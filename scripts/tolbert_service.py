from __future__ import annotations

import argparse
from collections import OrderedDict
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch
import yaml


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


class TolbertRuntime:
    def __init__(
        self,
        *,
        repo_root: Path,
        config_path: Path,
        checkpoint_path: Path,
        nodes_path: Path,
        label_map_path: Path,
        source_spans_paths: list[Path],
        cache_paths: list[Path],
        device: str,
    ) -> None:
        sys.path.insert(0, str(repo_root / "other_repos" / "TOLBERT"))
        sys.path.insert(0, str(Path("/data/TOLBERT_BRAIN")))

        from tolbert_brain.brain import BrainConfig, TolbertBrain

        cfg_yaml = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        level_sizes = {int(k): int(v) for k, v in cfg_yaml["level_sizes"].items()}
        brain_cfg = BrainConfig(
            base_model_name=cfg_yaml["base_model_name"],
            level_sizes=level_sizes,
            checkpoint_path=str(checkpoint_path),
            max_length=int(cfg_yaml.get("max_length", 256)),
            device=device,
            proj_dim=int(cfg_yaml.get("proj_dim", 256)),
        )
        self.device = device
        self.brain = TolbertBrain.from_config(brain_cfg)
        self.nodes_by_id = self._load_nodes(nodes_path)
        self.label_maps, self.position_to_head_level, self.head_level_to_position = self._load_label_maps(
            label_map_path
        )
        self.parent_to_children = self._build_parent_to_children(source_spans_paths)
        self.cache_descriptors = self._load_cache_descriptors(cache_paths)
        self.shard_names = [str(descriptor["name"]) for descriptor in self.cache_descriptors]
        self._loaded_shards: OrderedDict[Path, dict[str, Any]] = OrderedDict()
        self._max_loaded_shards = max(1, int(os.getenv("AGENT_KERNEL_TOLBERT_MAX_LOADED_SHARDS", "2")))
        self.max_index_level = max(self.head_level_to_position)

    def query(
        self,
        *,
        query_text: str,
        branch_results: int,
        global_results: int,
        confidence_threshold: float,
        top_branches: int = 1,
        branch_confidence_margin: float = 0.12,
        low_confidence_widen_threshold: float = 0.6,
        ancestor_branch_levels: int = 2,
        low_confidence_branch_multiplier: float = 2.0,
        low_confidence_global_multiplier: float = 2.0,
    ) -> dict[str, Any]:
        out = self.brain.embed_and_predict(query_text, return_level_logits=True)
        query_embedding = out["embedding"]
        if hasattr(query_embedding, "detach"):
            query_embedding = query_embedding.detach().cpu()
        query_embedding = torch.as_tensor(query_embedding, dtype=torch.float32)
        query_embedding = query_embedding / torch.linalg.vector_norm(query_embedding)

        level_logits = {
            int(level): torch.as_tensor(logits, dtype=torch.float32)
            for level, logits in out["level_logits"].items()
        }
        request_shards: dict[Path, dict[str, Any]] = {}
        path_prediction = self._decode_path(level_logits, confidence_threshold)

        branch_level = self._select_branch_level(
            path_prediction,
            low_confidence_widen_threshold=low_confidence_widen_threshold,
        )
        path_confidence = self._path_confidence(path_prediction)
        widened_branch_results = branch_results
        widened_global_results = global_results
        if path_confidence < low_confidence_widen_threshold:
            widened_branch_results = max(
                branch_results,
                int(round(branch_results * max(1.0, low_confidence_branch_multiplier))),
            )
            widened_global_results = max(
                global_results,
                int(round(global_results * max(1.0, low_confidence_global_multiplier))),
            )
        branch_position = self.head_level_to_position[branch_level]
        branch_candidates = self._branch_candidates(
            level_logits=level_logits,
            path_prediction=path_prediction,
            branch_level=branch_level,
            top_branches=top_branches,
            confidence_margin=branch_confidence_margin,
        )
        primary_candidate = branch_candidates[0]
        branch_results_payload = self._search_multidepth_branch(
            query_embedding,
            path_prediction=path_prediction,
            selected_branch_level=branch_level,
            branch_local_id=primary_candidate["local_id"],
            limit=widened_branch_results,
            ancestor_branch_levels=ancestor_branch_levels,
            request_shards=request_shards,
        )
        fallback_results_payload = self._search_fallback_branches(
            query_embedding,
            path_prediction=path_prediction,
            selected_branch_level=branch_level,
            branch_candidates=branch_candidates[1:],
            limit=widened_branch_results,
            ancestor_branch_levels=ancestor_branch_levels,
            request_shards=request_shards,
        )
        global_results_payload = self._search(
            query_embedding,
            limit=widened_global_results,
            request_shards=request_shards,
        )

        return {
            "backend": "tolbert_brain_service",
            "device": self.device,
            "index_shards": list(self.shard_names),
            "level_focus": self._level_focus(branch_level),
            "path_prediction": path_prediction,
            "selected_branch_level": branch_level,
            "branch_candidates": branch_candidates,
            "retrieval": {
                "branch_scoped": branch_results_payload,
                "fallback_scoped": fallback_results_payload,
                "global": global_results_payload,
            },
        }

    def _decode_path(
        self,
        level_logits: dict[int, torch.Tensor],
        confidence_threshold: float,
    ) -> dict[str, Any]:
        predicted_level_ids: dict[str, int] = {}
        confidence_by_level: dict[str, float] = {}
        labels_by_level: dict[str, str] = {}

        ordered_levels = sorted(self.label_maps)
        previous_level = None
        for level in ordered_levels:
            logits = level_logits[level]
            valid_local_ids = self._valid_local_ids(level, logits=logits)
            if previous_level is None:
                candidate_ids = valid_local_ids
            else:
                parent_local = predicted_level_ids[str(previous_level)]
                children = self.parent_to_children[level].get(parent_local)
                if not children:
                    raise RuntimeError(
                        f"No ontology child mapping for level {level} and parent class {parent_local}."
                    )
                candidate_ids = [child for child in children if child in valid_local_ids]
            if not candidate_ids:
                raise RuntimeError(f"No valid label ids available for decode level {level}.")
            child_indices = torch.tensor(candidate_ids, dtype=torch.long)
            local_id = int(child_indices[torch.argmax(logits[child_indices])].item())

            confidence = float(torch.softmax(logits, dim=-1)[local_id].item())
            if confidence < confidence_threshold:
                raise RuntimeError(
                    f"TOLBERT confidence {confidence:.3f} below threshold "
                    f"{confidence_threshold:.3f} at level {level}."
                )

            predicted_level_ids[str(level)] = local_id
            confidence_by_level[str(level)] = round(confidence, 4)
            node_id = self.label_maps[level][local_id]
            labels_by_level[str(level)] = self.nodes_by_id[node_id]["name"]
            previous_level = level

        return {
            "tree_version": "tol_v1",
            "decode_mode": "greedy_hierarchical_decode",
            "levels": ordered_levels,
            "predicted_level_ids": predicted_level_ids,
            "confidence_by_level": confidence_by_level,
            "labels_by_level": labels_by_level,
            "fallbacks": [],
        }

    def _select_branch_level(
        self,
        path_prediction: dict[str, Any],
        *,
        low_confidence_widen_threshold: float,
    ) -> int:
        ordered_levels = [int(level) for level in path_prediction["levels"]]
        branch_level = min(max(ordered_levels), self.max_index_level)
        while branch_level > min(ordered_levels):
            confidence = float(path_prediction["confidence_by_level"].get(str(branch_level), 0.0))
            if confidence >= low_confidence_widen_threshold:
                break
            branch_level -= 1
        return branch_level

    def _branch_candidates(
        self,
        *,
        level_logits: dict[int, torch.Tensor],
        path_prediction: dict[str, Any],
        branch_level: int,
        top_branches: int,
        confidence_margin: float,
    ) -> list[dict[str, Any]]:
        logits = level_logits[branch_level]
        probs = torch.softmax(logits, dim=-1)
        valid_local_ids = self._valid_local_ids(branch_level, logits=logits)
        if not valid_local_ids:
            raise RuntimeError(f"No valid label ids available for branch level {branch_level}.")
        candidate_tensor = torch.tensor(valid_local_ids, dtype=torch.long)
        candidate_probs = probs[candidate_tensor]
        max_candidates = min(top_branches, int(candidate_probs.numel()))
        top_probs, top_positions = torch.topk(candidate_probs, k=max_candidates)
        primary_confidence = float(path_prediction["confidence_by_level"][str(branch_level)])
        candidates: list[dict[str, Any]] = []
        for prob, position_tensor in zip(top_probs.tolist(), top_positions.tolist()):
            local_id = int(candidate_tensor[int(position_tensor)].item())
            if candidates and primary_confidence - float(prob) > confidence_margin:
                continue
            node_id = self.label_maps[branch_level][local_id]
            candidates.append(
                {
                    "level": branch_level,
                    "local_id": local_id,
                    "confidence": round(float(prob), 4),
                    "label": self.nodes_by_id[node_id]["name"],
                    "node_id": node_id,
                }
            )
        path_prediction["fallbacks"] = [
            {
                "level": str(candidate["level"]),
                "local_id": str(candidate["local_id"]),
                "label": str(candidate["label"]),
                "confidence": f"{candidate['confidence']:.4f}",
            }
            for candidate in candidates[1:]
        ]
        return candidates

    def _search_fallback_branches(
        self,
        query_embedding: torch.Tensor,
        *,
        path_prediction: dict[str, Any],
        selected_branch_level: int,
        branch_candidates: list[dict[str, Any]],
        limit: int,
        ancestor_branch_levels: int,
        request_shards: dict[Path, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not branch_candidates or limit <= 0:
            return []
        per_branch_limit = max(1, limit)
        results: list[dict[str, Any]] = []
        for candidate in branch_candidates:
            branch_results = self._search_multidepth_branch(
                query_embedding,
                path_prediction=path_prediction,
                selected_branch_level=selected_branch_level,
                branch_local_id=int(candidate["local_id"]),
                limit=per_branch_limit,
                ancestor_branch_levels=ancestor_branch_levels,
                request_shards=request_shards,
            )
            for item in branch_results:
                metadata = dict(item.get("metadata", {}))
                metadata["branch_candidate_level"] = int(candidate["level"])
                metadata["branch_candidate_label"] = str(candidate["label"])
                metadata["branch_candidate_confidence"] = float(candidate["confidence"])
                item["metadata"] = metadata
                results.append(item)
        results.sort(key=lambda item: (-item["score"], item["span_id"]))
        deduped: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in results:
            if item["span_id"] in seen:
                continue
            seen.add(item["span_id"])
            deduped.append(item)
        return deduped[:limit]

    def _search_multidepth_branch(
        self,
        query_embedding: torch.Tensor,
        *,
        path_prediction: dict[str, Any],
        selected_branch_level: int,
        branch_local_id: int,
        limit: int,
        ancestor_branch_levels: int,
        request_shards: dict[Path, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        branch_specs = [(self.head_level_to_position[selected_branch_level], branch_local_id, selected_branch_level)]
        levels = sorted(int(level) for level in path_prediction["levels"])
        selected_index = levels.index(selected_branch_level)
        for offset in range(1, max(0, ancestor_branch_levels) + 1):
            if selected_index - offset < 0:
                break
            ancestor_level = levels[selected_index - offset]
            ancestor_position = self.head_level_to_position[ancestor_level]
            ancestor_local_id = int(path_prediction["predicted_level_ids"][str(ancestor_level)])
            branch_specs.append((ancestor_position, ancestor_local_id, ancestor_level))

        results: list[dict[str, Any]] = []
        for branch_position, local_id, node_level in branch_specs:
            branch_results = self._search(
                query_embedding,
                limit=limit,
                branch_filter=(branch_position, local_id),
                request_shards=request_shards,
            )
            for item in branch_results:
                metadata = dict(item.get("metadata", {}))
                metadata["branch_scope_level"] = node_level
                metadata["branch_scope_position"] = branch_position
                metadata["branch_scope_local_id"] = local_id
                item["metadata"] = metadata
                results.append(item)
        results.sort(key=lambda item: (-item["score"], item["span_id"]))
        deduped: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in results:
            if item["span_id"] in seen:
                continue
            seen.add(item["span_id"])
            deduped.append(item)
        return deduped[:limit]

    @staticmethod
    def _path_confidence(path_prediction: dict[str, Any]) -> float:
        confidences = [
            float(value)
            for value in path_prediction.get("confidence_by_level", {}).values()
        ]
        if not confidences:
            return 0.0
        return min(confidences)

    def _search(
        self,
        query_embedding: torch.Tensor,
        *,
        limit: int,
        branch_filter: tuple[int, int] | None = None,
        request_shards: dict[Path, dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for descriptor in self.cache_descriptors:
            cache_path = Path(str(descriptor["path"]))
            if branch_filter is not None and not self._descriptor_matches_branch_filter(
                descriptor,
                branch_filter=branch_filter,
            ):
                continue
            if request_shards is not None and cache_path in request_shards:
                shard = request_shards[cache_path]
            else:
                shard = self._load_cache(cache_path)
                if request_shards is not None:
                    request_shards[cache_path] = shard
            embs = shard["embs"]
            metas = shard["metas"]
            if branch_filter is not None:
                level, value = branch_filter
                branch_indices = shard.get("branch_indices")
                indices: list[int]
                if isinstance(branch_indices, dict):
                    level_map = branch_indices.get(int(level))
                    if isinstance(level_map, dict):
                        cached_indices = level_map.get(int(value))
                        if isinstance(cached_indices, list):
                            indices = cached_indices
                        else:
                            indices = []
                    else:
                        indices = []
                else:
                    indices = [
                        idx
                        for idx, meta in enumerate(metas)
                        if isinstance(meta.get("node_path"), list)
                        and len(meta["node_path"]) > level
                        and int(meta["node_path"][level]) == value
                    ]
                if not indices:
                    continue
                try:
                    shard_embs = embs[indices]
                except TypeError:
                    shard_embs = [embs[idx] for idx in indices]
                scores = torch.matmul(shard_embs, query_embedding)
                top_k = min(limit, scores.numel())
                top_scores, top_pos = torch.topk(scores, k=top_k)
                for score, pos in zip(top_scores.tolist(), top_pos.tolist()):
                    results.append(self._meta_to_result(metas[indices[pos]], score))
            else:
                scores = torch.matmul(embs, query_embedding)
                top_k = min(limit, scores.numel())
                top_scores, top_pos = torch.topk(scores, k=top_k)
                for score, pos in zip(top_scores.tolist(), top_pos.tolist()):
                    results.append(self._meta_to_result(metas[pos], score))

        results.sort(key=lambda item: (-item["score"], item["span_id"]))
        return results[:limit]

    def _meta_to_result(self, meta: dict[str, Any], score: float) -> dict[str, Any]:
        text = str(meta.get("text", ""))
        meta_payload = dict(meta.get("meta") or {})
        span_type = str(meta_payload.get("span_type", "")).strip()
        if not span_type:
            kind = str(meta_payload.get("kind", "")).strip().lower()
            if kind == "paper":
                span_type = (
                    "doc:paper_paragraph"
                    if "paragraph_index" in meta_payload
                    else "doc:paper"
                )
            elif kind:
                span_type = f"{kind}:span"
            else:
                span_type = "span"
        if span_type in {"agent:command_template", "agent:setup_command", "agent:skill_fragment", "agent:episode_step"}:
            preview = text.strip()
        else:
            preview = " ".join(text.split())
            if len(preview) > 400:
                preview = preview[:397] + "..."
        return {
            "span_id": str(meta.get("span_id", "")),
            "text": preview,
            "source_id": str(meta.get("source_id", meta.get("span_id", ""))),
            "span_type": span_type,
            "score": round(float(score), 4),
            "node_path": meta.get("node_path", []),
            "metadata": meta_payload,
        }

    def _load_cache(self, path: Path) -> dict[str, Any]:
        cached = self._loaded_shards.get(path)
        if cached is not None:
            self._loaded_shards.move_to_end(path)
            return cached
        cache = torch.load(path, map_location="cpu")
        embs = torch.as_tensor(cache["embs"], dtype=torch.float32)
        embs = torch.nn.functional.normalize(embs, dim=-1)
        metas = cache["metas"]
        shard = {
            "name": path.name,
            "embs": embs,
            "metas": metas,
            # Precompute static branch filters once per loaded shard instead of
            # rebuilding matching row indices on every query.
            "branch_indices": self._build_branch_indices(metas),
        }
        self._loaded_shards[path] = shard
        self._loaded_shards.move_to_end(path)
        while len(self._loaded_shards) > self._max_loaded_shards:
            self._loaded_shards.popitem(last=False)
        return shard

    def _build_branch_indices(self, metas: list[dict[str, Any]]) -> dict[int, dict[int, list[int]]]:
        branch_indices: dict[int, dict[int, list[int]]] = {}
        for index, meta in enumerate(metas):
            node_path = meta.get("node_path")
            if not isinstance(node_path, list):
                continue
            for position, node_id in enumerate(node_path):
                if position == 0:
                    continue
                level_indices = branch_indices.setdefault(int(position), {})
                level_indices.setdefault(int(node_id), []).append(index)
        return branch_indices

    def _load_cache_descriptors(self, paths: list[Path]) -> list[dict[str, Any]]:
        descriptors: list[dict[str, Any]] = []
        for path in paths:
            if path.suffix.lower() == ".json":
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    descriptors.append({"path": str(path), "name": path.name, "branch_presence": {}})
                    continue
                shards = payload.get("shards") if isinstance(payload, dict) else None
                if isinstance(shards, list):
                    for shard in shards:
                        if not isinstance(shard, dict):
                            continue
                        shard_path = Path(str(shard.get("path", "")).strip())
                        if not str(shard_path):
                            continue
                        descriptors.append(
                            {
                                "path": str(shard_path),
                                "name": str(shard.get("name", shard_path.name)),
                                "branch_presence": dict(shard.get("branch_presence") or {}),
                            }
                        )
                    continue
                cache_paths = payload.get("cache_paths") if isinstance(payload, dict) else None
                if isinstance(cache_paths, list) and all(str(item).strip() for item in cache_paths):
                    descriptors.extend(
                        {
                            "path": str(Path(str(item))),
                            "name": Path(str(item)).name,
                            "branch_presence": {},
                        }
                        for item in cache_paths
                    )
                    continue
            descriptors.append({"path": str(path), "name": path.name, "branch_presence": {}})
        return descriptors

    def _descriptor_matches_branch_filter(
        self,
        descriptor: dict[str, Any],
        *,
        branch_filter: tuple[int, int],
    ) -> bool:
        position, value = branch_filter
        branch_presence = descriptor.get("branch_presence")
        if not isinstance(branch_presence, dict) or not branch_presence:
            return True
        candidates = branch_presence.get(str(position))
        if not isinstance(candidates, list):
            return True
        try:
            candidate_values = {int(candidate) for candidate in candidates}
        except Exception:
            return True
        return int(value) in candidate_values

    def _load_nodes(self, path: Path) -> dict[int, dict[str, Any]]:
        records = _load_jsonl(path)
        return {int(record["node_id"]): record for record in records}

    def _load_label_maps(
        self,
        path: Path,
    ) -> tuple[dict[int, dict[int, int]], dict[int, int], dict[int, int]]:
        data = json.loads(path.read_text(encoding="utf-8"))
        label_maps: dict[int, dict[int, int]] = {}
        position_to_head_level: dict[int, int] = {}
        head_level_to_position: dict[int, int] = {}
        for position_str, mapping in data.items():
            if not str(position_str).isdigit():
                continue
            position = int(position_str)
            normalized = {int(local_id): int(node_id) for local_id, node_id in mapping.items()}
            head_level = position
            label_maps[head_level] = normalized
            position_to_head_level[position] = head_level
            head_level_to_position[head_level] = position
        return label_maps, position_to_head_level, head_level_to_position

    def _valid_local_ids(self, level: int, *, logits: torch.Tensor) -> list[int]:
        return [
            local_id
            for local_id in sorted(self.label_maps.get(level, {}))
            if 0 <= int(local_id) < int(logits.numel())
        ]

    def _build_parent_to_children(self, source_spans_paths: list[Path]) -> dict[int, dict[int, list[int]]]:
        inverse_maps = {
            level: {node_id: local_id for local_id, node_id in mapping.items()}
            for level, mapping in self.label_maps.items()
        }
        parent_to_children: dict[int, dict[int, list[int]]] = {
            level: {} for level in self.label_maps if level > 1
        }
        for path in source_spans_paths:
            for record in _iter_jsonl(path):
                node_path = record.get("node_path")
                if not isinstance(node_path, list):
                    continue
                ordered_positions = [position for position in sorted(self.position_to_head_level) if len(node_path) > position]
                for pos_index in range(1, len(ordered_positions)):
                    parent_position = ordered_positions[pos_index - 1]
                    child_position = ordered_positions[pos_index]
                    parent_level = self.position_to_head_level[parent_position]
                    child_level = self.position_to_head_level[child_position]
                    if child_level not in parent_to_children:
                        continue
                    parent_global = int(node_path[parent_position])
                    child_global = int(node_path[child_position])
                    parent_local = inverse_maps[parent_level].get(parent_global)
                    child_local = inverse_maps[child_level].get(child_global)
                    if parent_local is None or child_local is None:
                        continue
                    children = parent_to_children[child_level].setdefault(parent_local, [])
                    if child_local not in children:
                        children.append(child_local)
        return parent_to_children

    def _level_focus(self, branch_level: int) -> str:
        labels = {
            1: "pillar",
            2: "theme",
            3: "corpus",
            4: "artifact",
            5: "leaf",
        }
        return labels.get(branch_level, "skill")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--nodes", required=True)
    parser.add_argument("--label-map", required=True)
    parser.add_argument("--source-spans", action="append", required=True)
    parser.add_argument("--cache-path", action="append", required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested_device = args.device
    if requested_device == "cuda" and not torch.cuda.is_available():
        print(
            json.dumps(
                {
                    "startup_error": (
                        "Requested CUDA for TOLBERT service, but torch.cuda.is_available() is false."
                    )
                }
            ),
            file=sys.stderr,
            flush=True,
        )
        raise RuntimeError("CUDA requested but unavailable in the TOLBERT runtime environment.")
    runtime = TolbertRuntime(
        repo_root=Path(args.repo_root),
        config_path=Path(args.config),
        checkpoint_path=Path(args.checkpoint),
        nodes_path=Path(args.nodes),
        label_map_path=Path(args.label_map),
        source_spans_paths=[Path(value) for value in args.source_spans],
        cache_paths=[Path(value) for value in args.cache_path],
        device=requested_device,
    )
    print(
        json.dumps(
            {
                "event": "startup_ready",
                "backend": "tolbert_brain_service",
                "device": requested_device,
            }
        ),
        file=sys.stderr,
        flush=True,
    )

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            response = runtime.query(
                query_text=request["query_text"],
                branch_results=int(request["branch_results"]),
                global_results=int(request["global_results"]),
                confidence_threshold=float(request["confidence_threshold"]),
                top_branches=int(request.get("top_branches", 1)),
                branch_confidence_margin=float(request.get("branch_confidence_margin", 0.12)),
                low_confidence_widen_threshold=float(
                    request.get("low_confidence_widen_threshold", 0.6)
                ),
                ancestor_branch_levels=int(request.get("ancestor_branch_levels", 2)),
                low_confidence_branch_multiplier=float(
                    request.get("low_confidence_branch_multiplier", 2.0)
                ),
                low_confidence_global_multiplier=float(
                    request.get("low_confidence_global_multiplier", 2.0)
                ),
            )
        except Exception as exc:
            response = {"error": str(exc)}
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
