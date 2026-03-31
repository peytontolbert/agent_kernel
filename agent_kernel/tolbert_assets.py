from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from .config import KernelConfig
from .improvement_common import artifact_payload_in_lifecycle_states, retained_artifact_payload
from .modeling.tolbert.delta import resolve_tolbert_runtime_checkpoint_path
from .retrieval_improvement import retained_retrieval_asset_controls
from .task_bank import TaskBank, load_discovered_tasks


@dataclass(frozen=True, slots=True)
class TolbertAssetPaths:
    output_dir: Path
    nodes_path: Path
    source_spans_path: Path
    model_spans_path: Path
    label_map_path: Path
    config_path: Path
    level_sizes_path: Path


def build_agentkernel_tolbert_assets(
    *,
    repo_root: Path,
    output_dir: Path,
    base_model_name: str = "bert-base-uncased",
    config: KernelConfig | None = None,
) -> TolbertAssetPaths:
    output_dir.mkdir(parents=True, exist_ok=True)

    builder = _AssetBuilder(repo_root=repo_root, config=config or KernelConfig())
    nodes, source_spans, model_spans, label_map, level_sizes = builder.build()

    paths = TolbertAssetPaths(
        output_dir=output_dir,
        nodes_path=output_dir / "nodes_agentkernel.jsonl",
        source_spans_path=output_dir / "source_spans_agentkernel.jsonl",
        model_spans_path=output_dir / "model_spans_agentkernel.jsonl",
        label_map_path=output_dir / "label_map_agentkernel.json",
        config_path=output_dir / "config_agentkernel.json",
        level_sizes_path=output_dir / "level_sizes_agentkernel.json",
    )

    _write_jsonl(paths.nodes_path, nodes)
    _write_jsonl(paths.source_spans_path, source_spans)
    _write_jsonl(paths.model_spans_path, model_spans)
    paths.label_map_path.write_text(json.dumps(label_map, indent=2), encoding="utf-8")
    paths.level_sizes_path.write_text(json.dumps(level_sizes, indent=2), encoding="utf-8")
    config = {
        "base_model_name": base_model_name,
        "level_sizes": level_sizes,
        "spans_files": [str(paths.model_spans_path)],
        "retrieval_asset_controls": dict(builder.asset_controls),
        "max_length": 256,
        "mask_probability": 0.15,
        "batch_size": 8,
        "num_workers": 0,
        "lr": 5.0e-5,
        "num_epochs": 10,
        "grad_clip": 1.0,
        "use_amp": True,
        "lambda_hier": 1.0,
        "lambda_path": 0.2,
        "lambda_contrast": 0.0,
        "contrast_temperature": 0.07,
        "output_dir": str(output_dir / "checkpoints"),
    }
    paths.config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return paths


def materialize_retained_retrieval_asset_bundle(
    *,
    repo_root: Path,
    config: KernelConfig | None = None,
    output_dir: Path | None = None,
    manifest_path: Path | None = None,
    base_model_name: str = "bert-base-uncased",
    cycle_id: str | None = None,
) -> Path:
    runtime_config = config or KernelConfig()
    payload = _retained_retrieval_payload(runtime_config)
    if payload is None:
        raise RuntimeError("retained retrieval proposals are required to materialize a Tolbert asset bundle")

    resolved_manifest_path = manifest_path or runtime_config.retrieval_asset_bundle_path
    if not resolved_manifest_path.is_absolute():
        resolved_manifest_path = repo_root / resolved_manifest_path
    resolved_output_dir = output_dir
    if resolved_output_dir is None:
        resolved_output_dir = resolved_manifest_path.parent / "tolbert_bundle"
    if not resolved_output_dir.is_absolute():
        resolved_output_dir = repo_root / resolved_output_dir

    paths = build_agentkernel_tolbert_assets(
        repo_root=repo_root,
        output_dir=resolved_output_dir,
        base_model_name=base_model_name,
        config=runtime_config,
    )
    runtime_paths = {
        "config_path": str(paths.config_path),
        "nodes_path": str(paths.nodes_path),
        "label_map_path": str(paths.label_map_path),
        "source_spans_paths": [str(paths.source_spans_path)],
        "checkpoint_path": _optional_existing_path(runtime_config.tolbert_checkpoint_path, repo_root=repo_root),
        "cache_paths": _existing_paths(runtime_config.tolbert_cache_paths, repo_root=repo_root),
    }
    manifest = {
        "spec_version": "asi_v1",
        "artifact_kind": "tolbert_retrieval_asset_bundle",
        "lifecycle_state": "retained",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cycle_id": cycle_id or "",
        "retrieval_proposals_path": str(runtime_config.retrieval_proposals_path),
        "asset_strategy": str(payload.get("asset_strategy", "")),
        "asset_controls": retained_retrieval_asset_controls(payload),
        "asset_rebuild_plan": payload.get("asset_rebuild_plan", {}),
        "bundle_output_dir": str(paths.output_dir),
        "materialized_paths": {
            "nodes_path": str(paths.nodes_path),
            "source_spans_path": str(paths.source_spans_path),
            "model_spans_path": str(paths.model_spans_path),
            "label_map_path": str(paths.label_map_path),
            "config_path": str(paths.config_path),
            "level_sizes_path": str(paths.level_sizes_path),
        },
        "runtime_paths": runtime_paths,
    }
    resolved_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return resolved_manifest_path


def retained_tolbert_runtime_paths(
    config: KernelConfig,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    artifact_path = config.tolbert_model_artifact_path
    if not artifact_path.is_absolute():
        artifact_path = repo_root / artifact_path
    model_bundle = _retained_tolbert_model_artifact(config, repo_root=repo_root)
    bundle = _retained_retrieval_asset_bundle(config, repo_root=repo_root)
    runtime_paths = {}
    if model_bundle is not None:
        runtime_paths = model_bundle.get("runtime_paths", {})
    if not runtime_paths and bundle is not None:
        runtime_paths = bundle.get("runtime_paths", {})
    if not isinstance(runtime_paths, dict):
        runtime_paths = {}
    source_spans_paths = runtime_paths.get("source_spans_paths", config.tolbert_source_spans_paths)
    cache_paths = runtime_paths.get("cache_paths", config.tolbert_cache_paths)
    checkpoint_path = ""
    if model_bundle is not None:
        checkpoint_path = resolve_tolbert_runtime_checkpoint_path(
            runtime_paths,
            artifact_path=artifact_path,
        )
    return {
        "tolbert_config_path": _string_or_default(runtime_paths.get("config_path"), config.tolbert_config_path),
        "tolbert_checkpoint_path": _string_or_default(
            checkpoint_path or runtime_paths.get("checkpoint_path"),
            config.tolbert_checkpoint_path,
        ),
        "tolbert_nodes_path": _string_or_default(runtime_paths.get("nodes_path"), config.tolbert_nodes_path),
        "tolbert_label_map_path": _string_or_default(runtime_paths.get("label_map_path"), config.tolbert_label_map_path),
        "tolbert_source_spans_paths": _string_list_or_default(source_spans_paths, config.tolbert_source_spans_paths),
        "tolbert_cache_paths": _string_list_or_default(cache_paths, config.tolbert_cache_paths),
    }


def _retained_tolbert_model_artifact(
    config: KernelConfig,
    *,
    repo_root: Path,
) -> dict[str, Any] | None:
    if not bool(config.use_tolbert_model_artifacts):
        return None
    path = config.tolbert_model_artifact_path
    if not path.is_absolute():
        path = repo_root / path
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return retained_artifact_payload(payload, artifact_kind="tolbert_model_bundle")


def _retained_retrieval_asset_bundle(
    config: KernelConfig,
    *,
    repo_root: Path,
) -> dict[str, Any] | None:
    path = config.retrieval_asset_bundle_path
    if not path.is_absolute():
        path = repo_root / path
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return retained_artifact_payload(payload, artifact_kind="tolbert_retrieval_asset_bundle")


def _retained_retrieval_payload(config: KernelConfig) -> dict[str, Any] | None:
    path = config.retrieval_proposals_path
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return retained_artifact_payload(payload, artifact_kind="retrieval_policy_set")


def _existing_paths(raw_paths: tuple[str, ...], *, repo_root: Path) -> list[str]:
    resolved: list[str] = []
    for raw_path in raw_paths:
        existing = _optional_existing_path(raw_path, repo_root=repo_root)
        if existing:
            resolved.append(existing)
    return resolved


def _optional_existing_path(raw_path: str | None, *, repo_root: Path) -> str:
    if not raw_path:
        return ""
    path = Path(raw_path)
    if not path.is_absolute():
        path = repo_root / path
    if not path.exists():
        return ""
    return str(path)


def _string_or_default(value: object, default: str | None) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return default


def _string_list_or_default(value: object, default: tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(value, list):
        normalized = tuple(str(item).strip() for item in value if str(item).strip())
        if normalized:
            return normalized
    if isinstance(value, tuple):
        normalized = tuple(str(item).strip() for item in value if str(item).strip())
        if normalized:
            return normalized
    return default


class _AssetBuilder:
    def __init__(self, *, repo_root: Path, config: KernelConfig) -> None:
        self.repo_root = repo_root
        self.config = config
        self.asset_controls = self._asset_controls()
        self.nodes: list[dict[str, Any]] = [
            {"node_id": 0, "level": 0, "parent_id": None, "name": "agent-kernel", "attributes": {}}
        ]
        self.spans: list[dict[str, Any]] = []
        self._next_node_id = 1
        self._level_local_ids: dict[int, dict[tuple[Any, ...], int]] = {}
        self._node_ids: dict[tuple[Any, ...], int] = {}
        self._level_one_specs = [
            ("artifacts", "Artifacts"),
            ("capabilities", "Capabilities"),
            ("docs", "Docs"),
            ("episodes", "Episodes"),
            ("failures", "Failures"),
            ("prompts", "Prompts"),
            ("procedures", "Procedures"),
            ("skills", "Skills"),
            ("tasks", "Tasks"),
            ("tools", "Tools"),
        ]

    def _prefer_failure_alignment(self) -> bool:
        return bool(self.asset_controls.get("prefer_failure_alignment", False))

    def build(
        self,
    ) -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
        dict[str, dict[str, int]],
        dict[str, int],
    ]:
        for key, label in self._level_one_specs:
            self._ensure_node(level=1, key=("level1", key), name=label, parent_id=0)
        self._build_tasks()
        self._build_discovered_tasks()
        if self._include_failure_catalog_spans():
            self._build_failure_catalog()
        self._build_docs()
        self._build_prompts()
        self._build_verifier_contracts()
        if self._include_skill_procedure_spans():
            self._build_skills()
        if self._include_tool_candidate_spans():
            self._build_tool_candidates()
        if self._include_episode_step_spans() or self._include_negative_episode_spans():
            self._build_episodes()

        label_map = {
            str(level): {str(local_id): node_id for _, local_id, node_id in sorted(entries, key=lambda item: item[1])}
            for level, entries in self._label_map_entries().items()
        }
        level_sizes = {
            str(level): len(entries)
            for level, entries in sorted(self._level_local_ids.items())
        }
        model_spans = [self._to_model_span(span) for span in self.spans]
        return self.nodes, self.spans, model_spans, label_map, level_sizes

    def _asset_controls(self) -> dict[str, object]:
        if not self.config.use_retrieval_proposals:
            return {}
        path = self.config.retrieval_proposals_path
        if not path.exists():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        return retained_retrieval_asset_controls(payload)

    def _include_failure_catalog_spans(self) -> bool:
        return bool(self.asset_controls.get("include_failure_catalog_spans", True))

    def _include_negative_episode_spans(self) -> bool:
        return bool(self.asset_controls.get("include_negative_episode_spans", True))

    def _include_tool_candidate_spans(self) -> bool:
        return bool(self.asset_controls.get("include_tool_candidate_spans", True))

    def _include_episode_step_spans(self) -> bool:
        return bool(self.asset_controls.get("include_episode_step_spans", True))

    def _include_skill_procedure_spans(self) -> bool:
        return bool(self.asset_controls.get("include_skill_procedure_spans", True))

    def _max_episode_step_spans_per_task(self) -> int:
        try:
            return max(1, int(self.asset_controls.get("max_episode_step_spans_per_task", 3)))
        except (TypeError, ValueError):
            return 3

    def _max_failure_spans_per_task(self) -> int:
        try:
            return max(1, int(self.asset_controls.get("max_failure_spans_per_task", 2)))
        except (TypeError, ValueError):
            return 2

    def _label_map_entries(self) -> dict[int, list[tuple[tuple[Any, ...], int, int]]]:
        entries: dict[int, list[tuple[tuple[Any, ...], int, int]]] = {}
        for level, local_ids in self._level_local_ids.items():
            level_entries = []
            for key, local_id in local_ids.items():
                level_entries.append((key, local_id, self._node_ids[key]))
            entries[level] = level_entries
        return entries

    def _build_tasks(self) -> None:
        for task in TaskBank().list():
            self._add_task_spans(task, discovered=False)

    def _build_discovered_tasks(self) -> None:
        for task in load_discovered_tasks(self.config.trajectories_root):
            self._add_task_spans(task, discovered=True)

    def _add_task_spans(self, task, *, discovered: bool) -> None:
        artifact_key = ("tasks", task.task_id)
        capability = str(task.metadata.get("capability", "unknown"))
        difficulty = str(task.metadata.get("difficulty", "unknown"))
        artifact_class = self._artifact_class_for_task(task)
        overview_lines = [
            f"task_id: {task.task_id}",
            f"prompt: {task.prompt}",
            f"workspace_subdir: {task.workspace_subdir}",
            f"success_command: {task.success_command}",
            f"capability: {capability}",
            f"difficulty: {difficulty}",
            f"artifact_class: {artifact_class}",
        ]
        if discovered:
            overview_lines.append("memory_source: discovered_task")
            discovery_failure_types = [
                str(value).strip()
                for value in task.metadata.get("discovery_failure_types", [])
                if str(value).strip()
            ]
            discovery_transition_failures = [
                str(value).strip()
                for value in task.metadata.get("discovery_transition_failures", [])
                if str(value).strip()
            ]
            if discovery_failure_types:
                overview_lines.append(
                    f"discovery_failure_types: {', '.join(discovery_failure_types)}"
                )
            if discovery_transition_failures:
                overview_lines.append(
                    "discovery_transition_failures: "
                    + ", ".join(discovery_transition_failures)
                )
        if task.expected_files:
            overview_lines.append(f"expected_files: {', '.join(task.expected_files)}")
        if task.expected_output_substrings:
            overview_lines.append(
                f"expected_output_substrings: {', '.join(task.expected_output_substrings)}"
            )
        self._add_span(
            artifact_key=artifact_key,
            span_kind="task_overview",
            span_id=f"task:{task.task_id}:overview",
            source_id=task.task_id,
            text="\n".join(overview_lines),
            span_type="agent:task",
            metadata={
                "task_id": task.task_id,
                "capability": capability,
                "difficulty": difficulty,
                "artifact_class": artifact_class,
                "memory_source": "discovered_task" if discovered else "",
            },
        )
        self._add_span(
            artifact_key=("capabilities", capability),
            span_kind="capability_task",
            span_id=f"capability:{capability}:task:{task.task_id}",
            source_id=task.task_id,
            text="\n".join(
                [
                    f"task_id: {task.task_id}",
                    f"capability: {capability}",
                    f"prompt: {task.prompt}",
                    f"success_command: {task.success_command}",
                ]
            ),
            span_type="agent:capability_task",
            metadata={"task_id": task.task_id, "capability": capability, "difficulty": difficulty},
        )
        self._add_span(
                artifact_key=("artifacts", artifact_class),
                span_kind="artifact_task",
                span_id=f"artifact:{artifact_class}:task:{task.task_id}",
                source_id=task.task_id,
                text="\n".join(
                    [
                        f"task_id: {task.task_id}",
                        f"artifact_class: {artifact_class}",
                        f"expected_files: {', '.join(task.expected_files) or 'none'}",
                        f"forbidden_files: {', '.join(task.forbidden_files) or 'none'}",
                    ]
                ),
                span_type="agent:artifact_task",
                metadata={"task_id": task.task_id, "artifact_class": artifact_class},
        )
        for index, command in enumerate(task.suggested_commands, start=1):
            tool_family = self._tool_family(command)
            self._add_span(
                artifact_key=artifact_key,
                span_kind="suggested_command",
                span_id=f"task:{task.task_id}:suggested:{index}",
                source_id=task.task_id,
                text=command,
                span_type="agent:command_template",
                metadata={
                    "task_id": task.task_id,
                    "command_index": index,
                    "tool_family": tool_family,
                    "artifact_class": artifact_class,
                    "capability": capability,
                    "memory_source": "discovered_task" if discovered else "",
                    "required_preconditions": [],
                    "touched_files": self._touched_files(command),
                    "verifier_success": True,
                },
            )
            self._add_span(
                artifact_key=("tools", tool_family),
                span_kind="tool_command_template",
                span_id=f"tool:{tool_family}:task:{task.task_id}:suggested:{index}",
                source_id=task.task_id,
                text=command,
                span_type="agent:tool_template",
                metadata={
                    "task_id": task.task_id,
                    "tool_family": tool_family,
                    "command_index": index,
                    "capability": capability,
                    "memory_source": "discovered_task" if discovered else "",
                },
            )
            self._add_span(
                artifact_key=("procedures", capability),
                span_kind="procedure_command",
                span_id=f"procedure:{task.task_id}:suggested:{index}",
                source_id=task.task_id,
                text=command,
                span_type="agent:procedure_span",
                metadata={
                    "task_id": task.task_id,
                    "tool_family": tool_family,
                    "artifact_class": artifact_class,
                    "capability": capability,
                    "command_index": index,
                    "memory_source": "discovered_task" if discovered else "",
                },
            )
        for index, command in enumerate(task.setup_commands, start=1):
            tool_family = self._tool_family(command)
            self._add_span(
                artifact_key=artifact_key,
                span_kind="setup_command",
                span_id=f"task:{task.task_id}:setup:{index}",
                source_id=task.task_id,
                text=command,
                span_type="agent:setup_command",
                metadata={
                    "task_id": task.task_id,
                    "command_index": index,
                    "tool_family": tool_family,
                    "memory_source": "discovered_task" if discovered else "",
                    "required_preconditions": [],
                    "touched_files": self._touched_files(command),
                },
            )

    def _build_verifier_contracts(self) -> None:
        path = self.config.verifier_contracts_path
        if not path.exists():
            return
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload.get("proposals", []) if isinstance(payload, dict) else []
        for index, record in enumerate(records, start=1):
            if not isinstance(record, dict):
                continue
            contract = record.get("contract", {})
            if not isinstance(contract, dict):
                continue
            task_id = str(record.get("source_task_id", "")).strip() or f"verifier_candidate_{index}"
            prompt = str(contract.get("prompt", "")).strip()
            success_command = str(contract.get("success_command", "")).strip()
            if not prompt and not success_command:
                continue
            artifact_key = ("docs", f"verifier:{task_id}")
            self._add_span(
                artifact_key=artifact_key,
                span_kind="verifier_contract",
                span_id=f"verifier:{task_id}:contract",
                source_id=task_id,
                text="\n".join(
                    [
                        f"task_id: {task_id}",
                        f"prompt: {prompt}",
                        f"success_command: {success_command}",
                        "expected_files: " + ", ".join(contract.get("expected_files", []) or ["none"]),
                        "forbidden_files: " + ", ".join(contract.get("forbidden_files", []) or ["none"]),
                    ]
                ),
                span_type="agent:verifier_contract",
                metadata={
                    "task_id": task_id,
                    "proposal_id": str(record.get("proposal_id", "")).strip(),
                    "memory_source": "verifier_candidate",
                },
            )

    def _build_docs(self) -> None:
        doc_paths = [self.repo_root / "README.md", *sorted((self.repo_root / "docs").glob("*.md"))]
        for path in doc_paths:
            artifact_key = ("docs", path.name)
            for index, chunk in enumerate(_split_markdown_chunks(path.read_text(encoding="utf-8")), start=1):
                self._add_span(
                    artifact_key=artifact_key,
                    span_kind="doc_chunk",
                    span_id=f"doc:{path.stem}:{index}",
                    source_id=str(path.relative_to(self.repo_root)),
                    text=chunk,
                    span_type="doc:readme_chunk",
                    metadata={"path": str(path.relative_to(self.repo_root)), "chunk_index": index},
                )

    def _build_failure_catalog(self) -> None:
        for failure_type, description in (
            ("command_failure", "command exited unsuccessfully"),
            ("missing_expected_file", "expected file was not produced"),
            ("missing_expected_output", "expected output substring was not produced"),
            ("timeout", "command or action timed out"),
            ("policy_terminated", "policy ended without satisfying the verifier"),
            ("repeated_failed_action", "agent repeated failing behavior"),
        ):
            self._add_span(
                artifact_key=("failures", failure_type),
                span_kind="failure_catalog",
                span_id=f"failure:{failure_type}:catalog",
                source_id="failure_catalog",
                text=f"failure_type: {failure_type}\ndescription: {description}",
                span_type="agent:failure_pattern",
                metadata={"failure_type": failure_type},
            )

    def _build_prompts(self) -> None:
        for path in sorted((self.repo_root / "prompts").glob("*.md")):
            artifact_key = ("prompts", path.name)
            for index, chunk in enumerate(_split_markdown_chunks(path.read_text(encoding="utf-8")), start=1):
                self._add_span(
                    artifact_key=artifact_key,
                    span_kind="prompt_chunk",
                    span_id=f"prompt:{path.stem}:{index}",
                    source_id=str(path.relative_to(self.repo_root)),
                    text=chunk,
                    span_type="agent:prompt",
                    metadata={"path": str(path.relative_to(self.repo_root)), "chunk_index": index},
                )

    def _build_skills(self) -> None:
        path = self.config.skills_path
        if not path.exists():
            return
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and artifact_payload_in_lifecycle_states(
            payload,
            artifact_kind="skill_set",
            allowed_states={"promoted", "retained"},
        ) is None:
            return
        skill_records = payload.get("skills", payload) if isinstance(payload, dict) else payload
        for record in skill_records:
            if not isinstance(record, dict):
                continue
            lifecycle_state = str(record.get("lifecycle_state", "")).strip()
            if lifecycle_state and lifecycle_state not in {"promoted", "retained"}:
                continue
            retention_decision = record.get("retention_decision", {})
            if isinstance(retention_decision, dict) and str(retention_decision.get("state", "")).strip() == "reject":
                continue
            task_id = str(record.get("source_task_id", record.get("task_id", "")))
            commands = [str(command) for command in record.get("procedure", {}).get("commands", [])]
            if not commands:
                commands = [str(command) for command in record.get("commands", [])]
            if not commands:
                continue
            artifact_key = ("skills", task_id)
            quality = float(record.get("quality", 1.0))
            capability = self._task_capability(task_id)
            self._add_span(
                artifact_key=artifact_key,
                span_kind="skill_sequence",
                span_id=f"skill:{task_id}:sequence",
                source_id=task_id,
                text="\n".join(commands),
                span_type="agent:skill_fragment",
                metadata={
                    "task_id": task_id,
                    "command_count": len(commands),
                    "skill_id": record.get("skill_id", f"skill:{task_id}:primary"),
                    "quality": quality,
                    "capability": capability,
                },
            )
            for index, command in enumerate(commands, start=1):
                self._add_span(
                    artifact_key=("procedures", capability),
                    span_kind="skill_procedure",
                    span_id=f"skill:{task_id}:procedure:{index}",
                    source_id=task_id,
                    text=command,
                    span_type="agent:procedure_span",
                    metadata={
                        "task_id": task_id,
                        "skill_id": record.get("skill_id", f"skill:{task_id}:primary"),
                        "quality": quality,
                        "capability": capability,
                        "tool_family": self._tool_family(command),
                        "required_preconditions": [],
                        "touched_files": self._touched_files(command),
                        "verifier_success": str(record.get("verifier", {}).get("termination_reason", "")) == "success",
                    },
                )

    def _build_episodes(self) -> None:
        step_span_counts: dict[str, int] = {}
        failure_span_counts: dict[str, int] = {}
        payloads = (
            self.config.sqlite_store().iter_trajectory_payloads()
            if self.config.uses_sqlite_storage()
            else []
        )
        if not payloads:
            for path in sorted(self.config.trajectories_root.glob("*.json")):
                payloads.append(json.loads(path.read_text(encoding="utf-8")))
        for data in payloads:
            task_id = str(data["task_id"])
            artifact_key = ("episodes", task_id)
            capability = self._task_capability(task_id)
            episode_success = bool(data.get("success", False))
            steps = data.get("steps", [])
            for step in steps:
                if not self._include_episode_step_spans():
                    break
                if step_span_counts.get(task_id, 0) >= self._max_episode_step_spans_per_task():
                    break
                index = int(step["index"])
                verification = step.get("verification") or {}
                command_text = str(step.get("content", ""))
                command_result = step.get("command_result") or {}
                failure_types = self._failure_types_from_reasons(verification.get("reasons", []))
                prefer_failure_alignment = self._prefer_failure_alignment()
                if prefer_failure_alignment and episode_success and not failure_types:
                    continue
                text_lines = [
                    f"task_id: {task_id}",
                    f"step_index: {index}",
                    f"thought: {step.get('thought', '')}",
                    f"action: {step.get('action', '')}",
                    f"content: {command_text}",
                    f"verification_passed: {verification.get('passed', False)}",
                    f"verification_reasons: {' | '.join(verification.get('reasons', []))}",
                    f"tool_family: {self._tool_family(command_text)}",
                    f"touched_files: {', '.join(self._touched_files(command_text)) or 'none'}",
                ]
                self._add_span(
                    artifact_key=artifact_key,
                    span_kind="episode_step",
                    span_id=f"episode:{task_id}:step:{index}",
                    source_id=task_id,
                    text="\n".join(text_lines),
                    span_type="agent:episode_step",
                    metadata={
                        "task_id": task_id,
                        "step_index": index,
                        "capability": capability,
                        "tool_family": self._tool_family(command_text),
                        "touched_files": self._touched_files(command_text),
                        "verification_passed": bool(verification.get("passed", False)),
                        "failure_types": failure_types,
                        "exit_code": command_result.get("exit_code"),
                    },
                )
                if command_text.strip():
                    self._add_span(
                        artifact_key=("procedures", capability),
                        span_kind="episode_procedure",
                        span_id=f"episode:{task_id}:procedure:{index}",
                        source_id=task_id,
                        text=command_text,
                        span_type="agent:procedure_span",
                        metadata={
                            "task_id": task_id,
                            "step_index": index,
                            "capability": capability,
                            "tool_family": self._tool_family(command_text),
                            "touched_files": self._touched_files(command_text),
                            "verifier_success": bool(verification.get("passed", False)),
                            "failure_types": failure_types,
                        },
                    )
                    step_span_counts[task_id] = step_span_counts.get(task_id, 0) + 1
                if not self._include_negative_episode_spans():
                    continue
                if not failure_types or failure_span_counts.get(task_id, 0) >= self._max_failure_spans_per_task():
                    continue
                self._add_span(
                    artifact_key=("failures", task_id),
                    span_kind="episode_failure",
                    span_id=f"failure:{task_id}:step:{index}",
                    source_id=task_id,
                    text="\n".join(
                        [
                            f"task_id: {task_id}",
                            f"step_index: {index}",
                            f"failure_types: {', '.join(failure_types)}",
                            f"command: {command_text}",
                            f"reasons: {' | '.join(verification.get('reasons', []))}",
                        ]
                    ),
                    span_type="agent:failure_pattern",
                    metadata={
                        "task_id": task_id,
                        "step_index": index,
                        "failure_types": failure_types,
                        "tool_family": self._tool_family(command_text),
                        "touched_files": self._touched_files(command_text),
                    },
                )
                failure_span_counts[task_id] = failure_span_counts.get(task_id, 0) + 1

    def _build_tool_candidates(self) -> None:
        path = self.config.tool_candidates_path
        if not path.exists():
            return
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and artifact_payload_in_lifecycle_states(
            payload,
            artifact_kind="tool_candidate_set",
            allowed_states={"replay_verified", "retained"},
        ) is None:
            return
        records = payload.get("candidates", payload) if isinstance(payload, dict) else payload
        for record in records:
            if not isinstance(record, dict):
                continue
            lifecycle_state = str(record.get("lifecycle_state", "")).strip()
            promotion_stage = str(record.get("promotion_stage", "")).strip()
            if lifecycle_state == "rejected":
                continue
            retention_decision = record.get("retention_decision", {})
            if isinstance(retention_decision, dict) and str(retention_decision.get("state", "")).strip() == "reject":
                continue
            if promotion_stage not in {"replay_verified", "promoted_tool"}:
                continue
            if promotion_stage == "replay_verified" and lifecycle_state not in {"replay_verified"}:
                continue
            if promotion_stage == "promoted_tool" and lifecycle_state not in {"retained"}:
                continue
            task_id = str(record.get("source_task_id", ""))
            commands = [str(command) for command in record.get("procedure", {}).get("commands", [])]
            if not task_id or not commands:
                continue
            capability = self._task_capability(task_id)
            quality = float(record.get("quality", 0.0))
            artifact_key = ("tools", task_id)
            self._add_span(
                artifact_key=artifact_key,
                span_kind="tool_candidate",
                span_id=f"tool:{task_id}:candidate",
                source_id=task_id,
                text="\n".join(commands),
                span_type="agent:tool_candidate",
                metadata={
                    "task_id": task_id,
                    "tool_id": str(record.get("tool_id", f"tool:{task_id}:primary")),
                    "quality": quality,
                    "capability": capability,
                    "benchmark_family": str(record.get("benchmark_family", "bounded")),
                },
            )
            for index, command in enumerate(commands, start=1):
                self._add_span(
                    artifact_key=("procedures", capability),
                    span_kind="tool_candidate_procedure",
                    span_id=f"tool:{task_id}:procedure:{index}",
                    source_id=task_id,
                    text=command,
                    span_type="agent:procedure_span",
                    metadata={
                        "task_id": task_id,
                        "tool_id": str(record.get("tool_id", f"tool:{task_id}:primary")),
                        "quality": quality,
                        "capability": capability,
                        "tool_family": self._tool_family(command),
                        "touched_files": self._touched_files(command),
                        "verifier_success": str(record.get("verifier", {}).get("termination_reason", "")) == "success",
                    },
                )

    def _add_span(
        self,
        *,
        artifact_key: tuple[str, str],
        span_kind: str,
        span_id: str,
        source_id: str,
        text: str,
        span_type: str,
        metadata: dict[str, Any],
    ) -> None:
        if not text.strip():
            return
        level_one_key = ("level1", artifact_key[0])
        level_two_key = ("level2", artifact_key[0], artifact_key[1])
        level_three_key = ("level3", artifact_key[0], artifact_key[1], span_kind)

        node_path = [
            0,
            self._ensure_node(level=1, key=level_one_key, name=self._display_name_for_level_one(artifact_key[0]), parent_id=0),
            self._ensure_node(level=2, key=level_two_key, name=artifact_key[1], parent_id=self._node_ids[level_one_key]),
            self._ensure_node(
                level=3,
                key=level_three_key,
                name=f"{artifact_key[1]}:{span_kind}",
                parent_id=self._node_ids[level_two_key],
            ),
        ]
        self.spans.append(
            {
                "span_id": span_id,
                "text": text,
                "source_id": source_id,
                "node_path": node_path,
                "meta": {
                    "span_type": span_type,
                    **metadata,
                },
            }
        )

    def _to_model_span(self, span: dict[str, Any]) -> dict[str, Any]:
        remapped = dict(span)
        node_path = list(span["node_path"])
        local_path = [0]
        for level, node_id in enumerate(node_path[1:], start=1):
            local_path.append(self._local_id_for_node(level, int(node_id)))
        remapped["node_path"] = local_path
        return remapped

    def _display_name_for_level_one(self, key: str) -> str:
        for value, label in self._level_one_specs:
            if key == value:
                return label
        return key

    def _ensure_node(self, *, level: int, key: tuple[Any, ...], name: str, parent_id: int) -> int:
        node_id = self._node_ids.get(key)
        if node_id is not None:
            return node_id
        local_ids = self._level_local_ids.setdefault(level, {})
        local_id = len(local_ids)
        local_ids[key] = local_id
        node_id = self._next_node_id
        self._next_node_id += 1
        self._node_ids[key] = node_id
        self.nodes.append(
            {
                "node_id": node_id,
                "level": level,
                "parent_id": parent_id,
                "name": name,
                "attributes": {"local_id": local_id},
            }
        )
        return node_id

    def _local_id_for_node(self, level: int, node_id: int) -> int:
        for node in self.nodes:
            if int(node["node_id"]) == node_id:
                return int(node["attributes"]["local_id"])
        raise KeyError(f"Unknown node_id at level {level}: {node_id}")

    @staticmethod
    def _tool_family(command: str) -> str:
        stripped = str(command).strip()
        if not stripped:
            return "none"
        token = stripped.split()[0]
        return {
            "printf": "text_write",
            "echo": "text_write",
            "cat": "file_read",
            "mv": "file_rename",
            "rm": "file_delete",
            "mkdir": "directory_create",
            "grep": "text_search",
            "test": "shell_assert",
        }.get(token, token)

    @staticmethod
    def _touched_files(command: str) -> list[str]:
        touched: list[str] = []
        normalized = str(command).replace("&&", " ").replace("||", " ")
        for token in normalized.split():
            cleaned = token.strip().strip("'\"")
            if "/" in cleaned or "." in cleaned:
                if cleaned.startswith("-"):
                    continue
                if cleaned not in touched:
                    touched.append(cleaned)
        return touched[:5]

    @staticmethod
    def _artifact_class_for_task(task: Any) -> str:
        files = list(getattr(task, "expected_files", []))
        if any("/" in path for path in files):
            return "nested_file"
        if any(path.endswith(".txt") for path in files):
            return "text_file"
        if files:
            return "filesystem_artifact"
        return "command_outcome"

    @staticmethod
    def _failure_types_from_reasons(reasons: list[str]) -> list[str]:
        failure_types: list[str] = []
        for reason in reasons:
            lowered = str(reason).lower()
            if "missing expected file" in lowered:
                failure_types.append("missing_expected_file")
            elif "forbidden file present" in lowered:
                failure_types.append("forbidden_file_present")
            elif "forbidden output present" in lowered:
                failure_types.append("forbidden_output_present")
            elif "exit code" in lowered:
                failure_types.append("command_failure")
            elif "timed out" in lowered:
                failure_types.append("timeout")
            elif "policy terminated" in lowered:
                failure_types.append("policy_terminated")
            elif "repeated failed action" in lowered:
                failure_types.append("repeated_failed_action")
        return sorted(set(failure_types))

    def _task_capability(self, task_id: str) -> str:
        try:
            task = TaskBank().get(task_id)
        except KeyError:
            return "unknown"
        return str(task.metadata.get("capability", "unknown"))


def _split_markdown_chunks(text: str) -> list[str]:
    chunks = []
    for raw_chunk in text.split("\n\n"):
        chunk = raw_chunk.strip()
        if chunk:
            chunks.append(" ".join(chunk.split()))
    return chunks


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
