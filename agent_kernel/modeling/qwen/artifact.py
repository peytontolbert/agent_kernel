from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from agent_kernel.config import KernelConfig
from agent_kernel.modeling.training.qwen_dataset import materialize_qwen_sft_dataset


def build_qwen_adapter_candidate_artifact(
    *,
    config: KernelConfig,
    repo_root: Path,
    output_dir: Path,
    base_model_name: str = "",
    focus: str | None = None,
    current_payload: object | None = None,
    supported_benchmark_families: list[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_manifest = materialize_qwen_sft_dataset(
        config=config,
        repo_root=repo_root,
        output_dir=output_dir / "dataset",
    )
    normalized_base_model = str(base_model_name or "").strip() or config.model_name
    adapter_output_dir = output_dir / "adapter"
    merged_output_dir = output_dir / "merged"
    adapter_manifest_path = output_dir / "qwen_adapter_manifest.json"
    adapter_output_dir.mkdir(parents=True, exist_ok=True)
    merged_output_dir.mkdir(parents=True, exist_ok=True)
    lineage = _artifact_lineage(
        current_payload=current_payload,
        active_artifact_path=config.qwen_adapter_artifact_path,
    )
    payload = {
        "spec_version": "asi_v1",
        "artifact_kind": "qwen_adapter_bundle",
        "lifecycle_state": "candidate",
        "generation_focus": str(focus or "coding_lane_sft").strip() or "coding_lane_sft",
        "runtime_role": "support_runtime",
        "training_objective": "qlora_sft",
        "base_model_name": normalized_base_model,
        "supported_benchmark_families": sorted(
            {
                str(value).strip()
                for value in (
                    supported_benchmark_families
                    if isinstance(supported_benchmark_families, list)
                    else ["repository", "project", "workflow", "tooling", "integration"]
                )
                if str(value).strip()
            }
        ),
        "runtime_policy": {
            "allow_primary_routing": False,
            "allow_shadow_routing": True,
            "allow_teacher_generation": True,
            "allow_post_liftoff_fallback": True,
            "require_retained_promotion_for_runtime_use": True,
        },
        "retention_gate": {
            "require_improvement_cycle_promotion": True,
            "require_non_regression": True,
            "require_base_model_match": True,
            "disallow_liftoff_authority": True,
        },
        "training_dataset_manifest": dataset_manifest,
        "training_backend": {
            "strategy": "external_qlora_backend",
            "launch_via": "scripts/run_training_backend.py",
            "artifact_env_function": "artifact_training_env",
        },
        "runtime_paths": {
            "dataset_manifest_path": str(dataset_manifest.get("manifest_path", "")),
            "train_dataset_path": str(dataset_manifest.get("train_dataset_path", "")),
            "eval_dataset_path": str(dataset_manifest.get("eval_dataset_path", "")),
            "served_model_name": normalized_base_model,
            "adapter_output_dir": str(adapter_output_dir),
            "merged_output_dir": str(merged_output_dir),
            "adapter_manifest_path": str(adapter_manifest_path),
        },
        "lineage": lineage,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
    }
    adapter_manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _artifact_lineage(
    *,
    current_payload: object | None,
    active_artifact_path: Path,
) -> dict[str, object]:
    parent_sha = ""
    if active_artifact_path.exists():
        try:
            parent_sha = hashlib.sha256(active_artifact_path.read_bytes()).hexdigest()
        except OSError:
            parent_sha = ""
    baseline = current_payload if isinstance(current_payload, dict) else {}
    return {
        "mode": "canonical_parent_mutation",
        "parent_artifact_path": str(active_artifact_path) if parent_sha else "",
        "parent_artifact_sha256": parent_sha,
        "candidate_artifact_strategy": "manifest_first_adapter_bundle",
        "promotion_policy": "canonical_replace_on_retain",
        "rejected_candidate_policy": "metrics_and_manifest_only",
        "base_model_delta": (
            {}
            if not isinstance(baseline, dict)
            else {
                key: value
                for key, value in {
                    "base_model_name": str(baseline.get("base_model_name", "")).strip(),
                    "runtime_role": str(baseline.get("runtime_role", "")).strip(),
                }.items()
                if value
            }
        ),
    }
