import json

from agent_kernel.config import KernelConfig
from agent_kernel.loop import AgentKernel
import pytest


def _write_decoder_ready_tolbert_artifact(tmp_path):
    bundle_path = tmp_path / "tolbert" / "hybrid_bundle_manifest.json"
    artifact_path = tmp_path / "tolbert" / "tolbert_model_artifact.json"
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_path.write_text(json.dumps({"artifact_kind": "tolbert_hybrid_runtime_bundle"}), encoding="utf-8")
    artifact_path.write_text(
        json.dumps(
            {
                "artifact_kind": "tolbert_model_bundle",
                "model_surfaces": {"decoder_surface": True},
                "hybrid_runtime": {
                    "primary_enabled": True,
                    "supports_decoder_surface": True,
                    "bundle_manifest_path": str(bundle_path),
                },
            }
        ),
        encoding="utf-8",
    )
    return artifact_path


def _write_universal_decoder_ready_tolbert_artifact(tmp_path):
    bundle_path = tmp_path / "tolbert" / "universal_bundle_manifest.json"
    artifact_path = tmp_path / "tolbert" / "tolbert_model_artifact.json"
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_path.write_text(json.dumps({"artifact_kind": "tolbert_hybrid_runtime_bundle"}), encoding="utf-8")
    artifact_path.write_text(
        json.dumps(
            {
                "artifact_kind": "tolbert_model_bundle",
                "model_surfaces": {"decoder_surface": True},
                "hybrid_runtime": {
                    "primary_enabled": False,
                    "supports_decoder_surface": True,
                    "bundle_manifest_path": "",
                },
                "universal_decoder_runtime": {
                    "materialized": True,
                    "bundle_manifest_path": str(bundle_path),
                    "training_objective": "universal_decoder_only",
                },
            }
        ),
        encoding="utf-8",
    )
    return artifact_path


def test_kernel_rejects_unknown_provider_at_startup(tmp_path):
    config = KernelConfig(
        provider="broken",
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )

    with pytest.raises(ValueError, match="unsupported provider"):
        AgentKernel(config=config)


def test_kernel_accepts_tolbert_provider_alias_at_startup(tmp_path):
    config = KernelConfig(
        provider="tolbert",
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )

    kernel = AgentKernel(config=config)

    assert kernel.config.normalized_provider() == "hybrid"


def test_kernel_rejects_nonpositive_timeout_at_startup(tmp_path):
    config = KernelConfig(
        provider="mock",
        command_timeout_seconds=0,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )

    with pytest.raises(ValueError, match="command_timeout_seconds must be > 0"):
        AgentKernel(config=config)


def test_kernel_rejects_nonpositive_runtime_history_window(tmp_path):
    config = KernelConfig(
        provider="mock",
        runtime_history_step_window=0,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )

    with pytest.raises(ValueError, match="runtime_history_step_window must be > 0"):
        AgentKernel(config=config)


def test_kernel_rejects_invalid_probability_thresholds_at_startup(tmp_path):
    config = KernelConfig(
        provider="mock",
        tolbert_confidence_threshold=1.5,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )

    with pytest.raises(ValueError, match="tolbert_confidence_threshold must be between 0 and 1"):
        AgentKernel(config=config)


def test_claimed_runtime_shape_requires_native_decoder_ready(tmp_path):
    artifact_path = _write_decoder_ready_tolbert_artifact(tmp_path)
    hybrid_config = KernelConfig(
        provider="hybrid",
        tolbert_model_artifact_path=artifact_path,
        workspace_root=tmp_path / "workspace_hybrid",
        trajectories_root=tmp_path / "trajectories_hybrid",
    )
    external_config = KernelConfig(
        provider="vllm",
        tolbert_model_artifact_path=artifact_path,
        workspace_root=tmp_path / "workspace_external",
        trajectories_root=tmp_path / "trajectories_external",
    )

    assert hybrid_config.claimed_runtime_shape() == "bounded_autonomous"
    assert external_config.claimed_runtime_shape() == "executable_floor"


def test_claimed_runtime_shape_accepts_materialized_universal_decoder_runtime(tmp_path):
    artifact_path = _write_universal_decoder_ready_tolbert_artifact(tmp_path)
    config = KernelConfig(
        provider="hybrid",
        tolbert_model_artifact_path=artifact_path,
        workspace_root=tmp_path / "workspace_hybrid",
        trajectories_root=tmp_path / "trajectories_hybrid",
    )

    posture = config.decoder_runtime_posture()

    assert config.claimed_runtime_shape() == "bounded_autonomous"
    assert posture["runtime_key"] == "universal_decoder_runtime"
    assert posture["training_objective"] == "universal_decoder_only"


def test_kernel_config_validation_allows_json_storage_and_unlimited_budget_group_caps(tmp_path):
    config = KernelConfig(
        provider="mock",
        storage_backend="json",
        delegated_job_max_active_per_budget_group=0,
        delegated_job_max_queued_per_budget_group=0,
        delegated_job_max_consecutive_selections_per_budget_group=0,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )

    config.ensure_directories()

    assert config.workspace_root.exists()
    assert config.trajectories_root.exists()


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"use_graph_memory": False}, "bounded_autonomous runtime requires use_graph_memory=True"),
        ({"use_world_model": False}, "bounded_autonomous runtime requires use_world_model=True"),
    ],
)
def test_kernel_rejects_weakened_bounded_autonomous_runtime_minimum(tmp_path, override, message):
    artifact_path = _write_decoder_ready_tolbert_artifact(tmp_path)
    config = KernelConfig(
        provider="hybrid",
        tolbert_model_artifact_path=artifact_path,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        **override,
    )

    with pytest.raises(ValueError, match=message):
        AgentKernel(config=config)
