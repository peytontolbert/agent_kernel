from agent_kernel.config import KernelConfig
from agent_kernel.loop import AgentKernel
import pytest


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
    config = KernelConfig(
        provider="mock",
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        **override,
    )

    with pytest.raises(ValueError, match=message):
        AgentKernel(config=config)
