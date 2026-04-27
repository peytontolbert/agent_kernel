from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
import os
import sys

from .extensions.strategy.kernel_catalog import kernel_catalog_string_list
from .modeling.model_python import preferred_model_python_path


_DEFAULT_UNATTENDED_ALLOWED_BENCHMARK_FAMILIES = tuple(
    kernel_catalog_string_list("runtime_defaults", "unattended_allowed_benchmark_families")
)
_DEFAULT_UNATTENDED_GENERATED_PATH_PREFIXES = tuple(
    kernel_catalog_string_list("runtime_defaults", "unattended_generated_path_prefixes")
)
_DEFAULT_UNATTENDED_TRUST_REQUIRED_BENCHMARK_FAMILIES = tuple(
    kernel_catalog_string_list("trust", "default_required_benchmark_families")
)


def _split_env_paths(name: str) -> tuple[str, ...]:
    raw = os.getenv(name, "")
    return tuple(part for part in raw.split(os.pathsep) if part)


def _split_env_csv(name: str) -> tuple[str, ...]:
    raw = os.getenv(name, "")
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def current_external_task_manifests_paths() -> tuple[str, ...]:
    return _split_env_paths("AGENT_KERNEL_EXTERNAL_TASK_MANIFESTS_PATHS")


def current_storage_governance_config() -> "KernelConfig":
    base = KernelConfig()
    return KernelConfig(
        storage_backend=os.getenv("AGENT_KERNEL_STORAGE_BACKEND", base.storage_backend),
        improvement_cycles_path=Path(
            os.getenv("AGENT_KERNEL_IMPROVEMENT_CYCLES_PATH", str(base.improvement_cycles_path))
        ),
        candidate_artifacts_root=Path(
            os.getenv("AGENT_KERNEL_CANDIDATE_ARTIFACTS_ROOT", str(base.candidate_artifacts_root))
        ),
        improvement_reports_dir=Path(
            os.getenv("AGENT_KERNEL_IMPROVEMENT_REPORTS_DIR", str(base.improvement_reports_dir))
        ),
        strategy_memory_nodes_path=Path(
            os.getenv("AGENT_KERNEL_STRATEGY_MEMORY_NODES_PATH", str(base.strategy_memory_nodes_path))
        ),
        strategy_memory_snapshots_path=Path(
            os.getenv("AGENT_KERNEL_STRATEGY_MEMORY_SNAPSHOTS_PATH", str(base.strategy_memory_snapshots_path))
        ),
        run_reports_dir=Path(
            os.getenv("AGENT_KERNEL_RUN_REPORTS_DIR", str(base.run_reports_dir))
        ),
        run_checkpoints_dir=Path(
            os.getenv("AGENT_KERNEL_RUN_CHECKPOINTS_DIR", str(base.run_checkpoints_dir))
        ),
        unattended_trust_ledger_path=Path(
            os.getenv("AGENT_KERNEL_UNATTENDED_TRUST_LEDGER_PATH", str(base.unattended_trust_ledger_path))
        ),
        tolbert_model_artifact_path=Path(
            os.getenv("AGENT_KERNEL_TOLBERT_MODEL_ARTIFACT_PATH", str(base.tolbert_model_artifact_path))
        ),
        storage_keep_cycle_export_files=int(
            os.getenv("AGENT_KERNEL_STORAGE_KEEP_CYCLE_EXPORT_FILES", str(base.storage_keep_cycle_export_files))
        ),
        storage_keep_report_export_files=int(
            os.getenv("AGENT_KERNEL_STORAGE_KEEP_REPORT_EXPORT_FILES", str(base.storage_keep_report_export_files))
        ),
        storage_keep_candidate_export_dirs=int(
            os.getenv("AGENT_KERNEL_STORAGE_KEEP_CANDIDATE_EXPORT_DIRS", str(base.storage_keep_candidate_export_dirs))
        ),
        storage_keep_run_report_files=int(
            os.getenv("AGENT_KERNEL_STORAGE_KEEP_RUN_REPORT_FILES", str(base.storage_keep_run_report_files))
        ),
        storage_keep_run_checkpoint_files=int(
            os.getenv("AGENT_KERNEL_STORAGE_KEEP_RUN_CHECKPOINT_FILES", str(base.storage_keep_run_checkpoint_files))
        ),
        storage_keep_namespace_candidate_dirs=int(
            os.getenv(
                "AGENT_KERNEL_STORAGE_KEEP_NAMESPACE_CANDIDATE_DIRS",
                str(base.storage_keep_namespace_candidate_dirs),
            )
        ),
        storage_max_cycle_export_records=int(
            os.getenv("AGENT_KERNEL_STORAGE_MAX_CYCLE_EXPORT_RECORDS", str(base.storage_max_cycle_export_records))
        ),
        storage_max_report_history_records=int(
            os.getenv(
                "AGENT_KERNEL_STORAGE_MAX_REPORT_HISTORY_RECORDS",
                str(base.storage_max_report_history_records),
            )
        ),
        storage_keep_tolbert_candidate_dirs=int(
            os.getenv(
                "AGENT_KERNEL_STORAGE_KEEP_TOLBERT_CANDIDATE_DIRS",
                str(base.storage_keep_tolbert_candidate_dirs),
            )
        ),
        storage_tolbert_candidate_budget_bytes=int(
            os.getenv(
                "AGENT_KERNEL_STORAGE_TOLBERT_CANDIDATE_BUDGET_BYTES",
                str(base.storage_tolbert_candidate_budget_bytes),
            )
        ),
        storage_tolbert_shared_store_budget_bytes=int(
            os.getenv(
                "AGENT_KERNEL_STORAGE_TOLBERT_SHARED_STORE_BUDGET_BYTES",
                str(base.storage_tolbert_shared_store_budget_bytes),
            )
        ),
    )


def _default_vllm_python_bin() -> str:
    preferred = preferred_model_python_path()
    if preferred is not None:
        return str(preferred)
    return sys.executable


@dataclass(slots=True)
class KernelConfig:
    provider: str = os.getenv("AGENT_KERNEL_PROVIDER", "vllm")
    model_name: str = os.getenv("AGENT_KERNEL_MODEL", "Qwen/Qwen3.5-9B")
    ollama_host: str = os.getenv("AGENT_KERNEL_OLLAMA_HOST", "http://127.0.0.1:11434")
    vllm_host: str = os.getenv("AGENT_KERNEL_VLLM_HOST", "http://127.0.0.1:8000")
    vllm_api_key: str = os.getenv("AGENT_KERNEL_VLLM_API_KEY", "")
    model_stack_host: str = os.getenv("AGENT_KERNEL_MODEL_STACK_HOST", "http://127.0.0.1:8001")
    model_stack_api_key: str = os.getenv("AGENT_KERNEL_MODEL_STACK_API_KEY", "")
    model_stack_model_dir: str = os.getenv("AGENT_KERNEL_MODEL_STACK_MODEL_DIR", "")
    model_stack_tokenizer_path: str = os.getenv("AGENT_KERNEL_MODEL_STACK_TOKENIZER_PATH", "")
    model_stack_repo_path: str = os.getenv("AGENT_KERNEL_MODEL_STACK_REPO_PATH", "other_repos/model-stack")
    vllm_autostart: bool = os.getenv("AGENT_KERNEL_VLLM_AUTOSTART", "1") == "1"
    vllm_start_command: str = os.getenv("AGENT_KERNEL_VLLM_START_COMMAND", "")
    vllm_start_extra_args: str = os.getenv("AGENT_KERNEL_VLLM_START_EXTRA_ARGS", "")
    vllm_python_bin: str = os.getenv("AGENT_KERNEL_VLLM_PYTHON_BIN", _default_vllm_python_bin())
    vllm_tensor_parallel_size: int = int(os.getenv("AGENT_KERNEL_VLLM_TENSOR_PARALLEL_SIZE", "1"))
    vllm_max_model_len: int = int(os.getenv("AGENT_KERNEL_VLLM_MAX_MODEL_LEN", "4096"))
    vllm_gpu_memory_utilization: float = float(
        os.getenv("AGENT_KERNEL_VLLM_GPU_MEMORY_UTILIZATION", "0.92")
    )
    vllm_enforce_eager: bool = os.getenv("AGENT_KERNEL_VLLM_ENFORCE_EAGER", "1") == "1"
    vllm_reasoning_parser: str = os.getenv("AGENT_KERNEL_VLLM_REASONING_PARSER", "")
    vllm_language_model_only: bool = os.getenv("AGENT_KERNEL_VLLM_LANGUAGE_MODEL_ONLY", "1") == "1"
    vllm_autostart_timeout_seconds: int = int(
        os.getenv("AGENT_KERNEL_VLLM_AUTOSTART_TIMEOUT_SECONDS", "180")
    )
    vllm_autostart_poll_interval_seconds: float = float(
        os.getenv("AGENT_KERNEL_VLLM_AUTOSTART_POLL_INTERVAL_SECONDS", "1")
    )
    vllm_runtime_log_path: Path = Path(
        os.getenv("AGENT_KERNEL_VLLM_RUNTIME_LOG_PATH", "var/runtime/vllm_runtime.log")
    )
    vllm_runtime_state_path: Path = Path(
        os.getenv("AGENT_KERNEL_VLLM_RUNTIME_STATE_PATH", "var/runtime/vllm_runtime.json")
    )
    vllm_runtime_lock_path: Path = Path(
        os.getenv("AGENT_KERNEL_VLLM_RUNTIME_LOCK_PATH", "var/runtime/vllm_runtime.lock")
    )
    use_tolbert_context: bool = os.getenv("AGENT_KERNEL_USE_TOLBERT_CONTEXT", "1") == "1"
    tolbert_mode: str = os.getenv("AGENT_KERNEL_TOLBERT_MODE", "full")
    use_skills: bool = os.getenv("AGENT_KERNEL_USE_SKILLS", "1") == "1"
    use_graph_memory: bool = os.getenv("AGENT_KERNEL_USE_GRAPH_MEMORY", "1") == "1"
    use_world_model: bool = os.getenv("AGENT_KERNEL_USE_WORLD_MODEL", "1") == "1"
    use_universe_model: bool = os.getenv("AGENT_KERNEL_USE_UNIVERSE_MODEL", "1") == "1"
    use_state_estimation_proposals: bool = (
        os.getenv("AGENT_KERNEL_USE_STATE_ESTIMATION_PROPOSALS", "1") == "1"
    )
    use_trust_proposals: bool = os.getenv("AGENT_KERNEL_USE_TRUST_PROPOSALS", "1") == "1"
    use_recovery_proposals: bool = os.getenv("AGENT_KERNEL_USE_RECOVERY_PROPOSALS", "1") == "1"
    use_delegation_proposals: bool = os.getenv("AGENT_KERNEL_USE_DELEGATION_PROPOSALS", "1") == "1"
    use_operator_policy_proposals: bool = os.getenv("AGENT_KERNEL_USE_OPERATOR_POLICY_PROPOSALS", "1") == "1"
    use_transition_model_proposals: bool = os.getenv("AGENT_KERNEL_USE_TRANSITION_MODEL_PROPOSALS", "1") == "1"
    use_tolbert_model_artifacts: bool = os.getenv("AGENT_KERNEL_USE_TOLBERT_MODEL_ARTIFACTS", "1") == "1"
    use_planner: bool = os.getenv("AGENT_KERNEL_USE_PLANNER", "1") == "1"
    use_role_specialization: bool = os.getenv("AGENT_KERNEL_USE_ROLE_SPECIALIZATION", "1") == "1"
    asi_coding_require_live_llm: bool = os.getenv("AGENT_KERNEL_ASI_CODING_REQUIRE_LIVE_LLM", "0") == "1"
    use_prompt_proposals: bool = os.getenv("AGENT_KERNEL_USE_PROMPT_PROPOSALS", "1") == "1"
    use_curriculum_proposals: bool = os.getenv("AGENT_KERNEL_USE_CURRICULUM_PROPOSALS", "1") == "1"
    use_retrieval_proposals: bool = os.getenv("AGENT_KERNEL_USE_RETRIEVAL_PROPOSALS", "1") == "1"
    use_research_library_context: bool = os.getenv("AGENT_KERNEL_USE_RESEARCH_LIBRARY_CONTEXT", "1") == "1"
    research_library_standalone_context: bool = (
        os.getenv("AGENT_KERNEL_RESEARCH_LIBRARY_STANDALONE_CONTEXT", "0") == "1"
    )
    research_library_config_path: Path = Path(
        os.getenv("AGENT_KERNEL_RESEARCH_LIBRARY_CONFIG_PATH", "config/research_library_sources.json")
    )
    research_library_status_path: Path = Path(
        os.getenv("AGENT_KERNEL_RESEARCH_LIBRARY_STATUS_PATH", "var/research_library/status.json")
    )
    research_library_context_max_chunks: int = int(
        os.getenv("AGENT_KERNEL_RESEARCH_LIBRARY_CONTEXT_MAX_CHUNKS", "6")
    )
    research_library_context_max_models: int = int(
        os.getenv("AGENT_KERNEL_RESEARCH_LIBRARY_CONTEXT_MAX_MODELS", "8")
    )
    research_library_context_max_repositories: int = int(
        os.getenv("AGENT_KERNEL_RESEARCH_LIBRARY_CONTEXT_MAX_REPOSITORIES", "4")
    )
    research_library_context_max_algorithms: int = int(
        os.getenv("AGENT_KERNEL_RESEARCH_LIBRARY_CONTEXT_MAX_ALGORITHMS", "4")
    )
    research_library_context_max_paper_hits: int = int(
        os.getenv("AGENT_KERNEL_RESEARCH_LIBRARY_CONTEXT_MAX_PAPER_HITS", "2")
    )
    research_library_paper_scan_file_limit: int = int(
        os.getenv("AGENT_KERNEL_RESEARCH_LIBRARY_PAPER_SCAN_FILE_LIMIT", "1")
    )
    research_library_paper_scan_row_limit: int = int(
        os.getenv("AGENT_KERNEL_RESEARCH_LIBRARY_PAPER_SCAN_ROW_LIMIT", "4096")
    )
    tolbert_tree_version: str = os.getenv("AGENT_KERNEL_TOLBERT_TREE_VERSION", "tol_v1")
    tolbert_branch_results: int = int(os.getenv("AGENT_KERNEL_TOLBERT_BRANCH_RESULTS", "3"))
    tolbert_global_results: int = int(os.getenv("AGENT_KERNEL_TOLBERT_GLOBAL_RESULTS", "2"))
    tolbert_top_branches: int = int(os.getenv("AGENT_KERNEL_TOLBERT_TOP_BRANCHES", "3"))
    tolbert_ancestor_branch_levels: int = int(
        os.getenv("AGENT_KERNEL_TOLBERT_ANCESTOR_BRANCH_LEVELS", "2")
    )
    tolbert_max_spans_per_source: int = int(
        os.getenv("AGENT_KERNEL_TOLBERT_MAX_SPANS_PER_SOURCE", "2")
    )
    tolbert_context_max_chunks: int = int(
        os.getenv("AGENT_KERNEL_TOLBERT_CONTEXT_MAX_CHUNKS", "8")
    )
    tolbert_context_char_budget: int = int(
        os.getenv("AGENT_KERNEL_TOLBERT_CONTEXT_CHAR_BUDGET", "2400")
    )
    tolbert_context_compile_budget_seconds: float = float(
        os.getenv("AGENT_KERNEL_TOLBERT_CONTEXT_COMPILE_BUDGET_SECONDS", "0")
    )
    tolbert_service_timeout_seconds: int = int(
        os.getenv("AGENT_KERNEL_TOLBERT_SERVICE_TIMEOUT_SECONDS", "15")
    )
    tolbert_service_startup_attempts: int = int(
        os.getenv("AGENT_KERNEL_TOLBERT_SERVICE_STARTUP_ATTEMPTS", "2")
    )
    llm_plan_max_items: int = int(
        os.getenv("AGENT_KERNEL_LLM_PLAN_MAX_ITEMS", "4")
    )
    llm_summary_max_chars: int = int(
        os.getenv("AGENT_KERNEL_LLM_SUMMARY_MAX_CHARS", "600")
    )
    tolbert_confidence_threshold: float = float(
        os.getenv("AGENT_KERNEL_TOLBERT_CONFIDENCE_THRESHOLD", "0.0")
    )
    tolbert_branch_confidence_margin: float = float(
        os.getenv("AGENT_KERNEL_TOLBERT_BRANCH_CONFIDENCE_MARGIN", "0.12")
    )
    tolbert_low_confidence_widen_threshold: float = float(
        os.getenv("AGENT_KERNEL_TOLBERT_LOW_CONFIDENCE_WIDEN_THRESHOLD", "0.6")
    )
    tolbert_low_confidence_branch_multiplier: float = float(
        os.getenv("AGENT_KERNEL_TOLBERT_LOW_CONFIDENCE_BRANCH_MULTIPLIER", "2.0")
    )
    tolbert_low_confidence_global_multiplier: float = float(
        os.getenv("AGENT_KERNEL_TOLBERT_LOW_CONFIDENCE_GLOBAL_MULTIPLIER", "2.0")
    )
    tolbert_deterministic_command_confidence: float = float(
        os.getenv("AGENT_KERNEL_TOLBERT_DETERMINISTIC_COMMAND_CONFIDENCE", "0.7")
    )
    tolbert_first_step_direct_command_confidence: float = float(
        os.getenv("AGENT_KERNEL_TOLBERT_FIRST_STEP_DIRECT_COMMAND_CONFIDENCE", "0.9")
    )
    tolbert_direct_command_min_score: int = int(
        os.getenv("AGENT_KERNEL_TOLBERT_DIRECT_COMMAND_MIN_SCORE", "1")
    )
    tolbert_skill_ranking_min_confidence: float = float(
        os.getenv("AGENT_KERNEL_TOLBERT_SKILL_RANKING_MIN_CONFIDENCE", "0.35")
    )
    tolbert_semantic_score_weight: float = float(
        os.getenv("AGENT_KERNEL_TOLBERT_SEMANTIC_SCORE_WEIGHT", "1.0")
    )
    tolbert_task_match_weight: float = float(
        os.getenv("AGENT_KERNEL_TOLBERT_TASK_MATCH_WEIGHT", "8.0")
    )
    tolbert_source_task_weight: float = float(
        os.getenv("AGENT_KERNEL_TOLBERT_SOURCE_TASK_WEIGHT", "7.0")
    )
    tolbert_distractor_penalty: float = float(
        os.getenv("AGENT_KERNEL_TOLBERT_DISTRACTOR_PENALTY", "4.0")
    )
    tolbert_python_bin: str = os.getenv(
        "AGENT_KERNEL_TOLBERT_PYTHON_BIN",
        "/home/peyton/miniconda3/envs/ai/bin/python",
    )
    tolbert_config_path: str | None = os.getenv(
        "AGENT_KERNEL_TOLBERT_CONFIG_PATH",
        "/data/agentkernel/var/tolbert/agentkernel/config_agentkernel.json",
    )
    tolbert_checkpoint_path: str | None = os.getenv(
        "AGENT_KERNEL_TOLBERT_CHECKPOINT_PATH",
        "/data/agentkernel/var/tolbert/agentkernel/checkpoints/tolbert_epoch10.pt",
    )
    tolbert_nodes_path: str | None = os.getenv(
        "AGENT_KERNEL_TOLBERT_NODES_PATH",
        "/data/agentkernel/var/tolbert/agentkernel/nodes_agentkernel.jsonl",
    )
    tolbert_label_map_path: str | None = os.getenv(
        "AGENT_KERNEL_TOLBERT_LABEL_MAP_PATH",
        "/data/agentkernel/var/tolbert/agentkernel/label_map_agentkernel.json",
    )
    tolbert_source_spans_paths: tuple[str, ...] = _split_env_paths("AGENT_KERNEL_TOLBERT_SOURCE_SPANS_PATHS") or (
        "/data/agentkernel/var/tolbert/agentkernel/source_spans_agentkernel.jsonl",
    )
    tolbert_cache_paths: tuple[str, ...] = _split_env_paths("AGENT_KERNEL_TOLBERT_CACHE_PATHS") or (
        "/data/agentkernel/var/tolbert/agentkernel/retrieval_cache/agentkernel__tolbert_epoch10.pt",
    )
    tolbert_cache_shard_size: int = int(
        os.getenv("AGENT_KERNEL_TOLBERT_CACHE_SHARD_SIZE", "50000")
    )
    use_paper_research_context: bool = os.getenv("AGENT_KERNEL_USE_PAPER_RESEARCH_CONTEXT", "1") == "1"
    paper_research_query_mode: str = os.getenv("AGENT_KERNEL_PAPER_RESEARCH_QUERY_MODE", "auto")
    paper_research_branch_results: int = int(
        os.getenv("AGENT_KERNEL_PAPER_RESEARCH_BRANCH_RESULTS", "2")
    )
    paper_research_global_results: int = int(
        os.getenv("AGENT_KERNEL_PAPER_RESEARCH_GLOBAL_RESULTS", "4")
    )
    paper_research_config_path: str | None = os.getenv(
        "AGENT_KERNEL_PAPER_RESEARCH_CONFIG_PATH",
        "/data/TOLBERT_BRAIN/configs/tolbert_brain_example.yaml",
    )
    paper_research_checkpoint_path: str | None = os.getenv(
        "AGENT_KERNEL_PAPER_RESEARCH_CHECKPOINT_PATH",
        "/data/TOLBERT_BRAIN/checkpoints/tolbert_brain/tolbert_epoch3.pt",
    )
    paper_research_nodes_path: str | None = os.getenv(
        "AGENT_KERNEL_PAPER_RESEARCH_NODES_PATH",
        "/data/TOLBERT_BRAIN/data/joint/nodes_joint.jsonl",
    )
    paper_research_label_map_path: str | None = os.getenv(
        "AGENT_KERNEL_PAPER_RESEARCH_LABEL_MAP_PATH",
        "/data/TOLBERT_BRAIN/data/joint/label_map_joint.json",
    )
    paper_research_source_spans_paths: tuple[str, ...] = _split_env_paths(
        "AGENT_KERNEL_PAPER_RESEARCH_SOURCE_SPANS_PATHS"
    ) or (
        "/data/TOLBERT_BRAIN/data/joint/code_spans_joint.jsonl",
        "/data/TOLBERT_BRAIN/data/joint/paper_spans_joint.jsonl",
    )
    paper_research_cache_paths: tuple[str, ...] = _split_env_paths(
        "AGENT_KERNEL_PAPER_RESEARCH_CACHE_PATHS"
    ) or (
        "/data/TOLBERT_BRAIN/checkpoints/tolbert_brain/retrieval_cache/paper_spans_joint_mapped__tolbert_epoch3.pt",
    )
    tolbert_device: str = os.getenv("AGENT_KERNEL_TOLBERT_DEVICE", "cuda")
    max_steps: int = int(os.getenv("AGENT_KERNEL_MAX_STEPS", "12"))
    max_task_steps_hard_cap: int = int(os.getenv("AGENT_KERNEL_MAX_TASK_STEPS_HARD_CAP", "4096"))
    frontier_task_step_floor: int = int(os.getenv("AGENT_KERNEL_FRONTIER_TASK_STEP_FLOOR", "50"))
    runtime_history_step_window: int = int(os.getenv("AGENT_KERNEL_RUNTIME_HISTORY_STEP_WINDOW", "32"))
    payload_history_step_window: int = int(os.getenv("AGENT_KERNEL_PAYLOAD_HISTORY_STEP_WINDOW", "12"))
    checkpoint_history_step_window: int = int(os.getenv("AGENT_KERNEL_CHECKPOINT_HISTORY_STEP_WINDOW", "24"))
    history_archive_summary_max_chars: int = int(
        os.getenv("AGENT_KERNEL_HISTORY_ARCHIVE_SUMMARY_MAX_CHARS", "1200")
    )
    timeout_seconds: int = int(os.getenv("AGENT_KERNEL_TIMEOUT_SECONDS", "20"))
    command_timeout_seconds: int = int(
        os.getenv(
            "AGENT_KERNEL_COMMAND_TIMEOUT_SECONDS",
            os.getenv("AGENT_KERNEL_TIMEOUT_SECONDS", "20"),
        )
    )
    llm_timeout_seconds: int = int(
        os.getenv(
            "AGENT_KERNEL_LLM_TIMEOUT_SECONDS",
            os.getenv("AGENT_KERNEL_TIMEOUT_SECONDS", "20"),
        )
    )
    sandbox_command_containment_mode: str = os.getenv(
        "AGENT_KERNEL_SANDBOX_COMMAND_CONTAINMENT_MODE",
        "auto",
    )
    sandbox_command_containment_tool: str = os.getenv(
        "AGENT_KERNEL_SANDBOX_COMMAND_CONTAINMENT_TOOL",
        "bwrap",
    )
    llm_retry_attempts: int = int(os.getenv("AGENT_KERNEL_LLM_RETRY_ATTEMPTS", "2"))
    llm_retry_backoff_seconds: float = float(
        os.getenv("AGENT_KERNEL_LLM_RETRY_BACKOFF_SECONDS", "0.5")
    )
    observe_feature_probe_max_tasks: int = int(
        os.getenv("AGENT_KERNEL_OBSERVE_FEATURE_PROBE_MAX_TASKS", "8")
    )
    compare_feature_max_tasks: int = int(
        os.getenv("AGENT_KERNEL_COMPARE_FEATURE_MAX_TASKS", "40")
    )
    persist_episode_memory: bool = os.getenv("AGENT_KERNEL_PERSIST_EPISODE_MEMORY", "1") == "1"
    persist_learning_candidates: bool = os.getenv("AGENT_KERNEL_PERSIST_LEARNING_CANDIDATES", "1") == "1"
    external_task_manifests_paths: tuple[str, ...] = _split_env_paths(
        "AGENT_KERNEL_EXTERNAL_TASK_MANIFESTS_PATHS"
    )
    workspace_root: Path = Path(os.getenv("AGENT_KERNEL_WORKSPACE_ROOT", "workspace"))
    trajectories_root: Path = Path(
        os.getenv("AGENT_KERNEL_TRAJECTORIES_ROOT", "trajectories/episodes")
    )
    skills_path: Path = Path(os.getenv("AGENT_KERNEL_SKILLS_PATH", "trajectories/skills/command_skills.json"))
    operator_classes_path: Path = Path(
        os.getenv("AGENT_KERNEL_OPERATOR_CLASSES_PATH", "trajectories/operators/operator_classes.json")
    )
    tool_candidates_path: Path = Path(
        os.getenv("AGENT_KERNEL_TOOL_CANDIDATES_PATH", "trajectories/tools/tool_candidates.json")
    )
    benchmark_candidates_path: Path = Path(
        os.getenv("AGENT_KERNEL_BENCHMARK_CANDIDATES_PATH", "trajectories/benchmarks/benchmark_candidates.json")
    )
    retrieval_proposals_path: Path = Path(
        os.getenv("AGENT_KERNEL_RETRIEVAL_PROPOSALS_PATH", "trajectories/retrieval/retrieval_proposals.json")
    )
    retrieval_asset_bundle_path: Path = Path(
        os.getenv("AGENT_KERNEL_RETRIEVAL_ASSET_BUNDLE_PATH", "trajectories/retrieval/retrieval_asset_bundle.json")
    )
    tolbert_model_artifact_path: Path = Path(
        os.getenv("AGENT_KERNEL_TOLBERT_MODEL_ARTIFACT_PATH", "trajectories/tolbert_model/tolbert_model_artifact.json")
    )
    tolbert_supervised_datasets_dir: Path = Path(
        os.getenv("AGENT_KERNEL_TOLBERT_SUPERVISED_DATASETS_DIR", "trajectories/tolbert_model/datasets")
    )
    qwen_adapter_artifact_path: Path = Path(
        os.getenv("AGENT_KERNEL_QWEN_ADAPTER_ARTIFACT_PATH", "trajectories/qwen_adapter/qwen_adapter_artifact.json")
    )
    qwen_supervised_datasets_dir: Path = Path(
        os.getenv("AGENT_KERNEL_QWEN_SUPERVISED_DATASETS_DIR", "trajectories/qwen_adapter/datasets")
    )
    tolbert_liftoff_report_path: Path = Path(
        os.getenv("AGENT_KERNEL_TOLBERT_LIFTOFF_REPORT_PATH", "trajectories/tolbert_model/liftoff_gate_report.json")
    )
    verifier_contracts_path: Path = Path(
        os.getenv("AGENT_KERNEL_VERIFIER_CONTRACTS_PATH", "trajectories/verifiers/verifier_contracts.json")
    )
    prompt_proposals_path: Path = Path(
        os.getenv("AGENT_KERNEL_PROMPT_PROPOSALS_PATH", "trajectories/prompts/prompt_proposals.json")
    )
    universe_constitution_path: Path = Path(
        os.getenv("AGENT_KERNEL_UNIVERSE_CONSTITUTION_PATH", "trajectories/universe/universe_constitution.json")
    )
    operating_envelope_path: Path = Path(
        os.getenv("AGENT_KERNEL_OPERATING_ENVELOPE_PATH", "trajectories/universe/operating_envelope.json")
    )
    universe_contract_path: Path = Path(
        os.getenv("AGENT_KERNEL_UNIVERSE_CONTRACT_PATH", "trajectories/universe/universe_contract.json")
    )
    world_model_proposals_path: Path = Path(
        os.getenv("AGENT_KERNEL_WORLD_MODEL_PROPOSALS_PATH", "trajectories/world_model/world_model_proposals.json")
    )
    state_estimation_proposals_path: Path = Path(
        os.getenv(
            "AGENT_KERNEL_STATE_ESTIMATION_PROPOSALS_PATH",
            "trajectories/state_estimation/state_estimation_proposals.json",
        )
    )
    trust_proposals_path: Path = Path(
        os.getenv("AGENT_KERNEL_TRUST_PROPOSALS_PATH", "trajectories/trust/trust_proposals.json")
    )
    recovery_proposals_path: Path = Path(
        os.getenv("AGENT_KERNEL_RECOVERY_PROPOSALS_PATH", "trajectories/recovery/recovery_proposals.json")
    )
    delegation_proposals_path: Path = Path(
        os.getenv("AGENT_KERNEL_DELEGATION_PROPOSALS_PATH", "trajectories/delegation/delegation_proposals.json")
    )
    operator_policy_proposals_path: Path = Path(
        os.getenv("AGENT_KERNEL_OPERATOR_POLICY_PROPOSALS_PATH", "trajectories/operator_policy/operator_policy_proposals.json")
    )
    transition_model_proposals_path: Path = Path(
        os.getenv("AGENT_KERNEL_TRANSITION_MODEL_PROPOSALS_PATH", "trajectories/transition_model/transition_model_proposals.json")
    )
    curriculum_proposals_path: Path = Path(
        os.getenv("AGENT_KERNEL_CURRICULUM_PROPOSALS_PATH", "trajectories/curriculum/curriculum_proposals.json")
    )
    improvement_cycles_path: Path = Path(
        os.getenv("AGENT_KERNEL_IMPROVEMENT_CYCLES_PATH", "trajectories/improvement/cycles.jsonl")
    )
    candidate_artifacts_root: Path = Path(
        os.getenv("AGENT_KERNEL_CANDIDATE_ARTIFACTS_ROOT", "trajectories/improvement/candidates")
    )
    improvement_reports_dir: Path = Path(
        os.getenv("AGENT_KERNEL_IMPROVEMENT_REPORTS_DIR", "trajectories/improvement/reports")
    )
    strategy_memory_nodes_path: Path = Path(
        os.getenv("AGENT_KERNEL_STRATEGY_MEMORY_NODES_PATH", "trajectories/improvement/strategy_memory/nodes.jsonl")
    )
    strategy_memory_snapshots_path: Path = Path(
        os.getenv("AGENT_KERNEL_STRATEGY_MEMORY_SNAPSHOTS_PATH", "trajectories/improvement/strategy_memory/snapshots.json")
    )
    semantic_hub_root: Path = Path(
        os.getenv("AGENT_KERNEL_SEMANTIC_HUB_ROOT", "trajectories/semantic_hub")
    )
    run_reports_dir: Path = Path(
        os.getenv("AGENT_KERNEL_RUN_REPORTS_DIR", "trajectories/reports")
    )
    learning_artifacts_path: Path = Path(
        os.getenv("AGENT_KERNEL_LEARNING_ARTIFACTS_PATH", "trajectories/learning/run_learning_artifacts.json")
    )
    capability_modules_path: Path = Path(
        os.getenv("AGENT_KERNEL_CAPABILITY_MODULES_PATH", "config/capabilities.json")
    )
    run_checkpoints_dir: Path = Path(
        os.getenv("AGENT_KERNEL_RUN_CHECKPOINTS_DIR", "trajectories/checkpoints")
    )
    delegated_job_queue_path: Path = Path(
        os.getenv("AGENT_KERNEL_DELEGATED_JOB_QUEUE_PATH", "trajectories/jobs/queue.json")
    )
    delegated_job_runtime_state_path: Path = Path(
        os.getenv("AGENT_KERNEL_DELEGATED_JOB_RUNTIME_STATE_PATH", "trajectories/jobs/runtime_state.json")
    )
    storage_backend: str = os.getenv("AGENT_KERNEL_STORAGE_BACKEND", "sqlite")
    runtime_database_path: Path = Path(
        os.getenv("AGENT_KERNEL_RUNTIME_DATABASE_PATH", "var/runtime/agentkernel.sqlite3")
    )
    storage_write_episode_exports: bool = os.getenv(
        "AGENT_KERNEL_STORAGE_WRITE_EPISODE_EXPORTS",
        "1",
    ) == "1"
    storage_write_learning_exports: bool = os.getenv(
        "AGENT_KERNEL_STORAGE_WRITE_LEARNING_EXPORTS",
        "0",
    ) == "1"
    storage_write_cycle_exports: bool = os.getenv(
        "AGENT_KERNEL_STORAGE_WRITE_CYCLE_EXPORTS",
        "1",
    ) == "1"
    storage_write_job_state_exports: bool = os.getenv(
        "AGENT_KERNEL_STORAGE_WRITE_JOB_STATE_EXPORTS",
        "1",
    ) == "1"
    storage_write_dataset_shards: bool = os.getenv(
        "AGENT_KERNEL_STORAGE_WRITE_DATASET_SHARDS",
        "0",
    ) == "1"
    storage_keep_terminal_job_records: int = int(
        os.getenv("AGENT_KERNEL_STORAGE_KEEP_TERMINAL_JOB_RECORDS", "256")
    )
    storage_prune_terminal_job_artifacts: bool = os.getenv(
        "AGENT_KERNEL_STORAGE_PRUNE_TERMINAL_JOB_ARTIFACTS",
        "1",
    ) == "1"
    storage_keep_cycle_export_files: int = int(
        os.getenv("AGENT_KERNEL_STORAGE_KEEP_CYCLE_EXPORT_FILES", "32")
    )
    storage_keep_report_export_files: int = int(
        os.getenv("AGENT_KERNEL_STORAGE_KEEP_REPORT_EXPORT_FILES", "64")
    )
    storage_keep_candidate_export_dirs: int = int(
        os.getenv("AGENT_KERNEL_STORAGE_KEEP_CANDIDATE_EXPORT_DIRS", "48")
    )
    storage_keep_run_report_files: int = int(
        os.getenv("AGENT_KERNEL_STORAGE_KEEP_RUN_REPORT_FILES", "256")
    )
    storage_keep_run_checkpoint_files: int = int(
        os.getenv("AGENT_KERNEL_STORAGE_KEEP_RUN_CHECKPOINT_FILES", "128")
    )
    storage_keep_namespace_candidate_dirs: int = int(
        os.getenv("AGENT_KERNEL_STORAGE_KEEP_NAMESPACE_CANDIDATE_DIRS", "16")
    )
    storage_max_cycle_export_records: int = int(
        os.getenv("AGENT_KERNEL_STORAGE_MAX_CYCLE_EXPORT_RECORDS", "512")
    )
    storage_max_report_history_records: int = int(
        os.getenv("AGENT_KERNEL_STORAGE_MAX_REPORT_HISTORY_RECORDS", "512")
    )
    storage_keep_tolbert_candidate_dirs: int = int(
        os.getenv("AGENT_KERNEL_STORAGE_KEEP_TOLBERT_CANDIDATE_DIRS", "2")
    )
    storage_tolbert_candidate_budget_bytes: int = int(
        os.getenv("AGENT_KERNEL_STORAGE_TOLBERT_CANDIDATE_BUDGET_BYTES", str(2 * 1024 * 1024 * 1024))
    )
    storage_tolbert_shared_store_budget_bytes: int = int(
        os.getenv("AGENT_KERNEL_STORAGE_TOLBERT_SHARED_STORE_BUDGET_BYTES", str(2 * 1024 * 1024 * 1024))
    )
    delegated_job_max_concurrency: int = int(
        os.getenv("AGENT_KERNEL_DELEGATED_JOB_MAX_CONCURRENCY", "3")
    )
    delegated_job_max_active_per_budget_group: int = int(
        os.getenv("AGENT_KERNEL_DELEGATED_JOB_MAX_ACTIVE_PER_BUDGET_GROUP", "2")
    )
    delegated_job_max_queued_per_budget_group: int = int(
        os.getenv("AGENT_KERNEL_DELEGATED_JOB_MAX_QUEUED_PER_BUDGET_GROUP", "8")
    )
    delegated_job_max_artifact_bytes: int = int(
        os.getenv("AGENT_KERNEL_DELEGATED_JOB_MAX_ARTIFACT_BYTES", str(8 * 1024 * 1024))
    )
    delegated_job_max_subprocesses_per_job: int = int(
        os.getenv("AGENT_KERNEL_DELEGATED_JOB_MAX_SUBPROCESSES_PER_JOB", "2")
    )
    delegated_job_max_consecutive_selections_per_budget_group: int = int(
        os.getenv("AGENT_KERNEL_DELEGATED_JOB_MAX_CONSECUTIVE_SELECTIONS_PER_BUDGET_GROUP", "0")
    )
    unattended_allowed_benchmark_families: tuple[str, ...] = _split_env_csv(
        "AGENT_KERNEL_UNATTENDED_ALLOWED_BENCHMARK_FAMILIES"
    ) or _DEFAULT_UNATTENDED_ALLOWED_BENCHMARK_FAMILIES
    unattended_allow_git_commands: bool = os.getenv(
        "AGENT_KERNEL_UNATTENDED_ALLOW_GIT_COMMANDS",
        "0",
    ) == "1"
    unattended_allow_generated_path_mutations: bool = os.getenv(
        "AGENT_KERNEL_UNATTENDED_ALLOW_GENERATED_PATH_MUTATIONS",
        "0",
    ) == "1"
    unattended_allow_http_requests: bool = os.getenv(
        "AGENT_KERNEL_UNATTENDED_ALLOW_HTTP_REQUESTS",
        "0",
    ) == "1"
    unattended_http_allowed_hosts: tuple[str, ...] = _split_env_csv(
        "AGENT_KERNEL_UNATTENDED_HTTP_ALLOWED_HOSTS"
    )
    unattended_http_timeout_seconds: int = int(
        os.getenv("AGENT_KERNEL_UNATTENDED_HTTP_TIMEOUT_SECONDS", "10")
    )
    unattended_http_max_body_bytes: int = int(
        os.getenv("AGENT_KERNEL_UNATTENDED_HTTP_MAX_BODY_BYTES", str(64 * 1024))
    )
    unattended_generated_path_prefixes: tuple[str, ...] = _split_env_csv(
        "AGENT_KERNEL_UNATTENDED_GENERATED_PATH_PREFIXES"
    ) or _DEFAULT_UNATTENDED_GENERATED_PATH_PREFIXES
    unattended_workspace_snapshot_root: Path = Path(
        os.getenv(
            "AGENT_KERNEL_UNATTENDED_WORKSPACE_SNAPSHOT_ROOT",
            "trajectories/recovery/workspaces",
        )
    )
    unattended_rollback_on_failure: bool = os.getenv(
        "AGENT_KERNEL_UNATTENDED_ROLLBACK_ON_FAILURE",
        "1",
    ) == "1"
    unattended_trust_enforce: bool = os.getenv(
        "AGENT_KERNEL_UNATTENDED_TRUST_ENFORCE",
        "1",
    ) == "1"
    unattended_trust_ledger_path: Path = Path(
        os.getenv(
            "AGENT_KERNEL_UNATTENDED_TRUST_LEDGER_PATH",
            "trajectories/reports/unattended_trust_ledger.json",
        )
    )
    unattended_trust_recent_report_limit: int = int(
        os.getenv("AGENT_KERNEL_UNATTENDED_TRUST_RECENT_REPORT_LIMIT", "50")
    )
    unattended_trust_required_benchmark_families: tuple[str, ...] = _split_env_csv(
        "AGENT_KERNEL_UNATTENDED_TRUST_REQUIRED_BENCHMARK_FAMILIES"
    ) or _DEFAULT_UNATTENDED_TRUST_REQUIRED_BENCHMARK_FAMILIES
    unattended_trust_bootstrap_min_reports: int = int(
        os.getenv("AGENT_KERNEL_UNATTENDED_TRUST_BOOTSTRAP_MIN_REPORTS", "5")
    )
    unattended_trust_breadth_min_reports: int = int(
        os.getenv("AGENT_KERNEL_UNATTENDED_TRUST_BREADTH_MIN_REPORTS", "10")
    )
    unattended_trust_min_distinct_families: int = int(
        os.getenv("AGENT_KERNEL_UNATTENDED_TRUST_MIN_DISTINCT_FAMILIES", "2")
    )
    unattended_trust_min_success_rate: float = float(
        os.getenv("AGENT_KERNEL_UNATTENDED_TRUST_MIN_SUCCESS_RATE", "0.7")
    )
    unattended_trust_max_unsafe_ambiguous_rate: float = float(
        os.getenv("AGENT_KERNEL_UNATTENDED_TRUST_MAX_UNSAFE_AMBIGUOUS_RATE", "0.1")
    )
    unattended_trust_max_hidden_side_effect_rate: float = float(
        os.getenv("AGENT_KERNEL_UNATTENDED_TRUST_MAX_HIDDEN_SIDE_EFFECT_RATE", "0.1")
    )
    unattended_trust_max_success_hidden_side_effect_rate: float = float(
        os.getenv("AGENT_KERNEL_UNATTENDED_TRUST_MAX_SUCCESS_HIDDEN_SIDE_EFFECT_RATE", "0.02")
    )
    min_skill_quality: float = float(os.getenv("AGENT_KERNEL_MIN_SKILL_QUALITY", "0.75"))

    @staticmethod
    def normalize_provider_name(provider: str) -> str:
        normalized = provider.strip().lower().replace("-", "_")
        if normalized == "tolbert":
            return "hybrid"
        if normalized == "modelstack":
            return "model_stack"
        return normalized

    @classmethod
    def executable_floor(cls, **overrides: object) -> "KernelConfig":
        floor_defaults: dict[str, object] = {
            "use_tolbert_context": False,
            "use_skills": False,
            "use_graph_memory": False,
            "use_world_model": False,
            "use_universe_model": False,
            "use_planner": False,
            "use_role_specialization": False,
            "use_research_library_context": False,
            "persist_learning_candidates": False,
        }
        floor_defaults.update(overrides)
        return cls(**floor_defaults)

    @classmethod
    def qwen_tolbert_reference_implementation(cls, **overrides: object) -> "KernelConfig":
        reference_defaults: dict[str, object] = {
            "provider": "vllm",
            "model_name": "Qwen/Qwen3.5-9B",
            "use_tolbert_context": True,
            "use_paper_research_context": False,
            "use_skills": False,
            "use_graph_memory": True,
            "use_world_model": True,
            "use_universe_model": True,
            "use_state_estimation_proposals": False,
            "use_trust_proposals": False,
            "use_recovery_proposals": False,
            "use_delegation_proposals": False,
            "use_operator_policy_proposals": False,
            "use_transition_model_proposals": False,
            "use_tolbert_model_artifacts": False,
            "use_planner": False,
            "use_role_specialization": False,
            "use_prompt_proposals": False,
            "use_curriculum_proposals": False,
            "use_retrieval_proposals": False,
            "use_research_library_context": False,
            "persist_episode_memory": True,
            "persist_learning_candidates": True,
            "asi_coding_require_live_llm": True,
        }
        reference_defaults.update(overrides)
        return cls(**reference_defaults)

    def normalized_provider(self) -> str:
        return self.normalize_provider_name(self.provider)

    def decoder_runtime_posture(self) -> dict[str, object]:
        provider = self.normalized_provider()
        runtime: dict[str, object] = {}
        surfaces: dict[str, object] = {}
        if bool(self.use_tolbert_model_artifacts):
            from .modeling.artifacts import (
                retained_tolbert_active_decoder_runtime,
                load_model_artifact,
                retained_tolbert_model_surfaces,
            )

            payload = load_model_artifact(self.tolbert_model_artifact_path)
            runtime = retained_tolbert_active_decoder_runtime(payload)
            surfaces = retained_tolbert_model_surfaces(payload)
        manifest_raw = str(runtime.get("bundle_manifest_path", "")).strip()
        manifest_path = Path(manifest_raw) if manifest_raw else Path()
        if manifest_raw and not manifest_path.is_absolute():
            manifest_path = (self.tolbert_model_artifact_path.parent / manifest_path).resolve()
        manifest_exists = bool(manifest_raw) and manifest_path.exists()
        runtime_key = str(runtime.get("runtime_key", "hybrid_runtime")).strip() or "hybrid_runtime"
        runtime_ready = (
            bool(runtime.get("materialized", False))
            if runtime_key == "universal_decoder_runtime"
            else bool(runtime.get("primary_enabled", False))
        )
        native_decoder_ready = (
            provider == "hybrid"
            and bool(self.use_tolbert_model_artifacts)
            and runtime_ready
            and bool(runtime.get("supports_decoder_surface", False))
            and bool(surfaces.get("decoder_surface", False))
            and manifest_exists
        )
        if native_decoder_ready:
            authority = "retained_native_decoder"
        elif provider in {"ollama", "vllm", "model_stack"}:
            authority = "external_decoder"
        elif provider == "mock":
            authority = "mock_decoder"
        else:
            authority = "unavailable_decoder"
        return {
            "provider": provider,
            "authority": authority,
            "native_decoder_ready": native_decoder_ready,
            "runtime_key": runtime_key,
            "training_objective": str(runtime.get("training_objective", "")).strip(),
            "bundle_manifest_path": str(manifest_path) if manifest_raw else "",
            "bundle_manifest_exists": manifest_exists,
            "primary_enabled": bool(runtime.get("primary_enabled", False)),
            "materialized": bool(runtime.get("materialized", False)),
            "supports_decoder_surface": bool(runtime.get("supports_decoder_surface", False)),
            "declares_decoder_surface": bool(surfaces.get("decoder_surface", False)),
        }

    def claimed_runtime_shape(self) -> str:
        if bool(self.use_universe_model) and bool(self.decoder_runtime_posture().get("native_decoder_ready", False)):
            return "bounded_autonomous"
        return "executable_floor"

    def validate(self) -> None:
        provider = self.normalized_provider()
        if provider not in {"mock", "ollama", "vllm", "hybrid", "model_stack"}:
            raise ValueError(f"unsupported provider: {self.provider}")
        if provider == "ollama" and not self.ollama_host.strip():
            raise ValueError("ollama_host must be non-empty when provider='ollama'")
        if provider == "vllm" and not self.vllm_host.strip():
            raise ValueError("vllm_host must be non-empty when provider='vllm'")
        if provider == "model_stack" and not self.model_stack_host.strip():
            raise ValueError("model_stack_host must be non-empty when provider='model_stack'")
        if provider == "vllm" and self.vllm_autostart and not self.vllm_python_bin.strip():
            raise ValueError("vllm_python_bin must be non-empty when vllm autostart is enabled")
        if not self.model_name.strip():
            raise ValueError("model_name must be non-empty")

        storage_backend = self.storage_backend.strip().lower()
        if storage_backend not in {"json", "sqlite"}:
            raise ValueError(f"unsupported storage backend: {self.storage_backend}")

        _require_positive_int("max_steps", self.max_steps)
        _require_positive_int("max_task_steps_hard_cap", self.max_task_steps_hard_cap)
        _require_positive_int("frontier_task_step_floor", self.frontier_task_step_floor)
        _require_positive_int("runtime_history_step_window", self.runtime_history_step_window)
        _require_positive_int("payload_history_step_window", self.payload_history_step_window)
        _require_positive_int("checkpoint_history_step_window", self.checkpoint_history_step_window)
        _require_positive_int("history_archive_summary_max_chars", self.history_archive_summary_max_chars)
        _require_positive_int("timeout_seconds", self.timeout_seconds)
        _require_positive_int("command_timeout_seconds", self.command_timeout_seconds)
        _require_positive_int("llm_timeout_seconds", self.llm_timeout_seconds)
        _require_non_negative_int("llm_retry_attempts", self.llm_retry_attempts)
        _require_non_negative_float("llm_retry_backoff_seconds", self.llm_retry_backoff_seconds)
        _require_positive_int("vllm_tensor_parallel_size", self.vllm_tensor_parallel_size)
        _require_positive_int("vllm_max_model_len", self.vllm_max_model_len)
        _require_positive_int("vllm_autostart_timeout_seconds", self.vllm_autostart_timeout_seconds)
        _require_non_negative_float(
            "vllm_autostart_poll_interval_seconds",
            self.vllm_autostart_poll_interval_seconds,
        )
        _require_probability("vllm_gpu_memory_utilization", self.vllm_gpu_memory_utilization)

        _require_positive_int("delegated_job_max_concurrency", self.delegated_job_max_concurrency)
        _require_positive_int("delegated_job_max_subprocesses_per_job", self.delegated_job_max_subprocesses_per_job)
        _require_non_negative_int(
            "delegated_job_max_active_per_budget_group",
            self.delegated_job_max_active_per_budget_group,
        )
        _require_non_negative_int(
            "delegated_job_max_queued_per_budget_group",
            self.delegated_job_max_queued_per_budget_group,
        )
        _require_non_negative_int(
            "delegated_job_max_consecutive_selections_per_budget_group",
            self.delegated_job_max_consecutive_selections_per_budget_group,
        )
        _require_positive_int("delegated_job_max_artifact_bytes", self.delegated_job_max_artifact_bytes)

        _require_positive_int("unattended_http_timeout_seconds", self.unattended_http_timeout_seconds)
        _require_positive_int("unattended_http_max_body_bytes", self.unattended_http_max_body_bytes)

        _require_probability("tolbert_confidence_threshold", self.tolbert_confidence_threshold)
        _require_probability("tolbert_branch_confidence_margin", self.tolbert_branch_confidence_margin)
        _require_probability(
            "tolbert_low_confidence_widen_threshold",
            self.tolbert_low_confidence_widen_threshold,
        )
        _require_probability(
            "tolbert_deterministic_command_confidence",
            self.tolbert_deterministic_command_confidence,
        )
        _require_probability(
            "tolbert_first_step_direct_command_confidence",
            self.tolbert_first_step_direct_command_confidence,
        )
        _require_probability(
            "tolbert_skill_ranking_min_confidence",
            self.tolbert_skill_ranking_min_confidence,
        )
        _require_probability("min_skill_quality", self.min_skill_quality)
        _require_probability("unattended_trust_min_success_rate", self.unattended_trust_min_success_rate)
        _require_probability(
            "unattended_trust_max_unsafe_ambiguous_rate",
            self.unattended_trust_max_unsafe_ambiguous_rate,
        )
        _require_probability(
            "unattended_trust_max_hidden_side_effect_rate",
            self.unattended_trust_max_hidden_side_effect_rate,
        )
        _require_probability(
            "unattended_trust_max_success_hidden_side_effect_rate",
            self.unattended_trust_max_success_hidden_side_effect_rate,
        )
        if self.claimed_runtime_shape() == "bounded_autonomous":
            if not bool(self.use_graph_memory):
                raise ValueError("bounded_autonomous runtime requires use_graph_memory=True")
            if not bool(self.use_world_model):
                raise ValueError("bounded_autonomous runtime requires use_world_model=True")

    def to_env(self) -> dict[str, str]:
        env: dict[str, str] = {}
        for field in fields(self):
            name = field.name
            env_name = _ENV_NAME_OVERRIDES.get(name, f"AGENT_KERNEL_{name.upper()}")
            env[env_name] = _serialize_env_value(name, getattr(self, name))
        return env

    def ensure_directories(self) -> None:
        self.validate()
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.trajectories_root.mkdir(parents=True, exist_ok=True)
        self.skills_path.parent.mkdir(parents=True, exist_ok=True)
        self.operator_classes_path.parent.mkdir(parents=True, exist_ok=True)
        self.tool_candidates_path.parent.mkdir(parents=True, exist_ok=True)
        self.benchmark_candidates_path.parent.mkdir(parents=True, exist_ok=True)
        self.retrieval_proposals_path.parent.mkdir(parents=True, exist_ok=True)
        self.retrieval_asset_bundle_path.parent.mkdir(parents=True, exist_ok=True)
        self.research_library_status_path.parent.mkdir(parents=True, exist_ok=True)
        self.tolbert_model_artifact_path.parent.mkdir(parents=True, exist_ok=True)
        self.tolbert_supervised_datasets_dir.mkdir(parents=True, exist_ok=True)
        self.qwen_adapter_artifact_path.parent.mkdir(parents=True, exist_ok=True)
        self.qwen_supervised_datasets_dir.mkdir(parents=True, exist_ok=True)
        self.tolbert_liftoff_report_path.parent.mkdir(parents=True, exist_ok=True)
        self.verifier_contracts_path.parent.mkdir(parents=True, exist_ok=True)
        self.prompt_proposals_path.parent.mkdir(parents=True, exist_ok=True)
        self.universe_constitution_path.parent.mkdir(parents=True, exist_ok=True)
        self.operating_envelope_path.parent.mkdir(parents=True, exist_ok=True)
        self.universe_contract_path.parent.mkdir(parents=True, exist_ok=True)
        self.world_model_proposals_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_estimation_proposals_path.parent.mkdir(parents=True, exist_ok=True)
        self.trust_proposals_path.parent.mkdir(parents=True, exist_ok=True)
        self.recovery_proposals_path.parent.mkdir(parents=True, exist_ok=True)
        self.delegation_proposals_path.parent.mkdir(parents=True, exist_ok=True)
        self.operator_policy_proposals_path.parent.mkdir(parents=True, exist_ok=True)
        self.transition_model_proposals_path.parent.mkdir(parents=True, exist_ok=True)
        self.curriculum_proposals_path.parent.mkdir(parents=True, exist_ok=True)
        self.improvement_cycles_path.parent.mkdir(parents=True, exist_ok=True)
        self.candidate_artifacts_root.mkdir(parents=True, exist_ok=True)
        self.improvement_reports_dir.mkdir(parents=True, exist_ok=True)
        self.strategy_memory_nodes_path.parent.mkdir(parents=True, exist_ok=True)
        self.strategy_memory_snapshots_path.parent.mkdir(parents=True, exist_ok=True)
        self.semantic_hub_root.mkdir(parents=True, exist_ok=True)
        self.run_reports_dir.mkdir(parents=True, exist_ok=True)
        self.learning_artifacts_path.parent.mkdir(parents=True, exist_ok=True)
        self.capability_modules_path.parent.mkdir(parents=True, exist_ok=True)
        self.run_checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.delegated_job_queue_path.parent.mkdir(parents=True, exist_ok=True)
        self.delegated_job_runtime_state_path.parent.mkdir(parents=True, exist_ok=True)
        self.runtime_database_path.parent.mkdir(parents=True, exist_ok=True)
        self.vllm_runtime_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.vllm_runtime_state_path.parent.mkdir(parents=True, exist_ok=True)
        self.vllm_runtime_lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.unattended_workspace_snapshot_root.mkdir(parents=True, exist_ok=True)
        self.unattended_trust_ledger_path.parent.mkdir(parents=True, exist_ok=True)

    def uses_sqlite_storage(self) -> bool:
        return self.storage_backend.strip().lower() == "sqlite"

    def sqlite_store(self):
        if not self.uses_sqlite_storage():
            raise ValueError(f"unsupported storage backend: {self.storage_backend}")
        from .storage import store_for_config

        return store_for_config(self)


_ENV_NAME_OVERRIDES = {
    "model_name": "AGENT_KERNEL_MODEL",
}


def _serialize_env_value(name: str, value: object) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        parts = [str(part) for part in value]
        separator = os.pathsep if name.endswith("_paths") else ","
        return separator.join(parts)
    return str(value)


def _require_positive_int(name: str, value: int) -> None:
    if int(value) <= 0:
        raise ValueError(f"{name} must be > 0")


def _require_non_negative_int(name: str, value: int) -> None:
    if int(value) < 0:
        raise ValueError(f"{name} must be >= 0")


def _require_non_negative_float(name: str, value: float) -> None:
    if float(value) < 0:
        raise ValueError(f"{name} must be >= 0")


def _require_probability(name: str, value: float) -> None:
    numeric = float(value)
    if numeric < 0.0 or numeric > 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
