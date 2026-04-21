from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable
from urllib import error as url_error
from urllib import request as url_request

from ..actions import CODE_EXECUTE
from ..extensions.capabilities import (
    capability_enabled,
    capability_policies_snapshot,
    capability_policy_satisfies,
    capability_registry_snapshot,
    capability_usage_summary,
    effective_http_allowed_hosts,
)
from ..config import KernelConfig
from ..extensions.operator_policy import operator_policy_snapshot
from .runtime_supervision import atomic_write_json
from ..schemas import EpisodeRecord, StepRecord, TaskSpec
from ..sandbox import sandbox_containment_status
from ..extensions.syntax_motor import build_syntax_motor_summary, syntax_preflight_check
from ..extensions.tolbert import _paper_research_runtime_paths
from ..extensions.tolbert_assets import retained_tolbert_runtime_paths
from ..extensions.trust import evaluate_unattended_trust, trust_policy_snapshot
from .vllm_runtime import ensure_vllm_runtime


@dataclass(slots=True)
class PreflightCheck:
    name: str
    passed: bool
    detail: str
    severity: str = "required"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PreflightReport:
    passed: bool
    checks: list[PreflightCheck]
    checked_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checked_at": self.checked_at,
            "checks": [check.to_dict() for check in self.checks],
        }


@dataclass(slots=True)
class WorkspaceFileSnapshot:
    relative_path: str
    size_bytes: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_unattended_preflight(
    config: KernelConfig,
    task: TaskSpec,
    *,
    repo_root: Path,
    urlopen: Callable[..., Any] | None = None,
) -> PreflightReport:
    opener = urlopen or url_request.urlopen
    checks = [
        _provider_health_check(config, opener=opener),
        _tolbert_assets_check(config, repo_root=repo_root),
        _paper_research_assets_check(config, repo_root=repo_root),
        _verifier_inputs_check(task),
        _workspace_readiness_check(config, task),
        _syntax_preflight_check(config, task),
        _execution_containment_check(config),
        _operator_policy_check(config, task),
        _trust_posture_check(config, task),
    ]
    return PreflightReport(
        passed=all(check.passed or check.severity != "required" for check in checks),
        checks=checks,
        checked_at=datetime.now(UTC).isoformat(),
    )


def provider_health_check(
    config: KernelConfig,
    *,
    urlopen: Callable[..., Any] | None = None,
) -> PreflightCheck:
    opener = urlopen or url_request.urlopen
    return _provider_health_check(config, opener=opener)


def classify_run_outcome(
    *,
    episode: EpisodeRecord | None,
    preflight: PreflightReport | None = None,
) -> tuple[str, list[str]]:
    if preflight is not None and not preflight.passed:
        failed_checks = _required_failed_preflight_checks(preflight)
        return "safe_stop", [f"preflight_failed:{name}" for name in failed_checks]
    if episode is None:
        return "safe_stop", ["no_episode_recorded"]
    if episode.success:
        return "success", ["verification_passed"]

    command_steps = [step for step in episode.steps if step.action == CODE_EXECUTE]
    if not command_steps:
        return "safe_stop", [episode.termination_reason or "policy_stopped_before_execution"]

    if all(_step_non_committal_failure(step) for step in command_steps):
        return "safe_stop", [episode.termination_reason or "non_committal_failure"]

    return "unsafe_ambiguous", [episode.termination_reason or "unverified_side_effect_risk"]


def build_unattended_task_report(
    *,
    task: TaskSpec,
    config: KernelConfig,
    episode: EpisodeRecord | None,
    preflight: PreflightReport | None,
    before_workspace_snapshot: dict[str, WorkspaceFileSnapshot] | None = None,
    clean_workspace: bool = True,
    outcome_override: str | None = None,
    outcome_reasons_override: list[str] | None = None,
    termination_reason_override: str | None = None,
    extra_uncertainties: list[str] | None = None,
) -> dict[str, Any]:
    outcome, outcome_reasons = classify_run_outcome(episode=episode, preflight=preflight)
    if outcome_override:
        outcome = outcome_override.strip()
    if outcome_reasons_override is not None:
        outcome_reasons = [reason.strip() for reason in outcome_reasons_override if reason.strip()]
    workspace = Path(episode.workspace) if episode is not None else config.workspace_root / task.workspace_subdir
    command_steps = [step for step in (episode.steps if episode is not None else []) if step.action == CODE_EXECUTE]
    after_workspace_snapshot = capture_workspace_snapshot(workspace)
    side_effects = summarize_workspace_side_effects(
        before_snapshot=before_workspace_snapshot or {},
        after_snapshot=after_workspace_snapshot,
        clean_workspace=clean_workspace,
        task=task,
    )
    operator_policy = unattended_operator_policy(config, task)
    trust_evaluation = evaluate_unattended_trust(
        config,
        benchmark_family=str(task.metadata.get("benchmark_family", "bounded")).strip() or "bounded",
        reports_dir=config.run_reports_dir,
    )
    trust_breadth_summary = _report_trust_breadth_summary(trust_evaluation, config=config)
    uncertainties = summarize_run_uncertainties(
        outcome=outcome,
        outcome_reasons=outcome_reasons,
        command_steps=command_steps,
        side_effects=side_effects,
        preflight=preflight,
    )
    if extra_uncertainties:
        uncertainties.extend(note.strip() for note in extra_uncertainties if note.strip())
    capability_usage = capability_usage_summary(task, command_steps)
    task_contract = _report_task_contract(task, episode)
    edit_synthesis = _report_edit_synthesis(task, episode)
    syntax_motor = build_syntax_motor_summary(
        task,
        workspace=workspace,
        task_contract=task_contract,
    )
    acceptance_packet = _report_acceptance_packet(
        task,
        episode,
        task_contract=task_contract,
        outcome=outcome,
        outcome_reasons=outcome_reasons,
        capability_usage=capability_usage,
    )
    trust_scope = _report_trust_scope(
        task=task,
        outcome=outcome,
        command_steps=command_steps,
        side_effects=side_effects,
    )
    supervision = _report_supervision(
        task=task,
        episode=episode,
        outcome=outcome,
        command_steps=command_steps,
        side_effects=side_effects,
        preflight=preflight,
    )
    return {
        "report_kind": "unattended_task_report",
        "generated_at": datetime.now(UTC).isoformat(),
        "task_id": task.task_id,
        "benchmark_family": str(task.metadata.get("benchmark_family", "bounded")),
        "prompt": task.prompt,
        "workspace": str(workspace),
        "outcome": outcome,
        "outcome_reasons": outcome_reasons,
        "success": bool(episode.success) if episode is not None else False,
        "termination_reason": (
            termination_reason_override.strip()
            if termination_reason_override
            else episode.termination_reason if episode is not None else "preflight_failed"
        ),
        "task_metadata": dict(task.metadata),
        "task_contract": task_contract,
        "trust_scope": trust_scope,
        "supervision": supervision,
        "edit_synthesis": edit_synthesis,
        "syntax_motor": syntax_motor,
        "acceptance_packet": acceptance_packet,
        "operator_policy": operator_policy,
        "trust": trust_evaluation,
        "trust_breadth_summary": trust_breadth_summary,
        "preflight": preflight.to_dict() if preflight is not None else None,
        "summary": {
            "command_steps": len(command_steps),
            "blocked_steps": sum(1 for step in command_steps if _step_blocked(step)),
            "timed_out_steps": sum(1 for step in command_steps if _step_timed_out(step)),
            "zero_exit_steps": sum(1 for step in command_steps if _step_exit_code(step) == 0),
            "verified_steps": sum(1 for step in command_steps if _step_verified(step)),
            "execution_source_summary": _execution_source_summary(command_steps),
            "workspace_files": len(after_workspace_snapshot),
            "created_files": len(side_effects["created_files"]),
            "modified_files": len(side_effects["modified_files"]),
            "deleted_files": len(side_effects["deleted_files"]),
            "accounted_change_files": len(side_effects["accounted_change_files"]),
            "unexpected_change_files": len(side_effects["unexpected_change_files"]),
            "hidden_side_effect_risk": side_effects["hidden_side_effect_risk"],
        },
        "commands": [_serialize_command_step(step) for step in command_steps],
        "capability_usage": capability_usage,
        "side_effects": side_effects,
        "uncertainties": uncertainties,
        "workspace_files": sorted(after_workspace_snapshot),
    }


def _report_task_contract(task: TaskSpec, episode: EpisodeRecord | None) -> dict[str, Any]:
    if episode is not None and isinstance(episode.task_contract, dict) and episode.task_contract:
        return dict(episode.task_contract)
    metadata = dict(task.metadata)
    return {
        "prompt": task.prompt,
        "workspace_subdir": task.workspace_subdir,
        "setup_commands": list(task.setup_commands),
        "success_command": task.success_command,
        "suggested_commands": list(task.suggested_commands),
        "expected_files": list(task.expected_files),
        "expected_output_substrings": list(task.expected_output_substrings),
        "forbidden_files": list(task.forbidden_files),
        "forbidden_output_substrings": list(task.forbidden_output_substrings),
        "expected_file_contents": dict(task.expected_file_contents),
        "max_steps": task.max_steps,
        "metadata": metadata,
        "synthetic_edit_plan": [
            dict(step)
            for step in metadata.get("synthetic_edit_plan", [])
            if isinstance(step, dict)
        ],
        "synthetic_edit_candidates": [
            dict(step)
            for step in metadata.get("synthetic_edit_candidates", [])
            if isinstance(step, dict)
        ],
    }


def _report_edit_synthesis(task: TaskSpec, episode: EpisodeRecord | None) -> dict[str, Any] | None:
    contract = _report_task_contract(task, episode)
    metadata = contract.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    selected = [
        dict(step)
        for step in contract.get("synthetic_edit_plan", [])
        if isinstance(step, dict)
    ]
    candidates = [
        dict(step)
        for step in contract.get("synthetic_edit_candidates", [])
        if isinstance(step, dict)
    ]
    if not selected and not candidates:
        return None
    return {
        "synthetic_worker": bool(metadata.get("synthetic_worker")),
        "source_integrator_task_id": str(metadata.get("source_integrator_task_id", "")).strip(),
        "selected_edits": selected,
        "candidate_sets": candidates,
    }


def _report_acceptance_packet(
    task: TaskSpec,
    episode: EpisodeRecord | None,
    *,
    task_contract: dict[str, Any],
    outcome: str,
    outcome_reasons: list[str],
    capability_usage: dict[str, Any],
) -> dict[str, Any]:
    metadata = dict(task_contract.get("metadata", {})) if isinstance(task_contract.get("metadata", {}), dict) else {}
    workflow_guard = dict(metadata.get("workflow_guard", {})) if isinstance(metadata.get("workflow_guard", {}), dict) else {}
    semantic_verifier = (
        dict(metadata.get("semantic_verifier", {}))
        if isinstance(metadata.get("semantic_verifier", {}), dict)
        else {}
    )
    selected_edits = [
        dict(step)
        for step in task_contract.get("synthetic_edit_plan", [])
        if isinstance(step, dict)
    ]
    candidate_sets = [
        dict(step)
        for step in task_contract.get("synthetic_edit_candidates", [])
        if isinstance(step, dict)
    ]
    test_commands = [
        dict(command)
        for command in semantic_verifier.get("test_commands", [])
        if isinstance(command, dict)
    ]
    report_rules = [
        dict(rule)
        for rule in semantic_verifier.get("report_rules", [])
        if isinstance(rule, dict)
    ]
    verifier_reasons: list[str] = []
    for step in episode.steps if episode is not None else []:
        verification = step.verification or {}
        if isinstance(verification, dict):
            verifier_reasons.extend(
                str(reason).strip()
                for reason in verification.get("reasons", [])
                if str(reason).strip()
            )
    if not verifier_reasons:
        verifier_reasons = [reason for reason in outcome_reasons if reason]
    return {
        "task_id": task.task_id,
        "synthetic_worker": bool(metadata.get("synthetic_worker")),
        "source_integrator_task_id": str(metadata.get("source_integrator_task_id", "")).strip(),
        "expected_branch": str(semantic_verifier.get("expected_branch", "")).strip(),
        "target_branch": str(
            workflow_guard.get("target_branch") or semantic_verifier.get("target_branch") or ""
        ).strip(),
        "required_merged_branches": [
            str(branch).strip()
            for branch in semantic_verifier.get("required_merged_branches", [])
            if str(branch).strip()
        ],
        "diff_base_ref": str(semantic_verifier.get("diff_base_ref", "")).strip(),
        "selected_edits": selected_edits,
        "candidate_edit_sets": candidate_sets,
        "tests": test_commands,
        "report_rules": report_rules,
        "verifier_result": {
            "kind": str(semantic_verifier.get("kind", "")).strip(),
            "passed": bool(episode.success) if episode is not None else outcome == "success",
            "outcome": outcome,
            "reasons": verifier_reasons,
        },
        "capability_usage": capability_usage,
    }


def _report_trust_scope(
    *,
    task: TaskSpec,
    outcome: str,
    command_steps: list[StepRecord],
    side_effects: dict[str, Any],
) -> str:
    metadata = dict(task.metadata)
    if outcome == "safe_stop" and not command_steps:
        return "coverage_only"
    is_retrieval_companion = bool(metadata.get("requires_retrieval")) and bool(
        str(metadata.get("source_task", "")).strip()
    )
    changed_files = (
        len(side_effects.get("created_files", []))
        + len(side_effects.get("modified_files", []))
        + len(side_effects.get("deleted_files", []))
    )
    if is_retrieval_companion and outcome == "safe_stop" and not command_steps and changed_files == 0:
        return "coverage_only"
    return "gated"


def _report_supervision(
    *,
    task: TaskSpec,
    episode: EpisodeRecord | None,
    outcome: str,
    command_steps: list[StepRecord],
    side_effects: dict[str, Any],
    preflight: PreflightReport | None,
) -> dict[str, Any]:
    metadata = dict(task.metadata)
    mode = _normalize_supervision_mode(metadata)
    operator_turns = _supervision_operator_turns(metadata)
    verifier_defined = _task_has_acceptance_contract(task, episode)
    changed_files = int(side_effects.get("changed_files_total", 0) or 0)
    hidden_side_effect_risk = bool(side_effects.get("hidden_side_effect_risk", False))
    unexpected_change_file_count = len(side_effects.get("unexpected_change_files", []))
    independent_execution = operator_turns <= 1
    contract_candidate = metadata.get("light_supervision_candidate")
    if isinstance(contract_candidate, bool):
        light_supervision_candidate = mode in {"light_supervision", "unattended"} and contract_candidate
    else:
        light_supervision_candidate = mode in {"light_supervision", "unattended"} and verifier_defined
    successful_outcome = bool(episode.success) if episode is not None else outcome == "success"
    clean_success = successful_outcome and not hidden_side_effect_risk and unexpected_change_file_count <= 0
    light_supervision_success = light_supervision_candidate and independent_execution and successful_outcome
    light_supervision_clean_success = light_supervision_candidate and independent_execution and clean_success
    contract_clean_failure_recovery_origin = bool(metadata.get("contract_clean_failure_recovery_origin", False))
    contract_clean_failure_recovery_step_floor = 0
    try:
        contract_clean_failure_recovery_step_floor = max(
            0,
            int(
                metadata.get(
                    "contract_clean_failure_recovery_step_floor",
                    metadata.get("step_floor", 0),
                )
                or 0
            ),
        )
    except (TypeError, ValueError):
        contract_clean_failure_recovery_step_floor = 0
    contract_clean_failure_recovery_candidate = (
        mode in {"light_supervision", "unattended"} and contract_clean_failure_recovery_origin
    )
    contract_clean_failure_recovery_success = (
        contract_clean_failure_recovery_candidate and independent_execution and successful_outcome
    )
    contract_clean_failure_recovery_clean_success = (
        contract_clean_failure_recovery_candidate and independent_execution and clean_success
    )
    return {
        "mode": mode,
        "operator_turns": operator_turns,
        "verifier_defined": verifier_defined,
        "independent_execution": independent_execution,
        "preflight_ran": preflight is not None,
        "preflight_passed": bool(preflight.passed) if preflight is not None else False,
        "command_steps": len(command_steps),
        "changed_files": changed_files,
        "hidden_side_effect_risk": hidden_side_effect_risk,
        "light_supervision_candidate": light_supervision_candidate,
        "light_supervision_success": light_supervision_success,
        "light_supervision_clean_success": light_supervision_clean_success,
        "contract_clean_failure_recovery_origin": contract_clean_failure_recovery_origin,
        "contract_clean_failure_recovery_step_floor": contract_clean_failure_recovery_step_floor,
        "contract_clean_failure_recovery_candidate": contract_clean_failure_recovery_candidate,
        "contract_clean_failure_recovery_success": contract_clean_failure_recovery_success,
        "contract_clean_failure_recovery_clean_success": contract_clean_failure_recovery_clean_success,
    }


def _normalize_supervision_mode(metadata: dict[str, Any]) -> str:
    for key in ("supervision_mode", "guidance_mode", "operator_guidance_mode"):
        raw = str(metadata.get(key, "")).strip().lower()
        if raw in {"unattended", "light_supervision", "guided"}:
            return raw
    operator_turns = _supervision_operator_turns(metadata)
    if operator_turns <= 0:
        return "unattended"
    if operator_turns <= 1:
        return "light_supervision"
    return "guided"


def _supervision_operator_turns(metadata: dict[str, Any]) -> int:
    for key in ("operator_turns", "operator_interventions", "guidance_turns"):
        raw = metadata.get(key)
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            continue
    return 0


def _task_has_acceptance_contract(task: TaskSpec, episode: EpisodeRecord | None) -> bool:
    contract = _report_task_contract(task, episode)
    return bool(
        str(contract.get("success_command", "")).strip()
        or contract.get("expected_files")
        or contract.get("expected_output_substrings")
        or contract.get("forbidden_files")
        or contract.get("forbidden_output_substrings")
        or contract.get("expected_file_contents")
    )


def write_unattended_task_report(
    *,
    task: TaskSpec,
    config: KernelConfig,
    episode: EpisodeRecord | None,
    preflight: PreflightReport | None,
    report_path: Path | None = None,
    before_workspace_snapshot: dict[str, WorkspaceFileSnapshot] | None = None,
    clean_workspace: bool = True,
    outcome_override: str | None = None,
    outcome_reasons_override: list[str] | None = None,
    termination_reason_override: str | None = None,
    extra_uncertainties: list[str] | None = None,
) -> Path:
    safe_task_id = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in task.task_id)
    target = report_path or (
        config.run_reports_dir
        / f"task_report_{safe_task_id}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}.json"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = build_unattended_task_report(
        task=task,
        config=config,
        episode=episode,
        preflight=preflight,
        before_workspace_snapshot=before_workspace_snapshot,
        clean_workspace=clean_workspace,
        outcome_override=outcome_override,
        outcome_reasons_override=outcome_reasons_override,
        termination_reason_override=termination_reason_override,
        extra_uncertainties=extra_uncertainties,
    )
    atomic_write_json(target, payload, config=config, govern_storage=False)
    return target


def capture_workspace_snapshot(workspace: Path) -> dict[str, WorkspaceFileSnapshot]:
    if not workspace.exists():
        return {}
    snapshot: dict[str, WorkspaceFileSnapshot] = {}
    for path in sorted(workspace.rglob("*")):
        if not path.is_file():
            continue
        relative_path = str(path.relative_to(workspace))
        snapshot[relative_path] = WorkspaceFileSnapshot(
            relative_path=relative_path,
            size_bytes=path.stat().st_size,
            sha256=_file_sha256(path),
        )
    return snapshot


def summarize_workspace_side_effects(
    *,
    before_snapshot: dict[str, WorkspaceFileSnapshot],
    after_snapshot: dict[str, WorkspaceFileSnapshot],
    clean_workspace: bool,
    task: TaskSpec | None = None,
) -> dict[str, Any]:
    before_paths = set(before_snapshot)
    after_paths = set(after_snapshot)
    created = sorted(after_paths - before_paths)
    deleted = sorted(before_paths - after_paths)
    modified = sorted(
        path
        for path in before_paths & after_paths
        if before_snapshot[path].sha256 != after_snapshot[path].sha256
    )
    unchanged = len(before_paths & after_paths) - len(modified)
    unexpected_created, unexpected_modified, unexpected_deleted = _unexpected_side_effects(
        task=task,
        created=created,
        modified=modified,
        deleted=deleted,
    )
    unexpected_change_files = sorted(set(unexpected_created + unexpected_modified + unexpected_deleted))
    managed_paths, delete_paths = _side_effect_contract_paths(task)
    accounted_change_files = sorted(
        path
        for path in [*created, *modified, *deleted]
        if path not in unexpected_change_files
    )
    return {
        "before_file_count": len(before_snapshot),
        "after_file_count": len(after_snapshot),
        "created_files": created,
        "modified_files": modified,
        "deleted_files": deleted,
        "unchanged_file_count": unchanged,
        "changed_files_total": len(created) + len(modified) + len(deleted),
        "clean_workspace_requested": clean_workspace,
        "workspace_reset_detected": bool(clean_workspace and before_snapshot and deleted),
        "contract_managed_paths": sorted(managed_paths),
        "contract_delete_paths": sorted(delete_paths),
        "accounted_change_files": accounted_change_files,
        "all_changes_accounted_for": not unexpected_change_files,
        "unexpected_created_files": unexpected_created,
        "unexpected_modified_files": unexpected_modified,
        "unexpected_deleted_files": unexpected_deleted,
        "unexpected_change_files": unexpected_change_files,
        "hidden_side_effect_risk": bool(unexpected_change_files),
        "changed_file_details": _changed_file_details(
            before_snapshot=before_snapshot,
            after_snapshot=after_snapshot,
            created=created,
            modified=modified,
            deleted=deleted,
            managed_paths=managed_paths,
            delete_paths=delete_paths,
        ),
    }


def summarize_run_uncertainties(
    *,
    outcome: str,
    outcome_reasons: list[str],
    command_steps: list[StepRecord],
    side_effects: dict[str, Any],
    preflight: PreflightReport | None,
) -> list[str]:
    notes: list[str] = []
    if outcome == "unsafe_ambiguous":
        notes.append(
            "run ended without verified success after at least one committal command; inspect side effects manually"
        )
    if side_effects["deleted_files"]:
        notes.append("workspace deletions were observed from before/after snapshots; command intent may require review")
    if side_effects["hidden_side_effect_risk"]:
        notes.append(
            "unexpected workspace changes fell outside the declared task contract; hidden side effects should be reviewed"
        )
    if any(_step_timed_out(step) for step in command_steps):
        notes.append("timed out commands may have produced partial side effects before termination")
    if any(_step_blocked(step) for step in command_steps):
        notes.append("blocked commands were refused before execution and do not prove user-intended side effects")
    if preflight is not None and not preflight.passed:
        notes.extend(outcome_reasons)
    return notes


def _provider_health_check(
    config: KernelConfig,
    *,
    opener: Callable[..., Any],
) -> PreflightCheck:
    provider = config.normalized_provider()
    if provider == "mock":
        return PreflightCheck(name="provider_health", passed=True, detail="mock provider requires no health probe")
    if provider == "hybrid":
        return PreflightCheck(
            name="provider_health",
            passed=True,
            detail="hybrid provider uses retained runtime surfaces and requires no external provider health probe",
        )
    if provider == "ollama":
        return _http_health_check(
            name="provider_health",
            detail_label="ollama",
            url=f"{config.ollama_host.rstrip('/')}/api/tags",
            opener=opener,
        )
    if provider == "vllm":
        status = ensure_vllm_runtime(config, opener=opener)
        return PreflightCheck(
            name="provider_health",
            passed=bool(status.ready),
            detail=str(status.detail),
        )
    return PreflightCheck(
        name="provider_health",
        passed=False,
        detail=f"unsupported provider {config.provider!r}",
    )


def _execution_containment_check(config: KernelConfig) -> PreflightCheck:
    status = sandbox_containment_status(config, cwd=config.workspace_root)
    mode = str(status.get("mode", "disabled")).strip() or "disabled"
    available = bool(status.get("available", False))
    detail = str(status.get("detail", "")).strip() or "execution containment status unavailable"
    if mode == "required":
        return PreflightCheck(name="execution_containment", passed=available, detail=detail)
    if mode == "disabled":
        return PreflightCheck(
            name="execution_containment",
            passed=True,
            detail=detail,
            severity="optional",
        )
    return PreflightCheck(
        name="execution_containment",
        passed=available,
        detail=detail,
        severity="optional",
    )


def _required_failed_preflight_checks(preflight: PreflightReport) -> list[str]:
    required_failures = [
        check.name
        for check in preflight.checks
        if not check.passed and str(check.severity).strip().lower() == "required"
    ]
    if required_failures:
        return required_failures
    return [check.name for check in preflight.checks if not check.passed]


def _http_health_check(
    *,
    name: str,
    detail_label: str,
    url: str,
    opener: Callable[..., Any],
    headers: dict[str, str] | None = None,
) -> PreflightCheck:
    req = url_request.Request(url=url, headers=headers or {}, method="GET")
    try:
        with opener(req, timeout=5) as response:
            status = getattr(response, "status", 200)
        return PreflightCheck(name=name, passed=True, detail=f"{detail_label} responded with status {status}")
    except (TimeoutError, url_error.URLError, OSError) as exc:
        return PreflightCheck(name=name, passed=False, detail=f"{detail_label} probe failed: {exc}")


def _tolbert_assets_check(config: KernelConfig, *, repo_root: Path) -> PreflightCheck:
    if not config.use_tolbert_context:
        return PreflightCheck(name="tolbert_assets", passed=True, detail="tolbert context disabled")
    runtime_paths = retained_tolbert_runtime_paths(config, repo_root=repo_root)
    required_paths: list[tuple[str, str | None]] = [
        ("tolbert_python_bin", config.tolbert_python_bin),
        ("tolbert_config_path", runtime_paths["tolbert_config_path"]),
        ("tolbert_checkpoint_path", runtime_paths["tolbert_checkpoint_path"]),
        ("tolbert_nodes_path", runtime_paths["tolbert_nodes_path"]),
        ("tolbert_label_map_path", runtime_paths["tolbert_label_map_path"]),
        *[
            (f"tolbert_source_spans_paths[{index}]", raw)
            for index, raw in enumerate(runtime_paths["tolbert_source_spans_paths"])
        ],
        *[(f"tolbert_cache_paths[{index}]", raw) for index, raw in enumerate(runtime_paths["tolbert_cache_paths"])],
    ]
    missing: list[str] = []
    for label, raw_path in required_paths:
        resolved = _resolve_repo_path(raw_path, repo_root=repo_root)
        if resolved is None or not resolved.exists():
            missing.append(f"{label}={resolved if resolved is not None else '<unset>'}")
    service_script = repo_root / "scripts" / "tolbert_service.py"
    if not service_script.exists():
        missing.append(f"tolbert_service_script={service_script}")
    if missing:
        return PreflightCheck(
            name="tolbert_assets",
            passed=False,
            detail="missing required tolbert assets: " + ", ".join(missing),
        )
    runtime_check = _tolbert_runtime_import_check(repo_root=repo_root)
    if not runtime_check.passed:
        return runtime_check
    return PreflightCheck(name="tolbert_assets", passed=True, detail="required tolbert assets and runtime imports present")


def _tolbert_runtime_import_check(*, repo_root: Path) -> PreflightCheck:
    tolbert_root = repo_root / "other_repos" / "TOLBERT"
    tolbert_brain_root = Path("/data/TOLBERT_BRAIN")
    search_roots = [tolbert_root, tolbert_brain_root]
    missing_roots = [str(path) for path in search_roots if not path.exists()]
    if missing_roots:
        return PreflightCheck(
            name="tolbert_assets",
            passed=False,
            detail="missing tolbert runtime roots: " + ", ".join(missing_roots),
        )
    for path in reversed(search_roots):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    missing_runtime_paths: list[str] = []
    expected_paths = {
        "tolbert": tolbert_root / "tolbert" / "__init__.py",
        "tolbert_brain.brain": tolbert_brain_root / "tolbert_brain" / "brain.py",
    }
    for label, path in expected_paths.items():
        if not path.exists():
            missing_runtime_paths.append(f"{label}={path}")
    if missing_runtime_paths:
        return PreflightCheck(
            name="tolbert_assets",
            passed=False,
            detail="missing tolbert runtime modules: " + ", ".join(missing_runtime_paths),
        )
    return PreflightCheck(name="tolbert_assets", passed=True, detail="tolbert runtime imports resolved")


def _paper_research_assets_check(config: KernelConfig, *, repo_root: Path) -> PreflightCheck:
    if not config.use_tolbert_context or not config.use_paper_research_context:
        return PreflightCheck(
            name="paper_research_assets",
            passed=True,
            detail="paper research context disabled",
            severity="optional",
        )
    mode = config.paper_research_query_mode.strip().lower()
    if mode in {"0", "false", "never", "off"}:
        return PreflightCheck(
            name="paper_research_assets",
            passed=True,
            detail="paper research context disabled by query mode",
            severity="optional",
        )
    runtime_paths = _paper_research_runtime_paths(config, repo_root=repo_root)
    if runtime_paths is not None:
        return PreflightCheck(
            name="paper_research_assets",
            passed=True,
            detail="paper research runtime assets present",
            severity="optional",
        )
    detail = "paper research runtime assets unavailable; auxiliary paper grounding will be skipped"
    severity = "optional"
    if mode in {"1", "always", "on", "true"}:
        detail = "paper research runtime assets unavailable while paper_research_query_mode requires them"
        severity = "required"
    return PreflightCheck(
        name="paper_research_assets",
        passed=False,
        detail=detail,
        severity=severity,
    )


def _verifier_inputs_check(task: TaskSpec) -> PreflightCheck:
    verifier_inputs = (
        len(task.expected_files)
        + len(task.expected_output_substrings)
        + len(task.forbidden_files)
        + len(task.forbidden_output_substrings)
        + len(task.expected_file_contents)
    )
    if verifier_inputs > 0:
        return PreflightCheck(
            name="verifier_inputs",
            passed=True,
            detail=f"task exposes {verifier_inputs} verifier input(s)",
        )
    if task.success_command:
        return PreflightCheck(
            name="verifier_inputs",
            passed=False,
            detail="task only defines success_command; verifier contract fields are empty",
        )
    return PreflightCheck(
        name="verifier_inputs",
        passed=False,
        detail="task does not define any verifier contract inputs",
    )


def _workspace_readiness_check(config: KernelConfig, task: TaskSpec) -> PreflightCheck:
    workspace_subdir = task.workspace_subdir.strip()
    workspace_path = Path(workspace_subdir)
    if workspace_path.is_absolute():
        return PreflightCheck(
            name="workspace_readiness",
            passed=False,
            detail=f"workspace_subdir must be relative: {workspace_subdir}",
        )
    if ".." in workspace_path.parts:
        return PreflightCheck(
            name="workspace_readiness",
            passed=False,
            detail=f"workspace_subdir must not escape workspace root: {workspace_subdir}",
        )
    target = config.workspace_root / workspace_path
    if target.exists() and not target.is_dir():
        return PreflightCheck(
            name="workspace_readiness",
            passed=False,
            detail=f"workspace path exists as non-directory: {target}",
        )
    try:
        config.ensure_directories()
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return PreflightCheck(
            name="workspace_readiness",
            passed=False,
            detail=f"workspace root is not writable: {exc}",
        )
    return PreflightCheck(
        name="workspace_readiness",
        passed=True,
        detail=f"workspace target is ready: {target}",
    )


def _syntax_preflight_check(config: KernelConfig, task: TaskSpec) -> PreflightCheck:
    workspace = config.workspace_root / task.workspace_subdir
    passed, status, detail = syntax_preflight_check(
        task,
        workspace=workspace if workspace.exists() else None,
    )
    severity = "optional" if status == "no_python_edit_targets" else "required"
    return PreflightCheck(
        name="syntax_preflight",
        passed=passed,
        detail=detail,
        severity=severity,
    )


def unattended_operator_policy(config: KernelConfig, task: TaskSpec) -> dict[str, Any]:
    benchmark_family = str(task.metadata.get("benchmark_family", "bounded")).strip() or "bounded"
    snapshot = operator_policy_snapshot(config)
    allowed_families = [
        str(family).strip()
        for family in snapshot.get("unattended_allowed_benchmark_families", [])
        if str(family).strip()
    ]
    task_origin = str(task.metadata.get("task_origin", "")).strip()
    family_allowed_by_manifest = False
    if benchmark_family not in set(allowed_families) and task_origin == "external_manifest":
        allowed_families.append(benchmark_family)
        family_allowed_by_manifest = True
    workflow_guard = _workflow_guard(task)
    return {
        "benchmark_family": benchmark_family,
        "allowed_benchmark_families": allowed_families,
        "family_allowed": benchmark_family in set(allowed_families),
        "family_allowed_by_manifest": family_allowed_by_manifest,
        "allow_git_commands": bool(snapshot.get("unattended_allow_git_commands", False)),
        "allow_http_requests": bool(snapshot.get("unattended_allow_http_requests", False)),
        "http_allowed_hosts": effective_http_allowed_hosts(config),
        "http_timeout_seconds": int(snapshot.get("unattended_http_timeout_seconds", 1)),
        "http_max_body_bytes": int(snapshot.get("unattended_http_max_body_bytes", 1)),
        "allow_generated_path_mutations": bool(
            snapshot.get("unattended_allow_generated_path_mutations", False)
        ),
        "generated_path_prefixes": list(snapshot.get("unattended_generated_path_prefixes", [])),
        "capabilities": capability_registry_snapshot(config),
        "capability_policies": capability_policies_snapshot(config),
        "trust_policy": trust_policy_snapshot(config),
        "workflow_guard": workflow_guard,
    }


def _operator_policy_check(config: KernelConfig, task: TaskSpec) -> PreflightCheck:
    policy = unattended_operator_policy(config, task)
    capability_registry = (
        dict(policy.get("capabilities", {}))
        if isinstance(policy.get("capabilities", {}), dict)
        else {}
    )
    benchmark_family = str(policy["benchmark_family"])
    if not bool(policy["family_allowed"]):
        return PreflightCheck(
            name="operator_policy",
            passed=False,
            detail=(
                f"benchmark family {benchmark_family!r} is not allowed for unattended execution; "
                f"allowed={policy['allowed_benchmark_families']}"
            ),
        )
    workflow_guard = policy["workflow_guard"]
    if isinstance(workflow_guard, dict) and workflow_guard.get("requires_git") and not bool(
        policy.get("allow_git_commands", False)
    ):
        return PreflightCheck(
            name="operator_policy",
            passed=False,
            detail="task workflow requires git commands but unattended git policy is disabled",
        )
    if isinstance(workflow_guard, dict) and workflow_guard.get("requires_http") and not bool(
        policy.get("allow_http_requests", False)
    ):
        return PreflightCheck(
            name="operator_policy",
            passed=False,
            detail="task workflow requires http requests but unattended http policy is disabled",
        )
    invalid_enabled_modules = (
        list(capability_registry.get("invalid_enabled_modules", []))
        if isinstance(capability_registry.get("invalid_enabled_modules", []), list)
        else []
    )
    if invalid_enabled_modules:
        details = [
            f"{str(module.get('module_id', '')).strip()}:{','.join(str(value) for value in module.get('issues', []))}"
            for module in invalid_enabled_modules
            if isinstance(module, dict)
        ]
        return PreflightCheck(
            name="operator_policy",
            passed=False,
            detail=f"enabled capability modules invalid: {details}",
        )
    required_capabilities = [
        str(value).strip()
        for value in workflow_guard.get("required_capabilities", [])
        if str(value).strip()
    ]
    missing_capabilities = [
        capability for capability in required_capabilities if not capability_enabled(config, capability)
    ]
    if missing_capabilities:
        return PreflightCheck(
            name="operator_policy",
            passed=False,
            detail=f"task workflow requires missing capabilities: {missing_capabilities}",
        )
    required_capability_policies = (
        dict(workflow_guard.get("required_capability_policies", {}))
        if isinstance(workflow_guard.get("required_capability_policies", {}), dict)
        else {}
    )
    unsatisfied_policy_requirements: list[str] = []
    for capability, requirements in required_capability_policies.items():
        normalized_capability = str(capability).strip()
        if not normalized_capability:
            continue
        requirement_map = dict(requirements) if isinstance(requirements, dict) else {}
        if not capability_policy_satisfies(config, normalized_capability, requirement_map):
            unsatisfied_policy_requirements.append(
                f"{normalized_capability}:{json.dumps(requirement_map, sort_keys=True)}"
            )
    if unsatisfied_policy_requirements:
        return PreflightCheck(
            name="operator_policy",
            passed=False,
            detail="task workflow requires unsatisfied capability policy scope: "
            f"{unsatisfied_policy_requirements}",
        )
    if (
        isinstance(workflow_guard, dict)
        and workflow_guard.get("touches_generated_paths")
        and not bool(policy.get("allow_generated_path_mutations", False))
    ):
        return PreflightCheck(
            name="operator_policy",
            passed=False,
            detail="task workflow requires generated-path mutations but unattended generated-path policy is disabled",
        )
    return PreflightCheck(
        name="operator_policy",
        passed=True,
        detail=f"benchmark family {benchmark_family!r} allowed under unattended operator policy",
    )


def _trust_posture_check(config: KernelConfig, task: TaskSpec) -> PreflightCheck:
    family = str(task.metadata.get("benchmark_family", "bounded")).strip() or "bounded"
    evaluation = evaluate_unattended_trust(
        config,
        benchmark_family=family,
        reports_dir=config.run_reports_dir,
    )
    detail = str(evaluation["detail"])
    breadth_detail = _trust_breadth_detail(
        _report_trust_breadth_summary(evaluation, config=config),
        family=family,
    )
    if breadth_detail:
        detail = f"{detail} | {breadth_detail}"
    bootstrap_override_detail = _trust_proof_bootstrap_override_detail(evaluation=evaluation, task=task)
    if bootstrap_override_detail:
        return PreflightCheck(
            name="trust_posture",
            passed=True,
            detail=f"{detail} | {bootstrap_override_detail}",
            severity="required",
        )
    return PreflightCheck(
        name="trust_posture",
        passed=bool(evaluation["passed"]),
        detail=detail,
        severity="required",
    )


def _trust_proof_bootstrap_override_detail(
    *,
    evaluation: dict[str, Any],
    task: TaskSpec,
) -> str:
    if bool(evaluation.get("passed", False)) or not bool(evaluation.get("required", False)):
        return ""
    family_assessment = (
        dict(evaluation.get("family_assessment", {}))
        if isinstance(evaluation.get("family_assessment", {}), dict)
        else {}
    )
    overall_assessment = (
        dict(evaluation.get("overall_assessment", {}))
        if isinstance(evaluation.get("overall_assessment", {}), dict)
        else {}
    )
    if not bool(family_assessment.get("passed", False)) or bool(overall_assessment.get("passed", False)):
        return ""
    overall_failures = [
        str(value).strip()
        for value in overall_assessment.get("failing_thresholds", [])
        if str(value).strip()
    ]
    if not overall_failures or any(
        "distinct_benchmark_families=" not in value or "min_distinct_families=" not in value
        for value in overall_failures
    ):
        return ""
    metadata = dict(task.metadata) if isinstance(task.metadata, dict) else {}
    if not (
        bool(metadata.get("decision_yield_contract_candidate", False))
        or bool(metadata.get("light_supervision_candidate", False))
        or str(metadata.get("task_origin", "")).strip() in {"semantic_hub", "external_manifest"}
    ):
        return ""
    coverage_summary = (
        dict(evaluation.get("coverage_summary", {}))
        if isinstance(evaluation.get("coverage_summary", {}), dict)
        else {}
    )
    family = str(evaluation.get("benchmark_family", "bounded")).strip() or "bounded"
    missing_counted_families = {
        str(value).strip()
        for value in coverage_summary.get("missing_required_counted_gated_families", [])
        if str(value).strip()
    }
    missing_clean_root_breadth = {
        str(value).strip()
        for value in coverage_summary.get("required_families_missing_clean_task_root_breadth", [])
        if str(value).strip()
    }
    if family not in missing_counted_families and family not in missing_clean_root_breadth:
        return ""
    reasons: list[str] = []
    if family in missing_counted_families:
        reasons.append("counted gated evidence")
    if family in missing_clean_root_breadth:
        reasons.append("clean task-root breadth")
    reasons_text = " and ".join(reasons)
    return (
        f"proof bootstrap override enabled for required family {family!r}; "
        f"allowing this clean-evidence candidate to accumulate missing {reasons_text}"
    )


def _report_trust_breadth_summary(
    evaluation: dict[str, Any],
    *,
    config: KernelConfig,
) -> dict[str, Any]:
    family = str(evaluation.get("benchmark_family", "bounded")).strip() or "bounded"
    policy = trust_policy_snapshot(config)
    family_summary = (
        dict(evaluation.get("family_summary", {}))
        if isinstance(evaluation.get("family_summary", {}), dict)
        else {}
    )
    coverage_summary = (
        dict(evaluation.get("coverage_summary", {}))
        if isinstance(evaluation.get("coverage_summary", {}), dict)
        else {}
    )
    clean_task_roots = [
        str(value).strip()
        for value in family_summary.get("clean_success_task_roots", [])
        if str(value).strip()
    ]
    min_distinct_task_roots = int(policy.get("family_breadth_min_distinct_task_roots", 0) or 0)
    distinct_clean_task_roots = int(family_summary.get("distinct_clean_success_task_roots", 0) or 0)
    required_family_clean_task_root_counts = (
        dict(coverage_summary.get("required_family_clean_task_root_counts", {}))
        if isinstance(coverage_summary.get("required_family_clean_task_root_counts", {}), dict)
        else {}
    )
    return {
        "benchmark_family": family,
        "required": bool(evaluation.get("required", False)),
        "status": str(evaluation.get("status", "")).strip(),
        "reports": int(family_summary.get("total", 0) or 0),
        "bootstrap_min_reports": int(policy.get("bootstrap_min_reports", 0) or 0),
        "breadth_min_reports": int(policy.get("breadth_min_reports", 0) or 0),
        "min_distinct_task_roots": min_distinct_task_roots,
        "distinct_clean_success_task_roots": distinct_clean_task_roots,
        "clean_success_task_roots": clean_task_roots,
        "remaining_distinct_task_roots": max(0, min_distinct_task_roots - distinct_clean_task_roots),
        "required_family_clean_task_root_count": int(required_family_clean_task_root_counts.get(family, 0) or 0),
        "light_supervision_reports": int(family_summary.get("light_supervision_candidate_count", 0) or 0),
        "light_supervision_successes": int(family_summary.get("light_supervision_success_count", 0) or 0),
        "light_supervision_clean_successes": int(
            family_summary.get("light_supervision_clean_success_count", 0) or 0
        ),
        "contract_clean_failure_recovery_reports": int(
            family_summary.get("contract_clean_failure_recovery_candidate_count", 0) or 0
        ),
        "contract_clean_failure_recovery_successes": int(
            family_summary.get("contract_clean_failure_recovery_success_count", 0) or 0
        ),
        "contract_clean_failure_recovery_clean_successes": int(
            family_summary.get("contract_clean_failure_recovery_clean_success_count", 0) or 0
        ),
    }


def _trust_breadth_detail(summary: dict[str, Any], *, family: str) -> str:
    if not summary or not bool(summary.get("required", False)):
        return ""
    min_distinct_task_roots = int(summary.get("min_distinct_task_roots", 0) or 0)
    if min_distinct_task_roots <= 0:
        return ""
    observed_roots = [
        str(value).strip()
        for value in summary.get("clean_success_task_roots", [])
        if str(value).strip()
    ]
    remaining = int(summary.get("remaining_distinct_task_roots", 0) or 0)
    observed_text = ",".join(observed_roots) if observed_roots else "<none>"
    return (
        f"{family} clean task-root breadth: observed={len(observed_roots)}/{min_distinct_task_roots} "
        f"remaining={remaining} roots=[{observed_text}]"
    )


def _resolve_repo_path(raw_path: str | None, *, repo_root: Path) -> Path | None:
    if not raw_path:
        return None
    path = Path(raw_path)
    if not path.is_absolute():
        path = repo_root / path
    return path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _workflow_guard(task: TaskSpec) -> dict[str, Any]:
    guard = task.metadata.get("workflow_guard", {})
    return dict(guard) if isinstance(guard, dict) else {}


def _unexpected_side_effects(
    *,
    task: TaskSpec | None,
    created: list[str],
    modified: list[str],
    deleted: list[str],
) -> tuple[list[str], list[str], list[str]]:
    managed_paths, delete_paths = _side_effect_contract_paths(task)
    allow_git_internal_changes = bool(task is not None and task.metadata.get("requires_git", False))
    unexpected_created = [
        path
        for path in created
        if path not in managed_paths and not _git_internal_path_allowed(path, allow_git_internal_changes)
    ]
    unexpected_modified = [
        path
        for path in modified
        if path not in managed_paths and not _git_internal_path_allowed(path, allow_git_internal_changes)
    ]
    unexpected_deleted = [
        path
        for path in deleted
        if path not in delete_paths and not _git_internal_path_allowed(path, allow_git_internal_changes)
    ]
    return unexpected_created, unexpected_modified, unexpected_deleted


def _side_effect_contract_paths(task: TaskSpec | None) -> tuple[set[str], set[str]]:
    if task is None:
        return set(), set()
    workflow_guard = _workflow_guard(task)
    managed_paths = _normalized_path_set(
        list(task.expected_files)
        + list(task.expected_file_contents)
        + list(task.forbidden_files)
        + [str(value) for value in workflow_guard.get("managed_paths", []) if str(value).strip()]
    )
    delete_paths = _normalized_path_set(
        list(task.forbidden_files)
        + [str(value) for value in workflow_guard.get("delete_paths", []) if str(value).strip()]
    )
    return managed_paths, delete_paths


def _normalized_path_set(paths: list[str]) -> set[str]:
    normalized: set[str] = set()
    for raw_path in paths:
        value = str(raw_path).strip().strip("/")
        if value:
            normalized.add(value)
    return normalized


def _git_internal_path_allowed(path: str, allow_git_internal_changes: bool) -> bool:
    if not allow_git_internal_changes:
        return False
    normalized = str(path).strip().strip("/")
    return normalized == ".git" or normalized.startswith(".git/")


def _changed_file_details(
    *,
    before_snapshot: dict[str, WorkspaceFileSnapshot],
    after_snapshot: dict[str, WorkspaceFileSnapshot],
    created: list[str],
    modified: list[str],
    deleted: list[str],
    managed_paths: set[str],
    delete_paths: set[str],
) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for path in created:
        after = after_snapshot.get(path)
        details.append(
            {
                "path": path,
                "change_type": "created",
                "before_sha256": "",
                "after_sha256": "" if after is None else after.sha256,
                "before_size_bytes": 0,
                "after_size_bytes": 0 if after is None else after.size_bytes,
                "accounted_for": path in managed_paths,
                "contract_status": "managed_create" if path in managed_paths else "unexpected_create",
            }
        )
    for path in modified:
        before = before_snapshot.get(path)
        after = after_snapshot.get(path)
        details.append(
            {
                "path": path,
                "change_type": "modified",
                "before_sha256": "" if before is None else before.sha256,
                "after_sha256": "" if after is None else after.sha256,
                "before_size_bytes": 0 if before is None else before.size_bytes,
                "after_size_bytes": 0 if after is None else after.size_bytes,
                "accounted_for": path in managed_paths,
                "contract_status": "managed_modify" if path in managed_paths else "unexpected_modify",
            }
        )
    for path in deleted:
        before = before_snapshot.get(path)
        details.append(
            {
                "path": path,
                "change_type": "deleted",
                "before_sha256": "" if before is None else before.sha256,
                "after_sha256": "",
                "before_size_bytes": 0 if before is None else before.size_bytes,
                "after_size_bytes": 0,
                "accounted_for": path in delete_paths,
                "contract_status": "allowed_delete" if path in delete_paths else "unexpected_delete",
            }
        )
    return details


def _serialize_command_step(step: StepRecord) -> dict[str, Any]:
    command_result = step.command_result or {}
    return {
        "index": step.index,
        "command": step.content,
        "exit_code": _step_exit_code(step),
        "timed_out": _step_timed_out(step),
        "blocked": _step_blocked(step),
        "verification_passed": _step_verified(step),
        "verification_reasons": list(step.verification.get("reasons", [])),
        "decision_source": str(step.decision_source).strip(),
        "tolbert_route_mode": str(step.tolbert_route_mode).strip(),
        "selected_retrieval_span_id": str(step.selected_retrieval_span_id or "").strip() or None,
        "retrieval_influenced": bool(step.retrieval_influenced),
        "retrieval_ranked_skill": bool(step.retrieval_ranked_skill),
        "trust_retrieval": bool(step.trust_retrieval),
        "path_confidence": float(step.path_confidence or 0.0),
        "capabilities_used": [
            str(value).strip()
            for value in command_result.get("capabilities_used", [])
            if str(value).strip()
        ],
    }


def _execution_source_summary(command_steps: list[StepRecord]) -> dict[str, int]:
    summary = {
        "decoder_generated": 0,
        "llm_generated": 0,
        "bounded_decoder_generated": 0,
        "synthetic_plan": 0,
        "deterministic_or_other": 0,
        "total_executed_commands": len(command_steps),
    }
    for step in command_steps:
        decision_source = str(step.decision_source).strip()
        if decision_source == "llm":
            summary["decoder_generated"] += 1
            summary["llm_generated"] += 1
        elif decision_source.endswith("_decoder"):
            summary["decoder_generated"] += 1
            summary["bounded_decoder_generated"] += 1
        elif decision_source == "synthetic_edit_plan_direct":
            summary["synthetic_plan"] += 1
        else:
            summary["deterministic_or_other"] += 1
    return summary


def _step_non_committal_failure(step: StepRecord) -> bool:
    exit_code = _step_exit_code(step)
    return _step_blocked(step) or _step_timed_out(step) or (exit_code is not None and exit_code != 0)


def _step_exit_code(step: StepRecord) -> int | None:
    command_result = step.command_result or {}
    value = command_result.get("exit_code")
    return int(value) if isinstance(value, int) else None


def _step_timed_out(step: StepRecord) -> bool:
    command_result = step.command_result or {}
    return bool(command_result.get("timed_out"))


def _step_blocked(step: StepRecord) -> bool:
    command_result = step.command_result or {}
    stderr = str(command_result.get("stderr", ""))
    return bool(command_result.get("exit_code") == 126 and "blocked " in stderr)


def _step_verified(step: StepRecord) -> bool:
    return bool(step.verification.get("passed"))
