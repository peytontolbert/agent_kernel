from __future__ import annotations

import re

from ...config import KernelConfig
from ...ops.episode_store import load_episode_document
from ...tasking.task_bank import _task_contract_from_memory
from .improvement_catalog import catalog_string_set
from .improvement_support_validation import TaskContractCatalog
from .state_estimation_improvement import STATE_ESTIMATION_PROPOSAL_AREAS
from .universe_improvement import UNIVERSE_PROPOSAL_AREAS

_WORLD_MODEL_PROPOSAL_AREAS = catalog_string_set("improvement", "world_model_proposal_areas")
_RECOVERY_PROPOSAL_AREAS = catalog_string_set("improvement", "recovery_proposal_areas")
_DELEGATION_PROPOSAL_AREAS = catalog_string_set("improvement", "delegation_proposal_areas")
_TRANSITION_MODEL_PROPOSAL_AREAS = catalog_string_set("transition_model", "proposal_areas")
_STATE_ESTIMATION_PROPOSAL_AREAS = set(STATE_ESTIMATION_PROPOSAL_AREAS)
_UNIVERSE_PROPOSAL_AREAS = set(UNIVERSE_PROPOSAL_AREAS)
_DERIVED_SOURCE_TASK_ID_SUFFIXES = (
    "_benchmark_candidate",
    "_verifier_candidate",
    "_episode_replay",
    "_verifier_replay",
    "_discovered",
    "_skill_replay",
    "_tool_replay",
    "_operator_replay",
    "_transition_pressure",
    "_repository_adjacent",
    "_repository_recovery",
    "_integration_recovery",
    "_tool_recovery",
    "_workflow_recovery",
    "_path_recovery",
)
_PARALLEL_WORKER_TASK_DELIMITER = "__worker__"


def is_stricter_contract(source_task, contract: dict[str, object]) -> bool:
    expected_files = set(str(path) for path in source_task.expected_files)
    source_forbidden = set(str(path) for path in source_task.forbidden_files)
    source_expected_contents = dict(source_task.expected_file_contents)
    source_forbidden_output = set(str(value) for value in source_task.forbidden_output_substrings)
    source_semantic_verifier = (
        dict(source_task.metadata.get("semantic_verifier", {}))
        if isinstance(source_task.metadata.get("semantic_verifier", {}), dict)
        else {}
    )

    contract_expected = set(str(path) for path in contract.get("expected_files", source_task.expected_files))
    contract_forbidden = set(str(path) for path in contract.get("forbidden_files", source_task.forbidden_files))
    contract_expected_contents = {
        str(path): str(content)
        for path, content in dict(contract.get("expected_file_contents", source_task.expected_file_contents)).items()
    }
    contract_forbidden_output = set(
        str(value) for value in contract.get("forbidden_output_substrings", source_task.forbidden_output_substrings)
    )
    contract_semantic_verifier = dict(contract.get("semantic_verifier", {})) if isinstance(contract.get("semantic_verifier", {}), dict) else {}

    if not expected_files.issubset(contract_expected):
        return False
    if not source_forbidden.issubset(contract_forbidden):
        return False
    if source_expected_contents.items() - contract_expected_contents.items():
        return False
    if not source_forbidden_output.issubset(contract_forbidden_output):
        return False
    if not _semantic_verifier_subsumes(source_semantic_verifier, contract_semantic_verifier):
        return False

    strictly_stronger = (
        contract_forbidden != source_forbidden
        or contract_expected_contents != source_expected_contents
        or contract_forbidden_output != source_forbidden_output
        or _semantic_verifier_is_stricter(source_semantic_verifier, contract_semantic_verifier)
    )
    return strictly_stronger


def _semantic_verifier_subsumes(source: dict[str, object], contract: dict[str, object]) -> bool:
    if not source:
        return True
    if not contract:
        return False
    for key in ("behavior_checks", "differential_checks", "repo_invariants"):
        source_values = source.get(key, [])
        if not isinstance(source_values, list):
            continue
        contract_values = contract.get(key, [])
        if not isinstance(contract_values, list):
            return False
        normalized_contract = [_normalized_semantic_value(item) for item in contract_values if isinstance(item, dict)]
        for item in source_values:
            if isinstance(item, dict) and _normalized_semantic_value(item) not in normalized_contract:
                return False
    return True


def _semantic_verifier_is_stricter(source: dict[str, object], contract: dict[str, object]) -> bool:
    for key in ("behavior_checks", "differential_checks", "repo_invariants"):
        source_values = source.get(key, [])
        contract_values = contract.get(key, [])
        if not isinstance(contract_values, list):
            continue
        source_count = len([item for item in source_values if isinstance(item, dict)]) if isinstance(source_values, list) else 0
        contract_count = len([item for item in contract_values if isinstance(item, dict)])
        if contract_count > source_count:
            return True
    return False


def _normalized_semantic_value(value: dict[str, object]) -> tuple[tuple[str, object], ...]:
    normalized: list[tuple[str, object]] = []
    for key, item in sorted(value.items()):
        if isinstance(item, dict):
            normalized.append((str(key), _normalized_semantic_value(item)))
        elif isinstance(item, list):
            normalized.append(
                (
                    str(key),
                    tuple(
                        _normalized_semantic_value(element) if isinstance(element, dict) else str(element)
                        for element in item
                    ),
                )
            )
        else:
            normalized.append((str(key), str(item)))
    return tuple(normalized)


def _safe_worker_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip().replace("/", "_"))
    return normalized.strip("_") or "worker"


def _resolve_parallel_worker_source_task(task_catalog: TaskContractCatalog, source_task_id: str):
    integrator_task_id, delimiter, worker_token = str(source_task_id).partition(_PARALLEL_WORKER_TASK_DELIMITER)
    if not delimiter or not integrator_task_id.strip() or not worker_token.strip():
        raise KeyError(source_task_id)
    normalized_worker_token = _safe_worker_name(worker_token)
    try:
        worker_tasks = task_catalog.bank.parallel_worker_tasks(integrator_task_id.strip())
    except KeyError as exc:
        raise KeyError(source_task_id) from exc
    for worker_task in worker_tasks:
        if worker_task.task_id == source_task_id:
            return worker_task
        metadata = dict(worker_task.metadata)
        workflow_guard = dict(metadata.get("workflow_guard", {})) if isinstance(metadata.get("workflow_guard", {}), dict) else {}
        worker_branch = str(workflow_guard.get("worker_branch", "")).strip()
        if worker_branch and _safe_worker_name(worker_branch) == normalized_worker_token:
            return worker_task
    raise KeyError(source_task_id)


def _resolve_episode_memory_source_task(task_catalog: TaskContractCatalog, source_task_id: str):
    config = KernelConfig()
    try:
        document = load_episode_document(config.trajectories_root, source_task_id, config=config)
    except FileNotFoundError as exc:
        raise KeyError(source_task_id) from exc
    contract = _task_contract_from_memory(document, task_id=source_task_id, bank=task_catalog.bank)
    if contract is None:
        raise KeyError(source_task_id)
    return contract


def significant_tokens(text: str) -> list[str]:
    tokens = []
    for token in str(text).lower().replace("/", " ").replace(".", " ").replace("'", " ").replace('"', " ").split():
        cleaned = token.strip()
        if len(cleaned) < 4:
            continue
        if cleaned in {"create", "containing", "string", "under", "same", "task", "with", "from"}:
            continue
        if cleaned not in tokens:
            tokens.append(cleaned)
    return tokens


def _resolve_source_task(task_catalog: TaskContractCatalog, source_task_id: str):
    pending = [str(source_task_id).strip()]
    seen: set[str] = set()
    while pending:
        candidate_task_id = str(pending.pop(0)).strip()
        if not candidate_task_id or candidate_task_id in seen:
            continue
        seen.add(candidate_task_id)
        try:
            return task_catalog.get(candidate_task_id)
        except KeyError:
            pass
        try:
            return _resolve_episode_memory_source_task(task_catalog, candidate_task_id)
        except KeyError:
            pass
        try:
            return _resolve_parallel_worker_source_task(task_catalog, candidate_task_id)
        except KeyError:
            pass
        for suffix in _DERIVED_SOURCE_TASK_ID_SUFFIXES:
            if candidate_task_id.endswith(suffix):
                stripped_task_id = candidate_task_id[: -len(suffix)].strip()
                if stripped_task_id:
                    pending.append(stripped_task_id)
    raise KeyError(source_task_id)


def semantic_compatibility_violations(
    *,
    subsystem: str,
    payload: dict[str, object],
    task_catalog: TaskContractCatalog,
) -> list[str]:
    from ... import improvement as core

    violations: list[str] = []
    if subsystem == "verifier":
        proposals = payload.get("proposals", [])
        if isinstance(proposals, list):
            for proposal in proposals:
                if not isinstance(proposal, dict):
                    continue
                source_task_id = str(proposal.get("source_task_id", "")).strip()
                contract = proposal.get("contract", {})
                if not source_task_id or not isinstance(contract, dict):
                    continue
                try:
                    source_task = _resolve_source_task(task_catalog, source_task_id)
                except KeyError:
                    violations.append(f"unknown source task for verifier proposal: {source_task_id}")
                    continue
                if not is_stricter_contract(source_task, contract):
                    violations.append(
                        f"verifier proposal for {source_task_id} does not strengthen the source contract"
                    )
        return violations

    if subsystem == "retrieval":
        proposals = payload.get("proposals", [])
        allowed = {
            "tolbert_branch_results",
            "tolbert_global_results",
            "tolbert_top_branches",
            "tolbert_ancestor_branch_levels",
            "tolbert_max_spans_per_source",
            "tolbert_context_max_chunks",
            "tolbert_confidence_threshold",
            "tolbert_branch_confidence_margin",
            "tolbert_low_confidence_widen_threshold",
            "tolbert_low_confidence_branch_multiplier",
            "tolbert_low_confidence_global_multiplier",
            "tolbert_deterministic_command_confidence",
            "tolbert_first_step_direct_command_confidence",
            "tolbert_direct_command_min_score",
            "tolbert_skill_ranking_min_confidence",
            "tolbert_distractor_penalty",
        }
        if isinstance(proposals, list):
            for proposal in proposals:
                if not isinstance(proposal, dict):
                    continue
                if not str(proposal.get("area", "")).strip():
                    violations.append("retrieval proposal must contain a non-empty area")
                if not str(proposal.get("reason", "")).strip():
                    violations.append("retrieval proposal must contain a non-empty reason")
                overrides = proposal.get("overrides", {})
                if not isinstance(overrides, dict) or not overrides:
                    violations.append("retrieval proposal must contain non-empty overrides")
                    continue
                for key, value in overrides.items():
                    if key not in allowed:
                        violations.append(f"retrieval proposal contains unsupported override: {key}")
                        continue
                    try:
                        numeric_value = float(value)
                    except (TypeError, ValueError):
                        violations.append(f"retrieval override {key} must be numeric")
                        continue
                    if "confidence" in key and not 0.0 <= numeric_value <= 1.0:
                        violations.append(f"retrieval override {key} must stay within [0.0, 1.0]")
                    if key in {
                        "tolbert_branch_results",
                        "tolbert_global_results",
                        "tolbert_top_branches",
                        "tolbert_ancestor_branch_levels",
                        "tolbert_max_spans_per_source",
                        "tolbert_context_max_chunks",
                        "tolbert_direct_command_min_score",
                    } and numeric_value < 0:
                        violations.append(f"retrieval override {key} must be non-negative")
        return violations

    if subsystem == "policy":
        proposals = payload.get("proposals", [])
        if isinstance(proposals, list):
            for proposal in proposals:
                if not isinstance(proposal, dict):
                    violations.append("every prompt proposal must be an object")
                    continue
                priority = proposal.get("priority", 0)
                if isinstance(priority, bool) or not isinstance(priority, int) or int(priority) <= 0:
                    violations.append("prompt proposal priority must be a positive integer")
                if not str(proposal.get("area", "")).strip():
                    violations.append("prompt proposal must contain a non-empty area")
                if not str(proposal.get("reason", "")).strip():
                    violations.append("prompt proposal must contain a non-empty reason")
                if not str(proposal.get("suggestion", "")).strip():
                    violations.append("prompt proposal must contain a non-empty suggestion")
        return violations

    if subsystem == "curriculum":
        proposals = payload.get("proposals", [])
        if isinstance(proposals, list):
            for proposal in proposals:
                if not isinstance(proposal, dict):
                    violations.append("every curriculum proposal must be an object")
                    continue
                priority = proposal.get("priority", 0)
                if isinstance(priority, bool) or not isinstance(priority, int) or int(priority) <= 0:
                    violations.append("curriculum proposal priority must be a positive integer")
                if not str(proposal.get("area", "")).strip():
                    violations.append("curriculum proposal must contain a non-empty area")
                if not str(proposal.get("reason", "")).strip():
                    violations.append("curriculum proposal must contain a non-empty reason")
                if not str(proposal.get("suggestion", "")).strip():
                    violations.append("curriculum proposal must contain a non-empty suggestion")
        return violations

    if subsystem == "world_model":
        proposals = payload.get("proposals", [])
        if isinstance(proposals, list):
            for proposal in proposals:
                if not isinstance(proposal, dict):
                    violations.append("every world_model proposal must be an object")
                    continue
                area = str(proposal.get("area", "")).strip()
                if area not in _WORLD_MODEL_PROPOSAL_AREAS:
                    violations.append("world_model proposal must contain a supported area")
                priority = proposal.get("priority", 0)
                if isinstance(priority, bool) or not isinstance(priority, int) or int(priority) <= 0:
                    violations.append("world_model proposal priority must be a positive integer")
                if not str(proposal.get("reason", "")).strip():
                    violations.append("world_model proposal must contain a non-empty reason")
                if not str(proposal.get("suggestion", "")).strip():
                    violations.append("world_model proposal must contain a non-empty suggestion")
        return violations

    if subsystem == "state_estimation":
        proposals = payload.get("proposals", [])
        if isinstance(proposals, list):
            for proposal in proposals:
                if not isinstance(proposal, dict):
                    violations.append("every state_estimation proposal must be an object")
                    continue
                area = str(proposal.get("area", "")).strip()
                if area not in _STATE_ESTIMATION_PROPOSAL_AREAS:
                    violations.append("state_estimation proposal must contain a supported area")
                priority = proposal.get("priority", 0)
                if isinstance(priority, bool) or not isinstance(priority, int) or int(priority) <= 0:
                    violations.append("state_estimation proposal priority must be a positive integer")
                if not str(proposal.get("reason", "")).strip():
                    violations.append("state_estimation proposal must contain a non-empty reason")
                if not str(proposal.get("suggestion", "")).strip():
                    violations.append("state_estimation proposal must contain a non-empty suggestion")
        return violations

    if subsystem == "universe":
        proposals = payload.get("proposals", [])
        if isinstance(proposals, list):
            for proposal in proposals:
                if not isinstance(proposal, dict):
                    violations.append("every universe proposal must be an object")
                    continue
                area = str(proposal.get("area", "")).strip()
                if area not in _UNIVERSE_PROPOSAL_AREAS:
                    violations.append("universe proposal must contain a supported area")
                priority = proposal.get("priority", 0)
                if isinstance(priority, bool) or not isinstance(priority, int) or int(priority) <= 0:
                    violations.append("universe proposal priority must be a positive integer")
                if not str(proposal.get("reason", "")).strip():
                    violations.append("universe proposal must contain a non-empty reason")
                if not str(proposal.get("suggestion", "")).strip():
                    violations.append("universe proposal must contain a non-empty suggestion")
        return violations

    if subsystem == "trust":
        proposals = payload.get("proposals", [])
        allowed_areas = {"safety", "breadth", "stability"}
        if isinstance(proposals, list):
            for proposal in proposals:
                if not isinstance(proposal, dict):
                    violations.append("every trust proposal must be an object")
                    continue
                area = str(proposal.get("area", "")).strip()
                if area not in allowed_areas:
                    violations.append("trust proposal must contain a supported area")
                priority = proposal.get("priority", 0)
                if isinstance(priority, bool) or not isinstance(priority, int) or int(priority) <= 0:
                    violations.append("trust proposal priority must be a positive integer")
                if not str(proposal.get("reason", "")).strip():
                    violations.append("trust proposal must contain a non-empty reason")
                if not str(proposal.get("suggestion", "")).strip():
                    violations.append("trust proposal must contain a non-empty suggestion")
        return violations

    if subsystem == "recovery":
        proposals = payload.get("proposals", [])
        if isinstance(proposals, list):
            for proposal in proposals:
                if not isinstance(proposal, dict):
                    violations.append("every recovery proposal must be an object")
                    continue
                area = str(proposal.get("area", "")).strip()
                if area not in _RECOVERY_PROPOSAL_AREAS:
                    violations.append("recovery proposal must contain a supported area")
                priority = proposal.get("priority", 0)
                if isinstance(priority, bool) or not isinstance(priority, int) or int(priority) <= 0:
                    violations.append("recovery proposal priority must be a positive integer")
                if not str(proposal.get("reason", "")).strip():
                    violations.append("recovery proposal must contain a non-empty reason")
                if not str(proposal.get("suggestion", "")).strip():
                    violations.append("recovery proposal must contain a non-empty suggestion")
        return violations

    if subsystem == "delegation":
        proposals = payload.get("proposals", [])
        if isinstance(proposals, list):
            for proposal in proposals:
                if not isinstance(proposal, dict):
                    violations.append("every delegation proposal must be an object")
                    continue
                area = str(proposal.get("area", "")).strip()
                if area not in _DELEGATION_PROPOSAL_AREAS:
                    violations.append("delegation proposal must contain a supported area")
                priority = proposal.get("priority", 0)
                if isinstance(priority, bool) or not isinstance(priority, int) or int(priority) <= 0:
                    violations.append("delegation proposal priority must be a positive integer")
                if not str(proposal.get("reason", "")).strip():
                    violations.append("delegation proposal must contain a non-empty reason")
                if not str(proposal.get("suggestion", "")).strip():
                    violations.append("delegation proposal must contain a non-empty suggestion")
        return violations

    if subsystem == "transition_model":
        proposals = payload.get("proposals", [])
        if isinstance(proposals, list):
            for proposal in proposals:
                if not isinstance(proposal, dict):
                    violations.append("every transition_model proposal must be an object")
                    continue
                area = str(proposal.get("area", "")).strip()
                if area not in _TRANSITION_MODEL_PROPOSAL_AREAS:
                    violations.append("transition_model proposal must contain a supported area")
                priority = proposal.get("priority", 0)
                if isinstance(priority, bool) or not isinstance(priority, int) or int(priority) <= 0:
                    violations.append("transition_model proposal priority must be a positive integer")
                if not str(proposal.get("reason", "")).strip():
                    violations.append("transition_model proposal must contain a non-empty reason")
                if not str(proposal.get("suggestion", "")).strip():
                    violations.append("transition_model proposal must contain a non-empty suggestion")
        return violations

    if subsystem in {"skills", "tooling"}:
        collection_key = "skills" if subsystem == "skills" else "candidates"
        records = payload.get(collection_key, [])
        if isinstance(records, list):
            for record in records:
                if not isinstance(record, dict):
                    continue
                source_task_id = str(record.get("source_task_id", record.get("task_id", ""))).strip()
                if not source_task_id:
                    continue
                try:
                    source_task = _resolve_source_task(task_catalog, source_task_id)
                except KeyError:
                    violations.append(f"unknown source task for {subsystem} artifact: {source_task_id}")
                    continue
                task_contract = record.get("task_contract", {})
                if isinstance(task_contract, dict) and task_contract:
                    expected_files = set(str(path) for path in source_task.expected_files)
                    contract_expected_files = set(
                        str(path) for path in task_contract.get("expected_files", source_task.expected_files)
                    )
                    if expected_files and contract_expected_files != expected_files:
                        violations.append(
                            f"{subsystem} artifact for {source_task_id} changes expected_files away from the source contract"
                        )
                procedure = record.get("procedure", {})
                commands = procedure.get("commands", []) if isinstance(procedure, dict) else []
                if not isinstance(commands, list) or not commands:
                    violations.append(f"{subsystem} artifact for {source_task_id} has no procedure commands")
                    continue
                normalized_commands = " ".join(str(command) for command in commands)
                expected_file_match = any(path and path in normalized_commands for path in source_task.expected_files)
                success_command_match = source_task.success_command and any(
                    token in normalized_commands
                    for token in significant_tokens(source_task.success_command)
                )
                if source_task.expected_files and not expected_file_match and not success_command_match:
                    violations.append(
                        f"{subsystem} artifact for {source_task_id} is not obviously aligned with source artifacts or verifier intent"
                    )
        return violations

    if subsystem == "operators":
        records = payload.get("operators", [])
        if isinstance(records, list):
            for record in records:
                if not isinstance(record, dict):
                    continue
                source_task_ids = [str(value) for value in record.get("source_task_ids", []) if str(value).strip()]
                if len(source_task_ids) < 2:
                    violations.append("operator artifact must aggregate at least two source tasks")
                    continue
                commands = core._operator_steps(record)
                if not isinstance(commands, list) or not commands:
                    violations.append("operator artifact must contain a non-empty step sequence")
                template_contract = core._operator_task_contract(record)
                if not isinstance(template_contract, dict) or not template_contract:
                    violations.append("operator artifact must contain a task contract")
                for source_task_id in source_task_ids:
                    try:
                        source_task = _resolve_source_task(task_catalog, source_task_id)
                    except KeyError:
                        violations.append(f"unknown source task for operators artifact: {source_task_id}")
                        continue
                    if isinstance(template_contract, dict):
                        contract_expected_files = set(
                            str(path) for path in template_contract.get("expected_files", source_task.expected_files)
                        )
                        if source_task.expected_files and not contract_expected_files:
                            violations.append(
                                f"operator artifact for {source_task_id} is missing expected file anchors"
                            )
        return violations

    if subsystem == "benchmark":
        proposals = payload.get("proposals", [])
        if isinstance(proposals, list):
            for proposal in proposals:
                if not isinstance(proposal, dict):
                    continue
                source_task_id = str(proposal.get("source_task_id", "")).strip()
                prompt = str(proposal.get("prompt", "")).strip().lower()
                if not source_task_id:
                    continue
                try:
                    source_task = _resolve_source_task(task_catalog, source_task_id)
                except KeyError:
                    violations.append(f"unknown source task for benchmark proposal: {source_task_id}")
                    continue
                source_terms = significant_tokens(source_task.prompt)
                if source_terms and not any(term in prompt for term in source_terms[:5]):
                    violations.append(
                        f"benchmark proposal for {source_task_id} is not semantically anchored to the source task prompt"
                    )
        return violations

    return violations


__all__ = [
    "is_stricter_contract",
    "semantic_compatibility_violations",
    "significant_tokens",
]
