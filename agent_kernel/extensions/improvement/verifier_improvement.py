from __future__ import annotations

import json
from pathlib import Path

from ...config import KernelConfig, current_external_task_manifests_paths
from .improvement_common import build_standard_proposal_artifact, retention_gate_preset
from ...memory import EpisodeMemory
from ...schemas import TaskSpec
from ...tasking.task_bank import TaskBank, _task_contract_from_memory
from ...verifier import synthesize_stricter_task


def synthesize_verifier_contracts(
    memory_root: Path,
    output_path: Path,
    *,
    limit: int = 20,
    strategy: str = "balanced",
) -> Path:
    memory = EpisodeMemory(memory_root)
    manifest_paths = current_external_task_manifests_paths()
    try:
        bank = TaskBank(
            config=KernelConfig(),
            external_task_manifests=manifest_paths if manifest_paths else None,
        )
    except TypeError:
        bank = TaskBank()
    proposals: list[dict[str, object]] = []
    grouped = _group_documents_by_task(memory.list_documents())
    for task_id, documents in grouped.items():
        contract = _best_contract_for_task(documents, task_id=task_id, bank=bank)
        if contract is None:
            continue
        strict = synthesize_stricter_task(contract, task_id=f"{task_id}_strict_contract")
        successful_documents = [document for document in documents if document.get("success")]
        failed_documents = [document for document in documents if not document.get("success")]
        failure_types = sorted(
            {
                str(value)
                for document in failed_documents
                for value in document.get("summary", {}).get("failure_types", [])
                if str(value).strip()
            }
        )
        transition_failures = sorted(
            {
                str(value)
                for document in failed_documents
                for value in document.get("summary", {}).get("transition_failures", [])
                if str(value).strip()
            }
        )
        evidence = {
            "successful_trace_count": len(successful_documents),
            "failed_trace_count": len(failed_documents),
            "failure_types": failure_types,
            "transition_failures": transition_failures,
            "strict_contract": _strictness_gain(contract, strict) > 0.0,
            "proposal_discrimination_estimate": _discrimination_gain_estimate(contract, strict, failed_documents),
            "false_failure_risk": _false_failure_risk(contract, strict),
            "requires_external_eval": True,
        }
        semantic_verifier = _strict_semantic_verifier(contract, strict, documents)
        proposal_contract = {
            "expected_files": strict.expected_files,
            "forbidden_files": strict.forbidden_files,
            "expected_file_contents": strict.expected_file_contents,
            "forbidden_output_substrings": strict.forbidden_output_substrings,
        }
        if semantic_verifier:
            proposal_contract["semantic_verifier"] = semantic_verifier
        proposals.append(
            {
                "proposal_id": f"verifier:{task_id}:strict",
                "source_task_id": task_id,
                "benchmark_family": str(contract.metadata.get("benchmark_family", "bounded")),
                "contract": proposal_contract,
                "evidence": evidence,
            }
        )
    proposals = _order_verifier_proposals(proposals, strategy=strategy)[:limit]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_standard_proposal_artifact(
        artifact_kind="verifier_candidate_set",
        generation_focus=strategy,
        retention_gate=retention_gate_preset("verifier"),
        proposals=proposals,
        extra_sections={"generation_strategy": strategy},
    )
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def _group_documents_by_task(documents: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for document in documents:
        task_id = str(document.get("task_id", "")).strip()
        if not task_id:
            continue
        grouped.setdefault(task_id, []).append(document)
    return grouped


def _best_contract_for_task(
    documents: list[dict[str, object]],
    *,
    task_id: str,
    bank: TaskBank,
) -> TaskSpec | None:
    successful_documents = [document for document in documents if document.get("success")]
    ordered_documents = successful_documents or documents
    for document in ordered_documents:
        contract = _task_contract_from_memory(document, task_id=task_id, bank=bank)
        if contract is not None:
            return contract
    return None


def _strictness_gain(source: TaskSpec, strict: TaskSpec) -> float:
    gain = 0.0
    if len(strict.forbidden_files) > len(source.forbidden_files):
        gain += 0.02 * (len(strict.forbidden_files) - len(source.forbidden_files))
    if len(strict.forbidden_output_substrings) > len(source.forbidden_output_substrings):
        gain += 0.02 * (
            len(strict.forbidden_output_substrings) - len(source.forbidden_output_substrings)
        )
    added_expected_content = len(set(strict.expected_file_contents.items()) - set(source.expected_file_contents.items()))
    if added_expected_content > 0:
        gain += 0.02 * added_expected_content
    source_semantic = _semantic_verifier_from_task(source)
    strict_semantic = _strict_semantic_verifier(source, strict, [])
    if strict_semantic:
        gain += _semantic_verifier_strictness_gain(source_semantic, strict_semantic)
    return gain


def _discrimination_gain_estimate(
    source: TaskSpec,
    strict: TaskSpec,
    failed_documents: list[dict[str, object]],
) -> float:
    failure_types = {
        str(value)
        for document in failed_documents
        for value in document.get("summary", {}).get("failure_types", [])
        if str(value).strip()
    }
    transition_failures = {
        str(value)
        for document in failed_documents
        for value in document.get("summary", {}).get("transition_failures", [])
        if str(value).strip()
    }
    return _strictness_gain(source, strict) + min(0.01 * len(failure_types), 0.05) + min(
        0.01 * len(transition_failures),
        0.03,
    )


def _false_failure_risk(source: TaskSpec, strict: TaskSpec) -> float:
    if not set(source.expected_files).issubset(strict.expected_files):
        return 1.0
    if set(source.forbidden_files) - set(strict.forbidden_files):
        return 1.0
    if source.expected_file_contents.items() - strict.expected_file_contents.items():
        return 1.0
    if set(source.forbidden_output_substrings) - set(strict.forbidden_output_substrings):
        return 1.0
    return 0.0


def _semantic_verifier_from_task(task: TaskSpec) -> dict[str, object]:
    verifier = task.metadata.get("semantic_verifier", {})
    return dict(verifier) if isinstance(verifier, dict) else {}


def _strict_semantic_verifier(
    source: TaskSpec,
    strict: TaskSpec,
    documents: list[dict[str, object]],
) -> dict[str, object]:
    verifier = _semantic_verifier_from_task(strict) or _semantic_verifier_from_task(source)
    if not verifier:
        return {}
    enriched = dict(verifier)
    behavior_checks = _normalized_behavior_checks(enriched.get("behavior_checks", []))
    for check in _behavior_checks_from_legacy_verifier(enriched):
        if check not in behavior_checks:
            behavior_checks.append(check)
    if behavior_checks:
        enriched["behavior_checks"] = behavior_checks
    differential_checks = _normalized_differential_checks(enriched.get("differential_checks", []))
    synthesized_differential_checks = _differential_checks_from_episode_evidence(
        source,
        documents=documents,
    )
    for check in synthesized_differential_checks:
        if check not in differential_checks:
            differential_checks.append(check)
    if differential_checks:
        enriched["differential_checks"] = differential_checks
    repo_invariants = _normalized_repo_invariants(enriched.get("repo_invariants", []))
    synthesized_invariants = _repo_invariants_from_legacy_verifier(enriched, documents=documents)
    for invariant in synthesized_invariants:
        if invariant not in repo_invariants:
            repo_invariants.append(invariant)
    if repo_invariants:
        enriched["repo_invariants"] = repo_invariants
    return enriched


def _semantic_verifier_strictness_gain(source: dict[str, object], strict: dict[str, object]) -> float:
    gain = 0.0
    source_behavior = _normalized_behavior_checks(source.get("behavior_checks", []))
    strict_behavior = _normalized_behavior_checks(strict.get("behavior_checks", []))
    if len(strict_behavior) > len(source_behavior):
        gain += 0.02 * (len(strict_behavior) - len(source_behavior))
    source_differential = _normalized_differential_checks(source.get("differential_checks", []))
    strict_differential = _normalized_differential_checks(strict.get("differential_checks", []))
    if len(strict_differential) > len(source_differential):
        gain += 0.02 * (len(strict_differential) - len(source_differential))
    source_invariants = _normalized_repo_invariants(source.get("repo_invariants", []))
    strict_invariants = _normalized_repo_invariants(strict.get("repo_invariants", []))
    if len(strict_invariants) > len(source_invariants):
        gain += 0.02 * (len(strict_invariants) - len(source_invariants))
    return gain


def _normalized_behavior_checks(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _normalized_differential_checks(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _normalized_repo_invariants(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _behavior_checks_from_legacy_verifier(verifier: dict[str, object]) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    for command in verifier.get("test_commands", []):
        if isinstance(command, dict):
            argv = [str(value).strip() for value in command.get("argv", []) if str(value).strip()]
            if not argv:
                continue
            checks.append(
                {
                    "label": str(command.get("label", "")).strip() or f"legacy test command: {' '.join(argv)}",
                    "argv": argv,
                    "expect_exit_code": 0,
                }
            )
            continue
        normalized = str(command).strip()
        if not normalized:
            continue
        checks.append(
            {
                "label": f"legacy test command: {normalized}",
                "argv": ["/bin/sh", "-lc", normalized],
                "expect_exit_code": 0,
            }
        )
    return checks


def _repo_invariants_from_legacy_verifier(
    verifier: dict[str, object],
    *,
    documents: list[dict[str, object]],
) -> list[dict[str, object]]:
    invariants: list[dict[str, object]] = []
    preserved_paths = [
        str(path).strip()
        for path in verifier.get("preserved_paths", [])
        if str(path).strip()
    ]
    for path in preserved_paths:
        invariants.append({"kind": "git_tracked_paths", "paths": [path]})
    clean_worktree = verifier.get("clean_worktree")
    if clean_worktree:
        allow_paths = {
            str(path).strip()
            for key in ("expected_changed_paths", "generated_paths", "resolved_conflict_paths")
            for path in verifier.get(key, [])
            if str(path).strip()
        }
        invariants.append({"kind": "git_clean", "allow_paths": sorted(allow_paths)})
    for document in documents:
        summary = document.get("summary", {})
        if not isinstance(summary, dict):
            continue
        for changed_path in summary.get("changed_paths", []):
            normalized = str(changed_path).strip()
            if normalized and normalized in preserved_paths:
                candidate = {"kind": "git_tracked_paths", "paths": [normalized]}
                if candidate not in invariants:
                    invariants.append(candidate)
    return invariants


def _differential_checks_from_episode_evidence(
    source: TaskSpec,
    *,
    documents: list[dict[str, object]],
) -> list[dict[str, object]]:
    verifier = _semantic_verifier_from_task(source)
    successful_records: list[dict[str, object]] = []
    failed_records: list[dict[str, object]] = []
    for document in documents:
        for record in _document_command_records(document):
            if bool(record.get("passed", False)):
                successful_records.append(record)
            else:
                failed_records.append(record)
    synthesized: list[dict[str, object]] = []
    seen_pairs: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
    candidate_pool = successful_records
    if not candidate_pool:
        candidate_pool = _legacy_test_command_records(verifier)
    for candidate in candidate_pool:
        candidate_argv = tuple(str(value).strip() for value in candidate.get("argv", []) if str(value).strip())
        if not candidate_argv:
            continue
        candidate_pattern = str(candidate.get("command_pattern", "")).strip()
        candidate_is_test = bool(candidate.get("test_command", False))
        matched_baseline = None
        for baseline in failed_records:
            baseline_argv = tuple(str(value).strip() for value in baseline.get("argv", []) if str(value).strip())
            if not baseline_argv:
                continue
            baseline_pattern = str(baseline.get("command_pattern", "")).strip()
            baseline_is_test = bool(baseline.get("test_command", False))
            if candidate_pattern and baseline_pattern and candidate_pattern != baseline_pattern and not (
                candidate_is_test and baseline_is_test
            ):
                continue
            pair_key = (candidate_argv, baseline_argv)
            if pair_key in seen_pairs:
                continue
            matched_baseline = baseline
            seen_pairs.add(pair_key)
            break
        if matched_baseline is None:
            continue
        candidate_exit_code = matched_baseline.get("candidate_exit_code")
        if candidate_exit_code is None:
            candidate_exit_code = candidate.get("exit_code")
        baseline_exit_code = matched_baseline.get("exit_code")
        rule: dict[str, object] = {
            "label": str(candidate.get("label", "")).strip()
            or str(matched_baseline.get("label", "")).strip()
            or f"differential check: {' '.join(candidate_argv)}",
            "candidate_argv": list(candidate_argv),
            "baseline_argv": list(
                str(value).strip() for value in matched_baseline.get("argv", []) if str(value).strip()
            ),
            "expect_same_exit_code": False,
            "expect_candidate_exit_code": 0 if candidate_exit_code is None else int(candidate_exit_code),
            "normalize_whitespace": True,
        }
        if baseline_exit_code is not None:
            rule["expect_baseline_exit_code"] = int(baseline_exit_code)
        candidate_stdout = str(candidate.get("stdout", "")).strip()
        baseline_stdout = str(matched_baseline.get("stdout", "")).strip()
        if candidate_stdout and baseline_stdout and candidate_stdout != baseline_stdout:
            rule["expect_stdout_difference"] = True
            rule["candidate_stdout_must_contain"] = [candidate_stdout]
            rule["baseline_stdout_must_contain"] = [baseline_stdout]
        candidate_stderr = str(candidate.get("stderr", "")).strip()
        baseline_stderr = str(matched_baseline.get("stderr", "")).strip()
        if candidate_stderr and baseline_stderr and candidate_stderr != baseline_stderr:
            rule["expect_stderr_difference"] = True
            rule["candidate_stderr_must_contain"] = [candidate_stderr]
            rule["baseline_stderr_must_contain"] = [baseline_stderr]
        candidate_file_expectations = _candidate_file_expectations_from_record(candidate)
        if candidate_file_expectations:
            rule["candidate_file_expectations"] = candidate_file_expectations
        touched_paths = sorted(
            {
                str(path).strip()
                for path in [*candidate.get("changed_paths", []), *matched_baseline.get("changed_paths", [])]
                if str(path).strip()
            }
        )
        if touched_paths:
            rule["touched_paths"] = touched_paths[:4]
        synthesized.append(rule)
        if len(synthesized) >= 2:
            break
    return synthesized


def _document_command_records(document: dict[str, object]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    task_contract = document.get("task_contract", {})
    task_contract = dict(task_contract) if isinstance(task_contract, dict) else {}
    summary = document.get("summary", {})
    summary = dict(summary) if isinstance(summary, dict) else {}
    for step in document.get("steps", []) if isinstance(document.get("steps", []), list) else []:
        if not isinstance(step, dict):
            continue
        if str(step.get("action", "")).strip() != "code_execute":
            continue
        command = str(step.get("content", "")).strip()
        if not command:
            continue
        command_result = step.get("command_result", {})
        command_result = dict(command_result) if isinstance(command_result, dict) else {}
        verification = step.get("verification", {})
        verification = dict(verification) if isinstance(verification, dict) else {}
        argv = _shell_argv_for_command(command)
        transition = step.get("state_transition", {})
        transition = dict(transition) if isinstance(transition, dict) else {}
        changed_paths = [
            str(path).strip()
            for path in transition.get("changed_paths", summary.get("changed_paths", []))
            if str(path).strip()
        ]
        edit_patches = [dict(patch) for patch in transition.get("edit_patches", []) if isinstance(patch, dict)]
        records.append(
            {
                "label": command,
                "argv": argv,
                "command_pattern": _command_pattern(command),
                "test_command": _is_test_like_command(command),
                "passed": bool(verification.get("passed", False)),
                "exit_code": command_result.get("exit_code"),
                "stdout": str(command_result.get("stdout", "")),
                "stderr": str(command_result.get("stderr", "")),
                "task_contract": dict(task_contract),
                "changed_paths": changed_paths,
                "edit_patches": edit_patches,
            }
        )
    return records


def _candidate_file_expectations_from_record(record: dict[str, object]) -> list[dict[str, object]]:
    task_contract = record.get("task_contract", {})
    task_contract = dict(task_contract) if isinstance(task_contract, dict) else {}
    metadata = task_contract.get("metadata", {})
    metadata = dict(metadata) if isinstance(metadata, dict) else {}
    verifier = metadata.get("semantic_verifier", {})
    verifier = dict(verifier) if isinstance(verifier, dict) else {}
    expected_file_contents = dict(task_contract.get("expected_file_contents", {}))
    expected_files = [str(path).strip() for path in task_contract.get("expected_files", []) if str(path).strip()]
    changed_paths = [str(path).strip() for path in record.get("changed_paths", []) if str(path).strip()]
    edit_patch_paths = [
        str(dict(patch).get("path", "")).strip()
        for patch in record.get("edit_patches", [])
        if isinstance(patch, dict) and str(dict(patch).get("path", "")).strip()
    ]
    candidate_paths: list[str] = []
    for path in [*changed_paths, *edit_patch_paths, *expected_files]:
        normalized = str(path).strip()
        if normalized and normalized not in candidate_paths:
            candidate_paths.append(normalized)
    expectations: list[dict[str, object]] = []
    for path in candidate_paths[:4]:
        expectation: dict[str, object] = {"path": path, "must_exist": True}
        if path in expected_file_contents:
            expectation["expected_content"] = str(expected_file_contents[path])
        report_rule = next(
            (
                dict(rule)
                for rule in verifier.get("report_rules", [])
                if isinstance(rule, dict) and str(rule.get("path", "")).strip() == path
            ),
            None,
        )
        if report_rule is not None:
            must_contain = [str(value).strip() for value in report_rule.get("must_mention", []) if str(value).strip()]
            if must_contain:
                expectation["must_contain"] = must_contain[:4]
        if expectation not in expectations:
            expectations.append(expectation)
    return expectations


def _legacy_test_command_records(verifier: dict[str, object]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for command in verifier.get("test_commands", []):
        if not isinstance(command, dict):
            continue
        argv = [str(value).strip() for value in command.get("argv", []) if str(value).strip()]
        if not argv:
            continue
        records.append(
            {
                "label": str(command.get("label", "")).strip() or "legacy test command",
                "argv": argv,
                "command_pattern": _command_pattern(" ".join(argv)),
                "test_command": True,
                "passed": True,
                "exit_code": 0,
                "stdout": "",
                "stderr": "",
            }
        )
    return records


def _shell_argv_for_command(command: str) -> list[str]:
    normalized = str(command).strip()
    if not normalized:
        return []
    return ["/bin/sh", "-lc", normalized]


def _command_pattern(command: str) -> str:
    normalized = str(command).strip()
    if not normalized:
        return ""
    if normalized.startswith("pytest"):
        return "pytest"
    if normalized.startswith("python -m pytest"):
        return "python -m pytest"
    if normalized.startswith("git "):
        parts = normalized.split()
        if len(parts) >= 2:
            return f"git {parts[1]}"
        return "git"
    match = normalized.split(None, 1)
    return match[0] if match else normalized


def _is_test_like_command(command: str) -> bool:
    normalized = str(command).strip()
    return bool(
        normalized.startswith("pytest")
        or normalized.startswith("python -m pytest")
        or normalized.startswith("test ")
        or "/test" in normalized
        or " verify" in normalized
    )


def _order_verifier_proposals(
    proposals: list[dict[str, object]],
    *,
    strategy: str,
) -> list[dict[str, object]]:
    if strategy == "false_failure_guard":
        return sorted(
            proposals,
            key=lambda proposal: (
                float(proposal.get("evidence", {}).get("false_failure_risk", 1.0)),
                -int(proposal.get("evidence", {}).get("failed_trace_count", 0)),
                -float(proposal.get("evidence", {}).get("discrimination_gain_estimate", 0.0)),
                str(proposal.get("proposal_id", "")),
            ),
        )
    if strategy == "strict_contract_growth":
        return sorted(
            proposals,
            key=lambda proposal: (
                -float(proposal.get("evidence", {}).get("discrimination_gain_estimate", 0.0)),
                -int(proposal.get("evidence", {}).get("successful_trace_count", 0)),
                float(proposal.get("evidence", {}).get("false_failure_risk", 1.0)),
                str(proposal.get("proposal_id", "")),
            ),
        )
    return sorted(proposals, key=lambda proposal: str(proposal.get("proposal_id", "")))
