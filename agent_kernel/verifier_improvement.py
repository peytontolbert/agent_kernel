from __future__ import annotations

import json
from pathlib import Path

from .config import KernelConfig, current_external_task_manifests_paths
from .improvement_common import build_standard_proposal_artifact, retention_gate_preset
from .memory import EpisodeMemory
from .schemas import TaskSpec
from .task_bank import TaskBank, _task_contract_from_memory
from .verifier import synthesize_stricter_task


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
        proposals.append(
            {
                "proposal_id": f"verifier:{task_id}:strict",
                "source_task_id": task_id,
                "benchmark_family": str(contract.metadata.get("benchmark_family", "bounded")),
                "contract": {
                    "expected_files": strict.expected_files,
                    "forbidden_files": strict.forbidden_files,
                    "expected_file_contents": strict.expected_file_contents,
                    "forbidden_output_substrings": strict.forbidden_output_substrings,
                },
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
