from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import combinations
import re
import shlex
from typing import TYPE_CHECKING

from ...syntax_motor import summarize_python_edit_step
from ..world.rollout import rollout_action_value

if TYPE_CHECKING:
    from agent_kernel.policy import SkillLibrary
    from agent_kernel.state import AgentState
    from agent_kernel.world_model import WorldModel


@dataclass(slots=True)
class DecodedActionCandidate:
    action: str
    content: str
    thought: str
    score: float
    reason: str
    selected_skill_id: str | None = None
    selected_retrieval_span_id: str | None = None
    retrieval_influenced: bool = False
    retrieval_ranked_skill: bool = False
    proposal_source: str = ""
    proposal_novel: bool = False
    proposal_metadata: dict[str, object] | None = None


def decode_bounded_action_candidates(
    *,
    state: "AgentState",
    world_model: "WorldModel",
    skill_library: "SkillLibrary",
    top_skill: dict[str, object] | None,
    retrieval_guidance: dict[str, list[str]],
    blocked_commands: list[str],
    decoder_policy: dict[str, object],
    rollout_policy: dict[str, object],
    command_score_fn,
    match_span_fn,
    normalize_command_fn,
    canonicalize_command_fn,
) -> list[DecodedActionCandidate]:
    blocked = {canonicalize_command_fn(command) for command in blocked_commands}
    seen: set[str] = set()
    candidates: list[DecodedActionCandidate] = []
    if _allow_stop_candidate(state, decoder_policy):
        stop_score = rollout_action_value(
            world_model_summary=state.world_model_summary,
            latent_state_summary=state.latent_state_summary,
            latest_transition=state.latest_state_transition,
            action="respond",
            content="Task appears complete under retained Tolbert primary control.",
            rollout_policy=rollout_policy,
            world_model=world_model,
        )
        candidates.append(
            DecodedActionCandidate(
                action="respond",
                content="Task appears complete under retained Tolbert primary control.",
                thought="Stop because the retained Tolbert decoder predicts the task contract is already satisfied.",
                score=stop_score,
                reason="stop_decision",
            )
        )

    if bool(decoder_policy.get("allow_retrieval_guidance", True)):
        for item in retrieval_guidance.get("recommended_command_spans", []):
            command = normalize_command_fn(str(item.get("command", "")), state.task.workspace_subdir)
            normalized = canonicalize_command_fn(command)
            if not normalized or normalized in seen or normalized in blocked:
                continue
            seen.add(normalized)
            score = _command_candidate_score(
                state=state,
                world_model=world_model,
                command=command,
                rollout_policy=rollout_policy,
                command_score_fn=command_score_fn,
            )
            candidates.append(
                DecodedActionCandidate(
                    action="code_execute",
                    content=command,
                    thought="Execute the highest-value retained Tolbert retrieval-guided command.",
                    score=score,
                    reason="retrieval_guidance",
                    selected_retrieval_span_id=str(item.get("span_id", "")).strip() or None,
                    retrieval_influenced=True,
                )
            )

    if bool(decoder_policy.get("allow_skill_commands", True)) and top_skill is not None:
        for command in skill_library._commands_for_skill(top_skill)[:1]:
            command = normalize_command_fn(command, state.task.workspace_subdir)
            normalized = canonicalize_command_fn(command)
            if not normalized or normalized in seen or normalized in blocked:
                continue
            seen.add(normalized)
            score = _command_candidate_score(
                state=state,
                world_model=world_model,
                command=command,
                rollout_policy=rollout_policy,
                command_score_fn=command_score_fn,
            )
            candidates.append(
                DecodedActionCandidate(
                    action="code_execute",
                    content=command,
                    thought="Execute the retained Tolbert skill-backed command with the best latent rollout value.",
                    score=score,
                    reason="skill_command",
                    selected_skill_id=str(top_skill.get("skill_id", "")).strip() or None,
                    selected_retrieval_span_id=match_span_fn(
                        command,
                        retrieval_guidance.get("recommended_command_spans", []),
                    ),
                    retrieval_ranked_skill=True,
                )
            )

    if bool(decoder_policy.get("allow_task_suggestions", True)):
        for command in state.task.suggested_commands[: int(decoder_policy.get("max_task_suggestions", 3))]:
            command = normalize_command_fn(command, state.task.workspace_subdir)
            normalized = canonicalize_command_fn(command)
            if not normalized or normalized in seen or normalized in blocked:
                continue
            seen.add(normalized)
            score = _command_candidate_score(
                state=state,
                world_model=world_model,
                command=command,
                rollout_policy=rollout_policy,
                command_score_fn=command_score_fn,
            )
            candidates.append(
                DecodedActionCandidate(
                    action="code_execute",
                    content=command,
                    thought="Execute the retained Tolbert task-suggested command with the best latent rollout value.",
                    score=score,
                    reason="task_suggestion",
                    selected_retrieval_span_id=match_span_fn(
                        command,
                        retrieval_guidance.get("recommended_command_spans", []),
                    ),
                )
            )

    return sorted(candidates, key=lambda item: (-item.score, item.action, item.content))


def decode_action_generation_candidates(
    *,
    state: "AgentState",
    world_model: "WorldModel",
    skill_library: "SkillLibrary",
    top_skill: dict[str, object] | None,
    retrieval_guidance: dict[str, list[str]],
    blocked_commands: list[str],
    proposal_policy: dict[str, object],
    rollout_policy: dict[str, object],
    command_score_fn,
    normalize_command_fn,
    canonicalize_command_fn,
) -> list[DecodedActionCandidate]:
    if not bool(proposal_policy.get("enabled", True)):
        return []
    benchmark_family = str(state.task.metadata.get("benchmark_family", "bounded")).strip() or "bounded"
    blocked = {canonicalize_command_fn(command) for command in blocked_commands}
    baseline_commands = _baseline_command_set(
        state=state,
        skill_library=skill_library,
        top_skill=top_skill,
        retrieval_guidance=retrieval_guidance,
        canonicalize_command_fn=canonicalize_command_fn,
        normalize_command_fn=normalize_command_fn,
    )
    seen: set[str] = set()
    candidates: list[DecodedActionCandidate] = []
    template_support = _template_support_by_kind(proposal_policy, benchmark_family)
    missing_expected = [
        str(path).strip()
        for path in state.world_model_summary.get("missing_expected_artifacts", [])
        if str(path).strip()
    ]
    unsatisfied_expected = {
        str(path).strip(): str(content)
        for path, content in state.task.expected_file_contents.items()
        if str(path).strip()
        and str(content)
        and (
            str(path).strip() in missing_expected
            or str(path).strip()
            in {
                str(value).strip()
                for value in state.world_model_summary.get("unsatisfied_expected_contents", [])
                if str(value).strip()
            }
        )
    }
    forbidden_present = [
        str(path).strip()
        for path in state.world_model_summary.get("present_forbidden_artifacts", [])
        if str(path).strip()
    ]
    raw_proposals: list[tuple[str, str, dict[str, object]]] = []
    raw_proposals.extend(
        _structured_edit_raw_proposals(
            state=state,
            benchmark_family=benchmark_family,
            unsatisfied_expected=unsatisfied_expected,
            proposal_policy=proposal_policy,
        )
    )
    structured_preview_metadata = [
        metadata
        for _, proposal_source, metadata in raw_proposals
        if proposal_source.startswith("structured_edit:") and isinstance(metadata, dict)
    ]
    _refresh_workspace_preview_hidden_gap_decode_metadata(structured_preview_metadata)
    structured_edit_paths = {
        str(metadata.get("target_path", "")).strip()
        for _, proposal_source, metadata in raw_proposals
        if proposal_source.startswith("structured_edit:") and isinstance(metadata, dict)
    }
    structured_edit_fallback_paths = {
        str(metadata.get("target_path", "")).strip()
        for _, proposal_source, metadata in raw_proposals
        if proposal_source.startswith("structured_edit:")
        and isinstance(metadata, dict)
        and bool(metadata.get("allow_expected_write_fallback", False))
    }
    workspace_preview_fallback_summary = _workspace_preview_fallback_summary_by_path(
        structured_preview_metadata
    )
    for metadata in structured_preview_metadata:
        path = str(metadata.get("target_path", "")).strip()
        if not path:
            continue
        fallback_summary = workspace_preview_fallback_summary.get(path)
        if fallback_summary:
            metadata.update(fallback_summary)
    for path, content in unsatisfied_expected.items():
        if path in structured_edit_paths and path not in structured_edit_fallback_paths:
            continue
        fallback_summary = workspace_preview_fallback_summary.get(path, {})
        raw_proposals.append(
            (
                _render_write_command(path, content),
                "expected_file_content",
                {
                    "benchmark_family": benchmark_family,
                    "template_kind": "expected_file_content",
                    "target_path": path,
                    "support": template_support.get("expected_file_content", 0),
                    "safety": "bounded_write_expected_content",
                    "verifier_aligned": True,
                    "fallback_from_workspace_preview": path in structured_edit_fallback_paths,
                    "provenance": _template_provenance(proposal_policy, benchmark_family, "expected_file_content"),
                    **fallback_summary,
                },
            )
        )
    for path in missing_expected:
        if path in unsatisfied_expected:
            continue
        raw_proposals.append(
            (
                _render_touch_command(path),
                "missing_expected_file",
                {
                    "benchmark_family": benchmark_family,
                    "template_kind": "missing_expected_file",
                    "target_path": path,
                    "support": template_support.get("missing_expected_file", 0),
                    "safety": "bounded_materialize_expected_file",
                    "verifier_aligned": True,
                    "provenance": _template_provenance(proposal_policy, benchmark_family, "missing_expected_file"),
                },
            )
        )
    for path in forbidden_present:
        raw_proposals.append(
            (
                _render_cleanup_command(path),
                "cleanup_forbidden_file",
                {
                    "benchmark_family": benchmark_family,
                    "template_kind": "cleanup_forbidden_file",
                    "target_path": path,
                    "support": template_support.get("cleanup_forbidden_file", 0),
                    "safety": "bounded_remove_forbidden_file",
                    "verifier_aligned": True,
                    "provenance": _template_provenance(proposal_policy, benchmark_family, "cleanup_forbidden_file"),
                },
            )
        )
    _annotate_preview_block_proposal_families(structured_preview_metadata)
    for command, proposal_source, metadata in raw_proposals:
        command = normalize_command_fn(command, state.task.workspace_subdir)
        normalized = canonicalize_command_fn(command)
        if not normalized or normalized in seen or normalized in blocked or normalized in baseline_commands:
            continue
        seen.add(normalized)
        support = _float_value(metadata.get("support"), 0.0)
        score = _command_candidate_score(
            state=state,
            world_model=world_model,
            command=command,
            rollout_policy=rollout_policy,
            command_score_fn=command_score_fn,
        )
        score += _float_value(proposal_policy.get("proposal_score_bias"), 0.0)
        score += _float_value(proposal_policy.get("novel_command_bonus"), 0.0)
        if bool(metadata.get("verifier_aligned", False)):
            score += _float_value(proposal_policy.get("verifier_alignment_bonus"), 0.0)
        if proposal_source in {"expected_file_content", "missing_expected_file"}:
            score += _float_value(proposal_policy.get("expected_file_template_bonus"), 0.0)
        if proposal_source == "expected_file_content" and bool(metadata.get("fallback_from_workspace_preview", False)):
            score += 8.0
            score += _expected_file_content_workspace_preview_fallback_adjustment(metadata)
        if proposal_source == "cleanup_forbidden_file":
            score += _float_value(proposal_policy.get("cleanup_template_bonus"), 0.0)
        score += _edit_kind_bonus(str(metadata.get("edit_kind", "")))
        if proposal_source.startswith("structured_edit:"):
            score += 3.0
            score -= _structured_edit_precision_penalty(metadata)
        if str(metadata.get("edit_source", "")) == "synthetic_edit_plan":
            score += 1.0
        score += _structured_edit_window_bonus(metadata)
        score += _structured_edit_chain_bonus(metadata)
        score += _structured_edit_overlap_recovery_bonus(metadata)
        score += _structured_edit_conflicting_alias_recovery_adjustment(metadata)
        score += _structured_edit_unrecoverable_overlap_penalty(metadata)
        score += _structured_edit_hidden_gap_penalty(metadata)
        score += _structured_edit_hidden_gap_region_ambiguity_penalty(metadata)
        score += _structured_edit_hidden_gap_bounded_alternative_penalty(metadata)
        score += _structured_edit_hidden_gap_current_ambiguity_penalty(metadata)
        score += _structured_edit_hidden_gap_current_target_equivalent_penalty(metadata)
        score += _structured_edit_hidden_gap_current_proof_penalty(metadata)
        score += _structured_edit_hidden_gap_current_partial_proof_penalty(metadata)
        score += _structured_edit_explicit_current_span_proof_adjustment(metadata)
        score += _structured_edit_block_alternative_adjustment(metadata)
        score += _structured_edit_same_window_block_adjustment(metadata)
        score += _structured_edit_exact_region_block_adjustment(metadata)
        score += _structured_edit_hidden_gap_region_block_adjustment(metadata)
        score += _structured_edit_bridged_hidden_gap_region_block_adjustment(metadata)
        score += _structured_edit_current_proof_region_block_adjustment(metadata)
        score += _structured_edit_same_span_block_proof_quality_adjustment(metadata)
        score += _structured_edit_bridge_run_frontier_adjustment(metadata)
        score += _structured_edit_exact_localized_bridge_competition_adjustment(metadata)
        score += _structured_edit_coverage_adjustment(metadata)
        score += min(1.5, support * 0.1)
        metadata["novel_command"] = True
        metadata["baseline_overlap"] = False
        candidates.append(
            DecodedActionCandidate(
                action="code_execute",
                content=command,
                thought="Execute the retained Tolbert action-generation proposal with the best verifier-aligned latent value.",
                score=score,
                reason=f"action_generation:{proposal_source}",
                proposal_source=proposal_source,
                proposal_novel=True,
                proposal_metadata=metadata,
            )
        )
    max_candidates = max(0, int(proposal_policy.get("max_candidates", 4) or 0))
    return sorted(candidates, key=lambda item: (-item.score, item.action, item.content))[:max_candidates]


def _allow_stop_candidate(state: "AgentState", decoder_policy: dict[str, object]) -> bool:
    if not bool(decoder_policy.get("allow_stop_decision", True)):
        return False
    completion_ratio = _float_value(state.world_model_summary.get("completion_ratio"), 0.0)
    min_completion = _float_value(decoder_policy.get("min_stop_completion_ratio"), 0.95)
    if completion_ratio < min_completion:
        return False
    for key in (
        "missing_expected_artifacts",
        "present_forbidden_artifacts",
        "changed_preserved_artifacts",
        "unsatisfied_expected_contents",
    ):
        if list(state.world_model_summary.get(key, [])):
            return False
    if _long_horizon_stop_blocked_by_learned_world_signal(state):
        return False
    return True


def _refresh_workspace_preview_hidden_gap_decode_metadata(
    proposals: list[dict[str, object]],
) -> None:
    workspace_preview_proposals = [
        proposal
        for proposal in proposals
        if str(proposal.get("edit_source", "")).strip()
        in {"workspace_preview", "workspace_preview_range"}
    ]
    if not workspace_preview_proposals:
        return
    _annotate_hidden_gap_region_ambiguity(workspace_preview_proposals)
    _annotate_hidden_gap_current_ambiguity(workspace_preview_proposals)
    _annotate_hidden_gap_current_target_equivalent_ties(workspace_preview_proposals)
    _annotate_hidden_gap_current_proof_requirement(workspace_preview_proposals)


def _workspace_preview_fallback_summary_template() -> dict[str, object]:
    return {
        "workspace_preview_hidden_gap_region_ambiguity_count": 0,
        "workspace_preview_hidden_gap_current_ambiguity_count": 0,
        "workspace_preview_hidden_gap_current_target_equivalent_count": 0,
        "workspace_preview_hidden_gap_current_proof_required_count": 0,
        "workspace_preview_best_block_proof_quality": 0,
        "workspace_preview_has_complete_current_proof": False,
        "workspace_preview_has_explicit_hidden_gap_current_proof": False,
        "workspace_preview_has_bridged_hidden_gap_region_block": False,
    }


def _update_workspace_preview_fallback_summary(
    summary: dict[str, object],
    proposal: dict[str, object],
) -> None:
    summary["workspace_preview_hidden_gap_region_ambiguity_count"] = max(
        _int_value(summary.get("workspace_preview_hidden_gap_region_ambiguity_count"), 0),
        _int_value(proposal.get("hidden_gap_region_ambiguity_count"), 0),
        _int_value(proposal.get("workspace_preview_hidden_gap_region_ambiguity_count"), 0),
    )
    summary["workspace_preview_hidden_gap_current_ambiguity_count"] = max(
        _int_value(summary.get("workspace_preview_hidden_gap_current_ambiguity_count"), 0),
        _int_value(proposal.get("hidden_gap_current_ambiguity_count"), 0),
        _int_value(proposal.get("workspace_preview_hidden_gap_current_ambiguity_count"), 0),
    )
    summary["workspace_preview_hidden_gap_current_target_equivalent_count"] = max(
        _int_value(
            summary.get("workspace_preview_hidden_gap_current_target_equivalent_count"),
            0,
        ),
        _int_value(proposal.get("hidden_gap_current_target_equivalent_count"), 0),
        _int_value(
            proposal.get("workspace_preview_hidden_gap_current_target_equivalent_count"),
            0,
        ),
    )
    summary["workspace_preview_hidden_gap_current_proof_required_count"] = max(
        _int_value(
            summary.get("workspace_preview_hidden_gap_current_proof_required_count"),
            0,
        ),
        _int_value(proposal.get("hidden_gap_current_proof_required_count"), 0),
        _int_value(
            proposal.get("workspace_preview_hidden_gap_current_proof_required_count"),
            0,
        ),
    )
    summary["workspace_preview_best_block_proof_quality"] = max(
        _int_value(summary.get("workspace_preview_best_block_proof_quality"), 0),
        _int_value(proposal.get("workspace_preview_best_block_proof_quality"), 0),
        (
            _preview_block_proposal_proof_quality_score(proposal)
            if str(proposal.get("edit_kind", "")).strip() == "block_replace"
            else 0
        ),
    )
    if bool(proposal.get("current_proof_complete", False)) or bool(
        proposal.get("workspace_preview_has_complete_current_proof", False)
    ):
        summary["workspace_preview_has_complete_current_proof"] = True
    if bool(proposal.get("explicit_hidden_gap_current_proof", False)) or bool(
        proposal.get("workspace_preview_has_explicit_hidden_gap_current_proof", False)
    ):
        summary["workspace_preview_has_explicit_hidden_gap_current_proof"] = True
    if bool(proposal.get("bridged_hidden_gap_region_block", False)) or bool(
        proposal.get("workspace_preview_has_bridged_hidden_gap_region_block", False)
    ):
        summary["workspace_preview_has_bridged_hidden_gap_region_block"] = True


def _annotate_workspace_preview_path_fallback_summary(
    proposals: list[dict[str, object]],
) -> None:
    workspace_preview_proposals = [
        proposal
        for proposal in proposals
        if str(proposal.get("edit_source", "")).strip()
        in {"workspace_preview", "workspace_preview_range"}
    ]
    if not workspace_preview_proposals:
        return
    summary = _workspace_preview_fallback_summary_template()
    for proposal in workspace_preview_proposals:
        _update_workspace_preview_fallback_summary(summary, proposal)
    for proposal in workspace_preview_proposals:
        proposal.update(summary)


def _workspace_preview_fallback_summary_by_path(
    proposals: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    summaries: dict[str, dict[str, object]] = {}
    for proposal in proposals:
        path = str(proposal.get("target_path", "")).strip()
        if not path:
            continue
        if str(proposal.get("edit_source", "")).strip() not in {
            "workspace_preview",
            "workspace_preview_range",
        }:
            continue
        summary = summaries.setdefault(path, _workspace_preview_fallback_summary_template())
        _update_workspace_preview_fallback_summary(summary, proposal)
    return summaries


def _long_horizon_stop_blocked_by_learned_world_signal(state: "AgentState") -> bool:
    if str(state.world_model_summary.get("horizon", "")).strip() != "long_horizon":
        return False
    learned = state.latent_state_summary.get("learned_world_state", {})
    learned = learned if isinstance(learned, dict) else {}
    learned_progress_signal = max(
        _float_value(learned.get("progress_signal"), 0.0),
        _float_value(learned.get("world_progress_score"), 0.0),
        _float_value(learned.get("decoder_world_progress_score"), 0.0),
        _float_value(learned.get("transition_progress_score"), 0.0),
    )
    learned_risk_signal = max(
        _float_value(learned.get("risk_signal"), 0.0),
        _float_value(learned.get("world_risk_score"), 0.0),
        _float_value(learned.get("decoder_world_risk_score"), 0.0),
        _float_value(learned.get("transition_regression_score"), 0.0),
    )
    return learned_risk_signal >= 0.55 and learned_risk_signal > learned_progress_signal


def _command_candidate_score(
    *,
    state: "AgentState",
    world_model: "WorldModel",
    command: str,
    rollout_policy: dict[str, object],
    command_score_fn,
) -> float:
    control_score = float(command_score_fn(state, command))
    rollout_score = rollout_action_value(
        world_model_summary=state.world_model_summary,
        latent_state_summary=state.latent_state_summary,
        latest_transition=state.latest_state_transition,
        action="code_execute",
        content=command,
        rollout_policy=rollout_policy,
        world_model=world_model,
    )
    return control_score + rollout_score


def _float_value(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int_value(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _baseline_command_set(
    *,
    state: "AgentState",
    skill_library: "SkillLibrary",
    top_skill: dict[str, object] | None,
    retrieval_guidance: dict[str, list[str]],
    canonicalize_command_fn,
    normalize_command_fn,
) -> set[str]:
    baseline: set[str] = set()
    for item in retrieval_guidance.get("recommended_command_spans", []):
        command = normalize_command_fn(str(item.get("command", "")), state.task.workspace_subdir)
        normalized = canonicalize_command_fn(command)
        if normalized:
            baseline.add(normalized)
    if top_skill is not None:
        for command in skill_library._commands_for_skill(top_skill)[:1]:
            normalized = canonicalize_command_fn(
                normalize_command_fn(command, state.task.workspace_subdir)
            )
            if normalized:
                baseline.add(normalized)
    for command in state.task.suggested_commands:
        normalized = canonicalize_command_fn(
            normalize_command_fn(command, state.task.workspace_subdir)
        )
        if normalized:
            baseline.add(normalized)
    return baseline


def _template_support_by_kind(
    proposal_policy: dict[str, object],
    benchmark_family: str,
) -> dict[str, int]:
    preferences = proposal_policy.get("template_preferences", {})
    family_preferences = preferences.get(benchmark_family, []) if isinstance(preferences, dict) else []
    support: dict[str, int] = {}
    for item in family_preferences:
        if not isinstance(item, dict):
            continue
        template_kind = str(item.get("template_kind", "")).strip()
        if not template_kind:
            continue
        support[template_kind] = int(item.get("support", 0) or 0)
    return support


def _template_provenance(
    proposal_policy: dict[str, object],
    benchmark_family: str,
    template_kind: str,
) -> list[str]:
    preferences = proposal_policy.get("template_preferences", {})
    family_preferences = preferences.get(benchmark_family, []) if isinstance(preferences, dict) else []
    for item in family_preferences:
        if not isinstance(item, dict):
            continue
        if str(item.get("template_kind", "")).strip() != template_kind:
            continue
        return [
            str(value).strip()
            for value in item.get("provenance", [])
            if str(value).strip()
        ][:8]
    return []


def _render_write_command(path: str, content: str) -> str:
    parent = path.rsplit("/", 1)[0] if "/" in path else ""
    pieces: list[str] = []
    if parent:
        pieces.append(f"mkdir -p {_shell_quote(parent)}")
    pieces.append(f"cat <<'__AK_EOF__' > {_shell_quote(path)}\n{content}\n__AK_EOF__")
    return " && ".join(pieces) if len(pieces) > 1 else pieces[0]


def _render_touch_command(path: str) -> str:
    parent = path.rsplit("/", 1)[0] if "/" in path else ""
    pieces: list[str] = []
    if parent:
        pieces.append(f"mkdir -p {_shell_quote(parent)}")
    pieces.append(f"touch {_shell_quote(path)}")
    return " && ".join(pieces) if len(pieces) > 1 else pieces[0]


def _render_cleanup_command(path: str) -> str:
    return f"rm -f {_shell_quote(path)}"


def _shell_quote(value: str) -> str:
    escaped = str(value).replace("'", "'\"'\"'")
    return f"'{escaped}'"


def _structured_edit_raw_proposals(
    *,
    state: "AgentState",
    benchmark_family: str,
    unsatisfied_expected: dict[str, str],
    proposal_policy: dict[str, object],
) -> list[tuple[str, str, dict[str, object]]]:
    raw: list[tuple[str, str, dict[str, object]]] = []
    planned_steps = _synthetic_edit_plan_steps(state)
    for path, target_content in unsatisfied_expected.items():
        step = planned_steps.get(path)
        support = _template_support_by_kind(proposal_policy, benchmark_family).get("structured_edit", 0)
        if step is not None:
            command = _render_structured_edit_command(step)
            if not command:
                continue
            edit_kind = str(step.get("edit_kind", "")).strip()
            syntax_motor = summarize_python_edit_step(
                step,
                expected_file_contents=state.task.expected_file_contents,
            )
            raw.append(
                (
                    command,
                    f"structured_edit:{edit_kind or 'rewrite'}",
                    {
                        "command": command,
                        "benchmark_family": benchmark_family,
                        "template_kind": "structured_edit",
                        "target_path": path,
                        "proposal_source": f"structured_edit:{edit_kind or 'rewrite'}",
                        "edit_kind": edit_kind or "rewrite",
                        "edit_kinds": [],
                        "edit_source": "synthetic_edit_plan",
                        "window_index": None,
                        "window_indices": [],
                        "support": support,
                        "safety": "bounded_structured_edit",
                        "verifier_aligned": True,
                        "provenance": ["task:synthetic_edit_plan"],
                        "syntax_motor": syntax_motor,
                        "target_symbol_fqn": (
                            str(syntax_motor.get("edited_symbol_fqn", "")).strip()
                            if isinstance(syntax_motor, dict)
                            else ""
                        ),
                        "enclosing_symbol_qualname": (
                            str(syntax_motor.get("enclosing_symbol_qualname", "")).strip()
                            if isinstance(syntax_motor, dict)
                            else ""
                        ),
                        "import_change_risk": bool(syntax_motor.get("import_change_risk", False))
                        if isinstance(syntax_motor, dict)
                        else False,
                        "signature_change_risk": bool(syntax_motor.get("signature_change_risk", False))
                        if isinstance(syntax_motor, dict)
                        else False,
                        "call_targets": list(syntax_motor.get("call_targets_after", []))[:8]
                        if isinstance(syntax_motor, dict)
                        else [],
                    },
                )
            )
            continue
        preview_windows = _workspace_file_preview_windows(state, path, target_content)
        if not preview_windows:
            continue
        preview_proposals = _workspace_preview_window_proposals(
            path=path,
            preview_windows=preview_windows,
        )
        if not preview_proposals:
            continue
        for proposal in preview_proposals:
            raw.append(
                (
                    str(proposal.get("command", "")),
                    str(proposal.get("proposal_source", "")),
                    {
                        "command": str(proposal.get("command", "")),
                        "benchmark_family": benchmark_family,
                        "template_kind": "structured_edit",
                        "target_path": path,
                        "proposal_source": str(proposal.get("proposal_source", "")),
                        "edit_kind": str(proposal.get("edit_kind", "")).strip() or "rewrite",
                        "edit_kinds": proposal.get("component_edit_kinds", []),
                        "edit_source": str(proposal.get("edit_source", "")).strip(),
                        "window_index": proposal.get("window_index"),
                        "window_indices": proposal.get("window_indices", []),
                        "edit_window_count": proposal.get("edit_window_count"),
                        "available_window_count": proposal.get("available_window_count"),
                        "retained_window_count": proposal.get("retained_window_count"),
                        "total_window_count": proposal.get("total_window_count"),
                        "partial_window_coverage": proposal.get("partial_window_coverage"),
                        "allow_expected_write_fallback": proposal.get("allow_expected_write_fallback"),
                        "exact_target_span": proposal.get("exact_target_span"),
                        "covered_window_indices": proposal.get("covered_window_indices", []),
                        "covered_window_count": proposal.get("covered_window_count"),
                        "exact_window_count": proposal.get("exact_window_count"),
                        "inexact_window_count": proposal.get("inexact_window_count"),
                        "overlap_alias_pair_count": proposal.get("overlap_alias_pair_count"),
                        "overlap_alias_window_count": proposal.get("overlap_alias_window_count"),
                        "safe_inexact_window_count": proposal.get("safe_inexact_window_count"),
                        "recovered_conflicting_alias": proposal.get("recovered_conflicting_alias"),
                        "overlap_component_alias_pair_count": proposal.get("overlap_component_alias_pair_count"),
                        "overlap_component_unresolved_alias_pair_count": proposal.get(
                            "overlap_component_unresolved_alias_pair_count"
                        ),
                        "overlap_component_conflicting_alias_pair_count": proposal.get(
                            "overlap_component_conflicting_alias_pair_count"
                        ),
                        "overlap_component_candidate_count": proposal.get("overlap_component_candidate_count"),
                        "overlap_component_frontier_gap": proposal.get("overlap_component_frontier_gap"),
                        "overlap_component_unrecoverable_count": proposal.get(
                            "overlap_component_unrecoverable_count"
                        ),
                        "overlap_component_unrecoverable_window_count": proposal.get(
                            "overlap_component_unrecoverable_window_count"
                        ),
                        "hidden_gap_broader_region_required": proposal.get(
                            "hidden_gap_broader_region_required"
                        ),
                        "hidden_gap_line_count": proposal.get("hidden_gap_line_count"),
                        "hidden_gap_region_ambiguity_count": proposal.get(
                            "hidden_gap_region_ambiguity_count"
                        ),
                        "hidden_gap_region_frontier_count": proposal.get(
                            "hidden_gap_region_frontier_count"
                        ),
                        "hidden_gap_region_frontier_gap": proposal.get(
                            "hidden_gap_region_frontier_gap"
                        ),
                        "hidden_gap_bounded_alternative_count": proposal.get(
                            "hidden_gap_bounded_alternative_count"
                        ),
                        "hidden_gap_bounded_alternative_gap": proposal.get(
                            "hidden_gap_bounded_alternative_gap"
                        ),
                        "hidden_gap_region_ambiguous": proposal.get(
                            "hidden_gap_region_ambiguous"
                        ),
                        "hidden_gap_current_ambiguity_count": proposal.get(
                            "hidden_gap_current_ambiguity_count"
                        ),
                        "hidden_gap_current_frontier_count": proposal.get(
                            "hidden_gap_current_frontier_count"
                        ),
                        "hidden_gap_current_frontier_gap": proposal.get(
                            "hidden_gap_current_frontier_gap"
                        ),
                        "hidden_gap_current_ambiguous": proposal.get(
                            "hidden_gap_current_ambiguous"
                        ),
                        "hidden_gap_current_target_equivalent_count": proposal.get(
                            "hidden_gap_current_target_equivalent_count"
                        ),
                        "hidden_gap_current_target_equivalent_gap": proposal.get(
                            "hidden_gap_current_target_equivalent_gap"
                        ),
                        "hidden_gap_current_target_equivalent_ambiguous": proposal.get(
                            "hidden_gap_current_target_equivalent_ambiguous"
                        ),
                        "hidden_gap_current_proof_required_count": proposal.get(
                            "hidden_gap_current_proof_required_count"
                        ),
                        "hidden_gap_current_proof_required": proposal.get(
                            "hidden_gap_current_proof_required"
                        ),
                        "hidden_gap_current_partial_proof_count": proposal.get(
                            "hidden_gap_current_partial_proof_count"
                        ),
                        "hidden_gap_current_partial_proof_missing_line_count": proposal.get(
                            "hidden_gap_current_partial_proof_missing_line_count"
                        ),
                        "hidden_gap_current_partial_proof_span_count": proposal.get(
                            "hidden_gap_current_partial_proof_span_count"
                        ),
                        "hidden_gap_current_partial_proof_opaque_spans": proposal.get(
                            "hidden_gap_current_partial_proof_opaque_spans"
                        ),
                        "hidden_gap_current_partial_proof_topologies": proposal.get(
                            "hidden_gap_current_partial_proof_topologies"
                        ),
                        "hidden_gap_current_partial_proof_admissible_opaque_span_count": proposal.get(
                            "hidden_gap_current_partial_proof_admissible_opaque_span_count"
                        ),
                        "hidden_gap_current_partial_proof_blocking_opaque_span_count": proposal.get(
                            "hidden_gap_current_partial_proof_blocking_opaque_span_count"
                        ),
                        "hidden_gap_current_partial_proof_coarse_opaque_span_count": proposal.get(
                            "hidden_gap_current_partial_proof_coarse_opaque_span_count"
                        ),
                        "hidden_gap_current_partial_proof_mixed_opaque_topology": proposal.get(
                            "hidden_gap_current_partial_proof_mixed_opaque_topology"
                        ),
                        "precision_penalty": proposal.get("precision_penalty"),
                        "same_window_exact_block_alternative": proposal.get("same_window_exact_block_alternative"),
                        "localized_exact_block_alternative_count": proposal.get("localized_exact_block_alternative_count"),
                        "block_alternative_count": proposal.get("block_alternative_count"),
                        "block_exact_alternative_count": proposal.get("block_exact_alternative_count"),
                        "block_max_covered_window_count": proposal.get("block_max_covered_window_count"),
                        "block_max_exact_window_count": proposal.get("block_max_exact_window_count"),
                        "block_min_precision_penalty": proposal.get("block_min_precision_penalty"),
                        "block_min_edit_window_count": proposal.get("block_min_edit_window_count"),
                        "block_frontier_max_exact_window_count": proposal.get("block_frontier_max_exact_window_count"),
                        "block_frontier_min_inexact_window_count": proposal.get("block_frontier_min_inexact_window_count"),
                        "block_frontier_min_span_line_count": proposal.get("block_frontier_min_span_line_count"),
                        "same_span_block_alternative_count": proposal.get(
                            "same_span_block_alternative_count"
                        ),
                        "same_span_block_quality_score": proposal.get(
                            "same_span_block_quality_score"
                        ),
                        "same_span_block_max_quality_score": proposal.get(
                            "same_span_block_max_quality_score"
                        ),
                        "same_span_block_quality_gap": proposal.get(
                            "same_span_block_quality_gap"
                        ),
                        "same_span_block_proof_frontier": proposal.get(
                            "same_span_block_proof_frontier"
                        ),
                        "exact_contiguous_region_block": proposal.get("exact_contiguous_region_block"),
                        "shared_anchor_exact_region_block": proposal.get("shared_anchor_exact_region_block"),
                        "shared_anchor_pair_kinds": proposal.get("shared_anchor_pair_kinds"),
                        "shared_anchor_exact_neighbor_count": proposal.get(
                            "shared_anchor_exact_neighbor_count"
                        ),
                        "shared_anchor_core_count": proposal.get(
                            "shared_anchor_core_count"
                        ),
                        "shared_anchor_hybrid_component_count": proposal.get(
                            "shared_anchor_hybrid_component_count"
                        ),
                        "shared_anchor_mixed_insert_delete": proposal.get(
                            "shared_anchor_mixed_insert_delete"
                        ),
                        "exact_hidden_gap_region_block": proposal.get("exact_hidden_gap_region_block"),
                        "bridged_hidden_gap_region_block": proposal.get("bridged_hidden_gap_region_block"),
                        "current_proof_region_block": proposal.get("current_proof_region_block"),
                        "current_proof_partial_region_block": proposal.get(
                            "current_proof_partial_region_block"
                        ),
                        "current_proof_partial_region_topology": proposal.get(
                            "current_proof_partial_region_topology"
                        ),
                        "current_proof_span_count": proposal.get("current_proof_span_count"),
                        "current_proof_complete": proposal.get("current_proof_complete"),
                        "current_proof_partial_coverage": proposal.get(
                            "current_proof_partial_coverage"
                        ),
                        "current_proof_covered_line_count": proposal.get(
                            "current_proof_covered_line_count"
                        ),
                        "current_proof_missing_line_count": proposal.get(
                            "current_proof_missing_line_count"
                        ),
                        "current_proof_missing_span_count": proposal.get(
                            "current_proof_missing_span_count"
                        ),
                        "current_proof_opaque_spans": proposal.get(
                            "current_proof_opaque_spans"
                        ),
                        "current_proof_opaque_span_count": proposal.get(
                            "current_proof_opaque_span_count"
                        ),
                        "workspace_preview_hidden_gap_region_ambiguity_count": proposal.get(
                            "workspace_preview_hidden_gap_region_ambiguity_count"
                        ),
                        "workspace_preview_hidden_gap_current_ambiguity_count": proposal.get(
                            "workspace_preview_hidden_gap_current_ambiguity_count"
                        ),
                        "workspace_preview_hidden_gap_current_target_equivalent_count": proposal.get(
                            "workspace_preview_hidden_gap_current_target_equivalent_count"
                        ),
                        "workspace_preview_hidden_gap_current_proof_required_count": proposal.get(
                            "workspace_preview_hidden_gap_current_proof_required_count"
                        ),
                        "workspace_preview_best_block_proof_quality": proposal.get(
                            "workspace_preview_best_block_proof_quality"
                        ),
                        "workspace_preview_has_complete_current_proof": proposal.get(
                            "workspace_preview_has_complete_current_proof"
                        ),
                        "workspace_preview_has_explicit_hidden_gap_current_proof": proposal.get(
                            "workspace_preview_has_explicit_hidden_gap_current_proof"
                        ),
                        "workspace_preview_has_bridged_hidden_gap_region_block": proposal.get(
                            "workspace_preview_has_bridged_hidden_gap_region_block"
                        ),
                        "current_proof_opaque_line_count": proposal.get(
                            "current_proof_opaque_line_count"
                        ),
                        "current_proof_opaque_internal_span_count": proposal.get(
                            "current_proof_opaque_internal_span_count"
                        ),
                        "current_proof_opaque_max_span_line_count": proposal.get(
                            "current_proof_opaque_max_span_line_count"
                        ),
                        "current_proof_admissible_opaque_span_count": proposal.get(
                            "current_proof_admissible_opaque_span_count"
                        ),
                        "current_proof_blocking_opaque_span_count": proposal.get(
                            "current_proof_blocking_opaque_span_count"
                        ),
                        "current_proof_coarse_internal_opaque_span_count": proposal.get(
                            "current_proof_coarse_internal_opaque_span_count"
                        ),
                        "current_proof_mixed_opaque_topology": proposal.get(
                            "current_proof_mixed_opaque_topology"
                        ),
                        "current_proof_opaque_boundary_touch_count": proposal.get(
                            "current_proof_opaque_boundary_touch_count"
                        ),
                        "current_proof_factorized_subregion": proposal.get(
                            "current_proof_factorized_subregion"
                        ),
                        "current_proof_factorized_subregion_index": proposal.get(
                            "current_proof_factorized_subregion_index"
                        ),
                        "current_proof_factorized_subregion_count": proposal.get(
                            "current_proof_factorized_subregion_count"
                        ),
                        "current_proof_parent_partial_region_topology": proposal.get(
                            "current_proof_parent_partial_region_topology"
                        ),
                        "current_proof_parent_opaque_span_count": proposal.get(
                            "current_proof_parent_opaque_span_count"
                        ),
                        "current_proof_parent_blocking_opaque_span_count": proposal.get(
                            "current_proof_parent_blocking_opaque_span_count"
                        ),
                        "explicit_hidden_gap_current_proof": proposal.get("explicit_hidden_gap_current_proof"),
                        "hidden_gap_current_from_line_span_proof": proposal.get(
                            "hidden_gap_current_from_line_span_proof"
                        ),
                        "explicit_current_span_proof": proposal.get("explicit_current_span_proof"),
                        "hidden_gap_target_from_expected_content": proposal.get(
                            "hidden_gap_target_from_expected_content"
                        ),
                        "hidden_gap_bridge_count": proposal.get("hidden_gap_bridge_count"),
                        "bridge_run_segment_count": proposal.get("bridge_run_segment_count"),
                        "bridge_run_alternative_count": proposal.get("bridge_run_alternative_count"),
                        "bridge_run_partial_alternative_count": proposal.get(
                            "bridge_run_partial_alternative_count"
                        ),
                        "bridge_run_exact_localized_alternative_count": proposal.get(
                            "bridge_run_exact_localized_alternative_count"
                        ),
                        "bridge_run_frontier_gap": proposal.get("bridge_run_frontier_gap"),
                        "bridge_run_exact_localized_gap": proposal.get("bridge_run_exact_localized_gap"),
                        "bridge_run_max_covered_window_count": proposal.get(
                            "bridge_run_max_covered_window_count"
                        ),
                        "synthetic_single_line_block_replace_count": proposal.get(
                            "synthetic_single_line_block_replace_count"
                        ),
                        "hidden_gap_current_line_count": proposal.get("hidden_gap_current_line_count"),
                        "hidden_gap_target_line_count": proposal.get("hidden_gap_target_line_count"),
                        "current_span_line_count": proposal.get("current_span_line_count"),
                        "target_span_line_count": proposal.get("target_span_line_count"),
                        "span_line_count": proposal.get("span_line_count"),
                        "component_edit_kinds": proposal.get("component_edit_kinds"),
                        "component_line_deltas": proposal.get("component_line_deltas"),
                        "structural_window_count": proposal.get("structural_window_count"),
                        "overlap_alias_pair_count": proposal.get("overlap_alias_pair_count"),
                        "overlap_alias_window_count": proposal.get("overlap_alias_window_count"),
                        "safe_inexact_window_count": proposal.get("safe_inexact_window_count"),
                        "step": proposal.get("step"),
                        "support": support,
                        "safety": "bounded_structured_edit",
                        "verifier_aligned": True,
                        "provenance": ["world_model:workspace_file_previews"],
                    },
                )
            )
    return raw


def _synthetic_edit_plan_steps(state: "AgentState") -> dict[str, dict[str, object]]:
    metadata = getattr(state.task, "metadata", {})
    plan = metadata.get("synthetic_edit_plan", []) if isinstance(metadata, dict) else []
    steps: dict[str, dict[str, object]] = {}
    if not isinstance(plan, list):
        return steps
    for item in plan:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path", "")).strip()
        edit_kind = str(item.get("edit_kind", "")).strip()
        if path and edit_kind and edit_kind != "rewrite":
            steps[path] = dict(item)
    return steps


def _workspace_file_preview_windows(
    state: "AgentState",
    path: str,
    target_content: str,
) -> list[dict[str, object]]:
    previews = state.world_model_summary.get("workspace_file_previews", {})
    if not isinstance(previews, dict):
        return []
    preview = previews.get(path)
    if not isinstance(preview, dict):
        return []
    raw_windows = preview.get("edit_windows")
    preview_windows = raw_windows if isinstance(raw_windows, list) and raw_windows else [preview]
    retained_window_count = max(1, _int_value(preview.get("retained_edit_window_count"), len(preview_windows)))
    total_window_count = max(
        retained_window_count,
        _int_value(preview.get("total_edit_window_count"), retained_window_count),
    )
    resolved: list[dict[str, object]] = []
    for index, window in enumerate(preview_windows):
        if not isinstance(window, dict):
            continue
        resolved_window = _resolve_workspace_preview_window(window, target_content=target_content)
        if resolved_window is not None:
            resolved_window["window_index"] = max(
                0,
                _int_value(window.get("window_index"), index),
            )
            resolved_window["retained_window_count"] = retained_window_count
            resolved_window["total_window_count"] = total_window_count
            resolved_window["partial_window_coverage"] = total_window_count > retained_window_count
            resolved.append(resolved_window)
    raw_exact_proofs = preview.get("exact_edit_window_proofs", [])
    if isinstance(raw_exact_proofs, list) and raw_exact_proofs:
        for proof_index, window in enumerate(raw_exact_proofs):
            if not isinstance(window, dict):
                continue
            resolved_window = _resolve_workspace_preview_exact_proof_window(
                window,
                proof_index=proof_index,
                target_content=target_content,
            )
            if resolved_window is None:
                continue
            resolved_window["retained_window_count"] = retained_window_count
            resolved_window["total_window_count"] = total_window_count
            resolved_window["partial_window_coverage"] = total_window_count > retained_window_count
            resolved.append(resolved_window)
    raw_bridge_runs = preview.get("bridged_edit_window_runs", [])
    if isinstance(raw_bridge_runs, list) and raw_bridge_runs:
        for bridge_index, window in enumerate(raw_bridge_runs):
            if not isinstance(window, dict):
                continue
            resolved_window = _resolve_workspace_preview_bridge_window(
                window,
                bridge_index=bridge_index,
                include_bridge_segments=True,
            )
            if resolved_window is None:
                continue
            resolved_window["retained_window_count"] = retained_window_count
            resolved_window["total_window_count"] = total_window_count
            resolved_window["partial_window_coverage"] = total_window_count > retained_window_count
            resolved.append(resolved_window)
    else:
        raw_bridge_windows = preview.get("bridged_edit_windows", [])
        if isinstance(raw_bridge_windows, list):
            for bridge_index, window in enumerate(raw_bridge_windows):
                if not isinstance(window, dict):
                    continue
                resolved_window = _resolve_workspace_preview_bridge_window(
                    window,
                    bridge_index=bridge_index,
                    include_bridge_segments=False,
                )
                if resolved_window is None:
                    continue
                resolved_window["retained_window_count"] = retained_window_count
                resolved_window["total_window_count"] = total_window_count
                resolved_window["partial_window_coverage"] = total_window_count > retained_window_count
                resolved.append(resolved_window)
    raw_current_proof_regions = preview.get("hidden_gap_current_proof_regions", [])
    if isinstance(raw_current_proof_regions, list):
        for region_index, region in enumerate(raw_current_proof_regions):
            if not isinstance(region, dict):
                continue
            resolved_region = _resolve_workspace_preview_current_proof_region(
                region,
                region_index=region_index,
            )
            if resolved_region is None:
                continue
            resolved_region["retained_window_count"] = retained_window_count
            resolved_region["total_window_count"] = total_window_count
            resolved_region["partial_window_coverage"] = total_window_count > retained_window_count
            resolved.append(resolved_region)
    return resolved


def _resolve_workspace_preview_exact_proof_window(
    preview: dict[str, object],
    *,
    proof_index: int,
    target_content: str,
) -> dict[str, object] | None:
    line_start = max(1, _int_value(preview.get("line_start"), 1))
    line_end = max(line_start - 1, _int_value(preview.get("line_end"), line_start - 1))
    target_line_start = max(1, _int_value(preview.get("target_line_start"), 1))
    target_line_end = max(
        target_line_start - 1,
        _int_value(preview.get("target_line_end"), target_line_start - 1),
    )
    if line_end < line_start and target_line_end < target_line_start:
        return None
    if target_line_end < target_line_start:
        resolved_target_content = ""
    else:
        expected_target_lines = _expected_target_lines_for_span(
            target_content,
            start_line=target_line_start,
            end_line=target_line_end,
        )
        if expected_target_lines is None:
            return None
        resolved_target_content = "".join(expected_target_lines)
    return {
        "window_index": _int_value(preview.get("window_index"), proof_index),
        "proof_index": proof_index,
        "truncated": True,
        "explicit_current_span_proof": bool(preview.get("explicit_current_span_proof", True)),
        "line_start": line_start,
        "line_end": line_end,
        "target_line_start": target_line_start,
        "target_line_end": target_line_end,
        "current_line_count": max(0, _int_value(preview.get("current_line_count"), line_end - line_start + 1)),
        "target_line_count": max(0, _int_value(preview.get("target_line_count"), target_line_end - target_line_start + 1)),
        "line_delta": _int_value(
            preview.get("line_delta"),
            max(0, target_line_end - target_line_start + 1) - max(0, line_end - line_start + 1),
        ),
        "target_content": resolved_target_content,
        "target_line_offset": target_line_start - 1,
        "exact_target_span": True,
        "full_target_content": target_content,
    }


def _resolve_workspace_preview_bridge_window(
    preview: dict[str, object],
    *,
    bridge_index: int,
    include_bridge_segments: bool,
) -> dict[str, object] | None:
    bridge_window_indices = sorted(
        {
            _int_value(index, -1)
            for index in preview.get("bridge_window_indices", [])
            if _int_value(index, -1) >= 0
        }
    )
    if len(bridge_window_indices) < 2:
        return None
    resolved_window = {
        "window_index": None,
        "bridge_index": bridge_index,
        "bridge_window_indices": bridge_window_indices,
        "truncated": bool(preview.get("truncated", True)),
        "line_start": max(1, _int_value(preview.get("line_start"), 1)),
        "line_end": max(
            _int_value(preview.get("line_start"), 1) - 1,
            _int_value(preview.get("line_end"), _int_value(preview.get("line_start"), 1) - 1),
        ),
        "target_line_start": max(1, _int_value(preview.get("target_line_start"), 1)),
        "target_line_end": max(
            _int_value(preview.get("target_line_start"), 1) - 1,
            _int_value(
                preview.get("target_line_end"),
                _int_value(preview.get("target_line_start"), 1) - 1,
            ),
        ),
        "hidden_gap_current_content": str(preview.get("hidden_gap_current_content", "")),
        "hidden_gap_target_content": str(preview.get("hidden_gap_target_content", "")),
        "hidden_gap_current_from_line_span_proof": bool(
            preview.get("hidden_gap_current_from_line_span_proof", False)
        ),
        "hidden_gap_target_from_expected_content": bool(
            preview.get("hidden_gap_target_from_expected_content", False)
        ),
        "explicit_hidden_gap_current_proof": bool(
            preview.get("explicit_hidden_gap_current_proof", False)
        ),
        "hidden_gap_current_line_start": max(
            1,
            _int_value(preview.get("hidden_gap_current_line_start"), 1),
        ),
        "hidden_gap_current_line_end": max(
            _int_value(preview.get("hidden_gap_current_line_start"), 1) - 1,
            _int_value(
                preview.get("hidden_gap_current_line_end"),
                _int_value(preview.get("hidden_gap_current_line_start"), 1) - 1,
            ),
        ),
        "hidden_gap_target_line_start": max(
            1,
            _int_value(preview.get("hidden_gap_target_line_start"), 1),
        ),
        "hidden_gap_target_line_end": max(
            _int_value(preview.get("hidden_gap_target_line_start"), 1) - 1,
            _int_value(
                preview.get("hidden_gap_target_line_end"),
                _int_value(preview.get("hidden_gap_target_line_start"), 1) - 1,
            ),
        ),
        "hidden_gap_current_line_count": max(
            0,
            _int_value(preview.get("hidden_gap_current_line_count"), 0),
        ),
        "hidden_gap_target_line_count": max(
            0,
            _int_value(preview.get("hidden_gap_target_line_count"), 0),
        ),
        "line_delta": _int_value(preview.get("line_delta"), 0),
    }
    if not include_bridge_segments:
        return resolved_window
    raw_segments = preview.get("bridge_segments", [])
    if not isinstance(raw_segments, list) or not raw_segments:
        return None
    resolved_segments: list[dict[str, object]] = []
    for segment in raw_segments:
        if not isinstance(segment, dict):
            return None
        resolved_segment = _resolve_workspace_preview_bridge_segment(segment)
        if resolved_segment is None:
            return None
        resolved_segments.append(resolved_segment)
    resolved_window["bridge_segments"] = resolved_segments
    return resolved_window


def _resolve_workspace_preview_bridge_segment(
    segment: dict[str, object],
) -> dict[str, object] | None:
    bridge_window_indices = sorted(
        {
            _int_value(index, -1)
            for index in segment.get("bridge_window_indices", [])
            if _int_value(index, -1) >= 0
        }
    )
    left_window_index = _int_value(segment.get("left_window_index"), -1)
    right_window_index = _int_value(segment.get("right_window_index"), -1)
    if len(bridge_window_indices) != 2:
        if left_window_index >= 0 and right_window_index >= 0:
            bridge_window_indices = [left_window_index, right_window_index]
        else:
            return None
    if bridge_window_indices[1] != bridge_window_indices[0] + 1:
        return None
    return {
        "bridge_window_indices": bridge_window_indices,
        "left_window_index": bridge_window_indices[0],
        "right_window_index": bridge_window_indices[1],
        "line_start": max(1, _int_value(segment.get("line_start"), 1)),
        "line_end": max(
            _int_value(segment.get("line_start"), 1) - 1,
            _int_value(segment.get("line_end"), _int_value(segment.get("line_start"), 1) - 1),
        ),
        "target_line_start": max(1, _int_value(segment.get("target_line_start"), 1)),
        "target_line_end": max(
            _int_value(segment.get("target_line_start"), 1) - 1,
            _int_value(
                segment.get("target_line_end"),
                _int_value(segment.get("target_line_start"), 1) - 1,
            ),
        ),
        "hidden_gap_current_line_start": max(
            1,
            _int_value(segment.get("hidden_gap_current_line_start"), 1),
        ),
        "hidden_gap_current_line_end": max(
            _int_value(segment.get("hidden_gap_current_line_start"), 1) - 1,
            _int_value(
                segment.get("hidden_gap_current_line_end"),
                _int_value(segment.get("hidden_gap_current_line_start"), 1) - 1,
            ),
        ),
        "hidden_gap_target_line_start": max(
            1,
            _int_value(segment.get("hidden_gap_target_line_start"), 1),
        ),
        "hidden_gap_target_line_end": max(
            _int_value(segment.get("hidden_gap_target_line_start"), 1) - 1,
            _int_value(
                segment.get("hidden_gap_target_line_end"),
                _int_value(segment.get("hidden_gap_target_line_start"), 1) - 1,
            ),
        ),
        "hidden_gap_current_content": str(segment.get("hidden_gap_current_content", "")),
        "hidden_gap_target_content": str(segment.get("hidden_gap_target_content", "")),
        "hidden_gap_current_from_line_span_proof": bool(
            segment.get("hidden_gap_current_from_line_span_proof", False)
        ),
        "hidden_gap_target_from_expected_content": bool(
            segment.get("hidden_gap_target_from_expected_content", False)
        ),
        "hidden_gap_current_line_count": max(
            0,
            _int_value(segment.get("hidden_gap_current_line_count"), 0),
        ),
        "hidden_gap_target_line_count": max(
            0,
            _int_value(segment.get("hidden_gap_target_line_count"), 0),
        ),
        "line_delta": _int_value(segment.get("line_delta"), 0),
        "explicit_hidden_gap_current_proof": bool(
            segment.get("explicit_hidden_gap_current_proof", False)
        ),
    }


def _resolve_workspace_preview_current_proof_region(
    preview: dict[str, object],
    *,
    region_index: int,
) -> dict[str, object] | None:
    window_indices = sorted(
        {
            _int_value(index, -1)
            for index in preview.get("window_indices", [])
            if _int_value(index, -1) >= 0
        }
    )
    if len(window_indices) < 2:
        return None
    raw_spans = preview.get("current_proof_spans", [])
    if not isinstance(raw_spans, list) or len(raw_spans) < 2:
        return None
    resolved_spans: list[dict[str, object]] = []
    resolved_opaque_spans: list[dict[str, object]] = []
    current_proof_covered_line_count = 0
    current_proof_missing_line_count = 0
    current_proof_missing_span_count = 0
    for span in raw_spans:
        if not isinstance(span, dict):
            return None
        current_line_start = max(1, _int_value(span.get("current_line_start"), 1))
        current_line_end = max(
            current_line_start - 1,
            _int_value(span.get("current_line_end"), current_line_start - 1),
        )
        target_line_start = max(1, _int_value(span.get("target_line_start"), 1))
        target_line_end = max(
            target_line_start - 1,
            _int_value(span.get("target_line_end"), target_line_start - 1),
        )
        resolved_spans.append(
            {
                "current_line_start": current_line_start,
                "current_line_end": current_line_end,
                "target_line_start": target_line_start,
                "target_line_end": target_line_end,
                "current_content": str(span.get("current_content", "")),
                "target_content": str(span.get("target_content", "")),
                "current_from_line_span_proof": bool(
                    span.get("current_from_line_span_proof", False)
                ),
                "target_from_expected_content": bool(
                    span.get("target_from_expected_content", False)
                ),
            }
        )
        current_line_count = max(0, current_line_end - current_line_start + 1)
        current_content = str(span.get("current_content", ""))
        current_from_line_span_proof = bool(span.get("current_from_line_span_proof", False))
        current_content_complete = (
            current_line_count <= 0 or len(current_content.splitlines()) == current_line_count
        )
        if current_from_line_span_proof or current_content_complete:
            current_proof_covered_line_count += current_line_count
        elif current_line_count > 0:
            current_proof_missing_line_count += current_line_count
            current_proof_missing_span_count += 1
    raw_opaque_spans = preview.get("current_proof_opaque_spans", [])
    if isinstance(raw_opaque_spans, list):
        for span in raw_opaque_spans:
            if not isinstance(span, dict):
                return None
            current_line_start = max(1, _int_value(span.get("current_line_start"), 1))
            current_line_end = max(
                current_line_start - 1,
                _int_value(span.get("current_line_end"), current_line_start - 1),
            )
            if current_line_end < current_line_start:
                continue
            target_line_start = max(1, _int_value(span.get("target_line_start"), 1))
            target_line_end = max(
                target_line_start - 1,
                _int_value(span.get("target_line_end"), target_line_start - 1),
            )
            resolved_opaque_spans.append(
                {
                    "current_line_start": current_line_start,
                    "current_line_end": current_line_end,
                    "target_line_start": target_line_start,
                    "target_line_end": target_line_end,
                    "reason": str(span.get("reason", "")).strip() or "unknown",
                }
            )
    preview_covered_line_count = max(
        0,
        _int_value(
            preview.get("current_proof_covered_line_count"),
            current_proof_covered_line_count,
        ),
    )
    preview_missing_line_count = max(
        0,
        _int_value(
            preview.get("current_proof_missing_line_count"),
            current_proof_missing_line_count,
        ),
    )
    preview_missing_span_count = max(
        0,
        _int_value(
            preview.get("current_proof_missing_span_count"),
            current_proof_missing_span_count,
        ),
    )
    preview_opaque_span_count = max(
        len(resolved_opaque_spans),
        _int_value(preview.get("current_proof_opaque_span_count"), len(resolved_opaque_spans)),
    )
    return {
        "current_proof_region": True,
        "proof_region_index": _int_value(preview.get("proof_region_index"), region_index),
        "window_indices": window_indices,
        "line_start": max(1, _int_value(preview.get("line_start"), 1)),
        "line_end": max(
            _int_value(preview.get("line_start"), 1) - 1,
            _int_value(preview.get("line_end"), _int_value(preview.get("line_start"), 1) - 1),
        ),
        "target_line_start": max(1, _int_value(preview.get("target_line_start"), 1)),
        "target_line_end": max(
            _int_value(preview.get("target_line_start"), 1) - 1,
            _int_value(
                preview.get("target_line_end"),
                _int_value(preview.get("target_line_start"), 1) - 1,
            ),
        ),
        "current_proof_span_count": max(
            len(resolved_spans),
            _int_value(preview.get("current_proof_span_count"), len(resolved_spans)),
        ),
        "current_proof_spans": resolved_spans,
        "current_proof_opaque_spans": resolved_opaque_spans,
        "current_proof_opaque_span_count": preview_opaque_span_count,
        "current_proof_complete": bool(
            preview.get("current_proof_complete", preview_missing_line_count <= 0)
        ),
        "current_proof_partial_coverage": bool(
            preview.get(
                "current_proof_partial_coverage",
                preview_covered_line_count > 0 and preview_missing_line_count > 0,
            )
        ),
        "current_proof_covered_line_count": preview_covered_line_count,
        "current_proof_missing_line_count": preview_missing_line_count,
        "current_proof_missing_span_count": preview_missing_span_count,
        "truncated": bool(preview.get("truncated", True)),
        "explicit_hidden_gap_current_proof": bool(
            preview.get("explicit_hidden_gap_current_proof", False)
        ),
        "hidden_gap_current_from_line_span_proof": bool(
            preview.get("hidden_gap_current_from_line_span_proof", False)
        ),
        "hidden_gap_target_from_expected_content": bool(
            preview.get("hidden_gap_target_from_expected_content", False)
        ),
    }


def _resolve_workspace_preview_window(
    preview: dict[str, object],
    *,
    target_content: str,
) -> dict[str, object] | None:
    truncated = bool(preview.get("truncated", False))
    if not truncated:
        content = str(preview.get("content", ""))
        if not content:
            return None
        line_start = max(1, _int_value(preview.get("line_start"), 1))
        target_line_start = max(1, _int_value(preview.get("target_line_start"), line_start))
        return {
            "baseline_content": content,
            "target_content": target_content,
            "full_target_content": target_content,
            "line_offset": line_start - 1,
            "target_line_offset": target_line_start - 1,
            "truncated": False,
            "line_delta": _int_value(preview.get("line_delta"), 0),
            "exact_target_span": True,
        }
    edit_content = str(preview.get("edit_content", ""))
    if not edit_content:
        return None
    target_edit_content = str(preview.get("target_edit_content", ""))
    line_start = max(1, _int_value(preview.get("line_start"), 1))
    line_end = max(line_start - 1, _int_value(preview.get("line_end"), line_start + len(edit_content.splitlines()) - 1))
    preferred_line_count = max(0, line_end - line_start + 1)
    target_window = target_edit_content
    explicit_target_line_start = _int_value(preview.get("target_line_start"), line_start)
    explicit_target_line_end = _int_value(
        preview.get("target_line_end"),
        explicit_target_line_start + len(target_window.splitlines()) - 1,
    )
    explicit_empty_target_span = (
        "target_line_start" in preview
        and "target_line_end" in preview
        and explicit_target_line_end < explicit_target_line_start
    )
    if not target_window and explicit_empty_target_span:
        target_window = ""
    elif not target_window:
        omitted_prefix_sha1 = str(preview.get("omitted_prefix_sha1", "")).strip() or _sha1_text("")
        omitted_suffix_sha1 = (
            str(preview.get("omitted_suffix_sha1", "")).strip()
            or str(preview.get("omitted_sha1", "")).strip()
        )
        if not omitted_suffix_sha1:
            return None
        target_window = _matching_target_preview_content(
            target_content,
            omitted_prefix_sha1=omitted_prefix_sha1,
            omitted_suffix_sha1=omitted_suffix_sha1,
            preferred_line_count=preferred_line_count,
            preferred_line_start=line_start,
        )
    if target_window is None:
        return None
    target_line_start = max(1, explicit_target_line_start)
    target_line_end = max(
        target_line_start - 1,
        _int_value(
            preview.get("target_line_end"),
            target_line_start + len(target_window.splitlines()) - 1,
        ),
    )
    return {
        "baseline_content": edit_content,
        "target_content": target_window,
        "full_target_content": target_content,
        "line_offset": line_start - 1,
        "target_line_offset": target_line_start - 1,
        "truncated": True,
        "line_delta": _int_value(
            preview.get("line_delta"),
            max(0, target_line_end - target_line_start + 1) - preferred_line_count,
        ),
        "exact_target_span": "target_line_start" in preview and "target_line_end" in preview,
    }


def _workspace_preview_window_proposals(
    *,
    path: str,
    preview_windows: list[dict[str, object]],
) -> list[dict[str, object]]:
    current_proof_regions = [
        preview_window
        for preview_window in preview_windows
        if bool(preview_window.get("current_proof_region", False))
    ]
    bridge_windows = [
        preview_window
        for preview_window in preview_windows
        if not bool(preview_window.get("current_proof_region", False))
        if isinstance(preview_window.get("bridge_window_indices"), list)
        and len(preview_window.get("bridge_window_indices", [])) >= 2
    ]
    proof_preview_windows = [
        preview_window
        for preview_window in preview_windows
        if bool(preview_window.get("explicit_current_span_proof", False))
        and not bool(preview_window.get("current_proof_region", False))
        and not (
            isinstance(preview_window.get("bridge_window_indices"), list)
            and len(preview_window.get("bridge_window_indices", [])) >= 2
        )
    ]
    retained_preview_windows = [
        preview_window
        for preview_window in preview_windows
        if not bool(preview_window.get("explicit_current_span_proof", False))
        and not bool(preview_window.get("current_proof_region", False))
        and not (
            isinstance(preview_window.get("bridge_window_indices"), list)
            and len(preview_window.get("bridge_window_indices", [])) >= 2
        )
    ]
    bridge_source_windows = retained_preview_windows + proof_preview_windows
    single_window_proposals: list[dict[str, object]] = []
    for preview_window in retained_preview_windows:
        exact_target_span = bool(
            preview_window.get("exact_target_span", "target_line_offset" in preview_window)
        )
        steps = _preview_window_candidate_steps(
            path=path,
            baseline_content=str(preview_window.get("baseline_content", "")),
            target_content=str(preview_window.get("target_content", "")),
            exact_target_span=exact_target_span,
        )
        if not steps:
            continue
        edit_source = (
            "workspace_preview_range"
            if bool(preview_window.get("truncated", False))
            else "workspace_preview"
        )
        covered_window_indices = _covered_preview_window_indices(preview_window, preview_windows)
        seen_commands: set[str] = set()
        for raw_step in steps:
            step = _offset_preview_step(
                raw_step,
                line_offset=_int_value(preview_window.get("line_offset"), 0),
                target_line_offset=_int_value(preview_window.get("target_line_offset"), 0),
            )
            command = _render_structured_edit_command(step)
            if not command or command in seen_commands:
                continue
            seen_commands.add(command)
            single_window_proposals.append(
                {
                    "command": command,
                    "proposal_source": f"structured_edit:{str(step.get('edit_kind', '')).strip() or 'rewrite'}",
                    "edit_kind": str(step.get("edit_kind", "")).strip() or "rewrite",
                    "edit_source": edit_source,
                    "window_index": _int_value(preview_window.get("window_index"), 0),
                    "edit_window_count": 1,
                    "covered_window_indices": covered_window_indices,
                    "covered_window_count": len(covered_window_indices),
                    "exact_target_span": exact_target_span,
                    "precision_penalty": _preview_step_precision_penalty(
                        step,
                        exact_target_span=exact_target_span,
                    ),
                    "exact_window_count": 1 if exact_target_span else 0,
                    "inexact_window_count": 0 if exact_target_span else 1,
                    "retained_window_count": _int_value(preview_window.get("retained_window_count"), 1),
                    "total_window_count": _int_value(preview_window.get("total_window_count"), 1),
                    "partial_window_coverage": bool(preview_window.get("partial_window_coverage", False)),
                    "step": step,
                }
            )
    exact_proof_block_replaces = _compose_workspace_preview_exact_proof_block_replace_proposals(
        path=path,
        proof_windows=proof_preview_windows,
    )
    direct_proposals = list(single_window_proposals)
    direct_proposals.extend(exact_proof_block_replaces)
    available_window_count = len(
        {
            _int_value(proposal.get("window_index"), -1)
            for proposal in direct_proposals
            if _int_value(proposal.get("window_index"), -1) >= 0
        }
    )
    for proposal in direct_proposals:
        proposal["available_window_count"] = available_window_count
    overlap_block_replaces = _compose_workspace_preview_overlap_block_replace_proposals(
        path=path,
        preview_windows=retained_preview_windows,
        available_window_count=available_window_count,
    )
    exact_region_block_replaces = _compose_workspace_preview_exact_region_block_replace_proposals(
        path=path,
        preview_windows=retained_preview_windows,
        available_window_count=available_window_count,
    )
    bridged_region_block_replaces = _compose_workspace_preview_bridged_region_block_replace_proposals(
        path=path,
        bridge_windows=bridge_windows,
        preview_windows=bridge_source_windows,
        available_window_count=available_window_count,
    )
    current_proof_region_block_replaces = _compose_workspace_preview_current_proof_region_block_replace_proposals(
        path=path,
        proof_regions=current_proof_regions,
        preview_windows=bridge_source_windows,
        available_window_count=available_window_count,
    )
    risk_summary = _workspace_preview_risk_summary(
        path=path,
        preview_windows=retained_preview_windows,
        available_window_count=available_window_count,
        proof_regions=current_proof_regions,
    )
    proposals = list(direct_proposals)
    proposals.extend(overlap_block_replaces)
    proposals.extend(exact_region_block_replaces)
    proposals.extend(bridged_region_block_replaces)
    proposals.extend(current_proof_region_block_replaces)
    proposals.extend(
        _compose_workspace_preview_frontier_block_replace_proposals(
            path=path,
            proposals=proposals,
            available_window_count=available_window_count,
        )
    )
    proposals = _prune_same_span_preview_block_proposals(proposals)
    composite = _compose_workspace_preview_window_proposal(path=path, proposals=proposals)
    if composite is not None:
        proposals.append(composite)
    proposals = _prune_cross_family_preview_bounded_duplicates(proposals)
    hidden_gap_region_ambiguity_count = _annotate_hidden_gap_region_ambiguity(proposals)
    hidden_gap_current_ambiguity_count = _annotate_hidden_gap_current_ambiguity(proposals)
    hidden_gap_current_target_equivalent_count = _annotate_hidden_gap_current_target_equivalent_ties(
        proposals
    )
    hidden_gap_current_proof_required_count = _annotate_hidden_gap_current_proof_requirement(
        proposals
    )
    _annotate_workspace_preview_risks(proposals, risk_summary)
    hidden_gap_current_partial_proof_count = max(
        0,
        _int_value(risk_summary.get("hidden_gap_current_partial_proof_count"), 0),
    )
    _annotate_preview_block_proposal_families(proposals)
    preview_window_count = len(
        {
            _int_value(proposal.get("window_index"), -1)
            for proposal in proposals
            if _int_value(proposal.get("window_index"), -1) >= 0
        }
    )
    path_has_exact_window = any(
        bool(proposal.get("exact_target_span", False))
        for proposal in proposals
        if proposal.get("window_index") is not None
    )
    path_has_inexact_window = any(
        not bool(proposal.get("exact_target_span", False))
        for proposal in proposals
        if proposal.get("window_index") is not None
    )
    partial_window_coverage = any(
        bool(proposal.get("partial_window_coverage", False))
        for proposal in proposals
    )
    unrecovered_overlap_component = _preview_has_unrecovered_overlap_component(
        preview_windows=retained_preview_windows,
        overlap_block_replaces=overlap_block_replaces,
    )
    safe_full_coverage = any(
        _int_value(proposal.get("covered_window_count"), 0) >= max(
            1,
            _int_value(proposal.get("retained_window_count"), preview_window_count or 1),
        )
        and not bool(proposal.get("partial_window_coverage", False))
        and _int_value(proposal.get("overlap_component_unresolved_alias_pair_count"), 0) <= 0
        and _int_value(proposal.get("hidden_gap_region_ambiguity_count"), 0) <= 1
        and _int_value(proposal.get("hidden_gap_current_ambiguity_count"), 0) <= 1
        and _int_value(proposal.get("hidden_gap_current_target_equivalent_count"), 0) <= 1
        and _int_value(proposal.get("hidden_gap_current_proof_required_count"), 0) <= 0
        and _int_value(proposal.get("safe_inexact_window_count"), 0)
        >= _int_value(proposal.get("inexact_window_count"), 0)
        and _int_value(proposal.get("overlap_alias_pair_count"), 0) > 0
        for proposal in proposals
    )
    if (
        preview_window_count > 1
        and path_has_exact_window
        and path_has_inexact_window
        and not safe_full_coverage
    ) or partial_window_coverage or unrecovered_overlap_component or hidden_gap_region_ambiguity_count > 1 or hidden_gap_current_ambiguity_count > 1 or hidden_gap_current_target_equivalent_count > 1 or hidden_gap_current_proof_required_count > 0 or hidden_gap_current_partial_proof_count > 0:
        for proposal in proposals:
            proposal["allow_expected_write_fallback"] = True
    _annotate_workspace_preview_path_fallback_summary(proposals)
    return proposals


def _preview_window_candidate_steps(
    *,
    path: str,
    baseline_content: str,
    target_content: str,
    exact_target_span: bool,
) -> list[dict[str, object]]:
    primary_step = _best_preview_edit_step(
        path=path,
        baseline_content=baseline_content,
        target_content=target_content,
        exact_target_span=exact_target_span,
    )
    if primary_step is None:
        return []
    steps = [primary_step]
    if exact_target_span and str(primary_step.get("edit_kind", "")).strip() != "block_replace":
        block_step = _derive_block_replace_step(
            path=path,
            baseline_content=baseline_content,
            target_content=target_content,
        )
        if block_step is not None:
            steps.append(block_step)
    return steps


def _compose_workspace_preview_window_proposal(
    *,
    path: str,
    proposals: list[dict[str, object]],
) -> dict[str, object] | None:
    if len(proposals) < 2:
        return None
    span_proposals: list[dict[str, object]] = []
    for proposal in proposals:
        step = proposal.get("step")
        if not isinstance(step, dict):
            continue
        current_span = _step_current_span(step)
        target_span = _step_target_span(step)
        if current_span is None or target_span is None:
            return None
        span_proposals.append(
            {
                "proposal": proposal,
                "step": step,
                "current_start": current_span[0],
                "current_end": current_span[1],
                "target_start": target_span[0],
                "target_end": target_span[1],
                "window_index": _int_value(proposal.get("window_index"), 0),
                "window_indices": _proposal_window_indices(proposal),
            }
        )
    if len(span_proposals) < 2:
        return None
    span_sorted = sorted(span_proposals, key=_window_subset_sort_key)
    best_subset = _best_non_overlapping_window_subset(span_sorted)
    if len(best_subset) < 2:
        return None
    execution_sorted = sorted(
        best_subset,
        key=lambda item: (
            -int(item["current_start"]),
            -int(item["target_start"]),
            -int(item["current_end"]),
            -int(item["target_end"]),
            int(item["window_index"]),
        ),
    )
    steps = [dict(item["step"]) for item in execution_sorted]
    command = _render_structured_edit_command(
        {
            "path": path,
            "edit_kind": "multi_edit",
            "steps": steps,
        }
    )
    if not command:
        return None
    covered_window_indices = sorted(
        {
            _int_value(index, -1)
            for item in best_subset
            for index in (item.get("proposal") or {}).get("covered_window_indices", [])
            if _int_value(index, -1) >= 0
        }
    )
    component_alias_pair_counts: dict[int, int] = {}
    component_unresolved_alias_pair_counts: dict[int, int] = {}
    component_conflicting_alias_pair_counts: dict[int, int] = {}
    component_candidate_counts: dict[int, int] = {}
    component_frontier_gaps: dict[int, float] = {}
    for item in best_subset:
        proposal = item.get("proposal") or {}
        component_id = _int_value(proposal.get("overlap_component_id"), -1)
        if component_id < 0:
            continue
        component_alias_pair_counts[component_id] = max(
            component_alias_pair_counts.get(component_id, 0),
            max(0, _int_value(proposal.get("overlap_component_alias_pair_count"), 0)),
        )
        component_unresolved_alias_pair_counts[component_id] = max(
            component_unresolved_alias_pair_counts.get(component_id, 0),
            max(0, _int_value(proposal.get("overlap_component_unresolved_alias_pair_count"), 0)),
        )
        component_conflicting_alias_pair_counts[component_id] = max(
            component_conflicting_alias_pair_counts.get(component_id, 0),
            max(0, _int_value(proposal.get("overlap_component_conflicting_alias_pair_count"), 0)),
        )
        component_candidate_counts[component_id] = max(
            component_candidate_counts.get(component_id, 0),
            max(0, _int_value(proposal.get("overlap_component_candidate_count"), 0)),
        )
        component_frontier_gaps[component_id] = max(
            component_frontier_gaps.get(component_id, 0.0),
            max(0.0, _float_value(proposal.get("overlap_component_frontier_gap"), 0.0)),
        )
    window_indices = sorted(
        {
            _int_value(index, -1)
            for item in best_subset
            for index in item.get("window_indices", [])
            if _int_value(index, -1) >= 0
        }
    )
    return {
        "command": command,
        "proposal_source": "structured_edit:multi_edit",
        "edit_kind": "multi_edit",
        "edit_source": (
            "workspace_preview_range"
            if any(str(item["proposal"].get("edit_source", "")).strip() == "workspace_preview_range" for item in best_subset)
            else "workspace_preview"
        ),
        "window_index": None,
        "window_indices": window_indices,
        "edit_window_count": len(window_indices) if window_indices else len(best_subset),
        "covered_window_indices": covered_window_indices,
        "covered_window_count": len(covered_window_indices),
        "available_window_count": max(
            _int_value((item["proposal"] or {}).get("available_window_count"), len(window_indices) or len(best_subset))
            for item in best_subset
        ),
        "retained_window_count": max(
            _int_value((item["proposal"] or {}).get("retained_window_count"), len(span_proposals))
            for item in best_subset
        ),
        "total_window_count": max(
            _int_value((item["proposal"] or {}).get("total_window_count"), len(span_proposals))
            for item in best_subset
        ),
        "partial_window_coverage": any(
            bool((item["proposal"] or {}).get("partial_window_coverage", False))
            for item in best_subset
        ),
        "explicit_current_span_proof": any(
            bool((item.get("proposal") or {}).get("explicit_current_span_proof", False))
            for item in best_subset
        ),
        "component_edit_kinds": [
            str(item["proposal"].get("edit_kind", "")).strip() or "rewrite"
            for item in best_subset
        ],
        "component_line_deltas": [
            _int_value((item["step"] or {}).get("line_delta"), 0)
            for item in best_subset
        ],
        "precision_penalty": sum(
            _int_value((item.get("proposal") or {}).get("precision_penalty"), 0)
            for item in best_subset
        ),
        "exact_window_count": sum(
            max(
                0,
                _int_value(
                    (item.get("proposal") or {}).get("exact_window_count"),
                    1 if bool((item.get("proposal") or {}).get("exact_target_span", False)) else 0,
                ),
            )
            for item in best_subset
        ),
        "inexact_window_count": sum(
            max(
                0,
                _int_value(
                    (item.get("proposal") or {}).get("inexact_window_count"),
                    0 if bool((item.get("proposal") or {}).get("exact_target_span", False)) else 1,
                ),
            )
            for item in best_subset
        ),
        "structural_window_count": sum(
            max(
                0,
                _int_value(
                    (item.get("proposal") or {}).get("structural_window_count"),
                    1 if _int_value((item["step"] or {}).get("line_delta"), 0) != 0 else 0,
                ),
            )
            for item in best_subset
        ),
        "overlap_alias_pair_count": sum(
            max(0, _int_value((item.get("proposal") or {}).get("overlap_alias_pair_count"), 0))
            for item in best_subset
        ),
        "overlap_alias_window_count": sum(
            max(0, _int_value((item.get("proposal") or {}).get("overlap_alias_window_count"), 0))
            for item in best_subset
        ),
        "safe_inexact_window_count": sum(
            max(0, _int_value((item.get("proposal") or {}).get("safe_inexact_window_count"), 0))
            for item in best_subset
        ),
        "recovered_conflicting_alias": any(
            bool((item.get("proposal") or {}).get("recovered_conflicting_alias", False))
            for item in best_subset
        ),
        "overlap_component_alias_pair_count": sum(component_alias_pair_counts.values()),
        "overlap_component_unresolved_alias_pair_count": sum(
            component_unresolved_alias_pair_counts.values()
        ),
        "overlap_component_conflicting_alias_pair_count": sum(
            component_conflicting_alias_pair_counts.values()
        ),
        "overlap_component_candidate_count": sum(component_candidate_counts.values()),
        "overlap_component_frontier_gap": sum(component_frontier_gaps.values()),
        "span_line_count": sum(
            max(
                _int_value((item.get("proposal") or {}).get("covered_window_count"), 1),
                _int_value(
                    (item.get("proposal") or {}).get("span_line_count"),
                    _int_value((item.get("proposal") or {}).get("covered_window_count"), 1),
                ),
            )
            for item in best_subset
        ),
        "step": {
            "path": path,
            "edit_kind": "multi_edit",
            "steps": steps,
            "edit_kinds": [
                str(item["proposal"].get("edit_kind", "")).strip() or "rewrite"
                for item in best_subset
            ],
            "edit_score": sum(
                int((item["proposal"].get("step") or {}).get("edit_score", 0))
                for item in best_subset
            ),
        },
    }


def _workspace_preview_risk_summary(
    *,
    path: str,
    preview_windows: list[dict[str, object]],
    available_window_count: int,
    proof_regions: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    span_windows: list[dict[str, object]] = []
    for preview_window in preview_windows:
        current_span = _preview_window_current_span(preview_window)
        target_span = _preview_window_target_span(preview_window)
        if current_span is None or target_span is None:
            continue
        span_windows.append(
            {
                "window": preview_window,
                "current_start": current_span[0],
                "current_end": current_span[1],
                "target_start": target_span[0],
                "target_end": target_span[1],
                "window_index": _int_value(preview_window.get("window_index"), 0),
            }
        )
    risk_summary = {
        "overlap_component_unrecoverable_count": 0,
        "overlap_component_unrecoverable_window_count": 0,
        "hidden_gap_broader_region_required": False,
        "hidden_gap_line_count": 0,
        "hidden_gap_current_partial_proof_count": 0,
        "hidden_gap_current_partial_proof_missing_line_count": 0,
        "hidden_gap_current_partial_proof_span_count": 0,
        "hidden_gap_current_partial_proof_opaque_spans": [],
        "hidden_gap_current_partial_proof_topologies": [],
        "hidden_gap_current_partial_proof_admissible_opaque_span_count": 0,
        "hidden_gap_current_partial_proof_blocking_opaque_span_count": 0,
        "hidden_gap_current_partial_proof_coarse_opaque_span_count": 0,
        "hidden_gap_current_partial_proof_mixed_opaque_topology": False,
    }
    if len(span_windows) < 2:
        return risk_summary
    for component_id, component_windows in enumerate(_overlap_window_components(span_windows)):
        if len(component_windows) < 2:
            continue
        if _preview_overlap_alias_pair_count(component_windows) <= 0:
            continue
        component_candidates = _overlap_block_replace_candidates_for_component(
            path=path,
            preview_windows=preview_windows,
            component_windows=component_windows,
            available_window_count=available_window_count,
            component_id=component_id,
        )
        if component_candidates:
            continue
        risk_summary["overlap_component_unrecoverable_count"] = _int_value(
            risk_summary.get("overlap_component_unrecoverable_count"),
            0,
        ) + 1
        risk_summary["overlap_component_unrecoverable_window_count"] = _int_value(
            risk_summary.get("overlap_component_unrecoverable_window_count"),
            0,
        ) + len(
            {
                _int_value(item.get("window_index"), -1)
                for item in component_windows
                if _int_value(item.get("window_index"), -1) >= 0
            }
        )
    hidden_gap_line_count = _workspace_preview_hidden_gap_line_count(span_windows)
    if hidden_gap_line_count > 0 and any(
        bool(window.get("partial_window_coverage", False))
        for window in preview_windows
    ):
        risk_summary["hidden_gap_broader_region_required"] = True
        risk_summary["hidden_gap_line_count"] = hidden_gap_line_count
    for proof_region in proof_regions or []:
        if not isinstance(proof_region, dict):
            continue
        if not bool(proof_region.get("current_proof_partial_coverage", False)):
            continue
        risk_summary["hidden_gap_current_partial_proof_count"] = _int_value(
            risk_summary.get("hidden_gap_current_partial_proof_count"),
            0,
        ) + 1
        risk_summary["hidden_gap_current_partial_proof_missing_line_count"] = max(
            _int_value(risk_summary.get("hidden_gap_current_partial_proof_missing_line_count"), 0),
            max(0, _int_value(proof_region.get("current_proof_missing_line_count"), 0)),
        )
        risk_summary["hidden_gap_current_partial_proof_span_count"] = max(
            _int_value(risk_summary.get("hidden_gap_current_partial_proof_span_count"), 0),
            max(0, _int_value(proof_region.get("current_proof_missing_span_count"), 0)),
        )
        region_current_start = max(1, _int_value(proof_region.get("line_start"), 1))
        region_current_end = max(
            region_current_start - 1,
            _int_value(proof_region.get("line_end"), region_current_start - 1),
        )
        region_target_start = max(1, _int_value(proof_region.get("target_line_start"), 1))
        region_target_end = max(
            region_target_start - 1,
            _int_value(proof_region.get("target_line_end"), region_target_start - 1),
        )
        opaque_metrics = _current_proof_region_opaque_span_metrics(
            proof_region,
            region_current_start=region_current_start,
            region_current_end=region_current_end,
            region_target_start=region_target_start,
            region_target_end=region_target_end,
        )
        risk_summary["hidden_gap_current_partial_proof_admissible_opaque_span_count"] = max(
            _int_value(
                risk_summary.get("hidden_gap_current_partial_proof_admissible_opaque_span_count"),
                0,
            ),
            max(0, _int_value(opaque_metrics.get("admissible_opaque_span_count"), 0)),
        )
        risk_summary["hidden_gap_current_partial_proof_blocking_opaque_span_count"] = max(
            _int_value(
                risk_summary.get("hidden_gap_current_partial_proof_blocking_opaque_span_count"),
                0,
            ),
            max(0, _int_value(opaque_metrics.get("blocking_opaque_span_count"), 0)),
        )
        risk_summary["hidden_gap_current_partial_proof_coarse_opaque_span_count"] = max(
            _int_value(
                risk_summary.get("hidden_gap_current_partial_proof_coarse_opaque_span_count"),
                0,
            ),
            max(0, _int_value(opaque_metrics.get("coarse_internal_opaque_span_count"), 0)),
        )
        risk_summary["hidden_gap_current_partial_proof_mixed_opaque_topology"] = bool(
            risk_summary.get("hidden_gap_current_partial_proof_mixed_opaque_topology", False)
        ) or bool(opaque_metrics.get("mixed_opaque_topology", False))
        existing_topologies = risk_summary.get("hidden_gap_current_partial_proof_topologies", [])
        if not isinstance(existing_topologies, list):
            existing_topologies = []
        opaque_topology = str(opaque_metrics.get("opaque_topology", "")).strip()
        if opaque_topology and opaque_topology not in existing_topologies:
            existing_topologies.append(opaque_topology)
        risk_summary["hidden_gap_current_partial_proof_topologies"] = existing_topologies
        existing_opaque_spans = risk_summary.get("hidden_gap_current_partial_proof_opaque_spans", [])
        if not isinstance(existing_opaque_spans, list):
            existing_opaque_spans = []
        raw_opaque_spans = proof_region.get("current_proof_opaque_spans", [])
        if isinstance(raw_opaque_spans, list):
            seen_spans = {
                (
                    _int_value(span.get("current_line_start"), 0),
                    _int_value(span.get("current_line_end"), -1),
                    _int_value(span.get("target_line_start"), 0),
                    _int_value(span.get("target_line_end"), -1),
                    str(span.get("reason", "")).strip(),
                )
                for span in existing_opaque_spans
                if isinstance(span, dict)
            }
            for span in raw_opaque_spans:
                if not isinstance(span, dict):
                    continue
                signature = (
                    _int_value(span.get("current_line_start"), 0),
                    _int_value(span.get("current_line_end"), -1),
                    _int_value(span.get("target_line_start"), 0),
                    _int_value(span.get("target_line_end"), -1),
                    str(span.get("reason", "")).strip(),
                )
                if signature in seen_spans:
                    continue
                seen_spans.add(signature)
                existing_opaque_spans.append(
                    {
                        "current_line_start": signature[0],
                        "current_line_end": signature[1],
                        "target_line_start": signature[2],
                        "target_line_end": signature[3],
                        "reason": signature[4] or "unknown",
                    }
                )
        risk_summary["hidden_gap_current_partial_proof_opaque_spans"] = existing_opaque_spans
    return risk_summary


def _workspace_preview_hidden_gap_line_count(
    span_windows: list[dict[str, object]],
) -> int:
    if len(span_windows) < 2:
        return 0
    target_start = min(int(item["target_start"]) for item in span_windows)
    target_end = max(int(item["target_end"]) for item in span_windows)
    target_lines = _merge_preview_subset_line_map(
        span_windows,
        content_key="target_content",
        span_key="target",
    )
    if not target_lines:
        return 0
    hidden_target_line_count = sum(
        1 for line_number in range(target_start, target_end + 1)
        if line_number not in target_lines
    )
    if hidden_target_line_count > 0:
        return hidden_target_line_count
    current_start = min(int(item["current_start"]) for item in span_windows)
    current_end = max(int(item["current_end"]) for item in span_windows)
    current_lines = _merge_preview_subset_line_map(
        span_windows,
        content_key="baseline_content",
        span_key="current",
    )
    if not current_lines:
        return 0
    return sum(
        1 for line_number in range(current_start, current_end + 1)
        if line_number not in current_lines
    )


def _annotate_workspace_preview_risks(
    proposals: list[dict[str, object]],
    risk_summary: dict[str, object],
) -> None:
    if not proposals:
        return
    overlap_component_unrecoverable_count = max(
        0,
        _int_value(risk_summary.get("overlap_component_unrecoverable_count"), 0),
    )
    overlap_component_unrecoverable_window_count = max(
        0,
        _int_value(risk_summary.get("overlap_component_unrecoverable_window_count"), 0),
    )
    hidden_gap_broader_region_required = bool(
        risk_summary.get("hidden_gap_broader_region_required", False)
    )
    hidden_gap_line_count = max(
        0,
        _int_value(risk_summary.get("hidden_gap_line_count"), 0),
    )
    hidden_gap_current_partial_proof_count = max(
        0,
        _int_value(risk_summary.get("hidden_gap_current_partial_proof_count"), 0),
    )
    hidden_gap_current_partial_proof_missing_line_count = max(
        0,
        _int_value(risk_summary.get("hidden_gap_current_partial_proof_missing_line_count"), 0),
    )
    hidden_gap_current_partial_proof_span_count = max(
        0,
        _int_value(risk_summary.get("hidden_gap_current_partial_proof_span_count"), 0),
    )
    hidden_gap_current_partial_proof_opaque_spans = risk_summary.get(
        "hidden_gap_current_partial_proof_opaque_spans", []
    )
    if not isinstance(hidden_gap_current_partial_proof_opaque_spans, list):
        hidden_gap_current_partial_proof_opaque_spans = []
    hidden_gap_current_partial_proof_topologies = risk_summary.get(
        "hidden_gap_current_partial_proof_topologies", []
    )
    if not isinstance(hidden_gap_current_partial_proof_topologies, list):
        hidden_gap_current_partial_proof_topologies = []
    hidden_gap_current_partial_proof_admissible_opaque_span_count = max(
        0,
        _int_value(
            risk_summary.get("hidden_gap_current_partial_proof_admissible_opaque_span_count"),
            0,
        ),
    )
    hidden_gap_current_partial_proof_blocking_opaque_span_count = max(
        0,
        _int_value(
            risk_summary.get("hidden_gap_current_partial_proof_blocking_opaque_span_count"),
            0,
        ),
    )
    hidden_gap_current_partial_proof_coarse_opaque_span_count = max(
        0,
        _int_value(
            risk_summary.get("hidden_gap_current_partial_proof_coarse_opaque_span_count"),
            0,
        ),
    )
    hidden_gap_current_partial_proof_mixed_opaque_topology = bool(
        risk_summary.get("hidden_gap_current_partial_proof_mixed_opaque_topology", False)
    )
    for proposal in proposals:
        proposal["overlap_component_unrecoverable_count"] = (
            overlap_component_unrecoverable_count
        )
        proposal["overlap_component_unrecoverable_window_count"] = (
            overlap_component_unrecoverable_window_count
        )
        current_proof_region_block = bool(proposal.get("current_proof_region_block", False))
        proposal["hidden_gap_broader_region_required"] = (
            hidden_gap_broader_region_required and not current_proof_region_block
        )
        proposal["hidden_gap_line_count"] = (
            0 if current_proof_region_block else hidden_gap_line_count
        )
        proposal["hidden_gap_current_partial_proof_count"] = (
            hidden_gap_current_partial_proof_count
        )
        proposal["hidden_gap_current_partial_proof_missing_line_count"] = (
            hidden_gap_current_partial_proof_missing_line_count
        )
        proposal["hidden_gap_current_partial_proof_span_count"] = (
            hidden_gap_current_partial_proof_span_count
        )
        proposal["hidden_gap_current_partial_proof_opaque_spans"] = list(
            hidden_gap_current_partial_proof_opaque_spans
        )
        proposal["hidden_gap_current_partial_proof_topologies"] = list(
            hidden_gap_current_partial_proof_topologies
        )
        proposal["hidden_gap_current_partial_proof_admissible_opaque_span_count"] = (
            hidden_gap_current_partial_proof_admissible_opaque_span_count
        )
        proposal["hidden_gap_current_partial_proof_blocking_opaque_span_count"] = (
            hidden_gap_current_partial_proof_blocking_opaque_span_count
        )
        proposal["hidden_gap_current_partial_proof_coarse_opaque_span_count"] = (
            hidden_gap_current_partial_proof_coarse_opaque_span_count
        )
        proposal["hidden_gap_current_partial_proof_mixed_opaque_topology"] = (
            hidden_gap_current_partial_proof_mixed_opaque_topology
        )


def _annotate_hidden_gap_region_ambiguity(
    proposals: list[dict[str, object]],
) -> int:
    if not proposals:
        return 0
    for proposal in proposals:
        proposal["hidden_gap_region_ambiguity_count"] = 0
        proposal["hidden_gap_region_frontier_count"] = 0
        proposal["hidden_gap_region_frontier_gap"] = 0.0
        proposal["hidden_gap_bounded_alternative_count"] = 0
        proposal["hidden_gap_bounded_alternative_gap"] = 0.0
        proposal["hidden_gap_region_ambiguous"] = False
        proposal["hidden_gap_current_ambiguity_count"] = 0
        proposal["hidden_gap_current_frontier_count"] = 0
        proposal["hidden_gap_current_frontier_gap"] = 0.0
        proposal["hidden_gap_current_ambiguous"] = False
        proposal["hidden_gap_current_target_equivalent_count"] = 0
        proposal["hidden_gap_current_target_equivalent_gap"] = 0.0
        proposal["hidden_gap_current_target_equivalent_ambiguous"] = False
        proposal["hidden_gap_current_proof_required_count"] = 0
        proposal["hidden_gap_current_proof_required"] = False
    broader_region_candidates = [
        proposal
        for proposal in proposals
        if str(proposal.get("proposal_source", "")).strip() == "structured_edit:block_replace"
        and bool(proposal.get("exact_target_span", False))
        and (
            max(
                0,
                _int_value(proposal.get("hidden_gap_current_line_count"), 0),
                _int_value(proposal.get("hidden_gap_target_line_count"), 0),
            ) > 0
            or (
                _int_value(proposal.get("inexact_window_count"), 0) > 0
                and _int_value(proposal.get("covered_window_count"), 0) >= max(
                    1,
                    _int_value(
                        proposal.get("retained_window_count"),
                        proposal.get("available_window_count"),
                    ),
                )
            )
        )
    ]
    if not broader_region_candidates:
        return 0
    best_rank = min(
        _hidden_gap_region_ambiguity_rank_key(candidate)
        for candidate in broader_region_candidates
    )
    ambiguous_frontier = [
        candidate
        for candidate in broader_region_candidates
        if _hidden_gap_region_ambiguity_rank_key(candidate) == best_rank
    ]
    ambiguous_frontier = _canonicalize_hidden_gap_annotation_candidates(
        ambiguous_frontier,
        ambiguity_rank_key=_hidden_gap_region_ambiguity_rank_key,
    )
    distinct_commands = {
        str(candidate.get("command", "")).strip()
        for candidate in ambiguous_frontier
        if str(candidate.get("command", "")).strip()
    }
    ambiguity_count = len(distinct_commands)
    frontier_count = max(1, len(distinct_commands))
    frontier_max_covered_window_count = max(
        _int_value(candidate.get("covered_window_count"), 0)
        for candidate in ambiguous_frontier
    )
    frontier_max_exact_window_count = max(
        _int_value(candidate.get("exact_window_count"), 0)
        for candidate in ambiguous_frontier
    )
    frontier_min_inexact_window_count = min(
        _int_value(candidate.get("inexact_window_count"), 0)
        for candidate in ambiguous_frontier
    )
    frontier_min_hidden_gap_line_count = min(
        max(
            0,
            _int_value(candidate.get("hidden_gap_current_line_count"), 0),
            _int_value(candidate.get("hidden_gap_target_line_count"), 0),
        )
        for candidate in ambiguous_frontier
    )
    frontier_min_span_line_count = min(
        _int_value(candidate.get("span_line_count"), 0)
        for candidate in ambiguous_frontier
    )
    for proposal in proposals:
        proposal["hidden_gap_region_frontier_count"] = frontier_count
        proposal["hidden_gap_region_frontier_gap"] = _hidden_gap_region_frontier_gap(
            proposal,
            frontier_max_covered_window_count=frontier_max_covered_window_count,
            frontier_max_exact_window_count=frontier_max_exact_window_count,
            frontier_min_inexact_window_count=frontier_min_inexact_window_count,
            frontier_min_hidden_gap_line_count=frontier_min_hidden_gap_line_count,
            frontier_min_span_line_count=frontier_min_span_line_count,
        )
    if ambiguity_count > 1:
        _annotate_hidden_gap_bounded_alternatives(
            proposals,
            broader_region_candidates=broader_region_candidates,
        )
    if ambiguity_count <= 1:
        return 0
    for proposal in proposals:
        proposal["hidden_gap_region_ambiguity_count"] = ambiguity_count
        proposal["hidden_gap_region_ambiguous"] = True
    return ambiguity_count


def _annotate_hidden_gap_current_ambiguity(
    proposals: list[dict[str, object]],
) -> int:
    if not proposals:
        return 0
    current_dominant_candidates = [
        proposal
        for proposal in proposals
        if str(proposal.get("proposal_source", "")).strip() == "structured_edit:block_replace"
        and bool(proposal.get("exact_target_span", False))
        and not bool(proposal.get("explicit_hidden_gap_current_proof", False))
        and _int_value(proposal.get("hidden_gap_current_line_count"), 0)
        > _int_value(proposal.get("hidden_gap_target_line_count"), 0)
    ]
    if not current_dominant_candidates:
        return 0
    best_rank = min(
        _hidden_gap_current_ambiguity_rank_key(candidate)
        for candidate in current_dominant_candidates
    )
    ambiguous_frontier = [
        candidate
        for candidate in current_dominant_candidates
        if _hidden_gap_current_ambiguity_rank_key(candidate) == best_rank
    ]
    ambiguous_frontier = _canonicalize_hidden_gap_annotation_candidates(
        ambiguous_frontier,
        ambiguity_rank_key=_hidden_gap_current_ambiguity_rank_key,
    )
    distinct_commands = {
        str(candidate.get("command", "")).strip()
        for candidate in ambiguous_frontier
        if str(candidate.get("command", "")).strip()
    }
    ambiguity_count = len(distinct_commands)
    frontier_count = max(1, len(distinct_commands))
    frontier_max_covered_window_count = max(
        _int_value(candidate.get("covered_window_count"), 0)
        for candidate in ambiguous_frontier
    )
    frontier_max_exact_window_count = max(
        _int_value(candidate.get("exact_window_count"), 0)
        for candidate in ambiguous_frontier
    )
    frontier_min_inexact_window_count = min(
        _int_value(candidate.get("inexact_window_count"), 0)
        for candidate in ambiguous_frontier
    )
    frontier_min_hidden_gap_target_line_count = min(
        max(0, _int_value(candidate.get("hidden_gap_target_line_count"), 0))
        for candidate in ambiguous_frontier
    )
    frontier_min_target_span_line_count = min(
        max(
            0,
            _int_value(
                candidate.get("target_span_line_count"),
                _int_value(candidate.get("span_line_count"), 0),
            ),
        )
        for candidate in ambiguous_frontier
    )
    for proposal in proposals:
        proposal["hidden_gap_current_frontier_count"] = frontier_count
        proposal["hidden_gap_current_frontier_gap"] = _hidden_gap_current_frontier_gap(
            proposal,
            frontier_max_covered_window_count=frontier_max_covered_window_count,
            frontier_max_exact_window_count=frontier_max_exact_window_count,
            frontier_min_inexact_window_count=frontier_min_inexact_window_count,
            frontier_min_hidden_gap_target_line_count=frontier_min_hidden_gap_target_line_count,
            frontier_min_target_span_line_count=frontier_min_target_span_line_count,
        )
    if ambiguity_count <= 1:
        return 0
    for proposal in proposals:
        proposal["hidden_gap_current_ambiguity_count"] = ambiguity_count
        proposal["hidden_gap_current_ambiguous"] = True
    return ambiguity_count


def _annotate_hidden_gap_current_proof_requirement(
    proposals: list[dict[str, object]],
) -> int:
    if not proposals:
        return 0
    proof_required_candidates = [
        proposal
        for proposal in proposals
        if str(proposal.get("proposal_source", "")).strip() == "structured_edit:block_replace"
        and bool(proposal.get("exact_target_span", False))
        and not bool(proposal.get("explicit_hidden_gap_current_proof", False))
        and not bool(proposal.get("bridged_hidden_gap_region_block", False))
        and not bool(proposal.get("exact_hidden_gap_region_block", False))
        and _int_value(proposal.get("hidden_gap_current_line_count"), 0)
        > _int_value(proposal.get("hidden_gap_target_line_count"), 0)
        and _int_value(proposal.get("inexact_window_count"), 0) > 0
    ]
    if not proof_required_candidates:
        return 0
    best_rank = min(
        _hidden_gap_current_ambiguity_rank_key(candidate)
        for candidate in proof_required_candidates
    )
    proof_required_frontier = [
        candidate
        for candidate in proof_required_candidates
        if _hidden_gap_current_ambiguity_rank_key(candidate) == best_rank
    ]
    proof_required_frontier = _canonicalize_hidden_gap_annotation_candidates(
        proof_required_frontier,
        ambiguity_rank_key=_hidden_gap_current_ambiguity_rank_key,
    )
    distinct_commands = {
        str(candidate.get("command", "")).strip()
        for candidate in proof_required_frontier
        if str(candidate.get("command", "")).strip()
    }
    if len(distinct_commands) != 1:
        return 0
    for proposal in proposals:
        proposal["hidden_gap_current_proof_required_count"] = 1
        proposal["hidden_gap_current_proof_required"] = True
    return 1


def _annotate_hidden_gap_current_target_equivalent_ties(
    proposals: list[dict[str, object]],
) -> int:
    if not proposals:
        return 0
    grouped_candidates: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for proposal in proposals:
        if str(proposal.get("proposal_source", "")).strip() != "structured_edit:block_replace":
            continue
        if not bool(proposal.get("exact_target_span", False)):
            continue
        if bool(proposal.get("explicit_hidden_gap_current_proof", False)):
            continue
        if bool(proposal.get("bridged_hidden_gap_region_block", False)):
            continue
        if bool(proposal.get("exact_hidden_gap_region_block", False)):
            continue
        if _int_value(proposal.get("hidden_gap_current_line_count"), 0) <= _int_value(
            proposal.get("hidden_gap_target_line_count"),
            0,
        ):
            continue
        signature = _hidden_gap_current_target_equivalent_signature(proposal)
        if signature is None:
            continue
        grouped_candidates.setdefault(signature, []).append(proposal)
    if not grouped_candidates:
        return 0
    max_group_count = 0
    for group in grouped_candidates.values():
        canonical_group = _canonicalize_hidden_gap_annotation_candidates(
            group,
            ambiguity_rank_key=_hidden_gap_current_ambiguity_rank_key,
        )
        distinct_commands = {
            str(candidate.get("command", "")).strip()
            for candidate in canonical_group
            if str(candidate.get("command", "")).strip()
        }
        target_equivalent_count = len(distinct_commands)
        if target_equivalent_count <= 1:
            continue
        max_group_count = max(max_group_count, target_equivalent_count)
        frontier_min_hidden_gap_current_line_count = min(
            max(0, _int_value(candidate.get("hidden_gap_current_line_count"), 0))
            for candidate in canonical_group
        )
        frontier_min_current_span_line_count = min(
            max(
                0,
                _int_value(
                    candidate.get("current_span_line_count"),
                    _int_value(candidate.get("span_line_count"), 0),
                ),
            )
            for candidate in canonical_group
        )
        frontier_min_inexact_window_count = min(
            max(0, _int_value(candidate.get("inexact_window_count"), 0))
            for candidate in canonical_group
        )
        for candidate in group:
            candidate["hidden_gap_current_target_equivalent_count"] = target_equivalent_count
            candidate["hidden_gap_current_target_equivalent_gap"] = (
                _hidden_gap_current_target_equivalent_gap(
                    candidate,
                    frontier_min_hidden_gap_current_line_count=frontier_min_hidden_gap_current_line_count,
                    frontier_min_current_span_line_count=frontier_min_current_span_line_count,
                    frontier_min_inexact_window_count=frontier_min_inexact_window_count,
                )
            )
            candidate["hidden_gap_current_target_equivalent_ambiguous"] = True
    return max_group_count


def _hidden_gap_region_ambiguity_rank_key(
    candidate: dict[str, object],
) -> tuple[int, int, int, int, int]:
    hidden_gap_line_count = max(
        0,
        _int_value(candidate.get("hidden_gap_current_line_count"), 0),
        _int_value(candidate.get("hidden_gap_target_line_count"), 0),
    )
    return (
        -_int_value(candidate.get("covered_window_count"), 0),
        -_int_value(candidate.get("exact_window_count"), 0),
        _int_value(candidate.get("inexact_window_count"), 0),
        hidden_gap_line_count,
        _int_value(candidate.get("span_line_count"), 0),
    )


def _hidden_gap_current_ambiguity_rank_key(
    candidate: dict[str, object],
) -> tuple[int, int, int, int, int]:
    return (
        -_int_value(candidate.get("covered_window_count"), 0),
        -_int_value(candidate.get("exact_window_count"), 0),
        _int_value(candidate.get("inexact_window_count"), 0),
        max(0, _int_value(candidate.get("hidden_gap_target_line_count"), 0)),
        max(
            0,
            _int_value(
                candidate.get("target_span_line_count"),
                _int_value(candidate.get("span_line_count"), 0),
            ),
        ),
    )


def _hidden_gap_current_target_equivalent_signature(
    proposal: dict[str, object],
) -> tuple[object, ...] | None:
    step = proposal.get("step")
    if not isinstance(step, dict):
        return None
    target_span = _step_target_span(step)
    if target_span is None:
        return None
    replacement = step.get("replacement")
    replacement_content = ""
    if isinstance(replacement, dict):
        replacement_content = str(replacement.get("content", ""))
    return (
        str(step.get("path", "")).strip(),
        int(target_span[0]),
        int(target_span[1]),
        replacement_content,
    )


def _canonicalize_hidden_gap_annotation_candidates(
    candidates: list[dict[str, object]],
    *,
    ambiguity_rank_key,
) -> list[dict[str, object]]:
    if len(candidates) < 2:
        return candidates
    same_span_groups: dict[
        tuple[int, int, int, int],
        list[tuple[int, dict[str, object]]],
    ] = {}
    passthrough_indices: set[int] = set()
    for index, candidate in enumerate(candidates):
        span_signature = _preview_block_proposal_span_signature(candidate)
        if span_signature is None:
            passthrough_indices.add(index)
            continue
        same_span_groups.setdefault(span_signature, []).append((index, candidate))
    if not same_span_groups:
        return candidates
    selected_indices = set(passthrough_indices)
    for span_candidates in same_span_groups.values():
        if len(span_candidates) == 1:
            selected_indices.add(span_candidates[0][0])
            continue
        best_index, _ = min(
            span_candidates,
            key=lambda item: (
                _same_span_preview_block_proposal_rank_key(item[1]),
                ambiguity_rank_key(item[1]),
                item[0],
            ),
        )
        selected_indices.add(best_index)
    if len(selected_indices) >= len(candidates):
        return candidates
    return [
        candidate
        for index, candidate in enumerate(candidates)
        if index in selected_indices
    ]


def _hidden_gap_region_frontier_gap(
    proposal: dict[str, object],
    *,
    frontier_max_covered_window_count: int,
    frontier_max_exact_window_count: int,
    frontier_min_inexact_window_count: int,
    frontier_min_hidden_gap_line_count: int,
    frontier_min_span_line_count: int,
) -> float:
    covered_window_count = max(0, _int_value(proposal.get("covered_window_count"), 0))
    exact_window_count = max(0, _int_value(proposal.get("exact_window_count"), 0))
    inexact_window_count = max(0, _int_value(proposal.get("inexact_window_count"), 0))
    hidden_gap_line_count = max(
        0,
        _int_value(proposal.get("hidden_gap_current_line_count"), 0),
        _int_value(proposal.get("hidden_gap_target_line_count"), 0),
    )
    span_line_count = max(
        0,
        _int_value(
            proposal.get("span_line_count"),
            covered_window_count,
        ),
    )
    frontier_gap = 0.0
    frontier_gap += max(0, frontier_max_covered_window_count - covered_window_count) * 3.0
    frontier_gap += max(0, frontier_max_exact_window_count - exact_window_count) * 2.0
    frontier_gap += max(0, inexact_window_count - frontier_min_inexact_window_count) * 3.0
    if hidden_gap_line_count > 0:
        frontier_gap += max(0, hidden_gap_line_count - frontier_min_hidden_gap_line_count) * 1.5
        frontier_gap += max(0, span_line_count - frontier_min_span_line_count) * 0.5
    else:
        frontier_gap += max(0, frontier_min_hidden_gap_line_count - hidden_gap_line_count) * 1.5
        frontier_gap += max(0, frontier_min_span_line_count - span_line_count) * 0.5
        edit_kind = str(proposal.get("edit_kind", "")).strip()
        if edit_kind == "multi_edit":
            frontier_gap += 1.0
        elif str(proposal.get("proposal_source", "")).strip() != "structured_edit:block_replace":
            frontier_gap += 0.5
    return frontier_gap


def _hidden_gap_current_frontier_gap(
    proposal: dict[str, object],
    *,
    frontier_max_covered_window_count: int,
    frontier_max_exact_window_count: int,
    frontier_min_inexact_window_count: int,
    frontier_min_hidden_gap_target_line_count: int,
    frontier_min_target_span_line_count: int,
) -> float:
    covered_window_count = max(0, _int_value(proposal.get("covered_window_count"), 0))
    exact_window_count = max(0, _int_value(proposal.get("exact_window_count"), 0))
    inexact_window_count = max(0, _int_value(proposal.get("inexact_window_count"), 0))
    hidden_gap_target_line_count = max(0, _int_value(proposal.get("hidden_gap_target_line_count"), 0))
    target_span_line_count = max(
        0,
        _int_value(
            proposal.get("target_span_line_count"),
            _int_value(proposal.get("span_line_count"), 0),
        ),
    )
    frontier_gap = 0.0
    frontier_gap += max(0, frontier_max_covered_window_count - covered_window_count) * 3.0
    frontier_gap += max(0, frontier_max_exact_window_count - exact_window_count) * 2.0
    frontier_gap += max(0, inexact_window_count - frontier_min_inexact_window_count) * 3.0
    frontier_gap += max(
        0,
        hidden_gap_target_line_count - frontier_min_hidden_gap_target_line_count,
    ) * 1.5
    frontier_gap += max(
        0,
        target_span_line_count - frontier_min_target_span_line_count,
    ) * 0.5
    if (
        _int_value(proposal.get("hidden_gap_current_line_count"), 0)
        <= _int_value(proposal.get("hidden_gap_target_line_count"), 0)
    ):
        frontier_gap += 0.5
    return frontier_gap


def _hidden_gap_current_target_equivalent_gap(
    proposal: dict[str, object],
    *,
    frontier_min_hidden_gap_current_line_count: int,
    frontier_min_current_span_line_count: int,
    frontier_min_inexact_window_count: int,
) -> float:
    hidden_gap_current_line_count = max(
        0,
        _int_value(proposal.get("hidden_gap_current_line_count"), 0),
    )
    current_span_line_count = max(
        0,
        _int_value(
            proposal.get("current_span_line_count"),
            _int_value(proposal.get("span_line_count"), 0),
        ),
    )
    inexact_window_count = max(0, _int_value(proposal.get("inexact_window_count"), 0))
    frontier_gap = 0.0
    frontier_gap += max(
        0,
        hidden_gap_current_line_count - frontier_min_hidden_gap_current_line_count,
    ) * 1.5
    frontier_gap += max(
        0,
        current_span_line_count - frontier_min_current_span_line_count,
    ) * 0.5
    frontier_gap += max(0, inexact_window_count - frontier_min_inexact_window_count) * 2.0
    return frontier_gap


def _annotate_hidden_gap_bounded_alternatives(
    proposals: list[dict[str, object]],
    *,
    broader_region_candidates: list[dict[str, object]],
) -> None:
    bounded_candidates = [
        proposal
        for proposal in proposals
        if str(proposal.get("proposal_source", "")).strip().startswith("structured_edit:")
        and str(proposal.get("edit_source", "")).strip()
        in {"workspace_preview", "workspace_preview_range"}
        and bool(proposal.get("exact_target_span", False))
        and max(
            0,
            _int_value(proposal.get("hidden_gap_current_line_count"), 0),
            _int_value(proposal.get("hidden_gap_target_line_count"), 0),
        )
        <= 0
        and _int_value(proposal.get("covered_window_count"), 0) > 0
        and not bool(proposal.get("partial_window_coverage", False))
    ]
    if not bounded_candidates:
        return
    best_rank = min(
        _hidden_gap_bounded_alternative_rank_key(candidate)
        for candidate in bounded_candidates
    )
    bounded_frontier = [
        candidate
        for candidate in bounded_candidates
        if _hidden_gap_bounded_alternative_rank_key(candidate) == best_rank
    ]
    bounded_frontier = _canonicalize_hidden_gap_annotation_candidates(
        bounded_frontier,
        ambiguity_rank_key=_hidden_gap_bounded_alternative_rank_key,
    )
    distinct_commands = {
        str(candidate.get("command", "")).strip()
        for candidate in bounded_frontier
        if str(candidate.get("command", "")).strip()
    }
    bounded_count = max(1, len(distinct_commands))
    frontier_max_covered_window_count = max(
        _int_value(candidate.get("covered_window_count"), 0)
        for candidate in bounded_frontier
    )
    frontier_max_exact_window_count = max(
        _int_value(candidate.get("exact_window_count"), 0)
        for candidate in bounded_frontier
    )
    frontier_min_inexact_window_count = min(
        _int_value(candidate.get("inexact_window_count"), 0)
        for candidate in bounded_frontier
    )
    frontier_min_unresolved_alias_pair_count = min(
        _int_value(candidate.get("overlap_component_unresolved_alias_pair_count"), 0)
        for candidate in bounded_frontier
    )
    frontier_min_precision_penalty = min(
        _int_value(candidate.get("precision_penalty"), 0)
        for candidate in bounded_frontier
    )
    frontier_min_span_line_count = min(
        _int_value(candidate.get("span_line_count"), 0)
        for candidate in bounded_frontier
    )
    for proposal in broader_region_candidates:
        proposal["hidden_gap_bounded_alternative_count"] = bounded_count
        proposal["hidden_gap_bounded_alternative_gap"] = _hidden_gap_bounded_alternative_gap(
            proposal,
            frontier_max_covered_window_count=frontier_max_covered_window_count,
            frontier_max_exact_window_count=frontier_max_exact_window_count,
            frontier_min_inexact_window_count=frontier_min_inexact_window_count,
            frontier_min_unresolved_alias_pair_count=frontier_min_unresolved_alias_pair_count,
            frontier_min_precision_penalty=frontier_min_precision_penalty,
            frontier_min_span_line_count=frontier_min_span_line_count,
        )


def _hidden_gap_bounded_alternative_rank_key(
    candidate: dict[str, object],
) -> tuple[int, int, int, int, int, int, int]:
    edit_kind = str(candidate.get("edit_kind", "")).strip()
    edit_kind_priority = 0 if edit_kind == "block_replace" else 1 if edit_kind == "multi_edit" else 2
    return (
        -_int_value(candidate.get("covered_window_count"), 0),
        -_int_value(candidate.get("exact_window_count"), 0),
        _int_value(candidate.get("inexact_window_count"), 0),
        _int_value(candidate.get("overlap_component_unresolved_alias_pair_count"), 0),
        _int_value(candidate.get("precision_penalty"), 0),
        edit_kind_priority,
        _int_value(candidate.get("span_line_count"), 0),
    )


def _hidden_gap_bounded_alternative_gap(
    proposal: dict[str, object],
    *,
    frontier_max_covered_window_count: int,
    frontier_max_exact_window_count: int,
    frontier_min_inexact_window_count: int,
    frontier_min_unresolved_alias_pair_count: int,
    frontier_min_precision_penalty: int,
    frontier_min_span_line_count: int,
) -> float:
    covered_window_count = max(0, _int_value(proposal.get("covered_window_count"), 0))
    exact_window_count = max(0, _int_value(proposal.get("exact_window_count"), 0))
    inexact_window_count = max(0, _int_value(proposal.get("inexact_window_count"), 0))
    unresolved_alias_pair_count = max(
        0,
        _int_value(proposal.get("overlap_component_unresolved_alias_pair_count"), 0),
    )
    precision_penalty = max(0, _int_value(proposal.get("precision_penalty"), 0))
    span_line_count = max(
        0,
        _int_value(
            proposal.get("span_line_count"),
            covered_window_count,
        ),
    )
    alternative_gap = 0.0
    alternative_gap += max(0, covered_window_count - frontier_max_covered_window_count) * 3.0
    alternative_gap += max(0, exact_window_count - frontier_max_exact_window_count) * 2.0
    alternative_gap += max(0, frontier_min_inexact_window_count - inexact_window_count) * 2.0
    alternative_gap += max(
        0,
        frontier_min_unresolved_alias_pair_count - unresolved_alias_pair_count,
    ) * 2.0
    alternative_gap += max(0, frontier_min_precision_penalty - precision_penalty) * 1.5
    alternative_gap += max(0, span_line_count - frontier_min_span_line_count) * 0.5
    return alternative_gap


def _compose_workspace_preview_overlap_block_replace_proposals(
    *,
    path: str,
    preview_windows: list[dict[str, object]],
    available_window_count: int,
) -> list[dict[str, object]]:
    if len(preview_windows) < 2:
        return []
    span_windows: list[dict[str, object]] = []
    for preview_window in preview_windows:
        current_span = _preview_window_current_span(preview_window)
        target_span = _preview_window_target_span(preview_window)
        if current_span is None or target_span is None:
            continue
        span_windows.append(
            {
                "window": preview_window,
                "current_start": current_span[0],
                "current_end": current_span[1],
                "target_start": target_span[0],
                "target_end": target_span[1],
                "window_index": _int_value(preview_window.get("window_index"), 0),
            }
        )
    if len(span_windows) < 2:
        return []
    candidates: list[dict[str, object]] = []
    for component_id, component_windows in enumerate(_overlap_window_components(span_windows)):
        if len(component_windows) < 2:
            continue
        component_candidates = _overlap_block_replace_candidates_for_component(
            path=path,
            preview_windows=preview_windows,
            component_windows=component_windows,
            available_window_count=available_window_count,
            component_id=component_id,
        )
        candidates.extend(component_candidates)
    ranked = sorted(candidates, key=_overlap_block_replace_rank_key)
    deduped: list[dict[str, object]] = []
    seen_signatures: set[tuple[str, tuple[int, ...]]] = set()
    for candidate in ranked:
        signature = _overlap_block_replace_signature(candidate)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        deduped.append(candidate)
    return deduped


def _compose_workspace_preview_exact_region_block_replace_proposals(
    *,
    path: str,
    preview_windows: list[dict[str, object]],
    available_window_count: int,
) -> list[dict[str, object]]:
    exact_windows: list[dict[str, object]] = []
    for preview_window in preview_windows:
        if not bool(preview_window.get("exact_target_span", False)):
            continue
        current_span = _preview_window_current_span(preview_window)
        target_span = _preview_window_target_span(preview_window)
        if current_span is None or target_span is None:
            continue
        exact_windows.append(
            {
                "window": preview_window,
                "current_start": current_span[0],
                "current_end": current_span[1],
                "target_start": target_span[0],
                "target_end": target_span[1],
                "window_index": _int_value(preview_window.get("window_index"), 0),
            }
        )
    if len(exact_windows) < 2:
        return []
    ranked_windows = sorted(exact_windows, key=_window_subset_sort_key)
    candidates: list[dict[str, object]] = []
    for subset_size in range(2, len(ranked_windows) + 1):
        for subset in combinations(ranked_windows, subset_size):
            ordered_subset = sorted(subset, key=_window_subset_sort_key)
            overlapping_subset = _window_subset_has_overlap(ordered_subset)
            shared_anchor_exact_region_block = _shared_anchor_exact_region_subset_is_safe(
                ordered_subset
            )
            shared_anchor_pair_kinds = _shared_anchor_exact_pair_kinds(ordered_subset)
            shared_anchor_exact_neighbor_count = _shared_anchor_exact_neighbor_count(
                ordered_subset
            )
            shared_anchor_core_count = _shared_anchor_exact_core_count(ordered_subset)
            if overlapping_subset and not shared_anchor_exact_region_block:
                continue
            if _subset_skips_intervening_retained_window(ordered_subset):
                continue
            merged_window = None
            if not overlapping_subset:
                merged_window = _merge_preview_window_subset(ordered_subset)
            if merged_window is None and shared_anchor_exact_region_block:
                merged_window = _recover_exact_shared_anchor_preview_window(ordered_subset)
            exact_hidden_gap_region_block = False
            if merged_window is None:
                merged_window = _recover_exact_hidden_gap_preview_window(ordered_subset)
                exact_hidden_gap_region_block = merged_window is not None
            if merged_window is None:
                continue
            needs_preview_offset = True
            if (
                exact_hidden_gap_region_block
                or shared_anchor_exact_region_block
                or str(merged_window.get("target_content", "")) == ""
            ):
                step = _build_block_replace_step_for_preview_region(
                    path=path,
                    preview_window=merged_window,
                )
                needs_preview_offset = False
            else:
                step = _derive_block_replace_step(
                    path=path,
                    baseline_content=str(merged_window.get("baseline_content", "")),
                    target_content=str(merged_window.get("target_content", "")),
                )
            if step is None:
                continue
            if needs_preview_offset:
                step = _offset_preview_step(
                    step,
                    line_offset=_int_value(merged_window.get("line_offset"), 0),
                    target_line_offset=_int_value(merged_window.get("target_line_offset"), 0),
                )
            command = _render_structured_edit_command(step)
            if not command:
                continue
            window_indices = [int(item["window_index"]) for item in ordered_subset]
            current_span_line_count = max(0, int(merged_window.get("current_line_count", 0)))
            target_span_line_count = max(0, int(merged_window.get("target_line_count", 0)))
            candidates.append(
                {
                    "command": command,
                    "proposal_source": "structured_edit:block_replace",
                    "edit_kind": "block_replace",
                    "edit_source": (
                        "workspace_preview_range"
                        if any(bool((item.get("window") or {}).get("truncated", False)) for item in ordered_subset)
                        else "workspace_preview"
                    ),
                    "window_index": None,
                    "window_indices": window_indices,
                    "edit_window_count": len(ordered_subset),
                    "covered_window_indices": window_indices,
                    "covered_window_count": len(window_indices),
                    "available_window_count": available_window_count,
                    "retained_window_count": max(
                        _int_value((item.get("window") or {}).get("retained_window_count"), available_window_count)
                        for item in ordered_subset
                    ),
                    "total_window_count": max(
                        _int_value((item.get("window") or {}).get("total_window_count"), available_window_count)
                        for item in ordered_subset
                    ),
                    "partial_window_coverage": any(
                        bool((item.get("window") or {}).get("partial_window_coverage", False))
                        for item in ordered_subset
                    ),
                    "exact_target_span": True,
                    "precision_penalty": _preview_step_precision_penalty(
                        step,
                        exact_target_span=True,
                    ),
                    "exact_window_count": len(ordered_subset),
                    "inexact_window_count": 0,
                    "hidden_gap_current_line_count": _int_value(
                        merged_window.get("hidden_gap_current_line_count"),
                        0,
                    ),
                    "hidden_gap_target_line_count": _int_value(
                        merged_window.get("hidden_gap_target_line_count"),
                        0,
                    ),
                    "current_span_line_count": current_span_line_count,
                    "target_span_line_count": target_span_line_count,
                    "span_line_count": max(current_span_line_count, target_span_line_count),
                    "exact_contiguous_region_block": not exact_hidden_gap_region_block,
                    "exact_hidden_gap_region_block": exact_hidden_gap_region_block,
                    "shared_anchor_exact_region_block": shared_anchor_exact_region_block,
                    "shared_anchor_pair_kinds": shared_anchor_pair_kinds,
                    "shared_anchor_exact_neighbor_count": shared_anchor_exact_neighbor_count,
                    "shared_anchor_core_count": shared_anchor_core_count,
                    "shared_anchor_hybrid_component_count": (
                        1 if shared_anchor_exact_region_block and exact_hidden_gap_region_block else 0
                    ),
                    "shared_anchor_mixed_insert_delete": any(
                        pair_kind in {"same_anchor_insert_delete", "adjacent_delete_insert"}
                        for pair_kind in shared_anchor_pair_kinds
                    ),
                    "hidden_gap_target_from_expected_content": bool(
                        merged_window.get("hidden_gap_target_from_expected_content", False)
                    ),
                    "step": step,
                }
            )
    ranked = sorted(candidates, key=_overlap_block_replace_rank_key)
    deduped: list[dict[str, object]] = []
    seen_signatures: set[tuple[str, tuple[int, ...]]] = set()
    for candidate in ranked:
        signature = _overlap_block_replace_signature(candidate)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        deduped.append(candidate)
        if len(deduped) >= 3:
            break
    return deduped


def _compose_workspace_preview_exact_proof_block_replace_proposals(
    *,
    path: str,
    proof_windows: list[dict[str, object]],
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    seen_commands: set[str] = set()
    for proof_window in proof_windows:
        line_start = max(1, _int_value(proof_window.get("line_start"), 1))
        line_end = max(line_start - 1, _int_value(proof_window.get("line_end"), line_start - 1))
        target_start = max(1, _int_value(proof_window.get("target_line_start"), 1))
        target_end = max(target_start - 1, _int_value(proof_window.get("target_line_end"), target_start - 1))
        full_target_content = str(proof_window.get("full_target_content", ""))
        if target_end < target_start:
            replacement_content = ""
        else:
            target_lines = _expected_target_lines_for_span(
                full_target_content,
                start_line=target_start,
                end_line=target_end,
            )
            if target_lines is None:
                continue
            replacement_content = "".join(target_lines)
        step = {
            "path": path,
            "edit_kind": "block_replace",
            "current_start_line": line_start,
            "current_end_line": line_end,
            "target_start_line": target_start,
            "target_end_line": target_end,
            "replacement": {
                "start_line": line_start,
                "end_line": line_end,
                "content": replacement_content,
                "after_lines": replacement_content.splitlines(),
            },
        }
        command = _render_structured_edit_command(step)
        if not command or command in seen_commands:
            continue
        seen_commands.add(command)
        window_index = _int_value(proof_window.get("window_index"), -1)
        current_span_line_count = max(0, line_end - line_start + 1)
        target_span_line_count = max(0, target_end - target_start + 1)
        candidates.append(
            {
                "command": command,
                "proposal_source": "structured_edit:block_replace",
                "edit_kind": "block_replace",
                "edit_source": "workspace_preview_range",
                "window_index": window_index,
                "edit_window_count": 1,
                "covered_window_indices": [window_index] if window_index >= 0 else [],
                "covered_window_count": 1 if window_index >= 0 else 0,
                "exact_target_span": True,
                "precision_penalty": _preview_step_precision_penalty(step, exact_target_span=True),
                "exact_window_count": 1,
                "inexact_window_count": 0,
                "retained_window_count": _int_value(proof_window.get("retained_window_count"), 1),
                "total_window_count": _int_value(proof_window.get("total_window_count"), 1),
                "partial_window_coverage": bool(proof_window.get("partial_window_coverage", False)),
                "explicit_current_span_proof": True,
                "current_span_line_count": current_span_line_count,
                "target_span_line_count": target_span_line_count,
                "span_line_count": max(current_span_line_count, target_span_line_count),
                "step": step,
            }
        )
    return candidates


def _compose_workspace_preview_bridged_region_block_replace_proposals(
    *,
    path: str,
    bridge_windows: list[dict[str, object]],
    preview_windows: list[dict[str, object]],
    available_window_count: int,
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    seen_signatures: set[tuple[str, tuple[int, ...]]] = set()
    indexed_windows = {
        _int_value(window.get("window_index"), -1): window
        for window in preview_windows
        if _int_value(window.get("window_index"), -1) >= 0
    }
    direct_bridge_runs = [
        preview_window
        for preview_window in bridge_windows
        if isinstance(preview_window.get("bridge_segments"), list)
        and len(preview_window.get("bridge_segments", [])) > 0
    ]
    for preview_window in direct_bridge_runs:
        candidate = _build_explicit_bridged_hidden_gap_run_candidate(
            path=path,
            bridge_window=preview_window,
            indexed_windows=indexed_windows,
            available_window_count=available_window_count,
        )
        if candidate is None:
            continue
        signature = _overlap_block_replace_signature(candidate)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        candidates.append(candidate)
    bridge_by_pair: dict[tuple[int, int], dict[str, object]] = {}
    for preview_window in bridge_windows:
        if preview_window in direct_bridge_runs:
            continue
        bridge_window_indices = tuple(
            sorted(
                {
                    _int_value(index, -1)
                    for index in preview_window.get("bridge_window_indices", [])
                    if _int_value(index, -1) >= 0
                }
            )
        )
        if len(bridge_window_indices) == 2:
            bridge_by_pair[bridge_window_indices] = preview_window
    sorted_indices = sorted(indexed_windows)
    for start_position, start_index in enumerate(sorted_indices):
        subset_indices = [start_index]
        for next_index in sorted_indices[start_position + 1 :]:
            if next_index != subset_indices[-1] + 1:
                break
            if (subset_indices[-1], next_index) not in bridge_by_pair:
                break
            subset_indices.append(next_index)
            candidate = _build_bridged_hidden_gap_region_candidate(
                path=path,
                subset_indices=subset_indices,
                indexed_windows=indexed_windows,
                bridge_by_pair=bridge_by_pair,
                available_window_count=available_window_count,
            )
            if candidate is None:
                continue
            signature = _overlap_block_replace_signature(candidate)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            candidates.append(candidate)
    return sorted(candidates, key=_overlap_block_replace_rank_key)


def _compose_workspace_preview_current_proof_region_block_replace_proposals(
    *,
    path: str,
    proof_regions: list[dict[str, object]],
    preview_windows: list[dict[str, object]],
    available_window_count: int,
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    seen_signatures: set[tuple[str, tuple[int, ...]]] = set()
    indexed_windows = {
        _int_value(window.get("window_index"), -1): window
        for window in preview_windows
        if _int_value(window.get("window_index"), -1) >= 0
    }
    for proof_region in proof_regions:
        candidate_regions = [proof_region]
        candidate_regions.extend(_factorize_current_proof_region_subregions(proof_region))
        for candidate_region in candidate_regions:
            candidate = _build_current_proof_region_candidate(
                path=path,
                proof_region=candidate_region,
                indexed_windows=indexed_windows,
                available_window_count=available_window_count,
            )
            if candidate is None:
                continue
            signature = _overlap_block_replace_signature(candidate)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            candidates.append(candidate)
    return sorted(candidates, key=_overlap_block_replace_rank_key)


def _current_proof_region_opaque_span_metrics(
    proof_region: dict[str, object],
    *,
    region_current_start: int,
    region_current_end: int,
    region_target_start: int,
    region_target_end: int,
) -> dict[str, object]:
    raw_opaque_spans = proof_region.get("current_proof_opaque_spans", [])
    if not isinstance(raw_opaque_spans, list):
        raw_opaque_spans = []
    opaque_span_count = 0
    opaque_line_count = 0
    opaque_boundary_touch_count = 0
    opaque_internal_span_count = 0
    opaque_max_span_line_count = 0
    admissible_opaque_span_count = 0
    coarse_internal_opaque_span_count = 0
    for span in raw_opaque_spans:
        if not isinstance(span, dict):
            continue
        current_line_start = max(1, _int_value(span.get("current_line_start"), 1))
        current_line_end = max(
            current_line_start - 1,
            _int_value(span.get("current_line_end"), current_line_start - 1),
        )
        target_line_start = max(1, _int_value(span.get("target_line_start"), 1))
        target_line_end = max(
            target_line_start - 1,
            _int_value(span.get("target_line_end"), target_line_start - 1),
        )
        if current_line_end < current_line_start or target_line_end < target_line_start:
            continue
        opaque_span_count += 1
        span_line_count = max(0, current_line_end - current_line_start + 1)
        opaque_line_count += span_line_count
        opaque_max_span_line_count = max(opaque_max_span_line_count, span_line_count)
        touches_boundary = (
            current_line_start <= region_current_start
            or current_line_end >= region_current_end
            or target_line_start <= region_target_start
            or target_line_end >= region_target_end
        )
        if touches_boundary:
            opaque_boundary_touch_count += 1
        else:
            opaque_internal_span_count += 1
            if span_line_count <= 1:
                admissible_opaque_span_count += 1
            else:
                coarse_internal_opaque_span_count += 1
    blocking_opaque_span_count = max(0, opaque_span_count - admissible_opaque_span_count)
    mixed_opaque_topology = admissible_opaque_span_count > 0 and blocking_opaque_span_count > 0
    opaque_topology = ""
    if opaque_span_count > 0:
        if opaque_boundary_touch_count > 0 and admissible_opaque_span_count > 0:
            opaque_topology = "mixed_boundary_internal_opaque"
        elif opaque_boundary_touch_count > 0:
            opaque_topology = "boundary_touch_opaque"
        elif coarse_internal_opaque_span_count > 0 and admissible_opaque_span_count > 0:
            opaque_topology = "mixed_internal_opaque"
        elif coarse_internal_opaque_span_count > 0:
            opaque_topology = "coarse_internal_opaque"
        elif opaque_span_count == 1:
            opaque_topology = "single_internal_opaque"
        else:
            opaque_topology = "sparse_internal_multi_opaque"
    return {
        "opaque_span_count": opaque_span_count,
        "opaque_line_count": opaque_line_count,
        "opaque_boundary_touch_count": opaque_boundary_touch_count,
        "opaque_internal_span_count": opaque_internal_span_count,
        "opaque_max_span_line_count": opaque_max_span_line_count,
        "admissible_opaque_span_count": admissible_opaque_span_count,
        "blocking_opaque_span_count": blocking_opaque_span_count,
        "coarse_internal_opaque_span_count": coarse_internal_opaque_span_count,
        "mixed_opaque_topology": mixed_opaque_topology,
        "opaque_topology": opaque_topology,
    }


def _factorize_current_proof_region_subregions(
    proof_region: dict[str, object],
) -> list[dict[str, object]]:
    if not bool(proof_region.get("current_proof_partial_coverage", False)):
        return []
    region_current_start = max(1, _int_value(proof_region.get("line_start"), 1))
    region_current_end = max(
        region_current_start - 1,
        _int_value(proof_region.get("line_end"), region_current_start - 1),
    )
    region_target_start = max(1, _int_value(proof_region.get("target_line_start"), 1))
    region_target_end = max(
        region_target_start - 1,
        _int_value(proof_region.get("target_line_end"), region_target_start - 1),
    )
    opaque_metrics = _current_proof_region_opaque_span_metrics(
        proof_region,
        region_current_start=region_current_start,
        region_current_end=region_current_end,
        region_target_start=region_target_start,
        region_target_end=region_target_end,
    )
    if (
        _int_value(opaque_metrics.get("admissible_opaque_span_count"), 0) <= 0
        or _int_value(opaque_metrics.get("blocking_opaque_span_count"), 0) <= 0
    ):
        return []
    raw_proof_spans = proof_region.get("current_proof_spans", [])
    if not isinstance(raw_proof_spans, list):
        return []
    proof_spans: list[dict[str, object]] = []
    for span in raw_proof_spans:
        if not isinstance(span, dict):
            continue
        current_line_start = max(1, _int_value(span.get("current_line_start"), 1))
        current_line_end = max(
            current_line_start - 1,
            _int_value(span.get("current_line_end"), current_line_start - 1),
        )
        target_line_start = max(1, _int_value(span.get("target_line_start"), 1))
        target_line_end = max(
            target_line_start - 1,
            _int_value(span.get("target_line_end"), target_line_start - 1),
        )
        if current_line_end < current_line_start or target_line_end < target_line_start:
            continue
        proof_spans.append(
            {
                "kind": "proof",
                "current_line_start": current_line_start,
                "current_line_end": current_line_end,
                "target_line_start": target_line_start,
                "target_line_end": target_line_end,
                "current_content": str(span.get("current_content", "")),
                "target_content": str(span.get("target_content", "")),
                "current_from_line_span_proof": bool(
                    span.get("current_from_line_span_proof", False)
                ),
                "target_from_expected_content": bool(
                    span.get("target_from_expected_content", False)
                ),
            }
        )
    if len(proof_spans) < 2:
        return []
    raw_opaque_spans = proof_region.get("current_proof_opaque_spans", [])
    if not isinstance(raw_opaque_spans, list):
        return []
    opaque_entries: list[dict[str, object]] = []
    for span in raw_opaque_spans:
        if not isinstance(span, dict):
            continue
        current_line_start = max(1, _int_value(span.get("current_line_start"), 1))
        current_line_end = max(
            current_line_start - 1,
            _int_value(span.get("current_line_end"), current_line_start - 1),
        )
        target_line_start = max(1, _int_value(span.get("target_line_start"), 1))
        target_line_end = max(
            target_line_start - 1,
            _int_value(span.get("target_line_end"), target_line_start - 1),
        )
        if current_line_end < current_line_start or target_line_end < target_line_start:
            continue
        span_line_count = max(0, current_line_end - current_line_start + 1)
        touches_boundary = (
            current_line_start <= region_current_start
            or current_line_end >= region_current_end
            or target_line_start <= region_target_start
            or target_line_end >= region_target_end
        )
        opaque_entries.append(
            {
                "kind": "opaque",
                "current_line_start": current_line_start,
                "current_line_end": current_line_end,
                "target_line_start": target_line_start,
                "target_line_end": target_line_end,
                "reason": str(span.get("reason", "")).strip() or "unknown",
                "line_count": span_line_count,
                "touches_boundary": touches_boundary,
                "admissible": not touches_boundary and span_line_count <= 1,
            }
        )
    if not opaque_entries:
        return []
    ordered_entries = sorted(
        [*proof_spans, *opaque_entries],
        key=lambda item: (
            _int_value(item.get("current_line_start"), 0),
            _int_value(item.get("current_line_end"), 0),
            0 if str(item.get("kind", "")) == "proof" else 1,
        ),
    )
    factorized_regions: list[dict[str, object]] = []
    chain_proof_spans: list[dict[str, object]] = []
    chain_opaque_spans: list[dict[str, object]] = []

    def flush_chain() -> None:
        if len(chain_proof_spans) < 2 or len(chain_opaque_spans) < 1:
            return
        factorized_regions.append(
            {
                "current_proof_region": True,
                "proof_region_index": _int_value(
                    proof_region.get("proof_region_index"),
                    len(factorized_regions),
                ),
                "window_indices": list(proof_region.get("window_indices", []))
                if isinstance(proof_region.get("window_indices"), list)
                else [],
                "line_start": _int_value(chain_proof_spans[0].get("current_line_start"), 1),
                "line_end": _int_value(chain_proof_spans[-1].get("current_line_end"), 0),
                "target_line_start": _int_value(chain_proof_spans[0].get("target_line_start"), 1),
                "target_line_end": _int_value(chain_proof_spans[-1].get("target_line_end"), 0),
                "current_proof_span_count": len(chain_proof_spans),
                "current_proof_spans": [
                    {
                        "current_line_start": _int_value(item.get("current_line_start"), 1),
                        "current_line_end": _int_value(item.get("current_line_end"), 0),
                        "target_line_start": _int_value(item.get("target_line_start"), 1),
                        "target_line_end": _int_value(item.get("target_line_end"), 0),
                        "current_content": str(item.get("current_content", "")),
                        "target_content": str(item.get("target_content", "")),
                        "current_from_line_span_proof": bool(
                            item.get("current_from_line_span_proof", False)
                        ),
                        "target_from_expected_content": bool(
                            item.get("target_from_expected_content", False)
                        ),
                    }
                    for item in chain_proof_spans
                ],
                "current_proof_opaque_spans": [
                    {
                        "current_line_start": _int_value(item.get("current_line_start"), 1),
                        "current_line_end": _int_value(item.get("current_line_end"), 0),
                        "target_line_start": _int_value(item.get("target_line_start"), 1),
                        "target_line_end": _int_value(item.get("target_line_end"), 0),
                        "reason": str(item.get("reason", "")).strip() or "unknown",
                    }
                    for item in chain_opaque_spans
                ],
                "current_proof_opaque_span_count": len(chain_opaque_spans),
                "current_proof_complete": False,
                "current_proof_partial_coverage": True,
                "current_proof_covered_line_count": sum(
                    max(
                        0,
                        _int_value(item.get("current_line_end"), 0)
                        - _int_value(item.get("current_line_start"), 1)
                        + 1,
                    )
                    for item in chain_proof_spans
                ),
                "current_proof_missing_line_count": sum(
                    max(0, _int_value(item.get("line_count"), 0))
                    for item in chain_opaque_spans
                ),
                "current_proof_missing_span_count": len(chain_opaque_spans),
                "truncated": bool(proof_region.get("truncated", True)),
                "explicit_hidden_gap_current_proof": bool(
                    proof_region.get("explicit_hidden_gap_current_proof", False)
                ),
                "hidden_gap_current_from_line_span_proof": bool(
                    proof_region.get("hidden_gap_current_from_line_span_proof", False)
                ),
                "hidden_gap_target_from_expected_content": bool(
                    proof_region.get("hidden_gap_target_from_expected_content", False)
                ),
                "current_proof_factorized_subregion": True,
                "current_proof_parent_partial_region_topology": str(
                    opaque_metrics.get("opaque_topology", "")
                ).strip(),
                "current_proof_parent_opaque_span_count": max(
                    0,
                    _int_value(opaque_metrics.get("opaque_span_count"), 0),
                ),
                "current_proof_parent_blocking_opaque_span_count": max(
                    0,
                    _int_value(opaque_metrics.get("blocking_opaque_span_count"), 0),
                ),
            }
        )

    for entry in ordered_entries:
        if str(entry.get("kind", "")) == "proof":
            chain_proof_spans.append(entry)
            continue
        if bool(entry.get("admissible", False)):
            if chain_proof_spans:
                chain_opaque_spans.append(entry)
            continue
        flush_chain()
        chain_proof_spans = []
        chain_opaque_spans = []
    flush_chain()
    deduped_regions: list[dict[str, object]] = []
    seen_spans: set[tuple[int, int, int, int]] = set()
    for region in factorized_regions:
        signature = (
            _int_value(region.get("line_start"), 0),
            _int_value(region.get("line_end"), -1),
            _int_value(region.get("target_line_start"), 0),
            _int_value(region.get("target_line_end"), -1),
        )
        if signature in seen_spans:
            continue
        seen_spans.add(signature)
        deduped_regions.append(region)
    total_subregion_count = len(deduped_regions)
    for subregion_index, region in enumerate(deduped_regions):
        region["current_proof_factorized_subregion_index"] = subregion_index
        region["current_proof_factorized_subregion_count"] = total_subregion_count
    return deduped_regions


def _build_current_proof_region_candidate(
    *,
    path: str,
    proof_region: dict[str, object],
    indexed_windows: dict[int, dict[str, object]],
    available_window_count: int,
) -> dict[str, object] | None:
    subset_indices = sorted(
        {
            _int_value(index, -1)
            for index in proof_region.get("window_indices", [])
            if _int_value(index, -1) >= 0
        }
    )
    if len(subset_indices) < 2:
        return None
    ordered_source_windows = [indexed_windows.get(index) for index in subset_indices]
    if any(window is None or not isinstance(window, dict) for window in ordered_source_windows):
        return None
    full_target_contents = {
        str((window or {}).get("full_target_content", ""))
        for window in ordered_source_windows
        if isinstance(window, dict) and str((window or {}).get("full_target_content", ""))
    }
    full_target_content = next(iter(full_target_contents)) if len(full_target_contents) == 1 else ""
    if not full_target_content:
        return None
    region_current_start = max(1, _int_value(proof_region.get("line_start"), 1))
    region_current_end = max(
        region_current_start - 1,
        _int_value(proof_region.get("line_end"), region_current_start - 1),
    )
    region_target_start = max(1, _int_value(proof_region.get("target_line_start"), 1))
    region_target_end = max(
        region_target_start - 1,
        _int_value(proof_region.get("target_line_end"), region_target_start - 1),
    )
    current_proof_partial_coverage = bool(
        proof_region.get("current_proof_partial_coverage", False)
    )
    opaque_metrics = _current_proof_region_opaque_span_metrics(
        proof_region,
        region_current_start=region_current_start,
        region_current_end=region_current_end,
        region_target_start=region_target_start,
        region_target_end=region_target_end,
    )
    partial_region_block = False
    partial_region_block_topology = ""
    if current_proof_partial_coverage:
        opaque_span_count = max(0, opaque_metrics["opaque_span_count"])
        opaque_line_count = max(0, opaque_metrics["opaque_line_count"])
        opaque_boundary_touch_count = max(0, opaque_metrics["opaque_boundary_touch_count"])
        opaque_internal_span_count = max(0, opaque_metrics["opaque_internal_span_count"])
        opaque_max_span_line_count = max(0, opaque_metrics["opaque_max_span_line_count"])
        opaque_topology = str(opaque_metrics.get("opaque_topology", "")).strip()
        missing_line_count = max(
            0,
            _int_value(proof_region.get("current_proof_missing_line_count"), opaque_line_count),
        )
        missing_span_count = max(
            0,
            _int_value(proof_region.get("current_proof_missing_span_count"), opaque_span_count),
        )
        region_line_count = max(0, region_current_end - region_current_start + 1)
        raw_proof_spans = proof_region.get("current_proof_spans", [])
        if not isinstance(raw_proof_spans, list):
            raw_proof_spans = []
        current_proof_span_count = max(
            len(raw_proof_spans),
            _int_value(proof_region.get("current_proof_span_count"), len(raw_proof_spans)),
        )
        if opaque_topology == "single_internal_opaque":
            partial_region_block = True
            partial_region_block_topology = opaque_topology
        elif (
            opaque_topology == "sparse_internal_multi_opaque"
            and opaque_span_count >= 2
            and opaque_max_span_line_count <= 1
            and missing_span_count == opaque_span_count
            and missing_line_count == opaque_line_count
            and opaque_line_count <= max(2, region_line_count // 3)
            and current_proof_span_count >= opaque_span_count + 1
        ):
            partial_region_block = True
            partial_region_block_topology = opaque_topology
        else:
            return None
    raw_spans = proof_region.get("current_proof_spans", [])
    if not isinstance(raw_spans, list) or len(raw_spans) < 2:
        return None
    coverage_spans: list[tuple[int, int]] = []
    target_coverage_spans: list[tuple[int, int]] = []
    exact_window_count = 0
    for window in ordered_source_windows:
        current_span = _preview_window_current_span(window)
        target_span = _preview_window_target_span(window)
        if current_span is None or target_span is None:
            return None
        coverage_spans.append(current_span)
        target_coverage_spans.append(target_span)
        if bool(window.get("exact_target_span", False)):
            exact_window_count += 1
    hidden_gap_current_from_line_span_proof = False
    hidden_gap_target_from_expected_content = False
    for span in raw_spans:
        if not isinstance(span, dict):
            return None
        current_start = max(1, _int_value(span.get("current_line_start"), 1))
        current_end = max(
            current_start - 1,
            _int_value(span.get("current_line_end"), current_start - 1),
        )
        target_start = max(1, _int_value(span.get("target_line_start"), 1))
        target_end = max(
            target_start - 1,
            _int_value(span.get("target_line_end"), target_start - 1),
        )
        coverage_spans.append((current_start, current_end))
        target_coverage_spans.append((target_start, target_end))
        current_content = str(span.get("current_content", ""))
        current_from_line_span_proof = bool(span.get("current_from_line_span_proof", False))
        if current_from_line_span_proof:
            if current_content:
                return None
        elif len(current_content.splitlines()) != max(0, current_end - current_start + 1):
            return None
        hidden_gap_current_from_line_span_proof = (
            hidden_gap_current_from_line_span_proof or current_from_line_span_proof
        )
        hidden_gap_target_from_expected_content = (
            hidden_gap_target_from_expected_content
            or bool(span.get("target_from_expected_content", False))
        )
    if partial_region_block:
        raw_opaque_spans = proof_region.get("current_proof_opaque_spans", [])
        if not isinstance(raw_opaque_spans, list):
            raw_opaque_spans = []
        for span in raw_opaque_spans:
            if not isinstance(span, dict):
                continue
            current_start = max(1, _int_value(span.get("current_line_start"), 1))
            current_end = max(
                current_start - 1,
                _int_value(span.get("current_line_end"), current_start - 1),
            )
            target_start = max(1, _int_value(span.get("target_line_start"), 1))
            target_end = max(
                target_start - 1,
                _int_value(span.get("target_line_end"), target_start - 1),
            )
            if current_end < current_start or target_end < target_start:
                continue
            coverage_spans.append((current_start, current_end))
            target_coverage_spans.append((target_start, target_end))
            hidden_gap_target_from_expected_content = True
    if not _sorted_spans_cover_region(coverage_spans, region_current_start, region_current_end):
        return None
    if not _sorted_spans_cover_region(target_coverage_spans, region_target_start, region_target_end):
        return None
    expected_target_lines = _expected_target_lines_for_span(
        full_target_content,
        start_line=region_target_start,
        end_line=region_target_end,
    )
    if expected_target_lines is None:
        return None
    target_content = "".join(expected_target_lines)
    current_line_count = max(0, region_current_end - region_current_start + 1)
    if current_line_count <= 0 or not target_content:
        return None
    step = _build_block_replace_step_for_preview_region(
        path=path,
        preview_window={
            "target_content": target_content,
            "line_offset": region_current_start - 1,
            "target_line_offset": region_target_start - 1,
            "current_line_count": current_line_count,
        },
    )
    if step is None:
        return None
    command = _render_structured_edit_command(step)
    if not command:
        return None
    current_span = _step_current_span(step)
    target_span = _step_target_span(step)
    if current_span is None or target_span is None:
        return None
    if current_span != (region_current_start, region_current_end):
        return None
    if target_span != (region_target_start, region_target_end):
        return None
    current_span_line_count = max(0, current_span[1] - current_span[0] + 1)
    target_span_line_count = max(0, target_span[1] - target_span[0] + 1)
    inexact_window_count = len(subset_indices) - exact_window_count
    return {
        "command": command,
        "proposal_source": "structured_edit:block_replace",
        "edit_kind": "block_replace",
        "edit_source": (
            "workspace_preview_range"
            if bool(proof_region.get("truncated", False))
            else "workspace_preview"
        ),
        "window_index": None,
        "window_indices": list(subset_indices),
        "edit_window_count": len(subset_indices),
        "covered_window_indices": list(subset_indices),
        "covered_window_count": len(subset_indices),
        "available_window_count": available_window_count,
        "retained_window_count": max(
            _int_value((window or {}).get("retained_window_count"), available_window_count)
            for window in ordered_source_windows
            if isinstance(window, dict)
        ),
        "total_window_count": max(
            _int_value((window or {}).get("total_window_count"), available_window_count)
            for window in ordered_source_windows
            if isinstance(window, dict)
        ),
        "partial_window_coverage": any(
            bool((window or {}).get("partial_window_coverage", False))
            for window in ordered_source_windows
            if isinstance(window, dict)
        ),
        "exact_target_span": True,
        "precision_penalty": _preview_step_precision_penalty(step, exact_target_span=True),
        "exact_window_count": exact_window_count,
        "inexact_window_count": inexact_window_count,
        "safe_inexact_window_count": inexact_window_count,
        "current_proof_region_block": True,
        "current_proof_partial_region_block": partial_region_block,
        "current_proof_partial_region_topology": partial_region_block_topology,
        "current_proof_span_count": max(
            len(raw_spans),
            _int_value(proof_region.get("current_proof_span_count"), len(raw_spans)),
        ),
        "current_proof_opaque_spans": list(proof_region.get("current_proof_opaque_spans", []))
        if isinstance(proof_region.get("current_proof_opaque_spans"), list)
        else [],
        "current_proof_opaque_span_count": max(
            max(0, opaque_metrics["opaque_span_count"]),
            _int_value(proof_region.get("current_proof_opaque_span_count"), 0),
        ),
        "current_proof_opaque_line_count": max(0, opaque_metrics["opaque_line_count"]),
        "current_proof_opaque_internal_span_count": max(
            0,
            opaque_metrics["opaque_internal_span_count"],
        ),
        "current_proof_opaque_max_span_line_count": max(
            0,
            opaque_metrics["opaque_max_span_line_count"],
        ),
        "current_proof_admissible_opaque_span_count": max(
            0,
            _int_value(opaque_metrics.get("admissible_opaque_span_count"), 0),
        ),
        "current_proof_blocking_opaque_span_count": max(
            0,
            _int_value(opaque_metrics.get("blocking_opaque_span_count"), 0),
        ),
        "current_proof_coarse_internal_opaque_span_count": max(
            0,
            _int_value(opaque_metrics.get("coarse_internal_opaque_span_count"), 0),
        ),
        "current_proof_mixed_opaque_topology": bool(
            opaque_metrics.get("mixed_opaque_topology", False)
        ),
        "current_proof_opaque_boundary_touch_count": max(
            0,
            opaque_metrics["opaque_boundary_touch_count"],
        ),
        "current_proof_factorized_subregion": bool(
            proof_region.get("current_proof_factorized_subregion", False)
        ),
        "current_proof_factorized_subregion_index": max(
            0,
            _int_value(proof_region.get("current_proof_factorized_subregion_index"), 0),
        ),
        "current_proof_factorized_subregion_count": max(
            0,
            _int_value(proof_region.get("current_proof_factorized_subregion_count"), 0),
        ),
        "current_proof_parent_partial_region_topology": str(
            proof_region.get("current_proof_parent_partial_region_topology", "")
        ).strip(),
        "current_proof_parent_opaque_span_count": max(
            0,
            _int_value(proof_region.get("current_proof_parent_opaque_span_count"), 0),
        ),
        "current_proof_parent_blocking_opaque_span_count": max(
            0,
            _int_value(proof_region.get("current_proof_parent_blocking_opaque_span_count"), 0),
        ),
        "current_proof_complete": bool(proof_region.get("current_proof_complete", True)),
        "current_proof_partial_coverage": bool(
            proof_region.get("current_proof_partial_coverage", False)
        ),
        "current_proof_covered_line_count": max(
            0,
            _int_value(proof_region.get("current_proof_covered_line_count"), 0),
        ),
        "current_proof_missing_line_count": max(
            0,
            _int_value(proof_region.get("current_proof_missing_line_count"), 0),
        ),
        "current_proof_missing_span_count": max(
            0,
            _int_value(proof_region.get("current_proof_missing_span_count"), 0),
        ),
        "explicit_hidden_gap_current_proof": bool(
            proof_region.get("explicit_hidden_gap_current_proof", False)
        ),
        "hidden_gap_current_from_line_span_proof": hidden_gap_current_from_line_span_proof,
        "hidden_gap_target_from_expected_content": hidden_gap_target_from_expected_content,
        "current_span_line_count": current_span_line_count,
        "target_span_line_count": target_span_line_count,
        "span_line_count": max(current_span_line_count, target_span_line_count),
        "step": step,
    }


def _build_explicit_bridged_hidden_gap_run_candidate(
    *,
    path: str,
    bridge_window: dict[str, object],
    indexed_windows: dict[int, dict[str, object]],
    available_window_count: int,
) -> dict[str, object] | None:
    subset_indices = sorted(
        {
            _int_value(index, -1)
            for index in bridge_window.get("bridge_window_indices", [])
            if _int_value(index, -1) >= 0
        }
    )
    if len(subset_indices) < 2:
        return None
    if any(right != left + 1 for left, right in zip(subset_indices, subset_indices[1:])):
        return None
    raw_segments = bridge_window.get("bridge_segments", [])
    if not isinstance(raw_segments, list) or not raw_segments:
        return None
    bridge_by_pair: dict[tuple[int, int], dict[str, object]] = {}
    for segment in raw_segments:
        if not isinstance(segment, dict):
            return None
        bridge_window_indices = tuple(
            sorted(
                {
                    _int_value(index, -1)
                    for index in segment.get("bridge_window_indices", [])
                    if _int_value(index, -1) >= 0
                }
            )
        )
        if len(bridge_window_indices) != 2 or bridge_window_indices[1] != bridge_window_indices[0] + 1:
            return None
        bridge_by_pair[bridge_window_indices] = segment
    if len(bridge_by_pair) != len(subset_indices) - 1:
        return None
    candidate = _build_bridged_hidden_gap_region_candidate(
        path=path,
        subset_indices=subset_indices,
        indexed_windows=indexed_windows,
        bridge_by_pair=bridge_by_pair,
        available_window_count=available_window_count,
    )
    if candidate is None:
        return None
    step = candidate.get("step")
    if not isinstance(step, dict):
        return None
    current_span = _step_current_span(step)
    target_span = _step_target_span(step)
    if current_span is None or target_span is None:
        return None
    if _int_value(bridge_window.get("line_start"), current_span[0]) != current_span[0]:
        return None
    if _int_value(bridge_window.get("line_end"), current_span[1]) != current_span[1]:
        return None
    if _int_value(bridge_window.get("target_line_start"), target_span[0]) != target_span[0]:
        return None
    if _int_value(bridge_window.get("target_line_end"), target_span[1]) != target_span[1]:
        return None
    if _int_value(
        bridge_window.get("hidden_gap_current_line_count"),
        candidate.get("hidden_gap_current_line_count"),
    ) != _int_value(candidate.get("hidden_gap_current_line_count"), 0):
        return None
    if _int_value(
        bridge_window.get("hidden_gap_target_line_count"),
        candidate.get("hidden_gap_target_line_count"),
    ) != _int_value(candidate.get("hidden_gap_target_line_count"), 0):
        return None
    candidate["bridge_run_segment_count"] = len(raw_segments)
    return candidate


def _build_bridged_hidden_gap_region_candidate(
    *,
    path: str,
    subset_indices: list[int],
    indexed_windows: dict[int, dict[str, object]],
    bridge_by_pair: dict[tuple[int, int], dict[str, object]],
    available_window_count: int,
) -> dict[str, object] | None:
    if len(subset_indices) < 2:
        return None
    if _subset_skips_intervening_retained_window(
        [{"window_index": index} for index in subset_indices]
    ):
        return None
    ordered_source_windows = [indexed_windows.get(index) for index in subset_indices]
    if any(window is None or not isinstance(window, dict) for window in ordered_source_windows):
        return None
    full_target_contents = {
        str((window or {}).get("full_target_content", ""))
        for window in ordered_source_windows
        if isinstance(window, dict) and str((window or {}).get("full_target_content", ""))
    }
    full_target_content = next(iter(full_target_contents)) if len(full_target_contents) == 1 else ""
    span_windows: list[dict[str, object]] = []
    for index, window in zip(subset_indices, ordered_source_windows):
        current_span = _preview_window_current_span(window)
        if current_span is None:
            return None
        span_windows.append(
            {
                "window": window,
                "current_start": current_span[0],
                "current_end": current_span[1],
                "window_index": index,
            }
        )
    first_bridge = bridge_by_pair.get((subset_indices[0], subset_indices[1]))
    if not isinstance(first_bridge, dict):
        return None
    region_current_start = max(1, _int_value(first_bridge.get("line_start"), 1))
    region_target_start = max(1, _int_value(first_bridge.get("target_line_start"), 1))
    if region_current_start != int(span_windows[0]["current_start"]):
        return None
    target_parts: list[str] = []
    hidden_gap_current_line_count = 0
    hidden_gap_target_line_count = 0
    hidden_gap_current_from_line_span_proof = False
    expected_window_target_start = region_target_start
    expected_region_target_end = region_target_start - 1
    for position, item in enumerate(span_windows):
        window = item.get("window") or {}
        target_content = str(window.get("target_content", ""))
        current_start = int(item["current_start"])
        current_end = int(item["current_end"])
        if position > 0:
            previous_item = span_windows[position - 1]
            if current_start <= int(previous_item["current_end"]):
                return None
        target_parts.append(target_content)
        window_target_line_count = len(target_content.splitlines())
        expected_region_target_end = expected_window_target_start + window_target_line_count - 1
        if position == len(span_windows) - 1:
            break
        next_index = subset_indices[position + 1]
        bridge = bridge_by_pair.get((subset_indices[position], next_index))
        if not isinstance(bridge, dict):
            return None
        if not bool(bridge.get("explicit_hidden_gap_current_proof", False)):
            return None
        if max(1, _int_value(bridge.get("line_start"), current_start)) != current_start:
            return None
        if max(1, _int_value(bridge.get("target_line_start"), expected_window_target_start)) != expected_window_target_start:
            return None
        next_current_start = int(span_windows[position + 1]["current_start"])
        next_window_target_line_count = len(
            str((span_windows[position + 1].get("window") or {}).get("target_content", "")).splitlines()
        )
        expected_current_gap_start = current_end + 1
        expected_current_gap_end = next_current_start - 1
        if _int_value(bridge.get("hidden_gap_current_line_start"), expected_current_gap_start) != expected_current_gap_start:
            return None
        if _int_value(bridge.get("hidden_gap_current_line_end"), expected_current_gap_end) != expected_current_gap_end:
            return None
        expected_hidden_gap_target_start = expected_window_target_start + window_target_line_count
        expected_hidden_gap_target_end = (
            expected_hidden_gap_target_start
            + max(0, _int_value(bridge.get("hidden_gap_target_line_count"), 0))
            - 1
        )
        if _int_value(bridge.get("hidden_gap_target_line_start"), expected_hidden_gap_target_start) != expected_hidden_gap_target_start:
            return None
        if _int_value(bridge.get("hidden_gap_target_line_end"), expected_hidden_gap_target_end) != expected_hidden_gap_target_end:
            return None
        hidden_current_content = str(bridge.get("hidden_gap_current_content", ""))
        hidden_target_content = str(bridge.get("hidden_gap_target_content", ""))
        current_from_line_span_proof = bool(
            bridge.get("hidden_gap_current_from_line_span_proof", False)
        )
        if bool(bridge.get("hidden_gap_target_from_expected_content", False)):
            if expected_hidden_gap_target_end < expected_hidden_gap_target_start:
                hidden_target_content = ""
            else:
                if not full_target_content:
                    return None
                expected_hidden_gap_target_lines = _expected_target_lines_for_span(
                    full_target_content,
                    start_line=expected_hidden_gap_target_start,
                    end_line=expected_hidden_gap_target_end,
                )
                if expected_hidden_gap_target_lines is None:
                    return None
                hidden_target_content = "".join(expected_hidden_gap_target_lines)
        if current_from_line_span_proof:
            if hidden_current_content:
                return None
        elif len(hidden_current_content.splitlines()) != max(
            0,
            expected_current_gap_end - expected_current_gap_start + 1,
        ):
            return None
        if len(hidden_target_content.splitlines()) != max(0, expected_hidden_gap_target_end - expected_hidden_gap_target_start + 1):
            return None
        target_parts.append(hidden_target_content)
        hidden_gap_current_line_count += max(0, _int_value(bridge.get("hidden_gap_current_line_count"), 0))
        hidden_gap_target_line_count += max(0, _int_value(bridge.get("hidden_gap_target_line_count"), 0))
        hidden_gap_current_from_line_span_proof = (
            hidden_gap_current_from_line_span_proof or current_from_line_span_proof
        )
        expected_bridge_target_end = expected_hidden_gap_target_end + next_window_target_line_count
        if _int_value(bridge.get("target_line_end"), expected_bridge_target_end) != expected_bridge_target_end:
            return None
        expected_window_target_start = expected_hidden_gap_target_end + 1
    target_content = "".join(target_parts)
    if not target_content:
        return None
    region_current_end = int(span_windows[-1]["current_end"])
    current_line_count = max(0, region_current_end - region_current_start + 1)
    if current_line_count <= 0:
        return None
    step = _build_block_replace_step_for_preview_region(
        path=path,
        preview_window={
            "target_content": target_content,
            "line_offset": region_current_start - 1,
            "target_line_offset": region_target_start - 1,
            "current_line_count": current_line_count,
        },
    )
    if step is None:
        return None
    command = _render_structured_edit_command(step)
    if not command:
        return None
    current_span = _step_current_span(step)
    target_span = _step_target_span(step)
    if current_span is None or target_span is None:
        return None
    if target_span[1] != max(region_target_start - 1, expected_region_target_end):
        return None
    current_span_line_count = max(0, current_span[1] - current_span[0] + 1)
    target_span_line_count = max(0, target_span[1] - target_span[0] + 1)
    exact_window_count = sum(
        1
        for window in ordered_source_windows
        if isinstance(window, dict) and bool(window.get("exact_target_span", False))
    )
    inexact_window_count = len(subset_indices) - exact_window_count
    return {
        "command": command,
        "proposal_source": "structured_edit:block_replace",
        "edit_kind": "block_replace",
        "edit_source": (
            "workspace_preview_range"
            if bool(first_bridge.get("truncated", False))
            else "workspace_preview"
        ),
        "window_index": None,
        "window_indices": list(subset_indices),
        "edit_window_count": len(subset_indices),
        "covered_window_indices": list(subset_indices),
        "covered_window_count": len(subset_indices),
        "available_window_count": available_window_count,
        "retained_window_count": max(
            _int_value((window or {}).get("retained_window_count"), available_window_count)
            for window in ordered_source_windows
            if isinstance(window, dict)
        ),
        "total_window_count": max(
            _int_value((window or {}).get("total_window_count"), available_window_count)
            for window in ordered_source_windows
            if isinstance(window, dict)
        ),
        "partial_window_coverage": any(
            bool((window or {}).get("partial_window_coverage", False))
            for window in ordered_source_windows
            if isinstance(window, dict)
        ),
        "exact_target_span": True,
        "precision_penalty": _preview_step_precision_penalty(
            step,
            exact_target_span=True,
        ),
        "exact_window_count": exact_window_count,
        "inexact_window_count": inexact_window_count,
        "safe_inexact_window_count": inexact_window_count,
        "bridged_hidden_gap_region_block": True,
        "explicit_hidden_gap_current_proof": True,
        "hidden_gap_current_from_line_span_proof": hidden_gap_current_from_line_span_proof,
        "hidden_gap_target_from_expected_content": any(
            bool((bridge_by_pair.get((left_index, right_index)) or {}).get("hidden_gap_target_from_expected_content", False))
            for left_index, right_index in zip(subset_indices, subset_indices[1:])
        ),
        "hidden_gap_current_line_count": hidden_gap_current_line_count,
        "hidden_gap_target_line_count": hidden_gap_target_line_count,
        "current_span_line_count": current_span_line_count,
        "target_span_line_count": target_span_line_count,
        "span_line_count": max(current_span_line_count, target_span_line_count),
        "hidden_gap_bridge_count": max(1, len(subset_indices) - 1),
        "step": step,
    }


def _compose_workspace_preview_frontier_block_replace_proposals(
    *,
    path: str,
    proposals: list[dict[str, object]],
    available_window_count: int,
) -> list[dict[str, object]]:
    seen_signatures: set[tuple[str, tuple[int, ...]]] = {
        _overlap_block_replace_signature(proposal)
        for proposal in proposals
        if str(proposal.get("proposal_source", "")).strip() == "structured_edit:block_replace"
    }
    block_proposals: list[dict[str, object]] = []
    for proposal in proposals:
        frontier_item = _frontier_block_replace_component(path=path, proposal=proposal)
        if frontier_item is None:
            continue
        block_proposals.append(
            {
                "proposal": proposal,
                "step": frontier_item["step"],
                "current_start": frontier_item["current_start"],
                "current_end": frontier_item["current_end"],
                "target_start": frontier_item["target_start"],
                "target_end": frontier_item["target_end"],
                "window_index": _int_value(proposal.get("window_index"), 0),
                "window_indices": _proposal_window_indices(proposal),
                "synthetic_single_line_block_replace": bool(
                    frontier_item.get("synthetic_single_line_block_replace", False)
                ),
            }
        )
    if len(block_proposals) < 2:
        return []
    candidates: list[dict[str, object]] = []
    for subset_size in range(2, len(block_proposals) + 1):
        for subset in combinations(block_proposals, subset_size):
            ordered_subset = sorted(subset, key=_window_subset_sort_key)
            if not _block_proposal_subset_is_contiguous(ordered_subset):
                continue
            step = _merge_block_replace_proposal_subset(path=path, ordered_subset=ordered_subset)
            if step is None:
                continue
            command = _render_structured_edit_command(step)
            if not command:
                continue
            window_indices = sorted(
                {
                    _int_value(index, -1)
                    for item in ordered_subset
                    for index in _proposal_window_indices(item.get("proposal") or {})
                    if _int_value(index, -1) >= 0
                }
            )
            covered_window_indices = sorted(
                {
                    _int_value(index, -1)
                    for item in ordered_subset
                    for index in (item.get("proposal") or {}).get("covered_window_indices", [])
                    if _int_value(index, -1) >= 0
                }
            )
            current_start = min(int(item["current_start"]) for item in ordered_subset)
            current_end = max(int(item["current_end"]) for item in ordered_subset)
            target_start = min(int(item["target_start"]) for item in ordered_subset)
            target_end = max(int(item["target_end"]) for item in ordered_subset)
            current_span_line_count = max(0, current_end - current_start + 1)
            target_span_line_count = max(0, target_end - target_start + 1)
            bridged_hidden_gap_region_block = any(
                bool((item.get("proposal") or {}).get("bridged_hidden_gap_region_block", False))
                for item in ordered_subset
            )
            exact_hidden_gap_region_block = (
                not bridged_hidden_gap_region_block
                and any(
                    bool((item.get("proposal") or {}).get("exact_hidden_gap_region_block", False))
                    for item in ordered_subset
                )
            )
            hidden_gap_current_line_count = sum(
                max(0, _int_value((item.get("proposal") or {}).get("hidden_gap_current_line_count"), 0))
                for item in ordered_subset
            )
            hidden_gap_target_line_count = sum(
                max(0, _int_value((item.get("proposal") or {}).get("hidden_gap_target_line_count"), 0))
                for item in ordered_subset
            )
            exact_contiguous_region_block = (
                not bridged_hidden_gap_region_block
                and not exact_hidden_gap_region_block
                and hidden_gap_current_line_count <= 0
                and hidden_gap_target_line_count <= 0
            )
            shared_anchor_metadata = _frontier_shared_anchor_metadata(
                ordered_subset=ordered_subset,
                covered_window_indices=covered_window_indices,
                exact_contiguous_region_block=exact_contiguous_region_block,
            )
            candidates.append(
                {
                    "command": command,
                    "proposal_source": "structured_edit:block_replace",
                    "edit_kind": "block_replace",
                    "edit_source": (
                        "workspace_preview_range"
                        if any(
                            str((item.get("proposal") or {}).get("edit_source", "")).strip()
                            == "workspace_preview_range"
                            for item in ordered_subset
                        )
                        else "workspace_preview"
                    ),
                    "window_index": None,
                    "window_indices": window_indices,
                    "edit_window_count": len(window_indices) if window_indices else len(ordered_subset),
                    "covered_window_indices": covered_window_indices,
                    "covered_window_count": len(covered_window_indices),
                    "available_window_count": max(
                        available_window_count,
                        max(
                            _int_value(
                                (item.get("proposal") or {}).get("available_window_count"),
                                available_window_count,
                            )
                            for item in ordered_subset
                        ),
                    ),
                    "retained_window_count": max(
                        _int_value(
                            (item.get("proposal") or {}).get("retained_window_count"),
                            available_window_count,
                        )
                        for item in ordered_subset
                    ),
                    "total_window_count": max(
                        _int_value(
                            (item.get("proposal") or {}).get("total_window_count"),
                            available_window_count,
                        )
                        for item in ordered_subset
                    ),
                    "partial_window_coverage": any(
                        bool((item.get("proposal") or {}).get("partial_window_coverage", False))
                        for item in ordered_subset
                    ),
                    "exact_target_span": True,
                    "precision_penalty": 0,
                    "exact_window_count": sum(
                        max(0, _int_value((item.get("proposal") or {}).get("exact_window_count"), 0))
                        for item in ordered_subset
                    ),
                    "inexact_window_count": sum(
                        max(0, _int_value((item.get("proposal") or {}).get("inexact_window_count"), 0))
                        for item in ordered_subset
                    ),
                    "overlap_alias_pair_count": sum(
                        max(0, _int_value((item.get("proposal") or {}).get("overlap_alias_pair_count"), 0))
                        for item in ordered_subset
                    ),
                    "overlap_alias_window_count": sum(
                        max(0, _int_value((item.get("proposal") or {}).get("overlap_alias_window_count"), 0))
                        for item in ordered_subset
                    ),
                    "safe_inexact_window_count": sum(
                        max(0, _int_value((item.get("proposal") or {}).get("safe_inexact_window_count"), 0))
                        for item in ordered_subset
                    ),
                    "recovered_conflicting_alias": any(
                        bool((item.get("proposal") or {}).get("recovered_conflicting_alias", False))
                        for item in ordered_subset
                    ),
                    "overlap_component_alias_pair_count": sum(
                        max(
                            0,
                            _int_value(
                                (item.get("proposal") or {}).get("overlap_component_alias_pair_count"),
                                0,
                            ),
                        )
                        for item in ordered_subset
                    ),
                    "overlap_component_unresolved_alias_pair_count": sum(
                        max(
                            0,
                            _int_value(
                                (item.get("proposal") or {}).get(
                                    "overlap_component_unresolved_alias_pair_count"
                                ),
                                0,
                            ),
                        )
                        for item in ordered_subset
                    ),
                    "overlap_component_conflicting_alias_pair_count": sum(
                        max(
                            0,
                            _int_value(
                                (item.get("proposal") or {}).get(
                                    "overlap_component_conflicting_alias_pair_count"
                                ),
                                0,
                            ),
                        )
                        for item in ordered_subset
                    ),
                    "overlap_component_candidate_count": sum(
                        max(
                            0,
                            _int_value(
                                (item.get("proposal") or {}).get("overlap_component_candidate_count"),
                                0,
                            ),
                        )
                        for item in ordered_subset
                    ),
                    "overlap_component_frontier_gap": sum(
                        max(
                            0.0,
                            _float_value(
                                (item.get("proposal") or {}).get("overlap_component_frontier_gap"),
                                0.0,
                            ),
                        )
                        for item in ordered_subset
                    ),
                    "current_span_line_count": current_span_line_count,
                    "target_span_line_count": target_span_line_count,
                    "span_line_count": max(current_span_line_count, target_span_line_count),
                    "exact_contiguous_region_block": exact_contiguous_region_block,
                    "exact_hidden_gap_region_block": exact_hidden_gap_region_block,
                    "shared_anchor_exact_region_block": shared_anchor_metadata[
                        "shared_anchor_exact_region_block"
                    ],
                    "shared_anchor_pair_kinds": shared_anchor_metadata[
                        "shared_anchor_pair_kinds"
                    ],
                    "shared_anchor_exact_neighbor_count": shared_anchor_metadata[
                        "shared_anchor_exact_neighbor_count"
                    ],
                    "shared_anchor_core_count": shared_anchor_metadata[
                        "shared_anchor_core_count"
                    ],
                    "shared_anchor_hybrid_component_count": shared_anchor_metadata[
                        "shared_anchor_hybrid_component_count"
                    ],
                    "shared_anchor_mixed_insert_delete": shared_anchor_metadata[
                        "shared_anchor_mixed_insert_delete"
                    ],
                    "bridged_hidden_gap_region_block": bridged_hidden_gap_region_block,
                    "current_proof_region_block": any(
                        bool((item.get("proposal") or {}).get("current_proof_region_block", False))
                        for item in ordered_subset
                    ),
                    "current_proof_span_count": sum(
                        max(0, _int_value((item.get("proposal") or {}).get("current_proof_span_count"), 0))
                        for item in ordered_subset
                    ),
                    "current_proof_complete": all(
                        bool((item.get("proposal") or {}).get("current_proof_complete", False))
                        for item in ordered_subset
                        if bool((item.get("proposal") or {}).get("current_proof_region_block", False))
                    ),
                    "current_proof_partial_coverage": any(
                        bool((item.get("proposal") or {}).get("current_proof_partial_coverage", False))
                        for item in ordered_subset
                    ),
                    "current_proof_covered_line_count": sum(
                        max(0, _int_value((item.get("proposal") or {}).get("current_proof_covered_line_count"), 0))
                        for item in ordered_subset
                    ),
                    "current_proof_missing_line_count": sum(
                        max(0, _int_value((item.get("proposal") or {}).get("current_proof_missing_line_count"), 0))
                        for item in ordered_subset
                    ),
                    "current_proof_missing_span_count": sum(
                        max(0, _int_value((item.get("proposal") or {}).get("current_proof_missing_span_count"), 0))
                        for item in ordered_subset
                    ),
                    "explicit_hidden_gap_current_proof": all(
                        bool((item.get("proposal") or {}).get("explicit_hidden_gap_current_proof", False))
                        for item in ordered_subset
                        if bool((item.get("proposal") or {}).get("bridged_hidden_gap_region_block", False))
                        or bool((item.get("proposal") or {}).get("current_proof_region_block", False))
                    ),
                    "hidden_gap_current_from_line_span_proof": any(
                        bool((item.get("proposal") or {}).get("hidden_gap_current_from_line_span_proof", False))
                        for item in ordered_subset
                    ),
                    "hidden_gap_target_from_expected_content": any(
                        bool((item.get("proposal") or {}).get("hidden_gap_target_from_expected_content", False))
                        for item in ordered_subset
                    ),
                    "hidden_gap_bridge_count": sum(
                        max(0, _int_value((item.get("proposal") or {}).get("hidden_gap_bridge_count"), 0))
                        for item in ordered_subset
                    ),
                    "hidden_gap_current_line_count": hidden_gap_current_line_count,
                    "hidden_gap_target_line_count": hidden_gap_target_line_count,
                    "synthetic_single_line_block_replace_count": sum(
                        1
                        for item in ordered_subset
                        if bool(item.get("synthetic_single_line_block_replace", False))
                    ),
                    "step": step,
                }
            )
    ranked = sorted(candidates, key=_overlap_block_replace_rank_key)
    deduped: list[dict[str, object]] = []
    for candidate in ranked:
        signature = _overlap_block_replace_signature(candidate)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        deduped.append(candidate)
        if len(deduped) >= 3:
            break
    return deduped


def _frontier_block_replace_component(
    *,
    path: str,
    proposal: dict[str, object],
) -> dict[str, object] | None:
    if not bool(proposal.get("exact_target_span", False)):
        return None
    step = proposal.get("step")
    if not isinstance(step, dict):
        return None
    current_span = _step_current_span(step)
    target_span = _step_target_span(step)
    if current_span is None or target_span is None:
        return None
    edit_kind = str(step.get("edit_kind", "")).strip()
    if edit_kind == "block_replace":
        return {
            "step": step,
            "current_start": current_span[0],
            "current_end": current_span[1],
            "target_start": target_span[0],
            "target_end": target_span[1],
            "synthetic_single_line_block_replace": False,
        }
    promoted_step = _promote_exact_single_line_replacement_to_block_replace(path=path, step=step)
    if promoted_step is None:
        return None
    promoted_current_span = _step_current_span(promoted_step)
    promoted_target_span = _step_target_span(promoted_step)
    if promoted_current_span is None or promoted_target_span is None:
        return None
    return {
        "step": promoted_step,
        "current_start": promoted_current_span[0],
        "current_end": promoted_current_span[1],
        "target_start": promoted_target_span[0],
        "target_end": promoted_target_span[1],
        "synthetic_single_line_block_replace": True,
    }


def _promote_exact_single_line_replacement_to_block_replace(
    *,
    path: str,
    step: dict[str, object],
) -> dict[str, object] | None:
    edit_kind = str(step.get("edit_kind", "")).strip()
    if edit_kind not in {"token_replace", "token_insert", "token_delete", "line_replace"}:
        return None
    if _int_value(step.get("line_delta"), 0) != 0:
        return None
    current_span = _step_current_span(step)
    target_span = _step_target_span(step)
    if current_span is None or target_span is None:
        return None
    if current_span[0] != current_span[1] or target_span[0] != target_span[1]:
        return None
    replacements = step.get("replacements", [])
    if not isinstance(replacements, list) or len(replacements) != 1:
        return None
    replacement = replacements[0]
    if not isinstance(replacement, dict):
        return None
    line_number = _int_value(replacement.get("line_number"), 0)
    if line_number <= 0:
        return None
    before_line = replacement.get("before_line")
    after_line = replacement.get("after_line")
    if before_line is None or after_line is None:
        return None
    promoted_replacement = {
        "start_line": current_span[0],
        "end_line": current_span[1],
        "before_lines": [str(before_line)],
        "after_lines": [str(after_line)],
    }
    return {
        "path": path,
        "edit_kind": "block_replace",
        "replacement": promoted_replacement,
        "current_start_line": current_span[0],
        "current_end_line": current_span[1],
        "target_start_line": target_span[0],
        "target_end_line": target_span[1],
        "line_delta": 0,
        "edit_score": _edit_candidate_score("block_replace", replacement=promoted_replacement),
    }


def _block_proposal_subset_is_contiguous(ordered_subset: list[dict[str, object]]) -> bool:
    if len(ordered_subset) < 2:
        return False
    seen_window_indices: set[int] = set()
    ordered_window_indices: list[int] = []
    previous: dict[str, object] | None = None
    for item in ordered_subset:
        proposal = item.get("proposal") or {}
        step = item.get("step")
        if not isinstance(step, dict):
            return False
        if str(step.get("edit_kind", "")).strip() != "block_replace":
            return False
        if (
            str(proposal.get("proposal_source", "")).strip() != "structured_edit:block_replace"
            and not bool(item.get("synthetic_single_line_block_replace", False))
        ):
            return False
        if not bool(proposal.get("exact_target_span", False)):
            return False
        window_indices = _proposal_window_indices(proposal)
        if not window_indices:
            return False
        if any(index in seen_window_indices for index in window_indices):
            return False
        seen_window_indices.update(window_indices)
        ordered_window_indices.extend(window_indices)
        if previous is not None:
            if int(item["current_start"]) != int(previous["current_end"]) + 1:
                return False
            if int(item["target_start"]) != int(previous["target_end"]) + 1:
                return False
        previous = item
    sorted_indices = sorted(ordered_window_indices)
    if any(right_index != left_index + 1 for left_index, right_index in zip(sorted_indices, sorted_indices[1:])):
        return False
    return True


def _merge_block_replace_proposal_subset(
    *,
    path: str,
    ordered_subset: list[dict[str, object]],
) -> dict[str, object] | None:
    if len(ordered_subset) < 2:
        return None
    before_lines: list[str] = []
    after_lines: list[str] = []
    current_start = int(ordered_subset[0]["current_start"])
    current_end = int(ordered_subset[-1]["current_end"])
    target_start = int(ordered_subset[0]["target_start"])
    target_end = int(ordered_subset[-1]["target_end"])
    for item in ordered_subset:
        step = item.get("step")
        if not isinstance(step, dict):
            return None
        replacement = step.get("replacement")
        if not isinstance(replacement, dict):
            return None
        current_span = _step_current_span(step)
        target_span = _step_target_span(step)
        if current_span is None or target_span is None:
            return None
        replacement_before_lines = [str(line) for line in replacement.get("before_lines", [])]
        replacement_after_lines = [str(line) for line in replacement.get("after_lines", [])]
        if len(replacement_before_lines) != max(0, current_span[1] - current_span[0] + 1):
            return None
        if len(replacement_after_lines) != max(0, target_span[1] - target_span[0] + 1):
            return None
        before_lines.extend(replacement_before_lines)
        after_lines.extend(replacement_after_lines)
    replacement = {
        "start_line": current_start,
        "end_line": current_end,
        "before_lines": before_lines,
        "after_lines": after_lines,
    }
    return {
        "path": path,
        "edit_kind": "block_replace",
        "replacement": replacement,
        "current_start_line": current_start,
        "current_end_line": current_end,
        "target_start_line": target_start,
        "target_end_line": target_end,
        "line_delta": len(after_lines) - len(before_lines),
        "edit_score": _edit_candidate_score("block_replace", replacement=replacement),
    }


def _frontier_shared_anchor_metadata(
    *,
    ordered_subset: list[dict[str, object]],
    covered_window_indices: list[int],
    exact_contiguous_region_block: bool,
) -> dict[str, object]:
    shared_anchor_core_count = 0
    shared_anchor_hybrid_component_count = 0
    allows_hybrid_shared_anchor_metadata = False
    shared_anchor_pair_kinds: list[str] = []
    shared_anchor_participating_window_count = 0
    for item in ordered_subset:
        proposal = item.get("proposal") or {}
        if not isinstance(proposal, dict):
            continue
        hybrid_component_count = _shared_anchor_hybrid_component_count_for_proposal(proposal)
        if hybrid_component_count > 0:
            shared_anchor_hybrid_component_count += hybrid_component_count
            allows_hybrid_shared_anchor_metadata = True
        if not bool(proposal.get("shared_anchor_exact_region_block", False)):
            continue
        allows_hybrid_shared_anchor_metadata = True
        pair_kinds = [
            str(pair_kind).strip()
            for pair_kind in proposal.get("shared_anchor_pair_kinds", [])
            if str(pair_kind).strip()
        ]
        shared_anchor_pair_kinds.extend(pair_kinds)
        proposal_core_count = max(
            0,
            _int_value(proposal.get("shared_anchor_core_count"), 0),
        )
        if proposal_core_count <= 0 and pair_kinds:
            proposal_core_count = 1
        shared_anchor_core_count += proposal_core_count
        proposal_window_indices = [
            _int_value(index, -1)
            for index in _proposal_window_indices(proposal)
            if _int_value(index, -1) >= 0
        ]
        proposal_window_count = len(proposal_window_indices)
        if proposal_window_count <= 0:
            proposal_window_count = max(
                0,
                _int_value(proposal.get("covered_window_count"), 0),
            )
        proposal_neighbor_count = max(
            0,
            _int_value(proposal.get("shared_anchor_exact_neighbor_count"), 0),
        )
        participating_window_count = max(0, proposal_window_count - proposal_neighbor_count)
        if participating_window_count <= 0 and pair_kinds and proposal_window_count > 0:
            participating_window_count = min(
                proposal_window_count,
                len(pair_kinds) + 1,
            )
        shared_anchor_participating_window_count += participating_window_count
    if not shared_anchor_pair_kinds or (
        not exact_contiguous_region_block and not allows_hybrid_shared_anchor_metadata
    ):
        return {
            "shared_anchor_exact_region_block": False,
            "shared_anchor_pair_kinds": [],
            "shared_anchor_exact_neighbor_count": 0,
            "shared_anchor_core_count": 0,
            "shared_anchor_hybrid_component_count": 0,
            "shared_anchor_mixed_insert_delete": False,
        }
    shared_anchor_exact_neighbor_count = max(
        0,
        len(covered_window_indices)
        - min(len(covered_window_indices), shared_anchor_participating_window_count),
    )
    return {
        "shared_anchor_exact_region_block": True,
        "shared_anchor_pair_kinds": shared_anchor_pair_kinds,
        "shared_anchor_exact_neighbor_count": shared_anchor_exact_neighbor_count,
        "shared_anchor_core_count": max(1, shared_anchor_core_count),
        "shared_anchor_hybrid_component_count": shared_anchor_hybrid_component_count,
        "shared_anchor_mixed_insert_delete": any(
            pair_kind in {"same_anchor_insert_delete", "adjacent_delete_insert"}
            for pair_kind in shared_anchor_pair_kinds
        ),
    }


def _shared_anchor_hybrid_component_count_for_proposal(
    proposal: dict[str, object],
) -> int:
    hybrid_component_count = 0
    if bool(proposal.get("recovered_conflicting_alias", False)) or max(
        0,
        _int_value(proposal.get("overlap_alias_pair_count"), 0),
    ) > 0:
        hybrid_component_count += 1
    if bool(proposal.get("exact_hidden_gap_region_block", False)) or bool(
        proposal.get("bridged_hidden_gap_region_block", False)
    ) or bool(
        proposal.get("current_proof_region_block", False)
    ):
        hybrid_component_count += 1
    return hybrid_component_count


def _subset_skips_intervening_retained_window(
    ordered_subset: list[dict[str, object]],
) -> bool:
    if len(ordered_subset) < 2:
        return False
    ordered_indices = [
        _int_value(item.get("window_index"), -1)
        for item in ordered_subset
        if _int_value(item.get("window_index"), -1) >= 0
    ]
    if len(ordered_indices) != len(ordered_subset):
        return False
    for left_index, right_index in zip(ordered_indices, ordered_indices[1:]):
        if right_index != left_index + 1:
            return True
    return False


def _recover_exact_hidden_gap_preview_window(
    ordered_subset: list[dict[str, object]],
) -> dict[str, object] | None:
    if len(ordered_subset) < 2:
        return None
    if _subset_skips_intervening_retained_window(ordered_subset):
        return None
    if not all(bool((item.get("window") or {}).get("exact_target_span", False)) for item in ordered_subset):
        return None
    full_target_content = _shared_preview_full_target_content(ordered_subset)
    if not full_target_content:
        return None
    current_start = min(int(item["current_start"]) for item in ordered_subset)
    current_end = max(int(item["current_end"]) for item in ordered_subset)
    target_start = min(int(item["target_start"]) for item in ordered_subset)
    target_end = max(int(item["target_end"]) for item in ordered_subset)
    expected_target_lines = _expected_target_lines_for_span(
        full_target_content,
        start_line=target_start,
        end_line=target_end,
    )
    if expected_target_lines is None:
        return None
    if not _subset_exact_windows_match_expected_target(ordered_subset, expected_target_lines, target_start):
        return None
    current_line_count = max(0, current_end - current_start + 1)
    target_line_count = len(expected_target_lines)
    visible_current_line_count = sum(
        len(str((item.get("window") or {}).get("baseline_content", "")).splitlines())
        for item in ordered_subset
    )
    visible_target_line_count = sum(
        len(str((item.get("window") or {}).get("target_content", "")).splitlines())
        for item in ordered_subset
    )
    hidden_gap_current_line_count = max(0, current_line_count - visible_current_line_count)
    hidden_gap_target_line_count = max(0, target_line_count - visible_target_line_count)
    if hidden_gap_current_line_count <= 0 and hidden_gap_target_line_count <= 0:
        return None
    return {
        "target_content": "".join(expected_target_lines),
        "line_offset": current_start - 1,
        "target_line_offset": target_start - 1,
        "current_line_count": current_line_count,
        "target_line_count": target_line_count,
        "hidden_gap_current_line_count": hidden_gap_current_line_count,
        "hidden_gap_target_line_count": hidden_gap_target_line_count,
        "hidden_gap_target_from_expected_content": True,
        "truncated": any(bool((item.get("window") or {}).get("truncated", False)) for item in ordered_subset),
        "exact_target_span": True,
    }


def _build_block_replace_step_for_preview_region(
    *,
    path: str,
    preview_window: dict[str, object],
) -> dict[str, object] | None:
    current_start_line = _int_value(preview_window.get("line_offset"), 0) + 1
    current_line_count = max(0, _int_value(preview_window.get("current_line_count"), 0))
    if current_start_line <= 0 or current_line_count <= 0:
        return None
    target_start_line = _int_value(preview_window.get("target_line_offset"), 0) + 1
    if target_start_line <= 0:
        return None
    after_lines = str(preview_window.get("target_content", "")).splitlines()
    current_end_line = current_start_line + current_line_count - 1
    target_end_line = target_start_line + len(after_lines) - 1
    replacement = {
        "start_line": current_start_line,
        "end_line": current_end_line,
        "before_lines": [""] * current_line_count,
        "after_lines": after_lines,
    }
    return {
        "path": path,
        "edit_kind": "block_replace",
        "replacement": replacement,
        "current_start_line": current_start_line,
        "current_end_line": current_end_line,
        "target_start_line": target_start_line,
        "target_end_line": max(target_start_line - 1, target_end_line),
        "line_delta": len(after_lines) - current_line_count,
        "edit_score": _edit_candidate_score("block_replace", replacement=replacement),
    }


def _overlap_block_replace_candidates_for_component(
    *,
    path: str,
    preview_windows: list[dict[str, object]],
    component_windows: list[dict[str, object]],
    available_window_count: int,
    component_id: int,
) -> list[dict[str, object]]:
    component_alias_pairs = _preview_overlap_alias_pairs(component_windows)
    component_conflicting_alias_pairs: set[tuple[int, int]] = set()
    candidates: list[dict[str, object]] = []
    for subset_size in range(2, len(component_windows) + 1):
        for subset in combinations(component_windows, subset_size):
            ordered_subset = sorted(subset, key=_window_subset_sort_key)
            if not _window_subset_has_overlap(ordered_subset):
                continue
            candidate_alias_pairs = _preview_overlap_alias_pairs(ordered_subset)
            merged_window = _merge_preview_window_subset(ordered_subset)
            recovered_conflicting_alias = False
            if merged_window is None:
                merged_window = _recover_conflicting_overlap_preview_window(ordered_subset)
                recovered_conflicting_alias = merged_window is not None
                if merged_window is None and candidate_alias_pairs:
                    component_conflicting_alias_pairs.update(candidate_alias_pairs)
            if merged_window is None:
                continue
            step = _derive_block_replace_step(
                path=path,
                baseline_content=str(merged_window.get("baseline_content", "")),
                target_content=str(merged_window.get("target_content", "")),
            )
            if step is None:
                continue
            step = _offset_preview_step(
                step,
                line_offset=_int_value(merged_window.get("line_offset"), 0),
                target_line_offset=_int_value(merged_window.get("target_line_offset"), 0),
            )
            command = _render_structured_edit_command(step)
            if not command:
                continue
            covered_window_indices = sorted(
                {
                    *_covered_preview_window_indices(merged_window, preview_windows),
                    *{
                        _int_value(item.get("window_index"), -1)
                        for item in ordered_subset
                        if _int_value(item.get("window_index"), -1) >= 0
                    },
                }
            )
            exact_window_count = sum(
                1
                for item in ordered_subset
                if bool((item.get("window") or {}).get("exact_target_span", False))
            )
            inexact_window_count = len(ordered_subset) - exact_window_count
            overlap_alias_pair_count = _preview_overlap_alias_pair_count(ordered_subset)
            overlap_alias_window_count = len(
                {
                    _int_value(item.get("window_index"), -1)
                    for item in ordered_subset
                    if _int_value(item.get("window_index"), -1) >= 0
                }
            ) if overlap_alias_pair_count > 0 else 0
            current_span_line_count = max(0, int(merged_window.get("current_line_count", 0)))
            target_span_line_count = max(0, int(merged_window.get("target_line_count", 0)))
            candidates.append(
                {
                    "command": command,
                    "proposal_source": "structured_edit:block_replace",
                    "edit_kind": "block_replace",
                    "edit_source": (
                        "workspace_preview_range"
                        if any(bool((item.get("window") or {}).get("truncated", False)) for item in ordered_subset)
                        else "workspace_preview"
                    ),
                    "window_index": None,
                    "window_indices": [int(item["window_index"]) for item in ordered_subset],
                    "edit_window_count": len(ordered_subset),
                    "covered_window_indices": covered_window_indices,
                    "covered_window_count": len(covered_window_indices),
                    "available_window_count": available_window_count,
                    "retained_window_count": max(
                        _int_value((item.get("window") or {}).get("retained_window_count"), available_window_count)
                        for item in ordered_subset
                    ),
                    "total_window_count": max(
                        _int_value((item.get("window") or {}).get("total_window_count"), available_window_count)
                        for item in ordered_subset
                    ),
                    "partial_window_coverage": any(
                        bool((item.get("window") or {}).get("partial_window_coverage", False))
                        for item in ordered_subset
                    ),
                    "exact_target_span": bool(merged_window.get("exact_target_span", False)),
                    "precision_penalty": _preview_step_precision_penalty(
                        step,
                        exact_target_span=bool(merged_window.get("exact_target_span", False)),
                    ),
                    "exact_window_count": exact_window_count,
                    "inexact_window_count": inexact_window_count,
                    "overlap_alias_pair_count": overlap_alias_pair_count,
                    "overlap_alias_window_count": overlap_alias_window_count,
                    "safe_inexact_window_count": inexact_window_count if overlap_alias_pair_count > 0 else 0,
                    "recovered_conflicting_alias": recovered_conflicting_alias,
                    "_overlap_component_alias_pairs": candidate_alias_pairs,
                    "overlap_component_alias_pair_count": len(component_alias_pairs),
                    "current_span_line_count": current_span_line_count,
                    "target_span_line_count": target_span_line_count,
                    "span_line_count": max(current_span_line_count, target_span_line_count),
                    "overlap_component_id": component_id,
                    "overlap_component_window_count": len(component_windows),
                    "step": step,
                }
            )
    ranked = sorted(candidates, key=_overlap_block_replace_rank_key)
    deduped: list[dict[str, object]] = []
    seen_signatures: set[tuple[str, tuple[int, ...]]] = set()
    for candidate in ranked:
        signature = _overlap_block_replace_signature(candidate)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        deduped.append(candidate)
        if len(deduped) >= 3:
            break
    component_candidate_count = len(deduped)
    frontier_alias_pair_count = max(
        (_int_value(candidate.get("overlap_alias_pair_count"), 0) for candidate in deduped),
        default=0,
    )
    frontier_alias_candidates = [
        candidate
        for candidate in deduped
        if _int_value(candidate.get("overlap_alias_pair_count"), 0) == frontier_alias_pair_count
    ]
    frontier_covered_window_count = max(
        (_int_value(candidate.get("covered_window_count"), 0) for candidate in frontier_alias_candidates),
        default=0,
    )
    frontier_covered_candidates = [
        candidate
        for candidate in frontier_alias_candidates
        if _int_value(candidate.get("covered_window_count"), 0) == frontier_covered_window_count
    ]
    frontier_exact_window_count = max(
        (_int_value(candidate.get("exact_window_count"), 0) for candidate in frontier_covered_candidates),
        default=0,
    )
    frontier_exact_candidates = [
        candidate
        for candidate in frontier_covered_candidates
        if _int_value(candidate.get("exact_window_count"), 0) == frontier_exact_window_count
    ]
    frontier_min_inexact_window_count = min(
        (_int_value(candidate.get("inexact_window_count"), 0) for candidate in frontier_exact_candidates),
        default=0,
    )
    frontier_inexact_candidates = [
        candidate
        for candidate in frontier_exact_candidates
        if _int_value(candidate.get("inexact_window_count"), 0) == frontier_min_inexact_window_count
    ]
    frontier_min_span_line_count = min(
        (
            _int_value(
                candidate.get("span_line_count"),
                _int_value(candidate.get("covered_window_count"), 1),
            )
            for candidate in frontier_inexact_candidates
        ),
        default=0,
    )
    component_conflicting_alias_pair_count = len(component_conflicting_alias_pairs)
    for candidate in deduped:
        candidate_alias_pairs = candidate.pop("_overlap_component_alias_pairs", set())
        unresolved_alias_pair_count = len(component_alias_pairs - candidate_alias_pairs)
        frontier_gap = 0.0
        frontier_gap += max(
            0,
            frontier_alias_pair_count - _int_value(candidate.get("overlap_alias_pair_count"), 0),
        ) * 8.0
        frontier_gap += max(
            0,
            frontier_covered_window_count - _int_value(candidate.get("covered_window_count"), 0),
        ) * 3.0
        frontier_gap += max(
            0,
            frontier_exact_window_count - _int_value(candidate.get("exact_window_count"), 0),
        ) * 4.0
        frontier_gap += max(
            0,
            _int_value(candidate.get("inexact_window_count"), 0) - frontier_min_inexact_window_count,
        ) * 3.0
        frontier_gap += max(
            0,
            _int_value(
                candidate.get("span_line_count"),
                _int_value(candidate.get("covered_window_count"), 1),
            )
            - frontier_min_span_line_count,
        ) * 0.5
        candidate["overlap_component_candidate_count"] = component_candidate_count
        candidate["overlap_component_unresolved_alias_pair_count"] = unresolved_alias_pair_count
        candidate["overlap_component_conflicting_alias_pair_count"] = (
            component_conflicting_alias_pair_count
        )
        candidate["overlap_component_frontier_gap"] = frontier_gap
    return deduped


def _overlap_window_components(
    span_windows: list[dict[str, object]],
) -> list[list[dict[str, object]]]:
    components: list[list[dict[str, object]]] = []
    remaining = list(span_windows)
    while remaining:
        seed = remaining.pop(0)
        component = [seed]
        queue = [seed]
        while queue:
            current = queue.pop(0)
            next_remaining: list[dict[str, object]] = []
            for candidate in remaining:
                if _preview_windows_share_overlap_component(current, candidate):
                    component.append(candidate)
                    queue.append(candidate)
                else:
                    next_remaining.append(candidate)
            remaining = next_remaining
        components.append(sorted(component, key=_window_subset_sort_key))
    return components


def _preview_windows_share_overlap_component(
    left: dict[str, object],
    right: dict[str, object],
) -> bool:
    left_current = (int(left["current_start"]), int(left["current_end"]))
    right_current = (int(right["current_start"]), int(right["current_end"]))
    left_target = (int(left["target_start"]), int(left["target_end"]))
    right_target = (int(right["target_start"]), int(right["target_end"]))
    return _spans_overlap(left_current, right_current) or _spans_overlap(left_target, right_target)


def _preview_has_unrecovered_overlap_component(
    *,
    preview_windows: list[dict[str, object]],
    overlap_block_replaces: list[dict[str, object]],
) -> bool:
    span_windows: list[dict[str, object]] = []
    for preview_window in preview_windows:
        current_span = _preview_window_current_span(preview_window)
        target_span = _preview_window_target_span(preview_window)
        if current_span is None or target_span is None:
            continue
        span_windows.append(
            {
                "window": preview_window,
                "current_start": current_span[0],
                "current_end": current_span[1],
                "target_start": target_span[0],
                "target_end": target_span[1],
                "window_index": _int_value(preview_window.get("window_index"), 0),
            }
        )
    if len(span_windows) < 2:
        return False
    best_component_unresolved_alias_counts: dict[int, int] = {}
    for proposal in overlap_block_replaces:
        component_id = _int_value(proposal.get("overlap_component_id"), -1)
        if component_id < 0:
            continue
        unresolved_alias_pair_count = max(
            0,
            _int_value(proposal.get("overlap_component_unresolved_alias_pair_count"), 0),
        )
        best_component_unresolved_alias_counts[component_id] = min(
            best_component_unresolved_alias_counts.get(component_id, unresolved_alias_pair_count),
            unresolved_alias_pair_count,
        )
    for component_id, component_windows in enumerate(_overlap_window_components(span_windows)):
        if len(component_windows) < 2:
            continue
        if _preview_overlap_alias_pair_count(component_windows) <= 0:
            continue
        if best_component_unresolved_alias_counts.get(component_id, 1) > 0:
            return True
    return False


def _annotate_preview_block_proposal_families(proposals: list[dict[str, object]]) -> None:
    preview_block_proposals = [
        proposal
        for proposal in proposals
        if str(proposal.get("edit_source", "")).strip() in {"workspace_preview", "workspace_preview_range"}
        and str(proposal.get("edit_kind", "")).strip() == "block_replace"
    ]
    localized_exact_block_windows = {
        _int_value(proposal.get("window_index"), -1)
        for proposal in preview_block_proposals
        if _int_value(proposal.get("window_index"), -1) >= 0
        and _int_value(proposal.get("edit_window_count"), 1) == 1
        and bool(proposal.get("exact_target_span", False))
    }
    block_alternative_count = len(preview_block_proposals)
    block_exact_alternative_count = sum(
        1 for proposal in preview_block_proposals if bool(proposal.get("exact_target_span", False))
    )
    block_max_covered_window_count = max(
        (_int_value(proposal.get("covered_window_count"), 0) for proposal in preview_block_proposals),
        default=0,
    )
    block_max_exact_window_count = max(
        (_int_value(proposal.get("exact_window_count"), 0) for proposal in preview_block_proposals),
        default=0,
    )
    block_min_precision_penalty = min(
        (_int_value(proposal.get("precision_penalty"), 0) for proposal in preview_block_proposals),
        default=0,
    )
    block_min_edit_window_count = min(
        (_int_value(proposal.get("edit_window_count"), 1) for proposal in preview_block_proposals),
        default=1,
    )
    block_frontier_max_exact_window_count = max(
        (_int_value(proposal.get("exact_window_count"), 0) for proposal in preview_block_proposals),
        default=0,
    )
    frontier_candidates = [
        proposal
        for proposal in preview_block_proposals
        if _int_value(proposal.get("exact_window_count"), 0) == block_frontier_max_exact_window_count
    ]
    block_frontier_min_inexact_window_count = min(
        (_int_value(proposal.get("inexact_window_count"), 0) for proposal in frontier_candidates),
        default=0,
    )
    frontier_min_inexact_candidates = [
        proposal
        for proposal in frontier_candidates
        if _int_value(proposal.get("inexact_window_count"), 0) == block_frontier_min_inexact_window_count
    ]
    block_frontier_min_span_line_count = min(
        (
            max(
                _int_value(proposal.get("covered_window_count"), 1),
                _int_value(proposal.get("span_line_count"), _int_value(proposal.get("covered_window_count"), 1)),
            )
            for proposal in frontier_min_inexact_candidates
        ),
        default=0,
    )
    same_span_block_groups: dict[tuple[int, int, int, int], list[dict[str, object]]] = {}
    for proposal in preview_block_proposals:
        span_signature = _preview_block_proposal_span_signature(proposal)
        if span_signature is None:
            continue
        same_span_block_groups.setdefault(span_signature, []).append(proposal)
    bridge_run_proposals = [
        proposal
        for proposal in preview_block_proposals
        if bool(proposal.get("bridged_hidden_gap_region_block", False))
    ]
    bridge_run_alternative_count = len(bridge_run_proposals)
    bridge_run_partial_alternative_count = sum(
        1 for proposal in bridge_run_proposals if bool(proposal.get("partial_window_coverage", False))
    )
    bridge_run_max_covered_window_count = max(
        (_int_value(proposal.get("covered_window_count"), 0) for proposal in bridge_run_proposals),
        default=0,
    )
    bridge_run_frontier_min_partial_window_coverage = min(
        (1 if bool(proposal.get("partial_window_coverage", False)) else 0 for proposal in bridge_run_proposals),
        default=1,
    )
    bridge_run_frontier_candidates = [
        proposal
        for proposal in bridge_run_proposals
        if (1 if bool(proposal.get("partial_window_coverage", False)) else 0)
        == bridge_run_frontier_min_partial_window_coverage
    ]
    bridge_run_frontier_max_covered_window_count = max(
        (_int_value(proposal.get("covered_window_count"), 0) for proposal in bridge_run_frontier_candidates),
        default=0,
    )
    bridge_run_frontier_candidates = [
        proposal
        for proposal in bridge_run_frontier_candidates
        if _int_value(proposal.get("covered_window_count"), 0) == bridge_run_frontier_max_covered_window_count
    ]
    bridge_run_frontier_max_exact_window_count = max(
        (_int_value(proposal.get("exact_window_count"), 0) for proposal in bridge_run_frontier_candidates),
        default=0,
    )
    bridge_run_frontier_candidates = [
        proposal
        for proposal in bridge_run_frontier_candidates
        if _int_value(proposal.get("exact_window_count"), 0) == bridge_run_frontier_max_exact_window_count
    ]
    bridge_run_frontier_min_inexact_window_count = min(
        (_int_value(proposal.get("inexact_window_count"), 0) for proposal in bridge_run_frontier_candidates),
        default=0,
    )
    bridge_run_frontier_candidates = [
        proposal
        for proposal in bridge_run_frontier_candidates
        if _int_value(proposal.get("inexact_window_count"), 0) == bridge_run_frontier_min_inexact_window_count
    ]
    bridge_run_frontier_min_span_line_count = min(
        (
            max(
                _int_value(proposal.get("covered_window_count"), 1),
                _int_value(proposal.get("span_line_count"), _int_value(proposal.get("covered_window_count"), 1)),
            )
            for proposal in bridge_run_frontier_candidates
        ),
        default=0,
    )
    bridge_run_frontier_max_segment_count = max(
        (_bridge_run_segment_count(proposal) for proposal in bridge_run_frontier_candidates),
        default=0,
    )
    bridge_run_exact_localized_alternatives = [
        proposal
        for proposal in preview_block_proposals
        if not bool(proposal.get("bridged_hidden_gap_region_block", False))
        and bool(proposal.get("exact_target_span", False))
        and _int_value(proposal.get("inexact_window_count"), 0) <= 0
        and _int_value(proposal.get("exact_window_count"), 0)
        >= max(1, _int_value(proposal.get("covered_window_count"), 1))
    ]
    bridge_run_exact_localized_alternative_count = len(bridge_run_exact_localized_alternatives)
    bridge_run_exact_localized_min_covered_window_count = min(
        (_int_value(proposal.get("covered_window_count"), 1) for proposal in bridge_run_exact_localized_alternatives),
        default=0,
    )
    bridge_run_exact_localized_min_span_line_count = min(
        (
            max(
                _int_value(proposal.get("covered_window_count"), 1),
                _int_value(proposal.get("span_line_count"), _int_value(proposal.get("covered_window_count"), 1)),
            )
            for proposal in bridge_run_exact_localized_alternatives
        ),
        default=0,
    )
    for proposal in proposals:
        window_index = _int_value(proposal.get("window_index"), -1)
        proposal["same_window_exact_block_alternative"] = window_index in localized_exact_block_windows
        proposal["localized_exact_block_alternative_count"] = len(localized_exact_block_windows)
        proposal["block_alternative_count"] = block_alternative_count
        proposal["block_exact_alternative_count"] = block_exact_alternative_count
        proposal["block_max_covered_window_count"] = block_max_covered_window_count
        proposal["block_max_exact_window_count"] = block_max_exact_window_count
        proposal["block_min_precision_penalty"] = block_min_precision_penalty
        proposal["block_min_edit_window_count"] = block_min_edit_window_count
        proposal["block_frontier_max_exact_window_count"] = block_frontier_max_exact_window_count
        proposal["block_frontier_min_inexact_window_count"] = block_frontier_min_inexact_window_count
        proposal["block_frontier_min_span_line_count"] = block_frontier_min_span_line_count
        same_span_alternatives = (
            same_span_block_groups.get(_preview_block_proposal_span_signature(proposal) or (), [])
            if str(proposal.get("edit_kind", "")).strip() == "block_replace"
            else []
        )
        same_span_quality_score = _preview_block_proposal_proof_quality_score(proposal)
        same_span_max_quality_score = max(
            (
                _preview_block_proposal_proof_quality_score(candidate)
                for candidate in same_span_alternatives
            ),
            default=same_span_quality_score,
        )
        proposal["same_span_block_alternative_count"] = len(same_span_alternatives)
        proposal["same_span_block_quality_score"] = same_span_quality_score
        proposal["same_span_block_max_quality_score"] = same_span_max_quality_score
        proposal["same_span_block_quality_gap"] = max(
            0,
            same_span_max_quality_score - same_span_quality_score,
        )
        proposal["same_span_block_proof_frontier"] = (
            len(same_span_alternatives) >= 2
            and same_span_max_quality_score > 0
            and same_span_quality_score >= same_span_max_quality_score
        )
        proposal["bridge_run_alternative_count"] = bridge_run_alternative_count
        proposal["bridge_run_partial_alternative_count"] = bridge_run_partial_alternative_count
        proposal["bridge_run_exact_localized_alternative_count"] = (
            bridge_run_exact_localized_alternative_count
        )
        proposal["bridge_run_max_covered_window_count"] = bridge_run_max_covered_window_count
        if bool(proposal.get("bridged_hidden_gap_region_block", False)):
            proposal["bridge_run_frontier_gap"] = _bridge_run_frontier_gap(
                proposal,
                frontier_min_partial_window_coverage=bridge_run_frontier_min_partial_window_coverage,
                frontier_max_covered_window_count=bridge_run_frontier_max_covered_window_count,
                frontier_max_exact_window_count=bridge_run_frontier_max_exact_window_count,
                frontier_min_inexact_window_count=bridge_run_frontier_min_inexact_window_count,
                frontier_min_span_line_count=bridge_run_frontier_min_span_line_count,
                frontier_max_segment_count=bridge_run_frontier_max_segment_count,
            )
            proposal["bridge_run_exact_localized_gap"] = _bridge_run_exact_localized_gap(
                proposal,
                localized_exact_min_covered_window_count=bridge_run_exact_localized_min_covered_window_count,
                localized_exact_min_span_line_count=bridge_run_exact_localized_min_span_line_count,
            )
        else:
            proposal["bridge_run_frontier_gap"] = 0.0
            proposal["bridge_run_exact_localized_gap"] = 0.0


def _prune_same_span_preview_block_proposals(
    proposals: list[dict[str, object]],
) -> list[dict[str, object]]:
    same_span_groups: dict[tuple[int, int, int, int], list[tuple[int, dict[str, object]]]] = {}
    for index, proposal in enumerate(proposals):
        if str(proposal.get("edit_source", "")).strip() not in {
            "workspace_preview",
            "workspace_preview_range",
        }:
            continue
        if str(proposal.get("edit_kind", "")).strip() != "block_replace":
            continue
        span_signature = _preview_block_proposal_span_signature(proposal)
        if span_signature is None:
            continue
        same_span_groups.setdefault(span_signature, []).append((index, proposal))
    if not same_span_groups:
        return proposals
    pruned_indices: set[int] = set()
    for candidates in same_span_groups.values():
        if len(candidates) < 2:
            continue
        max_quality_score = max(
            _preview_block_proposal_proof_quality_score(candidate)
            for _, candidate in candidates
        )
        if max_quality_score <= 0:
            continue
        frontier_candidates = [
            (index, candidate)
            for index, candidate in candidates
            if _preview_block_proposal_proof_quality_score(candidate) >= max_quality_score
        ]
        best_index, _ = min(
            frontier_candidates,
            key=lambda item: (
                _same_span_preview_block_proposal_rank_key(item[1]),
                item[0],
            ),
        )
        for index, _ in candidates:
            if index != best_index:
                pruned_indices.add(index)
    if not pruned_indices:
        return proposals
    return [
        proposal
        for index, proposal in enumerate(proposals)
        if index not in pruned_indices
    ]


def _prune_cross_family_preview_bounded_duplicates(
    proposals: list[dict[str, object]],
) -> list[dict[str, object]]:
    canonical_block_replace_commands: set[str] = set()
    canonical_block_replace_spans: set[tuple[int, int, int, int]] = set()
    for proposal in proposals:
        if str(proposal.get("proposal_source", "")).strip() != "structured_edit:block_replace":
            continue
        if str(proposal.get("edit_source", "")).strip() not in {
            "workspace_preview",
            "workspace_preview_range",
        }:
            continue
        command = str(proposal.get("command", "")).strip()
        if command:
            canonical_block_replace_commands.add(command)
        span_signature = _preview_structured_edit_proposal_span_signature(proposal)
        if span_signature is not None:
            canonical_block_replace_spans.add(span_signature)
    if not canonical_block_replace_commands and not canonical_block_replace_spans:
        return proposals
    deduped: list[dict[str, object]] = []
    for proposal in proposals:
        proposal_source = str(proposal.get("proposal_source", "")).strip()
        if proposal_source == "structured_edit:block_replace":
            deduped.append(proposal)
            continue
        if not proposal_source.startswith("structured_edit:"):
            deduped.append(proposal)
            continue
        if str(proposal.get("edit_source", "")).strip() not in {
            "workspace_preview",
            "workspace_preview_range",
        }:
            deduped.append(proposal)
            continue
        command = str(proposal.get("command", "")).strip()
        span_signature = _preview_structured_edit_proposal_span_signature(proposal)
        if command and command in canonical_block_replace_commands:
            continue
        if span_signature is not None and span_signature in canonical_block_replace_spans:
            continue
        deduped.append(proposal)
    return deduped


def _bridge_run_segment_count(proposal: dict[str, object]) -> int:
    return max(
        0,
        _int_value(
            proposal.get("bridge_run_segment_count"),
            proposal.get("hidden_gap_bridge_count"),
        ),
    )


def _bridge_run_frontier_gap(
    proposal: dict[str, object],
    *,
    frontier_min_partial_window_coverage: int,
    frontier_max_covered_window_count: int,
    frontier_max_exact_window_count: int,
    frontier_min_inexact_window_count: int,
    frontier_min_span_line_count: int,
    frontier_max_segment_count: int,
) -> float:
    partial_window_coverage = 1 if bool(proposal.get("partial_window_coverage", False)) else 0
    covered_window_count = max(0, _int_value(proposal.get("covered_window_count"), 0))
    exact_window_count = max(0, _int_value(proposal.get("exact_window_count"), 0))
    inexact_window_count = max(0, _int_value(proposal.get("inexact_window_count"), 0))
    span_line_count = max(
        covered_window_count,
        _int_value(proposal.get("span_line_count"), covered_window_count),
    )
    segment_count = _bridge_run_segment_count(proposal)
    gap = 0.0
    gap += max(0, partial_window_coverage - frontier_min_partial_window_coverage) * 6.0
    gap += max(0, frontier_max_covered_window_count - covered_window_count) * 3.0
    gap += max(0, frontier_max_exact_window_count - exact_window_count) * 2.0
    gap += max(0, inexact_window_count - frontier_min_inexact_window_count) * 3.0
    gap += max(0, span_line_count - frontier_min_span_line_count) * 0.5
    gap += max(0, frontier_max_segment_count - segment_count) * 0.5
    return gap


def _bridge_run_exact_localized_gap(
    proposal: dict[str, object],
    *,
    localized_exact_min_covered_window_count: int,
    localized_exact_min_span_line_count: int,
) -> float:
    if localized_exact_min_covered_window_count <= 0:
        return 0.0
    covered_window_count = max(1, _int_value(proposal.get("covered_window_count"), 1))
    exact_window_count = max(0, _int_value(proposal.get("exact_window_count"), 0))
    inexact_window_count = max(0, _int_value(proposal.get("inexact_window_count"), 0))
    span_line_count = max(
        covered_window_count,
        _int_value(proposal.get("span_line_count"), covered_window_count),
    )
    gap = 0.0
    if bool(proposal.get("partial_window_coverage", False)):
        gap += 4.0
    gap += max(0, covered_window_count - localized_exact_min_covered_window_count) * 1.5
    gap += max(0, covered_window_count - exact_window_count) * 1.5
    gap += inexact_window_count * 3.0
    gap += max(0, span_line_count - localized_exact_min_span_line_count) * 0.25
    return gap


def _merge_preview_window_subset(
    ordered_subset: list[dict[str, object]],
) -> dict[str, object] | None:
    if len(ordered_subset) < 2:
        return None
    current_start = min(int(item["current_start"]) for item in ordered_subset)
    current_end = max(int(item["current_end"]) for item in ordered_subset)
    current_lines = _merge_preview_subset_line_map(
        ordered_subset,
        content_key="baseline_content",
        span_key="current",
    )
    if current_lines is None:
        return None
    target_lines = _merge_preview_subset_line_map(
        ordered_subset,
        content_key="target_content",
        span_key="target",
    )
    if target_lines is None:
        return None
    if any(line_number not in current_lines for line_number in range(current_start, current_end + 1)):
        return None
    if target_lines:
        target_start = min(target_lines)
        target_end = max(target_lines)
        if any(line_number not in target_lines for line_number in range(target_start, target_end + 1)):
            return None
        merged_target_content = "".join(target_lines[line_number] for line_number in range(target_start, target_end + 1))
    else:
        target_start = min(int(item["target_start"]) for item in ordered_subset)
        target_end = target_start - 1
        merged_target_content = ""
    return {
        "baseline_content": "".join(current_lines[line_number] for line_number in range(current_start, current_end + 1)),
        "target_content": merged_target_content,
        "line_offset": current_start - 1,
        "target_line_offset": target_start - 1,
        "current_line_count": max(0, current_end - current_start + 1),
        "target_line_count": max(0, target_end - target_start + 1),
        "truncated": any(bool((item.get("window") or {}).get("truncated", False)) for item in ordered_subset),
        "exact_target_span": all(
            bool((item.get("window") or {}).get("exact_target_span", False))
            for item in ordered_subset
        ),
    }


def _shared_anchor_exact_region_subset_is_safe(
    ordered_subset: list[dict[str, object]],
) -> bool:
    if len(ordered_subset) < 2:
        return False
    if not all(bool((item.get("window") or {}).get("exact_target_span", False)) for item in ordered_subset):
        return False
    ordered_subset = sorted(ordered_subset, key=_window_subset_sort_key)
    has_shared_anchor_overlap = False
    requires_full_target_content = False
    for index, (left, right) in enumerate(zip(ordered_subset, ordered_subset[1:])):
        pair_kind = _shared_anchor_exact_pair_kind(left, right)
        if pair_kind is not None:
            has_shared_anchor_overlap = True
            if pair_kind in {
                "same_anchor_insert_delete",
                "adjacent_delete_insert",
            }:
                requires_full_target_content = True
            continue
        if _preview_windows_share_overlap_component(left, right):
            return False
    if not has_shared_anchor_overlap:
        return False
    current_lines = _merge_preview_subset_line_map(
        ordered_subset,
        content_key="baseline_content",
        span_key="current",
    )
    if current_lines is None:
        return False
    current_start = min(int(item["current_start"]) for item in ordered_subset)
    current_end = max(int(item["current_end"]) for item in ordered_subset)
    if any(line_number not in current_lines for line_number in range(current_start, current_end + 1)):
        return False
    synthesized_target_region = _synthesize_exact_shared_anchor_target_region(ordered_subset)
    if synthesized_target_region is None:
        return False
    full_target_content = _shared_preview_full_target_content(ordered_subset)
    if requires_full_target_content and not full_target_content:
        return False
    if not full_target_content:
        return True
    target_start = _int_value(synthesized_target_region.get("target_start"), 0)
    target_end = _int_value(
        synthesized_target_region.get("target_end"),
        target_start - 1,
    )
    if target_start <= 0:
        return False
    if target_end < target_start:
        return str(synthesized_target_region.get("target_content", "")) == ""
    expected_target_lines = _expected_target_lines_for_span(
        full_target_content,
        start_line=target_start,
        end_line=target_end,
    )
    if expected_target_lines is None:
        return False
    if _subset_exact_windows_match_expected_target(
        ordered_subset,
        expected_target_lines,
        target_start,
    ):
        return True
    return str(synthesized_target_region.get("target_content", "")) == "".join(expected_target_lines)


def _shared_anchor_exact_pair_kinds(
    ordered_subset: list[dict[str, object]],
) -> list[str]:
    pair_kinds: list[str] = []
    for left, right in zip(ordered_subset, ordered_subset[1:]):
        pair_kind = _shared_anchor_exact_pair_kind(left, right)
        if pair_kind is not None:
            pair_kinds.append(pair_kind)
    return pair_kinds


def _shared_anchor_exact_neighbor_count(
    ordered_subset: list[dict[str, object]],
) -> int:
    if len(ordered_subset) < 3:
        return 0
    participating_indices: set[int] = set()
    for index, (left, right) in enumerate(zip(ordered_subset, ordered_subset[1:])):
        if _shared_anchor_exact_pair_kind(left, right) is None:
            continue
        participating_indices.add(index)
        participating_indices.add(index + 1)
    if not participating_indices:
        return 0
    return max(0, len(ordered_subset) - len(participating_indices))


def _shared_anchor_exact_core_count(
    ordered_subset: list[dict[str, object]],
) -> int:
    if len(ordered_subset) < 2:
        return 0
    participating_indices: set[int] = set()
    for index, (left, right) in enumerate(zip(ordered_subset, ordered_subset[1:])):
        if _shared_anchor_exact_pair_kind(left, right) is None:
            continue
        participating_indices.add(index)
        participating_indices.add(index + 1)
    if not participating_indices:
        return 0
    ordered_indices = sorted(participating_indices)
    core_count = 1
    for left_index, right_index in zip(ordered_indices, ordered_indices[1:]):
        if right_index != left_index + 1:
            core_count += 1
    return core_count


def _shared_anchor_exact_pair_kind(
    left: dict[str, object],
    right: dict[str, object],
) -> str | None:
    left_current = (int(left["current_start"]), int(left["current_end"]))
    right_current = (int(right["current_start"]), int(right["current_end"]))
    left_target = (int(left["target_start"]), int(left["target_end"]))
    right_target = (int(right["target_start"]), int(right["target_end"]))
    left_delta = _int_value((left.get("window") or {}).get("line_delta"), 0)
    right_delta = _int_value((right.get("window") or {}).get("line_delta"), 0)
    if (
        left_current[0] == left_current[1] == right_current[0] == right_current[1]
        and left_current[0] == right_current[0]
        and {left_delta, right_delta} == {0, 1}
        and left_target[0] <= right_target[0]
        and max(left_target[1], left_target[0]) >= right_target[0]
    ):
        return "same_anchor_insert_replace"
    if (
        left_current[0] == left_current[1] == right_current[0] == right_current[1]
        and left_current[0] == right_current[0]
        and {left_delta, right_delta} == {-1, 1}
        and left_target[0] <= right_target[0]
        and max(left_target[1], left_target[0]) >= right_target[0]
    ):
        return "same_anchor_insert_delete"
    if left_target[1] >= left_target[0]:
        return None
    if left_delta >= 0 or right_delta > 0:
        if (
            left_delta < 0
            and right_delta > 0
            and right_current[0] == left_current[1] + 1
            and right_target[0] == left_target[0]
        ):
            return "adjacent_delete_insert"
        return None
    if right_current[0] != left_current[1] + 1:
        return None
    if right_target[0] != left_target[0]:
        return None
    if right_delta < 0 and right_target[1] < right_target[0]:
        return "adjacent_delete_delete"
    if right_delta != 0:
        return None
    return "adjacent_delete_replace"


def _recover_exact_shared_anchor_preview_window(
    ordered_subset: list[dict[str, object]],
) -> dict[str, object] | None:
    if not _shared_anchor_exact_region_subset_is_safe(ordered_subset):
        return None
    if not all(bool((item.get("window") or {}).get("exact_target_span", False)) for item in ordered_subset):
        return None
    current_lines = _merge_preview_subset_line_map(
        ordered_subset,
        content_key="baseline_content",
        span_key="current",
    )
    if current_lines is None:
        return None
    current_start = min(int(item["current_start"]) for item in ordered_subset)
    current_end = max(int(item["current_end"]) for item in ordered_subset)
    if any(line_number not in current_lines for line_number in range(current_start, current_end + 1)):
        return None
    synthesized_target_region = _synthesize_exact_shared_anchor_target_region(ordered_subset)
    if synthesized_target_region is None:
        return None
    target_start = _int_value(synthesized_target_region.get("target_start"), 0)
    target_end = _int_value(synthesized_target_region.get("target_end"), target_start - 1)
    target_content = str(synthesized_target_region.get("target_content", ""))
    target_line_count = max(0, _int_value(synthesized_target_region.get("target_line_count"), 0))
    return {
        "baseline_content": "".join(
            current_lines[line_number] for line_number in range(current_start, current_end + 1)
        ),
        "target_content": target_content,
        "line_offset": current_start - 1,
        "target_line_offset": target_start - 1,
        "current_line_count": max(0, current_end - current_start + 1),
        "target_line_count": target_line_count,
        "truncated": any(bool((item.get("window") or {}).get("truncated", False)) for item in ordered_subset),
        "exact_target_span": True,
    }


def _visible_preview_subset_target_line_numbers(
    ordered_subset: list[dict[str, object]],
) -> set[int]:
    line_numbers: set[int] = set()
    for item in ordered_subset:
        target_start = int(item["target_start"])
        target_end = int(item["target_end"])
        if target_end < target_start:
            continue
        line_numbers.update(range(target_start, target_end + 1))
    return line_numbers


def _synthesize_exact_shared_anchor_target_region(
    ordered_subset: list[dict[str, object]],
) -> dict[str, object] | None:
    full_target_content = _shared_preview_full_target_content(ordered_subset)
    if full_target_content:
        current_start = min(int(item["current_start"]) for item in ordered_subset)
        current_end = max(int(item["current_end"]) for item in ordered_subset)
        current_line_count = max(0, current_end - current_start + 1)
        target_start = min(int(item["target_start"]) for item in ordered_subset)
        target_line_count = max(
            0,
            current_line_count
            + sum(_int_value((item.get("window") or {}).get("line_delta"), 0) for item in ordered_subset),
        )
        target_end = target_start + target_line_count - 1
        if target_line_count <= 0:
            return {
                "target_start": target_start,
                "target_end": target_start - 1,
                "target_content": "",
                "target_line_count": 0,
            }
        expected_target_lines = _expected_target_lines_for_span(
            full_target_content,
            start_line=target_start,
            end_line=target_end,
        )
        if expected_target_lines is None:
            return None
        return {
            "target_start": target_start,
            "target_end": target_end,
            "target_content": "".join(expected_target_lines),
            "target_line_count": len(expected_target_lines),
        }
    target_lines_by_number: dict[int, tuple[tuple[int, int, int, int], str]] = {}
    target_start_candidates: list[int] = []
    for position, item in enumerate(ordered_subset):
        window = item.get("window") or {}
        target_start = int(item["target_start"])
        target_end = int(item["target_end"])
        if target_start <= 0:
            return None
        target_start_candidates.append(target_start)
        if target_end < target_start:
            if str(window.get("target_content", "")):
                return None
            continue
        target_lines = str(window.get("target_content", "")).splitlines(keepends=True)
        if len(target_lines) != target_end - target_start + 1:
            return None
        specificity = (
            max(0, target_end - target_start + 1),
            abs(_int_value((window or {}).get("line_delta"), 0)),
            max(0, int(item["current_end"]) - int(item["current_start"]) + 1),
            position,
        )
        for offset, target_line in enumerate(target_lines):
            line_number = target_start + offset
            existing = target_lines_by_number.get(line_number)
            if existing is None or specificity < existing[0]:
                target_lines_by_number[line_number] = (specificity, target_line)
    if not target_start_candidates:
        return None
    if not target_lines_by_number:
        target_start = min(target_start_candidates)
        return {
            "target_start": target_start,
            "target_end": target_start - 1,
            "target_content": "",
            "target_line_count": 0,
        }
    target_start = min(target_lines_by_number)
    target_end = max(target_lines_by_number)
    if any(line_number not in target_lines_by_number for line_number in range(target_start, target_end + 1)):
        return None
    synthesized_target_lines = [
        target_lines_by_number[line_number][1] for line_number in range(target_start, target_end + 1)
    ]
    return {
        "target_start": target_start,
        "target_end": target_end,
        "target_content": "".join(synthesized_target_lines),
        "target_line_count": len(synthesized_target_lines),
    }


def _merge_preview_subset_line_map(
    ordered_subset: list[dict[str, object]],
    *,
    content_key: str,
    span_key: str,
) -> dict[int, str] | None:
    merged: dict[int, str] = {}
    for item in ordered_subset:
        window = item.get("window") or {}
        content = str(window.get(content_key, ""))
        if span_key == "current":
            start_line = int(item["current_start"])
            end_line = int(item["current_end"])
        else:
            start_line = int(item["target_start"])
            end_line = int(item["target_end"])
        lines = content.splitlines(keepends=True)
        expected_line_count = max(0, end_line - start_line + 1)
        if len(lines) != expected_line_count:
            return None
        for offset, line in enumerate(lines):
            line_number = start_line + offset
            existing = merged.get(line_number)
            if existing is not None and existing != line:
                return None
            merged[line_number] = line
    return merged


def _recover_conflicting_overlap_preview_window(
    ordered_subset: list[dict[str, object]],
) -> dict[str, object] | None:
    if len(ordered_subset) < 2:
        return None
    if _preview_overlap_alias_pair_count(ordered_subset) <= 0:
        return None
    current_lines = _merge_preview_subset_line_map(
        ordered_subset,
        content_key="baseline_content",
        span_key="current",
    )
    if current_lines is None:
        return None
    current_start = min(int(item["current_start"]) for item in ordered_subset)
    current_end = max(int(item["current_end"]) for item in ordered_subset)
    if any(line_number not in current_lines for line_number in range(current_start, current_end + 1)):
        return None
    line_deltas = {
        _int_value((item.get("window") or {}).get("line_delta"), 0)
        for item in ordered_subset
    }
    if line_deltas != {0}:
        return None
    exact_anchor_deltas = {
        int(item["target_start"]) - int(item["current_start"])
        for item in ordered_subset
        if bool((item.get("window") or {}).get("exact_target_span", False))
    }
    if len(exact_anchor_deltas) != 1:
        return None
    full_target_content = _shared_preview_full_target_content(ordered_subset)
    if not full_target_content:
        return None
    anchor_delta = next(iter(exact_anchor_deltas))
    target_start = current_start + anchor_delta
    target_end = current_end + anchor_delta
    expected_target_lines = _expected_target_lines_for_span(
        full_target_content,
        start_line=target_start,
        end_line=target_end,
    )
    if expected_target_lines is None:
        return None
    if not _subset_exact_windows_match_expected_target(ordered_subset, expected_target_lines, target_start):
        return None
    return {
        "baseline_content": "".join(current_lines[line_number] for line_number in range(current_start, current_end + 1)),
        "target_content": "".join(expected_target_lines),
        "line_offset": current_start - 1,
        "target_line_offset": target_start - 1,
        "current_line_count": max(0, current_end - current_start + 1),
        "target_line_count": len(expected_target_lines),
        "truncated": any(bool((item.get("window") or {}).get("truncated", False)) for item in ordered_subset),
        "exact_target_span": True,
        "recovered_conflicting_alias": True,
    }


def _shared_preview_full_target_content(ordered_subset: list[dict[str, object]]) -> str:
    full_target_contents = {
        str((item.get("window") or {}).get("full_target_content", ""))
        for item in ordered_subset
        if str((item.get("window") or {}).get("full_target_content", ""))
    }
    if len(full_target_contents) != 1:
        return ""
    return next(iter(full_target_contents))


def _expected_target_lines_for_span(
    full_target_content: str,
    *,
    start_line: int,
    end_line: int,
) -> list[str] | None:
    if start_line <= 0 or end_line < start_line:
        return None
    target_lines = full_target_content.splitlines(keepends=True)
    if end_line > len(target_lines):
        return None
    expected_target_lines = target_lines[start_line - 1 : end_line]
    if len(expected_target_lines) != end_line - start_line + 1:
        return None
    return expected_target_lines


def _subset_exact_windows_match_expected_target(
    ordered_subset: list[dict[str, object]],
    expected_target_lines: list[str],
    target_start: int,
) -> bool:
    for item in ordered_subset:
        window = item.get("window") or {}
        if not bool(window.get("exact_target_span", False)):
            continue
        window_target_lines = str(window.get("target_content", "")).splitlines(keepends=True)
        relative_start = int(item["target_start"]) - target_start
        relative_end = relative_start + len(window_target_lines)
        if relative_start < 0 or relative_end > len(expected_target_lines):
            return False
        if expected_target_lines[relative_start:relative_end] != window_target_lines:
            return False
    return True


def _overlap_block_replace_signature(candidate: dict[str, object]) -> tuple[str, tuple[int, ...]]:
    window_indices = candidate.get("window_indices", [])
    if not isinstance(window_indices, list):
        window_indices = []
    return (
        str(candidate.get("command", "")),
        tuple(_int_value(index, -1) for index in window_indices),
    )


def _overlap_block_replace_rank_key(
    candidate: dict[str, object],
) -> tuple[int, int, int, int, int, int, tuple[int, ...]]:
    step = candidate.get("step") or {}
    window_indices = candidate.get("window_indices", [])
    if not isinstance(window_indices, list):
        window_indices = []
    return (
        -_int_value(candidate.get("overlap_alias_pair_count"), 0),
        -_int_value(candidate.get("covered_window_count"), 0),
        _int_value(candidate.get("inexact_window_count"), 0),
        _int_value(candidate.get("precision_penalty"), 0),
        _int_value(candidate.get("span_line_count"), 0),
        -_int_value(candidate.get("edit_window_count"), 0),
        _int_value(step.get("edit_score"), 9999),
        tuple(_int_value(index, -1) for index in window_indices),
    )


def _preview_window_current_span(preview_window: dict[str, object]) -> tuple[int, int] | None:
    if bool(preview_window.get("explicit_current_span_proof", False)):
        line_start = max(1, _int_value(preview_window.get("line_start"), 1))
        line_end = max(line_start - 1, _int_value(preview_window.get("line_end"), line_start - 1))
        return line_start, line_end
    line_start = _int_value(preview_window.get("line_offset"), 0) + 1
    baseline_lines = str(preview_window.get("baseline_content", "")).splitlines()
    if line_start <= 0 or not baseline_lines:
        return None
    return line_start, line_start + len(baseline_lines) - 1


def _preview_window_target_span(preview_window: dict[str, object]) -> tuple[int, int] | None:
    if bool(preview_window.get("explicit_current_span_proof", False)):
        line_start = max(1, _int_value(preview_window.get("target_line_start"), 1))
        line_end = max(line_start - 1, _int_value(preview_window.get("target_line_end"), line_start - 1))
        return line_start, line_end
    line_start = _int_value(preview_window.get("target_line_offset"), 0) + 1
    if line_start <= 0:
        return None
    target_lines = str(preview_window.get("target_content", "")).splitlines()
    if not target_lines:
        return line_start, line_start - 1
    return line_start, line_start + len(target_lines) - 1


def _sorted_spans_cover_region(
    spans: list[tuple[int, int]],
    region_start: int,
    region_end: int,
) -> bool:
    normalized = sorted(
        [
            (
                max(region_start, max(1, int(start))),
                min(region_end, max(int(start) - 1, int(end))),
            )
            for start, end in spans
            if max(int(start) - 1, int(end)) >= int(start)
            and max(int(start), region_start) <= min(region_end, max(int(start) - 1, int(end)))
        ],
        key=lambda span: (span[0], span[1]),
    )
    if region_end < region_start:
        return False
    if not normalized or normalized[0][0] != region_start:
        return False
    covered_end = region_start - 1
    for start, end in normalized:
        if start > covered_end + 1:
            return False
        if end > covered_end:
            covered_end = end
        if covered_end >= region_end:
            return True
    return covered_end >= region_end


def _span_contains(outer: tuple[int, int], inner: tuple[int, int]) -> bool:
    if inner[1] < inner[0]:
        if outer[1] < outer[0]:
            return outer[0] == inner[0]
        return outer[0] <= inner[0] <= outer[1] + 1
    if outer[1] < outer[0]:
        return False
    return outer[0] <= inner[0] and outer[1] >= inner[1]


def _preview_window_lines_match(
    *,
    source_lines: list[str],
    candidate_lines: list[str],
    relative_offset: int,
) -> bool:
    if relative_offset < 0:
        return False
    end_offset = relative_offset + len(candidate_lines)
    if end_offset > len(source_lines):
        return False
    return source_lines[relative_offset:end_offset] == candidate_lines


def _preview_window_covers_window(source_window: dict[str, object], candidate_window: dict[str, object]) -> bool:
    source_current = _preview_window_current_span(source_window)
    candidate_current = _preview_window_current_span(candidate_window)
    source_target = _preview_window_target_span(source_window)
    candidate_target = _preview_window_target_span(candidate_window)
    if source_current is None or candidate_current is None or source_target is None or candidate_target is None:
        return False
    if not _span_contains(source_current, candidate_current):
        return False
    if not _span_contains(source_target, candidate_target):
        return False
    if bool(source_window.get("explicit_current_span_proof", False)) or bool(candidate_window.get("explicit_current_span_proof", False)):
        return True
    source_baseline_lines = str(source_window.get("baseline_content", "")).splitlines()
    candidate_baseline_lines = str(candidate_window.get("baseline_content", "")).splitlines()
    current_offset = candidate_current[0] - source_current[0]
    if not _preview_window_lines_match(
        source_lines=source_baseline_lines,
        candidate_lines=candidate_baseline_lines,
        relative_offset=current_offset,
    ):
        return False
    source_target_lines = str(source_window.get("target_content", "")).splitlines()
    candidate_target_lines = str(candidate_window.get("target_content", "")).splitlines()
    target_offset = candidate_target[0] - source_target[0]
    return _preview_window_lines_match(
        source_lines=source_target_lines,
        candidate_lines=candidate_target_lines,
        relative_offset=target_offset,
    )


def _covered_preview_window_indices(
    source_window: dict[str, object],
    preview_windows: list[dict[str, object]],
) -> list[int]:
    source_index = _int_value(source_window.get("window_index"), -1)
    covered: set[int] = set()
    for candidate_window in preview_windows:
        candidate_index = _int_value(candidate_window.get("window_index"), -1)
        if candidate_index < 0:
            continue
        if candidate_index == source_index or _preview_window_covers_window(source_window, candidate_window):
            covered.add(candidate_index)
    if source_index >= 0:
        covered.add(source_index)
    return sorted(covered)


def _proposal_window_indices(proposal: dict[str, object]) -> list[int]:
    raw_window_indices = proposal.get("window_indices", [])
    if isinstance(raw_window_indices, list) and raw_window_indices:
        return sorted(
            {
                _int_value(index, -1)
                for index in raw_window_indices
                if _int_value(index, -1) >= 0
            }
        )
    window_index = _int_value(proposal.get("window_index"), -1)
    if window_index >= 0:
        return [window_index]
    return []


def _preview_overlap_alias_pair_count(
    ordered_subset: list[dict[str, object]],
) -> int:
    return len(_preview_overlap_alias_pairs(ordered_subset))


def _preview_overlap_alias_pairs(
    ordered_subset: list[dict[str, object]],
) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for index, left in enumerate(ordered_subset):
        for right in ordered_subset[index + 1 :]:
            if not _preview_window_pair_has_ambiguous_alias(left, right):
                continue
            left_index = _int_value(left.get("window_index"), -1)
            right_index = _int_value(right.get("window_index"), -1)
            if left_index < 0 or right_index < 0:
                continue
            pairs.add((min(left_index, right_index), max(left_index, right_index)))
    return pairs


def _preview_window_pair_has_ambiguous_alias(
    left: dict[str, object],
    right: dict[str, object],
) -> bool:
    left_current = (int(left["current_start"]), int(left["current_end"]))
    right_current = (int(right["current_start"]), int(right["current_end"]))
    left_target = (int(left["target_start"]), int(left["target_end"]))
    right_target = (int(right["target_start"]), int(right["target_end"]))
    if not (_span_intersects(left_current, right_current) or _span_intersects(left_target, right_target)):
        return False
    left_exact = bool((left.get("window") or {}).get("exact_target_span", False))
    right_exact = bool((right.get("window") or {}).get("exact_target_span", False))
    return not (left_exact and right_exact)



def _spans_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    if left[1] < left[0] or right[1] < right[0]:
        return False
    return left[0] <= right[1] and right[0] <= left[1]


def _window_subset_has_overlap(ordered_subset: list[dict[str, object]]) -> bool:
    for index, item in enumerate(ordered_subset):
        current_span = (int(item["current_start"]), int(item["current_end"]))
        target_span = (int(item["target_start"]), int(item["target_end"]))
        for other in ordered_subset[index + 1 :]:
            other_current_span = (int(other["current_start"]), int(other["current_end"]))
            other_target_span = (int(other["target_start"]), int(other["target_end"]))
            if _spans_overlap(current_span, other_current_span) or _spans_overlap(target_span, other_target_span):
                return True
    return False


def _window_subset_sort_key(item: dict[str, object]) -> tuple[int, int, int, int, int]:
    return (
        int(item["current_start"]),
        int(item["current_end"]),
        int(item["target_start"]),
        int(item["target_end"]),
        int(item["window_index"]),
    )


def _best_non_overlapping_window_subset(
    span_proposals: list[dict[str, object]],
) -> list[dict[str, object]]:
    if len(span_proposals) < 2:
        return []
    best: list[dict[str, object]] = []
    for subset_size in range(2, len(span_proposals) + 1):
        for subset in combinations(span_proposals, subset_size):
            ordered_subset = sorted(subset, key=_window_subset_sort_key)
            if not _window_subset_is_non_overlapping(ordered_subset):
                continue
            if _window_subset_rank_key(ordered_subset) < _window_subset_rank_key(best):
                best = ordered_subset
    return best


def _window_subset_is_non_overlapping(
    ordered_subset: list[dict[str, object]],
) -> bool:
    for index, previous in enumerate(ordered_subset):
        for current in ordered_subset[index + 1 :]:
            if not _preview_multi_edit_pair_is_safe(previous, current):
                return False
    return True


def _window_subset_rank_key(
    subset: list[dict[str, object]],
) -> tuple[int, int, int, float, int, int, int, int, tuple[int, ...]]:
    if not subset:
        return (1_000_000, 0, 0, 0.0, 0, 0, 0, 0, 0, ())
    exact_window_count = sum(
        max(
            0,
            _int_value(
                (item.get("proposal") or {}).get("exact_window_count"),
                1 if bool((item.get("proposal") or {}).get("exact_target_span", False)) else 0,
            ),
        )
        for item in subset
    )
    inexact_window_count = sum(
        max(
            0,
            _int_value(
                (item.get("proposal") or {}).get("inexact_window_count"),
                0 if bool((item.get("proposal") or {}).get("exact_target_span", False)) else 1,
            ),
        )
        for item in subset
    )
    safe_inexact_window_count = sum(
        max(0, _int_value((item.get("proposal") or {}).get("safe_inexact_window_count"), 0))
        for item in subset
    )
    overlap_alias_pair_count = sum(
        max(0, _int_value((item.get("proposal") or {}).get("overlap_alias_pair_count"), 0))
        for item in subset
    )
    unresolved_alias_pair_count = sum(
        max(
            0,
            _int_value(
                (item.get("proposal") or {}).get("overlap_component_unresolved_alias_pair_count"),
                0,
            ),
        )
        for item in subset
    )
    frontier_gap = sum(
        max(0.0, _float_value((item.get("proposal") or {}).get("overlap_component_frontier_gap"), 0.0))
        for item in subset
    )
    covered_window_count = len(
        {
            _int_value(index, -1)
            for item in subset
            for index in ((item.get("proposal") or {}).get("covered_window_indices", []) or item.get("window_indices", []))
            if _int_value(index, -1) >= 0
        }
    )
    span_line_count = sum(
        max(
            _int_value((item.get("proposal") or {}).get("covered_window_count"), 1),
            _int_value(
                (item.get("proposal") or {}).get("span_line_count"),
                _int_value((item.get("proposal") or {}).get("covered_window_count"), 1),
            ),
        )
        for item in subset
    )
    if inexact_window_count <= 0:
        precision_class = 0
    elif (
        safe_inexact_window_count >= inexact_window_count
        and overlap_alias_pair_count > 0
        and unresolved_alias_pair_count <= 0
    ):
        precision_class = 0
    elif safe_inexact_window_count > 0 and unresolved_alias_pair_count <= 0:
        precision_class = 1
    elif safe_inexact_window_count > 0:
        precision_class = 2
    else:
        precision_class = 3
    return (
        precision_class,
        unresolved_alias_pair_count,
        -covered_window_count,
        frontier_gap,
        -exact_window_count,
        sum(
            _int_value((item.get("proposal") or {}).get("precision_penalty"), 0)
            for item in subset
        ),
        span_line_count,
        -sum(int((item["proposal"].get("step") or {}).get("edit_score", 0)) for item in subset),
        tuple(
            _int_value(index, -1)
            for item in subset
            for index in item.get("window_indices", [])
            if _int_value(index, -1) >= 0
        ),
    )


def _preview_multi_edit_pair_is_safe(left: dict[str, object], right: dict[str, object]) -> bool:
    left_current = (int(left["current_start"]), int(left["current_end"]))
    right_current = (int(right["current_start"]), int(right["current_end"]))
    left_target = (int(left["target_start"]), int(left["target_end"]))
    right_target = (int(right["target_start"]), int(right["target_end"]))
    current_overlaps = _span_intersects(left_current, right_current)
    target_overlaps = _span_intersects(left_target, right_target)
    if target_overlaps and not current_overlaps:
        return _shifted_delete_anchor_pair_is_safe(left, right)
    if target_overlaps:
        return False
    if not current_overlaps:
        return True
    return _same_anchor_multi_edit_pair_is_safe(left, right)


def _same_anchor_multi_edit_pair_is_safe(left: dict[str, object], right: dict[str, object]) -> bool:
    if not _preview_subset_item_exact_target_span(left):
        return False
    if not _preview_subset_item_exact_target_span(right):
        return False
    if int(left["current_start"]) != int(left["current_end"]) or int(right["current_start"]) != int(right["current_end"]):
        return False
    if int(left["current_start"]) != int(right["current_start"]):
        return False
    left_step = _preview_subset_item_step(left)
    right_step = _preview_subset_item_step(right)
    if left_step is None or right_step is None:
        return False
    left_kind = str(left_step.get("edit_kind", "")).strip()
    right_kind = str(right_step.get("edit_kind", "")).strip()
    allowed_kinds = {"line_insert", "token_replace", "token_insert", "token_delete", "line_replace"}
    if left_kind not in allowed_kinds or right_kind not in allowed_kinds:
        return False
    if left_kind != "line_insert" and _int_value(left_step.get("line_delta"), 0) != 0:
        return False
    if right_kind != "line_insert" and _int_value(right_step.get("line_delta"), 0) != 0:
        return False
    return True


def _shifted_delete_anchor_pair_is_safe(left: dict[str, object], right: dict[str, object]) -> bool:
    if not _preview_subset_item_exact_target_span(left):
        return False
    if not _preview_subset_item_exact_target_span(right):
        return False
    earlier, later = sorted((left, right), key=lambda item: (int(item["current_start"]), int(item["current_end"]), int(item["window_index"])))
    earlier_target = (int(earlier["target_start"]), int(earlier["target_end"]))
    later_target = (int(later["target_start"]), int(later["target_end"]))
    if earlier_target[1] >= earlier_target[0]:
        return False
    earlier_step = _preview_subset_item_step(earlier)
    if earlier_step is None:
        return False
    if _int_value(earlier_step.get("line_delta"), 0) >= 0:
        return False
    if int(later["current_start"]) != int(earlier["current_end"]) + 1:
        return False
    if later_target[0] != earlier_target[0]:
        return False
    return True


def _preview_subset_item_exact_target_span(item: dict[str, object]) -> bool:
    proposal = item.get("proposal")
    if isinstance(proposal, dict) and "exact_target_span" in proposal:
        return bool(proposal.get("exact_target_span", False))
    window = item.get("window")
    if isinstance(window, dict):
        return bool(window.get("exact_target_span", False))
    return False


def _preview_subset_item_step(item: dict[str, object]) -> dict[str, object] | None:
    step = item.get("step")
    if isinstance(step, dict):
        return step
    window = item.get("window")
    if not isinstance(window, dict):
        return None
    return _best_preview_edit_step(
        path=str(window.get("path", "")),
        baseline_content=str(window.get("baseline_content", "")),
        target_content=str(window.get("target_content", "")),
        exact_target_span=bool(window.get("exact_target_span", False)),
    )


def _span_intersects(left: tuple[int, int], right: tuple[int, int]) -> bool:
    if left[1] < left[0] and right[1] < right[0]:
        return left[0] == right[0]
    if left[1] < left[0]:
        return right[0] <= left[0] <= right[1]
    if right[1] < right[0]:
        return left[0] <= right[0] <= left[1]
    return max(left[0], right[0]) <= min(left[1], right[1])


def _step_current_span(step: dict[str, object]) -> tuple[int, int] | None:
    start_line = _int_value(step.get("current_start_line"), 0)
    end_line = _int_value(step.get("current_end_line"), 0)
    if start_line > 0 and end_line >= start_line:
        return start_line, end_line
    edit_kind = str(step.get("edit_kind", "")).strip()
    if edit_kind in {"token_replace", "token_insert", "token_delete", "line_replace"}:
        replacements = step.get("replacements", [])
        if not isinstance(replacements, list) or not replacements:
            return None
        line_numbers = [
            _int_value(replacement.get("line_number"), 0)
            for replacement in replacements
            if isinstance(replacement, dict)
        ]
        valid = [line_number for line_number in line_numbers if line_number > 0]
        if not valid:
            return None
        return min(valid), max(valid)
    if edit_kind == "line_insert":
        insertion = step.get("insertion", {})
        if not isinstance(insertion, dict):
            return None
        line_number = _int_value(insertion.get("line_number"), 0)
        if line_number <= 0:
            return None
        return line_number, line_number
    if edit_kind == "line_delete":
        deletion = step.get("deletion", {})
        if not isinstance(deletion, dict):
            return None
        start_line = _int_value(deletion.get("start_line"), 0)
        end_line = _int_value(deletion.get("end_line"), 0)
        if start_line <= 0 or end_line < start_line:
            return None
        return start_line, end_line
    if edit_kind == "block_replace":
        replacement = step.get("replacement", {})
        if not isinstance(replacement, dict):
            return None
        start_line = _int_value(replacement.get("start_line"), 0)
        end_line = _int_value(replacement.get("end_line"), 0)
        if start_line <= 0 or end_line < start_line:
            return None
        return start_line, end_line
    return None


def _step_target_span(step: dict[str, object]) -> tuple[int, int] | None:
    start_line = _int_value(step.get("target_start_line"), 0)
    end_line = _int_value(step.get("target_end_line"), 0)
    if start_line > 0:
        return start_line, end_line
    return _step_current_span(step)


def _preview_block_proposal_span_signature(
    proposal: dict[str, object],
) -> tuple[int, int, int, int] | None:
    if str(proposal.get("edit_kind", "")).strip() != "block_replace":
        return None
    step = proposal.get("step")
    if not isinstance(step, dict):
        return None
    current_span = _step_current_span(step)
    target_span = _step_target_span(step)
    if current_span is None or target_span is None:
        return None
    return (
        int(current_span[0]),
        int(current_span[1]),
        int(target_span[0]),
        int(target_span[1]),
    )


def _preview_structured_edit_proposal_span_signature(
    proposal: dict[str, object],
) -> tuple[int, int, int, int] | None:
    proposal_source = str(proposal.get("proposal_source", "")).strip()
    if not proposal_source.startswith("structured_edit:"):
        return None
    step = proposal.get("step")
    if not isinstance(step, dict):
        return None
    edit_kind = str(step.get("edit_kind", "")).strip()
    if edit_kind == "multi_edit":
        child_steps = [
            child_step
            for child_step in step.get("steps", [])
            if isinstance(child_step, dict)
        ]
        if not child_steps:
            return None
        current_spans = [
            current_span
            for child_step in child_steps
            if (current_span := _step_current_span(child_step)) is not None
        ]
        target_spans = [
            target_span
            for child_step in child_steps
            if (target_span := _step_target_span(child_step)) is not None
        ]
        if not current_spans or not target_spans:
            return None
        return (
            min(int(span[0]) for span in current_spans),
            max(int(span[1]) for span in current_spans),
            min(int(span[0]) for span in target_spans),
            max(int(span[1]) for span in target_spans),
        )
    current_span = _step_current_span(step)
    target_span = _step_target_span(step)
    if current_span is None or target_span is None:
        return None
    return (
        int(current_span[0]),
        int(current_span[1]),
        int(target_span[0]),
        int(target_span[1]),
    )


def _preview_block_proposal_proof_quality_score(proposal: dict[str, object]) -> int:
    if str(proposal.get("edit_kind", "")).strip() != "block_replace":
        return 0
    score = 0
    partial_region_block = bool(proposal.get("current_proof_partial_region_block", False))
    if bool(proposal.get("shared_anchor_exact_region_block", False)):
        score += 3
    if bool(proposal.get("recovered_conflicting_alias", False)):
        score += 3
    if bool(proposal.get("exact_hidden_gap_region_block", False)):
        score += 4
    if bool(proposal.get("bridged_hidden_gap_region_block", False)):
        score += 5
    if bool(proposal.get("current_proof_region_block", False)):
        score += 5 if partial_region_block else 6
    if bool(proposal.get("current_proof_complete", False)):
        score += 8
    if bool(proposal.get("explicit_hidden_gap_current_proof", False)):
        score += 4
    if bool(proposal.get("hidden_gap_current_from_line_span_proof", False)):
        score += 3
    if bool(proposal.get("hidden_gap_target_from_expected_content", False)):
        score += 2
    score += min(6, max(0, _int_value(proposal.get("shared_anchor_hybrid_component_count"), 0)) * 3)
    score += min(4, max(0, _int_value(proposal.get("shared_anchor_core_count"), 0) - 1) * 2)
    score += min(3, max(0, _int_value(proposal.get("shared_anchor_exact_neighbor_count"), 0)))
    score += min(3, max(0, _int_value(proposal.get("current_proof_span_count"), 0) - 1))
    if bool(proposal.get("partial_window_coverage", False)):
        score -= 4
    if bool(proposal.get("current_proof_partial_coverage", False)):
        score -= 2 if partial_region_block else 6
    missing_span_count = max(0, _int_value(proposal.get("current_proof_missing_span_count"), 0))
    missing_line_count = max(0, _int_value(proposal.get("current_proof_missing_line_count"), 0))
    if partial_region_block:
        opaque_span_count = max(0, _int_value(proposal.get("current_proof_opaque_span_count"), 0))
        opaque_max_span_line_count = max(
            0,
            _int_value(proposal.get("current_proof_opaque_max_span_line_count"), 0),
        )
        topology = str(proposal.get("current_proof_partial_region_topology", "")).strip()
        score -= min(2, max(0, missing_span_count - 1))
        score -= min(2, max(0, missing_line_count - 1))
        score += min(2, max(0, _int_value(proposal.get("current_proof_opaque_internal_span_count"), 0)))
        score -= min(3, max(0, opaque_span_count - 1))
        score -= min(3, max(0, opaque_max_span_line_count - 1) * 2)
        if topology == "sparse_internal_multi_opaque":
            score -= 1
        score -= min(
            4,
            max(0, _int_value(proposal.get("current_proof_opaque_boundary_touch_count"), 0)) * 4,
        )
    else:
        score -= min(6, missing_span_count * 3)
        score -= min(6, missing_line_count)
    return score


def _same_span_preview_block_proposal_rank_key(
    proposal: dict[str, object],
) -> tuple[object, ...]:
    proof_quality_score = _preview_block_proposal_proof_quality_score(proposal)
    return (
        -proof_quality_score,
        *_same_span_preview_block_proposal_provenance_rank_key(proposal),
        -_int_value(proposal.get("shared_anchor_hybrid_component_count"), 0),
        -_int_value(proposal.get("shared_anchor_core_count"), 0),
        -_int_value(proposal.get("current_proof_span_count"), 0),
        0 if bool(proposal.get("current_proof_complete", False)) else 1,
        0 if bool(proposal.get("hidden_gap_target_from_expected_content", False)) else 1,
        0 if bool(proposal.get("hidden_gap_current_from_line_span_proof", False)) else 1,
        0 if bool(proposal.get("explicit_hidden_gap_current_proof", False)) else 1,
        _int_value(proposal.get("current_proof_missing_span_count"), 0),
        _int_value(proposal.get("current_proof_missing_line_count"), 0),
        *_overlap_block_replace_rank_key(proposal),
        str(proposal.get("command", "")),
    )


def _same_span_preview_block_proposal_provenance_rank_key(
    proposal: dict[str, object],
) -> tuple[object, ...]:
    return (
        0 if bool(proposal.get("recovered_conflicting_alias", False)) else 1,
        0 if bool(proposal.get("current_proof_region_block", False)) else 1,
        0 if bool(proposal.get("current_proof_complete", False)) else 1,
        0 if bool(proposal.get("explicit_hidden_gap_current_proof", False)) else 1,
        0 if bool(proposal.get("hidden_gap_current_from_line_span_proof", False)) else 1,
        0 if bool(proposal.get("bridged_hidden_gap_region_block", False)) else 1,
        0 if bool(proposal.get("exact_hidden_gap_region_block", False)) else 1,
        0 if bool(proposal.get("hidden_gap_target_from_expected_content", False)) else 1,
        0 if bool(proposal.get("shared_anchor_exact_region_block", False)) else 1,
        -_int_value(proposal.get("overlap_alias_pair_count"), 0),
        -_int_value(proposal.get("safe_inexact_window_count"), 0),
        -_int_value(proposal.get("shared_anchor_exact_neighbor_count"), 0),
    )


def _matching_target_preview_content(
    target_content: str,
    *,
    omitted_prefix_sha1: str,
    omitted_suffix_sha1: str,
    preferred_line_count: int,
    preferred_line_start: int,
) -> str | None:
    lines = target_content.splitlines(keepends=True)
    prefix_hashes = [_sha1_text("".join(lines[:index])) for index in range(len(lines) + 1)]
    suffix_hashes = [_sha1_text("".join(lines[index:])) for index in range(len(lines) + 1)]
    start_indices = [
        index for index, digest in enumerate(prefix_hashes) if digest == omitted_prefix_sha1
    ]
    end_indices = [
        index for index, digest in enumerate(suffix_hashes) if digest == omitted_suffix_sha1
    ]
    candidates: list[tuple[int, int, str]] = []
    for start_index in start_indices:
        for end_index in end_indices:
            if end_index <= start_index:
                continue
            visible_content = "".join(lines[start_index:end_index])
            if not visible_content:
                continue
            candidates.append((start_index, end_index, visible_content))
    if not candidates:
        return None
    _, _, best_visible_content = min(
        candidates,
        key=lambda item: (
            abs((item[1] - item[0]) - preferred_line_count),
            abs((item[0] + 1) - preferred_line_start),
            item[0],
            item[1],
        ),
    )
    return best_visible_content


def _offset_preview_step(
    step: dict[str, object],
    *,
    line_offset: int,
    target_line_offset: int = 0,
) -> dict[str, object]:
    if line_offset <= 0 and target_line_offset <= 0:
        return step
    shifted = dict(step)
    if _int_value(step.get("current_start_line"), 0) > 0:
        shifted["current_start_line"] = _int_value(step.get("current_start_line"), 0) + line_offset
        shifted["current_end_line"] = _int_value(step.get("current_end_line"), 0) + line_offset
    if _int_value(step.get("target_start_line"), 0) > 0:
        shifted["target_start_line"] = _int_value(step.get("target_start_line"), 0) + target_line_offset
        shifted["target_end_line"] = _int_value(step.get("target_end_line"), 0) + target_line_offset
    edit_kind = str(step.get("edit_kind", "")).strip()
    if edit_kind in {"token_replace", "token_insert", "token_delete", "line_replace"}:
        replacements = step.get("replacements", [])
        if not isinstance(replacements, list):
            return step
        shifted["replacements"] = [
            {
                **replacement,
                "line_number": _int_value(replacement.get("line_number"), 0) + line_offset,
            }
            for replacement in replacements
            if isinstance(replacement, dict)
        ]
        return shifted
    if edit_kind == "line_insert":
        insertion = step.get("insertion", {})
        if not isinstance(insertion, dict):
            return step
        shifted["insertion"] = {
            **insertion,
            "line_number": _int_value(insertion.get("line_number"), 0) + line_offset,
        }
        return shifted
    if edit_kind == "line_delete":
        deletion = step.get("deletion", {})
        if not isinstance(deletion, dict):
            return step
        shifted["deletion"] = {
            **deletion,
            "start_line": _int_value(deletion.get("start_line"), 0) + line_offset,
            "end_line": _int_value(deletion.get("end_line"), 0) + line_offset,
        }
        return shifted
    if edit_kind == "block_replace":
        replacement = step.get("replacement", {})
        if not isinstance(replacement, dict):
            return step
        shifted["replacement"] = {
            **replacement,
            "start_line": _int_value(replacement.get("start_line"), 0) + line_offset,
            "end_line": _int_value(replacement.get("end_line"), 0) + line_offset,
        }
    return shifted


def _best_preview_edit_step(
    *,
    path: str,
    baseline_content: str,
    target_content: str,
    exact_target_span: bool = True,
) -> dict[str, object] | None:
    candidates: list[dict[str, object]] = []
    token_edit = _derive_token_replace_step(path=path, baseline_content=baseline_content, target_content=target_content)
    if token_edit is not None:
        candidates.append(token_edit)
    token_insert = _derive_token_insert_step(path=path, baseline_content=baseline_content, target_content=target_content)
    if token_insert is not None:
        candidates.append(token_insert)
    token_delete = _derive_token_delete_step(path=path, baseline_content=baseline_content, target_content=target_content)
    if token_delete is not None:
        candidates.append(token_delete)
    insert_edit = _derive_line_insert_step(path=path, baseline_content=baseline_content, target_content=target_content)
    if insert_edit is not None:
        candidates.append(insert_edit)
    delete_edit = _derive_line_delete_step(path=path, baseline_content=baseline_content, target_content=target_content)
    if delete_edit is not None:
        candidates.append(delete_edit)
    block_edit = _derive_block_replace_step(path=path, baseline_content=baseline_content, target_content=target_content)
    if block_edit is not None:
        candidates.append(block_edit)
    line_edit = _derive_line_replace_step(path=path, baseline_content=baseline_content, target_content=target_content)
    if line_edit is not None:
        candidates.append(line_edit)
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: (
            _preview_edit_precision_rank(item, exact_target_span=exact_target_span),
            int(item.get("edit_score", 9999)),
            _edit_kind_rank(str(item.get("edit_kind", ""))),
            str(item.get("edit_kind", "")),
        ),
    )


def _preview_edit_precision_rank(
    step: dict[str, object],
    *,
    exact_target_span: bool,
) -> int:
    if exact_target_span:
        return 0
    edit_kind = str(step.get("edit_kind", "")).strip()
    current_span = _step_current_span(step)
    span_length = 0
    if current_span is not None:
        span_length = max(0, int(current_span[1]) - int(current_span[0]) + 1)
    replacement_count = 0
    replacements = step.get("replacements", [])
    if isinstance(replacements, list):
        replacement_count = len([item for item in replacements if isinstance(item, dict)])
    if edit_kind == "block_replace":
        return 0
    if edit_kind in {"line_delete", "line_insert"}:
        return 1
    if edit_kind == "line_replace":
        return 1 if span_length > 1 or replacement_count > 1 else 0
    if edit_kind in {"token_replace", "token_insert", "token_delete"}:
        return 2 if span_length > 1 or replacement_count > 1 else 0
    return 1


def _preview_step_precision_penalty(
    step: dict[str, object],
    *,
    exact_target_span: bool,
) -> int:
    if exact_target_span:
        return 0
    edit_kind = str(step.get("edit_kind", "")).strip()
    current_span = _step_current_span(step)
    span_length = 0
    if current_span is not None:
        span_length = max(0, int(current_span[1]) - int(current_span[0]) + 1)
    replacement_count = 0
    replacements = step.get("replacements", [])
    if isinstance(replacements, list):
        replacement_count = len([item for item in replacements if isinstance(item, dict)])
    if edit_kind == "block_replace":
        return 1
    if edit_kind in {"line_delete", "line_insert"}:
        return 2
    if edit_kind == "line_replace":
        return 2 if span_length > 1 or replacement_count > 1 else 1
    if edit_kind in {"token_replace", "token_insert", "token_delete"}:
        return 3 if span_length > 1 or replacement_count > 1 else 2
    return 2


def _derive_line_replace_step(
    *,
    path: str,
    baseline_content: str,
    target_content: str,
) -> dict[str, object] | None:
    if not baseline_content or baseline_content == target_content:
        return None
    baseline_lines = baseline_content.splitlines()
    target_lines = target_content.splitlines()
    if len(baseline_lines) != len(target_lines):
        return None
    replacements: list[dict[str, object]] = []
    for line_number, (before_line, after_line) in enumerate(zip(baseline_lines, target_lines), start=1):
        if before_line == after_line:
            continue
        replacements.append(
            {
                "line_number": line_number,
                "before_line": before_line,
                "after_line": after_line,
            }
        )
    if not replacements:
        return None
    return {
        "path": path,
        "edit_kind": "line_replace",
        "replacements": replacements,
        "current_start_line": min(item["line_number"] for item in replacements),
        "current_end_line": max(item["line_number"] for item in replacements),
        "target_start_line": min(item["line_number"] for item in replacements),
        "target_end_line": max(item["line_number"] for item in replacements),
        "line_delta": 0,
        "edit_score": _edit_candidate_score("line_replace", replacements=replacements),
    }


def _derive_token_replace_step(
    *,
    path: str,
    baseline_content: str,
    target_content: str,
) -> dict[str, object] | None:
    if not baseline_content or baseline_content == target_content:
        return None
    baseline_lines = baseline_content.splitlines()
    target_lines = target_content.splitlines()
    if len(baseline_lines) != len(target_lines):
        return None
    replacements: list[dict[str, object]] = []
    for line_number, (before_line, after_line) in enumerate(zip(baseline_lines, target_lines), start=1):
        if before_line == after_line:
            continue
        fragment = _line_edit_fragments(before_line, after_line)
        if fragment is None:
            return None
        before_fragment = str(fragment["before_fragment"])
        after_fragment = str(fragment["after_fragment"])
        if not before_fragment or not after_fragment:
            return None
        if before_line.count(before_fragment) != 1:
            return None
        replacements.append(
            {
                "line_number": line_number,
                "before_fragment": before_fragment,
                "after_fragment": after_fragment,
                "before_line": before_line,
                "after_line": after_line,
            }
        )
    if not replacements:
        return None
    return {
        "path": path,
        "edit_kind": "token_replace",
        "replacements": replacements,
        "current_start_line": min(item["line_number"] for item in replacements),
        "current_end_line": max(item["line_number"] for item in replacements),
        "target_start_line": min(item["line_number"] for item in replacements),
        "target_end_line": max(item["line_number"] for item in replacements),
        "line_delta": 0,
        "edit_score": _edit_candidate_score("token_replace", replacements=replacements),
    }


def _derive_token_insert_step(
    *,
    path: str,
    baseline_content: str,
    target_content: str,
) -> dict[str, object] | None:
    if not baseline_content or baseline_content == target_content:
        return None
    baseline_lines = baseline_content.splitlines()
    target_lines = target_content.splitlines()
    if len(baseline_lines) != len(target_lines):
        return None
    insertions: list[dict[str, object]] = []
    for line_number, (before_line, after_line) in enumerate(zip(baseline_lines, target_lines), start=1):
        if before_line == after_line:
            continue
        fragment = _line_edit_fragments(before_line, after_line)
        if fragment is None:
            return None
        before_fragment = str(fragment["before_fragment"])
        after_fragment = str(fragment["after_fragment"])
        if before_fragment or not after_fragment:
            return None
        prefix = str(fragment["prefix"])
        suffix = str(fragment["suffix"])
        if not prefix and not suffix:
            return None
        insertions.append(
            {
                "line_number": line_number,
                "prefix": prefix,
                "after_fragment": after_fragment,
                "suffix": suffix,
                "before_line": before_line,
                "after_line": after_line,
            }
        )
    if not insertions:
        return None
    return {
        "path": path,
        "edit_kind": "token_insert",
        "replacements": insertions,
        "current_start_line": min(item["line_number"] for item in insertions),
        "current_end_line": max(item["line_number"] for item in insertions),
        "target_start_line": min(item["line_number"] for item in insertions),
        "target_end_line": max(item["line_number"] for item in insertions),
        "line_delta": 0,
        "edit_score": _edit_candidate_score("token_insert", replacements=insertions),
    }


def _derive_token_delete_step(
    *,
    path: str,
    baseline_content: str,
    target_content: str,
) -> dict[str, object] | None:
    if not baseline_content or baseline_content == target_content:
        return None
    baseline_lines = baseline_content.splitlines()
    target_lines = target_content.splitlines()
    if len(baseline_lines) != len(target_lines):
        return None
    deletions: list[dict[str, object]] = []
    for line_number, (before_line, after_line) in enumerate(zip(baseline_lines, target_lines), start=1):
        if before_line == after_line:
            continue
        fragment = _line_edit_fragments(before_line, after_line)
        if fragment is None:
            return None
        before_fragment = str(fragment["before_fragment"])
        after_fragment = str(fragment["after_fragment"])
        if not before_fragment or after_fragment:
            return None
        prefix = str(fragment["prefix"])
        suffix = str(fragment["suffix"])
        if not prefix and not suffix:
            return None
        deletions.append(
            {
                "line_number": line_number,
                "prefix": prefix,
                "before_fragment": before_fragment,
                "suffix": suffix,
                "before_line": before_line,
                "after_line": after_line,
            }
        )
    if not deletions:
        return None
    return {
        "path": path,
        "edit_kind": "token_delete",
        "replacements": deletions,
        "current_start_line": min(item["line_number"] for item in deletions),
        "current_end_line": max(item["line_number"] for item in deletions),
        "target_start_line": min(item["line_number"] for item in deletions),
        "target_end_line": max(item["line_number"] for item in deletions),
        "line_delta": 0,
        "edit_score": _edit_candidate_score("token_delete", replacements=deletions),
    }


def _derive_block_replace_step(
    *,
    path: str,
    baseline_content: str,
    target_content: str,
) -> dict[str, object] | None:
    if not baseline_content or baseline_content == target_content:
        return None
    delta = _contiguous_line_delta(baseline_content=baseline_content, target_content=target_content)
    if delta is None:
        return None
    before_lines = delta["before_lines"]
    after_lines = delta["after_lines"]
    if not before_lines or not after_lines:
        return None
    baseline_start = int(delta["baseline_start"])
    baseline_end = int(delta["baseline_end"])
    target_start = int(delta["target_start"])
    target_end = int(delta["target_end"])
    if len(before_lines) <= 1 and len(after_lines) <= 1:
        return None
    replacement = {
        "start_line": baseline_start + 1,
        "end_line": max(baseline_start + 1, baseline_end),
        "before_lines": before_lines,
        "after_lines": after_lines,
    }
    return {
        "path": path,
        "edit_kind": "block_replace",
        "replacement": replacement,
        "current_start_line": baseline_start + 1,
        "current_end_line": max(baseline_start + 1, baseline_end),
        "target_start_line": target_start + 1,
        "target_end_line": max(target_start + 1, target_end),
        "line_delta": int(delta["line_delta"]),
        "edit_score": _edit_candidate_score("block_replace", replacement=replacement),
    }


def _derive_line_insert_step(
    *,
    path: str,
    baseline_content: str,
    target_content: str,
) -> dict[str, object] | None:
    if baseline_content == target_content:
        return None
    delta = _contiguous_line_delta(baseline_content=baseline_content, target_content=target_content)
    if delta is None:
        return None
    before_lines = delta["before_lines"]
    after_lines = delta["after_lines"]
    if before_lines or not after_lines:
        return None
    baseline_lines = baseline_content.splitlines()
    baseline_start = int(delta["baseline_start"])
    target_start = int(delta["target_start"])
    target_end = int(delta["target_end"])
    insertion = {
        "line_number": baseline_start + 1,
        "mode": "append" if baseline_start >= len(baseline_lines) else "before",
        "after_lines": after_lines,
    }
    return {
        "path": path,
        "edit_kind": "line_insert",
        "insertion": insertion,
        "current_start_line": baseline_start + 1,
        "current_end_line": baseline_start + 1,
        "target_start_line": target_start + 1,
        "target_end_line": max(target_start + 1, target_end),
        "line_delta": int(delta["line_delta"]),
        "edit_score": _edit_candidate_score("line_insert", insertion=insertion),
    }


def _derive_line_delete_step(
    *,
    path: str,
    baseline_content: str,
    target_content: str,
) -> dict[str, object] | None:
    if not baseline_content or baseline_content == target_content:
        return None
    delta = _contiguous_line_delta(baseline_content=baseline_content, target_content=target_content)
    if delta is None:
        return None
    before_lines = delta["before_lines"]
    after_lines = delta["after_lines"]
    if not before_lines or after_lines:
        return None
    baseline_start = int(delta["baseline_start"])
    baseline_end = int(delta["baseline_end"])
    target_start = int(delta["target_start"])
    deletion = {
        "start_line": baseline_start + 1,
        "end_line": max(baseline_start + 1, baseline_end),
        "before_lines": before_lines,
    }
    return {
        "path": path,
        "edit_kind": "line_delete",
        "deletion": deletion,
        "current_start_line": baseline_start + 1,
        "current_end_line": max(baseline_start + 1, baseline_end),
        "target_start_line": target_start + 1,
        "target_end_line": target_start,
        "line_delta": int(delta["line_delta"]),
        "edit_score": _edit_candidate_score("line_delete", deletion=deletion),
    }


def _contiguous_line_delta(*, baseline_content: str, target_content: str) -> dict[str, object] | None:
    baseline_lines = baseline_content.splitlines()
    target_lines = target_content.splitlines()
    prefix_length = 0
    while (
        prefix_length < len(baseline_lines)
        and prefix_length < len(target_lines)
        and baseline_lines[prefix_length] == target_lines[prefix_length]
    ):
        prefix_length += 1
    suffix_length = 0
    while (
        suffix_length < (len(baseline_lines) - prefix_length)
        and suffix_length < (len(target_lines) - prefix_length)
        and baseline_lines[len(baseline_lines) - 1 - suffix_length] == target_lines[len(target_lines) - 1 - suffix_length]
    ):
        suffix_length += 1
    baseline_start = prefix_length
    baseline_end = len(baseline_lines) - suffix_length
    target_end = len(target_lines) - suffix_length
    before_lines = baseline_lines[baseline_start:baseline_end]
    after_lines = target_lines[prefix_length:target_end]
    if not before_lines and not after_lines:
        return None
    return {
        "baseline_start": baseline_start,
        "baseline_end": baseline_end,
        "target_start": prefix_length,
        "target_end": target_end,
        "before_lines": before_lines,
        "after_lines": after_lines,
        "line_delta": len(after_lines) - len(before_lines),
    }


def _edit_candidate_score(
    edit_kind: str,
    *,
    replacements: list[dict[str, object]] | None = None,
    replacement: dict[str, object] | None = None,
    insertion: dict[str, object] | None = None,
    deletion: dict[str, object] | None = None,
) -> int:
    normalized_kind = str(edit_kind).strip() or "rewrite"
    if normalized_kind == "multi_edit":
        return 8
    if normalized_kind == "token_replace":
        ops = replacements or []
        fragment_chars = sum(len(str(item.get("before_fragment", ""))) + len(str(item.get("after_fragment", ""))) for item in ops)
        return 10 + len(ops) * 5 + fragment_chars
    if normalized_kind == "token_delete":
        ops = replacements or []
        fragment_chars = sum(len(str(item.get("before_fragment", ""))) for item in ops)
        return 12 + len(ops) * 5 + fragment_chars
    if normalized_kind == "token_insert":
        ops = replacements or []
        fragment_chars = sum(len(str(item.get("after_fragment", ""))) for item in ops)
        return 13 + len(ops) * 5 + fragment_chars
    if normalized_kind == "line_replace":
        ops = replacements or []
        changed_chars = sum(len(str(item.get("before_line", ""))) + len(str(item.get("after_line", ""))) for item in ops)
        return 30 + len(ops) * 12 + changed_chars
    if normalized_kind == "line_delete":
        removed = deletion or {}
        before_lines = [str(line) for line in removed.get("before_lines", [])]
        changed_chars = sum(len(line) for line in before_lines)
        return 34 + len(before_lines) * 10 + changed_chars
    if normalized_kind == "line_insert":
        inserted = insertion or {}
        raw_after_lines = inserted.get("after_lines", inserted.get("inserted_lines", []))
        after_lines = [str(line) for line in raw_after_lines]
        changed_chars = sum(len(line) for line in after_lines)
        return 38 + len(after_lines) * 12 + changed_chars
    if normalized_kind == "block_replace":
        block = replacement or {}
        before_lines = [str(line) for line in block.get("before_lines", [])]
        after_lines = [str(line) for line in block.get("after_lines", [])]
        changed_chars = sum(len(line) for line in before_lines) + sum(len(line) for line in after_lines)
        changed_lines = max(len(before_lines), len(after_lines))
        return 60 + changed_lines * 20 + changed_chars
    return 120


def _edit_kind_rank(edit_kind: str) -> int:
    order = {
        "multi_edit": 0,
        "token_replace": 0,
        "token_delete": 1,
        "token_insert": 2,
        "line_replace": 3,
        "multi_window_structured_edit": 4,
        "line_delete": 4,
        "line_insert": 5,
        "block_replace": 6,
        "rewrite": 7,
    }
    return order.get(str(edit_kind).strip() or "rewrite", 99)


def _structured_edit_window_bonus(metadata: dict[str, object]) -> float:
    edit_window_count = max(1, _int_value(metadata.get("edit_window_count"), 1))
    covered_window_count = max(
        edit_window_count,
        _int_value(metadata.get("covered_window_count"), edit_window_count),
    )
    if covered_window_count <= 1:
        return 0.0
    bonus = min(3.0, (covered_window_count - 1) * 1.5)
    available_window_count = max(
        covered_window_count,
        _int_value(metadata.get("available_window_count"), covered_window_count),
    )
    if available_window_count > 1:
        coverage_ratio = min(1.0, covered_window_count / available_window_count)
        bonus += coverage_ratio * 1.5
    if covered_window_count >= 3:
        bonus += min(3.0, (covered_window_count - 2) * 1.5)
    return bonus


def _structured_edit_chain_bonus(metadata: dict[str, object]) -> float:
    if str(metadata.get("edit_kind", "")).strip() != "multi_edit":
        return 0.0
    component_line_deltas = metadata.get("component_line_deltas", [])
    if not isinstance(component_line_deltas, list):
        component_line_deltas = []
    structural_window_count = sum(1 for value in component_line_deltas if _int_value(value, 0) != 0)
    component_kinds = metadata.get("component_edit_kinds", [])
    if not isinstance(component_kinds, list):
        component_kinds = []
    distinct_kinds = len(
        {
            str(value).strip()
            for value in component_kinds
            if str(value).strip()
        }
    )
    bonus = 0.0
    if structural_window_count > 0:
        bonus += 2.0 + min(1.5, max(0, structural_window_count - 1) * 0.75)
    if structural_window_count > 0 and distinct_kinds > 1:
        bonus += min(1.5, (distinct_kinds - 1) * 0.75)
    return bonus


def _structured_edit_overlap_recovery_bonus(metadata: dict[str, object]) -> float:
    edit_source = str(metadata.get("edit_source", "")).strip()
    if edit_source not in {"workspace_preview", "workspace_preview_range"}:
        return 0.0
    edit_kind = str(metadata.get("edit_kind", "")).strip()
    if edit_kind not in {"block_replace", "multi_edit"}:
        return 0.0
    covered_window_count = max(
        1,
        _int_value(
            metadata.get("covered_window_count"),
            metadata.get("edit_window_count"),
        ),
    )
    if covered_window_count < 2:
        return 0.0
    overlap_alias_pair_count = max(0, _int_value(metadata.get("overlap_alias_pair_count"), 0))
    overlap_alias_window_count = max(0, _int_value(metadata.get("overlap_alias_window_count"), 0))
    unresolved_alias_pair_count = max(
        0,
        _int_value(metadata.get("overlap_component_unresolved_alias_pair_count"), 0),
    )
    frontier_gap = max(0.0, _float_value(metadata.get("overlap_component_frontier_gap"), 0.0))
    exact_window_count = max(0, _int_value(metadata.get("exact_window_count"), 0))
    inexact_window_count = max(0, _int_value(metadata.get("inexact_window_count"), 0))
    bonus = 0.0
    if edit_kind == "block_replace":
        bonus += 2.0 + min(1.5, (covered_window_count - 2) * 0.75)
    elif overlap_alias_pair_count > 0:
        bonus += 1.5 + min(1.5, (covered_window_count - 2) * 0.75)
    if exact_window_count > 0 and inexact_window_count > 0:
        bonus += 1.0
    elif exact_window_count == covered_window_count:
        bonus += 0.5
    if overlap_alias_pair_count > 0:
        bonus += min(2.0, overlap_alias_pair_count * 0.75)
    if overlap_alias_window_count > 0:
        bonus += min(1.5, overlap_alias_window_count * 0.5)
    span_line_count = max(
        covered_window_count,
        _int_value(metadata.get("span_line_count"), covered_window_count),
    )
    span_overhang = max(0, span_line_count - covered_window_count)
    bonus += max(0.0, 1.5 - span_overhang * 0.5)
    if unresolved_alias_pair_count > 0:
        bonus -= min(2.5, unresolved_alias_pair_count * 0.75)
    if frontier_gap > 0:
        bonus -= min(2.0, frontier_gap * 0.25)
    return bonus


def _structured_edit_conflicting_alias_recovery_adjustment(metadata: dict[str, object]) -> float:
    edit_kind = str(metadata.get("edit_kind", "")).strip()
    if edit_kind not in {"block_replace", "multi_edit"}:
        return 0.0
    if not bool(metadata.get("recovered_conflicting_alias", False)):
        return 0.0
    covered_window_count = max(1, _int_value(metadata.get("covered_window_count"), 1))
    unresolved_alias_pair_count = max(
        0,
        _int_value(metadata.get("overlap_component_unresolved_alias_pair_count"), 0),
    )
    if (
        edit_kind == "block_replace"
        and bool(metadata.get("exact_contiguous_region_block", False))
        and unresolved_alias_pair_count <= 0
    ):
        return 2.5 + min(1.5, max(0, covered_window_count - 2) * 0.75)
    if (
        edit_kind == "block_replace"
        and bool(metadata.get("shared_anchor_exact_region_block", False))
        and max(0, _int_value(metadata.get("overlap_alias_pair_count"), 0)) > 0
        and unresolved_alias_pair_count <= 0
    ):
        bonus = 4.5 + min(2.0, max(0, covered_window_count - 2) * 0.75)
        bonus += _shared_anchor_core_bonus(metadata, per_extra_core=0.75, cap=1.5)
        bonus += min(
            1.5,
            max(0, _int_value(metadata.get("shared_anchor_hybrid_component_count"), 0)) * 0.75,
        )
        if bool(metadata.get("shared_anchor_mixed_insert_delete", False)):
            bonus += 0.5
        return bonus
    if edit_kind == "multi_edit":
        if unresolved_alias_pair_count <= 0:
            return 0.0
        penalty = -10.0 if covered_window_count <= 3 else -6.0
    elif covered_window_count <= 2:
        penalty = -14.0
    else:
        penalty = -8.0
    if unresolved_alias_pair_count > 0:
        penalty -= min(4.0, unresolved_alias_pair_count * 1.5)
    return penalty


def _structured_edit_unrecoverable_overlap_penalty(metadata: dict[str, object]) -> float:
    edit_source = str(metadata.get("edit_source", "")).strip()
    if edit_source not in {"workspace_preview", "workspace_preview_range"}:
        return 0.0
    unrecoverable_component_count = max(
        0,
        _int_value(metadata.get("overlap_component_unrecoverable_count"), 0),
    )
    if unrecoverable_component_count <= 0:
        return 0.0
    unrecoverable_window_count = max(
        unrecoverable_component_count,
        _int_value(metadata.get("overlap_component_unrecoverable_window_count"), 0),
    )
    retained_window_count = max(
        1,
        _int_value(
            metadata.get("retained_window_count"),
            metadata.get("available_window_count"),
        ),
    )
    covered_window_count = max(
        1,
        _int_value(
            metadata.get("covered_window_count"),
            metadata.get("edit_window_count"),
        ),
    )
    uncovered_window_count = max(0, retained_window_count - covered_window_count)
    penalty = 8.0 + unrecoverable_component_count * 3.0
    penalty += min(4.0, unrecoverable_window_count * 0.75)
    if uncovered_window_count > 0:
        penalty += uncovered_window_count * 2.5
    if bool(metadata.get("allow_expected_write_fallback", False)):
        penalty += 2.0
    if str(metadata.get("edit_kind", "")).strip() == "multi_edit":
        penalty += 1.5
    return -penalty


def _structured_edit_block_alternative_adjustment(metadata: dict[str, object]) -> float:
    edit_source = str(metadata.get("edit_source", "")).strip()
    if edit_source not in {"workspace_preview", "workspace_preview_range"}:
        return 0.0
    if str(metadata.get("edit_kind", "")).strip() != "block_replace":
        return 0.0
    alternative_count = max(0, _int_value(metadata.get("block_alternative_count"), 0))
    if alternative_count < 2:
        return 0.0
    covered_window_count = max(1, _int_value(metadata.get("covered_window_count"), 1))
    max_covered_window_count = max(
        covered_window_count,
        _int_value(metadata.get("block_max_covered_window_count"), covered_window_count),
    )
    exact_window_count = max(0, _int_value(metadata.get("exact_window_count"), 0))
    max_exact_window_count = max(
        exact_window_count,
        _int_value(metadata.get("block_max_exact_window_count"), exact_window_count),
    )
    precision_penalty = max(0.0, _float_value(metadata.get("precision_penalty"), 0.0))
    min_precision_penalty = min(
        precision_penalty,
        _float_value(metadata.get("block_min_precision_penalty"), precision_penalty),
    )
    edit_window_count = max(1, _int_value(metadata.get("edit_window_count"), 1))
    min_edit_window_count = min(
        edit_window_count,
        max(1, _int_value(metadata.get("block_min_edit_window_count"), edit_window_count)),
    )
    inexact_window_count = max(0, _int_value(metadata.get("inexact_window_count"), 0))
    span_line_count = max(
        covered_window_count,
        _int_value(metadata.get("span_line_count"), covered_window_count),
    )
    frontier_max_exact_window_count = max(
        exact_window_count,
        _int_value(metadata.get("block_frontier_max_exact_window_count"), exact_window_count),
    )
    frontier_min_inexact_window_count = max(
        0,
        _int_value(metadata.get("block_frontier_min_inexact_window_count"), inexact_window_count),
    )
    frontier_min_span_line_count = max(
        covered_window_count,
        _int_value(metadata.get("block_frontier_min_span_line_count"), span_line_count),
    )
    bonus = 0.0
    if covered_window_count == max_covered_window_count:
        bonus += 1.5
    else:
        bonus -= (max_covered_window_count - covered_window_count) * 1.5
    if exact_window_count == max_exact_window_count:
        bonus += 1.5
    else:
        bonus -= (max_exact_window_count - exact_window_count) * 1.25
    if precision_penalty <= min_precision_penalty:
        bonus += 0.5
    else:
        bonus -= min(2.0, (precision_penalty - min_precision_penalty) * 0.75)
    if edit_window_count == min_edit_window_count:
        bonus += 0.5
    else:
        bonus -= min(1.5, (edit_window_count - min_edit_window_count) * 0.75)
    if (
        frontier_max_exact_window_count >= 2
        and max_covered_window_count >= 2
        and exact_window_count == frontier_max_exact_window_count
    ):
        if inexact_window_count == frontier_min_inexact_window_count:
            bonus += 4.0
            if span_line_count == frontier_min_span_line_count:
                bonus += 1.0
            else:
                bonus -= min(2.0, (span_line_count - frontier_min_span_line_count) * 0.5)
        else:
            bonus -= 24.0 + min(8.0, (inexact_window_count - frontier_min_inexact_window_count) * 4.0)
    return bonus


def _structured_edit_same_window_block_adjustment(metadata: dict[str, object]) -> float:
    edit_source = str(metadata.get("edit_source", "")).strip()
    if edit_source not in {"workspace_preview", "workspace_preview_range"}:
        return 0.0
    edit_kind = str(metadata.get("edit_kind", "")).strip()
    if edit_kind == "block_replace":
        if not bool(metadata.get("same_window_exact_block_alternative", False)):
            return 0.0
        if max(1, _int_value(metadata.get("available_window_count"), 1)) <= 1:
            return 0.0
        if not bool(metadata.get("exact_target_span", False)):
            return 0.0
        bonus = 2.5
        if bool(metadata.get("allow_expected_write_fallback", False)):
            bonus += 1.0
        return bonus
    if not bool(metadata.get("same_window_exact_block_alternative", False)):
        return 0.0
    if max(1, _int_value(metadata.get("available_window_count"), 1)) <= 1:
        return 0.0
    penalty = 3.0 + _float_value(metadata.get("precision_penalty"), 0.0)
    if bool(metadata.get("allow_expected_write_fallback", False)):
        penalty += 1.0
    return -penalty


def _shared_anchor_core_bonus(
    metadata: dict[str, object],
    *,
    per_extra_core: float,
    cap: float,
) -> float:
    shared_anchor_core_count = max(
        0,
        _int_value(metadata.get("shared_anchor_core_count"), 0),
    )
    if shared_anchor_core_count <= 1:
        return 0.0
    return min(cap, max(0.0, shared_anchor_core_count - 1) * per_extra_core)


def _structured_edit_exact_region_block_adjustment(metadata: dict[str, object]) -> float:
    edit_source = str(metadata.get("edit_source", "")).strip()
    if edit_source not in {"workspace_preview", "workspace_preview_range"}:
        return 0.0
    if str(metadata.get("edit_kind", "")).strip() != "block_replace":
        return 0.0
    if not bool(metadata.get("exact_contiguous_region_block", False)):
        return 0.0
    covered_window_count = max(1, _int_value(metadata.get("covered_window_count"), 1))
    exact_window_count = max(0, _int_value(metadata.get("exact_window_count"), 0))
    inexact_window_count = max(0, _int_value(metadata.get("inexact_window_count"), 0))
    if covered_window_count < 2 or exact_window_count < covered_window_count or inexact_window_count > 0:
        return 0.0
    bonus = 3.0 + min(1.0, (covered_window_count - 2) * 0.5)
    if not bool(metadata.get("partial_window_coverage", False)):
        bonus += 0.5
    if bool(metadata.get("shared_anchor_exact_region_block", False)):
        shared_anchor_exact_neighbor_count = max(
            0,
            _int_value(metadata.get("shared_anchor_exact_neighbor_count"), 0),
        )
        bonus += 3.0
        bonus += min(1.5, shared_anchor_exact_neighbor_count * 0.5)
        bonus += _shared_anchor_core_bonus(metadata, per_extra_core=0.75, cap=1.5)
        if bool(metadata.get("shared_anchor_mixed_insert_delete", False)):
            bonus += 1.0
    return bonus


def _structured_edit_hidden_gap_region_block_adjustment(metadata: dict[str, object]) -> float:
    edit_source = str(metadata.get("edit_source", "")).strip()
    if edit_source not in {"workspace_preview", "workspace_preview_range"}:
        return 0.0
    if str(metadata.get("edit_kind", "")).strip() != "block_replace":
        return 0.0
    if not bool(metadata.get("exact_hidden_gap_region_block", False)):
        return 0.0
    covered_window_count = max(1, _int_value(metadata.get("covered_window_count"), 1))
    exact_window_count = max(0, _int_value(metadata.get("exact_window_count"), 0))
    inexact_window_count = max(0, _int_value(metadata.get("inexact_window_count"), 0))
    if covered_window_count < 2 or exact_window_count < covered_window_count or inexact_window_count > 0:
        return 0.0
    hidden_gap_current_line_count = max(0, _int_value(metadata.get("hidden_gap_current_line_count"), 0))
    hidden_gap_target_line_count = max(0, _int_value(metadata.get("hidden_gap_target_line_count"), 0))
    hidden_gap_line_count = max(hidden_gap_current_line_count, hidden_gap_target_line_count)
    if hidden_gap_line_count <= 0:
        return 0.0
    bonus = 4.0 + min(1.0, (covered_window_count - 2) * 0.5)
    bonus += min(1.0, hidden_gap_line_count * 0.5)
    if not bool(metadata.get("partial_window_coverage", False)):
        bonus += 0.5
    if bool(metadata.get("shared_anchor_exact_region_block", False)):
        bonus += 1.5
        bonus += _shared_anchor_core_bonus(metadata, per_extra_core=0.5, cap=1.0)
        bonus += min(
            1.0,
            max(0, _int_value(metadata.get("shared_anchor_hybrid_component_count"), 0)) * 0.5,
        )
        if bool(metadata.get("hidden_gap_target_from_expected_content", False)):
            bonus += 0.75
        if bool(metadata.get("shared_anchor_mixed_insert_delete", False)):
            bonus += 0.5
    return bonus


def _structured_edit_bridged_hidden_gap_region_block_adjustment(metadata: dict[str, object]) -> float:
    edit_source = str(metadata.get("edit_source", "")).strip()
    if edit_source not in {"workspace_preview", "workspace_preview_range"}:
        return 0.0
    if str(metadata.get("edit_kind", "")).strip() != "block_replace":
        return 0.0
    if not bool(metadata.get("bridged_hidden_gap_region_block", False)):
        return 0.0
    covered_window_count = max(1, _int_value(metadata.get("covered_window_count"), 1))
    if covered_window_count < 2:
        return 0.0
    hidden_gap_current_line_count = max(0, _int_value(metadata.get("hidden_gap_current_line_count"), 0))
    hidden_gap_target_line_count = max(0, _int_value(metadata.get("hidden_gap_target_line_count"), 0))
    hidden_gap_line_count = max(hidden_gap_current_line_count, hidden_gap_target_line_count)
    if hidden_gap_line_count <= 0:
        return 0.0
    hidden_gap_bridge_count = max(1, _int_value(metadata.get("hidden_gap_bridge_count"), 1))
    bonus = 5.0 + min(1.5, (covered_window_count - 2) * 0.5)
    bonus += min(1.5, hidden_gap_line_count * 0.5)
    bonus += min(1.0, (hidden_gap_bridge_count - 1) * 0.5)
    if bool(metadata.get("explicit_hidden_gap_current_proof", False)):
        bonus += 1.0
    if not bool(metadata.get("partial_window_coverage", False)):
        bonus += 0.5
    if bool(metadata.get("shared_anchor_exact_region_block", False)):
        bonus += 1.5
        bonus += _shared_anchor_core_bonus(metadata, per_extra_core=0.5, cap=1.0)
        bonus += min(
            1.0,
            max(0, _int_value(metadata.get("shared_anchor_hybrid_component_count"), 0)) * 0.5,
        )
        if bool(metadata.get("shared_anchor_mixed_insert_delete", False)):
            bonus += 0.5
    return bonus


def _structured_edit_current_proof_region_block_adjustment(metadata: dict[str, object]) -> float:
    edit_source = str(metadata.get("edit_source", "")).strip()
    if edit_source not in {"workspace_preview", "workspace_preview_range"}:
        return 0.0
    if str(metadata.get("edit_kind", "")).strip() != "block_replace":
        return 0.0
    if not bool(metadata.get("current_proof_region_block", False)):
        return 0.0
    partial_region_block = bool(metadata.get("current_proof_partial_region_block", False))
    if bool(metadata.get("current_proof_partial_coverage", False)) and not partial_region_block:
        return 0.0
    covered_window_count = max(1, _int_value(metadata.get("covered_window_count"), 1))
    if covered_window_count < 2:
        return 0.0
    current_proof_span_count = max(
        0,
        _int_value(metadata.get("current_proof_span_count"), 0),
    )
    if current_proof_span_count < 2:
        return 0.0
    if partial_region_block:
        opaque_span_count = max(0, _int_value(metadata.get("current_proof_opaque_span_count"), 0))
        opaque_line_count = max(0, _int_value(metadata.get("current_proof_opaque_line_count"), 0))
        opaque_max_span_line_count = max(
            0,
            _int_value(metadata.get("current_proof_opaque_max_span_line_count"), 0),
        )
        opaque_boundary_touch_count = max(
            0,
            _int_value(metadata.get("current_proof_opaque_boundary_touch_count"), 0),
        )
        if opaque_span_count <= 0 or opaque_boundary_touch_count > 0:
            return 0.0
        bonus = 7.5 + min(1.0, (covered_window_count - 2) * 0.5)
        bonus += min(1.0, (current_proof_span_count - 1) * 0.25)
        bonus -= min(1.5, max(0, opaque_line_count - 1) * 0.25)
        bonus -= min(1.5, max(0, opaque_span_count - 1) * 0.5)
        bonus -= min(1.0, max(0, opaque_max_span_line_count - 1) * 0.5)
        if str(metadata.get("current_proof_partial_region_topology", "")).strip() == (
            "sparse_internal_multi_opaque"
        ):
            bonus -= 0.5
    else:
        bonus = 6.0 + min(1.5, (covered_window_count - 2) * 0.5)
        bonus += min(2.0, (current_proof_span_count - 1) * 0.5)
    if bool(metadata.get("explicit_hidden_gap_current_proof", False)):
        bonus += 1.0
    if bool(metadata.get("hidden_gap_current_from_line_span_proof", False)):
        bonus += 0.5
    if not bool(metadata.get("partial_window_coverage", False)):
        bonus += 0.5
    if bool(metadata.get("shared_anchor_exact_region_block", False)):
        bonus += 1.5
        bonus += _shared_anchor_core_bonus(metadata, per_extra_core=0.5, cap=1.0)
        bonus += min(
            1.0,
            max(0, _int_value(metadata.get("shared_anchor_hybrid_component_count"), 0)) * 0.5,
        )
        if bool(metadata.get("hidden_gap_target_from_expected_content", False)):
            bonus += 0.75
        if bool(metadata.get("shared_anchor_mixed_insert_delete", False)):
            bonus += 0.5
    return bonus


def _structured_edit_same_span_block_proof_quality_adjustment(metadata: dict[str, object]) -> float:
    edit_source = str(metadata.get("edit_source", "")).strip()
    if edit_source not in {"workspace_preview", "workspace_preview_range"}:
        return 0.0
    if str(metadata.get("edit_kind", "")).strip() != "block_replace":
        return 0.0
    alternative_count = max(0, _int_value(metadata.get("same_span_block_alternative_count"), 0))
    max_quality_score = max(0, _int_value(metadata.get("same_span_block_max_quality_score"), 0))
    if alternative_count < 2 or max_quality_score <= 0:
        return 0.0
    quality_gap = max(
        0,
        _int_value(
            metadata.get("same_span_block_quality_gap"),
            max_quality_score - _int_value(metadata.get("same_span_block_quality_score"), 0),
        ),
    )
    if quality_gap <= 0:
        bonus = 2.5
        if bool(metadata.get("current_proof_complete", False)):
            bonus += 1.5
        if max(0, _int_value(metadata.get("shared_anchor_hybrid_component_count"), 0)) >= 2:
            bonus += 1.0
        if max(0, _int_value(metadata.get("shared_anchor_core_count"), 0)) >= 2:
            bonus += 1.0
        return bonus
    penalty = min(12.0, quality_gap * 0.75)
    if bool(metadata.get("current_proof_partial_coverage", False)):
        penalty += 2.0
    penalty += min(
        3.0,
        max(0, _int_value(metadata.get("current_proof_missing_span_count"), 0)) * 1.0,
    )
    penalty += min(
        2.0,
        max(0, _int_value(metadata.get("current_proof_missing_line_count"), 0)) * 0.5,
    )
    return -penalty


def _structured_edit_bridge_run_frontier_adjustment(metadata: dict[str, object]) -> float:
    edit_source = str(metadata.get("edit_source", "")).strip()
    if edit_source not in {"workspace_preview", "workspace_preview_range"}:
        return 0.0
    if str(metadata.get("edit_kind", "")).strip() != "block_replace":
        return 0.0
    if not bool(metadata.get("bridged_hidden_gap_region_block", False)):
        return 0.0
    frontier_gap = max(0.0, _float_value(metadata.get("bridge_run_frontier_gap"), 0.0))
    partial_window_coverage = bool(metadata.get("partial_window_coverage", False))
    exact_localized_alternative_count = max(
        0,
        _int_value(metadata.get("bridge_run_exact_localized_alternative_count"), 0),
    )
    exact_localized_gap = max(0.0, _float_value(metadata.get("bridge_run_exact_localized_gap"), 0.0))
    inexact_window_count = max(0, _int_value(metadata.get("inexact_window_count"), 0))
    adjustment = 0.0
    if frontier_gap <= 0.0:
        adjustment += 2.0
    else:
        adjustment -= min(6.0, frontier_gap)
    if exact_localized_alternative_count > 0:
        if partial_window_coverage:
            adjustment -= 3.0 + min(4.0, exact_localized_gap)
        elif inexact_window_count > 0:
            adjustment -= 1.5 + min(2.5, exact_localized_gap * 0.75)
        elif exact_localized_gap > 0.0:
            adjustment -= min(1.5, exact_localized_gap * 0.5)
    return adjustment


def _structured_edit_exact_localized_bridge_competition_adjustment(
    metadata: dict[str, object]
) -> float:
    edit_source = str(metadata.get("edit_source", "")).strip()
    if edit_source not in {"workspace_preview", "workspace_preview_range"}:
        return 0.0
    if str(metadata.get("edit_kind", "")).strip() != "block_replace":
        return 0.0
    if bool(metadata.get("bridged_hidden_gap_region_block", False)):
        return 0.0
    covered_window_count = max(1, _int_value(metadata.get("covered_window_count"), 1))
    exact_window_count = max(0, _int_value(metadata.get("exact_window_count"), 0))
    inexact_window_count = max(0, _int_value(metadata.get("inexact_window_count"), 0))
    if not bool(metadata.get("exact_target_span", False)):
        return 0.0
    if exact_window_count < covered_window_count or inexact_window_count > 0:
        return 0.0
    partial_bridge_run_count = max(
        0,
        _int_value(metadata.get("bridge_run_partial_alternative_count"), 0),
    )
    if partial_bridge_run_count <= 0:
        return 0.0
    bridge_run_max_covered_window_count = max(
        covered_window_count,
        _int_value(metadata.get("bridge_run_max_covered_window_count"), covered_window_count),
    )
    bonus = 2.0 + min(2.0, (bridge_run_max_covered_window_count - covered_window_count) * 0.75)
    if bool(metadata.get("same_window_exact_block_alternative", False)):
        bonus += 0.5
    return bonus


def _expected_file_content_workspace_preview_fallback_adjustment(
    metadata: dict[str, object]
) -> float:
    if not bool(metadata.get("fallback_from_workspace_preview", False)):
        return 0.0
    region_ambiguity_count = max(
        0,
        _int_value(metadata.get("workspace_preview_hidden_gap_region_ambiguity_count"), 0),
    )
    current_ambiguity_count = max(
        0,
        _int_value(metadata.get("workspace_preview_hidden_gap_current_ambiguity_count"), 0),
    )
    target_equivalent_count = max(
        0,
        _int_value(
            metadata.get("workspace_preview_hidden_gap_current_target_equivalent_count"),
            0,
        ),
    )
    proof_required_count = max(
        0,
        _int_value(
            metadata.get("workspace_preview_hidden_gap_current_proof_required_count"),
            0,
        ),
    )
    if (
        region_ambiguity_count <= 1
        and current_ambiguity_count <= 1
        and target_equivalent_count <= 1
        and proof_required_count <= 0
    ):
        return 0.0
    bonus = 0.0
    if region_ambiguity_count > 1:
        bonus += 8.0 + min(4.0, max(0, region_ambiguity_count - 1) * 2.0)
    if current_ambiguity_count > 1:
        bonus += 10.0 + min(4.0, max(0, current_ambiguity_count - 1) * 2.0)
    if target_equivalent_count > 1:
        bonus += 10.0 + min(4.0, max(0, target_equivalent_count - 1) * 2.0)
    if proof_required_count > 0:
        bonus += 12.0 + min(3.0, max(0, proof_required_count - 1) * 1.5)
    best_block_proof_quality = max(
        0,
        _int_value(metadata.get("workspace_preview_best_block_proof_quality"), 0),
    )
    if (
        bool(metadata.get("workspace_preview_has_complete_current_proof", False))
        or bool(metadata.get("workspace_preview_has_explicit_hidden_gap_current_proof", False))
        or bool(metadata.get("workspace_preview_has_bridged_hidden_gap_region_block", False))
    ):
        bonus -= min(4.0, max(0, best_block_proof_quality - 8) * 0.25)
    return max(0.0, bonus)


def _structured_edit_precision_penalty(metadata: dict[str, object]) -> float:
    if str(metadata.get("edit_kind", "")).strip() != "multi_edit":
        return 0.0
    exact_window_count = max(0, _int_value(metadata.get("exact_window_count"), 0))
    inexact_window_count = max(0, _int_value(metadata.get("inexact_window_count"), 0))
    if exact_window_count <= 0 or inexact_window_count <= 0:
        return 0.0
    safe_inexact_window_count = max(0, _int_value(metadata.get("safe_inexact_window_count"), 0))
    unresolved_alias_pair_count = max(
        0,
        _int_value(metadata.get("overlap_component_unresolved_alias_pair_count"), 0),
    )
    frontier_gap = max(0.0, _float_value(metadata.get("overlap_component_frontier_gap"), 0.0))
    risky_inexact_window_count = max(0, inexact_window_count - safe_inexact_window_count)
    if (
        risky_inexact_window_count <= 0
        and _int_value(metadata.get("overlap_alias_pair_count"), 0) > 0
        and unresolved_alias_pair_count <= 0
    ):
        return max(0.0, _float_value(metadata.get("precision_penalty"), 0.0) - 1.0)
    penalty = 4.0
    penalty += _float_value(metadata.get("precision_penalty"), 0.0) * 2.0
    penalty += risky_inexact_window_count * 3.0
    if safe_inexact_window_count > 0:
        penalty += safe_inexact_window_count * 0.5
    if unresolved_alias_pair_count > 0:
        penalty += unresolved_alias_pair_count * 2.0
    if frontier_gap > 0:
        penalty += min(4.0, frontier_gap)
    if bool(metadata.get("allow_expected_write_fallback", False)):
        penalty += 3.0
    return penalty


def _structured_edit_hidden_gap_penalty(metadata: dict[str, object]) -> float:
    edit_source = str(metadata.get("edit_source", "")).strip()
    if edit_source not in {"workspace_preview", "workspace_preview_range"}:
        return 0.0
    if bool(metadata.get("bridged_hidden_gap_region_block", False)) and bool(
        metadata.get("explicit_hidden_gap_current_proof", False)
    ):
        return 0.0
    if not bool(metadata.get("hidden_gap_broader_region_required", False)):
        return 0.0
    hidden_gap_line_count = max(1, _int_value(metadata.get("hidden_gap_line_count"), 0))
    retained_window_count = max(
        1,
        _int_value(
            metadata.get("retained_window_count"),
            metadata.get("available_window_count"),
        ),
    )
    covered_window_count = max(
        1,
        _int_value(
            metadata.get("covered_window_count"),
            metadata.get("edit_window_count"),
        ),
    )
    uncovered_window_count = max(0, retained_window_count - covered_window_count)
    penalty = 6.0 + min(6.0, hidden_gap_line_count * 1.5)
    if uncovered_window_count > 0:
        penalty += uncovered_window_count * 1.5
    if bool(metadata.get("allow_expected_write_fallback", False)):
        penalty += 1.5
    if str(metadata.get("edit_kind", "")).strip() == "multi_edit":
        penalty += 2.0
    return -penalty


def _structured_edit_hidden_gap_region_ambiguity_penalty(metadata: dict[str, object]) -> float:
    edit_source = str(metadata.get("edit_source", "")).strip()
    if edit_source not in {"workspace_preview", "workspace_preview_range"}:
        return 0.0
    ambiguity_count = max(0, _int_value(metadata.get("hidden_gap_region_ambiguity_count"), 0))
    frontier_count = max(0, _int_value(metadata.get("hidden_gap_region_frontier_count"), 0))
    frontier_gap = max(0.0, _float_value(metadata.get("hidden_gap_region_frontier_gap"), 0.0))
    if ambiguity_count <= 1 and frontier_gap <= 0.0:
        return 0.0
    penalty = 0.0
    if ambiguity_count > 1:
        penalty += 7.0 + min(4.0, (ambiguity_count - 1) * 2.0)
        if bool(metadata.get("hidden_gap_region_ambiguous", False)):
            penalty += 1.0
        if bool(metadata.get("allow_expected_write_fallback", False)):
            penalty += 1.5
        if bool(metadata.get("exact_hidden_gap_region_block", False)):
            penalty += 1.5
        if _int_value(metadata.get("inexact_window_count"), 0) > 0:
            penalty += 1.0
        if str(metadata.get("edit_kind", "")).strip() == "multi_edit":
            penalty += 2.0
    if frontier_gap > 0.0:
        penalty += min(4.0, frontier_gap)
        if frontier_count > 1:
            penalty += 1.0
    return -penalty if penalty > 0.0 else 0.0


def _structured_edit_hidden_gap_bounded_alternative_penalty(metadata: dict[str, object]) -> float:
    edit_source = str(metadata.get("edit_source", "")).strip()
    if edit_source not in {"workspace_preview", "workspace_preview_range"}:
        return 0.0
    if max(
        0,
        _int_value(metadata.get("hidden_gap_current_line_count"), 0),
        _int_value(metadata.get("hidden_gap_target_line_count"), 0),
    ) <= 0:
        return 0.0
    if bool(metadata.get("explicit_hidden_gap_current_proof", False)):
        return 0.0
    if max(0, _int_value(metadata.get("hidden_gap_region_ambiguity_count"), 0)) <= 1:
        return 0.0
    bounded_count = max(0, _int_value(metadata.get("hidden_gap_bounded_alternative_count"), 0))
    bounded_gap = max(0.0, _float_value(metadata.get("hidden_gap_bounded_alternative_gap"), 0.0))
    if bounded_count <= 0:
        return 0.0
    penalty = max(0.0, 7.0 - min(7.0, bounded_gap))
    penalty += min(2.0, max(0, _int_value(metadata.get("hidden_gap_region_ambiguity_count"), 0) - 1))
    if bool(metadata.get("allow_expected_write_fallback", False)):
        penalty += 1.0
    if bool(metadata.get("exact_hidden_gap_region_block", False)):
        penalty += 2.0
    if str(metadata.get("edit_kind", "")).strip() == "block_replace":
        penalty += 1.0
    if bounded_gap <= 0.0:
        penalty += 1.5
    if bounded_count > 1:
        penalty += 0.5
    return -penalty if penalty > 0.0 else 0.0


def _structured_edit_hidden_gap_current_ambiguity_penalty(metadata: dict[str, object]) -> float:
    edit_source = str(metadata.get("edit_source", "")).strip()
    if edit_source not in {"workspace_preview", "workspace_preview_range"}:
        return 0.0
    if bool(metadata.get("explicit_hidden_gap_current_proof", False)):
        return 0.0
    current_line_count = max(0, _int_value(metadata.get("hidden_gap_current_line_count"), 0))
    target_line_count = max(0, _int_value(metadata.get("hidden_gap_target_line_count"), 0))
    ambiguity_count = max(0, _int_value(metadata.get("hidden_gap_current_ambiguity_count"), 0))
    frontier_count = max(0, _int_value(metadata.get("hidden_gap_current_frontier_count"), 0))
    frontier_gap = max(0.0, _float_value(metadata.get("hidden_gap_current_frontier_gap"), 0.0))
    current_dominant = current_line_count > target_line_count
    if ambiguity_count <= 1 and (frontier_gap <= 0.0 or not current_dominant):
        return 0.0
    penalty = 0.0
    if ambiguity_count > 1:
        penalty += 8.0 + min(5.0, (ambiguity_count - 1) * 2.0)
        if bool(metadata.get("hidden_gap_current_ambiguous", False)):
            penalty += 1.0
        if bool(metadata.get("allow_expected_write_fallback", False)):
            penalty += 1.5
        if bool(metadata.get("exact_hidden_gap_region_block", False)):
            penalty += 1.0
        if str(metadata.get("edit_kind", "")).strip() == "multi_edit":
            penalty += 2.0
    if frontier_gap > 0.0 and current_dominant:
        penalty += min(4.0, frontier_gap)
        if frontier_count > 1:
            penalty += 1.0
    return -penalty if penalty > 0.0 else 0.0


def _structured_edit_hidden_gap_current_target_equivalent_penalty(metadata: dict[str, object]) -> float:
    edit_source = str(metadata.get("edit_source", "")).strip()
    if edit_source not in {"workspace_preview", "workspace_preview_range"}:
        return 0.0
    if bool(metadata.get("explicit_hidden_gap_current_proof", False)):
        return 0.0
    current_line_count = max(0, _int_value(metadata.get("hidden_gap_current_line_count"), 0))
    target_line_count = max(0, _int_value(metadata.get("hidden_gap_target_line_count"), 0))
    if current_line_count <= target_line_count:
        return 0.0
    target_equivalent_count = max(
        0,
        _int_value(metadata.get("hidden_gap_current_target_equivalent_count"), 0),
    )
    if target_equivalent_count <= 1:
        return 0.0
    target_equivalent_gap = max(
        0.0,
        _float_value(metadata.get("hidden_gap_current_target_equivalent_gap"), 0.0),
    )
    penalty = 7.0 + min(4.0, (target_equivalent_count - 1) * 1.5)
    if bool(metadata.get("hidden_gap_current_target_equivalent_ambiguous", False)):
        penalty += 1.0
    if bool(metadata.get("allow_expected_write_fallback", False)):
        penalty += 1.0
    if str(metadata.get("edit_kind", "")).strip() == "block_replace":
        penalty += 1.0
    if target_equivalent_gap <= 0.0:
        penalty += 2.0
    else:
        penalty += max(0.0, 2.5 - min(2.5, target_equivalent_gap))
    return -penalty


def _structured_edit_explicit_current_span_proof_adjustment(metadata: dict[str, object]) -> float:
    edit_source = str(metadata.get("edit_source", "")).strip()
    if edit_source not in {"workspace_preview", "workspace_preview_range"}:
        return 0.0
    if not bool(metadata.get("explicit_current_span_proof", False)):
        return 0.0
    edit_kind = str(metadata.get("edit_kind", "")).strip()
    covered_window_count = max(0, _int_value(metadata.get("covered_window_count"), 0))
    available_window_count = max(1, _int_value(metadata.get("available_window_count"), covered_window_count or 1))
    exact_window_count = max(0, _int_value(metadata.get("exact_window_count"), 0))
    bonus = 0.0
    if edit_kind == "block_replace":
        bonus = 5.0
        if bool(metadata.get("partial_window_coverage", False)):
            bonus += 1.5
    elif (
        edit_kind == "multi_edit"
        and covered_window_count >= available_window_count
        and exact_window_count >= covered_window_count
    ):
        bonus = 14.0
    return bonus


def _structured_edit_hidden_gap_current_proof_penalty(metadata: dict[str, object]) -> float:
    edit_source = str(metadata.get("edit_source", "")).strip()
    if edit_source not in {"workspace_preview", "workspace_preview_range"}:
        return 0.0
    if bool(metadata.get("explicit_hidden_gap_current_proof", False)):
        return 0.0
    if max(0, _int_value(metadata.get("hidden_gap_current_proof_required_count"), 0)) <= 0:
        return 0.0
    current_line_count = max(0, _int_value(metadata.get("hidden_gap_current_line_count"), 0))
    target_line_count = max(0, _int_value(metadata.get("hidden_gap_target_line_count"), 0))
    if current_line_count <= target_line_count:
        return 0.0
    penalty = 9.0
    if bool(metadata.get("hidden_gap_current_proof_required", False)):
        penalty += 1.0
    if bool(metadata.get("allow_expected_write_fallback", False)):
        penalty += 1.5
    if str(metadata.get("edit_kind", "")).strip() == "block_replace":
        penalty += 1.0
    if _int_value(metadata.get("inexact_window_count"), 0) > 0:
        penalty += 1.0
    penalty += min(2.0, max(0, current_line_count - target_line_count) * 0.5)
    bounded_gap = max(0.0, _float_value(metadata.get("hidden_gap_bounded_alternative_gap"), 0.0))
    if max(0, _int_value(metadata.get("hidden_gap_bounded_alternative_count"), 0)) > 0 and bounded_gap <= 0.0:
        penalty += 1.5
    frontier_gap = max(0.0, _float_value(metadata.get("hidden_gap_current_frontier_gap"), 0.0))
    if frontier_gap > 0.0:
        penalty += min(3.0, frontier_gap)
    return -penalty


def _structured_edit_hidden_gap_current_partial_proof_penalty(
    metadata: dict[str, object]
) -> float:
    edit_source = str(metadata.get("edit_source", "")).strip()
    if edit_source not in {"workspace_preview", "workspace_preview_range"}:
        return 0.0
    partial_proof_count = max(
        0,
        _int_value(metadata.get("hidden_gap_current_partial_proof_count"), 0),
    )
    if partial_proof_count <= 0:
        return 0.0
    missing_line_count = max(
        0,
        _int_value(metadata.get("hidden_gap_current_partial_proof_missing_line_count"), 0),
    )
    missing_span_count = max(
        0,
        _int_value(metadata.get("hidden_gap_current_partial_proof_span_count"), 0),
    )
    partial_region_block = bool(metadata.get("current_proof_partial_region_block", False))
    if partial_region_block:
        opaque_span_count = max(0, _int_value(metadata.get("current_proof_opaque_span_count"), 0))
        opaque_line_count = max(0, _int_value(metadata.get("current_proof_opaque_line_count"), 0))
        opaque_max_span_line_count = max(
            0,
            _int_value(metadata.get("current_proof_opaque_max_span_line_count"), 0),
        )
        opaque_boundary_touch_count = max(
            0,
            _int_value(metadata.get("current_proof_opaque_boundary_touch_count"), 0),
        )
        penalty = 1.5
        penalty += min(2.0, max(0, opaque_span_count - 1) * 0.75)
        penalty += min(1.5, max(0, opaque_line_count - 1) * 0.25)
        penalty += min(1.5, max(0, opaque_max_span_line_count - 1) * 0.75)
        penalty += min(1.0, max(0, missing_span_count - opaque_span_count) * 0.5)
        if opaque_boundary_touch_count > 0:
            penalty += 4.0
        if bool(metadata.get("allow_expected_write_fallback", False)):
            penalty += 0.5
        return -penalty
    penalty = 8.0
    penalty += min(4.0, partial_proof_count * 1.5)
    penalty += min(4.0, missing_line_count * 0.75)
    penalty += min(2.0, max(0, missing_span_count - 1) * 0.5)
    penalty += min(
        2.0,
        max(
            0,
            _int_value(metadata.get("hidden_gap_current_partial_proof_blocking_opaque_span_count"), 0),
        )
        * 1.0,
    )
    penalty += min(
        2.0,
        max(
            0,
            _int_value(metadata.get("hidden_gap_current_partial_proof_coarse_opaque_span_count"), 0),
        )
        * 1.0,
    )
    if bool(metadata.get("hidden_gap_current_partial_proof_mixed_opaque_topology", False)):
        penalty += 1.0
    if bool(metadata.get("allow_expected_write_fallback", False)):
        penalty += 1.5
    if str(metadata.get("edit_kind", "")).strip() == "block_replace":
        penalty += 1.0
    if str(metadata.get("edit_kind", "")).strip() == "multi_edit":
        penalty += 1.5
    return -penalty


def _structured_edit_coverage_adjustment(metadata: dict[str, object]) -> float:
    edit_source = str(metadata.get("edit_source", "")).strip()
    if edit_source not in {"workspace_preview", "workspace_preview_range"}:
        return 0.0
    edit_kind = str(metadata.get("edit_kind", "")).strip()
    retained_window_count = max(
        1,
        _int_value(
            metadata.get("retained_window_count"),
            metadata.get("available_window_count"),
        ),
    )
    covered_window_count = max(
        1,
        _int_value(
            metadata.get("covered_window_count"),
            metadata.get("edit_window_count"),
        ),
    )
    total_window_count = max(
        retained_window_count,
        _int_value(metadata.get("total_window_count"), retained_window_count),
    )
    penalty = 0.0
    if covered_window_count < retained_window_count:
        uncovered_window_count = retained_window_count - covered_window_count
        penalty += 4.5 + uncovered_window_count * 2.0
    if total_window_count > retained_window_count:
        hidden_window_count = total_window_count - retained_window_count
        penalty += 5.0 + hidden_window_count * 1.5
    if penalty <= 0.0:
        return 0.0
    if edit_kind == "multi_edit":
        penalty -= 1.0
    return -max(0.0, penalty)


def _edit_kind_bonus(edit_kind: str) -> float:
    return {
        "multi_edit": 9.0,
        "token_replace": 8.0,
        "token_delete": 7.5,
        "token_insert": 7.0,
        "line_replace": 6.0,
        "multi_window_structured_edit": 6.5,
        "line_delete": 5.0,
        "line_insert": 5.0,
        "block_replace": 4.0,
    }.get(str(edit_kind).strip(), 0.0)


def _line_edit_fragments(before_line: str, after_line: str) -> dict[str, str] | None:
    if before_line == after_line:
        return None
    prefix_length = 0
    while (
        prefix_length < len(before_line)
        and prefix_length < len(after_line)
        and before_line[prefix_length] == after_line[prefix_length]
    ):
        prefix_length += 1
    suffix_length = 0
    while (
        suffix_length < (len(before_line) - prefix_length)
        and suffix_length < (len(after_line) - prefix_length)
        and before_line[len(before_line) - 1 - suffix_length] == after_line[len(after_line) - 1 - suffix_length]
    ):
        suffix_length += 1
    before_fragment = before_line[prefix_length : len(before_line) - suffix_length if suffix_length else len(before_line)]
    after_fragment = after_line[prefix_length : len(after_line) - suffix_length if suffix_length else len(after_line)]
    if before_fragment == after_fragment:
        return None
    if not before_fragment and not after_fragment:
        return None
    if before_fragment == before_line and after_fragment == after_line:
        return None
    if "\n" in before_fragment or "\n" in after_fragment:
        return None
    prefix = before_line[:prefix_length]
    suffix = before_line[len(before_line) - suffix_length :] if suffix_length else ""
    return {
        "prefix": prefix,
        "before_fragment": before_fragment,
        "after_fragment": after_fragment,
        "suffix": suffix,
    }


def _render_structured_edit_command(step: dict[str, object]) -> str:
    path = str(step.get("path", "")).strip()
    if not path:
        return ""
    edit_kind = str(step.get("edit_kind", "rewrite")).strip() or "rewrite"
    if edit_kind == "multi_edit":
        raw_steps = step.get("steps", [])
        if not isinstance(raw_steps, list):
            return ""
        commands: list[str] = []
        for item in raw_steps:
            if not isinstance(item, dict):
                return ""
            command = _render_structured_edit_command(item)
            if not command:
                return ""
            commands.append(command)
        return " && ".join(commands)
    if edit_kind == "token_replace":
        replacements = step.get("replacements", [])
        if not isinstance(replacements, list):
            return ""
        commands = _render_token_replace_commands(path, replacements)
        return " && ".join(commands)
    if edit_kind == "token_insert":
        replacements = step.get("replacements", [])
        if not isinstance(replacements, list):
            return ""
        commands = _render_token_insert_commands(path, replacements)
        return " && ".join(commands)
    if edit_kind == "token_delete":
        replacements = step.get("replacements", [])
        if not isinstance(replacements, list):
            return ""
        commands = _render_token_delete_commands(path, replacements)
        return " && ".join(commands)
    if edit_kind == "line_replace":
        replacements = step.get("replacements", [])
        if not isinstance(replacements, list):
            return ""
        commands = _render_line_replace_commands(path, replacements)
        return " && ".join(commands)
    if edit_kind == "line_delete":
        deletion = step.get("deletion", {})
        if not isinstance(deletion, dict):
            return ""
        return _render_line_delete_command(path, deletion)
    if edit_kind == "line_insert":
        insertion = step.get("insertion", {})
        if not isinstance(insertion, dict):
            return ""
        return _render_line_insert_command(path, insertion)
    if edit_kind == "block_replace":
        replacement = step.get("replacement", {})
        if not isinstance(replacement, dict):
            return ""
        return _render_block_replace_command(path, replacement)
    return ""


def _render_line_replace_commands(path: str, replacements: list[dict[str, object]]) -> list[str]:
    commands: list[str] = []
    for replacement in replacements:
        try:
            line_number = int(replacement.get("line_number", 0))
        except (TypeError, ValueError):
            continue
        before_line = str(replacement.get("before_line", ""))
        after_line = str(replacement.get("after_line", ""))
        if line_number <= 0 or before_line == after_line:
            continue
        script = (
            f"{line_number}s#^{_sed_regex_escape(before_line)}$#"
            f"{_sed_replacement_escape(after_line)}#"
        )
        commands.append(f"sed -i {shlex.quote(script)} {shlex.quote(path)}")
    return commands


def _render_block_replace_command(path: str, replacement: dict[str, object]) -> str:
    try:
        start_line = int(replacement.get("start_line", 0))
        end_line = int(replacement.get("end_line", 0))
    except (TypeError, ValueError):
        return ""
    after_lines = [str(line) for line in replacement.get("after_lines", [])]
    if start_line <= 0 or end_line < start_line:
        return ""
    if not after_lines:
        script = f"{start_line},{end_line}d"
        return f"sed -i {shlex.quote(script)} {shlex.quote(path)}"
    replacement_body = "\\\n".join(_sed_block_text_escape(line) for line in after_lines)
    script = f"{start_line},{end_line}c\\\n{replacement_body}"
    return f"sed -i {shlex.quote(script)} {shlex.quote(path)}"


def _render_line_delete_command(path: str, deletion: dict[str, object]) -> str:
    try:
        start_line = int(deletion.get("start_line", 0))
        end_line = int(deletion.get("end_line", 0))
    except (TypeError, ValueError):
        return ""
    if start_line <= 0 or end_line < start_line:
        return ""
    script = f"{start_line},{end_line}d"
    return f"sed -i '{script}' {shlex.quote(path)}"


def _render_line_insert_command(path: str, insertion: dict[str, object]) -> str:
    try:
        line_number = int(insertion.get("line_number", 0))
    except (TypeError, ValueError):
        return ""
    raw_after_lines = insertion.get("after_lines", insertion.get("inserted_lines", []))
    after_lines = [str(line) for line in raw_after_lines]
    mode = str(insertion.get("mode", "before")).strip() or "before"
    if line_number <= 0 or not after_lines:
        return ""
    insertion_body = "\\\n".join(_sed_block_text_escape(line) for line in after_lines)
    if mode == "append":
        script = f"$a\\\n{insertion_body}"
    else:
        script = f"{line_number}i\\\n{insertion_body}"
    return f"sed -i {shlex.quote(script)} {shlex.quote(path)}"


def _render_token_replace_commands(path: str, replacements: list[dict[str, object]]) -> list[str]:
    commands: list[str] = []
    for replacement in replacements:
        try:
            line_number = int(replacement.get("line_number", 0))
        except (TypeError, ValueError):
            continue
        before_fragment = str(replacement.get("before_fragment", ""))
        after_fragment = str(replacement.get("after_fragment", ""))
        if line_number <= 0 or not before_fragment or before_fragment == after_fragment:
            continue
        script = (
            f"{line_number}s#{_sed_regex_escape(before_fragment)}#"
            f"{_sed_replacement_escape(after_fragment)}#"
        )
        commands.append(f"sed -i {shlex.quote(script)} {shlex.quote(path)}")
    return commands


def _render_token_insert_commands(path: str, replacements: list[dict[str, object]]) -> list[str]:
    commands: list[str] = []
    for replacement in replacements:
        try:
            line_number = int(replacement.get("line_number", 0))
        except (TypeError, ValueError):
            continue
        before_line = str(replacement.get("before_line", ""))
        after_line = str(replacement.get("after_line", ""))
        if line_number <= 0 or before_line == after_line:
            continue
        script = (
            f"{line_number}s#^{_sed_regex_escape(before_line)}$#"
            f"{_sed_replacement_escape(after_line)}#"
        )
        commands.append(f"sed -i {shlex.quote(script)} {shlex.quote(path)}")
    return commands


def _render_token_delete_commands(path: str, replacements: list[dict[str, object]]) -> list[str]:
    commands: list[str] = []
    for replacement in replacements:
        try:
            line_number = int(replacement.get("line_number", 0))
        except (TypeError, ValueError):
            continue
        before_line = str(replacement.get("before_line", ""))
        after_line = str(replacement.get("after_line", ""))
        if line_number <= 0 or before_line == after_line:
            continue
        script = (
            f"{line_number}s#^{_sed_regex_escape(before_line)}$#"
            f"{_sed_replacement_escape(after_line)}#"
        )
        commands.append(f"sed -i {shlex.quote(script)} {shlex.quote(path)}")
    return commands


def _sed_regex_escape(value: str) -> str:
    escaped = re.escape(value)
    return escaped.replace("#", r"\#")


def _sed_replacement_escape(value: str) -> str:
    return value.replace("\\", r"\\").replace("&", r"\&").replace("#", r"\#")


def _sed_block_text_escape(value: str) -> str:
    return value.replace("\\", r"\\")


def _sha1_text(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()
