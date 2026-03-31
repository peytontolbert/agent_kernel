from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

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
    for path, content in unsatisfied_expected.items():
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
                    "provenance": _template_provenance(proposal_policy, benchmark_family, "expected_file_content"),
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
        if proposal_source == "cleanup_forbidden_file":
            score += _float_value(proposal_policy.get("cleanup_template_bonus"), 0.0)
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
    return True


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
