from __future__ import annotations

from ..ops.episode_store import iter_episode_documents


def score_command(model, summary: dict[str, object], command: str) -> int:
    normalized = str(command)
    score = 0
    unresolved_paths = {
        str(path).strip()
        for key in (
            "missing_expected_artifacts",
            "unsatisfied_expected_contents",
            "present_forbidden_artifacts",
            "changed_preserved_artifacts",
            "workflow_expected_changed_paths",
            "workflow_generated_paths",
            "workflow_report_paths",
        )
        for path in summary.get(key, [])
        if str(path).strip()
    }
    for path in summary.get("missing_expected_artifacts", []):
        if path and path in normalized:
            score += model._control_int("expected_artifact_score_weight", 3) + 2
    for path in summary.get("expected_artifacts", []):
        if path and path in normalized:
            score += model._control_int("expected_artifact_score_weight", 3)
    cleanup_command = "rm " in normalized or "rm -f " in normalized or "unlink " in normalized
    for path in summary.get("present_forbidden_artifacts", []):
        if path and path in normalized and cleanup_command:
            score += model._control_int("forbidden_cleanup_score_weight", 4)
    for path in summary.get("preserved_artifacts", []):
        if path and path in normalized:
            score += model._control_int("preserved_artifact_score_weight", 2)
    for path in summary.get("forbidden_artifacts", []):
        if path and path in normalized:
            score -= model._control_int("forbidden_artifact_penalty", 5)
    if str(summary.get("horizon", "")) == "long_horizon":
        if "mkdir -p " in normalized or "cp " in normalized:
            score += model._control_int("long_horizon_scaffold_bonus", 1)
    for branch in summary.get("workflow_branch_targets", []):
        if branch and branch in normalized:
            score += model._control_int("workflow_branch_target_score_weight", 4)
    for path in summary.get("workflow_expected_changed_paths", []):
        if path and path in normalized:
            score += model._control_int("workflow_changed_path_score_weight", 3)
    for path in summary.get("workflow_generated_paths", []):
        if path and path in normalized:
            score += model._control_int("workflow_generated_path_score_weight", 3)
    for path in summary.get("workflow_report_paths", []):
        if path and path in normalized:
            score += model._control_int("workflow_report_path_score_weight", 2)
    for path in summary.get("workflow_preserved_paths", []):
        if path and path in normalized:
            score += model._control_int("workflow_preserved_path_score_weight", 2)
    if summary.get("workflow_required_tests") and ("test" in normalized or "./" in normalized):
        score += model._control_int("required_tests_score_weight", 2)
    if summary.get("workflow_required_merges") and "git merge" in normalized:
        score += model._control_int("required_merges_score_weight", 4)
    empirical = empirical_command_prediction(model, summary, command)
    direct_state_alignment = any(path in normalized for path in unresolved_paths)
    path_overlap_score = float(empirical.get("path_overlap_score", 0.0) or 0.0)
    empirical_support = 1.0 if direct_state_alignment else (0.5 if path_overlap_score > 0.0 else 0.0)
    predicted_progress_gain = float(empirical.get("predicted_progress_gain", 0.0) or 0.0)
    score += int(
        round(
            predicted_progress_gain
            * model._control_float("empirical_progress_gain_weight", 3.0)
            * empirical_support
        )
    )
    score += int(
        round(
            float(empirical.get("predicted_verifier_delta", 0.0) or 0.0)
            * model._control_float("empirical_verifier_delta_weight", 1.5)
            * empirical_support
        )
    )
    score += int(
        round(
            float(empirical.get("pass_rate", 0.0) or 0.0)
            * model._control_float("empirical_pass_rate_weight", 2.0)
            * empirical_support
        )
    )
    score -= int(
        round(
            max(0.0, 1.0 - float(empirical.get("pass_rate", 0.0) or 0.0))
            * model._control_float("empirical_low_pass_rate_penalty_weight", 2.5)
            * empirical_support
        )
    )
    score += int(round(path_overlap_score * model._control_float("empirical_path_overlap_weight", 2.0)))
    score -= int(
        round(
            float(empirical.get("regression_rate", 0.0) or 0.0)
            * model._control_float("empirical_regression_penalty_weight", 3.0)
            * empirical_support
        )
    )
    return score


def simulate_command_effect(model, summary: dict[str, object], command: str) -> dict[str, object]:
    normalized = str(command)
    predicted_outputs = [
        path for path in summary.get("expected_artifacts", []) if path and path in normalized
    ]
    predicted_conflicts = [
        path for path in summary.get("forbidden_artifacts", []) if path and path in normalized
    ]
    predicted_preserved = [
        path for path in summary.get("preserved_artifacts", []) if path and path in normalized
    ]
    predicted_workflow_paths = [
        path
        for path in (
            list(summary.get("workflow_expected_changed_paths", []))
            + list(summary.get("workflow_generated_paths", []))
            + list(summary.get("workflow_report_paths", []))
        )
        if path and path in normalized
    ]
    predicted_progress_gain = 0
    for path in summary.get("missing_expected_artifacts", []):
        if path and path in normalized:
            predicted_progress_gain += 1
    cleanup_command = "rm " in normalized or "rm -f " in normalized or "unlink " in normalized
    for path in summary.get("present_forbidden_artifacts", []):
        if path and path in normalized and cleanup_command:
            predicted_progress_gain += 1
    empirical = empirical_command_prediction(model, summary, command)
    empirical_progress_gain = float(empirical.get("predicted_progress_gain", 0.0) or 0.0)
    predicted_progress_gain = max(predicted_progress_gain, empirical_progress_gain)
    return {
        "predicted_outputs": predicted_outputs,
        "predicted_conflicts": predicted_conflicts,
        "predicted_preserved": predicted_preserved,
        "predicted_workflow_paths": predicted_workflow_paths,
        "predicted_progress_gain": predicted_progress_gain,
        "predicted_verifier_delta": float(empirical.get("predicted_verifier_delta", 0.0) or 0.0),
        "predicted_changed_paths": list(empirical.get("predicted_changed_paths", [])),
        "empirical_prior": empirical,
        "score": score_command(model, summary, command),
    }


def simulate_command_sequence_effect(
    model,
    summary: dict[str, object],
    commands: list[str],
) -> dict[str, object]:
    sequence = [str(command).strip() for command in commands if str(command).strip()]
    if not sequence:
        return {
            "sequence": [],
            "predicted_progress_gain": 0.0,
            "predicted_verifier_delta": 0.0,
            "predicted_changed_paths": [],
            "score": 0.0,
            "sequence_alignment_bonus": 0.0,
        }
    rolled_summary = dict(summary)
    predicted_changed_paths: list[str] = []
    predicted_progress_gain = 0.0
    predicted_verifier_delta = 0.0
    total_score = 0.0
    for command in sequence:
        effect = simulate_command_effect(model, rolled_summary, command)
        predicted_progress_gain += float(effect.get("predicted_progress_gain", 0.0) or 0.0)
        predicted_verifier_delta += float(effect.get("predicted_verifier_delta", 0.0) or 0.0)
        total_score += float(effect.get("score", 0.0) or 0.0)
        for path in effect.get("predicted_changed_paths", []):
            normalized = str(path).strip()
            if normalized and normalized not in predicted_changed_paths:
                predicted_changed_paths.append(normalized)
        rolled_summary = _rolled_summary_after_effect(rolled_summary, effect)
    sequence_alignment_bonus = _semantic_sequence_alignment_bonus(summary, sequence)
    total_score += sequence_alignment_bonus
    return {
        "sequence": sequence,
        "predicted_progress_gain": round(predicted_progress_gain, 4),
        "predicted_verifier_delta": round(predicted_verifier_delta, 4),
        "predicted_changed_paths": predicted_changed_paths,
        "score": round(total_score, 4),
        "sequence_alignment_bonus": round(sequence_alignment_bonus, 4),
    }


def empirical_command_prediction(model, summary: dict[str, object], command: str) -> dict[str, object]:
    learned = state_conditioned_command_prediction(model, summary, command)
    bootstrap = bootstrap_command_prediction(model, summary, command)
    semantic = semantic_command_prediction(model, summary, command)
    if learned and bootstrap:
        bootstrap_weight = min(
            float(max(0.0, model._control_float("bootstrap_prior_weight", 2.0))),
            float(bootstrap.get("sample_count", 0) or 0),
        )
        learned_weight = float(learned.get("sample_count", 0) or 0)
        semantic_weight = min(
            float(max(0.0, model._control_float("semantic_prior_weight", 2.0))),
            float(semantic.get("sample_count", 0) or 0),
        )
        total_weight = learned_weight + bootstrap_weight + semantic_weight
        if total_weight > 0.0:
            changed_path_counts = merge_changed_path_counts(
                dict(learned.get("changed_path_counts", {})),
                dict(bootstrap.get("changed_path_counts", {})),
                learned_weight=learned_weight,
                bootstrap_weight=bootstrap_weight,
            )
            changed_path_counts = merge_changed_path_counts(
                changed_path_counts,
                dict(semantic.get("changed_path_counts", {})),
                learned_weight=1.0,
                bootstrap_weight=semantic_weight,
            )
            combined = {
                "sample_count": int(learned.get("sample_count", 0) or 0),
                "bootstrap_weight": round(bootstrap_weight, 3),
                "semantic_weight": round(semantic_weight, 3),
                "pass_rate": round(
                    (
                        float(learned.get("pass_rate", 0.0) or 0.0) * learned_weight
                        + float(bootstrap.get("pass_rate", 0.0) or 0.0) * bootstrap_weight
                        + float(semantic.get("pass_rate", 0.0) or 0.0) * semantic_weight
                    )
                    / total_weight,
                    4,
                ),
                "regression_rate": round(
                    (
                        float(learned.get("regression_rate", 0.0) or 0.0) * learned_weight
                        + float(bootstrap.get("regression_rate", 0.0) or 0.0) * bootstrap_weight
                        + float(semantic.get("regression_rate", 0.0) or 0.0) * semantic_weight
                    )
                    / total_weight,
                    4,
                ),
                "predicted_progress_gain": round(
                    (
                        float(learned.get("predicted_progress_gain", 0.0) or 0.0) * learned_weight
                        + float(bootstrap.get("predicted_progress_gain", 0.0) or 0.0) * bootstrap_weight
                        + float(semantic.get("predicted_progress_gain", 0.0) or 0.0) * semantic_weight
                    )
                    / total_weight,
                    4,
                ),
                "predicted_verifier_delta": round(
                    (
                        float(learned.get("predicted_verifier_delta", 0.0) or 0.0) * learned_weight
                        + float(bootstrap.get("predicted_verifier_delta", 0.0) or 0.0) * bootstrap_weight
                        + float(semantic.get("predicted_verifier_delta", 0.0) or 0.0) * semantic_weight
                    )
                    / total_weight,
                    4,
                ),
                "prediction_source": "learned_state_model",
                "state_signature": str(learned.get("state_signature", "")).strip(),
                "command_pattern": str(learned.get("command_pattern", "")).strip()
                or str(bootstrap.get("command_pattern", "")).strip(),
                "learned_state_model": dict(learned),
                "bootstrap_prior": dict(bootstrap),
                "semantic_prior": dict(semantic),
            }
            combined.update(
                path_prediction_metrics(
                    summary,
                    changed_path_counts=changed_path_counts,
                )
            )
            return combined
    if semantic and learned:
        semantic_weight = min(
            float(max(0.0, model._control_float("semantic_prior_weight", 2.0))),
            float(semantic.get("sample_count", 0) or 0),
        )
        learned_weight = float(learned.get("sample_count", 0) or 0)
        total_weight = learned_weight + semantic_weight
        if total_weight > 0.0:
            changed_path_counts = merge_changed_path_counts(
                dict(learned.get("changed_path_counts", {})),
                dict(semantic.get("changed_path_counts", {})),
                learned_weight=learned_weight,
                bootstrap_weight=semantic_weight,
            )
            payload = {
                "sample_count": int(learned.get("sample_count", 0) or 0),
                "semantic_weight": round(semantic_weight, 3),
                "pass_rate": round(
                    (
                        float(learned.get("pass_rate", 0.0) or 0.0) * learned_weight
                        + float(semantic.get("pass_rate", 0.0) or 0.0) * semantic_weight
                    )
                    / total_weight,
                    4,
                ),
                "regression_rate": round(
                    (
                        float(learned.get("regression_rate", 0.0) or 0.0) * learned_weight
                        + float(semantic.get("regression_rate", 0.0) or 0.0) * semantic_weight
                    )
                    / total_weight,
                    4,
                ),
                "predicted_progress_gain": round(
                    (
                        float(learned.get("predicted_progress_gain", 0.0) or 0.0) * learned_weight
                        + float(semantic.get("predicted_progress_gain", 0.0) or 0.0) * semantic_weight
                    )
                    / total_weight,
                    4,
                ),
                "predicted_verifier_delta": round(
                    (
                        float(learned.get("predicted_verifier_delta", 0.0) or 0.0) * learned_weight
                        + float(semantic.get("predicted_verifier_delta", 0.0) or 0.0) * semantic_weight
                    )
                    / total_weight,
                    4,
                ),
                "prediction_source": "learned_state_model",
                "state_signature": str(learned.get("state_signature", "")).strip(),
                "command_pattern": str(learned.get("command_pattern", "")).strip(),
                "learned_state_model": dict(learned),
                "semantic_prior": dict(semantic),
            }
            payload.update(path_prediction_metrics(summary, changed_path_counts=changed_path_counts))
            return payload
    if semantic and bootstrap:
        semantic_weight = min(
            float(max(0.0, model._control_float("semantic_prior_weight", 2.0))),
            float(semantic.get("sample_count", 0) or 0),
        )
        bootstrap_weight = min(
            float(max(0.0, model._control_float("bootstrap_prior_weight", 2.0))),
            float(bootstrap.get("sample_count", 0) or 0),
        )
        total_weight = bootstrap_weight + semantic_weight
        if total_weight > 0.0:
            changed_path_counts = merge_changed_path_counts(
                dict(bootstrap.get("changed_path_counts", {})),
                dict(semantic.get("changed_path_counts", {})),
                learned_weight=bootstrap_weight,
                bootstrap_weight=semantic_weight,
            )
            payload = {
                "sample_count": int(bootstrap.get("sample_count", 0) or 0),
                "bootstrap_weight": round(bootstrap_weight, 3),
                "semantic_weight": round(semantic_weight, 3),
                "pass_rate": round(
                    (
                        float(bootstrap.get("pass_rate", 0.0) or 0.0) * bootstrap_weight
                        + float(semantic.get("pass_rate", 0.0) or 0.0) * semantic_weight
                    )
                    / total_weight,
                    4,
                ),
                "regression_rate": round(
                    (
                        float(bootstrap.get("regression_rate", 0.0) or 0.0) * bootstrap_weight
                        + float(semantic.get("regression_rate", 0.0) or 0.0) * semantic_weight
                    )
                    / total_weight,
                    4,
                ),
                "predicted_progress_gain": round(
                    (
                        float(bootstrap.get("predicted_progress_gain", 0.0) or 0.0) * bootstrap_weight
                        + float(semantic.get("predicted_progress_gain", 0.0) or 0.0) * semantic_weight
                    )
                    / total_weight,
                    4,
                ),
                "predicted_verifier_delta": round(
                    (
                        float(bootstrap.get("predicted_verifier_delta", 0.0) or 0.0) * bootstrap_weight
                        + float(semantic.get("predicted_verifier_delta", 0.0) or 0.0) * semantic_weight
                    )
                    / total_weight,
                    4,
                ),
                "prediction_source": "bootstrap_prior",
                "command_pattern": str(bootstrap.get("command_pattern", "")).strip(),
                "bootstrap_prior": dict(bootstrap),
                "semantic_prior": dict(semantic),
            }
            payload.update(path_prediction_metrics(summary, changed_path_counts=changed_path_counts))
            return payload
    if learned:
        payload = dict(learned)
        payload["prediction_source"] = "learned_state_model"
        payload["learned_state_model"] = dict(learned)
        return payload
    if semantic:
        payload = dict(semantic)
        payload["prediction_source"] = "semantic_memory"
        payload["semantic_prior"] = dict(semantic)
        return payload
    if bootstrap:
        payload = dict(bootstrap)
        payload["prediction_source"] = "bootstrap_prior"
        payload["bootstrap_prior"] = dict(bootstrap)
        return payload
    return {}


def semantic_command_prediction(model, summary: dict[str, object], command: str) -> dict[str, object]:
    del model
    normalized = str(command).strip()
    if not normalized:
        return {}
    semantic_episodes = summary.get("semantic_episodes", [])
    if not isinstance(semantic_episodes, list):
        semantic_episodes = []
    matched = []
    changed_path_counts: dict[str, float] = {}
    command_pattern_value = command_pattern(normalized)
    for raw in semantic_episodes:
        if not isinstance(raw, dict):
            continue
        episode = dict(raw)
        changed_paths = [
            str(value).strip()
            for value in episode.get("changed_paths", [])
            if str(value).strip()
        ]
        verifier_obligations = [
            str(value).strip()
            for value in episode.get("verifier_obligations", [])
            if str(value).strip()
        ]
        recovery_trace = episode.get("recovery_trace", {})
        recovery_trace = dict(recovery_trace) if isinstance(recovery_trace, dict) else {}
        path_matches = [path for path in changed_paths if path in normalized]
        report_match = any(path in normalized for path in summary.get("workflow_report_paths", []) if str(path).strip())
        recovery_command = str(recovery_trace.get("recovery_command", "")).strip()
        failed_command = str(recovery_trace.get("failed_command", "")).strip()
        direct_recovery_match = bool(recovery_command and recovery_command == normalized)
        repeated_failure_match = bool(failed_command and failed_command == normalized)
        if not (path_matches or report_match or direct_recovery_match or repeated_failure_match):
            continue
        success = bool(episode.get("success", False))
        obligation_pressure = min(2, len(verifier_obligations))
        if repeated_failure_match:
            progress_gain = 0.0
            verifier_delta = -1.0
            regression = 1.0
            pass_rate = 0.0
        elif success and (path_matches or direct_recovery_match):
            progress_gain = 1.0 + 0.5 * len(path_matches)
            verifier_delta = 1.0 + 0.5 * obligation_pressure
            regression = 0.0
            pass_rate = 1.0
        else:
            progress_gain = 0.5 if success else 0.0
            verifier_delta = 0.5 if (success and report_match) else 0.0
            regression = 0.0 if success else 0.5
            pass_rate = 1.0 if success else 0.0
        if not repeated_failure_match and success:
            for path in path_matches:
                changed_path_counts[path] = changed_path_counts.get(path, 0.0) + 1.0
        matched.append(
            {
                "progress_gain": progress_gain,
                "verifier_delta": verifier_delta,
                "regression": regression,
                "pass_rate": pass_rate,
            }
        )
    prototype_prediction = semantic_prototype_prediction(summary, normalized)
    if not matched:
        return prototype_prediction
    sample_count = len(matched)
    prediction = {
        "sample_count": sample_count,
        "pass_rate": round(sum(item["pass_rate"] for item in matched) / sample_count, 4),
        "regression_rate": round(sum(item["regression"] for item in matched) / sample_count, 4),
        "predicted_progress_gain": round(sum(item["progress_gain"] for item in matched) / sample_count, 4),
        "predicted_verifier_delta": round(sum(item["verifier_delta"] for item in matched) / sample_count, 4),
        "command_pattern": command_pattern_value,
        "changed_path_counts": changed_path_counts,
    }
    prediction.update(path_prediction_metrics(summary, changed_path_counts=changed_path_counts))
    if not prototype_prediction:
        return prediction
    prototype_weight = float(prototype_prediction.get("sample_count", 0) or 0)
    total_weight = float(sample_count) + prototype_weight
    changed_path_counts = merge_changed_path_counts(
        changed_path_counts,
        dict(prototype_prediction.get("changed_path_counts", {})),
        learned_weight=1.0,
        bootstrap_weight=1.0,
    )
    combined = {
        "sample_count": int(round(total_weight)),
        "pass_rate": round(
            (
                float(prediction.get("pass_rate", 0.0) or 0.0) * float(sample_count)
                + float(prototype_prediction.get("pass_rate", 0.0) or 0.0) * prototype_weight
            )
            / max(1.0, total_weight),
            4,
        ),
        "regression_rate": round(
            (
                float(prediction.get("regression_rate", 0.0) or 0.0) * float(sample_count)
                + float(prototype_prediction.get("regression_rate", 0.0) or 0.0) * prototype_weight
            )
            / max(1.0, total_weight),
            4,
        ),
        "predicted_progress_gain": round(
            (
                float(prediction.get("predicted_progress_gain", 0.0) or 0.0) * float(sample_count)
                + float(prototype_prediction.get("predicted_progress_gain", 0.0) or 0.0) * prototype_weight
            )
            / max(1.0, total_weight),
            4,
        ),
        "predicted_verifier_delta": round(
            (
                float(prediction.get("predicted_verifier_delta", 0.0) or 0.0) * float(sample_count)
                + float(prototype_prediction.get("predicted_verifier_delta", 0.0) or 0.0) * prototype_weight
            )
            / max(1.0, total_weight),
            4,
        ),
        "command_pattern": command_pattern_value,
        "changed_path_counts": changed_path_counts,
        "prototype_prior": dict(prototype_prediction),
    }
    combined.update(path_prediction_metrics(summary, changed_path_counts=changed_path_counts))
    return combined


def semantic_prototype_prediction(summary: dict[str, object], command: str) -> dict[str, object]:
    normalized = str(command).strip()
    if not normalized:
        return {}
    semantic_prototypes = summary.get("semantic_prototypes", [])
    if not isinstance(semantic_prototypes, list) or not semantic_prototypes:
        return {}
    matched = []
    changed_path_counts: dict[str, float] = {}
    for raw in semantic_prototypes:
        if not isinstance(raw, dict):
            continue
        prototype = dict(raw)
        application_commands = [
            str(value).strip()
            for value in prototype.get("application_commands", [])
            if str(value).strip()
        ]
        failed_commands = (
            dict(prototype.get("failed_commands", {}))
            if isinstance(prototype.get("failed_commands", {}), dict)
            else {}
        )
        if normalized in failed_commands:
            matched.append(
                {
                    "weight": float(max(1, int(failed_commands.get(normalized, 0) or 0))),
                    "pass_rate": 0.0,
                    "regression_rate": 1.0,
                    "progress_gain": 0.0,
                    "verifier_delta": -1.0,
                }
            )
            continue
        if normalized not in application_commands and not any(
            str(path).strip() and str(path).strip() in normalized
            for path in prototype.get("changed_paths", [])
        ):
            continue
        weight = float(max(1, int(prototype.get("success_count", 0) or 0)))
        obligation_pressure = min(2, len(list(prototype.get("verifier_obligations", []))))
        for path in prototype.get("changed_paths", []):
            normalized_path = str(path).strip()
            if normalized_path and normalized_path in normalized:
                changed_path_counts[normalized_path] = changed_path_counts.get(normalized_path, 0.0) + weight
        matched.append(
            {
                "weight": weight,
                "pass_rate": 1.0,
                "regression_rate": 0.0,
                "progress_gain": 1.0,
                "verifier_delta": 1.0 + 0.5 * obligation_pressure,
            }
        )
    if not matched:
        return {}
    total_weight = sum(item["weight"] for item in matched)
    prediction = {
        "sample_count": int(round(total_weight)),
        "pass_rate": round(sum(item["pass_rate"] * item["weight"] for item in matched) / total_weight, 4),
        "regression_rate": round(sum(item["regression_rate"] * item["weight"] for item in matched) / total_weight, 4),
        "predicted_progress_gain": round(sum(item["progress_gain"] * item["weight"] for item in matched) / total_weight, 4),
        "predicted_verifier_delta": round(sum(item["verifier_delta"] * item["weight"] for item in matched) / total_weight, 4),
        "command_pattern": command_pattern(normalized),
        "changed_path_counts": changed_path_counts,
    }
    prediction.update(path_prediction_metrics(summary, changed_path_counts=changed_path_counts))
    return prediction


def bootstrap_command_prediction(model, summary: dict[str, object], command: str) -> dict[str, object]:
    normalized = str(command).strip()
    if not normalized:
        return {}
    family = str(summary.get("benchmark_family", "bounded")).strip() or "bounded"
    horizon = str(summary.get("horizon", "bounded")).strip() or "bounded"
    pattern = command_pattern(normalized)
    aggregate = model._bootstrap_step_priors().get((family, horizon, pattern))
    if aggregate is None and horizon != "bounded":
        aggregate = model._bootstrap_step_priors().get((family, "bounded", pattern))
    if aggregate is None:
        aggregate = model._bootstrap_step_priors().get((family, "", pattern))
    if aggregate is None:
        aggregate = model._bootstrap_step_priors().get(("", "", pattern))
    if aggregate is None:
        return {}
    prediction = aggregate_prediction(
        model,
        summary,
        aggregate=aggregate,
        command_pattern_value=pattern,
    )
    prediction["bootstrap_configuration"] = True
    return prediction


def state_conditioned_command_prediction(model, summary: dict[str, object], command: str) -> dict[str, object]:
    normalized = str(command).strip()
    if not normalized:
        return {}
    family = str(summary.get("benchmark_family", "bounded")).strip() or "bounded"
    horizon = str(summary.get("horizon", "bounded")).strip() or "bounded"
    pattern = command_pattern(normalized)
    state_signature = model._summary_state_signature(summary)
    if not state_signature:
        return {}
    aggregate = model._state_conditioned_step_model().get((family, horizon, state_signature, pattern))
    if aggregate is None and horizon != "bounded":
        aggregate = model._state_conditioned_step_model().get((family, "bounded", state_signature, pattern))
    if aggregate is None:
        aggregate = model._state_conditioned_step_model().get((family, "", state_signature, pattern))
    if aggregate is None:
        aggregate = model._state_conditioned_step_model().get(("", "", state_signature, pattern))
    if aggregate is None:
        return {}
    prediction = aggregate_prediction(
        model,
        summary,
        aggregate=aggregate,
        command_pattern_value=pattern,
    )
    prediction["state_signature"] = state_signature
    prediction["test_pass_rate"] = round(
        float(aggregate.get("test_pass_count", 0) or 0)
        / float(max(1, int(aggregate.get("test_command_count", 0) or 0))),
        4,
    )
    prediction["patch_coverage_rate"] = round(
        float(aggregate.get("patch_touch_count", 0) or 0) / float(max(1, int(aggregate.get("count", 0) or 0))),
        4,
    )
    return prediction


def _rolled_summary_after_effect(summary: dict[str, object], effect: dict[str, object]) -> dict[str, object]:
    rolled = dict(summary)
    existing_expected = {
        str(value).strip()
        for value in rolled.get("existing_expected_artifacts", [])
        if str(value).strip()
    }
    missing_expected = [
        str(value).strip()
        for value in rolled.get("missing_expected_artifacts", [])
        if str(value).strip()
    ]
    predicted_outputs = {
        str(value).strip()
        for value in effect.get("predicted_outputs", [])
        if str(value).strip()
    }
    predicted_changed = {
        str(value).strip()
        for value in effect.get("predicted_changed_paths", [])
        if str(value).strip()
    }
    rolled["existing_expected_artifacts"] = sorted(existing_expected | predicted_outputs)
    rolled["missing_expected_artifacts"] = [
        path for path in missing_expected if path not in predicted_outputs
    ]
    for key in ("workflow_expected_changed_paths", "workflow_generated_paths", "workflow_report_paths"):
        rolled[key] = [
            str(value).strip()
            for value in rolled.get(key, [])
            if str(value).strip() and str(value).strip() not in predicted_changed
        ]
    completion_ratio = float(rolled.get("completion_ratio", 0.0) or 0.0)
    predicted_progress_gain = float(effect.get("predicted_progress_gain", 0.0) or 0.0)
    rolled["completion_ratio"] = min(1.0, round(completion_ratio + 0.15 * predicted_progress_gain, 4))
    return rolled


def _semantic_sequence_alignment_bonus(summary: dict[str, object], sequence: list[str]) -> float:
    if not sequence:
        return 0.0
    semantic_prototypes = summary.get("semantic_prototypes", [])
    if not isinstance(semantic_prototypes, list) or not semantic_prototypes:
        return 0.0
    bonus = 0.0
    normalized_sequence = [str(command).strip() for command in sequence if str(command).strip()]
    for raw in semantic_prototypes:
        if not isinstance(raw, dict):
            continue
        sequences = (
            dict(raw.get("command_sequences", {}))
            if isinstance(raw.get("command_sequences", {}), dict)
            else {}
        )
        for serialized, count in sequences.items():
            commands = [part.strip() for part in str(serialized).split(" || ") if part.strip()]
            if commands and normalized_sequence == commands[: len(normalized_sequence)]:
                bonus = max(bonus, 0.75 * float(count or 0))
    return bonus


def bootstrap_step_priors(model) -> dict[tuple[str, str, str], dict[str, object]]:
    if model._bootstrap_step_priors_cache is not None:
        return model._bootstrap_step_priors_cache
    aggregates: dict[tuple[str, str, str], dict[str, object]] = {}
    try:
        documents = iter_episode_documents(model.config.trajectories_root, config=model.config)
    except Exception:
        model._bootstrap_step_priors_cache = {}
        return model._bootstrap_step_priors_cache
    for document in documents[-256:]:
        if not isinstance(document, dict):
            continue
        task_metadata = document.get("task_metadata", {})
        task_metadata = dict(task_metadata) if isinstance(task_metadata, dict) else {}
        family = str(task_metadata.get("benchmark_family", "bounded")).strip() or "bounded"
        summary = document.get("summary", {})
        summary = dict(summary) if isinstance(summary, dict) else {}
        horizon = str(summary.get("horizon", task_metadata.get("difficulty", ""))).strip()
        if horizon != "long_horizon":
            horizon = "bounded"
        for step in document.get("steps", []) if isinstance(document.get("steps", []), list) else []:
            if not isinstance(step, dict):
                continue
            if str(step.get("action", "")).strip() != "code_execute":
                continue
            command = str(step.get("content", "")).strip()
            if not command:
                continue
            pattern = command_pattern(command)
            keys = {
                (family, horizon, pattern),
                (family, "", pattern),
                ("", "", pattern),
            }
            verification = step.get("verification", {})
            verification = dict(verification) if isinstance(verification, dict) else {}
            passed = bool(verification.get("passed", False))
            transition = step.get("state_transition", {})
            transition = dict(transition) if isinstance(transition, dict) else {}
            progress_gain = max(
                float(step.get("state_progress_delta", 0.0) or 0.0),
                float(transition.get("progress_delta", 0.0) or 0.0),
            )
            verifier_delta = transition_verifier_delta(transition)
            changed_paths = transition_changed_paths(transition)
            regressed = bool(changed_paths and list(transition.get("regressions", [])))
            for key in keys:
                aggregate = aggregates.setdefault(
                    key,
                    {
                        "count": 0,
                        "pass_count": 0,
                        "regression_count": 0,
                        "progress_gain_sum": 0.0,
                        "verifier_delta_sum": 0.0,
                        "changed_path_counts": {},
                    },
                )
                aggregate["count"] = int(aggregate.get("count", 0) or 0) + 1
                if passed:
                    aggregate["pass_count"] = int(aggregate.get("pass_count", 0) or 0) + 1
                if regressed:
                    aggregate["regression_count"] = int(aggregate.get("regression_count", 0) or 0) + 1
                aggregate["progress_gain_sum"] = float(aggregate.get("progress_gain_sum", 0.0) or 0.0) + progress_gain
                aggregate["verifier_delta_sum"] = float(aggregate.get("verifier_delta_sum", 0.0) or 0.0) + verifier_delta
                path_counts = aggregate.setdefault("changed_path_counts", {})
                if isinstance(path_counts, dict):
                    for path in changed_paths:
                        path_counts[path] = int(path_counts.get(path, 0) or 0) + 1
    model._bootstrap_step_priors_cache = aggregates
    return model._bootstrap_step_priors_cache


def state_conditioned_step_model(model) -> dict[tuple[str, str, str, str], dict[str, object]]:
    if model._state_conditioned_step_model_cache is not None:
        return model._state_conditioned_step_model_cache
    aggregates: dict[tuple[str, str, str, str], dict[str, object]] = {}
    try:
        documents = iter_episode_documents(model.config.trajectories_root, config=model.config)
    except Exception:
        model._state_conditioned_step_model_cache = {}
        return model._state_conditioned_step_model_cache
    for document in documents[-256:]:
        if not isinstance(document, dict):
            continue
        task_metadata = document.get("task_metadata", {})
        task_metadata = dict(task_metadata) if isinstance(task_metadata, dict) else {}
        family = str(task_metadata.get("benchmark_family", "bounded")).strip() or "bounded"
        summary = document.get("summary", {})
        summary = dict(summary) if isinstance(summary, dict) else {}
        horizon = str(summary.get("horizon", task_metadata.get("difficulty", ""))).strip()
        if horizon != "long_horizon":
            horizon = "bounded"
        state_signature = model._document_state_signature(document, family=family, horizon=horizon)
        if not state_signature:
            continue
        for step in document.get("steps", []) if isinstance(document.get("steps", []), list) else []:
            if not isinstance(step, dict):
                continue
            if str(step.get("action", "")).strip() != "code_execute":
                continue
            command = str(step.get("content", "")).strip()
            if not command:
                continue
            pattern = command_pattern(command)
            keys = {
                (family, horizon, state_signature, pattern),
                (family, "", state_signature, pattern),
                ("", "", state_signature, pattern),
            }
            verification = step.get("verification", {})
            verification = dict(verification) if isinstance(verification, dict) else {}
            passed = bool(verification.get("passed", False))
            transition = step.get("state_transition", {})
            transition = dict(transition) if isinstance(transition, dict) else {}
            progress_gain = max(
                float(step.get("state_progress_delta", 0.0) or 0.0),
                float(transition.get("progress_delta", 0.0) or 0.0),
            )
            verifier_delta = transition_verifier_delta(transition)
            changed_paths = transition_changed_paths(transition)
            regressed = bool(changed_paths and list(transition.get("regressions", [])))
            patch_touch_count = len(
                [
                    patch
                    for patch in transition.get("edit_patches", [])
                    if isinstance(patch, dict) and str(patch.get("path", "")).strip()
                ]
            )
            is_test_command = model._is_test_like_command(command)
            for key in keys:
                aggregate = aggregates.setdefault(
                    key,
                    {
                        "count": 0,
                        "pass_count": 0,
                        "regression_count": 0,
                        "progress_gain_sum": 0.0,
                        "verifier_delta_sum": 0.0,
                        "changed_path_counts": {},
                        "test_command_count": 0,
                        "test_pass_count": 0,
                        "patch_touch_count": 0,
                    },
                )
                aggregate["count"] = int(aggregate.get("count", 0) or 0) + 1
                if passed:
                    aggregate["pass_count"] = int(aggregate.get("pass_count", 0) or 0) + 1
                if regressed:
                    aggregate["regression_count"] = int(aggregate.get("regression_count", 0) or 0) + 1
                if is_test_command:
                    aggregate["test_command_count"] = int(aggregate.get("test_command_count", 0) or 0) + 1
                    if passed:
                        aggregate["test_pass_count"] = int(aggregate.get("test_pass_count", 0) or 0) + 1
                aggregate["patch_touch_count"] = int(aggregate.get("patch_touch_count", 0) or 0) + patch_touch_count
                aggregate["progress_gain_sum"] = float(aggregate.get("progress_gain_sum", 0.0) or 0.0) + progress_gain
                aggregate["verifier_delta_sum"] = float(aggregate.get("verifier_delta_sum", 0.0) or 0.0) + verifier_delta
                path_counts = aggregate.setdefault("changed_path_counts", {})
                if isinstance(path_counts, dict):
                    for path in changed_paths:
                        path_counts[path] = int(path_counts.get(path, 0) or 0) + 1
    model._state_conditioned_step_model_cache = aggregates
    return model._state_conditioned_step_model_cache


def aggregate_prediction(
    model,
    summary: dict[str, object],
    *,
    aggregate: dict[str, object],
    command_pattern_value: str,
) -> dict[str, object]:
    sample_count = max(1, int(aggregate.get("count", 0) or 0))
    changed_path_counts = dict(aggregate.get("changed_path_counts", {}))
    prediction = {
        "sample_count": sample_count,
        "pass_rate": round(float(aggregate.get("pass_count", 0) or 0) / float(sample_count), 4),
        "regression_rate": round(float(aggregate.get("regression_count", 0) or 0) / float(sample_count), 4),
        "predicted_progress_gain": round(float(aggregate.get("progress_gain_sum", 0.0) or 0.0) / float(sample_count), 4),
        "predicted_verifier_delta": round(float(aggregate.get("verifier_delta_sum", 0.0) or 0.0) / float(sample_count), 4),
        "command_pattern": command_pattern_value,
        "changed_path_counts": changed_path_counts,
    }
    prediction.update(path_prediction_metrics(summary, changed_path_counts=changed_path_counts))
    return prediction


def merge_changed_path_counts(
    learned: dict[str, object],
    bootstrap: dict[str, object],
    *,
    learned_weight: float,
    bootstrap_weight: float,
) -> dict[str, float]:
    merged: dict[str, float] = {}
    for path, count in learned.items():
        normalized = str(path).strip()
        if normalized:
            merged[normalized] = merged.get(normalized, 0.0) + float(count or 0.0) * max(learned_weight, 1.0)
    for path, count in bootstrap.items():
        normalized = str(path).strip()
        if normalized:
            merged[normalized] = merged.get(normalized, 0.0) + float(count or 0.0) * max(bootstrap_weight, 0.0)
    return merged


def path_prediction_metrics(
    summary: dict[str, object],
    *,
    changed_path_counts: dict[str, object],
) -> dict[str, object]:
    unresolved_paths = {
        str(path).strip()
        for key in (
            "missing_expected_artifacts",
            "unsatisfied_expected_contents",
            "present_forbidden_artifacts",
            "changed_preserved_artifacts",
            "workflow_expected_changed_paths",
            "workflow_generated_paths",
            "workflow_report_paths",
        )
        for path in summary.get(key, [])
        if str(path).strip()
    }
    path_overlap_score = float(
        sum(
            float(changed_path_counts.get(path, 0) or 0.0)
            for path in unresolved_paths
            if path in changed_path_counts
        )
    )
    predicted_changed_paths = [
        str(path)
        for path, _count in sorted(
            (
                (str(path).strip(), float(count or 0.0))
                for path, count in changed_path_counts.items()
                if str(path).strip()
            ),
            key=lambda item: (-item[1], item[0]),
        )[:4]
    ]
    return {
        "path_overlap_score": round(path_overlap_score, 4),
        "predicted_changed_paths": predicted_changed_paths,
    }


def transition_changed_paths(transition: dict[str, object]) -> list[str]:
    changed_paths: list[str] = []
    if isinstance(transition.get("changed_paths", []), list):
        for item in transition.get("changed_paths", []):
            normalized = str(item).strip()
            if normalized and normalized not in changed_paths:
                changed_paths.append(normalized)
        if changed_paths:
            return changed_paths
    for key in (
        "newly_materialized_expected_artifacts",
        "newly_satisfied_expected_contents",
        "cleared_forbidden_artifacts",
        "newly_changed_preserved_artifacts",
        "newly_updated_workflow_paths",
        "newly_updated_generated_paths",
        "newly_updated_report_paths",
        "regressions",
    ):
        for item in transition.get(key, []):
            normalized = str(item).strip()
            if normalized and normalized not in changed_paths:
                changed_paths.append(normalized)
    for patch in transition.get("edit_patches", []):
        if not isinstance(patch, dict):
            continue
        normalized = str(patch.get("path", "")).strip()
        if normalized and normalized not in changed_paths:
            changed_paths.append(normalized)
    return changed_paths


def transition_verifier_delta(transition: dict[str, object]) -> float:
    try:
        return float(transition.get("verifier_delta"))
    except (TypeError, ValueError):
        pass
    positive = 0
    for key in (
        "newly_materialized_expected_artifacts",
        "newly_satisfied_expected_contents",
        "cleared_forbidden_artifacts",
        "newly_updated_workflow_paths",
        "newly_updated_generated_paths",
        "newly_updated_report_paths",
    ):
        positive += len([item for item in transition.get(key, []) if str(item).strip()])
    negative = len([item for item in transition.get("regressions", []) if str(item).strip()])
    return float(positive - negative)


def command_pattern(command: str) -> str:
    normalized = str(command).strip()
    if not normalized:
        return ""
    if normalized.startswith("python -m pytest"):
        return "python -m pytest"
    if normalized.startswith("pytest"):
        return "pytest"
    if normalized.startswith("git "):
        parts = normalized.split()
        if len(parts) >= 2:
            return f"git {parts[1]}"
        return "git"
    head = normalized.split("&&", 1)[0].strip()
    token = head.split(None, 1)[0] if head else normalized
    return token
