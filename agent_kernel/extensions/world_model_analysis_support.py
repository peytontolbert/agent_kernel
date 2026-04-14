from __future__ import annotations

from ..schemas import TaskSpec


def prioritized_long_horizon_hotspots(
    model,
    *,
    task: TaskSpec,
    summary: dict[str, object],
    latest_transition: dict[str, object] | None = None,
    latent_state_summary: dict[str, object] | None = None,
    active_subgoal: str = "",
    max_items: int = 6,
) -> list[dict[str, object]]:
    horizon = str(
        summary.get("horizon", task.metadata.get("difficulty", task.metadata.get("horizon", "")))
    ).strip()
    if horizon != "long_horizon":
        return []
    latent = latent_state_summary if isinstance(latent_state_summary, dict) else {}
    learned = latent.get("learned_world_state", {})
    learned = learned if isinstance(learned, dict) else {}
    learned_progress_signal = max(
        model._safe_float(learned.get("progress_signal"), 0.0),
        model._safe_float(learned.get("world_progress_score"), 0.0),
        model._safe_float(learned.get("decoder_world_progress_score"), 0.0),
        model._safe_float(learned.get("transition_progress_score"), 0.0),
    )
    learned_risk_signal = max(
        model._safe_float(learned.get("risk_signal"), 0.0),
        model._safe_float(learned.get("world_risk_score"), 0.0),
        model._safe_float(learned.get("decoder_world_risk_score"), 0.0),
        model._safe_float(learned.get("transition_regression_score"), 0.0),
    )
    transition = latest_transition if isinstance(latest_transition, dict) else {}
    active_paths = {
        str(path).strip()
        for path in latent.get("active_paths", [])
        if str(path).strip()
    }
    regressed_paths = {
        str(path).strip()
        for path in transition.get("regressions", [])
        if str(path).strip()
    }
    no_progress = bool(transition.get("no_progress", False))
    unresolved_count = sum(
        len(list(summary.get(key, [])))
        for key in (
            "missing_expected_artifacts",
            "unsatisfied_expected_contents",
            "present_forbidden_artifacts",
            "changed_preserved_artifacts",
            "missing_preserved_artifacts",
        )
    )
    workflow_expected_paths = {
        str(path).strip()
        for path in summary.get("workflow_expected_changed_paths", [])
        if str(path).strip()
    }
    updated_workflow_paths = {
        str(path).strip() for path in summary.get("updated_workflow_paths", []) if str(path).strip()
    }
    workflow_report_paths = {
        str(path).strip() for path in summary.get("workflow_report_paths", []) if str(path).strip()
    }
    updated_report_paths = {
        str(path).strip() for path in summary.get("updated_report_paths", []) if str(path).strip()
    }
    workflow_generated_paths = {
        str(path).strip() for path in summary.get("workflow_generated_paths", []) if str(path).strip()
    }
    updated_generated_paths = {
        str(path).strip() for path in summary.get("updated_generated_paths", []) if str(path).strip()
    }
    should_surface = (
        learned_risk_signal >= 0.35
        and (
            learned_risk_signal > learned_progress_signal
            or no_progress
            or bool(regressed_paths)
            or unresolved_count >= 2
            or bool(active_paths)
            or bool(workflow_expected_paths - updated_workflow_paths)
            or bool(workflow_report_paths - updated_report_paths)
            or bool(workflow_generated_paths - updated_generated_paths)
        )
    )
    if not should_surface:
        return []

    active_path = model._active_subgoal_path(active_subgoal)
    hotspots: list[dict[str, object]] = []
    hotspot_index = 0

    def append_hotspot(
        *,
        path: str,
        subgoal: str,
        category: str,
        base_priority: int,
        signals: list[str],
    ) -> None:
        nonlocal hotspot_index
        normalized_path = str(path).strip()
        normalized_subgoal = str(subgoal).strip()
        if not normalized_path or not normalized_subgoal:
            return
        priority = base_priority
        ordered_signals: list[str] = []
        for signal in signals:
            normalized_signal = str(signal).strip()
            if normalized_signal and normalized_signal not in ordered_signals:
                ordered_signals.append(normalized_signal)
        if normalized_path in active_paths:
            priority += 12
            ordered_signals.append("active_path")
        if normalized_path in regressed_paths:
            priority += 18
            ordered_signals.append("state_regression")
        if no_progress:
            priority += 6
            ordered_signals.append("no_state_progress")
        if normalized_path == active_path:
            priority += 5
            ordered_signals.append("active_subgoal")
        if learned_risk_signal > learned_progress_signal:
            priority += 4
            ordered_signals.append("learned_risk")
        hotspots.append(
            {
                "path": normalized_path,
                "subgoal": normalized_subgoal,
                "category": category,
                "priority": priority,
                "signals": ordered_signals,
                "hotspot_index": hotspot_index,
            }
        )
        hotspot_index += 1

    for path in summary.get("changed_preserved_artifacts", []):
        append_hotspot(
            path=str(path),
            subgoal=f"preserve required artifact {path}",
            category="changed_preserved",
            base_priority=96,
            signals=["preserved_artifact_changed"],
        )
    for path in summary.get("missing_preserved_artifacts", []):
        append_hotspot(
            path=str(path),
            subgoal=f"preserve required artifact {path}",
            category="missing_preserved",
            base_priority=92,
            signals=["preserved_artifact_missing"],
        )
    for path in summary.get("present_forbidden_artifacts", []):
        append_hotspot(
            path=str(path),
            subgoal=f"remove forbidden artifact {path}",
            category="present_forbidden",
            base_priority=88,
            signals=["forbidden_artifact_present"],
        )
    for path in summary.get("unsatisfied_expected_contents", []):
        append_hotspot(
            path=str(path),
            subgoal=f"materialize expected artifact {path}",
            category="unsatisfied_expected_content",
            base_priority=84,
            signals=["expected_content_unsatisfied"],
        )
    for path in summary.get("missing_expected_artifacts", []):
        append_hotspot(
            path=str(path),
            subgoal=f"materialize expected artifact {path}",
            category="missing_expected_artifact",
            base_priority=78,
            signals=["expected_artifact_missing"],
        )
    for path in sorted(workflow_expected_paths - updated_workflow_paths):
        append_hotspot(
            path=path,
            subgoal=f"update workflow path {path}",
            category="workflow_path_pending",
            base_priority=74,
            signals=["workflow_path_pending"],
        )
    for path in sorted(workflow_generated_paths - updated_generated_paths):
        append_hotspot(
            path=path,
            subgoal=f"regenerate generated artifact {path}",
            category="generated_artifact_pending",
            base_priority=72,
            signals=["generated_artifact_pending"],
        )
    for path in sorted(workflow_report_paths - updated_report_paths):
        append_hotspot(
            path=path,
            subgoal=f"write workflow report {path}",
            category="workflow_report_pending",
            base_priority=68,
            signals=["workflow_report_pending"],
        )

    deduped: list[dict[str, object]] = []
    seen: set[str] = set()
    for entry in sorted(
        hotspots,
        key=lambda entry: (
            -int(entry.get("priority", 0) or 0),
            int(entry.get("hotspot_index", 0) or 0),
            str(entry.get("subgoal", "")),
        ),
    ):
        subgoal = str(entry.get("subgoal", "")).strip()
        if not subgoal or subgoal in seen:
            continue
        seen.add(subgoal)
        deduped.append(entry)
        if len(deduped) >= max(1, int(max_items or 0)):
            break
    return deduped


def bridged_targeted_preview_window_runs(
    *,
    bridged_windows: list[dict[str, object]],
) -> list[dict[str, object]]:
    runs: list[dict[str, object]] = []
    active_segments: list[dict[str, object]] = []
    active_window_indices: list[int] = []

    def flush_active_run() -> None:
        nonlocal active_segments, active_window_indices
        if not active_segments or len(active_window_indices) < 2:
            active_segments = []
            active_window_indices = []
            return
        first_segment = active_segments[0]
        last_segment = active_segments[-1]
        runs.append(
            {
                "truncated": True,
                "bridge_window_indices": list(active_window_indices),
                "line_start": int(first_segment["line_start"]),
                "line_end": int(last_segment["line_end"]),
                "target_line_start": int(first_segment["target_line_start"]),
                "target_line_end": int(last_segment["target_line_end"]),
                "hidden_gap_current_line_count": sum(
                    int(segment["hidden_gap_current_line_count"])
                    for segment in active_segments
                ),
                "hidden_gap_target_line_count": sum(
                    int(segment["hidden_gap_target_line_count"])
                    for segment in active_segments
                ),
                "line_delta": sum(int(segment["line_delta"]) for segment in active_segments),
                "explicit_hidden_gap_current_proof": all(
                    bool(segment["explicit_hidden_gap_current_proof"])
                    for segment in active_segments
                ),
                "hidden_gap_current_from_line_span_proof": any(
                    bool(segment.get("hidden_gap_current_from_line_span_proof", False))
                    for segment in active_segments
                ),
                "hidden_gap_target_from_expected_content": all(
                    bool(segment.get("hidden_gap_target_from_expected_content", False))
                    for segment in active_segments
                ),
                "bridge_segments": [dict(segment) for segment in active_segments],
            }
        )
        active_segments = []
        active_window_indices = []

    for bridge in bridged_windows:
        bridge_window_indices = sorted(
            {
                int(index)
                for index in bridge.get("bridge_window_indices", [])
                if isinstance(index, int)
            }
        )
        if len(bridge_window_indices) != 2 or bridge_window_indices[1] != bridge_window_indices[0] + 1:
            flush_active_run()
            continue
        segment = {
            "bridge_window_indices": bridge_window_indices,
            "left_window_index": bridge_window_indices[0],
            "right_window_index": bridge_window_indices[1],
            "line_start": int(bridge.get("line_start", 1)),
            "line_end": int(bridge.get("line_end", bridge.get("line_start", 1) - 1)),
            "target_line_start": int(bridge.get("target_line_start", 1)),
            "target_line_end": int(
                bridge.get("target_line_end", bridge.get("target_line_start", 1) - 1)
            ),
            "hidden_gap_current_line_start": int(bridge.get("hidden_gap_current_line_start", 1)),
            "hidden_gap_current_line_end": int(
                bridge.get(
                    "hidden_gap_current_line_end",
                    bridge.get("hidden_gap_current_line_start", 1) - 1,
                )
            ),
            "hidden_gap_target_line_start": int(bridge.get("hidden_gap_target_line_start", 1)),
            "hidden_gap_target_line_end": int(
                bridge.get(
                    "hidden_gap_target_line_end",
                    bridge.get("hidden_gap_target_line_start", 1) - 1,
                )
            ),
            "hidden_gap_current_content": str(bridge.get("hidden_gap_current_content", "")),
            "hidden_gap_target_content": str(bridge.get("hidden_gap_target_content", "")),
            "hidden_gap_current_from_line_span_proof": bool(
                bridge.get("hidden_gap_current_from_line_span_proof", False)
            ),
            "hidden_gap_target_from_expected_content": bool(
                bridge.get("hidden_gap_target_from_expected_content", False)
            ),
            "hidden_gap_current_line_count": int(bridge.get("hidden_gap_current_line_count", 0)),
            "hidden_gap_target_line_count": int(bridge.get("hidden_gap_target_line_count", 0)),
            "line_delta": int(bridge.get("line_delta", 0)),
            "explicit_hidden_gap_current_proof": bool(
                bridge.get("explicit_hidden_gap_current_proof", False)
            ),
        }
        if not active_segments:
            active_segments = [segment]
            active_window_indices = list(bridge_window_indices)
            continue
        if bridge_window_indices[0] != active_window_indices[-1]:
            flush_active_run()
            active_segments = [segment]
            active_window_indices = list(bridge_window_indices)
            continue
        active_segments.append(segment)
        active_window_indices.append(bridge_window_indices[1])
    flush_active_run()
    return runs


def hidden_gap_current_proof_regions(
    *,
    bridged_runs: list[dict[str, object]],
) -> list[dict[str, object]]:
    regions: list[dict[str, object]] = []
    for region_index, run in enumerate(bridged_runs):
        window_indices = [
            int(index)
            for index in run.get("bridge_window_indices", [])
            if isinstance(index, int)
        ]
        if len(window_indices) < 3:
            continue
        raw_segments = run.get("bridge_segments", [])
        if not isinstance(raw_segments, list):
            continue
        proof_spans: list[dict[str, object]] = []
        opaque_spans: list[dict[str, object]] = []
        current_proof_covered_line_count = 0
        current_proof_missing_line_count = 0
        current_proof_missing_span_count = 0
        for segment in raw_segments:
            if not isinstance(segment, dict):
                proof_spans = []
                break
            current_line_start = int(segment.get("hidden_gap_current_line_start", 1))
            current_line_end = int(
                segment.get(
                    "hidden_gap_current_line_end",
                    segment.get("hidden_gap_current_line_start", 1) - 1,
                )
            )
            current_line_count = max(0, current_line_end - current_line_start + 1)
            current_content = str(segment.get("hidden_gap_current_content", ""))
            current_from_line_span_proof = bool(
                segment.get("hidden_gap_current_from_line_span_proof", False)
            )
            current_content_complete = (
                current_line_count <= 0
                or len(current_content.splitlines()) == current_line_count
            )
            if current_from_line_span_proof or current_content_complete:
                current_proof_covered_line_count += current_line_count
            elif current_line_count > 0:
                current_proof_missing_line_count += current_line_count
                current_proof_missing_span_count += 1
                opaque_spans.append(
                    {
                        "current_line_start": current_line_start,
                        "current_line_end": current_line_end,
                        "target_line_start": int(
                            segment.get("hidden_gap_target_line_start", 1)
                        ),
                        "target_line_end": int(
                            segment.get(
                                "hidden_gap_target_line_end",
                                segment.get("hidden_gap_target_line_start", 1) - 1,
                            )
                        ),
                        "reason": "missing_current_proof",
                    }
                )
            proof_spans.append(
                {
                    "current_line_start": current_line_start,
                    "current_line_end": current_line_end,
                    "target_line_start": int(segment.get("hidden_gap_target_line_start", 1)),
                    "target_line_end": int(
                        segment.get(
                            "hidden_gap_target_line_end",
                            segment.get("hidden_gap_target_line_start", 1) - 1,
                        )
                    ),
                    "current_content": current_content,
                    "target_content": str(segment.get("hidden_gap_target_content", "")),
                    "current_from_line_span_proof": current_from_line_span_proof,
                    "target_from_expected_content": bool(
                        segment.get("hidden_gap_target_from_expected_content", False)
                    ),
                }
            )
        if len(proof_spans) < 2:
            continue
        regions.append(
            {
                "proof_region_index": region_index,
                "window_indices": window_indices,
                "line_start": int(run.get("line_start", 1)),
                "line_end": int(run.get("line_end", run.get("line_start", 1) - 1)),
                "target_line_start": int(run.get("target_line_start", 1)),
                "target_line_end": int(
                    run.get("target_line_end", run.get("target_line_start", 1) - 1)
                ),
                "current_proof_span_count": len(proof_spans),
                "current_proof_spans": proof_spans,
                "current_proof_opaque_spans": opaque_spans,
                "current_proof_opaque_span_count": len(opaque_spans),
                "current_proof_complete": current_proof_missing_line_count <= 0,
                "current_proof_partial_coverage": (
                    current_proof_covered_line_count > 0 and current_proof_missing_line_count > 0
                ),
                "current_proof_covered_line_count": current_proof_covered_line_count,
                "current_proof_missing_line_count": current_proof_missing_line_count,
                "current_proof_missing_span_count": current_proof_missing_span_count,
                "truncated": bool(run.get("truncated", True)),
                "explicit_hidden_gap_current_proof": bool(
                    run.get("explicit_hidden_gap_current_proof", False)
                ),
                "hidden_gap_current_from_line_span_proof": bool(
                    run.get("hidden_gap_current_from_line_span_proof", False)
                ),
                "hidden_gap_target_from_expected_content": bool(
                    run.get("hidden_gap_target_from_expected_content", False)
                ),
            }
        )
    return regions


def sparse_hidden_gap_current_proof_regions(
    model,
    *,
    windows: list[tuple[int, int, int, int]],
    bridged_windows: list[dict[str, object]],
    start_region_index: int = 0,
) -> list[dict[str, object]]:
    if len(windows) < 4:
        return []
    bridge_by_pair: dict[tuple[int, int], dict[str, object]] = {}
    for bridge in bridged_windows:
        if not isinstance(bridge, dict):
            continue
        bridge_window_indices = [
            int(index)
            for index in bridge.get("bridge_window_indices", [])
            if isinstance(index, int)
        ]
        if len(bridge_window_indices) != 2:
            continue
        bridge_by_pair[(bridge_window_indices[0], bridge_window_indices[1])] = bridge
    proof_pair_indices = sorted(
        {
            left
            for (left, right), bridge in bridge_by_pair.items()
            if right == left + 1
            and bridge_window_has_current_proof(bridge)
        }
    )
    if len(proof_pair_indices) < 2:
        return []
    regions: list[dict[str, object]] = []
    region_index = start_region_index
    for start_position, start_pair_index in enumerate(proof_pair_indices[:-1]):
        for end_pair_index in proof_pair_indices[start_position + 1 :]:
            if end_pair_index <= start_pair_index:
                continue
            window_start_index = start_pair_index
            window_end_index = end_pair_index + 1
            if window_end_index - window_start_index + 1 < 4:
                continue
            region = sparse_hidden_gap_current_proof_region(
                model,
                windows=windows,
                bridge_by_pair=bridge_by_pair,
                window_start_index=window_start_index,
                window_end_index=window_end_index,
                region_index=region_index,
            )
            if region is None:
                continue
            regions.append(region)
            region_index += 1
    return regions


def bridge_window_has_current_proof(bridge: dict[str, object]) -> bool:
    current_line_start = int(bridge.get("hidden_gap_current_line_start", 1))
    current_line_end = int(
        bridge.get("hidden_gap_current_line_end", current_line_start - 1)
    )
    current_line_count = max(0, current_line_end - current_line_start + 1)
    if current_line_count <= 0:
        return False
    if bool(bridge.get("hidden_gap_current_from_line_span_proof", False)):
        return True
    current_content = str(bridge.get("hidden_gap_current_content", ""))
    return len(current_content.splitlines()) == current_line_count


def sparse_hidden_gap_current_proof_region(
    model,
    *,
    windows: list[tuple[int, int, int, int]],
    bridge_by_pair: dict[tuple[int, int], dict[str, object]],
    window_start_index: int,
    window_end_index: int,
    region_index: int,
) -> dict[str, object] | None:
    if window_start_index < 0 or window_end_index >= len(windows):
        return None
    while window_start_index > 0 and windows_have_zero_gap(
        windows[window_start_index - 1],
        windows[window_start_index],
    ):
        window_start_index -= 1
    while window_end_index < len(windows) - 1 and windows_have_zero_gap(
        windows[window_end_index],
        windows[window_end_index + 1],
    ):
        window_end_index += 1
    if window_end_index - window_start_index + 1 < 4:
        return None
    proof_spans: list[dict[str, object]] = []
    opaque_spans: list[dict[str, object]] = []
    proof_pair_count = 0
    current_proof_covered_line_count = 0
    current_proof_missing_line_count = 0
    current_proof_missing_span_count = 0
    hidden_gap_target_from_expected_content = True
    explicit_hidden_gap_current_proof = True
    hidden_gap_current_from_line_span_proof = False
    saw_nonbridge_join = False
    for left_index in range(window_start_index, window_end_index):
        right_index = left_index + 1
        left_start, left_end, left_target_start, left_target_end = windows[left_index]
        right_start, right_end, right_target_start, right_target_end = windows[right_index]
        current_line_start = left_end + 1
        current_line_end = right_start
        target_line_start = left_target_end + 1
        target_line_end = right_target_start
        current_line_count = max(0, current_line_end - current_line_start + 1)
        target_line_count = max(0, target_line_end - target_line_start + 1)
        bridge = bridge_by_pair.get((left_index, right_index))
        if bridge is None:
            if current_line_count > 0 or target_line_count > 0:
                current_proof_missing_line_count += current_line_count
                if current_line_count > 0:
                    current_proof_missing_span_count += 1
                    opaque_spans.append(
                        {
                            "current_line_start": current_line_start,
                            "current_line_end": current_line_end,
                            "target_line_start": target_line_start,
                            "target_line_end": target_line_end,
                            "reason": "no_adjacent_pair_bridge",
                        }
                    )
            else:
                saw_nonbridge_join = True
            continue
        if current_line_count <= 0 and target_line_count <= 0:
            saw_nonbridge_join = True
            continue
        proof_pair_count += 1
        current_from_line_span_proof = bool(
            bridge.get("hidden_gap_current_from_line_span_proof", False)
        )
        target_from_expected_content = bool(
            bridge.get("hidden_gap_target_from_expected_content", False)
        )
        current_content = str(bridge.get("hidden_gap_current_content", ""))
        target_content = str(bridge.get("hidden_gap_target_content", ""))
        current_content_complete = (
            current_line_count <= 0 or len(current_content.splitlines()) == current_line_count
        )
        if current_from_line_span_proof or current_content_complete:
            current_proof_covered_line_count += current_line_count
        elif current_line_count > 0:
            current_proof_missing_line_count += current_line_count
            current_proof_missing_span_count += 1
            opaque_spans.append(
                {
                    "current_line_start": current_line_start,
                    "current_line_end": current_line_end,
                    "target_line_start": target_line_start,
                    "target_line_end": target_line_end,
                    "reason": "missing_current_proof",
                }
            )
        hidden_gap_current_from_line_span_proof = (
            hidden_gap_current_from_line_span_proof or current_from_line_span_proof
        )
        hidden_gap_target_from_expected_content = (
            hidden_gap_target_from_expected_content and target_from_expected_content
        )
        explicit_hidden_gap_current_proof = (
            explicit_hidden_gap_current_proof
            and bool(bridge.get("explicit_hidden_gap_current_proof", False))
        )
        proof_spans.append(
            {
                "current_line_start": current_line_start,
                "current_line_end": current_line_end,
                "target_line_start": target_line_start,
                "target_line_end": target_line_end,
                "current_content": current_content,
                "target_content": target_content,
                "current_from_line_span_proof": current_from_line_span_proof,
                "target_from_expected_content": target_from_expected_content,
            }
        )
    if proof_pair_count < 2:
        return None
    if not saw_nonbridge_join and current_proof_missing_line_count <= 0:
        return None
    region_line_start = windows[window_start_index][0] + 1
    region_line_end = windows[window_end_index][1]
    region_target_line_start = windows[window_start_index][2] + 1
    region_target_line_end = windows[window_end_index][3]
    return {
        "proof_region_index": region_index,
        "window_indices": list(range(window_start_index, window_end_index + 1)),
        "line_start": region_line_start,
        "line_end": region_line_end,
        "target_line_start": region_target_line_start,
        "target_line_end": region_target_line_end,
        "current_proof_span_count": len(proof_spans),
        "current_proof_spans": proof_spans,
        "current_proof_opaque_spans": opaque_spans,
        "current_proof_opaque_span_count": len(opaque_spans),
        "current_proof_complete": current_proof_missing_line_count <= 0,
        "current_proof_partial_coverage": (
            current_proof_covered_line_count > 0 and current_proof_missing_line_count > 0
        ),
        "current_proof_covered_line_count": current_proof_covered_line_count,
        "current_proof_missing_line_count": current_proof_missing_line_count,
        "current_proof_missing_span_count": current_proof_missing_span_count,
        "truncated": True,
        "explicit_hidden_gap_current_proof": explicit_hidden_gap_current_proof,
        "hidden_gap_current_from_line_span_proof": hidden_gap_current_from_line_span_proof,
        "hidden_gap_target_from_expected_content": hidden_gap_target_from_expected_content,
    }


def windows_have_zero_gap(
    left_window: tuple[int, int, int, int],
    right_window: tuple[int, int, int, int],
) -> bool:
    left_start, left_end, left_target_start, left_target_end = left_window
    right_start, right_end, right_target_start, right_target_end = right_window
    return (
        max(0, right_start - left_end) <= 0
        and max(0, right_target_start - left_target_end) <= 0
    )
