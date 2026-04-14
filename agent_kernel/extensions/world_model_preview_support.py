from __future__ import annotations

import difflib
import hashlib
from pathlib import Path


def workspace_state_summary(
    model,
    task,
    workspace: Path,
    *,
    workflow: dict[str, object],
    workspace_snapshot: dict[str, str],
) -> dict[str, object]:
    expected = sorted(set(task.expected_files) | set(task.expected_file_contents))
    existing_expected = [path for path in expected if (workspace / path).exists()]
    missing_expected = [path for path in expected if path not in existing_expected]
    satisfied_expected_contents = [
        path
        for path, content in task.expected_file_contents.items()
        if model._file_contents_match(workspace / path, content)
    ]
    unsatisfied_expected_contents = [
        path for path in task.expected_file_contents if path not in satisfied_expected_contents
    ]
    present_forbidden = [path for path in task.forbidden_files if (workspace / path).exists()]
    intact_preserved: list[str] = []
    changed_preserved: list[str] = []
    missing_preserved: list[str] = []
    for path in workflow.get("preserved_paths", []):
        current = model._fingerprint_path(workspace / path)
        baseline = workspace_snapshot.get(path, "")
        if not current:
            missing_preserved.append(path)
        elif baseline and current != baseline:
            changed_preserved.append(path)
        else:
            intact_preserved.append(path)

    updated_workflow_paths = [
        path
        for path in workflow.get("expected_changed_paths", [])
        if model._path_updated(workspace / path, workspace_snapshot.get(path, ""))
    ]
    updated_generated_paths = [
        path
        for path in workflow.get("generated_paths", [])
        if model._path_updated(workspace / path, workspace_snapshot.get(path, ""))
    ]
    updated_report_paths = [
        path
        for path in workflow.get("report_paths", [])
        if model._path_updated(workspace / path, workspace_snapshot.get(path, ""))
    ]
    preview_paths = prioritized_preview_paths(
        unsatisfied_expected_contents=unsatisfied_expected_contents,
        present_forbidden=present_forbidden,
        changed_preserved=changed_preserved,
        updated_workflow_paths=updated_workflow_paths,
        updated_generated_paths=updated_generated_paths,
        updated_report_paths=updated_report_paths,
        expected=expected,
    )
    preview_payload = workspace_file_previews(
        model,
        workspace,
        preview_paths,
        expected_file_contents={
            path: str(task.expected_file_contents.get(path, ""))
            for path in unsatisfied_expected_contents
            if str(task.expected_file_contents.get(path, ""))
        },
    )
    satisfied_obligations = (
        len(existing_expected)
        + len(satisfied_expected_contents)
        + (len(task.forbidden_files) - len(present_forbidden))
        + len(intact_preserved)
        + len(updated_workflow_paths)
        + len(updated_generated_paths)
        + len(updated_report_paths)
    )
    total_obligations = max(
        1,
        len(expected)
        + len(task.expected_file_contents)
        + len(task.forbidden_files)
        + len(workflow.get("preserved_paths", []))
        + len(workflow.get("expected_changed_paths", []))
        + len(workflow.get("generated_paths", []))
        + len(workflow.get("report_paths", [])),
    )
    return {
        "existing_expected_artifacts": existing_expected,
        "missing_expected_artifacts": missing_expected,
        "satisfied_expected_contents": satisfied_expected_contents,
        "unsatisfied_expected_contents": unsatisfied_expected_contents,
        "present_forbidden_artifacts": present_forbidden,
        "intact_preserved_artifacts": intact_preserved,
        "changed_preserved_artifacts": changed_preserved,
        "missing_preserved_artifacts": missing_preserved,
        "updated_workflow_paths": updated_workflow_paths,
        "updated_generated_paths": updated_generated_paths,
        "updated_report_paths": updated_report_paths,
        "completion_ratio": round(float(satisfied_obligations) / float(total_obligations), 3),
        "workspace_file_previews": preview_payload,
    }


def prioritized_preview_paths(
    *,
    unsatisfied_expected_contents: list[str],
    present_forbidden: list[str],
    changed_preserved: list[str],
    updated_workflow_paths: list[str],
    updated_generated_paths: list[str],
    updated_report_paths: list[str],
    expected: list[str],
) -> list[str]:
    prioritized: list[str] = []
    for path in (
        unsatisfied_expected_contents
        + present_forbidden
        + changed_preserved
        + updated_workflow_paths
        + updated_generated_paths
        + updated_report_paths
        + expected
    ):
        normalized = str(path).strip()
        if normalized and normalized not in prioritized:
            prioritized.append(normalized)
    return prioritized


def workspace_file_previews(
    model,
    workspace: Path,
    candidate_paths: list[str],
    *,
    expected_file_contents: dict[str, str] | None = None,
) -> dict[str, dict[str, object]]:
    previews: dict[str, dict[str, object]] = {}
    preview_targets = expected_file_contents or {}
    for index, path in enumerate(candidate_paths):
        if len(previews) >= model._max_workspace_file_previews():
            break
        max_bytes = (
            model._max_priority_workspace_preview_bytes()
            if index < model._max_priority_workspace_previews()
            else model._max_workspace_preview_bytes()
        )
        max_chars = (
            model._max_priority_workspace_preview_chars()
            if index < model._max_priority_workspace_previews()
            else model._max_workspace_preview_chars()
        )
        preview_windows = targeted_text_file_previews(
            model,
            workspace / path,
            expected_content=preview_targets.get(path, ""),
            max_bytes=max_bytes,
            max_chars=max_chars,
        )
        preview: dict[str, object] | None = None
        if preview_windows:
            preview = dict(preview_windows[0])
            preview["edit_windows"] = [dict(window) for window in preview_windows]
        if preview is None:
            preview = text_file_preview(
                model,
                workspace / path,
                max_bytes=max_bytes,
                max_chars=max_chars,
            )
        if preview is not None:
            previews[path] = preview
    return previews


def text_file_preview(
    model,
    path: Path,
    *,
    max_bytes: int,
    max_chars: int,
) -> dict[str, object] | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        payload = path.read_bytes()
    except OSError:
        return None
    if b"\0" in payload:
        return None
    try:
        full_content = payload.decode("utf-8")
    except UnicodeDecodeError:
        return None
    content = bounded_text_prefix(
        full_content,
        max_bytes=max_bytes,
        max_chars=max_chars,
    )
    truncated = content != full_content
    edit_content = content
    if truncated and content and not content.endswith("\n"):
        last_newline = content.rfind("\n")
        edit_content = content[: last_newline + 1] if last_newline >= 0 else ""
    return {
        "content": content,
        "truncated": truncated,
        "edit_content": edit_content,
        "line_start": 1,
        "line_end": len(edit_content.splitlines()),
        "target_line_start": 1,
        "target_line_end": len(edit_content.splitlines()),
        "line_delta": 0,
        "omitted_prefix_sha1": sha1_text(""),
        "omitted_suffix_sha1": (
            sha1_text(full_content[len(edit_content) :])
            if truncated
            else sha1_text("")
        ),
        "omitted_sha1": (
            sha1_text(full_content[len(edit_content) :])
            if truncated
            else ""
        ),
    }


def targeted_text_file_previews(
    model,
    path: Path,
    *,
    expected_content: str,
    max_bytes: int,
    max_chars: int,
) -> list[dict[str, object]]:
    if not expected_content:
        return []
    current_content = read_text_file(path)
    if current_content is None or current_content == expected_content:
        return []
    if bounded_text_prefix(current_content, max_bytes=max_bytes, max_chars=max_chars) == current_content:
        return []
    current_lines = current_content.splitlines(keepends=True)
    expected_lines = expected_content.splitlines(keepends=True)
    all_windows = targeted_preview_windows(
        current_lines=current_lines,
        expected_lines=expected_lines,
        max_bytes=max_bytes,
        max_chars=max_chars,
    )
    windows = all_windows[: model._max_targeted_preview_windows()]
    if not windows:
        return []
    total_window_count = len(all_windows)
    retained_window_count = len(windows)
    previews: list[dict[str, object]] = []
    for window_index, (start, end, target_start, target_end) in enumerate(windows):
        if start == 0 and end >= len(current_lines):
            continue
        visible_content = "".join(current_lines[start:end])
        target_visible_content = "".join(expected_lines[target_start:target_end])
        if not visible_content and not target_visible_content:
            continue
        previews.append(
            {
                "content": visible_content,
                "truncated": True,
                "window_index": window_index,
                "edit_content": visible_content,
                "target_edit_content": target_visible_content,
                "line_start": start + 1,
                "line_end": end,
                "target_line_start": target_start + 1,
                "target_line_end": target_end,
                "line_delta": (target_end - target_start) - (end - start),
                "retained_edit_window_count": retained_window_count,
                "total_edit_window_count": total_window_count,
                "partial_window_coverage": retained_window_count < total_window_count,
                "omitted_prefix_sha1": sha1_text("".join(current_lines[:start])),
                "omitted_suffix_sha1": sha1_text("".join(current_lines[end:])),
                "omitted_sha1": sha1_text("".join(current_lines[end:])),
            }
        )
    exact_proof_windows = exact_targeted_preview_proof_windows(
        windows=all_windows,
        retained_window_count=retained_window_count,
    )
    bridged_previews = bridged_targeted_preview_windows(
        current_lines=current_lines,
        expected_lines=expected_lines,
        windows=all_windows,
        max_bytes=max_bytes,
        max_chars=max_chars,
    )
    bridged_preview_runs = model._bridged_targeted_preview_window_runs(
        bridged_windows=bridged_previews
    )
    current_proof_regions = model._hidden_gap_current_proof_regions(
        bridged_runs=bridged_preview_runs
    )
    sparse_current_proof_regions = model._sparse_hidden_gap_current_proof_regions(
        windows=all_windows,
        bridged_windows=bridged_previews,
        start_region_index=len(current_proof_regions),
    )
    if sparse_current_proof_regions:
        current_proof_regions = dedupe_hidden_gap_current_proof_regions(
            [*current_proof_regions, *sparse_current_proof_regions]
        )
    if previews and exact_proof_windows:
        previews[0]["exact_edit_window_proofs"] = exact_proof_windows
    if previews and bridged_previews:
        previews[0]["bridged_edit_windows"] = bridged_previews
    if previews and bridged_preview_runs:
        previews[0]["bridged_edit_window_runs"] = bridged_preview_runs
    if previews and current_proof_regions:
        previews[0]["hidden_gap_current_proof_regions"] = current_proof_regions
    return previews


def dedupe_hidden_gap_current_proof_regions(
    regions: list[dict[str, object]],
) -> list[dict[str, object]]:
    deduped: list[dict[str, object]] = []
    seen: set[
        tuple[
            tuple[int, ...],
            tuple[tuple[int, int, bool], ...],
            tuple[tuple[int, int], ...],
            int,
            int,
            bool,
        ]
    ] = set()
    for region in regions:
        if not isinstance(region, dict):
            continue
        window_indices = tuple(
            int(index)
            for index in region.get("window_indices", [])
            if isinstance(index, int)
        )
        raw_spans = region.get("current_proof_spans", [])
        if not isinstance(raw_spans, list):
            raw_spans = []
        span_signature = tuple(
            (
                int(span.get("current_line_start", 1)),
                int(span.get("current_line_end", 0)),
                bool(span.get("current_from_line_span_proof", False)),
            )
            for span in raw_spans
            if isinstance(span, dict)
        )
        raw_opaque_spans = region.get("current_proof_opaque_spans", [])
        if not isinstance(raw_opaque_spans, list):
            raw_opaque_spans = []
        opaque_span_signature = tuple(
            (
                int(span.get("current_line_start", 1)),
                int(span.get("current_line_end", 0)),
            )
            for span in raw_opaque_spans
            if isinstance(span, dict)
        )
        signature = (
            window_indices,
            span_signature,
            opaque_span_signature,
            int(region.get("line_start", 1)),
            int(region.get("line_end", 0)),
            bool(region.get("current_proof_partial_coverage", False)),
        )
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(region)
    return deduped


def exact_targeted_preview_proof_windows(
    *,
    windows: list[tuple[int, int, int, int]],
    retained_window_count: int,
) -> list[dict[str, object]]:
    proofs: list[dict[str, object]] = []
    if retained_window_count >= len(windows):
        return proofs
    for window_index, (start, end, target_start, target_end) in enumerate(
        windows[retained_window_count:],
        start=retained_window_count,
    ):
        proofs.append(
            {
                "window_index": window_index,
                "explicit_current_span_proof": True,
                "line_start": start + 1,
                "line_end": end,
                "target_line_start": target_start + 1,
                "target_line_end": target_end,
                "current_line_count": max(0, end - start),
                "target_line_count": max(0, target_end - target_start),
                "line_delta": (target_end - target_start) - (end - start),
            }
        )
    return proofs


def bridged_targeted_preview_windows(
    *,
    current_lines: list[str],
    expected_lines: list[str],
    windows: list[tuple[int, int, int, int]],
    max_bytes: int,
    max_chars: int,
) -> list[dict[str, object]]:
    bridged: list[dict[str, object]] = []
    for index, (left_start, left_end, left_target_start, left_target_end) in enumerate(windows[:-1]):
        right_start, right_end, right_target_start, right_target_end = windows[index + 1]
        current_hidden_gap_line_count = max(0, right_start - left_end)
        target_hidden_gap_line_count = max(0, right_target_start - left_target_end)
        if current_hidden_gap_line_count <= 0 and target_hidden_gap_line_count <= 0:
            continue
        hidden_current_content = "".join(current_lines[left_end:right_start])
        hidden_target_content = "".join(expected_lines[left_target_end:right_target_start])
        if not hidden_current_content and not hidden_target_content:
            continue
        include_current_content = fits_preview_budget(
            hidden_current_content,
            max_bytes=max_bytes,
            max_chars=max_chars,
        )
        include_target_content = fits_dual_preview_budget(
            baseline_content=hidden_current_content,
            target_content=hidden_target_content,
            max_bytes=max_bytes,
            max_chars=max_chars,
        )
        bridged.append(
            {
                "truncated": True,
                "bridge_window_indices": [index, index + 1],
                "explicit_hidden_gap_current_proof": True,
                "line_start": left_start + 1,
                "line_end": right_end,
                "target_line_start": left_target_start + 1,
                "target_line_end": right_target_end,
                "hidden_gap_current_line_start": left_end + 1,
                "hidden_gap_current_line_end": right_start,
                "hidden_gap_target_line_start": left_target_end + 1,
                "hidden_gap_target_line_end": right_target_start,
                "hidden_gap_current_content": (
                    hidden_current_content if include_current_content else ""
                ),
                "hidden_gap_target_content": (
                    hidden_target_content if include_target_content else ""
                ),
                "hidden_gap_current_from_line_span_proof": not include_current_content,
                "hidden_gap_target_from_expected_content": not include_target_content,
                "hidden_gap_current_line_count": current_hidden_gap_line_count,
                "hidden_gap_target_line_count": target_hidden_gap_line_count,
                "line_delta": target_hidden_gap_line_count - current_hidden_gap_line_count,
            }
        )
    return bridged


def targeted_preview_windows(
    *,
    current_lines: list[str],
    expected_lines: list[str],
    max_bytes: int,
    max_chars: int,
) -> list[tuple[int, int, int, int]]:
    if not current_lines:
        return []
    matcher = difflib.SequenceMatcher(a=current_lines, b=expected_lines)
    windows: list[tuple[int, int, int, int]] = []
    for tag, current_start, current_end, expected_start, expected_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        start, end, target_start, target_end = expanded_preview_window(
            current_lines=current_lines,
            expected_lines=expected_lines,
            current_start=current_start,
            current_end=current_end,
            expected_start=expected_start,
            expected_end=expected_end,
            max_bytes=max_bytes,
            max_chars=max_chars,
        )
        if start is None or end is None or target_start is None or target_end is None:
            continue
        if windows:
            previous_start, previous_end, previous_target_start, previous_target_end = windows[-1]
            merged = merge_preview_windows(
                current_lines=current_lines,
                expected_lines=expected_lines,
                left=(previous_start, previous_end, previous_target_start, previous_target_end),
                right=(start, end, target_start, target_end),
                max_bytes=max_bytes,
                max_chars=max_chars,
            )
            if merged is not None:
                windows[-1] = merged
                continue
        windows.append((start, end, target_start, target_end))
    return windows


def expanded_preview_window(
    *,
    current_lines: list[str],
    expected_lines: list[str],
    current_start: int,
    current_end: int,
    expected_start: int,
    expected_end: int,
    max_bytes: int,
    max_chars: int,
) -> tuple[int | None, int | None, int | None, int | None]:
    if not current_lines and not expected_lines:
        return None, None, None, None
    if current_start < current_end:
        start = current_start
        end = current_end
    else:
        anchor = min(current_start, max(0, len(current_lines) - 1))
        start = anchor
        end = min(len(current_lines), anchor + 1)
    target_start = min(expected_start, len(expected_lines))
    target_end = max(target_start, expected_end)
    if start >= end and target_start >= target_end:
        return None, None, None, None
    if not fits_dual_preview_budget(
        baseline_content="".join(current_lines[start:end]),
        target_content="".join(expected_lines[target_start:target_end]),
        max_bytes=max_bytes,
        max_chars=max_chars,
    ):
        return None, None, None, None
    expanded = True
    while expanded:
        expanded = False
        if start > 0 and target_start > 0 and current_lines[start - 1] == expected_lines[target_start - 1]:
            if fits_dual_preview_budget(
                baseline_content="".join(current_lines[start - 1 : end]),
                target_content="".join(expected_lines[target_start - 1 : target_end]),
                max_bytes=max_bytes,
                max_chars=max_chars,
            ):
                start -= 1
                target_start -= 1
                expanded = True
        if (
            end < len(current_lines)
            and target_end < len(expected_lines)
            and current_lines[end] == expected_lines[target_end]
        ):
            if fits_dual_preview_budget(
                baseline_content="".join(current_lines[start : end + 1]),
                target_content="".join(expected_lines[target_start : target_end + 1]),
                max_bytes=max_bytes,
                max_chars=max_chars,
            ):
                end += 1
                target_end += 1
                expanded = True
    return start, end, target_start, target_end


def merge_preview_windows(
    *,
    current_lines: list[str],
    expected_lines: list[str],
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
    max_bytes: int,
    max_chars: int,
) -> tuple[int, int, int, int] | None:
    start = min(left[0], right[0])
    end = max(left[1], right[1])
    target_start = min(left[2], right[2])
    target_end = max(left[3], right[3])
    if not fits_dual_preview_budget(
        baseline_content="".join(current_lines[start:end]),
        target_content="".join(expected_lines[target_start:target_end]),
        max_bytes=max_bytes,
        max_chars=max_chars,
    ):
        return None
    return start, end, target_start, target_end


def read_text_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        payload = path.read_bytes()
    except OSError:
        return None
    if b"\0" in payload:
        return None
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError:
        return None


def fits_preview_budget(content: str, *, max_bytes: int, max_chars: int) -> bool:
    return len(content) <= max_chars and len(content.encode("utf-8")) <= max_bytes


def fits_dual_preview_budget(
    *,
    baseline_content: str,
    target_content: str,
    max_bytes: int,
    max_chars: int,
) -> bool:
    return fits_preview_budget(baseline_content, max_bytes=max_bytes, max_chars=max_chars) and (
        fits_preview_budget(target_content, max_bytes=max_bytes, max_chars=max_chars)
    )


def bounded_text_prefix(content: str, *, max_bytes: int, max_chars: int) -> str:
    if max_bytes <= 0 or max_chars <= 0 or not content:
        return ""
    pieces: list[str] = []
    used_bytes = 0
    used_chars = 0
    for char in content:
        char_bytes = len(char.encode("utf-8"))
        if used_chars >= max_chars or used_bytes + char_bytes > max_bytes:
            break
        pieces.append(char)
        used_bytes += char_bytes
        used_chars += 1
    return "".join(pieces)


def sha1_text(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()
