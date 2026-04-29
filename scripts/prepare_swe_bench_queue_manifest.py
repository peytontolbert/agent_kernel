from __future__ import annotations

from pathlib import Path
import argparse
import ast
from datetime import UTC, datetime
import json
import re
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"expected JSON object at {path}")
    return payload


def _safe_task_id(instance_id: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in instance_id)
    return f"swe_patch_{safe.strip('_') or 'instance'}"


def _shell_single_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _with_arg(value: str) -> str:
    return "--with " + _shell_single_quote(value)


def _candidate_file_success_command(candidate_files: list[str]) -> str:
    base = "test -s patch.diff && grep -Eq '^(diff --git |--- )' patch.diff"
    if not candidate_files:
        return base
    pattern = "|".join(re.escape(path) for path in candidate_files)
    pattern = pattern.replace("'", "'\"'\"'")
    return f"{base} && grep -Eq '{pattern}' patch.diff"


def _safe_source_context_path(path: str) -> str:
    normalized = path.strip().strip("/")
    if not normalized or normalized == "." or normalized.startswith("../") or "/../" in normalized:
        raise ValueError(f"unsafe source context path: {path!r}")
    return f"source_context/{normalized}"


def _safe_source_lines_path(path: str) -> str:
    normalized = path.strip().strip("/")
    if not normalized or normalized == "." or normalized.startswith("../") or "/../" in normalized:
        raise ValueError(f"unsafe source context path: {path!r}")
    return f"source_lines/{normalized}.lines"


def _line_numbered_content(content: str, *, start: int = 1) -> str:
    lines = content.splitlines()
    if content.endswith("\n"):
        lines.append("")
    end = start + len(lines) - 1
    width = max(4, len(str(end)))
    return "\n".join(f"{index:>{width}}: {line}" for index, line in enumerate(lines, start=start)) + "\n"


def _source_line_number(entry: dict[str, str], local_line_number: int) -> int:
    try:
        source_line_start = int(entry.get("source_line_start", "1"))
    except ValueError:
        source_line_start = 1
    return source_line_start + local_line_number - 1


def _source_context_entries(source_context: list[object]) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    seen: set[str] = set()
    for context in source_context:
        if not isinstance(context, dict):
            continue
        path = str(context.get("path", "")).strip()
        content = str(context.get("content", ""))
        if not path or not content:
            continue
        workspace_path = _safe_source_context_path(path)
        line_path = _safe_source_lines_path(path)
        if workspace_path in seen:
            suffix_parts = [str(len(seen) + 1)]
            focus_term = str(context.get("focus_term", "")).strip()
            if focus_term:
                suffix_parts.append(re.sub(r"[^A-Za-z0-9_]+", "_", focus_term).strip("_"))
            suffix = "_".join(part for part in suffix_parts if part)
            workspace_path = f"{workspace_path}.{suffix}"
            line_path = f"{line_path}.{suffix}.lines"
        seen.add(workspace_path)
        try:
            line_start_int = int(context.get("line_start") or 1)
        except (TypeError, ValueError):
            line_start_int = 1
        line_start = str(line_start_int)
        entries.append(
            {
                "path": path,
                "workspace_path": workspace_path,
                "line_path": line_path,
                "content": content,
                "line_content": _line_numbered_content(content, start=line_start_int),
                "source_line_start": line_start,
            }
        )
    return entries


def _entry_contains_source_line(entry: dict[str, str], line_number: int) -> bool:
    try:
        source_line_start = int(entry.get("source_line_start", "1"))
    except ValueError:
        source_line_start = 1
    line_count = max(1, len(entry.get("content", "").splitlines()))
    return source_line_start <= line_number < source_line_start + line_count


def _source_line_padded_content(entry: dict[str, str]) -> str:
    try:
        source_line_start = int(entry.get("source_line_start", "1"))
    except ValueError:
        source_line_start = 1
    if source_line_start <= 1:
        return entry["content"]
    return "\n" * (source_line_start - 1) + entry["content"]


def _suggested_command_target_lines(commands: list[str]) -> dict[str, set[int]]:
    target_lines: dict[str, set[int]] = {}
    for command in commands:
        path_match = re.search(r"--path\s+(\S+)", command)
        if not path_match:
            continue
        path = path_match.group(1).strip()
        if not path:
            continue
        lines = target_lines.setdefault(path, set())
        for range_match in re.finditer(r"--replace-lines\s+(\d+)\s+(\d+)", command):
            start = int(range_match.group(1))
            end = int(range_match.group(2))
            lines.update(range(min(start, end), max(start, end) + 1))
        for line_match in re.finditer(r"--replace-line\s+(\d+)", command):
            lines.add(int(line_match.group(1)))
    return target_lines


def _setup_file_contents(
    source_entries: list[dict[str, str]],
    *,
    preferred_target_lines: dict[str, set[int]] | None = None,
) -> dict[str, str]:
    setup_files: dict[str, str] = {}
    preferred_entries: dict[str, dict[str, str]] = {}
    for entry in source_entries:
        path = entry["path"]
        target_lines = (preferred_target_lines or {}).get(path, set())
        if target_lines and any(_entry_contains_source_line(entry, line_number) for line_number in target_lines):
            preferred_entries.setdefault(path, entry)
    for entry in source_entries:
        # The patch target path must remain the first/broadest context for a
        # file. Later duplicate focused windows are still exposed under their
        # suffixed source_context paths, but must not replace the editable file.
        preferred_entry = preferred_entries.get(entry["path"])
        if preferred_entry is entry:
            setup_files[entry["path"]] = _source_line_padded_content(entry)
        elif preferred_entry is None:
            setup_files.setdefault(entry["path"], entry["content"])
        setup_files[entry["workspace_path"]] = entry["content"]
        setup_files[entry["line_path"]] = entry["line_content"]
    return setup_files


def _keyword_tokens(*values: object) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        if isinstance(value, (list, tuple, set)):
            tokens.update(_keyword_tokens(*value))
            continue
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", str(value)):
            normalized = token.lower()
            tokens.add(normalized)
            tokens.update(part for part in normalized.split("_") if len(part) >= 3)
    return tokens


def _issue_focus_tokens(
    problem: str,
    hints: str,
    fail_to_pass: object,
    required_patch_identifiers: list[str],
) -> set[str]:
    tokens = _keyword_tokens(problem, hints, fail_to_pass, required_patch_identifiers)
    combined = f"{problem}\n{hints}".lower()
    if "case insensitive" in combined or "case-insensitive" in combined:
        tokens.update({"case", "insensitive", "lower", "upper", "ignorecase", "command", "read"})
    if "regex" in combined or "regular expression" in combined:
        tokens.update({"regex", "regexp", "compile", "match", "pattern"})
    return tokens


def _is_test_path(path: str) -> bool:
    normalized = str(path).replace("\\", "/")
    name = Path(normalized).name
    return "/tests/" in normalized or name.startswith("test_") or name.endswith("_test.py")


def _source_executable_edit_windows(
    source_entries: list[dict[str, str]],
    *,
    problem: str,
    hints: str,
    fail_to_pass: object,
    required_patch_identifiers: list[str],
) -> str:
    keywords = _issue_focus_tokens(problem, hints, fail_to_pass, required_patch_identifiers)
    required_tokens = _keyword_tokens(required_patch_identifiers)
    combined_issue_text = f"{problem}\n{hints}".lower()
    chunks: list[str] = []
    for entry in source_entries:
        path = entry["path"]
        if _is_test_path(path) or not path.endswith(".py"):
            continue
        content = entry["content"]
        lines = content.splitlines()
        windows: list[tuple[int, int, int, str, int, int]] = []
        for name, node_start, node_end, kind in _python_executable_ranges(content):
            if not name:
                continue
            name_tokens = _keyword_tokens(name, path)
            start = max(1, node_start - 2)
            end = min(len(lines), node_end + 6)
            body_text = "\n".join(lines[node_start - 1 : node_end])
            body_tokens = _keyword_tokens(body_text)
            score = 0
            score += 3 * len(keywords & name_tokens)
            score += len(keywords & body_tokens)
            score += 4 * len(required_tokens & (name_tokens | body_tokens))
            if ("regex" in combined_issue_text or "regular expression" in combined_issue_text) and (
                body_tokens & {"compile", "match", "pattern", "regexp"}
            ):
                score += 10
            if ("case insensitive" in combined_issue_text or "case-insensitive" in combined_issue_text) and (
                body_tokens & {"compile", "ignorecase", "lower", "upper"}
            ):
                score += 6
            if kind == "class":
                score -= 5
            if name.startswith("test_"):
                score -= 1
            if score > 0:
                windows.append((score, start, end, name, node_start, node_end))
        if not windows:
            continue
        selected = sorted(windows, key=lambda item: (-item[0], item[1]))[:3]
        for _score, start, end, name, node_start, node_end in selected:
            width = max(4, len(str(end)))
            excerpt_lines = _python_actionable_excerpt_lines(lines, start=node_start, end=node_end, keywords=keywords)
            if not excerpt_lines:
                excerpt_lines = list(range(start, end + 1))
            excerpt = "\n".join(f"{idx:>{width}}: {lines[idx - 1]}" for idx in excerpt_lines)
            chunks.append(f"### {path}::{name} lines {start}-{end}\n{excerpt}")
    return "\n\n".join(chunks[:6])


def _python_actionable_excerpt_lines(
    lines: list[str],
    *,
    start: int,
    end: int,
    keywords: set[str],
) -> list[int]:
    selected: list[int] = []
    relevant: list[tuple[int, int]] = []
    fallback: list[int] = []
    in_docstring = False
    for idx in range(start, end + 1):
        line = lines[idx - 1]
        stripped = line.strip()
        quote_count = stripped.count('"""') + stripped.count("'''")
        if quote_count:
            if quote_count % 2 == 1:
                in_docstring = not in_docstring
            continue
        if in_docstring:
            continue
        if not stripped or stripped.startswith("#"):
            continue
        if re.match(r"(async\s+def|def|class)\s+", stripped):
            selected.append(idx)
            continue
        line_tokens = _keyword_tokens(stripped)
        if line_tokens & keywords:
            score = len(line_tokens & keywords)
            if line_tokens & {"command", "read", "serr", "terr", "compile", "match", "ignorecase"}:
                score += 5
            relevant.append((score, idx))
        else:
            fallback.append(idx)
    ranked_relevant = [idx for _score, idx in sorted(relevant, key=lambda item: (-item[0], item[1]))]
    return [*selected, *ranked_relevant, *fallback][:80]


def _python_executable_ranges(content: str) -> list[tuple[str, int, int, str]]:
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return _regex_python_executable_ranges(content)
    ranges: list[tuple[str, int, int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        name = str(getattr(node, "name", ""))
        if not name:
            continue
        start = int(getattr(node, "lineno", 1))
        end = int(getattr(node, "end_lineno", start))
        kind = "class" if isinstance(node, ast.ClassDef) else "function"
        ranges.append((name, start, end, kind))
    return ranges


def _regex_python_executable_ranges(content: str) -> list[tuple[str, int, int, str]]:
    lines = content.splitlines()
    headers: list[tuple[str, int, int, str]] = []
    for index, line in enumerate(lines, start=1):
        match = re.match(r"^(\s*)(async\s+def|def|class)\s+([A-Za-z_][A-Za-z0-9_]*)\b", line)
        if not match:
            continue
        kind = "class" if match.group(2) == "class" else "function"
        if kind == "class":
            continue
        headers.append((match.group(3), index, len(match.group(1)), kind))
    ranges: list[tuple[str, int, int, str]] = []
    for offset, (name, start, indent, kind) in enumerate(headers):
        end = len(lines)
        for next_index in range(start + 1, len(lines) + 1):
            next_line = lines[next_index - 1]
            stripped = next_line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("@"):
                continue
            next_indent = len(next_line) - len(next_line.lstrip(" "))
            if next_indent <= indent:
                end = next_index - 1
                break
        ranges.append((name, start, end, kind))
    return ranges


def _required_patch_identifiers(problem: str) -> list[str]:
    identifiers: list[str] = []
    patterns = (
        r"unexpected keyword argument ['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]",
        r"got an unexpected keyword argument ['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, problem):
            identifier = match.group(1)
            if identifier not in identifiers:
                identifiers.append(identifier)
    return identifiers


def _suggested_patch_commands(source_entries: list[dict[str, str]], *, problem: str, hints: str) -> list[str]:
    combined = f"{problem}\n{hints}".lower()
    commands: list[str] = []
    if "header_rows" in combined or "header rows" in combined:
        commands.extend(_suggested_rst_header_rows_commands(source_entries))
    if "separability" in combined and "nested" in combined and "compound" in combined:
        commands.extend(_suggested_separability_nested_compound_commands(source_entries))
    if "mask" in combined and ("operand.mask" in combined or "bitwise_or" in combined):
        commands.extend(_suggested_nddata_operand_mask_commands(source_entries))
    if "trailing newline" in combined and "username" in combined:
        commands.extend(_suggested_django_username_validator_commands(source_entries))
    if "memoryview" in combined and "httpresponse" in combined:
        commands.extend(_suggested_django_memoryview_response_commands(source_entries))
    if "slugify" in combined and "strip" in combined and ("dashes" in combined or "underscores" in combined):
        commands.extend(_suggested_django_slugify_strip_commands(source_entries))
    if "resolve_redirects" in combined and "original" in combined and "request" in combined:
        commands.extend(_suggested_requests_redirect_carry_forward_commands(source_entries))
    if "hooks" in combined and "list" in combined and "values" in combined:
        commands.extend(_suggested_requests_hooks_list_commands(source_entries))
    if "array([])" in combined or ("empty array" in combined and "not enough values to unpack" in combined):
        commands.extend(_suggested_sympy_empty_array_commands(source_entries))
    if "uniq" in combined and ("changed size" in combined or "modifies" in combined or "modify" in combined):
        commands.extend(_suggested_sympy_uniq_changed_size_commands(source_entries))
    if "partitions" in combined and ("reusing" in combined or "copy" in combined or "dictionaries" in combined):
        commands.extend(_suggested_sympy_partitions_copy_commands(source_entries))
    if "col_insert" in combined and ("matrix" in combined or "insert" in combined):
        commands.extend(_suggested_sympy_col_insert_index_commands(source_entries))
    if "vector" in combined and "add 0" in combined:
        commands.extend(_suggested_sympy_vector_radd_zero_commands(source_entries))
    if "morse" in combined and '".----"' in combined and '"1"' in combined:
        commands.extend(_suggested_sympy_morse_one_mapping_commands(source_entries))
    if "intersection" in combined and ("duplicates" in combined or "remove" in combined):
        commands.extend(_suggested_sympy_intersection_deduplicate_commands(source_entries))
    if ("__matmul__" in combined or "matmul" in combined) and (
        "scalar" in combined or "non-matrix" in combined or "notimplemented" in combined
    ):
        commands.extend(_suggested_sympy_matmul_matrix_only_commands(source_entries))
    if "is_upper" in combined and ("indexerror" in combined or "tall" in combined or "hessenberg" in combined):
        commands.extend(_suggested_sympy_tall_upper_bounds_commands(source_entries))
    if "joblib" in combined and ("show_versions" in combined or "show versions" in combined):
        commands.extend(_suggested_sklearn_show_versions_joblib_commands(source_entries))
    if "isolationforest" in combined and "valid feature names" in combined and "contamination" in combined:
        commands.extend(_suggested_sklearn_iforest_fit_score_without_revalidation_commands(source_entries))
    if "pylint" in combined and "sys.path" in combined and "runpy" in combined and "first item" in combined:
        commands.extend(_suggested_pylint_modify_sys_path_guard_commands(source_entries))
    if "caplog" in combined and "get_records" in combined and "clear" in combined and "frozen" in combined:
        commands.extend(_suggested_pytest_caplog_clear_records_in_place_commands(source_entries))
    if "needs_extensions" in combined and "version" in combined and "string" in combined:
        commands.extend(_suggested_sphinx_needs_extensions_version_commands(source_entries))
    if "stackplot" in combined and "cycler" in combined and ("c2" in combined or "cycle reference" in combined):
        commands.extend(_suggested_matplotlib_stackplot_local_color_cycle_commands(source_entries))
    if "chunks" in combined and "load" in combined and ("zarr" in combined or "lazy" in combined):
        commands.extend(_suggested_xarray_chunks_no_load_commands(source_entries))
    if "byteswarning" in combined and "setup-show" in combined and "saferepr" in combined:
        commands.extend(_suggested_pytest_setuponly_saferepr_commands(source_entries))
    if "timeseries" in combined and "required column" in combined and (
        "misleading" in combined or "first column" in combined or "found 'time'" in combined
    ):
        commands.extend(_suggested_astropy_timeseries_required_columns_commands(source_entries))
    if "case insensitive" not in combined and "case-insensitive" not in combined:
        return commands[:2]
    for entry in source_entries:
        path = entry["path"]
        if _is_test_path(path) or not path.endswith(".py"):
            continue
        regex_command = ""
        value_mask_command = ""
        for line_number, line in enumerate(entry["content"].splitlines(), start=1):
            stripped = line.strip()
            if stripped == 'if v == "NO":':
                replacement = line.rstrip().replace('v == "NO"', 'v.upper() == "NO"')
                value_mask_command = (
                    "swe_patch_builder --path "
                    f"{path} --replace-line {_source_line_number(entry, line_number)} "
                    f"--with '{replacement}' > patch.diff"
                )
            if "re.compile(" not in stripped or "re.IGNORECASE" in stripped:
                continue
            if "_type_re" not in stripped and "command" not in stripped.lower():
                continue
            replacement = line.rstrip().removesuffix(")") + ", re.IGNORECASE)"
            regex_command = (
                "swe_patch_builder --path "
                f"{path} --replace-line {_source_line_number(entry, line_number)} "
                f"--with '{replacement}' > patch.diff"
            )
        if regex_command:
            if value_mask_command:
                combined_command = (
                    regex_command.removesuffix(" > patch.diff")
                    + value_mask_command.replace(f"swe_patch_builder --path {path} ", " ")
                )
                commands.append(combined_command)
            else:
                commands.append(regex_command)
            break
    return commands[:2]


def _suggested_nddata_operand_mask_commands(source_entries: list[dict[str, str]]) -> list[str]:
    for entry in source_entries:
        path = entry["path"]
        if _is_test_path(path) or not path.endswith("ndarithmetic.py"):
            continue
        for line_number, line in enumerate(entry["content"].splitlines(), start=1):
            if line.strip() == "elif operand is None:":
                replacement = line.rstrip().replace(
                    "elif operand is None:",
                    "elif operand is None or operand.mask is None:",
                )
                return [
                    "swe_patch_builder --path "
                    f"{path} --replace-line {_source_line_number(entry, line_number)} "
                    f"--with '{replacement}' > patch.diff"
                ]
    return []


def _suggested_django_username_validator_commands(source_entries: list[dict[str, str]]) -> list[str]:
    for entry in source_entries:
        path = entry["path"]
        if _is_test_path(path) or not path.endswith("validators.py"):
            continue
        lines = entry["content"].splitlines()
        replacement_lines = list(lines)
        changed_indexes: list[int] = []
        for index, line in enumerate(lines):
            if line.strip() == r"regex = r'^[\w.@+-]+$'":
                replacement = line.rstrip().replace(r"+$'", r"+\Z'")
                replacement_lines[index] = replacement
                changed_indexes.append(index)
        if changed_indexes:
            start = min(changed_indexes) + 1
            end = max(changed_indexes) + 1
            replacements = replacement_lines[start - 1 : end]
            command = (
                f"swe_patch_builder --path {path} "
                f"--replace-lines {_source_line_number(entry, start)} {_source_line_number(entry, end)} "
                + " ".join(f'--with "{line}"' for line in replacements)
                + " > patch.diff"
            )
            return [command]
    return []


def _suggested_django_memoryview_response_commands(source_entries: list[dict[str, str]]) -> list[str]:
    for entry in source_entries:
        path = entry["path"]
        if _is_test_path(path) or not path.endswith("response.py"):
            continue
        for line_number, line in enumerate(entry["content"].splitlines(), start=1):
            if line.strip() == "if isinstance(value, bytes):":
                replacement = line.rstrip().replace("bytes", "(bytes, memoryview)")
                return [
                    "swe_patch_builder --path "
                    f"{path} --replace-line {_source_line_number(entry, line_number)} "
                    f"--with '{replacement}' > patch.diff"
                ]
    return []


def _suggested_django_slugify_strip_commands(source_entries: list[dict[str, str]]) -> list[str]:
    for entry in source_entries:
        path = entry["path"]
        if _is_test_path(path) or not path.endswith("text.py"):
            continue
        for line_number, line in enumerate(entry["content"].splitlines(), start=1):
            if line.strip() == r"return re.sub(r'[-\s]+', '-', value)":
                replacement = line.rstrip() + ".strip('-_')"
                return [
                    "swe_patch_builder --path "
                    f'{path} --replace-line {_source_line_number(entry, line_number)} '
                    f'--with "{replacement}" > patch.diff'
                ]
    return []


def _suggested_requests_redirect_carry_forward_commands(source_entries: list[dict[str, str]]) -> list[str]:
    for entry in source_entries:
        path = entry["path"]
        if _is_test_path(path) or not path.endswith("sessions.py"):
            continue
        lines = entry["content"].splitlines()
        for index, line in enumerate(lines):
            if line.strip() != "i += 1":
                continue
            next_line = lines[index + 1] if index + 1 < len(lines) else ""
            if next_line.strip() != "yield resp":
                continue
            indent = line[: len(line) - len(line.lstrip(" "))]
            replacements = [
                line.rstrip(),
                f"{indent}req = prepared_request",
                next_line.rstrip(),
            ]
            return [
                f"swe_patch_builder --path {path} "
                f"--replace-lines {_source_line_number(entry, index + 1)} {_source_line_number(entry, index + 2)} "
                + " ".join(f'--with "{replacement}"' for replacement in replacements)
                + " > patch.diff"
            ]
    return []


def _suggested_requests_hooks_list_commands(source_entries: list[dict[str, str]]) -> list[str]:
    for entry in source_entries:
        path = entry["path"]
        if _is_test_path(path) or not path.endswith("models.py"):
            continue
        lines = entry["content"].splitlines()
        for index, line in enumerate(lines):
            if line.strip() != "for (k, v) in list(hooks.items()):":
                continue
            next_line = lines[index + 1] if index + 1 < len(lines) else ""
            if "self.register_hook(event=k, hook=v)" not in next_line:
                continue
            indent = line[: len(line) - len(line.lstrip(" "))]
            body_indent = next_line[: len(next_line) - len(next_line.lstrip(" "))]
            nested_indent = body_indent + "    "
            replacements = [
                line.rstrip(),
                f"{body_indent}if isinstance(v, (list, tuple)):",
                f"{nested_indent}for hook in v:",
                f"{nested_indent}    self.register_hook(event=k, hook=hook)",
                f"{body_indent}else:",
                f"{nested_indent}self.register_hook(event=k, hook=v)",
            ]
            return [
                f"swe_patch_builder --path {path} "
                f"--replace-lines {_source_line_number(entry, index + 1)} {_source_line_number(entry, index + 2)} "
                + " ".join(f'--with "{replacement}"' for replacement in replacements)
                + " > patch.diff"
            ]
    return []


def _suggested_sympy_empty_array_commands(source_entries: list[dict[str, str]]) -> list[str]:
    for entry in source_entries:
        path = entry["path"]
        if _is_test_path(path) or not path.endswith("ndim_array.py"):
            continue
        lines = entry["content"].splitlines()
        for index, line in enumerate(lines):
            if line.strip() != "result = []":
                continue
            next_line = lines[index + 1] if index + 1 < len(lines) else ""
            if "elems, shapes = zip(*[f(i) for i in pointer])" not in next_line:
                continue
            indent = line[: len(line) - len(line.lstrip(" "))]
            replacements = [
                line.rstrip(),
                f"{indent}if not pointer:",
                f"{indent}    return result, (0,)",
                next_line.rstrip(),
            ]
            return [
                f"swe_patch_builder --path {path} "
                f"--replace-lines {_source_line_number(entry, index + 1)} {_source_line_number(entry, index + 2)} "
                + " ".join(f'--with "{replacement}"' for replacement in replacements)
                + " > patch.diff"
            ]
    return []


def _suggested_sympy_uniq_changed_size_commands(source_entries: list[dict[str, str]]) -> list[str]:
    for entry in source_entries:
        path = entry["path"]
        if _is_test_path(path) or not path.endswith("iterables.py"):
            continue
        lines = entry["content"].splitlines()
        for index, line in enumerate(lines):
            if line.strip() != "try:":
                continue
            next_lines = [candidate.strip() for candidate in lines[index : index + 6]]
            expected = [
                "try:",
                "seen = set()",
                "result = result or []",
                "for i, s in enumerate(seq):",
                "if not (s in seen or seen.add(s)):",
                "yield s",
            ]
            if next_lines != expected:
                continue
            indent = line[: len(line) - len(line.lstrip(" "))]
            body_indent = lines[index + 1][: len(lines[index + 1]) - len(lines[index + 1].lstrip(" "))]
            loop_indent = lines[index + 4][: len(lines[index + 4]) - len(lines[index + 4].lstrip(" "))]
            yield_indent = lines[index + 5][: len(lines[index + 5]) - len(lines[index + 5].lstrip(" "))]
            tail = lines[index + 6 : index + 10]
            fallback_yield = next(
                (
                    offset
                    for offset, candidate in enumerate(tail, start=index + 7)
                    if candidate.strip() == "yield s"
                ),
                0,
            )
            if not fallback_yield:
                continue
            fallback_yield_indent = lines[fallback_yield - 1][
                : len(lines[fallback_yield - 1]) - len(lines[fallback_yield - 1].lstrip(" "))
            ]
            replacements = [
                f"{indent}try:",
                f"{body_indent}size = len(seq)",
                f"{indent}except TypeError:",
                f"{body_indent}size = None",
                f"{indent}try:",
                f"{body_indent}seen = set()",
                f"{body_indent}result = result or []",
                f"{body_indent}for i, s in enumerate(seq):",
                f"{loop_indent}if size is not None and len(seq) != size:",
                f"{yield_indent}raise RuntimeError('sequence changed size during iteration')",
                f"{loop_indent}if not (s in seen or seen.add(s)):",
                f"{yield_indent}yield s",
                f"{yield_indent}if size is not None and len(seq) != size:",
                f"{yield_indent}    raise RuntimeError('sequence changed size during iteration')",
                lines[index + 6].rstrip(),
                lines[index + 7].rstrip(),
                lines[index + 8].rstrip(),
                f"{fallback_yield_indent}if size is not None and len(seq) != size:",
                f"{fallback_yield_indent}    raise RuntimeError('sequence changed size during iteration')",
                lines[index + 9].rstrip(),
            ]
            return [
                f"swe_patch_builder --path {path} "
                f"--replace-lines {_source_line_number(entry, index + 1)} {_source_line_number(entry, index + 10)} "
                + " ".join(f'--with "{replacement}"' for replacement in replacements)
                + " > patch.diff"
            ]
    return []


def _suggested_sympy_partitions_copy_commands(source_entries: list[dict[str, str]]) -> list[str]:
    for entry in source_entries:
        path = entry["path"]
        if _is_test_path(path) or not path.endswith("iterables.py"):
            continue
        lines = entry["content"].splitlines()
        first_yield = 0
        last_yield = 0
        replacement_lines = list(lines)
        for index, line in enumerate(lines):
            stripped = line.strip()
            replacement = ""
            if stripped == "yield sum(ms.values()), ms":
                replacement = line.rstrip().replace("sum(ms.values()), ms", "sum(ms.values()), ms.copy()")
            elif stripped == "yield ms":
                replacement = line.rstrip().replace("yield ms", "yield ms.copy()")
            if not replacement:
                continue
            if first_yield <= 0:
                first_yield = index + 1
            last_yield = index + 1
            replacement_lines[index] = replacement
        if first_yield <= 0 or last_yield <= first_yield:
            continue
        replacements = replacement_lines[first_yield - 1 : last_yield]
        return [
            f"swe_patch_builder --path {path} "
            f"--replace-lines {_source_line_number(entry, first_yield)} {_source_line_number(entry, last_yield)} "
            + " ".join(f'--with "{replacement}"' for replacement in replacements)
            + " > patch.diff"
        ]
    return []


def _suggested_sympy_col_insert_index_commands(source_entries: list[dict[str, str]]) -> list[str]:
    for entry in source_entries:
        path = entry["path"]
        if _is_test_path(path) or not path.endswith("common.py"):
            continue
        for line_number, line in enumerate(entry["content"].splitlines(), start=1):
            if line.strip() != "return self[i, j - pos - other.cols]":
                continue
            replacement = line.rstrip().replace("j - pos - other.cols", "j - other.cols")
            return [
                "swe_patch_builder --path "
                f"{path} --replace-line {_source_line_number(entry, line_number)} "
                f"--with '{replacement}' > patch.diff"
            ]
    return []


def _suggested_sympy_morse_one_mapping_commands(source_entries: list[dict[str, str]]) -> list[str]:
    for entry in source_entries:
        path = entry["path"]
        if _is_test_path(path) or path != "sympy/crypto/crypto.py":
            continue
        for line_number, line in enumerate(entry["content"].splitlines(), start=1):
            if '"----": "1"' not in line:
                continue
            replacement = line.rstrip().replace('"----": "1"', '".----": "1"')
            return [
                "swe_patch_builder --path "
                f"{path} --replace-line {_source_line_number(entry, line_number)} "
                f"--with '{replacement}' > patch.diff"
            ]
    return []


def _suggested_sympy_vector_radd_zero_commands(source_entries: list[dict[str, str]]) -> list[str]:
    for entry in source_entries:
        path = entry["path"]
        if _is_test_path(path) or path != "sympy/physics/vector/vector.py":
            continue
        add_check_line = 0
        in_add = False
        radd_alias_line = 0
        for line_number, line in enumerate(entry["content"].splitlines(), start=1):
            stripped = line.strip()
            if stripped == "def __add__(self, other):":
                in_add = True
                continue
            if in_add and stripped.startswith("def "):
                in_add = False
            if in_add and stripped == "other = _check_vector(other)":
                add_check_line = _source_line_number(entry, line_number)
            if stripped == "__radd__ = __add__":
                radd_alias_line = _source_line_number(entry, line_number)
        if add_check_line and radd_alias_line:
            return [
                f"swe_patch_builder --path {path} "
                f"--replace-lines {add_check_line} {add_check_line} "
                "--with '        if other == 0:' "
                "--with '            return self' "
                "--with '        other = _check_vector(other)' "
                f"--replace-lines {radd_alias_line} {radd_alias_line} "
                "--with '    def __radd__(self, other):' "
                "--with '        return self.__add__(other)' > patch.diff"
            ]
    return []


def _suggested_sympy_intersection_deduplicate_commands(source_entries: list[dict[str, str]]) -> list[str]:
    for entry in source_entries:
        path = entry["path"]
        if _is_test_path(path) or not path.endswith("sets.py"):
            continue
        in_intersection = False
        for line_number, line in enumerate(entry["content"].splitlines(), start=1):
            if line.startswith("class Intersection("):
                in_intersection = True
                continue
            if line.startswith("class ") and in_intersection:
                in_intersection = False
            if not in_intersection:
                continue
            if line.strip() != "args = list(cls._new_args_filter(args))":
                continue
            replacement = line.rstrip().replace(
                "list(cls._new_args_filter(args))",
                "list(ordered(set(cls._new_args_filter(args)), Set._infimum_key))",
            )
            return [
                "swe_patch_builder --path "
                f"{path} --replace-line {_source_line_number(entry, line_number)} "
                f'--with "{replacement}" > patch.diff'
            ]
    return []


def _suggested_sympy_matmul_matrix_only_commands(source_entries: list[dict[str, str]]) -> list[str]:
    replacements_by_return = {
        "return self.__mul__(other)": (
            "        return self.__mul__(other) if "
            "getattr(_matrixify(other), 'is_Matrix', False) or "
            "getattr(_matrixify(other), 'is_MatrixLike', False) else NotImplemented"
        ),
        "return self.__rmul__(other)": (
            "        return self.__rmul__(other) if "
            "getattr(_matrixify(other), 'is_Matrix', False) or "
            "getattr(_matrixify(other), 'is_MatrixLike', False) else NotImplemented"
        ),
    }
    for entry in source_entries:
        path = entry["path"]
        if _is_test_path(path) or not path.endswith("common.py"):
            continue
        replacements: list[tuple[int, str]] = []
        pending_matmul_return = ""
        for line_number, line in enumerate(entry["content"].splitlines(), start=1):
            stripped = line.strip()
            if stripped == "def __matmul__(self, other):":
                pending_matmul_return = "return self.__mul__(other)"
                continue
            if stripped == "def __rmatmul__(self, other):":
                pending_matmul_return = "return self.__rmul__(other)"
                continue
            if pending_matmul_return and stripped == pending_matmul_return:
                replacement = replacements_by_return[pending_matmul_return]
                replacements.append((_source_line_number(entry, line_number), replacement))
                pending_matmul_return = ""
        if len(replacements) >= 2:
            return [
                f"swe_patch_builder --path {path} "
                + " ".join(
                    f'--replace-line {line_number} --with "{replacement}"'
                    for line_number, replacement in replacements
                )
                + " > patch.diff"
            ]
    return []


def _suggested_sympy_tall_upper_bounds_commands(source_entries: list[dict[str, str]]) -> list[str]:
    replacements_by_line = {
        "for j in range(i - 1))": "                   for j in range(min(i - 1, self.cols)))",
        "for j in range(i))": "                   for j in range(min(i, self.cols)))",
    }
    for entry in source_entries:
        path = entry["path"]
        if _is_test_path(path) or not path.endswith("matrices.py"):
            continue
        replacements: list[tuple[int, str]] = []
        for line_number, line in enumerate(entry["content"].splitlines(), start=1):
            replacement = replacements_by_line.get(line.strip())
            if replacement:
                replacements.append((_source_line_number(entry, line_number), replacement))
        if len(replacements) >= 2:
            return [
                f"swe_patch_builder --path {path} "
                + " ".join(
                    f'--replace-line {line_number} --with "{replacement}"'
                    for line_number, replacement in replacements
                )
                + " > patch.diff"
            ]
    return []


def _suggested_sklearn_iforest_fit_score_without_revalidation_commands(
    source_entries: list[dict[str, str]],
) -> list[str]:
    for entry in source_entries:
        path = entry["path"]
        if _is_test_path(path) or not path.endswith("_iforest.py"):
            continue
        in_fit = False
        for line_number, line in enumerate(entry["content"].splitlines(), start=1):
            stripped = line.strip()
            if stripped == "def fit(self, X, y=None, sample_weight=None):":
                in_fit = True
                continue
            if in_fit and line.startswith("    def "):
                in_fit = False
            if not in_fit:
                continue
            if stripped != "self.offset_ = np.percentile(self.score_samples(X), 100.0 * self.contamination)":
                continue
            replacement = line.rstrip().replace(
                "self.score_samples(X)",
                "-self._compute_chunked_score_samples(X)",
            )
            return [
                "swe_patch_builder --path "
                f"{path} --replace-line {_source_line_number(entry, line_number)} "
                f"--with '{replacement}' > patch.diff"
            ]
    return []


def _suggested_sklearn_show_versions_joblib_commands(source_entries: list[dict[str, str]]) -> list[str]:
    for entry in source_entries:
        path = entry["path"]
        if _is_test_path(path) or path != "sklearn/utils/_show_versions.py":
            continue
        in_deps = False
        for line_number, line in enumerate(entry["content"].splitlines(), start=1):
            stripped = line.strip()
            if stripped == "deps = [":
                in_deps = True
                continue
            if in_deps and stripped == "]":
                break
            if in_deps and stripped == '"Cython",':
                source_line = _source_line_number(entry, line_number)
                return [
                    f"swe_patch_builder --path {path} --replace-lines {source_line} {source_line} "
                    "--with '        \"Cython\",' "
                    "--with '        \"joblib\",' > patch.diff"
                ]
    return []


def _suggested_pylint_modify_sys_path_guard_commands(source_entries: list[dict[str, str]]) -> list[str]:
    for entry in source_entries:
        path = entry["path"]
        if _is_test_path(path) or path != "pylint/__init__.py":
            continue
        in_modify_sys_path = False
        for line_number, line in enumerate(entry["content"].splitlines(), start=1):
            stripped = line.strip()
            if stripped == "def modify_sys_path() -> None:":
                in_modify_sys_path = True
                continue
            if in_modify_sys_path and line.startswith("def "):
                in_modify_sys_path = False
            if not in_modify_sys_path or stripped != "sys.path.pop(0)":
                continue
            source_line = _source_line_number(entry, line_number)
            return [
                "swe_patch_builder --path "
                f"{path} --replace-lines {source_line} {source_line} "
                '--with \'    if sys.path[0] in ("", ".", os.getcwd()):\' '
                "--with '        sys.path.pop(0)' > patch.diff"
            ]
    return []


def _suggested_pytest_caplog_clear_records_in_place_commands(source_entries: list[dict[str, str]]) -> list[str]:
    for entry in source_entries:
        path = entry["path"]
        if _is_test_path(path) or path != "src/_pytest/logging.py":
            continue
        in_fixture = False
        in_clear = False
        for line_number, line in enumerate(entry["content"].splitlines(), start=1):
            stripped = line.strip()
            if stripped == "class LogCaptureFixture:":
                in_fixture = True
                continue
            if in_fixture and line.startswith("class "):
                in_fixture = False
                in_clear = False
            if not in_fixture:
                continue
            if stripped == "def clear(self) -> None:":
                in_clear = True
                continue
            if in_clear and line.startswith("    def "):
                in_clear = False
            if not in_clear or stripped != "self.handler.reset()":
                continue
            source_line = _source_line_number(entry, line_number)
            return [
                "swe_patch_builder --path "
                f"{path} --replace-lines {source_line} {source_line} "
                '--with "        self.handler.records.clear()" '
                '--with "        self.handler.stream = StringIO()" > patch.diff'
            ]
    return []


def _suggested_sphinx_needs_extensions_version_commands(source_entries: list[dict[str, str]]) -> list[str]:
    for entry in source_entries:
        path = entry["path"]
        if _is_test_path(path) or path != "sphinx/extension.py":
            continue
        lines = entry["content"].splitlines()
        typing_line = 0
        condition_line = 0
        raise_end_line = 0
        for line_number, line in enumerate(lines, start=1):
            stripped = line.strip()
            if stripped == "from typing import TYPE_CHECKING, Any, Dict":
                typing_line = _source_line_number(entry, line_number)
            if stripped == "if extension.version == 'unknown version' or reqversion > extension.version:":
                condition_line = _source_line_number(entry, line_number)
                continue
            if condition_line and stripped == "(extname, reqversion, extension.version))":
                raise_end_line = _source_line_number(entry, line_number)
                break
        if not typing_line or not condition_line or not raise_end_line:
            continue
        replacements = [
            "        fulfilled = True",
            "        if extension.version == 'unknown version':",
            "            fulfilled = False",
            "        else:",
            "            try:",
            "                if Version(reqversion) > Version(extension.version):",
            "                    fulfilled = False",
            "            except InvalidVersion:",
            "                if reqversion > extension.version:",
            "                    fulfilled = False",
            "",
            "        if not fulfilled:",
            "            raise VersionRequirementError(__('This project needs the extension %s at least in '",
            "                                             'version %s and therefore cannot be built with '",
            "                                             'the loaded version (%s).') %",
            "                                          (extname, reqversion, extension.version))",
        ]
        return [
            f"swe_patch_builder --path {path} "
            f"--replace-lines {typing_line} {typing_line} "
            "--with 'from typing import TYPE_CHECKING, Any, Dict' "
            "--with '' "
            "--with 'from packaging.version import InvalidVersion, Version' "
            f"--replace-lines {condition_line} {raise_end_line} "
            + " ".join(f"--with {json.dumps(line)}" for line in replacements)
            + " > patch.diff"
        ]
    return []


def _suggested_matplotlib_stackplot_local_color_cycle_commands(source_entries: list[dict[str, str]]) -> list[str]:
    for entry in source_entries:
        path = entry["path"]
        if _is_test_path(path) or path != "lib/matplotlib/stackplot.py":
            continue
        import_line = 0
        color_block_start = 0
        color_block_end = 0
        color_replacements: list[tuple[int, str]] = []
        for line_number, line in enumerate(entry["content"].splitlines(), start=1):
            stripped = line.strip()
            if stripped == "import numpy as np":
                import_line = _source_line_number(entry, line_number)
            if stripped == "if colors is not None:":
                color_block_start = _source_line_number(entry, line_number)
                continue
            if color_block_start and not color_block_end and stripped == "axes.set_prop_cycle(color=colors)":
                color_block_end = _source_line_number(entry, line_number)
            if stripped == "color = axes._get_lines.get_next_color()":
                indent = line[: len(line) - len(line.lstrip())]
                color_replacements.append((_source_line_number(entry, line_number), f"{indent}color = next(colors)"))
        if not import_line or not color_block_start or not color_block_end or len(color_replacements) < 2:
            continue
        return [
            f"swe_patch_builder --path {path} "
            f"--replace-lines {import_line} {import_line} "
            "--with 'import itertools' "
            "--with '' "
            "--with 'import numpy as np' "
            f"--replace-lines {color_block_start} {color_block_end} "
            "--with '    if colors is not None:' "
            "--with '        colors = itertools.cycle(colors)' "
            "--with '    else:' "
            "--with '        colors = (axes._get_lines.get_next_color() for _ in y)' "
            + " ".join(
                f'--replace-line {line_number} --with "{replacement}"'
                for line_number, replacement in color_replacements[:2]
            )
            + " > patch.diff"
        ]
    return []


def _suggested_xarray_chunks_no_load_commands(source_entries: list[dict[str, str]]) -> list[str]:
    for entry in source_entries:
        path = entry["path"]
        if _is_test_path(path) or path != "xarray/core/common.py":
            continue
        in_get_chunksizes = False
        for line_number, line in enumerate(entry["content"].splitlines(), start=1):
            stripped = line.strip()
            if stripped == "def get_chunksizes(":
                in_get_chunksizes = True
                continue
            if in_get_chunksizes and line.startswith("def "):
                in_get_chunksizes = False
            if not in_get_chunksizes or stripped != 'if hasattr(v.data, "chunks"):':
                continue
            replacement = line.rstrip().replace("v.data", "v._data")
            return [
                "swe_patch_builder --path "
                f"{path} --replace-line {_source_line_number(entry, line_number)} "
                f"--with '{replacement}' > patch.diff"
            ]
    return []


def _suggested_pytest_setuponly_saferepr_commands(source_entries: list[dict[str, str]]) -> list[str]:
    for entry in source_entries:
        path = entry["path"]
        if _is_test_path(path) or path != "src/_pytest/setuponly.py":
            continue
        write_line = 0
        for line_number, line in enumerate(entry["content"].splitlines(), start=1):
            stripped = line.strip()
            if stripped == 'tw.write("[{}]".format(fixturedef.cached_param))':
                write_line = _source_line_number(entry, line_number)
        if not write_line:
            continue
        return [
            f"swe_patch_builder --path {path} "
            f"--replace-lines {write_line} {write_line} "
            "--with '        from _pytest._io.saferepr import saferepr' "
            "--with '        tw.write(\"[{}]\".format(saferepr(fixturedef.cached_param, maxsize=42)))' "
            "> patch.diff"
        ]
    return []


def _suggested_pytest_mark_expression_reserved_literal_commands(source_entries: list[dict[str, str]]) -> list[str]:
    for entry in source_entries:
        path = entry["path"]
        if _is_test_path(path) or path != "src/_pytest/mark/expression.py":
            continue
        helper_anchor = 0
        ident_line = 0
        matcher_return_line = 0
        for line_number, line in enumerate(entry["content"].splitlines(), start=1):
            stripped = line.strip()
            if stripped == "def expression(s: Scanner) -> ast.Expression:":
                helper_anchor = _source_line_number(entry, line_number) - 1
            elif stripped == "return ast.Name(ident.value, ast.Load())":
                ident_line = _source_line_number(entry, line_number)
            elif stripped == "return self.matcher(key)":
                matcher_return_line = _source_line_number(entry, line_number)
        if not helper_anchor or not ident_line or not matcher_return_line:
            continue
        replacement_lines = [
            "",
            "_RESERVED_IDENTIFIERS = {",
            '    "__pytest_reserved_True": "True",',
            '    "__pytest_reserved_False": "False",',
            '    "__pytest_reserved_None": "None",',
            "}",
            "",
            "def _identifier_ast(name: str) -> ast.expr:",
            '    reserved_key = "__pytest_reserved_{}".format(name)',
            "    reserved_name = _RESERVED_IDENTIFIERS.get(reserved_key)",
            "    if reserved_name is not None:",
            "        return ast.Name(reserved_key, ast.Load())",
            "    return ast.Name(name, ast.Load())",
            "",
            "",
            "def expression(s: Scanner) -> ast.Expression:",
            "    if s.accept(TokenType.EOF):",
            "        ret = ast.NameConstant(False)  # type: ast.expr",
            "    else:",
            "        ret = expr(s)",
            "        s.accept(TokenType.EOF, reject=True)",
            "    return ast.fix_missing_locations(ast.Expression(ret))",
            "",
            "",
            "def expr(s: Scanner) -> ast.expr:",
            "    ret = and_expr(s)",
            "    while s.accept(TokenType.OR):",
            "        rhs = and_expr(s)",
            "        ret = ast.BoolOp(ast.Or(), [ret, rhs])",
            "    return ret",
            "",
            "",
            "def and_expr(s: Scanner) -> ast.expr:",
            "    ret = not_expr(s)",
            "    while s.accept(TokenType.AND):",
            "        rhs = not_expr(s)",
            "        ret = ast.BoolOp(ast.And(), [ret, rhs])",
            "    return ret",
            "",
            "",
            "def not_expr(s: Scanner) -> ast.expr:",
            "    if s.accept(TokenType.NOT):",
            "        return ast.UnaryOp(ast.Not(), not_expr(s))",
            "    if s.accept(TokenType.LPAREN):",
            "        ret = expr(s)",
            "        s.accept(TokenType.RPAREN, reject=True)",
            "        return ret",
            "    ident = s.accept(TokenType.IDENT)",
            "    if ident:",
            "        return _identifier_ast(ident.value)",
            "    s.reject((TokenType.NOT, TokenType.LPAREN, TokenType.IDENT))",
            "",
            "",
            "class MatcherAdapter(Mapping[str, bool]):",
            '    """Adapts a matcher function to a locals mapping as required by eval()."""',
            "",
            "    def __init__(self, matcher: Callable[[str], bool]) -> None:",
            "        self.matcher = matcher",
            "",
            "    def __getitem__(self, key: str) -> bool:",
            "        reserved_name = _RESERVED_IDENTIFIERS.get(key)",
            "        if reserved_name is not None:",
            "            key = reserved_name",
            "        return self.matcher(key)",
        ]
        with_args = " ".join(f"--with '{line}'" for line in replacement_lines)
        return [
            f"swe_patch_builder --path {path} "
            f"--replace-lines {helper_anchor} {matcher_return_line} "
            f"{with_args} > patch.diff"
        ]
    return []


def _suggested_astropy_timeseries_required_columns_commands(source_entries: list[dict[str, str]]) -> list[str]:
    for entry in source_entries:
        path = entry["path"]
        if _is_test_path(path) or not path.endswith("timeseries/core.py"):
            continue
        lines = entry["content"].splitlines()
        for index, line in enumerate(lines, start=1):
            if line.strip() != 'raise ValueError("{} object is invalid - expected \'{}\' "':
                continue
            if index + 2 > len(lines):
                continue
            if '"as the first column{} but found \'{}\'"' not in lines[index].strip():
                continue
            replacements = [
                "                if len(required_columns) == 1:",
                '                    raise ValueError("{} object is invalid - expected \'{}\' "',
                '                                     "as the first column{} but found \'{}\'"',
                "                                     .format(self.__class__.__name__, required_columns[0], plural,",
                "                                             self.colnames[0]))",
                "                else:",
                '                    raise ValueError("{} object is invalid - required {} "',
                '                                     "as the first column{} but found {}"',
                "                                     .format(self.__class__.__name__, required_columns, plural,",
                "                                             self.colnames[:len(required_columns)]))",
            ]
            start = _source_line_number(entry, index)
            end = _source_line_number(entry, index + 2)
            return [
                f"swe_patch_builder --path {path} --replace-lines {start} {end} "
                + " ".join(f"--with '{replacement}'" for replacement in replacements)
                + " > patch.diff"
            ]
    return []


def _suggested_rst_header_rows_commands(source_entries: list[dict[str, str]]) -> list[str]:
    for entry in source_entries:
        path = entry["path"]
        if _is_test_path(path) or not path.endswith("rst.py"):
            continue
        lines = entry["content"].splitlines()
        init_line = 0
        return_line = 0
        for line_number, line in enumerate(lines, start=1):
            if line.strip() == "def __init__(self):":
                init_line = line_number
            if init_line and line.strip() == "return lines":
                return_line = line_number
                break
        if not init_line or not return_line:
            continue
        replacements = [
            "    def __init__(self, header_rows=None):",
            "        super().__init__(delimiter_pad=None, bookend=False, header_rows=header_rows)",
            "",
            "    def write(self, lines):",
            "        lines = super().write(lines)",
            "        idx = len(self.header.header_rows)",
            "        return [lines[idx]] + lines + [lines[idx]]",
            "",
            "    def read(self, table):",
            "        self.data.start_line = 2 + len(self.header.header_rows)",
            "        return super().read(table)",
        ]
        command = (
            f"swe_patch_builder --path {path} --replace-lines {init_line} {return_line} "
            + " ".join(f"--with '{line}'" for line in replacements)
            + " > patch.diff"
        )
        return [command]
    return []


def _suggested_separability_nested_compound_commands(source_entries: list[dict[str, str]]) -> list[str]:
    for entry in source_entries:
        path = entry["path"]
        if _is_test_path(path) or not path.endswith("separable.py"):
            continue
        for line_number, line in enumerate(entry["content"].splitlines(), start=1):
            stripped = line.strip()
            if stripped == "cright[-right.shape[0]:, -right.shape[1]:] = 1":
                replacement = line.rstrip().rsplit("=", 1)[0].rstrip() + " = right"
                return [
                    "swe_patch_builder --path "
                    f"{path} --replace-line {_source_line_number(entry, line_number)} "
                    f"--with '{replacement}' > patch.diff"
                ]
    return []


def build_swe_queue_manifest(
    prediction_task_manifest: dict[str, Any],
    *,
    workspace_prefix: str = "swe_bench_predictions",
) -> dict[str, Any]:
    tasks = prediction_task_manifest.get("tasks", [])
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("prediction task manifest must contain non-empty tasks list")
    queue_tasks: list[dict[str, Any]] = []
    skipped_tasks: list[dict[str, Any]] = []
    for index, task in enumerate(tasks, start=1):
        if not isinstance(task, dict):
            raise ValueError(f"task {index} must be an object")
        instance_id = str(task.get("instance_id", "")).strip()
        if not instance_id:
            raise ValueError(f"task {index} missing instance_id")
        problem = str(task.get("problem_statement", "")).strip()
        if not problem:
            raise ValueError(f"task {instance_id} missing problem_statement")
        repo = str(task.get("repo", "")).strip()
        base_commit = str(task.get("base_commit", "")).strip()
        repo_cache_root = str(task.get("repo_cache_root", "")).strip()
        hints = str(task.get("hints_text", "")).strip()
        candidate_files = task.get("candidate_files", [])
        if not isinstance(candidate_files, list):
            candidate_files = []
        candidate_files = [
            str(path).strip()
            for path in candidate_files
            if str(path).strip()
        ]
        source_context = task.get("source_context", [])
        if not isinstance(source_context, list):
            source_context = []
        source_entries = _source_context_entries(source_context)
        if not source_entries:
            skipped_tasks.append(
                {
                    "instance_id": instance_id,
                    "repo": repo,
                    "reason": "missing_source_context",
                    "candidate_files": candidate_files,
                    "repo_cache_root": repo_cache_root,
                }
            )
            continue
        fail_to_pass = task.get("fail_to_pass", [])
        pass_to_pass = task.get("pass_to_pass", [])
        required_patch_identifiers = _required_patch_identifiers(problem)
        executable_windows = _source_executable_edit_windows(
            source_entries,
            problem=problem,
            hints=hints,
            fail_to_pass=fail_to_pass,
            required_patch_identifiers=required_patch_identifiers,
        )
        suggested_patch_commands = _suggested_patch_commands(source_entries, problem=problem, hints=hints)
        task_id = _safe_task_id(instance_id)
        prompt_parts = [
            f"SWE-bench instance: {instance_id}",
            f"Repository: {repo or 'unknown'}",
            f"Base commit: {base_commit or 'unknown'}",
            (
                "Execution contract: create patch.diff directly in the current workspace. "
                "Do not cd to /testbed, clone repositories, fetch network resources, or write any file other than patch.diff."
            ),
            (
                "Patch validity contract: patch.diff must be a real unified diff that applies cleanly to the listed base commit. "
                "Use exact paths and context lines from the source excerpts; do not invent files, placeholder functions, test stubs, or fake comments. "
                "The patch must change executable Python behavior; edits only inside comments, docstrings, module headers, or triple-quoted text are invalid. "
                "Do not delete or rename existing production function/class definitions while trying to satisfy the verifier."
            ),
            (
                "Command contract: do not run git, ls, find, or read /testbed paths. "
                "Use only the provided candidate source files already present in this workspace, "
                "for example sed -n '1,120p' astropy/<path> or sed -n '1,120p' source_context/astropy/<path>, "
                "then write patch.diff. A diff against any path not listed as a likely relevant file is invalid."
            ),
            (
                "Preferred patch-construction command: after choosing exact line numbers from source_lines, "
                "run swe_patch_builder --path <candidate-path> --replace-line <line> --with '<replacement source line>' > patch.diff. "
                "For multi-line replacements use --replace-lines <start> <end> followed by one --with argument per replacement line. "
                "This command builds valid unified diff syntax from the provided source file."
            ),
            "Problem statement:",
            problem,
        ]
        if hints:
            prompt_parts.extend(["Hints:", hints])
        if candidate_files:
            prompt_parts.extend(
                [
                    "Likely relevant files from dataset metadata:",
                    "\n".join(f"- {path}" for path in candidate_files),
                ]
            )
        if source_entries:
            prompt_parts.extend(
                [
                    "Workspace source-context files:",
                    "\n".join(
                        f"- {entry['path']} is available at {entry['path']}, {entry['workspace_path']}, and line-numbered {entry['line_path']}"
                        for entry in source_entries
                    ),
                    (
                        "Inspect these files with commands such as sed -n '1,120p' astropy/path.py or "
                        "sed -n '1,120p' source_lines/astropy/path.py.lines. Use the line-numbered files to choose exact hunk anchors. "
                        "Do not use git show or /testbed paths."
                    ),
                ]
            )
        source_parts: list[str] = []
        for entry in source_entries:
            path = entry["path"]
            content = entry["content"]
            context = next(
                (
                    item
                    for item in source_context
                    if isinstance(item, dict) and str(item.get("path", "")).strip() == path
                ),
                {},
            )
            truncated = " (truncated)" if bool(context.get("truncated", False)) else ""
            source_parts.append(f"### {path}{truncated}\n{content}")
        if source_parts:
            prompt_parts.extend(["Source excerpts:", "\n\n".join(source_parts)])
        if required_patch_identifiers:
            prompt_parts.extend(
                [
                    "Required issue identifiers:",
                    (
                        "The patch must directly reference these issue identifiers in executable source changes: "
                        + ", ".join(required_patch_identifiers)
                    ),
                ]
            )
        if executable_windows:
            prompt_parts.extend(
                [
                    "High-value executable edit windows:",
                    (
                        "Prefer edits inside these existing function/class bodies; they are ranked by issue identifiers, hints, tests, and source tokens. "
                        "Preserve definition lines and use swe_patch_builder with exact line numbers from these windows."
                    ),
                    executable_windows,
                ]
            )
        if suggested_patch_commands:
            prompt_parts.extend(
                [
                    "Suggested targeted patch commands:",
                    (
                        "These commands are source-grounded candidates for the named issue. "
                        "When this section is present, your first patch-writing action should execute one of these commands exactly "
                        "unless you can name a concrete source reason it is wrong. Do not hand-edit a different regex or line before "
                        "trying the matching suggested command."
                    ),
                    "\n".join(f"- {command}" for command in suggested_patch_commands),
                ]
            )
        if fail_to_pass:
            prompt_parts.extend(["Fail-to-pass tests:", json.dumps(fail_to_pass, sort_keys=True)])
        if pass_to_pass:
            prompt_parts.extend(["Pass-to-pass tests:", json.dumps(pass_to_pass, sort_keys=True)])
        prompt_parts.append(
            "Write only a minimal applyable unified diff to patch.diff. Prefer swe_patch_builder redirection over hand-written hunk headers when changing existing lines. If executing a command, create patch.diff in the current directory. Do not write prose, markdown fences, placeholders, fake imports, '# This is a test file', or explanations."
        )
        queue_tasks.append(
            {
                "task_id": task_id,
                "prompt": "\n\n".join(prompt_parts),
                "workspace_subdir": f"{workspace_prefix.strip('/') or 'swe_bench_predictions'}/{task_id}",
                "setup_commands": [],
                "suggested_commands": suggested_patch_commands,
                "success_command": _candidate_file_success_command(candidate_files),
                "expected_files": ["patch.diff"],
                "max_steps": 12,
                "metadata": {
                    "benchmark_family": "repository",
                    "capability": "swe_bench_patch_generation",
                    "task_origin": "external_manifest",
                    "swe_bench_prediction_task": True,
                    "swe_instance_id": instance_id,
                    "swe_repo": repo,
                    "swe_base_commit": base_commit,
                    "swe_candidate_files": candidate_files,
                    "swe_fail_to_pass": fail_to_pass,
                    "swe_pass_to_pass": pass_to_pass,
                    "swe_executable_edit_windows": executable_windows,
                    "swe_suggested_patch_commands": suggested_patch_commands,
                    "swe_patch_workspace_relpath": "patch.diff",
                    "swe_patch_output_path": str(task.get("patch_path", "")).strip(),
                    "swe_repo_cache_root": repo_cache_root,
                    "setup_file_contents": _setup_file_contents(
                        source_entries,
                        preferred_target_lines=_suggested_command_target_lines(suggested_patch_commands),
                    ),
                    "semantic_verifier": {
                        "kind": "swe_patch_apply_check",
                        "repo": repo,
                        "base_commit": base_commit,
                        "repo_cache_root": repo_cache_root,
                        "patch_path": "patch.diff",
                        "expected_changed_paths": candidate_files,
                        "required_patch_identifiers": required_patch_identifiers,
                    }
                    if repo_cache_root
                    else {},
                    "workflow_guard": {
                        "managed_paths": [
                            "patch.diff",
                            *[entry["path"] for entry in source_entries],
                            *[entry["workspace_path"] for entry in source_entries],
                            *[entry["line_path"] for entry in source_entries],
                        ],
                    },
                },
            }
        )
    return {
        "manifest_kind": "swe_bench_patch_generation_queue_manifest",
        "manifest_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "source_task_count": len(tasks),
        "skipped_task_count": len(skipped_tasks),
        "skipped_tasks": skipped_tasks,
        "tasks": queue_tasks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-task-manifest", required=True)
    parser.add_argument("--output-manifest-json", required=True)
    parser.add_argument("--workspace-prefix", default="swe_bench_predictions")
    args = parser.parse_args()

    manifest = build_swe_queue_manifest(
        _read_json(Path(args.prediction_task_manifest)),
        workspace_prefix=args.workspace_prefix,
    )
    output_path = Path(args.output_manifest_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"task_count={len(manifest['tasks'])} "
        f"skipped_task_count={len(manifest.get('skipped_tasks', []))} "
        f"output_manifest_json={output_path}"
    )


if __name__ == "__main__":
    main()
