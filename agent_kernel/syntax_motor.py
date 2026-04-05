from __future__ import annotations

import ast
import hashlib
import importlib.util
from pathlib import Path
import sys
from typing import Any

from .schemas import TaskSpec


_REPOSITORY_LIBRARY_CODE_GRAPH = Path("/data/repository_library/scripts/code_graph.py")
_REPOSITORY_LIBRARY_CACHE: tuple[type[Any] | None, str] | None = None
_UNSET = object()
_REPOSITORY_LIBRARY_CODE_GRAPH_CLASS_CACHE: type[Any] | None | object = _UNSET
_PARSE_CACHE: dict[tuple[str, str, str], dict[str, Any]] = {}
_REPOSITORY_GRAPH_CACHE: dict[str, Any | None] = {}


def summarize_python_edit_step(
    step: dict[str, Any],
    *,
    workspace: Path | None = None,
    expected_file_contents: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    if not isinstance(step, dict):
        return None
    path = str(step.get("path", "")).strip()
    if not path or not _is_python_path(path):
        return None
    expected_contents = dict(expected_file_contents or {})
    baseline_source, baseline_origin = _resolve_baseline_source(
        step,
        workspace=workspace,
        expected_file_contents=expected_contents,
    )
    target_source, target_origin = _resolve_target_source(
        step,
        workspace=workspace,
        expected_file_contents=expected_contents,
        baseline_source=baseline_source,
    )
    edited_region = _edited_region(step)
    baseline_parse = _parse_python_source(baseline_source, label=f"{path}:baseline")
    target_parse = _parse_python_source(target_source, label=f"{path}:target")
    baseline_symbol = _target_symbol_from_parse(baseline_parse, edited_region)
    target_symbol = _target_symbol_from_parse(target_parse, edited_region)
    primary_symbol = target_symbol or baseline_symbol
    imports_before = baseline_parse.get("imports", [])
    imports_after = target_parse.get("imports", [])
    imports_added = [item for item in imports_after if item not in set(imports_before)]
    imports_removed = [item for item in imports_before if item not in set(imports_after)]
    call_targets_before = _call_targets_for_symbol(
        baseline_parse.get("calls", []),
        baseline_symbol,
    )
    call_targets_after = _call_targets_for_symbol(
        target_parse.get("calls", []),
        target_symbol or baseline_symbol,
    )
    signature_changed = (
        isinstance(baseline_symbol, dict)
        and isinstance(target_symbol, dict)
        and str(baseline_symbol.get("signature", "")).strip()
        != str(target_symbol.get("signature", "")).strip()
    )
    repository_context = _repository_context_for_symbol(
        workspace=workspace,
        path=path,
        symbol=primary_symbol,
    )
    return {
        "path": path,
        "edit_kind": str(step.get("edit_kind", "rewrite")).strip() or "rewrite",
        "edited_region": edited_region,
        "baseline_source_origin": baseline_origin,
        "target_source_origin": target_origin,
        "baseline_syntax_valid": baseline_parse["error"] is None if baseline_source is not None else None,
        "target_syntax_valid": target_parse["error"] is None if target_source is not None else None,
        "baseline_target_symbol": baseline_symbol,
        "target_target_symbol": target_symbol,
        "edited_symbol_fqn": str(primary_symbol.get("fqn", "")).strip() or None
        if isinstance(primary_symbol, dict)
        else None,
        "enclosing_symbol_qualname": str(primary_symbol.get("qualname", "")).strip() or None
        if isinstance(primary_symbol, dict)
        else None,
        "imports_before": imports_before,
        "imports_after": imports_after,
        "imports_added": imports_added,
        "imports_removed": imports_removed,
        "exports_before": list(baseline_parse.get("exports", [])),
        "exports_after": list(target_parse.get("exports", [])),
        "star_imports_before": list(baseline_parse.get("star_imports", [])),
        "star_imports_after": list(target_parse.get("star_imports", [])),
        "call_targets_before": call_targets_before,
        "call_targets_after": call_targets_after,
        "signature_changed": signature_changed,
        "signature_change_risk": bool(signature_changed and call_targets_before),
        "import_change_risk": bool(imports_added or imports_removed),
        "module_symbols_before": baseline_parse.get("symbols", []),
        "module_symbols_after": target_parse.get("symbols", []),
        "syntax_ok": baseline_parse["error"] is None and target_parse["error"] is None,
        "parser": target_parse.get("parser", baseline_parse.get("parser", _python_parser_name())),
        "provider": target_parse.get("provider", baseline_parse.get("provider", "python_ast")),
        "repository_context": repository_context,
    }


def build_syntax_motor_summary(
    task: TaskSpec,
    *,
    workspace: Path | None = None,
    task_contract: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    contract = dict(task_contract or {})
    metadata = contract.get("metadata", {}) if isinstance(contract.get("metadata", {}), dict) else {}
    if not metadata:
        metadata = dict(getattr(task, "metadata", {}) or {})
    edit_plan = contract.get("synthetic_edit_plan", [])
    if not isinstance(edit_plan, list) or not edit_plan:
        edit_plan = metadata.get("synthetic_edit_plan", []) if isinstance(metadata, dict) else []
    python_steps = [dict(step) for step in edit_plan if isinstance(step, dict) and _is_python_path(step.get("path"))]
    if not python_steps:
        return None

    expected_file_contents = (
        dict(contract.get("expected_file_contents", {}))
        if isinstance(contract.get("expected_file_contents", {}), dict)
        else dict(getattr(task, "expected_file_contents", {}) or {})
    )
    files: list[dict[str, Any]] = []
    issues: list[str] = []
    targeted_symbol_count = 0
    target_syntax_failures = 0
    baseline_syntax_failures = 0
    for step in python_steps:
        file_summary = summarize_python_edit_step(
            step,
            workspace=workspace,
            expected_file_contents=expected_file_contents,
        )
        if file_summary is None:
            continue
        path = str(file_summary.get("path", "")).strip()
        if bool(file_summary.get("baseline_syntax_valid")) is False:
            baseline_syntax_failures += 1
            issues.append(f"baseline_syntax_error:{path}")
        if bool(file_summary.get("target_syntax_valid")) is False:
            target_syntax_failures += 1
            issues.append(f"target_syntax_error:{path}")
        if file_summary.get("baseline_target_symbol") or file_summary.get("target_target_symbol"):
            targeted_symbol_count += 1
        files.append(file_summary)
    if not files:
        return None
    return {
        "language": "python",
        "parser": _python_parser_name(),
        "target_file_count": len(files),
        "targeted_symbol_count": targeted_symbol_count,
        "baseline_syntax_failures": baseline_syntax_failures,
        "target_syntax_failures": target_syntax_failures,
        "syntax_preflight_passed": baseline_syntax_failures == 0 and target_syntax_failures == 0,
        "syntax_ok": baseline_syntax_failures == 0 and target_syntax_failures == 0,
        "files": files,
        "issues": issues,
    }


def syntax_preflight_check(task: TaskSpec, *, workspace: Path | None = None) -> tuple[bool, str, str]:
    summary = build_syntax_motor_summary(task, workspace=workspace)
    if summary is None:
        return True, "no_python_edit_targets", "no python synthetic edit targets to analyze"
    if bool(summary.get("syntax_preflight_passed", False)):
        targeted = int(summary.get("targeted_symbol_count", 0) or 0)
        count = int(summary.get("target_file_count", 0) or 0)
        return True, "syntax_ready", f"python syntax preflight passed for {count} file(s); targeted_symbols={targeted}"
    issues = ", ".join(str(issue) for issue in summary.get("issues", [])[:3])
    return False, "syntax_error", f"python syntax preflight found issues: {issues}"


def _is_python_path(path: object) -> bool:
    return str(path or "").strip().endswith(".py")


def _resolve_baseline_source(
    step: dict[str, Any],
    *,
    workspace: Path | None,
    expected_file_contents: dict[str, str],
) -> tuple[str | None, str]:
    path = str(step.get("path", "")).strip()
    if workspace is not None and path:
        candidate = workspace / path
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8"), "workspace"
    baseline_content = step.get("baseline_content")
    if isinstance(baseline_content, str) and baseline_content:
        return baseline_content, "baseline_content"
    expected = expected_file_contents.get(path)
    if isinstance(expected, str) and expected:
        return expected, "expected_file_contents"
    return None, "unavailable"


def _resolve_target_source(
    step: dict[str, Any],
    *,
    workspace: Path | None,
    expected_file_contents: dict[str, str],
    baseline_source: str | None,
) -> tuple[str | None, str]:
    target_content = step.get("target_content")
    if isinstance(target_content, str) and target_content:
        return target_content, "target_content"
    path = str(step.get("path", "")).strip()
    expected = expected_file_contents.get(path)
    if isinstance(expected, str) and expected:
        return expected, "expected_file_contents"
    if workspace is not None and path:
        candidate = workspace / path
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8"), "workspace"
    return baseline_source, "baseline_fallback" if baseline_source is not None else "unavailable"


def _edited_region(step: dict[str, Any]) -> dict[str, Any]:
    edit_kind = str(step.get("edit_kind", "rewrite")).strip() or "rewrite"
    line_numbers: list[int] = []
    replacements = step.get("replacements", [])
    if isinstance(replacements, list):
        for replacement in replacements:
            if not isinstance(replacement, dict):
                continue
            try:
                line_number = int(replacement.get("line_number", 0) or 0)
            except (TypeError, ValueError):
                line_number = 0
            if line_number > 0:
                line_numbers.append(line_number)
    insertion = step.get("insertion", {})
    if isinstance(insertion, dict):
        try:
            line_number = int(insertion.get("line_number", 0) or 0)
        except (TypeError, ValueError):
            line_number = 0
        if line_number > 0:
            line_numbers.append(line_number)
    start_line = min(line_numbers) if line_numbers else None
    end_line = max(line_numbers) if line_numbers else None
    if edit_kind == "rewrite":
        start_line = 1
        target_content = step.get("target_content")
        baseline_content = step.get("baseline_content")
        source = target_content if isinstance(target_content, str) and target_content else baseline_content
        if isinstance(source, str) and source:
            end_line = max(1, len(source.splitlines()))
    elif edit_kind == "line_insert" and isinstance(insertion, dict):
        inserted_lines = insertion.get("inserted_lines", [])
        count = len(inserted_lines) if isinstance(inserted_lines, list) and inserted_lines else 1
        if start_line is not None:
            end_line = start_line + max(0, count - 1)
    return {
        "start_line": start_line,
        "end_line": end_line if end_line is not None else start_line,
    }


def _parse_python_source(source: str | None, *, label: str) -> dict[str, Any]:
    if source is None:
        return {
            "tree": None,
            "imports": [],
            "symbols": [],
            "calls": [],
            "exports": [],
            "star_imports": [],
            "provider": "python_ast",
            "error": None,
        }
    module_name = _module_name_from_label(label)
    visitor_class, parser_name = _repository_library_module_visitor_class()
    provider_name = "repository_library" if visitor_class is not None else "python_ast"
    cache_key = (
        provider_name,
        module_name,
        hashlib.sha256(source.encode("utf-8")).hexdigest(),
    )
    cached = _PARSE_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)
    try:
        tree = ast.parse(source, filename=label)
    except SyntaxError as exc:
        return {
            "tree": None,
            "imports": [],
            "symbols": [],
            "calls": [],
            "exports": [],
            "star_imports": [],
            "provider": provider_name,
            "error": f"{exc.msg} line {exc.lineno}",
        }
    visitor = (
        visitor_class(module_name, label)
        if visitor_class is not None
        else _PythonModuleVisitor(module=module_name)
    )
    visitor.visit(tree)
    payload = {
        "tree": tree,
        "imports": _normalize_visitor_imports(visitor)[:8],
        "symbols": _normalize_visitor_symbols(visitor)[:16],
        "calls": _normalize_visitor_calls(visitor)[:32],
        "exports": _normalize_visitor_exports(visitor)[:16],
        "star_imports": _normalize_visitor_star_imports(visitor)[:8],
        "parser": parser_name,
        "provider": provider_name,
        "error": None,
    }
    _PARSE_CACHE[cache_key] = dict(payload)
    return payload


def _python_parser_name() -> str:
    return _repository_library_module_visitor_class()[1]


def _repository_library_module_visitor_class() -> tuple[type[Any] | None, str]:
    global _REPOSITORY_LIBRARY_CACHE
    if _REPOSITORY_LIBRARY_CACHE is not None:
        return _REPOSITORY_LIBRARY_CACHE
    if not _REPOSITORY_LIBRARY_CODE_GRAPH.is_file():
        _REPOSITORY_LIBRARY_CACHE = (None, "python_ast")
        return _REPOSITORY_LIBRARY_CACHE
    try:
        module_name = "_agentkernel_repository_library_code_graph"
        module = sys.modules.get(module_name)
        if module is None:
            spec = importlib.util.spec_from_file_location(module_name, _REPOSITORY_LIBRARY_CODE_GRAPH)
            if spec is None or spec.loader is None:
                raise ImportError("unable to load repository_library code_graph spec")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        visitor_class = getattr(module, "_ModuleVisitor", None)
        if not isinstance(visitor_class, type):
            raise AttributeError("repository_library _ModuleVisitor missing")
        _REPOSITORY_LIBRARY_CACHE = (visitor_class, "repository_library_python_ast")
    except Exception:
        _REPOSITORY_LIBRARY_CACHE = (None, "python_ast")
    return _REPOSITORY_LIBRARY_CACHE


def _normalize_visitor_imports(visitor: Any) -> list[str]:
    imports = getattr(visitor, "imports", {})
    modules = getattr(visitor, "import_modules", [])
    normalized: list[str] = []
    if isinstance(imports, dict):
        seen_targets: set[str] = set()
        for target in imports.values():
            value = str(target).strip()
            if value and value not in seen_targets:
                normalized.append(value)
                seen_targets.add(value)
    if isinstance(modules, list):
        for item in modules:
            value = str(item).strip()
            if value and value not in normalized:
                normalized.append(value)
    return normalized


def _normalize_visitor_symbols(visitor: Any) -> list[dict[str, Any]]:
    symbols = getattr(visitor, "symbols", [])
    normalized: list[dict[str, Any]] = []
    if not isinstance(symbols, list):
        return normalized
    for item in symbols:
        if isinstance(item, dict):
            payload = dict(item)
        else:
            payload = {
                "kind": str(getattr(item, "kind", "")).strip(),
                "fqn": str(getattr(item, "fqn", "")).strip(),
                "name": str(getattr(item, "name", "")).strip(),
                "qualname": str(getattr(item, "qualname", "")).strip(),
                "signature": str(getattr(item, "signature", "")).strip() or None,
                "returns": str(getattr(item, "returns", "")).strip() or None,
                "start_line": int(getattr(item, "line", 0) or 0),
                "end_line": int(getattr(item, "end_line", 0) or 0),
            }
        normalized.append(
            {
                "kind": str(payload.get("kind", "")).strip(),
                "fqn": str(payload.get("fqn", "")).strip(),
                "name": str(payload.get("name", "")).strip(),
                "qualname": str(payload.get("qualname", "")).strip(),
                "signature": _normalize_symbol_signature(
                    str(payload.get("name", "")).strip(),
                    str(payload.get("signature", "")).strip() or None,
                ),
                "returns": str(payload.get("returns", "")).strip() or None,
                "start_line": int(payload.get("start_line", 0) or 0),
                "end_line": int(payload.get("end_line", 0) or 0),
            }
        )
    return normalized


def _normalize_visitor_calls(visitor: Any) -> list[dict[str, str]]:
    calls = getattr(visitor, "calls", [])
    normalized: list[dict[str, str]] = []
    if not isinstance(calls, list):
        return normalized
    for item in calls:
        if not isinstance(item, tuple) or len(item) != 2:
            continue
        caller = str(item[0]).strip()
        callee = str(item[1]).strip()
        if not caller or not callee:
            continue
        normalized.append({"caller_fqn": caller, "callee": callee})
    return normalized


def _normalize_visitor_exports(visitor: Any) -> list[str]:
    exports = getattr(visitor, "exports", [])
    if not isinstance(exports, list):
        return []
    normalized: list[str] = []
    for item in exports:
        value = str(item).strip()
        if value and value not in normalized:
            normalized.append(value)
    return normalized


def _normalize_visitor_star_imports(visitor: Any) -> list[str]:
    star_imports = getattr(visitor, "star_imports", [])
    if not isinstance(star_imports, list):
        return []
    normalized: list[str] = []
    for item in star_imports:
        value = str(item).strip()
        if value and value not in normalized:
            normalized.append(value)
    return normalized


def _normalize_symbol_signature(name: str, signature: str | None) -> str | None:
    normalized_name = str(name).strip()
    normalized_signature = str(signature or "").strip()
    if not normalized_signature:
        return None
    if normalized_name and normalized_signature.startswith("("):
        return f"{normalized_name}{normalized_signature}"
    return normalized_signature


class _PythonModuleVisitor(ast.NodeVisitor):
    def __init__(self, *, module: str) -> None:
        self.module = module
        self.imports: dict[str, str] = {}
        self.import_modules: list[str] = []
        self.symbols: list[dict[str, Any]] = []
        self.calls: list[tuple[str, str]] = []
        self._stack: list[str] = []
        self._class_stack: list[str] = []

    def visit_Import(self, node: ast.Import) -> Any:  # type: ignore[override]
        for alias in node.names:
            asname = alias.asname or alias.name.split(".")[-1]
            self.imports[asname] = alias.name
            self.import_modules.append(alias.name.split(".")[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:  # type: ignore[override]
        module = str(node.module or "").strip()
        for alias in node.names:
            asname = alias.asname or alias.name
            target = f"{module}.{alias.name}" if module else alias.name
            self.imports[asname] = target
        if module:
            self.import_modules.append(module.split(".")[0])

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:  # type: ignore[override]
        self.symbols.append(
            {
                "kind": "class",
                "fqn": self._fqn(node.name),
                "name": node.name,
                "qualname": self._qualname(),
                "start_line": int(getattr(node, "lineno", 0) or 0),
                "end_line": int(getattr(node, "end_lineno", 0) or 0),
            }
        )
        self._stack.append(node.name)
        self._class_stack.append(node.name)
        self.generic_visit(node)
        self._class_stack.pop()
        self._stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:  # type: ignore[override]
        self._visit_function_like(node, kind="function")

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:  # type: ignore[override]
        self._visit_function_like(node, kind="async_function")

    def _visit_function_like(self, node: ast.FunctionDef | ast.AsyncFunctionDef, *, kind: str) -> None:
        payload = {
            "kind": kind,
            "fqn": self._fqn(node.name),
            "name": node.name,
            "qualname": self._qualname(),
            "signature": _function_signature(node),
            "start_line": int(getattr(node, "lineno", 0) or 0),
            "end_line": int(getattr(node, "end_lineno", 0) or 0),
        }
        returns = getattr(node, "returns", None)
        if returns is not None:
            try:
                payload["returns"] = ast.unparse(returns)
            except Exception:
                payload["returns"] = None
        self.symbols.append(payload)
        self._stack.append(node.name)
        fqn = self._fqn("")
        caller_fqn = fqn[:-1] if fqn.endswith(".") else fqn
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                callee_key = self._extract_callee_key(sub.func)
                if callee_key:
                    self.calls.append((caller_fqn, callee_key))
        self.generic_visit(node)
        self._stack.pop()

    def _qualname(self) -> str:
        if not self._stack:
            return ""
        return ".".join(self._stack)

    def _fqn(self, name: str) -> str:
        qualname = self._qualname()
        if qualname and name:
            return f"{self.module}.{qualname}.{name}"
        if qualname:
            return f"{self.module}.{qualname}"
        if name:
            return f"{self.module}.{name}"
        return self.module

    def _extract_callee_key(self, fn: ast.AST) -> str | None:
        if isinstance(fn, ast.Name):
            return fn.id
        if (
            isinstance(fn, ast.Attribute)
            and isinstance(fn.value, ast.Call)
            and isinstance(fn.value.func, ast.Name)
            and fn.value.func.id == "super"
        ):
            current_class = self._class_stack[-1] if self._class_stack else ""
            return f"{self.module}.{current_class}.{fn.attr}" if current_class else fn.attr
        if isinstance(fn, ast.Attribute):
            parts: list[str] = []
            current: ast.AST | None = fn
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
                parts.reverse()
                return ".".join(parts)
        return None


def _target_symbol_from_symbols(symbols: list[dict[str, Any]], region: dict[str, Any]) -> dict[str, Any] | None:
    start_line = region.get("start_line")
    if not isinstance(start_line, int) or start_line <= 0:
        return None
    best: dict[str, Any] | None = None
    best_span: int | None = None
    for symbol in symbols:
        if not isinstance(symbol, dict):
            continue
        if str(symbol.get("kind", "")).strip() not in {"function", "async_function", "class"}:
            continue
        node_start = symbol.get("start_line")
        node_end = symbol.get("end_line")
        if not isinstance(node_start, int) or not isinstance(node_end, int):
            continue
        if not (node_start <= start_line <= node_end):
            continue
        span = node_end - node_start
        if best is None or best_span is None or span < best_span:
            best = symbol
            best_span = span
    if best is None:
        return None
    payload = {
        "kind": str(best.get("kind", "")).strip(),
        "fqn": str(best.get("fqn", "")).strip() or None,
        "name": str(best.get("name", "")).strip(),
        "qualname": str(best.get("qualname", "")).strip() or None,
        "start_line": int(best.get("start_line", 0) or 0),
        "end_line": int(best.get("end_line", 0) or 0),
    }
    signature = str(best.get("signature", "")).strip()
    if signature:
        payload["signature"] = signature
    return payload


def _target_symbol_from_tree(tree: ast.AST | None, region: dict[str, Any]) -> dict[str, Any] | None:
    if tree is None:
        return None
    start_line = region.get("start_line")
    if not isinstance(start_line, int) or start_line <= 0:
        return None
    best: ast.AST | None = None
    best_span: int | None = None
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        node_start = getattr(node, "lineno", None)
        node_end = getattr(node, "end_lineno", None)
        if not isinstance(node_start, int) or not isinstance(node_end, int):
            continue
        if not (node_start <= start_line <= node_end):
            continue
        span = node_end - node_start
        if best is None or best_span is None or span < best_span:
            best = node
            best_span = span
    if best is None:
        return None
    payload = {
        "kind": "class" if isinstance(best, ast.ClassDef) else "function",
        "name": str(getattr(best, "name", "")).strip(),
        "start_line": int(getattr(best, "lineno", 0) or 0),
        "end_line": int(getattr(best, "end_lineno", 0) or 0),
    }
    if isinstance(best, (ast.FunctionDef, ast.AsyncFunctionDef)):
        payload["signature"] = _function_signature(best)
    return payload


def _target_symbol_from_parse(parsed: dict[str, Any], region: dict[str, Any]) -> dict[str, Any] | None:
    tree_symbol = _target_symbol_from_tree(parsed.get("tree"), region)
    line_symbol = _target_symbol_from_symbols(parsed.get("symbols", []), region)
    if tree_symbol is None:
        return line_symbol
    if line_symbol is None:
        return tree_symbol
    merged = dict(line_symbol)
    merged.update(
        {
            "kind": tree_symbol.get("kind", merged.get("kind")),
            "name": tree_symbol.get("name", merged.get("name")),
            "start_line": tree_symbol.get("start_line", merged.get("start_line")),
            "end_line": tree_symbol.get("end_line", merged.get("end_line")),
        }
    )
    if tree_symbol.get("signature"):
        merged["signature"] = tree_symbol.get("signature")
    return merged


def _call_targets_for_symbol(calls: list[dict[str, str]], symbol: dict[str, Any] | None) -> list[str]:
    if not isinstance(symbol, dict):
        return []
    caller_fqn = str(symbol.get("fqn", "")).strip()
    if not caller_fqn:
        return []
    targets: list[str] = []
    for item in calls:
        if not isinstance(item, dict):
            continue
        if str(item.get("caller_fqn", "")).strip() != caller_fqn:
            continue
        callee = str(item.get("callee", "")).strip()
        if callee and callee not in targets:
            targets.append(callee)
    return targets


def _repository_context_for_symbol(
    *,
    workspace: Path | None,
    path: str,
    symbol: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if workspace is None or not isinstance(symbol, dict):
        return None
    graph = _repository_library_code_graph_for_root(workspace)
    if graph is None:
        return None
    symbol_fqn = str(symbol.get("fqn", "")).strip()
    symbol_name = str(symbol.get("name", "")).strip()
    relative_path = str(path).strip()
    try:
        owner_files = graph.owners_of(symbol_name) if symbol_name else []
    except Exception:
        owner_files = []
    try:
        repo_callers = graph.who_calls(symbol_fqn) if symbol_fqn else []
    except Exception:
        repo_callers = []
    try:
        repo_callees = graph.calls_of(symbol_fqn) if symbol_fqn else []
    except Exception:
        repo_callees = []
    owner_matches_path = bool(relative_path and relative_path in set(owner_files))
    if not owner_files and relative_path:
        owner_files = [relative_path]
        owner_matches_path = True
    return {
        "available": True,
        "provider": "repository_library_code_graph",
        "owner_files": list(owner_files)[:8],
        "owner_matches_path": owner_matches_path,
        "repo_callers": list(repo_callers)[:8],
        "repo_callees": list(repo_callees)[:8],
    }


def _repository_library_code_graph_class() -> type[Any] | None:
    global _REPOSITORY_LIBRARY_CODE_GRAPH_CLASS_CACHE
    if _REPOSITORY_LIBRARY_CODE_GRAPH_CLASS_CACHE is not _UNSET:
        return _REPOSITORY_LIBRARY_CODE_GRAPH_CLASS_CACHE  # type: ignore[return-value]
    if not _REPOSITORY_LIBRARY_CODE_GRAPH.is_file():
        _REPOSITORY_LIBRARY_CODE_GRAPH_CLASS_CACHE = None
        return None
    try:
        module_name = "_agentkernel_repository_library_code_graph"
        module = sys.modules.get(module_name)
        if module is None:
            spec = importlib.util.spec_from_file_location(module_name, _REPOSITORY_LIBRARY_CODE_GRAPH)
            if spec is None or spec.loader is None:
                raise ImportError("unable to load repository_library code_graph spec")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        code_graph_class = getattr(module, "CodeGraph", None)
        if not isinstance(code_graph_class, type):
            raise AttributeError("repository_library CodeGraph missing")
        _REPOSITORY_LIBRARY_CODE_GRAPH_CLASS_CACHE = code_graph_class
    except Exception:
        _REPOSITORY_LIBRARY_CODE_GRAPH_CLASS_CACHE = None
    return _REPOSITORY_LIBRARY_CODE_GRAPH_CLASS_CACHE  # type: ignore[return-value]


def _repository_library_code_graph_for_root(workspace: Path) -> Any | None:
    root = str(Path(workspace).resolve())
    if root in _REPOSITORY_GRAPH_CACHE:
        return _REPOSITORY_GRAPH_CACHE[root]
    code_graph_class = _repository_library_code_graph_class()
    if code_graph_class is None:
        _REPOSITORY_GRAPH_CACHE[root] = None
        return None
    try:
        graph = code_graph_class.load_or_build(root)
    except Exception:
        graph = None
    _REPOSITORY_GRAPH_CACHE[root] = graph
    return graph


def _module_name_from_label(label: str) -> str:
    path = str(label).split(":", 1)[0]
    candidate = Path(path)
    if candidate.suffix == ".py":
        candidate = candidate.with_suffix("")
    parts = [part for part in candidate.parts if part not in {"", "."}]
    return ".".join(parts) or "__syntax_motor__"


def _function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    positional = [arg.arg for arg in node.args.args]
    kwonly = [arg.arg for arg in node.args.kwonlyargs]
    parts: list[str] = []
    if node.args.posonlyargs:
        parts.extend(arg.arg for arg in node.args.posonlyargs)
        parts.append("/")
    parts.extend(positional)
    if node.args.vararg is not None:
        parts.append(f"*{node.args.vararg.arg}")
    elif kwonly:
        parts.append("*")
    parts.extend(kwonly)
    if node.args.kwarg is not None:
        parts.append(f"**{node.args.kwarg.arg}")
    return f"{node.name}({', '.join(parts)})"
