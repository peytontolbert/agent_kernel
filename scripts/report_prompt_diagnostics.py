#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "agent_kernel"


def _iter_python_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.py") if path.is_file())


def _count_python_loc(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            count += 1
    return count


def _parse_module(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _ast_name(node: ast.AST | None) -> str:
    if node is None:
        return ""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _ast_name(node.value)
        return f"{base}.{node.attr}".strip(".")
    if isinstance(node, ast.Constant):
        return str(node.value)
    if isinstance(node, ast.Call):
        return _ast_name(node.func)
    if isinstance(node, ast.Subscript):
        return _ast_name(node.value)
    return ast.dump(node, include_attributes=False)


def _constant_string(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _keyword_map(call: ast.Call) -> dict[str, ast.AST]:
    return {
        keyword.arg: keyword.value
        for keyword in call.keywords
        if keyword.arg is not None
    }


def collect_python_stats(root: Path) -> dict[str, Any]:
    files = _iter_python_files(root)
    totals = Counter()
    per_file: list[dict[str, Any]] = []

    for path in files:
        tree = _parse_module(path)
        nodes = list(ast.walk(tree))
        counts = Counter(type(node).__name__ for node in nodes)
        totals.update(counts)
        per_file.append(
            {
                "path": str(path.relative_to(REPO_ROOT)),
                "python_loc": _count_python_loc(path),
                "functions": counts.get("FunctionDef", 0) + counts.get("AsyncFunctionDef", 0),
                "classes": counts.get("ClassDef", 0),
                "imports": counts.get("Import", 0) + counts.get("ImportFrom", 0),
                "ast_nodes": len(nodes),
            }
        )

    return {
        "root": str(root),
        "python_files": len(files),
        "python_loc": sum(item["python_loc"] for item in per_file),
        "ast_totals": {
            "functions": totals.get("FunctionDef", 0) + totals.get("AsyncFunctionDef", 0),
            "classes": totals.get("ClassDef", 0),
            "imports": totals.get("Import", 0) + totals.get("ImportFrom", 0),
            "calls": totals.get("Call", 0),
            "assignments": totals.get("Assign", 0) + totals.get("AnnAssign", 0),
            "branches": totals.get("If", 0) + totals.get("Match", 0),
        },
        "largest_files_by_loc": sorted(per_file, key=lambda item: (-item["python_loc"], item["path"]))[:15],
        "largest_files_by_ast_nodes": sorted(per_file, key=lambda item: (-item["ast_nodes"], item["path"]))[:15],
    }


class PromptDiagnosticsCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.context_payload_keys: list[dict[str, Any]] = []
        self.payload_runtime_additions: list[dict[str, Any]] = []
        self.prompt_file_reads: list[dict[str, Any]] = []
        self.prompt_call_arguments: list[dict[str, Any]] = []
        self.decision_prompt_mutations: list[dict[str, Any]] = []
        self.system_prompt_mutations: list[dict[str, Any]] = []
        self.function_returns: list[dict[str, Any]] = []

        self._class_stack: list[str] = []
        self._function_stack: list[str] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._class_stack.append(node.name)
        self.generic_visit(node)
        self._class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._function_stack.append(node.name)
        if self._class_stack == ["ContextBudgeter"] and node.name == "build_payload":
            self._collect_context_budget_payload(node)
        if self._class_stack == ["LLMDecisionPolicy"] and node.name == "decide":
            self._collect_decide_details(node)
        if self._class_stack == ["LLMDecisionPolicy"] and node.name in {
            "_prompt_policy_payload",
            "_active_prompt_adjustments",
            "_role_system_prompt",
            "_role_decision_prompt",
            "_role_directive",
            "_role_directive_overrides",
            "_software_work_phase_gate_brief",
            "_campaign_contract_brief",
        }:
            self._collect_function_returns(node)
        self.generic_visit(node)
        self._function_stack.pop()

    def visit_Assign(self, node: ast.Assign) -> None:
        if self._class_stack == ["LLMDecisionPolicy"] and self._function_stack == ["__init__"]:
            self._collect_prompt_file_reads(node)
        self.generic_visit(node)

    def _collect_context_budget_payload(self, node: ast.FunctionDef) -> None:
        for child in ast.walk(node):
            if not isinstance(child, ast.Assign):
                continue
            if len(child.targets) != 1 or not isinstance(child.targets[0], ast.Name):
                continue
            if child.targets[0].id != "payload" or not isinstance(child.value, ast.Dict):
                continue
            for key_node, value_node in zip(child.value.keys, child.value.values):
                key = _constant_string(key_node)
                if not key:
                    continue
                self.context_payload_keys.append(
                    {
                        "key": key,
                        "source_expr": _ast_name(value_node),
                        "line": child.lineno,
                    }
                )

    def _collect_prompt_file_reads(self, node: ast.Assign) -> None:
        value = node.value
        if not isinstance(value, ast.Call):
            return
        if not isinstance(value.func, ast.Attribute) or value.func.attr != "read_text":
            return
        joinpath_call = value.func.value
        if not isinstance(joinpath_call, ast.Call):
            return
        if not isinstance(joinpath_call.func, ast.Attribute) or joinpath_call.func.attr != "joinpath":
            return
        if _ast_name(joinpath_call.func.value) != "prompts_dir":
            return
        if not joinpath_call.args:
            return
        filename = _constant_string(joinpath_call.args[0])
        if not filename:
            return
        for target in node.targets:
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                self.prompt_file_reads.append(
                    {
                        "target": target.attr,
                        "file": filename,
                        "line": node.lineno,
                    }
                )

    def _collect_decide_details(self, node: ast.FunctionDef) -> None:
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and _ast_name(child.func) == "self.context_budgeter.build_payload":
                for name, value in _keyword_map(child).items():
                    self.prompt_call_arguments.append(
                        {
                            "argument": name,
                            "source_expr": _ast_name(value),
                            "line": child.lineno,
                        }
                    )
            if isinstance(child, ast.Call) and _ast_name(child.func) == "self.client.create_decision":
                for name, value in _keyword_map(child).items():
                    self.prompt_call_arguments.append(
                        {
                            "argument": f"create_decision.{name}",
                            "source_expr": _ast_name(value),
                            "line": child.lineno,
                        }
                    )
            if isinstance(child, ast.Assign):
                self._collect_payload_assignment(child)
                self._collect_prompt_mutation(child)
            if isinstance(child, ast.AugAssign):
                self._collect_augmented_prompt_mutation(child)

    def _collect_payload_assignment(self, node: ast.Assign) -> None:
        for target in node.targets:
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == "payload"
            ):
                key = _constant_string(target.slice)
                if not key:
                    continue
                self.payload_runtime_additions.append(
                    {
                        "key": key,
                        "source_expr": _ast_name(node.value),
                        "line": node.lineno,
                    }
                )

    def _collect_prompt_mutation(self, node: ast.Assign) -> None:
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            return
        target = node.targets[0].id
        if target not in {"decision_prompt", "system_prompt"}:
            return
        entry = {
            "target": target,
            "source_expr": _ast_name(node.value),
            "line": node.lineno,
        }
        if target == "decision_prompt":
            self.decision_prompt_mutations.append(entry)
        else:
            self.system_prompt_mutations.append(entry)

    def _collect_augmented_prompt_mutation(self, node: ast.AugAssign) -> None:
        if not isinstance(node.target, ast.Name):
            return
        target = node.target.id
        if target not in {"decision_prompt", "system_prompt"}:
            return
        entry = {
            "target": target,
            "source_expr": _ast_name(node.value),
            "line": node.lineno,
        }
        if target == "decision_prompt":
            self.decision_prompt_mutations.append(entry)
        else:
            self.system_prompt_mutations.append(entry)

    def _collect_function_returns(self, node: ast.FunctionDef) -> None:
        for child in ast.walk(node):
            if isinstance(child, ast.Return):
                self.function_returns.append(
                    {
                        "function": node.name,
                        "expr": _ast_name(child.value),
                        "line": child.lineno,
                    }
                )


class LLMRenderCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.provider_payload_edges: list[dict[str, Any]] = []
        self.render_prompt_edges: list[dict[str, Any]] = []
        self.compact_payload_keys: list[dict[str, Any]] = []
        self.minimal_payload_mutations: list[dict[str, Any]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.name == "_render_prompt":
            self._collect_render_prompt(node)
        elif node.name == "_compact_state_payload":
            self._collect_compact_payload(node)
        elif node.name == "_minimal_state_payload":
            self._collect_minimal_payload(node)
        elif node.name in {"_generate", "_chat_completion"}:
            self._collect_provider_payload(node)
        self.generic_visit(node)

    def _collect_render_prompt(self, node: ast.FunctionDef) -> None:
        for child in ast.walk(node):
            if isinstance(child, ast.Return):
                self.render_prompt_edges.append(
                    {
                        "function": node.name,
                        "expr": _ast_name(child.value),
                        "line": child.lineno,
                    }
                )

    def _collect_compact_payload(self, node: ast.FunctionDef) -> None:
        for child in ast.walk(node):
            if isinstance(child, ast.Return) and isinstance(child.value, ast.Dict):
                for key_node, value_node in zip(child.value.keys, child.value.values):
                    key = _constant_string(key_node)
                    if key:
                        self.compact_payload_keys.append(
                            {
                                "key": key,
                                "source_expr": _ast_name(value_node),
                                "line": child.lineno,
                            }
                        )

    def _collect_minimal_payload(self, node: ast.FunctionDef) -> None:
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Assign)
                and len(child.targets) == 1
                and isinstance(child.targets[0], ast.Subscript)
                and isinstance(child.targets[0].value, ast.Name)
                and child.targets[0].value.id == "compact"
            ):
                key = _constant_string(child.targets[0].slice)
                if key:
                    self.minimal_payload_mutations.append(
                        {
                            "key": key,
                            "source_expr": _ast_name(child.value),
                            "line": child.lineno,
                        }
                    )

    def _collect_provider_payload(self, node: ast.FunctionDef) -> None:
        for child in ast.walk(node):
            if not isinstance(child, ast.Assign):
                continue
            if len(child.targets) != 1 or not isinstance(child.targets[0], ast.Name):
                continue
            if child.targets[0].id != "payload" or not isinstance(child.value, ast.Dict):
                continue
            for key_node, value_node in zip(child.value.keys, child.value.values):
                key = _constant_string(key_node)
                if not key:
                    continue
                self.provider_payload_edges.append(
                    {
                        "function": node.name,
                        "key": key,
                        "source_expr": _ast_name(value_node),
                        "line": child.lineno,
                    }
                )


def _extract_config_fields(names: list[str]) -> list[dict[str, Any]]:
    tree = _parse_module(REPO_ROOT / "agent_kernel" / "config.py")
    results: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id in names:
                results.append(
                    {
                        "field": node.target.id,
                        "default_expr": _ast_name(node.value),
                        "line": node.lineno,
                    }
                )
    return sorted(results, key=lambda item: item["field"])


def build_prompt_influence_graph() -> dict[str, Any]:
    policy_collector = PromptDiagnosticsCollector()
    policy_collector.visit(_parse_module(REPO_ROOT / "agent_kernel" / "policy.py"))
    policy_collector.visit(_parse_module(REPO_ROOT / "agent_kernel" / "extensions" / "context_budget.py"))

    llm_collector = LLMRenderCollector()
    llm_collector.visit(_parse_module(REPO_ROOT / "agent_kernel" / "llm.py"))

    graph_edges = [
        "config.use_prompt_proposals -> LLMDecisionPolicy._prompt_policy_payload() gate",
        "config.prompt_proposals_path -> json.loads(read_text(...)) -> _prompt_policy_payload()",
        "_prompt_policy_payload() -> retained_role_directives(...) -> _role_directive_overrides()",
        "_role_directive_overrides() + role -> _role_directive(role)",
        "self.system_prompt + _role_directive(role) -> _role_system_prompt(role)",
        "self.decision_prompt + _role_directive(role) -> _role_decision_prompt(role)",
        "_prompt_policy_payload() -> retained_artifact_payload(...prompt_proposal_set...) -> dedupe_prompt_adjustments(...) -> _active_prompt_adjustments()",
        "state + retrieval + summaries + plan -> ContextBudgeter.build_payload(...)",
        "ContextBudgeter.build_payload(...) -> payload",
        "planner_recovery_rewrite_brief -> payload['planner_recovery_brief']",
        "software_work_phase_gate_brief -> payload['software_work_phase_gate_brief']",
        "campaign_contract_brief -> payload['campaign_contract_brief']",
        "planner_recovery_artifact -> payload['planner_recovery_artifact']",
        "_role_system_prompt(role) -> client.create_decision(system_prompt=...)",
        "_role_decision_prompt(role) + runtime mutations -> client.create_decision(decision_prompt=...)",
        "payload -> client.create_decision(state_payload=...)",
        "decision_prompt + json.dumps(compact/minimal state_payload) -> _render_prompt(...)",
        "_render_prompt(...) -> provider user message content / ollama prompt",
        "system_prompt -> provider system message content / ollama system",
    ]

    return {
        "config_fields": _extract_config_fields(
            [
                "use_prompt_proposals",
                "prompt_proposals_path",
                "provider",
                "model_name",
                "vllm_host",
                "ollama_host",
            ]
        ),
        "policy_function_returns": policy_collector.function_returns,
        "prompt_files": policy_collector.prompt_file_reads,
        "payload_build_arguments": policy_collector.prompt_call_arguments,
        "context_payload_keys": policy_collector.context_payload_keys,
        "payload_runtime_additions": policy_collector.payload_runtime_additions,
        "system_prompt_mutations": policy_collector.system_prompt_mutations,
        "decision_prompt_mutations": policy_collector.decision_prompt_mutations,
        "llm_render_prompt": llm_collector.render_prompt_edges,
        "compact_state_payload_keys": llm_collector.compact_payload_keys,
        "minimal_state_payload_mutations": llm_collector.minimal_payload_mutations,
        "provider_payload_edges": llm_collector.provider_payload_edges,
        "graph_edges": graph_edges,
    }


def collect_prompt_diagnostics() -> dict[str, Any]:
    return build_prompt_influence_graph()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report Python AST stats and LLM prompt/payload insertion points for agent_kernel.",
    )
    parser.add_argument(
        "--root",
        default=str(DEFAULT_ROOT),
        help="Directory to scan for Python AST diagnostics. Defaults to agent_kernel.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit full diagnostics as JSON.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"Path does not exist: {root}")
    if not root.is_dir():
        raise SystemExit(f"Path is not a directory: {root}")

    diagnostics = {
        "python": collect_python_stats(root),
        "prompt_diagnostics": collect_prompt_diagnostics(),
    }

    if args.json:
        print(json.dumps(diagnostics, indent=2, sort_keys=True))
        return

    python = diagnostics["python"]
    prompt = diagnostics["prompt_diagnostics"]

    print(f"root: {python['root']}")
    print(f"python_files: {python['python_files']}")
    print(f"python_loc: {python['python_loc']}")
    print(f"ast_functions: {python['ast_totals']['functions']}")
    print(f"ast_classes: {python['ast_totals']['classes']}")
    print(f"ast_calls: {python['ast_totals']['calls']}")
    print(f"ast_assignments: {python['ast_totals']['assignments']}")
    print()
    print("top_loc_files:")
    for item in python["largest_files_by_loc"][:8]:
        print(f"  - {item['path']}: loc={item['python_loc']} ast_nodes={item['ast_nodes']}")
    print()
    print("prompt_files:")
    for item in prompt["prompt_files"]:
        print(f"  - {item['target']} <= prompts/{item['file']} (policy.py:{item['line']})")
    print()
    print("prompt_influence_graph:")
    for item in prompt["graph_edges"]:
        print(f"  - {item}")
    print()
    print("config_fields:")
    for item in prompt["config_fields"]:
        print(f"  - {item['field']} <= {item['default_expr']} (config.py:{item['line']})")
    print()
    print("policy_function_returns:")
    for item in prompt["policy_function_returns"]:
        print(f"  - {item['function']} => {item['expr']} (policy.py:{item['line']})")
    print()
    print("payload_build_arguments:")
    for item in prompt["payload_build_arguments"]:
        print(f"  - {item['argument']} <= {item['source_expr']} (policy.py:{item['line']})")
    print()
    print("context_payload_keys:")
    for item in prompt["context_payload_keys"]:
        print(f"  - {item['key']} <= {item['source_expr']} (extensions/context_budget.py:{item['line']})")
    print()
    print("runtime_payload_additions:")
    for item in prompt["payload_runtime_additions"]:
        print(f"  - {item['key']} <= {item['source_expr']} (policy.py:{item['line']})")
    print()
    print("decision_prompt_mutations:")
    for item in prompt["decision_prompt_mutations"]:
        print(f"  - {item['source_expr']} (policy.py:{item['line']})")
    print()
    print("compact_state_payload_keys:")
    for item in prompt["compact_state_payload_keys"]:
        print(f"  - {item['key']} <= {item['source_expr']} (llm.py:{item['line']})")
    print()
    print("minimal_state_payload_mutations:")
    for item in prompt["minimal_state_payload_mutations"]:
        print(f"  - {item['key']} <= {item['source_expr']} (llm.py:{item['line']})")
    print()
    print("provider_payload_edges:")
    for item in prompt["provider_payload_edges"]:
        print(f"  - {item['function']}.{item['key']} <= {item['source_expr']} (llm.py:{item['line']})")


if __name__ == "__main__":
    main()
