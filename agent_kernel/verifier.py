from __future__ import annotations

import ast
import builtins
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib

from .schemas import (
    CommandResult,
    TaskSpec,
    VerificationResult,
    classify_command_result_failure,
    classify_verification_reason,
)


def structured_artifact_verifier_covers_success_command(task: TaskSpec) -> bool:
    metadata = task.metadata if isinstance(task.metadata, dict) else {}
    semantic_verifier = metadata.get("semantic_verifier", {})
    if not isinstance(semantic_verifier, dict):
        return False
    return str(semantic_verifier.get("kind", "")).strip() in {
        "swe_patch_apply_check",
    }


class Verifier:
    def verify(self, task: TaskSpec, workspace: Path, result: CommandResult) -> VerificationResult:
        reasons: list[str] = []
        total_checks = 0
        failed_checks = 0

        def _record(passed: bool, reason: str = "") -> None:
            nonlocal total_checks, failed_checks
            total_checks += 1
            if passed:
                return
            failed_checks += 1
            if reason:
                reasons.append(reason)

        _record(not result.timed_out, "command timed out")
        _record(result.exit_code == 0, f"exit code was {result.exit_code}")

        for relative_path in task.expected_files:
            _record((workspace / relative_path).exists(), f"missing expected file: {relative_path}")
        for relative_path in task.forbidden_files:
            _record(not (workspace / relative_path).exists(), f"forbidden file present: {relative_path}")
        for relative_path, expected_content in task.expected_file_contents.items():
            path = workspace / relative_path
            if not path.exists():
                _record(False, f"missing expected file content target: {relative_path}")
                continue
            actual_content = path.read_text(encoding="utf-8")
            _record(actual_content == expected_content, f"unexpected file content: {relative_path}")

        combined_output = f"{result.stdout}\n{result.stderr}"
        for needle in task.expected_output_substrings:
            _record(needle in combined_output, f"missing expected output: {needle}")
        for needle in task.forbidden_output_substrings:
            _record(needle not in combined_output, f"forbidden output present: {needle}")

        reasons.extend(self._semantic_verification_reasons(task, workspace, result=result))
        success_command_result = self._verify_success_command(task, workspace, skip=bool(reasons))
        if success_command_result:
            if not bool(success_command_result.get("passed", False)):
                reason = str(success_command_result.get("reason", "")).strip()
                if reason:
                    reasons.append(reason)
        semantic_failures = max(0, len(reasons) - failed_checks)
        total_checks += semantic_failures
        failed_checks += semantic_failures
        failure_codes = [
            code
            for code in (classify_verification_reason(reason) for reason in reasons)
            if code
        ]
        terminal_failure = classify_command_result_failure(result)
        terminal_failure_code = str(terminal_failure.get("failure_code", "")).strip()
        if terminal_failure_code and terminal_failure_code not in failure_codes:
            failure_codes.append(terminal_failure_code)
        process_score = 1.0 if total_checks <= 0 else max(0.0, float(total_checks - failed_checks) / float(total_checks))
        outcome_label = "success" if not reasons else (failure_codes[0] if failure_codes else "verification_failure")
        controllability = "agent"
        if any(code in {"timeout", "verification_command_execution_failed", "git_branch_inspection_failed"} for code in failure_codes):
            controllability = "runtime"

        return VerificationResult(
            passed=not reasons,
            reasons=reasons or ["verification passed"],
            command_result=result,
            process_score=round(process_score, 4),
            outcome_label=outcome_label,
            outcome_confidence=1.0,
            controllability=controllability,
            failure_codes=failure_codes,
            evidence=[
                {
                    "kind": "verification_summary",
                    "expected_files_checked": len(task.expected_files),
                    "forbidden_files_checked": len(task.forbidden_files),
                    "expected_contents_checked": len(task.expected_file_contents),
                    "expected_outputs_checked": len(task.expected_output_substrings),
                    "forbidden_outputs_checked": len(task.forbidden_output_substrings),
                    "semantic_verifier_kind": str(task.metadata.get("semantic_verifier", {}).get("kind", "")).strip()
                    if isinstance(task.metadata.get("semantic_verifier", {}), dict)
                    else "",
                }
            ]
            + ([terminal_failure] if terminal_failure else [])
            + ([success_command_result] if success_command_result else []),
        )

    def _verify_success_command(
        self,
        task: TaskSpec,
        workspace: Path,
        *,
        skip: bool,
    ) -> dict[str, object]:
        command = str(task.success_command).strip()
        if not command or skip:
            return {}
        if structured_artifact_verifier_covers_success_command(task):
            return {
                "kind": "success_command_result",
                "command": command,
                "passed": True,
                "skipped": True,
                "skip_reason": "structured_artifact_verifier_covers_success_command",
            }
        timed_out = False
        try:
            completed = subprocess.run(
                ["bash", "-lc", command],
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=20,
            )
            exit_code = int(completed.returncode)
            stdout = completed.stdout
            stderr = completed.stderr
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            exit_code = 124
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
        passed = (not timed_out) and exit_code == 0
        reason = ""
        if timed_out:
            reason = "success command timed out"
        elif exit_code != 0:
            reason = f"success command exited with code {exit_code}"
        return {
            "kind": "success_command_result",
            "command": command,
            "exit_code": exit_code,
            "timed_out": timed_out,
            "passed": passed,
            "reason": reason,
            "stdout_preview": str(stdout)[-1000:],
            "stderr_preview": str(stderr)[-1000:],
        }

    def _semantic_verification_reasons(
        self,
        task: TaskSpec,
        workspace: Path,
        *,
        result: CommandResult | None = None,
    ) -> list[str]:
        contract = task.metadata.get("semantic_verifier")
        if not isinstance(contract, dict):
            return []
        reasons: list[str] = []
        kind = str(contract.get("kind", "")).strip()
        if kind == "repo_chore_review":
            reasons.extend(self._verify_repo_chore_review(workspace, contract))
        elif kind == "git_repo_review":
            reasons.extend(self._verify_git_repo_review(workspace, contract))
        elif kind == "swe_patch_apply_check":
            reasons.extend(self._verify_swe_patch_apply_check(workspace, contract))
        reasons.extend(self._verify_behavior_checks(workspace, contract))
        reasons.extend(self._verify_differential_checks(workspace, contract))
        reasons.extend(self._verify_repo_invariants(workspace, contract))
        reasons.extend(
            self._verify_semantic_assertions(
                workspace,
                contract,
                stdout=str(result.stdout if result is not None else ""),
                stderr=str(result.stderr if result is not None else ""),
            )
        )
        return reasons

    def _verify_swe_patch_apply_check(self, workspace: Path, contract: dict[str, object]) -> list[str]:
        repo = str(contract.get("repo", "")).strip()
        base_commit = str(contract.get("base_commit", "")).strip()
        repo_cache_root = str(contract.get("repo_cache_root", "")).strip()
        patch_path = str(contract.get("patch_path", "patch.diff")).strip() or "patch.diff"
        patch_file = (workspace / patch_path).resolve()
        if not repo:
            return ["SWE patch verifier missing repo"]
        if not base_commit:
            return ["SWE patch verifier missing base_commit"]
        if not repo_cache_root:
            return ["SWE patch verifier missing repo_cache_root"]
        if not patch_file.exists():
            return [f"SWE patch verifier missing patch file: {patch_path}"]
        try:
            patch_text = patch_file.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return [f"SWE patch verifier could not read patch file: {exc}"]
        forbidden_patch_texts = [
            str(value)
            for value in contract.get("forbidden_patch_texts", [])
            if str(value).strip()
        ]
        normalized_patch_text = _normalize_patch_text_for_repeat_guard(patch_text)
        for forbidden_patch_text in forbidden_patch_texts:
            if normalized_patch_text and normalized_patch_text == _normalize_patch_text_for_repeat_guard(forbidden_patch_text):
                return ["SWE patch repeats prior official-failed patch exactly"]
        placeholder_patterns = (
            r"# This is a test file\.",
            r"\bimport os\b",
            r"\bprocess_data\(",
            r"\bplaceholder\b",
            r"\bdummy\b",
        )
        added_patch_text = "\n".join(self._patch_added_lines(patch_text))
        if self._patch_has_placeholder_output_replacement(patch_text):
            return ["SWE patch replaces real behavior with placeholder output"]
        if self._patch_adds_placeholder_success_print(patch_text):
            return ["SWE patch adds placeholder success print instead of behavior"]
        for pattern in placeholder_patterns:
            if re.search(pattern, added_patch_text, flags=re.IGNORECASE):
                return ["SWE patch diff contains placeholder/template content"]
        for identifier in contract.get("required_patch_identifiers", []):
            normalized_identifier = str(identifier).strip()
            if normalized_identifier and not re.search(rf"\b{re.escape(normalized_identifier)}\b", patch_text):
                return [f"SWE patch does not reference required issue identifier: {normalized_identifier}"]
        if not self._patch_has_meaningful_change(patch_text):
            return ["SWE patch diff has no meaningful content change"]
        whitespace_only_structured_reason = self._patch_suspicious_whitespace_only_structured_data_change(patch_text)
        if whitespace_only_structured_reason:
            return [whitespace_only_structured_reason]
        structured_regex_reason = self._patch_suspicious_structured_data_regex_escape_duplicate(patch_text)
        if structured_regex_reason:
            return [structured_regex_reason]
        if not self._patch_has_executable_change(patch_text):
            return ["SWE patch diff changes only comments/docstrings/non-executable text"]
        comment_to_code_reason = self._patch_suspicious_non_executable_to_code_replacement(patch_text)
        if comment_to_code_reason:
            return [comment_to_code_reason]
        if self._patch_double_escapes_raw_regex_whitespace(patch_text):
            return ["SWE patch suspiciously double-escapes raw regex whitespace"]
        isolated_replacement_reason = self._patch_isolated_one_line_production_python_replacement(patch_text)
        if isolated_replacement_reason:
            return [isolated_replacement_reason]
        language_mismatch_reason = self._patch_suspicious_non_python_language_mismatch(patch_text)
        if language_mismatch_reason:
            return [language_mismatch_reason]
        token_flip_reason = self._patch_suspicious_semantic_token_flips(patch_text)
        if token_flip_reason:
            return [token_flip_reason]
        config_key_replacement_reason = self._patch_suspicious_config_key_replacements(patch_text)
        if config_key_replacement_reason:
            return [config_key_replacement_reason]
        duplicate_call_reason = self._patch_suspicious_duplicate_surrounding_call_wrappers(patch_text)
        if duplicate_call_reason:
            return [duplicate_call_reason]
        duplicate_statement_reason = self._patch_suspicious_duplicate_existing_python_statements(patch_text)
        if duplicate_statement_reason:
            return [duplicate_statement_reason]
        hunk_replacement_reason = self._patch_suspicious_python_hunk_replacements(patch_text)
        if hunk_replacement_reason:
            return [hunk_replacement_reason]
        behavior_deletion_reason = self._patch_suspicious_python_behavior_deletion(patch_text)
        if behavior_deletion_reason:
            return [behavior_deletion_reason]
        call_removal_reason = self._patch_suspicious_python_call_statement_removal(patch_text)
        if call_removal_reason:
            return [call_removal_reason]
        signature_change_reason = self._patch_suspicious_python_signature_contract_change(patch_text)
        if signature_change_reason:
            return [signature_change_reason]
        class_config_reason = self._patch_suspicious_python_class_config_removal(patch_text)
        if class_config_reason:
            return [class_config_reason]
        control_flow_reason = self._patch_suspicious_python_control_flow_replacement(patch_text)
        if control_flow_reason:
            return [control_flow_reason]
        tuple_type_reason = self._patch_suspicious_python_tuple_type_construction(patch_text)
        if tuple_type_reason:
            return [tuple_type_reason]
        fragment_reason = self._patch_suspicious_python_repair_fragments(patch_text)
        if fragment_reason:
            return [fragment_reason]
        call_assignment_reason = self._patch_suspicious_python_call_assignment_collapse(patch_text)
        if call_assignment_reason:
            return [call_assignment_reason]
        container_api_reason = self._patch_suspicious_python_container_api_mismatch(patch_text)
        if container_api_reason:
            return [container_api_reason]
        tiny_mutation_reason = self._patch_suspicious_tiny_production_mutation(patch_text)
        if tiny_mutation_reason:
            return [tiny_mutation_reason]
        text_template_reason = self._patch_suspicious_text_template_replacements(patch_text)
        if text_template_reason:
            return [text_template_reason]
        disallowed_path_reason = self._patch_only_changes_disallowed_swe_paths(patch_text)
        if disallowed_path_reason:
            return [disallowed_path_reason]
        expected_changed_paths = [
            str(path).strip()
            for path in contract.get("expected_changed_paths", [])
            if str(path).strip()
        ]
        if expected_changed_paths:
            changed_paths = self._patch_changed_paths(patch_text)
            if not changed_paths:
                return ["SWE patch diff has no changed file paths"]
            expected_set = set(expected_changed_paths)
            actual_set = set(changed_paths)
            unexpected = sorted(actual_set - expected_set)
            if unexpected:
                return [f"SWE patch diff includes unexpected path: {path}" for path in unexpected]
        repo_path = self._swe_repo_cache_path(repo_cache_root, repo)
        if repo_path is None:
            return [f"SWE patch verifier missing repo cache: {repo}"]
        with tempfile.TemporaryDirectory(prefix="swe_patch_verify_") as tmp:
            worktree = Path(tmp) / "repo"
            clone = subprocess.run(
                ["git", "clone", "--shared", "--no-checkout", str(repo_path), str(worktree)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            if clone.returncode != 0:
                return [f"SWE patch verifier clone failed: {clone.stderr.strip()}"]
            checkout = subprocess.run(
                ["git", "-C", str(worktree), "checkout", "--detach", base_commit],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            if checkout.returncode != 0:
                return [f"SWE patch verifier checkout failed: {checkout.stderr.strip()}"]
            apply_check = subprocess.run(
                ["git", "-C", str(worktree), "apply", "--check", str(patch_file)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            if apply_check.returncode != 0:
                detail = apply_check.stderr.strip()
                return [f"SWE patch apply check failed: {detail}"]
            python_paths = [
                path
                for path in self._patch_changed_paths(patch_text)
                if path.endswith(".py") and (worktree / path).exists()
            ]
            original_python_sources = {
                path: (worktree / path).read_text(encoding="utf-8", errors="replace")
                for path in python_paths
            }
            apply_patch = subprocess.run(
                ["git", "-C", str(worktree), "apply", str(patch_file)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            if apply_patch.returncode != 0:
                detail = apply_patch.stderr.strip()
                return [f"SWE patch apply failed after check: {detail}"]
            for path in self._patch_changed_paths(patch_text):
                if path.endswith(".json") and (worktree / path).exists():
                    try:
                        json.loads((worktree / path).read_text(encoding="utf-8", errors="replace"))
                    except json.JSONDecodeError as exc:
                        return [f"SWE patch JSON syntax check failed in {path}: {exc.msg}"]
                if path.endswith(".toml") and (worktree / path).exists():
                    try:
                        tomllib.loads((worktree / path).read_text(encoding="utf-8", errors="replace"))
                    except tomllib.TOMLDecodeError as exc:
                        return [f"SWE patch TOML syntax check failed in {path}: {exc}"]
            if python_paths:
                compile_check = subprocess.run(
                    [sys.executable, "-m", "py_compile", *python_paths],
                    cwd=worktree,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                if compile_check.returncode != 0:
                    detail = compile_check.stderr.strip().splitlines()[-1] if compile_check.stderr.strip() else ""
                    return [f"SWE patch python syntax check failed: {detail}"]
                ast_changed = False
                for path in python_paths:
                    patched_source = (worktree / path).read_text(encoding="utf-8", errors="replace")
                    if _python_executable_ast_changed(original_python_sources.get(path, ""), patched_source):
                        ast_changed = True
                        break
                if not ast_changed:
                    return ["SWE patch python AST unchanged after ignoring docstrings/comments"]
                for path in python_paths:
                    if _is_python_test_path(path):
                        continue
                    original_source = original_python_sources.get(path, "")
                    patched_source = (worktree / path).read_text(encoding="utf-8", errors="replace")
                    if _python_string_literal_only_changed(original_source, patched_source):
                        return [f"SWE patch changes only Python string literal values in {path}"]
                    if _python_annotation_only_changed(original_source, patched_source):
                        return [f"SWE patch changes only Python type annotations in {path}"]
                    indentation_moves = _python_indentation_only_statement_moves(original_source, patched_source)
                    if indentation_moves:
                        preview = ", ".join(indentation_moves[:5])
                        return [f"SWE patch makes indentation-only statement moves in {path}: {preview}"]
                    removed_defs = _removed_python_definition_names(
                        original_source,
                        patched_source,
                    )
                    if removed_defs:
                        preview = ", ".join(removed_defs[:5])
                        return [f"SWE patch removes existing Python definitions in {path}: {preview}"]
                    unused_new_params = _unused_new_python_parameters(
                        original_source,
                        patched_source,
                    )
                    if unused_new_params:
                        preview = ", ".join(unused_new_params[:5])
                        return [f"SWE patch adds unused production function parameters in {path}: {preview}"]
                    before_invalid_init_returns = set(_python_init_return_value_names(original_python_sources.get(path, "")))
                    invalid_init_returns = sorted(
                        set(
                            _python_init_return_value_names(
                                (worktree / path).read_text(encoding="utf-8", errors="replace")
                            )
                        )
                        - before_invalid_init_returns
                    )
                    if invalid_init_returns:
                        preview = ", ".join(invalid_init_returns[:5])
                        return [f"SWE patch leaves invalid __init__ return values in {path}: {preview}"]
                    before_init_generators = set(_python_init_generator_names(original_python_sources.get(path, "")))
                    invalid_init_generators = sorted(
                        set(_python_init_generator_names((worktree / path).read_text(encoding="utf-8", errors="replace")))
                        - before_init_generators
                    )
                    if invalid_init_generators:
                        preview = ", ".join(invalid_init_generators[:5])
                        return [f"SWE patch leaves invalid __init__ generators in {path}: {preview}"]
                    removed_init_assignments = _python_removed_init_instance_assignments(
                        original_source,
                        patched_source,
                    )
                    if removed_init_assignments:
                        preview = ", ".join(removed_init_assignments[:5])
                        return [f"SWE patch removes constructor instance assignments in {path}: {preview}"]
                    before_unbound_locals = set(
                        _python_local_load_before_assignment_names(original_python_sources.get(path, ""))
                    )
                    introduced_unbound_locals = sorted(
                        set(
                            _python_local_load_before_assignment_names(
                                (worktree / path).read_text(encoding="utf-8", errors="replace")
                            )
                        )
                        - before_unbound_locals
                    )
                    if introduced_unbound_locals:
                        preview = ", ".join(introduced_unbound_locals[:5])
                        return [f"SWE patch introduces local use before assignment in {path}: {preview}"]
                    removed_registrations = _removed_python_module_registration_names(
                        original_python_sources.get(path, ""),
                        (worktree / path).read_text(encoding="utf-8", errors="replace"),
                    )
                    if removed_registrations:
                        preview = ", ".join(removed_registrations[:5])
                        return [f"SWE patch removes module registration assignments in {path}: {preview}"]
                    removed_module_assignments = _removed_python_module_state_assignment_names(
                        original_source,
                        patched_source,
                    )
                    if removed_module_assignments:
                        preview = ", ".join(removed_module_assignments[:5])
                        return [f"SWE patch removes module-level state assignments in {path}: {preview}"]
                    suspicious_attr_replacements = _python_suspicious_attribute_replacement_details(
                        original_python_sources.get(path, ""),
                        (worktree / path).read_text(encoding="utf-8", errors="replace"),
                    )
                    if suspicious_attr_replacements:
                        preview = ", ".join(suspicious_attr_replacements[:5])
                        return [f"SWE patch makes suspicious unknown attribute replacements in {path}: {preview}"]
                    private_attr_access = _python_introduced_private_attribute_accesses(
                        original_python_sources.get(path, ""),
                        (worktree / path).read_text(encoding="utf-8", errors="replace"),
                    )
                    if private_attr_access:
                        preview = ", ".join(private_attr_access[:5])
                        return [f"SWE patch introduces private attribute reads in {path}: {preview}"]
                    unknown_self_attr_access = _python_introduced_unknown_self_private_attribute_accesses(
                        original_source,
                        patched_source,
                    )
                    if unknown_self_attr_access:
                        preview = ", ".join(unknown_self_attr_access[:5])
                        return [f"SWE patch introduces unknown self private attribute reads in {path}: {preview}"]
                    exception_contract_regressions = _python_exception_contract_regression_details(
                        original_python_sources.get(path, ""),
                        (worktree / path).read_text(encoding="utf-8", errors="replace"),
                    )
                    if exception_contract_regressions:
                        preview = ", ".join(exception_contract_regressions[:5])
                        return [f"SWE patch makes suspicious exception contract changes in {path}: {preview}"]
                    redundant_normalizations = _python_redundant_decorated_normalization_details(
                        original_python_sources.get(path, ""),
                        (worktree / path).read_text(encoding="utf-8", errors="replace"),
                    )
                    if redundant_normalizations:
                        preview = ", ".join(redundant_normalizations[:5])
                        return [f"SWE patch adds redundant decorated normalizations in {path}: {preview}"]
                    nested_reducers = _python_nested_duplicate_reducer_details(original_source, patched_source)
                    if nested_reducers:
                        preview = ", ".join(nested_reducers[:5])
                        return [f"SWE patch nests duplicate numerical reducers in {path}: {preview}"]
                    self_recursive_properties = _python_introduced_self_recursive_property_accesses(
                        original_source,
                        patched_source,
                    )
                    if self_recursive_properties:
                        preview = ", ".join(self_recursive_properties[:5])
                        return [f"SWE patch introduces self-recursive property access in {path}: {preview}"]
                    nested_assignment_regressions = _python_nested_assignment_replacement_details(patch_text)
                    if nested_assignment_regressions:
                        preview = ", ".join(nested_assignment_regressions[:5])
                        return [f"SWE patch replaces container initialization with nested assignment in {path}: {preview}"]
                    statement_replacement_reason = self._patch_suspicious_python_statement_replacements(patch_text)
                    if statement_replacement_reason:
                        return [statement_replacement_reason]
                    introduced_unresolved_names = _introduced_python_unresolved_name_loads(
                        original_python_sources.get(path, ""),
                        (worktree / path).read_text(encoding="utf-8", errors="replace"),
                    )
                    if introduced_unresolved_names:
                        preview = ", ".join(introduced_unresolved_names[:5])
                        return [f"SWE patch introduces unresolved name reads in {path}: {preview}"]
                    call_arity_mismatches = _python_introduced_local_call_arity_mismatch_details(
                        original_source,
                        patched_source,
                        worktree,
                    )
                    if call_arity_mismatches:
                        preview = ", ".join(call_arity_mismatches[:5])
                        return [f"SWE patch changes calls beyond local function arity in {path}: {preview}"]
                    call_keyword_mismatches = _python_introduced_local_call_keyword_mismatch_details(
                        original_source,
                        patched_source,
                        worktree,
                    )
                    if call_keyword_mismatches:
                        preview = ", ".join(call_keyword_mismatches[:5])
                        return [f"SWE patch changes calls beyond local function keyword contract in {path}: {preview}"]
                    none_container_misuse = _python_introduced_none_container_misuse_details(
                        original_source,
                        patched_source,
                    )
                    if none_container_misuse:
                        preview = ", ".join(none_container_misuse[:5])
                        return [f"SWE patch introduces None container misuse in {path}: {preview}"]
                    function_object_arithmetic = _python_introduced_function_object_arithmetic_details(
                        original_source,
                        patched_source,
                    )
                    if function_object_arithmetic:
                        preview = ", ".join(function_object_arithmetic[:5])
                        return [f"SWE patch introduces function object arithmetic in {path}: {preview}"]
                    removed_return_paths = _python_removed_return_value_paths(
                        original_source,
                        patched_source,
                    )
                    if removed_return_paths:
                        preview = ", ".join(removed_return_paths[:5])
                        return [f"SWE patch removes production return value paths in {path}: {preview}"]
                    none_return_regressions = _python_introduced_none_return_value_paths(original_source, patched_source)
                    if none_return_regressions:
                        preview = ", ".join(none_return_regressions[:5])
                        return [f"SWE patch introduces None return value paths in {path}: {preview}"]
                    risky_boolean_flips = _python_suspicious_boolean_return_flip_names(
                        original_python_sources.get(path, ""),
                        (worktree / path).read_text(encoding="utf-8", errors="replace"),
                    )
                    if risky_boolean_flips:
                        preview = ", ".join(risky_boolean_flips[:5])
                        return [f"SWE patch makes suspicious isolated boolean return flips in {path}: {preview}"]
        return []

    @staticmethod
    def _patch_changed_paths(patch_text: str) -> list[str]:
        paths: list[str] = []
        for line in patch_text.splitlines():
            if line.startswith("diff --git "):
                parts = line.split()
                if len(parts) >= 4:
                    path = parts[3]
                    if path.startswith("b/"):
                        path = path[2:]
                    if path and path != "/dev/null" and path not in paths:
                        paths.append(path)
                continue
            if line.startswith("+++ "):
                path = line[4:].strip()
                if path.startswith("b/"):
                    path = path[2:]
                if path and path != "/dev/null" and path not in paths:
                    paths.append(path)
        return paths

    @staticmethod
    def _patch_has_meaningful_change(patch_text: str) -> bool:
        removed: list[str] = []
        added: list[str] = []
        for line in patch_text.splitlines():
            if line.startswith("--- ") or line.startswith("+++ "):
                continue
            if line.startswith("-"):
                removed.append(line[1:])
            elif line.startswith("+"):
                added.append(line[1:])
        if not removed and not added:
            return False
        return removed != added

    @staticmethod
    def _patch_added_lines(patch_text: str) -> list[str]:
        added: list[str] = []
        for line in patch_text.splitlines():
            if line.startswith("+++ "):
                continue
            if line.startswith("+"):
                added.append(line[1:])
        return added

    @staticmethod
    def _patch_has_executable_change(patch_text: str) -> bool:
        changed_lines: list[str] = []
        for line in patch_text.splitlines():
            if line.startswith("--- ") or line.startswith("+++ "):
                continue
            if line.startswith(("-", "+")):
                changed_lines.append(line[1:])
        if not changed_lines:
            return False
        return any(not _looks_like_non_executable_patch_line(line) for line in changed_lines)

    @staticmethod
    def _patch_double_escapes_raw_regex_whitespace(patch_text: str) -> bool:
        removed: list[str] = []
        for line in patch_text.splitlines():
            if line.startswith("-") and not line.startswith("--- "):
                removed.append(line[1:])
                continue
            if not line.startswith("+") or line.startswith("+++ "):
                continue
            payload = line[1:]
            if re.search(r"=\s*rf?[\"']", payload) and r"\\s" in payload:
                return True
            if re.search(r"\\\\[sSdDwWbB]", payload):
                token = re.search(r"\\\\[sSdDwWbB]", payload)
                if token and any(token.group(0).replace("\\\\", "\\") in old for old in removed[-8:]):
                    return True
        return False

    @staticmethod
    def _patch_suspicious_whitespace_only_structured_data_change(patch_text: str) -> str:
        current_path = ""
        changed: dict[str, dict[str, list[str]]] = {}
        for line in patch_text.splitlines():
            if line.startswith("+++ "):
                path = line[4:].strip()
                current_path = path[2:] if path.startswith("b/") else path
                if current_path == "/dev/null":
                    current_path = ""
                continue
            if not current_path:
                continue
            if line.startswith("--- ") or line.startswith("diff --git ") or line.startswith("@@ "):
                continue
            if not current_path.endswith((".json", ".yaml", ".yml", ".toml")):
                continue
            bucket = changed.setdefault(current_path, {"removed": [], "added": []})
            if line.startswith("-"):
                bucket["removed"].append(line[1:])
            elif line.startswith("+"):
                bucket["added"].append(line[1:])
        for path, payload in changed.items():
            removed = payload["removed"]
            added = payload["added"]
            if not removed or not added:
                continue
            removed_normalized = "".join("".join(removed).split())
            added_normalized = "".join("".join(added).split())
            if removed_normalized and removed_normalized == added_normalized:
                return f"SWE patch changes only whitespace in structured data file: {path}"
        return ""

    @staticmethod
    def _patch_suspicious_structured_data_regex_escape_duplicate(patch_text: str) -> str:
        current_path = ""
        hunk_lines: list[tuple[str, str]] = []

        def pattern_value(line: str) -> str:
            match = re.search(r'"pattern"\s*:\s*"([^"]+)"', line)
            return match.group(1) if match else ""

        def flush() -> str:
            if current_path.endswith((".json", ".yaml", ".yml")):
                existing_patterns = {
                    pattern_value(line)
                    for kind, line in hunk_lines
                    if kind in {" ", "-"} and pattern_value(line)
                }
                added_patterns = [
                    pattern_value(line)
                    for kind, line in hunk_lines
                    if kind == "+" and pattern_value(line)
                ]
                for added_pattern in added_patterns:
                    less_escaped = added_pattern.replace("\\\\\\\\", "\\\\")
                    if less_escaped != added_pattern and less_escaped in existing_patterns:
                        hunk_lines.clear()
                        return (
                            "SWE patch adds double-escaped duplicate regex pattern in structured data file: "
                            f"{current_path}"
                        )
            hunk_lines.clear()
            return ""

        for line in patch_text.splitlines():
            if line.startswith("+++ "):
                reason = flush()
                if reason:
                    return reason
                path = line[4:].strip()
                current_path = path[2:] if path.startswith("b/") else path
                if current_path == "/dev/null":
                    current_path = ""
                continue
            if line.startswith(("diff --git ", "--- ")):
                reason = flush()
                if reason:
                    return reason
                continue
            if line.startswith("@@ "):
                reason = flush()
                if reason:
                    return reason
                continue
            if not current_path.endswith((".json", ".yaml", ".yml")):
                continue
            if line.startswith("+") and not line.startswith("+++ "):
                hunk_lines.append(("+", line[1:]))
            elif line.startswith("-") and not line.startswith("--- "):
                hunk_lines.append(("-", line[1:]))
            else:
                hunk_lines.append((" ", line[1:] if line.startswith(" ") else line))
        return flush()

    @staticmethod
    def _patch_suspicious_python_behavior_deletion(patch_text: str) -> str:
        current_path = ""
        removed: list[str] = []
        added: list[str] = []

        def flush() -> str:
            if current_path.endswith(".py") and not _is_python_test_path(current_path):
                removed_exec = [
                    line.strip()
                    for line in removed
                    if line.strip() and not _looks_like_non_executable_patch_line(line)
                ]
                added_exec = [
                    line.strip()
                    for line in added
                    if line.strip() and not _looks_like_non_executable_patch_line(line)
                ]
                removed_behavior = [
                    line
                    for line in removed_exec
                    if re.search(r"\b(yield|return|raise|append|extend|update|ValidationError|Match\.create)\b", line)
                ]
                added_behavior = [
                    line
                    for line in added_exec
                    if re.search(r"\b(yield|return|raise|append|extend|update|ValidationError|Match\.create)\b", line)
                ]
                if len(removed_exec) >= 3 and len(added_exec) <= 1 and removed_behavior:
                    preview = " | ".join(_python_line_preview(line) for line in removed_behavior[:3])
                    removed.clear()
                    added.clear()
                    return (
                        "SWE patch deletes production behavior without enough replacement structure "
                        f"in {current_path}: {preview}"
                    )
                if len(removed_behavior) >= 2 and len(added_behavior) == 0:
                    preview = " | ".join(_python_line_preview(line) for line in removed_behavior[:3])
                    removed.clear()
                    added.clear()
                    return (
                        "SWE patch removes production validation/reporting behavior without replacement "
                        f"in {current_path}: {preview}"
                    )
            removed.clear()
            added.clear()
            return ""

        for line in patch_text.splitlines():
            if line.startswith("+++ "):
                reason = flush()
                if reason:
                    return reason
                path = line[4:].strip()
                current_path = path[2:] if path.startswith("b/") else path
                if current_path == "/dev/null":
                    current_path = ""
                continue
            if line.startswith(("diff --git ", "@@ ", "--- ")):
                reason = flush()
                if reason:
                    return reason
                continue
            if line.startswith("-"):
                removed.append(line[1:])
                continue
            if line.startswith("+"):
                added.append(line[1:])
                continue
        return flush()

    @staticmethod
    def _patch_suspicious_python_signature_contract_change(patch_text: str) -> str:
        current_path = ""
        removed_lines: list[str] = []
        added_lines: list[str] = []

        def param_count(signature: str) -> tuple[str, int] | None:
            match = re.search(r"\bdef\s+([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*(?:->|:)", signature)
            if not match:
                return None
            params = [
                item.strip()
                for item in match.group(2).split(",")
                if item.strip() and item.strip() not in {"/", "*"}
            ]
            return match.group(1), len(params)

        def flush() -> str:
            if current_path.endswith(".py") and not _is_python_test_path(current_path):
                removed_by_name: dict[str, int] = {}
                removed_text = " ".join(line.strip() for line in removed_lines)
                added_text = " ".join(line.strip() for line in added_lines)
                removed_defs = re.findall(r"\bdef\s+[A-Za-z_][A-Za-z0-9_]*\s*\([^)]*\)\s*(?:->\s*[^:]+)?\s*:", removed_text)
                added_defs = re.findall(r"\bdef\s+[A-Za-z_][A-Za-z0-9_]*\s*\([^)]*\)\s*(?:->\s*[^:]+)?\s*:", added_text)
                for line in removed_defs:
                    parsed = param_count(line.strip())
                    if parsed:
                        removed_by_name[parsed[0]] = parsed[1]
                for line in added_defs:
                    parsed = param_count(line.strip())
                    if not parsed:
                        continue
                    name, new_count = parsed
                    old_count = removed_by_name.get(name)
                    if old_count is not None and new_count < old_count:
                        removed_defs.clear()
                        added_defs.clear()
                        return (
                            "SWE patch reduces production function signature arity without callsite compatibility "
                            f"in {current_path}: {name} {old_count}->{new_count}"
                        )
            removed_lines.clear()
            added_lines.clear()
            return ""

        for line in patch_text.splitlines():
            if line.startswith("+++ "):
                reason = flush()
                if reason:
                    return reason
                path = line[4:].strip()
                current_path = path[2:] if path.startswith("b/") else path
                if current_path == "/dev/null":
                    current_path = ""
                continue
            if line.startswith(("diff --git ", "@@ ", "--- ")):
                reason = flush()
                if reason:
                    return reason
                continue
            if line.startswith("-"):
                removed_lines.append(line[1:])
            elif line.startswith("+"):
                added_lines.append(line[1:])
        return flush()

    @staticmethod
    def _patch_suspicious_python_class_config_removal(patch_text: str) -> str:
        current_path = ""
        removed: list[str] = []
        added: list[str] = []

        def flush() -> str:
            if current_path.endswith(".py") and not _is_python_test_path(current_path):
                removed_text = "\n".join(removed)
                added_text = "\n".join(added)
                if re.search(r"\bmodel_config\s*=", removed_text) and not re.search(r"\b(model_config|ConfigDict)\b", added_text):
                    removed.clear()
                    added.clear()
                    return f"SWE patch removes class model_config without equivalent replacement in {current_path}"
            removed.clear()
            added.clear()
            return ""

        for line in patch_text.splitlines():
            if line.startswith("+++ "):
                reason = flush()
                if reason:
                    return reason
                path = line[4:].strip()
                current_path = path[2:] if path.startswith("b/") else path
                if current_path == "/dev/null":
                    current_path = ""
                continue
            if line.startswith(("diff --git ", "@@ ", "--- ")):
                reason = flush()
                if reason:
                    return reason
                continue
            if line.startswith("-"):
                removed.append(line[1:])
            elif line.startswith("+"):
                added.append(line[1:])
        return flush()

    @staticmethod
    def _patch_suspicious_python_control_flow_replacement(patch_text: str) -> str:
        current_path = ""
        removed: list[str] = []
        added: list[str] = []

        def flush() -> str:
            if current_path.endswith(".py") and not _is_python_test_path(current_path):
                removed_text = "\n".join(line.strip() for line in removed)
                added_text = "\n".join(line.strip() for line in added)
                if re.search(r"\bcontinue\b", removed_text) and re.search(r"\breturn\s+False\b", added_text):
                    removed.clear()
                    added.clear()
                    return f"SWE patch replaces loop continuation with boolean return in production code: {current_path}"
            removed.clear()
            added.clear()
            return ""

        for line in patch_text.splitlines():
            if line.startswith("+++ "):
                reason = flush()
                if reason:
                    return reason
                path = line[4:].strip()
                current_path = path[2:] if path.startswith("b/") else path
                if current_path == "/dev/null":
                    current_path = ""
                continue
            if line.startswith(("diff --git ", "@@ ", "--- ")):
                reason = flush()
                if reason:
                    return reason
                continue
            if line.startswith("-"):
                removed.append(line[1:])
            elif line.startswith("+"):
                added.append(line[1:])
        return flush()

    @staticmethod
    def _patch_suspicious_python_tuple_type_construction(patch_text: str) -> str:
        current_path = ""
        for line in patch_text.splitlines():
            if line.startswith("+++ "):
                path = line[4:].strip()
                current_path = path[2:] if path.startswith("b/") else path
                if current_path == "/dev/null":
                    current_path = ""
                continue
            if not current_path.endswith(".py") or _is_python_test_path(current_path):
                continue
            if line.startswith("+") and not line.startswith("+++ ") and re.search(r"\btuple\s*\(\s*\[\s*(list|dict|set|tuple)\s*\]\s*\)", line):
                return f"SWE patch constructs type tuples through tuple([type]) in production code: {current_path}"
        return ""

    @staticmethod
    def _patch_suspicious_python_repair_fragments(patch_text: str) -> str:
        current_path = ""
        hunk_lines: list[tuple[str, str]] = []

        def flush() -> str:
            if current_path.endswith(".py") and not _is_python_test_path(current_path):
                removed = [line.strip() for kind, line in hunk_lines if kind == "-" and line.strip()]
                added = [line.strip() for kind, line in hunk_lines if kind == "+" and line.strip()]
                context = [line.strip() for kind, line in hunk_lines if kind == " " and line.strip()]
                added_text = "\n".join(added)
                removed_text = "\n".join(removed)
                context_text = "\n".join(context)
                if re.search(r"\b[A-Za-z_][A-Za-z0-9_]*\s*:\s*[A-Za-z_][A-Za-z0-9_\[\], |.]*,\s*$", added_text) and "def " not in added_text:
                    hunk_lines.clear()
                    return f"SWE patch adds bare type-annotation fragments outside a signature in {current_path}"
                if "hasattr(self," in added_text and re.search(r"\[[\"'][A-Za-z0-9_]+[\"']\]", removed_text):
                    hunk_lines.clear()
                    return (
                        "SWE patch replaces data extraction with unrelated self-attribute initialization "
                        f"in {current_path}"
                    )
                if "errors.append(" in added_text and "errors.append(" in context_text:
                    hunk_lines.clear()
                    return f"SWE patch duplicates existing error append flow in {current_path}"
            hunk_lines.clear()
            return ""

        for line in patch_text.splitlines():
            if line.startswith("+++ "):
                reason = flush()
                if reason:
                    return reason
                path = line[4:].strip()
                current_path = path[2:] if path.startswith("b/") else path
                if current_path == "/dev/null":
                    current_path = ""
                continue
            if line.startswith(("diff --git ", "--- ")):
                reason = flush()
                if reason:
                    return reason
                continue
            if line.startswith("@@ "):
                reason = flush()
                if reason:
                    return reason
                continue
            if not current_path.endswith(".py"):
                continue
            if line.startswith("+") and not line.startswith("+++ "):
                hunk_lines.append(("+", line[1:]))
            elif line.startswith("-") and not line.startswith("--- "):
                hunk_lines.append(("-", line[1:]))
            else:
                hunk_lines.append((" ", line[1:] if line.startswith(" ") else line))
        return flush()

    @staticmethod
    def _patch_suspicious_python_call_statement_removal(patch_text: str) -> str:
        current_path = ""
        hunk_lines: list[tuple[str, str]] = []

        def call_name(line: str) -> str:
            stripped = line.strip()
            match = re.match(
                r"^(?:await\s+)?((?:self|cls|super\(\)|[A-Za-z_][A-Za-z0-9_]*)"
                r"(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\s*\(",
                stripped,
            )
            if not match:
                return ""
            if re.match(r"^(?:return|yield|raise|if|while|for|with)\b", stripped):
                return ""
            return match.group(1)

        def flush() -> str:
            if current_path.endswith(".py") and not _is_python_test_path(current_path):
                removed = [
                    line.strip()
                    for kind, line in hunk_lines
                    if kind == "-" and line.strip() and not _looks_like_non_executable_patch_line(line)
                ]
                added = [
                    line.strip()
                    for kind, line in hunk_lines
                    if kind == "+" and line.strip() and not _looks_like_non_executable_patch_line(line)
                ]
                added_text = "\n".join(added)
                for line in removed:
                    callee = call_name(line)
                    if not callee:
                        continue
                    if re.search(rf"\b{re.escape(callee)}\s*\(", added_text):
                        continue
                    hunk_lines.clear()
                    return (
                        "SWE patch removes production call statement without preserving the call "
                        f"in {current_path}: {_python_line_preview(line)}"
                    )
            hunk_lines.clear()
            return ""

        for line in patch_text.splitlines():
            if line.startswith("+++ "):
                current_path = line[4:].strip()
                if current_path.startswith("b/"):
                    current_path = current_path[2:]
                continue
            if line.startswith(("diff --git ", "--- ")):
                reason = flush()
                if reason:
                    return reason
                continue
            if line.startswith("@@ "):
                reason = flush()
                if reason:
                    return reason
                continue
            if not current_path.endswith(".py"):
                continue
            if line.startswith("+") and not line.startswith("+++ "):
                hunk_lines.append(("+", line[1:]))
            elif line.startswith("-") and not line.startswith("--- "):
                hunk_lines.append(("-", line[1:]))
            else:
                hunk_lines.append((" ", line[1:] if line.startswith(" ") else line))
        return flush()

    @staticmethod
    def _patch_suspicious_python_call_assignment_collapse(patch_text: str) -> str:
        current_path = ""
        hunk_lines: list[tuple[str, str]] = []

        def flush() -> str:
            if current_path.endswith(".py") and not _is_python_test_path(current_path):
                removed = [line.strip() for kind, line in hunk_lines if kind == "-" and line.strip()]
                added = [line.strip() for kind, line in hunk_lines if kind == "+" and line.strip()]
                removed_text = "\n".join(removed)
                if (
                    len(removed) >= 3
                    and len(added) == 1
                    and re.search(r"=\s*[A-Za-z_][A-Za-z0-9_.]*\s*\(", removed_text)
                    and re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=\s*(self|cls|[A-Za-z_][A-Za-z0-9_]*)\.", added[0])
                ):
                    preview = _python_line_preview(added[0])
                    hunk_lines.clear()
                    return (
                        "SWE patch collapses a production call-assignment block into a bare attribute assignment "
                        f"in {current_path}: {preview}"
                    )
            hunk_lines.clear()
            return ""

        for line in patch_text.splitlines():
            if line.startswith("+++ "):
                reason = flush()
                if reason:
                    return reason
                path = line[4:].strip()
                current_path = path[2:] if path.startswith("b/") else path
                if current_path == "/dev/null":
                    current_path = ""
                continue
            if line.startswith(("diff --git ", "--- ")):
                reason = flush()
                if reason:
                    return reason
                continue
            if line.startswith("@@ "):
                reason = flush()
                if reason:
                    return reason
                continue
            if not current_path.endswith(".py"):
                continue
            if line.startswith("+") and not line.startswith("+++ "):
                hunk_lines.append(("+", line[1:]))
            elif line.startswith("-") and not line.startswith("--- "):
                hunk_lines.append(("-", line[1:]))
            else:
                hunk_lines.append((" ", line[1:] if line.startswith(" ") else line))
        return flush()

    @staticmethod
    def _patch_suspicious_python_container_api_mismatch(patch_text: str) -> str:
        current_path = ""
        hunk_lines: list[tuple[str, str]] = []

        def call_target(line: str, method: str) -> str:
            match = re.search(r"(?P<target>.+)\." + re.escape(method) + r"\s*\(", line.strip())
            if not match:
                return ""
            return re.sub(r"\s+", "", match.group("target"))

        def flush() -> str:
            if current_path.endswith(".py") and not _is_python_test_path(current_path):
                existing_update_targets = {
                    target
                    for kind, line in hunk_lines
                    if kind in {" ", "-"}
                    for target in [call_target(line, "update")]
                    if target
                }
                added_append_targets = {
                    target
                    for kind, line in hunk_lines
                    if kind == "+"
                    for target in [call_target(line, "append")]
                    if target
                }
                overlap = sorted(existing_update_targets & added_append_targets)
                if overlap:
                    preview = ", ".join(overlap[:3])
                    hunk_lines.clear()
                    return (
                        "SWE patch mixes append with existing update calls on the same production container "
                        f"in {current_path}: {preview}"
                    )
            hunk_lines.clear()
            return ""

        for line in patch_text.splitlines():
            if line.startswith("+++ "):
                reason = flush()
                if reason:
                    return reason
                path = line[4:].strip()
                current_path = path[2:] if path.startswith("b/") else path
                if current_path == "/dev/null":
                    current_path = ""
                continue
            if line.startswith(("diff --git ", "--- ")):
                reason = flush()
                if reason:
                    return reason
                continue
            if line.startswith("@@ "):
                reason = flush()
                if reason:
                    return reason
                continue
            if not current_path.endswith(".py"):
                continue
            if line.startswith("+") and not line.startswith("+++ "):
                hunk_lines.append(("+", line[1:]))
            elif line.startswith("-") and not line.startswith("--- "):
                hunk_lines.append(("-", line[1:]))
            else:
                hunk_lines.append((" ", line[1:] if line.startswith(" ") else line))
        return flush()

    @staticmethod
    def _patch_suspicious_tiny_production_mutation(patch_text: str) -> str:
        current_path = ""
        changed_paths: set[str] = set()
        added_exec: list[tuple[str, str]] = []
        removed_exec: list[tuple[str, str]] = []
        for line in patch_text.splitlines():
            if line.startswith("+++ "):
                path = line[4:].strip()
                current_path = path[2:] if path.startswith("b/") else path
                if current_path == "/dev/null":
                    current_path = ""
                continue
            if not current_path.endswith(".py") or _is_python_test_path(current_path):
                continue
            if line.startswith("+") and not line.startswith("+++ "):
                payload = line[1:]
                if payload.strip() and not _looks_like_non_executable_patch_line(payload):
                    added_exec.append((current_path, payload.strip()))
                    changed_paths.add(current_path)
                continue
            if line.startswith("-") and not line.startswith("--- "):
                payload = line[1:]
                if payload.strip() and not _looks_like_non_executable_patch_line(payload):
                    removed_exec.append((current_path, payload.strip()))
                    changed_paths.add(current_path)
                continue
        if len(changed_paths) != 1:
            return ""
        if len(added_exec) == 1 and not removed_exec:
            path, line = added_exec[0]
            if re.search(r"\[[^\]]+\]\s*\[[^\]]+\]\s*=", line):
                return (
                    "SWE patch is a single-line nested production mutation without surrounding repair structure: "
                    f"{path}: {_python_line_preview(line)}"
                )
        return ""

    @staticmethod
    def _patch_isolated_one_line_production_python_replacement(patch_text: str) -> str:
        current_path = ""
        changed_paths: set[str] = set()
        added: list[tuple[str, str]] = []
        removed: list[tuple[str, str]] = []
        for line in patch_text.splitlines():
            if line.startswith("+++ "):
                path = line[4:].strip()
                current_path = path[2:] if path.startswith("b/") else path
                if current_path == "/dev/null":
                    current_path = ""
                continue
            if line.startswith("+") and not line.startswith("+++ "):
                if current_path:
                    added.append((current_path, line[1:]))
                    changed_paths.add(current_path)
                continue
            if line.startswith("-") and not line.startswith("--- "):
                if current_path:
                    removed.append((current_path, line[1:]))
                    changed_paths.add(current_path)
                continue
        if len(added) != 1 or len(removed) != 1 or len(changed_paths) != 1:
            return ""
        path = next(iter(changed_paths))
        if not path.endswith(".py") or _is_python_test_path(path) or _is_disallowed_swe_solution_path(path):
            return ""
        old_line = removed[0][1].strip()
        new_line = added[0][1].strip()
        if not old_line or not new_line:
            return ""
        if old_line.startswith("#") or new_line.startswith("#"):
            return ""
        old_assignment = re.match(r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=", old_line)
        new_assignment = re.match(r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=", new_line)
        if (
            old_assignment
            and new_assignment
            and old_assignment.group("name") == new_assignment.group("name")
            and old_line != new_line
        ):
            old_value = old_line.split("=", 1)[1].strip()
            new_value = new_line.split("=", 1)[1].strip()
            if _looks_like_python_literal_constant(new_value) and not _looks_like_python_literal_constant(old_value):
                return (
                    "SWE patch replaces computed production assignment with a literal constant without enough "
                    f"repair structure: {path}: {_python_line_preview(old_line)} -> {_python_line_preview(new_line)}"
                )
            return (
                "SWE patch is an isolated one-line production Python assignment replacement without enough "
                f"repair structure: {path}: {_python_line_preview(old_line)} -> {_python_line_preview(new_line)}"
            )
        return (
            "SWE patch is an isolated one-line production Python replacement without enough "
            f"repair structure: {path}: {_python_line_preview(old_line)} -> {_python_line_preview(new_line)}"
        )

    @staticmethod
    def _patch_suspicious_non_executable_to_code_replacement(patch_text: str) -> str:
        current_path = ""
        removed: list[str] = []
        added: list[str] = []

        def flush() -> str:
            if not current_path.endswith(".py") or _is_python_test_path(current_path):
                removed.clear()
                added.clear()
                return ""
            removed_payload = [line.strip() for line in removed if line.strip()]
            added_payload = [line.strip() for line in added if line.strip()]
            if not removed_payload or not added_payload:
                removed.clear()
                added.clear()
                return ""
            removed_non_executable = all(_looks_like_non_executable_patch_line(line) for line in removed_payload)
            added_executable = [line for line in added_payload if not _looks_like_non_executable_patch_line(line)]
            if added_payload == ["pass"] and all(
                _looks_like_non_executable_patch_line(line) or _looks_like_plain_doc_text_patch_line(line)
                for line in removed_payload
            ):
                removed.clear()
                added.clear()
                return f"SWE patch replaces documentation-only production context with bare pass in {current_path}"
            if removed_non_executable and added_executable:
                preview = " | ".join(_python_line_preview(line) for line in added_executable[:3])
                removed.clear()
                added.clear()
                return (
                    "SWE patch replaces only non-executable production context with executable code "
                    f"in {current_path}: {preview}"
                )
            removed.clear()
            added.clear()
            return ""

        for line in patch_text.splitlines():
            if line.startswith("+++ "):
                reason = flush()
                if reason:
                    return reason
                path = line[4:].strip()
                current_path = path[2:] if path.startswith("b/") else path
                if current_path == "/dev/null":
                    current_path = ""
                continue
            if line.startswith(("diff --git ", "@@ ", "--- ")):
                reason = flush()
                if reason:
                    return reason
                continue
            if line.startswith("-"):
                removed.append(line[1:])
                continue
            if line.startswith("+"):
                added.append(line[1:])
                continue
            reason = flush()
            if reason:
                return reason
        return flush()

    @staticmethod
    def _patch_has_placeholder_output_replacement(patch_text: str) -> bool:
        added = [line[1:].strip() for line in patch_text.splitlines() if line.startswith("+") and not line.startswith("+++ ")]
        removed = [line[1:].strip() for line in patch_text.splitlines() if line.startswith("-") and not line.startswith("--- ")]
        if not added or not removed:
            return False
        placeholder_output_patterns = (
            r"^(print|click\.echo|click\.secho|logger\.\w+)\(\s*['\"]patch (applied|works|done|fixed)['\"]",
            r"^(print|click\.echo|click\.secho|logger\.\w+)\(\s*['\"](success|done|fixed)['\"]\s*\)",
        )
        if not any(re.search(pattern, line, flags=re.IGNORECASE) for pattern in placeholder_output_patterns for line in added):
            return False
        return any(
            re.search(r"\b(exception|error|warning|failed|raise|return|logger|click\.secho|click\.echo)\b", line, flags=re.IGNORECASE)
            for line in removed
        )

    @staticmethod
    def _patch_adds_placeholder_success_print(patch_text: str) -> bool:
        for line in Verifier._patch_added_lines(patch_text):
            stripped = line.strip()
            if re.match(
                r"^print\(\s*['\"](?:patch applied|success|done|fixed|works)['\"]\s*\)\s*$",
                stripped,
                flags=re.IGNORECASE,
            ):
                return True
        return False

    @staticmethod
    def _patch_suspicious_non_python_language_mismatch(patch_text: str) -> str:
        current_path = ""
        for line in patch_text.splitlines():
            if line.startswith("+++ "):
                path = line[4:].strip()
                current_path = path[2:] if path.startswith("b/") else path
                if current_path == "/dev/null":
                    current_path = ""
                continue
            if not line.startswith("+") or line.startswith("+++ "):
                continue
            if not current_path.endswith((".sh", ".bash", ".zsh", ".fish")):
                continue
            payload = line[1:].strip()
            if re.search(r"\b[A-Za-z_][\w]*\.[A-Za-z_][\w]*\(", payload) or re.search(r"^(from|import)\s+\w+", payload):
                return f"SWE patch inserts Python-looking code into non-Python file {current_path}: {_python_line_preview(payload)}"
        return ""

    @staticmethod
    def _patch_suspicious_semantic_token_flips(patch_text: str) -> str:
        current_path = ""
        removed: list[str] = []
        added: list[str] = []

        def flush() -> str:
            if removed and added:
                details = _suspicious_semantic_token_flip_details(removed, added)
                if details:
                    preview = ", ".join(details[:3])
                    return f"SWE patch makes suspicious semantic token flips in {current_path or 'patch.diff'}: {preview}"
            removed.clear()
            added.clear()
            return ""

        for line in patch_text.splitlines():
            if line.startswith("+++ "):
                reason = flush()
                if reason:
                    return reason
                path = line[4:].strip()
                current_path = path[2:] if path.startswith("b/") else path
                if current_path == "/dev/null":
                    current_path = ""
                continue
            if line.startswith(("diff --git ", "@@ ", "--- ")):
                reason = flush()
                if reason:
                    return reason
                continue
            if line.startswith("-"):
                removed.append(line[1:])
                continue
            if line.startswith("+"):
                added.append(line[1:])
                continue
            reason = flush()
            if reason:
                return reason
        return flush()

    @staticmethod
    def _patch_suspicious_config_key_replacements(patch_text: str) -> str:
        current_path = ""
        removed: list[str] = []
        added: list[str] = []

        def flush() -> str:
            if current_path.endswith((".yaml", ".yml", ".toml", ".ini")) and removed and added:
                details = _suspicious_config_key_replacement_details(removed, added)
                if details:
                    preview = ", ".join(details[:3])
                    return f"SWE patch makes suspicious config key replacements in {current_path}: {preview}"
            removed.clear()
            added.clear()
            return ""

        for line in patch_text.splitlines():
            if line.startswith("+++ "):
                reason = flush()
                if reason:
                    return reason
                path = line[4:].strip()
                current_path = path[2:] if path.startswith("b/") else path
                if current_path == "/dev/null":
                    current_path = ""
                continue
            if line.startswith(("diff --git ", "@@ ", "--- ")):
                reason = flush()
                if reason:
                    return reason
                continue
            if line.startswith("-"):
                removed.append(line[1:])
                continue
            if line.startswith("+"):
                added.append(line[1:])
                continue
            reason = flush()
            if reason:
                return reason
        return flush()

    @staticmethod
    def _patch_suspicious_duplicate_surrounding_call_wrappers(patch_text: str) -> str:
        current_path = ""
        hunk_lines: list[tuple[str, str]] = []

        def flush() -> str:
            if current_path.endswith(".py") and not _is_python_test_path(current_path):
                details = _python_duplicate_surrounding_call_wrapper_details(hunk_lines)
                if details:
                    preview = ", ".join(details[:3])
                    return f"SWE patch duplicates surrounding call wrappers in {current_path}: {preview}"
            hunk_lines.clear()
            return ""

        for line in patch_text.splitlines():
            if line.startswith("+++ "):
                reason = flush()
                if reason:
                    return reason
                path = line[4:].strip()
                current_path = path[2:] if path.startswith("b/") else path
                if current_path == "/dev/null":
                    current_path = ""
                continue
            if line.startswith(("diff --git ", "--- ")):
                reason = flush()
                if reason:
                    return reason
                continue
            if line.startswith("@@ "):
                reason = flush()
                if reason:
                    return reason
                continue
            if not current_path.endswith(".py"):
                continue
            if line.startswith("+") and not line.startswith("+++ "):
                hunk_lines.append(("+", line[1:]))
            elif line.startswith("-") and not line.startswith("--- "):
                hunk_lines.append(("-", line[1:]))
            else:
                hunk_lines.append((" ", line[1:] if line.startswith(" ") else line))
        return flush()

    @staticmethod
    def _patch_suspicious_duplicate_existing_python_statements(patch_text: str) -> str:
        current_path = ""
        hunk_lines: list[tuple[str, str]] = []

        def flush() -> str:
            if current_path.endswith(".py") and not _is_python_test_path(current_path):
                details = _python_duplicate_existing_statement_replacement_details(hunk_lines)
                if details:
                    preview = ", ".join(details[:3])
                    return f"SWE patch duplicates existing Python statements in {current_path}: {preview}"
            hunk_lines.clear()
            return ""

        for line in patch_text.splitlines():
            if line.startswith("+++ "):
                reason = flush()
                if reason:
                    return reason
                path = line[4:].strip()
                current_path = path[2:] if path.startswith("b/") else path
                if current_path == "/dev/null":
                    current_path = ""
                continue
            if line.startswith(("diff --git ", "--- ")):
                reason = flush()
                if reason:
                    return reason
                continue
            if line.startswith("@@ "):
                reason = flush()
                if reason:
                    return reason
                continue
            if not current_path.endswith(".py"):
                continue
            if line.startswith("+") and not line.startswith("+++ "):
                hunk_lines.append(("+", line[1:]))
            elif line.startswith("-") and not line.startswith("--- "):
                hunk_lines.append(("-", line[1:]))
            else:
                hunk_lines.append((" ", line[1:] if line.startswith(" ") else line))
        return flush()

    @staticmethod
    def _patch_suspicious_python_hunk_replacements(patch_text: str) -> str:
        current_path = ""
        hunk_lines: list[tuple[str, str]] = []

        def flush() -> str:
            if current_path.endswith(".py") and not _is_python_test_path(current_path):
                details = _python_suspicious_hunk_replacement_details(hunk_lines)
                if details:
                    preview = ", ".join(details[:3])
                    return f"SWE patch makes suspicious Python hunk replacements in {current_path}: {preview}"
            hunk_lines.clear()
            return ""

        for line in patch_text.splitlines():
            if line.startswith("+++ "):
                reason = flush()
                if reason:
                    return reason
                path = line[4:].strip()
                current_path = path[2:] if path.startswith("b/") else path
                if current_path == "/dev/null":
                    current_path = ""
                continue
            if line.startswith(("diff --git ", "--- ")):
                reason = flush()
                if reason:
                    return reason
                continue
            if line.startswith("@@ "):
                reason = flush()
                if reason:
                    return reason
                continue
            if not current_path.endswith(".py"):
                continue
            if line.startswith("+") and not line.startswith("+++ "):
                hunk_lines.append(("+", line[1:]))
            elif line.startswith("-") and not line.startswith("--- "):
                hunk_lines.append(("-", line[1:]))
            else:
                hunk_lines.append((" ", line[1:] if line.startswith(" ") else line))
        return flush()

    @staticmethod
    def _patch_suspicious_text_template_replacements(patch_text: str) -> str:
        current_path = ""
        removed: list[str] = []
        added: list[str] = []

        def flush() -> str:
            if current_path.endswith((".html", ".textfsm", ".template", ".tpl", ".jinja", ".jinja2", ".j2")) and removed and added:
                details = _suspicious_text_template_replacement_details(removed, added)
                if details:
                    preview = ", ".join(details[:3])
                    return f"SWE patch makes suspicious text template replacements in {current_path}: {preview}"
            removed.clear()
            added.clear()
            return ""

        for line in patch_text.splitlines():
            if line.startswith("+++ "):
                reason = flush()
                if reason:
                    return reason
                path = line[4:].strip()
                current_path = path[2:] if path.startswith("b/") else path
                if current_path == "/dev/null":
                    current_path = ""
                continue
            if line.startswith(("diff --git ", "@@ ", "--- ")):
                reason = flush()
                if reason:
                    return reason
                continue
            if line.startswith("-"):
                removed.append(line[1:])
                continue
            if line.startswith("+"):
                added.append(line[1:])
                continue
            reason = flush()
            if reason:
                return reason
        return flush()

    @staticmethod
    def _patch_only_changes_disallowed_swe_paths(patch_text: str) -> str:
        paths = Verifier._patch_changed_paths(patch_text)
        if not paths:
            return ""
        disallowed = [path for path in paths if _is_disallowed_swe_solution_path(path)]
        if len(disallowed) == len(paths):
            preview = ", ".join(disallowed[:5])
            return f"SWE patch changes only tests or auxiliary update artifacts: {preview}"
        return ""

    @staticmethod
    def _patch_suspicious_python_statement_replacements(patch_text: str) -> str:
        current_path = ""
        removed: list[str] = []
        added: list[str] = []

        def flush() -> str:
            if (
                current_path.endswith(".py")
                and not _is_python_test_path(current_path)
                and removed
                and added
            ):
                details = _python_suspicious_line_replacement_details(removed, added)
                if details:
                    preview = ", ".join(details[:3])
                    return f"SWE patch makes suspicious Python statement-kind replacements in {current_path}: {preview}"
            removed.clear()
            added.clear()
            return ""

        for line in patch_text.splitlines():
            if line.startswith("+++ "):
                reason = flush()
                if reason:
                    return reason
                path = line[4:].strip()
                if path.startswith("b/"):
                    path = path[2:]
                current_path = path if path != "/dev/null" else ""
                continue
            if line.startswith(("diff --git ", "@@ ", "--- ")):
                reason = flush()
                if reason:
                    return reason
                continue
            if line.startswith("-"):
                removed.append(line[1:])
                continue
            if line.startswith("+"):
                added.append(line[1:])
                continue
            reason = flush()
            if reason:
                return reason
        return flush()

    @staticmethod
    def _swe_repo_cache_path(repo_cache_root: str, repo: str) -> Path | None:
        root = Path(repo_cache_root)
        candidates = [
            root / repo,
            root / repo.replace("/", "__"),
            root / repo.split("/")[-1],
        ]
        for candidate in candidates:
            if candidate.exists() and candidate.is_dir():
                return candidate
        return None

    def _verify_repo_chore_review(self, workspace: Path, contract: dict[str, object]) -> list[str]:
        reasons: list[str] = []
        report_rules = contract.get("report_rules", [])
        if not isinstance(report_rules, list):
            return ["semantic verifier contract malformed: report_rules must be a list"]
        for rule in report_rules:
            if not isinstance(rule, dict):
                reasons.append("semantic verifier contract malformed: report rule must be an object")
                continue
            relative_path = str(rule.get("path", "")).strip()
            if not relative_path:
                reasons.append("semantic verifier contract malformed: report rule missing path")
                continue
            path = workspace / relative_path
            if not path.exists():
                reasons.append(f"semantic report missing: {relative_path}")
                continue
            content = path.read_text(encoding="utf-8").strip().lower()
            for phrase in rule.get("must_mention", []):
                expected = str(phrase).strip().lower()
                if expected and expected not in content:
                    reasons.append(f"semantic report missing phrase {expected!r}: {relative_path}")
            for covered_path in rule.get("covers", []):
                target = str(covered_path).strip()
                if not target:
                    continue
                if not _report_mentions_path(content, target):
                    reasons.append(f"semantic report does not cover {target}: {relative_path}")
        return reasons

    def _verify_git_repo_review(self, workspace: Path, contract: dict[str, object]) -> list[str]:
        reasons = self._verify_repo_chore_review(workspace, contract)
        git_dir = workspace / ".git"
        if not git_dir.exists():
            return reasons + ["git repository missing: .git"]

        expected_branch = str(contract.get("expected_branch", "")).strip()
        if expected_branch:
            branch = self._git_output(workspace, "branch", "--show-current")
            if branch is None:
                reasons.append("git branch inspection failed")
            elif branch.strip() != expected_branch:
                reasons.append(f"git branch mismatch: expected {expected_branch!r} got {branch.strip()!r}")

        diff_base_ref = str(contract.get("diff_base_ref", "")).strip()
        resolved_diff_base_ref = self._resolved_git_diff_base_ref(workspace, diff_base_ref or None)
        expected_changed_paths = [
            str(path).strip()
            for path in contract.get("expected_changed_paths", [])
            if str(path).strip()
        ]
        required_merged_branches = [
            str(path).strip()
            for path in contract.get("required_merged_branches", [])
            if str(path).strip()
        ]
        if expected_changed_paths:
            changed_paths = self._git_changed_paths(workspace, base_ref=resolved_diff_base_ref)
            expected_set = set(expected_changed_paths)
            allowed_required_branch_paths: set[str] = set()
            if resolved_diff_base_ref and required_merged_branches:
                for branch_name in required_merged_branches:
                    allowed_required_branch_paths.update(
                        self._git_changed_paths_between(
                            workspace,
                            start_ref=resolved_diff_base_ref,
                            end_ref=branch_name,
                        )
                    )
            actual_set = set(changed_paths)
            missing = sorted(expected_set - actual_set)
            unexpected = sorted(actual_set - expected_set - allowed_required_branch_paths)
            for path in missing:
                reasons.append(f"git diff missing expected path: {path}")
            for path in unexpected:
                reasons.append(f"git diff includes unexpected path: {path}")

        current_branch = expected_branch or str(contract.get("target_branch", "")).strip()
        for normalized in required_merged_branches:
            if not current_branch:
                reasons.append(
                    f"semantic verifier contract malformed: required_merged_branches needs expected_branch or target_branch"
                )
                break
            merged = self._git_returncode(
                workspace,
                "merge-base",
                "--is-ancestor",
                normalized,
                current_branch,
            )
            if merged != 0:
                reasons.append(f"required worker branch not accepted into {current_branch}: {normalized}")

        for preserved_path in contract.get("preserved_paths", []):
            normalized = str(preserved_path).strip()
            if normalized and normalized in self._git_changed_paths(workspace, base_ref=resolved_diff_base_ref):
                reasons.append(f"git diff unexpectedly changed preserved path: {normalized}")

        generated_paths = {
            str(path).strip()
            for path in contract.get("generated_paths", [])
            if str(path).strip()
        }
        changed_paths_for_generated = set(self._git_changed_paths(workspace, base_ref=resolved_diff_base_ref))
        for generated_path in sorted(generated_paths):
            if not (workspace / generated_path).exists():
                reasons.append(f"generated artifact missing: {generated_path}")
            elif generated_path not in changed_paths_for_generated:
                reasons.append(f"generated artifact not recorded in git diff: {generated_path}")

        unresolved_paths = self._git_unmerged_paths(workspace)
        if unresolved_paths:
            for path in unresolved_paths:
                reasons.append(f"git conflict remains unresolved: {path}")

        conflict_paths = [
            str(path).strip()
            for path in contract.get("resolved_conflict_paths", [])
            if str(path).strip()
        ]
        for conflict_path in conflict_paths:
            if self._file_contains_conflict_markers(workspace / conflict_path):
                reasons.append(f"conflict markers still present after merge resolution: {conflict_path}")

        if bool(contract.get("clean_worktree", False)):
            status_output = self._git_output(workspace, "status", "--porcelain")
            if status_output is None:
                reasons.append("git status inspection failed")
            elif status_output.strip():
                reasons.append("git worktree not clean after merge acceptance")

        for test_rule in contract.get("test_commands", []):
            if not isinstance(test_rule, dict):
                reasons.append("semantic verifier contract malformed: test command rule must be an object")
                continue
            argv = [str(value) for value in test_rule.get("argv", []) if str(value).strip()]
            if not argv:
                reasons.append("semantic verifier contract malformed: test command rule missing argv")
                continue
            label = str(test_rule.get("label", "test command")).strip() or "test command"
            try:
                completed = subprocess.run(
                    argv,
                    cwd=str(workspace),
                    text=True,
                    capture_output=True,
                    timeout=10,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                reasons.append(f"{label} failed to execute: {exc}")
                continue
            if completed.returncode != 0:
                reasons.append(f"{label} exited with code {completed.returncode}")
        return reasons

    def _verify_behavior_checks(self, workspace: Path, contract: dict[str, object]) -> list[str]:
        reasons: list[str] = []
        checks = contract.get("behavior_checks", [])
        if not isinstance(checks, list):
            return ["semantic verifier contract malformed: behavior_checks must be a list"]
        for rule in checks:
            if not isinstance(rule, dict):
                reasons.append("semantic verifier contract malformed: behavior check must be an object")
                continue
            argv = [str(value).strip() for value in rule.get("argv", []) if str(value).strip()]
            if not argv:
                reasons.append("semantic verifier contract malformed: behavior check missing argv")
                continue
            label = str(rule.get("label", "behavior check")).strip() or "behavior check"
            completed = self._run_semantic_command(
                workspace,
                argv=argv,
                cwd=str(rule.get("cwd", "")).strip(),
                timeout_seconds=self._semantic_timeout_seconds(rule, default=10),
            )
            if isinstance(completed, str):
                reasons.append(f"{label} failed to execute: {completed}")
                continue
            expected_exit_code = self._semantic_expected_exit_code(rule, default=0)
            if completed.returncode != expected_exit_code:
                reasons.append(
                    f"{label} exited with code {completed.returncode} (expected {expected_exit_code})"
                )
            reasons.extend(
                self._assert_command_output(
                    completed.stdout,
                    completed.stderr,
                    rule,
                    label=label,
                )
            )
            file_expectations = rule.get("file_expectations", [])
            if file_expectations is not None:
                if not isinstance(file_expectations, list):
                    reasons.append(f"semantic verifier contract malformed: {label} file_expectations must be a list")
                else:
                    reasons.extend(
                        self._apply_file_expectations(
                            workspace,
                            file_expectations,
                            label=label,
                        )
                    )
            repo_invariants = rule.get("repo_invariants", [])
            if repo_invariants is not None:
                if not isinstance(repo_invariants, list):
                    reasons.append(f"semantic verifier contract malformed: {label} repo_invariants must be a list")
                else:
                    reasons.extend(self._verify_repo_invariants(workspace, {"repo_invariants": repo_invariants}))
        return reasons

    def _verify_differential_checks(self, workspace: Path, contract: dict[str, object]) -> list[str]:
        reasons: list[str] = []
        checks = contract.get("differential_checks", [])
        if not isinstance(checks, list):
            return ["semantic verifier contract malformed: differential_checks must be a list"]
        for rule in checks:
            if not isinstance(rule, dict):
                reasons.append("semantic verifier contract malformed: differential check must be an object")
                continue
            candidate_argv = [str(value).strip() for value in rule.get("candidate_argv", []) if str(value).strip()]
            baseline_argv = [str(value).strip() for value in rule.get("baseline_argv", []) if str(value).strip()]
            if not candidate_argv or not baseline_argv:
                reasons.append("semantic verifier contract malformed: differential check missing candidate_argv or baseline_argv")
                continue
            label = str(rule.get("label", "differential check")).strip() or "differential check"
            timeout_seconds = self._semantic_timeout_seconds(rule, default=10)
            try:
                with tempfile.TemporaryDirectory(prefix="agentkernel-differential-") as temp_root:
                    temp_root_path = Path(temp_root)
                    candidate_workspace = self._prepare_differential_workspace(
                        workspace,
                        destination=temp_root_path / "candidate",
                    )
                    baseline_workspace = self._prepare_differential_workspace(
                        workspace,
                        destination=temp_root_path / "baseline",
                    )
                    candidate = self._run_semantic_command(
                        candidate_workspace,
                        argv=candidate_argv,
                        cwd=str(rule.get("cwd", "")).strip(),
                        timeout_seconds=timeout_seconds,
                    )
                    baseline = self._run_semantic_command(
                        baseline_workspace,
                        argv=baseline_argv,
                        cwd=str(rule.get("cwd", "")).strip(),
                        timeout_seconds=timeout_seconds,
                    )
                    candidate_reasons, baseline_reasons = self._verify_differential_file_expectations(
                        candidate_workspace,
                        baseline_workspace,
                        rule,
                        label=label,
                    )
            except OSError as exc:
                reasons.append(f"{label} differential workspace preparation failed: {exc}")
                continue
            if isinstance(candidate, str):
                reasons.append(f"{label} candidate command failed to execute: {candidate}")
                continue
            if isinstance(baseline, str):
                reasons.append(f"{label} baseline command failed to execute: {baseline}")
                continue
            normalize_whitespace = bool(rule.get("normalize_whitespace", True))
            if "expect_candidate_exit_code" in rule:
                expected_candidate_exit_code = self._semantic_expected_exit_code(
                    {"expect_exit_code": rule.get("expect_candidate_exit_code")},
                    default=0,
                )
                if candidate.returncode != expected_candidate_exit_code:
                    reasons.append(
                        f"{label} candidate exit code {candidate.returncode} (expected {expected_candidate_exit_code})"
                    )
            if "expect_baseline_exit_code" in rule:
                expected_baseline_exit_code = self._semantic_expected_exit_code(
                    {"expect_exit_code": rule.get("expect_baseline_exit_code")},
                    default=0,
                )
                if baseline.returncode != expected_baseline_exit_code:
                    reasons.append(
                        f"{label} baseline exit code {baseline.returncode} (expected {expected_baseline_exit_code})"
                    )
            if bool(rule.get("expect_same_exit_code", True)) and candidate.returncode != baseline.returncode:
                reasons.append(
                    f"{label} exit code mismatch: candidate {candidate.returncode} baseline {baseline.returncode}"
                )
            if bool(rule.get("expect_same_stdout", False)) and self._normalized_output(
                candidate.stdout,
                normalize_whitespace=normalize_whitespace,
            ) != self._normalized_output(
                baseline.stdout,
                normalize_whitespace=normalize_whitespace,
            ):
                reasons.append(f"{label} stdout differed from baseline")
            if bool(rule.get("expect_same_stderr", False)) and self._normalized_output(
                candidate.stderr,
                normalize_whitespace=normalize_whitespace,
            ) != self._normalized_output(
                baseline.stderr,
                normalize_whitespace=normalize_whitespace,
            ):
                reasons.append(f"{label} stderr differed from baseline")
            if bool(rule.get("expect_stdout_difference", False)) and self._normalized_output(
                candidate.stdout,
                normalize_whitespace=normalize_whitespace,
            ) == self._normalized_output(
                baseline.stdout,
                normalize_whitespace=normalize_whitespace,
            ):
                reasons.append(f"{label} stdout unexpectedly matched baseline")
            if bool(rule.get("expect_stderr_difference", False)) and self._normalized_output(
                candidate.stderr,
                normalize_whitespace=normalize_whitespace,
            ) == self._normalized_output(
                baseline.stderr,
                normalize_whitespace=normalize_whitespace,
            ):
                reasons.append(f"{label} stderr unexpectedly matched baseline")
            reasons.extend(
                self._assert_command_output(
                    candidate.stdout,
                    candidate.stderr,
                    {
                        "stdout_must_contain": rule.get("candidate_stdout_must_contain", []),
                        "stdout_must_not_contain": rule.get("candidate_stdout_must_not_contain", []),
                        "stderr_must_contain": rule.get("candidate_stderr_must_contain", []),
                        "stderr_must_not_contain": rule.get("candidate_stderr_must_not_contain", []),
                        "stdout_json_fields": rule.get("candidate_stdout_json_fields", []),
                        "stderr_json_fields": rule.get("candidate_stderr_json_fields", []),
                    },
                    label=f"{label} candidate",
                )
            )
            reasons.extend(
                self._assert_command_output(
                    baseline.stdout,
                    baseline.stderr,
                    {
                        "stdout_must_contain": rule.get("baseline_stdout_must_contain", []),
                        "stdout_must_not_contain": rule.get("baseline_stdout_must_not_contain", []),
                        "stderr_must_contain": rule.get("baseline_stderr_must_contain", []),
                        "stderr_must_not_contain": rule.get("baseline_stderr_must_not_contain", []),
                        "stdout_json_fields": rule.get("baseline_stdout_json_fields", []),
                        "stderr_json_fields": rule.get("baseline_stderr_json_fields", []),
                    },
                    label=f"{label} baseline",
                )
            )
            reasons.extend(candidate_reasons)
            reasons.extend(baseline_reasons)
        return reasons

    def _verify_repo_invariants(self, workspace: Path, contract: dict[str, object]) -> list[str]:
        reasons: list[str] = []
        invariants = contract.get("repo_invariants", [])
        if not isinstance(invariants, list):
            return ["semantic verifier contract malformed: repo_invariants must be a list"]
        for rule in invariants:
            if not isinstance(rule, dict):
                reasons.append("semantic verifier contract malformed: repo invariant must be an object")
                continue
            kind = str(rule.get("kind", "")).strip()
            if kind == "file_contains":
                reasons.extend(self._verify_file_contains_invariant(workspace, rule))
            elif kind == "file_unchanged":
                reasons.extend(self._verify_file_unchanged_invariant(workspace, rule))
            elif kind == "git_clean":
                reasons.extend(self._verify_git_clean_invariant(workspace, rule))
            elif kind == "git_no_unmerged":
                unresolved_paths = self._git_unmerged_paths(workspace)
                for path in unresolved_paths:
                    reasons.append(f"repo invariant violated: unresolved merge remains at {path}")
            elif kind == "git_tracked_paths":
                paths = [str(value).strip() for value in rule.get("paths", []) if str(value).strip()]
                if not paths:
                    reasons.append("semantic verifier contract malformed: git_tracked_paths invariant missing paths")
                    continue
                for path in paths:
                    if self._git_returncode(workspace, "ls-files", "--error-unmatch", path) != 0:
                        reasons.append(f"repo invariant violated: path is not tracked by git: {path}")
            else:
                reasons.append(f"semantic verifier contract malformed: unknown repo invariant kind {kind!r}")
        return reasons

    def _verify_semantic_assertions(
        self,
        workspace: Path,
        contract: dict[str, object],
        *,
        stdout: str,
        stderr: str,
    ) -> list[str]:
        assertions = contract.get("semantic_assertions", [])
        if assertions is None:
            return []
        if not isinstance(assertions, list):
            return ["semantic verifier contract malformed: semantic_assertions must be a list"]
        reasons: list[str] = []
        for rule in assertions:
            if not isinstance(rule, dict):
                reasons.append("semantic verifier contract malformed: semantic assertion must be an object")
                continue
            label = str(rule.get("label", "semantic assertion")).strip() or "semantic assertion"
            source = str(rule.get("source", "")).strip()
            if source == "stdout_text":
                reasons.extend(self._assert_text_semantics(stdout, rule, label=label))
            elif source == "stderr_text":
                reasons.extend(self._assert_text_semantics(stderr, rule, label=label))
            elif source == "stdout_json":
                reasons.extend(
                    self._assert_json_fields_from_text(
                        stdout,
                        rule.get("json_fields", []),
                        label=label,
                    )
                )
            elif source == "stderr_json":
                reasons.extend(
                    self._assert_json_fields_from_text(
                        stderr,
                        rule.get("json_fields", []),
                        label=label,
                    )
                )
            elif source == "workspace_file_text":
                path = str(rule.get("path", "")).strip()
                if not path:
                    reasons.append(f"semantic verifier contract malformed: {label} workspace_file_text missing path")
                    continue
                target = workspace / path
                if not target.exists():
                    reasons.append(f"{label} missing workspace file {path}")
                    continue
                reasons.extend(self._assert_text_semantics(target.read_text(encoding="utf-8"), rule, label=label))
            elif source == "workspace_file_json":
                path = str(rule.get("path", "")).strip()
                if not path:
                    reasons.append(f"semantic verifier contract malformed: {label} workspace_file_json missing path")
                    continue
                target = workspace / path
                if not target.exists():
                    reasons.append(f"{label} missing workspace file {path}")
                    continue
                reasons.extend(
                    self._assert_json_fields_from_text(
                        target.read_text(encoding="utf-8"),
                        rule.get("json_fields", []),
                        label=label,
                    )
                )
            elif source == "git_diff":
                text = self._git_output(workspace, "diff", "--")
                if text is None:
                    reasons.append(f"{label} git diff inspection failed")
                    continue
                reasons.extend(self._assert_text_semantics(text, rule, label=label))
            elif source == "git_status":
                text = self._git_output(workspace, "status", "--porcelain")
                if text is None:
                    reasons.append(f"{label} git status inspection failed")
                    continue
                reasons.extend(self._assert_text_semantics(text, rule, label=label))
            else:
                reasons.append(f"semantic verifier contract malformed: unknown semantic assertion source {source!r}")
        return reasons

    def _verify_file_contains_invariant(self, workspace: Path, rule: dict[str, object]) -> list[str]:
        reasons: list[str] = []
        path = str(rule.get("path", "")).strip()
        if not path:
            return ["semantic verifier contract malformed: file_contains invariant missing path"]
        target = workspace / path
        if not target.exists():
            return [f"repo invariant violated: missing file {path}"]
        content = target.read_text(encoding="utf-8")
        for needle in rule.get("must_contain", []):
            expected = str(needle).strip()
            if expected and expected not in content:
                reasons.append(f"repo invariant violated: {path} missing required content {expected!r}")
        for needle in rule.get("must_not_contain", []):
            forbidden = str(needle).strip()
            if forbidden and forbidden in content:
                reasons.append(f"repo invariant violated: {path} contains forbidden content {forbidden!r}")
        reasons.extend(
            self._assert_json_fields_from_text(
                content,
                rule.get("json_fields", []),
                label=f"repo invariant {path}",
            )
        )
        return reasons

    def _verify_file_unchanged_invariant(self, workspace: Path, rule: dict[str, object]) -> list[str]:
        path = str(rule.get("path", "")).strip()
        if not path:
            return ["semantic verifier contract malformed: file_unchanged invariant missing path"]
        target = workspace / path
        if not target.exists():
            return [f"repo invariant violated: missing preserved file {path}"]
        if "expected_content" in rule:
            expected_content = str(rule.get("expected_content", ""))
            actual_content = target.read_text(encoding="utf-8")
            if actual_content != expected_content:
                return [f"repo invariant violated: preserved file changed: {path}"]
            return []
        return []

    def _verify_git_clean_invariant(self, workspace: Path, rule: dict[str, object]) -> list[str]:
        status_output = self._git_output(workspace, "status", "--porcelain")
        if status_output is None:
            return ["git status inspection failed"]
        allow_paths = {
            str(value).strip()
            for value in rule.get("allow_paths", [])
            if str(value).strip()
        }
        changed_paths = [line[3:].strip() for line in status_output.splitlines() if line.strip()]
        unexpected = [path for path in changed_paths if path not in allow_paths]
        if unexpected:
            return [f"repo invariant violated: unexpected dirty paths present: {', '.join(sorted(unexpected))}"]
        return []

    def _verify_differential_file_expectations(
        self,
        candidate_workspace: Path,
        baseline_workspace: Path,
        rule: dict[str, object],
        *,
        label: str,
    ) -> tuple[list[str], list[str]]:
        candidate_expectations = rule.get("candidate_file_expectations", [])
        baseline_expectations = rule.get("baseline_file_expectations", [])
        if candidate_expectations is not None and not isinstance(candidate_expectations, list):
            return ([f"semantic verifier contract malformed: {label} candidate_file_expectations must be a list"], [])
        if baseline_expectations is not None and not isinstance(baseline_expectations, list):
            return ([], [f"semantic verifier contract malformed: {label} baseline_file_expectations must be a list"])
        return (
            self._apply_file_expectations(
                candidate_workspace,
                candidate_expectations if isinstance(candidate_expectations, list) else [],
                label=f"{label} candidate",
            ),
            self._apply_file_expectations(
                baseline_workspace,
                baseline_expectations if isinstance(baseline_expectations, list) else [],
                label=f"{label} baseline",
            ),
        )

    def _apply_file_expectations(
        self,
        workspace: Path,
        expectations: list[object],
        *,
        label: str,
    ) -> list[str]:
        reasons: list[str] = []
        for expectation in expectations:
            if not isinstance(expectation, dict):
                reasons.append(f"semantic verifier contract malformed: {label} file expectation must be an object")
                continue
            path = str(expectation.get("path", "")).strip()
            if not path:
                reasons.append(f"semantic verifier contract malformed: {label} file expectation missing path")
                continue
            must_exist = bool(expectation.get("must_exist", True))
            target = workspace / path
            if must_exist and not target.exists():
                reasons.append(f"{label} missing expected file {path}")
                continue
            if not must_exist:
                if target.exists():
                    reasons.append(f"{label} unexpectedly created file {path}")
                continue
            actual_content = target.read_text(encoding="utf-8") if target.exists() else ""
            if "expected_content" in expectation:
                expected_content = str(expectation.get("expected_content", ""))
                if actual_content != expected_content:
                    reasons.append(f"{label} file content mismatch for {path}")
            for needle in expectation.get("must_contain", []):
                normalized = str(needle).strip()
                if normalized and normalized not in actual_content:
                    reasons.append(f"{label} file {path} missing required content {normalized!r}")
            for needle in expectation.get("must_not_contain", []):
                normalized = str(needle).strip()
                if normalized and normalized in actual_content:
                    reasons.append(f"{label} file {path} contains forbidden content {normalized!r}")
            reasons.extend(
                self._assert_json_fields_from_text(
                    actual_content,
                    expectation.get("json_fields", []),
                    label=f"{label} file {path}",
                )
            )
        return reasons

    @staticmethod
    def _prepare_differential_workspace(workspace: Path, *, destination: Path) -> Path:
        shutil.copytree(
            workspace,
            destination,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", ".mypy_cache"),
        )
        return destination

    @staticmethod
    def _semantic_timeout_seconds(rule: dict[str, object], *, default: int) -> int:
        try:
            return max(1, int(rule.get("timeout_seconds", default) or default))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _semantic_expected_exit_code(rule: dict[str, object], *, default: int) -> int:
        try:
            return int(rule.get("expect_exit_code", default))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _assert_command_output(
        stdout: str,
        stderr: str,
        rule: dict[str, object],
        *,
        label: str,
    ) -> list[str]:
        reasons: list[str] = []
        for needle in rule.get("stdout_must_contain", []):
            expected = str(needle).strip()
            if expected and expected not in stdout:
                reasons.append(f"{label} stdout missing {expected!r}")
        for needle in rule.get("stdout_must_not_contain", []):
            forbidden = str(needle).strip()
            if forbidden and forbidden in stdout:
                reasons.append(f"{label} stdout contains forbidden text {forbidden!r}")
        for pattern in rule.get("stdout_regex_must_match", []):
            regex = str(pattern).strip()
            if regex and re.search(regex, stdout, flags=re.MULTILINE) is None:
                reasons.append(f"{label} stdout missing regex match {regex!r}")
        for pattern in rule.get("stdout_regex_must_not_match", []):
            regex = str(pattern).strip()
            if regex and re.search(regex, stdout, flags=re.MULTILINE) is not None:
                reasons.append(f"{label} stdout matched forbidden regex {regex!r}")
        for needle in rule.get("stderr_must_contain", []):
            expected = str(needle).strip()
            if expected and expected not in stderr:
                reasons.append(f"{label} stderr missing {expected!r}")
        for needle in rule.get("stderr_must_not_contain", []):
            forbidden = str(needle).strip()
            if forbidden and forbidden in stderr:
                reasons.append(f"{label} stderr contains forbidden text {forbidden!r}")
        for pattern in rule.get("stderr_regex_must_match", []):
            regex = str(pattern).strip()
            if regex and re.search(regex, stderr, flags=re.MULTILINE) is None:
                reasons.append(f"{label} stderr missing regex match {regex!r}")
        for pattern in rule.get("stderr_regex_must_not_match", []):
            regex = str(pattern).strip()
            if regex and re.search(regex, stderr, flags=re.MULTILINE) is not None:
                reasons.append(f"{label} stderr matched forbidden regex {regex!r}")
        reasons.extend(
            Verifier._assert_json_fields_from_text(
                stdout,
                rule.get("stdout_json_fields", []),
                label=f"{label} stdout",
            )
        )
        reasons.extend(
            Verifier._assert_json_fields_from_text(
                stderr,
                rule.get("stderr_json_fields", []),
                label=f"{label} stderr",
            )
        )
        return reasons

    @staticmethod
    def _assert_text_semantics(
        text: str,
        rule: dict[str, object],
        *,
        label: str,
    ) -> list[str]:
        reasons: list[str] = []
        for needle in rule.get("contains", []):
            normalized = str(needle).strip()
            if normalized and normalized not in text:
                reasons.append(f"{label} missing required text {normalized!r}")
        for needle in rule.get("not_contains", []):
            normalized = str(needle).strip()
            if normalized and normalized in text:
                reasons.append(f"{label} contains forbidden text {normalized!r}")
        regex = str(rule.get("regex", "")).strip()
        if regex and re.search(regex, text, flags=re.MULTILINE) is None:
            reasons.append(f"{label} missing regex match {regex!r}")
        not_regex = str(rule.get("not_regex", "")).strip()
        if not_regex and re.search(not_regex, text, flags=re.MULTILINE) is not None:
            reasons.append(f"{label} matched forbidden regex {not_regex!r}")
        if "line_count_min" in rule:
            try:
                if len(text.splitlines()) < int(rule.get("line_count_min", 0) or 0):
                    reasons.append(
                        f"{label} line count {len(text.splitlines())} below minimum {int(rule.get('line_count_min', 0) or 0)}"
                    )
            except (TypeError, ValueError):
                reasons.append(f"semantic verifier contract malformed: {label} line_count_min must be an integer")
        if "line_count_max" in rule:
            try:
                if len(text.splitlines()) > int(rule.get("line_count_max", 0) or 0):
                    reasons.append(
                        f"{label} line count {len(text.splitlines())} above maximum {int(rule.get('line_count_max', 0) or 0)}"
                    )
            except (TypeError, ValueError):
                reasons.append(f"semantic verifier contract malformed: {label} line_count_max must be an integer")
        capture_regex = str(rule.get("capture_regex", "")).strip()
        if capture_regex:
            match = re.search(capture_regex, text, flags=re.MULTILINE)
            if match is None:
                reasons.append(f"{label} missing capture regex {capture_regex!r}")
            else:
                try:
                    group_index = int(rule.get("capture_group", 1) or 1)
                except (TypeError, ValueError):
                    group_index = 1
                try:
                    captured = float(match.group(group_index))
                except (IndexError, TypeError, ValueError):
                    reasons.append(f"{label} capture regex {capture_regex!r} did not yield a numeric group")
                else:
                    if "min" in rule and captured < float(rule.get("min")):
                        reasons.append(
                            f"{label} captured value {captured!r} below minimum {float(rule.get('min'))!r}"
                        )
                    if "max" in rule and captured > float(rule.get("max")):
                        reasons.append(
                            f"{label} captured value {captured!r} above maximum {float(rule.get('max'))!r}"
                        )
                    if "equals" in rule and captured != float(rule.get("equals")):
                        reasons.append(
                            f"{label} captured value {captured!r} did not equal {float(rule.get('equals'))!r}"
                        )
        return reasons

    @staticmethod
    def _normalized_output(output: str, *, normalize_whitespace: bool) -> str:
        if not normalize_whitespace:
            return output
        return " ".join(str(output).split())

    @staticmethod
    def _assert_json_fields_from_text(
        text: str,
        expectations: object,
        *,
        label: str,
    ) -> list[str]:
        if expectations is None:
            return []
        if not isinstance(expectations, list):
            return [f"semantic verifier contract malformed: {label} json_fields must be a list"]
        if not expectations:
            return []
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            return [f"{label} is not valid JSON: {exc.msg}"]
        reasons: list[str] = []
        for expectation in expectations:
            if not isinstance(expectation, dict):
                reasons.append(f"semantic verifier contract malformed: {label} json field expectation must be an object")
                continue
            path = str(expectation.get("path", "")).strip()
            if not path:
                reasons.append(f"semantic verifier contract malformed: {label} json field expectation missing path")
                continue
            missing, actual = Verifier._json_field_value(payload, path)
            if missing:
                reasons.append(f"{label} JSON path missing: {path}")
                continue
            if "equals" in expectation and actual != expectation.get("equals"):
                reasons.append(
                    f"{label} JSON path {path!r} expected {expectation.get('equals')!r} got {actual!r}"
                )
            if "not_equals" in expectation and actual == expectation.get("not_equals"):
                reasons.append(
                    f"{label} JSON path {path!r} unexpectedly matched {expectation.get('not_equals')!r}"
                )
            if "one_of" in expectation:
                allowed = expectation.get("one_of")
                if not isinstance(allowed, list):
                    reasons.append(f"semantic verifier contract malformed: {label} json field one_of must be a list")
                elif actual not in allowed:
                    reasons.append(
                        f"{label} JSON path {path!r} expected one of {allowed!r} got {actual!r}"
                    )
            if "contains" in expectation:
                needle = expectation.get("contains")
                if isinstance(actual, str):
                    if str(needle) not in actual:
                        reasons.append(
                            f"{label} JSON path {path!r} missing substring {str(needle)!r}"
                        )
                elif isinstance(actual, list):
                    if needle not in actual:
                        reasons.append(f"{label} JSON path {path!r} missing item {needle!r}")
                else:
                    reasons.append(
                        f"semantic verifier contract malformed: {label} json field contains requires string or list at {path}"
                    )
            if "regex" in expectation:
                regex = str(expectation.get("regex", "")).strip()
                if not isinstance(actual, str):
                    reasons.append(
                        f"semantic verifier contract malformed: {label} json field regex requires string at {path}"
                    )
                elif regex and re.search(regex, actual, flags=re.MULTILINE) is None:
                    reasons.append(
                        f"{label} JSON path {path!r} missing regex match {regex!r}"
                    )
            if "not_regex" in expectation:
                regex = str(expectation.get("not_regex", "")).strip()
                if not isinstance(actual, str):
                    reasons.append(
                        f"semantic verifier contract malformed: {label} json field not_regex requires string at {path}"
                    )
                elif regex and re.search(regex, actual, flags=re.MULTILINE) is not None:
                    reasons.append(
                        f"{label} JSON path {path!r} matched forbidden regex {regex!r}"
                    )
            if expectation.get("truthy", False) and not actual:
                reasons.append(f"{label} JSON path {path!r} was not truthy")
            if expectation.get("falsy", False) and actual:
                reasons.append(f"{label} JSON path {path!r} was not falsy")
            if "min" in expectation:
                try:
                    if float(actual) < float(expectation.get("min")):
                        reasons.append(
                            f"{label} JSON path {path!r} expected >= {expectation.get('min')!r} got {actual!r}"
                        )
                except (TypeError, ValueError):
                    reasons.append(
                        f"semantic verifier contract malformed: {label} json field min requires numeric value at {path}"
                    )
            if "max" in expectation:
                try:
                    if float(actual) > float(expectation.get("max")):
                        reasons.append(
                            f"{label} JSON path {path!r} expected <= {expectation.get('max')!r} got {actual!r}"
                        )
                except (TypeError, ValueError):
                    reasons.append(
                        f"semantic verifier contract malformed: {label} json field max requires numeric value at {path}"
                    )
        return reasons

    @staticmethod
    def _json_field_value(payload: object, path: str) -> tuple[bool, object]:
        current = payload
        for part in str(path).split("."):
            token = part.strip()
            if not token:
                return True, None
            if isinstance(current, dict):
                if token not in current:
                    return True, None
                current = current[token]
                continue
            if isinstance(current, list):
                try:
                    index = int(token)
                except ValueError:
                    return True, None
                if index < 0 or index >= len(current):
                    return True, None
                current = current[index]
                continue
            return True, None
        return False, current

    @staticmethod
    def _run_semantic_command(
        workspace: Path,
        *,
        argv: list[str],
        cwd: str,
        timeout_seconds: int,
    ):
        run_cwd = workspace / cwd if cwd else workspace
        try:
            return subprocess.run(
                argv,
                cwd=str(run_cwd),
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return str(exc)

    @staticmethod
    def _git_output(workspace: Path, *args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=str(workspace),
                text=True,
                capture_output=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if completed.returncode != 0:
            return None
        return completed.stdout.strip()

    @staticmethod
    def _git_returncode(workspace: Path, *args: str) -> int | None:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=str(workspace),
                text=True,
                capture_output=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return completed.returncode

    def _git_changed_paths(self, workspace: Path, *, base_ref: str | None = None) -> list[str]:
        if base_ref:
            diff_output = self._git_output(workspace, "diff", "--name-only", "--relative", f"{base_ref}..HEAD")
            return sorted(line.strip() for line in (diff_output or "").splitlines() if line.strip())
        diff_output = self._git_output(workspace, "diff", "--name-only", "--relative")
        untracked_output = self._git_output(workspace, "ls-files", "--others", "--exclude-standard")
        changed = {
            line.strip()
            for line in ((diff_output or "").splitlines() + (untracked_output or "").splitlines())
            if line.strip()
        }
        return sorted(changed)

    def _git_changed_paths_between(self, workspace: Path, *, start_ref: str, end_ref: str) -> list[str]:
        diff_output = self._git_output(workspace, "diff", "--name-only", "--relative", f"{start_ref}..{end_ref}")
        return sorted(line.strip() for line in (diff_output or "").splitlines() if line.strip())

    def _resolved_git_diff_base_ref(self, workspace: Path, base_ref: str | None) -> str | None:
        normalized = str(base_ref or "").strip()
        if not normalized:
            return None
        if self._git_returncode(workspace, "rev-parse", "--verify", "--quiet", normalized) == 0:
            return normalized
        # Shared-repo fixtures historically use a logical "baseline" ref; if it is absent in the
        # live clone, the repository root commit is the intended semantic baseline.
        if normalized == "baseline":
            root_commit = self._git_output(workspace, "rev-list", "--max-parents=0", "HEAD")
            resolved_root = root_commit.splitlines()[-1].strip() if root_commit else ""
            if resolved_root:
                return resolved_root
        return normalized

    def _git_unmerged_paths(self, workspace: Path) -> list[str]:
        output = self._git_output(workspace, "diff", "--name-only", "--diff-filter=U")
        return sorted(line.strip() for line in (output or "").splitlines() if line.strip())

    @staticmethod
    def _file_contains_conflict_markers(path: Path) -> bool:
        if not path.exists() or not path.is_file():
            return False
        content = path.read_text(encoding="utf-8")
        return all(marker in content for marker in ("<<<<<<<", "=======", ">>>>>>>"))


def _report_mentions_path(content: str, relative_path: str) -> bool:
    tokens = _semantic_path_tokens(relative_path)
    return any(token in content for token in tokens)


def _semantic_path_tokens(relative_path: str) -> set[str]:
    path = Path(relative_path)
    tokens = {relative_path.lower()}
    for part in path.parts:
        normalized = part.strip().lower()
        if not normalized:
            continue
        tokens.add(normalized)
        stem = Path(normalized).stem
        if stem:
            tokens.add(stem)
        tokens.update(token for token in re.split(r"[^a-z0-9]+", stem) if token)
    return {token for token in tokens if token}


def _looks_like_non_executable_patch_line(line: str) -> bool:
    stripped = str(line).strip()
    if not stripped:
        return True
    if stripped.startswith("#"):
        return True
    if stripped.startswith(('"""', "'''")) or stripped.endswith(('"""', "'''")):
        return True
    if stripped.startswith(":"):
        return True
    if stripped.lower().startswith("author:"):
        return True
    return False


def _looks_like_plain_doc_text_patch_line(line: str) -> bool:
    stripped = str(line).strip()
    if not stripped:
        return True
    if re.match(r"^(from|import|class|def|return|raise|if|elif|else|for|while|try|except|with|async|await|pass|yield)\b", stripped):
        return False
    if re.search(r"(^|[A-Za-z_][A-Za-z0-9_]*)\s*(=|\(|\[|\{)", stripped):
        return False
    return bool(re.search(r"[A-Za-z]", stripped))


def _python_executable_ast_changed(before_source: str, after_source: str) -> bool:
    try:
        before_tree = ast.parse(before_source)
        after_tree = ast.parse(after_source)
    except SyntaxError:
        return True
    _strip_docstrings(before_tree)
    _strip_docstrings(after_tree)
    return ast.dump(before_tree, include_attributes=False) != ast.dump(after_tree, include_attributes=False)


def _python_introduced_local_call_arity_mismatch_details(
    before_source: str,
    after_source: str,
    repo_root: Path,
) -> list[str]:
    try:
        before_tree = ast.parse(before_source)
        after_tree = ast.parse(after_source)
    except SyntaxError:
        return []
    before_calls = _python_call_signature_set(before_tree)
    introduced_calls: list[tuple[str, str, int, list[str]]] = []
    wanted_names: set[str] = set()
    for node in ast.walk(after_tree):
        if not isinstance(node, ast.Call):
            continue
        func_expr = ast.unparse(node.func)
        leaf_name = _python_call_leaf_name(node.func)
        arg_count = len(node.args)
        keyword_names = [str(keyword.arg) for keyword in node.keywords if keyword.arg]
        call_signature = (func_expr, arg_count, len(node.keywords))
        if leaf_name and call_signature not in before_calls:
            introduced_calls.append((func_expr, leaf_name, arg_count, keyword_names))
            wanted_names.add(leaf_name)
    if not wanted_names:
        return []
    contracts = _python_repo_function_positional_arity_contracts(repo_root, wanted_names)
    details: list[str] = []
    keyword_contracts = _python_repo_function_keyword_contracts(repo_root, wanted_names)
    for func_expr, leaf_name, arg_count, keyword_names in introduced_calls:
        contract = contracts.get(leaf_name)
        if contract is None:
            continue
        min_args, max_args, has_varargs = contract
        if has_varargs:
            continue
        accepted_keywords, has_varkw = keyword_contracts.get(leaf_name, (set(), False))
        recognized_keyword_count = 0
        if has_varkw:
            recognized_keyword_count = len(keyword_names)
        elif accepted_keywords:
            recognized_keyword_count = sum(1 for name in keyword_names if name in accepted_keywords)
        supplied_required_slots = arg_count + recognized_keyword_count
        if arg_count < min_args and supplied_required_slots >= min_args:
            continue
        if arg_count < min_args or arg_count > max_args:
            details.append(f"{func_expr} called with {arg_count} positional args but local definition accepts {min_args}..{max_args}")
    return details


def _python_introduced_local_call_keyword_mismatch_details(
    before_source: str,
    after_source: str,
    repo_root: Path,
) -> list[str]:
    try:
        before_tree = ast.parse(before_source)
        after_tree = ast.parse(after_source)
    except SyntaxError:
        return []
    before_calls = _python_call_signature_set(before_tree)
    introduced_calls: list[tuple[str, str, list[str]]] = []
    wanted_names: set[str] = set()
    for node in ast.walk(after_tree):
        if not isinstance(node, ast.Call):
            continue
        keyword_names = [str(keyword.arg) for keyword in node.keywords if keyword.arg]
        if not keyword_names:
            continue
        func_expr = ast.unparse(node.func)
        leaf_name = _python_call_leaf_name(node.func)
        call_signature = (func_expr, len(node.args), len(node.keywords))
        if leaf_name and call_signature not in before_calls:
            introduced_calls.append((func_expr, leaf_name, keyword_names))
            wanted_names.add(leaf_name)
    if not wanted_names:
        return []
    contracts = _python_repo_function_keyword_contracts(repo_root, wanted_names)
    details: list[str] = []
    for func_expr, leaf_name, keyword_names in introduced_calls:
        contract = contracts.get(leaf_name)
        if contract is None:
            continue
        accepted_keywords, has_varkw = contract
        if has_varkw:
            continue
        unexpected = [name for name in keyword_names if name not in accepted_keywords]
        if unexpected:
            details.append(f"{func_expr} called with unsupported keywords: {', '.join(unexpected[:4])}")
    return details


def _python_call_signature_set(tree: ast.AST) -> set[tuple[str, int, int]]:
    signatures: set[tuple[str, int, int]] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            signatures.add((ast.unparse(node.func), len(node.args), len(node.keywords)))
    return signatures


def _python_call_leaf_name(func: ast.AST) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _python_repo_function_positional_arity_contracts(
    repo_root: Path,
    wanted_names: set[str],
) -> dict[str, tuple[int, int, bool]]:
    contracts: dict[str, tuple[int, int, bool]] = {}
    if not wanted_names:
        return contracts
    for path in repo_root.rglob("*.py"):
        if any(part.startswith(".") for part in path.relative_to(repo_root).parts):
            continue
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source)
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name not in wanted_names:
                continue
            min_args, max_args, has_varargs = _python_function_positional_arity_contract(node)
            existing = contracts.get(node.name)
            if existing is None:
                contracts[node.name] = (min_args, max_args, has_varargs)
                continue
            contracts[node.name] = (
                min(existing[0], min_args),
                max(existing[1], max_args),
                existing[2] or has_varargs,
            )
    return contracts


def _python_repo_function_keyword_contracts(
    repo_root: Path,
    wanted_names: set[str],
) -> dict[str, tuple[set[str], bool]]:
    contracts: dict[str, tuple[set[str], bool]] = {}
    if not wanted_names:
        return contracts
    for path in repo_root.rglob("*.py"):
        if any(part.startswith(".") for part in path.relative_to(repo_root).parts):
            continue
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source)
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name not in wanted_names:
                continue
            accepted, has_varkw = _python_function_keyword_contract(node)
            existing = contracts.get(node.name)
            if existing is None:
                contracts[node.name] = (set(accepted), has_varkw)
                continue
            contracts[node.name] = (set(existing[0]) | set(accepted), existing[1] or has_varkw)
    return contracts


def _python_function_keyword_contract(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[set[str], bool]:
    accepted: set[str] = set()
    args = list(node.args.args)
    if args and args[0].arg in {"self", "cls"}:
        args = args[1:]
    accepted.update(arg.arg for arg in args)
    accepted.update(arg.arg for arg in node.args.kwonlyargs)
    return accepted, node.args.kwarg is not None


def _python_function_positional_arity_contract(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[int, int, bool]:
    args = list(node.args.posonlyargs) + list(node.args.args)
    if args and args[0].arg in {"self", "cls"}:
        args = args[1:]
    positional_count = len(args)
    defaults_count = len(node.args.defaults)
    required_count = max(0, positional_count - defaults_count)
    return required_count, positional_count, node.args.vararg is not None


def _python_string_literal_only_changed(before_source: str, after_source: str) -> bool:
    return _python_ast_changed_only_after_normalization(
        before_source,
        after_source,
        normalizer=_normalize_python_string_constants,
    )


def _python_annotation_only_changed(before_source: str, after_source: str) -> bool:
    return _python_ast_changed_only_after_normalization(
        before_source,
        after_source,
        normalizer=_normalize_python_annotations,
    )


def _python_ast_changed_only_after_normalization(before_source: str, after_source: str, *, normalizer) -> bool:
    try:
        before_tree = ast.parse(before_source)
        after_tree = ast.parse(after_source)
    except SyntaxError:
        return False
    before_raw = ast.dump(before_tree, include_attributes=False)
    after_raw = ast.dump(after_tree, include_attributes=False)
    if before_raw == after_raw:
        return False
    before_norm = ast.parse(before_source)
    after_norm = ast.parse(after_source)
    normalizer(before_norm)
    normalizer(after_norm)
    return ast.dump(before_norm, include_attributes=False) == ast.dump(after_norm, include_attributes=False)


def _normalize_python_string_constants(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = "<str>"


def _normalize_python_annotations(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.returns = None
            for arg in (
                list(node.args.posonlyargs)
                + list(node.args.args)
                + list(node.args.kwonlyargs)
            ):
                arg.annotation = None
            if node.args.vararg is not None:
                node.args.vararg.annotation = None
            if node.args.kwarg is not None:
                node.args.kwarg.annotation = None
        elif isinstance(node, ast.AnnAssign):
            node.annotation = ast.Constant(value="<annotation>")


def _python_indentation_only_statement_moves(before_source: str, after_source: str) -> list[str]:
    before_lines = before_source.splitlines()
    after_lines = after_source.splitlines()
    if len(before_lines) != len(after_lines):
        return []
    moves: list[str] = []
    for before_line, after_line in zip(before_lines, after_lines):
        if before_line == after_line:
            continue
        if before_line.strip() and before_line.strip() == after_line.strip():
            moves.append(_python_line_preview(after_line.strip()))
    return moves


def _strip_docstrings(node: ast.AST) -> None:
    for child in ast.walk(node):
        body = getattr(child, "body", None)
        if not isinstance(body, list) or not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            body.pop(0)


def _is_python_test_path(path: str) -> bool:
    normalized = str(path).replace("\\", "/")
    name = Path(normalized).name
    return "/tests/" in normalized or name.startswith("test_") or name.endswith("_test.py")


def _is_disallowed_swe_solution_path(path: str) -> bool:
    normalized = str(path).replace("\\", "/").strip("/")
    name = Path(normalized).name
    if _is_python_test_path(normalized) or normalized.startswith(("test/", "tests/")):
        return True
    if "/test/" in f"/{normalized}/" or "/tests/" in f"/{normalized}/":
        return True
    if normalized.startswith(("doc/", "docs/")) or "/docs/" in f"/{normalized}/":
        return True
    if normalized.startswith("ci/") or "/ci/" in f"/{normalized}/":
        return True
    if normalized.startswith((".github/", "features/", "examples/", "galleries/examples/")):
        return True
    if "/examples/" in f"/{normalized}/" or "/galleries/examples/" in f"/{normalized}/":
        return True
    if "/mpl-data/" in f"/{normalized}/":
        return True
    if name.lower() in {
        "readme.md",
        "changelog.md",
        "changelog.rst",
        "changes.md",
        "changes.rst",
        "changes.txt",
        "makefile",
    }:
        return True
    if name.lower() in {"matplotlibrc"}:
        return True
    if normalized in {".secrets.baseline", ".gitignore", ".pre-commit-config.yaml"}:
        return True
    if "spellcheck" in normalized or "spelling" in normalized:
        return True
    if re.search(r"(^|/)(update|generate|regen|refresh)[_-].*\.(py|sh)$", normalized):
        return True
    if "snapshot" in normalized.lower() and normalized.startswith(("script/", "scripts/")):
        return True
    return name in {"conftest.py"}


def _python_suspicious_line_replacement_details(removed: list[str], added: list[str]) -> list[str]:
    if len(removed) != len(added):
        if len(removed) <= 4 and len(added) <= 4:
            old_joined = _python_join_continued_statement_lines(removed)
            new_joined = _python_join_continued_statement_lines(added)
            if old_joined and new_joined:
                removed_scale = _python_removed_arithmetic_scale_factor_detail(old_joined, new_joined)
                if removed_scale:
                    return [removed_scale]
        return []
    if len(removed) > 4:
        return []
    details: list[str] = []
    for old_line, new_line in zip(removed, added):
        old = str(old_line).strip()
        new = str(new_line).strip()
        if not old or not new or old.startswith("#") or new.startswith("#"):
            continue
        old_target = _python_simple_assignment_target(old)
        new_target = _python_simple_assignment_target(new)
        old_kind = _python_line_statement_kind(old)
        new_kind = _python_line_statement_kind(new)
        if old_target and new_kind in {"import", "from"}:
            details.append(f"assignment replaced by {new_kind} statement")
            continue
        if old_target and not new_target and new_kind not in {"annotation", "augassign"}:
            details.append(f"{old_target} assignment replaced by {_python_line_preview(new)}")
            continue
        dataclass_field = _python_dataclass_field_removed_detail(old, new)
        if dataclass_field:
            details.append(dataclass_field)
            continue
        if old_target and old_target != new_target:
            details.append(f"{old_target} replaced by {_python_line_preview(new)}")
            continue
        assignment_source = _python_assignment_source_replacement_detail(old, new)
        if assignment_source:
            details.append(assignment_source)
            continue
        capacity_state = _python_capacity_invariant_replaced_by_state_field_detail(old, new)
        if capacity_state:
            details.append(capacity_state)
            continue
        numeric_none = _python_numeric_literal_replaced_by_none_detail(old, new)
        if numeric_none:
            details.append(numeric_none)
            continue
        keyword_contract = _python_keyword_type_contract_replacement_detail(old, new)
        if keyword_contract:
            details.append(keyword_contract)
            continue
        identical_ternary = _python_identical_ternary_branch_detail(new)
        if identical_ternary:
            details.append(identical_ternary)
            continue
        annotation_removed = _python_annotation_removed_detail(old, new)
        if annotation_removed:
            details.append(annotation_removed)
            continue
        keyword_constant = _python_keyword_constant_replaced_by_expression_detail(old, new)
        if keyword_constant:
            details.append(keyword_constant)
            continue
        positional_keyword = _python_call_positional_arg_replaced_by_keyword_detail(old, new)
        if positional_keyword:
            details.append(positional_keyword)
            continue
        if not old_target and new_target and old.endswith(","):
            details.append(f"call/list argument replaced by {new_target} assignment")
            continue
        if not old_target and new_target:
            details.append(f"{_python_line_statement_kind(old)} replaced by {new_target} assignment")
            continue
        if old_kind == "raise" and new_kind == "raise" and old != new:
            details.append(f"raise expression replaced by {_python_line_preview(new)}")
            continue
        if old_kind == "except" and new_kind == "except" and old != new:
            details.append(f"except handler replaced by {_python_line_preview(new)}")
            continue
        if old_kind == "return" and new == "return None" and old != new:
            details.append("return expression replaced by return None")
            continue
        if (
            old_kind in {"if", "elif"}
            and new_kind == old_kind
            and old != new
            and "isinstance(" in new
            and ("KeyboardInterrupt" in new or "Exception" in new)
        ):
            details.append(f"{old_kind} condition replaced by exception-type check")
            continue
        if old_kind in {"if", "elif"} and new_kind == old_kind and _python_conditions_have_disjoint_names(old, new):
            details.append(f"{old_kind} condition replaced by unrelated condition")
            continue
        attr_condition = _python_condition_attribute_replacement_detail(old, new)
        if attr_condition:
            details.append(attr_condition)
            continue
        call_condition = _python_condition_introduces_call_detail(old, new)
        if call_condition:
            details.append(call_condition)
            continue
        bool_broaden = _python_condition_broadening_detail(old, new)
        if bool_broaden:
            details.append(bool_broaden)
            continue
        removed_guard = _python_condition_removed_none_guard_detail(old, new)
        if removed_guard:
            details.append(removed_guard)
            continue
        if old_kind == "expr" and new_kind == "expr":
            bool_expr_replacement = _python_boolean_expression_replacement_detail(old, new)
            if bool_expr_replacement:
                details.append(bool_expr_replacement)
                continue
        if old_kind == "return" and new_kind == "return" and _python_return_replaced_variable_with_constant(old, new):
            details.append(f"return variable replaced by {_python_line_preview(new)}")
            continue
        if old_kind == "return" and new_kind == "return" and _python_return_constant_contract_changed(old, new):
            details.append(f"return constant changed from {_python_return_expr(old)} to {_python_return_expr(new)}")
            continue
        if old_kind == "return" and new_kind == "return" and _python_return_simplifies_expression_detail(old, new):
            details.append(_python_return_simplifies_expression_detail(old, new))
            continue
        if old_kind == "return" and new_kind == "return" and _python_return_replaced_by_unrelated_expression_detail(old, new):
            details.append(_python_return_replaced_by_unrelated_expression_detail(old, new))
            continue
        if old_kind == "return" and new_kind == "return" and _python_return_introduces_self_reference(old, new):
            details.append(f"return expression introduces self-reference: {_python_line_preview(new)}")
            continue
        call_arg_change = _python_return_call_argument_contract_change_detail(old, new)
        if call_arg_change:
            details.append(call_arg_change)
            continue
        if old_kind == "expr" and new_kind in {"raise", "return", "if", "elif", "for", "while", "with", "import", "from"}:
            details.append(f"expression replaced by {new_kind} statement")
            continue
        literal_replacement = _python_expression_replaced_by_literal_detail(old, new)
        if literal_replacement:
            details.append(literal_replacement)
            continue
        call_replacement = _python_expression_call_replacement_detail(old, new)
        if call_replacement:
            details.append(call_replacement)
            continue
        format_removal = _python_format_call_removed_detail(old, new)
        if format_removal:
            details.append(format_removal)
            continue
        dict_value = _python_dict_value_replacement_detail(old, new)
        if dict_value:
            details.append(dict_value)
            continue
        message_rewrite = _python_message_constant_rewrite_detail(old, new)
        if message_rewrite:
            details.append(message_rewrite)
            continue
        membership_change = _python_membership_container_change_detail(old, new)
        if membership_change:
            details.append(membership_change)
            continue
        literal_shape = _python_literal_shape_replacement_detail(old, new)
        if literal_shape:
            details.append(literal_shape)
            continue
        generator_replacement = _python_generator_clause_replaced_by_unrelated_detail(old, new)
        if generator_replacement:
            details.append(generator_replacement)
            continue
        mutating_call_replacement = _python_expression_replaced_by_mutating_call_detail(old, new)
        if mutating_call_replacement:
            details.append(mutating_call_replacement)
            continue
        duplicate_reducer = _python_nested_duplicate_reducer_name(new)
        if duplicate_reducer:
            details.append(f"{new_kind} expression nests duplicate reducer {duplicate_reducer}")
            continue
        duplicate_call = _python_nested_duplicate_call_name(new)
        if duplicate_call:
            details.append(f"{new_kind} expression nests duplicate call {duplicate_call}")
            continue
        broad_casts = _python_broad_backend_cast_operand_names(old, new)
        if broad_casts:
            details.append(f"{new_kind} expression broadly casts existing operands: {', '.join(broad_casts[:4])}")
            continue
        tuple_change = _python_tuple_element_replacement_detail(old, new)
        if tuple_change:
            details.append(tuple_change)
            continue
        or_none = _python_return_super_call_or_none_added_detail(old, new)
        if or_none:
            details.append(or_none)
            continue
        removed_scale = _python_removed_arithmetic_scale_factor_detail(old, new)
        if removed_scale:
            details.append(removed_scale)
            continue
        arithmetic_change = _python_arithmetic_literal_contract_change_detail(old, new)
        if arithmetic_change:
            details.append(arithmetic_change)
            continue
        exclude_none = _python_exclude_none_serialization_change_detail(old, new)
        if exclude_none:
            details.append(exclude_none)
            continue
        bool_get = _python_bool_condition_introduces_unguarded_get_detail(old, new)
        if bool_get:
            details.append(bool_get)
            continue
        if old_kind == "assign" and new_kind in {"import", "from"}:
            details.append(f"assignment replaced by {new_kind} statement")
            continue
        if old_kind in {"if", "elif", "for", "while", "with", "return", "yield", "raise"} and new_kind != old_kind:
            details.append(f"{old_kind} statement replaced by {new_kind or _python_line_preview(new)}")
    return details


def _python_simple_assignment_target(line: str) -> str:
    if re.match(r"^(if|elif|while|for|with|return|yield|raise|assert|from|import|class|def|async)\b", line):
        return ""
    try:
        tree = ast.parse(line)
    except SyntaxError:
        tree = None
    if tree is not None and len(tree.body) == 1:
        statement = tree.body[0]
        target: ast.AST | None = None
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
        elif isinstance(statement, ast.AnnAssign):
            target = statement.target
        elif isinstance(statement, ast.AugAssign):
            target = statement.target
        if isinstance(target, ast.Name):
            return target.id
        if isinstance(target, ast.Attribute):
            parts: list[str] = []
            current: ast.AST | None = target
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
                return ".".join(reversed(parts))
        if isinstance(target, ast.Subscript):
            return ast.unparse(target)
        if isinstance(target, ast.Tuple):
            names = [ast.unparse(element) for element in target.elts]
            if names:
                return ", ".join(names)
    if any(operator in line for operator in ("==", "!=", "<=", ">=")):
        return ""
    match = re.match(r"^([A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*|\[[^\]]+\])?)\s*(?::[^=]+)?=(?!=)", line)
    if not match:
        return ""
    return match.group(1)


def _python_line_statement_kind(line: str) -> str:
    for keyword in ("if", "elif", "for", "while", "with", "return", "yield", "raise", "assert", "except", "import", "from"):
        if re.match(rf"^{keyword}\b", line):
            return keyword
    if re.match(r"^async\s+def\b", line):
        return "def"
    if re.match(r"^def\b", line):
        return "def"
    if re.match(r"^class\b", line):
        return "class"
    if _python_simple_assignment_target(line):
        return "assign"
    try:
        tree = ast.parse(line)
    except SyntaxError:
        tree = None
    if tree is not None and len(tree.body) == 1 and isinstance(tree.body[0], (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        return "assign"
    return "expr"


def _python_line_preview(line: str) -> str:
    compact = " ".join(str(line).strip().split())
    if len(compact) > 40:
        return compact[:37] + "..."
    return compact


def _python_duplicate_existing_statement_replacement_details(hunk_lines: list[tuple[str, str]]) -> list[str]:
    context = {
        " ".join(line.strip().split())
        for sign, line in hunk_lines
        if sign == " " and _python_executable_statement_for_duplicate_guard(line)
    }
    if not context:
        return []
    removed = [line for sign, line in hunk_lines if sign == "-"]
    added = [line for sign, line in hunk_lines if sign == "+"]
    if len(removed) != len(added) or len(added) > 4:
        return []
    details: list[str] = []
    for old_line, new_line in zip(removed, added):
        old_normalized = " ".join(old_line.strip().split())
        new_normalized = " ".join(new_line.strip().split())
        if not new_normalized or old_normalized == new_normalized:
            continue
        if (
            new_normalized in context
            and _python_executable_statement_for_duplicate_guard(old_line)
            and _python_executable_statement_for_duplicate_guard(new_line)
        ):
            details.append(f"{_python_line_preview(new_line)} duplicates existing hunk statement")
    return details


def _python_numeric_literal_replaced_by_none_detail(old_line: str, new_line: str) -> str:
    old = " ".join(str(old_line).strip().split())
    new = " ".join(str(new_line).strip().split())
    old_match = re.search(r"(?P<prefix>[A-Za-z_][A-Za-z0-9_\.]*\s*=\s*)[-+]?\d+(?:\.\d+)?\b", old)
    new_match = re.search(r"(?P<prefix>[A-Za-z_][A-Za-z0-9_\.]*\s*=\s*)None\b", new)
    if not old_match or not new_match:
        return ""
    if re.sub(r"\s+", "", old_match.group("prefix")) != re.sub(r"\s+", "", new_match.group("prefix")):
        return ""
    return f"numeric default {old_match.group('prefix').strip()} replaced by None"


def _python_identical_ternary_branch_detail(line: str) -> str:
    try:
        tree = ast.parse(str(line).strip())
    except SyntaxError:
        try:
            tree = ast.parse(f"_value = {str(line).strip()}")
        except SyntaxError:
            return ""
    for node in ast.walk(tree):
        if isinstance(node, ast.IfExp) and ast.dump(node.body, include_attributes=False) == ast.dump(
            node.orelse,
            include_attributes=False,
        ):
            return f"conditional expression has identical branches {_python_line_preview(line)}"
    return ""


def _python_keyword_type_contract_replacement_detail(old_line: str, new_line: str) -> str:
    old_match = re.search(r"\bcheck_type\s*=\s*([A-Za-z_][A-Za-z0-9_\.]*)", old_line)
    new_match = re.search(r"\bcheck_type\s*=\s*([A-Za-z_][A-Za-z0-9_\.]*)", new_line)
    if not old_match or not new_match:
        return ""
    old_value = old_match.group(1)
    new_value = new_match.group(1)
    if old_value == new_value:
        return ""
    return f"check_type contract changed from {old_value} to {new_value}"


def _python_removed_control_guard_details(hunk_lines: list[tuple[str, str]]) -> list[str]:
    details: list[str] = []
    added_control_indents = [
        len(line) - len(line.lstrip())
        for sign, line in hunk_lines
        if sign == "+" and _python_line_statement_kind(line.strip()) in {"if", "elif", "while", "for", "try", "except"}
    ]
    for sign, line in hunk_lines:
        if sign != "-":
            continue
        stripped = line.strip()
        if _python_line_statement_kind(stripped) not in {"if", "elif", "while", "for"} or not stripped.endswith(":"):
            continue
        indent = len(line) - len(line.lstrip())
        if any(added_indent <= indent for added_indent in added_control_indents):
            continue
        details.append(f"removed control guard {_python_line_preview(line)}")
    return details


def _python_exception_alias_mismatch_details(hunk_lines: list[tuple[str, str]]) -> list[str]:
    aliases: set[str] = set()
    for _sign, line in hunk_lines:
        match = re.match(r"^\s*except\b.*\bas\s+([A-Za-z_][A-Za-z0-9_]*)\s*:", line)
        if match:
            aliases.add(match.group(1))
    if not aliases:
        return []
    details: list[str] = []
    for sign, line in hunk_lines:
        if sign != "+":
            continue
        for name in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\.", line):
            if name in aliases or name in {"self", "cls", "super"}:
                continue
            if re.search(rf"\b{name}\b", line) and any(alias in line for alias in aliases):
                continue
            details.append(
                f"exception handler aliases {', '.join(sorted(aliases))} but added {_python_line_preview(line)}"
            )
            return details
    return details


def _python_nested_mapping_reinitialization_details(hunk_lines: list[tuple[str, str]]) -> list[str]:
    removed_nested: set[tuple[str, str]] = set()
    for sign, line in hunk_lines:
        if sign != "-":
            continue
        match = re.search(
            r"(?P<base>[A-Za-z_][A-Za-z0-9_\.]*)\s*\[\s*(?P<key>[^\]]+)\s*\]\s*\[",
            line,
        )
        if match:
            removed_nested.add((match.group("base"), " ".join(match.group("key").split())))
    if not removed_nested:
        return []
    added_base_assignments: set[tuple[str, str]] = set()
    added_nested_assignments: set[tuple[str, str]] = set()
    for sign, line in hunk_lines:
        if sign != "+":
            continue
        base_match = re.search(
            r"(?P<base>[A-Za-z_][A-Za-z0-9_\.]*)\s*\[\s*(?P<key>[^\]]+)\s*\]\s*=",
            line,
        )
        if base_match:
            added_base_assignments.add((base_match.group("base"), " ".join(base_match.group("key").split())))
        nested_match = re.search(
            r"(?P<base>[A-Za-z_][A-Za-z0-9_\.]*)\s*\[\s*(?P<key>[^\]]+)\s*\]\s*\[",
            line,
        )
        if nested_match:
            added_nested_assignments.add((nested_match.group("base"), " ".join(nested_match.group("key").split())))
    risky = sorted(removed_nested & added_base_assignments & added_nested_assignments)
    if not risky:
        return []
    base, key = risky[0]
    return [f"{base}[{key}] reinitialized before nested field updates"]


def _python_suspicious_hunk_replacement_details(hunk_lines: list[tuple[str, str]]) -> list[str]:
    details: list[str] = []
    removed_comments = [line for sign, line in hunk_lines if sign == "-" and line.strip().startswith("#")]
    added_control_flow = [
        line
        for sign, line in hunk_lines
        if sign == "+" and re.match(r"^\s*(return|raise|break|continue)\b", line)
    ]
    if removed_comments and added_control_flow:
        details.append(
            f"comment replaced by control-flow statement {_python_line_preview(added_control_flow[0])}"
        )

    context_targets = {
        target
        for sign, line in hunk_lines
        if sign == " " and (target := _python_simple_assignment_target(str(line).strip()))
    }
    context_statements = {
        " ".join(line.strip().split())
        for sign, line in hunk_lines
        if sign == " " and _python_executable_statement_for_duplicate_guard(line)
    }
    removed_statements = {
        " ".join(line.strip().split())
        for sign, line in hunk_lines
        if sign == "-" and _python_executable_statement_for_duplicate_guard(line)
    }
    context_literals = {
        literal
        for sign, line in hunk_lines
        if sign == " "
        for literal in re.findall(r"['\"]([^'\"]+)['\"]", line)
    }
    added_literals = {
        literal
        for sign, line in hunk_lines
        if sign == "+"
        for literal in re.findall(r"['\"]([^'\"]+)['\"]", line)
    }
    added_source_line_count = sum(1 for sign, line in hunk_lines if sign == "+" and str(line).strip())
    context_reuse_has_new_literal_scope = bool(added_literals - context_literals) and added_source_line_count >= 3
    for sign, line in hunk_lines:
        if sign != "+":
            continue
        stripped = str(line).strip()
        normalized = " ".join(stripped.split())
        if normalized in context_statements:
            if normalized in removed_statements or context_reuse_has_new_literal_scope:
                continue
            details.append(f"{_python_line_preview(line)} duplicates existing hunk statement")
            continue
        target = _python_simple_assignment_target(stripped)
        if target and target in context_targets:
            details.append(f"{target} assignment duplicates existing hunk target")

    removed = [line for sign, line in hunk_lines if sign == "-"]
    added = [line for sign, line in hunk_lines if sign == "+"]
    details.extend(_python_suspicious_line_replacement_details(removed, added))
    details.extend(_python_exception_alias_mismatch_details(hunk_lines))
    details.extend(_python_nested_mapping_reinitialization_details(hunk_lines))
    details.extend(_python_removed_control_guard_details(hunk_lines))
    for old_line in removed:
        old_stripped = old_line.strip()
        old_indent = len(old_line) - len(old_line.lstrip())
        if not old_stripped.startswith("elif ") or not old_stripped.endswith(":"):
            continue
        old_condition = old_stripped[len("elif ") : -1].strip()
        for new_line in added:
            new_stripped = new_line.strip()
            new_indent = len(new_line) - len(new_line.lstrip())
            if new_indent > old_indent and new_stripped == f"if {old_condition}:":
                details.append(f"elif branch nested as if {old_condition}")

    context_conditions = {
        re.sub(r"^(if|elif|while)\s+", "", line.strip()).rstrip(":")
        for sign, line in hunk_lines
        if sign == " " and _python_line_statement_kind(line.strip()) in {"if", "elif", "while"}
    }
    context_continuations = {
        " ".join(line.strip().split())
        for sign, line in hunk_lines
        if sign == " " and re.match(r"^(and|or)\b", line.strip())
    }
    for new_line in added:
        new_continuation = " ".join(new_line.strip().split())
        if new_continuation in context_continuations:
            details.append(f"boolean continuation duplicates existing hunk clause {new_continuation}")
    for new_line in added:
        new_kind = _python_line_statement_kind(new_line.strip())
        if new_kind not in {"if", "elif", "while"}:
            continue
        new_condition = re.sub(r"^(if|elif|while)\s+", "", new_line.strip()).rstrip(":")
        for context_condition in context_conditions:
            if context_condition and context_condition != new_condition and context_condition in new_condition:
                details.append(f"{new_kind} condition duplicates nested hunk condition")
                break

    removed_return_indent = [
        len(line) - len(line.lstrip())
        for line in removed
        if re.match(r"^\s*return\s+[A-Za-z_][\w.]*\s*\(", line)
    ]
    if removed_return_indent:
        min_removed_return_indent = min(removed_return_indent)
        for new_line in added:
            new_stripped = new_line.strip()
            new_indent = len(new_line) - len(new_line.lstrip())
            if new_stripped.startswith("return ") and new_indent > min_removed_return_indent:
                details.append(f"return constructor block collapsed into nested {_python_line_preview(new_line)}")
                break
    for old_line in removed:
        old_stripped = old_line.strip()
        if not re.match(r"^(del\s+)?self\.[A-Za-z_]\w*(?:\.[A-Za-z_]\w*|\[[^\]]+\])+", old_stripped):
            continue
        old_attrs = _python_expr_attribute_set(old_stripped.replace("del ", ""))
        for new_line in added:
            new_stripped = new_line.strip()
            if new_stripped.startswith("del "):
                if "._data" in new_stripped and "._data" not in old_stripped:
                    details.append(f"delete target replaced with unrelated private data deletion")
                    continue
                new_attrs = _python_expr_attribute_set(new_stripped.replace("del ", ""))
                if old_attrs and new_attrs and old_attrs.isdisjoint(new_attrs):
                    details.append(f"delete target replaced with unrelated private data deletion")
    return details


def _suspicious_text_template_replacement_details(removed: list[str], added: list[str]) -> list[str]:
    details: list[str] = []
    for old_line, new_line in zip(removed, added):
        old = old_line.strip()
        new = new_line.strip()
        if old and new and old != new and old.replace("'", '"') == new.replace("'", '"'):
            details.append(f"template line changes only quote style {_python_line_preview(new_line)}")
            continue
        if "->" in new and "->" not in old and re.search(r"\\[sSdDwW]", old):
            details.append(f"regex matcher replaced by state transition {_python_line_preview(new)}")
            continue
        if re.match(r"^(export\s+default\s+)?function\b", old) and re.match(r"^\{%\s*(for|if)\b", new):
            details.append(f"template control replaced executable function {_python_line_preview(old)}")
    return details


def _python_executable_statement_for_duplicate_guard(line: str) -> bool:
    stripped = str(line).strip()
    if not stripped or stripped.startswith("#"):
        return False
    if stripped in {")", "}", "]"} or stripped.endswith((",", "\\", "(", "[", "{")):
        return False
    if _python_line_statement_kind(stripped) in {"import", "from", "def", "class"}:
        return False
    try:
        tree = ast.parse(stripped)
    except SyntaxError:
        return False
    return len(tree.body) == 1


def _python_expression_replaced_by_literal_detail(old_line: str, new_line: str) -> str:
    try:
        old_tree = ast.parse(str(old_line).strip())
        new_tree = ast.parse(str(new_line).strip())
    except SyntaxError:
        return ""
    if len(old_tree.body) != 1 or len(new_tree.body) != 1:
        return ""
    old_statement = old_tree.body[0]
    new_statement = new_tree.body[0]
    if not isinstance(old_statement, ast.Expr) or not isinstance(new_statement, ast.Expr):
        return ""
    if isinstance(old_statement.value, ast.Constant):
        return ""
    if isinstance(new_statement.value, ast.Constant):
        return f"expression replaced by literal {_python_line_preview(new_line)}"
    return ""


def _python_expression_call_replacement_detail(old_line: str, new_line: str) -> str:
    try:
        old_tree = ast.parse(str(old_line).strip())
        new_tree = ast.parse(str(new_line).strip())
    except SyntaxError:
        return ""
    if len(old_tree.body) != 1 or len(new_tree.body) != 1:
        return ""
    old_statement = old_tree.body[0]
    new_statement = new_tree.body[0]
    if not isinstance(old_statement, ast.Expr) or not isinstance(new_statement, ast.Expr):
        return ""
    if not isinstance(old_statement.value, ast.Call) or not isinstance(new_statement.value, ast.Call):
        return ""
    old_name = _python_call_func_name(old_statement.value)
    new_name = _python_call_func_name(new_statement.value)
    if old_name and new_name and old_name != new_name:
        return f"call expression replaced by unrelated call {old_name}->{new_name}"
    return ""


def _python_expression_replaced_by_mutating_call_detail(old_line: str, new_line: str) -> str:
    old_expr = _python_expression_from_statement_line(old_line)
    new_expr = _python_expression_from_statement_line(new_line)
    try:
        try:
            old_node = ast.parse(old_expr, mode="eval").body
        except SyntaxError:
            if " for " not in old_expr:
                raise
            old_node = ast.parse(f"({old_expr})", mode="eval").body
        new_node = ast.parse(new_expr, mode="eval").body
    except SyntaxError:
        return ""
    if not isinstance(new_node, ast.Call):
        return ""
    call_name = _python_call_func_name(new_node)
    if not call_name.endswith((".add", ".append", ".extend", ".update", ".remove", ".discard", ".pop", ".clear")):
        return ""
    old_mentions_iteration = isinstance(old_node, (ast.GeneratorExp, ast.ListComp, ast.SetComp, ast.DictComp)) or " for " in old_expr
    if old_mentions_iteration:
        return f"iterable expression replaced by mutating call {call_name}"
    return ""


def _python_dataclass_field_removed_detail(old_line: str, new_line: str) -> str:
    try:
        old_tree = ast.parse(str(old_line).strip())
        new_tree = ast.parse(str(new_line).strip())
    except SyntaxError:
        return ""
    if len(old_tree.body) != 1 or len(new_tree.body) != 1:
        return ""
    old_statement = old_tree.body[0]
    new_statement = new_tree.body[0]
    if not isinstance(old_statement, ast.AnnAssign) or not isinstance(new_statement, ast.AnnAssign):
        return ""
    if not isinstance(old_statement.value, ast.Call):
        return ""
    if _python_call_func_name(old_statement.value).endswith(".field") and isinstance(new_statement.value, ast.Constant):
        return f"{ast.unparse(old_statement.target)} dataclass field replaced by constant default"
    return ""


def _python_assignment_source_replacement_detail(old_line: str, new_line: str) -> str:
    try:
        old_tree = ast.parse(str(old_line).strip())
        new_tree = ast.parse(str(new_line).strip())
    except SyntaxError:
        return ""
    if len(old_tree.body) != 1 or len(new_tree.body) != 1:
        return ""
    old_statement = old_tree.body[0]
    new_statement = new_tree.body[0]
    if not isinstance(old_statement, ast.Assign) or not isinstance(new_statement, ast.Assign):
        return ""
    if len(old_statement.targets) != 1 or len(new_statement.targets) != 1:
        return ""
    if ast.dump(old_statement.targets[0]) != ast.dump(new_statement.targets[0]):
        return ""
    old_attrs = _python_expr_attribute_set(ast.unparse(old_statement.value))
    new_calls = _python_call_names(new_statement.value)
    if old_attrs and new_calls and old_attrs.isdisjoint(new_calls):
        return f"{ast.unparse(old_statement.targets[0])} assignment source replaced by unrelated call"
    return ""


def _python_format_call_removed_detail(old_line: str, new_line: str) -> str:
    if ".format(" not in old_line or ".format(" in new_line:
        return ""
    try:
        old_node = ast.parse(_python_expression_from_statement_line(old_line), mode="eval").body
        new_node = ast.parse(_python_expression_from_statement_line(new_line), mode="eval").body
    except SyntaxError:
        return ""
    if isinstance(old_node, ast.Call) and isinstance(old_node.func, ast.Attribute) and old_node.func.attr == "format":
        base = ast.unparse(old_node.func.value)
        if ast.unparse(new_node) == base:
            return f"format call removed from {base}"
    return ""


def _python_dict_value_replacement_detail(old_line: str, new_line: str) -> str:
    old = str(old_line).strip().rstrip(",")
    new = str(new_line).strip().rstrip(",")
    if not (old.startswith(("'", '"')) and ":" in old and ":" in new):
        return ""
    try:
        old_expr = ast.parse("{" + old + "}", mode="eval").body
        new_expr = ast.parse("{" + new + "}", mode="eval").body
    except SyntaxError:
        return ""
    if not isinstance(old_expr, ast.Dict) or not isinstance(new_expr, ast.Dict):
        return ""
    if len(old_expr.keys) != 1 or len(new_expr.keys) != 1:
        return ""
    if ast.dump(old_expr.keys[0]) != ast.dump(new_expr.keys[0]):
        return f"dict key {ast.unparse(old_expr.keys[0])} replaced by {ast.unparse(new_expr.keys[0])}"
    old_value = old_expr.values[0]
    new_value = new_expr.values[0]
    if ast.dump(old_value) != ast.dump(new_value):
        return f"dict value for {ast.unparse(old_expr.keys[0])} replaced"
    return ""


def _python_message_constant_rewrite_detail(old_line: str, new_line: str) -> str:
    old_expr = _python_expression_from_statement_line(old_line)
    new_expr = _python_expression_from_statement_line(new_line)
    try:
        old_node = ast.parse(old_expr, mode="eval").body
        new_node = ast.parse(new_expr, mode="eval").body
    except SyntaxError:
        return ""
    if isinstance(old_node, ast.JoinedStr) and isinstance(new_node, ast.JoinedStr):
        old_text = re.sub(r"\{[^}]+\}", " ", old_expr)
        new_text = re.sub(r"\{[^}]+\}", " ", new_expr)
        old_words = set(re.findall(r"[A-Za-z_]{4,}", old_text))
        new_words = set(re.findall(r"[A-Za-z_]{4,}", new_text))
        shared = old_words & new_words
        if len(shared) <= 1:
            return "formatted message rewritten with unrelated text"
    if isinstance(old_node, ast.Name) and isinstance(new_node, ast.JoinedStr):
        return f"message constant {old_expr} replaced by formatted string"
    if isinstance(old_node, ast.Call) and isinstance(new_node, ast.Call):
        if _python_call_func_name(old_node) != _python_call_func_name(new_node):
            return ""
        for old_arg, new_arg in zip(old_node.args, new_node.args):
            if isinstance(old_arg, ast.Name) and isinstance(new_arg, ast.JoinedStr):
                return f"message constant {old_arg.id} replaced by formatted string"
    return ""


def _python_return_call_argument_contract_change_detail(old_line: str, new_line: str) -> str:
    try:
        old_node = ast.parse(_python_return_expr(old_line), mode="eval").body
        new_node = ast.parse(_python_return_expr(new_line), mode="eval").body
    except SyntaxError:
        return ""
    if not isinstance(old_node, ast.Call) or not isinstance(new_node, ast.Call):
        return ""
    old_name = _python_call_func_name(old_node)
    new_name = _python_call_func_name(new_node)
    if old_name != new_name:
        return ""
    if len(new_node.args) > len(old_node.args) or len(new_node.keywords) > len(old_node.keywords):
        return f"return {old_name} call broadens arguments"
    if old_node.args and new_node.args:
        old_arg = old_node.args[0]
        new_arg = new_node.args[0]
        if isinstance(old_arg, ast.Name) and isinstance(new_arg, ast.Attribute):
            return f"return {old_name} call changes first argument shape"
    return ""


def _python_condition_broadening_detail(old_line: str, new_line: str) -> str:
    old_kind = _python_line_statement_kind(old_line)
    new_kind = _python_line_statement_kind(new_line)
    if old_kind not in {"if", "elif"} or old_kind != new_kind:
        return ""
    old_expr = re.sub(r"^(if|elif)\s+", "", str(old_line).strip()).rstrip(":")
    new_expr = re.sub(r"^(if|elif)\s+", "", str(new_line).strip()).rstrip(":")
    if old_expr and old_expr in new_expr and old_expr != new_expr:
        return f"{old_kind} condition broadened with extra clause"
    return ""


def _python_condition_removed_none_guard_detail(old_line: str, new_line: str) -> str:
    old_kind = _python_line_statement_kind(old_line)
    new_kind = _python_line_statement_kind(new_line)
    if old_kind not in {"if", "elif"} or old_kind != new_kind:
        return ""
    old_expr = re.sub(r"^(if|elif)\s+", "", str(old_line).strip()).rstrip(":")
    new_expr = re.sub(r"^(if|elif)\s+", "", str(new_line).strip()).rstrip(":")
    try:
        old_node = ast.parse(old_expr, mode="eval").body
        ast.parse(new_expr, mode="eval")
    except SyntaxError:
        return ""
    guarded_names: set[str] = set()
    for node in ast.walk(old_node):
        if not isinstance(node, ast.Compare):
            continue
        if not any(isinstance(op, ast.IsNot) for op in node.ops):
            continue
        if not any(isinstance(comparator, ast.Constant) and comparator.value is None for comparator in node.comparators):
            continue
        if isinstance(node.left, ast.Name):
            guarded_names.add(node.left.id)
    for name in sorted(guarded_names):
        if f"{name} is not None" in old_expr and f"{name} is not None" not in new_expr and f"{name}." in new_expr:
            return f"{old_kind} condition removed None guard before {name} attribute access"
    return ""


def _python_boolean_expression_replacement_detail(old_line: str, new_line: str) -> str:
    old_expr = _python_expression_from_statement_line(old_line)
    new_expr = _python_expression_from_statement_line(new_line)
    try:
        old_node = ast.parse(old_expr, mode="eval").body
        new_node = ast.parse(new_expr, mode="eval").body
    except SyntaxError:
        return ""
    old_calls = _python_call_names(old_node)
    new_calls = _python_call_names(new_node)
    if any(call_name.endswith(".get") for call_name in new_calls - old_calls):
        return ""
    old_unparsed = ast.unparse(old_node)
    new_unparsed = ast.unparse(new_node)
    if isinstance(old_node, ast.BoolOp) and isinstance(new_node, ast.BoolOp):
        if old_unparsed and old_unparsed in new_unparsed and old_unparsed != new_unparsed:
            return "boolean expression broadened with extra clause"
        new_subtrees = {ast.dump(node, include_attributes=False) for node in ast.walk(new_node)}
        old_parts = [ast.dump(value, include_attributes=False) for value in old_node.values]
        if old_parts and all(part in new_subtrees for part in old_parts) and ast.dump(old_node) != ast.dump(new_node):
            return "boolean expression broadened with extra clause"
    old_attrs = _python_expr_attribute_set(old_expr)
    new_attrs = _python_expr_attribute_set(new_expr)
    booleanish = (ast.BoolOp, ast.Compare, ast.UnaryOp)
    if (
        isinstance(old_node, booleanish)
        and isinstance(new_node, booleanish)
        and new_calls
        and (old_calls or old_attrs)
        and old_calls.isdisjoint(new_calls)
        and old_attrs.isdisjoint(new_calls)
    ):
        return "boolean expression replaced by unrelated call"
    return ""


def _python_membership_container_change_detail(old_line: str, new_line: str) -> str:
    old_expr = _python_expression_from_statement_line(old_line)
    new_expr = _python_expression_from_statement_line(new_line)
    if " in " not in old_expr or " in " not in new_expr:
        return ""
    if re.search(r"\bin\s+set\(", old_expr) and not re.search(r"\bin\s+set\(", new_expr):
        return "membership container set wrapper removed"
    if " in (" in old_expr and " in [" in new_expr:
        return "membership container tuple replaced by list"
    return ""


def _python_literal_shape_replacement_detail(old_line: str, new_line: str) -> str:
    old = str(old_line).strip().rstrip(",")
    new = str(new_line).strip().rstrip(",")
    try:
        old_node = ast.parse(old, mode="eval").body
        new_node = ast.parse(new, mode="eval").body
    except SyntaxError:
        return ""
    if isinstance(old_node, ast.Dict) and isinstance(new_node, ast.Constant):
        return "dictionary literal replaced by scalar literal"
    return ""


def _python_annotation_removed_detail(old_line: str, new_line: str) -> str:
    try:
        old_tree = ast.parse(str(old_line).strip())
        new_tree = ast.parse(str(new_line).strip())
    except SyntaxError:
        return ""
    if len(old_tree.body) != 1 or len(new_tree.body) != 1:
        return ""
    old_statement = old_tree.body[0]
    new_statement = new_tree.body[0]
    if not isinstance(old_statement, ast.AnnAssign) or not isinstance(new_statement, ast.Assign):
        return ""
    old_target = ast.unparse(old_statement.target)
    new_targets = [ast.unparse(target) for target in new_statement.targets]
    if old_target in new_targets:
        return f"{old_target} annotation removed"
    return ""


def _python_keyword_constant_replaced_by_expression_detail(old_line: str, new_line: str) -> str:
    old = str(old_line).strip().rstrip(",")
    new = str(new_line).strip().rstrip(",")
    old_match = re.match(r"^([A-Za-z_]\w*)\s*=\s*(True|False|None|[-+]?\d+(?:\.\d+)?)$", old)
    new_match = re.match(r"^([A-Za-z_]\w*)\s*=\s*(.+)$", new)
    if old_match and new_match and old_match.group(1) == new_match.group(1):
        try:
            new_node = ast.parse(new_match.group(2), mode="eval").body
        except SyntaxError:
            return ""
        if not isinstance(new_node, ast.Constant):
            return f"{old_match.group(1)} keyword constant replaced by expression"
    return ""


def _python_call_positional_arg_replaced_by_keyword_detail(old_line: str, new_line: str) -> str:
    old_expr = _python_expression_from_statement_line(old_line)
    new_expr = _python_expression_from_statement_line(new_line)
    try:
        old_node = ast.parse(old_expr, mode="eval").body
        new_node = ast.parse(new_expr, mode="eval").body
    except SyntaxError:
        return ""
    if not isinstance(old_node, ast.Call) or not isinstance(new_node, ast.Call):
        return ""
    old_name = _python_call_func_name(old_node)
    new_name = _python_call_func_name(new_node)
    if old_name != new_name:
        return ""
    old_constants = {repr(arg.value) for arg in old_node.args if isinstance(arg, ast.Constant)}
    for keyword in new_node.keywords:
        if isinstance(keyword.value, ast.Constant) and repr(keyword.value.value) in old_constants:
            return f"{old_name} call converts positional argument to keyword {keyword.arg}"
    return ""


def _python_generator_clause_replaced_by_unrelated_detail(old_line: str, new_line: str) -> str:
    if " for " not in old_line or " for " not in new_line:
        return ""
    try:
        old_node = ast.parse(f"({str(old_line).strip()})", mode="eval").body
        new_node = ast.parse(f"({str(new_line).strip()})", mode="eval").body
    except SyntaxError:
        return ""
    if not isinstance(old_node, ast.GeneratorExp) or not isinstance(new_node, ast.GeneratorExp):
        return ""
    old_names = _python_ast_name_set(old_node) | _python_expr_attribute_set(str(old_line).strip())
    new_names = _python_ast_name_set(new_node) | _python_expr_attribute_set(str(new_line).strip())
    if old_names and new_names and old_names.isdisjoint(new_names):
        return "generator expression replaced by unrelated generator"
    return ""


def _python_conditions_have_disjoint_names(old_line: str, new_line: str) -> bool:
    old_expr = re.sub(r"^(if|elif)\s+", "", str(old_line).strip()).rstrip(":")
    new_expr = re.sub(r"^(if|elif)\s+", "", str(new_line).strip()).rstrip(":")
    old_names = _python_expr_name_set(old_expr)
    new_names = _python_expr_name_set(new_expr)
    if not old_names or not new_names:
        return False
    return old_names.isdisjoint(new_names)


def _python_condition_attribute_replacement_detail(old_line: str, new_line: str) -> str:
    old_kind = _python_line_statement_kind(old_line)
    new_kind = _python_line_statement_kind(new_line)
    if old_kind not in {"if", "elif"} or old_kind != new_kind or old_line == new_line:
        return ""
    old_expr = re.sub(r"^(if|elif)\s+", "", str(old_line).strip()).rstrip(":")
    new_expr = re.sub(r"^(if|elif)\s+", "", str(new_line).strip()).rstrip(":")
    old_attrs = _python_expr_attribute_set(old_expr)
    new_attrs = _python_expr_attribute_set(new_expr)
    if old_attrs and new_attrs and old_attrs.isdisjoint(new_attrs):
        return f"{old_kind} condition replaced by unrelated attribute condition"
    return ""


def _python_condition_introduces_call_detail(old_line: str, new_line: str) -> str:
    old_kind = _python_line_statement_kind(old_line)
    new_kind = _python_line_statement_kind(new_line)
    if old_kind not in {"if", "elif", "while"} or old_kind != new_kind or old_line == new_line:
        return ""
    old_expr = re.sub(r"^(if|elif|while)\s+", "", str(old_line).strip()).rstrip(":")
    new_expr = re.sub(r"^(if|elif|while)\s+", "", str(new_line).strip()).rstrip(":")
    try:
        old_node = ast.parse(old_expr, mode="eval").body
        new_node = ast.parse(new_expr, mode="eval").body
    except SyntaxError:
        return ""
    introduced_calls = sorted(_python_call_names(new_node) - _python_call_names(old_node))
    if introduced_calls:
        return f"{old_kind} condition introduces call {introduced_calls[0]}"
    return ""


def _python_return_replaced_variable_with_constant(old_line: str, new_line: str) -> bool:
    old_expr = str(old_line).strip()[len("return ") :].strip()
    new_expr = str(new_line).strip()[len("return ") :].strip()
    try:
        old_node = ast.parse(old_expr, mode="eval").body
        new_node = ast.parse(new_expr, mode="eval").body
    except SyntaxError:
        return False
    return isinstance(old_node, ast.Name) and isinstance(new_node, (ast.Name, ast.Constant)) and old_expr != new_expr


def _python_return_constant_contract_changed(old_line: str, new_line: str) -> bool:
    try:
        old_node = ast.parse(_python_return_expr(old_line), mode="eval").body
        new_node = ast.parse(_python_return_expr(new_line), mode="eval").body
    except SyntaxError:
        return False
    return (
        isinstance(old_node, ast.Constant)
        and isinstance(new_node, ast.Constant)
        and old_node.value != new_node.value
    )


def _python_return_simplifies_expression_detail(old_line: str, new_line: str) -> str:
    try:
        old_node = ast.parse(_python_return_expr(old_line), mode="eval").body
        new_node = ast.parse(_python_return_expr(new_line), mode="eval").body
    except SyntaxError:
        return ""
    if isinstance(old_node, (ast.BinOp, ast.Call, ast.Subscript)) and isinstance(new_node, (ast.Name, ast.Attribute, ast.Constant)):
        return f"return expression simplified to {_python_line_preview(new_line)}"
    return ""


def _python_return_replaced_by_unrelated_expression_detail(old_line: str, new_line: str) -> str:
    try:
        old_node = ast.parse(_python_return_expr(old_line), mode="eval").body
        new_node = ast.parse(_python_return_expr(new_line), mode="eval").body
    except SyntaxError:
        return ""
    old_names = _python_ast_name_set(old_node) | _python_expr_attribute_set(_python_return_expr(old_line))
    new_names = _python_ast_name_set(new_node) | _python_expr_attribute_set(_python_return_expr(new_line))
    if old_names and new_names and old_names.isdisjoint(new_names):
        return "return expression replaced by unrelated expression"
    return ""


def _python_return_expr(line: str) -> str:
    return str(line).strip()[len("return ") :].strip()


def _python_return_introduces_self_reference(old_line: str, new_line: str) -> bool:
    old_expr = _python_return_expr(old_line)
    new_expr = _python_return_expr(new_line)
    if "self." not in new_expr:
        return False
    return bool(_python_expr_attribute_set(new_expr) - _python_expr_attribute_set(old_expr))


def _python_nested_duplicate_reducer_name(line: str) -> str:
    fragment_name = _python_nested_duplicate_reducer_name_from_fragment(line)
    if fragment_name:
        return fragment_name
    try:
        tree = ast.parse(_python_expression_from_statement_line(line), mode="eval")
    except SyntaxError:
        return ""
    reducer_names = {
        "reduce_all",
        "reduce_any",
        "reduce_max",
        "reduce_mean",
        "reduce_min",
        "reduce_prod",
        "reduce_sum",
        "max",
        "mean",
        "min",
        "prod",
        "sum",
    }
    for outer in ast.walk(tree):
        if not isinstance(outer, ast.Call):
            continue
        outer_name = _python_call_func_name(outer)
        if outer_name.split(".")[-1] not in reducer_names:
            continue
        for child in ast.walk(ast.Expression(body=outer)):
            if child is outer or not isinstance(child, ast.Call):
                continue
            if _python_call_func_name(child) == outer_name:
                return outer_name
    return ""


def _python_nested_duplicate_call_name(line: str) -> str:
    try:
        tree = ast.parse(_python_expression_from_statement_line(line), mode="eval")
    except SyntaxError:
        return ""
    for outer in ast.walk(tree):
        if not isinstance(outer, ast.Call):
            continue
        outer_name = _python_call_func_name(outer)
        if not outer_name:
            continue
        for child in ast.walk(ast.Expression(body=outer)):
            if child is outer or not isinstance(child, ast.Call):
                continue
            if _python_call_func_name(child) == outer_name:
                return outer_name
    return ""


def _python_nested_duplicate_reducer_details(before_source: str, after_source: str) -> list[str]:
    before = _python_nested_duplicate_reducer_occurrences(before_source)
    after = _python_nested_duplicate_reducer_occurrences(after_source)
    return sorted(after - before)


def _python_introduced_self_recursive_property_accesses(before_source: str, after_source: str) -> list[str]:
    before = set(_python_self_recursive_property_accesses(before_source))
    after = set(_python_self_recursive_property_accesses(after_source))
    return sorted(after - before)


def _python_self_recursive_property_accesses(source: str) -> list[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    details: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if not any(
            (isinstance(decorator, ast.Name) and decorator.id == "property")
            or (isinstance(decorator, ast.Attribute) and decorator.attr == "property")
            for decorator in node.decorator_list
        ):
            continue
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Attribute)
                and child.attr == node.name
                and isinstance(child.value, ast.Name)
                and child.value.id == "self"
            ):
                details.append(f"{node.name} reads self.{node.name}")
                break
    return details


def _python_nested_assignment_replacement_details(patch_text: str) -> list[str]:
    details: list[str] = []
    removed: list[str] = []
    added: list[str] = []
    current_path = ""

    def flush() -> None:
        nonlocal removed, added
        if current_path.endswith(".py") and not _is_python_test_path(current_path):
            for old_line in removed:
                old_target = _python_assignment_target_expr(old_line)
                if not old_target:
                    continue
                for new_line in added:
                    new_target = _python_assignment_target_expr(new_line)
                    if not new_target or new_target == old_target:
                        continue
                    if new_target.startswith(old_target + "[") or new_target.startswith(old_target + "."):
                        details.append(
                            f"{_python_line_preview(old_line)} replaced by nested target {_python_line_preview(new_line)}"
                        )
        removed = []
        added = []

    for line in patch_text.splitlines():
        if line.startswith("+++ "):
            flush()
            path = line[4:].strip()
            current_path = path[2:] if path.startswith("b/") else path
            if current_path == "/dev/null":
                current_path = ""
            continue
        if line.startswith(("diff --git ", "--- ", "@@ ")):
            flush()
            continue
        if line.startswith("-"):
            removed.append(line[1:])
        elif line.startswith("+"):
            added.append(line[1:])
        else:
            flush()
    flush()
    return details


def _python_assignment_target_expr(line: str) -> str:
    try:
        tree = ast.parse(str(line).strip())
    except SyntaxError:
        return ""
    if len(tree.body) != 1:
        return ""
    stmt = tree.body[0]
    target: ast.AST | None = None
    if isinstance(stmt, ast.Assign):
        if len(stmt.targets) != 1:
            return ""
        target = stmt.targets[0]
    elif isinstance(stmt, (ast.AnnAssign, ast.AugAssign)):
        target = stmt.target
    if target is None:
        return ""
    try:
        return ast.unparse(target)
    except Exception:
        return ""


def _python_nested_duplicate_reducer_occurrences(source: str) -> set[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    occurrences: set[str] = set()
    for outer in ast.walk(tree):
        if not isinstance(outer, ast.Call):
            continue
        outer_name = _python_call_func_name(outer)
        if not _python_is_reducer_call_name(outer_name):
            continue
        for child in ast.walk(ast.Expression(body=outer)):
            if child is outer or not isinstance(child, ast.Call):
                continue
            if _python_call_func_name(child) == outer_name:
                occurrences.add(outer_name)
    return occurrences


def _python_nested_duplicate_reducer_name_from_fragment(line: str) -> str:
    calls = re.findall(r"\b((?:[A-Za-z_]\w*\.)*(?:reduce_all|reduce_any|reduce_max|reduce_mean|reduce_min|reduce_prod|reduce_sum))\s*\(", str(line))
    for name in calls:
        if _python_is_reducer_call_name(name) and calls.count(name) > 1:
            return name
    return ""


def _python_is_reducer_call_name(name: str) -> bool:
    reducer_names = {
        "reduce_all",
        "reduce_any",
        "reduce_max",
        "reduce_mean",
        "reduce_min",
        "reduce_prod",
        "reduce_sum",
        "max",
        "mean",
        "min",
        "prod",
        "sum",
    }
    return str(name).split(".")[-1] in reducer_names


def _python_broad_backend_cast_operand_names(old_line: str, new_line: str) -> list[str]:
    try:
        old_tree = ast.parse(_python_expression_from_statement_line(old_line), mode="eval")
        new_tree = ast.parse(_python_expression_from_statement_line(new_line), mode="eval")
    except SyntaxError:
        return []
    old_casted = _python_backend_cast_operand_names(old_tree)
    new_casted = _python_backend_cast_operand_names(new_tree)
    introduced = sorted(new_casted - old_casted)
    if len(introduced) < 2:
        return []
    old_names = _python_expr_name_set(_python_expression_from_statement_line(old_line))
    risky = [name for name in introduced if name in old_names and name not in {"input", "inputs", "x", "y", "tensor", "tensors"}]
    return risky if len(risky) >= 2 else []


def _python_backend_cast_operand_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        func_name = _python_call_func_name(node)
        if func_name not in {"backend.cast", "self.backend.cast"}:
            continue
        operand = node.args[0]
        if isinstance(operand, ast.Name):
            names.add(operand.id)
    return names


def _python_call_func_name(node: ast.Call) -> str:
    try:
        return ast.unparse(node.func)
    except Exception:
        return ""


def _python_expression_from_statement_line(line: str) -> str:
    stripped = str(line).strip()
    for keyword in ("return", "yield", "raise"):
        prefix = f"{keyword} "
        if stripped.startswith(prefix):
            return stripped[len(prefix) :].strip()
    if stripped.startswith(("if ", "elif ")):
        return re.sub(r"^(if|elif)\s+", "", stripped).rstrip(":")
    for operator in ("+=", "-=", "*=", "/=", "//=", "%=", "="):
        if operator in stripped and not any(compare in stripped for compare in ("==", "!=", "<=", ">=")):
            return stripped.split(operator, 1)[1].strip()
    return stripped


def _python_tuple_element_replacement_detail(old_line: str, new_line: str) -> str:
    old = str(old_line).strip().rstrip(",")
    new = str(new_line).strip().rstrip(",")
    if not (old.startswith("(") and new.startswith("(")):
        return ""
    try:
        old_node = ast.parse(old, mode="eval").body
        new_node = ast.parse(new, mode="eval").body
    except SyntaxError:
        return ""
    if not isinstance(old_node, ast.Tuple) or not isinstance(new_node, ast.Tuple):
        return ""
    if len(old_node.elts) != len(new_node.elts) or len(old_node.elts) < 2:
        return ""
    changed: list[str] = []
    for index, (old_elt, new_elt) in enumerate(zip(old_node.elts, new_node.elts)):
        old_text = ast.unparse(old_elt)
        new_text = ast.unparse(new_elt)
        if old_text != new_text:
            changed.append(f"{index}:{old_text}->{new_text}")
    if not changed:
        return ""
    return f"tuple element contract changed: {', '.join(changed[:3])}"


def _python_return_super_call_or_none_added_detail(old_line: str, new_line: str) -> str:
    old_expr = _python_expression_from_statement_line(old_line)
    new_expr = _python_expression_from_statement_line(new_line)
    if "super()" not in old_expr or " or None" not in new_expr:
        return ""
    try:
        old_node = ast.parse(old_expr, mode="eval").body
        new_node = ast.parse(new_expr, mode="eval").body
    except SyntaxError:
        return ""
    if _python_has_or_none_boolop(new_node) and not _python_has_or_none_boolop(old_node):
        return "return super call broadened with or None"
    return ""


def _python_has_or_none_boolop(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if not isinstance(child, ast.BoolOp) or not isinstance(child.op, ast.Or):
            continue
        if any(isinstance(value, ast.Constant) and value.value is None for value in child.values):
            return True
    return False


def _python_arithmetic_literal_contract_change_detail(old_line: str, new_line: str) -> str:
    old_expr = _python_expression_from_statement_line(old_line)
    new_expr = _python_expression_from_statement_line(new_line)
    try:
        old_node = ast.parse(old_expr, mode="eval").body
        new_node = ast.parse(new_expr, mode="eval").body
    except SyntaxError:
        return ""
    old_ops = _python_binary_op_literals(old_node)
    new_ops = _python_binary_op_literals(new_node)
    if old_ops == new_ops:
        return ""
    for key, old_literals in old_ops.items():
        new_literals = new_ops.get(key, [])
        if old_literals and new_literals and old_literals != new_literals:
            return f"arithmetic literal contract changed near {key}: {old_literals[:3]}->{new_literals[:3]}"
    return ""


def _python_removed_arithmetic_scale_factor_detail(old_line: str, new_line: str) -> str:
    old_target = _python_simple_assignment_target(old_line)
    new_target = _python_simple_assignment_target(new_line)
    if old_target and new_target and old_target != new_target:
        return ""
    old_expr = _python_expression_from_statement_line(old_line)
    new_expr = _python_expression_from_statement_line(new_line)
    try:
        old_node = ast.parse(old_expr, mode="eval").body
        new_node = ast.parse(new_expr, mode="eval").body
    except SyntaxError:
        return ""
    old_scales = _python_divisor_numeric_constant_sets(old_node)
    new_scales = _python_divisor_numeric_constant_sets(new_node)
    for old_scale in old_scales:
        if len(old_scale) < 2:
            continue
        if old_scale not in new_scales and not any(old_scale.issubset(new_scale) for new_scale in new_scales):
            preview = "*".join(sorted(old_scale))
            return f"arithmetic scale factor removed or weakened near {old_target or _python_line_statement_kind(old_line)}: {preview}"
    return ""


def _python_capacity_invariant_replaced_by_state_field_detail(old_line: str, new_line: str) -> str:
    old_target = _python_simple_assignment_target(old_line)
    new_target = _python_simple_assignment_target(new_line)
    if not old_target or old_target != new_target:
        return ""
    old_expr = _python_expression_from_statement_line(old_line)
    new_expr = _python_expression_from_statement_line(new_line)
    try:
        old_node = ast.parse(old_expr, mode="eval").body
        new_node = ast.parse(new_expr, mode="eval").body
    except SyntaxError:
        return ""
    if not _python_is_plain_name_or_attribute(old_node) or not _python_is_plain_name_or_attribute(new_node):
        return ""
    old_text = _python_expr_preview(old_node)
    new_text = _python_expr_preview(new_node)
    if old_text == new_text:
        return ""
    old_tokens = _python_identifier_tokens(old_node)
    new_tokens = _python_identifier_tokens(new_node)
    capacity_tokens = {
        "max",
        "maximum",
        "limit",
        "capacity",
        "cap",
        "size",
        "len",
        "length",
        "width",
        "height",
        "total",
    }
    state_tokens = {
        "previous",
        "prev",
        "current",
        "last",
        "next",
        "boundary",
        "cursor",
        "offset",
        "index",
        "idx",
        "state",
        "position",
        "pos",
    }
    if old_tokens & capacity_tokens and new_tokens & state_tokens:
        return f"capacity/limit invariant replaced by state field: {old_text}->{new_text}"
    return ""


def _python_is_plain_name_or_attribute(node: ast.AST) -> bool:
    return isinstance(node, ast.Name) or (
        isinstance(node, ast.Attribute) and _python_is_plain_name_or_attribute(node.value)
    )


def _python_expr_preview(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return node.__class__.__name__


def _python_identifier_tokens(node: ast.AST) -> set[str]:
    tokens: set[str] = set()
    for child in ast.walk(node):
        value = ""
        if isinstance(child, ast.Name):
            value = child.id
        elif isinstance(child, ast.Attribute):
            value = child.attr
        if not value:
            continue
        for part in re.findall(r"[A-Za-z][A-Za-z0-9]*", value):
            tokens.add(part.lower())
    return tokens


def _python_join_continued_statement_lines(lines: list[str]) -> str:
    payload = " ".join(str(line).strip() for line in lines if str(line).strip())
    return re.sub(r"\s+", " ", payload).strip()


def _python_divisor_numeric_constant_sets(node: ast.AST) -> set[frozenset[str]]:
    scales: set[frozenset[str]] = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.BinOp) or not isinstance(child.op, (ast.Div, ast.FloorDiv)):
            continue
        constants = frozenset(_python_numeric_constant_tokens(child.right))
        if constants:
            scales.add(constants)
    return scales


def _python_numeric_constant_tokens(node: ast.AST) -> set[str]:
    tokens: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, (int, float)) and not isinstance(child.value, bool):
            tokens.add(str(child.value))
    return tokens


def _python_binary_op_literals(node: ast.AST) -> dict[str, list[str]]:
    values: dict[str, list[str]] = {}
    for child in ast.walk(node):
        if not isinstance(child, ast.BinOp):
            continue
        names = sorted(_python_ast_name_set(child))
        if not names:
            continue
        literals = [ast.unparse(grandchild) for grandchild in ast.walk(child) if isinstance(grandchild, ast.Constant)]
        operators = [type(grandchild.op).__name__ for grandchild in ast.walk(child) if isinstance(grandchild, ast.BinOp)]
        if literals:
            values["+".join(names)] = [*operators, *literals]
    return values


def _python_exclude_none_serialization_change_detail(old_line: str, new_line: str) -> str:
    old_expr = _python_expression_from_statement_line(old_line)
    new_expr = _python_expression_from_statement_line(new_line)
    if "exclude_none" not in new_expr or "exclude_none" in old_expr:
        return ""
    try:
        new_node = ast.parse(new_expr, mode="eval").body
    except SyntaxError:
        return ""
    for call in (node for node in ast.walk(new_node) if isinstance(node, ast.Call)):
        if not any(keyword.arg == "exclude_none" and isinstance(keyword.value, ast.Constant) and keyword.value.value is True for keyword in call.keywords):
            continue
        func_name = _python_call_func_name(call)
        if func_name.endswith((".dict", ".model_dump", ".json", ".model_dump_json")):
            return f"serialization call adds exclude_none=True to {func_name}"
    return ""


def _python_bool_condition_introduces_unguarded_get_detail(old_line: str, new_line: str) -> str:
    if ".get(" not in new_line:
        return ""
    old_gets = set(re.findall(r"\b([A-Za-z_]\w*)\.get\s*\(", old_line))
    new_gets = set(re.findall(r"\b([A-Za-z_]\w*)\.get\s*\(", new_line))
    introduced_gets = sorted(new_gets - old_gets)
    if introduced_gets and re.match(r"^or\b", new_line.strip()) and re.match(r"^and\b", old_line.strip()):
        return f"boolean condition introduces unguarded mapping get {introduced_gets[0]}.get"
    old_expr = _python_expression_from_statement_line(old_line)
    new_expr = _python_expression_from_statement_line(new_line)
    try:
        old_node = ast.parse(old_expr, mode="eval").body
        new_node = ast.parse(new_expr, mode="eval").body
    except SyntaxError:
        return ""
    if not isinstance(new_node, ast.BoolOp) or not isinstance(new_node.op, ast.Or):
        return ""
    if not isinstance(old_node, ast.BoolOp) or not isinstance(old_node.op, ast.And):
        return ""
    introduced_gets = _python_call_names(new_node) - _python_call_names(old_node)
    for call_name in sorted(introduced_gets):
        if call_name.endswith(".get"):
            return f"boolean condition introduces unguarded mapping get {call_name}"
    return ""


def _python_call_names(node: ast.AST) -> set[str]:
    return {_python_call_func_name(call) for call in ast.walk(node) if isinstance(call, ast.Call)}


def _python_ast_name_set(node: ast.AST) -> set[str]:
    return {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}


def _python_expr_name_set(expr: str) -> set[str]:
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return set()
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}


def _python_expr_attribute_set(expr: str) -> set[str]:
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return set()
    attrs: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            attrs.add(ast.unparse(node))
    return attrs


def _suspicious_config_key_replacement_details(removed: list[str], added: list[str]) -> list[str]:
    if len(removed) != len(added) or len(removed) > 4:
        return []
    details: list[str] = []
    for old_line, new_line in zip(removed, added):
        old = str(old_line).rstrip()
        new = str(new_line).rstrip()
        old_match = re.match(r"^(\s*)([A-Za-z_][\w.-]*)\s*[:=]\s*(.+?)\s*$", old)
        new_match = re.match(r"^(\s*)([A-Za-z_][\w.-]*)\s*[:=]\s*(.+?)\s*$", new)
        if not old_match or not new_match:
            continue
        if old_match.group(1) != new_match.group(1):
            continue
        old_key = old_match.group(2)
        new_key = new_match.group(2)
        old_value = old_match.group(3).strip()
        new_value = new_match.group(3).strip()
        if old_key != new_key and old_value and new_value:
            details.append(f"{old_key} replaced by {new_key}")
    return details


def _suspicious_semantic_token_flip_details(removed: list[str], added: list[str]) -> list[str]:
    if len(removed) != len(added) or len(removed) > 4:
        return []
    token_pairs = (("female", "male"), ("woman", "man"), ("women", "men"))
    details: list[str] = []
    for old_line, new_line in zip(removed, added):
        old = str(old_line)
        new = str(new_line)
        for left, right in token_pairs:
            old_left = len(re.findall(rf"(?<![A-Za-z]){left}(?![A-Za-z])", old, flags=re.IGNORECASE))
            old_right = len(re.findall(rf"(?<![A-Za-z]){right}(?![A-Za-z])", old, flags=re.IGNORECASE))
            new_left = len(re.findall(rf"(?<![A-Za-z]){left}(?![A-Za-z])", new, flags=re.IGNORECASE))
            new_right = len(re.findall(rf"(?<![A-Za-z]){right}(?![A-Za-z])", new, flags=re.IGNORECASE))
            if old_left > new_left and new_right > old_right:
                details.append(f"{left} token replaced by {right}")
            elif old_right > new_right and new_left > old_left:
                details.append(f"{right} token replaced by {left}")
    return details


def _python_duplicate_surrounding_call_wrapper_details(hunk_lines: list[tuple[str, str]]) -> list[str]:
    details: list[str] = []
    for index, (prefix, line) in enumerate(hunk_lines):
        if prefix != "+":
            continue
        added = str(line).strip()
        added_call = _python_call_prefix_name(added)
        if not added_call:
            continue
        before_context = [
            str(context_line).strip()
            for context_prefix, context_line in hunk_lines[max(0, index - 4) : index]
            if context_prefix == " "
        ]
        after_context = [
            str(context_line).strip()
            for context_prefix, context_line in hunk_lines[index + 1 : index + 5]
            if context_prefix == " "
        ]
        if not any(_python_context_opens_call(context, added_call) for context in before_context):
            continue
        if not any(context == ")" or context.startswith(")") for context in after_context):
            continue
        removed_neighbors = [
            str(context_line).strip()
            for context_prefix, context_line in hunk_lines[max(0, index - 3) : index + 3]
            if context_prefix == "-"
        ]
        if any(added_call in removed for removed in removed_neighbors):
            continue
        details.append(f"{added_call} wrapper duplicated inside existing call")
    return details


def _python_call_prefix_name(line: str) -> str:
    match = re.match(r"^([A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)*)\s*\(", str(line).strip())
    return match.group(1) if match else ""


def _python_context_opens_call(line: str, call_name: str) -> bool:
    stripped = str(line).strip()
    return bool(
        re.match(rf"^(?:return|yield)\s+{re.escape(call_name)}\s*\(\s*$", stripped)
        or re.match(rf"^{re.escape(call_name)}\s*\(\s*$", stripped)
        or re.match(rf"^[A-Za-z_][\w.]*\s*=\s*{re.escape(call_name)}\s*\(\s*$", stripped)
    )


def _python_suspicious_attribute_replacement_details(before_source: str, after_source: str) -> list[str]:
    before_attrs = _python_attribute_names_by_base(before_source)
    before_replacements = _python_simple_assignment_attribute_reads(before_source)
    after_replacements = _python_simple_assignment_attribute_reads(after_source)
    details: list[str] = []
    for target, after_read in after_replacements.items():
        before_read = before_replacements.get(target)
        if before_read is None:
            continue
        before_base, before_attr = before_read
        after_base, after_attr = after_read
        if before_base != after_base or before_attr == after_attr:
            continue
        if after_attr not in before_attrs.get(after_base, set()):
            details.append(f"{target}: {before_base}.{before_attr} replaced by unknown {after_base}.{after_attr}")
    return sorted(details)


def _python_simple_assignment_attribute_reads(source: str) -> dict[str, tuple[str, str]]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    reads: dict[str, tuple[str, str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        value = node.value
        if isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name):
            reads[target.id] = (value.value.id, value.attr)
    return reads


def _python_attribute_names_by_base(source: str) -> dict[str, set[str]]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    attrs: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            attrs.setdefault(node.value.id, set()).add(node.attr)
    return attrs


def _python_introduced_private_attribute_accesses(before_source: str, after_source: str) -> list[str]:
    before = _python_private_attribute_accesses(before_source)
    after = _python_private_attribute_accesses(after_source)
    return sorted(after - before)


def _python_private_attribute_accesses(source: str) -> set[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    accesses: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute) or not isinstance(node.ctx, ast.Load):
            continue
        if not node.attr.startswith("_") or node.attr.startswith("__"):
            continue
        if isinstance(node.value, ast.Name) and node.value.id not in {"self", "cls"}:
            accesses.add(f"{node.value.id}.{node.attr}")
    return accesses


def _python_introduced_unknown_self_private_attribute_accesses(before_source: str, after_source: str) -> list[str]:
    before_attrs = _python_self_attribute_names(before_source)
    after_loads = _python_self_private_attribute_loads(after_source)
    return sorted(attr for attr in after_loads if attr not in before_attrs)


def _python_self_attribute_names(source: str) -> set[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    attrs: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in {"self", "cls"}:
            attrs.add(node.attr)
    return attrs


def _python_self_private_attribute_loads(source: str) -> set[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    attrs: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute) or not isinstance(node.ctx, ast.Load):
            continue
        if not isinstance(node.value, ast.Name) or node.value.id not in {"self", "cls"}:
            continue
        if node.attr.startswith("_") and not node.attr.startswith("__"):
            attrs.add(node.attr)
    return attrs


def _python_exception_contract_regression_details(before_source: str, after_source: str) -> list[str]:
    details: set[str] = set()
    for function_name, before_raises in _python_function_raise_signatures(before_source).items():
        after_raises = _python_function_raise_signatures(after_source).get(function_name, set())
        for before_raise in before_raises:
            if before_raise == "raise" and "raise" not in after_raises:
                replacement = sorted(after_raises - before_raises)
                if replacement:
                    details.add(f"{function_name}: bare re-raise replaced by {replacement[0]}")
    for function_name, before_conditions in _python_function_if_condition_signatures(before_source).items():
        after_conditions = _python_function_if_condition_signatures(after_source).get(function_name, set())
        if not before_conditions or not after_conditions:
            continue
        introduced = after_conditions - before_conditions
        if any("isinstance(" in condition and ("KeyboardInterrupt" in condition or "Exception" in condition) for condition in introduced):
            details.add(f"{function_name}: condition replaced by exception-type check")
    return sorted(details)


def _python_function_raise_signatures(source: str) -> dict[str, set[str]]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    signatures: dict[str, set[str]] = {}

    def _visit(node: ast.AST, prefix: str = "") -> None:
        for child in getattr(node, "body", []):
            if isinstance(child, ast.ClassDef):
                qualified = f"{prefix}.{child.name}" if prefix else child.name
                _visit(child, qualified)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualified = f"{prefix}.{child.name}" if prefix else child.name
                values: set[str] = set()
                for body_node in _python_function_body_nodes(child):
                    if isinstance(body_node, ast.Raise):
                        values.add("raise" if body_node.exc is None else ast.unparse(body_node.exc))
                signatures[qualified] = values
                _visit(child, qualified)

    _visit(tree)
    return signatures


def _python_function_if_condition_signatures(source: str) -> dict[str, set[str]]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    signatures: dict[str, set[str]] = {}

    def _visit(node: ast.AST, prefix: str = "") -> None:
        for child in getattr(node, "body", []):
            if isinstance(child, ast.ClassDef):
                qualified = f"{prefix}.{child.name}" if prefix else child.name
                _visit(child, qualified)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualified = f"{prefix}.{child.name}" if prefix else child.name
                values: set[str] = set()
                for body_node in _python_function_body_nodes(child):
                    if isinstance(body_node, ast.If):
                        values.add(ast.unparse(body_node.test))
                signatures[qualified] = values
                _visit(child, qualified)

    _visit(tree)
    return signatures


def _python_redundant_decorated_normalization_details(before_source: str, after_source: str) -> list[str]:
    before = _python_decorated_normalization_calls(before_source)
    after = _python_decorated_normalization_calls(after_source)
    introduced: list[str] = []
    for name, after_calls in after.items():
        before_calls = before.get(name, set())
        for call in sorted(after_calls - before_calls):
            introduced.append(f"{name}: {call}")
    return sorted(introduced)


def _python_decorated_normalization_calls(source: str) -> dict[str, set[str]]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    normalized: dict[str, set[str]] = {}

    def _decorator_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
        names: set[str] = set()
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                names.add(decorator.id)
            elif isinstance(decorator, ast.Attribute):
                names.add(decorator.attr)
            elif isinstance(decorator, ast.Call):
                func = decorator.func
                if isinstance(func, ast.Name):
                    names.add(func.id)
                elif isinstance(func, ast.Attribute):
                    names.add(func.attr)
        return names

    def _visit(node: ast.AST, prefix: str = "") -> None:
        for child in getattr(node, "body", []):
            if isinstance(child, ast.ClassDef):
                qualified = f"{prefix}.{child.name}" if prefix else child.name
                _visit(child, qualified)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualified = f"{prefix}.{child.name}" if prefix else child.name
                decorators = _decorator_names(child)
                calls: set[str] = set()
                if "lowercase" in decorators:
                    for body_node in _python_function_body_nodes(child):
                        if (
                            isinstance(body_node, ast.Call)
                            and isinstance(body_node.func, ast.Attribute)
                            and body_node.func.attr == "lower"
                        ):
                            calls.add(".lower()")
                if calls:
                    normalized[qualified] = calls
                _visit(child, qualified)

    _visit(tree)
    return normalized


def _removed_python_definition_names(before_source: str, after_source: str) -> list[str]:
    before_defs = _python_definition_names(before_source)
    after_defs = _python_definition_names(after_source)
    if not before_defs or not after_defs:
        return []
    return sorted(before_defs - after_defs)


def _python_definition_names(source: str) -> set[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    names: set[str] = set()

    def _visit(node: ast.AST, prefix: str = "") -> None:
        for child in getattr(node, "body", []):
            if isinstance(child, ast.ClassDef):
                qualified = f"{prefix}.{child.name}" if prefix else child.name
                names.add(qualified)
                _visit(child, qualified)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualified = f"{prefix}.{child.name}" if prefix else child.name
                names.add(qualified)
                _visit(child, qualified)

    _visit(tree)
    return names


def _unused_new_python_parameters(before_source: str, after_source: str) -> list[str]:
    before_defs = _python_function_parameters(before_source)
    after_defs = _python_function_parameters(after_source)
    if not before_defs or not after_defs:
        return []
    first_contexts = _python_function_first_name_contexts(after_source)
    unused: list[str] = []
    for name, after_params in sorted(after_defs.items()):
        before_params = before_defs.get(name, set())
        new_params = sorted(after_params - before_params)
        if not new_params:
            continue
        used_names = _python_function_used_names(after_source).get(name, set())
        function_first_contexts = first_contexts.get(name, {})
        for param in new_params:
            if param not in used_names or function_first_contexts.get(param) != "load":
                unused.append(f"{name}.{param}")
    return unused


def _python_init_return_value_names(source: str) -> list[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    invalid: list[str] = []

    def _visit(node: ast.AST, prefix: str = "") -> None:
        for child in getattr(node, "body", []):
            if isinstance(child, ast.ClassDef):
                qualified = f"{prefix}.{child.name}" if prefix else child.name
                _visit(child, qualified)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualified = f"{prefix}.{child.name}" if prefix else child.name
                if child.name == "__init__":
                    for body_item in child.body:
                        for descendant in ast.walk(body_item):
                            if (
                                isinstance(descendant, ast.Return)
                                and descendant.value is not None
                                and qualified not in invalid
                            ):
                                invalid.append(qualified)
                _visit(child, qualified)

    _visit(tree)
    return sorted(invalid)


def _python_init_generator_names(source: str) -> list[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    invalid: list[str] = []

    def _visit(node: ast.AST, prefix: str = "") -> None:
        for child in getattr(node, "body", []):
            if isinstance(child, ast.ClassDef):
                qualified = f"{prefix}.{child.name}" if prefix else child.name
                _visit(child, qualified)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualified = f"{prefix}.{child.name}" if prefix else child.name
                if child.name == "__init__":
                    for body_item in child.body:
                        for descendant in ast.walk(body_item):
                            if isinstance(descendant, (ast.Yield, ast.YieldFrom)) and qualified not in invalid:
                                invalid.append(qualified)
                _visit(child, qualified)

    _visit(tree)
    return sorted(invalid)


def _python_removed_init_instance_assignments(before_source: str, after_source: str) -> list[str]:
    def init_assignments(source: str) -> set[str]:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return set()
        assignments: set[str] = set()
        for class_node in ast.walk(tree):
            if not isinstance(class_node, ast.ClassDef):
                continue
            for node in class_node.body:
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name != "__init__":
                    continue
                for child in ast.walk(node):
                    targets: list[ast.expr] = []
                    if isinstance(child, ast.Assign):
                        targets.extend(child.targets)
                    elif isinstance(child, ast.AnnAssign):
                        targets.append(child.target)
                    elif isinstance(child, ast.AugAssign):
                        targets.append(child.target)
                    for target in targets:
                        if (
                            isinstance(target, ast.Attribute)
                            and isinstance(target.value, ast.Name)
                            and target.value.id == "self"
                        ):
                            assignments.add(f"{class_node.name}.__init__.self.{target.attr}")
        return assignments

    before = init_assignments(before_source)
    if not before:
        return []
    after = init_assignments(after_source)
    return sorted(before - after)


def _python_local_load_before_assignment_names(source: str) -> list[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    invalid: list[str] = []

    def _visit(node: ast.AST, prefix: str = "") -> None:
        for child in getattr(node, "body", []):
            if isinstance(child, ast.ClassDef):
                qualified = f"{prefix}.{child.name}" if prefix else child.name
                _visit(child, qualified)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualified = f"{prefix}.{child.name}" if prefix else child.name
                local_names, declared_external = _python_function_local_binding_names(child)
                local_names -= declared_external
                first_contexts: dict[str, str] = {}
                for body_item in child.body:
                    _record_first_body_name_contexts(body_item, first_contexts)
                for name in sorted(local_names):
                    if first_contexts.get(name) == "load":
                        invalid.append(f"{qualified}.{name}")
                _visit(child, qualified)

    _visit(tree)
    return sorted(invalid)


def _removed_python_module_registration_names(before_source: str, after_source: str) -> list[str]:
    before = _python_module_registration_assignment_names(before_source)
    after = _python_module_registration_assignment_names(after_source)
    if not before:
        return []
    return sorted(before - after)


def _removed_python_module_state_assignment_names(before_source: str, after_source: str) -> list[str]:
    before = _python_module_state_assignment_names(before_source)
    after = _python_module_state_assignment_names(after_source)
    if not before:
        return []
    return sorted(before - after)


def _python_module_registration_assignment_names(source: str) -> set[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    class_names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    registrations: set[str] = set()
    for node in tree.body:
        target_names: list[str] = []
        value: ast.AST | None = None
        if isinstance(node, ast.Assign):
            value = node.value
            for target in node.targets:
                if isinstance(target, ast.Name):
                    target_names.append(target.id)
        elif isinstance(node, ast.AnnAssign):
            value = node.value
            if isinstance(node.target, ast.Name):
                target_names.append(node.target.id)
        if not target_names or not isinstance(value, ast.Call):
            continue
        callee = _python_call_leaf_name(value.func)
        for target_name in target_names:
            if callee in class_names or target_name in {"check", "rule", "plugin", "app"}:
                registrations.add(target_name)
    return registrations


def _python_module_state_assignment_names(source: str) -> set[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    names: set[str] = set()
    for node in tree.body:
        target_names: list[str] = []
        value: ast.AST | None = None
        if isinstance(node, ast.Assign):
            value = node.value
            for target in node.targets:
                if isinstance(target, ast.Name):
                    target_names.append(target.id)
        elif isinstance(node, ast.AnnAssign):
            value = node.value
            if isinstance(node.target, ast.Name):
                target_names.append(node.target.id)
        if not target_names or value is None:
            continue
        if isinstance(value, (ast.Dict, ast.List, ast.Tuple, ast.Set)):
            names.update(target_names)
        elif isinstance(value, ast.Call):
            names.update(target_names)
    return names


def _python_call_leaf_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _introduced_python_unresolved_name_loads(before_source: str, after_source: str) -> list[str]:
    before = set(_python_unresolved_name_loads(before_source))
    after = set(_python_unresolved_name_loads(after_source))
    return sorted(after - before)


def _python_unresolved_name_loads(source: str) -> list[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    module_names = _python_module_bound_names(tree)
    builtin_names = set(dir(builtins))
    unresolved: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)):
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load) and child.id not in module_names and child.id not in builtin_names:
                unresolved.add(f"<module>.{child.id}")

    def _visit(node: ast.AST, prefix: str = "") -> None:
        for child in getattr(node, "body", []):
            if isinstance(child, ast.ClassDef):
                qualified = f"{prefix}.{child.name}" if prefix else child.name
                _visit(child, qualified)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualified = f"{prefix}.{child.name}" if prefix else child.name
                local_names, declared_external = _python_function_local_binding_names(child)
                parameter_names = _python_function_parameter_names(child)
                allowed = module_names | builtin_names | local_names | parameter_names | declared_external
                for body_node in _python_function_body_nodes(child):
                    if isinstance(body_node, ast.Name) and isinstance(body_node.ctx, ast.Load):
                        if body_node.id not in allowed:
                            unresolved.add(f"{qualified}.{body_node.id}")
                _visit(child, qualified)

    _visit(tree)
    return sorted(unresolved)


def _python_module_bound_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(str(alias.asname or alias.name).split(".", 1)[0])
            continue
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    names.add(str(alias.asname or alias.name).split(".", 1)[0])
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                names.add(child.id)
    return names


def _python_function_parameter_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    names: set[str] = set()
    names.update(arg.arg for arg in node.args.posonlyargs)
    names.update(arg.arg for arg in node.args.args)
    names.update(arg.arg for arg in node.args.kwonlyargs)
    if node.args.vararg is not None:
        names.add(node.args.vararg.arg)
    if node.args.kwarg is not None:
        names.add(node.args.kwarg.arg)
    return names


def _python_removed_return_value_paths(before_source: str, after_source: str) -> list[str]:
    before = _python_return_value_counts(before_source)
    after = _python_return_value_counts(after_source)
    removed: list[str] = []
    for name, before_count in before.items():
        if before_count > 0 and after.get(name, 0) < before_count:
            removed.append(name)
    return sorted(removed)


def _python_introduced_none_return_value_paths(before_source: str, after_source: str) -> list[str]:
    before = _python_none_return_value_counts(before_source)
    after = _python_none_return_value_counts(after_source)
    introduced: list[str] = []
    for name, after_count in after.items():
        if after_count > before.get(name, 0):
            introduced.append(name)
    return sorted(introduced)


def _python_introduced_none_container_misuse_details(before_source: str, after_source: str) -> list[str]:
    before_none = _python_none_assignment_names(before_source)
    after_none = _python_none_assignment_names(after_source)
    introduced_none = after_none - before_none
    if not introduced_none:
        return []
    try:
        tree = ast.parse(after_source)
    except SyntaxError:
        return []
    misused: set[str] = set()

    class Visitor(ast.NodeVisitor):
        def visit_Subscript(self, node: ast.Subscript) -> None:
            if isinstance(node.value, ast.Name) and node.value.id in introduced_none:
                misused.add(f"{node.value.id}:subscript")
            self.generic_visit(node)

        def visit_Attribute(self, node: ast.Attribute) -> None:
            if isinstance(node.value, ast.Name) and node.value.id in introduced_none:
                misused.add(f"{node.value.id}:attribute")
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            if isinstance(node.func, ast.Name) and node.func.id in introduced_none:
                misused.add(f"{node.func.id}:call")
            self.generic_visit(node)

        def visit_For(self, node: ast.For) -> None:
            if isinstance(node.iter, ast.Name) and node.iter.id in introduced_none:
                misused.add(f"{node.iter.id}:iteration")
            self.generic_visit(node)

        def visit_Compare(self, node: ast.Compare) -> None:
            for op, comparator in zip(node.ops, node.comparators):
                if isinstance(op, (ast.In, ast.NotIn)) and isinstance(comparator, ast.Name) and comparator.id in introduced_none:
                    misused.add(f"{comparator.id}:membership")
            self.generic_visit(node)

    Visitor().visit(tree)
    return sorted(misused)


def _python_none_assignment_names(source: str) -> set[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and node.value.value is None:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.value, ast.Constant) and node.value.value is None:
            if isinstance(node.target, ast.Name):
                names.add(node.target.id)
    return names


def _python_introduced_function_object_arithmetic_details(before_source: str, after_source: str) -> list[str]:
    try:
        before_tree = ast.parse(before_source)
        after_tree = ast.parse(after_source)
    except SyntaxError:
        return []
    function_names = {
        node.name
        for node in ast.walk(after_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if not function_names:
        return []
    before_details = _python_function_object_arithmetic_details(before_tree, function_names)
    after_details = _python_function_object_arithmetic_details(after_tree, function_names)
    return sorted(after_details - before_details)


def _python_function_object_arithmetic_details(tree: ast.AST, function_names: set[str]) -> set[str]:
    details: set[str] = set()

    def operand_names(node: ast.AST) -> set[str]:
        names: set[str] = set()
        if isinstance(node, ast.Name):
            names.add(node.id)
        for child in ast.iter_child_nodes(node):
            names.update(operand_names(child))
        return names

    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp):
            for name in sorted(operand_names(node) & function_names):
                details.add(f"{name}:binop")
    return details


def _python_none_return_value_counts(source: str) -> dict[str, int]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    counts: dict[str, int] = {}

    def _visit(node: ast.AST, prefix: str = "") -> None:
        for child in getattr(node, "body", []):
            if isinstance(child, ast.ClassDef):
                qualified = f"{prefix}.{child.name}" if prefix else child.name
                _visit(child, qualified)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualified = f"{prefix}.{child.name}" if prefix else child.name
                count = 0
                for body_node in _python_function_body_nodes(child):
                    if (
                        isinstance(body_node, ast.Return)
                        and isinstance(body_node.value, ast.Constant)
                        and body_node.value.value is None
                    ):
                        count += 1
                counts[qualified] = count
                _visit(child, qualified)

    _visit(tree)
    return counts


def _python_return_value_counts(source: str) -> dict[str, int]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    counts: dict[str, int] = {}

    def _visit(node: ast.AST, prefix: str = "") -> None:
        for child in getattr(node, "body", []):
            if isinstance(child, ast.ClassDef):
                qualified = f"{prefix}.{child.name}" if prefix else child.name
                _visit(child, qualified)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualified = f"{prefix}.{child.name}" if prefix else child.name
                count = 0
                for body_node in _python_function_body_nodes(child):
                    if isinstance(body_node, ast.Return) and body_node.value is not None:
                        count += 1
                counts[qualified] = count
                _visit(child, qualified)

    _visit(tree)
    return counts


def _python_suspicious_boolean_return_flip_names(before_source: str, after_source: str) -> list[str]:
    before = _python_terminal_boolean_returns(before_source)
    after = _python_terminal_boolean_returns(after_source)
    suspicious: list[str] = []
    for name, before_value in before.items():
        after_value = after.get(name)
        if after_value is not None and before_value is not after_value:
            suspicious.append(name)
    return sorted(suspicious)


def _python_terminal_boolean_returns(source: str) -> dict[str, bool]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    returns: dict[str, bool] = {}

    def _visit(node: ast.AST, prefix: str = "") -> None:
        for child in getattr(node, "body", []):
            if isinstance(child, ast.ClassDef):
                qualified = f"{prefix}.{child.name}" if prefix else child.name
                _visit(child, qualified)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualified = f"{prefix}.{child.name}" if prefix else child.name
                body = list(child.body)
                if (
                    body
                    and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)
                ):
                    body = body[1:]
                if body:
                    last = body[-1]
                    if (
                        isinstance(last, ast.Return)
                        and isinstance(last.value, ast.Constant)
                        and isinstance(last.value.value, bool)
                    ):
                        returns[qualified] = bool(last.value.value)
                _visit(child, qualified)

    _visit(tree)
    return returns


def _python_function_local_binding_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[set[str], set[str]]:
    local_names: set[str] = set()
    declared_external: set[str] = set()
    for child in _python_function_body_nodes(node):
        if isinstance(child, (ast.Global, ast.Nonlocal)):
            declared_external.update(str(name) for name in child.names if str(name).strip())
            continue
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
            local_names.add(child.id)
            continue
        if isinstance(child, ast.Import):
            for alias in child.names:
                local_names.add(str(alias.asname or alias.name).split(".", 1)[0])
            continue
        if isinstance(child, ast.ImportFrom):
            for alias in child.names:
                if alias.name == "*":
                    continue
                local_names.add(str(alias.asname or alias.name).split(".", 1)[0])
            continue
        if isinstance(child, ast.ExceptHandler) and child.name:
            local_names.add(str(child.name))
    return local_names, declared_external


def _python_function_body_nodes(node: ast.FunctionDef | ast.AsyncFunctionDef):
    def _walk(current: ast.AST):
        if isinstance(
            current,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
                ast.Lambda,
            ),
        ):
            return
        yield current
        for nested in ast.iter_child_nodes(current):
            yield from _walk(nested)

    for body_item in node.body:
        yield from _walk(body_item)


def _python_function_parameters(source: str) -> dict[str, set[str]]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    parameters: dict[str, set[str]] = {}

    def _visit(node: ast.AST, prefix: str = "") -> None:
        for child in getattr(node, "body", []):
            if isinstance(child, ast.ClassDef):
                qualified = f"{prefix}.{child.name}" if prefix else child.name
                _visit(child, qualified)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualified = f"{prefix}.{child.name}" if prefix else child.name
                args = set()
                args.update(arg.arg for arg in child.args.posonlyargs)
                args.update(arg.arg for arg in child.args.args)
                args.update(arg.arg for arg in child.args.kwonlyargs)
                if child.args.vararg is not None:
                    args.add(child.args.vararg.arg)
                if child.args.kwarg is not None:
                    args.add(child.args.kwarg.arg)
                parameters[qualified] = {arg for arg in args if arg not in {"self", "cls"}}
                _visit(child, qualified)

    _visit(tree)
    return parameters


def _python_function_used_names(source: str) -> dict[str, set[str]]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    used: dict[str, set[str]] = {}

    def _visit(node: ast.AST, prefix: str = "") -> None:
        for child in getattr(node, "body", []):
            if isinstance(child, ast.ClassDef):
                qualified = f"{prefix}.{child.name}" if prefix else child.name
                _visit(child, qualified)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualified = f"{prefix}.{child.name}" if prefix else child.name
                body_names: set[str] = set()
                for body_item in child.body:
                    for descendant in ast.walk(body_item):
                        if isinstance(descendant, ast.Name) and isinstance(descendant.ctx, ast.Load):
                            body_names.add(descendant.id)
                used[qualified] = body_names
                _visit(child, qualified)

    _visit(tree)
    return used


def _python_function_first_name_contexts(source: str) -> dict[str, dict[str, str]]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    contexts: dict[str, dict[str, str]] = {}

    def _visit(node: ast.AST, prefix: str = "") -> None:
        for child in getattr(node, "body", []):
            if isinstance(child, ast.ClassDef):
                qualified = f"{prefix}.{child.name}" if prefix else child.name
                _visit(child, qualified)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualified = f"{prefix}.{child.name}" if prefix else child.name
                function_contexts: dict[str, str] = {}
                for body_item in child.body:
                    _record_first_body_name_contexts(body_item, function_contexts)
                contexts[qualified] = function_contexts
                _visit(child, qualified)

    _visit(tree)
    return contexts


def _record_first_body_name_contexts(node: ast.AST, contexts: dict[str, str]) -> None:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
        return
    if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
        for generator in node.generators:
            _record_first_body_name_contexts(generator.target, contexts)
        _record_first_body_name_contexts(node.elt, contexts)
        for generator in node.generators:
            _record_first_body_name_contexts(generator.iter, contexts)
            for condition in generator.ifs:
                _record_first_body_name_contexts(condition, contexts)
        return
    if isinstance(node, ast.DictComp):
        for generator in node.generators:
            _record_first_body_name_contexts(generator.target, contexts)
        _record_first_body_name_contexts(node.key, contexts)
        _record_first_body_name_contexts(node.value, contexts)
        for generator in node.generators:
            _record_first_body_name_contexts(generator.iter, contexts)
            for condition in generator.ifs:
                _record_first_body_name_contexts(condition, contexts)
        return
    if isinstance(node, ast.Name) and node.id not in contexts:
        if isinstance(node.ctx, ast.Load):
            contexts[node.id] = "load"
        elif isinstance(node.ctx, (ast.Store, ast.Del)):
            contexts[node.id] = "store"
    for child in ast.iter_child_nodes(node):
        _record_first_body_name_contexts(child, contexts)


def _normalize_patch_text_for_repeat_guard(patch_text: str) -> str:
    lines = [line.rstrip() for line in str(patch_text or "").strip().splitlines()]
    return "\n".join(lines).strip()


def _looks_like_python_literal_constant(value: str) -> bool:
    stripped = str(value or "").strip().rstrip(",")
    if not stripped:
        return False
    try:
        node = ast.parse(stripped, mode="eval")
    except SyntaxError:
        return False
    return isinstance(node.body, ast.Constant)


def synthesize_stricter_task(
    task: TaskSpec,
    *,
    task_id: str | None = None,
    prompt_prefix: str = "Solve the same task under a stricter synthesized verifier.",
    benchmark_family: str = "verifier_memory",
    memory_source: str = "verifier",
    extra_metadata: dict[str, object] | None = None,
) -> TaskSpec:
    metadata = dict(task.metadata)
    metadata.update(
        {
            "benchmark_family": benchmark_family,
            "memory_source": memory_source,
            "origin_benchmark_family": str(task.metadata.get("benchmark_family", "bounded")),
            "verifier_strength": "synthesized_strict",
        }
    )
    if extra_metadata:
        metadata.update(extra_metadata)
    forbidden_files = list(task.forbidden_files)
    workspace_name = Path(task.workspace_subdir).name.strip()
    for relative_path in task.expected_files:
        nested_path = f"{workspace_name}/{relative_path}".strip("/")
        if workspace_name and nested_path != relative_path and nested_path not in forbidden_files:
            forbidden_files.append(nested_path)
    prompt = f"{prompt_prefix} {task.prompt}".strip()
    return TaskSpec(
        task_id=task_id or task.task_id,
        prompt=prompt,
        workspace_subdir=task.workspace_subdir,
        setup_commands=list(task.setup_commands),
        success_command=task.success_command,
        suggested_commands=list(task.suggested_commands),
        expected_files=list(task.expected_files),
        expected_output_substrings=list(task.expected_output_substrings),
        forbidden_files=forbidden_files,
        forbidden_output_substrings=list(task.forbidden_output_substrings),
        expected_file_contents=dict(task.expected_file_contents),
        max_steps=task.max_steps,
        metadata=metadata,
    )
