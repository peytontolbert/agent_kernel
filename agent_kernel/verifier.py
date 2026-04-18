from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from .schemas import CommandResult, TaskSpec, VerificationResult, classify_verification_reason


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
        semantic_failures = max(0, len(reasons) - failed_checks)
        total_checks += semantic_failures
        failed_checks += semantic_failures
        failure_codes = [
            code
            for code in (classify_verification_reason(reason) for reason in reasons)
            if code
        ]
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
            ],
        )

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
