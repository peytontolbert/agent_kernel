from __future__ import annotations

from pathlib import Path
import re
import subprocess

from .schemas import CommandResult, TaskSpec, VerificationResult


class Verifier:
    def verify(self, task: TaskSpec, workspace: Path, result: CommandResult) -> VerificationResult:
        reasons: list[str] = []

        if result.timed_out:
            reasons.append("command timed out")
        if result.exit_code != 0:
            reasons.append(f"exit code was {result.exit_code}")

        for relative_path in task.expected_files:
            if not (workspace / relative_path).exists():
                reasons.append(f"missing expected file: {relative_path}")
        for relative_path in task.forbidden_files:
            if (workspace / relative_path).exists():
                reasons.append(f"forbidden file present: {relative_path}")
        for relative_path, expected_content in task.expected_file_contents.items():
            path = workspace / relative_path
            if not path.exists():
                reasons.append(f"missing expected file content target: {relative_path}")
                continue
            actual_content = path.read_text(encoding="utf-8")
            if actual_content != expected_content:
                reasons.append(f"unexpected file content: {relative_path}")

        combined_output = f"{result.stdout}\n{result.stderr}"
        for needle in task.expected_output_substrings:
            if needle not in combined_output:
                reasons.append(f"missing expected output: {needle}")
        for needle in task.forbidden_output_substrings:
            if needle in combined_output:
                reasons.append(f"forbidden output present: {needle}")

        reasons.extend(self._semantic_verification_reasons(task, workspace))

        return VerificationResult(
            passed=not reasons,
            reasons=reasons or ["verification passed"],
            command_result=result,
        )

    def _semantic_verification_reasons(self, task: TaskSpec, workspace: Path) -> list[str]:
        contract = task.metadata.get("semantic_verifier")
        if not isinstance(contract, dict):
            return []
        kind = str(contract.get("kind", "")).strip()
        if kind == "repo_chore_review":
            return self._verify_repo_chore_review(workspace, contract)
        if kind == "git_repo_review":
            return self._verify_git_repo_review(workspace, contract)
        return []

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
