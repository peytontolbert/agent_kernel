from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .config import KernelConfig
from .schemas import TaskSpec
from .world_model_improvement import retained_world_model_controls, retained_world_model_planning_controls


class WorldModel:
    def __init__(self, config: KernelConfig | None = None) -> None:
        self.config = config or KernelConfig()
        self._controls_cache: dict[str, object] | None = None
        self._planning_controls_cache: dict[str, object] | None = None

    def summarize(
        self,
        task: TaskSpec,
        graph_summary: dict[str, object] | None = None,
        *,
        workspace: Path | None = None,
        workspace_snapshot: dict[str, str] | None = None,
    ) -> dict[str, object]:
        graph_summary = graph_summary or {}
        expected = sorted(set(task.expected_files) | set(task.expected_file_contents))
        horizon = "long_horizon" if str(task.metadata.get("difficulty", "")) == "long_horizon" else "bounded"
        workflow = self._workflow_summary(task)
        preserved = list(workflow.get("preserved_paths", []))
        summary = {
            "expected_artifacts": expected,
            "forbidden_artifacts": list(task.forbidden_files),
            "preserved_artifacts": preserved,
            "target_outputs": list(task.expected_output_substrings),
            "benchmark_family": str(task.metadata.get("benchmark_family", "bounded")),
            "difficulty": str(task.metadata.get("difficulty", "unknown")),
            "horizon": horizon,
            "graph_neighbor_count": len(list(graph_summary.get("neighbors", []))),
            "workflow_kind": str(workflow.get("kind", "")),
            "workflow_expected_changed_paths": list(workflow.get("expected_changed_paths", [])),
            "workflow_report_paths": list(workflow.get("report_paths", [])),
            "workflow_generated_paths": list(workflow.get("generated_paths", [])),
            "workflow_preserved_paths": list(workflow.get("preserved_paths", [])),
            "workflow_branch_targets": list(workflow.get("branch_targets", [])),
            "workflow_required_tests": list(workflow.get("required_tests", [])),
            "workflow_required_merges": list(workflow.get("required_merges", [])),
            "workflow_shared_repo": bool(workflow.get("shared_repo", False)),
        }
        if workspace is not None:
            summary.update(
                self._workspace_state_summary(
                    task,
                    workspace,
                    workflow=workflow,
                    workspace_snapshot=workspace_snapshot or {},
                )
            )
        return summary

    def score_command(self, summary: dict[str, object], command: str) -> int:
        normalized = str(command)
        score = 0
        for path in summary.get("missing_expected_artifacts", []):
            if path and path in normalized:
                score += self._control_int("expected_artifact_score_weight", 3) + 2
        for path in summary.get("expected_artifacts", []):
            if path and path in normalized:
                score += self._control_int("expected_artifact_score_weight", 3)
        cleanup_command = "rm " in normalized or "rm -f " in normalized or "unlink " in normalized
        for path in summary.get("present_forbidden_artifacts", []):
            if path and path in normalized and cleanup_command:
                score += self._control_int("forbidden_cleanup_score_weight", 4)
        for path in summary.get("preserved_artifacts", []):
            if path and path in normalized:
                score += self._control_int("preserved_artifact_score_weight", 2)
        for path in summary.get("forbidden_artifacts", []):
            if path and path in normalized:
                score -= self._control_int("forbidden_artifact_penalty", 5)
        if str(summary.get("horizon", "")) == "long_horizon":
            if "mkdir -p " in normalized or "cp " in normalized:
                score += self._control_int("long_horizon_scaffold_bonus", 1)
        for branch in summary.get("workflow_branch_targets", []):
            if branch and branch in normalized:
                score += self._control_int("workflow_branch_target_score_weight", 4)
        for path in summary.get("workflow_expected_changed_paths", []):
            if path and path in normalized:
                score += self._control_int("workflow_changed_path_score_weight", 3)
        for path in summary.get("workflow_generated_paths", []):
            if path and path in normalized:
                score += self._control_int("workflow_generated_path_score_weight", 3)
        for path in summary.get("workflow_report_paths", []):
            if path and path in normalized:
                score += self._control_int("workflow_report_path_score_weight", 2)
        for path in summary.get("workflow_preserved_paths", []):
            if path and path in normalized:
                score += self._control_int("workflow_preserved_path_score_weight", 2)
        if summary.get("workflow_required_tests") and ("test" in normalized or "./" in normalized):
            score += self._control_int("required_tests_score_weight", 2)
        if summary.get("workflow_required_merges") and "git merge" in normalized:
            score += self._control_int("required_merges_score_weight", 4)
        return score

    def simulate_command_effect(self, summary: dict[str, object], command: str) -> dict[str, object]:
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
        return {
            "predicted_outputs": predicted_outputs,
            "predicted_conflicts": predicted_conflicts,
            "predicted_preserved": predicted_preserved,
            "predicted_workflow_paths": predicted_workflow_paths,
            "predicted_progress_gain": predicted_progress_gain,
            "score": self.score_command(summary, command),
        }

    def score_retrieved_span(self, summary: dict[str, object], item: dict[str, object]) -> int:
        text = str(item.get("text", ""))
        metadata = item.get("metadata") or {}
        span_type = str(item.get("span_type", ""))
        score = 0
        for path in summary.get("expected_artifacts", []):
            if path and (path in text or path in str(metadata)):
                score += self._control_int("retrieved_expected_artifact_score_weight", 3)
        for path in summary.get("forbidden_artifacts", []):
            if path and (path in text or path in str(metadata)):
                score -= self._control_int("retrieved_forbidden_artifact_penalty", 4)
        for path in summary.get("preserved_artifacts", []):
            if path and (path in text or path in str(metadata)):
                score += self._control_int("retrieved_preserved_artifact_score_weight", 2)
        if span_type in {"agent:command_template", "agent:procedure"}:
            score += 1
        if str(summary.get("horizon", "")) == "long_horizon" and span_type in {"agent:task", "doc:readme_chunk"}:
            score += 1
        for path in summary.get("workflow_expected_changed_paths", []):
            if path and (path in text or path in str(metadata)):
                score += self._control_int("retrieved_workflow_changed_path_score_weight", 2)
        for path in summary.get("workflow_report_paths", []):
            if path and (path in text or path in str(metadata)):
                score += self._control_int("retrieved_workflow_report_path_score_weight", 1)
        return score

    def retained_planning_controls(self) -> dict[str, object]:
        if self._planning_controls_cache is not None:
            return dict(self._planning_controls_cache)
        payload = self._retained_payload()
        self._planning_controls_cache = retained_world_model_planning_controls(payload)
        return dict(self._planning_controls_cache)

    def _controls(self) -> dict[str, object]:
        if self._controls_cache is not None:
            return self._controls_cache
        payload = self._retained_payload()
        self._controls_cache = retained_world_model_controls(payload)
        return self._controls_cache

    def _retained_payload(self) -> object:
        if not bool(self.config.use_world_model):
            return {}
        path = self.config.world_model_proposals_path
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _control_int(self, field: str, default: int) -> int:
        value = self._controls().get(field, default)
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def capture_workspace_snapshot(self, task: TaskSpec, workspace: Path) -> dict[str, str]:
        snapshot: dict[str, str] = {}
        for path in self._observed_paths(task):
            snapshot[path] = self._fingerprint_path(workspace / path)
        return snapshot

    def describe_progress(self, summary: dict[str, object], *, command: str = "", step_index: int = 0) -> str:
        progress = float(summary.get("completion_ratio", 0.0))
        missing_expected = len(list(summary.get("missing_expected_artifacts", [])))
        forbidden_present = len(list(summary.get("present_forbidden_artifacts", [])))
        changed_preserved = len(list(summary.get("changed_preserved_artifacts", [])))
        prefix = f"step {step_index}: " if step_index else ""
        command_part = f"command={command!r}; " if command else ""
        return (
            f"{prefix}{command_part}"
            f"progress={progress:.2f} "
            f"missing_expected={missing_expected} "
            f"present_forbidden={forbidden_present} "
            f"changed_preserved={changed_preserved}"
        ).strip()

    def describe_transition(
        self,
        previous_summary: dict[str, object] | None,
        current_summary: dict[str, object] | None,
    ) -> dict[str, object]:
        previous = previous_summary or {}
        current = current_summary or {}
        previous_completion = float(previous.get("completion_ratio", 0.0))
        current_completion = float(current.get("completion_ratio", 0.0))
        progress_delta = round(current_completion - previous_completion, 3)

        def _new_items(key: str) -> list[str]:
            before = {str(item) for item in previous.get(key, [])}
            after = [str(item) for item in current.get(key, [])]
            return [item for item in after if item not in before]

        def _cleared_items(key: str) -> list[str]:
            before = {str(item) for item in previous.get(key, [])}
            after = {str(item) for item in current.get(key, [])}
            return sorted(item for item in before if item not in after)

        regressions = [
            *(_new_items("missing_expected_artifacts")),
            *(_new_items("present_forbidden_artifacts")),
            *(_new_items("changed_preserved_artifacts")),
        ]
        transition = {
            "progress_delta": progress_delta,
            "newly_materialized_expected_artifacts": _new_items("existing_expected_artifacts"),
            "newly_satisfied_expected_contents": _new_items("satisfied_expected_contents"),
            "cleared_forbidden_artifacts": _cleared_items("present_forbidden_artifacts"),
            "newly_changed_preserved_artifacts": _new_items("changed_preserved_artifacts"),
            "newly_updated_workflow_paths": _new_items("updated_workflow_paths"),
            "newly_updated_generated_paths": _new_items("updated_generated_paths"),
            "newly_updated_report_paths": _new_items("updated_report_paths"),
            "regressions": regressions,
            "no_progress": progress_delta <= 0 and not regressions,
        }
        transition["state_change_score"] = (
            len(transition["newly_materialized_expected_artifacts"])
            + len(transition["newly_satisfied_expected_contents"])
            + len(transition["cleared_forbidden_artifacts"])
            + len(transition["newly_updated_workflow_paths"])
            + len(transition["newly_updated_generated_paths"])
            + len(transition["newly_updated_report_paths"])
            - len(regressions)
        )
        return transition

    def _workspace_state_summary(
        self,
        task: TaskSpec,
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
            if self._file_contents_match(workspace / path, content)
        ]
        unsatisfied_expected_contents = [
            path for path in task.expected_file_contents if path not in satisfied_expected_contents
        ]
        present_forbidden = [path for path in task.forbidden_files if (workspace / path).exists()]
        intact_preserved: list[str] = []
        changed_preserved: list[str] = []
        missing_preserved: list[str] = []
        for path in workflow.get("preserved_paths", []):
            current = self._fingerprint_path(workspace / path)
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
            if self._path_updated(workspace / path, workspace_snapshot.get(path, ""))
        ]
        updated_generated_paths = [
            path
            for path in workflow.get("generated_paths", [])
            if self._path_updated(workspace / path, workspace_snapshot.get(path, ""))
        ]
        updated_report_paths = [
            path
            for path in workflow.get("report_paths", [])
            if self._path_updated(workspace / path, workspace_snapshot.get(path, ""))
        ]
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
        }

    @staticmethod
    def _workflow_summary(task: TaskSpec) -> dict[str, object]:
        contract = task.metadata.get("semantic_verifier", {})
        workflow_guard = task.metadata.get("workflow_guard", {})
        verifier = dict(contract) if isinstance(contract, dict) else {}
        guard = dict(workflow_guard) if isinstance(workflow_guard, dict) else {}
        report_paths = [
            str(rule.get("path", "")).strip()
            for rule in verifier.get("report_rules", [])
            if isinstance(rule, dict) and str(rule.get("path", "")).strip()
        ]
        required_tests = [
            str(rule.get("label", "")).strip() or "test command"
            for rule in verifier.get("test_commands", [])
            if isinstance(rule, dict)
        ]
        branch_targets = [
            value
            for value in (
                str(verifier.get("expected_branch", "")).strip(),
                str(guard.get("worker_branch", "")).strip(),
                str(guard.get("target_branch", "")).strip(),
            )
            if value
        ]
        return {
            "kind": str(verifier.get("kind", "")).strip(),
            "expected_changed_paths": [
                str(path).strip()
                for path in verifier.get("expected_changed_paths", [])
                if str(path).strip()
            ],
            "generated_paths": [
                str(path).strip()
                for path in verifier.get("generated_paths", [])
                if str(path).strip()
            ],
            "report_paths": report_paths,
            "required_tests": required_tests,
            "required_merges": [
                str(path).strip()
                for path in verifier.get("required_merged_branches", [])
                if str(path).strip()
            ],
            "preserved_paths": [
                str(path).strip()
                for path in verifier.get("preserved_paths", [])
                if str(path).strip()
            ],
            "branch_targets": branch_targets,
            "shared_repo": bool(guard.get("shared_repo_id")),
        }

    @staticmethod
    def _observed_paths(task: TaskSpec) -> list[str]:
        workflow = WorldModel._workflow_summary(task)
        paths: list[str] = []
        for path in (
            list(task.expected_files)
            + list(task.expected_file_contents.keys())
            + list(task.forbidden_files)
            + list(workflow.get("expected_changed_paths", []))
            + list(workflow.get("generated_paths", []))
            + list(workflow.get("report_paths", []))
            + list(workflow.get("preserved_paths", []))
        ):
            normalized = str(path).strip()
            if normalized and normalized not in paths:
                paths.append(normalized)
        return paths

    @staticmethod
    def _file_contents_match(path: Path, expected_content: str) -> bool:
        if not path.exists() or not path.is_file():
            return False
        try:
            return path.read_text(encoding="utf-8") == expected_content
        except OSError:
            return False

    @staticmethod
    def _path_updated(path: Path, baseline: str) -> bool:
        current = WorldModel._fingerprint_path(path)
        if not current:
            return False
        if not baseline:
            return True
        return current != baseline

    @staticmethod
    def _fingerprint_path(path: Path) -> str:
        if not path.exists():
            return ""
        try:
            if path.is_file():
                digest = hashlib.sha1(path.read_bytes()).hexdigest()
                return f"file:{digest}"
            if path.is_dir():
                items = sorted(str(child.relative_to(path)) for child in path.rglob("*"))
                digest = hashlib.sha1("\n".join(items).encode("utf-8")).hexdigest()
                return f"dir:{digest}"
        except OSError:
            return ""
        return "exists"
