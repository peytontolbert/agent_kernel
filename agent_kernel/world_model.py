from __future__ import annotations

import difflib
import hashlib
import json
from pathlib import Path

from .config import KernelConfig
from .schemas import TaskSpec
from .world_model_improvement import retained_world_model_controls, retained_world_model_planning_controls

_MAX_WORKSPACE_FILE_PREVIEWS = 6
_MAX_WORKSPACE_PREVIEW_BYTES = 4096
_MAX_WORKSPACE_PREVIEW_CHARS = 400
_MAX_PRIORITY_WORKSPACE_PREVIEWS = 4
_MAX_PRIORITY_WORKSPACE_PREVIEW_BYTES = 16384
_MAX_PRIORITY_WORKSPACE_PREVIEW_CHARS = 1600
_MAX_TARGETED_PREVIEW_WINDOWS = 3


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

    def prioritized_long_horizon_hotspots(
        self,
        task: TaskSpec,
        summary: dict[str, object],
        *,
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
            self._safe_float(learned.get("progress_signal"), 0.0),
            self._safe_float(learned.get("world_progress_score"), 0.0),
            self._safe_float(learned.get("decoder_world_progress_score"), 0.0),
            self._safe_float(learned.get("transition_progress_score"), 0.0),
        )
        learned_risk_signal = max(
            self._safe_float(learned.get("risk_signal"), 0.0),
            self._safe_float(learned.get("world_risk_score"), 0.0),
            self._safe_float(learned.get("decoder_world_risk_score"), 0.0),
            self._safe_float(learned.get("transition_regression_score"), 0.0),
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

        active_path = self._active_subgoal_path(active_subgoal)
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
        preview_paths = self._prioritized_preview_paths(
            unsatisfied_expected_contents=unsatisfied_expected_contents,
            present_forbidden=present_forbidden,
            changed_preserved=changed_preserved,
            updated_workflow_paths=updated_workflow_paths,
            updated_generated_paths=updated_generated_paths,
            updated_report_paths=updated_report_paths,
            expected=expected,
        )
        workspace_file_previews = self._workspace_file_previews(
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
            "workspace_file_previews": workspace_file_previews,
        }

    @staticmethod
    def _prioritized_preview_paths(
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

    @staticmethod
    def _active_subgoal_path(active_subgoal: str) -> str:
        normalized = str(active_subgoal).strip()
        for prefix in (
            "remove forbidden artifact ",
            "materialize expected artifact ",
            "preserve required artifact ",
            "update workflow path ",
            "regenerate generated artifact ",
            "write workflow report ",
        ):
            if normalized.startswith(prefix):
                return normalized.removeprefix(prefix).strip()
        return ""

    @staticmethod
    def _safe_float(value: object, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _workspace_file_previews(
        workspace: Path,
        candidate_paths: list[str],
        *,
        expected_file_contents: dict[str, str] | None = None,
    ) -> dict[str, dict[str, object]]:
        previews: dict[str, dict[str, object]] = {}
        preview_targets = expected_file_contents or {}
        for index, path in enumerate(candidate_paths):
            if len(previews) >= _MAX_WORKSPACE_FILE_PREVIEWS:
                break
            max_bytes = (
                _MAX_PRIORITY_WORKSPACE_PREVIEW_BYTES
                if index < _MAX_PRIORITY_WORKSPACE_PREVIEWS
                else _MAX_WORKSPACE_PREVIEW_BYTES
            )
            max_chars = (
                _MAX_PRIORITY_WORKSPACE_PREVIEW_CHARS
                if index < _MAX_PRIORITY_WORKSPACE_PREVIEWS
                else _MAX_WORKSPACE_PREVIEW_CHARS
            )
            preview_windows = WorldModel._targeted_text_file_previews(
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
                preview = WorldModel._text_file_preview(
                    workspace / path,
                    max_bytes=max_bytes,
                    max_chars=max_chars,
                )
            if preview is not None:
                previews[path] = preview
        return previews

    @staticmethod
    def _text_file_preview(
        path: Path,
        *,
        max_bytes: int = _MAX_WORKSPACE_PREVIEW_BYTES,
        max_chars: int = _MAX_WORKSPACE_PREVIEW_CHARS,
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
        content = WorldModel._bounded_text_prefix(
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
            "omitted_prefix_sha1": WorldModel._sha1_text(""),
            "omitted_suffix_sha1": (
                WorldModel._sha1_text(full_content[len(edit_content) :])
                if truncated
                else WorldModel._sha1_text("")
            ),
            "omitted_sha1": (
                WorldModel._sha1_text(full_content[len(edit_content) :])
                if truncated
                else ""
            ),
        }

    @staticmethod
    def _targeted_text_file_previews(
        path: Path,
        *,
        expected_content: str,
        max_bytes: int = _MAX_WORKSPACE_PREVIEW_BYTES,
        max_chars: int = _MAX_WORKSPACE_PREVIEW_CHARS,
    ) -> list[dict[str, object]]:
        if not expected_content:
            return []
        current_content = WorldModel._read_text_file(path)
        if current_content is None or current_content == expected_content:
            return []
        if WorldModel._bounded_text_prefix(current_content, max_bytes=max_bytes, max_chars=max_chars) == current_content:
            return []
        current_lines = current_content.splitlines(keepends=True)
        expected_lines = expected_content.splitlines(keepends=True)
        all_windows = WorldModel._targeted_preview_windows(
            current_lines=current_lines,
            expected_lines=expected_lines,
            max_bytes=max_bytes,
            max_chars=max_chars,
        )
        windows = all_windows[:_MAX_TARGETED_PREVIEW_WINDOWS]
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
                    "omitted_prefix_sha1": WorldModel._sha1_text("".join(current_lines[:start])),
                    "omitted_suffix_sha1": WorldModel._sha1_text("".join(current_lines[end:])),
                    "omitted_sha1": WorldModel._sha1_text("".join(current_lines[end:])),
                }
            )
        exact_proof_windows = WorldModel._exact_targeted_preview_proof_windows(
            windows=all_windows,
            retained_window_count=retained_window_count,
        )
        bridged_previews = WorldModel._bridged_targeted_preview_windows(
            current_lines=current_lines,
            expected_lines=expected_lines,
            windows=all_windows,
            max_bytes=max_bytes,
            max_chars=max_chars,
        )
        bridged_preview_runs = WorldModel._bridged_targeted_preview_window_runs(
            bridged_windows=bridged_previews
        )
        current_proof_regions = WorldModel._hidden_gap_current_proof_regions(
            bridged_runs=bridged_preview_runs
        )
        sparse_current_proof_regions = WorldModel._sparse_hidden_gap_current_proof_regions(
            windows=all_windows,
            bridged_windows=bridged_previews,
            start_region_index=len(current_proof_regions),
        )
        if sparse_current_proof_regions:
            current_proof_regions = WorldModel._dedupe_hidden_gap_current_proof_regions(
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

    @staticmethod
    def _dedupe_hidden_gap_current_proof_regions(
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

    @staticmethod
    def _exact_targeted_preview_proof_windows(
        *,
        windows: list[tuple[int, int, int, int]],
        retained_window_count: int,
    ) -> list[dict[str, object]]:
        proofs: list[dict[str, object]] = []
        if retained_window_count >= len(windows):
            return proofs
        for window_index, (start, end, target_start, target_end) in enumerate(windows[retained_window_count:], start=retained_window_count):
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

    @staticmethod
    def _bridged_targeted_preview_windows(
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
            include_current_content = WorldModel._fits_preview_budget(
                hidden_current_content,
                max_bytes=max_bytes,
                max_chars=max_chars,
            )
            include_target_content = WorldModel._fits_dual_preview_budget(
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

    @staticmethod
    def _bridged_targeted_preview_window_runs(
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

    @staticmethod
    def _hidden_gap_current_proof_regions(
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

    @staticmethod
    def _sparse_hidden_gap_current_proof_regions(
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
                and WorldModel._bridge_window_has_current_proof(bridge)
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
                region = WorldModel._sparse_hidden_gap_current_proof_region(
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

    @staticmethod
    def _bridge_window_has_current_proof(bridge: dict[str, object]) -> bool:
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

    @staticmethod
    def _sparse_hidden_gap_current_proof_region(
        *,
        windows: list[tuple[int, int, int, int]],
        bridge_by_pair: dict[tuple[int, int], dict[str, object]],
        window_start_index: int,
        window_end_index: int,
        region_index: int,
    ) -> dict[str, object] | None:
        if window_start_index < 0 or window_end_index >= len(windows):
            return None
        while window_start_index > 0 and WorldModel._windows_have_zero_gap(
            windows[window_start_index - 1],
            windows[window_start_index],
        ):
            window_start_index -= 1
        while window_end_index < len(windows) - 1 and WorldModel._windows_have_zero_gap(
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

    @staticmethod
    def _windows_have_zero_gap(
        left_window: tuple[int, int, int, int],
        right_window: tuple[int, int, int, int],
    ) -> bool:
        left_start, left_end, left_target_start, left_target_end = left_window
        right_start, right_end, right_target_start, right_target_end = right_window
        return (
            max(0, right_start - left_end) <= 0
            and max(0, right_target_start - left_target_end) <= 0
        )

    @staticmethod
    def _targeted_preview_windows(
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
            start, end, target_start, target_end = WorldModel._expanded_preview_window(
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
                merged = WorldModel._merge_preview_windows(
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

    @staticmethod
    def _expanded_preview_window(
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
        if not WorldModel._fits_dual_preview_budget(
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
                if WorldModel._fits_dual_preview_budget(
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
                if WorldModel._fits_dual_preview_budget(
                    baseline_content="".join(current_lines[start : end + 1]),
                    target_content="".join(expected_lines[target_start : target_end + 1]),
                    max_bytes=max_bytes,
                    max_chars=max_chars,
                ):
                    end += 1
                    target_end += 1
                    expanded = True
        return start, end, target_start, target_end

    @staticmethod
    def _merge_preview_windows(
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
        if not WorldModel._fits_dual_preview_budget(
            baseline_content="".join(current_lines[start:end]),
            target_content="".join(expected_lines[target_start:target_end]),
            max_bytes=max_bytes,
            max_chars=max_chars,
        ):
            return None
        return start, end, target_start, target_end

    @staticmethod
    def _read_text_file(path: Path) -> str | None:
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

    @staticmethod
    def _fits_preview_budget(content: str, *, max_bytes: int, max_chars: int) -> bool:
        return len(content) <= max_chars and len(content.encode("utf-8")) <= max_bytes

    @staticmethod
    def _fits_dual_preview_budget(
        *,
        baseline_content: str,
        target_content: str,
        max_bytes: int,
        max_chars: int,
    ) -> bool:
        return WorldModel._fits_preview_budget(baseline_content, max_bytes=max_bytes, max_chars=max_chars) and (
            WorldModel._fits_preview_budget(target_content, max_bytes=max_bytes, max_chars=max_chars)
        )

    @staticmethod
    def _bounded_text_prefix(content: str, *, max_bytes: int, max_chars: int) -> str:
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

    @staticmethod
    def _sha1_text(value: str) -> str:
        return hashlib.sha1(value.encode("utf-8")).hexdigest()
