from __future__ import annotations

import difflib
import hashlib
import json
from pathlib import Path

from .config import KernelConfig
from .schemas import TaskSpec
from .extensions.improvement.world_model_improvement import retained_world_model_controls, retained_world_model_planning_controls
from .extensions import world_model_analysis_support
from .extensions import world_model_preview_support
from .extensions import world_model_prediction_support

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
        self._bootstrap_step_priors_cache: dict[tuple[str, str, str], dict[str, object]] | None = None
        self._state_conditioned_step_model_cache: dict[tuple[str, str, str, str], dict[str, object]] | None = None

    @staticmethod
    def _max_workspace_file_previews() -> int:
        return _MAX_WORKSPACE_FILE_PREVIEWS

    @staticmethod
    def _max_workspace_preview_bytes() -> int:
        return _MAX_WORKSPACE_PREVIEW_BYTES

    @staticmethod
    def _max_workspace_preview_chars() -> int:
        return _MAX_WORKSPACE_PREVIEW_CHARS

    @staticmethod
    def _max_priority_workspace_previews() -> int:
        return _MAX_PRIORITY_WORKSPACE_PREVIEWS

    @staticmethod
    def _max_priority_workspace_preview_bytes() -> int:
        return _MAX_PRIORITY_WORKSPACE_PREVIEW_BYTES

    @staticmethod
    def _max_priority_workspace_preview_chars() -> int:
        return _MAX_PRIORITY_WORKSPACE_PREVIEW_CHARS

    @staticmethod
    def _max_targeted_preview_windows() -> int:
        return _MAX_TARGETED_PREVIEW_WINDOWS

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
            "workflow_behavior_checks": list(workflow.get("behavior_checks", [])),
            "workflow_differential_checks": list(workflow.get("differential_checks", [])),
            "workflow_repo_invariants": list(workflow.get("repo_invariants", [])),
            "workflow_shared_repo": bool(workflow.get("shared_repo", False)),
            "semantic_episodes": [
                dict(item)
                for item in graph_summary.get("semantic_episodes", [])[:4]
                if isinstance(item, dict)
            ]
            if isinstance(graph_summary.get("semantic_episodes", []), list)
            else [],
            "semantic_prototypes": [
                dict(item)
                for item in graph_summary.get("semantic_prototypes", [])[:4]
                if isinstance(item, dict)
            ]
            if isinstance(graph_summary.get("semantic_prototypes", []), list)
            else [],
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

    score_command = world_model_prediction_support.score_command

    simulate_command_effect = world_model_prediction_support.simulate_command_effect

    simulate_command_sequence_effect = world_model_prediction_support.simulate_command_sequence_effect

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

    def _control_float(self, field: str, default: float) -> float:
        value = self._controls().get(field, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    _empirical_command_prediction = world_model_prediction_support.empirical_command_prediction

    _bootstrap_command_prediction = world_model_prediction_support.bootstrap_command_prediction

    _state_conditioned_command_prediction = (
        world_model_prediction_support.state_conditioned_command_prediction
    )

    _bootstrap_step_priors = world_model_prediction_support.bootstrap_step_priors

    _state_conditioned_step_model = world_model_prediction_support.state_conditioned_step_model

    def _aggregate_prediction(
        self,
        summary: dict[str, object],
        *,
        aggregate: dict[str, object],
        command_pattern: str,
    ) -> dict[str, object]:
        return world_model_prediction_support.aggregate_prediction(
            self,
            summary,
            aggregate=aggregate,
            command_pattern_value=command_pattern,
        )

    _merge_changed_path_counts = staticmethod(world_model_prediction_support.merge_changed_path_counts)

    _path_prediction_metrics = staticmethod(world_model_prediction_support.path_prediction_metrics)

    _transition_changed_paths = staticmethod(world_model_prediction_support.transition_changed_paths)

    _transition_verifier_delta = staticmethod(world_model_prediction_support.transition_verifier_delta)

    _command_pattern = staticmethod(world_model_prediction_support.command_pattern)

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
        changed_paths = []
        for key in (
            "newly_materialized_expected_artifacts",
            "newly_satisfied_expected_contents",
            "cleared_forbidden_artifacts",
            "newly_changed_preserved_artifacts",
            "newly_updated_workflow_paths",
            "newly_updated_generated_paths",
            "newly_updated_report_paths",
            "regressions",
        ):
            for item in transition.get(key, []):
                normalized = str(item).strip()
                if normalized and normalized not in changed_paths:
                    changed_paths.append(normalized)
        transition["state_change_score"] = (
            len(transition["newly_materialized_expected_artifacts"])
            + len(transition["newly_satisfied_expected_contents"])
            + len(transition["cleared_forbidden_artifacts"])
            + len(transition["newly_updated_workflow_paths"])
            + len(transition["newly_updated_generated_paths"])
            + len(transition["newly_updated_report_paths"])
            - len(regressions)
        )
        transition["verifier_delta"] = int(transition["state_change_score"])
        transition["changed_paths"] = changed_paths
        transition["edit_patches"] = self._preview_edit_patches(
            previous_summary=previous,
            current_summary=current,
            changed_paths=changed_paths,
        )
        return transition

    def _preview_edit_patches(
        self,
        *,
        previous_summary: dict[str, object],
        current_summary: dict[str, object],
        changed_paths: list[str],
    ) -> list[dict[str, object]]:
        previous_previews = previous_summary.get("workspace_file_previews", {})
        previous_previews = previous_previews if isinstance(previous_previews, dict) else {}
        current_previews = current_summary.get("workspace_file_previews", {})
        current_previews = current_previews if isinstance(current_previews, dict) else {}
        patches: list[dict[str, object]] = []
        for path in changed_paths[:4]:
            before_preview = previous_previews.get(path, {})
            after_preview = current_previews.get(path, {})
            before_text = str(before_preview.get("content", "")) if isinstance(before_preview, dict) else ""
            after_text = str(after_preview.get("content", "")) if isinstance(after_preview, dict) else ""
            if not before_text and not after_text:
                continue
            if before_text and after_text:
                status = "modified"
            elif after_text:
                status = "created"
            else:
                status = "deleted"
            patch_text = "".join(
                difflib.unified_diff(
                    before_text.splitlines(keepends=True),
                    after_text.splitlines(keepends=True),
                    fromfile=f"a/{path}",
                    tofile=f"b/{path}",
                    n=1,
                )
            ).strip()
            if not patch_text:
                continue
            additions = sum(1 for line in patch_text.splitlines() if line.startswith("+") and not line.startswith("+++"))
            deletions = sum(1 for line in patch_text.splitlines() if line.startswith("-") and not line.startswith("---"))
            patches.append(
                {
                    "path": str(path).strip(),
                    "status": status,
                    "patch": patch_text[:1200],
                    "patch_summary": f"{status} {path} (+{additions} -{deletions})",
                    "before_excerpt": before_text[:240],
                    "after_excerpt": after_text[:240],
                }
            )
        return patches

    @staticmethod
    def _summary_state_signature(summary: dict[str, object]) -> str:
        return WorldModel._signature_from_fragments(WorldModel._summary_state_fragments(summary))

    def _document_state_signature(self, document: dict[str, object], *, family: str, horizon: str) -> str:
        return self._signature_from_fragments(
            self._document_state_fragments(document, family=family, horizon=horizon)
        )

    def _document_state_fragments(self, document: dict[str, object], *, family: str, horizon: str) -> list[str]:
        task_contract = document.get("task_contract", {})
        task_contract = dict(task_contract) if isinstance(task_contract, dict) else {}
        task_metadata = document.get("task_metadata", {})
        task_metadata = dict(task_metadata) if isinstance(task_metadata, dict) else {}
        metadata = task_contract.get("metadata", {})
        metadata = dict(metadata) if isinstance(metadata, dict) else {}
        verifier = task_metadata.get("semantic_verifier", metadata.get("semantic_verifier", {}))
        verifier = dict(verifier) if isinstance(verifier, dict) else {}
        fragments = [f"family:{family}", f"horizon:{horizon}"]
        expected_paths: list[str] = []
        for path in [*task_contract.get("expected_files", []), *dict(task_contract.get("expected_file_contents", {})).keys()]:
            normalized = str(path).strip()
            if normalized and normalized not in expected_paths:
                expected_paths.append(normalized)
        for path in expected_paths[:6]:
            fragments.append(f"expected_artifacts:{path}")
        for output in task_contract.get("expected_output_substrings", [])[:4]:
            normalized = str(output).strip()
            if normalized:
                fragments.append(f"target_output:{normalized[:120]}")
        for output in task_contract.get("forbidden_output_substrings", [])[:4]:
            normalized = str(output).strip()
            if normalized:
                fragments.append(f"forbidden_output:{normalized[:120]}")
        for path in task_contract.get("forbidden_files", [])[:4]:
            normalized = str(path).strip()
            if normalized:
                fragments.append(f"forbidden_artifacts:{normalized}")
        for key in ("expected_changed_paths", "generated_paths", "preserved_paths"):
            for path in verifier.get(key, [])[:6]:
                normalized = str(path).strip()
                if normalized:
                    summary_key = {
                        "expected_changed_paths": "workflow_expected_changed_paths",
                        "generated_paths": "workflow_generated_paths",
                        "preserved_paths": "workflow_preserved_paths",
                    }[key]
                    fragments.append(f"{summary_key}:{normalized}")
        for rule in verifier.get("report_rules", [])[:4]:
            if not isinstance(rule, dict):
                continue
            normalized = str(rule.get("path", "")).strip()
            if normalized:
                fragments.append(f"workflow_report_paths:{normalized}")
        for rule in verifier.get("test_commands", [])[:4]:
            if isinstance(rule, dict):
                label = str(rule.get("label", "")).strip() or " ".join(
                    str(value).strip() for value in rule.get("argv", [])[:2] if str(value).strip()
                )
            else:
                label = str(rule).strip()
            if label:
                fragments.append(f"test:{label}")
        for label in self._verifier_behavior_labels(verifier)[:4]:
            fragments.append(f"behavior:{label}")
        for label in self._verifier_differential_labels(verifier)[:4]:
            fragments.append(f"differential:{label}")
        for label in self._verifier_repo_invariant_labels(verifier)[:4]:
            fragments.append(f"invariant:{label}")
        for branch in verifier.get("required_merged_branches", [])[:4]:
            normalized = str(branch).strip()
            if normalized:
                fragments.append(f"merge:{normalized}")
        summary = document.get("summary", {})
        summary = dict(summary) if isinstance(summary, dict) else {}
        for patch in self._document_transition_patches(document, summary=summary)[:4]:
            path = str(patch.get("path", "")).strip()
            if path:
                fragments.append(f"patch_path:{path}")
            patch_summary = str(patch.get("patch_summary", "")).strip()
            if patch_summary:
                fragments.append(f"patch_summary:{patch_summary[:120]}")
            before_excerpt = str(patch.get("before_excerpt", "")).strip()
            if before_excerpt:
                fragments.append(f"before:{before_excerpt[:120]}")
            after_excerpt = str(patch.get("after_excerpt", "")).strip()
            if after_excerpt:
                fragments.append(f"after:{after_excerpt[:120]}")
        return fragments

    @staticmethod
    def _signature_from_fragments(fragments: list[str]) -> str:
        normalized: list[str] = []
        for fragment in fragments:
            value = str(fragment).strip()
            if value and value not in normalized:
                normalized.append(value)
        if not normalized:
            return ""
        return hashlib.sha256("\n".join(normalized).encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _summary_state_fragments(summary: dict[str, object]) -> list[str]:
        fragments: list[str] = []
        family = str(summary.get("benchmark_family", "bounded")).strip() or "bounded"
        horizon = str(summary.get("horizon", "bounded")).strip() or "bounded"
        fragments.extend([f"family:{family}", f"horizon:{horizon}"])
        for key in (
            "expected_artifacts",
            "forbidden_artifacts",
            "preserved_artifacts",
            "workflow_expected_changed_paths",
            "workflow_generated_paths",
            "workflow_report_paths",
            "workflow_preserved_paths",
        ):
            for path in summary.get(key, [])[:6]:
                normalized = str(path).strip()
                if normalized:
                    fragments.append(f"{key}:{normalized}")
        for output in summary.get("target_outputs", [])[:4]:
            normalized = str(output).strip()
            if normalized:
                fragments.append(f"target_output:{normalized[:120]}")
        for label in summary.get("workflow_required_tests", [])[:4]:
            normalized = str(label).strip()
            if normalized:
                fragments.append(f"test:{normalized}")
        for label in summary.get("workflow_behavior_checks", [])[:4]:
            normalized = str(label).strip()
            if normalized:
                fragments.append(f"behavior:{normalized}")
        for label in summary.get("workflow_differential_checks", [])[:4]:
            normalized = str(label).strip()
            if normalized:
                fragments.append(f"differential:{normalized}")
        for label in summary.get("workflow_repo_invariants", [])[:4]:
            normalized = str(label).strip()
            if normalized:
                fragments.append(f"invariant:{normalized}")
        for branch in summary.get("workflow_required_merges", [])[:4]:
            normalized = str(branch).strip()
            if normalized:
                fragments.append(f"merge:{normalized}")
        previews = summary.get("workspace_file_previews", {})
        previews = previews if isinstance(previews, dict) else {}
        for path in WorldModel._summary_signature_preview_paths(summary)[:4]:
            preview = previews.get(path, {})
            for excerpt in WorldModel._preview_signature_excerpts(preview)[:2]:
                fragments.append(f"preview:{path}:{excerpt}")
        return fragments

    @staticmethod
    def _summary_signature_preview_paths(summary: dict[str, object]) -> list[str]:
        ordered: list[str] = []
        for key in (
            "expected_artifacts",
            "workflow_expected_changed_paths",
            "workflow_report_paths",
            "workflow_generated_paths",
            "workflow_preserved_paths",
        ):
            for path in summary.get(key, [])[:6]:
                normalized = str(path).strip()
                if normalized and normalized not in ordered:
                    ordered.append(normalized)
        return ordered

    @staticmethod
    def _preview_signature_excerpts(preview: object) -> list[str]:
        excerpts: list[str] = []
        if not isinstance(preview, dict):
            return excerpts
        content = str(preview.get("content", "")).strip()
        if content:
            excerpts.append(content[:120])
        for window in preview.get("edit_windows", [])[:2]:
            if not isinstance(window, dict):
                continue
            window_content = str(window.get("content", "")).strip()
            if window_content:
                excerpts.append(window_content[:120])
        return excerpts

    def _document_transition_patches(
        self,
        document: dict[str, object],
        *,
        summary: dict[str, object],
    ) -> list[dict[str, object]]:
        patches = summary.get("edit_patches", [])
        if isinstance(patches, list) and patches:
            return [dict(item) for item in patches if isinstance(item, dict)]
        collected: list[dict[str, object]] = []
        for step in document.get("steps", []) if isinstance(document.get("steps", []), list) else []:
            if not isinstance(step, dict):
                continue
            transition = step.get("state_transition", {})
            transition = dict(transition) if isinstance(transition, dict) else {}
            for patch in transition.get("edit_patches", []):
                if isinstance(patch, dict):
                    collected.append(dict(patch))
        for fragment in document.get("fragments", []) if isinstance(document.get("fragments", []), list) else []:
            if not isinstance(fragment, dict):
                continue
            if str(fragment.get("kind", "")).strip() != "edit_patch":
                continue
            collected.append(dict(fragment))
        return collected

    @staticmethod
    def _verifier_behavior_labels(verifier: dict[str, object]) -> list[str]:
        labels: list[str] = []
        for rule in verifier.get("behavior_checks", [])[:4]:
            if not isinstance(rule, dict):
                continue
            label = str(rule.get("label", "")).strip() or " ".join(
                str(value).strip() for value in rule.get("argv", [])[:2] if str(value).strip()
            )
            if label:
                labels.append(label)
        return labels

    @staticmethod
    def _verifier_differential_labels(verifier: dict[str, object]) -> list[str]:
        labels: list[str] = []
        for rule in verifier.get("differential_checks", [])[:4]:
            if not isinstance(rule, dict):
                continue
            label = str(rule.get("label", "")).strip()
            if not label:
                candidate = " ".join(
                    str(value).strip() for value in rule.get("candidate_argv", [])[:2] if str(value).strip()
                )
                baseline = " ".join(
                    str(value).strip() for value in rule.get("baseline_argv", [])[:2] if str(value).strip()
                )
                label = f"{candidate} vs {baseline}".strip(" vs ")
            if label:
                labels.append(label)
            for key, prefix in (
                ("candidate_file_expectations", "candidate_file"),
                ("baseline_file_expectations", "baseline_file"),
            ):
                for expectation in rule.get(key, [])[:2]:
                    if not isinstance(expectation, dict):
                        continue
                    path = str(expectation.get("path", "")).strip()
                    if path:
                        labels.append(f"{prefix}:{path}")
        return labels[:6]

    @staticmethod
    def _verifier_repo_invariant_labels(verifier: dict[str, object]) -> list[str]:
        labels: list[str] = []
        for rule in verifier.get("repo_invariants", [])[:4]:
            if not isinstance(rule, dict):
                continue
            kind = str(rule.get("kind", "")).strip()
            path = str(rule.get("path", "")).strip()
            paths = ",".join(str(value).strip() for value in rule.get("paths", [])[:2] if str(value).strip())
            label = ":".join(part for part in (kind, path or paths) if part)
            if label:
                labels.append(label)
        return labels

    @staticmethod
    def _is_test_like_command(command: str) -> bool:
        normalized = str(command).strip()
        return bool(
            normalized.startswith("pytest")
            or normalized.startswith("python -m pytest")
            or normalized.startswith("test ")
            or "/test" in normalized
            or " verify" in normalized
        )

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
        return world_model_analysis_support.prioritized_long_horizon_hotspots(
            self,
            task=task,
            summary=summary,
            latest_transition=latest_transition,
            latent_state_summary=latent_state_summary,
            active_subgoal=active_subgoal,
            max_items=max_items,
        )

    _workspace_state_summary = world_model_preview_support.workspace_state_summary

    _prioritized_preview_paths = staticmethod(world_model_preview_support.prioritized_preview_paths)

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
            "behavior_checks": WorldModel._verifier_behavior_labels(verifier),
            "differential_checks": WorldModel._verifier_differential_labels(verifier),
            "repo_invariants": WorldModel._verifier_repo_invariant_labels(verifier),
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
        return world_model_preview_support.workspace_file_previews(
            WorldModel,
            workspace,
            candidate_paths,
            expected_file_contents=expected_file_contents,
        )

    @staticmethod
    def _text_file_preview(
        path: Path,
        *,
        max_bytes: int = _MAX_WORKSPACE_PREVIEW_BYTES,
        max_chars: int = _MAX_WORKSPACE_PREVIEW_CHARS,
    ) -> dict[str, object] | None:
        return world_model_preview_support.text_file_preview(
            WorldModel,
            path,
            max_bytes=max_bytes,
            max_chars=max_chars,
        )

    @staticmethod
    def _targeted_text_file_previews(
        path: Path,
        *,
        expected_content: str,
        max_bytes: int = _MAX_WORKSPACE_PREVIEW_BYTES,
        max_chars: int = _MAX_WORKSPACE_PREVIEW_CHARS,
    ) -> list[dict[str, object]]:
        return world_model_preview_support.targeted_text_file_previews(
            WorldModel,
            path,
            expected_content=expected_content,
            max_bytes=max_bytes,
            max_chars=max_chars,
        )

    _dedupe_hidden_gap_current_proof_regions = staticmethod(
        world_model_preview_support.dedupe_hidden_gap_current_proof_regions
    )

    @staticmethod
    def _exact_targeted_preview_proof_windows(
        *,
        windows: list[tuple[int, int, int, int]],
        retained_window_count: int,
    ) -> list[dict[str, object]]:
        return world_model_preview_support.exact_targeted_preview_proof_windows(
            windows=windows,
            retained_window_count=retained_window_count,
        )

    @staticmethod
    def _bridged_targeted_preview_windows(
        *,
        current_lines: list[str],
        expected_lines: list[str],
        windows: list[tuple[int, int, int, int]],
        max_bytes: int,
        max_chars: int,
    ) -> list[dict[str, object]]:
        return world_model_preview_support.bridged_targeted_preview_windows(
            current_lines=current_lines,
            expected_lines=expected_lines,
            windows=windows,
            max_bytes=max_bytes,
            max_chars=max_chars,
        )

    _bridged_targeted_preview_window_runs = staticmethod(
        world_model_analysis_support.bridged_targeted_preview_window_runs
    )

    _hidden_gap_current_proof_regions = staticmethod(
        world_model_analysis_support.hidden_gap_current_proof_regions
    )

    @staticmethod
    def _sparse_hidden_gap_current_proof_regions(
        *,
        windows: list[tuple[int, int, int, int]],
        bridged_windows: list[dict[str, object]],
        start_region_index: int = 0,
    ) -> list[dict[str, object]]:
        return world_model_analysis_support.sparse_hidden_gap_current_proof_regions(
            WorldModel,
            windows=windows,
            bridged_windows=bridged_windows,
            start_region_index=start_region_index,
        )

    _bridge_window_has_current_proof = staticmethod(
        world_model_analysis_support.bridge_window_has_current_proof
    )

    @staticmethod
    def _sparse_hidden_gap_current_proof_region(
        *,
        windows: list[tuple[int, int, int, int]],
        bridge_by_pair: dict[tuple[int, int], dict[str, object]],
        window_start_index: int,
        window_end_index: int,
        region_index: int,
    ) -> dict[str, object] | None:
        return world_model_analysis_support.sparse_hidden_gap_current_proof_region(
            WorldModel,
            windows=windows,
            bridge_by_pair=bridge_by_pair,
            window_start_index=window_start_index,
            window_end_index=window_end_index,
            region_index=region_index,
        )

    _windows_have_zero_gap = staticmethod(world_model_analysis_support.windows_have_zero_gap)

    @staticmethod
    def _targeted_preview_windows(
        *,
        current_lines: list[str],
        expected_lines: list[str],
        max_bytes: int,
        max_chars: int,
    ) -> list[tuple[int, int, int, int]]:
        return world_model_preview_support.targeted_preview_windows(
            current_lines=current_lines,
            expected_lines=expected_lines,
            max_bytes=max_bytes,
            max_chars=max_chars,
        )

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
        return world_model_preview_support.expanded_preview_window(
            current_lines=current_lines,
            expected_lines=expected_lines,
            current_start=current_start,
            current_end=current_end,
            expected_start=expected_start,
            expected_end=expected_end,
            max_bytes=max_bytes,
            max_chars=max_chars,
        )

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
        return world_model_preview_support.merge_preview_windows(
            current_lines=current_lines,
            expected_lines=expected_lines,
            left=left,
            right=right,
            max_bytes=max_bytes,
            max_chars=max_chars,
        )

    _read_text_file = staticmethod(world_model_preview_support.read_text_file)

    _fits_preview_budget = staticmethod(world_model_preview_support.fits_preview_budget)

    _fits_dual_preview_budget = staticmethod(world_model_preview_support.fits_dual_preview_budget)

    _bounded_text_prefix = staticmethod(world_model_preview_support.bounded_text_prefix)

    _sha1_text = staticmethod(world_model_preview_support.sha1_text)
