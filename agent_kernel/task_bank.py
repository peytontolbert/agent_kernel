from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import glob
import json
from pathlib import Path
import re
import shlex

from .config import KernelConfig
from .episode_store import iter_episode_documents
from .kernel_catalog import kernel_catalog_string_list, kernel_catalog_string_set
from .schemas import TaskSpec
from .task_budget import uplifted_task_max_steps
from .verifier import synthesize_stricter_task


_BUILTIN_TASK_MANIFEST_PATH = (
    Path(__file__).resolve().parent.parent / "datasets" / "task_bank" / "default_tasks.json"
)
_TASK_BANK_SYNTHESIS_RULES_PATH = (
    Path(__file__).resolve().parent.parent / "datasets" / "task_bank" / "synthesis_rules.json"
)


def _task_acceptance_contract_defined(task: TaskSpec) -> bool:
    metadata = dict(task.metadata)
    verifier = dict(metadata.get("semantic_verifier", {})) if isinstance(metadata.get("semantic_verifier", {}), dict) else {}
    return bool(
        task.success_command
        or task.expected_files
        or task.expected_output_substrings
        or task.forbidden_files
        or task.forbidden_output_substrings
        or task.expected_file_contents
        or verifier
    )


def _task_light_supervision_candidate(task: TaskSpec) -> bool:
    metadata = dict(task.metadata)
    if not _task_acceptance_contract_defined(task):
        return False
    memory_source = str(metadata.get("memory_source", "none")).strip().lower() or "none"
    if memory_source not in {"", "none"}:
        return False
    if bool(metadata.get("requires_retrieval", False)) and str(metadata.get("source_task", "")).strip():
        return False
    difficulty = str(metadata.get("difficulty", "")).strip().lower()
    if difficulty == "retrieval":
        return False
    task_origin = str(metadata.get("task_origin", "")).strip().lower()
    if task_origin in {
        "episode_replay",
        "skill_replay",
        "skill_transfer",
        "operator_replay",
        "tool_replay",
        "verifier_replay",
        "discovered_task",
        "transition_pressure",
        "benchmark_candidate",
        "verifier_candidate",
    }:
        return False
    return True


def _light_supervision_contract_kind(task: TaskSpec) -> str:
    metadata = dict(task.metadata)
    verifier = dict(metadata.get("semantic_verifier", {})) if isinstance(metadata.get("semantic_verifier", {}), dict) else {}
    if verifier:
        return "semantic_verifier"
    if (
        task.expected_files
        or task.expected_output_substrings
        or task.forbidden_files
        or task.forbidden_output_substrings
        or task.expected_file_contents
    ):
        return "workspace_acceptance"
    if task.success_command:
        return "success_command"
    return ""


def _annotate_light_supervision_contract(task: TaskSpec) -> TaskSpec:
    metadata = dict(task.metadata)
    candidate = _task_light_supervision_candidate(task)
    metadata["light_supervision_candidate"] = candidate
    metadata["decision_yield_family"] = str(metadata.get("benchmark_family", "bounded")).strip() or "bounded"
    metadata["decision_yield_contract_candidate"] = candidate
    contract_kind = _light_supervision_contract_kind(task)
    if contract_kind:
        metadata["light_supervision_contract_kind"] = contract_kind
    else:
        metadata.pop("light_supervision_contract_kind", None)
    task.metadata = metadata
    return task


def annotate_light_supervision_contract(task: TaskSpec) -> TaskSpec:
    return _annotate_light_supervision_contract(task)


@lru_cache(maxsize=1)
def _task_bank_synthesis_rules() -> dict[str, object]:
    if not _TASK_BANK_SYNTHESIS_RULES_PATH.exists() or not _TASK_BANK_SYNTHESIS_RULES_PATH.is_file():
        raise RuntimeError(f"task-bank synthesis rules missing: {_TASK_BANK_SYNTHESIS_RULES_PATH}")
    try:
        parsed = json.loads(_TASK_BANK_SYNTHESIS_RULES_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid task-bank synthesis rules: {_TASK_BANK_SYNTHESIS_RULES_PATH}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"task-bank synthesis rules must be an object: {_TASK_BANK_SYNTHESIS_RULES_PATH}")
    return parsed


def _task_bank_memory_task_rule(name: str) -> dict[str, object]:
    rules = _task_bank_synthesis_rules().get("memory_task_rules", {})
    if not isinstance(rules, dict):
        raise RuntimeError(f"task-bank memory_task_rules must be an object: {_TASK_BANK_SYNTHESIS_RULES_PATH}")
    rule = rules.get(name, {})
    if not isinstance(rule, dict):
        raise RuntimeError(f"task-bank memory task rule must be an object: {name}")
    return rule


def _memory_task_rule_text(rule_name: str, field: str, *, fallback: str = "") -> str:
    return str(_task_bank_memory_task_rule(rule_name).get(field, fallback)).strip() or fallback


def _memory_task_prompt(rule_name: str, *, prompt: str, **kwargs: object) -> str:
    template = _memory_task_rule_text(rule_name, "prompt_template", fallback="{prompt}")
    values = {"prompt": prompt}
    values.update({key: str(value) for key, value in kwargs.items()})
    try:
        rendered = template.format(**values)
    except KeyError as exc:
        raise RuntimeError(f"invalid task-bank memory prompt template: {rule_name}") from exc
    return str(rendered).strip() or prompt


def _memory_task_metadata(
    rule_name: str,
    *,
    source_task_id: str,
    origin_benchmark_family: str,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    metadata = _task_bank_memory_task_rule(rule_name).get("metadata", {})
    if not isinstance(metadata, dict):
        raise RuntimeError(f"task-bank memory task metadata must be an object: {rule_name}")
    payload = dict(metadata)
    payload.setdefault("memory_source_task", source_task_id)
    payload.setdefault("origin_benchmark_family", origin_benchmark_family)
    payload.setdefault("source_task", source_task_id)
    if extra:
        payload.update(extra)
    return payload


class TaskBank:
    def __init__(
        self,
        config: KernelConfig | None = None,
        external_task_manifests: tuple[str, ...] | None = None,
    ) -> None:
        self._tasks: dict[str, TaskSpec] = {}
        self._merge_bundled_tasks(_BUILTIN_TASK_MANIFEST_PATH)
        self._merge_external_tasks(
            external_task_manifests
            if external_task_manifests is not None
            else tuple(
                str(path)
                for path in getattr(config, "external_task_manifests_paths", ())
                if str(path).strip()
            )
        )

    def _merge_bundled_tasks(self, manifest_path: Path) -> None:
        if not manifest_path.exists() or not manifest_path.is_file():
            raise RuntimeError(f"bundled task manifest missing: {manifest_path}")
        loaded = 0
        for payload in self._iter_manifest_task_payloads(manifest_path):
            try:
                task = TaskSpec.from_dict(payload)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"invalid bundled task payload in {manifest_path}") from exc
            task.max_steps = uplifted_task_max_steps(
                task.max_steps,
                metadata=task.metadata,
                suggested_commands=task.suggested_commands,
            )
            task = _annotate_light_supervision_contract(task)
            if task.task_id in self._tasks:
                raise RuntimeError(f"duplicate bundled task id: {task.task_id}")
            self._tasks[task.task_id] = task
            loaded += 1
        if loaded == 0:
            raise RuntimeError(f"bundled task manifest is empty: {manifest_path}")

    @staticmethod
    def _task(*, capability: str, difficulty: str, **kwargs) -> TaskSpec:
        metadata = dict(kwargs.pop("metadata", {}))
        metadata.setdefault("capability", capability)
        metadata.setdefault("difficulty", difficulty)
        metadata.setdefault("benchmark_family", "bounded")
        task = TaskSpec(metadata=metadata, **kwargs)
        task.max_steps = uplifted_task_max_steps(
            task.max_steps,
            metadata=task.metadata,
            suggested_commands=task.suggested_commands,
        )
        return _annotate_light_supervision_contract(task)

    def get(self, task_id: str) -> TaskSpec:
        try:
            return deepcopy(self._tasks[task_id])
        except KeyError as exc:
            raise KeyError(f"Unknown task_id: {task_id}") from exc

    def list(self) -> list[TaskSpec]:
        return [deepcopy(self._tasks[task_id]) for task_id in sorted(self._tasks)]

    def _merge_external_tasks(self, manifest_paths: tuple[str, ...]) -> None:
        for manifest_path in self._external_manifest_files(manifest_paths):
            if not manifest_path.exists() or not manifest_path.is_file():
                continue
            for payload in self._iter_manifest_task_payloads(manifest_path):
                try:
                    task = TaskSpec.from_dict(payload)
                except (TypeError, ValueError):
                    continue
                if task.task_id in self._tasks:
                    continue
                metadata = dict(task.metadata)
                metadata.setdefault("benchmark_family", "external_manifest")
                metadata.setdefault("capability", "external_manifest")
                metadata.setdefault("task_origin", "external_manifest")
                metadata.setdefault("external_manifest_path", str(manifest_path))
                task.metadata = metadata
                task.max_steps = uplifted_task_max_steps(
                    task.max_steps,
                    metadata=task.metadata,
                    suggested_commands=task.suggested_commands,
                )
                task = _annotate_light_supervision_contract(task)
                self._tasks[task.task_id] = task

    @staticmethod
    def _external_manifest_files(manifest_paths: tuple[str, ...]) -> list[Path]:
        resolved: list[Path] = []
        seen: set[str] = set()
        for raw_path in manifest_paths:
            normalized = str(raw_path).strip()
            if not normalized:
                continue
            matches: list[Path]
            if any(char in normalized for char in "*?[]"):
                matches = [Path(value) for value in glob.glob(normalized, recursive=True)]
            else:
                candidate = Path(normalized)
                if candidate.is_dir():
                    matches = sorted(
                        [
                            *candidate.rglob("*.json"),
                            *candidate.rglob("*.jsonl"),
                        ]
                    )
                else:
                    matches = [candidate]
            for match in matches:
                key = str(match.resolve()) if match.exists() else str(match)
                if key in seen:
                    continue
                seen.add(key)
                resolved.append(match)
        return resolved

    @staticmethod
    def _iter_manifest_task_payloads(path: Path) -> list[dict[str, object]]:
        if path.suffix.lower() == ".jsonl":
            payloads: list[dict[str, object]] = []
            try:
                for line in path.read_text(encoding="utf-8").splitlines():
                    stripped = line.strip()
                    if not stripped:
                        continue
                    parsed = json.loads(stripped)
                    if isinstance(parsed, dict):
                        payloads.extend(TaskBank._task_payloads_from_parsed_manifest(parsed))
            except (OSError, json.JSONDecodeError):
                return []
            return payloads
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        if isinstance(parsed, dict):
            return TaskBank._task_payloads_from_parsed_manifest(parsed)
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        return []

    @staticmethod
    def _task_payloads_from_parsed_manifest(parsed: dict[str, object]) -> list[dict[str, object]]:
        tasks = parsed.get("tasks")
        if isinstance(tasks, list):
            return [item for item in tasks if isinstance(item, dict)]
        task = parsed.get("task")
        if isinstance(task, dict):
            return [task]
        if "task_id" in parsed:
            return [parsed]
        return []

    def parallel_worker_tasks(
        self,
        task_id: str,
        *,
        target_worker_count: int | None = None,
    ) -> list[TaskSpec]:
        task = self.get(task_id)
        metadata = dict(task.metadata)
        workflow_guard = metadata.get("workflow_guard", {})
        guard = dict(workflow_guard) if isinstance(workflow_guard, dict) else {}
        verifier = metadata.get("semantic_verifier", {})
        contract = dict(verifier) if isinstance(verifier, dict) else {}
        shared_repo_id = str(guard.get("shared_repo_id", "")).strip()
        required_branches = [
            str(value).strip()
            for value in contract.get("required_merged_branches", [])
            if str(value).strip()
        ]
        candidate_paths = _parallel_worker_candidate_paths(contract)
        expanded_required_branches = _expanded_required_worker_branches(
            required_branches,
            changed_paths=candidate_paths,
            target_worker_count=target_worker_count,
        )
        if not shared_repo_id or not expanded_required_branches:
            return []
        try:
            integrator_order = int(metadata.get("shared_repo_order", 0))
        except (TypeError, ValueError):
            integrator_order = 0
        branch_order = {branch: index for index, branch in enumerate(expanded_required_branches)}
        matches: list[tuple[int, int, str, TaskSpec]] = []
        for candidate_id, candidate in self._tasks.items():
            if candidate_id == task_id:
                continue
            candidate_metadata = dict(candidate.metadata)
            candidate_guard = candidate_metadata.get("workflow_guard", {})
            candidate_workflow_guard = dict(candidate_guard) if isinstance(candidate_guard, dict) else {}
            candidate_shared_repo_id = str(candidate_workflow_guard.get("shared_repo_id", "")).strip()
            worker_branch = str(candidate_workflow_guard.get("worker_branch", "")).strip()
            if candidate_shared_repo_id != shared_repo_id or worker_branch not in branch_order:
                continue
            try:
                candidate_order = int(candidate_metadata.get("shared_repo_order", 0))
            except (TypeError, ValueError):
                candidate_order = 0
            if candidate_order >= integrator_order:
                continue
            matches.append((branch_order[worker_branch], candidate_order, candidate_id, deepcopy(candidate)))
        matches.sort(key=lambda item: (item[0], item[1], item[2]))
        if matches and len(matches) >= len(expanded_required_branches):
            return [task for _, _, _, task in matches]
        return self._synthesized_parallel_worker_tasks(task, required_branches=expanded_required_branches)

    def _synthesized_parallel_worker_tasks(self, task: TaskSpec, *, required_branches: list[str]) -> list[TaskSpec]:
        metadata = dict(task.metadata)
        workflow_guard = metadata.get("workflow_guard", {})
        guard = dict(workflow_guard) if isinstance(workflow_guard, dict) else {}
        worker_specs = metadata.get("parallel_workers", [])
        if not isinstance(worker_specs, list) or not worker_specs:
            worker_specs = self._derive_worker_specs_from_integrator(task, required_branches=required_branches)
        if not worker_specs:
            return []
        benchmark_family = str(metadata.get("benchmark_family", "repo_sandbox")).strip() or "repo_sandbox"
        shared_repo_id = str(guard.get("shared_repo_id", "")).strip()
        target_branch = str(guard.get("target_branch", "main")).strip() or "main"
        bootstrap_commands = [
            str(command).strip()
            for command in metadata.get("shared_repo_bootstrap_commands", [])
            if str(command).strip()
        ]
        bootstrap_fixture_dir = str(metadata.get("shared_repo_bootstrap_fixture_dir", "")).strip()
        bootstrap_executable_paths = [
            str(path).strip()
            for path in metadata.get("shared_repo_bootstrap_executable_paths", [])
            if str(path).strip()
        ]
        bootstrap_managed_paths = [
            str(path).strip()
            for path in metadata.get("shared_repo_bootstrap_managed_paths", [])
            if str(path).strip()
        ]
        required_branch_set = set(required_branches)
        synthesized: list[tuple[int, TaskSpec]] = []
        for index, worker_spec in enumerate(worker_specs):
            if not isinstance(worker_spec, dict):
                continue
            worker_branch = str(worker_spec.get("worker_branch", "")).strip()
            if not worker_branch or worker_branch not in required_branch_set:
                continue
            expected_changed_paths = [
                str(path).strip()
                for path in worker_spec.get("expected_changed_paths", [])
                if str(path).strip()
            ]
            claimed_paths = [
                str(path).strip()
                for path in worker_spec.get("claimed_paths", expected_changed_paths)
                if str(path).strip()
            ]
            preserved_paths = [
                str(path).strip()
                for path in worker_spec.get("preserved_paths", [])
                if str(path).strip()
            ]
            test_commands = [
                dict(command)
                for command in worker_spec.get("test_commands", [])
                if isinstance(command, dict)
            ]
            report_rules = [
                dict(rule)
                for rule in worker_spec.get("report_rules", [])
                if isinstance(rule, dict)
            ]
            expected_files = [
                *expected_changed_paths,
                *preserved_paths,
                *[
                    str(rule.get("path", "")).strip()
                    for rule in report_rules
                    if str(rule.get("path", "")).strip()
                ],
            ]
            expected_file_contents = {
                str(path).strip(): str(content)
                for path, content in worker_spec.get("expected_file_contents", {}).items()
                if str(path).strip()
            } if isinstance(worker_spec.get("expected_file_contents", {}), dict) else {}
            edit_plan = [
                dict(step)
                for step in worker_spec.get("edit_plan", [])
                if isinstance(step, dict)
            ]
            edit_candidates = [
                dict(step)
                for step in worker_spec.get("edit_candidates", [])
                if isinstance(step, dict)
            ]
            prompt = str(worker_spec.get("prompt", "")).strip() or (
                f"On branch {worker_branch}, update {', '.join(expected_changed_paths) or 'the required paths'} "
                f"for integration into {target_branch}."
            )
            synthesized_task = self._task(
                task_id=f"{task.task_id}__worker__{_safe_worker_name(worker_branch)}",
                prompt=prompt,
                workspace_subdir=f"{task.workspace_subdir}__worker__{_safe_worker_name(worker_branch)}",
                setup_commands=[],
                suggested_commands=[
                    str(command).strip()
                    for command in worker_spec.get("suggested_commands", [])
                    if str(command).strip()
                ],
                success_command=str(worker_spec.get("success_command", "git branch --show-current")).strip()
                or "git branch --show-current",
                expected_files=expected_files,
                expected_file_contents=expected_file_contents,
                capability=str(metadata.get("capability", "repo_environment")),
                difficulty="git_worker_branch_synthesized",
                metadata={
                    "benchmark_family": benchmark_family,
                    "requires_git": True,
                    "shared_repo_order": 0,
                    "synthetic_worker": True,
                    "source_integrator_task_id": task.task_id,
                    "shared_repo_bootstrap_commands": bootstrap_commands,
                    "shared_repo_bootstrap_fixture_dir": bootstrap_fixture_dir,
                    "shared_repo_bootstrap_executable_paths": bootstrap_executable_paths,
                    "shared_repo_bootstrap_managed_paths": bootstrap_managed_paths,
                    "synthetic_edit_plan": edit_plan,
                    "synthetic_edit_candidates": edit_candidates,
                    "workflow_guard": {
                        "requires_git": True,
                        "shared_repo_id": shared_repo_id,
                        "target_branch": target_branch,
                        "worker_branch": worker_branch,
                        "claimed_paths": claimed_paths,
                    },
                    "semantic_verifier": {
                        "kind": "git_repo_review",
                        "expected_branch": worker_branch,
                        "diff_base_ref": f"origin/{target_branch}",
                        "expected_changed_paths": expected_changed_paths,
                        "preserved_paths": preserved_paths,
                        "clean_worktree": True,
                        "test_commands": test_commands,
                        "report_rules": report_rules,
                    },
                },
            )
            synthesized.append((required_branches.index(worker_branch), synthesized_task))
        synthesized.sort(key=lambda item: item[0])
        return [worker_task for _, worker_task in synthesized]

    def _derive_worker_specs_from_integrator(self, task: TaskSpec, *, required_branches: list[str]) -> list[dict[str, object]]:
        metadata = dict(task.metadata)
        workflow_guard = metadata.get("workflow_guard", {})
        guard = dict(workflow_guard) if isinstance(workflow_guard, dict) else {}
        target_branch = str(guard.get("target_branch", "main")).strip() or "main"
        verifier = metadata.get("semantic_verifier", {})
        contract = dict(verifier) if isinstance(verifier, dict) else {}
        report_paths = {
            str(rule.get("path", "")).strip()
            for rule in contract.get("report_rules", [])
            if isinstance(rule, dict) and str(rule.get("path", "")).strip()
        }
        generated_paths = {
            str(path).strip()
            for path in contract.get("generated_paths", [])
            if str(path).strip()
        }
        resolved_conflict_paths = {
            str(path).strip()
            for path in contract.get("resolved_conflict_paths", [])
            if str(path).strip()
        }
        changed_paths = [
            str(path).strip()
            for path in contract.get("expected_changed_paths", [])
            if str(path).strip()
        ]
        worker_candidate_paths = [
            path
            for path in changed_paths
            if path not in report_paths and path not in generated_paths and path not in resolved_conflict_paths
        ]
        if not worker_candidate_paths or not required_branches:
            return []
        branch_assignments = _assign_paths_to_branches(required_branches, worker_candidate_paths)
        preserved_paths = [
            str(path).strip()
            for path in contract.get("preserved_paths", [])
            if str(path).strip()
        ]
        bootstrap_file_contents = _bootstrap_file_contents(task)
        expected_file_contents = dict(task.expected_file_contents)
        worker_specs: list[dict[str, object]] = []
        for branch in required_branches:
            assigned_paths = branch_assignments.get(branch, [])
            if not assigned_paths:
                continue
            assigned_tests = _select_worker_test_commands(branch, assigned_paths, contract)
            report_rules = _derive_worker_report_rules(branch, assigned_paths, assigned_tests)
            report_paths = [
                str(rule.get("path", "")).strip()
                for rule in report_rules
                if str(rule.get("path", "")).strip()
            ]
            edit_plan = _derive_worker_edit_plan(
                branch=branch,
                assigned_paths=assigned_paths,
                assigned_tests=assigned_tests,
                expected_file_contents=expected_file_contents,
                bootstrap_file_contents=bootstrap_file_contents,
            )
            edit_candidates = _derive_worker_edit_candidates(
                branch=branch,
                assigned_paths=assigned_paths,
                assigned_tests=assigned_tests,
                expected_file_contents=expected_file_contents,
                bootstrap_file_contents=bootstrap_file_contents,
            )
            worker_expected_contents = {
                str(step.get("path", "")).strip(): str(step.get("target_content", ""))
                for step in edit_plan
                if str(step.get("path", "")).strip() and step.get("target_content") is not None
            }
            worker_specs.append(
                {
                    "worker_branch": branch,
                    "prompt": _worker_prompt(branch, assigned_paths),
                    "expected_changed_paths": [*assigned_paths, *report_paths],
                    "claimed_paths": [*assigned_paths, *report_paths],
                    "preserved_paths": [path for path in preserved_paths if path not in assigned_paths],
                    "test_commands": assigned_tests,
                    "report_rules": report_rules,
                    "edit_plan": edit_plan,
                    "edit_candidates": edit_candidates,
                    "expected_file_contents": worker_expected_contents,
                    "suggested_commands": _synthesized_worker_commands(
                        branch=branch,
                        target_branch=target_branch,
                        changed_paths=assigned_paths,
                        edit_plan=edit_plan,
                        expected_file_contents=worker_expected_contents,
                        test_commands=assigned_tests,
                        report_rules=report_rules,
                    ),
                }
            )
        return worker_specs


_SYNTHETIC_LINEAGE_TASK_SUFFIXES = tuple(
    kernel_catalog_string_list("task_bank", "synthetic_lineage_task_suffixes")
)

_SYNTHETIC_LINEAGE_BENCHMARK_FAMILIES = kernel_catalog_string_set(
    "task_bank",
    "synthetic_lineage_benchmark_families",
)
_SYNTHETIC_LINEAGE_MEMORY_SOURCES = kernel_catalog_string_set(
    "task_bank",
    "synthetic_lineage_memory_sources",
)


def _synthetic_lineage_seed_skipped(document: dict[str, object]) -> bool:
    task_id = str(document.get("task_id", "")).strip()
    if any(task_id.endswith(suffix) for suffix in _SYNTHETIC_LINEAGE_TASK_SUFFIXES):
        return True
    task_metadata = document.get("task_metadata", {})
    if not isinstance(task_metadata, dict):
        task_metadata = {}
    benchmark_family = str(task_metadata.get("benchmark_family", "")).strip()
    if benchmark_family in _SYNTHETIC_LINEAGE_BENCHMARK_FAMILIES:
        return True
    memory_source = str(task_metadata.get("memory_source", "")).strip()
    return memory_source in _SYNTHETIC_LINEAGE_MEMORY_SOURCES


def load_episode_replay_tasks(episodes_root: Path, *, limit: int | None = None) -> list[TaskSpec]:
    bank = TaskBank()
    tasks: list[TaskSpec] = []
    for data in iter_episode_documents(episodes_root):
        if not data.get("success"):
            continue
        if _synthetic_lineage_seed_skipped(data):
            continue
        task_id = str(data.get("task_id", "")).strip()
        if not task_id:
            continue
        contract = _task_contract_from_memory(data, task_id=task_id, bank=bank)
        if contract is None:
            continue
        executed_commands = list(data.get("summary", {}).get("executed_commands", []))
        metadata = dict(contract.metadata)
        metadata.update(
            _memory_task_metadata(
                "episode_replay",
                source_task_id=task_id,
                origin_benchmark_family=str(contract.metadata.get("benchmark_family", "bounded")),
                extra={
                    "episode_phase": str(data.get("episode_storage", {}).get("phase", "")).strip(),
                    "episode_relative_path": str(data.get("episode_storage", {}).get("relative_path", "")).strip(),
                },
            )
        )
        replay_task = TaskSpec(
            task_id=f"{task_id}{_memory_task_rule_text('episode_replay', 'task_id_suffix', fallback='_episode_replay')}",
            prompt=_memory_task_prompt("episode_replay", prompt=contract.prompt),
            workspace_subdir=(
                f"{task_id}{_memory_task_rule_text('episode_replay', 'workspace_suffix', fallback='_episode_replay')}"
            ),
            setup_commands=list(contract.setup_commands),
            success_command=contract.success_command,
            suggested_commands=executed_commands or list(contract.suggested_commands),
            expected_files=list(contract.expected_files),
            expected_output_substrings=list(contract.expected_output_substrings),
            forbidden_files=list(contract.forbidden_files),
            forbidden_output_substrings=list(contract.forbidden_output_substrings),
            expected_file_contents=dict(contract.expected_file_contents),
            max_steps=max(contract.max_steps, max(1, len(executed_commands) + 1)),
            metadata=metadata,
        )
        tasks.append(_annotate_light_supervision_contract(replay_task))
        if limit is not None and len(tasks) >= limit:
            break
    return tasks


def load_discovered_tasks(episodes_root: Path, *, limit: int | None = None) -> list[TaskSpec]:
    bank = TaskBank()
    tasks: list[TaskSpec] = []
    for data in iter_episode_documents(episodes_root):
        if _synthetic_lineage_seed_skipped(data):
            continue
        summary = data.get("summary", {})
        if not isinstance(summary, dict):
            summary = {}
        failure_types = [str(value).strip() for value in summary.get("failure_types", []) if str(value).strip()]
        transition_failures = [
            str(value).strip() for value in summary.get("transition_failures", []) if str(value).strip()
        ]
        if not failure_types and not transition_failures and bool(data.get("success")):
            continue
        task_id = str(data.get("task_id", "")).strip()
        if not task_id:
            continue
        contract = _task_contract_from_memory(data, task_id=task_id, bank=bank)
        if contract is None:
            continue
        strict_task = synthesize_stricter_task(
            contract,
            task_id=(
                f"{task_id}{_memory_task_rule_text('discovered_task', 'task_id_suffix', fallback='_discovered')}"
            ),
            extra_metadata=_memory_task_metadata(
                "discovered_task",
                source_task_id=task_id,
                origin_benchmark_family=str(contract.metadata.get("benchmark_family", "bounded")),
                extra={
                    "discovery_failure_types": failure_types,
                    "discovery_transition_failures": transition_failures,
                    "episode_phase": str(data.get("episode_storage", {}).get("phase", "")).strip(),
                    "episode_relative_path": str(data.get("episode_storage", {}).get("relative_path", "")).strip(),
                },
            ),
        )
        strict_task.prompt = _memory_task_prompt("discovered_task", prompt=strict_task.prompt)
        strict_task.workspace_subdir = (
            f"{contract.workspace_subdir}{_memory_task_rule_text('discovered_task', 'workspace_suffix', fallback='_discovered')}"
        )
        strict_task.max_steps = max(strict_task.max_steps, len(strict_task.suggested_commands) + 1)
        tasks.append(_annotate_light_supervision_contract(strict_task))
        if limit is not None and len(tasks) >= limit:
            break
    return tasks


def load_transition_pressure_tasks(episodes_root: Path, *, limit: int | None = None) -> list[TaskSpec]:
    bank = TaskBank()
    tasks: list[TaskSpec] = []
    for data in iter_episode_documents(episodes_root):
        if _synthetic_lineage_seed_skipped(data):
            continue
        summary = data.get("summary", {})
        if not isinstance(summary, dict):
            summary = {}
        transition_failures = [
            str(value).strip() for value in summary.get("transition_failures", []) if str(value).strip()
        ]
        if not transition_failures:
            continue
        task_id = str(data.get("task_id", "")).strip()
        if not task_id:
            continue
        contract = _task_contract_from_memory(data, task_id=task_id, bank=bank)
        if contract is None:
            continue
        pressure_task = synthesize_stricter_task(
            contract,
            task_id=(
                f"{task_id}{_memory_task_rule_text('transition_pressure', 'task_id_suffix', fallback='_transition_pressure')}"
            ),
            extra_metadata=_memory_task_metadata(
                "transition_pressure",
                source_task_id=task_id,
                origin_benchmark_family=str(contract.metadata.get("benchmark_family", "bounded")),
                extra={
                    "discovery_transition_failures": transition_failures,
                    "episode_phase": str(data.get("episode_storage", {}).get("phase", "")).strip(),
                    "episode_relative_path": str(data.get("episode_storage", {}).get("relative_path", "")).strip(),
                },
            ),
        )
        pressure_task.prompt = _memory_task_prompt(
            "transition_pressure",
            prompt=pressure_task.prompt,
            failure_modes=", ".join(transition_failures),
        )
        pressure_task.workspace_subdir = (
            f"{contract.workspace_subdir}{_memory_task_rule_text('transition_pressure', 'workspace_suffix', fallback='_transition_pressure')}"
        )
        pressure_task.max_steps = max(pressure_task.max_steps, len(pressure_task.suggested_commands) + 1)
        tasks.append(_annotate_light_supervision_contract(pressure_task))
        if limit is not None and len(tasks) >= limit:
            break
    return tasks


def _safe_worker_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip().replace("/", "_"))
    return normalized.strip("_") or "worker"


def _parallel_worker_candidate_paths(contract: dict[str, object]) -> list[str]:
    report_paths = {
        str(rule.get("path", "")).strip()
        for rule in contract.get("report_rules", [])
        if isinstance(rule, dict) and str(rule.get("path", "")).strip()
    }
    generated_paths = {
        str(path).strip()
        for path in contract.get("generated_paths", [])
        if str(path).strip()
    }
    resolved_conflict_paths = {
        str(path).strip()
        for path in contract.get("resolved_conflict_paths", [])
        if str(path).strip()
    }
    return [
        str(path).strip()
        for path in contract.get("expected_changed_paths", [])
        if str(path).strip()
        and str(path).strip() not in report_paths
        and str(path).strip() not in generated_paths
        and str(path).strip() not in resolved_conflict_paths
    ]


def _expanded_required_worker_branches(
    required_branches: list[str],
    *,
    changed_paths: list[str],
    target_worker_count: int | None,
) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for branch in required_branches:
        normalized = str(branch).strip()
        if normalized and normalized not in seen:
            deduped.append(normalized)
            seen.add(normalized)
    max_parallelism = max(len(deduped), min(len(changed_paths), _normalized_parallel_worker_count(target_worker_count)))
    if max_parallelism <= len(deduped):
        return deduped
    prioritized_paths = sorted(
        changed_paths,
        key=lambda path: (
            min(
                max((_branch_path_score(branch, path) for branch in deduped), default=0),
                10**6,
            ),
            path,
        ),
    )
    for path in prioritized_paths:
        if len(deduped) >= max_parallelism:
            break
        candidate = _synthetic_worker_branch_for_path(path, existing=seen)
        if candidate in seen:
            continue
        deduped.append(candidate)
        seen.add(candidate)
    return deduped


def _normalized_parallel_worker_count(value: int | None) -> int:
    try:
        parsed = int(value or 0)
    except (TypeError, ValueError):
        return 0
    return max(0, parsed)


def _synthetic_worker_branch_for_path(path: str, *, existing: set[str]) -> str:
    parts = [part for part in path.split("/") if part]
    if not parts:
        base = "worker/slice"
    else:
        top = _safe_worker_name(parts[0])
        leaf = _safe_worker_name(Path(parts[-1]).stem or parts[-1])
        if len(parts) > 1 and leaf == top:
            base = f"worker/{top}"
        else:
            base = f"worker/{top}-{leaf}"
    if base not in existing:
        return base
    suffix = 2
    while f"{base}-{suffix}" in existing:
        suffix += 1
    return f"{base}-{suffix}"


def _assign_paths_to_branches(required_branches: list[str], changed_paths: list[str]) -> dict[str, list[str]]:
    assignments: dict[str, list[str]] = {branch: [] for branch in required_branches}
    remaining_paths = list(changed_paths)
    for branch in required_branches:
        if not remaining_paths:
            break
        scored = sorted(
            (
                (_branch_path_score(branch, path), path)
                for path in remaining_paths
            ),
            key=lambda item: (-item[0], item[1]),
        )
        best_score, best_path = scored[0]
        if best_score > 0:
            assignments[branch].append(best_path)
            remaining_paths.remove(best_path)
    for path in list(remaining_paths):
        branch = max(
            required_branches,
            key=lambda candidate: (_branch_path_score(candidate, path), -required_branches.index(candidate)),
        )
        assignments[branch].append(path)
    return {branch: paths for branch, paths in assignments.items() if paths}


def _branch_path_score(branch: str, path: str) -> int:
    branch_tokens = set(_path_tokens(branch))
    path_tokens = set(_path_tokens(path))
    overlap = len(branch_tokens & path_tokens)
    path_parts = [part for part in path.split("/") if part]
    top_level_bonus = 2 if any(token == (path_parts[0] if path_parts else "") for token in branch_tokens) else 0
    leaf_bonus = 1 if path_tokens and branch_tokens and any(token in path_parts[-1] for token in branch_tokens) else 0
    return overlap * 5 + top_level_bonus + leaf_bonus


def _select_worker_test_commands(
    branch: str,
    assigned_paths: list[str],
    contract: dict[str, object],
) -> list[dict[str, object]]:
    tests = [
        dict(command)
        for command in contract.get("test_commands", [])
        if isinstance(command, dict)
    ]
    if not tests:
        return []
    branch_tokens = set(_path_tokens(branch))
    path_tokens = set(token for path in assigned_paths for token in _path_tokens(path))
    scored: list[tuple[int, dict[str, object]]] = []
    for test_command in tests:
        label_tokens = set(_path_tokens(str(test_command.get("label", ""))))
        argv_tokens = set(
            token
            for value in test_command.get("argv", [])
            for token in _path_tokens(str(value))
        ) if isinstance(test_command.get("argv", []), list) else set()
        score = len((branch_tokens | path_tokens) & (label_tokens | argv_tokens))
        scored.append((score, test_command))
    matched = [command for score, command in scored if score > 0]
    if matched:
        return matched
    if len(tests) == 1:
        return tests
    return []


def _worker_prompt(branch: str, assigned_paths: list[str]) -> str:
    owned = ", ".join(assigned_paths)
    template = str(
        _task_bank_synthesis_rules().get(
            "worker_prompt_template",
            "On branch {branch}, update only these worker-owned paths: {owned}. Keep unrelated paths unchanged, run any assigned tests, and commit the branch.",
        )
    )
    return template.format(branch=branch, owned=owned)


def _synthesized_worker_commands(
    *,
    branch: str,
    target_branch: str,
    changed_paths: list[str],
    edit_plan: list[dict[str, object]],
    expected_file_contents: dict[str, str],
    test_commands: list[dict[str, object]],
    report_rules: list[dict[str, object]],
) -> list[str]:
    if not changed_paths:
        return []
    write_commands: list[str] = []
    planned_writes = [
        dict(step)
        for step in edit_plan
        if isinstance(step, dict) and str(step.get("path", "")).strip() in changed_paths
    ]
    if not planned_writes:
        planned_writes = [
            {
                "path": path,
                "target_content": expected_file_contents.get(path),
            }
            for path in changed_paths
        ]
    for step in planned_writes:
        path = str(step.get("path", "")).strip()
        content = step.get("target_content")
        if not path:
            return []
        edit_kind = str(step.get("edit_kind", "rewrite")).strip() or "rewrite"
        if edit_kind == "block_replace":
            replacement = step.get("replacement", {})
            if not isinstance(replacement, dict):
                return []
            command = _render_block_replace_command(path, replacement)
            if not command:
                return []
            write_commands.append(command)
            continue
        if edit_kind == "line_insert":
            insertion = step.get("insertion", {})
            if not isinstance(insertion, dict):
                return []
            command = _render_line_insert_command(path, insertion)
            if not command:
                return []
            write_commands.append(command)
            continue
        if edit_kind == "line_delete":
            deletion = step.get("deletion", {})
            if not isinstance(deletion, dict):
                return []
            command = _render_line_delete_command(path, deletion)
            if not command:
                return []
            write_commands.append(command)
            continue
        if edit_kind == "token_replace":
            replacements = step.get("replacements", [])
            if not isinstance(replacements, list) or not replacements:
                return []
            write_commands.extend(_render_token_replace_commands(path, replacements))
            continue
        if edit_kind == "line_replace":
            replacements = step.get("replacements", [])
            if not isinstance(replacements, list) or not replacements:
                return []
            write_commands.extend(_render_line_replace_commands(path, replacements))
            continue
        if content is None:
            return []
        parent = Path(path).parent
        if str(parent) not in {"", "."}:
            write_commands.append(f"mkdir -p {shlex.quote(str(parent))}")
        write_commands.append(f"printf %s {shlex.quote(str(content))} > {shlex.quote(path)}")
    test_invocations = [
        " ".join(shlex.quote(str(part)) for part in command.get("argv", []))
        for command in test_commands
        if isinstance(command.get("argv", []), list) and command.get("argv", [])
    ]
    report_commands = [
        _render_report_write_command(rule)
        for rule in report_rules
        if _render_report_write_command(rule)
    ]
    git_add_paths = " ".join(shlex.quote(path) for path in changed_paths)
    if report_rules:
        git_add_paths = " ".join(
            [git_add_paths, *[shlex.quote(str(rule.get("path", "")).strip()) for rule in report_rules if str(rule.get("path", "")).strip()]]
        ).strip()
    commit_message = shlex.quote(f"worker update for {branch}")
    primary = " && ".join(
        [
            *write_commands,
            *test_invocations,
            *report_commands,
            f"git add {git_add_paths}",
            f"git commit -m {commit_message}",
        ]
    )
    return [
        primary,
        "git branch --show-current",
        f"git diff --name-only --relative {shlex.quote(f'origin/{target_branch}')}..HEAD",
    ]


def _derive_worker_edit_plan(
    *,
    branch: str,
    assigned_paths: list[str],
    assigned_tests: list[dict[str, object]],
    expected_file_contents: dict[str, str],
    bootstrap_file_contents: dict[str, str],
) -> list[dict[str, object]]:
    candidates = _derive_worker_edit_candidates(
        branch=branch,
        assigned_paths=assigned_paths,
        assigned_tests=assigned_tests,
        expected_file_contents=expected_file_contents,
        bootstrap_file_contents=bootstrap_file_contents,
    )
    return [
        dict(entry.get("selected", {}))
        for entry in candidates
        if isinstance(entry, dict) and isinstance(entry.get("selected"), dict)
    ]


def _derive_worker_edit_candidates(
    *,
    branch: str,
    assigned_paths: list[str],
    assigned_tests: list[dict[str, object]],
    expected_file_contents: dict[str, str],
    bootstrap_file_contents: dict[str, str],
) -> list[dict[str, object]]:
    test_expectations = _test_script_expectations(assigned_tests, bootstrap_file_contents)
    edit_candidates: list[dict[str, object]] = []
    for path in assigned_paths:
        baseline_content = bootstrap_file_contents.get(path, "")
        target_content = expected_file_contents.get(path)
        intent_source = "expected_file_contents"
        if target_content is None:
            target_content = test_expectations.get(path)
            if target_content is not None:
                intent_source = "assigned_tests"
                target_content = _merge_partial_target_into_baseline(
                    baseline_content=baseline_content,
                    target_content=target_content,
                )
        if target_content is None:
            target_content = _target_content_from_branch_intent(
                baseline_content=baseline_content,
                branch=branch,
                path=path,
            )
            if target_content is not None:
                intent_source = "branch_intent"
        if target_content is None:
            continue
        candidates: list[dict[str, object]] = []
        token_edit = _derive_token_replace_step(
            path=path,
            baseline_content=baseline_content,
            target_content=target_content,
            intent_source=intent_source,
        )
        if token_edit is not None:
            candidates.append(token_edit)
        insert_edit = _derive_line_insert_step(
            path=path,
            baseline_content=baseline_content,
            target_content=target_content,
            intent_source=intent_source,
        )
        if insert_edit is not None:
            candidates.append(insert_edit)
        delete_edit = _derive_line_delete_step(
            path=path,
            baseline_content=baseline_content,
            target_content=target_content,
            intent_source=intent_source,
        )
        if delete_edit is not None:
            candidates.append(delete_edit)
        block_edit = _derive_block_replace_step(
            path=path,
            baseline_content=baseline_content,
            target_content=target_content,
            intent_source=intent_source,
        )
        if block_edit is not None:
            candidates.append(block_edit)
        line_edit = _derive_line_replace_step(
            path=path,
            baseline_content=baseline_content,
            target_content=target_content,
            intent_source=intent_source,
        )
        if line_edit is not None:
            candidates.append(line_edit)
        candidates.append(
            _rewrite_edit_step(
                path=path,
                baseline_content=baseline_content,
                target_content=target_content,
                intent_source=intent_source,
            )
        )
        selected = _best_edit_candidate(candidates)
        if selected is not None:
            edit_candidates.append(
                {
                    "path": path,
                    "selected_kind": str(selected.get("edit_kind", "")).strip(),
                    "selected_score": int(selected.get("edit_score", 0)),
                    "selected": dict(selected),
                    "candidates": sorted(
                        [dict(candidate) for candidate in candidates],
                        key=lambda candidate: (
                            int(candidate.get("edit_score", 0)),
                            _edit_kind_rank(str(candidate.get("edit_kind", "rewrite"))),
                            str(candidate.get("edit_kind", "rewrite")),
                        ),
                    ),
                }
            )
    return edit_candidates


def _derive_line_replace_step(
    *,
    path: str,
    baseline_content: str,
    target_content: str,
    intent_source: str,
) -> dict[str, object] | None:
    if not baseline_content or baseline_content == target_content:
        return None
    baseline_lines = baseline_content.splitlines()
    target_lines = target_content.splitlines()
    if len(baseline_lines) != len(target_lines):
        return None
    replacements: list[dict[str, object]] = []
    for line_number, (before_line, after_line) in enumerate(zip(baseline_lines, target_lines), start=1):
        if before_line == after_line:
            continue
        replacements.append(
            {
                "line_number": line_number,
                "before_line": before_line,
                "after_line": after_line,
            }
        )
    if not replacements:
        return None
    return {
        "path": path,
        "baseline_content": baseline_content,
        "target_content": target_content,
        "edit_kind": "line_replace",
        "intent_source": intent_source,
        "replacements": replacements,
        "edit_score": _edit_candidate_score("line_replace", replacements=replacements),
    }


def _derive_token_replace_step(
    *,
    path: str,
    baseline_content: str,
    target_content: str,
    intent_source: str,
) -> dict[str, object] | None:
    if not baseline_content or baseline_content == target_content:
        return None
    baseline_lines = baseline_content.splitlines()
    target_lines = target_content.splitlines()
    if len(baseline_lines) != len(target_lines):
        return None
    replacements: list[dict[str, object]] = []
    for line_number, (before_line, after_line) in enumerate(zip(baseline_lines, target_lines), start=1):
        if before_line == after_line:
            continue
        fragment = _token_replacement_fragment(before_line, after_line)
        if fragment is None:
            return None
        before_fragment, after_fragment = fragment
        if before_line.count(before_fragment) != 1:
            return None
        replacements.append(
            {
                "line_number": line_number,
                "before_fragment": before_fragment,
                "after_fragment": after_fragment,
                "before_line": before_line,
                "after_line": after_line,
            }
        )
    if not replacements:
        return None
    return {
        "path": path,
        "baseline_content": baseline_content,
        "target_content": target_content,
        "edit_kind": "token_replace",
        "intent_source": intent_source,
        "replacements": replacements,
        "edit_score": _edit_candidate_score("token_replace", replacements=replacements),
    }


def _derive_block_replace_step(
    *,
    path: str,
    baseline_content: str,
    target_content: str,
    intent_source: str,
) -> dict[str, object] | None:
    if not baseline_content or baseline_content == target_content:
        return None
    baseline_lines = baseline_content.splitlines()
    target_lines = target_content.splitlines()
    prefix_length = 0
    while (
        prefix_length < len(baseline_lines)
        and prefix_length < len(target_lines)
        and baseline_lines[prefix_length] == target_lines[prefix_length]
    ):
        prefix_length += 1
    suffix_length = 0
    while (
        suffix_length < (len(baseline_lines) - prefix_length)
        and suffix_length < (len(target_lines) - prefix_length)
        and baseline_lines[len(baseline_lines) - 1 - suffix_length] == target_lines[len(target_lines) - 1 - suffix_length]
    ):
        suffix_length += 1
    baseline_start = prefix_length
    baseline_end = len(baseline_lines) - suffix_length
    target_start = prefix_length
    target_end = len(target_lines) - suffix_length
    baseline_block = baseline_lines[baseline_start:baseline_end]
    target_block = target_lines[target_start:target_end]
    if not baseline_block and not target_block:
        return None
    if len(baseline_block) <= 1 and len(target_block) <= 1:
        return None
    return {
        "path": path,
        "baseline_content": baseline_content,
        "target_content": target_content,
        "edit_kind": "block_replace",
        "intent_source": intent_source,
        "replacement": {
            "start_line": baseline_start + 1,
            "end_line": max(baseline_start + 1, baseline_end),
            "before_lines": baseline_block,
            "after_lines": target_block,
        },
        "edit_score": _edit_candidate_score(
            "block_replace",
            replacement={
                "start_line": baseline_start + 1,
                "end_line": max(baseline_start + 1, baseline_end),
                "before_lines": baseline_block,
                "after_lines": target_block,
            },
        ),
    }


def _derive_line_insert_step(
    *,
    path: str,
    baseline_content: str,
    target_content: str,
    intent_source: str,
) -> dict[str, object] | None:
    if not baseline_content or baseline_content == target_content:
        return None
    baseline_lines = baseline_content.splitlines()
    target_lines = target_content.splitlines()
    prefix_length = 0
    while (
        prefix_length < len(baseline_lines)
        and prefix_length < len(target_lines)
        and baseline_lines[prefix_length] == target_lines[prefix_length]
    ):
        prefix_length += 1
    suffix_length = 0
    while (
        suffix_length < (len(baseline_lines) - prefix_length)
        and suffix_length < (len(target_lines) - prefix_length)
        and baseline_lines[len(baseline_lines) - 1 - suffix_length] == target_lines[len(target_lines) - 1 - suffix_length]
    ):
        suffix_length += 1
    baseline_start = prefix_length
    baseline_end = len(baseline_lines) - suffix_length
    target_start = prefix_length
    target_end = len(target_lines) - suffix_length
    baseline_block = baseline_lines[baseline_start:baseline_end]
    target_block = target_lines[target_start:target_end]
    if baseline_block or not target_block:
        return None
    baseline_lines = baseline_content.splitlines()
    insertion = {
        "line_number": baseline_start + 1,
        "mode": "append" if baseline_start >= len(baseline_lines) else "before",
        "after_lines": target_block,
    }
    return {
        "path": path,
        "baseline_content": baseline_content,
        "target_content": target_content,
        "edit_kind": "line_insert",
        "intent_source": intent_source,
        "insertion": insertion,
        "edit_score": _edit_candidate_score("line_insert", insertion=insertion),
    }


def _rewrite_edit_step(
    *,
    path: str,
    baseline_content: str,
    target_content: str,
    intent_source: str,
) -> dict[str, object]:
    return {
        "path": path,
        "baseline_content": baseline_content,
        "target_content": target_content,
        "edit_kind": "rewrite",
        "intent_source": intent_source,
        "edit_score": _edit_candidate_score("rewrite", target_content=target_content),
    }


def _derive_line_delete_step(
    *,
    path: str,
    baseline_content: str,
    target_content: str,
    intent_source: str,
) -> dict[str, object] | None:
    if not baseline_content or baseline_content == target_content:
        return None
    baseline_lines = baseline_content.splitlines()
    target_lines = target_content.splitlines()
    prefix_length = 0
    while (
        prefix_length < len(baseline_lines)
        and prefix_length < len(target_lines)
        and baseline_lines[prefix_length] == target_lines[prefix_length]
    ):
        prefix_length += 1
    suffix_length = 0
    while (
        suffix_length < (len(baseline_lines) - prefix_length)
        and suffix_length < (len(target_lines) - prefix_length)
        and baseline_lines[len(baseline_lines) - 1 - suffix_length] == target_lines[len(target_lines) - 1 - suffix_length]
    ):
        suffix_length += 1
    baseline_start = prefix_length
    baseline_end = len(baseline_lines) - suffix_length
    target_end = len(target_lines) - suffix_length
    baseline_block = baseline_lines[baseline_start:baseline_end]
    target_block = target_lines[prefix_length:target_end]
    if not baseline_block or target_block:
        return None
    deletion = {
        "start_line": baseline_start + 1,
        "end_line": max(baseline_start + 1, baseline_end),
        "before_lines": baseline_block,
    }
    return {
        "path": path,
        "baseline_content": baseline_content,
        "target_content": target_content,
        "edit_kind": "line_delete",
        "intent_source": intent_source,
        "deletion": deletion,
        "edit_score": _edit_candidate_score("line_delete", deletion=deletion),
    }


def _best_edit_candidate(candidates: list[dict[str, object]]) -> dict[str, object] | None:
    valid = [
        dict(candidate)
        for candidate in candidates
        if isinstance(candidate, dict) and str(candidate.get("path", "")).strip()
    ]
    if not valid:
        return None
    return min(
        valid,
        key=lambda candidate: (
            int(candidate.get("edit_score", _edit_candidate_score(str(candidate.get("edit_kind", "rewrite"))))),
            _edit_kind_rank(str(candidate.get("edit_kind", "rewrite"))),
            str(candidate.get("edit_kind", "rewrite")),
        ),
    )


def _edit_candidate_score(
    edit_kind: str,
    *,
    replacements: list[dict[str, object]] | None = None,
    replacement: dict[str, object] | None = None,
    insertion: dict[str, object] | None = None,
    deletion: dict[str, object] | None = None,
    target_content: str = "",
) -> int:
    normalized_kind = str(edit_kind).strip() or "rewrite"
    score_rules = dict(_task_bank_synthesis_rules().get("edit_scores", {}))
    kind_rules = dict(score_rules.get(normalized_kind, {})) if isinstance(score_rules.get(normalized_kind, {}), dict) else {}
    base = int(kind_rules.get("base", 120 if normalized_kind == "rewrite" else 0) or 0)
    char_weight = int(kind_rules.get("char_weight", 1) or 1)
    if normalized_kind == "token_replace":
        ops = replacements or []
        fragment_chars = sum(len(str(item.get("before_fragment", ""))) + len(str(item.get("after_fragment", ""))) for item in ops)
        return base + len(ops) * int(kind_rules.get("per_replacement", 0) or 0) + fragment_chars * char_weight
    if normalized_kind == "line_replace":
        ops = replacements or []
        changed_chars = sum(len(str(item.get("before_line", ""))) + len(str(item.get("after_line", ""))) for item in ops)
        return base + len(ops) * int(kind_rules.get("per_replacement", 0) or 0) + changed_chars * char_weight
    if normalized_kind == "line_insert":
        inserted = insertion or {}
        raw_after_lines = inserted.get("after_lines", inserted.get("inserted_lines", []))
        after_lines = [str(line) for line in raw_after_lines]
        changed_chars = sum(len(line) for line in after_lines)
        return base + len(after_lines) * int(kind_rules.get("per_line", 0) or 0) + changed_chars * char_weight
    if normalized_kind == "line_delete":
        removed = deletion or {}
        before_lines = [str(line) for line in removed.get("before_lines", [])]
        changed_chars = sum(len(line) for line in before_lines)
        return base + len(before_lines) * int(kind_rules.get("per_line", 0) or 0) + changed_chars * char_weight
    if normalized_kind == "block_replace":
        block = replacement or {}
        before_lines = [str(line) for line in block.get("before_lines", [])]
        after_lines = [str(line) for line in block.get("after_lines", [])]
        changed_chars = sum(len(line) for line in before_lines) + sum(len(line) for line in after_lines)
        changed_lines = max(len(before_lines), len(after_lines))
        return base + changed_lines * int(kind_rules.get("per_line", 0) or 0) + changed_chars * char_weight
    return base + len(str(target_content)) * char_weight


def _edit_kind_rank(edit_kind: str) -> int:
    ordered_kinds = [
        str(value).strip()
        for value in _task_bank_synthesis_rules().get("edit_kind_order", [])
        if str(value).strip()
    ]
    order = {kind: index for index, kind in enumerate(ordered_kinds)}
    return order.get(str(edit_kind).strip() or "rewrite", 99)


def _token_replacement_fragment(before_line: str, after_line: str) -> tuple[str, str] | None:
    if before_line == after_line:
        return None
    prefix_length = 0
    while (
        prefix_length < len(before_line)
        and prefix_length < len(after_line)
        and before_line[prefix_length] == after_line[prefix_length]
    ):
        prefix_length += 1
    suffix_length = 0
    while (
        suffix_length < (len(before_line) - prefix_length)
        and suffix_length < (len(after_line) - prefix_length)
        and before_line[len(before_line) - 1 - suffix_length] == after_line[len(after_line) - 1 - suffix_length]
    ):
        suffix_length += 1
    before_fragment = before_line[prefix_length : len(before_line) - suffix_length if suffix_length else len(before_line)]
    after_fragment = after_line[prefix_length : len(after_line) - suffix_length if suffix_length else len(after_line)]
    if not before_fragment or before_fragment == after_fragment:
        return None
    if before_fragment == before_line and after_fragment == after_line:
        return None
    if "\n" in before_fragment or "\n" in after_fragment:
        return None
    return before_fragment, after_fragment


def _bootstrap_file_contents(task: TaskSpec) -> dict[str, str]:
    metadata = dict(task.metadata)
    fixture_dir = str(metadata.get("shared_repo_bootstrap_fixture_dir", "")).strip()
    managed_paths = [
        str(path).strip()
        for path in metadata.get("shared_repo_bootstrap_managed_paths", [])
        if str(path).strip()
    ]
    if fixture_dir:
        contents: dict[str, str] = {}
        fixture_root = (_BUILTIN_TASK_MANIFEST_PATH.parent / "shared_repo_fixtures" / fixture_dir).resolve()
        for relative_path in managed_paths:
            source = fixture_root / relative_path
            if source.exists() and source.is_file():
                contents[relative_path] = source.read_text(encoding="utf-8")
        return contents
    commands = [
        str(command).strip()
        for command in metadata.get("shared_repo_bootstrap_commands", [])
        if str(command).strip()
    ]
    if not commands:
        commands = list(task.setup_commands)
    contents: dict[str, str] = {}
    for command in commands:
        for path, content in _command_file_writes(command).items():
            contents[path] = content
    return contents


def _command_file_writes(command: str) -> dict[str, str]:
    writes: dict[str, str] = {}
    for segment in [part.strip() for part in command.split("&&") if part.strip()]:
        try:
            tokens = shlex.split(segment, posix=True)
        except ValueError:
            continue
        if not tokens or tokens[0] != "printf":
            continue
        content_start = 1
        if len(tokens) > 1 and tokens[1] == "%s":
            content_start = 2
        if ">" not in tokens[content_start:]:
            continue
        redirect_index = tokens.index(">", content_start)
        if redirect_index <= content_start or redirect_index + 1 >= len(tokens):
            continue
        path = str(tokens[redirect_index + 1]).strip()
        if not path:
            continue
        writes[path] = _decode_shell_literal("".join(tokens[content_start:redirect_index]))
    return writes


def _decode_shell_literal(value: str) -> str:
    text = str(value)
    try:
        return bytes(text, "utf-8").decode("unicode_escape")
    except UnicodeDecodeError:
        return text


def _merge_partial_target_into_baseline(*, baseline_content: str, target_content: str) -> str:
    if not baseline_content or not target_content:
        return target_content
    baseline_lines = baseline_content.splitlines()
    target_lines = target_content.splitlines()
    if len(target_lines) != 1 or len(baseline_lines) <= 1:
        return target_content
    target_line = target_lines[0]
    if target_line in baseline_lines:
        return baseline_content
    replacement_index = _baseline_line_replacement_index(baseline_lines, target_line)
    if replacement_index is None:
        return target_content
    merged_lines = list(baseline_lines)
    merged_lines[replacement_index] = _merge_target_into_baseline_line(
        baseline_lines[replacement_index],
        target_line,
    )
    merged = "\n".join(merged_lines)
    if baseline_content.endswith("\n"):
        merged += "\n"
    return merged


def _baseline_line_replacement_index(baseline_lines: list[str], target_line: str) -> int | None:
    if "=" in target_line:
        key, _, _ = target_line.partition("=")
        matches = [index for index, line in enumerate(baseline_lines) if line.partition("=")[0] == key]
        if len(matches) == 1:
            return matches[0]
    target_tokens = set(_path_tokens(target_line))
    if not target_tokens:
        return None
    scored = [
        (len(target_tokens & set(_path_tokens(line))), index)
        for index, line in enumerate(baseline_lines)
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    if not scored or scored[0][0] <= 0:
        return None
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        return None
    return scored[0][1]


def _merge_target_into_baseline_line(baseline_line: str, target_line: str) -> str:
    if not baseline_line or not target_line or target_line == baseline_line:
        return target_line or baseline_line
    if "=" in target_line and "=" in baseline_line:
        baseline_left, _, baseline_right = baseline_line.partition("=")
        target_left, _, target_right = target_line.partition("=")
        if set(_path_tokens(target_left)) & set(_path_tokens(baseline_left)):
            suffix = ""
            if baseline_right.rstrip().endswith(";") and not target_right.rstrip().endswith(";"):
                suffix = ";"
            spacing = " " if baseline_left.endswith(" ") or target_right.startswith(" ") else ""
            return f"{baseline_left}={spacing}{target_right.strip()}{suffix}"
    return target_line


def _render_line_replace_commands(path: str, replacements: list[dict[str, object]]) -> list[str]:
    commands: list[str] = []
    for replacement in replacements:
        try:
            line_number = int(replacement.get("line_number", 0))
        except (TypeError, ValueError):
            continue
        before_line = str(replacement.get("before_line", ""))
        after_line = str(replacement.get("after_line", ""))
        if line_number <= 0 or before_line == after_line:
            continue
        script = (
            f"{line_number}s#^{_sed_regex_escape(before_line)}$#"
            f"{_sed_replacement_escape(after_line)}#"
        )
        commands.append(f"sed -i {shlex.quote(script)} {shlex.quote(path)}")
    return commands


def _render_block_replace_command(path: str, replacement: dict[str, object]) -> str:
    try:
        start_line = int(replacement.get("start_line", 0))
        end_line = int(replacement.get("end_line", 0))
    except (TypeError, ValueError):
        return ""
    after_lines = [str(line) for line in replacement.get("after_lines", [])]
    if start_line <= 0 or end_line < start_line:
        return ""
    if not after_lines:
        script = f"{start_line},{end_line}d"
        return f"sed -i {shlex.quote(script)} {shlex.quote(path)}"
    replacement_body = "\\\n".join(_sed_block_text_escape(line) for line in after_lines)
    script = f"{start_line},{end_line}c\\\n{replacement_body}"
    return f"sed -i {shlex.quote(script)} {shlex.quote(path)}"


def _render_line_insert_command(path: str, insertion: dict[str, object]) -> str:
    try:
        line_number = int(insertion.get("line_number", 0))
    except (TypeError, ValueError):
        return ""
    raw_after_lines = insertion.get("after_lines", insertion.get("inserted_lines", []))
    after_lines = [str(line) for line in raw_after_lines]
    mode = str(insertion.get("mode", "before")).strip() or "before"
    if line_number <= 0 or not after_lines:
        return ""
    replacement_body = "\\\n".join(_sed_block_text_escape(line) for line in after_lines)
    if mode == "append":
        script = f"$a\\\n{replacement_body}"
    else:
        script = f"{line_number}i\\\n{replacement_body}"
    return f"sed -i {shlex.quote(script)} {shlex.quote(path)}"


def _render_line_delete_command(path: str, deletion: dict[str, object]) -> str:
    try:
        start_line = int(deletion.get("start_line", 0))
        end_line = int(deletion.get("end_line", 0))
    except (TypeError, ValueError):
        return ""
    if start_line <= 0 or end_line < start_line:
        return ""
    script = f"{start_line},{end_line}d"
    return f"sed -i '{script}' {shlex.quote(path)}"


def _render_token_replace_commands(path: str, replacements: list[dict[str, object]]) -> list[str]:
    commands: list[str] = []
    for replacement in replacements:
        try:
            line_number = int(replacement.get("line_number", 0))
        except (TypeError, ValueError):
            continue
        before_fragment = str(replacement.get("before_fragment", ""))
        after_fragment = str(replacement.get("after_fragment", ""))
        if line_number <= 0 or not before_fragment or before_fragment == after_fragment:
            continue
        script = (
            f"{line_number}s#{_sed_regex_escape(before_fragment)}#"
            f"{_sed_replacement_escape(after_fragment)}#"
        )
        commands.append(f"sed -i {shlex.quote(script)} {shlex.quote(path)}")
    return commands


def _sed_regex_escape(value: str) -> str:
    escaped = re.escape(value)
    return escaped.replace("#", r"\#")


def _sed_replacement_escape(value: str) -> str:
    return value.replace("\\", r"\\").replace("&", r"\&").replace("#", r"\#")


def _sed_block_text_escape(value: str) -> str:
    return value.replace("\\", r"\\")


def _test_script_expectations(
    test_commands: list[dict[str, object]],
    bootstrap_file_contents: dict[str, str],
) -> dict[str, str]:
    expectations: dict[str, str] = {}
    for command in test_commands:
        argv = command.get("argv", [])
        if not isinstance(argv, list):
            continue
        for argv_part in argv:
            script_path = str(argv_part).strip()
            if not script_path:
                continue
            script_content = bootstrap_file_contents.get(script_path)
            if not script_content:
                continue
            for path, content in _grep_expectations_from_script(script_content).items():
                expectations[path] = content
    return expectations


def _grep_expectations_from_script(script_content: str) -> dict[str, str]:
    expectations: dict[str, str] = {}
    for match in re.finditer(
        r"grep\s+-q\s+(?P<quote>['\"])\^(?P<needle>.+?)\$(?P=quote)\s+(?P<path>[^\s]+)",
        script_content,
    ):
        path = str(match.group("path")).strip().strip("'\"")
        if path:
            expectations[path] = _decode_shell_literal(str(match.group("needle"))) + "\n"
    for match in re.finditer(
        r"grep\s+-q\s+(?P<quote>['\"])(?P<needle>.+?)(?P=quote)\s+(?P<path>[^\s]+)",
        script_content,
    ):
        path = str(match.group("path")).strip().strip("'\"")
        needle = _decode_shell_literal(str(match.group("needle")))
        if not path or path in expectations:
            continue
        if needle.startswith("^") and needle.endswith("$"):
            continue
        expectations[path] = needle
    return expectations


def _target_content_from_branch_intent(
    *,
    baseline_content: str,
    branch: str,
    path: str,
) -> str | None:
    if not baseline_content:
        return None
    preferred_state = _preferred_branch_state(branch, path)
    if not preferred_state:
        return None
    candidate = baseline_content
    changed = False
    branch_rules = dict(_task_bank_synthesis_rules().get("branch_intent", {}))
    direct_replacements = [
        str(value).strip()
        for value in branch_rules.get("direct_replacements", [])
        if str(value).strip()
    ]
    replacements = {source: preferred_state for source in direct_replacements}
    replacements["todo"] = str(
        branch_rules.get("todo_replacement", "done" if preferred_state == "ready" else preferred_state)
    )
    replacements["draft"] = str(
        branch_rules.get("draft_replacement", "final" if preferred_state == "ready" else preferred_state)
    )
    for source, target in replacements.items():
        normalized_target = preferred_state if target == "{preferred_state}" else target
        candidate, count = re.subn(rf"\b{re.escape(source)}\b", normalized_target, candidate, flags=re.IGNORECASE)
        if count:
            changed = True
    if not changed and "=" in candidate and preferred_state not in candidate:
        rewritten_lines: list[str] = []
        for line in candidate.splitlines():
            if "=" in line:
                key, _, _ = line.partition("=")
                rewritten_lines.append(f"{key}={preferred_state}")
                changed = True
            else:
                rewritten_lines.append(line)
        candidate = "\n".join(rewritten_lines)
        if baseline_content.endswith("\n"):
            candidate += "\n"
    return candidate if changed and candidate != baseline_content else None


def _preferred_branch_state(branch: str, path: str) -> str:
    branch_rules = dict(_task_bank_synthesis_rules().get("branch_intent", {}))
    preferred_states = (
        dict(branch_rules.get("preferred_states", {}))
        if isinstance(branch_rules.get("preferred_states", {}), dict)
        else {}
    )
    for token in [*_path_tokens(branch), *_path_tokens(path)]:
        normalized = str(preferred_states.get(token, "")).strip()
        if normalized:
            return normalized
    return ""


def _path_tokens(value: str) -> list[str]:
    return [
        token
        for token in re.split(r"[^A-Za-z0-9]+", value.lower())
        if token
    ]


def _derive_worker_report_rules(
    branch: str,
    changed_paths: list[str],
    test_commands: list[dict[str, object]],
) -> list[dict[str, object]]:
    if not changed_paths:
        return []
    report_rules = dict(_task_bank_synthesis_rules().get("worker_report", {}))
    report_template = str(report_rules.get("path_template", "reports/{worker_name}_report.txt"))
    report_path = report_template.format(worker_name=_safe_worker_name(branch), branch=branch)
    must_mention = [
        *[str(value).strip() for value in report_rules.get("base_must_mention", []) if str(value).strip()],
        branch,
    ]
    test_cover_paths: list[str] = []
    for command in test_commands:
        label = str(command.get("label", "")).strip()
        if label:
            must_mention.extend(token for token in _path_tokens(label) if token not in must_mention)
        for argv_part in command.get("argv", []):
            path = str(argv_part).strip()
            if "/" in path:
                test_cover_paths.append(path)
    covers = [*changed_paths, *test_cover_paths]
    return [
        {
            "path": report_path,
            "must_mention": must_mention,
            "covers": covers,
        }
    ]


def _render_report_write_command(rule: dict[str, object]) -> str:
    path = str(rule.get("path", "")).strip()
    if not path:
        return ""
    parent = Path(path).parent
    must_mention = [
        str(value).strip()
        for value in rule.get("must_mention", [])
        if str(value).strip()
    ]
    covers = [
        str(value).strip()
        for value in rule.get("covers", [])
        if str(value).strip()
    ]
    body = " ".join([*must_mention, *covers]).strip()
    commands: list[str] = []
    if str(parent) not in {"", "."}:
        commands.append(f"mkdir -p {shlex.quote(str(parent))}")
    commands.append(f"printf %s {shlex.quote(body + chr(10))} > {shlex.quote(path)}")
    return " && ".join(commands)


def load_skill_replay_tasks(skills_path: Path, *, limit: int | None = None) -> list[TaskSpec]:
    if not skills_path.exists():
        return []
    bank = TaskBank()
    tasks: list[TaskSpec] = []
    payload = json.loads(skills_path.read_text(encoding="utf-8"))
    if not _artifact_payload_in_states(payload, {"promoted", "retained"}):
        return []
    skills = payload.get("skills", payload) if isinstance(payload, dict) else payload
    for skill in skills:
        source_task_id = str(skill.get("source_task_id", skill.get("task_id", ""))).strip()
        if not source_task_id:
            continue
        contract = _task_contract_from_memory(skill, task_id=source_task_id, bank=bank)
        if contract is None:
            continue
        procedure = list(skill.get("procedure", {}).get("commands", []))
        metadata = dict(contract.metadata)
        metadata.update(
            _memory_task_metadata(
                "skill_replay",
                source_task_id=source_task_id,
                origin_benchmark_family=str(contract.metadata.get("benchmark_family", "bounded")),
            )
        )
        replay_task = TaskSpec(
            task_id=f"{source_task_id}{_memory_task_rule_text('skill_replay', 'task_id_suffix', fallback='_skill_replay')}",
            prompt=_memory_task_prompt("skill_replay", prompt=contract.prompt),
            workspace_subdir=(
                f"{source_task_id}{_memory_task_rule_text('skill_replay', 'workspace_suffix', fallback='_skill_replay')}"
            ),
            setup_commands=list(contract.setup_commands),
            success_command=contract.success_command,
            suggested_commands=procedure or list(contract.suggested_commands),
            expected_files=list(contract.expected_files),
            expected_output_substrings=list(contract.expected_output_substrings),
            forbidden_files=list(contract.forbidden_files),
            forbidden_output_substrings=list(contract.forbidden_output_substrings),
            expected_file_contents=dict(contract.expected_file_contents),
            max_steps=max(contract.max_steps, max(1, len(procedure) + 1)),
            metadata=metadata,
        )
        tasks.append(_annotate_light_supervision_contract(replay_task))
        if limit is not None and len(tasks) >= limit:
            break
    return tasks


def load_verifier_replay_tasks(
    episodes_root: Path,
    skills_path: Path,
    *,
    limit: int | None = None,
) -> list[TaskSpec]:
    tasks: list[TaskSpec] = []
    for task in load_episode_replay_tasks(episodes_root, limit=limit):
        strict_task = synthesize_stricter_task(
            task,
            task_id=(
                f"{task.task_id}{_memory_task_rule_text('verifier_replay', 'task_id_suffix', fallback='_verifier_replay')}"
            ),
            extra_metadata=_memory_task_metadata(
                "verifier_replay",
                source_task_id=str(task.metadata.get("memory_source_task", task.task_id)),
                origin_benchmark_family=str(
                    task.metadata.get("origin_benchmark_family", task.metadata.get("benchmark_family", "bounded"))
                ),
                extra={
                    "verifier_source": "episode",
                    "memory_source_task": str(task.metadata.get("memory_source_task", task.task_id)),
                    "source_task": str(task.metadata.get("source_task", task.task_id)),
                },
            ),
        )
        strict_task.workspace_subdir = (
            f"{task.workspace_subdir}{_memory_task_rule_text('verifier_replay', 'workspace_suffix', fallback='_verifier_replay')}"
        )
        strict_task.max_steps = max(strict_task.max_steps, len(strict_task.suggested_commands) + 1)
        tasks.append(_annotate_light_supervision_contract(strict_task))
        if limit is not None and len(tasks) >= limit:
            return tasks
    remaining = None if limit is None else max(0, limit - len(tasks))
    for task in load_skill_replay_tasks(skills_path, limit=remaining):
        strict_task = synthesize_stricter_task(
            task,
            task_id=(
                f"{task.task_id}{_memory_task_rule_text('verifier_replay', 'task_id_suffix', fallback='_verifier_replay')}"
            ),
            extra_metadata=_memory_task_metadata(
                "verifier_replay",
                source_task_id=str(task.metadata.get("memory_source_task", task.task_id)),
                origin_benchmark_family=str(
                    task.metadata.get("origin_benchmark_family", task.metadata.get("benchmark_family", "bounded"))
                ),
                extra={
                    "verifier_source": "skill",
                    "memory_source_task": str(task.metadata.get("memory_source_task", task.task_id)),
                    "source_task": str(task.metadata.get("source_task", task.task_id)),
                },
            ),
        )
        strict_task.workspace_subdir = (
            f"{task.workspace_subdir}{_memory_task_rule_text('verifier_replay', 'workspace_suffix', fallback='_verifier_replay')}"
        )
        strict_task.max_steps = max(strict_task.max_steps, len(strict_task.suggested_commands) + 1)
        tasks.append(_annotate_light_supervision_contract(strict_task))
        if limit is not None and len(tasks) >= limit:
            break
    return tasks


def load_skill_transfer_tasks(
    skills_path: Path,
    *,
    limit: int | None = None,
    target_task_by_source: dict[str, str] | None = None,
) -> list[TaskSpec]:
    if not skills_path.exists():
        return []
    bank = TaskBank()
    payload = json.loads(skills_path.read_text(encoding="utf-8"))
    if not _artifact_payload_in_states(payload, {"promoted", "retained"}):
        return []
    skills = payload.get("skills", payload) if isinstance(payload, dict) else payload
    tasks: list[TaskSpec] = []
    for skill in skills:
        if not isinstance(skill, dict):
            continue
        source_task_id = str(skill.get("source_task_id", skill.get("task_id", ""))).strip()
        if not source_task_id:
            continue
        try:
            source_task = bank.get(source_task_id)
        except KeyError:
            continue
        procedure = list(skill.get("procedure", {}).get("commands", []))
        target_task = _resolve_transfer_target(
            bank,
            target_task_by_source=target_task_by_source,
            source_task_id=source_task_id,
            capability=str(source_task.metadata.get("capability", "unknown")),
            benchmark_family=str(source_task.metadata.get("benchmark_family", "bounded")),
        )
        if target_task is None:
            continue
        metadata = dict(target_task.metadata)
        metadata.update(
            _memory_task_metadata(
                "skill_transfer",
                source_task_id=source_task_id,
                origin_benchmark_family=str(target_task.metadata.get("benchmark_family", "bounded")),
                extra={"transfer_target_task": target_task.task_id},
            )
        )
        tasks.append(
            _annotate_light_supervision_contract(TaskSpec(
                task_id=(
                    f"{source_task_id}_to_{target_task.task_id}"
                    f"{_memory_task_rule_text('skill_transfer', 'task_id_suffix', fallback='_skill_transfer')}"
                ),
                prompt=_memory_task_prompt("skill_transfer", prompt=target_task.prompt),
                workspace_subdir=(
                    f"{target_task.workspace_subdir}{_memory_task_rule_text('skill_transfer', 'workspace_suffix', fallback='_skill_transfer')}"
                ),
                setup_commands=list(target_task.setup_commands),
                success_command=target_task.success_command,
                suggested_commands=procedure or list(target_task.suggested_commands),
                expected_files=list(target_task.expected_files),
                expected_output_substrings=list(target_task.expected_output_substrings),
                forbidden_files=list(target_task.forbidden_files),
                forbidden_output_substrings=list(target_task.forbidden_output_substrings),
                expected_file_contents=dict(target_task.expected_file_contents),
                max_steps=max(target_task.max_steps, len(procedure) + 1),
                metadata=metadata,
            ))
        )
        if limit is not None and len(tasks) >= limit:
            break
    return tasks


def load_operator_replay_tasks(
    operator_classes_path: Path,
    *,
    limit: int | None = None,
    target_task_by_operator: dict[str, str] | None = None,
) -> list[TaskSpec]:
    if not operator_classes_path.exists():
        return []
    bank = TaskBank()
    payload = json.loads(operator_classes_path.read_text(encoding="utf-8"))
    if not _artifact_payload_in_states(payload, {"promoted", "retained"}):
        return []
    operators = payload.get("operators", payload) if isinstance(payload, dict) else payload
    tasks: list[TaskSpec] = []
    for operator in operators:
        if not isinstance(operator, dict):
            continue
        source_task_ids = [str(value) for value in operator.get("source_task_ids", []) if str(value).strip()]
        capabilities = [str(value) for value in operator.get("applicable_capabilities", []) if str(value).strip()]
        families = [str(value) for value in operator.get("applicable_benchmark_families", []) if str(value).strip()]
        target_task = _resolve_operator_target(
            bank,
            target_task_by_operator=target_task_by_operator,
            operator_id=str(operator.get("operator_id", "")),
            source_task_ids=source_task_ids,
            capabilities=capabilities,
            benchmark_families=families,
        )
        if target_task is None:
            continue
        commands = instantiate_operator_commands(operator, target_task)
        metadata = dict(target_task.metadata)
        metadata.update(
            _memory_task_metadata(
                "operator_replay",
                source_task_id=",".join(source_task_ids),
                origin_benchmark_family=str(target_task.metadata.get("benchmark_family", "bounded")),
                extra={
                    "source_task": source_task_ids[0] if source_task_ids else "",
                    "transfer_target_task": target_task.task_id,
                    "operator_id": str(operator.get("operator_id", "")),
                },
            )
        )
        tasks.append(
            _annotate_light_supervision_contract(TaskSpec(
                task_id=(
                    f"{str(operator.get('operator_id', 'operator')).replace(':', '_')}_{target_task.task_id}"
                    f"{_memory_task_rule_text('operator_replay', 'task_id_suffix', fallback='_operator_replay')}"
                ),
                prompt=_memory_task_prompt("operator_replay", prompt=target_task.prompt),
                workspace_subdir=(
                    f"{target_task.workspace_subdir}{_memory_task_rule_text('operator_replay', 'workspace_suffix', fallback='_operator_replay')}"
                ),
                setup_commands=list(target_task.setup_commands),
                success_command=target_task.success_command,
                suggested_commands=commands or list(target_task.suggested_commands),
                expected_files=list(target_task.expected_files),
                expected_output_substrings=list(target_task.expected_output_substrings),
                forbidden_files=list(target_task.forbidden_files),
                forbidden_output_substrings=list(target_task.forbidden_output_substrings),
                expected_file_contents=dict(target_task.expected_file_contents),
                max_steps=max(target_task.max_steps, len(commands) + 1),
                metadata=metadata,
            ))
        )
        if limit is not None and len(tasks) >= limit:
            break
    return tasks


def load_benchmark_candidate_tasks(
    benchmark_candidates_path: Path,
    *,
    limit: int | None = None,
) -> list[TaskSpec]:
    if not benchmark_candidates_path.exists():
        return []
    bank = TaskBank()
    payload = json.loads(benchmark_candidates_path.read_text(encoding="utf-8"))
    if not _artifact_payload_in_states(payload, {"proposed", "retained"}):
        return []
    proposals = payload.get("proposals", payload) if isinstance(payload, dict) else payload
    tasks: list[TaskSpec] = []
    for proposal in proposals:
        source_task_id = str(proposal.get("source_task_id", "")).strip()
        prompt = str(proposal.get("prompt", "")).strip()
        if not source_task_id or not prompt:
            continue
        try:
            source_task = bank.get(source_task_id)
        except KeyError:
            continue
        metadata = dict(source_task.metadata)
        metadata.update(
            _memory_task_metadata(
                "benchmark_candidate",
                source_task_id=source_task_id,
                origin_benchmark_family=str(proposal.get("benchmark_family", metadata.get("benchmark_family", "bounded"))),
                extra={"candidate_kind": str(proposal.get("kind", ""))},
            )
        )
        tasks.append(
            _annotate_light_supervision_contract(TaskSpec(
                task_id=(
                    f"{source_task_id}{_memory_task_rule_text('benchmark_candidate', 'task_id_suffix', fallback='_benchmark_candidate')}"
                ),
                prompt=prompt,
                workspace_subdir=(
                    f"{source_task.workspace_subdir}{_memory_task_rule_text('benchmark_candidate', 'workspace_suffix', fallback='_benchmark_candidate')}"
                ),
                setup_commands=list(source_task.setup_commands),
                success_command=source_task.success_command,
                suggested_commands=list(source_task.suggested_commands),
                expected_files=list(source_task.expected_files),
                expected_output_substrings=list(source_task.expected_output_substrings),
                forbidden_files=list(source_task.forbidden_files),
                forbidden_output_substrings=list(source_task.forbidden_output_substrings),
                expected_file_contents=dict(source_task.expected_file_contents),
                max_steps=source_task.max_steps,
                metadata=metadata,
            ))
        )
        if limit is not None and len(tasks) >= limit:
            break
    return tasks


def load_verifier_candidate_tasks(
    verifier_contracts_path: Path,
    *,
    limit: int | None = None,
) -> list[TaskSpec]:
    if not verifier_contracts_path.exists():
        return []
    bank = TaskBank()
    payload = json.loads(verifier_contracts_path.read_text(encoding="utf-8"))
    if not _artifact_payload_in_states(payload, {"proposed", "retained"}):
        return []
    proposals = payload.get("proposals", payload) if isinstance(payload, dict) else payload
    tasks: list[TaskSpec] = []
    for proposal in proposals:
        source_task_id = str(proposal.get("source_task_id", "")).strip()
        contract = proposal.get("contract", {})
        if not source_task_id or not isinstance(contract, dict):
            continue
        try:
            source_task = bank.get(source_task_id)
        except KeyError:
            continue
        metadata = dict(source_task.metadata)
        metadata.update(
            _memory_task_metadata(
                "verifier_candidate",
                source_task_id=source_task_id,
                origin_benchmark_family=str(proposal.get("benchmark_family", metadata.get("benchmark_family", "bounded"))),
            )
        )
        tasks.append(
            _annotate_light_supervision_contract(TaskSpec(
                task_id=(
                    f"{source_task_id}{_memory_task_rule_text('verifier_candidate', 'task_id_suffix', fallback='_verifier_candidate')}"
                ),
                prompt=source_task.prompt,
                workspace_subdir=(
                    f"{source_task.workspace_subdir}{_memory_task_rule_text('verifier_candidate', 'workspace_suffix', fallback='_verifier_candidate')}"
                ),
                setup_commands=list(source_task.setup_commands),
                success_command=source_task.success_command,
                suggested_commands=list(source_task.suggested_commands),
                expected_files=list(contract.get("expected_files", source_task.expected_files)),
                expected_output_substrings=list(source_task.expected_output_substrings),
                forbidden_files=list(contract.get("forbidden_files", source_task.forbidden_files)),
                forbidden_output_substrings=list(
                    contract.get("forbidden_output_substrings", source_task.forbidden_output_substrings)
                ),
                expected_file_contents=dict(contract.get("expected_file_contents", source_task.expected_file_contents)),
                max_steps=source_task.max_steps,
                metadata=metadata,
            ))
        )
        if limit is not None and len(tasks) >= limit:
            break
    return tasks


def load_tool_replay_tasks(tool_candidates_path: Path, *, limit: int | None = None) -> list[TaskSpec]:
    if not tool_candidates_path.exists():
        return []
    bank = TaskBank()
    tasks: list[TaskSpec] = []
    payload = json.loads(tool_candidates_path.read_text(encoding="utf-8"))
    if not _artifact_payload_in_states(payload, {"replay_verified", "retained"}):
        return []
    records = payload.get("candidates", payload) if isinstance(payload, dict) else payload
    for record in records:
        promotion_stage = str(record.get("promotion_stage", "")).strip()
        record_lifecycle_state = str(record.get("lifecycle_state", "")).strip()
        if record_lifecycle_state == "rejected":
            continue
        if promotion_stage and promotion_stage not in {"replay_verified", "promoted_tool"}:
            continue
        if promotion_stage == "replay_verified" and record_lifecycle_state not in {"", "replay_verified"}:
            continue
        if promotion_stage == "promoted_tool" and record_lifecycle_state not in {"", "retained"}:
            continue
        source_task_id = str(record.get("source_task_id", "")).strip()
        if not source_task_id:
            continue
        contract = _task_contract_from_memory(record, task_id=source_task_id, bank=bank)
        if contract is None:
            continue
        procedure = list(record.get("procedure", {}).get("commands", []))
        if _tool_candidate_has_incomplete_shared_repo_integrator_bundle(record, contract=contract, procedure=procedure):
            continue
        metadata = dict(contract.metadata)
        metadata.update(
            _memory_task_metadata(
                "tool_replay",
                source_task_id=source_task_id,
                origin_benchmark_family=str(contract.metadata.get("benchmark_family", "bounded")),
            )
        )
        replay_task = TaskSpec(
            task_id=f"{source_task_id}{_memory_task_rule_text('tool_replay', 'task_id_suffix', fallback='_tool_replay')}",
            prompt=_memory_task_prompt("tool_replay", prompt=contract.prompt),
            workspace_subdir=(
                f"{source_task_id}{_memory_task_rule_text('tool_replay', 'workspace_suffix', fallback='_tool_replay')}"
            ),
            setup_commands=list(contract.setup_commands),
            success_command=contract.success_command,
            suggested_commands=procedure or list(contract.suggested_commands),
            expected_files=list(contract.expected_files),
            expected_output_substrings=list(contract.expected_output_substrings),
            forbidden_files=list(contract.forbidden_files),
            forbidden_output_substrings=list(contract.forbidden_output_substrings),
            expected_file_contents=dict(contract.expected_file_contents),
            max_steps=max(contract.max_steps, max(1, len(procedure) + 1)),
            metadata=metadata,
        )
        tasks.append(_annotate_light_supervision_contract(replay_task))
        if limit is not None and len(tasks) >= limit:
            break
    return tasks


def _retention_decision_state(payload: dict[str, object]) -> str:
    decision = payload.get("retention_decision", {})
    if not isinstance(decision, dict):
        return ""
    return str(decision.get("state", "")).strip()


def _artifact_payload_in_states(payload: object, allowed_states: set[str]) -> bool:
    if not isinstance(payload, dict):
        return True
    if _retention_decision_state(payload) == "reject":
        return False
    lifecycle_state = str(payload.get("lifecycle_state", "")).strip()
    if not lifecycle_state:
        return False
    return lifecycle_state in allowed_states


def instantiate_operator_commands(operator: dict[str, object], task: TaskSpec) -> list[str]:
    operator_kind = str(operator.get("operator_kind", "shell_procedure")).strip()
    template = operator.get("template_procedure", {})
    template_commands = []
    if isinstance(template, dict):
        template_commands = [str(command) for command in template.get("commands", []) if str(command).strip()]
    template_contract = operator.get("template_contract", {})
    commands = _instantiate_template_commands(
        template_commands,
        template_contract if isinstance(template_contract, dict) else {},
        task,
    )
    if commands:
        return commands
    commands = []
    expected_dirs = sorted(
        {str(Path(path).parent) for path in task.expected_files if str(Path(path).parent) not in {"", "."}}
    )
    if expected_dirs:
        commands.append(f"mkdir -p {' '.join(expected_dirs)}")
    if operator_kind in {"cleanup_write", "multi_emit", "single_emit", "rename"}:
        for path in task.forbidden_files:
            commands.append(f"rm -f {path}")
        for path, content in task.expected_file_contents.items():
            escaped = content.replace("\\", "\\\\").replace("'", "'\"'\"'")
            commands.append(f"printf '{escaped}' > {path}")
        for path in task.expected_files:
            if path not in task.expected_file_contents:
                commands.append(f": > {path}")
    return commands


def _find_transfer_target(
    bank: TaskBank,
    *,
    source_task_id: str,
    capability: str,
    benchmark_family: str,
    excluded_task_ids: set[str] | None = None,
) -> TaskSpec | None:
    excluded = set(excluded_task_ids or ())
    if source_task_id:
        excluded.add(source_task_id)
    for task in bank.list():
        if task.task_id in excluded:
            continue
        if str(task.metadata.get("capability", "")) != capability:
            continue
        if str(task.metadata.get("benchmark_family", "")) != benchmark_family:
            continue
        return task
    return None


def _find_operator_target(
    bank: TaskBank,
    *,
    source_task_ids: list[str],
    capabilities: list[str],
    benchmark_families: list[str],
) -> TaskSpec | None:
    capability_set = set(capabilities)
    family_set = set(benchmark_families)
    source_set = set(source_task_ids)
    for task in bank.list():
        if task.task_id in source_set:
            continue
        if capability_set and str(task.metadata.get("capability", "")) not in capability_set:
            continue
        if family_set and str(task.metadata.get("benchmark_family", "")) not in family_set:
            continue
        return task
    return None


def build_shared_transfer_target_maps(
    skills_path: Path,
    operator_classes_path: Path,
) -> tuple[dict[str, str], dict[str, str]]:
    bank = TaskBank()
    skill_targets: dict[str, str] = {}
    operator_targets: dict[str, str] = {}
    exclusions_by_class: dict[tuple[str, str], set[str]] = {}

    if skills_path.exists():
        payload = json.loads(skills_path.read_text(encoding="utf-8"))
        if not _artifact_payload_in_states(payload, {"promoted", "retained"}):
            payload = {"skills": []}
        skills = payload.get("skills", payload) if isinstance(payload, dict) else payload
        for skill in skills:
            if not isinstance(skill, dict):
                continue
            lifecycle_state = str(skill.get("lifecycle_state", "")).strip()
            if lifecycle_state == "rejected":
                continue
            decision_state = str((skill.get("retention_decision") or {}).get("state", "")).strip()
            if decision_state == "reject":
                continue
            source_task_id = str(skill.get("source_task_id", skill.get("task_id", ""))).strip()
            if not source_task_id:
                continue
            try:
                source_task = bank.get(source_task_id)
            except KeyError:
                continue
            key = (
                str(source_task.metadata.get("capability", "unknown")),
                str(source_task.metadata.get("benchmark_family", "bounded")),
            )
            exclusions_by_class.setdefault(key, set()).add(source_task_id)

    if operator_classes_path.exists():
        payload = json.loads(operator_classes_path.read_text(encoding="utf-8"))
        if not _artifact_payload_in_states(payload, {"promoted", "retained"}):
            payload = {"operators": []}
        operators = payload.get("operators", payload) if isinstance(payload, dict) else payload
        for operator in operators:
            if not isinstance(operator, dict):
                continue
            lifecycle_state = str(operator.get("lifecycle_state", "")).strip()
            if lifecycle_state == "rejected":
                continue
            decision_state = str((operator.get("retention_decision") or {}).get("state", "")).strip()
            if decision_state == "reject":
                continue
            capabilities = sorted(str(value) for value in operator.get("applicable_capabilities", []) if str(value).strip())
            families = sorted(
                str(value) for value in operator.get("applicable_benchmark_families", []) if str(value).strip()
            )
            if not capabilities or not families:
                continue
            key = (capabilities[0], families[0])
            exclusions_by_class.setdefault(key, set()).update(
                str(value) for value in operator.get("source_task_ids", []) if str(value).strip()
            )

    target_by_class: dict[tuple[str, str], str] = {}
    for key, excluded in exclusions_by_class.items():
        capability, benchmark_family = key
        target = _find_transfer_target(
            bank,
            source_task_id="",
            capability=capability,
            benchmark_family=benchmark_family,
            excluded_task_ids=excluded,
        )
        if target is not None:
            target_by_class[key] = target.task_id

    if skills_path.exists():
        payload = json.loads(skills_path.read_text(encoding="utf-8"))
        if not _artifact_payload_in_states(payload, {"promoted", "retained"}):
            payload = {"skills": []}
        skills = payload.get("skills", payload) if isinstance(payload, dict) else payload
        for skill in skills:
            if not isinstance(skill, dict):
                continue
            lifecycle_state = str(skill.get("lifecycle_state", "")).strip()
            if lifecycle_state == "rejected":
                continue
            decision_state = str((skill.get("retention_decision") or {}).get("state", "")).strip()
            if decision_state == "reject":
                continue
            source_task_id = str(skill.get("source_task_id", skill.get("task_id", ""))).strip()
            if not source_task_id:
                continue
            try:
                source_task = bank.get(source_task_id)
            except KeyError:
                continue
            key = (
                str(source_task.metadata.get("capability", "unknown")),
                str(source_task.metadata.get("benchmark_family", "bounded")),
            )
            if key in target_by_class:
                skill_targets[source_task_id] = target_by_class[key]

    if operator_classes_path.exists():
        payload = json.loads(operator_classes_path.read_text(encoding="utf-8"))
        if not _artifact_payload_in_states(payload, {"promoted", "retained"}):
            payload = {"operators": []}
        operators = payload.get("operators", payload) if isinstance(payload, dict) else payload
        for operator in operators:
            if not isinstance(operator, dict):
                continue
            lifecycle_state = str(operator.get("lifecycle_state", "")).strip()
            if lifecycle_state == "rejected":
                continue
            decision_state = str((operator.get("retention_decision") or {}).get("state", "")).strip()
            if decision_state == "reject":
                continue
            capabilities = sorted(str(value) for value in operator.get("applicable_capabilities", []) if str(value).strip())
            families = sorted(
                str(value) for value in operator.get("applicable_benchmark_families", []) if str(value).strip()
            )
            if not capabilities or not families:
                continue
            key = (capabilities[0], families[0])
            operator_id = str(operator.get("operator_id", "")).strip()
            if operator_id and key in target_by_class:
                operator_targets[operator_id] = target_by_class[key]

    return skill_targets, operator_targets


def _resolve_transfer_target(
    bank: TaskBank,
    *,
    target_task_by_source: dict[str, str] | None,
    source_task_id: str,
    capability: str,
    benchmark_family: str,
) -> TaskSpec | None:
    if target_task_by_source and source_task_id in target_task_by_source:
        try:
            return bank.get(target_task_by_source[source_task_id])
        except KeyError:
            return None
    return _find_transfer_target(
        bank,
        source_task_id=source_task_id,
        capability=capability,
        benchmark_family=benchmark_family,
    )


def _resolve_operator_target(
    bank: TaskBank,
    *,
    target_task_by_operator: dict[str, str] | None,
    operator_id: str,
    source_task_ids: list[str],
    capabilities: list[str],
    benchmark_families: list[str],
) -> TaskSpec | None:
    if target_task_by_operator and operator_id in target_task_by_operator:
        try:
            return bank.get(target_task_by_operator[operator_id])
        except KeyError:
            return None
    return _find_operator_target(
        bank,
        source_task_ids=source_task_ids,
        capabilities=capabilities,
        benchmark_families=benchmark_families,
    )


def _instantiate_template_commands(
    template_commands: list[str],
    template_contract: dict[str, object],
    task: TaskSpec,
) -> list[str]:
    if not template_commands:
        return []
    replacements = _template_replacements(template_contract, task)
    commands = list(template_commands)
    for source, target in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        commands = [command.replace(source, target) for command in commands]
    return commands


def _template_replacements(template_contract: dict[str, object], task: TaskSpec) -> dict[str, str]:
    replacements: dict[str, str] = {}
    source_expected_files = [str(path) for path in template_contract.get("expected_files", [])]
    source_forbidden_files = [str(path) for path in template_contract.get("forbidden_files", [])]
    source_expected_contents = {
        str(path): str(content)
        for path, content in dict(template_contract.get("expected_file_contents", {})).items()
    }
    for source_path, target_path in zip(sorted(source_expected_files), sorted(task.expected_files)):
        replacements[source_path] = target_path
        if source_path in source_expected_contents and target_path in task.expected_file_contents:
            source_content = source_expected_contents[source_path]
            target_content = task.expected_file_contents[target_path]
            replacements[source_content] = target_content
            replacements[_shell_escaped_content(source_content)] = _shell_escaped_content(target_content)
    for source_path, target_path in zip(sorted(source_forbidden_files), sorted(task.forbidden_files)):
        replacements[source_path] = target_path
    return replacements


def _shell_escaped_content(content: str) -> str:
    return str(content).replace("\\", "\\\\").replace("\n", "\\n")


def _task_contract_from_memory(
    payload: dict[str, object],
    *,
    task_id: str,
    bank: TaskBank,
) -> TaskSpec | None:
    contract = payload.get("task_contract")
    task_metadata = payload.get("task_metadata", {})
    if not isinstance(task_metadata, dict):
        task_metadata = {}
    if isinstance(contract, dict) and contract:
        try:
            fallback = bank.get(task_id)
            fallback_metadata = dict(fallback.metadata)
        except KeyError:
            fallback_metadata = {}
            fallback = None
        metadata = dict(fallback_metadata)
        if isinstance(contract.get("metadata", {}), dict):
            metadata.update(dict(contract.get("metadata", {})))

        capability = str(metadata.get("capability", "")).strip() or str(
            task_metadata.get("capability", "")
        ).strip() or str(fallback_metadata.get("capability", "unknown")).strip()
        difficulty = str(metadata.get("difficulty", "")).strip() or str(
            task_metadata.get("difficulty", "")
        ).strip() or str(fallback_metadata.get("difficulty", "unknown")).strip()
        benchmark_family = str(metadata.get("benchmark_family", "")).strip() or str(
            task_metadata.get("benchmark_family", "")
        ).strip() or str(fallback_metadata.get("benchmark_family", "bounded")).strip()

        metadata["capability"] = capability
        metadata["difficulty"] = difficulty
        metadata["benchmark_family"] = benchmark_family
        task = TaskSpec(
            task_id=task_id,
            prompt=str(contract.get("prompt", payload.get("prompt", getattr(fallback, "prompt", "")))),
            workspace_subdir=str(contract.get("workspace_subdir", getattr(fallback, "workspace_subdir", task_id))),
            setup_commands=list(contract.get("setup_commands", getattr(fallback, "setup_commands", []))),
            success_command=str(contract.get("success_command", getattr(fallback, "success_command", ""))),
            suggested_commands=list(contract.get("suggested_commands", getattr(fallback, "suggested_commands", []))),
            expected_files=list(contract.get("expected_files", getattr(fallback, "expected_files", []))),
            expected_output_substrings=list(
                contract.get("expected_output_substrings", getattr(fallback, "expected_output_substrings", []))
            ),
            forbidden_files=list(contract.get("forbidden_files", getattr(fallback, "forbidden_files", []))),
            forbidden_output_substrings=list(
                contract.get("forbidden_output_substrings", getattr(fallback, "forbidden_output_substrings", []))
            ),
            expected_file_contents=dict(
                contract.get("expected_file_contents", getattr(fallback, "expected_file_contents", {}))
            ),
            max_steps=int(contract.get("max_steps", getattr(fallback, "max_steps", 5)) or 5),
            metadata=metadata,
        )
        task.max_steps = uplifted_task_max_steps(
            task.max_steps,
            metadata=task.metadata,
            suggested_commands=task.suggested_commands,
        )
        return _annotate_light_supervision_contract(task)
    try:
        return bank.get(task_id)
    except KeyError:
        return None


def _tool_candidate_has_incomplete_shared_repo_integrator_bundle(
    record: dict[str, object],
    *,
    contract: TaskSpec,
    procedure: list[str],
) -> bool:
    bundle = record.get("shared_repo_bundle", {})
    if isinstance(bundle, dict) and str(bundle.get("role", "")).strip() == "integrator":
        return not bool(bundle.get("bundle_complete", False))
    metadata = dict(contract.metadata)
    try:
        shared_repo_order = int(metadata.get("shared_repo_order", 0) or 0)
    except (TypeError, ValueError):
        shared_repo_order = 0
    verifier = dict(metadata.get("semantic_verifier", {})) if isinstance(metadata.get("semantic_verifier", {}), dict) else {}
    required_merged_branches = [
        str(value).strip()
        for value in verifier.get("required_merged_branches", [])
        if str(value).strip()
    ]
    if shared_repo_order <= 0 or not required_merged_branches:
        return False
    observed_merged_branches = {
        branch for branch in required_merged_branches if any(branch in str(command) for command in procedure)
    }
    return len(observed_merged_branches) < len(required_merged_branches)
