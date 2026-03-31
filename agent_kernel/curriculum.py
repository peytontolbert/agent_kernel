from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from .config import KernelConfig, current_external_task_manifests_paths
from .improvement_common import retained_artifact_payload
from .memory import EpisodeMemory
from .schemas import EpisodeRecord, TaskSpec
from .task_bank import TaskBank


class CurriculumEngine:
    def __init__(self, memory_root: Path | None = None, config: KernelConfig | None = None) -> None:
        self.memory = EpisodeMemory(memory_root) if memory_root is not None else None
        self.config = config or KernelConfig()
        self._curriculum_controls_cache: dict[str, object] | None = None

    def generate_followup_task(self, episode: EpisodeRecord) -> TaskSpec:
        if episode.success:
            task = self.generate_adjacent_task(episode)
        else:
            task = self.generate_failure_driven_task(episode)
        return self._apply_curriculum_controls(self._with_curriculum_hint(task))

    def schedule_generated_seed_episodes(
        self,
        episodes: list[EpisodeRecord],
        *,
        curriculum_kind: str,
    ) -> list[EpisodeRecord]:
        controls = self._curriculum_controls()
        if not controls:
            return list(episodes)
        success_only = curriculum_kind == "adjacent_success"
        preferred_family = str(controls.get("preferred_benchmark_family", "")).strip()
        ranked: list[tuple[int, str, EpisodeRecord]] = []
        for episode in episodes:
            family = self._episode_benchmark_family(episode)
            score = 0
            if preferred_family and family == preferred_family:
                score += 4
            if success_only:
                if episode.success:
                    score += 2
            else:
                if not episode.success:
                    score += 3
                failure_types = self._failure_types(episode)
                score += min(2, len(failure_types))
                if str(episode.termination_reason).strip() == "repeated_failed_action":
                    score += 1
            ranked.append((score, episode.task_id, episode))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        limit = self._max_generated_seed_tasks(curriculum_kind)
        selected = [episode for _, _, episode in ranked]
        if limit > 0:
            selected = selected[:limit]
        return selected

    def generate_adjacent_task(self, episode: EpisodeRecord) -> TaskSpec:
        retrieval_context = self._retrieve_context(episode, success_only=True)
        metadata = {
            "parent_task": episode.task_id,
            "curriculum_kind": "adjacent_success",
            **retrieval_context["metadata"],
        }
        benchmark_family = str(retrieval_context["metadata"].get("benchmark_family", "bounded"))
        suggested_commands = self._merged_commands(
            [
                "printf 'hello completed\n' > status.txt",
                "cat status.txt",
            ],
            retrieval_context["successful_commands"],
        )
        if benchmark_family == "project":
            prompt = (
                f"Create a project handoff for {episode.task_id} by writing project/summary.txt containing "
                f"{episode.task_id} project handoff ready, project/status.txt containing project verified, "
                "and project/check.txt containing handoff complete."
            )
            if retrieval_context["summary"]:
                prompt += f" Reuse the validated pattern from {retrieval_context['summary']}."
            return TaskSpec(
                task_id=f"{episode.task_id}_project_adjacent",
                prompt=prompt,
                workspace_subdir=f"{episode.task_id}_project_adjacent",
                suggested_commands=self._merged_commands(
                    [
                        f"mkdir -p project && printf '{episode.task_id} project handoff ready\n' > project/summary.txt && printf 'project verified\n' > project/status.txt && printf 'handoff complete\n' > project/check.txt",
                        "cat project/summary.txt",
                        "cat project/status.txt",
                        "cat project/check.txt",
                    ],
                    retrieval_context["successful_commands"],
                ),
                success_command=(
                    f"test -f project/summary.txt && grep -q '^{episode.task_id} project handoff ready$' project/summary.txt && "
                    "test -f project/status.txt && grep -q '^project verified$' project/status.txt && "
                    "test -f project/check.txt && grep -q '^handoff complete$' project/check.txt"
                ),
                expected_files=["project/summary.txt", "project/status.txt", "project/check.txt"],
                expected_file_contents={
                    "project/summary.txt": f"{episode.task_id} project handoff ready\n",
                    "project/status.txt": "project verified\n",
                    "project/check.txt": "handoff complete\n",
                },
                metadata=metadata,
            )
        if benchmark_family == "repository":
            prompt = (
                f"Create a repository handoff for {episode.task_id} by writing repo/summary.txt containing "
                f"{episode.task_id} repository handoff ready, repo/status.txt containing repo verified, and "
                "repo/check.txt containing repository complete."
            )
            if retrieval_context["summary"]:
                prompt += f" Reuse the validated pattern from {retrieval_context['summary']}."
            return TaskSpec(
                task_id=f"{episode.task_id}_repository_adjacent",
                prompt=prompt,
                workspace_subdir=f"{episode.task_id}_repository_adjacent",
                suggested_commands=self._merged_commands(
                    [
                        f"mkdir -p repo && printf '{episode.task_id} repository handoff ready\n' > repo/summary.txt && printf 'repo verified\n' > repo/status.txt && printf 'repository complete\n' > repo/check.txt",
                        "cat repo/summary.txt",
                        "cat repo/status.txt",
                        "cat repo/check.txt",
                    ],
                    retrieval_context["successful_commands"],
                ),
                success_command=(
                    f"test -f repo/summary.txt && grep -q '^{episode.task_id} repository handoff ready$' repo/summary.txt && "
                    "test -f repo/status.txt && grep -q '^repo verified$' repo/status.txt && "
                    "test -f repo/check.txt && grep -q '^repository complete$' repo/check.txt"
                ),
                expected_files=["repo/summary.txt", "repo/status.txt", "repo/check.txt"],
                expected_file_contents={
                    "repo/summary.txt": f"{episode.task_id} repository handoff ready\n",
                    "repo/status.txt": "repo verified\n",
                    "repo/check.txt": "repository complete\n",
                },
                metadata=metadata,
            )
        if benchmark_family == "tooling":
            prompt = (
                f"Create a tool handoff for {episode.task_id} by writing tool/summary.txt containing "
                f"{episode.task_id} tool handoff ready, tool/status.txt containing tool verified, and "
                "tool/check.txt containing tool exchange complete."
            )
            if retrieval_context["summary"]:
                prompt += f" Reuse the validated pattern from {retrieval_context['summary']}."
            return TaskSpec(
                task_id=f"{episode.task_id}_tool_adjacent",
                prompt=prompt,
                workspace_subdir=f"{episode.task_id}_tool_adjacent",
                suggested_commands=self._merged_commands(
                    [
                        f"mkdir -p tool && printf '{episode.task_id} tool handoff ready\n' > tool/summary.txt && printf 'tool verified\n' > tool/status.txt && printf 'tool exchange complete\n' > tool/check.txt",
                        "cat tool/summary.txt",
                        "cat tool/status.txt",
                        "cat tool/check.txt",
                    ],
                    retrieval_context["successful_commands"],
                ),
                success_command=(
                    f"test -f tool/summary.txt && grep -q '^{episode.task_id} tool handoff ready$' tool/summary.txt && "
                    "test -f tool/status.txt && grep -q '^tool verified$' tool/status.txt && "
                    "test -f tool/check.txt && grep -q '^tool exchange complete$' tool/check.txt"
                ),
                expected_files=["tool/summary.txt", "tool/status.txt", "tool/check.txt"],
                expected_file_contents={
                    "tool/summary.txt": f"{episode.task_id} tool handoff ready\n",
                    "tool/status.txt": "tool verified\n",
                    "tool/check.txt": "tool exchange complete\n",
                },
                metadata=metadata,
            )
        if benchmark_family == "integration":
            prompt = (
                f"Create an integration handoff for {episode.task_id} by writing integration/summary.txt containing "
                f"{episode.task_id} integration handoff ready, integration/status.txt containing integration verified, "
                "and integration/check.txt containing integration complete."
            )
            if retrieval_context["summary"]:
                prompt += f" Reuse the validated pattern from {retrieval_context['summary']}."
            return TaskSpec(
                task_id=f"{episode.task_id}_integration_adjacent",
                prompt=prompt,
                workspace_subdir=f"{episode.task_id}_integration_adjacent",
                suggested_commands=self._merged_commands(
                    [
                        f"mkdir -p integration && printf '{episode.task_id} integration handoff ready\n' > integration/summary.txt && printf 'integration verified\n' > integration/status.txt && printf 'integration complete\n' > integration/check.txt",
                        "cat integration/summary.txt",
                        "cat integration/status.txt",
                        "cat integration/check.txt",
                    ],
                    retrieval_context["successful_commands"],
                ),
                success_command=(
                    f"test -f integration/summary.txt && grep -q '^{episode.task_id} integration handoff ready$' integration/summary.txt && "
                    "test -f integration/status.txt && grep -q '^integration verified$' integration/status.txt && "
                    "test -f integration/check.txt && grep -q '^integration complete$' integration/check.txt"
                ),
                expected_files=["integration/summary.txt", "integration/status.txt", "integration/check.txt"],
                expected_file_contents={
                    "integration/summary.txt": f"{episode.task_id} integration handoff ready\n",
                    "integration/status.txt": "integration verified\n",
                    "integration/check.txt": "integration complete\n",
                },
                metadata=metadata,
            )
        if benchmark_family == "workflow":
            prompt = (
                f"Create a workflow audit for {episode.task_id} by writing audit/summary.txt containing "
                f"{episode.task_id} workflow verified and audit/status.txt containing workflow complete."
            )
            if retrieval_context["summary"]:
                prompt += f" Reuse the validated pattern from {retrieval_context['summary']}."
            return TaskSpec(
                task_id=f"{episode.task_id}_workflow_adjacent",
                prompt=prompt,
                workspace_subdir=f"{episode.task_id}_workflow_adjacent",
                suggested_commands=self._merged_commands(
                    [
                        f"mkdir -p audit && printf '{episode.task_id} workflow verified\n' > audit/summary.txt && printf 'workflow complete\n' > audit/status.txt",
                        "cat audit/summary.txt",
                        "cat audit/status.txt",
                    ],
                    retrieval_context["successful_commands"],
                ),
                success_command=(
                    f"test -f audit/summary.txt && grep -q '^{episode.task_id} workflow verified$' audit/summary.txt && "
                    "test -f audit/status.txt && grep -q '^workflow complete$' audit/status.txt"
                ),
                expected_files=["audit/summary.txt", "audit/status.txt"],
                expected_file_contents={
                    "audit/summary.txt": f"{episode.task_id} workflow verified\n",
                    "audit/status.txt": "workflow complete\n",
                },
                metadata=metadata,
            )
        if episode.task_id == "hello_task":
            prompt = "Create status.txt containing the string hello completed."
            if retrieval_context["summary"]:
                prompt += f" Reuse the validated pattern from {retrieval_context['summary']}."
            return TaskSpec(
                task_id="hello_task_followup",
                prompt=prompt,
                workspace_subdir="hello_task_followup",
                suggested_commands=suggested_commands,
                success_command="test -f status.txt && grep -q 'hello completed' status.txt",
                expected_files=["status.txt"],
                metadata=metadata,
            )

        prompt = f"Create summary.txt stating that {episode.task_id} succeeded."
        if retrieval_context["summary"]:
            prompt += f" Reuse the validated pattern from {retrieval_context['summary']}."
        return TaskSpec(
            task_id=f"{episode.task_id}_adjacent",
            prompt=prompt,
            workspace_subdir=f"{episode.task_id}_adjacent",
            suggested_commands=self._merged_commands(
                [
                    f"printf '{episode.task_id} succeeded\n' > summary.txt",
                    "cat summary.txt",
                ],
                retrieval_context["successful_commands"],
            ),
            success_command=f"test -f summary.txt && grep -q '{episode.task_id} succeeded' summary.txt",
            expected_files=["summary.txt"],
            metadata=metadata,
        )

    def generate_failure_driven_task(self, episode: EpisodeRecord) -> TaskSpec:
        failure_types = self._failure_types(episode)
        latest_command = self._latest_command(episode)
        retrieval_context = self._retrieve_context(episode, success_only=False)
        metadata = {
            "parent_task": episode.task_id,
            "source_task": str(episode.task_metadata.get("source_task", episode.task_id)),
            "curriculum_kind": "failure_recovery",
            "failure_types": failure_types,
            "failure_pattern": self._failure_pattern(episode, latest_command),
            "failed_command": latest_command,
            **retrieval_context["metadata"],
        }
        benchmark_family = str(retrieval_context["metadata"].get("benchmark_family", "bounded"))
        prompt_suffix = ""
        if retrieval_context["summary"]:
            prompt_suffix = f" Use the validated pattern from {retrieval_context['summary']}."
        if retrieval_context["avoidance_note"]:
            prompt_suffix += f" Avoid {retrieval_context['avoidance_note']}."
        if latest_command:
            prompt_suffix += f" Do not repeat the failed command shape {latest_command!r}."

        failure_pattern = str(metadata["failure_pattern"])

        if failure_pattern == "workspace_prefixed_path":
            fallback_commands = [
                "printf 'path recovery complete\n' > resolved.txt",
                "cat resolved.txt",
            ]
            nested_path = f"{episode.task_id}/resolved.txt"
            return TaskSpec(
                task_id=f"{episode.task_id}_path_recovery",
                prompt=(
                    "Recover from a workspace-path mistake by creating resolved.txt containing path recovery complete "
                    "directly in the current workspace, not inside a nested task directory."
                    f"{prompt_suffix}"
                ),
                workspace_subdir=f"{episode.task_id}_path_recovery",
                suggested_commands=self._failure_recovery_commands(
                    fallback_commands,
                    retrieval_context["successful_commands"],
                    expected_files=["resolved.txt"],
                    failed_command=latest_command,
                ),
                success_command="test -f resolved.txt && grep -q '^path recovery complete$' resolved.txt",
                expected_files=["resolved.txt"],
                forbidden_files=[nested_path],
                expected_file_contents={"resolved.txt": "path recovery complete\n"},
                metadata=metadata,
            )

        if benchmark_family == "project":
            fallback_commands = [
                "mkdir -p project && printf 'project recovery complete\n' > project/recovery.txt && printf 'recovery verified\n' > project/check.txt && printf 'project recovered\n' > project/status.txt",
                "cat project/recovery.txt",
                "cat project/check.txt",
                "cat project/status.txt",
            ]
            return TaskSpec(
                task_id=f"{episode.task_id}_project_recovery",
                prompt=(
                    "Recover the project workspace by creating project/recovery.txt containing project recovery complete, "
                    "project/check.txt containing recovery verified, and project/status.txt containing project recovered."
                    f"{prompt_suffix}"
                ),
                workspace_subdir=f"{episode.task_id}_project_recovery",
                suggested_commands=self._failure_recovery_commands(
                    fallback_commands,
                    retrieval_context["successful_commands"],
                    expected_files=["project/recovery.txt", "project/check.txt", "project/status.txt"],
                    failed_command=latest_command,
                ),
                success_command=(
                    "test -f project/recovery.txt && grep -q '^project recovery complete$' project/recovery.txt && "
                    "test -f project/check.txt && grep -q '^recovery verified$' project/check.txt && "
                    "test -f project/status.txt && grep -q '^project recovered$' project/status.txt"
                ),
                expected_files=["project/recovery.txt", "project/check.txt", "project/status.txt"],
                expected_file_contents={
                    "project/recovery.txt": "project recovery complete\n",
                    "project/check.txt": "recovery verified\n",
                    "project/status.txt": "project recovered\n",
                },
                metadata=metadata,
            )
        if benchmark_family == "repository":
            fallback_commands = [
                "mkdir -p repo && printf 'repository recovery complete\n' > repo/recovery.txt && printf 'repo recovery verified\n' > repo/check.txt && printf 'repository recovered\n' > repo/status.txt",
                "cat repo/recovery.txt",
                "cat repo/check.txt",
                "cat repo/status.txt",
            ]
            return TaskSpec(
                task_id=f"{episode.task_id}_repository_recovery",
                prompt=(
                    "Recover the repository workspace by creating repo/recovery.txt containing repository recovery complete, "
                    "repo/check.txt containing repo recovery verified, and repo/status.txt containing repository recovered."
                    f"{prompt_suffix}"
                ),
                workspace_subdir=f"{episode.task_id}_repository_recovery",
                suggested_commands=self._failure_recovery_commands(
                    fallback_commands,
                    retrieval_context["successful_commands"],
                    expected_files=["repo/recovery.txt", "repo/check.txt", "repo/status.txt"],
                    failed_command=latest_command,
                ),
                success_command=(
                    "test -f repo/recovery.txt && grep -q '^repository recovery complete$' repo/recovery.txt && "
                    "test -f repo/check.txt && grep -q '^repo recovery verified$' repo/check.txt && "
                    "test -f repo/status.txt && grep -q '^repository recovered$' repo/status.txt"
                ),
                expected_files=["repo/recovery.txt", "repo/check.txt", "repo/status.txt"],
                expected_file_contents={
                    "repo/recovery.txt": "repository recovery complete\n",
                    "repo/check.txt": "repo recovery verified\n",
                    "repo/status.txt": "repository recovered\n",
                },
                metadata=metadata,
            )
        if benchmark_family == "tooling":
            fallback_commands = [
                "mkdir -p tool && printf 'tool recovery complete\n' > tool/recovery.txt && printf 'tool recovery verified\n' > tool/check.txt && printf 'tool recovered\n' > tool/status.txt",
                "cat tool/recovery.txt",
                "cat tool/check.txt",
                "cat tool/status.txt",
            ]
            return TaskSpec(
                task_id=f"{episode.task_id}_tool_recovery",
                prompt=(
                    "Recover the tool workspace by creating tool/recovery.txt containing tool recovery complete, "
                    "tool/check.txt containing tool recovery verified, and tool/status.txt containing tool recovered."
                    f"{prompt_suffix}"
                ),
                workspace_subdir=f"{episode.task_id}_tool_recovery",
                suggested_commands=self._failure_recovery_commands(
                    fallback_commands,
                    retrieval_context["successful_commands"],
                    expected_files=["tool/recovery.txt", "tool/check.txt", "tool/status.txt"],
                    failed_command=latest_command,
                ),
                success_command=(
                    "test -f tool/recovery.txt && grep -q '^tool recovery complete$' tool/recovery.txt && "
                    "test -f tool/check.txt && grep -q '^tool recovery verified$' tool/check.txt && "
                    "test -f tool/status.txt && grep -q '^tool recovered$' tool/status.txt"
                ),
                expected_files=["tool/recovery.txt", "tool/check.txt", "tool/status.txt"],
                expected_file_contents={
                    "tool/recovery.txt": "tool recovery complete\n",
                    "tool/check.txt": "tool recovery verified\n",
                    "tool/status.txt": "tool recovered\n",
                },
                metadata=metadata,
            )
        if benchmark_family == "integration":
            fallback_commands = [
                "mkdir -p integration && printf 'integration recovery complete\n' > integration/recovery.txt && printf 'integration recovery verified\n' > integration/check.txt && printf 'integration recovered\n' > integration/status.txt",
                "cat integration/recovery.txt",
                "cat integration/check.txt",
                "cat integration/status.txt",
            ]
            return TaskSpec(
                task_id=f"{episode.task_id}_integration_recovery",
                prompt=(
                    "Recover the integration workspace by creating integration/recovery.txt containing integration recovery complete, "
                    "integration/check.txt containing integration recovery verified, and integration/status.txt containing integration recovered."
                    f"{prompt_suffix}"
                ),
                workspace_subdir=f"{episode.task_id}_integration_recovery",
                suggested_commands=self._failure_recovery_commands(
                    fallback_commands,
                    retrieval_context["successful_commands"],
                    expected_files=["integration/recovery.txt", "integration/check.txt", "integration/status.txt"],
                    failed_command=latest_command,
                ),
                success_command=(
                    "test -f integration/recovery.txt && grep -q '^integration recovery complete$' integration/recovery.txt && "
                    "test -f integration/check.txt && grep -q '^integration recovery verified$' integration/check.txt && "
                    "test -f integration/status.txt && grep -q '^integration recovered$' integration/status.txt"
                ),
                expected_files=["integration/recovery.txt", "integration/check.txt", "integration/status.txt"],
                expected_file_contents={
                    "integration/recovery.txt": "integration recovery complete\n",
                    "integration/check.txt": "integration recovery verified\n",
                    "integration/status.txt": "integration recovered\n",
                },
                metadata=metadata,
            )

        if benchmark_family == "workflow":
            fallback_commands = [
                "mkdir -p recovery && printf 'workflow recovery complete\n' > recovery/summary.txt && printf 'recovered\n' > recovery/status.txt",
                "cat recovery/summary.txt",
                "cat recovery/status.txt",
            ]
            return TaskSpec(
                task_id=f"{episode.task_id}_workflow_recovery",
                prompt=(
                    "Recover the workflow by creating recovery/summary.txt containing workflow recovery complete "
                    "and recovery/status.txt containing recovered, without repeating the failed command shape."
                    f"{prompt_suffix}"
                ),
                workspace_subdir=f"{episode.task_id}_workflow_recovery",
                suggested_commands=self._failure_recovery_commands(
                    fallback_commands,
                    retrieval_context["successful_commands"],
                    expected_files=["recovery/summary.txt", "recovery/status.txt"],
                    failed_command=latest_command,
                ),
                success_command=(
                    "test -f recovery/summary.txt && grep -q '^workflow recovery complete$' recovery/summary.txt && "
                    "test -f recovery/status.txt && grep -q '^recovered$' recovery/status.txt"
                ),
                expected_files=["recovery/summary.txt", "recovery/status.txt"],
                expected_file_contents={
                    "recovery/summary.txt": "workflow recovery complete\n",
                    "recovery/status.txt": "recovered\n",
                },
                metadata=metadata,
            )

        if "missing_expected_file" in failure_types:
            fallback_commands = [
                "printf 'file recovery complete\n' > recovery.txt",
                "cat recovery.txt",
            ]
            return TaskSpec(
                task_id=f"{episode.task_id}_file_recovery",
                prompt=(
                    "Recover from a missing-file failure by creating recovery.txt "
                    "containing file recovery complete."
                    f"{prompt_suffix}"
                ),
                workspace_subdir=f"{episode.task_id}_file_recovery",
                suggested_commands=self._failure_recovery_commands(
                    fallback_commands,
                    retrieval_context["successful_commands"],
                    expected_files=["recovery.txt"],
                    failed_command=latest_command,
                ),
                success_command="test -f recovery.txt && grep -q '^file recovery complete$' recovery.txt",
                expected_files=["recovery.txt"],
                metadata=metadata,
            )

        if "command_failure" in failure_types and latest_command:
            fallback_commands = [
                "printf 'command recovery complete\n' > recovery.txt",
                "cat recovery.txt",
            ]
            return TaskSpec(
                task_id=f"{episode.task_id}_avoidance_recovery",
                prompt=(
                    "Recover from a failed command pattern by creating recovery.txt containing command recovery complete "
                    "without repeating the failed command structure."
                    f"{prompt_suffix}"
                ),
                workspace_subdir=f"{episode.task_id}_avoidance_recovery",
                suggested_commands=self._failure_recovery_commands(
                    fallback_commands,
                    retrieval_context["successful_commands"],
                    expected_files=["recovery.txt"],
                    failed_command=latest_command,
                ),
                success_command="test -f recovery.txt && grep -q '^command recovery complete$' recovery.txt",
                expected_files=["recovery.txt"],
                forbidden_output_substrings=[latest_command],
                expected_file_contents={"recovery.txt": "command recovery complete\n"},
                metadata=metadata,
            )

        fallback_commands = [
            "printf 'safe retry complete\n' > retry.txt",
            "cat retry.txt",
        ]
        return TaskSpec(
            task_id=f"{episode.task_id}_safe_retry",
            prompt=(
                "Recover from a failed command sequence by creating retry.txt "
                "containing safe retry complete."
                f"{prompt_suffix}"
            ),
            workspace_subdir=f"{episode.task_id}_safe_retry",
            suggested_commands=self._failure_recovery_commands(
                fallback_commands,
                retrieval_context["successful_commands"],
                expected_files=["retry.txt"],
                failed_command=latest_command,
            ),
            success_command="test -f retry.txt && grep -q '^safe retry complete$' retry.txt",
            expected_files=["retry.txt"],
            metadata=metadata,
        )

    def _retrieve_context(self, episode: EpisodeRecord, *, success_only: bool) -> dict[str, object]:
        documents = self._memory_documents()
        if not documents:
            return {
                "successful_commands": [],
                "summary": "",
                "avoidance_note": "",
                "metadata": {
                    "reference_task_ids": [],
                    "reference_commands": [],
                    "retrieved_failure_types": [],
                    "retrieved_transition_failures": [],
                },
            }

        target_failure_types = set(self._failure_types(episode))
        controls = self._curriculum_controls()
        preferred_family = str(controls.get("preferred_benchmark_family", "")).strip()
        family_only = bool(controls.get("failure_reference_family_only", False)) and not success_only
        episode_family = str(episode.task_metadata.get("benchmark_family", "")).strip()
        ranked: list[tuple[int, dict]] = []
        for document in documents:
            task_id = str(document.get("task_id", ""))
            if task_id == episode.task_id:
                continue
            document_family = str(
                document.get("task_metadata", {}).get(
                    "benchmark_family",
                    document.get("metadata", {}).get("benchmark_family", ""),
                )
            ).strip()
            if family_only and episode_family and document_family and document_family != episode_family:
                continue
            score = 0
            if document.get("success"):
                score += 4
            if preferred_family and document_family == preferred_family:
                score += 3
            if episode_family and document_family == episode_family:
                score += 2
            if task_id.startswith(f"{episode.task_id}_") or episode.task_id.startswith(f"{task_id}_"):
                score += 3
            summary = document.get("summary", {})
            doc_failure_types = set(summary.get("failure_types", [])) | set(summary.get("transition_failures", []))
            if target_failure_types and doc_failure_types.intersection(target_failure_types):
                score += 3
            if score:
                ranked.append((score, document))

        ranked.sort(key=lambda item: (-item[0], str(item[1].get("task_id", ""))))
        selected = [
            document
            for _, document in ranked
            if not success_only or document.get("success")
        ][:3]

        successful_commands: list[str] = []
        retrieved_failure_types: set[str] = set()
        retrieved_transition_failures: set[str] = set()
        for document in selected:
            summary = document.get("summary", {})
            retrieved_failure_types.update(summary.get("failure_types", []))
            retrieved_transition_failures.update(summary.get("transition_failures", []))
            for command in self._reference_commands_for_document(document):
                command_text = str(command).strip()
                if command_text and command_text not in successful_commands:
                    successful_commands.append(command_text)

        summary_label = ", ".join(str(document.get("task_id", "")) for document in selected[:2])
        avoidance_note = ", ".join(
            sorted(target_failure_types.intersection(retrieved_failure_types | retrieved_transition_failures))
        )
        return {
            "successful_commands": successful_commands[: self._success_reference_limit()],
            "summary": summary_label,
            "avoidance_note": avoidance_note,
            "metadata": {
                "reference_task_ids": [str(document.get("task_id", "")) for document in selected],
                "reference_commands": successful_commands[: self._success_reference_limit()],
                "retrieved_failure_types": sorted(retrieved_failure_types),
                "retrieved_transition_failures": sorted(retrieved_transition_failures),
                "benchmark_family": self._benchmark_family(episode.task_id, selected),
            },
        }

    @staticmethod
    def _reference_commands_for_document(document: dict[str, object]) -> list[str]:
        if not bool(document.get("success", False)):
            return []
        fragments = document.get("fragments", [])
        commands: list[str] = []
        if isinstance(fragments, list):
            for fragment in fragments:
                if not isinstance(fragment, dict):
                    continue
                if fragment.get("kind") != "command" or not fragment.get("passed", False):
                    continue
                command = str(fragment.get("command", "")).strip()
                if command and command not in commands:
                    commands.append(command)
        if commands:
            return commands
        summary = document.get("summary", {})
        if not isinstance(summary, dict):
            return []
        return [str(command).strip() for command in summary.get("executed_commands", []) if str(command).strip()]

    def _memory_documents(self) -> list[dict]:
        if self.memory is None:
            return []
        return self.memory.list_documents()

    @staticmethod
    def _merged_commands(primary: list[str], fallback: list[str]) -> list[str]:
        merged: list[str] = []
        for command in [*primary, *fallback]:
            command_text = str(command).strip()
            if command_text and command_text not in merged:
                merged.append(command_text)
        return merged

    @classmethod
    def _failure_recovery_commands(
        cls,
        fallback: list[str],
        retrieved: list[str],
        *,
        expected_files: list[str],
        failed_command: str,
    ) -> list[str]:
        anchors = cls._command_anchors(expected_files)
        normalized_failed = " ".join(str(failed_command).strip().split())
        aligned_retrieved: list[str] = []
        for command in retrieved:
            command_text = str(command).strip()
            normalized = " ".join(command_text.split())
            if not command_text or normalized == normalized_failed:
                continue
            if anchors and not any(anchor in command_text for anchor in anchors):
                continue
            if command_text not in aligned_retrieved:
                aligned_retrieved.append(command_text)
        return cls._merged_commands(fallback, aligned_retrieved)

    @staticmethod
    def _command_anchors(expected_files: list[str]) -> list[str]:
        anchors: list[str] = []
        for path in expected_files:
            normalized = str(path).strip()
            if not normalized:
                continue
            parts = [part for part in Path(normalized).parts if part not in {".", ""}]
            for part in parts:
                if part not in anchors:
                    anchors.append(part)
        return anchors

    def _with_curriculum_hint(self, task: TaskSpec) -> TaskSpec:
        if not self.config.use_curriculum_proposals:
            return task
        path = self.config.curriculum_proposals_path
        if not path.exists():
            return task
        payload = json.loads(path.read_text(encoding="utf-8"))
        retained = retained_artifact_payload(payload, artifact_kind="curriculum_proposal_set")
        if retained is None:
            return task
        proposals = retained.get("proposals", [])
        if not isinstance(proposals, list):
            return task
        family = str(task.metadata.get("benchmark_family", "bounded"))
        hint = ""
        for proposal in proposals:
            if not isinstance(proposal, dict):
                continue
            suggestion = str(proposal.get("suggestion", "")).strip()
            reason = str(proposal.get("reason", "")).strip()
            if family in reason or proposal.get("area") in {"failure_recovery", "benchmark_family"}:
                hint = suggestion
                break
        if not hint:
            return task
        metadata = dict(task.metadata)
        metadata["curriculum_proposal_hint"] = hint
        return replace(task, prompt=f"{task.prompt} Curriculum guidance: {hint}", metadata=metadata)

    def _apply_curriculum_controls(self, task: TaskSpec) -> TaskSpec:
        controls = self._curriculum_controls()
        if not controls:
            return task
        suggested_commands = [str(command).strip() for command in task.suggested_commands if str(command).strip()]
        if str(task.metadata.get("curriculum_kind", "")).strip() != "failure_recovery":
            capped = self._cap_commands(
                suggested_commands,
                self._adjacent_reference_limit(),
            )
            if capped == task.suggested_commands:
                return task
            metadata = dict(task.metadata)
            metadata["curriculum_behavior_controls"] = dict(controls)
            return replace(task, suggested_commands=capped, metadata=metadata)
        min_anchor_matches = self._failure_recovery_anchor_min_matches()
        filtered_commands: list[str] = []
        if min_anchor_matches > 1:
            anchors = self._command_anchors([*task.expected_files, *task.expected_file_contents.keys()])
            for index, command_text in enumerate(suggested_commands):
                anchor_matches = sum(1 for anchor in anchors if anchor in command_text)
                if index == 0 or anchor_matches >= min_anchor_matches:
                    filtered_commands.append(command_text)
        else:
            filtered_commands = list(suggested_commands)
        if not filtered_commands:
            filtered_commands = list(suggested_commands)
        filtered_commands = self._cap_commands(
            filtered_commands,
            self._failure_recovery_command_cap(),
        )
        metadata = dict(task.metadata)
        metadata["curriculum_behavior_controls"] = dict(controls)
        return replace(task, suggested_commands=filtered_commands, metadata=metadata)

    def _curriculum_controls(self) -> dict[str, object]:
        if self._curriculum_controls_cache is not None:
            return self._curriculum_controls_cache
        if not self.config.use_curriculum_proposals:
            self._curriculum_controls_cache = {}
            return self._curriculum_controls_cache
        path = self.config.curriculum_proposals_path
        if not path.exists():
            self._curriculum_controls_cache = {}
            return self._curriculum_controls_cache
        payload = json.loads(path.read_text(encoding="utf-8"))
        retained = retained_artifact_payload(payload, artifact_kind="curriculum_proposal_set")
        if retained is None:
            self._curriculum_controls_cache = {}
            return self._curriculum_controls_cache
        controls = retained.get("controls", {})
        self._curriculum_controls_cache = dict(controls) if isinstance(controls, dict) else {}
        return self._curriculum_controls_cache

    def _max_generated_seed_tasks(self, curriculum_kind: str) -> int:
        field = "max_generated_failure_recovery_tasks" if curriculum_kind == "failure_recovery" else "max_generated_adjacent_tasks"
        value = self._curriculum_controls().get(field, 0)
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 0

    def _success_reference_limit(self) -> int:
        value = self._curriculum_controls().get("success_reference_limit", 3)
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return 3

    def _adjacent_reference_limit(self) -> int:
        value = self._curriculum_controls().get("adjacent_reference_limit", self._success_reference_limit())
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return self._success_reference_limit()

    def _failure_recovery_anchor_min_matches(self) -> int:
        value = self._curriculum_controls().get("failure_recovery_anchor_min_matches", 1)
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return 1

    def _failure_recovery_command_cap(self) -> int:
        value = self._curriculum_controls().get("failure_recovery_command_cap", 4)
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return 4

    @staticmethod
    def _episode_benchmark_family(episode: EpisodeRecord) -> str:
        return str(episode.task_metadata.get("benchmark_family", "bounded")).strip() or "bounded"

    @staticmethod
    def _cap_commands(commands: list[str], limit: int) -> list[str]:
        limited: list[str] = []
        for command in commands:
            command_text = str(command).strip()
            if command_text and command_text not in limited:
                limited.append(command_text)
            if len(limited) >= max(1, limit):
                break
        return limited

    @staticmethod
    def _failure_types(episode: EpisodeRecord) -> list[str]:
        failure_types: set[str] = set()
        for step in episode.steps:
            for reason in step.verification.get("reasons", []):
                lowered = reason.lower()
                if "missing expected file" in lowered:
                    failure_types.add("missing_expected_file")
                elif "forbidden file present" in lowered:
                    failure_types.add("forbidden_file_present")
                elif "unexpected file content" in lowered:
                    failure_types.add("unexpected_file_content")
                elif "forbidden output present" in lowered:
                    failure_types.add("forbidden_output_present")
                elif "exit code" in lowered:
                    failure_types.add("command_failure")
                elif "timed out" in lowered:
                    failure_types.add("timeout")
                elif "repeated failed action" in lowered:
                    failure_types.add("repeated_failed_action")
            for signal in step.failure_signals:
                normalized = str(signal).strip()
                if normalized:
                    failure_types.add(normalized)
        if episode.termination_reason:
            failure_types.add(episode.termination_reason)
        return sorted(failure_types)

    @staticmethod
    def _latest_command(episode: EpisodeRecord) -> str:
        for step in reversed(episode.steps):
            if step.action == "code_execute" and step.content:
                return step.content
        return ""

    @staticmethod
    def _failure_pattern(episode: EpisodeRecord, latest_command: str) -> str:
        workspace_name = Path(episode.workspace).name.strip()
        if workspace_name and f"{workspace_name}/" in latest_command:
            return "workspace_prefixed_path"
        return "generic_recovery"

    @staticmethod
    def _benchmark_family(task_id: str, documents: list[dict]) -> str:
        for document in documents:
            task_metadata = document.get("task_metadata", {})
            family = str(task_metadata.get("benchmark_family", "")).strip()
            if family:
                return family
            metadata = document.get("metadata", {})
            family = str(metadata.get("benchmark_family", "")).strip()
            if family:
                return family
        try:
            try:
                manifest_paths = current_external_task_manifests_paths()
                bank = TaskBank(
                    config=KernelConfig(),
                    external_task_manifests=manifest_paths if manifest_paths else None,
                )
            except TypeError:
                bank = TaskBank()
            return str(bank.get(task_id).metadata.get("benchmark_family", "bounded"))
        except KeyError:
            return "bounded"
