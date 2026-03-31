from __future__ import annotations

from copy import deepcopy
import glob
import json
from pathlib import Path
import re
import shlex

from .config import KernelConfig
from .episode_store import iter_episode_documents
from .schemas import TaskSpec
from .verifier import synthesize_stricter_task


class TaskBank:
    def __init__(
        self,
        config: KernelConfig | None = None,
        external_task_manifests: tuple[str, ...] | None = None,
    ) -> None:
        self._tasks = {
            "hello_task": self._task(
                task_id="hello_task",
                prompt="Create hello.txt containing the string hello agent kernel.",
                workspace_subdir="hello_task",
                suggested_commands=[
                    "printf 'hello agent kernel\n' > hello.txt",
                    "cat hello.txt",
                ],
                success_command="test -f hello.txt && grep -q 'hello agent kernel' hello.txt",
                expected_files=["hello.txt"],
                expected_file_contents={"hello.txt": "hello agent kernel\n"},
                capability="file_write",
                difficulty="seed",
            ),
            "math_task": self._task(
                task_id="math_task",
                prompt="Create result.txt containing the number 42.",
                workspace_subdir="math_task",
                suggested_commands=[
                    "printf '42\n' > result.txt",
                    "cat result.txt",
                ],
                success_command="test -f result.txt && grep -q '^42$' result.txt",
                expected_files=["result.txt"],
                expected_file_contents={"result.txt": "42\n"},
                capability="file_write",
                difficulty="seed",
            ),
            "nested_file_task": self._task(
                task_id="nested_file_task",
                prompt="Create reports/status.txt containing the string ready.",
                workspace_subdir="nested_file_task",
                suggested_commands=[
                    "mkdir -p reports && printf 'ready\n' > reports/status.txt",
                    "cat reports/status.txt",
                ],
                success_command="test -f reports/status.txt && grep -q '^ready$' reports/status.txt",
                expected_files=["reports/status.txt"],
                expected_file_contents={"reports/status.txt": "ready\n"},
                capability="nested_filesystem",
                difficulty="bounded",
            ),
            "rename_task": self._task(
                task_id="rename_task",
                prompt="Rename draft.txt to final.txt and keep the existing contents.",
                workspace_subdir="rename_task",
                setup_commands=["printf 'renamed content\n' > draft.txt"],
                suggested_commands=[
                    "mv draft.txt final.txt",
                    "cat final.txt",
                ],
                success_command="test -f final.txt && grep -q '^renamed content$' final.txt",
                expected_files=["final.txt"],
                forbidden_files=["draft.txt"],
                expected_file_contents={"final.txt": "renamed content\n"},
                capability="filesystem_mutation",
                difficulty="bounded",
            ),
            "rewrite_task": self._task(
                task_id="rewrite_task",
                prompt="Overwrite note.txt so it contains only the string done.",
                workspace_subdir="rewrite_task",
                setup_commands=["printf 'todo\n' > note.txt"],
                suggested_commands=[
                    "printf 'done\n' > note.txt",
                    "cat note.txt",
                ],
                success_command="test -f note.txt && grep -q '^done$' note.txt",
                expected_files=["note.txt"],
                expected_file_contents={"note.txt": "done\n"},
                capability="file_edit",
                difficulty="bounded",
            ),
            "cleanup_task": self._task(
                task_id="cleanup_task",
                prompt="Remove temp.txt and create status.txt containing cleaned.",
                workspace_subdir="cleanup_task",
                setup_commands=["printf 'temporary\n' > temp.txt"],
                suggested_commands=[
                    "rm -f temp.txt && printf 'cleaned\n' > status.txt",
                    "cat status.txt",
                ],
                success_command="test ! -f temp.txt && test -f status.txt && grep -q '^cleaned$' status.txt",
                expected_files=["status.txt"],
                forbidden_files=["temp.txt"],
                expected_file_contents={"status.txt": "cleaned\n"},
                capability="cleanup",
                difficulty="bounded",
                metadata={"benchmark_family": "micro"},
            ),
            "release_bundle_task": self._task(
                task_id="release_bundle_task",
                prompt=(
                    "Prepare a release bundle: remove incoming/draft.txt, create release/notes.txt "
                    "containing shipped release, and create release/status.txt containing packaged."
                ),
                workspace_subdir="release_bundle_task",
                setup_commands=[
                    "mkdir -p incoming && printf 'draft release\\n' > incoming/draft.txt",
                ],
                suggested_commands=[
                    "mkdir -p release && rm -f incoming/draft.txt && printf 'shipped release\n' > release/notes.txt && printf 'packaged\n' > release/status.txt",
                    "cat release/notes.txt",
                    "cat release/status.txt",
                ],
                success_command=(
                    "test ! -f incoming/draft.txt && "
                    "test -f release/notes.txt && grep -q '^shipped release$' release/notes.txt && "
                    "test -f release/status.txt && grep -q '^packaged$' release/status.txt"
                ),
                expected_files=["release/notes.txt", "release/status.txt"],
                forbidden_files=["incoming/draft.txt"],
                expected_file_contents={
                    "release/notes.txt": "shipped release\n",
                    "release/status.txt": "packaged\n",
                },
                capability="workflow_environment",
                difficulty="environment",
                metadata={"benchmark_family": "workflow"},
            ),
            "config_sync_task": self._task(
                task_id="config_sync_task",
                prompt=(
                    "Synchronize the config workspace: create config/app.env containing MODE=prod "
                    "and PORT=8080, while preserving template.env."
                ),
                workspace_subdir="config_sync_task",
                setup_commands=["printf 'MODE=dev\nPORT=3000\n' > template.env"],
                suggested_commands=[
                    "mkdir -p config && printf 'MODE=prod\nPORT=8080\n' > config/app.env",
                    "cat config/app.env",
                ],
                success_command=(
                    "test -f template.env && "
                    "test -f config/app.env && grep -q '^MODE=prod$' config/app.env && "
                    "grep -q '^PORT=8080$' config/app.env"
                ),
                expected_files=["template.env", "config/app.env"],
                expected_file_contents={
                    "template.env": "MODE=dev\nPORT=3000\n",
                    "config/app.env": "MODE=prod\nPORT=8080\n",
                },
                capability="workflow_environment",
                difficulty="environment",
                metadata={"benchmark_family": "workflow"},
            ),
            "release_bundle_retrieval_task": self._task(
                task_id="release_bundle_retrieval_task",
                prompt=(
                    "Reproduce the established release workflow used elsewhere in this repo: remove the draft input, "
                    "write the shipped release notes, and mark the release as packaged."
                ),
                workspace_subdir="release_bundle_retrieval_task",
                setup_commands=[
                    "mkdir -p incoming && printf 'draft release\\n' > incoming/draft.txt",
                ],
                suggested_commands=[],
                success_command=(
                    "test ! -f incoming/draft.txt && "
                    "test -f release/notes.txt && grep -q '^shipped release$' release/notes.txt && "
                    "test -f release/status.txt && grep -q '^packaged$' release/status.txt"
                ),
                expected_files=["release/notes.txt", "release/status.txt"],
                forbidden_files=["incoming/draft.txt"],
                expected_file_contents={
                    "release/notes.txt": "shipped release\n",
                    "release/status.txt": "packaged\n",
                },
                capability="workflow_environment",
                difficulty="retrieval",
                metadata={
                    "benchmark_family": "workflow",
                    "requires_retrieval": True,
                    "source_task": "release_bundle_task",
                },
            ),
            "config_sync_retrieval_task": self._task(
                task_id="config_sync_retrieval_task",
                prompt=(
                    "Reproduce the canonical config synchronization procedure from earlier repo tasks: preserve "
                    "template.env and create the production config file under config/."
                ),
                workspace_subdir="config_sync_retrieval_task",
                setup_commands=["printf 'MODE=dev\nPORT=3000\n' > template.env"],
                suggested_commands=[],
                success_command=(
                    "test -f template.env && "
                    "test -f config/app.env && grep -q '^MODE=prod$' config/app.env && "
                    "grep -q '^PORT=8080$' config/app.env"
                ),
                expected_files=["template.env", "config/app.env"],
                expected_file_contents={
                    "template.env": "MODE=dev\nPORT=3000\n",
                    "config/app.env": "MODE=prod\nPORT=8080\n",
                },
                capability="workflow_environment",
                difficulty="retrieval",
                metadata={
                    "benchmark_family": "workflow",
                    "requires_retrieval": True,
                    "source_task": "config_sync_task",
                },
            ),
            "deployment_manifest_task": self._task(
                task_id="deployment_manifest_task",
                prompt=(
                    "Prepare the deployment workspace: remove staging/draft.txt, preserve config/base.env, "
                    "create deploy/manifest.txt containing deployment manifest ready, and create "
                    "deploy/checklist.txt containing deployment checklist complete."
                ),
                workspace_subdir="deployment_manifest_task",
                setup_commands=[
                    "mkdir -p staging config && printf 'draft deployment\\n' > staging/draft.txt && printf 'ENV=base\\n' > config/base.env",
                ],
                suggested_commands=[
                    "mkdir -p deploy && rm -f staging/draft.txt && printf 'deployment manifest ready\n' > deploy/manifest.txt && printf 'deployment checklist complete\n' > deploy/checklist.txt",
                    "cat deploy/manifest.txt",
                    "cat deploy/checklist.txt",
                ],
                success_command=(
                    "test ! -f staging/draft.txt && "
                    "test -f config/base.env && grep -q '^ENV=base$' config/base.env && "
                    "test -f deploy/manifest.txt && grep -q '^deployment manifest ready$' deploy/manifest.txt && "
                    "test -f deploy/checklist.txt && grep -q '^deployment checklist complete$' deploy/checklist.txt"
                ),
                expected_files=["config/base.env", "deploy/manifest.txt", "deploy/checklist.txt"],
                forbidden_files=["staging/draft.txt"],
                expected_file_contents={
                    "config/base.env": "ENV=base\n",
                    "deploy/manifest.txt": "deployment manifest ready\n",
                    "deploy/checklist.txt": "deployment checklist complete\n",
                },
                capability="project_environment",
                difficulty="long_horizon",
                metadata={"benchmark_family": "project"},
            ),
            "deployment_manifest_retrieval_task": self._task(
                task_id="deployment_manifest_retrieval_task",
                prompt=(
                    "Reproduce the canonical deployment manifest procedure used elsewhere in this repo: "
                    "preserve the base config, remove the staging draft, and produce the manifest and checklist."
                ),
                workspace_subdir="deployment_manifest_retrieval_task",
                setup_commands=[
                    "mkdir -p staging config && printf 'draft deployment\\n' > staging/draft.txt && printf 'ENV=base\\n' > config/base.env",
                ],
                suggested_commands=[],
                success_command=(
                    "test ! -f staging/draft.txt && "
                    "test -f config/base.env && grep -q '^ENV=base$' config/base.env && "
                    "test -f deploy/manifest.txt && grep -q '^deployment manifest ready$' deploy/manifest.txt && "
                    "test -f deploy/checklist.txt && grep -q '^deployment checklist complete$' deploy/checklist.txt"
                ),
                expected_files=["config/base.env", "deploy/manifest.txt", "deploy/checklist.txt"],
                forbidden_files=["staging/draft.txt"],
                expected_file_contents={
                    "config/base.env": "ENV=base\n",
                    "deploy/manifest.txt": "deployment manifest ready\n",
                    "deploy/checklist.txt": "deployment checklist complete\n",
                },
                capability="project_environment",
                difficulty="long_horizon",
                metadata={
                    "benchmark_family": "project",
                    "requires_retrieval": True,
                    "source_task": "deployment_manifest_task",
                },
            ),
            "report_rollup_task": self._task(
                task_id="report_rollup_task",
                prompt=(
                    "Prepare the report rollup: preserve inbox/day1.log and inbox/day2.log, create "
                    "reports/summary.txt containing rollup complete, and create reports/index.txt containing "
                    "2 sources processed."
                ),
                workspace_subdir="report_rollup_task",
                setup_commands=[
                    "mkdir -p inbox && printf 'source day1\\n' > inbox/day1.log && printf 'source day2\\n' > inbox/day2.log",
                ],
                suggested_commands=[
                    "mkdir -p reports && printf 'rollup complete\n' > reports/summary.txt && printf '2 sources processed\n' > reports/index.txt",
                    "cat reports/summary.txt",
                    "cat reports/index.txt",
                ],
                success_command=(
                    "test -f inbox/day1.log && grep -q '^source day1$' inbox/day1.log && "
                    "test -f inbox/day2.log && grep -q '^source day2$' inbox/day2.log && "
                    "test -f reports/summary.txt && grep -q '^rollup complete$' reports/summary.txt && "
                    "test -f reports/index.txt && grep -q '^2 sources processed$' reports/index.txt"
                ),
                expected_files=["inbox/day1.log", "inbox/day2.log", "reports/summary.txt", "reports/index.txt"],
                expected_file_contents={
                    "inbox/day1.log": "source day1\n",
                    "inbox/day2.log": "source day2\n",
                    "reports/summary.txt": "rollup complete\n",
                    "reports/index.txt": "2 sources processed\n",
                },
                capability="project_environment",
                difficulty="long_horizon",
                metadata={"benchmark_family": "project"},
            ),
            "report_rollup_retrieval_task": self._task(
                task_id="report_rollup_retrieval_task",
                prompt=(
                    "Reproduce the canonical report rollup procedure from earlier repo tasks: preserve both inbox "
                    "source logs and create the summary and index artifacts."
                ),
                workspace_subdir="report_rollup_retrieval_task",
                setup_commands=[
                    "mkdir -p inbox && printf 'source day1\\n' > inbox/day1.log && printf 'source day2\\n' > inbox/day2.log",
                ],
                suggested_commands=[],
                success_command=(
                    "test -f inbox/day1.log && grep -q '^source day1$' inbox/day1.log && "
                    "test -f inbox/day2.log && grep -q '^source day2$' inbox/day2.log && "
                    "test -f reports/summary.txt && grep -q '^rollup complete$' reports/summary.txt && "
                    "test -f reports/index.txt && grep -q '^2 sources processed$' reports/index.txt"
                ),
                expected_files=["inbox/day1.log", "inbox/day2.log", "reports/summary.txt", "reports/index.txt"],
                expected_file_contents={
                    "inbox/day1.log": "source day1\n",
                    "inbox/day2.log": "source day2\n",
                    "reports/summary.txt": "rollup complete\n",
                    "reports/index.txt": "2 sources processed\n",
                },
                capability="project_environment",
                difficulty="long_horizon",
                metadata={
                    "benchmark_family": "project",
                    "requires_retrieval": True,
                    "source_task": "report_rollup_task",
                },
            ),
            "release_packet_task": self._task(
                task_id="release_packet_task",
                prompt=(
                    "Prepare the project release packet: preserve notes/brief.txt, remove drafts/old.txt, create "
                    "plan/timeline.txt containing milestone freeze ready, create plan/owners.txt containing owner "
                    "delivery, and create reports/packet.txt containing release packet assembled."
                ),
                workspace_subdir="release_packet_task",
                setup_commands=[
                    "mkdir -p notes drafts && printf 'brief baseline\\n' > notes/brief.txt && printf 'outdated draft\\n' > drafts/old.txt",
                ],
                suggested_commands=[
                    "mkdir -p plan reports && rm -f drafts/old.txt && printf 'milestone freeze ready\n' > plan/timeline.txt && printf 'owner delivery\n' > plan/owners.txt && printf 'release packet assembled\n' > reports/packet.txt",
                    "cat plan/timeline.txt",
                    "cat plan/owners.txt",
                    "cat reports/packet.txt",
                ],
                success_command=(
                    "test -f notes/brief.txt && grep -q '^brief baseline$' notes/brief.txt && "
                    "test ! -f drafts/old.txt && "
                    "test -f plan/timeline.txt && grep -q '^milestone freeze ready$' plan/timeline.txt && "
                    "test -f plan/owners.txt && grep -q '^owner delivery$' plan/owners.txt && "
                    "test -f reports/packet.txt && grep -q '^release packet assembled$' reports/packet.txt"
                ),
                expected_files=["notes/brief.txt", "plan/timeline.txt", "plan/owners.txt", "reports/packet.txt"],
                forbidden_files=["drafts/old.txt"],
                expected_file_contents={
                    "notes/brief.txt": "brief baseline\n",
                    "plan/timeline.txt": "milestone freeze ready\n",
                    "plan/owners.txt": "owner delivery\n",
                    "reports/packet.txt": "release packet assembled\n",
                },
                capability="project_environment",
                difficulty="long_horizon",
                metadata={"benchmark_family": "project"},
            ),
            "release_packet_retrieval_task": self._task(
                task_id="release_packet_retrieval_task",
                prompt=(
                    "Reproduce the canonical project release packet from earlier tasks: preserve the brief, remove "
                    "the outdated draft, build the timeline and owner plan, and emit the packet report."
                ),
                workspace_subdir="release_packet_retrieval_task",
                setup_commands=[
                    "mkdir -p notes drafts && printf 'brief baseline\\n' > notes/brief.txt && printf 'outdated draft\\n' > drafts/old.txt",
                ],
                suggested_commands=[],
                success_command=(
                    "test -f notes/brief.txt && grep -q '^brief baseline$' notes/brief.txt && "
                    "test ! -f drafts/old.txt && "
                    "test -f plan/timeline.txt && grep -q '^milestone freeze ready$' plan/timeline.txt && "
                    "test -f plan/owners.txt && grep -q '^owner delivery$' plan/owners.txt && "
                    "test -f reports/packet.txt && grep -q '^release packet assembled$' reports/packet.txt"
                ),
                expected_files=["notes/brief.txt", "plan/timeline.txt", "plan/owners.txt", "reports/packet.txt"],
                forbidden_files=["drafts/old.txt"],
                expected_file_contents={
                    "notes/brief.txt": "brief baseline\n",
                    "plan/timeline.txt": "milestone freeze ready\n",
                    "plan/owners.txt": "owner delivery\n",
                    "reports/packet.txt": "release packet assembled\n",
                },
                capability="project_environment",
                difficulty="long_horizon",
                metadata={
                    "benchmark_family": "project",
                    "requires_retrieval": True,
                    "source_task": "release_packet_task",
                    "distractor_tasks": ["deployment_manifest_task", "report_rollup_task"],
                },
            ),
            "service_release_task": self._task(
                task_id="service_release_task",
                prompt=(
                    "Prepare the service release repo slice: preserve docs/overview.md, create app/release.txt "
                    "containing service release ready, create config/runtime.env containing MODE=release, "
                    "and create tests/smoke.txt containing smoke ready."
                ),
                workspace_subdir="service_release_task",
                setup_commands=[
                    "mkdir -p docs && printf 'service overview\\n' > docs/overview.md",
                ],
                suggested_commands=[
                    "mkdir -p app config tests && printf 'service release ready\n' > app/release.txt && printf 'MODE=release\n' > config/runtime.env && printf 'smoke ready\n' > tests/smoke.txt",
                    "cat app/release.txt",
                    "cat config/runtime.env",
                    "cat tests/smoke.txt",
                ],
                success_command=(
                    "test -f docs/overview.md && grep -q '^service overview$' docs/overview.md && "
                    "test -f app/release.txt && grep -q '^service release ready$' app/release.txt && "
                    "test -f config/runtime.env && grep -q '^MODE=release$' config/runtime.env && "
                    "test -f tests/smoke.txt && grep -q '^smoke ready$' tests/smoke.txt"
                ),
                expected_files=["docs/overview.md", "app/release.txt", "config/runtime.env", "tests/smoke.txt"],
                expected_file_contents={
                    "docs/overview.md": "service overview\n",
                    "app/release.txt": "service release ready\n",
                    "config/runtime.env": "MODE=release\n",
                    "tests/smoke.txt": "smoke ready\n",
                },
                capability="repo_environment",
                difficulty="cross_component",
                metadata={"benchmark_family": "repository"},
            ),
            "service_release_retrieval_task": self._task(
                task_id="service_release_retrieval_task",
                prompt=(
                    "Reproduce the canonical service release repo procedure from earlier tasks: preserve the "
                    "overview doc, create the release artifact, set release mode, and add the smoke marker."
                ),
                workspace_subdir="service_release_retrieval_task",
                setup_commands=[
                    "mkdir -p docs && printf 'service overview\\n' > docs/overview.md",
                ],
                suggested_commands=[],
                success_command=(
                    "test -f docs/overview.md && grep -q '^service overview$' docs/overview.md && "
                    "test -f app/release.txt && grep -q '^service release ready$' app/release.txt && "
                    "test -f config/runtime.env && grep -q '^MODE=release$' config/runtime.env && "
                    "test -f tests/smoke.txt && grep -q '^smoke ready$' tests/smoke.txt"
                ),
                expected_files=["docs/overview.md", "app/release.txt", "config/runtime.env", "tests/smoke.txt"],
                expected_file_contents={
                    "docs/overview.md": "service overview\n",
                    "app/release.txt": "service release ready\n",
                    "config/runtime.env": "MODE=release\n",
                    "tests/smoke.txt": "smoke ready\n",
                },
                capability="repo_environment",
                difficulty="cross_component",
                metadata={
                    "benchmark_family": "repository",
                    "requires_retrieval": True,
                    "source_task": "service_release_task",
                },
            ),
            "schema_alignment_task": self._task(
                task_id="schema_alignment_task",
                prompt=(
                    "Align the schema repo slice: preserve specs/base.md, create src/schema.txt containing "
                    "schema aligned, create fixtures/example.json containing {\"status\": \"aligned\"}, and "
                    "create docs/changelog.txt containing schema update recorded."
                ),
                workspace_subdir="schema_alignment_task",
                setup_commands=[
                    "mkdir -p specs && printf 'base schema spec\\n' > specs/base.md",
                ],
                suggested_commands=[
                    "mkdir -p src fixtures docs && printf 'schema aligned\n' > src/schema.txt && printf '{\"status\": \"aligned\"}\n' > fixtures/example.json && printf 'schema update recorded\n' > docs/changelog.txt",
                    "cat src/schema.txt",
                    "cat fixtures/example.json",
                    "cat docs/changelog.txt",
                ],
                success_command=(
                    "test -f specs/base.md && grep -q '^base schema spec$' specs/base.md && "
                    "test -f src/schema.txt && grep -q '^schema aligned$' src/schema.txt && "
                    "test -f fixtures/example.json && grep -q '^{\"status\": \"aligned\"}$' fixtures/example.json && "
                    "test -f docs/changelog.txt && grep -q '^schema update recorded$' docs/changelog.txt"
                ),
                expected_files=["specs/base.md", "src/schema.txt", "fixtures/example.json", "docs/changelog.txt"],
                expected_file_contents={
                    "specs/base.md": "base schema spec\n",
                    "src/schema.txt": "schema aligned\n",
                    "fixtures/example.json": "{\"status\": \"aligned\"}\n",
                    "docs/changelog.txt": "schema update recorded\n",
                },
                capability="repo_environment",
                difficulty="cross_component",
                metadata={"benchmark_family": "repository"},
            ),
            "schema_alignment_retrieval_task": self._task(
                task_id="schema_alignment_retrieval_task",
                prompt=(
                    "Reproduce the canonical schema alignment repo procedure from earlier tasks: preserve the "
                    "base spec, align the schema artifact, update the fixture, and record the changelog."
                ),
                workspace_subdir="schema_alignment_retrieval_task",
                setup_commands=[
                    "mkdir -p specs && printf 'base schema spec\\n' > specs/base.md",
                ],
                suggested_commands=[],
                success_command=(
                    "test -f specs/base.md && grep -q '^base schema spec$' specs/base.md && "
                    "test -f src/schema.txt && grep -q '^schema aligned$' src/schema.txt && "
                    "test -f fixtures/example.json && grep -q '^{\"status\": \"aligned\"}$' fixtures/example.json && "
                    "test -f docs/changelog.txt && grep -q '^schema update recorded$' docs/changelog.txt"
                ),
                expected_files=["specs/base.md", "src/schema.txt", "fixtures/example.json", "docs/changelog.txt"],
                expected_file_contents={
                    "specs/base.md": "base schema spec\n",
                    "src/schema.txt": "schema aligned\n",
                    "fixtures/example.json": "{\"status\": \"aligned\"}\n",
                    "docs/changelog.txt": "schema update recorded\n",
                },
                capability="repo_environment",
                difficulty="cross_component",
                metadata={
                    "benchmark_family": "repository",
                    "requires_retrieval": True,
                    "source_task": "schema_alignment_task",
                },
            ),
            "repo_sync_matrix_task": self._task(
                task_id="repo_sync_matrix_task",
                prompt=(
                    "Prepare the repository sync matrix: preserve docs/module_map.md, create src/runtime.txt "
                    "containing runtime synced, create tests/coverage.txt containing coverage expanded, create "
                    "config/deploy.env containing DEPLOY_MODE=sync, and create reports/matrix.txt containing "
                    "repository sync recorded."
                ),
                workspace_subdir="repo_sync_matrix_task",
                setup_commands=[
                    "mkdir -p docs && printf 'module map preserved\\n' > docs/module_map.md",
                ],
                suggested_commands=[
                    "mkdir -p src tests config reports && printf 'runtime synced\n' > src/runtime.txt && printf 'coverage expanded\n' > tests/coverage.txt && printf 'DEPLOY_MODE=sync\n' > config/deploy.env && printf 'repository sync recorded\n' > reports/matrix.txt",
                    "cat src/runtime.txt",
                    "cat tests/coverage.txt",
                    "cat config/deploy.env",
                    "cat reports/matrix.txt",
                ],
                success_command=(
                    "test -f docs/module_map.md && grep -q '^module map preserved$' docs/module_map.md && "
                    "test -f src/runtime.txt && grep -q '^runtime synced$' src/runtime.txt && "
                    "test -f tests/coverage.txt && grep -q '^coverage expanded$' tests/coverage.txt && "
                    "test -f config/deploy.env && grep -q '^DEPLOY_MODE=sync$' config/deploy.env && "
                    "test -f reports/matrix.txt && grep -q '^repository sync recorded$' reports/matrix.txt"
                ),
                expected_files=[
                    "docs/module_map.md",
                    "src/runtime.txt",
                    "tests/coverage.txt",
                    "config/deploy.env",
                    "reports/matrix.txt",
                ],
                expected_file_contents={
                    "docs/module_map.md": "module map preserved\n",
                    "src/runtime.txt": "runtime synced\n",
                    "tests/coverage.txt": "coverage expanded\n",
                    "config/deploy.env": "DEPLOY_MODE=sync\n",
                    "reports/matrix.txt": "repository sync recorded\n",
                },
                capability="repo_environment",
                difficulty="cross_component",
                metadata={"benchmark_family": "repository"},
            ),
            "repo_sync_matrix_retrieval_task": self._task(
                task_id="repo_sync_matrix_retrieval_task",
                prompt=(
                    "Reproduce the canonical repository sync matrix from earlier tasks: preserve the module map, "
                    "sync the runtime and deploy config, expand the coverage artifact, and emit the matrix report."
                ),
                workspace_subdir="repo_sync_matrix_retrieval_task",
                setup_commands=[
                    "mkdir -p docs && printf 'module map preserved\\n' > docs/module_map.md",
                ],
                suggested_commands=[],
                success_command=(
                    "test -f docs/module_map.md && grep -q '^module map preserved$' docs/module_map.md && "
                    "test -f src/runtime.txt && grep -q '^runtime synced$' src/runtime.txt && "
                    "test -f tests/coverage.txt && grep -q '^coverage expanded$' tests/coverage.txt && "
                    "test -f config/deploy.env && grep -q '^DEPLOY_MODE=sync$' config/deploy.env && "
                    "test -f reports/matrix.txt && grep -q '^repository sync recorded$' reports/matrix.txt"
                ),
                expected_files=[
                    "docs/module_map.md",
                    "src/runtime.txt",
                    "tests/coverage.txt",
                    "config/deploy.env",
                    "reports/matrix.txt",
                ],
                expected_file_contents={
                    "docs/module_map.md": "module map preserved\n",
                    "src/runtime.txt": "runtime synced\n",
                    "tests/coverage.txt": "coverage expanded\n",
                    "config/deploy.env": "DEPLOY_MODE=sync\n",
                    "reports/matrix.txt": "repository sync recorded\n",
                },
                capability="repo_environment",
                difficulty="cross_component",
                metadata={
                    "benchmark_family": "repository",
                    "requires_retrieval": True,
                    "source_task": "repo_sync_matrix_task",
                    "distractor_tasks": ["service_release_task", "schema_alignment_task"],
                },
            ),
            "repo_patch_review_task": self._task(
                task_id="repo_patch_review_task",
                prompt=(
                    "Prepare the repo patch review packet: preserve docs/context.md, update src/app.py to "
                    "contain STATUS=ready, update tests/status_check.txt to contain status ready covered, "
                    "create reports/diff_summary.txt containing src and tests updated, and create "
                    "reports/verification.txt containing deterministic checks passed."
                ),
                workspace_subdir="repo_patch_review_task",
                setup_commands=[
                    "mkdir -p docs src tests && printf 'repo context\\n' > docs/context.md && printf 'STATUS=pending\\n' > src/app.py && printf 'status pending\\n' > tests/status_check.txt",
                ],
                suggested_commands=[
                    "mkdir -p reports && printf 'STATUS=ready\n' > src/app.py && printf 'status ready covered\n' > tests/status_check.txt && printf 'updated src/app.py and tests/status_check.txt\n' > reports/diff_summary.txt && printf 'deterministic checks passed for docs/context.md preservation\n' > reports/verification.txt",
                    "cat src/app.py",
                    "cat tests/status_check.txt",
                    "cat reports/diff_summary.txt",
                    "cat reports/verification.txt",
                ],
                success_command=(
                    "test -f docs/context.md && grep -q '^repo context$' docs/context.md && "
                    "test -f src/app.py && grep -q '^STATUS=ready$' src/app.py && "
                    "test -f tests/status_check.txt && grep -q '^status ready covered$' tests/status_check.txt && "
                    "test -f reports/diff_summary.txt && grep -q '^updated src/app.py and tests/status_check.txt$' reports/diff_summary.txt && "
                    "test -f reports/verification.txt && grep -q '^deterministic checks passed for docs/context.md preservation$' reports/verification.txt"
                ),
                expected_files=[
                    "docs/context.md",
                    "src/app.py",
                    "tests/status_check.txt",
                    "reports/diff_summary.txt",
                    "reports/verification.txt",
                ],
                expected_file_contents={
                    "docs/context.md": "repo context\n",
                    "src/app.py": "STATUS=ready\n",
                    "tests/status_check.txt": "status ready covered\n",
                    "reports/diff_summary.txt": "updated src/app.py and tests/status_check.txt\n",
                    "reports/verification.txt": "deterministic checks passed for docs/context.md preservation\n",
                },
                capability="repo_environment",
                difficulty="delegated_review",
                metadata={
                    "benchmark_family": "repo_chore",
                    "semantic_verifier": {
                        "kind": "repo_chore_review",
                        "report_rules": [
                            {
                                "path": "reports/diff_summary.txt",
                                "must_mention": ["updated"],
                                "covers": ["src/app.py", "tests/status_check.txt"],
                            },
                            {
                                "path": "reports/verification.txt",
                                "must_mention": ["checks", "passed", "preservation"],
                                "covers": ["docs/context.md"],
                            },
                        ],
                    },
                },
            ),
            "repo_patch_review_retrieval_task": self._task(
                task_id="repo_patch_review_retrieval_task",
                prompt=(
                    "Reproduce the canonical repo patch review packet from earlier tasks: preserve the "
                    "repo context, update the app status, update the status check, and emit the diff and "
                    "verification reports."
                ),
                workspace_subdir="repo_patch_review_retrieval_task",
                setup_commands=[
                    "mkdir -p docs src tests && printf 'repo context\\n' > docs/context.md && printf 'STATUS=pending\\n' > src/app.py && printf 'status pending\\n' > tests/status_check.txt",
                ],
                suggested_commands=[],
                success_command=(
                    "test -f docs/context.md && grep -q '^repo context$' docs/context.md && "
                    "test -f src/app.py && grep -q '^STATUS=ready$' src/app.py && "
                    "test -f tests/status_check.txt && grep -q '^status ready covered$' tests/status_check.txt && "
                    "test -f reports/diff_summary.txt && grep -q '^updated src/app.py and tests/status_check.txt$' reports/diff_summary.txt && "
                    "test -f reports/verification.txt && grep -q '^deterministic checks passed for docs/context.md preservation$' reports/verification.txt"
                ),
                expected_files=[
                    "docs/context.md",
                    "src/app.py",
                    "tests/status_check.txt",
                    "reports/diff_summary.txt",
                    "reports/verification.txt",
                ],
                expected_file_contents={
                    "docs/context.md": "repo context\n",
                    "src/app.py": "STATUS=ready\n",
                    "tests/status_check.txt": "status ready covered\n",
                    "reports/diff_summary.txt": "updated src/app.py and tests/status_check.txt\n",
                    "reports/verification.txt": "deterministic checks passed for docs/context.md preservation\n",
                },
                capability="repo_environment",
                difficulty="retrieval",
                metadata={
                    "benchmark_family": "repo_chore",
                    "requires_retrieval": True,
                    "source_task": "repo_patch_review_task",
                    "semantic_verifier": {
                        "kind": "repo_chore_review",
                        "report_rules": [
                            {
                                "path": "reports/diff_summary.txt",
                                "must_mention": ["updated"],
                                "covers": ["src/app.py", "tests/status_check.txt"],
                            },
                            {
                                "path": "reports/verification.txt",
                                "must_mention": ["checks", "passed", "preservation"],
                                "covers": ["docs/context.md"],
                            },
                        ],
                    },
                },
            ),
            "repo_cleanup_review_task": self._task(
                task_id="repo_cleanup_review_task",
                prompt=(
                    "Prepare the cleanup review packet: remove tmp/debug.log, preserve config/base.env, "
                    "create src/service.py containing SERVICE_MODE=stable, create reports/branch.txt "
                    "containing branch cleanup/stable-ready, and create reports/review.txt containing "
                    "review packet complete."
                ),
                workspace_subdir="repo_cleanup_review_task",
                setup_commands=[
                    "mkdir -p tmp config && printf 'debug trace\\n' > tmp/debug.log && printf 'BASE_ENV=kept\\n' > config/base.env",
                ],
                suggested_commands=[
                    "mkdir -p src reports && rm -f tmp/debug.log && printf 'SERVICE_MODE=stable\n' > src/service.py && printf 'branch cleanup/stable-ready removes tmp/debug.log\n' > reports/branch.txt && printf 'review packet complete with config/base.env preserved\n' > reports/review.txt",
                    "cat src/service.py",
                    "cat reports/branch.txt",
                    "cat reports/review.txt",
                ],
                success_command=(
                    "test ! -f tmp/debug.log && "
                    "test -f config/base.env && grep -q '^BASE_ENV=kept$' config/base.env && "
                    "test -f src/service.py && grep -q '^SERVICE_MODE=stable$' src/service.py && "
                    "test -f reports/branch.txt && grep -q '^branch cleanup/stable-ready removes tmp/debug.log$' reports/branch.txt && "
                    "test -f reports/review.txt && grep -q '^review packet complete with config/base.env preserved$' reports/review.txt"
                ),
                expected_files=[
                    "config/base.env",
                    "src/service.py",
                    "reports/branch.txt",
                    "reports/review.txt",
                ],
                forbidden_files=["tmp/debug.log"],
                expected_file_contents={
                    "config/base.env": "BASE_ENV=kept\n",
                    "src/service.py": "SERVICE_MODE=stable\n",
                    "reports/branch.txt": "branch cleanup/stable-ready removes tmp/debug.log\n",
                    "reports/review.txt": "review packet complete with config/base.env preserved\n",
                },
                capability="repo_environment",
                difficulty="delegated_review",
                metadata={
                    "benchmark_family": "repo_chore",
                    "semantic_verifier": {
                        "kind": "repo_chore_review",
                        "report_rules": [
                            {
                                "path": "reports/branch.txt",
                                "must_mention": ["branch", "cleanup", "stable"],
                                "covers": ["tmp/debug.log"],
                            },
                            {
                                "path": "reports/review.txt",
                                "must_mention": ["review", "complete", "preserved"],
                                "covers": ["config/base.env"],
                            },
                        ],
                    },
                },
            ),
            "repo_cleanup_review_retrieval_task": self._task(
                task_id="repo_cleanup_review_retrieval_task",
                prompt=(
                    "Reproduce the canonical cleanup review packet from earlier tasks: remove the debug log, "
                    "preserve the base env, create the stable service artifact, and emit the branch and review reports."
                ),
                workspace_subdir="repo_cleanup_review_retrieval_task",
                setup_commands=[
                    "mkdir -p tmp config && printf 'debug trace\\n' > tmp/debug.log && printf 'BASE_ENV=kept\\n' > config/base.env",
                ],
                suggested_commands=[],
                success_command=(
                    "test ! -f tmp/debug.log && "
                    "test -f config/base.env && grep -q '^BASE_ENV=kept$' config/base.env && "
                    "test -f src/service.py && grep -q '^SERVICE_MODE=stable$' src/service.py && "
                    "test -f reports/branch.txt && grep -q '^branch cleanup/stable-ready removes tmp/debug.log$' reports/branch.txt && "
                    "test -f reports/review.txt && grep -q '^review packet complete with config/base.env preserved$' reports/review.txt"
                ),
                expected_files=[
                    "config/base.env",
                    "src/service.py",
                    "reports/branch.txt",
                    "reports/review.txt",
                ],
                forbidden_files=["tmp/debug.log"],
                expected_file_contents={
                    "config/base.env": "BASE_ENV=kept\n",
                    "src/service.py": "SERVICE_MODE=stable\n",
                    "reports/branch.txt": "branch cleanup/stable-ready removes tmp/debug.log\n",
                    "reports/review.txt": "review packet complete with config/base.env preserved\n",
                },
                capability="repo_environment",
                difficulty="retrieval",
                metadata={
                    "benchmark_family": "repo_chore",
                    "requires_retrieval": True,
                    "source_task": "repo_cleanup_review_task",
                    "distractor_tasks": ["repo_patch_review_task"],
                    "semantic_verifier": {
                        "kind": "repo_chore_review",
                        "report_rules": [
                            {
                                "path": "reports/branch.txt",
                                "must_mention": ["branch", "cleanup", "stable"],
                                "covers": ["tmp/debug.log"],
                            },
                            {
                                "path": "reports/review.txt",
                                "must_mention": ["review", "complete", "preserved"],
                                "covers": ["config/base.env"],
                            },
                        ],
                    },
                },
            ),
            "git_repo_status_review_task": self._task(
                task_id="git_repo_status_review_task",
                prompt=(
                    "In this git repo sandbox, create branch review/status-ready, update src/app.py to contain "
                    "STATUS=ready, update tests/status_check.txt to contain status ready covered, run the local "
                    "status check script, create reports/diff_summary.txt describing the changed repo paths, and "
                    "create reports/test_report.txt containing status check passed."
                ),
                workspace_subdir="git_repo_status_review_task",
                setup_commands=[
                    "mkdir -p docs src tests && printf 'repo context\\n' > docs/context.md && printf 'STATUS=pending\\n' > src/app.py && printf 'status pending covered\\n' > tests/status_check.txt && printf '#!/bin/sh\\nset -eu\\ngrep -q \"^STATUS=ready$\" src/app.py\\ngrep -q \"^status ready covered$\" tests/status_check.txt\\n' > tests/check_status.sh && chmod +x tests/check_status.sh && git init && git config user.email agent@example.com && git config user.name 'Agent Kernel' && git add docs/context.md src/app.py tests/status_check.txt tests/check_status.sh && git commit -m 'baseline repo sandbox'",
                ],
                suggested_commands=[
                    "git checkout -b review/status-ready && mkdir -p reports && printf 'STATUS=ready\n' > src/app.py && printf 'status ready covered\n' > tests/status_check.txt && tests/check_status.sh && printf 'updated src/app.py, tests/status_check.txt, reports/diff_summary.txt, and reports/test_report.txt on branch review/status-ready\n' > reports/diff_summary.txt && printf 'status check passed\n' > reports/test_report.txt",
                    "git branch --show-current",
                    "git diff --name-only --relative",
                    "cat reports/test_report.txt",
                ],
                success_command=(
                    "git branch --show-current | grep -q '^review/status-ready$' && "
                    "test -f reports/test_report.txt && grep -q '^status check passed$' reports/test_report.txt"
                ),
                expected_files=[
                    "docs/context.md",
                    "src/app.py",
                    "tests/status_check.txt",
                    "tests/check_status.sh",
                    "reports/diff_summary.txt",
                    "reports/test_report.txt",
                ],
                expected_file_contents={
                    "docs/context.md": "repo context\n",
                    "src/app.py": "STATUS=ready\n",
                    "tests/status_check.txt": "status ready covered\n",
                    "reports/test_report.txt": "status check passed\n",
                },
                capability="repo_environment",
                difficulty="git_workflow",
                metadata={
                    "benchmark_family": "repo_sandbox",
                    "requires_git": True,
                    "workflow_guard": {
                        "requires_git": True,
                    },
                    "semantic_verifier": {
                        "kind": "git_repo_review",
                        "expected_branch": "review/status-ready",
                        "expected_changed_paths": [
                            "reports/diff_summary.txt",
                            "reports/test_report.txt",
                            "src/app.py",
                            "tests/status_check.txt",
                        ],
                        "preserved_paths": ["docs/context.md", "tests/check_status.sh"],
                        "test_commands": [
                            {
                                "label": "status check script",
                                "argv": ["tests/check_status.sh"],
                            }
                        ],
                        "report_rules": [
                            {
                                "path": "reports/diff_summary.txt",
                                "must_mention": ["updated", "review/status-ready"],
                                "covers": [
                                    "src/app.py",
                                    "tests/status_check.txt",
                                    "reports/test_report.txt",
                                ],
                            },
                            {
                                "path": "reports/test_report.txt",
                                "must_mention": ["status", "check", "passed"],
                                "covers": ["tests/check_status.sh"],
                            },
                        ],
                    },
                },
            ),
            "git_repo_status_review_retrieval_task": self._task(
                task_id="git_repo_status_review_retrieval_task",
                prompt=(
                    "Reproduce the canonical git repo sandbox review workflow from earlier tasks: create the "
                    "review/status-ready branch, make the status-ready repo changes, run the local status check, "
                    "and emit the diff and test reports."
                ),
                workspace_subdir="git_repo_status_review_retrieval_task",
                setup_commands=[
                    "mkdir -p docs src tests && printf 'repo context\\n' > docs/context.md && printf 'STATUS=pending\\n' > src/app.py && printf 'status pending covered\\n' > tests/status_check.txt && printf '#!/bin/sh\\nset -eu\\ngrep -q \"^STATUS=ready$\" src/app.py\\ngrep -q \"^status ready covered$\" tests/status_check.txt\\n' > tests/check_status.sh && chmod +x tests/check_status.sh && git init && git config user.email agent@example.com && git config user.name 'Agent Kernel' && git add docs/context.md src/app.py tests/status_check.txt tests/check_status.sh && git commit -m 'baseline repo sandbox'",
                ],
                suggested_commands=[],
                success_command=(
                    "git branch --show-current | grep -q '^review/status-ready$' && "
                    "test -f reports/test_report.txt && grep -q '^status check passed$' reports/test_report.txt"
                ),
                expected_files=[
                    "docs/context.md",
                    "src/app.py",
                    "tests/status_check.txt",
                    "tests/check_status.sh",
                    "reports/diff_summary.txt",
                    "reports/test_report.txt",
                ],
                expected_file_contents={
                    "docs/context.md": "repo context\n",
                    "src/app.py": "STATUS=ready\n",
                    "tests/status_check.txt": "status ready covered\n",
                    "reports/test_report.txt": "status check passed\n",
                },
                capability="repo_environment",
                difficulty="retrieval",
                metadata={
                    "benchmark_family": "repo_sandbox",
                    "requires_git": True,
                    "requires_retrieval": True,
                    "source_task": "git_repo_status_review_task",
                    "workflow_guard": {
                        "requires_git": True,
                    },
                    "semantic_verifier": {
                        "kind": "git_repo_review",
                        "expected_branch": "review/status-ready",
                        "expected_changed_paths": [
                            "reports/diff_summary.txt",
                            "reports/test_report.txt",
                            "src/app.py",
                            "tests/status_check.txt",
                        ],
                        "preserved_paths": ["docs/context.md", "tests/check_status.sh"],
                        "test_commands": [
                            {
                                "label": "status check script",
                                "argv": ["tests/check_status.sh"],
                            }
                        ],
                        "report_rules": [
                            {
                                "path": "reports/diff_summary.txt",
                                "must_mention": ["updated", "review/status-ready"],
                                "covers": [
                                    "src/app.py",
                                    "tests/status_check.txt",
                                    "reports/test_report.txt",
                                ],
                            },
                            {
                                "path": "reports/test_report.txt",
                                "must_mention": ["status", "check", "passed"],
                                "covers": ["tests/check_status.sh"],
                            },
                        ],
                    },
                },
            ),
            "git_repo_test_repair_task": self._task(
                task_id="git_repo_test_repair_task",
                prompt=(
                    "In this git repo sandbox, create branch fix/release-ready, repair the failing deterministic "
                    "release test by updating src/release_state.txt to RELEASE_STATUS=ready, run the local release "
                    "test script, and create reports/diff_summary.txt and reports/test_report.txt describing the "
                    "accepted repair."
                ),
                workspace_subdir="git_repo_test_repair_task",
                setup_commands=[
                    "mkdir -p docs src tests && printf 'release notes preserved\\n' > docs/notes.md && printf 'RELEASE_STATUS=broken\\n' > src/release_state.txt && printf '#!/bin/sh\\nset -eu\\ngrep -q \"^RELEASE_STATUS=ready$\" src/release_state.txt\\n' > tests/test_release.sh && chmod +x tests/test_release.sh && git init && git config user.email agent@example.com && git config user.name 'Agent Kernel' && git add docs/notes.md src/release_state.txt tests/test_release.sh && git commit -m 'baseline failing release test'",
                ],
                suggested_commands=[
                    "git checkout -b fix/release-ready && mkdir -p reports && printf 'RELEASE_STATUS=ready\n' > src/release_state.txt && tests/test_release.sh && printf 'repaired failing deterministic release test by updating src/release_state.txt on branch fix/release-ready\n' > reports/diff_summary.txt && printf 'release test passed\n' > reports/test_report.txt",
                    "git branch --show-current",
                    "git diff --name-only --relative",
                    "cat reports/test_report.txt",
                ],
                success_command=(
                    "git branch --show-current | grep -q '^fix/release-ready$' && "
                    "test -f reports/test_report.txt && grep -q '^release test passed$' reports/test_report.txt"
                ),
                expected_files=[
                    "docs/notes.md",
                    "src/release_state.txt",
                    "tests/test_release.sh",
                    "reports/diff_summary.txt",
                    "reports/test_report.txt",
                ],
                expected_file_contents={
                    "docs/notes.md": "release notes preserved\n",
                    "src/release_state.txt": "RELEASE_STATUS=ready\n",
                    "reports/test_report.txt": "release test passed\n",
                },
                capability="repo_environment",
                difficulty="git_test_repair",
                metadata={
                    "benchmark_family": "repo_sandbox",
                    "requires_git": True,
                    "workflow_guard": {
                        "requires_git": True,
                    },
                    "semantic_verifier": {
                        "kind": "git_repo_review",
                        "expected_branch": "fix/release-ready",
                        "expected_changed_paths": [
                            "reports/diff_summary.txt",
                            "reports/test_report.txt",
                            "src/release_state.txt",
                        ],
                        "preserved_paths": ["docs/notes.md", "tests/test_release.sh"],
                        "test_commands": [
                            {
                                "label": "release test script",
                                "argv": ["tests/test_release.sh"],
                            }
                        ],
                        "report_rules": [
                            {
                                "path": "reports/diff_summary.txt",
                                "must_mention": ["repaired", "failing", "fix/release-ready"],
                                "covers": ["src/release_state.txt", "reports/test_report.txt"],
                            },
                            {
                                "path": "reports/test_report.txt",
                                "must_mention": ["release", "test", "passed"],
                                "covers": ["tests/test_release.sh"],
                            },
                        ],
                    },
                },
            ),
            "git_repo_test_repair_retrieval_task": self._task(
                task_id="git_repo_test_repair_retrieval_task",
                prompt=(
                    "Reproduce the canonical git repo sandbox repair workflow from earlier tasks: create the "
                    "fix/release-ready branch, repair the failing deterministic release test, run the local "
                    "release test script, and emit the diff and test reports."
                ),
                workspace_subdir="git_repo_test_repair_retrieval_task",
                setup_commands=[
                    "mkdir -p docs src tests && printf 'release notes preserved\\n' > docs/notes.md && printf 'RELEASE_STATUS=broken\\n' > src/release_state.txt && printf '#!/bin/sh\\nset -eu\\ngrep -q \"^RELEASE_STATUS=ready$\" src/release_state.txt\\n' > tests/test_release.sh && chmod +x tests/test_release.sh && git init && git config user.email agent@example.com && git config user.name 'Agent Kernel' && git add docs/notes.md src/release_state.txt tests/test_release.sh && git commit -m 'baseline failing release test'",
                ],
                suggested_commands=[],
                success_command=(
                    "git branch --show-current | grep -q '^fix/release-ready$' && "
                    "test -f reports/test_report.txt && grep -q '^release test passed$' reports/test_report.txt"
                ),
                expected_files=[
                    "docs/notes.md",
                    "src/release_state.txt",
                    "tests/test_release.sh",
                    "reports/diff_summary.txt",
                    "reports/test_report.txt",
                ],
                expected_file_contents={
                    "docs/notes.md": "release notes preserved\n",
                    "src/release_state.txt": "RELEASE_STATUS=ready\n",
                    "reports/test_report.txt": "release test passed\n",
                },
                capability="repo_environment",
                difficulty="retrieval",
                metadata={
                    "benchmark_family": "repo_sandbox",
                    "requires_git": True,
                    "requires_retrieval": True,
                    "source_task": "git_repo_test_repair_task",
                    "workflow_guard": {
                        "requires_git": True,
                    },
                    "semantic_verifier": {
                        "kind": "git_repo_review",
                        "expected_branch": "fix/release-ready",
                        "expected_changed_paths": [
                            "reports/diff_summary.txt",
                            "reports/test_report.txt",
                            "src/release_state.txt",
                        ],
                        "preserved_paths": ["docs/notes.md", "tests/test_release.sh"],
                        "test_commands": [
                            {
                                "label": "release test script",
                                "argv": ["tests/test_release.sh"],
                            }
                        ],
                        "report_rules": [
                            {
                                "path": "reports/diff_summary.txt",
                                "must_mention": ["repaired", "failing", "fix/release-ready"],
                                "covers": ["src/release_state.txt", "reports/test_report.txt"],
                            },
                            {
                                "path": "reports/test_report.txt",
                                "must_mention": ["release", "test", "passed"],
                                "covers": ["tests/test_release.sh"],
                            },
                        ],
                    },
                },
            ),
            "git_parallel_worker_api_task": self._task(
                task_id="git_parallel_worker_api_task",
                prompt=(
                    "In the shared repo sandbox on branch worker/api-status, update src/api_status.txt to "
                    "API_STATUS=ready, run the api suite, and commit the branch without touching any other "
                    "worker-owned paths."
                ),
                workspace_subdir="git_parallel_worker_api_task",
                setup_commands=[],
                suggested_commands=[
                    "printf 'API_STATUS=ready\n' > src/api_status.txt && tests/test_api.sh && git add src/api_status.txt && git commit -m 'worker api status ready'",
                    "git branch --show-current",
                    "git diff --name-only --relative origin/main..HEAD",
                ],
                success_command="git branch --show-current",
                expected_files=[
                    "docs/status.md",
                    "src/api_status.txt",
                    "tests/test_api.sh",
                    "tests/test_docs.sh",
                ],
                expected_file_contents={
                    "docs/status.md": "status pending documented\n",
                    "src/api_status.txt": "API_STATUS=ready\n",
                },
                capability="repo_environment",
                difficulty="git_worker_branch",
                metadata={
                    "benchmark_family": "repo_sandbox",
                    "requires_git": True,
                    "shared_repo_order": 0,
                    "shared_repo_bootstrap_commands": [
                        "mkdir -p docs src tests && printf 'API_STATUS=pending\\n' > src/api_status.txt && printf 'status pending documented\\n' > docs/status.md && printf '#!/bin/sh\\nset -eu\\ngrep -q \"^API_STATUS=ready$\" src/api_status.txt\\n' > tests/test_api.sh && printf '#!/bin/sh\\nset -eu\\ngrep -q \"^status ready documented$\" docs/status.md\\n' > tests/test_docs.sh && chmod +x tests/test_api.sh tests/test_docs.sh && git init && git checkout -b main && git config user.email agent@example.com && git config user.name 'Agent Kernel' && git add docs/status.md src/api_status.txt tests/test_api.sh tests/test_docs.sh && git commit -m 'baseline parallel sandbox' && git tag baseline",
                    ],
                    "shared_repo_bootstrap_managed_paths": [
                        "docs/status.md",
                        "src/api_status.txt",
                        "tests/test_api.sh",
                        "tests/test_docs.sh",
                    ],
                    "workflow_guard": {
                        "requires_git": True,
                        "shared_repo_id": "repo_sandbox_parallel_merge",
                        "target_branch": "main",
                        "worker_branch": "worker/api-status",
                        "claimed_paths": ["src/api_status.txt"],
                    },
                    "semantic_verifier": {
                        "kind": "git_repo_review",
                        "expected_branch": "worker/api-status",
                        "diff_base_ref": "origin/main",
                        "expected_changed_paths": ["src/api_status.txt"],
                        "preserved_paths": ["docs/status.md", "tests/test_api.sh", "tests/test_docs.sh"],
                        "clean_worktree": True,
                        "test_commands": [
                            {
                                "label": "api suite",
                                "argv": ["tests/test_api.sh"],
                            }
                        ],
                    },
                },
            ),
            "git_parallel_worker_docs_task": self._task(
                task_id="git_parallel_worker_docs_task",
                prompt=(
                    "In the shared repo sandbox on branch worker/docs-status, update docs/status.md to "
                    "status ready documented, run the docs suite, and commit the branch without touching any "
                    "other worker-owned paths."
                ),
                workspace_subdir="git_parallel_worker_docs_task",
                setup_commands=[],
                suggested_commands=[
                    "printf 'status ready documented\n' > docs/status.md && tests/test_docs.sh && git add docs/status.md && git commit -m 'worker docs status ready'",
                    "git branch --show-current",
                    "git diff --name-only --relative origin/main..HEAD",
                ],
                success_command="git branch --show-current",
                expected_files=[
                    "docs/status.md",
                    "src/api_status.txt",
                    "tests/test_api.sh",
                    "tests/test_docs.sh",
                ],
                expected_file_contents={
                    "docs/status.md": "status ready documented\n",
                    "src/api_status.txt": "API_STATUS=pending\n",
                },
                capability="repo_environment",
                difficulty="git_worker_branch",
                metadata={
                    "benchmark_family": "repo_sandbox",
                    "requires_git": True,
                    "shared_repo_order": 0,
                    "shared_repo_bootstrap_commands": [
                        "mkdir -p docs src tests && printf 'API_STATUS=pending\\n' > src/api_status.txt && printf 'status pending documented\\n' > docs/status.md && printf '#!/bin/sh\\nset -eu\\ngrep -q \"^API_STATUS=ready$\" src/api_status.txt\\n' > tests/test_api.sh && printf '#!/bin/sh\\nset -eu\\ngrep -q \"^status ready documented$\" docs/status.md\\n' > tests/test_docs.sh && chmod +x tests/test_api.sh tests/test_docs.sh && git init && git checkout -b main && git config user.email agent@example.com && git config user.name 'Agent Kernel' && git add docs/status.md src/api_status.txt tests/test_api.sh tests/test_docs.sh && git commit -m 'baseline parallel sandbox' && git tag baseline",
                    ],
                    "shared_repo_bootstrap_managed_paths": [
                        "docs/status.md",
                        "src/api_status.txt",
                        "tests/test_api.sh",
                        "tests/test_docs.sh",
                    ],
                    "workflow_guard": {
                        "requires_git": True,
                        "shared_repo_id": "repo_sandbox_parallel_merge",
                        "target_branch": "main",
                        "worker_branch": "worker/docs-status",
                        "claimed_paths": ["docs/status.md"],
                    },
                    "semantic_verifier": {
                        "kind": "git_repo_review",
                        "expected_branch": "worker/docs-status",
                        "diff_base_ref": "origin/main",
                        "expected_changed_paths": ["docs/status.md"],
                        "preserved_paths": ["src/api_status.txt", "tests/test_api.sh", "tests/test_docs.sh"],
                        "clean_worktree": True,
                        "test_commands": [
                            {
                                "label": "docs suite",
                                "argv": ["tests/test_docs.sh"],
                            }
                        ],
                    },
                },
            ),
            "git_parallel_merge_acceptance_task": self._task(
                task_id="git_parallel_merge_acceptance_task",
                prompt=(
                    "In this shared git repo sandbox, accept worker/api-status and worker/docs-status into main "
                    "without collisions, run the api and docs suites, and record the accepted merge packet."
                ),
                workspace_subdir="git_parallel_merge_acceptance_task",
                setup_commands=[],
                suggested_commands=[
                    "git merge --no-ff worker/api-status -m 'merge worker/api-status' && git merge --no-ff worker/docs-status -m 'merge worker/docs-status' && tests/test_api.sh && tests/test_docs.sh && mkdir -p reports && printf 'accepted worker/api-status for src/api_status.txt and worker/docs-status for docs/status.md into main without collisions\n' > reports/merge_report.txt && printf 'api suite passed; docs suite passed\n' > reports/test_report.txt && git add reports/merge_report.txt reports/test_report.txt && git commit -m 'record merge acceptance reports'",
                    "git branch --show-current",
                    "git diff --name-only --relative baseline..HEAD",
                    "cat reports/test_report.txt",
                ],
                success_command=(
                    "test -f reports/test_report.txt && "
                    "grep -q '^api suite passed; docs suite passed$' reports/test_report.txt"
                ),
                expected_files=[
                    "docs/status.md",
                    "src/api_status.txt",
                    "tests/test_api.sh",
                    "tests/test_docs.sh",
                    "reports/merge_report.txt",
                    "reports/test_report.txt",
                ],
                expected_file_contents={
                    "docs/status.md": "status ready documented\n",
                    "src/api_status.txt": "API_STATUS=ready\n",
                    "reports/merge_report.txt": (
                        "accepted worker/api-status for src/api_status.txt and worker/docs-status for "
                        "docs/status.md into main without collisions\n"
                    ),
                    "reports/test_report.txt": "api suite passed; docs suite passed\n",
                },
                capability="repo_environment",
                difficulty="git_parallel_merge",
                metadata={
                    "benchmark_family": "repo_sandbox",
                    "requires_git": True,
                    "shared_repo_order": 1,
                    "shared_repo_bootstrap_commands": [
                        "mkdir -p docs src tests && printf 'API_STATUS=pending\\n' > src/api_status.txt && printf 'status pending documented\\n' > docs/status.md && printf '#!/bin/sh\\nset -eu\\ngrep -q \"^API_STATUS=ready$\" src/api_status.txt\\n' > tests/test_api.sh && printf '#!/bin/sh\\nset -eu\\ngrep -q \"^status ready documented$\" docs/status.md\\n' > tests/test_docs.sh && chmod +x tests/test_api.sh tests/test_docs.sh && git init && git checkout -b main && git config user.email agent@example.com && git config user.name 'Agent Kernel' && git add docs/status.md src/api_status.txt tests/test_api.sh tests/test_docs.sh && git commit -m 'baseline parallel sandbox' && git tag baseline",
                    ],
                    "shared_repo_bootstrap_managed_paths": [
                        "docs/status.md",
                        "src/api_status.txt",
                        "tests/test_api.sh",
                        "tests/test_docs.sh",
                    ],
                    "workflow_guard": {
                        "requires_git": True,
                        "shared_repo_id": "repo_sandbox_parallel_merge",
                        "target_branch": "main",
                        "claimed_paths": [
                            "docs/status.md",
                            "src/api_status.txt",
                            "reports/merge_report.txt",
                            "reports/test_report.txt",
                        ],
                    },
                    "semantic_verifier": {
                        "kind": "git_repo_review",
                        "expected_branch": "main",
                        "diff_base_ref": "baseline",
                        "required_merged_branches": ["worker/api-status", "worker/docs-status"],
                        "expected_changed_paths": [
                            "docs/status.md",
                            "reports/merge_report.txt",
                            "reports/test_report.txt",
                            "src/api_status.txt",
                        ],
                        "preserved_paths": ["tests/test_api.sh", "tests/test_docs.sh"],
                        "clean_worktree": True,
                        "test_commands": [
                            {
                                "label": "api suite",
                                "argv": ["tests/test_api.sh"],
                            },
                            {
                                "label": "docs suite",
                                "argv": ["tests/test_docs.sh"],
                            },
                        ],
                        "report_rules": [
                            {
                                "path": "reports/merge_report.txt",
                                "must_mention": ["accepted", "without collisions", "main"],
                                "covers": ["src/api_status.txt", "docs/status.md"],
                            },
                            {
                                "path": "reports/test_report.txt",
                                "must_mention": ["api", "docs", "passed"],
                                "covers": ["tests/test_api.sh", "tests/test_docs.sh"],
                            },
                        ],
                    },
                },
            ),
            "git_conflict_worker_status_task": self._task(
                task_id="git_conflict_worker_status_task",
                prompt=(
                    "In the shared repo sandbox on branch worker/status-refresh, update src/shared_status.txt to "
                    "SERVICE_STATUS=worker-ready, run the worker refresh suite, and commit the branch."
                ),
                workspace_subdir="git_conflict_worker_status_task",
                setup_commands=[],
                suggested_commands=[
                    "printf 'SERVICE_STATUS=worker-ready\n' > src/shared_status.txt && tests/test_worker_refresh.sh && git add src/shared_status.txt && git commit -m 'worker status refresh'",
                    "git branch --show-current",
                    "git diff --name-only --relative origin/main..HEAD",
                ],
                success_command="git branch --show-current",
                expected_files=[
                    "docs/notes.md",
                    "scripts/generate_bundle.sh",
                    "src/shared_status.txt",
                    "tests/test_worker_refresh.sh",
                    "tests/test_service.sh",
                    "tests/test_bundle.sh",
                ],
                expected_file_contents={
                    "docs/notes.md": "notes pending\n",
                    "src/shared_status.txt": "SERVICE_STATUS=worker-ready\n",
                },
                capability="repo_environment",
                difficulty="git_worker_branch",
                metadata={
                    "benchmark_family": "repo_sandbox",
                    "requires_git": True,
                    "shared_repo_order": 0,
                    "shared_repo_bootstrap_commands": [
                        "mkdir -p docs scripts src tests && printf 'SERVICE_STATUS=baseline\\n' > src/shared_status.txt && printf 'notes pending\\n' > docs/notes.md && printf '#!/bin/sh\\nset -eu\\nmkdir -p dist\\ncat src/shared_status.txt docs/notes.md > dist/status_bundle.txt\\n' > scripts/generate_bundle.sh && printf '#!/bin/sh\\nset -eu\\ngrep -q \"^SERVICE_STATUS=worker-ready$\" src/shared_status.txt\\n' > tests/test_worker_refresh.sh && printf '#!/bin/sh\\nset -eu\\ngrep -q \"^SERVICE_STATUS=resolved$\" src/shared_status.txt\\n' > tests/test_service.sh && printf '#!/bin/sh\\nset -eu\\ngrep -q \"^SERVICE_STATUS=resolved$\" dist/status_bundle.txt\\ngrep -q \"^notes ready$\" dist/status_bundle.txt\\n' > tests/test_bundle.sh && chmod +x scripts/generate_bundle.sh tests/test_worker_refresh.sh tests/test_service.sh tests/test_bundle.sh && git init && git checkout -b main && git config user.email agent@example.com && git config user.name 'Agent Kernel' && git add docs/notes.md scripts/generate_bundle.sh src/shared_status.txt tests/test_worker_refresh.sh tests/test_service.sh tests/test_bundle.sh && git commit -m 'baseline generated sandbox' && git tag baseline",
                    ],
                    "shared_repo_bootstrap_managed_paths": [
                        "docs/notes.md",
                        "scripts/generate_bundle.sh",
                        "src/shared_status.txt",
                        "tests/test_worker_refresh.sh",
                        "tests/test_service.sh",
                        "tests/test_bundle.sh",
                    ],
                    "workflow_guard": {
                        "requires_git": True,
                        "shared_repo_id": "repo_sandbox_generated_conflict",
                        "target_branch": "main",
                        "worker_branch": "worker/status-refresh",
                        "claimed_paths": ["src/shared_status.txt"],
                    },
                    "semantic_verifier": {
                        "kind": "git_repo_review",
                        "expected_branch": "worker/status-refresh",
                        "diff_base_ref": "origin/main",
                        "expected_changed_paths": ["src/shared_status.txt"],
                        "preserved_paths": [
                            "docs/notes.md",
                            "scripts/generate_bundle.sh",
                            "tests/test_worker_refresh.sh",
                            "tests/test_service.sh",
                            "tests/test_bundle.sh",
                        ],
                        "clean_worktree": True,
                        "test_commands": [
                            {
                                "label": "worker refresh suite",
                                "argv": ["tests/test_worker_refresh.sh"],
                            }
                        ],
                    },
                },
            ),
            "git_generated_conflict_resolution_task": self._task(
                task_id="git_generated_conflict_resolution_task",
                prompt=(
                    "In this shared git repo sandbox, update main so worker/status-refresh conflicts on "
                    "src/shared_status.txt, resolve that conflict before acceptance, regenerate dist/status_bundle.txt, "
                    "run the selected service and bundle suites, and record the accepted merge packet."
                ),
                workspace_subdir="git_generated_conflict_resolution_task",
                setup_commands=[],
                suggested_commands=[
                    "printf 'SERVICE_STATUS=mainline-ready\n' > src/shared_status.txt && git add src/shared_status.txt && git commit -m 'mainline status change' && git merge -X theirs --no-ff worker/status-refresh -m 'merge worker/status-refresh' && printf 'SERVICE_STATUS=resolved\n' > src/shared_status.txt && printf 'notes ready\n' > docs/notes.md && scripts/generate_bundle.sh && tests/test_service.sh && tests/test_bundle.sh && mkdir -p reports && printf 'resolved worker/status-refresh merge conflict on src/shared_status.txt before acceptance into main\n' > reports/merge_report.txt && printf 'service suite passed; bundle suite passed\n' > reports/test_report.txt && git add docs/notes.md dist/status_bundle.txt reports/merge_report.txt reports/test_report.txt src/shared_status.txt && git commit -m 'resolve merge conflict and regenerate bundle'",
                    "git branch --show-current",
                    "git diff --name-only --relative baseline..HEAD",
                    "cat reports/test_report.txt",
                ],
                success_command=(
                    "test -f reports/test_report.txt && "
                    "grep -q '^service suite passed; bundle suite passed$' reports/test_report.txt"
                ),
                expected_files=[
                    "docs/notes.md",
                    "dist/status_bundle.txt",
                    "reports/merge_report.txt",
                    "reports/test_report.txt",
                    "scripts/generate_bundle.sh",
                    "src/shared_status.txt",
                    "tests/test_worker_refresh.sh",
                    "tests/test_bundle.sh",
                    "tests/test_service.sh",
                ],
                expected_file_contents={
                    "docs/notes.md": "notes ready\n",
                    "dist/status_bundle.txt": "SERVICE_STATUS=resolved\nnotes ready\n",
                    "reports/merge_report.txt": (
                        "resolved worker/status-refresh merge conflict on src/shared_status.txt before "
                        "acceptance into main\n"
                    ),
                    "reports/test_report.txt": "service suite passed; bundle suite passed\n",
                    "src/shared_status.txt": "SERVICE_STATUS=resolved\n",
                },
                capability="repo_environment",
                difficulty="git_conflict_resolution",
                metadata={
                    "benchmark_family": "repo_sandbox",
                    "requires_git": True,
                    "shared_repo_order": 1,
                    "shared_repo_bootstrap_commands": [
                        "mkdir -p docs scripts src tests && printf 'SERVICE_STATUS=baseline\\n' > src/shared_status.txt && printf 'notes pending\\n' > docs/notes.md && printf '#!/bin/sh\\nset -eu\\nmkdir -p dist\\ncat src/shared_status.txt docs/notes.md > dist/status_bundle.txt\\n' > scripts/generate_bundle.sh && printf '#!/bin/sh\\nset -eu\\ngrep -q \"^SERVICE_STATUS=worker-ready$\" src/shared_status.txt\\n' > tests/test_worker_refresh.sh && printf '#!/bin/sh\\nset -eu\\ngrep -q \"^SERVICE_STATUS=resolved$\" src/shared_status.txt\\n' > tests/test_service.sh && printf '#!/bin/sh\\nset -eu\\ngrep -q \"^SERVICE_STATUS=resolved$\" dist/status_bundle.txt\\ngrep -q \"^notes ready$\" dist/status_bundle.txt\\n' > tests/test_bundle.sh && chmod +x scripts/generate_bundle.sh tests/test_worker_refresh.sh tests/test_service.sh tests/test_bundle.sh && git init && git checkout -b main && git config user.email agent@example.com && git config user.name 'Agent Kernel' && git add docs/notes.md scripts/generate_bundle.sh src/shared_status.txt tests/test_worker_refresh.sh tests/test_service.sh tests/test_bundle.sh && git commit -m 'baseline generated sandbox' && git tag baseline",
                    ],
                    "shared_repo_bootstrap_managed_paths": [
                        "docs/notes.md",
                        "scripts/generate_bundle.sh",
                        "src/shared_status.txt",
                        "tests/test_worker_refresh.sh",
                        "tests/test_service.sh",
                        "tests/test_bundle.sh",
                    ],
                    "workflow_guard": {
                        "requires_git": True,
                        "touches_generated_paths": True,
                        "shared_repo_id": "repo_sandbox_generated_conflict",
                        "target_branch": "main",
                        "claimed_paths": [
                            "docs/notes.md",
                            "dist/status_bundle.txt",
                            "reports/merge_report.txt",
                            "reports/test_report.txt",
                            "src/shared_status.txt",
                        ],
                    },
                    "semantic_verifier": {
                        "kind": "git_repo_review",
                        "expected_branch": "main",
                        "diff_base_ref": "baseline",
                        "required_merged_branches": ["worker/status-refresh"],
                        "expected_changed_paths": [
                            "docs/notes.md",
                            "dist/status_bundle.txt",
                            "reports/merge_report.txt",
                            "reports/test_report.txt",
                            "src/shared_status.txt",
                        ],
                        "generated_paths": ["dist/status_bundle.txt"],
                        "resolved_conflict_paths": ["src/shared_status.txt"],
                        "preserved_paths": [
                            "scripts/generate_bundle.sh",
                            "tests/test_worker_refresh.sh",
                            "tests/test_bundle.sh",
                            "tests/test_service.sh",
                        ],
                        "clean_worktree": True,
                        "test_commands": [
                            {
                                "label": "service suite",
                                "argv": ["tests/test_service.sh"],
                            },
                            {
                                "label": "bundle suite",
                                "argv": ["tests/test_bundle.sh"],
                            },
                        ],
                        "report_rules": [
                            {
                                "path": "reports/merge_report.txt",
                                "must_mention": ["resolved", "merge conflict", "main"],
                                "covers": ["src/shared_status.txt"],
                            },
                            {
                                "path": "reports/test_report.txt",
                                "must_mention": ["service", "bundle", "passed"],
                                "covers": ["tests/test_service.sh", "tests/test_bundle.sh"],
                            },
                        ],
                    },
                },
            ),
            "api_contract_task": self._task(
                task_id="api_contract_task",
                prompt=(
                    "Prepare the API contract bundle: preserve requests/template.http, create api/request.json "
                    "containing {\"route\": \"/health\", \"method\": \"GET\"}, and create responses/expected.json "
                    "containing {\"status\": \"ok\"}."
                ),
                workspace_subdir="api_contract_task",
                setup_commands=[
                    "mkdir -p requests && printf 'GET /template\\n' > requests/template.http",
                ],
                suggested_commands=[
                    "mkdir -p api responses && printf '{\"route\": \"/health\", \"method\": \"GET\"}\n' > api/request.json && printf '{\"status\": \"ok\"}\n' > responses/expected.json",
                    "cat api/request.json",
                    "cat responses/expected.json",
                ],
                success_command=(
                    "test -f requests/template.http && grep -q '^GET /template$' requests/template.http && "
                    "test -f api/request.json && grep -q '^{\"route\": \"/health\", \"method\": \"GET\"}$' api/request.json && "
                    "test -f responses/expected.json && grep -q '^{\"status\": \"ok\"}$' responses/expected.json"
                ),
                expected_files=["requests/template.http", "api/request.json", "responses/expected.json"],
                expected_file_contents={
                    "requests/template.http": "GET /template\n",
                    "api/request.json": "{\"route\": \"/health\", \"method\": \"GET\"}\n",
                    "responses/expected.json": "{\"status\": \"ok\"}\n",
                },
                capability="tool_environment",
                difficulty="cross_tool",
                metadata={"benchmark_family": "tooling"},
            ),
            "api_contract_retrieval_task": self._task(
                task_id="api_contract_retrieval_task",
                prompt=(
                    "Reproduce the canonical API contract procedure from earlier tasks: preserve the request template, "
                    "write the health request payload, and create the expected ok response."
                ),
                workspace_subdir="api_contract_retrieval_task",
                setup_commands=[
                    "mkdir -p requests && printf 'GET /template\\n' > requests/template.http",
                ],
                suggested_commands=[],
                success_command=(
                    "test -f requests/template.http && grep -q '^GET /template$' requests/template.http && "
                    "test -f api/request.json && grep -q '^{\"route\": \"/health\", \"method\": \"GET\"}$' api/request.json && "
                    "test -f responses/expected.json && grep -q '^{\"status\": \"ok\"}$' responses/expected.json"
                ),
                expected_files=["requests/template.http", "api/request.json", "responses/expected.json"],
                expected_file_contents={
                    "requests/template.http": "GET /template\n",
                    "api/request.json": "{\"route\": \"/health\", \"method\": \"GET\"}\n",
                    "responses/expected.json": "{\"status\": \"ok\"}\n",
                },
                capability="tool_environment",
                difficulty="cross_tool",
                metadata={
                    "benchmark_family": "tooling",
                    "requires_retrieval": True,
                    "source_task": "api_contract_task",
                },
            ),
            "cli_exchange_task": self._task(
                task_id="cli_exchange_task",
                prompt=(
                    "Prepare the CLI exchange bundle: preserve prompts/base.txt, create commands/run.sh containing "
                    "echo tool ready, and create outputs/result.txt containing tool output ready."
                ),
                workspace_subdir="cli_exchange_task",
                setup_commands=[
                    "mkdir -p prompts && printf 'base prompt\\n' > prompts/base.txt",
                ],
                suggested_commands=[
                    "mkdir -p commands outputs && printf 'echo tool ready\n' > commands/run.sh && printf 'tool output ready\n' > outputs/result.txt",
                    "cat commands/run.sh",
                    "cat outputs/result.txt",
                ],
                success_command=(
                    "test -f prompts/base.txt && grep -q '^base prompt$' prompts/base.txt && "
                    "test -f commands/run.sh && grep -q '^echo tool ready$' commands/run.sh && "
                    "test -f outputs/result.txt && grep -q '^tool output ready$' outputs/result.txt"
                ),
                expected_files=["prompts/base.txt", "commands/run.sh", "outputs/result.txt"],
                expected_file_contents={
                    "prompts/base.txt": "base prompt\n",
                    "commands/run.sh": "echo tool ready\n",
                    "outputs/result.txt": "tool output ready\n",
                },
                capability="tool_environment",
                difficulty="cross_tool",
                metadata={"benchmark_family": "tooling"},
            ),
            "cli_exchange_retrieval_task": self._task(
                task_id="cli_exchange_retrieval_task",
                prompt=(
                    "Reproduce the canonical CLI exchange procedure from earlier tasks: preserve the base prompt, "
                    "write the run script, and create the ready output artifact."
                ),
                workspace_subdir="cli_exchange_retrieval_task",
                setup_commands=[
                    "mkdir -p prompts && printf 'base prompt\\n' > prompts/base.txt",
                ],
                suggested_commands=[],
                success_command=(
                    "test -f prompts/base.txt && grep -q '^base prompt$' prompts/base.txt && "
                    "test -f commands/run.sh && grep -q '^echo tool ready$' commands/run.sh && "
                    "test -f outputs/result.txt && grep -q '^tool output ready$' outputs/result.txt"
                ),
                expected_files=["prompts/base.txt", "commands/run.sh", "outputs/result.txt"],
                expected_file_contents={
                    "prompts/base.txt": "base prompt\n",
                    "commands/run.sh": "echo tool ready\n",
                    "outputs/result.txt": "tool output ready\n",
                },
                capability="tool_environment",
                difficulty="cross_tool",
                metadata={
                    "benchmark_family": "tooling",
                    "requires_retrieval": True,
                    "source_task": "cli_exchange_task",
                },
            ),
            "service_mesh_task": self._task(
                task_id="service_mesh_task",
                prompt=(
                    "Prepare the integration workspace: create gateway/routes.txt containing routes synced, "
                    "create services/api.env containing API_VERSION=v2 and QUEUE_ENABLED=yes, and create "
                    "reports/health.txt containing integration green."
                ),
                workspace_subdir="service_mesh_task",
                suggested_commands=[
                    "mkdir -p gateway services reports && printf 'routes synced\n' > gateway/routes.txt && printf 'API_VERSION=v2\nQUEUE_ENABLED=yes\n' > services/api.env && printf 'integration green\n' > reports/health.txt",
                    "cat gateway/routes.txt",
                    "cat services/api.env",
                    "cat reports/health.txt",
                ],
                success_command=(
                    "test -f gateway/routes.txt && grep -q '^routes synced$' gateway/routes.txt && "
                    "test -f services/api.env && grep -q '^API_VERSION=v2$' services/api.env && "
                    "grep -q '^QUEUE_ENABLED=yes$' services/api.env && "
                    "test -f reports/health.txt && grep -q '^integration green$' reports/health.txt"
                ),
                expected_files=["gateway/routes.txt", "services/api.env", "reports/health.txt"],
                expected_file_contents={
                    "gateway/routes.txt": "routes synced\n",
                    "services/api.env": "API_VERSION=v2\nQUEUE_ENABLED=yes\n",
                    "reports/health.txt": "integration green\n",
                },
                capability="integration_environment",
                difficulty="multi_system",
                metadata={"benchmark_family": "integration"},
            ),
            "incident_matrix_task": self._task(
                task_id="incident_matrix_task",
                prompt=(
                    "Prepare the incident integration bundle: create alerts/open.txt containing incident triaged, "
                    "create runbook/steps.txt containing rollback queued, and create summary/owner.txt containing "
                    "owner platform."
                ),
                workspace_subdir="incident_matrix_task",
                suggested_commands=[
                    "mkdir -p alerts runbook summary && printf 'incident triaged\n' > alerts/open.txt && printf 'rollback queued\n' > runbook/steps.txt && printf 'owner platform\n' > summary/owner.txt",
                    "cat alerts/open.txt",
                    "cat runbook/steps.txt",
                    "cat summary/owner.txt",
                ],
                success_command=(
                    "test -f alerts/open.txt && grep -q '^incident triaged$' alerts/open.txt && "
                    "test -f runbook/steps.txt && grep -q '^rollback queued$' runbook/steps.txt && "
                    "test -f summary/owner.txt && grep -q '^owner platform$' summary/owner.txt"
                ),
                expected_files=["alerts/open.txt", "runbook/steps.txt", "summary/owner.txt"],
                expected_file_contents={
                    "alerts/open.txt": "incident triaged\n",
                    "runbook/steps.txt": "rollback queued\n",
                    "summary/owner.txt": "owner platform\n",
                },
                capability="integration_environment",
                difficulty="multi_system",
                metadata={"benchmark_family": "integration"},
            ),
            "service_mesh_retrieval_task": self._task(
                task_id="service_mesh_retrieval_task",
                prompt=(
                    "Reproduce the canonical integration mesh procedure from earlier tasks: sync the gateway routes, "
                    "write the API integration env, and mark the health report green."
                ),
                workspace_subdir="service_mesh_retrieval_task",
                suggested_commands=[],
                success_command=(
                    "test -f gateway/routes.txt && grep -q '^routes synced$' gateway/routes.txt && "
                    "test -f services/api.env && grep -q '^API_VERSION=v2$' services/api.env && "
                    "grep -q '^QUEUE_ENABLED=yes$' services/api.env && "
                    "test -f reports/health.txt && grep -q '^integration green$' reports/health.txt"
                ),
                expected_files=["gateway/routes.txt", "services/api.env", "reports/health.txt"],
                expected_file_contents={
                    "gateway/routes.txt": "routes synced\n",
                    "services/api.env": "API_VERSION=v2\nQUEUE_ENABLED=yes\n",
                    "reports/health.txt": "integration green\n",
                },
                capability="integration_environment",
                difficulty="retrieval",
                metadata={
                    "benchmark_family": "integration",
                    "requires_retrieval": True,
                    "source_task": "service_mesh_task",
                    "distractor_tasks": ["incident_matrix_task"],
                },
            ),
            "incident_matrix_retrieval_task": self._task(
                task_id="incident_matrix_retrieval_task",
                prompt=(
                    "Reproduce the canonical incident integration bundle from prior tasks: triage the alert, queue "
                    "the rollback runbook, and assign the owner summary."
                ),
                workspace_subdir="incident_matrix_retrieval_task",
                suggested_commands=[],
                success_command=(
                    "test -f alerts/open.txt && grep -q '^incident triaged$' alerts/open.txt && "
                    "test -f runbook/steps.txt && grep -q '^rollback queued$' runbook/steps.txt && "
                    "test -f summary/owner.txt && grep -q '^owner platform$' summary/owner.txt"
                ),
                expected_files=["alerts/open.txt", "runbook/steps.txt", "summary/owner.txt"],
                expected_file_contents={
                    "alerts/open.txt": "incident triaged\n",
                    "runbook/steps.txt": "rollback queued\n",
                    "summary/owner.txt": "owner platform\n",
                },
                capability="integration_environment",
                difficulty="retrieval",
                metadata={
                    "benchmark_family": "integration",
                    "requires_retrieval": True,
                    "source_task": "incident_matrix_task",
                    "distractor_tasks": ["service_mesh_task"],
                },
            ),
            "queue_failover_task": self._task(
                task_id="queue_failover_task",
                prompt=(
                    "Prepare the failover integration bundle: create queues/consumer.env containing QUEUE_MODE=failover "
                    "and RETRIES=2, create orchestration/plan.txt containing failover staged, and create "
                    "summary/state.txt containing recovery ready."
                ),
                workspace_subdir="queue_failover_task",
                suggested_commands=[
                    "mkdir -p queues orchestration summary && printf 'QUEUE_MODE=failover\nRETRIES=2\n' > queues/consumer.env && printf 'failover staged\n' > orchestration/plan.txt && printf 'recovery ready\n' > summary/state.txt",
                    "cat queues/consumer.env",
                    "cat orchestration/plan.txt",
                    "cat summary/state.txt",
                ],
                success_command=(
                    "test -f queues/consumer.env && grep -q '^QUEUE_MODE=failover$' queues/consumer.env && "
                    "grep -q '^RETRIES=2$' queues/consumer.env && "
                    "test -f orchestration/plan.txt && grep -q '^failover staged$' orchestration/plan.txt && "
                    "test -f summary/state.txt && grep -q '^recovery ready$' summary/state.txt"
                ),
                expected_files=["queues/consumer.env", "orchestration/plan.txt", "summary/state.txt"],
                expected_file_contents={
                    "queues/consumer.env": "QUEUE_MODE=failover\nRETRIES=2\n",
                    "orchestration/plan.txt": "failover staged\n",
                    "summary/state.txt": "recovery ready\n",
                },
                capability="integration_environment",
                difficulty="multi_system",
                metadata={"benchmark_family": "integration"},
            ),
            "queue_failover_retrieval_task": self._task(
                task_id="queue_failover_retrieval_task",
                prompt=(
                    "Reproduce the canonical failover integration bundle from prior tasks: write the failover queue "
                    "config, stage the orchestration plan, and mark the recovery summary ready."
                ),
                workspace_subdir="queue_failover_retrieval_task",
                suggested_commands=[],
                success_command=(
                    "test -f queues/consumer.env && grep -q '^QUEUE_MODE=failover$' queues/consumer.env && "
                    "grep -q '^RETRIES=2$' queues/consumer.env && "
                    "test -f orchestration/plan.txt && grep -q '^failover staged$' orchestration/plan.txt && "
                    "test -f summary/state.txt && grep -q '^recovery ready$' summary/state.txt"
                ),
                expected_files=["queues/consumer.env", "orchestration/plan.txt", "summary/state.txt"],
                expected_file_contents={
                    "queues/consumer.env": "QUEUE_MODE=failover\nRETRIES=2\n",
                    "orchestration/plan.txt": "failover staged\n",
                    "summary/state.txt": "recovery ready\n",
                },
                capability="integration_environment",
                difficulty="multi_system",
                metadata={
                    "benchmark_family": "integration",
                    "requires_retrieval": True,
                    "source_task": "queue_failover_task",
                    "distractor_tasks": ["service_mesh_task", "incident_matrix_task"],
                },
            ),
            "handshake_seed_task": self._task(
                task_id="handshake_seed_task",
                prompt="Create handshake.txt containing the exact string kernel handshake ready.",
                workspace_subdir="handshake_seed_task",
                suggested_commands=[
                    "printf 'kernel handshake ready\n' > handshake.txt",
                    "cat handshake.txt",
                ],
                success_command="test -f handshake.txt && grep -q '^kernel handshake ready$' handshake.txt",
                expected_files=["handshake.txt"],
                expected_file_contents={"handshake.txt": "kernel handshake ready\n"},
                capability="retrieval_bootstrap",
                difficulty="bounded",
            ),
            "handshake_retrieval_task": self._task(
                task_id="handshake_retrieval_task",
                prompt=(
                    "Reproduce the canonical handshake artifact used elsewhere in this repo. "
                    "Create the standard handshake file with the exact established phrase."
                ),
                workspace_subdir="handshake_retrieval_task",
                suggested_commands=[],
                success_command="test -f handshake.txt && grep -q '^kernel handshake ready$' handshake.txt",
                expected_files=["handshake.txt"],
                expected_file_contents={"handshake.txt": "kernel handshake ready\n"},
                capability="retrieval_dependent",
                difficulty="retrieval",
                metadata={"requires_retrieval": True, "source_task": "handshake_seed_task"},
            ),
            "status_phrase_seed_task": self._task(
                task_id="status_phrase_seed_task",
                prompt="Create phrase.txt containing the exact string diagnostics green stable.",
                workspace_subdir="status_phrase_seed_task",
                suggested_commands=[
                    "printf 'diagnostics green stable\n' > phrase.txt",
                    "cat phrase.txt",
                ],
                success_command="test -f phrase.txt && grep -q '^diagnostics green stable$' phrase.txt",
                expected_files=["phrase.txt"],
                expected_file_contents={"phrase.txt": "diagnostics green stable\n"},
                capability="retrieval_bootstrap",
                difficulty="bounded",
            ),
            "status_phrase_fallback_seed_task": self._task(
                task_id="status_phrase_fallback_seed_task",
                prompt="Create phrase.txt containing the exact string diagnostics amber fallback.",
                workspace_subdir="status_phrase_fallback_seed_task",
                suggested_commands=[
                    "printf 'diagnostics amber fallback\n' > phrase.txt",
                    "cat phrase.txt",
                ],
                success_command="test -f phrase.txt && grep -q '^diagnostics amber fallback$' phrase.txt",
                expected_files=["phrase.txt"],
                expected_file_contents={"phrase.txt": "diagnostics amber fallback\n"},
                capability="retrieval_bootstrap",
                difficulty="bounded",
            ),
            "status_phrase_retrieval_task": self._task(
                task_id="status_phrase_retrieval_task",
                prompt=(
                    "Create the phrase artifact using the primary diagnostics wording previously established "
                    "in this repo. There is also a fallback wording elsewhere; do not use the fallback."
                ),
                workspace_subdir="status_phrase_retrieval_task",
                suggested_commands=[],
                success_command="test -f phrase.txt && grep -q '^diagnostics green stable$' phrase.txt",
                expected_files=["phrase.txt"],
                expected_file_contents={"phrase.txt": "diagnostics green stable\n"},
                capability="retrieval_dependent",
                difficulty="retrieval",
                metadata={
                    "requires_retrieval": True,
                    "source_task": "status_phrase_seed_task",
                    "distractor_tasks": ["status_phrase_fallback_seed_task"],
                },
            ),
            "status_phrase_fallback_retrieval_task": self._task(
                task_id="status_phrase_fallback_retrieval_task",
                prompt=(
                    "Create the phrase artifact using the fallback diagnostics wording established elsewhere "
                    "in this repo. Do not use the primary diagnostics wording."
                ),
                workspace_subdir="status_phrase_fallback_retrieval_task",
                suggested_commands=[],
                success_command="test -f phrase.txt && grep -q '^diagnostics amber fallback$' phrase.txt",
                expected_files=["phrase.txt"],
                expected_file_contents={"phrase.txt": "diagnostics amber fallback\n"},
                capability="retrieval_dependent",
                difficulty="retrieval",
                metadata={
                    "requires_retrieval": True,
                    "source_task": "status_phrase_fallback_seed_task",
                    "distractor_tasks": ["status_phrase_seed_task"],
                },
            ),
            "archive_command_seed_task": self._task(
                task_id="archive_command_seed_task",
                prompt="Create archive/out.txt containing archived using a single command.",
                workspace_subdir="archive_command_seed_task",
                suggested_commands=[
                    "mkdir -p archive && printf 'archived\n' > archive/out.txt",
                    "cat archive/out.txt",
                ],
                success_command="test -f archive/out.txt && grep -q '^archived$' archive/out.txt",
                expected_files=["archive/out.txt"],
                expected_file_contents={"archive/out.txt": "archived\n"},
                capability="retrieval_bootstrap",
                difficulty="bounded",
            ),
            "archive_retrieval_task": self._task(
                task_id="archive_retrieval_task",
                prompt=(
                    "Reproduce the established archive command pattern from earlier repo tasks. "
                    "Create archive/out.txt with the canonical archived content."
                ),
                workspace_subdir="archive_retrieval_task",
                suggested_commands=[],
                success_command="test -f archive/out.txt && grep -q '^archived$' archive/out.txt",
                expected_files=["archive/out.txt"],
                expected_file_contents={"archive/out.txt": "archived\n"},
                capability="retrieval_dependent",
                difficulty="retrieval",
                metadata={"requires_retrieval": True, "source_task": "archive_command_seed_task"},
            ),
            "avoidance_seed_task": self._task(
                task_id="avoidance_seed_task",
                prompt="Create safe.txt containing safe path only.",
                workspace_subdir="avoidance_seed_task",
                suggested_commands=[
                    "printf 'safe path only\n' > safe.txt",
                    "cat safe.txt",
                ],
                success_command="test -f safe.txt && grep -q '^safe path only$' safe.txt",
                expected_files=["safe.txt"],
                expected_file_contents={"safe.txt": "safe path only\n"},
                capability="retrieval_bootstrap",
                difficulty="bounded",
            ),
            "avoidance_retrieval_task": self._task(
                task_id="avoidance_retrieval_task",
                prompt=(
                    "Create the safe-path artifact using the prior successful pattern from this repo. "
                    "Avoid repeating previously failing command shapes."
                ),
                workspace_subdir="avoidance_retrieval_task",
                suggested_commands=[],
                success_command="test -f safe.txt && grep -q '^safe path only$' safe.txt",
                expected_files=["safe.txt"],
                expected_file_contents={"safe.txt": "safe path only\n"},
                capability="retrieval_dependent",
                difficulty="retrieval",
                metadata={"requires_retrieval": True, "source_task": "avoidance_seed_task"},
            ),
            "bundle_primary_seed_task": self._task(
                task_id="bundle_primary_seed_task",
                prompt="Create bundle/report.txt containing the exact string bundle verified primary.",
                workspace_subdir="bundle_primary_seed_task",
                suggested_commands=[
                    "mkdir -p bundle && printf 'bundle verified primary\n' > bundle/report.txt",
                    "cat bundle/report.txt",
                ],
                success_command="test -f bundle/report.txt && grep -q '^bundle verified primary$' bundle/report.txt",
                expected_files=["bundle/report.txt"],
                expected_file_contents={"bundle/report.txt": "bundle verified primary\n"},
                capability="retrieval_bootstrap",
                difficulty="bounded",
            ),
            "bundle_legacy_seed_task": self._task(
                task_id="bundle_legacy_seed_task",
                prompt="Create bundle/report.txt containing the exact string bundle verified legacy.",
                workspace_subdir="bundle_legacy_seed_task",
                suggested_commands=[
                    "mkdir -p bundle && printf 'bundle verified legacy\n' > bundle/report.txt",
                    "cat bundle/report.txt",
                ],
                success_command="test -f bundle/report.txt && grep -q '^bundle verified legacy$' bundle/report.txt",
                expected_files=["bundle/report.txt"],
                expected_file_contents={"bundle/report.txt": "bundle verified legacy\n"},
                capability="retrieval_bootstrap",
                difficulty="bounded",
            ),
            "bundle_retrieval_task": self._task(
                task_id="bundle_retrieval_task",
                prompt=(
                    "Recreate the canonical bundle report artifact used elsewhere in this repo. "
                    "A legacy bundle wording also exists; do not use the legacy wording."
                ),
                workspace_subdir="bundle_retrieval_task",
                suggested_commands=[],
                success_command="test -f bundle/report.txt && grep -q '^bundle verified primary$' bundle/report.txt",
                expected_files=["bundle/report.txt"],
                expected_file_contents={"bundle/report.txt": "bundle verified primary\n"},
                capability="retrieval_dependent",
                difficulty="retrieval",
                metadata={
                    "requires_retrieval": True,
                    "source_task": "bundle_primary_seed_task",
                    "distractor_tasks": ["bundle_legacy_seed_task"],
                },
            ),
            "bundle_legacy_retrieval_task": self._task(
                task_id="bundle_legacy_retrieval_task",
                prompt=(
                    "Recreate the legacy bundle report artifact used elsewhere in this repo. "
                    "Do not use the primary bundle wording."
                ),
                workspace_subdir="bundle_legacy_retrieval_task",
                suggested_commands=[],
                success_command="test -f bundle/report.txt && grep -q '^bundle verified legacy$' bundle/report.txt",
                expected_files=["bundle/report.txt"],
                expected_file_contents={"bundle/report.txt": "bundle verified legacy\n"},
                capability="retrieval_dependent",
                difficulty="retrieval",
                metadata={
                    "requires_retrieval": True,
                    "source_task": "bundle_legacy_seed_task",
                    "distractor_tasks": ["bundle_primary_seed_task"],
                },
            ),
            "checkpoint_blue_seed_task": self._task(
                task_id="checkpoint_blue_seed_task",
                prompt="Create checkpoint.txt containing the exact string checkpoint blue stable.",
                workspace_subdir="checkpoint_blue_seed_task",
                suggested_commands=[
                    "printf 'checkpoint blue stable\n' > checkpoint.txt",
                    "cat checkpoint.txt",
                ],
                success_command="test -f checkpoint.txt && grep -q '^checkpoint blue stable$' checkpoint.txt",
                expected_files=["checkpoint.txt"],
                expected_file_contents={"checkpoint.txt": "checkpoint blue stable\n"},
                capability="retrieval_bootstrap",
                difficulty="bounded",
            ),
            "checkpoint_green_seed_task": self._task(
                task_id="checkpoint_green_seed_task",
                prompt="Create checkpoint.txt containing the exact string checkpoint green stable.",
                workspace_subdir="checkpoint_green_seed_task",
                suggested_commands=[
                    "printf 'checkpoint green stable\n' > checkpoint.txt",
                    "cat checkpoint.txt",
                ],
                success_command="test -f checkpoint.txt && grep -q '^checkpoint green stable$' checkpoint.txt",
                expected_files=["checkpoint.txt"],
                expected_file_contents={"checkpoint.txt": "checkpoint green stable\n"},
                capability="retrieval_bootstrap",
                difficulty="bounded",
            ),
            "checkpoint_blue_retrieval_task": self._task(
                task_id="checkpoint_blue_retrieval_task",
                prompt=(
                    "Create the checkpoint artifact using the established blue stable wording. "
                    "A green checkpoint wording also exists elsewhere; do not use it."
                ),
                workspace_subdir="checkpoint_blue_retrieval_task",
                suggested_commands=[],
                success_command="test -f checkpoint.txt && grep -q '^checkpoint blue stable$' checkpoint.txt",
                expected_files=["checkpoint.txt"],
                expected_file_contents={"checkpoint.txt": "checkpoint blue stable\n"},
                capability="retrieval_dependent",
                difficulty="retrieval",
                metadata={
                    "requires_retrieval": True,
                    "source_task": "checkpoint_blue_seed_task",
                    "distractor_tasks": ["checkpoint_green_seed_task"],
                },
            ),
            "checkpoint_green_retrieval_task": self._task(
                task_id="checkpoint_green_retrieval_task",
                prompt=(
                    "Create the checkpoint artifact using the established green stable wording. "
                    "Do not use the blue checkpoint wording."
                ),
                workspace_subdir="checkpoint_green_retrieval_task",
                suggested_commands=[],
                success_command="test -f checkpoint.txt && grep -q '^checkpoint green stable$' checkpoint.txt",
                expected_files=["checkpoint.txt"],
                expected_file_contents={"checkpoint.txt": "checkpoint green stable\n"},
                capability="retrieval_dependent",
                difficulty="retrieval",
                metadata={
                    "requires_retrieval": True,
                    "source_task": "checkpoint_green_seed_task",
                    "distractor_tasks": ["checkpoint_blue_seed_task"],
                },
            ),
        }
        self._merge_external_tasks(
            external_task_manifests
            if external_task_manifests is not None
            else tuple(
                str(path)
                for path in getattr(config, "external_task_manifests_paths", ())
                if str(path).strip()
            )
        )

    @staticmethod
    def _task(*, capability: str, difficulty: str, **kwargs) -> TaskSpec:
        metadata = dict(kwargs.pop("metadata", {}))
        metadata.setdefault("capability", capability)
        metadata.setdefault("difficulty", difficulty)
        metadata.setdefault("benchmark_family", "bounded")
        return TaskSpec(metadata=metadata, **kwargs)

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


_SYNTHETIC_LINEAGE_TASK_SUFFIXES = (
    "_episode_replay",
    "_verifier_replay",
    "_discovered",
    "_transition_pressure",
    "_skill_replay",
    "_skill_transfer",
    "_tool_replay",
    "_benchmark_candidate",
    "_verifier_candidate",
)

_SYNTHETIC_LINEAGE_BENCHMARK_FAMILIES = {
    "episode_memory",
    "verifier_memory",
    "discovered_task",
    "transition_pressure",
    "skill_memory",
    "skill_transfer",
    "tool_memory",
    "operator_memory",
    "benchmark_candidate",
    "verifier_candidate",
}


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
    return memory_source in {
        "episode",
        "verifier",
        "discovered_task",
        "transition_pressure",
        "skill",
        "skill_transfer",
        "tool",
        "operator",
    }


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
            {
                "benchmark_family": "episode_memory",
                "memory_source": "episode",
                "memory_source_task": task_id,
                "origin_benchmark_family": str(contract.metadata.get("benchmark_family", "bounded")),
                "requires_retrieval": True,
                "source_task": task_id,
                "episode_phase": str(data.get("episode_storage", {}).get("phase", "")).strip(),
                "episode_relative_path": str(data.get("episode_storage", {}).get("relative_path", "")).strip(),
            }
        )
        replay_task = TaskSpec(
            task_id=f"{task_id}_episode_replay",
            prompt=(
                "Reproduce a previously successful task from episodic memory. "
                f"{contract.prompt}"
            ),
            workspace_subdir=f"{task_id}_episode_replay",
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
        tasks.append(replay_task)
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
            task_id=f"{task_id}_discovered",
            extra_metadata={
                "benchmark_family": "discovered_task",
                "memory_source": "discovered_task",
                "memory_source_task": task_id,
                "origin_benchmark_family": str(contract.metadata.get("benchmark_family", "bounded")),
                "source_task": task_id,
                "discovery_failure_types": failure_types,
                "discovery_transition_failures": transition_failures,
                "episode_phase": str(data.get("episode_storage", {}).get("phase", "")).strip(),
                "episode_relative_path": str(data.get("episode_storage", {}).get("relative_path", "")).strip(),
            },
        )
        strict_task.prompt = (
            "Solve a discovered robustness task derived from prior failures or bad state transitions. "
            f"{strict_task.prompt}"
        )
        strict_task.workspace_subdir = f"{contract.workspace_subdir}_discovered"
        strict_task.max_steps = max(strict_task.max_steps, len(strict_task.suggested_commands) + 1)
        tasks.append(strict_task)
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
            task_id=f"{task_id}_transition_pressure",
            extra_metadata={
                "benchmark_family": "transition_pressure",
                "memory_source": "transition_pressure",
                "memory_source_task": task_id,
                "origin_benchmark_family": str(contract.metadata.get("benchmark_family", "bounded")),
                "source_task": task_id,
                "discovery_transition_failures": transition_failures,
                "episode_phase": str(data.get("episode_storage", {}).get("phase", "")).strip(),
                "episode_relative_path": str(data.get("episode_storage", {}).get("relative_path", "")).strip(),
            },
        )
        pressure_task.prompt = (
            "Solve a transition-pressure task derived from prior bad state transitions. "
            f"Avoid repeating these failure modes: {', '.join(transition_failures)}. "
            f"{pressure_task.prompt}"
        )
        pressure_task.workspace_subdir = f"{contract.workspace_subdir}_transition_pressure"
        pressure_task.max_steps = max(pressure_task.max_steps, len(pressure_task.suggested_commands) + 1)
        tasks.append(pressure_task)
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
    return (
        f"On branch {branch}, update only these worker-owned paths: {owned}. "
        "Keep unrelated paths unchanged, run any assigned tests, and commit the branch."
    )


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
        if baseline_lines.count(before_line) != 1:
            return None
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
        if baseline_content.count(before_fragment) != 1:
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
    target_content: str = "",
) -> int:
    normalized_kind = str(edit_kind).strip() or "rewrite"
    if normalized_kind == "token_replace":
        ops = replacements or []
        fragment_chars = sum(len(str(item.get("before_fragment", ""))) + len(str(item.get("after_fragment", ""))) for item in ops)
        return 10 + len(ops) * 5 + fragment_chars
    if normalized_kind == "line_replace":
        ops = replacements or []
        changed_chars = sum(len(str(item.get("before_line", ""))) + len(str(item.get("after_line", ""))) for item in ops)
        return 30 + len(ops) * 12 + changed_chars
    if normalized_kind == "block_replace":
        block = replacement or {}
        before_lines = [str(line) for line in block.get("before_lines", [])]
        after_lines = [str(line) for line in block.get("after_lines", [])]
        changed_chars = sum(len(line) for line in before_lines) + sum(len(line) for line in after_lines)
        changed_lines = max(len(before_lines), len(after_lines))
        return 60 + changed_lines * 20 + changed_chars
    return 120 + len(str(target_content))


def _edit_kind_rank(edit_kind: str) -> int:
    order = {
        "token_replace": 0,
        "line_replace": 1,
        "block_replace": 2,
        "rewrite": 3,
    }
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
        before_line = str(replacement.get("before_line", ""))
        after_line = str(replacement.get("after_line", ""))
        if before_line == after_line:
            continue
        script = f"s#^{_sed_regex_escape(before_line)}$#{_sed_replacement_escape(after_line)}#"
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


def _render_token_replace_commands(path: str, replacements: list[dict[str, object]]) -> list[str]:
    commands: list[str] = []
    for replacement in replacements:
        before_fragment = str(replacement.get("before_fragment", ""))
        after_fragment = str(replacement.get("after_fragment", ""))
        if not before_fragment or before_fragment == after_fragment:
            continue
        script = f"s#{_sed_regex_escape(before_fragment)}#{_sed_replacement_escape(after_fragment)}#"
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
    replacements = {
        "pending": preferred_state,
        "broken": preferred_state,
        "todo": "done" if preferred_state == "ready" else preferred_state,
        "draft": "final" if preferred_state == "ready" else preferred_state,
        "stale": preferred_state,
        "wip": preferred_state,
    }
    for source, target in replacements.items():
        candidate, count = re.subn(rf"\b{re.escape(source)}\b", target, candidate, flags=re.IGNORECASE)
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
    for token in [*_path_tokens(branch), *_path_tokens(path)]:
        if token in {"ready", "done", "fixed", "complete", "final", "enabled"}:
            return "ready" if token == "fixed" else token
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
    report_path = f"reports/{_safe_worker_name(branch)}_report.txt"
    must_mention = ["updated", branch]
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
            {
                "benchmark_family": "skill_memory",
                "memory_source": "skill",
                "memory_source_task": source_task_id,
                "origin_benchmark_family": str(contract.metadata.get("benchmark_family", "bounded")),
                "source_task": source_task_id,
                "requires_skill_memory": True,
            }
        )
        replay_task = TaskSpec(
            task_id=f"{source_task_id}_skill_replay",
            prompt=(
                "Use the promoted reusable procedure from prior successful work to solve an equivalent task. "
                f"{contract.prompt}"
            ),
            workspace_subdir=f"{source_task_id}_skill_replay",
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
        tasks.append(replay_task)
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
            task_id=f"{task.task_id}_verifier_replay",
            extra_metadata={
                "benchmark_family": "verifier_memory",
                "memory_source": "verifier",
                "verifier_source": "episode",
                "memory_source_task": str(task.metadata.get("memory_source_task", task.task_id)),
                "source_task": str(task.metadata.get("source_task", task.task_id)),
                "origin_benchmark_family": str(
                    task.metadata.get("origin_benchmark_family", task.metadata.get("benchmark_family", "bounded"))
                ),
            },
        )
        strict_task.workspace_subdir = f"{task.workspace_subdir}_verifier_replay"
        strict_task.max_steps = max(strict_task.max_steps, len(strict_task.suggested_commands) + 1)
        tasks.append(strict_task)
        if limit is not None and len(tasks) >= limit:
            return tasks
    remaining = None if limit is None else max(0, limit - len(tasks))
    for task in load_skill_replay_tasks(skills_path, limit=remaining):
        strict_task = synthesize_stricter_task(
            task,
            task_id=f"{task.task_id}_verifier_replay",
            extra_metadata={
                "benchmark_family": "verifier_memory",
                "memory_source": "verifier",
                "verifier_source": "skill",
                "memory_source_task": str(task.metadata.get("memory_source_task", task.task_id)),
                "source_task": str(task.metadata.get("source_task", task.task_id)),
                "origin_benchmark_family": str(
                    task.metadata.get("origin_benchmark_family", task.metadata.get("benchmark_family", "bounded"))
                ),
            },
        )
        strict_task.workspace_subdir = f"{task.workspace_subdir}_verifier_replay"
        strict_task.max_steps = max(strict_task.max_steps, len(strict_task.suggested_commands) + 1)
        tasks.append(strict_task)
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
            {
                "benchmark_family": "skill_transfer",
                "memory_source": "skill_transfer",
                "memory_source_task": source_task_id,
                "origin_benchmark_family": str(target_task.metadata.get("benchmark_family", "bounded")),
                "source_task": source_task_id,
                "transfer_target_task": target_task.task_id,
            }
        )
        tasks.append(
            TaskSpec(
                task_id=f"{source_task_id}_to_{target_task.task_id}_skill_transfer",
                prompt=(
                    "Attempt cross-task transfer with the raw remembered procedure from a different task. "
                    f"{target_task.prompt}"
                ),
                workspace_subdir=f"{target_task.workspace_subdir}_skill_transfer",
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
            )
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
            {
                "benchmark_family": "operator_memory",
                "memory_source": "operator",
                "memory_source_task": ",".join(source_task_ids),
                "origin_benchmark_family": str(target_task.metadata.get("benchmark_family", "bounded")),
                "source_task": source_task_ids[0] if source_task_ids else "",
                "transfer_target_task": target_task.task_id,
                "operator_id": str(operator.get("operator_id", "")),
            }
        )
        tasks.append(
            TaskSpec(
                task_id=f"{str(operator.get('operator_id', 'operator')).replace(':', '_')}_{target_task.task_id}_operator_replay",
                prompt=(
                    "Use the induced operator class distilled from multiple successful procedures to solve a related task. "
                    f"{target_task.prompt}"
                ),
                workspace_subdir=f"{target_task.workspace_subdir}_operator_replay",
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
            )
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
            {
                "benchmark_family": "benchmark_candidate",
                "memory_source": "benchmark_candidate",
                "memory_source_task": source_task_id,
                "candidate_kind": str(proposal.get("kind", "")),
                "origin_benchmark_family": str(proposal.get("benchmark_family", metadata.get("benchmark_family", "bounded"))),
                "source_task": source_task_id,
            }
        )
        tasks.append(
            TaskSpec(
                task_id=f"{source_task_id}_benchmark_candidate",
                prompt=prompt,
                workspace_subdir=f"{source_task.workspace_subdir}_benchmark_candidate",
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
            )
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
            {
                "benchmark_family": "verifier_candidate",
                "memory_source": "verifier_candidate",
                "memory_source_task": source_task_id,
                "origin_benchmark_family": str(proposal.get("benchmark_family", metadata.get("benchmark_family", "bounded"))),
                "source_task": source_task_id,
            }
        )
        tasks.append(
            TaskSpec(
                task_id=f"{source_task_id}_verifier_candidate",
                prompt=source_task.prompt,
                workspace_subdir=f"{source_task.workspace_subdir}_verifier_candidate",
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
            )
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
        metadata = dict(contract.metadata)
        metadata.update(
            {
                "benchmark_family": "tool_memory",
                "memory_source": "tool",
                "memory_source_task": source_task_id,
                "origin_benchmark_family": str(contract.metadata.get("benchmark_family", "bounded")),
                "source_task": source_task_id,
                "requires_tool_memory": True,
            }
        )
        replay_task = TaskSpec(
            task_id=f"{source_task_id}_tool_replay",
            prompt=(
                "Use the promoted local shell procedure from prior successful work to solve an equivalent task. "
                f"{contract.prompt}"
            ),
            workspace_subdir=f"{source_task_id}_tool_replay",
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
        tasks.append(replay_task)
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
        metadata = dict(contract.get("metadata", {}))
        try:
            fallback = bank.get(task_id)
            fallback_metadata = dict(fallback.metadata)
        except KeyError:
            fallback_metadata = {}

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
        return TaskSpec(
            task_id=task_id,
            prompt=str(contract.get("prompt", payload.get("prompt", ""))),
            workspace_subdir=str(contract.get("workspace_subdir", task_id)),
            setup_commands=list(contract.get("setup_commands", [])),
            success_command=str(contract.get("success_command", "")),
            suggested_commands=list(contract.get("suggested_commands", [])),
            expected_files=list(contract.get("expected_files", [])),
            expected_output_substrings=list(contract.get("expected_output_substrings", [])),
            forbidden_files=list(contract.get("forbidden_files", [])),
            forbidden_output_substrings=list(contract.get("forbidden_output_substrings", [])),
            expected_file_contents=dict(contract.get("expected_file_contents", {})),
            max_steps=int(contract.get("max_steps", 5) or 5),
            metadata=metadata,
        )
    try:
        return bank.get(task_id)
    except KeyError:
        return None
