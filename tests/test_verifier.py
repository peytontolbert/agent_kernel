import json
from agent_kernel.config import KernelConfig
from agent_kernel.sandbox import Sandbox
from agent_kernel.schemas import CommandResult, TaskSpec
from agent_kernel.shared_repo import bootstrap_shared_repo_seed
from agent_kernel.task_bank import TaskBank
from agent_kernel.verifier import Verifier, synthesize_stricter_task
from urllib import request as url_request
import subprocess


def test_verifier_checks_files_and_output(tmp_path):
    (tmp_path / "artifact.txt").write_text("ok\n", encoding="utf-8")
    task = TaskSpec(
        task_id="test",
        prompt="verify artifact output",
        workspace_subdir="test",
        expected_files=["artifact.txt"],
        expected_output_substrings=["done"],
    )
    result = CommandResult(command="echo done", exit_code=0, stdout="done\n", stderr="")

    verification = Verifier().verify(task, tmp_path, result)

    assert verification.passed is True


def test_verifier_checks_forbidden_files_and_exact_content(tmp_path):
    (tmp_path / "final.txt").write_text("renamed content\n", encoding="utf-8")
    (tmp_path / "draft.txt").write_text("stale\n", encoding="utf-8")
    task = TaskSpec(
        task_id="rename",
        prompt="rename draft to final",
        workspace_subdir="rename",
        expected_files=["final.txt"],
        forbidden_files=["draft.txt"],
        expected_file_contents={"final.txt": "renamed content\n"},
    )
    result = CommandResult(command="mv draft.txt final.txt", exit_code=0, stdout="", stderr="")

    verification = Verifier().verify(task, tmp_path, result)

    assert verification.passed is False
    assert "forbidden file present: draft.txt" in verification.reasons


def test_verifier_checks_forbidden_output_substrings(tmp_path):
    task = TaskSpec(
        task_id="output",
        prompt="avoid warnings",
        workspace_subdir="output",
        forbidden_output_substrings=["warning"],
    )
    result = CommandResult(command="echo warning", exit_code=0, stdout="warning\n", stderr="")

    verification = Verifier().verify(task, tmp_path, result)

    assert verification.passed is False
    assert "forbidden output present: warning" in verification.reasons


def test_sandbox_blocks_privileged_commands(tmp_path):
    result = Sandbox(timeout_seconds=1).run("sudo ls", tmp_path)

    assert result.exit_code == 126
    assert "blocked privileged command pattern" in result.stderr


def test_sandbox_blocks_interactive_commands(tmp_path):
    result = Sandbox(timeout_seconds=1).run("vim notes.txt", tmp_path)

    assert result.exit_code == 126
    assert "blocked interactive command pattern" in result.stderr


def test_sandbox_blocks_unknown_host_executable(tmp_path):
    result = Sandbox(timeout_seconds=1).run("awk 'BEGIN { print 1 }'", tmp_path)

    assert result.exit_code == 126
    assert "blocked unsupported executable: awk" in result.stderr


def test_sandbox_wraps_local_commands_with_bubblewrap_when_available(monkeypatch, tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path,
        sandbox_command_containment_mode="required",
        sandbox_command_containment_tool="bwrap",
    )
    seen_argv: list[list[str]] = []

    class Completed:
        def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def fake_run(argv, **kwargs):
        del kwargs
        argv_list = [str(value) for value in argv]
        seen_argv.append(argv_list)
        if argv_list[-1] == "true":
            return Completed(0)
        return Completed(0, stdout="ready\n")

    monkeypatch.setattr("agent_kernel.sandbox.shutil.which", lambda value: "/usr/bin/bwrap" if value == "bwrap" else None)
    monkeypatch.setattr("agent_kernel.sandbox.subprocess.run", fake_run)

    result = Sandbox(timeout_seconds=1, config=config).run("printf 'ready\\n'", tmp_path)

    assert result.exit_code == 0
    assert result.stdout == "ready\n"
    assert seen_argv[0][0] == "/usr/bin/bwrap"
    assert seen_argv[1][0] == "/usr/bin/bwrap"
    assert seen_argv[1][-2:] == ["printf", "ready\\n"]


def test_sandbox_runs_bounded_chaining_and_redirection(tmp_path):
    result = Sandbox(timeout_seconds=1).run(
        "mkdir -p reports && printf 'ready\\n' > reports/status.txt && cat reports/status.txt",
        tmp_path,
    )

    assert result.exit_code == 0
    assert result.stdout == "ready\n"
    assert (tmp_path / "reports" / "status.txt").read_text(encoding="utf-8") == "ready\n"


def test_sandbox_allows_in_place_sed_workspace_edit(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "status.txt").write_text("STATUS=pending\nnotes preserved\n", encoding="utf-8")

    result = Sandbox(timeout_seconds=1).run(
        "sed -i 's/^STATUS=pending$/STATUS=ready/' src/status.txt && cat src/status.txt",
        tmp_path,
    )

    assert result.exit_code == 0
    assert result.stdout == "STATUS=ready\nnotes preserved\n"
    assert (tmp_path / "src" / "status.txt").read_text(encoding="utf-8") == "STATUS=ready\nnotes preserved\n"


def test_sandbox_blocks_unsupported_shell_operator(tmp_path):
    result = Sandbox(timeout_seconds=1).run("printf 'hi\\n' | cat", tmp_path)

    assert result.exit_code == 126
    assert "blocked unsupported shell operator: |" in result.stderr


def test_sandbox_blocks_workspace_escape_redirection(tmp_path):
    result = Sandbox(timeout_seconds=1).run("printf 'oops\\n' > ../escape.txt", tmp_path)

    assert result.exit_code == 126
    assert "blocked workspace escape path: ../escape.txt" in result.stderr


def test_sandbox_blocks_destructive_delete_without_task_contract(tmp_path):
    (tmp_path / "keep.txt").write_text("keep\n", encoding="utf-8")

    result = Sandbox(timeout_seconds=1).run("rm -f keep.txt", tmp_path)

    assert result.exit_code == 126
    assert "blocked destructive mutation without task contract" in result.stderr
    assert (tmp_path / "keep.txt").exists()


def test_sandbox_blocks_delete_of_workspace_root(tmp_path):
    task = TaskSpec(
        task_id="dangerous_cleanup",
        prompt="do not delete the workspace root",
        workspace_subdir="dangerous_cleanup",
        forbidden_files=["artifact.txt"],
    )

    result = Sandbox(timeout_seconds=1).run("rm -rf .", tmp_path, task=task)

    assert result.exit_code == 126
    assert "blocked destructive delete of workspace root" in result.stderr


def test_sandbox_runs_bounded_http_request(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path,
        unattended_allow_http_requests=True,
        unattended_http_timeout_seconds=2,
        unattended_http_max_body_bytes=1024,
    )

    class FakeResponse:
        def __init__(self):
            self.status = 200
            self.headers = {"content-type": "text/plain"}

        def read(self, limit=-1):
            del limit
            return b"hello http\n"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return False

    def fake_urlopen(req, timeout):
        assert isinstance(req, url_request.Request)
        assert req.get_method() == "GET"
        assert req.full_url == "https://example.com/health"
        assert timeout == 2
        return FakeResponse()

    result = Sandbox(timeout_seconds=1, config=config, urlopen=fake_urlopen).run(
        "http_request GET https://example.com/health > reports/health.txt",
        tmp_path,
    )

    assert result.exit_code == 0
    assert result.stdout == ""
    assert (tmp_path / "reports" / "health.txt").read_text(encoding="utf-8").startswith("status: 200\n")


def test_sandbox_blocks_http_host_outside_allowlist(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path,
        unattended_allow_http_requests=True,
        unattended_http_allowed_hosts=("api.github.com",),
    )

    result = Sandbox(timeout_seconds=1, config=config).run(
        "http_request GET https://example.com/health",
        tmp_path,
    )

    assert result.exit_code == 126
    assert "blocked http host by operator policy: example.com" in result.stderr


def test_sandbox_allows_http_host_from_enabled_module_scope(tmp_path):
    modules_path = tmp_path / "config" / "capabilities.json"
    modules_path.parent.mkdir(parents=True, exist_ok=True)
    modules_path.write_text(
        json.dumps(
            {
                "modules": [
                    {
                        "module_id": "github",
                        "enabled": True,
                        "capabilities": ["github_read"],
                        "settings": {"http_allowed_hosts": ["api.github.com"]},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path,
        capability_modules_path=modules_path,
        unattended_allow_http_requests=True,
    )

    class FakeResponse:
        def __init__(self):
            self.status = 200
            self.headers = {"content-type": "application/json"}

        def read(self, limit=-1):
            del limit
            return b"{}\n"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return False

    result = Sandbox(timeout_seconds=1, config=config, urlopen=lambda req, timeout: FakeResponse()).run(
        "http_request GET https://api.github.com/repos/openai/openai-python > reports/repo.json",
        tmp_path,
    )

    assert result.exit_code == 0
    assert (tmp_path / "reports" / "repo.json").exists()


def test_sandbox_blocks_unmanaged_delete_path_for_task_contract(tmp_path):
    task = TaskSpec(
        task_id="cleanup_task",
        prompt="remove temp and write status",
        workspace_subdir="cleanup_task",
        expected_files=["status.txt"],
        forbidden_files=["temp.txt"],
    )
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path,
    )

    result = Sandbox(timeout_seconds=1, config=config).run("rm -f stray.txt", tmp_path, task=task)

    assert result.exit_code == 126
    assert "blocked unmanaged delete path for task contract: stray.txt" in result.stderr


def test_sandbox_blocks_git_by_operator_policy(tmp_path):
    task = TaskSpec(
        task_id="repo_task",
        prompt="inspect repo",
        workspace_subdir="repo_task",
        expected_files=["reports/review.txt"],
        metadata={"benchmark_family": "repo_chore"},
    )
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path,
        unattended_allow_git_commands=False,
    )

    result = Sandbox(timeout_seconds=1, config=config).run("git status", tmp_path, task=task)

    assert result.exit_code == 126
    assert "blocked unattended git command by operator policy" in result.stderr


def test_sandbox_uses_retained_operator_policy_for_generated_path_mutation(tmp_path):
    operator_policy_path = tmp_path / "operator_policy" / "operator_policy_proposals.json"
    operator_policy_path.parent.mkdir(parents=True, exist_ok=True)
    operator_policy_path.write_text(
        json.dumps(
            {
                "artifact_kind": "operator_policy_set",
                "lifecycle_state": "retained",
                "controls": {
                    "unattended_allowed_benchmark_families": ["micro"],
                    "unattended_allow_git_commands": False,
                    "unattended_allow_http_requests": False,
                    "unattended_http_allowed_hosts": [],
                    "unattended_http_timeout_seconds": 10,
                    "unattended_http_max_body_bytes": 65536,
                    "unattended_allow_generated_path_mutations": True,
                    "unattended_generated_path_prefixes": ["build"],
                },
            }
        ),
        encoding="utf-8",
    )
    task = TaskSpec(
        task_id="generated_bundle_task",
        prompt="write generated output",
        workspace_subdir="generated_bundle_task",
        expected_files=["build/output.txt"],
        metadata={"benchmark_family": "micro"},
    )
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path,
        unattended_allow_generated_path_mutations=False,
        operator_policy_proposals_path=operator_policy_path,
    )

    result = Sandbox(timeout_seconds=1, config=config).run(
        "mkdir -p build && printf 'ready\\n' > build/output.txt",
        tmp_path,
        task=task,
    )

    assert result.exit_code == 0
    assert (tmp_path / "build" / "output.txt").read_text(encoding="utf-8") == "ready\n"


def test_sandbox_allows_workspace_relative_executable(tmp_path):
    script = tmp_path / "tests" / "check_status.sh"
    script.parent.mkdir(parents=True)
    script.write_text("#!/bin/sh\nprintf 'ok\\n'\n", encoding="utf-8")
    script.chmod(0o755)
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path,
        unattended_allow_git_commands=True,
    )

    result = Sandbox(timeout_seconds=1, config=config).run("tests/check_status.sh", tmp_path)

    assert result.exit_code == 0
    assert result.stdout == "ok\n"


def test_sandbox_blocks_write_outside_shared_repo_claim(tmp_path):
    task = TaskSpec(
        task_id="shared_repo_worker",
        prompt="touch owned path only",
        workspace_subdir="shared_repo_worker",
        expected_files=["src/api_status.txt", "docs/status.md"],
        metadata={
            "workflow_guard": {
                "claimed_paths": ["src/api_status.txt"],
            }
        },
    )
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path,
        unattended_allow_git_commands=True,
    )

    result = Sandbox(timeout_seconds=1, config=config).run(
        "printf 'ready\\n' > docs/status.md",
        tmp_path,
        task=task,
    )

    assert result.exit_code == 126
    assert "blocked out-of-claim write path for shared repo worker: docs/status.md" in result.stderr


def test_sandbox_blocks_sed_write_outside_shared_repo_claim(tmp_path):
    task = TaskSpec(
        task_id="shared_repo_worker",
        prompt="touch owned path only",
        workspace_subdir="shared_repo_worker",
        expected_files=["src/api_status.txt", "docs/status.md"],
        metadata={
            "workflow_guard": {
                "claimed_paths": ["src/api_status.txt"],
            }
        },
    )
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path,
        unattended_allow_git_commands=True,
    )
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "status.md").write_text("status pending documented\n", encoding="utf-8")

    result = Sandbox(timeout_seconds=1, config=config).run(
        "sed -i 's/^status pending documented$/status ready documented/' docs/status.md",
        tmp_path,
        task=task,
    )

    assert result.exit_code == 126
    assert "blocked out-of-claim write path for shared repo worker: docs/status.md" in result.stderr


def test_synthesized_stricter_task_adds_nested_forbidden_paths():
    task = TaskSpec(
        task_id="hello_task",
        prompt="Create hello.txt containing hello agent kernel.",
        workspace_subdir="hello_task",
        expected_files=["hello.txt"],
        expected_file_contents={"hello.txt": "hello agent kernel\n"},
    )

    strict = synthesize_stricter_task(task, task_id="hello_task_verifier_replay")

    assert strict.metadata["benchmark_family"] == "verifier_memory"
    assert strict.metadata["memory_source"] == "verifier"
    assert "hello_task/hello.txt" in strict.forbidden_files


def test_verifier_applies_semantic_repo_chore_review_checks(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "reports").mkdir()
    (tmp_path / "docs" / "context.md").write_text("repo context\n", encoding="utf-8")
    (tmp_path / "src" / "app.py").write_text("STATUS=ready\n", encoding="utf-8")
    (tmp_path / "tests" / "status_check.txt").write_text("status ready covered\n", encoding="utf-8")
    (tmp_path / "reports" / "diff_summary.txt").write_text(
        "updated src/app.py and tests/status_check.txt\n",
        encoding="utf-8",
    )
    (tmp_path / "reports" / "verification.txt").write_text(
        "deterministic checks passed for docs/context.md preservation\n",
        encoding="utf-8",
    )
    task = TaskSpec(
        task_id="repo_patch_review_task",
        prompt="prepare repo review packet",
        workspace_subdir="repo_patch_review_task",
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
        },
        metadata={
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
            }
        },
    )
    result = CommandResult(command="true", exit_code=0, stdout="", stderr="")

    verification = Verifier().verify(task, tmp_path, result)

    assert verification.passed is True


def test_verifier_rejects_semantically_incomplete_repo_chore_review(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "reports").mkdir()
    (tmp_path / "docs" / "context.md").write_text("repo context\n", encoding="utf-8")
    (tmp_path / "src" / "app.py").write_text("STATUS=ready\n", encoding="utf-8")
    (tmp_path / "tests" / "status_check.txt").write_text("status ready covered\n", encoding="utf-8")
    (tmp_path / "reports" / "diff_summary.txt").write_text("updated src/app.py\n", encoding="utf-8")
    (tmp_path / "reports" / "verification.txt").write_text("checks passed\n", encoding="utf-8")
    task = TaskSpec(
        task_id="repo_patch_review_task",
        prompt="prepare repo review packet",
        workspace_subdir="repo_patch_review_task",
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
        },
        metadata={
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
            }
        },
    )
    result = CommandResult(command="true", exit_code=0, stdout="", stderr="")

    verification = Verifier().verify(task, tmp_path, result)

    assert verification.passed is False
    assert "semantic report does not cover tests/status_check.txt: reports/diff_summary.txt" in verification.reasons
    assert "semantic report missing phrase 'preservation': reports/verification.txt" in verification.reasons


def test_verifier_applies_git_repo_review_checks(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "reports").mkdir()
    (tmp_path / "docs" / "context.md").write_text("repo context\n", encoding="utf-8")
    (tmp_path / "src" / "app.py").write_text("STATUS=pending\n", encoding="utf-8")
    (tmp_path / "tests" / "status_check.txt").write_text("status pending covered\n", encoding="utf-8")
    (tmp_path / "tests" / "check_status.sh").write_text(
        "#!/bin/sh\nset -eu\ngrep -q \"^STATUS=ready$\" src/app.py\ngrep -q \"^status ready covered$\" tests/status_check.txt\n",
        encoding="utf-8",
    )
    (tmp_path / "tests" / "check_status.sh").chmod(0o755)
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.email", "agent@example.com"], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.name", "Agent Kernel"], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "add", "docs/context.md", "src/app.py", "tests/status_check.txt", "tests/check_status.sh"], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "commit", "-m", "baseline"], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "checkout", "-b", "review/status-ready"], cwd=tmp_path, check=True, capture_output=True, text=True)
    (tmp_path / "src" / "app.py").write_text("STATUS=ready\n", encoding="utf-8")
    (tmp_path / "tests" / "status_check.txt").write_text("status ready covered\n", encoding="utf-8")
    (tmp_path / "reports" / "diff_summary.txt").write_text(
        "updated src/app.py, tests/status_check.txt, and reports/test_report.txt on branch review/status-ready\n",
        encoding="utf-8",
    )
    (tmp_path / "reports" / "test_report.txt").write_text("status check passed\n", encoding="utf-8")
    task = TaskSpec(
        task_id="git_repo_status_review_task",
        prompt="prepare git repo review packet",
        workspace_subdir="git_repo_status_review_task",
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
        metadata={
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
                    {"label": "status check script", "argv": ["tests/check_status.sh"]},
                ],
                "report_rules": [
                    {
                        "path": "reports/diff_summary.txt",
                        "must_mention": ["updated", "review/status-ready"],
                        "covers": ["src/app.py", "tests/status_check.txt", "reports/test_report.txt"],
                    },
                    {
                        "path": "reports/test_report.txt",
                        "must_mention": ["status", "check", "passed"],
                        "covers": ["tests/check_status.sh"],
                    },
                ],
            }
        },
    )
    result = CommandResult(command="git diff --name-only", exit_code=0, stdout="", stderr="")

    verification = Verifier().verify(task, tmp_path, result)

    assert verification.passed is True


def test_verifier_rejects_git_repo_review_mismatches(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "reports").mkdir()
    (tmp_path / "docs" / "context.md").write_text("repo context\n", encoding="utf-8")
    (tmp_path / "src" / "app.py").write_text("STATUS=pending\n", encoding="utf-8")
    (tmp_path / "tests" / "status_check.txt").write_text("status pending covered\n", encoding="utf-8")
    (tmp_path / "tests" / "check_status.sh").write_text(
        "#!/bin/sh\nset -eu\ngrep -q \"^STATUS=ready$\" src/app.py\ngrep -q \"^status ready covered$\" tests/status_check.txt\n",
        encoding="utf-8",
    )
    (tmp_path / "tests" / "check_status.sh").chmod(0o755)
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.email", "agent@example.com"], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.name", "Agent Kernel"], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "add", "docs/context.md", "src/app.py", "tests/status_check.txt", "tests/check_status.sh"], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "commit", "-m", "baseline"], cwd=tmp_path, check=True, capture_output=True, text=True)
    (tmp_path / "reports" / "diff_summary.txt").write_text("updated src/app.py\n", encoding="utf-8")
    (tmp_path / "reports" / "test_report.txt").write_text("status check passed\n", encoding="utf-8")
    task = TaskSpec(
        task_id="git_repo_status_review_task",
        prompt="prepare git repo review packet",
        workspace_subdir="git_repo_status_review_task",
        expected_files=["reports/diff_summary.txt", "reports/test_report.txt"],
        metadata={
            "semantic_verifier": {
                "kind": "git_repo_review",
                "expected_branch": "review/status-ready",
                "expected_changed_paths": [
                    "reports/diff_summary.txt",
                    "reports/test_report.txt",
                    "src/app.py",
                ],
                "preserved_paths": ["docs/context.md"],
                "test_commands": [
                    {"label": "status check script", "argv": ["tests/check_status.sh"]},
                ],
                "report_rules": [
                    {
                        "path": "reports/diff_summary.txt",
                        "must_mention": ["updated", "review/status-ready"],
                        "covers": ["src/app.py", "reports/test_report.txt"],
                    }
                ],
            }
        },
    )
    result = CommandResult(command="git diff --name-only", exit_code=0, stdout="", stderr="")

    verification = Verifier().verify(task, tmp_path, result)

    assert verification.passed is False
    assert any("git branch mismatch" in reason for reason in verification.reasons)
    assert any("status check script exited with code" in reason for reason in verification.reasons)


def test_verifier_accepts_git_repo_test_repair_workflow(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "reports").mkdir()
    (tmp_path / "docs" / "notes.md").write_text("release notes preserved\n", encoding="utf-8")
    (tmp_path / "src" / "release_state.txt").write_text("RELEASE_STATUS=broken\n", encoding="utf-8")
    (tmp_path / "tests" / "test_release.sh").write_text(
        "#!/bin/sh\nset -eu\ngrep -q \"^RELEASE_STATUS=ready$\" src/release_state.txt\n",
        encoding="utf-8",
    )
    (tmp_path / "tests" / "test_release.sh").chmod(0o755)
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.email", "agent@example.com"], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.name", "Agent Kernel"], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "add", "docs/notes.md", "src/release_state.txt", "tests/test_release.sh"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(["git", "commit", "-m", "baseline"], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "checkout", "-b", "fix/release-ready"], cwd=tmp_path, check=True, capture_output=True, text=True)
    (tmp_path / "src" / "release_state.txt").write_text("RELEASE_STATUS=ready\n", encoding="utf-8")
    (tmp_path / "reports" / "diff_summary.txt").write_text(
        "repaired failing deterministic release test by updating src/release_state.txt on branch fix/release-ready\n",
        encoding="utf-8",
    )
    (tmp_path / "reports" / "test_report.txt").write_text("release test passed\n", encoding="utf-8")
    task = TaskBank().get("git_repo_test_repair_task")
    result = CommandResult(command="git diff --name-only", exit_code=0, stdout="", stderr="")

    verification = Verifier().verify(task, tmp_path, result)

    assert verification.passed is True


def test_verifier_accepts_parallel_merge_repo_workflow(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path,
        unattended_allow_git_commands=True,
    )
    sandbox = Sandbox(timeout_seconds=10, config=config)
    bank = TaskBank()
    worker_api = bank.get("git_parallel_worker_api_task")
    worker_docs = bank.get("git_parallel_worker_docs_task")
    task = bank.get("git_parallel_merge_acceptance_task")
    bootstrap_shared_repo_seed(worker_api, workspace=tmp_path, config=config)
    subprocess.run(["git", "checkout", "-b", "worker/api-status"], cwd=tmp_path, check=True, capture_output=True, text=True)
    worker_api_result = sandbox.run(worker_api.suggested_commands[0], tmp_path, task=worker_api)
    assert worker_api_result.exit_code == 0, worker_api_result.stderr
    subprocess.run(["git", "checkout", "main"], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "checkout", "-b", "worker/docs-status"], cwd=tmp_path, check=True, capture_output=True, text=True)
    worker_docs_result = sandbox.run(worker_docs.suggested_commands[0], tmp_path, task=worker_docs)
    assert worker_docs_result.exit_code == 0, worker_docs_result.stderr
    subprocess.run(["git", "checkout", "main"], cwd=tmp_path, check=True, capture_output=True, text=True)
    result = sandbox.run(task.suggested_commands[0], tmp_path, task=task)

    verification = Verifier().verify(task, tmp_path, result)

    assert verification.passed is True


def test_verifier_accepts_release_train_repo_workflow(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path,
        unattended_allow_git_commands=True,
    )
    sandbox = Sandbox(timeout_seconds=10, config=config)
    bank = TaskBank()
    worker_api = bank.get("git_release_train_worker_api_task")
    worker_docs = bank.get("git_release_train_worker_docs_task")
    worker_ops = bank.get("git_release_train_worker_ops_task")
    task = bank.get("git_release_train_acceptance_task")
    bootstrap_shared_repo_seed(worker_api, workspace=tmp_path, config=config)
    subprocess.run(["git", "checkout", "-b", "worker/api-cutover"], cwd=tmp_path, check=True, capture_output=True, text=True)
    worker_api_result = sandbox.run(worker_api.suggested_commands[0], tmp_path, task=worker_api)
    assert worker_api_result.exit_code == 0, worker_api_result.stderr
    subprocess.run(["git", "checkout", "main"], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "checkout", "-b", "worker/docs-cutover"], cwd=tmp_path, check=True, capture_output=True, text=True)
    worker_docs_result = sandbox.run(worker_docs.suggested_commands[0], tmp_path, task=worker_docs)
    assert worker_docs_result.exit_code == 0, worker_docs_result.stderr
    subprocess.run(["git", "checkout", "main"], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "checkout", "-b", "worker/ops-cutover"], cwd=tmp_path, check=True, capture_output=True, text=True)
    worker_ops_result = sandbox.run(worker_ops.suggested_commands[0], tmp_path, task=worker_ops)
    assert worker_ops_result.exit_code == 0, worker_ops_result.stderr
    subprocess.run(["git", "checkout", "main"], cwd=tmp_path, check=True, capture_output=True, text=True)
    result = sandbox.run(task.suggested_commands[0], tmp_path, task=task)

    verification = Verifier().verify(task, tmp_path, result)

    assert verification.passed is True


def test_verifier_accepts_release_train_conflict_repo_workflow(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path,
        unattended_allow_git_commands=True,
        unattended_allow_generated_path_mutations=True,
    )
    sandbox = Sandbox(timeout_seconds=10, config=config)
    bank = TaskBank()
    worker_api = bank.get("git_release_train_conflict_worker_api_task")
    worker_docs = bank.get("git_release_train_conflict_worker_docs_task")
    worker_ops = bank.get("git_release_train_conflict_worker_ops_task")
    task = bank.get("git_release_train_conflict_acceptance_task")
    bootstrap_shared_repo_seed(worker_api, workspace=tmp_path, config=config)
    subprocess.run(["git", "checkout", "-b", "worker/api-release"], cwd=tmp_path, check=True, capture_output=True, text=True)
    worker_api_result = sandbox.run(worker_api.suggested_commands[0], tmp_path, task=worker_api)
    assert worker_api_result.exit_code == 0, worker_api_result.stderr
    subprocess.run(["git", "checkout", "main"], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "checkout", "-b", "worker/docs-release"], cwd=tmp_path, check=True, capture_output=True, text=True)
    worker_docs_result = sandbox.run(worker_docs.suggested_commands[0], tmp_path, task=worker_docs)
    assert worker_docs_result.exit_code == 0, worker_docs_result.stderr
    subprocess.run(["git", "checkout", "main"], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "checkout", "-b", "worker/ops-release"], cwd=tmp_path, check=True, capture_output=True, text=True)
    worker_ops_result = sandbox.run(worker_ops.suggested_commands[0], tmp_path, task=worker_ops)
    assert worker_ops_result.exit_code == 0, worker_ops_result.stderr
    subprocess.run(["git", "checkout", "main"], cwd=tmp_path, check=True, capture_output=True, text=True)
    result = sandbox.run(task.suggested_commands[0], tmp_path, task=task)

    verification = Verifier().verify(task, tmp_path, result)

    assert verification.passed is True


def test_verifier_accepts_parallel_merge_repo_workflow_with_required_branch_artifacts(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path,
        unattended_allow_git_commands=True,
    )
    sandbox = Sandbox(timeout_seconds=10, config=config)
    bank = TaskBank()
    worker_api = bank.get("git_parallel_worker_api_task")
    worker_docs = bank.get("git_parallel_worker_docs_task")
    task = bank.get("git_parallel_merge_acceptance_task")
    bootstrap_shared_repo_seed(worker_api, workspace=tmp_path, config=config)
    subprocess.run(["git", "checkout", "-b", "worker/api-status"], cwd=tmp_path, check=True, capture_output=True, text=True)
    worker_api_result = sandbox.run(worker_api.suggested_commands[0], tmp_path, task=worker_api)
    assert worker_api_result.exit_code == 0, worker_api_result.stderr
    (tmp_path / "reports").mkdir(exist_ok=True)
    (tmp_path / "reports" / "worker_api-status_report.txt").write_text(
        "worker/api-status updated src/api_status.txt\n",
        encoding="utf-8",
    )
    subprocess.run(
        ["git", "add", "reports/worker_api-status_report.txt"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "record worker api report"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(["git", "checkout", "main"], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "checkout", "-b", "worker/docs-status"], cwd=tmp_path, check=True, capture_output=True, text=True)
    worker_docs_result = sandbox.run(worker_docs.suggested_commands[0], tmp_path, task=worker_docs)
    assert worker_docs_result.exit_code == 0, worker_docs_result.stderr
    (tmp_path / "reports").mkdir(exist_ok=True)
    (tmp_path / "reports" / "worker_docs-status_report.txt").write_text(
        "worker/docs-status updated docs/status.md\n",
        encoding="utf-8",
    )
    subprocess.run(
        ["git", "add", "reports/worker_docs-status_report.txt"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "record worker docs report"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(["git", "checkout", "main"], cwd=tmp_path, check=True, capture_output=True, text=True)
    result = sandbox.run(task.suggested_commands[0], tmp_path, task=task)

    verification = Verifier().verify(task, tmp_path, result)

    assert verification.passed is True


def test_verifier_rejects_unresolved_generated_conflict_repo_workflow(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path,
        unattended_allow_git_commands=True,
        unattended_allow_generated_path_mutations=True,
    )
    sandbox = Sandbox(timeout_seconds=10, config=config)
    bank = TaskBank()
    worker = bank.get("git_conflict_worker_status_task")
    task = bank.get("git_generated_conflict_resolution_task")
    bootstrap_shared_repo_seed(worker, workspace=tmp_path, config=config)
    subprocess.run(["git", "checkout", "-b", "worker/status-refresh"], cwd=tmp_path, check=True, capture_output=True, text=True)
    worker_result = sandbox.run(worker.suggested_commands[0], tmp_path, task=worker)
    assert worker_result.exit_code == 0, worker_result.stderr
    subprocess.run(["git", "checkout", "main"], cwd=tmp_path, check=True, capture_output=True, text=True)
    (tmp_path / "src" / "shared_status.txt").write_text("SERVICE_STATUS=mainline-ready\n", encoding="utf-8")
    subprocess.run(["git", "add", "src/shared_status.txt"], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "commit", "-m", "mainline status change"], cwd=tmp_path, check=True, capture_output=True, text=True)
    merge_result = subprocess.run(
        ["git", "merge", "--no-ff", "worker/status-refresh", "-m", "merge worker/status-refresh"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert merge_result.returncode != 0
    (tmp_path / "reports").mkdir(exist_ok=True)
    (tmp_path / "reports" / "merge_report.txt").write_text(
        "resolved worker/status-refresh merge conflict on src/shared_status.txt before acceptance into main\n",
        encoding="utf-8",
    )
    (tmp_path / "reports" / "test_report.txt").write_text(
        "service suite passed; bundle suite passed\n",
        encoding="utf-8",
    )
    result = CommandResult(command="git merge worker/status-refresh", exit_code=1, stdout="", stderr="")

    verification = Verifier().verify(task, tmp_path, result)

    assert verification.passed is False
    assert any("git conflict remains unresolved" in reason for reason in verification.reasons)


def test_verifier_accepts_generated_conflict_repo_workflow_when_baseline_ref_is_missing(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path,
        unattended_allow_git_commands=True,
        unattended_allow_generated_path_mutations=True,
    )
    sandbox = Sandbox(timeout_seconds=10, config=config)
    bank = TaskBank()
    worker = bank.get("git_conflict_worker_status_task")
    task = bank.get("git_generated_conflict_resolution_task")
    bootstrap_shared_repo_seed(worker, workspace=tmp_path, config=config)
    subprocess.run(["git", "tag", "-d", "baseline"], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "checkout", "-b", "worker/status-refresh"], cwd=tmp_path, check=True, capture_output=True, text=True)
    worker_result = sandbox.run(worker.suggested_commands[0], tmp_path, task=worker)
    assert worker_result.exit_code == 0, worker_result.stderr
    subprocess.run(["git", "checkout", "main"], cwd=tmp_path, check=True, capture_output=True, text=True)
    result = sandbox.run(task.suggested_commands[0], tmp_path, task=task)

    verification = Verifier().verify(task, tmp_path, result)

    assert verification.passed is True
