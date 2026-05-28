import json
from agent_kernel.config import KernelConfig
from agent_kernel.sandbox import Sandbox
from agent_kernel.schemas import CommandResult, TaskSpec
from agent_kernel.ops.shared_repo import bootstrap_shared_repo_seed
from agent_kernel.tasking.task_bank import TaskBank
from agent_kernel.verifier import (
    Verifier,
    _introduced_python_unresolved_name_loads,
    _is_disallowed_swe_solution_path,
    _python_exception_contract_regression_details,
    _python_annotation_only_changed,
    _python_indentation_only_statement_moves,
    _python_introduced_none_return_value_paths,
    _python_introduced_none_container_misuse_details,
    _python_introduced_function_object_arithmetic_details,
    _python_introduced_local_call_arity_mismatch_details,
    _python_introduced_local_call_keyword_mismatch_details,
    _python_introduced_private_attribute_accesses,
    _python_introduced_unknown_self_private_attribute_accesses,
    _python_introduced_self_recursive_property_accesses,
    _python_nested_assignment_replacement_details,
    _python_nested_duplicate_reducer_details,
    _python_removed_return_value_paths,
    _python_redundant_decorated_normalization_details,
    _removed_python_module_state_assignment_names,
    _python_string_literal_only_changed,
    _python_suspicious_attribute_replacement_details,
    _python_suspicious_line_replacement_details,
    _python_suspicious_boolean_return_flip_names,
    _python_duplicate_surrounding_call_wrapper_details,
    _python_duplicate_existing_statement_replacement_details,
    _python_suspicious_hunk_replacement_details,
    _suspicious_text_template_replacement_details,
    _removed_python_module_registration_names,
    _suspicious_config_key_replacement_details,
    _suspicious_semantic_token_flip_details,
    synthesize_stricter_task,
)
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


def test_python_module_state_assignment_removal_and_unresolved_module_load_are_detected():
    before = 'registry: dict[str, object] = {"x": object()}\n'
    after = 'schema=subschema.get("schema", {})\n'

    assert _removed_python_module_state_assignment_names(before, after) == ["registry"]
    assert _introduced_python_unresolved_name_loads(before, after) == ["<module>.subschema"]


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


def test_verifier_enforces_success_command(tmp_path):
    (tmp_path / "patch.diff").write_text("diff --git a/right.py b/right.py\n", encoding="utf-8")
    task = TaskSpec(
        task_id="success_command",
        prompt="verify success command",
        workspace_subdir="success_command",
        success_command="grep -q 'missing.py' patch.diff",
        expected_files=["patch.diff"],
    )
    result = CommandResult(command="write patch", exit_code=0, stdout="", stderr="")

    verification = Verifier().verify(task, tmp_path, result)

    assert verification.passed is False
    assert "success command exited with code 1" in verification.reasons
    assert any(item["kind"] == "success_command_result" for item in verification.evidence)


def test_verifier_success_command_can_pass(tmp_path):
    (tmp_path / "patch.diff").write_text("diff --git a/right.py b/right.py\n", encoding="utf-8")
    task = TaskSpec(
        task_id="success_command",
        prompt="verify success command",
        workspace_subdir="success_command",
        success_command="grep -q 'right.py' patch.diff",
        expected_files=["patch.diff"],
    )
    result = CommandResult(command="write patch", exit_code=0, stdout="", stderr="")

    verification = Verifier().verify(task, tmp_path, result)

    assert verification.passed is True


def test_verifier_swe_patch_apply_check(tmp_path):
    repo_root = tmp_path / "repos" / "owner" / "repo"
    source = repo_root / "pkg" / "module.py"
    source.parent.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_root, check=True)
    source.write_text("def value(value):\n    return value\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    source.write_text("def value(value):\n    if value is None:\n        return 0\n    return value\n", encoding="utf-8")
    patch = subprocess.check_output(["git", "diff"], cwd=repo_root, text=True)
    subprocess.run(["git", "checkout", "--", "pkg/module.py"], cwd=repo_root, check=True)
    (tmp_path / "workspace").mkdir()
    (tmp_path / "workspace" / "patch.diff").write_text(patch, encoding="utf-8")
    task = TaskSpec(
        task_id="swe",
        prompt="verify swe patch",
        workspace_subdir="workspace",
        expected_files=["patch.diff"],
        metadata={
            "semantic_verifier": {
                "kind": "swe_patch_apply_check",
                "repo": "owner/repo",
                "base_commit": commit,
                "repo_cache_root": str(tmp_path / "repos"),
                "patch_path": "patch.diff",
            }
        },
    )

    verification = Verifier().verify(
        task,
        tmp_path / "workspace",
        CommandResult(command="write", exit_code=0, stdout="", stderr=""),
    )

    assert verification.passed is True


def test_verifier_swe_patch_apply_check_rejects_isolated_one_line_production_replacement(tmp_path):
    repo_root = tmp_path / "repos" / "owner" / "repo"
    source = repo_root / "pkg" / "module.py"
    source.parent.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_root, check=True)
    source.write_text("def value(x):\n    return x > 0\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    source.write_text("def value(x):\n    return x >= 0\n", encoding="utf-8")
    patch = subprocess.check_output(["git", "diff"], cwd=repo_root, text=True)
    (tmp_path / "workspace").mkdir()
    (tmp_path / "workspace" / "patch.diff").write_text(patch, encoding="utf-8")
    task = TaskSpec(
        task_id="swe",
        prompt="verify swe patch",
        workspace_subdir="workspace",
        expected_files=["patch.diff"],
        metadata={
            "semantic_verifier": {
                "kind": "swe_patch_apply_check",
                "repo": "owner/repo",
                "base_commit": commit,
                "repo_cache_root": str(tmp_path / "repos"),
                "patch_path": "patch.diff",
            }
        },
    )

    verification = Verifier().verify(
        task,
        tmp_path / "workspace",
        CommandResult(command="write", exit_code=0, stdout="", stderr=""),
    )

    assert verification.passed is False
    assert any("isolated one-line production Python replacement" in reason for reason in verification.reasons)


def test_verifier_swe_patch_apply_check_rejects_removed_production_call_statement(tmp_path):
    repo_root = tmp_path / "repos" / "owner" / "repo"
    source = repo_root / "pkg" / "module.py"
    source.parent.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_root, check=True)
    source.write_text(
        "from asyncio import Event, Lock\n\n"
        "class Dispatcher:\n"
        "    async def start(self, bot, workflow_data):\n"
        "        await self.emit_startup(bot=bot, **workflow_data)\n"
        "        self.running = True\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    source.write_text(
        "from asyncio import Event, Lock\n\n"
        "class Dispatcher:\n"
        "    async def start(self, bot, workflow_data):\n"
        "        self._running_lock = Lock()\n"
        "        self._stop_signal = None\n"
        "        self._stopped_signal = None\n"
        "        self.running = True\n",
        encoding="utf-8",
    )
    patch = subprocess.check_output(["git", "diff"], cwd=repo_root, text=True)
    (tmp_path / "workspace").mkdir()
    (tmp_path / "workspace" / "patch.diff").write_text(patch, encoding="utf-8")
    task = TaskSpec(
        task_id="swe",
        prompt="verify swe patch",
        workspace_subdir="workspace",
        expected_files=["patch.diff"],
        metadata={
            "semantic_verifier": {
                "kind": "swe_patch_apply_check",
                "repo": "owner/repo",
                "base_commit": commit,
                "repo_cache_root": str(tmp_path / "repos"),
                "patch_path": "patch.diff",
            }
        },
    )

    verification = Verifier().verify(
        task,
        tmp_path / "workspace",
        CommandResult(command="write", exit_code=0, stdout="", stderr=""),
    )

    assert verification.passed is False
    assert any("removes production call statement" in reason for reason in verification.reasons)


def test_verifier_swe_patch_apply_check_rejects_removed_init_instance_assignment(tmp_path):
    repo_root = tmp_path / "repos" / "owner" / "repo"
    source = repo_root / "pkg" / "module.py"
    source.parent.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_root, check=True)
    source.write_text(
        "from typing import TypeAlias\n\n"
        "class Formatter:\n"
        "    def __init__(self, pattern, format):\n"
        "        self.pattern = pattern\n"
        "        self.format = format\n"
        "    def __repr__(self):\n"
        "        return self.pattern\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    source.write_text(
        "from typing import TypeAlias\n\n"
        "class Formatter:\n"
        "    def __init__(self, pattern, format):\n"
        "        self.pattern = pattern\n"
        "    _Context: TypeAlias = str\n"
        "    _Other: TypeAlias = int\n"
        "    def __repr__(self):\n"
        "        return self.pattern\n",
        encoding="utf-8",
    )
    patch = subprocess.check_output(["git", "diff"], cwd=repo_root, text=True)
    (tmp_path / "workspace").mkdir()
    (tmp_path / "workspace" / "patch.diff").write_text(patch, encoding="utf-8")
    task = TaskSpec(
        task_id="swe",
        prompt="verify swe patch",
        workspace_subdir="workspace",
        expected_files=["patch.diff"],
        metadata={
            "semantic_verifier": {
                "kind": "swe_patch_apply_check",
                "repo": "owner/repo",
                "base_commit": commit,
                "repo_cache_root": str(tmp_path / "repos"),
                "patch_path": "patch.diff",
            }
        },
    )

    verification = Verifier().verify(
        task,
        tmp_path / "workspace",
        CommandResult(command="write", exit_code=0, stdout="", stderr=""),
    )

    assert verification.passed is False
    assert any("removes constructor instance assignments" in reason for reason in verification.reasons)
    assert any("Formatter.__init__.self.format" in reason for reason in verification.reasons)


def test_verifier_swe_patch_apply_check_rejects_non_executable_context_replaced_with_code(tmp_path):
    repo_root = tmp_path / "repos" / "owner" / "repo"
    source = repo_root / "pkg" / "module.py"
    source.parent.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_root, check=True)
    source.write_text(
        "def value(locale):\n"
        "    # TODO: support locale\n"
        "    return 'ok'\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    source.write_text(
        "def value(locale):\n"
        "    locale_obj = Locale.parse(locale)\n"
        "    return 'ok'\n",
        encoding="utf-8",
    )
    patch = subprocess.check_output(["git", "diff"], cwd=repo_root, text=True)
    (tmp_path / "workspace").mkdir()
    (tmp_path / "workspace" / "patch.diff").write_text(patch, encoding="utf-8")
    task = TaskSpec(
        task_id="swe",
        prompt="verify swe patch",
        workspace_subdir="workspace",
        expected_files=["patch.diff"],
        metadata={
            "semantic_verifier": {
                "kind": "swe_patch_apply_check",
                "repo": "owner/repo",
                "base_commit": commit,
                "repo_cache_root": str(tmp_path / "repos"),
                "patch_path": "patch.diff",
            }
        },
    )

    verification = Verifier().verify(
        task,
        tmp_path / "workspace",
        CommandResult(command="write", exit_code=0, stdout="", stderr=""),
    )

    assert verification.passed is False
    assert any("replaces only non-executable production context with executable code" in reason for reason in verification.reasons)


def test_verifier_swe_patch_apply_check_rejects_doc_text_replaced_with_bare_pass(tmp_path):
    repo_root = tmp_path / "repos" / "owner" / "repo"
    source = repo_root / "pkg" / "module.py"
    source.parent.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_root, check=True)
    source.write_text(
        '"""\n'
        "Calculate the solar position using the NREL SPA algorithm.\n"
        '"""\n'
        "VALUE = 1\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    source.write_text("pass\nVALUE = 1\n", encoding="utf-8")
    patch = subprocess.check_output(["git", "diff"], cwd=repo_root, text=True)
    (tmp_path / "workspace").mkdir()
    (tmp_path / "workspace" / "patch.diff").write_text(patch, encoding="utf-8")
    task = TaskSpec(
        task_id="swe",
        prompt="verify swe patch",
        workspace_subdir="workspace",
        expected_files=["patch.diff"],
        metadata={
            "semantic_verifier": {
                "kind": "swe_patch_apply_check",
                "repo": "owner/repo",
                "base_commit": commit,
                "repo_cache_root": str(tmp_path / "repos"),
                "patch_path": "patch.diff",
            }
        },
    )

    verification = Verifier().verify(
        task,
        tmp_path / "workspace",
        CommandResult(command="write", exit_code=0, stdout="", stderr=""),
    )

    assert verification.passed is False
    assert any("documentation-only production context with bare pass" in reason for reason in verification.reasons)


def test_verifier_swe_patch_apply_check_rejects_local_call_arity_hallucination(tmp_path):
    repo_root = tmp_path / "repos" / "owner" / "repo"
    source = repo_root / "pkg" / "module.py"
    helper = repo_root / "pkg" / "helper.py"
    source.parent.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_root, check=True)
    helper.write_text("def calculate_deltat(year, month):\n    return year + month\n", encoding="utf-8")
    source.write_text(
        "from pkg import helper\n\n"
        "def value(times):\n"
        "    delta_t = helper.calculate_deltat(times.year, times.month)\n"
        "    return delta_t\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    source.write_text(
        "from pkg import helper\n\n"
        "def value(times):\n"
        "    delta_t = helper.calculate_deltat(times.year, times.month, times.day, times.hour, times.minute, times.second)\n"
        "    return delta_t\n",
        encoding="utf-8",
    )
    assert _python_introduced_local_call_arity_mismatch_details(
        "from pkg import helper\n\n"
        "def value(times):\n"
        "    delta_t = helper.calculate_deltat(times.year, times.month)\n"
        "    return delta_t\n",
        source.read_text(encoding="utf-8"),
        repo_root,
    ) == ["helper.calculate_deltat called with 6 positional args but local definition accepts 2..2"]
    patch = subprocess.check_output(["git", "diff"], cwd=repo_root, text=True)
    (tmp_path / "workspace").mkdir()
    (tmp_path / "workspace" / "patch.diff").write_text(patch, encoding="utf-8")
    task = TaskSpec(
        task_id="swe",
        prompt="verify swe patch",
        workspace_subdir="workspace",
        expected_files=["patch.diff"],
        metadata={
            "semantic_verifier": {
                "kind": "swe_patch_apply_check",
                "repo": "owner/repo",
                "base_commit": commit,
                "repo_cache_root": str(tmp_path / "repos"),
                "patch_path": "patch.diff",
            }
        },
    )

    verification = Verifier().verify(
        task,
        tmp_path / "workspace",
        CommandResult(command="write", exit_code=0, stdout="", stderr=""),
    )

    assert verification.passed is False
    assert any(
        "changes calls beyond local function arity" in reason
        or "isolated one-line production Python assignment replacement" in reason
        for reason in verification.reasons
    )


def test_verifier_swe_patch_apply_check_rejects_local_call_keyword_mismatch(tmp_path):
    repo_root = tmp_path / "repos" / "owner" / "repo"
    source = repo_root / "pkg" / "module.py"
    source.parent.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_root, check=True)
    source.write_text(
        "def first(value, *, mode='safe'):\n"
        "    return value\n"
        "\n"
        "def second(value):\n"
        "    return value\n"
        "\n"
        "def run(value):\n"
        "    return first(value, mode='safe')\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    source.write_text(
        "def first(value, *, mode='safe'):\n"
        "    return value\n"
        "\n"
        "def second(value):\n"
        "    return value\n"
        "\n"
        "def run(value):\n"
        "    return second(value, mode='safe')\n",
        encoding="utf-8",
    )
    patch = subprocess.check_output(["git", "diff"], cwd=repo_root, text=True)
    (tmp_path / "workspace").mkdir()
    (tmp_path / "workspace" / "patch.diff").write_text(patch, encoding="utf-8")
    task = TaskSpec(
        task_id="swe",
        prompt="verify swe patch",
        workspace_subdir="workspace",
        expected_files=["patch.diff"],
        metadata={
            "semantic_verifier": {
                "kind": "swe_patch_apply_check",
                "repo": "owner/repo",
                "base_commit": commit,
                "repo_cache_root": str(tmp_path / "repos"),
                "patch_path": "patch.diff",
            }
        },
    )

    assert _python_introduced_local_call_keyword_mismatch_details(
        "def first(value, *, mode='safe'):\n"
        "    return value\n"
        "\n"
        "def second(value):\n"
        "    return value\n"
        "\n"
        "def run(value):\n"
        "    return first(value, mode='safe')\n",
        "def first(value, *, mode='safe'):\n"
        "    return value\n"
        "\n"
        "def second(value):\n"
        "    return value\n"
        "\n"
        "def run(value):\n"
        "    return second(value, mode='safe')\n",
        repo_root,
    ) == ["second called with unsupported keywords: mode"]


def test_verifier_local_call_arity_accepts_keyword_satisfied_required_args(tmp_path):
    repo_root = tmp_path / "repos" / "owner" / "repo"
    source = repo_root / "pkg" / "module.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "def build_message(message, subject):\n"
        "    return message, subject\n"
        "\n"
        "def run(message, subject):\n"
        "    return build_message(message, subject)\n",
        encoding="utf-8",
    )
    before = source.read_text(encoding="utf-8")
    after = (
        "def build_message(message, subject):\n"
        "    return message, subject\n"
        "\n"
        "def run(message, subject):\n"
        "    return build_message(message=message, subject=subject)\n"
    )

    assert _python_introduced_local_call_arity_mismatch_details(before, after, repo_root) == []
    assert _python_introduced_local_call_keyword_mismatch_details(before, after, repo_root) == []


def test_verifier_swe_patch_apply_check_rejects_removed_arithmetic_scale_factor(tmp_path):
    repo_root = tmp_path / "repos" / "owner" / "repo"
    source = repo_root / "pkg" / "module.py"
    source.parent.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_root, check=True)
    source.write_text(
        "def value(second, minute, hour, microsecond):\n"
        "    frac_of_day = (microsecond / 1e6 + (second + minute * 60 + hour * 3600)) * 1.0 / (3600 * 24)\n"
        "    return frac_of_day\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    source.write_text(
        "def value(second, minute, hour, microsecond):\n"
        "    frac_of_day = (microsecond / 1e6 + (second + minute * 60 + hour * 3600) / 3600)\n"
        "    return frac_of_day\n",
        encoding="utf-8",
    )
    patch = subprocess.check_output(["git", "diff"], cwd=repo_root, text=True)
    (tmp_path / "workspace").mkdir()
    (tmp_path / "workspace" / "patch.diff").write_text(patch, encoding="utf-8")
    task = TaskSpec(
        task_id="swe",
        prompt="verify swe patch",
        workspace_subdir="workspace",
        expected_files=["patch.diff"],
        metadata={
            "semantic_verifier": {
                "kind": "swe_patch_apply_check",
                "repo": "owner/repo",
                "base_commit": commit,
                "repo_cache_root": str(tmp_path / "repos"),
                "patch_path": "patch.diff",
            }
        },
    )

    verification = Verifier().verify(
        task,
        tmp_path / "workspace",
        CommandResult(command="write", exit_code=0, stdout="", stderr=""),
    )

    assert verification.passed is False
    assert any("arithmetic scale factor removed or weakened" in reason for reason in verification.reasons)


def test_verifier_swe_patch_apply_check_rejects_capacity_invariant_state_replacement(tmp_path):
    repo_root = tmp_path / "repos" / "owner" / "repo"
    source = repo_root / "pkg" / "module.py"
    source.parent.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_root, check=True)
    source.write_text(
        "class Packer:\n"
        "    def pack(self):\n"
        "        boundary = self.max_seq_len\n"
        "        return boundary\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    source.write_text(
        "class Packer:\n"
        "    def pack(self):\n"
        "        boundary = self.previous_sample_boundary\n"
        "        return boundary\n",
        encoding="utf-8",
    )
    patch = subprocess.check_output(["git", "diff"], cwd=repo_root, text=True)
    (tmp_path / "workspace").mkdir()
    (tmp_path / "workspace" / "patch.diff").write_text(patch, encoding="utf-8")
    task = TaskSpec(
        task_id="swe",
        prompt="verify swe patch",
        workspace_subdir="workspace",
        expected_files=["patch.diff"],
        metadata={
            "semantic_verifier": {
                "kind": "swe_patch_apply_check",
                "repo": "owner/repo",
                "base_commit": commit,
                "repo_cache_root": str(tmp_path / "repos"),
                "patch_path": "patch.diff",
            }
        },
    )

    verification = Verifier().verify(
        task,
        tmp_path / "workspace",
        CommandResult(command="write", exit_code=0, stdout="", stderr=""),
    )

    assert verification.passed is False
    assert any("capacity/limit invariant replaced by state field" in reason for reason in verification.reasons)


def test_verifier_swe_patch_apply_check_rejects_repeated_official_failed_patch(tmp_path):
    repo_root = tmp_path / "repos" / "owner" / "repo"
    source = repo_root / "pkg" / "module.py"
    source.parent.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_root, check=True)
    source.write_text("def value(value):\n    return value\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    source.write_text("def value(value):\n    return value + 1\n", encoding="utf-8")
    patch = subprocess.check_output(["git", "diff"], cwd=repo_root, text=True)
    (tmp_path / "workspace").mkdir()
    (tmp_path / "workspace" / "patch.diff").write_text(patch, encoding="utf-8")
    task = TaskSpec(
        task_id="swe",
        prompt="verify swe patch",
        workspace_subdir="workspace",
        expected_files=["patch.diff"],
        metadata={
            "semantic_verifier": {
                "kind": "swe_patch_apply_check",
                "repo": "owner/repo",
                "base_commit": commit,
                "repo_cache_root": str(tmp_path / "repos"),
                "patch_path": "patch.diff",
                "forbidden_patch_texts": [patch],
            }
        },
    )

    verification = Verifier().verify(
        task,
        tmp_path / "workspace",
        CommandResult(command="write", exit_code=0, stdout="", stderr=""),
    )

    assert verification.passed is False
    assert "SWE patch repeats prior official-failed patch exactly" in verification.reasons


def test_verifier_swe_patch_apply_check_rejects_literal_constant_assignment_guess(tmp_path):
    repo_root = tmp_path / "repos" / "owner" / "repo"
    source = repo_root / "pkg" / "module.py"
    source.parent.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_root, check=True)
    source.write_text("def value(x):\n    result = compute(x)\n    return result\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    source.write_text("def value(x):\n    result = 67.0\n    return result\n", encoding="utf-8")
    patch = subprocess.check_output(["git", "diff"], cwd=repo_root, text=True)
    (tmp_path / "workspace").mkdir()
    (tmp_path / "workspace" / "patch.diff").write_text(patch, encoding="utf-8")
    task = TaskSpec(
        task_id="swe",
        prompt="verify swe patch",
        workspace_subdir="workspace",
        expected_files=["patch.diff"],
        metadata={
            "semantic_verifier": {
                "kind": "swe_patch_apply_check",
                "repo": "owner/repo",
                "base_commit": commit,
                "repo_cache_root": str(tmp_path / "repos"),
                "patch_path": "patch.diff",
            }
        },
    )

    verification = Verifier().verify(
        task,
        tmp_path / "workspace",
        CommandResult(command="write", exit_code=0, stdout="", stderr=""),
    )

    assert verification.passed is False
    assert any("literal constant" in reason for reason in verification.reasons)


def test_verifier_static_guard_rejects_same_symbol_assignment_value_patch():
    patch = (
        "--- a/pkg/config.py\n"
        "+++ b/pkg/config.py\n"
        "@@ -1 +1 @@\n"
        "-            samples_per_save=25000,\n"
        "+            samples_per_save=None,\n"
    )

    assert "isolated one-line production Python assignment replacement" in (
        Verifier._patch_isolated_one_line_production_python_replacement(patch)
    )


def test_verifier_static_guards_reject_raw_completed_failure_shapes():
    assert Verifier._patch_adds_placeholder_success_print(
        "--- a/pkg/module.py\n+++ b/pkg/module.py\n@@ -1 +1 @@\n->>> value()\n+print(\"patch applied\")\n"
    )
    assert _python_introduced_self_recursive_property_accesses(
        "class Provider:\n    @property\n    def ready(self):\n        return True\n",
        "class Provider:\n    @property\n    def ready(self):\n        return True or self.ready\n",
    ) == ["ready reads self.ready"]
    nested_assignment = _python_nested_assignment_replacement_details(
        "--- a/pkg/module.py\n+++ b/pkg/module.py\n@@ -1 +1 @@\n-        self._cookies[name] = ''\n+        self._cookies[name][\"expires\"] = -1\n"
    )
    assert nested_assignment
    assert "nested target" in nested_assignment[0]
    assignment_to_expression = Verifier._patch_suspicious_python_hunk_replacements(
        "--- a/pkg/module.py\n"
        "+++ b/pkg/module.py\n"
        "@@ -1 +1 @@\n"
        "-        pkg_name = self._get_package_name(self._dep)\n"
        "+        self._conanfile.conf.get(\"tools.gnu:pkgconfigdeps_set_proper\", check_type=bool, default=False)\n"
    )
    assert "assignment replaced by" in assignment_to_expression
    comment_to_return = Verifier._patch_suspicious_python_hunk_replacements(
        "--- a/pkg/module.py\n"
        "+++ b/pkg/module.py\n"
        "@@ -1,4 +1,4 @@\n"
        " def check(value):\n"
        "-    # Continue into detailed validation.\n"
        "+    return True\n"
        "     if value:\n"
        "         return validate(value)\n"
    )
    assert "comment replaced by control-flow statement" in comment_to_return
    exception_alias = Verifier._patch_suspicious_python_hunk_replacements(
        "--- a/pkg/module.py\n"
        "+++ b/pkg/module.py\n"
        "@@ -1,6 +1,6 @@\n"
        " try:\n"
        "     run()\n"
        " except Exception as error:\n"
        "     logger.error(error)\n"
        "-    hook(exit_code=1)\n"
        "+    hook(exit_code=exc.code)\n"
    )
    assert "exception handler aliases error" in exception_alias
    nested_mapping = Verifier._patch_suspicious_python_hunk_replacements(
        "--- a/pkg/module.py\n"
        "+++ b/pkg/module.py\n"
        "@@ -1,4 +1,7 @@\n"
        " def unset(self, name):\n"
        "-    self._cookies[name]['expires'] = -1\n"
        "+    self._cookies[name] = \"\"\n"
        "+    self._cookies[name]['expires'] = -1\n"
        "+    self._cookies[name]['path'] = '/'\n"
    )
    assert "reinitialized before nested field updates" in nested_mapping
    numeric_none = Verifier._patch_suspicious_python_hunk_replacements(
        "--- a/pkg/module.py\n"
        "+++ b/pkg/module.py\n"
        "@@ -1 +1 @@\n"
        "-            samples_per_save=25000,\n"
        "+            samples_per_save=None,\n"
    )
    assert "numeric default samples_per_save= replaced by None" in numeric_none
    identical_ternary = Verifier._patch_suspicious_python_hunk_replacements(
        "--- a/pkg/module.py\n"
        "+++ b/pkg/module.py\n"
        "@@ -1 +1 @@\n"
        "-    host = await choose(source)\n"
        "+    host = await choose(source) if tmp_host == 'default' else await choose(source)\n"
    )
    assert "conditional expression has identical branches" in identical_ternary
    removed_guard = Verifier._patch_suspicious_python_hunk_replacements(
        "--- a/pkg/module.py\n"
        "+++ b/pkg/module.py\n"
        "@@ -1,5 +1,4 @@\n"
        " def apply(e):\n"
        "-    elif isinstance(e, Density):\n"
        "         new_args = transform(e.args)\n"
        "         return Density(*new_args)\n"
    )
    assert "removed control guard" in removed_guard
    duplicate_return = Verifier._patch_suspicious_python_hunk_replacements(
        "--- a/pkg/module.py\n"
        "+++ b/pkg/module.py\n"
        "@@ -1,5 +1,6 @@\n"
        "     if role == 'assistant':\n"
        "         return cls.from_assistant(text=content)\n"
        "+        return cls.from_assistant(text=content)\n"
        "     assert content is not None\n"
    )
    assert "duplicates existing hunk statement" in duplicate_return
    check_type_flip = Verifier._patch_suspicious_python_hunk_replacements(
        "--- a/pkg/module.py\n"
        "+++ b/pkg/module.py\n"
        "@@ -1 +1 @@\n"
        "-    value = conf.get('setting', check_type=str)\n"
        "+    value = conf.get('setting', check_type=bool)\n"
    )
    assert "check_type contract changed from str to bool" in check_type_flip
    template_quote = Verifier._patch_suspicious_text_template_replacements(
        "--- a/templates/head.html\n"
        "+++ b/templates/head.html\n"
        "@@ -1 +1 @@\n"
        "-{% if request.user_profile.theme == 'light' %}\n"
        "+{% if request.user_profile.theme == \"light\" %}\n"
    )
    assert "changes only quote style" in template_quote
    whitespace_only_json = Verifier._patch_suspicious_whitespace_only_structured_data_change(
        "--- a/config/schema.json\n"
        "+++ b/config/schema.json\n"
        "@@ -1,3 +1,3 @@\n"
        " {\n"
        "-  \"$ref\": \"#/definitions/item\"\n"
        "+    \"$ref\": \"#/definitions/item\"\n"
        " }\n"
    )
    assert "changes only whitespace" in whitespace_only_json
    duplicate_regex = Verifier._patch_suspicious_structured_data_regex_escape_duplicate(
        "--- a/config/schema.json\n"
        "+++ b/config/schema.json\n"
        "@@ -1,6 +1,9 @@\n"
        "     {\n"
        "      \"const\": \"*\"\n"
        "+    },\n"
        "+    {\n"
        "+      \"pattern\": \"^\\\\\\\\d{12}$\"\n"
        "     },\n"
        "     {\n"
        "      \"pattern\": \"^\\\\d{12}$\"\n"
    )
    assert "double-escaped duplicate regex pattern" in duplicate_regex
    behavior_deletion = Verifier._patch_suspicious_python_behavior_deletion(
        "--- a/pkg/rule.py\n"
        "+++ b/pkg/rule.py\n"
        "@@ -1,8 +1,4 @@\n"
        "     else:\n"
        "-        yield ValidationError(\n"
        "-            message='bad',\n"
        "-            path=path,\n"
        "-        )\n"
        "+        path=path\n"
    )
    assert "deletes production behavior" in behavior_deletion
    signature_contract = Verifier._patch_suspicious_python_signature_contract_change(
        "--- a/pkg/rule.py\n"
        "+++ b/pkg/rule.py\n"
        "@@ -1,5 +1,3 @@\n"
        "-    def validate(\n"
        "-        self, validator, keywords, instance, schema\n"
        "-    ) -> ValidationResult:\n"
        "+    def validate(self, validator, instance) -> ValidationResult:\n"
    )
    assert "reduces production function signature arity" in signature_contract
    class_config = Verifier._patch_suspicious_python_class_config_removal(
        "--- a/pkg/model.py\n"
        "+++ b/pkg/model.py\n"
        "@@ -1,5 +1,6 @@\n"
        " class InternalRequest(BaseModel):\n"
        "-    model_config = ConfigDict(populate_by_name=True)\n"
        "+    order_no: Decimal | None = None\n"
        "+    \"\"\"Order number.\"\"\"\n"
    )
    assert "removes class model_config" in class_config
    loop_return = Verifier._patch_suspicious_python_control_flow_replacement(
        "--- a/pkg/rule.py\n"
        "+++ b/pkg/rule.py\n"
        "@@ -1,5 +1,4 @@\n"
        "-            if setting != 'expected':\n"
        "-                continue\n"
        "+            if not isinstance(runtime, str): return False\n"
    )
    assert "replaces loop continuation" in loop_return
    tuple_type = Verifier._patch_suspicious_python_tuple_type_construction(
        "--- a/pkg/walk.py\n"
        "+++ b/pkg/walk.py\n"
        "@@ -1 +1 @@\n"
        "+        if isinstance(value, tuple([list]) + _SCALAR_TYPES):\n"
    )
    assert "tuple([type])" in tuple_type
    repair_fragments = Verifier._patch_suspicious_python_repair_fragments(
        "--- a/pkg/stream.py\n"
        "+++ b/pkg/stream.py\n"
        "@@ -1,6 +1,3 @@\n"
        "-                if usage['details'] is not None:\n"
        "-                    details = usage['details']\n"
        "+                if not hasattr(self, 'usage_cache'): self.usage_cache = {}\n"
    )
    assert "unrelated self-attribute initialization" in repair_fragments
    call_assignment_collapse = Verifier._patch_suspicious_python_call_assignment_collapse(
        "--- a/pkg/rule.py\n"
        "+++ b/pkg/rule.py\n"
        "@@ -1,7 +1,4 @@\n"
        "             else:\n"
        "-                step_validator = validator.evolve(\n"
        "-                    resolver=self.resolver,\n"
        "-                    schema=self.schema,\n"
        "-                )\n"
        "+                resolver=self.resolver\n"
    )
    assert "collapses a production call-assignment block" in call_assignment_collapse
    type_fragments = Verifier._patch_suspicious_python_repair_fragments(
        "--- a/pkg/getatt.py\n"
        "+++ b/pkg/getatt.py\n"
        "@@ -1,4 +1,5 @@\n"
        "     values[\n"
        "+        s: Any,\n"
        "+        paths: Sequence[Any],\n"
        "     ]\n"
    )
    assert "bare type-annotation fragments" in type_fragments
    duplicate_error = Verifier._patch_suspicious_python_repair_fragments(
        "--- a/pkg/plugin.py\n"
        "+++ b/pkg/plugin.py\n"
        "@@ -1,6 +1,12 @@\n"
        "     if invalid:\n"
        "         errors.append(\n"
        "             PluginError(meta, 'bad', entry)\n"
        "         )\n"
        "+        errors.append(\n"
        "+            PluginError(meta, 'bad', entry)\n"
        "+        )\n"
    )
    assert "duplicates existing error append" in duplicate_error
    container_api_mismatch = Verifier._patch_suspicious_python_container_api_mismatch(
        "--- a/pkg/tracker.py\n"
        "+++ b/pkg/tracker.py\n"
        "@@ -1,5 +1,6 @@\n"
        "         else:\n"
        "             errors[file][name].update([message])\n"
        "+            errors[file][name].append(message)\n"
    )
    assert "mixes append with existing update calls" in container_api_mismatch
    tiny_mutation = Verifier._patch_suspicious_tiny_production_mutation(
        "--- a/pkg/tracker.py\n"
        "+++ b/pkg/tracker.py\n"
        "@@ -1,3 +1,4 @@\n"
        "     if missing:\n"
        "         create()\n"
        "+        self._errors[file][name][message] = 1\n"
    )
    assert "single-line nested production mutation" in tiny_mutation


def test_verifier_static_guards_reject_invalid_toml_and_spellcheck_only_changes(tmp_path):
    repo_root = tmp_path / "repos" / "owner" / "repo"
    repo_root.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_root, check=True)
    (repo_root / "pyproject.toml").write_text("[build-system]\nrequires = []\n", encoding="utf-8")
    (repo_root / ".spellcheck-en-custom.txt").write_text("GiB\n", encoding="utf-8")
    (repo_root / "CHANGES.txt").write_text("old\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    task = TaskSpec(
        task_id="swe",
        prompt="verify swe patch",
        workspace_subdir="workspace",
        expected_files=["patch.diff"],
        metadata={
            "semantic_verifier": {
                "kind": "swe_patch_apply_check",
                "repo": "owner/repo",
                "base_commit": commit,
                "repo_cache_root": str(tmp_path / "repos"),
                "patch_path": "patch.diff",
            }
        },
    )

    (workspace / "patch.diff").write_text(
        "--- a/pyproject.toml\n"
        "+++ b/pyproject.toml\n"
        "@@ -1,2 +1,2 @@\n"
        "-[build-system]\n"
        "+version: 2\n"
        " requires = []\n",
        encoding="utf-8",
    )
    verification = Verifier().verify(task, workspace, CommandResult(command="write", exit_code=0, stdout="", stderr=""))
    assert verification.passed is False
    assert any("TOML syntax check failed" in reason for reason in verification.reasons)

    (workspace / "patch.diff").write_text(
        "--- a/.spellcheck-en-custom.txt\n+++ b/.spellcheck-en-custom.txt\n@@ -1 +1 @@\n-GiB\n+Gib\n",
        encoding="utf-8",
    )
    verification = Verifier().verify(task, workspace, CommandResult(command="write", exit_code=0, stdout="", stderr=""))
    assert verification.passed is False
    assert any("changes only tests or auxiliary update artifacts" in reason for reason in verification.reasons)

    (workspace / "patch.diff").write_text(
        "--- a/CHANGES.txt\n+++ b/CHANGES.txt\n@@ -1 +1 @@\n-old\n+new\n",
        encoding="utf-8",
    )
    verification = Verifier().verify(task, workspace, CommandResult(command="write", exit_code=0, stdout="", stderr=""))
    assert verification.passed is False
    assert any("changes only tests or auxiliary update artifacts" in reason for reason in verification.reasons)


def test_verifier_swe_patch_apply_check_rejects_bad_patch(tmp_path):
    repo_root = tmp_path / "repos" / "owner" / "repo"
    source = repo_root / "pkg" / "module.py"
    source.parent.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_root, check=True)
    source.write_text("def value():\n    return 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    (tmp_path / "workspace").mkdir()
    (tmp_path / "workspace" / "patch.diff").write_text(
        "diff --git a/pkg/missing.py b/pkg/missing.py\n--- a/pkg/missing.py\n+++ b/pkg/missing.py\n@@ -1 +1 @@\n-x\n+y\n",
        encoding="utf-8",
    )
    task = TaskSpec(
        task_id="swe",
        prompt="verify swe patch",
        workspace_subdir="workspace",
        expected_files=["patch.diff"],
        metadata={
            "semantic_verifier": {
                "kind": "swe_patch_apply_check",
                "repo": "owner/repo",
                "base_commit": commit,
                "repo_cache_root": str(tmp_path / "repos"),
                "patch_path": "patch.diff",
            }
        },
    )

    verification = Verifier().verify(
        task,
        tmp_path / "workspace",
        CommandResult(command="write", exit_code=0, stdout="", stderr=""),
    )

    assert verification.passed is False
    assert any(reason.startswith("SWE patch apply check failed") for reason in verification.reasons)


def test_verifier_swe_patch_apply_check_rejects_unexpected_paths(tmp_path):
    repo_root = tmp_path / "repos" / "owner" / "repo"
    source = repo_root / "pkg" / "module.py"
    source.parent.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_root, check=True)
    source.write_text("def value():\n    return 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    (tmp_path / "workspace").mkdir()
    (tmp_path / "workspace" / "patch.diff").write_text(
        "diff --git a/pkg/other.py b/pkg/other.py\n--- a/pkg/other.py\n+++ b/pkg/other.py\n@@ -1 +1 @@\n-x\n+y\n",
        encoding="utf-8",
    )
    task = TaskSpec(
        task_id="swe",
        prompt="verify swe patch",
        workspace_subdir="workspace",
        expected_files=["patch.diff"],
        metadata={
            "semantic_verifier": {
                "kind": "swe_patch_apply_check",
                "repo": "owner/repo",
                "base_commit": commit,
                "repo_cache_root": str(tmp_path / "repos"),
                "patch_path": "patch.diff",
                "expected_changed_paths": ["pkg/module.py"],
            }
        },
    )

    verification = Verifier().verify(
        task,
        tmp_path / "workspace",
        CommandResult(command="write", exit_code=0, stdout="", stderr=""),
    )

    assert verification.passed is False
    assert "SWE patch diff includes unexpected path: pkg/other.py" in verification.reasons


def test_verifier_swe_patch_apply_check_rejects_template_patch(tmp_path):
    repo_root = tmp_path / "repos" / "owner" / "repo"
    source = repo_root / "pkg" / "module.py"
    source.parent.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_root, check=True)
    source.write_text("def value():\n    return 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    (tmp_path / "workspace").mkdir()
    (tmp_path / "workspace" / "patch.diff").write_text(
        "diff --git a/pkg/module.py b/pkg/module.py\n--- a/pkg/module.py\n+++ b/pkg/module.py\n@@ -1,2 +1,3 @@\n+# This is a test file.\n def value():\n     return 1\n",
        encoding="utf-8",
    )
    task = TaskSpec(
        task_id="swe",
        prompt="verify swe patch",
        workspace_subdir="workspace",
        expected_files=["patch.diff"],
        metadata={
            "semantic_verifier": {
                "kind": "swe_patch_apply_check",
                "repo": "owner/repo",
                "base_commit": commit,
                "repo_cache_root": str(tmp_path / "repos"),
                "patch_path": "patch.diff",
                "expected_changed_paths": ["pkg/module.py"],
            }
        },
    )

    verification = Verifier().verify(
        task,
        tmp_path / "workspace",
        CommandResult(command="write", exit_code=0, stdout="", stderr=""),
    )

    assert verification.passed is False
    assert "SWE patch diff contains placeholder/template content" in verification.reasons


def test_verifier_swe_patch_apply_check_allows_placeholder_tokens_in_context_lines(tmp_path):
    repo_root = tmp_path / "repos" / "owner" / "repo"
    source = repo_root / "pkg" / "module.py"
    source.parent.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_root, check=True)
    source.write_text("import os\n\n\ndef value():\n    return 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    (tmp_path / "workspace").mkdir()
    (tmp_path / "workspace" / "patch.diff").write_text(
        "diff --git a/pkg/module.py b/pkg/module.py\n"
        "--- a/pkg/module.py\n"
        "+++ b/pkg/module.py\n"
        "@@ -1,5 +1,5 @@\n"
        " import os\n"
        " \n"
        " \n"
        " def value():\n"
        "-    return 1\n"
        "+    return 2\n",
        encoding="utf-8",
    )
    task = TaskSpec(
        task_id="swe",
        prompt="verify swe patch",
        workspace_subdir="workspace",
        expected_files=["patch.diff"],
        metadata={
            "semantic_verifier": {
                "kind": "swe_patch_apply_check",
                "repo": "owner/repo",
                "base_commit": commit,
                "repo_cache_root": str(tmp_path / "repos"),
                "patch_path": "patch.diff",
                "expected_changed_paths": ["pkg/module.py"],
            }
        },
    )

    verification = Verifier().verify(
        task,
        tmp_path / "workspace",
        CommandResult(command="write", exit_code=0, stdout="", stderr=""),
    )

    assert "SWE patch diff contains placeholder/template content" not in verification.reasons


def test_verifier_swe_patch_apply_check_skips_brittle_success_command(tmp_path):
    repo_root = tmp_path / "repos" / "owner" / "repo"
    source = repo_root / "pkg" / "module.py"
    source.parent.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_root, check=True)
    source.write_text("def value():\n    if name == 'dummy':\n        return 1\n    return 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    source.write_text("def value():\n    if name == 'dummy':\n        return 1\n    return 2\n", encoding="utf-8")
    patch = subprocess.check_output(["git", "diff"], cwd=repo_root, text=True)
    subprocess.run(["git", "checkout", "--", "pkg/module.py"], cwd=repo_root, check=True)
    (tmp_path / "workspace").mkdir()
    (tmp_path / "workspace" / "patch.diff").write_text(patch, encoding="utf-8")
    task = TaskSpec(
        task_id="swe",
        prompt="verify swe patch",
        workspace_subdir="workspace",
        success_command="test -s patch.diff && ! grep -Eiq 'dummy' patch.diff",
        expected_files=["patch.diff"],
        metadata={
            "semantic_verifier": {
                "kind": "swe_patch_apply_check",
                "repo": "owner/repo",
                "base_commit": commit,
                "repo_cache_root": str(tmp_path / "repos"),
                "patch_path": "patch.diff",
                "expected_changed_paths": ["pkg/module.py"],
            }
        },
    )

    verification = Verifier().verify(
        task,
        tmp_path / "workspace",
        CommandResult(command="write", exit_code=0, stdout="", stderr=""),
    )

    assert verification.passed is True
    assert any(
        evidence.get("skip_reason") == "structured_artifact_verifier_covers_success_command"
        for evidence in verification.evidence
    )


def test_verifier_swe_patch_apply_check_rejects_missing_required_issue_identifier(tmp_path):
    repo_root = tmp_path / "repos" / "owner" / "repo"
    source = repo_root / "pkg" / "module.py"
    source.parent.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_root, check=True)
    source.write_text("def write(lines):\n    return lines\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    (tmp_path / "workspace").mkdir()
    (tmp_path / "workspace" / "patch.diff").write_text(
        "diff --git a/pkg/module.py b/pkg/module.py\n"
        "--- a/pkg/module.py\n"
        "+++ b/pkg/module.py\n"
        "@@ -1,2 +1,2 @@\n"
        "-def write(lines):\n"
        "+def write(lines, **kwargs):\n"
        "     return lines\n",
        encoding="utf-8",
    )
    task = TaskSpec(
        task_id="swe",
        prompt="verify swe patch",
        workspace_subdir="workspace",
        expected_files=["patch.diff"],
        metadata={
            "semantic_verifier": {
                "kind": "swe_patch_apply_check",
                "repo": "owner/repo",
                "base_commit": commit,
                "repo_cache_root": str(tmp_path / "repos"),
                "patch_path": "patch.diff",
                "expected_changed_paths": ["pkg/module.py"],
                "required_patch_identifiers": ["header_rows"],
            }
        },
    )

    verification = Verifier().verify(
        task,
        tmp_path / "workspace",
        CommandResult(command="write", exit_code=0, stdout="", stderr=""),
    )

    assert verification.passed is False
    assert "SWE patch does not reference required issue identifier: header_rows" in verification.reasons


def test_verifier_swe_patch_apply_check_rejects_noop_patch(tmp_path):
    repo_root = tmp_path / "repos" / "owner" / "repo"
    source = repo_root / "pkg" / "module.py"
    source.parent.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_root, check=True)
    source.write_text('"""module doc"""\ndef value():\n    return 1\n', encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    (tmp_path / "workspace").mkdir()
    (tmp_path / "workspace" / "patch.diff").write_text(
        'diff --git a/pkg/module.py b/pkg/module.py\n'
        '--- a/pkg/module.py\n'
        '+++ b/pkg/module.py\n'
        '@@ -1,3 +1,3 @@\n'
        '-"""module doc"""\n'
        '+"""module doc"""\n'
        ' def value():\n'
        '     return 1\n',
        encoding="utf-8",
    )
    task = TaskSpec(
        task_id="swe",
        prompt="verify swe patch",
        workspace_subdir="workspace",
        expected_files=["patch.diff"],
        metadata={
            "semantic_verifier": {
                "kind": "swe_patch_apply_check",
                "repo": "owner/repo",
                "base_commit": commit,
                "repo_cache_root": str(tmp_path / "repos"),
                "patch_path": "patch.diff",
                "expected_changed_paths": ["pkg/module.py"],
            }
        },
    )

    verification = Verifier().verify(
        task,
        tmp_path / "workspace",
        CommandResult(command="write", exit_code=0, stdout="", stderr=""),
    )

    assert verification.passed is False
    assert "SWE patch diff has no meaningful content change" in verification.reasons


def test_verifier_swe_patch_apply_check_rejects_non_executable_patch(tmp_path):
    repo_root = tmp_path / "repos" / "owner" / "repo"
    source = repo_root / "pkg" / "module.py"
    source.parent.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_root, check=True)
    source.write_text('"""module doc"""\n:Author: Old\n\ndef value():\n    return 1\n', encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    (tmp_path / "workspace").mkdir()
    (tmp_path / "workspace" / "patch.diff").write_text(
        "diff --git a/pkg/module.py b/pkg/module.py\n"
        "--- a/pkg/module.py\n"
        "+++ b/pkg/module.py\n"
        "@@ -1,5 +1,5 @@\n"
        ' """module doc"""\n'
        "-:Author: Old\n"
        "+    Author: New\n"
        "\n"
        " def value():\n",
        encoding="utf-8",
    )
    task = TaskSpec(
        task_id="swe",
        prompt="verify swe patch",
        workspace_subdir="workspace",
        expected_files=["patch.diff"],
        metadata={
            "semantic_verifier": {
                "kind": "swe_patch_apply_check",
                "repo": "owner/repo",
                "base_commit": commit,
                "repo_cache_root": str(tmp_path / "repos"),
                "patch_path": "patch.diff",
                "expected_changed_paths": ["pkg/module.py"],
            }
        },
    )

    verification = Verifier().verify(
        task,
        tmp_path / "workspace",
        CommandResult(command="write", exit_code=0, stdout="", stderr=""),
    )

    assert verification.passed is False
    assert "SWE patch diff changes only comments/docstrings/non-executable text" in verification.reasons


def test_verifier_swe_patch_apply_check_rejects_double_escaped_raw_regex_whitespace(tmp_path):
    repo_root = tmp_path / "repos" / "owner" / "repo"
    source = repo_root / "pkg" / "module.py"
    source.parent.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_root, check=True)
    source.write_text(
        "import re\n\n"
        "def classify(line):\n"
        "    pattern = r\"^\\s*READ\\s*$\"\n"
        "    return re.compile(pattern).match(line)\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    (tmp_path / "workspace").mkdir()
    (tmp_path / "workspace" / "patch.diff").write_text(
        "diff --git a/pkg/module.py b/pkg/module.py\n"
        "--- a/pkg/module.py\n"
        "+++ b/pkg/module.py\n"
        "@@ -1,5 +1,5 @@\n"
        " import re\n"
        "\n"
        " def classify(line):\n"
        "-    pattern = r\"^\\s*READ\\s*$\"\n"
        "+    pattern = r\"^\\\\s*READ\\\\s*$\"\n"
        "     return re.compile(pattern).match(line)\n",
        encoding="utf-8",
    )
    task = TaskSpec(
        task_id="swe",
        prompt="verify swe patch",
        workspace_subdir="workspace",
        expected_files=["patch.diff"],
        metadata={
            "semantic_verifier": {
                "kind": "swe_patch_apply_check",
                "repo": "owner/repo",
                "base_commit": commit,
                "repo_cache_root": str(tmp_path / "repos"),
                "patch_path": "patch.diff",
                "expected_changed_paths": ["pkg/module.py"],
            }
        },
    )

    verification = Verifier().verify(
        task,
        tmp_path / "workspace",
        CommandResult(command="write", exit_code=0, stdout="", stderr=""),
    )

    assert verification.passed is False
    assert "SWE patch suspiciously double-escapes raw regex whitespace" in verification.reasons


def test_verifier_swe_patch_apply_check_rejects_python_syntax_breakage(tmp_path):
    repo_root = tmp_path / "repos" / "owner" / "repo"
    source = repo_root / "pkg" / "module.py"
    source.parent.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_root, check=True)
    source.write_text(
        "from pkg.other import (\n"
        "    UsefulThing,\n"
        ")\n"
        "\n"
        "def value():\n"
        "    return UsefulThing()\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    (tmp_path / "workspace").mkdir()
    (tmp_path / "workspace" / "patch.diff").write_text(
        "diff --git a/pkg/module.py b/pkg/module.py\n"
        "--- a/pkg/module.py\n"
        "+++ b/pkg/module.py\n"
        "@@ -1,5 +1,5 @@\n"
        " from pkg.other import (\n"
        "-    UsefulThing,\n"
        "+    def read(self, input, **kwargs):\n"
        " )\n"
        "\n"
        " def value():\n",
        encoding="utf-8",
    )
    task = TaskSpec(
        task_id="swe",
        prompt="verify swe patch",
        workspace_subdir="workspace",
        expected_files=["patch.diff"],
        metadata={
            "semantic_verifier": {
                "kind": "swe_patch_apply_check",
                "repo": "owner/repo",
                "base_commit": commit,
                "repo_cache_root": str(tmp_path / "repos"),
                "patch_path": "patch.diff",
                "expected_changed_paths": ["pkg/module.py"],
            }
        },
    )

    verification = Verifier().verify(
        task,
        tmp_path / "workspace",
        CommandResult(command="write", exit_code=0, stdout="", stderr=""),
    )

    assert verification.passed is False
    assert any(reason.startswith("SWE patch python syntax check failed: ") for reason in verification.reasons)


def test_verifier_swe_patch_apply_check_rejects_docstring_only_ast_patch(tmp_path):
    repo_root = tmp_path / "repos" / "owner" / "repo"
    source = repo_root / "pkg" / "module.py"
    source.parent.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_root, check=True)
    source.write_text(
        "def value(line):\n"
        "    \"\"\"Interpret a line.\n"
        "\n"
        "    Parameters\n"
        "    ----------\n"
        "    \"\"\"\n"
        "    return line\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    (tmp_path / "workspace").mkdir()
    (tmp_path / "workspace" / "patch.diff").write_text(
        "diff --git a/pkg/module.py b/pkg/module.py\n"
        "--- a/pkg/module.py\n"
        "+++ b/pkg/module.py\n"
        "@@ -1,7 +1,7 @@\n"
        " def value(line):\n"
        "     \"\"\"Interpret a line.\n"
        "\n"
        "-    Parameters\n"
        "+        return \"comment\"\n"
        "     ----------\n"
        "     \"\"\"\n"
        "     return line\n",
        encoding="utf-8",
    )
    task = TaskSpec(
        task_id="swe",
        prompt="verify swe patch",
        workspace_subdir="workspace",
        expected_files=["patch.diff"],
        metadata={
            "semantic_verifier": {
                "kind": "swe_patch_apply_check",
                "repo": "owner/repo",
                "base_commit": commit,
                "repo_cache_root": str(tmp_path / "repos"),
                "patch_path": "patch.diff",
                "expected_changed_paths": ["pkg/module.py"],
            }
        },
    )

    verification = Verifier().verify(
        task,
        tmp_path / "workspace",
        CommandResult(command="write", exit_code=0, stdout="", stderr=""),
    )

    assert verification.passed is False
    assert "SWE patch python AST unchanged after ignoring docstrings/comments" in verification.reasons


def test_verifier_swe_patch_apply_check_rejects_removed_source_definition(tmp_path):
    repo_root = tmp_path / "repos" / "owner" / "repo"
    source = repo_root / "pkg" / "module.py"
    source.parent.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_root, check=True)
    source.write_text(
        "class Reader:\n"
        "    def __init__(self):\n"
        "        pass\n"
        "\n"
        "    def write(self, lines):\n"
        "        return list(lines)\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    (tmp_path / "workspace").mkdir()
    (tmp_path / "workspace" / "patch.diff").write_text(
        "diff --git a/pkg/module.py b/pkg/module.py\n"
        "--- a/pkg/module.py\n"
        "+++ b/pkg/module.py\n"
        "@@ -1,6 +1,6 @@\n"
        " class Reader:\n"
        "     def __init__(self):\n"
        "         pass\n"
        "\n"
        "-    def write(self, lines):\n"
        "+        self.header = []\n"
        "         return list(lines)\n",
        encoding="utf-8",
    )
    task = TaskSpec(
        task_id="swe",
        prompt="verify swe patch",
        workspace_subdir="workspace",
        expected_files=["patch.diff"],
        metadata={
            "semantic_verifier": {
                "kind": "swe_patch_apply_check",
                "repo": "owner/repo",
                "base_commit": commit,
                "repo_cache_root": str(tmp_path / "repos"),
                "patch_path": "patch.diff",
                "expected_changed_paths": ["pkg/module.py"],
            }
        },
    )

    verification = Verifier().verify(
        task,
        tmp_path / "workspace",
        CommandResult(command="write", exit_code=0, stdout="", stderr=""),
    )

    assert verification.passed is False
    assert "SWE patch removes existing Python definitions in pkg/module.py: Reader.write" in verification.reasons


def test_verifier_swe_patch_apply_check_rejects_unused_new_source_parameter(tmp_path):
    repo_root = tmp_path / "repos" / "owner" / "repo"
    source = repo_root / "pkg" / "module.py"
    source.parent.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_root, check=True)
    source.write_text(
        "def parse_line(line):\n"
        "    return line.lower()\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    (tmp_path / "workspace").mkdir()
    (tmp_path / "workspace" / "patch.diff").write_text(
        "diff --git a/pkg/module.py b/pkg/module.py\n"
        "--- a/pkg/module.py\n"
        "+++ b/pkg/module.py\n"
        "@@ -1,2 +1,2 @@\n"
        "-def parse_line(line):\n"
        "+def parse_line(line, err_specs=None):\n"
        "     return line.lower()\n",
        encoding="utf-8",
    )
    task = TaskSpec(
        task_id="swe",
        prompt="verify swe patch",
        workspace_subdir="workspace",
        expected_files=["patch.diff"],
        metadata={
            "semantic_verifier": {
                "kind": "swe_patch_apply_check",
                "repo": "owner/repo",
                "base_commit": commit,
                "repo_cache_root": str(tmp_path / "repos"),
                "patch_path": "patch.diff",
                "expected_changed_paths": ["pkg/module.py"],
            }
        },
    )

    verification = Verifier().verify(
        task,
        tmp_path / "workspace",
        CommandResult(command="write", exit_code=0, stdout="", stderr=""),
    )

    assert verification.passed is False
    assert "SWE patch adds unused production function parameters in pkg/module.py: parse_line.err_specs" in verification.reasons


def test_verifier_swe_patch_apply_check_rejects_shadowed_new_source_parameter(tmp_path):
    repo_root = tmp_path / "repos" / "owner" / "repo"
    source = repo_root / "pkg" / "module.py"
    source.parent.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_root, check=True)
    source.write_text(
        "def parse_line(line):\n"
        "    err_specs = {}\n"
        "    if line:\n"
        "        err_specs['line'] = line\n"
        "    return err_specs\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    (tmp_path / "workspace").mkdir()
    (tmp_path / "workspace" / "patch.diff").write_text(
        "diff --git a/pkg/module.py b/pkg/module.py\n"
        "--- a/pkg/module.py\n"
        "+++ b/pkg/module.py\n"
        "@@ -1,5 +1,5 @@\n"
        "-def parse_line(line):\n"
        "+def parse_line(line, err_specs=None):\n"
        "     err_specs = {}\n"
        "     if line:\n"
        "         err_specs['line'] = line\n"
        "     return err_specs\n",
        encoding="utf-8",
    )
    task = TaskSpec(
        task_id="swe",
        prompt="verify swe patch",
        workspace_subdir="workspace",
        expected_files=["patch.diff"],
        metadata={
            "semantic_verifier": {
                "kind": "swe_patch_apply_check",
                "repo": "owner/repo",
                "base_commit": commit,
                "repo_cache_root": str(tmp_path / "repos"),
                "patch_path": "patch.diff",
                "expected_changed_paths": ["pkg/module.py"],
            }
        },
    )

    verification = Verifier().verify(
        task,
        tmp_path / "workspace",
        CommandResult(command="write", exit_code=0, stdout="", stderr=""),
    )

    assert verification.passed is False
    assert "SWE patch adds unused production function parameters in pkg/module.py: parse_line.err_specs" in verification.reasons


def test_verifier_swe_patch_apply_check_rejects_init_return_value(tmp_path):
    repo_root = tmp_path / "repos" / "owner" / "repo"
    source = repo_root / "pkg" / "module.py"
    source.parent.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_root, check=True)
    source.write_text(
        "class Reader:\n"
        "    def write(self, lines):\n"
        "        return list(lines)\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    (tmp_path / "workspace").mkdir()
    (tmp_path / "workspace" / "patch.diff").write_text(
        "diff --git a/pkg/module.py b/pkg/module.py\n"
        "--- a/pkg/module.py\n"
        "+++ b/pkg/module.py\n"
        "@@ -1,3 +1,7 @@\n"
        " class Reader:\n"
        "+    def __init__(self, header_rows=None):\n"
        "+        if header_rows is not None:\n"
        "+            return header_rows\n"
        "+\n"
        "     def write(self, lines):\n"
        "         return list(lines)\n",
        encoding="utf-8",
    )
    task = TaskSpec(
        task_id="swe",
        prompt="verify swe patch",
        workspace_subdir="workspace",
        expected_files=["patch.diff"],
        metadata={
            "semantic_verifier": {
                "kind": "swe_patch_apply_check",
                "repo": "owner/repo",
                "base_commit": commit,
                "repo_cache_root": str(tmp_path / "repos"),
                "patch_path": "patch.diff",
                "expected_changed_paths": ["pkg/module.py"],
            }
        },
    )

    verification = Verifier().verify(
        task,
        tmp_path / "workspace",
        CommandResult(command="write", exit_code=0, stdout="", stderr=""),
    )

    assert verification.passed is False
    assert "SWE patch leaves invalid __init__ return values in pkg/module.py: Reader.__init__" in verification.reasons


def test_verifier_swe_patch_apply_check_rejects_init_generator(tmp_path):
    repo_root = tmp_path / "repos" / "owner" / "repo"
    source = repo_root / "pkg" / "module.py"
    source.parent.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_root, check=True)
    source.write_text(
        "class Reader:\n"
        "    def __init__(self, rule):\n"
        "        self.rule = rule\n"
        "\n"
        "    def values(self):\n"
        "        return []\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    (tmp_path / "workspace").mkdir()
    (tmp_path / "workspace" / "patch.diff").write_text(
        "diff --git a/pkg/module.py b/pkg/module.py\n"
        "--- a/pkg/module.py\n"
        "+++ b/pkg/module.py\n"
        "@@ -1,6 +1,6 @@\n"
        " class Reader:\n"
        "     def __init__(self, rule):\n"
        "-        self.rule = rule\n"
        "+        yield from rule.values()\n"
        " \n"
        "     def values(self):\n"
        "         return []\n",
        encoding="utf-8",
    )
    task = TaskSpec(
        task_id="swe",
        prompt="verify swe patch",
        workspace_subdir="workspace",
        expected_files=["patch.diff"],
        metadata={
            "semantic_verifier": {
                "kind": "swe_patch_apply_check",
                "repo": "owner/repo",
                "base_commit": commit,
                "repo_cache_root": str(tmp_path / "repos"),
                "patch_path": "patch.diff",
                "expected_changed_paths": ["pkg/module.py"],
            }
        },
    )

    verification = Verifier().verify(
        task,
        tmp_path / "workspace",
        CommandResult(command="write", exit_code=0, stdout="", stderr=""),
    )

    assert verification.passed is False
    assert "SWE patch leaves invalid __init__ generators in pkg/module.py: Reader.__init__" in verification.reasons


def test_verifier_swe_patch_apply_check_rejects_local_use_before_assignment(tmp_path):
    repo_root = tmp_path / "repos" / "owner" / "repo"
    source = repo_root / "pkg" / "module.py"
    source.parent.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_root, check=True)
    source.write_text(
        "def raw_strict(value):\n"
        "    return bool(value)\n"
        "\n"
        "def raw_normal(value):\n"
        "    return True\n"
        "\n"
        "def choose(strict, values):\n"
        "    if strict:\n"
        "        raw_type_fn = raw_strict\n"
        "    else:\n"
        "        raw_type_fn = raw_normal\n"
        "    return all(raw_type_fn(value) for value in values)\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    (tmp_path / "workspace").mkdir()
    (tmp_path / "workspace" / "patch.diff").write_text(
        "diff --git a/pkg/module.py b/pkg/module.py\n"
        "--- a/pkg/module.py\n"
        "+++ b/pkg/module.py\n"
        "@@ -5,7 +5,7 @@\n"
        " def raw_normal(value):\n"
        "     return True\n"
        " \n"
        " def choose(strict, values):\n"
        "-    if strict:\n"
        "+    if not any(raw_type_fn(value) for value in values):\n"
        "         raw_type_fn = raw_strict\n"
        "     else:\n"
        "         raw_type_fn = raw_normal\n",
        encoding="utf-8",
    )
    task = TaskSpec(
        task_id="swe",
        prompt="verify swe patch",
        workspace_subdir="workspace",
        expected_files=["patch.diff"],
        metadata={
            "semantic_verifier": {
                "kind": "swe_patch_apply_check",
                "repo": "owner/repo",
                "base_commit": commit,
                "repo_cache_root": str(tmp_path / "repos"),
                "patch_path": "patch.diff",
                "expected_changed_paths": ["pkg/module.py"],
            }
        },
    )

    verification = Verifier().verify(
        task,
        tmp_path / "workspace",
        CommandResult(command="write", exit_code=0, stdout="", stderr=""),
    )

    assert verification.passed is False
    assert "SWE patch introduces local use before assignment in pkg/module.py: choose.raw_type_fn" in verification.reasons


def test_python_static_artifact_helpers_classify_live_failure_shapes():
    before_registration = (
        "class Rule:\n"
        "    def scan(self):\n"
        "        return False\n"
        "\n"
        "check = Rule()\n"
    )
    after_registration = (
        "class Rule:\n"
        "    def scan(self):\n"
        "        return False\n"
        "        self.evaluated_keys = ['rule']\n"
    )
    assert _removed_python_module_registration_names(before_registration, after_registration) == ["check"]

    before_unresolved = "def evaluate(flag):\n    return flag\n"
    after_unresolved = "def evaluate(flag):\n    return missing_flag or flag\n"
    assert _introduced_python_unresolved_name_loads(before_unresolved, after_unresolved) == [
        "evaluate.missing_flag"
    ]

    before_return = "def load(path):\n    if path:\n        return path\n    return None\n"
    after_return = "def load(path):\n    if path:\n        value = path\n    return None\n"
    assert _python_removed_return_value_paths(before_return, after_return) == ["load"]

    before_bool = "def condition(value):\n    if value:\n        return False\n    return True\n"
    after_bool = "def condition(value):\n    if value:\n        return False\n    return False\n"
    assert _python_suspicious_boolean_return_flip_names(before_bool, after_bool) == ["condition"]

    assert _is_disallowed_swe_solution_path("tests/test_feature.py") is True
    assert _is_disallowed_swe_solution_path("scripts/update_snapshot_results.sh") is True
    assert _is_disallowed_swe_solution_path(".github/workflows/release.yml") is True
    assert _is_disallowed_swe_solution_path("features/recovery.feature") is True
    assert _is_disallowed_swe_solution_path("galleries/examples/axes_grid1/parasite_simple.py") is True
    assert _is_disallowed_swe_solution_path("lib/matplotlib/mpl-data/matplotlibrc") is True
    assert _is_disallowed_swe_solution_path("ci/requirements/environment.yml") is True
    assert _is_disallowed_swe_solution_path("src/package/feature.py") is False
    assert _python_suspicious_line_replacement_details(
        ["    tags = ['functions']"],
        ["    validator.context.path.path[0] == 'Resources'"],
    ) == ["tags assignment replaced by validator.context.path.path[0] == 'Re..."]
    assert _python_suspicious_line_replacement_details(
        ["        skipped_checks,"],
        ["        end_line = attr_value.end_mark.line"],
    ) == ["call/list argument replaced by end_line assignment"]
    assert _suspicious_config_key_replacement_details(
        ["  logs_dir: ~/.local/share/app/logs"],
        ["  learning_rate: 2e-5"],
    ) == ["logs_dir replaced by learning_rate"]
    assert _python_duplicate_surrounding_call_wrapper_details(
        [
            (" ", "        grouped.append("),
            ("-", "            item.rename(target)"),
            ("+", "        grouped.append(item.rename(target))"),
            (" ", "        )"),
        ]
    ) == ["grouped.append wrapper duplicated inside existing call"]
    assert _python_duplicate_surrounding_call_wrapper_details(
        [
            (" ", "    return array_api_compat.result_type("),
            ("-", "        *map(preprocess_scalar_types, arrays_and_dtypes), xp=xp"),
            ("+", "        array_api_compat.result_type(*map(preprocess_scalar_types, arrays_and_dtypes), xp=xp)"),
            (" ", "    )"),
        ]
    ) == ["array_api_compat.result_type wrapper duplicated inside existing call"]
    assert _python_duplicate_existing_statement_replacement_details(
        [
            (" ", 'host.set_xlabel("Distance")'),
            (" ", 'host.set_ylabel("Density")'),
            ("-", 'par.set_ylabel("Temperature")'),
            ("+", 'host.set_xlabel("Distance")'),
            (" ", 'p1, = host.plot([0, 1, 2], [0, 1, 2], label="Density")'),
        ]
    ) == ['host.set_xlabel("Distance") duplicates existing hunk statement']
    before_attr = "def handle(line_info):\n    ifun = line_info.ifun\n    return ifun\n"
    after_attr = "def handle(line_info):\n    ifun = line_info.function\n    return ifun\n"
    assert _python_suspicious_attribute_replacement_details(before_attr, after_attr) == [
        "ifun: line_info.ifun replaced by unknown line_info.function"
    ]
    before_private = "def load(sheet):\n    dset = Dataset()\n    return dset\n"
    after_private = "def load(sheet):\n    dset = Dataset(headers=sheet.headers, data=sheet._data)\n    return dset\n"
    assert _python_introduced_private_attribute_accesses(before_private, after_private) == ["sheet._data"]
    before_self_private = "class EnumModel:\n    def find_member(self, value):\n        return None\n"
    after_self_private = "class EnumModel:\n    def find_member(self, value):\n        return self._default_members\n"
    assert _python_introduced_unknown_self_private_attribute_accesses(before_self_private, after_self_private) == [
        "_default_members"
    ]
    assert _suspicious_semantic_token_flip_details(
        ['            ("{{last_name}} {{first_name_female}} {{middle_name_female}}", 0.1),'],
        ['            ("{{last_name}} {{first_name_male}} {{middle_name_male}}", 0.1),'],
    ) == ["female token replaced by male"]
    before_reraise = "def parse():\n    try:\n        run()\n    except ValueError:\n        raise\n"
    after_reraise = "def parse():\n    try:\n        run()\n    except ValueError:\n        raise ValueError('Invalid date')\n"
    assert _python_exception_contract_regression_details(before_reraise, after_reraise) == [
        "parse: bare re-raise replaced by ValueError('Invalid date')"
    ]
    before_condition = "def train(data):\n    if not data:\n        raise Exception('missing')\n"
    after_condition = "def train(data):\n    if not isinstance(e, KeyboardInterrupt):\n        raise Exception('missing')\n"
    assert _python_exception_contract_regression_details(before_condition, after_condition) == [
        "train: condition replaced by exception-type check"
    ]
    before_lower = "@lowercase\ndef email(self):\n    return self.user_name()\n"
    after_lower = "@lowercase\ndef email(self):\n    return self.user_name().lower()\n"
    assert _python_redundant_decorated_normalization_details(before_lower, after_lower) == [
        "email: .lower()"
    ]
    assert _python_suspicious_line_replacement_details(
        ["            (filename, bp_line) = breakpoint.rsplit(':', 1)"],
        ["            import timeit"],
    ) == ["assignment replaced by import statement"]
    assert _python_string_literal_only_changed(
        "def log(file_path):\n    logger.info(f'Document at {file_path} Tring schema')\n",
        "def log(file_path):\n    logger.info(f'Document at {file_path} Trying schema')\n",
    )
    assert _python_annotation_only_changed(
        "def item(value=''):\n    return value\n",
        "def item(value: str = ''):\n    return value\n",
    )
    assert _python_indentation_only_statement_moves(
        "def f(flag):\n    if flag:\n        logger.info('x')\n    return 1\n",
        "def f(flag):\n    if flag:\n            logger.info('x')\n    return 1\n",
    ) == ["logger.info('x')"]
    assert _python_introduced_none_return_value_paths(
        "def value():\n    return 1\n",
        "def value():\n    if bad:\n        return None\n    return 1\n",
    ) == ["value"]
    assert _python_suspicious_line_replacement_details(
        ['                f"Parameter {param.name}"'],
        ["    arbitrary_types_allowed=True,"],
    ) == ["expr replaced by arbitrary_types_allowed assignment"]
    assert _python_suspicious_line_replacement_details(
        ["        raise e"],
        ["        raise ValueError('Invalid date')"],
    ) == ["raise expression replaced by raise ValueError('Invalid date')"]
    assert _python_suspicious_line_replacement_details(
        ["        return self.generator.random.uniform(min_value, max_value)"],
        ["        return None"],
    ) == ["return expression replaced by return None"]
    assert _python_suspicious_line_replacement_details(
        ["    if failed:"],
        ["    if not isinstance(e, KeyboardInterrupt):"],
    ) == ["if condition replaced by exception-type check"]
    assert _python_suspicious_line_replacement_details(
        ["    if slash_command in lm_provider_klass.unsupported_slash_commands:"],
        ["    if self.preferred_dir and os.path.exists(self.preferred_dir):"],
    ) == ["if condition replaced by unrelated condition"]
    assert _python_suspicious_line_replacement_details(
        ["        return persona"],
        ["        return JupyternautPersona"],
    ) == ["return variable replaced by return JupyternautPersona"]
    assert _python_suspicious_line_replacement_details(
        ["            return not (self.__class__._stream is BaseChatModel._stream)"],
        ["            return not (self.__class__._stream is BaseChatModel._stream) or self._supports_sync_streaming"],
    ) == ["return expression introduces self-reference: return not (self.__class__._stream is..."]
    assert _python_suspicious_line_replacement_details(
        ["            return tf.math.reduce_max(tf.math.abs(x), axis=axis, keepdims=keepdims"],
        ["            return tf.math.reduce_max(tf.math.reduce_max(tf.math.abs(x), axis=axis, keepdims=keepdims))"],
    ) == ["return expression nests duplicate reducer tf.math.reduce_max"]
    assert _python_nested_duplicate_reducer_details(
        "def norm(x, axis, keepdims):\n    return tf.math.reduce_max(tf.math.abs(x), axis=axis, keepdims=keepdims)\n",
        "def norm(x, axis, keepdims):\n    return tf.math.reduce_max(tf.math.reduce_max(tf.math.abs(x), axis=axis, keepdims=keepdims))\n",
    ) == ["tf.math.reduce_max"]
    assert _python_suspicious_line_replacement_details(
        ["        return self.backend.cast(inputs, dtype) * scale + offset"],
        ["        return self.backend.cast(inputs, dtype) * self.backend.cast(scale, dtype) + self.backend.cast(offset, dtype)"],
    ) == ["return expression broadly casts existing operands: offset, scale"]
    assert _python_suspicious_line_replacement_details(
        ["            (self.is_optional and not self.use_union_operator, IMPORT_OPTIONAL),"],
        ["            (self.is_list, IMPORT_LIST),"],
    ) == [
        "tuple element contract changed: 0:self.is_optional and (not self.use_union_operator)->self.is_list, 1:IMPORT_OPTIONAL->IMPORT_LIST"
    ]
    assert _python_suspicious_line_replacement_details(
        ["        return super().start_section(heading if namespace.no_color or not heading else self._bold_cyan(heading))"],
        [
            "        return super().start_section(heading if namespace.no_color or not heading else self._bold_cyan(heading) or None)"
        ],
    ) == ["return super call broadened with or None"]
    assert _python_suspicious_line_replacement_details(
        ["                position_axis += free_space / (len(line) + 1)"],
        ["                position_axis += free_space / (len(line) - 1)"],
    ) == ["arithmetic literal contract changed near free_space+len+line: ['Div', 'Add', '1']->['Div', 'Sub', '1']"]
    assert _python_suspicious_line_replacement_details(
        ["        return to_hashable(item.dict())"],
        ["        return to_hashable(item.dict(exclude_none=True))"],
    ) == ["serialization call adds exclude_none=True to item.dict"]
    assert _python_suspicious_line_replacement_details(
        ["            block.block_level_width(child_copy, parent_box)"],
        ["            axis, cross = 'height', 'width'"],
    ) == ["expr replaced by axis, cross assignment"]
    assert _python_suspicious_line_replacement_details(
        ["        match_submerged_margins(layoutgrids, fig)"],
        ["        ' match_submerged_margins(layoutgrids, fig)'"],
    ) == ["expression replaced by literal ' match_submerged_margins(layoutgrids..."]
    assert _python_suspicious_line_replacement_details(
        ["            self.args.extend(exec_config['args'])"],
        ["                raise ConfigException('exec: malformed response')"],
    ) == ["expression replaced by raise statement"]
    assert _python_suspicious_line_replacement_details(
        ["        self.add_line(l)"],
        ['        self._check_no_units([xmin, xmax], ["xmin", "xmax"])'],
    ) == ["call expression replaced by unrelated call self.add_line->self._check_no_units"]
    assert _python_suspicious_line_replacement_details(
        ["        return 9"],
        ["        return 10"],
    ) == ["return constant changed from 9 to 10"]
    assert _python_suspicious_line_replacement_details(
        ["        return self._size * padding[self._tickdir]"],
        ["        return self._tickdir"],
    ) == ["return expression simplified to return self._tickdir"]
    assert _python_suspicious_line_replacement_details(
        ["        if self._kv_cache:"],
        ["        if not self._response:"],
    ) == ["if condition replaced by unrelated attribute condition"]
    assert _python_suspicious_hunk_replacement_details(
        [
            (" ", "        image_node['caption'] = self.options.get('caption', None)"),
            ("+", '        image_node["caption"] = self.options.get("caption", None)'),
        ]
    ) == ["image_node['caption'] assignment duplicates existing hunk target"]
    assert _python_suspicious_hunk_replacement_details(
        [
            ("-", "        elif closed and len(vertices):"),
            ("+", "            if closed and len(vertices):"),
        ]
    ) == ["elif branch nested as if closed and len(vertices)"]
    assert _python_suspicious_hunk_replacement_details(
        [
            ("-", "        return ModelSettings("),
            ("+", "            return override"),
        ]
    ) == ["return constructor block collapsed into nested return override"]
    assert _python_suspicious_hunk_replacement_details(
        [
            ("+", "            if alias and alias == node.name:"),
            (" ", "                if alias == node.name:"),
        ]
    ) == ["if condition duplicates nested hunk condition"]
    assert _python_suspicious_hunk_replacement_details(
        [
            (" ", "        and not netloc  # Not UNC."),
            ("+", "        and not netloc  # Not UNC."),
        ]
    ) == ["boolean continuation duplicates existing hunk clause and not netloc # Not UNC."]
    assert _suspicious_text_template_replacement_details(
        [r"  ^\s+\(Watts\)"],
        [r"  ^\s*$$ -> Interfaces"],
    ) == [r"regex matcher replaced by state transition ^\s*$$ -> Interfaces"]
    assert _python_suspicious_line_replacement_details(
        ["            pattern.match(name) for pattern in self._good_names_rgxs_compiled"],
        ["            prop.rsplit('.', 1)[-1] for prop in config.property_classes"],
    ) == ["generator expression replaced by unrelated generator"]
    assert _python_suspicious_line_replacement_details(
        ["    while _node and _node.fromlineno == lineno:"],
        ["    while _node and _node.fromlineno == lineno and _node.nodes_of_class(nodes.AssignName):"],
    ) == ["while condition introduces call _node.nodes_of_class"]
    assert _python_suspicious_line_replacement_details(
        ["        self.names_under_always_false_test: set[str] = set()"],
        ["        self.names_under_always_false_test = set()"],
    ) == ["self.names_under_always_false_test annotation removed"]
    assert _python_suspicious_line_replacement_details(
        ["            line = line.encode(catalog.charset, 'backslashreplace')"],
        ['            line = line.encode(catalog.charset, errors="backslashreplace")'],
    ) == ["line.encode call converts positional argument to keyword errors"]
    assert _python_suspicious_line_replacement_details(
        ["                masked=False,"],
        ["                masked=not self.train_on_input,"],
    ) == ["masked keyword constant replaced by expression"]
    assert _python_suspicious_line_replacement_details(
        ["    password: str | None = dc.field(default=None, repr=False)"],
        ["    password: str | None = None"],
    ) == ["password dataclass field replaced by constant default"]
    assert _python_suspicious_line_replacement_details(
        ["    except astroid.InferenceError:"],
        ["    except StopIteration:"],
    ) == ["except handler replaced by except StopIteration:"]
    assert _python_suspicious_line_replacement_details(
        ["        request.all_data[parameter] = value.format(*m.groups())"],
        ["        request.all_data[parameter] = value"],
    ) == ["format call removed from value"]
    assert _python_suspicious_line_replacement_details(
        ['                        "choices": list(NAMING_STYLES.keys()),'],
        ['                        "default": default_style,'],
    ) == ["dict key 'choices' replaced by 'default'"]
    assert _python_suspicious_line_replacement_details(
        ["        return _get_axes(line_array)"],
        ["        return _get_axes(line_array.axes)"],
    ) == ["return _get_axes call changes first argument shape"]
    assert _python_suspicious_line_replacement_details(
        ["        return DependencyPackage(dependency, package)"],
        ["        return DependencyPackage(dependency, package, source_name=explicit_source)"],
    ) == ["return DependencyPackage call broadens arguments"]
    assert _python_suspicious_line_replacement_details(
        ['            msg = f"Failed to download {args.repo_id} with error: {e}"'],
        ['            msg = f"Repository {args.repo_id} is gated. Please check access."'],
    ) == ["formatted message rewritten with unrelated text"]
    assert _python_suspicious_line_replacement_details(
        ["    if format not in patterns:"],
        ['    if format not in patterns or format == "":'],
    ) == ["if condition broadened with extra clause"]
    assert _python_suspicious_line_replacement_details(
        ['                message.role != "assistant" or index != len(messages) - 1'],
        ['                message.role != "assistant" or (index != len(messages) - 1 and message.role != "user")'],
    ) == ["boolean expression broadened with extra clause"]
    assert _python_suspicious_line_replacement_details(
        ["                    isinstance(handler.type, nodes.Const) and handler.type.value is None"],
        ["                    not utils.inherit_from_std_ex(exc)"],
    ) == ["boolean expression replaced by unrelated call"]
    assert _python_suspicious_line_replacement_details(
        ["    K = result.gain_matrix"],
        ["    K = place_varga(A, B, placed_eigs, dtime=False)"],
    ) == ["K assignment source replaced by unrelated call"]
    assert _python_suspicious_line_replacement_details(
        ["        n not in set(consumed_nodes)"],
        ["        n not in consumed_nodes"],
    ) == ["membership container set wrapper removed"]
    assert _python_suspicious_line_replacement_details(
        ["        COMMAND_NOT_FOUND_PREFIX_MESSAGE"],
        ['        f"Command {command} not found"'],
    ) == ["message constant COMMAND_NOT_FOUND_PREFIX_MESSAGE replaced by formatted string"]
    assert _python_suspicious_line_replacement_details(
        ["                    io.write_error_line(COMMAND_NOT_FOUND_PREFIX_MESSAGE)"],
        ['                    io.write_error_line(f"The requested command does not exist in the <c1>{command}</> namespace.")'],
    ) == ["message constant COMMAND_NOT_FOUND_PREFIX_MESSAGE replaced by formatted string"]
    assert _python_suspicious_line_replacement_details(
        ['        segment.type in ("select_statement", "set_expression")'],
        ['        segment.type in ["table_reference", "object_reference"]'],
    ) == ["membership container tuple replaced by list"]
    assert _python_suspicious_line_replacement_details(
        ['        {"old_names": [deprecated_class], "shared": True},'],
        ['        "deprecated-class",'],
    ) == ["dictionary literal replaced by scalar literal"]
    assert _python_suspicious_hunk_replacement_details(
        [
            ('-', '                self.pyproject.dependency_groups.pop(group, None)'),
            ('+', '                del self.pyproject._data["tool"]["pdm"]'),
        ]
    ) == ["delete target replaced with unrelated private data deletion"]
    assert _python_suspicious_line_replacement_details(
        ["        isinstance(config, str) and custom_objects and custom_objects.get(config) is not None"],
        ["        isinstance(config, str) and custom_objects and custom_objects.get(config) is not None or config.get('registered_name') in custom_objects"],
    ) == ["boolean condition introduces unguarded mapping get config.get"]
    assert _python_suspicious_line_replacement_details(
        ["        and custom_objects.get(config) is not None"],
        ['        or config.get("registered_name") in custom_objects'],
    ) == ["boolean condition introduces unguarded mapping get config.get"]
    assert _is_disallowed_swe_solution_path("docs/source/users/index.md")
    assert _is_disallowed_swe_solution_path(".secrets.baseline")
    assert _is_disallowed_swe_solution_path("Makefile")
    assert _is_disallowed_swe_solution_path("CHANGELOG.rst")
    assert _python_suspicious_line_replacement_details(
        ["                c.get_name() for c in parent_state.class_subclasses"],
        ["            parent_state.class_subclasses.add(cls)"],
    ) == ["iterable expression replaced by mutating call parent_state.class_subclasses.add"]
    assert _python_suspicious_line_replacement_details(
        ["        cls.__fields__.update({var_name: new_field})"],
        ["    from reflex.utils.exceptions import VarNameError"],
    ) == ["expression replaced by from statement"]
    assert _python_suspicious_line_replacement_details(
        ["                if parent_state is not None and parent_state.event_handlers.get(name):"],
        ["                if parent_state.event_handlers.get(name):"],
    ) == ["if condition removed None guard before parent_state attribute access"]
    assert _python_suspicious_line_replacement_details(
        ["                    next_layer.append(child)"],
        ["    if sources in G:"],
    ) == ["expression replaced by if statement"]
    assert _suspicious_text_template_replacement_details(
        ["export default function MyApp({ Component, pageProps }) {"],
        ["{% for library_alias, library_path in  window_libraries %}"],
    ) == ["template control replaced executable function export default function MyApp({ Compo..."]


def test_python_static_guards_reject_none_container_and_function_object_arithmetic():
    assert _python_introduced_none_container_misuse_details(
        "def get_datetime_format(locale):\n    patterns = Locale.parse(locale).datetime_formats\n    if format not in patterns:\n        format = None\n    return patterns[format]\n",
        "def get_datetime_format(locale):\n    patterns = None\n    if format not in patterns:\n        format = None\n    return patterns[format]\n",
    ) == ["patterns:membership", "patterns:subscript"]
    assert _python_introduced_function_object_arithmetic_details(
        "def julian_day(utime):\n    return 1\n\ndef f(utime, delta_t):\n    jd = julian_day(utime)\n    jde = jd + delta_t\n    return jde\n",
        "def julian_day(utime):\n    return 1\n\ndef f(utime, delta_t):\n    jd = julian_day(utime)\n    jde = julian_day + delta_t * 1.0 / 86400\n    return jde\n",
    ) == ["julian_day:binop"]


def test_verifier_swe_patch_apply_check_allows_preexisting_init_return_value(tmp_path):
    repo_root = tmp_path / "repos" / "owner" / "repo"
    source = repo_root / "pkg" / "module.py"
    source.parent.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_root, check=True)
    source.write_text(
        "class Reader:\n"
        "    def __init__(self, rows=None):\n"
        "        if rows is not None:\n"
        "            return rows\n"
        "        self.rows = []\n"
        "\n"
        "    def write(self, lines):\n"
        "        return list(lines)\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    (tmp_path / "workspace").mkdir()
    (tmp_path / "workspace" / "patch.diff").write_text(
        "diff --git a/pkg/module.py b/pkg/module.py\n"
        "--- a/pkg/module.py\n"
        "+++ b/pkg/module.py\n"
        "@@ -5,4 +5,4 @@\n"
        "         self.rows = []\n"
        " \n"
        "     def write(self, lines):\n"
        "-        return list(lines)\n"
        "+        return tuple(lines)\n",
        encoding="utf-8",
    )
    task = TaskSpec(
        task_id="swe",
        prompt="verify swe patch",
        workspace_subdir="workspace",
        expected_files=["patch.diff"],
        metadata={
            "semantic_verifier": {
                "kind": "swe_patch_apply_check",
                "repo": "owner/repo",
                "base_commit": commit,
                "repo_cache_root": str(tmp_path / "repos"),
                "patch_path": "patch.diff",
                "expected_changed_paths": ["pkg/module.py"],
            }
        },
    )

    verification = Verifier().verify(
        task,
        tmp_path / "workspace",
        CommandResult(command="write", exit_code=0, stdout="", stderr=""),
    )

    assert verification.passed is True


def test_verifier_applies_behavior_check_regex_output_rules(tmp_path):
    task = TaskSpec(
        task_id="regex_behavior",
        prompt="verify regex behavior",
        workspace_subdir="regex_behavior",
        metadata={
            "semantic_verifier": {
                "behavior_checks": [
                    {
                        "label": "json report",
                        "argv": ["python", "-c", "print('status=ready\\ncount=3')"],
                        "stdout_regex_must_match": [r"status=ready", r"count=\d+"],
                        "stdout_regex_must_not_match": [r"error"],
                    }
                ]
            }
        },
    )
    result = CommandResult(command="true", exit_code=0, stdout="", stderr="")

    verification = Verifier().verify(task, tmp_path, result)

    assert verification.passed is True


def test_verifier_applies_json_field_regex_rules(tmp_path):
    task = TaskSpec(
        task_id="json_regex_behavior",
        prompt="verify json regex behavior",
        workspace_subdir="json_regex_behavior",
        metadata={
            "semantic_verifier": {
                "behavior_checks": [
                    {
                        "label": "json behavior",
                        "argv": [
                            "python",
                            "-c",
                            "import json; print(json.dumps({'status': 'ready', 'message': 'release ready for deploy'}))",
                        ],
                        "stdout_json_fields": [
                            {"path": "status", "regex": r"rea.*"},
                            {"path": "message", "not_regex": r"failed|error"},
                        ],
                    }
                ]
            }
        },
    )
    result = CommandResult(command="true", exit_code=0, stdout="", stderr="")

    verification = Verifier().verify(task, tmp_path, result)

    assert verification.passed is True


def test_verifier_emits_structured_failure_codes_and_process_score(tmp_path):
    task = TaskSpec(
        task_id="output",
        prompt="avoid warnings",
        workspace_subdir="output",
        expected_files=["artifact.txt"],
        forbidden_output_substrings=["warning"],
    )
    result = CommandResult(command="echo warning", exit_code=2, stdout="warning\n", stderr="")

    verification = Verifier().verify(task, tmp_path, result)
    payload = verification.to_payload()

    assert verification.passed is False
    assert verification.failure_codes[:3] == [
        "command_failure",
        "missing_expected_file",
        "forbidden_output_present",
    ]
    assert verification.outcome_label == "command_failure"
    assert verification.process_score < 1.0
    assert payload["failure_codes"] == verification.failure_codes
    assert payload["outcome_label"] == "command_failure"


def test_verifier_classifies_terminal_boundary_failures(tmp_path):
    task = TaskSpec(
        task_id="terminal_failure",
        prompt="run tests",
        workspace_subdir="terminal_failure",
    )
    result = CommandResult(
        command="pytest tests/test_config.py -q",
        exit_code=1,
        stdout="FAILED tests/test_config.py::test_parse - AssertionError: expected 3 got 2",
        stderr="",
    )

    verification = Verifier().verify(task, tmp_path, result)

    assert "command_failure" in verification.failure_codes
    assert "terminal_test_assertion_failure" in verification.failure_codes
    assert any(
        item.get("kind") == "terminal_failure_classification"
        and item.get("failure_class") == "test_assertion_failure"
        for item in verification.evidence
    )


def test_verifier_classifies_import_errors(tmp_path):
    task = TaskSpec(
        task_id="import_failure",
        prompt="run script",
        workspace_subdir="import_failure",
    )
    result = CommandResult(
        command="python script.py",
        exit_code=1,
        stdout="",
        stderr="ModuleNotFoundError: No module named 'yaml'",
    )

    verification = Verifier().verify(task, tmp_path, result)

    assert "terminal_import_error" in verification.failure_codes
    assert any(item.get("failure_class") == "import_error" for item in verification.evidence)


def test_verifier_classifies_no_op_edits(tmp_path):
    task = TaskSpec(
        task_id="noop_edit",
        prompt="produce patch",
        workspace_subdir="noop_edit",
        expected_files=["patch.diff"],
    )
    result = CommandResult(
        command="swe_patch_builder --path src/app.py --replace-line 10 --with pass > patch.diff",
        exit_code=2,
        stdout="",
        stderr="swe_patch_builder replacement produced no meaningful change",
    )

    verification = Verifier().verify(task, tmp_path, result)

    assert "terminal_no_op_edit" in verification.failure_codes
    assert any(item.get("failure_class") == "no_op_edit" for item in verification.evidence)


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


def test_sandbox_allows_bounded_sleep_and_times_out(tmp_path):
    quick = Sandbox(timeout_seconds=1).run("sleep 0", tmp_path)
    slow = Sandbox(timeout_seconds=1).run("sleep 2", tmp_path)

    assert quick.exit_code == 0
    assert slow.exit_code == 124
    assert slow.timed_out is True


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


def test_sandbox_normalizes_simple_cat_heredoc_to_bounded_redirection(tmp_path):
    result = Sandbox(timeout_seconds=1).run(
        "cat > reports/status.txt << 'EOF'\nready\nset\nEOF",
        tmp_path,
    )

    assert result.exit_code == 0
    assert result.stderr == ""
    assert (tmp_path / "reports" / "status.txt").read_text(encoding="utf-8") == "ready\nset\n"


def test_sandbox_normalizes_mkdir_cat_heredoc_to_bounded_redirection(tmp_path):
    result = Sandbox(timeout_seconds=1).run(
        "mkdir -p reports && cat > reports/status.txt << EOF\nready\nset\nEOF",
        tmp_path,
    )

    assert result.exit_code == 0
    assert result.stderr == ""
    assert (tmp_path / "reports" / "status.txt").read_text(encoding="utf-8") == "ready\nset\n"


def test_sandbox_treats_heredoc_body_as_literal_content_for_blocked_tokens(tmp_path):
    result = Sandbox(timeout_seconds=1).run(
        "cat > reports/patch.diff << 'EOF'\n"
        "diff --git a/docs/table.md b/docs/table.md\n"
        "--- a/docs/table.md\n"
        "+++ b/docs/table.md\n"
        "@@ -1 +1 @@\n"
        "-`table_io` more text\n"
        "+`table_io` more literal text\n"
        "EOF",
        tmp_path,
    )

    assert result.exit_code == 0
    assert result.stderr == ""
    assert "`table_io` more literal text" in (tmp_path / "reports" / "patch.diff").read_text(encoding="utf-8")


def test_sandbox_swe_patch_builder_creates_applyable_candidate_diff(tmp_path):
    repo_root = tmp_path / "repos" / "owner" / "repo"
    repo_source = repo_root / "pkg" / "module.py"
    repo_source.parent.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_root, check=True)
    repo_source.write_text("def value():\n    return 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    workspace = tmp_path / "workspace"
    workspace_source = workspace / "pkg" / "module.py"
    workspace_source.parent.mkdir(parents=True)
    workspace_source.write_text("def value():\n    return 1\n", encoding="utf-8")
    task = TaskSpec(
        task_id="swe_builder",
        prompt="build a patch",
        workspace_subdir="workspace",
        expected_files=["patch.diff"],
        metadata={
            "swe_candidate_files": ["pkg/module.py"],
            "semantic_verifier": {
                "kind": "swe_patch_apply_check",
                "repo": "owner/repo",
                "base_commit": commit,
                "repo_cache_root": str(tmp_path / "repos"),
                "patch_path": "patch.diff",
                "expected_changed_paths": ["pkg/module.py"],
            },
        },
    )

    result = Sandbox(timeout_seconds=1).run(
        "swe_patch_builder --path pkg/module.py --replace-line 2 --with '    return 2' > patch.diff",
        workspace,
        task=task,
    )
    verification = Verifier().verify(task, workspace, result)

    assert result.exit_code == 0
    assert "--- a/pkg/module.py" in (workspace / "patch.diff").read_text(encoding="utf-8")
    assert "+    return 2" in (workspace / "patch.diff").read_text(encoding="utf-8")
    assert verification.passed is True


def test_sandbox_swe_patch_builder_allows_literal_backticks_in_replacement(tmp_path):
    repo_root = tmp_path / "repos" / "owner" / "repo"
    repo_source = repo_root / "pkg" / "module.py"
    repo_source.parent.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_root, check=True)
    repo_source.write_text('def value():\n    """old docs"""\n    return 1\n', encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    workspace = tmp_path / "workspace"
    workspace_source = workspace / "pkg" / "module.py"
    workspace_source.parent.mkdir(parents=True)
    workspace_source.write_text('def value():\n    """old docs"""\n    return 1\n', encoding="utf-8")
    task = TaskSpec(
        task_id="swe_builder_backticks",
        prompt="build a patch",
        workspace_subdir="workspace",
        expected_files=["patch.diff"],
        metadata={
            "swe_candidate_files": ["pkg/module.py"],
            "semantic_verifier": {
                "kind": "swe_patch_apply_check",
                "repo": "owner/repo",
                "base_commit": commit,
                "repo_cache_root": str(tmp_path / "repos"),
                "patch_path": "patch.diff",
                "expected_changed_paths": ["pkg/module.py"],
            },
        },
    )

    result = Sandbox(timeout_seconds=1).run(
        'swe_patch_builder --path pkg/module.py --replace-line 2 '
        '--with \'    """new docs for ``value``"""\' > patch.diff',
        workspace,
        task=task,
    )

    assert result.exit_code == 0
    assert "``value``" in (workspace / "patch.diff").read_text(encoding="utf-8")


def test_sandbox_patch_builder_alias_creates_applyable_candidate_diff(tmp_path):
    repo_root = tmp_path / "repos" / "owner" / "repo"
    repo_source = repo_root / "pkg" / "module.py"
    repo_source.parent.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_root, check=True)
    repo_source.write_text("def value():\n    return 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    workspace = tmp_path / "workspace"
    workspace_source = workspace / "pkg" / "module.py"
    workspace_source.parent.mkdir(parents=True)
    workspace_source.write_text("def value():\n    return 1\n", encoding="utf-8")
    task = TaskSpec(
        task_id="patch_builder_alias",
        prompt="build a patch",
        workspace_subdir="workspace",
        expected_files=["patch.diff"],
        metadata={
            "swe_candidate_files": ["pkg/module.py"],
            "semantic_verifier": {
                "kind": "swe_patch_apply_check",
                "repo": "owner/repo",
                "base_commit": commit,
                "repo_cache_root": str(tmp_path / "repos"),
                "patch_path": "patch.diff",
                "expected_changed_paths": ["pkg/module.py"],
            },
        },
    )

    result = Sandbox(timeout_seconds=1).run(
        "patch_builder --path pkg/module.py --replace-line 2 --with '    return 2' > patch.diff",
        workspace,
        task=task,
    )
    verification = Verifier().verify(task, workspace, result)

    assert result.exit_code == 0
    assert "--- a/pkg/module.py" in (workspace / "patch.diff").read_text(encoding="utf-8")
    assert "+    return 2" in (workspace / "patch.diff").read_text(encoding="utf-8")
    assert verification.passed is True


def test_sandbox_swe_patch_builder_creates_multi_replacement_diff(tmp_path):
    repo_root = tmp_path / "repos" / "owner" / "repo"
    repo_source = repo_root / "pkg" / "module.py"
    repo_source.parent.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_root, check=True)
    repo_source.write_text("def left():\n    return 1\n\ndef right():\n    return 2\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    workspace = tmp_path / "workspace"
    workspace_source = workspace / "pkg" / "module.py"
    workspace_source.parent.mkdir(parents=True)
    workspace_source.write_text("def left():\n    return 1\n\ndef right():\n    return 2\n", encoding="utf-8")
    task = TaskSpec(
        task_id="swe_builder",
        prompt="build a patch",
        workspace_subdir="workspace",
        expected_files=["patch.diff"],
        metadata={
            "swe_candidate_files": ["pkg/module.py"],
            "semantic_verifier": {
                "kind": "swe_patch_apply_check",
                "repo": "owner/repo",
                "base_commit": commit,
                "repo_cache_root": str(tmp_path / "repos"),
                "patch_path": "patch.diff",
                "expected_changed_paths": ["pkg/module.py"],
            },
        },
    )

    result = Sandbox(timeout_seconds=1).run(
        "swe_patch_builder --path pkg/module.py "
        "--replace-line 2 --with '    return 10' "
        "--replace-line 5 --with '    return 20' > patch.diff",
        workspace,
        task=task,
    )
    verification = Verifier().verify(task, workspace, result)
    patch = (workspace / "patch.diff").read_text(encoding="utf-8")

    assert result.exit_code == 0
    assert "+    return 10" in patch
    assert "+    return 20" in patch
    assert verification.passed is True


def test_sandbox_swe_patch_builder_splits_embedded_newline_replacement(tmp_path):
    repo_root = tmp_path / "repos" / "owner" / "repo"
    repo_source = repo_root / "pkg" / "module.py"
    repo_source.parent.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_root, check=True)
    repo_source.write_text("def value():\n    return 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    workspace = tmp_path / "workspace"
    workspace_source = workspace / "pkg" / "module.py"
    workspace_source.parent.mkdir(parents=True)
    workspace_source.write_text("def value():\n    return 1\n", encoding="utf-8")
    task = TaskSpec(
        task_id="swe_builder",
        prompt="build a patch",
        workspace_subdir="workspace",
        expected_files=["patch.diff"],
        metadata={
            "swe_candidate_files": ["pkg/module.py"],
            "semantic_verifier": {
                "kind": "swe_patch_apply_check",
                "repo": "owner/repo",
                "base_commit": commit,
                "repo_cache_root": str(tmp_path / "repos"),
                "patch_path": "patch.diff",
                "expected_changed_paths": ["pkg/module.py"],
            },
        },
    )

    result = Sandbox(timeout_seconds=1).run(
        "swe_patch_builder --path pkg/module.py --replace-lines 1 2 "
        "--with 'def value():\n    return 2' > patch.diff",
        workspace,
        task=task,
    )
    patch = (workspace / "patch.diff").read_text(encoding="utf-8")
    verification = Verifier().verify(task, workspace, result)

    assert result.exit_code == 0
    assert "+    return 2" in patch
    assert verification.passed is True


def test_sandbox_swe_patch_builder_accepts_replace_line_range_alias(tmp_path):
    repo_root = tmp_path / "repos" / "owner" / "repo"
    repo_source = repo_root / "pkg" / "module.py"
    repo_source.parent.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_root, check=True)
    repo_source.write_text("def value():\n    total = 1\n    return total\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    workspace = tmp_path / "workspace"
    workspace_source = workspace / "pkg" / "module.py"
    workspace_source.parent.mkdir(parents=True)
    workspace_source.write_text("def value():\n    total = 1\n    return total\n", encoding="utf-8")
    task = TaskSpec(
        task_id="swe_builder_range_alias",
        prompt="build a patch",
        workspace_subdir="workspace",
        expected_files=["patch.diff"],
        metadata={
            "swe_candidate_files": ["pkg/module.py"],
            "semantic_verifier": {
                "kind": "swe_patch_apply_check",
                "repo": "owner/repo",
                "base_commit": commit,
                "repo_cache_root": str(tmp_path / "repos"),
                "patch_path": "patch.diff",
                "expected_changed_paths": ["pkg/module.py"],
            },
        },
    )

    result = Sandbox(timeout_seconds=1).run(
        "swe_patch_builder --path pkg/module.py --replace-line 2-3 "
        "--with '    total = 2' --with '    return total' > patch.diff",
        workspace,
        task=task,
    )
    patch = (workspace / "patch.diff").read_text(encoding="utf-8")
    verification = Verifier().verify(task, workspace, result)

    assert result.exit_code == 0
    assert "+    total = 2" in patch
    assert verification.passed is True


def test_sandbox_swe_patch_builder_preserves_original_indent_for_bare_replacement(tmp_path):
    repo_root = tmp_path / "repos" / "owner" / "repo"
    repo_source = repo_root / "pkg" / "module.py"
    repo_source.parent.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_root, check=True)
    repo_source.write_text("def value():\n    return 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    workspace = tmp_path / "workspace"
    workspace_source = workspace / "pkg" / "module.py"
    workspace_source.parent.mkdir(parents=True)
    workspace_source.write_text("def value():\n    return 1\n", encoding="utf-8")
    task = TaskSpec(
        task_id="swe_builder",
        prompt="build a patch",
        workspace_subdir="workspace",
        expected_files=["patch.diff"],
        metadata={
            "swe_candidate_files": ["pkg/module.py"],
            "semantic_verifier": {
                "kind": "swe_patch_apply_check",
                "repo": "owner/repo",
                "base_commit": commit,
                "repo_cache_root": str(tmp_path / "repos"),
                "patch_path": "patch.diff",
                "expected_changed_paths": ["pkg/module.py"],
            },
        },
    )

    result = Sandbox(timeout_seconds=1).run(
        "swe_patch_builder --path pkg/module.py --replace-line 2 --with 'return 2' > patch.diff",
        workspace,
        task=task,
    )
    patch_text = (workspace / "patch.diff").read_text(encoding="utf-8")
    verification = Verifier().verify(task, workspace, result)

    assert result.exit_code == 0
    assert "+    return 2" in patch_text
    assert verification.passed is True


def test_sandbox_swe_patch_builder_normalizes_single_space_indent_hint(tmp_path):
    repo_root = tmp_path / "repos" / "owner" / "repo"
    repo_source = repo_root / "pkg" / "module.py"
    repo_source.parent.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_root, check=True)
    repo_source.write_text("def value():\n    return 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    workspace = tmp_path / "workspace"
    workspace_source = workspace / "pkg" / "module.py"
    workspace_source.parent.mkdir(parents=True)
    workspace_source.write_text("def value():\n    return 1\n", encoding="utf-8")
    task = TaskSpec(
        task_id="swe_builder",
        prompt="build a patch",
        workspace_subdir="workspace",
        expected_files=["patch.diff"],
        metadata={
            "swe_candidate_files": ["pkg/module.py"],
            "semantic_verifier": {
                "kind": "swe_patch_apply_check",
                "repo": "owner/repo",
                "base_commit": commit,
                "repo_cache_root": str(tmp_path / "repos"),
                "patch_path": "patch.diff",
                "expected_changed_paths": ["pkg/module.py"],
            },
        },
    )

    result = Sandbox(timeout_seconds=1).run(
        "swe_patch_builder --path pkg/module.py --replace-line 2 --with ' return 2' > patch.diff",
        workspace,
        task=task,
    )
    patch_text = (workspace / "patch.diff").read_text(encoding="utf-8")
    verification = Verifier().verify(task, workspace, result)

    assert result.exit_code == 0
    assert "+    return 2" in patch_text
    assert "+ return 2" not in patch_text
    assert verification.passed is True


def test_sandbox_swe_patch_builder_rejects_invalid_python_replacement(tmp_path):
    workspace = tmp_path / "workspace"
    source = workspace / "pkg" / "module.py"
    source.parent.mkdir(parents=True)
    source.write_text("def value():\n    return 1\n", encoding="utf-8")
    task = TaskSpec(
        task_id="swe_builder_invalid_python",
        prompt="build a patch",
        workspace_subdir="workspace",
        expected_files=["patch.diff"],
        metadata={
            "swe_candidate_files": ["pkg/module.py"],
            "semantic_verifier": {
                "kind": "swe_patch_apply_check",
                "patch_path": "patch.diff",
                "expected_changed_paths": ["pkg/module.py"],
            },
        },
    )

    result = Sandbox(timeout_seconds=1).run(
        "swe_patch_builder --path pkg/module.py --replace-line 2 --with '    if (' > patch.diff",
        workspace,
        task=task,
    )

    assert result.exit_code == 2
    assert "swe_patch_builder replacement would produce invalid Python" in result.stderr
    assert (workspace / "patch.diff").read_text(encoding="utf-8") == ""


def test_sandbox_swe_patch_builder_allows_unparseable_baseline_excerpt(tmp_path):
    workspace = tmp_path / "workspace"
    source = workspace / "pkg" / "module.py"
    source.parent.mkdir(parents=True)
    source.write_text("def value():\n    return 1\n\ndef truncated():\n    call(\n", encoding="utf-8")
    task = TaskSpec(
        task_id="swe_builder_unparseable_baseline_excerpt",
        prompt="build a patch",
        workspace_subdir="workspace",
        expected_files=["patch.diff"],
        metadata={
            "swe_candidate_files": ["pkg/module.py"],
            "semantic_verifier": {
                "kind": "swe_patch_apply_check",
                "patch_path": "patch.diff",
                "expected_changed_paths": ["pkg/module.py"],
            },
        },
    )

    result = Sandbox(timeout_seconds=1).run(
        "swe_patch_builder --path pkg/module.py --replace-line 2 --with '    return 2' > patch.diff",
        workspace,
        task=task,
    )

    assert result.exit_code == 0
    assert result.stderr == ""
    assert "+    return 2" in (workspace / "patch.diff").read_text(encoding="utf-8")


def test_sandbox_swe_patch_builder_validates_unparseable_excerpt_against_base_source(tmp_path):
    repo_root = tmp_path / "repos" / "owner" / "repo"
    repo_source = repo_root / "pkg" / "module.py"
    repo_source.parent.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_root, check=True)
    repo_source.write_text("def value():\n    return 1\n\ndef other():\n    return 3\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    workspace = tmp_path / "workspace"
    source = workspace / "pkg" / "module.py"
    source.parent.mkdir(parents=True)
    source.write_text("def value():\n    return 1\n\ndef truncated():\n    call(\n", encoding="utf-8")
    task = TaskSpec(
        task_id="swe_builder_unparseable_excerpt_with_base",
        prompt="build a patch",
        workspace_subdir="workspace",
        expected_files=["patch.diff"],
        metadata={
            "swe_candidate_files": ["pkg/module.py"],
            "semantic_verifier": {
                "kind": "swe_patch_apply_check",
                "repo": "owner/repo",
                "base_commit": commit,
                "repo_cache_root": str(tmp_path / "repos"),
                "patch_path": "patch.diff",
                "expected_changed_paths": ["pkg/module.py"],
            },
        },
    )

    result = Sandbox(timeout_seconds=1).run(
        "swe_patch_builder --path pkg/module.py --replace-line 2 --with '    if (' > patch.diff",
        workspace,
        task=task,
    )

    assert result.exit_code == 2
    assert "swe_patch_builder replacement would produce invalid Python" in result.stderr
    assert (workspace / "patch.diff").read_text(encoding="utf-8") == ""


def test_sandbox_swe_patch_builder_generates_diff_from_base_source_when_workspace_excerpt_truncated(tmp_path):
    repo_root = tmp_path / "repos" / "owner" / "repo"
    repo_source = repo_root / "pkg" / "module.py"
    repo_source.parent.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_root, check=True)
    repo_source.write_text("def value():\n    total = 1\n    return total\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    workspace = tmp_path / "workspace"
    source = workspace / "pkg" / "module.py"
    source.parent.mkdir(parents=True)
    source.write_text("def value():\n    total = 1\n    ret", encoding="utf-8")
    task = TaskSpec(
        task_id="swe_builder_truncated_excerpt_with_base",
        prompt="build a patch",
        workspace_subdir="workspace",
        expected_files=["patch.diff"],
        metadata={
            "swe_candidate_files": ["pkg/module.py"],
            "semantic_verifier": {
                "kind": "swe_patch_apply_check",
                "repo": "owner/repo",
                "base_commit": commit,
                "repo_cache_root": str(tmp_path / "repos"),
                "patch_path": "patch.diff",
                "expected_changed_paths": ["pkg/module.py"],
            },
        },
    )

    result = Sandbox(timeout_seconds=1).run(
        "swe_patch_builder --path pkg/module.py --replace-line 3 --with '    return total + 1' > patch.diff",
        workspace,
        task=task,
    )
    patch_text = (workspace / "patch.diff").read_text(encoding="utf-8")
    verification = Verifier().verify(task, workspace, result)

    assert result.exit_code == 0
    assert "-    return total" in patch_text
    assert "+    return total + 1" in patch_text
    assert "-    ret+" not in patch_text
    assert verification.passed is True


def test_sandbox_swe_patch_builder_rejects_compile_only_python_errors(tmp_path):
    workspace = tmp_path / "workspace"
    source = workspace / "pkg" / "module.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "def value(obj):\n"
        "    return must_be(\n"
        "        \"x\", option=\"option\", obj=obj, id=\"test.E001\"\n"
        "    )\n",
        encoding="utf-8",
    )
    task = TaskSpec(
        task_id="swe_builder_compile_only_error",
        prompt="build a patch",
        workspace_subdir="workspace",
        expected_files=["patch.diff"],
        metadata={
            "swe_candidate_files": ["pkg/module.py"],
            "semantic_verifier": {
                "kind": "swe_patch_apply_check",
                "patch_path": "patch.diff",
                "expected_changed_paths": ["pkg/module.py"],
            },
        },
    )

    result = Sandbox(timeout_seconds=1).run(
        "swe_patch_builder --path pkg/module.py --replace-line 4 "
        "--with \"'x', option='option', obj=obj)\" > patch.diff",
        workspace,
        task=task,
    )

    assert result.exit_code == 2
    assert "keyword argument repeated" in result.stderr
    assert (workspace / "patch.diff").read_text(encoding="utf-8") == ""


def test_sandbox_swe_patch_builder_rejects_non_candidate_path(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "module.py").write_text("def value():\n    return 1\n", encoding="utf-8")
    task = TaskSpec(
        task_id="swe_builder_reject",
        prompt="build a patch",
        workspace_subdir="workspace",
        expected_files=["patch.diff"],
        metadata={"swe_candidate_files": ["pkg/allowed.py"]},
    )

    result = Sandbox(timeout_seconds=1).run(
        "swe_patch_builder --path pkg/module.py --replace-line 2 --with '    return 2' > patch.diff",
        tmp_path,
        task=task,
    )

    assert result.exit_code == 2
    assert "path is not in task candidate files: pkg/module.py" in result.stderr
    assert (tmp_path / "patch.diff").read_text(encoding="utf-8") == ""


def test_sandbox_swe_patch_builder_rejects_from_diff_for_swe_patch_tasks(tmp_path):
    repo_root = tmp_path / "repos" / "owner" / "repo"
    repo_source = repo_root / "pkg" / "module.py"
    repo_source.parent.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_root, check=True)
    repo_source.write_text("def value():\n    return 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    workspace = tmp_path / "workspace"
    workspace_source = workspace / "pkg" / "module.py"
    workspace_source.parent.mkdir(parents=True)
    workspace_source.write_text("def value():\n    return 1\n", encoding="utf-8")
    (workspace / "patch.diff").write_text(
        "diff --git a/pkg/module.py b/pkg/module.py\n"
        "--- a/pkg/module.py\n"
        "+++ b/pkg/module.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def value():\n"
        "-    return 1\n"
        "+    return 2\n",
        encoding="utf-8",
    )
    task = TaskSpec(
        task_id="swe_builder_from_diff",
        prompt="repair a patch",
        workspace_subdir="workspace",
        expected_files=["patch.diff"],
        metadata={
            "swe_candidate_files": ["pkg/module.py"],
            "semantic_verifier": {
                "kind": "swe_patch_apply_check",
                "repo": "owner/repo",
                "base_commit": commit,
                "repo_cache_root": str(tmp_path / "repos"),
                "patch_path": "patch.diff",
                "expected_changed_paths": ["pkg/module.py"],
            },
        },
    )

    result = Sandbox(timeout_seconds=1).run("swe_patch_builder --from-diff patch.diff > patch.diff", workspace, task=task)

    assert result.exit_code == 2
    assert "swe_patch_builder --from-diff is disabled for SWE patch tasks" in result.stderr
    assert (workspace / "patch.diff").read_text(encoding="utf-8") == ""


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


def test_verifier_applies_behavior_checks(tmp_path):
    task = TaskSpec(
        task_id="behavior_semantic_task",
        prompt="Run semantic behavior checks.",
        workspace_subdir="behavior_semantic_task",
        metadata={
            "semantic_verifier": {
                "kind": "behavioral_semantic",
                "behavior_checks": [
                    {
                        "label": "behavior smoke",
                        "argv": ["/bin/sh", "-lc", "printf 'READY\\n'"],
                        "expect_exit_code": 0,
                        "stdout_must_contain": ["READY"],
                        "stdout_must_not_contain": ["BROKEN"],
                    }
                ],
            }
        },
    )
    result = CommandResult(command="true", exit_code=0, stdout="", stderr="")

    verification = Verifier().verify(task, tmp_path, result)

    assert verification.passed is True


def test_verifier_behavior_checks_can_assert_file_expectations_and_repo_invariants(tmp_path):
    task = TaskSpec(
        task_id="behavior_side_effect_task",
        prompt="Run semantic behavior checks with workspace side effects.",
        workspace_subdir="behavior_side_effect_task",
        metadata={
            "semantic_verifier": {
                "kind": "behavioral_semantic",
                "behavior_checks": [
                    {
                        "label": "workspace repair",
                        "argv": [
                            "/bin/sh",
                            "-lc",
                            "mkdir -p reports && printf 'ready\\n' > reports/release_review.txt",
                        ],
                        "expect_exit_code": 0,
                        "file_expectations": [
                            {
                                "path": "reports/release_review.txt",
                                "must_exist": True,
                                "must_contain": ["ready"],
                            }
                        ],
                        "repo_invariants": [
                            {
                                "kind": "file_contains",
                                "path": "reports/release_review.txt",
                                "must_contain": ["ready"],
                                "must_not_contain": ["broken"],
                            }
                        ],
                    }
                ],
            }
        },
    )
    result = CommandResult(command="true", exit_code=0, stdout="", stderr="")

    verification = Verifier().verify(task, tmp_path, result)

    assert verification.passed is True


def test_verifier_applies_behavior_check_json_fields(tmp_path):
    task = TaskSpec(
        task_id="behavior_semantic_json_task",
        prompt="Run semantic behavior checks with structured JSON output.",
        workspace_subdir="behavior_semantic_json_task",
        metadata={
            "semantic_verifier": {
                "kind": "behavioral_semantic",
                "behavior_checks": [
                    {
                        "label": "json smoke",
                        "argv": ["/bin/sh", "-lc", "printf '{\"status\":\"ready\",\"metrics\":{\"pass_rate\":0.75},\"families\":[\"integration\",\"repo\"]}\\n'"],
                        "stdout_json_fields": [
                            {"path": "status", "equals": "ready"},
                            {"path": "metrics.pass_rate", "min": 0.7},
                            {"path": "families", "contains": "integration"},
                        ],
                    }
                ],
            }
        },
    )
    result = CommandResult(command="true", exit_code=0, stdout="", stderr="")

    verification = Verifier().verify(task, tmp_path, result)

    assert verification.passed is True


def test_verifier_rejects_behavior_check_json_field_mismatch(tmp_path):
    task = TaskSpec(
        task_id="behavior_semantic_json_fail_task",
        prompt="Reject incorrect structured JSON output.",
        workspace_subdir="behavior_semantic_json_fail_task",
        metadata={
            "semantic_verifier": {
                "kind": "behavioral_semantic",
                "behavior_checks": [
                    {
                        "label": "json smoke",
                        "argv": ["/bin/sh", "-lc", "printf '{\"status\":\"broken\",\"metrics\":{\"pass_rate\":0.25}}\\n'"],
                        "stdout_json_fields": [
                            {"path": "status", "equals": "ready"},
                            {"path": "metrics.pass_rate", "min": 0.7},
                        ],
                    }
                ],
            }
        },
    )
    result = CommandResult(command="true", exit_code=0, stdout="", stderr="")

    verification = Verifier().verify(task, tmp_path, result)

    assert verification.passed is False
    assert "json smoke stdout JSON path 'status' expected 'ready' got 'broken'" in verification.reasons
    assert "json smoke stdout JSON path 'metrics.pass_rate' expected >= 0.7 got 0.25" in verification.reasons


def test_verifier_applies_differential_checks(tmp_path):
    task = TaskSpec(
        task_id="differential_semantic_task",
        prompt="Run differential semantic checks.",
        workspace_subdir="differential_semantic_task",
        metadata={
            "semantic_verifier": {
                "kind": "behavioral_semantic",
                "differential_checks": [
                    {
                        "label": "stable output",
                        "candidate_argv": ["/bin/sh", "-lc", "printf 'match\\n'"],
                        "baseline_argv": ["/bin/sh", "-lc", "printf 'match\\n'"],
                        "expect_same_exit_code": True,
                        "expect_same_stdout": True,
                    }
                ],
            }
        },
    )
    result = CommandResult(command="true", exit_code=0, stdout="", stderr="")

    verification = Verifier().verify(task, tmp_path, result)

    assert verification.passed is True


def test_verifier_applies_richer_differential_checks(tmp_path):
    task = TaskSpec(
        task_id="differential_semantic_rich_task",
        prompt="Run richer differential semantic checks.",
        workspace_subdir="differential_semantic_rich_task",
        metadata={
            "semantic_verifier": {
                "kind": "behavioral_semantic",
                "differential_checks": [
                    {
                        "label": "candidate beats baseline",
                        "candidate_argv": ["/bin/sh", "-lc", "printf 'ready\\n'"],
                        "baseline_argv": ["/bin/sh", "-lc", "printf 'broken\\n'; exit 1"],
                        "expect_same_exit_code": False,
                        "expect_candidate_exit_code": 0,
                        "expect_baseline_exit_code": 1,
                        "expect_stdout_difference": True,
                        "candidate_stdout_must_contain": ["ready"],
                        "baseline_stdout_must_contain": ["broken"],
                    }
                ],
            }
        },
    )
    result = CommandResult(command="true", exit_code=0, stdout="", stderr="")

    verification = Verifier().verify(task, tmp_path, result)

    assert verification.passed is True


def test_verifier_applies_differential_json_output_checks(tmp_path):
    task = TaskSpec(
        task_id="differential_semantic_json_output_task",
        prompt="Run richer differential semantic checks on structured command output.",
        workspace_subdir="differential_semantic_json_output_task",
        metadata={
            "semantic_verifier": {
                "kind": "behavioral_semantic",
                "differential_checks": [
                    {
                        "label": "candidate structured status beats baseline",
                        "candidate_argv": ["/bin/sh", "-lc", "printf '{\"status\":\"ready\",\"score\":0.93}\\n'"],
                        "baseline_argv": ["/bin/sh", "-lc", "printf '{\"status\":\"broken\",\"score\":0.2}\\n'"],
                        "expect_same_exit_code": True,
                        "candidate_stdout_json_fields": [
                            {"path": "status", "equals": "ready"},
                            {"path": "score", "min": 0.9},
                        ],
                        "baseline_stdout_json_fields": [
                            {"path": "status", "equals": "broken"},
                            {"path": "score", "max": 0.3},
                        ],
                    }
                ],
            }
        },
    )
    result = CommandResult(command="true", exit_code=0, stdout="", stderr="")

    verification = Verifier().verify(task, tmp_path, result)

    assert verification.passed is True


def test_verifier_applies_differential_file_expectations_in_isolated_workspaces(tmp_path):
    (tmp_path / "reports").mkdir()
    task = TaskSpec(
        task_id="differential_semantic_file_task",
        prompt="Run differential semantic checks with file assertions.",
        workspace_subdir="differential_semantic_file_task",
        metadata={
            "semantic_verifier": {
                "kind": "behavioral_semantic",
                "differential_checks": [
                    {
                        "label": "candidate writes ready report",
                        "candidate_argv": ["/bin/sh", "-lc", "printf 'READY\\n' > reports/release_review.txt"],
                        "baseline_argv": ["/bin/sh", "-lc", "printf 'BROKEN\\n' > reports/release_review.txt"],
                        "expect_same_exit_code": True,
                        "candidate_file_expectations": [
                            {
                                "path": "reports/release_review.txt",
                                "expected_content": "READY\n",
                            }
                        ],
                        "baseline_file_expectations": [
                            {
                                "path": "reports/release_review.txt",
                                "must_contain": ["BROKEN"],
                            }
                        ],
                    }
                ],
            }
        },
    )
    result = CommandResult(command="true", exit_code=0, stdout="", stderr="")

    verification = Verifier().verify(task, tmp_path, result)

    assert verification.passed is True


def test_verifier_applies_differential_json_file_expectations(tmp_path):
    (tmp_path / "reports").mkdir()
    task = TaskSpec(
        task_id="differential_semantic_json_file_task",
        prompt="Run differential semantic checks with JSON file assertions.",
        workspace_subdir="differential_semantic_json_file_task",
        metadata={
            "semantic_verifier": {
                "kind": "behavioral_semantic",
                "differential_checks": [
                    {
                        "label": "candidate writes structured status",
                        "candidate_argv": ["/bin/sh", "-lc", "printf '{\"status\":\"ready\",\"score\":0.91}\\n' > reports/status.json"],
                        "baseline_argv": ["/bin/sh", "-lc", "printf '{\"status\":\"broken\",\"score\":0.12}\\n' > reports/status.json"],
                        "candidate_file_expectations": [
                            {
                                "path": "reports/status.json",
                                "json_fields": [
                                    {"path": "status", "equals": "ready"},
                                    {"path": "score", "min": 0.9},
                                ],
                            }
                        ],
                        "baseline_file_expectations": [
                            {
                                "path": "reports/status.json",
                                "json_fields": [
                                    {"path": "status", "equals": "broken"},
                                    {"path": "score", "max": 0.2},
                                ],
                            }
                        ],
                    }
                ],
            }
        },
    )
    result = CommandResult(command="true", exit_code=0, stdout="", stderr="")

    verification = Verifier().verify(task, tmp_path, result)

    assert verification.passed is True


def test_verifier_applies_repo_invariants(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "reports").mkdir()
    (tmp_path / "docs" / "context.md").write_text("context stable\n", encoding="utf-8")
    (tmp_path / "reports" / "status.txt").write_text("READY\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.email", "agent@example.com"], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.name", "Agent Kernel"], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "add", "docs/context.md", "reports/status.txt"], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "commit", "-m", "baseline"], cwd=tmp_path, check=True, capture_output=True, text=True)
    (tmp_path / "reports" / "status.txt").write_text("READY\n", encoding="utf-8")
    task = TaskSpec(
        task_id="repo_invariant_task",
        prompt="Validate repo invariants.",
        workspace_subdir="repo_invariant_task",
        metadata={
            "semantic_verifier": {
                "kind": "behavioral_semantic",
                "repo_invariants": [
                    {
                        "kind": "file_unchanged",
                        "path": "docs/context.md",
                        "expected_content": "context stable\n",
                    },
                    {
                        "kind": "file_contains",
                        "path": "reports/status.txt",
                        "must_contain": ["READY"],
                        "must_not_contain": ["BROKEN"],
                    },
                    {
                        "kind": "git_clean",
                        "allow_paths": ["reports/status.txt"],
                    },
                    {
                        "kind": "git_tracked_paths",
                        "paths": ["docs/context.md", "reports/status.txt"],
                    },
                    {"kind": "git_no_unmerged"},
                ],
            }
        },
    )
    result = CommandResult(command="true", exit_code=0, stdout="", stderr="")

    verification = Verifier().verify(task, tmp_path, result)

    assert verification.passed is True


def test_verifier_applies_json_repo_invariant(tmp_path):
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "status.json").write_text(
        "{\"status\":\"ready\",\"checks\":{\"passed\":3},\"families\":[\"integration\",\"repo\"]}\n",
        encoding="utf-8",
    )
    task = TaskSpec(
        task_id="repo_json_invariant_task",
        prompt="Validate JSON repo invariant.",
        workspace_subdir="repo_json_invariant_task",
        metadata={
            "semantic_verifier": {
                "kind": "behavioral_semantic",
                "repo_invariants": [
                    {
                        "kind": "file_contains",
                        "path": "reports/status.json",
                        "json_fields": [
                            {"path": "status", "equals": "ready"},
                            {"path": "checks.passed", "min": 3},
                            {"path": "families", "contains": "integration"},
                        ],
                    }
                ],
            }
        },
    )
    result = CommandResult(command="true", exit_code=0, stdout="", stderr="")

    verification = Verifier().verify(task, tmp_path, result)

    assert verification.passed is True


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
