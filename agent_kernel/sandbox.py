from __future__ import annotations

from dataclasses import dataclass
import difflib
import os
from pathlib import Path
import re
import shutil
import shlex
import subprocess
from urllib import error as url_error
from urllib import parse as url_parse
from urllib import request as url_request

from .extensions.capabilities import effective_http_allowed_hosts, segment_capabilities
from .config import KernelConfig
from .extensions.operator_policy import operator_policy_snapshot
from .schemas import CommandResult
from .schemas import TaskSpec


@dataclass(slots=True)
class _ParsedSegment:
    argv: list[str]
    stdout_path: str = ""
    append_stdout: bool = False


class Sandbox:
    _CONTAINMENT_PROBE_CACHE: dict[tuple[str, str], tuple[bool, str]] = {}
    ALLOWED_HOST_EXECUTABLES = frozenset(
        {
            "cat",
            "chmod",
            "cp",
            "echo",
            "false",
            "find",
            "git",
            "grep",
            "head",
            "http_request",
            "ls",
            "mkdir",
            "mv",
            "printf",
            "pwd",
            "rm",
            "sed",
            "sleep",
            "sort",
            "swe_patch_builder",
            "tail",
            "test",
            "touch",
            "true",
            "wc",
        }
    )
    BLOCKED_COMMAND_PATTERNS = (
        ("privileged", r"(^|\s)sudo(\s|$)"),
        ("privileged", r"(^|\s)su(\s|$)"),
        ("privileged", r"(^|\s)doas(\s|$)"),
        ("interactive", r"(^|\s)(vim|vi|nano|less|more|man|top|htop|watch)(\s|$)"),
        ("interactive", r"(^|\s)tail\s+-f(\s|$)"),
        ("interpreter", r"(^|\s)(python|python3|node|perl|ruby)(\s|$)"),
        ("shell", r"(^|\s)(bash|sh|zsh)(\s|$)"),
        ("network", r"(^|\s)(curl|wget|nc|netcat|telnet|ftp|sftp|scp|ssh|rsync)(\s|$)"),
    )
    _TEST_PATH_FLAGS = {"-d", "-e", "-f", "-r", "-s", "-w", "-x"}

    def __init__(
        self,
        timeout_seconds: int,
        *,
        config: KernelConfig | None = None,
        urlopen: object | None = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.config = config
        self.urlopen = urlopen or url_request.urlopen

    def run(self, command: str, cwd: Path, *, task: TaskSpec | None = None) -> CommandResult:
        try:
            segments = self._parse_segments(command)
        except ValueError as exc:
            return CommandResult(
                command=command,
                exit_code=126,
                stdout="",
                stderr=str(exc),
            )
        if not self._is_literal_cat_heredoc(command):
            blocked_reason = self._blocked_reason(command)
            if blocked_reason is not None:
                return CommandResult(
                    command=command,
                    exit_code=126,
                    stdout="",
                    stderr=blocked_reason,
                )

        cwd_resolved = cwd.resolve()
        cwd_resolved.mkdir(parents=True, exist_ok=True)
        env = self._bounded_env(cwd_resolved)
        combined_stdout = ""
        combined_stderr = ""
        last_exit_code = 0
        capabilities_used: set[str] = set()

        for segment in segments:
            path_error = self._path_policy_violation(segment, cwd_resolved, task=task)
            if path_error is not None:
                return CommandResult(
                    command=command,
                    exit_code=126,
                    stdout=combined_stdout,
                    stderr=f"{combined_stderr}{path_error}",
                )
            try:
                segment_result = self._run_segment(segment, cwd_resolved, env=env, task=task)
            except subprocess.TimeoutExpired as exc:
                return CommandResult(
                    command=command,
                    exit_code=124,
                    stdout=combined_stdout + self._coerce_text(exc.stdout),
                    stderr=combined_stderr + self._coerce_text(exc.stderr),
                    timed_out=True,
                )
            combined_stdout += segment_result.stdout
            combined_stderr += segment_result.stderr
            last_exit_code = segment_result.exit_code
            capabilities_used.update(segment_result.capabilities_used)
            if segment_result.exit_code != 0:
                return CommandResult(
                    command=command,
                    exit_code=last_exit_code,
                    stdout=combined_stdout,
                    stderr=combined_stderr,
                    capabilities_used=sorted(capabilities_used),
                )

        return CommandResult(
            command=command,
            exit_code=last_exit_code,
            stdout=combined_stdout,
            stderr=combined_stderr,
            capabilities_used=sorted(capabilities_used),
        )

    def containment_status(self, cwd: Path) -> dict[str, object]:
        cwd_resolved = cwd.resolve()
        cwd_resolved.mkdir(parents=True, exist_ok=True)
        return self._containment_status(cwd_resolved)

    def _run_segment(
        self,
        segment: _ParsedSegment,
        cwd: Path,
        *,
        env: dict[str, str],
        task: TaskSpec | None = None,
    ) -> CommandResult:
        command_name = Path(segment.argv[0]).name
        if command_name == "http_request":
            result = self._run_http_request_segment(segment, cwd)
            stdout = result.stdout
            stderr = result.stderr
            exit_code = result.exit_code
            timed_out = result.timed_out
        elif command_name == "swe_patch_builder":
            result = self._run_swe_patch_builder_segment(segment, cwd, task=task)
            stdout = result.stdout
            stderr = result.stderr
            exit_code = result.exit_code
            timed_out = result.timed_out
        else:
            containment = self._containment_status(cwd)
            argv = list(segment.argv)
            if bool(containment.get("active", False)):
                argv = self._contained_segment_argv(segment.argv, cwd, env=env, tool_path=str(containment["tool_path"]))
            completed = subprocess.run(
                argv,
                cwd=str(cwd),
                shell=False,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                env=env,
            )
            stdout = completed.stdout
            stderr = completed.stderr
            exit_code = completed.returncode
            timed_out = False
        if segment.stdout_path:
            target = self._resolve_workspace_path(segment.stdout_path, cwd)
            target.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if segment.append_stdout else "w"
            with target.open(mode, encoding="utf-8") as handle:
                handle.write(stdout)
            stdout = ""
        return CommandResult(
            command=" ".join(segment.argv),
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            capabilities_used=segment_capabilities(segment.argv, stdout_path=segment.stdout_path),
        )

    def _parse_segments(self, command: str) -> list[_ParsedSegment]:
        normalized = command.strip()
        if not normalized:
            raise ValueError("blocked empty command")
        heredoc_segment = self._parse_simple_cat_heredoc(normalized)
        if heredoc_segment is not None:
            return [heredoc_segment]
        if any(marker in normalized for marker in ("`", "$(", "${")):
            raise ValueError(f"blocked unsupported shell expansion: {normalized}")

        lexer = shlex.shlex(normalized, posix=True, punctuation_chars="&|;<>")
        lexer.whitespace_split = True
        tokens = list(lexer)
        if not tokens:
            raise ValueError("blocked empty command")

        segments: list[_ParsedSegment] = []
        argv: list[str] = []
        stdout_path = ""
        append_stdout = False
        index = 0
        while index < len(tokens):
            token = tokens[index]
            if token == "&&":
                segments.append(self._build_segment(argv, stdout_path=stdout_path, append_stdout=append_stdout))
                argv = []
                stdout_path = ""
                append_stdout = False
                index += 1
                continue
            if token in {"|", "||", ";", "&", "<", "<<"}:
                raise ValueError(f"blocked unsupported shell operator: {token}")
            if token in {">", ">>"}:
                if stdout_path:
                    raise ValueError(f"blocked multiple stdout redirects: {normalized}")
                index += 1
                if index >= len(tokens):
                    raise ValueError(f"blocked redirect without target: {normalized}")
                stdout_path = tokens[index]
                append_stdout = token == ">>"
                index += 1
                continue
            if stdout_path:
                raise ValueError(f"blocked trailing tokens after redirect target: {normalized}")
            argv.append(token)
            index += 1
        segments.append(self._build_segment(argv, stdout_path=stdout_path, append_stdout=append_stdout))
        return segments

    def _is_literal_cat_heredoc(self, command: str) -> bool:
        return self._parse_simple_cat_heredoc(command.strip()) is not None

    def _parse_simple_cat_heredoc(self, normalized: str) -> _ParsedSegment | None:
        mkdir_match = re.fullmatch(
            r"mkdir\s+-p\s+(?P<mkdir_path>\S+)\s+&&\s+cat\s+>\s+(?P<path>\S+)\s+"
            r"<<\s*['\"]?(?P<marker>[A-Za-z_][A-Za-z0-9_]*)['\"]?\n"
            r"(?P<body>.*)\n(?P=marker)",
            normalized,
            flags=re.DOTALL,
        )
        if mkdir_match is not None:
            body = mkdir_match.group("body")
            lines = body.splitlines()
            if not lines:
                lines = [""]
            return self._build_segment(
                ["printf", "%s\n", *lines],
                stdout_path=mkdir_match.group("path"),
                append_stdout=False,
            )
        match = re.fullmatch(
            r"cat\s+>\s+(?P<path>\S+)\s+<<\s*['\"]?(?P<marker>[A-Za-z_][A-Za-z0-9_]*)['\"]?\n"
            r"(?P<body>.*)\n(?P=marker)",
            normalized,
            flags=re.DOTALL,
        )
        if match is None:
            return None
        body = match.group("body")
        lines = body.splitlines()
        if not lines:
            lines = [""]
        return self._build_segment(
            ["printf", "%s\n", *lines],
            stdout_path=match.group("path"),
            append_stdout=False,
        )

    def _build_segment(self, argv: list[str], *, stdout_path: str, append_stdout: bool) -> _ParsedSegment:
        if not argv:
            raise ValueError("blocked empty command segment")
        command_name = argv[0].strip()
        if not command_name:
            raise ValueError("blocked empty executable")
        if Path(command_name).is_absolute():
            raise ValueError(f"blocked absolute executable path: {command_name}")
        return _ParsedSegment(
            argv=list(argv),
            stdout_path=stdout_path.strip(),
            append_stdout=append_stdout,
        )

    def _path_policy_violation(
        self,
        segment: _ParsedSegment,
        cwd: Path,
        *,
        task: TaskSpec | None,
    ) -> str | None:
        try:
            executable_error = self._executable_policy_violation(segment, cwd)
            if executable_error is not None:
                return executable_error
            if "/" in segment.argv[0]:
                self._resolve_workspace_path(segment.argv[0], cwd)
            for token in self._path_operands(segment):
                self._resolve_workspace_path(token, cwd)
            if segment.stdout_path:
                self._resolve_workspace_path(segment.stdout_path, cwd)
            workflow_error = self._workflow_policy_violation(segment, cwd, task=task)
            if workflow_error is not None:
                return workflow_error
        except ValueError as exc:
            return str(exc)
        return None

    def _executable_policy_violation(self, segment: _ParsedSegment, cwd: Path) -> str | None:
        raw_executable = str(segment.argv[0]).strip()
        command_name = Path(raw_executable).name
        if "/" not in raw_executable:
            if command_name in self.ALLOWED_HOST_EXECUTABLES:
                return None
            return f"blocked unsupported executable: {command_name}"
        resolved = self._resolve_workspace_path(raw_executable, cwd)
        if not resolved.exists():
            return f"blocked missing workspace executable: {raw_executable}"
        if not resolved.is_file():
            return f"blocked non-file workspace executable: {raw_executable}"
        if not os.access(resolved, os.X_OK):
            return f"blocked non-executable workspace file: {raw_executable}"
        return None

    def _path_operands(self, segment: _ParsedSegment) -> list[str]:
        command_name = Path(segment.argv[0]).name
        args = list(segment.argv[1:])
        if command_name in {"cat", "mkdir", "rm", "mv", "cp", "touch"}:
            return [arg for arg in args if not arg.startswith("-")]
        if command_name == "sed":
            return self._sed_path_operands(args)
        if command_name == "grep":
            operands = self._grep_path_operands(args)
            return operands
        if command_name == "test":
            return self._test_path_operands(args)
        if command_name == "http_request":
            return self._http_request_path_operands(args)
        if command_name == "swe_patch_builder":
            return self._swe_patch_builder_path_operands(args)
        return [arg for arg in args if self._looks_like_path(arg)]

    def _run_swe_patch_builder_segment(
        self,
        segment: _ParsedSegment,
        cwd: Path,
        *,
        task: TaskSpec | None,
    ) -> CommandResult:
        try:
            patch = self._build_swe_patch(segment.argv[1:], cwd, task=task)
        except ValueError as exc:
            return CommandResult(
                command=" ".join(segment.argv),
                exit_code=2,
                stdout="",
                stderr=f"{exc}\n",
            )
        return CommandResult(
            command=" ".join(segment.argv),
            exit_code=0,
            stdout=patch,
            stderr="",
        )

    def _build_swe_patch(self, args: list[str], cwd: Path, *, task: TaskSpec | None) -> str:
        from_diff = ""
        path = ""
        operations: list[tuple[int, int, list[str]]] = []
        current_start_line = 0
        current_end_line = 0
        current_replacement_lines: list[str] = []

        def flush_operation() -> None:
            nonlocal current_start_line, current_end_line, current_replacement_lines
            if current_start_line:
                operations.append((current_start_line, current_end_line, list(current_replacement_lines)))
                current_start_line = 0
                current_end_line = 0
                current_replacement_lines = []

        index = 0
        while index < len(args):
            token = args[index]
            if token == "--from-diff" and index + 1 < len(args):
                flush_operation()
                from_diff = args[index + 1].strip()
                index += 2
                continue
            if token == "--path" and index + 1 < len(args):
                path = args[index + 1].strip()
                index += 2
                continue
            if token == "--replace-line" and index + 1 < len(args):
                flush_operation()
                current_start_line = int(args[index + 1])
                current_end_line = current_start_line
                index += 2
                continue
            if token == "--replace-lines" and index + 2 < len(args):
                flush_operation()
                current_start_line = int(args[index + 1])
                current_end_line = int(args[index + 2])
                index += 3
                continue
            if token == "--with" and index + 1 < len(args):
                if not current_start_line:
                    raise ValueError("swe_patch_builder --with requires a preceding replacement range")
                current_replacement_lines.append(args[index + 1])
                index += 2
                continue
            raise ValueError(f"unsupported swe_patch_builder argument: {token}")
        flush_operation()
        if from_diff:
            if path or operations:
                raise ValueError("swe_patch_builder --from-diff cannot be combined with line replacement arguments")
            diff_path = self._resolve_workspace_path(from_diff, cwd)
            if not diff_path.is_file():
                raise ValueError(f"swe_patch_builder input diff missing: {from_diff}")
            return self._build_swe_patch_from_diff(diff_path.read_text(encoding="utf-8"), cwd, task=task)
        if not path:
            raise ValueError("swe_patch_builder requires --path")
        if not operations:
            raise ValueError("swe_patch_builder requires --replace-line or --replace-lines")
        normalized_path = path.strip().strip("/")
        if not normalized_path or normalized_path.startswith("../") or "/../" in normalized_path:
            raise ValueError(f"swe_patch_builder received unsafe path: {path}")
        if task is not None:
            allowed_paths = self._swe_candidate_paths(task)
            if allowed_paths and normalized_path not in allowed_paths:
                raise ValueError(f"swe_patch_builder path is not in task candidate files: {normalized_path}")
        source_path = self._resolve_workspace_path(normalized_path, cwd)
        if not source_path.is_file():
            raise ValueError(f"swe_patch_builder source file missing: {normalized_path}")
        original_lines = source_path.read_text(encoding="utf-8").splitlines(keepends=True)
        normalized_operations = sorted(operations, key=lambda item: item[0])
        previous_end_line = 0
        for start_line, end_line, replacement_lines in normalized_operations:
            if start_line < 1 or end_line < start_line:
                raise ValueError("swe_patch_builder received invalid replacement line range")
            if start_line <= previous_end_line:
                raise ValueError("swe_patch_builder received overlapping replacement ranges")
            if not replacement_lines:
                raise ValueError("swe_patch_builder requires at least one --with replacement line")
            if end_line > len(original_lines):
                raise ValueError(
                    f"swe_patch_builder replacement range {start_line}-{end_line} exceeds {normalized_path} line count {len(original_lines)}"
                )
            previous_end_line = end_line
        updated_lines = list(original_lines)
        for start_line, end_line, replacement_lines in reversed(normalized_operations):
            formatted_replacements = self._format_swe_replacement_lines(
                original_lines,
                start_line=start_line,
                replacement_lines=replacement_lines,
            )
            updated_lines[start_line - 1 : end_line] = formatted_replacements
        if updated_lines == original_lines:
            raise ValueError("swe_patch_builder replacement produced no meaningful change")
        return "".join(
            difflib.unified_diff(
                original_lines,
                updated_lines,
                fromfile=f"a/{normalized_path}",
                tofile=f"b/{normalized_path}",
            )
        )

    def _build_swe_patch_from_diff(self, diff_text: str, cwd: Path, *, task: TaskSpec | None) -> str:
        path, replacements = self._extract_swe_diff_replacements(diff_text)
        if not path:
            raise ValueError("swe_patch_builder could not find a changed path in input diff")
        normalized_path = path.strip().strip("/")
        if task is not None:
            allowed_paths = self._swe_candidate_paths(task)
            if allowed_paths and normalized_path not in allowed_paths:
                raise ValueError(f"swe_patch_builder path is not in task candidate files: {normalized_path}")
        source_path = self._resolve_workspace_path(normalized_path, cwd)
        if not source_path.is_file():
            raise ValueError(f"swe_patch_builder source file missing: {normalized_path}")
        original_lines = source_path.read_text(encoding="utf-8").splitlines(keepends=True)
        updated_lines = list(original_lines)
        for start_line, end_line, replacement_lines in sorted(replacements, reverse=True):
            if start_line < 1 or end_line < start_line or end_line > len(updated_lines):
                raise ValueError(f"swe_patch_builder input diff contains invalid range {start_line}-{end_line}")
            formatted_replacements = self._format_swe_replacement_lines(
                updated_lines,
                start_line=start_line,
                replacement_lines=replacement_lines,
            )
            updated_lines[start_line - 1 : end_line] = formatted_replacements
        if updated_lines == original_lines:
            raise ValueError("swe_patch_builder input diff produced no meaningful change")
        return "".join(
            difflib.unified_diff(
                original_lines,
                updated_lines,
                fromfile=f"a/{normalized_path}",
                tofile=f"b/{normalized_path}",
            )
        )

    @staticmethod
    def _format_swe_replacement_lines(
        source_lines: list[str],
        *,
        start_line: int,
        replacement_lines: list[str],
    ) -> list[str]:
        formatted: list[str] = []
        for offset, replacement in enumerate(replacement_lines):
            line = replacement if replacement.endswith("\n") else f"{replacement}\n"
            original_index = start_line - 1 + offset
            if (
                original_index < len(source_lines)
                and line.strip()
                and not line[:1].isspace()
            ):
                original_indent = re.match(r"^\s*", source_lines[original_index]).group(0)
                if original_indent:
                    line = f"{original_indent}{line}"
            formatted.append(line)
        return formatted

    @staticmethod
    def _extract_swe_diff_replacements(diff_text: str) -> tuple[str, list[tuple[int, int, list[str]]]]:
        lines = diff_text.splitlines()
        path = ""
        for line in lines:
            if line.startswith("+++ b/"):
                path = line.removeprefix("+++ b/").strip()
                break
        if not path:
            for line in lines:
                if line.startswith("diff --git "):
                    parts = line.split()
                    if len(parts) >= 4 and parts[3].startswith("b/"):
                        path = parts[3].removeprefix("b/").strip()
                        break
        replacements: list[tuple[int, int, list[str]]] = []
        hunk_header = re.compile(r"^@@\s+-(?P<start>\d+)(?:,(?P<count>\d+))?\s+\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))?\s+@@")
        index = 0
        while index < len(lines):
            match = hunk_header.match(lines[index])
            if match is None:
                index += 1
                continue
            old_cursor = int(match.group("start"))
            change_start = 0
            change_end = 0
            replacement: list[str] = []
            index += 1
            while index < len(lines) and not lines[index].startswith("@@ ") and not lines[index].startswith("diff --git "):
                line = lines[index]
                if not line:
                    index += 1
                    continue
                prefix = line[0]
                payload = line[1:]
                if prefix == " ":
                    if change_start:
                        break
                    old_cursor += 1
                elif prefix == "-":
                    if not change_start:
                        change_start = old_cursor
                    old_cursor += 1
                    change_end = old_cursor - 1
                elif prefix == "+":
                    if not change_start:
                        change_start = old_cursor
                        change_end = old_cursor
                    replacement.append(payload)
                elif line.startswith("\\ No newline"):
                    pass
                else:
                    break
                index += 1
            if change_start and change_end >= change_start and replacement:
                replacements.append((change_start, change_end, replacement))
            continue
        if not replacements:
            raise ValueError("swe_patch_builder could not extract a replacement from input diff")
        return path, replacements

    @staticmethod
    def _swe_candidate_paths(task: TaskSpec) -> set[str]:
        candidates = task.metadata.get("swe_candidate_files", [])
        verifier = task.metadata.get("semantic_verifier", {})
        if not candidates and isinstance(verifier, dict):
            candidates = verifier.get("expected_changed_paths", [])
        if not isinstance(candidates, list):
            return set()
        return {str(path).strip().strip("/") for path in candidates if str(path).strip()}

    def _workflow_policy_violation(
        self,
        segment: _ParsedSegment,
        cwd: Path,
        *,
        task: TaskSpec | None,
    ) -> str | None:
        destructive_error = self._destructive_mutation_violation(segment, cwd, task=task)
        if destructive_error is not None:
            return destructive_error
        if task is None or self.config is None:
            return None
        operator_policy = self._operator_policy_snapshot()
        command_name = Path(segment.argv[0]).name
        if command_name == "git" and not bool(operator_policy.get("unattended_allow_git_commands", False)):
            return "blocked unattended git command by operator policy"
        if command_name == "http_request" and not bool(
            operator_policy.get("unattended_allow_http_requests", False)
        ):
            return "blocked unattended http request by operator policy"

        write_targets, delete_targets = self._mutation_targets(segment)
        workflow_guard = self._workflow_guard(task)
        claimed_paths = self._normalize_paths(workflow_guard.get("claimed_paths", []))
        for raw_path in write_targets:
            relative_path = self._relative_workspace_path(raw_path, cwd)
            if self._is_generated_path(relative_path) and not bool(
                operator_policy.get("unattended_allow_generated_path_mutations", False)
            ):
                return f"blocked generated-path mutation by operator policy: {relative_path}"
            if claimed_paths and not self._path_within_claim(relative_path, claimed_paths):
                return f"blocked out-of-claim write path for shared repo worker: {relative_path}"
            if not self._write_target_allowed(relative_path, task, workflow_guard):
                return f"blocked unmanaged write path for task contract: {relative_path}"
        for raw_path in delete_targets:
            relative_path = self._relative_workspace_path(raw_path, cwd)
            if self._is_generated_path(relative_path) and not bool(
                operator_policy.get("unattended_allow_generated_path_mutations", False)
            ):
                return f"blocked generated-path mutation by operator policy: {relative_path}"
            if claimed_paths and not self._path_within_claim(relative_path, claimed_paths):
                return f"blocked out-of-claim delete path for shared repo worker: {relative_path}"
            if not self._delete_target_allowed(relative_path, task, workflow_guard):
                return f"blocked unmanaged delete path for task contract: {relative_path}"
        return None

    def _mutation_targets(self, segment: _ParsedSegment) -> tuple[list[str], list[str]]:
        command_name = Path(segment.argv[0]).name
        raw_args = list(segment.argv[1:])
        args = [arg for arg in raw_args if not arg.startswith("-")]
        write_targets: list[str] = [segment.stdout_path] if segment.stdout_path else []
        delete_targets: list[str] = []
        if command_name in {"mkdir", "touch"}:
            write_targets.extend(args)
        elif command_name == "rm":
            delete_targets.extend(args)
        elif command_name in {"mv", "cp"} and len(args) == 2:
            delete_targets.append(args[0] if command_name == "mv" else "")
            write_targets.append(args[1])
        elif command_name == "sed" and self._sed_in_place(raw_args):
            write_targets.extend(self._sed_path_operands(raw_args))
        return [path for path in write_targets if path], [path for path in delete_targets if path]

    def _destructive_mutation_violation(
        self,
        segment: _ParsedSegment,
        cwd: Path,
        *,
        task: TaskSpec | None,
    ) -> str | None:
        _, delete_targets = self._mutation_targets(segment)
        if not delete_targets:
            return None
        for raw_path in delete_targets:
            relative_path = self._relative_workspace_path(raw_path, cwd).strip()
            if relative_path in {"", "."}:
                return "blocked destructive delete of workspace root"
            if relative_path == ".." or relative_path.startswith("../"):
                return f"blocked destructive delete path: {relative_path}"
        if task is None:
            return "blocked destructive mutation without task contract"
        return None

    @staticmethod
    def _http_request_path_operands(args: list[str]) -> list[str]:
        operands: list[str] = []
        index = 0
        while index < len(args):
            token = args[index]
            if token == "--body-file" and index + 1 < len(args):
                operands.append(args[index + 1])
                index += 2
                continue
            if token == "--output" and index + 1 < len(args):
                operands.append(args[index + 1])
                index += 2
                continue
            index += 1
        return operands

    @staticmethod
    def _swe_patch_builder_path_operands(args: list[str]) -> list[str]:
        operands: list[str] = []
        index = 0
        while index < len(args):
            token = args[index]
            if token == "--from-diff" and index + 1 < len(args):
                operands.append(args[index + 1])
                index += 2
                continue
            if token == "--path" and index + 1 < len(args):
                operands.append(args[index + 1])
                index += 2
                continue
            index += 1
        return operands

    @staticmethod
    def _workflow_guard(task: TaskSpec) -> dict[str, object]:
        guard = task.metadata.get("workflow_guard", {})
        return dict(guard) if isinstance(guard, dict) else {}

    def _write_target_allowed(
        self,
        relative_path: str,
        task: TaskSpec,
        workflow_guard: dict[str, object],
    ) -> bool:
        managed_paths = self._managed_paths(task, workflow_guard)
        if relative_path in managed_paths:
            return True
        return any(path.startswith(f"{relative_path}/") for path in managed_paths)

    def _delete_target_allowed(
        self,
        relative_path: str,
        task: TaskSpec,
        workflow_guard: dict[str, object],
    ) -> bool:
        delete_paths = {
            *self._normalize_paths(task.forbidden_files),
            *self._normalize_paths(workflow_guard.get("delete_paths", [])),
        }
        return relative_path in delete_paths

    def _managed_paths(self, task: TaskSpec, workflow_guard: dict[str, object]) -> set[str]:
        return {
            *self._normalize_paths(task.expected_files),
            *self._normalize_paths(task.expected_file_contents),
            *self._normalize_paths(task.forbidden_files),
            *self._normalize_paths(workflow_guard.get("managed_paths", [])),
            *self._normalize_paths(workflow_guard.get("delete_paths", [])),
        }

    @staticmethod
    def _path_within_claim(relative_path: str, claimed_paths: set[str]) -> bool:
        return any(
            relative_path == claimed_path
            or relative_path.startswith(f"{claimed_path}/")
            or claimed_path.startswith(f"{relative_path}/")
            for claimed_path in claimed_paths
        )

    @staticmethod
    def _normalize_paths(values: object) -> set[str]:
        if isinstance(values, dict):
            iterable = values.keys()
        elif isinstance(values, (list, tuple, set)):
            iterable = values
        else:
            iterable = []
        return {str(value).strip().strip("/") for value in iterable if str(value).strip()}

    def _is_generated_path(self, relative_path: str) -> bool:
        normalized = relative_path.strip().strip("/")
        if not normalized:
            return False
        operator_policy = self._operator_policy_snapshot()
        for prefix in operator_policy.get("unattended_generated_path_prefixes", []):
            candidate = str(prefix).strip().strip("/")
            if not candidate:
                continue
            if normalized == candidate or normalized.startswith(f"{candidate}/"):
                return True
        return False

    def _operator_policy_snapshot(self) -> dict[str, object]:
        if self.config is None:
            return {}
        return operator_policy_snapshot(self.config)

    def _relative_workspace_path(self, token: str, cwd: Path) -> str:
        resolved = self._resolve_workspace_path(token, cwd)
        return str(resolved.relative_to(cwd))

    def _grep_path_operands(self, args: list[str]) -> list[str]:
        if not args:
            return []
        index = 0
        while index < len(args) and args[index].startswith("-"):
            index += 1
            if index < len(args) and args[index - 1] in {"-e", "-f", "-m"}:
                index += 1
        if index >= len(args):
            return []
        index += 1
        return [arg for arg in args[index:] if not arg.startswith("-")]

    def _sed_path_operands(self, args: list[str]) -> list[str]:
        if not args:
            return []
        index = 0
        while index < len(args):
            token = args[index]
            if token == "-i":
                index += 1
                if index < len(args) and not args[index].startswith("-") and not self._looks_like_sed_script(args[index]):
                    index += 1
                continue
            if token.startswith("-i") or token in {"-E", "-n"}:
                index += 1
                continue
            if token in {"-e", "-f"}:
                index += 2
                continue
            break
        if index < len(args) and self._looks_like_sed_script(args[index]):
            index += 1
        return [arg for arg in args[index:] if not arg.startswith("-")]

    @staticmethod
    def _sed_in_place(args: list[str]) -> bool:
        return any(token == "-i" or token.startswith("-i") for token in args)

    def _test_path_operands(self, args: list[str]) -> list[str]:
        operands: list[str] = []
        index = 0
        while index < len(args):
            token = args[index]
            if token == "!":
                index += 1
                continue
            if token in self._TEST_PATH_FLAGS and index + 1 < len(args):
                operands.append(args[index + 1])
                index += 2
                continue
            if self._looks_like_path(token):
                operands.append(token)
            index += 1
        return operands

    @staticmethod
    def _looks_like_path(token: str) -> bool:
        return "/" in token or token in {".", ".."} or token.startswith("./") or token.startswith("../")

    @staticmethod
    def _looks_like_sed_script(token: str) -> bool:
        return bool(token) and token[0] in {"s", "/", ":"}

    @staticmethod
    def _resolve_workspace_path(token: str, cwd: Path) -> Path:
        candidate = Path(token)
        resolved = candidate.resolve() if candidate.is_absolute() else (cwd / candidate).resolve()
        if resolved != cwd and cwd not in resolved.parents:
            raise ValueError(f"blocked workspace escape path: {token}")
        return resolved

    @staticmethod
    def _coerce_text(value: str | bytes | None) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return value

    @staticmethod
    def _bounded_env(cwd: Path) -> dict[str, str]:
        tmpdir = cwd / ".tmp"
        tmpdir.mkdir(parents=True, exist_ok=True)
        return {
            "PATH": os.getenv("PATH", ""),
            "HOME": str(cwd),
            "TMPDIR": str(tmpdir),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
        }

    def _containment_status(self, cwd: Path) -> dict[str, object]:
        mode = self._containment_mode()
        if mode == "disabled":
            return {
                "mode": mode,
                "available": False,
                "active": False,
                "tool": "",
                "tool_path": "",
                "detail": "OS-level command containment disabled by config",
            }
        tool_name = self._containment_tool_name()
        if not tool_name:
            return {
                "mode": mode,
                "available": False,
                "active": False,
                "tool": "",
                "tool_path": "",
                "detail": "OS-level command containment tool not configured",
            }
        if tool_name != "bwrap":
            return {
                "mode": mode,
                "available": False,
                "active": False,
                "tool": tool_name,
                "tool_path": "",
                "detail": f"unsupported containment tool configured: {tool_name}",
            }
        tool_path = shutil.which(tool_name)
        if not tool_path:
            return {
                "mode": mode,
                "available": False,
                "active": False,
                "tool": tool_name,
                "tool_path": "",
                "detail": f"OS-level command containment tool not found: {tool_name}",
            }
        available, detail = self._probe_bwrap(tool_path, cwd)
        return {
            "mode": mode,
            "available": available,
            "active": available,
            "tool": tool_name,
            "tool_path": tool_path,
            "detail": detail,
        }

    def _containment_mode(self) -> str:
        raw = str(getattr(self.config, "sandbox_command_containment_mode", "disabled") or "disabled").strip().lower()
        if raw in {"disabled", "off", "none"}:
            return "disabled"
        if raw in {"required", "require", "strict"}:
            return "required"
        return "auto"

    def _containment_tool_name(self) -> str:
        return str(getattr(self.config, "sandbox_command_containment_tool", "bwrap") or "bwrap").strip()

    def _probe_bwrap(self, tool_path: str, cwd: Path) -> tuple[bool, str]:
        cache_key = (tool_path, str(cwd))
        cached = self._CONTAINMENT_PROBE_CACHE.get(cache_key)
        if cached is not None:
            return cached
        probe_argv = self._contained_segment_argv(["true"], cwd, env=self._bounded_env(cwd), tool_path=tool_path)
        try:
            completed = subprocess.run(
                probe_argv,
                cwd=str(cwd),
                shell=False,
                text=True,
                capture_output=True,
                timeout=min(3, max(1, self.timeout_seconds)),
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            result = (False, f"bubblewrap probe failed: {exc}")
            self._CONTAINMENT_PROBE_CACHE[cache_key] = result
            return result
        if completed.returncode == 0:
            result = (True, f"OS-level command containment active via bubblewrap ({tool_path})")
            self._CONTAINMENT_PROBE_CACHE[cache_key] = result
            return result
        detail = completed.stderr.strip() or completed.stdout.strip() or "bubblewrap probe failed"
        result = (False, f"bubblewrap unavailable for this runtime: {detail}")
        self._CONTAINMENT_PROBE_CACHE[cache_key] = result
        return result

    def _contained_segment_argv(
        self,
        argv: list[str],
        cwd: Path,
        *,
        env: dict[str, str],
        tool_path: str,
    ) -> list[str]:
        command: list[str] = [
            tool_path,
            "--die-with-parent",
            "--new-session",
            "--unshare-all",
            "--share-net",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--ro-bind",
            "/",
            "/",
            "--bind",
            str(cwd),
            str(cwd),
            "--chdir",
            str(cwd),
            "--clearenv",
        ]
        for key in ("HOME", "LANG", "LC_ALL", "PATH", "TMPDIR"):
            value = str(env.get(key, "")).strip()
            if value:
                command.extend(["--setenv", key, value])
        command.append("--")
        command.extend(argv)
        return command

    def _blocked_reason(self, command: str) -> str | None:
        normalized = command.strip()
        for kind, pattern in self.BLOCKED_COMMAND_PATTERNS:
            if re.search(pattern, normalized):
                return f"blocked {kind} command pattern: {normalized}"
        return None

    def _run_http_request_segment(self, segment: _ParsedSegment, cwd: Path) -> CommandResult:
        try:
            request_spec = self._parse_http_request_args(segment.argv[1:], cwd)
        except ValueError as exc:
            return CommandResult(command=" ".join(segment.argv), exit_code=126, stdout="", stderr=str(exc))

        if self.config is not None:
            host_error = self._http_host_policy_violation(request_spec["url"])
            if host_error is not None:
                return CommandResult(command=" ".join(segment.argv), exit_code=126, stdout="", stderr=host_error)
            operator_policy = self._operator_policy_snapshot()
            timeout_seconds = max(1, int(operator_policy.get("unattended_http_timeout_seconds", 1)))
            max_body_bytes = max(1, int(operator_policy.get("unattended_http_max_body_bytes", 1)))
        else:
            timeout_seconds = max(1, self.timeout_seconds)
            max_body_bytes = 64 * 1024

        req = url_request.Request(
            url=str(request_spec["url"]),
            data=request_spec["body"],
            headers=request_spec["headers"],
            method=str(request_spec["method"]),
        )
        try:
            with self.urlopen(req, timeout=timeout_seconds) as response:
                raw = response.read(max_body_bytes + 1)
                if len(raw) > max_body_bytes:
                    return CommandResult(
                        command=" ".join(segment.argv),
                        exit_code=125,
                        stdout="",
                        stderr=f"http response exceeded max body bytes: {max_body_bytes}",
                    )
                body = raw.decode("utf-8", errors="replace")
                status = int(getattr(response, "status", 200))
                header_lines = [f"{key}: {value}" for key, value in response.headers.items()]
        except TimeoutError:
            return CommandResult(
                command=" ".join(segment.argv),
                exit_code=124,
                stdout="",
                stderr="http request timed out",
                timed_out=True,
            )
        except url_error.HTTPError as exc:
            error_body = exc.read(max_body_bytes).decode("utf-8", errors="replace")
            return CommandResult(
                command=" ".join(segment.argv),
                exit_code=1,
                stdout=error_body,
                stderr=f"http request failed with status {exc.code}",
            )
        except (url_error.URLError, OSError) as exc:
            return CommandResult(
                command=" ".join(segment.argv),
                exit_code=1,
                stdout="",
                stderr=f"http request failed: {exc}",
            )

        response_stdout = f"status: {status}\n"
        if header_lines:
            response_stdout += "\n".join(header_lines) + "\n\n"
        response_stdout += body
        return CommandResult(
            command=" ".join(segment.argv),
            exit_code=0,
            stdout=response_stdout,
            stderr="",
        )

    def _parse_http_request_args(self, args: list[str], cwd: Path) -> dict[str, object]:
        if len(args) < 2:
            raise ValueError("blocked malformed http_request: expected METHOD URL")
        method = str(args[0]).strip().upper()
        if method not in {"GET", "POST"}:
            raise ValueError(f"blocked unsupported http method: {method}")
        url = str(args[1]).strip()
        parsed = url_parse.urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError(f"blocked unsupported http scheme: {parsed.scheme or '<missing>'}")
        headers: dict[str, str] = {}
        body: bytes | None = None
        output_path = ""
        index = 2
        while index < len(args):
            token = str(args[index]).strip()
            if token == "--header":
                if index + 1 >= len(args):
                    raise ValueError("blocked malformed http_request: --header requires value")
                header = str(args[index + 1])
                if ":" not in header:
                    raise ValueError("blocked malformed http_request header: expected Name: Value")
                name, value = header.split(":", 1)
                headers[name.strip()] = value.strip()
                index += 2
                continue
            if token == "--body":
                if index + 1 >= len(args):
                    raise ValueError("blocked malformed http_request: --body requires value")
                if body is not None:
                    raise ValueError("blocked malformed http_request: multiple request bodies")
                body = str(args[index + 1]).encode("utf-8")
                index += 2
                continue
            if token == "--body-file":
                if index + 1 >= len(args):
                    raise ValueError("blocked malformed http_request: --body-file requires path")
                if body is not None:
                    raise ValueError("blocked malformed http_request: multiple request bodies")
                path = self._resolve_workspace_path(str(args[index + 1]), cwd)
                body = path.read_bytes()
                index += 2
                continue
            if token == "--output":
                if index + 1 >= len(args):
                    raise ValueError("blocked malformed http_request: --output requires path")
                output_path = str(args[index + 1]).strip()
                index += 2
                continue
            raise ValueError(f"blocked malformed http_request argument: {token}")
        if method == "GET" and body is not None:
            raise ValueError("blocked malformed http_request: GET does not accept request body")
        if output_path:
            raise ValueError("blocked http_request --output; use shell redirection into workspace instead")
        return {
            "method": method,
            "url": url,
            "headers": headers,
            "body": body,
        }

    def _http_host_policy_violation(self, url: str) -> str | None:
        if self.config is None:
            return None
        parsed = url_parse.urlparse(url)
        host = (parsed.hostname or "").strip().lower()
        if not host:
            return "blocked malformed http_request host"
        allowed_hosts = {
            str(value).strip().lower()
            for value in effective_http_allowed_hosts(self.config)
            if str(value).strip()
        }
        if not allowed_hosts:
            return None
        if host in allowed_hosts:
            return None
        if any(host.endswith(f".{candidate}") for candidate in allowed_hosts):
            return None
        return f"blocked http host by operator policy: {host}"


def sandbox_containment_status(config: KernelConfig, *, cwd: Path | None = None) -> dict[str, object]:
    sandbox = Sandbox(timeout_seconds=max(1, int(config.command_timeout_seconds)), config=config)
    probe_root = (cwd or config.workspace_root).resolve()
    probe_root.mkdir(parents=True, exist_ok=True)
    return sandbox.containment_status(probe_root)
