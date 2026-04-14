from __future__ import annotations

import re


def normalize_command_for_workspace(command: str, workspace_subdir: str) -> str:
    normalized = command.strip()
    workspace_name = workspace_subdir.strip().strip("/")
    if not workspace_name:
        return normalized

    mkdir_prefix = f"mkdir -p {workspace_name} && "
    if normalized.startswith(mkdir_prefix):
        normalized = normalized[len(mkdir_prefix):].strip()

    normalized = re.sub(rf"(?<![\w./-]){re.escape(workspace_name)}/", "", normalized)
    return normalized


def canonicalize_command(command: str) -> str:
    normalized = str(command).strip()
    if not normalized:
        return ""
    # Retrieval templates can encode newlines literally inside quoted strings,
    # while model outputs often use the shell-escaped `\\n` form. Normalize both
    # spellings so retrieval grounding and metrics attach to the same command.
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\n", "\\n").replace("\t", "\\t")
    return " ".join(normalized.split())
