from __future__ import annotations

import json
from pathlib import Path
import shlex
from typing import Any
from urllib import parse as url_parse

from ..config import KernelConfig
from .improvement.improvement_common import retained_artifact_payload
from ..extensions.strategy.kernel_catalog import kernel_catalog_mapping, kernel_catalog_string_set
from ..extensions.operator_policy import operator_policy_snapshot
from ..schemas import StepRecord, TaskSpec


_SHELL_OPERATORS = kernel_catalog_string_set("capabilities", "shell_operators")
_FILESYSTEM_COMMANDS = kernel_catalog_string_set("capabilities", "filesystem_commands")
_BUILTIN_CAPABILITY_NAMES = kernel_catalog_string_set("capabilities", "builtin_capability_names")
_TIER_ORDER = {
    str(key): int(value)
    for key, value in kernel_catalog_mapping("capabilities", "tier_order").items()
}
_ADAPTER_CATALOG: dict[str, dict[str, Any]] = {
    str(key): dict(value)
    for key, value in kernel_catalog_mapping("capabilities", "adapter_catalog").items()
    if isinstance(value, dict)
}


def builtin_capabilities(config: KernelConfig) -> dict[str, dict[str, Any]]:
    operator_policy = operator_policy_snapshot(config)
    return {
        "workspace_fs": {
            "kind": "builtin",
            "enabled": True,
            "scope": "workspace_only",
        },
        "workspace_exec": {
            "kind": "builtin",
            "enabled": True,
            "scope": "workspace_only",
        },
        "git_commands": {
            "kind": "builtin",
            "enabled": bool(operator_policy.get("unattended_allow_git_commands", False)),
            "scope": "workspace_only",
        },
        "generated_path_mutation": {
            "kind": "builtin",
            "enabled": bool(operator_policy.get("unattended_allow_generated_path_mutations", False)),
            "scope": "workspace_only",
        },
        "http_request": {
            "kind": "builtin",
            "enabled": bool(operator_policy.get("unattended_allow_http_requests", False)),
            "scope": {
                "allowed_hosts": effective_http_allowed_hosts(config),
                "timeout_seconds": int(operator_policy.get("unattended_http_timeout_seconds", 1)),
                "max_body_bytes": int(operator_policy.get("unattended_http_max_body_bytes", 1)),
            },
        },
    }


def load_capability_modules(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(payload, dict) and str(payload.get("artifact_kind", "")).strip() == "capability_module_set":
        retained = retained_artifact_payload(payload, artifact_kind="capability_module_set")
        if retained is None:
            return []
        modules = retained.get("modules", [])
    else:
        modules = payload.get("modules", []) if isinstance(payload, dict) else payload
    if not isinstance(modules, list):
        return []
    normalized: list[dict[str, Any]] = []
    for module in modules:
        if not isinstance(module, dict):
            continue
        module_id = str(module.get("module_id", "")).strip()
        if not module_id:
            continue
        adapter_kind = _adapter_kind_for_module(module)
        adapter = _ADAPTER_CATALOG.get(adapter_kind, {})
        capabilities = [
            str(capability).strip()
            for capability in module.get("capabilities", [])
            if str(capability).strip()
        ]
        if not capabilities:
            capabilities = [
                str(capability).strip()
                for capability in adapter.get("default_capabilities", [])
                if str(capability).strip()
            ]
        user_settings = dict(module.get("settings", {})) if isinstance(module.get("settings", {}), dict) else {}
        settings = _merge_adapter_settings(adapter_kind, user_settings)
        normalized_module = {
            "module_id": module_id,
            "adapter_kind": adapter_kind,
            "adapter_label": str(adapter.get("label", adapter_kind or "custom")).strip() or "custom",
            "enabled": bool(module.get("enabled", False)),
            "capabilities": capabilities,
            "user_settings": user_settings,
            "settings": settings,
        }
        issues = _validate_module(normalized_module)
        normalized.append(
            {
                **normalized_module,
                "valid": not issues,
                "issues": issues,
            }
        )
    return normalized


def capability_registry_snapshot(config: KernelConfig) -> dict[str, Any]:
    builtins = builtin_capabilities(config)
    modules = load_capability_modules(config.capability_modules_path)
    enabled_external_capabilities = sorted(
        {
            capability
            for module in modules
            if bool(module.get("enabled", False)) and bool(module.get("valid", False))
            for capability in module.get("capabilities", [])
        }
    )
    enabled_builtins = sorted(name for name, item in builtins.items() if bool(item.get("enabled", False)))
    enabled_all = sorted(set(enabled_builtins) | set(enabled_external_capabilities))
    return {
        "builtin": builtins,
        "modules": modules,
        "adapter_catalog": adapter_catalog_snapshot(),
        "invalid_enabled_modules": [
            {
                "module_id": str(module.get("module_id", "")).strip(),
                "adapter_kind": str(module.get("adapter_kind", "")).strip(),
                "issues": list(module.get("issues", [])),
            }
            for module in modules
            if bool(module.get("enabled", False)) and not bool(module.get("valid", False))
        ],
        "capability_policies": capability_policies_snapshot(config),
        "enabled_builtin_capabilities": enabled_builtins,
        "enabled_external_capabilities": enabled_external_capabilities,
        "enabled_capabilities": enabled_all,
        "modules_path": str(config.capability_modules_path),
    }


def capability_enabled(config: KernelConfig, capability: str) -> bool:
    normalized = str(capability).strip()
    if not normalized:
        return False
    registry = capability_registry_snapshot(config)
    return normalized in set(registry["enabled_capabilities"])


def effective_http_allowed_hosts(config: KernelConfig) -> list[str]:
    operator_policy = operator_policy_snapshot(config)
    allowed_hosts = {
        str(value).strip().lower()
        for value in operator_policy.get("unattended_http_allowed_hosts", [])
        if str(value).strip()
    }
    for module in load_capability_modules(config.capability_modules_path):
        if not bool(module.get("enabled", False)) or not bool(module.get("valid", False)):
            continue
        settings = module.get("settings", {})
        if not isinstance(settings, dict):
            continue
        for raw_value in settings.get("http_allowed_hosts", []) or []:
            value = str(raw_value).strip().lower()
            if value:
                allowed_hosts.add(value)
    return sorted(allowed_hosts)


def capability_policies_snapshot(config: KernelConfig) -> dict[str, list[dict[str, Any]]]:
    policies: dict[str, list[dict[str, Any]]] = {}
    for name, builtin in builtin_capabilities(config).items():
        policies[name] = [
            {
                "source": "builtin",
                "module_id": "builtin",
                "capability": name,
                "enabled": bool(builtin.get("enabled", False)),
                "scope": builtin.get("scope"),
                "http_allowed_hosts": (
                    list(builtin.get("scope", {}).get("allowed_hosts", []))
                    if isinstance(builtin.get("scope"), dict)
                    else []
                ),
                "repo_scopes": [],
                "account_scopes": [],
                "access_tier": "write" if name in {"workspace_fs", "workspace_exec", "git_commands"} else "read",
                "write_tier": "workspace_only" if name in {"workspace_fs", "workspace_exec", "git_commands"} else "none",
                "read_only": name == "http_request",
            }
        ]
    for module in load_capability_modules(config.capability_modules_path):
        if not bool(module.get("enabled", False)) or not bool(module.get("valid", False)):
            continue
        module_id = str(module.get("module_id", "")).strip()
        for capability in module.get("capabilities", []):
            normalized = str(capability).strip()
            if not normalized:
                continue
            merged_settings = _module_capability_settings(module, normalized)
            policies.setdefault(normalized, []).append(
                {
                    "source": "module",
                    "module_id": module_id,
                    "adapter_kind": str(module.get("adapter_kind", "")).strip(),
                    "capability": normalized,
                    "enabled": True,
                    "scope": merged_settings,
                    "http_allowed_hosts": _string_list(merged_settings.get("http_allowed_hosts", [])),
                    "repo_scopes": _string_list(
                        merged_settings.get("repo_scopes", merged_settings.get("allowed_repos", []))
                    ),
                    "account_scopes": _string_list(
                        merged_settings.get("account_scopes", merged_settings.get("allowed_accounts", []))
                    ),
                    "access_tier": _normalize_tier(merged_settings.get("access_tier"), default="read"),
                    "write_tier": str(merged_settings.get("write_tier", "none")).strip() or "none",
                    "read_only": bool(merged_settings.get("read_only", False)),
                }
            )
    return {capability: entries for capability, entries in sorted(policies.items())}


def enabled_corpus_sources(config: KernelConfig) -> list[dict[str, Any]]:
    allowed_hosts = set(effective_http_allowed_hosts(config))
    sources: dict[tuple[str, str], dict[str, Any]] = {}
    for module in load_capability_modules(config.capability_modules_path):
        if not bool(module.get("enabled", False)) or not bool(module.get("valid", False)):
            continue
        module_id = str(module.get("module_id", "")).strip()
        adapter_kind = str(module.get("adapter_kind", "")).strip()
        settings = dict(module.get("settings", {})) if isinstance(module.get("settings", {}), dict) else {}
        for url in _module_corpus_urls(module):
            host = _url_host(url)
            if not host or (allowed_hosts and host not in allowed_hosts and not any(host.endswith(f".{candidate}") for candidate in allowed_hosts)):
                continue
            key = (module_id, url)
            sources[key] = {
                "module_id": module_id,
                "adapter_kind": adapter_kind,
                "url": url,
                "host": host,
                "repo_scopes": _string_list(settings.get("repo_scopes", [])),
                "account_scopes": _string_list(settings.get("account_scopes", [])),
            }
    return [sources[key] for key in sorted(sources)]


def capability_policy_satisfies(config: KernelConfig, capability: str, requirements: dict[str, Any]) -> bool:
    normalized = str(capability).strip()
    if not normalized:
        return False
    requirement_map = dict(requirements) if isinstance(requirements, dict) else {}
    policies = capability_policies_snapshot(config).get(normalized, [])
    if not policies:
        return False
    return any(_policy_satisfies(policy, requirement_map) for policy in policies if bool(policy.get("enabled", False)))


def declared_task_capabilities(task: TaskSpec) -> list[str]:
    workflow_guard = task.metadata.get("workflow_guard", {})
    guard = dict(workflow_guard) if isinstance(workflow_guard, dict) else {}
    declared = {
        str(value).strip()
        for value in guard.get("required_capabilities", [])
        if str(value).strip()
    }
    if guard.get("requires_git"):
        declared.add("git_commands")
    if guard.get("requires_http"):
        declared.add("http_request")
    if guard.get("touches_generated_paths"):
        declared.add("generated_path_mutation")
    return sorted(declared)


def segment_capabilities(argv: list[str], *, stdout_path: str = "") -> list[str]:
    if not argv:
        return []
    command_name = Path(argv[0]).name
    capabilities = {"workspace_exec"}
    if stdout_path:
        capabilities.add("workspace_fs")
    if command_name == "git":
        capabilities.add("git_commands")
    if command_name == "http_request":
        capabilities.add("http_request")
    if command_name in _FILESYSTEM_COMMANDS or "/" in argv[0]:
        capabilities.add("workspace_fs")
    if any(_looks_like_path(token) for token in argv[1:]):
        capabilities.add("workspace_fs")
    return sorted(capabilities)


def command_capabilities(command: str) -> list[str]:
    normalized = str(command).strip()
    if not normalized:
        return []
    try:
        lexer = shlex.shlex(normalized, posix=True, punctuation_chars="&|;<>")
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return []
    if not tokens:
        return []
    capabilities: set[str] = set()
    argv: list[str] = []
    stdout_path = ""
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "&&":
            capabilities.update(segment_capabilities(argv, stdout_path=stdout_path))
            argv = []
            stdout_path = ""
            index += 1
            continue
        if token in {">", ">>"}:
            index += 1
            stdout_path = tokens[index] if index < len(tokens) else ""
            index += 1
            continue
        if token in _SHELL_OPERATORS:
            index += 1
            continue
        argv.append(token)
        index += 1
    capabilities.update(segment_capabilities(argv, stdout_path=stdout_path))
    return sorted(capabilities)


def capability_usage_summary(task: TaskSpec, command_steps: list[StepRecord]) -> dict[str, Any]:
    declared = declared_task_capabilities(task)
    used: set[str] = set()
    for step in command_steps:
        command_result = step.command_result or {}
        capabilities = command_result.get("capabilities_used")
        if isinstance(capabilities, list) and capabilities:
            used.update(str(value).strip() for value in capabilities if str(value).strip())
            continue
        used.update(command_capabilities(step.content))
    builtin = sorted(capability for capability in used if capability in _BUILTIN_CAPABILITY_NAMES)
    external = sorted(capability for capability in used if capability not in set(builtin))
    return {
        "baseline_capabilities": ["workspace_fs", "workspace_exec"],
        "required_capabilities": declared,
        "used_capabilities": sorted(used),
        "builtin_capabilities_used": builtin,
        "external_capabilities_used": external,
        "declared_but_unused_capabilities": sorted(set(declared) - used),
    }


def _looks_like_path(token: str) -> bool:
    value = str(token).strip()
    if not value or value.startswith("-"):
        return False
    if value in {".", ".."}:
        return True
    return "/" in value or "." in Path(value).name


def _string_list(values: object) -> list[str]:
    if isinstance(values, dict):
        iterable = values.keys()
    elif isinstance(values, (list, tuple, set)):
        iterable = values
    else:
        iterable = []
    return sorted({str(value).strip() for value in iterable if str(value).strip()})


def adapter_catalog_snapshot() -> dict[str, dict[str, Any]]:
    return {
        adapter_kind: {
            "label": str(adapter.get("label", adapter_kind)).strip() or adapter_kind,
            "default_capabilities": list(adapter.get("default_capabilities", [])),
            "default_settings": dict(adapter.get("default_settings", {})),
        }
        for adapter_kind, adapter in sorted(_ADAPTER_CATALOG.items())
    }


def _adapter_kind_for_module(module: dict[str, Any]) -> str:
    explicit = str(module.get("adapter_kind", "")).strip().lower()
    if explicit:
        return explicit
    inferred = str(module.get("module_id", "")).strip().lower()
    return inferred if inferred in _ADAPTER_CATALOG else "custom"


def _merge_adapter_settings(adapter_kind: str, raw_settings: object) -> dict[str, Any]:
    settings = dict(raw_settings) if isinstance(raw_settings, dict) else {}
    adapter = _ADAPTER_CATALOG.get(adapter_kind, {})
    merged = dict(adapter.get("default_settings", {}))
    merged.update(settings)
    if "capability_settings" in settings and isinstance(settings.get("capability_settings"), dict):
        merged["capability_settings"] = dict(settings["capability_settings"])
    return merged


def _module_capability_settings(module: dict[str, Any], capability: str) -> dict[str, Any]:
    user_settings = dict(module.get("user_settings", {})) if isinstance(module.get("user_settings", {}), dict) else {}
    adapter_kind = str(module.get("adapter_kind", "")).strip()
    adapter_settings = dict(_ADAPTER_CATALOG.get(adapter_kind, {}).get("default_settings", {}))
    per_capability_settings = (
        dict(user_settings.get("capability_settings", {}))
        if isinstance(user_settings.get("capability_settings", {}), dict)
        else {}
    )
    merged_settings = dict(adapter_settings)
    capability_defaults = (
        dict(_ADAPTER_CATALOG.get(adapter_kind, {}).get("capability_defaults", {}).get(capability, {}))
        if adapter_kind in _ADAPTER_CATALOG
        else {}
    )
    merged_settings.update(capability_defaults)
    merged_settings.update(user_settings)
    override = per_capability_settings.get(capability, {})
    if isinstance(override, dict):
        merged_settings.update(override)
    merged_settings.pop("capability_settings", None)
    return merged_settings


def _validate_module(module: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    adapter_kind = str(module.get("adapter_kind", "")).strip()
    if adapter_kind and adapter_kind not in _ADAPTER_CATALOG and adapter_kind != "custom":
        issues.append(f"unknown adapter_kind={adapter_kind}")
    capabilities = [
        str(capability).strip()
        for capability in module.get("capabilities", [])
        if str(capability).strip()
    ]
    if not capabilities:
        issues.append("no capabilities declared")
    settings = dict(module.get("settings", {})) if isinstance(module.get("settings", {}), dict) else {}
    http_hosts = _string_list(settings.get("http_allowed_hosts", []))
    corpus_seed_urls = _string_list(settings.get("corpus_seed_urls", []))
    corpus_modes = _string_list(settings.get("corpus_modes", []))
    if adapter_kind == "generic_http" and not http_hosts:
        issues.append("generic_http adapter requires http_allowed_hosts")
    for capability in capabilities:
        merged_settings = _module_capability_settings(module, capability)
        read_only = bool(merged_settings.get("read_only", False))
        access_tier = _normalize_tier(merged_settings.get("access_tier"), default="read")
        write_tier = str(merged_settings.get("write_tier", "none")).strip() or "none"
        repo_scopes = _string_list(
            merged_settings.get("repo_scopes", merged_settings.get("allowed_repos", []))
        )
        account_scopes = _string_list(
            merged_settings.get("account_scopes", merged_settings.get("allowed_accounts", []))
        )
        if read_only and access_tier in {"write", "admin"}:
            issues.append(f"{capability} declares read_only with access_tier={access_tier}")
        if read_only and write_tier != "none":
            issues.append(f"{capability} declares read_only with write_tier={write_tier}")
        channel_scopes = _string_list(merged_settings.get("channel_scopes", []))
        if capability in {"github_repo_write", "twitter_post", "gitlab_repo_write"} and not repo_scopes and not account_scopes:
            issues.append(f"{capability} requires repo_scopes or account_scopes")
        if capability == "github_repo_write" and write_tier == "none":
            issues.append("github_repo_write requires non-none write_tier")
        if capability == "github_repo_write" and read_only:
            issues.append("github_repo_write cannot be read_only")
        if capability == "gitlab_repo_write" and write_tier == "none":
            issues.append("gitlab_repo_write requires non-none write_tier")
        if capability == "gitlab_repo_write" and read_only:
            issues.append("gitlab_repo_write cannot be read_only")
        if capability == "slack_post" and not channel_scopes and not account_scopes:
            issues.append("slack_post requires channel_scopes or account_scopes")
        if capability == "slack_post" and write_tier == "none":
            issues.append("slack_post requires non-none write_tier")
        if capability == "slack_post" and read_only:
            issues.append("slack_post cannot be read_only")
        if capability.startswith("github_") and not http_hosts:
            issues.append(f"{capability} requires http_allowed_hosts")
        if capability.startswith("gitlab_") and not http_hosts:
            issues.append(f"{capability} requires http_allowed_hosts")
        if capability.startswith("slack_") and not http_hosts:
            issues.append(f"{capability} requires http_allowed_hosts")
        if capability.startswith("youtube_") and not http_hosts:
            issues.append(f"{capability} requires http_allowed_hosts")
        if capability.startswith("twitter_") and not http_hosts:
            issues.append(f"{capability} requires http_allowed_hosts")
    if adapter_kind == "github":
        allowed_modes = {"repo_metadata", "readme", "releases", "issues", "pulls", "account_profile", "account_repos", "seed_urls"}
        for mode in corpus_modes:
            if mode not in allowed_modes:
                issues.append(f"unsupported github corpus_mode={mode}")
    if adapter_kind == "gitlab":
        allowed_modes = {"project_metadata", "readme", "releases", "merge_requests", "account_profile", "seed_urls"}
        for mode in corpus_modes:
            if mode not in allowed_modes:
                issues.append(f"unsupported gitlab corpus_mode={mode}")
    if adapter_kind == "slack":
        allowed_modes = {"channel_history", "channel_pins", "account_profile", "seed_urls"}
        for mode in corpus_modes:
            if mode not in allowed_modes:
                issues.append(f"unsupported slack corpus_mode={mode}")
    for raw_url in corpus_seed_urls:
        host = _url_host(raw_url)
        if not host:
            issues.append(f"corpus_seed_urls contains malformed url: {raw_url}")
            continue
        if http_hosts and host not in http_hosts and not any(host.endswith(f".{candidate}") for candidate in http_hosts):
            issues.append(f"corpus_seed_url host not allowed by http_allowed_hosts: {host}")
    return sorted(set(issues))


def _normalize_tier(value: object, *, default: str) -> str:
    candidate = str(value).strip().lower()
    if candidate in _TIER_ORDER:
        return candidate
    return default


def _policy_satisfies(policy: dict[str, Any], requirements: dict[str, Any]) -> bool:
    if not requirements:
        return True
    required_hosts = _string_list(requirements.get("http_allowed_hosts", []))
    if required_hosts and not set(required_hosts).issubset(set(_string_list(policy.get("http_allowed_hosts", [])))):
        return False
    required_repos = _string_list(requirements.get("repo_scopes", requirements.get("allowed_repos", [])))
    if required_repos and not set(required_repos).issubset(set(_string_list(policy.get("repo_scopes", [])))):
        return False
    required_accounts = _string_list(requirements.get("account_scopes", requirements.get("allowed_accounts", [])))
    if required_accounts and not set(required_accounts).issubset(set(_string_list(policy.get("account_scopes", [])))):
        return False
    required_access_tier = _normalize_tier(requirements.get("access_tier"), default="none")
    policy_access_tier = _normalize_tier(policy.get("access_tier"), default="none")
    if _TIER_ORDER[policy_access_tier] < _TIER_ORDER[required_access_tier]:
        return False
    if bool(requirements.get("allow_write", False)) and bool(policy.get("read_only", False)):
        return False
    required_write_tier = str(requirements.get("write_tier", "")).strip()
    if required_write_tier and str(policy.get("write_tier", "")).strip() != required_write_tier:
        return False
    return True


def _module_corpus_urls(module: dict[str, Any]) -> list[str]:
    settings = dict(module.get("settings", {})) if isinstance(module.get("settings", {}), dict) else {}
    urls = _string_list(settings.get("corpus_seed_urls", []))
    adapter_kind = str(module.get("adapter_kind", "")).strip()
    corpus_modes = set(_string_list(settings.get("corpus_modes", [])))
    if adapter_kind == "github":
        for repo in _string_list(settings.get("repo_scopes", [])):
            if not corpus_modes or "repo_metadata" in corpus_modes:
                urls.append(f"https://api.github.com/repos/{repo}")
            if not corpus_modes or "readme" in corpus_modes:
                urls.append(f"https://api.github.com/repos/{repo}/readme")
            if "releases" in corpus_modes:
                urls.append(f"https://api.github.com/repos/{repo}/releases/latest")
            if "issues" in corpus_modes:
                urls.append(f"https://api.github.com/repos/{repo}/issues?state=open&per_page=20")
            if "pulls" in corpus_modes:
                urls.append(f"https://api.github.com/repos/{repo}/pulls?state=open&per_page=20")
        for account in _string_list(settings.get("account_scopes", [])):
            if not corpus_modes or "account_profile" in corpus_modes:
                urls.append(f"https://api.github.com/users/{account}")
            if "account_repos" in corpus_modes:
                urls.append(f"https://api.github.com/users/{account}/repos?sort=updated&per_page=20")
    if adapter_kind == "gitlab":
        for repo in _string_list(settings.get("repo_scopes", [])):
            encoded_repo = url_parse.quote(repo, safe="")
            if not corpus_modes or "project_metadata" in corpus_modes:
                urls.append(f"https://gitlab.com/api/v4/projects/{encoded_repo}")
            if not corpus_modes or "readme" in corpus_modes:
                urls.append(
                    f"https://gitlab.com/api/v4/projects/{encoded_repo}/repository/files/README.md/raw?ref=HEAD"
                )
            if "releases" in corpus_modes:
                urls.append(f"https://gitlab.com/api/v4/projects/{encoded_repo}/releases")
            if "merge_requests" in corpus_modes:
                urls.append(f"https://gitlab.com/api/v4/projects/{encoded_repo}/merge_requests?state=opened&per_page=20")
        for account in _string_list(settings.get("account_scopes", [])):
            if not corpus_modes or "account_profile" in corpus_modes:
                urls.append(f"https://gitlab.com/api/v4/users?username={url_parse.quote(account, safe='')}")
    if adapter_kind == "slack":
        for channel in _string_list(settings.get("channel_scopes", [])):
            if not corpus_modes or "channel_history" in corpus_modes:
                urls.append(
                    f"https://slack.com/api/conversations.history?channel={url_parse.quote(channel, safe='')}&limit=200"
                )
            if "channel_pins" in corpus_modes:
                urls.append(f"https://slack.com/api/pins.list?channel={url_parse.quote(channel, safe='')}")
        for account in _string_list(settings.get("account_scopes", [])):
            if not corpus_modes or "account_profile" in corpus_modes:
                urls.append(f"https://slack.com/api/users.info?user={url_parse.quote(account, safe='')}")
    return sorted({url for url in urls if url})


def _url_host(url: str) -> str:
    try:
        parsed = url_parse.urlparse(str(url).strip())
    except ValueError:
        return ""
    return (parsed.hostname or "").strip().lower()
