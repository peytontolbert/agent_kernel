from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from .config import KernelConfig
from .improvement_common import retained_artifact_payload
from .sandbox import sandbox_containment_status
from .schemas import TaskSpec
from .universe_improvement import (
    DEFAULT_UNIVERSE_ACTION_RISK_CONTROLS,
    DEFAULT_UNIVERSE_ENVIRONMENT_ASSUMPTIONS,
    UNIVERSE_ACTION_RISK_CONTROL_KEYS,
    UNIVERSE_ENVIRONMENT_ASSUMPTION_BOOL_FIELDS,
    UNIVERSE_ENVIRONMENT_ASSUMPTION_ENUM_FIELDS,
)

_DEFAULT_UNIVERSE_CONSTITUTION: dict[str, object] = {
    "artifact_kind": "universe_constitution",
    "lifecycle_state": "retained",
    "universe_id": "agent_kernel_bounded_autonomy",
    "stability": "stable",
    "governance_mode": "bounded_autonomous",
    "invariants": [
        "preserve verifier contract alignment",
        "prefer measurable reversible progress",
        "avoid destructive workspace resets",
        "keep autonomous actions inside bounded runtime policy",
    ],
    "forbidden_command_patterns": [
        "rm -rf /",
        "git reset --hard",
        "git checkout --",
        "curl -s",
        "curl -fsSL",
        "wget -qO-",
        "sudo ",
    ],
    "preferred_command_prefixes": [
        "rg ",
        "sed -n ",
        "cat ",
        "ls ",
        "test ",
        "pytest",
        "python -m pytest",
        "git diff",
        "git status",
    ],
    "governance": {
        "require_verification": True,
        "require_bounded_steps": True,
        "prefer_reversible_actions": True,
        "respect_task_forbidden_artifacts": True,
        "respect_preserved_artifacts": True,
    },
}
_DEFAULT_OPERATING_ENVELOPE: dict[str, object] = {
    "artifact_kind": "operating_envelope",
    "lifecycle_state": "retained",
    "action_risk_controls": dict(DEFAULT_UNIVERSE_ACTION_RISK_CONTROLS),
    "environment_assumptions": dict(DEFAULT_UNIVERSE_ENVIRONMENT_ASSUMPTIONS),
    "allowed_http_hosts": [],
    "writable_path_prefixes": [],
    "toolchain_requirements": ["git", "python", "pytest", "rg"],
    "learned_calibration_priors": {},
}
_CORE_DESTRUCTIVE_RESET_PATTERNS = {"rm -rf /", "git reset --hard", "git checkout --"}


class UniverseModel:
    def __init__(self, config: KernelConfig | None = None) -> None:
        self.config = config or KernelConfig()
        self._constitution_cache: dict[str, object] | None = None
        self._operating_envelope_cache: dict[str, object] | None = None

    def summarize(
        self,
        task: TaskSpec,
        *,
        world_model_summary: dict[str, object] | None = None,
        workspace: Path | None = None,
        planned_commands: list[str] | None = None,
    ) -> dict[str, object]:
        constitution = self._constitution()
        operating_envelope = self._operating_envelope()
        governance = constitution.get("governance", {})
        if not isinstance(governance, dict):
            governance = {}
        world = world_model_summary or {}
        action_risk_controls = self._normalized_action_risk_controls(operating_envelope.get("action_risk_controls", {}))
        environment_assumptions = self._normalized_environment_assumptions(
            operating_envelope.get("environment_assumptions", {})
        )
        runtime_attestation = self._runtime_attestation(
            task,
            workspace=workspace,
            world_model_summary=world,
        )
        environment_snapshot = self._normalized_environment_snapshot(
            runtime_attestation.get("environment_snapshot", {})
        )
        envelope_alignment = self._environment_alignment(
            assumptions=environment_assumptions,
            snapshot=environment_snapshot,
        )
        summary = {
            "universe_id": str(constitution.get("universe_id", "agent_kernel_bounded_autonomy")).strip()
            or "agent_kernel_bounded_autonomy",
            "stability": str(constitution.get("stability", "stable")).strip() or "stable",
            "governance_mode": str(constitution.get("governance_mode", "bounded_autonomous")).strip()
            or "bounded_autonomous",
            "constitution": {
                "artifact_kind": "universe_constitution",
                "invariants": self._string_list(constitution.get("invariants", [])),
                "forbidden_command_patterns": self._string_list(
                    constitution.get("forbidden_command_patterns", [])
                ),
                "preferred_command_prefixes": self._string_list(
                    constitution.get("preferred_command_prefixes", [])
                ),
                "governance": {
                    key: bool(governance.get(key, False))
                    for key in (
                        "require_verification",
                        "require_bounded_steps",
                        "prefer_reversible_actions",
                        "respect_task_forbidden_artifacts",
                        "respect_preserved_artifacts",
                    )
                },
            },
            "operating_envelope": {
                "artifact_kind": "operating_envelope",
                "action_risk_controls": action_risk_controls,
                "environment_assumptions": environment_assumptions,
                "allowed_http_hosts": self._string_list(operating_envelope.get("allowed_http_hosts", [])),
                "writable_path_prefixes": self._string_list(operating_envelope.get("writable_path_prefixes", [])),
                "toolchain_requirements": self._string_list(operating_envelope.get("toolchain_requirements", [])),
                "learned_calibration_priors": self._dict_copy(
                    operating_envelope.get("learned_calibration_priors", {})
                ),
            },
            "invariants": self._string_list(constitution.get("invariants", [])),
            "forbidden_command_patterns": self._string_list(constitution.get("forbidden_command_patterns", [])),
            "preferred_command_prefixes": self._string_list(constitution.get("preferred_command_prefixes", [])),
            "requires_verification": bool(governance.get("require_verification", True)),
            "requires_bounded_steps": bool(governance.get("require_bounded_steps", True)),
            "prefer_reversible_actions": bool(governance.get("prefer_reversible_actions", True)),
            "respect_task_forbidden_artifacts": bool(governance.get("respect_task_forbidden_artifacts", True)),
            "respect_preserved_artifacts": bool(governance.get("respect_preserved_artifacts", True)),
            "action_risk_controls": action_risk_controls,
            "environment_assumptions": environment_assumptions,
            "constitutional_compliance": self._constitutional_compliance(
                constitution=constitution,
                operating_envelope=operating_envelope,
            ),
            "envelope_alignment": envelope_alignment,
            "environment_alignment": envelope_alignment,
            "runtime_attestation": runtime_attestation,
            "environment_snapshot": environment_snapshot,
            "autonomy_scope": {
                "provider": str(self.config.provider),
                "tolbert_mode": str(self.config.tolbert_mode),
                "benchmark_family": str(task.metadata.get("benchmark_family", "bounded")),
                "difficulty": str(task.metadata.get("difficulty", "unknown")),
                "shared_repo": bool(world.get("workflow_shared_repo", False)),
                "unattended_git_mutation_allowed": bool(self.config.unattended_allow_git_commands),
                "unattended_http_allowed": bool(self.config.unattended_allow_http_requests),
            },
        }
        summary["plan_risk_summary"] = self.assess_plan_risk(
            summary,
            self._planned_commands(task, planned_commands=planned_commands),
        )
        return summary

    def score_command(self, summary: dict[str, object], command: str) -> int:
        normalized = str(command).strip().lower()
        if not normalized:
            return 0
        score = 0
        profile = self._command_profile(normalized)
        action_risk_controls = self._normalized_action_risk_controls(summary.get("action_risk_controls", {}))
        environment_assumptions = self._normalized_environment_assumptions(summary.get("environment_assumptions", {}))
        environment_snapshot = self._normalized_environment_snapshot(summary.get("environment_snapshot", {}))
        forbidden_patterns = [
            pattern.lower()
            for pattern in self._string_list(summary.get("forbidden_command_patterns", []))
        ]
        preferred_prefixes = [
            prefix.lower()
            for prefix in self._string_list(summary.get("preferred_command_prefixes", []))
        ]
        for pattern in forbidden_patterns:
            if pattern and pattern in normalized:
                score -= 12
        if bool(summary.get("prefer_reversible_actions", True)):
            for prefix in preferred_prefixes:
                if prefix and normalized.startswith(prefix):
                    score += 3
        if profile["read_only_discovery"]:
            score += int(action_risk_controls.get("read_only_discovery_bonus", 0))
        if profile["reversible_file_operation"] and bool(summary.get("prefer_reversible_actions", True)):
            score += int(action_risk_controls.get("reversible_file_operation_bonus", 0))
        if bool(summary.get("requires_verification", True)) and (
            normalized.startswith("pytest")
            or normalized.startswith("python -m pytest")
            or normalized.startswith("test ")
            or " verify" in normalized
        ):
            score += int(action_risk_controls.get("verification_bonus", 0))
        if profile["git_mutation"] and self._git_mutation_penalized(
            assumptions=environment_assumptions,
            snapshot=environment_snapshot,
        ):
            score -= int(action_risk_controls.get("git_mutation_penalty", 0))
        if profile["network_fetch"] and self._network_fetch_penalized(
            assumptions=environment_assumptions,
            snapshot=environment_snapshot,
            profile=profile,
        ):
            score -= int(action_risk_controls.get("network_fetch_penalty", 0))
        if profile["destructive_mutation"]:
            score -= int(action_risk_controls.get("destructive_mutation_penalty", 0))
        if profile["remote_execution"]:
            score -= int(action_risk_controls.get("remote_execution_penalty", 0))
        if profile["privileged_command"]:
            score -= int(action_risk_controls.get("privileged_command_penalty", 0))
        if profile["inline_destructive_interpreter"]:
            score -= int(action_risk_controls.get("inline_destructive_interpreter_penalty", 0))
        if profile["unbounded_execution"] and bool(summary.get("requires_bounded_steps", True)):
            score -= int(action_risk_controls.get("unbounded_execution_penalty", 0))
        if profile["workspace_scope_escape"] and bool(
            environment_assumptions.get("require_path_scoped_mutations", True)
        ):
            score -= int(action_risk_controls.get("scope_escape_penalty", 0))
        return score

    def simulate_command_governance(self, summary: dict[str, object], command: str) -> dict[str, object]:
        normalized = str(command).strip().lower()
        profile = self._command_profile(normalized)
        environment_assumptions = self._normalized_environment_assumptions(summary.get("environment_assumptions", {}))
        environment_snapshot = self._normalized_environment_snapshot(summary.get("environment_snapshot", {}))
        forbidden_matches = [
            pattern
            for pattern in self._string_list(summary.get("forbidden_command_patterns", []))
            if pattern and pattern.lower() in normalized
        ]
        preferred_matches = [
            prefix
            for prefix in self._string_list(summary.get("preferred_command_prefixes", []))
            if prefix and normalized.startswith(prefix.lower())
        ]
        risk_flags = [
            key
            for key in (
                "destructive_mutation",
                "git_mutation",
                "inline_destructive_interpreter",
                "network_fetch",
                "privileged_command",
                "remote_execution",
                "workspace_scope_escape",
                "unbounded_execution",
            )
            if profile[key]
        ]
        if profile["network_fetch"] and self._network_fetch_penalized(
            assumptions=environment_assumptions,
            snapshot=environment_snapshot,
            profile=profile,
        ):
            risk_flags.append("network_access_conflict")
        if profile["git_mutation"] and self._git_mutation_penalized(
            assumptions=environment_assumptions,
            snapshot=environment_snapshot,
        ):
            risk_flags.append("git_write_conflict")
        if profile["workspace_scope_escape"] and bool(
            environment_assumptions.get("require_path_scoped_mutations", True)
        ):
            risk_flags.append("path_scope_conflict")
        action_categories = [
            key
            for key in (
                "read_only_discovery",
                "reversible_file_operation",
                "verification_aligned",
            )
            if profile[key]
        ]
        return {
            "score": self.score_command(summary, command),
            "forbidden_pattern_matches": forbidden_matches,
            "preferred_prefix_matches": preferred_matches,
            "verification_aligned": profile["verification_aligned"]
            and bool(summary.get("requires_verification", True)),
            "read_only_git_command": self._is_read_only_git_command(normalized)
            if normalized.startswith("git ")
            else False,
            "action_categories": action_categories,
            "risk_flags": risk_flags,
            "environment_alignment": self._environment_alignment(
                assumptions=environment_assumptions,
                snapshot=environment_snapshot,
            ),
            "network_host": str(profile.get("network_host", "")).strip(),
        }

    def assess_plan_risk(self, summary: dict[str, object], commands: list[str]) -> dict[str, object]:
        normalized_commands = [str(command).strip() for command in commands if str(command).strip()]
        if not normalized_commands:
            return {
                "command_count": 0,
                "risk_level": "low",
                "cumulative_mutation_surface": 0,
                "rollback_coverage": bool(
                    summary.get("environment_assumptions", {}).get("require_rollback_on_mutation", False)
                ),
                "verifier_coverage": 0,
                "network_escalation": 0,
                "git_escalation": 0,
                "scope_escape_risk": 0,
            }
        reports = [self.simulate_command_governance(summary, command) for command in normalized_commands]
        risk_counts: dict[str, int] = {}
        cumulative_mutation_surface = 0
        negative_scores = 0
        max_command_risk = 0
        verification_coverage = 0
        for report in reports:
            score = int(report.get("score", 0))
            if score < 0:
                negative_scores += abs(score)
                max_command_risk = max(max_command_risk, abs(score))
            if bool(report.get("verification_aligned", False)):
                verification_coverage += 1
            flags = report.get("risk_flags", [])
            if isinstance(flags, list):
                for label in flags:
                    key = str(label).strip()
                    if not key:
                        continue
                    risk_counts[key] = risk_counts.get(key, 0) + 1
                cumulative_mutation_surface += sum(
                    1
                    for label in flags
                    if str(label).strip()
                    in {
                        "destructive_mutation",
                        "git_mutation",
                        "network_fetch",
                        "remote_execution",
                        "workspace_scope_escape",
                    }
                )
        if (
            risk_counts.get("destructive_mutation", 0) > 0
            or risk_counts.get("remote_execution", 0) > 0
            or risk_counts.get("network_access_conflict", 0) > 0
            or risk_counts.get("path_scope_conflict", 0) > 0
        ):
            risk_level = "high"
        elif negative_scores > 0 or risk_counts.get("git_mutation", 0) > 0:
            risk_level = "medium"
        else:
            risk_level = "low"
        return {
            "command_count": len(normalized_commands),
            "commands_evaluated": normalized_commands[:8],
            "risk_level": risk_level,
            "cumulative_mutation_surface": cumulative_mutation_surface,
            "rollback_coverage": bool(
                summary.get("environment_assumptions", {}).get("require_rollback_on_mutation", False)
            ),
            "verifier_coverage": verification_coverage,
            "network_escalation": risk_counts.get("network_fetch", 0)
            + risk_counts.get("network_access_conflict", 0),
            "git_escalation": risk_counts.get("git_mutation", 0)
            + risk_counts.get("git_write_conflict", 0),
            "scope_escape_risk": risk_counts.get("workspace_scope_escape", 0)
            + risk_counts.get("path_scope_conflict", 0),
            "max_command_risk": max_command_risk,
        }

    def _constitution(self) -> dict[str, object]:
        if self._constitution_cache is not None:
            return dict(self._constitution_cache)
        constitution = dict(_DEFAULT_UNIVERSE_CONSTITUTION)
        payload = self._load_runtime_artifact(
            self.config.universe_constitution_path,
            artifact_kind="universe_constitution",
        )
        if not payload:
            payload = self._legacy_contract_payload()
        if payload:
            constitution = {
                **constitution,
                **{
                    key: value
                    for key, value in payload.items()
                    if key
                    in {
                        "universe_id",
                        "stability",
                        "governance_mode",
                    }
                },
            }
            constitution["invariants"] = self._merged_string_list(
                constitution.get("invariants", []),
                payload.get("invariants", []),
            )
            constitution["forbidden_command_patterns"] = self._merged_string_list(
                constitution.get("forbidden_command_patterns", []),
                payload.get("forbidden_command_patterns", []),
            )
            constitution["preferred_command_prefixes"] = self._merged_string_list(
                constitution.get("preferred_command_prefixes", []),
                payload.get("preferred_command_prefixes", []),
            )
            constitution["governance"] = {
                **self._dict_copy(_DEFAULT_UNIVERSE_CONSTITUTION.get("governance", {})),
                **self._dict_copy(payload.get("governance", {})),
            }
        self._constitution_cache = constitution
        return dict(constitution)

    def _operating_envelope(self) -> dict[str, object]:
        if self._operating_envelope_cache is not None:
            return dict(self._operating_envelope_cache)
        envelope = dict(_DEFAULT_OPERATING_ENVELOPE)
        payload = self._load_runtime_artifact(
            self.config.operating_envelope_path,
            artifact_kind="operating_envelope",
        )
        if not payload:
            payload = self._legacy_contract_payload()
        if payload:
            envelope["action_risk_controls"] = {
                **self._normalized_action_risk_controls(
                    _DEFAULT_OPERATING_ENVELOPE.get("action_risk_controls", {})
                ),
                **self._normalized_action_risk_controls(payload.get("action_risk_controls", {})),
            }
            envelope["environment_assumptions"] = {
                **self._normalized_environment_assumptions(
                    _DEFAULT_OPERATING_ENVELOPE.get("environment_assumptions", {})
                ),
                **self._normalized_environment_assumptions(payload.get("environment_assumptions", {})),
            }
            envelope["allowed_http_hosts"] = self._string_list(
                payload.get("allowed_http_hosts", self.config.unattended_http_allowed_hosts)
            )
            envelope["writable_path_prefixes"] = self._string_list(
                payload.get("writable_path_prefixes", [])
            )
            envelope["toolchain_requirements"] = self._string_list(
                payload.get("toolchain_requirements", envelope.get("toolchain_requirements", []))
            )
            envelope["learned_calibration_priors"] = self._dict_copy(
                payload.get("learned_calibration_priors", {})
            )
        self._operating_envelope_cache = envelope
        return dict(envelope)

    def _legacy_contract_payload(self) -> dict[str, object]:
        return self._load_runtime_artifact(
            self.config.universe_contract_path,
            artifact_kind="universe_contract",
        )

    def _load_runtime_artifact(self, path: Path, *, artifact_kind: str) -> dict[str, object]:
        if not path.exists():
            return {}
        payload = self._load_json_payload(path)
        if not payload:
            return {}
        if str(payload.get("artifact_kind", "")).strip() == artifact_kind:
            if any(
                key in payload
                for key in ("lifecycle_state", "spec_version", "retention_gate", "retention_decision")
            ):
                return retained_artifact_payload(payload, artifact_kind=artifact_kind) or {}
            return payload
        return {}

    @staticmethod
    def _load_json_payload(path: Path) -> dict[str, object]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _dict_copy(value: object) -> dict[str, object]:
        return dict(value) if isinstance(value, dict) else {}

    @staticmethod
    def _string_list(values: object) -> list[str]:
        if not isinstance(values, list):
            return []
        return [str(value).strip() for value in values if str(value).strip()]

    @staticmethod
    def _merged_string_list(defaults: object, overrides: object) -> list[str]:
        seen: set[str] = set()
        merged: list[str] = []
        for source in (defaults, overrides):
            if not isinstance(source, list):
                continue
            for item in source:
                value = str(item).strip()
                if value and value not in seen:
                    seen.add(value)
                    merged.append(value)
        return merged

    @staticmethod
    def _is_read_only_git_command(command: str) -> bool:
        normalized = str(command).strip().lower()
        return normalized.startswith(
            (
                "git status",
                "git diff",
                "git log",
                "git show",
                "git branch --show-current",
                "git rev-parse",
            )
        )

    @staticmethod
    def _normalized_action_risk_controls(controls: object) -> dict[str, int]:
        normalized: dict[str, int] = {}
        if not isinstance(controls, dict):
            return normalized
        for key in sorted(UNIVERSE_ACTION_RISK_CONTROL_KEYS):
            value = controls.get(key)
            if isinstance(value, bool):
                continue
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                continue
            if parsed > 0:
                normalized[key] = parsed
        return normalized

    @staticmethod
    def _normalized_environment_assumptions(assumptions: object) -> dict[str, object]:
        normalized: dict[str, object] = {}
        if not isinstance(assumptions, dict):
            return normalized
        for key, allowed_values in UNIVERSE_ENVIRONMENT_ASSUMPTION_ENUM_FIELDS.items():
            value = str(assumptions.get(key, "")).strip().lower()
            if value in allowed_values:
                normalized[key] = value
        for key in UNIVERSE_ENVIRONMENT_ASSUMPTION_BOOL_FIELDS:
            if key in assumptions:
                normalized[key] = bool(assumptions.get(key, False))
        return normalized

    @staticmethod
    def _normalized_environment_snapshot(snapshot: object) -> dict[str, object]:
        if not isinstance(snapshot, dict):
            return {}
        normalized: dict[str, object] = {}
        for key in ("network_access_mode", "git_write_mode", "workspace_write_scope"):
            value = str(snapshot.get(key, "")).strip().lower()
            if value:
                normalized[key] = value
        allowed_http_hosts = snapshot.get("allowed_http_hosts", [])
        if isinstance(allowed_http_hosts, list):
            normalized["allowed_http_hosts"] = [
                str(value).strip().lower()
                for value in allowed_http_hosts
                if str(value).strip()
            ]
        return normalized

    def _runtime_attestation(
        self,
        task: TaskSpec,
        *,
        workspace: Path | None,
        world_model_summary: dict[str, object],
    ) -> dict[str, object]:
        workspace_path = (workspace or (self.config.workspace_root / task.workspace_subdir)).resolve()
        workspace_path.mkdir(parents=True, exist_ok=True)
        environment_snapshot = self._environment_snapshot(world_model_summary=world_model_summary)
        containment = sandbox_containment_status(self.config, cwd=workspace_path)
        repo_status = self._repo_status(workspace_path)
        toolchain_available = {
            name: bool(shutil.which(name))
            for name in ("git", "python", "pytest", "rg")
        }
        writable_paths = [
            str(path.resolve())
            for path in (
                self.config.workspace_root,
                self.config.trajectories_root,
                workspace_path,
            )
            if path.exists() and os.access(path, os.W_OK)
        ]
        return {
            "repo_dirty": bool(repo_status.get("dirty", False)),
            "repo_root": str(repo_status.get("root", "")).strip(),
            "writable_paths": writable_paths,
            "actual_network_reach_mode": str(environment_snapshot.get("network_access_mode", "blocked")),
            "available_hosts": list(environment_snapshot.get("allowed_http_hosts", [])),
            "toolchain_available": toolchain_available,
            "sandbox_mode": {
                "provider": str(self.config.provider),
                "execution_mode": "contained" if bool(containment.get("active", False)) else "direct",
                "containment_active": bool(containment.get("active", False)),
                "containment_reason": str(containment.get("reason", "")).strip(),
            },
            "environment_snapshot": environment_snapshot,
        }

    def _environment_snapshot(self, *, world_model_summary: dict[str, object]) -> dict[str, object]:
        if not self.config.unattended_allow_http_requests:
            network_access_mode = "blocked"
        elif self.config.unattended_http_allowed_hosts:
            network_access_mode = "allowlist_only"
        else:
            network_access_mode = "open"
        if not self.config.unattended_allow_git_commands:
            git_write_mode = "blocked"
        else:
            git_write_mode = "task_scoped"
        if bool(world_model_summary.get("workflow_shared_repo", False)):
            workspace_write_scope = "shared_repo_gated"
        elif self.config.unattended_allow_generated_path_mutations:
            workspace_write_scope = "generated_only"
        else:
            workspace_write_scope = "task_only"
        return {
            "network_access_mode": network_access_mode,
            "git_write_mode": git_write_mode,
            "workspace_write_scope": workspace_write_scope,
            "allowed_http_hosts": [
                str(value).strip().lower()
                for value in self.config.unattended_http_allowed_hosts
                if str(value).strip()
            ],
        }

    def _constitutional_compliance(
        self,
        *,
        constitution: dict[str, object],
        operating_envelope: dict[str, object],
    ) -> dict[str, object]:
        governance = self._dict_copy(constitution.get("governance", {}))
        forbidden_patterns = set(self._string_list(constitution.get("forbidden_command_patterns", [])))
        environment_assumptions = self._normalized_environment_assumptions(
            operating_envelope.get("environment_assumptions", {})
        )
        compliance = {
            "verification_required": bool(governance.get("require_verification", False)),
            "bounded_steps_required": bool(governance.get("require_bounded_steps", False)),
            "destructive_reset_prohibited": _CORE_DESTRUCTIVE_RESET_PATTERNS.issubset(forbidden_patterns),
            "path_scoped_mutations_required": bool(
                environment_assumptions.get("require_path_scoped_mutations", False)
            ),
            "rollback_required": bool(environment_assumptions.get("require_rollback_on_mutation", False)),
        }
        compliance["all_satisfied"] = all(bool(value) for value in compliance.values())
        return compliance

    def _environment_alignment(
        self,
        *,
        assumptions: dict[str, object],
        snapshot: dict[str, object],
    ) -> dict[str, object]:
        return {
            "network_access_aligned": self._mode_within_assumption(
                str(snapshot.get("network_access_mode", "")),
                str(assumptions.get("network_access_mode", "blocked")),
                order=("blocked", "allowlist_only", "open"),
            ),
            "git_write_aligned": self._mode_within_assumption(
                str(snapshot.get("git_write_mode", "")),
                str(assumptions.get("git_write_mode", "blocked")),
                order=("blocked", "operator_gated", "task_scoped"),
            ),
            "workspace_scope_aligned": self._mode_within_assumption(
                str(snapshot.get("workspace_write_scope", "")),
                str(assumptions.get("workspace_write_scope", "task_only")),
                order=("task_only", "generated_only", "shared_repo_gated"),
            ),
        }

    @staticmethod
    def _mode_within_assumption(actual: str, assumed: str, *, order: tuple[str, ...]) -> bool:
        rank = {value: index for index, value in enumerate(order)}
        if actual not in rank or assumed not in rank:
            return False
        return rank[actual] <= rank[assumed]

    def _planned_commands(self, task: TaskSpec, *, planned_commands: list[str] | None = None) -> list[str]:
        if isinstance(planned_commands, list) and planned_commands:
            return [str(command).strip() for command in planned_commands if str(command).strip()]
        commands = list(task.setup_commands)
        commands.extend(task.suggested_commands)
        if task.success_command:
            commands.append(task.success_command)
        return [str(command).strip() for command in commands if str(command).strip()]

    def _command_profile(self, normalized_command: str) -> dict[str, bool | str]:
        network_host = self._network_host(normalized_command)
        network_fetch = (
            normalized_command.startswith("curl ")
            or " curl " in normalized_command
            or normalized_command.startswith("wget ")
            or " wget " in normalized_command
        )
        remote_execution = (
            ("| sh" in normalized_command)
            or ("| bash" in normalized_command)
            or ("bash -c" in normalized_command and ("curl" in normalized_command or "wget" in normalized_command))
        )
        inline_destructive_interpreter = (
            "python -c" in normalized_command
            and (
                "shutil.rmtree" in normalized_command
                or ".unlink(" in normalized_command
                or "os.remove" in normalized_command
            )
        )
        destructive_mutation = any(
            token in normalized_command
            for token in (
                "rm -rf",
                "git reset --hard",
                "git checkout --",
                "git clean -fd",
                "shutil.rmtree",
            )
        )
        read_only_git_command = (
            self._is_read_only_git_command(normalized_command)
            if normalized_command.startswith("git ")
            else False
        )
        git_mutation = normalized_command.startswith("git ") and not read_only_git_command
        read_only_discovery = normalized_command.startswith(
            (
                "rg ",
                "sed -n ",
                "cat ",
                "ls ",
                "find ",
                "test ",
            )
        ) or read_only_git_command
        reversible_file_operation = normalized_command.startswith(("mkdir -p ", "cp ", "touch "))
        verification_aligned = (
            normalized_command.startswith("pytest")
            or normalized_command.startswith("python -m pytest")
            or normalized_command.startswith("test ")
            or " verify" in normalized_command
        )
        privileged_command = normalized_command.startswith("sudo ") or " sudo " in normalized_command
        workspace_scope_escape = bool(
            re.search(r"(?:^|[\s'\"=])(?:\.\./|/tmp/|/etc/|/var/|/root/|/home/)", normalized_command)
        )
        unbounded_execution = any(
            token in normalized_command
            for token in (
                "while true",
                "for (;;)",
                "for ((;;))",
                "yes |",
            )
        )
        return {
            "destructive_mutation": destructive_mutation,
            "git_mutation": git_mutation,
            "inline_destructive_interpreter": inline_destructive_interpreter,
            "network_fetch": network_fetch,
            "network_host": network_host,
            "privileged_command": privileged_command,
            "read_only_discovery": read_only_discovery,
            "remote_execution": remote_execution,
            "reversible_file_operation": reversible_file_operation,
            "workspace_scope_escape": workspace_scope_escape,
            "unbounded_execution": unbounded_execution,
            "verification_aligned": verification_aligned,
        }

    @staticmethod
    def _network_host(normalized_command: str) -> str:
        match = re.search(r"https?://([a-z0-9.-]+)", normalized_command)
        return str(match.group(1)).strip().lower() if match else ""

    def _repo_status(self, cwd: Path) -> dict[str, object]:
        if not shutil.which("git"):
            return {"dirty": False, "root": ""}
        try:
            root_result = subprocess.run(
                ["git", "-C", str(cwd), "rev-parse", "--show-toplevel"],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError:
            return {"dirty": False, "root": ""}
        root = root_result.stdout.strip() if root_result.returncode == 0 else ""
        if not root:
            return {"dirty": False, "root": ""}
        status_result = subprocess.run(
            ["git", "-C", root, "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
        )
        return {
            "dirty": bool(status_result.stdout.strip()) if status_result.returncode == 0 else False,
            "root": root,
        }

    def _git_mutation_penalized(
        self,
        *,
        assumptions: dict[str, object],
        snapshot: dict[str, object],
    ) -> bool:
        assumed_mode = str(assumptions.get("git_write_mode", "operator_gated"))
        actual_mode = str(snapshot.get("git_write_mode", "blocked"))
        return assumed_mode in {"blocked", "operator_gated"} or actual_mode == "blocked"

    def _network_fetch_penalized(
        self,
        *,
        assumptions: dict[str, object],
        snapshot: dict[str, object],
        profile: dict[str, bool | str],
    ) -> bool:
        assumed_mode = str(assumptions.get("network_access_mode", "blocked"))
        actual_mode = str(snapshot.get("network_access_mode", "blocked"))
        host = str(profile.get("network_host", "")).strip().lower()
        allowed_hosts = {
            str(value).strip().lower()
            for value in snapshot.get("allowed_http_hosts", [])
            if str(value).strip()
        }
        if assumed_mode == "blocked" or actual_mode == "blocked":
            return True
        if assumed_mode == "allowlist_only" or actual_mode == "allowlist_only":
            return not host or host not in allowed_hosts
        return False
