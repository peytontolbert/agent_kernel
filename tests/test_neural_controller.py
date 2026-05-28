import argparse
import json
from pathlib import Path

import pytest

from agent_kernel.config import KernelConfig
from agent_kernel.loop import AgentKernel
from agent_kernel.llm import MockLLMClient
from agent_kernel.neural_controller import (
    FULL_KERNEL_CONTROL_TOKENS,
    attach_neural_controller_verified_contract_metadata,
    artifact_command_target_from_encoder,
    artifact_command_target_from_task,
    artifact_slot_target_from_encoder,
    artifact_slot_target_from_task,
    build_neural_controller_advisory,
    build_neural_controller_encoder_text,
    compact_neural_controller_shadow,
    command_copy_target_from_encoder,
    guarded_neural_controller_source,
    load_neural_controller_manifest,
    localized_edit_candidate_from_pointer_token,
    localized_edit_candidate_index_from_token,
    localized_edit_candidate_pointer_token,
    localized_edit_candidates_from_encoder,
    materialization_candidate_from_pointer_token,
    materialization_candidate_index_from_token,
    materialization_candidates_from_encoder,
    augment_encoder_with_active_materialization_target,
    augment_encoder_with_plan_source_inspection_candidates,
    neural_controller_exec_kind_family,
    neural_controller_surfaces,
    neural_controller_shadow_promotion_readiness,
    parse_neural_controller_line_protocol,
    plan_source_inspection_candidates_from_encoder,
    repair_line_protocol_with_command_copy_target,
    select_verified_neural_controller_shadow,
    source_inspection_candidates_from_encoder,
    summarize_neural_controller_shadow_documents,
    summarize_neural_controller_shadow_steps,
)
from agent_kernel.modeling.neural_controller_runtime import _line_protocol_prediction_complete
from agent_kernel.policy import LLMDecisionPolicy
from agent_kernel.policy import Policy
from agent_kernel.extensions.policy_runtime_support import PolicyRuntimeSupport
from agent_kernel.schemas import ActionDecision, EpisodeRecord, StepRecord, TaskSpec
from agent_kernel.state import AgentState
from agent_kernel.tasking.task_bank import TaskBank
from agent_kernel.extensions.extractors import build_episode_summary
from agent_kernel.learning_compiler import compile_episode_learning_candidates
from scripts.build_agentkernel_controller_direct_command_dataset import (
    DIRECT_CODE_CONTROL_TOKENS,
    _copy_pointer_candidate,
    _direct_command_rows,
)
from scripts.train_agentkernel_lite_encdec import LocalBpeTokenizer
from scripts.build_agentkernel_controller_long_horizon_dataset import _control_tokens
from scripts.build_agentkernel_controller_long_horizon_dataset import _command_argument_slots
from scripts.build_agentkernel_controller_long_horizon_dataset import _execution_intent_token
from scripts.build_agentkernel_controller_slot_curriculum import build_slot_curriculum
from scripts.evaluate_neural_controller_shadow_dataset import evaluate_dataset
from scripts.evaluate_neural_controller_shadow_dataset import _target_line_protocol
from scripts.evaluate_neural_controller_shadow_dataset import summarize_family_metrics
from scripts.compare_neural_controller_family_metrics import compare_family_metrics
from scripts.merge_agentkernel_lite_datasets import DatasetWriter
from scripts.merge_agentkernel_lite_datasets import _union_lists
from scripts.report_neural_controller_rowwise_frontier import report_rowwise_frontier
from scripts.evaluate_neural_controller_rowwise_selector import evaluate_rowwise_selector
from scripts.report_neural_controller_rowwise_selector_readiness import report_selector_readiness
from scripts.build_neural_controller_selector_retained_candidate_packet import (
    build_selector_retained_candidate_packet,
)
from scripts.report_neural_controller_runtime_contract_metrics import summarize_runtime_contract_metrics
from scripts.report_neural_controller_selector_activation_gate import report_selector_activation_gate


def test_source_inspection_candidates_expand_plan_workflow_paths() -> None:
    encoder = "\n".join(
        [
            "Active subgoal: update workflow path src/main.py",
            "Plan: update workflow path src/main.py | update workflow path tests/test_main.py | materialize expected artifact patch.diff",
        ]
    )

    candidates = plan_source_inspection_candidates_from_encoder(encoder)

    assert "cat src/main.py" in candidates
    assert "cat source_lines/src/main.py.lines" in candidates
    assert "cat src/main.py tests/test_main.py" in candidates
    assert "cat src/main.py tests/test_main.py 2>/dev/null || echo 'Files not found or empty'" in candidates


def test_encoder_augmentation_inserts_plan_source_inspection_candidates() -> None:
    encoder = "\n".join(
        [
            "Active subgoal: update workflow path src/main.py",
            "Plan: update workflow path src/main.py | update workflow path tests/test_main.py | validate expected artifacts and forbidden artifacts before termination",
            "Command copy target: cat src/main.py",
        ]
    )

    augmented = augment_encoder_with_plan_source_inspection_candidates(encoder)

    assert "Source inspection candidate commands:" in augmented
    assert "Source inspection candidate 1:" in augmented
    assert "cat src/main.py tests/test_main.py" in augmented
from scripts.select_neural_controller_candidate import select_candidate
from scripts.select_neural_controller_checkpoints import checkpoint_label
from scripts.build_neural_controller_preservation_replay import build_preservation_replay
from scripts.compose_neural_controller_guarded_report import compose_guarded_report
from scripts.report_neural_controller_retained_promotion_gate import build_retained_promotion_gate


def _write_controller_manifest(tmp_path):
    dataset_path = tmp_path / "dataset_manifest.json"
    dataset_path.write_text(
        json.dumps(
            {
                "artifact_kind": "agentkernel_controller_trace_dataset",
                "objective": "agentkernel_controller_trace_policy",
                "agentkernel_special_tokens": [
                    "<AK_ACTION_SPACE_CODE>",
                    "<AK_ACTION_SPACE_ARTIFACT>",
                    "<AK_RETRIEVE>",
                    "<AK_RET_CODE>",
                    "<AK_VERIFY>",
                    "<AK_OOD>",
                    "<AK_SAFE_STOP>",
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "controller_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "artifact_kind": "agentkernel_lite_encdec_bundle",
                "model_family": "agentkernel_lite_encdec_v1",
                "model_dir": str(tmp_path / "model"),
                "tokenizer_dir": str(tmp_path / "tokenizer"),
                "dataset_manifest_path": str(dataset_path),
                "parameter_count": 142_656_135,
                "model_config": {
                    "retrieval_head_dim": 256,
                    "agent_policy_heads": True,
                },
                "training_summary": {
                    "dataset_objective": "agentkernel_controller_trace_x5_plus_retrieval",
                    "completed_steps": 3000,
                },
                "replaces_surfaces": ["chat_decision_generation"],
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def test_neural_controller_manifest_maps_full_kernel_surfaces(tmp_path):
    manifest_path = _write_controller_manifest(tmp_path)

    manifest = load_neural_controller_manifest(manifest_path)

    assert manifest is not None
    assert manifest.has_neural_retrieval is True
    assert manifest.has_policy_heads is True
    assert manifest.is_full_kernel_controller_trace is True
    surfaces = neural_controller_surfaces(manifest)
    assert "encoder_neural_retrieval_query_embedding" in surfaces
    assert "controller_action_validity_head" in surfaces
    assert "full_kernel_artifact_repair_policy" in surfaces


def test_full_kernel_control_tokens_cover_runtime_and_improvement_loop():
    required = {
        "<AK_BOOTSTRAP>",
        "<AK_MEMORY_READ>",
        "<AK_WORLD_STATE>",
        "<AK_GOVERNANCE>",
        "<AK_CONTEXT_COMPILE>",
        "<AK_PLAN>",
        "<AK_DECIDE>",
        "<AK_ACTION_SPACE_CODE>",
        "<AK_ACTION_SPACE_ARTIFACT>",
        "<AK_ACTION_SPACE_RETRIEVAL>",
        "<AK_EXECUTE>",
        "<AK_VERIFY>",
        "<AK_WORLD_UPDATE>",
        "<AK_MEMORY_WRITE>",
        "<AK_LEARN_COMPILE>",
        "<AK_IMPROVE_SELECT>",
        "<AK_IMPROVE_GENERATE>",
        "<AK_IMPROVE_EVALUATE>",
        "<AK_RETAIN>",
        "<AK_REJECT>",
        "<AK_VALIDATE_PRESENT>",
        "<AK_VALIDATE_ABSENT>",
        "<AK_READ_SOURCE>",
        "<AK_EXEC_KIND_MATERIALIZE_ARTIFACT>",
        "<AK_EXEC_KIND_VERIFY_PRESENT>",
        "<AK_EXEC_KIND_VERIFY_ABSENT>",
        "<AK_EXEC_KIND_INSPECT_SOURCE>",
        "<AK_EXEC_KIND_LOCALIZED_EDIT>",
        "<AK_EXEC_KIND_RUN_CHECK>",
        "<AK_COPY_COMMAND_TARGET>",
        "<AK_COPY_ARTIFACT_TARGET>",
        "<AK_COPY_ARTIFACT_PATH>",
        "<AK_COPY_ARTIFACT_CONTENT>",
        "<AK_SAFE_STOP>",
        "<AK_CLOSEOUT>",
    }

    assert required.issubset(set(FULL_KERNEL_CONTROL_TOKENS))


def test_training_tokenizer_defaults_cover_full_kernel_control_tokens():
    missing = set(FULL_KERNEL_CONTROL_TOKENS) - set(LocalBpeTokenizer.default_agentkernel_special_tokens)
    assert not missing


def test_neural_controller_encoder_declares_kernel_phase_loop():
    encoder_text = build_neural_controller_encoder_text(
        state_payload={
            "task": {
                "task_id": "phase_loop_task",
                "prompt": "write an artifact",
                "workspace_subdir": "phase_loop_task",
                "suggested_commands": ["printf 'ok\\n' > result.txt"],
                "metadata": {"benchmark_family": "unit"},
            },
            "history": [],
            "active_subgoal": "direct bounded workspace action",
        }
    )

    loop_line = next(line for line in encoder_text.splitlines() if line.startswith("<AK_LOOP>"))
    assert "<AK_BOOTSTRAP>" in loop_line
    assert "<AK_MEMORY_READ>" in loop_line
    assert "<AK_CONTEXT_COMPILE>" in loop_line
    assert "<AK_DECIDE>" in loop_line


def test_neural_controller_encoder_preserves_long_command_copy_target():
    long_command = (
        "mkdir -p reports && "
        + " && ".join(
            f"printf 'segment {index}\\n' > reports/segment_{index}.txt"
            for index in range(30)
        )
    )
    encoder_text = build_neural_controller_encoder_text(
        state_payload={
            "task": {
                "task_id": "long_command_task",
                "prompt": "write many report segments",
                "workspace_subdir": "long_command_task",
                "suggested_commands": [long_command],
                "metadata": {"benchmark_family": "unit"},
            },
            "history": [],
            "active_subgoal": "direct bounded workspace action",
        }
    )

    assert command_copy_target_from_encoder(encoder_text) == long_command.replace("\n", "\\n")


def test_neural_controller_encoder_adds_artifact_command_target_for_active_subgoal():
    encoder_text = build_neural_controller_encoder_text(
        state_payload={
            "task": {
                "task_id": "artifact_task",
                "prompt": "write expected artifact",
                "expected_files": ["reports/status.txt"],
                "expected_file_contents": {"reports/status.txt": "ready"},
            },
            "active_subgoal": "materialize expected artifact reports/status.txt",
            "world_model_summary": {
                "missing_expected_artifacts": ["reports/status.txt"],
                "present_forbidden_artifacts": ["tmp/debug.log"],
            },
        }
    )

    target = artifact_command_target_from_encoder(encoder_text)

    assert target == "mkdir -p reports && printf %s 'ready' > reports/status.txt"
    assert "Current artifact target path: reports/status.txt" in encoder_text
    assert "Current artifact target content: ready" in encoder_text
    assert "Validation target present paths: reports/status.txt" in encoder_text
    assert "Validation target absent paths: tmp/debug.log" in encoder_text
    assert "Validation present commands: test -f reports/status.txt" in encoder_text
    assert "Validation absent commands: test ! -f tmp/debug.log" in encoder_text
    assert "Next-step target candidates: materialize:reports/status.txt | verify_absent:tmp/debug.log" in encoder_text
    assert artifact_slot_target_from_encoder(encoder_text) == ("reports/status.txt", "ready")
    assert artifact_command_target_from_task(
        active_subgoal="materialize expected artifact reports/status.txt",
        expected_file_contents={"reports/status.txt": "ready"},
    ) == target
    assert artifact_slot_target_from_task(
        active_subgoal="materialize expected artifact reports/status.txt",
        expected_file_contents={"reports/status.txt": "ready"},
    ) == ("reports/status.txt", "ready")


def test_neural_controller_encoder_adds_source_inspection_candidates():
    encoder_text = build_neural_controller_encoder_text(
        state_payload={
            "task": {
                "task_id": "inspect_recovery_task",
                "prompt": "inspect recovery before deciding",
                "expected_files": ["recovery.txt"],
                "expected_file_contents": {"recovery.txt": "file recovery complete\n"},
                "suggested_commands": [
                    "printf 'file recovery complete\\n' > recovery.txt",
                    "cat recovery.txt",
                ],
            },
            "history": [
                {"action": "code_execute", "content": "head -n 20 recovery.txt"},
            ],
            "active_subgoal": "materialize expected artifact recovery.txt",
            "world_model_summary": {
                "existing_expected_artifacts": ["recovery.txt"],
                "expected_artifacts": ["recovery.txt"],
                "missing_expected_artifacts": [],
            },
        }
    )

    assert "Next-step target candidates: materialize:recovery.txt | verify_present:recovery.txt" in encoder_text
    assert (
        "Source inspection candidate commands: cat recovery.txt | head -n 20 recovery.txt"
        in encoder_text
    )
    assert source_inspection_candidates_from_encoder(encoder_text) == [
        "cat recovery.txt",
        "cat source_lines/recovery.txt.lines",
        "head -n 20 recovery.txt",
    ]


def test_neural_controller_encoder_adds_combined_source_candidates_from_plan():
    encoder_text = build_neural_controller_encoder_text(
        state_payload={
            "task": {
                "task_id": "source_plan_task",
                "prompt": "inspect related files",
            },
            "plan": [
                "update workflow path app.py",
                "update workflow path tests/test_app.py",
                "materialize expected artifact patch.diff",
            ],
        }
    )

    assert "Source inspection candidate commands: cat app.py tests/test_app.py" in encoder_text
    assert source_inspection_candidates_from_encoder(encoder_text)[:3] == [
        "cat app.py tests/test_app.py",
        "cat source_lines/tests/test_app.py.lines",
        "cat app.py",
    ]


def test_neural_controller_encoder_adds_localized_edit_candidates():
    encoder_text = build_neural_controller_encoder_text(
        state_payload={
            "task": {
                "task_id": "edit_task",
                "prompt": "edit a status file",
                "suggested_commands": ["sed -i '1s#pending#ready#' reports/status.txt"],
            },
            "history": [
                {
                    "action": "code_execute",
                    "content": "sed -i '2s#draft#final#' reports/status.txt",
                }
            ],
        }
    )

    assert "Localized edit candidate commands:" in encoder_text
    assert "Localized edit candidate 1: sed -i '1s#pending#ready#' reports/status.txt" in encoder_text
    assert "Localized edit candidate 1 fields: path=reports/status.txt ; old=pending ; new=ready" in encoder_text
    assert localized_edit_candidates_from_encoder(encoder_text) == [
        "sed -i '1s#pending#ready#' reports/status.txt",
    ]


def test_neural_controller_encoder_adds_trajectory_position():
    encoder_text = build_neural_controller_encoder_text(
        state_payload={
            "task": {"task_id": "multi_step", "prompt": "finish sequence"},
            "trajectory_step_index": 2,
            "trajectory_step_count": 7,
        }
    )

    assert "Trajectory position: step 3 of 7" in encoder_text


def test_neural_controller_parses_numbered_localized_edit_candidates_after_truncation():
    encoder_text = "\n".join(
        [
            "Localized edit candidate commands: sed -i '1s#old#new#' a.txt | sed -i '2s#too",
            "Localized edit candidate 1: sed -i '1s#old#new#' a.txt",
            "Localized edit candidate 2: sed -i '2s#pending#ready#' reports/status.txt",
        ]
    )

    assert localized_edit_candidates_from_encoder(encoder_text) == [
        "sed -i '1s#old#new#' a.txt",
        "sed -i '2s#pending#ready#' reports/status.txt",
    ]


def test_neural_controller_expands_localized_edit_candidate_pointer():
    encoder_text = "\n".join(
        [
            "Localized edit candidate 1: sed -i '1s#old#new#' a.txt",
            "Localized edit candidate 2: sed -i '2s#pending#ready#' reports/status.txt",
        ]
    )
    token = localized_edit_candidate_pointer_token(2)

    assert token == "<AK_COPY_LOCALIZED_EDIT_CANDIDATE_2>"
    assert localized_edit_candidate_index_from_token(token) == 2
    assert localized_edit_candidate_from_pointer_token(token, encoder_text) == (
        "sed -i '2s#pending#ready#' reports/status.txt"
    )

    repaired, warnings = repair_line_protocol_with_command_copy_target(
        {
            "tokens": ["<AK_EXEC_KIND_LOCALIZED_EDIT>", token],
            "action": "code_execute",
            "content": token,
        },
        encoder_text=encoder_text,
    )

    assert warnings == []
    assert repaired["content"] == "sed -i '2s#pending#ready#' reports/status.txt"
    assert repaired["target_path"] == "reports/status.txt"
    assert repaired["edit_old"] == "pending"
    assert repaired["edit_new"] == "ready"
    assert repaired["localized_edit_candidate_expanded"] is True


def test_neural_controller_repairs_ungrounded_localized_edit_to_frontier_candidate():
    encoder_text = "\n".join(
        [
            "Localized edit candidate 1: sed -i '1s#old#new#' a.txt",
            "Localized edit candidate 2: sed -i '2s#pending#ready#' reports/status.txt",
        ]
    )

    repaired, warnings = repair_line_protocol_with_command_copy_target(
        {
            "tokens": ["<AK_EXEC_KIND_LOCALIZED_EDIT>"],
            "action": "code_execute",
            "content": "sed -i '3s#stale#done#' unrelated.txt",
        },
        encoder_text=encoder_text,
    )

    assert warnings == []
    assert repaired["content"] == "sed -i '1s#old#new#' a.txt"
    assert repaired["target_path"] == "a.txt"
    assert repaired["edit_old"] == "old"
    assert repaired["edit_new"] == "new"
    assert repaired["localized_edit_candidate_repaired"] is True


def test_neural_controller_orders_localized_edit_candidates_by_success_frontier():
    encoder_text = build_neural_controller_encoder_text(
        state_payload={
            "task": {
                "task_id": "frontier_edit_task",
                "prompt": "finish ordered edits",
                "expected_file_contents": {
                    "reports/checklist.md": "status: ready\n- packet published",
                    "reports/packet.md": "title: validation handoff\nsummary: complete",
                },
                "success_command": (
                    "grep -q '^status: ready$' reports/checklist.md && "
                    "grep -q '^- packet published$' reports/checklist.md && "
                    "grep -q '^title: validation handoff$' reports/packet.md"
                ),
            },
            "history": [
                {
                    "action": "code_execute",
                    "content": "sed -i '1s#status\\ pending#status: ready#' reports/checklist.md",
                }
            ],
        }
    )

    candidates = localized_edit_candidates_from_encoder(encoder_text)

    assert candidates[0] == "sed -i '$a\\\\n- packet published' reports/checklist.md"


def test_neural_controller_orders_localized_edit_candidates_to_next_path_after_completed_frontier():
    encoder_text = build_neural_controller_encoder_text(
        state_payload={
            "task": {
                "task_id": "frontier_next_path_task",
                "prompt": "finish ordered edits",
                "expected_file_contents": {
                    "reports/checklist.md": "status: ready",
                    "reports/packet.md": "title: validation handoff\nsummary: complete",
                },
                "success_command": (
                    "grep -q '^status: ready$' reports/checklist.md && "
                    "grep -q '^title: validation handoff$' reports/packet.md"
                ),
            },
            "history": [
                {
                    "action": "code_execute",
                    "content": "sed -i '1s#status\\ pending#status: ready#' reports/checklist.md",
                }
            ],
        }
    )

    candidates = localized_edit_candidates_from_encoder(encoder_text)

    assert candidates[0] == "sed -i '1s#^title:\\ draft\\ handoff$#title: validation handoff#' reports/packet.md"


def test_neural_controller_orders_residual_append_after_expected_paths_touched():
    encoder_text = build_neural_controller_encoder_text(
        state_payload={
            "task": {
                "task_id": "residual_append_task",
                "prompt": "finish residual checklist append",
                "expected_files": ["reports/summary.md", "reports/checklist.md", "reports/verify.txt"],
                "expected_file_contents": {
                    "reports/summary.md": "title: validation handoff",
                    "reports/checklist.md": "checklist:\n- project handoff ready\n- verification complete\n- packet published",
                    "reports/verify.txt": "verification complete",
                },
            },
            "history": [
                {"action": "code_execute", "content": "sed -i '1s#^title:\\ draft\\ handoff$#title: validation handoff#' reports/summary.md"},
                {"action": "code_execute", "content": "sed -i '2s#^\\-\\ pending\\ validation$#- project handoff ready#' reports/checklist.md"},
                {"action": "code_execute", "content": "sed -i '3s#^\\-\\ pending\\ signoff$#- verification complete#' reports/checklist.md"},
                {"action": "code_execute", "content": "sed -i '1s#^verification\\ pending$#verification complete#' reports/verify.txt"},
            ],
        }
    )

    candidates = localized_edit_candidates_from_encoder(encoder_text)

    assert candidates[0] == "sed -i '$a\\\\n- packet published' reports/checklist.md"


def test_controller_dataset_targets_emit_full_kernel_phase_tokens():
    direct_tokens = DIRECT_CODE_CONTROL_TOKENS.split()
    for token in (
        "<AK_DECIDE>",
        "<AK_ACTION_SPACE_CODE>",
        "<AK_EXECUTE>",
        "<AK_VERIFY>",
        "<AK_WORLD_UPDATE>",
        "<AK_MEMORY_WRITE>",
    ):
        assert token in direct_tokens

    long_horizon_tokens = _control_tokens(
        {
            "decision_source": "artifact_repair",
            "verification": {"passed": False},
        },
        "code_execute",
        "python scripts/swe_patch_builder.py --instance x",
        "artifact_missing_after_response",
        terminal_step=True,
    ).split()
    for token in (
        "<AK_DECIDE>",
        "<AK_ACTION_SPACE_ARTIFACT>",
        "<AK_ARTIFACT_REPAIR>",
        "<AK_PATCH_BUILD>",
        "<AK_EXEC_KIND_RUN_CHECK>",
        "<AK_EXECUTE>",
        "<AK_VERIFY>",
        "<AK_WORLD_UPDATE>",
        "<AK_MEMORY_WRITE>",
        "<AK_SAFE_STOP>",
    ):
        assert token in long_horizon_tokens


def test_controller_dataset_failed_intermediate_step_is_repair_not_safe_stop():
    tokens = _control_tokens(
        {
            "decision_source": "artifact_repair",
            "verification": {"passed": False},
        },
        "code_execute",
        "python scripts/swe_patch_builder.py --instance x",
        "artifact_missing_after_response",
        terminal_step=False,
    ).split()

    assert "<AK_ARTIFACT_REPAIR>" in tokens
    assert "<AK_OOD>" in tokens
    assert "<AK_SAFE_STOP>" not in tokens


def test_controller_dataset_exec_kind_tokens_classify_code_execute_intent():
    cases = {
        "mkdir -p reports && printf %s 'ready' > reports/status.txt": "<AK_EXEC_KIND_MATERIALIZE_ARTIFACT>",
        "test -f reports/status.txt": "<AK_EXEC_KIND_VERIFY_PRESENT>",
        "test ! -f staging/draft.txt": "<AK_EXEC_KIND_VERIFY_ABSENT>",
        "cat source_lines/context.txt": "<AK_EXEC_KIND_INSPECT_SOURCE>",
        "sed -i '1s#old#new#' project/plan.md": "<AK_EXEC_KIND_LOCALIZED_EDIT>",
        "python -m pytest tests/test_policy.py": "<AK_EXEC_KIND_RUN_CHECK>",
    }
    for content, expected_token in cases.items():
        assert _execution_intent_token("code_execute", content) == expected_token


def test_controller_dataset_exec_kind_token_is_added_before_execute():
    tokens = _control_tokens(
        {
            "decision_source": "artifact_repair",
            "verification": {"passed": True},
        },
        "code_execute",
        "test ! -f staging/draft.txt",
        "artifact_contract_success",
    ).split()

    assert "<AK_EXEC_KIND_VERIFY_ABSENT>" in tokens
    assert "<AK_VALIDATE_ABSENT>" in tokens
    assert tokens.index("<AK_VALIDATE_ABSENT>") < tokens.index("<AK_EXEC_KIND_VERIFY_ABSENT>")
    assert tokens.index("<AK_EXEC_KIND_VERIFY_ABSENT>") < tokens.index("<AK_EXECUTE>")


def test_controller_dataset_argument_slots_extract_common_code_execute_shapes():
    assert _command_argument_slots("code_execute", "test ! -f staging/draft.txt") == {
        "target_path": "staging/draft.txt",
        "verify_polarity": "absent",
    }
    assert _command_argument_slots("code_execute", "test -f reports/status.txt") == {
        "target_path": "reports/status.txt",
        "verify_polarity": "present",
    }
    assert _command_argument_slots("code_execute", "mkdir -p reports && printf %s 'ready' > reports/status.txt") == {
        "target_path": "reports/status.txt",
        "target_content": "ready",
    }
    assert _command_argument_slots("code_execute", "sed -i '1s#old#new#' project/plan.md") == {
        "target_path": "project/plan.md",
        "edit_old": "old",
        "edit_new": "new",
    }


def test_neural_controller_line_protocol_parses_argument_slots():
    parsed = parse_neural_controller_line_protocol(
        "\n".join(
            [
                "<AK_DECIDE> <AK_EXEC_KIND_VERIFY_ABSENT>",
                "Action: code_execute",
                "Artifact-Failure-Mode: artifact_contract_success",
                "Target-Path: staging/draft.txt",
                "Verify-Polarity: absent",
                "Content: test ! -f staging/draft.txt",
            ]
        )
    )

    assert parsed["target_path"] == "staging/draft.txt"
    assert parsed["verify_polarity"] == "absent"


def test_neural_controller_runtime_stops_after_complete_line_protocol():
    assert _line_protocol_prediction_complete(
        "\n".join(
            [
                "<AK_ACTION_SPACE_CODE> <AK_PATCH_BUILD>",
                "Thought: build the required artifact",
                "Action: code_execute",
                "Content: patch_builder --path pkg/a.py --replace-line 1 --with 'x = 2' > patch.diff",
                "Done: false",
            ]
        )
    )
    assert not _line_protocol_prediction_complete(
        "\n".join(
            [
                "Thought: still planning",
                "Action: code_execute",
            ]
        )
    )


def test_neural_controller_line_protocol_infers_missing_slots_from_content():
    verify = parse_neural_controller_line_protocol(
        "\n".join(
            [
                "<AK_DECIDE> <AK_EXEC_KIND_VERIFY_ABSENT>",
                "Action: code_execute",
                "Content: test ! -f staging/draft.txt",
            ]
        )
    )
    materialize = parse_neural_controller_line_protocol(
        "\n".join(
            [
                "<AK_DECIDE> <AK_EXEC_KIND_MATERIALIZE_ARTIFACT>",
                "Action: code_execute",
                "Content: mkdir -p reports && printf %s 'ready\\n' > reports/status.txt",
            ]
        )
    )
    edit = parse_neural_controller_line_protocol(
        "\n".join(
            [
                "<AK_DECIDE> <AK_EXEC_KIND_LOCALIZED_EDIT>",
                "Action: code_execute",
                "Content: sed -i '1s#old#new#' project/plan.md",
            ]
        )
    )

    assert verify["target_path"] == "staging/draft.txt"
    assert verify["verify_polarity"] == "absent"
    assert materialize["target_path"] == "reports/status.txt"
    assert materialize["target_content"] == "ready\\n"
    assert edit["target_path"] == "project/plan.md"
    assert edit["edit_old"] == "old"
    assert edit["edit_new"] == "new"


def test_neural_controller_normalizes_misplaced_artifact_pointer_in_slot_fields():
    encoder_text = "\n".join(
        [
            "<AK_CONTEXT>",
            "Artifact command target: mkdir -p reports && printf %s 'ready\\n' > reports/status.txt",
            "Artifact target path: reports/status.txt",
            "Artifact target content: ready\\n",
        ]
    )
    line_protocol = {
        "tokens": ["<AK_ACTION_SPACE_CODE>", "<AK_COPY_ARTIFACT_TARGET>"],
        "action": "code_execute",
        "target_path": "<AK_COPY_ARTIFACT_TARGET>",
        "target_content": "<AK_COPY_ARTIFACT_TARGET>",
        "content": "<AK_COPY_ARTIFACT_TARGET>",
    }

    repaired, warnings = repair_line_protocol_with_command_copy_target(
        line_protocol,
        encoder_text=encoder_text,
    )

    assert repaired["target_path"] == "reports/status.txt"
    assert repaired["target_content"] == "ready\\n"
    assert repaired["content"] == "mkdir -p reports && printf %s 'ready\\n' > reports/status.txt"
    assert repaired["artifact_pointer_slot_normalized"] is True
    assert warnings == []
    compact = compact_neural_controller_shadow(
        {"ready": True, "line_protocol": repaired},
        selected_action="code_execute",
        selected_content=repaired["content"],
    )
    assert compact["artifact_pointer_slot_normalized"] is True


def test_neural_controller_expands_materialization_candidate_pointer():
    encoder_text = "\n".join(
        [
            "Artifact command target: mkdir -p reports && printf %s 'ready\\n' > reports/status.txt",
            "Command copy target: mkdir -p reports logs && printf 'ready\\n' > reports/status.txt && printf 'ok\\n' > logs/status.txt",
            "Materialization candidate 1: mkdir -p reports && printf %s 'ready\\n' > reports/status.txt",
            "Materialization candidate 2: mkdir -p reports logs && printf 'ready\\n' > reports/status.txt && printf 'ok\\n' > logs/status.txt",
        ]
    )
    token = "<AK_COPY_MATERIALIZE_CANDIDATE_2>"

    assert materialization_candidates_from_encoder(encoder_text) == [
        "mkdir -p reports && printf %s 'ready\\n' > reports/status.txt",
        "mkdir -p reports logs && printf 'ready\\n' > reports/status.txt && printf 'ok\\n' > logs/status.txt",
    ]
    assert materialization_candidate_index_from_token(token) == 2
    assert materialization_candidate_from_pointer_token(token, encoder_text) == (
        "mkdir -p reports logs && printf 'ready\\n' > reports/status.txt && printf 'ok\\n' > logs/status.txt"
    )

    repaired, warnings = repair_line_protocol_with_command_copy_target(
        {
            "tokens": ["<AK_EXEC_KIND_VERIFY_PRESENT>", token],
            "action": "code_execute",
            "content": "test -f reports/status.txt",
        },
        encoder_text=encoder_text,
    )

    assert warnings == []
    assert repaired["content"] == (
        "mkdir -p reports logs && printf 'ready\\n' > reports/status.txt && printf 'ok\\n' > logs/status.txt"
    )
    assert repaired["target_path"] == "reports/status.txt"
    assert repaired["target_content"] == "ready\\n"
    assert repaired["tokens"][0] == "<AK_EXEC_KIND_MATERIALIZE_ARTIFACT>"
    assert repaired["materialization_candidate_expanded"] is True


def test_neural_controller_encoder_exposes_active_materialization_target():
    command = (
        "mkdir -p semantic_open_world && printf '%s\\n' '# evidence' > "
        "semantic_open_world/integration_evidence.md && printf '%s\\n' "
        "'{\"task_id\":\"integration\"}' > semantic_open_world/integration_task.json"
    )
    encoder_text = build_neural_controller_encoder_text(
        state_payload={
            "task": {
                "task_id": "semantic_task",
                "prompt": "Write `semantic_open_world/integration_task.json` and evidence.",
                "workspace_subdir": "semantic_task",
                "suggested_commands": [command],
                "metadata": {"benchmark_family": "integration"},
            },
            "active_subgoal": "materialize expected artifact semantic_open_world/integration_task.json",
            "world_model_summary": {},
        }
    )

    assert "Active materialization target: " + command in encoder_text
    candidates = materialization_candidates_from_encoder(encoder_text)
    assert candidates[0] == command


def test_neural_controller_augments_stored_encoder_with_active_materialization_target():
    command = (
        "mkdir -p semantic_open_world && printf '%s\\n' '# evidence' > "
        "semantic_open_world/integration_evidence.md && printf '%s\\n' "
        "'{\"task_id\":\"integration\"}' > semantic_open_world/integration_task.json"
    )
    encoder_text = "\n".join(
        [
            "Active subgoal: materialize expected artifact semantic_open_world/integration_task.json",
            "Command copy target: " + command,
            "Materialization candidate commands: printf '%s\\n' '# evidence' > semantic_open_world/integration_evidence.md",
        ]
    )
    augmented = augment_encoder_with_active_materialization_target(encoder_text)

    assert "Active materialization target: " + command in augmented
    assert augmented.index("Active materialization target:") < augmented.index("Materialization candidate commands:")


def test_neural_controller_repairs_materialize_to_grounded_artifact_target():
    encoder_text = "\n".join(
        [
            "<AK_CONTEXT>",
            "Artifact command target: mkdir -p alerts && printf %s 'incident triaged\\n' > alerts/open.txt",
            "Artifact target path: alerts/open.txt",
            "Artifact target content: incident triaged\\n",
        ]
    )
    line_protocol = {
        "tokens": ["<AK_EXEC_KIND_MATERIALIZE_ARTIFACT>"],
        "action": "code_execute",
        "content": "test -f alerts/open.txt",
        "target_path": "alerts/open.txt",
        "target_content": "incident triaged\\n",
    }

    repaired, warnings = repair_line_protocol_with_command_copy_target(
        line_protocol,
        encoder_text=encoder_text,
    )

    assert warnings == []
    assert repaired["content"] == "mkdir -p alerts && printf %s 'incident triaged\\n' > alerts/open.txt"
    assert repaired["artifact_command_target_repaired"] is True
    compact = compact_neural_controller_shadow(
        {"ready": True, "line_protocol": repaired},
        selected_action="code_execute",
        selected_content=repaired["content"],
    )
    assert compact["artifact_command_target_repaired"] is True
    assert compact["content_exact_agreement"] is True


def test_neural_controller_preserves_valid_materialize_when_artifact_target_has_extra_command():
    encoder_text = "\n".join(
        [
            "<AK_CONTEXT>",
            "Artifact command target: mkdir -p plans && printf '{\"ok\":true}\\n' > plans/packet.json && printf '# ok\\n' > plans/packet.md",
            "Artifact target path: plans/packet.json",
            "Artifact target content: {\"ok\":true}\\n",
        ]
    )
    line_protocol = {
        "tokens": ["<AK_EXEC_KIND_MATERIALIZE_ARTIFACT>"],
        "action": "code_execute",
        "content": "mkdir -p plans && printf %s '{\"ok\":true}\\n' > plans/packet.json",
        "target_path": "plans/packet.json",
        "target_content": "{\"ok\":true}\\n",
    }

    repaired, warnings = repair_line_protocol_with_command_copy_target(
        line_protocol,
        encoder_text=encoder_text,
    )

    assert warnings == []
    assert repaired["content"] == "mkdir -p plans && printf %s '{\"ok\":true}\\n' > plans/packet.json"
    assert "artifact_command_target_repaired" not in repaired


def test_neural_controller_does_not_repair_materialize_without_grounded_slot_match():
    encoder_text = "\n".join(
        [
            "<AK_CONTEXT>",
            "Artifact command target: mkdir -p alerts && printf %s 'incident triaged\\n' > alerts/open.txt",
            "Artifact target path: alerts/open.txt",
            "Artifact target content: incident triaged\\n",
        ]
    )
    line_protocol = {
        "tokens": ["<AK_EXEC_KIND_MATERIALIZE_ARTIFACT>"],
        "action": "code_execute",
        "content": "mkdir -p runbook && printf %s 'step one\\n' > runbook/steps.txt",
        "target_path": "runbook/steps.txt",
        "target_content": "step one\\n",
    }

    repaired, warnings = repair_line_protocol_with_command_copy_target(
        line_protocol,
        encoder_text=encoder_text,
    )

    assert warnings == []
    assert repaired["content"] == "mkdir -p runbook && printf %s 'step one\\n' > runbook/steps.txt"
    assert "artifact_command_target_repaired" not in repaired


def test_neural_controller_preserves_materialize_with_same_redirect_path():
    encoder_text = "Command copy target: printf 'ready\\n' > docs/status.md && tests/test_docs.sh"
    line_protocol = {
        "tokens": ["<AK_EXEC_KIND_MATERIALIZE_ARTIFACT>"],
        "action": "code_execute",
        "content": "printf 'wrong\\n' > docs/status.md",
        "target_path": "docs/status.md",
    }

    repaired, warnings = repair_line_protocol_with_command_copy_target(
        line_protocol,
        encoder_text=encoder_text,
    )

    assert warnings == []
    assert repaired["content"] == "printf 'wrong\\n' > docs/status.md"
    assert "command_copy_target_repaired" not in repaired


def test_neural_controller_repairs_low_conf_artifact_output_to_command_copy_materialize():
    encoder_text = "Command copy target: printf 'ready\\n' > docs/status.md && tests/test_docs.sh"
    line_protocol = {
        "tokens": [
            "<AK_VALIDATE_PRESENT>",
            "<AK_EXEC_KIND_VERIFY_PRESENT>",
            "<AK_CONF_LOW>",
            "<AK_ARTIFACT_REPAIR>",
        ],
        "action": "code_execute",
        "content": "printf 'wrong\\n' > docs/other.md",
        "target_path": "docs/other.md",
        "verify_polarity": "present",
    }

    repaired, warnings = repair_line_protocol_with_command_copy_target(
        line_protocol,
        encoder_text=encoder_text,
    )

    assert warnings == []
    assert repaired["content"] == "printf 'ready\\n' > docs/status.md && tests/test_docs.sh"
    assert repaired["target_path"] == "docs/status.md"
    assert "<AK_EXEC_KIND_MATERIALIZE_ARTIFACT>" in repaired["tokens"]
    assert "<AK_EXEC_KIND_VERIFY_PRESENT>" not in repaired["tokens"]
    assert repaired["command_copy_target_repaired"] is True
    assert repaired["materialize_exec_kind_repaired"] is True


def test_neural_controller_does_not_repair_valid_validation_to_source_candidate():
    encoder_text = "\n".join(
        [
            "Validation present commands: test -f gateway/routes.txt",
            "Source inspection candidate commands: cat gateway/routes.txt",
        ]
    )
    line_protocol = {
        "tokens": [
            "<AK_EXEC_KIND_VERIFY_PRESENT>",
            "<AK_CONF_LOW>",
            "<AK_ARTIFACT_REPAIR>",
        ],
        "action": "code_execute",
        "content": "test -f gateway/routes.txt",
        "target_path": "gateway/routes.txt",
        "verify_polarity": "present",
    }

    repaired, warnings = repair_line_protocol_with_command_copy_target(
        line_protocol,
        encoder_text=encoder_text,
    )

    assert warnings == []
    assert repaired["content"] == "test -f gateway/routes.txt"
    assert "source_inspection_candidate_repaired" not in repaired


def test_neural_controller_preserves_validation_content_when_exec_kind_is_source():
    encoder_text = "\n".join(
        [
            "Validation present commands: test -f gateway/routes.txt",
            "Source inspection candidate commands: cat gateway/routes.txt",
        ]
    )
    line_protocol = {
        "tokens": [
            "<AK_EXEC_KIND_INSPECT_SOURCE>",
            "<AK_CONF_LOW>",
            "<AK_ARTIFACT_REPAIR>",
        ],
        "action": "code_execute",
        "content": "test -f gateway/routes.txt",
        "target_path": "gateway/routes.txt",
    }

    repaired, warnings = repair_line_protocol_with_command_copy_target(
        line_protocol,
        encoder_text=encoder_text,
    )

    assert warnings == []
    assert repaired["content"] == "test -f gateway/routes.txt"
    assert "<AK_EXEC_KIND_VERIFY_PRESENT>" in repaired["tokens"]
    assert "source_inspection_candidate_repaired" not in repaired
    assert repaired["validation_exec_kind_repaired"] is True


def test_neural_controller_repairs_verify_present_polarity_with_grounded_path():
    encoder_text = "Validation present commands: test -f gateway/routes.txt | test -f reports/health.txt"
    line_protocol = {
        "tokens": ["<AK_EXEC_KIND_VERIFY_PRESENT>"],
        "action": "code_execute",
        "content": "test ! -f gateway/routes.txt",
        "target_path": "gateway/routes.txt",
        "verify_polarity": "absent",
    }

    repaired, warnings = repair_line_protocol_with_command_copy_target(
        line_protocol,
        encoder_text=encoder_text,
    )

    assert warnings == []
    assert repaired["content"] == "test -f gateway/routes.txt"
    assert repaired["target_path"] == "gateway/routes.txt"
    assert repaired["verify_polarity"] == "present"
    assert repaired["validation_command_repaired"] is True


def test_neural_controller_repairs_single_verify_absent_candidate_for_unstable_probe():
    encoder_text = "Validation absent commands: test ! -f scratch/old_payload.json"
    line_protocol = {
        "tokens": ["<AK_EXEC_KIND_VERIFY_ABSENT>"],
        "action": "code_execute",
        "content": "test ! -f old_task/test_info.json",
        "target_path": "old_task/test_info.json",
        "verify_polarity": "absent",
    }

    repaired, warnings = repair_line_protocol_with_command_copy_target(
        line_protocol,
        encoder_text=encoder_text,
    )

    assert warnings == []
    assert repaired["content"] == "test ! -f scratch/old_payload.json"
    assert repaired["target_path"] == "scratch/old_payload.json"
    assert repaired["verify_polarity"] == "absent"
    assert repaired["validation_command_repaired"] is True


def test_neural_controller_repairs_single_verify_absent_candidate_for_wrong_absent_path():
    encoder_text = "Validation absent commands: test ! -f scratch/old_payload.json"
    line_protocol = {
        "tokens": ["<AK_EXEC_KIND_VERIFY_ABSENT>"],
        "action": "code_execute",
        "content": "test ! -f api_contract_task/template.http",
        "target_path": "api_contract_task/template.http",
        "verify_polarity": "absent",
    }

    repaired, warnings = repair_line_protocol_with_command_copy_target(
        line_protocol,
        encoder_text=encoder_text,
    )

    assert warnings == []
    assert repaired["content"] == "test ! -f scratch/old_payload.json"
    assert repaired["target_path"] == "scratch/old_payload.json"
    assert repaired["verify_polarity"] == "absent"
    assert repaired["validation_command_repaired"] is True


def test_neural_controller_repairs_verify_absent_wrong_path_to_first_grounded_absent_candidate():
    encoder_text = "Validation absent commands: test ! -f forbidden/a.txt | test ! -f forbidden/b.txt"
    line_protocol = {
        "tokens": ["<AK_EXEC_KIND_VERIFY_ABSENT>"],
        "action": "code_execute",
        "content": "test ! -f docs/wrong.txt",
        "target_path": "docs/wrong.txt",
        "verify_polarity": "absent",
    }

    repaired, warnings = repair_line_protocol_with_command_copy_target(
        line_protocol,
        encoder_text=encoder_text,
    )

    assert warnings == []
    assert repaired["content"] == "test ! -f forbidden/a.txt"
    assert repaired["target_path"] == "forbidden/a.txt"
    assert repaired["verify_polarity"] == "absent"
    assert repaired["validation_command_repaired"] is True


def test_neural_controller_repairs_verify_absent_to_present_when_same_grounded_path_has_no_absent_contract():
    encoder_text = "Validation present commands: test -f gateway/routes.txt"
    line_protocol = {
        "tokens": ["<AK_EXEC_KIND_VERIFY_ABSENT>"],
        "action": "code_execute",
        "content": "test ! -f gateway/routes.txt",
        "target_path": "gateway/routes.txt",
        "verify_polarity": "absent",
    }

    repaired, warnings = repair_line_protocol_with_command_copy_target(
        line_protocol,
        encoder_text=encoder_text,
    )

    assert warnings == []
    assert repaired["content"] == "test -f gateway/routes.txt"
    assert repaired["target_path"] == "gateway/routes.txt"
    assert repaired["verify_polarity"] == "present"
    assert "<AK_EXEC_KIND_VERIFY_PRESENT>" in repaired["tokens"]
    assert "<AK_EXEC_KIND_VERIFY_ABSENT>" not in repaired["tokens"]
    assert repaired["validation_exec_kind_repaired"] is True


def test_neural_controller_does_not_repair_verify_absent_to_present_when_absent_contract_exists():
    encoder_text = "\n".join(
        [
            "Validation present commands: test -f gateway/routes.txt",
            "Validation absent commands: test ! -f gateway/routes.txt",
        ]
    )
    line_protocol = {
        "tokens": ["<AK_EXEC_KIND_VERIFY_ABSENT>"],
        "action": "code_execute",
        "content": "test ! -f gateway/routes.txt",
        "target_path": "gateway/routes.txt",
        "verify_polarity": "absent",
    }

    repaired, warnings = repair_line_protocol_with_command_copy_target(
        line_protocol,
        encoder_text=encoder_text,
    )

    assert warnings == []
    assert repaired["content"] == "test ! -f gateway/routes.txt"
    assert "validation_exec_kind_repaired" not in repaired


def test_neural_controller_repairs_source_probe_to_present_validation_when_no_source_contract_exists():
    encoder_text = "Validation present commands: test -f gateway/routes.txt"
    line_protocol = {
        "tokens": ["<AK_EXEC_KIND_INSPECT_SOURCE>"],
        "action": "code_execute",
        "content": "cat gateway/routes.txt",
        "target_path": "gateway/routes.txt",
    }

    repaired, warnings = repair_line_protocol_with_command_copy_target(
        line_protocol,
        encoder_text=encoder_text,
    )

    assert warnings == []
    assert repaired["content"] == "test -f gateway/routes.txt"
    assert repaired["target_path"] == "gateway/routes.txt"
    assert repaired["verify_polarity"] == "present"
    assert "<AK_EXEC_KIND_VERIFY_PRESENT>" in repaired["tokens"]
    assert "<AK_EXEC_KIND_INSPECT_SOURCE>" not in repaired["tokens"]
    assert repaired["validation_exec_kind_repaired"] is True


def test_neural_controller_repairs_source_probe_to_present_validation_even_with_direct_artifact_probe_candidate():
    encoder_text = "\n".join(
        [
            "Validation present commands: test -f gateway/routes.txt",
            "Source inspection candidate commands: cat gateway/routes.txt",
            "Contract: use the direct task command; do not inspect source_lines or retrieve code context.",
        ]
    )
    line_protocol = {
        "tokens": ["<AK_EXEC_KIND_INSPECT_SOURCE>"],
        "action": "code_execute",
        "content": "cat gateway/routes.txt",
        "target_path": "gateway/routes.txt",
    }

    repaired, warnings = repair_line_protocol_with_command_copy_target(
        line_protocol,
        encoder_text=encoder_text,
    )

    assert warnings == []
    assert repaired["content"] == "test -f gateway/routes.txt"
    assert repaired["target_path"] == "gateway/routes.txt"
    assert repaired["verify_polarity"] == "present"
    assert "<AK_EXEC_KIND_VERIFY_PRESENT>" in repaired["tokens"]
    assert repaired["validation_exec_kind_repaired"] is True


def test_neural_controller_repairs_malformed_localized_edit_to_single_grounded_candidate():
    encoder_text = "Localized edit candidate commands: sed -i '1s#pending#ready#' reports/status.txt"
    line_protocol = {
        "tokens": ["<AK_EXEC_KIND_LOCALIZED_EDIT>"],
        "action": "code_execute",
        "content": "sed -i '1s#pending#ready# reports/reports/status.txt",
        "target_path": "reports/reports/status.txt",
    }

    repaired, warnings = repair_line_protocol_with_command_copy_target(
        line_protocol,
        encoder_text=encoder_text,
    )

    assert warnings == []
    assert repaired["content"] == "sed -i '1s#pending#ready#' reports/status.txt"
    assert repaired["target_path"] == "reports/status.txt"
    assert repaired["localized_edit_candidate_repaired"] is True


def test_neural_controller_does_not_convert_source_probe_when_source_contract_is_authoritative():
    encoder_text = "\n".join(
        [
            "Validation present commands: test -f recovery.txt",
            "Source inspection candidate commands: cat recovery.txt",
        ]
    )
    line_protocol = {
        "tokens": ["<AK_EXEC_KIND_INSPECT_SOURCE>"],
        "action": "code_execute",
        "content": "cat recovery.txt",
        "target_path": "recovery.txt",
    }

    repaired, warnings = repair_line_protocol_with_command_copy_target(
        line_protocol,
        encoder_text=encoder_text,
    )

    assert warnings == []
    assert repaired["content"] == "cat recovery.txt"
    assert "validation_exec_kind_repaired" not in repaired


def test_neural_controller_repairs_inspect_source_to_matching_candidate():
    encoder_text = "\n".join(
        [
            "<AK_CONTEXT>",
            "Source inspection candidate commands: cat recovery.txt | head -n 20 other.txt",
        ]
    )
    line_protocol = {
        "tokens": ["<AK_EXEC_KIND_INSPECT_SOURCE>"],
        "action": "code_execute",
        "content": "test ! -f recovery.txt",
        "target_path": "recovery.txt",
    }

    repaired, warnings = repair_line_protocol_with_command_copy_target(
        line_protocol,
        encoder_text=encoder_text,
    )

    assert warnings == []
    assert repaired["content"] == "cat recovery.txt"
    assert repaired["target_path"] == "recovery.txt"
    assert repaired["source_inspection_candidate_repaired"] is True
    compact = compact_neural_controller_shadow(
        {"ready": True, "line_protocol": repaired},
        selected_action="code_execute",
        selected_content="cat recovery.txt",
    )
    assert compact["source_inspection_candidate_repaired"] is True
    assert compact["content_exact_agreement"] is True


def test_neural_controller_expands_source_inspection_prefix_to_grounded_candidate():
    encoder_text = "Source inspection candidate commands: cat app.py tests/test_app.py"
    line_protocol = {
        "tokens": ["<AK_EXEC_KIND_INSPECT_SOURCE>"],
        "action": "code_execute",
        "content": "cat app.py",
    }

    repaired, warnings = repair_line_protocol_with_command_copy_target(
        line_protocol,
        encoder_text=encoder_text,
    )

    assert warnings == []
    assert repaired["content"] == "cat app.py tests/test_app.py"
    assert repaired["target_path"] == "tests/test_app.py"
    assert repaired["source_inspection_candidate_repaired"] is True


def test_neural_controller_repairs_inspect_source_to_single_grounded_candidate():
    encoder_text = "\n".join(
        [
            "<AK_CONTEXT>",
            "Active subgoal: update workflow path package/module.py",
        ]
    )
    line_protocol = {
        "tokens": ["<AK_EXEC_KIND_INSPECT_SOURCE>"],
        "action": "code_execute",
        "content": "cat source_lines/src/stale/base.py.lines",
        "target_path": "source_lines/src/stale/base.py.lines",
    }

    repaired, warnings = repair_line_protocol_with_command_copy_target(
        line_protocol,
        encoder_text=encoder_text,
    )

    assert warnings == []
    assert repaired["content"] == "cat package/module.py"
    assert repaired["target_path"] == "package/module.py"
    assert repaired["source_inspection_candidate_repaired"] is True


def test_neural_controller_prefers_source_lines_candidate_when_prediction_uses_source_lines():
    encoder_text = "Source inspection candidate commands: cat package/module.py"
    line_protocol = {
        "tokens": ["<AK_EXEC_KIND_INSPECT_SOURCE>"],
        "action": "code_execute",
        "content": "cat source_lines/src/stale/base.py.lines",
        "target_path": "source_lines/src/stale/base.py.lines",
    }

    repaired, warnings = repair_line_protocol_with_command_copy_target(
        line_protocol,
        encoder_text=encoder_text,
    )

    assert warnings == []
    assert repaired["content"] == "cat source_lines/package/module.py.lines"
    assert repaired["target_path"] == "source_lines/package/module.py.lines"
    assert repaired["source_inspection_candidate_repaired"] is True


def test_neural_controller_prefers_source_lines_for_unstable_generated_source_path():
    encoder_text = "Source inspection candidate commands: cat django/db/models/enums.py"
    line_protocol = {
        "tokens": ["<AK_EXEC_KIND_INSPECT_SOURCE>"],
        "action": "code_execute",
        "content": "cat django/db/models/models/models.py",
        "target_path": "django/db/models/models/models.py",
    }

    repaired, warnings = repair_line_protocol_with_command_copy_target(
        line_protocol,
        encoder_text=encoder_text,
    )

    assert warnings == []
    assert repaired["content"] == "cat source_lines/django/db/models/enums.py.lines"
    assert repaired["target_path"] == "source_lines/django/db/models/enums.py.lines"
    assert repaired["source_inspection_candidate_repaired"] is True


def test_neural_controller_repairs_low_conf_validation_to_source_inspection_candidate():
    encoder_text = "Source inspection candidate commands: cat recovery.txt"
    line_protocol = {
        "tokens": [
            "<AK_EXEC_KIND_MATERIALIZE_ARTIFACT>",
            "<AK_ARTIFACT_REPAIR>",
            "<AK_CONF_LOW>",
        ],
        "action": "code_execute",
        "content": "test ! -f recovery.txt",
        "target_path": "recovery.txt",
    }

    repaired, warnings = repair_line_protocol_with_command_copy_target(
        line_protocol,
        encoder_text=encoder_text,
    )

    assert warnings == []
    assert repaired["content"] == "cat recovery.txt"
    assert "<AK_EXEC_KIND_INSPECT_SOURCE>" in repaired["tokens"]
    assert "<AK_EXEC_KIND_MATERIALIZE_ARTIFACT>" not in repaired["tokens"]
    assert repaired["low_conf_artifact_repair_source_inspection_repaired"] is True


def test_controller_dataset_sed_i_is_localized_edit_not_source_inspect():
    tokens = _control_tokens(
        {
            "decision_source": "artifact_repair",
            "verification": {"passed": True},
        },
        "code_execute",
        "sed -i '1s#old#new#' project/plan.md",
        "artifact_contract_success",
    ).split()

    assert "<AK_EXEC_KIND_LOCALIZED_EDIT>" in tokens
    assert "<AK_SOURCE_INSPECT>" not in tokens
    assert "<AK_ACTION_SPACE_RETRIEVAL>" not in tokens


def test_direct_command_dataset_split_does_not_hold_out_entire_tasks():
    rows = _direct_command_rows(repeat=10, eval_ratio=0.2)
    by_source: dict[str, set[str]] = {}
    for row in rows:
        source_id = str(row.get("source_id", ""))
        if not source_id.startswith("task_bank:"):
            continue
        by_source.setdefault(source_id, set()).add(str(row.get("split", "train")))

    assert by_source
    assert all("train" in splits for splits in by_source.values())
    assert any("eval" in splits for splits in by_source.values())


def test_direct_command_copy_pointer_is_reserved_for_copy_risk_commands():
    assert _copy_pointer_candidate("printf '42\\n' > result.txt") is False
    assert _copy_pointer_candidate(
        "mkdir -p checks manifests reports && "
        "printf '{\"channel\":\"stable\"}\\n' > manifests/release.json && "
        "printf '{\"status\":\"ok\"}\\n' > reports/status.json && "
        "printf 'audit checks green\\n' > checks/status.txt"
    ) is True


def test_neural_controller_advisory_payload_ready(tmp_path):
    manifest = load_neural_controller_manifest(_write_controller_manifest(tmp_path))

    advisory = build_neural_controller_advisory(
        manifest=manifest,
        mode="advisory",
        guarded_fallback_families=("inspect_source",),
        guarded_candidate_manifest_path="candidate.json",
        guarded_selector_policy="candidate_contract_improves",
    )
    payload = advisory.to_payload()

    assert payload["enabled"] is True
    assert payload["ready"] is True
    assert payload["mode"] == "advisory"
    assert "<AK_ACTION_SPACE_CODE>" in payload["action_space_tokens"]
    assert "verification_need" in payload["policy_heads"]
    assert payload["guarded_fallback_families"] == ["inspect_source"]
    assert payload["guarded_candidate_manifest_path"] == "candidate.json"
    assert payload["guarded_selector_policy"] == "candidate_contract_improves"


def test_guarded_neural_controller_source_falls_back_by_candidate_family():
    baseline = {
        "tokens": ["<AK_EXEC_KIND_INSPECT_SOURCE>"],
        "action": "code_execute",
        "content": "cat recovery.txt",
    }
    candidate = {
        "tokens": ["<AK_EXEC_KIND_INSPECT_SOURCE>"],
        "action": "code_execute",
        "content": "test ! -f recovery.txt",
    }

    selected = guarded_neural_controller_source(
        candidate_line_protocol=candidate,
        baseline_line_protocol=baseline,
        fallback_families=("inspect_source",),
    )

    assert neural_controller_exec_kind_family(candidate["tokens"]) == "inspect_source"
    assert selected["source"] == "baseline"
    assert selected["line_protocol"]["content"] == "cat recovery.txt"


def test_verified_neural_controller_shadow_selector_requires_contract_improvement():
    baseline = {
        "example_id": "row",
        "artifact_failure_mode": "missing_expected_file",
        "content_exact_agreement": False,
    }
    weak_candidate = {
        "example_id": "row",
        "artifact_failure_mode": "missing_expected_file",
        "content_exact_agreement": False,
    }
    contract_candidate = {
        "example_id": "row",
        "artifact_failure_mode": "artifact_contract_success",
        "content_exact_agreement": False,
    }

    selected = select_verified_neural_controller_shadow(
        baseline_label="v64",
        baseline_shadow=baseline,
        candidate_shadows=[("weak", weak_candidate), ("contract", contract_candidate)],
    )

    assert selected["source"] == "contract"
    assert selected["accepted_candidate_sources"] == ["contract"]
    assert selected["shadow"]["rowwise_selector_source"] == "contract"

    protected = select_verified_neural_controller_shadow(
        baseline_label="v64",
        baseline_shadow={**baseline, "artifact_failure_mode": "artifact_contract_success"},
        candidate_shadows=[("contract", contract_candidate)],
    )

    assert protected["source"] == "v64"
    assert protected["accepted_candidate_sources"] == []


def test_attach_neural_controller_verified_contract_metadata_adds_runtime_contract_status():
    metadata = {
        "neural_controller_shadow": {
            "example_id": "row",
            "content_exact_agreement": False,
        }
    }

    updated = attach_neural_controller_verified_contract_metadata(
        metadata,
        verification={"passed": True, "reasons": ["verification passed"], "failure_codes": []},
    )

    shadow = updated["neural_controller_shadow"]
    assert shadow["runtime_artifact_failure_mode"] == "artifact_contract_success"
    assert shadow["runtime_contract_success"] is True
    assert "runtime_artifact_failure_mode" not in metadata["neural_controller_shadow"]

    guarded_metadata = {
        "neural_controller_shadow": {
            "ready": True,
            "guarded_selected_source": "candidate",
            "guarded_selector_policy": "candidate_contract_improves",
        }
    }
    guarded = attach_neural_controller_verified_contract_metadata(
        guarded_metadata,
        verification={"passed": True, "reasons": ["verification passed"], "failure_codes": []},
    )

    guarded_shadow = guarded["neural_controller_shadow"]
    assert guarded_shadow["rowwise_selector_source"] == "candidate"
    assert guarded_shadow["rowwise_selector_policy"] == "candidate_contract_improves"
    assert guarded_shadow["runtime_selector_selected_source"] == "candidate"
    assert guarded_shadow["runtime_selector_selected_contract_success"] is True

    failed = attach_neural_controller_verified_contract_metadata(
        metadata,
        verification={
            "passed": False,
            "reasons": ["missing expected file: patch.diff"],
            "failure_codes": ["missing_expected_file"],
        },
    )

    assert failed["neural_controller_shadow"]["runtime_artifact_failure_mode"] == "missing_expected_file"
    assert failed["neural_controller_shadow"]["runtime_contract_success"] is False


def test_runtime_contract_metrics_count_verified_neural_controller_shadow_signal():
    documents = [
        {
            "steps": [
                {
                    "proposal_metadata": {
                        "neural_controller_shadow": {
                            "runtime_artifact_failure_mode": "artifact_contract_success",
                            "runtime_contract_success": True,
                            "rowwise_selector_source": "v64_guarded",
                            "rowwise_selector_policy": "candidate_contract_improves",
                            "guarded_baseline_prediction": {"content_preview": "cat a.txt"},
                            "guarded_candidate_prediction": {"content_preview": "cat b.txt"},
                        },
                        "neural_controller_guarded_candidate_dry_run": {
                            "attempted": True,
                            "skipped": False,
                            "candidate_verification_passed": True,
                            "candidate_artifact_failure_mode": "artifact_contract_success",
                        },
                        "neural_controller_guarded_selected_dry_run": {
                            "attempted": True,
                            "selected_verification_passed": False,
                        },
                        "neural_controller_guarded_dry_run_switch": {
                            "applied": True,
                        },
                    },
                    "verification": {"passed": True},
                },
                {
                    "proposal_metadata": {
                        "neural_controller_shadow": {
                            "runtime_artifact_failure_mode": "missing_expected_file",
                            "runtime_contract_success": False,
                            "rowwise_selector_source": "v28_replay",
                            "rowwise_selector_policy": "candidate_contract_improves",
                        }
                    },
                    "verification": {"passed": False},
                },
                {
                    "proposal_metadata": {"neural_controller_shadow": {"ready": True}},
                    "verification": {"passed": False},
                },
            ]
        }
    ]

    summary = summarize_runtime_contract_metrics(documents)

    assert summary["shadow_steps"] == 3
    assert summary["runtime_contract_steps"] == 2
    assert summary["runtime_contract_task_count"] == 0
    assert summary["runtime_contract_success_steps"] == 1
    assert summary["runtime_contract_coverage_rate"] == 0.666667
    assert summary["guarded_comparison_steps"] == 1
    assert summary["guarded_baseline_observed_steps"] == 1
    assert summary["guarded_candidate_observed_steps"] == 1
    assert summary["guarded_candidate_dry_run_attempts"] == 1
    assert summary["guarded_candidate_dry_run_successes"] == 1
    assert summary["guarded_candidate_dry_run_success_rate"] == 1.0
    assert summary["guarded_candidate_dry_run_mode_counts"] == {"artifact_contract_success": 1}
    assert summary["guarded_selected_dry_run_attempts"] == 1
    assert summary["guarded_selected_dry_run_successes"] == 0
    assert summary["guarded_selected_dry_run_success_rate"] == 0.0
    assert summary["guarded_dry_run_switches_applied"] == 1
    assert summary["runtime_artifact_failure_mode_counts"] == {
        "artifact_contract_success": 1,
        "missing_expected_file": 1,
    }
    assert summary["rowwise_selector_source_counts"] == {"v28_replay": 1, "v64_guarded": 1}
    assert summary["selector_signal_ready"] is True


def test_loop_persists_runtime_contract_status_on_neural_controller_shadow(tmp_path):
    class ShadowPolicy(Policy):
        def decide(self, state):
            del state
            return ActionDecision(
                thought="write expected artifact",
                action="code_execute",
                content="mkdir -p out && printf ok > out/result.txt",
                proposal_metadata={
                    "neural_controller_shadow": {
                        "ready": True,
                        "rowwise_selector_source": "v64_guarded",
                        "rowwise_selector_policy": "candidate_contract_improves",
                    }
                },
            )

    config = KernelConfig(
        provider="mock",
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        use_world_model=False,
        use_universe_model=False,
        use_planner=False,
        use_graph_memory=False,
        use_tolbert_context=False,
        use_retrieval_proposals=False,
        use_role_specialization=False,
        max_steps=1,
    )
    task = TaskSpec(
        task_id="neural_runtime_contract_bridge",
        prompt="write expected artifact",
        workspace_subdir="neural_runtime_contract_bridge",
        expected_file_contents={"out/result.txt": "ok"},
        max_steps=1,
    )

    episode = AgentKernel(config=config, policy=ShadowPolicy()).run_task(task)

    shadow = episode.steps[0].proposal_metadata["neural_controller_shadow"]
    assert episode.success is True
    assert shadow["runtime_artifact_failure_mode"] == "artifact_contract_success"
    assert shadow["runtime_contract_success"] is True


def test_loop_guarded_candidate_dry_run_verifies_candidate_in_isolated_workspace(tmp_path):
    class ShadowPolicy(Policy):
        def decide(self, state):
            del state
            return ActionDecision(
                thought="selected command is weak; guarded candidate is contract satisfying",
                action="code_execute",
                content="true",
                proposal_metadata={
                    "neural_controller_shadow": {
                        "ready": True,
                        "guarded_selected_source": "baseline",
                        "guarded_selector_policy": "candidate_contract_improves",
                        "guarded_candidate_prediction": {
                            "action": "code_execute",
                            "content": "mkdir -p out && printf ok > out/result.txt",
                            "content_preview": "mkdir -p out && printf ok > out/result.txt",
                            "control_tokens": ["<AK_EXEC_KIND_MATERIALIZE_ARTIFACT>"],
                            "exec_kind_family": "materialize_artifact",
                        },
                        "guarded_baseline_prediction": {
                            "action": "code_execute",
                            "content": "true",
                            "content_preview": "true",
                            "control_tokens": ["<AK_EXEC_KIND_RUN_CHECK>"],
                            "exec_kind_family": "run_check",
                        },
                    }
                },
            )

    config = KernelConfig(
        provider="mock",
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        use_world_model=False,
        use_universe_model=False,
        use_planner=False,
        use_graph_memory=False,
        use_tolbert_context=False,
        use_retrieval_proposals=False,
        use_role_specialization=False,
        neural_controller_guarded_dry_run_compare=True,
        max_steps=1,
    )
    task = TaskSpec(
        task_id="neural_guarded_candidate_dry_run",
        prompt="write expected artifact",
        workspace_subdir="neural_guarded_candidate_dry_run",
        expected_file_contents={"out/result.txt": "ok"},
        max_steps=1,
    )

    episode = AgentKernel(config=config, policy=ShadowPolicy()).run_task(task)

    metadata = episode.steps[0].proposal_metadata
    dry_run = metadata["neural_controller_guarded_candidate_dry_run"]
    assert episode.success is False
    assert dry_run["attempted"] is True
    assert dry_run["candidate_verification_passed"] is True
    assert dry_run["candidate_artifact_failure_mode"] == "artifact_contract_success"
    assert not (config.workspace_root / task.workspace_subdir / "out/result.txt").exists()


def test_loop_guarded_candidate_dry_run_switches_when_candidate_passes_selected_fails(tmp_path):
    class ShadowPolicy(Policy):
        def decide(self, state):
            del state
            return ActionDecision(
                thought="switch to verified guarded candidate",
                action="code_execute",
                content="true",
                proposal_metadata={
                    "neural_controller_shadow": {
                        "ready": True,
                        "guarded_selected_source": "baseline",
                        "guarded_selector_policy": "candidate_contract_improves",
                        "guarded_candidate_prediction": {
                            "action": "code_execute",
                            "content": "mkdir -p out && printf ok > out/result.txt",
                            "content_preview": "mkdir -p out && printf ok > out/result.txt",
                            "control_tokens": ["<AK_EXEC_KIND_MATERIALIZE_ARTIFACT>"],
                            "exec_kind_family": "materialize_artifact",
                        },
                        "guarded_baseline_prediction": {
                            "action": "code_execute",
                            "content": "true",
                            "content_preview": "true",
                            "control_tokens": ["<AK_EXEC_KIND_RUN_CHECK>"],
                            "exec_kind_family": "run_check",
                        },
                    }
                },
            )

    config = KernelConfig(
        provider="mock",
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        use_world_model=False,
        use_universe_model=False,
        use_planner=False,
        use_graph_memory=False,
        use_tolbert_context=False,
        use_retrieval_proposals=False,
        use_role_specialization=False,
        neural_controller_guarded_dry_run_compare=True,
        neural_controller_guarded_dry_run_switch=True,
        max_steps=1,
    )
    task = TaskSpec(
        task_id="neural_guarded_candidate_dry_run_switch",
        prompt="write expected artifact",
        workspace_subdir="neural_guarded_candidate_dry_run_switch",
        expected_file_contents={"out/result.txt": "ok"},
        max_steps=1,
    )

    episode = AgentKernel(config=config, policy=ShadowPolicy()).run_task(task)

    metadata = episode.steps[0].proposal_metadata
    shadow = metadata["neural_controller_shadow"]
    assert episode.success is True
    assert metadata["neural_controller_guarded_candidate_dry_run"]["candidate_verification_passed"] is True
    assert metadata["neural_controller_guarded_selected_dry_run"]["selected_verification_passed"] is False
    assert metadata["neural_controller_guarded_dry_run_switch"]["applied"] is True
    assert shadow["rowwise_selector_source"] == "candidate_dry_run"
    assert shadow["runtime_contract_success"] is True
    assert (config.workspace_root / task.workspace_subdir / "out/result.txt").read_text(encoding="utf-8") == "ok"


def test_kernel_config_allows_guarded_neural_controller_without_primary_authority(tmp_path):
    gate_path = tmp_path / "selector_gate.json"
    gate_path.write_text(
        json.dumps(
            {
                "report_kind": "neural_controller_selector_activation_gate",
                "production_guarded_selector_activation_ready": True,
                "primary_authority_ready": False,
            }
        ),
        encoding="utf-8",
    )
    config = KernelConfig(
        provider="mock",
        use_neural_controller=True,
        neural_controller_mode="guarded",
        neural_controller_guarded_fallback_families=("inspect_source",),
        neural_controller_guarded_selector_policy="candidate_contract_improves",
        neural_controller_selector_activation_gate_path=gate_path,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )

    config.validate()


def test_kernel_config_rejects_contract_selector_without_activation_gate(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_neural_controller=True,
        neural_controller_mode="guarded",
        neural_controller_guarded_selector_policy="candidate_contract_improves",
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )

    with pytest.raises(ValueError, match="production-ready selector activation gate"):
        config.validate()


def test_kernel_config_rejects_unknown_neural_controller_guarded_selector_policy(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_neural_controller=True,
        neural_controller_mode="guarded",
        neural_controller_guarded_selector_policy="exact_oracle",
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )

    with pytest.raises(ValueError, match="unsupported neural_controller_guarded_selector_policy"):
        config.validate()


def test_guarded_neural_controller_advisory_derives_runtime_from_report(tmp_path):
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    baseline_dir.mkdir()
    candidate_dir.mkdir()
    baseline_manifest = _write_controller_manifest(baseline_dir)
    candidate_manifest = _write_controller_manifest(candidate_dir)
    baseline_report = tmp_path / "baseline_report.json"
    candidate_report = tmp_path / "candidate_report.json"
    baseline_report.write_text(json.dumps({"manifest_path": str(baseline_manifest)}), encoding="utf-8")
    candidate_report.write_text(json.dumps({"manifest_path": str(candidate_manifest)}), encoding="utf-8")
    guarded_report = tmp_path / "guarded_report.json"
    guarded_report.write_text(
        json.dumps(
            {
                "report_kind": "neural_controller_guarded_composition",
                "baseline_report_path": str(baseline_report),
                "candidate_report_path": str(candidate_report),
                "fallback_families": ["inspect_source"],
            }
        ),
        encoding="utf-8",
    )
    config = KernelConfig(
        provider="mock",
        use_neural_controller=True,
        neural_controller_mode="guarded",
        neural_controller_guarded_report_path=guarded_report,
        neural_controller_guarded_selector_policy="candidate_contract_improves",
    )

    advisory = PolicyRuntimeSupport(config=config, repo_root=tmp_path).neural_controller_advisory().to_payload()

    assert advisory["mode"] == "guarded"
    assert advisory["manifest_path"] == str(baseline_manifest)
    assert advisory["guarded_candidate_manifest_path"] == str(candidate_manifest)
    assert advisory["guarded_fallback_families"] == ["inspect_source"]
    assert advisory["guarded_selector_policy"] == "candidate_contract_improves"


def test_kernel_config_rejects_unretained_neural_controller_primary(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_neural_controller=True,
        neural_controller_mode="primary",
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )

    with pytest.raises(ValueError, match="retained promotion gate"):
        config.validate()


def test_kernel_config_allows_primary_with_retained_neural_controller_gate(tmp_path):
    gate_path = tmp_path / "neural_controller_retained_gate.json"
    gate_path.write_text(
        json.dumps(
            {
                "report_kind": "neural_controller_retained_promotion_gate",
                "primary_authority_ready": True,
            }
        ),
        encoding="utf-8",
    )
    config = KernelConfig(
        provider="mock",
        use_neural_controller=True,
        neural_controller_mode="primary",
        neural_controller_retained_promotion_gate_path=gate_path,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )

    config.validate()


def test_retained_promotion_gate_reports_contract_ready_without_primary(tmp_path):
    guarded_path = tmp_path / "guarded.json"
    guarded_path.write_text(
        json.dumps(
            {
                "family_metrics": {
                    "materialize_artifact": {
                        "content_exact_rate": 0.4,
                        "contract_content_rate": 0.9,
                    },
                    "localized_edit": {
                        "content_exact_rate": 0.0,
                        "contract_content_rate": 0.0,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    flip_path = tmp_path / "flip.json"
    flip_path.write_text(
        json.dumps(
            {
                "guarded": {
                    "path": str(guarded_path),
                    "content_exact_rate": 0.74,
                    "contract_content_rate": 0.82,
                }
            }
        ),
        encoding="utf-8",
    )

    report = build_retained_promotion_gate(
        argparse.Namespace(
            flip_report=flip_path,
            output=tmp_path / "gate.json",
            min_content_rate=0.8,
            retained_evidence_ready=False,
            allow_contract_content_primary=False,
        )
    )

    assert report["contract_content_ready"] is True
    assert report["strict_content_ready"] is False
    assert report["primary_authority_ready"] is False
    assert report["family_authority_profile"]["ready"] is False
    assert "localized_edit_content_exact_rate_below_gate" in report["family_authority_profile"]["blockers"]
    assert report["family_authority_profile"]["families"]["materialize_artifact"]["ready"] is True
    assert "contract_content_ready_but_not_authorized_for_primary" in report["blockers"]


def test_policy_adds_neural_controller_advisory_to_payload(tmp_path):
    class CapturingClient(MockLLMClient):
        def __init__(self) -> None:
            super().__init__()
            self.last_payload = None
            self.last_decision_prompt = ""

        def create_decision(self, *, system_prompt, decision_prompt, state_payload):
            del system_prompt
            self.last_decision_prompt = decision_prompt
            self.last_payload = state_payload
            return {
                "thought": "write expected file",
                "action": "code_execute",
                "content": "printf 'hello agent kernel\\n' > hello.txt",
                "done": False,
            }

    client = CapturingClient()
    config = KernelConfig(
        provider="mock",
        asi_coding_require_live_llm=True,
        use_neural_controller=True,
        neural_controller_mode="advisory",
        neural_controller_manifest_path=_write_controller_manifest(tmp_path),
    )
    policy = LLMDecisionPolicy(client, config=config)

    decision = policy.decide(AgentState(task=TaskBank().get("hello_task")))

    assert decision.action == "code_execute"
    assert client.last_payload["neural_controller_advisory"]["ready"] is True
    assert "full_kernel_action_space_policy" in client.last_payload["neural_controller_advisory"]["surfaces"]
    assert "Neural controller advisory:" in client.last_decision_prompt
    assert decision.proposal_metadata["neural_controller_advisory"]["ready"] is True


def test_policy_adds_optional_neural_controller_shadow_prediction_to_payload(tmp_path, monkeypatch):
    class CapturingClient(MockLLMClient):
        def __init__(self) -> None:
            super().__init__()
            self.last_payload = None
            self.last_decision_prompt = ""

        def create_decision(self, *, system_prompt, decision_prompt, state_payload):
            del system_prompt
            self.last_decision_prompt = decision_prompt
            self.last_payload = state_payload
            return {
                "thought": "write expected file",
                "action": "code_execute",
                "content": "printf 'hello agent kernel\\n' > hello.txt",
                "done": False,
            }

    def fake_generate_neural_controller_text(**kwargs):
        assert "encoder_text" in kwargs
        return {
            "generated_text": "<AK_ACTION_SPACE_ARTIFACT> <AK_PATCH_BUILD> <AK_VERIFY>\n"
            "Action: code_execute\n"
            "Content: python scripts/patch_builder.py --path pkg/core.py",
            "generated_token_count": 14,
            "policy_heads": {"query_confidence": 0.7, "needs_verification": 0.9},
        }

    monkeypatch.setattr(
        "agent_kernel.extensions.policy_runtime_support.generate_neural_controller_text",
        fake_generate_neural_controller_text,
    )
    client = CapturingClient()
    config = KernelConfig(
        provider="mock",
        asi_coding_require_live_llm=True,
        use_neural_controller=True,
        neural_controller_mode="shadow",
        neural_controller_shadow_generate=True,
        neural_controller_manifest_path=_write_controller_manifest(tmp_path),
    )
    policy = LLMDecisionPolicy(client, config=config)

    decision = policy.decide(AgentState(task=TaskBank().get("hello_task")))

    shadow = client.last_payload["neural_controller_shadow"]
    assert shadow["ready"] is True
    assert shadow["generated_token_count"] == 14
    assert shadow["policy_heads"]["needs_verification"] == 0.9
    assert shadow["line_protocol"]["action"] == "code_execute"
    assert "<AK_PATCH_BUILD>" in shadow["line_protocol"]["tokens"]
    assert decision.proposal_metadata["neural_controller_advisory"]["shadow_generated"] is True
    compact_shadow = decision.proposal_metadata["neural_controller_shadow"]
    assert compact_shadow["predicted_action"] == "code_execute"
    assert compact_shadow["action_agreement"] is True
    assert compact_shadow["policy_heads"]["needs_verification"] == 0.9
    assert decision.proposal_metadata["neural_controller_advisory"]["shadow_attempted"] is True


def test_policy_skips_neural_controller_shadow_during_artifact_diagnostic_repair(tmp_path, monkeypatch):
    class CapturingClient(MockLLMClient):
        def __init__(self) -> None:
            super().__init__()
            self.last_payload = None

        def create_decision(self, *, system_prompt, decision_prompt, state_payload):
            del system_prompt, decision_prompt
            self.last_payload = state_payload
            return {
                "thought": "write patch",
                "action": "code_execute",
                "content": "patch_builder --path pkg/module.py --replace-line 1 --with 'value = 2' > patch.diff",
                "done": False,
            }

    def fail_generate_neural_controller_text(**kwargs):
        del kwargs
        raise AssertionError("artifact diagnostic repair should not run neural shadow generation")

    monkeypatch.setattr(
        "agent_kernel.extensions.policy_runtime_support.generate_neural_controller_text",
        fail_generate_neural_controller_text,
    )
    client = CapturingClient()
    config = KernelConfig(
        provider="mock",
        asi_coding_require_live_llm=True,
        use_neural_controller=True,
        neural_controller_mode="shadow",
        neural_controller_shadow_generate=True,
        neural_controller_manifest_path=_write_controller_manifest(tmp_path),
    )
    policy = LLMDecisionPolicy(client, config=config)
    state = AgentState(
        task=TaskSpec(
            task_id="neural_shadow_skip_artifact_diagnostic_task",
            prompt="write patch.diff",
            workspace_subdir="neural_shadow_skip_artifact_diagnostic_task",
            expected_files=["patch.diff"],
            metadata={
                "artifact_repair_contract": {
                    "artifact_path": "patch.diff",
                    "builder_commands": ["patch_builder"],
                },
                "setup_file_contents": {
                    "source_context/pkg/module.py": "value = 1\n",
                    "source_lines/pkg/module.py.lines": "1: value = 1\n",
                },
            },
        )
    )
    state.history.append(
        StepRecord(
            index=1,
            thought="read compact context",
            action="code_execute",
            content="python3 - <<'AGENT_KERNEL_ARTIFACT_CONTEXT'\n# context\npass\nAGENT_KERNEL_ARTIFACT_CONTEXT",
            selected_skill_id=None,
            command_result={"exit_code": 0},
            verification={"passed": False, "reasons": ["missing expected file: patch.diff"]},
            decision_source="artifact_policy_timeout_context_read",
        )
    )

    decision = policy.decide(state)

    assert decision.action == "code_execute"
    assert "neural_controller_advisory" in client.last_payload
    assert "neural_controller_shadow" not in client.last_payload


def test_policy_guarded_neural_controller_shadow_selects_fallback_payload(tmp_path, monkeypatch):
    class CapturingClient(MockLLMClient):
        def __init__(self) -> None:
            super().__init__()
            self.last_payload = None
            self.last_decision_prompt = ""

        def create_decision(self, *, system_prompt, decision_prompt, state_payload):
            del system_prompt
            self.last_decision_prompt = decision_prompt
            self.last_payload = state_payload
            return {
                "thought": "write expected file",
                "action": "code_execute",
                "content": "cat recovery.txt",
                "done": False,
            }

    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    baseline_dir.mkdir()
    candidate_dir.mkdir()
    baseline_manifest = _write_controller_manifest(baseline_dir)
    candidate_manifest = _write_controller_manifest(candidate_dir)

    def fake_generate_neural_controller_text(**kwargs):
        manifest_path = str(kwargs["manifest_path"])
        if "candidate" in manifest_path:
            return {
                "generated_text": "<AK_EXEC_KIND_INSPECT_SOURCE>\n"
                "Action: code_execute\n"
                "Content: test ! -f recovery.txt",
                "generated_token_count": 9,
            }
        return {
            "generated_text": "<AK_EXEC_KIND_INSPECT_SOURCE>\n"
            "Action: code_execute\n"
            "Content: cat recovery.txt",
            "generated_token_count": 7,
        }

    monkeypatch.setattr(
        "agent_kernel.extensions.policy_runtime_support.generate_neural_controller_text",
        fake_generate_neural_controller_text,
    )
    client = CapturingClient()
    config = KernelConfig(
        provider="mock",
        asi_coding_require_live_llm=True,
        use_neural_controller=True,
        neural_controller_mode="guarded",
        neural_controller_shadow_generate=True,
        neural_controller_manifest_path=baseline_manifest,
        neural_controller_guarded_candidate_manifest_path=candidate_manifest,
        neural_controller_guarded_fallback_families=("inspect_source",),
        neural_controller_guarded_selector_policy="candidate_contract_improves",
    )
    policy = LLMDecisionPolicy(client, config=config)

    decision = policy.decide(AgentState(task=TaskBank().get("hello_task")))

    shadow = client.last_payload["neural_controller_shadow"]
    assert shadow["guarded"]["selected_source"] == "baseline"
    assert shadow["guarded"]["selector_policy"] == "candidate_contract_improves"
    assert shadow["line_protocol"]["content"] == "cat recovery.txt"
    assert shadow["guarded"]["baseline_line_protocol"]["content"] == "cat recovery.txt"
    assert shadow["guarded"]["candidate_raw_line_protocol"]["content"] == "test ! -f recovery.txt"
    assert shadow["guarded"]["candidate_line_protocol"]["content"] == "cat hello.txt"
    assert "Guarded mode is active" in client.last_decision_prompt
    advisory = decision.proposal_metadata["neural_controller_advisory"]
    assert advisory["guarded_selected_source"] == "baseline"
    assert advisory["guarded_candidate_family"] == "inspect_source"
    assert advisory["guarded_fallback_families"] == ["inspect_source"]
    assert advisory["guarded_selector_policy"] == "candidate_contract_improves"
    compact_shadow = decision.proposal_metadata["neural_controller_shadow"]
    assert compact_shadow["predicted_content_preview"] == "cat recovery.txt"
    assert compact_shadow["guarded_selected_source"] == "baseline"
    assert compact_shadow["guarded_selector_policy"] == "candidate_contract_improves"
    assert compact_shadow["guarded_baseline_prediction"]["content_preview"] == "cat recovery.txt"
    assert compact_shadow["guarded_candidate_prediction"]["content_preview"] == "cat hello.txt"
    assert compact_shadow["guarded_candidate_prediction"]["exec_kind_family"] == "inspect_source"


def test_episode_summary_and_learning_candidates_include_neural_controller_shadow_metrics(tmp_path):
    step = StepRecord(
        index=1,
        thought="use selected command",
        action="code_execute",
        content="printf 'hello agent kernel\\n' > hello.txt",
        selected_skill_id=None,
        command_result={
            "command": "printf 'hello agent kernel\\n' > hello.txt",
            "exit_code": 0,
            "stdout": "",
            "stderr": "",
            "timed_out": False,
        },
        verification={"passed": True, "reasons": ["verification passed"]},
        proposal_metadata={
            "neural_controller_shadow": {
                "ready": True,
                "predicted_action": "code_execute",
                "predicted_content_preview": "printf 'hello agent kernel\\n' > hello.txt",
                "control_tokens": ["<AK_ACTION_SPACE_CODE>", "<AK_VERIFY>"],
                "policy_heads": {"needs_verification": 0.9},
                "action_agreement": True,
                "content_exact_agreement": True,
            }
        },
    )
    episode = EpisodeRecord(
        task_id="neural_shadow_summary_task",
        prompt="create hello.txt",
        workspace=str(tmp_path),
        success=True,
        steps=[step],
        task_metadata={"benchmark_family": "unit"},
    )

    summary = build_episode_summary(episode)
    candidates = compile_episode_learning_candidates(
        episode,
        episode_storage={"root": str(tmp_path), "relative_path": "neural_shadow_summary_task.json"},
    )

    assert summary["neural_controller_shadow"]["shadow_steps"] == 1
    assert summary["neural_controller_shadow"]["verified_action_agreement_steps"] == 1
    assert summary["neural_controller_shadow"]["control_token_counts"]["<AK_VERIFY>"] == 1
    assert candidates[0]["neural_controller_shadow_steps"] == 1
    assert candidates[0]["neural_controller_verified_action_agreement_steps"] == 1


def test_neural_controller_shadow_summary_supports_persisted_dict_steps():
    steps = [
        {
            "proposal_metadata": {
                "neural_controller_shadow": {
                    "ready": True,
                    "action_agreement": True,
                    "content_exact_agreement": False,
                    "control_tokens": ["<AK_RETRIEVE>", "<AK_VERIFY>"],
                }
            },
            "verification": {"passed": True},
        },
        {
            "proposal_metadata": {
                "neural_controller_shadow": {
                    "ready": False,
                    "action_agreement": False,
                    "error": "No module named torch.nn",
                    "warnings": ["runtime_dependency_missing"],
                    "control_tokens": ["<AK_SAFE_STOP>"],
                }
            },
            "verification": {"passed": False},
        },
    ]

    summary = summarize_neural_controller_shadow_steps(steps)

    assert summary["shadow_steps"] == 2
    assert summary["ready_steps"] == 1
    assert summary["verified_ready_steps"] == 1
    assert summary["verified_action_agreement_steps"] == 1
    assert summary["error_steps"] == 1
    assert summary["warning_steps"] == 1
    assert summary["control_token_counts"]["<AK_RETRIEVE>"] == 1
    assert summary["control_token_counts"]["<AK_SAFE_STOP>"] == 1


def test_compact_neural_controller_shadow_preserves_error_diagnostics():
    compact = compact_neural_controller_shadow(
        {
            "ready": False,
            "error": "No module named torch.nn",
            "warnings": ["runtime_dependency_missing"],
        },
        selected_action="code_execute",
    )

    assert compact["ready"] is False
    assert compact["error"] == "No module named torch.nn"
    assert compact["warnings"] == ["runtime_dependency_missing"]
    assert compact["action_agreement"] is False


def test_compact_neural_controller_shadow_preserves_scalar_control_diagnostics():
    compact = compact_neural_controller_shadow(
        {
            "ready": True,
            "line_protocol": {"action": "code_execute", "content": "printf 'ok\\n' > out.txt"},
            "scalar_control": {
                "encoder": {
                    "source_mean": 0.42,
                    "potential_mean": 0.12,
                    "update_norm": 0.0,
                }
            },
        },
        selected_action="code_execute",
        selected_content="printf 'ok\\n' > out.txt",
    )

    assert compact["scalar_control"]["encoder"]["source_mean"] == 0.42
    assert compact["scalar_control"]["encoder"]["update_norm"] == 0.0


def test_compact_neural_controller_shadow_marks_content_comparison_evidence():
    compact = compact_neural_controller_shadow(
        {
            "ready": True,
            "line_protocol": {
                "action": "code_execute",
                "content": "printf 'ok\\n' > status.txt",
            },
        },
        selected_action="code_execute",
        selected_content="printf 'ok\\n' > status.txt",
    )

    assert compact["selected_content_preview"] == "printf 'ok\\n' > status.txt"
    assert compact["predicted_content"] == "printf 'ok\\n' > status.txt"
    assert compact["selected_content"] == "printf 'ok\\n' > status.txt"
    assert compact["content_comparison_evaluated"] is True
    assert compact["content_exact_agreement"] is True


def test_neural_controller_expands_source_inspection_candidate_pointer():
    encoder_text = "\n".join(
        [
            "Source inspection candidate commands: cat wrong.py | cat correct.py",
            "Source inspection candidate 1: cat wrong.py",
            "Source inspection candidate 2: cat correct.py",
        ]
    )
    repaired, warnings = repair_line_protocol_with_command_copy_target(
        {
            "tokens": ["<AK_EXEC_KIND_INSPECT_SOURCE>"],
            "action": "code_execute",
            "content": "<AK_COPY_SOURCE_INSPECT_CANDIDATE_3>",
        },
        encoder_text=encoder_text,
    )

    assert warnings == []
    assert repaired["content"] == "cat correct.py"
    assert repaired["target_path"] == "correct.py"
    assert repaired["source_inspection_candidate_expanded"] is True


def test_neural_controller_source_pointer_control_token_overrides_freeform_content():
    encoder_text = "\n".join(
        [
            "Source inspection candidate commands: cat wrong.py | cat correct.py",
            "Source inspection candidate 1: cat wrong.py",
            "Source inspection candidate 2: cat correct.py",
        ]
    )
    repaired, warnings = repair_line_protocol_with_command_copy_target(
        {
            "tokens": [
                "<AK_EXEC_KIND_MATERIALIZE_ARTIFACT>",
                "<AK_COPY_SOURCE_INSPECT_CANDIDATE_3>",
            ],
            "action": "code_execute",
            "content": "printf 'file recovery complete\\n' > recovery.txt",
        },
        encoder_text=encoder_text,
    )

    assert warnings == []
    assert repaired["content"] == "cat correct.py"
    assert repaired["target_path"] == "correct.py"
    assert repaired["tokens"][0] == "<AK_EXEC_KIND_INSPECT_SOURCE>"
    assert repaired["source_inspection_candidate_expanded"] is True


def test_neural_controller_expands_validation_candidate_pointer():
    encoder_text = "\n".join(
        [
            "Validation present commands: test -f wrong.txt | test -f correct.txt",
            "Validation present candidate 1: test -f wrong.txt",
            "Validation present candidate 2: test -f correct.txt",
        ]
    )
    repaired, warnings = repair_line_protocol_with_command_copy_target(
        {
            "tokens": ["<AK_EXEC_KIND_VERIFY_PRESENT>"],
            "action": "code_execute",
            "content": "<AK_COPY_VALIDATE_PRESENT_CANDIDATE_2>",
        },
        encoder_text=encoder_text,
    )

    assert warnings == []
    assert repaired["content"] == "test -f correct.txt"
    assert repaired["target_path"] == "correct.txt"
    assert repaired["verify_polarity"] == "present"
    assert repaired["validation_command_expanded"] is True


def test_neural_controller_validation_pointer_control_token_overrides_freeform_content():
    encoder_text = "\n".join(
        [
            "Validation present commands: test -f wrong.txt | test -f correct.txt",
            "Validation present candidate 1: test -f wrong.txt",
            "Validation present candidate 2: test -f correct.txt",
        ]
    )
    repaired, warnings = repair_line_protocol_with_command_copy_target(
        {
            "tokens": [
                "<AK_EXEC_KIND_MATERIALIZE_ARTIFACT>",
                "<AK_COPY_VALIDATE_PRESENT_CANDIDATE_2>",
            ],
            "action": "code_execute",
            "content": "mkdir -p docs && printf ok > docs/wrong.txt",
        },
        encoder_text=encoder_text,
    )

    assert warnings == []
    assert repaired["content"] == "test -f correct.txt"
    assert repaired["target_path"] == "correct.txt"
    assert repaired["tokens"][0] == "<AK_EXEC_KIND_VERIFY_PRESENT>"
    assert repaired["verify_polarity"] == "present"
    assert repaired["validation_command_expanded"] is True


def test_neural_controller_repairs_validate_present_to_existing_expected_artifact():
    encoder_text = "\n".join(
        [
            'World: {"existing_expected_artifacts": ["docs/module_map.md"], "missing_expected_artifacts": ["src/runtime.txt"]}',
            "Validation present commands: test -f src/runtime.txt | test -f docs/module_map.md",
        ]
    )
    repaired, warnings = repair_line_protocol_with_command_copy_target(
        {
            "tokens": ["<AK_VALIDATE_PRESENT>", "<AK_EXEC_KIND_VERIFY_PRESENT>"],
            "action": "code_execute",
            "content": "test -f docs/runtime.md",
        },
        encoder_text=encoder_text,
    )

    assert warnings == []
    assert repaired["content"] == "test -f docs/module_map.md"
    assert repaired["target_path"] == "docs/module_map.md"
    assert repaired["verify_polarity"] == "present"
    assert repaired["validation_command_repaired"] is True


def test_neural_controller_world_state_validation_repair_handles_nested_json():
    encoder_text = "\n".join(
        [
            (
                'World: {"existing_expected_artifacts": ["docs/module_map.md"], '
                '"semantic_episodes": [{"recovery_trace": {"nested": true}}]}'
            ),
            "Validation present commands: test -f src/runtime.txt | test -f docs/module_map.md",
        ]
    )
    repaired, warnings = repair_line_protocol_with_command_copy_target(
        {
            "tokens": ["<AK_VALIDATE_PRESENT>", "<AK_EXEC_KIND_VERIFY_PRESENT>"],
            "action": "code_execute",
            "content": "test -f tmp/runtime.txt",
        },
        encoder_text=encoder_text,
    )

    assert warnings == []
    assert repaired["content"] == "test -f docs/module_map.md"
    assert repaired["target_path"] == "docs/module_map.md"
    assert repaired["validation_command_repaired"] is True


def test_neural_controller_world_state_validation_repair_handles_compacted_json():
    encoder_text = "\n".join(
        [
            (
                'World: {"existing_expected_artifacts": ["docs/module_map.md"], '
                '"semantic_episodes": [{"recovery_trace": {"nested": true}}]...'
            ),
            "Validation present commands: test -f config/deploy.env | test -f docs/module_map.md",
        ]
    )
    repaired, warnings = repair_line_protocol_with_command_copy_target(
        {
            "tokens": ["<AK_VALIDATE_PRESENT>", "<AK_EXEC_KIND_VERIFY_PRESENT>"],
            "action": "code_execute",
            "content": "test -f tmp/runtime.txt",
        },
        encoder_text=encoder_text,
    )

    assert warnings == []
    assert repaired["content"] == "test -f docs/module_map.md"
    assert repaired["target_path"] == "docs/module_map.md"
    assert repaired["validation_command_repaired"] is True


def test_neural_controller_repairs_validate_absent_to_present_forbidden_artifact():
    encoder_text = "\n".join(
        [
            'World: {"present_forbidden_artifacts": ["tmp/debug.log"], "forbidden_artifacts": ["tmp/debug.log"]}',
            "Validation absent commands: test ! -f tmp/debug.log",
        ]
    )
    repaired, warnings = repair_line_protocol_with_command_copy_target(
        {
            "tokens": ["<AK_VALIDATE_ABSENT>", "<AK_EXEC_KIND_MATERIALIZE_ARTIFACT>"],
            "action": "code_execute",
            "content": "printf ok > tmp/debug.log",
        },
        encoder_text=encoder_text,
    )

    assert warnings == []
    assert repaired["content"] == "test ! -f tmp/debug.log"
    assert repaired["target_path"] == "tmp/debug.log"
    assert repaired["verify_polarity"] == "absent"
    assert repaired["validation_command_repaired"] is True


def test_neural_controller_validation_intent_overrides_localized_pointer_content():
    encoder_text = "\n".join(
        [
            'World: {"present_forbidden_artifacts": ["staging/draft.txt"]}',
            "Validation absent commands: test ! -f staging/draft.txt",
            "Localized edit candidate 1: sed -i '1s#^ENV=base\\ pending$#ENV=base#' config/base.env",
        ]
    )
    repaired, warnings = repair_line_protocol_with_command_copy_target(
        {
            "tokens": [
                "<AK_VALIDATE_ABSENT>",
                "<AK_EXEC_KIND_LOCALIZED_EDIT>",
            ],
            "action": "code_execute",
            "content": "<AK_COPY_LOCALIZED_EDIT_CANDIDATE_1>",
        },
        encoder_text=encoder_text,
    )

    assert warnings == []
    assert repaired["content"] == "test ! -f staging/draft.txt"
    assert repaired["target_path"] == "staging/draft.txt"
    assert repaired["tokens"][0] == "<AK_VALIDATE_ABSENT>"
    assert repaired["tokens"][1] == "<AK_EXEC_KIND_VERIFY_ABSENT>"
    assert repaired["validation_command_repaired"] is True


def test_neural_controller_shadow_readiness_blocks_content_authority_without_content_evidence():
    summary = {
        "episodes_with_shadow": 5,
        "ready_steps": 25,
        "shadow_steps": 25,
        "action_agreement_rate": 1.0,
        "verified_action_agreement_rate": 1.0,
        "content_comparison_steps": 0,
        "content_exact_agreement_rate": 0.0,
        "error_rate": 0.0,
        "warning_rate": 0.0,
    }

    readiness = neural_controller_shadow_promotion_readiness(summary)

    assert readiness["shadow_compare_ready"] is True
    assert readiness["kernel_guarded_content_ready"] is False
    assert "insufficient_content_comparison_steps" in readiness["content_authority_blockers"]


def test_neural_controller_shadow_document_summary_and_readiness_gate():
    documents = [
        {
            "task_id": "episode_1",
            "steps": [
                {
                    "proposal_metadata": {
                        "neural_controller_shadow": {
                            "ready": True,
                            "action_agreement": True,
                            "content_exact_agreement": True,
                            "control_tokens": ["<AK_VERIFY>"],
                        }
                    },
                    "verification": {"passed": True},
                }
            ],
        },
        {
            "task_id": "episode_2",
            "summary": {
                "neural_controller_shadow": {
                    "shadow_steps": 1,
                    "ready_steps": 1,
                    "action_agreement_steps": 1,
                    "content_exact_agreement_steps": 1,
                    "verified_ready_steps": 1,
                    "verified_action_agreement_steps": 1,
                    "control_token_counts": {"<AK_PATCH_BUILD>": 1},
                }
            },
        },
    ]

    summary = summarize_neural_controller_shadow_documents(documents)
    readiness = neural_controller_shadow_promotion_readiness(
        summary,
        min_episodes=2,
        min_ready_steps=2,
        min_content_comparison_steps=2,
        min_action_agreement_rate=1.0,
        min_verified_action_agreement_rate=1.0,
    )

    assert summary["episode_count"] == 2
    assert summary["episodes_with_shadow"] == 2
    assert summary["ready_rate"] == 1.0
    assert summary["action_agreement_rate"] == 1.0
    assert summary["error_rate"] == 0.0
    assert summary["warning_rate"] == 0.0
    assert summary["verified_action_agreement_rate"] == 1.0
    assert summary["contract_content_agreement_rate"] == 1.0
    assert summary["control_token_counts"]["<AK_PATCH_BUILD>"] == 1
    assert readiness["shadow_compare_ready"] is True
    assert readiness["content_authority_ready"] is True
    assert readiness["primary_authority_ready"] is False


def test_neural_controller_shadow_summary_counts_verified_artifact_contract_content():
    documents = [
        {
            "task_id": "episode_1",
            "steps": [
                {
                    "proposal_metadata": {
                        "neural_controller_shadow": {
                            "ready": True,
                            "action_agreement": True,
                            "content_comparison_evaluated": True,
                            "content_exact_agreement": False,
                            "artifact_failure_mode": "artifact_contract_success",
                        }
                    },
                    "verification": {"passed": True},
                }
            ],
        }
    ]

    summary = summarize_neural_controller_shadow_documents(documents)

    assert summary["content_exact_agreement_steps"] == 0
    assert summary["contract_content_agreement_steps"] == 1
    assert summary["contract_content_agreement_rate"] == 1.0


def test_neural_controller_shadow_document_summary_reads_policy_trace_reports():
    documents = [
        {
            "report_kind": "unattended_task_report",
            "task_id": "report_task",
            "policy_trace": [
                {
                    "verification_passed": True,
                    "neural_controller": {
                        "shadow": {
                            "ready": True,
                            "action_agreement": True,
                            "control_tokens": ["<AK_VERIFY>"],
                        }
                    },
                }
            ],
        }
    ]

    summary = summarize_neural_controller_shadow_documents(documents)

    assert summary["episodes_with_shadow"] == 1
    assert summary["shadow_steps"] == 1
    assert summary["verified_action_agreement_steps"] == 1


def test_neural_controller_shadow_readiness_blocks_runtime_errors():
    summary = {
        "episodes_with_shadow": 5,
        "ready_steps": 25,
        "shadow_steps": 26,
        "action_agreement_rate": 0.95,
        "verified_action_agreement_rate": 0.95,
        "error_rate": 1 / 26,
        "warning_rate": 0.0,
    }

    readiness = neural_controller_shadow_promotion_readiness(summary)

    assert readiness["shadow_compare_ready"] is False
    assert "shadow_error_rate_above_gate" in readiness["blockers"]


def test_neural_controller_shadow_readiness_separates_action_and_content_authority():
    summary = {
        "episodes_with_shadow": 5,
        "ready_steps": 25,
        "shadow_steps": 25,
        "action_agreement_rate": 1.0,
        "verified_action_agreement_rate": 1.0,
        "content_comparison_steps": 25,
        "content_exact_agreement_rate": 0.0,
        "error_rate": 0.0,
        "warning_rate": 0.0,
    }

    readiness = neural_controller_shadow_promotion_readiness(summary)

    assert readiness["shadow_compare_ready"] is True
    assert readiness["content_authority_ready"] is False
    assert readiness["kernel_guarded_content_ready"] is False
    assert readiness["primary_authority_ready"] is False
    assert "content_exact_agreement_rate_below_gate" in readiness["content_authority_blockers"]


def test_neural_controller_shadow_readiness_separates_repaired_content_from_pure_authority():
    summary = {
        "episodes_with_shadow": 5,
        "ready_steps": 5,
        "shadow_steps": 5,
        "action_agreement_rate": 1.0,
        "verified_action_agreement_rate": 1.0,
        "content_comparison_steps": 5,
        "content_exact_agreement_rate": 1.0,
        "unrepaired_content_exact_agreement_rate": 0.8,
        "command_copy_target_repaired_rate": 0.2,
        "error_rate": 0.0,
        "warning_rate": 0.2,
    }

    readiness = neural_controller_shadow_promotion_readiness(
        summary,
        min_episodes=5,
        min_ready_steps=5,
        min_content_exact_agreement_rate=0.8,
    )

    assert readiness["shadow_compare_ready"] is True
    assert readiness["kernel_guarded_content_ready"] is True
    assert readiness["content_authority_ready"] is False
    assert readiness["pure_content_authority_ready"] is False
    assert "command_copy_target_repairs_present" in readiness["pure_content_authority_blockers"]


def test_neural_controller_does_not_override_content_without_copy_token():
    encoder_text = "\n".join(
        [
            "<AK_CONTEXT>",
            "Command copy target: mkdir -p data && printf '{\"ok\": true}\\n' > data/status.json",
        ]
    )
    line_protocol = {
        "tokens": ["<AK_ACTION_SPACE_CODE>", "<AK_VERIFY>"],
        "action": "code_execute",
        "content": "mkdir -p data && printf '{\"ok\": true\\n' > data/status.json",
    }

    repaired, warnings = repair_line_protocol_with_command_copy_target(
        line_protocol,
        encoder_text=encoder_text,
    )

    assert command_copy_target_from_encoder(encoder_text) == (
        "mkdir -p data && printf '{\"ok\": true}\\n' > data/status.json"
    )
    assert repaired["content"] == line_protocol["content"]
    assert "raw_content_before_command_copy_repair" not in repaired
    assert "command_copy_target_repaired" not in repaired
    assert warnings == []


def test_neural_controller_does_not_repair_canonically_equal_command_target():
    encoder_text = "\n".join(
        [
            "<AK_CONTEXT>",
            "Command copy target: mkdir -p reports && printf 'ready\\n' > reports/status.txt",
        ]
    )
    line_protocol = {
        "tokens": ["<AK_ACTION_SPACE_CODE>", "<AK_VERIFY>"],
        "action": "code_execute",
        "content": "mkdir -p reports && printf 'ready\n' > reports/status.txt",
    }

    repaired, warnings = repair_line_protocol_with_command_copy_target(
        line_protocol,
        encoder_text=encoder_text,
    )

    assert repaired == line_protocol
    assert warnings == []


def test_neural_controller_expands_model_command_copy_pointer_without_repair_warning():
    encoder_text = "\n".join(
        [
            "<AK_CONTEXT>",
            "Command copy target: mkdir -p data && printf 'ok\\n' > data/status.txt",
        ]
    )
    line_protocol = {
        "tokens": ["<AK_ACTION_SPACE_CODE>", "<AK_COPY_COMMAND_TARGET>"],
        "action": "code_execute",
        "content": "<AK_COPY_COMMAND_TARGET>",
    }

    repaired, warnings = repair_line_protocol_with_command_copy_target(
        line_protocol,
        encoder_text=encoder_text,
    )

    assert repaired["content"] == "mkdir -p data && printf 'ok\\n' > data/status.txt"
    assert repaired["command_copy_target_expanded"] is True
    assert "command_copy_target_repaired" not in repaired
    assert warnings == []


def test_neural_controller_expands_model_artifact_copy_pointer_without_repair_warning():
    encoder_text = "\n".join(
        [
            "<AK_CONTEXT>",
            "Artifact command target: mkdir -p reports && printf %s 'ready\\n' > reports/status.txt",
            "Artifact target path: reports/status.txt",
            "Artifact target content: ready\\n",
        ]
    )
    line_protocol = {
        "tokens": ["<AK_ACTION_SPACE_CODE>", "<AK_COPY_ARTIFACT_TARGET>"],
        "action": "code_execute",
        "target_path": "<AK_COPY_ARTIFACT_PATH>",
        "target_content": "<AK_COPY_ARTIFACT_CONTENT>",
        "content": "<AK_COPY_ARTIFACT_TARGET>",
    }

    repaired, warnings = repair_line_protocol_with_command_copy_target(
        line_protocol,
        encoder_text=encoder_text,
    )

    assert repaired["content"] == "mkdir -p reports && printf %s 'ready\\n' > reports/status.txt"
    assert repaired["target_path"] == "reports/status.txt"
    assert repaired["target_content"] == "ready\\n"
    assert repaired["artifact_command_target_expanded"] is True
    assert repaired["artifact_path_target_expanded"] is True
    assert repaired["artifact_content_target_expanded"] is True
    assert warnings == []


def test_neural_controller_shadow_dataset_eval_emits_manifest_scoped_documents(tmp_path, monkeypatch):
    manifest_path = tmp_path / "agentkernel_controller_manifest.json"
    dataset_path = tmp_path / "eval.jsonl"
    manifest_path.write_text(
        json.dumps({"training_summary": {"eval_dataset_path": str(dataset_path)}}),
        encoding="utf-8",
    )
    encoder_text = "\n".join(
        [
            "<AK_CONTEXT>",
            "Command copy target: mkdir -p data && printf 'ok\\n' > data/status.txt",
        ]
    )
    dataset_path.write_text(
        json.dumps(
            {
                "example_id": "direct_command:status:0",
                "task_type": "controller_action_policy_copy_pointer",
                "encoder_text": encoder_text,
                "decoder_text": "\n".join(
                    [
                        "<AK_DECIDE> <AK_ACTION_SPACE_CODE> <AK_COPY_COMMAND_TARGET>",
                        "Action: code_execute",
                        "Artifact-Failure-Mode: artifact_contract_success",
                        "Content: <AK_COPY_COMMAND_TARGET>",
                    ]
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_generate_neural_controller_text(**kwargs):
        return {
            "generated_text": "\n".join(
                [
                    "<AK_DECIDE> <AK_ACTION_SPACE_CODE> <AK_COPY_COMMAND_TARGET>",
                    "Action: code_execute",
                    "Artifact-Failure-Mode: artifact_contract_success",
                    "Content: <AK_COPY_COMMAND_TARGET>",
                ]
            ),
            "generated_token_count": 12,
            "policy_heads": {"action_validity": 0.99},
            "scalar_control": {},
        }

    monkeypatch.setattr(
        "scripts.evaluate_neural_controller_shadow_dataset.generate_neural_controller_text",
        fake_generate_neural_controller_text,
    )
    report = evaluate_dataset(
        manifest_path=manifest_path,
        dataset_path=dataset_path,
        output_path=tmp_path / "report.json",
        repo_root=tmp_path,
        device="cpu",
        limit=8,
        task_type="",
        max_new_tokens=64,
        max_encoder_tokens=256,
    )

    summary = report["summary"]
    shadow = report["documents"][0]["steps"][0]["proposal_metadata"]["neural_controller_shadow"]
    assert summary["content_comparison_steps"] == 1
    assert summary["content_exact_agreement_rate"] == 1.0
    assert shadow["manifest_path"] == str(manifest_path.resolve())
    assert shadow["command_copy_target_expanded"] is True
    assert shadow["content_comparison_evaluated"] is True
    assert shadow["target_control_tokens"] == [
        "<AK_DECIDE>",
        "<AK_ACTION_SPACE_CODE>",
        "<AK_COPY_COMMAND_TARGET>",
    ]
    assert shadow["control_token_subset_agreement"] is True


def test_neural_controller_shadow_dataset_eval_does_not_repair_literal_targets():
    row = {
        "encoder_text": "\n".join(
            [
                "<AK_CONTEXT>",
                "Command copy target: printf 'first\\n' > first.txt",
            ]
        ),
        "decoder_text": "\n".join(
            [
                "<AK_DECIDE> <AK_ACTION_SPACE_CODE>",
                "Action: code_execute",
                "Artifact-Failure-Mode: artifact_contract_success",
                "Content: printf 'second\\n' > second.txt",
            ]
        ),
    }

    target = _target_line_protocol(row)

    assert target["content"] == "printf 'second\\n' > second.txt"


def test_neural_controller_shadow_dataset_family_metrics_separate_operation_families():
    documents = [
        {
            "steps": [
                {
                    "proposal_metadata": {
                        "neural_controller_shadow": {
                            "content_exact_agreement": True,
                            "exec_kind_agreement": True,
                            "target_exec_kind": "<AK_EXEC_KIND_VERIFY_PRESENT>",
                            "target_target_path": "reports/status.txt",
                            "slot_agreements": {"target_path": True},
                        }
                    }
                }
            ]
        },
        {
            "steps": [
                {
                    "proposal_metadata": {
                        "neural_controller_shadow": {
                            "content_exact_agreement": False,
                            "artifact_failure_mode": "artifact_contract_success",
                            "exec_kind_agreement": False,
                            "target_exec_kind": "<AK_EXEC_KIND_VERIFY_ABSENT>",
                            "target_target_path": "tmp/debug.log",
                            "target_verify_polarity": "absent",
                            "slot_agreements": {"target_path": True, "verify_polarity": False},
                        }
                    }
                }
            ]
        },
    ]

    metrics = summarize_family_metrics(documents)

    assert metrics["verify_present"]["content_exact_rate"] == 1.0
    assert metrics["verify_absent"]["content_exact_rate"] == 0.0
    assert metrics["verify_absent"]["contract_content_rate"] == 1.0
    assert metrics["verify_absent"]["slot_rates"]["target_path"] == 1.0
    assert metrics["verify_absent"]["slot_rates"]["verify_polarity"] == 0.0
    assert metrics["_macro"]["min_content_exact_rate"] == 0.0
    assert metrics["_macro"]["min_contract_content_rate"] == 1.0


def test_neural_controller_family_compare_rejects_family_regression_despite_summary_gain():
    baseline_report = {
        "summary": {"content_exact_agreement_rate": 0.12},
        "family_metrics": {
            "materialize_artifact": {
                "total": 10,
                "content_exact_rate": 0.3,
                "exec_kind_agreement_rate": 0.8,
            },
            "verify_absent": {
                "total": 10,
                "content_exact_rate": 0.5,
                "exec_kind_agreement_rate": 1.0,
            },
            "_macro": {
                "macro_content_exact_rate": 0.4,
                "macro_exec_kind_agreement_rate": 0.9,
            },
        },
    }
    candidate_report = {
        "summary": {"content_exact_agreement_rate": 0.13},
        "family_metrics": {
            "materialize_artifact": {
                "total": 10,
                "content_exact_rate": 0.4,
                "exec_kind_agreement_rate": 0.8,
            },
            "verify_absent": {
                "total": 10,
                "content_exact_rate": 0.0,
                "exec_kind_agreement_rate": 0.2,
            },
            "_macro": {
                "macro_content_exact_rate": 0.2,
                "macro_exec_kind_agreement_rate": 0.5,
            },
        },
    }

    comparison = compare_family_metrics(
        baseline_report=baseline_report,
        candidate_report=candidate_report,
        baseline_label="v20",
        candidate_label="v21",
    )

    assert comparison["accepted"] is False
    assert comparison["recommendation"] == "keep_baseline"
    assert comparison["summary_delta"]["content_exact_agreement_rate"]["delta"] > 0
    assert any(
        regression.get("family") == "verify_absent"
        and regression.get("metric") == "content_exact_rate"
        for regression in comparison["regressions"]
    )


def test_neural_controller_candidate_selector_keeps_strict_baseline_with_diagnostic_rank(tmp_path):
    baseline_report = {
        "summary": {
            "content_exact_agreement_rate": 0.12,
            "content_exact_agreement_steps": 12,
        },
        "family_metrics": {
            "materialize_artifact": {
                "total": 10,
                "content_exact_rate": 0.3,
                "exec_kind_agreement_rate": 0.8,
            },
            "verify_absent": {
                "total": 10,
                "content_exact_rate": 0.5,
                "exec_kind_agreement_rate": 1.0,
            },
            "_macro": {
                "macro_content_exact_rate": 0.4,
                "macro_exec_kind_agreement_rate": 0.9,
            },
        },
    }
    regressive_candidate = {
        "summary": {
            "content_exact_agreement_rate": 0.2,
            "content_exact_agreement_steps": 20,
        },
        "family_metrics": {
            "materialize_artifact": {
                "total": 10,
                "content_exact_rate": 0.5,
                "exec_kind_agreement_rate": 0.9,
            },
            "verify_absent": {
                "total": 10,
                "content_exact_rate": 0.4,
                "exec_kind_agreement_rate": 1.0,
            },
            "_macro": {
                "macro_content_exact_rate": 0.45,
                "macro_exec_kind_agreement_rate": 0.95,
            },
        },
    }
    weak_non_regressive_candidate = {
        "summary": {
            "content_exact_agreement_rate": 0.13,
            "content_exact_agreement_steps": 13,
        },
        "family_metrics": {
            "materialize_artifact": {
                "total": 10,
                "content_exact_rate": 0.3,
                "exec_kind_agreement_rate": 0.8,
            },
            "verify_absent": {
                "total": 10,
                "content_exact_rate": 0.5,
                "exec_kind_agreement_rate": 1.0,
            },
            "_macro": {
                "macro_content_exact_rate": 0.4,
                "macro_exec_kind_agreement_rate": 0.9,
            },
        },
    }
    baseline_path = tmp_path / "baseline.json"
    regressive_path = tmp_path / "v_bad_slot_eval132_shadow_report.json"
    accepted_path = tmp_path / "v_good_slot_eval132_shadow_report.json"
    baseline_path.write_text(json.dumps(baseline_report), encoding="utf-8")
    regressive_path.write_text(json.dumps(regressive_candidate), encoding="utf-8")
    accepted_path.write_text(json.dumps(weak_non_regressive_candidate), encoding="utf-8")

    selection = select_candidate(
        baseline_report_path=baseline_path,
        candidate_report_paths=[regressive_path, accepted_path],
        baseline_label="baseline",
    )

    assert selection["strict_recommendation"] == "accept_candidate"
    assert selection["accepted_candidate_label"] == "v_good"
    assert selection["diagnostic_rank"][0]["candidate_label"] == "v_good"
    bad_row = next(row for row in selection["candidates"] if row["candidate_label"] == "v_bad")
    assert bad_row["accepted"] is False


def test_neural_controller_checkpoint_label_uses_run_and_step():
    assert (
        checkpoint_label(
            Path(
                "artifacts/agentkernel_controller/seq2seq_controller_v25/checkpoints/step_00000060.pt"
            )
        )
        == "seq2seq_controller_v25_step_00000060"
    )
    assert checkpoint_label(Path("tmp/candidate.pt")) == "tmp_candidate"


def test_neural_controller_preservation_replay_selects_baseline_wins(tmp_path):
    def document(example_id, *, exact, exec_kind, target_exec_kind):
        return {
            "steps": [
                {
                    "proposal_metadata": {
                        "neural_controller_shadow": {
                            "example_id": example_id,
                            "content_exact_agreement": exact,
                            "exec_kind_agreement": exec_kind,
                            "target_exec_kind": target_exec_kind,
                        }
                    }
                }
            ]
        }

    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    eval_path = tmp_path / "eval.jsonl"
    source_manifest_path = tmp_path / "manifest.json"
    baseline_path.write_text(
        json.dumps(
            {
                "documents": [
                    document(
                        "inspect-1",
                        exact=True,
                        exec_kind=True,
                        target_exec_kind="<AK_EXEC_KIND_INSPECT_SOURCE>",
                    ),
                    document(
                        "verify-1",
                        exact=True,
                        exec_kind=True,
                        target_exec_kind="<AK_EXEC_KIND_VERIFY_ABSENT>",
                    ),
                ]
            }
        ),
        encoding="utf-8",
    )
    candidate_path.write_text(
        json.dumps(
            {
                "documents": [
                    document(
                        "inspect-1",
                        exact=False,
                        exec_kind=False,
                        target_exec_kind="<AK_EXEC_KIND_INSPECT_SOURCE>",
                    ),
                    document(
                        "verify-1",
                        exact=False,
                        exec_kind=False,
                        target_exec_kind="<AK_EXEC_KIND_VERIFY_ABSENT>",
                    ),
                ]
            }
        ),
        encoding="utf-8",
    )
    eval_path.write_text(
        json.dumps(
            {
                "example_id": "inspect-1",
                "encoder_text": "Source inspection candidate commands: cat recovery.txt",
                "decoder_text": "<AK_EXEC_KIND_INSPECT_SOURCE>\nAction: code_execute\nContent: cat recovery.txt",
                "action": "code_execute",
            }
        )
        + "\n"
        + json.dumps(
            {
                "example_id": "verify-1",
                "encoder_text": "Validation absent commands: test ! -f tmp/debug.log",
                "decoder_text": "<AK_EXEC_KIND_VERIFY_ABSENT>\nAction: code_execute\nContent: test ! -f tmp/debug.log",
                "action": "code_execute",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    source_manifest_path.write_text(
        json.dumps({"agentkernel_special_tokens": ["<AK_EXEC_KIND_INSPECT_SOURCE>"]}),
        encoding="utf-8",
    )

    manifest = build_preservation_replay(
        argparse.Namespace(
            baseline_report=str(baseline_path),
            candidate_report=str(candidate_path),
            eval_dataset=str(eval_path),
            source_manifest=str(source_manifest_path),
            output_dir=str(tmp_path / "replay"),
            objective="unit_preservation",
            family_include="inspect_source",
            metric="either",
            repeat=3,
            distill_loss_weight=4.5,
        )
    )

    train_rows = [
        json.loads(line)
        for line in Path(manifest["train_dataset_path"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert manifest["train_examples"] == 3
    assert manifest["eval_examples"] == 1
    assert manifest["preservation_replay"]["family_counts"] == {"inspect_source": 1}
    assert manifest["preservation_replay"]["distill_loss_weight"] == 4.5
    assert {row["source_id"] for row in train_rows} == {"inspect-1"}
    assert {row["distill_loss_weight"] for row in train_rows} == {4.5}
    assert manifest["agentkernel_special_tokens"] == ["<AK_EXEC_KIND_INSPECT_SOURCE>"]


def test_guarded_neural_controller_report_falls_back_on_regressed_family(tmp_path):
    def report(path, *, inspect_exact, materialize_exact):
        documents = []
        for example_id, target_exec_kind, exact in (
            ("inspect-1", "<AK_EXEC_KIND_INSPECT_SOURCE>", inspect_exact),
            ("materialize-1", "<AK_EXEC_KIND_MATERIALIZE_ARTIFACT>", materialize_exact),
        ):
            documents.append(
                {
                    "task_id": example_id,
                    "steps": [
                        {
                            "proposal_metadata": {
                                "neural_controller_shadow": {
                                    "example_id": example_id,
                                    "ready": True,
                                    "action_agreement": True,
                                    "verified_action_agreement": True,
                                    "content_exact_agreement": exact,
                                    "exec_kind_agreement": True,
                                    "selected_action": "code_execute",
                                    "selected_content_preview": "x",
                                    "proposed_action": "code_execute",
                                    "proposed_content_preview": "x" if exact else "y",
                                    "target_exec_kind": target_exec_kind,
                                }
                            },
                            "verification": {"passed": True},
                        }
                    ],
                }
            )
        path.write_text(
            json.dumps(
                {
                    "documents": documents,
                    "summary": summarize_neural_controller_shadow_documents(documents),
                    "family_metrics": summarize_family_metrics(documents),
                }
            ),
            encoding="utf-8",
        )

    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    report(baseline_path, inspect_exact=True, materialize_exact=False)
    report(candidate_path, inspect_exact=False, materialize_exact=True)

    guarded = compose_guarded_report(
        baseline_report_path=baseline_path,
        candidate_report_path=candidate_path,
        output_path=tmp_path / "guarded.json",
        baseline_label="baseline",
        candidate_label="candidate",
        min_family_total=1,
    )

    assert guarded["fallback_families"] == ["inspect_source"]
    assert guarded["source_counts"] == {"baseline": 1, "candidate": 1}
    assert guarded["summary"]["content_exact_agreement_steps"] == 2


def test_rowwise_frontier_reports_recoverable_baseline_misses(tmp_path):
    def write_report(path, rows):
        documents = []
        for example_id, target_exec_kind, exact in rows:
            shadow = {
                "example_id": example_id,
                "ready": True,
                "action_agreement": True,
                "verified_action_agreement": True,
                "content_exact_agreement": exact,
                "exec_kind_agreement": True,
                "selected_action": "code_execute",
                "selected_content_preview": "x" if exact else "miss",
                "target_exec_kind": target_exec_kind,
            }
            documents.append(
                {
                    "task_id": example_id,
                    "steps": [
                        {
                            "proposal_metadata": {"neural_controller_shadow": shadow},
                            "verification": {"passed": True},
                        }
                    ],
                }
            )
        path.write_text(
            json.dumps(
                {
                    "documents": documents,
                    "summary": summarize_neural_controller_shadow_documents(documents),
                    "family_metrics": summarize_family_metrics(documents),
                }
            ),
            encoding="utf-8",
        )

    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    retained_path = tmp_path / "retained.json"
    inspect = "<AK_EXEC_KIND_INSPECT_SOURCE>"
    materialize = "<AK_EXEC_KIND_MATERIALIZE_ARTIFACT>"
    write_report(baseline_path, [("row-a", inspect, True), ("row-b", materialize, False)])
    write_report(candidate_path, [("row-a", inspect, False), ("row-b", materialize, True)])
    write_report(retained_path, [("row-a", inspect, True), ("row-b", materialize, True)])

    report = report_rowwise_frontier(
        baseline_report_path=baseline_path,
        candidate_report_paths=[candidate_path],
        retained_report_path=retained_path,
        output_path=tmp_path / "frontier.json",
        baseline_label="baseline",
        candidate_labels=["candidate"],
        selector_dataset_output_path=tmp_path / "selector.jsonl",
    )

    assert report["frontier_summary"]["content_exact_agreement_steps"] == 2
    assert report["source_counts"] == {"baseline": 1, "candidate": 1}
    assert report["selector_dataset_summary"] == {
        "accepted": 1,
        "accepted_by_candidate": {"candidate": 1},
        "rejected": 1,
        "rows": 2,
    }
    assert report["family_recovery_counts"]["materialize_artifact"]["baseline_miss_recovered"] == 1
    assert not [row for row in report["rows"] if row["retained_only_unrecovered"]]
    selector_rows = [
        json.loads(line)
        for line in (tmp_path / "selector.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["accept_candidate"] for row in selector_rows] == [False, True]
    assert selector_rows[1]["candidate_improves_strict"] is True

    selector_report = evaluate_rowwise_selector(
        baseline_report_path=baseline_path,
        candidate_report_paths=[candidate_path],
        selector_dataset_path=tmp_path / "selector.jsonl",
        output_path=tmp_path / "selector_report.json",
        baseline_label="baseline",
        candidate_labels=["candidate"],
    )

    assert selector_report["summary"]["content_exact_agreement_steps"] == 2
    assert selector_report["source_counts"] == {"baseline": 1, "candidate": 1}
    selected = {row["example_id"]: row["selected_source"] for row in selector_report["rows"]}
    assert selected == {"row-a": "baseline", "row-b": "candidate"}

    readiness = report_selector_readiness(
        baseline_report_path=baseline_path,
        selector_report_path=tmp_path / "selector_report.json",
        retained_report_path=retained_path,
        output_path=tmp_path / "readiness.json",
        baseline_label="baseline",
    )

    assert readiness["reaches_retained_strict"] is True
    assert readiness["primary_authority_ready"] is False
    assert readiness["recommendation"] == "promote_selector_to_retained_candidate_packet"

    packet = build_selector_retained_candidate_packet(
        readiness_path=tmp_path / "readiness.json",
        selector_dataset_path=tmp_path / "selector.jsonl",
        selector_report_path=tmp_path / "selector_report.json",
        output_path=tmp_path / "selector_packet.json",
    )

    assert packet["retained_candidate_ready"] is True
    assert packet["recommended_runtime_mode"] == "guarded"
    assert packet["primary_authority_ready"] is False
    assert packet["metrics"]["source_switch_count"] == 1

    runtime_metrics_path = tmp_path / "runtime_contract_metrics.json"
    runtime_metrics_path.write_text(
        json.dumps(
            {
                "report_kind": "neural_controller_runtime_contract_metrics",
                "summary": {
                    "runtime_contract_steps": 1,
                    "runtime_contract_task_count": 1,
                    "runtime_contract_success_steps": 1,
                    "runtime_contract_success_rate": 1.0,
                    "selector_signal_ready": True,
                },
            }
        ),
        encoding="utf-8",
    )
    activation = report_selector_activation_gate(
        retained_candidate_packet_path=tmp_path / "selector_packet.json",
        runtime_contract_metrics_path=runtime_metrics_path,
        output_path=tmp_path / "activation.json",
    )

    assert activation["guarded_selector_activation_ready"] is True
    assert activation["production_guarded_selector_activation_ready"] is False
    assert activation["recommended_runtime_mode"] == "guarded"
    assert activation["primary_authority_ready"] is False
    assert "runtime_contract_steps_below_production_gate" in activation["production_blockers"]


def test_dataset_merge_unions_agentkernel_special_tokens():
    tokens = _union_lists(
        [
            {"agentkernel_special_tokens": ["<AK_DECIDE>", "<AK_COPY_COMMAND_TARGET>"]},
            {"agentkernel_special_tokens": ["<AK_DECIDE>", "<AK_COPY_ARTIFACT_TARGET>"]},
        ],
        "agentkernel_special_tokens",
    )

    assert tokens == ["<AK_DECIDE>", "<AK_COPY_COMMAND_TARGET>", "<AK_COPY_ARTIFACT_TARGET>"]


def test_dataset_writer_default_distill_loss_weight_is_configurable(tmp_path):
    (tmp_path / "merge").mkdir()
    writer = DatasetWriter(
        tmp_path / "merge",
        output_format="jsonl",
        parquet_shard_size=100,
        default_distill_loss_weight=0.0,
    )
    try:
        row = writer._normalize(
            {
                "encoder_text": "Task: inspect",
                "decoder_text": "<AK_EXEC_KIND_INSPECT_SOURCE>\nAction: code_execute\nContent: cat a.txt",
            },
            split="train",
        )
        explicit = writer._normalize(
            {
                "encoder_text": "Task: preserve",
                "decoder_text": "<AK_EXEC_KIND_INSPECT_SOURCE>\nAction: code_execute\nContent: cat b.txt",
                "distill_loss_weight": 8.0,
            },
            split="train",
        )
    finally:
        writer.close()

    assert row["distill_loss_weight"] == 0.0
    assert explicit["distill_loss_weight"] == 8.0


def test_slot_curriculum_builder_oversamples_slot_rows_and_keeps_slot_eval(tmp_path):
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    manifest_path = tmp_path / "manifest.json"
    slot_row = {
        "example_id": "slot-train",
        "split": "train",
        "source_type": "unit",
        "source_id": "slot-train",
        "task_type": "controller_long_horizon",
        "encoder_text": "Task: edit file",
        "decoder_text": "\n".join(
            [
                "<AK_DECIDE> <AK_EXEC_KIND_LOCALIZED_EDIT>",
                "Action: code_execute",
                "Target-Path: project/plan.md",
                "Edit-Old: old",
                "Edit-New: new",
                "Content: sed -i '1s#old#new#' project/plan.md",
            ]
        ),
        "action": "code_execute",
        "weight": 1.0,
    }
    no_slot_row = {
        **slot_row,
        "example_id": "plain-train",
        "decoder_text": "<AK_DECIDE>\nAction: respond\nContent: done",
        "action": "respond",
    }
    eval_slot_row = {**slot_row, "example_id": "slot-eval", "split": "eval"}
    train_path.write_text(
        json.dumps(slot_row, sort_keys=True) + "\n" + json.dumps(no_slot_row, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    eval_path.write_text(json.dumps(eval_slot_row, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "artifact_kind": "agentkernel_controller_long_horizon_dataset",
                "train_dataset_path": str(train_path),
                "eval_dataset_path": str(eval_path),
                "agentkernel_special_tokens": ["<AK_DECIDE>", "<AK_EXEC_KIND_LOCALIZED_EDIT>"],
            }
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "slot_curriculum"
    manifest = build_slot_curriculum(
        argparse.Namespace(
            dataset_manifest=str(manifest_path),
            output_dir=str(output_dir),
            objective="unit_slot_curriculum",
            base_repeat=2,
            target_content_bonus=0,
            verify_bonus=0,
            verify_present_bonus=0,
            edit_bonus=3,
            family_bonus_json="",
            max_repeat=8,
            output_format="jsonl",
            parquet_shard_size=50000,
        )
    )

    train_rows = [
        json.loads(line)
        for line in Path(manifest["train_dataset_path"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    eval_rows = [
        json.loads(line)
        for line in Path(manifest["eval_dataset_path"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert manifest["train_examples"] == 5
    assert manifest["eval_examples"] == 1
    assert {row["example_id"].split(":slot_curriculum:")[0] for row in train_rows} == {"slot-train"}
    assert eval_rows[0]["example_id"].startswith("slot-eval:slot_curriculum:")
    assert manifest["slot_curriculum"]["source_counts"]["train_slot_rows"] == 1
    assert manifest["slot_curriculum"]["slot_counts"]["train"]["edit_old"] == 1


def test_slot_curriculum_builder_applies_generic_operation_family_bonus(tmp_path):
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    manifest_path = tmp_path / "manifest.json"
    verify_absent_row = {
        "example_id": "verify-absent-train",
        "split": "train",
        "source_type": "unit",
        "source_id": "verify-absent-train",
        "task_type": "controller_long_horizon",
        "encoder_text": "Task: verify forbidden artifact absent",
        "decoder_text": "\n".join(
            [
                "<AK_DECIDE> <AK_EXEC_KIND_VERIFY_ABSENT>",
                "Action: code_execute",
                "Target-Path: tmp/debug.log",
                "Verify-Polarity: absent",
                "Content: test ! -f tmp/debug.log",
            ]
        ),
        "action": "code_execute",
        "weight": 1.0,
    }
    materialize_row = {
        **verify_absent_row,
        "example_id": "materialize-train",
        "source_id": "materialize-train",
        "decoder_text": "\n".join(
            [
                "<AK_DECIDE> <AK_EXEC_KIND_MATERIALIZE_ARTIFACT>",
                "Action: code_execute",
                "Target-Path: reports/status.txt",
                "Target-Content: ready",
                "Content: mkdir -p reports && printf %s 'ready' > reports/status.txt",
            ]
        ),
    }
    train_path.write_text(
        json.dumps(verify_absent_row, sort_keys=True) + "\n"
        + json.dumps(materialize_row, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    eval_path.write_text("", encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "artifact_kind": "agentkernel_controller_long_horizon_dataset",
                "train_dataset_path": str(train_path),
                "eval_dataset_path": str(eval_path),
                "agentkernel_special_tokens": [
                    "<AK_DECIDE>",
                    "<AK_EXEC_KIND_VERIFY_ABSENT>",
                    "<AK_EXEC_KIND_MATERIALIZE_ARTIFACT>",
                ],
            }
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "family_curriculum"
    manifest = build_slot_curriculum(
        argparse.Namespace(
            dataset_manifest=str(manifest_path),
            output_dir=str(output_dir),
            objective="unit_family_curriculum",
            base_repeat=1,
            target_content_bonus=0,
            verify_bonus=0,
            verify_present_bonus=0,
            edit_bonus=0,
            family_bonus_json='{"verify_absent": 3}',
            max_repeat=8,
            output_format="jsonl",
            parquet_shard_size=50000,
        )
    )

    train_rows = [
        json.loads(line)
        for line in Path(manifest["train_dataset_path"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    family_by_base = {}
    for row in train_rows:
        base_id = row["example_id"].split(":slot_curriculum:")[0]
        family_by_base.setdefault(base_id, []).append(row["slot_curriculum_operation_family"])

    assert len(family_by_base["verify-absent-train"]) == 4
    assert len(family_by_base["materialize-train"]) == 1
    assert set(family_by_base["verify-absent-train"]) == {"verify_absent"}
    assert manifest["slot_curriculum"]["repeat_policy"]["family_bonus"] == {"verify_absent": 3}
    assert manifest["slot_curriculum"]["repeated_family_counts"]["train"]["verify_absent"] == 4
    assert manifest["slot_curriculum"]["repeated_family_counts"]["train"]["materialize_artifact"] == 1
