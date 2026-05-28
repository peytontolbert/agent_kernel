#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_kernel.neural_controller import FULL_KERNEL_CONTROL_TOKENS, build_neural_controller_encoder_text
from agent_kernel.tasking.task_bank import TaskBank


DIRECT_CODE_CONTROL_TOKENS = (
    "<AK_DECIDE> <AK_ACTION_SPACE_CODE> <AK_NO_RETRIEVAL> "
    "<AK_EXECUTE> <AK_VERIFY> <AK_WORLD_UPDATE> <AK_MEMORY_WRITE> <AK_CONF_HIGH>"
)
DIRECT_COPY_POINTER_CONTROL_TOKENS = (
    "<AK_DECIDE> <AK_ACTION_SPACE_CODE> <AK_NO_RETRIEVAL> <AK_COPY_COMMAND_TARGET> "
    "<AK_EXECUTE> <AK_VERIFY> <AK_WORLD_UPDATE> <AK_MEMORY_WRITE> <AK_CONF_HIGH>"
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _split_for_copy(copy_index: int, repeat: int, eval_ratio: float) -> str:
    if repeat <= 1:
        return "train"
    eval_stride = max(2, round(1.0 / max(0.01, min(0.5, float(eval_ratio)))))
    return "eval" if copy_index % eval_stride == 0 else "train"


def _copy_pointer_candidate(command: str) -> bool:
    command = str(command)
    return (
        len(command) >= 180
        or command.count("&&") >= 3
        or command.count('{"') >= 2
        or command.count("\\n") >= 4
    )


def _direct_command_rows(*, repeat: int, eval_ratio: float) -> list[dict[str, object]]:
    bank = TaskBank()
    rows: list[dict[str, object]] = []
    for task_id in sorted(bank._tasks):
        task = bank._tasks[task_id]
        metadata = dict(task.metadata)
        if not bool(metadata.get("light_supervision_candidate", False)):
            continue
        if str(metadata.get("requires_retrieval", "")).lower() == "true":
            continue
        family = str(metadata.get("benchmark_family", "bounded")).strip() or "bounded"
        if family == "repo_sandbox":
            continue
        command = task.suggested_commands[0] if task.suggested_commands else ""
        if not command:
            continue
        command = command.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\n")
        encoder_text = build_neural_controller_encoder_text(
            state_payload={
                "task": {
                    "task_id": task.task_id,
                    "prompt": task.prompt,
                    "workspace_subdir": task.workspace_subdir,
                    "suggested_commands": [command],
                    "success_command": task.success_command,
                    "expected_files": list(task.expected_files),
                    "expected_file_contents": dict(task.expected_file_contents),
                    "metadata": {"benchmark_family": family},
                },
                "history": [],
                "active_subgoal": "direct bounded workspace action",
            }
        )
        decoder_text = "\n".join(
            [
                DIRECT_CODE_CONTROL_TOKENS,
                "Action: code_execute",
                "Artifact-Failure-Mode: artifact_contract_success",
                f"Content: {command}",
            ]
        )
        base = {
            "action": "code_execute",
            "answer_confidence_target": 0.9,
            "decoder_text": decoder_text,
            "encoder_text": encoder_text,
            "needs_verification_target": 0.85,
            "ood_evidence_target": 0.05,
            "ood_query_target": 0.05,
            "paper_action_validity_target": 0.95,
            "query_confidence_target": 0.9,
            "retrieval_coverage_target": 0.0,
            "retrieval_doc_text": "",
            "retrieval_loss_weight": 0.0,
            "retrieval_query_text": "",
            "source_id": f"task_bank:{task.task_id}:direct_command",
            "source_type": "agentkernel_controller_direct_command_repair",
            "split": "train",
            "task_type": "controller_action_policy",
            "weight": 2.0,
        }
        for copy_index in range(max(1, repeat)):
            row = dict(base)
            row["example_id"] = f"direct_command:{task.task_id}:{copy_index}"
            row["split"] = _split_for_copy(copy_index, max(1, repeat), eval_ratio)
            if _copy_pointer_candidate(command):
                row["decoder_text"] = "\n".join(
                    [
                        DIRECT_COPY_POINTER_CONTROL_TOKENS,
                        "Action: code_execute",
                        "Artifact-Failure-Mode: artifact_contract_success",
                        "Content: <AK_COPY_COMMAND_TARGET>",
                    ]
                )
                row["task_type"] = "controller_action_policy_copy_pointer"
                row["weight"] = 2.4
            rows.append(row)
    rows.extend(_synthetic_command_copy_rows(repeat=max(1, repeat // 4), eval_ratio=eval_ratio))
    return rows


def _synthetic_command_copy_rows(*, repeat: int, eval_ratio: float) -> list[dict[str, object]]:
    specs = [
        (
            "json_status",
            "Create data/status.json with ok true.",
            "mkdir -p data && printf '{\"ok\": true}\\n' > data/status.json",
            "data/status.json",
        ),
        (
            "json_metrics",
            "Create reports/metrics.json with score 1.",
            "mkdir -p reports && printf '{\"score\": 1}\\n' > reports/metrics.json",
            "reports/metrics.json",
        ),
        (
            "env_pair",
            "Create config/app.env with MODE and PORT.",
            "mkdir -p config && printf 'MODE=prod\\nPORT=8080\\n' > config/app.env",
            "config/app.env",
        ),
        (
            "yaml_flag",
            "Create config/runtime.yaml with enabled true.",
            "mkdir -p config && printf 'enabled: true\\n' > config/runtime.yaml",
            "config/runtime.yaml",
        ),
        (
            "nested_report",
            "Create reports/status.txt containing ready.",
            "mkdir -p reports && printf 'ready\\n' > reports/status.txt",
            "reports/status.txt",
        ),
        (
            "plain_result",
            "Create result.txt containing 42.",
            "printf '42\\n' > result.txt",
            "result.txt",
        ),
    ]
    rows: list[dict[str, object]] = []
    for task_id, prompt, command_raw, expected_file in specs:
        command = command_raw.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\n")
        encoder_text = build_neural_controller_encoder_text(
            state_payload={
                "task": {
                    "task_id": f"synthetic_copy_{task_id}",
                    "prompt": prompt,
                    "workspace_subdir": f"synthetic_copy_{task_id}",
                    "suggested_commands": [command],
                    "success_command": f"test -f {expected_file}",
                    "expected_files": [expected_file],
                    "expected_file_contents": {},
                    "metadata": {"benchmark_family": "bounded"},
                },
                "history": [],
                "active_subgoal": "direct bounded workspace action",
            }
        )
        decoder_text = "\n".join(
            [
                DIRECT_CODE_CONTROL_TOKENS,
                "Action: code_execute",
                "Artifact-Failure-Mode: artifact_contract_success",
                f"Content: {command}",
            ]
        )
        for copy_index in range(repeat):
            decoder_text_for_copy = decoder_text
            task_type = "controller_action_policy"
            weight = 2.5
            if _copy_pointer_candidate(command):
                decoder_text_for_copy = "\n".join(
                    [
                        DIRECT_COPY_POINTER_CONTROL_TOKENS,
                        "Action: code_execute",
                        "Artifact-Failure-Mode: artifact_contract_success",
                        "Content: <AK_COPY_COMMAND_TARGET>",
                    ]
                )
                task_type = "controller_action_policy_copy_pointer"
                weight = 2.8
            rows.append(
                {
                    "action": "code_execute",
                    "answer_confidence_target": 0.9,
                    "decoder_text": decoder_text_for_copy,
                    "encoder_text": encoder_text,
                    "example_id": f"synthetic_command_copy:{task_id}:{copy_index}",
                    "needs_verification_target": 0.85,
                    "ood_evidence_target": 0.05,
                    "ood_query_target": 0.05,
                    "paper_action_validity_target": 0.95,
                    "query_confidence_target": 0.9,
                    "retrieval_coverage_target": 0.0,
                    "retrieval_doc_text": "",
                    "retrieval_loss_weight": 0.0,
                    "retrieval_query_text": "",
                    "source_id": f"synthetic_command_copy:{task_id}",
                    "source_type": "agentkernel_controller_direct_command_repair",
                    "split": _split_for_copy(copy_index, repeat, eval_ratio),
                    "task_type": task_type,
                    "weight": weight,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="artifacts/agentkernel_controller/direct_command_repair_v1")
    parser.add_argument("--repeat", type=int, default=80)
    parser.add_argument("--eval-ratio", type=float, default=0.10)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    rows = _direct_command_rows(repeat=max(1, int(args.repeat)), eval_ratio=float(args.eval_ratio))
    if not rows:
        raise SystemExit("no direct command rows generated")
    eval_rows = [row for row in rows if row.get("split") == "eval"]
    train_rows = [row for row in rows if row.get("split") != "eval"]
    if not eval_rows:
        eval_rows = rows[:1]
        train_rows = rows[1:]
    train_path = output_dir / "agentkernel_lite_encdec_train.jsonl"
    eval_path = output_dir / "agentkernel_lite_encdec_eval.jsonl"
    manifest_path = output_dir / "agentkernel_lite_encdec_dataset_manifest.json"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(eval_path, eval_rows)
    manifest = {
        "artifact_kind": "agentkernel_lite_encdec_distill_dataset",
        "dataset_format": "jsonl",
        "decoder_format": "line",
        "objective": "agentkernel_controller_direct_command_repair",
        "manifest_path": str(manifest_path.resolve()),
        "train_dataset_path": str(train_path.resolve()),
        "eval_dataset_path": str(eval_path.resolve()),
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "total_examples": len(rows),
        "source_counts": {"agentkernel_controller_direct_command_repair": len(rows)},
        "action_counts": {"code_execute": len(rows)},
        "agentkernel_special_tokens": list(FULL_KERNEL_CONTROL_TOKENS),
        "schema": {
            "encoder_text": "direct bounded task state with explicit no-retrieval contract",
            "decoder_text": "direct code action-space tokens followed by line protocol action/content target",
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"direct_command_rows={len(rows)} train={len(train_rows)} eval={len(eval_rows)} manifest={manifest_path}"
    )


if __name__ == "__main__":
    main()
