from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.build_neural_controller_retained_output_distill import build_retained_output_distill


def test_build_retained_output_distill_rewrites_decoder_to_selected_content(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    eval_path = tmp_path / "eval.jsonl"
    output_dir = tmp_path / "out"
    report_path.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "task_id": "row-1",
                        "steps": [
                            {
                                "proposal_metadata": {
                                    "neural_controller_shadow": {
                                        "example_id": "row-1",
                                        "content_exact_agreement": True,
                                        "selected_action": "code_execute",
                                        "selected_content": "cat correct.py",
                                        "target_control_tokens": [
                                            "<AK_DECIDE>",
                                            "<AK_READ_SOURCE>",
                                            "<AK_EXEC_KIND_INSPECT_SOURCE>",
                                        ],
                                        "target_target_path": "correct.py",
                                        "artifact_failure_mode": "missing_expected_file",
                                    }
                                }
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    eval_path.write_text(
        json.dumps(
            {
                "example_id": "row-1",
                "encoder_text": "Context",
                "decoder_text": "Content: cat stale.py",
                "action": "code_execute",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = build_retained_output_distill(
        argparse.Namespace(
            retained_report=str(report_path),
            eval_dataset=str(eval_path),
            source_manifest="",
            output_dir=str(output_dir),
            objective="test",
            repeat=2,
            distill_loss_weight=1.0,
            allow_preview_fallback=False,
            only_content_wins=True,
            pointerize_candidates=False,
        )
    )

    assert manifest["train_examples"] == 2
    train_rows = [
        json.loads(line)
        for line in Path(manifest["train_dataset_path"]).read_text(encoding="utf-8").splitlines()
    ]
    assert all("Content: cat correct.py" in row["decoder_text"] for row in train_rows)
    assert all("Target-Path: correct.py" in row["decoder_text"] for row in train_rows)


def test_build_retained_output_distill_can_pointerize_source_candidate(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    eval_path = tmp_path / "eval.jsonl"
    output_dir = tmp_path / "out"
    report_path.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "task_id": "row-1",
                        "steps": [
                            {
                                "proposal_metadata": {
                                    "neural_controller_shadow": {
                                        "example_id": "row-1",
                                        "content_exact_agreement": True,
                                        "selected_action": "code_execute",
                                        "selected_content": "cat correct.py",
                                        "target_control_tokens": ["<AK_EXEC_KIND_INSPECT_SOURCE>"],
                                        "target_target_path": "correct.py",
                                    }
                                }
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    eval_path.write_text(
        json.dumps(
            {
                "example_id": "row-1",
                "encoder_text": "Source inspection candidate commands: cat wrong.py | cat correct.py",
                "decoder_text": "Content: cat stale.py",
                "action": "code_execute",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = build_retained_output_distill(
        argparse.Namespace(
            retained_report=str(report_path),
            eval_dataset=str(eval_path),
            source_manifest="",
            output_dir=str(output_dir),
            objective="test",
            repeat=1,
            distill_loss_weight=1.0,
            allow_preview_fallback=False,
            only_content_wins=True,
            pointerize_candidates=True,
        )
    )
    row = json.loads(Path(manifest["train_dataset_path"]).read_text(encoding="utf-8").strip())
    audit = manifest["retained_output_distill"]["pointer_grounding_audit"]

    assert "<AK_COPY_SOURCE_INSPECT_CANDIDATE_3>" in manifest["agentkernel_special_tokens"]
    assert "<AK_COPY_SOURCE_INSPECT_CANDIDATE_3>" in row["decoder_text"]
    assert "Content: <AK_COPY_SOURCE_INSPECT_CANDIDATE_3>" in row["decoder_text"]
    assert audit["pointer_rows"] == 2
    assert audit["pointer_tokens"] == 4
    assert audit["invalid_pointer_tokens"] == 0


def test_build_retained_output_distill_can_pointerize_materialization_candidate(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    eval_path = tmp_path / "eval.jsonl"
    output_dir = tmp_path / "out"
    selected = "mkdir -p reports && printf %s 'ready\\n' > reports/status.txt"
    report_path.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "task_id": "row-1",
                        "steps": [
                            {
                                "proposal_metadata": {
                                    "neural_controller_shadow": {
                                        "example_id": "row-1",
                                        "content_exact_agreement": True,
                                        "selected_action": "code_execute",
                                        "selected_content": selected,
                                        "target_control_tokens": ["<AK_EXEC_KIND_MATERIALIZE_ARTIFACT>"],
                                        "target_target_path": "reports/status.txt",
                                    }
                                }
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    eval_path.write_text(
        json.dumps(
            {
                "example_id": "row-1",
                "encoder_text": "\n".join(
                    [
                        "Artifact command target: " + selected,
                        "Materialization candidate 1: " + selected,
                    ]
                ),
                "decoder_text": "Content: test -f reports/status.txt",
                "action": "code_execute",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = build_retained_output_distill(
        argparse.Namespace(
            retained_report=str(report_path),
            eval_dataset=str(eval_path),
            source_manifest="",
            output_dir=str(output_dir),
            objective="test",
            repeat=1,
            distill_loss_weight=1.0,
            allow_preview_fallback=False,
            only_content_wins=True,
            pointerize_candidates=True,
        )
    )
    row = json.loads(Path(manifest["train_dataset_path"]).read_text(encoding="utf-8").strip())
    audit = manifest["retained_output_distill"]["pointer_grounding_audit"]

    assert "<AK_COPY_MATERIALIZE_CANDIDATE_1>" in manifest["agentkernel_special_tokens"]
    assert "<AK_COPY_MATERIALIZE_CANDIDATE_1>" in row["decoder_text"]
    assert "Content: <AK_COPY_MATERIALIZE_CANDIDATE_1>" in row["decoder_text"]
    assert audit["pointer_rows"] == 2
    assert audit["pointer_tokens"] == 4
    assert audit["invalid_pointer_tokens"] == 0


def test_build_retained_output_distill_can_filter_pointer_family(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    eval_path = tmp_path / "eval.jsonl"
    output_dir = tmp_path / "out"
    materialize = "mkdir -p reports && printf %s 'ready\\n' > reports/status.txt"
    report_path.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "task_id": "materialize-row",
                        "steps": [
                            {
                                "proposal_metadata": {
                                    "neural_controller_shadow": {
                                        "example_id": "materialize-row",
                                        "content_exact_agreement": True,
                                        "selected_action": "code_execute",
                                        "selected_content": materialize,
                                    }
                                }
                            }
                        ],
                    },
                    {
                        "task_id": "source-row",
                        "steps": [
                            {
                                "proposal_metadata": {
                                    "neural_controller_shadow": {
                                        "example_id": "source-row",
                                        "content_exact_agreement": True,
                                        "selected_action": "code_execute",
                                        "selected_content": "cat correct.py",
                                    }
                                }
                            }
                        ],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    eval_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "example_id": "materialize-row",
                        "encoder_text": "Materialization candidate 1: " + materialize,
                        "decoder_text": "Content: stale",
                        "action": "code_execute",
                    }
                ),
                json.dumps(
                    {
                        "example_id": "source-row",
                        "encoder_text": "Source inspection candidate commands: cat correct.py",
                        "decoder_text": "Content: stale",
                        "action": "code_execute",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = build_retained_output_distill(
        argparse.Namespace(
            retained_report=str(report_path),
            eval_dataset=str(eval_path),
            source_manifest="",
            output_dir=str(output_dir),
            objective="test",
            repeat=2,
            distill_loss_weight=1.0,
            allow_preview_fallback=False,
            only_content_wins=True,
            pointerize_candidates=True,
            only_pointer_family="materialize",
        )
    )
    rows = [
        json.loads(line)
        for line in Path(manifest["train_dataset_path"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(rows) == 2
    assert all("<AK_COPY_MATERIALIZE_CANDIDATE_1>" in row["decoder_text"] for row in rows)
