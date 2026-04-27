from pathlib import Path
import importlib.util
import json
import sys


def _load_predictions_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "prepare_swe_bench_predictions.py"
    spec = importlib.util.spec_from_file_location("prepare_swe_bench_predictions", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _patch() -> str:
    return (
        "diff --git a/pkg/file.py b/pkg/file.py\n"
        "--- a/pkg/file.py\n"
        "+++ b/pkg/file.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )


def test_validate_swe_predictions_accepts_complete_records():
    module = _load_predictions_module()

    failures = module.validate_swe_predictions(
        [
            {
                "instance_id": "repo__pkg-1",
                "model_name_or_path": "agentkernel",
                "model_patch": _patch(),
            }
        ]
    )

    assert failures == []


def test_validate_swe_predictions_rejects_duplicate_and_empty_patch():
    module = _load_predictions_module()

    failures = module.validate_swe_predictions(
        [
            {"instance_id": "repo__pkg-1", "model_name_or_path": "agentkernel", "model_patch": _patch()},
            {"instance_id": "repo__pkg-1", "model_name_or_path": "agentkernel", "model_patch": ""},
        ]
    )

    assert "duplicate instance_id: repo__pkg-1" in failures
    assert "record 2 field model_patch must be a non-empty string" in failures


def test_build_swe_predictions_from_manifest_reads_patch_path(tmp_path):
    module = _load_predictions_module()
    patch_path = tmp_path / "patch.diff"
    patch_path.write_text(_patch(), encoding="utf-8")

    records = module.build_swe_predictions_from_manifest(
        {
            "base_dir": str(tmp_path),
            "predictions": [
                {
                    "instance_id": "repo__pkg-1",
                    "model_name_or_path": "agentkernel",
                    "patch_path": "patch.diff",
                }
            ],
        }
    )

    assert records == [
        {
            "instance_id": "repo__pkg-1",
            "model_name_or_path": "agentkernel",
            "model_patch": _patch(),
        }
    ]


def test_build_swe_predictions_from_nested_prediction_manifest(tmp_path):
    module = _load_predictions_module()
    patch_path = tmp_path / "patch.diff"
    patch_path.write_text(_patch(), encoding="utf-8")

    records = module.build_swe_predictions_from_manifest(
        {
            "report_kind": "swe_bench_prediction_task_manifest",
            "prediction_manifest": {
                "base_dir": str(tmp_path),
                "predictions": [
                    {
                        "instance_id": "repo__pkg-1",
                        "model_name_or_path": "agentkernel",
                        "patch_path": "patch.diff",
                    }
                ],
            },
        }
    )

    assert records[0]["instance_id"] == "repo__pkg-1"
    assert records[0]["model_patch"] == _patch()


def test_prepare_swe_bench_predictions_cli_build_and_validate(tmp_path, monkeypatch, capsys):
    module = _load_predictions_module()
    manifest_path = tmp_path / "manifest.json"
    output_path = tmp_path / "predictions.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "predictions": [
                    {
                        "instance_id": "repo__pkg-1",
                        "model_name_or_path": "agentkernel",
                        "model_patch": _patch(),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_swe_bench_predictions.py",
            "build",
            "--manifest-json",
            str(manifest_path),
            "--output-jsonl",
            str(output_path),
        ],
    )

    module.main()

    assert "prediction_count=1" in capsys.readouterr().out
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_swe_bench_predictions.py",
            "validate",
            "--predictions-jsonl",
            str(output_path),
        ],
    )
    module.main()
    assert f"verified_swe_predictions={output_path}" in capsys.readouterr().out
