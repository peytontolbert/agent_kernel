from pathlib import Path
import importlib.util
import json
import sys


def _load_downloader_module():
    repo_root = Path(__file__).resolve().parents[1]
    scripts_dir = repo_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    script_path = scripts_dir / "download_a8_benchmark_datasets.py"
    spec = importlib.util.spec_from_file_location("download_a8_benchmark_datasets", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_download_a8_status_reports_existing_dataset_and_missing_credentialed_source(tmp_path):
    module = _load_downloader_module()
    _write_json(tmp_path / "benchmarks/swe/data.json", [{"instance_id": "x"}])
    manifest = {
        "sources": [
            {
                "benchmark": "swe_bench_verified",
                "label": "SWE-Bench Verified",
                "kind": "huggingface_dataset",
                "local_path": "benchmarks/swe/data.json",
                "required_for_a8": True,
            },
            {
                "benchmark": "mle_bench",
                "label": "MLE-Bench",
                "kind": "git_repo",
                "local_path": "benchmarks/mle_bench/repo",
                "requires_credentials": True,
                "required_for_a8": True,
            },
        ]
    }
    manifest_path = tmp_path / "config/sources.json"
    status_path = tmp_path / "benchmarks/status.json"
    _write_json(manifest_path, manifest)

    result = module.download_a8_benchmark_datasets(
        root=tmp_path,
        source_manifest=manifest_path,
        output_status=status_path,
        dry_run=False,
    )

    by_benchmark = {source["benchmark"]: source for source in result["sources"]}
    assert by_benchmark["swe_bench_verified"]["status"] == "available"
    assert by_benchmark["swe_bench_verified"]["rows"] == 1
    assert by_benchmark["mle_bench"]["status"] == "needs_credentials"
    assert status_path.exists()


def test_download_a8_status_skips_large_dataset_sources_without_flag(tmp_path):
    module = _load_downloader_module()
    manifest_path = tmp_path / "config/sources.json"
    _write_json(
        manifest_path,
        {
            "sources": [
                {
                    "benchmark": "codeforces",
                    "label": "Codeforces",
                    "kind": "huggingface_dataset",
                    "dataset_id": "example/large",
                    "local_path": "benchmarks/codeforces/info.json",
                    "large": True,
                    "required_for_a8": True,
                }
            ]
        },
    )

    result = module.download_a8_benchmark_datasets(
        root=tmp_path,
        source_manifest=manifest_path,
        output_status=tmp_path / "status.json",
    )

    source = result["sources"][0]
    assert source["status"] == "missing"
    assert source["last_action"]["action"] == "skipped_large_source"


def test_downloader_cli_writes_status(tmp_path, monkeypatch, capsys):
    module = _load_downloader_module()
    manifest_path = tmp_path / "config/sources.json"
    status_path = tmp_path / "status.json"
    _write_json(manifest_path, {"sources": []})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download_a8_benchmark_datasets.py",
            "--root",
            str(tmp_path),
            "--source-manifest",
            str(manifest_path),
            "--output-status",
            str(status_path),
        ],
    )

    module.main()

    assert status_path.exists()
    assert "status_counts" in capsys.readouterr().out
