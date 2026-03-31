from __future__ import annotations

from io import StringIO
from pathlib import Path
import importlib.util
import json
import os
import sys

from agent_kernel.modeling.tolbert.runtime_status import hybrid_runtime_status


def _load_script_module(name: str):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hybrid_runtime_status_reports_partial_torch_installation_cleanly() -> None:
    status = hybrid_runtime_status()

    assert isinstance(status.available, bool)
    assert isinstance(status.reason, str)
    if not status.available:
        assert status.reason


def test_train_hybrid_runtime_script_fails_closed_when_full_torch_is_unavailable(monkeypatch) -> None:
    module = _load_script_module("train_hybrid_tolbert_runtime.py")
    monkeypatch.setattr(module, "maybe_reexec_under_model_python", lambda **kwargs: False)
    monkeypatch.setattr(
        module,
        "hybrid_runtime_status",
        lambda: type(
            "Status",
            (),
            {
                "available": False,
                "reason": "missing full torch runtime",
                "to_dict": lambda self: {"available": False},
            },
        )(),
    )
    monkeypatch.setattr(sys, "argv", ["train_hybrid_tolbert_runtime.py"])
    err_stream = StringIO()
    monkeypatch.setattr(sys, "stderr", err_stream)

    try:
        module.main()
    except SystemExit as exc:
        assert int(exc.code) == 1
    else:  # pragma: no cover
        raise AssertionError("expected train_hybrid_tolbert_runtime.py to exit")

    payload = json.loads(err_stream.getvalue())
    assert payload["artifact_kind"] == "tolbert_hybrid_runtime_error"
    assert payload["available"] is False
    assert payload["runtime_status"]["available"] is False


def test_train_hybrid_runtime_script_prefers_retained_dataset_manifest_and_runtime_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_script_module("train_hybrid_tolbert_runtime.py")
    monkeypatch.setattr(module, "maybe_reexec_under_model_python", lambda **kwargs: False)
    dataset_path = tmp_path / "dataset" / "hybrid.jsonl"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_text("", encoding="utf-8")
    dataset_manifest_path = tmp_path / "dataset" / "hybrid.manifest.json"
    dataset_manifest_path.write_text(
        json.dumps(
            {
                "artifact_kind": "tolbert_hybrid_training_dataset",
                "dataset_path": str(dataset_path),
                "config": {"hidden_dim": 20, "d_state": 5, "sequence_length": 3},
            }
        ),
        encoding="utf-8",
    )
    seen: dict[str, object] = {}

    monkeypatch.setattr(
        module,
        "hybrid_runtime_status",
        lambda: type("Status", (), {"available": True, "reason": "", "to_dict": lambda self: {"available": True}})(),
    )

    def _fake_train_from_dataset(**kwargs):
        seen.update(kwargs)
        return {"artifact_kind": "tolbert_hybrid_runtime_bundle", "checkpoint_path": kwargs["runtime_paths"]["checkpoint_path"]}

    monkeypatch.setattr(module, "train_hybrid_runtime_from_dataset", _fake_train_from_dataset)
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_hybrid_tolbert_runtime.py", "--device", "cpu"],
    )
    monkeypatch.setenv("AGENTKERNEL_TOLBERT_DATASET_MANIFEST_PATH", str(dataset_manifest_path))
    monkeypatch.setenv("AGENTKERNEL_TOLBERT_TRAIN_CONFIG_PATH", str(tmp_path / "runtime" / "train_config.json"))
    monkeypatch.setenv(
        "AGENTKERNEL_TOLBERT_CHECKPOINT_OUTPUT_PATH",
        str(tmp_path / "runtime" / "checkpoints" / "tolbert_runtime.pt"),
    )
    monkeypatch.setenv(
        "AGENTKERNEL_TOLBERT_BUNDLE_MANIFEST_PATH",
        str(tmp_path / "runtime" / "manifests" / "tolbert_bundle.json"),
    )
    (tmp_path / "runtime").mkdir(parents=True, exist_ok=True)
    Path(os.environ["AGENTKERNEL_TOLBERT_TRAIN_CONFIG_PATH"]).write_text(
        json.dumps({"hidden_dim": 28, "d_state": 7, "sequence_length": 4}),
        encoding="utf-8",
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    assert Path(seen["dataset_path"]) == dataset_path
    assert seen["config"].hidden_dim == 28
    assert str(seen["runtime_paths"]["checkpoint_path"]).endswith("tolbert_runtime.pt")
    assert str(seen["runtime_paths"]["bundle_manifest_path"]).endswith("tolbert_bundle.json")
