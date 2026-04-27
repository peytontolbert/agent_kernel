from pathlib import Path
import importlib.util
import json
import sys


def _load_runner_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_autonomous_benchmark_harness.py"
    spec = importlib.util.spec_from_file_location("run_autonomous_benchmark_harness", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_prepare_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "prepare_autonomous_benchmark_harness.py"
    spec = importlib.util.spec_from_file_location("prepare_autonomous_benchmark_harness", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _harness_spec(output_name: str) -> dict:
    return {
        "spec_version": "asi_v1",
        "report_kind": "autonomous_benchmark_harness_spec",
        "benchmark": "codeforces",
        "autonomy_contract": {
            "operator_role": "launch_and_monitor_only",
            "selection_mode": "fixture",
            "kernel_owned_phases": ["write_artifact"],
            "prohibited_manual_interventions": ["manual packet editing"],
            "countable_evidence": ["verified fixture artifact"],
        },
        "phases": [
            {
                "name": "write_artifact",
                "kind": "command",
                "argv": [
                    sys.executable,
                    "-c",
                    f"from pathlib import Path; Path({output_name!r}).write_text('ok', encoding='utf-8')",
                ],
                "expected_outputs": [output_name],
            }
        ],
    }


def test_validate_harness_spec_accepts_command_phases():
    module = _load_runner_module()

    assert module.validate_harness_spec(_harness_spec("artifact.txt")) == []


def test_validate_harness_spec_rejects_unknown_kernel_owned_phase():
    module = _load_runner_module()
    spec = _harness_spec("artifact.txt")
    spec["autonomy_contract"]["kernel_owned_phases"] = ["missing_phase"]

    failures = module.validate_harness_spec(spec)

    assert "autonomy_contract.kernel_owned_phases references unknown phase: missing_phase" in failures


def test_check_harness_prerequisites_rejects_missing_account_gate(tmp_path):
    module = _load_runner_module()
    spec = _harness_spec("artifact.txt")
    spec["prerequisites"] = [
        {
            "blocking": True,
            "kind": "account",
            "name": "codeforces_account",
            "proof_path": str(tmp_path / "missing_account.json"),
            "required_env": ["CODEFORCES_HANDLE"],
            "satisfied_by": "env_or_proof",
        }
    ]

    failures = module.check_harness_prerequisites(spec, env={})

    assert failures == [
        "codeforces_account unsatisfied: required_env=CODEFORCES_HANDLE "
        f"proof_path={tmp_path / 'missing_account.json'} satisfied_by=env_or_proof"
    ]


def test_run_harness_spec_executes_phases_and_writes_log(tmp_path):
    module = _load_runner_module()
    log_path = tmp_path / "run_log.json"

    log = module.run_harness_spec(
        _harness_spec("artifact.txt"),
        cwd=tmp_path,
        log_json=log_path,
    )

    assert log["success"] is True
    assert (tmp_path / "artifact.txt").read_text(encoding="utf-8") == "ok"
    saved_log = json.loads(log_path.read_text(encoding="utf-8"))
    assert saved_log["phase_results"][0]["name"] == "write_artifact"
    assert saved_log["phase_results"][0]["returncode"] == 0


def test_run_harness_spec_exposes_active_phase_to_running_command(tmp_path):
    module = _load_runner_module()
    log_path = tmp_path / "run_log.json"
    spec = _harness_spec("artifact.txt")
    spec["phases"][0]["expected_outputs"] = ["artifact.txt", "observed_phase.txt"]
    spec["phases"][0]["argv"] = [
        sys.executable,
        "-c",
        (
            "import json\n"
            "from pathlib import Path\n"
            f"log_path = Path({str(log_path)!r})\n"
            "payload = json.loads(log_path.read_text(encoding='utf-8'))\n"
            "Path('observed_phase.txt').write_text(payload['active_phase']['name'], encoding='utf-8')\n"
            "Path('artifact.txt').write_text('ok', encoding='utf-8')\n"
        ),
    ]

    module.run_harness_spec(spec, cwd=tmp_path, log_json=log_path)

    saved_log = json.loads(log_path.read_text(encoding="utf-8"))
    assert (tmp_path / "observed_phase.txt").read_text(encoding="utf-8") == "write_artifact"
    assert "active_phase" not in saved_log
    assert saved_log["phase_results"][0]["pid"] > 0
    assert saved_log["phase_results"][0]["elapsed_seconds"] >= 0


def test_run_harness_spec_rejects_missing_required_input(tmp_path):
    module = _load_runner_module()
    spec = _harness_spec("artifact.txt")
    spec["phases"][0]["required_inputs"] = ["missing_input.txt"]
    log_path = tmp_path / "run_log.json"

    try:
        module.run_harness_spec(spec, cwd=tmp_path, log_json=log_path)
    except SystemExit as exc:
        assert "harness phase missing required inputs" in str(exc)
    else:
        raise AssertionError("missing required input should stop the harness")

    saved_log = json.loads(log_path.read_text(encoding="utf-8"))
    assert saved_log["failed_phase"] == "write_artifact"
    assert saved_log["phase_results"][0]["missing_inputs"] == [
        "write_artifact missing required input: missing_input.txt"
    ]


def test_harness_status_reports_initial_missing_input(tmp_path):
    module = _load_runner_module()
    spec = _harness_spec("artifact.txt")
    spec["phases"][0]["required_inputs"] = ["missing_input.txt"]

    status = module.harness_status(spec, cwd=tmp_path, env={})

    assert status["runnable"] is False
    assert status["initial_input_failures"] == ["write_artifact missing required input: missing_input.txt"]


def test_harness_status_allows_required_input_produced_by_prior_phase(tmp_path):
    module = _load_runner_module()
    spec = _harness_spec("artifact.txt")
    spec["autonomy_contract"]["kernel_owned_phases"] = ["write_artifact", "consume_artifact"]
    spec["phases"].append(
        {
            "name": "consume_artifact",
            "kind": "command",
            "argv": [sys.executable, "-c", "from pathlib import Path; Path('done.txt').write_text('ok')"],
            "required_inputs": ["artifact.txt"],
            "expected_outputs": ["done.txt"],
        }
    )

    status = module.harness_status(spec, cwd=tmp_path, env={})

    assert status["runnable"] is True
    assert status["initial_input_failures"] == []


def test_harness_status_reports_later_missing_external_input(tmp_path):
    module = _load_runner_module()
    spec = _harness_spec("artifact.txt")
    spec["autonomy_contract"]["kernel_owned_phases"] = ["write_artifact", "consume_external"]
    spec["phases"].append(
        {
            "name": "consume_external",
            "kind": "command",
            "argv": [sys.executable, "-c", "print('ok')"],
            "required_inputs": ["external_summary.json"],
        }
    )

    status = module.harness_status(spec, cwd=tmp_path, env={})

    assert status["runnable"] is False
    assert status["initial_input_failures"] == [
        "consume_external missing required input: external_summary.json"
    ]


def test_harness_status_preflight_reports_uncountable_evidence(tmp_path):
    module = _load_runner_module()
    spec = _harness_spec("artifact.txt")
    spec["phases"][0]["preflight_argv"] = [sys.executable, "-c", "raise SystemExit('bad evidence')"]

    without_preflight = module.harness_status(spec, cwd=tmp_path, env={}, run_preflight=False)
    with_preflight = module.harness_status(spec, cwd=tmp_path, env={}, run_preflight=True)

    assert without_preflight["runnable"] is True
    assert with_preflight["runnable"] is False
    assert with_preflight["preflight_failures"] == ["write_artifact preflight failed: bad evidence"]


def test_run_harness_cli_validates_spec(tmp_path, monkeypatch, capsys):
    module = _load_runner_module()
    harness_path = tmp_path / "harness.json"
    harness_path.write_text(json.dumps(_harness_spec("artifact.txt")), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_autonomous_benchmark_harness.py",
            "validate",
            "--harness-json",
            str(harness_path),
        ],
    )

    module.main()

    assert f"verified_autonomous_benchmark_harness={harness_path}" in capsys.readouterr().out


def test_run_harness_cli_status_reports_harness_dir(tmp_path, monkeypatch, capsys):
    module = _load_runner_module()
    harness_dir = tmp_path / "harnesses"
    harness_dir.mkdir()
    (harness_dir / "ok.json").write_text(json.dumps(_harness_spec("artifact.txt")), encoding="utf-8")
    missing_spec = _harness_spec("artifact.txt")
    missing_spec["phases"][0]["required_inputs"] = ["missing_input.txt"]
    (harness_dir / "blocked.json").write_text(json.dumps(missing_spec), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_autonomous_benchmark_harness.py",
            "status",
            "--harness-dir",
            str(harness_dir),
            "--cwd",
            str(tmp_path),
            "--json",
        ],
    )

    module.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["harness_count"] == 2
    assert {item["runnable"] for item in payload["statuses"]} == {True, False}


def test_run_harness_cli_status_preflight_blocks_invalid_phase(tmp_path, monkeypatch, capsys):
    module = _load_runner_module()
    harness_path = tmp_path / "harness.json"
    spec = _harness_spec("artifact.txt")
    spec["phases"][0]["preflight_argv"] = [sys.executable, "-c", "raise SystemExit('bad evidence')"]
    harness_path.write_text(json.dumps(spec), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_autonomous_benchmark_harness.py",
            "status",
            "--harness-json",
            str(harness_path),
            "--cwd",
            str(tmp_path),
            "--preflight",
            "--json",
        ],
    )

    module.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["statuses"][0]["runnable"] is False
    assert payload["statuses"][0]["preflight_failures"] == ["write_artifact preflight failed: bad evidence"]


def test_run_summary_only_harness_executes_adapter_and_writes_verified_packet(tmp_path):
    prepare_module = _load_prepare_module()
    runner_module = _load_runner_module()
    repo_root = Path(__file__).resolve().parents[1]
    summary_path = tmp_path / "codeforces_summary.json"
    packet_path = tmp_path / "codeforces_packet.json"
    log_path = tmp_path / "harness_log.json"
    account_proof_path = tmp_path / "codeforces_account.json"
    summary_path.write_text(json.dumps({"rating_equivalent": 3100}), encoding="utf-8")
    account_proof_path.write_text(json.dumps({"handle": "fixture"}), encoding="utf-8")
    run_spec = {
        "spec_version": "asi_v1",
        "report_kind": "a8_benchmark_run_spec",
        "benchmark": "codeforces",
        "ready_to_run": True,
        "prerequisites": [
            {
                "blocking": True,
                "kind": "account",
                "name": "codeforces_account",
                "proof_path": str(account_proof_path),
                "required_env": ["CODEFORCES_HANDLE"],
                "satisfied_by": "env_or_proof",
            }
        ],
        "runner": {"kind": "summary_only", "summary_source": str(summary_path)},
        "adapter": {
            "script": "scripts/run_a8_benchmark_adapter.py",
            "summary_json": str(summary_path),
            "output_packet_json": str(packet_path),
            "conservative_comparison_report": True,
        },
        "open_limits": ["fixture"],
    }
    harness = prepare_module.build_autonomous_harness_from_run_spec(run_spec, spec_path="fixture.json")

    status = runner_module.harness_status(harness, cwd=repo_root, env={}, run_preflight=True)
    log = runner_module.run_harness_spec(harness, cwd=repo_root, log_json=log_path)

    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    assert status["runnable"] is True
    assert log["success"] is True
    assert packet["report_kind"] == "a8_benchmark_result"
    assert packet["benchmark"] == "codeforces"
    assert packet["metrics"]["rating_equivalent"] == 3100
    assert packet["metrics"]["conservative_comparison_report"] is True
