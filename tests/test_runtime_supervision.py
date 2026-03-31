from pathlib import Path
import os
import signal
import subprocess

from agent_kernel.runtime_supervision import atomic_write_text, terminate_process_tree


def test_atomic_write_text_replaces_existing_file(tmp_path):
    target = tmp_path / "state.json"
    target.write_text("old", encoding="utf-8")

    atomic_write_text(target, "new", encoding="utf-8")

    assert target.read_text(encoding="utf-8") == "new"
    leftovers = [path for path in tmp_path.iterdir() if path.name != "state.json"]
    assert leftovers == []


def test_terminate_process_tree_uses_process_group_and_escalates(monkeypatch):
    seen_signals: list[tuple[int, int]] = []

    class FakeProcess:
        def __init__(self):
            self.pid = 4321
            self._poll = None
            self.wait_calls = 0

        def poll(self):
            return self._poll

        def wait(self, timeout=None):
            self.wait_calls += 1
            if self.wait_calls == 1:
                raise subprocess.TimeoutExpired(cmd="fake", timeout=timeout)
            self._poll = -9
            return -9

        def terminate(self):
            raise AssertionError("terminate should not be used when killpg works")

        def kill(self):
            raise AssertionError("kill should not be used when killpg works")

    monkeypatch.setattr(os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(
        os,
        "killpg",
        lambda pgid, sig: seen_signals.append((pgid, sig)),
    )

    process = FakeProcess()
    terminate_process_tree(process, grace_seconds=0.01)

    assert seen_signals == [(4321, signal.SIGTERM), (4321, signal.SIGKILL)]
    assert process.poll() == -9
