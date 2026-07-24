import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from ternarystem.deployment.process import record_process, terminate_verified_group, verify_process


def test_dedicated_process_group_is_verified_and_fully_terminated(tmp_path):
    process = subprocess.Popen(["sh", "-c", "sleep 30 & wait"], start_new_session=True)
    identity = tmp_path / "identity.json"
    try:
        record_process(process.pid, identity, "sleep 30")
        assert verify_process(identity)["pgid"] == process.pid
        terminate_verified_group(identity, timeout_seconds=1)
        process.wait(timeout=2)
        assert process.returncode == -15
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_vanished_leader_with_surviving_group_is_cleaned(tmp_path):
    child_file = tmp_path / "child.pid"
    process = subprocess.Popen(
        ["sh", "-c", f"sleep 30 & echo $! > {child_file}; sleep 0.3"],
        start_new_session=True,
    )
    identity = tmp_path / "identity.json"
    try:
        record_process(process.pid, identity, "sleep 30")
        process.wait(timeout=2)
        child = int(child_file.read_text())
        assert Path(f"/proc/{child}").exists()
        terminate_verified_group(identity, timeout_seconds=1)
        deadline = time.monotonic() + 2
        while Path(f"/proc/{child}").exists() and time.monotonic() < deadline:
            stat = Path(f"/proc/{child}/stat").read_text().split()
            if stat[2] == "Z":
                break
            time.sleep(0.05)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_term_resistant_group_is_escalated(tmp_path):
    process = subprocess.Popen(
        ["sh", "-c", "trap '' TERM; while :; do sleep 1; done"],
        start_new_session=True,
    )
    identity = tmp_path / "identity.json"
    try:
        record_process(process.pid, identity, "trap")
        time.sleep(0.1)
        terminate_verified_group(identity, timeout_seconds=0.1)
        process.wait(timeout=2)
        assert process.returncode == -9
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_deployment_launcher_closes_lock_before_spawn():
    text = Path("scripts/vast/deployment_2_start.sh").read_text(encoding="utf-8")
    assert text.index("exec 8>&-") < text.index("setsid nohup")


def test_stale_identity_refuses_to_signal_reused_or_unrelated_pid(tmp_path):
    process = subprocess.Popen(["sleep", "30"], start_new_session=True)
    identity = tmp_path / "identity.json"
    try:
        payload = record_process(process.pid, identity, "sleep 30")
        payload["start_ticks"] += 1
        identity.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="identity does not match"):
            terminate_verified_group(identity, timeout_seconds=0.1)
        assert process.poll() is None
        os.killpg(process.pid, 15)
        process.wait(timeout=2)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
