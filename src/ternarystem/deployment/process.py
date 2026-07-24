"""Linux PID/PGID identity and bounded process-group supervision."""

from __future__ import annotations

import json
import os
import signal
import time
from pathlib import Path

from ternarystem.training import atomic_json_save


def process_identity(pid: int) -> dict:
    stat_path = Path(f"/proc/{pid}/stat")
    cmdline_path = Path(f"/proc/{pid}/cmdline")
    if not stat_path.is_file():
        raise ValueError("process does not exist")
    fields = stat_path.read_text(encoding="utf-8").split()
    if len(fields) < 22:
        raise ValueError("process stat is malformed")
    return {
        "pid": pid,
        "pgid": int(fields[4]),
        "session_id": int(fields[5]),
        "start_ticks": int(fields[21]),
        "cmdline": cmdline_path.read_bytes().replace(b"\0", b" ").decode(errors="replace").strip(),
    }


def record_process(pid: int, path: str | Path, required_token: str) -> dict:
    identity = process_identity(pid)
    if identity["pgid"] != pid or identity["session_id"] != pid:
        raise ValueError("supervised leader must own its process group and session")
    if required_token not in identity["cmdline"]:
        raise ValueError("supervised process command identity does not match")
    identity["required_token"] = required_token
    atomic_json_save(identity, path)
    return identity


def verify_process(path: str | Path) -> dict:
    try:
        recorded = json.loads(Path(path).read_text(encoding="utf-8"))
        current = process_identity(int(recorded["pid"]))
    except (OSError, KeyError, TypeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("process identity is missing, corrupt, stale, or exited") from error
    for key in ("pid", "pgid", "session_id", "start_ticks"):
        if current[key] != recorded.get(key):
            raise ValueError("process identity no longer matches recorded leader")
    token = recorded.get("required_token")
    if not isinstance(token, str) or token not in current["cmdline"]:
        raise ValueError("process command identity no longer matches")
    if current["pgid"] != current["pid"] or current["session_id"] != current["pid"]:
        raise ValueError("recorded process does not own a dedicated session")
    return current


def _group_members(pgid: int) -> list[int]:
    members = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text(encoding="utf-8").split()
            if int(fields[4]) == pgid and fields[2] != "Z":
                members.append(int(entry.name))
        except (OSError, ValueError, IndexError):
            continue
    return members


def _orphaned_group_identity(path: str | Path) -> dict:
    try:
        recorded = json.loads(Path(path).read_text(encoding="utf-8"))
        pid = int(recorded["pid"])
        pgid = int(recorded["pgid"])
        session_id = int(recorded["session_id"])
    except (OSError, KeyError, TypeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("orphaned process identity is invalid") from error
    if Path(f"/proc/{pid}").exists():
        raise ValueError("leader PID still exists but its identity does not match")
    members = _group_members(pgid)
    if not members:
        raise ValueError("recorded process group has already exited")
    for member in members:
        fields = Path(f"/proc/{member}/stat").read_text(encoding="utf-8").split()
        if int(fields[5]) != session_id:
            raise ValueError("orphaned process group contains a foreign session")
    return recorded


def terminate_verified_group(path: str | Path, timeout_seconds: float = 10.0) -> None:
    try:
        identity = verify_process(path)
    except ValueError:
        identity = _orphaned_group_identity(path)
    pgid = identity["pgid"]
    if pgid in {0, 1, os.getpgrp()}:
        raise ValueError("refusing to signal an unsafe process group")
    os.killpg(pgid, signal.SIGTERM)
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not _group_members(pgid):
            return
        time.sleep(0.05)
    if _group_members(pgid):
        os.killpg(pgid, signal.SIGKILL)
    kill_deadline = time.monotonic() + min(timeout_seconds, 2.0)
    while time.monotonic() < kill_deadline:
        if not _group_members(pgid):
            return
        time.sleep(0.05)
    raise ValueError("supervised process group did not exit after SIGKILL")
