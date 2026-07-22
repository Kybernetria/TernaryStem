"""Local JSON experiment records."""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch

from ternarystem.data import split_hash
from ternarystem.training.persistence import atomic_json_save


def _git(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def base_record(config: dict, seed: int, device: str | None = None) -> dict:
    status = _git("status", "--porcelain")
    cuda_available = torch.cuda.is_available()
    gpu_model = None
    if cuda_available:
        try:
            gpu_device = device if device is not None and device.startswith("cuda") else None
            gpu_model = torch.cuda.get_device_name(gpu_device)
        except (AssertionError, RuntimeError, ValueError):
            gpu_model = None
    return {
        "schema_version": 3,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "config": config,
        "seed": seed,
        "split_sha256": split_hash(),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "device": device,
            "cuda_available": cuda_available,
            "gpu_model": gpu_model,
        },
        "software": {
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
            "numpy": _package_version("numpy"),
            "soundfile": _package_version("soundfile"),
            "pyyaml": _package_version("PyYAML"),
            "demucs": _package_version("demucs"),
            "musdb": _package_version("musdb"),
            "museval": _package_version("museval"),
        },
        "quality": None,
        "runtime": None,
        "checkpoint_sha256": None,
        "checkpoint_hashes": {"latest": None, "best": None},
        "packed_model_sha256": None,
    }


def save_record(path: str | Path, record: dict) -> None:
    atomic_json_save(record, path)
