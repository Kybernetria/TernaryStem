"""Frozen MUSDB18-HQ development split utilities."""

from __future__ import annotations

import hashlib
import json
from importlib.resources import files
from pathlib import Path
from typing import Iterable


def _path() -> Path:
    return Path(str(files("ternarystem.data").joinpath("musdb18_split.json")))


def load_split() -> dict:
    return json.loads(_path().read_text(encoding="utf-8"))


def split_hash() -> str:
    return hashlib.sha256(_path().read_bytes()).hexdigest()


def validate_track_names(track_names: Iterable[str]) -> tuple[list[str], list[str]]:
    """Return the deterministic (train, validation) split, rejecting incomplete datasets."""
    names = sorted(set(track_names))
    validation = load_split()["validation"]
    missing = sorted(set(validation) - set(names))
    if missing:
        raise ValueError(f"dataset is missing validation tracks: {missing}")
    train = sorted(set(names) - set(validation))
    if len(names) != 100 or len(train) != 86:
        raise ValueError(f"expected 100 official train tracks (86 train), got {len(names)}")
    return train, validation
