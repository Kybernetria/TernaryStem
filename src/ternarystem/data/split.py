"""Frozen MUSDB18-HQ development split utilities."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from importlib.resources import files
from pathlib import Path


def _path() -> Path:
    return Path(str(files("ternarystem.data").joinpath("musdb18_split.json")))


def load_split() -> dict:
    return json.loads(_path().read_text(encoding="utf-8"))


def split_hash() -> str:
    return hashlib.sha256(_path().read_bytes()).hexdigest()


def validate_development_data_root(root: str | Path) -> Path:
    """Reject any path that could expose the official MUSDB test partition."""
    supplied = Path(root)
    if supplied.name.casefold() != "train":
        raise ValueError("data root must be the canonical MUSDB18-HQ train directory")
    if supplied.is_symlink() or any(parent.is_symlink() for parent in supplied.parents):
        raise ValueError("symlinked dataset paths are not allowed")
    resolved = supplied.resolve(strict=False)
    if resolved.name.casefold() != "train" or "test" in {
        part.casefold() for part in resolved.parts
    }:
        raise ValueError("official MUSDB test paths are forbidden")
    for sibling in resolved.parent.iterdir() if resolved.parent.is_dir() else ():
        if sibling.name.casefold() == "test":
            raise ValueError("official MUSDB test sibling must be absent from the training host")
    return resolved


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
