"""Streaming MUSDB18-HQ chunks with deterministic dynamic stem remixing."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import IterableDataset, get_worker_info

STEMS = ("vocals", "drums", "bass", "other")


@dataclass(frozen=True)
class StemSampleSpec:
    track: str
    stem: str
    start: int
    gain: float
    swap_channels: bool
    invert_polarity: bool


@dataclass(frozen=True)
class SampleSpec:
    sample_index: int
    epoch: int
    anchor_track: str
    stems: tuple[StemSampleSpec, ...]

    @property
    def stable_id(self) -> str:
        encoded = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def generate_sample_spec(
    *,
    track_names: list[str],
    frame_counts: dict[str, int],
    chunk_samples: int,
    seed: int,
    epoch: int,
    sample_index: int,
    remix: bool,
    augment: bool,
) -> SampleSpec:
    """Generate deterministic sample metadata without opening or decoding audio."""
    rng = random.Random(seed + epoch * 1_000_003 + sample_index * 97_409)
    anchor = rng.choice(track_names)
    aligned_start = None
    if not remix:
        frames = frame_counts[anchor]
        aligned_start = rng.randrange(max(1, frames - chunk_samples + 1))
    stems = []
    for stem in STEMS:
        track = rng.choice(track_names) if remix else anchor
        start = (
            rng.randrange(max(1, frame_counts[track] - chunk_samples + 1))
            if aligned_start is None
            else aligned_start
        )
        gain = 10 ** (rng.uniform(-3.0, 3.0) / 20.0) if augment else 1.0
        swap = rng.random() < 0.5 if augment else False
        invert = rng.random() < 0.5 if augment else False
        stems.append(StemSampleSpec(track, stem, start, gain, swap, invert))
    return SampleSpec(sample_index, epoch, anchor, tuple(stems))


class MUSDBChunkDataset(IterableDataset):
    def __init__(
        self,
        root: str | Path,
        track_names: list[str],
        chunk_samples: int,
        epoch_chunks: int,
        seed: int = 20250218,
        remix: bool = True,
        augment: bool = True,
        include_metadata: bool = False,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.track_names = list(track_names)
        self.chunk_samples = chunk_samples
        self.epoch_chunks = epoch_chunks
        self.seed = seed
        self.remix = remix
        self.augment = augment
        self.include_metadata = include_metadata
        self.epoch = 0
        if not self.track_names:
            raise ValueError("track_names cannot be empty")
        self.frame_counts = {}
        for name in self.track_names:
            frames = set()
            for stem in STEMS:
                path = self.root / name / f"{stem}.wav"
                if not path.is_file():
                    raise FileNotFoundError(path)
                frames.add(sf.info(path).frames)
            if len(frames) != 1:
                raise ValueError(f"stem lengths disagree for track: {name}")
            self.frame_counts[name] = frames.pop()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def sample_spec(self, sample_index: int) -> SampleSpec:
        return generate_sample_spec(
            track_names=self.track_names,
            frame_counts=self.frame_counts,
            chunk_samples=self.chunk_samples,
            seed=self.seed,
            epoch=self.epoch,
            sample_index=sample_index,
            remix=self.remix,
            augment=self.augment,
        )

    def ordered_sample_metadata(self) -> list[dict]:
        return [
            {
                "sample_id": spec.stable_id,
                "sample_index": spec.sample_index,
                "track": spec.anchor_track,
                "start": spec.stems[0].start,
            }
            for spec in (self.sample_spec(index) for index in range(self.epoch_chunks))
        ]

    def _read(self, spec: StemSampleSpec) -> np.ndarray:
        path = self.root / spec.track / f"{spec.stem}.wav"
        info = sf.info(path)
        if info.samplerate != 44100 or info.channels != 2:
            raise ValueError(f"expected stereo 44.1 kHz audio: {path}")
        audio, _ = sf.read(
            path,
            start=spec.start,
            frames=self.chunk_samples,
            dtype="float32",
            always_2d=True,
        )
        if len(audio) < self.chunk_samples:
            audio = np.pad(audio, ((0, self.chunk_samples - len(audio)), (0, 0)))
        source = audio.T.copy() * spec.gain
        if spec.swap_channels:
            source = source[::-1].copy()
        if spec.invert_polarity:
            source *= -1
        return source

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        workers = worker.num_workers if worker else 1
        for sample_index in range(worker_id, self.epoch_chunks, workers):
            spec = self.sample_spec(sample_index)
            stacked = torch.from_numpy(np.stack([self._read(stem) for stem in spec.stems]))
            if self.include_metadata:
                metadata = {
                    "sample_id": spec.stable_id,
                    "sample_index": spec.sample_index,
                    "track": spec.anchor_track,
                    "start": spec.stems[0].start,
                }
                yield stacked.sum(0), stacked, metadata
            else:
                yield stacked.sum(0), stacked
