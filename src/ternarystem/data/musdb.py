"""Streaming MUSDB18-HQ chunks with deterministic dynamic stem remixing."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import IterableDataset, get_worker_info

STEMS = ("vocals", "drums", "bass", "other")


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
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.track_names = list(track_names)
        self.chunk_samples = chunk_samples
        self.epoch_chunks = epoch_chunks
        self.seed = seed
        self.remix = remix
        self.augment = augment
        self.epoch = 0
        if not self.track_names:
            raise ValueError("track_names cannot be empty")
        for name in self.track_names:
            for stem in STEMS:
                if not (self.root / name / f"{stem}.wav").is_file():
                    raise FileNotFoundError(self.root / name / f"{stem}.wav")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _read(
        self, name: str, stem: str, rng: random.Random, start: int | None = None
    ) -> np.ndarray:
        path = self.root / name / f"{stem}.wav"
        info = sf.info(path)
        if info.samplerate != 44100 or info.channels != 2:
            raise ValueError(f"expected stereo 44.1 kHz audio: {path}")
        if start is None:
            start = rng.randrange(max(1, info.frames - self.chunk_samples + 1))
        audio, _ = sf.read(
            path, start=start, frames=self.chunk_samples, dtype="float32", always_2d=True
        )
        if len(audio) < self.chunk_samples:
            audio = np.pad(audio, ((0, self.chunk_samples - len(audio)), (0, 0)))
        return audio.T.copy()

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        workers = worker.num_workers if worker else 1
        # Derive each sample from its global index so worker count does not change
        # the canonical training/validation examples.
        for sample_index in range(worker_id, self.epoch_chunks, workers):
            rng = random.Random(
                self.seed + self.epoch * 1_000_003 + sample_index * 97_409
            )
            anchor = rng.choice(self.track_names)
            aligned_start = None
            if not self.remix:
                frames = sf.info(self.root / anchor / f"{STEMS[0]}.wav").frames
                aligned_start = rng.randrange(max(1, frames - self.chunk_samples + 1))
            sources = []
            for stem in STEMS:
                track = rng.choice(self.track_names) if self.remix else anchor
                source = self._read(track, stem, rng, aligned_start)
                if self.augment:
                    source *= 10 ** (rng.uniform(-3.0, 3.0) / 20.0)
                    if rng.random() < 0.5:
                        source = source[::-1].copy()
                    if rng.random() < 0.5:
                        source *= -1
                sources.append(source)
            stacked = torch.from_numpy(np.stack(sources))
            yield stacked.sum(0), stacked
