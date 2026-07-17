from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf
import torch

from ternarystem.models import Separator, SeparatorConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Separate a stereo WAV into four stems")
    parser.add_argument("input", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("separated"))
    args = parser.parse_args()
    audio, sample_rate = sf.read(args.input, always_2d=True, dtype="float32")
    if sample_rate != 44100 or audio.shape[1] != 2:
        raise SystemExit("input must be stereo 44.1 kHz audio")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = SeparatorConfig(**payload["config"])
    model = Separator(config).eval()
    model.load_state_dict(payload["state_dict"])
    with torch.inference_mode():
        estimates = model(torch.from_numpy(audio.T).unsqueeze(0))[0]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, stem in zip(("vocals", "drums", "bass", "other"), estimates):
        sf.write(args.output_dir / f"{name}.wav", stem.T.numpy(), sample_rate)
