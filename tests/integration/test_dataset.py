import numpy as np
import soundfile as sf
import torch

from ternarystem.data import MUSDBChunkDataset, STEMS


def test_streaming_remix_is_reproducible_and_consistent(tmp_path):
    for track_index, track in enumerate(("track-a", "track-b")):
        directory = tmp_path / track
        directory.mkdir()
        for stem_index, stem in enumerate(STEMS):
            audio = np.full((80, 2), 0.01 * (track_index + 1) * (stem_index + 1), np.float32)
            sf.write(directory / f"{stem}.wav", audio, 44100, subtype="FLOAT")
    kwargs = dict(
        root=tmp_path,
        track_names=["track-a", "track-b"],
        chunk_samples=64,
        epoch_chunks=2,
        seed=7,
        remix=True,
        augment=True,
    )
    first = MUSDBChunkDataset(**kwargs)
    second = MUSDBChunkDataset(**kwargs)
    mixture, sources = next(iter(first))
    other_mixture, other_sources = next(iter(second))
    assert mixture.shape == (2, 64)
    assert sources.shape == (4, 2, 64)
    torch.testing.assert_close(mixture, sources.sum(0))
    torch.testing.assert_close(mixture, other_mixture)
    torch.testing.assert_close(sources, other_sources)
    first.set_epoch(1)
    next_mixture, _ = next(iter(first))
    assert not torch.equal(mixture, next_mixture)
