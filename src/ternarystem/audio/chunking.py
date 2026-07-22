"""Chunked inference with weighted overlap-add."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor


def overlap_add(
    waveform: Tensor,
    separate: Callable[[Tensor], Tensor],
    chunk_samples: int,
    overlap: float = 0.5,
) -> Tensor:
    """Apply ``separate`` to chunks; callable returns ``[B, sources, C, N]``."""
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")
    hop = max(1, int(chunk_samples * (1 - overlap)))
    length = waveform.shape[-1]
    output = weight_sum = None
    for start in range(0, length, hop):
        valid = min(chunk_samples, length - start)
        chunk = torch.nn.functional.pad(waveform[..., start : start + valid], (0, chunk_samples - valid))
        estimate = separate(chunk)[..., :valid]
        window = torch.hann_window(
            chunk_samples + 2, device=waveform.device, dtype=estimate.dtype
        )[1:-1][:valid]
        # At long chunk sizes the first nonzero Hann coefficient can round to zero
        # in FP32. A representable floor keeps boundary samples reconstructible.
        window = window.clamp_min(torch.finfo(estimate.dtype).eps)
        if output is None:
            output = estimate.new_zeros(*estimate.shape[:-1], length)
            weight_sum = estimate.new_zeros(length)
        output[..., start : start + valid] += estimate * window
        weight_sum[start : start + valid] += window
        if start + valid >= length:
            break
    assert output is not None and weight_sum is not None
    return output / weight_sum.clamp_min(1e-8)
