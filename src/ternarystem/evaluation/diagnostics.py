"""Development-only separation diagnostics (not museval/BSSEval)."""

from __future__ import annotations

import torch
from torch import Tensor

STEM_NAMES = ("vocals", "drums", "bass", "other")


class DevelopmentDiagnostics:
    """Accumulate chunk-level global SDR and waveform L1 by source.

    SDR follows the project's training diagnostic definition: each source/chunk is
    reduced over channel and time, then values are averaged. It is never BSSEval.
    """

    def __init__(self, sources: int, eps: float = 1e-8) -> None:
        self.sources = sources
        self.eps = eps
        self._sdr_sum = torch.zeros(sources, dtype=torch.float64)
        self._baseline_sdr_sum = torch.zeros(sources, dtype=torch.float64)
        self._l1_sum = torch.zeros(sources, dtype=torch.float64)
        self._baseline_l1_sum = torch.zeros(sources, dtype=torch.float64)
        self._count = 0

    def update(self, estimates: Tensor, targets: Tensor, mixture: Tensor) -> None:
        if estimates.shape != targets.shape or estimates.ndim != 4:
            raise ValueError("estimates and targets must have shape [batch, sources, channels, time]")
        if estimates.shape[1] != self.sources or mixture.shape != (
            estimates.shape[0],
            estimates.shape[2],
            estimates.shape[3],
        ):
            raise ValueError("mixture or source shape does not match estimates")
        baseline = mixture.unsqueeze(1) / self.sources
        dimensions = (2, 3)
        signal = targets.double().square().sum(dim=dimensions)

        def sdr(values: Tensor) -> Tensor:
            error = (targets.double() - values.double()).square().sum(dim=dimensions)
            return 10 * torch.log10((signal + self.eps) / (error + self.eps))

        self._sdr_sum += sdr(estimates).sum(0).cpu()
        self._baseline_sdr_sum += sdr(baseline).sum(0).cpu()
        self._l1_sum += (targets - estimates).abs().double().mean(dim=dimensions).sum(0).cpu()
        self._baseline_l1_sum += (
            (targets - baseline).abs().double().mean(dim=dimensions).sum(0).cpu()
        )
        self._count += estimates.shape[0]

    def compute(self) -> dict:
        if self._count == 0:
            raise ValueError("no development diagnostic samples were accumulated")
        names = [*STEM_NAMES[: self.sources]]
        names.extend(f"source_{index}" for index in range(len(names), self.sources))

        def values(tensor: Tensor) -> dict[str, float]:
            return {name: float(value / self._count) for name, value in zip(names, tensor)}

        per_stem_sdr = values(self._sdr_sum)
        baseline_sdr = values(self._baseline_sdr_sum)
        return {
            "label": "development diagnostics (not BSSEval)",
            "global_sdr": sum(per_stem_sdr.values()) / self.sources,
            "per_stem_global_sdr": per_stem_sdr,
            "per_stem_waveform_l1": values(self._l1_sum),
            "equal_share_baseline": {
                "definition": "mixture / number_of_sources",
                "global_sdr": sum(baseline_sdr.values()) / self.sources,
                "per_stem_global_sdr": baseline_sdr,
                "per_stem_waveform_l1": values(self._baseline_l1_sum),
            },
        }
