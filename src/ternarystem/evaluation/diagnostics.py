"""Development-only separation diagnostics (not museval/BSSEval)."""

from __future__ import annotations

import torch
from torch import Tensor

STEM_NAMES = ("vocals", "drums", "bass", "other")


class DevelopmentDiagnostics:
    """Accumulate chunk-level global SDR and waveform L1 by source.

    Canonical global SDR sums signal/error energy across fixed chunks before taking
    the ratio. The historical mean-of-chunk SDR is retained separately. Neither is BSSEval.
    """

    def __init__(self, sources: int, eps: float = 1e-8) -> None:
        self.sources = sources
        self.eps = eps
        self._sdr_sum = torch.zeros(sources, dtype=torch.float64)
        self._baseline_sdr_sum = torch.zeros(sources, dtype=torch.float64)
        self._signal_energy = torch.zeros(sources, dtype=torch.float64)
        self._error_energy = torch.zeros(sources, dtype=torch.float64)
        self._baseline_error_energy = torch.zeros(sources, dtype=torch.float64)
        self._l1_sum = torch.zeros(sources, dtype=torch.float64)
        self._baseline_l1_sum = torch.zeros(sources, dtype=torch.float64)
        self._count = 0
        self._sample_ids: list[str] = []
        self._track_energies: dict[str, dict[str, Tensor]] = {}

    def update(
        self,
        estimates: Tensor,
        targets: Tensor,
        mixture: Tensor,
        sample_ids: list[str] | None = None,
        track_names: list[str] | None = None,
    ) -> None:
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

        estimate_error = (targets.double() - estimates.double()).square().sum(dim=dimensions)
        baseline_error = (targets.double() - baseline.double()).square().sum(dim=dimensions)

        def sdr(error: Tensor) -> Tensor:
            return 10 * torch.log10((signal + self.eps) / (error + self.eps))

        self._sdr_sum += sdr(estimate_error).sum(0).cpu()
        self._baseline_sdr_sum += sdr(baseline_error).sum(0).cpu()
        self._signal_energy += signal.sum(0).cpu()
        self._error_energy += estimate_error.sum(0).cpu()
        self._baseline_error_energy += baseline_error.sum(0).cpu()
        self._l1_sum += (targets - estimates).abs().double().mean(dim=dimensions).sum(0).cpu()
        self._baseline_l1_sum += (
            (targets - baseline).abs().double().mean(dim=dimensions).sum(0).cpu()
        )
        batch = estimates.shape[0]
        if (sample_ids is None) != (track_names is None):
            raise ValueError("sample_ids and track_names must be provided together")
        if sample_ids is not None and track_names is not None:
            if len(sample_ids) != batch or len(track_names) != batch:
                raise ValueError("development metadata length does not match batch")
            self._sample_ids.extend(sample_ids)
            for index, track in enumerate(track_names):
                energies = self._track_energies.setdefault(
                    track,
                    {
                        "signal": torch.zeros(self.sources, dtype=torch.float64),
                        "error": torch.zeros(self.sources, dtype=torch.float64),
                        "baseline_error": torch.zeros(self.sources, dtype=torch.float64),
                    },
                )
                energies["signal"] += signal[index].cpu()
                energies["error"] += estimate_error[index].cpu()
                energies["baseline_error"] += baseline_error[index].cpu()
        self._count += batch

    def compute(self) -> dict:
        if self._count == 0:
            raise ValueError("no development diagnostic samples were accumulated")
        names = [*STEM_NAMES[: self.sources]]
        names.extend(f"source_{index}" for index in range(len(names), self.sources))

        def values(tensor: Tensor) -> dict[str, float]:
            return {name: float(value / self._count) for name, value in zip(names, tensor)}

        per_stem_mean_chunk_sdr = values(self._sdr_sum)
        baseline_mean_chunk_sdr = values(self._baseline_sdr_sum)

        def energy_sdr(error: Tensor) -> dict[str, float]:
            ratio = 10 * torch.log10(
                (self._signal_energy + self.eps) / (error + self.eps)
            )
            return {name: float(value) for name, value in zip(names, ratio)}

        per_stem_global_sdr = energy_sdr(self._error_energy)
        baseline_global_sdr = energy_sdr(self._baseline_error_energy)
        track_energies = {}
        for track, energies in sorted(self._track_energies.items()):
            track_energies[track] = {
                kind: {name: float(value) for name, value in zip(names, tensor)}
                for kind, tensor in energies.items()
            }
        return {
            "label": "development diagnostics (not BSSEval)",
            "ordered_sample_ids": list(self._sample_ids),
            "per_track_float64_energies": track_energies,
            "float64_energies": {
                "signal": {name: float(value) for name, value in zip(names, self._signal_energy)},
                "error": {name: float(value) for name, value in zip(names, self._error_energy)},
                "baseline_error": {
                    name: float(value)
                    for name, value in zip(names, self._baseline_error_energy)
                },
            },
            "sdr_aggregation": "sum signal/error energy across chunks, then take ratio",
            "global_sdr": sum(per_stem_global_sdr.values()) / self.sources,
            "per_stem_global_sdr": per_stem_global_sdr,
            "mean_chunk_sdr": sum(per_stem_mean_chunk_sdr.values()) / self.sources,
            "per_stem_mean_chunk_sdr": per_stem_mean_chunk_sdr,
            "per_stem_waveform_l1": values(self._l1_sum),
            "equal_share_baseline": {
                "definition": "mixture / number_of_sources",
                "global_sdr": sum(baseline_global_sdr.values()) / self.sources,
                "per_stem_global_sdr": baseline_global_sdr,
                "mean_chunk_sdr": sum(baseline_mean_chunk_sdr.values()) / self.sources,
                "per_stem_mean_chunk_sdr": baseline_mean_chunk_sdr,
                "per_stem_waveform_l1": values(self._baseline_l1_sum),
            },
        }
