import copy

import pytest
import torch

from ternarystem.models import (
    TERNARYSTEM_V2_EXPECTED_TRAINABLE_PARAMETERS,
    TernaryStemV2Config,
    TernaryStemV2System,
)


def tiny_config(**kwargs):
    values = {
        "channels": (4, 8, 12),
        "tdf_reduction": 4,
        "tdf_min_hidden": 4,
        "temporal_dilations": (1, 2),
        "temporal_kernel": 3,
        "n_fft": 64,
        "hop_length": 16,
        "frequency_bins": 33,
        "expected_trainable_parameters": None,
    }
    values.update(kwargs)
    return TernaryStemV2Config(**values)


def test_full_spectrum_odd_shapes_forward_backward_and_consistency():
    model = TernaryStemV2System(tiny_config())
    mixture = torch.randn(1, 2, 257, requires_grad=True)
    spectra = model.spectrograms(mixture)
    estimates = model.stft.synthesis(spectra, mixture.shape[-1])
    assert spectra.shape == (1, 4, 2, 33, 17)
    assert estimates.shape == (1, 4, 2, 257)
    torch.testing.assert_close(estimates.sum(1), mixture, atol=2e-5, rtol=2e-5)
    estimates.square().mean().backward()
    assert mixture.grad is not None
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_silence_is_exact_and_cartesian_masks_are_bounded():
    model = TernaryStemV2System(tiny_config()).eval()
    silence = torch.zeros(1, 2, 257)
    with torch.inference_mode():
        mixture_spectrum = model.stft.analysis(silence)
        parts = torch.view_as_real(mixture_spectrum).permute(0, 1, 4, 2, 3).flatten(1, 2)
        raw = model.network(parts)
        masks = torch.complex(torch.tanh(raw.real), torch.tanh(raw.imag))
        output = model(silence)
    assert masks.real.abs().max() <= 1
    assert masks.imag.abs().max() <= 1
    torch.testing.assert_close(output, torch.zeros_like(output), rtol=0, atol=0)


def test_temporal_stage_has_explicit_long_range_receptive_field():
    model = TernaryStemV2System(tiny_config(temporal_dilations=(1, 2, 4)))
    assert model.network.temporal.receptive_field_frames == 15


def test_standard_candidate_has_frozen_parameter_count():
    model = TernaryStemV2System()
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    assert trainable == TERNARYSTEM_V2_EXPECTED_TRAINABLE_PARAMETERS
    assert 2_000_000 <= trainable <= 3_000_000


def test_tiny_candidate_can_reduce_fixed_example_loss():
    torch.manual_seed(23)
    model = TernaryStemV2System(tiny_config(channels=(4,)))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    mixture = torch.randn(1, 2, 129) * 0.05
    target = mixture[:, None].expand(-1, 4, -1, -1) / 4
    losses = []
    for _ in range(8):
        optimizer.zero_grad(set_to_none=True)
        loss = (model(mixture) - target).square().mean()
        losses.append(float(loss.detach()))
        loss.backward()
        optimizer.step()
    assert losses[-1] < losses[0]


def test_checkpoint_state_round_trip():
    source = TernaryStemV2System(tiny_config())
    restored = TernaryStemV2System(tiny_config())
    restored.load_state_dict(copy.deepcopy(source.state_dict()), strict=True)
    mixture = torch.randn(1, 2, 257)
    with torch.inference_mode():
        torch.testing.assert_close(restored(mixture), source(mixture), rtol=0, atol=0)


def test_unused_v2_precision_selector_fails_closed():
    with pytest.raises(ValueError, match="unused layer precision selectors"):
        TernaryStemV2System(
            tiny_config(layer_precisions={"encoder.0.block.tfc.typo": "w8a8"})
        )
