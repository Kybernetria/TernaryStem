import copy

import torch

from ternarystem.models import SCNetConfig, SCNetSystem
from ternarystem.models._vendor.scnet.SCNet import SCNet as PinnedUpstreamSCNet


def tiny_config():
    return SCNetConfig(
        dims=(4, 8),
        nfft=64,
        hop_size=16,
        win_size=64,
        band_SR=(0.25, 0.375, 0.375),
        band_stride=(1, 2, 4),
        band_kernel=(3, 2, 4),
        conv_depths=(1, 1, 1),
        compress=4,
        num_dplayer=2,
    )


def test_adapter_core_is_synthetic_upstream_equivalent():
    config = tiny_config()
    torch.manual_seed(19)
    upstream = PinnedUpstreamSCNet(**config.upstream_kwargs()).eval()
    adapter = SCNetSystem(config).eval()
    adapter.core.load_state_dict(copy.deepcopy(upstream.state_dict()), strict=True)
    waveform = torch.randn(1, 2, 512)
    with torch.inference_mode():
        expected = upstream(waveform)
        actual = adapter.upstream_forward(waveform)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_pinned_scnet_loss_is_finite_and_decreases_on_fixed_tiny_example():
    torch.manual_seed(21)
    model = SCNetSystem(tiny_config())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    mixture = torch.randn(1, 2, 512) * 0.05
    targets = mixture[:, None].expand(-1, 4, -1, -1) / 4
    losses = []
    for _ in range(4):
        optimizer.zero_grad(set_to_none=True)
        estimates = model.training_estimates(mixture)
        loss = model.training_loss(estimates, targets)
        assert torch.isfinite(loss)
        losses.append(float(loss.detach()))
        loss.backward()
        optimizer.step()
    assert losses[-1] < losses[0]


def test_adapter_reorders_sources_and_applies_exact_waveform_consistency():
    model = SCNetSystem(tiny_config()).eval()
    waveform = torch.randn(1, 2, 512)
    with torch.inference_mode():
        upstream = model.upstream_forward(waveform)
        adapted = model(waveform)
    reordered = upstream[:, (3, 0, 1, 2)]
    expected = reordered + (waveform - reordered.sum(1)).unsqueeze(1) / 4
    torch.testing.assert_close(adapted, expected)
    torch.testing.assert_close(adapted.sum(1), waveform, atol=1e-6, rtol=1e-6)
