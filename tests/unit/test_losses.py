import torch

from ternarystem.losses import global_sdr


def test_global_sdr_improves_for_better_estimate():
    target = torch.randn(2, 4, 2, 100)
    assert global_sdr(target * 0.9, target) > global_sdr(target * 0.5, target)
