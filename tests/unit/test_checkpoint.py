import copy

import pytest
import torch

from ternarystem.models import Separator, SeparatorConfig
from ternarystem.training import resume_training, warm_start_model


def tiny_config(**kwargs):
    values = {
        "channels": (4,),
        "tdf_bottleneck": 4,
        "n_fft": 32,
        "hop_length": 8,
        "frequency_bins": 16,
    }
    values.update(kwargs)
    return SeparatorConfig(**values)


def test_fp_checkpoint_warm_starts_qat_without_copying_optimizer_or_epoch():
    fp_model = Separator(tiny_config())
    payload = {"state_dict": copy.deepcopy(fp_model.state_dict()), "epoch": 9}
    qat_model = Separator(tiny_config(layer_precisions={"tdf_linear": "w4a8"}))
    missing = warm_start_model(qat_model, payload)
    assert missing
    assert all(".activation_quant." in key for key in missing)
    torch.testing.assert_close(
        qat_model.network.encoder[0][1].tdf.layers[0].weight,
        fp_model.network.encoder[0][1].tdf.layers[0].weight,
    )


def test_warm_start_rejects_missing_model_weight():
    model = Separator(tiny_config())
    state = copy.deepcopy(model.state_dict())
    del state["network.input_projection.weight"]
    with pytest.raises(ValueError, match="non-quantizer"):
        warm_start_model(model, {"state_dict": state})


def test_resume_restores_model_optimizer_and_next_epoch():
    source = Separator(tiny_config())
    source_optimizer = torch.optim.AdamW(source.parameters(), lr=1e-3)
    source(torch.randn(1, 2, 128)).square().mean().backward()
    source_optimizer.step()
    payload = {
        "state_dict": copy.deepcopy(source.state_dict()),
        "optimizer": copy.deepcopy(source_optimizer.state_dict()),
        "epoch": 2,
    }
    restored = Separator(tiny_config())
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=9e-3)
    assert resume_training(restored, restored_optimizer, payload) == 3
    for expected, actual in zip(source.parameters(), restored.parameters()):
        torch.testing.assert_close(actual, expected)
    assert restored_optimizer.param_groups[0]["lr"] == 1e-3
