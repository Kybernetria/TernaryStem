import copy

import pytest
import torch

from ternarystem.models import Separator, SeparatorConfig
from ternarystem.training import build_scheduler, resume_training, warm_start_model


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


def test_resume_restores_scheduler_and_warm_start_uses_fresh_schedule():
    source = Separator(tiny_config())
    optimizer = torch.optim.AdamW(source.parameters(), lr=1e-3)
    train_config = {
        "epochs": 10,
        "scheduler": {"name": "cosine", "t_max": 10, "eta_min": 1e-5},
    }
    scheduler = build_scheduler(optimizer, train_config)
    assert scheduler is not None
    optimizer.step()
    scheduler.step()
    payload = {
        "state_dict": copy.deepcopy(source.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "scheduler": copy.deepcopy(scheduler.state_dict()),
        "epoch": 0,
    }

    restored = Separator(tiny_config())
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=9e-3)
    restored_scheduler = build_scheduler(restored_optimizer, train_config)
    assert restored_scheduler is not None
    assert resume_training(restored, restored_optimizer, payload, restored_scheduler) == 1
    assert restored_scheduler.state_dict() == scheduler.state_dict()
    assert restored_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]

    fresh_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    fresh_scheduler = build_scheduler(fresh_optimizer, train_config)
    warm_start_model(restored, payload)
    assert fresh_scheduler is not None
    assert fresh_scheduler.last_epoch == 0
    assert fresh_optimizer.param_groups[0]["lr"] == 1e-3


def test_resume_requires_scheduler_state_when_scheduler_is_configured():
    model = Separator(tiny_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    with pytest.raises(ValueError, match="scheduler state"):
        resume_training(
            model,
            optimizer,
            {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": 0},
            scheduler,
        )
