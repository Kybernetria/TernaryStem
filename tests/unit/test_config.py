from pathlib import Path

import pytest

from ternarystem.config import load_config, model_config


@pytest.mark.parametrize(
    "name",
    [
        "fp32",
        "fp32_complex_mask",
        "fp32_medium",
        "htdemucs_distillation",
        "ternary_qat",
        "w4a8",
        "w8a8",
        "mixed",
    ],
)
def test_remote_smoke_configs_resolve_with_fp32_boundaries(name):
    config = load_config(Path("configs/smoke") / f"{name}.yaml")
    resolved = model_config(config)
    assert resolved.frequency_bins <= resolved.n_fft // 2 + 1
    assert resolved.precision_for("projections", "input_projection") == "fp32"
    assert resolved.precision_for("projections", "output_projection") == "fp32"


def test_matched_smoke_fp32_configs_only_differ_in_output_parameterization():
    direct = load_config("configs/smoke/fp32.yaml")
    mask = load_config("configs/smoke/fp32_complex_mask.yaml")
    direct["model"]["output_parameterization"] = "complex_mask"
    assert direct == mask


@pytest.mark.parametrize(
    "path",
    [
        "configs/colab/fp32_mask_medium.yaml",
        "configs/colab/fp32_mask_large_pilot.yaml",
    ],
)
def test_colab_fp32_candidates_use_complex_masks_and_fp32_boundaries(path):
    config = load_config(path)
    resolved = model_config(config)
    assert resolved.output_parameterization == "complex_mask"
    assert not resolved.layer_precisions
    assert resolved.precision_for("projections", "input_projection") == "fp32"
    assert resolved.precision_for("projections", "output_projection") == "fp32"


def test_matched_colab_distillation_configs_only_differ_in_enabled_flag():
    control = load_config("configs/colab/fp32_mask_control.yaml")
    distilled = load_config("configs/colab/htdemucs_distillation.yaml")
    control["distillation"]["enabled"] = True
    assert control == distilled


def test_matched_remote_fp32_configs_only_differ_in_output_parameterization():
    direct = load_config("configs/remote/fp32_direct.yaml")
    mask = load_config("configs/remote/fp32_complex_mask.yaml")
    direct_mode = direct["model"].pop("output_parameterization")
    mask_mode = mask["model"].pop("output_parameterization")
    assert direct_mode == "direct_estimate"
    assert mask_mode == "complex_mask"
    assert direct == mask


def test_removed_global_quantized_flag_is_rejected():
    with pytest.raises(ValueError, match="layer_precisions"):
        model_config({"model": {"channels": [4], "quantized": True}})
