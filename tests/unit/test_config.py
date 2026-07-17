from pathlib import Path

import pytest

from ternarystem.config import load_config, model_config


@pytest.mark.parametrize(
    "name", ["fp32", "fp32_medium", "ternary_qat", "w4a8", "w8a8", "mixed"]
)
def test_remote_smoke_configs_resolve_with_fp32_boundaries(name):
    config = load_config(Path("configs/smoke") / f"{name}.yaml")
    resolved = model_config(config)
    assert resolved.frequency_bins <= resolved.n_fft // 2 + 1
    assert resolved.precision_for("projections", "input_projection") == "fp32"
    assert resolved.precision_for("projections", "output_projection") == "fp32"


def test_removed_global_quantized_flag_is_rejected():
    with pytest.raises(ValueError, match="layer_precisions"):
        model_config({"model": {"channels": [4], "quantized": True}})
