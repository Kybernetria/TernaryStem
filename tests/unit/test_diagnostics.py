import torch

from ternarystem.evaluation import DevelopmentDiagnostics
from ternarystem.evaluation.comparison import summarize_record


def test_development_diagnostics_include_stems_and_equal_share_baseline():
    mixture = torch.tensor([[[2.0, 2.0], [2.0, 2.0]]])
    targets = mixture[:, None].repeat(1, 4, 1, 1) / 4
    estimates = targets.clone()
    diagnostics = DevelopmentDiagnostics(4)
    diagnostics.update(estimates, targets, mixture)
    result = diagnostics.compute()
    assert result["label"] == "development diagnostics (not BSSEval)"
    assert set(result["per_stem_global_sdr"]) == {"vocals", "drums", "bass", "other"}
    assert result["global_sdr"] > 80
    assert result["equal_share_baseline"]["global_sdr"] > 80
    assert result["per_stem_waveform_l1"]["vocals"] == 0


def test_comparison_handles_new_and_legacy_records_without_calling_sdr_bsseval():
    record = {
        "config": {"model": {"output_parameterization": "complex_mask"}, "quant": {}},
        "training": [
            {
                "epoch": 0,
                "validation_global_sdr": 1.0,
                "validation_development_diagnostics": {
                    "global_sdr": 1.0,
                    "per_stem_global_sdr": {"vocals": 1.0},
                    "equal_share_baseline": {"global_sdr": 0.25},
                },
            },
            {"epoch": 1, "validation_global_sdr": 0.5},
        ],
    }
    summary = summarize_record(record, "mask")
    assert summary["output_parameterization"] == "complex_mask"
    assert summary["best"]["model_minus_equal_share_db"] == 0.75
    assert summary["final"]["equal_share_development_global_sdr"] is None
    assert "not BSSEval" in summary["metric_label"]
