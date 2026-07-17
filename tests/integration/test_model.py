import torch

from ternarystem.models import Separator, SeparatorConfig


def test_separator_shapes_and_waveform_consistency():
    config = SeparatorConfig(
        channels=(4, 8), tdf_bottleneck=4, n_fft=64, hop_length=16, frequency_bins=32
    )
    model = Separator(config).eval()
    mixture = torch.randn(1, 2, 256)
    with torch.inference_mode():
        estimates = model(mixture)
    assert estimates.shape == (1, 4, 2, 256)
    torch.testing.assert_close(estimates.sum(1), mixture, atol=2e-5, rtol=2e-5)


def test_mixed_precision_model_backward_and_fp32_boundaries():
    config = SeparatorConfig(
        channels=(4, 8),
        tdf_bottleneck=4,
        n_fft=32,
        hop_length=8,
        frequency_bins=16,
        layer_precisions={
            "tdf_linear": "ternary",
            "bottleneck_conv": "w8a8",
            "encoder_conv": "w4a8",
            "decoder_conv": "w4a8",
        },
    )
    model = Separator(config)
    assert type(model.network.input_projection) is torch.nn.Conv2d
    assert type(model.network.output_projection) is torch.nn.Conv2d
    assert isinstance(model.network.encoder[0][1].tdf.layers[0], torch.nn.Linear)
    assert hasattr(model.network.encoder[0][1].tdf.layers[0], "activation_quant")
    mixture = torch.randn(1, 2, 128)
    model(mixture).square().mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)


def test_exact_layer_precision_overrides_family_and_boundary_default():
    config = SeparatorConfig(
        channels=(4,),
        tdf_bottleneck=4,
        n_fft=32,
        hop_length=8,
        frequency_bins=16,
        layer_precisions={
            "tdf_linear": "w8a8",
            "encoder.0.1.tdf.layers.0": "fp32",
            "input_projection": "w8a8",
        },
    )
    model = Separator(config)
    assert type(model.network.encoder[0][1].tdf.layers[0]) is torch.nn.Linear
    assert hasattr(model.network.encoder[0][1].tdf.layers[2], "activation_quant")
    assert hasattr(model.network.input_projection, "activation_quant")
    assert type(model.network.output_projection) is torch.nn.Conv2d
