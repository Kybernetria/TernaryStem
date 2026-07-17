import torch

from ternarystem.quant import (
    ActivationFakeQuant,
    W4A8Linear,
    W8A8Linear,
    fake_symmetric_weight,
    fake_ternary,
    symmetric_weight_values,
    ternary_values,
)


def test_adaptive_scale_is_reconstruction_optimal():
    weight = torch.tensor([[1.0, -0.8, 0.1, 0.0], [0.2, -0.4, 2.0, -1.0]])
    quantized, scale, values = ternary_values(weight, zero_ratio=0.4)
    expected = (weight * values).sum(1, keepdim=True) / values.square().sum(1, keepdim=True).clamp_min(1)
    torch.testing.assert_close(scale, expected)
    torch.testing.assert_close(quantized, scale * values)
    assert set(values.unique().tolist()) <= {-1.0, 0.0, 1.0}


def test_fake_ternary_uses_identity_ste():
    weight = torch.randn(4, 8, requires_grad=True)
    fake_ternary(weight).sum().backward()
    torch.testing.assert_close(weight.grad, torch.ones_like(weight))


def test_symmetric_weight_quantization_group_scales_and_ranges():
    weight = torch.tensor([[1.0, -0.5, 0.1, 0.2, 8.0, -4.0, 0.3, 0.0]])
    quantized, scales, values = symmetric_weight_values(weight, bits=4, group_size=4)
    assert scales.shape == (1, 2)
    assert int(values.min()) >= -7 and int(values.max()) <= 7
    torch.testing.assert_close(quantized[:, :4], values[:, :4] * scales[:, :1])
    torch.testing.assert_close(quantized[:, 4:], values[:, 4:] * scales[:, 1:])


def test_uniform_weight_fake_quant_uses_identity_ste():
    weight = torch.randn(3, 7, requires_grad=True)
    fake_symmetric_weight(weight, bits=4, group_size=4).sum().backward()
    torch.testing.assert_close(weight.grad, torch.ones_like(weight))


def test_w4a8_and_w8a8_modules_keep_latent_weights_and_backpropagate():
    for layer in (W4A8Linear(7, 3, group_size=4), W8A8Linear(7, 3)):
        original = layer.weight.detach().clone()
        inputs = torch.randn(2, 7, requires_grad=True)
        layer(inputs).sum().backward()
        torch.testing.assert_close(layer.weight.detach(), original)
        assert layer.weight.grad is not None
        assert inputs.grad is not None


def test_activation_fake_quant_has_int8_grid_and_gradient():
    inputs = torch.linspace(-2, 2, 101, requires_grad=True)
    quantizer = ActivationFakeQuant(method="static", clip=2.0).eval()
    output = quantizer(inputs)
    scaled = output.detach() / (2.0 / 127)
    torch.testing.assert_close(scaled, scaled.round(), atol=1e-5, rtol=0)
    output.sum().backward()
    torch.testing.assert_close(inputs.grad, torch.ones_like(inputs))
