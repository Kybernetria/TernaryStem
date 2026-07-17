import torch

from ternarystem.export import (
    export_state_dict,
    pack_bitnet_i2s,
    pack_int4,
    pack_ternary,
    unpack_bitnet_i2s,
    unpack_int4,
    unpack_ternary,
)


def test_pack_round_trip_with_partial_byte():
    values = torch.tensor([-1, 0, 1, 1, 0, -1, 1], dtype=torch.int8)
    packed = pack_ternary(values)
    assert len(packed) == 2
    torch.testing.assert_close(unpack_ternary(packed, values.numel()), values)


def test_int4_round_trip_with_partial_byte():
    values = torch.tensor([-7, -1, 0, 1, 7], dtype=torch.int8)
    packed = pack_int4(values)
    assert len(packed) == 3
    torch.testing.assert_close(unpack_int4(packed, values.numel()), values)


def test_bitnet_i2s_interleaved_round_trip():
    values = (torch.arange(3 * 256).reshape(3, 256) % 3 - 1).to(torch.int8)
    packed = pack_bitnet_i2s(values)
    assert len(packed) == values.numel() // 4
    torch.testing.assert_close(unpack_bitnet_i2s(packed, 3, 256), values)


def test_mixed_export_is_deterministic_and_records_quant_metadata(tmp_path):
    state = {
        "network.input_projection.weight": torch.randn(2, 4, 1, 1),
        "network.encoder.0.1.tdf.layers.0.weight": torch.randn(3, 8),
        "network.encoder.0.1.tfc.layers.0.weight": torch.randn(2, 2, 3, 3),
        "network.output_projection.weight": torch.randn(8, 2, 1, 1),
    }
    precisions = {"tdf_linear": "w4a8", "bottleneck_conv": "w8a8"}
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    manifest = export_state_dict(state, first, layer_precisions=precisions, w4_group_size=4)
    export_state_dict(state, second, layer_precisions=precisions, w4_group_size=4)
    assert first.read_bytes() == second.read_bytes()
    assert first.with_suffix(".npz.json").read_bytes() == second.with_suffix(".npz.json").read_bytes()
    tensors = manifest["tensors"]
    assert tensors["network.input_projection.weight"]["kind"] == "fp32"
    assert tensors["network.encoder.0.1.tdf.layers.0.weight"]["bits"] == 4
    assert tensors["network.encoder.0.1.tfc.layers.0.weight"]["bits"] == 8
