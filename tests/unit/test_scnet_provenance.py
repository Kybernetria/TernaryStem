import hashlib
from pathlib import Path

from ternarystem.models import SCNET_EXPECTED_TRAINABLE_PARAMETERS, SCNetConfig, SCNetSystem

ROOT = Path(__file__).parents[2]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_vendored_scnet_sources_and_license_match_pinned_blobs():
    assert sha256(ROOT / "third_party/SCNet/LICENSE") == (
        "0bdf1b69335198118ab16cfc50d337b496b8c6d90e83beeaba4643781ab62513"
    )
    assert sha256(ROOT / "src/ternarystem/models/_vendor/scnet/SCNet.py") == (
        "85a15ea5d28285a0cf0a24d6266a28d043c5a655d47aa41684ef256d84e7bc4a"
    )
    assert sha256(ROOT / "src/ternarystem/models/_vendor/scnet/separation.py") == (
        "43402dc6579436d3b5abb921990572684beed8fa10b377a112892b438f40713b"
    )


def test_standard_scnet_exact_trainable_parameter_convention():
    model = SCNetSystem(SCNetConfig())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    assert trainable == total == SCNET_EXPECTED_TRAINABLE_PARAMETERS
