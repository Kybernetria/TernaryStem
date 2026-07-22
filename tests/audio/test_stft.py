import torch

from ternarystem.audio import STFT, mixture_consistency, overlap_add


def test_stft_exact_reconstruction():
    torch.manual_seed(1)
    waveform = torch.rand(2, 2, 8192) * 2 - 1
    transform = STFT(n_fft=512, hop_length=128)
    reconstructed = transform.synthesis(transform.analysis(waveform), waveform.shape[-1])
    assert (waveform - reconstructed).abs().max() <= 2e-5


def test_mixture_consistency_sums_to_mixture():
    mixture = torch.randn(2, 2, 8, 6, dtype=torch.complex64)
    estimates = torch.randn(2, 4, 2, 8, 6, dtype=torch.complex64)
    projected = mixture_consistency(estimates, mixture)
    torch.testing.assert_close(projected.sum(1), mixture)


@torch.no_grad()
def test_overlap_add_identity():
    waveform = torch.randn(1, 2, 1500)
    result = overlap_add(waveform, lambda chunk: chunk.unsqueeze(1), 512)
    torch.testing.assert_close(result[:, 0], waveform, atol=1e-6, rtol=1e-6)


@torch.no_grad()
def test_overlap_add_preserves_boundaries_for_six_second_chunks():
    chunk_samples = 6 * 44100
    waveform = torch.randn(1, 2, chunk_samples + 17)
    result = overlap_add(waveform, lambda chunk: chunk.unsqueeze(1), chunk_samples)
    torch.testing.assert_close(result[:, 0], waveform, atol=1e-6, rtol=1e-6)
