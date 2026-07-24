# Pinned SCNet provenance and interface audit

## Source

- Authoritative repository: `https://github.com/starrytong/SCNet`
- Pinned commit: `5d95bf96b19c3eede63248d171efeca8e3abb948`
- Commit subject: `Merge pull request #33 from AdrLfv/fix/mono_audio`
- License: MIT, copyright 2024 starrytong; preserved at `third_party/SCNet/LICENSE`.
- SHA-256 of the pinned commit's sorted `git ls-tree -r --full-tree` text: `b0f5a840238d03f8fd9238d7cbda9967d9270339170dde293afa157610d98eb3`.

Vendored executable core files are intentionally limited to `scnet/SCNet.py` and
`scnet/separation.py`. They depend only on PyTorch. Project training, data, checkpoint,
evaluation, and inference orchestration are not copied from upstream. The vendored files
remain byte-identical to the pinned blobs; project behavior is implemented in a separate
adapter.

| Upstream file | SHA-256 |
|---|---|
| `LICENSE` | `0bdf1b69335198118ab16cfc50d337b496b8c6d90e83beeaba4643781ab62513` |
| `conf/config.yaml` | `bc16895f69551cd8becbba384ae33e6e615a4792649b866036884f4a9a4bb4b0` |
| `scnet/SCNet.py` | `85a15ea5d28285a0cf0a24d6266a28d043c5a655d47aa41684ef256d84e7bc4a` |
| `scnet/separation.py` | `43402dc6579436d3b5abb921990572684beed8fa10b377a112892b438f40713b` |
| `scnet/loss.py` | `ca8c79f4a52107b1ca37362fd4b727e133ead7d569c67954f3260829d1e4a14c` |
| `scnet/apply.py` | `68b60535ad50b9b6cb0da4e4a0e7a269ff1349d345de30e666bdbda3d8fc8a31` |
| `scnet/augment.py` | `f03328615927bf167ffd2e648bc022ef8cced6d3d157a5a281a73569d525113d` |
| `requirements.txt` | `5af27b6912eddb99793d94936f3ab53e344fb09bd139d75c0969c54086a821bd` |

No upstream checkpoint, audio, or dataset was downloaded.

## Frozen upstream core

The standard universal configuration at the pinned commit uses:

- sources in `drums, bass, other, vocals` order;
- stereo waveform input and direct waveform estimates;
- a 4,096-point STFT, 1,024-sample hop, 4,096-sample window, centered and normalized;
- real/imaginary STFT features flattened into four input channels;
- per-example feature mean/std normalization inside the model;
- sparse low/mid/high frequency down/up-sampling with proportions
  `0.175/0.392/0.433`, strides `1/4/16`, and kernels `3/4/16`;
- dimensions `4/32/64/128`, convolution depths `3/2/1`, compression 4;
- six alternating dual-path bidirectional-LSTM/FFT conversion layers;
- direct complex estimates followed by iSTFT, without a mask or mixture-consistency
  projection in the upstream core.

The upstream training recipe uses 11-second segments, track-level mixture-derived
normalization in the data loader, shift/remix/scale/channel/sign augmentations, Adam,
a spectral RMSE loss, mixed-precision autocast, and several EMAs. Deployment #2 does
not silently inherit those orchestration choices: every retained or replaced choice must
be explicit in the project config and complete-system comparison.

## Parameter convention

The exact count obtained with PyTorch by summing all trainable tensors from
`SCNet(**conf/config.yaml:model)` is **10,578,768**. There are no frozen parameters,
so total and trainable counts are identical under this convention.

This does **not** reproduce the previously expected “approximately 10.08M” figure.
The repository must use the source-derived exact count and disclose the discrepancy;
it must not alter the topology merely to force the expected number.

## Canonical project adapter

`src/ternarystem/models/scnet.py` owns behavior outside the upstream-equivalent core:

1. retain the pinned upstream topology and output head unchanged;
2. reorder outputs to canonical `vocals, drums, bass, other` order;
3. apply equal additive waveform mixture consistency;
4. expose common project checkpoint, evaluation, and inventory metadata.

Because the canonical system adds output reordering and mixture consistency, project
results are labeled `scnet_5d95bf96_adapted_v1`; raw-core equivalence and adapter
behavior are tested separately. SCNet remains an FP quality/Pareto reference and is
not automatically eligible for ternary deployment.

## Open contract correction

The pinned upstream architecture emits direct estimates rather than bounded complex
masks. Applying a bounded-mask head would modify the output topology and invalidate
upstream-core equivalence. Deployment documentation must therefore treat bounded
complex masks as Candidate A's requirement and explicitly record Candidate B's direct
estimate semantics instead of claiming both are identical systems.
