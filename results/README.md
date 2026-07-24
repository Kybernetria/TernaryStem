# Results

Machine-readable experiment records and compact reports belong here. Do not commit checkpoints, generated audio, datasets, or benchmark binaries.

Recorded evidence:

- `int8_kernel_benchmark.json`: production FBGEMM INT8 shape benchmark used by the interim Gate 0 report.
- `operator_shapes.json`: candidate model operator inventory.
- `quant_probe_*.json`: deterministic synthetic software probes; not music evidence.
- `remote/2026-07-17-selective-ternary/`: MUSDB18-HQ development-split FP, sensitivity, and matched selective-ternary QAT smoke records; not BSSEval or a quality claim.
- `remote/2026-07-23-vast-fp32-plateau/`: RTX 5090 FP32 complex-mask baseline stopped after 310,000 chunks at a best 5.4480 dB development `global_sdr`; Gate 1 failed and no QAT ran.

Results must follow `docs/BENCHMARK_PROTOCOL.md`; absent measurements are `null`, never zero or invented.
