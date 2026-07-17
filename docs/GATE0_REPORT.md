# Gate 0 Interim Report

Status: **open — BitNet.cpp is promising for large-K shapes but does not consistently beat the production FBGEMM INT8 baseline by the required 1.3x**.

## Environment

Measured 2026-07-17 on an AMD Ryzen 7 7730U (x86-64 AVX2), GCC 13.3, Release build, one thread. Weight packing occurred before timing. Timings are medians of 11 warm samples from `runtime/benchmarks/gemm_benchmark.cpp`.

## Kernel results

| M (rows) | N (outputs) | K | INT8 AVX2 ms | Packed ternary AVX2 ms | Ternary speedup |
|---:|---:|---:|---:|---:|---:|
| 64 | 256 | 512 | 0.450 | 0.545 | 0.83x |
| 1 | 1024 | 1024 | 0.055 | 0.066 | 0.84x |
| 16 | 1024 | 1024 | 0.885 | 1.058 | 0.84x |
| 256 | 64 | 1024 | 0.884 | 1.055 | 0.84x |
| 256 | 1024 | 64 | 1.051 | 1.347 | 0.78x |

The AVX2 ternary path reads packed two-bit weights directly and expands 32 weights in registers using nibble lookup tables. It never materializes or caches a fully unpacked weight matrix. Exact INT32 agreement with the scalar reference passes, including vector and remainder paths.

These INT8 numbers are a project-local AVX2 baseline, not an optimized library baseline. The local packed implementation is not sufficient.

## BitNet.cpp adaptation result

Microsoft BitNet commit `6f228769736934053af73b6a313a6dfc30ef89bf` (MIT license) was built outside the repository under `/tmp`. `runtime/benchmarks/bitnet_comparison.cpp` adapts its I2_S GEMM without vendoring it. The adapter:

- transforms weights offline into BitNet's 128-element interleaved I2_S layout;
- maps ternary values to codes `{0,1,2}`;
- includes the required activation-sum correction in timed execution;
- checks every output exactly against the scalar ternary reference before timing.

| M | N | K | BitNet I2_S ms | Local INT8 AVX2 ms | Speedup over INT8 |
|---:|---:|---:|---:|---:|---:|
| 64 | 256 | 512 | 0.073 | 0.448 | 6.13x |
| 1 | 1024 | 1024 | 0.010 | 0.056 | 5.51x |
| 16 | 1024 | 1024 | 0.130 | 0.900 | 6.93x |
| 256 | 64 | 1024 | 0.174 | 0.889 | 5.10x |
| 256 | 1024 | 128 | 0.518 | 1.946 | 3.75x |

### Production INT8 comparison

PyTorch 2.13's FBGEMM quantized Linear was then measured with one thread, offline prepacking/input quantization, and timed dispatch plus output requantization. The fastest available production backend is the relevant gate baseline.

| M | N | K | BitNet I2_S ms | FBGEMM INT8 ms | BitNet speedup |
|---:|---:|---:|---:|---:|---:|
| 64 | 256 | 512 | 0.073 | 0.089 | 1.22x |
| 1 | 1024 | 1024 | 0.010 | 0.035 | 3.49x |
| 16 | 1024 | 1024 | 0.130 | 0.157 | 1.21x |
| 256 | 64 | 1024 | 0.174 | 0.163 | 0.94x |
| 256 | 1024 | 128 | 0.518 | 0.357 | 0.69x |

The gate is therefore not met across representative shapes. The comparison is still favorable to BitNet because its timed path performs activation-sum correction but not per-channel scale/bias/requantization, while FBGEMM includes output requantization. BitNet I2_S also requires K divisible by 128; other layers need padding, another kernel, or mixed precision. Selective dispatch may remain useful for matrix-vector and large-K operations, but cannot justify an all-ternary runtime yet.

## Quantization plumbing probe

The deterministic synthetic regression probe (`scripts/quant_probe.py`) confirms that STE training executes and reduces loss. It is not music quality evidence:

| Method | Initial MSE | Final MSE | Improvement | Zero fraction |
|---|---:|---:|---:|---:|
| Adaptive 40% target | 1.3145 | 1.2369 | 1.06x | 40.6% |
| BitNet-style abs-mean | 1.1870 | 0.5958 | 1.99x | 60.7% |

Full Python tests pass (11), native scalar/AVX2 correctness passes, and 26 candidate model operator calls were captured in `results/operator_shapes.json`.

## Next decision work

1. Add matched scale/bias/requantization to the BitNet adapter and benchmark in one process.
2. Use operator inventory to estimate selective BitNet versus FBGEMM dispatch, including padding.
3. Begin the planned W4A8/W8A8 fallback path rather than assuming all-ternary deployment.
4. Run the reduced music-separation QAT probe on licensed local or remote data.
5. Validate BitNet's NEON path on ARM hardware.

No Gate 0 pass is claimed. If optimized packed kernels remain below 1.3x, follow the planned W4A8/W8A8 deployment pivot while retaining ternary quality experiments.
