# GPU deployment #2 — locked FP architecture screen

## 1. Purpose and hard boundaries

Deployment #2 answers one question: can a stronger, still plausibly deployable FP32 architecture produce a credible path past the frozen 7.5 dB development `global_sdr` gate within a solo-developer budget?

This deployment is **not** a QAT run and is not an official test evaluation. It must:

- use only the legally supplied MUSDB18-HQ `train/` partition and the canonical 86/14 split;
- keep the official 50-track test partition absent and inaccessible;
- run FP32 qualifying candidates only;
- use Candidate A's bounded complex masks and exact mixture consistency; Candidate B preserves the pinned SCNet direct-estimate head and adds canonical waveform mixture consistency in a separately labeled adapter;
- preserve the project direction of a universal four-stem model;
- record quality, parameters, operator shapes, VRAM, throughput, GPU-hours, and dollars;
- stop when a pre-registered quality or cost gate fails;
- never start sensitivity, PTQ, QAT, distillation, or an extra-data experiment;
- plan for at most 45 billed instance-hours and stop absolutely before 50 billed hours at the approved $0.80/hour offer (`$36` planned, `$40` absolute).

Every setup, preflight, retry, validation, synchronization, and idle minute counts against the persistent cumulative ledger. A restart must not reset it.

## 2. Evidence from deployment #1

The checkpoint-free evidence is in `results/remote/2026-07-23-vast-fp32-plateau/`.

- Source: clean commit `497d4f9aec2d01b9902aa53a428570b28936891c`.
- Model: 632,208-parameter FP32 complex-mask TFC-TDF U-Net.
- Exposure: 31 completed epochs / 310,000 six-second chunks.
- Best: 5.4480 dB at record epoch index 23 (24th completed epoch).
- Same-checkpoint per-stem development `global_sdr`: vocals 6.1231, drums 5.9805, bass 5.4315, other 4.2568 dB.
- Equal-share overall development `global_sdr`: 1.2455 dB.
- Gate: 7.5 dB.
- No sensitivity or QAT ran.

The current model directly predicts only 1,024 of the 2,049 bins produced by its 4,096-point real STFT. It zero-pads the omitted estimates before mixture consistency, which assigns the unresolved residual equally among stems. Its TFC blocks provide local temporal convolutions, but it has no distinct configurable long-range temporal separator. Deployment #2 must not assume that width alone fixes these limitations.

## 3. Frozen candidates

Exactly three FP systems enter the 10k commissioning rung. Only candidates A and B may continue beyond commissioning.

### Control C — reproducibility sentinel

The existing `configs/remote/fp32_complex_mask.yaml` architecture and recipe. Its purpose is to catch data, metric, software, or training drift. It receives at most 10,000 chunks and is never promoted.

### Candidate A — TernaryStem-v2

A universal, quantization-instrumented model designed for the existing deployment direction:

- proper TFC-TDF-v3-style residual blocks rather than the simplified current block;
- all 2,049 STFT bins modeled, with odd-dimension-safe down/up-sampling;
- an explicit long-range temporal stage built from precision-factory Conv/Linear operations;
- bounded complex masks and exact waveform mixture consistency;
- exactly 2,398,783 trainable parameters under the frozen config (within the required 2.0M–3.0M range);
- FP32 training in deployment #2;
- every eligible Conv/Linear family named and included in the operator/precision inventory;
- norms, STFT/iSTFT, mask construction, projections unless later proven safe, mixture consistency, and reconstruction retained FP32.

The exact topology and resolved config must be committed and immutable before rental. Full-spectrum, temporal, and block changes are bundled into a deployable-system candidate; results must not be presented as a causal ablation of any one change.

### Candidate B — pinned SCNet architecture

Use the authoritative MIT-licensed implementation at:

- repository: `https://github.com/starrytong/SCNet`;
- reviewed source commit: `5d95bf96b19c3eede63248d171efeca8e3abb948`;
- source-derived parameter convention: exactly 10,578,768 trainable parameters for the pinned standard universal model. This conflicts with the earlier approximate 10.08M expectation and is disclosed rather than forcing a topology change.

The audit is recorded in `docs/provenance/SCNET_5d95bf96.md`; the MIT notice and byte-pinned core files are preserved. The upstream core emits direct waveform estimates in drums/bass/other/vocals order and has no mask or mixture-consistency projection. The project system is therefore explicitly identified as `scnet_5d95bf96_adapted_v1`: its core remains source-equivalent, while its adapter reorders to vocals/drums/bass/other and applies waveform mixture consistency. It must never be described as an exact published-system reproduction. Published SCNet numbers are context only because data exposure, training recipe, postprocessing, and metrics differ.

Candidate B is a quality/Pareto reference, not automatically a ternary deployment candidate. Its recurrent operator shapes and sequential cost must be inventoried before promotion.

## 4. Comparison contract

This is a Pareto comparison of two frozen systems, not a parameter-matched causal architecture ablation.

Both candidates must use:

- the canonical 86 training and 14 development tracks;
- identical deterministic training sample indices where architecture interfaces permit;
- seed `20250218`;
- the same stem definitions and source ordering;
- dynamic cross-track remix, gain, stereo swap, and polarity policy;
- FP32 qualifying precision;
- a predeclared maximum-horizon scheduler from the first launch;
- equivalent optimizer-update and effective-batch semantics, using gradient accumulation only if physical batches differ;
- the existing 140-chunk diagnostic for continuity;
- a separate track-balanced or whole-track development confirmation path;
- no official test, extra data, teacher, shifts, ensembles, or source-specific model bag.

Architecture-specific preprocessing and losses may remain only when required for faithful operation and must be frozen before results are observed. In that case, conclusions compare complete systems, not architecture alone. Score must be reported against both chunks and paid GPU-hours.

## 5. Required local work before rental

Deployment is `NO-GO` until all items pass.

1. Import the deployment #1 non-audio record and document its checkpoint hashes.
2. Add an architecture registry and common separator/checkpoint interface.
3. Implement candidates A and B with immutable resolved configs and provenance.
4. Assert exact parameter counts and generate MAC/operator-shape inventories.
5. Add architecture-aware strict checkpoint loading.
6. Save and restore Python, NumPy, Torch CPU, and CUDA RNG state.
7. Add a stop-at-completed-rung control that does not alter the resolved maximum schedule.
8. Add a persistent cumulative billed-time/dollar ledger with planned and absolute cutoffs.
9. Add atomic serial-rung manifests and off-instance synchronization.
10. Persist validation sample/track IDs and per-track signal/error energies.
11. Add track-balanced or whole-track development-only evaluation; it must reject the official test path.
12. Keep the deployment #2 runner free of sensitivity/QAT branches.
13. Fix the orphaned background-sleep/lock cleanup observed after deployment #1 smoke.
14. Run Ruff, all Python tests, native correctness tests, and CUDA-required dry runs.

Required candidate tests include finite forward/backward, silence, four-stem ordering, exact mixture reconstruction, full-spectrum odd-bin handling, overlap-add boundaries, loss decrease on a tiny dataset, duplicate-run determinism, interrupted-versus-uninterrupted resume equivalence, checkpoint round-trip, no train/development overlap, stable validation IDs, SCNet upstream-reference equivalence, and cost/rung refusal behavior.

### 5.1 Current readiness matrix

The authoritative executable check is `python scripts/deployment_2.py readiness`. Missing evidence returns nonzero.

| # | Prerequisite | Current state |
|---:|---|---|
| 1 | Deployment #1 non-audio evidence and hashes | PASS |
| 2 | Architecture registry/common construction | PASS |
| 3 | Frozen Candidate A topology/count | PASS locally |
| 4 | Pinned SCNet provenance/core/adapter | PASS locally |
| 5 | Strict architecture-aware checkpoints and RNG state | PASS locally |
| 6 | Rung stop without scheduler mutation | PASS locally |
| 7 | Persistent cumulative billing ledger | PASS locally |
| 8 | Atomic rung state and verified sync receipts | PASS locally |
| 9 | Stable validation IDs and float64 track energies | PASS locally |
| 10 | Official-test path denial before audio access | PASS locally |
| 11 | FP-only allowlist and gate-policy hashes | PASS locally |
| 12 | Verified process-group supervision, transaction crash recovery, receipt-last sync, and no inherited launcher lock | PASS locally; 31 focused matrix tests recorded in `results/readiness/controller_fake_matrix.json` |
| 13 | Clean frozen source commit bound by `.deployment2-lock.json` | PASS in the current checkout; regenerate the ignored lock after any authorized source change |
| 14 | Authorized local CUDA-required evidence | **OPEN — no authorized local GPU evidence exists** |

Therefore deployment remains **NO-GO** solely on the authorized CUDA-required readiness evidence in item 14. A rental or CPU fallback must not be used to turn that item green.

## 6. Paid rungs and gates

The runner uses cumulative chunks and one persistent maximum-horizon scheduler. It must sync completed checkpoints before every stop.

### Rung 0 — commissioning, 10k chunks each

Run C, A, and B for 10,000 chunks.

Pass requirements:

- clean pinned source and accepted dataset terms;
- CUDA execution with no CPU fallback;
- finite loss/gradients and zero skipped FP32 updates;
- strict checkpoint resume and repeatable validation;
- no reconstruction, silence, or shape failure;
- measured VRAM, chunks/s, validation seconds, and operator inventory;
- control C reproduces deployment #1's early curve within a predeclared tolerance;
- projected worst-case deployment, including 20% overhead, fits within 45 billed hours.

Do not eliminate A or B solely for early rank at 10k. Stop the deployment if either cannot fit the frozen physical/effective-batch contract or if the cost projection fails.

### Rung 1 — 30k chunks each for A and B

Both candidates continue to 30,000 cumulative chunks.

Eligibility requires each system to beat its own equal-share overall diagnostic, avoid material per-stem collapse, and retain a positive training/validation learning signal. Do not use the official test set or published SCNet checkpoint for promotion.

### Rung 2 — 100k chunks each for A and B

Both eligible candidates continue to 100,000 cumulative chunks to reduce early-rank reversal risk.

A candidate remains viable only if it:

- has best development `global_sdr` at least 5.40 dB or improves by at least 0.25 dB over its 30k best;
- has a positive predeclared recent validation slope unless it already reaches 7.5 dB;
- has no non-finite or persistent single-stem failure;
- remains within the worst-case cost ledger.

Select one system after 100k. A clear quality winner requires at least a 0.15 dB paired development advantage with the predeclared track-grouped interval excluding zero. If quality is statistically inconclusive, select A only as the documented deployment/ternary tie-break, not as a quality winner. If only B is viable, B may continue as the FP quality reference, but that does not authorize QAT.

### Rung 3 — 200k chunks for the selected system

The selected viable system continues from 100k to 200k. Promote it from 200k to 300k only if its 200k best improves by at least 0.20 dB over the 100k best and exceeds deployment #1's verified 5.4480 dB, unless 7.5 dB is already reached.

### Rung 4 — 300k chunks for the selected system

After promotion, continue to 300k and run the broader development-only confirmation. Continuing beyond 300k requires all of:

- at least 6.3 dB development `global_sdr`;
- positive credible learning-curve extrapolation toward 7.5 dB;
- acceptable per-stem and whole-track development behavior;
- enough ledger reserve to finish before the planned $36 cutoff.

Otherwise stop and publish a negative result.

### Final continuation

Only the selected system may continue, using the same optimizer, scheduler, samples, and checkpoint lineage. Stop at the first qualifying checkpoint or the planned 45-hour cutoff. The operator must terminate before 50 billed hours / $40 under all circumstances.

## 7. Success and failure decisions

Deployment #2 succeeds scientifically if it produces a reproducible Pareto comparison and obeys the budget, even when both models fail.

An FP candidate qualifies for later quantization work only if one exact checkpoint:

- reaches at least 7.5 dB frozen development `global_sdr`;
- beats equal share overall and for every stem;
- has no material per-stem collapse;
- passes broader development-only confirmation;
- has a plausible operator/CPU profile;
- is checksum-pinned and independently backed up.

No per-stem maxima from different epochs may be combined. The development diagnostic must never be called BSSEval.

If A qualifies, deployment #3 may run corrected sensitivity and matched selective ternary/W4/W8 QAT from A's exact checkpoint. If only B qualifies, record the quality result and inspect its operator/quantization feasibility; do not quantize an inadequate A to preserve the narrative.

## 8. Known work required before any future QAT

The existing sensitivity path must be corrected before deployment #3:

- calibrate activation observers explicitly before evaluation;
- freeze calibrated ranges;
- use canonical energy-aggregated `DevelopmentDiagnostics`, not averages of batch SDRs;
- aggregate saturation over the full evaluation;
- update `prepare_qat.py` to consume only the corrected metric;
- inventory non-BitNet-compatible shapes such as odd full-spectrum dimensions;
- benchmark ternary, W4, W8, and FP fallbacks per real operator shape.

The current native runtime is not end-to-end. Parameter coverage or packed size alone is not a deployment-speed claim. A complete graph schema, loader, scalar reference, missing operators, numerical agreement tests, and end-to-end CPU benchmark remain required before release.

## 9. Data policy

Deployment #2 is MUSDB18-HQ-only. MoisesDB, Slakh2100, MedleyDB, or any other data is a separately versioned experiment after architecture selection. MedleyDB overlap with MUSDB must be denied explicitly; Slakh has no real vocals; MoisesDB introduces CC BY-NC-SA obligations and taxonomy mapping. Code and model-weight licensing remain separate decisions.
