# New-session continuation prompt

Copy the prompt below into a new coding-agent session opened at the TernaryStem repository root.

```text
Continue TernaryStem development locally for GPU deployment #2. Treat the repository as the source of truth. Do not rent or launch a GPU instance, access the official MUSDB test partition, expose credentials, download restricted datasets, start sensitivity/QAT, or commit unless the operator explicitly asks.

Repository

- Path: the current TernaryStem repository root
- Last committed source at handoff: 497d4f9aec2d01b9902aa53a428570b28936891c
- The working tree intentionally contains uncommitted deployment #1 evidence and documentation. Preserve all user changes: do not reset, clean, checkout, restore, or overwrite them.
- `docs/NEW_SESSION_PROMPT.md` was recreated at the operator's request after it disappeared during the prior inspection.

Start by

1. Run:
   - `git status --short --branch`
   - `git diff --check`
   - `git diff`
   - `git log -3 --oneline`
2. Read completely:
   - `README.md`
   - `docs/DEPLOYMENT_2.md`
   - `docs/VAST_AI.md`
   - `docs/BENCHMARK_PROTOCOL.md`
   - `docs/STATUS.md`
   - `docs/PLAN.md`
   - `docs/MODEL_CARD.md`
   - `results/remote/2026-07-23-vast-fp32-plateau/README.md`
3. Inspect `results/remote/2026-07-23-vast-fp32-plateau/experiment.json` without reproducing private absolute paths.
4. Reinspect relevant model, data, training, evaluation, checkpoint, quantization, export, runtime, and Vast code before editing. Do not trust stale summaries over current files.
5. Report the exact dirty state and any mismatch with this handoff before making changes.

Verified deployment #1 evidence

- Run ID: `20260723T164310Z-497d4f9aec2d`.
- Clean detached source: `497d4f9aec2d01b9902aa53a428570b28936891c`.
- RTX 5090 32 GB; PyTorch 2.11.0+cu128.
- Official MUSDB18-HQ `train/` only, canonical 86/14 split; official test was not extracted on the training path.
- Existing 632,208-parameter FP32 bounded-complex-mask TFC-TDF U-Net directly models 1,024 of 2,049 bins.
- 31 completed epochs / 310,000 six-second chunks.
- Best checkpoint: record epoch 23, the 24th completed epoch after 240,000 chunks.
- Best development `global_sdr`: 5.447996 dB versus 1.245475 dB equal share.
- Same-checkpoint vocals/drums/bass/other development `global_sdr`: 6.123076 / 5.980548 / 5.431550 / 4.256812 dB.
- Gate 1 is 7.5 dB; failure margin 2.0520 dB. The operator stopped the run. No sensitivity or QAT ran.
- Sustained throughput was about 53–54 chunks/s; approximately 5.39 GiB peak allocated and 6.42 GiB peak reserved.
- Off-instance bundle was copied to Filen and content-checked. Checkpoints are not in Git.
- Hashes: best.pt `7c9a1c8c03f6f494a736a83f7753ad35ae7bad6f1c88b7b2b962a4b102aa0f8f`; latest.pt `56d0843998297accdd82314f6bc6b2f90ace6527a1d949d147af430aca250cc6`.
- The former Vast instance could not be reached by SSH at the end. Never assume it is stopped; before any future rental, ask the operator to verify the Vast dashboard to avoid billing.

Scientific and legal boundaries

- The current FP model is incapable and must not be quantized.
- Omitted high-frequency estimates are zero-padded before mixture consistency, effectively equal-sharing unresolved residual.
- Current TFC blocks have local temporal convolutions but no distinct configurable long-range separator. Width alone is not a justified fix.
- Deployment #2 is an FP-only architecture/Pareto screen. QAT requires a separately approved later deployment after one exact FP checkpoint reaches 7.5 dB and passes broader development confirmation.
- Development `global_sdr` is not museval or BSSEval.
- Published SCNet/Band-SCNet/HTDemucs scores are context only because protocols and data conditions differ.
- Deployment #2 is MUSDB18-HQ-only. Do not integrate MoisesDB, Slakh2100, MedleyDB, proprietary data, or official MUSDB test audio.
- Repository code is Apache-2.0; pinned SCNet source is MIT. Dataset and model-weight rights are separate. Never claim repository licensing authorizes checkpoint redistribution.
- Never commit datasets, checkpoints, generated audio, credentials, private keys, or secret-bearing/private paths.

Locked candidates

Control C — reproducibility sentinel

- Existing `configs/remote/fp32_complex_mask.yaml` system unchanged.
- At most 10,000 chunks and never promoted.

Candidate A — TernaryStem-v2

- Universal four-stem separator.
- Proper TFC-TDF-v3-style residual blocks.
- All 2,049 bins modeled with odd-safe down/up-sampling.
- Explicit long-range temporal stage made from precision-factory Conv/Linear families.
- Bounded complex masks and exact mixture consistency.
- Exact frozen trainable count 2,398,783 (within 2.0M–3.0M).
- FP32 for deployment #2; eligible Conv/Linear families remain named and instrumentable for future selective ternary/W4/W8 work.
- Keep STFT/iSTFT, losses, required norms, projections unless later proven safe, mask construction, mixture consistency, and reconstruction FP32.

Candidate B — pinned SCNet

- Authoritative upstream: `https://github.com/starrytong/SCNet`.
- Pin commit `5d95bf96b19c3eede63248d171efeca8e3abb948`.
- Preserve MIT attribution and review transitive material.
- The pinned standard universal core has exactly 10,578,768 trainable parameters under the source-derived convention; disclose that this conflicts with the earlier approximate 10.08M expectation.
- Integrate through the common interfaces with the smallest adapter.
- Record preprocessing, normalization, topology, source ordering, output semantics, config, recurrent/sequential costs, and parameter derivation.
- Separate an upstream-equivalent core from the canonical project adapter in tests. If topology or output heads change, label it `SCNet-adapted`; never claim an adaptation is an exact published reproduction.
- It is a quality/Pareto candidate, not automatically a ternary candidate.

Three-reviewer outcome

Three independent read-only reviewers unanimously returned `request_changes` / `NO-GO`. Incorporate these corrections into the implementation sequence:

1. The common contract must cover more than model construction: architecture identity, waveform output, preprocessing and architecture-specific loss hooks, source order, parameter/inventory metadata, checkpointing, evaluation, effective-batch semantics, and training counters.
2. Perform the pinned SCNet provenance/interface audit before freezing an A-shaped interface. Preserve upstream source/hash/license records and distinguish core equivalence from adapter behavior.
3. Use a versioned checkpoint schema with immutable architecture/config/provenance hashes, completed chunks, optimizer updates, rung lineage, scaler/optimizer/scheduler state, and Python/NumPy/Torch CPU/all-CUDA RNG state.
4. Freeze every ambiguous promotion rule in a machine-readable gate-policy config before results: Control C tolerance, material stem collapse, positive learning signal, recent-slope window/estimator, paired interval method and confidence level, credible extrapolation, missing/NaN refusal, tie behavior, and reserve formula.
5. Separate deterministic sample-spec generation from audio loading. Persist stable validation IDs with track/start metadata and per-track/per-stem float64 signal/error energies.
6. Build deployment #2 in a separate allowlisted FP-only path. It must reject QAT, sensitivity, PTQ, distillation, AMP, extra data, official test, arbitrary configs, and unapproved checkpoint lineage before spawning training.
7. Implement a crash-safe serial rung state machine and persistent cumulative billed-time/dollar ledger. Missing, corrupt, conflicting, or unwritable state fails closed. Promotion requires an atomically published and independently verified synchronization receipt.
8. Fix inherited lock descriptors and process supervision. Track and verify PID/PGID identity; terminate and wait for complete helper groups without harming the active trainer or unrelated processes.
9. Add an offline readiness command bound to exact commit/config/provenance hashes. Missing evidence returns nonzero. Actual CUDA-required readiness remains open until run on an authorized local GPU; do not substitute a rental or CPU fallback.

Implementation order

Stage 1 — legacy sentinel and common system contract

- First run the unchanged legacy focused/full tests to establish the starting baseline.
- Add an explicit legacy architecture ID for Control C and narrowly compatible migration for old configs/checkpoints.
- Add a model/system registry and common adapter protocol used by training, validation, checkpointing, preflight, inference, inventory, export where applicable, and records.
- Add strict selector-consumption validation so misspelled exact precision paths cannot remain silently unused.
- Keep Control C behavior and legacy checkpoints covered.

Suggested files:
- `src/ternarystem/models/registry.py`
- `src/ternarystem/models/interface.py`
- `src/ternarystem/config.py`
- `src/ternarystem/training/checkpoint.py`
- `scripts/train.py`, `validate.py`, `separate.py`, `inventory.py`
- `tests/unit/test_registry.py`
- `tests/unit/test_checkpoint.py`
- `tests/integration/test_legacy_control.py`

Stage 2 — SCNet provenance/interface spike

- Inspect the pinned upstream source lawfully without downloading data or weights.
- Vendor only required source/material, preserve the MIT notice, record source/blob hashes and local patches.
- Freeze upstream preprocessing, normalization, topology, source order, outputs, dependencies, and parameter convention before adapting.
- Add deterministic synthetic upstream-core equivalence fixtures independent of published checkpoints/audio.

Suggested files:
- `third_party/SCNet/LICENSE`
- `docs/provenance/SCNET_5d95bf96.md`
- `src/ternarystem/models/scnet.py`
- `configs/deployment_2/scnet.yaml`
- `tests/integration/test_scnet_adapter.py`
- `tests/unit/test_scnet_provenance.py`

Stage 3 — checkpoint determinism and effective-batch/rung counters

- Save/restore Python, NumPy, Torch CPU, and all CUDA RNG states.
- Record deterministic backend settings.
- Preserve optimizer, scheduler, scaler, sample exposure, optimizer updates, and effective-batch semantics.
- Add gradient accumulation only if required, with correctly scaled loss and accumulation-safe checkpoints.
- Add architecture-neutral stop-at-completed-chunks/rung control without changing the immutable maximum-horizon scheduler config.

Tests:
- duplicate-run determinism;
- interrupted versus uninterrupted resume;
- physical-batch/accumulation equivalence;
- wrong-architecture/config/provenance refusal;
- scheduler horizon unchanged by rung stop.

Stage 4 — Candidate A

- Freeze exact topology and config before observing paid results.
- Implement full 2,049-bin odd-safe processing, residual TFC-TDF-v3 blocks, and explicit temporal stage.
- Assert exact trainable count in 2.0M–3.0M.
- Inventory temporal receptive field, parameters, MAC convention, activation/operator shapes, precision family, and FP32 boundaries.
- Ensure every eligible Conv/Linear is named and covered by precision/inventory instrumentation.

Tests:
- finite forward/backward;
- exact parameter assertion;
- true 4,096-point STFT / 2,049-bin odd shapes and odd frame counts;
- silence, bounded masks, four-stem order, exact mixture reconstruction;
- STFT/iSTFT and overlap-add tolerances;
- tiny-data loss decrease;
- checkpoint round trip;
- complete quantizable-family/selector coverage.

Stage 5 — Candidate B final adapter and generalized inventory

- Finish common adapter without changing upstream topology/head unless explicitly labeled adapted.
- Generalize inventory beyond Conv2d/Linear to recurrent operators, call multiplicity, sequence shapes, and sequential costs.
- Assert the exact frozen Candidate B parameter convention.

Stage 6 — development evaluation

- Preserve the frozen 140-chunk continuity diagnostic exactly.
- Persist ordered stable sample IDs and per-track/per-stem energies.
- Add predeclared paired track-grouped uncertainty and a track-balanced or whole-track confirmation over all 14 development tracks.
- Fail before audio access when given official test, parent/sibling/symlinked test paths, or a noncanonical partition.
- Deployment #2 must never import or invoke `scripts/evaluate.py`.

Tests:
- stable IDs across worker counts and restart;
- no training/development overlap;
- paired A/B identities;
- exact energy aggregation and silent-target behavior;
- official-test denial before `musdb.DB` or audio opening.

Stage 7 — gate policy, ledger, and serial runner

- Create a versioned machine-readable gate policy implementing every locked paid-rung rule.
- Add a deployment-wide append-only ledger covering setup, idle, validation, retry, and sync time across launches/replacements.
- Planned cutoff: 7.5 billed hours / $6 at $0.80/hour.
- Absolute cutoff: 8.75 billed hours / $7. Never let synchronization block absolute shutdown.
- Use integer seconds/cents or another deterministic accounting convention.
- Implement atomic states such as pending -> running -> checkpointed -> remotely_committed -> gate_passed/stopped.
- Manifests include deployment/candidate/rung IDs, parent/checkpoint/config/provenance hashes, counters, metrics, decision, ledger snapshot, and sync receipt.
- Upload immutable generations to staging, verify checksums, then publish a small committed receipt/pointer last.
- Never promote an unsynced checkpoint. Restart never resets the ledger or repeats an already committed rung.

Tests:
- all pass/fail/tie boundary decisions;
- NaN/missing-data refusal;
- restart/replacement persistence and clock rollback;
- corrupt/missing/unwritable ledger refusal;
- crashes before/after checkpoint, rename, upload, receipt, and state transition;
- concurrent sync refusal/serialization;
- cumulative planned and absolute cutoff behavior;
- no prohibited branch reachable.

Stage 8 — Vast reliability

- Add a separate deployment #2 setup/start/stop/controller; do not route through deployment #1 `run_pipeline.sh`.
- Close launcher/pipeline lock descriptors before spawning unrelated descendants.
- Supervise dedicated verified process groups; terminate, wait, and boundedly escalate helper cleanup.
- Preserve the active trainer when cleaning stale helpers.

Tests using temporary directories/fake trainers only:
- immediate launcher/pipeline lock reacquisition;
- smoke/approval exit and restart;
- stale/reused PID metadata;
- vanished leader with surviving grandchildren;
- TERM-resistant helper cleanup;
- no unrelated process signaled;
- no helper inherits lock descriptors.

Stage 9 — verification and documentation

- Run focused tests after each stage, then Ruff, all pytest tests, CMake/build/CTest, scalar/AVX2 correctness, dry runs, and any authorized local CUDA-required matrix.
- Update `docs/DEPLOYMENT_2.md`, `docs/STATUS.md`, `docs/PLAN.md`, `docs/MODEL_CARD.md`, configs, provenance, and tests as evidence changes.
- Add an exact readiness matrix against all 14 prerequisites in `docs/DEPLOYMENT_2.md`.
- Do not weaken quality, cost, legal, or test-partition gates to pass tests.
- Do not commit unless the operator explicitly asks. Suggest small commit boundaries and report exact dirty state.

Locked paid rungs — implement but do not launch

- 10k chunks: C, A, B commissioning and cost projection; C never promotes.
- 30k cumulative: A and B.
- 100k cumulative: A and B; require >=5.40 dB or +0.25 dB over 30k plus the frozen positive recent slope.
- Select one at 100k. A quality winner needs >=+0.15 dB paired advantage with the predeclared track-grouped interval excluding zero. If inconclusive, A may win only as the documented deployment/ternary tie-break, not as a quality winner.
- 200k cumulative: selected candidate; promotion requires +0.20 dB over its 100k best and exceeding verified 5.4480 dB unless already at 7.5.
- 300k cumulative: broader confirmation. Continue only at >=6.3 dB with frozen credible-positive extrapolation and sufficient reserve.
- Stop at the first qualifying >=7.5 dB checkpoint or planned cutoff.
- If no model qualifies, publish the negative result; never force QAT.

Known future-QAT blockers — document only, do not implement as a reason to launch QAT

- `scripts/sensitivity.py` lacks explicit observer calibration/freeze.
- It averages batch SDR rather than canonical energy-aggregated `DevelopmentDiagnostics`.
- Saturation retains last-forward values instead of run-wide aggregation.
- `prepare_qat.py` must eventually consume only corrected canonical sensitivity output.
- Odd full-spectrum shapes may not be BitNet I2_S-compatible; future work needs real-shape ternary/W4/W8/FP fallbacks.
- Packed parameters do not constitute a working or fast end-to-end native graph.

Working method

- Make changes in small reviewable stages and preserve unrelated dirty files.
- Read before editing and run focused tests after every stage.
- If a source file changes concurrently, stop and report it rather than overwriting it.
- Never invent quality, cost, hardware, latency, legal, or test evidence.
- Keep the recommendation `NO-GO` until every mandatory prerequisite passes. Recommend `GO` only when the exact readiness matrix is fully green.

Begin with Stage 1 only: inspect current files and tests, report the exact dirty state, then propose the smallest first edit group and focused tests. Do not start GPU, cloud, QAT, dataset-download, or official-test commands.
```

## Maintenance note

Keep this prompt synchronized with `docs/DEPLOYMENT_2.md`. The deployment contract controls whenever the two differ.
