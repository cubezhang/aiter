---
name: mi325x-poc-tuning
description: Build, baseline, accept, deploy, benchmark, diagnose, and tune reproducible AI, distributed-training, diffusion, and HPC POCs on AMD Instinct MI325X (CDNA3, gfx942) servers and clusters. Use for MI325X hardware/ROCm readiness, AMD Customer Acceptance Guide execution, 256GB HBM3E model fit, vLLM/SGLang/ATOM serving, SLO-constrained TPM/TTFT/TPOT cliff searches, cold-start/warmup/repeat campaigns, interrupted-run recovery, CUDAGraph batch buckets, chunked prefill, MTP/speculative decoding, FP8 FNUZ/Quark quantization, AITER, Primus/Megatron/PyTorch/JAX training, RCCL/RDMA/MoE communication, SPX/DPX/QPX/CPX and NPS partitioning, SR-IOV, multi-node scaling, ROCm profiling, kernel bottlenecks, performance regression, power/thermal tuning, inventory onboarding, or POC result comparison. Do not substitute MI300X, MI308X, or MI350/MI355 recipes without the portability gates in this skill.
---

# MI325X POC tuning

Run an evidence-driven loop from immutable discovery through formal acceptance, workload enablement, controlled tuning, and reproducible handoff. Treat MI325X as a runtime-detected platform: never assume eight GPUs, 256 GB usable HBM per visible device, SPX/NPS1, full-mesh links, bare metal, firmware, or software versions.

## Companion skills

Use installed siblings only when their scope applies:

- Read `../serving-llms-on-instinct/SKILL.md` for vLLM model-fit, launch, and endpoint checks.
- Read `../magpie-kernel-evaluator/SKILL.md` for kernel/framework benchmarks and gap analysis.
- Read `../tracelens-analysis-orchestrator/SKILL.md` for PyTorch trace analysis.

This skill owns MI325X identity, safety, acceptance, precision, partitioning, training/HPC, and evidence rules. Its gates override a companion's generic defaults.

## Hard platform boundaries

1. Require PCI device `1002:74a5`, market name containing `MI325X`, and LLVM target `gfx942`. A mismatch is a stop condition.
2. MI325X is CDNA3 with 256 GB HBM3E, 6 TB/s peak bandwidth, 304 active CUs, and a 1000 W peak board envelope. Detect usable values; product peaks are not measured POC results.
3. Baseline BF16/FP16 or the exact documented FNUZ FP8 path. Reject native MXFP4/MXFP6/MXFP8, OCP-FP8-only, or `gfx950` code paths. Keep checkpoint storage, weight compute, activation compute, KV cache, and collective quantization separate.
4. Do not silently reuse MI300X thresholds: both are `gfx942`, but MI325X has HBM3E capacity/bandwidth and its own `74a5` acceptance page. Conversely, do not port MI350/MI355 CDNA4 kernels, OCP FP8, 160 KB LDS, or MX formats.
5. Start read-only. Never install, reload drivers, alter partition/NPS modes, clocks, power, BIOS, boot parameters, NUMA balancing, network policy, containers, or production services without explicit approval and rollback.
6. Pin model/dataset revisions, container digest, ROCm/driver, framework/engine, AITER/RCCL, topology, partition mode, command, request/training shape, and correctness gates before comparison.
7. Stop on missing GPUs, PCIe degradation, XGMI faults, new uncorrectable ECC/RAS, bad-page growth, memory corruption, thermal/power throttling, driver resets, or correctness failure.

Read [references/mi325x-platform-constraints.md](references/mi325x-platform-constraints.md) before selecting precision, kernels, parallelism, partitions, or images.

## Workflow

### 0. Define the POC contract

Record:

- authorized hosts/IPs, SSH account, bare-metal/host/guest role, expected physical and logical GPU counts;
- dedicated/shared status, maintenance window, privileges, and allowed change classes;
- workload type: inference, training/fine-tuning, diffusion/multimodal, HPC/kernel, or formal acceptance;
- immutable model/dataset/workflow and licenses/access;
- precision/quantization, context/resolution/sequence length, batch/concurrency, TP/PP/DP/EP/CP plan;
- primary objective and SLOs: TTFT/TPOT/goodput, samples/s, tokens/s/GPU, MFU, job completion, bandwidth, power, or cost;
- correctness/quality, stability duration, evidence directory, and rollback owner.

If scope is incomplete, perform only read-only discovery and return a proposed matrix.

### 1. Capture immutable node or cluster baselines

Read [references/acceptance-gates.md](references/acceptance-gates.md). Preview, then collect:

```bash
python3 scripts/collect_mi325x_snapshot.py \
  --host <user@host> --expected-gpus 8 --dry-run

python3 scripts/collect_mi325x_snapshot.py \
  --host <user@host> --expected-gpus 8 \
  --output <artifact-dir>/snapshot.json
```

For multiple nodes, use an inventory without passwords or private keys:

```bash
python3 scripts/collect_mi325x_cluster.py \
  --inventory inventory.json --output-dir <artifact-dir>/nodes --dry-run
```

The collectors are read-only and avoid `sudo`; unavailable privileged evidence remains `REVIEW`, not inferred `PASS`.

### 2. Classify readiness

Use exactly one tier:

- `POC_READINESS`: identity, runtime, bounded health, existing-tool smoke, and workload stability.
- `FORMAL_NODE_ACCEPTANCE`: exact AMD/OEM MI325X sequence and thresholds.
- `FORMAL_CLUSTER_ACCEPTANCE`: node acceptance plus NIC/RDMA/fabric and multi-node collectives/workload.

Run gates in order: identity → host/runtime → PCIe/XGMI/RAS → RVS/AGFHC → TransferBench/rocBLAS/BabelStream/RCCL → RDMA/multi-node → workload soak. A later fast model run cannot compensate for a failed hardware gate.

### 3. Freeze platform and capacity

Read [references/partition-and-virtualization.md](references/partition-and-virtualization.md). Record physical GPU BDFs/serials, logical partitions, SPX/DPX/QPX/CPX, NPS mode, XGMI and GPU/NIC/NUMA mapping.

Estimate fit before downloading or loading:

```bash
python3 scripts/plan_mi325x_hbm.py \
  --physical-gpus 8 --hbm-per-gpu-gib 256 --usable-fraction 0.90 \
  --weights-gib 720 --runtime-gib 80 --kv-or-activation-gib 400 \
  --tensor-parallel 8
```

Use measured free HBM and workload-specific cache/activation estimates. Do not multiply logical CPX devices by 256 GB or count host RAM as HBM.

### 4. Select and preflight a workload path

Read the matching reference:

- Inference: [references/inference-playbook.md](references/inference-playbook.md)
- Training/fine-tuning/HPC: [references/training-hpc-playbook.md](references/training-hpc-playbook.md)
- Multi-node/EP/RDMA: [references/network-and-collectives.md](references/network-and-collectives.md)

For every container or environment:

1. resolve tag to digest and pin all versions;
2. confirm expected MI325X count and `gfx942` inside the environment;
3. import the framework/engine and kernel packages;
4. smoke-test required GEMM, attention/MLA, MoE, norm, collective, optimizer, or diffusion operators;
5. stop on ABI errors, missing code objects, JIT failure, NaN/Inf, unintended fallback, or device mismatch;
6. save the preflight log before a large model/dataset load.

### 5. Establish a correctness-first baseline

- Use the smallest topology that fits and the most official exact MI325X/gfx942 recipe.
- Prefer AMD-published, digest-pinned containers; record host driver/user-space compatibility.
- Start with default partitioning and clocks unless the contract explicitly tests them.
- Confirm effective kernels, precision, graph mode, parallel groups, rank mapping, and device placement from logs/traces.
- Run workload-specific correctness before measuring performance.

### 6. Benchmark a controlled matrix

Read [references/benchmark-method.md](references/benchmark-method.md). At minimum include:

- latency/small-scale point;
- expected production point;
- saturation/scaling knee;
- stress point: long context, large global batch, MoE all-to-all, large image/video, or HPC problem size.

Warm up compilation and caches. Run at least three measured repetitions unless the contract says otherwise. Save raw trials, telemetry, logs, command, environment, and correctness.

For SLO-constrained LLM serving, read [references/slo-serving-campaign.md](references/slo-serving-campaign.md) before launching the sweep. Use its campaign stages, per-request evidence, cliff-search, local-batch/graph alignment, MTP, HBM, interrupted-run recovery, and promotion rules. Do not treat a 750-request screening win as full acceptance.

### 7. Tune one hypothesis group at a time

Read [references/tuning-playbook.md](references/tuning-playbook.md). Use this order:

1. correctness, stability, and fit;
2. host/runtime/container consistency;
3. framework/engine/image;
4. precision and quantization;
5. TP/PP/DP/EP/CP and instance packing;
6. scheduler, batching, KV/activation/checkpointing;
7. attention/GEMM/MoE/optimizer/diffusion kernels;
8. RCCL, rank/NUMA/NIC binding and overlap;
9. partition/NPS mode as an approved maintenance experiment;
10. kernel/profile-guided changes and multi-node topology.

Keep a known-good launch artifact and exact rollback. Reject gains that exceed noise only in a profiled run, change correctness, or violate another SLO.

At an inference SLO cliff, sweep adjacent offered-load points even when throughput appears monotonic. Prefer a point with repeatable SLO margin over a higher-concurrency point that crosses the hard boundary intermittently. Align per-replica steady decode batch sizes with captured graph buckets before adding more load.

### 8. Compare and decide

Write results using [references/poc-result.schema.json](references/poc-result.schema.json), then compare:

```bash
python3 scripts/compare_poc_results.py \
  --baseline baseline.json --candidate candidate.json \
  --objective balanced --output comparison.json
```

An undeclared frozen-field difference is inconclusive. A failed correctness/preflight/stability gate is rejected even if faster.

For repeated inference summaries that use the schema described in the SLO-serving reference, run:

```bash
python3 scripts/summarize_slo_campaign.py \
  --summary <run-1>/summary.json --summary <run-2>/summary.json \
  --output-dir <campaign-dir> --target-tpm 1200000 \
  --max-mean-ttft-ms 2000 --minimum-rounds 2 --require-pass
```

Set `--minimum-rounds 3` for ordinary promotion unless the active contract explicitly permits fewer. Add `--require-backend-balance` only when backend imbalance is a hard gate rather than an observation.

### 9. Escalate profiling minimally

Use this ladder:

1. workload metrics and logs;
2. AMD SMI/process/CPU/network telemetry;
3. PyTorch profiler;
4. TraceLens;
5. Magpie framework/kernel analysis;
6. ROCm Systems Profiler for CPU/GPU/communication gaps;
7. ROCm Compute Profiler for selected kernels and roofline/counters.

Profile bounded representative windows. Counter replay and tracing perturb performance; keep profiled and unprofiled results separate.

### 10. Deliver and onboard

Report hardware/software fingerprint, gate status, workload contract, baseline, accepted/rejected experiments, variance, quality/stability, exact commands/digests, rollback, artifacts, and remaining risks. Label evidence as:

- `MI325X_MEASURED`
- `MI325X_DOCUMENTED`
- `GFX942_INFERRED`
- `UNSUPPORTED_PORT` (never accepted)

Use [references/portal-mapping.md](references/portal-mapping.md) for inventory and POC-system onboarding. Keep IP/host identity data-driven and verified by remote `hostname -f` plus `hostname -I`; never infer IPs from aliases.

## Stop conditions

Stop and request direction when:

- authorized SSH target, expected physical/logical GPU count, shared status, or maintenance scope is unknown;
- identity is not MI325X/`74a5`/`gfx942`;
- the next action needs `sudo`, install, driver reload, reboot, partition, clock/power, firmware/BIOS, network, firewall, container/service replacement, or production port;
- formal thresholds conflict with OEM guidance or the exact platform configuration is unknown;
- model/dataset revision, workload shape, correctness gate, or comparable baseline is missing;
- a result depends on mutable/nightly artifacts or undocumented flags without an experimental label.

## Reference routing

- Read [references/official-sources.md](references/official-sources.md) before version/image/flag choices.
- Read [references/mi325x-platform-constraints.md](references/mi325x-platform-constraints.md) before porting recipes.
- Read [references/acceptance-gates.md](references/acceptance-gates.md) for readiness and formal thresholds.
- Read [references/partition-and-virtualization.md](references/partition-and-virtualization.md) for SPX/DPX/QPX/CPX, NPS, and SR-IOV.
- Read [references/inference-playbook.md](references/inference-playbook.md) for vLLM, SGLang, ATOM, AITER, model fit, and quantization.
- Read [references/slo-serving-campaign.md](references/slo-serving-campaign.md) for closed-loop inference campaigns, per-request evidence, SLO-cliff search, graph-bucket alignment, MTP, HBM utilization, failure recovery, and promotion.
- Read [references/training-hpc-playbook.md](references/training-hpc-playbook.md) for Primus, Megatron, PyTorch/JAX, fine-tuning, and HPC.
- Read [references/network-and-collectives.md](references/network-and-collectives.md) for RCCL, RDMA, EP/MoE, and multi-node work.
- Read [references/benchmark-method.md](references/benchmark-method.md) and [references/tuning-playbook.md](references/tuning-playbook.md) for measurements and diagnosis.
- Read [references/portal-mapping.md](references/portal-mapping.md) for inventory/POC onboarding.
