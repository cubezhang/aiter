# Reproducible benchmark method

## Freeze the contract

Record host/GPU fingerprint, physical/logical GPU count, partition/NPS mode, topology, power/clocks, ROCm/driver/firmware, framework/engine/AITER/RCCL, container digest, model/dataset/workflow revisions, precision dimensions, parallelism, command/environment, client placement, input shape, warmup/repetitions, SLOs, and correctness.

Do not compare different frozen fields unless each difference is listed in `metadata.changes`.

## Four-point matrix

| Point | Inference | Training/HPC |
|---|---|---|
| Latency/small | low concurrency, representative ISL/OSL | small topology or latency-sensitive problem |
| Production | expected RPS/concurrency | target global batch/problem size |
| Saturation/scale | sweep to SLO knee | node/GPU scaling sweep |
| Stress | long context/MoE/multimodal | max sequence, checkpoint, all-to-all, large problem |

Warm up JIT, graph capture, autotuning and caches. Run at least three measured repetitions and retain all trials. Randomize candidate order when thermal/shared-system drift matters.

## Metrics

Inference: successes/failures, request/token throughput, SLO goodput, TTFT/TPOT/ITL/E2E p50/p95/p99, HBM, utilization, power/temperature, CPU/client saturation, quality.

Training: tokens/s/GPU, samples/s, step p50/p95, MFU, forward/backward/optimizer/communication breakdown, HBM, loss/gradient/overflow, checkpoint I/O, scaling efficiency, energy.

HPC/diffusion: define stage latency, throughput, achieved FLOPS/bandwidth, numeric/visual accuracy, first-run versus steady state, memory, power, and scaling.

## Correctness

Use golden prompts/tool schemas/model evaluation, loss/convergence tolerances, numeric residuals, output-image/video validation, no NaN/Inf/corruption, and stability/RAS gates appropriate to the workload. Performance with failed correctness is rejected.

## Variance and evidence

- Report individual trials, median, spread, and tails.
- Mark gains within noise inconclusive.
- Separate cold/warm, profiled/unprofiled, tuned/untuned-cache runs.
- Verify benchmark client, storage, CPU, and network are not limiting unintentionally.
- Preserve JSON, logs, telemetry, commands, config, traces, and digests.
- Freeze and hash the load generator and summarizer before measurement; never edit an active runner.
- Include initial dispatch through final drain in the measurement window and keep slow requests.
- Attribute token work per backend; equal request counts are not necessarily balanced work.
- Recover an interrupted run only when raw positions are complete, unique and valid; otherwise retain it as failure evidence under a new label.
- Near a hard SLO, require repeatable margin rather than accepting a median that hides failed rounds.

## Machine result

Write one JSON per configuration following `poc-result.schema.json`. Compare with `scripts/compare_poc_results.py`; its verdict is a gate, not a substitute for raw-trial review.
