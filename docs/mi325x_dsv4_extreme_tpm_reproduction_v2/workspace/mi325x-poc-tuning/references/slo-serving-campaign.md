# SLO-constrained inference campaign

## Contents

1. Separate campaign stages
2. Freeze the contract and harness
3. Execute an auditable run
4. Search the SLO cliff
5. Align replicas and graph buckets
6. Distinguish prefill controls
7. Evaluate speculative decoding
8. Convert HBM capacity into useful work
9. Preserve routing and workload validity
10. Recover interrupted or failed runs
11. Promote a candidate
12. MI325X DeepSeek-V4 case study

## 1. Separate campaign stages

Label every result as exactly one stage:

| Stage | Purpose | May support final acceptance? |
|---|---|---|
| Smoke | Endpoint, shape, output and basic correctness | No |
| Warmup | JIT, graph capture, autotuning and cache stabilization | No |
| Screening | Reject/promote knobs with a fixed bounded manifest | No |
| Reproduction | Repeat a promoted point, including cold restart | Not by itself |
| Full acceptance | Contract manifest, restart sequence and required repetitions | Yes |

Do not upgrade evidence by renaming it. A short screening can establish that a target
has been reached experimentally, but not that a larger formal campaign has passed.
When the user changes the immediate priority, preserve the frozen contract and record
the reduced priority gate separately instead of silently weakening the contract.

## 2. Freeze the contract and harness

Freeze before comparison:

- digest-pinned server and load-generator images;
- source commit plus complete dirty-worktree patch;
- model, tokenizer, quantization metadata, dataset and manifest hashes;
- request body, sampling, exact output length, streaming parsing and retry policy;
- topology, visible devices, precision dimensions, MTP depth, cache policy and router;
- offered-load model, warmup, repetition count, measurement window and SLOs;
- result schema, summarizer, telemetry, health/RAS checks and rollback.

Treat the harness as part of the benchmark. Hash it before the first measured run and
do not edit it while load is active. A runner edit during an active process can leave
complete raw requests but fail the orchestration tail; recover from raw evidence and
label the incident rather than pretending the run ended normally.

Use a deterministic finite manifest for acceptance-like work. Attempt every entry
exactly once. Do not retry away failures, sort by favorable prompt length, drop slow
drain requests, or change thinking/tool allocation between candidates.

## 3. Execute an auditable run

For every measured run:

1. Resolve the exact service/router containers and backend ports.
2. Refuse non-empty result directories and ambiguous service ownership.
3. Capture router state, container inspect, RAS/ECC and server-log start offsets.
4. Start GPU utilization/HBM/power/temperature telemetry.
5. Dispatch the fixed manifest under the selected offered load.
6. Preserve one record per attempt with monotonic start/first-token/last-token/end times.
7. Capture router, RAS, kernel delta and backend log deltas after the run.
8. Validate manifest positions, worker shards, output length, cache hits and backend attribution.
9. Compute metrics from raw records, including the full initial dispatch through final drain.
10. Hash summaries and raw evidence.

At minimum record total/input/output TPM, success rate, TTFT/TPOT/E2E mean and tails,
prompt/output-token distributions, backend token work, HBM and RAS/stability. Request
counts alone do not demonstrate balanced work when prompt lengths vary.

## 4. Search the SLO cliff

Use coarse-to-fine closed-loop concurrency or rate search:

1. Find an obviously legal point and the first repeatably illegal point.
2. Search the interval with steps of 2–3, then adjacent points.
3. Repeat each candidate with the same manifest and warm state.
4. Cold-restart the promoted configuration and repeat it.

Do not assume throughput increases monotonically with concurrency. At saturation, one
less request can improve both throughput and latency by reducing padding, queueing,
collective skew or graph-bucket mismatch. Always test adjacent points around the knee.

A hard-boundary result needs margin. If repeated Mean TTFT values straddle 2 seconds,
the configuration is a cliff diagnostic, not a stable winner. Prefer the nearest lower
load that still clears the throughput objective on every required repeat.

## 5. Align replicas and graph buckets

For `R` independent replicas and total closed-loop concurrency `C`, start with local
steady decode load near `C/R`. Test total concurrency divisible by `R`; symmetric local
load often reduces router oscillation and produces more repeatable graph selection.

Capture graph buckets around observed local decode batch sizes, not only powers of two.
For example, if two replicas sustain about 28 requests each, test an exact bucket at 28
and bounded neighbors. Confirm from logs that graphs capture and replay without eager
fallback or excessive memory cost.

Graph changes and prefill changes can interact. Isolate them first, then test the
combination when each has a plausible mechanism. A combination may win even when each
isolated screen is marginal; retain the component results so the interaction is visible.

## 6. Distinguish prefill controls

Do not conflate:

- attention prefill kernel chunk size: changes internal kernel work partitioning;
- scheduler long-prefill threshold: splits a request across scheduler iterations;
- maximum batched tokens: controls total admitted work;
- chunked-prefill enablement: allows partial prefill behavior;
- prefix cache: changes effective workload and must follow the contract.

A larger kernel chunk can reduce kernel overhead while leaving request order intact. A
smaller scheduler threshold can improve interleaving but also starve partial prefills,
inflate tail TTFT or expose mixed-prefill/decode bugs. Increasing batched tokens may fit
in HBM yet worsen both TPM and TTFT because long prefill monopolizes compute.

Inspect scheduler invariants when partial prefill mixes with speculative decode. If a
compact per-decode array is indexed by full-batch positions, deferred rows can shift new
decode indices out of range. Fix the indexing domain minimally, add an assertion/test,
and rerun the exact failure shape before measuring performance.

## 7. Evaluate speculative decoding

Do not select MTP/speculative depth from acceptance rate alone. For every depth record:

- accepted drafts / proposed drafts;
- distribution of 0..K accepted drafts;
- average output tokens per target forward;
- target/draft forward cost and HBM;
- end-to-end TPM, TTFT and TPOT under the fixed workload.

Increasing depth usually lowers per-draft acceptance. A deeper setting can produce more
tokens per forward yet lose end to end because verification and predictor cost grows.
Compare at least off/K1/K2 when feasible. If off was not measured, report only the best
among tested depths; do not claim absolute gain from MTP.

Verify exact output limits under speculative execution. A verification step can cross
`max_tokens`; trim surplus emitted tokens before usage accounting and require every
successful request to match the frozen completion length.

## 8. Convert HBM capacity into useful work

Treat MI325X's 256 GB HBM as an option set, not an automatic speedup. It can enable:

- smaller TP and more replicas;
- larger KV capacity or context;
- larger graph/workspace footprints;
- higher batch limits;
- model variants that otherwise spill or fail to fit.

Measure each option end to end. More replicas can shrink local batches enough that
kernel efficiency and communication savings are lost. Larger batches can increase
prefill blocking. Record that a configuration fits separately from whether it improves
SLO goodput. Keep the topology frozen for direct comparison or version the contract.

## 9. Preserve routing and workload validity

Use transparent routing and attribute every response to a backend. Compare request count,
active count and token work. Least-connections can balance requests while token TPM remains
skewed because prompt lengths differ.

Do not add prompt-length/content-aware routing when the contract forbids favorable
partitioning. Prefer contract-neutral fixes: symmetric total concurrency, exact local
graph buckets, better queue accounting or a newly versioned routing experiment.

## 10. Recover interrupted or failed runs

Before rerunning, inspect raw artifacts:

- If every manifest position exists exactly once and raw records are complete, rerun only
  the deterministic summarizer and collect missing post-run evidence. Mark orchestration
  metadata as recovered.
- If records are missing, duplicated, truncated or lack output/usage, keep the partial run
  as failure evidence and rerun under a new label.
- If a few requests remain stuck beyond the established tail envelope, inspect backend
  activity, power, RAS and logs. Stop the client as a stability failure; never summarize
  the partial set as a valid full run.
- If a candidate crashes, stop only its tagged containers, preserve logs and verify GPUs
  return to a clean state before the next hypothesis.

Never modify the active runner to repair a live campaign. Patch after the process is
stopped, validate syntax, and use a new run label.

## 11. Promote a candidate

Promote only when:

- every required repeat passes throughput, SLO, success, output and integrity gates;
- the improvement exceeds observed noise and survives final drain;
- no OOM, graph fallback, engine death, RAS delta or hidden retry occurs;
- cold restart reproduces the mechanism;
- backend imbalance meets the active contract or is explicitly observational;
- exact deployment and rollback artifacts exist.

Use at least three measured screening repetitions by default. Two repetitions can support
an explicitly reduced exploratory priority gate, but not a formal stability claim. Use the
full manifest and restart schedule for final acceptance.

## 12. MI325X DeepSeek-V4 case study

This measured case illustrates the method; it is not a portable default configuration.

Frozen workload: two TP4 replicas, FP8 checkpoint/KV, Quick Reduce INT4, MTP2, exact
1024-token streaming output, fixed 750-request screen, least-connections routing.

| Experiment | Result | Reusable conclusion |
|---|---|---|
| Five full baseline rounds at C53 | Median 1,152,744 TPM; Mean TTFT legal | Stable baseline was 4.10% below 1.2M |
| MTP1, C57 screen | 1,150,464 TPM; 76.6% acceptance; 1.77 tokens/forward | Higher acceptance did not offset lower yield in this screen |
| MTP2 | About 59%/draft; about 2.18 tokens/forward | Best tested speculative depth |
| MTP3, C53 screen | 1,065,749 TPM; about 44.7%/draft; about 2.34 tokens/forward | More tokens/forward still lost end to end in this screen |
| Four TP2 replicas, C53 | Fit with substantial HBM headroom; 1,146,691 TPM; TTFT 2.021s | Smaller TP/more replicas fit but small local batches erased benefit |
| Max batched tokens 262K, C55 | Fit; 1,094,585 TPM; TTFT 2.529s | HBM capacity did not become SLO goodput |
| Scheduler long-prefill threshold 16K, C57 | 1,154,105 TPM; TTFT 2.650s after bug fix | Scheduler slicing worsened tail and exposed compact-index bug |
| 32K prefill kernel + exact local graph buckets, C57 | 1.226M–1.233M; Mean TTFT 1.969–2.082s | Throughput improved, but hard TTFT cliff was unstable |
| Same configuration, symmetric C56 | 1,243,062 and 1,247,987 TPM; Mean TTFT 1.931 and 1.930s | One less request improved throughput and produced SLO margin |

The C56 two-round median was 1,245,524 TPM with 0.28% CV. One backend-imbalance
observation was 5.13%, so this remained a screening/reproduction result rather than the
full contract acceptance. The critical skill point was to test local graph/batch symmetry
and adjacent load, not to keep increasing concurrency or HBM allocation.

The MTP1/MTP3 rows were bounded candidate screens at different listed loads, and some
candidate rows were warmup measurements. Use them as directional rejection evidence, not
as a formal matched MTP-off/K1/K2/K3 comparison. MTP-off remained unmeasured in this case.
