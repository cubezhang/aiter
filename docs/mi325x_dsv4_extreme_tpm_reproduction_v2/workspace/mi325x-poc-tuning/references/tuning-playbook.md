# Symptom-driven MI325X tuning

## Model does not fit or startup OOM

1. Verify physical/logical devices, free HBM and partition mode.
2. Size weights, KV/activations, runtime/graphs, communication and reserve.
3. Reduce avoidable context/batch/cache; test eager mode if graph capture OOMs.
4. Use the smallest TP that fits; compare FP8 FNUZ/INT8/exact INT4 paths with quality gates.
5. For training, test sharding/recomputation/sequence parallelism and microbatch separately.
6. Reject native MXFP4/MXFP6/MXFP8 on `gfx942`.

## Low inference throughput

Sweep offered load, `max-num-batched-tokens`, sequences, chunked prefill, prefix cache, graph mode, instance packing, TP/DP, then kernels. Check tokenizer/client/host/network first. Compare AITER backends only after log verification.

At the saturation knee, test adjacent loads and replica-symmetric points. One less request can increase TPM when it aligns the local decode batch with a captured graph bucket or reduces queue/collective skew. Capture bounded graph buckets around the observed per-replica batch instead of assuming only powers of two.

## High TTFT

Separate queueing, tokenization/preprocessing, prompt length, chunked prefill, prefix-cache hit, prefill attention, TP collective, cold JIT/graph and network. Report cold and steady state separately.

Keep attention-kernel chunk size, scheduler long-prefill threshold and maximum batched tokens as separate experiments. The first changes kernel partitioning; the second changes request interleaving; the third changes admitted work. A setting that fits HBM can still worsen TTFT by monopolizing compute.

## High TPOT/ITL

Check decode attention, KV layout/precision, small-batch kernels, graph mode, TP/RCCL latency, host launch gaps, speculative acceptance, clocks/power. FP8 KV can accelerate a kernel yet lose end to end through casts; use a paired trace.

For speculative decoding, record accepted-token distribution and average tokens per target forward for every depth, then decide from end-to-end TPM/TTFT/TPOT. Higher draft acceptance or more tokens per forward alone does not prove a gain.

## Low training throughput/MFU

Break down input, forward, backward, optimizer, collectives, bubbles and checkpointing. Tune microbatch/accumulation, recomputation, TP/PP/DP/EP/CP, bucket sizes/overlap, fused kernels, precision and input pipeline one group at a time. Compare equal global tokens and convergence.

## Poor scaling

Establish single-node/minimum-fit baseline, run exact collective/message sizes, verify rank/NUMA/NIC binding, inspect overlap and imbalance, then modify communication settings. For MoE inspect all-to-all-v and expert imbalance. Do not infer workload scaling from all-reduce peak.

## HBM bandwidth below expectation

Validate clocks/power/thermals, partition/NPS/locality, access pattern, alignment/coalescing, cache behavior, concurrency and measurement method. Compare against official acceptance recipe before custom kernels. Partition changes are a separate approved maintenance experiment.

## Low utilization or gaps

Classify CPU/input/tokenizer/storage bound, scheduler/launch overhead, rank synchronization, network wait, memory-bound kernels, or insufficient work. Use Systems Profiler before Compute Profiler.

## RAS, resets, or throttling

Stop performance work. Capture before/after ECC/bad pages, dmesg, XGMI/PCIe, power/temperature/clocks, workload and firmware/driver. Do not hide instability by lowering load or resetting counters.

## Candidates, not defaults

- AITER master enablement and exact MHA/MLA/MoE backends
- `ROCM_AITER_MLA` versus `ROCM_AITER_TRITON_MLA` on `gfx942`
- `TORCH_BLAS_PREFER_HIPBLASLT=1`
- graph/compilation modes
- FP8 FNUZ weights/KV/training
- Quick Reduce collective quantization
- TP/DP/EP/DP Attention and instance packing
- Primus MI300X/MI325X-specific environment switches in the exact recipe
- deterministic/performance clock mode
- SPX/NPS1 versus supported partition/locality profile

Verify each against current pinned docs/source, run correctness, measure the fixed matrix, then keep or roll back. Do not accumulate old environment variables across image upgrades.
