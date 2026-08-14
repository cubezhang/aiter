# Training, fine-tuning, and HPC playbook

## Contents

1. Training path selection
2. Primus and distributed training
3. Memory and parallelism
4. Training measurements
5. Fine-tuning
6. HPC and kernel POCs
7. Correctness and acceptance

## 1. Training path selection

Prefer AMD Primus for a reproducible supported training baseline. Current Primus supports MI325X with Megatron-LM, TorchTitan, JAX MaxText, Megatron Bridge, and related backends. Use direct PyTorch/JAX only when the POC requires custom code or a control.

Pin image digest, Primus/repository release, backend commit, ROCm, PyTorch/JAX, Transformer Engine, FlashAttention, hipBLASLt, Triton/CK, RCCL, dataset/tokenizer/model, and config YAML.

## 2. Primus and distributed training

Start with the official image and a built-in GEMM verification, then a small mock-data training job. Current AMD examples provide MI300X configuration files for the shared `gfx942` path and explicitly note MI325X/MI300X environment candidates. Treat an MI300X-named config as `GFX942_INFERRED` until its effective MI325X values and performance are verified.

Never copy network variables. Detect NICs, RDMA devices, GID, routes, and GPU/NIC locality. The current examples suggest variables such as `NCCL_IB_HCA`, `NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME`, and a RoCE GID only after site-specific discovery.

Run order:

1. single GPU forward/backward correctness;
2. one-node 8-GPU or contract topology;
3. short checkpoint save/resume;
4. multi-node smoke;
5. production-shape warmup and measured steps;
6. bounded stability/scale run.

## 3. Memory and parallelism

Account for parameters, gradients, optimizer states, master weights, activations, temporary workspaces, communication buckets, and fragmentation.

Tune:

- tensor parallel (TP);
- pipeline parallel (PP) and virtual pipeline stages;
- data parallel (DP);
- expert parallel (EP);
- context/sequence parallel (CP/SP);
- FSDP/ZeRO sharding;
- activation checkpointing/recomputation;
- microbatch/global batch and gradient accumulation;
- optimizer/offload only when host/network/storage capacity supports it.

The larger 256 GB HBM may reduce sharding or recomputation, increase microbatch, or permit fewer GPUs per model. Measure whether the additional HBM improves tokens/s/GPU or merely increases batch latency.

For FP8 training, verify FNUZ-compatible Transformer Engine/kernels, scaling strategy, loss curves, overflow/NaN, convergence proxy, and checkpoint portability. Keep BF16 control runs.

## 4. Training measurements

Record:

- tokens/s, samples/s, tokens/s/GPU;
- step time p50/p95 and warmup/compile time;
- model FLOP utilization/MFU with formula and peak basis;
- forward/backward/optimizer/communication time;
- HBM peak, activation/optimizer footprint;
- all-reduce/reduce-scatter/all-gather/all-to-all time and overlap;
- checkpoint save/load time and storage throughput;
- loss, gradient norm, overflow/skipped updates, accuracy proxy;
- power/energy per token or sample and throttling;
- scaling efficiency versus single-node/minimum topology.

Compare equal global batch, sequence length, tokens, optimizer, data order, precision, and convergence gate.

## 5. Fine-tuning

For LoRA/QLoRA/SFT/DPO/RL or custom post-training, record base model revision, adapter rank/targets, quantization, optimizer, dataset/revision, prompt template, sequence packing, evaluation suite, and merge/export path.

Validate training loss and held-out quality, resume, adapter load/merge, and inference compatibility. Do not call a lower loss at different effective batch/tokens a performance improvement.

## 6. HPC and kernel POCs

Establish vendor library baselines before custom kernels:

- rocBLAS/hipBLASLt GEMM;
- rocFFT/rocSPARSE/rocSOLVER as applicable;
- BabelStream/STREAM-like memory bandwidth;
- TransferBench/RCCL for movement;
- rocHPL/HPCG or domain application;
- HIP/Triton/CK custom kernel correctness and Magpie comparison.

Compile explicitly for `gfx942`. Freeze problem sizes, strides/layout, precision, warmup, synchronization, clocks/power, and validation tolerance. Report achieved bandwidth/FLOPS plus percent of an appropriate measured or product peak, not only kernel time.

Use ROCm Compute Profiler for dominant kernels and roofline/counters after Systems Profiler proves the workload is GPU-kernel bound.

## 7. Correctness and acceptance

Reject training/HPC candidates with divergent loss, NaN/Inf, excessive numeric error, failed resume, nondeterministic corruption, memory errors, RAS growth, rank hangs, or unstable scaling. Keep seeds and determinism settings documented; bitwise equality is not always expected, so define tolerances before tuning.
