# MI325X inference playbook

## Contents

1. Engine selection
2. Container and ABI preflight
3. vLLM
4. SGLang
5. ATOM and vLLM-ATOM
6. Quantization and HBM
7. Parallelism and MoE
8. Diffusion and multimodal
9. Acceptance

## 1. Engine selection

Use the narrowest official exact path:

| Need | First candidate |
|---|---|
| Broad OpenAI-compatible LLM serving | ROCm vLLM |
| Exact SGLang-supported model/features | ROCm SGLang |
| Exact ROCm-native model recipe | native ATOM |
| vLLM scheduler/API plus supported ATOM kernels | vLLM-ATOM plugin |
| Diffusion/video | current ROCm xDiT/ComfyUI/PyTorch recipe |
| Portability/control | PyTorch/Transformers baseline |

Keep engines as separate measured configurations. Do not translate flags mechanically.

## 2. Container and ABI preflight

Prefer an AMD-published image for the exact release and resolve its digest. Current ROCm 7.14 docs publish a CDNA vLLM 0.23.0 image with PyTorch 2.11/Python 3.14, but recheck before use.

Inside the pinned image:

```bash
python - <<'PY'
import platform, torch
print(platform.python_version(), torch.__version__, torch.version.hip)
print(torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(i, torch.cuda.get_device_name(i), getattr(p, "gcnArchName", None), p.total_memory)
PY
```

Then print engine/AITER/RCCL versions and test representative attention, GEMM, MoE/norm and collective paths. Require eight MI325X/`gfx942` devices only when the contract expects a standard unpartitioned node.

## 3. vLLM

Start with current official ROCm vLLM, `CUDA_VISIBLE_DEVICES`, AITER master enablement where documented, and no legacy sub-flag cargo cult. Confirm effective attention/MoE/graph backends in logs.

Select attention deliberately:

- MHA: AITER/CK/Triton path supported by the pinned image.
- MLA: on MI300X/MI325X `gfx942`, current AMD guidance identifies `ROCM_AITER_TRITON_MLA` as a possible 2–3% TPS candidate versus `ROCM_AITER_MLA`; measure it.
- DSA/sparse MLA: use its exact recipe and block-size requirements; ordinary MLA rules do not prove DSA support.

Sweep one group at a time:

- context, `max-num-seqs`, `max-num-batched-tokens`;
- `gpu-memory-utilization` and instance packing;
- chunked prefill and prefix caching;
- graph/compilation mode;
- scheduler/async scheduling;
- FP8 model/KV cache;
- TP/DP/EP/DP Attention;
- Quick Reduce only after a non-quantized collective baseline.

Quick Reduce INT4/INT6/INT8 quantizes communication, not model compute. Quality-gate it.

## 4. SGLang

Use the current AMD SGLang image/recipe and pin its digest. Verify model architecture, attention/MoE backend, AITER, graph mode, tokenizer/chat/tool/reasoning behavior, and `gfx942` kernels before load.

Do not assume vLLM environment variables apply. Establish the same request contract and correctness suite, then compare scheduling, prefix cache, radix cache, speculative decoding, TP/DP/EP, and disaggregation as separate experiments.

## 5. ATOM and vLLM-ATOM

ATOM's general supported-GPU table currently names MI300X `gfx942`, not MI325X. Treat MI325X as `GFX942_INFERRED` until an exact current recipe/model names it and local preflight passes.

Choose exactly one mode:

- Native ATOM owns API, scheduling, KV cache, execution and AITER.
- vLLM-ATOM keeps vLLM's API/scheduler and replaces supported execution paths.

Require exact architecture/quantization registry coverage and TP divisibility. Resolve mutable images/nightlies. Verify plugin/model/backend registration from logs. Reject MI355X MXFP4/A4W4 recipes on MI325X.

Use vanilla vLLM as the control when ATOM support is inferred. Benchmark health, chat/completions, streaming, tool/reasoning, long context, and model-specific quality identically.

## 6. Quantization and HBM

Known candidates:

- BF16/FP16 baseline;
- FNUZ FP8 W8A8 or exact Quark/compressed-tensors checkpoint;
- FP8 KV cache;
- INT8;
- exact AWQ/GPTQ/INT4 W4A16 path;
- Quark INT4 storage + FP8 compute only with exact metadata/kernel support.

Reject native MX formats. A compressed filename does not prove the executed compute format. Measure weight load, per-rank HBM, KV capacity, casts/conversions, quality, long-context stability, and end-to-end latency/goodput.

The 256 GB HBM can reduce TP or permit more KV cache/instances. Compare:

- smallest TP that fits;
- TP8 full-node;
- several TP2/TP4 or single-GPU replicas;
- BF16 versus FP8 capacity and conversion cost.

Report `fits`, `stable`, and `improves SLO goodput` as separate conclusions. More
replicas can reduce local batch efficiency; a larger token budget can increase prefill
blocking. Capacity headroom is a reason to test a topology or batch, not evidence that it
is faster.

## 7. Parallelism and MoE

- Dense: TP for fit/latency; DP replicas for aggregate throughput; PP across nodes only when fit/topology requires.
- MoE: run all-to-all/all-to-all-v first; compare TP, TP+EP and DP Attention+EP.
- High-concurrency MLA+MoE: DP Attention can avoid replicated KV; measure crossover.
- Ultra-sparse MoE: all-to-all may dominate; do not default to EP.
- Disaggregation/MORI/DeepEP/Infera: only with exact MI325X/gfx942 and NIC support; separately validate transport.

Record visible devices, physical BDFs, logical partitions, rank groups, CPU/NIC binding, rendezvous, and communication backend.

## 8. Diffusion and multimodal

Freeze media count, dimensions, frames, workflow/graph, model revisions, custom nodes, seed/sampler, and preprocessing. Separate host decode/resize/tokenization, vision encoder, diffusion transformer/UNet, VAE, LLM prefill, and decode.

Use an exact AMD xDiT/ComfyUI/PyTorch route. Measure first-run compilation separately, avoid untrusted custom nodes in the baseline, and test data/tensor/sequence parallelism only after correctness. Record images/s, videos/s, latency, HBM, host memory, and output quality.

## 9. Acceptance

Require fixed workload, zero unexpected failures, correct backend/precision, no new RAS errors/resets, stable memory/power/thermal state, repeated metric improvement beyond noise, and pinned artifacts. A faster kernel without end-to-end/SLO gain is rejected or inconclusive.

For closed-loop serving near a hard SLO, search adjacent concurrency and prefer a
repeatable margin point. Keep screening, cold-restart reproduction and full acceptance as
different evidence levels. Read `slo-serving-campaign.md` through the main skill routing
before designing the campaign.
