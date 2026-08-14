# MI325X platform constraints and portability gates

## Contents

1. Identity and hardware
2. Precision and quantization
3. Kernel and binary compatibility
4. Memory and model fit
5. Topology and communication
6. Power, thermals, and storage
7. Portability checklist

## 1. Identity and hardware

Require all three signals:

- PCI vendor/device `1002:74a5`
- AMD SMI market name containing `MI325X`
- LLVM target `gfx942`

Official per-accelerator reference values are 256 GB HBM3E, up to 6 TB/s peak memory bandwidth, 304 active CUs across eight XCDs, 1,216 matrix cores, 256 MB last-level Infinity Cache, PCIe 5.0 x16, and up to 1000 W board power. The accelerator product page lists eight Infinity Fabric links, while the eight-GPU platform datasheet describes seven 128 GB/s scale-up links per GPU in its ring. A standard UBB 2.0 platform commonly exposes eight OAM GPUs and 2,048 GB aggregate HBM. Detect the OEM topology; SR-IOV or accelerator partitioning changes visibility.

Product specifications are capability ceilings, not acceptance measurements. Use the exact AMD/OEM MI325X page for thresholds.

## 2. Precision and quantization

MI325X is CDNA3 `gfx942`:

| Path | Rule |
|---|---|
| BF16 / FP16 | Known-good training and inference baselines |
| FP32 / TF32 / FP64 | Supported; select for accuracy/HPC rather than LLM efficiency |
| FP8 | Native FNUZ E4M3/E5M2 path; require matching scales/kernels |
| INT8 | Native candidate with accuracy gate |
| OCP FP8-only checkpoint/kernel | Not native `gfx942`; require documented conversion/path |
| MXFP8 / MXFP6 / MXFP4 | No native CDNA3 support; reject `gfx950` recipes |
| AWQ/GPTQ/INT4 W4A16 | Integer-weight compression, not native FP4; exact kernel and quality gate required |
| INT4-FP8 two-level Quark | Storage/compute combination; verify actual FP8 FNUZ compute and conversion overhead |
| FP8 KV cache | Capacity/bandwidth candidate; benchmark end to end |
| Quick Reduce INT4/6/8 | Collective quantization, not model precision |

Record independently: checkpoint storage, weight compute, activation compute, KV/cache precision, gradient/optimizer precision, and collective quantization. Model metadata alone does not prove executed kernels.

FNUZ lacks infinities and differs from OCP FP8 semantics. Before accepting FP8, inspect immutable quantization metadata, verify the engine's `gfx942` path, test NaN/Inf and accuracy, and compare conversion/cast overhead against BF16.

## 3. Kernel and binary compatibility

Pin host driver/firmware, ROCm user space, Python/glibc, PyTorch/JAX, vLLM/SGLang/ATOM/Primus, AITER/CK/Triton/hipBLASLt, RCCL, and communication add-ons. Resolve every container tag to a digest.

`gfx942` coverage is necessary but not sufficient. Run representative required operators before a large load. Do not reuse `gfx950` binaries, AITER tuning CSVs, graph artifacts, JIT caches, or online-tuning results. MI300X artifacts are also not automatically optimal because MI325X HBM speed/capacity changes shapes and bottlenecks.

Promote a path to `MI325X_MEASURED` only after logs/traces prove the backend and fixed-workload correctness/performance passes.

## 4. Memory and model fit

Do not equate advertised 256 GB with a safe allocator budget. Reserve HBM for runtime, graphs, fragmentation, KV cache or activations, collectives, and transient workspaces. Use measured free memory after environment startup.

For inference:

```text
per-rank HBM = sharded weights + KV cache + graph/workspace + temporary communication + reserve
```

For training:

```text
per-rank HBM = parameters + gradients + optimizer states + master weights + activations + communication buckets + reserve
```

Account for ZeRO/FSDP sharding, recomputation, sequence/context parallelism, MoE expert placement, and uneven layers. A model that fits eight MI325X GPUs may be faster as several smaller replicas; measure TP cost versus instance packing.

Logical CPX partitions divide the same physical resources and HBM. Do not multiply visible logical-device count by 256 GB.

## 5. Topology and communication

Detect physical BDFs, XGMI hops, logical partitions, CPU/NUMA locality, GPU/NIC proximity, RDMA devices, and rank mapping. A standard eight-GPU platform is designed as an Infinity Fabric island, but validate actual links and OEM wiring.

Use RCCL as the baseline for TP/DP collectives. Run all-reduce, all-gather, reduce-scatter, broadcast, send/recv, and all-to-all/all-to-all-v for MoE/EP as applicable. Peak collective bandwidth does not prove application scaling; correlate overlap, imbalance, host gaps, and end-to-end metrics.

Do not copy an MI350/MI355 MORI/DeepEP or disaggregation recipe. Verify exact MI325X/gfx942, NIC, ROCm, PyTorch, and library support first.

## 6. Power, thermals, and storage

MI325X has a high power envelope. Record power cap, clocks, temperature, throttle reasons, PSU/rack limits, and cooling state. Do not change deterministic/performance mode or power caps during the first baseline.

Model/dataset fit also requires storage capacity, inode headroom, read throughput, `/dev/shm`, container writable-layer space, checkpoint save bandwidth, and temporary download/unpack space. Avoid filling a shared filesystem; use an approved cache and cleanup plan.

## 7. Portability checklist

Before reusing a recipe, ask:

- Does it explicitly name MI325X or only MI300X/MI308X/MI350X/MI355X?
- Is the device `74a5` and target `gfx942`?
- Does it require OCP FP8, MX formats, CDNA4 instructions, 160 KB LDS, or 8 TB/s HBM?
- Is the partition/NPS mode and visible physical/logical GPU count identical?
- Are ROCm, driver, framework, AITER, RCCL, model/dataset, precision, and container digest compatible?
- Are TP/PP/DP/EP/CP, context/sequence, concurrency/batch, and graph modes the same?
- Was the gain kernel-level or end-to-end, and did correctness/SLO pass?

Unknown answers make the recipe a hypothesis seed, not a baseline.
