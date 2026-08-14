# Official sources and freshness policy

Checked 2026-08-11. Reopen live sources before installation, mutation, or version-specific flags.

## Hardware and architecture

- Product specifications: https://www.amd.com/en/products/accelerators/instinct/mi300/mi325x.html
- MI325X platform datasheet: https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/instinct-mi325x-platform-datasheet.pdf
- Current GPU specs: https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html
- CDNA 3 architecture white paper: https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf
- HIP FP8/FNUZ types: https://rocm.docs.amd.com/projects/HIP/en/latest/reference/fp8_numbers.html

## Acceptance, partitioning, and operations

- Exact MI325X acceptance page and thresholds: https://instinct.docs.amd.com/projects/system-acceptance/en/latest/gpus/mi325x.html
- Customer Acceptance Guide: https://instinct.docs.amd.com/projects/system-acceptance/en/latest/
- System validation: https://instinct.docs.amd.com/projects/system-acceptance/en/latest/common/system-validation.html
- Network/cluster validation: https://instinct.docs.amd.com/projects/system-acceptance/en/latest/network/validation.html
- GPU cluster networking: https://instinct.docs.amd.com/projects/gpu-cluster-networking/en/latest/
- AMD SMI partitioning: https://rocm.docs.amd.com/projects/amdsmi/en/develop/conceptual/partition.html
- AMD SMI CLI: https://rocm.docs.amd.com/projects/amdsmi/en/latest/how-to/amdsmi-cli-tool.html

## ROCm and frameworks

- Current ROCm compatibility matrix: https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html
- Current release notes: https://rocm.docs.amd.com/en/latest/about/release-notes.html
- ROCm AI ecosystem: https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/
- MI300/MI325 workload optimization: https://rocm.docs.amd.com/en/docs-7.2.4/how-to/rocm-for-ai/inference-optimization/workload.html
- vLLM serving: https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/inference/vllm.html
- vLLM V1 optimization: https://rocm.docs.amd.com/en/docs-7.2.4/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html
- SGLang serving: https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/inference/sglang.html
- Model quantization: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/model-quantization.html
- AITER: https://github.com/ROCm/aiter
- ATOM: https://rocm.docs.amd.com/projects/atom/en/latest/

## Training and HPC

- Primus documentation: https://rocm.docs.amd.com/projects/primus/en/latest/
- Primus installation/support: https://rocm.docs.amd.com/projects/primus/en/latest/01-getting-started/installation.html
- Primus + Megatron recipe: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/primus-megatron.html
- ROCm training hub: https://rocm.docs.amd.com/en/develop/how-to/rocm-for-ai/training/index.html
- RCCL: https://rocm.docs.amd.com/projects/rccl/en/develop/
- ROCm Systems Profiler: https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/
- ROCm Compute Profiler: https://rocm.docs.amd.com/projects/rocprofiler-compute/en/develop/

## Freshness rules

1. Exact MI325X/OEM page beats generic MI300X guidance.
2. Current compatibility/release beats a versioned recipe for installation; preserve the recipe version when reproducing.
3. `gfx942` MI300X guidance is an inference until validated on MI325X.
4. MI350/MI355 `gfx950` MX/OCP FP8 paths are non-portable.
5. Pin images, repositories, models and datasets to immutable digests/revisions.
6. Save source URL and access date with the result.
7. If current sources conflict, stop and report the conflict.
