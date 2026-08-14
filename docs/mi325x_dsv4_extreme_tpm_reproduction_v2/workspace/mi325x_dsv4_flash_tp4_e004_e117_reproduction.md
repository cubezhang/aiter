# MI325X DeepSeek-V4-Flash-FP8 TP4 Reproduction

This directory contains the retained general-purpose E004/E117 configuration for DeepSeek-V4-Flash-FP8 on MI325X, together with the original files required to reproduce the results.

## Reference Result

Reference performance of the retained configuration:

* ATOM short benchmark, 140 requests: `18042.34 tok/s`
* Mean TPOT: `20.76 ms`
* Original Locust benchmark, 750 requests: `1,580,708 TPM`
* Locust TTFT: `35.43 ms`
* Locust TPOT: `29.69 ms`

This configuration uses dual TP4 instances with a Router and is designed for mixed workloads containing both short and long requests.

## Included Files

* Original startup script: `docs/mi325x_dsv4_flash_tp4/commands/e004_start_dual_tp4_131k_singlecontainer.sh`
* Original stop script: `docs/mi325x_dsv4_flash_tp4/commands/e004_stop_dual_tp4_131k_singlecontainer.sh`
* ATOM benchmark script: `docs/mi325x_dsv4_flash_tp4/commands/e004_run_atom_6100_1024_u70.sh`
* Original Locust script: `docs/mi325x_dsv4_flash_tp4/commands/e003_run_original_locust.sh`
* Retained 30-row A8W8 tuning table: `aiter/configs/model_configs/dsv4_flash_mi325x_tp4_general_a8w8.csv`
* C57 top-k source code: `docs/mi325x_dsv4_flash_tp4/topk_per_row_kernels.cu`
* Router: `docs/mi325x_dsv4_flash_tp4/router.py`
* Locust benchmark program: `docs/mi325x_dsv4_flash_tp4/locustfile.py`
* ATOM source and commit: `docs/mi325x_dsv4_flash_tp4/atom_origin.txt`, `atom_commit.txt`
* Container image digest: `docs/mi325x_dsv4_flash_tp4/container_image.txt`
* File checksums: `docs/mi325x_dsv4_flash_tp4/sha256sums.txt`

## Host Requirements

* 8 × MI325X GPUs
* Docker must have access to `/dev/kfd` and `/dev/dri`
* ROCm must be properly configured
* The host must be configured with `iommu=pt`
* `kernel.numa_balancing=0`
* The model must be located at:

```bash
/data/DeepSeek-V4-Flash-FP8
```

> Do not set `AITER_REBUILD=1`. This variable forces AITER JIT modules to be rebuilt and may cause startup failures due to missing prebuilt modules or differences in the compilation environment.

## 1. Clone Source

```bash
cd /data

git clone https://github.com/cubezhang/aiter.git
cd aiter

git switch mi325-dsv4-ops
git pull --ff-only origin mi325-dsv4-ops

export REPO_ROOT=$PWD
export RUN_ROOT=/data/models/mi325_dsv4_perf_tuning_20260803_065439
```

Check the versions required for this reproduction:

```bash
cat docs/mi325x_dsv4_flash_tp4/container_image.txt
cat docs/mi325x_dsv4_flash_tp4/atom_origin.txt
cat docs/mi325x_dsv4_flash_tp4/atom_commit.txt
cat docs/mi325x_dsv4_flash_tp4/sha256sums.txt
```

## 2. Required Host Files

The original E004 script uses the following validated host dependency paths:

```bash
# Model
/data/DeepSeek-V4-Flash-FP8

# Official ATOM source
/data/models/mi325_dsv4_reuse_tuning_20260801_103501/02_source/ATOM_official

# C57 Top-k kernel
/data/models/mi325_dsv4_model_tuning_round2_20260802/57_topk_ob_hybrid_bpp_20260802_141900/aiter/csrc/kernels/topk_per_row_kernels.cu

# Router
/data/models/mi325_dsv4_reuse_tuning_20260801_103501/08_dual_tp4/router.py

# Retained A8W8 tuning table
/data/models/mi325_dsv4_perf_tuning_20260803_065439/experiments/e077_bpreshuffle_18row_overlay.csv
```

The files uploaded to GitHub contain copies and checksums of these dependencies. The original E004 script intentionally retains the validated absolute paths so that the same environment can reproduce the configuration directly.

## 3. Pull Container Image

The E004 script automatically creates the required Docker containers. First, pull the validated image:

```bash
docker pull rocm/atom-dev:nightly_202607271535
```

## 4. Start the E004 Dual TP4 Service

Stop any existing containers with the same names:

```bash
bash \
  "$REPO_ROOT/docs/mi325x_dsv4_flash_tp4/commands/e004_stop_dual_tp4_131k_singlecontainer.sh"
```

Create and start the containers using the original E004 script:

```bash
bash \
  "$REPO_ROOT/docs/mi325x_dsv4_flash_tp4/commands/e004_start_dual_tp4_131k_singlecontainer.sh"
```

The script automatically creates:

* `mi325_dsv4_e004_dual`
* TP4 instance on GPUs `0,1,2,3`, listening on port `18000`
* TP4 instance on GPUs `4,5,6,7`, listening on port `18001`
* `mi325_dsv4_e004_router`
* Least-connections Router, listening on port `18080`

Check the containers:

```bash
docker ps --filter name=mi325_dsv4_e004
```

Check the logs:

```bash
docker logs -f mi325_dsv4_e004_dual
docker logs -f mi325_dsv4_e004_router
```

## 5. Verify the Service

```bash
curl -sS http://127.0.0.1:18000/v1/models
curl -sS http://127.0.0.1:18001/v1/models
curl -sS http://127.0.0.1:18080/router/status
```

Verify the service with a streaming request:

```bash
curl -N http://127.0.0.1:18080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "DeepSeek-V3.2",
    "messages": [
      {
        "role": "user",
        "content": "你好，请介绍一下你自己。"
      }
    ],
    "max_tokens": 256,
    "stream": true
  }'
```

## 6. ATOM Benchmark

Benchmark configuration:

* Input length: `6100`
* Output length: `1024`
* Concurrency: `70`

Run the validated benchmark script:

```bash
bash \
  "$REPO_ROOT/docs/mi325x_dsv4_flash_tp4/commands/e004_run_atom_6100_1024_u70.sh"
```

If the full 700-request benchmark script is available, run:

```bash
bash \
  "$REPO_ROOT/docs/mi325x_dsv4_flash_tp4/commands/e152_run_atom_6100_1024_u70_700.sh"
```

Results are saved to:

```bash
$RUN_ROOT/atom_benchmark/
```

## 7. Locust Benchmark

The original Locust benchmark uses a mixed dataset with an average input length of approximately 6000 tokens and an output length of 1024 tokens.

Run:

```bash
bash \
  "$REPO_ROOT/docs/mi325x_dsv4_flash_tp4/commands/e003_run_original_locust.sh"
```

Results are saved to:

```bash
$RUN_ROOT/locust/
```

## 8. Stop the Service

```bash
bash \
  "$REPO_ROOT/docs/mi325x_dsv4_flash_tp4/commands/e004_stop_dual_tp4_131k_singlecontainer.sh"
```

Alternatively, stop and remove the containers directly:

```bash
docker rm -f \
  mi325_dsv4_e004_dual \
  mi325_dsv4_e004_router
```

## Configuration Summary

* Dual TP4: GPUs `0-3` and `4-7`
* Router: least-connections, port `18080`
* MTP: `2`
* `max-model-len=131072`
* `max-num-batched-tokens=131072`
* `max-num-seqs=128`
* `gpu-memory-utilization=0.83`
* CUDAGraph sizes: `[1,2,4,8,16,24,32,48,64,72,96,128]`
* AITER Quick Reduce: `INT4`
* Prefix cache: disabled
