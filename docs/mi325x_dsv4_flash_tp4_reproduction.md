# MI325X DeepSeek-V4-Flash-FP8 TP4 reproduction

This recipe reproduces the MI325X TP4 serving result obtained with
DeepSeek-V4-Flash-FP8.

## Reference result

- Total token throughput: 18075.76 tokens/s
- Output token throughput: 2583.95 tokens/s
- Mean TPOT: 25.36 ms
- Mean TTFT: 1177.41 ms
- Successful requests: 700/700
- GPU utilization: 100%

Workload:

- Input length: 6144
- Output length: 1024
- Concurrency: 70
- Requests: 700
- Ignore EOS: enabled

## Reproduction fingerprint

- GPU: 4 x AMD Instinct MI325X, gfx942, 304 CUs
- Host kernel: 6.8.0-136-generic
- Kernel command line must contain `iommu=pt`
- `kernel.numa_balancing` must be `0`
- Container image:
  `rocm/atom-dev@sha256:3beb8c2db7beac1a8865d7b96e79c7e89c8769995757d02570042619f712fac5`
- ATOM commit:
  `b31cc35409c8c8a00869adfe07253396b8000578`
- AITER optimization baseline:
  `dd616c67f7beeaade01461114a61621c82ba8af5`
- M210 table SHA256:
  `9c62a50656c5aec4fccc83bf7dce4e9d0d4878db944a12236d0d7dbb1cad55bb`

## Host preparation

Verify:

```bash
grep -o 'iommu=pt' /proc/cmdline
sysctl kernel.numa_balancing
```

The model must be available on the host as:

```text
/data/DeepSeek-V4-Flash-FP8
```

## Create the container

```bash
docker run -dit \
  --name atom_dsv4_repro \
  --ipc=host \
  --network=host \
  --privileged \
  --device=/dev/kfd \
  --device=/dev/dri \
  --security-opt seccomp=unconfined \
  --security-opt apparmor=unconfined \
  --group-add video \
  --shm-size=128G \
  -e NCCL_IB_GID_INDEX=3 \
  -w /workspace \
  -v /data:/data/models \
  rocm/atom-dev@sha256:3beb8c2db7beac1a8865d7b96e79c7e89c8769995757d02570042619f712fac5 \
  /bin/bash
```

## Install the tuned AITER branch

```bash
docker exec -it atom_dsv4_repro bash
```
Then run inside the container:
```bash
git clone \
  --branch mi325-dsv4-ops \
  --single-branch \
  https://github.com/cubezhang/aiter.git \
  /app/aiter-tuning-dsv4
```

## Start the server

Enter the container and run:

```bash
cd /app/ATOM

unset AITER_CONFIG_GEMM_BF16
unset AITER_CONFIG_FMOE
unset AITER_USE_CK_MOE_SORTING
unset AITER_USE_FLYDSL_MOE_SORTING
unset AITER_REBUILD

export PYTHONPATH=/app/aiter-tuning-dsv4
export AITER_JIT_DIR=/app/aiter-tuning-dsv4/aiter/jit
export AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE=/app/aiter-tuning-dsv4/aiter/configs/model_configs/dsv4_flash_mi325x_tp4_m210_a8w8_blockscale_bpreshuffle_tuned_gemm.csv
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export AITER_BF16_FP8_MOE_BOUND=256
export AITER_LOG_LEVEL=WARNING

export ATOM_MOE_GU_ITLV=0
export ATOM_PCP_MOE_MERGE=1
export ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=1024
export ATOM_NUMA_BIND=1
export ATOM_NUMA_NODE=0,0,0,0
export HIP_VISIBLE_DEVICES=0,1,2,3

python -m atom.entrypoints.openai_server \
  --model /data/models/DeepSeek-V4-Flash-FP8 \
  --kv-cache-dtype fp8 \
  --index-cache-dtype fp8 \
  -tp 4 \
  -pcp 1 \
  -dp 1 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192 \
  --max-num-batched-tokens 131072 \
  --max-num-seqs 512 \
  --scheduler-delay-factor 0.0 \
  --method mtp \
  --num-speculative-tokens 2 \
  --cudagraph-capture-sizes '[1,2,4,8,16,32,48,64,70,128,256,512]'
```

## Benchmark

```bash
python -m atom.benchmarks.benchmark_serving \
  --model=/data/models/DeepSeek-V4-Flash-FP8 \
  --backend=vllm \
  --base-url=http://127.0.0.1:8000 \
  --dataset-name=random \
  --random-input-len=6144 \
  --random-output-len=1024 \
  --random-range-ratio=1.0 \
  --num-prompts=700 \
  --num-warmups=140 \
  --max-concurrency=70 \
  --request-rate=inf \
  --ignore-eos \
  --save-result \
  --percentile-metrics=ttft,tpot,itl,e2el \
  --metric-percentiles=50,90,95,99
```
