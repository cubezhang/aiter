#!/usr/bin/env bash
set -euo pipefail

# Start a tagged serving candidate with controlled scheduler/topology knobs.
# The caller must stop the currently active stack first. Container names and
# logs are tagged so every candidate remains independently inspectable.

service_tag=${1:?usage: start_candidate_service.sh SERVICE_TAG}
topology=${CANDIDATE_TOPOLOGY:-dual_tp4}
capture_sizes=${CANDIDATE_CAPTURE_SIZES:-[1,2,4,8,16,24,25,26,27,28,29,30,31,32,48,64,72,96,128]}
attn_prefill_chunk_size=${CANDIDATE_ATTN_PREFILL_CHUNK_SIZE:-16384}
max_num_batched_tokens=${CANDIDATE_MAX_NUM_BATCHED_TOKENS:-131072}
max_num_seqs=${CANDIDATE_MAX_NUM_SEQS:-128}
long_prefill_token_threshold=${CANDIDATE_LONG_PREFILL_TOKEN_THRESHOLD:-0}
scheduler_delay_factor=${CANDIDATE_SCHEDULER_DELAY_FACTOR:-0.0}
gpu_memory_utilization=${CANDIDATE_GPU_MEMORY_UTILIZATION:-0.83}
speculative_method=${CANDIDATE_SPECULATIVE_METHOD:-mtp}
num_speculative_tokens=${CANDIDATE_NUM_SPECULATIVE_TOKENS:-2}
model_root=${CANDIDATE_MODEL_ROOT:-/data/DeepSeek-V4-Flash-FP8}
source_root=${CANDIDATE_SOURCE_ROOT:-/data/models/mi325_dsv4_reuse_tuning_20260801_103501/02_source/ATOM_official}
scheduler_patch=${CANDIDATE_SCHEDULER_PATCH-/data/aiter/mi325_dsv4_extreme_20260812/patches/scheduler.py}
extra_atom_args=${CANDIDATE_EXTRA_ATOM_ARGS:-}
eplb_enable=${CANDIDATE_EPLB_ENABLE:-0}
eplb_config=${CANDIDATE_EPLB_CONFIG:-'{"num_redundant_experts":0,"placement_policy":"naive","rebalance_interval":200,"load_window_size":200}'}
mori_shmem_heap_size=${CANDIDATE_MORI_SHMEM_HEAP_SIZE:-}
ar_use_new=${CANDIDATE_AR_USE_NEW:-}
ar_m_min=${CANDIDATE_AR_M_MIN:-100}
ar_m_max=${CANDIDATE_AR_M_MAX:-220}
ar_comm_patch=${CANDIDATE_AR_COMM_PATCH:-/data/models/mi325_dsv4_perf_tuning_20260803_065439/patches/e142_communication_op.py}
ar_parallel_state_patch=${CANDIDATE_AR_PARALLEL_STATE_PATCH:-/data/models/mi325_dsv4_perf_tuning_20260803_065439/patches/e142_parallel_state.py}

if [[ $model_root != /* || ! -f $model_root/config.json ]]; then
  echo "CANDIDATE_MODEL_ROOT must be an absolute model directory with config.json: $model_root" >&2
  exit 66
fi
model_expert_dtype=$(jq -r '.expert_dtype // ""' "$model_root/config.json")
enable_triton_moe=${CANDIDATE_TRITON_MOE:-}
if [[ -z $enable_triton_moe && $model_expert_dtype == fp4 ]]; then
  # ATOM's gfx94x MXFP4 path selects Triton internally, but its swizzle module
  # imports are gated by this explicit environment flag.
  enable_triton_moe=1
fi
if [[ -n $enable_triton_moe && $enable_triton_moe != 0 && $enable_triton_moe != 1 ]]; then
  echo "CANDIDATE_TRITON_MOE must be 0, 1, or unset" >&2
  exit 64
fi

case $topology in
  dual_tp4)
    tensor_parallel_size=4
    data_parallel_size=1
    enable_expert_parallel=0
    enable_dp_attention=0
    gpu_groups='0,1,2,3;4,5,6,7'
    backend_ports='18000;18001'
    router_backends='http://127.0.0.1:18000,http://127.0.0.1:18001'
    ;;
  dual_dp4_ep)
    # Four data-parallel attention ranks share an EP4 MoE shard group.  This
    # keeps the same two independent four-GPU instances as dual_tp4 while
    # exposing ATOM's DPA/EP path as an isolated topology candidate.
    tensor_parallel_size=1
    data_parallel_size=4
    enable_expert_parallel=1
    enable_dp_attention=1
    gpu_groups='0,1,2,3;4,5,6,7'
    backend_ports='18000;18001'
    router_backends='http://127.0.0.1:18000,http://127.0.0.1:18001'
    ;;
  single_dp8_ep)
    # Official ATOM-style node-wide data/expert parallelism: attention is
    # replicated per GPU while all eight ranks shard the MoE experts.
    tensor_parallel_size=1
    data_parallel_size=8
    enable_expert_parallel=1
    enable_dp_attention=1
    gpu_groups='0,1,2,3,4,5,6,7'
    backend_ports='18000'
    router_backends='http://127.0.0.1:18000'
    ;;
  quad_tp2)
    tensor_parallel_size=2
    data_parallel_size=1
    enable_expert_parallel=0
    enable_dp_attention=0
    gpu_groups='0,1;2,3;4,5;6,7'
    backend_ports='18000;18001;18002;18003'
    router_backends='http://127.0.0.1:18000,http://127.0.0.1:18001,http://127.0.0.1:18002,http://127.0.0.1:18003'
    ;;
  *)
    echo "unsupported CANDIDATE_TOPOLOGY: $topology" >&2
    exit 64
    ;;
esac

if [[ $enable_expert_parallel == 1 && -z $mori_shmem_heap_size ]]; then
  # A 32K dispatch budget per DP rank exceeds MoRI's 4 GiB static default.
  # ROCm's documented production setting for that budget is a 16 GiB heap.
  mori_shmem_heap_size=16G
fi

if [[ $eplb_enable != 0 && $eplb_enable != 1 ]]; then
  echo "CANDIDATE_EPLB_ENABLE must be 0 or 1" >&2
  exit 64
fi
if [[ $eplb_enable == 1 && $enable_expert_parallel != 1 ]]; then
  echo "CANDIDATE_EPLB_ENABLE=1 requires an expert-parallel topology" >&2
  exit 64
fi
jq -e 'type == "object"' <<<"$eplb_config" >/dev/null || {
  echo "CANDIDATE_EPLB_CONFIG must be a JSON object" >&2
  exit 64
}

atom_image=rocm/atom-dev@sha256:66df2fb1c537f52d47f2d3bf973b8da15278dd2ee362b5307e3d29dfad69fa7d
dual_name=mi325_dsv4_${service_tag}_dual
router_name=mi325_dsv4_${service_tag}_router
patch_root=/data/aiter/mi325_dsv4_extreme_20260812
tuning_root=/data/models/mi325_dsv4_perf_tuning_20260803_065439
topk_source=/data/models/mi325_dsv4_model_tuning_round2_20260802/57_topk_ob_hybrid_bpp_20260802_141900/aiter/csrc/kernels/topk_per_row_kernels.cu

if [[ $source_root != /* || ! -f $source_root/atom/model_engine/arg_utils.py ]]; then
  echo "CANDIDATE_SOURCE_ROOT must be an absolute ATOM source directory: $source_root" >&2
  exit 66
fi
source_mounts=(--mount "type=bind,src=$source_root,dst=/app/ATOM,readonly")
if [[ -n $scheduler_patch ]]; then
  if [[ $scheduler_patch != /* || ! -f $scheduler_patch ]]; then
    echo "CANDIDATE_SCHEDULER_PATCH must be empty or an absolute file" >&2
    exit 66
  fi
  source_mounts+=(--mount "type=bind,src=$scheduler_patch,dst=/app/ATOM/atom/model_engine/scheduler.py,readonly")
fi

ar_mounts=()
if [[ -n $ar_use_new ]]; then
  if [[ $ar_use_new != 0 && $ar_use_new != 1 ]]; then
    echo "CANDIDATE_AR_USE_NEW must be 0, 1, or unset" >&2
    exit 64
  fi
  if [[ ! -f $ar_comm_patch || ! -f $ar_parallel_state_patch ]]; then
    echo "AllReduce candidate patches are missing" >&2
    exit 66
  fi
  ar_mounts+=(
    --mount "type=bind,src=$ar_comm_patch,dst=/app/aiter-test/aiter/dist/communication_op.py,readonly"
    --mount "type=bind,src=$ar_parallel_state_patch,dst=/app/aiter-test/aiter/dist/parallel_state.py,readonly"
  )
fi

for name in "$dual_name" "$router_name"; do
  if docker container inspect "$name" >/dev/null 2>&1; then
    echo "refusing existing candidate container: $name" >&2
    exit 65
  fi
done

docker run --detach --name "$dual_name" \
  --network host --ipc host --shm-size 128g --security-opt label=disable \
  --device /dev/kfd --device /dev/dri \
  --env AITER_QUICK_REDUCE_QUANTIZATION=INT4 \
  --env "SERVICE_TAG=$service_tag" \
  --env "CAPTURE_SIZES=$capture_sizes" \
  --env "ATTN_PREFILL_CHUNK_SIZE=$attn_prefill_chunk_size" \
  --env "MAX_NUM_BATCHED_TOKENS=$max_num_batched_tokens" \
  --env "MAX_NUM_SEQS=$max_num_seqs" \
  --env "LONG_PREFILL_TOKEN_THRESHOLD=$long_prefill_token_threshold" \
  --env "SCHEDULER_DELAY_FACTOR=$scheduler_delay_factor" \
  --env "GPU_MEMORY_UTILIZATION=$gpu_memory_utilization" \
  --env "SPECULATIVE_METHOD=$speculative_method" \
  --env "NUM_SPECULATIVE_TOKENS=$num_speculative_tokens" \
  --env "MODEL_ROOT=$model_root" \
  --env "EXTRA_ATOM_ARGS=$extra_atom_args" \
  --env "ENABLE_TRITON_MOE=$enable_triton_moe" \
  --env "TENSOR_PARALLEL_SIZE=$tensor_parallel_size" \
  --env "DATA_PARALLEL_SIZE=$data_parallel_size" \
  --env "ENABLE_EXPERT_PARALLEL=$enable_expert_parallel" \
  --env "ENABLE_DP_ATTENTION=$enable_dp_attention" \
  --env "EPLB_ENABLE=$eplb_enable" \
  --env "EPLB_CONFIG=$eplb_config" \
  --env "MORI_SHMEM_HEAP_SIZE=$mori_shmem_heap_size" \
  --env "AITER_TP_ALLREDUCE_USE_NEW=$ar_use_new" \
  --env "AITER_TP_ALLREDUCE_M_MIN=$ar_m_min" \
  --env "AITER_TP_ALLREDUCE_M_MAX=$ar_m_max" \
  --env "GPU_GROUPS=$gpu_groups" \
  --env "BACKEND_PORTS=$backend_ports" \
  "${source_mounts[@]}" \
  "${ar_mounts[@]}" \
  --mount "type=bind,src=$topk_source,dst=/app/aiter-test/csrc/kernels/topk_per_row_kernels.cu,readonly" \
  --mount "type=bind,src=$tuning_root,dst=/work" \
  --mount "type=bind,src=$patch_root/logs,dst=/service_logs" \
  --mount "type=bind,src=$model_root,dst=$model_root,readonly" \
  "$atom_image" bash -lc '
set -euo pipefail
unset AITER_REBUILD AITER_CONFIG_GEMM_BF16 AITER_CONFIG_FMOE AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE ATOM_USE_TRITON_MOE
if [[ -n $ENABLE_TRITON_MOE ]]; then
  export ATOM_USE_TRITON_MOE="$ENABLE_TRITON_MOE"
fi
export AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE=/work/experiments/e077_bpreshuffle_18row_overlay.csv
rm -f /app/aiter-test/aiter/jit/module_top_k_per_row.so
cd /app/ATOM
start_backend() {
  local devices=$1
  local port=$2
  # Each independent DP backend needs its own rendezvous port.  ATOM defaults
  # every server process to 29500, which collides when two DP groups share the
  # host network namespace.
  local dp_master_port=$((29500 + (port - 18000) * 100))
  local spec_args=()
  local parallel_args=(
    --data-parallel-size "$DATA_PARALLEL_SIZE"
    --data-parallel-master-port "$dp_master_port"
  )
  local eplb_args=()
  local extra_args=()
  if [[ -n $EXTRA_ATOM_ARGS ]]; then
    read -r -a extra_args <<< "$EXTRA_ATOM_ARGS"
  fi
  if [[ $SPECULATIVE_METHOD != none ]]; then
    spec_args=(--method "$SPECULATIVE_METHOD" --num-speculative-tokens "$NUM_SPECULATIVE_TOKENS")
  fi
  if [[ $ENABLE_EXPERT_PARALLEL == 1 ]]; then
    parallel_args+=(--enable-expert-parallel)
  fi
  if [[ $ENABLE_DP_ATTENTION == 1 ]]; then
    parallel_args+=(--enable-dp-attention)
  fi
  if [[ $EPLB_ENABLE == 1 ]]; then
    eplb_args=(--eplb-enable --eplb-config "$EPLB_CONFIG")
  fi
  ATOM_DP_MASTER_PORT="$dp_master_port" HIP_VISIBLE_DEVICES="$devices" \
    python -m atom.entrypoints.openai_server \
    --model "$MODEL_ROOT" --served-model-name DeepSeek-V4 \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" "${parallel_args[@]}" \
    "${eplb_args[@]}" "${spec_args[@]}" \
    --kv-cache-dtype fp8 --max-model-len 131072 \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --attn-prefill-chunk-size "$ATTN_PREFILL_CHUNK_SIZE" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --long-prefill-token-threshold "$LONG_PREFILL_TOKEN_THRESHOLD" \
    --scheduler-delay-factor "$SCHEDULER_DELAY_FACTOR" \
    --cudagraph-capture-sizes "$CAPTURE_SIZES" \
    --no-enable-prefix-caching "${extra_args[@]}" \
    --host 127.0.0.1 --server-port "$port" \
    > "/service_logs/${SERVICE_TAG}_${port}_server.log" 2>&1 &
  backend_pid=$!
}
IFS=";" read -r -a groups <<< "$GPU_GROUPS"
IFS=";" read -r -a ports <<< "$BACKEND_PORTS"
[[ ${#groups[@]} -eq ${#ports[@]} ]] || exit 30
pids=()
for index in "${!groups[@]}"; do
  start_backend "${groups[$index]}" "${ports[$index]}"
  pids+=("$backend_pid")
  for _ in $(seq 1 180); do
    curl -fsS --max-time 3 "http://127.0.0.1:${ports[$index]}/v1/models" >/dev/null && break
    kill -0 "$backend_pid" 2>/dev/null || exit 31
    sleep 3
  done
  curl -fsS --max-time 3 "http://127.0.0.1:${ports[$index]}/v1/models" >/dev/null || exit 32
done
trap "kill ${pids[*]} 2>/dev/null || true; wait ${pids[*]} 2>/dev/null || true" TERM INT EXIT
wait -n "${pids[@]}"
'

for _ in $(seq 1 240); do
  all_healthy=true
  IFS=';' read -r -a ports <<< "$backend_ports"
  for port in "${ports[@]}"; do
    curl -fsS --max-time 3 "http://127.0.0.1:$port/v1/models" >/dev/null || all_healthy=false
  done
  if $all_healthy; then
    break
  fi
  docker inspect "$dual_name" --format '{{.State.Status}} {{.State.ExitCode}}' | grep -q '^running ' || exit 41
  sleep 3
done
for port in "${ports[@]}"; do
  curl -fsS --max-time 3 "http://127.0.0.1:$port/v1/models" >/dev/null
done

docker run --detach --name "$router_name" \
  --network host --ipc host --security-opt label=disable \
  --env "ROUTER_BACKENDS=$router_backends" \
  --env ROUTER_POLICY=least_connections \
  --mount "type=bind,src=$patch_root/router.py,dst=/router.py,readonly" \
  "$atom_image" bash -lc 'exec python -m uvicorn router:app --app-dir / --host 127.0.0.1 --port 18080 --no-access-log'

for _ in $(seq 1 60); do
  if curl -fsS --max-time 3 http://127.0.0.1:18080/router/status >/dev/null; then
    docker inspect "$dual_name" "$router_name"
    exit 0
  fi
  sleep 1
done
exit 51
