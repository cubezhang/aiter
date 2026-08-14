#!/usr/bin/env bash
set -euo pipefail

# End-to-end reproduction for the currently validated MI325X DeepSeek-V4 point:
#   2 x TP4, MTP2, FP8 KV, 32K attention-prefill chunk, exact graph buckets,
#   closed-loop concurrency 56, fixed 750-request screening manifest.
#
# This script intentionally does not run the full 3997-request x 5 acceptance
# campaign. Set MEASURE_MANIFEST and REPEATS explicitly only after agreeing on
# that contract phase.

usage() {
  cat <<'EOF'
Usage: run_120w_reproduction.sh [CAMPAIGN_TAG]

Default workflow:
  1. Validate pinned images, model assets, manifests, tools, GPUs and free ports.
  2. Capture pre-deployment platform/configuration evidence and SHA256 hashes.
  3. Deploy two TP4 ATOM backends plus least-connections Router.
  4. Run one fixed C53/750-request warmup (excluded from campaign summary).
  5. Run two fixed C56/750-request measurement rounds.
  6. Generate JSON, Markdown and TSV campaign summaries.
  7. Keep the validated service online by default.

Environment overrides:
  REPEATS=2                  Number of measured repeats.
  CONCURRENCY=56             Measured closed-loop concurrency.
  WARMUP_CONCURRENCY=53      Warmup concurrency.
  MEASURE_MANIFEST=...       Default: fixed screen_750.jsonl.
  WARMUP_MANIFEST=...        Default: fixed screen_750.jsonl.
  RUN_ROOT=...               Parent evidence directory.
  KEEP_SERVICE=1             1 keeps service online; 0 stops it after summary.
  STOP_EXISTING_TAG=tag      Explicitly stop/remove only this prior task tag.
  PULL_IMAGES=0              1 pulls the two digest-pinned images first.
  TARGET_TPM=1200000         Priority TPM gate.
  MAX_MEAN_TTFT_MS=2000      Priority Mean TTFT gate.
  SPECULATIVE_METHOD=mtp      mtp (default), dspark, or none.
  NUM_SPECULATIVE_TOKENS=2    Draft tokens; ignored when method is none.
  MODEL_ROOT=...              Default: /data/DeepSeek-V4-Flash-FP8.
  EXPECTED_MODEL_SHARDS=46    Expected model shard count identity gate.
  EXPECTED_MODEL_SHARD_BYTES=294038841472

Examples:
  ./tools/run_120w_reproduction.sh repro_20260812_01
  STOP_EXISTING_TAG=e016repro REPEATS=3 ./tools/run_120w_reproduction.sh repro_3x
  KEEP_SERVICE=0 ./tools/run_120w_reproduction.sh disposable_repro

The script refuses non-empty output directories, unknown occupied ports and
implicit container replacement. Existing evidence is never overwritten.
EOF
}

log() {
  printf '[%s] %s\n' "$(date --utc +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

if [[ ${1:-} == -h || ${1:-} == --help ]]; then
  usage
  exit 0
fi
if [[ $# -gt 1 ]]; then
  usage >&2
  exit 64
fi

workspace=/data/mi325_0811/mi325_dsv4_opt_0811
asset_root=/data/models/mi325_dsv4_extreme_tpm_20260812
manifest_root=$asset_root/02_manifests/generated
source_root=/data/models/mi325_dsv4_reuse_tuning_20260801_103501/02_source/ATOM_official
patch_root=/data/aiter/mi325_dsv4_extreme_20260812
tuning_root=/data/models/mi325_dsv4_perf_tuning_20260803_065439
topk_source=/data/models/mi325_dsv4_model_tuning_round2_20260802/57_topk_ob_hybrid_bpp_20260802_141900/aiter/csrc/kernels/topk_per_row_kernels.cu
model_root=${MODEL_ROOT:-/data/DeepSeek-V4-Flash-FP8}
expected_model_shards=${EXPECTED_MODEL_SHARDS:-46}
expected_model_shard_bytes=${EXPECTED_MODEL_SHARD_BYTES:-294038841472}
model_expert_dtype=$(jq -r '.expert_dtype // ""' "$model_root/config.json")

atom_image=rocm/atom-dev@sha256:66df2fb1c537f52d47f2d3bf973b8da15278dd2ee362b5307e3d29dfad69fa7d
locust_image=locust-awcloud@sha256:7df3afaaaf1cca96eb9898ba7f23d9c5e4f531d86e8c02ab9f2f488eb9448631
capture_sizes='[1,2,4,8,16,24,25,26,27,28,29,30,31,32,48,64,72,96,128]'
attn_prefill_chunk_size=32768
max_num_batched_tokens=131072
max_num_seqs=128
gpu_memory_utilization=0.83
speculative_method=${SPECULATIVE_METHOD:-mtp}
num_speculative_tokens=${NUM_SPECULATIVE_TOKENS:-2}

repeats=${REPEATS:-2}
concurrency=${CONCURRENCY:-56}
warmup_concurrency=${WARMUP_CONCURRENCY:-53}
measure_manifest=${MEASURE_MANIFEST:-$manifest_root/screen_750.jsonl}
warmup_manifest=${WARMUP_MANIFEST:-$manifest_root/screen_750.jsonl}
run_root=${RUN_ROOT:-$asset_root/11_c56_120w_reproduction}
keep_service=${KEEP_SERVICE:-1}
pull_images=${PULL_IMAGES:-0}
target_tpm=${TARGET_TPM:-1200000}
max_mean_ttft_ms=${MAX_MEAN_TTFT_MS:-2000}
campaign_tag=${1:-c56_120w_$(date --utc +%Y%m%dT%H%M%SZ)}
service_tag=${campaign_tag,,}
service_tag=${service_tag//-/_}
campaign_dir=$run_root/$campaign_tag
service_container=mi325_dsv4_${service_tag}_dual
router_container=mi325_dsv4_${service_tag}_router

if [[ ! $service_tag =~ ^[a-z0-9_]+$ ]]; then
  echo "campaign tag must normalize to lowercase letters, digits or underscores: $service_tag" >&2
  exit 64
fi
if [[ ! $repeats =~ ^[1-9][0-9]*$ || ! $concurrency =~ ^[1-9][0-9]*$ ]]; then
  echo "REPEATS and CONCURRENCY must be positive integers" >&2
  exit 64
fi
if [[ $speculative_method != mtp && $speculative_method != dspark && $speculative_method != none ]]; then
  echo "SPECULATIVE_METHOD must be mtp, dspark, or none" >&2
  exit 64
fi
if [[ ! $num_speculative_tokens =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_SPECULATIVE_TOKENS must be a positive integer" >&2
  exit 64
fi
if [[ $keep_service != 0 && $keep_service != 1 ]]; then
  echo "KEEP_SERVICE must be 0 or 1" >&2
  exit 64
fi

required_commands=(curl docker git jq python3 readlink rocm-smi sha256sum ss)
for command_name in "${required_commands[@]}"; do
  command -v "$command_name" >/dev/null || {
    echo "missing required command: $command_name" >&2
    exit 69
  }
done

required_files=(
  "$workspace/tools/start_candidate_service.sh"
  "$workspace/tools/run_strict_point.sh"
  "$workspace/tools/summarize_strict_run.py"
  "$workspace/tools/summarize_120w_campaign.py"
  "$measure_manifest"
  "$warmup_manifest"
  "$patch_root/patches/scheduler.py"
  "$patch_root/router.py"
  "$topk_source"
  "$tuning_root/experiments/e077_bpreshuffle_18row_overlay.csv"
  "$model_root/config.json"
)
for required_file in "${required_files[@]}"; do
  [[ -f $required_file ]] || {
    echo "missing required file: $required_file" >&2
    exit 66
  }
done
[[ -d $source_root ]] || { echo "missing source directory: $source_root" >&2; exit 66; }
mapfile -t model_shard_rows < <(
  find "$model_root" -maxdepth 1 -type f -name 'model-*.safetensors' \
    -printf '%f\t%s\n' | sort
)
model_shard_bytes=0
for shard_row in "${model_shard_rows[@]}"; do
  model_shard_bytes=$((model_shard_bytes + ${shard_row#*$'\t'}))
done
if [[ ${#model_shard_rows[@]} -ne $expected_model_shards || $model_shard_bytes -ne $expected_model_shard_bytes ]]; then
  echo "model shard identity mismatch: count=${#model_shard_rows[@]}/$expected_model_shards bytes=$model_shard_bytes/$expected_model_shard_bytes" >&2
  exit 66
fi

mkdir -p "$run_root"
if [[ -e $campaign_dir ]]; then
  echo "refusing existing campaign directory: $campaign_dir" >&2
  exit 65
fi

if [[ -n ${STOP_EXISTING_TAG:-} ]]; then
  old_tag=${STOP_EXISTING_TAG,,}
  old_tag=${old_tag//-/_}
  [[ $old_tag =~ ^[a-z0-9_]+$ ]] || { echo "invalid STOP_EXISTING_TAG" >&2; exit 64; }
  old_service=mi325_dsv4_${old_tag}_dual
  old_router=mi325_dsv4_${old_tag}_router
  for old_container in "$old_router" "$old_service"; do
    if docker container inspect "$old_container" >/dev/null 2>&1; then
      docker stop "$old_container"
      docker rm "$old_container"
    fi
  done
fi

for port in 18000 18001 18080; do
  if ss -ltnH "sport = :$port" | read -r _; then
    echo "port $port is already listening; set exact STOP_EXISTING_TAG or stop it manually" >&2
    exit 67
  fi
done

mkdir "$campaign_dir"
printf '%s\n' "${model_shard_rows[@]}" > "$campaign_dir/model_shards.tsv"

log "validating digest-pinned deployment images"
if [[ $pull_images == 1 ]]; then
  docker pull "$atom_image"
  docker pull "$locust_image"
fi
docker image inspect "$atom_image" > "$campaign_dir/atom_image_inspect.json"
docker image inspect "$locust_image" > "$campaign_dir/locust_image_inspect.json"

log "capturing pre-deployment GPU, RAS, topology and software evidence"
for card in $(seq 0 7); do
  rocm-smi -d "$card" --showrasinfo
done > "$campaign_dir/ras_predeploy.txt" 2>&1
rocm-smi --showproductname --showuniqueid --showuse --showmemuse --showpower --showtemp \
  > "$campaign_dir/gpu_predeploy.txt" 2>&1
rocm-smi --showtopo > "$campaign_dir/gpu_topology.txt" 2>&1 || true
uname -a > "$campaign_dir/uname.txt"
docker version > "$campaign_dir/docker_version.txt"
git -C "$source_root" rev-parse HEAD > "$campaign_dir/atom_source_commit.txt"
git -C "$source_root" status --short > "$campaign_dir/atom_source_status.txt"
git -C "$source_root" diff --binary > "$campaign_dir/atom_source_worktree.patch"

sha256sum \
  "$workspace/tools/start_candidate_service.sh" \
  "$workspace/tools/run_strict_point.sh" \
  "$workspace/tools/locust_strict_contract.py" \
  "$workspace/tools/summarize_strict_run.py" \
  "$workspace/tools/summarize_120w_campaign.py" \
  "$measure_manifest" "$warmup_manifest" \
  "$patch_root/patches/scheduler.py" "$patch_root/router.py" \
  "$source_root/atom/model_engine/model_runner.py" \
  "$source_root/atom/entrypoints/openai/api_server.py" \
  "$source_root/atom/entrypoints/openai/chat_encoders.py" \
  "$source_root/atom/model_ops/fused_moe_triton.py" \
  "$topk_source" "$tuning_root/experiments/e077_bpreshuffle_18row_overlay.csv" \
  "$model_root/config.json" \
  > "$campaign_dir/input_sha256.txt"

python3 - "$campaign_dir/campaign_config.json" <<PY
import json
import pathlib

value = {
    "schema_version": 1,
    "campaign_tag": "$campaign_tag",
    "service_tag": "$service_tag",
    "topology": "2xTP4",
    "atom_image": "$atom_image",
    "locust_image": "$locust_image",
    "model": "$model_root",
    "model_shard_count": $expected_model_shards,
    "model_shard_bytes": $expected_model_shard_bytes,
    "model_expert_dtype": "$model_expert_dtype",
    "triton_moe_required": $( [[ $model_expert_dtype == fp4 ]] && echo True || echo False ),
    "kv_cache_dtype": "fp8",
    "requested_kv_cache_dtype": "fp8",
    "effective_kv_cache_dtype": "runtime-detected",
    "quick_reduce_quantization": "INT4",
    "speculative_method": "$speculative_method",
    "num_speculative_tokens": $( [[ $speculative_method == none ]] && echo 0 || echo "$num_speculative_tokens" ),
    "attn_prefill_chunk_size": $attn_prefill_chunk_size,
    "cudagraph_capture_sizes": $capture_sizes,
    "max_num_batched_tokens": $max_num_batched_tokens,
    "max_num_seqs": $max_num_seqs,
    "gpu_memory_utilization": $gpu_memory_utilization,
    "prefix_cache": False,
    "router_policy": "least_connections",
    "warmup_manifest": "$warmup_manifest",
    "warmup_concurrency": $warmup_concurrency,
    "measure_manifest": "$measure_manifest",
    "measure_concurrency": $concurrency,
    "repeats": $repeats,
    "target_tpm": float("$target_tpm"),
    "max_mean_ttft_ms": float("$max_mean_ttft_ms"),
}
pathlib.Path("$campaign_dir/campaign_config.json").write_text(
    json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8"
)
PY

service_started=0
stop_service() {
  if [[ $service_started == 1 ]]; then
    docker stop "$router_container" "$service_container" >/dev/null 2>&1 || true
    docker rm "$router_container" "$service_container" >/dev/null 2>&1 || true
  fi
}
on_exit() {
  rc=$?
  if [[ $rc -ne 0 || $keep_service == 0 ]]; then
    stop_service
  fi
  exit "$rc"
}
trap on_exit EXIT

export CANDIDATE_TOPOLOGY=dual_tp4
export CANDIDATE_CAPTURE_SIZES=$capture_sizes
export CANDIDATE_ATTN_PREFILL_CHUNK_SIZE=$attn_prefill_chunk_size
export CANDIDATE_MAX_NUM_BATCHED_TOKENS=$max_num_batched_tokens
export CANDIDATE_MAX_NUM_SEQS=$max_num_seqs
export CANDIDATE_LONG_PREFILL_TOKEN_THRESHOLD=0
export CANDIDATE_SCHEDULER_DELAY_FACTOR=0.0
export CANDIDATE_GPU_MEMORY_UTILIZATION=$gpu_memory_utilization
export CANDIDATE_SPECULATIVE_METHOD=$speculative_method
export CANDIDATE_NUM_SPECULATIVE_TOKENS=$num_speculative_tokens
export CANDIDATE_MODEL_ROOT=$model_root

log "deploying $service_container and $router_container from pinned ATOM image"
service_started=1
"$workspace/tools/start_candidate_service.sh" "$service_tag" \
  > "$campaign_dir/service_start.log" 2>&1
docker inspect "$service_container" "$router_container" \
  > "$campaign_dir/service_inspect_deployed.json"
curl -fsS http://127.0.0.1:18080/router/status \
  > "$campaign_dir/router_deployed.json"

runtime_kv_cache_dtype=fp8
if grep -Fq 'Falling back to a bf16 KV cache.' \
  "$patch_root/logs/${service_tag}_18000_server.log"; then
  runtime_kv_cache_dtype=bf16
fi
python3 - "$campaign_dir/campaign_config.json" "$runtime_kv_cache_dtype" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
value = json.loads(path.read_text(encoding="utf-8"))
value["effective_kv_cache_dtype"] = sys.argv[2]
path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
PY

export STRICT_SERVICE_TAG=$service_tag
warmup_label=${campaign_tag}_WARMUP_C${warmup_concurrency}
warmup_dir=$campaign_dir/$warmup_label
log "running excluded warmup: label=$warmup_label concurrency=$warmup_concurrency"
"$workspace/tools/run_strict_point.sh" \
  "$warmup_manifest" "$warmup_concurrency" "$warmup_label" "$warmup_dir" \
  > "$campaign_dir/${warmup_label}.console.log" 2>&1

run_dirs=()
for repetition in $(seq 1 "$repeats"); do
  label=${campaign_tag}_C${concurrency}_R${repetition}
  run_dir=$campaign_dir/$label
  log "running measured round $repetition/$repeats: label=$label concurrency=$concurrency"
  "$workspace/tools/run_strict_point.sh" \
    "$measure_manifest" "$concurrency" "$label" "$run_dir" \
    > "$campaign_dir/${label}.console.log" 2>&1
  run_dirs+=("$run_dir")
done

log "aggregating priority-gate results and evidence hashes"
summary_args=()
for run_dir in "${run_dirs[@]}"; do
  summary_args+=(--run-dir "$run_dir")
done
python3 "$workspace/tools/summarize_120w_campaign.py" \
  "${summary_args[@]}" \
  --output-dir "$campaign_dir" \
  --target-tpm "$target_tpm" \
  --max-mean-ttft-ms "$max_mean_ttft_ms" \
  > "$campaign_dir/campaign_summary.console.json"

log "capturing post-campaign GPU, RAS, container and router evidence"
for card in $(seq 0 7); do
  rocm-smi -d "$card" --showrasinfo
done > "$campaign_dir/ras_postcampaign.txt" 2>&1
rocm-smi --showuse --showmemuse --showpower --showtemp \
  > "$campaign_dir/gpu_postcampaign.txt" 2>&1
docker inspect "$service_container" "$router_container" \
  > "$campaign_dir/service_inspect_postcampaign.json"
curl -fsS http://127.0.0.1:18080/router/status \
  > "$campaign_dir/router_postcampaign.json"
sha256sum "$campaign_dir"/campaign_summary.json "$campaign_dir"/campaign_summary.md \
  "$campaign_dir"/campaign_summary.tsv > "$campaign_dir/output_sha256.txt"

campaign_pass=$(jq -r '.aggregate.all_rounds_pass' "$campaign_dir/campaign_summary.json")
log "campaign complete: $campaign_dir"
log "summary: $campaign_dir/campaign_summary.md"
if [[ $keep_service == 1 ]]; then
  log "service kept online: $service_container and $router_container"
fi
if [[ $campaign_pass != true ]]; then
  echo "one or more measured rounds failed the configured priority/quality gates" >&2
  exit 2
fi
