#!/usr/bin/env bash
set -euo pipefail

# Start the frozen DeepSeek-V4-Flash 2 x TP4 service on this host.
# No arguments are accepted. The script pauses before every stage.

asset_root=/data/models/mi325_dsv4_extreme_tpm_20260812
reference_evidence=$asset_root/14_c56_full_single_acceptance/c56_full_single_20260813
split_root=$asset_root/15_c56_split_reproduction
workspace=/data/mi325_0811/mi325_dsv4_opt_0811
source_root=/data/models/mi325_dsv4_reuse_tuning_20260801_103501/02_source/ATOM_official
model_root=/data/DeepSeek-V4-Flash-FP8
start_runner=$workspace/tools/start_candidate_service.sh
state_file=$split_root/active_service.json

atom_image=rocm/atom-dev@sha256:66df2fb1c537f52d47f2d3bf973b8da15278dd2ee362b5307e3d29dfad69fa7d
atom_image_id=sha256:66df2fb1c537f52d47f2d3bf973b8da15278dd2ee362b5307e3d29dfad69fa7d
expected_atom_commit=5e491a90bd7e860518e0026d38c0ec1101dfb4a8
expected_atom_patch_sha256=1df4ac16f77443bc4284c993120b16607e0d3fb74b80c5cfffa29e805ff53956
expected_start_runner_sha256=c8765f7386785e542391b42d88e4c2ce2f75f68b563d88666b93e8b5238a9837
expected_model_shards=46
expected_model_shard_bytes=294038841472
known_report_service_tag=c56_full_single_20260813

service_tag=c56_split_service_$(date --utc +%Y%m%d_%H%M%S)
service_container=mi325_dsv4_${service_tag}_dual
router_container=mi325_dsv4_${service_tag}_router
service_evidence=$split_root/services/$service_tag
server_log_root=/data/aiter/mi325_dsv4_extreme_20260812/logs
step_number=0
new_service_started=0
old_service_stopped=0
old_stopped_tag=

log() {
  printf '[%s] %s\n' "$(date --utc +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

step() {
  step_number=$((step_number + 1))
  printf '\n========== START STEP %d: %s ==========\n' "$step_number" "$*"
  read -r -p '按 Enter 继续，按 Ctrl-C 停止：' _
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

require_file() {
  [[ -f $1 ]] || die "missing required file: $1"
}

cleanup_failed_start() {
  rc=$?
  if [[ $rc -ne 0 && $new_service_started == 1 ]]; then
    log '启动失败，清理本脚本创建的精确容器名称'
    docker stop "$router_container" "$service_container" >/dev/null 2>&1 || true
    docker rm "$router_container" "$service_container" >/dev/null 2>&1 || true
  fi
  if [[ $rc -ne 0 && $old_service_stopped == 1 && -f $state_file ]]; then
    recorded_tag=$(jq -r '.service_tag // empty' "$state_file" 2>/dev/null || true)
    if [[ $recorded_tag == "$old_stopped_tag" ]]; then
      mkdir -p "$service_evidence"
      mv "$state_file" "$service_evidence/invalidated_previous_active_service.json"
      log '旧服务已停止；旧 active_service.json 已保留并失效化'
    fi
  fi
  exit "$rc"
}
trap cleanup_failed_start EXIT

[[ $# -eq 0 ]] || die '该脚本不接受任何参数；请直接运行 01_start_service.sh'

step '检查本机命令、固定文件和 Docker 权限'
for command_name in awk bash cp curl docker find git jq lspci mkdir mv readlink rg rocm-smi seq sha256sum ss wc; do
  require_command "$command_name"
done
docker info >/dev/null 2>&1 || die 'Docker daemon 不可用，或当前用户无权访问 Docker socket'

require_file "$start_runner"
require_file "$reference_evidence/input_sha256.txt"
require_file "$reference_evidence/atom_source_worktree.patch"
require_file "$model_root/config.json"
[[ -d $source_root ]] || die "missing ATOM source directory: $source_root"
[[ ! -e $service_evidence ]] || die "refusing existing service evidence directory: $service_evidence"

step '只读确认 8 x MI325X、1002:74a5、gfx942 和 perf_determinism'
pci_count=$(lspci -nn | awk '/1002:74a5/{count++} END{print count+0}')
[[ $pci_count -eq 8 ]] || die "expected 8 x 1002:74a5, found $pci_count"

product_info=$(rocm-smi --showproductname 2>&1)
printf '%s\n' "$product_info"
market_count=$(awk '/Card Series:.*MI325X/{count++} END{print count+0}' <<<"$product_info")
gfx_count=$(awk '/GFX Version:.*gfx942/{count++} END{print count+0}' <<<"$product_info")
[[ $market_count -eq 8 ]] || die "expected 8 MI325X market names, found $market_count"
[[ $gfx_count -eq 8 ]] || die "expected 8 gfx942 devices, found $gfx_count"

perf_info=$(rocm-smi --showperflevel --showclocks 2>&1)
printf '%s\n' "$perf_info"
perf_count=$(awk '/Performance Level: perf_determinism/{count++} END{print count+0}' <<<"$perf_info")
[[ $perf_count -eq 8 ]] || die "expected perf_determinism on all 8 GPUs, found $perf_count; no state was changed"
rocm-smi --showuse --showmemuse --showpower --showtemp
rocm-smi --showtopo || true

step '验证固定 ATOM 镜像、模型、源码和部署 Runner'
actual_atom_image_id=$(docker image inspect "$atom_image" --format '{{.Id}}' 2>/dev/null) ||
  die "missing pinned ATOM image: $atom_image"
[[ $actual_atom_image_id == "$atom_image_id" ]] || die "ATOM image ID mismatch: $actual_atom_image_id"
docker image inspect "$atom_image" --format '{{.Id}} {{json .RepoDigests}}'

model_shard_count=$(find "$model_root" -maxdepth 1 -type f -name 'model-*.safetensors' | wc -l)
model_shard_bytes=$(find "$model_root" -maxdepth 1 -type f -name 'model-*.safetensors' \
  -printf '%s\n' | awk '{total += $1} END{print total+0}')
[[ $model_shard_count -eq $expected_model_shards ]] ||
  die "model shard count mismatch: $model_shard_count/$expected_model_shards"
[[ $model_shard_bytes -eq $expected_model_shard_bytes ]] ||
  die "model shard bytes mismatch: $model_shard_bytes/$expected_model_shard_bytes"

(
  cd /
  sha256sum -c "$reference_evidence/input_sha256.txt"
)

actual_start_runner_sha256=$(sha256sum "$start_runner" | awk '{print $1}')
[[ $actual_start_runner_sha256 == "$expected_start_runner_sha256" ]] ||
  die "start runner hash mismatch: $actual_start_runner_sha256"
actual_atom_commit=$(git -C "$source_root" rev-parse HEAD)
[[ $actual_atom_commit == "$expected_atom_commit" ]] || die "ATOM commit mismatch: $actual_atom_commit"
actual_atom_patch_sha256=$(git -C "$source_root" diff --binary | sha256sum | awk '{print $1}')
[[ $actual_atom_patch_sha256 == "$expected_atom_patch_sha256" ]] ||
  die "ATOM worktree patch mismatch: $actual_atom_patch_sha256"
saved_atom_patch_sha256=$(sha256sum "$reference_evidence/atom_source_worktree.patch" | awk '{print $1}')
[[ $saved_atom_patch_sha256 == "$expected_atom_patch_sha256" ]] ||
  die "saved evidence patch mismatch: $saved_atom_patch_sha256"
bash -n "$start_runner"

step_number=$((step_number + 1))
printf '\n========== START STEP %d: 确认端口所有权并精确替换旧服务（如有） ==========\n' "$step_number"
occupied_ports=()
for port in 18000 18001 18080; do
  if [[ -n $(ss -ltnH "sport = :$port") ]]; then
    occupied_ports+=("$port")
  fi
done

is_running_pinned_pair() {
  local candidate_tag=$1
  local candidate_service=mi325_dsv4_${candidate_tag}_dual
  local candidate_router=mi325_dsv4_${candidate_tag}_router

  docker container inspect "$candidate_service" "$candidate_router" >/dev/null 2>&1 || return 1
  [[ $(docker container inspect "$candidate_service" --format '{{.State.Running}}') == true ]] || return 1
  [[ $(docker container inspect "$candidate_router" --format '{{.State.Running}}') == true ]] || return 1
  [[ $(docker container inspect "$candidate_service" --format '{{.Image}}') == "$atom_image_id" ]] || return 1
  [[ $(docker container inspect "$candidate_router" --format '{{.Image}}') == "$atom_image_id" ]] || return 1
}

old_tag=
old_tag_source=
if [[ ${#occupied_ports[@]} -gt 0 ]]; then
  log "检测到占用端口：${occupied_ports[*]}"
  docker ps --format '{{.Names}}\t{{.Status}}\t{{.Image}}'

  if [[ -f $state_file ]]; then
    state_tag=$(jq -r '.service_tag // empty' "$state_file")
    if [[ $state_tag =~ ^[a-z0-9_]+$ ]] && is_running_pinned_pair "$state_tag"; then
      old_tag=$state_tag
      old_tag_source=active_service_state
    fi
  fi
  if [[ -z $old_tag ]] && is_running_pinned_pair "$known_report_service_tag"; then
    old_tag=$known_report_service_tag
    old_tag_source=known_report_service
  fi

  if [[ -z $old_tag ]]; then
    discovered_tags=()
    while IFS= read -r running_name; do
      if [[ $running_name =~ ^mi325_dsv4_([a-z0-9_]+)_dual$ ]]; then
        discovered_tag=${BASH_REMATCH[1]}
        if is_running_pinned_pair "$discovered_tag"; then
          discovered_tags+=("$discovered_tag")
        fi
      fi
    done < <(docker ps --format '{{.Names}}')

    if [[ ${#discovered_tags[@]} -eq 1 ]]; then
      old_tag=${discovered_tags[0]}
      old_tag_source=uniquely_discovered_pinned_pair
      log "发现唯一一对未登记但名称成对、运行中且镜像匹配的本任务容器：$old_tag"
    elif [[ ${#discovered_tags[@]} -eq 0 ]]; then
      die '端口被占用，但未发现可验证的本任务 dual/router 容器对；拒绝停止'
    else
      printf '发现多对可能的本任务容器，无法唯一确认：\n' >&2
      printf '  %s\n' "${discovered_tags[@]}" >&2
      die '候选容器不唯一；拒绝停止'
    fi
  fi

  old_service=mi325_dsv4_${old_tag}_dual
  old_router=mi325_dsv4_${old_tag}_router
  [[ $(docker container inspect "$old_service" --format '{{.Image}}') == "$atom_image_id" ]] ||
    die "old service image mismatch: $old_service"
  [[ $(docker container inspect "$old_router" --format '{{.Image}}') == "$atom_image_id" ]] ||
    die "old router image mismatch: $old_router"
  printf '识别来源：%s\n' "$old_tag_source"
  printf '即将仅停止并删除以下两个精确容器：\n  %s\n  %s\n' "$old_service" "$old_router"
  read -r -p '按 Enter 停止上述占用任务，按 Ctrl-C 取消：' _
  docker stop "$old_router" "$old_service"
  docker rm "$old_router" "$old_service"
  old_service_stopped=1
  old_stopped_tag=$old_tag
else
  log '端口 18000、18001、18080 均空闲'
  read -r -p '按 Enter 继续启动服务，按 Ctrl-C 停止：' _
fi

for port in 18000 18001 18080; do
  [[ -z $(ss -ltnH "sport = :$port") ]] || die "port $port remains occupied"
done

step '创建启动证据并部署 2 x TP4 ATOM 与 least-connections Router'
mkdir -p "$service_evidence"
if [[ -f $state_file ]]; then
  cp "$state_file" "$service_evidence/previous_active_service.json"
fi

for card in $(seq 0 7); do
  rocm-smi -d "$card" --showrasinfo
done > "$service_evidence/ras_before_start.txt" 2>&1
rocm-smi --showproductname --showuniqueid --showuse --showmemuse --showpower --showtemp \
  > "$service_evidence/gpu_before_start.txt" 2>&1
rocm-smi --showperflevel --showclocks > "$service_evidence/clocks_before_start.txt" 2>&1
rocm-smi --showtopo > "$service_evidence/gpu_topology.txt" 2>&1 || true
docker image inspect "$atom_image" > "$service_evidence/atom_image_inspect.json"
git -C "$source_root" rev-parse HEAD > "$service_evidence/atom_source_commit.txt"
git -C "$source_root" status --short > "$service_evidence/atom_source_status.txt"
git -C "$source_root" diff --binary > "$service_evidence/atom_source_worktree.patch"
sha256sum \
  "$start_runner" \
  "$reference_evidence/atom_source_worktree.patch" \
  "$model_root/config.json" \
  > "$service_evidence/service_input_sha256.txt"

export CANDIDATE_TOPOLOGY=dual_tp4
export CANDIDATE_CAPTURE_SIZES='[1,2,4,8,16,24,25,26,27,28,29,30,31,32,48,64,72,96,128]'
export CANDIDATE_ATTN_PREFILL_CHUNK_SIZE=32768
export CANDIDATE_MAX_NUM_BATCHED_TOKENS=131072
export CANDIDATE_MAX_NUM_SEQS=128
export CANDIDATE_LONG_PREFILL_TOKEN_THRESHOLD=0
export CANDIDATE_SCHEDULER_DELAY_FACTOR=0.0
export CANDIDATE_GPU_MEMORY_UTILIZATION=0.83
export CANDIDATE_SPECULATIVE_METHOD=mtp
export CANDIDATE_NUM_SPECULATIVE_TOKENS=2
export CANDIDATE_MODEL_ROOT=$model_root

new_service_started=1
"$start_runner" "$service_tag" > "$service_evidence/service_start.log" 2>&1
docker inspect "$service_container" "$router_container" > "$service_evidence/service_inspect.json"
curl -fsS http://127.0.0.1:18080/router/status > "$service_evidence/router_status.json"

step '验证服务健康、固定配置和 Effective BF16 KV'
for port in 18000 18001; do
  curl -fsS --max-time 5 "http://127.0.0.1:$port/v1/models" >/dev/null ||
    die "backend health failed: $port"
done
curl -fsS --max-time 5 http://127.0.0.1:18080/v1/models >/dev/null || die 'router health failed'
jq -e '
  .policy == "least_connections" and
  (.backends | has("http://127.0.0.1:18000")) and
  (.backends | has("http://127.0.0.1:18001"))
' "$service_evidence/router_status.json" >/dev/null || die 'router policy/backend validation failed'

for port in 18000 18001; do
  server_log=$server_log_root/${service_tag}_${port}_server.log
  require_file "$server_log"
  rg -q 'Falling back to a bf16 KV cache' "$server_log" ||
    die "expected gfx942 FP8-to-BF16 KV fallback not found: $server_log"
done

service_container_id=$(docker inspect "$service_container" --format '{{.Id}}')
router_container_id=$(docker inspect "$router_container" --format '{{.Id}}')
started_at_utc=$(date --utc +%Y-%m-%dT%H:%M:%SZ)
mkdir -p "$split_root"
service_state_candidate=$service_evidence/service_state.json
jq -n \
  --arg service_tag "$service_tag" \
  --arg service_container "$service_container" \
  --arg router_container "$router_container" \
  --arg service_container_id "$service_container_id" \
  --arg router_container_id "$router_container_id" \
  --arg atom_image "$atom_image" \
  --arg atom_image_id "$atom_image_id" \
  --arg model "$model_root" \
  --arg evidence_dir "$service_evidence" \
  --arg started_at_utc "$started_at_utc" '
  {
    schema_version: 1,
    service_tag: $service_tag,
    service_container: $service_container,
    router_container: $router_container,
    service_container_id: $service_container_id,
    router_container_id: $router_container_id,
    atom_image: $atom_image,
    atom_image_id: $atom_image_id,
    model: $model,
    topology: "2xTP4",
    backend_ports: [18000, 18001],
    router_port: 18080,
    router_policy: "least_connections",
    requested_kv_cache_dtype: "fp8",
    effective_kv_cache_dtype: "bf16",
    speculative_method: "mtp",
    num_speculative_tokens: 2,
    evidence_dir: $evidence_dir,
    started_at_utc: $started_at_utc
  }
' > "$service_state_candidate"
sha256sum "$(readlink -f "$0")" > "$service_evidence/start_script.sha256"

for card in $(seq 0 7); do
  rocm-smi -d "$card" --showrasinfo
done > "$service_evidence/ras_after_start.txt" 2>&1
cmp -s "$service_evidence/ras_before_start.txt" "$service_evidence/ras_after_start.txt" ||
  die 'RAS changed while starting service'

state_tmp=$split_root/.active_service.${service_tag}.tmp
cp "$service_state_candidate" "$state_tmp"
mv "$state_tmp" "$state_file"

trap - EXIT
printf '\nService START PASS\n'
printf 'Service tag:      %s\n' "$service_tag"
printf 'Service:          %s\n' "$service_container"
printf 'Router:           %s\n' "$router_container"
printf 'State file:       %s\n' "$state_file"
printf 'Evidence:         %s\n' "$service_evidence"
printf 'Requested KV:     FP8\n'
printf 'Effective KV:     BF16 on gfx942\n'
printf '\n下一步运行：\n  %s/02_run_test.sh\n' "$asset_root"
