#!/usr/bin/env bash
set -euo pipefail

# Run the frozen C53 warm-up and C56/3997-request measured round against the
# service recorded by 01_start_service.sh. No arguments are accepted.

asset_root=/data/models/mi325_dsv4_extreme_tpm_20260812
reference_evidence=$asset_root/14_c56_full_single_acceptance/c56_full_single_20260813
split_root=$asset_root/15_c56_split_reproduction
workspace=/data/mi325_0811/mi325_dsv4_opt_0811
model_root=/data/DeepSeek-V4-Flash-FP8
manifest_root=$asset_root/02_manifests/generated
state_file=$split_root/active_service.json
point_runner=$workspace/tools/run_strict_point.sh
campaign_summarizer=$workspace/tools/summarize_120w_campaign.py
locust_image=locust-awcloud@sha256:7df3afaaaf1cca96eb9898ba7f23d9c5e4f531d86e8c02ab9f2f488eb9448631
locust_image_id=sha256:7df3afaaaf1cca96eb9898ba7f23d9c5e4f531d86e8c02ab9f2f488eb9448631
atom_image_id=sha256:66df2fb1c537f52d47f2d3bf973b8da15278dd2ee362b5307e3d29dfad69fa7d
expected_manifest_sha256=cca558fc66a2db34ba87bee64a4d2b0351310dee8cbd4820672231f40f4f88ad
expected_point_runner_sha256=b8d257f2c1d2df18b92fdc50f04e99ac659772fe2163e86b0dda06314f877338
expected_campaign_summarizer_sha256=c91523ea8676ea5d37f47d4a85e6276e9061ded398289b1cad9137779edb1c90
target_tpm=1240000
max_mean_ttft_ms=2000
campaign_tag=c56_split_test_$(date --utc +%Y%m%d_%H%M%S)
campaign_dir=$split_root/test_runs/$campaign_tag
warmup_label=${campaign_tag}_WARMUP_C53
warmup_dir=$campaign_dir/$warmup_label
run_label=${campaign_tag}_C56_R1
run_dir=$campaign_dir/$run_label
server_log_root=/data/aiter/mi325_dsv4_extreme_20260812/logs
step_number=0

log() {
  printf '[%s] %s\n' "$(date --utc +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

step() {
  step_number=$((step_number + 1))
  printf '\n========== TEST STEP %d: %s ==========\n' "$step_number" "$*"
  read -r -p '按 Enter 继续，按 Ctrl-C 停止：' _
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

require_file() {
  [[ -f $1 ]] || die "missing required file: $1"
}

[[ $# -eq 0 ]] || die '该脚本不接受任何参数；请直接运行 02_run_test.sh'

step '读取启动脚本状态并检查固定测试输入'
for command_name in awk bash cmp cp curl docker jq lspci mkdir mv python3 readlink rg rocm-smi seq sha256sum wc; do
  require_command "$command_name"
done
docker info >/dev/null 2>&1 || die 'Docker daemon 不可用，或当前用户无权访问 Docker socket'
require_file "$state_file"
require_file "$point_runner"
require_file "$campaign_summarizer"
require_file "$manifest_root/full_3997.jsonl"
require_file "$manifest_root/screen_750.jsonl"
require_file "$reference_evidence/input_sha256.txt"
[[ ! -e $campaign_dir ]] || die "refusing existing campaign directory: $campaign_dir"

jq -e '
  .schema_version == 1 and
  (.service_tag | type == "string") and
  .topology == "2xTP4" and
  .router_policy == "least_connections" and
  .requested_kv_cache_dtype == "fp8" and
  .effective_kv_cache_dtype == "bf16"
' "$state_file" >/dev/null || die 'active service state is invalid'

service_tag=$(jq -r '.service_tag' "$state_file")
service_container=$(jq -r '.service_container' "$state_file")
router_container=$(jq -r '.router_container' "$state_file")
expected_service_container=mi325_dsv4_${service_tag}_dual
expected_router_container=mi325_dsv4_${service_tag}_router
[[ $service_tag =~ ^[a-z0-9_]+$ ]] || die "invalid service tag in state: $service_tag"
[[ $service_container == "$expected_service_container" ]] || die 'service container name/state mismatch'
[[ $router_container == "$expected_router_container" ]] || die 'router container name/state mismatch'

actual_point_runner_sha256=$(sha256sum "$point_runner" | awk '{print $1}')
[[ $actual_point_runner_sha256 == "$expected_point_runner_sha256" ]] ||
  die "point runner hash mismatch: $actual_point_runner_sha256"
actual_campaign_summarizer_sha256=$(sha256sum "$campaign_summarizer" | awk '{print $1}')
[[ $actual_campaign_summarizer_sha256 == "$expected_campaign_summarizer_sha256" ]] ||
  die "campaign summarizer hash mismatch: $actual_campaign_summarizer_sha256"
actual_manifest_sha256=$(sha256sum "$manifest_root/full_3997.jsonl" | awk '{print $1}')
[[ $actual_manifest_sha256 == "$expected_manifest_sha256" ]] ||
  die "full manifest hash mismatch: $actual_manifest_sha256"
(
  cd /
  sha256sum -c "$reference_evidence/input_sha256.txt"
)
bash -n "$point_runner"

actual_locust_image_id=$(docker image inspect "$locust_image" --format '{{.Id}}' 2>/dev/null) ||
  die "missing pinned Locust image: $locust_image"
[[ $actual_locust_image_id == "$locust_image_id" ]] || die "Locust image ID mismatch: $actual_locust_image_id"

step '重新确认 MI325X 身份和 perf_determinism'
pci_count=$(lspci -nn | awk '/1002:74a5/{count++} END{print count+0}')
[[ $pci_count -eq 8 ]] || die "expected 8 x 1002:74a5, found $pci_count"
product_info=$(rocm-smi --showproductname 2>&1)
market_count=$(awk '/Card Series:.*MI325X/{count++} END{print count+0}' <<<"$product_info")
gfx_count=$(awk '/GFX Version:.*gfx942/{count++} END{print count+0}' <<<"$product_info")
[[ $market_count -eq 8 ]] || die "expected 8 MI325X market names, found $market_count"
[[ $gfx_count -eq 8 ]] || die "expected 8 gfx942 devices, found $gfx_count"
perf_info=$(rocm-smi --showperflevel --showclocks 2>&1)
printf '%s\n' "$perf_info"
perf_count=$(awk '/Performance Level: perf_determinism/{count++} END{print count+0}' <<<"$perf_info")
[[ $perf_count -eq 8 ]] || die "expected perf_determinism on all 8 GPUs, found $perf_count"

step '验证状态文件对应的服务、Router、镜像和 Effective BF16 KV'
docker container inspect "$service_container" "$router_container" >/dev/null 2>&1 ||
  die 'recorded service containers do not exist'
[[ $(docker inspect "$service_container" --format '{{.State.Running}}') == true ]] || die 'service container is not running'
[[ $(docker inspect "$router_container" --format '{{.State.Running}}') == true ]] || die 'router container is not running'
[[ $(docker inspect "$service_container" --format '{{.State.OOMKilled}}') == false ]] || die 'service container was OOM-killed'
[[ $(docker inspect "$service_container" --format '{{.Image}}') == "$atom_image_id" ]] || die 'service image ID mismatch'
[[ $(docker inspect "$router_container" --format '{{.Image}}') == "$atom_image_id" ]] || die 'router image ID mismatch'
[[ $(docker inspect "$service_container" --format '{{.Id}}') == "$(jq -r '.service_container_id' "$state_file")" ]] ||
  die 'service container ID differs from start state'
[[ $(docker inspect "$router_container" --format '{{.Id}}') == "$(jq -r '.router_container_id' "$state_file")" ]] ||
  die 'router container ID differs from start state'

for port in 18000 18001; do
  curl -fsS --max-time 5 "http://127.0.0.1:$port/v1/models" >/dev/null || die "backend health failed: $port"
  server_log=$server_log_root/${service_tag}_${port}_server.log
  require_file "$server_log"
  rg -q 'Falling back to a bf16 KV cache' "$server_log" || die "BF16 KV fallback not proven: $server_log"
done
curl -fsS --max-time 5 http://127.0.0.1:18080/v1/models >/dev/null || die 'router health failed'
router_status=$(curl -fsS --max-time 5 http://127.0.0.1:18080/router/status)
jq -e '
  .policy == "least_connections" and
  (.backends | has("http://127.0.0.1:18000")) and
  (.backends | has("http://127.0.0.1:18001"))
' <<<"$router_status" >/dev/null || die 'router status mismatch'

step '创建测试证据目录和冻结配置'
mkdir -p "$campaign_dir"
cp "$state_file" "$campaign_dir/service_state_before_test.json"
docker inspect "$service_container" "$router_container" > "$campaign_dir/service_inspect_before_test.json"
printf '%s\n' "$router_status" > "$campaign_dir/router_before_test.json"
rocm-smi --showuse --showmemuse --showpower --showtemp > "$campaign_dir/gpu_before_test.txt" 2>&1
rocm-smi --showperflevel --showclocks > "$campaign_dir/clocks_before_test.txt" 2>&1
for card in $(seq 0 7); do
  rocm-smi -d "$card" --showrasinfo
done > "$campaign_dir/ras_before_campaign.txt" 2>&1

sha256sum \
  "$point_runner" \
  "$workspace/tools/locust_strict_contract.py" \
  "$workspace/tools/summarize_strict_run.py" \
  "$campaign_summarizer" \
  "$manifest_root/full_3997.jsonl" \
  "$manifest_root/screen_750.jsonl" \
  "$model_root/config.json" \
  "$campaign_dir/service_state_before_test.json" \
  > "$campaign_dir/test_input_sha256.txt"

jq -n \
  --arg campaign_tag "$campaign_tag" \
  --arg service_tag "$service_tag" \
  --arg measure_manifest "$manifest_root/full_3997.jsonl" \
  --arg warmup_manifest "$manifest_root/screen_750.jsonl" \
  --arg manifest_sha256 "$expected_manifest_sha256" \
  --arg locust_image "$locust_image" \
  --argjson target_tpm "$target_tpm" \
  --argjson max_mean_ttft_ms "$max_mean_ttft_ms" '
  {
    schema_version: 1,
    evidence_stage: "full_manifest_formal_single_round",
    campaign_tag: $campaign_tag,
    service_tag: $service_tag,
    topology: "2xTP4",
    warmup_concurrency: 53,
    measure_concurrency: 56,
    warmup_manifest: $warmup_manifest,
    measure_manifest: $measure_manifest,
    measure_manifest_sha256: $manifest_sha256,
    repeats: 1,
    completion_tokens_exact: 1024,
    target_tpm: $target_tpm,
    max_mean_ttft_ms: $max_mean_ttft_ms,
    max_mean_tpot_ms: 20,
    min_success_rate: 0.995,
    requested_kv_cache_dtype: "fp8",
    effective_kv_cache_dtype: "bf16",
    locust_image: $locust_image
  }
' > "$campaign_dir/campaign_config.json"
sha256sum "$(readlink -f "$0")" > "$campaign_dir/test_script.sha256"

step '执行排除计分的 C53 / 750-request Warm-up'
export STRICT_SERVICE_TAG=$service_tag
"$point_runner" \
  "$manifest_root/screen_750.jsonl" 53 "$warmup_label" "$warmup_dir" \
  > "$campaign_dir/${warmup_label}.console.log" 2>&1
jq -e '
  .integrity.pass == true and
  .overall.attempted == 750 and
  .overall.successful == 750 and
  .overall.completion_tokens.min == 1024 and
  .overall.completion_tokens.max == 1024 and
  .integrity.cached_tokens_total == 0
' "$warmup_dir/summary.json" >/dev/null || die 'warm-up integrity/quality gate failed'
log 'Warm-up 完成且排除计分'

step '执行 C56 / 完整 3,997-request 正式单轮'
"$point_runner" \
  "$manifest_root/full_3997.jsonl" 56 "$run_label" "$run_dir" \
  > "$campaign_dir/${run_label}.console.log" 2>&1

step '生成活动汇总并执行 124W 独立验收'
set +e
python3 "$campaign_summarizer" \
  --run-dir "$run_dir" \
  --output-dir "$campaign_dir" \
  --target-tpm "$target_tpm" \
  --max-mean-ttft-ms "$max_mean_ttft_ms" \
  > "$campaign_dir/campaign_summary.console.json"
summarizer_rc=$?
set -e

for card in $(seq 0 7); do
  rocm-smi -d "$card" --showrasinfo
done > "$campaign_dir/ras_after_campaign.txt" 2>&1
docker inspect "$service_container" "$router_container" > "$campaign_dir/service_inspect_after_test.json" 2>&1 || true
curl -fsS http://127.0.0.1:18080/router/status > "$campaign_dir/router_after_test.json" 2>&1 || true
[[ $summarizer_rc -eq 0 ]] || die "campaign summarizer failed with rc=$summarizer_rc"

jq -e \
  --arg manifest_sha256 "$expected_manifest_sha256" \
  --argjson target_tpm "$target_tpm" \
  --argjson max_mean_ttft_ms "$max_mean_ttft_ms" '
    .manifest_sha256 == $manifest_sha256 and
    .integrity.pass == true and
    .integrity.expected_requests == 3997 and
    .integrity.actual_requests == 3997 and
    .integrity.raw_worker_files == 8 and
    .integrity.expected_thinking == 1599 and
    .integrity.actual_thinking == 1599 and
    .integrity.cached_tokens_total == 0 and
    (.errors | length) == 0 and
    .overall.attempted == 3997 and
    .overall.successful == 3997 and
    .overall.success_rate >= 0.995 and
    .overall.total_tpm >= $target_tpm and
    .overall.ttft_ms.mean < $max_mean_ttft_ms and
    .overall.tpot_ms.mean < 20 and
    .overall.completion_tokens.min == 1024 and
    .overall.completion_tokens.max == 1024 and
    .backend_balance.tpm_imbalance < 0.05
  ' "$run_dir/summary.json" >/dev/null || die 'formal measured-run gate failed'

raw_line_count=$(wc -l "$run_dir"/requests_worker_*.jsonl | awk '/ total$/{print $1}')
[[ $raw_line_count -eq 3997 ]] || die "raw JSONL line count mismatch: $raw_line_count/3997"
cmp -s "$run_dir/ras_before.txt" "$run_dir/ras_after.txt" || die 'formal-run RAS before/after differs'
if rg -n -i \
  'oom|traceback|segfault|engine.*(dead|death)|fatal|uncorrectable|xgmi.*error|gpu reset|amdgpu.*error' \
  "$run_dir/kernel_delta.log" \
  "$run_dir/server_18000_delta.log" \
  "$run_dir/server_18001_delta.log"; then
  die 'runtime error signature found in formal measured-run logs'
fi

[[ $(docker inspect "$service_container" --format '{{.State.Running}}' 2>/dev/null) == true ]] ||
  die 'service container is not running after the formal test'
[[ $(docker inspect "$router_container" --format '{{.State.Running}}' 2>/dev/null) == true ]] ||
  die 'router container is not running after the formal test'
[[ $(docker inspect "$service_container" --format '{{.State.OOMKilled}}' 2>/dev/null) == false ]] ||
  die 'service container was OOM-killed during the formal test'
for port in 18000 18001; do
  curl -fsS --max-time 5 "http://127.0.0.1:$port/v1/models" >/dev/null ||
    die "backend is unhealthy after the formal test: $port"
done
curl -fsS --max-time 5 http://127.0.0.1:18080/v1/models >/dev/null ||
  die 'router is unhealthy after the formal test'

cmp -s "$campaign_dir/ras_before_campaign.txt" "$campaign_dir/ras_after_campaign.txt" ||
  die 'campaign-level RAS before/after differs'
sha256sum \
  "$campaign_dir/campaign_summary.json" \
  "$campaign_dir/campaign_summary.md" \
  "$campaign_dir/campaign_summary.tsv" \
  "$run_dir/summary.json" \
  > "$campaign_dir/test_output_sha256.txt"

state_tmp=$split_root/.active_service.test_update.tmp
jq \
  --arg last_test_tag "$campaign_tag" \
  --arg last_test_dir "$campaign_dir" \
  --arg last_test_at_utc "$(date --utc +%Y-%m-%dT%H:%M:%SZ)" '
  .last_test_tag = $last_test_tag |
  .last_test_dir = $last_test_dir |
  .last_test_at_utc = $last_test_at_utc
' "$state_file" > "$state_tmp"
mv "$state_tmp" "$state_file"

step '显示正式结果和证据路径'
jq '{
  run_label,
  concurrency,
  manifest_sha256,
  integrity,
  overall: {
    attempted: .overall.attempted,
    successful: .overall.successful,
    measurement_seconds: .overall.measurement_seconds,
    input_tokens: .overall.input_tokens,
    output_tokens: .overall.output_tokens,
    total_tokens: .overall.total_tokens,
    total_tpm: .overall.total_tpm,
    ttft_ms: .overall.ttft_ms,
    tpot_ms: .overall.tpot_ms
  },
  backend_balance,
  errors,
  mean_slo_legal,
  p95_slo_legal
}' "$run_dir/summary.json"

summary_sha256=$(sha256sum "$run_dir/summary.json" | awk '{print $1}')
printf '\nTest REPRODUCTION PASS\n'
printf 'Evidence stage:     full 3,997-request formal single round\n'
printf 'Service tag:        %s\n' "$service_tag"
printf 'Campaign:           %s\n' "$campaign_dir"
printf 'Measured run:       %s\n' "$run_dir"
printf 'summary.json SHA:   %s\n' "$summary_sha256"
printf 'Requested KV:       FP8\n'
printf 'Effective KV:       BF16 on gfx942\n'
printf 'Service retained:   yes\n'
