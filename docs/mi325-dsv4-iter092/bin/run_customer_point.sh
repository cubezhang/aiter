#!/usr/bin/env bash
set -euo pipefail

root=$(readlink -f "$(dirname "${BASH_SOURCE[0]}")/..")
export CAMPAIGN_ROOT="$root"
# shellcheck disable=SC1091
source "$root/config/contract.env"

wave=${1:?usage: run_customer_point.sh WAVE LABEL PROFILE C REP BLOCKS RUN_DIR}
label=${2:?usage: run_customer_point.sh WAVE LABEL PROFILE C REP BLOCKS RUN_DIR}
profile=${3:?usage: run_customer_point.sh WAVE LABEL PROFILE C REP BLOCKS RUN_DIR}
concurrency=${4:?usage: run_customer_point.sh WAVE LABEL PROFILE C REP BLOCKS RUN_DIR}
repetition=${5:?usage: run_customer_point.sh WAVE LABEL PROFILE C REP BLOCKS RUN_DIR}
blocks=${6:?usage: run_customer_point.sh WAVE LABEL PROFILE C REP BLOCKS RUN_DIR}
run_dir=${7:?usage: run_customer_point.sh WAVE LABEL PROFILE C REP BLOCKS RUN_DIR}

[[ $wave =~ ^[a-z0-9_]+$ ]] || { echo "invalid wave: $wave" >&2; exit 64; }
[[ $label =~ ^[a-z0-9_]+$ ]] || { echo "invalid label: $label" >&2; exit 64; }
[[ $profile =~ ^[a-z0-9_]+$ ]] || { echo "invalid profile: $profile" >&2; exit 64; }
[[ $concurrency =~ ^[1-9][0-9]*$ ]] || { echo "invalid concurrency: $concurrency" >&2; exit 64; }
[[ $repetition =~ ^[1-9][0-9]*$ ]] || { echo "invalid repetition: $repetition" >&2; exit 64; }
[[ $blocks =~ ^[1-9][0-9]*$ ]] || { echo "invalid score blocks: $blocks" >&2; exit 64; }
[[ $run_dir == "$root/runs/"* ]] || { echo "run dir is outside campaign: $run_dir" >&2; exit 64; }
[[ ! -e $run_dir ]] || { echo "refusing existing run dir: $run_dir" >&2; exit 65; }

event() {
  printf '[%s] %s\n' "$(date --utc +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$root/logs/progress.log"
}

capture_ras() {
  local target=$1
  for card in $(seq 0 7); do
    rocm-smi -d "$card" --showrasinfo
  done > "$target" 2>&1
}

capture_gpu() {
  local target=$1
  {
    date --utc +%Y-%m-%dT%H:%M:%SZ
    rocm-smi --showproductname --showperflevel --showuse --showmemuse --showpower --showtemp
  } > "$target" 2>&1
}

mkdir -p "$run_dir"
client_name="custv2_${label}_$(date --utc +%Y%m%d%H%M%S)"
[[ ${#client_name} -le 120 ]] || { echo "client container name too long" >&2; exit 64; }
docker container inspect "$client_name" >/dev/null 2>&1 && {
  echo "refusing existing client container: $client_name" >&2
  exit 65
}

actual_script_hash=$(sha256sum "$LOCUST_SCRIPT" | awk '{print $1}')
[[ $actual_script_hash == "$LOCUST_SCRIPT_SHA256" ]] || {
  echo "customer Locust script hash mismatch" >&2
  exit 66
}
actual_dataset_count=$(find "$DATASET_DIR" -maxdepth 1 -type f -name '*.json' | wc -l)
[[ $actual_dataset_count -eq $DATASET_FILE_COUNT ]] || {
  echo "dataset count mismatch: $actual_dataset_count/$DATASET_FILE_COUNT" >&2
  exit 66
}
actual_client_image=$(docker image inspect "$LOCUST_IMAGE" --format '{{.Id}}')
[[ $actual_client_image == "$LOCUST_IMAGE_ID" ]] || {
  echo "Locust image ID mismatch: $actual_client_image" >&2
  exit 66
}

service_state=$root/state/active_service.json
[[ -s $service_state ]] || { echo "missing active service state" >&2; exit 67; }
[[ $(jq -r '.profile_id' "$service_state") == "$profile" ]] || {
  echo "active service profile mismatch" >&2
  exit 67
}
service_container=$(jq -r '.service_container' "$service_state")
router_container=$(jq -r '.router_container' "$service_state")
service_id=$(jq -r '.service_container_id' "$service_state")
router_id=$(jq -r '.router_container_id' "$service_state")
[[ $(docker inspect "$service_container" --format '{{.Id}}') == "$service_id" ]] || exit 67
[[ $(docker inspect "$router_container" --format '{{.Id}}') == "$router_id" ]] || exit 67
curl -fsS --max-time 5 "$SERVICE_URL/v1/models" >/dev/null

started_at=$(date --utc +%Y-%m-%dT%H:%M:%SZ)
started_epoch=$(date +%s)
jq -n \
  --arg contract "$CONTRACT_ID" --arg wave "$wave" --arg label "$label" \
  --arg profile "$profile" --arg started "$started_at" --arg client "$client_name" \
  --arg script_hash "$actual_script_hash" --arg dataset_hash "$DATASET_AGGREGATE_SHA256" \
  --argjson concurrency "$concurrency" --argjson repetition "$repetition" \
  --argjson score_blocks "$blocks" --argjson tpot_limit_ms "$LOW_LATENCY_TPOT_MS" \
  --slurpfile service "$service_state" \
  '{schema_version:1,contract_id:$contract,wave:$wave,label:$label,
    service_profile:$profile,concurrency:$concurrency,repetition:$repetition,
    score_blocks:$score_blocks,started_at_utc:$started,client_container:$client,
    customer_script_sha256:$script_hash,dataset_aggregate_sha256:$dataset_hash,
    strict_result_used:false,tpot_limit_ms:$tpot_limit_ms,service:$service[0]}' > "$run_dir/metadata.json"

curl -fsS --max-time 5 "$SERVICE_URL/router/status" > "$run_dir/router_before.json"
docker inspect "$service_container" "$router_container" > "$run_dir/service_inspect.json"
capture_ras "$run_dir/ras_before.txt"
capture_gpu "$run_dir/gpu_before.txt"
amd-smi bad-pages --json > "$run_dir/bad_pages_before.json"

telemetry_loop() {
  while true; do
    {
      printf '\n===== %s =====\n' "$(date --utc +%Y-%m-%dT%H:%M:%SZ)"
      rocm-smi --showuse --showmemuse --showpower --showtemp
    } >> "$run_dir/gpu_telemetry.txt" 2>&1 || true
    sleep 15
  done
}
telemetry_loop &
telemetry_pid=$!
client_created=0
cleanup() {
  rc=$?
  kill "$telemetry_pid" >/dev/null 2>&1 || true
  wait "$telemetry_pid" >/dev/null 2>&1 || true
  if [[ $client_created == 1 ]] && docker container inspect "$client_name" >/dev/null 2>&1; then
    docker stop --time 30 "$client_name" >/dev/null 2>&1 || true
    docker logs "$client_name" > "$run_dir/container_stdout.log" 2>&1 || true
    docker rm "$client_name" >/dev/null 2>&1 || true
  fi
  if [[ $rc -ne 0 ]]; then
    event "POINT_PARTIAL label=$label profile=$profile C=$concurrency rc=$rc dir=$run_dir"
  fi
  exit "$rc"
}
trap cleanup EXIT INT TERM

event "POINT_CLIENT_START label=$label profile=$profile C=$concurrency blocks=$blocks container=$client_name"
docker run --detach --name "$client_name" \
  --network host \
  --mount "type=bind,src=$LOCUST_SCRIPT,dst=/contract/locustfile-request-count-0916.py,readonly" \
  --mount "type=bind,src=$DATASET_DIR,dst=/contract/llm_test_datasets-prod,readonly" \
  --mount "type=bind,src=$run_dir,dst=/run" \
  --workdir /run \
  --entrypoint /root/locust_test/.venv/bin/locust \
  "$LOCUST_IMAGE" \
  -f /contract/locustfile-request-count-0916.py \
  --headless --processes "$LOCUST_PROCESSES" --master-port 6568 --master-bind-port 6568 \
  --host "$SERVICE_URL" \
  --users "$concurrency" --spawn-rate "$concurrency" \
  --stop-timeout 120 \
  --csv /run/locust --csv-full-history \
  --logfile /run/locust.log \
  --model-name "$MODEL_NAME" --max-tokens "$MAX_TOKENS" \
  --include-usage true --thinking-enabled false \
  --dataset-dir /contract/llm_test_datasets-prod \
  --interval "$concurrency" --sla-interval 300 \
  > "$run_dir/client_container_id.txt"
client_created=1

deadline=$((started_epoch + POINT_GUARD_SECONDS))
last_progress_block=-1
score_seen=0
while (( $(date +%s) < deadline )); do
  if "$root/bin/parse_customer_result.py" "$run_dir" \
      --concurrency "$concurrency" --blocks "$blocks" --probe >/dev/null 2>&1; then
    score_seen=1
    break
  fi
  if ! docker inspect "$client_name" --format '{{.State.Running}}' 2>/dev/null | rg -q '^true$'; then
    echo "Locust client exited before score window completed" >&2
    exit 70
  fi
  current_block=$(rg -o "历史均值--5x \([0-9]+ 次统计--统计间隔 $concurrency\)" \
      "$run_dir/locust.log" 2>/dev/null | tail -n 1 | sed -E 's/.*\(([0-9]+) 次.*/\1/' || true)
  if [[ $current_block =~ ^[0-9]+$ && $current_block -ne $last_progress_block ]]; then
    last_progress_block=$current_block
    event "POINT_PROGRESS label=$label profile=$profile C=$concurrency blocks=$current_block/$blocks"
  fi
  sleep 3
done
[[ $score_seen == 1 ]] || { echo "point guard timeout before scoring window" >&2; exit 71; }

date --utc +%Y-%m-%dT%H:%M:%SZ > "$run_dir/score_cutoff_detected_at_utc"
"$root/bin/parse_customer_result.py" "$run_dir" \
  --concurrency "$concurrency" --blocks "$blocks" --probe > "$run_dir/score_at_cutoff.json"
event "POINT_SCORE_CUTOFF label=$label profile=$profile C=$concurrency; stopping exact client without scoring drain"
docker stop --time "$POINT_STOP_SECONDS" "$client_name" > "$run_dir/client_stop.txt"
docker logs "$client_name" > "$run_dir/container_stdout.log" 2>&1 || true
docker inspect "$client_name" > "$run_dir/client_inspect_after.json" 2>/dev/null || true
docker rm "$client_name" > "$run_dir/client_remove.txt"
client_created=0

kill "$telemetry_pid" >/dev/null 2>&1 || true
wait "$telemetry_pid" >/dev/null 2>&1 || true
capture_ras "$run_dir/ras_after.txt"
capture_gpu "$run_dir/gpu_after.txt"
amd-smi bad-pages --json > "$run_dir/bad_pages_after.json"
curl -fsS --max-time 5 "$SERVICE_URL/router/status" > "$run_dir/router_after.json"
journalctl -k --since "@$started_epoch" --no-pager > "$run_dir/kernel_window.txt" 2>&1 || true
rg -i 'amdgpu.*(page fault|gpu reset|fatal)|VM_L2_PROTECTION_FAULT|xgmi.*(fatal|error)|GPU reset' \
  "$run_dir/kernel_window.txt" > "$run_dir/kernel_gpu_faults.txt" || true

set +e
"$root/bin/parse_customer_result.py" "$run_dir" \
  --concurrency "$concurrency" --blocks "$blocks" > "$run_dir/result.console.json"
parse_rc=$?
set -e
if [[ -s $run_dir/result.json ]]; then
  total=$(jq -r '.first_complete_score.total_tpm_w // 0' "$run_dir/result.json")
  tpot=$(jq -r '.first_complete_score.tpot_ms // 0' "$run_dir/result.json")
  valid=$(jq -r '.valid' "$run_dir/result.json")
  event "POINT_RESULT label=$label profile=$profile C=$concurrency total_tpm_w=$total tpot_ms=$tpot valid=$valid"
fi
trap - EXIT INT TERM
exit "$parse_rc"
