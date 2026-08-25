#!/usr/bin/env bash
set -euo pipefail

repro_root=$(readlink -f "$(dirname "${BASH_SOURCE[0]}")/..")
profile=official_clean_1e7659f_dual_tp4_mtp2_leg_noprefix_flysort
image_id=sha256:a5cfa1ab503af6e0f55e0ed83cd7e999edbea6940a38dba44aaeee9a22758976
state_file="$repro_root/state/active_service.json"

case ${1:-} in
  ""|--preflight-only) ;;
  *)
    echo "usage: $0 [--preflight-only]" >&2
    exit 2
    ;;
esac

mkdir -p \
  "$repro_root/logs" \
  "$repro_root/runs" \
  "$repro_root/state" \
  "$repro_root/service/logs" \
  "$repro_root/service/work"

"$repro_root/bin/verify_external_inputs.sh"

[[ -f $state_file ]] || {
  echo "missing active service state; run ./bin/restore_iter092_service.sh first" >&2
  exit 1
}
jq -e --arg profile "$profile" --arg image_id "$image_id" '
  .profile_id == $profile and
  .atom_image_id == $image_id and
  .topology == "2xTP4" and
  .router_policy == "least_connections" and
  .effective_kv_cache_dtype == "bf16" and
  .num_speculative_tokens == 2 and
  .prefix_caching == false
' "$state_file" >/dev/null || {
  echo "active service state does not match the iteration-92 winner" >&2
  exit 1
}

for role in service router; do
  container_name=$(jq -er --arg key "${role}_container" '.[$key]' "$state_file")
  expected_container_id=$(jq -er --arg key "${role}_container_id" '.[$key]' "$state_file")
  actual_container_state=$(
    docker inspect "$container_name" --format '{{.Id}} {{.State.Running}}'
  )
  [[ $actual_container_state == "$expected_container_id true" ]] || {
    echo "$role container does not match active service state" >&2
    exit 1
  }
done

for health_url in \
  http://127.0.0.1:18000/v1/models \
  http://127.0.0.1:18001/v1/models \
  http://127.0.0.1:18080/v1/models \
  http://127.0.0.1:18080/router/status; do
  curl -fsS --max-time 5 "$health_url" >/dev/null || {
    echo "service health check failed: $health_url" >&2
    exit 1
  }
done

echo "Iteration-92 service preflight passed; this script will not start or replace it."
[[ ${1:-} == --preflight-only ]] && exit 0

warm_stamp=$(date --utc +%Y%m%d_%H%M%S)
warm_dir="$repro_root/runs/i092_cold_warmup_c60_attempt_1_${warm_stamp}"
"$repro_root/bin/run_customer_point.sh" \
  tpot_phase10_c65_extreme \
  i092_cold_warmup_c60 \
  "$profile" \
  60 1 3 \
  "$warm_dir"

stable_zero=0
drain_deadline=$((SECONDS + 300))
while (( stable_zero < 3 && SECONDS < drain_deadline )); do
  active_requests=$(
    curl -fsS --max-time 3 http://127.0.0.1:18080/router/status |
      jq '[.backends[].active] | add'
  )
  if (( active_requests == 0 )); then
    stable_zero=$((stable_zero + 1))
  else
    stable_zero=0
  fi
  sleep 1
done
(( stable_zero >= 3 )) || { echo "service did not drain after warmup" >&2; exit 1; }

formal_stamp=$(date --utc +%Y%m%d_%H%M%S)
formal_dir="$repro_root/runs/i092_final_replay_attempt_1_${formal_stamp}"
"$repro_root/bin/run_customer_point.sh" \
  tpot_phase10_c65_extreme \
  i092_final_replay \
  "$profile" \
  60 1 20 \
  "$formal_dir"

jq '{
  valid,
  integrity_valid,
  tpot_slo_pass,
  concurrency,
  first_complete_score,
  kernel_gpu_fault_count
}' "$formal_dir/result.json"

echo "Formal result: $formal_dir/result.json"
