#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 MANIFEST CONCURRENCY RUN_LABEL RUN_DIR" >&2
  exit 64
fi

manifest_path=$(readlink -f "$1")
concurrency=$2
run_label=$3
run_dir=$(readlink -m "$4")
dataset_path=/data/models/mi325_dsv4_extreme_tpm_20260812/02_manifests/dataset
tool_path=/data/mi325_0811/mi325_dsv4_opt_0811/tools
locust_image=locust-awcloud@sha256:7df3afaaaf1cca96eb9898ba7f23d9c5e4f531d86e8c02ab9f2f488eb9448631
container_name="strict-${run_label//[^a-zA-Z0-9_.-]/-}"
service_tag=${STRICT_SERVICE_TAG:-e005}
service_container=mi325_dsv4_${service_tag}_dual
router_container=mi325_dsv4_${service_tag}_router

mkdir -p "$run_dir"
if find "$run_dir" -mindepth 1 -maxdepth 1 | read -r _; then
  echo "refusing non-empty run directory: $run_dir" >&2
  exit 65
fi

start_epoch=$(date +%s)
start_utc=$(date --utc +%Y-%m-%dT%H:%M:%SZ)
curl -fsS http://127.0.0.1:18080/router/status -o "$run_dir/router_before.json"
mapfile -t backend_ports < <(
  jq -r '.backends | keys[] | capture(":(?<port>[0-9]+)$").port' \
    "$run_dir/router_before.json" | sort -n
)
if [[ ${#backend_ports[@]} -lt 1 ]]; then
  echo "expected at least one router backend" >&2
  exit 66
fi
declare -A server_start_lines
for port in "${backend_ports[@]}"; do
  server_log=/service_logs/${service_tag}_${port}_server.log
  server_start_lines[$port]=$(docker exec "$service_container" sh -lc "wc -l < $server_log")
done
docker inspect "$service_container" "$router_container" > "$run_dir/service_inspect_before.json"
for card in $(seq 0 7); do
  rocm-smi -d "$card" --showrasinfo
done > "$run_dir/ras_before.txt" 2>&1

monitor_gpu() {
  while true; do
    date --utc +%Y-%m-%dT%H:%M:%SZ
    rocm-smi --showuse --showmemuse --showpower --showtemp 2>&1
    sleep 5
  done
}
monitor_gpu > "$run_dir/gpu_telemetry.txt" &
monitor_pid=$!

set +e
docker run --rm --name "$container_name" --network host \
  --mount "type=bind,src=$tool_path,dst=/work/tools,readonly" \
  --mount "type=bind,src=$dataset_path,dst=/work/dataset,readonly" \
  --mount "type=bind,src=$(dirname "$manifest_path"),dst=/work/manifests,readonly" \
  --mount "type=bind,src=$run_dir,dst=/work/result" \
  --entrypoint bash "$locust_image" -lc \
  "/root/locust_test/.venv/bin/locust -f /work/tools/locust_strict_contract.py --headless --processes 8 --expect-workers 8 --only-summary --loglevel INFO --host http://127.0.0.1:18080 -u $concurrency -r $concurrency --run-time 7200s --stop-timeout 120 --strict-manifest /work/manifests/$(basename "$manifest_path") --strict-dataset-dir /work/dataset --strict-result-dir /work/result --strict-worker-count 8 --strict-run-id $run_label" \
  > "$run_dir/locust.log" 2>&1
locust_rc=$?
set -e

kill "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true
curl -fsS http://127.0.0.1:18080/router/status -o "$run_dir/router_after.json" || true
for card in $(seq 0 7); do
  rocm-smi -d "$card" --showrasinfo
done > "$run_dir/ras_after.txt" 2>&1
journalctl -k --since "@$start_epoch" --no-pager > "$run_dir/kernel_delta.log" 2>&1 || true
for port in "${backend_ports[@]}"; do
  server_log=/service_logs/${service_tag}_${port}_server.log
  start_line=${server_start_lines[$port]}
  docker exec "$service_container" sh -lc \
    "sed -n '$((start_line + 1)),\$p' $server_log" \
    > "$run_dir/server_${port}_delta.log" 2>&1 || true
done

end_utc=$(date --utc +%Y-%m-%dT%H:%M:%SZ)
python3 - "$run_dir/run_metadata.json" "$run_label" "$concurrency" "$manifest_path" "$start_utc" "$end_utc" "$locust_rc" <<'PY'
import hashlib
import json
import pathlib
import sys

output, label, concurrency, manifest, started, ended, rc = sys.argv[1:]
manifest_path = pathlib.Path(manifest)
value = {
    "schema_version": 1,
    "run_label": label,
    "concurrency": int(concurrency),
    "manifest": str(manifest_path),
    "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    "started_at_utc": started,
    "ended_at_utc": ended,
    "locust_exit_code": int(rc),
    "locust_image": "locust-awcloud@sha256:7df3afaaaf1cca96eb9898ba7f23d9c5e4f531d86e8c02ab9f2f488eb9448631",
    "endpoint": "http://127.0.0.1:18080/v1/chat/completions",
    "worker_count": 8,
}
pathlib.Path(output).write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
PY

python3 "$tool_path/summarize_strict_run.py" \
  --manifest "$manifest_path" --result-dir "$run_dir" \
  --concurrency "$concurrency" --run-label "$run_label"

# Locust's built-in exit policy returns 1 for any single request failure,
# whereas the frozen contract permits up to 0.5% failures and evaluates them
# from raw records. Preserve locust_exit_code in run_metadata.json, but let the
# strict summarizer/integrity result govern orchestration continuity.
exit 0
