#!/usr/bin/env bash
set -euo pipefail

model_dir=/data/DeepSeek-V4-Flash-FP8
service_image=rocm/atom-dev:nightly_202608201458@sha256:a5cfa1ab503af6e0f55e0ed83cd7e999edbea6940a38dba44aaeee9a22758976
service_image_id=sha256:a5cfa1ab503af6e0f55e0ed83cd7e999edbea6940a38dba44aaeee9a22758976
expected_gpu_count=8

for required_command in docker curl jq python3 rocm-smi amd-smi journalctl ss; do
  command -v "$required_command" >/dev/null || {
    echo "missing service command: $required_command" >&2
    exit 1
  }
done

[[ -e /dev/kfd && -d /dev/dri ]] || {
  echo "missing AMD GPU devices: /dev/kfd or /dev/dri" >&2
  exit 1
}

asic_json=$(amd-smi static --asic --json)
jq -e --argjson expected_gpu_count "$expected_gpu_count" '
  (.gpu_data | length) == $expected_gpu_count and
  all(.gpu_data[];
    .asic.vendor_id == "0x1002" and
    .asic.device_id == "0x74a5" and
    (.asic.market_name | contains("MI325X")) and
    .asic.target_graphics_version == "gfx942"
  )
' <<<"$asic_json" >/dev/null || {
  echo "platform mismatch: require 8 x MI325X (1002:74a5, gfx942)" >&2
  exit 1
}

partition_json=$(amd-smi partition --current --json)
jq -e --argjson expected_gpu_count "$expected_gpu_count" '
  (.current_partition | length) == $expected_gpu_count and
  all(.current_partition[];
    .memory == "NPS1" and .accelerator_type == "SPX"
  )
' <<<"$partition_json" >/dev/null || {
  echo "partition mismatch: require all 8 GPUs in SPX/NPS1" >&2
  exit 1
}

[[ -d $model_dir ]] || { echo "missing model directory: $model_dir" >&2; exit 1; }

actual_service_image_id=$(docker image inspect "$service_image" --format '{{.Id}}')
[[ $actual_service_image_id == "$service_image_id" ]] || {
  echo "service image ID mismatch: $actual_service_image_id" >&2
  exit 1
}

echo "Service inputs verified: 8 x MI325X SPX/NPS1, model, and pinned ATOM image."
