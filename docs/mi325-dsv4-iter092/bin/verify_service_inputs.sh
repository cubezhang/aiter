#!/usr/bin/env bash
set -euo pipefail

repro_root=$(readlink -f "$(dirname "${BASH_SOURCE[0]}")/..")
# shellcheck disable=SC1091
source "$repro_root/config/service.env"

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
jq -e --argjson expected_gpu_count "$EXPECTED_GPU_COUNT" '
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
jq -e --argjson expected_gpu_count "$EXPECTED_GPU_COUNT" '
  (.current_partition | length) == $expected_gpu_count and
  all(.current_partition[];
    .memory == "NPS1" and .accelerator_type == "SPX"
  )
' <<<"$partition_json" >/dev/null || {
  echo "partition mismatch: require all 8 GPUs in SPX/NPS1" >&2
  exit 1
}

[[ $SERVICE_MODEL_DIR == /* ]] || {
  echo "SERVICE_MODEL_DIR must be an absolute host path: $SERVICE_MODEL_DIR" >&2
  exit 1
}
[[ -d $SERVICE_MODEL_DIR ]] || {
  echo "missing model directory: $SERVICE_MODEL_DIR" >&2
  exit 1
}

actual_service_image_id=$(docker image inspect "$SERVICE_IMAGE" --format '{{.Id}}')
[[ $actual_service_image_id == "$SERVICE_IMAGE_ID" ]] || {
  echo "service image ID mismatch: $actual_service_image_id" >&2
  exit 1
}

printf 'Service inputs verified: %s x MI325X SPX/NPS1, model=%s, pinned ATOM image.\n' \
  "$EXPECTED_GPU_COUNT" "$SERVICE_MODEL_DIR"
