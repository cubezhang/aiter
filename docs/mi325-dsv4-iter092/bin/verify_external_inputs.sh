#!/usr/bin/env bash
set -euo pipefail

repro_root=$(readlink -f "$(dirname "${BASH_SOURCE[0]}")/..")
locust_script="$repro_root/contract/locustfile-request-count-0916.py"
dataset_dir="$repro_root/input/llm_test_datasets-prod"
model_dir=/data/DeepSeek-V4-Flash-FP8

expected_script_sha=f01146774dd3b05b6a105918f444cec03ed9d5575e2c92d40ac58522bb327850
expected_dataset_count=3997
expected_dataset_sha=c774104d35e521baddb4dc7d57646ac9aac68eeedb57c6962f9d1c85cbe0d7d8
service_image=rocm/atom-dev@sha256:a5cfa1ab503af6e0f55e0ed83cd7e999edbea6940a38dba44aaeee9a22758976
service_image_id=sha256:a5cfa1ab503af6e0f55e0ed83cd7e999edbea6940a38dba44aaeee9a22758976
locust_image=locust-awcloud@sha256:7df3afaaaf1cca96eb9898ba7f23d9c5e4f531d86e8c02ab9f2f488eb9448631
locust_image_id=sha256:7df3afaaaf1cca96eb9898ba7f23d9c5e4f531d86e8c02ab9f2f488eb9448631
expected_gpu_count=8

for required_command in docker curl jq rg sha256sum rocm-smi amd-smi journalctl ss; do
  command -v "$required_command" >/dev/null || {
    echo "missing command: $required_command" >&2
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
[[ -f $locust_script ]] || { echo "missing private Locust script: $locust_script" >&2; exit 1; }
[[ -d $dataset_dir ]] || { echo "missing private dataset directory: $dataset_dir" >&2; exit 1; }

actual_script_sha=$(sha256sum "$locust_script" | awk '{print $1}')
[[ $actual_script_sha == "$expected_script_sha" ]] || {
  echo "Locust script SHA256 mismatch: $actual_script_sha" >&2
  exit 1
}

actual_dataset_count=$(find "$dataset_dir" -maxdepth 1 -type f -name '*.json' | wc -l)
[[ $actual_dataset_count -eq $expected_dataset_count ]] || {
  echo "dataset count mismatch: $actual_dataset_count/$expected_dataset_count" >&2
  exit 1
}

actual_dataset_sha=$(
  (
    cd "$dataset_dir"
    find . -maxdepth 1 -type f -name '*.json' -print0 |
      sort -z |
      xargs -0 sha256sum
  ) | sha256sum | awk '{print $1}'
)
[[ $actual_dataset_sha == "$expected_dataset_sha" ]] || {
  echo "dataset aggregate SHA256 mismatch: $actual_dataset_sha" >&2
  exit 1
}

actual_service_image_id=$(docker image inspect "$service_image" --format '{{.Id}}')
[[ $actual_service_image_id == "$service_image_id" ]] || {
  echo "service image ID mismatch: $actual_service_image_id" >&2
  exit 1
}
actual_locust_image_id=$(docker image inspect "$locust_image" --format '{{.Id}}')
[[ $actual_locust_image_id == "$locust_image_id" ]] || {
  echo "Locust image ID mismatch: $actual_locust_image_id" >&2
  exit 1
}

echo "External inputs verified: 8 x MI325X SPX/NPS1, model, client script, 3997 datasets, and both images."
