#!/usr/bin/env bash
set -euo pipefail

repro_root=$(readlink -f "$(dirname "${BASH_SOURCE[0]}")/..")
locust_script="$repro_root/contract/locustfile-request-count-0916.py"
dataset_dir="$repro_root/input/llm_test_datasets-prod"
expected_script_sha=f01146774dd3b05b6a105918f444cec03ed9d5575e2c92d40ac58522bb327850
expected_dataset_count=3997
expected_dataset_sha=c774104d35e521baddb4dc7d57646ac9aac68eeedb57c6962f9d1c85cbe0d7d8

for required_command in docker curl jq python3 rg sha256sum rocm-smi journalctl; do
  command -v "$required_command" >/dev/null || {
    echo "missing test command: $required_command" >&2
    exit 1
  }
done

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

"$repro_root/bin/verify_locust_image.sh"

echo "Test inputs verified: private Locust script, 3997 datasets, and client image."
