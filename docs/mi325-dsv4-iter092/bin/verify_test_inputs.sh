#!/usr/bin/env bash
set -euo pipefail

repro_root=$(readlink -f "$(dirname "${BASH_SOURCE[0]}")/..")
export CAMPAIGN_ROOT="$repro_root"
# shellcheck disable=SC1091
source "$repro_root/config/contract.env"

for required_command in docker curl jq python3 rg sha256sum rocm-smi journalctl; do
  command -v "$required_command" >/dev/null || {
    echo "missing test command: $required_command" >&2
    exit 1
  }
done

[[ -n $LOCUST_SCRIPT ]] || {
  echo "set CUSTOMER_TEST_ROOT or LOCUST_SCRIPT to the local private workload" >&2
  exit 1
}
[[ -n $DATASET_DIR ]] || {
  echo "set CUSTOMER_TEST_ROOT or DATASET_DIR to the local private workload" >&2
  exit 1
}
[[ $LOCUST_SCRIPT == /* && $DATASET_DIR == /* ]] || {
  echo "LOCUST_SCRIPT and DATASET_DIR must be absolute host paths" >&2
  exit 1
}
[[ -f $LOCUST_SCRIPT ]] || { echo "missing private Locust script: $LOCUST_SCRIPT" >&2; exit 1; }
[[ -d $DATASET_DIR ]] || { echo "missing private dataset directory: $DATASET_DIR" >&2; exit 1; }

actual_script_sha=$(sha256sum "$LOCUST_SCRIPT" | awk '{print $1}')
[[ $actual_script_sha == "$LOCUST_SCRIPT_SHA256" ]] || {
  echo "Locust script SHA256 mismatch: $actual_script_sha" >&2
  exit 1
}

actual_dataset_count=$(find "$DATASET_DIR" -maxdepth 1 -type f -name '*.json' | wc -l)
[[ $actual_dataset_count -eq $DATASET_FILE_COUNT ]] || {
  echo "dataset count mismatch: $actual_dataset_count/$DATASET_FILE_COUNT" >&2
  exit 1
}

actual_dataset_sha=$(
  (
    cd "$DATASET_DIR"
    find . -maxdepth 1 -type f -name '*.json' -print0 |
      LC_ALL=C sort -z |
      xargs -0 sha256sum
  ) | sha256sum | awk '{print $1}'
)
[[ $actual_dataset_sha == "$DATASET_AGGREGATE_SHA256" ]] || {
  printf 'dataset aggregate SHA256 mismatch:\n  DATASET_DIR=%s\n  actual=%s\n  expected=%s\n' \
    "$DATASET_DIR" "$actual_dataset_sha" "$DATASET_AGGREGATE_SHA256" >&2
  exit 1
}

LOCUST_IMAGE="$LOCUST_IMAGE" LOCUST_IMAGE_ID="$LOCUST_IMAGE_ID" \
  "$repro_root/bin/verify_locust_image.sh"

printf 'Test inputs verified:\n  LOCUST_SCRIPT=%s\n  DATASET_DIR=%s\n  LOCUST_IMAGE=%s\n' \
  "$LOCUST_SCRIPT" "$DATASET_DIR" "$LOCUST_IMAGE"
