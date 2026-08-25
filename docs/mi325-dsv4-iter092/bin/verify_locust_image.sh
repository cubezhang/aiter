#!/usr/bin/env bash
set -euo pipefail

repro_root=$(readlink -f "$(dirname "${BASH_SOURCE[0]}")/..")
export CAMPAIGN_ROOT="$repro_root"
# shellcheck disable=SC1091
source "$repro_root/config/contract.env"

actual_image_id=$(docker image inspect "$LOCUST_IMAGE" --format '{{.Id}}')
if [[ -n $LOCUST_IMAGE_ID ]]; then
  [[ $actual_image_id == "$LOCUST_IMAGE_ID" ]] || {
    echo "Locust image ID mismatch: $actual_image_id" >&2
    exit 1
  }
  evidence_class=FROZEN_FORMAL_IMAGE
else
  expected_lock_sha=$(sha256sum "$repro_root/requirements-locust.lock" | awk '{print $1}')
  actual_lock_sha=$(
    docker image inspect "$LOCUST_IMAGE" \
      --format '{{index .Config.Labels "io.aiter.iter092.locust-lock-sha256"}}'
  )
  [[ $actual_lock_sha == "$expected_lock_sha" ]] || {
    echo "Locust compatibility image lock mismatch: $actual_lock_sha" >&2
    exit 1
  }
  evidence_class=SOURCE_REBUILT_COMPATIBILITY
fi

docker run --rm \
  --entrypoint /root/locust_test/.venv/bin/python \
  "$LOCUST_IMAGE" -c '
import importlib.metadata as metadata
import sys
expected = {
    "locust": "2.37.4",
    "gevent": "24.11.1",
    "requests": "2.32.3",
    "numpy": "2.3.3",
    "Flask": "3.1.1",
}
assert sys.version.split()[0] == "3.12.11", sys.version
for package, version in expected.items():
    assert metadata.version(package) == version, (package, metadata.version(package))
'

printf 'Locust image verified: %s %s %s\n' \
  "$LOCUST_IMAGE" "$actual_image_id" "$evidence_class"
