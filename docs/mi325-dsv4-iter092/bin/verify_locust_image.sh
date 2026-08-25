#!/usr/bin/env bash
set -euo pipefail

repro_root=$(readlink -f "$(dirname "${BASH_SOURCE[0]}")/..")
export CAMPAIGN_ROOT="$repro_root"
# shellcheck disable=SC1091
source "$repro_root/config/contract.env"

actual_image_id=$(docker image inspect "$LOCUST_IMAGE" --format '{{.Id}}')
[[ $LOCUST_IMAGE == locust-awcloud:1.5 ]] || {
  echo "unsupported Locust image: $LOCUST_IMAGE" >&2
  exit 1
}
[[ $actual_image_id == "$LOCUST_IMAGE_ID" ]] || {
  printf 'Locust image ID mismatch:\n  image=%s\n  actual=%s\n  expected=%s\n' \
    "$LOCUST_IMAGE" "$actual_image_id" "$LOCUST_IMAGE_ID" >&2
  exit 1
}

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
  "$LOCUST_IMAGE" "$actual_image_id" FROZEN_FORMAL_IMAGE
