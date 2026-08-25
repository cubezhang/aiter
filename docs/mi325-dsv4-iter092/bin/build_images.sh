#!/usr/bin/env bash
set -euo pipefail

repro_root=$(readlink -f "$(dirname "${BASH_SOURCE[0]}")/..")
service_tag=rocm/atom-dev:iter092-replay
client_tag=locust-awcloud:iter092-rebuild
service_source=rocm/atom-dev:nightly_202608201458@sha256:a5cfa1ab503af6e0f55e0ed83cd7e999edbea6940a38dba44aaeee9a22758976
service_image_id=sha256:a5cfa1ab503af6e0f55e0ed83cd7e999edbea6940a38dba44aaeee9a22758976
lock_file="$repro_root/requirements-locust.lock"
lock_sha=$(sha256sum "$lock_file" | awk '{print $1}')

# A zero-layer Docker build export creates a new manifest ID. Pull and retag
# the immutable public manifest so the formal launcher still sees a5cfa1...
docker pull "$service_source"
docker tag "$service_source" "$service_tag"

DOCKER_BUILDKIT=1 docker build --pull \
  --file "$repro_root/Dockerfile.locust" \
  --build-arg "LOCUST_LOCK_SHA256=$lock_sha" \
  --tag "$client_tag" \
  "$repro_root"

actual_service_id=$(docker image inspect "$service_tag" --format '{{.Id}}')
[[ $actual_service_id == "$service_image_id" ]] || {
  echo "service image ID mismatch: $actual_service_id" >&2
  exit 1
}

# Revision checks do not need GPU access. Importing AITER does: its import path
# probes rocminfo, so keep that GPU-aware preflight in REPRODUCTION.md.
docker run --rm --entrypoint bash "$service_tag" -lc '
  test "$(git -C /app/ATOM rev-parse HEAD)" = \
    1e7659fde32eeaa0d9aa868c3e90847e5e46a51c
  test "$(git -C /app/aiter-test rev-parse HEAD)" = \
    eb84cb02200b1707f1076edf3f4930d3626adfb2
'

actual_lock_sha=$(
  docker image inspect "$client_tag" \
    --format '{{index .Config.Labels "io.aiter.iter092.locust-lock-sha256"}}'
)
[[ $actual_lock_sha == "$lock_sha" ]] || {
  echo "Locust lock label mismatch: $actual_lock_sha" >&2
  exit 1
}

docker run --rm \
  --entrypoint /root/locust_test/.venv/bin/python \
  "$client_tag" -c '
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
print("Locust compatibility client verified")
'

printf 'SERVICE_IMAGE=%s\nSERVICE_IMAGE_ID=%s\n' "$service_tag" "$actual_service_id"
printf 'LOCUST_IMAGE=%s\nLOCUST_IMAGE_ID=%s\nLOCUST_LOCK_SHA256=%s\n' \
  "$client_tag" "$(docker image inspect "$client_tag" --format '{{.Id}}')" "$lock_sha"
