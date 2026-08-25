#!/usr/bin/env bash
set -euo pipefail

service_tag=rocm/atom-dev:iter092-replay
service_source=rocm/atom-dev:nightly_202608201458@sha256:a5cfa1ab503af6e0f55e0ed83cd7e999edbea6940a38dba44aaeee9a22758976
service_image_id=sha256:a5cfa1ab503af6e0f55e0ed83cd7e999edbea6940a38dba44aaeee9a22758976

# A Docker build export creates a new manifest ID even with no filesystem
# layer. Pull and retag the immutable public manifest for strict replay.
docker pull "$service_source"
docker tag "$service_source" "$service_tag"

actual_service_id=$(docker image inspect "$service_tag" --format '{{.Id}}')
[[ $actual_service_id == "$service_image_id" ]] || {
  echo "service image ID mismatch: $actual_service_id" >&2
  exit 1
}

# These source checks need no GPU and do not import AITER.
docker run --rm --entrypoint bash "$service_tag" -lc '
  test "$(git -C /app/ATOM rev-parse HEAD)" = \
    1e7659fde32eeaa0d9aa868c3e90847e5e46a51c
  test "$(git -C /app/aiter-test rev-parse HEAD)" = \
    eb84cb02200b1707f1076edf3f4930d3626adfb2
'

printf 'SERVICE_IMAGE=%s\nSERVICE_IMAGE_ID=%s\n' \
  "$service_tag" "$actual_service_id"
