#!/usr/bin/env bash
set -euo pipefail

repro_root=$(readlink -f "$(dirname "${BASH_SOURCE[0]}")/..")
client_tag=locust-awcloud:iter092-rebuild
lock_file="$repro_root/requirements-locust.lock"
lock_sha=$(sha256sum "$lock_file" | awk '{print $1}')

DOCKER_BUILDKIT=1 docker build --pull \
  --file "$repro_root/Dockerfile.locust" \
  --build-arg "LOCUST_LOCK_SHA256=$lock_sha" \
  --tag "$client_tag" \
  "$repro_root"

LOCUST_IMAGE="$client_tag" LOCUST_IMAGE_ID= \
  "$repro_root/bin/verify_locust_image.sh" >/dev/null

printf 'LOCUST_IMAGE=%s\nLOCUST_IMAGE_ID=%s\nLOCUST_LOCK_SHA256=%s\n' \
  "$client_tag" "$(docker image inspect "$client_tag" --format '{{.Id}}')" "$lock_sha"
