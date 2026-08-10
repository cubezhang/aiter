#!/usr/bin/env bash
set -euo pipefail
for name in mi325_dsv4_e004_router mi325_dsv4_e004_dual; do
  if docker ps --format '{{.Names}}' | rg -qx "$name"; then
    docker stop --timeout 30 "$name"
  fi
  if docker ps -a --format '{{.Names}}' | rg -qx "$name"; then
    docker rm "$name"
  fi
done
