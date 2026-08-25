#!/usr/bin/env bash
set -euo pipefail

repro_root=$(readlink -f "$(dirname "${BASH_SOURCE[0]}")/..")

# Convenience wrapper only. Service and test setup have independent entrypoints.
"$repro_root/bin/prepare_service_image.sh"
"$repro_root/bin/build_locust_image.sh"
