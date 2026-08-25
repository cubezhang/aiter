#!/usr/bin/env bash
set -euo pipefail

repro_root=$(readlink -f "$(dirname "${BASH_SOURCE[0]}")/..")

# Backward-compatible aggregate check. The service and test workflows call
# their narrower verifiers directly and remain independent.
"$repro_root/bin/verify_service_inputs.sh"
"$repro_root/bin/verify_test_inputs.sh"

echo "All service and test inputs verified."
