#!/usr/bin/env bash
set -euo pipefail

root=$(readlink -f "$(dirname "${BASH_SOURCE[0]}")/..")
profile=official_clean_1e7659f_dual_tp4_mtp2_leg_noprefix_flysort

# Service-only gate: intentionally independent of Locust and datasets.
"$root/bin/verify_service_inputs.sh"

export TPOT_WORKFLOW_ROOT="$root/service"
export TPOT_CAMPAIGN_ROOT="$root"
export TPOT_ITERATION_ID=92

exec python3 "$root/service/actions/restore_iter092_winner_export.py" "$profile"
