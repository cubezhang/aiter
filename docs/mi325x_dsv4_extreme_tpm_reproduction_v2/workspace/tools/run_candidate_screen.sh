#!/usr/bin/env bash
set -euo pipefail

service_tag=${1:?usage: run_candidate_screen.sh SERVICE_TAG CANDIDATE_LABEL [CONCURRENCY...]}
candidate_label=${2:?usage: run_candidate_screen.sh SERVICE_TAG CANDIDATE_LABEL [CONCURRENCY...]}
shift 2
if [[ $# -eq 0 ]]; then
  concurrencies=(53 55)
else
  concurrencies=("$@")
fi

export STRICT_SERVICE_TAG=$service_tag
runner=/data/mi325_0811/mi325_dsv4_opt_0811/tools/run_strict_point.sh
manifest=/data/models/mi325_dsv4_extreme_tpm_20260812/02_manifests/generated/screen_750.jsonl
root=/data/models/mi325_dsv4_extreme_tpm_20260812/03_runs
progress=$root/${candidate_label}_SCREEN_PROGRESS.tsv

printf 'repetition\tconcurrency\trun_label\ttotal_tpm\tsuccess_rate\tttft_mean_ms\tttft_p95_ms\ttpot_mean_ms\ttpot_p95_ms\tmean_legal\tintegrity\tbackend_imbalance\n' > "$progress"
for concurrency in "${concurrencies[@]}"; do
  for repetition in 1 2 3; do
    label=${candidate_label}_SCREEN_C${concurrency}_R${repetition}
    run_dir=$root/$label
    summary=$run_dir/summary.json
    if [[ ! -s $summary ]]; then
      "$runner" "$manifest" "$concurrency" "$label" "$run_dir" \
        > "$root/${label}.console.log" 2>&1
    fi
    python3 - "$progress" "$repetition" "$concurrency" "$label" "$summary" <<'PY'
import json
import sys

progress, repetition, concurrency, label, summary = sys.argv[1:]
d = json.load(open(summary, encoding="utf-8"))
o = d["overall"]
values = [
    repetition, concurrency, label, f'{o["total_tpm"]:.9f}',
    f'{o["success_rate"]:.9f}', f'{o["ttft_ms"]["mean"]:.9f}',
    f'{o["ttft_ms"]["p95"]:.9f}', f'{o["tpot_ms"]["mean"]:.9f}',
    f'{o["tpot_ms"]["p95"]:.9f}', str(d["mean_slo_legal"]),
    str(d["integrity"]["pass"]),
    f'{d["backend_balance"]["tpm_imbalance"]:.9f}',
]
with open(progress, "a", encoding="utf-8") as handle:
    handle.write("\t".join(values) + "\n")
print("\t".join(values), flush=True)
PY
  done
done
