#!/usr/bin/env bash
set -euo pipefail

runner=/data/mi325_0811/mi325_dsv4_opt_0811/tools/run_strict_point.sh
manifest_root=/data/models/mi325_dsv4_extreme_tpm_20260812/02_manifests/generated
run_root=/data/models/mi325_dsv4_extreme_tpm_20260812/10_final_cold_restart
progress=$run_root/V11_FINAL_C53_PROGRESS.tsv

export STRICT_SERVICE_TAG=finalv11
mkdir -p "$run_root"

warmup_label=V11_FINAL_WARMUP_C53
warmup_dir=$run_root/$warmup_label
if [[ ! -s "$warmup_dir/summary.json" ]]; then
  "$runner" "$manifest_root/screen_750.jsonl" 53 "$warmup_label" "$warmup_dir" \
    > "$run_root/${warmup_label}.console.log" 2>&1
fi

printf 'repetition\tconcurrency\trun_label\ttotal_tpm\tsuccess_rate\tttft_mean_ms\tttft_p95_ms\tttft_p99_ms\ttpot_mean_ms\ttpot_p95_ms\ttpot_p99_ms\tmean_legal\tintegrity\tbackend_imbalance\n' > "$progress"
for repetition in 1 2 3 4 5; do
  label=V11_FINAL_C53_R$repetition
  run_dir=$run_root/$label
  summary=$run_dir/summary.json
  if [[ ! -s "$summary" ]]; then
    "$runner" "$manifest_root/full_3997.jsonl" 53 "$label" "$run_dir" \
      > "$run_root/${label}.console.log" 2>&1
  fi
  python3 - "$progress" "$repetition" "$label" "$summary" <<'PY'
import json
import sys

progress, repetition, label, summary = sys.argv[1:]
d = json.load(open(summary, encoding="utf-8"))
o = d["overall"]
v = [
    repetition,
    "53",
    label,
    f'{o["total_tpm"]:.9f}',
    f'{o["success_rate"]:.9f}',
    f'{o["ttft_ms"]["mean"]:.9f}',
    f'{o["ttft_ms"]["p95"]:.9f}',
    f'{o["ttft_ms"]["p99"]:.9f}',
    f'{o["tpot_ms"]["mean"]:.9f}',
    f'{o["tpot_ms"]["p95"]:.9f}',
    f'{o["tpot_ms"]["p99"]:.9f}',
    str(d["mean_slo_legal"]),
    str(d["integrity"]["pass"]),
    f'{d["backend_balance"]["tpm_imbalance"]:.9f}',
]
with open(progress, "a", encoding="utf-8") as handle:
    handle.write("\t".join(v) + "\n")
print("\t".join(v))
PY
done
