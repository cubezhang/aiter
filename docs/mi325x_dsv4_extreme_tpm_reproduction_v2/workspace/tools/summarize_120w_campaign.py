#!/usr/bin/env python3
"""Aggregate repeat MI325X C56 runs and evaluate the priority 120W gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_summary(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"missing run summary: {summary_path}")
    value = json.loads(summary_path.read_text(encoding="utf-8"))
    value["_run_dir"] = str(run_dir.resolve())
    value["_summary_sha256"] = sha256(summary_path)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate repeated strict runs and generate campaign evidence."
    )
    parser.add_argument("--run-dir", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-tpm", type=float, default=1_200_000.0)
    parser.add_argument("--max-mean-ttft-ms", type=float, default=2_000.0)
    parser.add_argument("--max-mean-tpot-ms", type=float, default=20.0)
    parser.add_argument("--min-success-rate", type=float, default=0.995)
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="return 2 unless every measured round passes the priority and quality gates",
    )
    args = parser.parse_args()

    runs = [load_summary(path) for path in args.run_dir]
    if not runs:
        raise ValueError("at least one run is required")

    rows: list[dict[str, Any]] = []
    for run in runs:
        overall = run["overall"]
        completion = overall["completion_tokens"]
        priority_pass = bool(
            overall["total_tpm"] >= args.target_tpm
            and overall["ttft_ms"]["mean"] < args.max_mean_ttft_ms
        )
        quality_pass = bool(
            run["integrity"]["pass"]
            and overall["success_rate"] >= args.min_success_rate
            and overall["tpot_ms"]["mean"] < args.max_mean_tpot_ms
            and completion["min"] == 1024.0
            and completion["max"] == 1024.0
            and run["integrity"]["cached_tokens_total"] == 0
        )
        rows.append(
            {
                "run_label": run["run_label"],
                "run_dir": run["_run_dir"],
                "summary_sha256": run["_summary_sha256"],
                "concurrency": run["concurrency"],
                "total_tpm": overall["total_tpm"],
                "ttft_mean_ms": overall["ttft_ms"]["mean"],
                "ttft_p95_ms": overall["ttft_ms"]["p95"],
                "tpot_mean_ms": overall["tpot_ms"]["mean"],
                "tpot_p95_ms": overall["tpot_ms"]["p95"],
                "success_rate": overall["success_rate"],
                "successful": overall["successful"],
                "attempted": overall["attempted"],
                "integrity": run["integrity"]["pass"],
                "cached_tokens_total": run["integrity"]["cached_tokens_total"],
                "backend_tpm_imbalance": run["backend_balance"]["tpm_imbalance"],
                "backend_balance_lt_5pct": run["backend_balance"]["target_lt_5pct"],
                "priority_pass": priority_pass,
                "quality_pass": quality_pass,
                "pass": priority_pass and quality_pass,
            }
        )

    tpms = [float(row["total_tpm"]) for row in rows]
    ttfts = [float(row["ttft_mean_ms"]) for row in rows]
    tpots = [float(row["tpot_mean_ms"]) for row in rows]
    tpm_mean = statistics.fmean(tpms)
    tpm_stdev = statistics.stdev(tpms) if len(tpms) > 1 else 0.0
    result = {
        "schema_version": 1,
        "gate": {
            "target_tpm_gte": args.target_tpm,
            "mean_ttft_ms_lt": args.max_mean_ttft_ms,
            "mean_tpot_ms_lt": args.max_mean_tpot_ms,
            "success_rate_gte": args.min_success_rate,
            "completion_tokens_exact": 1024,
            "integrity_required": True,
            "cached_tokens_required": 0,
            "backend_imbalance_lt_5pct_observational_for_priority_gate": True,
        },
        "rounds": rows,
        "aggregate": {
            "round_count": len(rows),
            "all_rounds_pass": all(row["pass"] for row in rows),
            "all_priority_pass": all(row["priority_pass"] for row in rows),
            "all_quality_pass": all(row["quality_pass"] for row in rows),
            "tpm_median": statistics.median(tpms),
            "tpm_mean": tpm_mean,
            "tpm_min": min(tpms),
            "tpm_max": max(tpms),
            "tpm_stdev": tpm_stdev,
            "tpm_cv": tpm_stdev / tpm_mean if tpm_mean else None,
            "ttft_mean_across_rounds_ms": statistics.fmean(ttfts),
            "ttft_worst_round_mean_ms": max(ttfts),
            "tpot_mean_across_rounds_ms": statistics.fmean(tpots),
            "gain_vs_historical_1120000": statistics.median(tpms) / 1_120_000 - 1,
            "ratio_vs_h200_1380000": statistics.median(tpms) / 1_380_000,
            "gap_to_h200_1380000": 1_380_000 - statistics.median(tpms),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "campaign_summary.json"
    md_path = args.output_dir / "campaign_summary.md"
    tsv_path = args.output_dir / "campaign_summary.tsv"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    columns = [
        "run_label",
        "concurrency",
        "total_tpm",
        "ttft_mean_ms",
        "tpot_mean_ms",
        "success_rate",
        "integrity",
        "backend_tpm_imbalance",
        "priority_pass",
        "quality_pass",
        "pass",
        "summary_sha256",
    ]
    with tsv_path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(columns) + "\n")
        for row in rows:
            handle.write("\t".join(str(row[column]) for column in columns) + "\n")

    aggregate = result["aggregate"]
    table_lines = [
        "| Run | C | Total TPM | Mean TTFT | Mean TPOT | Success | Imbalance | Pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        imbalance = row["backend_tpm_imbalance"]
        imbalance_text = f"{imbalance:.2%}" if imbalance is not None else "n/a"
        table_lines.append(
            f"| {row['run_label']} | {row['concurrency']} | {row['total_tpm']:,.3f} "
            f"| {row['ttft_mean_ms']:,.3f} ms | {row['tpot_mean_ms']:,.6f} ms "
            f"| {row['success_rate']:.4%} | {imbalance_text} | {row['pass']} |"
        )
    markdown = f"""# MI325X 120W TPM reproduction summary

Priority gate: Total TPM >= {args.target_tpm:,.0f}, Mean TTFT < {args.max_mean_ttft_ms:,.0f} ms.

{chr(10).join(table_lines)}

- All rounds pass: {aggregate['all_rounds_pass']}
- TPM median / min / max: {aggregate['tpm_median']:,.3f} / {aggregate['tpm_min']:,.3f} / {aggregate['tpm_max']:,.3f}
- TPM standard deviation / CV: {aggregate['tpm_stdev']:,.3f} / {aggregate['tpm_cv']:.4%}
- Mean TTFT across rounds: {aggregate['ttft_mean_across_rounds_ms']:,.3f} ms
- Worst round Mean TTFT: {aggregate['ttft_worst_round_mean_ms']:,.3f} ms
- Gain vs historical MI325 1,120,000 TPM: {aggregate['gain_vs_historical_1120000']:.2%}
- Ratio vs H200 1,380,000 TPM: {aggregate['ratio_vs_h200_1380000']:.2%}
- Gap to H200: {aggregate['gap_to_h200_1380000']:,.3f} TPM

The backend imbalance <5% field is reported for contract planning but is not part
of the user's current priority gate. Screening results are not a substitute for
the full 3997-request x 5 cold-restart acceptance procedure.
"""
    md_path.write_text(markdown, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 2 if args.require_pass and not aggregate["all_rounds_pass"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
