#!/usr/bin/env python3
"""Aggregate strict per-run inference summaries against configurable SLO gates."""

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


def parse_comparison(value: str) -> tuple[str, float]:
    try:
        label, number = value.rsplit("=", 1)
        if not label:
            raise ValueError
        return label, float(number)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("comparison must be LABEL=TPM") from exc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-tpm", type=float, required=True)
    parser.add_argument("--max-mean-ttft-ms", type=float, required=True)
    parser.add_argument("--max-mean-tpot-ms", type=float, default=20.0)
    parser.add_argument("--min-success-rate", type=float, default=0.995)
    parser.add_argument("--completion-tokens", type=int, default=1024)
    parser.add_argument("--max-backend-imbalance", type=float, default=0.05)
    parser.add_argument("--require-backend-balance", action="store_true")
    parser.add_argument("--minimum-rounds", type=int, default=3)
    parser.add_argument("--comparison", action="append", type=parse_comparison, default=[])
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()

    if args.minimum_rounds < 1:
        parser.error("--minimum-rounds must be positive")

    rows: list[dict[str, Any]] = []
    for path in args.summary:
        value = json.loads(path.read_text(encoding="utf-8"))
        overall = value["overall"]
        integrity = value["integrity"]
        completion = overall["completion_tokens"]
        imbalance = value.get("backend_balance", {}).get("tpm_imbalance")
        balance_pass = imbalance is not None and imbalance < args.max_backend_imbalance
        priority_pass = bool(
            overall["total_tpm"] >= args.target_tpm
            and overall["ttft_ms"]["mean"] < args.max_mean_ttft_ms
        )
        quality_pass = bool(
            integrity["pass"]
            and overall["success_rate"] >= args.min_success_rate
            and overall["tpot_ms"]["mean"] < args.max_mean_tpot_ms
            and completion["min"] == float(args.completion_tokens)
            and completion["max"] == float(args.completion_tokens)
            and integrity.get("cached_tokens_total", 0) == 0
            and (balance_pass or not args.require_backend_balance)
        )
        rows.append(
            {
                "summary": str(path.resolve()),
                "summary_sha256": sha256(path),
                "run_label": value["run_label"],
                "concurrency": value["concurrency"],
                "total_tpm": overall["total_tpm"],
                "ttft_mean_ms": overall["ttft_ms"]["mean"],
                "ttft_p95_ms": overall["ttft_ms"]["p95"],
                "tpot_mean_ms": overall["tpot_ms"]["mean"],
                "tpot_p95_ms": overall["tpot_ms"]["p95"],
                "success_rate": overall["success_rate"],
                "attempted": overall["attempted"],
                "successful": overall["successful"],
                "integrity": integrity["pass"],
                "cached_tokens_total": integrity.get("cached_tokens_total", 0),
                "backend_tpm_imbalance": imbalance,
                "backend_balance_pass": balance_pass,
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
    enough_rounds = len(rows) >= args.minimum_rounds
    all_rounds_pass = enough_rounds and all(row["pass"] for row in rows)
    median_tpm = statistics.median(tpms)
    comparisons = {
        label: {
            "reference_tpm": reference,
            "ratio": median_tpm / reference,
            "gain": median_tpm / reference - 1,
            "gap": reference - median_tpm,
        }
        for label, reference in args.comparison
    }
    result = {
        "schema_version": 1,
        "gate": {
            "target_tpm_gte": args.target_tpm,
            "mean_ttft_ms_lt": args.max_mean_ttft_ms,
            "mean_tpot_ms_lt": args.max_mean_tpot_ms,
            "success_rate_gte": args.min_success_rate,
            "completion_tokens_exact": args.completion_tokens,
            "integrity_required": True,
            "cached_tokens_required": 0,
            "minimum_rounds": args.minimum_rounds,
            "backend_imbalance_lt": args.max_backend_imbalance,
            "backend_balance_required": args.require_backend_balance,
        },
        "rounds": rows,
        "aggregate": {
            "round_count": len(rows),
            "enough_rounds": enough_rounds,
            "all_rounds_pass": all_rounds_pass,
            "tpm_median": median_tpm,
            "tpm_mean": tpm_mean,
            "tpm_min": min(tpms),
            "tpm_max": max(tpms),
            "tpm_stdev": tpm_stdev,
            "tpm_cv": tpm_stdev / tpm_mean if tpm_mean else None,
            "ttft_mean_across_rounds_ms": statistics.fmean(ttfts),
            "ttft_worst_round_mean_ms": max(ttfts),
            "tpot_mean_across_rounds_ms": statistics.fmean(tpots),
            "comparisons": comparisons,
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "slo_campaign_summary.json"
    md_path = args.output_dir / "slo_campaign_summary.md"
    tsv_path = args.output_dir / "slo_campaign_summary.tsv"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    columns = [
        "run_label", "concurrency", "total_tpm", "ttft_mean_ms",
        "tpot_mean_ms", "success_rate", "integrity", "backend_tpm_imbalance",
        "backend_balance_pass", "priority_pass", "quality_pass", "pass",
        "summary_sha256",
    ]
    with tsv_path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(columns) + "\n")
        for row in rows:
            handle.write("\t".join(str(row[column]) for column in columns) + "\n")

    table = [
        "| Run | C | Total TPM | Mean TTFT | Mean TPOT | Success | Imbalance | Pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        imbalance = row["backend_tpm_imbalance"]
        imbalance_text = f"{imbalance:.2%}" if imbalance is not None else "n/a"
        table.append(
            f"| {row['run_label']} | {row['concurrency']} | {row['total_tpm']:,.3f} "
            f"| {row['ttft_mean_ms']:,.3f} ms | {row['tpot_mean_ms']:,.6f} ms "
            f"| {row['success_rate']:.4%} | {imbalance_text} | {row['pass']} |"
        )
    comparison_lines = []
    for label, comparison in comparisons.items():
        comparison_lines.append(
            f"- {label}: ratio {comparison['ratio']:.2%}, gain {comparison['gain']:.2%}, "
            f"gap {comparison['gap']:,.3f} TPM"
        )
    markdown = f"""# SLO-constrained inference campaign

Gate: Total TPM >= {args.target_tpm:,.0f}, Mean TTFT < {args.max_mean_ttft_ms:,.0f} ms,
Mean TPOT < {args.max_mean_tpot_ms:,.3f} ms, Success >= {args.min_success_rate:.2%}.

{chr(10).join(table)}

- Enough rounds ({len(rows)}/{args.minimum_rounds}): {enough_rounds}
- All rounds pass: {all_rounds_pass}
- TPM median / min / max: {median_tpm:,.3f} / {min(tpms):,.3f} / {max(tpms):,.3f}
- TPM standard deviation / CV: {tpm_stdev:,.3f} / {result['aggregate']['tpm_cv']:.4%}
- Mean / worst-round Mean TTFT: {statistics.fmean(ttfts):,.3f} / {max(ttfts):,.3f} ms
- Backend balance required: {args.require_backend_balance}
{chr(10).join(comparison_lines)}
"""
    md_path.write_text(markdown, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 2 if args.require_pass and not all_rounds_pass else 0


if __name__ == "__main__":
    raise SystemExit(main())
