#!/usr/bin/env python3
"""Validate and summarize one strict-contract load run from raw request JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * probability
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def describe(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0 if values else None,
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
    return rows


def segment(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("valid_success")]
    elapsed_s = (
        (max(row["response_end_perf_ns"] for row in rows) - min(row["started_perf_ns"] for row in rows)) / 1e9
        if rows
        else 0.0
    )
    input_tokens = sum(int(row["prompt_tokens"]) for row in valid)
    output_tokens = sum(int(row["completion_tokens"]) for row in valid)
    return {
        "attempted": len(rows),
        "successful": len(valid),
        "success_rate": len(valid) / len(rows) if rows else 0.0,
        "measurement_seconds": elapsed_s,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_tpm": input_tokens * 60 / elapsed_s if elapsed_s else 0.0,
        "output_tpm": output_tokens * 60 / elapsed_s if elapsed_s else 0.0,
        "total_tpm": (input_tokens + output_tokens) * 60 / elapsed_s if elapsed_s else 0.0,
        "prompt_tokens": describe([float(row["prompt_tokens"]) for row in valid]),
        "completion_tokens": describe([float(row["completion_tokens"]) for row in valid]),
        "ttft_ms": describe([float(row["ttft_ms"]) for row in valid]),
        "tpot_ms": describe([float(row["tpot_ms"]) for row in valid]),
        "e2e_ms": describe([float(row["e2e_ms"]) for row in valid]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--run-label", required=True)
    args = parser.parse_args()

    manifest = load_jsonl(args.manifest)
    raw_paths = sorted(args.result_dir.glob("requests_worker_*.jsonl"))
    records = [row for path in raw_paths for row in load_jsonl(path)]
    expected_positions = {int(row["position"]) for row in manifest}
    actual_positions = [int(row["manifest_position"]) for row in records]
    duplicate_positions = sorted({position for position in actual_positions if actual_positions.count(position) > 1})
    missing_positions = sorted(expected_positions - set(actual_positions))
    unexpected_positions = sorted(set(actual_positions) - expected_positions)
    manifest_by_position = {int(row["position"]): row for row in manifest}
    mismatches = []
    for row in records:
        expected = manifest_by_position.get(int(row["manifest_position"]))
        if expected is None:
            continue
        for field, record_field in [
            ("request_id", "request_id"),
            ("relative_filename", "relative_filename"),
            ("file_sha256", "file_sha256"),
            ("thinking", "thinking"),
        ]:
            if row.get(record_field) != expected.get(field):
                mismatches.append({"position": row["manifest_position"], "field": field})

    integrity_pass = all(
        [
            len(records) == len(manifest),
            not duplicate_positions,
            not missing_positions,
            not unexpected_positions,
            not mismatches,
            len(raw_paths) == 8,
        ]
    )
    overall = segment(records)
    valid = [row for row in records if row.get("valid_success")]
    expected_thinking = sum(bool(row["thinking"]) for row in manifest)
    actual_thinking = sum(bool(row["thinking"]) for row in records)
    cached_total = sum(int(row.get("cached_tokens") or 0) for row in records)
    mean_slo_legal = bool(
        integrity_pass
        and overall["success_rate"] >= 0.995
        and overall["ttft_ms"]["mean"] is not None
        and overall["ttft_ms"]["mean"] < 2000.0
        and overall["tpot_ms"]["mean"] is not None
        and overall["tpot_ms"]["mean"] < 20.0
        and all(row.get("completion_tokens") == 1024 for row in valid)
        and cached_total == 0
    )
    p95_slo_legal = bool(
        integrity_pass
        and overall["success_rate"] >= 0.995
        and overall["ttft_ms"]["p95"] is not None
        and overall["ttft_ms"]["p95"] < 2000.0
        and overall["tpot_ms"]["p95"] is not None
        and overall["tpot_ms"]["p95"] < 20.0
        and all(row.get("completion_tokens") == 1024 for row in valid)
        and cached_total == 0
    )
    errors: dict[str, int] = {}
    for row in records:
        if not row.get("valid_success"):
            key = str(row.get("error") or "unknown")
            errors[key] = errors.get(key, 0) + 1

    backend_balance: dict[str, Any] = {}
    backend_tpms: list[float] = []
    for backend in sorted({str(row.get("backend")) for row in records if row.get("backend")}):
        backend_rows = [row for row in records if row.get("backend") == backend]
        backend_valid = [row for row in backend_rows if row.get("valid_success")]
        backend_tokens = sum(int(row["total_tokens"]) for row in backend_valid)
        backend_tpm = backend_tokens * 60 / overall["measurement_seconds"] if overall["measurement_seconds"] else 0.0
        backend_tpms.append(backend_tpm)
        backend_balance[backend] = {
            "attempted": len(backend_rows),
            "successful": len(backend_valid),
            "total_tokens": backend_tokens,
            "total_tpm_shared_window": backend_tpm,
            "ttft_ms": describe([float(row["ttft_ms"]) for row in backend_valid]),
            "tpot_ms": describe([float(row["tpot_ms"]) for row in backend_valid]),
        }
    if len(backend_tpms) >= 2 and statistics.fmean(backend_tpms):
        backend_imbalance = (max(backend_tpms) - min(backend_tpms)) / statistics.fmean(backend_tpms)
    else:
        backend_imbalance = None

    result = {
        "schema_version": 1,
        "run_label": args.run_label,
        "concurrency": args.concurrency,
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "raw_files": [{"path": str(path), "sha256": sha256(path)} for path in raw_paths],
        "integrity": {
            "pass": integrity_pass,
            "expected_requests": len(manifest),
            "actual_requests": len(records),
            "raw_worker_files": len(raw_paths),
            "duplicate_positions": duplicate_positions[:100],
            "missing_positions": missing_positions[:100],
            "unexpected_positions": unexpected_positions[:100],
            "manifest_mismatches": mismatches[:100],
            "expected_thinking": expected_thinking,
            "actual_thinking": actual_thinking,
            "thinking_ratio": actual_thinking / len(records) if records else 0.0,
            "cached_tokens_total": cached_total,
        },
        "overall": overall,
        "thinking": segment([row for row in records if row.get("thinking")]),
        "non_thinking": segment([row for row in records if not row.get("thinking")]),
        "worker_attempts": {
            str(index): sum(int(row["worker_index"]) == index for row in records) for index in range(8)
        },
        "backend_balance": {
            "backends": backend_balance,
            "tpm_imbalance": backend_imbalance,
            "target_lt_5pct": backend_imbalance is not None and backend_imbalance < 0.05,
        },
        "errors": errors,
        "mean_slo_legal": mean_slo_legal,
        "p95_slo_legal": p95_slo_legal,
    }
    summary_path = args.result_dir / "summary.json"
    summary_path.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    def value(path: str) -> float:
        current: Any = result
        for part in path.split("."):
            current = current[part]
        return float(current)

    markdown = f"""# Strict run summary: {args.run_label}

- Concurrency: {args.concurrency}
- Requests: {overall['successful']}/{overall['attempted']} valid ({overall['success_rate']:.4%})
- Measurement: {overall['measurement_seconds']:.3f} s
- Total TPM: {overall['total_tpm']:,.3f}
- Input TPM: {overall['input_tpm']:,.3f}
- Output TPM: {overall['output_tpm']:,.3f}
- TTFT mean / p95 / p99: {value('overall.ttft_ms.mean'):.3f} / {value('overall.ttft_ms.p95'):.3f} / {value('overall.ttft_ms.p99'):.3f} ms
- TPOT mean / p95 / p99: {value('overall.tpot_ms.mean'):.6f} / {value('overall.tpot_ms.p95'):.6f} / {value('overall.tpot_ms.p99'):.6f} ms
- Mean-SLO legal: {mean_slo_legal}
- P95-SLO legal: {p95_slo_legal}
- Integrity: {integrity_pass}
- Thinking: {actual_thinking}/{len(records)} ({actual_thinking / len(records) if records else 0:.4%})
- Cached tokens: {cached_total}
"""
    (args.result_dir / "summary.md").write_text(markdown, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if integrity_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
