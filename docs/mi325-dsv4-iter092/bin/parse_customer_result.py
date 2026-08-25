#!/usr/bin/env python3
"""Parse the first complete CUSTOMER_LOCUST_LONGRUN_V1 scoring window."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path


HEADER_RE = re.compile(
    r"历史均值--5x\s*\((\d+)\s*次统计--统计间隔\s*(\d+)\)"
)
SCORE_RE = re.compile(
    r"Total_TPM\s+([0-9.]+)\s*w\s*\|\s*"
    r"Input_TPM\s+([0-9.]+)\s*w\s*\|\s*"
    r"Output_TPM\s+([0-9.]+)\s*w\s*\|\s*"
    r"TTFT\s+([0-9.]+)\s*ms\s*\|\s*"
    r"TPOT\s+([0-9.]+)\s*ms\s*\|\s*"
    r"E2E\s+([0-9.]+)\s*s\s*\|\s*"
    r"Input_tokens\s+([0-9.]+)\s*\|\s*"
    r"Output_tokens\s+([0-9.]+)"
)
TIMESTAMP_RE = re.compile(r"^\[([^]]+)\]")


def first_score(lines: list[str], concurrency: int, blocks: int) -> tuple[dict | None, int]:
    max_complete = 0
    for index, line in enumerate(lines):
        header = HEADER_RE.search(line)
        if not header:
            continue
        count, interval = map(int, header.groups())
        if interval == concurrency:
            max_complete = max(max_complete, count)
        if count != blocks or interval != concurrency:
            continue
        for score_index in range(index + 1, min(index + 8, len(lines))):
            score_match = SCORE_RE.search(lines[score_index])
            if not score_match:
                continue
            values = list(map(float, score_match.groups()))
            timestamp = TIMESTAMP_RE.search(line)
            return {
                "history_count": count,
                "interval": interval,
                "header_line_number": index + 1,
                "score_line_number": score_index + 1,
                "logged_at": timestamp.group(1) if timestamp else None,
                "total_tpm_w": values[0],
                "input_tpm_w": values[1],
                "output_tpm_w": values[2],
                "ttft_ms": values[3],
                "tpot_ms": values[4],
                "e2e_s": values[5],
                "mean_input_tokens": values[6],
                "mean_output_tokens": values[7],
            }, max_complete
    return None, max_complete


def aggregate_stats(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(newline="", errors="replace") as handle:
        for row in csv.DictReader(handle):
            if row.get("Name") == "Aggregated":
                return {
                    "request_count": int(float(row.get("Request Count") or 0)),
                    "failure_count": int(float(row.get("Failure Count") or 0)),
                    "requests_per_second": float(row.get("Requests/s") or 0),
                    "median_response_time_ms": float(row.get("Median Response Time") or 0),
                    "p99_response_time_ms": float(row.get("99%") or 0),
                }
    return {}


def no_bad_pages(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(errors="replace"))
    except (json.JSONDecodeError, OSError):
        return False
    if not isinstance(payload, list) or len(payload) != 8:
        return False
    keys = ("retired", "pending", "un_res")
    return all(
        all(item.get(key) == "No bad pages found." for key in keys)
        for item in payload
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--blocks", type=int, default=20)
    parser.add_argument("--tpot-limit-ms", type=float)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    metadata_path = run_dir / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
    concurrency = args.concurrency or int(metadata.get("concurrency", 0))
    blocks = args.blocks or int(metadata.get("score_blocks", 20))
    tpot_limit_ms = (
        args.tpot_limit_ms
        if args.tpot_limit_ms is not None
        else metadata.get("tpot_limit_ms")
    )
    if concurrency <= 0:
        parser.error("concurrency is required in metadata.json or --concurrency")

    log_path = run_dir / "locust.log"
    lines = log_path.read_text(errors="replace").splitlines() if log_path.exists() else []
    score, max_complete = first_score(lines, concurrency, blocks)
    if args.probe:
        if score:
            print(json.dumps(score, ensure_ascii=False, sort_keys=True))
            return 0
        return 2

    stats = aggregate_stats(run_dir / "locust_stats.csv")
    ras_before = run_dir / "ras_before.txt"
    ras_after = run_dir / "ras_after.txt"
    ras_unchanged = (
        ras_before.exists()
        and ras_after.exists()
        and ras_before.read_bytes() == ras_after.read_bytes()
    )
    kernel_fault_path = run_dir / "kernel_gpu_faults.txt"
    kernel_fault_lines = []
    if kernel_fault_path.exists():
        kernel_fault_lines = [line for line in kernel_fault_path.read_text(errors="replace").splitlines() if line.strip()]

    failures = stats.get("failure_count")
    integrity_valid = bool(
        score
        and failures == 0
        and ras_unchanged
        and no_bad_pages(run_dir / "bad_pages_before.json")
        and no_bad_pages(run_dir / "bad_pages_after.json")
        and not kernel_fault_lines
    )
    slo_pass = bool(
        score
        and tpot_limit_ms is not None
        and score["tpot_ms"] <= float(tpot_limit_ms)
    )
    eligible = bool(integrity_valid and slo_pass)
    result = {
        "schema_version": 1,
        "contract_id": metadata.get("contract_id", "CUSTOMER_LOCUST_LONGRUN_V1"),
        "run_dir": str(run_dir),
        "label": metadata.get("label"),
        "wave": metadata.get("wave"),
        "service_profile": metadata.get("service_profile"),
        "concurrency": concurrency,
        "score_blocks_required": blocks,
        "scored_complete_requests": blocks * 5 * concurrency,
        "max_observed_complete_5x_blocks": max_complete,
        "first_complete_score": score,
        "locust_aggregate_after_graceful_stop": stats,
        "ras_unchanged": ras_unchanged,
        "bad_pages_clear_before": no_bad_pages(run_dir / "bad_pages_before.json"),
        "bad_pages_clear_after": no_bad_pages(run_dir / "bad_pages_after.json"),
        "kernel_gpu_fault_count": len(kernel_fault_lines),
        "valid": integrity_valid,
        "integrity_valid": integrity_valid,
        "tpot_limit_ms": tpot_limit_ms,
        "tpot_slo_pass": slo_pass,
        "eligible": eligible,
        "low_latency_23ms": bool(score and score["tpot_ms"] <= 23.0),
        "strict_result_used": False,
        "partial_tail_and_final_drain_scored": False,
    }
    (run_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if integrity_valid else 3


if __name__ == "__main__":
    sys.exit(main())
