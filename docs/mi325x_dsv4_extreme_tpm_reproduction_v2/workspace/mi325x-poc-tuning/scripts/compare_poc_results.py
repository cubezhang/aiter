#!/usr/bin/env python3
"""Compare machine-readable MI325X POC results with quality and stability gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


FROZEN_FIELDS = (
    "workload_type", "host_fingerprint", "cluster_fingerprint", "gpu_model",
    "physical_gpu_count", "visible_device_count", "gfx_version", "partition_mode",
    "memory_partition_mode", "rocm_version", "driver_version", "framework_or_engine",
    "framework_or_engine_version", "container_digest", "artifact_id", "artifact_revision",
    "dataset_revision", "precision", "checkpoint_storage_format", "weight_compute_format",
    "activation_compute_format", "cache_or_gradient_format", "collective_quantization",
    "tensor_parallel", "pipeline_parallel", "data_parallel", "expert_parallel",
    "context_parallel", "workload_fingerprint",
)

HIGHER_HINTS = (
    "throughput", "goodput", "tokens_per_second", "tokens_per_gpu", "samples_per_second",
    "images_per_second", "bandwidth", "gflops", "tflops", "mfu", "efficiency",
    "successful_requests",
)
LOWER_HINTS = (
    "_ms", "latency", "step_time", "time_per", "failed_", "errors", "resets",
    "energy_per", "power_avg", "temperature_max",
)


def load(path: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: top-level JSON must be an object")
    for field in ("metadata", "correctness", "stability", "metrics"):
        if not isinstance(value.get(field), dict):
            raise ValueError(f"{path}: missing object {field}")
    return value


def direction(name: str) -> str:
    lowered = name.lower()
    if any(hint in lowered for hint in HIGHER_HINTS):
        return "higher"
    if any(hint in lowered for hint in LOWER_HINTS):
        return "lower"
    return "neutral"


def improvement(old: float, new: float, way: str) -> float | None:
    if old == 0:
        return None
    raw = (new - old) / abs(old) * 100.0
    return round(-raw if way == "lower" else raw, 4)


def objective_match(name: str, objective: str) -> bool:
    lowered = name.lower()
    if objective == "throughput":
        return any(item in lowered for item in ("throughput", "tokens_per", "samples_per", "images_per", "gflops", "tflops"))
    if objective == "latency":
        return "_ms" in lowered or "latency" in lowered or "step_time" in lowered or "time_per" in lowered
    if objective == "efficiency":
        return "efficiency" in lowered or "mfu" in lowered or "energy_per" in lowered
    if objective == "scaling":
        return "scaling" in lowered or "throughput" in lowered or "tokens_per" in lowered
    return direction(name) != "neutral" and not any(item in lowered for item in ("power", "temperature", "memory"))


def compare(
    baseline: dict[str, Any], candidate: dict[str, Any], objective: str,
    max_regression_pct: float, min_improvement_pct: float,
    allow_metadata_mismatch: bool,
) -> dict[str, Any]:
    changes = {str(item) for item in candidate["metadata"].get("changes", [])}
    mismatches = []
    for field in FROZEN_FIELDS:
        old = baseline["metadata"].get(field)
        new = candidate["metadata"].get(field)
        if old != new:
            mismatches.append({"field": field, "baseline": old, "candidate": new, "declared": field in changes})
    undeclared = [row for row in mismatches if not row["declared"]]

    common = sorted(set(baseline["metrics"]) & set(candidate["metrics"]))
    rows = []
    regressions = []
    improvements = []
    for name in common:
        old, new = baseline["metrics"][name], candidate["metrics"][name]
        if isinstance(old, bool) or isinstance(new, bool) or not isinstance(old, (int, float)) or not isinstance(new, (int, float)):
            continue
        way = direction(name)
        delta = improvement(float(old), float(new), way)
        rows.append({"metric": name, "baseline": old, "candidate": new, "direction": way, "improvement_pct": delta})
        if delta is None or way == "neutral":
            continue
        core = objective_match(name, objective) or "failed" in name.lower() or "errors" in name.lower()
        if core and delta < -max_regression_pct:
            regressions.append(name)
        if objective_match(name, objective) and delta >= min_improvement_pct:
            improvements.append(name)

    correctness_ok = baseline["correctness"].get("passed") is True and candidate["correctness"].get("passed") is True
    stability_ok = baseline["stability"].get("passed") is True and candidate["stability"].get("passed") is True
    preflight_ok = candidate["metadata"].get("kernel_preflight_status") == "PASS"
    evidence_ok = candidate["metadata"].get("evidence_level") != "UNSUPPORTED_PORT"
    metadata_ok = not undeclared or allow_metadata_mismatch
    accepted = all((correctness_ok, stability_ok, preflight_ok, evidence_ok, metadata_ok, not regressions, bool(improvements)))

    if not correctness_ok:
        recommendation = "REJECT: correctness failed"
    elif not stability_ok:
        recommendation = "REJECT: stability/RAS gate failed"
    elif not preflight_ok:
        recommendation = "REJECT: kernel/ABI preflight is not PASS"
    elif not evidence_ok:
        recommendation = "REJECT: unsupported portability path"
    elif undeclared and not allow_metadata_mismatch:
        recommendation = "INCONCLUSIVE: undeclared frozen metadata differs"
    elif regressions:
        recommendation = "REJECT: blocking metric regression"
    elif not improvements:
        recommendation = "INCONCLUSIVE: objective improvement below threshold"
    else:
        recommendation = "ACCEPT: objective improved with all gates passing"

    return {
        "accepted": accepted, "recommendation": recommendation, "objective": objective,
        "thresholds": {"max_regression_pct": max_regression_pct, "min_improvement_pct": min_improvement_pct},
        "correctness_ok": correctness_ok, "stability_ok": stability_ok,
        "kernel_preflight_ok": preflight_ok, "evidence_ok": evidence_ok,
        "metadata_mismatches": mismatches, "undeclared_metadata_mismatches": undeclared,
        "blocking_regressions": regressions, "objective_improvements": improvements,
        "metrics": rows,
        "review_required": ["Inspect raw trials and variance", "Confirm client/profiler/system load", "Confirm quality and RAS evidence"],
    }


def self_test() -> int:
    metadata = {field: None for field in FROZEN_FIELDS}
    metadata.update({
        "workload_type": "inference", "host_fingerprint": "host", "gpu_model": "MI325X",
        "physical_gpu_count": 8, "visible_device_count": 8, "gfx_version": "gfx942",
        "partition_mode": "SPX", "memory_partition_mode": "NPS1", "container_digest": "sha256:test",
        "artifact_id": "model", "artifact_revision": "abc", "workload_fingerprint": "shape",
        "kernel_preflight_status": "PASS", "evidence_level": "MI325X_MEASURED", "changes": [],
    })
    baseline = {"metadata": metadata, "correctness": {"passed": True}, "stability": {"passed": True}, "metrics": {"output_tokens_per_second": 1000, "ttft_p99_ms": 100}}
    candidate = json.loads(json.dumps(baseline))
    candidate["metrics"].update({"output_tokens_per_second": 1080, "ttft_p99_ms": 102})
    result = compare(baseline, candidate, "balanced", 5, 2, False)
    if not result["accepted"]:
        print(json.dumps(result, indent=2))
        return 1
    broken = json.loads(json.dumps(candidate))
    broken["stability"]["passed"] = False
    if compare(baseline, broken, "balanced", 5, 2, False)["accepted"]:
        return 1
    print("self-test: PASS")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline")
    p.add_argument("--candidate")
    p.add_argument("--objective", choices=("throughput", "latency", "efficiency", "scaling", "balanced"), default="balanced")
    p.add_argument("--max-regression-pct", type=float, default=5.0)
    p.add_argument("--min-improvement-pct", type=float, default=2.0)
    p.add_argument("--allow-metadata-mismatch", action="store_true")
    p.add_argument("--output")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        return self_test()
    if not args.baseline or not args.candidate:
        p.error("--baseline and --candidate are required unless --self-test")
    if args.max_regression_pct < 0 or args.min_improvement_pct < 0:
        p.error("percentage thresholds must be non-negative")
    try:
        result = compare(load(args.baseline), load(args.candidate), args.objective, args.max_regression_pct, args.min_improvement_pct, args.allow_metadata_mismatch)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    rendered = json.dumps(result, indent=2)
    if args.output:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    sys.exit(main())
