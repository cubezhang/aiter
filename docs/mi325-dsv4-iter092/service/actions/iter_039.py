#!/usr/bin/env python3
"""Iteration 39: pinned 20260816 image baseline on the closest C34 profile."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import traceback
from typing import Any

from common import (
    CAMPAIGN,
    ROOT,
    WORK,
    atomic_json,
    read_json,
    reapply_retained_host_rules,
    run_point,
    start_profile,
)
from iter_030 import PROFILE_ROOT, atomic_text


C34_PROFILE = "original_graph33_40_m102_a8w8nonsquare"
C34_TPOT_MS = 21.60
IMAGE = "rocm/atom-dev@sha256:9939480d4ff9230b09e288af343b38e1e02a2fc16e0a2522b2cc2297df9f2f59"
IMAGE_ID = "sha256:9939480d4ff9230b09e288af343b38e1e02a2fc16e0a2522b2cc2297df9f2f59"
CANDIDATE_PROFILE = f"{C34_PROFILE}_latestimage"


def ensure_candidate_profile() -> str:
    parent = PROFILE_ROOT / f"{C34_PROFILE}.env"
    candidate = PROFILE_ROOT / f"{CANDIDATE_PROFILE}.env"
    if not parent.is_file():
        raise RuntimeError(f"C34 parent profile is missing: {parent}")
    content = "\n".join(
        [
            f"source {parent}",
            f"PROFILE_ID={CANDIDATE_PROFILE}",
            "PROFILE_DESCRIPTION=C34_seeded_pinned_20260816_ATOM_AITER_image_baseline",
            f"CANDIDATE_ATOM_IMAGE={IMAGE}",
            f"CANDIDATE_ATOM_IMAGE_ID={IMAGE_ID}",
            "CANDIDATE_FMOE_CONFIG=",
        ]
    ) + "\n"
    if candidate.exists() and candidate.read_text() != content:
        raise RuntimeError(f"refusing non-identical latest-image profile: {candidate}")
    if not candidate.exists():
        atomic_text(candidate, content)
    return CANDIDATE_PROFILE


def verify_runtime(candidate: str) -> dict[str, Any]:
    active = read_json(CAMPAIGN / "state/active_service.json")
    if (
        active.get("profile_id") != candidate
        or active.get("atom_image_id") != IMAGE_ID
        or active.get("fmoe_config") not in (None, "")
    ):
        raise RuntimeError("latest-image baseline runtime provenance mismatch")
    container = str(active["service_container"])
    head = subprocess.run(
        ["docker", "exec", container, "git", "-C", "/app/aiter-test", "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    if head != "878d60d779fa774496fe075175d0e99f1d49eea9":
        raise RuntimeError(f"unexpected AITER commit in latest-image baseline: {head}")
    return {"active_service": active, "aiter_commit": head}


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    previous_id = int(os.environ["TPOT_PREVIOUS_ITERATION"])
    current_id = int(os.environ["TPOT_ITERATION_ID"])
    previous = read_json(ROOT / f"config_snapshots/iter_{previous_id:03d}.json")
    previous_spec = previous["restore_spec"]
    previous_profile = str(previous_spec["service_profile"])
    candidate = ensure_candidate_profile()
    runtime: dict[str, Any] | None = None
    warmup: dict[str, Any] | None = None
    endpoint_probe: dict[str, Any] | None = None
    screen: dict[str, Any] | None = None
    formal: dict[str, Any] | None = None
    failure: dict[str, Any] | None = None

    try:
        start_profile(candidate)
        reapply_retained_host_rules(previous_spec)
        runtime = verify_runtime(candidate)
        warmup = run_point(candidate, "latestimage_warmup", "tpot_phase8_latest_image", 1)
        endpoint_probe = run_point(candidate, "latestimage_probe", "tpot_phase8_latest_image", 1)
        probe_pass = bool(
            endpoint_probe["valid"]
            and endpoint_probe["total_tpm_w"] >= 132.0
            and endpoint_probe["ttft_ms"] <= 30.0
            and endpoint_probe["measured_tpot"] <= 22.0
        )
        if probe_pass:
            screen = run_point(candidate, "latestimage_screen", "tpot_phase8_latest_image", 3)
        formal_gate = bool(
            screen
            and screen["valid"]
            and screen["total_tpm_w"] >= 134.5
            and screen["ttft_ms"] <= 30.0
            and screen["measured_tpot"] <= 20.8
        )
        if formal_gate:
            formal = run_point(candidate, "latestimage_formal20", "tpot_phase8_latest_image", 20)
        measurement = formal or screen or endpoint_probe
    except Exception as exc:
        failure = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        start_profile(previous_profile)
        reapply_retained_host_rules(previous_spec)
        measurement = run_point(
            previous_profile,
            "latestimage_failure_control",
            "tpot_phase8_latest_image_failure_control",
            1,
        )
        measurement["control_valid"] = bool(measurement["valid"])
        measurement["valid"] = False

    retain = bool(
        failure is None
        and measurement["score_blocks"] >= 3
        and measurement["valid"]
        and measurement["total_tpm_w"] >= 135.0
        and measurement["ttft_ms"] <= 30.0
        and measurement["measured_tpot"] < C34_TPOT_MS
    )
    desired = candidate if retain else previous_profile
    if read_json(CAMPAIGN / "state/active_service.json").get("profile_id") != desired:
        start_profile(desired)
    reapply_retained_host_rules(previous_spec)
    measurement["promotion_eligible"] = retain

    evidence = {
        "schema_version": 1,
        "hypothesis": "the pinned newer ATOM/AITER image improves C34 through aggregate kernel/runtime updates without a private table",
        "parent_profile": C34_PROFILE,
        "parent_measured_tpot_ms": C34_TPOT_MS,
        "previous_profile": previous_profile,
        "candidate_profile": candidate,
        "runtime": runtime,
        "warmup": warmup,
        "endpoint_probe": endpoint_probe,
        "screen3": screen,
        "formal20": formal,
        "failure": failure,
        "promotion_eligible": retain,
        "retained_profile": desired,
    }
    evidence_path = WORK / "latest_image_campaign.json"
    atomic_json(evidence_path, evidence)
    measurement["latest_image_evidence"] = str(evidence_path)
    restore_spec = copy.deepcopy(previous_spec)
    restore_spec["service_profile"] = desired
    restore_spec[f"iteration_{current_id}"] = {
        "latest_image_evidence": str(evidence_path),
        "promotion_eligible": retain,
    }
    atomic_json(WORK / "restore_spec.json", restore_spec)
    atomic_json(WORK / "measurement_result.json", measurement)
    print(f"MEASUREMENT_RESULT {json.dumps(measurement, sort_keys=True)}", flush=True)


if __name__ == "__main__":
    main()
