#!/usr/bin/env python3
"""Iteration 42: retry the pinned newer image with a guarded cross-image transaction."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import time
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
from iter_039 import C34_PROFILE, C34_TPOT_MS, IMAGE, IMAGE_ID, verify_runtime


PORTS = (18000, 18001, 18080)


def ensure_candidate_profile(parent_profile: str) -> str:
    parent = PROFILE_ROOT / f"{parent_profile}.env"
    candidate_profile = f"{parent_profile}_latestimage_retry"
    candidate = PROFILE_ROOT / f"{candidate_profile}.env"
    if not parent.is_file():
        raise RuntimeError(f"latest-image retry parent profile missing: {parent}")
    content = "\n".join(
        [
            f"source {parent}",
            f"PROFILE_ID={candidate_profile}",
            "PROFILE_DESCRIPTION=guarded_pinned_20260816_AITER_runtime_image_retry",
            f"CANDIDATE_ATOM_IMAGE={IMAGE}",
            f"CANDIDATE_ATOM_IMAGE_ID={IMAGE_ID}",
            "CANDIDATE_FMOE_CONFIG=",
        ]
    ) + "\n"
    if candidate.exists() and candidate.read_text() != content:
        raise RuntimeError(f"refusing non-identical latest-image retry profile: {candidate}")
    if not candidate.exists():
        atomic_text(candidate, content)
    return candidate_profile


def occupied_ports() -> list[int]:
    occupied: list[int] = []
    for port in PORTS:
        result = subprocess.run(
            ["ss", "-ltnH", f"sport = :{port}"],
            text=True,
            capture_output=True,
            check=True,
        )
        if result.stdout.strip():
            occupied.append(port)
    return occupied


def exact_active_pair() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    state_path = CAMPAIGN / "state/active_service.json"
    if not state_path.is_file():
        listeners: dict[int, str] = {}
        for port in occupied_ports():
            result = subprocess.run(
                ["ss", "-ltnpH", f"sport = :{port}"],
                text=True,
                capture_output=True,
                check=False,
            )
            listeners[port] = result.stdout.strip()
        docker_ps = subprocess.run(
            [
                "docker",
                "ps",
                "--no-trunc",
                "--format",
                "{{.ID}}\t{{.Names}}\t{{.Image}}\t{{.Status}}",
            ],
            text=True,
            capture_output=True,
            check=False,
        ).stdout.strip()
        raise RuntimeError(
            "service ports are occupied but state/active_service.json is missing; "
            "refusing to stop unknown listeners. "
            f"listeners={listeners!r} running_containers={docker_ps!r}"
        )
    active = read_json(state_path)
    names = [str(active["router_container"]), str(active["service_container"])]
    inspected = json.loads(
        subprocess.run(
            ["docker", "inspect", *names],
            text=True,
            capture_output=True,
            check=True,
        ).stdout
    )
    expected_ids = {
        str(active["router_container"]): str(active["router_container_id"]),
        str(active["service_container"]): str(active["service_container_id"]),
    }
    actual_ids = {str(item["Name"]).lstrip("/"): str(item["Id"]) for item in inspected}
    if actual_ids != expected_ids:
        raise RuntimeError(f"active container identity mismatch: actual={actual_ids} expected={expected_ids}")
    if not all(bool(item["State"]["Running"]) for item in inspected):
        raise RuntimeError("active service pair is not fully running")
    return active, inspected


def stop_exact_active_pair(evidence_name: str) -> dict[str, Any]:
    active, inspected = exact_active_pair()
    evidence = {
        "active": active,
        "inspect": inspected,
        "occupied_ports_before": occupied_ports(),
    }
    atomic_json(WORK / f"{evidence_name}_before.json", evidence)
    names = [str(active["router_container"]), str(active["service_container"])]
    subprocess.run(["docker", "stop", *names], text=True, check=True)
    subprocess.run(["docker", "rm", *names], text=True, check=True)
    for _ in range(24):
        if not occupied_ports():
            break
        time.sleep(0.5)
    remaining = occupied_ports()
    if remaining:
        raise RuntimeError(f"ports remain occupied after exact pair stop: {remaining}")
    return evidence


def transition_profile(profile: str, desired_image_id: str, evidence_name: str) -> dict[str, Any]:
    transition: dict[str, Any] = {"target_profile": profile, "target_image_id": desired_image_id}
    ports = occupied_ports()
    if ports:
        active, _ = exact_active_pair()
        transition["source_profile"] = active.get("profile_id")
        transition["source_image_id"] = active.get("atom_image_id")
        if active.get("atom_image_id") != desired_image_id:
            transition["cross_image_stop"] = stop_exact_active_pair(evidence_name)
    start_profile(profile)
    active = read_json(CAMPAIGN / "state/active_service.json")
    if active.get("profile_id") != profile or active.get("atom_image_id") != desired_image_id:
        raise RuntimeError("post-transition profile/image provenance mismatch")
    transition["result"] = active
    atomic_json(WORK / f"{evidence_name}_transaction.json", transition)
    return transition


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    previous_id = int(os.environ["TPOT_PREVIOUS_ITERATION"])
    current_id = int(os.environ["TPOT_ITERATION_ID"])
    previous = read_json(ROOT / f"config_snapshots/iter_{previous_id:03d}.json")
    previous_spec = previous["restore_spec"]
    previous_profile = str(previous_spec["service_profile"])
    previous_active = read_json(CAMPAIGN / "state/active_service.json")
    previous_image_id = str(previous_active["atom_image_id"])
    parent_profile = previous_profile if previous_profile.endswith("_densecg1_64") else C34_PROFILE
    candidate = ensure_candidate_profile(parent_profile)
    state = read_json(ROOT / "test_job_state.json")
    accepted_tpots = [
        float(item["measured_tpot"])
        for item in state.get("completed_results", [])
        if item.get("accepted")
    ]
    parent_best = min([C34_TPOT_MS, *accepted_tpots])

    transaction: dict[str, Any] | None = None
    runtime: dict[str, Any] | None = None
    warmup: dict[str, Any] | None = None
    endpoint_probe: dict[str, Any] | None = None
    screen: dict[str, Any] | None = None
    formal: dict[str, Any] | None = None
    failure: dict[str, Any] | None = None

    try:
        transaction = transition_profile(candidate, IMAGE_ID, "to_newer_image")
        reapply_retained_host_rules(previous_spec)
        runtime = verify_runtime(candidate)
        warmup = run_point(candidate, "latestimage_retry_warmup", "tpot_phase8_latest_image_retry", 1)
        endpoint_probe = run_point(candidate, "latestimage_retry_probe", "tpot_phase8_latest_image_retry", 1)
        probe_pass = bool(
            endpoint_probe["valid"]
            and endpoint_probe["total_tpm_w"] >= 115.0
            and endpoint_probe["ttft_ms"] <= 45.0
            and endpoint_probe["measured_tpot"] <= 40.0
        )
        if probe_pass:
            screen = run_point(candidate, "latestimage_retry_screen", "tpot_phase8_latest_image_retry", 3)
        formal_gate = bool(
            screen
            and screen["valid"]
            and screen["total_tpm_w"] >= 134.5
            and screen["ttft_ms"] <= 30.0
            and screen["measured_tpot"] <= 21.2
        )
        if formal_gate:
            formal = run_point(candidate, "latestimage_retry_formal20", "tpot_phase8_latest_image_retry", 20)
        measurement = formal or screen or endpoint_probe
    except Exception as exc:
        failure = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        try:
            transition_profile(previous_profile, previous_image_id, "failure_restore_previous")
            reapply_retained_host_rules(previous_spec)
            measurement = run_point(
                previous_profile,
                "latestimage_retry_failure_control",
                "tpot_phase8_latest_image_retry_failure_control",
                1,
            )
            measurement["control_valid"] = bool(measurement["valid"])
        except Exception as restore_exc:
            failure["restore_failure"] = {
                "type": type(restore_exc).__name__,
                "message": str(restore_exc),
                "traceback": traceback.format_exc(),
            }
            raise
        measurement["valid"] = False

    retain = bool(
        failure is None
        and measurement["score_blocks"] >= 3
        and measurement["valid"]
        and measurement["total_tpm_w"] >= 135.0
        and measurement["ttft_ms"] <= 30.0
        and measurement["measured_tpot"] < parent_best
    )
    desired_profile = candidate if retain else previous_profile
    desired_image_id = IMAGE_ID if retain else previous_image_id
    active = read_json(CAMPAIGN / "state/active_service.json")
    if active.get("profile_id") != desired_profile or active.get("atom_image_id") != desired_image_id:
        transition_profile(desired_profile, desired_image_id, "final_retention_transition")
    reapply_retained_host_rules(previous_spec)
    measurement["promotion_eligible"] = retain

    evidence = {
        "schema_version": 1,
        "hypothesis": "newer pinned AITER/runtime image improves C34 once the cross-image launcher guard is handled transactionally",
        "previous_profile": previous_profile,
        "previous_image_id": previous_image_id,
        "parent_profile": parent_profile,
        "parent_best_tpot_ms": parent_best,
        "candidate_profile": candidate,
        "candidate_image_id": IMAGE_ID,
        "transaction": transaction,
        "runtime": runtime,
        "warmup": warmup,
        "endpoint_probe": endpoint_probe,
        "screen3": screen,
        "formal20": formal,
        "failure": failure,
        "promotion_eligible": retain,
        "retained_profile": desired_profile,
    }
    evidence_path = WORK / "latest_image_retry_campaign.json"
    atomic_json(evidence_path, evidence)
    measurement["latest_image_retry_evidence"] = str(evidence_path)
    restore_spec = copy.deepcopy(previous_spec)
    restore_spec["service_profile"] = desired_profile
    restore_spec[f"iteration_{current_id}"] = {
        "latest_image_retry_evidence": str(evidence_path),
        "promotion_eligible": retain,
    }
    atomic_json(WORK / "restore_spec.json", restore_spec)
    atomic_json(WORK / "measurement_result.json", measurement)
    print(f"MEASUREMENT_RESULT {json.dumps(measurement, sort_keys=True)}", flush=True)


if __name__ == "__main__":
    main()
