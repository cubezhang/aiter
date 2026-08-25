#!/usr/bin/env python3
"""Iteration 30: tune all six exact-M102 BF16 projection fallbacks."""

from __future__ import annotations

import copy
import datetime as dt
import json
import os
import pathlib
import re
import subprocess
import tempfile
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


TUNING_ROOT = pathlib.Path("/data/models/mi325_dsv4_perf_tuning_20260803_065439")
EXPERIMENTS = TUNING_ROOT / "experiments"
PROFILE_ROOT = CAMPAIGN / "config/profiles"
INPUT = EXPERIMENTS / "codex_tpot20_m102_bf16_projection_input.csv"
BASE_BF16 = pathlib.Path(
    "/data/models/mi325_dsv4_reuse_tuning_20260801_103501/02_source/aiter-test/aiter/configs/bf16_tuned_gemm.csv"
)
FINAL_BF16 = EXPERIMENTS / "codex_tpot20_m102_bf16_projection_overlay_v1.csv"
IMAGE = "rocm/atom-dev@sha256:66df2fb1c537f52d47f2d3bf973b8da15278dd2ee362b5307e3d29dfad69fa7d"
REQUIRED = {
    ("102", "64", "4096"),
    ("102", "256", "4096"),
    ("102", "512", "4096"),
    ("102", "1024", "4096"),
    ("102", "2048", "4096"),
    ("102", "32320", "4096"),
}


def atomic_text(path: pathlib.Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def selected_rows(path: pathlib.Path) -> list[str]:
    selected: list[str] = []
    for line in path.read_text().splitlines()[1:]:
        fields = line.split(",")
        if (
            len(fields) >= 18
            and fields[0] == "gfx942"
            and fields[1] == "304"
            and tuple(fields[2:5]) in REQUIRED
            and fields[5:10]
            == ["False", "torch.bfloat16", "torch.bfloat16", "False", "False"]
            and float(fields[15]) <= 0.01
        ):
            selected.append(line)
    return selected


def config_keys(path: pathlib.Path) -> list[tuple[str, ...]]:
    keys: list[tuple[str, ...]] = []
    for line in path.read_text().splitlines()[1:]:
        fields = line.split(",")
        if len(fields) >= 10:
            keys.append(tuple(fields[:10]))
    return keys


def valid_overlay(path: pathlib.Path) -> bool:
    if not path.is_file():
        return False
    keys = config_keys(path)
    if len(keys) != len(set(keys)):
        return False
    counts = {shape: 0 for shape in REQUIRED}
    for key in keys:
        if (
            key[0] == "gfx942"
            and key[1] == "304"
            and tuple(key[2:5]) in REQUIRED
            and list(key[5:10])
            == ["False", "torch.bfloat16", "torch.bfloat16", "False", "False"]
        ):
            counts[tuple(key[2:5])] += 1
    return all(count == 1 for count in counts.values())


def stop_active_service_for_tuning(expected_profile: str, evidence_dir: pathlib.Path) -> None:
    active = read_json(CAMPAIGN / "state/active_service.json")
    if active.get("profile_id") != expected_profile:
        raise RuntimeError(f"BF16 tuner active profile mismatch: {active.get('profile_id')}")
    names = [str(active["router_container"]), str(active["service_container"])]
    ids = [str(active["router_container_id"]), str(active["service_container_id"])]
    for name, expected_id in zip(names, ids):
        actual = subprocess.run(
            ["docker", "inspect", name, "--format", "{{.Id}}"],
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        if actual != expected_id:
            raise RuntimeError(f"refusing to stop unexpected container {name}")
    inspect = subprocess.run(
        ["docker", "inspect", *names], text=True, capture_output=True, check=True
    )
    (evidence_dir / "previous_service_inspect.json").write_text(inspect.stdout)
    subprocess.run(["docker", "stop", *names], text=True, check=True)
    subprocess.run(["docker", "rm", *names], text=True, check=True)


def publish_overlay(tuned: pathlib.Path) -> None:
    rows = selected_rows(tuned)
    if len(rows) != 6 or {tuple(line.split(",")[2:5]) for line in rows} != REQUIRED:
        raise RuntimeError(f"BF16 tuner did not produce six strict rows: {tuned}")
    merged = "\n".join(BASE_BF16.read_text().splitlines() + rows) + "\n"
    if FINAL_BF16.exists():
        if FINAL_BF16.read_text() != merged:
            raise RuntimeError(f"refusing to overwrite non-identical BF16 overlay: {FINAL_BF16}")
    else:
        atomic_text(FINAL_BF16, merged)
    if not valid_overlay(FINAL_BF16):
        raise RuntimeError(f"published BF16 overlay failed validation: {FINAL_BF16}")


def run_tuner(previous_profile: str) -> dict[str, Any]:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    tune_dir = EXPERIMENTS / f"tpot_i030_m102_bf16_tune_{stamp}"
    tune_dir.mkdir(parents=True, exist_ok=False)
    tuned = tune_dir / "m102_bf16_tuned.csv"
    profile = tune_dir / "m102_bf16_profile.csv"
    stop_active_service_for_tuning(previous_profile, tune_dir)
    relative = tune_dir.relative_to(TUNING_ROOT)
    command = [
        "docker", "run", "--rm", "--name", f"tpot_i030_m102_bf16_tuner_{stamp}",
        "--ipc=host", "--shm-size=128G", "--device", "/dev/kfd", "--device", "/dev/dri",
        "--group-add", "video", "--mount", f"type=bind,src={TUNING_ROOT},dst=/work",
        IMAGE, "bash", "-lc",
        (
            "cd /app/aiter-test && /opt/venv/bin/python "
            "csrc/gemm_a16w16/gemm_a16w16_tune.py "
            "--input_file /work/experiments/codex_tpot20_m102_bf16_projection_input.csv "
            f"--tuned_file /work/{relative}/m102_bf16_tuned.csv "
            f"--profile_file /work/{relative}/m102_bf16_profile.csv "
            "--mp 8 --shape_grouped --libtype all --warmup 10 --iters 120 --errRatio 0.01"
        ),
    ]
    subprocess.run(command, text=True, check=True)
    publish_overlay(tuned)

    verify_logs: list[str] = []
    for repeat in (1, 2):
        verify_log = tune_dir / f"verify_{repeat}.log"
        verify_command = [
            "docker", "run", "--rm", "--name", f"tpot_i030_m102_bf16_v{repeat}_{stamp}",
            "--ipc=host", "--shm-size=128G", "--device", "/dev/kfd", "--device", "/dev/dri",
            "--group-add", "video", "--mount", f"type=bind,src={TUNING_ROOT},dst=/work",
            IMAGE, "bash", "-lc",
            (
                "cd /app/aiter-test && /opt/venv/bin/python "
                "csrc/gemm_a16w16/gemm_a16w16_tune.py "
                f"--run_config /work/{relative}/m102_bf16_tuned.csv "
                "--warmup 10 --iters 160 --errRatio 0.01"
            ),
        ]
        with verify_log.open("w", encoding="utf-8") as handle:
            result = subprocess.run(
                verify_command, text=True, stdout=handle, stderr=subprocess.STDOUT
            )
        if result.returncode:
            raise RuntimeError(f"BF16 repeat verification failed rc={result.returncode}: {verify_log}")
        text = verify_log.read_text(errors="replace")
        if len(re.findall(r"\|\s+OK\s*$", text, re.MULTILINE)) < 6:
            raise RuntimeError(f"BF16 verification markers missing: {verify_log}")
        if re.search(r"\|\s+(?:ERROR|MISMATCH)\s*$", text, re.MULTILINE | re.IGNORECASE):
            raise RuntimeError(f"BF16 verification reported failure: {verify_log}")
        verify_logs.append(str(verify_log))
    return {
        "reused": False,
        "tune_dir": str(tune_dir),
        "tuned": str(tuned),
        "profile": str(profile),
        "verify_logs": verify_logs,
        "final_bf16_overlay": str(FINAL_BF16),
        "strict_shape_rows": sorted(REQUIRED),
    }


def ensure_candidate_profile(previous_profile: str) -> str:
    candidate = f"{previous_profile}_bf16m102proj"
    previous_file = PROFILE_ROOT / f"{previous_profile}.env"
    candidate_file = PROFILE_ROOT / f"{candidate}.env"
    if not previous_file.is_file():
        raise RuntimeError(f"retained profile file is missing: {previous_file}")
    relative_overlay = FINAL_BF16.relative_to(TUNING_ROOT)
    content = (
        f"source {previous_file}\n"
        f"PROFILE_ID={candidate}\n"
        "PROFILE_DESCRIPTION=C68_exact_M102_BF16_projection_GEMMs\n"
        f"CANDIDATE_BF16_CONFIG=/work/{relative_overlay}\n"
        "CANDIDATE_REQUIRE_BF16_M102_PROJECTIONS=1\n"
    )
    if candidate_file.exists():
        if candidate_file.read_text() != content:
            raise RuntimeError(f"refusing to overwrite non-identical candidate profile: {candidate_file}")
    else:
        atomic_text(candidate_file, content)
    return candidate


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    previous_id = int(os.environ["TPOT_PREVIOUS_ITERATION"])
    current_id = int(os.environ["TPOT_ITERATION_ID"])
    previous = read_json(ROOT / f"config_snapshots/iter_{previous_id:03d}.json")
    previous_spec = previous["restore_spec"]
    previous_profile = str(previous_spec["service_profile"])
    previous_best = 21.75
    if previous.get("accepted_candidate"):
        previous_best = float(previous["measurement_result"]["measured_tpot"])

    try:
        if valid_overlay(FINAL_BF16):
            tuning: dict[str, Any] = {
                "reused": True,
                "final_bf16_overlay": str(FINAL_BF16),
                "strict_shape_rows": sorted(REQUIRED),
            }
        else:
            tuning = run_tuner(previous_profile)
        if not valid_overlay(FINAL_BF16):
            raise RuntimeError(f"required BF16 overlay is not valid: {FINAL_BF16}")
        candidate = ensure_candidate_profile(previous_profile)
        start_profile(candidate)
        reapply_retained_host_rules(previous_spec)
        active = read_json(CAMPAIGN / "state/active_service.json")
        expected_config = f"/work/{FINAL_BF16.relative_to(TUNING_ROOT)}"
        if active.get("profile_id") != candidate:
            raise RuntimeError(f"BF16 active profile mismatch: {active.get('profile_id')}")
        if active.get("bf16_config") != expected_config:
            raise RuntimeError(f"BF16 config provenance mismatch: {active.get('bf16_config')}")
        warmup = run_point(candidate, "bf16_m102_warmup", "tpot_phase7_bf16_m102", 1)
        warmup.update({"profile": candidate, "variant": "BF16_M102_PROJECTIONS", "stage": "warmup"})
        screen = run_point(candidate, "bf16_m102_screen", "tpot_phase7_bf16_m102", 5)
        screen.update({"profile": candidate, "variant": "BF16_M102_PROJECTIONS", "stage": "screen"})
        formal = None
        if (
            screen["valid"]
            and screen["total_tpm_w"] >= 134.0
            and screen["ttft_ms"] <= 30.0
            and screen["measured_tpot"] <= 21.25
        ):
            formal = run_point(candidate, "bf16_m102_formal20", "tpot_phase7_bf16_m102", 20)
            formal.update({"profile": candidate, "variant": "BF16_M102_PROJECTIONS", "stage": "formal20"})
        measurement = formal or screen
    except Exception:
        start_profile(previous_profile)
        reapply_retained_host_rules(previous_spec)
        raise

    retain = bool(
        measurement["valid"]
        and measurement["total_tpm_w"] >= 135.0
        and measurement["ttft_ms"] <= 30.0
        and measurement["measured_tpot"] < previous_best
    )
    desired = candidate if retain else previous_profile
    if read_json(CAMPAIGN / "state/active_service.json").get("profile_id") != desired:
        start_profile(desired)
    reapply_retained_host_rules(previous_spec)

    evidence = {
        "schema_version": 1,
        "hypothesis": "the exact M102 graph falls back for six BF16 projection shapes, including the N32320 vocab projection; tune them without replacing retained A8W8 or FMoE kernels",
        "previous_profile": previous_profile,
        "previous_best_tpot_ms": previous_best,
        "candidate_profile": candidate,
        "tuning": tuning,
        "warmup": warmup,
        "screen": screen,
        "formal20": formal,
        "retained_profile": desired,
    }
    evidence_path = WORK / "bf16_m102_projection_campaign.json"
    atomic_json(evidence_path, evidence)
    measurement["bf16_m102_projection_evidence"] = str(evidence_path)

    restore_spec = copy.deepcopy(previous_spec)
    restore_spec["service_profile"] = desired
    restore_spec[f"iteration_{current_id}"] = {
        "bf16_m102_projection_evidence": str(evidence_path)
    }
    atomic_json(WORK / "restore_spec.json", restore_spec)
    atomic_json(WORK / "measurement_result.json", measurement)
    print(f"MEASUREMENT_RESULT {json.dumps(measurement, sort_keys=True)}", flush=True)


if __name__ == "__main__":
    main()
