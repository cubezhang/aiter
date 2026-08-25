"""Durable helpers shared by terminal-independent TPOT tuning actions."""

from __future__ import annotations

import datetime as dt
import json
import os
import pathlib
import re
import subprocess
from typing import Any


ROOT = pathlib.Path(os.environ["TPOT_WORKFLOW_ROOT"])
CAMPAIGN = pathlib.Path(os.environ["TPOT_CAMPAIGN_ROOT"])
ITERATION = int(os.environ["TPOT_ITERATION_ID"])
WORK = ROOT / "work" / f"iter_{ITERATION:03d}"
RUN_POINT = CAMPAIGN / "bin/run_customer_point.sh"
START_PROFILE = CAMPAIGN / "bin/start_service_profile.sh"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def log(message: str) -> None:
    print(f"[{utc_now()}] {message}", flush=True)


def read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_optional_json(path: pathlib.Path) -> dict[str, Any] | None:
    """Return None only when a state file has never been created.

    Invalid or unreadable existing state remains a hard failure so service
    ownership is never guessed from incomplete provenance.
    """
    try:
        return read_json(path)
    except FileNotFoundError:
        return None


def atomic_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def start_profile(profile: str) -> None:
    command = [str(START_PROFILE), profile]
    log(f"COMMAND {json.dumps(command)}")
    subprocess.run(command, text=True, check=True)
    active = read_json(CAMPAIGN / "state/active_service.json")
    if active.get("profile_id") != profile:
        raise RuntimeError(f"active profile mismatch: {active.get('profile_id')} != {profile}")


def run_point(profile: str, suffix: str, phase: str, blocks: int) -> dict[str, Any]:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    label = f"tpot_i{ITERATION:03d}_{suffix}_c68"
    run_dir = CAMPAIGN / "runs" / f"{label}_attempt_1_{stamp}"
    command = [str(RUN_POINT), phase, label, profile, "68", "1", str(blocks), str(run_dir)]
    log(f"COMMAND {json.dumps(command)}")
    subprocess.run(command, text=True, check=True)
    result = read_json(run_dir / "result.json")
    score = result["first_complete_score"]
    return {
        "iteration_id": ITERATION,
        "measured_tpot": float(score["tpot_ms"]),
        "total_tpm_w": float(score["total_tpm_w"]),
        "ttft_ms": float(score["ttft_ms"]),
        "valid": bool(result["valid"]),
        "score_blocks": blocks,
        "source_result": str(run_dir / "result.json"),
    }


def backend_descendants(port: int) -> set[int]:
    output = subprocess.run(
        ["ps", "-eo", "pid=,ppid=,args="], text=True, capture_output=True, check=True
    ).stdout
    parents: dict[int, int] = {}
    roots: list[int] = []
    for line in output.splitlines():
        fields = line.strip().split(None, 2)
        if len(fields) < 3:
            continue
        pid, ppid = int(fields[0]), int(fields[1])
        parents[pid] = ppid
        if "atom.entrypoints.openai_server" in fields[2] and f"--server-port {port}" in fields[2]:
            roots.append(pid)
    if len(roots) != 1:
        raise RuntimeError(f"backend {port}: expected one root PID, found {roots}")
    descendants = {roots[0]}
    changed = True
    while changed:
        changed = False
        for pid, ppid in parents.items():
            if ppid in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    return descendants


def set_backend_allowed_pool(port: int, cpus: set[int]) -> dict[str, int]:
    if not cpus:
        raise RuntimeError(f"backend {port}: empty allowed CPU pool")
    descendants = backend_descendants(port)
    tids: set[int] = set()
    for pid in descendants:
        task_dir = pathlib.Path(f"/proc/{pid}/task")
        if task_dir.is_dir():
            tids.update(int(task.name) for task in task_dir.iterdir() if task.name.isdigit())
    changed = 0
    for tid in tids:
        try:
            os.sched_setaffinity(tid, cpus)
            changed += 1
        except ProcessLookupError:
            pass
    log(f"BACKEND_ALLOWED_POOL backend={port} pids={len(descendants)} tids={changed}")
    return {"pids": len(descendants), "tids": changed}


def write_irq_affinity(irq: str | int, affinity: str | int) -> None:
    path = pathlib.Path(f"/proc/irq/{irq}/smp_affinity_list")
    path.write_text(str(affinity))


def irq_affinity(irq: str | int) -> dict[str, str | None]:
    def optional(name: str) -> str | None:
        try:
            return pathlib.Path(f"/proc/irq/{irq}/{name}").read_text().strip()
        except OSError:
            return None

    return {
        "requested": optional("smp_affinity_list"),
        "effective": optional("effective_affinity_list"),
        "node": optional("node"),
    }


def reapply_retained_host_rules(spec: dict[str, Any]) -> None:
    for irq, affinity in spec.get("managed_irq_affinity", {}).items():
        write_irq_affinity(irq, affinity)
    for rule in spec.get("process_affinity_rules", []):
        if rule.get("type") == "backend_descendant_allowed_pool":
            set_backend_allowed_pool(int(rule["server_port"]), {int(cpu) for cpu in rule["cpus"]})


def irq_counts() -> dict[str, dict[str, Any]]:
    lines = pathlib.Path("/proc/interrupts").read_text().splitlines()
    cpu_columns = len(lines[0].split()) if lines else 0
    result: dict[str, dict[str, Any]] = {}
    for line in lines[1:]:
        match = re.match(r"\s*(\d+):\s+(.*)", line)
        if not match:
            continue
        fields = match.group(2).split()
        counts = [int(value) for value in fields[:cpu_columns] if value.isdigit()]
        result[match.group(1)] = {
            "total": sum(counts),
            "per_cpu": counts,
            "description": " ".join(fields[cpu_columns:]),
        }
    return result
