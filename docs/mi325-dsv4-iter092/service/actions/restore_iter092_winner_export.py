#!/usr/bin/env python3
"""Idempotently restore the pinned iteration-92 contract winner."""

from __future__ import annotations

import datetime as dt
import json
import os
import pathlib
import subprocess
import sys


PROFILE = "official_clean_1e7659f_dual_tp4_mtp2_leg_noprefix_flysort"
VARIANT = "dual_tp4_mtp2_leg_noprefix_flysort"
IMAGE_ID = "sha256:a5cfa1ab503af6e0f55e0ed83cd7e999edbea6940a38dba44aaeee9a22758976"
CAMPAIGN = pathlib.Path(
    os.environ.get(
        "TPOT_CAMPAIGN_ROOT",
        "/data/hxh/0823/deepseek_v4_iter092_final",
    )
).resolve()
STATE = CAMPAIGN / "state/active_service.json"
ACTION_DIR = pathlib.Path(__file__).resolve().parent


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=False)


def _url_ok(url: str) -> bool:
    return _run(["curl", "-fsS", "--max-time", "5", url]).returncode == 0


def _healthy() -> bool:
    try:
        state = json.loads(STATE.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if state.get("profile_id") != PROFILE or state.get("atom_image_id") != IMAGE_ID:
        return False
    for role in ("service", "router"):
        name = str(state.get(f"{role}_container", ""))
        expected_id = str(state.get(f"{role}_container_id", ""))
        if not name or not expected_id:
            return False
        actual = _run(["docker", "inspect", name, "--format", "{{.Id}} {{.State.Running}}"])
        if actual.returncode or actual.stdout.strip() != f"{expected_id} true":
            return False
    return all(
        _url_ok(url)
        for url in (
            "http://127.0.0.1:18000/v1/models",
            "http://127.0.0.1:18001/v1/models",
            "http://127.0.0.1:18080/v1/models",
        )
    )


def main() -> None:
    requested = sys.argv[1] if len(sys.argv) == 2 else ""
    if requested != PROFILE:
        raise SystemExit(f"unsupported iteration-92 restore profile: {requested!r}")
    if _healthy():
        print(f"SERVICE_REUSE profile={PROFILE}", flush=True)
        return

    # common.py resolves these at import time.  Defaults make the restorer
    # independently callable while preserving explicit worker-provided values.
    os.environ.setdefault("TPOT_WORKFLOW_ROOT", str(ACTION_DIR.parent))
    os.environ.setdefault("TPOT_CAMPAIGN_ROOT", str(CAMPAIGN))
    os.environ.setdefault("TPOT_ITERATION_ID", "92")
    sys.path.insert(0, str(ACTION_DIR))
    import latest_1e_stack

    latest_1e_stack.activate()
    import official_clean

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    transition = official_clean.launch_clean(
        VARIANT, f"restore_iter092_winner_{stamp}_{os.getpid()}"
    )
    if transition.get("profile") != PROFILE or not _healthy():
        raise RuntimeError("iteration-92 winner restore did not produce the pinned healthy profile")
    print(f"SERVICE_RESTORED profile={PROFILE}", flush=True)


if __name__ == "__main__":
    main()
