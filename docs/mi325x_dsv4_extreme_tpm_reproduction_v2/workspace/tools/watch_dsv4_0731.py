#!/usr/bin/env python3
"""Watch an attached downloader child and resume its paused supervisor safely."""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import download_dsv4_0731 as downloader  # noqa: E402


def process_state(pid: int) -> str | None:
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[2]
    except (OSError, IndexError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--child-pid", type=int, required=True)
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()

    manifest = downloader.load_manifest()
    total = sum(int(item["size"]) for item in manifest["files"])
    previous = downloader.scan(manifest)["effective_bytes"]
    previous_at = time.monotonic()

    while True:
        progress = downloader.scan(manifest)
        now = time.monotonic()
        elapsed = max(now - previous_at, 1e-6)
        rate = max(0, progress["effective_bytes"] - previous) / elapsed
        child_state = process_state(args.child_pid)
        parent_state = process_state(args.parent_pid)
        print(
            time.strftime("%Y-%m-%d %H:%M:%S"),
            f"{100 * progress['effective_bytes'] / total:.2f}%",
            f"rate={rate / 2**20:.2f}MiB/s",
            f"files={progress['complete_files']}/{downloader.EXPECTED_FILES}",
            f"parent={parent_state or 'MISSING'} child={child_state or 'MISSING'}",
            flush=True,
        )

        child_finished = child_state in (None, "Z")
        files_finished = progress["complete_files"] == downloader.EXPECTED_FILES
        if child_finished or files_finished:
            if parent_state is not None:
                os.kill(args.parent_pid, signal.SIGCONT)
                print(
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    "resumed supervisor",
                    f"reason={'files-complete' if files_finished else 'child-finished'}",
                    flush=True,
                )
            return 0

        if parent_state is None:
            print(time.strftime("%Y-%m-%d %H:%M:%S"), "supervisor disappeared", flush=True)
            return 2

        previous = progress["effective_bytes"]
        previous_at = now
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
