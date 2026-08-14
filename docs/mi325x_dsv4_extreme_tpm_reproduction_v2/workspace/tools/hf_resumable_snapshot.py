#!/usr/bin/env python3
"""Hugging Face snapshot downloader with process-restart-safe HTTP partials.

huggingface_hub 1.24 deliberately uses a UUID-suffixed temporary file opened
with ``wb``. That is safe on filesystems with broken locks, but it means a
supervisor restart cannot resume a multi-GB file. This wrapper is for a local
NVMe filesystem and a single supervised downloader, so it restores the older
fixed-partial behavior and HTTP Range resume semantics.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import huggingface_hub.file_download as file_download
from huggingface_hub import snapshot_download


UUID_PARTIAL_RE = re.compile(r"^(?P<base>.+\.[0-9a-f]{40,64})\.[0-9a-f]{8}\.incomplete$")


def recover_uuid_partials(local_dir: Path) -> tuple[int, int]:
    """Promote the largest UUID partial for each blob to its stable name."""
    cache = local_dir / ".cache" / "huggingface" / "download"
    if not cache.exists():
        return 0, 0

    groups: dict[Path, list[Path]] = {}
    for path in cache.glob("*.incomplete"):
        match = UUID_PARTIAL_RE.match(path.name)
        if match:
            stable = path.with_name(match.group("base") + ".incomplete")
            groups.setdefault(stable, []).append(path)

    promoted = 0
    recovered = 0
    for stable, unique_paths in groups.items():
        candidates = list(unique_paths)
        if stable.exists():
            candidates.append(stable)
        winner = max(candidates, key=lambda item: item.stat().st_size)
        winner_size = winner.stat().st_size
        if winner != stable:
            if stable.exists():
                displaced = stable.with_name(stable.name + f".superseded.{os.getpid()}")
                os.replace(stable, displaced)
            os.replace(winner, stable)
            promoted += 1
            recovered += winner_size
    return promoted, recovered


def resumable_download_to_tmp_and_move(
    incomplete_path: Path,
    destination_path: Path,
    url_to_download: str,
    headers: dict[str, str],
    expected_size: int | None,
    filename: str,
    force_download: bool,
    etag: str | None,
    xet_file_data: object | None,
    tqdm_class: object | None = None,
) -> None:
    """Download to a stable partial and retain it on failure for Range resume."""
    if destination_path.exists() and not force_download:
        return

    incomplete_path.parent.mkdir(parents=True, exist_ok=True)
    if force_download:
        incomplete_path.unlink(missing_ok=True)

    resume_size = incomplete_path.stat().st_size if incomplete_path.exists() else 0
    if expected_size is not None and resume_size > expected_size:
        incomplete_path.unlink()
        resume_size = 0

    if expected_size is not None and resume_size == expected_size:
        file_download._chmod_and_move(incomplete_path, destination_path)
        return

    if expected_size is not None:
        remaining = max(0, expected_size - resume_size)
        file_download._check_disk_space(remaining, incomplete_path.parent)
        file_download._check_disk_space(remaining, destination_path.parent)

    # Xet is deliberately disabled by the supervisor. Regular HTTP supports
    # Range requests and ``http_get`` already retries transient failures from
    # the current file offset within one process.
    with incomplete_path.open("ab") as handle:
        file_download.http_get(
            url_to_download,
            handle,
            resume_size=resume_size,
            headers=headers,
            expected_size=expected_size,
            displayed_filename=filename,
            tqdm_class=tqdm_class,
        )
    file_download._chmod_and_move(incomplete_path, destination_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=48)
    args = parser.parse_args()

    args.local_dir.mkdir(parents=True, exist_ok=True)
    promoted, recovered = recover_uuid_partials(args.local_dir)
    print(
        f"Recovered {promoted} stable partials ({recovered} bytes) before download",
        flush=True,
    )

    file_download._download_to_tmp_and_move = resumable_download_to_tmp_and_move
    result = snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=args.local_dir,
        max_workers=args.max_workers,
    )
    print(result, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
