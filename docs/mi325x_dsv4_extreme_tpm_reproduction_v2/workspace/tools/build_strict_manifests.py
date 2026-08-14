#!/usr/bin/env python3
"""Build deterministic manifests for the MI325X strict TPM contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SEED = "MI325X-DSV4-EXTREME-20260812-v1"


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def keyed(label: str, relative_name: str) -> str:
    return digest_bytes(f"{label}\0{SEED}\0{relative_name}".encode())


def canonical(row: dict[str, Any]) -> bytes:
    return (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def write_manifest(path: Path, rows: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    payload = b"".join(canonical(row) for row in rows)
    path.write_bytes(payload)
    actual_hash = digest_bytes(payload)
    metadata = {
        "schema_version": 1,
        "kind": kind,
        "seed": SEED,
        "count": len(rows),
        "thinking_count": sum(bool(row["thinking"]) for row in rows),
        "non_thinking_count": sum(not bool(row["thinking"]) for row in rows),
        "manifest_sha256": actual_hash,
        "manifest_path": path.name,
    }
    (path.with_suffix(path.suffix + ".meta.json")).write_text(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (path.with_suffix(path.suffix + ".sha256")).write_text(f"{actual_hash}  {path.name}\n", encoding="ascii")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    files = sorted(args.dataset.glob("*.json"), key=lambda p: p.name)
    if len(files) != 3997:
        raise SystemExit(f"expected 3997 JSON files, found {len(files)}")

    records: list[dict[str, Any]] = []
    content_hashes: list[str] = []
    total_bytes = 0
    for source in files:
        raw = source.read_bytes()
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            raise ValueError(f"dataset entry is not a message list: {source}")
        file_hash = digest_bytes(raw)
        content_hashes.append(file_hash)
        total_bytes += len(raw)
        records.append(
            {
                "relative_filename": source.name,
                "file_sha256": file_hash,
                "order_key": keyed("order", source.name),
                "think_key": keyed("think", source.name),
                "screen_key": keyed("screen", source.name),
                "screen_think_key": keyed("screen_think", source.name),
            }
        )

    if total_bytes != 104_950_488:
        raise SystemExit(f"dataset bytes changed: {total_bytes}")
    content_set_hash = digest_bytes(("\n".join(sorted(content_hashes)) + "\n").encode())
    if content_set_hash != "815a66cb3ca54b276d5fed9ffbdcc0312be612df0f8e95e8998a1fdbd100d620":
        raise SystemExit(f"dataset content hash set changed: {content_set_hash}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    full_order = sorted(records, key=lambda row: (row["order_key"], row["relative_filename"]))
    full_thinking = {row["relative_filename"] for row in sorted(records, key=lambda row: row["think_key"])[:1599]}
    full_rows = [
        {
            "position": position,
            "request_id": f"full-{position:04d}",
            "relative_filename": row["relative_filename"],
            "file_sha256": row["file_sha256"],
            "thinking": row["relative_filename"] in full_thinking,
        }
        for position, row in enumerate(full_order)
    ]

    screen_selected = sorted(records, key=lambda row: (row["screen_key"], row["relative_filename"]))[:750]
    screen_thinking = {
        row["relative_filename"]
        for row in sorted(screen_selected, key=lambda row: (row["screen_think_key"], row["relative_filename"]))[:300]
    }
    screen_order = sorted(screen_selected, key=lambda row: (row["order_key"], row["relative_filename"]))
    screen_rows = [
        {
            "position": position,
            "request_id": f"screen-{position:04d}",
            "relative_filename": row["relative_filename"],
            "file_sha256": row["file_sha256"],
            "thinking": row["relative_filename"] in screen_thinking,
        }
        for position, row in enumerate(screen_order)
    ]

    # Smoke is diagnostic only. Its 10 entries include both modes and preserve full-manifest flags.
    smoke_candidates = [row for row in full_rows if row["thinking"]][:5] + [row for row in full_rows if not row["thinking"]][:5]
    smoke_rows = [dict(row, position=position, request_id=f"smoke-{position:04d}") for position, row in enumerate(smoke_candidates)]

    metadata = {
        "full": write_manifest(args.output_dir / "full_3997.jsonl", full_rows, "full"),
        "screen": write_manifest(args.output_dir / "screen_750.jsonl", screen_rows, "screen"),
        "smoke": write_manifest(args.output_dir / "smoke_10.jsonl", smoke_rows, "smoke"),
        "dataset": {
            "count": len(files),
            "bytes": total_bytes,
            "content_hash_set_sha256": content_set_hash,
        },
    }
    (args.output_dir / "manifest_index.json").write_text(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
