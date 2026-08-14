#!/usr/bin/env python3
"""Run the read-only MI325X node collector for a validated JSON inventory."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import ipaddress
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


def load_inventory(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    nodes = value.get("nodes") if isinstance(value, dict) else None
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("inventory.nodes must be a non-empty array")
    names: set[str] = set()
    targets: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise ValueError(f"nodes[{index}] must be an object")
        name = str(node.get("name", "")).strip()
        target = str(node.get("ssh_target", "")).strip()
        port = int(node.get("port", 22))
        expected = int(node.get("expected_gpus", 8))
        if not name or not target:
            raise ValueError(f"nodes[{index}] requires name and ssh_target")
        if name in names or target in targets:
            raise ValueError(f"duplicate node name or ssh_target: {name}, {target}")
        if not 1 <= port <= 65535 or expected < 1:
            raise ValueError(f"invalid port/expected_gpus for {name}")
        names.add(name)
        targets.add(target)
        normalized.append({**node, "name": name, "ssh_target": target, "port": port, "expected_gpus": expected})
    return normalized


def literal_target_ip(target: str) -> str | None:
    host = target.rsplit("@", 1)[-1]
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    try:
        return str(ipaddress.ip_address(host))
    except ValueError:
        return None


def run_node(collector: Path, node: dict[str, Any], output_dir: Path, timeout: int) -> dict[str, Any]:
    output = output_dir / f"{node['name']}.json"
    argv = [
        sys.executable, str(collector), "--host", node["ssh_target"],
        "--port", str(node["port"]), "--expected-gpus", str(node["expected_gpus"]),
        "--timeout", str(timeout), "--output", str(output),
    ]
    result = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    snapshot: dict[str, Any] | None = None
    if output.is_file():
        try:
            snapshot = json.loads(output.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            snapshot = None
    assessment = snapshot.get("assessment", {}) if snapshot else {}
    reported_ips = assessment.get("reported_ips", []) if isinstance(assessment, dict) else []
    target_ip = literal_target_ip(node["ssh_target"])
    if target_ip is None:
        ip_status = "REVIEW"
        ip_detail = "SSH target is a hostname; compare DNS and reported interface addresses manually"
    elif target_ip in reported_ips:
        ip_status = "PASS"
        ip_detail = f"target IP {target_ip} reported by remote hostname -I"
    else:
        ip_status = "FAIL"
        ip_detail = f"target IP {target_ip} not in remote reported IPs {reported_ips}"
    return {
        "name": node["name"], "ssh_target": node["ssh_target"], "role": node.get("role"),
        "collector_returncode": result.returncode, "snapshot": str(output),
        "assessment": assessment.get("overall", "FAIL") if isinstance(assessment, dict) else "FAIL",
        "verified_hostname": assessment.get("verified_hostname") if isinstance(assessment, dict) else None,
        "verified_fqdn": assessment.get("verified_fqdn") if isinstance(assessment, dict) else None,
        "reported_ips": reported_ips, "ip_status": ip_status, "ip_detail": ip_detail,
        "stderr": result.stderr[-4000:].strip(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        nodes = load_inventory(args.inventory)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.timeout < 1 or args.workers < 1:
        parser.error("timeout and workers must be positive")

    literal_ips = [literal_target_ip(node["ssh_target"]) for node in nodes]
    actual_ips = [item for item in literal_ips if item is not None]
    if len(actual_ips) != len(set(actual_ips)):
        print("error: duplicate literal target IPs in inventory", file=sys.stderr)
        return 2

    collector = Path(__file__).with_name("collect_mi325x_snapshot.py")
    if args.dry_run:
        print(json.dumps({
            "read_only": True,
            "collector": str(collector),
            "nodes": [{
                "name": node["name"], "ssh_target": node["ssh_target"],
                "port": node["port"], "expected_physical_gpus": node["expected_gpus"],
                "literal_target_ip": literal_target_ip(node["ssh_target"]),
            } for node in nodes],
        }, indent=2))
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(args.workers, len(nodes))) as pool:
        futures = {pool.submit(run_node, collector, node, args.output_dir, args.timeout): node for node in nodes}
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda row: row["name"])
    accepted = all(
        row["collector_returncode"] == 0 and row["assessment"] != "FAIL" and row["ip_status"] != "FAIL"
        for row in rows
    )
    summary = {"schema_version": 1, "read_only": True, "accepted_for_review": accepted, "nodes": rows}
    summary_path = args.output_dir / "cluster-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
