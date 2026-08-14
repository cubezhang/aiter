#!/usr/bin/env python3
"""Collect a read-only MI325X node snapshot locally or over SSH."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any


COMMANDS: tuple[tuple[str, str], ...] = (
    ("hostname", "hostname"),
    ("hostname_fqdn", "hostname -f"),
    ("hostname_ips", "hostname -I"),
    ("ip_addresses", "ip -json address show"),
    ("routes", "ip -json route show"),
    ("os_release", "cat /etc/os-release"),
    ("kernel", "uname -a"),
    ("kernel_cmdline", "cat /proc/cmdline"),
    ("virtualization", "systemd-detect-virt 2>/dev/null || true"),
    ("host_memory_kib", "awk '/^MemTotal:/ {print $2}' /proc/meminfo"),
    ("cpu_numa", "lscpu"),
    ("numa_hardware", "numactl --hardware"),
    ("numa_balancing", "cat /proc/sys/kernel/numa_balancing"),
    ("device_nodes", "ls -l /dev/kfd /dev/dri/render*"),
    ("user_identity", "id"),
    ("mi325x_pci", "lspci -nn -d 1002:74a5"),
    ("mi325x_pcie_verbose", "lspci -d 1002:74a5 -vvv"),
    ("amd_smi_version", "amd-smi version --json"),
    ("amd_smi_list", "amd-smi list -e --json"),
    ("amd_smi_static", "amd-smi static --asic --vram --driver --board --partition --json"),
    ("amd_smi_partition", "amd-smi partition --current"),
    ("amd_smi_topology", "amd-smi topology --json"),
    ("amd_smi_xgmi", "amd-smi xgmi --link-status"),
    ("amd_smi_ecc", "amd-smi metric --ecc --ecc-blocks --json"),
    ("amd_smi_bad_pages", "amd-smi bad-pages --json"),
    ("amd_smi_processes", "amd-smi process --json"),
    ("amd_smi_metrics", "amd-smi metric --usage --power --clock --temperature --json"),
    (
        "rocm_version",
        "if [ -r /opt/rocm/.info/version ]; then cat /opt/rocm/.info/version; "
        "elif command -v hipconfig >/dev/null 2>&1; then hipconfig --version; else exit 127; fi",
    ),
    ("hipconfig", "hipconfig --full"),
    ("gpu_archs", "rocm_agent_enumerator"),
    ("rocminfo_head", "rocminfo | sed -n '1,360p'"),
    ("python_version", "python3 --version"),
    ("glibc_version", "ldd --version | head -1"),
    ("storage_capacity", "df -hT"),
    ("storage_inodes", "df -i"),
    ("block_devices", "lsblk -o NAME,TYPE,SIZE,FSTYPE,MOUNTPOINTS"),
    ("rdma_links", "if command -v rdma >/dev/null 2>&1; then rdma -j link; else exit 127; fi"),
    ("rdma_devices", "if command -v ibv_devices >/dev/null 2>&1; then ibv_devices; else exit 127; fi"),
    ("docker_version", "docker version --format '{{json .}}'"),
    ("docker_info", "docker info --format '{{json .}}'"),
    (
        "pytorch",
        "python3 -c 'import json,torch; print(json.dumps({"
        "\"torch\":torch.__version__,\"hip\":getattr(torch.version,\"hip\",None),"
        "\"available\":torch.cuda.is_available(),\"devices\":torch.cuda.device_count(),"
        "\"names\":[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],"
        "\"archs\":[getattr(torch.cuda.get_device_properties(i),\"gcnArchName\",None) "
        "for i in range(torch.cuda.device_count())]}))'",
    ),
    ("kernel_warnings", "dmesg --level=err,warn 2>/dev/null | tail -n 300"),
)

MAX_CAPTURE_CHARS = 120_000
TARGET_PATTERN = re.compile(r"^[A-Za-z0-9_.@:\-\[\]]+$")


def _validate_target(target: str) -> None:
    if target and (target.startswith("-") or not TARGET_PATTERN.fullmatch(target)):
        raise ValueError("SSH target contains unsupported characters")


def _run(command: str, host: str, port: int, timeout: int) -> dict[str, Any]:
    started = time.monotonic()
    if host:
        argv = [
            "ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
            "-o", f"ConnectTimeout={min(timeout, 15)}", "-p", str(port), host, command,
        ]
    elif os.name == "nt":
        return {
            "command": command, "returncode": 127, "stdout": "",
            "stderr": "Local collection requires Linux; use --host for a remote MI325X node.",
            "duration_ms": 0,
        }
    else:
        argv = ["/bin/sh", "-lc", command]

    try:
        result = subprocess.run(
            argv, capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout, check=False,
        )
        return {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout[:MAX_CAPTURE_CHARS].strip(),
            "stderr": result.stderr[:MAX_CAPTURE_CHARS].strip(),
            "duration_ms": round((time.monotonic() - started) * 1000),
        }
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        if isinstance(exc, subprocess.TimeoutExpired):
            stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout
            stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr
            code = 124
            message = ((stderr or "") + f"\nTimed out after {timeout}s").strip()
        else:
            stdout, message, code = "", str(exc), 127
        return {
            "command": command, "returncode": code,
            "stdout": (stdout or "")[:MAX_CAPTURE_CHARS].strip(),
            "stderr": message[:MAX_CAPTURE_CHARS],
            "duration_ms": round((time.monotonic() - started) * 1000),
        }


def _status(name: str, status: str, detail: str) -> dict[str, str]:
    return {"name": name, "status": status, "detail": detail}


def _derive(results: dict[str, dict[str, Any]], expected_gpus: int) -> dict[str, Any]:
    pci_text = str(results["mi325x_pci"]["stdout"])
    static_text = str(results["amd_smi_static"]["stdout"])
    list_text = str(results["amd_smi_list"]["stdout"])
    arch_text = "\n".join(
        [str(results["gpu_archs"]["stdout"]), str(results["rocminfo_head"]["stdout"]), str(results["pytorch"]["stdout"])]
    )
    nodes_text = str(results["device_nodes"]["stdout"])
    pcie_text = str(results["mi325x_pcie_verbose"]["stdout"])
    memory_text = str(results["host_memory_kib"]["stdout"]).strip()
    cmdline = str(results["kernel_cmdline"]["stdout"])

    pci_count = len([line for line in pci_text.splitlines() if line.strip()])
    mi325_mentions = len(re.findall(r"MI325X", static_text + list_text, re.IGNORECASE))
    gfx942_mentions = len(re.findall(r"gfx942", arch_text, re.IGNORECASE))
    link_pairs = re.findall(r"LnkSta:.*?Speed\s+32GT/s.*?Width\s+x16", pcie_text, re.IGNORECASE)
    fatal_errors = len(re.findall(r"FatalErr\+", pcie_text))
    required_boot = ("pci=realloc=off", "pci=bfsort", "iommu=pt", "numa_balancing=disable")
    missing_boot = [item for item in required_boot if item not in cmdline]

    try:
        host_memory_tib = round(int(memory_text.splitlines()[0]) / (1024**3), 3)
    except (ValueError, IndexError):
        host_memory_tib = None

    gates = [
        _status("mi325x_pci_count", "PASS" if pci_count == expected_gpus else "FAIL", f"detected={pci_count}, expected={expected_gpus}"),
        _status("mi325x_market_name", "PASS" if mi325_mentions >= expected_gpus else "REVIEW", f"MI325X mentions={mi325_mentions}"),
        _status("gfx942", "PASS" if gfx942_mentions >= expected_gpus else "REVIEW", f"gfx942 mentions={gfx942_mentions}"),
        _status("device_nodes", "PASS" if "/dev/kfd" in nodes_text and "/dev/dri/render" in nodes_text else "FAIL", "requires /dev/kfd and render nodes"),
        _status("pcie_links", "PASS" if len(link_pairs) >= expected_gpus and fatal_errors == 0 else "REVIEW", f"32GT/s x16={len(link_pairs)}, FatalErr+={fatal_errors}"),
        _status("rocm_detected", "PASS" if results["rocm_version"]["returncode"] == 0 else "FAIL", str(results["rocm_version"]["stdout"] or results["rocm_version"]["stderr"])),
        _status("host_memory_formal", "PASS" if host_memory_tib is not None and host_memory_tib >= 2.5 else "REVIEW", f"detected_tib={host_memory_tib}; current standard MI325X formal page uses >=2.5 TiB"),
        _status("formal_boot_args", "PASS" if not missing_boot else "REVIEW", f"missing={missing_boot}; confirm current OEM/AMD guide before changes"),
        _status("docker_detected", "PASS" if results["docker_version"]["returncode"] == 0 else "REVIEW", "required for container-first AI paths"),
    ]
    statuses = {gate["status"] for gate in gates}
    overall = "FAIL" if "FAIL" in statuses else ("REVIEW" if "REVIEW" in statuses else "PASS")
    return {
        "overall": overall,
        "physical_pci_gpu_count": pci_count,
        "host_memory_tib": host_memory_tib,
        "verified_hostname": str(results["hostname"]["stdout"]),
        "verified_fqdn": str(results["hostname_fqdn"]["stdout"]),
        "reported_ips": str(results["hostname_ips"]["stdout"]).split(),
        "gates": gates,
        "manual_review": [
            "HBM size per physical versus logical device",
            "partition/NPS mode and parent physical BDF mapping",
            "XGMI topology/link status and GPU/NIC/NUMA locality",
            "ECC/RAS deltas and bad pages",
            "firmware/OEM bill of materials and power/cooling",
            "RVS/AGFHC, TransferBench, rocBLAS, BabelStream and RCCL",
            "RDMA and multi-node validation",
            "framework/kernel ABI preflight and workload correctness",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="", help="SSH target as [user@]host; omit on Linux GPU host")
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--expected-gpus", type=int, default=8, help="Expected physical MI325X PCI devices")
    parser.add_argument("--timeout", type=int, default=30, help="Per-command seconds")
    parser.add_argument("--output", help="Optional local JSON path")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        _validate_target(args.host)
    except ValueError as exc:
        parser.error(str(exc))
    if not 1 <= args.port <= 65535 or args.expected_gpus < 1 or args.timeout < 1:
        parser.error("port, expected-gpus, and timeout must be positive and valid")

    if args.dry_run:
        print(json.dumps({
            "target": args.host or "local", "expected_physical_gpus": args.expected_gpus,
            "commands": [{"name": name, "command": command} for name, command in COMMANDS],
        }, indent=2))
        return 0

    results = {name: _run(command, args.host, args.port, args.timeout) for name, command in COMMANDS}
    snapshot = {
        "schema_version": 1,
        "collected_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "target": args.host or "local",
        "expected_physical_gpus": args.expected_gpus,
        "read_only": True,
        "commands": results,
        "assessment": _derive(results, args.expected_gpus),
    }
    rendered = json.dumps(snapshot, indent=2, ensure_ascii=False)
    if args.output:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    sys.exit(main())
