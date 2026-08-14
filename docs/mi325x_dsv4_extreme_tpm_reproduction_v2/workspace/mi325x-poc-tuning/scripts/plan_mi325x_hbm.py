#!/usr/bin/env python3
"""Conservative MI325X HBM fit and replica-topology preflight."""

from __future__ import annotations

import argparse
import json
import sys


def nonnegative(value: str) -> float:
    number = float(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return number


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--physical-gpus", type=int, required=True)
    p.add_argument("--hbm-per-gpu-gib", type=nonnegative, default=256.0)
    p.add_argument("--usable-fraction", type=float, default=0.90)
    p.add_argument("--weights-gib", type=nonnegative, required=True, help="Total weights/checkpoint resident HBM per replica")
    p.add_argument("--runtime-gib", type=nonnegative, required=True, help="Total sharded graph/workspace/optimizer HBM per replica")
    p.add_argument("--kv-or-activation-gib", type=nonnegative, required=True, help="Total sharded KV cache or activations per replica")
    p.add_argument("--replicated-gib-per-rank", type=nonnegative, default=0.0)
    p.add_argument("--reserve-gib-per-rank", type=nonnegative, default=8.0)
    p.add_argument("--tensor-parallel", type=int, required=True)
    p.add_argument("--replicas", type=int, default=1)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    if args.physical_gpus < 1 or args.tensor_parallel < 1 or args.replicas < 1:
        p.error("physical-gpus, tensor-parallel, and replicas must be positive")
    if not 0 < args.usable_fraction <= 1:
        p.error("usable-fraction must be in (0, 1]")

    required_gpus = args.tensor_parallel * args.replicas
    sharded_total = args.weights_gib + args.runtime_gib + args.kv_or_activation_gib
    per_rank = sharded_total / args.tensor_parallel + args.replicated_gib_per_rank + args.reserve_gib_per_rank
    usable = args.hbm_per_gpu_gib * args.usable_fraction
    per_rank_headroom = usable - per_rank
    topology_ok = required_gpus <= args.physical_gpus
    memory_ok = per_rank_headroom >= 0
    verdict = "PREFLIGHT_PASS" if topology_ok and memory_ok else "PREFLIGHT_FAIL"
    result = {
        "verdict": verdict,
        "inputs": vars(args) | {"json": None},
        "required_physical_gpus": required_gpus,
        "usable_hbm_per_rank_gib": round(usable, 3),
        "estimated_hbm_per_rank_gib": round(per_rank, 3),
        "headroom_per_rank_gib": round(per_rank_headroom, 3),
        "memory_ok": memory_ok,
        "topology_ok": topology_ok,
        "caveats": [
            "Assumes even sharding across tensor-parallel ranks.",
            "Pipeline/expert/data/context parallelism and logical partitions require separate accounting.",
            "Use measured free HBM and real peak telemetry before acceptance.",
            "Logical CPX devices do not each own 256 GiB.",
        ],
    }
    result["inputs"].pop("json", None)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Verdict: {verdict}")
        print(f"Required physical GPUs: {required_gpus}/{args.physical_gpus}")
        print(f"Estimated per-rank HBM: {per_rank:.2f} GiB")
        print(f"Usable per-rank HBM:    {usable:.2f} GiB")
        print(f"Per-rank headroom:      {per_rank_headroom:.2f} GiB")
    return 0 if verdict == "PREFLIGHT_PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
