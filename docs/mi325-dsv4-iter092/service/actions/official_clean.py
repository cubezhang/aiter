"""Guarded launcher for a clean, image-matched official ATOM stack.

This path deliberately omits every legacy scheduler, GEMM table, TopK source,
AllReduce patch, compiler override and ATOM source bind.  The contract adapter
(the small HTTP router) is retained so the immutable customer client and
dataset remain unchanged.
"""

from __future__ import annotations

import datetime as dt
import csv
import hashlib
import json
import os
import pathlib
import re
import subprocess
import time
from typing import Any

from common import CAMPAIGN, WORK, atomic_json, read_json, read_optional_json
from iter_042 import occupied_ports, stop_exact_active_pair


IMAGE = "rocm/atom-dev@sha256:066841903594e8d691b1c528123dfd2e17eacd293acfde947d4fb02afffd57db"
IMAGE_ID = "sha256:066841903594e8d691b1c528123dfd2e17eacd293acfde947d4fb02afffd57db"
ATOM_COMMIT = "ffdf3bb5ddf8a15a64d521a2eeb3bc7438f9ac21"
AITER_COMMIT = "d78f4797ba4e16927d83f5ede412767dd5430f5a"
MODEL = os.environ.get("ITER092_MODEL_DIR", "/data/DeepSeek-V4-Flash-FP8")
ROUTER = str(pathlib.Path(__file__).resolve().parent.parent / "router.py")
SERVICE_LOG_ROOT = CAMPAIGN / "logs/official_clean"
DENSE_CAPTURE_SIZES = "[1,2,4,8,12,16,20,24,28,32,33,34,35,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100,104,108,112,116,120,124,128,132,136,140,144,148,152,156,160,164,168,172,176,180,184,188,192,196,200,204,208,212,216,220,224,228,232,236,240,244,248,252,256]"
M102_OVERLAY = pathlib.Path(__file__).resolve().parent.parent / "config_overlays/latest_ffdf_m102_a8w8.csv"
M102_OVERLAY_SHA256 = "c4a18782ccf999036c18ea79066813f931c53d6cf799ece4e8908e11605425f9"
# Set only by a guarded iteration after an exact-shape operator probe.  Keeping
# this empty preserves the clean official stack for every existing variant.
FMOE_OVERLAY_OVERRIDE: pathlib.Path | None = None
# Optional guarded ATOM source worktree used only by an iteration that records
# its exact base commit and patch digest.  Existing clean variants leave this
# unset and continue to execute the image-owned /app/ATOM tree.
ATOM_SOURCE_OVERRIDE: pathlib.Path | None = None
# Optional guarded AITER HSA directory used for a branch-only gfx942 code
# object plus its registry row.  The image-owned Python/C++ dispatch stays
# pinned; only the two audited HSA files are mounted for the selected variant.
AITER_FMOE_HSA_OVERRIDE: pathlib.Path | None = None
# Optional guarded replacement for the image-owned AITER fused_moe.py.  Used
# only after an exact operator probe of an auditable patch on the image's
# exact AITER commit.
AITER_FMOE_SOURCE_OVERRIDE: pathlib.Path | None = None
TRITON_FP8_MOE_BLOCK_M_OVERRIDE = 16
TRITON_FP8_MOE_CONFIG_JSON_OVERRIDE = "{}"


def prepare_previous_pair(evidence_name: str) -> dict[str, Any]:
    """Resolve the previous service without guessing ownership.

    A missing state file is normal on a clean first launch only when all
    service ports are free. Occupied ports still require an exact recorded
    container pair before anything may be stopped.
    """
    ports = occupied_ports()
    if ports:
        return stop_exact_active_pair(evidence_name)
    return {
        "ports_free": True,
        "stale_active_service": read_optional_json(
            CAMPAIGN / "state/active_service.json"
        ),
    }


SERVICE_SCRIPT = r"""
set -euo pipefail
unset AITER_REBUILD AITER_CONFIG_GEMM_BF16 AITER_CONFIG_FMOE \
  AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE ATOM_USE_TRITON_MOE \
  ATOM_USE_TRITON_FP8_MOE \
  AITER_TP_ALLREDUCE_USE_NEW AITER_QUICK_REDUCE_QUANTIZATION
export AITER_LOG_LEVEL=WARNING
export ATOM_FUSE_SHARED_EXPERT="$FUSE_SHARED_EXPERT"
export ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD="$DUAL_STREAM_MOE_TOKEN_THRESHOLD"
export AITER_USE_FLYDSL_MOE_SORTING="$USE_FLYDSL_MOE_SORTING"
export AITER_USE_CK_MOE_SORTING="$USE_CK_MOE_SORTING"
export ATOM_USE_TRITON_FP8_MOE="$USE_TRITON_FP8_MOE"
export ATOM_TRITON_FP8_MOE_BLOCK_M="$TRITON_FP8_MOE_BLOCK_M"
export ATOM_TRITON_FP8_MOE_CONFIG_JSON="$TRITON_FP8_MOE_CONFIG_JSON"
if [[ -n $A8W8_OVERLAY ]]; then
  export AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE="$A8W8_OVERLAY"
fi
if [[ -n $FMOE_OVERLAY ]]; then
  export AITER_CONFIG_FMOE="$FMOE_OVERLAY"
fi
cd /app/ATOM
start_backend() {
  local devices=$1
  local port=$2
  local dp_master_port=$((29500 + (port - 18000) * 100))
  local spec_args=()
  local graph_args=()
  local cache_args=(--enable-prefix-caching)
  local extra_args=()
  if [[ $SPEC_TOKENS -gt 0 ]]; then
    spec_args=(--method mtp --num-speculative-tokens "$SPEC_TOKENS")
  fi
  if [[ -n $CAPTURE_SIZES ]]; then
    graph_args=(--cudagraph-capture-sizes "$CAPTURE_SIZES")
  fi
  if [[ $ENABLE_PREFIX_CACHING == 0 ]]; then
    cache_args=(--no-enable-prefix-caching)
  fi
  if [[ -n $EXTRA_ATOM_ARGS ]]; then
    read -r -a extra_args <<< "$EXTRA_ATOM_ARGS"
  fi
  if [[ -n $GPU_HW_QUEUES ]]; then
    export GPU_MAX_HW_QUEUES="$GPU_HW_QUEUES"
  else
    unset GPU_MAX_HW_QUEUES
  fi
  if [[ -n $MORI_MODE ]]; then
    export MORI_SHMEM_MODE="$MORI_MODE"
  else
    unset MORI_SHMEM_MODE
  fi
  if [[ -n $QUICK_REDUCE_QUANTIZATION ]]; then
    export AITER_QUICK_REDUCE_QUANTIZATION="$QUICK_REDUCE_QUANTIZATION"
  else
    unset AITER_QUICK_REDUCE_QUANTIZATION
  fi
  if [[ -n $USE_TRITON_MOE ]]; then
    export ATOM_USE_TRITON_MOE="$USE_TRITON_MOE"
  else
    unset ATOM_USE_TRITON_MOE
  fi
  ATOM_DP_MASTER_PORT="$dp_master_port" HIP_VISIBLE_DEVICES="$devices" \
    python -m atom.entrypoints.openai_server \
      --model "$MODEL_ROOT" --served-model-name DeepSeek-V4 \
      --tensor-parallel-size "$TP_SIZE" \
      --data-parallel-size "$DP_SIZE" \
      "${spec_args[@]}" "${graph_args[@]}" \
      --max-num-batched-tokens "$MAX_BATCH_TOKENS" \
      --attn-prefill-chunk-size "$ATTN_PREFILL_CHUNK" \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      "${cache_args[@]}" "${extra_args[@]}" \
      --kv-cache-dtype fp8 --max-model-len 131072 \
      --host 127.0.0.1 --server-port "$port" \
      > "/service_logs/${SERVICE_TAG}_${port}_server.log" 2>&1 &
  backend_pid=$!
}
IFS=";" read -r -a groups <<< "$GPU_GROUPS"
IFS=";" read -r -a ports <<< "$BACKEND_PORTS"
[[ ${#groups[@]} -eq ${#ports[@]} ]] || exit 30
pids=()
for index in "${!groups[@]}"; do
  start_backend "${groups[$index]}" "${ports[$index]}"
  pids+=("$backend_pid")
  for _ in $(seq 1 300); do
    curl -fsS --max-time 3 "http://127.0.0.1:${ports[$index]}/v1/models" >/dev/null && break
    kill -0 "$backend_pid" 2>/dev/null || exit 31
    sleep 3
  done
  curl -fsS --max-time 3 "http://127.0.0.1:${ports[$index]}/v1/models" >/dev/null || exit 32
done
trap "kill ${pids[*]} 2>/dev/null || true; wait ${pids[*]} 2>/dev/null || true" TERM INT EXIT
wait -n "${pids[@]}"
"""


def _run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if check and result.returncode:
        raise RuntimeError(
            f"command failed rc={result.returncode}: {command}: {result.stderr[-2000:]}"
        )
    return result


def _curl_ok(url: str) -> bool:
    return subprocess.run(
        ["curl", "-fsS", "--max-time", "3", url],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def _container_running(name: str) -> bool:
    result = _run(["docker", "inspect", name, "--format", "{{.State.Running}}"], check=False)
    return result.returncode == 0 and result.stdout.strip() == "true"


def remove_exact(names: list[str]) -> None:
    existing = [
        name
        for name in names
        if _run(["docker", "container", "inspect", name], check=False).returncode == 0
    ]
    if not existing:
        return
    _run(["docker", "stop", *existing], check=False)
    _run(["docker", "rm", *existing], check=False)


def _variant(variant: str) -> tuple[int, str, str, str]:
    if variant.startswith("tp8_"):
        return 8, "0,1,2,3,4,5,6,7", "18000", "http://127.0.0.1:18000"
    if variant.startswith("dual_dp4_"):
        return (
            1,
            "0,1,2,3;4,5,6,7",
            "18000;18001",
            "http://127.0.0.1:18000,http://127.0.0.1:18001",
        )
    if variant.startswith("dual_tp4_mtp"):
        return (
            4,
            "0,1,2,3;4,5,6,7",
            "18000;18001",
            "http://127.0.0.1:18000,http://127.0.0.1:18001",
        )
    raise RuntimeError(f"unsupported official clean variant: {variant}")


def _variant_options(variant: str) -> dict[str, Any]:
    """Translate an auditable variant name into only documented ATOM knobs."""
    tokens = set(variant.split("_"))
    legacy_envelope = "leg" in tokens
    mtp_match = re.search(r"(?:^|_)mtp(\d+)(?:_|$)", variant)
    if "no" in tokens and "mtp" in tokens:
        spec_tokens = 0
    elif "nomtp" in tokens or "no-mtp" in tokens:
        spec_tokens = 0
    elif mtp_match:
        spec_tokens = int(mtp_match.group(1))
    else:
        spec_tokens = 0
    tied_batch_match = re.search(r"(?:^|_)bt(\d+)(?:_|$)", variant)
    max_batch_match = re.search(r"(?:^|_)mb(\d+)(?:_|$)", variant)
    chunk_match = re.search(r"(?:^|_)pc(\d+)(?:_|$)", variant)
    max_seq_match = re.search(r"(?:^|_)seq(\d+)(?:_|$)", variant)
    gpu_mem_match = re.search(r"(?:^|_)gm(\d+)(?:_|$)", variant)
    hw_queue_match = re.search(r"(?:^|_)hq(\d+)(?:_|$)", variant)
    # Explicit probe tokens override the inherited legacy envelope so a
    # single variant can isolate scheduler/capture boundaries audibly.
    batch_tokens = int(
        max_batch_match.group(1)
        if max_batch_match
        else tied_batch_match.group(1)
        if tied_batch_match
        else 131072
        if legacy_envelope
        else 16384
    )
    prefill_chunk_tokens = int(
        chunk_match.group(1)
        if chunk_match
        else tied_batch_match.group(1)
        if tied_batch_match
        else 32768
        if legacy_envelope
        else 16384
    )
    max_num_seqs = (
        int(max_seq_match.group(1))
        if max_seq_match
        else 128
        if legacy_envelope
        else 512
    )
    gpu_memory_utilization = 0.83 if legacy_envelope else (
        int(gpu_mem_match.group(1)) / 100.0 if gpu_mem_match else 0.9
    )
    prefix_caching = "noprefix" not in tokens
    extra_args: list[str] = []
    if not prefix_caching:
        extra_args.extend(("--state-checkpoint-interval-tokens", "0"))
    if "dpa" in tokens:
        extra_args.append("--enable-dp-attention")
    if "ep" in tokens:
        extra_args.append("--enable-expert-parallel")
    if "eager" in tokens:
        # Isolation probe for EP graph-capture faults.  This deliberately
        # disables both torch.compile and CUDA/HIP graph capture while keeping
        # the model, FP8 format, DP-attention and routed-expert EP unchanged.
        extra_args.append("--enforce-eager")
    if "tboall" in tokens:
        extra_args.extend(("--enable-tbo", "all"))
    elif "tbo" in tokens:
        extra_args.extend(("--enable-tbo", "prefill"))
    if "ll" in tokens:
        extra_args.extend(("--all2all-backend", "low-latency"))
    if "rapid" in tokens:
        extra_args.append("--enable-rapidserve")
    if "mega" in tokens:
        extra_args.extend(("--moe-backend", "mega"))
    if "oqptpc" in tokens:
        # Re-quantize only the fused DeepSeek expert module from checkpoint
        # per-1x128 FP8 to the officially supported PTPC/per-channel FP8
        # runtime format.  Keep every non-MoE layer in its source format.
        extra_args.extend(
            (
                "--online_quant_config",
                '{"layer_quant_config":{"*ffn.experts*":"ptpc_fp8"}}',
            )
        )
    if "oqmxfp4" in tokens:
        # Re-quantize only the fused expert module to the official MXFP4
        # runtime format.  The guarded V4 source override is still required
        # so the hardcoded source-format lookup does not turn this into a no-op.
        extra_args.extend(
            (
                "--online_quant_config",
                '{"layer_quant_config":{"*ffn.experts*":"mxfp4"}}',
            )
        )
    return {
        "spec_tokens": spec_tokens,
        "data_parallel_size": 4 if variant.startswith("dual_dp4_") else 1,
        "batch_tokens": batch_tokens,
        "prefill_chunk_tokens": prefill_chunk_tokens,
        "max_num_seqs": max_num_seqs,
        "gpu_memory_utilization": gpu_memory_utilization,
        "prefix_caching": prefix_caching,
        "capture_sizes": "" if "eager" in tokens else DENSE_CAPTURE_SIZES,
        "extra_args": extra_args,
        "gpu_hw_queues": (
            hw_queue_match.group(1)
            if hw_queue_match
            else "5" if "tbo" in tokens or "tboall" in tokens else ""
        ),
        "mori_mode": "ISOLATION" if "ep" in tokens else "",
        "mori_shmem_heap_size": "16G" if "ep" in tokens else "",
        "atom_gc_threshold": "20000,50,50" if "gc20k" in tokens else "",
        "quick_reduce_quantization": "INT4" if legacy_envelope or "qrint4" in tokens else "",
        "a8w8_overlay": str(M102_OVERLAY) if legacy_envelope or "overlay102" in tokens else "",
        "fmoe_overlay": (
            str(FMOE_OVERLAY_OVERRIDE)
            if "fmoeoverlay" in tokens and FMOE_OVERLAY_OVERRIDE is not None
            else ""
        ),
        # Official V4 supports either a fused 257-expert/top-7 dispatch or a
        # standalone shared expert overlapped with routed 256-expert/top-6 MoE.
        "fuse_shared_expert": "0" if "nofuseshared" in tokens else "1",
        "dual_stream_moe_token_threshold": "1024",
        "use_flydsl_moe_sorting": "1" if "flysort" in tokens else "0",
        "use_ck_moe_sorting": "1" if "cksort" in tokens else "0",
        "use_triton_moe": (
            "0" if "aitermoe" in tokens else "1" if "tritonmoe" in tokens else ""
        ),
        "use_triton_fp8_moe": "1" if "tritonfp8" in tokens else "0",
        "triton_fp8_moe_block_m": int(TRITON_FP8_MOE_BLOCK_M_OVERRIDE),
        "triton_fp8_moe_config_json": str(TRITON_FP8_MOE_CONFIG_JSON_OVERRIDE),
    }


def _ras_snapshot() -> str:
    chunks: list[str] = []
    for card in range(8):
        result = _run(["rocm-smi", "-d", str(card), "--showrasinfo"], check=False)
        chunks.append(result.stdout + result.stderr)
    return "".join(chunks)


def _bad_pages_clear() -> tuple[bool, Any]:
    result = _run(["amd-smi", "bad-pages", "--json"], check=False)
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError:
        return False, result.stdout + result.stderr
    keys = ("retired", "pending", "un_res")
    clear = bool(
        isinstance(value, list)
        and len(value) == 8
        and all(item.get(key) == "No bad pages found." for item in value for key in keys)
    )
    return clear, value


def launch_clean(variant: str, evidence_name: str) -> dict[str, Any]:
    SERVICE_LOG_ROOT.mkdir(parents=True, exist_ok=True)
    tp_size, groups, ports, router_backends = _variant(variant)
    options = _variant_options(variant)
    overlay_path = str(options["a8w8_overlay"])
    fmoe_overlay_path = str(options["fmoe_overlay"])
    atom_source_path = ""
    atom_source_evidence: dict[str, Any] | None = None
    aiter_hsa_path = ""
    aiter_hsa_mode = ""
    aiter_hsa_evidence: dict[str, Any] | None = None
    aiter_source_file_path = ""
    aiter_source_evidence: dict[str, Any] | None = None
    source_override_tokens = {"tritonfp8", "v4oqexpert", "headsrc"}
    selected_source_tokens = source_override_tokens.intersection(
        set(variant.split("_"))
    )
    if selected_source_tokens:
        if ATOM_SOURCE_OVERRIDE is None:
            raise RuntimeError(
                "variant requires guarded ATOM_SOURCE_OVERRIDE: "
                f"{sorted(selected_source_tokens)}"
            )
        source = pathlib.Path(ATOM_SOURCE_OVERRIDE).resolve()
        if not (source / ".git").exists():
            # A linked git worktree has a .git text file; a normal checkout has
            # a directory.  exists() accepts both and rejects arbitrary trees.
            raise RuntimeError(f"guarded ATOM source is not a git worktree: {source}")
        head = _run(["git", "-C", str(source), "rev-parse", "HEAD"]).stdout.strip()
        if head != ATOM_COMMIT:
            raise RuntimeError(f"ATOM override base mismatch: {head} != {ATOM_COMMIT}")
        diff = _run(["git", "-C", str(source), "diff", "--binary", "HEAD"]).stdout
        untracked = _run(
            ["git", "-C", str(source), "ls-files", "--others", "--exclude-standard"]
        ).stdout.splitlines()
        patch_material = diff.encode("utf-8")
        for relative in sorted(untracked):
            candidate = (source / relative).resolve()
            if not candidate.is_relative_to(source) or not candidate.is_file():
                raise RuntimeError(f"invalid untracked ATOM override file: {relative}")
            patch_material += b"\nUNTRACKED " + relative.encode("utf-8") + b"\n"
            patch_material += candidate.read_bytes()
        clean_exact_commit = bool(
            "headsrc" in selected_source_tokens and not patch_material.strip()
        )
        if not patch_material.strip() and not clean_exact_commit:
            raise RuntimeError("ATOM source override has no auditable patch")
        atom_source_path = str(source)
        atom_source_evidence = {
            "path": atom_source_path,
            "base_commit": head,
            "patch_sha256": (
                hashlib.sha256(patch_material).hexdigest()
                if patch_material.strip()
                else None
            ),
            "clean_exact_commit": clean_exact_commit,
            "tree": _run(["git", "-C", str(source), "rev-parse", "HEAD^{tree}"])
            .stdout.strip(),
            "untracked_files": sorted(untracked),
            "status": _run(["git", "-C", str(source), "status", "--short"]).stdout.splitlines(),
        }
    if "emsort256" in set(variant.split("_")):
        if AITER_FMOE_HSA_OVERRIDE is None:
            raise RuntimeError("emsort256 variant requires guarded AITER_FMOE_HSA_OVERRIDE")
        hsa = pathlib.Path(AITER_FMOE_HSA_OVERRIDE).resolve()
        binary = hsa / "fmoe_bf16_a16_blockscaleFp8_g1u1_vs_silu_1tg_emsort_16x256.co"
        manifests = [
            hsa / "fmoe_bf16_blockscaleFp8_g1u1_silu.csv",
            hsa / "fmoe_bf16_blockscaleBf16_g1u1_silu.csv",
        ]
        if not binary.is_file() or not all(manifest.is_file() for manifest in manifests):
            raise RuntimeError(f"missing guarded emsort256 HSA files under {hsa}")
        branch_head = _run(["git", "-C", str(hsa), "rev-parse", "HEAD"]).stdout.strip()
        expected_branch = "05ddabb5781610b1b0c66e2f4e797062710c9ace"
        if branch_head != expected_branch:
            raise RuntimeError(
                f"emsort256 branch mismatch: {branch_head} != {expected_branch}"
            )
        kernel_name = (
            "_ZN5aiter58fmoe_bf16_a16_blockscaleFp8_g1u1_vs_silu_1tg_"
            "emsort_16x256E"
        )
        if any(
            kernel_name not in manifest.read_text(errors="strict")
            for manifest in manifests
        ):
            raise RuntimeError(
                "guarded emsort256 manifests do not both register the kernel"
            )
        aiter_hsa_path = str(hsa)
        aiter_hsa_mode = "emsort256"
        aiter_hsa_evidence = {
            "path": aiter_hsa_path,
            "branch_commit": branch_head,
            "binary": str(binary),
            "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "manifests": [str(manifest) for manifest in manifests],
            "manifest_sha256": {
                manifest.name: hashlib.sha256(manifest.read_bytes()).hexdigest()
                for manifest in manifests
            },
            "kernel_name": kernel_name,
        }
    if "pyisa32" in set(variant.split("_")):
        if AITER_FMOE_HSA_OVERRIDE is None:
            raise RuntimeError("pyisa32 variant requires guarded AITER_FMOE_HSA_OVERRIDE")
        hsa = pathlib.Path(AITER_FMOE_HSA_OVERRIDE).resolve()
        binary = hsa / "fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_32x256.co"
        if not binary.is_file():
            raise RuntimeError(f"missing guarded pyisa32 binary under {hsa}")
        branch_head = _run(["git", "-C", str(hsa), "rev-parse", "HEAD"]).stdout.strip()
        expected_branch = "8d928bea6b6692cbfe4dc3c17ac98692a474b726"
        if branch_head != expected_branch:
            raise RuntimeError(
                f"pyisa32 branch mismatch: {branch_head} != {expected_branch}"
            )
        binary_sha256 = hashlib.sha256(binary.read_bytes()).hexdigest()
        expected_sha256 = "83472db09b2e6e5e8c52a21b5858722e6f70e0f47d617977dfbf3621a619dfeb"
        if binary_sha256 != expected_sha256:
            raise RuntimeError(
                f"pyisa32 binary mismatch: {binary_sha256} != {expected_sha256}"
            )
        aiter_hsa_path = str(hsa)
        aiter_hsa_mode = "pyisa32"
        aiter_hsa_evidence = {
            "path": aiter_hsa_path,
            "branch_commit": branch_head,
            "binary": str(binary),
            "binary_sha256": binary_sha256,
            "kernel_name": (
                "_ZN5aiter47fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_32x256E"
            ),
        }
    if "aitercache" in set(variant.split("_")):
        if AITER_FMOE_SOURCE_OVERRIDE is None:
            raise RuntimeError(
                "aitercache variant requires guarded AITER_FMOE_SOURCE_OVERRIDE"
            )
        source_file = pathlib.Path(AITER_FMOE_SOURCE_OVERRIDE).resolve()
        if source_file.name != "fused_moe.py" or source_file.parent.name != "aiter":
            raise RuntimeError(f"invalid guarded AITER source file: {source_file}")
        if not source_file.is_file():
            raise RuntimeError(f"missing guarded AITER source file: {source_file}")
        source_root = pathlib.Path(
            _run(
                ["git", "-C", str(source_file.parent), "rev-parse", "--show-toplevel"]
            ).stdout.strip()
        ).resolve()
        source_head = _run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"]
        ).stdout.strip()
        if source_head != AITER_COMMIT:
            raise RuntimeError(
                f"AITER source override base mismatch: {source_head} != {AITER_COMMIT}"
            )
        relative = source_file.relative_to(source_root)
        source_diff = _run(
            ["git", "-C", str(source_root), "diff", "--binary", "HEAD", "--", str(relative)]
        ).stdout
        if not source_diff.strip():
            raise RuntimeError("guarded AITER source override has no auditable patch")
        required_tokens = (
            "_moe_sorting_buf_cache",
            "_scale_t_cache",
            "_get_moe_sorting_bufs",
        )
        source_text = source_file.read_text(errors="strict")
        missing_tokens = [token for token in required_tokens if token not in source_text]
        if missing_tokens:
            raise RuntimeError(
                f"guarded AITER cache source missing tokens: {missing_tokens}"
            )
        aiter_source_file_path = str(source_file)
        aiter_source_evidence = {
            "path": aiter_source_file_path,
            "base_commit": source_head,
            "patch_sha256": hashlib.sha256(source_diff.encode("utf-8")).hexdigest(),
            "file_sha256": hashlib.sha256(source_file.read_bytes()).hexdigest(),
            "upstream_pr": "ROCm/aiter#3204 (closed, unmerged; guarded experiment)",
        }
    fmoe_overlay_sha256 = ""
    if fmoe_overlay_path:
        fmoe_path = pathlib.Path(fmoe_overlay_path).resolve()
        if not fmoe_path.is_file() or not fmoe_path.stat().st_size:
            raise RuntimeError(f"missing/empty guarded FMoE overlay: {fmoe_path}")
        if fmoe_path.suffix != ".csv":
            raise RuntimeError(f"guarded FMoE overlay must be CSV: {fmoe_path}")
        fmoe_overlay_path = str(fmoe_path)
        fmoe_overlay_sha256 = hashlib.sha256(fmoe_path.read_bytes()).hexdigest()
    overlay_sha256 = ""
    overlay_gate: dict[str, Any] | None = None
    if overlay_path:
        overlay_sha256 = hashlib.sha256(pathlib.Path(overlay_path).read_bytes()).hexdigest()
        if overlay_sha256 != M102_OVERLAY_SHA256:
            raise RuntimeError(
                f"M102 overlay hash mismatch: {overlay_sha256} != {M102_OVERLAY_SHA256}"
            )
        with pathlib.Path(overlay_path).open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        keys = {(row["gfx"], int(row["cu_num"]), int(row["M"]), int(row["N"]), int(row["K"])) for row in rows}
        expected_keys = {
            ("gfx942", 304, 102, 1536, 4096),
            ("gfx942", 304, 102, 8192, 1024),
        }
        if keys != expected_keys or any(float(row["errRatio"]) != 0.0 for row in rows):
            raise RuntimeError(f"M102 overlay schema/key gate failed: keys={keys} rows={rows}")
        binary_gate = _run(
            [
                "docker",
                "run",
                "--rm",
                "--entrypoint",
                "bash",
                IMAGE,
                "-lc",
                "test -f /app/aiter-test/hsa/gfx942/fp8gemm_blockscale/"
                "fp8gemm_bf16_blockscale_BpreShuffle_32x128.co && "
                "test -f /app/aiter-test/hsa/gfx942/fp8gemm_blockscale/"
                "fp8gemm_bf16_blockscale_BpreShuffle_48x128.co",
            ],
            check=False,
        )
        if binary_gate.returncode:
            raise RuntimeError("latest image lacks the gfx942 ASM code objects required by M102 overlay")
        overlay_gate = {
            "schema_rows": len(rows),
            "exact_keys": sorted([list(key) for key in keys]),
            "err_ratio_zero": True,
            "latest_image_gfx942_code_objects": True,
        }
    spec_tokens = int(options["spec_tokens"])
    batch_tokens = int(options["batch_tokens"])
    prefill_chunk_tokens = int(options["prefill_chunk_tokens"])
    max_num_seqs = int(options["max_num_seqs"])
    gpu_memory_utilization = float(options["gpu_memory_utilization"])
    profile = f"official_clean_{ATOM_COMMIT[:7]}_{variant}"
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    tag = f"custv1_{profile}_{stamp}"
    service = f"mi325_dsv4_{tag}_dual"
    router = f"mi325_dsv4_{tag}_router"
    transition = {
        "variant": variant,
        "profile": profile,
        "tag": tag,
        "image": IMAGE,
        "image_id": IMAGE_ID,
        "atom_commit": ATOM_COMMIT,
        "aiter_commit": AITER_COMMIT,
        "atom_source_override": atom_source_evidence,
        "aiter_hsa_override": aiter_hsa_evidence,
        "aiter_source_override": aiter_source_evidence,
        "legacy_overrides": (
            ([{"type": "a8w8_exact_m102", "path": overlay_path, "sha256": overlay_sha256}]
             if overlay_path else [])
            + ([{"type": "fmoe_exact_shape", "path": fmoe_overlay_path,
                 "sha256": fmoe_overlay_sha256}]
               if fmoe_overlay_path else [])
        ),
        "overlay_portability_gate": overlay_gate,
        "official_defaults": {
            "max_num_batched_tokens": batch_tokens,
            "attn_prefill_chunk_size": prefill_chunk_tokens,
            "max_num_seqs": max_num_seqs,
            "gpu_memory_utilization": gpu_memory_utilization,
            "prefix_caching": options["prefix_caching"],
        },
        "documented_extra_args": options["extra_args"],
        "documented_environment": {
            "GPU_MAX_HW_QUEUES": options["gpu_hw_queues"],
            "MORI_SHMEM_MODE": options["mori_mode"],
            "MORI_SHMEM_HEAP_SIZE": options["mori_shmem_heap_size"],
            "ATOM_GC_THRESHOLD": options["atom_gc_threshold"],
            "AITER_QUICK_REDUCE_QUANTIZATION": options["quick_reduce_quantization"],
            "ATOM_FUSE_SHARED_EXPERT": options["fuse_shared_expert"],
            "ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD": options[
                "dual_stream_moe_token_threshold"
            ],
            "AITER_USE_FLYDSL_MOE_SORTING": options["use_flydsl_moe_sorting"],
            "AITER_USE_CK_MOE_SORTING": options["use_ck_moe_sorting"],
            "ATOM_USE_TRITON_MOE": options["use_triton_moe"],
            "ATOM_USE_TRITON_FP8_MOE": options["use_triton_fp8_moe"],
            "ATOM_TRITON_FP8_MOE_BLOCK_M": options["triton_fp8_moe_block_m"],
            "ATOM_TRITON_FP8_MOE_CONFIG_JSON": options["triton_fp8_moe_config_json"],
        },
        "official_recipe_capture_sizes": options["capture_sizes"],
    }
    started_epoch = int(time.time())
    ras_before = _ras_snapshot()
    bad_before_clear, bad_before = _bad_pages_clear()
    if not bad_before_clear:
        raise RuntimeError(f"bad pages present before official transition: {bad_before}")
    transition["previous_pair"] = prepare_previous_pair(evidence_name)
    atomic_json(WORK / f"{evidence_name}_transition.json", transition)
    try:
        overlay_mount_args = (
            ["--mount", f"type=bind,src={overlay_path},dst=/tuning/a8w8_overlay.csv,readonly"]
            if overlay_path
            else []
        )
        fmoe_mount_args = (
            ["--mount", f"type=bind,src={fmoe_overlay_path},dst=/tuning/fmoe_overlay.csv,readonly"]
            if fmoe_overlay_path
            else []
        )
        atom_source_mount_args = (
            ["--mount", f"type=bind,src={atom_source_path},dst=/app/ATOM,readonly"]
            if atom_source_path
            else []
        )
        emsort_hsa_mount_args = (
            [
                "--mount",
                "type=bind,src="
                f"{aiter_hsa_path}/fmoe_bf16_a16_blockscaleFp8_g1u1_vs_silu_1tg_"
                "emsort_16x256.co,dst=/app/aiter-test/hsa/gfx942/fmoe/silu/"
                "fmoe_bf16_a16_blockscaleFp8_g1u1_vs_silu_1tg_emsort_16x256.co,readonly",
                "--mount",
                "type=bind,src="
                f"{aiter_hsa_path}/fmoe_bf16_blockscaleFp8_g1u1_silu.csv,"
                "dst=/app/aiter-test/hsa/gfx942/fmoe/silu/"
                "fmoe_bf16_blockscaleFp8_g1u1_silu.csv,readonly",
                "--mount",
                "type=bind,src="
                f"{aiter_hsa_path}/fmoe_bf16_blockscaleBf16_g1u1_silu.csv,"
                "dst=/app/aiter-test/hsa/gfx942/fmoe/silu/"
                "fmoe_bf16_blockscaleBf16_g1u1_silu.csv,readonly",
            ]
            if aiter_hsa_path and aiter_hsa_mode == "emsort256"
            else []
        )
        pyisa_hsa_mount_args = (
            [
                "--mount",
                "type=bind,src="
                f"{aiter_hsa_path}/fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_"
                "32x256.co,dst=/app/aiter-test/hsa/gfx942/fmoe/silu/"
                "fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_32x256.co,readonly",
            ]
            if aiter_hsa_path and aiter_hsa_mode == "pyisa32"
            else []
        )
        aiter_source_mount_args = (
            [
                "--mount",
                f"type=bind,src={aiter_source_file_path},"
                "dst=/app/aiter-test/aiter/fused_moe.py,readonly",
            ]
            if aiter_source_file_path
            else []
        )
        _run(
            [
                "docker",
                "run",
                "--detach",
                "--name",
                service,
                "--network",
                "host",
                "--ipc",
                "host",
                "--shm-size",
                "128g",
                "--security-opt",
                "label=disable",
                "--device",
                "/dev/kfd",
                "--device",
                "/dev/dri",
                "--env",
                f"SERVICE_TAG={tag}",
                "--env",
                f"GPU_GROUPS={groups}",
                "--env",
                f"BACKEND_PORTS={ports}",
                "--env",
                f"TP_SIZE={tp_size}",
                "--env",
                f"DP_SIZE={options['data_parallel_size']}",
                "--env",
                f"SPEC_TOKENS={spec_tokens}",
                "--env",
                f"CAPTURE_SIZES={options['capture_sizes']}",
                "--env",
                f"MAX_BATCH_TOKENS={batch_tokens}",
                "--env",
                f"ATTN_PREFILL_CHUNK={prefill_chunk_tokens}",
                "--env",
                f"MAX_NUM_SEQS={max_num_seqs}",
                "--env",
                f"GPU_MEMORY_UTILIZATION={gpu_memory_utilization}",
                "--env",
                f"ENABLE_PREFIX_CACHING={1 if options['prefix_caching'] else 0}",
                "--env",
                f"EXTRA_ATOM_ARGS={' '.join(options['extra_args'])}",
                "--env",
                f"GPU_HW_QUEUES={options['gpu_hw_queues']}",
                "--env",
                f"MORI_MODE={options['mori_mode']}",
                "--env",
                f"MORI_SHMEM_HEAP_SIZE={options['mori_shmem_heap_size']}",
                "--env",
                f"ATOM_GC_THRESHOLD={options['atom_gc_threshold']}",
                "--env",
                f"QUICK_REDUCE_QUANTIZATION={options['quick_reduce_quantization']}",
                "--env",
                f"A8W8_OVERLAY={'/tuning/a8w8_overlay.csv' if overlay_path else ''}",
                "--env",
                f"FMOE_OVERLAY={'/tuning/fmoe_overlay.csv' if fmoe_overlay_path else ''}",
                "--env",
                f"FUSE_SHARED_EXPERT={options['fuse_shared_expert']}",
                "--env",
                "DUAL_STREAM_MOE_TOKEN_THRESHOLD="
                f"{options['dual_stream_moe_token_threshold']}",
                "--env",
                f"USE_FLYDSL_MOE_SORTING={options['use_flydsl_moe_sorting']}",
                "--env",
                f"USE_CK_MOE_SORTING={options['use_ck_moe_sorting']}",
                "--env",
                f"USE_TRITON_MOE={options['use_triton_moe']}",
                "--env",
                f"USE_TRITON_FP8_MOE={options['use_triton_fp8_moe']}",
                "--env",
                f"TRITON_FP8_MOE_BLOCK_M={options['triton_fp8_moe_block_m']}",
                "--env",
                "TRITON_FP8_MOE_CONFIG_JSON="
                f"{options['triton_fp8_moe_config_json']}",
                "--env",
                f"MODEL_ROOT={MODEL}",
                "--mount",
                f"type=bind,src={SERVICE_LOG_ROOT},dst=/service_logs",
                "--mount",
                f"type=bind,src={MODEL},dst={MODEL},readonly",
                *overlay_mount_args,
                *fmoe_mount_args,
                *atom_source_mount_args,
                *emsort_hsa_mount_args,
                *pyisa_hsa_mount_args,
                *aiter_source_mount_args,
                IMAGE,
                "bash",
                "-lc",
                SERVICE_SCRIPT,
            ]
        )
        port_list = ports.split(";")
        deadline = time.monotonic() + 1200
        while time.monotonic() < deadline:
            if not _container_running(service):
                logs = _run(["docker", "logs", "--tail", "200", service], check=False)
                raise RuntimeError(f"official service exited: {logs.stdout[-4000:]}{logs.stderr[-4000:]}")
            if all(_curl_ok(f"http://127.0.0.1:{port}/v1/models") for port in port_list):
                break
            time.sleep(3)
        else:
            raise RuntimeError("official service readiness timeout")

        _run(
            [
                "docker",
                "run",
                "--detach",
                "--name",
                router,
                "--network",
                "host",
                "--ipc",
                "host",
                "--security-opt",
                "label=disable",
                "--env",
                f"ROUTER_BACKENDS={router_backends}",
                "--env",
                "ROUTER_POLICY=least_connections",
                "--mount",
                f"type=bind,src={ROUTER},dst=/router.py,readonly",
                IMAGE,
                "bash",
                "-lc",
                "exec python -m uvicorn router:app --app-dir / --host 127.0.0.1 --port 18080 --no-access-log",
            ]
        )
        for _ in range(60):
            if _curl_ok("http://127.0.0.1:18080/router/status"):
                break
            time.sleep(1)
        else:
            raise RuntimeError("official router readiness timeout")

        if atom_source_path:
            # A linked worktree's .git file points at a host-side gitdir that
            # is intentionally not exposed inside the serving container.  The
            # exact host commit was already checked above; prove here that the
            # container received that same tree as a read-only /app/ATOM bind.
            inspect_value = json.loads(
                _run(["docker", "inspect", service]).stdout
            )[0]
            atom_mounts = [
                mount
                for mount in inspect_value.get("Mounts", [])
                if mount.get("Destination") == "/app/ATOM"
            ]
            if len(atom_mounts) != 1:
                raise RuntimeError(
                    f"expected one guarded /app/ATOM mount, got {atom_mounts}"
                )
            atom_mount = atom_mounts[0]
            if (
                pathlib.Path(str(atom_mount.get("Source", ""))).resolve()
                != pathlib.Path(atom_source_path).resolve()
                or atom_mount.get("RW") is not False
            ):
                raise RuntimeError(
                    f"guarded /app/ATOM mount provenance mismatch: {atom_mount}"
                )
            atom_commit = _run(
                ["git", "-C", atom_source_path, "rev-parse", "HEAD"]
            ).stdout.strip()
            transition["container_atom_source_mount"] = atom_mount
        else:
            atom_commit = _run(
                ["docker", "exec", service, "git", "-C", "/app/ATOM", "rev-parse", "HEAD"]
            ).stdout.strip()
        aiter_commit = _run(
            ["docker", "exec", service, "git", "-C", "/app/aiter-test", "rev-parse", "HEAD"]
        ).stdout.strip()
        if atom_commit != ATOM_COMMIT or aiter_commit != AITER_COMMIT:
            raise RuntimeError(f"official commit mismatch ATOM={atom_commit} AITER={aiter_commit}")

        log_text = "\n".join(
            (SERVICE_LOG_ROOT / f"{tag}_{port}_server.log").read_text(errors="replace")
            for port in port_list
        )
        forbidden = (
            "unexpected keyword argument",
            "Engine Core: load model runner failed",
            "VM_L2_PROTECTION_FAULT",
        )
        hits = [needle for needle in forbidden if needle in log_text]
        if hits:
            raise RuntimeError(f"official clean fatal log hits: {hits}")
        effective_kv = "bf16" if "Falling back to a bf16 KV cache" in log_text else "fp8"

        ras_after = _ras_snapshot()
        bad_after_clear, bad_after = _bad_pages_clear()
        kernel = _run(
            ["journalctl", "-k", "--since", f"@{started_epoch}", "--no-pager"],
            check=False,
        ).stdout
        fault_re = re.compile(
            r"amdgpu.*(?:page fault|gpu reset|fatal)|VM_L2_PROTECTION_FAULT|xgmi.*(?:fatal|error)",
            re.IGNORECASE,
        )
        kernel_faults = [line for line in kernel.splitlines() if fault_re.search(line)]
        if ras_after != ras_before or not bad_after_clear or kernel_faults:
            raise RuntimeError(
                "official clean hardware gate failed: "
                f"ras_changed={ras_after != ras_before} bad_pages={bad_after} "
                f"kernel_faults={kernel_faults[-20:]}"
            )
        (WORK / f"{evidence_name}_ras_before.txt").write_text(ras_before)
        (WORK / f"{evidence_name}_ras_after.txt").write_text(ras_after)
        (WORK / f"{evidence_name}_kernel_window.txt").write_text(kernel)

        service_id = _run(["docker", "inspect", service, "--format", "{{.Id}}"] ).stdout.strip()
        router_id = _run(["docker", "inspect", router, "--format", "{{.Id}}"] ).stdout.strip()
        config_hash = hashlib.sha256(
            json.dumps(transition, sort_keys=True).encode("utf-8")
        ).hexdigest()
        state = {
            "schema_version": 1,
            "profile_id": profile,
            "profile_description": "clean official ATOM defaults with only the immutable customer router adapter",
            "profile_sha256": config_hash,
            "service_tag": tag,
            "service_container": service,
            "router_container": router,
            "service_container_id": service_id,
            "router_container_id": router_id,
            "atom_image": IMAGE,
            "atom_image_id": IMAGE_ID,
            "source_root": atom_source_path or "image:///app/ATOM",
            "source_patch_sha256": (
                atom_source_evidence["patch_sha256"] if atom_source_evidence else ""
            ),
            "source_commit": ATOM_COMMIT,
            "aiter_hsa_override": aiter_hsa_evidence,
            "aiter_source_override": aiter_source_evidence,
            "model": MODEL,
            "a8w8_bpreshuffle_config": overlay_path,
            "a8w8_bpreshuffle_config_sha256": overlay_sha256,
            "fmoe_config": "",
            "fmoe_config_sha256": "",
            "bf16_config": "",
            "bf16_config_sha256": "",
            "aiter_compiler_variant": "official_image_default",
            "topology": (
                "2x(DP4/TP1/EP4)"
                if variant.startswith("dual_dp4_")
                else "TP8" if tp_size == 8 else "2xTP4"
            ),
            "router_policy": "least_connections",
            "requested_kv_cache_dtype": "fp8",
            "effective_kv_cache_dtype": effective_kv,
            "speculative_method": "none" if spec_tokens == 0 else "mtp",
            "num_speculative_tokens": spec_tokens,
            "prefix_caching": options["prefix_caching"],
            "documented_extra_args": options["extra_args"],
            "official_clean": True,
            "evidence_dir": str(WORK),
            "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        atomic_json(CAMPAIGN / "state/active_service.json", state)
        transition["active_service"] = state
        transition["effective_kv_cache_dtype"] = effective_kv
        atomic_json(WORK / f"{evidence_name}_ready.json", transition)
        return transition
    except Exception:
        remove_exact([router, service])
        raise
