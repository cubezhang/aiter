#!/usr/bin/env bash
set -euo pipefail

AITER_ROOT="${AITER_ROOT:-/app/aiter-tuning-dsv4}"
BASE_JIT_DIR="${BASE_AITER_JIT_DIR:-/app/aiter-test/aiter/jit}"
TUNED_JIT_DIR="${AITER_JIT_DIR:-${AITER_ROOT}/aiter/jit}"

QK_MODULE="module_fused_qk_norm_rope_cache_quant_shuffle"
QK_SO="${QK_MODULE}.so"

if [[ ! -d "${AITER_ROOT}/aiter" ]]; then
    echo "ERROR: tuned AITER source not found: ${AITER_ROOT}" >&2
    exit 1
fi

if [[ ! -d "${BASE_JIT_DIR}" ]]; then
    echo "ERROR: official-image JIT directory not found: ${BASE_JIT_DIR}" >&2
    exit 1
fi

mkdir -p "${TUNED_JIT_DIR}"

echo "Seeding prebuilt AITER modules from the official image..."

while IFS= read -r -d '' module; do
    name="$(basename "${module}")"

    # This module contains branch-specific changes and must be rebuilt.
    if [[ "${name}" == "${QK_SO}" ]]; then
        continue
    fi

    if [[ ! -e "${TUNED_JIT_DIR}/${name}" ]]; then
        cp -a "${module}" "${TUNED_JIT_DIR}/${name}"
    fi
done < <(
    find "${BASE_JIT_DIR}" \
        -maxdepth 1 \
        -type f \
        -name 'module_*.so' \
        -print0
)

export PYTHONPATH="${AITER_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export AITER_JIT_DIR="${TUNED_JIT_DIR}"

# Never let TP workers rebuild shared JIT modules concurrently.
unset AITER_REBUILD

echo "Building ${QK_MODULE} in a single process..."

python3 - <<'PY'
import importlib
import os

from aiter.jit.core import build_module, get_args_of_build

name = "module_fused_qk_norm_rope_cache_quant_shuffle"
args = get_args_of_build(name)

build_module(
    name,
    args["srcs"],
    args["flags_extra_cc"],
    args["flags_extra_hip"],
    args["blob_gen_cmd"],
    args["extra_include"],
    args["extra_ldflags"],
    args["verbose"],
    args["is_python_module"],
    args["is_standalone"],
    args["torch_exclude"],
    args["third_party"],
    hipify=args.get("hipify", False),
    flags_extra_hip_per_source=args.get(
        "flags_extra_hip_per_source", None
    ),
)

module = importlib.import_module(name)

print("Built module:", module.__file__)
print("AITER_JIT_DIR:", os.environ["AITER_JIT_DIR"])
PY

test -f "${TUNED_JIT_DIR}/${QK_SO}"

echo
echo "AITER initialization completed successfully."
echo "Keep AITER_REBUILD unset when starting the TP4 server."
