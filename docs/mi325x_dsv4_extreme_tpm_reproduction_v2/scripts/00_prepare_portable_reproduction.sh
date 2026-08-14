#!/usr/bin/env bash
set -euo pipefail

REPRO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

: "${MODEL_DIR:?export MODEL_DIR=/data/DeepSeek-V4-Flash-FP8}"
: "${LOCUST_DATASET_DIR:?export LOCUST_DATASET_DIR=/path/to/private/dataset}"
: "${LOCUST_MANIFEST_DIR:?export LOCUST_MANIFEST_DIR=/path/to/private/manifests}"

RUN_ROOT=${RUN_ROOT:-"${REPRO_ROOT}/runtime"}
ASSET_ROOT="${RUN_ROOT}/assets"
WORKSPACE="${RUN_ROOT}/workspace"
ATOM_ROOT="${RUN_ROOT}/ATOM"

mkdir -p "${RUN_ROOT}" "${ASSET_ROOT}"

test -d "${MODEL_DIR}"
test -d "${LOCUST_DATASET_DIR}"
test -d "${LOCUST_MANIFEST_DIR}"
test -f "${REPRO_ROOT}/atom_origin.txt"
test -f "${REPRO_ROOT}/atom_commit.txt"
test -f "${REPRO_ROOT}/atom/worktree.patch"

if [[ ! -d "${ATOM_ROOT}/.git" ]]; then
  git clone "$(cat "${REPRO_ROOT}/atom_origin.txt")" "${ATOM_ROOT}"
fi

git -C "${ATOM_ROOT}" fetch --all --tags
git -C "${ATOM_ROOT}" checkout --detach "$(cat "${REPRO_ROOT}/atom_commit.txt")"

if [[ -s "${REPRO_ROOT}/atom/worktree.patch" ]]; then
  git -C "${ATOM_ROOT}" apply --check "${REPRO_ROOT}/atom/worktree.patch"
  git -C "${ATOM_ROOT}" apply "${REPRO_ROOT}/atom/worktree.patch"
fi

rm -rf "${ASSET_ROOT}/optimization" "${WORKSPACE}" "${RUN_ROOT}/scripts"

cp -a "${REPRO_ROOT}/optimization" "${ASSET_ROOT}/"
cp -a "${REPRO_ROOT}/workspace" "${WORKSPACE}"
cp -a "${REPRO_ROOT}/scripts" "${RUN_ROOT}/scripts"

find "${RUN_ROOT}/scripts" "${WORKSPACE}" \
  -type f \( -name '*.sh' -o -name '*.py' \) \
  -exec sed -i \
    -e "s#/data/models/mi325_dsv4_extreme_tpm_20260812/02_manifests/dataset#${LOCUST_DATASET_DIR}#g" \
    -e "s#/data/models/mi325_dsv4_extreme_tpm_20260812/02_manifests/generated#${LOCUST_MANIFEST_DIR}#g" \
    -e "s#/data/models/mi325_dsv4_model_tuning_round2_20260802/57_topk_ob_hybrid_bpp_20260802_141900/aiter/csrc/kernels/topk_per_row_kernels.cu#${ASSET_ROOT}/optimization/topk/topk_per_row_kernels.cu#g" \
    -e "s#/data/models/mi325_dsv4_perf_tuning_20260803_065439#${ASSET_ROOT}/optimization#g" \
    -e "s#/data/models/mi325_dsv4_reuse_tuning_20260801_103501/02_source/ATOM_official#${ATOM_ROOT}#g" \
    -e "s#/data/mi325_0811/mi325_dsv4_opt_0811#${WORKSPACE}#g" \
    -e "s#/data/models/mi325_dsv4_extreme_tpm_20260812#${ASSET_ROOT}#g" \
    {} +

printf '%s\n' \
  "RUN_ROOT=${RUN_ROOT}" \
  "ASSET_ROOT=${ASSET_ROOT}" \
  "WORKSPACE=${WORKSPACE}" \
  "ATOM_ROOT=${ATOM_ROOT}" \
  "MODEL_DIR=${MODEL_DIR}" \
  "LOCUST_DATASET_DIR=${LOCUST_DATASET_DIR}" \
  "LOCUST_MANIFEST_DIR=${LOCUST_MANIFEST_DIR}" \
  > "${RUN_ROOT}/reproduction.env"

echo "Prepared: ${RUN_ROOT}"
echo "Start:    bash ${RUN_ROOT}/scripts/01_start_service.sh"
echo "Test:     bash ${RUN_ROOT}/scripts/02_run_test.sh"
