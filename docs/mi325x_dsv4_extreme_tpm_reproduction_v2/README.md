# MI325X DeepSeek-V4-Flash-FP8 Extreme TPM Reproduction

This directory is a portable reproduction package for the frozen MI325X DeepSeek-V4-Flash-FP8 dual-TP4 optimization result.

The repository includes the service scripts, test scripts, AITER optimization artifacts, Top-k source, communication patches, ATOM patch, pinned ATOM revision, and image metadata.

## Requirements

- One host with 8 × AMD Instinct MI325X GPUs
- Docker with access to `/dev/kfd` and `/dev/dri`
- ROCm host environment compatible with the pinned container image
- `iommu=pt` enabled
- `kernel.numa_balancing=0`
- Git access to the ATOM source remote recorded in `atom_origin.txt`
- The following private external assets:

```text
MODEL_DIR
  DeepSeek-V4-Flash-FP8 model directory

LOCUST_DATASET_DIR
  Locust dataset directory

LOCUST_MANIFEST_DIR
  Private C53/C56 request-manifest directory
```

The model and Locust request data are intentionally not included in this public repository. Request manifests may contain confidential request content or credentials and must be supplied through an approved private channel.

Do not set `AITER_REBUILD=1`. It forces AITER JIT recompilation and can invalidate the frozen runtime environment.

## Included artifacts

```text
scripts/00_prepare_portable_reproduction.sh
  Creates the portable runtime tree and prepares ATOM source.

scripts/01_start_service.sh
  Starts the frozen service configuration.

scripts/02_run_test.sh
  Runs the frozen test procedure.

atom_origin.txt
atom_commit.txt
atom/worktree.patch
  Pinned ATOM source and required patch.

optimization/
  A8W8 tuning configurations, communication patches, sparse decode
  patches, Quick Reduce experiments, and Top-k source.

workspace/
  Frozen service, benchmark, validation, and summary tooling.
```

## Reproduce on a new MI325X server

Clone the repository:

```bash
git clone --branch mi325-dsv4-ops \
  git@github.com:cubezhang/aiter.git \
  /data/aiter-mi325-repro

cd /data/aiter-mi325-repro
```

Set the private asset locations:

```bash
export MODEL_DIR=/data/DeepSeek-V4-Flash-FP8
export LOCUST_DATASET_DIR=/data/llm_test_datasets-prod
export LOCUST_MANIFEST_DIR=/data/mi325-private-manifests
export RUN_ROOT=/data/mi325-dsv4-repro-runtime
```

Prepare the portable runtime tree:

```bash
bash \
  docs/mi325x_dsv4_extreme_tpm_reproduction_v2/scripts/00_prepare_portable_reproduction.sh
```

The bootstrap script performs the following actions:

1. Creates `RUN_ROOT`.
2. Clones the ATOM source at the revision recorded in `atom_commit.txt`.
3. Applies `atom/worktree.patch`.
4. Copies all included optimization artifacts into `RUN_ROOT`.
5. Rewrites the frozen scripts in `RUN_ROOT` to use the new server paths.
6. Writes `RUN_ROOT/reproduction.env` with the resolved paths.

Start the service:

```bash
bash "$RUN_ROOT/scripts/01_start_service.sh"
```

Run the frozen test:

```bash
bash "$RUN_ROOT/scripts/02_run_test.sh"
```

## Runtime layout

After bootstrap, the generated runtime tree is:

```text
$RUN_ROOT/
├── ATOM/                 # Pinned ATOM source with frozen patch applied
├── assets/
│   └── optimization/     # Included tuning tables, patches, and Top-k source
├── workspace/            # Frozen benchmark and service tooling
├── scripts/              # Rewritten start and test scripts
└── reproduction.env      # Resolved paths used for this reproduction
```

All generated logs, service state, benchmark results, and test results are written below `RUN_ROOT`.

## Verify the prepared runtime

```bash
cat "$RUN_ROOT/reproduction.env"

test -d "$RUN_ROOT/ATOM"
test -d "$RUN_ROOT/assets/optimization"
test -d "$RUN_ROOT/workspace"
test -x "$RUN_ROOT/scripts/01_start_service.sh"
test -x "$RUN_ROOT/scripts/02_run_test.sh"
```

## Stop the service

Use the stop logic included in the generated service scripts, or remove only the exact containers created by this reproduction run.

Do not remove unrelated containers, images, model files, or shared datasets.

## Security and reproducibility notes

- Never commit model weights, JSONL request manifests, API keys, tokens, or private datasets.
- Verify private asset provenance before testing.
- Keep the pinned image, ATOM revision, and patch unchanged when reproducing the frozen result.
- Any change to model revision, ATOM revision, Docker image, tuning table, request manifest, GPU topology, or runtime environment creates a new test configuration and must be measured separately.
