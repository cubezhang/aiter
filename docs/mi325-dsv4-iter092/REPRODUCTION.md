# MI325X DeepSeek-V4 Iteration-92 reproduction

This handoff has two independent workflows:

1. prepare and start the inference service;
2. prepare and run the private customer Locust test.

The service workflow never reads or validates the Locust script, customer
dataset, or Locust client image. Missing test inputs therefore do not block
service startup.

The accepted C60 result was:

- total throughput: 144.3996 ten-thousand tokens/minute (1,443,996 TPM)
- TPOT: 19.70 ms
- client TTFT: 361.94 ms
- scored completions: 6000, failures: 0
- hardware: 8 x AMD Instinct MI325X (`1002:74a5`, `gfx942`)

The public handoff excludes the model, customer dataset, frozen customer
Locust script, image archives, raw request logs, and the full optimization
ledger.

## Clone the public branch

```bash
git clone --branch mi325-dsv4-ops --single-branch \
  https://github.com/cubezhang/aiter.git
cd aiter/docs/mi325-dsv4-iter092
```

## Workflow A: start the service without test inputs

### A1. Service prerequisites

Only these external inputs are required to start the service:

| Service dependency | Required value |
|---|---|
| Hardware | 8 unpartitioned MI325X GPUs in SPX/NPS1 mode |
| GPU identity | `1002:74a5`, `gfx942` |
| GPU devices | `/dev/kfd` and `/dev/dri` |
| Model | `/data/DeepSeek-V4-Flash-FP8` |
| Service image | public pinned ATOM image shown below |
| Free service ports | 18000, 18001, and 18080 |

Required host commands are `docker`, `curl`, `jq`, `python3`, `rocm-smi`,
`amd-smi`, `journalctl`, and `ss`.

The accepted runtime is pinned to:

```text
rocm/atom-dev:nightly_202608201458@sha256:a5cfa1ab503af6e0f55e0ed83cd7e999edbea6940a38dba44aaeee9a22758976
ATOM  1e7659fde32eeaa0d9aa868c3e90847e5e46a51c
AITER eb84cb02200b1707f1076edf3f4930d3626adfb2
```

The older official image
`rocm/atom-dev:nightly_202607271535@sha256:66df2fb1c537f52d47f2d3bf973b8da15278dd2ee362b5307e3d29dfad69fa7d`
was used by the earlier `iter_030` profile and is not the Iteration-92 winner.

### A2. Prepare only the service image

This command works with an empty local Docker cache. It pulls the immutable
public manifest, retags it for replay, validates its image ID, and checks the
embedded ATOM and AITER commits. It does not build or inspect the Locust image.

```bash
./bin/prepare_service_image.sh
```

Expected service image ID:

```text
sha256:a5cfa1ab503af6e0f55e0ed83cd7e999edbea6940a38dba44aaeee9a22758976
```

`Dockerfile` records the accepted parent. Strict replay uses direct pull and
retag because even a zero-layer Docker build export creates a new manifest ID.

Optional GPU-aware import validation:

```bash
docker run --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --security-opt seccomp=unconfined \
  --entrypoint bash \
  rocm/atom-dev:iter092-replay -lc '
  /opt/rocm-7.2.4/bin/rocminfo | grep -m 8 gfx942
  python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
  python -c "import atom, aiter; print(atom.__file__); print(aiter.__file__)"
'
```

Without the GPU devices, `import aiter` fails when its `rocminfo` probe cannot
detect the architecture. That does not indicate an image or revision mismatch.

### A3. Start or reuse the service

```bash
./bin/restore_iter092_service.sh
```

The launcher runs `verify_service_inputs.sh` internally. That verifier checks
only the MI325X identity and partitioning, model directory, service commands,
and pinned service image. It never checks test files or the Locust image.

On a clean first launch, `state/active_service.json` does not exist and all
four service ports must be free. If a state file is missing while any service
port is occupied, the launcher prints the listener and running-container
evidence and exits without stopping anything. Resolve ownership of those
listeners before retrying; never remove an unidentified service merely to free
the ports.

Verify the endpoints:

```bash
curl -fsS http://127.0.0.1:18000/v1/models | jq .
curl -fsS http://127.0.0.1:18001/v1/models | jq .
curl -fsS http://127.0.0.1:18080/v1/models | jq .
curl -fsS http://127.0.0.1:18080/router/status | jq .
```

The service consists of two TP4 replicas on GPUs 0-3 and 4-7 plus a
least-connections router on port 18080. It uses MTP2, C60, dense graph buckets,
the M102 A8W8 overlay, FlyDSL MoE sorting, no prefix caching, and Quick Reduce
INT4. The accepted run used effective BF16 KV cache.

At this point the service is usable. Workflow B is optional and can be
performed later or on a separate test preparation schedule.

## Workflow B: prepare and run the customer test

Workflow B requires an already healthy service from Workflow A, but it never
starts, stops, or replaces that service.

### B1. Point to the existing local test environment

| Test dependency | Required location or identity |
|---|---|
| Customer Locust script | `$CUSTOMER_TEST_ROOT/contract/locustfile-request-count-0916.py` |
| Script SHA256 | `f01146774dd3b05b6a105918f444cec03ed9d5575e2c92d40ac58522bb327850` |
| Dataset | `$CUSTOMER_TEST_ROOT/input/llm_test_datasets-prod/*.json` |
| Dataset files | 3997 |
| Dataset aggregate SHA256 | `c774104d35e521baddb4dc7d57646ac9aac68eeedb57c6962f9d1c85cbe0d7d8` |
| Frozen Locust image | local `locust-awcloud:1.5` |
| Frozen image ID | `sha256:7df3afaaaf1cca96eb9898ba7f23d9c5e4f531d86e8c02ab9f2f488eb9448631` |
| Test-only port | 6568 for Locust master/worker coordination |

Do not copy the private script or dataset into this Git repository. On the
captured source host, the runner defaults directly to:

```text
/data/hxh/0823/deepseek_v4_iter092_final
```

Therefore the exact local test needs only the already loaded frozen image:

```bash
export LOCUST_IMAGE='locust-awcloud:1.5'
export LOCUST_IMAGE_ID='sha256:7df3afaaaf1cca96eb9898ba7f23d9c5e4f531d86e8c02ab9f2f488eb9448631'
```

On another host, override the captured root:

```bash
export CUSTOMER_TEST_ROOT='/absolute/path/to/the/captured/test/environment'
```

The expected layout under `CUSTOMER_TEST_ROOT` is:

```text
$CUSTOMER_TEST_ROOT/
├── contract/locustfile-request-count-0916.py
└── input/llm_test_datasets-prod/*.json
```

If the two inputs do not share a root, export their absolute paths directly:

```bash
export LOCUST_SCRIPT='/absolute/path/locustfile-request-count-0916.py'
export DATASET_DIR='/absolute/path/llm_test_datasets-prod'
```

The test container mounts both host paths read-only. Results and logs still go
under this repository's ignored `runs/` directory.

The normalized dataset aggregate is calculated as:

```bash
dataset_dir=${DATASET_DIR:-$CUSTOMER_TEST_ROOT/input/llm_test_datasets-prod}
(
  cd "$dataset_dir"
  find . -maxdepth 1 -type f -name '*.json' -print0 |
    sort -z |
    xargs -0 sha256sum
) | sha256sum
```

### B2. Verify the local formal test inputs

```bash
./bin/verify_test_inputs.sh
```

This checks the external script path and SHA256, all 3997 external dataset
files and their aggregate SHA256, and the exact local Locust image ID. It does
not copy private inputs, prepare the service, or replace the running service.

If `locust-awcloud:1.5` is not present, transfer it from the captured host:

```bash
# Captured source host
docker save -o locust-awcloud-iter092.tar locust-awcloud:1.5
sha256sum locust-awcloud-iter092.tar

# Test host
docker load -i locust-awcloud-iter092.tar
docker image inspect locust-awcloud:1.5 --format '{{.Id}}'
```

### B3. Optional public compatibility client

When the frozen local image is unavailable, build the public compatibility
client and explicitly select compatibility mode:

```bash
./bin/build_locust_image.sh
export LOCUST_IMAGE='locust-awcloud:iter092-rebuild'
export LOCUST_IMAGE_ID=
./bin/verify_test_inputs.sh
```

`Dockerfile.locust` starts from the digest-pinned public
`python:3.12.11-slim-bookworm` image and installs the observed Python
environment from `requirements-locust.lock`. The lock SHA256 is:

```text
7c6d030fc6b3547c84b6b78a5a0626e2479c0e17e225b47e2e4f2b024b67aecb
```

The rebuilt client is independently buildable from public sources, but PyPI
wheels are not vendored, so its Docker manifest ID is not expected to remain
byte-identical across BuildKit exports.

### B4. Run the accepted workload sequence

```bash
# Read-only check of test inputs and the already running service:
./bin/run_iter092_repro.sh --preflight-only

# Three C60 warmup blocks, idle fence, then twenty formal C60 blocks:
./bin/run_iter092_repro.sh
```

The runner fails closed unless `state/active_service.json`, both service
containers, all four endpoints, the pinned image ID, and the accepted profile
agree. It does not invoke `restore_iter092_service.sh`.

The run takes roughly 35 minutes. New artifacts are written under `runs/` and
ignored by Git. Random-with-replacement sampling and per-request thinking
selection mean the exact value 144.3996 is not deterministic. The target is
approximately 144 ten-thousand TPM with TPOT no greater than 20.00 ms.

The public rebuilt Locust client produces new compatibility evidence, not the
same frozen-image evidence as the historical result. For promotion-quality
evidence, perform three independent formal runs and require every run to pass
the 20.00 ms TPOT gate.

## Optional compatibility wrappers

To prepare both images in one command:

```bash
./bin/build_images.sh
```

To verify both service and test inputs in one command:

```bash
./bin/verify_external_inputs.sh
```

These aggregate wrappers are conveniences only. Neither independent workflow
uses `verify_external_inputs.sh`.
