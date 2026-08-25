# MI325X DeepSeek-V4 Iteration-92 reproduction

This is the public, dependency-only handoff for the accepted C60 result:

- total throughput: 144.3996 ten-thousand tokens/minute (1,443,996 TPM)
- TPOT: 19.70 ms
- client TTFT: 361.94 ms
- scored completions: 6000, failures: 0
- hardware: 8 x AMD Instinct MI325X (`1002:74a5`, `gfx942`)

The public handoff intentionally excludes the model, customer dataset, frozen
customer Locust script, image archives, raw request logs, and the full 92-step
optimization ledger.

## Pinned runtime

The accepted service used:

```text
rocm/atom-dev@sha256:a5cfa1ab503af6e0f55e0ed83cd7e999edbea6940a38dba44aaeee9a22758976
ATOM  1e7659fde32eeaa0d9aa868c3e90847e5e46a51c
AITER eb84cb02200b1707f1076edf3f4930d3626adfb2
```

`Dockerfile` is deliberately layer-free. Build it to fetch and locally tag the
exact accepted runtime without changing its image ID:

```bash
docker build --pull -t rocm/atom-dev:iter092-replay .

docker image inspect rocm/atom-dev:iter092-replay --format '{{.Id}}'
```

Expected ID:

```text
sha256:a5cfa1ab503af6e0f55e0ed83cd7e999edbea6940a38dba44aaeee9a22758976
```

Validate the embedded source revisions:

```bash
docker run --rm --entrypoint bash rocm/atom-dev:iter092-replay -lc '
  test "$(git -C /app/ATOM rev-parse HEAD)" = \
    1e7659fde32eeaa0d9aa868c3e90847e5e46a51c
  test "$(git -C /app/aiter-test rev-parse HEAD)" = \
    eb84cb02200b1707f1076edf3f4930d3626adfb2
  python -c "import atom, aiter; print(atom.__file__); print(aiter.__file__)"
'
```

## External private inputs

Provision these without committing them to the public repository:

| Dependency | Required location or identity |
|---|---|
| Model | `/data/DeepSeek-V4-Flash-FP8` |
| Customer Locust script | `contract/locustfile-request-count-0916.py` |
| Script SHA256 | `f01146774dd3b05b6a105918f444cec03ed9d5575e2c92d40ac58522bb327850` |
| Dataset | `input/llm_test_datasets-prod/*.json` |
| Dataset files | 3997 |
| Dataset aggregate SHA256 | `c774104d35e521baddb4dc7d57646ac9aac68eeedb57c6962f9d1c85cbe0d7d8` |
| Locust image | `locust-awcloud@sha256:7df3afaaaf1cca96eb9898ba7f23d9c5e4f531d86e8c02ab9f2f488eb9448631` |

The dataset aggregate is calculated as:

```bash
(
  cd input/llm_test_datasets-prod
  find . -maxdepth 1 -type f -name '*.json' -print0 |
    sort -z |
    xargs -0 sha256sum
) | sha256sum
```

The frozen Locust image contains Python 3.12.11, Locust 2.37.4, gevent
24.11.1, requests 2.32.3, NumPy 2.3.3, and Flask 3.1.1. The versions are also
listed in `requirements-locust.txt`; rebuilding a client from that list is a
compatibility client, not the frozen formal client image.

If the Locust image is not available from a registry, transfer it from the
source host:

```bash
# Source host
docker save -o locust-awcloud-iter092.tar locust-awcloud:1.5
sha256sum locust-awcloud-iter092.tar

# Reproduction host
docker load -i locust-awcloud-iter092.tar
docker image inspect locust-awcloud:1.5 --format '{{.Id}}'
```

## Host prerequisites

- 8 unpartitioned MI325X GPUs in SPX/NPS1 mode, exposed as `gfx942`
- `/dev/kfd` and `/dev/dri`
- Docker access
- commands: `curl`, `jq`, `rg`, `sha256sum`, `rocm-smi`, `amd-smi`,
  `journalctl`, and `ss`
- free ports: 18000, 18001, 18080, and 6568
- CPU governor `performance`, NUMA balancing disabled, THP `madvise`, GPU
  performance level `perf_determinism`

Do not copy IRQ numbers between hosts. Recreate the GPU/NUMA affinity policy
from the target host topology.

## Verify external inputs

After placing the private inputs:

```bash
./bin/verify_external_inputs.sh
```

This check fails closed unless `amd-smi` reports exactly eight MI325X devices
with PCI identity `1002:74a5`, target `gfx942`, and SPX/NPS1 partitioning.

## Start the service

```bash
./bin/verify_external_inputs.sh
./bin/restore_iter092_service.sh

curl -fsS http://127.0.0.1:18000/v1/models | jq .
curl -fsS http://127.0.0.1:18001/v1/models | jq .
curl -fsS http://127.0.0.1:18080/v1/models | jq .
curl -fsS http://127.0.0.1:18080/router/status | jq .
```

The launcher creates two independent TP4 replicas on GPUs 0-3 and 4-7, then a
least-connections router on port 18080. It uses MTP2, C60, dense graph buckets,
the M102 A8W8 overlay, FlyDSL MoE sorting, no prefix caching, and Quick Reduce
INT4. Although FP8 KV was requested, the accepted logs show effective BF16 KV.

## Reproduce the accepted run sequence

The faithful sequence is cold service restore, three C60 warmup blocks, a
three-sample idle fence, and twenty C60 formal blocks:

```bash
# Optional read-only check before starting the client workload:
./bin/run_iter092_repro.sh --preflight-only

# Run only the warmup, idle fence, and formal client workload:
./bin/run_iter092_repro.sh
```

Service lifecycle and client testing are intentionally separate. The test
script never starts, stops, or replaces the service. It fails closed unless
`state/active_service.json`, both running containers, all four HTTP endpoints,
the pinned image ID, and the accepted Iteration-92 profile agree. Run
`./bin/restore_iter092_service.sh` from the preceding section first.

The run takes roughly 35 minutes. New artifacts are written below `runs/` and
are ignored by Git. The random-with-replacement workload and per-request
thinking choice mean the exact value 144.3996 is not deterministic; the target
is approximately 144 ten-thousand TPM with TPOT no greater than 20.00 ms.

For promotion-quality evidence, execute three independent formal runs and
require every run to pass the 20.00 ms TPOT gate.
