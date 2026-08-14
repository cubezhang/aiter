# Inventory and POC portal mapping

## Resource inventory

Map each verified host/IP to `ResourceNode`: FQDN, management/data addresses, hardware role, OS, online/maintenance state, labels, agent version, and snapshot URI. Never derive IP from an SSH alias; verify remote `hostname -f`, `hostname -I`, interface addresses and routes.

Map physical MI325X devices to `AcceleratorDevice`: serial, BDF, device ID `74a5`, `gfx942`, HBM, firmware, partition/NPS, health and node. Map logical partitions separately with parent physical BDF and partition ID.

## Environment baseline

Persist ROCm/driver/kernel/firmware, Python/framework/engine/AITER/RCCL, image digest, variables, launch/config, topology/partition, applied patches, preflight and acceptance status. Create a new baseline when a frozen field changes.

## Test template and run

Template: workload type, model/dataset/workflow revision, shapes, batch/concurrency, parallelism, warmup/repetitions, SLOs, correctness and required metrics.

Run: project, nodes/GPU set, baseline/template, actual command/config, timestamps/status, metrics/correctness, raw artifacts, MLflow/EvalScope IDs and conclusion.

Use `poc-result.schema.json` as the portable artifact.

## Storage and audit

- POC portal: ownership, approvals, lifecycle, canonical links
- EvalScope: applicable standardized tests
- MLflow: parameters, metrics and comparisons
- NAS/NFS/object store: raw snapshots, logs, telemetry, traces, checkpoints and reports

Persist operator, target, before/after, reason, approval, command, rollback and validation for changes. Never store plaintext SSH, HF, registry, VPN or network credentials.
