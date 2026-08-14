# MI325X acceptance gates

## Contents

1. Scope and tiers
2. Identity and host
3. PCIe, XGMI, RAS
4. Functional validation
5. Official performance thresholds
6. Cluster and workload gates
7. Reporting

## 1. Scope and tiers

Choose `POC_READINESS`, `FORMAL_NODE_ACCEPTANCE`, or `FORMAL_CLUSTER_ACCEPTANCE`. Formal acceptance requires an approved maintenance window and the current AMD/OEM recipe. The AMD guide estimates hours for AGFHC, TransferBench, rocHPL, RCCL, and workload tests; do not launch them on shared nodes without approval.

Record authorized host/IP/account, expected physical/logical GPUs, bare-metal/host/guest role, OEM bill of materials, current users/workloads, privileges, and allowed changes.

## 2. Identity and host

Read-only identity:

```bash
lspci -nn -d 1002:74a5
amd-smi list -e --json
amd-smi static --asic --vram --driver --board --partition --json
rocminfo
```

Formal current MI325X page expects eight `74a5` devices for the standard UBB platform, supported OS/ROCm, at least 2.5 TiB host memory, and exact required boot arguments. Treat OEM-specific deviations as `REVIEW`; never modify boot arguments during discovery.

Verify `/dev/kfd`, render nodes, user groups, container runtime, host memory, hugepages, IOMMU/ACS/NUMA state, Python/frameworks, partition mode, storage, and `/dev/shm`.

## 3. PCIe, XGMI, RAS

```bash
lspci -d 1002:74a5 -vvv
amd-smi topology --json
amd-smi xgmi --link-status
amd-smi metric --ecc --ecc-blocks --json
amd-smi bad-pages --json
dmesg --level=err,warn
```

The current MI325X acceptance page expects PCIe 32 GT/s, width x16, and no `FatalErr+`. Require intact expected XGMI links. Capture ECC/RAS and bad pages before/after stress; new uncorrectable errors, memory corruption, link degradation, or device loss are hard stops.

## 4. Functional validation

Use the exact current platform files:

- RVS `gpup_single.conf`
- MI325X `gst_single.conf`, `iet_single.conf`, `pebb_single.conf`
- memory, PEQT, and PBQT tests
- AGFHC/CVS levels for the chosen acceptance scope
- bounded real workload after microbenchmarks

Installing tools or changing configs is mutable. Present package/source, command, duration, impact, output path, and rollback first.

## 5. Official performance thresholds

As an evidence snapshot checked 2026-08-11, the current AMD MI325X acceptance page publishes these standard-platform gates:

| Test | Current published pass criterion |
|---|---|
| TransferBench all-to-all | at least 32.9 GB/s |
| TransferBench P2P unidirectional | at least 33.9 GB/s |
| TransferBench P2P bidirectional | at least 43.9 GB/s |
| RCCL 8-GPU all-reduce, 8B–8GiB | bus bandwidth at least 304 GB/s |
| rocBLAS FP32 recipe | at least 94,100 GFLOP/s |
| rocBLAS BF16 recipe | at least 130,600 GFLOP/s |
| rocBLAS INT8 recipe | at least 162,700 GFLOP/s |

TransferBench config tests and BabelStream have additional published thresholds. Read the live MI325X page and execute its exact commands; do not transplant these numbers to a different GPU count, partition mode, VM, message size, or OEM topology. If the live page changes, the live page wins.

## 6. Cluster and workload gates

Before multi-node work:

- pass every node independently;
- verify unique host/IP/BDF inventory and consistent stack/digests;
- map GPU↔NIC↔NUMA locality;
- verify RDMA links, MTU, GID, routes, drivers/firmware, and switch policy;
- run OFED read/write/send bandwidth/latency along local and switched paths;
- run multi-node RCCL for real collective/message sizes;
- run a bounded distributed workload soak.

The current AMD guide recommends 400G-class backend NICs to avoid bottlenecks and Llama 3.1 405B/JAX as a cluster-validation workload. Adapt to validated NIC hardware and the actual POC; do not guess contractual network thresholds.

## 7. Reporting

For every gate record `PASS`, `FAIL`, `REVIEW`, or `NOT_RUN`, exact command/tool version, start/end time, raw evidence, threshold source/date, deviations/owner, and remediation. Never infer `PASS` from a later workload.
