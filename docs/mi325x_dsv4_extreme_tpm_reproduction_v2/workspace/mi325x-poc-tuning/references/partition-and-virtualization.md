# Partitioning and virtualization

## Contents

1. Two partition dimensions
2. Discovery
3. Workload selection
4. Change control
5. SR-IOV boundaries

## 1. Two partition dimensions

MI325X has eight XCDs. Accelerator partitioning groups compute into logical devices:

- SPX: one logical GPU across all XCDs
- DPX: two logical GPUs
- QPX: four logical GPUs
- CPX: eight logical GPUs, one XCD each

Only use modes the device/firmware reports; not every ASIC exposes every nominal profile.

Memory partitioning controls HBM NUMA layout:

- NPS1: all eight HBM stacks interleaved
- NPS2: two groups of four stacks
- NPS4: localized groups
- NPS8: defined by APIs but support is device/firmware-specific

Compute and memory modes are separate but constrained combinations. Query capabilities rather than relying on a static matrix.

## 2. Discovery

Read-only/current-state queries may require elevated visibility on some hosts:

```bash
amd-smi static --partition --json
amd-smi partition --current
sudo amd-smi partition --memory
sudo amd-smi partition --accelerator -g 0
numactl --hardware
amd-smi topology --json
```

Record physical BDF plus partition ID. All logical partitions of one physical GPU share the Bus:Device identity; a simple device count can overstate physical accelerators and HBM.

## 3. Workload selection

- SPX/NPS1 is the safest first full-GPU baseline for large models and conventional multi-GPU jobs.
- CPX/localized NPS may benefit many independent bandwidth-bound jobs or explicit locality, but requires partition-aware rank and memory placement.
- DPX/QPX can balance isolation and per-job capacity.
- Never select partitioning solely to increase visible device count.

Compare identical workloads with physical-resource-normalized metrics and account for scheduler isolation, cache/HBM locality, graph compilation, and per-process overhead.

## 4. Change control

Partition changes are maintenance operations:

1. enumerate supported profiles and compatible NPS modes;
2. capture current profile, processes, topology, serial/BDF mapping, and rollback profile;
3. stop all workloads on the entire physical GPU/hive;
4. obtain approval for the exact `amd-smi set` command;
5. memory partition changes require driver reload; this affects all GPUs in the hive;
6. re-enumerate devices and rebuild handles/rank maps;
7. rerun identity, RAS, XGMI, RCCL, and workload correctness.

In AMD SMI 7.13+, the prior reset shortcut is removed; current docs use an explicit `amdgpu` module reload or library driver-reload API. Never guess the command for a pinned version.

## 5. SR-IOV boundaries

MI325X supports SR-IOV and up to 64 partitions at product level. A guest can query limited partition information but cannot change host settings; reported guest modes may intentionally hide the host profile. Distinguish physical function, virtual function, mVF, and bare-metal evidence.

Pin host driver, guest driver/user space, hypervisor, firmware, VF profile, IOMMU, and resource assignment. Check release-specific warnings; ROCm 7.2.4 explicitly warns MI325X KVM SR-IOV users against amdgpu 30.20.0. Recheck the current compatibility matrix before deployment.
