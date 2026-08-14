# Network, RCCL, and collective playbook

## Contents

1. Inventory and topology
2. Single-node baseline
3. Multi-node RDMA
4. RCCL and workload communication
5. MoE and disaggregation
6. Tuning and acceptance

## 1. Inventory and topology

For every node record verified management/data IPs, FQDN, NIC/BDF/RDMA device, link speed/state, firmware/driver, MTU, NUMA node, GPU BDF/serial and nearest NIC. Avoid interface-name assumptions. Preserve route and GID output.

Pass each node's hardware acceptance before cluster tests. Enforce identical software/container digests and workload artifacts across nodes.

## 2. Single-node baseline

Run RCCL tests on the exact physical GPU group and partition mode:

- `all_reduce_perf`
- `all_gather_perf`
- `reduce_scatter_perf`
- `broadcast_perf`
- `sendrecv_perf`
- `alltoall_perf` and `alltoallv_perf` for MoE/EP

Sweep message sizes representative of model layers and communication buckets. Save algorithm/protocol/channel decisions, bus and algorithm bandwidth, rank mapping, topology dump, and errors.

The official MI325X standard-node acceptance uses an 8-GPU all-reduce threshold at large message size. Apply it only to the exact official command and topology.

## 3. Multi-node RDMA

Validate in layers:

1. link/routing and RDMA state;
2. NIC-local host-memory bandwidth/latency;
3. NIC-to-NIC through switch;
4. GPU-to-NIC and GPU-to-GPU with ROCm memory;
5. all node pairs/diameter;
6. multi-node RCCL;
7. workload.

Use `ib_write_bw`, `ib_send_bw`, `ib_write_lat`, `ib_send_lat`, and `ib_read_lat` per current AMD guidance. The guide recommends 400G-class backends to avoid bottlenecks, but actual pass criteria come from the validated NIC/OEM/network design.

Changing ACS, PFC/DCQCN/ECN, routing, MTU, GID, IRQ/RPS, hugepages, or driver settings is a shared-infrastructure change requiring network-owner approval and rollback.

## 4. RCCL and workload communication

Do not treat `NCCL_*` names as NVIDIA-only; RCCL supports NCCL-compatible variables, but each variable must be current and evidence-based. Start from defaults/topology discovery. Tune one group:

- rank-to-GPU/NIC/NUMA binding;
- interface/HCA/GID selection;
- channels/protocols/algorithms;
- communication buffer sizes;
- compute/communication overlap;
- collective fusion/bucket sizes;
- TP/PP/DP/EP/CP group layout.

Correlate microbenchmarks with actual collective traces and end-to-end scaling. Peak all-reduce cannot explain MoE all-to-all or pipeline bubbles.

## 5. MoE and disaggregation

Before EP/MoE acceptance:

- verify all-to-all and all-to-all-v;
- measure expert-token imbalance and dropped/rerouted tokens;
- verify dispatcher/combine correctness;
- profile overlap and host gaps;
- compare RCCL baseline with MORI/DeepEP only when exact MI325X/gfx942/NIC support is documented and smoke-tested.

For prefill/decode disaggregation or tiered KV, pin orchestration and transport, separate control/data planes, measure KV transfer time and failure recovery, and compare with colocated serving. Current public MORI recipes may target MI355X; do not infer MI325X support from the framework name.

## 6. Tuning and acceptance

Measure bandwidth/latency distributions, tail outliers, retransmits/errors, collective time, overlap, job throughput, scaling efficiency, and failure recovery. Accept only when every node/link remains healthy and the workload improves without correctness or stability loss.
