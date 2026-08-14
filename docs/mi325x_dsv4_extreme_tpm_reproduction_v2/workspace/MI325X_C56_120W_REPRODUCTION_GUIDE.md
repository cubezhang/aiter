# MI325X DeepSeek-V4 120W TPM 复现指南

## 目标与边界

本流程复现当前已验证的优先目标：

```text
Total TPM >= 1,200,000 tokens/min
Mean TTFT < 2,000 ms
Mean TPOT < 20 ms
Success Rate >= 99.5%
每个有效请求 Completion Tokens == 1024
Manifest 完整，Cached Tokens == 0
```

默认执行固定 750-request screening 两轮，不等同于完整 3997-request × 5
Cold Restart 合同验收。双 TP4 TPM imbalance 会完整披露，但按当前约定不作为
“先达成 120W”阶段的失败条件。

## 已验证部署参数

```text
Topology                 2 × TP4
Backend A                GPU 0,1,2,3 / 127.0.0.1:18000
Backend B                GPU 4,5,6,7 / 127.0.0.1:18001
Router                   least-connections / 127.0.0.1:18080
Checkpoint               DeepSeek-V4-Flash-FP8
KV cache                 FP8
Quick Reduce             INT4
MTP                      2 speculative tokens
Max model length         131072
Max batched tokens       131072
Max sequences            128
GPU memory utilization   0.83
Attention prefill chunk  32768
Prefix cache             disabled
CUDAGraph buckets        1,2,4,8,16,24,25,26,27,28,29,30,31,32,48,64,72,96,128
Warmup                   C53, fixed screen_750 manifest
Measurement              C56, fixed screen_750 manifest, 2 repeats
```

## 执行

当前若没有服务占用 18000、18001、18080：

```bash
cd /data/mi325_0811/mi325_dsv4_opt_0811
./tools/run_120w_reproduction.sh repro_20260812_01
```

若当前运行的是任务所属 `e016repro` 服务，并明确需要由新部署替换：

```bash
STOP_EXISTING_TAG=e016repro \
REPEATS=2 \
./tools/run_120w_reproduction.sh repro_20260812_01
```

脚本不会自动停止未知容器，也不会覆盖已有结果目录。默认测试成功后保留新服务；
使用 `KEEP_SERVICE=0` 可在生成报告后停止并移除本次服务。

## 阶段与输出

脚本依次执行：

1. 检查必要命令、模型、Manifest、补丁、端口和 digest-pinned 镜像。
2. 保存 GPU、拓扑、RAS、内核、Docker、ATOM commit/worktree diff 和输入哈希。
3. 部署双 TP4 backend 与 least-connections Router，并等待健康。
4. 执行 C53 warmup；warmup 证据保留，但不计最终汇总。
5. 连续执行 C56 测量轮，每个 Manifest 条目恰好尝试一次，禁止 retry 隐藏失败。
6. 每轮保存逐请求 JSONL、服务日志增量、GPU 遥测、RAS、kernel delta、Router 前后状态。
7. 生成 campaign JSON、Markdown、TSV 和 SHA256。

默认输出目录：

```text
/data/models/mi325_dsv4_extreme_tpm_20260812/11_c56_120w_reproduction/<CAMPAIGN_TAG>/
```

关键文件：

```text
campaign_config.json             实际部署与门槛参数
input_sha256.txt                 模型配置、Manifest、补丁、脚本等输入哈希
atom_source_commit.txt           ATOM commit
atom_source_status.txt           ATOM dirty worktree 状态
atom_source_worktree.patch       实际部署源码相对 commit 的 patch
service_inspect_deployed.json    部署容器、镜像、mount、环境变量
<WARMUP_LABEL>/                  warmup 原始证据
<MEASURE_LABEL>/                 每轮逐请求与系统证据
campaign_summary.json            机器可读总报告
campaign_summary.md              人工阅读报告
campaign_summary.tsv             表格结果
output_sha256.txt                汇总报告哈希
```

进程退出码：

```text
0   所有测量轮通过当前优先目标及质量门槛
2   至少一轮未通过；报告和测试后证据仍会生成，失败候选服务自动停止
64+ 参数、资产、镜像、端口或运行环境预检失败
```

## 切换完整 Manifest

在双方确认完整合同计划后，可以显式切换为 3997 请求；不要把它与默认 screening
结果混为同一等级证据：

```bash
MEASURE_MANIFEST=/data/models/mi325_dsv4_extreme_tpm_20260812/02_manifests/generated/full_3997.jsonl \
REPEATS=3 \
STOP_EXISTING_TAG=e016repro \
./tools/run_120w_reproduction.sh c56_full3997_3x
```

正式 5 轮 Cold Restart 验收还要求每轮前重建服务，不能只把 `REPEATS` 改为 5；
应在确认合同版本后使用独立 final-acceptance 编排。
