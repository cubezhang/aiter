# DeepSeek-V4-Flash MI325X：服务启动与正式测试

复现流程已经拆成两个不需要参数的交互式脚本：

```text
01_start_service.sh  # 检查环境并启动 2 × TP4 服务和 Router
02_run_test.sh       # 对已启动服务执行 Warm-up 和 3,997-request 正式单轮
```

两个脚本都会在每个阶段开始前暂停。确认后按 Enter，取消时按 Ctrl-C。

## 一、运行顺序

### 1. 启动服务

```bash
cd /data/models/mi325_dsv4_extreme_tpm_20260812
./01_start_service.sh
```

只有看到下面的输出，才继续测试：

```text
Service START PASS
```

### 2. 执行测试

```bash
cd /data/models/mi325_dsv4_extreme_tpm_20260812
./02_run_test.sh
```

只有看到下面的输出，才表示完整正式单轮通过：

```text
Test REPRODUCTION PASS
```

两个脚本都不接受参数。服务标签和测试标签由脚本自动生成，已有证据不会被覆盖。

## 二、启动脚本做什么

`01_start_service.sh` 依次完成：

1. 检查 Docker 权限和本机固定文件；
2. 确认恰好 8 张 `1002:74a5` / MI325X / `gfx942`；
3. 确认 8 卡均为 `perf_determinism`；
4. 校验 ATOM 镜像 digest、模型分片、ATOM commit、dirty-worktree patch 和部署 Runner；
5. 检查 `18000`、`18001`、`18080` 端口；
6. 如端口被旧测试服务占用，依次尝试识别状态文件服务、原报告服务，或唯一一对名称成对且固定镜像匹配的运行中 `_dual`/`_router` 容器；
7. 停止前先打印两个精确容器名称；此时按 Enter 会立即停止并删除这两个占用任务，按 Ctrl-C 取消；候选为零或多于一对时拒绝停止；
8. 启动 GPU `0,1,2,3` 和 `4,5,6,7` 上的两个 TP4 后端；
9. 启动 least-connections Router；
10. 验证两个后端和 Router 健康；
11. 验证 Requested FP8 KV 在 `gfx942` 上实际回退为 BF16；
12. 保存服务启动证据和活动服务状态。

固定服务配置：

| 项目 | 值 |
| --- | --- |
| Model | `/data/DeepSeek-V4-Flash-FP8` |
| Topology | 2 × TP4 |
| Backend Ports | `18000`、`18001` |
| Router Port | `18080` |
| Router | least-connections |
| Speculative Decoding | MTP2 |
| Attention Prefill Chunk | 32,768 |
| Max Batched Tokens | 131,072 |
| Max Sequences | 128 |
| GPU Memory Utilization | 0.83 |
| Prefix Cache | 关闭 |
| Quick Reduce | INT4 |
| Requested KV | FP8 |
| Effective KV | BF16 |

服务状态写入：

```text
/data/models/mi325_dsv4_extreme_tpm_20260812/15_c56_split_reproduction/active_service.json
```

测试脚本不猜测服务标签，必须读取并重新验证该状态文件。

## 三、测试脚本做什么

`02_run_test.sh` 依次完成：

1. 读取 `active_service.json`；
2. 核对服务/Router 容器名称、容器 ID、镜像 ID和运行状态；
3. 重新检查 MI325X、`gfx942` 和 `perf_determinism`；
4. 核对 Router 策略、两个后端和 BF16 KV 回退；
5. 校验 Locust 镜像、Manifest、Runner、Summarizer 和冻结输入 Hash；
6. 执行 C53 / 750-request Warm-up，排除计分；
7. 执行 C56 / 完整 3,997-request 正式单轮；
8. 生成活动汇总；
9. 验证 124W TPM、Mean TTFT、Mean TPOT、成功率、输出长度、缓存、Thinking、后端平衡、RAS 和错误日志；
10. 确认正式请求排空后两个后端、Router 和容器仍健康且未 OOM；
11. 保存测试证据并保持服务在线。

固定测试合同：

| 项目 | 值 |
| --- | --- |
| Warm-up | C53 / `screen_750.jsonl` / 750 requests |
| 正式轮 | C56 / `full_3997.jsonl` / 3,997 requests |
| 正式轮次数 | 1 |
| Completion Tokens | 每请求严格 1,024 |
| Total TPM | ≥ 1,240,000 |
| Mean TTFT | < 2,000 ms |
| Mean TPOT | < 20 ms |
| 成功率 | ≥ 99.5% |
| Cached Tokens | 0 |
| Thinking | 1,599 / 3,997 |
| Backend Token 不平衡 | < 5% |
| RAS | 前后无变化 |

## 四、证据目录

服务启动证据：

```text
/data/models/mi325_dsv4_extreme_tpm_20260812/15_c56_split_reproduction/services/<服务标签>/
```

测试证据：

```text
/data/models/mi325_dsv4_extreme_tpm_20260812/15_c56_split_reproduction/test_runs/<测试标签>/
```

测试目录中的关键文件：

```text
campaign_config.json
campaign_summary.json
campaign_summary.md
test_input_sha256.txt
test_output_sha256.txt
service_state_before_test.json
<测试标签>_WARMUP_C53/
<测试标签>_C56_R1/summary.json
<测试标签>_C56_R1/requests_worker_*.jsonl
<测试标签>_C56_R1/gpu_telemetry.txt
<测试标签>_C56_R1/kernel_delta.log
<测试标签>_C56_R1/ras_before.txt
<测试标签>_C56_R1/ras_after.txt
```

## 五、重复测试

服务启动成功后，可以多次运行：

```bash
./02_run_test.sh
```

每次都会先执行独立 Warm-up，再创建新的测试标签和证据目录。重复测试不等于冷重启轮次，因为服务没有重启。

需要冷重启复现时，重新依次运行：

```bash
./01_start_service.sh
./02_run_test.sh
```

启动脚本会根据状态文件精确替换上一轮服务。

## 六、停止服务

查看当前服务：

```bash
jq '{service_tag,service_container,router_container}' \
  /data/models/mi325_dsv4_extreme_tpm_20260812/15_c56_split_reproduction/active_service.json
```

确认状态文件内容后，可以精确停止：

```bash
STATE=/data/models/mi325_dsv4_extreme_tpm_20260812/15_c56_split_reproduction/active_service.json
SERVICE=$(jq -r '.service_container' "$STATE")
ROUTER=$(jq -r '.router_container' "$STATE")
docker stop "$ROUTER" "$SERVICE"
docker rm "$ROUTER" "$SERVICE"
```

不要用模糊名称、通配符或批量容器清理命令。

## 七、安全与报告边界

- 两个脚本都不会修改 GPU 时钟、Performance Level、功耗上限、分区、驱动、固件、BIOS、内核、NUMA 或网络配置。
- 硬件身份、镜像、Hash、容器 ID、RAS 或数据完整性不匹配时立即停止。
- Requested KV 是 FP8；本机 MI325X `gfx942` 的 Effective KV 必须按 BF16 报告。
- 正式结果是完整 3,997-request 单轮，不是五轮冷重启 Final Winner 验收。
- P95 TTFT 会披露，但当前硬门是 Mean TTFT < 2,000 ms。
