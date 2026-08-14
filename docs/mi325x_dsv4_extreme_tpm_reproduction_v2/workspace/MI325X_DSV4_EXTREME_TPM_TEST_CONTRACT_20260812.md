# MI325X DeepSeek-V4 Extreme TPM 最终测试合同

合同版本：`1.0`  
冻结时间：`2026-08-11 UTC`  
计划验收日期：`2026-08-12 UTC`  
执行环境：单机独占，8 × AMD Instinct MI325X  
主目标：在真实固定 POC 工作负载和硬延迟 SLO 下最大化可持续总 TPM

本合同是 2026-08-12 Extreme Throughput Lane 的唯一验收口径。合同冻结后，模型、数据集、请求生成规则、采样规则、输出长度、精度、拓扑、TPM 公式、延迟公式或通过门槛发生变化时，结果不得与本合同下的基线直接比较；必须创建新合同版本并重新建立基线。

---

## 1. 权威性能数字与单位

所有首页、Ledger、图表和口头结论必须优先写完整数值，禁止只写容易混淆的 `W`、`M` 或“万”。

| 对象 | 权威 TPM | 中文辅助写法 | 用途 |
|---|---:|---:|---|
| MI325 历史基线 | `1,120,000 tokens/min` | 112 万 TPM | 本轮历史增益锚点 |
| H200 对照 | `1,380,000 tokens/min` | 138 万 TPM | 只作背景对照，不参与本机 SLO 搜索 |

历史 MI325/H200 比例为 `1,120,000 / 1,380,000 = 81.16%`。

历史 `1,120,000 tokens/min` 并非由本合同“强制生成满 1024 tokens”的完整口径产生，因此本轮必须另建严格口径基线 `E000_STRICT`。最终同时报告：

```text
Gain_vs_Historical = Final_SLO_TPM / 1,120,000 - 1
Gain_vs_E000_Strict = Final_SLO_TPM / E000_STRICT_SLO_TPM - 1
```

不得用历史基线替代 `E000_STRICT`，也不得隐去两者的工作负载差异。

---

## 2. POC 类型、权限与变更边界

本轮分级为 `POC_READINESS`，不是 AMD/OEM Formal Node Acceptance。

已授权：

- 本机整晚独占使用，8 张 GPU 均可用于本任务。
- 停止、启动和重建任务范围内的 E004、Router、Locust 与实验容器。
- 创建任务专用容器、脚本、日志、配置、补丁和 profiler 产物。
- 修改实验代码并保留精确 patch、commit、checksum 与回滚方法。
- 在任务范围内连续运行负载、正确性、稳定性与只读遥测。

未经额外批准禁止：

- 修改 GPU 时钟、Performance Level 或功耗上限。
- 修改驱动、ROCm、固件、BIOS、内核或启动参数。
- 修改 SPX/DPX/QPX/CPX、NPS 或 SR-IOV 模式。
- 修改 NUMA balancing、IOMMU、HugePage、GTT/TTM 等系统级状态。
- 修改网络、MTU、防火墙、路由、ACS、RDMA/PFC/ECN 策略。
- 删除或覆盖无关的用户文件、历史日志和现有 dirty worktree 修改。

出现需要上述操作的候选时，先记录为 `REQUIRES_APPROVAL`，不得擅自执行。

---

## 3. 硬件与平台冻结

### 3.1 身份

| 字段 | 冻结值 |
|---|---|
| 物理 GPU 数量 | `8` |
| GPU | `AMD Instinct MI325X` |
| PCI ID | `1002:74a5`，共 8 个设备 |
| LLVM/GFX target | `gfx942` |
| Accelerator partition | 全部 `SPX` |
| Memory partition | 全部 `NPS1` |
| GPU 0–3 NUMA affinity | NUMA node 0 |
| GPU 4–7 NUMA affinity | NUMA node 1 |
| GPU 间链路 | 8 卡均由 XGMI 互连；正式运行前重新采集状态 |
| Host OS | Ubuntu 24.04 |
| Kernel | `6.8.0-136-generic` |
| Host ROCm user space | `7.2.4` |
| AMD GPU driver | `6.16.13` |
| IOMMU | 内核命令行包含 `iommu=pt` |
| NUMA balancing | `kernel.numa_balancing=0` |

正式测试前后必须采集：PCIe link、XGMI、ECC/RAS、bad pages、dmesg、GPU 温度/功耗/时钟、HBM、CPU/NUMA、磁盘空间和容器状态。出现新 uncorrectable ECC、bad-page 增长、XGMI/PCIe 降级、GPU reset、内存损坏或设备丢失时立即停止性能工作。

### 3.2 容器与运行时

| 组件 | 冻结值 |
|---|---|
| ATOM 镜像 | `rocm/atom-dev:nightly_202607271535` |
| ATOM 镜像 digest | `sha256:66df2fb1c537f52d47f2d3bf973b8da15278dd2ee362b5307e3d29dfad69fa7d` |
| Locust 镜像 | `locust-awcloud:1.5` |
| Locust 镜像 digest | `sha256:7df3afaaaf1cca96eb9898ba7f23d9c5e4f531d86e8c02ab9f2f488eb9448631` |
| ATOM commit | `5e491a90bd7e860518e0026d38c0ec1101dfb4a8` |
| ATOM API compat patch SHA256 | `548eb7ce398a0adc4034e4cd1303ff4c12e8bee123b09a0f9b099c051b302318` |
| AITER image commit | `242fee05fcdb0781f9d338c03b194c96fc5e6d03` |
| PyTorch | `2.10.0+rocm7.2.4.git3d3aa833` |
| Torch HIP | `7.2.53211` |
| Locust | `2.37.4`，8 workers |

夜版镜像虽已用 digest 固定，但 native ATOM 的公开通用支持口径未明确把 MI325X 列为独立平台，因此初始证据级别为 `GFX942_INFERRED`。只有在本机算子预检、正确性、稳定性和固定工作负载全部通过后，最终候选才可标为 `MI325X_MEASURED`。

---

## 4. 模型与精度冻结

### 4.1 模型

| 字段 | 冻结值 |
|---|---|
| 模型路径 | `/data/DeepSeek-V4-Flash-FP8` |
| Served model name | `DeepSeek-V4` |
| Architecture | `DeepseekV4ForCausalLM` |
| Model type | `deepseek_v4` |
| 权重 shard | 46 个，共 `294,038,841,472` 字节 |
| 46-shard 聚合 SHA256 | `9193f2ac37312214784d7113ee81dc1e1126c3e4dfbd4bd0ebdd9600409bfdc3` |
| `config.json` SHA256 | `52b5a1aa87606cb5be4f3158d706594edb1c4ce97ce6b1cd6079f15df075d7f5` |
| `model.safetensors.index.json` SHA256 | `7e975ba3bef8947a94e7da0abd60888375b232b4dfad883d59653e65c6ba522a` |
| `tokenizer.json` SHA256 | `8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf` |
| `tokenizer_config.json` SHA256 | `0277d76786fdb1f25e766db8ec0439a5b05caeb93d4d96098600810496faa3a4` |
| `chat_template.jinja` SHA256 | `ac6fcd0bda270d198fa360cad4f4b20a3191812477fd4abc886aa365f174b86c` |

46-shard 聚合哈希算法为：在模型目录内按相对文件名排序，对每个 `model-*.safetensors` 计算 SHA256，再对完整 `sha256sum` 输出计算 SHA256。

### 4.2 精度维度

| 维度 | 冻结值 |
|---|---|
| Checkpoint storage | FP8，block size `[128,128]` |
| FP8 format | E4M3 FNUZ 路径，scale format `ue8m0` |
| Activation scheme | dynamic |
| Weight/activation compute | 当前验证的 A8W8 blockscale 路径 |
| KV cache | FP8 |
| Collective quantization | AITER Quick Reduce INT4 |
| MTP | 2 speculative tokens |

禁止替换模型、量化元数据、精度、KV 格式或采样质量来换取吞吐。任何精度或 collective quantization 变化必须成为独立实验，先通过正确性，且不得直接与冻结基线混为同一配置。

### 4.3 已验证的叠加资产

| 资产 | SHA256 |
|---|---|
| C57 `topk_per_row_kernels.cu` | `ea047ba7b951deea107dcf79c951587d1ba233354362cfb2a2c8e0848625900c` |
| Least-connections `router.py` | `eeab689806ad68c7c8f61d3cd6513b71b86f1765e0435381749a4a1334de8c1c` |
| A8W8 tuning CSV | `af374befa66082ef02d2799ca9f7d58c70f5dffe6187774907f826e86eab5d9f` |
| E004 startup script | `cc5cb916eeda56075bde1886a3e99516f298239b5a34c30389e3aeaa45c5889d` |
| E004 stop script | `bde8bdf129481c46de91a3b0450eee507fba0d014c54b0f3ccd802b93c41627f` |

---

## 5. 服务拓扑与 E000 起始配置

```text
Locust client, 8 workers
        |
        v
Least-connections Router :18080
        |-------------------------|
        v                         v
TP4 A :18000                  TP4 B :18001
GPU 0,1,2,3                  GPU 4,5,6,7
NUMA 0                       NUMA 1
```

E000 起始配置：

| 参数 | 值 |
|---|---|
| Topology | `2 × TP4` |
| `max-model-len` | `131072` |
| `max-num-batched-tokens` | `131072` |
| `max-num-seqs` | `128` |
| `gpu-memory-utilization` | `0.83` |
| Chunked Prefill | 已启用 |
| Prefill chunk size | `16384` |
| Long prefill threshold | `0` |
| Scheduler delay factor | `0.0` |
| Prefix cache | 禁用 |
| CUDAGraph | 启用 |
| Capture sizes | `[1,2,4,8,16,24,32,48,64,72,96,128]` |
| Router | least connections |

当前基线已经启用 Chunked Prefill，因此“仅开启 Chunked Prefill”不得算作新优化。只有参数、调度或实现发生明确变化且 SLO Max TPM 提升时，才可作为候选。

---

## 6. 数据集冻结与确定性 Manifest

### 6.1 官方数据集

| 字段 | 冻结值 |
|---|---|
| 容器内路径 | `/root/locust_test/llm_test_datasets-prod` |
| JSON 文件数 | `3997` |
| 总字节数 | `104,950,488` |
| JSON 解析 | 3997/3997 有效 |
| 文件名记录长度均值 | `6083.36` tokens，仅作预期参考 |
| Manifest 文件 SHA256 | `2af4007801256b2f652f963b8a64f957ed86536054bf1707654a20e652c2469a` |
| 内容哈希集合 SHA256 | `815a66cb3ca54b276d5fed9ffbdcc0312be612df0f8e95e8998a1fdbd100d620` |

每轮实际 `prompt_tokens` 以服务端 usage 为准，必须报告 Mean/P50/P90/P95/P99/Min/Max。不得修改、重写、裁剪或按 prompt 内容进行有利分流。

### 6.2 Manifest 生成

固定种子字符串为：

```text
MI325X-DSV4-EXTREME-20260812-v1
```

为避免 Python RNG 版本差异，顺序和 thinking 标记使用 SHA256 排序，不依赖进程随机数：

```text
order_key(file) = SHA256("order\0" + seed + "\0" + relative_filename)
think_key(file) = SHA256("think\0" + seed + "\0" + relative_filename)
```

- 完整 Manifest：全部 3997 个文件按 `order_key` 升序排列，每个文件恰好出现一次。
- 完整 Manifest Thinking：按 `think_key` 选最小的 1599 个文件为 thinking，比例 `1599/3997 = 40.005%`；其余 2398 个为 non-thinking。
- Screening Manifest：从完整顺序中选定 750 个固定请求，按独立 `screen` key 形成稳定、无重复的样本；按独立 thinking key 固定 300 个 thinking、450 个 non-thinking，恰好 40%。
- Manifest 一旦生成即写入 JSONL，记录相对文件名、文件 SHA256、顺序号、thinking 标记和 manifest SHA256。E000 与所有候选必须使用完全相同的对应 manifest。

Screening 结果只能用于淘汰或晋级，不能作为正式 SLO Max TPM。Winner、最终 Cliff 和验收数字必须来自完整 3997 请求 Manifest。

---

## 7. 请求与采样合同

每个性能请求使用 `/v1/chat/completions`、HTTP streaming，并固定：

```json
{
  "model": "DeepSeek-V4",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    "<原始数据集消息，原样追加>"
  ],
  "stream": true,
  "stream_options": {"include_usage": true},
  "max_tokens": 1024,
  "ignore_eos": true,
  "temperature": 1.0,
  "top_p": 1.0,
  "top_k": -1,
  "n": 1
}
```

Thinking 规则：

- Manifest 标记为 thinking 的 40% 请求增加 `"chat_template_kwargs":{"thinking":true}`。
- Non-thinking 请求保持历史脚本行为，不传 `chat_template_kwargs`。
- 不按 prompt 内容、长度、预期难度或历史耗时决定 thinking。

其他冻结规则：

- 不传 `stop`，并设置 `ignore_eos=true`，确保每个成功请求实际完成 1024 completion tokens。
- 不修改 temperature/top-p/top-k，不把采样改为 greedy。
- ATOM 当前 Chat API 虽声明 `seed` 字段，但该版本没有把 request seed 传入 `SamplingParams`；因此不宣称输出文本逐 token 可复现。请求集合、thinking 分配、采样参数和输出 token 数仍完全冻结。
- 禁止 retry 隐藏失败。若诊断需要重试，原始 attempt 仍计为失败，重试单独记录且不得加入正式 TPM。
- Prefix cache 必须关闭；每轮记录配置和 `cached_tokens`，发现非零人工缓存命中则该轮无效。

---

## 8. 负载模型与搜索方法

使用 closed-loop concurrency：每个 Locust 用户在前一个请求完成后立即发送下一个请求，`wait_time=0`。Concurrency 是首要 offered-load 控制量；如后续加入 request-rate 模式，必须作为独立字段和独立曲线，不得与 closed-loop 结果混合。

### 8.1 Warm-up

- 每次 Cold Restart 后先等待两个 TP4 backend 和 Router 健康。
- 使用固定 warm-up manifest 覆盖预定 graph/batch 范围。
- 等待 JIT、graph capture、tuning/cache 活动稳定，并确认无 eager fallback/编译错误。
- Warm-up 请求不计 TPM、TTFT、TPOT 或成功率。
- Warm-up 后重置 client、Router 实验计数与遥测窗口；不得清除或隐藏 RAS/系统错误。

### 8.2 E000 严格基线

从历史 Users=52 附近开始，先对 E004 起始配置执行至少 3 次独立完整 Manifest 测量，建立：

```text
E000_STRICT_FIXED_LOAD_TPM
E000_STRICT_MEAN_SLO_MAX_TPM
E000_STRICT_P95_SLO_MAX_TPM
E000_STRICT_SLO_CLIFF
```

不得假定历史 `1,120,000 tokens/min` 就是严格口径 E000。

### 8.3 Coarse + Fine Search

初始粗扫以 C0=52 为中心，建议点：

```text
39, 47, 52, 57, 62, 68
```

根据第一个违反 TTFT、TPOT 或成功率的点确定合法/非法区间，再用步长 2–3，最后步长 1 搜索。无需测试明显远离临界区的全部并发。

每个配置：

1. Correctness preflight。
2. 固定 750-request Screening，至少 3 次。
3. 无收益或违反门槛立即 rollback。
4. 有明确收益则用完整 3997-request Manifest 重新搜索 SLO Cliff。
5. 新 Winner 重新运行其最大合法负载，禁止只在旧 concurrency 比较。

### 8.4 稳定合法点

普通正式点至少 3 次完整 Manifest：

- 每一轮都必须满足 Mean-SLO 与成功率门槛，才是 Mean-SLO Legal。
- 每一轮还必须满足 P95-SLO，才是 P95-SLO Legal。
- 正式 TPM 报告 3 轮中位数，同时报告每轮值、Min/Max、标准差和 CV。
- TPM CV >3% 或边界结论不一致时扩展到 5 轮。

Final Winner 必须按第 13 节执行 Cold Restart 后 5 轮完整验收。

---

## 9. 指标定义

### 9.1 TPM

本项目 TPM 是成功请求的输入 token 与输出 token 之和：

```text
Valid_Total_Tokens = Σ(prompt_tokens + completion_tokens), successful requests only
Measurement_Minutes = (last acceptance request completed - first acceptance request dispatched) / 60
Total_TPM = Valid_Total_Tokens / Measurement_Minutes
Input_TPM = Σ(prompt_tokens) / Measurement_Minutes
Output_TPM = Σ(completion_tokens) / Measurement_Minutes
```

测量窗口包含服务端排队和客户端持有的固定并发负载，不包含 warm-up，也不得剔除慢请求、尾部 drain 或合法长 prompt。

`SLO-Constrained Maximum TPM` 是满足相应 SLO 的最高稳定合法并发点的正式 Total TPM 中位数。

### 9.2 TTFT

```text
TTFT = 首个真实生成 token 到达客户端的时间 - HTTP 请求开始时间
```

首 token 必须是包含非空 `delta.content` 或 `delta.reasoning_content` 的生成事件，不能把 role-only、空 delta、usage 或 `[DONE]` 当作首 token。TTFT 包含本机 Client→Router→Backend 的 dispatch 与服务端排队。

### 9.3 TPOT

```text
TPOT = (最后一个生成 token 到达时间 - 第一个生成 token 到达时间) / (completion_tokens - 1)
```

不得把最终 usage、HTTP close 或 `[DONE]` 尾部时间加入最后 token 时间。若流式 chunk 聚合多个 token，以 usage 的 `completion_tokens=1024` 为分母，并保留 chunk/usage 一致性诊断。

### 9.4 E2E 与 Success

```text
E2E = 完整响应结束时间 - 请求开始时间
Success Rate = valid_successful_requests / attempted_requests
```

有效成功请求必须同时满足：

- HTTP 200。
- SSE 可完整解析并正常结束。
- 最终 usage 存在且数值有效。
- `completion_tokens == 1024`。
- 输出非空，无 NaN/Inf、乱码损坏或协议异常。

### 9.5 分位数

必须保存逐请求原始记录，并直接计算：

```text
TTFT Mean/P50/P90/P95/P99/Min/Max
TPOT Mean/P50/P90/P95/P99/Min/Max
E2E  Mean/P50/P90/P95/P99/Min/Max
```

禁止用“50 请求平均值”“五分钟平均值”或 Locust synthetic metric 的分位数冒充逐请求分位数。

---

## 10. 硬 SLO 与判定

所有比较使用未四舍五入的原始浮点值；展示时四舍五入不改变通过/失败结论。

### 10.1 Mean-SLO Legal

每一轮必须同时满足：

```text
Mean TTFT < 2000 ms
Mean TPOT < 20.000 ms
Success Rate >= 99.5%
Output tokens per valid request == 1024
```

任一条件失败，该轮为 `INVALID_FOR_EXTREME_LANE`。

### 10.2 P95-SLO Legal

每一轮必须同时满足：

```text
P95 TTFT < 2000 ms
P95 TPOT < 20.000 ms
Success Rate >= 99.5%
Output tokens per valid request == 1024
```

最终必须分开报告：

```text
Mean-SLO Max TPM
P95-SLO Max TPM
```

P95 不满足时不得把 Mean-SLO 数字命名为统一的“SLO Max TPM”。

### 10.3 其他硬门槛

- 不得修改 Dataset、模型、Sampling、Thinking 比例或精度。
- 不得人工制造 cache hit。
- 不得截断输出或选择性丢弃慢请求。
- 不得把失败请求产生的 token 计入有效 TPM。
- 两个 TP4 backend 都必须实际处理请求。
- TP4 load imbalance 目标 `<5%`，争取 `<2%`：

```text
Imbalance = abs(A_TPM - B_TPM) / ((A_TPM + B_TPM) / 2)
```

- 无新 uncorrectable ECC、bad pages、XGMI fault、GPU reset、OOM、NaN/Inf 或数据损坏。
- 正式性能轮不得附带 profiler；profiled 数据只用于诊断。

---

## 11. Correctness 与稳定性门槛

每个实验先完成：

1. 两个 `/v1/models` 和 Router health/status 检查。
2. 32 个冻结 golden prompts 的 greedy correctness suite；保存 E000 输出与哈希。
3. Streaming、usage、1024-token、thinking/non-thinking、长 prompt 冒烟。
4. 代表性 Attention/MLA、MoE/Top-K、A8W8 GEMM、norm、sampling 与 TP collective 路径预检。
5. 检查日志中的 backend、graph hit/eager fallback、精度、JIT/ABI 与 NaN/Inf。

候选若改变输出、协议、token usage、模型行为或正确性，即使 TPM 更高也拒绝。随机主负载不要求输出文本逐 token 相同，但采样参数、thinking 标记、输出 token 数和结构正确性必须一致。

稳定性记录：

- 每轮 attempted/success/failed/invalid/cancelled。
- Router 与 backend HTTP errors 的测试前后 delta，禁止使用长寿命累计值代替本轮值。
- GPU HBM、busy、功耗、温度、时钟及 throttle reason。
- ECC、bad pages、XGMI/PCIe、dmesg before/after。
- Client CPU/RSS、RPS、事件循环/worker 饱和。
- Backend A/B 请求数、TPM、active、queue/batch/KV 指标及 imbalance。

---

## 12. 调优与 Winner 管理

优先顺序：

```text
P0 Scheduler / Batch / Queue / 双 TP4 平衡 / Graph launch gaps
P1 KV / Prefill-Decode 调度 / MLA-MoE-GEMM / TP4 communication
P2 Router / CPU-NUMA / RCCL overlap
P3 GTT-TTM-HugePage（需要额外系统变更批准）
```

每个 Experiment 只改变一个声明的 hypothesis group。必须保存：

- Parent Winner ID。
- 完整 command/env/config diff。
- Correctness 和 kernel preflight。
- Fixed-load Screening。
- 新 SLO Cliff 搜索。
- 逐轮 JSON、原始请求 CSV/JSONL、日志、遥测和回滚命令。

Winner 的唯一主排序字段为任务专用的：

```text
slo_goodput_total_tokens_per_minute
```

同时保留兼容展示字段 `SLO_MAX_TPM`。通用比较器必须附加本合同硬 SLO evaluator；通用比较器的默认 5% regression 容忍度不得覆盖 TTFT/TPOT/成功率硬门槛。

候选晋级要求：

- Correctness PASS。
- Stability/RAS PASS。
- Kernel/ABI preflight PASS。
- 所有冻结字段一致，变化字段明确声明。
- SLO Max TPM 增益超过测量噪声。
- 无另一硬指标回退或作弊条件。

---

## 13. 2026-08-12 最终验收流程

最终候选必须执行：

```text
1. 保存当前容器 inspect、命令、镜像、挂载、环境、日志和回滚脚本
2. 停止任务范围内的两个 TP4 service 与 Router
3. 重新创建/启动最终配置
4. 重新加载模型
5. 等待两个 backend 完全 ready
6. 执行固定 warm-up，不计正式指标
7. 启动 Router，确认计数从验收窗口基线开始
8. 采集 RAS/PCIe/XGMI/HBM/温度/功耗 before snapshot
9. 在最终合法 concurrency 上连续运行 5 轮完整 3997-request Manifest
10. 每轮之间确认无残留 active request、无错误增长和无配置漂移
11. 采集 after snapshot
12. 验证全部哈希、原始记录、指标和门槛
```

Final Acceptance 必须全部满足：

- 5/5 轮 Success Rate ≥99.5%。
- 5/5 轮有效请求输出均为 1024 tokens。
- 5/5 轮 Mean TTFT <2000 ms。
- 5/5 轮 Mean TPOT <20.000 ms。
- 若声明 P95-SLO Final，则 5/5 轮 P95 TTFT/TPOT 同时满足对应门槛。
- 配置、模型、数据集、manifest、sampling、thinking 和容器 digest 无漂移。
- 两个 TP4 均参与，负载无异常偏置。
- 无新 RAS/XGMI/PCIe/driver reset/OOM/NaN/Inf。
- 最终 TPM 取 5 轮中位数，同时披露全部 5 个值、Min/Max、标准差和 CV。

只要一轮硬门槛失败，不得选择性删除该轮。必须报告失败、定位原因；只有证明为独立客户端/基础设施异常并完整保留证据时，才可在报告中标注后追加重跑，原失败轮仍不可隐藏。

---

## 14. 证据和存储布局

正式 Run Root：

```text
/data/models/mi325_dsv4_extreme_tpm_20260812/
```

建议目录：

```text
00_contract/
01_platform_snapshot/
02_fingerprints/
03_manifests/
04_correctness/
05_e000_strict/
06_slo_curve/
07_experiments/
08_profiles/
09_winner_stack/
10_final_cold_restart/
11_report/
rollback/
```

职责划分：

- `/data/aiter`：源码与经过版本控制的服务脚本；保留既有 dirty 修改，不直接覆盖。
- 当前工作区：自动搜索工具、合同、Ledger、分析与最终报告。
- `/data/models/mi325_dsv4_extreme_tpm_20260812`：原始实验和验收证据。
- Locust 容器数据集：只读源；复制出的 manifest/数据文件必须保持内容哈希一致。

每个实验至少包含：

```text
metadata.json
result.json              # 符合 poc-result.schema.json
requests.jsonl/csv       # 逐请求原始数据
summary.json
command.sh
environment.txt/json
config.diff
router_before.json
router_after.json
gpu_telemetry.csv
ras_before.json/txt
ras_after.json/txt
server_A.log
server_B.log
client.log
checksums.sha256
rollback.sh 或明确 rollback 指令
```

任何 API key、SSH key、token 或凭据不得写入报告和公开证据。

---

## 15. 报告与图表

最终报告文件：

```text
MI325X_DSV4_EXTREME_TPM_REPORT.md
```

首页固定展示：

```text
Topology: 2 × TP4
Dataset: 3997-file official POC dataset + manifest SHA256
Input: measured prompt-token distribution, mean approximately 6000
Output: exactly 1024 tokens per valid request
Thinking: fixed 40.005% full-manifest assignment
Sampling: temperature=1.0, top_p=1.0, top_k=-1, n=1
Historical MI325: 1,120,000 tokens/min
H200 reference: 1,380,000 tokens/min
E000 Strict: <measured value>
Final Mean-SLO: <measured value>
Final P95-SLO: <measured value or explicitly not achieved>
Gain vs Historical: <measured>
Gain vs E000 Strict: <measured>
TTFT Mean/P95/P99
TPOT Mean/P95/P99
Success Rate
TP4 Imbalance
```

必须绘制并附原始数据：

- Concurrency vs Total TPM。
- Total TPM vs Mean/P95 TTFT。
- Total TPM vs Mean/P95 TPOT。
- Concurrency vs Success Rate。
- TP4 A/B TPM 与 imbalance。
- 必要时 Queue/Batch/KV/GPU busy 与 SLO Cliff 的关联图。

最终报告必须列出所有 rejected/inconclusive 候选及原因，不得只展示 Winner。

---

## 16. 验收签字条件

仅当以下内容齐全时，本合同下的验收数字才有效：

- 合同版本与 SHA256。
- 平台、镜像、模型、数据集、manifest、脚本和配置指纹。
- E000 Strict 三轮或以上原始证据。
- 完整 Saturation Curve 与合法/非法边界。
- Final Cold Restart 五轮原始证据。
- Mean-SLO 与 P95-SLO 分离结论。
- Correctness、Success、RAS/Stability 全部通过。
- 精确回滚和复现命令。
- 最终报告中的 TPM 使用完整 `tokens/min` 数值。

不满足任一硬门槛时，最高 TPM 只能标为 `RAW/INVALID`，不得标为 Extreme Lane Winner。
