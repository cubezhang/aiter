# 新增最高优先级目标：SLO约束下的极限TPM攻坚

## 前置条件
1、测试数据集所在位置：locust容器内部， /root/locust_test/llm_test_datasets-prod
2、原来我们已经优化过的测试脚本和测试日志保存地址（容器外）：/data/models/
3、原来我们已经优化过的测试指引：mi325x_dsv4_flash_tp4_e004_e117_reproduction.mdtion.md

## 一、增加第二条独立性能赛道

除原有：

```text
MI325X 4卡 VS H200 4卡公平性能对标
```

之外，新增：

# Extreme Throughput Lane

目标：

> 在完全固定真实POC数据集、生成长度和采样规则的前提下，以 TTFT 和 TPOT 为硬约束，将系统可持续稳定承载的 TPM 推到最高。

当前已知基线：

```text
Topology:
2 × TP4 Instance

Dataset:
固定官方POC Dataset

Average Input:
≈ 6000 tokens

Output:
1024 tokens

Current TPM:
≈ 1,120,000 TPM
```

当前基线定义为：

```text
EXTREME_BASELINE = 1.12M TPM
```

后续所有优化均以此为起点计算增益。

---

# 二、最终优化目标重新定义

本赛道不再以：

```text
最大Raw Throughput
```

为目标。

而是：

```text
Maximize TPM
```

Subject to：

```text
TTFT < 2000 ms
TPOT < 20 ms
Success Rate >= 99.5%
Output Length保持原始测试定义
Dataset不变
Sampling不变
模型不变
精度不变
```

形式化目标：

```text
TPM_max =
MAX(TPM)

Subject To:

TTFT_SLO < 2s
TPOT_SLO < 20ms
Error Rate <= 0.5%
No Dataset Modification
No Sampling Degradation
No Artificial Cache Hit
No Output Truncation
```

---

# 三、延迟SLO不能只看平均值

为了防止：

```text
平均TTFT = 1.8s
但P99 = 8s
```

这种“吞吐看起来很高、实际POC不可用”的结果，本次同时记录：

```text
TTFT Mean
TTFT P50
TTFT P90
TTFT P95
TTFT P99

TPOT Mean
TPOT P50
TPOT P90
TPOT P95
TPOT P99
```

正式极限值至少要求：

```text
Mean TTFT < 2s
Mean TPOT < 20ms
```

同时把：

```text
P95 TTFT
P95 TPOT
```

作为关键稳定性指标。

优先争取：

```text
P95 TTFT < 2s
P95 TPOT < 20ms
```

如果当前业务基线尚无法做到P95同时满足，则必须明确区分：

```text
Mean-SLO Max TPM
```

和：

```text
P95-SLO Max TPM
```

禁止混为一个数字。

---

# 四、核心方法改变：寻找“SLO悬崖”

本次Extreme Lane最重要的工作不是直接改Kernel。

第一步必须画出：

# Throughput-Latency Saturation Curve

即：

```text
Offered Load
      ↑
      |
TPM ↑ |
      |
      |               X  ← Latency Cliff
      |            XXX
      |         XXX
      |      XXX
      |   XXX
      |XXX
      +------------------------→ Concurrency
```

对于每一个Concurrency / Request Rate，记录：

```text
TPM
TTFT
TPOT
P95 TTFT
P95 TPOT
Queue Depth
GPU Busy
HBM BW
Batch Size
KV Usage
```

---

# 五、首先找到当前112万TPM对应的真实极限点

禁止假设：

```text
112万TPM = 当前硬件最大能力
```

它可能只是：

```text
当前Benchmark参数产生的性能点
```

首先对现有Known Good配置进行负载扫描。

例如：

```text
Concurrency = C0 × 0.75
Concurrency = C0 × 0.90
Concurrency = C0
Concurrency = C0 × 1.10
Concurrency = C0 × 1.20
Concurrency = C0 × 1.30
```

具体步长根据当前Locust设置决定。

目标找到：

```text
TPOT开始突破20ms的位置

以及

TTFT开始突破2s的位置
```

然后在临界区间进行更细粒度搜索。

例如：

```text
70
75
80
85
90
95
100
```

而不是全范围暴力Sweep。

最终确定：

```text
CURRENT_SLO_CLIFF
```

例如：

```text
Concurrency 82
TPM 1.12M
TTFT 1.83s
TPOT 19.4ms
```

这才是真正的Extreme Baseline。

---

# 六、以后所有优化只评价一个核心指标

定义：

# SLO-Constrained Max TPM

即：

> 在不突破TTFT 2秒和TPOT 20ms约束的情况下，系统能够稳定达到的最大TPM。

今后不能再简单比较：

```text
Config A @ Concurrency 70

VS

Config B @ Concurrency 70
```

因为一个好的优化可能：

```text
Concurrency 70性能只提升1%

但

最大可承载Concurrency：
70 → 90
```

这才是真正有价值的Serving优化。

所以每个Winner最终必须回答：

```text
它把SLO Cliff向右推了多少？
```

---

# 七、Extreme Lane的阶段目标

以当前：

```text
1.12M TPM
```

为基线。

设置：

## Level 1

```text
1.20M TPM
```

相当于突破当前基线约：

```text
+7%
```

---

## Level 2

```text
1.25M TPM
```

约：

```text
+12%
```

这是本次必须重点争取的第一目标。

---

## Level 3

```text
1.30M TPM
```

约：

```text
+16%
```

属于高价值目标。

---

## Level 4

```text
1.35M TPM
```

约：

```text
+20%
```

属于攻坚目标。

---

## Stretch Goal

```text
1.40M TPM+
```

相对于1.12M意味着约25%的系统级有效容量提升。

只有在：

```text
TTFT < 2s
TPOT < 20ms
```

仍成立时才算有效。

禁止为了达到1.4M而突破Latency SLO。

---

# 八、Extreme Lane调优顺序完全重新排列

Extreme Lane优先级定义为：

```text
P0：Batch / Scheduler / Queue
P0：双TP4实例负载均衡
P0：Graph / CPU Launch Bubble

P1：KV Cache管理
P1：Prefill / Decode调度
P1：MoE / MLA / GEMM Kernel
P1：TP4内部通信

P2：Router调度
P2：CPU / NUMA
P2：RCCL overlap

P3：GTT / TTM / HugePage
```

原因：

当前任务是：

```text
6000 Input
+
1024 Decode
+
高并发
```

因此真正需要优化的是：

> 单位时间内整个Serving Pipeline能够持续处理多少有效Token，同时又不让排队延迟和Decode间隔突破SLO。

---

# 九、第一主战场：Batch Size与Scheduler

这是本赛道最高ROI区域。

重点参数：

```text
max_num_seqs
max_num_batched_tokens
prefill batch
decode batch
chunked prefill
scheduler delay
scheduler policy
queue size
decode scheduling
```

目标不是简单：

```text
Batch越大越好
```

而是寻找：

```text
TPM最高
且
TTFT < 2s
且
TPOT < 20ms
```

的临界Batch。

---

# 十、关键策略：Batch要运行在Latency Cliff附近

如果Batch过小：

```text
GPU未吃满
TPM低
TPOT很好看
```

属于：

```text
Under Utilization
```

如果Batch过大：

```text
GPU吃满
TPM可能继续增加
但TTFT / TPOT爆炸
```

属于：

```text
Over Saturation
```

我们真正寻找：

```text
SLO Sweet Spot
```

即：

```text
GPU接近满载
+
Batch接近最大
+
Queue可控
+
TTFT略低于2s
+
TPOT略低于20ms
```

例如最终可以故意运行在：

```text
TTFT ≈ 1.6～1.9s
TPOT ≈ 17～19.5ms
```

而不是浪费性能运行在：

```text
TTFT = 300ms
TPOT = 8ms
```

因为本任务明确要求：

> 在满足SLO的情况下最大化TPM。

只要真实业务SLO没有被突破，就应该充分使用Latency Budget。

---

# 十一、第二主战场：2 × TP4负载平衡

当前存在：

```text
TP4 Instance A
TP4 Instance B
```

必须单独记录：

```text
A TPM
B TPM

A Queue Depth
B Queue Depth

A GPU Busy
B GPU Busy

A TTFT
B TTFT
```

检查是否存在：

```text
Instance A = 610K TPM
Instance B = 510K TPM
```

这种不均衡。

如果存在，即使单个GPU Kernel已经很快：

> 整体仍然损失大量TPM。

---

# 十二、Router必须纳入核心性能路径

必须Profile：

```text
Client
 ↓
Router
 ↓
TP4 A
TP4 B
```

统计：

```text
Router Queue Time
Dispatch Time
Backend Queue Depth
Backend Active Requests
Backend Batch Size
```

测试Router策略：

```text
Round Robin

Least Outstanding Requests

Least Queue Depth

Weighted Load

Latency-aware Routing
```

但不得根据Prompt内容做作弊式分流。

Router只能依据：

```text
负载
队列
运行状态
```

进行合法调度。

目标：

> 两个TP4实例始终尽可能同时处于最佳Saturation Point。

---

# 十三、不要追求每个TP4自身最高TPM

需要优化：

```text
TPM_Total =
TPM_TP4_A
+
TPM_TP4_B
```

有时：

```text
A = 590K
B = 590K
```

比：

```text
A = 650K
B = 490K
```

更有价值。

因此本次最重要的资源指标之一：

```text
TP4 Load Imbalance %
```

定义：

```text
abs(A - B) / ((A+B)/2)
```

目标：

```text
< 5%
```

争取：

```text
< 2%
```

---

# 十四、第三主战场：Prefill与Decode竞争

当前工作负载特点：

```text
Average Input ≈ 6000
Output = 1024
```

属于：

> Prefill负载明显存在，同时Decode持续时间也很长。

因此最容易出现：

```text
Prefill抢占Decode资源
```

导致：

```text
TPOT突然恶化
```

或者：

```text
Decode优先级过高
```

导致：

```text
TTFT爆炸
```

所以需要Profile：

```text
Prefill GPU Time
Decode GPU Time
Prefill Queue
Decode Queue
```

寻找最佳资源平衡。

---

# 十五、Chunked Prefill作为P0实验

如果Framework支持当前DeepSeek-V4路径下稳定使用：

重点测试：

```text
Chunked Prefill
```

目标：

> 避免6000-token Prompt一次性Prefill阻塞Decode。

通过适当拆分Prefill，可以：

```text
降低Decode阻塞
→ TPOT下降
→ 允许提高Concurrency
→ TPM提高
```

因此评价Chunked Prefill时：

禁止只看单请求Prefill速度。

必须看：

```text
SLO-Constrained Max TPM
```

是否提高。

---

# 十六、Graph Capture评价方法改变

Graph优化的真正价值可能不是：

```text
GPU Kernel本身快5%
```

而是：

```text
减少CPU Launch Bubble
→ Decode间隙下降
→ TPOT下降
→ 可提高Batch
→ TPM提高
```

因此Graph相关实验重点记录：

```text
GPU idle gap
CPU launch latency
Graph hit rate
Eager fallback
TPOT
```

---

# 十七、Kernel优化必须围绕TPOT Budget

对于：

```text
1024 Decode Tokens
```

TPOT是非常敏感的。

因此优先寻找：

```text
每一个Decode Step都会执行
```

的热点Kernel。

比只在Prefill阶段执行一次的低占比Kernel更重要。

Profiler必须分别输出：

```text
PREFILL TOP KERNELS
```

以及：

```text
DECODE TOP KERNELS
```

不要混在一个Top GPU Time表中。

---

# 十八、特别关注DeepSeek-V4 MoE路径

需要分析Decode阶段：

```text
Router / Top-K
Expert Selection
MoE GEMM
Expert Communication
Reduce
```

各自GPU时间。

因为1024 token生成意味着这些路径会被执行大量次数。

如果每一步减少：

```text
几十微秒
```

累计到：

```text
1024 Decode Steps
×
大量并发请求
```

可能成为可观的TPM增益。

---

# 十九、通信优化评价标准

不要只跑：

```text
RCCL Bandwidth Benchmark
```

然后认为通信优化成功。

真正的判断标准：

```text
TPOT下降了吗？
```

以及：

```text
SLO Cliff提高了吗？
```

重点记录：

```text
Compute Time
Communication Time
Exposed Communication Time
Hidden Communication Time
```

核心目标：

> 减少Exposed Communication。

而不是仅仅提高理论通信带宽。

---

# 二十、Extreme Lane实验流程

每个Experiment按以下流程：

```text
1. 修改一个优化点

2. Correctness Check

3. 使用原始Dataset和原始Sampling进行快速E2E Screening

4. 判断TPM / TTFT / TPOT方向

5. 若无收益：
   Rollback

6. 若有明显收益：
   重新寻找该配置新的SLO Cliff

7. 记录新的SLO-Constrained Max TPM

8. 加入Winner Stack
```

因此：

> 每次有效优化后，都必须重新寻找最大Concurrency。

禁止一直固定旧Concurrency测试到最后。

---

# 二十一、为什么必须重新搜索Concurrency

例如：

Baseline：

```text
Concurrency = 80

TPM = 1.12M
TTFT = 1.85s
TPOT = 19.2ms
```

优化Kernel后：

```text
Concurrency = 80

TPM = 1.16M
TTFT = 1.60s
TPOT = 17.2ms
```

此时不能说：

```text
优化收益只有3.6%
```

因为系统还有Latency Budget没有使用。

继续提高：

```text
Concurrency = 90
```

可能达到：

```text
TPM = 1.25M
TTFT = 1.88s
TPOT = 19.3ms
```

真正的Serving Capacity增益实际是：

```text
1.12M
→
1.25M
```

这才是本项目要寻找的收益。

---

# 二十二、建立自动SLO Search程序

建议专门开发：

```text
tools/find_max_tpm_under_slo.py
```

输入：

```text
TTFT_LIMIT=2000
TPOT_LIMIT=20
```

自动完成：

```text
启动负载
↓
逐渐增加Concurrency / Request Rate
↓
观察TTFT/TPOT
↓
接近SLO
↓
缩小搜索步长
↓
找到最大合法负载
```

采用：

```text
Coarse Search
+
Binary/Fine Search
```

而不是人工逐个试。

最终自动输出：

```text
Max Legal Concurrency
Max Legal Request Rate
Max TPM
TTFT
TPOT
P95 TTFT
P95 TPOT
```

---

# 二十三、Extreme Performance Ledger

额外增加：

```text
SLO_MAX_TPM
```

字段。

例如：

| Exp | Optimization | Fixed-load TPM | SLO Max TPM | TTFT | TPOT | Max Concurrency |
|---|---|---:|---:|---:|---:|---:|
| E000 | Baseline | 1.12M | 1.12M | 1.83s | 19.2ms | 80 |
| E012 | Scheduler | 1.15M | 1.21M | 1.89s | 19.1ms | 87 |
| E019 | Graph | 1.18M | 1.25M | 1.84s | 18.9ms | 91 |
| E026 | MoE | 1.21M | 1.31M | 1.91s | 19.4ms | 96 |

真正排序依据：

```text
SLO Max TPM
```

而不是：

```text
Fixed-load TPM
```

---

# 二十四、10小时重新分配

## 0～0.5h

```text
环境冻结
+
3轮112万基线复现
```

---

## 0.5～1.25h

# 建立SLO Saturation Curve

找出：

```text
当前真正Max TPM
当前Latency Cliff
当前最佳Concurrency
```

---

## 1.25～2h

# Profile + Historical Winner Diff

确定：

```text
Prefill瓶颈
Decode瓶颈
Scheduler瓶颈
Router瓶颈
Communication瓶颈
```

---

## 2～4.5h

# P0攻坚

集中：

```text
Scheduler
Batch
Chunked Prefill
Graph
KV
双TP4负载均衡
```

目标：

```text
1.20M～1.25M
```

---

## 4.5～6.5h

# Decode Hot Path攻坚

集中：

```text
MoE
MLA
Top-K
GEMM
Sampling
Communication
```

目标：

```text
1.25M～1.30M+
```

---

## 6.5～7.5h

# 双TP4 / Router / RCCL优化

目标：

```text
减少TP4不均衡
降低Queue
增加总系统Capacity
```

---

## 7.5～8.75h

# Winner Interaction Search

组合：

```text
Best Scheduler
+
Best Batch
+
Best Graph
+
Best Kernel
+
Best Router
```

每形成新Winner：

重新运行：

```text
SLO Search
```

目标：

```text
1.30M～1.35M+
```

---

## 8.75～10h

# Final Maximum TPM Search

使用完整固定POC Dataset：

在：

```text
TTFT < 2s
TPOT < 20ms
```

边界附近精细寻找：

```text
MAX TPM
```

最后连续：

```text
5轮
```

验证最终性能。

---

# 二十五、最终极限验收必须Cold Restart

最终配置必须：

```text
停止两个TP4 Container
↓
重新启动
↓
重新加载模型
↓
正常Warm-up
↓
启动Router
↓
运行完整Dataset
```

禁止依赖特殊长期Warm状态。

---

# 二十六、Extreme Lane最终报告

单独输出：

```text
MI325X_DSV4_EXTREME_TPM_REPORT.md
```

报告第一页直接显示：

```text
Topology:
2 × TP4

Dataset:
XXXXXXXX

Avg Input:
≈ 6000

Output:
1024

Sampling:
XXXXXXXX

Baseline:
1.12M TPM

Final:
X.XXM TPM

Gain:
+XX.X%

TTFT:
XXXX ms

TPOT:
XX.XX ms

P95 TTFT:
XXXX ms

P95 TPOT:
XX.XX ms
```

并绘制：

```text
Concurrency
VS
TPM
```

以及：

```text
TPM
VS
TTFT
```

```text
TPM
VS
TPOT
```

三条曲线。

---

# 二十七、核心判断原则

从现在开始，每做一个优化，都不要只问：

```text
TPM提高了吗？
```

必须问：

```text
这个优化是否降低了每个Token的资源成本？

是否降低了Scheduler / Queue / Kernel / Communication开销？

是否释放了Latency Budget？

能不能因此提高Concurrency？

提高Concurrency以后，新的SLO Max TPM是多少？
```

最终唯一核心评价指标：

# SLO-Constrained Maximum TPM

而不是：

# Raw Maximum TPM

---

# 二十八、最终目标排序

本项目目标优先级：

```text
Priority 1

保持：
TTFT < 2s
TPOT < 20ms
真实Dataset
真实Sampling
1024 Output
```

↓

```text
Priority 2

最大化TPM
```

↓

```text
Priority 3

降低P95/P99和性能波动
```

↓

```text
Priority 4

提高MI325/H200性能比例
```

只要某项优化：

```text
TPM ↑
但
TTFT > 2s
```

或者：

```text
TPOT > 20ms
```

立即判定：

```text
INVALID FOR EXTREME LANE
```

不进入Winner Stack。

---

# 二十九、最终攻坚思想

当前112万TPM不是终点。

我们真正要寻找的是：

> **这套2×TP4 MI325X系统，在平均输入约6000 token、输出1024 token的真实POC业务下，能够在TTFT < 2秒和TPOT < 20ms时稳定承载的物理性能边界究竟在哪里。**

本轮性能攻坚的核心不是：

> 让一次Benchmark跑得最快。

而是：

> **不断降低每个请求、每个Decode Step以及每个Batch消耗的系统资源，从而让系统在相同Latency Budget中容纳更多并发请求，最终把SLO Performance Cliff尽可能向右推。**