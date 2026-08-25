# CUSTOMER_LOCUST_LONGRUN_V1

本合同是 MI325X / DeepSeek-V4-Flash-FP8 后续长程优化的唯一性能测试合同。它从客户 2026-08-14 实际使用的通用 Locust 方法派生，不使用、混入或换算此前 strict 3997/固定 1024 输出的结果。

## 冻结输入

- 客户脚本：`locustfile-request-count-0916.py`
- 脚本 SHA256：`f01146774dd3b05b6a105918f444cec03ed9d5575e2c92d40ac58522bb327850`
- 数据集：3997 个 JSON 文件，随机有放回抽样
- 数据集归一化聚合 SHA256：`c774104d35e521baddb4dc7d57646ac9aac68eeedb57c6962f9d1c85cbe0d7d8`
- Locust 镜像：`locust-awcloud@sha256:7df3afaaaf1cca96eb9898ba7f23d9c5e4f531d86e8c02ab9f2f488eb9448631`
- 运行环境：Python 3.12.11、Locust 2.37.4、gevent 24.11.1、requests 2.32.3
- 模型请求名：`DeepSeek-V4`
- `max_tokens=16384`，不设置 `ignore_eos`，由模型自然 EOS
- `include_usage=true`
- 每个请求由冻结脚本以 `random.random() < 0.4` 独立决定是否开启 thinking
- 命令行保留 `--thinking-enabled false` 以复现客户界面标签；该标签不改变上述逐请求 40% 随机 payload 行为
- 8 个 Locust worker 进程；每个测试点使用全新 Locust 容器/进程
- 并发数 `C` 同时作为 `--users`、`--spawn-rate` 和脚本 `--interval`

## 唯一计分规则

每个测试点从一个全新的 Locust 进程开始。冻结脚本每累计 `5*C` 个完成请求形成一个 `5x` 分段；取日志中第一次出现的：

`历史均值--5x (20 次统计--统计间隔 C)`

以及它紧随其后的 `Total_TPM` 行，作为该点唯一得分。因此得分是前 20 个完整 `5x` 分段 TPM 的算术平均，覆盖恰好前 `100*C` 个进入完整分段的完成请求。

- 命中计分行后立即、优雅停止本测试点的精确 Locust 容器。
- 停止期间完成的在途请求、最终 drain、重复打印的历史均值均不进入得分。
- 尾部不完整分段不计分，但原始日志、CSV 和请求统计全部保留。
- 客户 123.0312W 是本合同的来源锚点，但它当时只含 11 个 `5x` 分段；新合同的 20 段结果不得与其伪装成完全同口径复测。

## 有效性与排序

一个测试点只有在以下条件全部满足时才有效：

- 精确命中首次 20 段计分行，且统计间隔与请求并发一致；
- Locust failure count 为 0；
- 测试前后 RAS/ECC 计数不变；
- 测试窗口没有新增 GPU page fault、reset、XGMI fatal 或 amdgpu fatal；
- 8 张卡仍为 MI325X/gfx942、SPX/NPS1、`perf_determinism`；
- 服务模型、镜像、源码、启动参数和容器身份均有证据。

有效点以 `Total_TPM` 为吞吐排序主指标，同时保存 Input/Output TPM、TTFT、TPOT、E2E、输入/输出长度和并发。`TPOT <= 23 ms` 只作为低延迟边界标签，不作为吞吐极限点的有效性门槛。

## 搜索与接受纪律

- 先在客户原始冻结服务重建 C56 新基线，再做并发粗扫；不得用 strict 结果选择并发。
- 所有服务候选均使用本合同做同并发 A/B；短跑只能诊断，不能晋升。
- 晋升候选至少三次独立新进程复验，并报告均值、最差值、标准差和 CV。
- 最终保留三条边界：最大有效吞吐、低延迟平衡点、长稳/冷启动可复现点。
- 每次只改变可归因的一组变量；服务切换、失败、部分结果和负结果都保留。
- GPU/服务异常立即停止当前点并隔离，不把异常样本纳入性能结论。
- 任务没有按天数或 Terminal 会话定义的截止时间；控制面会持续读取后续 wave，直到人工明确结束。

## 资源和安全边界

- 独占 8 张 GPU；只启停本 campaign 精确识别的容器。
- 不修改宿主机驱动，不重启宿主机，不修改功耗/时钟，不改变 NPS 或 accelerator partition。
- 所有新增证据写入 `/data`；根分区空间紧张时禁止拉取新镜像或构建大层。
- 旧客户 `locust-8089` 容器和交互式 Locust 进程不属于本 campaign，禁止触碰。
