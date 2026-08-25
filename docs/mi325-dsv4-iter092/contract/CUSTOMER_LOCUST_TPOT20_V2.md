# CUSTOMER_LOCUST_TPOT20_V2

本合同是 `CUSTOMER_LOCUST_LONGRUN_V1` 的 SLO 收紧版本。V1 的原始结果和
最大吞吐证据全部保留；V2 不回写或重解释 V1 结果。

## 冻结负载与计分口径

- 客户原始 `locustfile-request-count-0916.py`、数据集、请求体、随机有放回
  抽样、40% thinking 概率、`max_tokens=16384`、自然结束、8 个 Locust
  进程和客户端镜像全部保持冻结。
- 每个测试点从全新的 Locust 进程开始。
- 唯一计分值仍是第一次出现的 20 个完整 `5x` 历史均值及其紧随的
  `Total_TPM` 行，覆盖恰好前 `100*C` 个进入完整分段的完成请求。
- 停止后的 drain、尾部不完整分段、失败请求删除或任何 strict 结果均不计分。

## TPOT 20.00ms 硬门槛

- `TPOT <= 20.00 ms` 才能成为 V2 的 SLO 合格点；`TPOT > 20.00 ms` 为
  `SLO_FAIL`，不得晋升、不得参与 V2 最佳 TPM 排序。
- 判定使用客户日志计分行中的原始两位小数值，并以 `20.00` 为包含等号的
  上限；不得把 20.01ms 四舍五入或容差放宽成合格。
- 吞吐排序只在完整性门槛全部通过且 `TPOT <= 20.00ms` 的点之间进行。
- 单轮贴线不是正式结论。晋升候选至少三次独立新 Locust 进程复验，三轮
  每一轮均须 `TPOT <= 20.00ms`；报告均值、最差值、标准差和 CV。
- 靠近悬崖的点优先保留可复现余量；若三轮跨越 20.00ms，标记为 cliff
  diagnostic，选择相邻较低负载。

## 有效性与安全

每轮还必须同时满足：Locust failure count 为 0；测试前后 RAS/ECC 不变；
无新增 bad page、GPU page fault/reset/XGMI fatal/amdgpu fatal；8 张卡仍为
MI325X/gfx942、SPX/NPS1、`perf_determinism`；服务、模型、镜像、源码、启动
参数和容器身份证据齐全。

TPOT 超限但完整性健康的测试点保留为有效测量和负证据，控制器记录为
`POINT_SLO_FAIL`，但绝不进入 `best_tpot20`。硬件或正确性失败则隔离。

## 搜索顺序

1. 复用当前已验证服务，在 C52、C54、C56、C58 建立合法/非法区间。
2. 测试相邻 C53、C55、C57，解析双副本对称性和并发悬崖。
3. 对 C52-C58 各做三个全新客户端进程的 20 段测量；按三轮全通过选择
   当前服务的最高 TPM 点。
4. 再以获胜并发为控制，逐项测试 MTP 深度、batch/prefill 和图 bucket，
   每次只改变一个可归因变量。
5. 最终执行冷启动复现和长稳验证。

任务继续由 systemd 控制面运行，不依赖 Terminal 会话；关键状态同步到
`logs/progress.log` 和 `logs/monitor.log`。
