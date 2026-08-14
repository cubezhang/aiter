# MI325X DeepSeek-V4 Extreme TPM 测试合同 v1.1 补充条款

生效时间：`2026-08-12 UTC`  
上位合同：`MI325X_DSV4_EXTREME_TPM_TEST_CONTRACT_20260812.md`（v1.0，SHA256 `bd2a31a4333cdc94e50d5e1fbc95c9253a1aabb54ecc95a897944ae648247a79`）

本文件按用户在 2026-08-12 的明确指示创建新合同版本。除下述条款外，v1.0 的模型、数据集、Manifest、请求、采样、输出长度、精度、拓扑、TPM 公式、成功率、正确性、稳定性、RAS、证据保存和禁止事项全部继续有效。原 v1.0 文件及其 SHA256 不修改。

## 1. 主验收目标

主排序目标仍为在硬 SLO 下最大化可持续总 TPM：

```text
Mean TTFT < 2000 ms
Mean TPOT < 20.000 ms
Success Rate >= 99.5%
Output tokens per valid request == 1024
```

满足以上全部条件的稳定合法点按 `slo_goodput_total_tokens_per_minute` 排序，取 TPM 最大者。

## 2. P95/P99 指标降级为观察项

TTFT、TPOT 与 E2E 的 P95/P99 仍须从逐请求原始记录计算并披露，用于诊断尾延迟和解释 SLO cliff，但不再作为本版本的严格验收失败条件。

因此：

- 不要求单独建立 P95-SLO Max TPM。
- 不要求最终 5 轮的 P95 TTFT 或 P95 TPOT 小于 v1.0 的阈值。
- P95/P99 超过 2000 ms 或 20.000 ms 时必须如实报告，但只要 Mean-SLO、成功率、输出长度和其他硬门槛全部通过，该轮仍可判为合法。
- 禁止删除、裁剪或弱化逐请求尾延迟原始证据。

## 3. 最终验收轮次

最终候选仍须 Cold Restart、固定 warm-up，并执行连续 5 轮完整 3997-request Manifest。5/5 轮必须同时满足第 1 节四项硬门槛以及 v1.0 的配置冻结、双 TP4 参与、负载平衡、完整性和 RAS/Stability 门槛。

最终 TPM 仍取 5 轮中位数，并披露五个单轮值、Min/Max、标准差和 CV。P95/P99 作为观察指标随报告附上，不参与 PASS/FAIL。

## 4. 基线与可比性

v1.0 已采集的 E000 Strict 原始轮次可作为历史证据，但 v1.1 的最终验收数字必须来自本补充条款生效后的 Cold Restart 五轮。最终报告必须明确标注 `Contract v1.1 / Mean-SLO acceptance`，不得将其误称为 v1.0 的 P95-SLO Final。
