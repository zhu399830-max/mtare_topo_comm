# GSE-Graph 持久双向因果几何变化 Teacher 证明 V1

状态：`PASS_GSE_CAUSAL_CHANGE_POINT_PROOF_V1`  
日期：2026-08-26  
正式证据：`results/gate3_semantics/gate3_20260826_gse_causal_change_point_proof_v1_seed0`

## 结论

方案 A 可行。原 Teacher 的短帧几何波动可以替换为稀疏、持久、双向一致且可由过去观测确认的结构变化点，同时保留足够的监督容量。

这项 PASS 只证明新 Teacher 的定义和数据容量成立；尚未证明模型已经学会新语义，也未证明在线拓扑图或探索性能改善。

## 方法

沿每条物理边分别重算正向和反向 native-mesh 宽度/高度序列。继续使用原来的 `5 m` 比较跨度和 `1 m` 宽度/高度变化阈值，不引入结果调出的新尺度。

一个几何变化点只有在以下条件同时成立时才保留：

1. 变化信号至少持续一个 `5 m` 比较跨度；
2. 同一物理边正反两个方向的区间重叠；
3. 两个方向在同一几何量上给出一致的增减方向；
4. 两个方向各有且仅有一个过去窗口检测与该边界对应；
5. 端点 `10 m` 内只合并到唯一 degree-2 节点；terminal/junction 保持优先，多义端点拒绝生成标签。

因果检测只比较已经完成的两个五采样窗口。检测发生在跨过变化之后，再把事件位置回投到估计边界；不读取未来传感器帧。

## 结果

完整范围为 80 个 C01--C08 世界、8,039 条物理边、16,078 条有向穿越、252,430 个几何帧和 188,126 个五帧序列。

| 证据阶段 | 数量 |
|---|---:|
| 单方向持久变化 episode | 1,010 |
| 正反方向一致候选 | 435 |
| 双方向唯一因果候选 | 108 |
| 最终结构变化点 | 76 |
| degree-2 端点节点 | 45 |
| 边内部变化点 | 31 |

旧 Teacher 有 3,035 个 transition identity；其中只有 103 个能在现有 `10 m` 节点半径内对应到新 identity，2,932 个没有稳定对应，支持它们主要是短促 mesh/端点波动而非结构节点的诊断。

proof 初始产生 1,090 条因果标签。与原事件优先级逐帧合并时，39 条 junction 和 20 条 terminal 保持优先，因此生产 Teacher 最终保留 1,031 条 geometry-transition 标签。76 个 identity 全部保留：C01--C06 为 59 个，C07--C08 为 17 个；每个 identity 有 6--23 条标签。最终标签数高于既有每类至少 500 条的训练容量门槛。

双向因果检测延迟为 `7.0--10.5 m`，中位数 `8.0 m`。在线节点必须使用回投后的边界位置，不能把检测时刻的机器人位置当成结构节点位置。

## 风险与下一步

1,031 条相邻帧不是 1,031 个独立结构样本；真正的独立变化 identity 只有 76 个，selection 中为 17 个。训练必须按 identity 平衡，并同时报告逐 identity 覆盖和置信区间，不能只报告帧级准确率。

下一步是生成不可覆盖的 corrected Teacher manifest：保留 junction、terminal 和 turn，删除全部旧 geometry-transition 帧标签，再只写入上述 1,031 条新标签及 76 个 identity。该 manifest 通过完整一致性和泄漏审计后，才能训练新的结构事件头；C09/C10 和 M-TARE 在模型及阈值冻结前继续不读取。

## 复现与图

- 正式 run seal：`54bdc8155c3113b554240c63f568257dc8209f35e9cd0e98c28b6b6daf7b5f60`
- 23/23 文件逐哈希一致，source before/after 完全相同；结果约 5.0 MB。
- 论文图包：`docs/figures/gse_graph/gse_causal_change_point_teacher.{png,pdf,svg,csv}`
- 图包同时保留 source JSON、provenance、生成脚本和 SHA-256；manifest：`95d29a693251637e9f03c7367be6db3c4e22d125b2fb1039bcd3af5818c42b79`
- C09/C10/M-TARE/model inference/training/optimizer step 均为 0。
