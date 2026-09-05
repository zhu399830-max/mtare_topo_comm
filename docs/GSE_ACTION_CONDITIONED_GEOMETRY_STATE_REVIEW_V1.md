# GSE-Graph 方法差异审查 V1

日期：2026-08-27  
状态：`NAIVE_STATE_CHANGE_ROUTE_FORMALLY_FAILED_AND_SUPERSEDED`

## 结论

论文不能把“出口识别”“连续几何描述符”或“描述符变化点生成拓扑节点”单独作为核心创新。这三条路线都已有直接或高度相近的工作。

保留的候选方法是 **Action-Conditioned Geometry State Graph（动作条件几何状态图）**：从因果 LiDAR 序列估计局部可通行几何状态及出口动作集合；只有稳定的几何状态或可执行动作集合发生变化时才提出节点；歧义关联必须拒绝合并；边只有在机器人真实穿越后才确认。图中保存可供探索使用的宽度、净空、坡度、曲率、出口状态及不确定性。

这仍只是有创新潜力的候选边界，不是已经证明的论文贡献。下一步必须先用现有冻结观测做 C01--C08 只读可行性验证。如果“几何 + 动作”不能在低误触发条件下优于任一单项信息源，停止该路线，不进入新训练。

## 相关工作与贡献矩阵

| 相关工作 | 输入与节点产生方式 | 已覆盖内容 | 本文必须保持的差异 |
|---|---|---|---|
| Cano 等，*Autonomous Navigation in Large-Scale Underground Environments Based on a Purely Topological Understanding of Tunnel Networks*，JFR 2026 | 3D LiDAR 深度图预测隧道/交叉口/终点及跟踪出口；交叉口和终点作为节点，隧道作为边 | 地下 LiDAR 学习出口、纯拓扑导航、低带宽图 | 不把出口检测当创新；学习显式连续几何状态和不确定性；节点由状态/动作变化产生；在线边必须经过实际穿越验证；最终比较 Cano-like 基线。来源：[DOI](https://onlinelibrary.wiley.com/doi/10.1002/rob.70157) |
| Chapoulie 等，*Topological Segmentation of Indoors/Outdoors Sequences of Spherical Views*，2012 | 球面图像全局结构描述符；自定义变化点检测划分拓扑地点 | “结构描述符 + 变化点 = 拓扑地点” | 普通变化点本身不作为贡献；分段对象必须是可执行动作集合与显式度量几何联合状态，并证明其直接改善在线图与探索。来源：[论文 PDF](https://citeseerx.ist.psu.edu/document?doi=674bb87ec898718fcd8a87e5601c8ea727f66fbf&repid=rep1&type=pdf) |
| Hu 等，*Active Layered Topology Mapping Driven by Road Intersection*，KBS 2025 | LiDAR 主动检测道路交叉口；节点含 metric/semantic/topology 三层并做 ReID | 交叉口语义、度量属性、节点重识别和在线拓扑 | 场景改成矿井本身不足以构成创新；必须使用因果 3D LiDAR 的连续可通行状态、出口动作变化、拒绝式关联和穿越后边确认，并报告多机器人探索收益。来源：[DOI](https://doi.org/10.1016/j.knosys.2025.113305) |
| Unsupervised Subterranean Junction Recognition，2020 | 单帧 2D LiDAR 与谱聚类估计墙面/分支 | 无监督地下交叉口识别 | 不以交叉口计数为贡献；覆盖坡道、转弯、净空/宽度变化和时序不确定性，并完成关联、建图与闭环。来源：[arXiv](https://arxiv.org/abs/2006.04225) |
| Learning Topometric Semantic Maps from Occupancy Grids，2020 | 完整 occupancy grid 中学习房间、门、走廊，再导出拓扑 | 学习结构语义并生成拓扑 | 本文输入为在线因果局部 LiDAR，不读取完整地图或未来；输出同时约束可执行动作和验证边。来源：[arXiv](https://arxiv.org/abs/2001.03676) |
| Robotic Exploration through Semantic Topometric Mapping，2024 | 将已探索栅格图分割为交叉口、路径、终点和 frontier | 结构语义 topometric map 用于探索 | 不依赖先建稠密栅格再分割；直接由局部 LiDAR 状态维护稀疏图，并用不确定性拒绝错误合并。来源：[arXiv](https://arxiv.org/abs/2406.18381) |
| Scan Context++，2021 及 LiDAR place recognition | 点云描述符用于地点检索和回环定位 | 结构描述符与地点关联 | place descriptor 仅是关联证据之一；它不能自行生成节点或未经穿越生成边；必须单独报告 false loop merge。来源：[arXiv](https://arxiv.org/abs/2109.13494) |
| DeepExplorer，RSS 2023 | 图像特征空间探索；最新观测可成为节点，视觉地点识别补回环 | 学习特征空间中的拓扑探索 | 不把每帧/关键帧作为节点；节点必须对应稳定的结构/动作状态变化，图属性可解释且可用于地下通行决策。来源：[论文](https://roboticsproceedings.org/rss19/p099.html) |

## 候选方法的最小独特机制

在线观测状态固定为：

```text
S_t = {
  local_axis,
  width, height, slope, curvature,
  exit/action tokens,
  place descriptor,
  uncertainty
}
```

图更新固定为：

```text
因果 LiDAR 历史
→ 估计 S_t
→ 检测持续的几何状态或动作集合变化
→ 提出 provisional structural node
→ 依据状态、出口 token、位置与不确定性做拒绝式关联
→ 机器人真实穿越后确认 edge 及其几何属性
```

这里真正要检验的不是“类别分对没有”，而是三件事：

1. 状态改变是否能在走廊误触发不超过冻结预算时覆盖真实结构事件；
2. 几何与出口动作信息联合后，是否显著优于只用几何或只用出口；
3. 这种节点证据能否降低离线图的漏节点、重复节点和错误回环，并最终改善探索效率和通信量。

## 只读可行性验证的预注册边界

- 数据：C01--C06 拟合分区 142,184 条观测；C07--C08 选择分区 45,942 条观测；共 188,126 条，C09/C10/M-TARE 零读取。
- 输入：三个冻结 seed 的 146 维部署可用观测特征；不得读取 GT identity、世界坐标、未来帧或 teacher 几何作为算法输入。
- Teacher：既有 12 帧因果 episode membership，仅用于评分和拟合分区阈值。
- 方法：比较 geometry-only、action-only、combined 三个预声明状态差异；按 traversal 内因果历史计算，不训练模型。
- 触发：拟合分区固定走廊误触发预算确定阈值；连续高分只合并成一次事件触发。
- 选择分区报告：触发 precision、episode recall、junction/terminal/turn/geometry-transition identity coverage，以及每个拓扑 family 的结果。
- 候选路线通过条件：selection 上 combined 的结构触发 precision 不低于 `0.98`、每个 eligible corridor observation 的错误触发比例不高于 `0.01`、episode recall 相对两个单项基线的较高者至少提高 `0.05`；turn 与 geometry-transition 均至少覆盖 1 个 identity。任何输入/Teacher/分区漂移立即失败。

## 明确排除的表述

- 不再称现有五类事件分类器为论文主方法；正式 V1R 已证明其 turn/geometry-transition 不可靠。
- 不声称首次使用几何变化点建立拓扑图。
- 不声称首次用 LiDAR 识别地下出口或交叉口。
- 不把程序化 Teacher、M-TARE 接口或 Cano 生成器本身写成方法创新。
- 在离线可行性证据通过前，不开始新训练、不读取严格测试、不进入建图调参或闭环。

## 2026-08-27 正式可行性结果

唯一正式 run `gate3_20260827_gse_action_conditioned_state_feasibility_v1_seed0` 系统执行正常、输入未漂移、零推理/优化/禁用数据读取，但科学结果为 FAIL：

- combined 结构触发 precision=`0.552632`，未达到`0.98`；
- combined episode recall=`0.031111`，相对更强单项基线变化=`-0.005185`，未达到`+0.05`；
- combined 只覆盖 turn `2/95`、geometry-transition `1/17` identities；
- 12-observation双块差异只有`39,128/188,126`观测可用，短 traversal 和起始决策点大量缺少历史；
- 原5维 exit summary丢弃了每个出口的方向、宽度、vertical profile与descriptor关系。

因此本文件提出的“压缩状态摘要 + 固定变化点”路线被正式否定。其图和指标保留作失败分析，不得调阈值重跑。后继方法改为因子化表示：junction/terminal是动作集合改变的决策节点；turn和连续几何变化保存在trace-verified edge geometry profile中，不再强行当作拓扑节点。后继权威方案见`docs/GSE_GRAPH_FACTORIZED_RESEARCH_PLAN_V2.md`。
