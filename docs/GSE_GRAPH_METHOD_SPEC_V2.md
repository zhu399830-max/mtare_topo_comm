# ERCSS 方法实施规范 V2

状态：`AUTHORITATIVE_PHASE3_CANDIDATE`

## 研究问题

五帧因果 LiDAR 在相对里程计配准后，是否足以恢复机器人当前已经观测到的、与自身连通的稀疏隧道骨架，并用该骨架的分叉、终止和连续几何变化直接产生拓扑节点？

## 方法边界

学生输入只有五帧 `16×720 range + valid mask` 和五帧相对平移/偏航。数据准备阶段可从世界位姿计算相对变换，但输出只保留当前机器人坐标，不保存绝对位姿、world、TNG、edge、traversal 或 identity。

Teacher 使用训练世界的 TNG、spline、geometry parameters 和 native mesh：每条物理 edge 独立按 1 m 采样；只保留过去五个传感器位姿中处于 16 线垂直视场、50 m 范围且 native-mesh LOS 可见的点；再从当前轴线最近可见点出发取唯一连通分量。非可见间隙不得补边，不同 edge 或重叠隧道不得按 tunnel ID 合并。

模型候选输出局部骨架节点、连接、每段 width/height/slope/curvature 与不确定性。稳定的局部分叉、终止和几何变化可以提出节点；全局 edge 仍只能由机器人真实穿越提交。

## 当前只允许的工作

先做零训练 Teacher/表示可行性审计。固定 C01--C06 为 fit 人口、C07--C08 为 selection 人口，共 80 worlds、252,430 个唯一帧、188,126 条五帧序列、16,078 条声明 directed traversals（其中 16,076 有序列）。C09、C10、M-TARE、graph replay、planner 和训练全部禁止。

审计必须证明：

- 相对配准对任意全局 SE(2) 变换不变，当前帧必须是窗口最后一帧；
- 物理 edge 采样和图连接唯一，全部 Teacher 点有因果视场与 native-mesh LOS 支持；
- 可见骨架严格 ego-connected，不跨越不可见间隙，不按 tunnel ID 折叠；
- fit 和 selection 的 junction、terminal、turn、geometry-transition 身份覆盖足够；
- 节点/segment/trace 容量存在固定有限上界且 selection 不溢出；
- 输入、工具和输出哈希不漂移，训练、推理和测试读取均为零。

可行性 FAIL 时停止 ERCSS，不缩短历史、不扩大视场、不降低覆盖门、不用图或规划器补偿。PASS 也只允许实现模型 readiness，不能进入在线图。

