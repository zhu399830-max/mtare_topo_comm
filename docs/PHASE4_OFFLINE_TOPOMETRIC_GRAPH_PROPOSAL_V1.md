# Phase 4 Offline Topometric Graph Proposal V1

状态：`C08_DEVELOPMENT_AUTHORIZED_TRAJECTORY_CONTRACT_PASS_REPLAY_NOT_YET_EXECUTED`  
日期：2026-08-13  
前置结论：Phase 3 / Gate 2 `GATE_PASS`  

## 1. 唯一研究问题

冻结的 M1D 局部结构语义输出，能否在连续、因果 LiDAR 轨迹上构建紧凑且可解释的
`structural node + verified edge + exit stub` 图，并在未参与图参数选择的 C09 拓扑上保持
出口、连通性和路径距离？

本阶段只做离线因果 replay，不控制机器人、不修改 M-TARE、不选择全局目标。

## 2. 输入与边界

学生侧每帧只使用：

```text
16x720 range + valid mask
当前 sensor pose（只用于世界坐标锚定和实际运动边，不输入 M1D）
历史图状态
```

冻结 M1D 输出：

```text
720-bin outgoing direction logits
branch count logits
interior / junction / terminal role logits
128D z_role
```

`z_role` 只作结构作用一致性证据，不是 `z_place`，不得用于地点身份或单独触发回环。
完整 graph/spline 只用于生成连续轨迹、oracle 与评价，禁止进入 M1D 或在线图更新。

## 3. 精确数据卡草案

### 3.1 图参数开发集

固定 3 个 train split C08 parent，每个 parent 一条 doubled-edge continuous replay：

| Parent | GT nodes/edges | route length | frames @ 2 m |
|---|---:|---:|---:|
| S01_flat_tree_small_C08 | 54/54 | 1,601.266 m | 801 |
| S06_3d_branch_medium_C08 | 92/92 | 2,826.012 m | 1,414 |
| S10_3d_complex_C08 | 164/166 | 5,114.364 m | 2,558 |
| 合计 | — | 9,541.643 m | 4,773 |

这些数值来自已 PASS 的 graph-edge/spline/mesh 正式合同，不再使用节点间直线估计。每条边由
`graph node -> <=0.5 m 显式连接段 -> spline subarc -> 显式连接段 -> graph node` 构成，并各
正反遍历一次。轨迹只用于选择规则持续性和关联阈值，不得查看 C09/C10 后回调参数。

### 3.2 冻结图验证集

固定全部 10 个 validation split C09 parent，每个 parent 一条 doubled-edge continuous replay：

| Parent | GT nodes/edges | route length | frames @ 2 m |
|---|---:|---:|---:|
| S01_flat_tree_small_C09 | 58/57 | 1,865.974 m | 933 |
| S02_3d_tree_small_C09 | 58/57 | 1,713.007 m | 857 |
| S03_flat_unicyclic_small_C09 | 60/60 | 1,881.516 m | 941 |
| S04_3d_unicyclic_small_C09 | 59/59 | 2,017.536 m | 1,009 |
| S05_flat_branch_medium_C09 | 96/96 | 2,695.245 m | 1,348 |
| S06_3d_branch_medium_C09 | 85/85 | 2,653.675 m | 1,327 |
| S07_flat_loop_rich_C09 | 137/138 | 3,682.825 m | 1,842 |
| S08_3d_loop_rich_C09 | 134/135 | 4,338.826 m | 2,170 |
| S09_flat_complex_C09 | 164/166 | 5,060.904 m | 2,531 |
| S10_3d_complex_C09 | 172/174 | 5,532.937 m | 2,767 |
| 合计 | — | 31,442.449 m | 15,725 |

本表是授权前使用节点直线距离得到的历史估计，已因 C08 几何审计证明该口径会低估 spline
弧长而作废。C09 未获读取/执行授权；未来必须另做相同 edge/spline/connector 合同后才能
形成精确 C09 数据卡，不得沿用表中里程或帧数。

C09 已用于 M1D checkpoint 选择，因此本实验只能评价“冻结感知条件下的图构建泛化”，
不能宣称严格端到端未见拓扑泛化。C10、M-TARE benchmark、LAMP/SubT 和历史 NPZ
读取均为 0。C10 只在图方法及所有阈值冻结后另立一次严格测试卡。

### 3.3 独立采样单位

- 当前获批 topology parent：3 个 C08；
- 当前连续 trajectory：3 条，每 parent 1 条；
- 当前精确 raw/effective frame：4,773 / 4,773；
- spacing：doubled-edge centerline route 上约 2 m；
- 每条轨迹覆盖每条 GT edge 两个方向，用于验证 stub 被实际通过后才转为 edge；
- trajectory seed、sensor seed 和起点必须在正式卡中固定。

## 4. 主方法、基线与必要备选

### 4.1 主方法：规则持续性的 Structural Topometric Graph

```text
M1D frame outputs
-> circular direction component decoding + confidence
-> temporal hysteresis / stable structural event
-> structural node or distance anchor
-> spatial + role + heading-set association
-> verified traversed edge
-> remaining outgoing components as exit stubs
```

节点保存 `xyz anchor / role probabilities / branch directions / confidence / z_role audit / visited`。
边只在机器人真实运动通过两个节点后建立。回环候选必须同时满足空间距离、结构角色和
方向集合一致；`z_role` 可拒绝明显不一致候选，但不得单独合并节点。

Exit stub 第一版状态固定为：

```text
unobserved -> observed -> attempted -> traversed
                         -> temporarily_failed -> blocked
```

离线 replay 只会自然产生 `observed/traversed`；其他状态保留 schema，不伪造规划行为。

### 4.2 基线

同一连续 LiDAR、pose、轨迹和 graph builder，使用冻结 `RangeExitBaseline` 输出代替 M1D。
这回答学习语义是否真正改善图，而不是换了轨迹或图算法。

### 4.3 Oracle 上界

使用 GT direction/role 输入同一个 graph builder。Oracle 只证明图接口和规则上限，不作为
传感器方法性能。如果 Oracle 无法形成合格图，必须先修图定义，禁止归因给 M1D。

### 4.4 必要备选

仅当规则持续性在开发集表现出明确节点抖动或系统性漏检时，另立 change-point detection
proposal。当前禁止并行引入 GNN、GRU、Transformer、learned node score 或 place descriptor。

## 5. 参数冻结顺序

1. 先用 3 个 C08 开发轨迹比较有限、预声明的规则参数网格；
2. 目标函数必须同时覆盖 exit preservation、connectivity、node redundancy、path distortion；
3. 冻结唯一参数组和失败分类；
4. C09 只执行一次，不根据结果回调；
5. M1D seeds 0/1/2 全部独立 replay，报告中位数和最差世界，不在 C09 选择 seed。

旧原型的 `stable_frames=1 / travel=6 m / loop radius=4 m / heading=20 deg` 只能作为候选中心，
不能直接升级为正式参数，因为它来自单一开发 world 的 81 组同轨迹 sweep。

## 6. 指标与 Gate 判据草案

必须逐 world、逐 seed 保存原始值，并报告 median、P10 和 worst case：

- exit preservation precision/recall/F1；
- GT component connectivity preservation；
- verified-edge correctness；
- node redundancy（predicted structural nodes / matched GT structural nodes）；
- path-distance distortion（图最短路相对 GT 路径）；
- remote-frontier reachability；
- node/stub temporal churn；
- runtime、peak memory、graph bytes。

建议先通过 C08 开发分布冻结数值门槛，不能现在凭空写一个宽松阈值。硬完整性门可先固定：

- 13/13 trajectory deterministic replay；
- 0 非因果 GT 输入；
- 0 C10/M-TARE/规划器读取或修改；
- 0 未经实际通过建立的 verified edge；
- 所有图、trajectory、decision trace、逐帧 semantic outputs、failure reason 和 seal 完整保存。

## 7. 科研可视化

只保存以下可核验图：

1. 每个 world 一张 GT / B0 / M1D / Oracle 全图叠图；
2. 每个 world 的 node/stub churn 随距离曲线；
3. connectivity 与 path-distortion 分布；
4. 固定规则抽取的 worst-case 失败页，不挑最好看样本；
5. C08 参数选择 Pareto 图，C09 不画调参图。

每张图必须绑定机器可读 metric、world、trajectory、seed、坐标单位和颜色含义。

## 8. 实施步骤与当前停止点

1. 用户明确修改 `docs/PLAN.md` 的实施上限，允许 Phase 4；
2. operational Gate 从 2 切到 4（Gate 3 显式 direction/count/role 已在 Phase 3 多任务合同中完成，
   需在决策日志中明确这是证据映射，不是跳过语义验证）；
3. 建立并批准完整 topology-replay data card；
4. 冻结 trajectory generator、M1D inference adapter、graph builder、baselines 和 metrics；
5. preflight 后先执行 C08 development run；
6. 用户审阅并冻结参数；
7. 另立 C09 validation run；
8. Gate 4 PASS 后仍不得自动进入 M-TARE shadow 或高层规划。

当前停止点：C08 trajectory mesh contract 已正式 PASS 并封存，轨迹坐标已生成；LiDAR
raycast、M1D/B0/oracle 推理和在线图更新仍为 0。下一步必须先给出正式 C08 replay 的精确
数据/方法/成本/证据并经用户确认，再执行 4,773-frame replay；C09/C10 和 M-TARE 仍禁止。
