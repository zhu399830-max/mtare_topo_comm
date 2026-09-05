# GSE-Graph V3：几何基元条件的结构假设图方法提案

状态：`DRAFT_REQUIRES_EXPLICIT_METHOD_BOUNDARY_DECISION`

本文件不是新的训练授权，也不把任何旧结果改写成成功。它只把当前证据支持的主方法、已知失败点、可复用实现和下一次快速验证写清楚。用户确认方法边界前，不创建新训练、C08、graph replay 或 M-TARE run。

## 1. 论文要回答的问题

> 机器人能否从过去五帧因果 LiDAR 学到三维扫掠几何基元及其局部组合，用这些结构语义决定哪里产生节点、节点有哪些可执行端口，并利用图上已经发生的运动与穿越证据安全处理重复地下结构，最终形成可用于多机器人探索的稀疏拓扑图？

主链为：

```text
五帧因果 LiDAR + 相对里程计
→ 学习三维扫掠基元（轴线、截面、坡度、曲率、不确定性）
→ 学习局部基元组合和结构事件
→ 建立本地确认的结构节点及端口
→ 结构描述候选 + 位姿/邻接/已穿越边联合关联
→ 安全回环合并或拒绝合并
→ 真实穿越后建立带几何属性的边
→ GSE-Graph 全局探索
→ 保留 M-TARE Local Planner
```

学习几何决定图的**生成内容**；图上下文解决局部观察无法解决的**全局身份歧义**。两者不能再混成一个局部 place descriptor 任务。

## 2. 为什么不是继续训练局部地点描述子

当前两级快速门使用完全相同的 C01 人口：10 个 topology parent、180 个五帧观察、100 个物理节点、80 个同节点正对。

| 接口 | 用时 | degree accuracy | 几何 RMSE | 99% precision 下安全关联 |
|---|---:|---:|---:|---:|
| 单一 44 维共享读出 | 12.24 s | 0.99444 | 0.14321 | 0 |
| 几何/关联双读出 | 8.20 s | 0.97778 | 0.10883 | 0 |

双读出已经把不同节点关联距离的中位数拉到 1.15288，但仍有不同物理节点距离精确为 0。地下世界含有重复截面、重复分叉布局和平行/分层通道；局部观察即使几何正确，也不总能唯一确定全局地点。继续增加读出层训练时间不能消除这个可观测性问题。

这不否定几何基元路线：已有三 seed 正式证据表明，学习模型相对非学习拟合在 surface 和连续几何上有稳定改善；Teacher 节点组合在同一 tiny 人口可得 precision/recall=`1.0/0.9625`；完整 C07 的 Teacher/解析描述候选加固定图位置门可得 `2960 TP / 0 FP`、precision/recall=`1.0/0.933459`。

## 3. 旧图条件路线为什么仍然失败

仓库已有的 Factorized GSE-Graph 是必须正面对照的历史路线，不能换名重复。

它的正证据：

- route-conditioned association capacity 在 C07--C08 三 seed safe recall 为`0.4197/0.7847/0.5401`；
- C09 consensus corrective 的 balanced precision/recall=`1.0/0.5077`，runtime false accept=`0.9865%`；
- 它证明“描述子 + 动作 token + 已穿越通道几何”作为候选验证因子是可行的。

它的失败发生在节点与边的生命周期：

- C09 决策触发 precision=`0.97536<0.98`；
- 274 个真实节点最终只唯一提交 192 个；另有 31 个 verifier reject、16 个 proposal missing、12 个中心超出 4 m、10 个歧义/提交逻辑失败和 6 个重复提交；
- 13 条真实 trace relation 最终只恢复 2 条边；8 个端点未提交、3 个端点 proposal 缺失；
- 最好一次空间修正 node F1=`0.8233`，但 edge recall=`0.1538`，不能用于探索主张。

旧状态机要求同一假设获得两条独立 traversal 支持才提交，并从同一 traversal 的首末**已提交语义触发**推导边。这把“新节点存在”“是否与旧节点相同”“能否作为执行边端点”混成一个门，导致高精度关联以大量节点和边漏失为代价。

## 4. 新方法的四个类型化输出

```text
PrimitiveStructuralObservation
  sequence_index
  graph_pose_mean / graph_pose_covariance
  event_probabilities: corridor / junction / terminal
  primitive_ports[]
  structural_descriptor
  uncertainty

PrimitivePortToken
  direction_robot
  swept_axis_controls
  cross_section_half_axes
  shape_exponent
  slope / curvature
  traversability / confidence

GraphAssociationFactor
  structural_similarity
  port_assignment_consistency
  pose_residual_and_uncertainty
  incident_edge_geometry_consistency
  neighborhood_action_consistency
  posterior / ambiguity

VerifiedGeometricEdge
  from_node / to_node
  executed_port
  length
  ordered_axis_width_height_slope_curvature_profile
  traversal_count / execution_state
```

学生 forward 禁止输入 world、TNG node/edge/traversal identity、绝对 GT pose、未来帧或 Teacher descriptor。`graph_pose`来自部署定位/相对里程计，只进入在线图关联，不进入局部几何学生以偷学世界身份。

## 5. 节点生成：学习结构，而不是出口数量规则

一个 permutation-invariant primitive-set encoder 消费最多 64 个预测端点及其置信度，输出：

1. `corridor/junction/terminal`结构事件概率；
2. 最多四个结构端口 token；
3. 旋转不变的局部结构描述和不确定性。

监督来自程序构造的 incident primitive star、degree、截面、相对端口夹角和五帧可见性。degree 是辅助监督，不以`degree>=3`硬规则在部署时生成 junction。节点触发必须是过去五帧内稳定的事件 episode；单帧尖峰只保留 transient hypothesis。

连续转弯、宽高、坡度和曲率变化进入已执行边的 geometry profile，不强制制造大量“语义类别节点”。必要的 metric anchor 只服务定位和曲线近似，不计作结构节点。

## 6. 关键状态机：新建节点与回环合并分离

```text
TRANSIENT_EVENT
  └─ 五帧稳定且结构置信通过 → LOCAL_CONFIRMED_NODE

LOCAL_CONFIRMED_NODE
  ├─ 没有历史候选 → 保留为新的图节点
  ├─ 唯一候选且联合因子安全 → LOOP_MERGED
  └─ 多候选/低置信/冲突 → 保留独立节点 + ALIAS_PROVISIONAL

EXECUTED_DEPARTURE(node, port)
  └─ 到达下一个 LOCAL_CONFIRMED_NODE → VERIFIED_GEOMETRIC_EDGE
```

重要区别：

- 第一次观测到稳定结构事件即可建立**本地确认节点**，不要求第二条独立轨迹；
- 不确定的是“它是否是旧节点”，不是“这个结构节点是否存在”；
- 拒绝回环合并时保留一个独立节点，因此不会为了低 false-loop 而删除可执行连通性；
- 后续反向/不同路线重访可继续给 alias hypothesis 增加证据，再做延迟合并；
- edge 由离开节点时选择的端口、连续执行轨迹和到达节点闭合，不要求同一 traversal 内恰好再出现两个语义触发。

## 7. 图条件联合关联

局部结构描述只负责提出候选，不能单独提交回环。对每个候选对构造对称因子：

- primitive-star 描述距离；
- 端口集合的一对一几何匹配代价；
- 当前定位与候选节点位姿分布的 Mahalanobis 残差；
- 当前 approach geometry 与候选 incident verified edge 的宽、高、坡度、曲率匹配；
- 已执行邻接动作与候选剩余端口的一致性；
- 各来源不确定性、缺失端口和历史拒绝状态。

小型 factor verifier 学习输出 match posterior；固定空间半径只用于限制计算候选，不作为地点分类器。阈值和拒绝间隔只在 C07 冻结。出现多个接受候选、第一/第二候选间隔不足或任一硬物理矛盾时拒绝合并。

Teacher identity仅构造正负 pair并事后评分。真实回放绝不读取 identity 决定候选、合并或边。

## 8. 边的建立与几何属性

机器人从本地确认节点选择一个端口后开始记录 trace；抵达下一个本地确认节点时提交一条 edge。edge 保存整段因果预测的 axis、width、height、slope、curvature、不确定性和执行结果。

同一 physical edge 的反向穿越用于验证/更新属性，但不能先验创建反向边。未走过的出口保留为 exit stub，是全局探索目标候选，不冒充已经存在的边。

这使拓扑图在第一次探索时就有用：它表示“已经走通的连接”和“尚未走过的结构端口”，而不是等走完全部地图后才建图。

## 9. 高效验证漏斗

所有阶段在前一级通过后才能进入下一级。

### A. 零训练状态机与容量证明（分钟级）

只用 C01--C06 fit 和 C07，C08--C10/M-TARE 为 0：

1. Teacher event + Teacher association：状态机 node/edge precision/recall 必须均为 1；否则是实现错误；
2. Teacher event + 当前 primitive descriptor + 图因子：验证关联/提交上限；
3. 当前事件输出 + Teacher association：单独测节点 proposal；
4. 完整当前输出：只作归因，不作为最终模型结果；
5. 把旧 two-trace commit 与新 local-confirmed/strict-loop-merge 并排回放，必须解释节点和边漏失变化。

硬门：false loop merge<=1%，node precision>=0.98，edge precision>=0.98，node/edge recall均非零；oracle状态机必须完全恢复所有可观察节点和已完成trace关系。

### B. 事件头 tiny overfit（秒到分钟）

复用相同 180 观察/100 节点，只训练 event/port set head，不再训练局部全局身份：

- event/degree accuracy>=0.99；
- port count与端口集合匹配F1>=0.99；
- loss下降>=95%；
- permutation/yaw/repeat全部通过。

### C. 单 seed 短训（小时内）

只用 fit 子集和 C07，验证结构事件、端口集合、连续几何和联合关联均有非空安全区域。失败即停，不启动三seed。

### D. 单 seed 完整 C07

必须同时达到：

- junction/terminal macro-F1相对 Cano-like/非学习事件提高至少5个百分点；
- node association precision>=0.98、false loop<=1%、recall>=0.25；
- offline node/edge F1相对规则图提高至少5个百分点；
- edge recall不低于0.60，避免重现历史`0.1538`；
- 十个 topology family 均有非零正确节点与边。

### E. 三 seed、一次 C08、闭环

只有 D 通过才运行三seed；至少2/3 seed通过后冻结模型/阈值，一次零适配 C08。之后才进入 M-TARE 单机器人和2/3/4机器人闭环。

## 10. 对照与消融

主对照：

1. 原始 M-TARE；
2. Cano-like出口模型 + 规则图；
3. 非学习基元事件图；
4. 历史 Factorized exit-token/profile graph；
5. 完整 primitive-conditioned GSE-Graph；
6. GT-TNG oracle。

必要消融：

- 去掉显式 primitive geometry；
- 五帧改为单帧；
- 去掉 incident-edge geometry factor；
- 去掉 pose/adjacency graph context；
- 去掉 uncertainty/refusal；
- 恢复旧 two-independent-trace node commit；
- 恢复旧 same-traversal first/last-trigger edge commit。

最后两项直接验证新状态机是否解决历史节点/边召回失败，不能省略。

## 11. 与现有工作和历史代码的边界

- 相对 Cano：学习的是显式三维扫掠基元组合，结构语义直接产生节点/端口，且图在线生成；不是出口方向检测接已有图。
- 相对 LiDAR place recognition：局部 descriptor 不被赋予不可完成的唯一地点身份；贡献是结构语义、图条件因子和可拒绝回环。
- 相对 MR-TopoMap/传统拓扑图：edge携带学习的完整执行几何，未走端口与已验证边严格区分；图由结构生成而非固定距离采样。
- 相对仓库旧 Factorized GSE：把 exit-token/action hazard 替换为三维 primitive composition，并修正新建、回环和edge commit三个生命周期耦合错误。

## 12. 代码复用与替换清单

| 资产 | 处理 |
|---|---|
| P1a/P1b LiDAR、primitive Teacher、observability sidecar | 原样复用 |
| frozen primitive geometry backbone | 作为初始化与基线，是否解冻由后续单独门决定 |
| `gse_structural_node_evidence.py` | 保留为Teacher容量/评分，不直接作为部署identity |
| `gse_hypothesis_graph.py` | 复用类型和证据账本，增加local-confirmed与alias状态 |
| `gse_factorized_association.py` | 复用对称pair verifier框架，输入换为primitive-star/port/graph因子 |
| `factorized_gse_graph.py` | 复用 traversed geometry profile 与执行状态 |
| `gse_trace_commit_replay.py` | 保留旧策略作消融；新建V2状态机，不原地修改历史行为 |
| `gse_metrics.py`、offline replay、图可视化 | 复用并补充alias/duplicate/edge lifecycle指标 |
| 旧exit-only、slot、endpoint metric、anchor和factorized runs | 保留为基线、消融与失败分析 |

## 13. 当前决策点

推荐冻结本提案作为下一方法边界，先实现第9节A的零训练状态机/容量证明。预计只需数小时开发与分钟级运行；它若不能在oracle输入下完整恢复图，立即修实现或停止，不会启动训练。

需要用户明确确认的是方法本身：

> 是否接受“学习几何结构负责节点/端口生成；地点回环由图条件联合因子验证；首次稳定结构节点可本地确认；边由真实departure-to-arrival轨迹闭合”作为GSE-Graph V3主方法？

确认后才把本文状态改为`AUTHORITATIVE_PHASE3_METHOD`、建立第9节A的Data Card/spec并执行。
