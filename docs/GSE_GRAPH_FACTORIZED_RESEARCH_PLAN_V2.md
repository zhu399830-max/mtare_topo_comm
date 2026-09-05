# Factorized GSE-Graph 论文研究计划 V2

状态：`ACTIVE_AUTHORITATIVE_METHOD_REVISION`

## 核心问题

能否把因果 LiDAR 学到的结构语义分解为两类不同信息，并由它们直接建立可靠的在线图：

1. **动作语义**决定哪里是必须做选择的拓扑节点；
2. **连续几何语义**描述机器人真实走过的边，而不是被错误离散成大量事件类别。

这次修订不回到 Cano-like 出口检测，也不再训练一个五类事件分类器。目标表示为：

```text
因果 LiDAR
→ 完整出口/动作 token + 决策点 hazard
→ junction / terminal provisional node
→ 实际穿越
→ width / clearance / slope / curvature / axis profile
→ verified geometric edge
→ route-conditioned refusal-aware association
→ Factorized GSE-Graph
```

## 为什么需要因子化

turn、坡度变化、宽高变化和曲率变化通常改变的是一条通道的执行代价与可通行性，并不必然增加或删除可选动作。把它们与 junction/terminal 一起强制分类成拓扑节点，导致类别定义、触发时机和图结构混在一起。

现有正式证据给出清楚分界：

- junction/terminal 因果触发已达到`0.9914`结构触发precision，identity coverage分别为`140/146`和`125/128`；
- 连续宽/高/坡度/曲率相对非学习估计改善`24.07%/52.00%/49.67%/85.31%`；
- turn/geometry-transition强制分类仅`6/95`和`0/17`；
- 压缩状态变化点的combined precision仅`0.5526`，episode recall仅`0.0311`。

所以失败发生在“如何离散成节点”，不是“有没有学到几何”。

## 类型化图接口

```text
DecisionNodeObservation
  decision_hazard
  exit_tokens[]
  local_axis
  place_descriptor
  uncertainty

TraversedGeometryProfile
  ordered_samples[]
    arc_m
    axis
    width_m / height_m
    slope_deg / curvature
    uncertainty

FactorizedDecisionNode
  provisional_or_persistent
  action_signature
  incident_verified_edges[]
  association_confidence

FactorizedVerifiedEdge
  from_node / to_node
  length_m
  geometry_profile
  traversability_summary
  execution_state
```

节点提交规则：

- 只有稳定的junction/terminal决策证据可以产生结构节点；
- 普通turn/geometry-transition只进入edge profile或内部metric polyline，不进入结构node F1；
- 模糊重访建立provisional node，禁止强制loop merge；
- edge只有机器人从一个决策节点实际走到另一个决策节点后才确认；
- 关联可使用当前动作token、place descriptor、空间距离，以及已经执行得到的incident-edge geometry fingerprint，但禁止GT identity和未来信息。

## 与现有工作的差异

- 相对Cano：不只输出出口或依赖用户已有拓扑；系统在线生成图，并学习可执行边的度量几何与不确定性。
- 相对普通change-point拓扑分段：变化点不是贡献，且连续几何变化不再被一律离散成节点。
- 相对LiDAR place recognition：descriptor不直接决定loop；动作集合、incident edge geometry和拒绝机制共同决定关联。
- 相对道路intersection layered map：重点不是把metric/semantic/topology三层并列存储，而是用实际穿越积累的地下通道几何反过来约束节点关联和探索可执行性。

## 当前证据身份

现有junction/terminal触发与连续几何精度是**方法重构的组件证据**，不是Factorized GSE-Graph整体PASS。旧五类分类、普通变化点、多变量逐帧风险和完整token关联失败全部保留为必要消融/失败分析。

任何正式新结论必须从下列顺序取得：

1. C01--C08只读inventory：构造decision-node episodes、完整exit token、可用incoming/outgoing geometry profile和困难负对；
2. C01--C06拟合、C07--C08选择的route-conditioned association capacity proof；
3. 冻结Factorized接口与阈值后，才允许一次C09资格；C10继续严格隔离；
4. 只有decision node、association和geometry profile同时通过，才实现离线图；
5. 离线图通过后再进入单/多机器人闭环。

## 下一门槛

route-conditioned association必须同时满足：

- precision `>=0.98`；
- false loop merge `<=1%`；
- nonzero accepted pairs；
- recall `>=0.25`；
- 十个topology family没有零precision或系统性强制合并；
- 相对current-observation association和distance/angle规则关联均有独立消融。

如果加入已执行边的几何profile仍不能达到安全关联，停止“学习关联”主张；保留学习几何作为edge representation，并重新评估论文创新强度，不进入planner调参。

## Capacity proof冻结接口（2026-08-27）

identity-balanced Teacher PASS后的容量证明不把两个incoming edge profile直接要求为相似，因为同一个junction从不同支路进入时，宽度和坡度本来可以不同。route-conditioned关系固定为：将一侧5帧因果序列已经实际走过的incoming width/slope profile，与另一侧观测到的6个exit token opening-width/vertical-profile逐个匹配；两方向结果排序后形成严格对称、token permutation invariant的relation。完整模型同时读取learned event probabilities、axis relation、place descriptor、uncertainty和完整exit-token set relation。

绝对pose、world/parent/identity、Teacher event和objective geometry profile禁止进入学生。`<=16 m`空间距离只定义同world runtime candidate域，不作为identity-balanced跨world structural-alias classifier的数值输入，避免学生用不同world坐标系识别negative。正式容量proof固定C01--C06 fit `792 identities / 1,584 pairs`，C07--C08 selection `274 / 548`；三seed各训练full与no-route-geometry两个小模型，参数量分别`8,017/6,673`，backbone和全部perception outputs冻结。

除原安全门槛外，full模型三seed平均safe recall必须比descriptor-only至少高`0.05`、比no-route-geometry至少高`0.02`，且任一seed相对no-route退化不得超过`0.02`。否则即使某个seed偶然达到precision门槛，也停止route-conditioned association贡献，不进入C09或图回放。

## 论文证据保留

- `gse_action_conditioned_state_feasibility`的PNG/PDF/SVG和source JSON保留为“为什么不采用普通变化点”的失败图；
- junction/terminal触发图、连续几何误差图、route-conditioned association ROC/precision-recall、最终结构图和coverage-time曲线必须保留PNG及矢量源；
- 每张正式论文图必须绑定生成脚本、机器可读源数据和SHA-256。
