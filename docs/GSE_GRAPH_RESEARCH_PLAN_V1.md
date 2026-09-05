# GSE-Graph 论文实施计划 V1

状态：`ACTIVE_AUTHORITATIVE_RESEARCH_ROUTE`

唯一目标：完成一篇以“从因果 LiDAR 序列学习局部几何结构语义，并由该语义直接建立在线 topometric graph”为核心贡献的可投稿论文。

## 研究问题

在不读取完整地图、未来帧或 GT identity 的部署条件下，学习得到的中心轴、出口布局、宽度、净空、坡度、曲率、结构事件、位置描述子与不确定性，能否：

1. 比出口方向模型和非学习几何估计更准确地描述局部结构；
2. 以高精度触发 junction、terminal、turn 和 geometry-transition 节点，并完成重访关联；
3. 通过只由真实穿越验证的边，形成比规则图更正确、更稀疏的因果拓扑；
4. 在保留 M-TARE local planner、避障、控制和传感器接口时改善单/多机器人探索。

## 固定数据边界

- 独立单位是 topology parent，固定 `80 train / 10 validation / 10 strict test`。
- 已封存单帧数据为 train `20,000 clusters / 100,000 frames`、validation `2,500 clusters / 12,500 frames`。
- 新序列只从 train/validation parent 的全部 directed traversals 构造：每 `1 m` 弧长一个 anchor、每个样本仅含当前与过去共 5 帧。
- TNG 提供 node、degree、tunnel、edge 和结构事件 identity；spline 提供轴线、弧长、坡度与曲率；mesh/geometry parameters 提供宽度与净空。
- 宽度/高度教师固定为 sealed native mesh 上 route-frame 的左右/上下窄扇区射线中位数；geometry-transition 是相邻 5 m 横截面中位数发生至少 1 m 的持续变化。解析 tunnel radius 只保留为非学习基线，不再作为主教师。
- 训练集只读哨兵已证明五类事件均非空。对 open junction 中本来不唯一或 mesh 缺口导致的不完整横截面，只屏蔽 width/height 两维损失，不删除样本，也不补造半径标签；event、axis、slope、curvature 和 association 教师继续有效。
- C10、M-TARE benchmark 和其它 formal test world 在模型、温度、拒绝阈值及图参数冻结前保持零读取。
- 只读 inventory 已确认：train `8039 edges / 16078 directed traversals / 188126 sequences / 252430 unique frames`，validation `1027 / 2054 / 24462 / 32678`；合计 `9066 edges / 18132 directed traversals / 212588 sequences / 285108 unique frames / 1062940 referenced frames / 276028.677900 m`。事件分布和正负关联对数量仍须在正式 manifest 中冻结，禁止预估数进入训练合同。

## 主方法

```text
5-frame causal LiDAR
→ circular range-image encoder + causal temporal aggregation
→ geometric event probabilities + metric geometry + exit token set
→ learned event triggering and uncertainty-aware node/exit association
→ provisional or persistent structural nodes
→ trace-verified topometric edges with geometry attributes
→ global target selection
→ unchanged M-TARE local execution
```

模型必须输出类型化 `GeometricSemanticObservation`：事件概率、局部轴、宽度、高度、坡度、曲率、place descriptor、可变数量 `ExitGeometryToken` 和不确定性。模糊关联建立 provisional node，禁止强制 loop merge；edge 只能由物理穿越建立。

## 对照与消融

主对照固定为：原始 M-TARE；现有 Cano-like 出口模型 + 规则因果图；非学习几何事件图；完整 GSE-Graph。GT-TNG 仅作诊断参考，不称性能上界。

必要消融为：无显式几何、单帧、距离/角度规则关联、无不确定性拒绝、无 edge 几何属性。旧 V9/V5 运行作为 exit-only 与规则图失败机制证据，不再扩展为主方法。

## 训练完成后的冻结离线协议

- 每个 seed 只用其完整 `24,462` 条 validation 输出做一次校准：事件温度最小化 NLL；事件置信度与 uncertainty 采用同一个 reliability 阈值并最大化五类 macro-F1；出口存在阈值最大化集合匹配 F1。
- place/exit descriptor 按 parent 内、只看过去观测的最近邻计算。阈值选择先满足 association precision `>=0.98` 和 false accept `<=1%`，再最大化接受的正确重访数；禁止零接受的虚假 PASS。
- 非学习连续几何主对照固定为当前帧 LiDAR 的 horizontal PCA、五个等占用纵向截面、robust 横截面包络、截面中心线/二次曲线拟合；不读取 Teacher、pose、world identity 或未来帧。另保留出口 range-sector + 固定几何事件图作为完整非学习图基线。
- validation 每个 world 的双向 physical traversals 按 traversal ID 确定性构成 Euler circuit；全部 directed traversal 精确消费一次。模型只按该物理连续路线获得当前及历史 LiDAR，edge 只在真实正行程 trace 后建立，禁止 teleport edge。
- Teacher structural graph 将 corridor 与不可观测 degree-2 TNG 点压缩，只保留 junction、terminal、turn 与 geometry-transition identity；未被学习事件发现但成为分支/末端的 metric anchor 保留并计作错误。重复预测节点、重复边和 unmapped edge 均进入 precision 分母，不允许集合去重掩盖过建图。
- 图超参数只在 validation 做一次共享选择。固定 `3^5=243` 组：stable frames=`2/3/4`、minimum event travel=`2/4/6m`、metric anchor interval=`20/30/40m`、association radius=`8/12/16m`、ambiguity margin=`0.02/0.05/0.10`。每个 seed 的温度、置信度、descriptor 和出口几何容差由上述校准固定，不进入该网格。
- 选择规则先拒绝 association 不安全的配置；再依次最大化 `min(node F1, edge F1)`、两者均值、component/cycle-rank 准确性和最小 grid index。训练完成前该网格已冻结；C10 与 M-TARE 仍为零读取。
- 完整 GSE、GSE+规则关联消融和 exit-only+规则图分别对三个冻结 seed 运行同一 `243` 组结构网格；非学习几何事件图是确定性方法，只运行一个 observation seed，但仍运行同一 `243` 组结构网格。GSE+规则关联消融保留对应 seed 已校准的事件阈值和不确定性拒绝，只替换关联器。规则图不读取 learned place/exit descriptor，仅使用事件类型、空间距离、出口数量和出口朝向；等距或近等距候选进入 provisional node，不允许强制回环合并。
- 基线不得因 association 不安全而被事后移除。完整 GSE 先执行 association 安全约束，再按上述顺序选参；基线直接按 node/edge 与图不变量表现选参，同时原样报告其错误关联。这使较弱基线仍留在论文比较中，避免只与被筛选后的有利子集比较。
- 离线图科学门槛在读取结果前固定：完整 GSE 的 node F1 和 edge F1 必须分别比 `exit-only+规则图` 与 `非学习几何事件图` 中对应最强者至少高 `0.05`；其跨 validation world/seed 的 connected-component 和 cycle-rank 平均有符号误差绝对值均不得超过 `0.25`；同时必须发生非空关联尝试，且关联 precision `>=0.98`、false loop merge `<=1%`。任一条件失败即停止严格测试和闭环实验。
- 离线结果必须保存全部参数 sweep、唯一选中配置、逐 world/seed 的 nodes、edges、decision trace、图不变量和 GT-TNG 诊断；GT-TNG 仅作为 identity-aware 诊断参考，不参与参数选择，也不称可部署或理论上界。
- 论文定性拓扑图在结果前固定为validation的`S01/S06/S10 C09`、seed0和全部四种实际方法，分别覆盖小型平面树、3D中型分支和3D复杂拓扑。只画压缩后的结构图；objective graph用灰色虚线、正确/错误预测边分别着色、未映射节点单独标记并用节点颜色显示高度。禁止按结果另挑world或seed。

## 进入闭环的硬门槛

- 结构事件 macro-F1 相对 exit-only baseline 提高至少 5 个百分点；
- 上述事件门槛以三seed平均绝对提升为主：平均至少`+0.05`，至少`2/3`个seed各自达到
  `+0.05`，且任何seed不得低于其对应exit-only seed；
- 连续几何量采用跨单位可比的预注册门槛：分别计算
  `1 - learned_MAE / nonlearning_MAE`，四项（宽度、高度、坡度、曲率）等权平均至少
  `10%`，且任一单项不得相对基线退化超过 `5%`；全部单项、有效样本数与原始 MAE
  同时报告，禁止只展示有利指标；
- 节点关联 precision `>= 0.98`，false loop merge `<= 1%`；
- 离线 graph node/edge F1 各自相对两个主要可部署基线中对应最强者至少提高 5 个百分点；
- connected components 与 cycle rank 的跨 validation world/seed 平均有符号误差绝对值各不超过 `0.25`；
- 无 GT identity、未来帧、validation/test 反向泄漏。

任一学习门槛不通过，则停止 GSE 闭环，不通过调 planner 掩盖感知或关联失败。

## 实施顺序

1. 当前 30-case exit-only/规则图基线自然完成并封存；生成只读统计和论文图，不修改旧 run。
2. 建立相关工作贡献矩阵、正式类型接口、序列 inventory 与 Data Card。
3. 实现 objective teacher、困难负样本、关联对和数据泄漏测试。
4. 实现 5 帧 GSE 模型、损失、validation-only 校准与 seeds `0/1/2` 训练。
5. 通过离线感知与图门槛，再实现在线 GSE graph 和全部消融。
6. 完成未见程序化拓扑、单机器人、2/3/4 机器人闭环与统计。
7. 生成方法图、结果表、topology 图、coverage-time 曲线、失败案例、复现包和投稿 PDF。

## 证据与清理

论文图统一保存在 `docs/figures/gse_graph/`，每张图保留 PNG、矢量 PDF/SVG、生成脚本、机器可读源数据、选择规则和 SHA-256。不得只留截图。

方法主图已固定为 `gse_method_overview`：明确区分 learned encoder、typed semantic observation、deterministic online graph 与 unchanged M-TARE local planner；不把完整拓扑或规划策略冒充为端到端学习输出。

清理只按逐目录 allowlist 执行。先记录路径、大小、状态、替代证据和 hash；保留所有基线、消融、失败分析、最终模型、数据、图和复现材料。只有被后继 run 完整替代且与论文无关的 cache、smoke 包、重复 sensor shard、孤立 RUNNING 目录和可再生 raw bag 才能删除。当前活跃 run 完成前不做 I/O 密集清理。
