# GSE-Graph 方法规范 V1

状态：`TYPED_DUAL_COMPOSER_READINESS_PASS_AWAITING_THREE_SEED_TRAINING_CARD`

## 1. 论文方法问题

GSE-Graph 要证明的不是“LiDAR 能分类路口”，而是：

> 从过去五帧 LiDAR 学到的几何结构状态，能否直接产生可解释、可拒绝、可关联的结构事件，并用真实穿越边构成稀疏在线拓扑图？

主链固定为：

```text
5-frame causal LiDAR
  -> learned geometry-semantic state
  -> typed causal event hypotheses
  -> uncertainty-aware node generation and association
  -> edge only after physical traversal
  -> online topometric graph
```

## 2. 学习的几何结构状态

`GeometrySemanticState` 只保留部署时可得、可解释的量：

- 最多 6 个可通行出口 token：相对机器人前向的方位、开口宽度、垂直轮廓、置信度和几何不确定性；
- 相邻帧出口关系：`persistent / reveal / withdraw / dustbin`；
- 通道几何序列：宽度、高度、坡度、曲率及其因果变化；
- place/exit descriptor，仅用于后续节点与出口关联，不得旁路产生结构事件。

机器人前向是已知 route-frame 坐标锚，不是学习贡献。现有 `local_axis`
只作诊断输出；未通过独立 yaw 扰动并超过常数前向基线前，不进入论文贡献。

## 3. 结构事件因子化

不再使用一个 `hidden context -> five-class event` 头代表主方法。事件按物理语义拆成两个可审计的 Composer。

### 3.1 Action-Set Relation Composer

输入仅为五帧出口 token 及其几何、count probability 和跨帧 transport，不读
encoder context、旧 event logits、identity、pose、world、TNG 或未来帧。

它产生：

- `junction` evidence：多个稳定可执行出口及其相对布局；
- `terminal` evidence：单一稳定返回出口与路程终止关系；
- `provisional/refuse`：数量、关系或几何不一致时禁止强制提交。

出口数量不能单独决定事件；开口宽度、垂直轮廓、方位关系、持续/新生/消失与不确定性必须共同进入。

### 3.2 Metric-Change Composer

输入仅为五帧的宽、高、坡度、曲率、一阶变化及不确定性。它产生：

- `geometry-transition`：持久宽/高变化；
- `turn`：持久曲率/方位变化；
- 因果检测延迟和沿已执行轨迹的事件位置回投。

`geometry-transition` 监督必须使用已封存的持久、双向一致、过去窗口可确认的
corrected causal change-point Teacher，不得恢复旧的 12,534 帧双边窗口 mask。

### 3.3 事件提交

Composer 输出先产生带位置和不确定性的 hypothesis，再由在线关联决定新建、合并或保留
provisional node。重复观测应通过同一物理事件关联合并，不依赖一个只在开发世界有效的手写
debounce 补丁。

## 4. 在线拓扑图

### 4.1 节点

- `junction / terminal`：决策节点，来自 Action-Set Relation Composer；
- `geometry-transition / turn`：度量结构锚点，来自 Metric-Change Composer；
- `provisional`：低置信候选，不允许 loop merge 或全局任务分配。

节点保存 event type、轨迹回投位置、出口 tokens、宽/净空/坡度/曲率、descriptor、不确定性和执行状态。

### 4.2 关联

学习式关联只能使用：place descriptor 相似度、exit-token 一对一几何对应、event type 兼容性、已执行路程给出的空间候选域和不确定性拒绝。模糊匹配必须保留 provisional，不得强行 loop merge。

### 4.3 边

语义观测可以创建节点候选，但边只能在机器人真实从一个已确认节点穿越到另一个已确认节点后生成。边保存长度、宽度、净空、坡度、曲率和执行状态。

## 5. 与 Cano-like 方法的边界

Cano-like 基线使用出口峰值/数量识别路口并进行拓扑导航。GSE-Graph 的差异必须由独立证据支撑：

1. 度量几何和跨帧物理关系实际进入事件生成，不是并行输出；
2. 节点关联使用出口级几何对应与不确定性拒绝，不是固定距离合并；
3. 几何变化和转弯以因果锚点进图，边由真实穿越验证；
4. 删除 metric geometry、transport 或 refusal 会分别使事件或图指标退化。

若这四项无法由消融和闭环收益证明，不得声称方法区别于 Cano。

### 5.1 方法成立的必要创新链

相关工作复核已经排除三种单独主张：Cano 已覆盖LiDAR出口检测、时序稳定化和纯拓扑导航；PRISM-TopoMap已覆盖学习式地点关联与在线拓扑定位；Sequential Probabilistic Descriptor已覆盖时序概率描述子和不确定性过滤。因此 Action-Set Composer、learned descriptor 或 refusal 中任意一项单独通过，都不能确立GSE-Graph。

主方法必须同时具备并由独立消融证明：

1. 从因果LiDAR学习可执行出口关系；
2. 从因果LiDAR学习沿路线的宽、高、坡度和曲率变化；
3. 两类显式状态直接组成typed structural event，而非由hidden context旁路分类；
4. typed event与出口几何共同控制节点生成和拒绝关联；
5. edge只由真实穿越提交，最终带来可测的图质量与探索收益。

其中第2--3项是区别于Cano-like路线的必要条件，不是可选增强。若只能证明junction/terminal而不能证明turn/geometry-transition，当前候选方法不得进入主论文闭环。

## 6. 固定数据与 Teacher

- C01--C06：60 worlds，142,184 observations；
- C07：10 worlds，21,548 observations；
- C08：10 worlds，24,394 observations；
- C09/C10 及 M-TARE：方法、checkpoint、校准和阈值冻结前禁止读取。

corrected causal Teacher 的精确事件人口：

| split | corridor | junction | terminal | turn | geometry-transition |
|---|---:|---:|---:|---:|---:|
| C01--C06 | 114,617 | 19,743 | 5,551 | 1,482 | 791 |
| C07 | 17,113 | 3,189 | 900 | 279 | 67 |
| C08 | 19,234 | 3,676 | 1,074 | 237 | 173 |

change-point 独立 identity 为 C01--C06/C07/C08=`59/5/12`；必须报告 identity coverage，不得用相邻帧数放大有效样本。

## 7. 实现顺序

1. 封存 V2R5 三种子 token/transport/metric-geometry 组件基线；
2. 用 corrected causal Teacher 对 V2R5 显式输出做零主模型更新的容量与冲突审计；允许在C07拟合预注册固定线性诊断探针，但必须原样迁移C08，探针不得成为部署checkpoint；
3. 实现两个 Composer readiness，验证禁止旁路、排列/旋转等变、past-only mask、因果回投和 finite backward；
4. 每个 V2R5 seed 冻结 backbone/token/transport/geometry，只训练小型 Composer；
5. C07 选 checkpoint/校准/refusal，C08 一次零适配评价；
6. 感知门通过后才冻结 observation adapter 和在线图接口。

第一版禁止 Composer loss 反传到 backbone，以证明收益确实来自显式几何组合。通过后可以另立端到端消融，不得反过来用端到端结果补救因果证据。

## 8. 方法确立标准

以下全部满足前，状态不得改为 `METHOD_ESTABLISHED`：

- V2R5 显式组件在 corrected Teacher 上的三seed可观测性与冲突人口已封存；C07诊断探针与阈值不得利用C08反馈；
- 两个 Composer 的类型接口、参数量、loss、拒绝和位置回投全部冻结；
- 三 seed C07/C08 相对 V2R5 独立 event head 的 corrected-Teacher macro-F1 平均至少提高 5 个百分点；
- turn 和 geometry-transition 的 frame-F1 与 identity coverage 均提高；
- 在 precision `>=0.98`、false accept `<=1%` 时 recall `>=0.25`；
- 去掉 metric geometry、transport 或 refusal 存在跨 seed 可重复退化；
- graph adapter 的 def-use 证据确认事件和关联真正消费上述显式量。

任一核心项失败，停止进入未见拓扑图，不允许通过调图或 planner 掩盖。

## 9. 固定对照与消融

1. M-TARE original（后续闭环）；
2. Cano-like exit count + rule graph；
3. V2R5 independent hidden-context event head；
4. count + bearing only Composer；
5. full GSE-Graph；
6. full minus metric geometry；
7. full minus transport；
8. full minus refusal；
9. full single-frame；
10. GT-TNG oracle 上界。

当前 V2R5 的训练人口、输出和失败图均保留，但它不是主方法 checkpoint。

## 10. 唯一允许的方法级表示修正

若三 seed 完整显式 token-set、transport 和当前 metric geometry 在C07拟合、C08原样迁移的固定诊断中仍无法在
corrected Teacher 上产生非平凡 geometry-transition precision/recall，结论是当前
`GeometrySemanticState` 缺少纵向几何信息，不是 Composer 容量不足。

此时不能删除 Metric-Change Composer 并退回出口关系方法。唯一允许的方法级表示修正是先做零训练可见性/Teacher proof，再决定是否新增
`RouteGeometryProfile`：从当前 LiDAR 显式预测沿前向可见通道的宽、高、坡度和曲率纵向剖面。
该剖面才可进入 Metric-Change Composer。

禁止的 fallback：扩大隐藏 context event head、将 descriptor 输入 Composer、降低安全阈值、在 C08/C09 调参或用图/planner补偿。若 RouteGeometryProfile 的可见性/Teacher proof 也失败，停止当前方法方向并重新评估论文贡献。
