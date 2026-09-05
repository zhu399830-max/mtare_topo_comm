# 地下结构语义混合监督学习方案 V1

> 2026-08-29权威覆盖：主监督由planner/event标签改为程序化地图真实构造provenance；学生学习显式扫掠超椭圆基元、组合关系和跨帧对应，而不是把五类事件作为主输出。最新数据、接口、损失和停止规则见`docs/PRIMITIVE_RELATION_STRUCTURAL_GRAPH_PLAN_V1.md`。本文继续保留原始数据、因果隔离和Teacher/学生边界原则。

本文件是 MASTER PLAN V3 的 Gate 1--3 方法细化。它固定“原始数据是什么、学生看什么、老师看什么、AI 标什么、模型学什么”，避免再次把数据规模、teacher、结构类别、表示学习和拓扑节点混为一体。

当前 Gate 仍为 Gate 0，V3 数据集、teacher、模型和 checkpoint 均为 `NONE`。本文件不是数据导出或训练许可。

2026-08-10 补充：更具体的数据池隔离、程序化世界生成、Zarr schema、5-world contract pilot、输入候选值和相关工作定位已固定在 `docs/GATE1_DATA_METHOD_V1.md`。若两份文件在数据实现细节上冲突，以后者及 `contracts/gate1_dataset_contract_draft_v1.json` 为准；当前仍只允许设计审阅。

## 1. 总体决定

采用混合监督 teacher-student 路线：

```text
地下 LiDAR/位姿/真实 ray origin                 完整地图/碰撞模型/local planner
          │                                               │
          ├─ 因果学生输入                                 ├─ 客观核心 teacher
          │                                               │
          │                         标准多视图/BEV/planner overlay
          │                                               │
          │                              冻结 AI 标注器提出结构角色
          │                                               │
          │                         硬规则检查 + 人工金标 + abstain
          │                                               │
          └────────────── 有监督轻量结构模型 ───────────────┘
                                      │
                 R(theta), D(theta), exit components,
                 G_local, structural role, uncertainty
                                      │
                  时间持续性 -> 节点/边/exit stub -> 拓扑规划
```

核心原则：

- planner/地图产生的客观结构事实是核心监督；
- AI 只提出高层结构属性、辅助解释和异常候选，不能无条件覆盖客观标签；
- 人工金标与规则检查决定 AI 标签能否进入训练；
- 学生模型始终只接收部署时在线可得的因果观测；
- offline teacher/AI 可使用完整地图或后验轨迹，但这些信息不得进入学生输入；
- 第一版不端到端学习节点、拓扑图或规划器。

## 2. 四类信息必须分开

### 2.1 原始事实来源

原始数据以带时间戳的 LiDAR 点云为准，同时保存：

```text
scan timestamp
per-scan pose/state estimation
真实 LiDAR ray origin
必要时的 per-point relative time/intensity
terrain/collision 输出
world、trajectory、seed、start 和 source bag provenance
```

ROS bag/原始记录是可追溯来源，派生 `.npz`、Zarr/HDF5 shard 或 BEV 都不是独立原始数据。旧 V3/V4/V5 `.npz` 继续禁止进入新训练。

### 2.2 学生输入

学生只看当前局部点云及其因果历史。绝对 pose、world ID、trajectory ID 和未来帧只能作为 metadata，禁止作为模型特征。

主 Route A：由点云构建 terrain-relative 2.5D BEV，候选通道为 observed mask、多高度占据、terrain-relative min/max height、height span、point density、可靠时的 slope/roughness、observation age/count 和 history valid mask。只有保存真实 ray origin 才能构建 free/unknown。

备选 Route B：直接局部点云编码。点至少包含 robot-frame `x,y,z`，可用时附 intensity、relative time、observation age 和 scan index；历史点云先变换到当前机器人坐标。只有 Route A 被多高度、坡地、悬空结构或复杂洞穴的实验证据否定后才启用 Route B。

### 2.3 客观 Teacher 输入

Teacher 可以离线使用完整仿真地图、碰撞模型、terrain map、local-planner rollout 和已冻结的后验轨迹。Teacher 的可用信息必须单独写入 provenance，不能与学生输入混存为同名字段。

Teacher 输出使用显式 validity mask；不可判断不是负样本。完整地图或未来状态只能定义离线 target，不能成为在线输入。

### 2.4 AI 标注输入

通用多模态 AI 不直接读取未知含义的 `.npz` 数组。每个标注单元生成固定 annotation bundle：

```text
当前残缺点云俯视图
因果历史累计点云俯视图
固定相机参数的前/侧/等距三维视图
完整 oracle 局部地图
planner 可达方向、距离和碰撞结果叠图
不包含 world 名、绝对坐标和 split 名的样本 ID
冻结标签 schema 与术语解释
```

AI 输出必须是版本化 JSON，记录 labeler 名称/版本、prompt hash、原始响应、结构化结果、逐字段 confidence、依据、异常提示和 `abstain`。低置信或违反硬约束的标签不得自动进入训练。

## 3. 数据样本单位与结构

样本单位是空间独立的 `place_cluster`，不是单帧文件。相邻帧可组成同一位置的不同 coverage/history view，但不能重复计为独立地点。

建议逻辑结构：

```text
identity:
  sample_id, spatial_cluster_id, world, trajectory, timestamp, split
student:
  current_points, causal_history_points, relative_transforms,
  ray_origins, observation_age, valid_masks, optional_bev
teacher:
  direction_traversable, reachable_distance, direction_valid_mask,
  exit_components, local_connectivity, teacher_confidence
ai_auxiliary:
  structural_attributes, confidence, abstain, audit_state
provenance:
  source_bag, source_topics, exporter_version, teacher_version,
  labeler_version, prompt_hash, human_review_state
```

模型输入范围、历史长度、voxel/BEV 分辨率和方向预测距离不凭经验随意冻结。Gate 0 先审计 LiDAR 有效范围、local-planner horizon、collision resolution、机器人宽度和实时预算；Gate 1 再根据这些 contract 提交候选值与小样本可视化。

## 4. 标签体系

### 4.1 核心客观监督

| 标签 | 定义 | 主要来源 | 模型/规则用途 |
|---|---|---|---|
| `R(theta)` | 机器人坐标系固定方向的可通性 | planner-consistent rollout | 方向可通 head |
| `D(theta)` | 有效方向上的可达距离 | planner/collision | 距离 head |
| `E` | 连续可通方向组成的 circular exit component | `R,D` 与地图连通性 | 确定性出口提取 |
| `G_local` | 局部出口之间的可连接关系 | 完整地图局部路径 | 连接 head/规则 |
| validity/confidence | teacher 是否有资格监督 | 观测、planner 和审计 | masked loss/uncertainty |

第一版模型直接学习 `R(theta)`、`D(theta)`、必要的 connectivity 和 uncertainty；`E` 优先由冻结的 circular connected-component 规则从方向输出得到，避免同时学习两个互相矛盾的出口定义。

### 4.2 AI/人工辅助结构属性

高层角色采用多标签属性，避免把重叠结构强塞进单一类别：

```text
topological_degree: 1 / 2 / 3 / 4+
passage_shape: straight / turn / branch / chamber
width_state: narrow / normal / wide / bottleneck
vertical_state: flat / slope / multi_height
special_state: dead_end / transition / ambiguous
```

其中 degree、dead-end 等能从客观 teacher 推导的字段必须做一致性检查。AI 的主要价值是提出 passage/width/vertical 等有解释力但边界模糊的属性，而不是猜连续距离或隐藏自由空间。

`node_score`、具体 node label、最终 frontier utility 和 place identity 不进入第一版监督。

## 5. 标注与质量控制

### 5.1 Gate 1 标注 pilot

在 Gate 0 通过并批准 Data Card 后，先做小规模标注质量实验，不训练正式模型：

- 至少来自 3 个地下开发 world；
- 按结构事件和空间位置分层抽取约 300--500 个 place cluster；
- 所有样本生成客观 teacher 和 AI 标注建议；
- 至少 150 个分层样本形成独立人工金标；
- 稀有结构、AI abstain、低置信和规则冲突样本全部人工复核；
- strict test world 不进入 pilot，也不展示给 AI 标注器。

上述数量是标注可行性 pilot 的容量目标，不是正式训练数据充分性结论。实际 world、轨迹、距离和样本数必须写入获批 Data Card。

### 5.2 机器硬检查

- exit component 必须受 `R(theta)` 和 `D(theta)` 支持；
- degree/exit count 与结构属性不得无解释冲突；
- 无 ray origin 时 free/unknown 标签必须无效；
- teacher invalid 方向不得被填成 0；
- 标签旋转后必须按机器人坐标同步 circular shift；
- world 名、绝对 heading、trajectory ID 不得进入 AI prompt 或模型特征；
- AI 修改客观标签必须进入冲突队列，禁止静默覆盖。

### 5.3 Pilot 通过规则

正式阈值在 pilot run spec 中由用户批准。第一版建议检查：AI 辅助属性对人工金标的 macro-F1、稀有类 recall、逐字段置信度校准、abstain 率、规则冲突率，以及 planner teacher 对人工复核的方向/出口一致性。

若 AI 标签不通过，它只能作为人工界面提示，不进入 loss；核心 planner/map 监督路线仍可继续。若客观 teacher 不通过 planner consistency，则整个 Gate 1 停止，不能靠 AI 标签绕过。

## 6. 正式数据规模原则

Gate 1 必须先提交完整 inventory，再冻结正式规模。容量规划的优选目标是：

- 训练至少 6 个独立地下 world，validation 至少 2 个地下 world；
- strict test 至少 2 个完全隔离地下 world；
- 每个开发 world 至少 4 条独立轨迹，并改变 start/seed/coverage；
- 按约 0.5--1.0 m 的候选空间间隔构建 cluster，最终间隔由机器人尺度和局部 planner contract 决定；
- 第一版容量目标约 10,000--30,000 个有效 place cluster，并单独报告有效 junction、branch、dead end、chamber、bottleneck、slope 和 multi-height 事件。

这些是准备 Data Card 时的目标，不是脱离 inventory 的硬性充分条件。若可用独立地下 world 不足，应增加程序化地下几何变体或审计合适的地下数据源；不得用 forest/campus/generic indoor 填补地下验证，也不得把相邻帧和数据增强计为独立样本。

## 7. 模型主路线

### 7.1 Route A：监督式轻量 BEV 结构模型

```text
terrain-relative causal BEV
  -> lightweight CNN/ResNet encoder
  -> polar/radial directional aggregation
  -> R(theta), D(theta), connectivity, uncertainty heads
  -> deterministic exit component extraction
  -> auxiliary structural-attribute head and z_role
```

核心 loss 只覆盖获得批准的 objective target：masked BCE/focal for traversability、masked Huber for distance、必要时的 masked connectivity loss 和 uncertainty calibration。AI/人工结构属性只作为低权重 auxiliary loss，并必须做 `objective-only` 对 `objective + AI auxiliary` ablation。

Gate 2 评价表示稳定性，不建图、不调节点阈值：同位置不同 coverage、一致旋转、稀疏/遮挡扰动、跨地下 world 检索与 objective probe。Gate 3 再冻结并评价显式语义 head。

### 7.2 Route B：直接点云编码

只有 Route A 在预先定义的信息保真审计中对多高度、坡地、悬空或洞穴结构持续失败，才启用 point-based encoder。输入仍使用同一份原始点云和同一套 teacher/split；禁止同时重建一套不同数据集。

Sparse 3D 只有直接 point route 仍被证据否定且在线预算允许时才考虑，不作为并行第三路线。

### 7.3 自监督的重新定位

自监督不再是 Gate 2 第一版主方法。masked geometry reconstruction、same-place consistency 和 rotation equivariance 只在以下情况启用：

- 有大量合格但未标注的开发地下数据；
- 监督 baseline 已建立；
- 单独 run spec 明确它是 warm start 或辅助正则；
- 与纯监督模型做冻结 split 的 ablation；
- strict test 仍完全禁止进入 SSL。

自监督未带来跨 world 稳定性或 downstream objective 提升时立即删除，不继续叠加 loss。

## 8. 固定对照和归因

Gate 2/3 的最小对照链：

```text
B0  online partial-BEV geometry/planner rule baseline（不学习）
M1  objective teacher supervised BEV model（主方法）
A1  M1 + audited AI structural auxiliary（单独 ablation）
R1  direct point model（仅 BEV failure 后启用）
S1  supervised + SSL warm start（仅标签不足且获批时启用）
```

同一阶段最多运行主方法、明确 baseline 和一个已被证据触发的备选。A1 用于回答 AI 辅助标签是否有增益；不能因其名称含 AI 就成为默认主方法。

## 9. Gate 对齐

- Gate 0：冻结传感器、planner/collision 接口、地下 benchmark 和 baseline，不选训练样本。
- Gate 1：冻结 point-cloud source、因果输入、objective teacher、AI schema、人工审计、split 和数据充分性；只做标注/teacher pilot，不训练正式模型。
- Gate 2：训练和验证监督式稳定结构表示；不建拓扑图。
- Gate 3：验证显式方向语义、出口、连接、角色和 uncertainty；不让 semantic class 直接决定节点。
- Gate 4：使用冻结语义输出，通过规则与持续性构建 node/edge/exit stub。
- Gate 5--8：保持原 shadow、单机、多机和最终公平对比边界。

改变监督策略不允许跳过 Gate 0/1，也不自动提高任何历史结果的状态。

## 10. 停止条件

出现下列任一情况立即停止受影响工作并报告：

- 原始记录缺 ray origin、时间同步或可靠 pose，导致输入 contract 不成立；
- AI/人工标签 schema 不能稳定区分结构属性；
- teacher 与 local planner 一致性不达标；
- 数据只有少量 world/轨迹，空间重复冒充规模；
- BEV 与点云路线同时扩张而没有预先定义的 failure 证据；
- AI 标签提高离线分类但不改善 objective semantics 或图指标；
- test world 进入 AI prompt、人工开发审阅、统计或选择流程；
- 通过增加类别、head、loss 或阈值掩盖标签缺陷。

## 11. 当前下一步

本节已被 `docs/PRIMITIVE_RELATION_STRUCTURAL_GRAPH_PLAN_V1.md` 取代。当前唯一下一步是 Gate 3 的 P0 构造监督可行性验证：只用解析微场景和少量只读 C01 样本，检查程序构造记录能否生成唯一、视场裁剪、无未来信息泄漏的基元、变换、组合关系、端口和表面来源监督。P0 不训练、不建图、不运行 M-TARE，也不读取 C07--C10。
