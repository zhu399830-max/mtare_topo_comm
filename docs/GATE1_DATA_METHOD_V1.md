# Gate 1 数据与方法设计 V1

状态：`DRAFT_FOR_USER_REVIEW`  
日期：2026-08-10  
适用范围：MASTER PLAN V3 Gate 1--3  
当前项目 Gate：Gate 0，`GATE_MIXED`

本文件回答六个问题：比较地图能否参与训练、数据从哪里来、输入到底是什么、标签如何产生、AI 标注与对比学习如何取舍、先生成什么再训练什么。

本文件只是一份设计合同。它不授权启动 CPU/Gazebo/Isaac 数据采集、导出数据、调用 AI 标注、训练模型或进入 Gate 1。

## 1. 最终决定

主方法不是“AI 打标签”和“对比学习”二选一，而是：

```text
新建程序化地下世界的完整几何/拓扑/碰撞信息
                    │
                    ├─ 客观离线 teacher：R(theta), D(theta), exits, G_local
                    │
真实部署可见的残缺 LiDAR + 因果历史
                    │
                    └─ 轻量 BEV/polar student ──> 结构事实 + uncertainty
                                                 │
                                   时间持续性与规则建拓扑图
```

- 第一版采用客观监督学习。teacher 由程序化真值、机器人尺寸、碰撞检查和冻结的 local-planner 规则产生。
- AI 不是核心 teacher。第一版 5-TNG contract pilot 不把 AI 标签放进 loss。以后 AI 只能提出结构名称、歧义样本和审计建议，且必须允许 `abstain`。
- 对比/自监督不是主训练方法。监督 baseline 通过后，才允许对合格的未标注真实地下开发数据做独立消融。
- 最终部署输入仍是 LiDAR 点云构建的因果 terrain-relative 2.5D BEV；原始点云必须完整保存，便于以后验证 BEV 是否丢失多高度信息。
- M-TARE 原比较地图只用于最终同条件闭环比较，禁止以任何形式进入开发。

## 2. 数据池与不可跨越的边界

| 数据池 | 用途 | 是否可训练 | 隔离要求 |
|---|---|---:|---|
| `SYN_TRAIN` | 新建程序化地下世界，监督训练 | 是 | 与 validation/test 按完整 world/seed 隔离 |
| `SYN_VAL` | 模型、阈值和 checkpoint 选择 | 否，只验证 | 完整 held-out world |
| `SYN_DEV_TEST` | 未见拓扑组合的开发期测试 | 否 | 不参与选择 |
| `REAL_SSL_TRAIN` | 经审计的 LAMP/SubT 未标注开发站点 | 仅可选 SSL | 不得包含 real test 站点 |
| `REAL_SITE_TEST` | 完整未接触真实站点 | 否 | 不进 prompt、统计或人工开发审阅 |
| `MTARE_BENCHMARK_ONLY` | 原 M-TARE 地图公平闭环比较 | 否 | 绝对 benchmark-only |
| `STRICT_HIDDEN_TEST` | 新建封存世界/拓扑族 | 否 | 冻结后开发者不可见 |

`MTARE_BENCHMARK_ONLY` 禁止用于监督、SSL、归一化、AI prompt、人工标注 pilot、数据增强调参、teacher 阈值标定、checkpoint 选择和失败后查看再改模型。由于现有原 M-TARE 世界已经被历史实验接触，它们只能称为“M-TARE parity benchmark”，不能冒充 strict unseen test。

最终报告必须同时包含：

1. 在原 M-TARE 地图上、相同 start/seed/sensor/local planner/runtime 的 parity comparison；
2. 在至少两个完全封存的新世界/拓扑族上的 strict generalization；
3. 在完整未接触真实地下站点上的 domain generalization（具备合格数据时）。

## 3. 要生成的不是五张地图，而是“图先行”的地下世界族

2026-08-10 路线修订：Phase 1--3 的主传感器级合成后端为 Open3D CPU RaycastingScene，不再以 Isaac 为前置条件。优先直接复用能够固定版本和重放的 Cano TNG/表面点云/native mesh 代码，只通过项目 adapter 补 provenance、pose、ray origin 和标签。必须先生成可追溯结构图，再生成经用途门禁的 perception mesh；动态 Gazebo collision mesh 可由另一合格后端生成，但两者必须共享 topology/centerline provenance。Isaac 只保留为可选后验传感器域实验。

### 3.1 生成顺序

```text
seed + topology grammar
  -> 有向/无向结构图、中心线、出口与连接真值
  -> 管廊/洞穴截面、宽高、曲率、坡度和连接腔体
  -> collision/free-space/navigation geometry
  -> clutter、表面粗糙度、材质、传感器噪声随机化
  -> CPU idealized LiDAR scan + valid mask + pose + ray origin
  -> 完整地图上的 planner/collision teacher
```

每个 world 必须保存 generator 版本、`topology_seed/geometry_seed/clutter_seed/trajectory_seed/sensor_seed`、参数、TNG、中心线、perception mesh hash、collision mesh hash、可通区域、clearance、坡度/高度和导出器版本。CPU perception 与 Gazebo collision 资产必须共享 topology/centerline hash；若 geometry 不同，必须明确记录并做固定 pose sensor parity，禁止只凭相同 world 名称假定同域。

### 3.2 必须覆盖的结构族

- 直道、弯道、T/Y/X 分叉；
- 环路、死胡同、宽窄突变、瓶颈；
- chamber/大厅及多出口腔体；
- 坡道、不同高度通道、竖井邻域；
- 不同截面、宽度、曲率、粗糙度和遮挡；
- 不影响拓扑真值但影响观测的石块、设备、支护和 LiDAR 噪声。

划分原子是 TNG topology parent，不是 mesh、相邻帧或 observation。同一 TNG 的全部 geometry、clutter、trajectory 和派生数据必须进入同一 split。数据增强产生的旋转或稀疏点云不增加“独立拓扑数”。

## 4. 先做 5-TNG contract pilot，不直接做正式训练集

第一步只验证数据链是否正确：

| Pilot topology parent | 最低结构覆盖 |
|---|---|
| `P01_straight_turn` | 直道、连续弯道 |
| `P02_branch_deadend` | T/Y 分叉、死胡同 |
| `P03_loop_bottleneck` | 环路、宽窄突变、瓶颈 |
| `P04_chamber_multiexit` | chamber、三个以上出口 |
| `P05_slope_multiheight` | 坡道、多高度/竖向邻域 |

单图 CPU 合同 PASS 后，即时 perception pilot 修正为 5 个独立 TNG parent、每个 TNG 1 个 native perception mesh、每个 TNG 50 个 canonical anchor、每 anchor 3 个固定 yaw view，共 `5 × 1 × 50 × 3 = 750` diagnostic observation。它对应 5 个 mesh world和 250 个 canonical anchor，只用于验证跨拓扑生成、sensor、teacher、泄漏与证据合同，不用于声称模型充分训练。

旧的每 TNG 2 个 geometry、10 mesh、1,500 observation 设计不是删除，而是延后为 paired invariance pilot。原因是 native Cano mesh 已被动态导航门禁否决，当前无法用 footprint/planner rollout 认证 operational connectivity 不变；必须先有 navigation-grade collision backend，再单独审批该配对实验。

每个位置必须构造三类配对：

- 同一局部 TNG、不同且已认证 geometry 的正对，以及同一位置不同残缺 coverage，用于检验结构稳定性；
- 当前可见几何相似、但 oracle 隐藏连接不同的困难负对，用于防止模型背世界外观；
- 同一观测的旋转副本，用于检验输出的 circular equivariance。

完整 TNG 只作 oracle/teacher/evaluation。不可见连接必须标记 `UNKNOWN/MASKED`；`tng_id`、geometry variant ID、绝对位置和完整图禁止进入 student tensor。若几何变化导致 footprint 可通性、exit set、cycle 或 dead-end 改变，该 pair 必须改标为 connectivity-changing variant，禁止作为正对。

Pilot 的退出条件：

- 5 类 world 都能由 seed 重放，结构图、mesh、collision 与导出 hash 完整；
- 从原始 LiDAR 重建出的模型输入与在线实现逐通道一致；
- teacher 与冻结 local-planner/collision 判定的一致性通过预定阈值；
- world-disjoint 泄漏检查为零；
- 每类结构都有点云、BEV、oracle、R/D 和 exit overlay 的科研可视化；
- M-TARE benchmark world 未出现在任何 manifest 中。

Pilot 未通过时只修数据生成器/teacher/contract，不训练模型。

## 5. 原始存储与模型输入

### 5.1 原始 episode 存储

主存储建议使用 episode-level Zarr，避免每个样本一个 `.npz` 造成百万小文件和 schema 漂移。逻辑字段：

```text
points_xyz[N_total, 3]          float32, sensor frame
point_offsets[T + 1]            int64
point_intensity[N_total]        optional
point_ring[N_total]             optional
point_relative_time[N_total]    optional
ray_origin[T, 3]                float32
pose_world_sensor[T, 7]         float64
timestamp_ns[T]                 int64
world_id, episode_id, seed
geometry_hash, generator_version, sensor_config_hash
```

`manifest.parquet` 或 `manifest.jsonl` 记录 observation、place cluster、split、来源和 teacher 版本。派生 tensor 用独立 shard-level Zarr。`.npz` 只允许作为少量 debug export，不是正式数据集格式。

### 5.2 第一版在线输入

候选 contract：机器人中心 40 m × 40 m、0.25 m/cell、160 × 160；当前帧与过去 4 s 因果融合各一组 8-channel BEV。最终范围必须在 Gate 0 根据 LiDAR 有效距离、local-planner horizon 和实时预算确认。

每组通道固定为：

1. `observed_surface`：真实回波覆盖；
2. `ray_free_evidence`：仅由真实 ray origin 与射线形成；
3. `height_min_rel_terrain`；
4. `height_max_rel_terrain`；
5. `height_span`；
6. `log_point_density`；
7. `local_ground_slope`；
8. `observation_age`。

第一版不另设 roughness 通道：局部高度跨度与点密度已提供粗糙程度证据。只有输入信息审计证明这两个通道不足时，才能通过版本化 schema 变更新增 roughness。

未知区域由 observed/free/valid 的组合表达。无真实 ray origin 时，`ray_free_evidence` 必须 invalid，禁止把“没有点”当作 free。

模型输入严禁包含绝对坐标、world ID、trajectory ID、未来帧、完整地图或 oracle teacher。历史点先变换到当前机器人坐标。

## 6. 客观 teacher 与标签

固定 32 个机器人坐标系方向 bin。teacher 从完整几何、机器人 footprint、clearance、碰撞检查和冻结 local-planner 产生：

- `R(theta)[32]`：每个方向是否可通；
- `D(theta)[32]`：可通方向上的最大可达距离；
- `direction_valid[32]`：是否有资格进入 loss；
- `exit_components[<=8]`：由相邻可通方向的 circular component 确定；
- `G_local[8,8]`：局部出口之间是否在 oracle 局部区域内连通；
- `teacher_confidence`、`clearance`、版本与 provenance。

不可判断方向必须 mask，不得填成负样本。出口优先由冻结规则从 R/D 提取，第一版不同时训练两个相互竞争的出口定义。

`z_role` 不使用 AI 直接定义。它由 objective task 学出，再用 same-place consistency、rotation equivariance、跨 world probe 和后续图指标验证。

## 7. 模型和训练顺序

### 7.1 主模型 M1

```text
current BEV + causal BEV
  -> lightweight CNN/ResNet
  -> polar/radial aggregation
  -> R, D, optional G_local, uncertainty heads
  -> deterministic exit extraction
```

核心 loss：masked BCE/focal for R、masked Huber for D 和 uncertainty calibration。第一版只训练 `R/D/U`；`G_local` 仅作 oracle probe，真实 traverse/geometry verification 后再建立 edge。第一版不加 GRU，不端到端建图，不直接输出最终 waypoint。

### 7.2 对照顺序

| ID | 方法 | 回答的问题 |
|---|---|---|
| `B0` | partial-BEV 几何/规则，不学习 | 学习是否真的有必要 |
| `B1` | Cano-like range/depth-image exit CNN | 相对最接近已有方法是否有增益 |
| `M1` | objective-supervised BEV/polar model | 主方法是否学到 planner-consistent 结构事实 |
| `A1` | M1 + same-place/rotation auxiliary | 任务相关自监督是否改善跨 world 稳定性 |
| `A2` | M1 + audited AI role auxiliary | 仅在 AI pilot 通过后检验高层语义增益 |

实施顺序固定为：先用小规模数据完成 `B1` 复现和感知合同核验，再运行 `B0` 与 `M1` 主比较；`A1/A2` 不自动启用。B1 与 M1 必须使用同一 canonical LiDAR、world split 和原始 observation。

### 7.3 何时允许对比学习

只有同时满足以下条件才提案：

- M1 的 objective-supervised baseline 已经稳定；
- 有合格且与 test 隔离的 LAMP/SubT 开发站点；
- 监督数据或 sim-to-real 稳定性出现明确瓶颈；
- 单独 run spec 和 Data Card 获批；
- 与 M1 使用完全相同的 frozen split 做消融。

首选辅助是同一 place 不同 coverage 的一致性与旋转等变性，而不是不区分任务结构的通用 instance contrast。SSL 不能产生 oracle 隐藏连通性或 planner 可通距离真值。

### 7.4 AI 的准确位置

AI 只能用于：高层多标签结构名称建议、歧义检测、可视化审计和人工复核排序。它不允许直接猜 `D(theta)`、隐藏 free space 或 `G_local`。

AI 标签进入 loss 前必须另做 300--500 place 的 annotation pilot、至少 150 place 的分层人工金标，并报告 macro-F1、稀有类 recall、校准、abstain 和冲突率。未通过则 AI 只作为检查工具。

## 8. 正式数据容量建议

5-TNG perception pilot 与后续 navigation-grade geometry/parity 门禁通过后，才提交正式 Data Card。第一版容量上限为：

- 80 个训练、10 个 validation、10 个 development-test TNG topology parent；
- 每个 TNG 至少 2 个认证 geometry variant，共 200 个 mesh realization；
- 每个 TNG 500 个 matched canonical anchor，每个 geometry-specific anchor 5 个残缺 coverage/view；
- 共 100 个独立拓扑、50,000 个 canonical anchor、100,000 个 geometry-specific place instance 和 500,000 个 observation；
- 使用 20/40/80 train-TNG 的嵌套学习曲线逐级扩展，不一次性盲目生成；
- 另有至少两个封存 strict-test 拓扑族，每族候选 10 个 world，不计入 50 万开发 observation；
- 真实数据按完整 site 切分，LAMP/SubT 只在 provenance 与 ray/pose 质量审计通过后进入相应池。

这些数字是容量上限，不是提前宣布“足够”。每档必须同时报告 topology parent、mesh realization、canonical anchor、geometry-specific place、observation、拓扑统计和固定 validation 学习曲线；若尚未饱和，应增加拓扑族和 topology seed，而不是仅在同一 TNG 采更多帧。全量 `16×160×160` float16 BEV 约需 381 GiB，正式存储默认保存 raw episode 并确定性在线派生或使用压缩 shard，禁止无计划缓存全部未压缩 tensor。TNG 反事实配对与防泄漏细则见 `docs/TNG_COUNTERFACTUAL_DATA_CONTRACT_V1.md`。

## 9. 与最终 M-TARE 比较的链路

```text
离线：新建世界监督学习 -> 固定模型/阈值
在线：LiDAR -> causal BEV -> R/D/exits/uncertainty
     -> 时间关联 -> node/edge/exit-stub topology
     -> 全局目标与多机器人任务分配
     -> /way_point
     -> 原 M-TARE localPlanner/pathFollower/control
```

原 M-TARE 方法走原 `tare_planner_node`；新方法替换该全局节点，但共用 sensor、state、terrain/collision、local planner、controller、world/start/seed/runtime。比较的是“全局表示与规划”，不是换了一套底层运动能力。

## 10. 已识别的新颖性风险

已有工作已经做过“程序化地下世界 + 合成 3D LiDAR + 轻量 CNN 检测通道出口 + 纯拓扑导航”。因此，本项目不能把“用神经网络从点云发现出口并建拓扑图”当作主要创新。

必须把贡献压在：

1. 因果残缺观测下，与保留的 local planner 一致的 R/D/G_local 和 uncertainty；
2. 可解释结构事实如何形成稳定 node/edge/exit-stub 图，并用 preservation/connectivity 指标验证；
3. 该图如何真正替换 M-TARE 全局 frontier 规划，而不是只做感知 demo；
4. 多机器人通信约束下的任务分配、冗余和探索效率；
5. parity benchmark 与 sealed strict test 双重证据。

如果这些贡献不能在 Gate 3--8 被验证，项目应停止宣称方法创新，而不是继续增加网络、loss 或类别。

## 11. 下一步与审批点

当前只请求用户审核这份设计，尤其是：benchmark 隔离、5-world pilot、输入 40 m/0.25 m/4 s 候选值、客观监督主路线，以及 AI/SSL 的降级位置。

Gate 0 冻结后，下一份材料必须是 5-world pilot 的 Data Card，列明生成参数、准确样本量、磁盘/算力、teacher 校验阈值、可视化清单和失败停止条件。用户批准前不启动生成。

2026-08-10 实施补充：能直接运行和复现的 Cano generator/adapter/CPU raycast 优先作为 baseline 使用；当前 checkout 没有论文 CNN 训练代码，故 B1 是按论文接口实现的 adapted reproduction。Isaac 接入部分已被 `docs/CPU_RAYCAST_LIDAR_BACKEND_V1.md` 取代。单图 CPU Raycast LiDAR contract 已 PASS；当前只有修正后的 5-parent/5-mesh/750-observation perception pilot 提案，尚未批准或执行。
