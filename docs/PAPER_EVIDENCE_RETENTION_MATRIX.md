# 论文证据保留矩阵

状态：`ACTIVE_RETENTION_POLICY`  
主线：五帧因果LiDAR学习扫掠几何基元及关系，并形成执行验证的在线结构语义图。

## 1. 使用原则

旧工作不按“成功/失败”决定去留，而按能否支持论文主张决定：

- **正式对比**：必须在最终冻结数据、传感器、轨迹、评价指标和随机种子下重新运行；历史数字不能直接与新方法拼表。
- **消融**：只改变一个明确模块，其余输入和训练/评价合同相同。
- **失败分析**：系统正常完成且`error=null`，结果能说明某种表示或机制为什么不够。
- **复现附录**：依赖、环境、脚本接口、freeze或绘图错误；不得作为方法优劣证据。

在最终表格与附录映射完成前，不删除下列run的图、metrics、summary、checkpoint、轨迹、图结构或seal。

这也适用于所有此前已完成的出口模型、事件槽、Composer、规则关联、几何资格和旧闭环工作：能在统一合同下回答论文问题的，重跑后进入正式对比或单变量消融；运行完整且能否定某个方法假设的，进入失败分析；只反映脚本、依赖或资源问题的，不包装成算法结果。既有论文图片及其矢量源、生成脚本、原始metrics和seal在最终排版完成前一律保留。

### 1.1 进入论文的硬条件

旧资产只有满足下列口径之一才进入论文，避免“实验很多但结论不成立”：

1. **主表对比**：与完整方法使用相同世界、传感器输入、因果历史、轨迹、seed、预算和评价代码；不满足时必须按最终合同重跑。
2. **模块消融**：以完整方法为母体，只删除或替换一个组件；数据、训练轮数、阈值选择和图/规划后端保持相同。
3. **设计失败分析**：运行正常完成、输入没有泄漏、指标能够直接否定一个假设，并能解释主方法为何采用当前设计。
4. **系统复现附录**：只说明环境、资源和fail-closed机制，不进入性能排名，也不包装成算法失败。

每个被采用的旧run必须最终登记为：`论文问题 → 对照变量 → 指标/图片 → 原始run与seal → 正文/补充材料位置`。无法完成这条映射的资产不占论文篇幅，但在清理前仍保留紧凑证据。

## 2. 最终正式对比

| 方法 | 论文作用 | 当前资产 | 最终要求 |
|---|---|---|---|
| 原始M-TARE | 原系统基线 | Gazebo、定位、局部规划、控制接口均保留 | 在最终单/多机器人同世界同seed合同下重跑 |
| Cano-like出口模型+规则图 | “只学出口，不学几何基元关系”基线 | 三个M1D checkpoint、ROS导出、旧因果图与纵向切片可复用 | 冻结新测试后重新推理和建图；旧数字只作实现依据 |
| 非学习鲁棒基元拟合 | “有几何但不学习”基线 | C07正式完成：64,644序列，primitive F1=`0.2983`、coverage=`0.1753`、attachment F1=`0.0058`、overlap F1=`0`、Chamfer=`17.31 m`；`gate3_20260830_primitive_relation_nonlearning_c07_v1_seed0` | 数字与图已冻结；主模型必须在同一输入和指标上超过该下界 |
| 完整基元关系模型 | 主方法 | P1数据、P1b Teacher、P2模型/readiness已完成 | 三seed训练、C07选择、C08一次迁移、离线图和闭环尚未完成 |
| GT-TNG oracle | 上界与系统可达性检查 | TNG、splines、teacher和oracle图接口保留 | 只作评价/规划上界，禁止进入学生输入或阈值选择 |

## 3. 必要消融

| 消融 | 要回答的问题 | 可复用证据/实现 |
|---|---|---|
| 单帧 vs 五帧 | 时间配准是否改善遮挡下的基元和关系恢复 | 当前五帧reader、relative odometry和temporal head |
| 8/16/32槽 | 容量是否影响真实可见结构 | `gate3_20260830_primitive_slot_capacity_audit_v1r_seed0`：overflow=`264/15/0`，正式冻结32槽 |
| 去掉构造参数监督 | 仅靠表面/射线重建能否学到正确可解释参数 | 六类loss中的参数项可独立关闭，必须保持其他项不变 |
| 去掉表面重建 | 参数回归是否会过拟合Teacher而不符合扫描 | 六类loss中的surface/ray项可独立关闭 |
| 去掉关系头 | 显式学习连接是否优于最近邻几何关联 | 主模型几何输出+冻结非学习关联器 |
| 去掉时序对应 | 跨帧实体一致性是否降低重复节点和错误关联 | temporal correspondence/dustbin Teacher已完整 |
| 去掉不确定性拒绝 | 拒绝机制是否换取更低false loop merge | uncertainty head和provisional-node策略 |
| 去掉edge几何属性 | 宽高、坡度、曲率是否改善探索目标和通信 | 后续同图同planner消融 |
| 三种截面 | 是否只记住椭圆生成器外观 | ellipse、rounded-rectangle、C1-mixed共240个配对实现 |

## 4. 可进入失败分析的旧科学实验

这些run系统均完成且具有方法含义，可用于“为什么从出口/事件槽转向底层基元关系”的实验链。它们不自动成为最终主表的公平对照。

| 旧路线与证据 | 主要结果 | 可支持的论文结论 |
|---|---|---|
| Spatial Event Set：`gate3_20260828_gse_spatial_event_set_capacity_v1_seed0` | 三seed F1=`0.130/0.186/0.144`，匹配位置MAE约`2.49–2.63 m` | 直接把扫描压成少量离散事件槽，几何定位和集合分解不足 |
| Geometry-Anchored Joint：`gate3_20260828_gse_geometry_anchored_joint_capacity_v1_seed0` | 三seed F1=`0.086/0.093/0.069`，低于旧规则baseline `0.390` | 只有几何锚点、没有正确基元分解和关系监督仍不够 |
| Structured Polar Multi-depth：`gate3_20260828_gse_structured_polar_multidepth_capacity_v1_seed0` | 三seed F1=`0.167/0.144/0.145`，MAE约`2.0 m` | 方位分箱和多深度槽不能稳定表达组合隧道结构 |
| Exact-one Event：`gate3_20260829_gse_structured_exact_one_event_training_v1_seed0` | C08 precision=`0.991`但recall=`0.357`、F1=`0.583` | 强制每次只提交一个事件可抑制重复，却会漏掉大量真实结构 |
| Relational Exit Transport：`gate3_20260829_gse_relational_exit_transport_training_v1_seed0` | ensemble F1=`0.665`，低于baseline `0.744`；仍有18个duplicate | 出口token传输和关系提交不能替代底层物理基元身份 |
| Relational Commit Policy：`gate3_20260829_gse_relational_commit_policy_feasibility_v1_seed0` | C08 precision=`0.997`、recall=`0.569`、duplicate=0 | 高精度拒绝能保证安全提交，但固定事件接口召回上限明显 |
| Dual Composer：`gate3_20260829_gse_dual_composer_three_seed_training_v1_seed0` | C08 F1=`0.529`，旧baseline=`0.589`，平均下降`0.0597` | 把预测几何再交给事件Composer没有产生增益，且增加规则/模块层级 |
| RouteGeometryProfile：`gate3_20260829_gse_route_geometry_profile_proof_v1_seed0` | 单帧平均仅支持约`1.55–1.89/5`段；transition完整五段为0 | 从单帧直接监督固定前向剖面受可见性限制，因此改用五帧因果配准与可见基元 |
| 旧离线图关联：`gate4_20260826_gse_offline_topology_validation_v3_seed0` | 最好precision=`0.9646`，false-loop=`3.54%`，未达`0.98/1%` | 距离/事件关联会错误合并相似走廊，必须学习关系并允许拒绝合并 |
| Route-conditioned residual：`gate3_20260828_gse_route_conditioned_event_residual_training_v1_seed0` | node recall=`0.675`、edge recall=`0.308`，false loop=0 | 小型规则残差可保守但无法恢复完整图边，不能作为核心创新 |

对应的正式图文件、figure source和seal一并保留；论文正文只选能形成连续论证的2–4项，其余放补充材料。

### 4.1 已存在、禁止清理的候选论文图

| 证据主题 | 已存在的可复现图片 | 预定用途 |
|---|---|---|
| 程序化地图与组合节点 | `results/gate3_semantics/gate3_20260830_primitive_construction_supervision_feasibility_v1_seed0/previews/primitive_construction_graph.{png,pdf,svg}` | 数据生成图中的真实C01拓扑/扫掠基元示例 |
| 三种截面与C1变换 | `results/gate3_semantics/gate3_20260830_swept_superellipse_contract_v1_seed0/previews/swept_superellipse_contract.{png,pdf,svg}` | 方法图中的椭圆、圆角矩形及连续变换 |
| LiDAR与逐射线基元来源 | `results/gate3_semantics/gate3_20260830_primitive_provenance_corrective_v1_seed0/previews/primitive_provenance_teacher.{png,pdf,svg}` | Teacher生成图，展示同一扫描的range和primitive identity |
| 离散事件槽容量失败 | `results/gate3_semantics/gate3_20260828_gse_spatial_event_set_capacity_v1_seed0/metrics/capacity/gse_spatial_event_set_capacity.{png,pdf,svg}` | 正文方法动机或补充失败分析 |
| 几何锚点事件失败 | `results/gate3_semantics/gate3_20260828_gse_geometry_anchored_joint_capacity_v1_seed0/metrics/capacity/gse_geometry_anchored_joint_capacity_v1.{png,pdf,svg}` | 与底层基元分解对照 |
| 极坐标多深度槽失败 | `results/gate3_semantics/gate3_20260828_gse_structured_polar_multidepth_capacity_v1_seed0/metrics/capacity/gse_structured_polar_multidepth_capacity_v1.{png,pdf,svg}` | 说明固定方位/深度槽的表达上限 |
| Exact-one精度/召回权衡 | `results/gate3_semantics/gate3_20260829_gse_structured_exact_one_event_training_v1_seed0/metrics/selection/gse_structured_exact_one_event_selection_v1.{png,pdf,svg}` | 不确定性拒绝与漏检分析 |
| 出口关系传输 | `results/gate3_semantics/gate3_20260829_gse_relational_exit_transport_training_v1_seed0/metrics/selection/gse_relational_exit_transport_selection_v1.{png,pdf,svg}` | “只学出口关系”基线失败证据 |
| 保守提交策略 | `results/gate3_semantics/gate3_20260829_gse_relational_commit_policy_feasibility_v1_seed0/metrics/feasibility/gse_relational_commit_policy_feasibility_v1.{png,pdf,svg}` | 拒绝机制的高精度/低召回边界 |
| Dual Composer | `results/gate3_semantics/gate3_20260829_gse_dual_composer_three_seed_training_v1_seed0/metrics/evaluation/gse_dual_composer_three_seed_training_v1.{png,pdf,svg}` | 规则Composer不增益的消融动机 |
| 单帧固定剖面可观测性 | `results/gate3_semantics/gate3_20260829_gse_route_geometry_profile_proof_v1_seed0/metrics/profile_proof/gse_route_geometry_profile_proof_v1.{png,pdf,svg}` | 采用五帧因果输入的依据 |
| 保守规则残差图 | `results/gate3_semantics/gate3_20260828_gse_route_conditioned_event_residual_training_v1_seed0/artifacts/evaluation/gse_route_conditioned_event_residual.{png,pdf,svg}` | 规则方法节点/边召回失败分析 |
| 同输入非学习基元下界 | 正式论文包：`docs/figures/gse_graph/primitive_relation_nonlearning_baseline.{png,pdf,svg}`；密封源：`results/gate3_semantics/gate3_20260830_primitive_relation_nonlearning_c07_v1_seed0/previews/nonlearning_c07_baseline.{png,pdf,svg}` | 主文Figure 3感知基线图 |

每组图片旁的`figure_source.json`、summary、配置和run seal是图片可复现性的组成部分，必须与图片一起保留。旧离线图关联run目前只有密封的`parameter_sweep.jsonl`和summary，没有合格图片；若进入正文，应从这两个密封源生成新的确定性图，不能手工抄数。

## 5. 主方法的数据与实现证据

| 证据 | 已证明内容 | 论文位置 |
|---|---|---|
| `primitive_relation_method_overview` | 当前主方法完整链条；无实验结果值，manifest=`d2d6a23f...` | 正文Figure 1，禁止由旧event/exit图替代 |
| `primitive_relation_dataset_overview` | P1a/P1b密封源中的实际C01地图、截面、range与训练期来源编码；固定frame0且零C09/C10，manifest=`a517298d...` | 正文Figure 2与数据/Teacher说明 |
| 程序化world/TNG/mesh可视化 | 树形、含环、复杂三维、坡道和stacked tunnel的环境多样性 | 数据集图和方法输入图 |
| `gate3_20260830_swept_superellipse_contract_v1_seed0` | ellipse/rounded/C1混合的统一连续参数化与解析精度 | 方法中的基元定义、补充材料 |
| `gate3_20260830_geometry_variant_inventory_v1r2_seed0` | 80父世界、8,039 edges、24,117三形状基元 | 数据统计与可复现性 |
| P1a corrected | 757,290帧、8.724B rays、逐射线构造来源无损 | 数据生成流程与Teacher provenance |
| P1b V1R | 564,378个五帧序列及几何/连接/重叠/时序标签 | 监督生成图与数据统计 |
| P2 readiness V1/V1R | 揭示并修复Teacher槽位近似平局；V1R 30/30、全部梯度有效 | 实现细节/补充材料，不作为性能结果 |
| 非学习基元C07 baseline | 手工拟合高precision但低coverage且关系失败，说明学习组合关系的必要性；正式论文包manifest=`54e080ae...`，密封源图继续保留 | 正文Figure 3、baseline表、方法动机和失败案例 |

## 6. 只进入复现附录的错误

以下内容保留summary、trace和seal，但不进入方法对比图表：

- P1b V1漏传`source_run`；
- Torch sidecar选错、缺`pytest`；
- CUDA确定性缺`CUBLAS_WORKSPACE_CONFIG`；
- frozen SHA未刷新、JSON读取类型、绘图库调用或环境依赖错误；
- 任何在读取科学样本或产生指标前就停止的run。

它们只能证明fail-closed和复现治理有效，不能证明某个模型更好或更差。

## 7. 清理约束

1. 先为每个候选run记录目录大小、科学/系统状态、替代证据、保留文件和SHA-256。
2. 正式基线、消融、失败分析和论文图片对应的资产全部保留。
3. 可删除对象仅限可再生重复sensor shard、无科学输出的smoke/cache、被后继run逐项覆盖的冗余大包；先保留summary、配置、日志、图、checkpoint/轨迹摘要和seal。
4. 不使用通配符删除`results/`，不在最终论文图表映射完成前删除历史checkpoint或图。

## 8. 预定论文证据结构

| 论文位置 | 采用的现有工作 | 作用 |
|---|---|---|
| 主结果表 | 原始M-TARE、Cano-like出口+规则图、同输入非学习基元拟合、完整基元关系图、GT-TNG oracle | 分别回答原系统、只学出口、有几何但不学习、完整方法和可达上界的差异 |
| 感知消融表 | 单帧/五帧、8/16/32槽、去参数监督、去表面/射线重建、去关系、去时序、去不确定性 | 逐项证明几何基元、组合关系、时序和拒绝机制不是装饰模块 |
| 图构建消融表 | 最近距离关联、出口规则关联、学习关系关联、去edge几何属性 | 证明学习到的结构关系是否真正改变节点/边，而非只改善局部回归指标 |
| 方法动机图 | 非学习拟合低覆盖、事件槽/Composer失败、旧离线图false loop | 用代表性失败说明为何需要底层基元、关系学习与拒绝错误合并 |
| 补充材料 | 其余完整科学失败、资源/环境fail-closed记录 | 给出完整研究轨迹和复现边界，不稀释正文主线 |

最终只挑选最能支撑因果链的少量失败进入正文；“做过”不是采用理由，“能独立支撑论文结论”才是。
