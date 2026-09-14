# 当前代码导航：从地图到面片结构图

日期：2026-09-08。配套正文：[CURRENT_METHOD_WALKTHROUGH.md](CURRENT_METHOD_WALKTHROUGH.md)。

本表描述当前实际接口，不是承诺已经全部端到端运行。状态：**历史实物**＝旧正式资产/运行；**真实接口**＝至少有当前真实读取或导出；**合成软件**＝只确认合成/回归；**缺接线**＝组件存在但新链路未完成。函数链接指向完整源码，可按函数名搜索。

## 1. 数据与坐标

| 文件 / 主要对象 | 输入 → 输出、格式 | 状态与注意点 |
|---|---|---|
| [worldgen/tng.py](../src/mtare_topo/data/worldgen/tng.py) / generate_tng、TunnelNetworkGraph | SeedBundle、TNGParameters → 节点/边拓扑 JSON | 早期生成路径；不等于当前所有 P1a 数据直接调用它 |
| [worldgen/tunnel_mesh.py](../src/mtare_topo/data/worldgen/tunnel_mesh.py) / TunnelMeshParameters | TNG、几何参数 → 体素并集边界网格 | 历史造图实现；不要将默认机器人参数当成当前真实部署合同 |
| [generate_tng.py](../tools/v3/worldgen/generate_tng.py) / main | 配置 JSON → 新 topology-parent JSON | 只生成拓扑，不生成数据集或训练 |
| [primitive_construction_supervisor.py](../src/mtare_topo/teacher/primitive_construction_supervisor.py) / build_primitive_construction_graph、EndpointComposition | 原轴线/构造 → 双端及组合锚点记录 | 构造教师来源；真实身份不进入模型 |
| [primitive_relation_materialization.py](../src/mtare_topo/data/primitive_relation_materialization.py) / load_p1a_realized_construction | 封存 P1a JSON → PrimitiveConstructionGraph、SweptSuperellipsePrimitive 元组 | 当前实际几何消费入口；核对顺序、端点、引用 |
| [swept_superellipse_field.py](../src/mtare_topo/teacher/swept_superellipse_field.py) / SweptSuperellipsePrimitive | 轴线、截面半轴、指数 → 有限扫掠体几何表示 | 是旧通道体，不是新观测表面片 |
| [gse_sequence_inventory.py](../src/mtare_topo/data/gse_sequence_inventory.py) / enumerate_directed_traversals | world 路线结构 → 有向穿越清单 | 双向独立；不意味着连续全局路线 |
| [gse_sensor_export.py](../src/mtare_topo/data/gse_sensor_export.py) / world_unique_frame_poses | 路线弧长/记录 → 扫描位姿 | 已有离线数据路径；不是部署定位 |
| [primitive_relation_sensor_export.py](../src/mtare_topo/data/primitive_relation_sensor_export.py) / PrimitiveSensorFrame、render_primitive_sensor_frame | raycaster、pose → 距离、有效性、来源字段 | 历史实物；教师来源与学生距离分开 |
| [csg_mesh_provenance.py](../src/mtare_topo/teacher/csg_mesh_provenance.py) / mesh_swept_superellipse、CSGMeshProvenanceRaycaster.ray_exit_hits | 扫掠体 → 封闭三角网格；射线及初始内外状态 → 并集退出回波 | Open3D交点排序/状态更新，不是最近内部三角面；原参数需绑定 |
| [cano_sensor_smoke.py](../src/mtare_topo/data/cano_sensor_smoke.py) | 激光设置 → 16×720 射线与扫描工具 | 共用传感器几何；具体扫描路径以原 spec 为准 |
| [primitive_relation_sequences.py](../src/mtare_topo/data/primitive_relation_sequences.py) | 同段帧 → 五帧因果序列、相对运动 | yaw+三维平移；不能称完整 roll/pitch 补偿 |
| [primitive_relation_model.py](../src/mtare_topo/representation/primitive_relation_model.py) / register_causal_lidar_points、_memory | B×5×2×16×720 与运动 → 对齐点、900×128 token | 历史编码器复用，不复用旧连接头作新预测 |
| [gse_dual_path_encoder_adapter_v1.py](../src/mtare_topo/representation/gse_dual_path_encoder_adapter_v1.py) / expand_features | 紧凑公共特征 → 原点坐标与逐点上下文、布局索引 | 当前真实特征接口；索引是传感器槽位，不是 GT ID |
| [gse_surface_encoder_checkpoint_v1.py](../src/mtare_topo/representation/gse_surface_encoder_checkpoint_v1.py) / load_surface_encoder | 权重 bytes、期望 SHA → 冻结编码器 | 严格加载，不临时选更有利 checkpoint |
| [gse_surface_feature_cache_v1.py](../src/mtare_topo/data/gse_surface_feature_cache_v1.py) / bind_feature_cache | 缓存、清单、原输入 → 认证 compact | 真实接口；原 CPU/CUDA 精确比较问题已独立版本处理 |
| [gse_surface_training_join_v1.py](../src/mtare_topo/data/gse_surface_training_join_v1.py) / join_training_observation | 特征/原扫描/目标/manifest → SurfaceTrainingExample + loss context | 真实单例通过；目前关系射线默认未知，grid 仅进入损失路径 |

## 2. 教师：最新路径与保留旧路径

| 文件 / 函数 | 输入 → 输出 | 证据边界 |
|---|---|---|
| [gse_reference_anchor_targets_v1.py](../src/mtare_topo/teacher/gse_reference_anchor_targets_v1.py) / produce_junction_reference_targets | 来源绑定观测 → 部分构造路口位置 | 旧端面见证对侧面进入场景不足 |
| [gse_reference_anchor_targets_v2.py](../src/mtare_topo/teacher/gse_reference_anchor_targets_v2.py) | 旧接口 + 内部见证 → 路口参考 | V6 全人口所用路径，不等于 V8 |
| [gse_reference_anchor_targets_v3.py](../src/mtare_topo/teacher/gse_reference_anchor_targets_v3.py) / produce_junction_reference_targets | 原观测 + 明确 mesh 设置 + lateral evidence → 锚点 | 新增侧面进入/离开见证；目前合成 |
| [gse_reference_opening_targets_v1.py](../src/mtare_topo/teacher/gse_reference_opening_targets_v1.py) / _produce_opening_reference_targets | 10m 窗口参考 + crossing/surface → 开口位置/方向，宽高 None | 不是天然洞口尺寸真值，不产生固定图节点 |
| [gse_reference_opening_targets_v3.py](../src/mtare_topo/teacher/gse_reference_opening_targets_v3.py) / produce_opening_reference_targets | 来源绑定诊断 + 轴线归属 → 部分开口 | 当前联合路径调用；未知候选保留 |
| [gse_reference_terminal_targets_v1.py](../src/mtare_topo/teacher/gse_reference_terminal_targets_v1.py) / produce_terminal_reference_targets | 唯一来源端帽首次返回 → 构造尽头位置 | 看不到端面不标尽头；部分参考 |
| [gse_reference_continuations_v1.py](../src/mtare_topo/teacher/gse_reference_continuations_v1.py) | typed 实际双端 → 参考连续段/未解连接 | 不同分量不代表物理隔断 |
| [gse_construction_continuations_v1.py](../src/mtare_topo/teacher/gse_construction_continuations_v1.py) | 全构造、canonical SHA → 来源完整的连续段 | 不 snap 非零偏移；保留同隧道双 incidence |
| [gse_branch_transition_witness_v1.py](../src/mtare_topo/teacher/gse_branch_transition_witness_v1.py) | 同一原射线方向/首次返回 → 同节点进出支持 | 方向兼容不等于负标签 |
| [gse_observed_operand_entries_v1.py](../src/mtare_topo/teacher/gse_observed_operand_entries_v1.py) | 完整操作体网格 + 原射线 → 侧面等入口 | 5601 射线合成核查成立；真实配置待绑定 |
| [gse_lateral_branch_evidence_v1.py](../src/mtare_topo/teacher/gse_lateral_branch_evidence_v1.py) / bound_lateral_evidence | 原扫描/构造/入口 → surface entry 与 departure | 同一来源对唯一节点、方向与遮挡保护；不是场景名规则 |
| [gse_joint_reference_targets_v6.py](../src/mtare_topo/teacher/gse_joint_reference_targets_v6.py) / produce_joint_reference_targets | 旧锚点/开口/尽头 → 部分联合记录 | 2676 真实观察已封存；False=0 |
| [gse_joint_reference_targets_v7.py](../src/mtare_topo/teacher/gse_joint_reference_targets_v7.py) / _produce_terminal_exclusion | 参考连续段 + 正归属 + 观测否定见证 → 可证 None→False | 原 T 失败仍保留；不准冒称全人口成功 |
| [gse_joint_reference_targets_v8.py](../src/mtare_topo/teacher/gse_joint_reference_targets_v8.py) / produce_joint_reference_targets | anchor V3 + entry 对应 + departure 排除 → 联合记录 | 四实际合成场景通过，未真实全批导出 |
| [gse_surface_target_adapter_v1.py](../src/mtare_topo/teacher/gse_surface_target_adapter_v1.py) / observed_targets | 部分 JSON → SurfaceLossTargets 张量与掩码 | None 不进 BCE；根可达有效性当前不提供 |
| [gse_reference_exclusion_binding_v1.py](../src/mtare_topo/teacher/gse_reference_exclusion_binding_v1.py) / bound_reference_exclusion | 原观测 grid + 参考绑定 + 查询 → 有证据排除 | 只为损失/独立诊断；不能给前向教师候选 |

## 3. 面片、模型与训练

| 文件 / 对象 | 输入 → 输出/形状 | 状态与限制 |
|---|---|---|
| [gse_surface_patches_v1.py](../src/mtare_topo/representation/gse_surface_patches_v1.py) / extract_surface_patches | N×3、valid、frame index → M 个 PCA 面片 | 固定 .5m/10m，最多4096；退化不删除 |
| 同文件 / build_patch_neighbors | 面片 → M×8 近邻、几何关系 | 不表示连通性；默认射线证据未知 |
| [gse_surface_ray_evidence_v1.py](../src/mtare_topo/representation/gse_surface_ray_evidence_v1.py) / build_surface_ray_grid、query_patch_gaps | 原射线 → .25m 观测网格 → gap 证据 | 观测代理，不是车身通行证明 |
| [gse_surface_relation_model_v1.py](../src/mtare_topo/representation/gse_surface_relation_model_v1.py) / collate_surface_patches | 面片列表 → B×M×18、B×M×8×9 | 无真实 ID，无教师筛选 |
| 同文件 / _MessageLayer | 128维自身/邻居 +9维关系 →128维 | B 自身更新，C 邻接消息；三层 |
| 同文件 / SurfaceRelationModelV1.forward_compact | compact + patches → SurfaceRelationPrediction | A/B/C 共同查询头；真实正式训练0 |
| 同文件 / SurfaceRelationPrediction | 32锚点、64开口、64×32归属等 | 无事件分类头；不确定性等未监督 |
| [gse_surface_losses_v1.py](../src/mtare_topo/representation/gse_surface_losses_v1.py) / _assign、surface_relation_losses | prediction + target → 八项独立归一化损失 | 几何匹配；未知不补负；不训可靠性头 |
| [gse_surface_partial_losses_v1.py](../src/mtare_topo/representation/gse_surface_partial_losses_v1.py) | 部分目标 + 来源 grid → 有界负查询约束 | 不自动完整背景资格 |
| [gse_surface_partial_losses_v2.py](../src/mtare_topo/representation/gse_surface_partial_losses_v2.py) | 上述 + 确认参考 → 重复预测约束 | 不代替 membership 负例 |
| [gse_surface_training_context_v1.py](../src/mtare_topo/representation/gse_surface_training_context_v1.py) / validate_context、source_bound_loss | 原例与独立 context → 校验和损失 | 教师只进此路，不进入 model.forward |
| [gse_surface_training_v1.py](../src/mtare_topo/representation/gse_surface_training_v1.py) / train_paired_surface_models | 同缓存人口/预算 → A/B/C 初末状态/损失记录 | 合成测试；未完成正式真实 A/B/C 实验 |
| 同文件 / SurfaceTrainingBudget | 更新数/seed/lr/累积 → 固定预算 | 2000 是诊断预算，不保证收敛 |

## 4. 从预测到图与评分

| 文件 / 对象 | 输入 → 输出 | 现有证据/问题 |
|---|---|---|
| [gse_surface_observation_v1.py](../src/mtare_topo/topology/gse_surface_observation_v1.py) / SurfaceObservationV1 | 源帧、锚点、开口、关系 → typed observation | 分开存在与未知，不携带 GT 身份 |
| [gse_surface_prediction_binding_v1.py](../src/mtare_topo/topology/gse_surface_prediction_binding_v1.py) / SurfacePredictionBindingV1 | 帧 packet、compact、模型 → 来源绑定观察 | 推理填实际 ray gap；与训练入口不一致 |
| 同文件 / SurfaceForwardReadoutV1 | logits/有效性 → 读出概率及掩码 | 有未训支持/有效性头，不能当已校准决策 |
| [gse_surface_graph_v1.py](../src/mtare_topo/topology/gse_surface_graph_v1.py) / KnownSurfacePoseV1、SurfaceSegmentedGraphV1 | observation + 已知 pose → 本地节点/轨迹边/decision trace | 短时跟踪，非跨访问融合；未校准配置 |
| 同文件 / SurfaceExecutionStartV1、depart | 明确执行开始凭据 + 当前开口 → 待完成穿越 | DTO 本身不认证真实控制器；须有实际轨迹 |
| [gse_registration.py](../src/mtare_topo/topology/gse_registration.py) | 局部几何/配准配置 → 验证证据 | 独立复用组件存在；新 surface graph 未调用 |
| [gse_graph_frames.py](../src/mtare_topo/topology/gse_graph_frames.py) | 三维坐标变换 → 合法参考系 | 该模块有 SE3 校验，不表示历史输入已有 full-SE3 运动 |
| [gse_surface_detection_metrics_v1.py](../src/mtare_topo/evaluation/gse_surface_detection_metrics_v1.py) / detection_counts | 置信筛选后的点集/目标 → 检出计数 | 与 representation scorer 的先匹配顺序不同 |
| [gse_surface_prediction_scoring_v1.py](../src/mtare_topo/evaluation/gse_surface_prediction_scoring_v1.py) | 结构观察 + 部分目标 → 检测诊断 | 完整/部分区域须分开 |
| [gse_surface_scoring_v1.py](../src/mtare_topo/representation/gse_surface_scoring_v1.py) / score_surface_predictions | 全查询 + targets + 显式阈值 → 分项评分 | 不用训练匹配；几何平局拒绝，未训置信仅诊断 |
| [gse_objective_spatial_graph_score.py](../src/mtare_topo/evaluation/gse_objective_spatial_graph_score.py) | 历史图与真实结构 → 空间图评分 | 旧 event 分组与新无类别锚点尚需统一，不能直接排名 |

## 5. 代表性反例与验证入口

| 文件 | 检查什么 | 如何解释 |
|---|---|---|
| [check_gse_joint_producer_backend.py](../tests/v3/unit/check_gse_joint_producer_backend.py) | 实际合成 T 几何/扫描 | 几何场景来源，不是人工伪造的 tensor 答案 |
| [check_gse_terminal_T_lateral_entry.py](../tests/v3/unit/check_gse_terminal_T_lateral_entry.py) | 侧面入口与解析解、遮挡、多来源 | 5601 射线原因证据 |
| [check_gse_terminal_exclusion_v7.py](../tests/v3/unit/check_gse_terminal_exclusion_v7.py) | 原 V7 同 T 失败 | 失败保留，不能把预期失败删掉制造绿灯 |
| [check_gse_lateral_anchor_v3.py](../tests/v3/unit/check_gse_lateral_anchor_v3.py) | 侧面见证是否恢复锚点 | 不等于归属已全接通 |
| [check_gse_joint_lateral_v8.py](../tests/v3/unit/check_gse_joint_lateral_v8.py) | 四场景联合归属 | 合成验证，不是实际人口负例数 |
| [test_gse_surface_patches_v1.py](../tests/v3/unit/test_gse_surface_patches_v1.py) | 面片、退化、确定性等 | 本次回归集 |
| [test_gse_surface_relation_model_v1.py](../tests/v3/unit/test_gse_surface_relation_model_v1.py) | A/B/C 输入输出与软件保护 | 本次回归集 |
| [test_gse_surface_losses_v1.py](../tests/v3/unit/test_gse_surface_losses_v1.py) | 匹配、未知、多归属、恒正退化 | 本次回归集，部分合成梯度更新不计正式训练 |
| [test_gse_surface_graph_v1.py](../tests/v3/unit/test_gse_surface_graph_v1.py) | 分段/轨迹/中间节点保护 | 本次回归集，不证明真实建图收益 |
| [test_gse_surface_prediction_scoring_v1.py](../tests/v3/unit/test_gse_surface_prediction_scoring_v1.py) | 预测评分 | 本次回归集 |
| [test_gse_surface_scoring_v1.py](../tests/v3/unit/test_gse_surface_scoring_v1.py) | 独立张量评分 | 本次回归集 |

本次执行并取得终态：以上最后六个 pytest 文件，106 passed in 2.36s。实际命令见正文第17节。合成 V8 的结果来自已记录运行；本次未为报告重新导出真实标签。

## 6. 读代码的最短顺序

先读 join_training_observation，确定学生/教师分流；再读 extract_surface_patches 与 _MessageLayer，理解 B/C；然后看 SurfaceRelationPrediction 与 observed_targets，区分“输出字段存在”和“真有监督”；之后读 surface_relation_losses 和 SurfaceSegmentedGraphV1；最后看两个评分入口，理解为什么目前还不能给科学比较表。

不要先读数百个 run spec，否则容易把正确留痕误当成主方法结果。历史完整目录元数据在正文附录，重点结果在第12节。

## 7. 当前源码版本快照

工作树不是只由 Git HEAD 描述的干净快照。以下 SHA-256 是本次文档核查时的核心源码内容，不覆盖全仓库，也不替代正式训练冻结。

| 核心源码 | SHA-256 |
|---|---|
| [representation/gse_surface_patches_v1.py](../src/mtare_topo/representation/gse_surface_patches_v1.py) | 835901304377e62623850b17879ba73ea8b93505e9418c1b923694ea60c4b50a |
| [representation/gse_surface_relation_model_v1.py](../src/mtare_topo/representation/gse_surface_relation_model_v1.py) | 901e4f08784078fa661e324e27c9771529acc3ec46c5cc9188306ba1e826ae7d |
| [representation/gse_surface_losses_v1.py](../src/mtare_topo/representation/gse_surface_losses_v1.py) | 77266db1f50f6edd3d7f35adcd38ee105375521d3a01abb2ebb9334dd9c9a6b9 |
| [representation/gse_surface_training_v1.py](../src/mtare_topo/representation/gse_surface_training_v1.py) | 4dcf5dd025885a35ebcb94c68eddb1ad965271f7b160011003715725224bf745 |
| [representation/gse_surface_scoring_v1.py](../src/mtare_topo/representation/gse_surface_scoring_v1.py) | 1db17185f5ddb0bb5e81b8bf7a1ae52c4afd4d779bf313dff14f650d48a82c11 |
| [representation/gse_dual_path_encoder_adapter_v1.py](../src/mtare_topo/representation/gse_dual_path_encoder_adapter_v1.py) | 5527dd18dbb9bb66bbae1b1f5f0295839ef9b775ee2ea1b9a1c5e378aa829292 |
| [data/gse_surface_training_join_v1.py](../src/mtare_topo/data/gse_surface_training_join_v1.py) | f916400ad89f368177e3ae847760f9b97b8c814883820717b1d580b2e394c858 |
| [teacher/gse_joint_reference_targets_v8.py](../src/mtare_topo/teacher/gse_joint_reference_targets_v8.py) | 50110f775cea0d3db10b8a59f3e5a7967e1b14b2a0e9d469be35b574b500dff8 |
| [teacher/gse_lateral_branch_evidence_v1.py](../src/mtare_topo/teacher/gse_lateral_branch_evidence_v1.py) | b47abce4ad05e87abe3a8ead307601a9c713749cf56347f4eadd46890d73eeac |
| [teacher/gse_observed_operand_entries_v1.py](../src/mtare_topo/teacher/gse_observed_operand_entries_v1.py) | a1e768bffee8d5b0e06dff9dca1b4c3ec0fa78799c28151a26b4196a7072f11a |
| [teacher/gse_surface_target_adapter_v1.py](../src/mtare_topo/teacher/gse_surface_target_adapter_v1.py) | dbb179738f109d4283408bc3afa18ce88db3b61f5f92725ab3bceacd60dbfc82 |
| [topology/gse_surface_prediction_binding_v1.py](../src/mtare_topo/topology/gse_surface_prediction_binding_v1.py) | a4d3d2dcd87a8a5d991a08a1ff37e90a9b5d889b95284d3c6fa71c7634d9ec85 |
| [topology/gse_surface_graph_v1.py](../src/mtare_topo/topology/gse_surface_graph_v1.py) | 277c5b6a3b204e1d6ca5e02efa2c9c53c4e422e569a23ccc3217aabb0b228fb8 |
| [evaluation/gse_surface_detection_metrics_v1.py](../src/mtare_topo/evaluation/gse_surface_detection_metrics_v1.py) | 86feb355194978f51058f24184a5ba14f7a88d5db303f3526e1828462cdeb157 |
