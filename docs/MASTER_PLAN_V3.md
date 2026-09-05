# 地下结构语义拓扑探索项目：MASTER PLAN V3

> 2026-08-29权威覆盖：Gate 3主方法已改为程序构造监督的显式扫掠几何基元、端口关系和跨帧对应学习，详见`docs/PRIMITIVE_RELATION_STRUCTURAL_GRAPH_PLAN_V1.md`。本文中旧BEV方向语义、事件类别和规则节点路线只保留治理背景；发生冲突时执行新计划。

> 执行优先级说明（2026-08-10）：用户已发布 `docs/PLAN.md` 作为当前最终执行方针，日常阶段推进以 `PLAN.md` 与 `PROGRESS.md` 为准。本文件保留治理、数据隔离和历史 Gate 细节；若方法顺序、当前范围或 baseline 与 `PLAN.md` 冲突，以 `PLAN.md` 为准并记录决策，不得自行混合。

从现在开始，项目不再以“一个实验失败后继续叠加网络、loss 或阈值”的方式推进。研究目标、阶段边界、test 隔离、M-TARE 公平比较条件和可复现证据不可漂移。未通过当前 Gate，不得自动进入下一 Gate。

## 0. 最终研究目标

在相同 M-TARE 仿真环境、机器人、传感器和局部规划器条件下，使用学习得到的地下局部结构特征和结构语义构建增量拓扑图，并利用拓扑图完成全局 frontier 选择与多机器人任务分配，替代 M-TARE 原有全局规划层。

```text
LiDAR + state estimation
  -> 局部因果结构观测
  -> 学习式结构表示
  -> 可解释结构语义
  -> 增量拓扑图
  -> 全局 frontier/exit selection
  -> 多机器人任务分配
  -> 下一拓扑目标与安全局部 waypoint
  -> M-TARE local planner
  -> 执行反馈更新拓扑图
```

## 1. 系统边界与公平比较

保留 M-TARE 的仿真器、LiDAR/状态估计、terrain analysis、碰撞检测、local planner、短程避障、waypoint 执行和底层控制。替换全局环境表示、全局 frontier 表示、全局目标选择和多机器人全局任务分配。

最终比较为 `M-TARE original global planning` 对 `our semantic-topological global planning`。两边必须使用相同 world、起点、机器人数量、随机种子、传感器、local planner、碰撞参数和运行时间。

## 2. 三层核心假设

- H1：残缺在线 LiDAR 能学习出比纯表面几何更稳定的方向可通性、出口、连接关系和局部结构角色。
- H2：结构语义图能在稳定段少建冗余节点、结构变化处保留节点、不丢未探索出口并保持连通，优于固定距离、纯几何变化和传统 keyframe。
- H3：在相同 local planner 下，semantic topology、graph frontier planning 和 multi-robot allocation 能减少重复探索、无效移动和任务冲突，并改善 coverage-time、最终覆盖率、完成率、路径长度或重复探索率中的至少一项。

三层假设必须逐层验证，不能混在一次实验中。

## 3. 项目治理和证据

正式文档位于 `docs/`，V3 结果位于 `results/gate0_baseline/` 到 `results/gate8_final/`，机器可读状态为 `results/project_status.json`。

每个实验至少保存 config、完整命令、随机种子、输入 manifest、运行环境、原始 log、metrics、summary 和必要 preview；模型保存 checkpoint；图实验保存 trajectory、nodes、edges、exit states 和 decision trace；闭环保存 coverage-time、path、planner decisions、runtime 和失败原因。

图只为回答研究问题服务，不得为了展示而随意绘图。每个 preview 必须注明数据 split、样本标识、方法、单位和它支持或反驳的假设。

### 问题停止与反馈协议

任何阶段一旦发现数据缺陷、样本独立性不足、split 泄漏、teacher 与部署规则不一致、baseline 无效、指标不稳定或实现阻塞，立即停止受影响工作，不得先偷偷修正再报告。

反馈必须明确给出：具体证据、影响的 Gate/结论、问题归因、可选方案及代价、推荐方案和需要用户决定的事项。未经用户确认，不得替换数据、改变 split/teacher/metric、降低阈值、增加复杂模型或转做另一实验。

### 数据实验前置审批

任何数据导出、自监督或训练开始前，必须先提交 data card，至少列出：

```text
原始数据源和许可用途
world 数量及名称
每个 world 的独立轨迹数
每条轨迹时长与移动距离
原始帧数、空间下采样规则和有效样本数
结构类型覆盖及不平衡
train/validation/test 的完整划分
teacher 来源和有效 mask
泄漏与历史污染审计
预计磁盘、时间和计算成本
```

样本数量不能只报帧数，必须同时报告独立 world、独立轨迹、空间覆盖和有效结构事件数量。相邻帧重复不能冒充数据规模。data card 未经用户确认，不得训练。

最终 benchmark/test world 必须 world-disjoint，并禁止进入监督训练、SSL、teacher/阈值标定、归一化统计、数据增强调节和 checkpoint 选择。开发 world 可以采集训练数据和运行调试；最终性能声明只使用冻结后从未参与 V3 开发的 test world。

### 混合监督主路线（2026-08-10 修订）

V3 的第一学习路线改为有监督 teacher-student，而不是先以自监督为主。原始事实来源是带时间戳、位姿和真实 ray origin 的地下 LiDAR 点云；第一版学生输入是从点云构建的 terrain-relative causal 2.5D BEV。完整地图、collision 和 M-TARE local planner 生成 `R(theta)`、`D(theta)`、exit component 与 `G_local` 等客观核心监督。

冻结的多模态 AI 只对标准化点云/BEV/完整地图/planner overlay 提出高层结构属性、置信度和异常建议，必须经过硬规则与人工金标审计。AI 不直接猜连续距离、隐藏 free-space，也不得静默覆盖 planner/map 标签。AI 标注未通过质量门时不进入 loss，核心 objective supervision 仍可独立推进。

Gate 2 主方法为轻量 BEV CNN/ResNet + polar representation 的监督式结构表示学习；自监督降为有合格未标注开发数据时的可选 warm start/正则，并必须与纯监督模型消融。直接点云 encoder 只有 BEV 被多高度、坡地、悬空或洞穴结构证据否定后才启用。详细 contract、标注流程、模型和停止条件见 `docs/SUPERVISED_STRUCTURE_LEARNING_V1.md`。

### 固定实施主线：论文生成器 + CPU Raycast + 结构语义拓扑（2026-08-10 修订）

优先直接使用能够固定来源、环境、seed 和输出的论文公开代码，通过最薄 adapter 补足 provenance 与导出，不为提前制造“创新”而重写可复现 baseline。当前 Cano 固定 commit 可复用 TNG、表面点云和 native mesh 生成；其 checkout 未包含出口 CNN 训练实现，因此后续感知 baseline 必须标为 `Cano-like adapted reproduction`。生成器必须输出 topology、centerline 和经用途审计的 perception/collision geometry；两类 mesh 可不同，但必须共享 topology/centerline provenance。

生成随机性按 `topology_seed / geometry_seed / clutter_seed / trajectory_seed / sensor_seed` 分离。split 的原子是 TNG topology parent：同一 TNG 的全部 geometry 和 observation 永不跨 split。每个 TNG 至少生成两个经机器人 footprint/planner 认证、不改变 operational connectivity 的几何变体，用于“同拓扑、不同几何”配对评价；改变可通连接的变体不得作为正样本。完整 TNG 仅作 oracle/teacher/evaluation，学生只能看到 LiDAR 形成的当前与因果观测，不可见连接必须 mask。详细合同见 `docs/TNG_COUNTERFACTUAL_DATA_CONTRACT_V1.md`。

Phase 1--3 的 canonical LiDAR 主后端为 Open3D CPU idealized raycasting：明确 pose/ray origin、16×720 first-return range 和 valid mask。第一学习 baseline `B1` 是 Cano-like range-image 360-degree exit CNN；主模型 `M1` 才结合本项目方向，使用 current+causal terrain-relative BEV，训练 planner-consistent `R(theta)`、`D(theta)` 与 uncertainty。Gazebo 固定 pose parity 是正式训练前必要门禁；Isaac 降级为可选后验域实验。出口 component 和结构角色由冻结解释器产生；二次创新集中在 stable exit-stub graph、M-TARE global replacement 和 multi-robot allocation，禁止端到端绕过 H1--H3。

正式容量按 80 train、10 validation、10 development-test world，上限 500,000 observation 设计，通过 20/40/80 train-world 嵌套学习曲线逐级扩展。M-TARE 官方五图、项目现有地下 benchmark 和 sealed strict test 均不进入训练。当前传感器接口见 `docs/CPU_RAYCAST_LIDAR_BACKEND_V1.md`；旧 Isaac 蓝图只保留历史/可选域实验参考；发表门槛见 `docs/PUBLICATION_EVIDENCE_PLAN_V1.md`。

## Gate 0：冻结 M-TARE baseline 与接口

本 Gate 不训练模型。确认原全局 planner 输入输出、local planner 输入、waypoint、coverage、位姿、terrain/collision 和多机器人通信接口。至少选择 3 个地下训练/开发 world 和 2 个完全未见地下 test world，并运行若干有效重复 baseline。无运动、日志缺失和 planner 异常的 run 标记 INVALID。

证据：`interface_contract.json`、`world_manifest.json`、`baseline_runs/`、`baseline_metrics.json`、`coverage_curves/`。

通过条件：明确新模块从哪里接管、向 local planner 输出什么，并获得冻结条件下的有效 baseline。

## Gate 1：数据与 Teacher

本 Gate 只回答“模型看什么、老师教什么”，不训练正式模型。

原始数据 contract 以地下局部点云、scan/point 时间戳、状态估计、相对变换和真实 ray origin 为基础。样本单位是空间独立 `place_cluster`，不是单个相邻帧或 `.npz` 文件。绝对 pose、world/trajectory ID 和未来信息只作 provenance，禁止进入学生模型。

学生主 Route A 为 terrain-relative 2.5D BEV，包含可靠在线可得的 observed surface、多高度占据、terrain-relative height、height span、可靠时的 slope/roughness、observation age/count 和历史有效 mask。只有存在真实 ray origin 时才能加入 free/unknown evidence，不得从 LAMP 累积 KeyedScan 伪造 free-space。

备选 Route B 是直接局部点云，只有 BEV 被证据证明在多高度、洞穴或坡地上成为瓶颈时才启用。

核心 teacher 只保留方向可通性 `R(theta)`、方向可达距离 `D(theta)`、连续出口 component `E` 和局部连接 `G_local`。openness、bottleneck、turn 仅作辅助，node_score 不作核心监督。

Teacher 主 Route A 是 planner-consistent teacher；备选 Route B 是 terrain-derived teacher，且必须与 Route A 抽样对比。

AI-assisted annotation 只标多标签结构属性，如 topological degree、straight/turn/branch/chamber、width/bottleneck、slope/multi-height 和 ambiguity。每次标注 pilot 必须在 Data Card 中冻结 annotation bundle、AI labeler/version、prompt、JSON schema、调用样本数、人工金标数、abstain/冲突规则和成本。strict test 不得进入 Gate 1 AI 标注、prompt 调试或人工开发审阅。

证据：point-cloud source contract、causal student-input contract、teacher contract、AI annotation contract、dataset manifest、split definition、teacher-planner consistency、AI-human/rule consistency、replay parity 和 data previews。

Gate 1 通过前必须单独给出数据充分性结论。至少覆盖 Gate 0 冻结的 3 个地下开发 world，并具有多条独立轨迹；是否达到正式训练规模由 world/轨迹/空间/结构事件统计共同决定，不允许只因 `.npz` 数量达到某个数字就宣称充分。规模不足时结论必须为 `GATE_FAIL` 或 `GATE_MIXED`，不得进入 Gate 2 正式训练。

## Gate 2：学习稳定结构表示

本 Gate 只回答残缺观测能否学习稳定结构，不建拓扑图。

主 Route A：轻量 BEV CNN/ResNet 加 polar/radial-direction representation，通过 Gate 1 冻结的客观结构标签进行监督式表示学习。第一版直接学习 masked `R(theta)`、`D(theta)`、必要的 local connectivity 和 uncertainty；exit component 优先由冻结 circular grouping 从方向输出确定。AI/人工结构属性只作为低权重辅助监督，并做 objective-only 对 objective+AI ablation。

明确 baseline 是 online partial-BEV geometry/planner rule，不学习。备选 Point-based encoder 仅在 BEV 信息保真审计明确失败后尝试；Sparse 3D 仅在 point route 仍失败且实时预算允许时尝试。不得并行开发三条路线或为 point route 重建不同 split。

自监督不再是第一版主方法。只有存在大量合格但未标注的地下开发数据、纯监督 baseline 已建立且用户批准独立 run spec 时，才可使用 masked geometry reconstruction、same-place different-observation consistency 和 rotation equivariance/invariance 作为 warm start/辅助正则。严格 test 不得进入 SSL；无跨 world 稳定性或 downstream objective 增益时删除自监督分支。

必须证明同结构不同 coverage 接近、相似 coverage 不同结构分开、旋转可控、遮挡/稀疏稳定且跨 world 有效。

## Gate 3：显式结构语义

核心输出为 direction traversability、reachable distance、exit components、local connectivity、structural-role representation 和 uncertainty。`z_role` 描述结构作用，不等于具体地点身份 `z_place`。

方向/距离/连接 head 只使用 Gate 1 冻结的客观 teacher；结构角色可使用通过 AI/人工审计的辅助属性，但必须报告不使用 AI 属性时的 objective-only baseline。语义分类不能直接决定 node，`node_score` 仍不学习。

时间主 Route A 是把因果历史变换到当前坐标并累计证据，服务静态出口和可达性。只有持续结构事件确实需要时，才启用备选 Route B：GRU、TCN 或小型 causal Transformer。

directional semantics 和 `z_role` 分开输出；`z_place` 可留到未来专门的 place-recognition 阶段。

## Gate 4：离线拓扑图

图对象固定为 structural node、anchor keyframe、verified edge 和一等对象 exit stub。结构节点由持续结构事件产生；稳定长段由 anchor 保持连通和边长受控。

Exit stub 状态为 unobserved、observed、attempted、traversed、temporarily_failed 和 blocked。机器人实际通过后，stub 才转换为 edge 和新 node。

节点主 Route A 是持续出口/连接变化、距离 anchor 和迟滞组成的规则 baseline。只有它不稳定时才尝试 change-point detection。暂不学习 node_score。

评价必须包括 exit preservation、graph connectivity、node redundancy、path-distance distortion、verified-edge correctness 和 remote-frontier reachability，不能只看 node F1。

## Gate 5：Shadow 在线运行

M-TARE 原 planner 控制机器人，新系统只旁路输出 semantics、nodes、edges、exit stubs 和 global target proposal。检查实时性、图爆炸、节点抖动、出口翻转和目标稳定性。Shadow 稳定后才能控制机器人。

## Gate 6：单机器人拓扑全局规划

开始替换 M-TARE global planner。主 Route A 使用简单可解释评分：exploration potential 减 graph travel cost 和 repeated/retry penalty。只有尺度和边界使规则不稳定时才尝试备选 Sugeno fuzzy decision，并与规则方法对照。

```text
current node -> all unexplored exit stubs -> utility -> target exit
-> Dijkstra/A* -> next node/exit -> safe waypoint -> M-TARE local planner
```

## Gate 7：多机器人拓扑协同

单机通过后才开始。第一版假设共同世界坐标已知；断联独立建图和选任务，重连交换增量节点/边/exit 状态并做几何与 ID 消解。

主 Route A 为连通时中央 Hungarian；备选 Route B 为受限通信下 distributed auction/lease；Route C 可用 Sugeno 只计算每台机器人对每个 exit 的 utility，再交给 Hungarian 或 auction。模糊控制不承担地图融合、节点匹配和分配算法本身。

第一版 utility 只含 information potential、graph travel cost 和 redundancy，之后再通过 ablation 加通信、冲突、风险和定位不确定性。

## Gate 8：最终 M-TARE 对比

固定比较：A 原始 M-TARE；B semantic topology + rule-based；C semantic topology + Sugeno；D multi-robot semantic topology + 选定分配方法。

指标包括 coverage-time、final coverage、完成率、团队和单机器人路径长度、重复探索、remote-frontier success、stuck time、碰撞/拒绝、通信字节和运行/GPU 延迟。无有效 baseline 的 run 不进入统计。

## 4. Place Recognition / Loop Closure 边界

早期 Gate 不混入 place recognition。未来需要时采用：`z_place` 检索候选，经 temporal/graph filtering、局部点云配准和 geometry verification 后才能添加 loop edge。`z_role` 只可辅助过滤，禁止因 role 相似直接合并节点。

## 5. Gate 决策规则

每个 Gate 结束必须输出 `GATE_PASS`、`GATE_MIXED` 或 `GATE_FAIL`，回答研究问题、证据、最强 baseline、改进、失败案例、问题归因和下一步理由。FAIL 只能修当前 Gate；MIXED 必须明确可继续部分和保留风险；PASS 后也只能在用户确认后更新 current_gate。

每个 Gate 最多维护一个主方法、一个明确 baseline 和一个必要备选。禁止 test 调参、失败后自动跳 Gate、无证据并行复杂架构和无限增加 loss/head。

## 6. 现有项目重新定位

可复用资产：M-TARE 环境、rosbag、StandardFrame、SurfaceEvidenceBuilder、adapter、terrain/collision 审计、teacher 经验、训练代码、graph 原型、online node、shadow/closed-loop 框架、指标与可视化。

历史 baseline：ForcedGlobalBottleneckNet、旧 64D embedding、旧 canonical role、旧 semantic 模型、旧节点阈值、旧 offline topology 和已有 closed-loop。已有在线实现不等于 Gate 4/5/6 已通过。

V3 从 Gate 0 的 baseline、benchmark 和 interface freeze 重新开始，但先审计已有资产；满足要求的证据可以复用。当前初始化只建立文档和 Gate 对齐审计，不训练、不运行新的 closed-loop，也不自动开始 Gate 0 实验。
