# 地下结构语义拓扑探索：项目主流程 V2

## 1. 最终目标

本项目的目标是让机器人在地下环境中，从在线 LiDAR 和位姿估计中学习稳定的局部结构表示，形成可解释的结构语义，持续建立可用于导航的拓扑图，并在图上进行全局探索规划，以替代 M-TARE 原有的全局规划层。

M-TARE 的仿真器、状态估计、地形分析、碰撞检查和局部规划器继续保留。新系统负责全局结构理解、拓扑图维护、全局 frontier 选择和下一局部 waypoint 的生成。

完整链路固定为：

```text
多源地下数据
  -> 统一因果局部结构地图
  -> 自监督结构表示预训练
  -> 显式结构语义监督训练
  -> 结构事件与关键帧联合建图
  -> 几何验证的节点匹配与回环
  -> 拓扑图上的全局 frontier 规划
  -> M-TARE 局部规划器执行
  -> 执行反馈更新拓扑图
```

## 2. 数据体系

### 2.1 数据来源

训练和验证数据必须覆盖多种地下结构，而不能只依赖原 M-TARE 的五个仿真 world。

数据分为三类：

1. M-TARE 地下仿真：`tunnel`、`garage`、`unseen_mine`，以及后续新增的隧道、矿井和洞穴 world。
2. 外部地下仿真：`external_cave`、SubTGraph operational 系列和其他可获得的地下 world。
3. 真实地下数据：LAMP tunnel/urban/ku/perlin，以及在点云和可靠位姿问题解决后可使用的 SubT-MRS 数据。

没有完整 teacher 的真实数据仍可用于自监督预训练，不应因缺标签而被排除。

### 2.2 采集单位

一个样本不是一张完整地图，而是一个带因果历史的局部关键帧：

```text
当前扫描 + 过去若干扫描 + 相对位姿 + 时间间隔
```

每个 world 应采集多条独立探索轨迹，改变起点、朝向、随机种子和探索策略。建议每个 world 至少 20--50 条轨迹，并按 0.5--1.0 m 空间间隔导出关键帧。第一阶段目标为 5 万--20 万个有效关键帧；样本的 world、轨迹和结构多样性比重复导出相邻帧更重要。

### 2.3 数据划分

同时执行两个互补协议。

场景内部署协议：同一目标 benchmark world 的完整轨迹按约 70%/15%/15% 分为 train/validation/test，用来回答模型能否在目标场景的新轨迹上工作。

跨地下世界协议：多个地下 world 训练，一个完整未见地下 world 验证，另一个完整未见地下 world 最终测试，用来回答跨环境泛化能力。

禁止随机打散相邻帧。一个轨迹或空间区域只能属于一个 split。最终 test world、test 轨迹和 test 运行反馈不能参与阈值选择、训练或 checkpoint 选择。

## 3. 统一局部结构输入

在线和离线必须调用同一个 builder，并通过回放一致性测试。输入统一表达在当前机器人坐标系中，历史帧按真实相对位姿对齐到当前帧。

第一版采用可实时运行的 2.5D 多高度层 BEV，而不是单一二维 surface mask。建议包含：

- observed、unknown、ray-free-space 和 occupied evidence；
- 多个高度层的占据证据；
- min/mean/max height、height span；
- slope、roughness 和可通行证据；
- 观测次数或时间衰减；
- 每个历史帧的有效 mask、时间差和相对位姿。

所有 channel 必须有明确单位、坐标方向、缺失值规则和归一化范围。world ID、轨迹 ID、绝对位置和未来观测禁止作为模型输入。

## 4. 模型方法

### 4.1 主干结构

采用“2.5D BEV 空间编码 + 极坐标方向编码 + 因果时序融合”的模型：

```text
每帧 2.5D BEV
  -> 共享权重的 2D CNN/轻量残差编码器
  -> 以机器人为中心投影为径向环 x 角方向 token
  -> circular convolution 保留方向周期性
  -> causal temporal Transformer 或 GRU 融合历史
  -> 当前局部结构表示
```

模型保留两类输出，不能再把固定 canonical role 当作唯一结构表示。

第一类是可解释语义头：

- 32 或 64 个角方向的 traversability；
- 每个方向的 reachable distance；
- 连续出口/分支的概率、方向和宽度；
- openness、bottleneck、main-path continuity；
- structural-change/event probability；
- 预测置信度或不确定性。

第二类是 128D 学习型结构 embedding，用于节点匹配、相似结构检索和回环候选生成。该 embedding 必须通过 metric/contrastive supervision 学习“相同拓扑、不同覆盖应接近；相似覆盖、不同拓扑应远离”。

固定 canonicalizer 可以保留为可解释的诊断描述符，但不再作为唯一的节点身份或唯一的拓扑 embedding。

### 4.2 训练分两阶段

阶段 A：自监督预训练。使用全部可用地下扫描，包括无 teacher 的 LAMP/SubT 数据，联合使用：

- masked local-map reconstruction；
- 相邻时间帧和同地点不同视角的对比学习；
- 旋转等变/不变约束；
- 历史顺序和相对位姿一致性；
- 同一位置不同采样密度、遮挡和噪声下的一致性。

阶段 B：有监督语义微调。使用仿真完整地图、局部规划器可达性和真实执行轨迹产生 teacher，训练显式语义头、event head 和结构 embedding。

总 loss 由以下部分组成：

```text
L = L_traversability
  + L_reachable_distance
  + L_exit_component
  + L_static_structure
  + L_event
  + L_metric_embedding
  + L_uncertainty
  + L_temporal_consistency
```

类别不平衡使用 focal/BCE、有效 mask 和按结构类型平衡采样处理。不能仅靠普通 frame-IID batch；必须按 world、轨迹、覆盖率和分支数平衡。

### 4.3 Teacher 原则

teacher 优先来自与部署局部规划器一致的可达性，而不是简单把完整 mesh 压成二维障碍图。

仿真 teacher 应结合：

- 完整几何和地形；
- 机器人尺寸与碰撞膨胀；
- 坡度、台阶、高度和粗糙度限制；
- 从中心位置执行局部规划/可达性查询；
- 连通区域、分支方向和沿分支可达距离；
- 相邻关键帧之间持续存在的结构变化。

事件标签只有在机器人真实移动后、变化持续多个关键帧时才有效。teacher 生成器必须单独通过人工可视化、局部规划器一致性和基础统计审计，未通过时禁止启动正式训练。

## 5. 拓扑建图方法

### 5.1 节点创建

采用混合节点策略。满足任一条件时生成候选节点：

- 预测到持续的结构变化或新分支；
- 行驶距离达到最大 keyframe 间距；
- 累积转角超过阈值；
- 模型不确定性显著上升；
- 局部规划连续失败；
- 检测到回环候选。

结构事件决定节点的语义重要性，距离 keyframe 保证图连通。阈值必须在 validation 轨迹上标定，并报告灵敏度，不能继续使用无验证的单一常数。

每个节点保存位姿及协方差、128D embedding、显式出口语义、局部结构子图、观测时间、来源扫描和出口状态。

### 5.2 边和回环

普通边只在机器人真实完成节点间移动后建立，并记录长度、代价、风险和执行状态。

回环流程固定为：

```text
embedding 检索候选
  -> 空间/时间和图结构过滤
  -> 局部点云或局部地图配准
  -> 相对位姿与图一致性验证
  -> 通过后合并节点或增加回环边
```

禁止仅凭 embedding/role 相似和固定空间半径直接合并节点。

### 5.3 出口状态

每个结构出口必须区分：

```text
unobserved / observed / attempted / traversed / blocked / temporarily_failed
```

看到一个分支不等于已经探索；局部规划失败不等于永久不可通。状态随新观测和执行反馈更新。

## 6. 拓扑全局规划方法

全局 planner 从全图所有未探索出口中选择目标，而不是只在当前局部 32 个方向中取最大值。

候选效用定义为：

```text
utility = expected_information_gain
        + semantic_novelty
        - graph_path_cost
        - traversability_risk
        - localization_uncertainty
        - repeated_exploration_penalty
        - recent_failure_penalty
```

使用 Dijkstra/A* 在已验证拓扑边上寻找从当前节点到目标 frontier 节点的路径。全局层只发布路径中的下一局部 waypoint，M-TARE local planner 继续负责实时避障和短程轨迹执行。

局部执行结果必须反馈到图中：成功则增加/确认边并消费对应出口；失败则更新风险和临时失败次数，在多次独立失败后才标记 blocked。若学习模块无有效 frontier 或不确定性过高，应切换到安全 fallback，而不是默认向 sector 0 行驶或永久停止。

## 7. 评测协议

离线模型评测至少包括：

- direction/exit precision、recall、PR-AUC 和校准误差；
- reachable-distance error；
- event precision/recall 与空间容差内召回；
- same-topology/different-coverage 与 hard-negative 排序；
- history、pose、rotation、density、遮挡和噪声消融；
- 未见轨迹和未见地下 world 结果。

拓扑图评测至少包括：

- 节点事件 precision/recall；
- 节点冗余与漏检；
- 边正确率和连通率；
- 回环 precision/recall；
- 未探索出口状态准确率；
- 返回远端 frontier 的成功率。

闭环探索评测必须在相同 world、起点、随机种子、传感器、运行时间和局部规划器配置下比较原始 M-TARE baseline 与新方法，并执行多个重复运行。核心指标为覆盖率随时间曲线、最终覆盖面积、探索成功率、路径长度、重复探索率、卡住时间、碰撞/拒绝次数和计算实时性。

`status=COMPLETED` 只表示进程完成，不能作为算法通过结论。baseline 未移动或日志不完整的 run 必须判定为无效比较。

## 8. 分阶段验收门槛

### Gate 1：数据可用

多源地下轨迹清单、统一输入契约、在线/离线回放一致性、split 无泄漏、数据可视化和统计全部通过。

### Gate 2：Teacher 可用

teacher 与局部规划器可达性一致，分支和事件标签通过人工抽检及自动指标；常数预测和简单 coverage 基线被显著超过。

### Gate 3：表示可用

未见轨迹和未见 world 上，学习 embedding 能分开 hard negatives，并在旋转、遮挡和密度变化下保持稳定；历史和相对位姿消融造成可测性能下降。

### Gate 4：拓扑图可用

离线回放图在节点、边、回环和 frontier 状态上达到预定 precision/recall，图保持连通且冗余受控。

### Gate 5：Shadow 可用

在线 shadow 模式实时运行，预测、图和规划决策稳定，且不控制机器人。

### Gate 6：Closed-loop 可用

新方法在多个同条件重复运行中稳定工作，并在覆盖率、重复探索或路径效率上至少一项显著优于 M-TARE baseline，同时安全指标不下降。

只有通过前一 Gate，才能进入下一阶段。

## 9. 现有资产的定位

保留并复用：局部地图与坐标变换代码、LAMP/M-TARE 适配器、rosbag 记录脚本、ROS 启动框架、shadow/closed-loop 接口、可视化和指标工具。

降级为历史 feasibility 证据：五-world v4/v5 数据集、当前 semantic epoch-70 checkpoint、固定 canonical role 作为主 embedding、当前手工节点阈值和简单 unknown-gain waypoint 策略。它们不删除，但不能继续作为最终方案或性能结论。

## 10. 当前唯一正确的下一步

在修改模型或继续调阈值前，先完成地下数据资产审计和新数据集协议：列出每个数据源的 world、轨迹数、时长、点云、位姿、teacher 可用性、坐标系和许可用途；确定目标 benchmark world 及 train/validation/test 轨迹；然后实现统一关键帧导出和无泄漏 split。

## 11. 项目迄今已完成的工作

本节记录现有仓库已经做过的工作及其真实结论。它是历史审计，不代表所有实验都应进入 V2 主方案。

### 11.1 多源数据和局部地图对齐

项目首先审计了 SubT-MRS、LAMP 和 M-TARE 数据接口，并实现了统一的 `StandardFrame`、坐标变换、raycast 和 `LocalStructuralMapBuilder`。

已完成：

- LAMP keyed scan 与 pose graph 适配；
- M-TARE rosbag 和在线 `/registered_scan`、`/state_estimation_at_scan` 适配；
- LAMP 与 M-TARE 生成统一 `[8, 100, 100]` float32 局部结构地图；
- M-TARE bag adapter 与 runtime adapter 在回放样本上达到一致；
- contract、坐标变换、raycast 和 replay parity 测试；
- LAMP/M-TARE 混合 batch 的小型 CNN forward/loss/backward smoke train。

未完成：SubT-MRS UGV2 原始 bag 只有 raw Velodyne packets 和 IMU，缺少已确认可用的 PointCloud2 与全局 pose/path/tf，因此只完成审计，没有进入统一地图导出和正式训练。

### 11.2 结构表面数据集与 bottleneck 表示

项目实现了 `surface_evidence_v1`，用 surface mask、log density、mean height 和 height span 表示局部表面，并生成过 LAMP 和 M-TARE 结构数据集。

随后训练和验证了：

```text
ForcedGlobalBottleneckNet
  -> 128D global latent
  -> StructuralProjectionHead
  -> 64D structure representation
```

该阶段完成了局部表面补全、强制全局 bottleneck、扰动一致性、跨位置检索、M-TARE transfer 和 offline topology-node 实验。多环境结构数据集还加入了 LAMP tunnel/urban/perlin2/ku 等来源。

真实结论是：这条支线证明了局部结构可以压缩，并提供了一定结构变化信号；但离线节点实验仍有 robustness、节点冗余和跨环境稳定性问题，因此 128D/64D 表示不能直接视为最终拓扑表示。

### 11.3 M-TARE 数据记录和迁移验证

项目已经在 campus、indoor、garage、tunnel、unseen mine、external cave 和 SubTGraph 等环境中记录或运行过 M-TARE 数据，保存了 rosbag、配置、topic、节点、容器和运行日志。

这部分完成了：

- M-TARE 原输入 topic 的确认；
- campus/indoor 未见环境的输入导出和冻结模型推理；
- 不同 world 的启动脚本、world 注入和运行审计；
- 原始 TARE baseline 运行记录；
- semantic 模式与 baseline 的部分轨迹和覆盖比较工具。

限制是：若 baseline 未移动、日志不完整或只有 `execution_only` 状态，则对应运行不能用于性能结论。

### 11.4 拓扑语义 teacher 和数据集迭代

项目从 M-TARE 五个仿真 world 构建过完整地图 teacher，并在局部位置生成 student observation 与监督字段。主要 world 为 tunnel、garage、forest、campus 和 indoor。

数据集经历了多次修正：

- v1：完整地图生成方向、距离、面积、独立出口和早期 role；
- v2/v3：改为从 M-TARE 在线 `/registered_scan` 形成 student observation，并修正 traversability；
- supervision audit v2：发现旧出口标签单 bin、方向指标和 history 使用等问题，结论为 `NOT_READY`；
- v4：重新定义连续出口 sector、64D canonical role、静态/动态 mask、history time delta 和数据 contract；
- v5：重新表达 128D rotation-canonical role 和持续结构事件标签。

当前 v4/v5 共有 train=3024、val=356、test=1469 个样本。数据 contract 和小样本可拟合性已经通过，但数据规模、world 多样性、teacher 的 2D/准 2D 假设和 split 仍不符合最终地下部署目标。

### 11.5 拓扑语义模型迭代

项目训练过 current-only、unordered/causal history、ordered GRU 和 role-invariant temporal 等多个模型版本，并预测 direction、distance、area、exit、静态结构、动态事件和 role。

早期模型的正式结论包括 `TOPOLOGICAL_SEMANTIC_MODEL_FAIL` 和 `TOPOLOGICAL_SEMANTIC_V2_FAIL`。发现的问题包括：

- direction 指标曾接近 all-positive 常数基线；
- exit 头在部分运行中完全不预测正样本；
- role 与 coverage 的相关性高于与 teacher topology 的相关性；
- history、pose 和时间顺序消融影响很弱；
- campus 开放环境泛化失败；
- 动态结构事件不稳定。

随后实现了 `PolarEncoder + PolarHeads + fixed canonicalizer` 的 `Semantic` 模型。地下 tunnel/garage 的两折轨迹交叉验证得到较好的 topology ranking 和 direction/exit error，并训练了全量地下部署候选 `semantic_bottleneck_underground_deployment_epoch70.pt`。

该结果证明同一批地下 world 的未见轨迹上存在可学习信号，但没有证明全新地下 world 泛化；coverage 影响仍然明显，部署权重也看过全部 tunnel/garage 训练轨迹。

### 11.6 在线语义拓扑和闭环控制

项目实现了 ROS1 `semantic_topology_global_node.py`，它替换 `/way_point` 的全局来源，同时保留 M-TARE local planner。节点支持：

- 在线 surface evidence 构建和 frozen model 推理；
- direction/distance/exit/static 语义平滑；
- structural-event 节点创建；
- 已行驶边和简单 loop closure；
- traversed/blocked exit 状态；
- 当前节点和远端节点的 semantic frontier；
- 图上 shortest path 回退；
- active target、卡住 watchdog 和短程 waypoint；
- RViz marker、decision trace 和 topology snapshot；
- shadow 与 closed-loop 两种模式。

该节点已经在 indoor、garage、tunnel、unseen mine、external cave 和 SubTGraph operational world 中运行过，并生成了节点和边。

真实结论是：ROS 接口、实时推理、图状态和 waypoint 闭环已经打通；但当前图仍由手工阈值和简化出口状态驱动，规划主要是短程 semantic frontier 控制，尚未达到可靠替代 M-TARE 全局规划器的标准。

### 11.7 当前项目状态总结

当前已经具备的资产是：

```text
数据适配与记录
+ 统一局部地图基础
+ 多版结构/语义模型实验
+ teacher 和数据 contract 审计经验
+ ROS 在线推理接口
+ 增量图与 waypoint 原型
+ shadow/closed-loop 仿真框架
+ 结果、日志和可视化工具
```

当前尚未完成的核心目标是：

```text
足够规模和多样性的地下训练数据
+ 可靠的 2.5D/3D 地下 teacher
+ 能跨轨迹和跨地下 world 泛化的结构表示
+ 可验证的节点、边和回环
+ 完整拓扑全局 frontier planner
+ 与原 M-TARE 的有效同条件重复对比
```

因此，现阶段应定义为“端到端原型和多轮可行性审计已经完成，最终地下结构语义拓扑规划系统尚未完成”。
