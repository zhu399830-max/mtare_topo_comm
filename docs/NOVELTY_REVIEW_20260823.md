# 结构语义因果拓扑探索：新颖性审计（2026-08-23）

状态：`SUPERSEDED_METHOD_BOUNDARY_RETAINED_AS_EXIT_ONLY_BASELINE_REVIEW`

> 2026-08-23 更新：本文后半部分对“当前可声明方法”的描述仅适用于旧 V9/V5 exit-only 基线。论文主方法已切换为学习显式几何结构、事件与关联的 GSE-Graph；最新新颖性边界见 `docs/GSE_GRAPH_CONTRIBUTION_MATRIX_V1.md`。

用途：约束论文主张和后续对照，不代替正式实验结论。检索日期为2026-08-23；投稿前必须再次更新。
已核对书目保存在`docs/references/structural_topology_novelty_20260823.bib`。SAGE v1的Crossref作者顺序与论文PDF冲突，当前BibTeX按PDF顺序并显式要求最终出版前复核。

## 当前可辩护的研究问题

不能把“采用拓扑图”“用语义做探索”或“减少地图通信”单独写成创新。当前真正需要实验支持的主张是：

> 在仅有因果历史和局部LiDAR结构语义的条件下，维护带verified edge、未完成exit stub及执行反馈生命周期的
> persistent topometric graph，能否作为M-TARE全局表示/目标层的轻量替代，并在完全保留其local planner、
> path follower、避障和控制时，提高地下探索的覆盖效率、路径效率、稳定性或通信效率？

论文必须把贡献拆成两部分验证：局部结构语义是否提供可用的未见出口信息；因果图及exit生命周期是否把该信息
转化为比原M-TARE coarse global map更有效的长期决策。只提高出口检测F1或只展示一张拓扑图均不充分。

## 最近且直接相关的工作

| 工作 | 已有能力 | 与本项目的关键差异 | 本项目必须提供的证据 |
|---|---|---|---|
| [M-TARE / representation granularity](https://doi.org/10.1126/scirobotics.adf0970) | 高分辨率局部图+低分辨率全局图；单机TSP、多机VRP及受限通信pursuit；大量仿真和实地证据 | 当前项目保留其局部执行栈，只替换全局表示/目标选择；不能把完整M-TARE称为弱基线 | 同sensor/local stack/budget的配对对照；规划运行时间、coverage-time、路径与通信量；明确只替换global layer |
| [Cano et al., JFR 2026](https://doi.org/10.1002/rob.70157) | 程序化地下环境生成、合成LiDAR训练、CNN预测可通行隧道方向、纯拓扑指令导航、仿真与真实地下验证 | 已覆盖“学习隧道方向+纯拓扑地下导航”；本项目不能据此声称首次学习地下拓扑 | 证明不是复刻其CNN/导航：客观teacher、persistent causal graph、exit lifecycle、M-TARE闭环替换及因果消融 |
| [GRID-FAST, JINT 2024](https://doi.org/10.1007/s10846-024-02180-6) | 从2D grid map提取intersection/path/dead-end/unexplored-path结构语义，生成紧凑topometric map | 其结构语义来自已建grid map；本项目目标是从局部LiDAR与历史因果构图，并用于在线探索目标 | 报告partial-observation/causal约束、未观测exit stub和执行反馈；不要把“结构语义拓扑”本身写成首次 |
| [Similarity Score Map and Topological Memory, RA-L 2024](https://eunsunlee.github.io/SimScoreMap/) | 多机器人共享视觉topological memory，用特征相似度估计访问频率并选择低相似frontier | 已实现topological-memory global goal planning；其节点依赖视觉相似/室内场景 | 多机器人比较必须报告图通信字节、冲突、重复覆盖；强调地下结构出口身份和trace-verified connectivity，而非泛化的graph memory |
| [Balanced Collaborative Exploration via Distributed Topological Graph Voronoi, 2025](https://arxiv.org/abs/2510.24067) | GVD/frontier/coverage混合拓扑图，分布式weighted graph Voronoi负载均衡及理论界 | 已覆盖topological multi-robot allocation与在线完整性；本项目的Hungarian分配本身不构成足够新颖性 | 多机器人贡献应聚焦结构语义exit单位、低带宽因果图共享和执行反馈；必须与原M-TARE/independent策略公平比较 |
| [UAVs Beneath the Surface / LTVMap](https://arxiv.org/abs/2206.08185) | 地下多UAV的SphereMap/FacetMap及低带宽topological-volumetric LTVMap，真实SubT部署 | 已证明轻量拓扑-体积表示适合地下多机器人通信 | 不能笼统声称首次低带宽地下拓扑协作；必须量化消息字节、图增量、性能—通信折中和UGV结构假设 |
| [Hierarchical graph terrain-aware ground-air exploration, 2025](https://arxiv.org/abs/2505.14859) | 几何+语义traversability层级图、置信度frontier共享、真实地下异构协作 | 已覆盖语义层级图与地下异构协作，但语义主要是地形可通行性 | 本项目要限定为tunnel-network structural semantics及exit lifecycle，不泛称一般semantic traversability |
| [SAGE: Decoupling Spatial Logic from Metric Scale, 2025 preprint](https://doi.org/10.21203/rs.3.rs-8307666/v1) | 学习尺度无关拓扑逻辑并用于zero-shot multi-robot exploration | 对“拓扑抽象带来尺度泛化”的主张形成直接竞争；正式同行评审状态需投稿前复核 | 严格unseen拓扑/尺度实验、规则图与学习部分边界、M-TARE受控替换和失败案例必须完整；不能只报开发地图 |

## 论文贡献边界

在当前实现和证据下，候选贡献应写为：

1. 一条可追溯的程序化topology→局部LiDAR→结构出口监督链，学习部分限定为exit direction；count/role为冻结几何接口，禁止写成端到端结构理解。
2. 一个因果在线topometric graph：persistent structural nodes、trace-verified edges、unresolved exit stubs、visited/confidence和执行结果生命周期；禁止写成完整未知地图预测。
3. 一个严格受控的M-TARE global-target-layer replacement：传感器和localPlanner/pathFollower/control完全相同，用配对block证明收益或失败机制。
4. 若Gate 7通过，再增加基于共享因果图的多机器人分配和通信证据；Hungarian算法本身不是创新，结构出口任务单位、增量图通信与闭环反馈才是可检验贡献。

## 会直接否定高水平主张的缺口

- V5若不能在正式30-case中相对原M-TARE产生稳定且有置信区间支持的收益，论文只能定位为失败机制/系统研究，不能宣称更优planner。
- 没有Gate 7真实1/2/3/4 robot闭环，不能声称多机器人贡献；CPU单测不算实验。
- C10或其它sealed strict worlds若参与开发选择，泛化结论失效。
- 没有真实地下闭环时，不能使用field-robotics superiority措辞；纯仿真更适合聚焦RA-L/会议故事。
- 若不报告Cano、GRID-FAST、Similarity-Score Topological Memory、Topological Graph Voronoi、LTVMap与SAGE，相关工作和新颖性论证不完整。

## 投稿前复核任务

1. 下载并逐节核对上述论文的正式版本、实验场景、开源许可和具体指标；当前表格只基于论文/官方项目页。
2. 更新2026年下半年新论文，并核对SAGE和Graph Voronoi工作的最终同行评审状态。
3. 将每条最终contribution绑定到sealed run、case、metric、统计表和图；没有证据的措辞从摘要和结论中删除。
4. 在最终论文中把Cano代码/数据生成器、M-TARE/AEE、第三方模型和本项目新增部分分开归属。
