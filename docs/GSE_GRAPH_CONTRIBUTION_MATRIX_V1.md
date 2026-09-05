# GSE-Graph 相关工作与贡献矩阵 V1

> 本文件记录旧事件/Composer路线的历史新颖性审计。当前程序构造监督基元关系主方法的权威边界已迁移到`docs/PRIMITIVE_RELATION_NOVELTY_MATRIX_V1.md`；不得把下方Composer差异继续写成投稿主张。

状态：`ACTIVE_NOVELTY_CONTROL_METHOD_GAP_OPEN`  
最近原始来源复核：2026-08-29

本矩阵取代把“出口方向 + 因果规则图”当作最终创新的旧边界。投稿前必须用正式论文版本再次复核。

| 现有工作缺口 | GSE-Graph 的方法差异 | 独立消融 | 必须出现的闭环收益 |
|---|---|---|---|
| Neural Topological SLAM从视觉学习可探索方向并建立ghost nodes，已经覆盖“学习候选拓扑目标”，但regular node仍按视觉定位/新地点更新，未学习地下三维几何关系 | GSE不再声称首次学习ghost/hypothesis；五帧本体LiDAR输出宽高坡度、出口token和route-conditioned relation，先形成provisional geometry-semantic hypothesis，只有trace-consistent双向执行证据才提交verified graph | NTS-style exit-only ghost graph；去掉连续几何；语义proposal直接提交 | 必须同时降低最终committed graph错误和无效验证成本；只提高proposal recall或展示ghost nodes不构成贡献 |
| Multi-Hypothesis Topological SLAM及更早deterministic topological SLAM已维护多个拓扑解释，并通过机器人遍历验证loop/path hypothesis | GSE不声称首次执行验证；区别必须来自学习式3D隧道几何证据如何缩小假设、route-conditioned association如何拒绝相似平行隧道，以及trace geometry如何决定提交/拒绝 | 无学习hypothesis tree；无route-conditioned relation；无反向提交证据 | 在相同执行预算下，必须减少候选数/验证行程并提高committed node/edge F1和false-loop安全性 |
| Cano 已学习隧道出口方向、按出口数量判断intersection/tunnel/dead-end、跟踪出口并做纯拓扑导航；因此“LiDAR出口峰+tracking+按数量建图”已经被覆盖 | 最终目标差异必须是Geometry-Relation Event Composer：5帧因果LiDAR预测route-frame出口token、宽高坡度曲率、persistent/reveal/withdraw、descriptor与uncertainty，并由这些显式量组成事件、触发节点和拒绝关联；当前V2R5仍是独立context event head，只能作基线，不能填入本栏贡献 | V2R5独立event head；exit-only；去掉显式几何；去掉transport；单帧 | composer必须相对V2R5和Cano-like同时提高事件与node/edge F1，并减少错误回环；只提高出口检测不构成贡献 |
| GRID-FAST 从已建立的 2-D grid map 提取结构语义 | 不依赖完整 grid；仅由当前/历史 LiDAR 和位姿在线产生结构事件 | 非学习几何事件图 | 在相同局部执行栈下更早形成正确稀疏图，并改善 coverage-time |
| LiDAR place recognition 通常解决检索/回环，但不决定探索图节点和出口状态 | place descriptor 与 event type、exit-token set、空间门控联合决定关联；模糊样本拒绝合并 | 距离/角度规则关联；无 uncertainty reject | association precision `>=0.98`，false loop merge `<=1%`，错误回环不造成闭环安全/连通退化 |
| PRISM-TopoMap学习多模态place recognition并用scan matching维护在线location graph，重点是无全局坐标的定位与回环 | GSE不把固定位置/keyframe当结构节点；因果LiDAR先预测junction/terminal/turn/transition及出口几何，事件决定节点生成，执行trace决定edge | 学习关联改为距离/角度；去掉显式几何 | 必须独立报告结构node/edge F1和false loop，而不能只用place localization或地图覆盖质量证明 |
| Sequential Probabilistic Descriptor 已用连续图像/LiDAR、概率嵌入和不确定性过滤高风险地点匹配 | GSE不把“时序descriptor + uncertainty reject”单独作为贡献；descriptor只服务结构节点关联，事件必须由可解释出口关系和路线连续几何产生，edge仍由真实穿越提交 | deterministic descriptor；无拒绝；descriptor-only keyframe graph | 必须证明拒绝机制和结构几何共同降低false loop且不把召回压到不可用；单独提高place-recognition校准不构成GSE贡献 |
| OVTG用CLIP视觉语言嵌入和GAT在在线位置图中对齐开放词汇语义，但新节点仍由空间新奇度触发，edge连接时序前驱或邻近可通行节点 | GSE无语言目标或对象语义；五帧本体LiDAR的地下几何事件直接决定稀疏结构节点，出口与place descriptor决定拒绝感知混淆，只有完成物理穿越才提交edge | 固定metric-anchor节点；规则关联；无显式几何 | 必须证明学习几何改变客观结构node/edge F1并减少false loop，不能只报语义导航SR/SPL |
| Bio-Inspired Hybrid Map把含学习特征的spatial-implicit local frames作为factor-graph节点，并以累计行程与视点变化创建新local frame | GSE不是把每个学习局部帧作为keyframe节点；其显式隧道事件决定稀疏结构节点，出口token参与关联，edge只在穿越证据到达另一节点后提交 | 固定metric-anchor图；无显式几何；规则关联 | 必须证明收益来自事件驱动的节点/关联，而不是学习特征keyframe加RRT*导航 |
| FHT-Map以CNN视觉特征和局部LiDAR扫描构造main/support两类节点，并用视觉信息丰富度、节点密度及自由空间路径修补控制节点和edge | GSE不依赖RGB、累计free-space map或异步A*补边；五帧本体LiDAR预测的地下结构事件触发节点，出口token参与关联，连接只来自执行trace | 固定metric-anchor图；exit-only；规则关联；无显式几何 | 必须提高客观结构node/edge F1和探索效率，不能只比较存储、重定位或任意路径规划能力 |
| Semantic Topometric Mapping从已经累计的二维occupancy grid规则分割intersection、pathway、dead-end和frontier，并利用这些结构语义选择探索目标 | GSE不等待完整二维栅格分割；本体五帧LiDAR直接预测三维隧道几何事件、出口集合和重访描述子，学习输出决定结构节点与关联 | 非学习几何事件图；exit-only规则图 | 必须先在客观结构图上提高node/edge F1，再在相同局部执行栈下改善coverage-time，不能只证明结构代价函数有效 |
| Learned frontier/ARiADNE 学习目标价值或策略 | GSE 不学习 frontier value；学习对象是可解释结构与关联，planner 只消费未穿越出口和 verified edges | 移除 edge geometry；固定规则图 | 收益应来自图质量与结构状态，而非额外 policy capacity |
| 2025 Heterogeneous Topological Graph Exploration从二维probability grid、ESDF skeleton、frontier/border/viewpoint确定性构图，再用GNN和RL选择viewpoint | GSE不学习全局探索策略，也不从二维grid/skeleton预定义节点；学习对象是局部地下结构事件和关联，连接仍由真实穿越确认 | 原M-TARE；非学习几何事件图；无显式几何 | 在保持M-TARE planner不变时，离线图质量和闭环收益必须同时出现，排除策略容量或二维skeleton构图带来的收益 |
| AIM-Mapping从局部栅格提取结构特征并用特权信息训练多机器人RL目标分配，但topological representation仍由几何距离和boundary candidates构造 | GSE不学习策略、不使用全局/特权grid；学习输出本身定义隧道结构事件、出口关系和重访关联，M-TARE局部执行保持不变 | 原M-TARE；规则关联；非学习几何事件图 | 学习几何必须先改善离线图，再在同一planner下改善单/多机器人探索，禁止把策略容量当结构语义收益 |
| M-TARE 使用多分辨率 metric representation | 保留其 local planner/control，只替换 global structural representation 与目标接口 | 原始 M-TARE | 单/多机器人在相同传感器、局部规划、时间预算下报告 coverage、travel、redundancy、latency 与通信 |
| LTVMap、Similarity-Score Memory、Graph Voronoi、SAGE 已覆盖低带宽/共享拓扑或多机分配 | 新主张不放在“共享图/Hungarian”；放在 geometry-semantic event、uncertainty-aware association 和 trace-verified execution graph | independent graph；无 uncertainty；无 edge geometry | 2/3/4 机器人效率、重复覆盖、冲突和通信量与结构图质量一致改善 |
| GBPlanner由累积点云构造凸多面体区域和frontier拓扑，几何直接参与节点/边，但不从因果LiDAR学习结构事件或地点/出口关联 | GSE以局部五帧观测预测可解释隧道几何和事件，只有真实穿越提交edge；不由已知空间polyhedron overlap生成连接 | 非学习几何事件图；无显式几何 | 在同一M-TARE局部栈下，学习语义必须改善离线node/edge F1并在闭环减少重复行程 |
| SG-SLAM学习/利用语义图增强LiDAR SLAM、重定位和loop closure，目标是全局一致定位与点云地图 | GSE不替代SLAM；学习输出决定探索图的结构节点、未穿越出口和拒绝合并，评价目标是拓扑探索而非里程计/配准精度 | 距离/角度关联；无uncertainty reject | place/exit关联精度和false merge先独立过门槛，再证明探索收益 |
| 2026 SATE用UAV俯视图U-Net traversability、semantic-seeded Voronoi和拓扑稀疏正则指导地面二维探索 | GSE只用机器人本体地下LiDAR、无俯视先验或dense safe-area heatmap；学习的是隧道几何事件与出口关系，图连接由执行验证 | 非学习Voronoi/规则图边界对照；无几何语义 | 未见三维隧道拓扑上保持图结构和闭环收益，不能只报告节点更少 |
| MLATC从连续3D点云用ART winner clustering在线学习多层位置/法向/可通行图，主问题是大规模最近邻搜索效率 | GSE监督学习的是局部结构事件、metric geometry和跨重访descriptor；节点由事件触发，edge不由两个空间winner相邻产生 | 固定距离/角度关联；规则metric anchor图 | 除运行时间外必须报告客观node/edge F1、cycle rank和false loop merge |
| Topo-Field从RGB-D序列训练layout-object-position neural field，再查询field建topometric map并用LLM验证edge | GSE无dense neural field、LFM/LLM或对象语义；用本体LiDAR即时输出隧道结构事件，edge只由机器人实际穿越验证 | 无显式几何；无edge geometry | 在未见地下拓扑上证明因果在线图和探索收益，而非已采集室内场景的查询规划 |
| 地下segmented-map/keyframe方法从累积LiDAR LOS区域生成SER和frontier拓扑 | GSE不先建立分割地图；局部学习事件决定结构节点，描述子与出口set处理重访，连续几何成为edge属性 | 非学习几何事件图；exit-only规则图 | 更早且更准确形成稀疏图，并用相同局部规划器改善coverage-time |

## 2026-08-24 定向重合检查

本轮核对的直接原始来源包括：

- Cano et al., *Autonomous Navigation in Large-Scale Underground Environments Based on a Purely Topological Understanding of Tunnel Networks*, JFR 2026, DOI `10.1002/rob.70157`；
- Yang et al., *Graph-Based Topological Exploration Planning in Large-Scale 3D Environments*, arXiv `2103.16829`；
- Wang et al., *Leveraging Semantic Graphs for Efficient and Robust LiDAR SLAM*, arXiv `2503.11145`；
- Ding et al., *Structure-Aware Topological Exploration: A Semantic Seeded Voronoi Approach for Unstructured Environments*, Electronics 2026, DOI `10.3390/electronics15051033`。
- Ofuchi et al., *MLATC: Fast Hierarchical Topological Mapping from 3D LiDAR Point Clouds Based on Adaptive Resonance Theory*, arXiv `2511.22238`；
- Hou et al., *Topo-Field: Topometric Mapping With Brain-Inspired Hierarchical Layout-Object-Position Fields*, RA-L 2025；
- Sohn et al., *Topological Exploration using Segmented Map with Keyframe Contribution in Subterranean Environments*, arXiv `2309.08397`。
- Muravyev et al., *PRISM-TopoMap: Online Topological Mapping with Place Recognition and Scan Matching*, RA-L 2025, DOI `10.1109/LRA.2025.3541454`, arXiv `2404.01674`, BibTeX `muravyev2025prism`；
- Cheng et al., *Asymmetric Information Enhanced Mapping Framework for Multirobot Exploration based on Deep Reinforcement Learning*（AIM-Mapping）, T-RO 2025, DOI `10.1109/TRO.2025.3619045`, arXiv `2404.18089`, BibTeX `cheng2025asymmetric`。
- Fredriksson et al., *Robotic Exploration through Semantic Topometric Mapping*, arXiv `2406.18381`, BibTeX `fredriksson2024semanticexploration`；
- Li et al., *Learning to Explore Efficiently: Heterogeneous Topological Graphs and Lightweight Global Reasoning for Robotic Exploration*, RA-L 2025, DOI `10.1109/LRA.2025.3622906`, BibTeX `li2025heterogeneous`。
- Dang and Huber, *Bio-Inspired Hybrid Map: Spatial Implicit Local Frames and Topological Map for Mobile Cobot Navigation*, arXiv `2507.04649`（IROS 2025版本边界仍以作者公开稿为准）, BibTeX `dang2025bioinspired`。
- Song et al., *FHT-Map: Feature-based Hierarchical Topological Map for Relocalization and Path Planning*, arXiv `2310.13899`, BibTeX `song2023fhtmap`。
- Li et al., *OVTG: Open-Vocabulary Topological Semantic Graph Construction Method for Unknown Environments*, RobCE 2026, DOI `10.1145/3820709.3820713`, BibTeX `li2026ovtg`。
- Pan et al., *Sequential Probabilistic Descriptor via Uncertainty-Aware Multi-Modal Fusion for Safety-Critical Place Recognition*, RA-L 2026, DOI `10.1109/LRA.2026.3669806`。

2026-08-28新增直接边界：Chaplot et al.的Neural Topological SLAM已从学习式explorable-area prediction生成ghost nodes；Tully et al.的Multi-Hypothesis Topological SLAM已维护edge-ordered graph hypotheses；deterministic topological SLAM已有通过实际路径遍历确认/拒绝loop hypothesis。因此“学习候选+执行验证”不能作为新颖性主张。当前尚未发现同时满足“机器人本体因果3D LiDAR、显式隧道连续几何和exit relation、route-conditioned拒绝关联、trace-consistent node/edge提交”四项机制的同构工作，但这只是更窄的待验证组合主张。若学习几何不能减少验证候选/行程或提高最终committed graph质量，则双层方法也必须停止。

2026-08-29代码审计确认当前V2R5尚未实现上表目标差异：其event head与token/transport/geometry并行，在线图由独立event触发。故贡献矩阵中的composer是待实现、待消融的方法合同，不是已完成贡献；V2R5归入Cano-like之上的强学习基线。

## 2026-08-29 方法生存条件

原始来源复核得到三个不能再退让的边界：

1. Cano 已覆盖本体3D LiDAR出口方向、出口时序稳定化、路口/隧道/尽头判断和轻量拓扑图；因此 Action-Set Relation Composer 单独成立时只能算更强的 Cano-like 基线。
2. PRISM-TopoMap 和 Sequential Probabilistic Descriptor 已分别覆盖学习式在线地点关联、时序概率描述子与不确定性过滤；因此 descriptor 或 refusal 单独成立也不是贡献。
3. Semantic Topometric Mapping 已覆盖结构语义驱动探索，但其语义来自累计二维栅格分割；GSE 的可辨识差异必须是从因果本体3D LiDAR学习连续隧道度量几何，并让该几何与出口关系直接产生稀疏结构事件。

据此，`Metric-Change Composer` 及其可观测输入不再是可删除的附加模块，而是论文方法成立的必要组成。若当前宽/高/坡度/曲率标量状态在三seed正式审计中不能恢复 corrected change points，必须先执行 `RouteGeometryProfile` 可见性和Teacher唯一性证明；不得退回“出口计数 + learned descriptor + uncertainty”的已覆盖组合。只有以下完整链条的独立消融和闭环收益同时成立，才允许把GSE-Graph写成主贡献：

```text
causal LiDAR
-> learned metric route geometry + executable exit relations
-> typed structural events
-> uncertainty-refusing association
-> traversal-verified edges
-> measurable graph and exploration gain
```

## 直接否定论文主张的条件

- 学习几何相对非学习估计器的连续量 MAE 未降低 10%；
- event macro-F1 相对 exit-only 未提高 5 个百分点；
- association precision 低于 0.98 或 false loop merge 超过 1%；
- graph node/edge F1 相对 exit-only 规则图未提高 5 个百分点；
- 闭环收益只能通过修改 M-TARE local planner、评分或测试后阈值得到；
- 相同输入、输出和建图机制已被已发表工作完整覆盖。
