# 程序构造监督基元关系图：相关工作与贡献矩阵 V1

状态：`ACTIVE_NOVELTY_CONTROL`
原始来源复核日期：2026-09-01

## 1. 不允许声称的“创新”

现有工作已经覆盖下列单点，因此论文不得声称：

- 首次从点云学习几何基元或超二次曲面；
- 首次从程序化构造数据学习逆向几何程序；
- 首次联合学习部件几何和部件关系图；
- 首次从自由空间距离场或墙面层级构造在线结构图；
- 首次端到端联合预测矢量几何及其拓扑关系；
- 首次用学习式地点识别维护在线拓扑图；
- 首次在地下环境用LiDAR检测隧道方向并建立拓扑图；
- 首次用稀疏拓扑表示进行地下或多机器人探索。

可检验的论文主张必须是完整链条：

```text
部分可见的五帧本体LiDAR
→ 显式扫掠隧道基元
→ 每端点O(E)共享三维组合锚点 / 断开重叠 / 跨帧对应及不确定性
→ 基元关系直接形成持久结构节点和候选端口
→ 模糊关联拒绝合并，真实穿越才提交全局edge
→ 在不改变M-TARE局部执行栈时改善图质量与探索
```

## 2. 逐项贡献边界

| 原始工作已完成什么 | 与本方法最接近的重合 | 本方法必须证明的差异 | 独立消融或对照 | 论文生存证据 |
|---|---|---|---|---|
| [SPFN, CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Li_Supervised_Fitting_of_Geometric_Primitives_to_3D_Point_Clouds_CVPR_2019_paper.html)从点云监督预测可变数量平面/球/柱/锥及参数；[CPFN, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Le_CPFN_Cascaded_Primitive_Fitting_Networks_for_High-Resolution_Point_Clouds_ICCV_2021_paper.html)进一步处理高分辨率与小基元 | 集合式显式基元预测、Hungarian匹配、表面拟合 | 目标不是静态CAD物体分解，而是从部分可见因果扫描恢复有限扫掠隧道基元，并同时恢复可执行连接、断开重叠和时间身份 | 同输入非学习拟合；去参数监督；去表面/射线重建；去关系和时序 | 几何误差、目标coverage和关系F1必须同时改善；只提高Chamfer不够 |
| [Superquadrics Revisited, CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Paschalidou_Superquadrics_Revisited_Learning_3D_Shape_Parsing_Beyond_Cuboids_CVPR_2019_paper.html)已用可变数量超二次曲面做无监督形状解析；[ResFit, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Ganeshan_Residual_Primitive_Fitting_of_3D_Shapes_with_SuperFrusta_CVPR_2026_paper.html)以SuperFrusta和残差迭代得到紧凑高保真分解 | 超椭圆/超二次基元、存在概率、表面重建、紧凑分解 | 扫掠轴线、端口与物理连接是导航结构实体；输出跨时间持续并成为在线图，而不是完整物体的可编辑重建 | 非学习拟合；32槽容量；重建-only模型 | 必须在未见拓扑和三截面变体上恢复结构关系并改善下游图，而非仅展示好看的基元 |
| [CSGNet, CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Sharma_CSGNet_Neural_Shape_CVPR_2018_paper.html)、[BSP-Net, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Chen_BSP-Net_Generating_Compact_Meshes_via_Binary_Space_Partitioning_CVPR_2020_paper.html)和[CAPRI-Net, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Yu_CAPRI-Net_Learning_Compact_CAD_Shapes_With_Adaptive_Primitive_Assembly_CVPR_2022_paper.html)已学习CSG程序、凸分解或自适应primitive assembly | 程序/CSG提供结构监督，基元通过组合解释表面 | 不预测完整隐藏地图或完整CSG程序；Teacher只裁剪到当前五帧可观测实体，学生预测局部物理关系，执行trace决定全局连接 | 去construction参数监督；reconstruction-only；禁止未来/完整地图输入 | 因果/泄漏审计必须PASS，且关系监督相对纯重建产生独立图收益 |
| [ArcPro, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Huang_ArcPro_Architectural_Programs_for_Structured_3D_Abstraction_of_Sparse_Points_CVPR_2025_paper.html)已通过前向程序合成“稀疏点云—建筑程序”训练对，学习逆向建筑程序 | 这是“程序生成数据监督结构恢复”的直接近邻，不能回避 | ArcPro恢复静态建筑程序；本方法处理移动机器人局部部分观测、时序遮挡、断开但重叠隧道、在线关联和执行验证探索图 | 单帧/五帧；无temporal；无disconnected-overlap；直接提交edge | 必须证明时间/断开关系降低重复节点和false loop，并产生闭环探索收益 |
| [StructureNet, TOG 2019](https://arxiv.org/abs/1908.00575)已用顺序不变n-ary graph联合学习部件几何、邻接/对称关系和结构生成；[Regularized Primitive Graph Learning, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Wang_Regularized_Primitive_Graph_Learning_for_Unified_Vector_Mapping_ICCV_2023_paper.html)已联合预测primitive及pairwise关系 | 部件集合、关系矩阵、顺序不变图学习 | 不是类别对象生成或二维矢量地图；每个隧道端点预测一个sensor-polar共享三维composition anchor与尺度，物理连接由对称Gaussian兼容度导出而非独立`O(E^2)`分类；关系还包含空间重叠硬负例和跨帧身份，并直接约束因果机器人图 | 旧独立pair head；anchor改为直接端点距离；去overlap hard-negative；去temporal；规则图基线 | attachment/overlap/temporal指标和node/edge F1都必须提升，且`O(E)`表示必须保持高精度非空连接；只报latent、压缩率或重建质量均不成立 |
| [TopoNet, arXiv 2023](https://arxiv.org/abs/2304.05277)已用场景图网络联合推理车道连接及交通元素关系；[DAGMapper, ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Homayounfar_DAGMapper_Learning_to_Map_by_Discovering_Lane_Topology_ICCV_2019_paper.pdf)已把车道几何和拓扑变化联合建模 | “先得到几何实例、再学习实例关系”及端到端geometry-to-graph都不是首次 | 本方法处理无视觉纹理的三维地下自由空间，输入是五帧本体LiDAR；学习的是可组合端点锚点而不是道路邻接矩阵，关系包含上下层断开重叠、跨帧身份和拒绝，并且网络预测不能直接提交可执行edge | composition-anchor与旧pairwise relation、直接junction/port grouping对照；去disconnected-overlap；单帧替代五帧 | 未见地下拓扑的关系F1、安全提交、false loop和最终图F1必须同时成立；道路数据上的成功不能替代地下证据 |
| [PRISM-TopoMap, RA-L 2025](https://arxiv.org/abs/2404.01674)用学习式多模态地点识别和scan matching在线维护局部对齐location graph；2026时序概率描述子已用不确定性过滤危险地点匹配 | 学习descriptor、时序输入、不确定性拒绝和在线loop closure均已有 | descriptor和拒绝不是单独贡献；节点内容/候选端口由学习基元关系产生，且不确定匹配保留provisional node，edge仍需真实穿越 | 距离/角度关联；descriptor-only keyframe图；去不确定性 | association precision≥0.98、false loop≤1%，同时保持可用召回和正确图不变量 |
| [Hydra, RSS 2022](https://www.roboticsproceedings.org/rss18/p050.pdf)已从局部ESDF/GVD增量抽取place graph并聚类房间；[S-Graphs+, RA-L 2023](https://arxiv.org/abs/2212.11770)已把keyframe、墙面、房间和楼层放入可优化层级图 | 从几何/自由空间形成持久节点、关系和回环约束已有成熟解析方案 | 不先构建稠密ESDF，也不假设平面墙和房间层级；从局部因果LiDAR学习隧道扫掠实体及物理关系，保留provisional节点，并只用真实穿越确认全局edge | Hydra式自由空间骨架/非学习几何图；去学习关系；相同轨迹的keyframe图 | 在相同输入预算下证明更早或更准的结构节点、更低图/通信开销，并保持可执行连通性；仅展示稀疏图不够 |
| [Cano et al., JFR 2026](https://onlinelibrary.wiley.com/doi/10.1002/rob.70157)已从3D LiDAR学习隧道分布，按隧道/路口/尽头建立纯拓扑地图并做仿真和真实导航 | 合成地下LiDAR、隧道拓扑、轻量图和探索均已覆盖 | 不以出口方向或出口数量作为结构语义；学习对象是具有轴线/截面/端口的物理基元和连接关系，并处理其论文明确指出的loop closure困难 | Cano-like出口模型+规则图；去显式几何；去关系 | 完整方法须相对Cano-like提高几何、关系、图F1并降低错误回环；只提高出口检测不构成贡献 |
| [地下Segmented Map拓扑探索](https://arxiv.org/abs/2309.08397)从累积3D dense map和LiDAR LOS生成SER、keyframe拓扑及frontier，并有仿真/实地收益 | 地下LiDAR、结构区域、拓扑探索 | 本方法不先构建完整dense map；从当前五帧因果观测产生结构实体并早期更新图 | 非学习几何图；dense-map/keyframe基线（若接口可公平复现） | 必须展示更早/更准的结构图及相同局部栈下coverage-time收益 |
| [M-TARE, Science Robotics 2023](https://biorobotics.ri.cmu.edu/papers/paperUploads/scirobotics.adf0970.pdf)已证明多分辨率表示可提高大规模单/多机器人探索效率 | 全局表示、目标分配、多机器人地下探索 | 保留其定位、局部地图、局部规划、避障和控制，只替换全局结构表示、目标单位和图增量共享 | 原始M-TARE；完整方法去edge几何；独立图/共享图 | 同world/start/seed/sensor/runtime配对下报告coverage、路程、冗余、延迟、通信和失败 |
| 2026年的[Hickory](https://arxiv.org/abs/2606.01545)已把superquadric对象用于层次场景表示和地图对齐 | superquadric实体、机器人地图、关系/对齐 | 本方法不是物体级语义图或已知对象对齐，而是从隧道自由空间边界恢复可通行扫掠结构并用于探索图生成 | 仅descriptor/map-alignment关联；无端口关系 | 必须以端口、图结构和探索指标证明差异，不能只报配准或表示紧凑度 |

## 3. 当前新颖性判断

截至本次原始来源复核，没有发现同时满足以下全部条件的同构方法：

1. 输入仅为机器人本体五帧因果3D LiDAR与相对里程计；
2. 输出可交换的显式扫掠隧道基元；
3. 以每端点共享三维组合锚点和尺度联合学习端口连接、断开重叠、时间对应与不确定性，而非输出独立pair矩阵；
4. 这些关系直接形成在线持久结构节点和候选端口；
5. 模糊关联拒绝合并，只有物理穿越提交全局edge；
6. 在不改变局部执行栈的公平单/多机器人地下探索中证明收益。

这是“组合机制尚未发现完全重合”的判断，不是“首次基元学习”或“首次拓扑探索”的宽泛主张。若P2/P3不能证明学习关系改善图结构，或闭环收益只能靠规划器调参得到，则该组合贡献也不成立。

## 4. 必须进入论文的对照逻辑

```text
SPFN / Superquadrics / ArcPro / StructureNet / TopoNet
  界定：基元、程序监督、部件关系本身都不是首次

Cano / Hydra / S-Graphs+ / PRISM-TopoMap / Segmented-Map Exploration
  界定：出口拓扑、学习地点关联和地下拓扑探索本身都不是首次

我们的必要证据
  学习的隧道基元关系
  → 更可靠的持久结构图
  → 不改局部执行栈的探索收益
```
