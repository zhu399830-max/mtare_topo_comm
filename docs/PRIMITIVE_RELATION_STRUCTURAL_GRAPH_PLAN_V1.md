# 程序构造监督的几何基元关系结构图计划 V1

状态：`AUTHORITATIVE_METHOD_PLAN`
日期：2026-08-29

## 1. 单一论文目标

完成一篇以如下问题为中心、达到 ICRA/IROS/RA-L 投稿完整度的论文：

> 能否利用程序化地下地图的真实构造程序，训练模型从因果 LiDAR 恢复显式扫掠几何基元、基元间组合与跨帧变换关系，并把这些学习结果持续融合为执行验证的结构语义拓扑图，从而改善单机器人和多机器人地下探索？

论文主链固定为：

```text
程序化地图构造程序
  -> 基元/变换/组合 provenance
  -> 因果 LiDAR
  -> learned explicit geometric primitives
  -> learned primitive relations and temporal correspondence
  -> persistent structural-semantic graph
  -> traversal-verified topometric graph
  -> M-TARE global exploration / original local planner
```

不再以五类事件分类、出口方向识别、固定距离节点、dense地图骨架化或通用GNN/RL规划作为论文主方法。

## 2. 创新边界

单独的超二次曲面拟合、CSG组合、部件关系图、地点拓扑图和低带宽探索均已有工作。可辩护贡献必须是完整因果链：

1. **程序构造监督的因果几何基元学习**：地图生成程序提供真实基元、SE(3)变换、组合树和表面来源；学生只看部署时可得的LiDAR与相对里程计。
2. **基元关系形成结构语义图谱**：模型学习端口连接、连续、分离、遮挡和跨帧对应；结构语义存在于几何实体及关系图中，不依赖手写junction/turn/terminal分类规则。
3. **执行验证的在线探索图**：学习输出只能产生候选端口和候选节点，真实穿越才提交全局edge；不确定匹配拒绝合并，并量化探索效率、错误回环和通信收益。

最近相关路线必须在投稿前再次复核：SPFN、Superquadrics Revisited、BSP-Net、CAPRI-Net、StructureNet/Seg&Struct、Primitive Graph Learning、Cano、PRISM-TopoMap、M-TARE与地下拓扑探索工作。不得声称首次学习基元、首次拓扑探索或首次低带宽地下协同。

## 3. 数据与构造监督

### 3.1 当前可复用开发数据

- 独立拓扑父世界：C01--C08共80个；C01--C06拟合、C07选择、C08一次零适配开发迁移。
- 当前有效资产：252,430个唯一16x720 LiDAR帧、188,126个五帧因果序列。
- 精确序列数：C01--C06=`142,184`，C07=`21,548`，C08=`24,394`。
- C09已有历史开发暴露，只可作污染域诊断；C10和正式M-TARE比较世界在方法冻结前禁止读取。
- 原LiDAR、TNG、splines、mesh、relative-pose registration、raycaster和训练基础设施继续复用。

### 3.2 必须补充的形状多样性

当前生成器主要产生椭圆拱顶和平底，不能独立证明矩形—椭圆基元学习。保持80个拓扑父世界不变，为每个父世界冻结三种几何实现：

1. 原椭圆截面；
2. 圆角矩形截面；
3. 截面宽高、形状指数和坡曲沿中心轴C1连续变化的混合截面。

拟议开发规模为240个几何实现、757,290个LiDAR帧和564,378个五帧序列；独立统计单位仍是80个拓扑父世界，几何变体不得跨父世界split。正式导出前必须通过只读inventory、资源测算、Data Card和preflight，计数漂移立即停止。

必须包含：弯曲、坡道、收缩/扩张、T/Y/X连接、chamber、遮挡、不同高度的空间重叠但物理不连接隧道，以及相似截面但不同连接关系的困难负样本。

### 3.3 Construction Program Supervisor

“Teacher”重新定义为地图真实构造记录，而不是事件规则：

```text
PrimitiveConstructionGraph
  primitives[]
  sweep_centerlines[]
  cross_section_parameters[]
  transforms_SE3[]
  composition_operations[]
  endpoint_connections[]
  physical_component_identity[]
  surface_to_primitive_provenance[]
```

Teacher必须按当前五帧传感器视场、ray support和遮挡裁剪。不可见基元、未来连接、世界ID、TNG identity、绝对pose和完整地图不得进入学生forward。TNG只用于独立拓扑评价，不作为模型输入。

## 4. 主模型

### 4.1 显式几何输出

每个观察最多输出32个可交换基元槽。该容量由C01--C06真实五帧全扫描的最大可见基元数`16`乘预注册25%余量后，从`8/16/32`中选择；C07/C08冻结迁移最大为`23/19`，32槽零overflow。旧8槽接口被正式容量证据取代，不允许按命中数截断Teacher：

```text
SweptSuperellipsePrimitive
  axis_control_xyz_m[3,3]
  endpoint_half_axes_m[2,2]
  endpoint_shape_exponent[2]
  existence_probability
  endpoint_descriptor[2,D]
  geometry_uncertainty
```

三个控制点表达局部弯曲轴；端点宽高和形状指数表达沿程渐变。重力与轴切线确定截面局部坐标，避免无意义的完整旋转自由度。

### 4.2 关系与时序输出

```text
PrimitiveRelationObservation
  endpoint_attachment_probability[32,2,32,2]
  temporal_correspondence[5,32,32+dustbin]
  disconnected_overlap_probability[32,32]
  relation_uncertainty
```

学生网络由现有circular range encoder、五帧因果时序聚合、primitive set decoder和relation transformer组成。关系头根据显式几何、局部特征和相对变换联合预测连接，不使用固定距离/角度规则代替学习。

`unexplored`不是可由五帧LiDAR唯一辨识的几何属性：同一局部观测在首次到达和重访时具有不同执行状态。因此不把`unexplored_port`作为感知输出或Teacher标签。模型预测未配对端点、连接概率和不确定性；在线图再根据机器人真实穿越trace确定端口的`unexplored/attempted/verified`状态。该状态不得回流为感知训练标签。

### 4.3 训练顺序

1. 单基元参数与可微表面重建预训练；
2. 多基元集合分解和最小描述长度约束；
3. 端口连接、非连接重叠及组合关系训练；
4. 五帧跨帧对应、SE(3)等变和遮挡拒绝训练；
5. 全部损失联合微调，seeds固定为`0/1/2`。

总损失只包含有独立证据的六项：基元集合参数、LiDAR表面重建、射线自由空间、端口关系、跨帧等变和不确定性校准。禁止在失败后增加事件分类loss或手写拓扑修正规则。

## 5. 从关系图谱到在线拓扑图

- 基元是局部几何实体，端口连接是结构关系，跨帧对应形成持久实体。
- 多端口连接簇自然表达分支；稳定未配对端点形成provisional port，在线执行trace再判断是否待探索；曲率和截面变化保存在edge几何，不额外制造事件类别。
- 学习输出可创建`provisional node/port`；只有机器人实际穿越才创建`verified edge`。
- place descriptor、几何一致性、关系结构和执行trace共同关联节点；关联不确定时保留并行provisional节点，禁止强制loop merge。
- M-TARE定位、局部规划、避障、控制和传感器合同不变，只替换全局表示、目标选择与多机器人图增量共享。

唯一保留的系统硬规则是：因果输入、真实穿越验证edge、拒绝不确定合并、严格数据隔离和安全执行合同。

## 6. 对照、消融与科学门

### 6.1 固定对照

1. 原始M-TARE；
2. Cano-like出口模型+规则图；
3. 非学习鲁棒超椭圆拟合+同一图接口；
4. 完整主方法；
5. GT-TNG oracle上界。

旧Composer、集合出口、稀疏relation与ERCSS仅作为失败机制、表示消融或历史基线，不继续扩展。

### 6.2 必要消融

- 单帧替代五帧；
- 去掉构造程序参数监督，仅保留重建；
- 去掉扫描重建，仅做参数监督；
- 去掉relation head，改用相同几何的最近邻关联；
- 去掉时序等变；
- 去掉不确定性拒绝；
- 去掉edge几何属性。

### 6.3 离线资格门

C07选择域必须同时满足：

- 表面/ray重建误差相对非学习基元拟合下降至少10%；
- 轴线、宽高、形状指数、坡度和曲率的有效目标MAE总体下降至少10%；
- 端口连接F1相对非学习关联提高至少5个百分点；
- 离线topology node/edge F1相对Cano-like规则图提高至少5个百分点；
- 关联precision不低于0.98，false loop merge不超过1%；
- 至少2/3 seeds同方向通过，ensemble不得掩盖单seed系统性失败。

全部checkpoint、阈值和图参数冻结后，C08只执行一次零适配迁移。任一主门失败即停止主方法，不通过规划器调参、阈值搜索、删除帧或增加规则补偿。

## 7. 实施阶段与唯一下一步

### P0：方法与数据合同

冻结相关工作贡献矩阵、`PrimitiveConstructionGraph` schema、三种几何实现定义、可见性裁剪、父世界split和资源上界。只做单元/合成证明和只读inventory，不训练。

**当前唯一下一步**：实现并审计一个零训练construction-supervision feasibility pilot，范围只含解析微场景和C01的少量只读开发世界；回答现有生成资产能否无歧义导出基元、变换、组合、surface provenance和可见端口。不得读取C07--C10或启动模型训练。

### P1：数据生成与质量证明

P0 PASS后才生成新几何实现和LiDAR，冻结Data Card、manifest、hash、传感器合同、碰撞安全、形状分布和泄漏审计。

### P2：模型readiness与三seed训练

先完成解析ellipse/rounded-rectangle/taper/curve/T/Y/X/stacked-overlap tests和真实batch有限梯度，再创建一个正式三seedrun。C01--C06提供梯度、C07选择、C08一次迁移。

### P3：离线结构语义图谱

感知门PASS后才运行因果图回放，保存逐帧基元、关系矩阵、节点、端口、verified edges、错误回环和完整图指标。

### P4：严格测试与真实数据

所有方法冻结后打开C10。公开SubT-MRS等真实地下数据先按完整site做许可、sensor、pose和污染审计；至少一个tunnel site和一个cave site零微调评价。真实离线数据只证明感知和图泛化，不冒充实地闭环探索。

### P5：单/多机器人闭环

- 单机器人：5方法×10个strict worlds×5 paired seeds=`250` cases。
- 多机器人：3个主要方法×2/3/4机器人×10 worlds×5 paired seeds=`450` cases。
- 报告coverage-time、完成率、路程、重复探索、错误回环、通信字节、规划延迟、卡死和失败原因。

### P6：论文交付

完成方法图、数据图、基元重建、关系图谱、拓扑对照、coverage-time曲线、消融表、失败案例、真实结果、复现包、补充材料和投稿PDF。所有论文数字必须绑定不可覆盖run和SHA-256证据。

## 8. 清理与停止政策

- 保留程序化world/TNG/spline/mesh/LiDAR、最终Teacher、三个seed、核心基线、必要消融、失败归因、论文图和复现材料。
- 删除前先生成逐目录清单，记录大小、状态、替代证据和hash；禁止通配符删除results根目录。
- ERCSS V1R3治理阻塞被本方法决策取代，不再执行；已有V1/V1R/V1R2仍保存为系统失败记录。
- 如果P0证明生成资产无法提供可靠构造监督，先修数据生成合同；如果P2学习几何不能改善P3图指标，停止投稿主张并重新评估研究问题，禁止退回出口方向+规则图冒充新方法。
