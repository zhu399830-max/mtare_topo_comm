# 局部结构监督 V2：教模型组合通道，不教地点编号

日期：2026-09-05。状态：`DRAFT_FOR_BOUNDED_TEACHER_PILOT`。本文只做源码与既有证据复核；没有读取新数据、物化教师、训练或修改旧实验。

执行依据为 [PLAN](PLAN.md) 最新并行覆盖及 [并行收敛 V2](GSE_GRAPH_CONVERGENCE_PARALLEL_V2.md)。仍在 Phase 3：**几何组合是否帮助识别局部结构及端口归属？** 允许与 GPU 几何对照并行准备结构实验，不要求所有几何误差先达标。

## 1. 结论和唯一推荐

现有数据已经保留了足够丰富的构造记录、可见片段和原始射线来源，**不需要重新生成地图或全量数据**。但现有 11 个教师字段没有直接给出“当前结构区域”和“局部通道口”标签，不能用一次字段改名补齐。

推荐下一项仅为：**在原 180 个 C01 观察、900 个原始帧上，完成“预测中心 query 定义区域参照”的全候选监督 pilot；用同一个观察中的所有有证据结构作集合目标，而不是让旧采样节点决定唯一答案。** 输出标签、证据与 UNKNOWN，随后直接做 GT／预测基元的同 query 组合头容量对照。这是结构训练的前置标签工作，不再追加一轮泛化诊断，也不等待几何模型完全准确。

先完成“候选区域／门面定义”的合成合同，再冻结这个 pilot 的精确 Data Card/spec。不能先读取实际射线，再根据支持率选择门面位置。若当前定义无法产生足够完整事件，报告 `TEACHER_INSUFFICIENT`，不是模型失败，更不是全 UNKNOWN 的 PASS。

本次只创建本文。未来 pilot 的新文件和 run 须单独登记；本文不是数据读取或训练授权。

## 2. 原 180 个观察是什么，不是什么

复用既有密封清单，不重新按分数、支持量或类别挑样本：C01 的 S01–S10、`c1_mixed`、每父地图 18 个五帧观察，共 180 观察、900 唯一源帧、1,452 可见片段、100 个旧评分节点。三种数量不能互相替代；新增有效事件／端口数量目前未知。

既有 [局部教师对照](GSE_LOCAL_TEACHER_AUDIT_V1.md) 确认：

- 1,452 个可见片段中，382 个属于旧清单指定的评分节点，1,070 个属于其他结构。**这只是事后归因，不是训练输入筛选器，也不能直接生成当前端口二值标签。**
- 全部 180 观察都含其他结构片段；178 观察有角投影重叠。角投影重叠不等于通道物理相交。
- 40 个旧分叉观察均包含全部 incident 基元，但只有 6 个满足全部旧物理端点支持。这不证明另外 34 个分叉看不见；也不证明 40 个完整事件都已有合格监督。

原清单的 20／120／38／2 是构造 degree 分布，不是已通过可观测性验证的 terminal／corridor／junction 分类人口。C01 tiny 结果只证明接口容量；不能称独立泛化。

## 3. 三种“端”必须分开

| 名称 | 精确定义 | 可否直接当局部端口 |
|---|---|---|
| 构造端点 | 一个 physical graph edge 对应的扫掠基元的两端；真实 composition 记录在这些端之间建立 incidence | 不可。真实接头可能在远处、被遮挡，或者多个端盖落在同一连接区域 |
| 可见裁剪端 | 五帧回波归属到基元后，投影到采样轴线的最小／最大支持位置；中点是支持弧长中间的采样轴点 | 不可。裁剪端会随视点变化，普通走廊也有两个裁剪端 |
| 局部通道口 | 指定局部结构区域与一条通道之间、具有独立几何边界及观测证据的开口 | 是目标概念，但必须先定义区域和边界；不等于数组的 endpoint 维 |

来源：[可见窗口教师](../src/mtare_topo/data/primitive_relation_dataset.py:120)、[缓存支持恢复](../src/mtare_topo/data/primitive_frame_support.py:94)、[构造监督](../src/mtare_topo/teacher/primitive_construction_supervisor.py:180)。同 tunnel ID 下的不同 graph edge 必须保留独立基元；不能按 tunnel 合并掉相反方向的 incidence。

## 4. 逐字段依赖与可推断边界

| 现有字段／文件 | 实际内容 | 允许生成的事实 | 不能直接推断 |
|---|---|---|---|
| `frame_row`、`source_global_sequence_index` | 五帧引用及源序列身份 | 精确因果人口、行对齐、帧去重 | 事件类别、空间独立性、已行驶距离 |
| `primitive_index`、`primitive_mask` | 五帧至少有一个回波来源的基元及 padding | 全部可见片段集合；teacher-only 槽位映射 | 当前节点的成员集合、完整可见端口 |
| `support_ray_count` | 五帧中属于该基元的回波计数；多源回波保留多重归属 | 有表面支持，支持量统计 | 支撑跨过接头、可通宽度、封闭端帽 |
| `temporal_visibility` | 每帧是否有该基元来源回波 | 五帧持续／间歇支持；缺席置未知 | 同一开口持续被看见、帧间连接关系 |
| `axis_control_current_sensor_m` | 共同当前传感器系中的三个可见轴线采样点 | 局部相对方向、高差、布局及几何回归目标 | 完整 spline、物理端点位置、端口中心 |
| `endpoint_half_axes_m`、`endpoint_shape_exponent` | 在可见支持区间两端的截面参数 | 可见局部宽高和形状监督 | 原构造端盖或局部开口的真实多边形 |
| `endpoint_neighbor` | 可见基元之间、按构造端点编码的真实附件；每端最多三个邻端 | 全可见集合的构造关系参考 | 连接处在本观察中可见；当前区域身份 |
| `disconnected_overlap_packed` | 无构造附件的基元，在至少一帧共享某个 720 方位列 | 角向 alias 分组、困难案例统计 | 同高度／同距离／同扫描线、物理重叠或应合并 |
| 旧 `endpoint_observed` | 原物理端点附近的表面回波支持 | 原端点任务的可观测掩码 | 新通道口或事件的必要条件 |
| `primitive_membership_code` + codebook | 每条 first-return 的完整来源基元集合，0 为无合格命中 | 点级来源、多源歧义、可见支持重建；仅教师使用 | 射线沿途所有穿过的基元、唯一表面／端帽类别 |
| realized construction | 完整基元几何、composition 的全部 `member_endpoints` | teacher-only 物理 incidence、几何候选及终端构造事实 | 当前观测已支持这些事实；可部署候选预筛 |

`endpoint_neighbor` 的槽号编码和 `disconnected_overlap_packed` 的 little-endian 展开见 [压缩存储](../src/mtare_topo/data/primitive_relation_storage.py:30)。角重叠生成按 `ray_index % 720` 收集方位，见 [关系目标](../src/mtare_topo/data/primitive_relation_targets.py:50)；不能把其负例叫“非相交通道”。

现成自洽检查可复用 [audit_support／decode_and_validate_relations](../src/mtare_topo/data/gse_local_teacher_audit.py:36)。它们验证存储与构造事实，不返回事件标签。

## 5. 现有射线组件可以直接用到哪一步

[gse_portal_ray_evidence](../src/mtare_topo/teacher/gse_portal_ray_evidence.py:154) 检查：有限 first-return 射线是否在首次回波之前穿过给定截面的严格内部。正反两个方向都可提供证据；无回波不作为自由空间；没有独占证据就是 UNKNOWN，不是负例。

可直接复用的输出是：`crossing_ray_count`、`exclusive_ray_count`、`witnessed`、`witness_ray_index`、inward／outward counts 和 ambiguous 数量。它们提供可追溯的正证据，不能证明机器人通过、完整事件或所有出口。

当前 `construction_cap_candidates` 有三个明确限制：

1. 它生成**全部给定可见基元的物理端盖**，没有生成真正的 junction throat／局部门面。最多 32 基元、64 候选。全部候选必须一起竞争，不允许先按目标 node ID 留下几个。
2. 同一射线穿过串联或重合候选时，当前算法不会给任何候选独占见证。这是保守的证据算法限制，不能改称物理连接不存在；不能为提高支持率悄悄只比较 incident 截面。
3. 输入截面必须与原 mesh 一致。P1 源 mesh 是轴向 0.05 m、角向 64 段，解析 field 0.025 m 不是同一采样。必须检查真实多边形边界，不能用更大的理想椭圆“补开口”。

此外，要保留源 float32 range 的完整 ULP，再减固定 caster／测量误差界；**新增 `range_error_bound_m` 必须从源算法数值合同确定，不能为了让射线通过而设为 0 或调节。** 组件不会自动给出真实测量误差。

## 6. V2 目标合同：参照明确、证据明确、未知明确

### 6.1 唯一推荐：区域由预测中心 query 指定，不由旧节点或机器人最近节点指定

部署输入是原始因果观测派生的全部预测基元，最多 32 个基元／64 个方向 token，保留共同当前传感器坐标。输入中没有 GT node、edge、traversal、primitive ID、旧 incident mask 或教师选好的 region。

不再将三分类定义成“机器人此刻所在的唯一区域”。机器人可能在走廊里同时看到两个交叉口，或者正位于交界面；强求一个 `current_region` 会把选目标与识别结构混在一起。本草稿明确采用**局部结构实例集合**：每个预测中心 query 回答“这个中心对应什么结构，哪些可见通道属于它”。字段名用 `region_query`，不继续模糊地叫 current_region。

最小新接口如下，不增加新 LiDAR 骨干：

```text
全部 32 预测基元 → 64 个方向 token（端只是方向锚点，不冒充真实端口）
  → 共享小型 proposal 分支：每个方向 token 提议一个局部结构中心
  → 每个中心 query 读取全部 token 的共同坐标组合
  → center_xyz、event、region_presence、该 query 对全部 token 的 membership
```

最多 64 个 proposal 来自已有 64 个方向锚点，不按真实节点数量决定 query 数。不先用 GT 判断哪个端靠近节点；所有候选都参与中心与成员预测。中心分支只是当前观测内三维位置的估计，不输出 place ID，不查全局地图。

教师侧为同一观察保留**全部**有可见基元成员的 composition，不只取旧 `target_node`。实际字段 `composition_operations[].anchor_xyz_m` 可变换为中心监督目标；若缺失，只允许核对所有成员的 `composition_anchor_xyz_m` 是否给出同一锚点，不能拿裁剪端均值代替。构造身份仅用于组合关系和独立评分。是否能监督该中心／事件，仍由原观察的自由空间／开口证据决定；“构造文件里有节点”不是 center_valid。

预测和目标用集合匹配在 **loss／评分端** 对齐。GT 中心不能作为查询坐标喂给主方法，也不能 teacher-force 到 forward 后再声称学会区域定位。无监督依据的目标中心不罚预测偏差；未匹配预测是否可当假阳性须有局部标签完整性证据，不能把未知区域当背景。

这比原单个全局 `event_head` 多一个中心 proposal／query 条件接口，必须明确实施并合成验证，不能仅更换旧 loss 标签。若先用真实中心查询测试组合容量，只能叫 **GT-query oracle 上界**，必须单列，不算 GT几何／预测几何主对照的部署输入。

### 6.2 教师输出、歧义与稳定建图

```text
region_targets[]:
  center_current_sensor_m, center_valid
  event_target, event_valid                  每实例 corridor / junction / terminal
  directional_member_target, member_valid   哪些片段方向属于此实例
  local_port_geometry, port_geometry_valid  真开口位置与形状；与裁剪端分开
  evidence_refs, unknown_reasons
observation_label_completeness              是否允许给额外 proposal 负例
```

多个相互分离且可观测的区域不是 UNKNOWN，而是多个目标；部署输出亦不应只保留最近一个。精确对称时，任意 query 槽号交换不影响集合目标。若两个区域的中心或端口对应无法由当前证据区分，标记的是**对应关系未知**，不是把所有已观察通道或整个观察删掉。

与图接口的衔接：相对运动将各次预测中心变换到共同部署坐标，稳定中心和端口布局可以产生本地节点，不要求第二次到访。不能唯一匹配旧全局节点时保留 provisional node／候选关联；局部结构存在和全局 loop identity 分开。不因局部多实例而拒绝所有节点，也不因相似性强而强行合并。

`active_region_for_navigation` 是图层根据部署位姿、已验证连接及执行状态选择的目标，不是教师喂给模型的 region。普通走廊的 degree-2 切分点不自动产生语义节点；必要 metric anchor 仍另计。此合同不把构造文件任意切段包装为新的 junction。

**剩余真正缺口是局部开口证据适配，不是区域参照继续未定义。** 源物理端盖可作为候选起点，但必须证明其与局部开口的关系，零厚度接头、重合门面、普通走廊切段及分层交叉要有合成反例。没有合格开口／中心证据时保留 unknown reason，不拿旧采样身份补齐。

### 6.3 可立即编码的标签决策表

| 所需标签 | 正标签要求 | 负标签要求／UNKNOWN |
|---|---|---|
| 表面片段可见 | 已有 mask／count／temporal 一致 | 无支持只说明片段未观察；不说明通道不存在 |
| 构造附件参考 | 已有全可见关系与 construction 精确一致 | 只用于附件参考，不冒充可观测连接负例 |
| 候选截面穿越见证 | 有合格 exclusive ray 穿过严格内部 | 无见证、串联竞争、擦边或 on-plane 均 UNKNOWN |
| 区域 query 的端口归属 | 候选对应所匹配局部区域的边界；独立开口有因果证据；构造对应一致 | 只有候选开口及其所属另一局部区域均已被明确识别，才可对本区域给 0。仅“不在旧incident集合”不足 |
| junction | 区域中心／范围有依据；至少三个**不同的局部通道口**被证实且属于同一局部自由区域 | 两个可见分支不能当 corridor 的充分证据；多个物理端盖不能重复计口 |
| corridor | 区域参照有效；两端通道口被证实，区域边界检查足以排除遗漏分支 | 仅完整地图 degree=2，或只看见两个方向，均不够；任意degree-2构造分段不当语义节点 |
| terminal | 区域参照有效；进入通道和封闭端部有合格局部证据，且不是遮挡形成的假尽头 | degree=1 不足；没有穿越射线不是封闭证据。现有 codebook 不保存端帽面类别，不能直接从 membership 判封闭 |

这些条件是保守标签语义，不是已完成的全可见性算法。特别是 corridor／terminal 的完整边界证据尚缺实现；不能靠所有未定样本不进 loss 而宣称完整教师完成。

端口**成员**和端口**位置**分开监督：某条通道确定属于当前结构，也不意味着裁剪端正好就是开口中心。没有位置证据时仅训练成员标签，不强制回归错误端点。

### 6.4 一个可以直接失败的 pilot，不再以“继续检查”为终点

固定执行顺序如下，均在原 180 行内：

1. 根据已有 primitive_index／mask 收集每观察的全部可见 operands；读取所有含这些成员的 composition，**包括旧目标之外的组**。该列表只进 teacher-builder，不进 model.forward。
2. 把候选构造中心及所有物理 cap 变换到当前传感器系；按真实源 mesh 参数生成候选截面。保留 `candidate_kind=physical_cap`，不预先赋值 `local_port`。
3. 以原五帧有限射线统一计算全部候选的正证据。保存每条见证的 frame／ray、交点和首次回波距离，保留串联、重合、遮挡与边界原因；不按GT组独立运行竞争以“补出”独占证据。
4. 独立门面适配器验证候选是否代表局部区域的不同出口、是否只是同一开口的重复端盖。只有这项通过，才把构造成员变成该 region 的可监督端口。中心坐标标签同时附其可观察结构支持，不因构造坐标有限就设 valid。
5. 输出全部 region 候选／有效目标／UNKNOWN 人口，并叠加原 first-return 与候选门面，检查多region及原点在走廊时仍输出目标集合，不再强选一个“当前GT节点”。

**决定性软件反例**：同一原点看到近、远两个分叉且均有充分证据时应输出两个实例；只改旧采样 target_node 不得改变任何目标集合；互换重复结构的身份编号不改几何标签；分层但角向重叠不能变成同一实例；两个物理cap重合只允许一个开口或UNKNOWN，不能制造junction。

如果第4步只有“它在同一个GT composition里”而没有实际开口依据，pilot 在该步明确失败，不能直接进入训练。若 native/P1 源端盖互遮导致没有合格证据，记录原图和射线为原因：**不换 mesh、不重渲染、不移动门面到有利位置、不扩大误差容差**。另行改进的是可观测证据算法或标签范围，不是传感器世界。

具体待实现的窄接口是 `resolve_local_opening(candidate_sections, finite_ray_witnesses, source_geometry_contract)`：输出开口等价／独立／未知及证据；它不读取 target_node，也不返回真实全局identity。射线组件已存在，缺的是这个从cap到局部开口的适配，以及新的中心query监督接口。未知人口的真实数量必须等这一固定pilot测得；目前不虚构三类事件已满足180观察训练门。

## 7. 最小补充读取清单：只原 180／900，不开全量

本节是未来 pilot 的精确字段草稿；当前审查没有打开这些源数组。完整任务名、180 行及 source/frame 索引直接绑定既有 C01 180 清单和其 SHA-256，不通过遍历目录发现地图。不得替换为新 C02 人口。

| 源 | 逻辑读取范围 | 字段／元数据 |
|---|---|---|
| P1b teacher | 原 180 个序列行 | 上表 11 字段；另 `relative_translation_current_sensor_m`、`relative_yaw_current_sensor_deg` 用于五帧共系与交叉核对 |
| P1a sensor | 原清单去重的 900 帧 | `range_m` float32、`valid_mask` uint8、`primitive_membership_code` uint16；保留全部 16×720 方向，不采样挑有利射线 |
| P1a pose，仅教师 | 同 900 帧 | `sensor_xyz_m`、`yaw_deg`，将构造候选和射线放入同一当前传感器系；绝对 pose 不进 student |
| codebook | 仅 10 个精确 C01 mixed 文件 | `primitive_ids`、`source_sets`、schema／parent／realization；多源集合完整保留 |
| realized construction | 仅相同 10 个文件 | `realized_primitives`、`base_construction.composition_operations` 及源 mesh 参数／版本；无全世界 SDF、无 mesh 重导出 |
| provenance | 精确源 seals／相关 task 的 Zarr 头 | `.zgroup`、`.zattrs`、上述数组 `.zarray` 及实际访问 chunks，全部与冻结输入 hash 核对 |

900×16×720 = 10,368,000 条既有射线记录；逻辑数不能说成新采集扫描。纯 range／valid／membership 三字段的未压缩量约 72.6 MB，可逐 task、逐帧／射线块处理，不需要全世界稠密场。Zarr chunk 可能包含邻接未选帧，必须报告实际解码块，不将其用于统计或标签；源 seal 仅索引显式 task/字段，不遍历 C07–C10。

不读取 `axis_xyz_m`、route arc、tangent、traversal identity 作为学生提示；不读取 checkpoint，不重跑 raycaster，不新建完整教师数据集。若仅做 aperture 正证据，membership_code 不是几何穿越判定的必要输入；本 pilot 若读它，用途限定为现有表面归属／多源一致性及可见候选校验，不能称沿射线全路径身份。

## 8. GT／预测基元的最小受控组合实验

标签有效后马上做小实验，不设置“轴线 MAE 必须先小于某值”的新门槛。**标签有效性**和**预测几何质量**是不同问题。

1. GT 几何容量分支：输入原观察中全部可见 GT 裁剪片段，随机置换槽位，去除全部构造身份。不能只给当前区域成员。教师 mask 决定 GT 上界的存在人口，必须明确这是理想检测参考。
2. 预测几何分支：输入全部 32 预测，不用 teacher mask、GT 排名或 GT 匹配结果选择有效候选。额外候选会影响组合结果，不能在前向前删掉。只在 loss／评分端做几何匹配及反向对应，转移 teacher-only 监督掩码。
3. 两分支用同一组合头架构、初始化／训练预算和同一标签集合；另报告 GT→GT、GT头→预测几何的冻结替换，分离几何误差与重训练适应。它们不是独立泛化分数。
4. 最小直接消融是去掉相对位置／夹角／高差组合输入，保持 unary 几何及训练预算。若无组合同样好，就停止扩大组合，不靠更长训练解释贡献。

现有 [GeometricCompositionHead](../src/mtare_topo/representation/gse_composition.py:85) 可以复用 shared-frame 特征、轻量消息聚合和 masked losses。**但它的单个 event 输出和每 token 一个 membership logit 没有显式 region reference。** V2 推荐薄改为上节共享 query 条件头；GT几何和预测几何都必须自行预测中心，不能一边给 GT-query、一边要求自行定位后直接比较。本文明确提出这一接口改动，尚未实施或冻结训练。

新 raw axis 小头只输出轴线：它没有同步重估宽高／存在置信度。第一轮 GT／预测共同使用的字段应取确实具有统一定义的轴线、方向及共同布局；若拼接旧截面／存在概率，必须另列“混合属性”因素，不能叫完整 raw 几何结果。方向退化保留未知，不造切向量。

原 180 只有 mixed 实现；**目前没有精确配对的另一几何实现人口**。因此这一 pilot 无法给出“去配对一致性”实验证据。不能把同一帧旋转、重复两次或相邻帧当作跨几何实现正对。先证明局部监督可学，再另登记同父地图的真实配对实现与共同可观测部分；这不阻塞当前 GT／预测／无组合容量检查。

## 9. 验收与交付，不让 UNKNOWN 制造假成功

软件须先通过：所有候选参与、ID 改名不改 student、slot 置换只置换端口输出、共同 yaw 后标签等变、非 incident stacked 通道不合并、T／Y／直廊／封闭端帽／遮挡假尽头、重合与串联门面、源 float32 擦边、原点落边界、未来帧拒绝。区域语义和门面定义尚未通过时不能开始真实标签物化。

pilot 必须交付全部 180 行的标签、掩码、原因、射线索引与候选几何；按父地图和独立结构报告 known／unknown、正／负端口、完整事件以及缺失类别。保留所有原观察，未知只是标签有效性，不是样本删除。

训练容量线沿用已批准方案的事件 macro-F1≥0.95、端口 F1≥0.90；同时必须列出三类真实有效样本数、各父地图覆盖及 precision／recall。以下情况不得 PASS：

- 某个必需类别没有有效标签，却跳过该类计算 macro-F1；
- 端口只有正例或只保留极少容易样本，却称完整归属识别成功；
- 全部 UNKNOWN、零有效监督或零预测，只凭 loss=0 通过；
- 只有少量独立节点重复观察，将观察数当独立事件数；
- 用 GT 匹配后的 1,452 个槽参与前向，隐藏 32×180 候选中的剩余项。

如有效人口不足，状态为 `TEACHER_INSUFFICIENT_FOR_EVENT_PORT_CAPACITY`，列出缺的是区域定义、门面证据、终端证据或类别覆盖。只暂停受影响的结构训练；GPU 几何受控对照和图软件仍可继续。禁止据此自动扩到 C07–C10、改半径／阈值、选更容易的节点或启动三种子。

最终需要的不是更漂亮的 degree 准确率，而是：模型在未经教师预筛的杂乱可见通道中，找出同一局部结构的成员，明确哪些方向是通道口，并把不确定判断保留给后续建图验证。
