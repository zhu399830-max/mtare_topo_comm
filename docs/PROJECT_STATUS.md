# PROJECT STATUS

## 2026-09-05：已确认位置与语义匹配捷径，不再凭低loss判断学会

同快照CPU归因3.477秒完成，50seal/全部30图保留。原分数精确复现；联合匹配可挑离中心59.23米但类别正确的候选，位置/属性未绑定是实际问题。预测成员联合F1仍0.158，尚有额外漏失。已公开选择单变量几何绑定修正，正实现独立loss/训练接线与新卡；不同时改数据、权重、阈值或扩种子，不把归因高分替代原模型成绩。

## 2026-09-05：并行完成多实例预测载体

GPU100.44秒三组结果已封存后，新增无决策候选转换层，全部64候选、坐标和UNKNOWN保留，77项相关测试通过。该软件不筛候选、不填宽高、不提交图，不能称拓扑图已建好。实际输入生产绑定与节点/端口决策仍缺；当前研究仍需用已缓存最终预测查明中心与成员对应问题，不扩大训练。

## 2026-09-05：真实训练有中心定位收益，但仍不能可靠识别结构成员

三组共900次更新、1080次评估已在100.44秒完成，不是在后台等待长训练。GT/预测/去关系中心误差约0.077/2.329/2.521米，成员F1约0.535/0.155/0；预测成员召回仅9.34%，三组事件两类macro-F1约0.491/0.491/0.496。关系帮助了10/10父地图的中心定位，但理想几何下成员仍大量漏失、交汇也未学好，因此不宣布方法或科学门通过。

97项seal复核、213项软件测试、全部60张SVG/2160面板与三份最终权重已保留。唯一run `gate3_20260905_gse_partial_structure_training_v1_seed0`，seal `07f212c90a25785b55a173f0a1f66451b3bb47b9ac6b9b2ded71e78d949d5b1e`。当前停止加轮数/三seed/闭环，先准备只读现有缓存及训练对应诊断；“训练与评分选了不同候选”尚未证实。纯候选图适配软件可并行，不拿GT筛选或补宽高。旧冻结文件和run不改，完整论文目标仍active。下文训练准备状态为历史；详见`GSE_PARTIAL_STRUCTURE_TRAINING_RESULT_V1.md`。

## 2026-09-05：训练输入已实际封存，开始固定预算三分支小头实验准备

已完成缓存导出，不再停在“还需统计有效标签”：原180观察/900帧/1452片段，GT与预测均有2152正成员、13489负成员，3正/21负无法唯一对应而保留未知；1206中心、886事件不变，终点仍0。GT/预测数值有效方向候选2892/11520。唯一run `gate3_20260905_gse_partial_structure_export_v1_seed0`耗时0.3677379169967s，17项seal复核，SHA256 `db28c9df03373dfa17a3222688f85d95679e6a0cbdabe339fdf74f15f35b514e`。下文“export待执行”为历史；旧run和图不修改。

接下来只读取四份封存导出文件，独立卡/spec后优先5090训练GT轴、预测轴、预测去显式关系三个小头。seed0/hidden64/Adam0.001/batch18，同初始化与顺序，各300步共900；初始/最终各180共1080小头评估，0骨干/新扫描。上限1800s、host/GPU各4GiB、结果0.5GB。独立评分以中心几何匹配，不复用loss matches；成员阈值0.5不校准，不选best。当前没有新增训练准确率，完整三类、检测、泛化、建图或科学PASS均未宣称。

## 2026-09-05：新支持标签完成，小头训练的软件已接通

已完成真实180观察支持标签和全部论文可用原图保留。缓存身份桥、loss方向对应及三分支小头训练核心通过合并292项回归，尚未读真实缓存启动新训练。下一步是一次精确缓存目标转换/有效分母统计，随后5090只训练小头，不再重复扫描或骨干。完整三类和完整建图未PASS；详见`GSE_PARTIAL_STRUCTURE_HEAD_SOFTWARE_V1.md`。

## 2026-09-05：局部结构训练答案已实际生成，但完整事件人口不足

11.306s完成原180/900支持标签：1206中心观测、304独立中心、2155正成员；只有6个交汇实例形成14次完整事件，终点尚无封闭表面标签。因此接下来只做中心/partial成员接口学习，不声称完整三分类可验收。134项回归和35seal复核，全部180双视图保留。V1失败源代码在256a059、修正源在b15f01b，不改旧run。

两条软件支线正在推进缓存精确对应和仅loss侧方向映射；下一GPU项将只训小头，不重跑900帧渲染/骨干。图坐标接口已实现，但完整 learned observation/图接线和闭环仍未完成。详见`GSE_SUPPORTED_CONSTRUCTION_TEACHER_RESULT_V1.md`。

## 2026-09-05：局部支持教师单次执行停止于精度接口，修正原因明确

V1在1.483s停止，13项失败seal复核；0新模型/optimizer/完整标签。旧P1b把相对运动存为float32，新执行器漏掉该转换便与float64计算值逐元素比较。首18观察转换回原存储格式后完全相等；修正必须另立V1R，旧run及版本保留，不放宽容差或数据合同。129项新教师相关软件测试通过，但缺少此非整数反例；先补测试再执行同180/900/1452。完整结构方法与三类验收仍未PASS。

## 2026-09-05：学习与图接口继续补齐，真实结构标签仍是关键路径

新多实例结构小头26,826参数已实现，当前未训练；不再用单一全局事件代表同一观察里的多个交汇区域。独立审查发现并修复3个可复现损失/数值问题，另拒绝负成员自动推导存在。CPU配准已有真实Open3D合成算法验证，并绑定扫描时刻和部署位姿，35项专用环境测试通过。截面正证据核只能提供局部射线事实，不能直接给完整结构标签。当前0新数据/0新模型训练/0实际图回放，GPU无后台长训练；下一主线是有效结构标签及原180/900pilot，不再用重复几何训练替代。论文目标保持active。

## 2026-09-05：GPU优先任务已跑完，局部结构训练仍待有效标签

正式同预算对照30.38秒完成，1080步、180观察/900帧；原坐标/均值坐标误差1.780/2.457m，横向则2.336/1.959m，不能宣布全面胜出。39项seal、全部11图和两分支权重保留。CPU端口/6DoF接口70项测试通过，含坐标输入共82项主代理复核。教师草稿已完成，发现构造端点不等于真实可见开口；暂停受影响的事件训练，不把未知标签强行补为负例。下一步局部区域/开口软件合同及原180/900pilot准备，与图接线并行。当前GPU实验进程已经结束，无结构训练在后台运行；完整论文目标保持active。

## 2026-09-05：C02开发对照完成，转为主线并行推进

真实冻结推理20.07秒完成：180观察、900帧，原始点头坐标误差2.03m对旧3.92m，10/10地图改善；横向仍2.64m，尚不等于结构和建图成功。全部11图、28项seal保留并复核，309回归通过。详见`GSE_HEAD_DEVELOPMENT_INFERENCE_V1.md`。

用户批准加速并优先5090。新持续目标保留完整论文/严格未见/真实迁移/单机与2/3/4机器人范围。GPU同预算几何对照准备、CPU标签合同、CPU图软件同时推进；标签通过后直接验证组合，不等待前端完美。仍Phase3，不启动未经资格的真实图/闭环。

## 2026-09-05：小头跨父地图检查已有精确人口

C02固定选样已完成，180观察、900唯一帧和1489片段，来自10父地图/180条有向穿越。只读索引与mask，0模型/几何/扫描，13项seal通过。接下来独立登记冻结推理，不重训。C02未用于新小头拟合但旧骨干已见，不能称为整模型严格未见；当前无新增模型精度或图结果。

## 2026-09-05：已排除完全固定候选库解释，下一步检查小头跨地图表现

用已保存输出做固定错配对照，2.966秒完成。原始点头正确配对1.780米，错配5.220米，10张父地图都成立；方向与横向偏差也明显依赖对应。这不是完全固定输出，但仍不能排除拟合记忆或运动相关性，也不是检测/建图成功。横向偏差仍2.336米，多余候选未评分，偏移分支停止扩展。

22项seal已复核，完整图片和结果保存，446项回归通过。对象/列表读取器错误导致的V1失败保留，V1R仅修接口，0重训。下一步只准备C02小头开发检查的精确元数据/选样卡；旧骨干训练已见C02，结果不得叫整模型严格未见。C07--C10与闭环不动。详见`GSE_AXIS_OBSERVATION_DEPENDENCE_V1.md`。

## 2026-09-05：原始点小训练取得拟合改善，但偏移分支未通过

180观察、两种读出各540步，执行15.977s。相同几何评分下无偏移坐标误差1.780m，偏移2.826m，旧模型3.612m；偏移版全部10父地图不如无偏移，停止扩大该分支训练。原始点版仍有4.111m三维距离误差且评分不罚多余候选，尚不能宣布结构识别/建图成功。

原run最终绘图报错，全部权重/预测/日志已保存且不可修改；独立补全在修正目录初始化后17.372s完成，0更新/推理，全部匹配与判定复现，180观察图及22项seal全部通过。423项指定回归通过。当前下一步是已缓存输出的观测依赖与布局对照，不重训、不开放测试地图或闭环。所有论文图片及旧工作保留。详见`GSE_POINT_AXIS_PROBE_V1.md`。

## 2026-09-05：前端不再只有诊断，已有可训练的小型点级读出实现

新增35,460参数的原始点归属/slot条件轴线偏移，与旧骨干通过独立adapter连接。48项正式软件测试全部通过，完整57,600点/32slot前后骨干不变、梯度隔离正常；用时2.120s，RSS约0.95GiB，11项seal已复核。不是重新训练成功或模型准确率提升。

多源回波不能只给唯一轴目标的问题在标签生成前处理为slot条件偏移，不更改现有教师。下一动作是同180观察的几何小训练规格、卡和对照执行器，验证新读出是否真的学到更准确几何；暂不训练尚未验证标签的事件头。Phase 3不变，C07--C10、图和闭环未开放，旧模型和论文图保留。详见`GSE_POINT_AXIS_READOUT_V1.md`。

## 2026-09-05：找到真实前端表达限制，已从猜测转为小范围修正依据

同180观察坐标支持对照完成：旧均值点池不能表达3153/4356个目标轴点，其中2688个可由原始点重构；465个连原始点凸包也在外。说明旧“先平均坐标再加权输出”确实限制几何恢复，同时否定只把坐标数组换成raw就一定能解决问题。

下一步只验证保留原始点区分和表面到轴线估计的最小读出，不推倒骨干、不启动三个种子长训练。该结果不是模型准确率，点归属与完整事件/端口监督仍待验证。运行16.254s、0模型/optimizer/新标签，27项seal复核，381项指定回归通过；旧模型、论文图和失败实验保留。Phase 3未科学PASS，C07--C10与闭环未开放。详见`GSE_COORDINATE_SUPPORT_AUDIT_V1.md`。

## 2026-09-05：确认几何布局也不准，前端薄改先验证具体原因

同180观察分解已完成，不再停留在“误差可能只是长度”判断：横向RMS中位2.263m、无向角中位37.933°；上一轮行/对应关系/分数完全复现。该结果只否决冻结当前几何直接接组合头的做法，不否决学习基元主线。26项seal已复核，11张图及旧论文资产保留。

现有坐标均值池化在合成上下层反例中丢失了可恢复高度的支持；下一步用原180观察对照均值池与原始点池，0模型/optimizer/新标签，明确实际影响后再薄改读出。新增45项测试/指定回归313项通过。Phase 3不变，模型成功/图/闭环未宣布PASS；详见`GSE_AXIS_ERROR_DECOMPOSITION_V1.md`。

## 2026-09-05：完成真实几何对照，正在定位前端与监督接口的误差来源

同180观察的现有教师支持/连接核对通过，预测与教师全部1452片段的裁剪轴点误差均值10.882m、半轴误差0.539m。不能直接把它当节点定位误差；独立审查未发现单位/端点反转错误，但评分人口与旧存在门不同，轴向截取误差尚未分离。因此暂停受影响的组合头训练假设，下一步仅做同样本零训练的误差分解，不再先跑长训练。

10张完整XY/XZ图、逐样本指标与24项seal已保存复核，0新增推理/标签/权重更新/测试世界访问。Phase 3仍未获科学PASS，旧模型和论文图片不动。详见`GSE_LOCAL_TEACHER_AUDIT_V1.md`。

## 2026-09-05：字段恢复已正式完成，进入真实/预测几何监督衔接

原 180 个 C01 观察的共同坐标轴线、截面、存在性和不确定性已恢复；旧特征与置信度逐元素复现，18 个重复观察的原始字段完全一致。唯一 run 用时 6.137s，error=null；198 次冻结推理、0 更新、0 新标签。23 项 seal 已复核，完整代码回归 255/255 通过。

这是原有模型的正确复用与输入恢复，不是模型重新训练成功或正式建图成功。下一步核对同 180 观察的现有教师几何/组合及歧义，定义可监督局部成员与 UNKNOWN；不直接从 degree 或旧物理端点支持生成事件标签。当前 Phase 3 不变，旧数据、模型与论文图全部保留。

## 2026-09-05：监督证据组件与原始五帧读取接口已落地

新增正射线截面证据和精确范围模型 reader，40 项针对性测试通过。独立审查抓到并修正浮点端盖误判、门面边界不稳定和 mesh 采样适配错误；尚未在真实数据上生成新标签。完整事件/终端/端口监督仍未验证，不能宣布教师通过。冻结 Torch 已实际完成 CUDA 标量检查。

下一动作是同 180 观察共同坐标字段恢复的卡/spec和冻结推理，先与旧缓存复核，不重新训练骨干。原始数据、论文图和旧实验均未删除；当前 Phase 3 不变，无新科学 PASS。详见 `GSE_NODE_OBSERVABILITY_READINESS_V1.md`。

## 2026-09-05：组合输入与图内核已实现；当前修正监督接口，不在长训练

本轮 82/82 针对性测试通过；同 C01 的 180 观察只读清单已完成并密封。40 个分叉观察均含全部 incident 通道几何，但旧物理端点条件仅 6 个完全满足。合成反例证明不能据此判断其余分叉不可见；新输入已保留可见基元而不依赖旧端点分类器。当前障碍是节点级监督有效性与旧缓存字段不足，不是新训练再次失败。下一步只做同样本监督/字段恢复 readiness，数据相关动作另冻结精确卡/spec。

主机显卡查询正常，已纠正旧“CUDA 不可用”的硬件解读。库存 V1 的 JSON 解析故障已独立修复，V1R 数值清单和 14 项 seal 有效；其中把端点支持称作事件必要条件的解释明确撤回，详见 DECISION_LOG。未新增训练、测试世界读取或真实闭环；旧论文图、模型及失败证据均保留。当前 Phase 3 未获科学 PASS。

## 2026-09-05：实现首批验证完成，不再处于等待批准状态

组合接口/结构损失与因果图内核已实现，56/56 合成及旧模块回归通过；唯一软件 run 已完成并 seal，0 数据集帧、0 训练、0 M-TARE。论文方法尚未得到新的精度或探索收益证据。旧 180 缓存缺共同坐标和端口监督；下一步限定相同行的 reader allowlist 与元数据/教师有效性核对。CUDA 不可用但 CPU 软件工作可推进。详见 `GSE_COMPOSITION_IMPLEMENTATION_STATUS_V1.md`。


## 2026-09-05：新方案已批准，首批软件验证正在实施

权威方法：`docs/GSE_GRAPH_COMPOSITION_EXECUTION_PLAN_V1.md`。不再等待方法选择。正在实现共同坐标几何组合接口和首次确认/实际穿越的因果图；本批 0 地图、0 数据集帧、0 训练、0 M-TARE，尚无新的科学性能结论。Git 基线已建立；CPU Torch 可用，CUDA 当前不可用。旧模型、图片、失败结果全部保留。


## 2026-09-04：GSE-Graph V3方法草案已就绪，当前只等方法边界决定

- 草案文件：`docs/GSE_GRAPH_PRIMITIVE_CONDITIONED_HYPOTHESIS_GRAPH_PROPOSAL_V1.md`。它把当前三维基元学习与历史图关联/执行证据整合，但明确不沿用旧生命周期。
- 新方法不再训练“局部几何=全局地点身份”；学习基元组合负责结构节点和端口，图上下文仅验loop merge，真实穿越建边。
- 独立新节点可在稳定事件后立即成为local-confirmed；“暂不合并旧节点”不再使新节点与执行边一起丢失。
- 这针对旧Factorized路线的实证失败：274真节点仅唯一提交192，13真边仅恢复2，而关联本身曾在C09达到balanced P/R=`1.0/0.5077`。
- 当前不得运行新训练/C08/graph/M-TARE。用户确认草案中的方法边界后，第一步仅为分钟级零训练oracle状态机和因子容量证明。

## 2026-09-04：局部place-identity学生已停止；等待冻结图上下文联合关联方法

- 两级快速门均已完成：V1共`12.24s`，V1R1共`8.20s`，都在长训练前否定了当前接口。
- 保留的正结论：几何基元网络有真实几何改善；节点结构类型信号可学；Teacher节点组合描述具有高分辨能力；描述候选+16 m图一致性在C07可得P/R=`1.0/0.933459`。
- 停止的假设：仅靠冻结局部基元输出的小聚合头，可独立学会全局地点身份。V1R1仍无99% precision安全关联，且存在不同节点距离0。
- 推荐方法边界：学习结构几何负责节点生成/端口属性；地点关联使用学习候选与里程计、图邻接、真实穿越证据的联合因子。这需要新方法决策，不属于原读出头的常规修复。
- 当前Phase 3/P2、C08/graph/M-TARE关闭。最新run=`results/gate3_semantics/gate3_20260904_gse_structural_node_tiny_overfit_v1r1_seed0`；seal-list SHA-256=`d19fb26fd7f14dc2c423e4ff89e62f4b7d6dd96640c986acc5895a22a2bbf803`。

## 2026-09-04：快速tiny overfit否定当前多任务读出，正在做最小读出corrective

- 同规模正式run在`12.24 s`内给出科学FAIL，成功避免了数小时/数天的三seed试错。
- 正证据：节点degree accuracy=`99.44%`，底层LiDAR几何基元确实含有结构类型信号；Teacher在同tiny节点上关联P/R=`1.0/0.9625`，结构目标可分。
- 失败证据：学生几何RMSE=`0.14321`，loss下降只有`69.45%`，99% precision下无安全关联。一个44维向量同时承担了不同尺度的几何回归和身份度量，是直接阻塞。
- 当前Phase 3/P2未结束，C08、正式图和M-TARE继续关闭。唯一下一步是冻结骨干与数据，拆分归一化几何头和独立关联头，重做一次同180/100/500的快速门。
- run=`results/gate3_semantics/gate3_20260904_gse_structural_node_tiny_overfit_v1_seed0`；seal-list SHA-256=`57fc01e05cff81644cf0f840844eeed7ef610a4186dff703276537f512c05588`。

## 2026-09-04：节点级表示加图一致性容量PASS；训练前进入tiny overfit

- C07节点描述子提出3013对候选，其中2960真/53假；固定16 m图位置门删除全部53假匹配且不删除真匹配，最终precision/recall=`1.0/0.933459`。
- 每节点最坏1 m定位误差仍0 FP；S01--S10均有真关联。fit图门P/R=`0.998197/0.931416`。
- 这解决的是“表示与安全候选机制是否有信息”，不是学生网络成功；当前仍不允许C08、正式图或M-TARE。
- 当前唯一下一步为家庭平衡小样本tiny overfit，数百步内检查LiDAR/冻结基元输出能否拟合44维节点证据和同节点关系；失败即停止该学生接口，成功才进入单seed短训。
- 权威run seal-list SHA-256=`7ee483fe66954b676bfd4f52b81743d7838ca7b294b3a995c1929322390a3240`。

## 2026-09-04：节点级表示方向成立，但V1R漏检研究门；当前做图一致性快速归因

- 零训练节点证据已把C07从endpoint metric best precision `0.1187--0.1982`推进到precision/recall=`0.982410/0.933459`，2960个真匹配、53个误匹配，十类地图全覆盖；oracle P/R=`1/1`。
- V1证明support ray count不能当地点身份；V1R修正后只影响不确定性，运行约88秒完成，说明分级证伪机制有效。
- V1R程序漏实现预注册false fraction `<=0.01`，实际为`0.017590`；且6条短traversal无五帧窗口，有效fit观察为36,312而非36,318。因此状态为`FORMAL_MACHINE_PASS_RESEARCH_CONTRACT_FAIL`，尚不允许训练、C08或建图。
- 当前唯一动作：密封结果合同重评分和53个FP的图一致性候选归因；若既有16 m空间门、邻接一致性和执行轨迹约束仍不能把误接受压到1%以下，则停止直接节点描述关联并重新设计候选机制。
- V1R seal-list SHA-256=`12d81fe24342c7712d3ee99655f860fb0de45c12adb1f98b57b13cd9994e84a0`。

## 2026-09-04：端点关系度量三seed正式0/3 FAIL；转节点级快速可分性审计

- 唯一正式run已完成`30,699`次更新并完整封存；C07三seedbest F1=`0.252722/0.163434/0.280575`，均高于非学习基线`0.006380`，但best-point precision只有`0.173568/0.118684/0.198213`。
- 三seed在precision`>=0.98`时均无非空真连接，complete-link安全簇也为0；正式决定`STOP_ENDPOINT_RELATION_METRIC_BEFORE_C08_AND_GRAPH`。这条路线只保留为辅助特征、对照和失败分析。
- 当前仍为Phase 3/P2，C08、在线图和M-TARE关闭。唯一下一步是C01--C07零训练`Structural Node Evidence Funnel`：先把基元组合在时间上聚合成节点观测，再判断结构节点产生和重访关联是否存在高精度可分区域。
- 新候选必须依次通过零训练可分性、小样本过拟合、单seed短训、单seed完整C07，才允许三seed正式训练；禁止再次用长三seedrun试探未经确认的假设。
- 权威run=`results/gate3_semantics/gate3_20260904_primitive_endpoint_relation_metric_three_seed_training_v1_seed0`；seal-list SHA-256=`ab4f3ccf737ca7afb57e204002aab13ab9221cbdf339958fb7426fa4520d14d3`。

## 2026-09-04：endpoint relation metric实现与真实batch训练资格PASS

- 新方法已移除任意slot编号和dustbin竞争：每个端点学习连续关系embedding，pair相似度由共享度量产生；同组合拉近、不同组合与叠置隧道拉远。
- 固定batch128证明全部426,818个head参数可训练，2,635,631参数骨干冻结；置换、yaw、重复、精确对称、complete-link确定性及16/4 GiB资源合同全部通过。
- 当前仍为Phase 3/P2。readiness不代表C07精度成功；唯一下一步是三seed正式训练与完整C07门，C08、结构图和M-TARE继续关闭。
- 权威run seal-list SHA-256=`ed55c036f3d0a7271002c15a778e940721085bb0a5ed07b095874c884e14ae87`。

## 2026-09-04：归因确认任意槽关系不可安全泛化；下一候选为端点关系度量

- 冻结C07归因完整系统PASS，精确复现原三seed失败；三个seed均显示dustbin主导（真实连接端点误入dustbin`72.1%--90.5%`）和槽身份混淆（Hungarian对齐准确率`44.2%--48.0%`）。
- 去掉margin和其余置信乘法仍不能达到precision>=0.98；硬槽关系precision仅`17.9%--21.9%`，soft affinity也没有2/3跨seed安全结果。因此不能把失败归咎于单一阈值或校准，当前local-slot架构科学停止。
- 当前仍为Phase 3/P2，C08、在线图和M-TARE关闭。下一步只做新接口readiness：endpoint relation metric直接监督“同一局部组合/不同组合”，不预测任意槽编号；解码用保守拒绝和transitive cluster合同，edge仍只允许真实穿越提交。
- 权威归因run seal-list SHA-256=`d87f601a7cfc2d529df26aba94d14565f41ee24e6160de20a835671fc2e18f69`。

## 2026-09-04：三seed训练系统完成，局部组合槽关系科学FAIL，当前只做C07因子归因

- 正式训练完成`3 × 10,233 = 30,699`次head-only更新，RUN_STATE=`COMPLETED`且`error=null`；三个模型、输入哈希、资源上限和零C08/graph/M-TARE合同全部有效。
- 完整未见C07上，三个seed的attachment best-F1=`0.138322/0.141336/0.137033`，说明比非学习基线有弱排序增益；安全门只通过seed2，且仅`6`个真连接，最终`1/3<2/3`，所以方法尚不能建立可靠拓扑图。
- 失败集中在部署置信度：中位数及90分位endpoint confidence均为0，绝大多数pair因槽不一致或某个乘法因子归零而没有可排序分数。阈值实现已核验为tie-safe，不存在拆分0分tie虚增F1的问题。
- 当前Phase 3/P2唯一允许动作是冻结C07-only零训练归因；它将分解best-slot概率、dustbin/second-slot margin、slot presence、primitive existence、endpoint evidence和entropy，判断是槽关系没学会还是部署解码器把信号压没。结论出来前禁止重训、C08、在线图和M-TARE。
- 正式run seal-list SHA-256=`f393c3c61e22c40546f450bfae9a3b4538b358b1829f3198b73d2db29600c39e`。

## 2026-09-03：局部组合关系模型已具备训练资格，科学性能尚待三seed验证

- 已完成32槽+dustbin关系头、交换不变Teacher匹配、同槽关系解码和歧义拒绝接口。新head为`1,001,507`参数/81张量，只消费冻结几何基元输出；旧backbone的`2,635,631`参数保持冻结。
- 正式真实batch128 readiness通过全部22项检查：81/81梯度有效、backbone无梯度且state hash不变，重复逐位一致，slot permutation/loss/yaw误差均在预注册容差内，safe score精确对称。
- 资源实测支持batch128：CUDA allocated/process约`7.09/13.64 GiB`、host RSS约`1.94 GiB`。C07仅核人口未前向，optimizer/checkpoint/C08/graph/M-TARE均为0。
- V1的CPU-generator/CUDA-output错误作为系统失败保留；V1R只修设备桥并正式PASS。这不能证明连接精度，下一步仍需三seed完整训练和C07门。
- 当前Phase 3/P2，唯一下一步是三seed训练实现与Data Card；只有至少2/3 seed在同一C07人口获得相对非学习F1增益、precision>=0.98且非空真连接，并无槽塌缩/overlap失控，才允许一次C08迁移。
- 权威V1R run seal-list SHA-256=`45488881d3ef76f8d81aef6a54f71c7f116aac8ea9e57fd968412fc2a4ea463a`。

## 2026-09-03：局部组合槽目标成立；当前进入新关系模型训练前实现

- fit/C07共`491,196`条Teacher序列已经完成零训练全量审计。每一条可观测物理连接都能被“端点分到同一个局部槽”精确表示；非clique、关系还原错误、断开重叠隧道误合并和非确定性均为0。
- fit最大需要18个实际组合簇；加25%预留后需要23个，故固定选择32槽。未见C07最大19槽，容量溢出为0。这证明新接口在Teacher层面可实现，但尚未证明LiDAR模型能预测它。
- 当前仍为Phase 3/P2。底层几何backbone、五帧时序输入、端点可观测性和旧失败模型继续复用；只替换失败的“全局坐标锚点回归”为交换不变的局部槽分配。C08、在线图和M-TARE仍未开放。
- 唯一下一步是关系模型readiness：实现32槽+dustbin输出、置换不变匹配、置信拒绝、同槽连接解码以及真实batch梯度/内存/确定性检查。readiness通过后才允许建立三seed训练Data Card。
- 权威run=`results/gate3_semantics/gate3_20260903_local_composition_slot_teacher_feasibility_v1_seed0`；27项seal全PASS，seal-list SHA-256=`ee6ddaecbcce9959602e11f976aaea4d9144de354f98557877ddd8567a01145c`。

## 2026-09-03：C07组合锚点归因完成；主瓶颈为预测坐标，下一步审计局部组合槽

- 三seed C07归因正式系统PASS且科学结论一致：完整模型在precision≥0.98时均无真连接；预测anchor距离仅seed2保留1个真连接，而Teacher anchor距离三seed均以precision/F1=`1.0/1.0`恢复全部`442,936`个真连接。
- 三seed预测端点平均偏离Teacher anchor约`11.15--11.81 m`，head修正后仍为`10.53--11.26 m`；Teacher primitive端点自身到anchor的q99仅`0.1773 m`。故构造Teacher有效，失败是学生几何坐标精度不足及独立坐标回归接口不适合物理连接。
- 现有几何学习仍是正证据：相对非学习拟合，冻结backbone显著改善axis与surface并提高覆盖；组合anchor和compatibility也把attachment F1提高到`0.202--0.226`。这些资产保留为表征消融，不允许解释为建图PASS。
- 当前仍在Phase 3/P2，C08和结构图关闭。唯一下一步是零训练局部组合槽Teacher可行性：验证可观测attachment能否无损表示为少量交换组合簇，且不同高度/平行非连接隧道不会被传递闭包误合并。通过后才实现学习式slot assignment；失败则该关系表示构成科学阻塞。
- 权威run=`results/gate3_semantics/gate3_20260903_primitive_composition_anchor_failure_attribution_v1r2_seed0`；`RUN_STATE=COMPLETED`、`error=null`、28项seal，seal-list SHA-256=`4825b95e6b0c7c9fa15756e2f7681f8c0b1007351ea75490c5be1c6adb03a2f2`。

## 2026-09-03：组合锚点C07三seed 0/3 FAIL，几何关系有信号但不具备安全建图资格

- 正式训练已完整完成三个seed和`30,699`次更新；原run的最终评估因浮点连乘产生最大`2.98e-8`的末位非对称而在首批系统失败。新的零训练corrective复用原checkpoint和阈值，完整执行`193,932`行C07推理，证明修正分数全部finite且逐位对称，源49项seal与输入未漂移。
- 三seed attachment F1=`0.22550/0.20233/0.21629`，相对基线`0.00638`的增益均超过5个百分点；这说明学习式共享锚点显著优于非学习局部拟合，不是随机模型。
- 但三seed最佳F1点precision仅=`14.60%/13.76%/13.29%`；precision>=`0.98`时TP全部为0。预测锚点相对原预测端点的MAE改善仅=`5.58%/4.64%/5.36%`，三个seed都未达到10%。最终通过数=`0/3`，模型科学状态为FAIL。
- 机制上，训练loss中真连接兼容度改善的同时，叠置/近邻非连接困难项持续恶化，最终产生数十万overlap FP。当前模型能够粗略排序相关端点，但不能把真实连接与相近不连通隧道可靠分开，故不能进入C08或拓扑图。
- 当前唯一允许动作改为C07-only零训练归因：用Teacher anchor/proposal/evidence/scale逐项替换，比较纯锚点距离与Gaussian分数，并按距离和结构族分层，确认是不可观测目标、坐标回归能力、置信尺度还是证据头造成失败。不得通过降0.98门、挑seed、重训或图规则补偿。
- 权威corrective run=`results/gate3_semantics/gate3_20260903_primitive_composition_anchor_c07_evaluation_corrective_v1_seed0`；RUN_STATE=`COMPLETED`、error=null、系统证据PASS，seal-list SHA-256=`2b9a91c62885148962cd3070bab68487a96776626704e5f0c1cfd5933de3af53`。

## 2026-09-03：组合锚点正式训练seed0/1完成，seed2运行中

- Gate 3/P2仍只回答一个问题：共享组合锚点能否把已经可学习的地下基元几何转化为precision>=0.98且非空的物理连接，并在三个seed中至少两个超过同输入非学习基线。
- seed0已完成3轮/10,233步，C07选择总loss连续下降`1.968394 -> 1.950769 -> 1.934105`，固定选择器保留epoch2；anchor NLL和relation loss也分别降到`2.529543/0.516868`。这只是可靠收敛证据，尚不是连接F1或安全提交结果。
- seed0存在明确科学风险：兼容性loss改善`9.22%`，但非连接重叠隧道的hard-negative loss恶化`16.00%`。这可能在离散评估中表现为错误合并；不据此改loss或补训，继续由冻结的0.98精度、非空TP及overlap FP门裁决。
- seed0训练合同、冻结backbone、人口、步数、checkpoint一致性和资源全部PASS；峰值GPU进程`14,742,978,560 bytes`、主机RSS`2,694,041,600 bytes`。总控现已自动执行seed1，未改数据、Teacher、损失、门槛或种子。
- seed0 `selected.pt`与`epoch_02.pt`哈希均为`af37f87cae32ee475e1dd63a66f30eb0d161afadcd66f210d5f55c45dc97e1a5`，冻结骨干前后状态哈希一致。
- 当前仍不允许读取C08或进入在线图/M-TARE。下一动作是让seed1/2按同一合同完成，并由冻结评估器一次性给出三seed离散科学门；只有至少2/3通过才开放下一阶段。
- seed1首轮现已完成3,411步，fit/C07=`426,552/64,644`行；C07 total=`2.022705`，比seed0同阶段高`2.76%`，其中anchor/关系仅高`0.38%/1.36%`、不确定性校准高`5.24%`。这表明第二随机种子尚未异常掉队，但不是科学PASS。资源门通过，checkpoint SHA-256=`a0da178f2f5d650c95fdc7dc1d73e33531e82551c03414c6255fc9b13e7847dc`，epoch1已自动开始。
- seed1第二轮完成后累计6,822步，C07 total/anchor/relation/compatibility相对首轮分别改善`0.48%/0.15%/1.49%/4.77%`；但overlap hard-negative恶化`7.20%`，说明两个seed都出现“真连接拟合改善、近邻非连接分离变差”的同向风险。checkpoint SHA-256=`cad6789319f92142eff7914244445a0ee44b0c3f5c2819c8eb4e0f7a193c117f`；seed1末轮已自动开始，离散安全评估仍未执行。
- seed1现已完成3轮/10,233步，epoch2以C07 total=`2.006676`被固定选择器保留；相对首轮total/relation/compatibility改善`0.79%/1.65%/6.71%`，但overlap hard-negative累计恶化`11.73%`。资源、人口、步数、冻结状态与selected-exact九项合同全部通过；`selected.pt` SHA-256=`a30ceb79d15f502ebf1b48984e09d90f9ed9ccbb38afe5a1efe7e84354b641a2`。
- 总控已自动进入seed2。当前两个完整seed均显示连续目标收敛和困难重叠项恶化，最终是否科学PASS仍必须由三seed部署式attachment F1、precision>=0.98非空TP、anchor MAE与overlap FP共同裁决；C08、graph、M-TARE继续关闭。

## 2026-09-03：组合锚点关系头正式训练readiness PASS

- 当前仍为Gate 3/P2。唯一科学问题未变：22,278参数sensor-polar组合锚点头能否在完整C07上以precision>=0.98产生非空真连接，并在三个seed中至少两个超过同输入基线。
- readiness run精确绑定fit/C07=`426,552/64,644`行和三个冻结source checkpoint；真实batch-128的8/8 head张量梯度有限非零，`2,635,631`个backbone参数保持冻结，同批重复输出与loss逐位一致。
- 资源实测为CUDA allocation `7.13 GiB`、GPU process `13.63 GiB`、host RSS `2.06 GiB`，支持固定batch-128；本run为0 optimizer、0 checkpoint、0 C07 model forward、0 C08/graph/M-TARE。
- 下一步是冻结并执行唯一三seedhead-only训练/评价run，不允许据readiness提前声称模型成功或打开C08。

## 2026-09-03：491,196行组合锚点Teacher sidecar正式PASS；进入head-only三seed训练设计

- V1处理完21个C07任务后系统FAIL：代码把论文图的`80 m`横轴上限误当成数据有效性门，首个超限任务`S08_3d_loop_rich_C07__c1_mixed`只有1个端点达到`80.3915 m`。这属于可视化实现错误，不否定锚点；失败run保持不可修改，0 optimizer/model/C08/graph/M-TARE。
- V1R删除该伪门，不裁剪或改写任何Teacher值；超出显示范围的目标只计入overflow并另存真实最大值。22项测试含`143 m`回归样例和原失败真实任务均PASS，Data Card/spec重新冻结且preflight 0/0。
- 正式V1R完整物化C01--C06 fit=`60 parents/180 tasks/426,552 rows`、C07=`10/30/64,644`，总计`70/210/491,196`。fit/C07 active endpoints=`6,912,786/1,088,828`，全部`3,301,484/525,077`个真连接对的两端锚点逐位相同，最大距离=`0 m`。
- 每个目标均按P1b槽位、当前sensor xyz/yaw重新推导两次并在Zarr重开后全量第三次核对；float32相对float64最大误差=`3.81465e-6 m`，inactive精确为0，全部冻结输入执行前后不变。sidecar本体`48,241,720 bytes`，完整run约`58 MiB`。
- 17/17正式checks与2,259项独立seal复核全PASS，seal-list SHA-256=`2045f4555d7fbe069489cae2baf92f0d72a86765a0f3eb2f0c460996dd1cc6c0`。这证明训练监督完整、无歧义和可复现，不代表模型已经学会连接。
- 当前唯一NEXT是冻结并执行sensor-polar O(E)组合锚点head-only三seed训练：每个seed复用对应冻结几何/时序backbone，只训练22,278参数新头；C01--C06拟合、C07选择，precision≥0.98且非零TP等科学门不变。至少2/3 seed通过前，C08、图和M-TARE继续关闭。
- PASS run=`results/gate3_semantics/gate3_20260903_primitive_composition_anchor_target_sidecar_v1r_seed0`；失败V1=`results/gate3_semantics/gate3_20260903_primitive_composition_anchor_target_sidecar_v1_seed0`。

## 2026-09-03：O(E)组合锚点模型V2 readiness PASS；下一步物化fit/C07紧凑Teacher sidecar

- V1在真实预测槽上暴露近竖直/退化轴局部帧未定义，V1R修正后又在图源NumPy序列化后置失败；两次均为0 optimizer/C07/C08并已封存。V1R2完成科学检查后正式FAIL：12个预测端点轴段短于0.1 mm，其中1个属于真实匹配且可观测端点；轴锚帧导致query-permutation误差`0.001587`、yaw compatibility误差`9.81384`，不能进入训练。
- V2没有删除退化候选，也没有放宽`3e-5`新增头等变门。它改用当前传感器极坐标`[radial,lateral,gravity]`三维修正基，删除不稳定tangent/segment/bend输入，并把可学习Gaussian温度初始化为`softplus(-5)=0.006715`以免放大float32噪声。
- 正式V2 readiness覆盖同一C01真实五帧行：11 active primitives、17 observed endpoints、8个无向正连接、6个overlap hard negatives。退化轴端点12个、匹配/观测各1个，但观测端点径向退化为0；全部20项tests和17项正式checks PASS。
- 新head只有`22,278`个可训练参数，每行输出256值，相对1,984个独立pair值缩减7.75倍。新增头query-permutation误差=`1.5259e-5`，yaw anchor/scale/compatibility误差=`3.8147e-6/5.9605e-8/9.5367e-6`；冻结上游导致的绝对anchor误差仅`4.1008e-5 m<1e-4 m`。
- 当前唯一NEXT是零模型、零训练物化C01--C06 fit与C07 selection的composition-anchor target sidecar，并逐任务验证构造anchor、当前sensor变换、primitive order、inactive zero和observability对齐。sidecar PASS后才冻结head-only三seed训练；C08、图和M-TARE继续关闭。
- PASS run=`results/gate3_semantics/gate3_20260903_primitive_composition_anchor_model_readiness_v2_seed0`；17项seal-list SHA-256=`3fc29fb93c7f332c619192daae595239e8b523f8f2ee68c36413f1adbe6a907d`。

## 2026-09-03：预测端点几何在正确候选下仍不安全；正式开放O(E)组合锚点修正头

- 三个冻结模型已在完整C07上各运行64,644条序列，共193,932次确定性TF32-off前向；旧learned-pair指标在deployed与proposal-oracle两种掩码下逐seed精确复现，系统证据PASS，28项seal-list SHA-256=`9eda771e22bdf9dbb886736c003c3a932c67b6a107e2fe390213764f7147919b`。
- 即使Teacher直接给出正确基元候选，预测端点距离的最佳attachment F1也只有`0.289743/0.282983/0.293032`；三个seed在precision≥`0.98`时安全真连接均为0。fit-only的`0.499973 m`真值几何阈值用于预测端点后，proposal-oracle precision只有`0.222877/0.203807/0.230984`，并产生大量重叠隧道误连接。
- 人话结论：连接关系本身定义正确，候选冗余会放大错误，但最根本的问题是模型预测的轴线端点没有落到同一个真实组合位置。简单距离、旧pair分数和正确候选都不能安全建边，不能靠调阈值或图规则修复。
- 按预注册分叉，停止“直接使用预测端点距离即可构图”的路线，正式允许最小`O(E)`组合锚点残差与不确定性头readiness：每个端点只预测一个共享组合锚点修正和置信度，再由概率兼容性形成连接簇。不得恢复独立`O(E²)`pair head、稠密K槽、手写事件规则或读取C08。
- 当前唯一下一步是零训练模型/loss/readiness：验证端点置换与旋转等变、反向traversal一致、共享锚点正例收敛梯度、stacked/overlap负例分离、真实batch有限反传、学生forward无Teacher身份。通过后才允许单独冻结训练Data Card。
- run=`results/gate3_semantics/gate3_20260903_primitive_predicted_geometry_association_diagnostic_v1_seed0`；总耗时`1429.96 s`，峰值主机RSS=`9,447,988 KiB`、CUDA reserved=`13,985,906,688 bytes`，0 optimizer/C08/C09/C10/graph/M-TARE。

## 2026-09-02：可观测C07非学习基线冻结；三seed训练可正式启动

- 修正后的公平基线已覆盖全部64,644条C07序列，attachment目标正例精确为442,936，隐藏82,141条不参与评分；run seal-list SHA-256=`01100d79bcc91182974259e25ef5d548227d783720202e7dd56a552f67fea58e`。
- 同输入非学习attachment F1=`0.006380`，precision=`0.05114`，recall=`0.003402`。它说明解析拟合能恢复少量真实连接，但远不足以安全建图；同时为学习模型提供了冻结的`+0.05 F1`比较起点。
- 当前没有数据、Teacher、实现或基线阻塞。唯一三seed关系层训练run=`results/gate3_semantics/gate3_20260902_primitive_relation_observable_three_seed_training_v1_seed0`已在preflight 0/0和53/53冻结测试后`RUNNING`，当前为seed0 epoch0；C08、图和M-TARE仍为0。

## 2026-09-02：端点可观测关系训练接口合格；当前准备三seed正式训练

- 正式readiness run=`results/gate3_semantics/gate3_20260902_primitive_relation_observable_readiness_v1r_seed0`在42/42测试和18/18检查后PASS，seal-list SHA-256=`57e3d5f991326262965b57bb942ee51518ea6c927cf67c31702e1dd0ddc27294`。
- 新接口没有退回旧V1目标：V2的稀疏cardinality/最小描述长度仍逐位保留；只有双端有当前五帧射线证据的物理连接参与关系监督，隐藏连接保持unknown。部署时由新endpoint-evidence head从学生特征预测证据，不依赖Teacher mask。
- 三个旧checkpoint只贡献已经通过C07的几何/时序表示。每个seed精确迁移133个状态张量并重置64个关系/evidence张量；正式真实batch证明64个可训练张量全部有有限非零梯度、冻结参数全部无梯度。
- 这只证明“修正后的训练接口可信”，尚未证明关系已经学会。Gate 3仍未恢复，C07训练门、非空precision>=0.98真连接和2/3 seeds条件保持不变；C08、图和M-TARE继续关闭。
- 当前唯一下一步是冻结并执行observable relation三seed训练，不增加事件规则、不调低阈值、不读取C08。

## 2026-09-02：连接失败的主要Teacher错配已量化，进入可观测关系sidecar

- C01--C07正式零训练审计覆盖70个父世界、210个几何任务、491,196条序列和3,826,561个无向连接标签；10项检查全PASS、C08+及graph/M-TARE读取为0。run=`results/gate3_semantics/gate3_20260902_primitive_attachment_teacher_observability_v1_seed0`，17项seal-list SHA-256=`151cdff6b18755a8458172247cf4b1d4e7d317230a7f443b904d1c4c133ccacf`。
- 旧Teacher确实把完整构造图连接直接投给“基元在窗口中出现”的样本，没有验证相连端点本身可见。0.25 m下fit/C07双端可见率只有`84.28%/84.36%`，解释了为什么proposal oracle仍无法让关系分数安全排序。
- 五帧方向没有被否决：fit/C07物理连接identity至少一次双端可见覆盖为`95.17%/95.21%`。按预注册门归类为可修正窗口标签错配。下一步保留双端可见pair监督，把其余pair设为unknown，真实穿越才向持久图提交edge；新关系模型通过C07之前继续关闭C08和建图。
- 紧凑sidecar已经正式PASS：`210`个任务、`491,196`条序列只占`892,360 bytes`，fit/C07精确保留`2,782,487/442,936`个可观测正pair，并把`518,997/82,141`个隐藏正pair设为unknown；可观测候选负例仍充足。run seal-list SHA-256=`52f379fe2f1fdbeca46b8278f9136bb40dc5bcd13217116f4f88e7e75a5060da`。当前进入masked relation reader/loss/model readiness，尚未启动训练。

## 2026-09-02：V2 C07正式0/3 FAIL；几何学习成立，物理连接学习未达到建图资格

- V1R已完成三个seed各7轮，共`561,456`次参数更新；正式run无系统错误，资源、数据、checkpoint和67项SHA-256证据全部有效。证据清单SHA-256=`74a834a1c5a07b42e74a803cb4c008200579ed34153fefec050d77c7bc5a5c82`。
- C07三个seed的表面重建改善约`49%--56%`，连续几何macro改善约`21%--39%`，target coverage约`87%--98%`，跨帧对应accuracy约`80%--87%`。这些结果支持“模型能从五帧LiDAR学习显式地下几何基元和持久身份”。
- attachment F1只有`0.072331/0.032406/0.055452`；相对基线增益只有seed0超过5个百分点。更关键的是三个seed在precision≥`0.98`时均没有一个真实连接可提交。当前方法不能把学到的几何稳定组合成物理关系，因此不能生成论文主张所需的学习式拓扑边。
- Gate 3/P2状态为`GATE_FAIL`；C08读取为0，C09/C10、graph和M-TARE均保持关闭。下一步不是重训或调规划器，而是只用C07和既有checkpoint完成关系层分解归因，确定可证伪的最小修正或停止当前主方法。
- 唯一C07-only归因run已在18项测试和preflight 0/0后启动：三个冻结seed各读取64,644行一次，零训练。它将严格区分候选数量/身份、pairwise解码、关系排序与不确定性校准；结果不会被表述为部署性能，也不会解锁C08或图。
- 初始归因V1因batch rank张量未广播在首批立即系统FAIL并封存，未形成模型结论。V1R只显式扩展rank源到正式batch大小，19/19测试和preflight 0/0通过后已启动并越过原错误；数据、模型、Teacher、分数、解码和诊断顺序不变。
- V1R完成全部193,932条推理后，仅在外层required-file枚举发生未定义`SEEDS`的post-compute NameError。新的CPU-only evidence corrective没有重跑模型，验证源23项seal、全部三seed/逐任务/三格式图、资源、隔离和上游hash后PASS；25项新seal SHA-256=`cfab8a74cb329b05198503c92e70d68c6230b44023f8ddf2d757b2ed9e37afd7`。
- 归因结论为当前关系分数在Teacher proposal oracle下仍失败：三个seed的oracle attachment F1仅`0.2009/0.1793/0.1630`，precision≥0.98安全真连接仍全为0；精确阈值、Teacher基数和best-link均不能修复。当前relation Transformer正式停止，下一步先审计连接Teacher的五帧可观测性，再决定研究接口而非继续换网络重训。

## 2026-09-01：V2资源证据修正版运行中；模型科学实验重新从seed0开始

- 原run启动后发现系统证据缺陷：名为`process_memory`的值来自`nvidia-smi`，不能同时证明16 GiB主机RAM门。它在任何epoch/checkpoint完成前被主动停止并封存，属于资源监控失败，不支持也不否定模型。
- V1R没有改学习方法，只增加独立主机`VmHWM/RUSAGE_CHILDREN`监控并明确区分PyTorch显存、GPU进程显存和主机RSS。48/48测试含超限强制停止负控制，preflight 0/0。
- 新唯一run已启动全新seed0 epoch0；当前约2.29 GiB主机RSS、3.05 GiB GPU进程显存。训练仍为每轮426,552序列、每seed 7轮、3 seeds共561,456步；C08及建图保持关闭。
- seed0 epoch0现已完整完成并写出checkpoint：26,736步，训练/C07人口精确为426,552/64,644，全部loss与梯度有限；耗时4,251.64秒。GPU进程峰值15.028 GiB、PyTorch allocation 7.34 GiB、主机RSS约2.53 GiB，三类资源证据均通过。当前自动运行epoch1稀疏集合阶段；尚不能用单轮内部loss判断科学成败。
- seed0 epoch1稀疏集合阶段也已完整完成：累计53,472步，C07 primitive-set/surface loss相对epoch0分别下降13.87%/17.51%，checkpoint SHA-256=`ff250b9e889c9b508e633ead7f5c2a943264352ecf0e95e24786e333a5c862ea`；GPU进程峰值约15.02 GiB、主机VmHWM约2.53 GiB，均通过资源门。当前自动运行epoch2端点关系阶段；冗余基元是否真正消除、safe attachment是否非空仍须等待三seed正式C07评估，尚无科学PASS。
- seed0 epoch2端点关系阶段完成后，C07关系loss改善43.21%，但几何集合、表面和不确定性loss同时大幅恶化，属于“学会关系时忘掉几何”的阶段性任务冲突。它不终止冻结训练，因为epoch4--6原定就是六任务联合恢复；但当前checkpoint不可用于建图，最终若不能同时恢复几何且保留关系收益则V2失败。累计80,208步，checkpoint SHA-256=`2789d133bed156d3a8c46ef26de04cc66e353cb55d5c18e9d35a1e00c1cd06c9`，资源门继续PASS；当前自动运行epoch3时序/不确定性阶段。
- seed0 epoch3时序/不确定性阶段完成：时序和不确定性C07 loss分别改善71.89%和99.89%，但关系loss相对上一轮恶化67.24%，几何仍未恢复，进一步确认“单项能学、共享表示互相覆盖”。累计106,944步，checkpoint SHA-256=`34dee5d1d78dc153ba7f9eb4d2f3a7efc7fbc40bb9677163c7f9c3d67b255343`，资源门PASS。当前进入epoch4--6六任务联合训练；只有联合阶段同时保住全部能力，seed0才可能进入正式C07离散评估。
- seed0 epoch4首轮联合训练显著缓解任务冲突：几何集合/表面均恢复到此前最佳，关系和时序相对epoch3分别改善27.71%/30.32%，总loss改善40.89%。尚存两点风险：关系未回到专训最低值，自由空间loss小幅恶化；且这些仍是连续loss，不等于离散基元、attachment F1和98%精度非空命中通过。累计133,680步，checkpoint SHA-256=`9cce9ed4111157a33f1abbaf2dc283b876036637b95004dda31be2b21680c14f`，资源门PASS；当前自动运行epoch5第二轮联合训练。
- seed0 epoch5第二轮联合训练没有继续单调改善：几何基本稳定且自由空间/不确定性改善，但关系、时序和C07 total分别恶化12.59%/9.01%/7.44%。当前最佳仍为epoch4的最低C07 total，不允许按单项挑epoch5；这说明多任务平衡可达到但仍有波动。累计160,416步，checkpoint SHA-256=`64df7c37e638986bda737b667ad1832ba0749a33d1512f3e8ddeb7865ada3b0b`，资源门PASS；当前自动运行epoch6最后一轮联合训练，离散科学门尚未执行。
- seed0已完成7轮/187,152步。最后一轮训练关系loss正常但C07关系loss恶化190.97%、总loss恶化81.44%，证明后期对C01--C06过拟合；冻结选择按规则选中epoch4，`selected.pt`与`epoch_04.pt` SHA-256同为`9cce9ed4111157a33f1abbaf2dc283b876036637b95004dda31be2b21680c14f`。独立host RSS峰值约2.53 GiB、GPU进程峰值15.028 GiB，资源PASS；这仍不是离散科学PASS。seed1已从全新初始化自动开始，C08/C09/C10/graph/M-TARE仍为0。
- seed1 epoch0完成26,736步，C07 total比seed0同阶段高3.99%，但表面和关系连续loss略好；没有异常掉队，随机初始化差异仍在正常范围。checkpoint SHA-256=`c5714581bbcaaec06c11382d7ba58c4a6d74a7b7adea2188cebaee4215fb19db`，资源门PASS；当前自动运行seed1 epoch1稀疏集合阶段。
- seed1 epoch1也已完成，累计53,472步。C07表面loss改善31.51%、总loss改善3.19%、不确定性loss下降，但primitive-set loss小幅恶化3.39%，说明表面拟合更好尚不等于重复基元已被消除；最终必须由离散基元数、coverage和attachment门判断。checkpoint SHA-256=`8267d59f891bac1d75207315c6d518afa69a3c8d4ff5e99e35c5da0c3d66d59d`，资源门继续PASS；当前自动运行seed1 epoch2端点关系阶段，C08仍关闭。
- seed1 epoch2连接关系专训已完成，累计80,208步。C07端点关系loss改善51.46%到`0.336576`，比seed0同阶段更低，证明关系目标在第二个随机种子也可学习；但几何集合、表面和不确定性同时大幅回退，复现了共享表示任务覆盖。该checkpoint不能建图；当前自动进入epoch3时序/不确定性阶段，之后三轮联合训练必须同时恢复几何和保留关系。checkpoint SHA-256=`dfd4d8fdc27001e3d41b8db029227036a0ae58533c8515093f52e5755ce69d39`，资源门PASS，C08仍关闭。
- seed1 epoch3完成后累计106,944步。C07时序/不确定性分别改善78.63%/98.58%，说明五帧对应和置信度同样可学习；但连接关系又恶化153.98%，几何只部分恢复。两个seed至此都给出相同机制证据：单项能力存在，问题在共享表示不能自然同时保住它们。当前进入epoch4首轮六任务联合训练；checkpoint SHA-256=`19e925a14fe48289e4367b8ce65f1d896a8bba787fc58d99b6e764f7aad83840`，资源门PASS，C08仍关闭。
- seed1 epoch4首轮联合训练恢复了几何、自由空间和时序，但C07连接关系恶化到`1.183414`，相对关系专训最佳差251.61%；训练集关系loss却为`0.292223`。这表明seed1当前不是“没学连接”，而是连接规律在训练地图上成立、到未见C07上崩坏。checkpoint SHA-256=`a92a8b66cadd009639bafb97712a93710b3f38aec333cdd73b0a1b611528d351`，资源门PASS；当前运行epoch5第二轮联合训练，剩余两轮必须恢复关系才能保留该seed通过可能，C08仍关闭。
- seed1 epoch5把C07关系loss从`1.183414`恢复到`0.951534`，但仍比专训最佳差182.71%；几何保持稳定，总loss降到当前最低`0.295728`。说明联合训练正在部分修复关系，但未见域泛化差距仍明显。checkpoint SHA-256=`aecc8a1afa10140883363d2b97fc9a71177a8770e20cb22ca7e198a66bdb0c1c`，资源门PASS；当前运行seed1最后一轮epoch6，C08仍关闭。
- seed1已完成完整7轮/187,152步。最后一轮把C07关系loss恢复到`0.573549`并把总loss降到全程最低`0.253593`，冻结选择器因此选择epoch6；`selected.pt` SHA-256=`0ee826131ff5781f5598da437745d2d9d0e367b0acf773caecdac374c90b3168`。但关系仍比专训最佳差70.41%，表面/时序也较上一轮回退，说明形成的是多任务折中而非稳定全恢复。host RSS峰值约2.53 GiB、GPU进程峰值15.028 GiB，资源PASS；离散科学门仍未执行。seed2已从全新初始化自动开始，C08仍关闭。
- seed2 epoch0几何预训练已完成26,736步，训练/C07人口和batch继续精确，全部loss与gradient有限。C07 surface=`0.145497`，优于seed0/seed1同阶段约22%/19%；primitive-set=`0.213505`则差约24%/21%，total=`0.547025`略优。这属于健康的随机初始化差异：底层表面恢复有效，但连接头尚未专训，不能声称结构关系成功。checkpoint SHA-256=`8bf4145ed0416b6dcd9ad2b4a4d2b8314d4b1c64c0f0a0c6548cebf75767b372`，CUDA allocation峰值约7.46 GiB；当前自动运行seed2 epoch1稀疏集合/MDL阶段，C08/C09/C10/graph/M-TARE仍为0。
- seed2 epoch1稀疏集合/MDL已完成，累计53,472步。C07基元集合、表面和自由空间loss分别改善`28.90%/2.29%/29.79%`，说明针对第一版候选过密的修正同时改善集合稀疏性和几何解释，没有以破坏表面为代价。C07 total与seed0同阶段几乎相同、比seed1低3.91%，跨初始化没有异常。checkpoint SHA-256=`00a1395ccb0bbb2fed40a8b0b0ca973c85cffe7cc1119f0d2d199bd93186bddf`，资源门PASS；当前自动运行seed2 epoch2端点关系专训，离散门尚未执行，C08及图仍关闭。
- seed2 epoch2端点关系专训已完成，累计80,208步。C07连接关系loss从`0.697169`降到`0.227114`，改善`67.42%`，说明三个随机初始化都能学到“哪些端口物理相连”的训练信号；但形状集合、表面和不确定性同时明显退化，属于专任务覆盖而不是可直接建图的结果。checkpoint SHA-256=`9ae9037beafc3a0426c9e4c133b68053a93e2006e51ab5c19c0577a84864b29a`，资源门PASS。三seed训练总进度为`454,512/561,456=80.95%`，当前运行seed2 epoch3时序/不确定性阶段；最终仍须后三轮联合训练和冻结离散门，C08及图保持关闭。
- seed2 epoch3时序/不确定性专训已完成，累计106,944步。C07时序和不确定性loss分别改善`79.49%/98.50%`，时序结果略优于另外两个种子，证明第三个初始化也能从五帧中追踪同一结构并输出可训练的不确定性；连接loss同时恶化`330.19%`，表面和自由空间也回退，进一步确认共享网络存在稳定的任务覆盖。checkpoint SHA-256=`0feb0499dbfa7eca2b4e4de6acaef457a1829fdc15e4a27879b994b0003c4b2f`，资源门PASS。三seed总进度=`481,248/561,456=85.71%`，当前已自动运行seed2 epoch4首轮六任务联合训练；只有联合阶段让几何、连接、时序和拒绝共存，才可能通过离散门并进入建图。
- seed2 epoch4首轮联合训练已完成，累计133,680步。C07形状集合、表面、自由空间和时序loss分别改善`76.89%/53.24%/63.43%/13.41%`，说明联合训练成功恢复几何和五帧身份；但连接loss继续恶化`42.80%`到`1.395239`，是训练集`0.275771`的`5.06x`。因此当前明确问题是连接规律没有迁移到未见C07拓扑，而不是优化器没有拟合训练集；四种能力尚未共存。checkpoint SHA-256=`fb84118298b3733d49590eb2f148be283e6f80bc624b85f4b7361b58c5a41f3f`，资源门PASS。三seed总进度=`507,984/561,456=90.48%`，当前运行seed2 epoch5第二轮联合训练；剩余两轮仍按冻结日程执行，C08和图保持关闭。
- seed2 epoch5第二轮联合训练已完成，累计160,416步。C07连接loss从`1.395239`恢复到`0.929426`（改善`33.39%`），形状集合、表面、不确定性和总loss也继续改善；但连接仍为训练集的`3.13x`且比专训最佳高`309.23%`，自由空间恶化`80.58%`、时序略退`3.47%`。因此模型正在恢复连接泛化，但四种能力仍互相拉扯，当前仍不可建图。checkpoint SHA-256=`3687bae95655af9e213950ba238b22eee23a8ef7fb8ca06a483acdd810bc76cf`，资源门PASS。三seed总进度=`534,720/561,456=95.24%`，当前运行seed2 epoch6最后一轮联合训练；结束后直接执行冻结C07离散门，C08和图仍关闭。
- seed2 epoch6及全部三seed训练已完成，总计561,456步。最后一轮C07连接loss反弹`29.13%`、总loss反弹`13.66%`，尽管自由空间和几何继续改善；固定选择器因此正确保留更均衡的epoch5，selected SHA-256=`3687bae95655af9e213950ba238b22eee23a8ef7fb8ca06a483acdd810bc76cf`。这证明连接泛化在联合训练中不稳定，不能靠继续加epoch解决；是否仍满足离散F1和98%精度非空提交必须由正式评估判定。seed2独立host RSS峰值约2.54 GiB，三seed资源均PASS。三seed冻结C07离散评估已自动启动；C08、图和M-TARE保持关闭。
- C07 seed0离散结果为FAIL，但失败边界比V1明显推进：表面和连续几何分别改善56.37%/20.55%，primitive F1=`0.599041`、coverage=`0.873565`，attachment F1=`0.072331`且相对基线增益6.65个百分点，前五项全部通过。唯一失败是98%精度下安全attachment仍为零真阳性；模型能给连接排序，却没有一个连接达到可自动落边的可靠度。结果SHA-256=`4c62f04f60eaadef411f752c1b4e2a0bb804375e884b2eb470192c9b4d24be8e`。seed1评估已自动开始；至少2/3 seed通过的总门尚未判定，C08和图仍关闭。
- 新增Hydra/S-Graphs和TopoNet/DAGMapper原始来源边界后，论文不再声称首次从几何构图、首次层级图或首次联合几何—拓扑学习；独立主张只保留因果三维隧道基元物理关系、拒绝合并、traversal-only edge及其探索收益。成熟解析图构造将作为借鉴骨架、基线或安全验证，禁止用它替代失败的学习关系而沿用原主张。矩阵、正文、BibTeX和决策日志已同步，当前训练源码哈希无漂移。
- run=`results/gate3_semantics/gate3_20260901_primitive_relation_sparse_port_three_seed_training_v1r_seed0`。

## 2026-08-31：V2三seed正式训练运行中；当前是seed0第1轮

- 这不是再次readiness。唯一不可覆盖正式run已经启动，44/44冻结测试通过，seed0正在用C01--C06的426,552个五帧序列执行epoch0几何预训练。
- 全合同为3个独立随机seed、每个7轮，共561,456次参数更新。C07只负责选checkpoint和阈值；C08当前没有被打开，C09/C10、在线图和M-TARE也没有运行。
- 通过条件保持严格：至少2/3 seed同时超过同输入非学习基线的几何、连接、检测和覆盖门，并在precision>=0.98时真正命中至少一条attachment。模型全部拒绝连接不能算安全通过。
- 当前只证明正式管线启动正常，尚无首轮loss或科学结果。run=`results/gate3_semantics/gate3_20260831_primitive_relation_sparse_port_three_seed_training_v1_seed0`。

## 2026-08-31：稀疏端点关系V2 readiness PASS；模型尚未训练

- V2把V1已证实的两项故障直接编码进可训练目标：未匹配槽不再被class-balanced存在性丢掉真实先验，精确cardinality/冗余描述长度压制重复候选；端点关系Transformer使用位置、外向切线、截面、形状与描述子上下文，并输出关系不确定性。
- 正式readiness只读一条C04源序列的三种配对几何，共15帧；42/42测试、193/193参数梯度、六类loss、稀疏梯度方向、置换/反转/旋转/对称/重复和资源门全部PASS。0 optimizer、0 checkpoint、0 C07/C08/C09/C10、0 graph/M-TARE。
- 当前结论仅为“实现与训练接口合格”，不是“关系已经学会”。下一唯一动作是冻结V2七轮三seed训练；只有C07至少2/3通过原科学门，才允许一次C08迁移。
- run seal-list SHA-256=`81c017d4943792e93a8bd2f18cb3ccaa0bf96ff68deac7f6fd901622bf8ff2af`。

## 2026-08-31：V1R2归因完成；当前进入稀疏端点关系架构readiness

- C07-only V1R2已完整结束：三个冻结seed、每seed`64,644`行，两遍TF32-off重评加一遍归因，共`581,796`次推理；run=`COMPLETED`、`error=null`，0 optimizer/C08/C09/C10/graph/M-TARE。
- 修正后三seed仍仅`1/3`通过V1门。每seed约206.6--206.8万激活候选对54.4万Teacher目标，使端点pair空间扩大约14.04倍；这是当前关系假阳性的第一来源。
- Teacher对齐proposal oracle把attachment F1提高到`0.2049/0.2567/0.2327`，但三seed在precision>=0.98时仍全部零真实attachment提交。因此“只修候选存在性”被否决；当前V1的关系分数尚不能安全生成拓扑连接。
- 当前唯一允许动作是零训练sparse-port relation architecture readiness：复用有效的五帧LiDAR、时序和几何编码，增加稀疏基数/最小描述长度、端点相对几何上下文、relation transformer及关系不确定性。readiness通过前不得启动新三seed训练、读取C08或建图。
- 正式归因图=`results/gate3_semantics/gate3_20260831_primitive_relation_v1_failure_attribution_v1r2_seed0/previews/primitive_relation_v1_failure_attribution.png`；32项seal复核有效，seal-list SHA-256=`ca2600041eb72b70d5ff0118a900526297cbc867b9ab83396e1c2a12be115b93`。

## 2026-08-31：旧C07评估发现cuDNN TF32合同漂移；正确C07重评与归因运行中

- V1R1完整seed0归因无法复现旧formal三类离散计数并按合同FAIL。只读代码/环境证据定位为：训练器关闭matmul和cuDNN TF32，但独立旧评估器未关闭cuDNN TF32，当前默认值为true。
- 因此旧V1仍是重要失败证据，但其精确C07数字不能继续标为满足“TF32 off”声明；尚不能据此选择下一模型修订。数据、Teacher、checkpoint训练和C08隔离均未受影响。
- 正在执行唯一C07-only V1R2：相同三个checkpoint、64,644行/seed，先正确重评再做proposal-oracle归因，共581,796推理行，0训练/C08/C09/C10/graph/M-TARE；11/11测试和preflight 0/0已通过。

## 2026-08-31：当前V1基元关系模型在C07以1/3 seed通过，正式科学停止

- 三seed六轮正式训练和冻结C07评估均已完整结束，run状态=`COMPLETED`、`error=null`；只有seed1通过五项相对基线门，seed0/seed2的attachment F1增益不足，最终`1/3<2/3`。
- 三seed sampled-surface Chamfer改善`43.17%/50.56%/53.80%`、连续几何macro改善`42.78%/45.54%/44.81%`、target coverage均约`99.9%`，因此显式3D扫掠基元学习得到正证据。
- 但模型每seed预测约206万候选而真目标只有544,414，primitive precision约26.3%；端口关系在大量假候选的两两组合中失去选择性。attachment F1只有`0.0292/0.0810/0.0408`，三个seed在precision>=0.98时均零安全提交。
- 结论：当前V1不能进入C08或在线结构图；C08 rows read=`0`，C09/C10、graph、M-TARE均为0。当前唯一允许工作是密封C07输出的只读失败归因，不得修改阈值、挑seed或以规则/规划补偿。
- 结果图=`results/gate3_semantics/gate3_20260830_primitive_relation_three_seed_training_v1_seed0/previews/primitive_relation_comparison.png`；证据清单SHA-256=`81f43937e813f11b1b8cf337edd05f842ba83a189cf94ed54feb5260b7fe2aa7`。

## 2026-08-30：非学习几何基元下界已完成；正式进入三seed学习模型训练

同输入非学习基线已经在完整C07上正式冻结：10个独立父世界、30个三形状任务、64,644个五帧序列，模型推理、优化、C08/C09/C10、图和M-TARE均为0。run为`results/gate3_semantics/gate3_20260830_primitive_relation_nonlearning_c07_v1_seed0`，状态COMPLETED、error为空，45项证据seal SHA为`605df99e2c542642bdc45d9530a10c89df1512cde95651c97f1a822688a637f7`。

它提供了清晰的论文下界：预测基元precision为`99.99%`，但recall只有`17.53%`、F1为`29.83%`；attachment F1为`0.58%`，disconnected-overlap F1为0，时序对应accuracy为`15.55%`，表面Chamfer为`17.31 m`。结论不是“基线运行成功所以方法有效”，而是手工局部拟合只能找出少量明显基元，无法恢复完整几何组合和连接关系。

主论文已建立与当前方法一致的新骨架`docs/PRIMITIVE_RELATION_STRUCTURAL_GRAPH_MANUSCRIPT_DRAFT_V1.md`；旧exit-only稿只保留为基线文本。新稿已经写清构造程序Teacher、三形状扫掠基元、五帧集合/关系模型、执行验证图、同输入对照、单/多机器人闭环和必要消融，所有尚无密封结果的结论都保持`AUTO`占位。逐项完成证据由`docs/PRIMITIVE_RELATION_PAPER_COMPLETION_MATRIX_V1.md`追踪。

最新原始论文复核进一步收窄了创新边界：ArcPro已用程序合成的稀疏点云—建筑程序配对学习逆程序，StructureNet已联合学习部件几何和关系，Cano、PRISM-TopoMap、地下Segmented Map和M-TARE也分别覆盖出口拓扑、学习地点图及地下探索。因此投稿不能声称上述单点首次；权威贡献矩阵`docs/PRIMITIVE_RELATION_NOVELTY_MATRIX_V1.md`要求完整证明“部分可见因果LiDAR的扫掠隧道基元关系 → 持久结构图 → traversal-only edge → 不改局部栈的探索收益”。

当前唯一下一步是正式三seed训练。训练固定使用C01--C06的60个父世界、180个任务、426,552个五帧序列；C07的10个父世界、30个任务、64,644个序列只用于checkpoint和阈值选择。只有三个seed在C07至少2/3满足预注册增益门，才读取C08的73,182个序列做一次零适配迁移。C09/C10、在线图和M-TARE仍关闭。

训练Data Card/spec已通过preflight 0/0，正式run `results/gate3_semantics/gate3_20260830_primitive_relation_three_seed_training_v1_seed0`现为RUNNING。27项冻结测试全部通过，seed0已完成6/6轮，seed1已自动开始；当前只有内部loss趋势，尚无最终冻结几何/关系性能结论，也没有读取C08。

seed0前三轮现已完整完成：每轮26,736个训练batch/426,552行和522个C07 batch/64,644行全部精确，累计80,208次optimizer step；六类loss、总loss和梯度范数均finite，`epoch_00.pt`到`epoch_02.pt`均已保存。关系预训练使C07端口关系loss下降25.96%；随后时序/不确定性预训练使C07 temporal loss从`2.099932`降到`0.454086`（下降78.38%）、uncertainty loss从`0.177309`降到`0.002560`（下降98.56%），总loss降到`0.395260`。单任务阶段造成关系、表面与ray项回退，必须由epoch3--5联合训练恢复，因此仍不能判科学PASS。第三轮耗时4,206.29秒，进程显存峰值14.936 GiB，满足16 GiB硬门。当前自动进入seed0第4轮首轮联合训练，C08仍未读取。

seed0第4轮首轮联合训练现已完整完成：累计`106,944`次optimizer step，训练与C07计数仍逐项精确，`epoch_03.pt`已保存。C07总loss由`0.395260`降至`0.300798`；surface、ray和port-relation loss分别恢复到`0.165779`、`0.043213`和`0.575632`，说明联合训练正在修复单任务预训练造成的几何/关系回退；temporal和uncertainty loss为`0.573748`与`0.000516`。本轮用时4,296.16秒，进程显存峰值14.489 GiB，满足16 GiB门。当前已自动进入seed0第5轮联合训练；内部loss不是最终论文指标，必须等完整六轮后用冻结的Chamfer、连续几何MAE、attachment F1、primitive F1与coverage正式判定，C08仍未读取。

seed0第5轮联合训练也已完整结束：累计`133,680`次optimizer step，训练/C07仍精确为`426,552/64,644`行和`26,736/522`个batch，所有loss与gradient norm有限。`epoch_04.pt`大小`23,807,103 bytes`、SHA-256为`3ea307e0431c6f9896fb3ed10fb848ce625cd8404eb779c4bb117177f53cbec2`。C07总loss由`0.300798`小幅降到`0.298724`，primitive和temporal项改善，但surface与port-relation项回升；这说明训练仍数值稳定，但不能据此宣称结构恢复已超过基线。该轮用时4,299.70秒，进程显存峰值14.616 GiB，满足16 GiB门。当前自动执行seed0第6轮即最后一轮，C08仍未读取。

seed0第6轮联合训练已经完成，六轮合计`160,416`次optimizer step。最后一轮训练总loss降至`0.215987`，但C07选择总loss回升到`0.313710`，所以冻结选择器正确保留epoch4的`0.298724`；`selected.pt`与`epoch_04.pt`的SHA-256完全相同，都是`3ea307e0431c6f9896fb3ed10fb848ce625cd8404eb779c4bb117177f53cbec2`。seed0总耗时25,562.59秒，峰值进程GPU内存14.998 GiB，低于16 GiB上限。总控已在06:30自动启动seed1；这仍只是一个种子的训练/选模完整性，不是相对非学习基线的科学PASS，C08/C09/C10、在线图和M-TARE仍未读取。

seed1 epoch0几何预训练也已完整完成：训练/C07分别精确为`426,552/64,644`行与`26,736/522`个batch，累计`26,736`次optimizer step，全部损失与梯度有限。C07总loss为`0.696785`，高于seed0同阶段的`0.627985`，但三seed合同禁止据此丢弃种子或改变训练；只有最终冻结指标决定是否通过。`epoch_00.pt`大小`22,320,635 bytes`，SHA-256为`5bbd5dc9a2d93221f938592fee50ac2e1d2a7cb10dd62989c2d2ca5a7b2cc591`；本轮用时4,225.29秒，峰值进程GPU内存14.996 GiB，低于16 GiB门。seed1 epoch1关系预训练已自动开始，C08/C09/C10、在线图和M-TARE仍未读取。

seed1 epoch1关系预训练现已完整完成：累计`53,472`次optimizer step，训练/C07计数继续逐项精确，所有loss与gradient norm均有限。C07 port-relation loss从`0.697757`降至`0.577823`，下降`17.19%`，总loss降至`0.624677`；几何、ray和时序项在关系单任务阶段回升，必须由后续联合阶段恢复，不能据此判科学PASS。`epoch_01.pt`大小`23,003,383 bytes`，SHA-256为`5c897bbf5fdb450cc37988d462815fd292cbba9413d4e3700ac643f77d3a8435`；本轮用时4,212.53秒，峰值进程GPU内存14.563 GiB，低于16 GiB门。seed1 epoch2时序/不确定性预训练已自动开始，C08/C09/C10、在线图和M-TARE仍未读取。

seed1现已继续完成epoch2--4，累计`133,680`次optimizer step；每轮训练/C07仍严格为`426,552/64,644`行和`26,736/522`个batch，全部loss与gradient norm有限。epoch2使C07 temporal/uncertainty由`2.086685/0.218156`降至`0.437748/0.012942`，随后epoch3--4联合训练把一度回升的surface/ray/port-relation恢复到`0.149446/0.051772/0.501851`，当前C07总loss最低为epoch4的`0.291615`。epoch2/3/4 checkpoint SHA-256依次为`282e4bfa...a123e`、`6cf00643...c46cb`、`b83cc0d7...6fb4e`；各轮峰值进程GPU内存均低于16 GiB。当前正在执行seed1最后一轮epoch5；这只证明分阶段训练收敛与资源合同正常，最终相对基线结论仍等待冻结评估器，C08仍未读取。

seed1第6轮及整个seed现已完整结束，六轮合计`160,416`次optimizer step。最后一轮C07总loss由epoch4的`0.291615`小幅降至`0.289074`，固定选择器因此保留epoch5；`selected.pt`与`epoch_05.pt` SHA-256均为`ab7768ea4eb6b5ec47095f0fa0df1761109b391810db476e336e715783ecada8`。人话解释是：最后一轮对自由空间、基元参数和时间一致性有改善，但表面重建和端点连接关系反而变差，模型仍存在多任务顾此失彼；这只是风险信号，是否失败必须由冻结评估器把最终几何和连接指标与同输入非学习基线逐项比较。seed1总耗时24,721.29秒，峰值进程GPU内存14.996 GiB，资源合同通过。总控已自动启动seed2；C08/C09/C10、在线图和M-TARE仍未读取。

seed2 epoch0几何预训练现已完整结束：训练/C07精确为`426,552/64,644`行和`26,736/522`个batch，累计`26,736`次optimizer step，全部损失和梯度有限。C07总loss为`0.682251`，介于seed0/seed1同阶段的`0.627985/0.696785`之间；人话解释是第三种子早期收敛有随机差异但没有异常掉队，连接关系和时间一致性尚未进入专门训练阶段，所以此时不能判断结构恢复成败。`epoch_00.pt`大小`22,320,635 bytes`、SHA-256为`f6db9b609d2ec2fd24bf4593dea1176481486790f75c32751bf7ba564d997f95`；本轮用时4,187.12秒，峰值进程GPU内存14.996 GiB，低于16 GiB门。seed2 epoch1关系预训练已自动开始；C08/C09/C10、在线图和M-TARE仍未读取。

seed2 epoch1关系预训练现已完整结束：累计`53,472`次optimizer step，训练/C07人口逐项精确，全部损失和梯度有限。C07端点连接loss由`0.703490`降至`0.686872`，仅改善`2.36%`，而seed0/seed1同阶段约改善`25.96%/17.19%`；表面重建loss同时由`0.167282`升至`0.533279`。人话解释是这个种子在专门学习隧道端点怎么连接时只学到少量信号，却明显忘掉一部分几何，属于后三轮联合训练必须修复的具体风险；它尚不否定最终方法，因为冻结日程本来允许单任务阶段回退，并要求联合阶段恢复。`epoch_01.pt`大小`23,003,383 bytes`、SHA-256为`4130ff198553c93fc02db21128391146161254b95622c959d2eb72a89144dfce`；本轮用时4,156.06秒，峰值进程GPU内存14.715 GiB。seed2 epoch2时序/不确定性预训练已自动开始；C08/C09/C10、在线图和M-TARE仍未读取。

seed2 epoch2时序/不确定性预训练现已完整结束：累计`80,208`次optimizer step，训练/C07人口逐项精确，全部损失和梯度有限。C07时序loss由`2.105678`降至`0.628625`，改善`70.15%`；不确定性loss由`0.103742`降至`0.002108`，改善`97.97%`。人话解释是模型已经明显学会在五帧之间追踪同一几何基元，并改善置信度判断；但表面重建虽然从上一轮`0.533279`恢复到`0.363073`，仍远高于第一轮`0.167282`，所以“学习连接时忘掉几何”仍是后三轮联合训练必须解决的主风险。`epoch_02.pt`大小`23,807,103 bytes`、SHA-256为`8753e4b7a8389cd03d14d3f32b1dfd4f4960209edba31aca9c9b5bdce7aaa761`；本轮用时4,206.75秒，峰值进程GPU内存14.909 GiB。seed2 epoch3首轮联合训练已自动开始；C08/C09/C10、在线图和M-TARE仍未读取。

seed2 epoch3首轮联合训练现已完整结束：累计`106,944`次optimizer step，训练/C07继续精确为`426,552/64,644`行和`26,736/522`个batch，全部损失与梯度有限。C07表面重建、自由空间和端点连接loss分别从`0.363073/0.229778/0.669983`降到`0.158167/0.039639/0.565359`，时序和不确定性保持在`0.542759/0.000626`，总loss降到`0.301643`。人话解释是：首轮联合训练已经把“顾着学连接而忘了几何”的问题修回来，而且连接和五帧追踪能力也没有因此失效。不过这些仍是训练内部loss，模型是否真正超过非学习基线，仍必须等六轮结束后用冻结物理指标判定。`epoch_03.pt` SHA-256为`bd7bebc934ec211df8f550a45a399a798cfd4370c4ed1429619984951182f728`；本轮用时4,271.80秒，峰值进程GPU内存14.469 GiB，资源门通过。seed2 epoch4已自动开始；C08/C09/C10、在线图和M-TARE仍未读取。

seed2 epoch4第二轮联合训练现已完整结束：累计`133,680`次optimizer step，训练/C07人口继续逐项精确，全部损失与梯度有限。C07端点连接和时序loss从`0.565359/0.542759`略降到`0.561436/0.542357`，但表面重建、自由空间和基元参数loss从`0.158167/0.039639/0.503309`回升到`0.172958/0.043921/0.727826`，总loss回升到`0.341573`。人话解释是：模型仍然记得部件如何连接，也能在五帧中追踪同一部件，但第二轮联合训练让它更难准确还原部件的形状参数；这是模型多任务之间互相拉扯，不是数据、Teacher或系统故障。固定选择器当前应保留更均衡的epoch3，最后一轮仍按预注册日程执行，不提前停止、改权重或补训。`epoch_04.pt` SHA-256为`4efa3d05eeebe6158e6af56da4dbc9164546076d3dbcf8780844a461bf02801a`；本轮用时4,311.65秒，峰值进程GPU内存14.616 GiB，资源门通过。seed2 epoch5最后一轮已自动开始；C08/C09/C10、在线图和M-TARE仍未读取。

seed2 epoch5及三种子全部六轮训练现已完整结束。seed2共`160,416`次optimizer step，三种子总计`481,248`次；每轮训练/C07人口和batch逐项精确，全部损失与梯度有限。seed2最后一轮将C07总loss从`0.341573`恢复至`0.314999`，连接、基元参数和时序均好于epoch4，但仍没有超过epoch3的`0.301643`，因此固定选择器正确保留epoch3。`epoch_05.pt`/`selected.pt` SHA-256分别为`56fd2d7d5ab197bc183a2d61588be039652da2e0b5c334ecde9c1bc6610694f4`/`bd7bebc934ec211df8f550a45a399a798cfd4370c4ed1429619984951182f728`；seed2总用时25,418.28秒，峰值进程GPU内存14.996 GiB，资源门通过。人话解释是：训练流程已完整跑完，实验没有系统失败，但多任务顾此失彼仍是模型风险；不能用内部loss宣称成功。冻结评估器已自动启动C07正式门检，当前进程级只读审计未见C08打开；C09/C10、在线图和M-TARE仍未读取。

C07正式门检已完成seed0，结果为正式FAIL。模型的surface Chamfer=`9.8386 m`，相对非学习基线改善`43.17%`；连续几何macro改善`42.78%`；primitive F1=`0.4167`、target coverage=`99.92%`、时序对应accuracy=`90.32%`，都说明五帧LiDAR确实学到了比手工拟合更完整的几何结构。但attachment precision/recall/F1只有`2.56%/3.40%/2.92%`，对baseline的绝对F1增益为`2.34`个百分点，没有达到预注册`5`个百分点。具体机制是模型为了覆盖`544,414`个真实基元预测了`2,066,264`个候选，过多候选进入关系组合后产生`680,094`个错误attachment；安全阈值下又零提交。因此当前有证据支持“显式几何可学”，但不支持“学习关系已可安全建图”。seed1/2仍按原冻结合同评估，不改阈值、指标或checkpoint；C08/C09/C10、在线图和M-TARE仍未读取。

历史证据已按论文角色完成第一轮清点：10组已有PNG/PDF/SVG图片及其`figure_source.json`已逐文件确认存在，覆盖事件槽、几何锚点、极坐标多深度、exact-one、出口关系传输、保守提交、Dual Composer、单帧剖面、规则残差和同输入非学习基元。准确路径和用途已写入`docs/PAPER_EVIDENCE_RETENTION_MATRIX.md`；这些图、生成源、原始metrics和seal在论文完成前禁止清理。旧离线拓扑关联只有密封数值，没有合格图，后续只能从原始sweep确定性生成。

当前主方法图也已正式生成并嵌入主论文：`docs/figures/gse_graph/primitive_relation_method_overview.{png,pdf,svg}`。它把训练期程序构造监督与部署期输入明确分开，并完整显示五帧LiDAR、扫掠基元、物理关系、持久结构图、真实穿越确认边及不变M-TARE局部栈。结构化source、provenance、确定性publisher和SHA清单齐全，manifest SHA-256为`d2d6a23fe2a44ddad17b8f131b3bfa0219310ebd8fd4df1dcf9d21da8b10b63d`；不包含任何未完成实验的性能数字。

数据与地图Figure 2也已从P1a/P1b密封源确定性生成并嵌入论文：`docs/figures/gse_graph/primitive_relation_dataset_overview.{png,pdf,svg}`。固定样本是字典序首个fit世界`S01_flat_tree_small_C01`、`c1_mixed`、global frame 0，图中同时展示实际生成的55条物理隧道轴线/56个节点、三种连续截面、学生输入的range图和训练期来源编码。图源NPZ、summary、provenance、publisher及SHA清单齐全，manifest SHA-256为`a517298d533dabed8ae848435b1378f1168edb4efb8839ec81ff1b32e5d7b17b`；选择不依赖模型结果，C09/C10读取为0。

同输入非学习C07基线现已发布并嵌入论文Figure 3：`docs/figures/gse_graph/primitive_relation_nonlearning_baseline.{png,pdf,svg}`。三种图从正式密封run逐字节复制，同时生成紧凑机器指标、来源说明、确定性publisher和SHA-256清单；五项清单文件均复核通过，manifest为`54e080ae5c40484ad9a684d6b8e3690d86a28f1881380a0bec64106cd4dd3772`。该图只使用C07，C08/C09/C10读取为0。历史成功和失败资产继续按正式对比、单变量消融、科学失败分析、复现错误四类管理；论文排版和补充材料映射完成前不删除对应图片、指标、配置或seal。

一项等待期论文整理被主动停止：Spatial Event Set、Geometry-Anchored Joint和Structured Polar Multi-depth三个历史summary的人口核对为`45,942=21,548+24,394`，合并了旧C07+C08，并非纯C07。没有打开当前P1a/P1b C08 shard、没有执行当前模型或生成新图，但为保持当前P2“C07通过前不重新消费C08结果”的清晰隔离，暂不出版该合成图；旧原图和密封证据继续保留，待当前模型与阈值冻结后再处理。主训练未受影响。

## 2026-08-30：底层基元关系模型已具备正式训练资格；模型尚未训练

当前Gate 3/P2的readiness已经PASS。模型输入只含五帧16×720 LiDAR range/valid和当前传感器坐标下的相对里程计；输出32个可交换扫掠基元的轴线、端点截面、存在性、不确定性，以及端口连接、物理不连接重叠和跨帧对应。模型共有1,969,516个参数。

第一次正式V1没有训练，唯一失败是Teacher槽号影响Hungarian近似平局：槽位重新编号后某个关系loss变化`1.7345e-05`，超过冻结门限`1e-6`。V1R没有放宽门限，而是在匹配前用基元自身几何和端口关系生成与槽号无关的规范顺序。V1R正式达到30/30测试、141/141参数梯度finite且nonzero、六类loss finite、槽位重排和端点反向误差均为0、重复推理误差0；峰值显存约362 MiB。run为`results/gate3_semantics/gate3_20260830_primitive_relation_model_readiness_v1r_seed0`，seal SHA为`39f72889a7333a6c9ee060ebf13a317cafb0a8ef1762aee9a21144f148e3968f`。

这只证明实现可以科学地训练，不证明模型精度。当前唯一下一步是先建立同输入的非学习几何拟合baseline，再冻结正式三seed训练Data Card/spec。训练只允许C01--C06拟合，C07选择，C08一次零适配迁移；C09/C10、在线图和M-TARE仍关闭。历史出口模型、规则图、Composer、事件槽和ERCSS资产在论文证据矩阵建立前不得删除：有效结果用于基线/消融，具有方法含义的失败用于失败分析，纯环境或脚本错误只进入复现附录。

## 2026-08-30：P1训练数据与Teacher完整闭合；当前进入底层基元关系模型readiness

P1b系统修正版已正式PASS：80个父世界、240个三形状任务、757,290个源帧和564,378个五帧序列全部完成；每个样本从五帧LiDAR与相对里程计对应到最多32个可交换扫掠基元槽，实际最大为24。Teacher包含8,779,620个有向端口连接、3,023,794个“视觉重叠但物理不连接”关系和538,391个跨帧dustbin对应，三类监督都不是空壳。

初始V1失败只因真实分片pilot漏传`source_run`，发生在任何正式Teacher分片写入前。V1R只补该路径字段并绑定V1失败证据；真实分片双重复哈希相同。V1R 10项正式检查全部通过，Teacher数据为283,883,437 bytes，RUN_STATE为COMPLETED、error为空、零训练/推理/图/C09/C10；41,390项seal SHA为`f629b511e9a945bbe15249e3bc212be57a01b1aa0118d0d7d228824aa25f7d47`。

当前进入P2，但还没有训练模型。唯一下一步是模型readiness：实现五帧LiDAR和相对里程计的圆周编码、32槽显式基元集合、端口关系与跨帧对应，并在解析场景和C01--C06真实batch上证明排列不变、端点反向等价、旋转/相对运动一致、六项loss有限且所有应训练参数有有限梯度。该审计PASS后，才允许一个正式三随机种子训练run；C07只选checkpoint/阈值，C08只做一次零适配迁移，C09/C10和图继续关闭。

## 2026-08-30：P1a科学与资源合同全部闭合；当前启动P1b五帧关系Teacher

P1a源run的20.857740 GiB容量FAIL保持不可修改。唯一无损corrective现已正式PASS：80个父世界、240个三形状分片、757,290帧、8,723,980,800条射线、24,117个基元和402次姿态修正全部保留，2,640个数组逐chunk raw-byte一致。数据分片降至19,388,655,191 bytes，满足20 GiB上限；9项检查全部通过，RUN_STATE为COMPLETED且error为空。127,380项seal SHA-256为`79fd988ac8c205d74c93e4858b7b579571a06502e778f31634046b48791a0668`。

当前唯一任务是绑定该seal冻结并执行P1b：从逐射线构造来源生成564,378个五帧样本，每个样本最多32个可交换基元槽，监督局部轴线/截面、端口连接、视觉重叠但物理不连接的困难负例、跨帧对应和相对里程计。学生输入仍只有五帧range/valid与相对里程计；绝对pose、world、TNG和primitive identity不会进入forward。正式同解释器的准备测试为10/10 PASS。

P1b完成前仍不训练、不建图、不读取C09/C10或M-TARE。若完整计数、32槽容量、关系标签、源seal、确定性或12 GiB资源任一失败，P1停止；全部PASS才进入P2模型readiness和三随机种子训练。

## 2026-08-30：P1a科学数据完整、仅容量门FAIL；无损存储corrective已完成

当前不是从头开始，也还没有训练模型。正式P1a已经完成240/240分片：757,290帧LiDAR、8,723,980,800条射线、24,117个三几何基元和402次配对姿态修正全部精确。数据、身份、原点、确定性、多来源和tree hash共11项检查全部通过；唯一失败是压缩分片22,395,827,779 bytes=20.857740 GiB，超过预注册20 GiB。runner正常退出且error为空，原run以resource-only FAIL和124,740项seal不可修改保存。

因为唯一失败项正是容量，预注册的无损corrective合法启用并已完成。Data Card/spec通过0错误0警告preflight，新不可覆盖run使用12个CPU把每个Zarr array改为zstd9 byte-shuffle，并逐目标chunk复读比较原始dtype字节；codebook/construction逐字节复制，源/目标tree均记录。2项内置测试和正式资源门均已通过。

资源fallback的代码已提前准备但没有抢跑：新修正器只把物理压缩改为zstd9 byte-shuffle，严格保留11个数组的schema、chunk和所有逻辑值，并对每个写后chunk按原始dtype字节复读比较；codebook和construction文档也必须逐字节相同。2项合成测试全部通过；一个完整真实分片也达到11/11数组和group attrs逐块bit-exact，物理大小比例0.871286，临时输出已清理。只有原run完成、封存且确认唯一失败项就是20 GiB容量时，才能冻结并运行新的不可覆盖corrective；若存在任何数据/身份/确定性失败，该fallback自动禁用。

进入S05后，完整人口首次出现单射线同时属于4个构造基元：codebook 119绑定四条独立edge primitive并在数组中出现1条射线。前置固定窗口审计的最大3只是抽样观测，并非冻结验收阈值。正式codebook合同允许并完整保存1--255来源，32槽限制的是五帧可见基元总数，两者没有冲突。P1b遍历任意长度来源集合，新增四来源无损回归后准备期完整单元测试为10/10 PASS，因此正式P1a继续而不改变数据或阈值。

P1b补齐部署可得、当前传感器坐标表达的五帧相对里程计后，完整真实分片完成双重复试跑：1,292帧生成860个五帧样本，最大17个可见基元，得到14,616个端点连接标签、5,891个视觉重叠但物理不连接的困难关系和1,013个遮挡消失对应；两次完整输出哈希完全相同，每次约半分钟。P1b正式执行器、压缩格式、Data Card冻结器与10项单元测试均已准备，但被锁定为必须等待P1a科学PASS后才能启动。

后续P2的泄漏安全数据接口也已准备并通过3项测试：模型输入对象只包含五帧range/valid和相对里程计，不暴露绝对pose、world/TNG或primitive identity；这些构造身份只留在Teacher目标侧做集合匹配。P1a/P1b的parent、partition、geometry或frame row一旦不一致即停止。当前没有借此启动训练。

方法接口同时剥离了不可辨识的“是否未探索”感知标签：模型只学习几何和关系，在线图依据真实穿越历史维护端口执行状态。该阶段已由上方corrective PASS取代。

## 2026-08-30：Teacher容量冻结为32，当前进入P1正式数据导出

80个开发父世界、三种几何、三类结构窗口的正式全扫描已经完成。C01--C06最大可见基元数为16；按预先约定的25%余量需要20槽，因此固定候选中选择32。C07/C08最大分别为23和19，不需要用迁移域反向调整。8槽会丢264个窗口，16槽仍会丢15个，32槽没有丢失。

当前P0前置链已经闭合：构造身份、三形状、80-world规模、高速CSG、端点姿态与32槽容量全部通过。唯一下一步是P1正式导出757,290帧与564,378个五帧序列，保存LiDAR range/valid、逐射线基元来源codebook、32槽局部基元参数、端口组合和时间可见性监督。训练、图、C09/C10与M-TARE仍关闭，直到P1完整性与泄漏审计通过。

## 2026-08-30：端点姿态合同已正式修复，当前回到Teacher槽位容量选择

P1预审发现坡道端点的world-vertical传感器偏置会越过有限构造端盖：三种几何合计402次失败，对应134个源姿态。修改几何端盖虽然能包住姿态，却造成61个非关联隧道重叠，因此该方案正式否决，不能进入Teacher。

最终方案不改地图，只对实际越界的134个端点帧沿原有向路线做最小内移。正式全量资格覆盖80个C01--C08世界、252,430个源姿态和757,290个三形状检查；修复后越界为0，252,296个无问题姿态逐元素不变，帧身份/数量/顺序全部不变，最大位移0.300015m，最小相邻间距0.699985m。当前唯一下一步是绑定该seal重跑五帧可见基元容量审计；其结果将冻结模型输出槽数16或32，之后才能正式导出P1训练数据。

## 2026-08-30：P1导出前发现固定8槽Teacher会删除真实可见基元

C01的真实五帧全扫描证明，普通走廊/交叉口/端点窗口分别可见`7/12/9`个构造基元；交叉口多出的基元具有至少153条回波支持，因此不能视作偶发噪声。当前数据codebook本身可以无损保存这些来源，但论文方法规范的8-slot decoder无法容纳12个目标。

受影响结论是P1 Teacher和后续模型容量，不影响已通过的CSG距离/身份后端。完整80-world导出暂不启动。当前唯一下一步是冻结一个C01--C08三形状、每世界三角色窗口的零训练容量审计：只用C01--C06选择最小带25%余量的`8/16/32`容量，C07/C08只验证不适配迁移；若32仍不足则固定槽接口科学失败，转为可变长度集合而不是删Teacher。

## 2026-08-30：P0全部数据生成前置合同PASS；进入P1三形状训练数据导出

高速Teacher阻塞已经解除。正式C01证明在6,144条走廊/交叉口/端点射线上达到`99.8698%`有效性一致、`98.8607%`可用覆盖、`100%`基元身份一致，range p99=`2.45 cm`且无误差超过5cm。AABB稀疏查询与完整查询逐hit哈希一致，说明加速没有改变标签；相对逐步解析Teacher加速`15.56x`，完整P1按2倍复杂度保护、32 workers估计`18.25 h`。

当前Phase 3/P1的唯一任务是冻结并执行80个C01--C08父世界、三种面积保持截面实现的数据导出：757,290帧、564,378个五帧因果序列、每帧16×720 range/valid/primitive membership及构造Teacher。C09/C10、训练、graph和M-TARE继续关闭，直到P1对计数、身份、歧义mask、因果窗口、重复确定性、磁盘和无泄漏全部PASS。

## 2026-08-30：P0表示与80-parent数据合同通过；正在实现可扩展身份射线后端

论文主线的底层数据表示现已闭合：一个swept-superellipse schema连续覆盖ellipse、rounded rectangle和C1 mixed section；三种配对实现由parent/edge hash决定并保持截面面积。80个C01--C08父世界可产生8,039 edge ×3=`24,117`个带身份基元，计划规模为757,290帧和564,378个因果序列。5个非共轴组合端口使用“anchor在incident tunnel自由空间内”的物理合同，8个stale degree字段以实际edge incidence替代但完整留档，不向Teacher伪造edge。

当前唯一阻塞是工程吞吐：native range渲染外推3.61小时且磁盘约8.77 GB，均可行；现有Python逐步隐式provenance外推80.86天，不允许启动P1。下一步实现独立闭合primitive mesh的多命中CSG union ray backend：从一次高速射线返回的有序face/primitive hits恢复occupancy，跳过内部重叠面，只提交离开union的第一个有身份交点。解析场景与C01必须同时证明语义等价和足够吞吐，之后才能冻结P1导出Data Card。

## 2026-08-30：逐射线构造身份链已修复，P0进入截面形状泛化

正式corrective已证明：edge-level隐式几何可以在组合隧道的真实union出口上返回完整primitive membership，且不会把多义交叉面硬分配给单个基元。C01的69,120条固定射线中，62,311条可用于唯一基元监督，5,365条多义命中被明确拒绝，完整复跑确定性一致；因此历史Poisson无身份的问题不再阻塞新Teacher生成。

当前仍是Gate 3/P0，而不是开始训练。现有实现只覆盖当前C01圆形扫掠管，尚未证明论文要求的椭圆、圆角矩形和C1连续混合截面。唯一下一步是把provenance field推广为shape-generic swept superellipse，完成ellipse/rounded-rectangle/taper/curve/T/Y/X/stacked-overlap解析合同，并对80个既有拓扑父世界做只读形状变体inventory和资源测算。只有这些合同通过，才允许冻结P1数据生成Data Card。

## 2026-08-30：历史资产不能直接提供逐回波构造监督

P0证明现有graph/spline/radius足以把C01的55条edge恢复为55个独立扫掠基元，并通过56个节点组合覆盖110个端口。失败发生在表面来源链：统一Poisson mesh删除了逐面构造身份，旧LiDAR没有命中triangle/surface/primitive ID，所以历史range不能无歧义对应到Teacher基元。

当前仍是Gate 3/P0。唯一下一步是在生成器adapter中保留Poisson之前的基元/交叉表面provenance，并让Teacher raycaster输出每个有效回波的primitive-hit identity；先做解析场景和一个C01的小型重生成合同，PASS后才进入P1数据生成。

## 2026-08-29：主方法重构为显式几何基元学习与关系图谱

当前论文不再以出口分类、五类结构事件Composer或直接骨架Teacher为主方法。新的单一主张是：程序化地图的真实构造程序提供截面基元、中心轴扫掠、SE(3)变换、组合连接和surface provenance；学生从五帧因果LiDAR与相对里程计学习显式扫掠超椭圆基元、端口关系和跨帧对应，再持续融合为结构语义图谱。结构语义由几何实体及其关系表达，不由手写junction/turn/terminal规则决定。

现有80个C01--C08拓扑父世界、252,430帧、188,126个序列及注册/raycast/mesh/TNG基础设施继续复用。当前生成器主要是椭圆拱顶和平底，后续必须在保持父世界split的前提下补充圆角矩形和连续形变几何；正式规模、资源和hash在新Data Card中冻结。C09只作历史污染域，C10和M-TARE正式比较继续隔离。

Composer已科学FAIL，RouteGeometryProfile已因可观测性FAIL；ERCSS只完成基础代码且全量审计三次在科学前系统停止。用户的新方法决定取代V1R3治理纠正，故V1R3不再执行，旧run完整保留。

当前唯一下一步不是训练，而是零训练construction-supervision feasibility pilot：验证现有/扩展生成器能否无歧义导出基元、变换、组合树、端口连接、surface provenance和按LiDAR可见性裁剪的学生目标。该proof通过后才可设计新几何数据导出；失败则先修生成合同，不允许靠事件规则或规划器补偿。

## 2026-08-29：单帧前向几何剖面不可行，方法转向五帧里程计配准骨架

RouteGeometryProfile正式proof已把失败分成两部分。客观Teacher在62,180个正反向物理位置上高度一致，说明mesh/spline标签可靠；但单张16线LiDAR在转弯和几何变化附近平均只能同时看见约1.5--1.9个五米段。turn物理身份覆盖在fit/selection仅`21.5%/29.5%`，transition为`76.3%/58.8%`，远低于冻结80%门，且没有任何transition观察具有完整0--20m Teacher。当前“单帧前向剖面”表示被正式否定。

不再修改距离或覆盖率重跑。下一候选利用系统真实可用但旧模型没有使用的相对里程计，将过去五帧LiDAR配准到当前机器人坐标，直接学习稀疏可见隧道骨架片段及其连续截面属性。它不从dense占据图后处理骨架，不把出口数量当结构，也不预测未经观测的全局地图；全局边仍必须由机器人真实穿越确认。

进入模型前先做一次零训练可行性审计：证明五帧相对位姿完备、配准确定、可见骨架Teacher唯一、结构身份覆盖足够且固定输出容量无溢出。该审计失败则停止此候选；通过也只允许实现模型readiness，C09/C10和拓扑图仍关闭。

## 2026-08-29：双Composer完整训练失败，下一步转向路线几何剖面证明

正式三种子训练已完整结束，不是程序崩溃。新双Composer在C08的三种子结构事件macro-F1为`0.5353/0.5245/0.5292`，旧V2R5基线为`0.5815/0.5979/0.5888`，平均反而下降`5.97`个百分点。ensemble完整方法为`0.5371`，旧基线为`0.5957`。完整方法在C07找不到同时满足高精度和最低召回的非空提交阈值，因此C08安全提交为0；若删除拒绝模块，precision只剩`0.3280`且错误接收率为`67.20%`。

消融进一步给出方法级根因：去掉度量几何后macro-F1升到`0.5594`，去掉跨帧transport后为`0.5371`、与完整方法几乎相同。现有每帧width/height/slope/curvature四个全局标量不能表达“变化发生在路线哪里”，transport也没有形成独立可重复收益。因此这不是继续加轮数或调阈值可以解决的问题，当前显式状态不能成为论文主方法，拓扑图继续关闭。

按冻结方法规范，下一步只做零训练`RouteGeometryProfile`证明：从因果LiDAR和客观spline/mesh Teacher检验沿前向可见通道的宽、高、坡度、曲率纵向剖面是否可观测、唯一、无未来泄漏。证明通过后才实现新表示；若该证明失败，停止当前GSE方向并重新评估论文创新，而不是退回Cano已覆盖的出口计数图。

## 2026-08-29：联合出口集合接口已通过训练前资格

新的候选感知接口已经从“许多局部峰再阈值筛选”改成“先判断出口数量，再把一份归一化概率质量分给整个出口集合”。额外假方向会从真实出口夺走质量，重复和ghost在数学上都受到直接惩罚；推理始终输出预测数量对应的连续出口，不再依赖一个无法达到高精度的存在性阈值。

全部188,126条开发因果序列和396,913个出口的引用、数量分布与隔离合同通过。1--4出口真实batch可有限反向；环绕0度和仅相隔4度的出口都能同时保留；旋转、批次排列、重复和五帧历史检查通过。初次run唯一失败是把5.96e-8浮点误差要求等于0，V1R按原合同修正后正式PASS。

这一步只确立“方法接口可以训练”，尚未证明性能。下一步用相同C01--C06训练、C07选择、C08零适配迁移和三个随机种子训练；若连续出口集合、数量、拒绝安全和几何仍不能改善，就停止该候选，不进入拓扑图。

## 2026-08-29：V2仍不足以确立最终方法，转向联合连续出口集合

V2三种子完整训练证明柔性角度和困难假峰不是无效修改：严格峰值排序比V1提高约4个百分点，轴线、宽高和曲率继续稳定泛化。但模型每帧仍形成约20个局部峰，没有任何置信度阈值能达到建图所需的99.5%精度；坡度也仍未过门。因此当前感知模块不能冻结，拓扑图仍不开放。

离线oracle进一步证明不能只补一个出口数量分类器。即使给出真实出口数，再从当前场选同样数量的峰，严格方位命中也只有约四分之一；误差来自“数量”和“峰位置”两个接口。继续调整BCE/focal、阈值或NMS只会重复同一路线。

下一候选主方法改为因果圆周出口集合过程：网络联合预测出口数量和1--4个连续圆周方位，用集合级无序匹配训练，几何属性与每个连续出口绑定；稳定集合变化再触发拓扑节点，真实穿越才建立边。这仍复用已经学会的五帧圆周backbone和连续结构几何，不从头推倒。先做数学与真实batch readiness，通过后才训练。

## 2026-08-29：V2损失已通过训练前验证，准备三种子正式训练

V1不是完全没学会结构：轴线、宽高和曲率已经迁移到C08，出口峰也通常落在真方向附近。失败来自把真出口压在唯一2度格子上训练，同时没有直接惩罚高置信假峰；每帧因此产生约20个局部峰，无法达到建图要求的高精度。

新的最小修正已经在不训练的条件下通过。它允许真方向附近形成连续监督，并专门选择最高分假峰施加抑制；三个旧模型的真实LiDAR样本都给出正确梯度且数值稳定。仅靠放宽匹配需要约20度，已明确拒绝，因此后续仍用严格中心峰指标检验是否真的修好。

当前仍不能建拓扑图。唯一下一步是用完全相同的数据、网络、三个随机种子和10轮训练重跑V2，只改变出口峰损失；C07用于选择，C08只做一次迁移检查。若严格峰值和0.995精度门仍失败，就停止该修正而不让图或规划器掩盖问题。

## 2026-08-28：180×2结构事件场通过全量训练前资格

新rasterizer已经在全部开发标签上证明无损：每个事件直接落到确定的方位/深度槽，同方向近远事件都保留，坐标重建误差低于0.03毫米。新模型不再用attention加权向量决定方向，而是从固定方位中心和有界残差构造坐标；旋转、free-range、top-k、确定性和反向传播均通过。

当前接口具备正式训练资格，但性能仍未知。下一步是三颗种子直接监督360个dense slots：正样本位置由Teacher rasterizer确定，空槽直接监督为无事件；验证仍只用C07--C08选择模型和固定阈值。只有三种子均超过旧出口基线预注册门，才进入离线拓扑图。

## 2026-08-28：新表示已收敛为180方位×2深度的结构事件场

Teacher审计发现地下结构事件确实会沿同一视线层叠：近处路口与远处终点可同时位于同一个2度方位、同一个高度带。全开发集共有277对同方位事件、251对连高度带也相同，但每帧每方位最多两个。因此单方位单token会丢真值，方位×高度也不够；两个按距离排序的深度槽是最小无损方案。

新接口将直接在180×2个固定几何槽上学习事件存在性和属性，方位只允许±1度局部残差，距离只允许落在对应五帧free-range内。这样训练监督不再依赖Hungarian自由query去猜方向，也不会因attention方向向量相消导致远程事件收缩。下一步只做rasterizer与模型readiness；训练和拓扑图仍未开放。

## 2026-08-28：根因已从“置信度”收敛为自由query空间候选失败

完整只读归因已经排除三条便宜修复：去重几乎不改变结果；让全部16个query参加仍只能覆盖约12%的真实事件；忽略事件类型也只增加不到0.5个百分点。约2.9万个目标在每个种子中都没有4米内候选，且方位和径向误差都很大。因此不能再修阈值、NMS、类别头或存在性校准。

下一模型改为结构化polar proposal：在180个LiDAR方位bin上直接学习dense eventness/type/range/elevation，候选方位由局部几何峰值产生，只学习有限局部残差；距离使用对应bin的free-range support，而不再把扩散attention向量相消后当坐标。实现前先做Teacher polar-slot可行性审计，确认同bin/邻bin事件冲突和局部support能否无损表达全部131,424个开发标签。审计通过前不训练、不建图。

## 2026-08-28：几何锚定联合模型正式失败，当前转入候选生成目标归因

三颗完整种子均已完成，系统、数据和隔离合同正常，但新模型没有超过现有出口基线。它能把少量成功匹配事件定位到约2.2--2.5米误差，说明显式polar/free-range几何不是完全无效；真正失败的是16个自由候选槽对稀疏结构事件的真假判别和覆盖，最佳阈值已经到0.95仍有大量假候选，同时召回只有5%--9%。

因此当前不能建立论文主方法拓扑图。该结果否定的是这一版自由query联合目标，不否定“LiDAR几何结构语义直接建图”的研究问题。唯一下一步是对三种子45,942帧输出做只读候选归因，量化重复、类型错配、径向错位、空帧误报、事件基数和assignment冲突。归因完成前禁止重训、调阈值、改Teacher、访问C09/C10/M-TARE或用图参数掩盖感知失败。

## 2026-08-28：几何锚定结构模型已具备正式训练资格

V2零训练资格已经把“模型能不能表示这些标签”闭环证明完成：131,424个可观测terminal/junction标签全部落在五帧自由空间包络内，最紧样本仍有正余量；模型输入只有range/valid，264,134个参数的旋转等变、query无序集合匹配、确定性和反向传播全部通过。Teacher V2和Torch实现的support profile逐元素相同。

因此当前不再卡在接口或Teacher。下一步是真正联合训练encoder与set head，而不是继续使用旧冻结特征。训练仍严格限定C01--C06梯度、C07--C08 checkpoint/置信度选择，三seed 0/1/2；C09/C10/M-TARE、拓扑图和planner继续封闭。只有新模型在同类型4m set指标、连续位置误差和多事件召回上达到预注册门，才会进入离线拓扑图。

## 2026-08-28：可观测Teacher V2已冻结，回到新encoder资格验证

Teacher—传感器错配已经用与模型无关的固定规则纠正。正式V2保留全部188,126条五帧观察和全部1,076个结构identity，只删除目标仰俯角超出冻结16线LiDAR ±15°视场的1,631个token；保留131,424个terminal/junction标签，并记录每个删除项的完整来源。转换过程没有读取扫描、模型输出、错误列表或测试世界。

这一步没有把难例偷偷删掉：删除规则在运行前冻结，仅对应硬件无法观测的垂直方向；3D复杂family删除更多正是其坡度导致，而所有identity仍有可学习观察。正式资产、论文图、删除明细和1,298项seal均已验证。项目现返回Gate 3，唯一下一步是以V2重做geometry-anchored零训练资格；只有物理支撑、旋转、集合监督和梯度全部PASS，才会建立三seed联合训练Data Card。

## 2026-08-28：新encoder数值与接口合格，训练被Teacher—传感器视场不一致阻塞

GeometryAnchoredSpatialEventEncoder已实现为真正从五帧organized LiDAR学习的圆周空间encoder，而不是旧encoder外挂head。它只接收range/valid，保留高度投影与180个方位bin，16个query的位置由可见free-range锚定；264,134个参数在真实0--5事件集合上完成有限前反向，旋转、query排列、确定性和物理距离边界全部通过。

全量零训练证明发现旧Teacher的连续mesh射线并不等于16线LiDAR可观测性：133,055个标签中1,631个目标仰俯角超出固定±15°视场，正式V1因此有641个anchor失败。加入五帧包络后仍不能解释577个视场外标签，证明不能靠模型或后处理绕开。该问题属于Teacher，不属于model/loss/metric/system。

推荐且已按用户持续自动授权选择的修正是：保留旧Teacher和FAIL证据，新建V2只过滤固定垂直视场外token，188,126条观察一条不删、1,076个结构identity全部保留；同时把anchor定义为五帧局部free-range包络并复用原0.25m LOS容差。扩大LiDAR视场会重做252,430帧与全部基线，改变硬件合同，故拒绝。当前唯一下一步是Teacher V2正式导出与新readiness；两者通过前不训练、不建图。

## 2026-08-28：只读归因排除重复query，下一模型必须显式学习远程空间结构

固定4米去重几乎没有删除候选，三seed最多只删0.96%，F1不变；即使把置信度完全拿掉、让16个query全部参加，同类型4米覆盖也只有20.0%--25.5%。因此不是阈值太严、query太多或缺NMS，旧编码器输出本身没有把远处结构事件放到正确位置附近。

距离分层给出直接证据：0--4米目标仍能召回约40%--56%，12米后快速下降，32米后接近失效，40--50米几乎为零。下一组件改为坐标感知的极坐标结构编码器：角度和高度显式编码，当前与历史自由距离作为径向锚点，事件深度预测相对于真实可见射线范围，而不是从压缩context凭空回归0--50米。

当前只允许先做零训练readiness：接口不得读取pose/world/TNG，圆周旋转必须等变，预测深度必须受观测free-range约束，空集合与1--5事件匹配、有限梯度和Teacher join全部通过。通过后才建立joint encoder+set三seed训练Data Card；不得回到旧decoder调参，也不得进入图或planner。

## 2026-08-28：空间事件集合头容量失败，阻塞已定位到表示或集合生成机制

三套固定旧编码器上的16-query集合解码器已完整训练和封存。它们能把一部分terminal/junction放到正确位置附近，成功匹配后的平均误差约2.5米，但会生成大量重复或错位事件，三seed F1仅13.0%、18.6%、14.4%，全部低于旧互斥单中心baseline的39.1%。这不是训练未完成：共13,344个decoder更新、0个encoder更新，输入哈希、环境和程序均正常。

该证据否定“旧局部语义encoder已经足够，只需外挂多事件head”，但没有否定GSE-Graph主问题。纯解析几何baseline F1只有0.3%，旧学习模型仍明显有效；下一步需要确认集合头失败来自query重复/基数失准，还是旧encoder在训练时已经丢弃50米范围内的空间事件布局。

现在禁止直接重训、解冻backbone或进入拓扑图。唯一下一项是C01--C08既有输出的只读失败归因：按目标距离、集合基数、类型和世界family统计；量化同类型4米内query重复、预测半径分布、NMS/oracle去重上界和旧单中心覆盖。只有证据表明简单集合约束足够，才允许一个预注册corrective；否则下一主模型必须从训练目标开始学习空间结构encoder。

准备阶段曾有一次广泛文件搜索枚举到历史C09输出的字段名和形状，但未读取或使用数值；该进程已隔离，未进入Data Card、输入哈希、训练、阈值或正式证据。C09继续只算历史开发域，C10仍是严格未见测试。

## 2026-08-28：根因已定，结构语义从互斥类别升级为空间事件集合

正式归因没有再调模型，而是把失败观测、三种子出口token、训练前后概率和客观图邻接逐项对齐。结果显示98个关系端点中96个在50米尺度内直接邻接相反事件类型，48个terminal全部直接连着junction。一个漏检稳定看成身后junction，另一个虽然三个seed都给出强route-stop线索，但结构置信度从0.967分裂到0.016。

因此现在不再要求每条LiDAR序列只能回答“junction还是terminal”。新表示必须回答“周围有哪些结构事件、各自在机器人坐标系哪里”：同一帧可以同时有前方terminal和身后junction，每个事件带相对三维位置、局部轴、宽高坡曲、出口token、descriptor和不确定性。图仍只在真实穿越后建边，模糊事件不强行合并。

当前还没有训练这个新head。下一项是Teacher可行性审计：用C01--C08 TNG+spline+native mesh+冻结pose确认50米内真实LOS事件集合是否唯一、基数是否可控、两split是否有足够共现和是否覆盖现有失败。审计不过就不训练；C09/C10/M-TARE继续隔离。

## 2026-08-28：路线条件小头已正式失败，停止调参并回到表示/Teacher接口审计

三种子643参数小头已完整运行。它把C07--C08正确事件episode从928提高到931、误触发从94降到90，但两个决定关系覆盖的低支持端点仍为0/2；接回同一图合同后，正确节点由193降到185，节点召回从70.44%降到67.52%。边仍为4条且全部正确、没有假回环，因此结果不是系统故障，而是“普通事件略好、稀有拓扑结构反而少”的有效科学反证。

该路线现在停止：不加训练步、不调0.97阈值、不扩大643参数head。固定route-flow AUC较高但不能转化为关键节点，说明瓶颈可能是16维汇总丢失六出口之间的关系，也可能是objective terminal/junction标签与机器人当下可执行动作并不一致。下一步只做训练前后token级归因，依据证据在“保留完整出口集合关系的学习头”和“重新定义可观测结构事件Teacher”中只选一个。

历史12帧模型已经做过并在C09以2.464%误触发率失败，不能重复包装。C09/C10/M-TARE继续禁止进入本次归因和任何新选择。

## 2026-08-28：路线条件特征已证明有效，下一步进入最小学习头实现

固定、零训练的路线停止分数在C01--C06和C07--C08分别得到`0.949`和`0.935`的terminal-vs-junction AUC，远超预注册`0.80`门槛。它只使用机器人坐标系中的前/后/侧出口置信度，不读TNG身份或绝对位置；这说明现有LiDAR token里已经有足够信息，缺的是把出口与执行方向建立关系。

下一模型会把五帧route-flow作为显式输入，并用一个零初始化小残差同时修正“是否是决策节点”和“junction还是terminal”。原LiDAR encoder、ActionSet时序context、空间位置、关联、执行验证和0.97阈值保持冻结。只有它在C07--C08找回至少一个低支持端点且不增加误触发、接回完整图仍不退化，才成为GSE-Graph主方法组件；否则保留该AUC作为分析而停止训练路线。

## 2026-08-28：问题不是“terminal分类器太弱”，而是缺少路线条件

最新正式审计把两个低支持terminal放回冻结特征空间比较。一个被模型高置信判成junction，而且三种子最近邻几乎全是真junction；另一个在物理出口空间最像普通走廊。这说明仅重训junction/terminal类别头会逼模型记住不可见的TNG身份，不能形成可泛化的结构语义。

真实拓扑解释了混叠：这两个dead-end离degree-3/4路口只有约13.5m和11.7m，远小于50m LiDAR量程。机器人在dead-end仍看得到身后的分支；无路线条件的出口集合自然把它看成junction。现有token其实含有解决线索：第二个terminal三个seed的前向出口质量都为0，后向质量约1.6；第一个也总体后向大于前向。

因此下一方法不是普通类别残差，而是route-conditioned exit flow：相对于机器人执行方向，显式区分前方未穿越出口、侧向分支和身后已执行入口，并用五帧变化触发terminal/junction事件。先做98个关系端点的零训练AUC/分布审计；只有C01--C06拟合域和C07--C08选择域方向一致，才实现新head。

## 2026-08-28：最小结构触发纠偏已跑完，安全但没有恢复稀有拓扑端点

三种子正式训练已经完成：每个种子只训练129个结构置信度残差参数200步，原LiDAR/时序模型完全冻结。安全选择器最终都退回零残差，因此没有破坏现有模型，也没有通过“调阈值换召回”。C07--C08最终仍得到193个正确节点、4条正确执行边、零假回环，图安全合同全部保持；但两个低支持关系端点仍一个都没找回，所以该corrective科学FAIL，不能作为论文主方法收益。

失败并非单一的“样本少”。第一个低支持terminal的结构概率已经高达0.99，模型确实触发了节点，但把terminal错判成junction；只修改“是否触发”的logit永远无法修正类别。第二个terminal只有一个可见位置，结构概率仅0.36，属于结构可观测性或事件边界问题。两者需要不同证据，继续增加训练轮数或扩大结构head都没有科学依据。

当前下一步是只读可观测性审计，而不是再训练。它会在C01--C08中找出与这两类失败对应的所有端点，比较五帧出口布局、宽度/净空/坡度变化、事件类别margin和困难负样本。若类别信息存在但conditional head没学到，下一模型改事件类别表示；若单帧/五帧输入本身不可分，则修正时间采样或显式几何token。C09/C10和M-TARE继续隔离。

## 2026-08-28：训练修正已收缩为一个129参数的结构触发残差

现有模型其实已经把连续观测合成episode，并不是简单逐帧分类。但它仍按episode计权：一个短暂可见端点只有1个episode，一个大路口有6--8个episode，所以同一真实结构的训练权重仍差约6.2倍。正式接口审计在C01--C08两个分区都复现了这个阶梯。

因此不复制稀有帧，也不重训整个LiDAR网络。下一模型只在冻结的128维因果结构context上增加一个零初始化的线性residual，修正“是否触发结构节点”的logit；junction/terminal类别、三维中心、关联、执行资格和连边全部保持原样。训练目标仍包含原负样本与全部episode MIL，额外加入“每个关系端点identity等权”的episode损失。零初始化必须逐元素复现现有模型，保证改动可以单独消融。

C07--C08选择门槛固定为：至少恢复一个原来两个低支持端点；高支持端点、全体正确episode和false trigger不得退化；接回冻结图后节点/边precision保持至少98%，边召回不得低于当前4/13。任一条件失败就停止这个corrective，不改阈值或图规则。

## 2026-08-28：已证明训练样本权重是稀有拓扑端点漏检的系统原因

我们没有直接按C09失败个例调模型，而是回到C01--C08开发数据检验同一现象。结果很强：C01--C06中只有1--3条监督观测的端点，模型只触发30%，最终只有20%进入图；有11--30条观测的端点则100%触发并全部进入图。C07--C08中两个低支持端点一个也没触发，而高支持端点约91%能触发。这个方向在两个开发分区一致，预先规定的70个百分点差距门槛也远超10个百分点要求。

因此下一轮不是重做网络，也不是降低0.97阈值，而是修正训练单位：现有损失按观测行计数，同一个大路口可贡献几十行，短暂可见的终点或关系端点只有1--3行，梯度几乎被淹没。新corrective将保留原逐行事件任务，同时增加按真实结构identity和关系端点均衡的train-only batch，使每个端点有相近的学习机会。C07--C08只用于选择是否改善低支持召回且不损害整体precision；C09不再用于新版本选择，C10继续严格封存。

这项修正只针对“有没有产生正确结构proposal”。当前仍有提交和资格阶段损失，不能在同一次训练里顺便改图规则。先证明端点均衡确实恢复稀有节点，再以同一冻结图合同重放，才能判断是否增加真实拓扑关系。

## 2026-08-28：已知道拓扑关系为什么少，主因不是连边器

现在已经把8条C09客观关系的16个关键端点逐个追到底。16个端点中，11个曾被模型正确触发，8个形成了跨轨迹提交节点，5个通过执行几何资格并进入最终图。丢失的11个端点可明确分为：5个没有触发、3个触发但没有稳定提交、3个提交后被资格规则拒绝。最终7条缺失关系中，4条首先受“没有触发”限制，2条受“没有提交”限制，1条受资格规则限制。

这排除了一个容易走错的方向：不应继续改边组装器。原始回放的6条边中只有1条连接两个不同的真实结构并已被保留；另外5条其实把同一个真实节点的两个重复假设连在一起。边少的根因是关系两端的结构节点没有稳定生存到图中，而不是节点齐全后连线失败。

下一步回到C01--C08开发世界，审计“每个真实关系端点有多少训练窗口、当前模型是否触发、是否提交”。C09显示4个无触发terminal只有1--3条监督窗口，提示逐帧损失可能系统性忽略稀有端点；但不能直接据C09修改模型。只有开发数据也证明这种支持量偏斜与漏检一致，才预注册一次按结构identity/关系端点均衡的训练，同时保持普通事件损失和安全拒绝门不变。

## 2026-08-28：C09节点保持零误报，但关系端点覆盖不足

冻结的GSE执行端点图已经在C09完成一次正式验证。它生成74个结构节点，74个都能在4米内唯一对应客观结构；生成的1条边也正确，且没有错误回环。节点召回为56.92%，通过当前25%下限。这说明模型不是“没有学到结构”，安全节点也不是只在开发世界成立。

失败发生在拓扑关系覆盖：客观上有8条可观察关系，最终只恢复1条，边召回12.5%。对原始回放的复核显示，边生成器没有误删第二条正确边——原始阶段本来就只有1条真实边，其余5条原始边是同一真实节点的重复连接或错误事件污染。8条关系共16个关键端点，最终只恢复5个；其中1条关系两端齐全、3条只有一端、4条两端都缺。

因此下一步不改连边器，也不通过降门槛制造通过结果。我们会对16个端点做一次只读因果漏斗，确定它们分别停在结构事件未触发、跨轨迹未关联、未提交还是执行端点资格拒绝。只有根因明确后才在开发世界设计一次针对性修正；C09只用于验证失败归因，不用于选参数，C10和闭环继续隔离。

## 2026-08-28：开发图已通过，方法冻结后进入C09验证准备

执行轨迹端点几何关闭了最后3条开发集错误边。核心不是调一个更松的阈值，而是把图位置和结构证据分开：LiDAR模型决定“这里是否出现结构事件、事件大致位于当前隧道哪一端”；机器人完成穿越后，实际轨迹给出该端点的位置和各incident edge向外方向。多条边的端点必须在固定1米采样误差推导出的2米范围内汇聚；两条边若近似直通而没有观察到分支，junction保持延迟状态；只有三条已执行边，或两条向外方向点积为正的明确分支，才允许提交。

在C01--C06，最终图598个节点全部正确、13条边全部正确，节点/边召回为75.51%/36.11%；C07--C08为193/193节点、4/4边，召回70.44%/30.77%。两部分precision均为100%，固定25%召回门也全部通过。在线4米关联、模型checkpoint和“只有真实穿越才建边”都没有改变。

实验口径保持诚实：C07--C08参与了机制确认，因此这里只是开发容量；C09以前也用于旧方法validation，不能宣称全项目从未读取，但本次冻结的endpoint-geometry机制尚未用C09适配。下一步是在C09做一次冻结验证，只运行必要推理，不训练、不调阈值。C10仍是最终严格未见测试，只有C09通过后才允许读取。

## 2026-08-28：重复节点合并已让节点过线，现在只剩3条开发集错误边

已穿越隧道的“哪一端”可以作为不依赖真值的稳定拓扑标识。全部C01--C08结构观测中，这个标识没有一次对应两个不同真实节点；反向穿越会得到同一个端点标识，而同一隧道的另一端仍保持隔离。用它合并提交后的重复节点后，开发集节点precision达到`98.05%`，选择集达到`100%`，选择集恢复4/13条真实边且没有错误边。节点侧已经满足当前论文门槛。

这一步不能再训练一个duplicate classifier，因为当前端点候选只有`9/5`组正例、没有任何困难负例；硬训练会产生不可验证模型。我们保留确定性的执行端点合并，它只在真实穿越完成后工作，不扩大4m在线半径，也不预测未走过的边。

剩余问题已缩小为开发集18条输出边中的3条错误边，导致edge precision为`83.33%`。选择集4条边全部正确，说明核心方向可行，但不能忽略开发集安全缺口。下一步逐条检查这3条边两端的事件一致性、学习出口关系、place descriptor和执行端点证据，寻找一个在C01--C06可验证、对C07--C08只复核的fail-closed edge verifier；在此之前仍不进入C09或闭环。

## 2026-08-28：当前不是重新训练识别器，而是在解决“同一路口被画成两个点”

最近的正式审计证明，学习到的三维位置和三seed一致性已经能恢复大量旧关联器漏掉的结构观测；如果只接受由至少三条真实隧道共同支撑的路口，节点precision达到`99.83%`、边precision达到`100%`，说明安全建图容量成立。但C07--C08只恢复3/13条真实边，距离固定4条门槛只差1条。

关键的新发现是：两隧道支撑组里的错误大多数不是“模型把走廊看成路口”，而是同一个真实路口从不同入口观察后被提交成两个节点。若把它们拿去训练真假路口分类器，会错误压制真实结构。因此这条训练计划已经停止。现在转为hypothesis级重复合并：只有节点已经由真实穿越提交后，才根据共享的物理隧道关系、学习位置、place descriptor和出口token判断是否为同一结构；有歧义就不合并。

这一步仍然属于GSE-Graph核心创新：学习语义负责结构节点与关系，真实执行负责边，累计几何语义负责跨视角节点对应。下一项正式工作先证明这种重复候选有足够正负样本和family覆盖，然后实现fail-closed合并器并重跑同一客观评分。4m在线候选半径、edge assembler、C09/C10及M-TARE保持不变。

## 2026-08-28：评分尺子已纠正，空间语义图有效但还差可靠性闭环

我们现在按“预测节点在三维空间是否真的落到真实结构中心附近”评价图，而不再只看触发那一帧有没有Teacher identity。新的评分更严格：旧标量中心图只有153/179个节点正确，空间语义图有197/207个正确；节点F1从`0.675`提高到`0.819`，图综合F1从`0.397`提高到`0.543`。这进一步证明模型学到的几何结构位置确实改善了拓扑图，不是复制Cano出口规则。

但主方法仍未过论文门槛。10个错误提交中，6个是真实路口但中心误差超过4m，2个是同一路口的重复节点，2个把转弯/普通走廊误当路口。边生成器本身没有再丢边：13条真实关系里只有2条两端节点都恢复，这2条都正确建边。因此下一步不改“真实穿越才建边”，而是用三seed空间一致性恢复被拒绝的节点，同时用事件置信度和不确定性拒绝假节点，并合并可证明重复的节点。所有选择先在C01--C06完成，C07--C08只复核；C09/C10和M-TARE仍不读取。

## 2026-08-28：剩余拓扑损失主要来自旧关联器，而不是几何或边组装

正式漏斗审计逐个追踪了274个真实结构节点。新三维中心已把238个节点的跨穿越观测拉入4m范围，但其中31个仍被旧关联器拒绝；这是当前最大的单项损失。另有16个节点没有触发、12个中心仍超4m、7个只有一条穿越证据、10个在多候选/提交阶段停留。207个提交节点的精度损失仅由6个重复节点、2个无identity假节点和1个非decision identity组成。

边层没有独立算法故障：13条真实关系中，漏掉的11条全部因为端点未提议或未提交；只要两端节点成立，现有“真实穿越才建边”逻辑没有再丢边。因此不改edge assembler。下一步把新学习到的中心距离与seed不确定性作为关联特征，并用C01--C06同一世界、4m内、不同identity的真实混淆对构造困难负样本；先证明Teacher数量和family覆盖够用，不能直接训练。

## 2026-08-28：学习几何已显著改善拓扑，但节点安全与边召回仍未资格化

新的三维结构中心接回相同因果图后，C07--C08节点F1从`0.715`升到`0.823`，边F1从`0.118`升到`0.267`，综合F1提高`0.129`；正确覆盖的真实结构节点从162个增加到198个，输出的2条边全部正确。这说明“从LiDAR学习几何结构位置，再用它建图”不是无效创新，收益也不是来自扩大半径或重选阈值。

但当前207个提交节点中仍有9个重复或错误项，使节点precision只有`0.9565`；13条可观察真实边只恢复2条，edge recall为`0.1538`。因此离线图仍是科学FAIL，不能读C09或进入闭环。下一步只解剖现有失败：真实节点是否被触发、能否跨穿越关联、是否得到两条独立轨迹支持、最后是否被组装成执行边。审计后只允许修真正被证据定位的机制，不做阈值/半径搜索。

## 2026-08-28：GSE空间结构中心完整资格PASS，恢复离线建图验证

三seed纵向纠正已在冻结横向/高度空间decoder的条件下完成。C07--C08上，结构中心3D MAE由解析投影基线`3.752 m`降至`2.221 m`；同一真实结构从不同穿越方向观察时，预测中心落在4m关联范围内的比例由`52.40%`升至`80.29%`，相对中心误差改善`42.50%`。三个seed均改善，横向和高度输出逐元素不变，七项门槛全部通过。

这解除的是“模型能否从因果LiDAR学习稳定结构位置”的阻塞，不等于拓扑图已经通过。下一步将新中心接入原双层trace-commit图，在完全不改4m关联半径、提交逻辑、edge必须真实穿越等合同下重放C07--C08，直接检查重复节点、错误合并、node/edge F1和图连通性。C09/C10/M-TARE仍保持隔离，只有离线图通过后才允许前进。

## 2026-08-28：横向/高度空间语义已解决，下一步只修junction前向中心

正式残差审计证明当前空间decoder不是“差一点所以继续盲调”。把前向预测换成真值后，保留学习横向/高度的跨穿越4m一致率达到`99.993%`；反过来保留预测前向、把横向/高度换成真值，仍只有`61.519%`。junction分层只有`38.998%`，terminal已有`88.636%`。三种子median/medoid均不如当前平均，排除聚合器作为主要修复点。

因此下一方法严格缩小为一个空间条件纵向残差头：输入仍是五帧36方位格、2层垂直特征及冻结scalar前向值；现有spatial encoder、横向/高度输出全部冻结。只在C01--C06训练纵向direct与跨穿越一致性，C07--C08用原门槛选checkpoint。若仍不能通过，不再继续中心回归；C09/C10/M-TARE和图继续隔离。

## 2026-08-28：空间结构位置学习成立，跨穿越4m一致性尚差0.403个百分点

五帧range-image空间decoder已完成三seed训练。它保留36个机器人方位格和2层垂直特征，并冻结原纵向中心预测。C07--C08上左右MAE由`0.453 m`降至`0.209 m`，上下MAE由`0.582 m`降至`0.136 m`，整体中心MAE由`3.752 m`降至`3.516 m`，跨视角误差改善`13.97%`。这正式证明LiDAR空间feature map能学习旧pooled context缺失的结构中心位置。

但固定4m关联覆盖只提高`9.597`个百分点，未达到`10`点门槛，因此资格run仍为科学FAIL，离线图没有启动。当前不是从头重做，也不是模型无效；唯一剩余问题是少量同一节点跨incident traversal的中心离群。下一步只读分解这些离群，区分纵向、横向/高度和seed disagreement，再决定是否允许一次有根因依据的一致性纠正。C09/C10/M-TARE继续隔离。

## 2026-08-28：冻结action-set上下文无法学习横向中心，中心小头路线停止

局部3D V2已正式完成。纵向继承、相对误差和全局中心MAE均改善，但lateral/up输出没有任何可选epoch优于零初始化，4m覆盖增益仍停在8.95个百分点。运行无程序、数据或泄漏异常，因此这是有效的模型接口容量反证。

不再继续改小头、loss或阈值。下一表示直接从圆环LiDAR range-image的空间feature map解码机器人局部3D中心，保留方位布局而非只消费128D pooled action context；数据、Teacher、split和4m门槛保持不变。只有该专用空间decoder通过，才恢复离线图。

## 2026-08-28：scalar中心纠正停止，转向显式局部3D结构中心

双批次纠正保住并改善了单点MAE，跨视角误差也改善12.30%，三个seed一致；但同节点跨视角4m覆盖只增加8.95个百分点，距离固定10点门槛差1.05点，正式科学FAIL且没有运行图。该结果证明监督和采样已有效，剩余限制来自输出只允许沿route tangent移动。

下一接口直接预测机器人局部`forward/lateral/up`三维结构中心。现有route tangent定义forward，水平法向定义lateral，重力方向定义up，不需重生成LiDAR；旧scalar hidden层和longitudinal权重直接继承，横向/竖向从零开始。若3D接口仍不能过同一跨视角/4m门槛，停止事件中心回归路线并重新评估主表示。

## 2026-08-28：跨穿越监督产生正增益，但替代原采样导致回归退化

跨穿越配对模型三seed正式完成，程序和数据均无异常。相对中心误差改善8.08%、4m关联覆盖提高5.89个百分点，证明“同一结构不同穿越视角应对齐”的监督方向正确；但没有达到预定10%/10点，单点MAE还从3.520m退化到3.821m，因此按门槛科学FAIL且未启动图回放。

失败来自训练批次而非新数据不足：identity-balanced pair batch替代了原row-balanced direct batch。下一次有界纠正从已通过V1R3权重初始化，同时保留两种独立批次；checkpoint只有在MAE不劣于V1R3时才按跨视角误差选择。若仍失败，停止当前scalar offset头，不继续改loss权重。

## 2026-08-28：结构中心学习PASS、部署图明显改善但仍未资格化

三seed结构中心偏移头已经训练完成，旧LiDAR/action-set主干完全冻结。C07--C08中心偏移MAE从`5.496 m`降至`3.520 m`，改善`35.95%`，说明LiDAR因果表示确实学到了“结构中心在机器人前后多远”，不是规则坐标回填。

把该输出接入在线图后，重复提交明显减少：节点从293降至179，节点precision由61.77%提高到90.50%，node F1由63.84%提高到71.52%，错误loop merge仍为0。但离98%安全要求还有差距，边precision/recall只有25.00%/7.69%，因此尚不能进入C09或闭环。

失败分解显示16个真实junction仍被拆开，其中13个因预测中心相距超过固定4m。初次客观中心上界误用了masked普通走廊行的`(0,0,0)`填充值，凭空聚集出9个假节点和2条伪边；该诊断结论已撤销。有效结构行用客观中心、无效行保持部署坐标后，node precision/recall=`98.81%/90.88%`、edge precision/recall=`100%/46.15%`、false loop=0，证明图逻辑本身有资格容量。当前唯一下一步是加入同一traversal的运动/时序一致损失来提高学习中心精度，再重做冻结部署图审计。

## 2026-08-28：trace提交有效但关联坐标错误；下一主任务为学习结构中心偏移

双层图首次正式C01--C08 proof为科学FAIL。执行验证确实把直接ghost图节点precision从42.12%提高到61.77%、节点F1从57.92%提高到63.84%，并将false loop merge降为0；但同一真实结构仍被重复提交，C07--C08 293个节点只对应181个唯一节点，边仅恢复3/13，总体提升未达到5个百分点。

原因已被定位为方法接口缺失：当前学习输出含事件、轴线、宽高、坡度、曲率、descriptor和exit tokens，却没有结构中心相对sensor的位置；从不同incident tunnel观察同一节点时，sensor poses天然可能超过4m，安全关联器只能拒绝并造成node split。教师纵向中心投影的只读上界达到99.76%候选对纯度、98.38%提交节点precision和88.69%recall，证明新增目标有直接可验证价值。

当前Phase仍为3。唯一NEXT是生成C01--C08 signed event-center offset监督接口，并训练小型回归头；C01--C06拟合/选择、C07--C08一次验证，旧action-set+sensor-pose图作为消融。C09/C10、正式topology replay、M-TARE和planner仍不读取。

## 2026-08-28：直接节点生成路线停止，主方法重构为语义假设图与执行提交图

route-conditioned结构化出口计数在C01--C06的900组固定配置中`0`组安全通过；最低召回条件下最优精度仅84.85%。因此当前冻结出口token既不能通过黑盒分类器，也不能通过显式计数规则直接提交最终junction/terminal节点。C07--C08未用于结果选择，C09/C10/M-TARE仍为0读取。

当前证据仍保留三项有效组件：连续几何学习显著改善、Factorized route-conditioned association在C09安全PASS、trace-verified edge合同已实现。下一方法不再要求单帧/短序列感知直接决定最终节点，而是维护两层：学习语义只建立provisional结构假设；机器人真实穿越、反向观察和incident action一致后才提交verified node/edge。假设错误计入效率，提交错误仍受1%安全门约束。

这一路线与NTS ghost node和传统多假设拓扑SLAM存在邻近，论文创新只能落在“因果3D LiDAR显式隧道几何语义 + route-conditioned关系 + trace-consistent提交”的完整机制及其闭环收益，不能泛称首次提出学习拓扑假设或执行验证。

## 2026-08-28：学习式出口集合有结构信号，但黑盒节点分类仍不安全

三seed轻量action-set模型已完整训练，原LiDAR/出口模型没有更新。正式纠正评估在C07--C08得到49.82%事件召回和十个family覆盖，但满足最低召回时总精度只有96.59%、假节点3.41%；junction是主要缺口，precision仅95.20%。因此不能读取C09或建立完整图。

这不是“完全没学到”：terminal precision/recall=`98.71%/90.16%`，junction/terminal身份覆盖=`69.86%/94.53%`。失败说明单纯把六个出口token集合压成一个节点类别仍丢失了机器人从哪个入口到达的route condition。

当前Phase仍为3。唯一下一步是零训练结构化容量审计：用在线可得的来向显式扣除incoming action，再由剩余学习式出口数量/几何和五帧稳定性决定junction/terminal。保持C01--C06/C07--C08隔离及原安全门槛；C09/C10/M-TARE继续0读取。

## 2026-08-28：事件分类式节点门已停止，转向学习式出口集合直接生成节点

预注册的5帧fallback没有读取C09：它在C07--C08扫描全部1,001个固定阈值后，找不到同时满足开发域安全裕量和最低两类召回的配置。因此不能靠把旧事件模型换阈值解决12帧的2.46%假节点问题，categorical node gate正式停止。

论文主线不变且更明确：junction/terminal本质是机器人可执行出口集合的结构。下一候选直接消费已经学习出的exit tokens（方向、宽度、垂直轮廓、descriptor、confidence）及其因果稳定性来生成节点；连续宽高坡度曲率仍属于已穿越edge，冻结关联仍负责安全合并。这不是从头训练，也不是退回规则图。

当前只允许C01--C08只读action-set容量证明。若学习式token集合不能在开发隔离域满足节点precision/false/recall合同，则停止该节点生成主张并重评论文方法；C09、完整图、C10和M-TARE继续禁止。

## 2026-08-28：关联已通过，但12帧节点生成未通过1%假节点预算

Factorized关联仍保持正式PASS；当前阻塞已经前移到“哪里生成junction/terminal节点”。新部署接口证明24,462条C09序列全部只消费当前及过去LiDAR，Teacher事件和identity在推理阶段读取为0，turn和geometry-transition不会生成节点。

12帧三seed触发器在C09得到487个节点触发、475个正确，整体precision/recall=`0.9754/0.8796`。junction和terminal的identity coverage达到`92.96%/91.53%`，说明模型确实学到结构语义；但12个错误触发使false fraction=`2.464%`，超过论文固定的1%，因此科学FAIL，不能进入完整图回放。

V1只因冻结输出archive行顺序假设错误而system FAIL；V2按global sequence ID和parent ID双射修正后完整执行，未改变数据、模型、阈值或门槛。下一步自动采用预注册fallback：在C07--C08选择5帧junction+terminal专用概率门，冻结后一次应用C09；若仍不安全，停止当前节点分类路线，不用图或规划器调参掩盖。

## 2026-08-28：学习式节点关联已通过，下一步验证整张离线拓扑图

Factorized GSE-Graph的关联阻塞已经解除。C07--C08只选择得到`2-of-3`模型共识和`4.0 m`运行候选距离约束；冻结后在C09达到balanced precision/recall=`1.000/0.508`，真实runtime precision/false-loop/recall=`0.9901/0.00987/0.5583`，满足原始安全与非空召回门槛。

结果来自一个不可覆盖的两进程正式run：选择进程无法访问C09，验证进程只在选择完成后读取封存C09 scores；22项seal完整、输入未变、C10和M-TARE均为0。原单seed失败仍保留，说明多模型一致性和metric locality都是必要机制，而非选择最好seed。

现在还不能宣称拓扑图已经完成。下一步是在Phase 3中让这些语义输出真正生成/关联decision nodes，让edge只由真实穿越提交，并检查node/edge F1、false loop、连通分量和cycle rank。该离线图资格通过后，才允许进入M-TARE闭环；C10仍保持严格隔离。

## 2026-08-27：C09关联安全资格FAIL，进入最小鲁棒聚合可行性审计

最终统一坡度模型在C01--C08容量很强，但单seed安全性没有迁移到C09：balanced和真实runtime两域都出现超过1%的false merge。当前不能建立论文主张所需的可靠拓扑图，离线图与闭环已停止。

失败证据完整且可用于论文：三seed仍有25.38%--85.11%召回，但复杂近邻节点会被相关性误合并。下一步不训练新模型、不用C09调阈值，而是在C07--C08 selection域预注册选择三seed共识数和metric distance cap；若selection域或再次冻结C09验证失败，正式停止当前Factorized association路线。

## 2026-08-27：Gate 3统一坡度关联容量PASS，进入C09未见拓扑资格准备

最终感知和关联现在使用同一套坡度语义。V1R证明三套统一观测只改变坡度列，所有非路线对照保持逐字节一致；完整路线条件关联在三seed上precision均不低于`0.99537`、false accept最高`0.463%`、recall为`41.97%/78.47%/54.01%`。

这仍是C01--C08开发容量证据，不是在线拓扑图PASS。当前唯一下一步是使用已经冻结的V1R模型、归一化和阈值执行一次C09只读资格，同时检查identity-balanced结构混淆与真实过去候选。任何失败都停止进入图回放；C10、M-TARE和闭环保持禁止。

## 2026-08-27：Gate 3统一坡度接口修正准备中

当前没有从头训练主模型，也尚未读取C09。已定位容量V1与最终感知合同之间唯一的集成缺口：route-conditioned关联使用的坡度仍来自旧GSE回归列，而论文最终坡度来自已冻结的physics-guided corrective与`0.89`风险校准。

只读证据已确认C01--C08全部`188,126`条五帧序列可用最终坡度一一替换，原CUDA checkpoint在C07--C08逐元素精确复现，且统一观测仅第10列改变、其余145列逐字节不变。接口模块、builder、单次runner和专项测试已经实现，`9/9 PASS`。

当前唯一下一步是冻结Data Card/spec、通过preflight并执行一个不可覆盖的V1R容量run。它保持原数据、Teacher、模型结构、优化、阈值和验收标准不变；只有V1R继续满足安全精度与route gain后，才进入C09未见拓扑资格。

## 2026-08-27：class-mass审计否定equal-class MIL，categorical detector路线停止

- 类别数量确实不平衡，但最终解析gradient mass没有压低transition：transition占`28.2380%`，高于junction的`18.0855%`；turn占`52.4928%`。equal-class会把transition/turn梯度推到`72.3886%/25.4180%`，不是安全的小修复。
- 正式audit为科学FAIL、`error=null`，optimizer/model inference/C09/C10/strict/M-TARE均为0；12项seal SHA=`5b888c284f9d5c808c96dee4de6a7964ecaff02beedb4da0fa8f57da13e8953f`。
- 因此不执行class-balanced V2，不再扩大五类事件分类器。下一阶段仍在Gate 3，只允许方法/新颖性审查和C01--C08只读可行性proof：把已经学会的连续几何signature用于因果分段/change-point，再由真实traversal验证边。

## 2026-08-27：Gate 3十二帧episode detector V1R科学FAIL

- 三种子完整运行，`error=null`，39,996次detector更新、0次backbone更新、0次C09/C10/strict/M-TARE读取；35项seal复核通过，SHA=`bf49586ddd57a5a78ee248a36ae7a5505eefbf44fa5cb492d6b5ff5b1b62e599`。
- 连续事件压成单trigger的执行逻辑通过安全行为：precision=`0.991437`、episode recall=`0.771852`，junction/terminal identity coverage通过；但条件结构语义失败，macro-F1=`0.654245`、turn=`6/95`、transition=`0/17`。
- V1仅作为system failure保留：NumPy dtype在第一个optimizer step前被错误传给PyTorch；该run没有科学结论，也未被V1R复用。
- 当前仍停留Gate 3，禁止拓扑图与planner实验。唯一下一步是对sealed V1R输出做只读class-loss/gradient-mass审计；只有证明稀有类别被目标函数压制，才允许同数据、同容量、同门槛的class-balanced MIL修订。

## 2026-08-27：12帧数据引用已全量证明，模型实现保留方向布局

- 正式V1R证明新输入不需要新数据：188,126条观测的1,818,662个past-only引用全部在同一parent和同一traversal内解析，81,486条具完整12帧，早期观测以5--11帧左填充mask表达。
- 完整traversal口径明确为`16,078 total / 16,076 observed / 2 zero-observation`。原V1因把total等同observed而FAIL，引用内层结果仍正确；V1R array digest与V1完全一致，排除方法或数据静默变化。
- 12帧detector代码现在同时消费每帧128D pooled feature与36-bin环形方位feature。该设计直接吸收旧directional head对turn有益、全局平均event head失败的证据；只使用pooled特征的版本保留为必要消融。
- 当前下一项是实现训练缓存和episode-preserving batch sampler，随后冻结三seed Data Card/spec。V1R 12/12 seal SHA=`12590398...9d124`；C09/C10/M-TARE仍为零读取。

## 2026-08-27：根因确认为frame/episode监督错位，主方法转为12帧因果事件检测

- 正式监督审计PASS：当前Teacher共有`5,306`段结构episode，change-point的`1,031`个正标签中`86.42%`位于完整确认前、`60.23%`的5帧历史已不含回投边界。逐帧macro-F1作为主门槛会惩罚一个本来只需在整段中正确触发一次的拓扑事件。
- 现有数据足够支持修复：全部`152/152`方向change confirmation均已有12帧过去LiDAR，12帧来源是已封存unique-frame引用，不需要重建世界、重发射ray或扩大测试暴露。
- 冻结的新接口是：帧级proposal保持provisional；masked 12-frame past-only decoder负责confirmation；episode bag内至少一次正确类别触发；确认后回投节点位置；edge仍只能由真实traversal建立。身份/TNG/未来帧只用于Teacher和评价，禁止进入学生输入。
- 当前仍为Gate 3表示修订，C09、在线图和闭环继续停止。审计18/18 seal SHA=`781ea53c...b8b7d9`，图像已目视复核。下一工作是先实现episode类型、12帧引用物化、MIL loss和因果状态机合同，再建立训练Data Card/spec。

## 2026-08-27：低容量容量证明失败，表示学习退回事件监督/因果输入设计

- 固定多lag几何风险读出器已完整运行并科学FAIL：C07--C08 event macro-F1=`0.610400`，transition precision/F1=`0.036070/0.069420`，turn F1=`0.256124`；没有非空阈值满足开放集节点安全合同。
- 运行本身可信：拟合收敛，`error=null`，16/16证据哈希匹配，零C09/C10/strict-test/M-TARE读取，输入封存前后完全一致。失败属于model/target organization，不是资源、程序或数据漂移。
- 当前仍停留Gate 3，不能建立论文主张所需的可靠结构节点，也不能进入C09拓扑回放或闭环。已学到的连续几何回归、junction/terminal和exit-token组件继续保留为方法组件、基线和消融。
- 当前唯一工作是只读审计逐帧Teacher与持久事件episode的时序错配，并冻结一个真正的因果事件检测合同。禁止继续增加分类器容量、选择seed、放宽1%错误预算或调图参数。

## 2026-08-27：根因已收敛到历史长度与风险组织，开始最后一次容量证明

- 正式只读审计证明较长过去历史有用：冻结几何变化AUC由4m的`0.6802`升至10m的`0.7645`，Teacher在10m为`0.8515`；这与Teacher约7--10.5m的因果确认延迟一致。
- 失败不是旧标签残留：10m时101个高风险corridor只有10个来自删除的旧transition，91个是普通corridor；所有单一lag仍为`0/17`安全覆盖。因此不能靠选lag、手工阈值或恢复旧Teacher建节点。
- 最后一次允许的表示容量proof采用固定2/4/6/8/10/12m多变量几何delta、原模型条件事件证据和不确定性，训练一个确定性低容量五分类风险读出器；位置、world、identity、TNG和未来帧严格禁止作为输入。它必须同时恢复transition、turn和宏观事件质量，而不是只优化一个数字。
- 审计正式状态PASS、18/18 seal SHA=`56f6b82c...e6269`，图像人工复核无裁切。capacity proof的执行器、runner、Data Card freezer与14项合同测试已就绪；正式结果未产生前仍不得进入C09或拓扑回放。

## 2026-08-27：有界主干适应也未恢复结构节点，当前转向机制审计

- 最后编码块三seed正式run已科学FAIL而非系统失败：ensemble几何delta MAE=`0.348158`，较冻结基线改善`40.71%`；但事件macro-F1=`0.680096`、change-point=`0/17`、turn=`23/95`，未达到预注册节点语义门槛。三个seed都在epoch 6结束，未出现NaN/OOM，且参数/梯度边界完全符合合同。
- 该结果证明问题不是“冻结得太多”这么简单：模型稳定读出连续宽高坡度曲率变化，却仍不能在1%错误预算下决定哪个变化应成为拓扑节点。现阶段不能进入C09、离线图或闭环，也不能通过选seed、降低阈值或扩大主干训练掩盖。
- 正式run为`FAIL_GSE_CAUSAL_GEOMETRY_DELTA_LAST_BLOCK_TRAINING_V1`，`52,236`步、C09/C10/M-TARE读取0，26/26 seal复核PASS，seal SHA=`27c9ec4c...03d53`。
- 当前唯一工作是零训练因果风险冲突审计；其执行器、runner、freezer和多时间尺度低容量风险接口已实现，相关纯函数/合成合同`14/14 PASS`。审计会先正式判断较长历史是否有实质信号及标量风险为何不安全，再决定是否允许真实数据风险proof。

## 2026-08-27：几何变化已经学到，结构节点触发仍未学到

- 正式多任务训练证明LiDAR中连续几何信息可学习：C07--C08上宽度、高度、坡度、曲率变化的归一化MAE由`0.587167`降至`0.355174`，ensemble改善`39.51%`，三个seed均改善超过40%。这部分可作为论文方法动机和后续edge几何属性证据。
- 但它尚不能建立可靠拓扑节点：ensemble事件macro-F1=`0.683669`，比既有baseline低`0.004235`，geometry-transition F1=`0`、change-point identity coverage=`0/17`。因此当前不能进入C09拓扑图或闭环。
- 当前唯一下一步是预声明的最小回退：只允许原LiDAR编码器最后一个残差块适应几何变化/事件联合目标，其他旧网络全部冻结。正式FAIL run无系统错误、`52,236`步完整，26文件seal SHA=`aed1d39e...47e4`。
- 组件成功/事件失败已形成可投稿失败分析图包并写入正文Figure 8；6个绘图资产逐哈希一致，manifest SHA=`e8e24f34...2f2948`，因此该阶段即使不产生最终主方法也保留为可复现消融证据。

## 2026-08-27：五帧LiDAR中存在可迁移几何变化信号，已允许显式多任务学习

- 正式只读审计精确覆盖80个C01--C08 worlds、188,126 observations和122,765个有效四步变化样本；Teacher与冻结三seed平均预测的C07--C08 AUC分别为`0.742067/0.680215`，预测AUC的fit-selection gap仅`0.008180`，全部达到预注册可观测性门。
- 这不是事件检测PASS：固定约1% corridor FPR时，冻结几何标量对17个选择域change-point身份覆盖仍为0，说明现有表征包含信号，但需要显式delta监督、事件联合损失与不确定性拒绝才能转成可靠节点。
- 当前唯一下一步是训练`Δwidth/height/slope/curvature + structural event`多任务decoder，仍只用C01--C06拟合/C07--C08选择。C09/C10、拓扑回放和planner继续冻结。正式证据18文件seal SHA=`c7578762...fa50`，论文动机图三格式和源数据已保留。

## 2026-08-27：新Teacher有效，但冻结表征上的分类头无法学习change-point

- 正式三种子训练完整、可复现且科学FAIL：ensemble macro-F1=`0.680608`，相对旧directional corrected-label baseline变化`-0.007296`；change-point身份覆盖`0/17`，turn=`15/95`。开放集precision=`99.01%`、false accept=`0.989%`、recall=`67.81%`，证明节点安全门本身可工作。
- 三个seed的change-point覆盖均为0；36,000步、0 backbone update、0 C09/C10/M-TARE、`error=null`，26文件seal SHA=`4ca5aa25...17e5e`。结论只否定“冻结backbone+纯事件分类头”，不否定corrected Teacher或完整GSE-Graph主张。
- 当前唯一研究问题改为：把连续几何变化作为显式学习目标，是否能让五帧LiDAR表征跨开发世界泛化，并进一步转化为安全的持久结构事件。下一步先做causal geometry-delta observability/capacity proof，再决定冻结特征回归头或主干微调，不进入图或planner。

## 2026-08-27：corrected causal Teacher完成，模型学习成为唯一问题

- V1R正式状态`PASS_GSE_CORRECTED_CAUSAL_TEACHER_MANIFEST_V1R`：80 worlds、16,078 traversals、188,126 observations、1,031 causal change-point labels、76 change-point identities、1,534 total structural identities和74,064 pairs全部精确。
- 稀疏global ID逐元素保留，源/输出digest一致；22/22科学/系统检查与17/17完整seal均通过，seal SHA=`3f840e11...7f49`。V1错误checker的FAIL证据继续保留，不覆盖。
- 当前操作阶段返回Gate 3。唯一研究问题是：identity-balanced、5帧因果LiDAR结构事件模型能否在C07--C08学出新的持久change-point，同时保持约1%开放集节点错误合同；在此之前不读取C09/C10、不进入图回放。

## 2026-08-26：corrected Teacher V1仅因全局索引验收写错而FAIL

- 实际输出188,126条，`global_sequence_index`唯一、无重复、严格沿sealed源顺序；但源索引属于包含C01--C10的212,588条总清单，C01--C08子集自然保留C09/C10的20,102个空槽，范围为`0..208227`，不能等于错误预期的连续`0..188125`。
- 其余全部精确通过：1,031新标签、76新identity、59条节点优先级抑制、各事件/identity数量、74,064个pair、80 worlds/16,078 traversals，以及C09/C10/M-TARE/model/training零读取。失败分类为system/metric checker，Teacher科学结果未被否定。
- 推荐V1R保持所有输出值不变，仅把索引合同改为逐元素等于sealed源train子集并要求唯一/严格单调；备选重编号会破坏稳定引用，不推荐。V1永久保留FAIL，未作静默重跑。
- 只读代码/配置审计表明下游已经统一使用`compact_to_global_sequence_index`或键值映射，不要求稀疏ID直接充当数组行号；因此修复边界只涉及V1 executor与对应Data Card/spec/checker，不需要改模型、训练器或拓扑回放。

## 2026-08-26：操作阶段暂回Gate 2生成corrected causal Teacher

- 当前只执行一次不可覆盖的Teacher生成：80个C01--C08世界、16,078条directed traversals、188,126条观测；保留junction/terminal/turn，移除旧transition语义并应用已封存的1,031条因果标签和76个结构identity。
- 主方法是方案A的持久、双向一致、严格因果change-point；旧Teacher是对照，失败时不改阈值、不换数据、不重跑。预期证据为完整观测、74,064条关联pair、identity清单、逐标签优先级审计、环境/命令日志和SHA-256 seal。
- Gate 3历史结果与C09失败证据不变；C09/C10/M-TARE/模型/训练均不读取。Teacher PASS后才返回Gate 3进行identity-balanced结构语义学习。

## 2026-08-26：几何变化节点方案A定义证明PASS，进入corrected Teacher生成

- 当前研究结论：geometry-transition可以保留为拓扑节点，但必须使用持久、双向一致、过去证据确认并回投边界的change-point；旧短帧transition Teacher正式降级为失败基线，不再进入主方法标签。
- C01--C08完整proof将旧3,035个identity收敛为76个稳定change-points和1,031条最终优先级一致标签，全部10个family非空；fit/selection identity=`59/17`。正式run seal SHA=`54bdc815...f60`，论文图manifest=`95d29a69...2b79`。
- 当前唯一研究问题：用新1,031条标签重建的Teacher和identity-balanced结构事件模型，能否在C07--C08保持约1%节点错误合同并恢复change-point identity coverage；在此之前不重放C09、不读C10、不启动拓扑/闭环。

## 2026-08-26：操作阶段暂回Gate 3训练稀有事件corrective

- 冻结三seed特征、60个C01--C06 fit世界和20个C07--C08 selection世界不变；训练仅更新轻量292→128→64→5残差事件头，backbone update=0。
- C09 V4失败证据保持有效，C09/C10/M-TARE继续0读取。选择域corrective未PASS前不回到离线图。

## 2026-08-26：已学会岔路/尽头，尚未学会转弯/几何变化

- 三seed事件×节点融合在C07--C08正式PASS：99.00% precision、0.995% false accept、49.39% frame recall，seal SHA=`509cbe125cf8ecfa3ba93370fd161e71155af0a66eaf475e78e7fd7b24e15ca1`。
- 但C09唯一地点覆盖揭示类别断层：junction/terminal约`93.0%/96.6%`，turn/geometry-transition仅`3.8%/1.5%`。后两类占Teacher节点主体，因此离线拓扑仍0个安全配置，不能进入C10或闭环。
- 当前唯一研究问题：在冻结LiDAR主干和出口关联的条件下，身份/episode平衡的事件corrective是否能真正学出turn与geometry-transition，而不是只提高同一批junction/terminal帧的置信度。

## 2026-08-26：重复建点已修正，但结构事件召回仍阻塞论文主图

- V4正式结果完整且为科学FAIL：四方法按预声明在10个C09 world上执行，主方法243组无安全配置；305/305 seal哈希通过，SHA=`cdb98789d8b98a1d15007f94f72379f1c0814c4fc8288ab59777cacbe9e44f8d`。
- 连续事件episode修正后，图不再在同一岔路附近周期性重复建点，连通分量误差归零、cycle signed bias降到约`0.20`；但81组完整只读检查仍无安全配置，最佳节点召回仅`18.69%`，node/edge F1约`0.291/0.055`，最接近安全的回环错误率仍为`2.06%`。
- 当前唯一研究问题：三seed因果结构事件融合能否在不使用GT身份、未来帧、C10或M-TARE的情况下提高唯一结构节点召回，同时保持节点生成和关联的开放集安全。不能则停止当前事件头路线并重新设计结构事件表示。

## 2026-08-26：开放集节点门开发域PASS，等待完整C09 corrective

- 节点门已冻结：三seed V2 matchability等权均值，阈值`0.982292910416921`；C07--C08 precision/false accept/recall=`0.990056/0.009944/0.349010`，10 family全部通过，零C09/C10读取。
- C09单配置只读原型把系统false merges从48降到14并达到precision `0.98450`，证明纠正方向有效，但false-loop `1.55%`仍不能宣称安全；必须跑完243组以及三类基线。
- 下一唯一问题：带冻结节点门的完整GSE是否存在预注册安全配置，并同时达到node/edge F1增益和图不变量合同。

## 2026-08-26：C09完整在线图未通过；pair verifier可用但节点生成开放集失败

- 当前结论：GSE-Graph尚不能进入C10或闭环。正式V3在全部243组图配置上都未达到系统级98% precision / 1% false-loop合同；正式FAIL与seal均已冻结。
- 关键分解：最佳安全方向配置的48次false merge中，只有8次是两个真实结构节点直接错配；36次源于corridor被模型误判为稳定结构事件，4次源于候选节点身份污染。结构identity条件下的关联约99.09%，所以继续提高pair threshold不是正确修复。
- 唯一下一研究问题：三seed结构事件的开放集一致性判定能否在不使用C10、未来帧或GT identity的情况下阻止corridor假节点，并让完整图同时满足关联安全、node/edge相对基线增益和图不变量。
- 正式证据：`results/gate4_topology/gate4_20260826_gse_offline_topology_validation_v3_seed0`，status=`FAIL_GSE_OFFLINE_TOPOLOGY_VALIDATION_V3`，seal SHA=`a13a837a72b3ed0bc142b1601c578fbb5c0bfd70241d76a708d5cc62d25b4d6e`。

## 2026-08-26：三seed等权exit-token关联已在真实线上候选域正式PASS

- 新只读immutable calibration以13文件seal正式PASS，冻结`distance<=16m`、三seed等权mean和threshold `0.9431912303`；P/false/R=`0.990054/0.009946/0.287442`，全部10 family通过。
- 该PASS没有改写V1R/V2单seedFAIL，没有训练、模型推理或C09读取。它只解除新的ensemble关联接口进入C09离线图资格。
- 当前下一步是实现并预注册C09离线拓扑V3：模型/验证器/阈值/radius全部冻结，只在10个C09 world评价在线node/edge、错误loop、component与cycle rank，并与exit-only规则图和非学习几何事件图比较；失败则不读C10、不进闭环。

## 2026-08-26：V2单seedFAIL，线上候选域内ensemble出现可验证PASS候选

- V2完整使用逐出口descriptor/heading/width/profile并在三seed上均降低高召回错误，但没有单seed满足`<=1%`，正式run保持科学FAIL；40文件seal SHA=`7864f8c4...`。
- 阈值评估器遗漏了已写入合同的16m在线候选过滤：selection全量45,372 pair中13,903条距离大于16m，线上不会成为候选。该缺陷使V1R/V2的旧阈值结论不能作为线上资格，但不改变其原spec下的FAIL状态。
- 在31,469条真实线上候选上，V2三seed均值达到precision `99.0054%`、false merge `0.9946%`、recall `28.7442%`且全部10类非空安全；V1R均值仍失败。推荐新建只读ensemble calibration run正式复核并冻结，成本为分钟级、0训练/0C09。

## 2026-08-26：open-set V1R科学FAIL，下一步补齐真正的exit-token集合关联

- V1R修复了唯一system错误并正式证明pair/feature按稀疏global identity精确对齐；三seed均完整训练，旧run不复用，C09/C10/M-TARE和backbone更新全部为0。37文件seal SHA=`dd4c0b60...`。
- 方法从旧在线图的`51.43%`显著提高到25%召回附近的`97.59%/98.10%/98.01%` precision，但错误合并仍为`2.41%/1.90%/1.99%`，未达到固定`<=1%`；因此离线拓扑、C10和闭环继续停止。
- 失败不是数据量、编号或资源问题。当前verifier只消费出口置信度的五项汇总，遗漏冻结GSE已经输出并受监督训练的逐token heading/width/vertical profile/32D descriptor，导致相似总体几何难以区分。
- 推荐下一步仍属于用户选择的方案A：不重训803k参数GSE主干，只新增旋转/置换不变的exit-token set matcher，并在同一fit/selection pair上进行hard-negative-aware校准；保持原安全与召回门槛，不以三seedensemble的post-hoc `1.204%`错误率冒充PASS。

## 2026-08-26：方案A代码合同完成，pair容量口径发现阻塞性不一致

- 新增开放集关联类型实现：单观测matchability、左右严格对称的pair verifier、只使用部署可得GSE输出的特征，以及禁止全拒绝的总体/逐family阈值合同；10/10专项测试通过。
- sealed C01--C08真实重放否定旧只读inventory中的`79,478/25,360`条“16m严格过去”负对。正确3D严格过去计数为`57,066/18,111`；取消过去限制也只有`71,277/22,843`，说明旧统计的距离/时序语义被错误记录。
- 推荐修复是不改物理定义：正式采用同parent、严格过去、3D欧氏距离`<=16m`，总fit/selection pairs为`135,232/45,372`。该容量仍充足，且最接近在线图真实可用信息；另一选择是先找回并正式定义旧统计使用的未知距离口径，成本更高且有离线/在线错配风险。
- 当前没有material run、训练、C09/C10或M-TARE读取。必须先冻结正确pair口径，之后才能生成Data Card/spec并训练三seed verifier。

## 2026-08-25：open-set corrective数据容量充分，方法决策仍未执行

- C01--C06/C07--C08只读元数据盘点确认`102,874/32,249`条无结构身份open-set观测、`39,310/13,693`条identity-valid观测和`3,380/1,113`个完全不相交身份，覆盖全部10个topology family。
- 保留现有`78,166/27,261`条正例与different-identity困难负例，再为每个16m内open-set query确定性配一个严格过去结构候选，可形成`157,644 fit / 52,621 selection` pair。最小8m范围也有`59,628/18,992`个open-set困难负例。
- 该证据排除“open-set样本不足”，但尚未证明现有冻结特征可在实质召回下达到98% precision。未获得方法决策前不实现verifier、不生成冻结模型、不重跑C09。

## 2026-08-25：离线拓扑V2在学习关联安全门槛正式FAIL

- 唯一正式run `results/gate4_topology/gate4_20260825_gse_offline_topology_validation_v2_seed0`完成全部243组完整GSE扫描后，以`FAIL_GSE_OFFLINE_TOPOLOGY_VALIDATION_V2`停止；12/12 seal复核PASS，seal SHA=`4883061636c365f7210b9c5ae2f3993ecd667327a69b6ec41a2a67049b6452b3`。运行`1083.23s`、峰值child RSS `1,226,636 KiB`、零模型更新、零C10/M-TARE读取。
- 243组中没有一组满足在线association precision `>=0.98`与false loop `<=0.01`。最高precision仅`0.514286`（36 correct / 34 false），因此按预注册选择器在第一方法后立即停止；其余三个方法和node/edge科学比较没有执行，不得从partial sweep推断主方法相对基线结果。
- 对最高precision配置的三seed逐决策只读审计显示70次merge中36次正确、34次失败；34次失败的目标均没有稳定GT结构身份，32次查询本身是普通位置被误触发为结构事件，2次有身份查询匹配到先前误触发建立的无身份节点。没有观察到“已有真实结构A合并到已有真实结构B”，但错误开放集节点仍会污染图，不能改写为安全回环。
- 根因是model/calibration contract缺失：place descriptor训练与阈值校准都排除了`association_valid=false`观测，因而没有学习或验证open-set rejection。仅把这些负样本加入阈值可得到三seed`46/3/2`个零假接受正确匹配，但召回约`0.998%/0.064%/0.041%`，属于近乎全拒绝，不能作为论文修复。
- 当前停止离线图、C10与闭环。推荐新建train-only open-set association verifier：冻结GSE backbone，用C01--C06拟合、C07--C08选择，将无结构身份查询和相似平行隧道作为显式困难负样本；先证明非空高精度且有实质召回，再重新执行新的C09完整拓扑资格。该方法变化需明确研究决策。

## 2026-08-25：风险校准感知正式PASS，执行位置进入离线拓扑V2

- 唯一V2正式run `results/gate3_semantics/gate3_20260825_gse_risk_calibrated_perception_validation_v2_seed0`已完成为`PASS_GSE_RISK_CALIBRATED_PERCEPTION_VALIDATION_V2`；38条seal覆盖除seal自身外的全部文件，seal SHA=`5194dd6c489a31bdab0998d1cec1a0df27b677668afd04b8c128aebf26e0478e`，零optimizer/model update、零C10/M-TARE读取。
- 共享残差scale由20个C07--C08 worlds、45,942序列的固定101值网格选为`0.89`。C09三seed坡度MAE=`0.779108/0.765864/0.760941°`，均值`0.768638°`，相对五帧先验均值改善`47.042%`；最差S02退化`4.319%`，满足原逐世界最多5%合同。
- 完整感知门槛同时PASS：事件macro-F1平均增益`+0.154469`，关联precision约`0.990`且false accept不超过`1%`；宽/高/坡度/曲率相对非学习估计分别改善`24.071%/51.998%/49.675%/85.307%`，四字段平均`52.763%`。
- 论文图`gse_perception_validation`与`gse_slope_risk_calibration`已按PNG/PDF/SVG/CSV/JSON/provenance/SHA-256合同发布并目视复核。C09属于暴露过V1失败的development validation；最终泛化结论仍只能来自未读取的C10及后续sealed闭环世界。
- 当前NEXT：冻结并执行一次C09离线拓扑V2，固定10 worlds、2,054 directed traversals、24,462 sequences、四方法、243组参数、24,300 replays和59,442,660 typed updates；C10/M-TARE继续为0读取。

## 2026-08-25：GSE结构语义仍在Gate 3，V1全残差逐世界FAIL，V2风险校准待正式执行

- 已封存V1：`results/gate3_semantics/gate3_20260824_gse_corrected_perception_validation_v1_seed0`，状态`FAIL_GSE_CORRECTED_PERCEPTION_VALIDATION_V1`，29/29 seal，SHA=`63a903ba7f7c75891bf9665f359a7dd6637788ec541832456941d1c5ab0398c4`。
- 科学失败不是总体坡度无收益，而是`S02_3d_tree_small_C09`相对五帧先验退化`13.381%`；C09总体三seed平均仍改善`48.364%`。汇总器字段错误属于独立system defect，已修并新增回归测试，旧run保持不可修改。
- 当前主候选为C07--C08-only maximin residual calibration。固定101值网格选出共享scale=`0.89`；只读全量proof使C09 slope MAE=`0.768638°`、总体改善`47.042%`、最差world退化`4.319%`，并使完整事件/四几何/关联门槛同时PASS。
- C09已暴露V1失败，因此V2明确属于development method revision；最终论文泛化结论只能由仍未读取的C10和后续sealed M-TARE worlds提供。离线拓扑、闭环和C10在V2正式证据完成前继续停止。
- NEXT：执行`configs/v3/gate3/gse_risk_calibrated_perception_validation_v2.json`的final preflight，创建并执行唯一不可覆盖V2 run；PASS后发布完整感知/坡度消融图并进入离线拓扑。

更新时间：2026-08-24（train-only坡度corrective三seed正式PASS；返回C09完整资格）

> 当前执行优先级：`docs/PLAN.md` 与 `docs/PROGRESS.md` 高于旧 Gate 路线。本文件中的旧 Gate 证据继续保留，但不得据此跳过 Phase 1--3 或现在修改 M-TARE。

## 当前状态

```text
当前执行 Phase：冻结坡度corrective后的C09完整感知重新资格
当前 operational Gate：Gate 3 semantics validation（旧FAIL保留，新run只替换slope）
已完成实验：主GSE V1R三seed`PASS`；旧C09除slope外全部PASS；train-only坡度corrective V1R现正式`PASS`。三个corrective seed把C07--C08 slope MAE从五帧解析`1.664656°`降到`0.862578/0.862454/0.863412°`，平均改善`48.1686%`，10 topology family全部改善；115文件seal SHA=`d8683e5b...`。下一唯一操作是预检并执行一次新的24,462序列C09完整资格，C10/M-TARE仍禁止。
当前实施上限：完成三seed训练、C09 validation-only感知资格与固定243组离线拓扑选择；在方法、校准、关联阈值和图参数冻结前禁止读取C10与M-TARE正式测试世界
论文完整交付检查：docs/PAPER_COMPLETION_MATRIX_V1.md（当前状态ACTIVE_NOT_COMPLETE）
```

## 2026-08-24 — 投稿前新颖性边界补入两个直接结构拓扑探索对照

- 2026年7月的OVTG已纳入最新直接近邻：它将CLIP开放词汇特征和GAT融入在线拓扑图，进一步否定了宽泛的“学习语义+拓扑”新颖性。其节点仍由空间新奇度触发、edge由时序/邻近可通行性建立，未覆盖GSE几何事件触发节点和物理穿越后建边的组合。贡献矩阵、正文和BibTeX已更新，引用key完整性PASS。
- 新增原始来源复核：Semantic Topometric Mapping从累计二维occupancy grid规则分割intersection/pathway/dead-end/frontier；2025 RA-L Heterogeneous Topological Graph Exploration从二维probability grid、ESDF skeleton和frontier/border/viewpoint确定性构图，再用GNN/RL选择目标。
- 两者证明“结构语义+拓扑探索”本身不能作为新颖性主张，但均未同时采用本体五帧因果LiDAR、显式三维隧道几何事件/出口token、学习关联决定结构节点、真实穿越才提交edge。
- 贡献矩阵、论文related work与BibTeX已更新。当前主张收窄为上述四机制的组合并保持不变；训练数据、模型、阈值、C09/C10隔离及活跃run均未修改。
- 论文方法部分已从冻结实现补齐可复现细节：803,058参数、128维时序/地点表示、6个8-head出口queries、全部loss权重与归一化、batch64、AdamW、BF16、梯度裁剪、三seed和validation-loss checkpoint规则。独立复核指出并已修正logits/probabilities、cosine axis surrogate、train-derived归一化类权重、heteroscedastic variance和`1e-8` tie语义；六项复核全部关闭，代码实算参数量与正文一致，BibTeX引用无缺失。
- 论文实验方法现进一步写明C09-only校准：NLL事件温度、`min(calibrated confidence,1-uncertainty)`拒绝、matched-token出口阈值、parent内past-only descriptor匹配、非空`precision>=0.98/false accept<=0.01`选择，以及第五帧M1D/非学习PCA+五截面对照和三seed感知门槛。该文本逐项来自冻结evaluator，不读取任何结果或C10。
- 离线图方法/正文独立审计已纠正一项重要过度表述：GSE global selector、M-TARE adapter和多机器人交换尚未实现，现明确标为planned；当前edge仅由C09预记录Euler traversal trace确认。旧单帧计数、anchor触发、edge字段、243-grid方法差异、两个主baseline与pending ablations均已对齐冻结实现，八组复核关闭。
- 活跃论文草稿顶部现有显式证据边界：三seed训练仍在进行，C09感知/离线图只属于已实现且预注册的待执行评估，C10、GSE专用M-TARE adapter、单/多机器人闭环及其消融均无当前结果；`AUTO`占位符只有绑定completed sealed来源后才可转成论文结论。摘要与主图说明中的edge合同同步收窄为当前可证的`traversal evidence`，闭环替换明确为planned integration。
- 投稿检索新增Dang与Huber的Bio-Inspired Hybrid Map：其学习特征local frame进入factor graph，属于比普通place recognition更直接的近邻；但新节点仍由累计行程/视点变化触发，未让显式隧道事件/出口token决定结构节点与关联。贡献矩阵、正文与BibTeX已正面加入该边界，当前四机制组合主张维持，仍须由预注册消融和闭环收益证明。
- 同轮检索补入FHT-Map：其CNN视觉特征与局部LiDAR进入main nodes，support nodes及累计free-space路径修补支持规划；节点由视觉丰富度/密度与几何规则生成。该工作进一步否定“学习特征+层级拓扑”这一宽泛新颖性，GSE主张继续严格限定为LiDAR几何事件直接触发结构节点、exit-aware关联与trace-only connectivity。
- 论文图保留合同已再次逐manifest执行：方法总览、数据集、Teacher分布、exit-token审计四包共24个清单内文件全部SHA-256一致，PNG/PDF/SVG与机器源数据/provenance均存在；训练、感知、拓扑和闭环图继续禁止在正式sealed结果前占位造图。
- C09感知证据新增独立拒绝分析发布器：完整保留三seed事件拒绝、place/exit association阈值曲线和选定错误接受率，固定四panel而不挑seed/曲线；仅接受sealed perception PASS且同时复核C10/M-TARE零读取。发布器已加入spec freezer工具哈希，GSE专项联动`118/118 PASS`，当前未生成任何虚构结果图。
- C09现增加diagnostic-only uncertainty证据：按三seed各自全24,462观测的10个等频bin，同时保留几何方差校准与event-error转移曲线。五类事件对照另保留M1D/GSE三seed全量P/R/F1/support，不在制表阶段重新选择。逐世界证据再固定保留10个C09 parent×3 seed的事件增益和四字段几何改进；发布层强制每个parent重放全局冻结温度/阈值，并要求paired geometry valid-count逐字段相等，禁止挑世界或换有效子集。六套C09证据由总发布器统一提交；Luna审计后已从静态目标回滚加强为目录快照差分、精确子包文件集合和独立source-seal复核，成功才写43文件总索引/哈希。正式图表仍为0，必须等C09 sealed PASS。
- 离线拓扑source证据也经Luna审计修正：新共享门禁要求manifest精确覆盖对应run全部文件、路径不越界，并由evaluator/runner双层检查四source身份、PASS与禁止读取。真实dataset `33,083/33,083`和Teacher `17/17`完整覆盖已PASS。每方法另强制243行sweep、精确selected文件集及`7290/7290/7290/2430=24300` replay、`59,442,660`更新总数；PASS后两图包以16文件原子提交，训练曲线以7文件原子提交。专项总计`151/151 PASS`，training/perception待完成后再验证。
- C09出口token证据已修正为与M1D相同的20°方向匹配合同：GSE在validation-selected presence threshold上报告direction precision/recall/F1、matched angular error、count exact/MAE；新发布器固定三seed配对四panel，不把query presence F1冒充方向F1，无匹配时保留缺失值。该项仅作诊断、不改变预注册感知gate；共享metric及publisher进入spec hash，GSE专项更新为`121/121 PASS`。

## 2026-08-23 — Gate 6 V3在77/90因系统时钟回拨FAIL，V3R恢复已冻结

- 原正式run完成并归档77个case后，case077的事后bag统计检出一个`-0.181s` planner runtime；其余597个runtime均在`0.116--0.459s`。主机在相同运行窗口记录`Clock change detected`，故归因为系统计时仪器故障。旧run保持`FAILED`，1201项seal SHA-256=`1c1e51e6a386351e9900edd262fdcb0411f61969f79c7d634579979cf48e4e09`。
- 不采用删除负值或只重跑失败baseline。V3R预声明重跑完整`tunnel_env23`九例并执行未完成尾部十二例，共21例；最终组合只使用旧run中69个不受影响PASS与21个新case，旧区组八例和case077 partial一律排除。
- V3R与predecessor保持aggregate FAIL。V1R正式只读审计验证两个seal、V3R 21个case全树、90-case身份、10×9、30/30/30和60个V9机制文件；状态`PASS_COMPOSED_FRONTIER_ATTEMPT_AUDIT_V1R`，13项seal SHA-256=`0c93c7ccb2e7700a6ec2cfc4bd59b6a9538c90c859fc4032c8ac3902863e6ea4`。
- V5 V1R probe在22帧后因Python3.8无`math.ulp`系统FAIL，未形成case summary或机制结论。兼容修复只把`math.ulp(1.0)`替换为等价`sys.float_info.epsilon`；宿主联动和冻结镜像Python3.8 import均PASS，科研合同不变。
- V5 V1R2从头运行同一case052并正式PASS：901周期、16节点、15条verified edge、移动199.881m、0 fallback/失败周期。出口执行反馈修正在frame51触发，frame52后路线继续增长，最终route arc 204.990m；28/28 seal复核PASS，seal SHA-256=`cfd62eac87614a60ed717dbad244ea2eb69ffbbe362f1782b4df0b1950926516`。
- 六例readiness首次V1在Gazebo前因runner漏导入`_bootstrap`系统FAIL，0/6案例，10项seal=`b2d8520c...042d`。V1R只修启动路径并从0/6执行，6/6 PASS：节点8--20、verified edges 7--20、移动134.748--274.717m、最大fallback 2.1566%、0 planner failure；110项seal SHA-256=`d301f54c1129af5e25a6a0e7d6cc7269ef6f71aa9987e08affaf66b171a8387d`。
- 30-case V1在Gazebo前因paired-source返回值未透传两个schedule hash而0/30系统FAIL，9项seal SHA-256=`93bdb0de2cc618fb2e38e5bfb123a4a5626704e8b2ceebbbf750254903537c98`。V1R补齐只读字段后，真实完整pre-Gazebo proof已验证30 cases、10 blocks、10/10/10 checkpoints、22+8来源、两前置seal、镜像/二进制及30条命令。
- 30-case V1R随后在首个case执行入口因V2 adapter未导出V1 `run_case`而0/30 FAIL，15项seal SHA-256=`749ade2608d07d077e7525fc312d9b9a0fb3cd53870441a1aa16a9f16e1a5ba8`。V1R2显式绑定冻结V1执行器，actual-main sentinel proof已到达首个case调用并验证异常封存，Gazebo未在proof中启动。
- 最终单机器人comparison的论文导出已扩展为：五张既有定量/定性图，加一张四方法coverage-time图，以及family summary、两组paired effects、V5机制计数三组CSV/Markdown/LaTeX表。120条curve逐文件验证原始source seal，图表禁止结果挑选和手工填数；18/18联动测试PASS，20/20 proposal hash一致。
- V5精确配对30-case proposal、Data Card和双seal finalizer已新增；精确要求22个predecessor M1D+8个recovery M1D、10 blocks和10/10/10 checkpoint配额。18项hash一致，来源未seal时正确拒绝，未创建正式run。
- 最终只读comparison proposal/finalizer也已完成：独立重验schedule双hash、10×9 block、30/30/30 family、69+21 source、V5 10/10/10 checkpoint和逐summary身份；输出七指标区组统计、3张定量图及2张固定身份轨迹/拓扑图，均为PNG/PDF。联动18/18测试PASS，16项comparison冻结hash一致。

## 2026-08-23 — Gate 6正式矩阵76/90，V1R3审计冻结等待源seal

- 唯一90-case run持续运行且未重启、重排或修改；当前70/90，case070已启动。case069 V9移动901.947m、0 proxy、0 fallback，但93次frontier事件中78次nonmatching，现有ledger仅捕获1次clearing；该例直接否定V4-only作为充分修复，支持verified re-anchor+图事件执行反馈的最小组合。当前25个V9中21个有proxy，792事件中439 nonmatching，780/780 traversal通过V1R3。
- Luna复核发现V1R2仍缺target exact schema和traversal↔event双射；V1R2未执行并被V1R3替代。V1R3补齐这两项并移除无科学依据的一小时run-age限制；33/33测试、17项冻结哈希和当前13个sealed V9案例/419 traversals真实回放PASS。
- 新增case033覆盖1052.375m³、移动222.181m，但1829帧出现旧节点arrival proxy且末尾1790帧route arc不增长；21个frontier事件中8个非匹配。当前13个V9案例合计10个有proxy，443事件中221个非匹配，继续支持回锚加执行反馈的组合机制候选，但选择仍等待30/30。
- 同一tunnel/env71 block的case035 original M-TARE覆盖10020.125m³、移动1150.232m、规划p95 0.320s；V9 case033覆盖1052.375m³但p95仅0.011s。当前证据定位为“V9计算快、图状态执行差”，正式效果结论仍等待完整block统计和修正后配对矩阵。
- case037 garage/env53/V9-seed2覆盖4774.125m³且p95 0.011574s，但1592帧出现proxy、末尾1515帧无route增长；31事件中16非匹配。当前14例为11 proxy cases，474事件中237非匹配，448/448 traversal通过V1R3。
- case038同块original M-TARE覆盖18470.125m³，V9 case037仅4774.125m³；case039与case040 V9又分别仅覆盖1397.500/226.375m³并出现2590/2834 proxy帧。当前16例为13 proxy cases，483事件中240 nonmatching，459/459 traversal通过。
- tunnel/env23同块case041 original M-TARE覆盖9200.250m³，V9 case040仅226.375m³；case042 garage/env37 baseline覆盖11162.875m³。当前原V9远未达到论文主张，必须经组合修正和精确配对矩阵重新资格。
- oracle case043/045仅覆盖632.000/463.375m³并显著低于baseline，故冻结身份保持“完整地图诊断”而非性能上界；它不决定主方法Gate，但其执行失败必须在论文中如实报告。
- case048 V9仅覆盖442.500m³并有2521 proxy帧，但route arc末尾仍增长；这排除把arc growth单独当作健康判据。当前17例为14 proxy cases，491事件中243 nonmatching，465/465 traversal通过。
- V9样本现为20/30：17个proxy cases，580事件中295 nonmatching，547/547 traversal通过。case050证明低proxy也可由exit lifecycle主导，case052复现极端回锚停滞；组合修正两部分均有独立证据。
- 完整final snapshot没有stub创建帧，故V1R2明确记录该历史状态不可直接重建，并以冻结planner只选择observed stub的源码合同绑定资格；不使用后验GT或虚构时间戳。
- 当前NEXT不变：完成并seal原90-case，执行compatibility与V1R3只读审计，用30/30分布选择V4-only或最小组合修正，再跑probe/readiness/精确配对30-case。C09/C10保持隔离。

## 2026-08-23 — Gate 6正式矩阵运行中，回锚与出口执行反馈两类缺陷均有因果证据

- 唯一90-case run保持不可修改并持续运行；当前29/90完成，主进程与串行Docker case健康。case027 original M-TARE最终覆盖7229.0m³、移动1138.836m、规划p95 0.321s；case028同场景独立重复覆盖6887.375m³、移动1135.957m、p95 0.288s，均完整PASS。该run最终会因已冻结V2 case status与V1 analyzer接口不一致而系统FAIL，但90个case本身仍将完整归档并封存。
- 新增无GT的frontier-attempt执行结果审计：case026的68次可分类出口尝试只有8次沿目标stub离开，50次从其它方向离开、10次回到同节点；60次非匹配中58次目标连续作用路径至少4m、56次至少8m，排除短暂目标切换作为主要解释。node4/stub3为19次方向不一致、4次回到同节点、0次成功。node0↔node4被验证往返37次，而轨迹活动范围仅约31.82×7.48m。该证据把第二缺陷定位为活动出口没有根据执行结果退出候选集合，而不是仅凭低coverage推测。专项测试12/12 PASS。
- 该新证据已单独建立不可执行proposal：90 summaries+30 traces+30 snapshots，9项工具hash冻结，7/7专项测试PASS；运行中源负对照被source-seal finalizer正确拒绝且无正式输出。它不修改已冻结compatibility audit，源seal后才允许一次物化和执行。
- case026没有回锚proxy且移动595.247m，但覆盖仅351.375m³、冗余0.999825；2983帧为frontier_exit，最常选stub累计1383帧，最终11节点/47 stubs/26 observed。该证据否定“V4回锚是充分修复”，新增frontier lifecycle审计将与回锚审计共同覆盖30个V9 case，57/57回归PASS。
- 已完成案例的只读机制审计显示，10个V9案例中8个存在重复`graph_backtrack`抵达旧节点但`current_node`不切换；两个tunnel案例各连续2953帧、最终仅移动约20m。两个零proxy案例移动785--926m。新增tunnel/env53/seed0移动279.514m，但1827帧命中proxy且尾部1782帧无route-arc增长。fallback接近0，故缺陷定位为图状态转移而非模型空输出；当前样本不完整，不作总体效果结论。
- V4只增加基于上一周期目标、现有trace-verified edge、正物理trace、既有loop-merge半径和严格路径一致性的回锚转移；不使用GT、不新增阈值。ROS V4节点、单case wrapper、matched probe、六例readiness、30例精确配对runner、只读统计bridge和最终合并比较均已实现。
- 兼容audit已扩展为验证30/30个V9 sealed trace/snapshot并输出逐案例状态陈旧/路线增长/尾部停滞机理表；源seal出现后一次性物化正式card/spec的fail-closed finalizer已实现，未完成源、哈希漂移或重复输出均拒绝。相关回归54/54 PASS，新增文件在ROS Python3.8中编译通过。V4尚未真实运行，不能宣称修复有效。当前唯一NEXT是让90-case原run自然结束并seal；随后按顺序执行兼容audit、matched probe、六例readiness、V4 30-case和最终统计。
- 兼容audit proposal已冻结90 summaries+30 traces+30 snapshots、零raw bag/C09/C10/training的范围；只等待源最终seal。schedule绑定已明确拆成文件SHA-256=`830f49d1...92201af7`与canonical内容SHA-256=`6aea0921...4b27f1a`，后续兼容audit、V4配对和最终比较同时验证两者。

最新readiness证据：V1R2的高空输出由部署时错误使用registered-map-point重栅格化导致。V1R3把V9输入恢复为训练使用的原始16×350 organized `/velodyne_points`及冻结350→720算子，其他科学变量不变。6/6 case全部通过，移动`20.121--279.053 m`，节点`2--19`，verified edges`1--19`，每例901个非hold周期、post-warmup fallback均为0、零失败周期。正式状态`PASS_AEE_COMPOSITE_V9_READINESS_V1R3`，seal SHA-256=`30a213d156d1f64bc3be3a8a4524df9ea6ac06463ed95b596c5d9d156f7b804d`，C09/C10读取为0。当前唯一下一步是将既有90-case矩阵绑定V9 deployment与AEE原始输入记录合同后执行。

最新V9证据：三个seed均从原始M1D checkpoint重新训练，没有复用V8R失败checkpoint。学习分支负责出口方向，冻结B0负责出口数量，固定规则负责terminal/interior/junction角色。held-out sparse方向F1=`0.847874/0.847615/0.843185`，count/role macro-F1=`0.775478/0.722601`；三seed全部科学与tensor合同通过，状态`PASS_AEE_CORRECTIVE_COMPOSITE_V9`。耗时`528.695 s`，65项seal SHA-256=`5d397b2c296029dfb2decdb57d9fbb9332e1713cbf1faaf46ee90cd8f6a85468`，C09/C10/formal benchmark读取为0。该PASS解除表示层阻塞；C08因果图已有历史sealed PASS，按权威计划下一步是AEE 6-case readiness，不重复C08参数选择。

最新representation证据：V7在保持encoder及方向分支的条件下，把既有embedding MLP改为逐方位非线性后再环形汇聚；三seed方向F1仍为`0.827722/0.824020/0.817702`，但count F1=`0.376367/0.446549/0.387015`、role F1=`0.553312/0.574543/0.561693`，整体状态为`FAIL_AEE_CORRECTIVE_SPATIAL_POOLING_V7`。因此后端分类读出变体停止，下一步必须适配encoder并显式保留方向输出。run耗时`487.452 s`，证据清单SHA-256=`d9f86c494779b217d8da1a396adc21488d069878360878753da38d4db137b62b`，C09/C10/benchmark读取为0。

最新V8R证据：same-view teacher retention与仅开放encoder.5使三seed方向F1达到`0.847874/0.847615/0.843185`，role达到`0.653891/0.658635/0.634710`，说明方法方向正确；但count仍为`0.487912/0.539757/0.484828`，整体`FAIL_AEE_CORRECTIVE_ENCODER_RETENTION_V8R`。在继续扩大encoder前必须检查mask后的单帧LiDAR是否仍包含完整拓扑count所需证据。seal=`1da86766b002e9f6fc2fa576fee60148a7ce49ca8d69d4281775e551f466c15d`，C09/C10/benchmark读取为0。

最新representation证据：三seed各完成10 epochs/470 steps，执行与封印正常，但held-out sparse Cano方向F1=`0.495606/0.516503/0.506227`，较source下降`0.139905/0.155336/0.162172`；dense方向下降约`0.195--0.220`。AEE train-fit方向F1仅`0.0227--0.0548`且空输出`91.7%--96.9%`。720-bin标签正质量约4.5%，普通BCE常数最优logit约`-3.05`，构成方向空输出塌缩的直接机制证据。run seal=`ba5eff2b0300c0e0334a66eb0ad7124cf3528ce6dab62e4fb7fbd687ec442454`，62项独立验证，C09/C10/benchmark读取为0。

最新证据：V7R2正式run完成3 worlds、25 windows、624 directed traversals、4773 frames和三档14319 frame audits，数值安全、连续性、seam、isolation、resolution identity与150 patches均按实现通过。但事后contract review确认executor在window mask前对全轨迹调用support solve；权威CSV join显示4,307个窗口外帧全部被修改z（S01/S06/S10=`721/1286/2300`），幅度`-4.716372..-3.060038m`，违反spec的`Optimize only window frames`。S10固定12帧native-mesh探针又为12/12 objective branch LOS失败。因此run保留为immutable machine evidence，但项目结论重分类为`FORMAL_MACHINE_PASS_RESEARCH_CONTRACT_FAIL`。详见`docs/C08_V7R2_CONTRACT_REVIEW.md`。

当前唯一科研问题：

> 在冻结的十个world×environment随机块中，修正图回锚后的结构语义拓扑全局规划器能否相对原始M-TARE提高coverage-time效率、降低规划延迟并保持闭环稳定？

当前工程状态：same-pose sensor接口、C13--C24 topology-only资格和V1R4 20-parent感知网格均已PASS。V1的metrics目录缺陷已由V1R修复并通过30/30测试；V1R连续完成3个世界1500帧，随后在`S02_3d_tree_small_C14`发现冻结姿态净空`0.765985m < 0.8m`而正确停止。独立只读审计确认固定10000帧中共有7帧不合格。进一步对全部6058个候选cluster/30290帧的只读proof得到6043个5/5帧合格cluster；在该eligible集合上20/20 worlds均可重新选出100 clusters/500 frames，事件零遗漏且原隧道配额/间隔合同通过。该proof为0写入/训练/模型/C09/C10；V1R为565/565 seal PASS，V1/V1R partial数据均禁止复用。

V1R2现状：新增类型化cluster/frame eligibility、完整6058候选事件/隧道分母、eligible-only结构选择、最终materialization资格重放、6058-row provenance和4 GiB进程树/Docker门禁；37/37相关测试及sidecar compile PASS。纠正后的权威计数为15个不合格cluster内21个失败帧，不是早期汇总的19；6043 eligible、30215 eligible poses、20/20 worlds和10000 selected frames不变。proposal preflight只有正式run与Data Card approval两项预期错误，0 warning，V1R2 run目录不存在。

V1R2正式结果：Cano阶段20/20 worlds、6058/6043/15 clusters、21失败候选帧、2000 selected clusters和10000最终帧全部PASS，耗时349.851秒、峰值RSS 759226368 bytes。AEE第1条tunnel seed11完整PASS；第2条seed23采满bag后因旧order pairing把启动期raw扫描与更晚registered pose错配，最大约0.4秒而FAIL。失败bag只读证明有3002个唯一单调近时pair，首3000 pair最大delta 0.003秒，raw索引7--3006。整体V1R2为immutable FAIL，8.0GB、3432/3432 seal，SHA-256=`067c84ae60c3470e30cef50dc336c00674459ea1642e3572df4cc77898f6de6b`；训练/模型/C09/C10均为0。

V1R3正式结果：单调近时pairing实现与152项AEE回归测试通过，真实seed23包生产函数回放为3000 pairs、max delta 0.003秒。正式run的Cano 20/20、10000帧再次PASS；AEE seed11的3000 pairs、100 retained、max delta 0.003秒、1148.690825m和sensor shard均PASS。失败仅因V1R3 handoff wrapper把稳定ownership状态从V1改名，旧archive consumer正确拒绝；run为immutable FAIL，3417项seal SHA-256=`2236082aaa2c606cc4fc4f17e85fc888c2446c58f3d6ace401503031ca6d3ac9`，partial禁止复用。V1R4只恢复稳定ownership名称，24项专项测试和preflight 0/0通过。

V1R4正式结果：20/20 Cano worlds与10000帧、10/10 AEE trajectories与1000帧全部PASS；每条AEE均有3000严格单调exact-odom pairs，10条最大delta不超过0.004秒，最短轨迹1136.763062m。总计11000 scans/labels、6000 train/5000 validation、1000 unique AEE frame IDs；耗时7701.050秒，结果20.766GB，峰值进程树RSS729878528 bytes。3578/3578 seal独立验证PASS，SHA-256=`dd6d5ac8a5a48223d2693734700b11396dd12a3aec63e4bbeaf02cf6f5fbaa59`；训练/模型/M-TARE/C09/C10均为0。

下一阶段接口阻塞：冻结AEE xacro/plugin证明原始`/velodyne_points`确实是可重新采集的organized `350×16` PointCloud2并保留NaN，但350水平样本位于闭区间`[-π,+π]`，首尾方向重复；现有重采样原语假设半开圆周且source index 0为0°。直接连接会造成180°起点错位并忽略重复端点。该sensor-interface contract必须先明确精确角重排和±π重复束合并规则；拓扑PASS不受影响。

从第1帧重跑的完整只读proof为`PASS_AEE_DAE_SUPPORT_TEACHER_READONLY_PROOF_V1`：10条轨迹、30000 raw、6000 effective、train/validation各3000、6000唯一frame全部得到唯一非空且direction/count/role一致的客观标签；耗时2371.436秒，teacher shard/model inference/training/C09/C10均为0。旧V1失败点、seed53 sample485等距点和seed71 sample428错误吸附点全部越过。真实DAE中garage/tunnel有880/168个surface identity；只读审计确认绝大多数是实际断开的geometry islands，而非浮点碎裂，因此未引入任意坐标容差。

正式结果：`results/gate2_representation/gate2_20260821_aee_dae_multilayer_objective_teacher_export_v1r_seed20260820/`为`PASS_AEE_DAE_MULTILAYER_OBJECTIVE_TEACHER_EXPORT_V1R`。10 shards、6000 labels、6000 unique frame IDs、train/validation各3000；耗时2385.476秒。train role 0/1/2=`2083/722/195`，validation=`1014/928/1058`；训练、推理、C09/C10均为0。独立复核46/46 seal及全部shard字段/哈希PASS，四张coverage/role图完整非空。

根因诊断发现两项合同不一致：第一，Cano方向监督是出口中心的`3°` Gaussian（平均target密度`0.042693`），AEE teacher shard保存整片可通行扇区（tunnel=`0.179390`、garage=`0.620927`）；第二，AEE valid density只有`0.4617/0.4827`，Cano约`0.9976`，冻结encoder从首层起即把AEE表示压成近常量，AEE/Cano z_role centroid cosine仅`0.196--0.217`。frame/yaw/heading/count/role编码未发现错位。

当前唯一工作：把AEE原始traversable mask保留为几何证据，同时按相同component center和Cano冻结`3°` Gaussian生成学习target；6000/6000只读canonicalization proof已通过，target均值train/validation=`0.047020/0.046394`。随后设计完整encoder域适配与AEE-mask-matched Cano增强；新训练前必须冻结Data Card/spec并preflight。

## 最新执行与复核：perception-mesh M0 exact-byte replay FAIL

证据目录：`results/gate0_baseline/gate0_20260811_cano_100_parent_perception_mesh_contract_m0_seed0/`。执行前 98/98 单元、12/12 E1 external、17/17 V2R、391/391 V2 source seal 和 preflight 0/0 通过；唯一正式 run 在 S01 primary/replay 的 OBJ SHA-256 不相等处按 stop rule 结束，S02--S10 未生成，无重试。

S01 两次 topology、operation trace、effective geometry parameters 和 axis 完全一致；两个 mesh 各自均通过 readability/axis/component 合同，均为 69,908 vertices、139,822 triangles、零 degenerate、单组件。双向 nearest-vertex maximum 为 `0.599708/0.571740 m`，AABB 最大坐标端点差 `0.110840 m`，surface-area relative difference `0.0425%`。完整 train 图无空白或断裂。run 43.508 s、21.9 MB、33/33 evidence hash；validation/dev-test 与 LiDAR/data/training/model/simulator/M-TARE 计数均为 0。

根因是 Open3D Poisson 不保证 vertex ordering，而 Cano 随后按当前 vertex array 对每坐标加入 `Uniform[-0.2,0.2] m` 噪声；同一空间位置理论三维差异上界为 `0.692820 m`，所以 exact OBJ hash 不是正确 native replay 指标。完整复核见 `docs/CANO_100_PARENT_PERCEPTION_MESH_M0_REVIEW.md`。M0R proposal 只改 replay metric，尚未批准、实现或执行。

## 最新执行与复核：100-parent V2R recipe split PASS

证据目录：`results/gate0_baseline/gate0_20260811_cano_100_topology_parent_recipe_reclassification_v2r_seed0/`。正式执行前 93/93 单元测试、11/11 冻结 E1 external、391/391 sealed-V2 source hash 和 preflight 0 错误/0 警告通过；唯一正式命令返回 `PASS_CANO_100_TOPOLOGY_PARENT_RECIPE_RECLASSIFICATION_V2R`，`RUN_STATE=COMPLETED`。

本轮只读 120 个 sealed candidate。requested grown/connector 被正确固定为生成 recipe，observed cycle rank 保留为 graph GT；所有 120 个 candidate 通过非 exact-cycle 合同和 `actual cycle rank >= requested connector count`。每层固定 C01--C10 后形成 100 parent：80 train、10 validation、10 development-test；100/100 canonical identity、100/100 coordinate-free WL hash 唯一，100/100 含结构事件，50/50 为合格 3D parent。

执行范围核验：新 topology/replay/graph/spline/preview、mesh、LiDAR、label、formal dataset、training、model、trajectory 和 M-TARE change 全为 0。结果只有 3 个 manifest、3 个 metrics 和治理证据，目录 250,816 bytes，17/17 evidence hash 通过。该 PASS 冻结的是 topology parent 身份与 split，不代表 Phase 1 mesh PASS，更不代表已有数据集或模型。

## 最新执行与复核：V2 生成全部成功，exact-family 合同 FAIL

证据目录：`results/gate0_baseline/gate0_20260811_cano_100_topology_parent_candidate_audit_v2_bounded_resampling_seed0/`。唯一正式 run 完成 120/120 generation 和 120/120 replay；114/120 通过 exact-cycle 合同，各层 valid/12=`11/12/12/12/12/12/9/12/10/12`，S07 少 1 个，最终 99 parent/80-10-9，状态 FAIL。

bounded resampling 本身有效：936 条 tunnel operation 平均 1.1357 draws、最大 5，0 条耗尽 20-draw budget，V1 的 60 个 generation failure 降为 0。6 个 invalid 唯一理由是 actual cycle rank 比 requested connector count 多 1；这来自 grown tunnel 接入已有 intersection，并非坏图或 replay 失败。

99/99 canonical identity、99/99 WL hash、99/99 structure-event 均唯一/通过；391/391 evidence hash 复核。10 张 train-only 完整图无空白/断裂，flat/3D 高度合同视觉一致。完整复核和 V2R 建议见 `docs/CANO_100_TOPOLOGY_PARENT_AUDIT_V2_REVIEW.md`。正式 mesh、LiDAR、label、dataset、training、model、trajectory、M-TARE 仍为 0。

## 最新执行与复核：100-parent native-random candidate audit V1 FAIL

证据目录：`results/gate0_baseline/gate0_20260811_cano_100_topology_parent_candidate_audit_v1_seed0/`。正式命令完成全部 120 个预声明 candidate，状态 `FAILED`，运行 555.74 s、约 44.8 MB、269 项 evidence hash 通过。没有 mesh、anchor、LiDAR、label、训练、模型或 M-TARE 修改。

120 个候选中 60 个通过当前合同；各层 valid/12=`7/9/6/9/3/9/3/9/1/4`，只能形成 56 train/4 validation/0 development-test，不能作为数据集。flat 成功率 20/60，3D 为 40/60；所有 60 个失败均发生在 grown tunnel 的单次参数抽样，没有 connector failure。

成功侧证据：60 个 canonical identity 与 coordinate-free WL hash 都为 60/60 唯一，60/60 含 degree>=3 事件；10 张固定 train-only 全图显示 flat z=0、3D 多高度和随规模增长的分支/回环。失败根因不是五模板重复，而是当前 bounded adapter 比上游无界 while 更严格，只允许每条 grown 抽一套参数。

另发现当前 family contract 过宽：S01 C07 请求 0 connector 却 cycle rank=1，S07 C04 请求 2 connector 却 cycle rank=3。后续需使用 exact cycle-rank。完整复核与 corrective 建议见 `docs/CANO_100_TOPOLOGY_PARENT_AUDIT_V1_REVIEW.md`。

## 最新执行与复核：五拓扑 selector-v2 CPU LiDAR pilot PASS

证据目录：`results/gate0_baseline/gate0_20260811_cano_five_topology_cpu_contract_pilot_v3_selector_coverage_seed0/`。用户批准后，v3 从头生成全部五图，未复用 v2b 的局部 shard；86/86 单元测试、9/9 冻结 E1 外部测试和零错误零警告 preflight 通过，唯一正式命令无重试完成。

实际产出 5 个 world bundle/mesh、250 anchors、750 个 16×720 静态观测、5 个诊断 NPZ、5 张完整地图与 30 页全观测图；两个独立 RaycastingScene 共 17.28M primary rays，最大 range 差异为 `0 m`，100 项 evidence hash 全部通过。冻结规则总体 P/R/F1=`0.8872/0.9431/0.9143`，平均角误差 `3.501°`；P05 slope-multiheight 最低 F1=`0.8374`，P04 chamber-multiexit 的过检最明显。

已逐张检查全部 35 张图，没有空白批次或明显传感器损坏。可视化误差与 P04/P05 指标一致，因此当前不回调冻结规则。完整方法、逐图指标和边界见 `docs/FIVE_TOPOLOGY_CPU_LIDAR_CONTRACT_V3.md`。

边界：本轮正式 dataset/training/model/trajectory/online graph/M-TARE change 均为 0。它证明五类结构的理想感知合同可运行，不证明已学到结构语义、未见拓扑泛化、Gazebo parity 或规划收益。下一步只设计 100 topology parent 及 80/10/10 parent-level split，未经批准不批量生成。

## 最新执行与复核：selector v2 完整空间覆盖 PASS

证据目录：`results/gate0_baseline/gate0_20260811_cano_anchor_selector_coverage_audit_v2_seed0/`。selector v2 保留结构事件，按 tunnel spline 弧长分配非事件配额，并在 0.5 m 弧长格上用确定性二元 MILP 同时满足精确配额、全局 5 m 间距和 7.5 m 完整 spline 覆盖。86/86 单元、9/9 冻结 E1 外部测试与零错误零警告 preflight 通过后，唯一正式 run PASS 并封存。

五图共 250 anchors；总体最小间距 `5.006145889 m`，最坏同 tunnel 或共享事件覆盖半径 `6.391727627 m`，所有弧长配额、结构事件、tunnel coverage 和 replay hash 均通过。已人工逐张检查五张完整 X-Y/X-Z 图：P01 长直段、P03 上回环和连接段、P05 高低坡均连续覆盖，未再出现 v1 的长空洞。详见 `docs/ANCHOR_SELECTOR_COVERAGE_AUDIT_V2.md`。

边界：本轮 mesh、ray、formal dataset、training、model、trajectory、topology-runtime 和 M-TARE change 全为 0。它只修复“去哪里采 LiDAR”，不等于已经获得 LiDAR 数据或完成结构语义学习。该步骤随后已由上方五拓扑 v3 PASS 结果接续。

## 上一轮执行与复核：selector v1 数量合同 PASS，完整空间覆盖 FAIL

证据目录：`results/gate0_baseline/gate0_20260811_cano_anchor_selector_zero_raycast_audit_v1_seed0/`。正式 run 只复用 P01--P04 graph/spline，并构造一次 P05 graph/spline；无 mesh、ray、正式数据或训练。85/85 单元、9/9 外部测试及 preflight 通过后只执行一次。机器合同为 PASS：五图各 50 anchors、全局最小间距 `5.4985 m`、所有 replay hash 一致，32 项证据 hash 通过。

但五张完整图揭示 earliest-feasible round-robin 从每条 spline 一端开始填，达到 50 后停止。P01 尾段和 P03 connector 上段明显无普通 anchor。对 spline knots 计算到同 source tunnel 最近 anchor 的最大距离，P01--P05 分别为 `32.780/16.564/92.306/10.501/17.558 m`。因此 `selected-per-tunnel count spread <=3` 是不完整甚至误导的均衡指标：长 tunnel 应按 arc length 获得更多名额。项目层把该结果登记为 `CONTRACT_PASS_RESEARCH_REVIEW_FAIL`；详细复核见 `docs/ANCHOR_SELECTOR_VISUAL_REVIEW_V1.md`，禁止据此重跑 LiDAR。

## 最新执行结果：五拓扑 v2b 在 P04 anchor 选择处失败

证据目录：`results/gate0_baseline/gate0_20260811_cano_five_topology_cpu_contract_pilot_v2b_runner_paths_seed0/`。runner proposal/data-card 路径修复后，84/84 单元测试、8/8 冻结 E1 外部测试及零错误零警告 preflight 通过；唯一批准命令真实执行一次，固定 checkout 导入与全部运行前身份门禁通过。

P01、P02、P03 各完成 50 anchors、150 views、双场景确定性检查、分支 LOS、完整 shard 与 world map；冻结规则 F1 分别为 `1.0000`、`0.9447`、`0.9154`。P04 已生成 graph/splines/native mesh 且 chamber 非空，但 `select_canonical_anchors` 在 5 m 全局欧氏间距下由当前最远点贪心只能选到 38/50，随后按 stop rule 抛错；P05 未开始。总运行 32.24 s、37 MB，51 项封存 SHA-256 全部复核通过，无重试。

P04 并非中心线太短：四条中心线共 326.13 m、655 个 spline knots。根因是“事件点先占位 + 逐次最大化最近距离”的贪心只保证得到不可继续追加的极大 packing，不保证目标 cardinality。修复必须保留事件与 5 m 约束，同时显式保证 50 个并审计分支覆盖；不能用降到 38、减小间距或换 seed 获得 PASS。前三图的 450 个观测只属于失败 run 的诊断证据，正式数据、训练、模型、轨迹、在线图和 M-TARE 修改仍为 0。

## 较早执行结果：五拓扑 v1 在生成前被源码身份门禁拦截

证据目录：`results/gate0_baseline/gate0_20260811_cano_five_topology_cpu_contract_pilot_v1_seed0/`。84/84 单元测试、5/5 外部测试与零警告 preflight 后，批准命令只执行一次；执行器在创建任何 world 之前发现实际导入文件位于 E1 的 `site-packages/subt_proc_gen/tunnel.py`，不在固定 `external/procedural-subt-gen/src` 下，因此 1.35 s 内主动失败并封存。

本次实际 world、mesh、anchor、observation、NPZ shard 和可视化均为 0，不能评价五类拓扑或算法。根因是 runner 的 `PYTHONPATH` 漏掉固定 checkout。现在已把固定 checkout 放在执行器路径首位，并增加“进入 RUNNING 前由 E1 子进程报告实际模块文件”的门禁；修复后 84/84 单元测试与 6/6 外部测试通过。原批准的一次执行已经消耗，未自动重试；新的 corrective run 必须使用新批准和新目录。

## 最新可运行结果：LiDAR 到 Online Topometric Graph 纵向切片

工程原型 v2 位于 `results/prototypes/cano_seed0_lidar_to_topometric_vertical_slice_v2_graph_refined/`。它复用已有 seed-0 Cano world，沿完整 source graph 生成 240 帧、1,676.46 m 的因果遍历；每帧只把 16×720 range/valid mask 交给透明 range-sector 规则，完整 spline/graph 只供并行 oracle 与评测使用。两路输出进入相同的在线图更新器，最终得到节点、物理移动边以及 unexplored/traversed exit stubs。

当前帧规则的出口 P/R/F1 为 `0.7480/0.7525/0.7503`，平均匹配角误差 `6.985°`。规则在线图为 43 nodes/50 edges，oracle 为 44 nodes/50 edges；空间角色 node F1=`0.8276`，直接几何 edge F1=`0.94`。结果目录保存完整 240 帧 NPZ、逐帧 JSON、两份图 JSON、81 组关联参数 sweep、全图 PNG、24 帧诊断 PNG、60 状态 GIF 和 13/13 可复核证据 hash。

边界必须同时阅读：这是一个已见 world 的开发原型；没有训练模型，图参数由同一轨迹的 81 组候选选出，且尚未闭环或修改 M-TARE。因此它证明“完整软件接口能工作”，不证明 unseen topology 泛化、学习优于规则或探索收益。v1 弱结果也保留，便于核验参数选择前后的差异。

## 最新路线决定：能复现的 baseline 先直接使用，Isaac 不再阻塞

用户确认停止把 Isaac 当作 Phase 1--3 前置条件。未实现的 async Writer v3 提案已标记 `RETIRED_BY_USER_ROUTE_DECISION_NOT_IMPLEMENTED_NOT_EXECUTED`；历史 Isaac 失败目录和分类保持不可变，未来只有出现明确传感器域研究需要时才可另立提案。

当前直接复用边界：固定 `procedural-subt-gen@b6c776...` 与 E1 环境运行 TNG、表面点云和 native mesh；项目 read-only adapter 补 graph/splines/provenance；现有 Open3D 0.19 CPU raycast 生成局部 first-return range；现有 5 m spline teacher 生成 outgoing-branch 标签。固定 checkout 没有发现论文出口 CNN 训练代码，因此后续 B1 必须写成 `Cano-like adapted reproduction`，不能声称运行了原作者模型代码。

二次创新不混进数据 smoke：先复现 range-image exit baseline；之后才验证 planner-consistent `R/D/U`、稳定 exit-stub 图、M-TARE global planner replacement 和 multi-robot allocation。方法边界见 `docs/CPU_RAYCAST_LIDAR_BACKEND_V1.md`。

CPU 单图合同已按批准范围只执行一次并 PASS：`results/gate0_baseline/gate0_20260810_cano_cpu_raycast_lidar_contract_v1_seed0/`。只读复用 1 个 seed-0 Cano world、24 poses（8/8/8）、每 pose 11,520 rays、两次独立 scene；24/24 通过，双场景与旧参考最大差异均为 0 m，最小净空 2.0585 m，分支 LOS 24/24，46/46 封存 hash 通过。正式数据/训练/model 计数保持 0。

该 PASS 只把 native mesh 认定为下一步理想化 CPU 感知 pilot 的可用资产；它仍不是动态导航 mesh，也没有证明 Gazebo/真实 LiDAR parity。下一步只能准备五拓扑 pilot 的独立提案和卡，未经新批准不得生成。

## 最新执行结果：精确官方 local-USDA 在同步 standalone 路径 callback=0

证据目录：`results/gate0_baseline/gate0_20260810_isaac_official_local_usda_writer_control_v2_seed0/`。68/68 V3 测试、零错误零警告 preflight、资产/镜像/官方示例/代码 hash 均通过后，只执行一次，无重试。精确官方 sensor 创建成功 1 个，Writer attach 成功 1 个，完成 300 次同步 standalone render update；但 callback、payload、valid GMO、positive/zero element 和 complete scan 全部为 0，shutdown 日志出现两次 Writer schedule drain timeout。

run 为 `FAIL_OFFICIAL_LOCAL_USDA_WRITER_RUNTIME_CONTROL`，结果目录 160 KB，15/15 evidence hash 通过，没有缺失证据，也没有 NPZ、图片、checkpoint 或模型。Cano world、custom project sensor、正式数据、标签、训练和模型均为 0。

封存 summary 按预设规则写为 `HOST_OR_HEADLESS_REPLICATOR_RUNTIME_BLOCKED`，该文件不回写。后续只读核对 NVIDIA 固定镜像中的正式测试发现：`test_lidar_sensor.py` 继承 `AsyncTestCase`，依次执行 async stage、viewport readiness、`next_update_async()` 和最终 flush；这不同于 v2 的同步 standalone loop。因此项目层结论收窄为“同步 standalone scheduling blocked；general async Kit runtime untested”。完整证据见 `docs/ISAAC_ASYNC_WRITER_RUNTIME_AUDIT_V1.md`。

后续 async 提案曾被形成，但现已由用户路线决定退休，未实现、未执行；该未验证分支不再是当前 blocker。

## 最新执行结果：官方 Writer control 未创建传感器

证据目录：`results/gate0_baseline/gate0_20260810_isaac_official_lidar_writer_control_v1_seed0/`。58/58 测试、preflight、镜像/官方示例/代码 hash 均通过，只运行一次、无重试。`Lidar.create(config="Example_Rotary")` 调用 `get_assets_root_path()`，但批准命令的 `--network none` 无法访问运行时配置的 NVIDIA 6.0 远端资产根，抛出 `Could not find assets root folder`。镜像与主机只读搜索均确认没有本地 Example_Rotary USD。

所以本次没有创建 LiDAR、没有 attach Writer、没有进入 300-frame 回调判据；缺少 `runtime_control.json` 与 stage，结果为 `FAIL_OFFICIAL_WRITER_RUNTIME_CONTROL_MISSING_EVIDENCE`。13/13 封存证据 hash 通过，目录 124 KB，点云样本、数据、标签、训练和模型全部为 0。summary 中 `built_in_control_sensors=1` 是计划/尝试数，不是成功创建数；Kit 仍把 Python异常掩盖为内部退出码 0，权威判据是 missing evidence 与 failure traceback。

## 最新执行结果：精确官方 USDA 来源冻结 PASS

证据目录：`results/gate0_baseline/gate0_20260810_isaac_official_lidar_asset_freeze_v1_seed0/`。唯一一次请求返回 HTTP 200、effective URL 精确一致、0 redirect；文件 15,137 bytes，SHA-256 `0812faf5c310f40316d5a11ab0c6786e19e18ac12cea10edc2e7ee44fc56c8c6`。断网 OpenUSD 25.11 解析 PASS：default prim 为 `Example_Rotary`、类型 `OmniLidar`、遍历 8 prim，sublayer/reference/composition/external-asset dependency 全部为 0。16/16 evidence hash 通过，GPU/simulation/Writer/sensor/sample/data/label/training/model 计数为 0。

这只解除“官方资产不自包含”的阻塞，不证明 Writer runtime 健康。后续 v2 已执行并由上方结果更新其状态。

## 较早执行结果：Writer v2 callback=0

证据目录：`results/gate0_baseline/gate0_20260810_cano_exact_profile_writer_probe_v2_seed0/`。新审批、冻结 hash、53/53 测试和 preflight 全部通过；只执行一次。local USDA、69 项 SensorChecker 和属性回读再次 PASS，但 300 frame 内 callback=0、zero-element=0，关闭时 writer schedule drain 超时。完整扫描、NPZ 和图片均为 0，runner 正确记录 `complete_diagnostic_scans=0`；19 项证据 hash 全部通过。

固定镜像的 standalone 官方示例与 v2 使用相同 Writer 调度写法，所以当前不能继续靠改 custom profile 猜原因。后续官方 control 已执行，但先被未冻结的远端传感器资产依赖阻塞，仍未回答 runtime A/B。

## 最新执行结果：USDA 参数层 PASS，直接 GMO 读取层 FAIL

证据目录：`results/gate0_baseline/gate0_20260810_cano_exact_profile_creation_probe_v1_seed0/`。v1 在 0 个 Cano world、1 个解析盒、1 sensor、1 pose 范围内成功创建 `OmniLidar`，69 个 SensorChecker 参数与全部冻结属性回读通过，旧 `Config not found` 消失；但直接 `LidarSensor.get_data` 在 300 frame 内持续返回非法 GMO magic，未保存完整扫描或图片，真实 complete scan 数为 0。

封存 summary 的 `complete_diagnostic_scans=1` 是外层错误依赖被 Kit shutdown 掩盖为 0 的退出码所致，不作为物理证据，封存结果不回写。Writer-callback v2 和证据计数修正当时通过 25/25 静态测试；v2 后来已执行并由上方最新结果记录。完整审计见 `docs/ISAAC_RTX_GMO_WRITER_AUDIT_V1.md`。

## 最新只读审计：自定义 profile 应使用本地 OmniLidar USDA

审计固定 Docker image `nvcr.io/nvidia/isaac-sim:6.0.1@sha256:783444...30aa9` 的新/旧 API、supported registry、官方测试、Lidar Core 文档与生成 schema。结论是：`config=` 只按官方 USD 资产白名单解析，任意 JSON 文件不会因挂载到 profile 目录而加入该 registry；6.0.1 的官方自定义入口是 `Lidar.create(usd_path=...)`。

推荐修复保持 RTX Lidar Core backend 和全部冻结参数不变，用本地 `OmniLidar` USDA 显式写出 16 emitters、16 channels、channelId 1..16、7200 Hz pattern firing、10 Hz scan、0.3--50 m 与原误差/回波/角度数组。不得修改 NVIDIA registry，也不得切换内置近似型号。完整审计见 `docs/ISAAC_RTX_EXACT_PROFILE_CREATION_AUDIT_V1.md`。

历史 v1 与 v2 均已执行失败，各自授权不得重用。两次 complete scan 均为 0，正式 dataset/training/model 均为 0；当前只允许按已批准冻结规格执行一次官方 runtime control，不自动恢复 24 poses。

## 最新执行结果：Cano 原样 smoke FAIL_BLOCKED

证据目录：`results/gate0_baseline/gate0_20260810_cano_source_executability_smoke_v1_seed0/`。

- 固定 remote 与 commit 核对通过，原 `snippet_2.py` 运行前后 hash 一致，tracked 源码无改动；
- 仓库无许可证正文，只有 `pyproject.toml` 的 MIT classifier，第三方集成/修改发布/再分发仍阻塞；
- requirements/pyproject 未锁版本并漏声明 `cv2`；本次隔离环境解析为 NumPy 2.5.2、SciPy 1.18.0、Open3D 0.19.0、PyVista 0.48.4；
- 唯一一次原样执行在第 1 条 grown tunnel 失败。`Spline3D` 将 shape `(N,1)` 的距离数组传给 `scipy.interpolate.splrep`，触发 `TypeError`；最小控制复现确认一维输入可调用，但本次未修改源码；
- 完成 topology/world/point cloud/mesh 均为 0。原脚本本身也只显示 graph/spline，不生成 mesh/point cloud、不序列化 TNG、无 seed 接口；
- 只读源码进一步确认：`snippet_3.py` 计算固定手工拓扑的 point cloud/mesh 但不保存，`generate_environments.py` 才保存 `mesh.stl/axis.txt/fta_dist.txt/model.sdf`。它们本次未执行；
- 因此不能声称 Cano 可执行、可复现、已有单图、graph/mesh 一致或 Phase 1 PASS。没有进入 adapter、批量、Isaac、LiDAR、标签或训练。

推荐下一动作是另立受控兼容性矩阵审计，在不改脚本的前提下测试少量预冻结历史 NumPy/SciPy 组合；并行向作者索取 environment lock 与明确 LICENSE。兼容性通过后再单独申请原生 `generate_environments.py` 单图导出 smoke。该动作需要用户确认，当前保持停止。

## 最新追加结果：冻结 E1 环境恢复原脚本进程执行

证据目录：`results/gate0_baseline/gate0_20260810_cano_dependency_compatibility_matrix_v1_seed0/`。

- E0 当前依赖失败不重跑；E1 固定为 Python 3.12、NumPy 1.26.4、SciPy 1.12.0、OpenCV-headless 4.8.1.78；
- E1 安装与 `pip check` 通过，原样 `scripts/snippet_2.py` 在 46.31 s 内退出 0，达到 10 个 grown 调用和 5 个 connector 调用；源码 hash 和 tracked 状态不变；
- stop-on-first-pass 生效，E2 未创建、未执行；
- 第 5 个 connector 明确耗尽 1000 次 trial。生成器会返回 `(False, None)`，但原脚本不检查返回值，因此退出 0 只支持 process executability，不支持最终 topology 成功、tunnel 数、graph connectivity 或 topology validity；
- 本 run 导出 world/point cloud/mesh/LiDAR/label/training sample 全部为 0，Phase 1 仍未 PASS；许可证、seed 和原生 topology export 问题仍存在。

该阶段当时的下一步是提交原生 `generate_environments.py` 单图导出 smoke；该动作随后已执行并由下方更新结果取代。

## 最新追加结果：原生文件输出成功，但 mesh 与 topology GT 失败

证据目录：`results/gate0_baseline/gate0_20260810_cano_native_export_smoke_v1_seed0/`。

- 用户批准后只执行一次原始 `generate_environments.py`：1 个临时环境、请求 3 grown/1 connector；13.70 s 退出 0，生成 `mesh.obj/axis.txt/fta_dist.txt/model.sdf`，总计 9,851,416 bytes；
- E1 freeze 与 `pip check` 通过，第三方 remote/commit/entrypoint hash 和 tracked clean 前后不变，21 项原生运行证据 hash 通过；
- mesh 可读取但只有部分执行价值：61,421 vertices、122,862 triangles、7 components，非 watertight、非 edge/vertex manifold、不可定向且 self-intersecting；
- `axis.txt` 有 4 个 tunnel ID，但 3,888 条 tunnel 行全部冲突；972 个唯一坐标均被重复赋给多个 tunnel ID，源码预测的全局轴点复制缺陷得到实证；
- 原入口没有 graph、生成成功返回值或 seed，故不能声称请求的 topology 成功、graph-mesh 一致、可重放或可用于训练；
- 完整三视图只支持检查实际 mesh extent 与 axis 对齐，不支持导航可行性或 GT 有效性；该 world 为失败证据，正式数据集、LiDAR、标签、训练样本均保持 0。

停止决定：Phase 1 不推进，不重跑随机 world，也不进入 Isaac。推荐下一步是另行审批 read-only audited adapter 单图：项目侧显式检查 tunnel 返回值、固定 RNG、调用已有 `WorldInfo` 和 per-tunnel axis accessor 导出标准 graph/spline/axis，并把单组件、watertight、manifold、无自交设为 mesh 硬门；不修改第三方核心。

## 最新追加结果：read-only adapter 的 GT 通过，原生 mesh 后端失败

证据目录：`results/gate0_baseline/gate0_20260810_cano_readonly_audited_adapter_smoke_v1_seed0/`。

- seed 0 下仅生成一次候选，没有整图重试；3 grown 与 1 connector 均返回标准 `(True, Tunnel)`；
- graph/spline/axis/WorldInfo/provenance 合同全部通过：52 nodes、52 edges、单组件、cycle rank 1；4 条 spline 共 1,679 点，端点最大误差 `5.69e-14`；1,042 条 tunnel-interior axis 行跨 ID 冲突为 0；
- 原 Cano mesh 含 58,825 vertices、117,675 triangles，只有 1 个组件且 vertex-manifold，但非 edge-manifold、非 watertight、不可定向并 self-intersecting，严格 mesh gate 失败；
- 首次 validator 计算完成后因 `numpy.bool_` JSON 序列化失败。旧验证器 hash、失败日志、deviation 和仅类型转换的修正均已保留；同一 bundle validation-only rerun 得到上述结论，没有重生成、改 seed、修 mesh、改指标或降阈值；
- E1/源码保持不变，外部合同测试 3/3、V3 测试 41/41、证据 manifest 27 项与 bundle hash 全部通过；正式 dataset/LiDAR/label/training 仍全为 0。

方法判断：adapter 已证明 Cano topology 可作为候选 GT 来源，但原生 Poisson mesh 不能作为本项目 geometry backend。推荐下一步保留 Cano topology/spline/GT，桥接项目已通过 navigation-grade 审计的 graph/spline-to-mesh 后端；方法和论文必须明确写成 `Cano topology + project geometry backend`，并另做单图 graph-mesh/footprint/replay smoke。未经用户确认不实施。

## 最新追加结果：24-pose 准备通过，Isaac 自定义 RTX 配置注册失败

证据目录：`results/gate0_baseline/gate0_20260810_cano_native_mesh_lidar_label_smoke_v1_seed0/`。

- 运行前冻结一张已有 seed-0 Cano world 和 24 个静态 pose：8 tunnel interior、8 junction transition、8 terminal approach；没有生成新 world；
- 5 m spline 球交标签全部满足预期 branch 数 2/3/1；CPU Open3D 在原始未修 mesh 上的 24/24 位姿净空和分支视线检查通过，保存了 24 个 CPU reference NPZ；这些仍是诊断证据，不是正式数据；
- Isaac Sim 6.0.1、RTX 5090 D、Warp 和 `isaacsim.sensors.experimental.rtx-1.4.6` 均成功启动；前两次包装失败分别暴露 CLI/异常可见性与容器 UID 写权限问题，原日志和 recovery JSON 全部保留；
- 权限修正后的决定性执行明确输出：`Config 'MTARE_VLP16_720_50M_V1' not found for OmniLidar`。随后 GenericModelOutput magic number 无效，300 rendered frames 内没有 fresh complete scan；
- 按冻结 stop condition 停止。RTX NPZ=0、正式 dataset world/sample=0、training sample/model=0；没有改用内置 sensor、CPU 伪装 RTX、mesh repair、换 world/pose 或降低阈值；
- 因无 RTX 数据，RTX/CPU 指标和 24 张 RTX 可视化均标为 `NOT_EXECUTED`，不能制作替代结果图。

当前判断：该结果否决的是“当前自定义 JSON 的注册路径”，不是 Cano mesh 的射线适用性本身。下一步先做只读 API/registry 审计，确认 Isaac 6.0.1 是要求 USD sensor asset、不同配置搜索目录，还是需显式注册扩展。任何修复规格必须保持同一 profile 和同一 24 poses，并重新获得运行确认；不得静默改用近似内置型号。

本轮已完成接口/世界/历史 baseline 的证据审计和机器可读草案，但尚未“冻结”。审计结果位于 `results/gate0_baseline/gate0_20260810_interface_benchmark_audit_v1_seed0/`，校验器通过，全部 V3 单元测试 18/18 通过。

最重要的新结论：准确替换单元是整个 `tare_planner_node`，保留独立 `localPlanner` 与 `pathFollower`；现有 5 个地下候选均有历史开发暴露，严格测试世界数量为 0；仿真与 TARE 存在随机源但启动文件没有 seed 接口；SubTGraph operational 01 原 TARE 记录为 0 m 运动，必须判 INVALID。

V3 clean-room 状态：已启用。当前没有 V3 训练数据集、teacher、模型或 checkpoint。旧数据和结果不得进入 V3 训练或选择流程。

问题反馈协议：已启用。发现问题必须停止受影响任务并先向用户报告；任何数据构建或训练必须先提交 data card 并获得用户确认。

## 最新的数据与方法结论（设计完成，未实施）

原 M-TARE 对比地图已明确为 `MTARE_BENCHMARK_ONLY`：只在模型和系统冻结后进行相同 world/start/seed/sensor/local-planner/runtime 的 parity comparison，严禁用于监督、SSL、归一化、AI prompt、人工开发审阅、阈值或 checkpoint 选择。由于历史实验已经接触这些地图，它们不算 strict unseen test；最终还必须有至少两个封存的新世界/拓扑族。

Gate 1 主路线固定为“新建程序化地下世界 + 客观 planner/map teacher + 因果 LiDAR student”。先从 topology graph 生成中心线、出口和连接真值，再生成同源 collision/free-space geometry，并由相同 geometry hash 导出 Isaac 和 Gazebo 版本。LAMP/SubT 只在逐 site 审计后作为可选真实域 SSL 或完整站点测试，不承担隐藏连接 teacher。

学习不是 AI 标签与对比学习二选一。第一版由完整几何、机器人 footprint、collision 和冻结 local-planner contract 监督 32 方向 `R(theta)`、`D(theta)`、exit component 和 `G_local`；AI 降为结构名称建议与审计工具，SSL 降为监督 baseline 之后的独立消融。学生候选输入是 40 m × 40 m、0.25 m/cell 的当前与 4 s 因果历史 terrain-relative BEV；这些尺寸仍需 Gate 0 的传感器与 planner horizon 审计确认。

正式数据前先申请修正后的 5-TNG perception contract pilot：每 TNG 一个 native perception mesh，共 5 个 mesh world、250 个 canonical anchor 和 750 个固定 yaw observation，只验证跨拓扑生成、sensor、teacher、防泄漏和科研可视化，不训练。原 2-geometry/1,500-observation paired pilot 延后到 navigation-grade collision backend 可做 footprint/planner certification 后。详细设计见 `docs/GATE1_DATA_METHOD_V1.md`、`docs/TNG_COUNTERFACTUAL_DATA_CONTRACT_V1.md` 和待审提案。

当前 TNG/Isaac/Gate-1 合同、治理、生成器和 mesh 测试合计 41/41 通过。Isaac Sim 6.0.1 Compatibility Checker 与 g001 单图 USD 导入已通过；该结果仍不表示 Gate 0 或 Gate 1 通过。

已发现直接的新颖性风险：已有 2026 工作已经实现程序化地下世界、合成 3D LiDAR、CNN 出口检测与纯拓扑导航。本项目不能只声称“从点云学出口并建图”；差异必须由 planner-consistent 结构事实与 uncertainty、部分观测下稳定 exit-stub 图、M-TARE 全局替换、多机器人分配和双重隔离评测证明。

### 最新实施收敛

固定主线为：合法复用或独立复现论文程序化生成器，输出 topology/centerline/common mesh；同源导出 Isaac USD 与 Gazebo SDF；Isaac 采 canonical LiDAR；先复现 Cano-like B1，再训练 causal-BEV `R/D/U` 主模型；冻结解释器形成结构语义；确定性状态机构建 node/edge/exit-stub；替换 `tare_planner_node` 高层并保留原 local planner/control。

本机已检测到 RTX 5090 D 32 GB、驱动 580.173.02 和 Docker，Docker GPU 透传探针通过。官方 Isaac Sim 6.0.1 镜像已拉取，digest 为 `sha256:783444c706538aa76cf5126e911ddc5e618779e6105305ad4af4260362a30aa9`；官方离线 Compatibility Checker 返回 `PASSED`。2022 生成器的论文 GitHub 地址目前不能匿名克隆，也没有找到可信归档；2024 新版 graph-to-mesh 代码仍未定位，因此采用不复制第三方实现的 clean-room TNG 核心。

clean-room TNG 核心现已实现独立 seed namespace、三维 RGTG 生成、CTG 闭环、最大度约束、采样边—边/节点—边净距、拓扑统计、canonical hash 和 TNG-parent split 防泄漏。第一次母图 `tng_faf57ba981e3c324` 在追加审计后因节点—非相邻边距离仅 1.089 m（合同要求 3.0 m）被否决并保留为负证据；抽象图 `tng_762e384fdc6b6fa0` 通过图阶段，但其 3.451 m 净距不足以容纳约 5 m 宽 tunnel mesh。

用户授权后已生成 mesh-aware parent `tng_3eca286d5e4c4cd3`，采样净距 6.894 m。`g000` 的 clean-room marching-tetrahedra mesh 离线通过：111,864 vertices、223,736 triangles、0 degenerate face、单一 watertight component、genus 3 与 cycle rank 3 一致，重放 hash 相同。Isaac v1 导入发现 USD 数组跨行缺逗号并保留为失败证据；修复后的 v2 经用户重新授权后在官方 Isaac Sim 6.0.1 中导入通过：10 次 update、有效 extent、Z-up/metre、碰撞 API 齐全。总测试 40/40 通过。

Isaac headless Replicator 的两次 PNG 输出均失败，第二次明确停在 renderer scheduled-frame/writer drain；所以没有把任何图片标为 Isaac RTX 截图。当前完整地图可视化直接读取已验收 OBJ 的全部 111,864 个表面顶点，展示 X–Y、X–Z 和斜视投影，并明确记录 offline provenance。

用户指出 `g000` 与未来仿真差距较大后，没有进入批量生成。navigation-grade 审计发现原 TNG 生成器的 CTG connector 漏掉坡度约束，导致历史 parent 的一条闭环边超过 0.22 rad；该 parent 和 `g000` 只保留为工程烟雾证据。修复后单独生成 `tng_84998d00587e03dc`，最大边坡度 0.180 rad，仍为 24 nodes、26 edges、cycle rank 3，净距 6.894 m。

新 `g001_navigation_grade` 使用 cubic-Hermite 水平中心线、按水平弧长分配高程、global-Z 地面剖面和扩大分支洞室。最终结果为 159,784 vertices、319,576 triangles、单一 watertight component、genus 3；最大曲线路径坡度 0.180 rad、最小转弯半径 2.837 m、曲线净距 6.857 m（要求 6.405 m），43,953 个半径 0.8 m 的机器人实体探针零失败；完整重放 hash 一致。官方 Isaac Sim 6.0.1 导入通过，并读取到 26 条/1,127 点导航中心线。四次被坡度、转弯半径和 genus 合同否决的失败均已保留。

容量上限候选仍为 80/10/10 个 train/validation/development-test TNG parent、每 TNG 至少两个认证 geometry variant：共 100 个独立拓扑、200 个 mesh、50,000 个 canonical anchor、100,000 个 geometry-specific place instance 和约 500,000 observation，按 20/40/80 train-TNG 学习曲线扩展。但该正式 paired 容量依赖 navigation-grade collision certification；当前先做待审的 5-TNG/5-mesh/750-observation perception pilot，仍不训练。M-TARE 官方五图、现有地下 benchmark 和 sealed test 均不进入训练。

详细实施见 `docs/ISAAC_STRUCTURAL_TOPOLOGY_IMPLEMENTATION_V1.md`，发表证据见 `docs/PUBLICATION_EVIDENCE_PLAN_V1.md`。仅系统拼接不足以发好期刊；RA-L 是完成严格闭环与真实验证后的现实目标，JFR 需要多 site field evidence，T-RO 需要一般性重大算法贡献。

## 已冻结的未来学习路线

用户已批准将 Gate 1--3 调整为混合监督路线。原始事实来源是地下 LiDAR 点云、时间戳、位姿和真实 ray origin；第一版学生输入是点云构建的 terrain-relative causal 2.5D BEV；完整地图、collision 和 M-TARE local planner 生成方向可通性、可达距离、出口与局部连接等核心监督。

AI 只对标准 annotation bundle 提出多标签结构属性、置信度、abstain 和异常候选，必须经硬规则与人工金标审计。AI 标签不允许静默覆盖客观 teacher。Gate 2 主路线改为监督式轻量 BEV 结构表示，自监督降为可选 warm start/正则；直接点云 encoder 只有 BEV 被证据否定后启用。

总方法说明：`docs/SUPERVISED_STRUCTURE_LEARNING_V1.md`。精确数据合同：`docs/GATE1_DATA_METHOD_V1.md`。这项方法调整不授权 Gate 1，也没有建立数据集或模型。

本次方案一致性与治理测试 13/13 通过，证据位于 `results/gate0_baseline/gate0_20260810_supervised_pointcloud_ai_strategy_v1_seed0/`。

## 本轮已经完成

在不移动或删除旧资产的前提下，已经建立 V3 正式源码/配置/测试骨架、仓库结构说明、实验规程、data card 模板、旧代码/结果地图和失败知识库。实验治理已变成可执行检查：run 必须绑定当前 Gate 与用户批准，数据操作必须绑定已批准 data card，strict test 不能泄漏，已有结果不能覆盖，状态工具不能推进 Gate。

治理测试结果：10/10 通过，证据位于 `results/gate0_baseline/gate0_20260808_repository_scaffold_governance_v1_seed0/`。本轮没有导出数据、创建 `.npz`、训练模型、运行 ROS/仿真或启动 closed-loop。

这次结构重构不改变 Gate 判断。Gate 0 仍为 `GATE_MIXED`，因为治理工具不能替代 interface contract、benchmark 冻结和有效 baseline。

工程风险：项目根目录的 `.git` 是空目录，当前不是有效 Git 仓库；两个 `external/` 子项目各有独立 Git。进入旧代码实质迁移前需要决定是否初始化根仓库以及如何管理外部仓库。本轮没有改 Git 元数据。

## Gate 0 已有证据

- 已从固定 Docker image 内原始源码核验：`tare_planner_node` 以 1 Hz 读取注册点云、扫描时刻位姿、terrain map 与反馈，发布 map-frame `/way_point`；
- 已核验独立 `localPlanner` 接收 `/way_point` 和 `/terrain_map`，发布 vehicle-frame `/path`，`pathFollower` 再发布 `/cmd_vel`；
- 已形成 15 项 interface contract 草案，包含 topic、type、frame、producer、consumer、rate、状态与证据；
- 已完成 9 个 world 的资产 hash、地下相关性、历史暴露和资格清点；
- 已将 9 条历史原 TARE run 分类为 VALID_CANDIDATE、INVALID、INSUFFICIENT_EVIDENCE 或 MISSING；
- 已形成 3 个主开发 world、2 个开发回归 world、1 个 smoke world 和 2 个空 strict-test 槽位的 proposal。

## Gate 0 缺少的关键证据

- interface contract 仍是 `DRAFT_FOR_USER_REVIEW`，`/terrain_map_ext` 是否进入公共全局接口、completion、runtime 和目标级 local rejection 尚未冻结；
- 当前没有任何 demonstrably clean strict-test world，必须新增并隔离两个地下世界；
- Gazebo/LiDAR/TARE 的统一 seed 控制尚未实现和验证，不能做成对重复统计；
- reference-surface coverage、reachable denominator 和 coverage-time AUC 尚未实现和可视化核验；
- SubTGraph operational 01 的 spawn/terrain/no-motion 故障未定位，operational 02 缺原 TARE baseline；
- 没有任何历史 run 可直接进入最终 mean/std；每个正式 world 仍需 5 次、600 s 的受控重跑；
- 多机器人消息只完成原接口 inventory，V3 语义拓扑分配消息留到 Gate 7。

## 现有结果的真实定位

- Gate 1：输入对齐和部分 teacher 审计可复用，但 teacher 与 planner 一致性、地下数据规模和 split 不满足 V3，状态 GATE_MIXED。
- Gate 2：旧 bottleneck/embedding 有 feasibility 信号，但未按 V3 数据和 unseen-world 协议验证，状态 GATE_MIXED。
- Gate 3：多版语义模型曾明确 FAIL；地下同地图轨迹结果不能替代新 world 证据，状态 GATE_FAIL。
- Gate 4：已有离线节点原型，但 robustness、exit preservation 和图质量证据不足，状态 GATE_MIXED。
- Gate 5：已有 shadow 运行，但缺系统稳定性统计，状态 GATE_MIXED。
- Gate 6：已有 closed-loop 原型，但不是有效公平对比，状态 GATE_FAIL。
- Gate 7：多机器人语义拓扑协同尚未开始，状态 GATE_FAIL。
- Gate 8：最终对比尚未开始，状态 GATE_FAIL。

## 当前推荐路线

主 Route：按 `docs/PLAN.md` 只推进 Phase 1。V2R 已冻结 100 个 topology parent 及 80/10/10 split。M0 已否定 native OBJ 字节重放，下一步只评审 M0R 的 noise-bounded geometric replay；不进入 M1、LiDAR 或训练。

clean-room `g000/g001` 继续作为生成、USD 和 Isaac 接口的工程预检与失败知识，不计入 Cano Phase 1 world 数，也不触发动力学/LiDAR 主线。

## 下一步唯一任务

> 核验并决定是否批准 `configs/v3/gate0/cano_100_parent_perception_mesh_contract_m0r_geometric_replay.proposal.json`。唯一变化是 replay metric：输入/axis exact，双向 nearest-vertex max `<=0.75 m`、AABB 端点差 `<=0.5 m`、面积差 `<=1%`；parents/seeds/native method/20 mesh/10 train 图和所有零范围不变。

Gate 0 通过前不会设计新 `.npz`、训练模型或选择网络；Gate 1 开始后也将从原始数据重新构建，而不是修补旧 v3/v4/v5 数据集。

## 关键路径

- 总纲：`docs/MASTER_PLAN_V3.md`
- 当前权威执行计划：`docs/PLAN.md`
- 当前阶段进度：`docs/PROGRESS.md`
- 仓库结构：`docs/REPOSITORY_STRUCTURE.md`
- 实验规程：`docs/EXPERIMENT_PROTOCOL.md`
- 数据卡模板：`docs/DATA_CARD_TEMPLATE.md`
- 混合监督学习方案：`docs/SUPERVISED_STRUCTURE_LEARNING_V1.md`
- Gate 1 数据与方法合同：`docs/GATE1_DATA_METHOD_V1.md`
- Isaac 结构语义拓扑实施蓝图：`docs/ISAAC_STRUCTURAL_TOPOLOGY_IMPLEMENTATION_V1.md`
- 论文贡献与发表证据计划：`docs/PUBLICATION_EVIDENCE_PLAN_V1.md`
- 论文学习路线：`docs/RELATED_WORK_READING_GUIDE.md`
- 单图烟雾测试运行方法：`docs/RUN_SINGLE_TNG_MESH_SMOKE.md`
- Navigation-grade 几何方法与结果：`docs/NAVIGATION_GRADE_GEOMETRY_V1.md`
- 决策日志：`docs/DECISION_LOG.md`
- 机器状态：`results/project_status.json`
- Gate 审计：`results/v3_gate_audit/`
- 本轮 Gate-0 接口与 benchmark 草案：`results/gate0_baseline/gate0_20260810_interface_benchmark_audit_v1_seed0/`
- 旧资产地图：`legacy/`

## 2026-08-11 最新阶段覆盖说明

上文“当前推荐路线/下一步唯一任务”中的 M0R 状态已过时。当前权威状态如下：

- M0F 已 PASS 并封存十个 train recipe sentinel 的不可变原生 perception mesh；复核见 `docs/CANO_100_PARENT_PERCEPTION_MESH_M0F_REVIEW.md`。
- 这不是训练集：正式 LiDAR、监督标签、NPZ 样本、CNN/GNN、在线结构语义图和 M-TARE 高层替换仍全部为 0。
- 当前唯一待决策项是 M1 全 100 parent 物化提案：`configs/v3/gate0/cano_100_parent_perception_mesh_m1_immutable_assets.proposal.json`。
- M1 proposal/card 均为 `PENDING_USER_APPROVAL_NOT_IMPLEMENTED_NOT_EXECUTED`。批准前不得实现或执行；M1 PASS 后也只能另提 CPU LiDAR 与客观标签合同。

## 2026-08-11 最新阶段覆盖说明：M1R 全量感知资产 PASS

上文 M0R/M1 pending 状态均已过时。M1 首次全量运行在第 45 个 parent 因 1 个零面积三角面严格停止；随后批准的 M1R 从头物化全部 100 parent，并对每个 OBJ 强制执行只删除 `doubled_area <= 1e-12` face 行、顶点精确不变的 sanitation 合同。

M1R 正式 PASS：100/100 primary、100/100 sanitation、0 replay，10 recipe 各 10 个，split 为 80 train/10 validation/10 development-test。总规模 `12,608,078` vertices、`25,217,511` triangles、0 degenerate；100 个 final OBJ hash 唯一，1001 项 evidence 封存。80 张 train 图全部人工核验通过且标记为 `M1R TRAIN ONLY`；validation/development-test 未渲染、未人工查看。

本次 sanitation 删除数为 0，因为 M1 的单个共线面未在全量重生成中重现；这被解释为 Poisson 运行级非确定性。正式结论只覆盖当前封存资产，不宣称生成器逐次确定，也不把这些资产升级为动态导航碰撞网格。

当前正式 dataset、LiDAR observation、teacher label、training sample、model、trajectory、online graph 和 M-TARE change 仍全部为 0。下一步只准备 CPU synthetic ↔ Gazebo fixed-pose LiDAR parity 审批材料；parity 与独立 Data Card 通过前禁止批量正式采样和训练。

## 2026-08-12：固定姿态 CPU↔Gazebo LiDAR parity 正式 PASS

用户批准的 run `gate0_20260812_cano_cpu_gazebo_fixed_pose_lidar_parity_v1_seed0` 只读使用 M1R 的 S01/S06/S10 C01 三个 train parent。24 个 pose 在任何扫描前由 graph/spline/FTA 固定，聚合为 8 tunnel-interior、8 junction-transition、8 terminal-approach；每 pose 保存 1 个 Open3D CPU reference 和丢弃 2 个 warm-up 后的 3 个 Gazebo Classic CPU-ray scan。

解析 box 首先 PASS。Cano 24/24 pose 均通过冻结的 shape/frame/ring、重复性、valid agreement、MAE、P95、P99 和水平最小距离门槛。全批最差指标为 valid agreement=`1.0`、MAE=`8.0222507e-6 m`、P95=`2.4795532e-5 m`、P99=`4.6730042e-5 m`、水平最小距离差=`8.4668398e-5 m`，三次重复最大差为 0。6 页可视化覆盖全部 24 pose；70/70 evidence hash 和无遗留 parity 容器检查通过。

结论边界：这证明固定静态理想射线合同的 CPU/Gazebo 一致性，不证明噪声、动态运动、学习、结构语义、在线拓扑或规划收益。7 个 parity NPZ 是诊断证据，不是数据集。正式 dataset/teacher/training/model/trajectory/online graph/M-TARE change 继续为 0。项目保持 Gate 0 `GATE_MIXED`，等待用户确认该阶段结果；不得自动推进。

## 2026-08-12：Phase 2 监督 Range 数据方案待弧长确认

用户已经确认 parity 阶段结果并允许起草方案。当前拟定 80 train/10 validation parent、22,500 place cluster、112,500 frame；学生只读 16×720 range/valid，客观 teacher 使用 5 m spline outgoing branch 与 mesh LOS。C10 development-test、M-TARE、历史 NPZ 和 parity NPZ 均不进入该操作。方案见 `docs/PHASE2_SUPERVISED_RANGE_DATA_V1.md`。

冻结轨迹清单时发现：Cano `distances[]` 与封存 `points[]` 的累计欧氏弧长存在系统差异。train 为 121,386.151 m 对 122,176.571 m，validation 为 15,721.222 m 对 15,826.232 m；单 world 最大差 1.181%。这只使采样距离口径和 per-world quota 尚不能冻结，不否定 parent split、事件数和总体容量。推荐用 `points[]` float64 累计欧氏弧长，因为实际 pose、mesh raycast 和 teacher 都在这些坐标上工作。

当前等待用户明确确认该口径。正式 dataset、teacher、training、model、trajectory、online graph 和 M-TARE change 仍全部为 0；没有 run spec、preflight、create_run 或数据执行。

弧长口径随后已获用户明确批准。继续冻结 registry 时发现 role-overlap 缺陷：junction-first 互斥分层使 train 22 个和 validation 8 个 terminal 事件虽有附近 incident-tunnel 候选，却没有 terminal-role 候选。推荐保留 near-junction/near-terminal 多标签，以 terminal-first primary role 仅做配额；候选容量仍高于固定目标且可恢复全部事件的同角色覆盖。当前等待该规则确认，registry 尚未创建。

用户随后确认 multi-label + terminal-first primary role。90-world registry 已冻结：80 train/10 validation、22,500 clusters、112,500 frames；每 world 的 points-based 长度、三角色容量/配额、事件数和最小候选数均明确。机器审计为 90 ID 唯一、配额精确闭合、0 超容量、junction/terminal 最小候选 3/1、C10 row=0。registry SHA-256=`d64edd758d86d1192dff870cdba08d392388cb48cfd755f428acdfb3ff4e8420`。当前只等待完整 Data Card 的数据实现/导出批准。

## 2026-08-12：Gate 1 数据导出机器 PASS，但科研证据复核为 GATE_MIXED

批准的唯一正式 run `gate1_20260812_cano_phase2_supervised_range_dataset_v1_seed0` 已完成且执行器返回 `PASS_CANO_PHASE2_SUPERVISED_RANGE_DATASET_V1`。它只读使用 80 个 train 与 10 个 validation parent，C10 development-test 和 M-TARE benchmark 读取均为 0；导出 22,500 个 place cluster、112,500 个 16×720 range/valid frame、112,500 份 objective teacher，形成 90 个 Zarr shard。train/validation primary-role 配额精确为 `16350/3000/650` 与 `2050/375/75`（interior/junction/terminal），训练样本消费和模型数均为 0。运行约 1425.35 s，封存前约 3.34 GB，正式 run evidence seal 为 12,790/12,790。

正式运行后的独立科研复核发现证据合同缺口，因此不得把执行器 PASS 升级为 Gate 1 scientific PASS：

- 每帧 independent-scene replay 在执行路径中被强制检查，但 `manifest.jsonl` 与 summary 没有持久化逐帧 pass/max-difference 或总体最大值，无法仅凭封存证据独立复核 112,500 次 replay；
- 61 个 rejected candidate 只保留逐 world 数量，没有 candidate ID 和具体拒绝原因，replacement 路径不可完整审计；
- 10 页 train preview 只显示 range 与 role/branch count，没有已批准设计要求的 valid mask、teacher heading/target 和 LOS；
- 10 张 `train_recipe_coverage` 实际是 C01--C08 selected-cluster 数量柱状图，不是空间 spline/cluster coverage；整体 role/branch/range/valid distribution 图缺失。

问题分类为 evidence/metric/visualization implementation，不否定已封存数据的数量、split、角色配额、最低 clearance、LOS runtime checks 或零 benchmark 泄漏。当前 Gate 1 结论固定为 `GATE_MIXED`，现有 3.34 GB 数据保持只读且不重导出。推荐另立一次 corrective evidence audit：读取现有 Zarr/manifest，按相同封存 mesh 对已选帧做独立 scene 复投并持久化误差，确定性重建 rejected candidate 明细，并生成包含 range、valid、teacher、LOS 的样本页、真实空间覆盖图和分布图。预计约 24 分钟、额外证据不超过约 1 GiB、训练为 0；未经用户明确批准不实现、不 preflight、不执行。

## 2026-08-12：纠正证据审计 PASS，但空间图否定 V1 selector

用户批准的一次只读 corrective audit 已正式完成。run `gate1_20260812_cano_phase2_dataset_evidence_audit_v1_seed0` 在约 913.71 s 内机器 PASS：90/90 worlds、22,500 selected clusters、112,500 frames、61 rejected candidates 全部精确重建；selected ID/order 与 sealed manifest 完全一致；stored→scene-A 和 scene-A→scene-B 最大 range 差均为 0，replay/teacher failures 均为 0；634 junction 与 575 terminal event 保持覆盖。原 dataset 12,790 项 seal 在运行前后完全一致，M1R 1,001 项 seal 通过；新 run 126/126 seal，约 62 MB，零训练/模型/C10/M-TARE 读取。

10 页 range+valid+teacher+LOS、10 张真实 spline/cluster coverage 和 1 张分布图均成功生成。人工复核这些正确空间图时发现新的 data-selector 缺陷：当前 event-first 后按 role 内固定 candidate 顺序填满 quota，未执行 per-tunnel 或 arc-length 均衡。90 个 world 中 50 个至少有一条 tunnel 完全无 selected cluster，共漏 58 条 tunnel；train 为 46/80 world、54 tunnel，validation 为 4/10 world、4 tunnel。即使只看已有 selected 的 tunnel，各 world 最大候选→最近 selected 同 tunnel 弧长距离中位数为 80 m、P95 为 222 m、最大 330 m；候选格本身是 5 m。

结论：corrective evidence audit 本身 PASS，证明现有 V1 数据、teacher 和替换路径可复现；但 V1 dataset 的科研可用性 FAIL，因为数量/事件配额不能替代空间代表性。Gate 1 保持 `GATE_MIXED`，该 V1 dataset 禁止训练。推荐下一步仅做零 ray/零数据 selector-V2 feasibility audit：保留 80/10、22,500 clusters、角色配额和事件覆盖，加入每 tunnel 最低覆盖、按长度配额和同 tunnel 最远点填充；先报告可实现覆盖半径，再决定是否批准约 3.2 GB 的 V2 重导出。未经用户新批准不实施。

## 2026-08-12：零-ray selector V2 可行性正式 PASS

用户以“继续做”批准一次 selector-V2 implementation + formal feasibility audit，边界为零 mesh、零 raycast、零数据导出、零训练。V2 对每个 world 使用稀疏二元 MILP，同时固定原 interior/junction/terminal 配额、长度比例 per-tunnel 整数配额、junction/terminal 事件覆盖和每个 5 m candidate 到同 tunnel selected 的 `<=10 m` 覆盖约束；每个 world 独立求解两次作 deterministic replay。

正式 run `gate1_20260812_cano_phase2_selector_v2_feasibility_audit_seed0` PASS：90/90 world、27,247 candidate、22,500 selected；80/10 split 不变，634/575 junction/terminal events 全覆盖，0 tunnel 缺失，0 deterministic replay failure。89 个 world 最大覆盖距离为 5 m，仅 S03 C01 为 10 m；V1 baseline 为 50/90 world 共漏 58 tunnel、最大 330 m。126/126 unit regression 先通过；正式 run 约 7.11 s、14 MB、112/112 seal，M1R 1,001 项和源 dataset 12,790 项 seal 均通过。10 张 train-only 全图人工抽查确认长灰段消失，validation 未渲染。

V1/V2 selected ID overlap 为 18,662，V1-only/V2-only 各 3,838，Jaccard=0.70856。因为 17.1% 的 cluster 身份变化，不能只改 manifest 或把旧 Zarr 冒充 V2。selector V2 feasibility PASS 不等于 V2 dataset 已生成；当前正式可训练 dataset 仍为 NONE，Gate 1 保持 `GATE_MIXED`。下一步只能起草并审批 V2 Data Card/完整重导出（预计 25--30 分钟、约 3.2 GB、零训练），并从导出时直接保存 replay/rejection/完整可视化证据。

V2 Data Card 与 proposal 随后已起草并保持 pending：`configs/v3/gate1/data_cards/cano_phase2_supervised_range_dataset_v2.json` 和 `configs/v3/gate1/cano_phase2_supervised_range_dataset_v2.proposal.json`。关键修正是先对全部 27,247 candidate cluster / 136,235 candidate frame 做双 scene eligibility，再在 eligible subset 上重新求解冻结 V2 约束，最终存 22,500 cluster / 112,500 frame；禁止 same-role sequential replacement。卡的结构性 approval-probe 校验为 0 error/0 warning，但尚无用户批准，故未实现 exporter、未 preflight、未 create_run、未导出数据。

## 2026-08-12：V2 全候选导出在第 23 个 world 因冻结 tunnel quota 不可行而 FAIL

用户明确批准一次 V2 exporter implementation 和一次正式全量导出。批准记录、专用 exporter/runner、eligibility-aware selector 和 run spec 均已冻结；104/104 V3 unit、Data Card validator 和 preflight 全部 0 error/0 warning。一世界真实冒烟中 S01 C01 的 163 个候选有 2 个因 clearance 淘汰，剩余集合仍可满足冻结约束。

唯一正式 run `gate1_20260812_cano_phase2_supervised_range_dataset_v2_seed0` 按顺序执行全候选双 scene 审查、合格集重求解和最终 shard 写入。前 22 个 world 完成；第 23 个 `S03_flat_unicyclic_small_C05` 中 tunnel 1 原有 38 个候选，7 个因至少一帧 `minimum_horizontal_clearance_below_0p8m` 被客观淘汰，只剩 31 个，但 selector feasibility 阶段在未知 eligibility 时冻结的 tunnel quota 要求 32 个。运行按 stop rule 在 MILP 前停止，没有改阈值、补样本、换配额或训练。状态为 `FAIL_CANO_PHASE2_SUPERVISED_RANGE_DATASET_V2`，约 223.12 s、500.10 MB、22 个完成 shard、2,943 项 seal；训练/模型/C10/M-TARE 读取均为 0。

该结果否定的是“在 eligibility 之前冻结精确 tunnel quota”这一数据选择顺序，不否定 clearance/LOS teacher、80/10 split、角色配额或 V2 空间覆盖思想。只读反事实显示：若在全部 eligibility 已知后按 eligible capacity 重新做同一长度比例整数配额，该 world 的 tunnel quota 从 `32/48/38/23/10` 变为 `27/50/39/24/11`，仍可选 151 个、保持角色配额、10 个事件和 5 m 最大覆盖；这只是诊断，不是获准方法或正式结果。

Gate 1 继续 `GATE_MIXED`，V1 仍禁止训练，失败 V2 partial artifacts 只作失败证据。推荐 V2R：先完成 90 world 全部 eligibility 并封存，再从每 world eligible capacity 计算长度比例 tunnel quota，仍固定角色配额、全事件、每 tunnel 和原候选格 `<=10 m` 覆盖；任何 world 不可行即 FAIL。需要新的 Data Card 和用户明确批准，禁止自动重跑。

## 2026-08-12：V2R eligibility-first 正式数据集机器 PASS，等待用户核验 Gate

用户以“赶紧搞啊这节奏太慢了”明确批准推荐 V2R 一次实施和一次完整重跑。V2R 保持 80 train/10 validation、原角色配额、64 个客观拒绝、634/575 结构事件、0.8 m clearance、完整 LOS、双 scene replay、每 tunnel 和原候选格 `<=10 m`；唯一方法修正是在每个 world 全候选 eligibility 已知后，按 eligible capacity 计算长度比例 tunnel quota。105/105 V3 unit、Data Card validator 和 preflight 均 0 error/0 warning。

正式 run `gate1_20260812_cano_phase2_supervised_range_dataset_v2r_seed0` 机器 PASS：90/90 worlds，27,247 candidate clusters、136,235 candidate frames、272,470 双 scene scans；27,183 eligible、64 rejected，最终精确选 22,500 clusters/112,500 frames，train/validation 为 20,000/2,500。角色精确为 train `16,350/3,000/650`、validation `2,050/375/75`（interior/junction/terminal）。634 junction 与 575 terminal events 全覆盖，最大原候选→同 tunnel selected 距离为 10 m；selector deterministic replay、selected replay、selected teacher failure 均为 0。

数据保存为 90 个 Zarr shard；完整证据包括 136,235 frame audits、27,247 eligibility、64 rejection、22,500 cluster manifest、112,500 frame manifest、10 页完整 train 样本、10 张 train 空间图和 1 张 train 分布图。运行约 1,933.50 s，封印前 3,478,508,257 bytes，最终目录约 3,482,677,507 bytes；19,338/19,338 SHA-256 复核通过。C10/M-TARE 读取、训练样本消费和模型数均为 0。

当前可训练候选数据集更新为 V2R，但按治理不得自动进入 Phase 3 或宣称 Gate 1 最终通过。Gate 1 暂保持 `GATE_MIXED`，等待用户核验数据卡、空间图和分布图并明确确认。下一步唯一任务是对 V2R 结果做人工/科研核验；确认后才可另立 Phase 3 CNN baseline 训练方案与批准，不启动 GNN、拓扑图或 M-TARE 修改。

## 2026-08-12：用户确认 Gate 1 并批准 Phase 3 多任务正式训练

用户在收到 V2R 完整结果及 Phase 3 多任务结构语义训练卡后回复“继续做”。Gate 1 以 V2R `PASS_CANO_PHASE2_SUPERVISED_RANGE_DATASET_V2R` 收口，operational Gate 切换为 2。批准的一次训练执行固定为 B1/M1 × seeds 0/1/2、最多 30 epochs，严格只读 80 train/10 validation；C10、M-TARE、在线图和规划修改为 0。训练执行前仍必须冻结 tools/source hash、通过全量测试和 preflight。

## 2026-08-12：Gate 2 正式命令在训练前环境探针暂态失败

冻结 run spec 的 preflight 为 0 error/0 warning，正式目录 `gate2_20260812_cano_phase3_multitask_structural_semantics_v1_seed0` 创建并按冻结命令启动。命令在运行器的 environment-identity 子进程立即返回 1；此时训练 loader、优化器和 epoch 均未开始，optimizer step、checkpoint 和模型结果均为 0。

原运行器虽使用 `capture_output`，但异常路径没有持久化 stderr，故第一次返回 1 的具体底层原因不可恢复。同一只读 identity 命令随后原样执行成功，确认 torch 2.9.0+cu129、CUDA 12.9、RTX 5090 D、Zarr 2.18.7 等冻结身份当前匹配，且没有残留 GPU 训练进程。因此证据只支持“启动瞬间的未诊断暂态探针失败”，不支持更换数据、模型或环境。

本次 run 固定为 `FAIL_PRETRAINING_ENVIRONMENT_PROBE_TRANSIENT_UNDIAGNOSED`，不能产生 Gate 2 科研结论。推荐只修复 runner 的探针 stderr/stdout 持久化与一次有界重试，重新冻结 hash/preflight，并由用户明确授权一次 replacement execution；数据、B1/M1、种子、epoch、loss 和验收阈值均不改变。当前停止训练工作等待决定。

## 2026-08-12：replacement 启动保存完整证据并定位 MKL 进程环境冲突

用户回复“继续”批准 runner-only 修复和一次 replacement execution。新 runner 保存每次 environment probe 的 stdout/stderr/return code/duration，并最多在 2 秒后重试一次；训练环境 13/13 专项测试通过，E1/Open3D 环境 118/118 回归通过（4 个 PyTorch 项在 E1 按设计跳过、已在训练环境通过），replacement preflight 0 error/0 warning。

正式 `gate2_20260812_cano_phase3_multitask_structural_semantics_v1r_seed0` 的两个 probe 均在约 0.25 s 返回 1，完整 stderr 一致为 `MKL_THREADING_LAYER=INTEL is incompatible with libgomp.so.1`。运行器按冻结停止条件退出；训练仍未开始，optimizer/epoch/checkpoint 均为 0。

只读根因审计证明：外层 shell 与 Python `os.environ` 都看不到 `MKL_THREADING_LAYER`；但 runner 导入 Matplotlib/NumPy 后，不显式传 `env` 的子进程稳定继承到 C 层 MKL 环境并失败。相同 child 显式使用 `env=os.environ.copy()` 后成功；NumPy-first 导入也成功。正式 trainer 本身 NumPy-first，runner 启动 trainer 已使用 `env=os.environ.copy()`，其 `--help` 合同复现通过。

因此问题固定为 environment probe 的 process-environment contract，不是数据、teacher、模型或指标问题。推荐只让 probe 与 trainer 一样显式传 `os.environ.copy()`；不设置 `MKL_THREADING_LAYER=GNU`，不改变数学运行时。第二次授权已消费，当前停止，等待用户批准该精确修复与新的 replacement execution。

## 2026-08-12：正式训练进入模型阶段后因 CuBLAS 确定性合同停止

用户回复“继续做”批准 probe 显式 `env=os.environ.copy()` 和新 replacement。13/13 Phase-3 专项测试、真实环境探针与 preflight 0/0 通过；`v1r2` 首次真正进入 GPU 训练。

B1 三种子完整结束，最佳方向 F1 分别为 `0.901932/0.894351/0.903747`，均超过冻结 B0=`0.795709`；epochs 为 13/12/11，strict-test 与 M-TARE 读取为 0。M1 seed 0 完成 13 epochs，最佳方向 checkpoint 为 epoch 7：方向 F1=`0.820010`、role macro-F1=`0.875548`、三类 recall=`0.929/0.949/0.925`、count 1--4 macro-F1=`0.675744`、同 cluster cosine mean/p05=`0.980767/0.903407`。固定 masking cosine=`0.205687`，旋转 direction max logit error=`0.004879`，两项未达门槛。

M1 seed 0 日志同时出现 PyTorch 明确警告：虽然调用了 deterministic algorithms，但 CuBLAS 因进程启动前未设置 `CUBLAS_WORKSPACE_CONFIG` 而不确定。该警告只在多任务线性 head 反向传播触发。它否定 M1 seed 可重放的 integrity 合同，故在 M1 seed 1 首个 epoch/checkpoint 落盘前人工停止；无残留进程，M1 seed 2 未启动。

本 run 固定为 `GATE_FAIL`（system/reproducibility integrity），不得把 M1 seed 0 诊断指标升级为三种子结论，也不得在原目录续跑。推荐只在 trainer child 启动前冻结 `CUBLAS_WORKSPACE_CONFIG=:4096:8`，先做真实 M1 backward 无警告验证，再经用户新批准从头执行完整六子运行；数据、模型、loss、阈值和 checkpoint 规则不变。

## 2026-08-12：Gate 2 v1r3 deterministic replacement 完整执行，结论 GATE_MIXED

v1r3 已在 `CUBLAS_WORKSPACE_CONFIG=:4096:8` 下从头完整执行 B1/M1 × seeds 0/1/2。六个子实验 exit code 均为 0，run 状态 `COMPLETED`，执行完整性 PASS；62 项 evidence SHA-256 全部复核通过，结果约 86 MB，GPU 已释放，strict C10、M-TARE、graph 和 planner 读取/修改均为 0。

M1 direction F1=`0.820010/0.868793/0.873366`，中位数 `0.868793`，优于冻结 B0=`0.795709`；role macro-F1 中位数 `0.875548`，count 1--4 macro-F1 中位数 `0.760694`，同 cluster cosine mean/p05 中位数 `0.980767/0.899985`，均通过。固定 10% 方位列 masking 的 `z_role` cosine 中位数仅 `0.205687`，未达到 `0.85`，因此 Gate 2 为 `GATE_MIXED`，不能进入 Phase 4。

只读 256-frame no-update 诊断显示，6° rotation 的 CPU maximum absolute logit error 为 `6.68e-6/6.68e-6/7.63e-6`，满足 `<=2e-5`；CUDA 为毫量级，而 `z_role` rotation cosine 在两种设备均约 1。故 rotation 失败归因 metric/device numerical contract；masking 低值在 CPU/CUDA 一致，归因 model/input robustness。该临时诊断不是 formal run，后续需另立 no-update audit 证据。

当前推荐：保持 Gate 2，先批准 formal no-update stability audit；若结果确认，再批准只增加 train-only ray-column dropout augmentation 的 corrective 训练。不得降低 masking 阈值，不得改变数据/split/teacher/network/head/loss/checkpoint，不得进入图或 M-TARE。

## 2026-08-13：Phase 3 局部结构语义模型通过 Gate 2

正式 corrective 采用与 M1 完全相同的 circular CNN、128D `z_role`、direction/count/role heads、loss、AdamW、checkpoint 指标和干净 validation，只在训练输入中加入 `p=0.5`、随机 phase、modulo-10 的 10% ray-column dropout。数据仍为 80 train/10 validation worlds、100000/12500 frames；C10、M-TARE benchmark、图和规划器读取均为 0。

原 run 因主机休眠触发外层墙钟 timeout，仅留下完整 seed0/1。新 recovery run 不覆盖旧证据，只复用经哈希和配置校验的 seed0/1，并按同一合同补 seed2。三 seed 聚合方向 F1=`0.887325`（floor `0.870524`）超过 B0=`0.795709`；role/count 中位数=`0.882346/0.746956`；同 cluster mean/p05=`0.982660/0.902049`；masking cosine=`0.999586`；CPU rotation max error=`9.54e-6`。全部冻结门槛通过，Gate 2=`GATE_PASS`。

证据目录：`results/gate2_representation/gate2_20260813_cano_phase3_masking_corrective_recovery_seed2_v1_seed2`，19/19 seal 通过。当前模型证明的是未见 procedural topology 上的局部 outgoing direction、branch count、structure role 与稳定结构表示；尚未建立正式在线拓扑图，也未替换 M-TARE 高层规划。当前实施上限仍为 Phase 3，等待用户明确批准后才可规划下一 Gate。

## 2026-08-13：S10 局部 floor-following pose feasibility audit 完成

用户批准严格 diagnostic Data Card 治理修正后，正式 run `gate4_20260813_cano_s10_local_floor_pose_feasibility_v1_seed0` 完成。审计单位是 9 个非独立 route failure pose，不把 549 个参数候选伪称为独立样本；Data Card 因此保持 raw/effective=`9/9`，另记 parameter evaluations=`549`。治理测试与相关专项测试 `33/33` 通过，preflight 0 error/0 warning。

高度扫描为每位姿 61 个 z offset，每候选 720 条水平射线加上/下竖直射线。frame 6、39、2551 分别在 `+0.40..0.45 m`、`+0.05..0.10 m`、`+0.20..0.25 m` 出现可行区间；frame 105、108、600、676、2507、2519 无可行 z。未解点在贴地高度下的最佳水平净空为 `0.7355/0.7677/0.7858/0.7887/0.6511/0.5814 m`；将 z 抬到水平净空合格时，向下距离又变为 `1.3515/1.1496/1.0950/1.0959/1.3047/1.1208 m`，不再满足 `1.00±0.05 m` floor-following 合同。

因此完整轨迹不能只靠 z 修正。这是 data/pose geometry 问题，不是模型或 teacher 问题。正式 run `COMPLETED/PASS_CANO_S10_LOCAL_FLOOR_POSE_FEASIBILITY_AUDIT_V1`，21/21 seal；PASS 仅表示审计执行完整，Gate 4 继续 `GATE_MIXED`。下一步推荐局部 x/y/z 可行性审计，未经新批准不执行。

## 2026-08-13：S10 局部 lateral-pose feasibility audit PASS

用户明确批准后，正式 run `gate4_20260813_cano_s10_local_lateral_pose_feasibility_v1_seed0` 对9个失败pose执行`21×21`x/y网格。Data Card仍以9个route pose为raw/effective单位，3,969只记为parameter evaluations。26/26专项测试、preflight 0/0、正式run 21/21 seal全部通过，执行约2.42s。

每个候选x/y先从mesh求局部floor，把sensor放到floor上1m，再核验720条水平ray和上/下证据。frame 6/39/105/108/600/676/2507/2519/2551的可行单元数为`441/441/271/297/343/342/271/414/441`，全部属于至少3单元的四邻域连通块。对原失败点的最近稳健修正最大只需`0.224m`。

结论：S10 perception mesh存在广泛可行局部通道，不需重生成；旧固定FTA高度/中心线位姿合同才是问题。但局部点可行不等于连续轨迹已修复，Gate 4仍为`GATE_MIXED`。下一步需另行批准完整floor-following+平滑lateral corrective trajectory contract。

## 2026-08-13：完整 floor-following trajectory corrective 因 S10 地面支撑孔洞 FAIL

用户批准后，28/28专项测试和preflight 0/0通过，唯一正式run `gate4_20260813_cano_c08_floor_following_trajectory_corrective_v1_seed0` 启动。S01和S06分别801/801、1414/1414全帧合格；S10在floor solve阶段遇到frame338/2275向下首命中8.20/8.28m而停止。该两帧是edge0065/tunnel3同一物理点，距所有lateral anchors至少48m且收到的偏移为0，故排除了平滑核跨通道污染。

邻帧地面距离稳定在0.93--1.08m，该点突跳到8m以下的另一表面；直接使用会造成约7m的轨迹跳变。这是data/mesh floor-support问题，不是model、teacher或metric问题。run按合同FAILED、17项证据封存，不续跑。Gate 4保持`GATE_MIXED`。

## 2026-08-13：S10 footprint-scale floor-support audit 否定局部绕行

正式run `gate4_20260813_cano_s10_floor_support_hole_audit_v1_seed0` 已完成2个非独立重复观测×1,681网格。3,362条floor-query ray全部没有找到与`axis_z+FTA`相差不超过0.25m的mesh地面，故没有任何支撑候选可继续做水平/顶部审计。三面板完整热力图人工检查确认两个观测全域高度误差大、共同可行图全空。

正式run为COMPLETED/PASS_CANO_S10_FLOOR_SUPPORT_HOLE_AUDIT_V1，15/15 seal；PASS仅表示诊断完整。结论是小范围轨迹绕行不可行，当前S10 perception mesh不具备完整中心线地面支撑。下一步推荐重生成并重做完整几何资格链，不手工补面。Gate 4仍为`GATE_MIXED`。

## 2026-08-13：S10 replacement 正式命令在materialization前因runner环境合同失败

43/43双环境专项测试和preflight 0/0通过后，唯一授权命令创建run `gate4_20260813_cano_s10_perception_mesh_replacement_v1_seed0`。executor在导入历史materializer时因`subt_proc_gen`不可见立即退出，没有执行topology reconstruction、Poisson、sanitation、OBJ写入或floor query。

只读根因审计确认：历史M1R runner用`/tmp/mtare_cano_compat_e1_py312_np126_sp112/bin/python`，并通过`PYTHONPATH=external/procedural-subt-gen/src:src`导入固定checkout；新runner错用了zarr E1且遗漏upstream src。这是system/runner environment-contract问题，不是data、mesh、model或metric失败。run已FAILED并11项seal，不静默重跑。

## 2026-08-13：runner-only修正后发现历史冻结E1已漂移

用户回复“继续”后，新增`run_cano_s10_perception_mesh_replacement_v1r.py`及独立v1r Data Card/spec。新runner恢复历史compat E1、固定upstream PYTHONPATH，并在物化前保存E1、upstream和import identity。43/43单元测试通过，且`subt_proc_gen.tunnel`可从固定checkout导入。

完整外部导入测试随后在`subt_proc_gen.mesh_generation -> subt_proc_gen.perlin`处报`ModuleNotFoundError: perlin_numpy`。历史`pip_freeze_E1.txt`和安装日志均证明应存在来自commit `5e26837db14042e51166eb6cad4c0df2c1907016`的`perlin-numpy`，但当前compat E1中`pip show`和文件扫描均为空。这是/tmp冻结环境漂移，不是runner路径问题。正式v1r run未创建，生成、Poisson、OBJ和floor ray仍为0；重建新不可变E1属于新的系统变更，等待用户批准。

## 2026-08-13：不可变E1恢复成功，原生S10 replacement未通过floor support

用户批准后，新E1从历史freeze恢复98/98包；标准化freeze SHA与封存证据同为`e917255c...`，完整`subt_proc_gen.mesh_generation`和固定commit `perlin-numpy`可导入。43/43单元测试、1/1外部环境测试、preflight 0/0通过。

正式run `gate4_20260813_cano_s10_perception_mesh_replacement_v1r_seed0`只物化一次。新OBJ `f22ebb...`保持全部source identities，mesh audit与sanitation PASS，但2558 downward queries只有2234在`axis_z+FTA±0.25m`内。324失败组成53段，最长144帧；原frame338/2275仍失败。run FAILED且seal通过。这否定继续用相同native Poisson随机重生成修复动态floor资格；下一步应明确分离perception mesh和collision/floor geometry，需新批准。

## 2026-08-13：TNG 独立地面条带可行性预检定位到路口融合缺陷

经用户“继续分析”授权，仅实现并单测了确定性 TNG/spline 地面支撑原型，未创建正式 run。原型从完整 spline、冻结 tunnel radius 与 FTA 生成每隧道地面条带，并只为显式 incident node 增加最长 `0.5 m` 的连接片；不读取轨迹生成几何。新增及相关合同测试共 `25/25` 通过，三个 C08 世界重复构建字节完全一致。

4773 帧内存预检没有达到正式执行条件：S01/S06/S10 的地面失败分别为 `1/21/44`；结合原始轴线做 primitive-face 归因时，S01 唯一失败是首帧端点边界漏射，S06 的 `21/21`、S10 的 `45/45` 异常首命中全部来自另一条隧道地面。所有错误隧道对均共享一个显式拓扑节点，失败点距该节点约 `0.337--4.800 m`，非 incident/叠层隧道错误数为 0。

因此问题不是条带宽度、射线阈值、非拓扑隧道穿插或自身三角化缺面，而是共享路口附近多张独立倾斜地面互相覆盖，没有形成连续单值路口地面。继续加宽、按期望 tunnel 过滤首命中或仅在 evaluator 中分场景，会掩盖真实 collision 几何缺陷，不能作为 `DYNAMIC_NAVIGATION_GATE` 证据。当前条带 v1 方法停止，不创建 spec/Data Card/formal run；Gate 4 仍为 `GATE_MIXED`。推荐下一步先设计并单测“incident-node 局部单值融合面 + 非 incident 3D 隔离”，再提交精确几何定义和 4773 帧资格方案审批。

## 2026-08-13：结构尺度预检否定“只融合地面且保持原轨迹高度”合同

用户回复“继续”批准路口单值融合原型与内存预检，不包含正式 run。实现前先按结构尺度审计融合半径，未使用失败帧调参：分别取每节点 incident tunnel 最大地面半宽的 `1.0/1.5/2.0` 倍，并把相应 incident 地面顶点统一到节点地面高度。原始失败计数为 S01/S06/S10=`1/21/45`；三种尺度下分别变为 S01=`1/1/1`、S06=`196/996/1261`、S10=`110/1357/1958`。在最小 `1.0×` 尺度下，三个世界已经分别有 `7/3/24` 对节点融合域互相重叠。

这证明当前验收合同自相矛盾：统一路口地面只能为同一 XY 提供一个物理高度，而原 4773 帧仍按各 tunnel 的 `spline_z+FTA` 独立定高；斜坡路口中两者不能同时满足 `down=1.00±0.05 m`。因此本轮未继续实现、未创建正式结果，也未修改轨迹。问题归类为 geometry method 与 trajectory/metric interface，不是 model、teacher、split 或 leakage。

推荐下一步把“碰撞地面生成”和“floor-following 轨迹求解”作为一个有顺序但不泄漏的合同：先仅从 TNG/spline/radius/FTA 生成统一几何，再在冻结 XY/arcs/frame indices 上从该几何求 sensor z，并完整重审 down/up/horizontal/continuity。若要求保持原 sensor z 完全不变，则只能放弃统一真实地面，不能形成动态导航证据。该方法变更需要用户明确批准。

## 2026-08-13：geometry-first 证明固定 XY 无可行融合尺度，推荐 3D union

用户批准 geometry-first + trajectory-afterward 原型后，先完成不落盘的可行性分解。V1 独立条带的双向所属层 floor search 可为 S06/S10 全帧找到地面，但首命中切换造成 S06/S10 最大步长 `2.302/2.552 m`，超过旧最大步长 `2.0 m + 0.1 m` 合同；S01 首帧另需端帽。

简单节点平面融合在重新求 z 后仍造成最高 `16--33 m` 跨层，因此淘汰。进一步加入三项正确拓扑隔离——仅 incident tunnel、仅节点投影弧长附近 spline、仅三维节点邻域——消除了跨层串扰。此时解析局部 tunnel-floor 下包络给出严格尺度矛盾：要覆盖全部原重叠异常，S06/S10 至少需要结构半宽 `1.011/1.026×`；但固定 XY 轨迹在 S10 `0.5×` 已有6个超限步长、最大步长回归 `0.369 m`，S06 从 `0.75×` 起超限。覆盖下界与连续性上界没有交集。

因此方案 A 的“统一地面后保持 XY 完全不变、只求 z”被否定。问题属于 geometry/trajectory interface：三维隧道体相交后不能由二维单值下包络代表完整可导航路口。推荐 fallback B：对局部 tunnel swept volumes 做三维并集/等值面提取，随后保持 world、traversal、帧数与路由顺序，允许路口局部联合优化 XY/z，再做完整 collision 资格。该升级未经批准，本轮没有实现、formal run、轨迹写入、推理、图或训练；Gate 4 保持 `GATE_MIXED`。

## 2026-08-15 — Route-conditioned support corrective preflight ready

- corrective代码、teacher/graph/sensor坐标接口与正式runner已实现；33项相关测试PASS。完整geometry只读proof为4773/4773帧三档PASS，4307个窗口外pose逐元素不变；native双场景4773帧finite/LOS/determinism proof也PASS。
- 新geometry Data Card/spec preflight为0 error/0 warning。正式run尚未创建，当前Gate仍为`GATE_MIXED`；只读proof不能替代一次不可覆盖、带日志/hash/seal的正式qualification。
- 冻结Torch sidecar与三个M1D checkpoints身份已核对，但geometry正式输出尚不存在，causal replay不得创建或执行。唯一下一步是取得geometry one-shot execution批准。

## 2026-08-16 — V8 formal run 因监控误操作 FAILED

- V8正式run在有效运行约72.14分钟时被监控误判为超过12h并错误SIGTERM；`ps etime`的18h27m包含主机/会话暂停，而冻结runner的monotonic计时没有超限。
- run已不可变封存为FAILED，77/77 seal一致。现有正式部分证据包括S01三档801/801 PASS和60/150 patches；不能据此宣称4773帧正式qualification PASS。
- 分类为agent/operator process-control failure，不否定已通过的完整只读geometry/native proofs。Gate保持`GATE_MIXED`，causal replay继续阻塞。任何V8R完整恢复必须由用户重新明确批准。

## 2026-08-17 — V8R formal geometry qualification PASS

- V8R唯一正式run完成：4773/4773帧在三档分辨率全部安全，150/150 patches完整，25 windows、624 endpoints/traversals计数正确；4307个窗口外pose逐元素不变。
- RUN_STATE=`COMPLETED`，有效运行14044.69s，结果约1.510GB，178/178 SHA-256 seal复核通过。S01/S06/S10最小水平净空均大于4.95m，down误差处于1e-14m量级，连续性和resolution identity全部PASS。
- C08 geometry replay blocker已解除；Gate仍为`GATE_MIXED`，因为M1D/B0/oracle causal replay尚未正式执行。下一步仅准备并preflight绑定V8R hashes的replay spec，执行仍需单独批准。

## 2026-08-17 — C08 causal replay V2 preflight ready

- 新Data Card/spec已绑定V8R trajectory、native mesh、冻结M1D checkpoints、两个sidecar及执行工具hash；preflight为0 error/0 warning。
- 正式范围固定为4773个sensor frames、109,969,920条双场景rays、14319个inference frames和3645次graph replays。finite scan、双场景一致性、branch LOS及teacher完整性是硬门禁，native clearance仅为诊断。
- 尚未调用`create_run.py`，正式输出目录不存在。Gate 4继续`GATE_MIXED`；唯一下一步是取得一次正式回放执行批准。

## 2026-08-17 — C08 causal replay V2 formal FAIL（checkpoint mode case mismatch）

- 4773帧sensor阶段完整PASS：109,969,920条双场景rays全部finite且逐帧一致，teacher与branch LOS硬门禁全部通过。该部分证明V8R pose与native LiDAR接口成立。
- 推理在0帧停止。三个冻结checkpoint的hash与seed均正确，实际mode字段统一为大写`M1D`；executor错误要求小写`m1d`。因此这是system/interface validation bug，不是checkpoint漂移、模型或数据失败。
- run固定为`FAILED`，0 inference、0 graph replay、539/539 seal通过，不续跑或覆盖。Gate 4保持`GATE_MIXED`。推荐另行批准V2R：仅修正身份字段比较、加回归测试并完整重新preflight/one-shot执行；否则接受V2 FAIL并停止。

## 2026-08-17 — C08 causal replay V2R formal PASS

- V2R只修正checkpoint mode validator为训练器真实冻结值`M1D`并增加回归测试；三个checkpoint hash/seed/mode全部通过。新spec preflight 0 error/0 warning，正式run从sensor开始完整重算且不复用V2资产。
- 4773 sensor frames、109,969,920 dual-scene rays、14,319 inference frames和3645 graph replays全部完成；oracle integrity全grid PASS，C09/C10/M-TARE读取为0，训练/optimizer step为0。
- 唯一冻结参数为`sf2_tr8_lr6_hh20_te45_da20`（2 stable frames、8m event travel、6m loop radius、20deg heading merge、45deg turn、20m distance anchor）。mean composite：B0 0.743541，M1D seeds 0/1/2为0.847797/0.824948/0.806043，oracle 0.847775。
- RUN_STATE=`COMPLETED`，约172.4MB、1174.31s、583/583 seal。该结果只完成C08 development参数冻结，Gate 4仍为`GATE_MIXED`；不得自动读取C09/C10或修改planner/M-TARE。

## 2026-08-17 — C08 V2R research review

- V2R科研复核结论为`C08_DEVELOPMENT_PASS_WITH_ROBUSTNESS_CAVEATS`。M1D相对B0有明确聚合提升，但不是每个seed/world都提升；S10最弱且结构节点过生成。
- 冻结tuple与两个heading-angle变体完全并列最优，12组参数距最优0.005以内；冻结值是确定性开发选择，不是唯一可识别真值。
- Oracle exit F1仍显著更高，但加权composite不是oracle上界；禁止“模型超过oracle”的解释。Gate 4继续`GATE_MIXED`。
- 下一步需用户另行批准C09只读continuous trajectory/geometry/count contract。正式C09 validation仍未授权，历史15,725帧估计不得直接使用。

## 2026-08-17 — C09 read-only contract audit PASS

- 10个C09精确为1027 edges、2054 traversals、31654.752491m、15833 frames、71 junction windows、1331 window frames；历史15725帧估计作废。
- route、connector、arc incidence、window isolation和route-conditioned support语义预检全部PASS；0 overlap/conflict/nonincident frame。未生成正式资产，未运行LiDAR/inference/graph，C10/M-TARE读取为0。
- 三分辨率正式geometry qualification预计639 patches、47499逐分辨率帧审计；按C08线性约16.62h，原12h cap不可复用。Gate 4保持`GATE_MIXED`，等待用户批准或否决24h cap的一次性C09 geometry qualification。

## 2026-08-17 — C09 geometry qualification V1 配置路径错误 FAIL

- V1 Data Card/spec已获批且preflight 0/0，唯一formal run在冻结输入检查时因历史C08 seal文件路径写错立即停止；正确对象是`artifacts/evidence_sha256.txt`，冻结hash没有写错。
- executor未启动，C09 geometry frame评估为0，C09/C10/M-TARE读取、LiDAR、模型推理、graph replay和training均为0。该FAIL属于system/run-spec path error，不是否定C09 geometry、support方法或已训练M1D。
- V1已不可变标记`FAILED`并完成10-file seal。Gate 4仍为`GATE_MIXED`；必须由用户明确批准一个只修路径并加回归检查的新V1R，或停止，不得在V1目录重试。

## 2026-08-19 — C09 geometry V1R external interruption / V1R2 approved

- V1R因主机重启中断在7/10 worlds、8311/15833 frames、258/639 patches；S08仅完成45/48 patches，S09/S10未开始。无最终summary、metrics或seal，因此不构成PASS或geometry FAIL，旧目录禁止续跑与部分复用。
- V1R2 Data Card/spec已按用户“批准”冻结，完整重算原合同并加入`systemd-inhibit --what=sleep:shutdown --mode=block`。重建sidecar与原冻结hash一致，meshing contract及35项geometry/trajectory/runner回归全部PASS。
- Gate 4保持`GATE_MIXED`。当前状态为`PHASE_4_C09_GEOMETRY_V1R2_APPROVED_PREFLIGHT_PENDING`；只有V1R2全量完成并seal后才能判断C09 replay eligibility，且即使PASS也不自动运行模型或拓扑回放。

## 2026-08-19 — C09 geometry V1R2 RUNNING

- V1R2 preflight 0/0后已创建并启动唯一正式run；RUN_STATE=`RUNNING`，完整重算且未读取V1R部分资产，sleep/shutdown inhibitor有效。
- 当前任务仅为geometry qualification与证据封存。Gate 4仍为`GATE_MIXED`，C09模型推理和拓扑回放继续阻塞。

## 2026-08-20 — C09 V1R2 geometry complete but formal FAIL on frozen patch-count defect

- 几何主体完整PASS：10 worlds、15833 frames、47499 resolution audits、0 failed frames、14502 exterior poses exact；426个实际physical-window-layer-resolution patches全部生成，506/506 seal通过。
- 正式runner因冻结合同错误要求639 patches而将RUN_STATE写为`FAILED`。正确语义是71 windows含142个unique physical window-layers（65个双层、3个三层、3个单层）× 3 resolutions = 426；错误639来自把213 incident support arcs当作physical layers。
- Gate 4保持`GATE_MIXED`，C09 replay eligibility未解除。V1R2保持不可变FAIL，等待用户选择只读corrective evidence audit、完整重算或停止；不得自动执行模型推理或拓扑图回放。

## 2026-08-20 — C09 corrective evidence audit V1 formal FAIL before evaluation

- 426合同的真实分布已由测试确认，但正式audit在读取首个list-root manifest时因误用object-only `load_json()`停止；未进入任何证据结论。
- run固定为`FAILED`并11/11 seal。V1R2的506-entry seal前后完全一致，几何主体证据未损坏。Gate 4保持`GATE_MIXED`，C09 replay仍禁止。
- 推荐新V1R只修loader并补真实入口回归测试；需新批准、spec与不可覆盖run，不能续跑V1。

## 2026-08-20 — C09 geometry evidence qualification restored by V1R

- loader-only V1R通过真实入口测试、正式preflight和唯一不可覆盖执行。其审计确认正确合同为71个窗口、142个物理窗口层、三档共426个patch；15833帧与47499档位审计全部通过，14502个窗口外pose完全不变。
- V1R2的506项源封印在执行前后相同，新V1R的15项封印全部复核通过。V1R2和V1仍保留不可变FAIL，V1R只对其证据合同作独立纠正。
- Gate 4保持`GATE_MIXED`。C09已恢复模型推理/因果拓扑回放资格；下一步仅允许冻结C08选定tuple后执行C09 validation，不允许重新训练、参数搜索、C10、planner或M-TARE。

## 2026-08-20 — C09完整冻结回放已完成

- 正式C09 run完成15833帧LiDAR、47499帧冻结M1D推理和50次固定参数因果建图，机器合同PASS，1881项封印全部通过。所有图连通率与已验证边正确率均为1.0。
- 三seed M1D平均综合分相对B0提高0.011171，主要来自terminal reachability提高0.123228；但出口F1下降0.013899、冗余质量下降0.041508，且5/10世界的平均综合分低于B0。
- 项目判断为`C09_FROZEN_GRAPH_VALIDATION_MIXED`，Gate 4保持`GATE_MIXED`。完整链路已跑出，但不能宣称M1D在C09一致优于规则基线，也不能把C09称为严格端到端未见测试。下一步需用户决定是否单独授权C10。

## 2026-08-20 — 用户授权整篇论文路线，进入拓扑全局规划实现

- 旧 `PLAN.md` 的 Phase-4-only 实施上限已被用户本次明确目标覆盖；Phase 5--9 现在允许按 Gate 逐级实施，但 material run 的 Data Card/spec/preflight/一次性批准不变。
- Phase 5 的单一问题是：冻结因果拓扑能否稳定产生 M-TARE 可执行的全局目标，并在保留原 local planner/control 的公平条件下达到或超过原 M-TARE。
- 已实现 `src/mtare_topo/planning/topological_frontier.py` 与 `src/mtare_topo/integration/mtare_handoff.py`。规划只使用因果图节点、observed exit stubs、trace-verified edges 和显式失败状态，不读取 GT edge identity；handoff 固定 map-frame waypoint、finish、runtime 和 map-clearing 语义。
- 7项规划/交接测试、4项既有因果图回归和4项shadow/proposal合同测试全部通过（15/15）；15张真实C08最终图读取/决策为15/15。尚未执行正式逐帧shadow、Gazebo、原M-TARE基线或closed-loop。
- 下一步冻结C08 shadow：3 worlds、3 trajectories、4,773 frames；B0、M1D seeds0/1/2、oracle共23,865 planner cycles。C09不参与参数选择，C10继续封存。

## 2026-08-20 — C08 formal planner shadow PASS

- 唯一正式run `results/gate5_shadow/gate5_20260820_cano_c08_topological_planner_shadow_v1_seed0` 完成4773 unique frames与23865 planner cycles，机器结果为`PASS_CANO_C08_TOPOLOGICAL_PLANNER_SHADOW_V1`，30/30证据封印通过。
- 所有目标有限且在4m交接半径内，所有回退路径只用有物理trace的verified edge，15条流的完整双重重放完全确定；没有发布`/way_point`、没有M-TARE closed loop、没有读取C09/C10。
- 真实ROS历史bag又验证了16x720输入适配：完整四元数逆变换后31个跨全程抽样帧均为0错环；只用yaw会在倾斜区产生最多4710点错环，故接口已强制完整xyzw姿态。下一步是Oracle feasibility与原M-TARE公平baseline提案，不是直接宣称探索性能提升。

## 2026-08-20 — M1D checkpoint 已正式具备ROS运行资格

- 正式run `results/gate5_shadow/gate5_20260820_m1d_ros_deployment_export_v1_seed0`完成三个冻结M1D checkpoint的无损部署转换，状态为`PASS_M1D_ROS_DEPLOYMENT_EXPORT_V1`。
- 三个模型共126个tensor hash全部相等；Torch 2.9与ROS Python3.8/Torch2.0固定probe输出及离散决策全部通过。原checkpoint未修改，训练、optimizer、数据帧、world读取和机器人控制均为0。
- 22/22封印文件独立复核通过。此前`pathlib._local`阻塞已作为serialization/system问题解除，不改变模型科研结论。
- Gate 5仍未结束：尚未运行Gazebo闭环或原M-TARE公平基线。下一步只做固定seed启动、Oracle ROS节点和统一记录器实现/测试；闭环属于新的material run，必须重新展示精确world/start/seed/runtime/成本/证据并取得单独批准。

## 2026-08-20 — Gate 5闭环接口完成到seed边界

- Gazebo固定seed启动文件已通过ROS Noetic展开检查，保留原localPlanner、terrain、sensor和control，且replacement system不包含`tare_planner_node`。
- layered GT-map oracle现已具备与M1D节点相同的在线因果图和M-TARE waypoint/recovery接口；28项闭环相关回归全部PASS。
- 当前阻塞不是模型能否运行，而是原TARE baseline内部仍有未受Gazebo seed控制的`std::random_device`与`rand()`路径。正式配对重复实验前必须显式控制这些随机源；否则相同logical seed不是真正的相同随机条件。
- 推荐最小变更为在冻结TARE源码中加入`planner_seed`并统一绑定现有随机采样，不改变候选、代价、阈值或规划逻辑。该baseline可复现性变更尚待用户明确批准。
- 三方法统一ROS bag合同及post-run inventory审计已完成；原TARE的namespaced finish话题已由源码确认并同步到两个replacement节点。必需话题缺失、0消息或类型不一致均直接判INVALID，当前相关测试30/30 PASS。
- 单机器人开发矩阵已作为非执行proposal冻结到50个严格配对case：2 worlds × 5 environment seeds ×（原M-TARE + 3 M1D checkpoint seeds + Oracle），每case 600秒。总计30000秒模拟时间；32项相关测试PASS。该文件不是Data Card或执行授权，只有planner seed阻塞解除后才可进入正式preflight。

## 2026-08-20 — M-TARE显式seed实现已通过，完整回放资格V1为系统FAIL

- 原TARE内部所有已发现随机入口已由显式`~planner_seed`统一控制，源码扫描、编译、独立RNG同/异seed probe、ROS参数边界和33项相关回归均通过；该补丁未改变规划算法变量。
- 正式完整系统资格只包含tunnel和原M-TARE三次60秒回放，seeds为11/11/23；计划比较300个scan/pose指纹与60个waypoint，零模型推理/训练/C09/C10。Data Card/spec已获批准且preflight通过。
- 唯一V1 run在首条trial启动阶段失败：runner未source TARE devel环境，导致roslaunch无法定位实际存在的planner binary。失败发生在planner cycle之前，不能评价same-seed系统确定性，也不否定种子补丁的源码/二进制证据。
- V1已不可变封存为`FAIL_MTARE_PLANNER_SEED_QUALIFICATION_V1`。Gate 5继续阻塞；推荐V2仅修复环境source并原样重做三次资格，必须取得新批准。

## 2026-08-20 — V2证明双工作空间可启动，但错误复用kTestID而FAIL

- `--extend`已经同时保留AEE与TARE catkin工作空间，V2真实启动到patched `tare_planner_node`，因此V1的binary discovery阻塞已解除。
- V2 runner把随机seed 11传给通信模式字段`kTestID`，节点按源码合同拒绝长度2字符串。单机器人full-comms必须固定`kTestID='0'`；随机性只应由Gazebo seed与新`planner_seed`控制。
- V2未形成same-seed复现结论，保持不可变系统FAIL。下一步推荐V3只纠正参数映射及V3状态标签，需新批准后执行。

## 2026-08-20 — V3已准备到执行批准边界

- 参数映射器固定`kTestID='0'`且不改变11/11/23 Gazebo/planner seeds；错误或重复seed字段会拒绝执行。V3 runner拥有版本正确的状态与seal合同。
- 16项回归全部PASS；精确Data Card/spec proposal与模拟preflight通过。正式V3 run尚未创建，等待用户执行批准。

## 2026-08-20 — V3揭示kTestID完整编码，V4组件已就绪

- V3已封存为系统FAIL：`kTestID='0'`仍不满足原源码无条件读取第3–4字符的要求。完整单机器人full-comms编码经所有源码读取点确认是`0001`。
- V4只把该通信模式字段改为`0001`，不改变Gazebo/planner seeds或科研合同；19项测试PASS。正式V4执行仍需新批准。

## 2026-08-20 — 原M-TARE完整系统同seed并非逐帧确定

- V4成功完成三次合法60秒回放，所有记录话题健康；异seed差异成立，但两个seed11在100个审计frame row中0个完全相同，20个waypoint仅首个相同，故正式exact-replay资格FAIL。
- 初始同pose扫描已有约9mm平均点差；闭环后平均位姿分离0.151m、航点最大分离1.600m。这证明非确定性会影响全局路线，而不是无关的日志/时间戳噪声。
- 不能据此否定结构拓扑方法；它否定的是“单次严格配对即公平”的实验设计。推荐改成多重复随机统计，并把topological abstraction对扫描扰动的稳定性作为显式评价维度。

- 可执行推荐规模为90 cases：两个world、五个environment seeds；M-TARE与Oracle各三次execution repeat，M1D用三个冻结training seeds各一次。这样三类方法每world均有15次运行，可用world/env-seed block与execution/model variation做层级统计。zstd-10实测可把完整bag降到26.3%，总存储约103--120GB。

## 2026-08-20：Gate 5 `GATE_MIXED`收口，Gate 6随机闭环获批

Gate 5可继续的部分包括：C08 23,865次shadow cycle、确定性拓扑目标选择、M-TARE waypoint/recovery交接、分层完整地图Oracle、统一coverage evaluator、ROS Python3.8 checkpoint部署和原M-TARE显式planner seed。保留风险是Gazebo/LiDAR/ROS完整闭环同seed不满足逐帧bitwise identity，因此旧50-case单次精确配对路线被否定。

用户明确批准推荐A后，operational Gate切到6。正式设计为tunnel/garage两个开发世界，5个环境seed，每个world×seed block运行原M-TARE三次、M1D三个冻结checkpoint各一次、Oracle三次，共90 cases、54,000仿真秒。主方法是M1D因果拓扑全局规划，baseline是原M-TARE，fallback/上界是不可部署的完整开发地图Oracle。主要通过证据为90/90合同健康、lossless bag校验、0.5m coverage-time AUC区组差异与置信区间，以及轨迹/航点稳定性；科研效果可以是正、混合或负，但任何系统、输入、topic、archive或证据失败都使正式run FAIL且不重试。

Gate-6专用Data Card/matrix/spec已冻结，25/25相关测试通过；派生ROS Python3.8入口编译、90-case frozen-input审计和磁盘门禁均通过。正式preflight为0 error，仅有“该operation不强制Data Card但仍会snapshot”的预期warning。用户的一次性执行批准已绑定，下一步是创建并启动唯一run。

Gate-6 V1已执行并在首case之后正式FAIL。600秒仿真、3000同步帧、统一topic audit、指标、拓扑snapshot和1.62GB raw bag均生成；失败点是ROS镜像缺少`zstd` CLI，故formal completed cases=0，run封存为FAILED且24/24 seal通过。更关键的是首个M1D case全程0m，只有1 node/0 edge；冻结方向头在首帧及均匀101帧都没有任何概率达到0.5（最大约0.2614--0.2617），而冻结B0在101/101帧都找到单一前向出口。分类同时包含system/package缺陷和model-domain/interface缺陷。不能仅补zstd后重跑90 cases。

进一步只读检查原M-TARE实际移动bag后，三个M1D seeds在99帧上的空方向为99/97/99，证明问题不是静止起点特例；B0为99/99非空。完整地图teacher抽查20/20帧给出一致的1--3个客观出口。因此长期B0 fallback会把方法变成几何baseline，不满足论文主张；推荐临时重新打开Gate2，只冻结encoder/embedding并用AEE tunnel objective teacher适配heads、AEE garage验证，同时做Cano retention门禁。精确方案与pending Data Card已经形成，等待用户批准。

## 2026-08-20 — AEE传感器导出正式run已创建并进入执行

- 当前Gate/问题：Gate 2纠正性表示学习；先验证原M-TARE能否在AEE tunnel/garage生成10条满足合同的移动传感器轨迹。
- 固定数据：seeds 11/23/37/53/71，各world 5条；每条3000 raw、固定每5帧保留1帧，总计30000 raw/6000 effective，单轨迹总移动不少于50m。
- 实现与预检：collector/exporter、全局唯一frame ID、host zstd无损归档和治理Gate 2纠正规则已测试；治理25项+AEE 10项PASS，preflight 0 error/0 warning。
- 唯一run：`results/gate2_representation/gate2_20260820_aee_domain_sensor_export_v1_seed20260820`已由`create_run.py`创建。当前NEXT仅为执行和监控；无teacher、模型、训练、C09/C10读取，任一case失败不重试。

## 2026-08-20 — AEE sensor export V1在入口前因Python环境FAIL

- 冻结命令调用`/usr/bin/python3`，该解释器没有NumPy；runner在模块导入时退出1，尚未进入`main()`。正式结果为0/10轨迹、0 raw/effective帧、0仿真/teacher/inference/training/C09/C10。
- V1保持不可修改并封存为`FAIL_AEE_DOMAIN_SENSOR_EXPORT_V1_PRE_ENTRY_PYTHON_ENVIRONMENT`；10/10证据哈希从项目根目录复核通过，没有重试。
- 只读验证表明`/home/zeng-workstation/anaconda3/bin/python`含NumPy 2.1.3，runner的`--help`及10轨迹合同导入通过。推荐只更换外层Python绝对路径，其他数据/方法/门槛/成本不变，新建replacement spec与run；需用户明确批准一次replacement执行。

### V1R replacement已准备到批准边界

新增V1R薄封装，仅校验宿主Python 3.13.5、NumPy 2.1.3及Python/NumPy init/core二进制哈希，然后把新RUN_ID转发给未修改V1 runner。Luna独立只读审查确认采集、world/seed、帧数、50m、timeout、无重试和archive合同均未改变。41/41测试PASS；proposal preflight唯一错误是缺少replacement明确批准，未创建第二个run。

## 2026-08-20 — V1R首轨迹完成但宿主归档权限FAIL；V1R2已获批

- V1R的第一条tunnel/seed11轨迹已产生3000 raw与600 effective帧，移动1141.027337 m，topic/range/shard审计均PASS。系统随后因root-owned case目录拒绝宿主创建`raw.bag.zst`而停止；正式aggregate为0/10，V1R保持不可修改FAIL且18/18 seal一致。
- 缺陷分类为container-to-host filesystem ownership，不影响数据、teacher或模型结论。V1R2仅在成功collector之后执行固定UID/GID ownership handoff，并把归档改为拒绝已存在输出且禁止force覆盖。
- 37/37单元/合同测试与一次真实Docker-host权限演练PASS。用户已批准此精确replacement；下一步是正式preflight、单次create和执行，仍为10条轨迹、30000 raw/6000 effective、约3小时/20GB。

## 2026-08-20 — V1R2授权证据时间戳无效，已在首轨迹完成前停止

- V1R2冻结spec中的`approved_at=21:30`晚于约21:07的run启动，属于治理/证据元数据错误。不能在不可覆盖正式结果中静默修改。
- runner和Docker case已停止；0条轨迹完成、0帧计入正式数据，1.7 GiB partial内容不复用。`FAIL_AEE_DOMAIN_SENSOR_EXPORT_V1R2_INVALID_FUTURE_AUTHORIZATION_TIMESTAMP`已写入RUN_STATE/metrics，17/17 seal验证通过。
- 下一步仅允许准备V1R3 metadata-only replacement：新run ID与真实、非未来的授权记录；复用不变的V1R2 permission/archive代码和全部科研合同。未获新明确批准前不创建run。

## 2026-08-20 — AEE适配后的部署与readiness执行路径已补齐

- 新三seedouter runner把6000 AEE teacher、Cano retention、三个source checkpoint和Torch/CUDA环境全部绑定，只有八项seed级门禁全PASS才输出aggregate PASS。
- 适配checkpoint的`M1D_AEE_HEAD_ADAPTED_V1`模式现可无损导出到ROS Torch2.0，在线节点显式白名单接受；旧`M1D`默认行为保持。跨runtime工具继续验证全部tensor hash、输出误差与方向/count/role决策。
- 在线空方向fallback已实现为显式且可审计的current-scan B0：learned非空时0调用，learned空时替代方向/count/role但保留embedding；每帧和snapshot记录fallback，供≤5%门禁直接计算。
- 宿主bag finalizer及6-case readiness runner已完成，解决旧ROS镜像无zstd问题，并硬检移动、图、verified edge、non-hold目标和fallback率。67/67相关测试及ROS Python3.8编译PASS；这些是实现证据，不是正式实验结果。

## 2026-08-21 — Canonical-target full-encoder domain adaptation已到正式批准边界

- 这不是从头开始：继续使用已封存PASS的AEE V1R3 sensor 6000帧、DAE多层objective teacher 6000标签、Cano V2R与三个原M1D checkpoint；V1R3 head-only scientific FAIL继续作为直接baseline。
- AEE原始binary traversable mask仍作为客观几何证据保存；学习target改为与Cano完全一致的exit-component center与固定3° circular Gaussian。6000帧只读canonicalization proof已确认component count恒等。
- 新增确定性AEE-mask-matched Cano view。每epoch仍只有3000 Cano与3000 AEE独立样本；额外3000 Cano view明确标为augmentation。dense/matched Cano各0.5 loss weight，AEE为1.0，因此两域总权重仍为1:1；验证集不增强。
- 新trainer允许encoder、embedding与三个semantic heads全部适配，不再要求旧z_role字节不变。正式Gate只使用PLAN预声明的branch count 1--4 macro-F1；5--6完整保存为diagnostic但不决定PASS。
- 新数据、trainer、outer runner、全模型finite/change、确定性pairing、样本/视图计数、validation隔离和count Gate相关39/39 CPU测试PASS，三个新入口通过py_compile。
- Data Card/spec proposal已冻结并preflight。当前只有8项预期authorization/approval错误、0 warning；没有创建run、没有GPU训练、没有新C09/C10读取。
- 逐条读取sealed sensor shard后的口径已纠正为实测：每条effective trajectory持续`599.000--599.005s`，全体相邻effective frame位移中位数`2.022131m`（均值`1.897445m`），替代旧卡沿用的660s/0.2m估计。零数据pre-entry proof同时验证15项tool hash、持久venv、Torch/CUDA/GPU/Zarr环境、pip-freeze、pip-check、child argv和run目录不存在，0 dataset/checkpoint/optimizer/C09/C10/file write。
- NEXT：向用户展示精确数据、方法、成本与门槛并取得本次material run的明确批准；批准后生成final Data Card/spec、再次preflight 0/0，并只创建执行一个不可覆盖三seed run。

## 2026-08-21 — Canonical-target full-encoder V2正式科学FAIL，暴露单AEE训练世界泛化不足

- 用户明确批准后，final Data Card/spec绑定真实时间`2026-08-21T19:35:00+08:00`；41/41相关测试及preflight 0 error/0 warning后只创建并执行一个不可覆盖run。
- seed0完整执行10 epochs/470 optimizer steps，环境、source/checkpoint/tool、确定性mask pairing、每epoch3000 Cano + 3000 AEE独立样本和3000 augmentation views、finite tensor及全部声明参数组change均通过。seed1/2按科研门禁未启动。
- 正式结果：AEE garage direction F1=`0.017102` vs B0=`0.301158`，empty=`0.972`，count 1--4 macro-F1=`0.281081`，role macro-F1=`0.362580`；Cano C09 direction从`0.889860`降到`0.735424`。四项效果门禁与retention均FAIL。
- run=`FAIL_AEE_ENCODER_DOMAIN_ADAPTATION_V2`，duration=`168.711s`，约11MiB；26/26 seal复核，seal SHA-256=`4d0ff2349b8f27b19547a3e8ebf6abec14bcb5b6693bd8872120886fc215a348`；C10/later读取为0。
- 只读诊断：source模型在AEE tunnel/garage均为100% empty且z_role feature std约`0.001`；V2后tunnel direction F1=`0.317246`、empty=`0.602`、role F1=`0.694783`、z feature std=`0.047807`，说明encoder collapse解除并在训练world学习；但garage direction F1=`0.017102`、empty=`0.972`。失败主要是单AEE训练world到不同geometry world的泛化，而不是仍未训练或表示继续塌缩。
- 数据充分性问题：5条tunnel trajectories只有一个独立geometry world，不能用3000相邻/重复观测替代跨world结构多样性；当前AEE镜像只冻结tunnel与garage两个world，garage又是validation，不能静默并入train。
- NEXT需用户决定：推荐先做同pose AEE real scan ↔ DAE ideal raycast的只读sensor-operator parity audit，再据结果选择冻结structural encoder的paired front-end domain adapter；备选为把多个Cano开发world正式移植到AEE/Gazebo采集真实sensor-domain数据，成本更高但数据证据更强；或降级为B0 topology并削弱论文学习主张。

## 2026-08-21 — 64帧同位姿审计完成：传感器栅格失配与跨几何泛化问题同时存在

- V1在任何数据读取前因runner残留必读`data_card`字段而失败，射线、B0、M1D、训练及C09/C10均为0；原run封存为pre-execution system FAIL。V1R只修复该入口检查，8/8测试和preflight 0/0后完成。
- V1R固定审计tunnel/garage各32帧，覆盖两世界全部10条轨迹；生成737,280条同pose DAE射线，执行128次B0预测和三checkpoint共384次M1D推理。14项证据已封存，seal SHA-256=`6d4b944bc2208e4a28981b616b8bb3517ca3824ec866b438d7f600e43ac4793d`，训练/C09/C10为0。
- 两世界真实scan仅约48%像素有效，理想DAE为95.5%/99.8%；真实与理想valid agreement仅0.507/0.486，且相邻列有效状态几乎总是相反。这证明一个跨世界一致的稀疏角采样/栅格接口差异真实存在。
- 理想scan使tunnel三个seed的方向F1均从接近0提高到0.167/0.566/0.384，空输出率均下降；garage只有seed0/1小幅恢复，seed2仍F1=0、empty=1.0。故冻结的“两个world、全部seed均恢复”判据不通过，正式结论为`MORE_INDEPENDENT_AEE_GEOMETRY_REQUIRED`。
- 科学解释不是“只有数据问题”或“只有传感器问题”，而是二者并存：统一前端有证据基础，但即使给理想观察也不能消除garage结构泛化失败。推荐下一步把显式稀疏角接口与多个独立AEE/Gazebo开发几何放入同一纠正设计；未经用户路线决定不启动新训练。
## 2026-08-26 — 历史判断：曾定位为空间时序表示，后被Teacher审计推翻

正式292D rare-event corrective已科学FAIL并完整封存：macro-F1=`0.706287`，开放集precision/false/recall=`0.990160/0.009840/0.389469`，junction/terminal/turn/transition identity coverage=`0.931507/0.953125/0.115789/0`，seal SHA=`c5b58a612f22ebbe9ac86ca4021909133dcf4d5403a23baff318923ad5374c9c`。C09/C10/M-TARE读取与backbone更新均为0。

当时的只读审计只证明标签在50m LiDAR中可见，冻结事件分类逐帧召回也已有`90.21%/82.88%`；它排除了“完全不可观测”，但没有证明Teacher identity稳定。Directional head随后FAIL，并由持久性/双向性审计证明旧transition Teacher本身有缺陷，因此“不重做Teacher”的历史判断已被顶部最新结论覆盖。
## 2026-08-26 — Directional Structural Event Head科学FAIL；主阻塞转为Teacher/metric

唯一正式run完成三seed和36,000次head optimizer step，zero backbone/C09/C10/M-TARE，资源与seal全部合格，但科学FAIL：ensemble macro-F1=`0.695935`、节点precision/false/recall=`0.990073/0.009927/0.487987`、turn coverage=`0.189474`、geometry-transition coverage=`0.002688`。run seal SHA=`ea7507d53802126ea3da6141604a2f7812727718fbda4fea281db833151c7aab`。

只读Teacher审计确认transition不是稳定的结构事件集合：C07--C08的744 identities中90.59%位于edge端点10m内、55.24%区间不超过2m、19.22%仅单向出现，并占全部结构identity的66.85%。当前暂停受影响训练与图回放。推荐下一路线是A：先证明持久、双向一致、端点归并的因果geometry change-point Teacher，再重建受影响标签；需用户明确选择A或B。
## 2026-08-26 — transition失败分析已形成论文级可复现图包

`docs/figures/gse_graph/gse_transition_teacher_audit.*`已从sealed Teacher/dataset/verifier/292D/directional五个来源自动生成并目视通过。PNG/PDF/SVG、CSV/JSON、provenance和manifest齐全，manifest SHA=`800008a52a1a0fcda07063af34dda67aecf3b87d924d81b379af66735c39b055`。该图保留为失败分析/Teacher修正动机，不冒充最终方法结果。本条记录时仍等待A/B；用户随后已选A且完整proof PASS，以文件顶部最新状态为准。
# 2026-08-27 Factorized GSE-Graph方法重构

- 普通`geometry/action summary + causal change-point`已由唯一正式C01--C08 proof科学否定：precision=`0.552632`、episode recall=`0.031111`、相对强单项baseline变化=`-0.005185`；run无系统错误、零禁用读取，18项seal SHA=`e77543a2eda4d55403cc52f09aaa06189e7cf53d625adc13db36227fbe3a9692`。
- 失败不是“没有学到几何”：现有冻结证据仍为junction/terminal identity coverage=`140/146`、`125/128`，连续宽/高/坡度/曲率相对非学习估计改善=`24.071%/51.998%/49.675%/85.307%`。
- 当前方法把结构语义因子化：junction/terminal决定decision node；turn与连续几何形成verified edge profile。节点关联候选使用完整exit token、place descriptor、空间距离和已执行incident-edge geometry fingerprint，并在多义时拒绝合并。
- 权威设计：`docs/GSE_GRAPH_FACTORIZED_RESEARCH_PLAN_V2.md`。当前操作阶段仍为Gate 3；下一步仅做C01--C08关联inventory/capacity proof准备，C09/C10/M-TARE/图网格/闭环禁止。

## 2026-08-27 Factorized association Teacher缺陷

- Inventory正式FAIL但系统正常：inbound geometry profile identity coverage fit/selection=`0.981061/0.981752`，full tokens 3/3有效；真正失败项仅为per-family hard-negative population。
- 旧decision pair分布fit=`23,359/27`、selection=`8,083/71`（positive/negative），多数family零negative，不能用于开放集关联训练或安全结论。
- 当前阻塞分类：`DATA_TEACHER_SAMPLE_UNIT_DEFECT`。不训练capacity model。推荐修订为identity-balanced structural-alias pair Teacher，细节见`docs/GSE_FACTORIZED_ASSOCIATION_TEACHER_PROPOSAL_V1.md`。
- 正式run：`gate3_20260827_gse_factorized_association_inventory_v1_seed0`，17项seal SHA=`fc7781239b3e947ddc193f85ef9fec14c1ba1fae8fb96f2df02930cdcbe2cbeb`。

## 2026-08-27 Factorized association Teacher manifest PASS

- 旧pair cache的negative coverage缺陷已由新的identity-balanced structural-alias Teacher解除；旧FAIL不修改，继续作为为什么需要新Teacher的直接基线。
- 正式population：C01--C08 `188,126` observations，decision identities fit/selection=`792/274`，每identity一个positive和一个hard negative，总计`1,066` units / `2,132` pair records。
- positive来源为different-edge `563`、reverse-view `491`、distinct-observation `7`、singleton terminal circular-shift augmentation `5`。最后5个明确不是physical revisit；所有identity均保留。
- hard negative全部split-isolated、不同identity、同event、同incident degree，并由只用于Teacher选择的masked objective geometry profile确定；该profile及identity/world/parent标签禁止作为学生特征。
- run=`results/gate3_semantics/gate3_20260827_gse_factorized_association_teacher_manifest_v1_seed0`，status=`PASS_GSE_FACTORIZED_ASSOCIATION_TEACHER_MANIFEST_V1`，18项seal SHA=`047b5d16b0083fcab56a2a04d286655b2711e148b704d98f30d958e7b1d90f5b`。
- 当前Phase仍为Gate 3；Teacher PASS不等于association model或topology PASS。下一步是route-conditioned association三seedcapacity proof的Data Card/spec和预注册门槛，C09/C10/M-TARE继续0读取。

## 2026-08-27 Factorized route-conditioned association capacity PASS

- C01--C06 fit `792 identities/1,584 pairs`、C07--C08 selection `274/548`；三seed frozen perception各训练full/no-route小关联器，原backbone零更新。
- full route-conditioned三seed均满足安全门槛，safe recall=`0.762774/0.645985/0.620438`，precision=`0.990521/1/1`；physical-only复核同样PASS。
- 平均safe recall从descriptor-only `0.218978`和no-route `0.199513`提高到`0.676399`，证明执行边width/slope到候选exit width/vertical-profile关系有独立贡献。
- 正式run 61项seal SHA=`91ef98c0bc5b4cd82492dd42499a57a1bfe22d020664fbb0189aa2345cee11cc`，C09/C10/M-TARE读取为0。
- 当前Phase保持Gate 3但转入`C09_FACTORIZED_ASSOCIATION_QUALIFICATION_PREPARATION`。下一步只能冻结C09只读runtime+structural-alias资格；模型和阈值禁止再选择，C10/图/闭环继续隔离。

## 2026-08-28：空间多事件Teacher feasibility PASS

- 原V1在任何world/raycast前因JSON array/object读取接口错误system FAIL；12项seal SHA=`79ee5e76b258f9be96a59caaec367ab51f537f7d8ae217db77d88f64f14d6c1d`，永久保留且不用于科学结论。
- V1R只修列表解析，完整审计80个C01--C08世界、188,126条五帧因果观测；1,076/1,076个objective terminal/junction全部可见且反向heading覆盖100%。
- fit/selection多事件帧=`17,811/6,153`，terminal+junction同帧=`10,052/3,282`，最大集合基数5，四个稀有失败行全部覆盖；所有预注册门PASS。
- 正式V1R 19项seal SHA=`c557e1d184697b57e7e3a103e32581ce5df43a319da5a129698b419d3d4117c2`，论文图SHA=`5bf7fe520446d780aa98781cfffc66af2c310f42d10f031b7ddb57988aa418b7`。
- 当前仍为Gate 3。下一步是独立Data Card/spec下导出16槽空间事件set Teacher；不训练、不读C09/C10/M-TARE。

## 2026-08-28：空间多事件Teacher export PASS

- 唯一Gate-2 run输出80 shards、188,126唯一行、133,055 visible tokens和1,076个Teacher-only identities；terminal/junction=`30,789/102,266`。
- 81,069个零事件行保留，固定16槽零截断，实际最大集合5；全部80个world与feasibility统计逐项相同。
- run仅16 MiB；1,634项seal SHA=`29d821e2c9935a7e9f24dc9aa56ed14c01131a397c922c5f6f47330c8caef930`，论文图SHA=`4f7278394a89dc28827adcba55b88efe6acb2e04c5a7b6eca77ed43c82e9d343`。
- 已返回Gate 3。下一步仅实现set decoder及loss/readiness，不训练、不读取C09/C10/M-TARE。

## 2026-08-28：SpatialEventSetDecoder readiness V1R PASS

- V1全数据接口正常，但query permutation error=`2.9373e-4`违反`1e-6`门；V1 10项seal永久保留。
- V1R只增加content-canonical assignment，正式误差=`1.1921e-7`，旋转误差=`8.2970e-5m`，93,638参数decoder对真实0--5事件batch finite backward。
- V1R 10项seal SHA=`a9625fa3eac099fa2a63a259f55e98f5200aacc49df60bc921e9571f76a62c78`。下一步为三seeddecoder-only capacity训练Data Card，仍禁止C09/C10/M-TARE。
## 2026-08-28 — StructuredPolarMultiDepth capacity结论

- 当前Phase/Gate：`GSE_GRAPH_GATE3_STRUCTURED_POLAR_OBJECTNESS_FAILURE_ATTRIBUTION_PREPARATION` / `GATE_FAIL`。
- 正式训练：C01--C06 `142,184` rows、C07--C08 `45,942` rows，三seed总`27,288` steps；0 C09/C10/M-TARE。
- 科学结果：三seed F1=`0.166922/0.144133/0.145364`，best baseline=`0.390133`；precision=`0.137586/0.103357/0.129531`，不能用于结构节点生成。
- 保留正面证据仅限matched xyz MAE=`2.007/2.007/2.053m`；same-bin second-depth recall后来证明全部由slot0匹配，不能声称双深度表示已学会。
- 当前阻塞：dense presence置信度产生`44,078/68,447/36,887` false positives。下一唯一工作是既有输出的objectness/proposal failure attribution；禁止图回放和规划器补偿。
- 证据：`results/gate3_semantics/gate3_20260828_gse_structured_polar_multidepth_capacity_v1_seed0`，33项seal SHA=`3186e311508a82fa5e6a8342806729ee4c9335cb9407b322f6420c2cf73503ee`。

## 2026-08-28 — Structured-polar路线停止，转向可执行full-token因果事件

- V1归因发现proposal oracle高但confidence不可分；V1R补齐预测slot provenance后，正式否定objectness-only refit。
- 三seed top16 slot1=`0/16/0`，正式slot1=`0/0/0`，second-depth由slot1匹配=`0/0/0`，distinct-slot pair=`0/0/0`。
- 当前Phase=`GSE_GRAPH_GATE3_FULL_EXIT_ACTION_TOKEN_CAUSAL_EVENT_FEASIBILITY_PREPARATION`；阻塞=`STRUCTURED_POLAR_SLOT_COLLAPSE_AND_OBJECTNESS_FAILURE`。
- 当前唯一问题：完整可执行exit/action geometry token的5帧状态变化能否形成唯一、可观测、可学习且区别于固定summary/change-point的结构事件Teacher。
- 下一工作只读C01--C08；C09/C10/M-TARE、图和闭环仍禁止。
- V1/V1R seal SHA=`d0ad99d0dfeeb57d6ad7c4d947d286623d46a92b0e415bef2129f71abd6ac925` / `8c9cf3714d8ceca617c57905573c8b7e86f7127dd2bcfa90accb84ab2d9d5ae5`。

## 2026-08-28 — 当前状态：关系式可执行exit-token事件模型readiness准备

- 可执行action-set feasibility正式PASS：C07--C08 Teacher upper macro-F1=`0.995411`，episode support=`0.985915`。
- 三seed相邻帧descriptor transport precision=`0.981889/0.983521/0.984711`，recall=`0.988804/0.990447/0.991645`。
- 旧pooling模型precision=`0.965870`未达到节点触发安全门，不能进入graph；失败根因是token correspondence在时间聚合前被压缩。
- 当前Phase=`GSE_GRAPH_GATE3_RELATIONAL_EXIT_TOKEN_TRANSPORT_READINESS_PREPARATION`，阻塞=`RELATIONAL_EVENT_MODEL_NOT_IMPLEMENTED`。
- 下一唯一任务：实现零训练`RelationalExitTokenTransportEventModel`接口、因果/置换等变、transport矩阵、finite backward、provisional/refusal合同和单元/真实batch readiness；随后才可冻结三seed训练Data Card。
- 证据run=`results/gate3_semantics/gate3_20260828_gse_exit_action_transport_feasibility_v1_seed0`；17项seal SHA=`10fa5df309c81d11254f3302280200fe19b7b77967f70b49f1f66f62da5f2f23`。

## 2026-08-29 — 当前状态：关系式exit-token模型三seed训练准备

- 零训练readiness V1R正式PASS；V1 API失败保留且不计科学结果。
- 240,101参数模型通过全量因果引用、token/seed permutation、padding、repeat、transport归一化、typed provisional/refusal和真实batch finite backward。
- 当前Phase=`GSE_GRAPH_GATE3_RELATIONAL_EXIT_TOKEN_TRANSPORT_TRAINING_PREPARATION`，阻塞=`RELATIONAL_EVENT_MODEL_CAPACITY_UNPROVEN`。
- 下一唯一任务：冻结三seedData Card、trainer和selection evaluator；C01--C06提供梯度，C07--C08只选checkpoint/threshold，安全precision门不降低。
- readiness run=`results/gate3_semantics/gate3_20260829_gse_relational_exit_transport_readiness_v1r_seed0`，seal SHA=`853b040875f3184e4c3389402655198e4bd13c43533d0e23d73e69861f32d01a`。

## 2026-08-29 — 当前状态：冻结输出commit-policy归因准备

- 关系模型三seed正式训练已完整结束，系统无错且严格隔离测试集，但ensemble科学FAIL。
- 单seedF1有`0.828--0.892`学习信号；ensemble因seed校准和无状态commit降至F1=`0.665`、precision=`0.947`。
- 26个非正确trigger中18个是同一结构事件重复建点风险，另有5个走廊误报和3个类型错误。
- 当前Phase=`GSE_GRAPH_GATE3_RELATIONAL_EVENT_COMMIT_POLICY_FEASIBILITY_PREPARATION`；阻塞=`STATELESS_COMMIT_AND_SEED_DISAGREEMENT`。
- 下一唯一任务：零训练、零新模型推理的C07--C08 past-only状态提交可行性，检查去重、稳定触发和不确定性拒绝；继续禁止C09/C10/M-TARE/graph/planner。
- 正式训练run=`results/gate3_semantics/gate3_20260829_gse_relational_exit_transport_training_v1_seed0`，seal SHA=`de4cb8d37bc834236e6a1739b7e2d8d91f963a2299d7dcd51e42b7a7605aad6f`。

## 2026-08-29 — 当前状态：exact-one结构化事件loss readiness准备

- 冻结输出状态机审计正式FAIL：C07安全通过但F1=`0.753519<0.793847`；C08 F1=`0.796339`不能替代双侧合同。
- 去重机制有效，C07/C08 precision=`0.996241/0.997093`且duplicate=0；剩余核心是junction安全召回不足。
- 当前Phase=`GSE_GRAPH_GATE3_STRUCTURED_EXACT_ONE_EVENT_LOSS_READINESS_PREPARATION`；阻塞=`EPISODE_MAX_MIL_DOES_NOT_PENALIZE_MULTIPLE_COMMITS`。
- 下一唯一任务：实现并验证参数无关的exact-one episode likelihood，要求一个正确commit、其余帧拒绝；零optimizer、新模型推理和测试世界读取。
- commit audit run=`results/gate3_semantics/gate3_20260829_gse_relational_commit_policy_feasibility_v1_seed0`，seal SHA=`bc8771dd710f33053531f47202c4ee0a15482b58273469307a5584306fadaa47`。

## 2026-08-29 — 当前状态：V2 exact-one三seed训练准备

- exact-one loss readiness 12/12门PASS；单峰排序、重复峰梯度、permutation、极端数值和真实完整episode backward均成立。
- 当前Phase=`GSE_GRAPH_GATE3_STRUCTURED_EXACT_ONE_EVENT_TRAINING_PREPARATION`；阻塞=`STRUCTURED_EXACT_ONE_EVENT_CAPACITY_UNPROVEN`。
- 下一唯一任务：同一240,101参数模型三seed、6轮训练；C01--C06梯度，C07 checkpoint/policy选择，C08一次迁移，继续禁止C09/C10/M-TARE/graph。
- readiness run=`results/gate3_semantics/gate3_20260829_gse_structured_exact_one_event_loss_readiness_v1_seed0`，seal SHA=`ced78ead9cf7773f51ee698d4ae4ae6f6b19a631530054797e151b9c47b07426`。

## 2026-08-29 — 当前状态：exit-token validity/action-set track可行性准备

- V2 exact-one三seed训练正式FAIL；它消除duplicate但C07/C08 F1降至`0.573/0.583`，直接事件分类路线停止。
- 当前Phase=`GSE_GRAPH_GATE3_EXIT_TOKEN_VALIDITY_ACTION_SET_TRACK_FEASIBILITY_PREPARATION`；阻塞=`DIRECT_EVENT_HEAD_SAFETY_RECALL_TRADEOFF`。
- 下一唯一问题：学习式几何出口token的validity与descriptor tracking能否形成稳定、可执行、可解释的action set，并可靠决定结构节点。
- 下一任务只读C01--C08预测token与Teacher token；0 optimizer/new model/C09/C10/M-TARE/graph。
- V2 run=`results/gate3_semantics/gate3_20260829_gse_structured_exact_one_event_training_v1_seed0`，seal SHA=`51d3a8964ca7c556ccafaad93ee1392804ec429273f73db497cf6b791da3306b`。

## 2026-08-29 — 当前状态：Dense Circular Traversability Field Teacher可行性准备

- frozen free-query token validity V1R2系统正常但科学FAIL；六query在99.5%精度下只保留约`9.39%/0.001%/0.64%`真实出口。
- 五帧descriptor轨迹和三seed heading/width/profile共识均未改善，最佳高精度recall=`9.24%/0.042%/1.08%`，AP gain三seed全部为负。
- action-set Teacher上界`0.995411`和已知真出口descriptor transport仍超过`0.98`，表明应移除free-query objectness而不是放弃几何结构语义。
- 当前Phase=`GSE_GRAPH_GATE3_DENSE_CIRCULAR_TRAVERSABILITY_TEACHER_FEASIBILITY_PREPARATION`；阻塞=`FREE_QUERY_GHOST_SLOT_VALIDITY_FAILURE`。
- 下一唯一问题：396,913个visible directed exits能否无损编码为固定180-bin圆周可通行几何场，并由连续分量在不使用identity/future的情况下恢复出口集合。
- 下一任务只读C01--C08 Teacher与range几何；0 optimizer/new inference/C09/C10/M-TARE/graph/planner。通过才允许导出Teacher。
- V1R2 run=`results/gate3_semantics/gate3_20260829_gse_token_validity_action_track_feasibility_v1r2_seed0`；seal SHA=`41c23bdf50f90cfec427d93add4ac7e6eecc001c41fb2f73655643c105986729`。

## 2026-08-29 — 当前状态：Circular Executable-Geometry Peak Field Teacher导出准备

- 宽扇区connected-components在12.027%观测丢失出口实例，已由中心peak语义取代；该修改有22,626行失败证据，不是无依据换名。
- 180-bin peak field正式V1R PASS：396,913 exits的same/adjacent-bin collision均0，几何、旋转、slot permutation和因果reference全部通过。
- raw-range envelope AP约0.052且安全recall为0，证明后续仍需学习，不把Teacher无损性冒充模型性能。
- 当前Phase=`GSE_GRAPH_GATE2_CIRCULAR_EXIT_GEOMETRY_FIELD_TEACHER_EXPORT_PREPARATION`；阻塞=`CIRCULAR_PEAK_FIELD_TEACHER_NOT_EXPORTED`。
- 下一唯一任务：一次性导出80个world shards，只保存180-bin peak targets和因果provenance，不复制LiDAR，不读取C09/C10/M-TARE；导出PASS后返回Gate 3 readiness。
- feasibility run=`results/gate3_semantics/gate3_20260829_gse_circular_exit_geometry_field_feasibility_v1r_seed0`；seal SHA=`ddad17c78f26ec05754c35ed88dc640473f30e9ea7c0b021995b2ed02fab663e`。

## 2026-08-29 — 当前状态：Circular Peak Field模型readiness准备

- Gate-2 Teacher export一次正式PASS：80 shards、188,126 rows、396,913 peaks；全部回读和seal通过，Teacher仅22.243 MiB。
- 当前Phase=`GSE_GRAPH_GATE3_CIRCULAR_EXIT_GEOMETRY_FIELD_MODEL_READINESS_PREPARATION`；阻塞=`CIRCULAR_PEAK_FIELD_LEARNABILITY_CONTRACT_UNPROVEN`。
- 下一唯一问题：五帧因果LiDAR模型能否以圆周等变、无identity/future的接口学习180-bin peak presence和几何属性，并在真实batch上得到finite masked loss/backward。
- 下一任务为零训练readiness；不创建checkpoint，不选阈值，不读取C09/C10/M-TARE，不运行graph/planner。
- export run=`results/gate2_representation/gate2_20260829_gse_circular_exit_geometry_field_teacher_export_v1_seed0`；seal SHA=`8c37f11a9e08b3cf0c48d005ae32aa2d79afcbba7c2330333ee9996605657f76`。

## 2026-08-29 — 当前状态：Circular Peak Geometry三seed训练准备

- readiness正式PASS：769,268参数、0 free query、全量causal join正确、真实masked backward及circular equivariance全部通过。
- 当前Phase=`GSE_GRAPH_GATE3_CIRCULAR_PEAK_GEOMETRY_THREE_SEED_TRAINING_PREPARATION`；阻塞=`CIRCULAR_PEAK_GEOMETRY_LEARNING_CAPACITY_UNPROVEN`。
- 下一唯一问题：从C01--C06 LiDAR学习的固定圆周峰值和完整几何语义，能否在C07选定后无适配迁移到C08，并以高精度恢复可执行出口集合及连续结构量。
- 下一任务：冻结三seed training Data Card/spec；C07只选checkpoint/threshold，C08只评估一次；C09/C10/M-TARE/graph/planner继续禁止。
- readiness run=`results/gate3_semantics/gate3_20260829_gse_circular_peak_geometry_model_readiness_v1_seed0`；seal SHA=`d06089680b97544a251e30e9c0214fa6c14df31fb3d30ef0c3d97852adc74c69`。

## 2026-08-29 — 当前状态：Soft Angular Peak + Hard-Negative readiness准备

- V1三seed系统完整但科学FAIL：连续axis/width/height/curvature有效，单格peak objectness和slope失败，禁止进入association/graph。
- 当前Phase=`GSE_GRAPH_GATE3_SOFT_ANGULAR_PEAK_HARD_NEGATIVE_READINESS_PREPARATION`；阻塞=`HARD_SINGLE_BIN_GHOST_PEAK_FAILURE`。
- 下一唯一问题：软圆周方位目标和显式困难负峰排序，能否在不改backbone/数据/split的情况下对近邻正确峰给连续梯度并直接压低已观察到的高置信假峰。
- 下一任务零训练、只读V1 predictions和真实batch；0 optimizer/checkpoint/C09/C10/M-TARE/graph/planner。PASS后才允许V2训练Data Card。
- V1 run=`results/gate3_semantics/gate3_20260829_gse_circular_peak_geometry_three_seed_training_v1_seed0`；seal SHA=`5477c4b7ddc32313d186e1a8a5b7dbb0dc8867cf9b52bb1b2fffb737b87ef018`。

## 2026-08-29 — 当前状态：Set Process冻结输出失败归因准备

- set-process三seed正式训练完整结束，系统无错；结构数量准确率C07/C08=`96.71%/95.99%`，raw结构macro-F1=`94.65%/93.87%`。
- 安全连续出口集合不可部署：99.5% precision下C07/C08 exit recall仅`0.044%/0.031%`，接受观测仅`10/8`；不得进入图。
- 全局axis/width/height/curvature仍具泛化，slope约`3.4deg`失败；极少数接受出口几何很准但不具总体代表性。
- 当前Phase=`GSE_GRAPH_GATE3_CIRCULAR_EXIT_SET_PROCESS_FAILURE_ATTRIBUTION_PREPARATION`；阻塞=`SAFE_EXIT_SET_RECALL_COLLAPSE_CAUSE_UNRESOLVED`。
- 下一唯一任务：零训练、零新推理地审计冻结C07--C08三seed预测，区分连续定位容量与confidence ranking/calibration失败；禁止C09/C10/M-TARE/graph/planner。
- 正式run=`results/gate3_semantics/gate3_20260829_gse_circular_exit_set_process_three_seed_training_v1_seed0`；seal SHA=`18a5dcb717ec7133aa0b27056fc18e844510ddbf738048ef8fd05ae481413f92`。

## 2026-08-29 — 当前状态：Circular Slot Transport readiness准备

- 冻结归因正式PASS：当前模型并非只缺confidence；2deg完整集合仅约23--25%，三/四出口发生严重模式漏失，selected-bin residual与训练位置错位。
- 现有count head和五帧geometry backbone继续保留；单一intensity与selected-bin residual部署接口停止。
- 当前Phase=`GSE_GRAPH_GATE3_CARDINALITY_CONDITIONED_CIRCULAR_SLOT_TRANSPORT_READINESS_PREPARATION`；阻塞=`ONE_TO_ONE_CONTINUOUS_SLOT_TRANSPORT_CONTRACT_UNPROVEN`。
- 下一唯一问题：K个圆周方位slot的一一匹配能否在无free-query existence的情况下同时保证多出口覆盖、连续定位、圆周等变和真实finite backward。
- 下一任务零训练readiness；C09/C10/M-TARE/graph/planner继续禁止。
- attribution run=`results/gate3_semantics/gate3_20260829_gse_circular_exit_set_process_failure_attribution_v1_seed0`；seal SHA=`7ff969054736b21c31d61177a5c69b87c1a02b537a1144b0ab97c3a269662b1f`。

## 2026-08-29 — 当前状态：Circular Slot Transport三seed训练准备

- readiness V1R正式PASS；V1仅为布尔审计system failure，无科学结论。
- 787,328参数decoder通过全量Teacher/causal audit、bijective duplicate gradient、confidence concentration、真实1--4 batch、rotation/slot/batch permutation和repeat合同。
- 当前Phase=`GSE_GRAPH_GATE3_CARDINALITY_CONDITIONED_CIRCULAR_SLOT_TRANSPORT_THREE_SEED_TRAINING_PREPARATION`；阻塞=`CIRCULAR_SLOT_TRANSPORT_LEARNING_CAPACITY_UNPROVEN`。
- 下一唯一问题：该一一连续slot decoder能否在C07/C08同时恢复安全出口集合，尤其解决三/四出口完整集合失败，并保留完整几何。
- 下一任务冻结三seedData Card/trainer/evaluator；C01--C06梯度，C07选择，C08一次迁移，继续禁止C09/C10/M-TARE/graph/planner。
- readiness run=`results/gate3_semantics/gate3_20260829_gse_cardinality_conditioned_circular_slot_transport_readiness_v1r_seed0`；seal SHA=`32ab6cb9ea65177c38ef5ef0b2eb88f61ef123943ffe740694a7d6de4cc00846`。

## 2026-08-29 — 当前状态：Circular Slot Transport冻结输出失败归因准备

- 三seed正式训练系统完整但科学FAIL：34,110 steps，best epochs=`8/8/9`，三个C07 best loss=`2.3343/2.3335/2.3286`，不是未收敛或单seed偶然失败。
- 相对set-process，整体exact-set在2deg提高约`9.09/5.20`个百分点，10deg提高约`18.27/16.84`个百分点；三出口10deg提高到`34.46%/28.53%`，slot一一覆盖具有明确容量。
- 但三出口2deg仅`1.30%/0.71%`、四出口严格为0；99.5% precision只能接受`5/4`帧，安全recall约`0.022%/0.016%`，不得进入图。
- 当前Phase=`GSE_GRAPH_GATE3_CARDINALITY_CONDITIONED_CIRCULAR_SLOT_TRANSPORT_FAILURE_ATTRIBUTION_PREPARATION`；阻塞=`STRICT_SLOT_LOCALIZATION_AND_CONFIDENCE_FAILURE_CAUSE_UNRESOLVED`。
- 下一唯一任务：零训练、零新推理地审计冻结C07--C08三seedslot mass，区分mean decoder、多峰分布、seed alignment和confidence不可分；禁止C09/C10/M-TARE/graph/planner。
- 正式run seal SHA=`472630ffd88a1331ead2abfdaf8f3469c6e1501d97d041b2d109685253f06455`；图SHA=`c36d02071e5604b3a991c5d40cd29a306bf11195590a8e81095fd9b1a4b9871a`。

## 2026-08-29 — 当前状态：Cyclic-Ordered Unimodal Slot Transport readiness准备

- 冻结归因正式PASS并排除mean/argmax解码、seed alignment和count coupling为主因；后处理confidence最高安全recall仍低于0.5%。
- 根因锁定为K>=3时任意permutation监督造成slot分布换位/扩散：三/四出口目标两bin质量及严格角误差明显恶化。
- 当前Phase=`GSE_GRAPH_GATE3_CYCLIC_ORDERED_UNIMODAL_SLOT_TRANSPORT_READINESS_PREPARATION`；阻塞=`CYCLIC_ORDERED_UNIMODAL_SLOT_CONTRACT_UNPROVEN`。
- 下一唯一问题：圆周顺序保持的cyclic transport与单峰proper circular likelihood能否消除换位歧义，同时保持旋转等变、distinctness、可信concentration和真实finite backward。
- 下一任务零训练readiness；禁止checkpoint/optimizer/C09/C10/M-TARE/graph/planner。PASS后才允许一次三seedtraining Data Card。
- attribution seal SHA=`06bee093f8a18e3eecd4bbbe407db97416ce986ac6169c9bf45235683bb3242c`；图SHA=`576b9b351956f518e434910a562e83b61d49f0de4d79ba1e2422555e1297bed2`。

## 2026-08-29 — COUST表示合同PASS，最终方法等待三seed效果验证

- V1公开保留metric FAIL；V1R只修正权威metric的量纲/对象，不改模型、数据、阈值或人口。
- V1R 11项科学检查全部PASS，完整覆盖80 worlds和真实1--4出口batch；0训练和0严格测试读取。
- 这只证明方法可训练、可复现且满足圆周因果合同，不证明论文效果。当前Phase仍为3，graph/planner继续禁止。
- 唯一下一步为COUST三seed训练。成败由K>=3严格集合定位、安全拒绝和连续几何相对旧slot transport的独立提升决定。
- V1/V1R seal SHA分别为`fa455c5ed5ddd0dc45454e0ba12e51073a7f1c8ccb501d5d5b93dbe4bcda512b`与`788f89bcee0ecfddf03e4cadbdc65c86ae2e868d73d72ca7bce752f7a494719f`。

## 2026-08-29 — COUST只改善简单结构，复杂结构与置信度失败

- 三seed训练完整、无系统错误；总体strict exact-set提升，但K=3低于旧方法、K=4近零，不能确立为论文方法。
- C07 concentration阈值无法迁移到C08，安全precision由1降至0.818；禁止进入拓扑图。
- COUST保留为失败消融：说明圆周顺序与单峰先验本身不足，且可能强化常见双出口偏置。
- 当前Phase仍为3。唯一下一步为冻结C07/C08输出归因K>=3 collapse/order/cyclic-start/seed alignment/kappa；0训练、0新推理、0测试/图。
- 正式seal SHA=`47e4a84354ad58498e24de839fab7a3b33fb0dbd0846213a12dcfd5114bffd80`；图SHA=`d5f533ebf25b1c60431c0d30600cf2c7c2c1cd64fc6852bc1c933389b4e26129`。

## 2026-08-29 — 当前状态：Joint Cyclic Gap Simplex readiness准备

- COUST冻结归因正式PASS；排除cross-seed cyclic alignment、单seed选择和简单slot-spacing collapse为主要修复方向。
- seed2粗间距已经接近Teacher，但K=3 strict exact-set仍低于0.6%；K=3 concentration/kappa AUC在两开发split为近随机或反向，失败位于精确定位/置信度目标。
- 当前Phase=`GSE_GRAPH_GATE3_JOINT_CYCLIC_GAP_SIMPLEX_READINESS_PREPARATION`；阻塞=`JOINT_COMPLEX_EXIT_SET_REPRESENTATION_CONTRACT_UNPROVEN`。
- 下一唯一问题：用一个旋转等变phase和K个正且总和360度的gap simplex联合生成出口集合，能否在不使用identity/future的情况下消除独立slot的duplicate/missing/crossing自由度并提供可学习的几何不确定性。
- 下一任务仅实现和正式执行零训练readiness；C09/C10/M-TARE/graph/planner继续禁止。
- attribution run=`results/gate3_semantics/gate3_20260829_gse_coust_complex_cardinality_failure_attribution_v1_seed0`；16项seal SHA=`bc6d05a03a2cd48d88e3f3a498a6286d6b4d670b1c79be5f8ae58102c3ff866c`。

## 2026-08-29 — 当前状态：Joint Cyclic Gap Simplex V2 readiness准备

- V1不是数据/系统失败；joint closure、gradient和真实backward成立，但phase可观测性与拒绝接口不完整，禁止训练。
- 当前Phase=`GSE_GRAPH_GATE3_JOINT_CYCLIC_GAP_SIMPLEX_READINESS_V2_PREPARATION`；阻塞=`PROPER_PHASE_OBSERVABILITY_AND_REFUSAL_CONTRACT_UNPROVEN`。
- V2唯一改变为proper phase likelihood、phase concentration绑定拒绝，以及distribution/resultant-space rotation authority；人口、Teacher、gap表示、789,650参数和隔离不变。
- 下一任务正式执行V2零训练readiness；PASS只允许三seed训练Data Card。
- V1 seal SHA=`e2e636523e49ef0db711c8232a5f050a9740def9e5ab924bf8015d510497a942`。

## 2026-08-29 — 当前状态：Joint Cyclic Gap Simplex三seed训练准备

- V2 readiness 11/11 PASS；proper phase、低可观测拒绝、closed gap、真实finite backward和权威rotation合同成立。
- 当前Phase=`GSE_GRAPH_GATE3_JOINT_CYCLIC_GAP_SIMPLEX_THREE_SEED_TRAINING_PREPARATION`；阻塞=`JOINT_CYCLIC_GAP_SIMPLEX_LEARNING_CAPACITY_UNPROVEN`。
- 下一唯一问题：联合闭合出口集合能否在C07/C08真正恢复K>=3 strict 2deg集合并得到可迁移安全拒绝，同时保留局部/全局几何。
- 下一任务冻结三seedData Card/trainer/evaluator；C09/C10/M-TARE/graph/planner继续禁止。
- readiness seal SHA=`42c335f0ebdb7d0c97fef67746513d1d96bc28ba2861c5117ce482080f002b14`。

## 2026-08-29 — 当前状态：Joint Cyclic Gap Simplex数值稳定纠正准备

- singleton closing gap合同已由0度纠正为360度，并由正式V3 readiness重新资格PASS；方法人口、Teacher和测试隔离未变。
- 首次三seedV1不产生科学结论：seed0完整成功，seed1在epoch4发生CUDA gather越界，seed2/selection未运行；失败前checkpoint全部finite。
- 当前Phase=`GSE_GRAPH_GATE3_JOINT_CYCLIC_GAP_SIMPLEX_NUMERICAL_STABILITY_CORRECTIVE_PREPARATION`；阻塞=`PHASE_RESULTANT_SINGULAR_GRADIENT_OR_NONFINITE_BEARING_SYSTEM_FAILURE`。
- 下一唯一问题：低concentration/近零phase resultant能否在保持phase probability与rotation合同的同时提供有限、确定性的角度反向传播，并在optimizer前拒绝任何非finite梯度。
- 下一任务为stable atan2与finite fail-closed readiness；PASS后V1R从seed0/1/2全部重训。C09/C10/M-TARE/graph/planner继续禁止。
- V3 readiness seal=`b9e896962e0288b817c3e8838e7c7fe0a215a212cbe2b74ad9b8db70203cf605`；V1 training FAIL seal=`d7fe9281a168a1d6ca30b7a48268e598868b4797041410814b58e5a7eb1fe499`。

## 2026-08-29 — 当前状态：JCGS seed1同步CUDA索引诊断准备

- stable-phase V4 readiness正式PASS，但V1R在seed1 epoch4精确重复CUDA gather越界；有限梯度防护未触发，排除“坏梯度先更新、下一批才崩溃”的主要假设。
- 当前Phase=`GSE_GRAPH_GATE3_JOINT_CYCLIC_GAP_SIMPLEX_SYNCHRONOUS_INDEX_FAILURE_ATTRIBUTION_PREPARATION`；阻塞=`ASYNC_CUDA_GATHER_INDEX_SOURCE_UNRESOLVED`。
- 下一唯一问题：哪个确切gather、哪组有限索引在seed1 epoch4越界，是Teacher phase index、periodic geometry sample还是其他分支。
- 下一任务固定seed1、最多5 epochs、同步CUDA诊断；只在C01--C06训练顺序和必要C07验证上重放，禁止C08/C09/C10/M-TARE/graph/planner。
- V4 seal=`4e649135b383d3e3fc47999daf034d75f7f4e379ac24fe2c2bc417eab2632559`；V1R FAIL seal=`a5f169e51b2ecd08341dfcbc63e45af1da2ff6692c2a3e2a0d0373ad6f9ad91c`。

## 2026-08-29 — 当前状态：JCGS周期索引已纠正，V1R2三seed训练准备

- 同步诊断已把seed1故障定位到phase likelihood gather：真实负零角度被浮点remainder表示为圆周端点bin 180；不是Teacher越界、非有限模型输出或坏梯度。
- canonical periodic coordinate把180等价映射为0，并由真实故障样本、CPU坐标审计和128行CUDA完整反向共同验证。
- V5的确定性cuBLAS环境遗漏作为不可覆盖system FAIL保留；V5R仅恢复环境，正式PASS且0 optimizer/checkpoint/test/graph。
- 当前Phase=`GSE_GRAPH_GATE3_JOINT_CYCLIC_GAP_SIMPLEX_V1R2_THREE_SEED_TRAINING_PREPARATION`；阻塞=`JOINT_CYCLIC_GAP_SIMPLEX_LEARNING_CAPACITY_UNPROVEN`。
- 下一唯一问题：正确实现的联合闭合出口集合能否在三个seed和未见开发世界中恢复K=3/4严格几何与可迁移拒绝。未证明前仍不得进入graph/planner。
- attribution/V5/V5R seal SHA=`12ad57ef81c8d796b7bcbffa0130247f565b3abefd5b1802a515b6c1a26e788c` / `8e7527c7258f810471e2effa8e8efc18523a3fe092413e5d77f8b09c938c798a` / `14a249d19860be0c6c986a1ba66e606b8e633da2d7c3da187e9b085b031ef152`。

## 2026-08-29 — 当前状态：JCGS科学FAIL，冻结输出归因准备

- V1R2系统完整：三个seed、34,110 steps、三个checkpoint与C07/C08预测全部封存；旧CUDA越界未复现。
- 方法失败：overall exact2仅`0.2008/0.1572`，K3/K4从2°到10°均0，安全拒绝只接受1帧且不迁移，slope仍失败。
- 当前Phase=`GSE_GRAPH_GATE3_JOINT_CYCLIC_GAP_SIMPLEX_FROZEN_FAILURE_ATTRIBUTION_PREPARATION`；阻塞=`JOINT_EXIT_SET_REPRESENTATION_COLLAPSE_UNATTRIBUTED`。
- 下一唯一问题：失败主要来自cardinality长尾塌缩、phase定位、gap形状、跨seed循环对齐还是confidence目标；只有归因后才能设计下一候选。
- graph/planner继续禁止。V1R2 seal SHA=`2290a4262da57a35d9755368913f8f0c8fa6a5614934a1c282648aa1d560533b`。

## 2026-08-29 — 当前状态：轴锚定几何事件—关系方法可行性准备

- JCGS冻结归因正式PASS并精确复现C07/C08结果；oracle count和Teacher-aligned seed ensemble均不能恢复复杂结构。
- K>=3的phase与gap shape在两开发split同时失败；完整出口集合回归不再作为论文主方法，JCGS/COUST/Slot Transport/Set Process转为表示消融。
- 当前Phase=`GSE_GRAPH_GATE3_AXIS_ANCHORED_EVENT_RELATION_TEACHER_FEASIBILITY_PREPARATION`；阻塞=`AXIS_ANCHORED_EVENT_RELATION_TEACHER_CONTRACT_UNPROVEN`。
- 当前候选方法：以机器人运动方向和学习局部轴线固定参考系，学习结构事件及相对continuation/side-branch关系、分支几何、descriptor和拒绝；结构节点由稳定事件提交，edge只由真实穿越创建。
- 下一唯一任务：C01--C08只读Teacher feasibility；核验轴锚因果可得、reverse traversal闭合、关系唯一、K1--K4/turn/geometry-transition人口、stacked/parallel无identity泄漏，以及旧关系事件证据的可复用边界。
- 禁止训练、新推理、C09/C10/M-TARE/graph/planner。归因run 18项seal SHA=`c627c1265f26b2dbcc49de542b9befae1a838f14d258bbe6772e17a2b424d1c4`；图SHA=`b61df1f3d0e6660b8dd29b3d6c3bc90c9dbde71b8cb010b27e890326a216899b`。

## 2026-08-29：方法方向已经确立，尚未完成模型确立

Axis-Anchored Geometry-Semantic Event Relation所需的监督合同已经正式通过：现有LiDAR数据中确实存在可因果获得的前进轴、五类结构事件、跨方向重访的唯一出口关系、出口出现/消失过程以及宽度、高度、坡度和曲率。它不是继续做出口方向分类，而是学习“当前是什么结构、各分支相对入射方向是什么关系、结构如何随机器人经过而显现”，这些输出将直接决定节点和出口关联；边仍只由真实穿越建立。

本次审计覆盖全部188,126个开发样本与4,493个结构事件，12项条件全PASS，且没有读测试世界或训练模型。因此“数据能否支持该创新方法”已经解除；旧完整集合回归路线正式保留为失败消融。

但还不能说最终模型已经成功：typed网络、损失、过去时记忆和几何等变性尚需零训练实现验证，随后还要三种子训练并以结构事件F1、连续几何误差、关联precision/false merge和离线图F1判断是否真正优于基线。当前唯一工作是method readiness，不进入拓扑图。

## 2026-08-29：主方法网络接口已完成，下一步是真正训练

现在已经不再是“连方法都没定”。819,196参数的主模型已经实现并通过正式训练前验证：它从五帧LiDAR同时学习五类结构事件、分支在时间上的持续/出现/消失、分支几何、整体隧道几何和地点/分支描述符。真值identity只出现在训练loss中，部署forward无法接收它；过去关系记忆也不读取未来。

真实数据有限反向、旋转等变、batch排列、反向行进轴闭合和确定性均通过。GPU默认低精度路径曾使旋转合同失败，现以确定性FP32环境解决，没有降低指标。

尚未完成的是三种子能力训练，所以目前只能说“方法与实现成立”，不能说“效果成立”。下一步将用C01--C06训练、C07选择、C08零适配验证；若五类事件、时间关系、几何和高精度关联不能超过预注册基线，该方法仍会在建图前停止。

## 2026-08-29：训练不会删除轨迹开头，也不会伪造时间关系

全数据检查发现轨迹最前面的少量五帧样本缺少部分观察级出口关系标签。这些帧仍有真实LiDAR、当前结构事件和几何标签，因此不能删除。当前正式合同对缺失的关系位置做mask，同时保留该帧的事件、分支和几何训练。

新的拆分loss已经在真实GPU数据上通过：每个样本参加核心语义/几何训练；地点和分支描述符由身份平衡batch单独训练。事件和时间关系的不均衡权重只按C01--C06统计一次，验证集不会参与。

这解除的是完整训练的数据接口阻塞，不是性能结论。下一步仍是三个随机种子的全量训练。

## 2026-08-29：关系表示缺陷已由全量V2纠正，方法进入训练实现

在实现训练Teacher时发现，旧V1/V1R把absent/persistent/reveal/withdraw当成互斥类别，但客观序列中同一2度方向格可能是旧出口withdraw、另一个物理出口同时reveal。全量精确计数为fit/C07/C08=`202/20/33`个方向格；因此旧接口虽通过小批readiness，仍不能作为最终训练方法。

V2采用三个独立二元关系通道persistent/reveal/withdraw，允许跨通道共存但继续拒绝同通道碰撞。正式run逐行读取C01--C08全部80 worlds和`142,184/21,548/24,394`条fit/C07/C08观察，分别得到有效关系pair=`448,152/66,752/77,490`，并无损保留全部255个同时reveal+withdraw案例。模型为`819,067`参数；15项unit、19项方法合同、masked/split CUDA backward、旋转/排列/确定性均PASS。

运行验证33,083项数据seal，峰值RAM=`2,080,928 KiB`，0 optimizer/checkpoint/C09/C10/M-TARE/graph/planner。当前Phase仍为`GSE_GRAPH_GATE3_AXIS_ANCHORED_EVENT_RELATION_THREE_SEED_TRAINING_PREPARATION`，阻塞仍为`AXIS_ANCHORED_EVENT_RELATION_SCIENTIFIC_PERFORMANCE_UNPROVEN`。现在确立的是无损可训练表示，不是最终效果；唯一NEXT是训练Data Card、trainer/evaluator和seeds 0/1/2能力验证。

正式run=`results/gate3_semantics/gate3_20260829_gse_axis_anchored_event_relation_readiness_v2_seed0`；seal SHA=`260e7ff9321111e87a2403c0e4f738e13543dc18419e837af4249f11651823a3`；论文图SHA=`ded447e672b46f42ec28c6adadc6ef6de36be5134559424732cae42a8ad8465e`。

## 2026-08-29：三种子训练证明当前方法尚未确立，密集关系场正式停止

Axis-Anchored V1完成三seed、每seed10轮和总计`36,690`次更新，运行工程完整且`error=null`。统一评估在C07/C08得到event macro-F1=`0.662292/0.633742`、axis误差=`82.7165°/83.0120°`；persistent/reveal/withdraw和current branch field均找不到同时满足安全precision与最低recall的阈值。combined association的C07 recall仅`0.004586`，迁移C08后false merge=`0.041667`。连续几何中的width/height/curvature有可复用信号，但slope显著回归。

当前结论是候选方法科学FAIL，而非模型训练未完成。机制证据指向两处实现退化：axis丢失五帧逐方位融合，极稀疏关系使用数千倍正样本权重的dense BCE。当前run只作为失败分析和消融保留，不进入图。Phase仍为Gate 3；唯一下一步是零训练V2归因/readiness，验证复用时序geometry backbone并采用稀疏圆周关系峰、困难负样本和显式跨帧transport的最小修正。

正式run=`results/gate3_semantics/gate3_20260829_gse_axis_anchored_event_relation_three_seed_training_v1_seed0`；证据清单SHA=`9e9e9f250e35a5fbc354a24dd57153067f7fd22100afc69f67be71b6eadbf5c9`。

## 2026-08-29：正式归因确定V2不是重练，而是更换关系表示

冻结归因精确复现Axis-Anchored V1失败，并证明五帧directional-temporal predecessor把轴向误差从约`83°`降到约`5.5°`。关系pair是否含persistent/reveal/withdraw具有`0.84--0.92` AUC，但dense exact-bin AP很低，特别是reveal/withdraw只有`0.006--0.009`；当前数千倍正样本权重造成高召回与大面积误报。oracle-count的±8°局部定位仍保留`35--46%`跨世界稳定信号。

因此唯一V2候选固定为稀疏圆周关系token与显式跨帧transport，而非增加epoch或调阈值：五帧逐方位特征生成最多6个token，带dustbin的匹配表示persistent/reveal/withdraw，圆周软支持只改善proposal定位，不直接承担关系决策。下一步先做零训练readiness，方法仍未最终确立。

正式run=`results/gate3_semantics/gate3_20260829_gse_axis_anchored_v2_failure_attribution_v1_seed0`；seal SHA=`1cf6ca4c7f8d3c9c4fb3a49526a48a8fd7a688854526802ced9ae6b000f04d0e`；图SHA=`19100bef12934de9eb887d840167982530b7c00d786c0deef7abac5a5070b8f0`。

## 2026-08-29：稀疏关系transport具备训练资格，效果仍待证明

新的`783,675`参数模型已经实现为五帧逐方位骨干、每帧6个圆周token和previous→current/dustbin transport。正式readiness证明旧强骨干53个state keys逐值兼容；同pair reveal+withdraw、persistent、dustbin、rotation、permutation、reverse和past-only语义成立。固定8条真实变化样本上所有参数梯度均存在且有限。

因此接口与训练路径已经确立，但科学方法仍未最终确立。下一步是一次三seed训练：只有token定位/数量、三类关系、axis、事件、几何与安全关联共同在C07/C08达到冻结门槛，才允许进入离线拓扑图。

正式run=`results/gate3_semantics/gate3_20260829_gse_sparse_circular_relation_transport_readiness_v1_seed0`；seal SHA=`293a0abda403697772f09f811127a26b78a30b1f463a6cc64f7703c136d793f6`；图SHA=`e304de6bdc3f544819dce51d83d1036f301ec1c867fdfb220098eb1c1d5c0e05`。

## 2026-08-29：训练前最后两个接口缺口已纠正

稀疏关系候选现在显式预测每个因果帧的0--6结构数量，不再用固定top-6或存在性阈值决定图分支。正式count corrective证明数量目标等于客观presence和唯一identity计数，数量loss确实到达新增头，因果/旋转/排列/确定性均通过。

随后真实完整训练loss发现一个工程缺陷：四维vertical profile加width只有五个回归目标，但头输出六个uncertainty。训练在任何optimizer step前停止；最小纠正只删除多余输出，参数从784,578变为784,513。新的正式corrective证明所有核心和descriptor loss、全部参数梯度与既有不变量通过。

因此当前是“候选方法具备完整训练资格”，不是“方法效果已经成立”。训练器已开始实现；图与规划继续关闭。

## 2026-08-29：Sparse Relation V2R2正式训练执行中

首次V2运行因4小时外层时限低于三seed实测需求，在seed0两轮且无checkpoint/C08/科学结论时主动停止封存。V2R启动后发现评估器只在成功定位端点内部计算relation recall，遗漏真实端点未进入FN；该运行在零epoch处停止，禁止形成性能结论。

零训练corrective现已正式PASS：完整C01--C08 Teacher relation正例由因果identity集合独立统计，fit/C07/C08有效pair=`448152/66752/77490`，所有persistent/reveal/withdraw人口均非零；synthetic missing-endpoint由candidate-only recall=`1.0`纠正为objective recall=`0.4`和3 FN。21/21 unit、8/8 formal checks、18项seal全部通过，seal SHA=`24551f44cf15fa7cf1f3c9a7a334a1bd4d84bd552fe45246b759f49bf99482fb`。

当前Phase=`GSE_GRAPH_GATE3_SPARSE_CIRCULAR_RELATION_TRANSPORT_V2R2_THREE_SEED_TRAINING_RUNNING`；阻塞=`SPARSE_CIRCULAR_RELATION_TRANSPORT_V2_SCIENTIFIC_PERFORMANCE_UNPROVEN`。V2R2保持784,513参数模型、C01--C06训练、C07选择、C08一次迁移和全部预注册门槛不变，只绑定8小时外层时限和完整Teacher召回口径。方法性能仍未确立，拓扑图与规划器继续关闭。

## 2026-08-29：训练系统阻塞已由等价cache生命周期纠正

V2R2/V2R3均没有产生完整三seed科学结果。正式资源V1否决减小evaluation batch，因为batch128使加权validation loss改变；V1R在不改batch256的条件下完成全部10个C07 worlds和21,548条观察，输出与loss逐项误差均为0，最大进程显存约9.71 GiB，低于16 GiB合同。

训练器现仅在独立validation/final-inference world边界释放未使用CUDA cache，25项测试通过。当前Phase=`GSE_GRAPH_GATE3_SPARSE_CIRCULAR_RELATION_TRANSPORT_V2R4_TRAINING_PREPARATION`；阻塞仍是`SPARSE_CIRCULAR_RELATION_TRANSPORT_V2_SCIENTIFIC_PERFORMANCE_UNPROVEN`。下一步从头完成三个seed，不能把资源PASS表述为方法成立。

## 2026-08-29：V2R4训练world显存硬门FAIL

V2R4只完成seed0前三轮，训练过程显存达到`16,386 MiB`，按16 GiB硬门停止并封存。前三轮展示学习趋势但没有三seed统一selection，不能用于确立方法。当前Phase=`GSE_GRAPH_GATE3_SPARSE_RELATION_TRAINING_CACHE_RESOURCE_CORRECTIVE`；下一步只允许seed0单epoch精确权重/指标parity和60-world资源证明。

## 2026-08-29：训练world缓存纠正精确PASS

单epoch正式证明得到77/77 tensor exact、全部epoch指标误差0，最大进程显存12.017 GiB。训练器和runner现直接记录并执行16 GiB硬门。当前Phase=`GSE_GRAPH_GATE3_SPARSE_RELATION_V2R5_THREE_SEED_TRAINING_PREPARATION`；下一步从头三seed，科学性能仍未确立。

## 2026-08-29：V2R5重新分类为主方法前的学习式token/transport基线

V2R5仍在正常运行，但只读审计证明它没有把预测`local_axis`用于出口结构关系：token在sensor azimuth上产生，关系仅为persistent/reveal/withdraw。与此同时，fit/C07/C08的route-aligned pose使水平axis几乎恒为robot-forward；常数前向基线的三维axis平均误差仅`2.79198/3.26039/3.33944deg`，比当前模型更好。

当前Phase=`GSE_GRAPH_GATE3_SPARSE_RELATION_V2R5_BASELINE_TRAINING_RUNNING`；阻塞=`FINAL_ROUTE_FRAME_GEOMETRY_RELATION_METHOD_NOT_ESTABLISHED`。V2R5完成后保留全部checkpoint、预测、指标和论文图作为基线/消融，但旧`axis<=10deg`与旧selection不得开放graph。下一步必须使已知运动方向只承担坐标锚，学习的分支关系和几何语义直接决定node/exit association；若声称学习axis，必须增加非平凡yaw验证并超过常数基线。

附加核心阻塞：当前`context→event`独立分类头直接触发图节点，token/transport/metric geometry并未组成事件。因此最终主方法必须实现geometry-relation-conditioned event composer，并通过对独立event head、无显式几何和无transport三项消融；否则即使V2R5旧指标通过，也只能作为基线而不是论文核心方法。

## 2026-08-29：主方法已有冻结候选规范，尚未完成有效性证明

V2R5仍正常运行，但其event head使用旧的12,534帧fit transition mask，不是已封存的corrected causal change-point Teacher。因而V2R5仅保留token、transport、metric geometry和descriptor组件作为可复用基线，独立event head降为对照。

主方法候选已冻结为`docs/GSE_GRAPH_METHOD_SPEC_V1.md`：Action-Set Relation Composer管理junction/terminal/provisional，Metric-Change Composer管理turn/geometry-transition与位置回投，后续学习关联只使用descriptor、exit-token几何一致性、event type、执行路程候选域和uncertainty refusal；边只由真实穿越建立。

当前Phase仍为`GSE_GRAPH_GATE3_SPARSE_RELATION_V2R5_BASELINE_TRAINING_RUNNING`；阻塞更精确为`FROZEN_METHOD_CANDIDATE_NOT_YET_VALIDATED`。下一步不是建图，而是V2R5封存后对corrected Teacher执行零训练容量/冲突审计，再决定是否允许双Composer readiness。

已完成一次不作正式结论的seed0接口诊断：显式出口集合对junction/terminal在C07/C08的AUC约为`0.98`，turn为`0.69/0.72`，但宽高及出口几何四帧差对corrected geometry-transition仅`0.588/0.593`。因此Action-Set Composer有明确容量信号，Metric-Change Composer尚有实质风险；正式审计须使用三seed完整token/transport状态而不是简单均值。

同一seed0又执行了一次全量642维显式状态的C07→C08线性可分性诊断：junction/terminal AUC=`0.9479/0.9955`，turn=`0.7338`，geometry-transition=`0.5744`；后两者AP仅`0.0264/0.0198`。这不是formal run，但排除了“仅因均值池化才看不到change-point”的简单解释。三seed重复前不改方法状态。

相关工作原始来源复核进一步确认方法尚未确立：Cano已覆盖出口感知、时序稳定和纯拓扑；PRISM-TopoMap覆盖学习式在线地点关联；已有Semantic Topometric Mapping覆盖结构语义探索；时序概率descriptor与uncertainty filtering也已有独立工作。故Action-Set、descriptor和refusal都不能单独承担创新。

当前唯一可投稿候选被收窄为完整因果链：学习的路线连续度量几何与可执行出口关系必须直接组成typed events，事件和几何控制节点生成/拒绝关联，edge只由真实穿越提交，并最终改善图质量和探索。Metric-Change因此是必要条件；若V2R5三seed正式显式审计仍不能恢复corrected change points，必须先验证RouteGeometryProfile，而不能退回出口计数或用planner补偿。V2R5 seed1当前完成6/10轮，运行正常；方法状态仍为`FROZEN_CANDIDATE_NOT_ESTABLISHED`。

正式可观测性审计代码现已就绪：只消费642维显式状态，C07拟合固定诊断probe，C08一次原样迁移；旧独立event head仅作baseline。真实seed0的45,942行全量smoke和10项unit均PASS。C07/C08 transition独立identity只有`5/12`，因此正式证据必须同时报告AUC、AP、迁移gap和identity coverage，不能从5个C07 identity声称最终泛化。V2R5 seed1现已完成8/10轮，进程正常；三seedseal前不会创建审计run。

seed0端到端readiness进一步得到C08 explicit AUC=`0.9490/0.9966/0.7422/0.6577`（junction/terminal/turn/transition），但98% precision迁移阈值下transition identity coverage仍为`0/12`，所以这只证明非随机容量，不证明可部署性能。该运行还发现草案metric缺陷：probe在C07拟合，C07训练AUC接近1，不能用其与C08的gap作泛化门。正式card尚未freeze，已把gap改为完整诊断，并继续用untouched C08 AUC/AP、三seed一致性和identity报告决定是否允许Composer readiness；后续安全门未改。

三seedensemble也已在freeze前纠正：不平均可能错位的跨seed 642维token状态，而是每seed独立拟合四个probe后对同事件score等权平均。因此正式审计为12个诊断probe，不存在额外“mean-feature模型”；数据、阈值和C08隔离不变。

V2R5当前已完成seed0和seed1，seed1共`12,230`步、best epoch=`8`；seed2已自动启动且资源正常。审计代码沿trainer导出链确认`geometry`为模型预测而非Teacher真值，并删除了未使用的跨seed feature mean；当前12项单元测试PASS。正式Data Card/spec仍必须等三seed完整seal后冻结，图与规划继续关闭。

历史实现复用边界也已核清：旧ActionSet detector使用三seed联合输入且descriptor直接进入事件，旧CausalGeometryDelta head使用hidden context与baseline event logits，二者均违反新方法的显式因果接口，不能充当主Composer。后续只能复用其集合池化、因果mask和loss工具；审计PASS后仍需实现每seed独立、descriptor-free的Action-Set Composer与explicit-geometry-only Metric-Change Composer。

Composer freezer已完成一次零落盘内存模拟：使用真实source Data Card的C07/C08 worlds与trajectories，只模拟尚未发生的baseline completed/seal/hash条件；生成的Data Card和run spec均通过当前governance validator，expected counts为`10/10` worlds、`21,548/24,394` observations、12 probes及零C09/C10/M-TARE/graph。正式两个JSON确认仍不存在，三seedseal前没有提前freeze或create run。

真实未完成态保护也已验证：freezer现先检查V2R5 `RUN_STATE=COMPLETED`且`error=null`，再读取final summary/seal；当前`RUNNING`调用明确拒绝且零文件落盘。治理与Composer联合回归仍为37项全PASS，训练进程未受影响。

已新增面向论文主线的统一报告`docs/GSE_GRAPH_TRAINING_METHOD_FAILURE_REPORT.md`：明确Teacher由程序化TNG/spline/mesh/directed traversal客观生成，解释corrected causal Teacher、数据划分、七类已试方法的原理、量化失败依据、共同根因、双Composer候选及下一判定。报告共225行，引用10个现有正式summary/seal/PNG，全部本地链接存在；后续正式结果只追加到这一报告，避免继续产生口径不一致的零散说明。

同一内容现已整理为可直接浏览器汇报的`docs/GSE_GRAPH_METHOD_EXPERIMENT_REPORT.html`。应用户要求，报告扩展为无需项目背景也可阅读的中文技术版：新增术语表、100个程序化地下世界的TNG→spline→mesh→LiDAR生成过程、正反有向穿越与1m/五帧因果采样、90-world完整导出与当前80-world开发读取边界、Teacher逐项标签公式与代码、学生/Teacher防泄漏边界，以及V2R5实际训练伪代码。当前为567行、36,443字节、11章节、7张正式实验图、5段代码；全部本地引用存在。Chrome实测首屏与长表格正常，可打印为17页、2.89MB临时PDF。更新时V2R5 seed2完成7/10轮，训练继续按固定checkpoint规则运行。

## 2026-08-29：V2R5三种子完整封存为科学失败基线

V2R5无系统错误完成三种子各10轮、共`36,690` optimizer steps；每个seed均有`12,230`步、20个C07/C08预测包和最佳epoch=`8/8/7`。正式run状态为`COMPLETED`、`error=null`、90项证据seal SHA=`2d3ddb4bec55244dc50be739fb4b0ce70298972fa7fc50a5ddc2c463924e3e59`，C09/C10/M-TARE/graph读取或执行均为0。

科学结果为FAIL：C07/C08旧事件头macro-F1=`0.641491/0.590305`，turn F1=`0.327559/0.162264`，旧transition F1=`0.099701/0.047411`；学习几何相对非学习估计器的宏改进仅`4.374%/7.795%`，低于10%合同，且slope分别回归`109.7%/90.1%`。C08高精度结构拒绝precision=`0.989156`但recall=`0.387289`；combined association precision=`0.976744`、recall=`0.011257`、false merge=`0.023256`；persistent/reveal/withdraw在高精度合同下均接受0。故V2R5只保留为token/transport/geometry/descriptor组件基线，不允许开放graph。

面向导师汇报的HTML已按科研证据链重构：新增真实复杂3D世界XY/XZ图、mesh-spline对应图、16×720 LiDAR样本、24帧传感器合同图和早期规则在线图；每条失败路线均按研究假设、实验设计、对照、量化结果、否定依据和保留组件展开，并新增V2R5最终失败表及三个当前科学障碍。正式Composer观测审计Data Card/spec已在完整seal后冻结；首次preflight只因`results/project_status.json`残留已结束run的running marker而拒绝，现已按真实状态清空，下一步重新preflight。

## 2026-08-29：显式双Composer可观测性正式PASS，但高精度节点覆盖仍未通过

状态同步后formal preflight为0 error/0 warning，唯一不可覆盖run=`results/gate3_semantics/gate3_20260829_gse_composer_observability_v1_seed0`已完成。审计在C07/C08=`10/10` worlds、`21,548/24,394` corrected-Teacher rows上拟合`3 seeds × 4 events = 12`个固定线性probe；C08零适配、三seed只做score ensemble，0主模型更新、0 checkpoint更新、0 C09/C10/M-TARE/graph。运行`200.20 s`、peak RSS=`1,221,012 KiB`，18项seal SHA=`a6cf980b450ba233c57a89a42f2d5723a061f88a5d8215e050466f7229eb7436`。

显式ensemble在C08的junction/terminal/turn/transition AUC=`0.965620/0.999131/0.768548/0.737538`；turn/transition AP=`0.151206/0.017807`，分别为prevalence的`15.56x/2.51x`，至少2个seed transition AUC≥0.60，全部预注册容量门PASS。因此双Composer readiness获准，不触发RouteGeometryProfile fallback。

PASS只证明显式状态包含可迁移排序信息，不证明可提交节点。C07选择的高精度阈值迁移到C08后，junction/terminal/turn/transition物理identity覆盖=`73/77, 69/69, 4/52, 0/12`；特别是transition仍无法在高精度下提交。下一步必须实现每seed独立、descriptor-free的Action-Set Relation Composer和explicit-geometry-only Metric-Change Composer，并用precision/recall/identity coverage与三个必要消融决定能否进入graph。

导师汇报HTML现扩展为711行、53,939字节、13张真实/正式实验图和5段代码，所有本地引用存在；Chrome完整打印为28页。可携带PDF=`docs/GSE_GRAPH_METHOD_EXPERIMENT_REPORT.pdf`，大小6,240,322 bytes，SHA-256=`3d3902252d55c277d7510bc338fba890795410f99c6f9db27ad6d48e8b9535b7`。

## 2026-08-29：双Composer受限接口与可训练路径正式PASS

新主方法模块`src/mtare_topo/representation/gse_typed_composers.py`已独立实现，未薄封装历史ActionSet/CausalGeometryDelta事件头。Action-Set Relation Composer只能读取五帧出口显式几何、count和transport；Metric-Change Composer只能读取五帧预测通道几何及不确定性，并输出严格落在过去/当前五帧内的回投分布。descriptor、hidden context、旧event logits、pose/world/TNG/identity/future在typed输入层不可表达。

正式run精确核对C07 10 worlds、21,548条Teacher/预测索引，并对覆盖corridor/junction/terminal/turn/geometry-transition的10个固定真实样本执行forward/loss/backward。Action/Metric参数量=`102,500/36,485`；15/15检查和6/6 unit通过，token permutation、global rotation和两个batch permutation最大误差均`2.98e-8`，repeat误差0；两模块全部参数均有finite gradient。

run=`results/gate3_semantics/gate3_20260829_gse_typed_composer_readiness_v1_seed0`，状态`COMPLETED`、`error=null`、peak RSS=`711,088 KiB`、19项seal逐项复核零mismatch，seal SHA=`2111d45d13377487acc026e8601375c91caa08e49f78de6f71a58ddb8b2c438d`。optimizer/checkpoint/C08/C09/C10/M-TARE/graph/planner均为0。

这解除的是实现与因果接口阻塞，不是科学性能阻塞。下一步冻结三seed训练Data Card和执行链：每个Composer只能使用对应V2R5 seed的冻结显式输出，C01--C06拟合、C07选择checkpoint/温度/refusal、C08一次零适配；必须同时改善corrected-Teacher macro-F1、turn/transition frame及identity coverage，并满足precision≥0.98、false accept≤1%、recall≥0.25和三项必要消融，才可开放离线图。

## 2026-08-29：geometry-only缓存闭合三seed训练输入来源

正式缓存V1在任何fit资产写入前，因当前重推`token_count_probability`与历史float16归档最大差`7.629e-6`按全字段逐位合同FAIL并封存。随后只读同进程重复/历史对照证明：当前重复前向所有字段逐位一致；八个主体字段跨历史也逐位一致；只有CUDA softmax/sigmoid派生的count/transport-row/reveal概率存在`7.63e-6/4.88e-4/1.91e-6`末位差异，count/transport argmax和reveal@0.5全部不变。

V1R据binary16表示固定唯一corrective：主体字段继续bit-exact，三个派生概率绝对差不得超过`np.finfo(float16).eps=0.0009765625`且所有离散决策相同。正式V1R完成3×80世界和564,378次冻结forward；主体最大误差0、派生最大误差`0.00048828125`、60/60 development seed-world决策全部一致。

最终240个世界级NPZ总计473 MiB，只含11个allow-list字段，不含旧事件输出、hidden context、place/token descriptor、pose/world/TNG/identity。每seed人口精确为fit/C07/C08=`142,184/21,548/24,394`；peak GPU=`4,559,283,200 bytes`、peak RSS=`2,247,960 KiB`、运行347.45s。run=`results/gate3_semantics/gate3_20260829_gse_explicit_composer_cache_export_v1r_seed0`，265项seal SHA=`edc613469f91c15f87be39c87cc3416e544e989fe267263fe768668dfe65faed`。

因此Composer训练不再重复执行感知骨干，且训练输入文件本身可审计地禁止旁路。下一步是实现corrected-Teacher label/backprojection packer、每seed小Composer trainer、C07-only checkpoint/温度/refusal校准、C08一次迁移及no-metric/no-transport/no-refusal消融；graph仍关闭。

## 2026-08-29：双Composer因果监督物化V1R完成

V1的80个监督NPZ和summary已经完整写出且七项科学检查全PASS，但绘图函数把字典对象直接作为Matplotlib分类轴输入，最终抛出`unhashable type: dict`。该失败发生在训练前，正式V1封存为系统FAIL；其科学summary明确为PASS，模型更新/推理/test/graph均为0。

V1R只把绘图输入改成显式key/value列表，并新增真实三格式输出单测。正式run精确得到80 worlds、188,126 rows，fit/C07/C08=`142,184/21,548/24,394`；五类事件=`150,964/26,608/7,525/1,998/1,031`；turn/transition有效回投帧=`1,174/410`，覆盖全部`392/76`个身份。run=`results/gate3_semantics/gate3_20260829_gse_composer_supervision_v1r_seed0`，99项证据seal SHA=`3c2b8511fb11b8d8c91784871be5c12ce13d26db737997e55c45165f7cf6ff54`。

监督资格现已解除；下一步实现并冻结三seed双Composer训练与三项必要消融。该PASS不代表主方法性能通过，也不开放graph。

## 2026-08-29：导师汇报版按科研问题—证据—决策链重做

新增`docs/GSE_GRAPH_ADVISER_PROGRESS_REPORT.html`及可直接汇报的PDF。该版本不再按内部实验编号堆叠历史，而是用13节说明：论文问题与Cano差异、TNG→spline→mesh→LiDAR地图生成、Teacher来源与因果修正、早期出口规则图、自由三维候选/几何锚定/稠密关系/V2R5四组关键失败、显式几何可观测性、双Composer当前实现、真实障碍与后续停止条件。

报告包含9张已有真实实验图，其中并列展示树形、单环和复杂三维三类生成地图；每个失败实验均明确假设、做法、预注册对照、实际数字和由此排除的路线。Chrome打印检查为19页，所有9个本地图片引用存在，页面接触表复核无图表溢出。PDF大小`5,498,107 bytes`，HTML/PDF SHA-256分别为`26beb6936b26b6f9c019b0e60517a6d80f963da5ac1612eb413fd9b7f46452cd`与`0f8465283ff2b299eac4ad13c0e2b5e89460cd850aa8849a258e3bfeb656b8c3`。原28页报告保留为技术附录，未修改实验数据或科学结论。
# 2026-08-29：ERCSS 基础实现通过，等待全量零训练可行性审计

五帧相对配准和可见骨架 Teacher 的基础代码已经完成。配准只输出当前机器人坐标，已证明对全局平移/旋转不变并禁止使用未来帧；Teacher 将每条物理 edge 独立采样，通过过去五帧的 native-mesh 视线筛选后只保留机器人所在的可见连通分量，不跨越遮挡间隙。

13项单元测试全部通过。这只证明接口定义正确，不代表 LiDAR 对所有结构事件都有足够信息。下一步是一次全量、零训练审计，统计 junction、terminal、turn 和 geometry-transition 的身份覆盖与骨架容量；未通过前不训练、不建图、不读取 C09/C10 或 M-TARE。
# 2026-08-30：ERCSS 科学可行性仍未知，阻塞在最终 run 治理

五帧配准和可见骨架代码现有10项专项测试全部通过，Open3D sidecar也已证明可在不安装Torch的情况下导入纯几何模块。正式全量审计仍没有读到任何世界：V1缺Open3D，V1R被旧包初始化隐式Torch依赖拦截，V1R2又被冻结器继承的旧测试哈希拦截。

因此不能说ERCSS成功或失败；当前只有Composer和单帧RouteGeometryProfile的科学FAIL是有效结论。V1R2已触发预注册停止条件，不能再自动创建run。推荐的唯一纠正是V1R3只重新冻结当前全部工具SHA，保持80 worlds、188,126窗口、37,162事件、Teacher、可见性、80%身份覆盖和容量门完全不变。该治理覆盖需要用户明确决定。
# 2026-08-30：V1R3已技术预备，正式状态仍等待明确治理覆盖

最后一次工具哈希纠正的冻结器和runner已经准备好，并加入强制授权令牌。只读自检确认它会从当前12个工具文件重新计算SHA，不再复制V1R2的旧测试哈希；自检没有创建Data Card、spec或run，也没有读取任何数据。

因此当前科学状态没有变化：ERCSS全量可行性未知。只有用户明确允许V1R3后才能冻结和执行；否则按停止规则结束ERCSS候选。

# 2026-09-02：可观测基元关系三种子训练，seed 0 第一轮完整落盘

正式run=`results/gate3_semantics/gate3_20260902_primitive_relation_observable_three_seed_training_v1_seed0`继续运行。seed 0 epoch 0已完整覆盖fit/C07=`426,552/64,644`条五帧序列、`26,736/522`批和`26,736`次optimizer step；冻结几何/时序状态检查通过后写出`epoch_00.pt`。单轮耗时`6,357.084 s`，CUDA allocated峰值`7,669,031,936 bytes`，GPU进程峰值`15,443,427,328 bytes`，未越过16 GiB合同。

训练关系objective=`0.154206`，C07 relation-selection loss=`0.321828`（port relation=`0.640978`、uncertainty=`0.002677`），显示训练/C07泛化差距；该轮不能单独确立或否决方法。seed 0 epoch 1已自动启动，C08/C09/C10、graph、M-TARE仍为0读取/0执行；必须等待三个seed完整训练和最终精确precision/recall/F1门禁。

# 2026-09-02：可观测基元关系 seed 0 完整训练，seed 1 自动接续

同一正式run的seed 0已完成三轮，每轮精确覆盖fit/C07=`426,552/64,644`条五帧序列和`26,736/522`批，总计`80,208`个optimizer steps。最佳checkpoint为epoch 2；C07 relation-selection loss按轮为`0.321828/0.240373/0.203239`，相对首轮下降`36.847%`，port-relation loss为`0.640978/0.478451/0.404346`，显示关系分支在未见C07上的连续改善。

运行合同完整：可训练/冻结参数=`742,149/1,893,482`，冻结状态前后SHA-256均为`cdfcbf238a464bee773060f17e5e5ef315653ca72a5504d53b5b0594bca706de`；C08与C09/C10读取均为0；CUDA allocated峰值`7,672,127,488 bytes`、GPU进程峰值`15,516,827,648 bytes`、host RSS峰值`2,689,691,648 bytes`，均未超过16 GiB合同。seed 1已自动启动。

这些结果只证明seed 0训练稳定和冻结几何未漂移，不代表科学门PASS。必须等seed 1/2完成后，对三个selected checkpoint执行完整C07阈值扫描；至少两seed同时满足surface与连续几何相对非学习基线改善不少于10%、attachment F1增加不少于5个百分点、primitive F1和coverage不退化，并在precision不低于0.98时具有非零真连接，才允许读取C08或进入graph。

# 2026-09-02：可观测基元关系 seed 1 第一轮完整落盘

seed 1 epoch 0已完整覆盖fit/C07=`426,552/64,644`条五帧序列、`26,736/522`批和`26,736`个optimizer steps，耗时`6,348.128 s`。CUDA allocated峰值`7,673,094,144 bytes`、GPU进程峰值`15,443,427,328 bytes`，均未超过16 GiB合同；epoch 1已自动启动。

该seed首轮C07 relation-selection loss=`0.650768`（port relation=`1.298934`、uncertainty=`0.002601`），显著高于seed 0首轮的`0.321828`。这证明当前存在真实跨seed方差，不能以seed 0代替三seed一致性；但seed 1仍有两轮，且seed 0曾从首轮持续改善到第三轮，因此此处不提前作科学FAIL，也不改变学习率、轮数、数据或选择规则。C08、C09/C10、graph和M-TARE继续保持0读取/0执行。

# 2026-09-02：可观测基元关系 seed 1 第二轮完整落盘

seed 1 epoch 1已完整覆盖fit/C07=`426,552/64,644`条五帧序列和`26,736/522`批，累计`53,472`个optimizer steps，单轮耗时`6,341.734 s`。C07 relation-selection loss从epoch 0的`0.650768`降到`0.548976`，降幅`15.642%`；port-relation loss从`1.298934`降到`1.095994`，降幅`15.624%`；uncertainty calibration error从`0.00260148`降到`0.00195666`，降幅`24.787%`。

该轮CUDA allocated峰值=`7,673,094,144 bytes`、GPU进程峰值=`15,518,924,800 bytes`，仍未越过16 GiB合同。seed 1 epoch 2已自动开始。这些数据证明seed 1继续在C07上改善，但其损失仍明显高于seed 0同轮，因此不预告三seed科学门PASS，不改变冻结训练合同，C08、graph和M-TARE仍关闭。

# 2026-09-02：可观测基元关系 seed 1 完整训练，seed 2 自动接续

seed 1已完成三轮，每轮精确覆盖fit/C07=`426,552/64,644`条五帧序列和`26,736/522`批，总计`80,208`个optimizer steps。C07 relation-selection loss为`0.650768/0.548976/0.570588`；epoch 1相对epoch 0降低`15.642%`，但epoch 2相对最优epoch 1回升`3.937%`，因此runner正确选择epoch 1而不是最后一轮。这是轻微过拟合/跨轮波动证据，不得只报告首两轮的下降。

运行合同完整：可训练/冻结参数=`742,149/1,893,482`，冻结状态前后SHA-256均为`bfa0a762ce80919da9aed65204fcccf2ed63fc6af05bc6ee1b0ba5153dbfa1eb`；C08与C09/C10读取均为0；CUDA allocated峰值=`7,673,094,144 bytes`、GPU进程峰值=`15,518,924,800 bytes`，未越过16 GiB合同。seed 2已自动开始。seed 1最优损失仍明显高于seed 0的`0.203239`，三seed最终precision/recall/F1门仍未判定，C08、graph和M-TARE继续关闭。

# 2026-09-02：可观测基元关系 seed 2 第一轮完整落盘

seed 2 epoch 0已完整覆盖fit/C07=`426,552/64,644`条五帧序列、`26,736/522`批和`26,736`个optimizer steps，耗时`6,356.791 s`；checkpoint SHA-256=`4f953d72aa223421ad7ad826d555decca1ea8c316243718c3b8587e19d325370`。训练合同明确保持三轮、冻结几何/时序状态、dual-endpoint可观测Teacher和C07-only selection，epoch 1已自动接续。

首轮C07 relation-selection loss=`0.844769`（port relation=`1.686916`、uncertainty=`0.00262146`），高于seed 0/1首轮的`0.321828/0.650768`，跨seed方差进一步增大；但seed 2仍有两轮，完整precision/recall/F1和几何门尚未执行，因此不提前判科学FAIL。CUDA allocated峰值=`7,673,307,136 bytes`、GPU进程峰值=`15,443,427,328 bytes`，均低于16 GiB合同；C08、C09/C10、graph和M-TARE继续保持0读取/0执行。

# 2026-09-02：可观测基元关系 seed 2 第二轮完整落盘

seed 2 epoch 1继续精确覆盖fit/C07=`426,552/64,644`条序列和`26,736/522`批，累计`53,472`个optimizer steps，单轮耗时`6,342.912 s`；checkpoint SHA-256=`a41b31f1523fd5505770669ddebb1794ff97d94fd7b0d381915ff11dc52c6ee5`。C07 relation-selection loss从`0.844769`降到`0.339378`，降幅`59.826%`；port-relation loss从`1.686916`降到`0.677207`，降幅`59.855%`。该轮已优于seed 1最佳`0.548976`，但仍弱于seed 0最佳`0.203239`，最终关系F1与安全精度仍需统一评估。

CUDA allocated峰值=`7,673,307,136 bytes`、GPU进程峰值=`15,518,924,800 bytes`，均未越过16 GiB合同。epoch 2已自动开始；训练设置、Teacher、数据split和选择规则保持冻结，C08、C09/C10、graph和M-TARE继续为0读取/0执行。

# 2026-09-02：可观测基元关系三种子训练全部完成，精确 C07 评估启动

同一不可覆盖run的seed 2 epoch 2已完成，第三轮C07 relation-selection loss=`0.261973`（port relation=`0.522612`、uncertainty=`0.00133434`），较epoch 1的`0.339378`下降`22.808%`；runner选择epoch 2。seed 2共完成`80,208`个optimizer steps，冻结状态前后SHA-256均为`37c15277c2ae224503c9a7007c49ca63925f959028773a8d4573f5b799eae633`，GPU进程峰值=`15,518,924,800 bytes`，C08与C09/C10读取为0。

至此seed 0/1/2各完成三轮，总计`240,624`个optimizer steps；最佳C07选择损失分别为`0.203239/0.548976/0.261973`，最佳轮次为`2/1/2`。这只证明训练与冻结合同完成，不是科学PASS。正式runner已自动启动精确C07评估，对三个selected checkpoint逐一扫描全部`64,644`条C07序列并计算真实几何、attachment precision/recall/F1、安全阈值和2/3种子判定；评估完成前C08、graph和M-TARE继续关闭。

# 2026-09-02：可观测基元关系 C07 资格门 0/3，通过几何但系统性失败于安全连接

正式精确评估已完成，seed 0/1/2均覆盖同一`64,644`条C07序列、`442,936`个可观测连接正例，passing seeds=`0/3`，总决定=`STOP_OBSERVABLE_RELATION_BEFORE_C08_AND_GRAPH`。三个seed的surface improvement=`56.38%/49.19%/52.62%`，连续几何macro improvement=`20.49%/38.90%/29.11%`，primitive F1 gain=`30.07/14.08/19.25`个百分点，target coverage gain=`69.83/80.06/74.82`个百分点；程序构造监督对显式几何的收益在三seed一致成立。

失败集中在物理连接：普通attachment precision=`5.65%/3.16%/3.23%`、recall=`13.74%/12.79%/14.90%`、F1=`0.0801/0.0506/0.0531`；相对同输入非学习基线的F1 gain=`7.37/4.42/4.67`个百分点，仅seed 0达到5个百分点。更关键的是，固定安全分数`p_attachment × 双端点证据 × (1-uncertainty)`在precision≥0.98条件下三个seed全部`TP=0, FP=0`，即全拒绝而非可用高精度连接。因此问题是关系候选/证据与置信度的系统性模型失败，不是训练中断、资源超限或单seed偶然。

正式run状态为COMPLETED、`error=null`，总计`240,624`个optimizer steps，C08/C09/C10/M-TARE读取和graph replay均为0；源码integrity前后逐字节一致，结果大小约`199,961,200 bytes`，54项证据seal文件SHA-256=`544b6be586bdb472a12c8912b16a8cf87105ed7e4ff44302e1df239527f0553c`。按冻结停止门不开放C08和graph，不挑seed、不降0.98门、不增加手写关系规则。下一步只允许基于本次密封C07输出做零训练、零新推理的失败归因和方法去留决策。

# 2026-09-02：最终 C07 评估发现 cuDNN TF32 数值合同漂移，0/3精确判定暂不合规

在失败归因前复核执行路径发现：每个seed的训练与逐轮C07 selection都调用`_configure_determinism`并显式设置matmul/cuDNN TF32为False、float32 matmul precision为highest；但独立最终评估脚本没有调用该配置，runner环境只设置`CUBLAS_WORKSPACE_CONFIG`，当前Torch 2.9新进程默认`cuda.matmul.allow_tf32=False`而`cudnn.allow_tf32=True`。现有20项相关测试未覆盖最终评估进程的该状态。

这使上一节0/3结果成为完整、密封但数值合同不合规的系统证据：它可用于提示风险，不能作为最终C07科学Gate或方法修订依据。训练checkpoint、数据/Teacher、冻结几何和C08零读取不受影响；不需要重训。唯一最小纠正是使用同三个selected checkpoint、同`64,644`条C07序列、同阈值与门，在显式TF32-off进程中重跑两遍评估，共`387,864`次模型前向，0 optimizer、0 C08/C09/C10/graph/M-TARE。纠正完成前C07状态恢复为未判定，failure attribution暂停。

# 2026-09-02：TF32-off C07纠正正式确认0/3，直接关系头停止并转入失败归因

同三个冻结checkpoint已在训练一致的数值合同下完成纠正评估：确定性算法开启、matmul TF32关闭、cuDNN TF32关闭、float32 matmul precision=`highest`。每个seed完整处理`64,644`条C07序列并执行两遍，共`387,864`次前向、0 optimizer；C08/C09/C10、graph和M-TARE读取或执行均为0。

纠正后的seed 0/1/2表面改善=`56.376%/49.189%/52.600%`，连续几何macro改善=`20.514%/38.906%/29.045%`，证明显式扫掠几何学习仍稳定成立；但普通attachment precision仅=`5.654%/3.155%/3.230%`，recall=`13.745%/12.795%/14.892%`。固定安全分数在precision≥0.98门下三seed全部`TP=0, FP=0`，passing seeds=`0/3`，最终科学决定为`STOP_OBSERVABLE_RELATION_BEFORE_C08_AND_GRAPH`。

源纠正run在科学计算、metrics、RUN_STATE和28项seal均写完后，最后一行将`_seal()`返回的整数再次传给`len()`而在外层进程抛出TypeError。没有修改源run，也没有重跑GPU；独立零推理证据纠正run逐项验证28/28源hash、三seed人口、数值合同、资源、隔离和上游输入/工具，复制紧凑证据并生成新的28项seal。该run状态`COMPLETED`、`error=null`、证据seal SHA-256=`ea5b0833c1ba2c1e03e70e3ede38dc85eae38c23e08d78d44a78c915265c698b`。

当前P2科学状态因此正式为FAIL：失败对象是“当前直接端口关系头”，不是程序构造监督的几何学习子主张。按冻结停止门，C08、在线结构图和M-TARE继续关闭；下一步只允许使用密封C07/checkpoint做关系失败归因，区分候选存在性、双端点可观测证据、组合空间、关系表示和置信度校准。不得挑seed、降低0.98门、删除帧或用手写规则补成PASS。

证据run=`results/gate3_semantics/gate3_20260902_primitive_relation_observable_c07_tf32_evidence_corrective_v1_seed0`。

# 2026-09-02：C07失败归因确认直接pair关系分数本身失败，转向连接超边Teacher可行性

正式归因对三个冻结checkpoint各执行一次完整C07前向，共`193,932`行、0 optimizer，并逐项复现纠正后的attachment和safe计数。全部九个预注册条件的passing seeds均为0；即使使用evaluation-only proposal oracle删除全部冗余候选，或只保留冻结关系分数的best-link，三个seed仍都找不到任何precision≥0.98的真实连接。

量化上，部署态每帧平均激活槽位=`16.14/29.01/23.27`，冗余激活总数=`567,814/1,344,267/1,001,555`；proposal oracle把关系F1从正式的`0.0801/0.0506/0.0531`提高到`0.2215/0.1725/0.2075`，说明候选爆炸是损失来源之一，但oracle raw与best-link的安全TP仍全部为0。端点可见性头在oracle matched endpoints上的F1稳定为`0.89188`，证明“是否看到端点”不是主阻塞。正式诊断为`RELATION_SCORE_FAILS_EVEN_WITH_OBSERVABLE_PROPOSAL_ORACLE`，决定为`STOP_DIRECT_PAIR_RELATION_HEAD_AND_REASSESS_ARCHITECTURE`。

问题归类为模型关系表示，不是数据中断、Teacher泄漏、阈值网格、资源或单seed偶然。继续重加权独立pair BCE或降低0.98门均被证据否决；用距离/角度手写连接也违反主方法。计划内证据最强的下一候选是`Primitive Connection Hypergraph`：网络不再独立分类所有端点pair，而是学习把每个可见基元端点分配到少量可交换连接簇；同簇端点自然形成T/Y/X多元组合关系，未连接端点进入dustbin/singleton，最终仍导出原attachment接口。

在实现或训练该head前，唯一下一步是零训练、零新模型推理的C01--C06 fit与C07 selection Teacher feasibility：证明observable attachment是无冲突的端点集合划分、每端点至多属于一个物理连接簇、观测裁剪后簇仍为clique、容量可从fit-only冻结且C07零overflow，并量化pair到cluster监督的组合压缩。C08/C09/C10、graph和M-TARE继续关闭。

归因run=`results/gate3_semantics/gate3_20260902_primitive_relation_observable_failure_attribution_v1_seed0`；27项seal SHA-256=`fef3891cad80602437d8c45afd9d00c8505b0921c8824984f78c278cf7d6cf76`。

# 2026-09-02：连接超图Teacher语义成立，但稠密全局槽解码器因组合压缩失败而停止

正式零训练审计完整读取C01--C06 fit的60个parent/180个配对几何shard/426,552行和C07的10个parent/30个shard/64,644行，共491,196行紧凑Teacher；读取传感器帧、模型前向、optimizer、C08/C09/C10、graph和M-TARE均为0。全部fit/C07标签都通过active destination、cross-primitive、canonical storage、symmetry和clique transitivity；因此程序构造Teacher确实把每个端点唯一分到至多一个物理连接簇。C07的442,936个双端点可观测正连接被逐项精确复现。

失败只发生在预注册的稠密全局slot复杂度门。fit单行最多19个物理连接簇，按冻结25%余量需要24个slot，只能从候选`4/8/16/32`选择`K=32`；C07最大20且零overflow。32槽endpoint-to-cluster head在固定32基元合同上需1,005,969,408个输出值，高于独立pair head的974,532,864个，压缩比只有0.96875；按真实active人口，cluster assignment候选256,051,648，也约为pair候选64,686,160的3.96倍。它把pair组合爆炸转移到了global cluster slots，按预注册规则判`STOP_PRIMITIVE_CONNECTION_HYPERGRAPH_CANDIDATE`，不得缩减余量、挑K=16或把该run改写成PASS。

该结果不否定连接簇物理语义，反而正式证明其唯一性和clique闭合；否定的是固定K稠密查询实现。下一候选收窄为`Learned Composition Anchor Field`：每个可见基元端点只预测一个当前传感器坐标系中的共享构造锚点及不确定性，输出规模随端点线性增长；同一物理节点的端点通过共享概率锚点形成连接簇，隐藏/冲突锚点拒绝提交，不使用固定距离/角度手写连接。实现前先做零训练Teacher/readiness，验证每端点锚点唯一、坐标变换因果、同簇一致、异簇特别是stacked tunnel可分、非学习端点基线和线性输出合同。若锚点Teacher不可观测或不可分，则停止该候选，不训练。

正式run=`results/gate3_semantics/gate3_20260902_primitive_connection_hypergraph_teacher_feasibility_v1_seed0`；23项seal SHA-256=`61e6ce19bbcae565f2b2b0d0f678917acf6ee5cc1388c5780d428c79cdec3e87`。该run为有效预注册科学FAIL；outer runner因executor按科学FAIL返回2而未把inner summary复制到outer字段，但完整inner summary、per-task表、图和seal均已保存，不影响停止结论。

# 2026-09-03：O(E) composition-anchor Teacher readiness正式PASS，先验证预测几何基线再训练

新的零训练readiness完整处理fit/C07共70 parents、210 tasks和491,196行，只读取构造JSON、当前sensor pose/yaw、P1b紧凑Teacher和endpoint-observability；LiDAR range、模型前向、optimizer、C08/C09/C10、graph和M-TARE均为0。每个primitive endpoint都恰好属于一个composition operation；同一物理连接的anchor在当前sensor坐标中最大距离为0，最近不同连接及overlap hard negative anchor仍相隔0.809503 m，证明共享锚点标签唯一且不会把stacked/nonincident tunnel合并。

以Teacher裁剪轴端点距离作为完全非学习的可观测性上界，在fit按precision>=0.98选出的唯一阈值为0.499973 m，fit达到P/R/F1=1/1/1；阈值原样迁移C07后TP=442,606、FP=0、FN=330，precision=1、recall=0.999255、F1=0.999627，899,848个overlap hard-negative pairs中误连为0。该结果证明局部LiDAR支持的几何端点包含极强连接信号；它同时构成论文方法风险：在Teacher真值基元上，关系几乎可由距离直接恢复，不能把额外relation learning的存在本身当作贡献。

composition-anchor输出固定为每端点`xyz+uncertainty`，32基元时每行256值，相比独立pair head的1,984值减少7.75倍；pairwise shared-anchor probability只在输出后确定性计算，不再学习O(E²)独立logits。正式run状态COMPLETED、error=null、全部检查PASS，耗时77.11s、peak RSS=2,508,700 KiB、source unchanged；27项seal SHA-256=`4e48e809d5462cd98fb9b811746a7dbd0b831b06f06fa80f789aee1d4b17e1d8`。

下一步在实现或训练anchor head前，先对三个冻结observable-relation checkpoint做一次C07-only、零optimizer的预测几何关联诊断：分别在deployed proposals和proposal-oracle matched slots上，用预测axis endpoint距离形成相同的非学习关系分数，与现有learned pair score比较安全precision/recall。如果预测几何本身已经安全恢复连接，则anchor head没有独立必要性，论文应把关系形成定位为学习几何的确定性组合并重新收窄贡献；如果只有Teacher几何可分而预测几何失败，则实现O(E) anchor residual/uncertainty head有直接证据。该诊断前仍不读C08、不训练、不建图。
