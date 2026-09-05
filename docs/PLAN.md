# 地下结构拓扑多机器人探索：最终执行计划 V1

## 2026-09-05 最新权威覆盖：用户批准几何组合学习与因果结构图实施

19:40软件推进覆盖：多实例中心query小头已实现（26,826参数，全部64候选，无GT中心/候选过滤），未知/存在/Hungarian平局及旋转临界反例已补。截面正证据去重核与Open3D实际局部配准/扫描位姿绑定已完成合成测试，均不代表真实结构标签、完整图或科学PASS。当前确切缺口是有效局部事件/成员标签的操作定义与原180/900对应，不是再扩大几何训练。不得把没有现成region外壳字段解释为不能学习，也不新增任意region多面体作为前置；现有形状/相对关系监督可以复用，但完整构造incidence若作为潜在结构监督属于标签语义改变，不能静默替代已批准的可观测目标。下一动作优先完成该标签边界和精确pilot准备；图软件可并行补持久节点坐标、端口及轨迹证据接线。详见`GSE_REGION_QUERY_SOFTWARE_V1.md`与教师草稿第10节。当前无真实训练/正式图进程。

19时后执行更新：GPU同预算坐标对照已完成，原C01的180观察/900帧，两分支各540步，30.3838秒。raw/mean坐标MAE1.779605/2.457090m，10/10父地图raw更好；但横向2.335539/1.959483m、方向16.846121/16.321092度，属于混合指标证据，不扩大训练或宣布几何科学PASS。39项seal复核、11图全部保留，见`GSE_COORDINATE_CONTROL_V1.md`。下文“GPU对照准备”为历史。当前下一动作是局部结构监督草稿中的区域/开口合成合同与原180/900精确pilot准备，同时继续CPU图接口接线；现有构造附件不能冒充可观察端口标签，未通过标签合同不启动结构训练。多实例中心query只是明确推荐草稿，未实施/训练；原180只有mixed，不能伪造跨几何配对消融。GPU仅执行已经有有效输入合同的任务，不为占卡扩张实验。

2026-09-05 19时并行收敛覆盖：执行`GSE_GRAPH_CONVERGENCE_PARALLEL_V2.md`，用户明确加速、多线并行、优先5090。当前唯一研究问题仍为几何组合是否改善局部结构/端口与图；允许同一问题下GPU几何对照、教师/组合准备、图软件三条依赖支线并行，不再把几何精度全部过门作为受控结构实验的前置。数据/教师/训练各自精确card与冻结不省略，正式GPUrun单队列，运行期间源不可改。

C02冻结推理已完成20.068515s，180观察/900帧/1489片段，raw/old坐标MAE2.031974/3.922121m，10/10父地图更好；横向2.643606/2.765252m，4271多余候选未检测惩罚，非结构/泛化PASS。28seal复核，309项回归通过，详见`GSE_HEAD_DEVELOPMENT_INFERENCE_V1.md`。下文“C02推理待执行”为历史。下一GPU项原C01同180、raw-v-mean坐标同预算540步各分支，独立卡后执行；offset停止不变。教师线完成有效标签后直接进入GT/预测/无组合/无配对约束短实验，不能GT筛成员。图线目前仅软件，无真实图/闭环。C07--C10不自动开放。

2026-09-05 C02元数据更新：预登记中点分位选样已单次完成，10父地图/180观察/900唯一帧/1489可见片段/180有向穿越；0几何坐标/扫描/模型读取，0.120676s，13项seal复核。样本不按标签或分数选取，实际全人口23878序列；帧索引间隔不等于已测物理距离。当前唯一下一动作是据此精确清单冻结raw小头与旧模型的C02开发推理卡/规格并验证执行器：每模型180主观察+首父18重复，0optimizer，既有轴线目标仅评分。元数据卡不能授权推理，必须用新独立data_export卡、两checkpoint哈希、预检和不可覆盖run。它仍是小头未参加拟合的父地图开发检查，不是骨干未见测试；C07--C10/事件训练/图/闭环不开放，偏移训练停止不变。

2026-09-05 观察对应诊断更新：唯一V1R在2.966s完成相同180缓存输出的固定父地图内shift9对照，raw正确/错配MAE1.779605/5.219674m、横向2.335539/6.530963m、无向角16.846121/34.271080°，10/10父地图正确对应更好。输出不是完全固定候选库，但不能区分传感器/运动相关性或记忆，仍非检测/泛化PASS；4308个剩余候选未惩罚，偏移分支STOP不变。V1的JSON对象/列表读取错误在0评分处停止，V1R只修该接口；完整执行合成回归及446项指定回归通过，22seal复核。当前唯一下一动作：C02十个父地图的精确封存元数据范围与小头冻结推理采样/card准备，只读已声明元数据、不先读数组或运行模型。C02必须称“小头未参与拟合的开发父地图”，旧骨干已经训练C01--C06，不能称整模型严格未见。先固定人口，再检查raw拟合改善能否迁移，不扩大偏移训练。C07--C10、事件训练、图与闭环仍关闭，科学Phase3未升级。详见`GSE_AXIS_OBSERVATION_DEPENDENCE_V1.md`；下文缓存对照准备已完成。

2026-09-05 固定预算训练更新：同180观察的两分支1080步已完成，15.977s；无偏移raw坐标MAE1.779605m，加slot偏移2.825677m，旧冻结模型同几何matcher3.611578m。偏移版10/10父地图都不如raw，维持STOP_BEFORE_EXPANSION，不加轮数或三种子。原run因最后parents字段覆盖绘图失败，权重/预测/日志保留；独立零训练补全V1R已恢复全部180图和原判定，22项seal复核。补全V1的目录初始化错误及原run均保持FAILED，不重写历史。当前唯一下一动作是已缓存预测的输入依赖与布局检查规格/软件：固定同父地图错配对照，区分观测相关恢复与多候选匹配收益，并报告方向/横向及剩余候选；不新训、不自动以raw候选成功替代失败分支。完整事件/端口监督、C07--C10和闭环仍关闭。详见`GSE_POINT_AXIS_PROBE_V1.md`。下文16:48“准备小训练”已完成；未修改科学验收门。

2026-09-05 16:48更新：最小点级轴线读出及冻结骨干adapter已实现，35,460参数。正式软件run 48/48 PASS、2.120s、RSS约0.95GiB，11项seal复核；包含实际骨干类的57,600点/32slot随机权重前向反向，0真实数据/checkpoint/optimizer。多源回波风险在生成teacher前发现并处理为slot条件偏移，不把源归属强制改单标签。当前唯一下一动作是原180 C01观察/900帧的独立几何小训练卡、对照规格和执行器：新点级偏移对无偏移raw读出，同输入同预算，旧冻结几何作参考；先核对指定冻结seed0特征复用、现有P1b几何监督和小批资源。尚未运行新训练，精确预算/验收必须先冻结。这里只开放几何前端小实验准备，不需要用未验证的事件标签训练；完整事件/端口监督、C07--C10和闭环仍关闭。详见`GSE_POINT_AXIS_READOUT_V1.md`；下文16:26软件readiness是已完成历史，Phase 3科学门未升级。

2026-09-05 16:26更新：同180观察/900原始帧的坐标支持对照在16.254s完成。4356控制点中旧均值池3153个被证明不可表达；其中2688个在原始点池有非负重构见证，但465个连原始点凸包也在外。27项seal复核，0模型/optimizer/新标签。结论是旧读出有真实表达限制，不是raw池替换已学好或全部误差已解释。当前唯一下一动作改为最小原XYZ旁路/点级归属与表面到轴线读出的软件规格和合成验证；不能把旧token权重广播到原点冒充新表达能力，不能继续仅靠回波凸组合表达所有轴点。先验证点级区分、凸包外轴线及梯度/退化，再冻结同180数据的独立短训练卡；不直接重训、不修改教师或阈值。完整节点/端口监督仍未通过。详见`GSE_COORDINATE_SUPPORT_AUDIT_V1.md`。Phase 3、C07--C10和闭环边界不变；下文15:53“坐标支持对照”为已完成历史。

2026-09-05 15:53更新：同180行几何分解正式完成，旧行/匹配/分数精确复现；横向RMS中位2.263m、无向角中位37.933°、有限两段折线双向距离中位5.201m，不能再解释为仅裁剪范围问题。冻结预测直接接组合头的训练继续停止。当前唯一下一动作是同180观察的原始坐标池可表达性软件/精确卡/spec：比较旧900均值点池与未合并返回点池，0模型/optimizer/新标签；原900源帧的range/valid读取必须独立卡，旧metadata-only卡不能授权。现有池化已由纯合成反例证明存在压平上下层的可表达性限制，但未证明该限制导致此次真实错误；先量化再薄改几何读出，不推倒骨干或自动重训。详见`GSE_AXIS_ERROR_DECOMPOSITION_V1.md`。C07--C10和闭环未开放；下文15:33“误差分解”为已完成历史。

2026-09-05 15:33更新：同180行现有教师/缓存对照已完成（`gse_local_teacher_audit_v1`），旧数据支持与全部可见构造连接一致，但预测裁剪轴点平均欧氏误差10.882m，不能据此直接训练组合头，也不能把该值当横向定位误差。全部180行含非当前节点片段，178行有角投影重叠；后者不是物理碰撞/连接。当前唯一下一动作改为同180行、零推理/零训练的轴向/横向/方向/裁剪长度误差分解，保留全部样本和原分数，另冻结精确只读卡/spec。以此区分前端几何失真和裁剪/接口问题；不增加训练规模、不自动创建事件标签、不读取C07--C10或闭环。下文“现有教师核对”为已完成历史，详见`GSE_LOCAL_TEACHER_AUDIT_V1.md`。

用户完整提交并要求实施 `docs/GSE_GRAPH_COMPOSITION_EXECUTION_PLAN_V1.md`，该文件现在是方法与阶段执行依据。保留几何基元主线；组合学习负责局部事件/端口，地点合并使用部署可得几何验证，首次稳定节点不等待重访，实际运动依次建立连接。

当前 Phase 3 首批仅合成组合接口与因果状态机软件测试（0 世界/数据集帧/训练），然后核对 180 观察的共同坐标与端口标签。阶段升级受用户方案的科学验收约束；常规步骤沿用持续授权，不再等待方法选项回复。完整训练、测试世界与闭环尚未开放。

2026-09-05 实施更新：软件内核已完成首批验证，同 180 观察的只读元数据核对也已完成。当前唯一下一步是节点级可观测监督合同与相同行字段恢复 readiness；不得把旧物理端点支持掩码当作节点/端口可见性。40 个分叉观察的全部 incident 基元均存在，但仅 6 个满足全部旧端点支持；这不等于其余 34 个分叉不可观。新的可见几何输入桥不依赖物理端点分类分数，旧桥与旧阈值保留作比较。节点标签、真实扫描补充和训练仍须各自精确 Data Card/spec；本轮库存卡只允许元数据核对。主机 RTX 5090 D 已可查询，先前 CUDA 不可用是沙箱访问限制，不能继续记为硬件故障。

覆盖旧“等待方法边界批准”“局部短训失败即证明理论不可观”“loss 降 95% 为通用门”“C08 可重新作为严格未见”的要求/解释；旧 run、指标和 seal 不变。C08/C09 历史开发暴露必须披露，C10 当前资产读取仍为0。新标准仅适用于新 run。

2026-09-05 后续：通道口正射线证据与精确模型输入 reader 已完成合成验证；不等于节点教师通过。唯一下一动作是为同 180 观察冻结共同坐标字段恢复 Data Card/spec，使用指定冻结 seed0、0 optimizer，核对旧缓存一致性；元数据库存卡不能授权此导出。详见 `GSE_NODE_OBSERVABILITY_READINESS_V1.md`。节点/端口有效标签未完成前不训练。用户允许按任务合理分工，不再指定简单任务必须由 Luna 承担。

2026-09-05 字段恢复完成：唯一正式 run 在 6.137s 内完成 180 主推理和 18 重复；900 唯一帧、旧特征/置信度逐元素一致、六项新字段重复一致、权重不变、0 optimizer/新标签。23 项 seal 已复核。上述“冻结字段恢复卡/spec”为已完成历史；当前唯一下一动作是冻结同 180 观察的现有教师几何/组合/歧义元数据核对，结合已恢复预测定义可监督的局部成员与 UNKNOWN，不自动把 degree 或物理端点掩码当事件教师。禁止因组件 PASS 扩大训练、读取 C07--C10 或启动闭环。


## 2026-09-04 最新执行覆盖：双读出tiny corrective仍FAIL，停止局部几何单独识别地点

V1失败后预注册的唯一corrective已使用完全相同的密封C01缓存，将结构几何回归和地点关联embedding拆开，仍只训练180观察、100节点、80正对和500步。V1R预检因Data Card用汇总world别名而未通过，没有创建run；V1R1仅将S01--S10逐世界列出，方法和数据不变，预检通过后正式执行。

V1R1在`8.20 s`内系统完整完成但科学FAIL：总loss reduction=`0.56397`，几何RMSE=`0.10883`，degree accuracy=`0.97778`，precision>=0.99的非空关联仍不存在。关联embedding虽将不同节点距离中位数拉到`1.15288`，但仍有不同物理节点距离精确为0；在重复管廊几何中，局部外观不能唯一决定地点身份。

因此按失败政策停止“冻结基元输出+小读出单独学习place identity”接口，不进入单seed或三seed长训练。当前需要的方法边界决策是：保留已证明可学的三维基元几何和结构事件，用它们直接决定节点生成与端口属性；地点合并改为学习描述候选+里程计/图邻接/执行一致性的联合裁决，不再要求局部几何独立解决不可观的全局地点身份。这是实质方法调整，必须在新规格冻结后才能继续实验。

## 2026-09-04 最新执行覆盖：tiny overfit快速FAIL，只允许分离几何回归与地点关联头

家族平衡的C01 tiny overfit已在`12.24 s`内完成，不是长训练或系统故障。固定人口为S01--S10各18个五帧观察，共180观察、100节点和80个同节点正对；只从密封seed0几何基元骨干生成`180×64×44`特征，C07--C10、graph replay和M-TARE读取均为0。

当前聚合头节点度数准确率达`0.99444`，证明底层基元包含terminal/corridor/junction结构信号；但总loss只降`69.45%<95%`，几何RMSE=`0.14321>0.01`，且precision>=0.99时无任何安全同节点接受。源特征中同节点/不同节点距离中位数为`0.22881/0.22244`，直接距离不可分；而同一tiny人口的Teacher描述可达precision/recall=`1.0/0.9625`，所以是学生读出接口失败，不是基元组合目标无解。

当前唯一允许修正是：冻结数据、骨干、500步和验收线，仅把一个44维多任务读出拆为（a）按维固定归一化的结构几何回归头；（b）独立的关联embedding头。合并仍必须通过已冻结的16 m图一致性候选门，真实穿越仍是edge提交的唯一来源。先重做同规模corrective tiny overfit；若仍失败，停止该学生接口，不开启单seed或三seed长训练。

## 2026-09-04 最新执行覆盖：图一致性快速门PASS，允许进入tiny overfit

正式零训练归因复现V1R合同失败后，使用历史已冻结的16 m图位置候选门重新裁决同一节点描述。C07 descriptor-only为TP/FP=`2960/53`、precision/recall=`0.982410/0.933459`；16 m门后为`2960/0`、`1.0/0.933459`，召回不损失。每节点独立1 m最坏定位误差仍为0 FP；S01--S10全部保留真关联。fit从TP/FP=`18266/370`改善为`18266/33`，precision=`0.998197`。53个原FP最小真实间距为`19.1678 m`，完整清单已保存。

这只证明节点级几何证据加图候选机制在Teacher/解析几何上具备容量，不代表LiDAR学生模型已经成功。当前漏斗进入下一层：tiny overfit必须在极小、家庭平衡的C01--C06子集上证明网络能从因果LiDAR/冻结基元输出恢复该44维节点证据，并在同批正负节点对上快速逼近Teacher；若数百步内不能显著下降并过拟合，则在任何完整seed训练前停止。C08--C10和M-TARE继续关闭。

## 2026-09-04 最新执行覆盖：节点级快速证伪找到有效表示，先修合同再加图一致性

`Structural Node Evidence Funnel`在不训练、不推理、不读C08--C10的条件下，于约88秒完成fit/C07全量节点对审计。完整基元组合oracle在C07得到precision/recall=`1.0/1.0`；修正“ray support只是不确定性、不是地点身份”后，五帧因果节点描述在C07得到precision=`0.982410`、recall=`0.933459`，十个结构family均有真阳性。这证明“基元组合形成节点描述”具有强可分信号，显著优于已停止的端点相似度路线。

但V1R执行器漏实现预注册的accepted-pair false fraction `<=0.01`门：实际为`53/3013=0.017590`；同时fit原始36,318个traversal-realization中有6个因物理行程太短而没有五帧序列，有效观察应为36,312。故V1R只能记为`FORMAL_MACHINE_PASS_RESEARCH_CONTRACT_FAIL`，不得作为论文PASS。当前唯一允许步骤是零训练合同重评分和53个C07误合并的图一致性归因；复用既有16 m候选门、已验证邻接/执行轨迹一致性与不确定性拒绝，仍在C01--C07选择/验证，先把误接受压到1%以下，再进入小样本过拟合。禁止因此直接启动长训练。

## 2026-09-04 最新执行覆盖：端点相似度0/3正式FAIL，改用分级快速证伪

无槽位的endpoint relation metric已完成唯一三seed正式run和完整C07评价，而不是仍在训练。三个seed的best attachment F1分别为`0.252722/0.163434/0.280575`，相对非学习基线`0.006380`均有真实学习增益；但对应precision只有`0.173568/0.118684/0.198213`，三个seed在precision`>=0.98`时安全TP全部为0，最终`0/3`通过。正式决定为`STOP_ENDPOINT_RELATION_METRIC_BEFORE_C08_AND_GRAPH`。这说明局部端点embedding可以粗排关系，但不能作为节点合并或边提交依据；现有模型只保留为辅助特征和失败消融。C08、在线图和M-TARE继续关闭。

当前唯一允许步骤改为C01--C07零训练的`Structural Node Evidence Funnel`：复用sealed Teacher、冻结模型和已有缓存，把同一物理结构的多端点、多帧几何组合先聚合成节点级观测，再分别审计结构节点产生、节点重访关联和困难平行/叠置隧道拒绝。学生候选只可使用因果LiDAR导出的基元几何、五帧时序、描述子与不确定性；Teacher identity只用于事后评分，真实穿越仍是未来edge提交的唯一来源。人口固定为C01--C06 `426,552`和C07 `64,644`序列，C08--C10为0读取。该步骤必须在训练前给出高精度非空可分区域、family覆盖、五帧相对单帧增益和无identity泄漏；否则停止新的节点训练。

此后所有新候选采用固定漏斗：零训练可分性审计 → 小样本过拟合 → 单seed短训练 → 单seed完整C07 → 三seed正式run → C08一次迁移。三seed只用于确认已经通过前四级的方法，不再用于探索未确认假设。

## 2026-09-04 最新执行覆盖：端点关系度量readiness PASS，进入三seed训练实现

无任意槽编号的endpoint relation metric已在固定C01真实batch128上正式PASS：`426,818`个head参数/34张量全部梯度有限非零，`2,635,631`参数backbone冻结；primitive permutation、sensor yaw、重复、精确对称、single-threshold complete-link和16/4 GiB资源合同全部通过。当前唯一允许步骤是建立并冻结三seed训练/evaluator：C01--C06完整`426,552`序列提供梯度，C07完整`64,644`序列选择checkpoint和唯一关系阈值，在同一`442,936`真连接人口上至少2/3 seed同时满足相对非学习F1增益`>=0.05`、precision`>=0.98`且TP>0，并报告overlap FP及complete-link簇。C08、图和M-TARE在该门PASS前继续关闭。

## 2026-09-04 最新执行覆盖：停止任意局部槽，转向端点关系度量readiness

正式C07-only因子归因已在三个冻结checkpoint上完整PASS并精确复现源指标。真实连接端点的dustbin argmax比例为`72.12%/86.82%/90.49%`，Hungarian对齐后的best-slot准确率为`46.15%/44.16%/48.02%`；只保留硬槽身份时precision仅`21.50%/17.89%/21.93%`，soft affinity也只有`1/3` seed出现任意precision>=0.98非空尾部。因此`endpoint -> arbitrary 32 slots/dustbin`主路线停止，不能用去margin、改阈值或soft score直接进入图。当前唯一允许步骤改为零训练的端点关系度量readiness：复用冻结几何骨干，以同一Teacher composition clique做监督对比正例，以不同clique、dustbin和disconnected-overlap做困难负例，验证无任意槽编号的endpoint embedding/pair metric、保守transitive clustering、梯度、置换/yaw、资源和无泄漏合同。通过readiness前不训练、不读C08、不建图。

## 2026-09-04 最新执行覆盖：局部组合槽三seed失败归因

当前仍为Phase 3/P2。32槽局部组合Teacher与零训练readiness保持PASS；正式三seedhead-only训练已完成全部`30,699`次更新，冻结C07评估只通过`1/3`个seed，未达到`2/3`门，因此C08、结构图和M-TARE继续关闭。当前唯一允许步骤是对三个冻结checkpoint执行C07-only、零训练的分数因子归因，区分槽分配、dustbin竞争、slot presence、primitive existence、endpoint evidence、entropy和连乘校准中的首要瓶颈，并复核所有阈值均不拆分同分数。归因不得修改模型、数据、Teacher、门槛或读取C08；只有归因给出跨seed一致且能被独立消融验证的最小修正，才允许另立readiness。下方早于本节的“P1导出/三seed训练是下一步”等状态均为已完成历史，不再是当前动作。

## 2026-08-29 最新权威覆盖：程序构造监督的几何基元关系结构图

用户已将论文主方法修正为：从程序化地图底层的截面/扫掠基元、SE(3)变换和组合关系出发，训练模型从五帧因果LiDAR恢复显式几何基元及其关系，并由学习关系形成结构语义图谱和执行验证的在线拓扑图。完整决策见`docs/PRIMITIVE_RELATION_STRUCTURAL_GRAPH_PLAN_V1.md`，该文件与本节共同覆盖下方旧的Composer、事件分类、RouteGeometryProfile和ERCSS候选指令；旧内容只保留为基线、消融和失败证据。

当前仍为Phase 3。P0构造监督、三形状合同、80-parent inventory、高速身份射线后端、有限端盖最小姿态修复和32槽Teacher容量均已正式PASS。全量姿态资格只修改`134/252,430`个实际越界端点帧，三变体`757,290`次检查越界为0；容量审计在C01--C06得到最大16、25%余量20，故从`8/16/32`冻结32槽，C07/C08最大`23/19`且零overflow。旧8槽方法接口被该正式证据取代。单一研究问题是：程序构造监督的显式基元与关系学习能否在不读取完整地图身份或未来信息的情况下，改善未见拓扑上的几何恢复与离线图结构。当前唯一允许步骤是按冻结split与32槽合同建立P1数据Data Card，并用已通过的闭合基元CSG后端导出三种配对几何实现的757,290帧/564,378个五帧序列及逐回波primitive provenance；在P1完整性、因果性、无泄漏和资源合同PASS前不得训练、建图、读取C09/C10或运行M-TARE。

状态：`AUTHORITATIVE`  
用户最近确认日期：2026-08-20  
适用范围：用户已明确把目标扩展为完成整篇论文流程。2026-08-23 起，文末第 22 节和 `docs/GSE_GRAPH_RESEARCH_PLAN_V1.md` 是最新权威路线；旧 Phase/Gate 内容只保留历史、基线和治理证据。每个 material run 仍须独立 Data Card、run spec、preflight 与不可覆盖证据；用户已授予既定 GSE-Graph 论文范围内的持续执行授权，不再逐次索取常规批准。C10 在方法、校准和阈值冻结前继续隔离。

本文件是项目当前最高优先级执行方针。每次实现、实验、清理、数据生成或训练前必须先读本文件与 `docs/PROGRESS.md`。旧文档保留研究细节和历史证据；发生冲突时以本文件为准，并在决策日志中记录，不得自行拼接两条路线。

## 1. 最终目标

```text
LiDAR
→ 学习局部可通行结构
→ 在线 Topometric Graph
→ 多机器人全局探索目标
→ M-TARE 原 Local Planner
```

核心问题：能否利用程序化地下环境天然提供的 topology ground truth，从局部 LiDAR 学到稳定结构表示，并利用在线结构拓扑图改善 M-TARE 多机器人全局探索？

训练与最终比较必须分离：

```text
训练：Cano procedural TNG → native perception mesh → CPU Raycast LiDAR + GT → Structural Model
测试：M-TARE Gazebo → LiDAR → Model → Online Topometric Graph
      → Global Target → M-TARE Local Planner
```

最终再部署真实机器人。M-TARE 原对比地图禁止进入训练、验证、归一化、阈值和 checkpoint 选择。

## 2. 当前禁止事项

- GNN/RL 全局策略、模糊控制、ghost prediction；
- learned communication、end-to-end navigation；
- 替换 M-TARE 的 local planner、避障、状态估计、terrain、控制或传感器公平合同；
- 自定义复杂结构类别和未经 baseline 证明的复杂 loss；
- 未通过当前 Phase 就批量进入下一 Phase。

第一版只证明：LiDAR 能识别局部 outgoing traversable branches，并能明确对应 GT TNG。Phase 3 失败时，禁止开始在线拓扑和 M-TARE 集成。

## 3. 工具与方法身份

| 模块 | 身份 |
|---|---|
| Cano generator | 可复现 baseline 数据生成工具，不是创新 |
| Open3D CPU Raycast | Phase 1--3 主合成 LiDAR 与可重复采样平台 |
| Isaac Sim | 可选后验传感器域实验；不是主线前置条件 |
| Structural Encoder | 核心学习模块 |
| Topometric Graph | 核心在线世界表示 |
| Global Topological Explorer | 核心规划方法 |
| M-TARE | baseline 与保留的 local planner |
| Multi-robot system | 最终验证 |

论文主线固定为：Learn structural-topological representations from procedural topology supervision and use them as a compact global representation for communication-efficient multi-robot underground exploration.

复用原则：能固定来源、环境、seed 和输出并通过合同的公开代码先原样或通过最薄 adapter 使用。当前 Cano 生成器代码可直接复用 TNG、表面点云和 native mesh 生成；项目已有 CPU raycast 与 spline 标签可直接复用。当前 checkout 不含论文出口 CNN 训练实现，故感知网络只能称为 `Cano-like adapted reproduction`，不得冒充原作者代码复现。二次创新只在 baseline 通过后加入 planner-consistent semantics/uncertainty、exit-stub 图、M-TARE 高层替换和多机器人分配。

## 4. Phase 0：工程边界与证据制度

必须维护：

- `docs/PLAN.md`：阶段、输入、输出、方法和验收；
- `docs/PROGRESS.md`：`DONE / FAILED / BLOCKED / NEXT`；
- `results/phaseN/` 或对应 V3 Gate 结果目录：不可覆盖证据。

每个 Phase 完成后必须更新进度、给出运行命令、保存科研可视化和定量指标、报告失败、明确 PASS/FAIL。任何 Phase 未 PASS，不得自行进入下一 Phase。

### 4.1 三类几何门禁不得混用（2026-08-10 修订）

程序化世界的几何资格按用途拆分，不能再用一个“严格 mesh”结论同时替代三类证据：

1. `TOPOLOGY_GT_GATE`：graph、spline、per-tunnel axis、branch/terminal/intersection 对应与重放；
2. `LIDAR_SENSOR_GATE`：固定内部位姿能否产生有限、可复现、无明显穿墙/伪开口且与轴线标签一致的 LiDAR；
3. `DYNAMIC_NAVIGATION_GATE`：机器人 footprint、地面接触、碰撞、低速轨迹跟踪与卡死。

非 watertight、非 manifold 或 self-intersection 仍是必须记录的几何风险，但它们只直接否决严格 collision/navigation 资产；是否否决感知数据，必须由 `LIDAR_SENSOR_GATE` 的任务相关证据决定。感知 mesh 与未来 collision mesh 可以是两个明确记录 provenance 的资产，禁止把其中一个的 PASS 冒充另一个。

正式数据前允许一种特殊的 `sensor_smoke`：只使用 1 个既有失败证据 world 和少量固定 pose，split 为 `NONE_DIAGNOSTIC_SMOKE_ONLY`，正式数据集计数仍为 0。它必须绑定专用 smoke card、保存全部样本并禁止训练、调参、换 world 或自动换后端。

### 4.2 可运行纵向链路原型（2026-08-10 用户指令例外）

为避免项目长期只修局部门禁而没有可检查的系统，允许在**一个已有开发 world** 上先实现一次不训练、不闭环、不修改 M-TARE 的接口纵向切片：

```text
CPU LiDAR → 当前帧规则出口 → 因果 Online Topometric Graph
                    ↘ spline-oracle 上界（仅作接口对照）
```

该例外只允许验证数据结构、因果更新、节点/边/exit-stub 表达、指标和可视化是否能完整工作，不代表提前通过 Phase 3/4。规则和图关联参数如果在同一轨迹上选择，必须标为 development tuning；其结果不得用于 unseen-topology、学习模型、规划收益或论文性能声明。正式路线仍按 Phase 1 五拓扑合同、Phase 2 数据、Phase 3 学习感知逐步推进。

## 5. Phase 1：Cano 程序化地图与 GT TNG

指定来源：`https://github.com/LorenzoCanoAn/procedural-subt-gen`。当前核验默认分支 `refactor`，固定候选 commit：`b6c77621187404b4dfab1249c7a1b40f63ad9ab3`。

优先阅读并原样跑通：

```text
scripts/snippet_2.py
src/subt_proc_gen/tunnel.py
src/subt_proc_gen/mesh_generation.py
```

第一阶段不修改原算法。若导出不完整，只在本项目写 adapter，不修改第三方核心。第三方仓库当前未在顶层显示 LICENSE：允许先做隔离、本机、原样可执行性审计；在许可证或作者许可明确前，不把其源码复制进 `src/`、不修改后再发布、不声称已获开源授权。

标准 world bundle：

```text
dataset/phase1_cano/world_000/
  mesh.*
  graph.json
  splines.*
  metadata.json
```

`graph.json` 至少包含 node id/xyz、edge、intersection、tunnel id。`metadata.json` 必须包含固定 commit、依赖、seed、生成参数、文件 hash 和 adapter 版本。

执行顺序：

1. 固定源码 commit、依赖和许可证状态；
2. 原样运行 `snippet_2.py`；
3. 生成 1 个 world smoke 并核验原始 TNG/point cloud/mesh；
4. 实现只读 adapter 和标准 bundle；
5. 固定 seed 重放并比较 hash；
6. 先做 5-world contract pilot，不训练；
7. pilot PASS 后再生成 100 个 topology parent，固定为 80 train / 10 validation / 10 test；
8. test topology 在训练、调参和人工开发审阅中隔离。

当前检查点（2026-08-11）：单 world 24-pose CPU 合同、selector v2、五拓扑 750-view CPU contract 和 100-parent V2R recipe split 均已 PASS。V2 以 bounded resampling 完成 120/120 generation/replay；V2R 只读固定 C01--C10 后得到 100 个互异 parent 和 80/10/10 split，actual cycle rank 作为 GT 而非 connector 数的同义词。正式 mesh、数据和模型仍为 0。下一步只设计 100-parent perception-mesh proposal，获批前不生成 mesh、不采 LiDAR、不训练。

Phase 1 最终验收：100 个不同 topology、10 张带 provenance 的 TNG/mesh 对照图、标准 graph.json、自动脚本、固定 seed 重放，以及 graph/mesh 连通、闭环和分支一致性证明。训练感知资产必须通过 `TOPOLOGY_GT_GATE + LIDAR_SENSOR_GATE`；动态导航资产另需 `DYNAMIC_NAVIGATION_GATE`。证据位于 `results/phase1/` 或当前治理映射的不可覆盖 Gate 结果目录。未达到 100 个合格 topology 不得宣称 Phase 1 PASS。

当前 clean-room `g000/g001` 只作生成/Isaac 合同与失败知识，不替代 Cano Phase 1 数据，也不计入 100 个 topology。

## 6. Phase 2：CPU Raycast LiDAR 与客观标签

固定链路：

```text
Cano TNG/graph/splines → audited native perception mesh
→ Open3D CPU RaycastingScene → 16×720 range + valid mask + ray origin
```

CPU backend 不生成拓扑，只按冻结 pose/ray origin 对 native perception mesh 发射理想射线。第一合同固定 16 个 elevation、720 个 azimuth、0.3--50 m、first return、无噪声；每个 sample 必须关联 robot pose、ray origin、nearest tunnel axis、当前 outgoing branches 和 provenance。完整 TNG/mesh 只用于 teacher 与审计，禁止进入学生输入；visited/unvisited 只按因果轨迹状态计算。

正式数据前依次通过：单图 24-pose CPU LiDAR contract、5-topology contract pilot、固定 pose 的 CPU synthetic ↔ Gazebo LiDAR parity。噪声、dropout、motion distortion 或 domain randomization 只能在 parity 证据后另立消融。若 Cano native mesh 的 `LIDAR_SENSOR_GATE` 失败，才触发同 pose 的 project navigation-grade geometry A/B；不得提前重写可用 baseline。Isaac 只保留为未来可选额外传感器域实验。

第一标签固定为 360° outgoing-exit / traversable-direction prediction，不先设计复杂 latent loss。第一输入优先选择最快跑通的 range/depth representation；已有 LocalStructuralMap 只在 baseline 后作为消融候选。

先做 smoke 和 Data Card；确认数据/标签有效后生成 100k samples，再根据学习曲线扩展到 500k+。样本规模同时报告 topology、trajectory、独立 place、结构事件和帧数，禁止相邻帧冒充独立样本。

## 7. Phase 3：Structural Perception Baseline

第一模型固定为小 CNN 或 ResNet18 量级：

```text
LiDAR range/depth → encoder → structural feature z → exit head
```

必须比较 geometry/raycast rule baseline 与 learned model。指标固定为 exit precision、recall、F1 和 angular error，并按 unseen topology、width、roughness、curvature 分层报告。

Phase 3 PASS 条件：在严格未见 topology 上明显优于随机，并达到可用于后续在线图构建的 precision/recall；预测必须能回溯到 GT TNG。失败时只允许检查数据、标签、输入和 baseline，不修改 M-TARE、不引入复杂 planner。

## 8. Phase 4--9（2026-08-20 起按 Gate 逐级实施）

- Phase 4：可解释 `OnlineTopometricGraph`；节点保留 xyz anchor、feature、branches、visited 和 confidence；简单 distance + structure association；与 GT 比 node/edge P/R 和 connectivity。
- Phase 5：单机器人 rule-based topological frontier；比较 M-TARE original、Rule-based Topological、GT-TNG Oracle。Oracle 无法达到 M-TARE 时停止路线。
- Phase 6：只预测 branch structural potential，不预测完整未知地图；标签由完整 GT TNG 的 future junction/graph length/reachable topology 自动计算。
- Phase 7：单机器人稳定后才做多机器人；同步 nodes/edges/visited/potential/target；Hungarian 或 greedy；不学习通信协议。
- Phase 8：1/2/3/4 robots，多地图多 seed；比较 Original、Rule-based、Ours、Oracle；报告 completion time、coverage-time AUC、distance、redundancy、latency、bytes、success 和结构图指标。
- Phase 9：程序化 CPU LiDAR 训练、M-TARE Gazebo 正式测试，不在 Gazebo 大规模 fine-tune；Isaac 只作可选额外域验证；最后从 2 台真实机器人扩展。

### 8.1 2026-08-13 Phase 4 执行授权

用户在审阅 `docs/PHASE4_OFFLINE_TOPOMETRIC_GRAPH_PROPOSAL_V1.md` 的数据、方法、基线、成本和停止边界后回复“继续”，明确解除原 Phase 3 实施上限。当前只授权 Phase 4 的 C08 离线开发步骤；C09 冻结验证需开发参数收口后另行确认，C10、M-TARE、shadow、closed-loop 和全局目标选择仍禁止。

Gate 3 显式语义不另行重复训练：冻结 M1D 的 direction/count/role 三头及其 C09 验证证据已经完成该语义合同。该证据映射不允许在 Gate 4 修改 checkpoint、语义阈值、loss 或数据 split。

Phase 4 当前唯一问题、精确 3-C08 数据范围、主方法、B0 baseline、GT oracle、指标和证据要求以 `docs/PHASE4_OFFLINE_TOPOMETRIC_GRAPH_PROPOSAL_V1.md` 为准。先过连续 spline trajectory 的零 GPU 合同，再执行 C08 development replay；任何 graph/spline 对应、轨迹穿墙、因果性、checkpoint、GT leakage 或 verified-edge 证据问题必须停止报告。

## 7.1 2026-08-12 Phase 3 结构语义主模型修订

V2R 数据集已经客观提供 direction target、branch count 和
`interior/junction/terminal` role。为保持项目中心问题，Phase 3 不得把单一
exit head 冒充完整结构语义。模型与对照固定为：

```text
B0  frozen range-sector geometry rule（不学习）
B1  Cano-like range-image encoder + direction head（对标消融）
M1  range+valid -> shared circular CNN encoder -> 128D z_role
                         |-> 720-bin outgoing-direction head
                         |-> branch-count auxiliary head
                         `-> interior/junction/terminal role head
```

`z_role` 是由客观多任务监督学习的结构作用表示，不是地点身份，也不是
loop-closure descriptor。必须通过同一 place 五视角一致性、跨 world role/objective
linear probe、旋转等变和遮挡稳定性来证明它具有结构信息；不能仅凭存在一个
128D 中间层就宣称“已提取结构特征”。

方向 head 是后续 exit stub 的基础，role head 用于结构事件证据，branch-count
head 只作辅助一致性。训练集中 count=5/6 仅有 11/2 帧，因此第一版不得宣称
1--6 各类均可靠；Gate 判据只覆盖常见 1--4，5--6 单独报告为稀有诊断。

本修订不提前进入建图：Phase 3 只训练并验证局部结构特征/语义；在线
Topometric Graph、全局目标选择和 M-TARE 高层替换仍按 Phase 4 以后执行。

## 9. 历史阶段记录：Phase 1 topology/mesh 起点

本节及第 10--15 节中的“当前唯一任务”均为当时的历史执行指针，已由第 16--19 节及
文末最新权威更新覆盖；不得再把这些历史措辞作为当前执行指令。

当时的 Phase：`PHASE_1_100_TOPOLOGY_PARENT_RECIPE_SPLIT_PASS_MESH_M0_FAIL_M0R_PENDING_APPROVAL`。

当前已有一个可运行但不计 Gate PASS 的工程纵向切片：`results/prototypes/cano_seed0_lidar_to_topometric_vertical_slice_v2_graph_refined/`。它在已有 seed-0 world 上沿完整图的双边遍历采样 240 帧，每帧 16×720 CPU LiDAR；当前帧 range-sector 规则和 spline oracle 分别输入同一因果在线图。规则出口 F1 为 `0.7503`，图节点 F1 为 `0.8276`，直接几何边 F1 为 `0.94`。图关联参数来自同一轨迹 81 组候选，因此这里只证明接口与图构建可以工作，不证明跨拓扑泛化。

五拓扑 CPU contract pilot 已完成：冻结 range rule 在 5 个固定 topology 上的总体 branch F1=`0.9143`，P05 slope-multiheight 最低 F1=`0.8374`；5 mesh、250 anchors、750 static views、17.28M 双场景 rays 和全部 35 张可视化均已封存复核。静态 anchors 没有因果运动边，因此本 pilot 只报告 branch 指标，不伪造 node/edge 在线图指标；纵向切片的图参数仍只属于单开发 world。

第一版 100-parent 候选审计已经执行完并封存于 `results/gate0_baseline/gate0_20260811_cano_100_topology_parent_candidate_audit_v1_seed0/`。它完整尝试 120 个预声明候选，但只有 60 个通过 V1 合同，60 个失败全部发生在 grown tunnel generation；flat/3D 有效率分别为 `20/60` 与 `40/60`。60 个成功结构的 canonical identity 和 coordinate-free WL hash 均为 60/60 唯一，说明生成器有结构多样性，但 V1 的单次参数抽样过严。另有 2 个候选暴露 `cycle_rank >= connector_count` 会接受额外非预期环，V1 结果不得作为最终 split 或训练数据。

V2 已按批准范围完整执行并封存：120/120 topology generation 和 replay 成功，bounded draw 最大只用 5；但 exact-cycle family 合同把 6 个含额外自然环的合法图拒绝，S07 只有 9/12，最终 99 parent/80-10-9，因此仍是 FAIL。该结果证明生成器和抽样方法可用，也证明 requested connector count 不能定义最终图的精确 cycle rank。

V2R 已完成正式只读恢复并冻结 100 parent/80-10-10。M0 在首个 train S01 证明 topology/parameters/axis exact、两个 native mesh 单独质量 PASS，但否定 OBJ 字节重放；原因是 Poisson ordering 后的上游 ±0.2 m/axis vertex noise。当时的历史下一步（已被后续权威更新覆盖）是评审 M0R：保持十个 C01 train parent、seeds、原生方法、20 mesh 和零 validation/test/LiDAR/训练不变，只用理论噪声界定义几何重放。M0R 未批准前不实现或重跑；M1 全 100 仍未授权。M-TARE benchmark worlds继续隔离；CPU↔Gazebo parity 与正式 Data Card 通过前不采 100k LiDAR、不训练 CNN/GNN、不修改 M-TARE。

2026-08-10 已完成的 24-pose sensor smoke 冻结了 1 张 seed-0 Cano world、24 poses（8/8/8）、16×720/50 m profile 和 5 m spline label。24/24 pose、label 与 CPU reference 准备通过，但 Isaac Sim 6.0.1 报自定义 config 未找到，0 个 RTX scan 完成；结果为 `FAIL_CUSTOM_RTX_CONFIG_NOT_REGISTERED`。这次失败没有检验到 Cano mesh 的 RTX 射线能力，正式 dataset/training/model 仍全部为 0。

2026-08-10 v1 已证明本地 `OmniLidar` USDA 创建、69 项 SensorChecker 和精确属性回读通过，但直接 `LidarSensor.get_data` 返回非法 GMO，完整扫描为 0。固定镜像的官方测试使用 Replicator Writer `renderProduct` callback 消费 GMO；完整失败边界和证据偏差见 `docs/ISAAC_RTX_GMO_WRITER_AUDIT_V1.md`。

Writer v2 已按批准范围执行一次并失败：local USDA/checker/read-back 通过，但 300 frame 内 callback=0，shutdown 出现 pending writer schedule drain timeout，完整扫描为 0。该结果不能通过重跑或继续修改 custom probe 规避。

用户批准的 `configs/v3/gate0/isaac_official_lidar_writer_control_v1.json` 已执行一次并停止。实验在 `Lidar.create(config="Example_Rotary")` 阶段失败：该配置映射到远端 `.../Assets/Isaac/6.0/Isaac/Sensors/NVIDIA/Example_Rotary.usda`，而冻结命令为 `--network none`；镜像和主机均无本地副本。因此传感器、Writer 和 300-frame 判据均未实际建立，不能据此分类 runtime 或 custom coupling。

`configs/v3/gate0/isaac_official_lidar_asset_freeze_v1.json` 已执行 PASS：精确 NVIDIA URL 只请求 1 次，HTTP 200、0 redirect，得到 15,137-byte USDA，SHA-256 `0812faf5...56c8c6`。断网 OpenUSD 25.11 解析出默认 `Example_Rotary`/`OmniLidar`，四类 dependency 均为空，16 项 evidence hash 通过；没有 GPU、Writer、sensor、数据、标签或训练。

`configs/v3/gate0/isaac_official_local_usda_writer_control_v2.json` 已按批准范围执行一次并停止。精确官方 local USDA 创建成功、Writer attach 成功、同步 standalone 更新 300 frame，但 callback、valid GMO、positive element 和 complete scan 均为 0，shutdown 有两次 writer-drain timeout；15/15 封存证据 hash 通过，结果目录 160 KB，无点云、图片、checkpoint 或模型，无重试。

该 run 的封存 summary 保留预先声明的 `HOST_OR_HEADLESS_REPLICATOR_RUNTIME_BLOCKED`，但后续只读源码审计发现 NVIDIA 的正式 Writer 单测使用 `AsyncTestCase + create_new_stage_async + wait_for_viewport_async + next_update_async`，不同于本次同步 standalone loop。故当前科学结论必须收窄为：同步 standalone Writer scheduling 已失败，通用 async Kit runtime 尚未测试。审计见 `docs/ISAAC_ASYNC_WRITER_RUNTIME_AUDIT_V1.md`。

用户已批准停止让 Isaac 阻塞主线。`configs/v3/gate0/isaac_official_async_writer_control_v3.proposal.json` 已退休，未实现、未执行；历史 Isaac 结果继续不可变保存。

单图 CPU 合同已按批准只执行一次并 PASS：1 个已有 world、24 poses、双 scene 与旧 reference 最大差异均为 0 m，24/24 净空/branch LOS 通过，46/46 evidence hash 通过，正式数据和训练仍为 0。

五拓扑 selector-v2 corrective run 已于 2026-08-11 PASS，详见 `docs/FIVE_TOPOLOGY_CPU_LIDAR_CONTRACT_V3.md`。其 750 个静态观测保持 diagnostic 身份，不转入正式数据集。下一份批准材料必须只针对 100 topology parent 的生成与验收；LiDAR 批量采样、模型训练和规划集成分别另行审批。

## 10. 2026-08-11 当前权威更新：M0F PASS，M1 待审批

本节覆盖第 9 节中已经过时的 M0R 当前状态。M0--M0R4 的 replay 路线已终止；原因不是降低科研要求，而是下游实际读取封存 OBJ，独立 Poisson remeshing 的 pairwise equality 不是任务所需复现单位。

M0F 已按批准范围正式 PASS：十个 train recipe sentinel 各一次 primary、零 replay，10/10 source/quality/map 通过，10 个 OBJ SHA 唯一，110/110 evidence hash 通过。正式 LiDAR、标签、样本、训练、模型和 M-TARE change 仍全部为 0。审查见 `docs/CANO_100_PARENT_PERCEPTION_MESH_M0F_REVIEW.md`。

当时的历史下一步（已被后续权威更新覆盖）是评审 `configs/v3/gate0/cano_100_parent_perception_mesh_m1_immutable_assets.proposal.json`。M1 提案固定 100 parent、80/10/10 split、100 primary、0 replay；只人工查看 80 张 train 完整图，validation/development-test 只做冻结自动检查并封存。M1 未批准前不实现、不执行；M1 即使 PASS 也不自动进入 LiDAR 或训练。

## 11. 2026-08-11 当前权威更新：M1 FAIL 已由 M1R 合同修复并完成

本节覆盖第 9--10 节中过时的“mesh 待审批”状态。M1 在第 45 个 parent 因 1 个严格零面积三角面停止，且暴露预览标题硬编码为 M0。失败 run 保持封存，不回写。

经新批准，M1R 对全部 100 parent 从头执行统一 `doubled_area <= 1e-12` sanitation，并修正阶段 provenance。正式结果 `PASS_CANO_100_PARENT_PERCEPTION_MESH_M1R`：100/100 primary、100/100 sanitation、80 train previews、0 replay；总计 `12,608,078` vertices、`25,217,511` final triangles、0 degenerate，100 个 OBJ hash 唯一，1001 项证据封存。validation/development-test 未渲染。

本轮 sanitation 实际删除 0 个面，说明 M1 的单个退化面受 Poisson 运行级非确定性影响；M1R 的有效修复是把确定性审计放入每次资产物化必经路径，而非保证每次都会删除面。正式复核见 `docs/CANO_100_PARENT_PERCEPTION_MESH_M1R_REVIEW.md`。

当时的历史下一步（已被后续权威更新覆盖）改为：先形成 CPU synthetic ↔ Gazebo fixed-pose LiDAR parity proposal/card，明确 pose、ray、frame、匹配指标、阈值和可视化后交用户审批。未经批准不执行 Gazebo，不生成正式 100k 数据，不训练 CNN/GNN，不建立或替换 M-TARE 高层规划器。

## 12. 2026-08-12 当前权威更新：CPU↔Gazebo fixed-pose parity PASS

本节覆盖第 9--11 节中过时的 parity 待审批/待执行状态。用户批准的唯一正式 run
`gate0_20260812_cano_cpu_gazebo_fixed_pose_lidar_parity_v1_seed0` 已完成并封存为
`PASS_CANO_CPU_GAZEBO_FIXED_POSE_LIDAR_PARITY_V1`。

固定范围为 1 个解析 box、3 个 train parent、24 个预选 pose、24 个 Open3D CPU reference 和
72 个 Gazebo 保存帧；每个传感器先丢弃 2 个 warm-up 帧。24/24 pose 全部通过，角色严格为
8 interior / 8 junction / 8 terminal。Gazebo 三次重复的 valid mask 完全相同，最大 range 差为 0；
CPU/Gazebo 最差 pose 的 valid agreement=`1.0`、MAE=`8.0223e-6 m`、P95=`2.4796e-5 m`、
P99=`4.6730e-5 m`、水平最小距离差=`8.4668e-5 m`。70/70 evidence SHA-256 复核通过，
6 页固定图完整覆盖全部 24 pose，未发现空白或挑样本问题。

该 PASS 只证明封存 Cano 感知 OBJ 上的理想静态 CPU ray 与当前 Gazebo Classic CPU ray 插件
在固定合同下具有一致传感器域；不证明有噪声、运动畸变、动态可导航性、结构语义学习、在线图或
探索规划收益。parity NPZ 永不并入训练集，正式数据、teacher label、training sample、model、
trajectory、online graph 和 M-TARE change 仍全部为 0。

当前不自动进入下一 Gate。唯一待用户确认事项是是否接受本 parity PASS 并允许起草 Phase 2 正式
Data Card；新卡必须先明确 80 个 train parent 上的独立 place/trajectory、空间采样、原始/有效样本数、
16×720 student input、graph/spline outgoing-branch teacher、80/10/10 隔离、存储格式、成本和可视化。
未经新批准不得采正式 LiDAR、生成 teacher、训练 CNN/GNN 或修改 M-TARE。

## 13. 2026-08-12 当前权威更新：Phase 2 数据方案已起草，弧长口径待确认

本节覆盖第 12 节中“等待 parity PASS 确认”的状态。用户回复“那继续吧”，确认接受 parity
阶段结果并允许起草 Phase 2 数据方案；这不是数据导出、teacher 生成或训练批准。

已形成 `docs/PHASE2_SUPERVISED_RANGE_DATA_V1.md` 及 pending Data Card/proposal。拟定范围为
80 train parent、10 validation parent、20,000/2,500 个不重叠 place cluster 和
100,000/12,500 个 16×720 frame；C10 development-test 与 M-TARE benchmark 本操作读取为 0。
学生输入只含 range/valid，teacher 是 5 m spline-sphere outgoing branch + mesh LOS；本阶段不使用
AI 标签、BEV、GNN、对比学习或训练。

冻结 90-world registry 前发现一个数据口径问题：Cano `spline.discretize()` 的 `distances[]` 与
封存 `points[]` 累计欧氏弧长在 train/validation 合计分别相差 790.421 m/105.010 m，单 world
最大相对差 1.181%。现有 pose 插值、raycast 和 teacher 均按实际 `points[]` 世界坐标工作，故推荐
正式采样统一使用 `points[]` 的 float64 累计欧氏弧长，旧 `distances[]` 仅保留 provenance。

当前唯一待用户决定项是是否接受该弧长口径。确认后只允许补齐 90-world immutable registry、
每 world 精确配额和最终 Data Card，再单独提交数据导出批准；当前不得实现 exporter、生成数据或
teacher、运行 preflight/create_run、训练模型或修改 M-TARE。

## 14. 2026-08-12 当前权威更新：弧长获批，role overlap 待决定

用户已经明确批准第 13 节推荐的 `points[]` 累计欧氏弧长口径。继续做 90-world 精确审计时发现，
原 junction-first 互斥 role 会让 train 22 个、validation 8 个 terminal 事件失去同角色候选；
这些事件均有空间候选，缺陷来自 role overlap 处理。

推荐数据 manifest 同时保存 `near_junction` 与 `near_terminal`，并只为固定配额定义
`primary_role = terminal > junction > interior`。该反事实保持 80/10 worlds、22,500 clusters、
112,500 frames 和原角色目标不变，候选容量为 train `20007/3307/803`、validation
`2625/407/98`，全部 junction/terminal 事件恢复同 primary-role 覆盖。

当前唯一待用户决定项是是否批准该 multi-label + terminal-first primary-role 规则。批准前不得冻结
registry 或 per-world quota；正式数据导出、teacher、run 和训练仍未授权。

## 15. 2026-08-12 当前权威更新：90-world registry 完成，等待最终 Data Card 审批

用户回复“继续”，确认第 14 节 multi-label + terminal-first primary-role 规则。registry
`configs/v3/gate1/manifests/cano_phase2_world_registry_v1.json` 已冻结，SHA-256 为
`d64edd758d86d1192dff870cdba08d392388cb48cfd755f428acdfb3ff4e8420`。

机器审计确认 80 train/10 validation、90 个唯一 ID、20,000/2,500 clusters、100,000/12,500
frames，所有 per-world/role quota 不超容量，junction/terminal 每事件最少 3/1 个候选，C10 row=0。
完整 `v3_data_card_v1` 现等待用户审批。该审批若通过，只授权实现 selector/teacher/exporter、测试、
run spec/preflight 和一次正式 data export + objective teacher；不授权训练或 M-TARE 修改。

## 16. 2026-08-20 当前权威更新：整篇论文流程获准继续，进入 Phase 5

本节覆盖此前“Phase 5--9 只定义不实施”和“planner/M-TARE 禁止”的范围限制。用户明确要求持续完成
整篇论文：以结构语义构建的在线拓扑图替换 M-TARE 全局规划器，完成单机器人、多机器人、严格测试、
基线对照、统计分析和论文证据。该授权扩大实施范围，但不取消 Gate、数据隔离或一次性 formal run 治理。

Gate 4 以 `GATE_MIXED` 收口：C08 开发链路与 C09 冻结回放机器 PASS，但 C09 上 M1D 仅小幅提高
composite，同时 exit-F1 和冗余指标退化。C10 继续封存，直到规划方法和所有开发参数冻结后才作为严格测试。

当前 Phase 5 的唯一研究问题是：冻结因果拓扑图能否通过 M-TARE 公共接口持续产生有限、可达、确定的
全局目标，并在同一 localPlanner/pathFollower 下达到或超过原 M-TARE 的探索效果。准确替换单元为
`tare_planner_node`；保留 simulator、LiDAR、terrain、独立 `localPlanner`、`pathFollower` 和控制。

执行顺序固定为：

1. 正式实现纯规则拓扑 frontier selector、verified-edge Dijkstra 回退、local rejection/retry 状态和
   `/way_point`、`/map_clearing`、`exploration_finish`、`/runtime` 交接合同；
2. 在 3 个 C08 development worlds 的 4,773 帧上做 5 方法逐帧 shadow replay，共 23,865 个 planner cycle；
3. shadow 合同通过后，在 M-TARE development worlds 做 GT-TNG Oracle 可行性与原 M-TARE 公平基线；
4. Oracle 不可达到原 M-TARE 时停止规划路线；否则进入单机器人替换、多机器人分配与最终严格测试；
5. C09 可作冻结后的非严格验证，C10 与新增 sealed worlds 只作最终测试，不用于修正方法。

本次用户指令允许实现与提案准备；任何正式 shadow/closed-loop/benchmark 执行仍须按项目规程在 preflight
后展示精确数据、方法、成本和证据，并消费一次不可覆盖执行批准。

## 17. 2026-08-20 Gate 5 收口并由用户授权进入 Gate 6

Gate 5 以 `GATE_MIXED` 收口。C08 的 23,865 次 shadow planner cycle、ROS 交接、分层地图
Oracle、统一 coverage evaluator、M1D ROS checkpoint 无损导出以及原 M-TARE 显式 planner seed
补丁均已通过；但完整 Gazebo/LiDAR/ROS 闭环在相同 seed 下不是逐帧 bitwise deterministic，不能继续
使用原 50-case 单次精确配对设计。这个限制属于仿真/调度随机性，不否定 shadow 接口或拓扑方法。

用户在看到两个备选后明确批准方案 A，因此 operational Gate 切换到 Gate 6。唯一获准的正式设计为：
2 个 development worlds（tunnel、garage）×5 个环境 seed（11/23/37/53/71），每个 block 包含
原 M-TARE 3 次执行、M1D checkpoint seeds 0/1/2 各 1 次、完整开发地图 Oracle 3 次，共 90 cases；
每 case 600 仿真秒，总计 54,000 仿真秒。运行顺序由 seed 20260820 一次性随机化，禁止结果驱动重排
或失败重试。统计单位为 10 个 world×environment-seed blocks；执行重复和 checkpoint seed 只用于估计
系统/模型变异，不冒充独立世界。C09/C10、训练、checkpoint/参数选择保持为 0。

Gate 6 的唯一研究问题是：在保留相同 simulator、LiDAR、terrain、localPlanner、pathFollower 和控制栈时，
冻结 M1D 因果拓扑全局规划相对原 M-TARE 是否改善 coverage-time 效率，并具有可报告的闭环稳定性。
正式 run 仍须先冻结 Gate-6 Data Card/spec、通过 preflight，再消费上述一次性批准。

## 18. 2026-08-20 Gate 6 域失败后授权重新打开 Gate 2 corrective representation

Gate 6 V1 的首个600秒case证明三个冻结M1D checkpoint不能直接部署到AEE传感域；后续只读移动bag复核中，
seed0/1/2在99帧上的空方向分别为99/97/99，而冻结B0为99/99非空。长期B0 fallback会实质替代学习语义，
不能支撑论文核心主张。完整地图teacher在20个移动pose上20/20给出非空且count一致的客观方向。

用户在审阅精确方案后回复“批准”，明确授权临时把operational Gate从6切回2，执行AEE域适配数据导出、
完整地图teacher生成和三个seed的head-only adaptation。固定范围为AEE tunnel训练、AEE garage验证，
seeds 11/23/37/53/71，共10条原M-TARE轨迹、30,000 raw同步帧、每5帧取1帧形成6,000 effective samples。
encoder和embedding head必须冻结，只训练direction/count/role heads。用户在发现合同冲突后再次回复“批准”，明确允许
sealed Cano V2R C09 validation只读用于每epoch抗遗忘评估及checkpoint并列选择，禁止进入训练、loss、归一化、
阈值或超参数调整；C10和后续sealed worlds读取为0。该C09例外只适用于本次Gate-2 head adaptation，不能外推。

Gate 2纠正PASS后仍不得自动恢复90-case。必须先回到Gate 6做2 worlds×3 adapted checkpoints×180秒的
6-case readiness；每case移动至少5m、图至少2 nodes/1 verified edge、至少1个非hold waypoint，且首20帧后
learned-empty/B0 fallback不超过5%。只有readiness PASS并获正式执行批准后，才冻结新的90-case矩阵。

## 19. 2026-08-21 AEE sensor-interface + multi-geometry corrective route

Head-only V1R3和full-encoder V2均已科学FAIL；同pose sensor-operator audit进一步证明AEE 350水平束到Cano
720列的接口差异与单训练world几何不足同时存在。用户选择推荐方案A：保留学习语义主线，同时纠正显式角采样
接口并增加独立开发几何；不降级为B0主方法。

正式接口必须从保留物理束身份的16×350组织化扫描开始。只有相邻两个源束均有真实回波时才允许重采样插值；
真实无回波不得补造。旧registered-scan短缝补齐只作诊断，不得作为正式训练输入。

新几何先执行独立topology-only资格：十个冻结结构层各预声明C13--C24共120个seed，沿用Gate-0 V2生成与验收
规则；每层按数值顺序取前两个有效且身份唯一的候选，分别作为corrective train/validation。任一层不足两个即FAIL，
禁止追加C25、删复杂层或降低合同。该资格run仍需在展示精确120/20计数、约2小时CPU/2GB成本和证据后获得一次
正式批准；通过前mesh、LiDAR、teacher、training、C09/C10和M-TARE读取均为0。

2026-08-21状态更新：上述唯一topology-only run已获精确批准并正式PASS。120/120候选完成，十层有效数为
`11/11/10/12/11/12/10/12/10/12`；固定选择10个corrective-train和10个corrective-validation parent，20个
coordinate-bearing topology identity全部唯一且split-disjoint。376/376 seal通过；mesh、LiDAR、teacher、training、
model、C09/C10和M-TARE计数仍为0。下一允许工作是为这20个冻结parent实现并只读资格化native perception mesh、
空间分离pose、16×350射线、16×720重采样和不变teacher合同，形成精确Data Card/spec与成本证据；不得把本次
topology审批扩展为正式数据生成或训练审批。

进入实现前的只读镜像审计补充：AEE raw `/velodyne_points`确实以`width=16,height=350`、NaN no-return形式组织，
但其350水平样本覆盖闭区间`[-π,+π]`，首尾是重复物理方向；现有通用重采样原语的半开圆周假设不能直接应用。
当前唯一允许步骤收窄为先明确并测试exact-angle canonicalizer和重复端点策略。不得修改冻结xacro，也不得在该接口
决定前生成20-world mesh/sensor正式资产。

用户随后批准方案A：全部350 raw records继续作为provenance，`±π`重复方向按既有first-return同格规则合并，
其余目标列仅在左右真实方向均有效时插值；AEE与Cano共享相同闭区间source-angle合同。实现与17项测试已PASS。
下一允许工作恢复为独立的20-parent immutable perception-mesh资格准备；它必须与后续11,000-frame数据导出分成
两个material runs，任何mesh失败都不得产生正式sensor/teacher样本。

## 20. 2026-08-23 当前权威更新：Gate 6 单机器人正式矩阵与 V4 回锚纠正

本节覆盖第 9--19 节中所有历史“当前唯一任务”与旧等待审批措辞。用户已授予既定论文范围内持续执行权；
常规的 Data Card、spec、preflight、不可覆盖 run 和证据 seal 仍必须执行，但不再逐项索取重复确认。改变数据
范围、split、teacher、指标、阈值，读取 C09/C10 作开发，或删除/覆盖正式证据仍必须停止报告。

当前 Phase 为 `PHASE_6_90_CASE_SINGLE_ROBOT_MATRIX`。唯一正式 V9 90-case run
`gate6_20260822_aee_composite_v9_stochastic_v3_seed20260820` 正在执行；固定范围为 2 个 development worlds、
5 个环境 seed、10 个统计 block，原 M-TARE/M1D/Oracle 各 30 case，每 case 600 仿真秒。原 run 不修改、
不重排、不重试；C09/C10 保持隔离。

冻结 runner 已确认存在 V2 case status 与 V1 analyzer 的接口不兼容。原 run 完成 90 个 case 后应保留其系统 FAIL
和 seal，不得回写；随后只读兼容 audit 仅在内存深拷贝中映射 status 并生成独立证据。机制审计已定位 M1D 的
verified graph backtrack 抵达旧节点后 `current_node` 不切换。V4 显式因果回锚、matched probe、六例 readiness、
严格复用源 schedule 的 30-case runner 与最终配对统计均已实现并通过单元/ROS Python 3.8 静态验证，但真实闭环
效果仍未知，不得提前宣称修复成功。

当前唯一执行顺序为：原 90-case 完成并 seal；只读 V2 兼容统计及30个V9案例的回锚/exit-stub lifecycle全量审计；
根据预声明的全量机制证据决定继续 matched V4-only probe，还是先形成同时覆盖生命周期缺陷的最小组合修正；
通过 matched probe 后再做2 worlds×3 checkpoints readiness、精确配对30-case和sealed corrected comparison。单机器人结论完成后才可按新 Data Card/spec
进入 Gate 7 多机器人和后续严格测试；不得自动把 Gate 6 局部结果表述为整篇论文完成。

## 21. 2026-08-23 当前权威更新：V3R透明恢复与组合 V5 单机器人结论链

本节覆盖第20节中“V4-only或组合修正尚待决定”的历史分支，但不改动其数据、基线、统计单位、指标或隔离合同。
原V9 90-case run在完成77例后因主机时钟回拨在case077事后统计阶段按冻结合同FAIL并seal。不得删除负值、复用
case077 partial、只重跑失败baseline或覆盖旧run。当前唯一恢复run固定重跑完整受影响`tunnel_env23`九例和未执行
尾部十二例；最终只允许由旧run中69个不受影响PASS案例与新run的21个案例组成透明90-case来源，并由独立只读
composition audit验证10个block、30/30/30方法配额、69+21来源配额、schedule双hash和全部来源seal。

30个缺陷V9案例的预声明机制审计已使V4-only充分性假设失效：除verified backtrack后的陈旧current-node外，还存在
frontier目标出口与实际verified departure不一致、same-node loop后旧stub继续被选择的生命周期缺陷。当前方法候选固定
为V5：保留V4 verified re-anchor，并把不一致执行结果写入既有retry ledger后确定性重选；不增加阈值，不改变感知模型、
图参数、M-TARE localPlanner/pathFollower/control、传感器、地图、checkpoint或评分指标。

当前唯一执行顺序为：完成并seal 21-case V3R；执行69+21组合性能/双机制审计；按outcome-blind规则执行一个matched
V5 probe；执行2 worlds×3 checkpoints readiness；执行与缺陷V9精确身份配对的V5 30-case；执行最终只读comparison。
最终comparison必须同时输出V5对原M-TARE主比较、V5对缺陷V9因果前后比较、GT-map诊断、七指标十block统计、
来源seal验证的120条coverage curve、固定身份轨迹/拓扑图和非手填CSV/Markdown/LaTeX表格。Gate 6给出明确结论后
才允许物化Gate 7多机器人Data Card/spec；C09/C10在方法与开发参数冻结前继续隔离。

## 22. 2026-08-23 当前权威更新：GSE-Graph 成为唯一论文主线

本节覆盖第 20--21 节作为“最终方法路线”的地位，但不终止正在运行的 V5 30-case，也不改写任何旧结果。V5 完成后只作为 `Cano-like exit-only perception + deterministic causal graph` 基线、消融和失败机制证据；不再继续扩展为论文主方法。

当前唯一研究问题改为：从 5 帧因果 LiDAR 中学习到的显式几何结构语义与不确定性，能否直接决定结构节点生成、重访/出口关联和边的几何属性，并在 trace-verified edge 合同下形成比出口模型与非学习几何图更准确、更有效的在线 topometric graph？

完整方法、固定数据、基线、离线门槛、闭环顺序、清理和论文图保存合同以 `docs/GSE_GRAPH_RESEARCH_PLAN_V1.md` 为准。实现顺序固定为：旧基线自然封存；新颖性矩阵与序列 inventory；Data Card；teacher/类型接口；5 帧模型三 seed；离线关联与图门槛；单/多机器人闭环；严格测试；复现包与投稿 PDF。

用户已明确授予既定 GSE-Graph 论文范围内的持续执行权，不再为常规实现、只读 inventory、Data Card/spec/preflight 或满足前置条件后的唯一 material run重复请求批准。若出现会改变数据隔离、teacher 定义、研究问题、方法主张、验收阈值或结论有效性的实质问题，仍须停止受影响动作并报告证据。任何论文图必须保留矢量源、生成脚本、机器可读源数据与 SHA-256；任何删除必须按逐目录 allowlist，并在活跃旧 run 完成后执行。

## 23. 2026-08-27 当前权威更新：Factorized GSE-Graph替代五类节点与普通变化点

本节覆盖第22节中“junction、terminal、turn和geometry-transition全部作为同类结构节点”的方法语义，但不改写旧run。五类categorical episode detector及class-mass审计已按预声明停止规则FAIL；随后C01--C08普通几何/动作摘要变化点proof系统正常但科学FAIL：combined precision=`0.552632`、episode recall=`0.031111`、相对更强单项基线变化=`-0.005185`。因此不得通过调阈值、加epoch或进入图参数搜索继续该路线。

当前主方法改为Factorized GSE-Graph：junction/terminal是改变可执行动作集合的decision nodes；turn、宽高、坡度、净空和曲率作为trace-verified edge geometry profile或内部metric polyline，不再强制进入结构node F1。现有连续几何平均改善`52.763%`及junction/terminal安全触发只作组件证据，整体方法仍未通过。

新唯一问题是：完整exit/action token、place descriptor、空间约束与已经实际穿越得到的incident-edge geometry fingerprint，能否在拒绝多义匹配的同时实现precision `>=0.98`、false loop merge `<=1%`和recall `>=0.25`的route-conditioned node association？在此之前C09/C10/M-TARE、离线图网格与闭环继续禁止。

完整接口、差异边界、证据身份与下一门槛以`docs/GSE_GRAPH_FACTORIZED_RESEARCH_PLAN_V2.md`为准。下一步只允许C01--C08只读association inventory/capacity proof准备，不允许新backbone训练或测试世界读取。

## 24. 2026-08-27 当前权威更新：结构关联 Teacher 完整，进入容量证明

Factorized association inventory确认旧runtime pair cache存在Teacher失衡：fit仅`27`个negative、selection仅`71`个negative，多数topology family为零。该旧缓存继续作为失败基线，禁止复制少量负对或扩大16 m runtime候选半径来掩盖。

identity-balanced structural-alias Teacher manifest正式PASS。固定C01--C08 `188,126`条观测、`1,066`个decision identities（fit `792`、selection `274`），每个identity生成一个positive和一个同split/event/incident-degree的objective-geometry最近hard negative，共`2,132`条pair records。positive由`563`个不同physical edge、`491`个reverse view、`7`个distinct observation及`5`个短terminal固定circular-shift augmentation组成；最后5个必须单独报告，不得称为physical revisit。fit/selection缺少完整5帧geometry profile的identity仍为`15/5`，保留显式mask，不删除。

唯一正式run为`gate3_20260827_gse_factorized_association_teacher_manifest_v1_seed0`，18项seal完整覆盖，SHA-256=`047b5d16b0083fcab56a2a04d286655b2711e148b704d98f30d958e7b1d90f5b`；optimizer、model inference、C09/C10、strict test和M-TARE读取均为0。该PASS只解除association Teacher阻塞，不证明模型关联或拓扑图有效。

当前唯一研究问题收窄为：在不向学生输入identity、world、parent、Teacher event或objective geometry profile的条件下，低容量route-conditioned association model能否在C07--C08达到precision `>=0.98`、false accept/false loop merge `<=0.01`且recall `>=0.25`，并优于距离/角度规则关联。下一步只允许冻结该capacity proof的输入列、baseline、三seed训练/选择单位、拒绝阈值和Data Card；通过前C09/C10、离线图、M-TARE与闭环继续禁止。

## 25. 2026-08-27 当前权威更新：route-conditioned association容量PASS，进入C09资格准备

唯一容量run `gate3_20260827_gse_factorized_association_capacity_v1_seed0`正式PASS。完整模型三seed在C07--C08 identity-balanced structural-alias selection上的precision=`0.990521/1.0/1.0`、false accept=`0.009479/0/0`、safe recall=`0.762774/0.645985/0.620438`；删除唯一selection singleton augmentation positive后仍全部通过。

三seed平均safe recall为：descriptor-only=`0.218978`、no-route-geometry=`0.199513`、full route-conditioned=`0.676399`，完整方法分别提高`0.457421/0.476886`，远高于预注册`0.05/0.02`。这证明收益来自incoming-edge geometry到candidate exit-token的关系，而非只靠place descriptor或同架构容量。61项seal完整覆盖，SHA-256=`91ef98c0bc5b4cd82492dd42499a57a1bfe22d020664fbb0189aa2345cee11cc`；总small-head optimizer steps=`1,239`、backbone updates/inference/C09/C10/M-TARE均为0。

该结果只在C01--C08开发/选择域冻结模型和阈值，不能冒充未见拓扑或在线图PASS。当前唯一下一步是准备一次C09 Factorized association qualification：只读冻结C09观测与真实runtime strictly-past `<=16 m`候选，同时建立C09 identity-balanced structural-alias审计；三seed模型、归一化和阈值不再选择或修改。C09必须同时验证balanced hard negatives与真实runtime candidate，报告per-family/physical-only/ambiguity rejection；失败即停止关联主张。C10、离线图参数、M-TARE和闭环继续禁止，直到C09资格PASS。

## 26. 2026-08-27 当前权威更新：先统一最终坡度接口，再进入C09

C09准备阶段发现容量V1的146维观测来自旧exit-token组件导出，其中坡度列仍是原GSE回归头；论文最终感知接口已经在独立正式run中冻结为physics-guided五帧坡度corrective加validation-only残差比例`0.89`。full关联恰好使用incoming slope与candidate exit vertical profile的关系，因此直接把V1模型用于最终C09会形成训练/部署接口漂移；若关联用旧坡度而edge属性用新坡度，则会形成不可接受的双重几何语义。

该问题不推翻V1在固定旧接口上的组件容量证据，但阻止直接进入C09。只读证明确认corrective缓存完整覆盖C01--C08的`142,184/45,942`条fit/selection序列，合计`188,126`个唯一global sequence IDs，并与关联Teacher逐身份完全对应。原checkpoint在冻结CUDA路径重放三个selection输出均逐元素完全一致；CPU重放的最大`0.034°`差异仅为设备浮点路径，不可用于正式资产。

当前唯一允许步骤改为一次独立V1R unified-interface corrective：三seed各在原CUDA路径重放`188,126`条坡度序列，固定乘`0.89`残差比例，只替换146维观测第10列并证明其余145列逐字节不变；随后保持Teacher、pairs、full/no-route架构、optimizer、checkpoint/threshold选择和全部科学门槛不变，重跑六个小关联头。descriptor与no-route输出还必须与V1逐字节一致，证明改动只影响route-conditioned分支。V1R失败则停止关联主张；PASS后才允许冻结C09资格。C09/C10/M-TARE/图/规划仍为零读取。

## 27. 2026-08-27 当前权威更新：统一坡度V1R正式PASS，恢复C09资格准备

唯一run `gate3_20260827_gse_factorized_association_capacity_v1r_seed0`正式PASS。三seed均完成`188,126`条冻结坡度CUDA重放，C07--C08 archived selection逐元素完全一致；每套146维观测只改变column 10，其他145列逐字节相同。descriptor与no-route selection archives也与V1逐字节完全相同，证明纠正范围只影响route-conditioned geometry relation。

最终统一接口下full三seedprecision=`1.0/0.995370/1.0`、false accept=`0/0.004630/0`、safe recall=`0.419708/0.784672/0.540146`，physical-only仍全部PASS。平均safe recall=`0.581509`，相对descriptor=`0.218978`提高`0.362530`，相对no-route=`0.199513`提高`0.381995`。相较旧接口V1平均recall有所下降，但仍远高于所有预注册门槛，因此不得选择旧坡度以追求更高数字。

run耗时`23.161s`，small-head optimizer steps=`1,239`，冻结坡度inference sequences=`564,378`，GSE backbone inference/update和C09/C10/M-TARE读取均为0；69项seal完整覆盖，SHA-256=`64e5bfa8ba9fa1ff64bf491f042cc2d1a33ae0c13242437ce75a39a5e6c5eb40`。V1R模型、归一化和阈值现在取代V1作为最终接口的开发冻结点。下一步恢复为一次C09只读资格：balanced alias与真实strictly-past `<=16 m` runtime candidates必须同时评估，禁止重新选择。

## 28. 2026-08-27 当前权威更新：单seed C09资格科学FAIL，转向选择域鲁棒聚合与metric一致性

唯一C09 qualification完整执行但科学FAIL，`error=null`。balanced alias三seedprecision=`0.989130/0.971831/1.0`、false accept=`0.010870/0.028169/0`、recall=`0.700000/0.530769/0.253846`；runtime precision=`0.970588/0.962160/0.983932`、false accept=`0.029412/0.037840/0.016068`、recall=`0.851075/0.572187/0.263211`。因此任何单seed都不能同时满足冻结1%错误预算，当前单seed关联主张停止，不进入离线图或planner。

失败不是无召回，而是复杂近邻节点的相关性误合并：runtime false positives集中于S08/S09；S02/S03本来就没有runtime negatives，已正确标记为precision不可识别。run精确覆盖C09 `24,462`序列、`130/260` balanced identities/pairs及`4,085/4,201` runtime queries/pairs，13,383次小头评分、0优化/选择/C10/M-TARE，25项seal SHA=`ca31e1d36958ae2291bb19c5c079fa7621b3b168e2fe5272de46ca00b99ac338`。失败图保留为论文failure analysis，不删除或覆盖。

可行方向中，禁止方案是使用C09重新调每seed threshold或选择最好seed。当前唯一允许的最小方法revision是：在C07--C08重新构造同定义runtime candidates，联合identity-balanced alias域，仅从selection域固定选择`k-of-3` seed consensus和不超过16 m的metric distance cap；目标仍为precision `>=0.98`、false accept `<=0.01`、recall `>=0.25`及family覆盖。C09只读诊断表明该机制有容量，但不得决定具体`k`或distance。若C07--C08不存在安全配置或冻结配置再次C09 FAIL，停止Factorized association主张并重新评估论文方法，不用图规划调参掩盖。

## 29. 2026-08-28 当前权威更新：鲁棒关联资格PASS，进入离线结构图资格

唯一两进程corrective run `gate3_20260828_gse_factorized_consensus_metric_corrective_v1_seed0`正式PASS。第一进程只能读取C07--C08：精确覆盖`45,942`序列、`8,839`个decision queries和`9,380`个runtime pairs；固定网格中8个配置满足更严格开发裕量，确定性规则选择`2-of-3`模型共识和`4.0 m` runtime distance cap。selection runtime precision/false accept/recall=`0.995027/0.004973/0.607356`。

第一进程退出并证明`c09_worlds_read=0`后，第二进程才读取已封存C09分数。C09 balanced precision/recall=`1.0/0.507692`，physical-only同样PASS；runtime precision/false accept/recall=`0.990135/0.009865/0.558281`，全部原始门槛通过。S02/S03没有runtime negatives，继续显式标记为precision不可识别，不作为完美安全证据。

正式run耗时`31.744s`、选择阶段峰值RSS=`1,563,052 KiB`、22项seal完整覆盖，SHA-256=`295004061d1a0ef5e83f29e483625573516d0571b1e038bfc6656b640c67c714`；输入未变化，optimizer/model update/checkpoint selection/C10/M-TARE均为0。原单seedC09 FAIL保留为必要基线与失败分析，不覆盖。

该PASS解除学习式decision-node association阻塞，但不证明生成的整张拓扑图正确。当前唯一下一步是在Phase 3内冻结一次离线结构图资格：完整GSE必须实际生成/关联decision nodes，以真实穿越提交edges及其geometry profile，并与原M-TARE、exit-only规则图、非学习几何事件图和GT-TNG oracle比较node/edge F1、false loop、连通分量与cycle rank。不得复用旧五类节点243-grid结果冒充Factorized方法；在离线图PASS前仍禁止C10、M-TARE闭环和planner调参。

## 30. 2026-08-28 当前权威更新：互斥事件分类改为相对位置多事件集合

本节覆盖第23节中“每条因果观测只输出一个junction或terminal类别”的接口，不覆盖已经通过的连续几何、三维中心、关联和执行验证证据。正式route-conditioned corrective证明：低维路线特征可把普通正确episode从`928`提高到`931`、false trigger从`94`降到`90`，但两个稀有关系端点仍为`0/2`，完整图node recall由`0.704380`降到`0.675182`，因此该小头科学FAIL并停止。

只读归因进一步证明，98个关系端点中96个与相反事件类型直接相连且欧氏距离小于50m；48个terminal端点全部在这一传感器量程尺度内直接连着junction。一个失败terminal的三个seed稳定输出相邻junction，另一个在三个seed间出现`0.967/0.106/0.016`级结构质量分裂。互斥scene-level标签会把“当前位置是terminal”和“传感器同时观察到身后junction”错误当成只能二选一。

后续主接口改为变量长度、带相对位置的结构事件集合：

```text
SpatialStructureEventToken
  event_type: junction | terminal | geometry_transition
  relative_axis_xyz_m
  local_axis / width / height / slope / curvature
  exit_tokens[]
  descriptor / confidence / uncertainty
```

同一因果LiDAR序列可以同时输出多个事件token；学生不得读取TNG identity、绝对world/pose或未来帧。在线图按token的相对三维位置产生/关联候选节点，模糊匹配保持provisional；edge仍只能由真实穿越建立，执行端点anchor、false-loop拒绝和M-TARE local planner合同不变。

不得立即训练。唯一下一步是C01--C08只读`spatial multi-event Teacher feasibility`：从TNG、spline、native perception mesh和冻结pose构造50m内且有任务相关LOS的事件集合，审计每帧集合基数、junction/terminal共现、跨方向重复、set matching唯一性、C01--C06/C07--C08覆盖和泄漏。只有Teacher唯一、两split均有足够多事件共现并能覆盖现有稀有失败，才允许新Data Card和集合预测head；否则停止多事件路线并重新定义可观测action event。C09/C10/M-TARE继续禁止。

## 31. 2026-08-28 当前权威更新：空间多事件Teacher可行性正式PASS

本节完成第30节规定的唯一前置证明，并覆盖其“Teacher可行性待定”的状态。首次V1运行在读取任何world或执行任何raycast之前，因为把四行JSON数组交给object-only治理读取器而system FAIL；原run不可修改，12项证据seal保留。V1R只改为标准JSON数组解析并增加回归检查，数据、50m范围、0.25m LOS margin、16-token预声明容量和全部科学门槛不变。

V1R完整只读C01--C08 `80`个世界、`16,078`条directed traversals和`188,126`条五帧因果观测。50m内共有`326,974`个TNG事件候选，native perception mesh LOS确认`133,055`个可见、`193,919`个被遮挡。全部`1,076/1,076`个degree-1 terminal或degree>=3 junction identity至少可见一次，并全部具有相差120--240度的反向观测；最大单帧集合基数为`5`，远低于预声明容量16，set identity/relative position零歧义。

C01--C06 fit有`17,811`个多事件帧和`10,052`个terminal+junction共现帧；C07--C08 selection有`6,153/3,282`个，10个topology family在两split均存在共现。四个稀有失败行的objective terminal全部可见；global row `44299`和`110361`还分别同时看到terminal与junction。这直接否定互斥scene label，同时证明多事件监督不是少量个例。

正式run为`gate3_20260828_gse_spatial_multi_event_teacher_feasibility_v1r_seed0`，19项seal逐项零mismatch，seal文件SHA=`c557e1d184697b57e7e3a103e32581ce5df43a319da5a129698b419d3d4117c2`，论文图SHA=`5bf7fe520446d780aa98781cfffc66af2c310f42d10f031b7ddb57988aa418b7`。0训练、0模型推理、0学生输入读取、0完整Teacher导出，C09/C10/M-TARE零读取。

下一唯一工作是在新Data Card下导出C01--C08空间事件set Teacher：每行最多保留预声明的16个`event_type + relative_xyz + mask + Teacher-only identity`，不按观测到的最大5缩小容量，不丢弃零事件帧；学生仍只读取既有五帧LiDAR。junction/terminal作为结构decision event，geometry-transition继续作为连续edge geometry语义，不在本轮伪装成已验证的结构node。由于该动作属于`teacher_generation`，治理状态暂时回到Gate 2执行唯一导出；这不改变研究问题或重新开放旧训练。Teacher导出完整性PASS后返回Gate 3，才允许设计固定query的集合预测head和三seed训练；C09/C10/M-TARE继续禁止。

## 32. 2026-08-28 当前权威更新：16槽空间事件Teacher导出正式PASS

唯一Gate-2 teacher-generation run `gate2_20260828_gse_spatial_multi_event_teacher_export_v1_seed0`正式PASS。80个Zarr shards逐世界精确重放feasibility V1R：C01--C06/C07--C08仍为`142,184/45,942`条观测，全部`188,126`个global sequence identities唯一；可见token精确为terminal `30,789`、junction `102,266`，合计`133,055`。

固定16槽合同零截断；`81,069`个零事件行原样保留，单/多事件行=`83,093/23,964`，实际最大基数仍为5。identity index和TNG只存在Teacher资产，学生输入没有被复制或读取；geometry-transition继续保留为edge geometry profile。0训练、推理、normalization或threshold selection，C09/C10/M-TARE零读取。

1,634项seal逐项零mismatch，seal文件SHA=`29d821e2c9935a7e9f24dc9aa56ed14c01131a397c922c5f6f47330c8caef930`；论文图SHA=`4f7278394a89dc28827adcba55b88efe6acb2e04c5a7b6eca77ed43c82e9d343`。Gate状态返回Phase 3。

下一唯一任务是先实现并冻结零训练的`SpatialEventSetDecoder`接口和匹配/loss/readiness：16个query从既有五帧circular directional feature直接预测presence、terminal/junction、robot-relative xyz、descriptor和uncertainty；禁止使用绝对pose、world、TNG identity或未来帧。只有circular equivariance、空集合、1--5事件匹配、query permutation、masked padding、有限梯度和Teacher join测试全部PASS，才允许建立三seedData Card。正式训练前必须预注册4m三维匹配合同、precision/recall与旧互斥分类/非学习规则baseline，不得用C07--C08事后选择query容量。

## 33. 2026-08-28 当前权威更新：SpatialEventSetDecoder readiness正式PASS

V1全量Teacher/data join、旋转和finite backward均PASS，但query permutation loss error=`2.9373e-4>1e-6`，定位为exact assignment用任意query编号打破近等价匹配。V1保留system-interface科学FAIL证据，不进入训练。V1R只把预测按内容规范排序后再匹配，不改变Teacher、loss权重、query数或阈值。

V1R正式PASS：80 shards、188,126 rows、133,055 tokens及全部0--5基数逐项复现；decoder固定`93,638`参数/16 queries。query permutation loss error=`1.1921e-7`，circular rotation最大误差=`8.2970e-5m`，真实Teacher batch finite backward。10项seal SHA=`a9625fa3eac099fa2a63a259f55e98f5200aacc49df60bc921e9571f76a62c78`，0 optimizer/trained inference/C09/C10/M-TARE。

下一步允许冻结三seedcapacity训练Data Card：每个seed复用对应冻结五帧encoder，只训练各自93,638参数decoder；C01--C06拟合、C07--C08选择。匹配正确必须同event type且3D误差`<=4m`，同时报告precision、recall、false positives、位置MAE和多事件子集；旧互斥scene detector加单中心是主baseline，非学习几何事件器是第二baseline。若集合方法不能同时改善多事件召回和位置/安全，不解冻backbone或进入图回放，先停止复核表示容量。

## 34. 2026-08-28 当前权威更新：冻结旧encoder的集合头容量正式FAIL

唯一正式run完成三seed各`4,448` decoder optimizer steps，总计`13,344`；三个旧五帧encoder均为0步更新。C01--C06 fit=`142,184`行，C07--C08 selection=`45,942`行/`33,563` visible tokens/`6,153` multi-event rows；正式进程C09/C10/M-TARE零读取，输入前后哈希一致且`error=null`。

三seed在预注册同类型、3D误差`<=4m`合同下P/R/F1分别为`0.142821/0.119209/0.129951`、`0.211947/0.166076/0.186228`、`0.161066/0.129667/0.143671`，multi-event recall最高仅`0.154335`。成功匹配位置MAE约`2.49--2.63m`，但大量query无法对应Teacher。旧互斥单中心baseline为P/R/F1=`0.765334/0.262849/0.391306`、MAE=`1.737m`，严格优于全部set seed；非学习几何F1=`0.003171`。

因此当前阻塞改为`SPATIAL_EVENT_SET_REPRESENTATION_OR_DUPLICATE_QUERY_FAILURE`。不得重试调loss、threshold、query数、epoch或直接解冻backbone，也不得进入C09/C10、图回放或planner。唯一下一步是C01--C08既有输出的只读failure attribution：按距离、set cardinality、type和family分层，量化duplicate queries、径向分布、同类型NMS与oracle-dedup上界。只有去重上界能超过旧baseline时才允许最小repulsion/cardinality corrective；否则必须重新设计并监督一个真正保留远程空间结构的encoder。正式seal SHA=`bdcbbf1552c6f592376d4f85912f18c90ddb6eea28568a0db1c7bdfafc404322`，论文图SHA=`fa340e856e59feece57db0e48b210ce9f9ad1f4726d3ee75722fabe7b26c8ef1`。

## 35. 2026-08-28 当前权威更新：失败归因要求geometry-anchored空间encoder

V1R只读审计精确复用C07--C08 `45,942`行和三个sealed selection archives。4m同类型NMS只抑制`0/28/260`个候选，F1不变；全部16 query的typed position oracle recall也只有`0.199684/0.255102/0.212764`，没有达到旧互斥baseline recall `0.262849 + 0.10`。因此duplicate-query和confidence/cardinality两条最小corrective均被预注册决策门否定。

目标距离从0--4m增加到32--40m时，三seed实际recall从约`0.403--0.562`降至`0.005--0.012`，40--50m近零。当前阻塞唯一分类为`ENCODER_SPATIAL_REPRESENTATION_INSUFFICIENT_NEW_ENCODER_OBJECTIVE_REQUIRED`。下一组件固定为`GeometryAnchoredSpatialEventEncoder`：从五帧organized LiDAR保留polar angle/elevation布局，显式加入sin/cos位置和可见free-range profile；query仍输出terminal/junction集合，但径向距离必须表示为对应注意力射线free-range内的bounded fraction，空间encoder与set目标联合训练。

不得立即训练。唯一下一步是零训练接口/readiness：验证输入只含range/valid，circular roll与robot-frame xyz严格等变，深度不超过对应free-range anchor，空集合/1--5事件assignment和loss有限，query permutation、padding、Teacher join、参数量和反向传播全部确定。readiness PASS后才允许C01--C06 fit/C07--C08 selection的joint encoder+set三seedData Card；C09/C10/M-TARE、graph replay和planner继续禁止。V1R seal SHA=`2cf2bd47a0607ea5c22f5b563d8c99b49f45a57cb15c765ae06c64de553343aa`，论文图SHA=`1a798eea42ace02aaec1c70a3ef60b0e8c9c5d352044f05b77cf07cfc35a3ef3`。

## 36. 2026-08-28 当前权威更新：GeometryAnchored readiness暴露Teacher垂直FOV错配

唯一正式V1 run在0训练条件下精确复现C01--C08 `80` worlds、`188,126` rows、`133,055` Teacher tokens和`1,076` identities。新`GeometryAnchoredSpatialEventEncoder`固定`264,134`参数，只接受五帧organized range/valid；真实cardinality 0--5 finite backward，query permutation error=`0`，18度circular rotation xyz error=`4.77e-7m`，seed replay逐位相同，预测径向距离严格不超过free-range anchor。所有model/interface/system门均PASS。

唯一科学FAIL是Teacher support：`641`个token超出当前polar-cell free range。复用原Teacher `0.25m` LOS margin后仍有`639`项，`619`项目标仰俯角超出冻结16线LiDAR的`[-15°,15°]`垂直FOV。五帧free-range envelope仍有`585`项失败，其中`577`项视场外，故不能用时序anchor、NMS、loss或训练调参修复。问题分类为`TEACHER_SENSOR_VERTICAL_FOV_MISMATCH`。V1不可修改，12项seal SHA=`39610c5fb4317da031614e52760ea1aa86ccc893fb5664d4fb79c33b541be5ce`。

本节覆盖第35节“原Teacher可直接进入joint training”的假设。禁止扩大LiDAR FOV，因为那会改变冻结传感器合同并重生成252,430帧与全部基线；禁止按模型失败列表删标签。唯一允许的纠正是Teacher V2按事前固定传感器FOV删除全部`1,631`个视场外token，不删除任何观察或identity。V2精确预期为`131,424` tokens，terminal/junction=`30,714/100,710`，fit/selection=`98,279/33,145`，cardinality 0--5=`82,326/82,183/21,756/1,724/128/9`，仍保留`1,076` identities。

encoder的物理anchor同时改为五帧局部free-range envelope并复用Teacher既有`0.25m` LOS margin；学生输入仍只有range/valid，无pose/TNG/future。下一唯一工作是正式Teacher V2导出和随后一次新readiness；二者全部PASS后才允许冻结joint encoder+set三seed训练Data Card。C09/C10/M-TARE、graph replay和planner继续禁止。

## 37. 2026-08-28 当前权威更新：Observable Spatial Event Teacher V2正式PASS

唯一Gate-2 run `gate2_20260828_gse_observable_spatial_event_teacher_v2_seed0`正式PASS，`error=null`且source unchanged。80个V1 shards按固定规则逐项转换：保留全部`188,126` observations和全部`1,076` event identities，只删除target elevation不在inclusive `[-15°,15°]`的`1,631`个tokens。删除terminal/junction=`75/1,556`，V2保留=`30,714/100,710`，总计`131,424`；fit/selection=`98,279/33,145`。

V2 cardinality 0--5精确为`82,326/82,183/21,756/1,724/128/9`，零事件行与global sequence identity全部保留。filter不读取organized scan、模型预测、checkpoint、confidence或失败列表；1,631个删除项逐行保存原world/global ID/slot/type/identity/distance/elevation，避免不可审计的样本清洗。0训练/推理/normalization/threshold selection及C09/C10/M-TARE读取。

1,298项seal逐项零mismatch，seal SHA=`736261034852c779d26e2619735755ce1bbd8bc7a415dbc68257923e7cfa5772`；论文图PNG SHA=`1648cc4b0a291eaec27b7336872a5d6ba4d170b7597d26b1ecbfd806e5e0a0f6`。治理状态返回Gate 3。

下一唯一任务是重做GeometryAnchored zero-training readiness：anchor必须使用五帧因果free-range envelope、固定一bin局部圆周邻域和原Teacher `0.25m` LOS margin，仍只输入range/valid；V2全部`131,424` tokens必须有表达支持，同时保持V1已经通过的rotation/permutation/determinism/finite-backward门。PASS后才允许joint encoder+set三seedData Card。

## 38. 2026-08-28 当前权威更新：GeometryAnchored readiness V2正式PASS

唯一正式run `gate3_20260828_gse_geometry_anchored_spatial_event_readiness_v2_seed0`完整审计C01--C08 `80` worlds、`188,126` unique observations、`131,424` observable Teacher tokens和`1,076` identities。五帧因果free-range、固定±1 polar bin局部包络及既有`0.25m` LOS margin覆盖全部tokens，unsupported=`0`；最小support margin=`2.2888e-5m`，NumPy审计器与Torch模型profile逐元素误差=`0`。

`GeometryAnchoredSpatialEventEncoder`固定`264,134`参数，只接受`range_m/valid_mask`。真实cardinality 0--5 finite backward；query permutation loss error=`1.1921e-7`，18度rotation xyz error=`7.1526e-7m`，rotation invariants=`3.8147e-6`，seed replay bit exact，预测range严格受anchor约束。所有人口、source、数值、物理与隔离门PASS；0 optimizer/trained inference/threshold selection和C09/C10/M-TARE读取。

12项seal逐项零mismatch，SHA=`de4a734e64339d0f73bd3ee415f7337126d323574e26ee7e2f8a8e477349740d`。该结果只解除joint training前置阻塞，不是模型效果结论。

下一唯一任务是冻结三seed joint encoder+set capacity Data Card和实现：每个seed从随机初始化训练完整264,134参数模型，C01--C06提供全部梯度与fit-only权重，C07--C08只选择minimum set loss checkpoint和固定`0.05:0.05:0.95` confidence；禁止C09/C10/M-TARE。主门保持同类型3D `<=4m` precision/recall/F1、多事件recall和matched position MAE，并与旧互斥center、旧冻结encoder set及非学习几何baseline同表比较。若joint模型仍不能超过旧互斥F1至少5个百分点或多事件召回无实质提升，停止该主表示，不进入graph。

## 39. 2026-08-28 当前权威更新：GeometryAnchored joint capacity正式FAIL

唯一正式run `gate3_20260828_gse_geometry_anchored_joint_capacity_v1_seed0`按冻结Data Card完成三个seed各8轮、`9,096`步，完整训练`264,134`参数，总计`27,288` optimizer steps。C01--C06提供`142,184`行和`98,279` tokens的全部梯度，C07--C08提供`45,942`行和`33,145` tokens的checkpoint/固定网格选择；C09/C10/M-TARE、graph和planner零读取。三个训练进程正常结束，输入未变化，`error=null`。

三seed同类型4m匹配P/R/F1为`0.0831/0.0889/0.0859`、`0.1389/0.0701/0.0932`、`0.1108/0.0496/0.0685`，macro-F1=`0.0944/0.0888/0.0755`。同一Observable Teacher V2人口上，旧互斥center baseline F1/macro-F1=`0.3901/0.4010`，最佳冻结encoder set F1=`0.1869`，非学习几何F1=`0.0032`。三seed只有matched position MAE=`2.440/2.231/2.512m<=4m`通过；precision、recall、分类型recall、F1/macro-F1提升和multi-event recall门全部失败。

正式结论为`MODEL_OBJECTNESS_AND_PROPOSAL_GENERATION_FAILURE`：显式polar/free-range anchor保留部分局部定位能力，但16个自由query配合加权presence集合损失不能可靠决定“哪里真的存在结构事件”。不得增加训练轮数、扩大threshold网格、调loss权重、改Teacher或直接进入graph。下一唯一工作是零训练、零阈值选择的C07--C08输出归因，固定分解空帧误报、query重复、事件类型错配、角度/径向误差、cardinality和assignment；根据证据只允许一个最小proposal/objectness corrective。

31项seal逐项零mismatch，SHA=`ded71ce5e58f1dff71bec150e3a6d5a32547e87e54733b71e425dd01b2f40c38`；论文失败图SHA=`600895c5f481a061a3ed7ed94ed82f6faca038b2dd4c611f35614219e896e697`。

## 40. 2026-08-28 当前权威更新：自由query空间proposal被正式否定

V1只读归因在全部匹配完成后因population cardinality的NumPy `uint64`安全转换错误退出，0科学结论，永久封存为system FAIL。V1R只增加显式`int64`转换并补真实类型测试；C07--C08 `45,942` rows、`33,145` targets、三seed outputs、0.95 threshold、4m matching/NMS、baseline和四条决策门均未改变。

V1R结果排除便宜corrective。固定4m同类型NMS只删除`72/71/2`个候选，F1几乎不变；全部16 query的typed one-to-one oracle recall仅=`0.122824/0.125147/0.116217`，position-only oracle=`0.123518/0.128556/0.120984`，均未达到旧互斥baseline recall `0.262905 + 0.10`。三个seed分别有`29,051/28,884/29,135`个Teacher targets没有4m内位置候选；最近同类型候选的方位中位误差=`75.18°/64.23°/67.32°`，径向中位误差=`14.99/10.97/15.47m`。0--4m覆盖接近1，超过4m后急剧下降。

因此唯一决策=`STRUCTURED_POLAR_PROPOSAL_REQUIRED`。下一表示不再使用16个learned free queries和attention加权方向向量；改为在180个显式LiDAR方位bin上产生dense eventness/type/range/elevation，候选由确定性 circular local maxima/top-k选出，方位和距离只允许相对对应bin/free-range的有界残差。这样几何坐标不会因扩散attention向量相消而塌缩。

不得立即训练。下一唯一任务是C01--C08 Observable Teacher V2的polar-slot可行性审计：量化同bin和±1-bin事件冲突、最小方位分离、各bin event cardinality、现有五帧local free-range support及2度circular roll索引确定性。只有全部131,424 tokens可无损分配且无不可解冲突，才实现dense proposal readiness和新训练Data Card。V1R 15项seal SHA=`2b9e30d94929be05ec1bbd8753afb86f188af2c3ccefdd41d566e87e9f39d056`，图SHA=`58dbbf48a9b0b2100cc8227fce29eb1e4db9761918753ba3808513de4c6bc537`。

## 41. 2026-08-28 当前权威更新：polar Teacher要求两个radial-depth slots

正式可行性run完整审计`188,126` observations和`131,424` tokens。2度量化内same-azimuth pairs=`277`，±1-bin pairs=`801`；same-azimuth-and-elevation pairs=`251`。单帧同azimuth与同azimuth-elevation occupancy最大均为`2`，最小真实事件角分离仅`0.004788°`。冲突样例表明同一视线可以同时包含约4m junction与约42m terminal，所以这不是量化噪声，而是论文需要表达的层级结构关系。

180-bin单槽会丢277对，180×4 elevation单槽仍会丢251对；二者均被否定。最小无损接口固定为`180 azimuth bins × 2 radial-depth slots`：每个bin内Teacher事件按距离升序分配slot0/1，方位中心固定并只学习±1度残差，elevation连续回归，range为对应五帧local free-range的bounded fraction。在线从360 slots按confidence选top-k，不使用neighbor NMS；输出必须保留azimuth bin和depth slot provenance。

全部tokens量化残差`<1°`，34度旋转索引mismatch=`0`；sealed readiness的unsupported=`0`、minimum support margin=`2.2888e-5m`继续有效。因此下一唯一任务是实现dense Teacher rasterizer、`StructuredPolarMultiDepthEventEncoder`和零训练readiness。必须证明131,424 tokens无overflow地round-trip、rotation equivariance、free-range bound、top-k determinism、finite backward及range/valid-only接口，之后才允许新的三seedData Card。

正式run 16项seal SHA=`8190da57f2e85ba2ed8beaebf7e6f99beb860d0862b9aa9ff4c381864b5b2505`；论文图SHA=`a42b42f9fd42ab556ed24807ba6c6b0978266bb34f5cd3e483c002e4b88a08fa`。

## 42. 2026-08-28 当前权威更新：StructuredPolarMultiDepth readiness正式PASS

新`StructuredPolarMultiDepthEventEncoder`固定`172,430`参数，只输入五帧`range_m/valid_mask`。它输出180方位×2深度的360个dense slots，每槽包含presence、terminal/junction type、free-range bounded radial fraction、±1度azimuth residual、±15度elevation、64维descriptor和uncertainty；在线兼容接口从360 slots按confidence确定性取top-16，不做neighbor NMS并保留bin/slot provenance。

全量rasterizer审计精确覆盖80 worlds、188,126 observations、131,424 tokens和1,076 identities。277个同方位第二深度token全部进入slot1，overflow=`0`；最大xyz round-trip误差=`2.9281e-5m`，最大radial fraction=`0.9999995`，最大azimuth residual=`0.999981°`。NumPy/Torch free-range profile parity=`0`。

真实cardinality 0--5与same-ray pair batch完成finite backward。20度rotation的dense logits/dense xyz/top16 xyz误差=`7.0781e-8/1.0967e-5m/6.6757e-6m`，bin/slot shift exact；range/azimuth bound、top16 uniqueness、repeat和seed replay逐位一致。所有训练前门PASS，0 optimizer/trained inference/threshold selection及C09/C10/M-TARE读取。

下一唯一任务允许为该dense接口冻结一次三seedcapacity Data Card和实现：C01--C06直接slot监督，C07--C08 minimum dense loss checkpoint与固定0.05 confidence grid；评估仍使用相同type+4m one-to-one set指标，并同时比较exclusive center、旧free-query joint、旧frozen set和nonlearning。若不能三seed超过旧exclusive F1/macro-F1至少5个百分点、precision>=0.90及multi-event recall实质提高，则停止进入graph。

正式run 11项seal SHA=`34a55cc0728de7a6835545a1732bcc7d135a71b7d3fee67eb6d918adf29a8736`。

## 43. 2026-08-28 当前权威更新：StructuredPolarMultiDepth capacity正式FAIL

唯一正式run `gate3_20260828_gse_structured_polar_multidepth_capacity_v1_seed0`按冻结Data Card完成三个seed各8轮、`9,096`步，总计`27,288` optimizer steps；完整训练`172,430`参数的180方位×2距离槽模型。C01--C06 fit=`142,184` observations/`98,279` tokens/`198` second-depth tokens，C07--C08 selection=`45,942/33,145/79`。三个训练进程和统一评估均正常结束，source unchanged、`error=null`，C09/C10/M-TARE/graph/planner零读取。

三seed在固定type+4m一对一合同下P/R/F1=`0.137586/0.212159/0.166922`、`0.103357/0.238045/0.144133`、`0.129531/0.165606/0.145364`；macro-F1=`0.175017/0.162460/0.163409`，matched position MAE=`2.007/2.007/2.053m`。最佳共同人口baseline F1/macro-F1=`0.390133/0.401037`。三seed均只有定位门通过；precision、recall、junction recall、F1/macro-F1和multi-event增益失败。

79个同方位远目标的总体recall=`0.088608/0.506329/0.240506`，但该指标当时未保留“由哪个预测depth slot完成匹配”的语义，不能单独证明双距离槽被利用。总体仍产生`44,078/68,447/36,887`个false positives。该容量run的直接结论只限于：固定方位坐标恢复约2m定位能力，但当前dense presence不能可靠区分360个槽中的真事件；双槽学习有效性由后续V1R归因裁决。

不得进入graph、C09/C10或planner，也不得直接加epoch、扩大threshold、调loss、slot数或top-k。唯一下一步是对sealed C07--C08 top16 outputs做零训练、零新推理的objectness attribution：固定量化continuous confidence separability、candidate oracle recall、slot0/slot1误报、同bin双槽误报、空帧cardinality、range/type分层和高precision可达性。只有proposal oracle明显超过baseline且存在无新超参的可分离机制时，才允许一个最小objectness corrective；否则停止dense event路线并回到可执行exit/action token作为稀疏结构事件。

33项seal逐项复核PASS，seal SHA=`3186e311508a82fa5e6a8342806729ee4c9335cb9407b322f6420c2cf73503ee`；论文失败图SHA=`a16582801d99e9dd6e500061216e8fd797dd4fcb80bb32d3c812c738ebe85ac6`。

## 44. 2026-08-28 当前权威更新：depth-slot语义归因停止structured-polar路线

V1 objectness归因系统PASS并证明all-top16 typed oracle recall=`0.674672/0.685352/0.619038`，高于exclusive recall `0.262905+0.10`；但candidate AP仅`0.060851/0.055909/0.051180`，candidate recall约0.25时precision仅`0.081500/0.061346/0.066669`。one-per-bin不改变F1，Teacher-cardinality上界F1仅`0.257505/0.227063/0.231015`。V1初步决定objectness refit。

随后发现V1 oracle没有验证预测depth-slot语义，故不得直接据此重训。V1R保持全部输入和匹配不变，只为每个匹配保留预测slot provenance。三个selection archive各有`735,072`个top16候选；slot1候选数=`0/16/0`，seed1仅16个slot1且confidence中位数=`0.021324`。0.95正式阈值slot1选中数=`0/0/0`。

更决定性地，全部typed oracle匹配中slot1贡献=`0/1/0`；79个second-depth targets由slot1匹配的数量=`0/0/0`。同方位近/远目标同时匹配`18/7/4`组，但全部由两个slot0候选完成，真正distinct slot0+slot1=`0/0/0`。因此高oracle recall不是双距离表示学习证据，V1的`FROZEN_GEOMETRY_OBJECTNESS_REFIT_REQUIRED`被正式取代。

当前决策固定为`STOP_STRUCTURED_POLAR_MULTIDEPTH_ROUTE_USE_EXECUTABLE_EXIT_TOKENS`。不再进行slot-permutation/objectness三seed重训；该路线同时存在slot collapse与置信度不可分两个失败，已经超出最小corrective。保留Teacher可行性、readiness、capacity与V1/V1R图作为表示消融和失败分析。

下一唯一任务回到Factorized GSE-Graph的可执行语义，不回到单纯出口方向：只读构造C01--C08 full exit/action geometry token causal-event feasibility。输入状态必须包含完整出口方向/开口宽度/垂直轮廓、局部轴线、width/height/slope/curvature及uncertainty；Teacher事件由当前可执行动作集合的birth/death/split/merge和junction/terminal commit定义。先证明5帧因果可观测性、事件唯一性、正负支持、family覆盖和与固定距离/summary变化点的差异，PASS后才允许新模型Data Card。edge仍只由真实穿越建立；C09/C10/M-TARE/graph replay继续禁止。

V1 17项seal SHA=`d0ad99d0dfeeb57d6ad7c4d947d286623d46a92b0e415bef2129f71abd6ac925`，图SHA=`8771cc9b52fa28f94298dfb73c169cef0f7730eea3989ced2b67496f116afeb6`。V1R 17项seal SHA=`8c9cf3714d8ceca617c57905573c8b7e86f7127dd2bcfa90accb84ab2d9d5ae5`，图SHA=`4fd94ed836e030e2a949f98b74cb2473082053202a87c2a8834ca0af173ba263`。

## 45. 2026-08-28 当前权威更新：可执行exit/action token transport可行性正式PASS

唯一正式run完整验证C01--C08的80个dataset shard tree，覆盖`252,430`个原始帧、`188,126`个五帧因果观测和`396,913`个可见directed-exit tokens。C01--C06 fit=`142,184`观测/`130,080`相邻对/`3,282` decision episodes；C07--C08 selection=`45,942/41,970/1,136`。全部输入source unchanged，C09/C10/M-TARE、optimizer、新模型推理、threshold selection和graph replay均为0。

可执行可见出口集合的保守规则（1个出口支持terminal、2个保留corridor、3个及以上支持junction）在C07--C08得到decision macro-F1=`0.995411`；`1,120/1,136=0.985915` decision episodes至少包含一帧明确可执行支持。该结果说明objective junction/terminal绝大多数可以由局部行动集合定义，不需要预测远程TNG节点；剩余16个episode必须由后续uncertainty/refusal保留为provisional，禁止强制提交。

对每个seed先用冻结训练一致的heading+width cost把可见Teacher identity附着到预测token（只用于评分），再完全不使用identity、仅用32维预测descriptor在相邻帧做一对一transport。C07--C08三seed precision/recall分别为`0.981889/0.988804`、`0.983521/0.990447`、`0.984711/0.991645`，全部通过0.98门。证明旧模型已经学到稳定的物理出口表征，失败不等于LiDAR没有结构信息。

对照结果保持不变：固定5维summary change point precision/episode recall=`0.552632/0.031111`；旧ActionSetNodeDetector在每帧先mean/max池化后macro-F1/precision/episode recall=`0.743847/0.965870/0.498239`，未满足0.995安全合同。因此下一模型不得再次逐帧池化完整集合。

当前唯一允许的实现为`RelationalExitTokenTransportEventModel`：保留5帧×3seed×6token，显式构造相邻帧token affinity/transport矩阵，用set/temporal attention聚合persistent、revelation、withdrawal、route-stop及几何上下文；输出junction/terminal/provisional commit和uncertainty/refusal。Teacher只监督episode内至少一次正确commit；edge仍仅由真实穿越创建。下一步先做零训练readiness和exact Data Card，禁止直接重训旧pooling head或进入graph/C09/C10/M-TARE。

正式run 17项seal逐项PASS，seal SHA=`10fa5df309c81d11254f3302280200fe19b7b77967f70b49f1f66f62da5f2f23`；论文图SHA=`86a7a8aff0b62f1f985155a413f697439fcd12a684c05aff9554fcb972089b5a`。

## 46. 2026-08-29 当前权威更新：RelationalExitTokenTransport readiness V1R正式PASS

V1使用不存在的`torch.flatnonzero`在真实batch loss前退出，未产生科学结论；0 optimizer/C09/C10/M-TARE，12项seal SHA=`9f4dcb8b6a3fe165d60e52155435d78f0ebc8053e112dd574764f34825dc89f9`。V1R仅替换为严格等价的`torch.nonzero(..., as_tuple=False).flatten()`并绑定原FAIL，不改变模型、数据、seed、检查或容差。

`RelationalExitTokenTransportEventModel`固定`240,101`参数。输入为`[B,5,3,6,40]`完整raw exit tokens、`[B,5,3,8]`最终learned axis/width/height/slope/curvature/uncertainty和past-only mask；输出corridor/junction/terminal、commit、provisional、uncertainty及4组`[3,6,6]`相邻帧soft transport。Teacher identity、event、world、pose、future和objective geometry不进入forward。

全量`188,126`引用中past-only same-traversal suffix违规=`0`，history长度1--5。真实8行batch上token permutation event/commit误差=`2.9802e-8/5.9605e-8`，seed permutation=`0/0`，masked history=`0/0`，repeat=`0/0`。transport归一化误差=`1.1921e-7`且std=`0.010695`，非均匀退化；event sum误差=`5.9605e-8`，commit+provisional误差=`0`，真实episode MIL loss=`3.583166`并finite backward。

readiness 11门全部PASS，允许下一步冻结一次三seedtraining Data Card。训练只能使用C01--C06梯度，C07--C08选择checkpoint和一个固定候选threshold grid；主要安全门必须保留aggregate precision`>=0.995`、junction/terminal precision`>=0.99`、episode recall`>=0.25`，并相对旧pooling macro-F1至少提高5个百分点。任一seed/ensemble不能满足时停止该事件模型，不进入graph/C09/C10/M-TARE。

V1R 17项seal SHA=`853b040875f3184e4c3389402655198e4bd13c43533d0e23d73e69861f32d01a`；readiness图SHA=`dd9942c61ddf96164f9526e989d7239babb08ec6bd51b10f76f7fc8a5ca94b50`。

## 47. 2026-08-29 当前权威更新：关系模型学到事件，但无状态commit接口正式FAIL

唯一正式run完成三个seed各6轮、`6,666`步，总计`19,998` optimizer steps；C01--C06=`142,184`观测/`3,282`事件，C07--C08=`45,942/1,136`，模型参数`240,101`。三个训练、评估和seal系统正常，`error=null`、source unchanged，C09/C10/M-TARE/graph/planner零读取。

单seed在统一0.985阈值的macro-F1=`0.827639/0.891699/0.838536`，说明关系表示存在真实学习信号；但precision=`0.869691/0.913255/0.732331`且校准不一致。概率均值ensemble在可满足最低recall的最佳precision点只有precision/recall/macro-F1=`0.946612/0.405810/0.664556`，低于旧pooling macro-F1=`0.743847`，没有任何固定阈值满足0.995 aggregate和0.99 per-event precision。

487个ensemble triggers中461正确；26个错误分解为18个同一Teacher episode内的重复trigger、5个corridor false trigger和3个event-type错误。当前失败分类为`STATELESS_COMMIT_DEBOUNCE_AND_UNCERTAINTY_REJECTION_FAILURE`，不是Teacher/data/system failure，也不能据此声称主方法有效。

禁止进入C09/C10、graph或planner，禁止通过降低precision门、挑单seed或重训掩盖。下一唯一任务是对冻结C07--C08输出做零训练、past-only commit-policy feasibility：显式检查稳定触发、短间隙去重、seed consensus/disagreement refusal和provisional状态，并以world-held-out方式审计，不修改checkpoint。只有预注册状态机同时恢复安全门、recall和相对pooling增益，才允许冻结一次正式commit-policy；否则停止该事件模型。

正式run 31项seal SHA=`de4cb8d37bc834236e6a1739b7e2d8d91f963a2299d7dcd51e42b7a7605aad6f`。

## 48. 2026-08-29 当前权威更新：手写commit状态机不足，转向结构化exact-one学习

正式零训练审计在C07=`21,548`观测/`533`事件上枚举每类336个固定past-only策略，再将唯一选中策略不变应用到C08=`24,394/603`。策略为junction/terminal均2-of-3 consensus与3帧稳定；概率阈值分别0.9/0.7，release均2帧。

状态机解决了主要安全问题：C07 precision/false/recall=`0.996241/0.003759/0.497186`，C08=`0.997093/0.002907/0.568823`；两侧duplicate均从18降至0，仅各剩1个corridor false。C08 macro-F1=`0.796339`通过0.793847门，但C07 macro-F1=`0.753519`失败，主要受junction F1=`0.547038`限制。不能用C08单侧通过、扩大grid或降低+0.05门宣称成功。

决策=`STATE_MACHINE_INSUFFICIENT_REQUIRE_STRUCTURED_ONE_COMMIT_LEARNING_OR_STOP`。选择有明确因果依据的结构化学习而非停止全部研究：V1 MIL只最大化episode中最高一帧，完全不惩罚同episode其他高峰；下一V2 loss必须直接最大化“整个positive episode恰好一次正确class commit且其余帧不commit”的概率，并对corridor要求零commit。它不增加Teacher、未来帧或推理输入，也不通过规划器补偿。

下一唯一任务是零训练structured exact-one loss readiness：证明单正确峰优于零峰/双峰/错误类峰、episode permutation不变、极端概率数值稳定、真实完整episode batch finite backward，并保持240,101参数模型、数据、split和C09/C10/M-TARE隔离。readiness失败则停止；PASS后才允许新的V2三seedData Card。

正式审计16项seal SHA=`bc8771dd710f33053531f47202c4ee0a15482b58273469307a5584306fadaa47`；论文图SHA=`ab5b993bc1fb38622d8183242817a6267005f2dc2ce1df9dd6cc33b9f16658fa`。

## 49. 2026-08-29 当前权威更新：structured exact-one loss readiness正式PASS

正式readiness保持`RelationalExitTokenTransportEventModel`架构和`240,101`参数完全不变，只新增参数无关的log-space episode likelihood。corridor要求零junction/terminal commit；每个positive episode要求恰好一次正确class commit，且其余全部为no-structural-commit。

合成证据中one-correct/zero/two-correct/wrong-class loss=`0.049946/3.546859/3.264189/4.596299`。旧max-MIL对第二个non-max positive peak的commit梯度精确为0，新loss梯度=`0.831963`，直接修复已观察到的重复峰监督缺失。episode row permutation error=`0`；±40 extreme logits loss/gradient finite。

真实C01--C06 deterministic 128-row complete-episode batch包含23 positive rows、105 corridor rows和3个完整episodes；新loss=`2.827224`并对全部参数finite backward。全量人口仍为`188,126`，fit/selection=`142,184/45,942`；0 optimizer、trained inference、checkpoint、threshold、C09/C10/M-TARE/graph/planner。

当前决策=`ALLOW_STRUCTURED_EXACT_ONE_EVENT_THREE_SEED_DATA_CARD`。下一训练保持架构、输入、Teacher、batch、优化器、epoch和安全门，唯一科学变量为V2 loss。为增强开发证据，C01--C06提供梯度，C07只选minimum exact-one NLL checkpoint和commit policy，C08只进行一次不适配迁移；C09/C10/M-TARE继续禁止。若C07或C08不能同时达到原安全门与macro-F1>=0.793847，则停止V2，不进入graph。

正式readiness 15项seal SHA=`ced78ead9cf7773f51ee698d4ae4ae6f6b19a631530054797e151b9c47b07426`；图SHA=`16a79c5554d519426e002fd9ef0469690853f844ee2402781667c603c0cfb0ca`。

## 50. 2026-08-29 当前权威更新：exact-one训练过度拒绝，停止直接事件分类路线

唯一正式V2 run完成三个seed各6轮和`6,666`步，总`19,998` optimizer steps；C01--C06 fit=`142,184`，C07 checkpoint/policy=`21,548`，C08 one-shot transfer=`24,394`。三个checkpoint分别选epoch `4/4/5`，C08 checkpoint/policy更新均为0，source unchanged、error=null、C09/C10/M-TARE/graph/planner零读取。

V2 stateful C07 precision/recall/macro-F1=`1.0/0.332083/0.573256`，junction/terminal recall=`0.264423/0.572650`；C08=`0.990783/0.356551/0.582755`，junction/terminal recall=`0.300429/0.547445`。C07 F1失败，C08同时安全与F1失败。相较V1 stateful C07/C08 F1=`0.753519/0.796339`，exact-one虽然保持duplicate=0，却显著过度拒绝，不能作为主方法。

当前决策=`STOP_DIRECT_JUNCTION_TERMINAL_EVENT_CLASSIFICATION_ROUTE`。不再调整loss权重、epoch、grid、seed或状态机。V1/V2 checkpoint、状态策略和两张失败图只保留作事件头/结构化loss消融。

下一主方法回到更可解释的几何结构关系：不直接监督junction/terminal分类，而是学习每个出口token是否是真实可执行结构、跨帧保持物理出口track，并由稳定action-set cardinality/关系确定节点。输入仍包含heading、opening width、vertical profile、descriptor及局部axis/width/height/slope/curvature；边仍只能由真实穿越建立。

下一唯一任务是零训练C01--C08 token-validity/action-set-track feasibility：为每个预测token生成evaluation-only Teacher validity/identity，量化正负规模、confidence与几何关系可分性、descriptor track稳定性、stable-track cardinality对junction/terminal的上界及与V1/V2事件头的差异。C09/C10/M-TARE继续禁止；只有稳定track action set有明确容量才允许新的token-validity Data Card。

正式V2 run 32项seal SHA=`51d3a8964ca7c556ccafaad93ee1392804ec429273f73db497cf6b791da3306b`；失败图SHA=`3b2c99f55e89d375f03ae30f2a202a4e66955859702a75895733084749f06156`。

## 51. 2026-08-29 当前权威更新：free-query token validity不可修复，改为连续圆周可通行几何场

V1 Data Card在run创建前因validation world字符串不符合治理schema而preflight停止，未创建run。V1R只修Data Card表达并正式创建，但执行器把已经是inner summary的V2文件再次索引`result`，在计算指标前system FAIL；该run永久保留且无科学结论。V1R2只增加outer/inner summary兼容读取并绑定V1R失败证据，方法、输入、门槛和人口不变。

V1R2正式覆盖80个C01--C08 worlds、252,430 raw frames和188,126 causal observations。每个seed的6-query人口精确为1,128,756 slots，其中Teacher附着positive/negative=`396,913/731,843`；C07/C08 observations=`21,548/24,394`。全部shard/cache hash一致，source unchanged、`error=null`，0 optimizer、新推理、threshold fitting、graph、C09/C10/M-TARE。

Teacher action-set容量仍成立：selection decision macro-F1=`0.995411`、episode support=`0.985915`；三seed descriptor transport最小precision/recall=`0.981889/0.988804`。但冻结token有效性不可部署：selection单帧confidence在precision>=0.995时recall仅`0.093888/0.000010/0.006379`；past-only五帧或三seed几何共识的最佳recall仅`0.092435/0.000423/0.010820`，AP相对单帧最佳变化为`-0.005735/-0.002602/-0.006024`。两种corrective都没有改善，说明假query是跨帧、跨seed稳定的系统性ghost slots，而不是可由平滑或共识消除的随机噪声。

当前决策=`STOP_FROZEN_FREE_QUERY_TOKEN_VALIDITY_ROUTE_USE_DENSE_CIRCULAR_TRAVERSABILITY_FIELD`。停止六自由query的objectness/refusal/threshold训练；现有exit模型、关系事件V1、exact-one V2及本图均保留为论文消融。新的表示方向为`Dense Circular Traversability Field`：对固定圆周方位直接预测可执行free-space validity、opening geometry、vertical clearance/slope和uncertainty，再以连续方位分量确定ExitGeometryToken，避免自由query凭空生成候选；descriptor只在确定的分量上学习。

下一唯一任务是零训练C01--C08 Teacher feasibility：验证全部396,913 visible directed exits能否无损投影到固定180-bin圆周field，量化角跨度、同bin/相邻bin冲突、连通分量合并/分裂、反向traversal、坡道/stacked geometry与五帧因果一致性。必须同时比较原始LiDAR几何阈值基线。任何不可唯一表达或Teacher依赖未来/identity才能解码的情况都停止该表示；通过后才允许Teacher导出Data Card。C09/C10/M-TARE、模型训练、graph和planner继续禁止。

正式V1R2 run 17项seal SHA=`41c23bdf50f90cfec427d93add4ac7e6eecc001c41fb2f73655643c105986729`；论文诊断图SHA=`2d176ed727b6207ef5bdf7fc0e0565d61022551bec498ed90176bb8fa36c3236`。

## 52. 2026-08-29 当前权威更新：Circular Executable-Geometry Peak Field可行性正式PASS

只读预诊断否定了原“宽开口扇区→连通分量”语义：188,126个观测中`22,626`行出口数不一致，`22,381`行不同出口扇区重叠；有效开口角跨度median/p90/p99=`117.762°/140.506°/174.696°`。因此第51节的connected-component解码被本节明确取代，不再把重叠开口当成单个出口。

新表示固定为180个2°方位bin的`Circular Executable-Geometry Peak Field`。每个visible directed exit在最近中心bin产生一个peak，并在该peak保存signed sub-bin residual、opening width及独立valid mask、四值vertical profile；出口由圆周局部极大值产生，不由宽扇区连通分量或free query产生。全部396,913 exits在188,126行上same-bin collision=`0`且adjacent-bin collision=`0`；cardinality 1/2/3/4=`7,525/154,279/24,458/1,864`。

初次V1科学run除round-trip门外全部通过，但评估器对接近357°的float32单位向量使用了与encoder不同的角度精度，在半bin边界读取相邻bin，伪造heading/width/profile误差。V1作为system/metric FAIL保留。V1R只新增并共享唯一float64 `heading_unit_to_bearing_bins`合同，绑定V1证据；数据、180 bins、Teacher、基线和门槛不变。

V1R正式PASS：heading round-trip最大误差=`2.9802e-8°`，width/profile误差=`0/0`；slot permutation mismatch=`0`；20°rotation presence/geometry mismatch=`0/0`，residual最大误差=`5.3262e-6°`；全部五帧local/global reference连续且future violation=`0`。全部source hash一致、source unchanged、`error=null`，0 Teacher export、optimizer、新推理、threshold selection、graph、C09/C10/M-TARE。

当前raw-range envelope非学习基线在fit/selection的peak AP仅`0.052647/0.051493`，precision>=0.995时recall均为0。该结果说明Teacher表示无损，但不能声称原始距离阈值已经解决出口语义；后续学习必须在同一人口上显著超过该基线，并同时满足安全precision/recall。

当前决策=`ALLOW_CIRCULAR_EXIT_GEOMETRY_FIELD_TEACHER_EXPORT_DATA_CARD`。下一唯一任务是Gate-2 Teacher导出：按80个world独立Zarr shard保存180-bin presence/residual/width-mask/profile、global sequence与五帧references，保留全部width-invalid peaks，不复制range frames；导出必须逐shard往返、旋转和source join一致，且只生成一次。导出PASS后才允许模型readiness/Data Card；C09/C10/M-TARE、训练、graph和planner继续禁止。

正式V1R 17项seal SHA=`ddad17c78f26ec05754c35ed88dc640473f30e9ea7c0b021995b2ed02fab663e`；论文图SHA=`360d2a8539941380b2f336281cb50328e1217f96648cad2b5b6e80fc334444ec`。

## 53. 2026-08-29 当前权威更新：Circular Peak Field Teacher正式导出PASS

唯一正式Gate-2运行按world独立导出80个Teacher shards。人口精确为188,126个因果观测、396,913个exit peaks；C01--C06 fit为60 worlds/142,184 observations/299,872 peaks，C07--C08 selection为20/45,942/97,041。宽度有效/无效peak=`389,026/7,887`，每行1/2/3/4个peak的观测数=`7,525/154,279/24,458/1,864`。

每个shard只保存180-bin presence、sub-bin heading residual、opening width及独立mask、四值vertical profile、exit count和past-only sequence/frame references。没有复制LiDAR、valid scan、pose、node/exit identity或未来target。80个shard全部重新打开并逐数组`array_equal`；最大heading round-trip误差=`2.9802e-8°`，source unchanged，全部禁用操作为0。

Teacher压缩数据实际仅`22.243 MiB`，完整run为36 MiB，远低于1 GiB上限。独立复核2,949条seal全部匹配，`error=null`，正式决策=`ALLOW_CIRCULAR_EXIT_GEOMETRY_FIELD_MODEL_READINESS`。

当前返回Gate 3但仍不允许直接训练。下一唯一任务是零训练model readiness：固定一个五帧因果circular range-image encoder与180-bin dense peak heads，证明输入/输出shape、圆周旋转等变性、past-only mask、presence不平衡loss、peak-only masked geometry loss、无identity/future forward和真实batch finite backward。PASS后才允许精确训练Data Card；C09/C10/M-TARE/graph/planner继续禁止。

正式run=`results/gate2_representation/gate2_20260829_gse_circular_exit_geometry_field_teacher_export_v1_seed0`；2,949项seal SHA=`8c37f11a9e08b3cf0c48d005ae32aa2d79afcbba7c2330333ee9996605657f76`；Teacher图SHA=`044cccf8b8e8fe68e24446ae7b371a2662c4bfb270f03fd3a5f456d655cc0780`。

## 54. 2026-08-29 当前权威更新：Circular Peak Geometry Model readiness正式PASS

正式readiness覆盖全部80 worlds/188,126 observations/396,913 peaks的Teacher-source join及五帧reference，并从真实数据确定性选择8行，覆盖1/2/3/4出口、wrap bin、width-invalid及fit/selection。全部local/global references连续且past-only，Teacher中无LiDAR、pose或identity数组。

新`CircularPeakGeometrySemanticNet`固定769,268参数，输入仅`[B,5,2,16,720]`因果LiDAR；没有free-query参数。输出180-bin peak presence、sub-bin heading、opening width、vertical profile、32D descriptor、六维geometry uncertainty，以及local axis、width/height/slope/curvature、128D place descriptor和observation uncertainty。它不再把出口方向冒充完整结构语义。

真实8行batch包含20个peaks和2个width-invalid peaks；所有输出、五项loss及全部parameter gradients finite。invalid-width和nonpeak profile任意扰动使total loss变化精确为0。40-column circular roll的dense/axis/global最大误差=`9.5367e-7/9.4771e-6/3.3379e-6`，batch permutation误差=`9.5367e-7`，repeat exact；改变最老历史帧使peak logits变化`0.033266`，证明五帧路径连通。

当前决策=`ALLOW_CIRCULAR_PEAK_GEOMETRY_MODEL_TRAINING_DATA_CARD`。下一训练必须从头开始，不复用曾读取C07--C08的旧checkpoint：C01--C06提供梯度，C07只选择checkpoint和唯一threshold，C08只做一次零适配迁移。三个seeds固定0/1/2；raw-range和旧free-query模型保留作对照。必须预注册高精度peak detection、action-set cardinality、heading/width/profile及axis/width/height/slope/curvature指标；失败不得用graph/planner补偿。

正式run=`results/gate3_semantics/gate3_20260829_gse_circular_peak_geometry_model_readiness_v1_seed0`；17项seal SHA=`d06089680b97544a251e30e9c0214fa6c14df31fb3d30ef0c3d97852adc74c69`；readiness图SHA=`38aa751b6f67a0a19c8cf5e6a5373fc1835eb85a5a6c055b33034c49b3b954ab`。

## 55. 2026-08-29 当前权威更新：Circular Peak Geometry V1训练学习连续结构但硬峰接口正式FAIL

唯一正式run完成3 seeds × 10 epochs，每seed 11,370、总34,110 optimizer steps。best epochs=`8/9/8`，C07 masked losses=`-2.101174/-2.078040/-2.107434`；784,266 development inference observations，source unchanged、`error=null`，C08 checkpoint observations=0，C09/C10/M-TARE/graph/planner均0。

连续结构输出存在真实泛化：ensemble C07/C08 axis mean error=`4.5309°/4.7078°`，width MAE=`0.7206/0.7347 m`，height=`0.4001/0.5407 m`，curvature=`0.004823/0.004699 per m`；这些通过原绝对门。slope MAE=`3.1404°/3.2321°`，未通过2°门。

主要失败是硬peak objectness。2°exact local-peak AP仅`0.069034/0.057519`，三single-seed同样失败；没有任何threshold同时满足precision>=0.995，因此selected peaks=0、recall/action F1=0，peak geometry不可评分。正式决策=`STOP_CIRCULAR_PEAK_GEOMETRY_BEFORE_GRAPH`，不得进入descriptor association或graph。

只读归因显示网络并非没有方位信号：dense exact-bin AP=`0.196756/0.180411`；最近local maximum相对Teacher中心的offset median/p90/p99=`1/4/8 bins`，即`2°/8°/16°`。允许one-to-one角度匹配后，±4° AP=`0.652/0.615`，±10° AP=`0.900/0.899`。但每帧平均约19.7个local maxima，高分假峰仍破坏0.995 precision。top-4+NMS只有在约20°容差时才恢复>0.5 safe recall，不能作为V1成功或只改评分口径。

失败分类为`SINGLE_BIN_HARD_TARGET_AND_EQUAL_MASS_BCE_HIGH_CONFIDENCE_GHOST_PEAK_FAILURE`。下一唯一任务是正式零训练failure attribution/readiness：冻结V1 predictions，只读量化offset、tolerance AP、NMS/top-K极限、负峰score tail与soft angular target+hard-negative ranking loss的数学/真实batch梯度。V2必须保持已有效的axis/global geometry backbone和split，只改变peak target/loss；readiness PASS后才允许一次V2 training Data Card。

正式V1 run=`results/gate3_semantics/gate3_20260829_gse_circular_peak_geometry_three_seed_training_v1_seed0`；88项seal SHA=`5477c4b7ddc32313d186e1a8a5b7dbb0dc8867cf9b52bb1b2fffb737b87ef018`；失败图SHA=`4569ea433eba92c73ab2e47a244e3d1231e190c282150cd94cf599ec41756289`。

## 56. 2026-08-29 当前权威更新：Soft Angular Peak + Hard Negative V2 readiness正式PASS

正式零训练readiness精确复现V1的C07/C08 exact local-peak AP=`0.069034/0.057519`，证明归因没有更换人口或评分口径。冻结输出显示最近真峰偏差p50/p90/p99=`2°/8°/16°`，±10° one-to-one AP=`0.900297/0.898578`，但仅靠top-4/NMS仍需约20°容差才能得到可用safe recall，故禁止把放宽匹配当成修复。

V2只替换presence supervision：半径4 bin（由C07-only p90固定）的圆周soft heatmap、focal loss，以及等数量最高分非支持bin的pairwise hard-negative ranking。合成检查中correct/1-bin/far/hard-ghost loss=`0.000071/5.634/19.880/3.643`；真中心梯度为负、ghost梯度为正。三个冻结V1 backbone各取真实batch均finite backward，hard-negative最小抑制梯度=`0.05116`。

全部9项检查PASS，C08只做确认且没有选择半径；0 optimizer、checkpoint、新推理、threshold、graph、C09/C10/M-TARE。当前决策=`ALLOW_SOFT_ANGULAR_PEAK_HARD_NEGATIVE_V2_TRAINING_DATA_CARD`。

下一唯一任务是冻结并执行一次V2三seed训练：C01--C06提供梯度，C07选checkpoint与唯一threshold，C08仅一次零适配迁移；保持V1模型、geometry heads、seeds 0/1/2、10 epochs和连续几何门，只改变peak presence loss。必须继续报告严格exact-bin指标，不能用宽容差宣称PASS。

正式run=`results/gate3_semantics/gate3_20260829_gse_soft_angular_peak_hard_negative_readiness_v1_seed0`；17项seal SHA=`1badc88c95982af3dd38e10199fb936d20b0cfc550fc67eaf8f00d3292a1072c`；论文诊断图SHA=`04a98075951f94c41dfa5d04cd4ac65818895669d49e6ceb7504b03d2a1bbf4f`。

## 57. 2026-08-29 当前权威更新：V2提升排序但独立局部峰路线正式FAIL

唯一正式V2 run完成3 seeds × 10 epochs，每seed 11,370、总34,110 optimizer steps；best epochs均为8，C07 losses=`-1.315687/-1.349687/-1.393289`。总耗时5,255秒，source unchanged、`error=null`，C08 checkpoint observations=0，C09/C10/M-TARE/graph/planner均0。

V2相对V1确有提升：strict exact local-peak AP从C07/C08=`0.069034/0.057519`提高到`0.110522/0.097221`，但仍低于raw AP+0.10门=`0.151493`，且没有任何threshold达到precision>=0.995，故selected peak、recall和action F1均为0。连续axis/width/height/curvature仍有效，C07/C08 axis=`5.121°/5.266°`、width=`0.741/0.774m`、height=`0.392/0.536m`、curvature=`0.00421/0.00440 per m`；slope=`3.250°/3.337°`继续失败。

只读oracle-count上界进一步排除“只加count head”：使用真实每帧出口数并从V2 field做top-K circular NMS，strict命中率最高仅C07/C08=`0.2588/0.2416`；±4°才约`0.782/0.765`。因此根因同时包含cardinality和离散峰定位，不能继续调threshold、NMS或同类peak loss。

当前决策=`STOP_INDEPENDENT_LOCAL_PEAK_CLASSIFICATION_USE_CAUSAL_CIRCULAR_EXIT_SET_PROCESS`。下一表示固定为因果圆周出口集合过程：显式预测1--4出口cardinality，以连续圆周方位和集合级permutation-invariant transport/assignment联合监督出口集合；出口几何只绑定到被选择的连续元素，uncertainty支持拒绝。保留有效的causal circular backbone和global geometry heads，停止独立180-bin BCE/focal local-max threshold接口。

下一唯一任务是零训练set-process readiness：证明连续方位的wrap/rotation、cardinality、permutation invariant matching、相近多出口分离、真实1--4出口batch finite backward，以及0 identity/future/test/graph。PASS后才允许新的三seed训练Data Card。

正式V2 run=`results/gate3_semantics/gate3_20260829_gse_circular_peak_geometry_three_seed_training_v2_seed0`；88项seal SHA=`35b5efabab9559e448186a618efb4164fe9f79fe45ae63a3e868fae5d6f920ee`；失败图SHA=`3850a80c51ee989f7ab6a5feedb74ae81f739856e9de4be022418f7045ea6195`。

## 58. 2026-08-29 当前权威更新：Causal Circular Exit Set Process readiness V1R正式PASS

新接口复用五帧causal circular backbone和全部geometry heads，新增唯一128→4 cardinality head，总参数769,784；不存在free-query参数。180个方位logits不再独立二分类，而是组成总质量为1的finite-set intensity；1--4 cardinality决定输出元素数，radius-one diverse decode由Teacher same/adjacent-bin collision=0固定，推理不使用存在性threshold。

V1覆盖80 worlds、188,126 observations、396,913 exits及cardinality 1/2/3/4=`7,525/154,279/24,458/1,864`。10项方法/数据检查PASS；唯一FAIL是synthetic set-NLL旋转差`5.9605e-8`被执行器要求exact zero，而Data Card原合同为`<=3e-5`。V1作为metric/system FAIL永久保留。

V1R只把该项比较改为`<=1e-6`，数据、模型、损失、8条真实行和其他检查不变。correct/one-bin-shift/duplicate/ghost set NLL=`0.895887/16.895887/1.242459/2.290151`；正确/错误cardinality loss=`3.58e-7/16.0`。wrap附近`179/1`和最小安全间隔`0/2`均同时解码；真实1--4出口batch的set/cardinality/geometry输出与全部梯度finite。

40-column rotation的intensity/count/dense/axis/global误差最大=`1.40e-9/2.98e-8/9.54e-7/9.48e-6/3.34e-6`，batch permutation=`9.54e-7`，repeat=0，历史敏感度=`0.033266`。全部11项检查PASS，0 optimizer/checkpoint/threshold/C09/C10/M-TARE/graph/planner。

当前决策=`ALLOW_CAUSAL_CIRCULAR_EXIT_SET_PROCESS_TRAINING_DATA_CARD`。下一训练保持C01--C06/C07/C08 split、seeds 0/1/2、10 epochs、circular augmentation及geometry门；科学变量仅为normalized set likelihood+cardinality代替independent presence。评价必须同时报告continuous one-to-one bearing、cardinality/action F1、拒绝后的precision/coverage、peak geometry和global geometry；不得用oracle count或宽容差冒充部署性能。

正式V1R run=`results/gate3_semantics/gate3_20260829_gse_circular_exit_set_process_readiness_v1r_seed0`；17项seal SHA=`966a5d92e2ef81b355b809bdd0c1ae416f6734d892555dc37d3448b98bf5583b`；readiness图SHA=`8e542e0b55ae32fada098bfb71fa17ea9226339ee1be9bec0585f966682533cb`。

## 59. 2026-08-29 当前权威更新：Set Process学会结构数量但安全连续出口集合正式FAIL

唯一正式run完成3 seeds × 10 epochs，每seed11,370、总34,110 optimizer steps；best epochs=`9/8/8`，C07 validation loss=`0.791381/0.680189/0.693892`。总耗时5,235秒，source unchanged、`error=null`，C08 checkpoint observations=0，C09/C10/M-TARE/graph/planner均0。

显式cardinality有效：C07/C08 raw count accuracy=`0.967143/0.959867`，terminal/corridor/junction raw macro-F1=`0.946488/0.938677`。这证明五帧LiDAR可以学习结构类别，不支持“模型什么都没学到”的结论。

部署出口集合失败：C07在precision>=0.995下只能接受10/21,548个观测、20/45,504个出口，recall=`0.000440`；同一阈值迁移到C08只接受8/24,394个观测、16/51,537个出口，recall=`0.000310`。deployed action macro-F1约为0，完整集合覆盖远低于0.50。极少数被接受出口的bearing/width/profile误差虽为C07=`0.760deg/0.130m/0.049m`、C08=`0.647deg/0.228m/0.056m`，但该选择人口过小，不能证明普遍几何能力。全局axis/width/height/curvature仍可用，slope=`3.412deg/3.491deg`继续高于2deg门。

当前决策=`STOP_NORMALIZED_FINITE_SET_INTENSITY_BEFORE_GRAPH_DIAGNOSE_LOCALIZATION_AND_CONFIDENCE`。不得用raw cardinality成功直接建图，也不得降低0.995 precision、2deg bearing或0.50 recall门。下一唯一任务是冻结输出零训练归因：报告全人口的最优一一匹配角误差、2--20deg容差曲线、按cardinality分层、三seed一致性、mass/margin/entropy/残差置信度的正确集合可分性，以及同一预测bin上的geometry误差。归因必须区分表示定位不足与置信度排序不足；只读C07--C08，0 optimizer/新推理/C09/C10/M-TARE/graph/planner。只有证据指向明确可修机制后才允许下一候选readiness。

正式run=`results/gate3_semantics/gate3_20260829_gse_circular_exit_set_process_three_seed_training_v1_seed0`；88项seal SHA=`18a5dcb717ec7133aa0b27056fc18e844510ddbf738048ef8fd05ae481413f92`；失败图SHA=`29095d2986ae11940b4ef4ffbdf13969f8e809e02ec0ae0318d02fc96aa5221d`。

## 60. 2026-08-29 当前权威更新：冻结输出归因锁定slot transport corrective

正式零训练归因覆盖C07/C08全部`21,548/24,394`观测与`45,504/51,537`真实出口，精确复现cardinality结果且全部容差曲线单调。2deg下exit precision/recall=`0.4463/0.4446`和`0.4242/0.4217`，完整集合率=`0.2504/0.2258`；10deg下完整集合率约`0.6553/0.6520`，说明总体存在近邻方位信号但不满足正式精度。

错误高度依赖cardinality：C07/C08三出口完整集合率在10deg仅`0.0479/0.0521`，四出口为`0/0`；因此整体10deg结果被占多数的双出口走廊掩盖，单一normalized intensity存在多模式覆盖失败。selected-bin连续残差相对bin center的median改善仅`0.003/-0.003deg`，而在Teacher真值bin读取同一残差头的mean error仅`0.469/0.478deg`，证明训练位置与推理解码位置错位。10种冻结confidence/margin/entropy/seed-consensus分数在C07 0.995 precision下最佳exact-set recall=`0.001854`，迁移C08=`0.001453`，不能靠后处理置信度修复。

当前决策=`STOP_SINGLE_INTENSITY_AND_SELECTED_BIN_RESIDUAL_USE_CARDINALITY_CONDITIONED_CIRCULAR_SLOT_TRANSPORT`。新候选保留已验证的五帧circular backbone、global geometry和count head；对每个预测cardinality K生成K个sensor-azimuth anchored circular slot distributions，用permutation-invariant one-to-one transport监督覆盖每个出口，以圆周期望直接输出连续bearing，并将geometry/confidence绑定到matched slot。slot数量由count决定，不使用free-query existence/objectness；一一分配必须惩罚duplicate和missing mode。

下一唯一任务是零训练slot-transport readiness：证明1--4出口可表达、wrap/rotation、slot permutation、distinctness、连续bearing、matched geometry、confidence concentration和真实1--4 batch finite backward；0 optimizer/checkpoint/C09/C10/M-TARE/graph/planner。readiness失败则停止该decoder，不能直接训练。

正式归因run=`results/gate3_semantics/gate3_20260829_gse_circular_exit_set_process_failure_attribution_v1_seed0`；18项seal SHA=`7ff969054736b21c31d61177a5c69b87c1a02b537a1144b0ab97c3a269662b1f`；归因图SHA=`892dd013e9ce907850f2ddbfad2123a31b93202ab0e9c53a9e1fe7d5411bc4d6`。

## 61. 2026-08-29 当前权威更新：Circular Slot Transport readiness V1R正式PASS

V1执行到网络对称性审计时，通用maximum-error函数对布尔`decoded_valid_mask`做减法，Torch拒绝该运算；run在科学summary前system FAIL，12项seal SHA=`6e47b51a028064c0008e6c4682d4da8d5a278fcc36d796f3e21efc7ad46a9bc7`。V1R只将布尔差改为logical mismatch并增加0/1回归测试，绑定V1；数据、模型和科学门不变。

V1R正式覆盖80 worlds、188,126 observations、396,913 exits和1/2/3/4 cardinality=`7,525/154,279/24,458/1,864`。新模型787,328参数，0 query/objectness。合成correct/duplicate assignment loss=`4.00002/9.75002`，duplicate missing-mode gradient=`-0.09375`；sharp/uniform concentration=`0.99998/3.21e-8`，rotation和slot permutation loss error=0。

真实8行覆盖1/2/3/4出口、wrap与invalid width，全部输出/loss/gradient finite。slot mass/resultant/count/concentration/slot geometry/batch permutation最大误差均不超过`9.54e-7`，repeat exact，五帧history sensitivity=`0.06444`。bearing角诊断在低浓度随机分布上为0.0036deg，但用于合同的未归一化resultant rotation error仅`1.24e-8`，避免在方向未定义时用角度制造伪失败。

readiness后将逐样本置换参考loss机械向量化；非退化合成batch的七项loss数值误差`<=1e-6`且slot-logit gradient `allclose(atol=1e-7,rtol=1e-6)`，现有7项测试PASS。科学定义不变。

当前决策=`ALLOW_CARDINALITY_CONDITIONED_CIRCULAR_SLOT_TRANSPORT_TRAINING_DATA_CARD`。下一训练从头三seed，继续C01--C06梯度、C07 checkpoint与唯一set-confidence threshold选择、C08一次零适配迁移；必须按1/2/3/4 cardinality分别报告完整集合与bearing/geometry，不能用双出口多数类掩盖三/四出口失败。C09/C10/M-TARE/graph/planner继续禁止。

正式V1R run=`results/gate3_semantics/gate3_20260829_gse_cardinality_conditioned_circular_slot_transport_readiness_v1r_seed0`；17项seal SHA=`32ab6cb9ea65177c38ef5ef0b2eb88f61ef123943ffe740694a7d6de4cc00846`；readiness图SHA=`8cdb43a46fd1827b4f64857f143dfe86904ff7f98f28e71b887f310d587f2445`。

## 62. 2026-08-29 当前权威更新：Slot Transport改善粗粒度多出口覆盖但严格定位与拒绝正式FAIL

唯一正式run完成3 seeds × 10 epochs、总34,110 optimizer steps，三个最佳epoch=`8/8/9`，C07 validation loss=`2.334300/2.333513/2.328594`，跨seed高度稳定。总耗时5,793秒，source unchanged、`error=null`，C08 checkpoint observations=0，C09/C10/M-TARE/graph/planner均0；88项seal全部复核通过。

该candidate产生了真实但不足以部署的改善。相对上一set-process，C07/C08总体exact-set在2deg从`0.2504/0.2258`提高到`0.3413/0.2778`，10deg从`0.6553/0.6520`提高到`0.8380/0.8204`；三出口10deg从`0.0479/0.0521`提高到`0.3446/0.2853`。这证明cardinality-conditioned bijective slots显著减少模式漏失，不能把本run解释为模型完全未学习。

严格合同仍失败。三出口2deg exact-set仅`0.0130/0.0071`，四出口2deg为`0/0`、10deg仅`0.0415/0.0130`。以count confidence × minimum slot concentration做whole-set refusal时，为达到precision>=0.995，C07/C08只能接受`5/4`个观测，exit recall=`0.000220/0.000155`；confidence没有分离整组正确与错误。全局axis/width/height/curvature仍通过，但slope MAE=`3.393/3.483deg`继续失败。

当前决策=`STOP_SLOT_TRANSPORT_BEFORE_GRAPH_DIAGNOSE_STRICT_LOCALIZATION_AND_CONFIDENCE`。保留slot transport为有效粗粒度多出口表示与后续消融，但不得确立为最终感知方法、不得降低2deg/0.995门或进入graph。下一唯一任务是冻结输出零训练归因：分别比较单seed与ensemble、circular mean与slot argmax/local mode、target附近mass、多峰/熵/集中度、seed对齐误差、按cardinality与slot排序的误差，以及预注册confidence的safe precision-recall。归因必须判断失败主要来自分布解码偏差、slot分布多峰、seed融合对齐还是不可分confidence；只读C07--C08，0 optimizer/新推理/C09/C10/M-TARE/graph/planner。

正式run=`results/gate3_semantics/gate3_20260829_gse_cardinality_conditioned_circular_slot_transport_three_seed_training_v1_seed0`；88项seal SHA=`472630ffd88a1331ead2abfdaf8f3469c6e1501d97d041b2d109685253f06455`；失败图SHA=`c36d02071e5604b3a991c5d40cd29a306bf11195590a8e81095fd9b1a4b9871a`。

## 63. 2026-08-29 当前权威更新：冻结归因排除后处理并锁定Cyclic-Ordered Unimodal Slot Transport

正式零训练归因覆盖C07/C08全部`21,548/24,394`观测。它精确复现ensemble 2deg exact-set=`0.341284574/0.277773223`，所有2/4/10deg曲线单调，0 optimizer、新推理、C09/C10/M-TARE/graph/planner；16项seal全部复核PASS。

简单解码、融合和cardinality均不是主因。argmax/local-mode的最佳2deg exact-set=`0.3238/0.2858`，没有在两侧比mean稳定提高3个百分点；最佳单seed=`0.2977/0.2704`，低于ensemble；oracle count仅得到`0.3440/0.2790`，改善不到0.3个百分点。

错误来自slot distribution本身随cardinality扩散。C07/C08三出口matched bearing error中位数=`5.692/6.449deg`、四出口=`8.472/10.436deg`；对应Teacher两bin概率质量中位数从双出口`0.411/0.387`降到三出口`0.131/0.124`和四出口`0.088/0.063`。最佳inference-only confidence为negative maximum entropy，AUC=`0.8068`，但在C07选定precision>=0.995后recall仅`0.004637`，迁移C08=`0.003105`，后处理仍不足。

当前决策=`USE_CYCLIC_ORDERED_UNIMODAL_SLOT_TRANSPORT_READINESS`。新候选保留五帧circular encoder、count和global geometry；对按圆周排序的K个Teacher exits只允许K个orientation-preserving cyclic assignments，不再允许K!任意换位；每个slot由equivariant raw angular evidence产生mean，再投影为单峰von-Mises/discrete circular distribution，concentration由proper likelihood学习并直接作为定位不确定性。该修正同时针对K>=3换位歧义、目标mass扩散和confidence失配；K=2的循环assignment与原全排列等价，可形成清晰消融。

下一唯一任务是零训练COUST readiness：证明循环换位不变而反向/crossing assignment受罚，单峰归一化与极端concentration稳定，rotation/slot cyclic permutation等变，duplicate/missing-mode有非零梯度，真实1--4 batch finite backward，并保持无free-query/identity/future。PASS后才允许新三seedData Card；失败则停止该修正。

正式归因run=`results/gate3_semantics/gate3_20260829_gse_cardinality_conditioned_circular_slot_transport_failure_attribution_v1_seed0`；16项seal SHA=`06bee093f8a18e3eecd4bbbe407db97416ce986ac6169c9bf45235683bb3242c`；图SHA=`576b9b351956f518e434910a562e83b61d49f0de4d79ba1e2422555e1297bed2`。

## 64. 2026-08-29 当前权威更新：COUST readiness V1R正式PASS，允许训练验证但方法尚未确立

COUST候选将K出口按sensor azimuth排序，只允许K个orientation-preserving cyclic assignments；每个slot的equivariant raw angular evidence先产生圆周均值，再投影为proper discrete von-Mises单峰分布，并由learned concentration表达定位不确定性。总参数788,618，无free query/objectness。

V1的10项科学检查通过，唯一失败是将float32 `atan2`得到的bearing角度诊断与dimensionless tensor error共用`3e-5`门：概率质量旋转误差仅`2.67e-7`，但角度诊断为`0.003601deg`。V1作为正式科学FAIL保留，未训练、未读测试、17项seal均复核。

V1R不改变模型、数据、容差或其他检查，只将真正由transport/loss消费的normalized circular probability distribution设为旋转权威量，角度仍完整保留为诊断。覆盖80 worlds、188,126 observations、396,913 exits，cardinality 1/2/3/4=`7,525/154,279/24,458/1,864`；真实8行覆盖1--4出口并完成finite backward。

全部11项检查PASS：cyclic relabel loss error=`1.82e-12`，reversed assignment loss=`11.2176`高于correct=`2.06e-5`，duplicate missing-target gradient=`-0.3333`；uniform/sharp concentration=`3.65e-17/0.99950`，sharp分布恰有一个local maximum；网络authoritative distribution/tensor最大误差=`7.63e-6`，repeat=0，history sensitivity=`0.06444`。0 optimizer/checkpoint/threshold/C09/C10/M-TARE/graph/planner，source unchanged；17项seal复核通过。

当前候选只完成结构可行性证明，尚不能称为最终论文方法。下一唯一任务是新建三seed训练Data Card和正式训练：C01--C06提供梯度，C07选择checkpoint及唯一拒绝阈值，C08一次零适配迁移。只有count-3/4严格2deg集合精度、safe precision/recall和连续几何显著超过旧slot transport，COUST才可确立；否则停止并保留为失败消融，仍不得进入graph。

V1 run=`results/gate3_semantics/gate3_20260829_gse_cyclic_ordered_unimodal_slot_transport_readiness_v1_seed0`，seal SHA=`fa455c5ed5ddd0dc45454e0ba12e51073a7f1c8ccb501d5d5b93dbe4bcda512b`。V1R run=`results/gate3_semantics/gate3_20260829_gse_cyclic_ordered_unimodal_slot_transport_readiness_v1r_seed0`，seal SHA=`788f89bcee0ecfddf03e4cadbdc65c86ae2e868d73d72ca7bce752f7a494719f`，图SHA=`d25c9b2dffb2077baa2960430c953e54c1f225bfaf84d0c48825dafd19908481`。

## 65. 2026-08-29 当前权威更新：COUST改善总体单/双出口但复杂结构与置信度正式FAIL

唯一正式run完成3 seeds × 10 epochs、总34,110 optimizer steps；三个最佳epoch均为8，C07 validation loss=`3.203331/3.257464/2.120927`。总耗时5,947秒，source unchanged、`error=null`，C08 checkpoint observations=0，C09/C10/M-TARE/graph/planner均0；88项seal全部复核通过。

COUST产生真实但错误分布的收益：C07/C08总体2deg exact-set由旧Slot Transport的`0.3413/0.2778`提高到`0.4156/0.3870`，绝对增益`+0.0743/+0.1092`；raw count accuracy=`0.9534/0.9465`，raw structural-action macro-F1=`0.9281/0.9239`。安全接受样本中的bearing/width/profile MAE=`0.523deg/0.790m/0.158m`和`0.853deg/0.604m/0.185m`，满足局部几何门。

核心创新假设未成立。三出口2deg exact-set降为`0.00240/0.00179`，低于旧方法`0.01300/0.00714`；四出口为`0/0.00433`，10deg也仅`0/0.02597`。因此总体提升来自占多数的单/双出口，orientation-preserving cyclic assignment与单峰投影没有解决K>=3结构解析。

learned concentration也没有形成可迁移拒绝。C07在precision=1时只接受56观测、exit recall=`0.00246`；同一threshold迁移C08接受132观测但precision降至`0.81818`、recall=`0.00419`。deployed action macro-F1约`0.0021/0.0043`。global slope MAE=`3.435/3.514deg`继续失败，其他global geometry通过。

当前决策=`STOP_COUST_BEFORE_GRAPH_DIAGNOSE_COMPLEX_CARDINALITY_COLLAPSE_AND_CONFIDENCE_TRANSFER`。COUST保留为“圆周顺序/单峰先验提高常见简单结构但伤害复杂结构”的重要失败消融，不得确立为主方法、不得进入graph、不得只用seed2、放宽2deg/0.995门或重新调训练。

下一唯一任务是冻结输出零训练归因：按seed与ensemble比较K>=3槽位间角距、预测圆周次序、slot collapse/duplicate、cyclic start不一致、kappa/concentration与真实误差关系，以及seed2低loss为何未转化为复杂集合恢复。只读C07--C08现有预测，0 optimizer/新推理/C09/C10/M-TARE/graph/planner；归因必须决定问题来自matching group仍不唯一、slot feature共享/容量、单峰投影偏置、ensemble循环对齐或confidence学习目标。

正式run=`results/gate3_semantics/gate3_20260829_gse_cyclic_ordered_unimodal_slot_transport_three_seed_training_v1_seed0`；88项seal SHA=`47e4a84354ad58498e24de839fab7a3b33fb0dbd0846213a12dcfd5114bffd80`；失败图SHA=`d5f533ebf25b1c60431c0d30600cf2c7c2c1cd64fc6852bc1c933389b4e26129`。

## 66. 2026-08-29 当前权威更新：COUST失败来自复杂集合定位/置信度目标，不是seed对齐

正式冻结归因覆盖C07/C08全部`21,548/24,394`观测和`45,504/51,537`出口，只读取三个sealed prediction trees。运行0 optimizer、0新推理、0 C09/C10/M-TARE/graph/planner，输入未变化，系统PASS。

跨seed循环对齐不是主因。将ensemble改为不受循环顺序限制的最优对齐，K=3 strict 2deg exact-set仅提高C07/C08=`0.000684/0.001785`，远低于预注册3个百分点；最好单seed也只有`0.004107/0.005653`，不能恢复复杂集合。seed2的K=3预测/目标最小角间隔中位比已经为`1.040/1.021`，但2deg exact-set仍低于0.6%，说明低loss和正确粗间距没有转化为精确出口中心。

简单“槽位挤在一起”也不足以解释ensemble失败：K=3预测最小角间隔低于真实一半的比例仅`0.1784/0.1734`，未达到0.25归因门；unrestricted assignment最优解为cyclic的比例=`0.9403/0.9430`。真正失败发生在每个复杂出口的严格局部定位与置信度目标：K=3 minimum concentration对exact-set的AUC仅C07/C08=`0.202/0.434`，kappa AUC=`0.350/0.399`，在复杂人口上甚至不随正确性单调增加。

当前决策=`STOP_INDEPENDENT_CIRCULAR_SLOT_DISTRIBUTIONS_USE_JOINT_CYCLIC_GAP_SIMPLEX_READINESS`。COUST及旧Slot Transport保留为表示消融；不再改ensemble、挑seed2、调kappa、阈值或训练轮数。下一候选保留已验证的五帧circular backbone、cardinality和连续全局几何，但用一个圆周phase与K个严格正、总和360度的gap simplex联合生成完整有序出口集合；这从表示层禁止duplicate/missing/crossing，不再让K个slot独立定位。先做零训练readiness与相关工作差异审计，证明wrap/rotation、cyclic relabel、gap closure、真实1--4 batch finite backward及无identity/future；PASS后才允许训练。

正式run=`results/gate3_semantics/gate3_20260829_gse_coust_complex_cardinality_failure_attribution_v1_seed0`；16项seal独立复核PASS，SHA=`bc6d05a03a2cd48d88e3f3a498a6286d6b4d670b1c79be5f8ae58102c3ff866c`。

## 67. 2026-08-29 当前权威更新：Joint Gap V1暴露phase可观测性合同缺失，V2修正准备

V1在相同80-world/188,126-observation Teacher人口上完成系统运行。正gap closure、collapsed-gap梯度、proper scale、真实1--4出口finite backward、因果历史和隔离全部PASS；唯一方法门FAIL为随机初始化下的输出旋转合同。phase mass本身rotation error仅`1.40e-9`，但其resultant接近零时`atan2`角度数学上未定义，bearing诊断=`0.00720deg`，依赖该角度采样的geometry诊断=`1.97e-4`。

进一步接口审计发现V1 loss未对phase distribution使用proper likelihood，且whole-set confidence未包含phase concentration；因此不能只把FAIL降格为浮点误差。V1固定为`PHASE_OBSERVABILITY_AND_REFUSAL_CONTRACT_FAILURE`，不允许直接训练。

唯一最小V2修正为：对K个cyclic phase候选加入proper circular phase NLL；whole-set refusal显式乘以selected phase concentration；rotation权威量改为实际由模型学习的phase distribution与unnormalized resultant，低concentration时的degree bearing/geometry仍保留诊断但不得判为有效结构。人口、backbone、gap simplex、Teacher、参数规模、真实8行与全部隔离保持不变。V2通过后才允许三seedData Card。

V1 run=`results/gate3_semantics/gate3_20260829_gse_joint_cyclic_gap_simplex_readiness_v1_seed0`；17项seal SHA=`e2e636523e49ef0db711c8232a5f050a9740def9e5ab924bf8015d510497a942`。

## 68. 2026-08-29 当前权威更新：Joint Cyclic Gap Simplex V2 readiness正式PASS

V2保持V1的80 worlds、188,126 observations、396,913 exits、789,650参数、固定8行真实batch和全部隔离，只增加proper phase likelihood、phase-concentration refusal及distribution/resultant rotation authority。11项检查全部PASS。

合成正确/均匀/错误phase NLL=`0.000355/5.192957/16.000355`，均匀phase真值bin梯度=`-0.99444`；collapsed-gap localization=`0.776857`显著高于正确值近0，gap gradient L1=`0.001787`。所有K分支gap为正且closure error最大`1.19e-7`；真实1--4出口forward/loss/backward全部finite。

权威rotation误差：phase mass=`5.59e-9`、resultant=`1.80e-8`、gap/scales=`5.96e-8`、batch permutation=`9.54e-7`，repeat=0。低phase concentration严格上界whole-set confidence。degree bearing=`0.00665deg`与低phase sampled geometry=`1.83e-4`继续完整记录为未定义/低置信诊断，不进入权威PASS。

当前决策=`ALLOW_JOINT_CYCLIC_GAP_SIMPLEX_THREE_SEED_DATA_CARD`。下一训练从头seeds 0/1/2、C01--C06梯度、C07 checkpoint及唯一whole-set refusal选择、C08一次零适配迁移；保持10 epochs、同backbone和全局几何门。必须按1/2/3/4出口报告strict 2deg集合，并至少显著恢复COUST失败的K>=3；否则停止，不进入graph。

正式run=`results/gate3_semantics/gate3_20260829_gse_joint_cyclic_gap_simplex_readiness_v2_seed0`；17项seal独立复核PASS，SHA=`42c335f0ebdb7d0c97fef67746513d1d96bc28ba2861c5117ce482080f002b14`。

## 69. 2026-08-29 当前权威更新：singleton closure已纠正，首次三seed训练因phase数值路径系统FAIL

训练前审计发现`circular_gaps_from_ordered_bearings`对K=1使用普通roll差，错误返回0度；联合圆周集合的唯一closing gap应为360度。该错误会让全部7,525个singleton观测承担固定360度gap误差。实现已纠正并增加回归，V3重新资格在相同80 worlds、188,126 observations、396,913 exits上PASS；singleton gap=`360.0deg`，其余V2 11项检查全部保持PASS，0 optimizer/checkpoint/test/graph/planner。V3 18项seal SHA=`b9e896962e0288b817c3e8838e7c7fe0a215a212cbe2b74ad9b8db70203cf605`。

首次三seed正式训练随后从头执行。seed0完整完成10 epochs/11,370 steps、best epoch=8、C07 loss=`-2.641473`，并导出45,942条C07/C08预测；seed1在epoch4训练过程中触发CUDA scatter/gather index out-of-bounds，run按规则停止，seed2和统一selection均未执行。失败前seed1 best epoch=3、loss=`-1.735134`，两个checkpoint权重均全部finite；没有科学性能结论，C09/C10/M-TARE/graph/planner仍为0。失败run 36项seal SHA=`d7fe9281a168a1d6ca30b7a48268e598868b4797041410814b58e5a7eb1fe499`。

错误发生在phase/gap解码后的CUDA gather，并以异步device assert在下一操作报告。当前分类为`PHASE_RESULTANT_SINGULAR_GRADIENT_OR_NONFINITE_BEARING_SYSTEM_FAILURE`：`atan2(0,0)`允许NaN梯度，现有训练器又未在optimizer step前拒绝nonfinite gradient，下一forward可将NaN bearing转成越界索引。下一唯一任务是加入与float32 phase-mass精度绑定的stable atan2 backward、显式finite bearing/gradient fail-closed和对应单元/真实batch重新资格；不得续跑旧目录或把seed0当科学结果。修正资格PASS后，创建V1R并从三个seed全部重训。

## 70. 2026-08-29 当前权威更新：stable phase不足，必须同步定位真实越界kernel

V4对普通`atan2`前向增加低resultant有界backward，并在periodic sampling前拒绝nonfinite bearing、在optimizer step前拒绝nonfinite gradient。11项unit tests及完整80-world/188,126-observation/396,913-exit readiness均PASS；真实1--4 batch finite，0训练/test/graph。18项seal SHA=`4e649135b383d3e3fc47999daf034d75f7f4e379ac24fe2c2bc417eab2632559`。

V1R按相同数据和门从三个seed全部重启。seed0再次完整完成10 epochs/11,370 steps和45,942条预测，best epoch=8、C07 loss=`-2.197774`。seed1仍在epoch4相同时间点触发CUDA scatter/gather index out-of-bounds；finite-gradient guard没有先触发，说明越界发生在forward/loss索引本身而不是非有限梯度进入optimizer。seed2和selection仍为0，故没有科学结论。V1R 36项seal SHA=`a5f169e51b2ecd08341dfcbc63e45af1da2ff6692c2a3e2a0d0373ad6f9ad91c`。

当前停止继续猜测或整组重训。下一唯一任务是一个seed1、最多5 epochs的`CUDA_LAUNCH_BLOCKING=1`冻结诊断，保持完全相同的数据顺序、augmentation、model和loss，只让device assert在真实kernel调用点同步报告。诊断必须保存精确epoch/world/batch、调用行、输入索引范围和0 C08/C09/C10/M-TARE/graph；定位后才允许决定是target index canonicalization、periodic sampler边界还是其他实现缺陷。

## 71. 2026-08-29 当前权威更新：真实越界已归因并纠正，允许一次V1R2能力训练

同步诊断在完全相同seed1顺序上于第5轮、5,673次成功optimizer step之后精确复现越界。真实调用为`joint_cyclic_gap_simplex_loss`的phase-log-mass gather；故障批次是`S10_3d_complex_C01`、epoch4、batch9、shift360，包含global sequence `180664`。该行的真实角度为`-2.842170943040401e-14°`。由于浮点`remainder`在圆周零点将它舍入为连续bin `180.0`，旧实现直接转为非法索引180；Teacher active bins仍全部在0--179，故障不是Teacher语义或模型非有限输出。

纠正采用canonical periodic coordinate：保留插值fraction，但将整数圆周坐标modulo 180，使同一圆周端点180映射回0；它不把普通越界截断到边界。该逻辑同时用于模型periodic sampling与phase likelihood。12项unit tests PASS，并加入真实负零舍入回归。

首次V5 readiness因外层遗漏训练环境已有的`CUBLAS_WORKSPACE_CONFIG=:4096:8`，在进入loss前由PyTorch确定性门拒绝，0 optimizer/checkpoint，固定为system FAIL。V5R只恢复该冻结环境变量，数据、模型、代码修正和科学门不变。正式V5R证明旧坐标恰有1个raw out-of-bounds、canonical out-of-bounds=0、索引范围0--179；真实128行CUDA loss forward/backward及全部梯度finite，既有11项方法readiness与12项unit regression全部PASS，0 optimizer/checkpoint/C09/C10/M-TARE/graph/planner。

当前决策=`ALLOW_ONE_FROM_SCRATCH_JCGS_V1R2_THREE_SEED_TRAINING`。这只证明实现可训练，不证明论文方法成立。V1R2必须从seeds 0/1/2全部重训，C01--C06共142,184观测提供梯度，C07 21,548观测选择checkpoint和唯一拒绝门，C08 24,394观测只作零适配迁移。只有K=3/4 strict 2deg完整集合、安全拒绝、连续几何和最终结构语义相对旧方法达到预注册门，才允许确立方法并进入graph；否则停止JCGS。

归因seal SHA=`12ad57ef81c8d796b7bcbffa0130247f565b3abefd5b1802a515b6c1a26e788c`；V5 system FAIL seal SHA=`8e7527c7258f810471e2effa8e8efc18523a3fe092413e5d77f8b09c938c798a`；V5R PASS seal SHA=`14a249d19860be0c6c986a1ba66e606b8e633da2d7c3da187e9b085b031ef152`。

## 72. 2026-08-29 当前权威更新：JCGS训练系统PASS但复杂结构科学FAIL，停止进入图

V1R2从头完成seeds 0/1/2各10 epochs、总34,110 optimizer steps；best epochs=`8/8/7`，C07 losses=`-2.197774/-2.543406/-2.266707`。三seed均导出C07/C08预测，总model inference observations=`784,266`；source unchanged、`error=null`、C08 checkpoint observations=0、C09/C10/M-TARE/graph/planner=0。周期端点修复在原seed1故障深度及完整训练上未再失败。

科学结果否决JCGS。C07/C08 overall strict 2deg exact-set=`0.200807/0.157211`，相对COUST下降`0.214776/0.229770`。K=3和K=4在2/4/10deg完整集合全部为0；K=1与K=4 count accuracy也为0，K=3仅`0.5534/0.5332`，只有K=2 count约`0.989/0.984`且2deg exact-set=`0.2468/0.1944`。因此joint gap表示强化了多数双出口模式，没有学习复杂出口集合。

安全拒绝完全不可用。C07在precision=1时只接受1/21,548观测、exit recall=`4.40e-5`；阈值迁移C08仍只接受1/24,394且为错误，precision/recall=`0/0`。deployed action macro-F1约`3.8e-5/3.4e-5`。global axis=`9.75°/10.52°`，slope=`3.37°/3.46°`，连续几何合同也未全部通过。

当前决策=`STOP_JOINT_CYCLIC_GAP_SIMPLEX_BEFORE_GRAPH_AND_ATTRIBUTE_FROZEN_OUTPUTS`。禁止建图、调阈值、挑seed、增epoch或复用C08适配。下一唯一任务是只读冻结输出分解：per-seed/ensemble cardinality confusion、oracle count、phase-only与gap-shape误差、oracle phase/gap、cyclic alignment、gap collapse和confidence separability。归因必须决定保留backbone/global geometry、保留或废弃gap约束，以及下一方法是否转向结构事件/出口关系而非精确完整集合回归。

正式run=`results/gate3_semantics/gate3_20260829_gse_joint_cyclic_gap_simplex_three_seed_training_v1r2_seed0`；89项seal SHA=`2290a4262da57a35d9755368913f8f0c8fa6a5614934a1c282648aa1d560533b`。

## 73. 2026-08-29 当前权威更新：JCGS归因否决完整集合回归，转向轴锚定事件—关系分解

正式冻结归因逐帧读取C07/C08全部`21,548/24,394`观测、`45,504/51,537`出口及三个seed预测，精确复现formal exact-set@2deg=`0.2008074995/0.1572107895`。运行只用CPU，0 optimizer、0新推理、0 checkpoint selection、0 C09/C10/M-TARE/graph/planner，source unchanged。

cardinality不是主因：使用真实出口数量后exact-set@2deg只提高`0.01211/0.01099`，且K3/K4在10deg仍为0。跨seed循环起点也不是主因：使用Teacher只作诊断地对齐三个seed，exact2额外提高仅`0.000046/0`。因此不得通过class weight、oracle count、改ensemble或挑seed修补JCGS。

真正故障在复杂结构的绝对phase与gap shape同时崩溃。K>=3把phase替换为真值后，预测gap在10deg仍`0/0`完整恢复；把gap替换为真值后，预测phase在10deg也只有`0.11268/0.12027`。K3 gap MAE中位数=`40.05/38.27deg`，phase error中位数=`71.79/86.55deg`；K4同样失败。安全confidence仍不能迁移：formal score在C07达到0.995 precision只接受1帧，迁移C08为0 precision。

当前决策=`REPLACE_COMPLETE_SET_REGRESSION_WITH_AXIS_ANCHORED_EVENT_RELATION_FACTORIZATION`。保留已经跨方法反复成立的五帧causal circular encoder、结构cardinality/event信号、width/height/curvature和旧descriptor transport证据；JCGS、COUST、Slot Transport、Set Process作为表示消融保留。停止把“从一个任意圆周phase一次性回归完整出口集合”作为主方法。

下一候选固定为`Axis-Anchored Geometry-Semantic Event Relation`：以当前运动方向和学习局部轴线构成可观测轴锚，不再学习任意global phase；先分类corridor/junction/terminal/turn/geometry-transition事件，再预测相对于入射轴的continuation/side-branch关系、分支几何、descriptor与不确定性；五帧transport只做persistent/reveal/withdraw关系，exact-once commit产生结构节点，边仍仅由真实穿越建立。它不是旧pooling事件分类器，也不是出口方向分类：几何关系必须直接决定节点类型、出口关联和edge attributes。

不得直接训练。下一唯一任务是只读C01--C08 Teacher feasibility：证明轴锚在因果输入中可获得、相对branch relation唯一、reverse traversal变换闭合、K1--K4及turn/geometry-transition标签人口足够、困难平行/stacked结构不依赖identity，并核对旧RelationalExitTokenTransport/commit证据能否作为初始化或仅作基线。PASS后才实现零训练method readiness；失败则停止该分解。正式归因run=`results/gate3_semantics/gate3_20260829_gse_joint_cyclic_gap_simplex_failure_attribution_v1_seed0`；18项seal SHA=`c627c1265f26b2dbcc49de542b9befae1a838f14d258bbe6772e17a2b424d1c4`；图SHA=`b61df1f3d0e6660b8dd29b3d6c3bc90c9dbde71b8cb010b27e890326a216899b`。

## 74. 2026-08-29 当前权威更新：轴锚定事件—关系Teacher合同正式成立

正式只读审计覆盖C01--C08全部80 worlds、252,430 raw frames、188,126五帧因果observations、396,913 visible exit tokens和4,493个独立association identities；33,083项源seal全部重验，12项预注册检查全PASS，source unchanged，0 optimizer/新推理/checkpoint selection/C09/C10/M-TARE/graph/planner。

运动轴在sensor frame中可直接确定：最大横向分量=`1.28e-15`，最小前向分量=`0.87236`。全部4,493个结构identity均存在唯一完整branch star；563个junction identities全部至少有4个route views，且fit/selection中K3/K4 junction比例均超过95%。fit/selection分别包含`2814/995`个reveal与`3032/1080`个withdraw关系，五类事件、global geometry和exit-width均有足够监督。最小可见出口角间隔=`5.0339deg`，高于冻结2deg分辨合同。

旧descriptor transport在fit/selection的最低precision/recall均超过0.98，证明关系描述符具有学习信号；旧stateful commit precision=`0.99624/0.99709`但recall=`0.49719/0.56882`，只能作为安全基线/消融，不能冒充主方法收益。7个junction identity在局部观测中只有K2，必须作为provisional/reveal案例保留，禁止删除。

当前Phase=`GSE_GRAPH_GATE3_AXIS_ANCHORED_EVENT_RELATION_METHOD_READINESS`，阻塞=`AXIS_ANCHORED_EVENT_RELATION_MODEL_CONTRACT_UNPROVEN`。方法方向现已由Teacher证据确立，但网络实现和性能尚未确立。唯一NEXT是冻结typed observation/token/memory接口、pairwise axis-relative relation head、permutation/rotation/reverse-traversal合同、masked multi-task loss与真实batch finite backward；readiness PASS后才允许三seed训练。正式run=`results/gate3_semantics/gate3_20260829_gse_axis_anchored_event_relation_teacher_feasibility_v1_seed0`；19项seal SHA=`709930fc5c2e7774883bd5493851b1d9f60d07a79e0926b1662cdfee3219a090`；论文图SHA=`6407e28a89e8b193b97f79b06a9ec96ac767636d80c28f660bc90196df9a1569`。

## 75. 2026-08-29 当前权威更新：Axis-Anchored Event Relation模型接口正式PASS

主模型已实现为819,196参数的五帧circular LiDAR encoder：global head输出五类结构事件、local axis、width/height/slope/curvature、place descriptor和uncertainty；dense axis-relative head对四个相邻历史步逐bin输出absent/persistent/reveal/withdraw，并为当前persistent/reveal分支输出heading residual、opening width、vertical profile、descriptor和heteroscedastic uncertainty。past-only branch union仅按过去到当前递推；forward唯一参数为`scans`，identity/pose/world/traversal/future/graph均被接口禁止。

正式readiness固定读取两个C01 fit parents中的7个observations/29个唯一raw frames，覆盖五事件、四关系、place与branch正负identity pair。7项unit tests及15项formal checks全部PASS；真实CUDA全部输出、7项loss和全部gradient finite。relation/event rotation max error=`2.68e-7/2.98e-8`，batch permutation=`2.98e-8`，reverse-axis roundtrip与repeat=`0`。

预审发现RTX 5090默认TF32/CuDNN会使relation rotation error=`9.64e-5`超过冻结`3e-5`门；没有放宽门槛，而是冻结deterministic algorithms并关闭TF32，误差降至`2.68e-7`。这项环境合同必须进入后续训练spec。

当前Phase=`GSE_GRAPH_GATE3_AXIS_ANCHORED_EVENT_RELATION_THREE_SEED_TRAINING_PREPARATION`，阻塞=`AXIS_ANCHORED_EVENT_RELATION_SCIENTIFIC_PERFORMANCE_UNPROVEN`。方法和可训练接口现已确立，性能仍未确立。唯一NEXT是冻结三seed Data Card/trainer/evaluator：C01--C06梯度，C07 checkpoint/阈值选择，C08一次零适配迁移；必须按五事件、四关系、连续几何、descriptor association和安全拒绝分别报告，不能用总体平均掩盖稀有turn/transition或relation失败。正式run 18项seal SHA=`022b0c0afec897ae5a3d33c3ddd30bb76d0c648038a8553b8c7a740b9d5931e5`；论文图SHA=`df85811305e643d6ef9f70e73b66b2cbd6c4a3c77e1c1e1248035e4d3d594a80`。

## 76. 2026-08-29 当前权威更新：缺失早期关系标签的训练合同已纠正

全人口预审确认：全部188,126条样本均有五帧LiDAR与当前事件/几何Teacher，但轨迹起始处并非每个raw reference都有观察级exit Teacher。禁止删除这些样本或伪造关系。训练合同固定为保留全部样本；event/global/current-branch geometry始终监督，四个相邻关系位置仅在前后帧都存在同traversal观察级Teacher时有效。

精确有效关系pair positions为fit/C07/C08=`448,152/66,752/77,490`；完整四关系行=`94,144/13,828/16,388`，包含变化的行=`8,112/1,319/1,677`。relation bin人口fit为absent/persistent/reveal/withdraw=`79,711,009/934,760/10,583/11,008`。类别权重只从C01--C06冻结为inverse-square-root normalized：event=`[0.266182,0.607611,1.145899,2.217725,0.762583]`，relation=`[0.021964,0.202824,1.906186,1.869026]`；C07/C08不得重算。

V1R保持模型819,196参数和原7-observation真实batch不变，只增加`-1` relation mask、独立current branch mask及core/identity-balanced loss拆分。8项unit与新增masked CUDA backward正式PASS；core total=`5.41459`、descriptor total=`3.47317`，全部gradient finite，原15项模型不变量保持PASS。

当前唯一NEXT仍是三seed训练实现与Data Card，但trainer必须执行全部core rows和独立identity-balanced descriptor batches，并逐项保存有效mask人口。V1R正式seal SHA=`cce9ff427daa2f5399a592f10c5fd75bd9c7957b91e7bc42db380a92126c0c14`。

## 77. 2026-08-29 当前权威更新：多标签关系V2无损覆盖全人口，允许训练实现

训练Teacher实现阶段发现互斥关系类别与真实物理事件冲突：fit/C07/C08分别有`202/20/33`个2度方向格同时发生不同exit identity的reveal和withdraw。V1/V1R小批资格不足以覆盖该情形，保留为接口失败与必要消融，禁止用于正式训练。

唯一修正是把persistent/reveal/withdraw改成三个独立binary channels；不同通道可以共存，同通道重复仍fail closed。五帧因果LiDAR、180个方向格、事件/轴线/连续几何/descriptor heads、split、Teacher identity和全部测试隔离均不变。模型参数由`819,196`变为`819,067`。

V2正式逐行rasterize 80 worlds/188,126 observations：fit/C07/C08有效pair=`448,152/66,752/77,490`、完整四步rows=`94,144/13,828/16,388`；persistent/reveal/withdraw positives分别为fit=`934,760/10,583/11,008`、C07=`139,346/1,636/1,728`、C08=`161,597/2,107/2,181`。全部255个跨通道冲突可表示，零同通道冲突。15项unit、19项方法合同和真实CUDA finite backward全部PASS。

当前决策=`ALLOW_AXIS_ANCHORED_EVENT_RELATION_THREE_SEED_DATA_CARD`。下一唯一任务是实现完整trainer/evaluator并冻结训练Data Card：C01--C06提供梯度和fit-only weights，C07选择checkpoint与拒绝阈值，C08只做一次零适配迁移；seeds固定0/1/2。必须分别报告五事件macro-F1、三关系指标、width/height/slope/curvature误差、place/branch association precision与false merge、uncertainty coverage。任何一项核心结构能力不能超过预注册基线即停止，不进入graph/planner。

正式run=`results/gate3_semantics/gate3_20260829_gse_axis_anchored_event_relation_readiness_v2_seed0`；seal SHA=`260e7ff9321111e87a2403c0e4f738e13543dc18419e837af4249f11651823a3`；图SHA=`ded447e672b46f42ec28c6adadc6ef6de36be5134559424732cae42a8ad8465e`。
## 78. 2026-08-29 当前权威更新：三种子训练FAIL，停止密集关系场

正式Axis-Anchored三种子训练完成`36,690`步且`error=null`，但C07/C08五类event、axis、三通道关系、current branch field、连续几何整体、关联与拒绝合同均未全部通过。axis误差约`83°`；在precision `>=0.98`与recall `>=0.25`合同下三个关系通道均无可用阈值。因此当前方法没有确立，禁止进入graph、planner、C09/C10或M-TARE。

冻结代码与历史同人口对照把最小修正限定为两点：恢复已验证的五帧逐方位directional temporal融合；把极稀疏三通道dense BCE改为圆周软峰、困难负样本、显式峰值解码与跨帧bearing transport。persistent/reveal/withdraw仍为可共存语义，不得退回互斥类别或单纯presence。

唯一NEXT是C01--C08零训练机制归因/readiness，固定使用本次三seed输出和既有Peak/Slot封存证据，不读取C09/C10/M-TARE，不创建图，不训练。只有该证明同时确认方向可恢复、关系表示可分离且没有Teacher语义冲突后，才允许最小V2训练Data Card。
## 79. 2026-08-29 当前权威更新：V2归因PASS，冻结稀疏圆周关系transport路线

正式归因在C07/C08完整冻结人口上确认：当前axis失败来自丢失五帧逐方位融合；dense relation失败来自极稀疏关系与数千倍正权重造成的定位/校准崩溃。pair-level relation presence仍有AUC `0.84--0.92`，oracle-count的±8°局部定位在三个通道上为约`35--46%`且跨C07/C08稳定，因此允许新的稀疏proposal readiness，但不允许直接训练。

下一typed表示固定为最多6个圆周exit/relation tokens和相邻帧带dustbin transport：matched token为persistent、dustbin→current为reveal、previous→dustbin为withdraw；不同通道同bearing可共存。axis和token feature必须消费五帧directional-temporal融合；圆周软支持和困难负样本只用于proposal定位，不能替代transport与拒绝。

唯一NEXT是零训练V2模型readiness和Data Card。必须覆盖输入隔离、同bin reveal+withdraw、跨bin运动、wrap、permutation、rotation、reverse traversal、dustbin、有限梯度、确定性和真实C01--C08 batch；C09/C10/M-TARE/graph/planner继续禁止。
## 80. 2026-08-29 当前权威更新：Sparse Relation Transport V2 readiness PASS

V2 typed模型固定为`783,675`参数、最多6个圆周token和四个相邻帧transport。53个已验证五帧backbone键从冻结Slot checkpoint逐值加载；previous token匹配current token为persistent，匹配dustbin为withdraw，current新token为reveal。Teacher identity只允许在loss-side assignment出现，forward仅接受`[B,5,2,16,720]` LiDAR。

正式readiness以8条固定fit观察完成真实CUDA forward/backward，所有trainable parameter gradient存在且finite；16项输入、关系、旋转、排列、反向、确定性、人口与隔离检查全部通过，0 optimizer/checkpoint/test/graph。

唯一NEXT是三seed训练Data Card/trainer/evaluator。C01--C06全部梯度；C07 checkpoint/阈值；C08一次零适配。必须独立评价圆周proposal、token cardinality、persistent/reveal/withdraw transport、axis、五事件、几何、descriptor association与拒绝；未通过前禁止graph/planner/C09/C10/M-TARE。

## 81. 2026-08-29 当前权威更新：Sparse Relation V2完整训练接口重新资格PASS

训练候选以六个proposal为固定容量，但新增903参数的0--6 count head决定有效结构token数。早期无Teacher帧和relation pair显式mask；withdraw/reveal权重按fit persistent与稀有对象人口平方根冻结为小于10，停止旧dense BCE数千倍权重。

真实完整loss首次执行发现四维profile加width只有五个几何目标，旧头却输出六个uncertainty。训练在零optimizer step停止；唯一修正将token geometry head输出从11改为10，减少65参数。最终模型784,513参数。8条固定真实观察上核心与identity-balanced descriptor loss、全部参数gradient、count因果/rotation/permutation/repeat全部PASS。

当前唯一NEXT是冻结三seed训练。每seed从对应Slot seed checkpoint只加载53个已验证backbone键，新heads由本seed初始化；10 epochs、C01--C06梯度、C07 checkpoint/阈值、C08唯一零适配。必须报告deployable count-conditioned token set、三类transport、五事件、axis、连续与token geometry、association和拒绝。训练FAIL则回到表示归因，不用图补偿。

## 82. 2026-08-29 当前权威更新：完整Teacher召回口径PASS，V2R2三seed训练执行中

首次V2训练在seed0两轮后由实测每轮约8.5分钟证明外层4小时不足以容纳3×10轮及最终评估；在科学输出前主动停止并封存为系统失败。只把外层时限改为8小时的V2R启动后，训练前并行代码审查发现旧评估器的relation recall只以已成功定位的预测端点为正例人口，遗漏端点不会进入FN，可能虚高关系召回；V2R同样在零epoch、零checkpoint和零科学结果处停止封存。

正式objective-recall corrective逐行读取C01--C08全部80 worlds、188,126 observations及`448,152/66,752/77,490`个有效因果pair。独立Teacher正例人口为fit persistent/reveal/withdraw=`934,760/10,583/11,008`，C07=`139,346/1,636/1,728`，C08=`161,597/2,107/2,181`。缺失端点构造把旧candidate-only recall=`1.0`纠正为objective recall=`0.4`并记录3 FN；21项unit和8项formal checks全部PASS，0训练/推理/阈值/test/graph/planner。seal SHA=`24551f44cf15fa7cf1f3c9a7a334a1bd4d84bd552fe45246b759f49bf99482fb`。

当前唯一NEXT为正在执行的V2R2三seed训练。它保持模型、loss、数据、seed、epoch、checkpoint选择和科学阈值不变，只绑定8小时外层时限与完整Teacher token/relation recall定义；结果未完成前方法性能仍未确立，graph/planner继续禁止。

## 83. 2026-08-29 当前权威更新：资源根因收口，V2R4完整训练获准

V2R2在科学计算前暴露structural-refusal分母接口不一致，V2R3随后在seed0首轮验证阶段因资源监控停止；两者均未形成三seed统一指标，不能证明或否定候选方法。资源归因V1证明把验证batch从256改为128会使加权validation loss最大变化`0.0190654`，因此禁止用减小batch改变checkpoint选择。

V1R保持batch256并在独立C07 world边界释放未使用CUDA allocator cache，完整执行10 worlds/21,548 observations。相对sealed batch256基线，全部输出和loss最大误差均为0；最大allocated/reserved/process memory分别为`4.286/9.057/9.709 GiB`，全部低于16 GiB。25项unit tests PASS，0 optimizer/checkpoint/C08/C09/C10/M-TARE/graph/planner；21项seal独立复核，SHA=`9aac7577efa75b235364fc738b26a3a24e958f6d2e0f4f65380f733b246f13f0`。

训练器只加入验证和最终推理world边界的unused-cache release；模型、数据、loss、batch、seed、epoch、阈值和评估器均不改变。当前唯一NEXT是V2R4从头执行seeds 0/1/2。方法仍未确立；只有C07选择和C08零适配的事件、token、关系、几何、关联与安全拒绝全部满足预注册门，才允许离线图。

## 84. 2026-08-29 当前权威更新：V2R4训练world缓存越界，停止并做精确parity corrective

V2R4 seed0完成3个epoch后，训练world序列中的nvidia process memory达到`16,386 MiB`，超过冻结16 GiB上限2 MiB。按硬门立即SIGTERM子进程；runner封存为system FAIL，seed1/2、统一selection、C08和graph均未执行。13项seal独立复核，SHA=`d83d3c436346a911e6d7e30cc9e9ee84d7b2a4a970a9331ee987525ddbb761e6`。

前三轮C07 event macro-F1=`0.5189/0.5605/0.6119`、axis=`8.522/6.862/6.391deg`、count accuracy=`0.9321/0.9468/0.9579`，turn与geometry-transition开始恢复；但这些只是系统失败run中的趋势，不得作为方法PASS或checkpoint选择。

唯一NEXT是训练world cache parity/resource corrective：seed0精确epoch0、60 fit worlds、142,184 core rows、6,347 descriptor rows、1,223 optimizer steps及完整C07；每world边界释放unused cache，全部77个模型tensor必须与sealed V2R3 epoch0逐元素相同，全部train/C07指标误差`<=3e-6`且三种显存量均`<=16GiB`。PASS后才允许新的三seedrun；失败转process isolation，不提高资源门或减小batch。

## 85. 2026-08-29 当前权威更新：训练world cache精确等价PASS，允许V2R5

正式corrective完成seed0精确epoch0：60 fit worlds、142,184 core rows、6,347 descriptor rows、1,223 optimizer steps和完整10-world/21,548-row C07 validation。相对sealed V2R3 epoch0，77/77模型tensor逐元素完全相同，全部train/C07数值指标最大误差为0。

60个训练world的最大allocated/reserved/nvidia-process memory=`7.383/11.363/12.017 GiB`，全部低于16 GiB；120次训练world cache边界调用完整。26项测试、输入/工具不变、20项seal独立复核PASS，SHA=`ee6e2ce6c0b74f75795c0a6cd84cee134c465c39d988ab9e9f5f5d3f69329f93`；论文候选图PNG/PDF/SVG保留。

正式训练器进一步只增加每训练world的nvidia process memory记录，runner在`>16GiB`时fail closed；27项测试PASS。唯一NEXT是V2R5从头seeds0/1/2完整训练。仍不得把资源与等价PASS表述为方法科学成立。

## 86. 2026-08-29 当前权威更新：V2R5只可确立token/transport基线，不能确立轴锚定主方法

V2R5运行期间的只读代码与Teacher审计发现两个此前未进入科学门的接口缺口。第一，`SparseCircularRelationTransportNet`虽然输出`local_axis`并单独计算axis loss，但proposal仍在sensor azimuth上产生，transport只建模persistent/reveal/withdraw；`local_axis`没有参与token坐标化、continuation/side-branch关系或节点触发。第二，当前轨迹pose本来沿traversal切线布置，`local_axis_robot`的水平分量在fit/C07/C08上几乎恒为robot-forward。

精确人口为fit/C07/C08=`142,184/21,548/24,394`。三个split的水平角绝对偏差最大仅约`7.4e-14deg`。不读取LiDAR、恒输出`[1,0,0]`的常数前向基线平均三维axis误差为`2.79198/3.26039/3.33944deg`，已经显著优于V2R5当前训练曲线约`5.96--6.91deg`；因此原`axis<=10deg`检查不能证明学习了中心轴，也不能支持论文轴语义贡献。

当前决定：V2R5继续完整执行，因为三个seed的事件、count、token、transport、geometry、descriptor与refusal结果是必要的learned exit-token/temporal-transport基线和消融；但无论其旧selection状态如何，都不得据此直接进入主方法图或宣称Axis-Anchored GSE成立。V2R5完成后的唯一NEXT先做正式方法差距审计并冻结修订接口：把机器人运动方向明确作为已知route-frame坐标锚，而非学习贡献；学习输出必须是相对该锚的结构分支关系、几何属性、事件、descriptor和不确定性，并由这些量直接决定node/exit association。若仍保留learned local axis，必须在独立yaw扰动的验证合同下优于常数/运动方向基线。

## 87. 2026-08-29 当前权威更新：独立event head不满足“几何语义决定图结构”因果链

只读def-use审计进一步确认：V2R5的`event_head=Linear(context,5)`直接从全局encoder context产生事件，与token count、token bearing/geometry、persistent/reveal/withdraw transport和global geometry heads并行；这些几何量没有进入event logits。现有`OnlineTopometricGraph.update`又以`observation.event`及其置信度作为structural node触发条件，width/height/slope/curvature/local_axis只进入trace或edge属性。因而当前链路是“多任务网络的独立事件分类器触发图”，不是论文要求的“学习几何结构语义直接决定节点结构”。

该缺口不否定V2R5作为强基线的价值，但否定其直接成为最终主方法。最终候选必须新增可审计的`Geometry-Relation Event Composer`：事件只能由部署时可得的count-conditioned exit tokens、相对route-frame方位、persistent/reveal/withdraw、token/global metric geometry及其因果变化和不确定性组成；graph继续只消费composer事件、学习关联和真实穿越边。独立`context→event`头保留为必要消融。必须分别证明(1) composer相对独立event head提升五类macro-F1，尤其turn/geometry-transition，(2) 去掉显式几何或transport会降低事件与图F1，(3) graph输入几何字段实际参与节点/关联决策。上述合同PASS前不得称GSE-Graph主方法确立。

## 88. 2026-08-29 当前权威更新：主方法收敛为两类因果事件因子

进一步只读核对确认，V2R5的event loss仍消费原deduplicated Zarr中的旧双边窗口transition mask，fit geometry-transition为`12,534`帧；它没有消费已正式PASS的corrected causal Teacher。后者全C01--C08仅`1,031`帧/`76`个identity，fit/C07/C08分别为`791/67/173`帧和`59/5/12` identities。因此V2R5的event head限定为旧Teacher基线，token/transport/metric-geometry/descriptor才是可复用组件。

最终候选方法不再是单一五类逐帧头，而是`docs/GSE_GRAPH_METHOD_SPEC_V1.md`冻结的双Composer因子化：`Action-Set Relation Composer`仅用出口token几何、count和transport产生junction/terminal/provisional；`Metric-Change Composer`仅用宽/高/坡度/曲率的因果序列产生turn/geometry-transition及检测延迟回投。descriptor仅进入后续节点/出口关联，边仍只由真实穿越创建。

该规范现在是冻结候选，不是已验证方法。V2R5封存后的唯一NEXT是将其显式输出与corrected causal Teacher对齐，执行零训练容量/冲突审计；PASS后才可实现双Composer readiness和三seed训练。C09/C10/graph/M-TARE继续为0。
## 89. 2026-08-29 当前权威更新：ERCSS 五帧配准骨架成为唯一 Phase 3 候选

RouteGeometryProfile 已正式科学 FAIL：客观 Teacher 正反向一致，但单帧 LiDAR 对 0--20 m 五段剖面的结构身份覆盖不足。双 Composer、单帧 Profile 和密集出口关系路线停止扩展，保留为论文基线和失败消融。

当前唯一候选为 `Ego-Motion Registered Causal Structural Skeleton (ERCSS)`，完整合同见 `docs/GSE_GRAPH_METHOD_SPEC_V2.md`。学生用部署可得的相对里程计把五帧 LiDAR 配准到当前坐标，学习已经观测且与自身连通的稀疏隧道骨架、连接和连续几何；不从 dense 地图后处理骨架，不使用绝对位姿/TNG/identity 作为学生输入，不预测未观测全局边。

唯一 NEXT 是 C01--C08 零训练 Teacher/表示可行性审计。固定 80 worlds、252,430 frames、188,126 五帧序列；C01--C06 fit、C07--C08 selection；C09/C10/M-TARE/graph/planner 禁止。审计 PASS 只允许模型 readiness，FAIL 则停止 ERCSS 并重新评估论文贡献。
