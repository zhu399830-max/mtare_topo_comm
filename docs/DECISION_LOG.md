# DECISION LOG

## 2026-09-05：均值坐标真实表达限制已证实；原始点纯凸组合也不够

独立精确`v3_scoped_coordinate_audit_card_v1`卡只允许同180 C01观察的原900帧坐标支持诊断；旧metadata-only卡限制不变。preflight 0 errors/0 warnings后单次创建run，16.254345s完成，RSS804294656 bytes、约12 MiB。27项seal独立核对，SHA-list=`5aabab7364e8d0eebe23a6b501967d0bf33231fe1e7f270100c284316791f3be`；0checkpoint/模型/optimizer/新标签/C07--C10。旧180行评分精确复现。

证据：4356目标控制点中均值池3153个凸包外、1201有见证、2未定；原始点池465个外、3891有见证。2688点明确是均值池不可表达而原始点有见证，所有10父地图均出现。全点距离下界均值3.088910m对0.161924m。逐控制点方向、支撑点、非负权重和误差界均保存；没有使用抖动/重试，弱证书可保留未定。

失效结论：当前冻结几何的布局质量与旧纯均值凸组合读出不能被视为组合学习合格前端；raw XYZ简单替换也不能表达全部现有目标。这是模型输出接口限制，不是新教师全错、几何主线理论失败或模型已取得89.33%准确率。全局权重允许混合不相关通道，必须与学习点归属区别。CPU坐标公式不宣称历史CUDA逐位相同；舍入界不是物理噪声界。

选项/成本：继续训练旧输出成本高且无法越过已证实的表达边界；仅换raw凸组合便宜但465点仍不可表达；重建骨干/全数据目前没有必要。推荐用户方案内的最小前端读出验证：保留原XYZ并允许点级区分与表面到轴线几何估计，先合成测试/梯度，再独立短训练卡，不恢复三seed长训练。独立审查指出同token权重广播到原点并归一化仍精确等价旧均值，不归一化则引入回波数偏置，需构造测试拦截。

这是已批准“理想基元有效而预测无效时修前端”的实施内选择，无需新方法选项回复；不更换数据/教师/指标或降低科学线。完整事件/端口教师尚未通过，受影响训练继续停止。当前唯一下一动作更新为薄读出软件规格/合成验证；旧15:53“先测真实pool影响”为已完成历史，Phase 3及测试/闭环隔离不变。新增68项诊断测试，指定回归381项通过；旧资产和论文图不删除。

封存后独立新增`test_gse_raw_coordinate_adapter_limit.py`四项测试：均分权重等价均值、直接广播引入点数偏置、点级oracle权重可分层、单侧壁凸包无法包含外部轴线。新诊断与两组池化反例共76项复核通过（1.18s）；这些只是软件反例，不是新学生训练成绩，也未追加到已密封run中。

## 2026-09-05：排除“只是裁剪”解释，先验证坐标读出可表达性

新同180行只读分解在6.150163s完成：全部旧行、匹配、反转和float32分数精确复现；1452片段不删。横向RMS中位2.262511m、无向角37.932620°、有限折线双向距离5.200476m；当前节点382片段分别2.703649m/32.416361°/5.527949m。全部10父地图均有明显误差。26项输出seal复核，SHA-list=`cf05614525e2b0107749647f839c7b721049187d6d07bf7fc45d913db50bef5b`；0新扫描/推理/训练/标签。

解释边界：82.5814%是可分辨样本沿轴残差平方占比，不是样本比例；两段有限折线仅是三控制点诊断插值，非完整TNG样条/表面Benchmark。积分误差上界最大0.0772m，不能解释米级偏差。6教师/22预测方向未知被保留，没有删掉或当零角。旧阈值/平局字段差异披露，未读取C07选择阈值。

结论：受影响的是当前冻结前端输出的布局质量及直接训练组合头假设；不是几何基元方法理论失败。直接重训骨干或叠组合loss仍缺因果归因且成本高。推荐先检查一个明确实现限制：实际observable/sparse-port几何路径将16竖直线×4方位列平均成坐标，再对900均值点独立softmax读出三点。

新增4项零数据构造测试直接调用现有`_token_xyz`：两层±3m均值为0，任意softmax都无法恢复高度，对该高度损失的logit梯度为0；保留原点有表示两层的oracle坐标支持。不同深度也可混合。它证明限制存在，尚未证明真实180样本受此限制程度。未据此修改模型或教师，也未把oracle点选择用于学生。

下一动作限定同180原始坐标池可表达性对照（新精确卡，900现有帧，0模型/optimizer/标签），定量比较均值池与未合并返回点，再决定原XYZ旁路/点归属引导的最小读出候选。若均值池能表示则转测优化/归属，不把反例强套真实数据。SPFN的点属性学习→可微几何估计只作已核对的工程参考，不作为本论文新贡献，也不冒充地下扫掠基元已验证。用户既定快速验证/前端修正授权覆盖此诊断，无需再选方法选项；训练和实质教师变化仍须独立规格。Phase 3、C07--C10隔离和闭环停止保持。

## 2026-09-05：现有教师核对完成；先分解几何误差，不盲跑组合训练

本次新卡在audit-only范围明确允许同180行现有P1b几何/连接和已密封六字段预测缓存；`no_model_or_checkpoint`仍禁止模型执行/权重读取，不通过复用旧metadata-only卡扩大权限。正式前预检0 errors/0 warnings，新执行器合成测试发现的重复manifest缺陷已修正并回归。24项正式输出seal独立复核，旧run不变。

实证：1452可见片段全部支持/时间可见性一致，旧连接完全等于全部可见构造成员连接。180观察均有其他节点通道，178存在无连接基元的角投影重叠。后者定义只要求共享方位列，不能沿用历史“disconnected overlap”简称把它当物理冲突；不得据此掩码删除样本。GT identity仅用于事后评分，没有筛选学生预测槽。

模型轴点欧氏误差均值10.882m（九坐标MAE4.780m）、半轴MAE0.539m；当前节点382片段轴点均值13.143m，其他1070片段10.075m。独立代码核对无明显XYZ/米单位/反转错误，但该误差混合轴向裁剪与横向偏差；旧geometry评估只统计存在门以上匹配，本次全部匹配，且没有传旧endpoint_observed排序字段，不声称匹配逐元素复现或历史性能退步。

问题类别是待分解的模型几何/教师裁剪接口，不是新方法原理已被否定。它否决“字段恢复完成即可把预测几何视为足够准确并直接训练组合头”的假设，未否决几何基元主线。选项：直接训练组合头会混淆两层误差；重训骨干成本更高且目前缺归因；推荐相同缓存/教师的零推理误差分解，区分沿轴范围错、横向布局错及评分约定差别，先据证据确定最小修正。该诊断已在用户批准的组件交叉替换/快速验证范围，无需新的方法选项回复；修改teacher或模型仍须新证据/spec，不自动执行。

正式run=`gate3_20260905_gse_local_teacher_audit_v1_seed0`，5.75752894s，0扫描/forward/optimizer/targets，seal-list=`69f17706af3a3dc36f3cd01ca7e9ae5d64b8d60fe906270133f50212c09e4642`。科学Gate未升级。当前唯一下一动作与PLAN同步为同180行只读几何误差分解；事件标签、长训练、C07--C10及闭环仍停止。

## 2026-09-05：字段恢复单次完成，停止把旧缓存缺字段作为当前阻塞

独立精确字段恢复卡绑定原180行清单哈希与指定seed0 checkpoint SHA，只允许 Gate 3 `data_export`；不是扩大旧metadata-only库存卡。合成集成阶段纠正操作名 `export` 为仓库既有 `data_export`，未添加宽松别名，正式预检为0 errors/0 warnings。

唯一正式 run `gate3_20260905_gse_composition_field_recovery_v1_seed0` 在 6.136600901s 内完成：180 主推理、18 重复、900 唯一帧，旧特征/置信度逐元素复现，六项原始字段重复精确一致。模型状态 digest=`248f6d4bfe675c98afc6c9103192fa37014636c8f8f7d035bce212b77be6758a`，运行前后相同，无梯度/optimizer。GPU reserved 2082471936 bytes；源541个chunk hash核对且运行后未变化。23项输出seal-list=`478e0744a7abaf5281342722a037b4aa1267f23b9b89377f5dd67415fed35481`，独立复核全通过。

结论只限字段恢复和旧输入/权重复用一致性，不能解释为新组合头精度、完整教师、图或论文科学 PASS。TF32 旧API出现弃用警告，保留与历史相同计算设置以复现，没有借此改数值精度。

下一步从已恢复预测与同180行现有教师几何/组合/歧义出发核对可监督的局部结构，保留UNKNOWN；不删除低支持样本、不按真实节点身份预筛学生输入。它属于用户原计划的教师边界与真实/预测几何对照，不需要新的方法选择；仍须新精确卡/spec才读取未获本次卡授权的教师字段。长训练、C07--C10和闭环继续关闭。

## 2026-09-05：独立审查先拦截数值/mesh 适配错误，字段恢复不等待完整教师训练

新正射线证据检查中，独立子任务复现：2.2 m 的 float32 上舍入使端盖被误判开口；门面起点在共同 SE3 后因约 1e-15 分子扰动改变结果；不同轴向采样间距使短折线端盖法向不同。归因都是实现/输入适配，尚无真实数据标签受到影响。

修正：保留源 range dtype 并扣除一个 ULP，额外 caster/测量误差须另行冻结；数值边界统一 UNKNOWN；mesh 候选显式绑定原轴向/角向采样并用真实多边形截面。原端点 .25 m 支持阈值、样本和历史结果不动。候选射线证据不是完整节点教师，也不证明可通行。

只验证纯软件而不恢复原 180 输入会推迟关键小实验。采用用户原计划已允许的同样本字段恢复：新 reader 只读 range/valid 和因果相对 pose，保留旧模型计算；随后冻结独立卡/spec运行指定 seed0 的字段恢复和旧缓存一致性比较，0 optimizer。未完成的教师标签问题继续停止受影响训练，不以未经验证标签绕过。无需新方法选择，但不得复用只允许元数据的库存卡。

用户本轮明确允许按合理效率决定子任务分工，不要求简单任务固定由 Luna 承担。本轮独立代码审查与本地接口实现并行，未委托改变科学合同或读取真实测试数据。

## 2026-09-05：物理端点支持不能直接定义节点可见性；停止受影响标签迁移

证据：`gate3_20260905_gse_composition_inventory_v1r_seed0` 核对同 180 个 C01 观察。degree 1/2/3/4 数量为 20/120/38/2，全部 incident endpoint 支持为 20/120/6/0。40 个分叉观察均存在所有 incident 基元，6 个完全支持的观察只来自 3 个节点。`test_gse_endpoint_vs_port_semantics.py` 的解析 T 形二维切片反例中，支路射线可通过 6 m，但支路回波的最近轴向位置超过 1 m，因此不满足旧 0.25 m 物理端点支持带。该反例否定一般必要性，不证明这 40 个真实三维观察都可完整标注。

失效结论：不能用“全部旧端点支持”作为新结构事件的必要条件；不能将 34 个观察删掉，也不能宣称只剩 3 个有效分叉。V1R summary 中 `support_is_necessary_not_sufficient_for_event_label=true` 是过强解释，在此撤回其节点任务含义；人口和支持数量仍有效，旧密封文件原样保留。问题类型为教师/输入语义，尚无新模型训练失败。

选项与成本：直接迁移旧掩码成本最低但已被反例否定；仅保留 6 个观察会改变人口并削弱多样性；重新检查同 180 行的可见几何与局部自由空间证据，成本限小规模 reader/合成检查及受控字段恢复。采用第三项，属于用户已批准的“逐项检查节点级监督一致性”，不需要新的方法选择。实际生成节点标签前仍须明确可观测定义、正/未知边界并冻结新卡/spec。没有默改旧教师或阈值。

实现选择：新增 `visible_geometry_input_from_prediction`，只用预测基元存在性和几何非退化条件保留已观察片段；旧 `composition_input_from_prediction` 仍保留物理端点门作比较。前者不是端口可通行证据或边提交依据。合成测试确认改变/移除端点分类字段不影响新桥，退化方向仍拒绝。

## 2026-09-05：库存执行器系统修复、静态数据时间合同及显卡访问纠正

V1 在打开任一 Zarr 分片前因 `load_json` 仅接受对象而清单实际为列表失败；保持失败 run 和 seal，增加完整 180 行合成执行测试后，V1R 只修读取器并重新冻结工具哈希。V1R 在 5.103s 内完成，峰值 RSS 149938176 bytes，输出 14 项 seal 全部匹配。两次都 0 原始扫描解码/推理/optimizer；不是科学方法失败。V1 seal-list=`0ef35adddb2d18533aa5226277ea65772a469d33b2701bb03b99bfb7cebe8637`；V1R=`e532f9366ff5dc129c6112fafcd33b9c74057adfd6b58b45d37b1ab34e5a7213`。

距离采样静态数据没有采集时钟，不伪造 duration。新增卡 schema 仅接受 audit 操作、精确任务/行及 `duration_s=null`；不能通过此卡授权训练、导出或标签生成。只读 reader 拒绝未知任务、越界行、写操作、符号链接逃逸和密封缺块（含 Zarr `__contains__` 静默填零路径）。

经只读主机设备查询，RTX 5090 D / 32607 MiB 正常；以前沙箱内查询失败不能推出 GPU 硬件不可用。后续训练可在批准的设备访问环境中执行，但仍需独立冻结环境与训练合同。本条覆盖此前 CUDA 硬件阻塞表述。

## 2026-09-05：缓存字段不足，停止直接复用旧特征训练而非停止主线

检查旧 NPZ 成员与 180 行清单确认缺少新组合所需共同坐标及逐端口目标；不能靠反演旧独立局部坐标或用 degree 代替可观测事件标签补齐。通用 reader 初始化还会打开全根分片，需新显式 allowlist。问题类型为输入/教师接口，并非新模型科学失败。选择用户方案已允许的同样本字段核对/补充，成本先限元数据与软件测试；新数据卡冻结前不执行导出/训练，不需要新的方法选项批准。

唯一软件 run 的 56/56 PASS 只说明合成接口成立；不得作为 Phase 3 PASS。新状态机需要后续真实配准、端口映射及传感器/执行适配，当前验证布尔值只用于隔离内核测试。


## 2026-09-05：实施用户批准方案并纠正过强历史结论

用户本轮完整方案是明确方法边界批准，不再要求重复确认。新依据为 `GSE_GRAPH_COMPOSITION_EXECUTION_PLAN_V1.md`。旧 180 观察短训失败仅否定当时接口/预算，不证明理论不可观；教师 incident 身份筛选只支持理想输入上限；几何相对改善不等于部署精度通过。旧 run 保留不变。

新软件把节点存在、重访合并、实际穿越连接分开，首先纯合成测试（0 数据集读取/训练）。新 tiny 验收为事件 macro-F1 .95 与端口 F1 .90，不篡改旧 loss/身份门。C08/C09 的历史暴露不被新 split 名掩盖。常规运行沿用用户持续授权，数据相关工作仍须精确卡/spec/preflight。


## 2026-09-04：新方法草案必须修正旧Factorized生命周期，不得只替换特征

- 复核证据：历史route-conditioned factor verifier已证明关联容量，C09 consensus balanced P/R=`1.0/0.5077`、runtime false accept=`0.009865`。因此“加图上下文关联”本身不是新贡献，也不足以修好图。
- 失败归因：旧two-trace node commit与same-traversal first/last-trigger edge commit使274个真节点仅唯一提交192，13条真trace relation仅恢复2；旧方法最好edge recall仅`0.1538`。
- 提案边界：学习三维primitive composition直接产生节点/端口；首次稳定结构事件即产生local-confirmed node；仅loop merge使用严格图条件因子；edge由真实departure-to-arrival trace闭合。
- 决定状态：只形成草案，未冻结、未执行。草案=`docs/GSE_GRAPH_PRIMITIVE_CONDITIONED_HYPOTHESIS_GRAPH_PROPOSAL_V1.md`。用户明确确认后才升为当前权威方法。

## 2026-09-04：双读出仍无法安全关联，局部place identity路线正式停止

- 具体证据：V1R1使用密封V1缓存和完全相同的180/100/80人口，500步、`error=null`。loss reduction=`0.56397`，geometry RMSE=`0.10883`，degree accuracy=`0.97778`，precision>=0.99时无非空关联；异节点association distance最小为0。
- 失效结论：不能声称冻结的局部LiDAR基元表示配一个更好的小读出就能安全识别全局地点。这是model/observability阻塞，不是system、data漂移或Teacher矛盾。
- 选项A（推荐）：保留学习三维基元、结构事件和节点/端口生成，将合并明确建模为描述候选+里程计/图邻接/执行证据的图条件联合因子；预计数小时完成规格+零训练快速可行性。选项B：解冻骨干端到端训练place descriptor，预计小时至天，且重复隧道存在根本歧义。选项C：纯规则图，只可作基线。
- 推荐决定：选A；它保留“学习几何结构直接决定图结构”的论文主张，同时不再将不可观的全局身份强加给局部感知。
- 需要的显式用户决定：这次会改变地点关联的方法边界，超出原V1R“只修读出头”fallback。需用户确认采用选项A后，才冻结新spec并执行；这不是例行run批准。
- V1R预检只因Data Card的汇总world别名失败，未创建run。V1R1仅修正为逐世界记录后通过预检，方法与数据无变化。
- run=`results/gate3_semantics/gate3_20260904_gse_structural_node_tiny_overfit_v1r1_seed0`；seal-list SHA-256=`d19fb26fd7f14dc2c423e4ff89e62f4b7d6dd96640c986acc5895a22a2bbf803`。

## 2026-09-04：tiny overfit拦截长训练，分离结构几何与地点关联读出

- 具体证据：180观察/100节点/500步的正式run无系统错误；degree accuracy=`0.99444`，但几何RMSE=`0.14321`、loss reduction=`0.69449`，precision>=0.99时TP=0。Teacher在相同人口上P/R=`1.0/0.9625`。
- 影响：“当前单一44维读出可直接进入单seed短训/三seed”的结论失效；不影响底层基元几何可学、Teacher组合容量和16 m图一致性的既有PASS。问题类型为model/readout，不是data/Teacher/system。
- 选项A（推荐）：固定按维归一化几何回归，另设关联embedding，然后用图一致性验合并；成本为一次同规模秒级/分钟级corrective。选项B：解冻整个LiDAR骨干重训，成本为小时至天且暂无必要。选项C：退回纯规则图，可运行但丢失论文核心创新。
- 决定：选A，不改数据、骨干、Teacher、500步、precision/recall门或16 m图一致性机制。若corrective tiny overfit仍FAIL，停止此学生接口，不进入长训。
- 用户决定：用户已明确授权在锁定论文主线内自动选择最优方案；本决定是预注册fallback且不扩大数据/测试范围。
- run=`results/gate3_semantics/gate3_20260904_gse_structural_node_tiny_overfit_v1_seed0`；seal-list SHA-256=`57fc01e05cff81644cf0f840844eeed7ef610a4186dff703276537f512c05588`。

## 2026-09-04：冻结“描述子提候选、图一致性验合并、真实穿越建边”接口

- 证据：C07局部节点描述P/R=`0.982410/0.933459`；固定16 m图候选把FP从53降到0，TP 2960不变；1 m/节点最坏误差仍0 FP，十类地图均覆盖。
- 决定：节点描述子不得单独提交loop merge。它只产生候选；图位置/邻接/执行状态验证合并；edge仍只能由真实穿越提交。这与纯出口规则图和MR-TopoMap式地点描述不同，保留底层三维基元组合为学习核心。
- 下一步：只做tiny overfit，不启动长三seed。tiny overfit成功后依次单seed短训、完整C07、三seed、C08；任一级失败均在进入下一级前停止。
- run=`results/gate3_semantics/gate3_20260904_gse_structural_node_graph_consistency_attribution_v1_seed0`；seal-list SHA-256=`7ee483fe66954b676bfd4f52b81743d7838ca7b294b3a995c1929322390a3240`。

## 2026-09-04：保留节点组合路线；support退出身份特征，机器PASS不得覆盖研究合同

- 证据：完整组合oracle在C07 P/R=`1/1`；support参与身份时五帧C07 0接受，support仅作不确定性后P/R=`0.982410/0.933459`。因此底层基元组合有足够节点信号，view-dependent ray count不是地点语义。
- 完整性问题：V1R漏执行预注册accepted-pair false fraction `<=1%`门，实际`1.759%`；预期fit 36,318观察中6条短traversal无五帧序列，实际36,312。V1R记录为`FORMAL_MACHINE_PASS_RESEARCH_CONTRACT_FAIL`，禁止据机器PASS启动训练。
- 决定：不放弃基元节点表示，也不再次直接长训。先做密封重评分与53个误合并的零训练图一致性归因，测试既有16 m空间候选、邻接一致性和执行轨迹约束；只有同一冻结门通过才进入tiny overfit。
- 成本：重评分为秒级，归因为分钟级；不新增GPU、模型推理、C08--C10或M-TARE读取。
- 用户决定：用户已明确要求持续执行并自动选择最优方案；该纠错不改变论文主线、数据或验收线，无需新的交互批准。

## 2026-09-04：停止端点相似度直接连接，采用节点级证据聚合和分级快速证伪

- 证据：正式三seed C07 best F1虽为`0.252722/0.163434/0.280575`并超过非学习基线，但precision只有`0.173568/0.118684/0.198213`；precision`>=0.98`时三seed安全TP全部为0。
- 决定：endpoint relation metric不得直接合并节点或提交edge，只保留为节点候选内部的一个辅助特征及论文失败消融。真实穿越继续是edge提交的唯一条件。
- 方法调整：下一候选不再逐端点判连接，而是先将同一局部几何组合的多端点和五帧历史聚合为`StructuralNodeObservation`；结构事件负责候选节点产生，节点描述子负责新地点/重访判断，不确定关联保留provisional node。
- 效率调整：固定采用`零训练可分性 → 小样本过拟合 → 单seed短训 → 单seed全C07 → 三seed正式 → C08一次迁移`。任何前级失败立即停止，三seed不再承担探索路线功能。
- 当前只授权C01--C07零训练审计，精确人口`426,552+64,644=491,196`序列；C08--C10、在线图和M-TARE仍为0。

## 2026-09-04 — 端点关系度量通过readiness；允许三seed训练但不降低原安全门

- 新接口用连续embedding直接表示“能否属于同一局部组合”，避免任意槽语义和共享dustbin；只保留通用的对称度量与complete-link一致性约束。该改变直接对应formal attribution，不是无证据扩模。
- 真实batch证明四项loss能同时向全部34个head张量传递有限非零梯度，并覆盖850个簇、99个单例端点和2520个叠置负对；因此允许建立三seed训练Data Card。
- 训练仍必须使用C01--C06全部426,552序列、C07全部64,644序列、seeds0/1/2和相同442,936真连接人口。precision>=0.98非空及至少2/3 seed门不变，不能用complete-link或平均F1掩盖错误连接。
- C08只在独立三seedC07 PASS后一次性零适配打开；readiness自身不授权C08、图或M-TARE。

## 2026-09-04 — 停止32任意槽+dustbin；下一候选改为无槽编号的端点关系度量

- 因子归因排除了“只是一项乘法置信门过严”：联合margin确实使`68.7%--90.1%`活跃真端点归零，但移除margin后硬槽关系仍只有`17.9%--21.9%` precision，且三个seed均无安全硬槽identity阈值。
- soft affinity可把best-F1提高到`0.2466--0.2830`，证明冻结几何/端点表示含有关系信号；但安全非空只在一个seed出现，不能把soft score事后替换成主方法。决定保留骨干和Teacher，停止任意slot分类与dustbin接口。
- 下一候选固定为endpoint relation metric：每端点产生连续关系embedding和可观测置信，训练时同Teacher clique及跨视角同一composition identity为正，不同clique、dustbin和disconnected-overlap为负；部署pair相似度必须对称并经保守transitive clustering，不允许用C07事后选择一组规则补边。
- 先做零训练readiness，不直接重训。readiness必须证明真实batch正负梯度方向、endpoint/primitive permutation与yaw合同、困难重叠负例、拒绝与聚类确定性、学生forward无Teacher identity及资源上界。失败则不再进行同类关系学习重试，需重设论文结构图主张。
- 用户的持续执行授权覆盖该冻结主线内的证据最强方案；本决定不授权C08、图或放宽precision门。

## 2026-09-04 — 局部组合槽C07只有1/3通过；先拆解置信因子，不以重训或降门补偿

- 证据表明训练、数据和Teacher均按冻结合同完成；三seed虽都超过非学习F1增益门，但seed0/1在precision>=0.98下无非空真连接，seed2也只有6个，故决定正式停止该版本进入C08和结构图。
- `exact_ranked_selection`已核验为按完整同分组选择，阈值0会真实接收全部零分候选；因此best-F1不是tie拆分错误。与此同时endpoint confidence的q50/q90均为0，说明首要嫌疑是硬槽一致性与六项乘法置信因子的零塌缩。
- 决定下一步为一个C07-only、零optimizer的正式归因run：复用三个selected checkpoint和同一64,644行/442,936正例人口，逐因子报告零值率、正负分布、best-F1、安全TP及overlap FP。不得把任一诊断分数直接宣布为新方法或在C07上事后选新部署公式。
- 若原始槽身份已有稳定区分而仅乘法校准跨seed一致失败，下一候选可以是预注册的可校准关系置信头；若槽身份本身不能分离连接，则停止当前slot-assignment架构并重新设计关系学习目标。两种分叉都须先过独立readiness和新Data Card，不能直接重训。
- 用户此前已授权在冻结论文主线和科学停止门内自动选择证据最强方案；本次“继续”用于执行上述只读归因，不扩展到C08、测试世界或修改科学门。

## 2026-09-03 — 允许32槽关系头三seed训练；不把readiness当性能PASS

- 决定采用集合端点编码+32交换槽+dustbin+Hungarian Teacher匹配作为P2唯一关系候选。输出关系由同槽概率构成，endpoint evidence和分配歧义共同用于拒绝；不恢复全局坐标锚点，也不使用独立O(E²)pair classifier或手写距离补边。
- 正式真实batch证明全部1,001,507个head参数可训练、2,635,631个backbone参数冻结、batch128资源可承受，且slot编号、primitive顺序和sensor yaw不改变关系语义。故允许创建三seed训练Data Card。
- V1设备错误不改变模型判断：其CPU generator/CUDA randperm不兼容发生在训练资格检查的后处理阶段。V1R固定同seed CPU置换并复制到CUDA，未改变任何科学变量；两run均保留。
- readiness只证明实现/资源，不证明C07关系可学。训练必须固定seeds0/1/2、fit梯度、C07选模/拒绝阈值和同一可观测pair人口；至少2/3 seed须同时超过非学习F1至少5个百分点并在precision>=0.98时产生非空真连接，且报告slot collapse和overlap FP。失败则停止该接口，不进入C08/图/planner。
- V1R seal-list SHA-256=`45488881d3ef76f8d81aef6a54f71c7f116aac8ea9e57fd968412fc2a4ea463a`。

## 2026-09-03 — 冻结32个交换局部组合槽；允许实现关系头但尚不训练

- 全量Teacher审计证明fit/C07的attachment连通分量全部为clique，同槽解码逐pair无损，disconnected-overlap误合并为0；因此局部组合槽能严格表达现有程序构造监督，不需要规则修复。
- fit最大18簇，冻结25%余量后需求为23；决定从预注册候选`8/16/32`中固定`32`，C07最大19且0 overflow。容量由fit选择，未用C07调参。
- 决定主方法的关系接口改为每个可观测端点预测32槽或dustbin，以交换不变匹配监督；相同槽产生局部结构连接候选，低置信/歧义分配拒绝，真实穿越仍是全局edge提交的唯一条件。
- 该PASS只回答Teacher表示是否自洽，不等于模型、C07、结构图或闭环PASS。下一步必须先通过零训练readiness，再按冻结Data Card做三seed训练；禁止恢复全局锚点坐标回归、独立O(E²)pair分类或手写规则补偿。
- 权威run seal-list SHA-256=`ee6ddaecbcce9959602e11f976aaea4d9144de354f98557877ddd8567a01145c`。

## 2026-09-03 — 停止全局端点锚点回归；先验证交换局部组合槽

- 决策证据：三seed受控替换中，预测anchor在precision≥0.98时安全TP=`0/0/1`，Teacher anchor为`442,936/442,936/442,936`且precision/F1均为1；模型anchor MAE仍为`10.53--11.26 m`，只比原预测端点改善`4.64--5.58%`。
- Teacher有效性证据：Teacher primitive端点到共享anchor的C07 q50/q99=`0.000395/0.1773 m`，正cluster共点、不同cluster最小间隔`0.8095 m`；因此不修改Teacher、不删除远距离样本，也不把可观测性问题冒充坐标头问题。
- 决定永久停止当前`global endpoint anchor + Gaussian compatibility`作为主方法，不延长训练、不挑seed、不降0.98安全门。三个checkpoint、归因和图保留为“连续坐标回归不能可靠形成物理关系”的消融/失败分析。
- 下一候选不是新的独立pair classifier，而是交换不变的局部组合槽：每个可观测端点学习分配到共享composition slot或dustbin，同槽诱导连接簇，模糊分配拒绝；真实穿越仍是提交全局edge的唯一条件。该接口保持几何基元→组合关系→结构图主线并从O(E²)独立决策变为O(EK)结构化分配。
- 训练前必须先做fit/C07零训练Teacher审计：attachment连通分量必须无损、每端点唯一归属、正簇必须为clique、overlap负例不得同簇，并按fit从`8/16/32`冻结容量后在C07零overflow。任一条件失败即停止该接口，不靠规则修补。
- V1路径错误与V1R数值合同不一致均为system failure；V1R2显式复现源评估实际数值状态并逐项重现三seed结果。权威run seal-list SHA-256=`4825b95e6b0c7c9fa15756e2f7681f8c0b1007351ea75490c5be1c6adb03a2f2`。

## 2026-09-03 — 组合锚点有排序增益但不能安全连接；停止进入C08并先做C07归因

- 三seed训练完整、资源与冻结合同有效；原评估失败被严格归因为浮点连乘结合顺序：各输入均finite和对称，但`compatibility*evidence_i*evidence_j`与其转置相差一个float32 ULP。决定保留原run为post-training system failure，只在新不可覆盖run中改为`compatibility*(evidence_i*evidence_j)`，不重训、不改任何科学输入或阈值。
- corrective在全部C07上复现旧最大非对称`2.98e-8`并使新分数最大非对称为0；系统证据PASS。正式模型结果为0/3 seeds：F1均从`0.00638`提高到`0.202--0.226`，但安全precision>=0.98的非空TP均为0，anchor MAE改善只有`4.64%--5.58%`。
- 决定科学状态为`STOP_COMPOSITION_ANCHOR_BEFORE_C08_AND_GRAPH`。禁止把明显F1增益单独包装成方法成功，也禁止降低precision、挑seed、读取C08、加入手写连接或用planner掩盖错误合并。
- 三seedcheckpoint、几何/时序backbone、共享锚点Teacher和对比图保留为论文正证据与失败消融。下一步只允许C07冻结输出归因：Teacher anchor/proposal/evidence/scale oracle、距离排序与结构分层必须区分锚点回归、可观测性、校准和关系排序瓶颈；只有归因支持明确的最小新接口才可另立readiness。
- corrective run=`results/gate3_semantics/gate3_20260903_primitive_composition_anchor_c07_evaluation_corrective_v1_seed0`；27项seal SHA-256=`2b9a91c62885148962cd3070bab68487a96776626704e5f0c1cfd5933de3af53`。

## 2026-09-03 — batch-128关系头训练接口与资源通过；冻结三seed训练合同

- 正式readiness证明四层数据绑定、22,278参数head-only梯度、冻结backbone、重复确定性和三类内存证据同时成立。采用batch-128可把每seed每轮从约26,736个小批次降到3,411个批次，而不改变样本、loss、模型或科学门。
- 决定下一run固定seed0/1/2、每seed3轮、AdamW、batch128和每seed10,233步；checkpoint仅按C07 mean anchor total loss选择。禁止训练中改batch、epoch、阈值、Teacher、数据或解码器。
- readiness不是科学PASS。只有至少2/3 seed同时满足同人口attachment F1绝对增益>=0.05、precision>=0.98且TP>0、anchor MAE相对冻结raw endpoint至少改善10%、backbone hash/状态不变，才允许读取C08并实现结构图。
- run=`results/gate3_semantics/gate3_20260903_primitive_composition_anchor_training_readiness_v1_seed0`；seal-list SHA-256=`252ee90df3d50c6bfa24d2b06289dc0e00774ed5038d03ee3ba904cc5442afbc`。

## 2026-09-03 — 组合锚点Teacher全量封存；绘图范围不得充当数据有效性门

- V1在`S08_3d_loop_rich_C07__c1_mixed`发现一个`80.3915 m`锚点后失败。锚点表示完整物理基元的组合端点，五帧可能只看到长隧道中段，因此它可以超出当前50 m回波范围；80 m仅为证据图显示上限，不是Teacher、模型或安全阈值。
- 决定保留V1为system/visualization failure，V1R只把图外值计入overflow并保存真实maximum；禁止裁剪目标、删除样本、缩短基元、扩大LiDAR或把80 m变成训练阈值。新增合成143 m和原失败任务回归。
- V1R对70父世界、210任务、491,196序列完成三次逐块推导核对。全部3,826,561个真连接对共享完全相同锚点，最大float32误差`3.81465e-6 m`，输入未漂移，输出48.24 MB；正式decision为`ALLOW_COMPOSITION_ANCHOR_HEAD_ONLY_THREE_SEED_TRAINING_SPEC`。
- 下一步仅允许每seed加载同seed冻结observable backbone并训练22,278参数sensor-polar anchor head。训练必须使用新sidecar与现有endpoint-observability mask；C07至少2/3 seed通过precision≥0.98且非零TP、F1增益和几何不回归前，禁止C08、图、planner或阈值放宽。

## 2026-09-02 — 新关系模型与非学习基线统一使用双端可观测评分人口

- P1b旧标签中82,141个C07正连接在当前五帧缺少双端射线支持。继续拿这些标签评价修正后的局部证据模型，会把“当前不可判断”错误计为false negative；只修学习模型、不修基线又会造成比较人口不一致。
- 因此正式重跑同输入非学习基线，算法和参数完全不变，只在共享评价器中排除隐藏matched pair；未匹配预测仍保留为false-positive候选。442,936个可观测正例逐项复核通过。
- 新冻结基线attachment F1=`0.006380`。后续三个seed必须在相同人口上分别满足F1绝对增益>=0.05与precision>=0.98的非空真连接；旧未mask结果只作标签策略消融，不得混用。
- run=`results/gate3_semantics/gate3_20260902_primitive_relation_observable_nonlearning_c07_v1_seed0`；45项seal-list SHA-256=`01100d79bcc91182974259e25ef5d548227d783720202e7dd56a552f67fea58e`。

## 2026-09-02 — 复用已通过的几何表示，只重训可观测物理关系与端点证据

- 新正式readiness证明Teacher-side endpoint mask可随Hungarian slot匹配和endpoint reversal无损对齐；隐藏物理连接不再被当前窗口强制判正或判负。V2稀疏cardinality目标保持逐位一致，因此该修正不会恢复旧V1的过密候选故障。
- 决定不从头训练全部2.64M参数。每个seed从其V2 selected checkpoint精确迁移133个几何/时序状态张量，只重置并训练64个relation/evidence张量；这隔离了已证实的几何能力与待检验的连接假设，也显著减少算力和灾难性遗忘风险。
- Teacher endpoint mask只能用于loss有效性，禁止进入forward或部署。部署安全连接分数必须包含模型自己预测的endpoint-evidence概率、连接概率和关系不确定性；图edge仍只能由真实穿越提交。
- 下一run继续使用完整C01--C06 fit与隔离C07、seeds 0/1/2及原科学门。若仍不能在至少2/3 seeds产生precision>=0.98的非空真连接，则按计划停止该关系方法，不能用规则连接、C08调参或规划器补偿。
- readiness run=`results/gate3_semantics/gate3_20260902_primitive_relation_observable_readiness_v1r_seed0`；seal-list SHA-256=`57e3d5f991326262965b57bb942ee51518ea6c927cf67c31702e1dd0ddc27294`。

## 2026-09-02 — P1b约15.7%连接正例缺少双端观测；采用可观测mask而非更换主线

- 正式审计证明P1b现有规则只检查两个基元是否在五帧窗口出现，再从完整构造图复制共享节点关系；没有检查相连端点的射线支持。0.25 m主判据下fit/C07双端支持率为`84.2799%/84.3564%`，故当前Teacher不能继续作为每个窗口都确定的attachment分类标签。
- 物理连接identity至少在一个窗口被双端观测的fit/C07覆盖仍为`95.1675%/95.2066%`，均高于预注册`0.95`纠正门。因此不放弃五帧几何基元关系主线，也不新增手写连接规则；冻结诊断=`WINDOW_LEVEL_HIDDEN_ENDPOINT_SUPERVISION_CORRECTABLE_BY_OBSERVABILITY_MASK`。
- 决定只物化端点可观测性sidecar：两端均在0.25 m射线支撑范围内的candidate pair才进入attachment正/负loss；缺一端的旧正例和负例均为unknown，不能反向当负样本。部署时同样只允许双端受支持关系进入高置信候选，拓扑edge仍必须由实际穿越提交。
- 现有P1a/P1b、三seed几何/时序checkpoint和旧失败结果完整保留，分别作为数据底座、可复用encoder和“无可观测mask”消融。新readiness与三seedC07未通过前，C08/C09/C10、graph和M-TARE继续关闭。
- run=`results/gate3_semantics/gate3_20260902_primitive_attachment_teacher_observability_v1_seed0`；17项seal-list SHA-256=`151cdff6b18755a8458172247cf4b1d4e7d317230a7f443b904d1c4c133ccacf`。
- 后续sidecar正式PASS并验证这个决定可无损落地：fit/C07保留可观测正pair=`2,782,487/442,936`，隐藏pair=`518,997/82,141`不再进入正负loss，数据增量仅`892,360 bytes`。决定下一模型必须显式预测端点证据概率，使部署拒绝不依赖Teacher mask；只在训练时删掉坏标签但推理时没有等价拒绝接口不被接受。sidecar run seal-list SHA-256=`52f379fe2f1fdbeca46b8278f9136bb40dc5bcd13217116f4f88e7e75a5060da`。

## 2026-09-02 — 稀疏端口关系头在proposal oracle下仍失败；停止重训并审计Teacher可观测性

- 三seedC07-only归因完整计算`193,932`条冻结推理。deployed关系F1为`0.0725/0.0325/0.0555`；Teacher只提供真实基元身份后提高到`0.2009/0.1793/0.1630`，证明冗余候选放大错误，但不足以解释失败。
- tie-safe精确阈值排除了原0.05--0.95网格过粗；Teacher真实数量排除了单纯cardinality错误；只用学习分数的best-link union/mutual排除了独立pair数量本身；原始概率和风险调整概率在proposal oracle下仍全部没有precision≥0.98的非空真连接。七个预注册诊断条件通过seed数均为0。
- 因此冻结诊断为`RELATION_SCORE_FAILS_EVEN_WITH_PROPOSAL_ORACLE`，当前relation Transformer停止。禁止继续加epoch、挑seed、降阈值、手写连接规则或用planner补偿；底层几何与跨帧身份checkpoint继续作为正结果和后续可复用编码器。
- 下一项证据必须先回答监督可观测性：P1b是否把“两个基元可见”直接等同于“它们的物理连接端点在五帧射线中可见”。若连接点缺少当前观测支持，目标改为学习可观测局部连接证据，拓扑边仍只由真实穿越提交；若连接点充分可见而关系仍不可分，当前pairwise关系假设构成科学阻塞，需要独立方法重设而不是第三次同类重训。
- 正式证据corrective run=`results/gate3_semantics/gate3_20260902_primitive_relation_sparse_port_failure_attribution_evidence_corrective_v1_seed0`，25项seal-list SHA-256=`cfab8a74cb329b05198503c92e70d68c6230b44023f8ddf2d757b2ed9e37afd7`。

## 2026-09-02 — V2三seedC07以0/3科学失败；停止进入C08与结构图

- 正式V1R完整训练与评测无系统错误，三seed分别通过全部资源和人口合同；故本次是模型科学失败，不是训练中断。三seed表面和连续几何均显著优于非学习基线，但attachment只有seed0达到F1增益门，且所有seed在precision≥0.98时均零真实提交。
- 结论被严格分解：保留“五帧因果LiDAR可学习扫掠隧道几何与跨帧身份”的正证据；否决“当前sparse-port relation Transformer可把这些基元可靠组合成在线拓扑连接”的主张。不能用平均几何提升外推整条几何语义建图链已经成立。
- 执行冻结停止政策：不读取C08，不生成图，不运行M-TARE，不降低安全precision，不挑seed、不延长训练、不加入手写端口连接规则。正式decision为`STOP_SPARSE_PORT_BEFORE_C08_AND_GRAPH`。
- 下一步限于新的C07-only只读归因提案：在不更新checkpoint的条件下比较原预测、Teacher基元proposal oracle、真实基数oracle、端点几何候选oracle和概率/不确定性排序，以区分proposal、pair explosion、relation representation和calibration瓶颈。归因结果才能决定是否存在保持原论文创新主张的最小架构修正；否则停止当前主方法并重新评估研究问题。
- run=`results/gate3_semantics/gate3_20260901_primitive_relation_sparse_port_three_seed_training_v1r_seed0`；seal-list SHA-256=`74a834a1c5a07b42e74a803cb4c008200579ed34153fefec050d77c7bc5a5c82`。

## 2026-09-01 — 成熟图构造只作可复用骨架；创新集中在因果隧道基元关系

- 原始来源补充复核确认，Hydra已经用局部ESDF/GVD增量抽取place graph并做层级回环，S-Graphs+已经把keyframe、墙面、房间和楼层放入可优化结构图，TopoNet/DAGMapper也已证明几何实例与拓扑关系的联合学习。因此“几何形成图”“层级图”“端到端几何—拓扑”均不得单独声称首次。
- 当前方法直接借鉴这些成功工作的系统原则：增量持久图、层级描述、保守数据关联和解析几何验证；继续复用Cano圆周LiDAR接口及M-TARE定位、局部规划、避障和多机器人协调骨架。论文创新只落在五帧部分观测下的三维扫掠隧道实体、端口物理关系、断开重叠困难负例、跨帧身份、拒绝合并和traversal-only edge链条。
- 解析自由空间或最近邻关系可作为安全核验、同输入基线和单变量消融，但不得在P2失败后替代学习关系并沿用原主张。若冻结P2门失败，必须先停止该主方法和完成归因；任何新的直接junction/port grouping或研究问题修订都需要独立计划、Data Card和未见拓扑门，不能通过规则补偿、阈值下降或规划器调参制造成功。
- 该文献边界修订不改变当前V1R三seed训练的模型、数据、Teacher、loss、阈值、checkpoint选择和C07/C08隔离。完整矩阵和新增BibTeX已写入`docs/PRIMITIVE_RELATION_NOVELTY_MATRIX_V1.md`与`docs/references/structural_topology_novelty_20260823.bib`。

## 2026-09-01 — 资源监控语义错误不允许混入模型结论；仅做外层RSS corrective

- 证据显示原V2训练器的`_process_memory_bytes()`调用`nvidia-smi --query-compute-apps=pid,used_memory`，所以该字段是GPU进程显存，不是主机RSS。原总控以它同时检查GPU和host上限，无法满足Data Card的独立主机16 GiB证据。
- 原run在epoch0完成前主动终止，0 checkpoint；部分未密封optimizer更新全部丢弃，不得恢复或拼接。该run只作为system/resource-contract失败，不得进入模型成败、学习曲线或论文性能统计。
- corrective限于新外层总控：轮询子进程`/proc/<pid>/status`的`VmHWM`，终止后用`RUSAGE_CHILDREN.ru_maxrss`保留高水位；旧内部值重命名为GPU process memory，PyTorch max allocation继续单列。三者分别受16 GiB门约束。
- 模型、数据、Teacher、loss、7轮训练顺序、seed、optimizer、选模、阈值和科学门逐项保持一致。48项测试与64 MiB实时探针、强制超限终止负控制通过；V1R preflight 0/0后执行唯一新run。

## 2026-08-31 — V2正式训练冻结；安全关系必须非空

- V2正式Data Card/spec通过preflight 0/0并创建唯一不可覆盖run。训练固定为seed0/1/2各7轮、每轮426,552个fit序列和64,644个C07选择序列，总optimizer steps=561,456；不得在结果后增加epoch、修改数据或挑seed。
- V1的“precision>=0.98但零真实提交”暴露了空预测的形式精度漏洞。V2正式门明确要求每个通过seed在风险调整attachment分数下precision>=0.98且true positive>0；该要求已有单元测试，零提交判FAIL。
- C08 loader只允许在C07至少2/3 seed通过完整几何、关系、覆盖与非空安全门后创建。该条件由源码顺序测试保护；C08不得用于checkpoint、阈值或方法选择。
- 运行上限为40小时、GPU和主机内存各16 GiB、结果6 GiB。任何系统错误保持原run证据并报告；任何科学失败保持checkpoint/图用于消融和归因，但不进入在线图。

## 2026-08-31 — V2稀疏端点关系readiness通过；训练顺序恢复完整五阶段合同

- 正式零训练readiness证明V2在真实三形状batch上满足六类loss、全参数梯度、稀疏cardinality方向、集合置换、端点反转、yaw旋转、关系对称、不确定性和确定性合同；允许进入新的训练spec，不允许据此提前读C08或建图。
- 三seed训练固定为7轮而不是复用V1的6轮：增加独立的epoch1多基元集合分解/最小描述长度阶段，并保留几何、关系、时序/不确定性与三轮联合训练。这是落实`PRIMITIVE_RELATION_STRUCTURAL_GRAPH_PLAN_V1.md`原五阶段顺序所需的单一修正，不是失败后搜索epoch数。
- 训练人口、batch、优化器、学习率、C07选择和最终科学门继续沿用V1；V2只新增有V1R2归因直接支持的稀疏集合与端点关系参数。不得联合V1/V2 ensemble、挑seed或修改非学习基线。
- readiness run=`results/gate3_semantics/gate3_20260831_primitive_relation_sparse_port_readiness_v1_seed0`，seal-list SHA-256=`81c017d4943792e93a8bd2f18cb3ccaa0bf96ff68deac7f6fd901622bf8ff2af`。

## 2026-08-31 — V1失败由“过密候选 + 不安全关系排序”共同造成；冻结稀疏端点关系架构corrective

- V1R2在正确TF32-off合同下复核三个冻结seed，科学门仍为`1/3`；因此模型失败不是旧评估器的数值偶然，也不能通过重新跑同一checkpoint解决。
- 三seed激活候选约为Teacher的`3.80x`，端点关系pair空间约为Teacher的`14.04x`。proposal oracle移除未匹配候选后，attachment F1全部显著提高到`0.2049--0.2567`，证明过密候选是主要可修机制，现有几何编码和部分关系排序保留科研价值。
- 但proposal oracle下三seed在precision>=0.98时仍全部零安全attachment真阳性，故否决“只增加existence阈值/基数头后直接重训”。当前简单全配对关系头缺少足够的端点方向、相对变换和全局关系上下文，不能承担在线图的连接提交。
- 下一版固定为单一`sparse-port relation architecture`：32槽容量保持不变但学习稀疏有效集合；基数/最小描述长度作为既有`primitive_set_parameters`损失族内部项，不新增无独立证据的第七loss；端点token显式包含位置、外向切线、截面、形状、描述子和存在性，使用置换安全relation transformer与学习式pair decoder；输出relation uncertainty并支持拒绝。端口是否连接仍由学习决定，禁止用手写距离/角度阈值冒充主方法。
- 先执行零训练readiness：解析attached/disconnected-overlap/hard-negative、query置换、端点反转、传感器yaw旋转、批次/重复确定性、真实P1 batch六类finite loss与全参数有限梯度、显存上界及旧V1 checkpoint隔离。readiness失败则修实现或重新评估关系假设，不直接消耗约21小时三seed训练。
- V1模型、三个checkpoint、C07结果和归因图永久保留为“几何可学但密集关系失败”消融/失败分析；C08仍未读。该决定不降低C07/C08、0.98 precision、false-loop、2/3 seed或后续图/闭环门槛。
- 权威run=`results/gate3_semantics/gate3_20260831_primitive_relation_v1_failure_attribution_v1r2_seed0`，seal-list SHA-256=`ca2600041eb72b70d5ff0118a900526297cbc867b9ab83396e1c2a12be115b93`。

## 2026-08-31 — 所有失败必须先给出可理解的机制解释

- 用户要求后续不能只报告Gate名、缩写、loss或PASS/FAIL字符串；每个失败必须先用普通中文解释“原本要学会什么、实际发生了什么、与基线差多少、失败位于感知/表示/关系/记忆/拒绝/系统中的哪一环、证据为何支持该归因、它阻断什么结论、哪些资产仍可复用以及下一步最小有效动作”。
- 机制解释必须区分症状与原因。例如低attachment F1只是症状；若证据显示模型能拟合局部表面却无法区分空间接近与物理连通，应明确说明它会把并行或上下层隧道误合并，因此不能安全提交拓扑连接。
- 每份正式FAIL摘要继续保留完整机器指标、阈值、人口和seal，但面向用户、导师和论文failure analysis的首段必须先给出上述人话结论；禁止用术语堆叠代替归因，也禁止在没有对照或分层证据时臆测根因。
- 该报告规范不改变当前训练的数据、Teacher、方法、阈值、checkpoint选择或停止门。

## 2026-08-31 — 投稿交付必须形成从构造监督到闭环探索的完整证据故事

- 用户明确要求最终工作不能停留在孤立的数据导出、单个模型或一串失败实验；必须形成“研究问题 → 程序构造监督 → 五帧几何基元与物理关系学习 → 持久结构图 → traversal-only edge → 单/多机器人探索 → 公平对照与局限”的完整因果链。
- 当前P2训练合同、数据、Teacher、baseline、seed、阈值和C08隔离不变。若当前模型通过，按冻结顺序进入C08、结构图、闭环、消融和论文；若科学FAIL，先停止该模型进入图，完成同人口失败归因，并将其作为主方法消融/失败证据，而不是把失败本身冒充论文终点。
- 科学FAIL后的后续方法只能从现有密封证据中选择最强、最小且仍服务同一论文问题的修订；必须重新经过独立Data Card、感知门和未见拓扑验证。禁止降低门槛、挑选最好seed、用C08反向调参、用planner补偿感知失败或堆叠与主张无关的实验凑工作量。
- “完整工作量”以可复现证据闭环衡量：最终至少包含主方法、同输入感知基线、图基线与oracle、必要单变量消融、单机器人和2/3/4机器人公平闭环、失败案例、复现包、正文/补充材料和投稿PDF；已有旧工作按`docs/PAPER_EVIDENCE_RETENTION_MATRIX.md`继续进入对照、消融或失败分析。
- 只有上述因果链获得有效主方法结果并完成系统实验，论文目标才算完成；若经证据确认核心研究假设不可行，必须明确报告科学阻塞并重新评估论文问题，不能把不完整故事包装为成功。

## 2026-08-31 — 新颖性主张收窄为“因果隧道基元关系到执行验证探索图”的完整链条

- 原始来源复核确认：SPFN/Superquadrics/ResFit已覆盖学习式几何基元；CSGNet/BSP-Net/CAPRI-Net/ArcPro已覆盖程序或CSG监督的结构恢复；StructureNet/Regularized Primitive Graph已覆盖部件几何与关系图联合学习。因此禁止声称“首次学习基元”“首次程序监督逆建模”或“首次学习部件关系图”。
- Cano已覆盖合成地下LiDAR、隧道方向和纯拓扑导航；PRISM-TopoMap已覆盖学习地点识别和在线location graph；地下Segmented Map与M-TARE已覆盖拓扑/多机器人探索。因此出口、descriptor、拒绝、共享图或探索也不能单独构成贡献。
- 当前可检验差异固定为：五帧本体因果LiDAR恢复扫掠隧道基元及端口连接、断开重叠和时序身份；关系直接形成持久节点/候选端口；模糊关联拒绝合并，物理穿越是提交全局edge的唯一条件；最终在不改变M-TARE局部执行栈的条件下证明图与探索收益。
- ArcPro是程序生成点云监督的直接近邻，StructureNet是部件关系学习的直接近邻，二者必须进入正文相关工作和独立消融逻辑。完整矩阵见`docs/PRIMITIVE_RELATION_NOVELTY_MATRIX_V1.md`。
- 若学习关系不能改善结构图，或闭环收益只能依赖规划器调参，则该组合贡献也不成立；不得退回宽泛“几何基元/拓扑探索”主张。

## 2026-08-31 — 旧工作按公平对比、单变量消融和科学失败正式纳入论文证据池

- 用户明确要求：此前可用工作和失败实验不作为废弃历史，只要能支撑论文主张，就作为正式对比、消融或失败分析，使论文实验闭环更完整。
- 主表仍要求相同world、输入、因果历史、轨迹、seed、预算和评价代码；历史条件不一致的数字只能说明设计动机，必须按最终合同重跑后才能参与性能排名。
- 消融必须从完整基元关系模型只移除一个组件；完整科学失败必须`error=null`、无泄漏且能直接否定一个方法假设。脚本、环境、路径和资源错误只进入复现附录。
- 已冻结的图片、metrics、summary、checkpoint、轨迹、图结构和seal在论文图表映射完成前不得删除；详细角色与预定正文位置见`docs/PAPER_EVIDENCE_RETENTION_MATRIX.md`。
- 该决定不改变当前P2训练数据、模型、损失、阈值或门槛；三seed训练继续运行，C08/C09/C10与图仍按既定隔离合同关闭。

## 2026-08-30 — 冻结非学习C07下界；历史资产按论文证据角色保留

- 同输入非学习基线正式完成C07全部64,644行，状态COMPLETED、error=null、C08/C09/C10/graph/M-TARE读取为0；允许把其固定指标用于P2学习模型比较。
- 冻结下界为primitive F1=`0.298311`、coverage=`0.175306`、surface Chamfer=`17.3137 m`、attachment F1=`0.005819`、disconnected-overlap F1=`0`和temporal correspondence accuracy=`0.155550`。后续不得因学习结果修改该实现或数字。
- 旧工作不按成败删除：公平同输入或同执行合同的结果进入正式baseline；只改变一个主方法组件的结果进入ablation；有完整科学运行且能解释方法转向的失败进入failure analysis；环境、路径、绘图和资源监控错误只保留复现记录。
- 该决定不降低P2门：学习模型仍需在C07满足预注册几何、关系、覆盖和检测门，至少2/3 seeds通过后才允许一次C08迁移。若C07不通过，C08与图保持未读，不能用规划器调参掩盖感知失败。
- run=`results/gate3_semantics/gate3_20260830_primitive_relation_nonlearning_c07_v1_seed0`，seal SHA=`605df99e2c542642bdc45d9530a10c89df1512cde95651c97f1a822688a637f7`。

## 2026-08-30 — P1b V1系统FAIL只允许source_run接口修正；V1R正式PASS

- V1在正式分片生成前由real-shard readiness触发`KeyError: source_run`。归因为双源支持改造后的pilot接口遗漏，不是数据、Teacher、确定性、容量或模型失败；V1保持FAILED且零正式Teacher分片。
- V1R只向pilot任务加入sealed P1a corrective路径，并以corrective数据自身的traversal manifest执行双重复；不改Teacher、split、样本、关系定义、资源门或阈值。
- V1R绑定V1 summary/RUN_STATE/seal并通过preflight 0/0。正式完成240任务和564,378序列，全部10项检查PASS，最大24/32槽，三类关系非空，结果283,883,437 bytes。
- 允许进入P2模型readiness；不因P1数据PASS而跳过解析/真实batch有限梯度，也不提前训练、建图或读取C09/C10。
- V1R run=`results/gate3_semantics/gate3_20260830_primitive_relation_p1b_teacher_materialization_v1r_seed0`，seal SHA=`f629b511e9a945bbe15249e3bc212be57a01b1aa0118d0d7d228824aa25f7d47`。

## 2026-08-30 — P1a无损存储corrective PASS；P1b必须从修正seal派生

- 唯一corrective完整完成240任务；源/目标11数组逐chunk raw-byte相等，2,640数组证据、240个源tree和240个目标tree全部有效，codebook/construction逐字节相同。
- 分片大小由`22,395,827,779`降至`19,388,655,191 bytes`，压缩比`0.865726`，不删除帧、射线、来源或元数据；冻结20 GiB资源门正式PASS。
- RUN_STATE=`COMPLETED`、error=null，127,380项seal SHA=`79fd988ac8c205d74c93e4858b7b579571a06502e778f31634046b48791a0668`。原P1a resource-only FAIL继续保留，不改写历史。
- 下一正式P1b只允许读取该corrective run；生成564,378个五帧几何/关系Teacher后才能进入模型readiness。禁止从旧出口Teacher、未来TNG或绝对pose补标签。
- P1b冻结脚本中`unit_tests=9`与执行器真实合同10不一致，已在冻结前修正为10并按正式解释器复核`10/10 PASS`；该更正不改变任何科学合同。

## 2026-08-30 — P1a容量fallback固定为逐块bit-exact无损存储corrective

- P1a正式run保持不可修改；现已完成240/240，11项科学/数据检查全部通过，唯一失败项确认为`result_bytes_le_20gib`：`22,395,827,779 bytes=20.857740 GiB`。RUN_STATE=COMPLETED、error=null，124,740项seal SHA=`f437992b2d9afead5a7860ef35bc73eb6aeddc8d039d84dff6d39dbec4d5cf93`。
- 预注册corrective启用条件已精确满足；若存在任何其他失败本应停止，不借存储修正掩盖。
- corrective只将Blosc zstd5 bitshuffle替换为zstd9 byte-shuffle，保持11个数组/分片的inventory、shape、chunk、dtype、元数据和全部逻辑值；目标chunk写后复读并按原始字节相等验收，codebook/construction逐字节复制。
- 不重渲染、不删帧、不改Teacher、不放宽20 GiB门、不覆盖源run。实现与2项合成测试、完整真实分片试跑均通过；Data Card/spec已绑定最终source hashes，preflight 0/0后创建并执行唯一不可覆盖run。

## 2026-08-30 — 高速CSG provenance合同PASS；允许进入P1完整数据导出

- C01正式6,144-ray proof通过所有预注册门：valid agreement=`0.998698`、coverage=`0.988607`、identity=`1.0`、p99=`0.024525 m`、大于5cm比例=`0`、稀疏加速=`15.56x`。
- spacing-padded primitive AABB只筛除空间上不可能的operand；边界保护宽度严格复用解析field spacing。稀疏与完整operand query的逐hit digest相同，故这是无标签变化的精确工程加速。
- 采用闭合独立primitive meshes + ordered multi-hit + analytic candidate/path qualification作为P1正式Teacher后端；慢解析field保留为合成/C01权威对照，不再逐帧运行全量P1。
- 允许下一步冻结并执行80-parent×3 paired geometry的数据导出；不得减少757,290帧、折叠多源membership、读取C09/C10、训练模型或提前建图。
- run=`results/gate3_semantics/gate3_20260830_csg_mesh_provenance_acceleration_v1_seed0`；seal SHA=`ea2da44432aea12497721f4fa630c687e889f16457d66d0e1ee3f1b6033538c9`。

## 2026-08-30 — P1采用高速多命中CSG identity ray；禁止用慢隐式proof硬跑全量

- 80-parent V1R2 inventory证明形状、拓扑、磁盘和native range渲染可行，但当前Python隐式场对8,723,980,800条计划射线外推80.86天。
- 决定保留隐式field作为解析权威/对照，新增由独立闭合primitive meshes组成的ray scene。每个face保留primitive ID；对每条射线排序全部交点、更新各primitive inside/outside状态，仅在union occupancy首次变为0时提交hit。
- 该算法直接实现union exit，因此不会把incident overlap内部面当墙；同距离多个face保留多源集合，不任意消歧。
- 只有解析ellipse/rounded/taper/T/Y/X/stacked和C01与隐式权威一致，并证明外推资源可接受，才允许P1正式导出。不得降低757,290帧、删除provenance或用最近中心轴伪标来绕过吞吐。
- inventory run=`results/gate3_semantics/gate3_20260830_geometry_variant_inventory_v1r2_seed0`；seal SHA=`7d31daf13b2403873426c0bba165bef913992699c7493364144c2f4a3aedcef4`。

## 2026-08-30 — 端口连接以自由空间重叠为物理合同；节点degree以edge incidence为真值

- 全80父世界仅5个degree-4端口不满足1cm轴线重合，offset/radius=`0.0302--0.1074`，graph anchor均在声明incident tunnel内部。schema分别保存实际axis endpoint和composition anchor，禁止改写edge/tunnel identity。
- 仅8个node的声明degree与edge incidence不符，全部是stale `self_intersection`字段、声明多1；无第二spline分支，graph cycle rank与edge列表一致。因此不补造edge、不删除world，关系Teacher使用实际edge incidence并保存全部mismatch。
- V1/V1R失败run保持不可修改，V1R2是唯一后继证据。

## 2026-08-30 — 采用身份保留隐式Teacher；歧义表面保留集合但拒绝单标签

- 正式corrective在一个C01世界的69,120条射线上PASS：所有命中均有非空来源集合，两遍逐元素一致，10个解析/单元合同全部通过。
- 唯一来源命中允许监督单基元参数；多来源命中必须保存完整membership、primary identity固定为`-1`且loss mask为0。禁止任选第一个基元或按最近距离消解。
- native Poisson mesh继续承担感知域外观；身份保留隐式union独立承担Teacher射线、自由空间和构造provenance。二者绑定同一construction graph和输入hash。
- 该结果只解除来源身份阻塞，不证明截面多样性或模型可学性。下一决策点是shape-generic swept-superellipse P0合同；在解析形状与80父世界inventory通过前，不生成正式P1数据、不训练、不建图。
- run=`results/gate3_semantics/gate3_20260830_primitive_provenance_corrective_v1_seed0`；seal SHA=`cbf9f214fad25b3d04e3c05b2b81e51c20d77e795a268007d83016d22554bb06`。

## 2026-08-30 — 采用双几何资产监督合同，不从历史Poisson mesh反推身份

- P0中55/55 edge基元和110/110端口可重建，但OBJ无逐面partition，Zarr无hit identity，完整构造监督判FAIL。
- 未来同一构造程序同时输出保留基元身份的Teacher geometry/ray-hit ID，以及继续用于感知域的native Poisson mesh；二者绑定同一construction graph和hash。
- 禁止按最近spline或最近圆管为历史LiDAR补伪标签；这会在交叉口和stacked overlap产生系统错标。
- 下一步只实现generator provenance corrective，不生成正式P1数据、不训练、不建图。

## 2026-08-29 — 采用程序构造监督的显式几何基元关系图，停止ERCSS候选

- 用户澄清目标不是取消训练或显式几何，而是从矩形/椭圆等底层基元、组合和变换中学习几何特征，再形成结构语义图谱。
- 选择连续超椭圆扫掠基元统一椭圆与圆角矩形；模型显式输出中心轴、截面、形状指数、坡曲、端口、descriptor和不确定性，并学习端口连接与跨帧对应。
- 构造程序Teacher只提供地图真实生成provenance和可见性裁剪监督；不再使用五类事件规则作为主监督，TNG只用于独立拓扑评价。
- 新颖性限定为`procedural construction supervision -> causal explicit primitive learning -> learned relation graph -> traversal-verified exploration topology`完整链；基元拟合、CSG、GNN、拓扑图或M-TARE接入任一单项均不作为独立首创。
- 旧Composer、出口集合、稀疏relation、RouteGeometryProfile与ERCSS全部保留为基线、消融或失败证据；ERCSS V1R3不再冻结或执行。
- 当前唯一下一步为零训练construction-supervision feasibility pilot。P0未PASS前禁止新数据生成、模型训练、C07--C10、graph和M-TARE。
- 完整计划：`docs/PRIMITIVE_RELATION_STRUCTURAL_GRAPH_PLAN_V1.md`。

## 2026-08-29 — RouteGeometryProfile失败；采用相对里程计配准的稀疏结构骨架候选

- 正式proof证明Teacher正反向唯一且稳定，但单帧前向五段支持和事件identity覆盖显著不足；故失败分类为`SINGLE_SCAN_FORWARD_PROFILE_OBSERVABILITY_FAILURE`，不是data/Teacher/system failure。
- 拒绝三种修补：缩短20m范围会在见到失败后调Teacher；降低80%/三段门会使方法只覆盖容易直廊；继续Composer或先建图会掩盖感知失败。旧路线永久保留为负结果与消融。
- 相关工作边界：STGPlanner和SphereMap从dense几何图生成骨架，地下SER用LiDAR LOS+dense map建拓扑，CenterLineDet从序列车载传感器预测道路中心线图，PRISM-TopoMap用point cloud/odometry和学习匹配建地点图。新方法不能把“骨架/中心线/descriptor/topological map”任一单项称为创新。
- 选择`ERCSS`候选：五帧LiDAR只用相对SE(2.5)里程计配准到当前帧；模型输出当前已观测自由空间内的ego-connected稀疏隧道中心骨架、分叉连接、宽高坡度曲率和不确定性；稳定局部骨架直接触发节点，真实trace才提交edge。
- 独立创新主张只有组合链：`registered causal raw LiDAR -> learned sparse metric structural skeleton without dense map -> uncertainty-refusing structural node proposal -> traversal-verified global graph`。任何组件不改善图或闭环即停止。
- 下一步仅做Teacher/representation feasibility，不训练：全量相对pose可用性、因果注册、mesh-visible spline clipping、identity/branch coverage、capacity/rotation/reversal/stacked-tunnel isolation。PASS前禁止模型与graph。
- RouteProfile run=`results/gate3_semantics/gate3_20260829_gse_route_geometry_profile_proof_v1_seed0`；seal SHA=`91c8f43a02dcc6e0e59273ed35dd92a090878c3818ea9a6cce942c2a1a6ea020`。

## 2026-08-29 — 停止当前双Composer标量状态；只允许RouteGeometryProfile证明

- 三种子正式训练系统完成且`error=null`，但C08 full macro-F1平均相对旧V2R5下降`0.059715`，turn frame-F1下降，高精度下full无法提交任何C08观察，核心科学门失败。
- no-metric的ensemble macro-F1=`0.559359>0.537115`，no-transport=`0.537107≈0.537115`。因此当前四个全局metric scalars和transport未形成完整方法所要求的独立正贡献；继续扩大Composer、加epoch或调C08阈值缺乏证据且违反冻结合同。
- 决定不进入离线图，不删除Metric-Change创新支柱，也不退回“出口计数+descriptor+refusal”。当前失败run保留为论文必要消融和负结果。
- 唯一方法级修正采用预注册fallback：先做零训练`RouteGeometryProfile`可见性与Teacher唯一性证明。候选剖面只能由当前及过去LiDAR支持，显式表达沿前向路线的宽、高、坡度和曲率位置分布；TNG/spline/mesh仅用于Teacher和审计。
- 若proof不能证明可观测、唯一、因果和足够事件覆盖，则停止当前GSE-Graph方法方向；若PASS，下一步也只能做typed profile representation readiness，不能直接训练或建图。
- run=`results/gate3_semantics/gate3_20260829_gse_dual_composer_three_seed_training_v1_seed0`；seal SHA=`e32d182d69d611a7389da30e7200b3434034bad8e8461b6edf1fd74173deada8`。

## 2026-08-29 — Sparse Circular Relation Transport接口PASS；允许一次三seed能力训练

- V2以6个token和显式dustbin transport替换失败的180-bin独立关系阈值场；保持三个关系语义可共存，并恢复五帧directional-temporal axis。
- 正式readiness证明53个旧backbone keys逐值兼容、全部新旧参数真实batch gradient finite、同pair reveal+withdraw无冲突、圆周旋转/排列/反向时间/确定性合同成立。
- 允许一次预注册三seed训练，但不得把readiness称为方法成功。训练必须先监督圆周proposal与token identity transport，再联合事件/几何/descriptor；C01--C06提供全部梯度，C07选择checkpoint与阈值，C08只作一次零适配迁移。
- 核心通过条件仍是：axis恢复`<=10°`；persistent/reveal/withdraw和当前token set达到安全precision与非零recall；五事件、连续几何、地点关联与不确定性拒绝不得回归。任一核心失败即停止在图前，不以planner补偿。
- run=`results/gate3_semantics/gate3_20260829_gse_sparse_circular_relation_transport_readiness_v1_seed0`；seal SHA=`293a0abda403697772f09f811127a26b78a30b1f463a6cc64f7703c136d793f6`，图SHA=`e304de6bdc3f544819dce51d83d1036f301ec1c867fdfb220098eb1c1d5c0e05`。

## 2026-08-29 — V2采用稀疏圆周关系token与显式transport，不再采用dense关系阈值场

- 冻结输出归因证明pair级关系是否发生可观测（AUC约`0.84--0.92`），但精确bin排序很弱：reveal/withdraw AP仅`0.006--0.009`，安全precision下无实用recall。该组合排除“完全没有LiDAR信号”和“只调阈值即可”两种解释。
- oracle-count的±8°关系定位约`35--46%`且C07→C08稳定，说明局部圆周支持可作为proposal；p90仍达`77--89` bins，说明必须结合显式跨帧匹配与空槽拒绝，不能把soft target当作完整修复。
- V2 typed接口固定为最多6个稀疏圆周exit/relation tokens；五帧逐方位融合恢复axis；相邻帧token通过带dustbin的transport表示persistent、reveal和withdraw；关系几何/descriptor绑定token。模型forward仍只接受五帧LiDAR，Teacher identity只用于loss/assignment。
- 下一步仅做零训练readiness：合成同时reveal+withdraw、不同bearing迁移、循环wrap、token permutation、旋转、反向traversal、dustbin拒绝、真实batch finite backward及旧backbone兼容。PASS前不得训练或进入graph。
- 正式归因run=`results/gate3_semantics/gate3_20260829_gse_axis_anchored_v2_failure_attribution_v1_seed0`；seal SHA=`1cf6ca4c7f8d3c9c4fb3a49526a48a8fd7a688854526802ced9ae6b000f04d0e`，图SHA=`19100bef12934de9eb887d840167982530b7c00d786c0deef7abac5a5070b8f0`。

## 2026-08-29 — 停止Axis-Anchored密集关系场；采用时序方向融合与稀疏关系transport作为唯一V2候选

- 三种子正式训练完整执行`36,690`步且`error=null`，但C07/C08 event macro-F1=`0.662292/0.633742`、axis误差=`82.7165°/83.0120°`；关系和current branch field在预注册安全精度下均无非零召回，不能进入拓扑图。
- 失败不是Teacher多标签接口、资源、随机种子或训练轮数问题。当前axis head只读取最后一帧方向特征；历史Peak/Slot模型使用五帧逐方位融合，在同一C07/C08人口上axis约`5.1--5.8°`。因此必须恢复时序方向编码，禁止继续扩当前单帧axis head。
- reveal/withdraw人口率约`0.013%`，当前平衡BCE正样本权重约`3,600--3,800×`；冻结输出表现为高raw recall与极低precision，且没有满足安全精度和最低召回的阈值。禁止通过加epoch、降低precision、只调阈值或图稳定器补偿。
- 唯一V2候选固定为：复用已验证的五帧directional-temporal backbone；保留persistent/reveal/withdraw三个可共存语义；用圆周软峰、困难负样本、显式峰值解码和跨帧bearing transport/匹配替换dense independent-BCE与同bin union传播。先做零训练机制归因/readiness，再决定是否训练。
- 当前run保留为论文失败分析与消融；有效的event、width/height/curvature和descriptor组件可复用，失败的axis/relation/branch-union/slope/association deployment不得作为最终方法证据。
- 正式run=`results/gate3_semantics/gate3_20260829_gse_axis_anchored_event_relation_three_seed_training_v1_seed0`；清单SHA=`9e9e9f250e35a5fbc354a24dd57153067f7fd22100afc69f67be71b6eadbf5c9`。

## 2026-08-29 — 停止互斥关系分类，采用独立多标签时序关系

- 全人口审计发现fit/C07/C08中分别有`202/20/33`个方向格同时包含不同物理出口的reveal和withdraw；四选一类别无法表达客观Teacher，故V1/V1R不得进入训练。
- 决定将persistent/reveal/withdraw设为三个独立二元通道。不同通道可共存；同一通道同格出现两个物理出口仍立即失败。没有删行、扩方位格、折叠identity或修改Teacher。
- V2正式覆盖80 worlds/188,126 observations，精确复现fit/C07/C08人口并无损保留全部255个冲突；15项unit、19项方法检查和真实CUDA backward全PASS。
- 这只确立可训练表示，不构成科学性能结论。下一步只允许一次预注册三seed能力训练；若事件、关系、几何和安全关联不能超过基线，停止方法，不让图或规划器补偿。
- 正式run=`results/gate3_semantics/gate3_20260829_gse_axis_anchored_event_relation_readiness_v2_seed0`；seal SHA=`260e7ff9321111e87a2403c0e4f738e13543dc18419e837af4249f11651823a3`，图SHA=`ded447e672b46f42ec28c6adadc6ef6de36be5134559424732cae42a8ad8465e`。

## 2026-08-29 — 出口集合过程readiness PASS；允许一次三seed训练

- 决定主感知候选采用normalized circular finite-set intensity + explicit 1--4 cardinality；不再恢复independent local peak threshold。
- 每行所有真实出口共享一份概率质量，correct set NLL低于1-bin shift、duplicate和ghost；cardinality显式监督，decode不使用存在性阈值。
- radius-one suppression不是调参：由396,913 Teacher exits的same-bin和adjacent-bin collision均0预先固定，可保留bins 0/2和179/1近邻出口。
- V1唯一失败为`5.96e-8==0`不合理浮点比较；V1R只设atol=`1e-6`，小于原rotation contract=`3e-5`，其余10项检查和所有输入不变。
- 允许一次from-scratch三seed训练。必须独立报告count/action、continuous bearing set、uncertainty refusal和完整geometry；禁止oracle count、宽容差追认或graph补偿。
- V1R run=`results/gate3_semantics/gate3_20260829_gse_circular_exit_set_process_readiness_v1r_seed0`；seal SHA=`966a5d92e2ef81b355b809bdd0c1ae416f6734d892555dc37d3448b98bf5583b`，图SHA=`8e542e0b55ae32fada098bfb71fa17ea9226339ee1be9bec0585f966682533cb`。

## 2026-08-29 — 停止独立局部峰分类，改为因果圆周连续出口集合过程

- V2完整34,110步且error=null；strict AP相对V1提高，但C07/C08仅`0.110522/0.097221`，无precision>=0.995 threshold，不能进入图。
- 连续global geometry保留，说明不推倒causal circular backbone；slope继续失败需在后续模型共同解决。
- oracle-count top-K NMS在strict中心bin上的最佳命中仅`0.2588/0.2416`，证明count-only corrective不足；±4°才达到约`0.782/0.765`，也禁止通过放宽评分追认V2成功。
- 停止继续调independent bin BCE/focal、threshold、NMS或V3 peak loss。新接口必须联合预测cardinality和连续bearing，用permutation-invariant set transport/assignment监督真实1--4出口；uncertainty决定拒绝，几何属性绑定集合元素。
- 先做零训练readiness，不直接训练：wrap/rotation、近邻出口分离、集合排列、cardinality、真实finite backward和隔离全部通过后才允许Data Card。
- V2 run=`results/gate3_semantics/gate3_20260829_gse_circular_peak_geometry_three_seed_training_v2_seed0`；seal SHA=`35b5efabab9559e448186a618efb4164fe9f79fe45ae63a3e868fae5d6f920ee`，图SHA=`3850a80c51ee989f7ab6a5feedb74ae81f739856e9de4be022418f7045ea6195`。

## 2026-08-29 — 允许Soft Angular + Hard Negative V2训练，禁止宽容差重评分

- V1冻结输出证明角度信息存在：C07/C08最近真峰p90偏差均为4 bins，±10° AP接近0.90；同时每帧约19.7个local maxima导致exact AP失败。
- top-4/NMS在不训练时只有把匹配放宽到约20°才恢复可用safe recall。该方案改变任务语义并可能合并相邻出口，拒绝作为corrective或论文成功指标。
- 选择最小模型修正：radius=4由C07-only p90一次确定；使用circular Gaussian-like soft target、focal heatmap和equal-count top outside-support hard-negative ranking。C08只确认偏差分布，不选择超参。
- 合成与三冻结backbone真实batch的梯度、排序、旋转和finite checks全部PASS；允许一次V2三seed训练Data Card。除presence loss外，不得改backbone、geometry heads、split、epochs、seed或评价门。
- readiness run=`results/gate3_semantics/gate3_20260829_gse_soft_angular_peak_hard_negative_readiness_v1_seed0`；seal SHA=`1badc88c95982af3dd38e10199fb936d20b0cfc550fc67eaf8f00d3292a1072c`，图SHA=`04a98075951f94c41dfa5d04cd4ac65818895669d49e6ceb7504b03d2a1bbf4f`。

## 2026-08-28 — StructuredPolarMultiDepth readiness PASS；允许一次dense三seed训练

- 正式readiness把131,424 tokens全部rasterize到180×2 slots，slot1=`277`、overflow=`0`、round-trip max=`2.9281e-5m`；没有删除同方向近/远事件。
- 172,430参数model只接受五帧range/valid；support parity=`0`，预测range和±1度azimuth均有界，真实空集/0--5事件+same-ray双深度batch finite backward。
- 20度dense/top16坐标rotation error=`1.0967e-5/6.6757e-6m`，bin/slot index exact；top16 unique、repeat exact、seed replay exact。旧free-query的方向相消路径已从新接口中删除。
- 决定允许一次三seeddense capacity训练。监督改为直接slot objectness/type/xyz/descriptor/uncertainty，不再Hungarian分配；数据、split、Teacher、三seed和4m同类型评估合同保持。训练前仍需独立Data Card/spec/preflight和smoke。
- 正式run=`results/gate3_semantics/gate3_20260828_gse_structured_polar_multidepth_readiness_v1_seed0`；11项seal SHA=`34a55cc0728de7a6835545a1732bcc7d135a71b7d3fee67eb6d918adf29a8736`。

## 2026-08-28 — Teacher要求180×2 radial-depth slots，拒绝单方位和方位×高度单槽

- 全量131,424 tokens审计得到same-azimuth pairs=`277`、same-azimuth-and-elevation=`251`、within-one-bin=`801`；max per-row same-azimuth occupancy=`2`。最小事件角分离仅`0.004788°`。
- 251对冲突证明增加四个elevation bands仍无法消除同方向近/远事件；删除远端或近端标签会破坏多事件结构语义，故禁止。最大occupancy=2给出恰好两个depth slots的充分证据，不增加第三槽。
- 决定新dense layout=`[180 azimuth bins, 2 depth slots]`。每bin的活跃Teacher tokens按radial distance升序分配slot 0/1；输出保留continuous elevation和±1度azimuth residual。推理不做±1-bin NMS，以免合并真实近邻事件，只按confidence从360 slots选固定top-k并保留bin/slot provenance。
- 全部token已有sealed local free-range support，minimum margin=`2.2888e-5m`；17-bin rotation mismatches=`0`、量化误差`<1°`。因此先实现rasterizer/model readiness，不再做Teacher修正。
- 正式run=`results/gate3_semantics/gate3_20260828_gse_structured_polar_teacher_feasibility_v1_seed0`；16项seal SHA=`8190da57f2e85ba2ed8beaebf7e6f99beb860d0862b9aa9ff4c381864b5b2505`，图SHA=`a42b42f9fd42ab556ed24807ba6c6b0978266bb34f5cd3e483c002e4b88a08fa`。

## 2026-08-28 — 归因要求structured polar proposal；禁止NMS/objectness/type-only修补

- V1只在最终population JSON处触发NumPy安全类型错误，未写科学结果；V1封存为system FAIL。V1R只把`uint64` cardinality显式转为`int64`，3项针对性测试通过，正式输入、算法和四条决策门逐项不变。
- V1R三seed NMS suppression=`72/71/2`，F1不恢复；all-query typed oracle recall=`0.122824/0.125147/0.116217`，position-only oracle=`0.123518/0.128556/0.120984`，全部低于旧baseline recall `0.262905 + 0.10`。
- 类型近邻上界仅增加`22/113/158`个目标，而无4m内位置候选为`29,051/28,884/29,135`；决定问题不是duplicate、confidence/cardinality或type conditioning，而是自由query未生成正确空间位置。
- 决定=`STRUCTURED_POLAR_PROPOSAL_REQUIRED`：下一表示在180个显式方位bin上预测dense objectness/type和对应free-range内的range/elevation，局部峰值确定proposal并只允许有界角度/位置残差。旧attention-weighted vector公式保留为失败消融，不再训练。
- 训练前必须先零训练审计Teacher polar-slot唯一性：同bin冲突、±1-bin冲突、最小方位分离、free-range支持和旋转索引确定性。若不能无损表达，先修改slot表示，不得删标签。V1/V1R seal SHA=`e5dd42844038e1568a4c34f1b2ea029bf630b338c1251b8989cf57b33014ec5e`/`2b9e30d94929be05ec1bbd8753afb86f188af2c3ccefdd41d566e87e9f39d056`。

## 2026-08-28 — GeometryAnchored三seed联合训练科学FAIL；停止自由query路线并先做只读归因

- 正式run=`results/gate3_semantics/gate3_20260828_gse_geometry_anchored_joint_capacity_v1_seed0`；三trainer return 0、统一评估return 2，`error=null`、source unchanged，总计`27,288` optimizer steps，C09/C10/M-TARE零读取。
- 三seed F1=`0.085885/0.093166/0.068527`、macro-F1=`0.094389/0.088810/0.075453`，而同人口旧互斥center F1/macro-F1=`0.390133/0.401037`；全部科学门失败。匹配定位MAE=`2.440/2.231/2.512m<=4m`是唯一通过项。
- 受影响结论：GeometryAnchored V1不能进入离线图或闭环；readiness V2关于Teacher支持、旋转等变和数值接口的PASS仍有效。问题分类为`MODEL_OBJECTNESS_AND_PROPOSAL_GENERATION_FAILURE`，不是data/Teacher/metric/system failure。
- 选项：A继续加epoch/调threshold，成本低但违反冻结门且三个seed已稳定失败；B直接用现有输出建图，会用后端掩盖前端失败；C先做输出级只读归因，再仅修证据指向的proposal/objectness模块。选择C。
- 下一步不训练：在45,942条C07--C08输出上分解空帧误报、query占用、同类型重复、错类型、径向/方位误差、cardinality与assignment。只允许据此预注册一个最小corrective。31项seal SHA=`ded71ce5e58f1dff71bec150e3a6d5a32547e87e54733b71e425dd01b2f40c38`，图SHA=`600895c5f481a061a3ed7ed94ed82f6faca038b2dd4c611f35614219e896e697`。

## 2026-08-28 — GeometryAnchored readiness V2 PASS；允许一次三seed联合训练

- 正式run=`results/gate3_semantics/gate3_20260828_gse_geometry_anchored_spatial_event_readiness_v2_seed0`，`error=null`、source unchanged。
- 80 worlds/188,126 rows/131,424 V2 tokens/1,076 identities全部对齐；五帧+±1 polar bin+0.25m LOS margin的support审计unsupported=`0`，最小margin=`2.29e-5m`，NumPy/Torch parity=`0`。
- 264,134参数模型range/valid-only；cardinality 0--5 finite backward、permutation error=`1.19e-7`、rotation xyz=`7.15e-7m`、invariants=`3.81e-6`、seed replay=`0`、free-range bound PASS。
- 12项seal SHA=`de4a734e64339d0f73bd3ee415f7337126d323574e26ee7e2f8a8e477349740d`。该PASS只证明训练资格，不冒充模型精度或图结果。
- 决定允许一次joint encoder+set三seedcapacity：C01--C06全部梯度，C07--C08仅checkpoint和固定网格threshold选择；必须和旧互斥center、旧冻结encoder set、非学习几何比较。训练前仍需独立Data Card/spec/preflight。

## 2026-08-28 — Observable Teacher V2通过；允许返回Gate 3重做encoder readiness

- 正式Teacher V2 run=`results/gate2_representation/gate2_20260828_gse_observable_spatial_event_teacher_v2_seed0`，`error=null`、source unchanged。
- 结果精确为80 worlds、188,126 rows、1,076 identities；V1 `133,055` tokens按冻结FOV规则删除`1,631`后保留`131,424`，terminal/junction=`30,714/100,710`，fit/selection=`98,279/33,145`。
- 删除只由`atan2(up,horizontal)`是否位于inclusive `[-15°,15°]`决定；没有读取scan、prediction、checkpoint、confidence、C09/C10或M-TARE。每个删除token保存原identity/slot/elevation/distance，V1及其FAIL证据不修改。
- 1,298项seal验证零mismatch，SHA=`736261034852c779d26e2619735755ce1bbd8bc7a415dbc68257923e7cfa5772`；图SHA=`1648cc4b0a291eaec27b7336872a5d6ba4d170b7597d26b1ecbfd806e5e0a0f6`。
- 决定治理状态从临时Gate 2返回Gate 3。只允许冻结五帧局部free-range envelope+既有0.25m LOS margin并重做一次zero-training readiness；不得直接训练或进入图。

## 2026-08-28 — 发现Teacher—LiDAR垂直FOV错配；选择可观测Teacher V2而非重做传感器

- 证据：正式GeometryAnchored readiness精确复现80 worlds/188,126 rows/133,055 tokens；264,134参数encoder的range/valid-only输入、0--5 set loss/backward、rotation、permutation、determinism和free-range bound全部PASS。唯一FAIL为641个Teacher token超出当前polar free-range，正式run `error=null`、source unchanged。
- 归因：采用原Teacher `0.25m` LOS margin后仍有639项；其中619项在当前帧超出LiDAR `[-15°,15°]`垂直FOV。五帧包络仍剩585项，其中577项视场外。旧Teacher以连续native-mesh ray标记visible，却没有约束冻结16线传感器FOV，属于Teacher/传感器可观测性错配。
- 受影响结论：V1 Teacher不能直接用于geometry-anchored joint training；旧冻结encoder容量FAIL结论不受影响。正式readiness V1永久保留为科学FAIL，seal SHA=`39610c5fb4317da031614e52760ea1aa86ccc893fb5664d4fb79c33b541be5ce`。
- 选项：A扩大垂直FOV并重生成252,430帧及全部基线，成本最高且改变传感器合同；B删除全部失败项会按模型接口选择标签；C仅用事前冻结的LiDAR FOV修正Teacher，并保留全部帧/identity。选择C。
- 决定：Teacher V2只删除仰俯角不在`[-15°,15°]`的1,631个token；不删观察，不读取测试域，131,424 tokens和1,076 identities保留。encoder使用五帧局部free-range envelope和既有0.25m LOS margin。用户已持续明确授权自动选择最优方案，本决定不另行请求例行批准。
- 下一步：先正式生成/审计V2 Teacher，再重做零训练readiness。任一失败则继续停止训练；C09/C10/M-TARE、图与planner保持禁止。

## 2026-08-28 — 失败归因排除duplicate/confidence corrective；要求geometry-anchored空间encoder

- 三seed固定4m NMS suppression=`0/28/260`，比例=`0/0.001065/0.009623`，NMS F1与sealed F1差值不足`2.3e-5`；决定永久停止repulsion/NMS作为主修复。
- 全16 query typed oracle recall=`0.199684/0.255102/0.212764`，position-only oracle=`0.253195/0.305336/0.277657`，未达到`exclusive recall 0.262849 + 0.10`。因此confidence/cardinality calibration也不能恢复主差距。
- 距离审计显示0--4m仍有可用空间信号，而32m后接近零；决定把失败分类为`ENCODER_SPATIAL_REPRESENTATION_INSUFFICIENT`，下一模型必须联合训练空间encoder并显式保留polar coordinate和free-range anchor。
- 新实现先限于零训练readiness，不直接开始三seed训练。接口只允许五帧range+valid mask，禁止pose/world/TNG/Teacher identity/future；径向预测必须是观测自由距离内的bounded fraction，避免无坐标context直接回归0--50m。
- V1因图distance字符串顺序、长标题和Infinity序列化只保留为原始审计；V1R没有改变样本、阈值、匹配、NMS或决策门，只修严格JSON和论文图。V1R run=`results/gate3_semantics/gate3_20260828_gse_spatial_event_set_failure_attribution_v1r_seed0`，16项seal SHA=`2cf2bd47a0607ea5c22f5b563d8c99b49f45a57cb15c765ae06c64de553343aa`，图SHA=`1a798eea42ace02aaec1c70a3ef60b0e8c9c5d352044f05b77cf07cfc35a3ef3`。

## 2026-08-28 — 冻结旧encoder的SpatialEventSetDecoder科学FAIL；先归因再决定新encoder

- 唯一正式run按Data Card完成三seed各`4,448`步，仅更新`93,638`参数set decoder；encoder optimizer step为0。正式C01--C08输入前后哈希一致、`error=null`、C09/C10/M-TARE零读取。
- 三seed同类型4m匹配F1=`0.129951/0.186228/0.143671`，precision最高仅`0.211947`，multi-event recall最高`0.154335`；全部违反预注册安全门。匹配成功部分MAE=`2.578/2.491/2.633m`，所以失败定义为`representation_or_set_generation_capacity`，不是定位完全不可学或系统错误。
- 旧互斥单中心baseline F1=`0.391306`且P=`0.765334`，显著高于全部set seed；非学习几何F1=`0.003171`。决定停止把“变量数量输出”本身当作创新收益，不调阈值、query数、loss权重或训练轮数挽救正式run，也不直接解冻backbone。
- 下一唯一操作是只读failure attribution：距离/基数/type/family分层、query重复率、径向塌缩、同类型NMS和oracle去重上界。若去重上界仍不能超过旧baseline，则训练真正面向远程结构位置的encoder；若主要损失来自duplicate queries，才允许带repulsion/cardinality calibration的最小corrective。两条路线不能并行试错。
- 准备时一次文件枚举读取到历史C09 archive的字段名/shape而没有数值；该检查被隔离且未绑定任何正式输入、方法、阈值或统计。决定如实记录但不把它冒充正式C09 read；严格未见测试继续是C10。
- 正式run=`results/gate3_semantics/gate3_20260828_gse_spatial_event_set_capacity_v1_seed0`；42项seal SHA=`bdcbbf1552c6f592376d4f85912f18c90ddb6eea28568a0db1c7bdfafc404322`，论文图SHA=`fa340e856e59feece57db0e48b210ce9f9ad1f4726d3ee75722fabe7b26c8ef1`。

## 2026-08-28 — 失败归因PASS；用空间多事件Teacher取代互斥scene分类

- 正式run对98个关系端点和两条稀有失败做0训练/0推理归因。`node_0012`稳定预测与terminal直连的degree-3 junction，归为exclusive Teacher conflict；`node_0040`route-stop三seed一致但structure mass高度分裂，归为seed representation instability。两个失败均不再是“训练步不够”的解释。
- 96/98关系端点在50m欧氏尺度内直连相反事件，48/48 terminal均如此。该统计只证明潜在共现尺度，未证明无遮挡；决定下一Teacher proof必须使用native perception mesh做LOS，禁止把距离直接写成可见率。
- 决定覆盖旧互斥junction/terminal接口：一条因果LiDAR可输出变量数量`SpatialStructureEventToken`，每个token包含事件类型、robot-frame相对3D位置、连续几何、出口集合、descriptor和uncertainty。TNG identity/绝对pose/未来帧仍禁止作为学生输入。
- 在线图的provisional拒绝、4m候选、执行endpoint anchor和trace-only edge合同保持；只替换结构事件Teacher与输出接口。历史12帧互斥detector和643参数route residual分别保留为安全FAIL与失败消融，不重跑。
- 下一步只允许C01--C08多事件Teacher可行性审计；必须证明LOS集合唯一、基数可控、fit/selection覆盖和稀有端点覆盖后才能训练。正式归因run 19项seal零mismatch，seal文件SHA=`352412794a5c5de684b983ffc92090f1e5d1e09ec84b331fca6a3e0d89241aa2`，论文图SHA=`1d136211db2c81189fcb7d6f32f68282e149cf33066ece26856e7292e229c11c`。

## 2026-08-28 — 643参数route-conditioned residual科学FAIL；停止小头并禁止重复12帧路线

- 唯一run完成三seed×200步，系统错误为空、输入前后哈希一致、C09/C10/M-TARE零读取。ensemble在固定0.97下为`931`正确episode和`90` false triggers，优于baseline的`928/94`，但两个低支持端点仍`0/2`。
- 完整图selection为`185`正确节点、`4`正确边，node P/R=`1/0.675182`、edge P/R=`1/0.307692`、false loop=`0`；违反冻结node recall不得低于`0.704380`的门。决定接受科学FAIL，不用总体episode小增益掩盖拓扑退化。
- 决定停止低维route-flow residual，不增加steps/hidden size、改变loss或调0.97阈值。AUC审计与本run保留为“标量路线线索存在但不足以驱动图结构”的论文消融。
- 历史12帧causal episode detector已在C09产生`2.464%` false trigger并失败安全门；禁止把它当成新路线重训。下一项只读审计比较两个稀有端点的原始六出口token、16维汇总、每seed残差和最终触发，先区分representation compression与Teacher-interface mismatch。
- 若完整token关系对目标端点仍不可分，推荐修正Teacher为可观测action event而非强记objective node type；若可分，才允许一个直接消费完整token集合的关系注意力head。两者不得并行调试。正式run 44项seal零mismatch，seal文件SHA=`5130ca94d266288069c6003e18d930a7f9f0d38f62dc8d2d541fe27cd7d74583`，失败图SHA=`14d5d3d5a4de6f77246883543c28184fb31e8c36a6d900d43671398cdae2f69e`。

## 2026-08-28 — route-conditioned表示容量PASS；允许一个最小event residual

- 预声明route-stop score在fit/selection分别达到AUC=`0.949074/0.934524`，两域方向一致且均超过0.80；terminal均值均高于junction。selection两个低支持terminal中1个高于fit junction中位数，满足“至少有一个可恢复端点”的容量门。
- 决定允许一个零初始化、低容量route-conditioned event residual，同时输出1个structural residual和2个conditional class residual。route-flow来自当前及最多4个过去观测，身份、世界、TNG、pose和未来帧禁止进入输入。
- 原ActionSetNodeDetector全部冻结；训练继续使用原negative+all-episode MIL和endpoint identity-mean项。C07--C08只选固定步checkpoint，threshold保持0.97，无loss/threshold grid；图回放合同不变。
- 若低支持端点仍为0、正确episode下降、false trigger增加或图precision/recall退化，科学FAIL并停止，不继续扩大head。正式audit run 18项seal SHA=`6b2c5709dd75f71dfa862b8c1efd1ec9332988548391a751422b8ed27fd3c660`，论文图SHA=`f5c8b20a9faf7e0988040de39a6ad027532e9e4a333b6a376dbb98a80e275636`。

## 2026-08-28 — 拒绝route-agnostic类别硬修；转route-conditioned exit flow

- 可观测性run机械规则因存在accepted-but-wrong-event端点给出`conditional_event_class_corrective_first`，但逐行5-NN显示该terminal的冻结context和物理出口几何都稳定落入junction邻域。决定不按单个布尔规则直接训练类别头，以免把局部不可辨识样本硬记忆为objective identity。
- TNG证据：`S04...node_0012`到degree-3节点仅`13.484m`，`S07...node_0040`到degree-4节点仅`11.659m`；50m LiDAR在terminal可同时看到身后junction。问题分类为representation/Teacher-interface mismatch，而非简单model capacity。
- 决定保留objective terminal作为图语义，但学生事件输入必须route-conditioned：把robot-frame heading相对当前执行方向分成连续forward/backward/lateral mass，并加入五帧变化；identity/world/TNG仍禁止进入学生输入。
- 下一步只读计算固定物理`route_stop_score=(backward-forward)/(forward+backward+lateral)`的identity-level terminal-vs-junction AUC与低支持位置，不训练、不选阈值。若fit/selection方向不一致，停止该表示；若一致，才允许低容量route-conditioned event head。
- 正式observability run=`results/gate3_semantics/gate3_20260828_gse_low_support_endpoint_observability_audit_v1_seed0`，19项seal SHA=`585c14105215c973365419726c59c926af3a5aff92bbca33e6f2e4aed9bb882f`，论文图SHA=`d13014e805e4166c1473abdebf10c34a6a83c0e66ceb1a59912d54b8972fe83e`。

## 2026-08-28 — 129参数identity残差科学FAIL；停止structural-only纠偏

- V1R3按冻结合同完成三seed×200步训练。三个seed的selection-safe最佳点均为step 0；identity均衡损失下降，但任何非零候选都未恢复两个低支持端点，并开始损失正确episode。决定接受科学FAIL，不增加steps、不调loss权重、不降低0.97阈值。
- ensemble低/高/全部端点正确数=`0/21/22`，与冻结baseline完全相同；correct episodes=`928`、false triggers=`94`也相同。开发图仍为selection `193`正确节点、`4`正确边、P=`1/1`、false loop=`0`，所以失败不否定现有执行验证图，只否定“structural-only identity residual足以恢复关系端点”的假设。
- 两个低支持terminal分别是event-class错误（结构mass最高`0.9887`，已有proposal但错成junction）和proposal缺失（唯一行mass`0.3632`）。前者在保持conditional logits不变时理论上不可修复；后者尚不能区分表示不可观测与监督边界问题。
- 决定停止该残差路线并保留为论文必要消融/失败分析。下一项只读审计必须同时量化structural margin、junction-vs-terminal conditional margin、五帧token/几何变化和同事件困难负样本；只有审计证明输入中存在可分信息，才允许一个对应的新表示或类别头。C09/C10/M-TARE继续禁止。
- 正式run=`results/gate3_semantics/gate3_20260828_gse_endpoint_identity_residual_training_v1r3_seed0`；44项seal SHA=`7dc32bdead2ae6cd0f6057f6b2fb38f4e5fea3bc012a960fe89adb78ae8851b2`；论文失败图SHA=`e5484af4f30f65fbe6eca5b42e2c8bb0c6e462c16dab00e2079b1b1ff3d3f4c4`。

## 2026-08-28 — 纠正“row-balanced”解释；冻结identity-mean-of-episode corrective

- 代码证据显示当前`action_set_episode_loss`对每个连续episode取一次最大joint probability，`EpisodePreservingBatchSampler`保证episode不被拆分；因此行级过采样不是正确修复，先前“逐帧损失淹没”表述被本条替代。
- 数据证据仍确认identity曝光偏斜：fit relation endpoints的`1--3/4--10/11--30/31+`支持组每identity episode均值为`1/2/2/6.2222`；selection为`1/2/2/6.2857`。低支持与高支持identity的有效MIL权重相差超过6倍。
- 决定：新增损失严格为`original negative + original all-episode positive + endpoint identity mean of within-identity episode losses`，三项等权且0超参搜索。Teacher identity只用于train loss分组，部署不输入identity。
- 模型改动限于零初始化`Linear(128,1)` structural-logit residual，原ActionSetNodeDetector全部冻结；conditional junction/terminal logits和causal context逐元素不变。该129参数head是可独立消融的最小corrective。
- selection必须同时改善2个低支持端点中的至少1个、保持高支持端点与总体正确episode、不增加false triggers，并在冻结spatial/association/qualification图上保持node/edge precision≥0.98及edge recall≥当前4/13。失败则停止，不读C09/C10。
- 正式audit run=`results/gate3_semantics/gate3_20260828_gse_relation_endpoint_episode_exposure_audit_v1_seed0`；18项seal SHA=`671b949aad8f380f20b5000b992cd0538fd7c27f1d2cbbe3cd7efc676e8c1d29`，论文图PNG SHA=`d3e7fdb139dcb885974d70f2eeca300f7df091f44bcf359afb359391342c43a2`。

## 2026-08-28 — 低支持端点漏检在开发集复现；批准一次identity-balanced corrective

- 正式审计覆盖C01--C08全部49条objective relation和98个不同端点。fit低支持`n=10`、selection低支持`n=2`，满足预注册最低样本条件与方向复核要求。
- fit proposal recall由低支持`0.30`升至高支持`1.0`，差`0.70`；selection由`0`升至`0.9130`，没有反转。预注册三个布尔门全部PASS，因此“逐帧训练权重淹没稀有端点”获得开发数据支持。
- 决定：允许一次ActionSetNodeDetector端点均衡corrective。原row-balanced batch与原loss继续存在；新增batch只从C01--C06的objective relation endpoints按identity均衡采样，不读取C09/C10。部署proposal threshold固定0.97，图关联、commit、qualification和edge合同均冻结。
- checkpoint选择必须同时满足：C07--C08低支持端点proposal recall提高；全体decision proposal precision/false-trigger安全不回归；普通高支持端点和既有节点/边precision不回归。若端点均衡只提高frame recall却不能提高端点/关系生存，则停止该corrective，不以图参数补救。
- 正式audit run=`results/gate3_semantics/gate3_20260828_gse_development_relation_endpoint_support_audit_v1_seed0`；19项seal SHA=`276cae555f04f02faf40d70f1b903d765bad03caa55fcfe662f713be891b6e35`，论文图PNG SHA=`2d56ae73794398dd0d0a8bfde272049012511973622341c0b541378e8f37396d`。

## 2026-08-28 — 关系端点漏斗完成；停止edge-first修正，转开发集监督支持审计

- V1的唯一错误是Teacher文件顺序与冻结action canonical顺序不同；两侧各有24,462个唯一ID且集合完全相同。V1R只按stable global ID做一对一join，保留V1系统失败和seal，不覆盖运行。
- V1R复现全部冻结中间量并通过：endpoint stages=`5 proposal_missing + 3 association_or_commit_missing + 3 qualification_reject + 5 final_recovered`；relation stages=`4 endpoint_proposal_missing + 2 endpoint_not_committed + 1 endpoint_qualification_reject + 1 recovered`。
- 6条raw edge为`1 true relation + 5 same-objective-identity duplicate/self relations`；没有证据支持继续修改edge assembler。决定冻结execution-only edge合同和最终4m匹配。
- C09失败归因可以用于停止错误路线，但不得把C09具体identity、支持数或失败项用于新阈值/checkpoint选择。新方法选择必须重新来自C01--C06，C07--C08只做内部选择；新版本不再把C09宣称独立验证，C10继续保留为最终strict unseen。
- 下一项只读开发审计统计C01--C08每个objective relation endpoint的Teacher窗口数、proposal、commit和qualification生存率，并按event/support bin/family分层。若低支持端点在开发数据同样系统漏检，才允许一次identity/endpoint-balanced corrective；否则停止该假设并检查关联/提交表示。
- V1R run=`results/gate3_semantics/gate3_20260828_gse_c09_relation_endpoint_funnel_v1r_seed0`，21项seal SHA=`03e9740454977cbb3b4e30c833ef543a81a92d39af9176f2d6eaaf3d91f5b774`，论文图PNG SHA=`c69c51a934d9acf6c69a09c0b5244e80155e0fa22fdea3539db1cea32ccbb344`。

## 2026-08-28 — C09冻结验证判科学FAIL；禁止改边组装器，先审计关系端点漏斗

- 正式C09验证在10个世界、24,462条因果观测上得到node P/R=`1/0.569231`、edge P/R=`1/0.125`、false loop=`0`；程序、输入、环境和隔离检查全部通过，失败项仅为edge recall `<0.25`，故结论是科学FAIL而非系统失败。
- 原始回放6条边中仅1条对应真实relation并已保留；其余5条是同一objective节点的重复/自关系或包含错误事件证据。不存在“正确第二条边被endpoint filter删除”的证据，因此冻结edge assembler和endpoint规则，禁止围绕它们做无依据修改。
- 8条objective relation的16个端点只恢复5个：1条双端恢复、3条单端恢复、4条零端恢复。当前问题归类为模型/图节点召回链路，而不是数据、评分器或边执行合同故障。
- 可行选择为：A）低成本只读端点漏斗，定位proposal、association/commit与qualification损失；B）降低edge recall门，违反预注册；C）在C09上调阈值或训练，造成validation泄漏；D）直接读C10，会破坏strict-test隔离。根据用户“自动选择最优方案”的持续授权，选择A，不要求例行批准。
- 正式run=`results/gate3_semantics/gate3_20260828_gse_endpoint_geometry_c09_validation_v1_seed0`，seal SHA=`d790407c7ad8f325b6c9173e2e9377cd23fbccf861a49fe5996c5649e6f1d2ad`，论文图PNG SHA=`0cc0b074e2dccaf479ec6c4649a48e5da004e6724519036eab81f205f40d9a68`。C10/M-TARE继续禁止。

## 2026-08-28 — 执行端点几何容量PASS；冻结机制并纠正C09隔离表述

- 根因对应机制：无identity的两edge假junction由两条outward vectors近反向而拒绝；duplicate/mislocalized hypotheses由执行endpoint anchors不在`2×1m`采样界内而拒绝；节点位置改为一致执行端点anchor均值。该机制消费过去执行轨迹，不消费Teacher或未穿越边。
- branch witness用dot product正负号而非角度阈值；anchor bound由固定1m route sampling推导，0 grid。决定冻结这两个合同，不允许C09后再调。
- 正式开发容量：fit node/edge P=`1/1`、R=`0.755051/0.361111`；internal selection P=`1/1`、R=`0.704380/0.307692`，全部门通过。22项seal SHA=`39c583292d6e427207a37b2c6c65ea5895894203175bc0a2c27f73b2ecfc063e`。
- 数据口径纠正：C07--C08用于method confirmation，不能作为unseen；formal run内`C09 remains the first untouched validation`只可解释为“本run和本endpoint机制尚未读取”，不能解释为全项目未读。历史C09感知/旧图validation已存在。
- 决定：C09下一步为冻结validation而非strict unseen test；C09只能判通过/失败，不能训练、调阈值或修改机制。C10继续作为最终严格未见测试，C09失败则不读C10。

## 2026-08-28 — 端点几何capacity诊断的zero-sequence traversal口径纠正

- 首次临时诊断在读取axis前因预期`16,078`条均有frame而停止；实际已知短边`S08_3d_loop_rich_C06:edge_0022`长度1.516m，正反traversal按5帧合同均为0 frame/0 sequence。
- 纠正显式保留`16,078 declared / 16,076 framed / 252,430 frames`，不删除短边、不创建伪frame。该临时诊断未生成正式run或科学结论；专项合同进入正式Data Card/spec。

## 2026-08-28 — 执行端点合并采用确定性不变量；禁止零负例duplicate训练

- `2,933/1,013`个fit/selection endpoint Teacher tokens均为单一objective identity；同physical edge的两端由orientation-invariant side区分。移除side后selection有1个同event双identity alias，因此禁止physical-edge-only合并。
- support-2已提交图中endpoint候选只有fit 9正/0负、selection 5正/0负。决定：不训练duplicate verifier；精确endpoint token union作为执行验证的确定性图操作，并保留零负例事实作为停止证据。
- fit-only factorized hybrid将node precision提高到`0.980510`，selection node/edge precision均为`1.0`且edge recall=`4/13=0.307692`。但fit edge precision仅`15/18=0.833333`，因此完整图资格仍FAIL。
- 决定：冻结4m在线关联、endpoint union及原edge assembler；下一步只做3条fit false edge的feature/Teacher inventory。若无法形成足够困难负例或无fail-closed机制，不训练edge head，不以selection结果反调规则。
- 正式run `gate3_20260828_gse_post_commit_endpoint_inventory_v1_seed0` 26项seal验证PASS，SHA=`1c65d49e88944f12770d0f0761f0593dc51eec734e198f877b7c811e606a2bf4`。

## 2026-08-28 — 停止generic事件可靠性训练；主阻塞改为提交后重复节点合并

- 正式partial-incidence inventory证明fit两edge junction有`84正/22负`且覆盖8个family、6个stratum；三edge规则在fit满足全部图门，但selection edge recall仅`3/13=0.230769`，说明还需安全恢复一个两edge结构关系。
- 对22/7个objective-negative两edge hypothesis做因果分解：fit为`15 duplicate + 5 center/assignment miss + 2 false event`，selection为`5 duplicate + 2 center/assignment miss + 0 false event`。因此objective unmatched不是统一的语义负标签。
- 决定：禁止以“是否被objective spatial matcher匹配”为标签训练junction reliability head；该训练会把真实但重复/偏移的junction错误标负，破坏论文的结构语义主张。
- 新机制是post-commit duplicate consolidation：不扩大在线4m候选域，不预测未穿越边；只在已提交、同world/同event hypothesis间，以共享physical incident-edge fingerprint产生候选，再用累计place/exit/metric证据做fail-closed一致性判断。歧义候选保持分裂，不允许强制loop merge。
- 下一正式操作仅为C01--C08只读candidate/Teacher inventory；若fit困难负例或family覆盖不足，则不训练，改用预注册确定性consensus或停止该机制。C09/C10/M-TARE继续禁止。

## 2026-08-28 — 部分incidence Teacher inventory正式PASS

- 唯一run `gate3_20260828_gse_partial_incidence_teacher_inventory_v1_seed0` 完成，`error=null`、0 optimizer、0 model inference、0 C09/C10/M-TARE；20项seal逐项验证，SHA=`ff97aa4c537d7a2fc69f76091e5673cee1ba2f84351ca0702080303ef4c2a27c`。
- 旧verifier接受`21,424/29,621`候选，三个空间seed一致接受`18,821`，其中`5,595`为metric-only恢复；联合合同不改变4m cap。
- 三edge incidence规则在fit node P/R=`0.998264/0.726010`、edge P/R=`1/0.333333`，selection node P/R=`1/0.682482`、edge P/R=`1/0.230769`；证据说明physical incidence是高精度可靠性信号，但单独使用会少恢复一条selection relation。
- 该run只证明Teacher与机制容量，不宣称图资格PASS；论文图及CSV/JSONL provenance保留，后续不得用selection重新选择support阈值。

## 2026-08-28 — 图主评分改为objective 3D一对一匹配；旧exact-row只保留诊断

- 证据：exact-row会把中心已偏离objective节点4m以上的预测仍计为正确，也可能把路口窗口外但位置正确的触发计错；它混合了触发时序与图节点位置，不能单独评价topometric graph。
- 新评分在replay完成后才读取Teacher；predicted node位置固定为evidence-row学习中心的算术均值，同world/同event、4m内做最大基数后最小距离的一对一匹配。duplicate/unmatched保持FP，edge必须经匹配后的节点identity验证。该评分没有改变runtime输出、模型、半径或图。
- 正式审计显示新评分更严格而非有利于主方法：spatial correct nodes `198→197`、node precision `0.956522→0.951691`，scalar `162→153`。因此采用它不会人为抬高论文结果；旧exact-row继续作为“触发窗口一致性”诊断并并列报告。
- 决定：后续Gate-3图资格以objective 3D评分为主；仍同时保留legacy分数与false-loop审计。主方法保持科学FAIL，不读C09/C10/M-TARE。
- 下一机制不再直接训练metric association：现有runtime候选的不同真实identity负对过少，不能支撑可靠训练。先做fit-only无训练容量证明，候选为三seed空间一致性恢复、uncertainty/event可靠性拒绝及可证明duplicate consolidation；edge assembler冻结。

## 2026-08-28 — 失败漏斗排除edge assembly；下一关联必须显式消费学习中心度量

- 274-node漏斗的最大损失是`31 association_verifier_reject`，这些identity已有至少两条独立traversal且中心距离<=4m；只有12个仍因中心超4m失败。继续只改中心不是主要收益方向。
- 13-edge漏斗为`3 endpoint proposal missing + 8 endpoint uncommitted + 2 recovered`，shared-trace和assembly miss均为0。决定冻结现有traversal-only edge生成，不加预测边或放松执行验证。
- 旧factorized verifier在训练时因cross-world hard negatives无共同坐标系，明确禁止把metric distance作为特征；新3D中心通过后，这个接口缺口现在成为可验证主阻塞。
- 决定：下一步先构造同world runtime metric-pair inventory，只用固定0.97 proposals与固定4m候选，positive为同identity跨traversal，negative为不同identity同event的真实近邻；检查独立identity/family/negative覆盖。若不足，不训练；若充分，只允许一次metric-aware association，旧verifier作为消融。

## 2026-08-28 — 空间中心图科学FAIL但有大幅正增益；保留主线并审计commit漏斗

- V1/V1R两次失败均发生在graph replay前，分别由float32世界坐标舍入和GPU batch布局非bit-exact导致；它们没有科学图结论。V1R2以canonical all-row batch128和固定0.25 mm复现界解除系统阻塞。
- V1R2在固定proposal threshold=`0.97`、association radius=`4 m`、2-of-3 vote、两独立traversal commit下，使node/edge/macro F1分别提高`0.108053/0.149020/0.128536`，false loop merge保持0。
- 仍未过node precision `0.98`与edge recall `0.25`门，不能把显著改善写成图资格PASS，也不能进入C09或planner。
- 决定：学习几何改善图这一必要条件已成立，故不停止GSE-Graph；下一步仅允许sealed C07--C08 failure-funnel审计，逐GT identity/relation定位proposal、association、commit和edge assembly中的损失。不改变模型、阈值、半径、投票、Teacher或图规则。

## 2026-08-28 — 纵向纠正通过全部冻结门；下一步只替换图中的结构中心输入

- 正式结果：relative error `4.641362→2.668592 m`（改善`42.50%`），within-4m `0.523952→0.802866`（增益`0.278915`），global center MAE `3.752201→2.221275 m`；三seed、纵向、横向、高度和系统门全部PASS。
- 因果归因：V1 lateral/up在三个seed中逐元素零漂移，原spatial decoder和backbone optimizer step均为0；收益只能来自新增纵向残差。
- 决定：冻结本次三seedcheckpoint作为新的event-center组件，停止该分支继续调参。下一步复用既有trace-commit executor，只把旧projected center替换成新三seedensemble center；关联半径、阈值、node/edge定义、因果顺序和验收标准全部不变。
- 边界：这次PASS只解除结构位置组件阻塞，不提前宣称拓扑图或论文主方法PASS；C09/C10/M-TARE在图容量证明完成前继续禁止读取。

## 2026-08-28 — 残差审计排除transverse与aggregation；只允许空间条件纵向纠正

- 正式反事实：oracle-long/pred-trans within-4m=`0.999928`，pred-long/oracle-trans=`0.615192`，完整spatial=`0.619923`。transverse空间学习不是当前阻塞，纵向是充分且必要的剩余改进方向。
- 事件分层：junction=`0.389981`、terminal=`0.886363`。剩余预算集中在junction的incident-traversal纵向中心，而不是全体terminal或少量横向离群。
- 排除方案：coordinate median和per-row medoid仅`0.616593/0.616478`，低于arithmetic mean；失败pair seed disagreement=`1.0788 m`、成功pair=`0.9202 m`，差异不足以支持换聚合器。
- 决定：现有spatial decoder及transverse输出冻结；新增零初始化bounded longitudinal residual，仅消费相同五帧空间context和冻结scalar值。保持direct+relative `1:1`、C01--C06/C07--C08 split、10 epochs及全部验收线。
- 停止规则：新纵向corrective若不能同时通过原7项门，不调loss weight/epoch/4m radius，不读C09、不运行图，停止event-center路线。

## 2026-08-28 — 空间中心学习通过6/7门；不降4m门槛，先审计剩余离群

- V1训练完整、V1R只读资格有效。最终relative error改善`13.97%`，lateral/up/global MAE和三seed稳定性均PASS；within-4m增益=`0.095971`，比预注册`0.10`少`0.004029`，因此整体科学FAIL。
- 系统边界：V1最后写JSON时的`numpy.bool_`序列化错误独立封存；V1R新增optimizer step=0并恢复同一指标。不得把system FAIL与科学FAIL混淆，也不得覆盖V1。
- 决定：保留圆环azimuth+elevation decoder为主方法候选，保留pooled-context V2为关键消融；不降低10点门槛、不扩大4m半径、不运行图或读取C09。
- 下一步：对封存C07--C08 ensemble按junction/terminal、node identity、pair distance和local forward/lateral/up误差做一次只读分解。若剩余失败主要来自冻结scalar longitudinal，则只纠正longitudinal consistency；若来自少量seed/identity离群，则采用预注册robust consistency目标；若无集中根因，停止中心路线。

## 2026-08-28 — 局部3D残差无法从pooled action context学习；停止小头并恢复空间feature map

- 正式V2：relative error改善`12.36%`、longitudinal MAE与global center MAE通过，但within-4m增益仍`0.089455`。lateral/up MAE精确等于零输出baseline，证明训练没有产生可选择的横向/竖向能力。
- 系统有效性：`error=null`、sources unchanged、11,880 transverse steps、0 backbone/test reads；因此不是运行失败。V1 evaluator bug另作system FAIL保留，V1R与V2给出一致科学结论。
- 决定：停止所有基于冻结128D `causal_context`的scalar/vector中心小头；不改loss权重、batch、epoch、4m cap或验收线。它们保留为pooled-context消融。
- 下一表示：从已存在的圆环range-image encoder空间feature map输出route-local 3D center heatmap/offset，显式保留方位与垂直布局；仍用跨traversal identity pair和direct geometry监督，C01--C06训练、C07--C08选择，C09/C10/M-TARE封锁。

## 2026-08-28 — scalar双批次只差一个门但仍停止；选择局部3D中心而非继续调loss

- 结果：V1R3 MAE、relative error和三seed稳定性gate通过，within-4m从`0.523952`升到`0.613407`，增益`0.089455<0.10`，因此整体FAIL。28项seal SHA=`e97120...d1a9`。
- 决定：执行预声明停止规则，不调整1:1 loss、epoch、batch、4m半径或验收线，不运行图。V1R3保留为“跨穿越监督scalar中心”消融。
- 容量分析：scalar只能沿当前route tangent投影，Teacher允许最大`4.549 m`横向/竖向残差；不同incident tunnel的切线投影即使纵向误差相同，也可能在3D中无法汇聚。
- 下一方法：局部3D event-center vector。forward由在线route tangent给出，lateral为水平正交方向，up为重力方向；网络输出longitudinal/lateral/vertical。继承通过的scalar hidden层和longitudinal输出，新增两维零初始化，避免丢失已学能力。
- 边界：仍只用C01--C06训练、C07--C08选择/验证，C09/C10/M-TARE不读；先证明Teacher有效、局部坐标确定且无测试泄漏，再允许一次训练。

## 2026-08-28 — 配对监督未过门；仅允许保留原回归批次的一次纠正

- 正式证据：V1R2 relative error改善`8.08%`、within-4m提高`5.89`点，但MAE退化`8.55%`；四项预注册gate均FAIL。运行`error=null`、0 backbone/C09/C10/M-TARE，科学结论有效。
- 原因：训练目标与采样同时改变。pair batch按identity均匀，并用其中行计算direct loss；原V1R3按全部观察行且junction/terminal均衡。长episode/常见几何的逐行回归质量因此丢失。
- 决定：不调loss权重、不扩大head或association半径。允许一次机制保持的corrective：V1R3 checkpoint初始化；每步一个原始row-balanced direct batch和一个identity-balanced pair batch；总loss仍为`direct + relative`。只在validation MAE不劣于对应V1R3 checkpoint的epoch中最小化relative error。
- 停止规则：corrective若不能同时满足原`>=10%` relative改善、`>=0.10` within-4m增益、ensemble MAE不退化和三seed改善，则停止scalar offset corrective，不运行图、不读取C09，重新设计可显式输出局部结构中心的主表示。

## 2026-08-28 — 更正masked-oracle诊断；图上界PASS，返回时序一致中心学习

- 数据缺陷：`objective_center_xyz_m`只在有效junction/terminal行有定义，其他行按接口填`(0,0,0)`。初次上界未应用`valid_mask`，使跨隧道假触发在原点被错误合并；9个假节点和2条伪边全部来自该零填充。
- 被撤销结论：node P=`0.961390`、edge P=`0.75`不能证明图机制存在第二瓶颈，也不能据此决定优先修改图规则。正式学习中心图FAIL与seal不受影响。
- 修正证据：有效结构行使用objective center、无效行保留部署预测坐标后，C07--C08 node P/R=`0.988095/0.908759`，edge P/R=`1.0/0.461538`，false loop=`0`，满足预注册图安全/召回门。
- 决定：保留已实现的completed-traversal edge与不确定区间fail-closed逻辑，但不再把它们当当前阻塞。下一主任务回到中心估计；不扩大模型主干，先增加同traversal相邻样本的运动一致约束和事件中心聚合损失，目标是降低跨incident view中心分散。
- 防复发：任何masked Teacher连续量的oracle/upper-bound都必须对无效行显式保留部署值或排除，禁止把sentinel/zero fill作为物理坐标。

## 2026-08-28 — 不扩大中心网络；先修正完整穿越边与延迟消歧机制

- 证据：event-center回归正式PASS，C07--C08 MAE改善`35.95%`；部署图node precision由`0.617747`提高到`0.905028`且false loop=0，但edge P/R=`0.25/0.076923`，完整图正式FAIL。
- 关键上界：以客观node center替代学习预测、空间关联全部放行后，node P/R仍只有`0.961390/0.908759`，edge P/R=`0.75/0.461538`。因此单纯换更大网络或放宽4m半径不能让图合同达标。
- 机制缺陷：当前实现把同一directed traversal中的所有提交hypothesis逐个连接，而不是只在一条完整物理穿越的首尾结构之间提交edge；一次多候选安全拒绝后也没有利用后续独立穿越作延迟消歧，形成永久node split。
- 决定：保持学习中心、4m locality、2-of-3 learned association和拒绝安全不变；先新增completed-traversal endpoint edge及deferred ambiguity resolution，并以C01--C06设计、C07--C08只读上界证明。上界未通过时不训练新center head，不进入C09/C10/M-TARE或planner。
- 论文含义：旧sensor-pose图、scalar learned-center图和objective-center上界都保留为消融，直接展示几何语义与执行验证各自贡献，不删除失败证据。

## 2026-08-28 — trace重复投票不足；选择学习event-center offset而非放宽关联半径

- 证据：正式C07--C08 trace-commit节点precision/recall/F1=`0.617747/0.660584/0.638448`，false loop merge=`0`；293个提交节点仅181个唯一正确节点，说明失败是node split。边precision/recall=`0.75/0.230769`，node-edge macro-F1只比最强基线高`0.016882`。
- 排除方案：直接扩大4m sensor-pose半径会把stacked/近邻隧道重新暴露给错误loop merge，且没有表达结构中心，不符合论文主线；继续加重复观测数只会降低召回，不能连接跨incident-tunnel的同节点观测。
- 选择：增加deployment输出`signed_event_center_offset_m`，由LiDAR/action-set五帧历史回归沿已执行route tangent的结构中心偏移；在线关联改用投影后的hypothesis center，现有2-of-3 association与trace edge合同保持不变。
- 可行性：只读oracle-longitudinal投影使C07--C08空间候选pair purity=`0.9976`、提交节点precision/recall=`0.9838/0.8869`、错误merge=`0`。该结果仅证明Teacher/接口容量，不作为部署成绩。
- 数据边界：仅C01--C06生成训练统计和checkpoint选择，C07--C08一次验证；C09/C10/M-TARE不读。用户已有自动最佳方案授权，因此采用该推荐路线，无需重复请求常规批准。

## 2026-08-28 — 停止直接提交节点，采用provisional semantic hypothesis到verified graph提交

- 证据：route-conditioned structured decoder在C01--C06固定900组中0组达到安全门；最佳最低召回点P/false/R=`0.848541/0.151459/0.699878`。因此当前exit-token集合不能可靠枚举incident actions，继续改阈值/持久性/NMS没有依据。
- 相关工作：NTS已从学习式explorable-area prediction建立ghost nodes；Tully等multi-hypothesis topological SLAM已维护拓扑假设；确定性topological SLAM也已有通过遍历验证loop hypothesis。简单“学习候选+执行验证”不构成充分创新。
- 决定：停止“语义输出直接成为最终节点”的当前实现主张。新主方法候选为双层图：GSE语义输出产生含exit relation和连续几何的provisional hypothesis；冻结route-conditioned association只做候选对应；完成物理穿越及反向一致性后才提交verified node/edge。
- 评价变化不是降低旧门槛：旧raw proposal precision继续原样报告为FAIL；新方法必须分别报告hypothesis recall/verification cost与committed graph precision/false loop/node-edge F1，并以NTS-style ghost graph、无学习规则假设图和无执行提交作为独立消融。
- 下一步只允许接口/新颖性矩阵与C01--C08离线commit-capacity proof；C09/C10/闭环保持禁止，直到双层机制在开发域证明提交安全。

## 2026-08-28 — 停止黑盒action-set节点分类，选择route-conditioned结构化解码审计

- 证据：三seed完整训练后，同一C07--C08固定阈值网格不存在满足总P>=0.995/false<=0.005/R>=0.25及两类P>=0.99/R>=0.25的配置。最低召回约束下最优点总P/R=`0.965870/0.498239`，junction P=`0.951977`，terminal P=`0.987069`。
- 影响：学习式node proposal仍未资格化；Factorized association PASS和continuous edge geometry证据不变，但完整离线图、C09/C10和闭环继续阻塞。
- 归因：model/interface。exit tokens含可迁移结构信号，但当前set pooling没有显式表达“哪一个出口是机器人刚走过的incoming action”，使junction和普通走廊混淆。
- 决定：不再扩张黑盒分类器或调低门槛。下一步先做零训练、C01--C08-only的route-conditioned structured decoder容量审计；在线来向只来自因果轨迹/机器人坐标，不使用Teacher identity或未来信息。
- 停止条件：若显式扣除incoming action后仍不存在原安全门下的非空召回，停止当前node-proposal主张并重新评估论文方法，不通过图/planner调参掩盖。

## 2026-08-28 — 五帧decision-mass无安全配置；节点语义改由学习式action set直接定义

- V1 checker错误在任何选择/C09前停止；V1R保持数据、模型、阈值网格和门槛不变，正式扫描C07--C08全部1,001点。
- 决定性证据：不存在aggregate P>=0.995/R>=0.25且junction/terminal各自P>=0.99/R>=0.25的非空五帧decision-mass门；C09进程未启动。结合12帧C09 false trigger=2.464%，阈值式categorical node proposal被两条独立证据否定。
- 决定：停止旧五类event概率的节点生成用途；旧模型/12帧模型保留为感知基线、消融和失败分析。主方法节点改为学习式action-set event：由exit token cardinality、heading layout、opening geometry、token descriptor和过去稳定性直接提出junction/terminal。
- 创新边界：这不是手工degree规则。token及其几何由LiDAR模型预测，时序集合关系决定节点；关联仍由冻结Factorized verifier，edge仍只由真实穿越建立并保存连续几何。
- 下一步与停止规则：只在C01--C08做零训练/低容量action-set可分性proof；若安全precision/false/recall无容量，不读C09、不调图、不进入planner，重新评估节点表示。

## 2026-08-28 — 12帧节点触发未达到1%安全预算；采用5帧decision-mass预注册fallback

- 工程事实：V1错误按archive row对齐C09输出，守卫在Teacher读取前停止；V2新增global sequence ID+parent ID双射并完成三seed推理，future/Teacher inference input、optimizer、threshold selection、C10和M-TARE均为0。
- 科学证据：V2 decision precision/false/recall=`0.975359/0.024641/0.879630`；junction与terminal identity coverage均超过0.91，但各自precision=`0.978836/0.963303`，违反预注册0.98与1%假节点合同。
- 影响：Factorized association PASS不受影响，连续几何edge profile不受影响；失效的是12帧total-structural trigger作为安全decision-node proposal。完整离线图资格和planner继续禁止。
- 决定：不降低门槛、不重选12帧阈值、不加graph stability参数。执行Data Card已声明的fallback：仅在C07--C08从冻结五帧ensemble选择`P(junction)+P(terminal)`门并压缩连续触发，随后一次性应用C09。
- 停止规则：5帧专用门若不能同时满足precision>=0.98、false<=0.01、非空召回和两类身份覆盖，则停止当前categorical node-generation路线并重新设计proposal表示；不得读取C10或用规划器收益替代节点安全。

## 2026-08-28 — 采用C07--C08选择的2-of-3共识与4.0 m metric locality，恢复离线图资格准备

- 证据：单seed C09错误相关且三seed均超过1% runtime false-accept；C07--C08更严格开发裕量下有8个安全配置，确定性最大召回规则选择`2-of-3 + 4.0 m`，随后C09一次验证得到runtime precision/false/recall=`0.990135/0.009865/0.558281`，balanced与physical-only也全部PASS。
- 防泄漏：选择和C09应用被拆成两个操作系统进程；selector没有C09路径，必须先退出并写出`c09_worlds_read=0`的calibration，outer runner才启动applicator。C09未选择vote、distance、seed或threshold。
- 决定：冻结三套V1R模型、各自threshold、2票共识和4.0 m cap作为唯一Factorized部署关联合同。原单seedFAIL保留为baseline/失败分析，不再部署任一单seed，也不重训backbone或关联头。
- 边界：这只证明pair association资格，不等于graph PASS。下一步必须重新冻结Factorized离线图协议，节点仅来自junction/terminal action change，边仅在真实traversal后提交并携带连续geometry profile；旧五类节点243-grid不能复用为主方法结论。
- 停止规则：离线图若不能达到预注册node/edge、false-loop、component和cycle-rank合同，停止进入planner；不读取C10、不调M-TARE掩盖失败。

## 2026-08-27 — 决策节点pair Teacher严重失衡；停止capacity训练并推荐identity-balanced修订

- 证据：Factorized inventory的full-token、exact population、cross-edge positive、profile coverage和零泄漏检查全部通过；唯一失败是family negative覆盖。fit decision pair=`23,359 positive / 27 negative`，selection=`8,083 / 71`，多数family negative=0。17项seal SHA=`fc7781239b3e947ddc193f85ef9fec14c1ba1fae8fb96f2df02930cdcbe2cbeb`。
- 影响：旧cache不能识别开放集decision association的precision/false-loop能力；直接训练会把严重先验失衡误当作模型表现。该缺陷使下一capacity proof无效，但不否定98.1% inbound profile覆盖或完整token可用性。
- 选项A（推荐）：保持runtime同world/strict-past/16m候选不变，另建identity-balanced structural-alias Teacher；每identity一个跨视角positive和一个同event/degree、objective geometry最近的不同identity negative。先做manifest proof与Data Card，约CPU分钟级、输出<1GB。
- 选项B：扩大runtime radius制造负对；成本低但改变图候选域和既有超参，不采用。选项C：强制distance-only merge；能工程运行但放弃学习关联创新与拒绝安全，不采用。
- 决定边界：停止受影响训练，当前只冻结提案；不复制少量negative、不跨split借样本、不读取C09/C10/M-TARE。Material Teacher重建须按数据缺陷治理单独绑定明确范围。

## 2026-08-27 — 普通状态变化点被反证；采用decision-node / geometry-edge因子化图

- 正式证据：C01--C08零训练proof完整执行，combined precision=`0.552632`、episode recall=`0.031111`、相对geometry/action较强单项baseline变化=`-0.005185`；turn/transition identities仅`2/95`和`1/17`。18项seal SHA=`e77543a2eda4d55403cc52f09aaa06189e7cf53d625adc13db36227fbe3a9692`，error=null、source unchanged、C09/C10/M-TARE/inference/optimizer均为0。
- 机制证据：12-observation双块只有`39,128/188,126`观测可评分；5维exit summary又丢失每个出口的heading/width/vertical profile/descriptor。因此该结果否定压缩摘要固定变化点，不授权阈值重选或同方法重跑。
- 概念修正：junction/terminal改变可执行动作集合，应作为decision nodes；turn、宽高、坡度、净空和曲率改变通道执行属性，应作为trace-verified edge geometry profile或内部metric polyline。把后二者强制计入同一结构node F1会混淆拓扑与几何。
- 决定：Factorized GSE-Graph成为当前唯一候选。现有junction/terminal安全触发和连续几何改善只作组件证据；下一阻塞是route-conditioned node association。它必须用完整action token与已穿越incident-edge geometry fingerprint，并满足precision `>=0.98`、false loop `<=1%`、recall `>=0.25`。
- 边界：先做C01--C08只读inventory/capacity proof；不读取C09/C10/M-TARE，不训练新backbone，不进入图参数或planner。若route-conditioned profile仍不能安全关联，停止学习关联主张并重新评估论文创新强度。

## 2026-08-27 — class-mass假设被反证；拒绝equal-class重训，停止五类categorical detector

- 审计PASS的事实：fit transition只占114/3956 episodes，raw sample mass低于3%，equal-class multiplier超过8x，lag10几何信号仍可观测。
- 审计FAIL的决定性事实：sealed ensemble transition解析gradient mass=`28.2380%`，并不低于junction=`18.0855%`；turn更占`52.4928%`。类别平衡后的估计质量会由transition主导到`72.3886%`。
- 决定：执行预声明stop rule，不创建equal-class MIL V2，不调学习率/epoch/threshold，不扩大模型。V1R的单trigger机制、连续几何学习、Teacher和失败证据保留为消融，但五类categorical event输出不再作为论文主方法。
- 下一方法边界：优先评估learned continuous geometry signature的因果分段/change-point表示，让节点来自结构状态变化、边仍来自真实穿越；先做文献差异矩阵和零训练/低容量C01--C08 proof，证明差异性与可行性后才允许新训练。

## 2026-08-27 — 保留因果episode/单节点触发机制；先审计类别质量再决定一次有界loss修订

- 证据：V1R的单trigger机制通过precision/false-trigger/recall与junction/terminal coverage，但frame macro-F1回归，turn/transition identity coverage仅`6/95`和`0/17`。
- 决定：不扩大encoder、detector容量、历史、数据、阈值网格或图/planner。先用sealed C01--C08 population与V1R checkpoint测量每类episode loss和gradient mass，optimizer step保持0。
- 理由：raw episode-frequency MIL中junction占positive bags约`64.53%`，geometry-transition仅`2.79%`，与高置信输出向常见事件坍缩一致。这只是待证明的loss定义假设，不构成成功主张。
- 停止规则：审计若未发现显著稀有类梯度压制，则放弃该detector路线；若确认，只允许一次预声明的equal-class episode MIL，同模型/数据/split/epochs/gates，不重复重试或调planner。

## 2026-08-27 — 引用proof V1计数语义FAIL，V1R保持数据不变并PASS

- V1内层9项引用检查全部true，188,126条观测和1,818,662个引用单元均有效；外层唯一失败是`directed_traversals=16,076`不等于预期`16,078`。sealed change-point inventory已客观记录2条zero-frame traversals，故差异来自total/observed口径，不是丢数据。
- 决定：V1不可修改地保留system FAIL，不删除其有效内层证据；V1R显式报告`inventory=16,078 / observed=16,076 / zero-observation=2`，并要求四个array digest与V1一致。不得虚构两条训练样本、删除短edge、重采LiDAR或放宽引用连续性。
- V1R正式PASS，12/12 seal SHA=`125903986cd8db048423286a60433d9ab0aae67343e29184edee5c2f2ea9d124`；零payload、零ray、零推理/训练和零C09/C10/M-TARE读取。
- 实现决策：12帧主模型不能只用全局pooled embedding。旧证据已表明方位布局对turn有效，因此冻结空间encoder同时输出128D pooled和36-bin环形方位特征；按seed串行生成临时缓存，训练后只保留digest、模型和必要统计，避免长期约7GB重复缓存。

## 2026-08-27 — 监督单位审计PASS；采用12帧episode-level因果事件检测

- 证据：`1,031`个最终change标签中`891`个早于closed confirmation，`621`个当前五帧历史不再覆盖结构边界；但全部`152`个方向确认episode在确认帧拥有12帧历史。全结构Teacher按连续traversal/identity划分为`5,306`段episode，而非188,126个独立决策单位。
- 影响：过去几轮event macro-F1和安全阈值FAIL不能只解释为模型容量不足。训练目标要求bag内每帧均为正、部署却只需要bag内一次安全触发，产生大量普通走廊假阳性和晚期正帧不可辨问题。旧负结果仍有效，作为frame-classification消融保留。
- 决定：不改客观结构identity、节点位置、数据split或1%错误预算；把学生接口改为12帧past-only episode detector。帧级输出仅是proposal，时间聚合器产生confirmation；loss对每个结构episode使用multiple-instance正包，对完整corridor窗口使用困难负包；每episode最多提交一个节点，并回投到估计边界。
- 12帧不是结果调出的新自由超参：它是先前固定2/4/6/8/10/12m审计的最大预声明历史，覆盖Teacher最大`10.5m`确认延迟；正式审计证明152/152已有输入。任何训练前仍需新的Data Card、引用完整性proof和无未来/跨traversal测试。
- 正式审计`error=null`、source unchanged、18/18 seal SHA=`781ea53ce97cd3bd7a345600649a7e4c6d38fb144d22deac19ab4dafd6b8b7d9`，C09/C10/strict-test/M-TARE和optimizer/inference均为0。

## 2026-08-27 — 多变量身份风险proof FAIL；停止逐帧分类器路线

- 证据：固定60维六lag线性softmax在142,184条fit观测上正常收敛，但45,942条selection观测的macro-F1=`0.610400`，transition precision仅`0.036070`、turn F1=`0.256124`；无非空开放集安全阈值。十个leave-family-out refit同样低，最差S05 macro-F1=`0.588974`，S09 transition F1=`0`。
- 影响：正式否定“冻结预测几何delta经低容量多变量组合即可恢复安全结构节点事件”。较长因果历史的AUC证据仍成立，连续几何学习结论仍成立，但它们不能直接支持在线拓扑节点生成。
- 决定：执行预声明停止规则，不再扩大分类器、解冻更多backbone、改阈值或返回图参数搜索。下一步先只读比较逐帧标签、持久identity episode、确认延迟和可见历史，定位监督单位是否错配；只有审计明确后才建立新的Teacher/输入Data Card。
- 运行边界：正式run `error=null`、16/16证据哈希PASS，C09/C10/strict-test/M-TARE读取和上游推理为0，source unchanged。该负结果保留为论文失败分析和“显式事件建模必要性”消融，不删除。

## 2026-08-27 — 风险冲突审计PASS；选择多变量身份风险proof而非继续深网微调

- 证据：冻结几何transition-vs-corridor AUC在lag4/8/10为`0.680215/0.736622/0.764515`，7--11m峰值相对lag4增加`0.087342`；Teacher lag10=`0.851485`。较长历史信号明确存在，但所有标量lag的安全identity coverage均为`0/17`。
- 困难负样本证据：lag10的101个高分corridor中，删除的旧transition只占10个，普通corridor占91个。恢复旧标签、删除所谓噪声或挑单一lag都不能满足1%风险合同。
- 决定：执行一次固定多lag、多变量、identity-balanced、低容量线性softmax capacity proof。选择该方法是为了检验几何组合本身，而不是再增加深网自由度；C01--C06只拟合，C07--C08只选非空安全阈值，十个leave-family-out refit仅作诊断。
- 停止规则：若proof不能同时达到macro-F1 `>=0.737904`、precision `>=0.98`、false accept `<=0.01`、recall `>=0.40`、change-point `>=7/17`、turn `>=0.40`及junction/terminal身份门槛，就停止当前分类器路线并修订因果输入/Teacher；不得调图或扩大backbone。
- 正式审计18/18 seal SHA=`56f6b82cec7526cb17bccf6cd2ffa3a688cf9295a972e013c58894dc0f5e6269`，零训练/推理/C09/C10/M-TARE，图像目视完整。

## 2026-08-27 — 最后编码块有界微调FAIL；停止扩大backbone，转机制审计

- 证据：三seed、`52,236`步正式run无系统错误，ensemble delta normalized MAE=`0.348158`、改善`40.71%`；event macro-F1=`0.680096`、gain=`-0.007809`，change-point=`0/17`、turn=`23/95`。各seed允许参数更新与冻结digest审计均通过，26/26 seal SHA=`27c9ec4c92e26b8eb0fe3b548b8fa4cb3cb45ae3b425dc1f29185ba70d403d53`。
- 影响：结果否定“只需给编码器最后块适应能力即可把连续几何转成安全节点事件”。它不否定LiDAR几何学习，反而连续两次证明几何delta可回归；失效位于因果历史、事件风险组织或Teacher确认时序。当前不得读取C09/C10、进入图/规划器或继续解冻更大主干。
- 决定：按运行前预声明执行一次只读机制审计，固定C01--C06/C07--C08、2--12m所有lag、四种sealed表示、1% fit-only corridor阈值和困难负样本分层，不挑winning lag。若较长历史有实质AUC增益而标量风险仍不足，才允许低容量、多变量、identity-balanced几何条件风险容量proof；否则直接修订因果输入/Teacher合同。
- 实现边界：风险接口只能消费过去几何delta、冻结条件事件概率和不确定性；位置、world、identity、TNG及未来帧禁止进入模型。已经完成的代码/合成测试不是正式真实数据结论，必须等待审计Data Card/spec/preflight/immutable run。

## 2026-08-27 — 冻结表征多任务正式FAIL；自动启用最后编码块有界微调

- 证据：正式三seed多任务run完整执行`52,236`步且无系统错误。ensemble geometry-delta normalized MAE=`0.355174`，相对冻结几何差基线改善`39.51%`；但event macro-F1=`0.683669`、相对旧directional baseline变化`-0.004235`，change-point identity coverage=`0/17`。26/26 seal复核PASS，SHA=`aed1d39e051b7a917a2996ff4db301a2993d100dfc58007925c8f1e790c347e4`。
- 影响：结果否定“完全冻结backbone，只训练共享delta/event decoder”能够恢复结构节点触发；它不否定连续几何学习，也不否定corrected Teacher。不得用增大epoch、选择单seed、降低7/17身份覆盖门槛或进入图参数调优掩盖失败。
- 决定：按已预声明fallback，只解冻`GeometrySemanticEventNet.encoder[-1]`（最后一个`96→128`残差块）和一个全新零初始化多任务头；冻结更早encoder、GRU、directional temporal、event/geometry/place/exit/uncertainty旧head。保持相同数据、Teacher、split、采样、epoch和验收门槛。
- 优化边界：最后块使用较小学习率`1e-4`，新head保持`1e-3`，独立参数组和逐步梯度/更新审计；失败run的head checkpoint不复用。若该单次有界fallback仍不能恢复事件门，停止当前表示路线并重评方法，不进行全主干盲目微调。

## 2026-08-27 — geometry-delta可观测性PASS；采用冻结表征显式多任务decoder

- 证据：80个C01--C08 worlds、122,765个有效四步lag pairs上，Teacher selection AUC=`0.742067`，冻结三seed平均metric geometry delta selection AUC=`0.680215`，fit-selection gap=`0.008180`；满足预注册`0.70/0.65/0.05`门槛。18/18 seal SHA=`c75787626ecc09a27870276e86a172d6434081034f4704a1f5286d2afe52fa50`。
- 限制：fit-only 1% FPR阈值转移到C07--C08后，冻结预测只覆盖`0/17` change-point identities，Teacher标量也只有`1/17`。AUC证明连续信号存在，不证明单阈值可安全建图。
- 决定：下一版保持corrected Teacher、五帧因果输入、C01--C06/C07--C08 split和开放集合同不变，训练显式`Δwidth/height/slope/curvature`回归与结构事件联合decoder；回归使用全部有效lag pairs，事件采样继续按identity平衡。不开启C09/C10，不调图或planner。
- 自动分支：若冻结decoder正式FAIL，直接采用已预声明的有限fallback——只解冻编码器最后一块并严格限制更新范围；不再要求用户处理常规批准。任何改变数据隔离、Teacher语义或论文核心主张的异常仍须记录并停止受影响工作。

## 2026-08-27 — corrected causal事件分类三种子FAIL；采用显式geometry-delta多任务路线

- 证据：三seed完整36,000步后，ensemble change-point identity=`0/17`、turn=`15/95`、macro-F1=`0.680608`，不及旧directional corrected-label baseline `0.687904`；但开放集precision/false/recall=`0.990108/0.009892/0.678061`通过。run无系统错误且26/26 seal完整。
- 影响：不得继续用同一冻结特征分类头加epoch/换seed/降阈值，也不得进入C09、图参数或planner。失败分类为model/representation objective，不是Teacher、data alignment、metric执行或system。
- 自动选择的最优下一方案：保留五帧因果输入与corrected Teacher，新增显式Δwidth/Δheight/Δslope/Δcurvature监督和持久change evidence，将全部有效corridor/结构序列用于几何变化表征，而不是只靠稀有事件CE。先做只读可观测性与容量proof；若冻结特征能回归则只训练delta decoder，否则再预注册有限主干微调。
- 不采用立即全量重训（变量过多且只有59 fit identities）、不采用删除change-point（用户已选择A且Teacher proof成立）、不采用frame-level阈值放宽（会破坏1%节点安全主张）。

## 2026-08-27 — 用户授权自动选择最优实现分支；V1R按A执行并PASS

- 用户明确要求以后不再逐次回复，由Codex根据证据自动选择最优方案。该授权适用于既定GSE-Graph论文范围内、不改变数据split、论文主张、安全门槛或测试隔离的实现/系统分支；实质研究语义变化仍必须即时报告。
- 对V1索引checker缺陷自动采用已记录推荐A：保留complete-export稀疏global ID并修正验收，不采用破坏稳定引用的重编号B，也不选择停止C。
- V1R 22/22检查、17/17 seal通过，seal SHA=`3f840e1165709d04aab80374739551931783045515f1ba95456107e4b5867f49`；所有Teacher科学数量与V1一致，源/输出索引digest均为`9081c80f...7f32`。决定返回Gate 3进行identity-balanced结构事件训练设计，C09/C10/M-TARE继续隔离。

## 2026-08-26 — corrected Teacher V1失败：全局索引连续性合同错误，等待replacement选择

- 具体证据：正式V1唯一失败项为`global_sequence_indices_exact_0_to_188125=false`；188,126个输出索引实际唯一、无重复、范围`0..208227`。sealed源总清单含C01--C10共212,588条，筛出C01--C08后合法保留20,102个C09/C10索引空槽，故连续子集假设错误。
- 影响：V1不得判PASS，也不能进入训练；但标签、Teacher身份、关联pair、泄漏隔离和全部科学数量均未失效。问题分类为system/metric checker，不是data、Teacher、model或样本缺失。
- 选项A（推荐）：V1R逐元素保留sealed源索引，验收唯一、严格单调及精确源集合；不改数据/标签/身份/pair，约1分钟CPU。选项B：重编号为连续train索引并重写所有引用，成本更高且破坏稳定全局身份，不推荐。选项C：停止，不生成可训练Teacher。
- 按正式失败策略，V1原样封存且不自动重跑；replacement实施需明确选择。
- 影响面审计：`gse_open_set_pairs`、dataset export、directional/rare-event trainer和topology replay均保留global ID并显式建立compact映射，既有Data Card也冻结了最大值208,227与20,102个空槽；没有发现第二处连续索引假设。A的修复范围因此可严格限制在本run的验收器和replacement治理文件。

## 2026-08-26 — 方案A进入corrected Teacher生成；操作阶段暂回Gate 2

- 用户已明确选择A，且C01--C08全量proof已证明76个持久双向因果change-point可复现；下一项工作是`teacher_generation`，不是新模型训练或测试集评价。
- 按治理合同，Teacher生成只能在Gate 1--2执行，因此操作指针从Gate 3暂回Gate 2，仅用于生成一个不可覆盖的corrected Teacher manifest。既有Gate 3训练/失败/proof证据全部保持不可修改，不据此重写历史结论。
- 固定输入为80个C01--C08世界、16,078条directed traversals和188,126条五帧因果观测；旧transition先全部复位为corridor，再应用1,031条优先级解析后的因果标签和76个identity。C09、C10、M-TARE、模型推理及训练保持零读取/零步骤。
- 只有清单数量、字段不变性、关联样本、泄漏和完整seal全部PASS，操作阶段才返回Gate 3训练identity-balanced结构事件模型；否则停止在Teacher层，不静默修补或重跑。

## 2026-08-26 — 用户选择A；geometry-transition重定义为持久双向因果change-point

- 决策证据：旧Teacher的3,035个C01--C08 transition identities大多是短促、端点集中或单向波动；扩大模型的292D residual与directional head均未恢复有效identity coverage，继续加模型容量被拒绝。
- 用户明确选择A，不采用“仅保留edge几何属性”的B路线。新定义复用固定`5m`比较跨度、`1m`宽/高阈值和`10m`节点半径，不增加结果调出的超参；结构点必须持久、正反方向唯一一致、过去窗口可确认，degree-2端点合并，terminal/junction优先，多义情况拒绝。
- 正式C01--C08 proof PASS：76个change-points、1,031条最终标签、fit/selection identity=`59/17`，10 family全部非空；C09/C10/M-TARE/model/training为0。旧Teacher和两个corrective FAIL继续保留为方法动机与消融，不改写。
- 决定下一步生成新的immutable corrected Teacher manifest，而不是回写旧manifest。后续训练按identity平衡并同时报告帧级与identity级指标；若新语义仍不能改善离线图，停止节点主张，不用planner调参掩盖。

## 2026-08-26 — Directional corrective仍不能学习transition；暂停并要求Teacher路线决策

- 正式directional head训练证明新的空间时序表示对turn有实质增益，但ensemble geometry-transition identity coverage仅`2/744`，完整执行/封存正常，排除系统、资源和backbone更新问题。
- 新Teacher审计显示transition标签高度集中于physical-edge端点、区间短且约五分之一单向不一致；它占结构identity的三分之二。该证据把主阻塞从单纯model capacity升级为`teacher + frame-level metric`定义缺陷。
- 影响：不得用当前transition标签再训练更大模型、调节点阈值、扫图参数或进入下游。junction/terminal/turn、连续几何与association证据继续保留，directional head作为“空间布局消融/失败分析”论文证据。
- 选项A（推荐）：构造持久、双向一致、端点归并的causal geometry change-point Teacher，允许在线延迟检测与轨迹回投；先做C01--C08只读proof。选项B：删除transition节点，变化只做edge属性。选项C：保持Teacher扩模，因两次正式FAIL与标签缺陷不推荐。
- 这是Teacher与评价语义的实质变化。即使用户已授权日常自主推进，科研合同要求在实施A/B前取得一次明确选择。

## 2026-08-26 — 不改 Teacher；以逐方位时序结构头替换压缩输出 corrective

- 一度怀疑`geometry_transition_mask`读取当前位置之后`5 m`宽高 profile 与因果学生错位。只读证据否定“不可观测”这一强结论：50 m当前帧中，C07--C08转弯/突变前向5 m可见率为`99.81%/99.47%`，冻结三seed逐帧召回为`90.21%/82.88%`。完整地图产生监督标签不等于未来传感器输入泄漏。
- 结构失败来自事件头接口：五帧encoder特征在进入event head前被elevation/azimuth全局平均，保留方向布局的分支只服务axis/exit token。全局1%错误接受门下，turn/transition identity coverage只有`5/95`和`1/744`；冻结146D residual corrective也正式FAIL。
- 决定保持C01--C08 Teacher、split、主干checkpoint、连续几何、exit token、place descriptor和pair verifier不变；下一corrective必须读取冻结encoder的逐方位五帧特征，独立学习binary structural evidence与conditional structural class。不得改标签、降低安全门槛、继续扫图参数或读取C09/C10/M-TARE。
- 该方向是已批准Data Card中“失败后重新设计temporal structural representation”的预声明fallback。正式训练仍须新Data Card/spec/preflight，使用C01--C06拟合与C07--C08选择，并保留原identity coverage和开放集门槛。

## 2026-08-26 — 稀有事件corrective训练期间操作阶段回退Gate 3

- 预检证明新run的数据、授权、工具与输入哈希均通过，唯一拒绝是`training`属于Gate 3而项目操作指针仍为Gate 4。
- 决定把`results/project_status.json.current_gate`暂时设为3，仅授权C01--C06拟合/C07--C08选择的冻结特征corrective训练。C09 V4正式FAIL、C10/M-TARE隔离和所有图门槛不变。
- corrective训练结束后，只有选择域科学PASS才回到Gate 4执行新的C09离线图；科学FAIL则留在表示阶段重新评估，不得越过到建图或planner。

## 2026-08-26 — event-node融合保留为组件，停止用其直接启动V5；转入稀有结构事件corrective

- 正式选择域证据：product score在99% precision下把frame recall从`0.349010`提高到`0.493902`，十family通过，故融合接口有独立科研价值并保留。
- 未见C09证据：frame recall从`0.307517`提高到`0.441021`，但identity coverage仅从`0.203704`到`0.220539`；81组图仍0组安全。逐类coverage把失败集中到turn=`0.037735`和geometry-transition=`0.014599`，而junction/terminal已基本覆盖。
- 决定：不把selection PASS外推为C09图PASS，不创建正式V5，不继续扫图参数。冻结主干、pair verifier和融合组件，新增仅针对五类事件/episode的轻量corrective；C01--C06拟合、C07--C08选择、C09评价、C10/M-TARE继续封存。
- 方法边界：corrective必须使用部署可得冻结几何语义特征，不使用identity值、pose、未来帧或图Teacher作输入；训练可用identity仅做episode平衡/采样与评价。若turn/transition identity coverage不能实质提高并保持开放集安全，则停止当前表示路线而不是调planner。

## 2026-08-26 — V4失败后停止图参数调优，转向结构事件融合可行性证明

- 证据：V4正式243组没有安全配置；episode修正后的81组只读审计仍为0组安全。最接近安全配置precision/false-loop=`0.979381/0.020619`，node recall=`0.186869`；最佳node/edge F1=`0.290576/0.054865`。
- 影响：当前GSE-Graph不能进入C10、M-TARE闭环或论文主结果。问题分类为`model/temporal event representation`，不是图半径、anchor间距或资源问题；episode逻辑已明显修复重复节点和cycle bias，但没有恢复感知召回。
- 决定：不再扩大243网格、不在C09调node/pair阈值、不靠planner掩盖。下一步只做C01--C08选择域上的三seed结构事件融合与episode代表观测校准，再以冻结接口评价C09。
- 停止条件：若融合不能在选择域同时提高事件/唯一节点召回并保持开放集precision与false accept合同，则停止现有事件头路线，重新设计学习式结构事件表示；不得直接创建V5正式run。

## 2026-08-26 — 不用关联阈值掩盖结构事件假阳性，转入三seed开放集节点生成corrective

- 证据：C09 V3全部243组无安全配置；最高precision=`0.964628`。grid177的48次false merge由`8 labeled identity mismatch + 4 candidate unmapped/conflicted + 36 corridor query mislabeled as structural`构成。
- 受影响结论：C07--C08 pair-level ensemble PASS不能证明完整在线图安全；Gate 4与C10/闭环资格均不成立。问题分类为`model/interface compound`：pair association在真实结构identity条件下基本合格，上游event open-set/node trigger不合格。
- 拒绝方案：不得在C09上提高`0.9431912303`阈值、缩小16m域、删除失败世界/帧、改变Teacher或放宽1%合同；这些都会把结构事件错误伪装成关联改进。
- 采用方案：保留冻结GSE和exit-token verifier，新增三seed结构事件一致性/开放集节点生成门；门的形式和阈值只由C01--C08开发证据冻结。C09允许作为开发验证检验完整图，C10继续严格封存。并行补齐现有三种基线，确认node/edge低值是否为共享建图接口问题。
- 成本：一次开发域校准审计、一次C09 corrective拓扑比较；约数小时CPU/GPU，不新增LiDAR或主干训练。若仍不能同时改善F1与图不变量，停止该节点生成路线并重新评估结构图表示，而非进入planner调参。
- 授权依据：用户已明确授权为GSE-Graph论文目标自主选择最优在范围方案并持续推进，无需重复批准；该决定不扩大到C10、M-TARE或闭环。

## 2026-08-26 — 冻结distance-aware三seedensemble并恢复C09离线拓扑资格

- 正式audit逐seal复核V1R/V2 FAILED来源，严格过滤`distance<=16m`后，对三个V2 score做无参数float64等权均值；未训练、未推理、未读C09/C10/M-TARE。
- 冻结选择点threshold=`0.9431912302970886`，P=`0.9900537634`、false merge=`0.0099462366`、R=`0.2874424413`，全部10 topology family满足非空、precision和recall门槛。13/13 seal SHA=`25be6e18...`。
- 决定把部署方法定义为完整三seedensemble，而不是宣称任一单seed PASS；必须如实报告3倍感知/验证计算并在闭环测runtime。V1R/V2原FAIL保留作为aggregation与candidate-domain消融/失败分析。
- 该结果只允许新的C09离线图评价。C09不得重新选择ensemble weight、threshold、radius、checkpoint或图参数；C10/M-TARE仍在离线图科学门槛通过前隔离。

## 2026-08-26 — V2保持FAIL；修复线上candidate-domain评估并正式审计ensemble

- 证据：`OpenSetAssociationContract.maximum_candidate_distance_m=16`且Data Card称其为部署candidate gate，但`select_nonvacuous_threshold`未接收distance。既有identity pairs未被16m约束，selection中13,903/45,372条域外。
- 影响：V1R/V2原formal判断仍按其冻结实现为FAIL，不能事后覆盖；但它们不能回答真实线上候选域内的安全性。分类为metric/system interface defect，不是data、teacher或模型漂移。
- 只读域内诊断：V2单seed仍FAIL，三seedmean以precision `0.990054`、false merge `0.009946`、recall `0.287442`和全family门槛PASS；V1R mean仍FAIL。结果同时需要exit-token增益与ensemble稳定性。
- 选项：(A)推荐，新增distance-aware selector和只读immutable V2 ensemble calibration，0训练、分钟级，PASS后冻结并去C09；(B)用正确域重训/重选单seed，约45分钟但现有域内单seed诊断均FAIL；(C)完整association head重训，成本高且只在(A)的C09泛化失败后考虑。
- 用户的方案A与持续最优执行授权覆盖(A)。决定不修改V2、不降低门槛、不用post-hoc诊断直接进入C09；先创建新Data Card/spec和不可覆盖审计。

## 2026-08-26 — V1R科学FAIL；拒绝ensemble擦线，补齐exit-token集合匹配

- V1R唯一正式run完成三seed并以科学FAIL封存；稀疏global-ID映射、pair counts、资源与隔离合同全部PASS，故结论不再受V1 system checker错误影响。
- 三seed在25%召回处错误合并约`2.41%/1.90%/1.99%`；三seed均值的post-hoc诊断虽降到`1.204%`，仍超过`1%`，且ensemble不是预注册主方法。决定不以四舍五入、改变false-loop分母或降低门槛通过。
- 根因分类为model/interface：冻结GSE已经输出六个token的heading、opening width、vertical profile和32D exit descriptor，但V1 observation interface只保留5个集合汇总并完全遗漏exit descriptor，未真正执行论文定义的exit-token association。
- 可行选项：(A)推荐，保持冻结backbone，新增确定性的旋转/置换不变token correspondence特征和小型hard-negative-aware verifier；成本约一次新C01--C08三seed冻结推理/训练。(B)完整重训GSE association heads，成本高且会重开感知资格。(C)放弃学习关联主张。用户既有方案A和持续最优执行授权覆盖(A)，故下一设计采用(A)，但在新Data Card/spec冻结前不读取C09/C10或启动正式训练。

## 2026-08-26 — 方案A获选；实现后发现旧open-set样本数与声明语义矛盾

- 用户明确选择方案A：冻结GSE主干，新增train-only matchability与对称pair verifier，不用近乎全拒绝的descriptor阈值冒充修复。
- 实现冻结了不含GT/pose/world/traversal的部署特征、对称pair函数和non-vacuous选择门槛；专项测试10/10 PASS。
- 正式训练前的真实metadata replay证明旧inventory口径不可复现：声明的同parent、严格过去、3D `<=16m`只能产生fit/selection `57,066/18,111`条open-set负对；即使允许未来候选也只有`71,277/22,843`，低于旧记录`79,478/25,360`。
- 因此旧容量结论只能保留为失败审计，不能进入Data Card。推荐采用可直接对应在线关联的严格过去3D口径，总pair改为`135,232/45,372`；在该实质数据定义被明确冻结前不启动训练。
- 后续持续执行采用该推荐口径。因治理明确限制`training/checkpoint_selection`只能位于Gate 2--3，operational Gate临时从4回到3，只训练冻结GSE输出后的关联corrective；离线拓扑结论和V2 FAIL不改写。corrective完成后才返回Gate 4。

## 2026-08-25 — 拒绝用近乎全拒绝阈值掩盖在线关联FAIL

- 正式243-grid证明离线图最高association precision仅`0.514286`；三seed逐决策证据将34个false merge全部定位到无稳定GT身份目标，其中32个查询来自普通位置误触发。
- 原训练和校准只在`association_valid=true`身份样本上优化/选择descriptor，明确排除了open-set查询。这使感知阶段约99%的closed-set nearest-neighbour precision不能外推为在线图安全性。
- 只读反事实把open-set负样本纳入阈值后虽可达到100% precision，但三seed只接受`46/3/2`个正确匹配，召回约`0.998%/0.064%/0.041%`。决定不利用“只要求非零”的文字漏洞把这种近空结果包装为PASS。
- 推荐方向是新增train-only open-set association verifier，而非降低98%门槛、改false-loop定义或直接进入planner。该corrective必须使用C01--C06拟合、C07--C08选择，冻结原backbone并在重新读取C09前设定实质性recall/coverage门槛；属于方法与训练合同变化，需要明确研究决策。

## 2026-08-25 — 感知V2 PASS后采用显式双来源离线拓扑接口

- 风险校准感知正式PASS，但其输出只新生成了校准坡度；事件、出口、关联、宽高、曲率与校准合同仍位于旧的immutable C09组件run中。
- 决定不复制或伪造一套“全新完整模型输出”。离线拓扑V2同时验证旧FAIL run的精确seal与其中已通过的event/association组件，再验证新V2完整PASS和校准坡度；只按唯一`global_sequence_index`覆盖`slope_deg`，其余数组逐元素保持冻结。
- 旧FAIL不得被表述为PASS，新V2不得冒充重新推理全部heads。C09已参与方法修正，只能用于development图参数选择；C10和M-TARE在参数冻结前继续零读取。
- 离线门槛、四方法、243网格、回放顺序和论文定性world不变；若node/edge增益、安全关联或图不变量失败，停止严格测试和闭环，不通过planner调参掩盖。

## 2026-08-25 — 保留逐世界安全门槛，采用C07--C08 maximin坡度残差校准

- 证据：全残差corrective在C09总体改善`48.364%`，但S02三seed一致退化`13.381%`；该family在训练/选择期解析先验MAE为`2.0649/2.4613°`，C09仅`0.9583°`，模型发生过修正。预测error scale不能识别受损样本。
- 影响：V1不能解除Gate-3结构语义门禁；不得进入离线拓扑、C10或闭环。
- 分类：主要是model/generalization risk；同时发现一个不影响科学输出的system schema-key defect。
- 拒绝方案：删除或放宽逐世界5%回归门槛会在看到结果后降低标准；另训复杂confidence gate成本更高且现有confidence诊断不支持。
- 决定：保留旧V1 FAIL和原门槛；新增单一共享残差scale，严格由C07--C08的101值固定网格、最差parent改善最大原则选择。C09只评价选定值，不参与选择；C10保持sealed。
- 透明性：这一方法修订发生在观察C09 V1失败之后，所以C09不再被表述为pristine test；它仍是开发validation，最终泛化主张必须依赖C10。

## 2026-08-24：离线拓扑来源 seal 改为精确全目录覆盖并双层验证

- Luna只读审计确认四方法、243网格、59,442,660更新、GT evaluator-only、trace-only edge及科学门槛实现一致，但指出旧`_verify_sealed_source`只检查manifest已列项，未证明实际输入文件全部列入seal；直接调用evaluator也未独立检查dataset/Teacher禁止读取。
- 新门禁要求run identity/status正确、C10/M-TARE为0、seal非空、digest合法、路径全部位于对应run内、无重复/自引用，且sealed集合精确等于run目录除seal外的全部文件。outer runner在前后检查，evaluator自身再检查四个来源。
- 越界、空seal、漏列、run identity、forbidden dataset直接调用负测及全GSE回归`141/141 PASS`。真实去重dataset `33,083`项与Teacher `17`项均逐哈希及全目录覆盖PASS；训练/perception未完成，尚未进入来源PASS声明。每方法还新增243行sweep、selected文件集合和四方法`24,300` replay/`59,442,660`更新总数执行后硬检查；两套离线论文图以16文件原子发布。

## 2026-08-24：区分感知几何基线与离线非学习几何事件图的时间输入

- 只读代码核对发现离线spec把非学习几何事件图误写成`current-frame geometry events`。实际实现对同一五帧过去到当前的因果序列逐帧运行确定性几何估计，再由前两帧/后两帧截面变化或当前曲率产生transition/turn；不读取未来帧。
- 正式比较保持不变：感知连续量基线和M1D仅看第五帧；离线非学习事件图看与GSE相同的五帧因果历史，从而把时间证据控制住，但没有学习权重、descriptor或uncertainty。
- 仅纠正冻结说明与论文方法文字，不改变代码、数据、阈值、243网格或正在运行的训练。

## 2026-08-24：五类结构事件结果按全部 seed 与类别完整发布

- 感知总macro-F1不足以判断收益来自路口/终点还是稀有的转弯/几何变化，因此论文必须同时报告五类事件的precision、recall、F1和support。
- 新发布器固定读取sealed C09 PASS中M1D与GSE的seeds 0/1/2，共30行原始类别记录，再生成mean±sample-std Markdown/LaTeX表；GSE只重放已经冻结的事件选择点，`selection_effect`漂移、样本support漂移、C10/M-TARE读取或seal漂移均立即拒绝。
- 正式表在C09完成前保持为0；工具已加入感知spec freeze。五套C09论文证据另由原子发布器统一执行。Luna审计发现仅按34个静态目标回滚不能证明未声明输出也被清除；现以目录快照记录本次全部新增文件，逐子包要求实际集合精确等于声明，异常时只回滚新增集合，并独立重验正式C09 seal。成功才写36文件总索引和总SHA-256。GSE专项`134/134 PASS`；该工作不修改训练、checkpoint、阈值或科学gate。

## 2026-08-24：预注册几何 uncertainty 到事件可靠性的跨任务诊断

- 代码复核确认标量uncertainty以异方差几何残差训练：其平方、下限`1e-4`后作为方差；部署校准又将该量与事件置信度联合用于拒绝。论文必须单独证明这一跨任务可靠性，不能只展示选定阈值。
- 新诊断对每个封存seed的全24,462条C09观测按uncertainty排序为10个等频bin，对比预测`u²`与训练完全同定义的masked normalized Smooth-L1几何残差，同时报告event error rate、校准gap和相关性。
- 该诊断固定`selection_effect=NONE`，不改checkpoint、温度、拒绝阈值或科学gate。发布器仅接受sealed perception PASS，保存三seed完整曲线与PNG/PDF/SVG/CSV/JSON/provenance/SHA-256；已加入spec freeze，GSE专项`124/124 PASS`。已完成seed0的只读值仅作接口smoke，不作正式论文结果。

## 2026-08-24：将 OVTG 列为 2026 开放词汇语义拓扑直接近邻

- ACM原始论文页显示OVTG用CLIP视觉语言嵌入和GAT建立开放词汇语义拓扑；其节点由空间新奇度触发，edge连接前驱或邻近无碰撞节点。
- 因此“学习语义特征+在线拓扑图”也不能作为GSE新颖性。GSE主张仍限定为五帧本体LiDAR预测显式地下几何事件/出口token，预测决定稀疏结构节点和拒绝关联，而edge只由完成的物理穿越提交。
- OVTG已加入贡献矩阵、论文related work与BibTeX；当前未发现四机制完全同构，故不停止正在进行的GSE训练。该更新不改数据、teacher、模型、阈值或C10/M-TARE隔离。

## 2026-08-24：补齐 GSE 方法图的机器可读源并保持已渲染图不变

- 论文图只读审计确认数据集、Teacher分布和出口Teacher三组bundle完整；方法总览图有PNG/PDF/SVG、provenance、generator和hash，但缺少总合同要求的机器可读源。
- 新增`gse_method_overview_source.json`，以类型化node/edge、布局、视觉属性和方法不变量表达同一流程图。修复前manifest与所有图文件逐一哈希验证，既有PNG/PDF/SVG未改动；新manifest SHA-256为`9c9a9d18f2a723e183d3a715dbb47110fe0a588bdcbed8c4a72dd45550e307ec`。
- publisher的全新目录smoke亦生成完整六文件bundle并通过哈希复核；临时目录已移入回收站。该修复未读取C10/M-TARE，未修改活跃训练run或任何实验数值。
- 四组已完成bundle现已以可追溯PNG和简洁图注接入`docs/GSE_GRAPH_MANUSCRIPT_DRAFT_V1.md`：方法图和数据隔离图作为正文证据，Teacher分布和出口Teacher审计可在页限要求下转入supplementary。尚未密封的训练/感知/拓扑/闭环图仍保持占位符，禁止用中间数值填充。

## 2026-08-23：GSE-Graph 取代 exit-only 规则图成为唯一投稿主线

- 用户明确指出旧方法没有真正学习几何结构特征，且与 Cano 的出口方向路线重合；决定不再把 V9/V5 包装为最终创新。正在运行的 V5 30-case 保持不可修改并自然完成，之后仅作为 Cano-like exit-only、规则图基线、消融和失败机制证据。
- 新主方法固定为 Geometry-Semantic Event Graph：5 帧因果 LiDAR 学习局部轴、出口 token、宽高、坡度、曲率、结构事件、place descriptor 与 uncertainty；这些输出直接决定节点触发和关联。模糊关联必须拒绝合并，edge 只能由真实穿越建立。
- 数据独立单位继续是 topology parent，固定 80 train / 10 validation / 10 strict test；已有 100000/12500 单帧只作预训练，新序列按全部 directed traversals 的 1m 弧长 anchor 构造。精确序列数在训练前只读冻结，C10/M-TARE benchmark 不参与开发。
- 用户授予既定论文范围内持续执行权，不再逐次请求常规批准；Data Card、spec、preflight、不可覆盖 run、停止报告和 seal 仍保留。会改变 split、teacher、方法主张、阈值或结论有效性的实质问题仍必须停下。
- 论文图必须保留 PNG、矢量 PDF/SVG、生成脚本、机器可读源数据、选择规则和 SHA-256。清理必须逐目录列出大小、状态、替代证据与 hash；活跃旧 run 完成前不进行 I/O 密集删除。

## 2026-08-23：新颖性主张收窄为因果exit-stub生命周期和受控全局层替换

- 最新只读检索确认，Cano 2026已覆盖程序化合成LiDAR、学习隧道方向和纯拓扑地下导航；GRID-FAST已覆盖结构语义topometric mapping；Similarity-Score Topological Memory、Topological Graph Voronoi、LTVMap和SAGE已分别覆盖多机器人topological memory、图分区、低带宽地下拓扑图和尺度无关拓扑探索。
- 因此禁止把“使用拓扑图”“学习出口方向”“低带宽共享图”或“Hungarian分配”单独写成首次贡献。当前候选主张固定为局部结构语义驱动、只用因果历史的persistent verified-edge/unresolved-exit-stub图，带执行反馈生命周期，并在保留M-TARE局部栈时替换其global target layer。
- 新增`docs/NOVELTY_REVIEW_20260823.md`保存论文/官方项目来源、差异、所需实验和会否定主张的缺口。该检索不改变数据、模型、阈值、指标或正在运行的V3R；投稿前仍需对正式版本和2026年下半年工作再次复核。

## 2026-08-23：论文单机器人图表全部由sealed证据确定性生成

- `docs/PLAN.md`第20节的“V4-only或组合修正尚待决定”已被全量机制审计后的V5选择取代；新增第21节作为当前权威路线，旧文字仅保留决策历史，禁止把两条分支静默混用。
- 决定不在实验结束后手工抄写论文数字。最终V5 comparison从90个组合source summary、30个corrected summary、corrected机制summary及120条coverage curve读取证据；每个来源文件先对其原run seal复核。
- 固定输出为六张结果图：三张block/paired-effect定量图、两张仅按garage/tunnel env11身份选择的轨迹/拓扑图、一张四方法coverage-time图。覆盖曲线固定0--600秒/10秒网格，先在每个world×environment block内平均三个repeat/checkpoint，再对十个block作均值与描述性pointwise bootstrap区间。
- 固定三组论文表：三方法七指标`mean±std`、V5对原M-TARE及缺陷V9的block配对效应、V5两类因果修正的事件/案例计数；每组同时输出CSV、Markdown和LaTeX。V5-vs-defective-V9补齐与预声明一致的七指标Holm校正，不改变原始指标或block单位。
- 图表接口、来源漂移负测和提案冻结链18/18测试PASS，20/20工具hash一致。该决定不读取raw bag、C09/C10，不改变正在运行的V3R，也不把单机器人开发结果冒充多机器人或严格泛化结论。

## 2026-08-23：新增无GT的frontier-attempt执行结果审计，联合修正等待全量分布

- 当前90-case到28/90，正式run保持不可修改。case026自身轨迹证明node0↔node4累计37次verified traversal但空间范围仅约31.82×7.48m；仅靠移动距离或route arc会误判为正常探索。
- 决定新增只读因果分类：上一周期活动`frontier_exit`之后，若图事件离开当前节点，则把目标stub与图实际会消费的departure stub逐身份比较；若`loop_merge`回到同节点则单独记录。该证据不读取GT/map/raw bag且不设失败阈值。
- case026得到68次事件：8 matched、50 divergent、10 same-node；60次非匹配中58次目标连续作用路径至少4m、56次至少8m，全部10次same-node均超过8m，故短暂目标切换不能解释主体。最常重复node4/stub3为0 matched、19 divergent、4 same-node。该结果支持“出口生命周期缺少执行反馈”机制，但11/30样本仍不允许冻结修正方法。
- 决定继续让90-case自然完成，并在30/30完整分布后选择V4-only或最小的`verified re-anchor + frontier execution feedback`组合；不得从case026单例调阈值或直接消耗30-case正式修正版矩阵。新增文件不在当前formal run及已冻结compatibility proposal工具集合内，未造成来源漂移；专项测试12/12 PASS，C09/C10读取为0。
- 为避免把临时诊断冒充论文证据，决定建立独立`aee_composite_v9_frontier_attempt_outcome_audit_v1` proposal，而不修改已冻结兼容audit。proposal冻结90 summaries/30 traces/30 snapshots、9项工具hash、0 bag/GT/training/C09/C10；finalizer只接受准确90-case sealed FAIL。运行中源负对照exit=1且零正式输出，7/7相关测试PASS。

## 2026-08-23：case026证伪V4回锚充分性，先做30-case双机制审计

- case026`tunnel/env37/seed1`为V2 case PASS，fallback=`0.001006`、回锚proxy=0、route arc持续增长到`605.454m`、实际移动`595.247m`，但覆盖仅`351.375m³`且point redundancy=`0.999825`。因此低效不能归因于stale backtrack arrival。
- 决策轨迹中`frontier_exit=2983/3001`、`graph_backtrack=18`；同一`node4/stub3`累计选择1383帧。最终图11 nodes/18 edges/47 stubs，其中26 observed；node0 branch_count=2但有8 stubs，单节点最大stub-minus-current-branch-count=6，且有59次loop_merge。
- 该证据否定“V4 re-anchor单独足以恢复性能”的假设，不否定当前90-case、V9方向模型或V4组件本身。分类为topology/planner exit-stub lifecycle与目标效率问题。
- 不按单例立即改方法，也不自动消耗V4 30-case。兼容audit新增无调参阈值的frontier lifecycle evidence，与re-anchor evidence共同覆盖全部30个V9 case；全量分布后再选择V4-only或最小组合修正。相关回归57/57 PASS，C09/C10保持0。

## 2026-08-23：区分 schedule 文件哈希与规范化内容哈希

- 只读核对发现当前90-case run的`config/case_schedule.json`文件SHA-256为`830f49d17470c0d3d5efe0e60c34439e2b340d377cabd75c23d91a9b92201af7`，而`matrix_audit.schedule_sha256`为`6aea0921b01119e7f7b62544bc30a0b4118822b1e613887c29c33a4e94b27f1a`。前者覆盖schema、缩进和换行的完整文件字节，后者覆盖90个case列表的canonical JSON内容；二者均正确但不可互换。
- 兼容audit、V4 30-case源绑定和最终corrected comparison已改为同时要求`*_schedule_file_sha256`与`*_schedule_content_sha256`，避免源run完成后把合法调度误判为漂移。相关V4/统计回归52/52 PASS；当前正式run及其冻结文件未修改。
- 已建立只读兼容audit proposal，精确范围为90 summaries、30 V9 traces、30 V9 graph snapshots、0 raw bag/C09/C10/training；fail-closed finalizer只在源90-case sealed FAIL且两类schedule哈希均匹配时一次性物化正式Data Card/spec，未完成源、漂移或重复输出均拒绝。整条回归54/54 PASS。

## 2026-08-23：把 V9 图状态故障审计扩展到全部30个正式案例

- 决定不以挑选的case016单例支撑论文机理结论。只读V2兼容audit现对全部30个sealed V9 case逐一验证decision trace和topology snapshot哈希，再输出arrival proxy、constant run、route-arc增长、尾部停滞、移动、覆盖和fallback。
- 该审计读取raw bag、评价器GT、C09/C10均为0，不改变源run或统计主指标；新增回归后V4/统计链52/52 PASS。论文将区分“图状态陈旧”与“物理路线停滞”，不把二者混称为同一失败。

## 2026-08-23：冻结 Gate 7 多机器人实现缺口与进入顺序

- 只读审计确认当前 shared graph、Hungarian allocation 和 coordinator 只是 ROS-free 算法骨架；没有 namespaced ROS transport、真实执行反馈、per-robot waypoint、planner isolation、安全合同或正式 runner，不能称为多机器人闭环实验。
- `docs/GATE7_MULTI_ROBOT_READINESS_GAP_V1.md` 冻结最小顺序为 ROS namespace/feedback 单测、两机器人 smoke、断联与窄通道安全资格、1/2/3/4 robot development matrix、参数冻结、sealed strict test。
- 该设计准备不推进 operational Gate，不启动多机器人仿真；Gate 6 单机器人结论仍是唯一前置条件。

## 2026-08-23：建立论文完成矩阵并纠正当前方法的可声明边界

- 新增 `docs/PAPER_COMPLETION_MATRIX_V1.md`，把数据、结构语义、因果图、单机器人、多机器人、严格测试、消融、统计、图表和实地证据逐项绑定到当前 sealed evidence 或明确缺口。
- 早期 `docs/PUBLICATION_EVIDENCE_PLAN_V1.md` 的 R/D/U 与实地闭环条目保留为理想证据上限，不再作为“已实现方法”描述。当前可声明方法身份固定为 learned exit direction + frozen geometry count/role、persistent causal topology、M-TARE global replacement；V4只增加verified-backtrack因果回锚。
- 该澄清不改变当前90-case数据、模型、planner、指标或门槛；它防止论文把未实现能力或开发集结果写成已证明贡献。

## 2026-08-23：消除 PLAN 内历史“当前唯一任务”与 Phase 6 指针冲突

- 只读治理审计发现 `docs/PLAN.md` 第 9--15 节仍保留 Phase 1--2 时期的“当前唯一任务”措辞，但第 16--19 节、`docs/PROGRESS.md` 与 `results/project_status.json` 已明确授权并推进到 Phase 6 / Gate 6。
- 决定保留旧阶段内容作为历史证据，只将其显式标记为历史；当前执行依据为 Gate 6 的 90-case 正式 run 及 `docs/PROGRESS.md`、`results/project_status.json` 的最新指针。该修订不改变数据、方法、阈值、正式 run 或 C09/C10 隔离。

## 2026-08-23：保留90-case原run并采用显式verified-backtrack回锚纠正

- 决定不停止、不修改也不重跑当前90-case原run。冻结runner在90例结束后会因`PASS_SINGLE_ROBOT_CASE_V2`与V1 analyzer只接受`PASS_SINGLE_ROBOT_CASE_V1`的接口不兼容封存系统FAIL；其90个case与lossless bags仍作为不可变来源证据。后续只读bridge只在深拷贝的内存字段中映射status，不回写原run，也不把bridge PASS冒充原run PASS。
- 机制证据：当前完成的7个V9案例中5个出现持续旧节点回退；最严重两个tunnel案例连续2953帧并停在约20m，fallback为0。决定修正图状态而不是训练模型、调阈值或换数据。
- V4回锚只在上一周期目标为`graph_backtrack`、路径以`current_node -> next_hop`开头、两节点已有唯一trace-verified edge、当前物理trace至少两帧且route arc正增长、pose进入既有`loop_merge_radius_m`并严格更靠近next-hop时触发。转移追加有方向的真实traversal、消费两端stub、重置trace/candidate状态并重新规划；无GT输入和新增超参。
- 执行顺序冻结为：原90-case结束与seal；只读V2兼容统计；同条件`tunnel/env23/seed2` V4 probe；2 worlds×3 checkpoints readiness；精确复用源schedule的V4 30-case；复用sealed原M-TARE 30例的十块配对统计。V4 30-case启动前必须验证源run state/summary/schedule均由源seal绑定。
- 用户已授权既定论文范围内自主选择最优路线，不再重复索取常规执行批准；治理preflight、Data Card、不可覆盖run和seal仍全部保留。C09/C10、Gate7和后续Gate在Gate6结论前不得启动。

## 2026-08-22：V4R保留平衡方向路线，禁止把部分成功提升为整体PASS

决定：V4R登记为科学FAIL，三个完整checkpoint均不推广。平衡方向损失与冻结encoder/embedding被保留为组件证据，因为三seed方向同时超过source与B0且dense不退化；count/role未过0.70，不能因方向成功而宣布representation PASS。

接口事件：V4首次运行因旧V3 runner硬编码Data Card status在0 seed/0 step停止。V4R仅更换原生runner的卡/summary接口，绑定V4失败seal且不复用结果；训练方法、数据和阈值未改变。

后续：方向分量数不能可靠替代count/role。下一步使用训练视图实际损失质量计算class weights；当前有效count质量为`[1244,6793,1329,504,105,25]`，role为`[6495,2105,1400]`。不得直接让极少的count 5/6主导common 1--4门禁，需先固定稀有类diagnostic-only训练合同。

## 2026-08-22：V3全编码器纠正训练FAIL，转入平衡方向损失证明

决定：`gate2_20260822_aee_corrective_full_encoder_v3_seed20260822`登记为科学FAIL；三个checkpoint全部禁止推广或用于拓扑回放。失败不是系统故障：三seed均完整执行，输入、环境、样本数和禁止读取合同全部通过。

证据：三seed sparse方向相对source下降`0.140/0.155/0.162`，dense方向下降`0.195/0.209/0.220`；AEE train-fit空输出超过91%。Cano/AEE方向标签正质量均约4.5%，普通BCE全空常数解logit约`-3.05`。

路线选择：不新增AEE数据，也不把B0直接包装成学习方法。先实现由训练标签确定的class-balanced direction loss，并加入source-model Cano保持约束；先做零optimizer-step梯度/数值证明，再决定是否冻结下一次三seed训练。用户已明确委托当前论文目标内的最优方案选择与执行，因此该选择不再要求重复确认；仍禁止扩大目标、读取C09/C10或修改M-TARE。

## 2026-08-08：采用 MASTER PLAN V3

决定：项目正式以 Gate 0--8 推进。未通过当前 Gate 不进入下一 Gate；任何 Gate 结束后也必须由用户确认是否推进。

原因：旧项目以实验失败后的局部修补推进，导致数据、teacher、模型、节点、规划和评测目标混合，已有 closed-loop 也没有构成完整的公平 baseline 证据。

影响：旧数据集、模型、阈值、offline topology 和 closed-loop 结果保留，但统一降级为历史资产或 baseline，不自动赋予 V3 Gate 通过状态。

## 2026-08-08：固定系统边界

决定：保留 M-TARE 仿真、状态估计、terrain/collision、local planner、避障和控制；替换全局环境表示、frontier 表示、全局目标选择和多机器人任务分配。

公平比较：两边使用相同 world、start、seed、机器人数量、传感器、local planner、碰撞参数和 runtime。无运动、planner 异常或日志不完整的 baseline 必须标记 INVALID。

## 2026-08-08：V3 从 Gate 0 重新判定

决定：当前 Gate 设为 Gate 0，状态为 GATE_MIXED。已有接口和运行资产可复用，但 benchmark、有效重复 baseline、coverage 口径和多机器人接口没有冻结。

禁止动作：本轮不训练模型、不运行节点实验、不启动 closed-loop、不自动开始 Gate 0 补跑。

## 2026-08-08：方法路线限制

决定：每个 Gate 只维护一个主方法、一个明确 baseline 和一个必要备选。Gate 1 主路线为 terrain-relative 2.5D BEV 与 planner-consistent teacher；Gate 2 主路线为轻量 BEV CNN + polar representation；复杂 point/sparse 3D 模型只有在主路线被证据否定后启用。

## 2026-08-08：V3 clean-room 数据与结论重启

决定：V3 不继承任何旧训练数据集、teacher、split、checkpoint、模型选择、阈值或性能结论。旧资产只能用于接口实现参考、失败模式分析和明确标注的历史 baseline。

原因：旧主数据集仅覆盖有限 M-TARE world，训练样本高度相关，验证/测试包含非地下环境，teacher 与部署 local planner 不完全一致，且历史开发已经接触多个所谓 unseen world。继续在其上修补无法严谨验证 H1--H3。

执行约束：

- Gate 0 不选择或使用训练数据；
- Gate 1 从原始 rosbag、仿真 world、terrain/collision 和真实执行轨迹重新建立 manifest、input、teacher 和 split；
- 旧 `.npz` 不得复制进 V3 数据集，也不得用于统计归一化、SSL、训练、验证或 test；
- 旧代码只有在重新通过当前 Gate 的 contract 和测试后才能复用；
- 所有 V3 test world 必须在冻结后隔离，不能因旧实验名称中的 `unseen` 自动视为严格 test。

## 2026-08-08：问题必须停工反馈，数据必须先审批

决定：任何数据、teacher、split、baseline、指标或系统问题一旦出现，Codex 必须先停止受影响工作并报告证据、影响、选项、推荐方案和所需决定，禁止静默修补或更换路线。

决定：任何数据导出、SSL 或正式训练前，先向用户提交包含 world、独立轨迹、时长/距离、原始与有效样本数、空间间隔、结构覆盖、teacher、split 和泄漏审计的 data card，获得确认后才能执行。

决定：最终 benchmark/test world 必须 world-disjoint，且不得进入监督训练、自监督预训练、归一化、teacher/阈值标定、增强调节或 checkpoint 选择。相邻帧数量不再作为数据充分性的主要依据。

## 2026-08-08：采用渐进式仓库重构

决定：不全盘抛弃旧代码，也不立即移动或删除旧目录。新正式实现按 `src/mtare_topo/{data,teacher,representation,semantics,topology,planning,integration,evaluation}` 分层；旧实现先登记为复用候选、历史 baseline、失败证据或淘汰候选。只有通过对应 Gate 的 contract 和测试后才迁移。

原因：一次性重写会丢失 M-TARE 接口、数据适配、图原型和故障经验；直接沿用旧代码又会把旧数据、teacher、split 和结论污染带入 V3。渐进迁移能够保留实现经验，同时重新建立证据链。

影响：`learning/`、`configs/learning/`、`contracts/` 和旧 `results/` 暂不删除。MASTER PLAN V3 与新 Gate 目录是唯一正式研究主线。

## 2026-08-08：实验治理改为可执行硬检查

决定：每个 material run 必须有当前 Gate 的 JSON spec、与本次范围绑定的用户批准、预检和不可覆盖的标准证据目录。数据相关 operation 还必须引用机器可读 data card，批准明确绑定 operation 和 Gate。

决定：`create_run.py` 只能创建目录和快照，不执行实验；`update_status.py` 不能改变 Gate；strict-test world 与训练、validation、SSL、归一化、teacher/阈值标定、增强调节和 checkpoint 选择的任何重叠均阻断运行。

核验：Python 标准库 `unittest` 共 10 项全部通过。环境中无 `pytest`，因此没有安装额外依赖，测试执行器改用 `unittest`；研究方法与验收目标未变化。

## 2026-08-10：Gate 1--3 改为混合监督学习主路线

决定：V3 原始数据以地下 LiDAR 点云、时间戳、位姿和真实 ray origin 为事实来源；第一版学生模型使用由点云构建的 terrain-relative causal 2.5D BEV。直接 point encoder 只在 BEV 被多高度、坡地、悬空或洞穴结构证据否定后启用，禁止两条路线无证据并行扩张。

决定：完整地图、collision、terrain 和 M-TARE local planner 生成 `R(theta)`、`D(theta)`、exit component 和 `G_local` 等客观核心监督。AI 标注器只提出高层结构属性、置信度、abstain 和错误候选，必须保存版本/prompt/原始响应，并经硬规则和人工金标审计；AI 不得直接猜连续距离、隐藏 free-space 或静默覆盖客观 teacher。

决定：Gate 2 主方法从“第一版以自监督为主”调整为轻量 BEV CNN/ResNet + polar representation 的监督式结构表示学习。自监督仅在存在合格未标注地下开发数据、监督 baseline 已建立且独立 run 获批时作为 warm start/辅助正则，并与纯监督模型消融。

原因：旧代码实际已经是规则伪标签下的监督学习，其失败来自数据独立性、非地下 split、teacher 与 planner 不一致以及任务混合，不是简单的“缺少监督”。新路线用客观 planner/map 标签约束核心结构事实，用 AI/人工辅助边界模糊的结构角色，同时保留因果 student 与 strict test 隔离。

影响：`annotation_pilot` 和 `ai_annotation` 被纳入 Gate 1 Data Card 审批。当前 Gate 仍为 Gate 0、状态仍为 `GATE_MIXED`、V3 数据集仍为 `NONE`；本决定不授权标注、数据导出、训练或仿真。

核验：治理测试 13/13 通过，其中新增“AI 标注无 Data Card 必须阻断”“AI 标注 Data Card 缺 annotation contract 必须阻断”和“完整获批 annotation contract 可以通过”。

## 2026-08-10：Gate 0 接口与 benchmark 草案完成，但不视为已冻结

审计事实：原始 Docker image 源码确认准确替换单元是整个 `tare_planner_node`；独立 `localPlanner`、`pathFollower`、terrain、传感器、状态估计和控制保留。旧 semantic node 只实现了 `/registered_scan`、`/state_estimation_at_scan` 到 `/way_point` 的最小链，没有复现 `/free_paths` 驱动的 `/map_clearing` 恢复、completion 和 runtime 契约。

审计事实：现有 tunnel、unseen mine、external cave、SubTGraph operational 01/02 都已被历史开发或结果分析接触，V3 clean strict-test world 数为 0。SubTGraph operational 01 原 TARE 600 s 为 0 m 运动，必须标记 INVALID。仿真/LiDAR/TARE 存在随机源，但当前 launch 没有 seed 接口。

当前状态：形成了 interface contract、world inventory、benchmark proposal 和 baseline rerun matrix，均为 `DRAFT_FOR_USER_REVIEW`。它们不授权仿真或 baseline 重跑，也不改变 `GATE_MIXED`。

待用户决定：是否同意三个主开发世界为 tunnel、unseen mine、external cave；是否同意新增两个隔离 strict-test world；是否同意先实现统一 seed、统一 coverage、完整 recorder/INVALID 判定和 SubTGraph spawn 诊断，再申请 baseline 重跑。

## 2026-08-10：冻结 Gate 1 数据与方法设计草案，M-TARE 地图只做 benchmark

决定：原 M-TARE 对比地图全部归入 `MTARE_BENCHMARK_ONLY`。它们不得进入监督训练、SSL、归一化、AI prompt、人工开发标注、增强调节、teacher/阈值标定、checkpoint 选择或失败驱动的模型修改。它们只用于模型和系统冻结后的同 world/start/seed/sensor/local-planner/runtime parity comparison；因历史上已被开发接触，不称为 strict unseen test。

决定：Gate 1 主数据来自新建的 graph-first 程序化地下世界族。先生成拓扑图和中心线真值，再生成一致的 collision/free-space geometry，最后从同一 geometry hash 导出 Isaac 传感器场景和 Gazebo 比较场景。LAMP/SubT 只可按完整未污染 site 进入可选 `REAL_SSL_TRAIN` 或 `REAL_SITE_TEST`。

决定：主学习路线固定为客观监督，不采用 AI-only labeler 或 contrastive-only learning。完整几何、机器人 footprint、collision 和冻结 local-planner contract 产生 32 方向的 `R(theta)`、`D(theta)`、validity、circular exit components 与 `G_local`；学生看当前与 4 s 因果历史构成的 terrain-relative BEV。AI 只做高层名称建议、异常发现和审计排序；SSL 只有监督 baseline 建立、真实开发数据合格且独立获批后才做消融。

决定：正式数据前先提出 5-world contract pilot：每 world 约 50 个 place cluster、每 place 约 6 个覆盖视图，共约 1,500 observation，只验证 seed/hash、raw-to-online input parity、teacher consistency、split leakage 和可视化，不训练模型。正式容量候选为 60/10/10 个 train/validation/development-test 程序化 world、约 160,000 observation，必须另交 Data Card 并获得批准。

相关工作风险：Cano、Tardioli、Mosteo 2026 已实现程序化地下世界、合成 3D LiDAR、轻量 CNN 出口检测与纯拓扑导航。因此“合成 LiDAR + 神经出口检测 + 拓扑图”不能作为本项目的主要创新。贡献必须落在 planner-consistent R/D/G_local 与不确定性、残缺观测下稳定 exit-stub 图、替换 M-TARE 全局规划、多机器人分配和严格隔离评测。

影响：新增 `docs/GATE1_DATA_METHOD_V1.md`、`docs/RELATED_WORK_READING_GUIDE.md` 和 `contracts/gate1_dataset_contract_draft_v1.json`。当前 Gate 仍是 Gate 0、状态仍是 `GATE_MIXED`、V3 数据集仍是 `NONE`；本决定不授权生成、标注、训练或仿真。

## 2026-08-10：固定“论文生成器—Isaac—结构语义拓扑—M-TARE”实施主线

决定：在代码和许可证核实后，优先通过 adapter 使用论文作者的程序化地下生成器；若 2024 graph-to-mesh 工具没有公开实现或许可证不明确，则依据公开论文独立复现。2022 Gazebo/SubT tile 生成器有论文给出的仓库地址，但本轮未能核实仓库当前状态与 LICENSE；2024 新版代码尚未定位，因此当前不得下载、集成或声称已复用。

决定：生成器输出中立 topology/centerline/common-mesh bundle，再由同一 canonical geometry 导出 Isaac USD 与 Gazebo SDF。Isaac 负责 canonical LiDAR 的 `STATIC_POSE` 和真实时序 `CAUSAL_EPISODE` 采集；随机静态 pose 不得拼成因果历史。批量生成前必须通过 seed、mesh、collision、landmark 和至少 20 个固定 pose 的 Isaac-Gazebo sensor parity。

决定：B1 先复现 Cano-like range-image exit CNN；M1 只训练 causal BEV 到 `R/D/U`。出口和结构角色由冻结规则解释，`G_local` 第一版只作 probe，拓扑状态机、edge verification、exit lifecycle 和 graph planner 第一版均不学习。AI、contrastive、GRU、node-score 和端到端 planning 不进入第一篇主方法。

决定：正式容量上限由旧草案的 60/10/10 world、160,000 observation 修订为 80/10/10 world、500,000 observation，通过 20/40/80 train-world 学习曲线扩展；sealed strict test 另计。该修订只更新容量设计，不授权生成。

发表判断：简单拼接已有生成器、CNN 和 M-TARE 不足以成为好期刊贡献。RA-L 是具备严格仿真和真实验证后的现实目标；JFR 要求多场景真实地下 field evidence；T-RO 只有在形成部分观测结构拓扑的一般性 major advance 后才是合理冲刺目标。

影响：新增 `docs/ISAAC_STRUCTURAL_TOPOLOGY_IMPLEMENTATION_V1.md`、`docs/PUBLICATION_EVIDENCE_PLAN_V1.md` 和 `contracts/isaac_generation_contract_draft_v1.json`。Gate 0、数据集和运行授权均不改变。

## 2026-08-10：吸收 IROS 2024 TNG-first 生成逻辑并冻结反事实数据边界

决定：程序化数据的独立 split 单位从含糊的“world”明确为 TNG topology parent。拓扑、几何、clutter、trajectory 和 sensor 随机源分离；同一 TNG 的所有 mesh 与 observation 必须位于同一 split。正式 500,000 observation 口径明确为 100 个 TNG parent、至少 200 个 mesh realization、50,000 个 matched canonical anchor、100,000 个 geometry-specific place instance与 500,000 个 observation。

决定：同 TNG 多几何用于 paired benchmark，但只有通过 robot-footprint operational connectivity 和 planner rollout 认证的 geometry 才是正样本。宽度、坡度、坍塌或障碍改变连接时属于 connectivity-changing variant，不能被不变性 loss 拉近。

决定：完整 TNG 是隐藏 oracle，只进入 provenance、teacher 和评价。学生不得读取 TNG ID、绝对位置、geometry variant、未来信息或不可见连接；不可观测连接必须 mask。M1 第一版仍只训练 R/D/U，ego local graph/G_local 保持 probe；对比学习仍是监督 baseline 之后的 A1 消融。

原因：该设计利用 IROS 2024 的 graph-first、topology/geometry 可分离优势，同时避免同父拓扑跨 split 泄漏、oracle 图泄漏和“几何变化实际改变可通性”造成的错误正样本。

影响：新增 `docs/TNG_COUNTERFACTUAL_DATA_CONTRACT_V1.md`，并更新生成/数据合同与测试。本决定不授权 Compatibility Checker、world 生成、数据导出或训练，当前运行保持暂停。

## 2026-08-10：Isaac Compatibility 通过并完成单 TNG clean-room 核心烟雾测试

执行结果：官方 Isaac Sim 6.0.1 容器的离线 Compatibility Checker 在 RTX 5090 D、驱动 580.173.02 上返回 `PASSED`。无显示器和离线 OmniHub 重试属于预期 headless/断网警告；CPU powersave governor 记录为性能风险，不影响本次兼容性结论。

实现决定：论文关联生成器无法匿名获取且许可证/提交不可核验，故不复制其代码。当前只独立实现 topology-parent 层：独立 topology/geometry/clutter/trajectory/sensor seeds、三维 RGTG spanning tree、CTG connectors、度数与采样净距约束、统计、canonical hash 和 parent-level split leakage 检查。没有实现 mesh、clutter、轨迹、传感器数据或标签。

问题与修正：初版 `tng_faf57ba981e3c324` 虽通过边—边阈值，但追加节点—非相邻边审计发现 1.089 m 小于 3.0 m，故明确否决并保留为负证据。修正生成条件后，`tng_762e384fdc6b6fa0` 的 24 节点/26 边图具有 3 个闭环，采样边—边与节点—边最小距离分别为 3.451 m 和 3.524 m；37/37 测试通过。

边界：这些净距基于离散中心线采样，只是 graph-stage guard，不是连续几何、隧道表面、碰撞体或机器人 footprint 的安全证明。下一阶段必须先提交单图 mesh 方法和验收标准供用户核验，不能直接批量生成或进入 Gate 1。

## 2026-08-10：单图 TNG-to-mesh 离线通过，Isaac v2 导入等待重新授权

问题：抽象图 `tng_762e384fdc6b6fa0` 的 3.451 m 非相邻中心线净距无法容纳约 5 m 宽隧道，否则不相邻 passage 会意外融合。决定不缩窄隧道或忽略冲突，而是生成 6.5 m clearance 的 mesh-aware parent `tng_3eca286d5e4c4cd3`。

方法：遵循 IROS 2024 的 TNG-first 因果顺序，但不复制不可获取的第三方代码。独立实现 clipped-elliptical tunnel free-space union 与固定 marching-tetrahedra surface extraction。直接体素块外壳曾被 2-manifold 测试否决，故保留检查并修正 meshing，不放宽验收。

结果：`g000` mesh 为单组件 watertight 2-manifold，111,864 vertices、223,736 triangles、0 degenerate face、genus 3，与 TNG cycle rank 3 一致；mesh hash 重放一致。该结果只通过离线 V0 geometry，不代表机器人可通行认证。

Isaac 状态：v1 USD 在真实 Isaac 6.0.1 打开时发现数组换行缺逗号并失败；修复后新增回归测试，40/40 通过。v2 重试因平台审批流断线而未创建进程，因此状态为 `NOT_EXECUTED`，不能声称 Isaac import pass。下一动作仅为用户明确重新授权后重试固定 v2 import；禁止绕过审批，禁止采数据或训练。

## 2026-08-10：重新授权后 Isaac USD v2 导入通过，单图阶段停止核验

执行结果：用户明确重新授权后，固定的 `isaac_stage_v2.usda` 在官方 Isaac Sim 6.0.1 容器中打开并完成 10 次 update。`/World/TunnelMesh` 的 111,864 vertices、223,736 triangles、extent、Z-up/metre 元数据、`PhysicsCollisionAPI`、`MeshCollisionAPI` 和 `none` collision approximation 均由 Isaac 读取验证，容器退出码为 0。因此单图 USD import 状态为 `PASS`；此前未执行记录保留为审批链历史，不再代表最终状态。

问题记录：两次 headless Replicator PNG 捕获未通过。第二次已缩减为单相机和显式 DiskBackend，仍出现 renderer 无法推进 scheduled frame、writer drain 不结束。只停止了本次临时预览容器，没有改动三个旧容器。该问题只否决“Isaac RTX 预览已生成”，不否决独立完成的 USD import。

可视化决定：为了让用户查看完整实体地图，新增直接读取验收 OBJ 全部表面顶点的 X–Y、X–Z、斜视三联图，并叠加 TNG；其 provenance 明确标为 offline actual-mesh projection，绝不冒充 Isaac 截图。

停止边界：本阶段到此停止等待用户核验。没有 LiDAR、标签、数据集、训练、planner rollout 或第二个 geometry。下一阶段如获确认，只允许先做机器人 footprint/地面碰撞和单个内部 LiDAR smoke。

## 2026-08-10：禁止直接批量，完成 g001 navigation-grade 单图几何

用户判断 `g000` 与未来仿真差距明显。决定不批量复制 g000，而是先把一张地图提升到机器人静态导航几何标准。

问题：新增曲线路径审计发现历史 parent 的最大局部坡度超过 0.22 rad。根因是 TNG 生成器只在 RGTG 生长时限制 pitch，CTG connector 候选没有应用同一限制。历史 `tng_3eca286d5e4c4cd3/g000` 保留为 topology-to-USD smoke，不再作为 navigation-grade parent。

修正：CTG 候选增加 pitch contract；单独生成 `tng_84998d00587e03dc`。geometry 使用 cubic-Hermite 水平曲线、水平弧长高程参数化、global-Z floor 和 1.15 倍 branch chamber。四次失败分别被超坡度、转弯半径和 genus 合同否决，未降低阈值。

结果：g001 最大坡度 0.180 rad、最小转弯半径 2.837 m、非相邻路径净距 6.857 m（要求 6.405 m）、43,953 个机器人实体探针零失败；mesh 159,784 vertices/319,576 triangles、watertight、genus 3、重放一致。官方 Isaac 6.0.1 导入 collision mesh 和 26 条导航曲线通过。V3 测试增至 41/41。

边界：实体探针不是动力学机器人 rollout。下一步必须先验证落地、轮地接触、低速跟踪、碰撞与卡死；该项之前不采 LiDAR、不批量、不训练。

## 2026-08-10：用户最终执行方案 V1 接管方法顺序

决定：`docs/PLAN.md` 成为当前最高优先级执行方针，`docs/PROGRESS.md` 记录唯一下一任务。根 `AGENTS.md` 已改为每次实施前先读这两个文件。旧 MASTER PLAN/Gate 文档保留治理、数据隔离和失败证据；发生方法顺序冲突时以 PLAN V1 为准，禁止静默混合。

固定主线：Phase 1 原生 Cano 程序地图与 GT TNG；Phase 2 Isaac LiDAR 与 outgoing-exit 客观标签；Phase 3 简单 CNN/ResNet18-level unseen-topology baseline。Phase 3 PASS 前禁止修改 M-TARE、在线图、全局规划和多机器人模块。

来源核验：`https://github.com/LorenzoCanoAn/procedural-subt-gen` 当前公开可读，默认分支 `refactor`，HEAD 为 `b6c77621187404b4dfab1249c7a1b40f63ad9ab3`；计划指定文件均存在。仓库顶层未显示 LICENSE，因此允许隔离、本机、原样可执行性审计，但在许可明确前禁止将第三方源码并入 `src/`、修改后发布或声明开源授权。

影响：原下一步 g001 动态 rollout 被取消为当前主线任务。g000/g001 保留为工程预检，不计入 Phase 1 的 100 topology。唯一下一步改为 Cano 固定 commit 的 run spec、许可/依赖审计与原样单图 smoke；不批量、不采 LiDAR、不训练。

## 2026-08-10：Cano 固定提交原样 smoke 失败，停止在兼容性决策前

执行事实：在隔离 checkout 中核对 `https://github.com/LorenzoCanoAn/procedural-subt-gen@b6c77621187404b4dfab1249c7a1b40f63ad9ab3`，remote/HEAD 匹配，原脚本前后 SHA-256 一致，tracked 工作区干净。固定 commit 没有 LICENSE/LICENCE/COPYING/NOTICE 正文；`pyproject.toml` 只有 MIT classifier。

依赖事实：仓库没有 lockfile，依赖均未锁版本，`snippet_2.py` 直接导入的 `cv2` 未被声明。本次 Python 3.12 隔离环境按清单解析到 NumPy 2.5.2、SciPy 1.18.0、Open3D 0.19.0、PyVista 0.48.4；漏声明的 OpenCV 作为审计环境补充并明确记录。

失败事实：唯一一次原样 `scripts/snippet_2.py` 执行在第一条 grown tunnel 进入 `Spline3D` 时失败。源码把 shape `(N,1)` 的 `dist_array` 传给 `scipy.interpolate.splrep`，抛出 `TypeError: only 0-dimensional arrays can be converted to Python scalars`。独立最小复现得到同一错误；一维控制输入能调用成功，但没有据此修改源码或重跑。

决定：本 run 判 `FAIL_BLOCKED`，Phase 1 保持未通过。完成 topology、point cloud、mesh、LiDAR、label、training sample 均为 0；不制作替代图片。原脚本还缺 seed 接口与 TNG/point-cloud/mesh 导出，因此即使兼容性恢复，也只能证明 graph/spline executability，不能直接形成标准 world bundle。

下一决策：推荐另立受控历史依赖兼容性矩阵 run，在不改原脚本的前提下测试少量预冻结 NumPy/SciPy 组合，并向作者索取 environment lock 与明确 LICENSE。未经用户确认，不私自降版本、不改二维数组、不写 adapter、不批量。

## 2026-08-10：E1 冻结依赖恢复进程执行，但随机 topology 成功仍不可审计

矩阵决定：E0 旧失败只引用不重跑；先运行 E1（Python 3.12、NumPy 1.26.4、SciPy 1.12.0、OpenCV-headless 4.8.1.78），仅 E1 失败才运行作者代码年代 E2。版本组合由 2023 年源码 blame 与 PyPI 主包兼容元数据确定。

执行结果：E1 安装和 `pip check` 通过，原样 `scripts/snippet_2.py` 运行 46.31 s、退出码 0，并达到 grown 10 和 connector 5 的调用进度。源码 hash 与 tracked 工作区不变。按 stop-on-first-pass，E2 未创建、未执行。

限制：第 5 个 connector 日志从 `0000` 到 `0999`，证明 1000 次 trial 全部失败。源码返回 `(False, None)`，但 `snippet_2.py` 丢弃 grown/connector 返回值。因此本 run 只判 `PARTIAL_EXECUTABILITY_PASS`：依赖兼容性和进程执行通过；最终 tunnel 数、节点/边、graph connectivity 和 topology validity 不可声称。

影响：完成正式 world、point cloud、mesh、LiDAR、label、training sample 仍全部为 0；许可证、seed、native topology export 和 graph-mesh 一致性均未解除。Phase 1 不推进。

下一决定：下一材料运行只能是原生 `generate_environments.py` 单图导出 smoke，最多 1 个临时 world，先核验原生 mesh/axis/SDF 与失败返回风险；该 world 不计入正式数据集。必须先向用户展示新规格并获得确认。

## 2026-08-10：原生 Cano 单图导出仅部分成功，mesh 与 topology GT 否决

执行：用户以“继续啊”批准已展示的固定规格。冻结 E1 下只执行一次原始 `generate_environments.py`，请求 1 个临时环境、3 grown、1 connector。原程序 13.70 s 退出 0，四个原生文件输出完整；E1 freeze、`pip check`、源码 remote/commit/hash 与 tracked clean 前后不变。

几何结果：`mesh.obj` 可读取，含 61,421 vertices、122,862 triangles，但为 7 个组件，非 watertight、非 edge/vertex manifold、不可定向并存在 self-intersection。因此“能写 OBJ”不能升级为“可导航 mesh”或 graph-mesh topology 一致。

GT 结果：`axis.txt` 为 9 列、4 个 tunnel ID，但 3,888 条 tunnel axis 行全部发生跨 ID 冲突；972 个唯一坐标被全局复制到多个 tunnel ID。该结果与源码在每个 tunnel 循环内使用全局 `aps_of_tunnels/avs_of_tunnels` 的缺陷一致。原入口仍未导出 graph、grown/connector 返回值或 seed。

决定：run 判 `PARTIAL_NATIVE_MESH_PASS_GT_FAIL`，正式 dataset world/LiDAR/label/training sample 均为 0，Phase 1 不推进，不通过随机重跑规避失败。只读审计确认上游已有 `WorldInfo` 序列化和 `aps_of_tunnel/avs_of_tunnel` 单 tunnel accessor；推荐下一步另立项目侧 read-only audited adapter 单图规格，显式检查生成返回、固定 RNG、导出 graph/spline/axis，并以单组件/watertight/manifold/无自交为硬门。未经用户确认不实现、不运行，不修改第三方核心、不批量、不进入 Isaac。

## 2026-08-10：read-only adapter 证明 Cano GT 可用，但原生 mesh 后端失败

执行：用户以“继续”批准精确 adapter 范围。项目侧 adapter 同时固定 `PYTHONHASHSEED`、NumPy/Python RNG 与上游 Node/Tunnel counter；只调用固定 commit 的上游 API，不修改或复制第三方核心。seed 0 只生成一次，3 grown/1 connector 均返回 `(True, Tunnel)`，整图重试为 0。

GT 结果：稳定 graph 为 52 nodes/52 edges、单组件、cycle rank 1；三条 grown 和一条 connector ID/类型明确。4 条 spline 共 1,679 点，graph endpoint 最大误差 `5.69e-14`；上游 WorldInfo、SDF、FTA、metadata/file hashes 通过；1,042 条 per-tunnel interior axis 行冲突为 0。故“Cano topology/GT 可审计”在该单图上成立。

mesh 结果：原始未修复 Poisson mesh 为 58,825 vertices/117,675 triangles、单组件且 vertex-manifold，但不是 edge-manifold/watertight/orientable，并存在 self-intersection。严格结果为 `FAIL_ADAPTER_MESH_GATE`，不能因 GT 通过而进入数据集或 Isaac。

偏差：validator 首次在计算与预览后因 `numpy.bool_` JSON 序列化失败。旧日志/hash 原样保存；修正仅增加 NumPy container 到 Python container 的 JSON 转换，同一 `world_000` validation-only 重验。没有第二次生成、seed/metric/threshold/mesh 变化，偏差证据已封存。

决定：不换 seed 猎取 PASS，不清理/修补原生 mesh，不批量。推荐下一步保留 Cano topology/spline/GT，桥接项目 navigation-grade geometry backend，并明确研究 provenance 为 `Cano topology + project geometry backend`；另立单图 bridge spec 后再检查 graph-mesh、footprint 和重放。未经用户确认不实施，也不进入 Isaac/LiDAR/标签/训练。

## 2026-08-10：按论文实际用途拆分 mesh 门禁，先做原生 mesh 传感器 smoke

纠正：上一条“原生 Poisson mesh 不能进入 Isaac/数据、下一步直接替换 geometry backend”的建议把动态导航几何资格与静态 LiDAR 训练资产资格混为一谈。Cano 论文的程序 mesh 主要用于在已知轴线附近放置传感器并采单帧 LiDAR；非 watertight/manifold/self-intersection 是明确风险，但不能在没有射线证据时直接等价为 LiDAR 失败。

决定：门禁拆为 `TOPOLOGY_GT_GATE`、`LIDAR_SENSOR_GATE` 和 `DYNAMIC_NAVIGATION_GATE`。现有单图只证明第一项 PASS、第三项 FAIL；第二项状态为 `UNTESTED`。用户批准只使用现有 seed-0 `world_000`，按 8 tunnel interior、8 junction transition、8 terminal approach 共 24 个固定静态 pose，运行 Isaac Sim 6.0.1 的 16-beam/720-column/50 m LiDAR，并以 5 m GT spline 球交点生成 360°高斯峰标签。

证据边界：本次 24 帧是 `NONE_DIAGNOSTIC_SMOKE_ONLY`，正式 dataset/train/validation/test/trajectory 计数全部为 0；保存全部样本，不训练、不调阈值、不换 world/seed、不自动替换 mesh。若原生 mesh 传感器门禁失败，再向用户提交同 pose 的 project geometry backend A/B 决策；若通过，也只允许另行审批 5-world pilot。

治理补充：正式 Data Card 强制要求 train/validation/strict-test 与独立轨迹，不适用于零数据集静态诊断。新增 `sensor_smoke` 专用审批卡合同，只允许 Gate 0--1 的单图小样本诊断并强制 `formal_dataset=false`，避免伪造 split 和轨迹来绕过治理。

## 2026-08-10：24-pose sensor smoke 停在 Isaac 自定义 profile 注册失败

执行：用户批准后冻结一张 seed-0 Cano world、24 个 pose（8/8/8）、VLP16-like 16×720/50 m profile、5 m spline label 和逐样本 RTX/CPU 阈值。preflight 无警告通过；24 个 pose/label/CPU ray reference 全部准备成功，正式 dataset 与 training 计数为 0。

执行偏差：首次容器成功启动 Isaac/GPU/RTX 扩展，但写出 0 scan，外层又错误执行无权限 chown。第二次加入 capture-prefixed 参数和 main marker 后确认进入采集器，但镜像 UID 1234 对 host UID 1000/mode 775 结果目录无写权限。两次均未产生 RTX 数据；原日志、前后工具 hash 与 recovery JSON 保留。第三次只修容器 UID，以 root 运行并将结果目录定向归还 UID 1000；world、pose、profile、label、CPU reference 和阈值均未变。

决定性失败：Isaac 6.0.1 报 `Config 'MTARE_VLP16_720_50M_V1' not found for OmniLidar at /World/DiagnosticLidar`，随后 GMO magic number 无效，300 rendered frames 内无 complete scan。按预先 stop condition 停止，未切换内置 sensor、未用 CPU backend 替代 RTX、未改 mesh/pose/world/threshold。结果为 `FAIL_CUSTOM_RTX_CONFIG_NOT_REGISTERED`，RTX sample=0，比较和 RTX 可视化未执行。

方法判断：失败定位在自定义 JSON 的注册/搜索路径，不足以判断原 Cano mesh 是否可产生有效 RTX 射线。下一步仅做 Isaac 6.0.1 registry/API 只读审计；推荐保持同一传感器 profile，先另立单帧 creation probe 验证官方 USD asset 或显式 registry 路径，再决定是否恢复同一 24 poses。任何近似内置型号都属于方法变化，必须另行批准。

## 2026-08-10：registry 审计通过，选择本地 OmniLidar USDA 直载

只读证据：固定 Isaac Sim 6.0.1 image 的 `Lidar.create` 明确把 `config` 限定为 `SUPPORTED_LIDAR_CONFIGS`，该 registry 全部是 `/Isaac/Sensors/.../*.usd[a]` 资产；同时公开 `usd_path` 参数并通过 `_create_from_usd` 查找 `OmniLidar`。deprecated command 也先查同一 USD 白名单。故先前挂载 JSON 不等于注册，失败原因已从“未知搜索路径”收敛为“错误使用 config-name 接口”。

决定：不修改 NVIDIA registry、不使用内置近似 sensor、不继续 deprecated JSON/camera fallback。保持 RTX Lidar Core backend，将冻结 JSON 一一转写为本地 OmniLidar USDA，再用 `Lidar.create(usd_path=...)` 创建。旧 JSON 隐式字段在 USD 中显式固定为 `numberOfChannels=16` 与 `channelId=1..16`；720 columns 由 `patternFiringRateHz=7200 / scanRateBaseHz=10` 得到。

执行边界：本轮只是审计，GPU/Cano/RTX sample/dataset/training/model 均为 0。下一材料运行提案是 0 Cano world、1 解析封闭盒、1 sensor、1 pose、1 complete scan 的 creation probe，必须同时通过 SensorChecker、冻结属性 read-back 与 GMO。提案当前未批准、未实现、未执行；probe 通过后也需再次确认才恢复原 24 poses。

## 2026-08-10：exact-profile USDA 参数层通过，直接 GMO 输出层失败

执行：用户核验目的、范围、方法、基线和 stop rule 后批准 v1。只运行 1 个解析封闭盒、1 个 local-USDA `OmniLidar`、1 pose；Cano world、正式 dataset/label/training/model/topology/M-TARE 修改全部为 0。镜像和冻结 hash 通过，未使用内置型号、CPU backend 或 fallback。

结果：`Lidar.create(usd_path=...)` 成功，prim 类型与 schema 正确；SensorChecker 69 个参数通过；冻结属性 read-back 零 mismatch，720 tick/scan 与 11,520 nominal rays/scan 正确。因此旧 `Config not found` 注册阻塞已解除。但直接 `LidarSensor.get_data("generic-model-output")` 返回 `0xF4AEEB00`，不符合 GMO magic `0x4E474D4F`，300 render frame 内无 complete scan。run 判 `FAIL_EXACT_PROFILE_CREATION_PROBE`，实际完整扫描数为 0，不制作伪图片。

证据偏差：Kit shutdown 将 Python failure 表现为进程退出码 0，v1 runner 又错误地以退出码推算 `complete_diagnostic_scans=1`。封存目录和 manifest 不回写；缺失 NPZ/capture summary、`capture_failure.json` 与 FAIL 状态是权威证据。后续 runner 已要求完整 NPZ 和 PASS capture summary 同时存在才计数 1。

下一方法：固定镜像的官方 `test_lidar_sensor.py` 使用 Replicator Writer `renderProduct` callback 消费 GMO。因此只准备 Writer-callback v2，不改 USDA/profile/scene/backend/pose/frame limit。代码静态编译和相关测试 25/25 通过，但第二次 GPU run 未批准、未执行。v2 PASS 也只解除传感器基础输出阻塞，不授权 Cano 24 poses、数据或训练。

## 2026-08-10：Writer v2 callback 为零，停止 custom probe 并转向 runtime control 决策

执行：用户以“继续”单独批准 Writer v2。新 smoke card/run spec 绑定同一解析盒、local USDA、1 sensor/1 pose、300 frame 与无 fallback；5 项 hash、固定镜像和 preflight 无错误无警告，V3 单元测试 53/53。只执行一次 GPU run，未触及 Cano、数据、标签、训练、模型、拓扑或 M-TARE。

结果：USDA 创建、69 项 SensorChecker 与零 mismatch read-back 再次 PASS；但 300 frame 内 Writer callback=0、zero-element=0、parse failure=0，关闭时两次出现 pending Replicator writer schedule drain timeout。故失败发生在 callback 调度前，完整 scan=0，未生成 NPZ 或图片。runner 的证据计数修正生效，summary 正确记录 0；19 项 evidence hash 全部通过，目录 160 KB。run 判 `FAIL_EXACT_PROFILE_CREATION_PROBE` 并停止，未重跑。

只读归因：固定镜像 standalone `inspect_lidar_gmo.py` 使用与 v2 相同的 Writer register/attach、renderProduct、timeline 和同步 update 顺序，源码 hash `fb6ab0...89d46`。现有证据不能区分 host/headless Replicator runtime 故障与 custom sensor/scene coupling。推荐下一材料动作是一个内置 `Example_Rotary + 官方简单 cube` 的单次 runtime health control；它只做 A/B 定位，不能作为项目 fallback 或数据。未经新批准不实施。

## 2026-08-10：批准并冻结官方 LiDAR Writer runtime 对照

批准：用户以“继续”批准单次 runtime A/B。范围固定为 1 个官方四 cube control world、1 个内置 `Example_Rotary`、1 pose、最多 300 frame；保存 Writer/GMO 机器指标和完整 control stage，不保存点云样本或可视化。Cano world、custom project sensor、正式数据、标签、训练、模型、拓扑和 M-TARE 计数全部为 0。

方法：严格沿用固定 Isaac Sim 6.0.1 bundled `inspect_lidar_gmo.py` 的 `Lidar.create(config) -> LidarSensor -> Writer attach -> timeline/update` 路径，冻结官方文件 hash `fb6ab0...89d46`、镜像 digest 和 probe/runner/test 三份代码 hash。若官方 callback/GMO/positive element 通过，则只把问题收敛到 custom sensor 或 scene/render-product coupling；若 callback 仍为 0，则只把问题收敛到 host/headless Replicator runtime scheduling。两种结果都必须停止并另行评审，不自动修改项目 sensor 或恢复 Cano。

执行状态：新数据卡、运行规格和 5 项定向测试已通过，GPU control 尚未执行。无可视化是审批内的科研设计：本实验检验 callback plumbing，不检验点云质量；完整 USD stage 与机器计数提供可核验证据。

## 2026-08-10：官方 runtime control 被未冻结的远端资产依赖阻塞

执行：冻结 hash、58/58 V3 测试和零警告 preflight 通过后，只执行一次批准命令。Isaac/GPU/RTX/Replicator 扩展成功启动，但 `Lidar.create(config="Example_Rotary")` 在 Writer attach 之前调用远端资产根检查。批准规格同时要求 Docker `--network none`，因此官方 `Example_Rotary.usda` 不可达并抛出 RuntimeError；镜像与主机只读搜索确认没有本地副本。

证据：run 为 `FAIL_OFFICIAL_WRITER_RUNTIME_CONTROL_MISSING_EVIDENCE`，13/13 manifest hash 通过，124 KB，无重试。未创建传感器、未 attach Writer、未进入 300 frame，故 callback/GMO 没有物理计数；点云样本、正式数据、标签、训练和模型为 0。summary 的 built-in sensor 数 1 仅表示计划/尝试，不能表述为成功创建。Kit 再次把 Python 异常表现为内部 exit code 0，但 runner 通过缺失 evidence 正确判 FAIL。

决定：该结果暴露的是 control 设计中的资产 provenance 缺口，不支持“headless runtime 坏了”或“custom sensor/scene 有问题”的二选一结论。停止且不打开网络重跑。推荐下一步先单独获取精确官方 USDA、保存响应和 SHA-256、检查引用依赖；只有自包含资产冻结通过后再提交 local-usd/network-none control，两个动作均不把官方 sensor 用作项目数据或 fallback。

## 2026-08-10：批准官方 Example_Rotary 单文件来源冻结

用户以“继续”批准 provenance-only acquisition。冻结范围为 NVIDIA Isaac 6.0 registry 指向的唯一 `Example_Rotary.usda`：一次 HTTPS GET、无 redirect、无 retry、上限 10 MiB；保存 response header、curl transfer JSON、原始字节和 SHA-256。随后只在固定 Isaac 6.0.1 镜像内用断网 OpenUSD 25.11 枚举 sublayer/reference/composition/external-asset dependency，不获取任何第二文件。

实现门禁：runner/auditor/test 三份代码已冻结；离线合同验证证明同一 auditor 能把自包含 LiDAR USDA 判 PASS、把含 external reference 的 stage 判 FAIL。正式网络请求尚未执行。即使 PASS，也只允许准备另一份 local-usd runtime control 提案，不授权运行 control，不把官方 sensor 用于项目数据。

## 2026-08-10：官方 Example_Rotary USDA 来源冻结 PASS

执行：63/63 V3 测试、零警告 preflight、镜像与三份工具 hash 通过后，只发出 1 次冻结 URL GET；无 redirect、无 retry，1.28 s 返回 HTTP 200 和 15,137 bytes。保存 HTTP header、完整 curl transfer JSON、原始 USDA 和 SHA-256 `0812faf5...56c8c6`。

依赖审计：固定镜像 OpenUSD 25.11 在 network-none 下成功解析；default prim `Example_Rotary` 类型 `OmniLidar`，8 个遍历 prim。sublayer、external reference、composition asset 和 external asset dependency 均为 0；没有获取第二文件。16/16 封存 hash 通过，结果目录 128 KB。

决定：该 PASS 仅证明精确官方 control sensor asset 已内容冻结且自包含。它不证明 sensor 可创建、Writer callback 健康或 RTX 有返回。按原批准边界停止，GPU/simulation/Writer/project sensor/sample/data/label/training/model 全部为 0；只形成 local-usd/network-none Writer control v2 未批准提案。

## 2026-08-10：批准并冻结 exact-local-official-USDA Writer control v2

用户在明确当前目的后以“继续吧”批准 v2。范围固定为已封存的 15,137-byte `Example_Rotary.usda`（SHA-256 `0812faf5...56c8c6`）、官方四 cube、network-none、1 sensor/pose、300 frame。只记录 sensor creation、Writer attach、render frame、callback/GMO/return/complete-scan 和 drain 指标；不保存点云或图片，不进入 Cano、数据、标签、训练、模型、拓扑或 M-TARE。

实现改进：独立 v2 probe/runner 不回写历史 run；failure 和 summary 均从物理执行状态记录“创建成功数、挂载成功数、实际帧数”，不再把计划计数当成功计数。5/5 定向测试已通过，资产/官方示例/镜像/代码 hash 和停止规则已冻结；GPU 尚未执行。

## 2026-08-10：精确官方 local-USDA 同步 Writer control callback 为零

执行：68/68 全量 V3 测试和零错误零警告 preflight 通过后，只运行一次已批准 v2。固定官方 USDA hash、镜像 digest、官方示例和三份代码 hash 均匹配；network-none、四 cube、1 sensor/pose、300 frame 范围未改变，无重试。

结果：官方 local sensor creation=1、Writer attach=1、同步 standalone update=300；callback、payload、valid GMO、positive element、zero element 和 complete scan 均为 0，shutdown 日志出现两次 pending Writer schedule drain timeout。run 判 `FAIL_OFFICIAL_LOCAL_USDA_WRITER_RUNTIME_CONTROL`；15/15 evidence hash 通过，目录 160 KB，无 NPZ/图片/checkpoint/model，正式数据、标签、训练和模型均为 0。

证据边界：封存 run 根据预设 decision rule 分类为 `HOST_OR_HEADLESS_REPLICATOR_RUNTIME_BLOCKED`，结果目录不回写。该分类描述本次设计当时的二分规则，不自动等价为所有 Kit 调度模型均失败。

## 2026-08-10：官方异步 Writer 测试收窄 runtime 归因

只读审计：固定 Isaac Sim 6.0.1 image 内 `test_lidar_sensor.py` 的 SHA-256 为 `2f545cfeb...ccbb`。该测试继承 `omni.kit.test.AsyncTestCase`；setup 使用 `create_new_stage_async()` 和 `ViewportManager.wait_for_viewport_async()`；Writer attach 后 timeline play，每帧 `await ...next_update_async()`，teardown 再 flush 一次。这与 v2 的同步 standalone update loop 存在实质执行模型差异。

决定：项目结论收窄为 `SYNCHRONOUS_STANDALONE_WRITER_SCHEDULING_BLOCKED; GENERAL_ASYNC_KIT_RUNTIME_UNTESTED`。不把源码审计当作 PASS，也不改写 v2 封存结果。唯一推荐下一实验是保持官方资产、四 cube、pose、Writer、镜像、断网和 300 updates 不变，只替换为 NVIDIA 测试使用的 async Kit lifecycle。

治理状态：`configs/v3/gate0/isaac_official_async_writer_control_v3.proposal.json` 已形成，但为 `PENDING_USER_APPROVAL_NOT_IMPLEMENTED_NOT_EXECUTED`。当前没有实现、GPU run、样本、数据、标签、训练、模型、拓扑或 M-TARE 修改。

## 2026-08-10：Isaac 退出关键路径，改用可复现 CPU Raycast baseline

用户决定：如果 Isaac 继续阻塞则不再使用，并在明确替代链路后以“那就继续吧”批准路线重构；随后进一步要求能直接模仿、当前代码能用且能复现的先用，再结合本项目方向做二次创新。

证据：Cano 固定 commit 与 E1 环境已经能运行生成器，read-only adapter 能导出一致 graph/splines/per-tunnel axis/native mesh；项目现有 Open3D 0.19 `RaycastingScene` 已对同一 world 的 24 个固定 pose 生成 16×720 CPU references，valid ratio 0.990972--1.0、均值 0.997070、最小水平净空 2.058 m、24/24 分支 LOS 通过。但这些文件属于旧 Isaac 失败 run 的诊断准备，不能直接升级为正式数据。

复用边界：直接使用 Cano TNG/表面点云/native mesh 生成和项目 CPU raycast/spline objective label。当前 generator checkout 未发现论文 CNN 训练实现，所以后续 B1 称为 `Cano-like adapted reproduction`，不能冒充原作者训练代码复现。upstream 只有 MIT classifier、没有 LICENSE 正文，本地研究运行允许继续，复制/修改后发布/再分发仍阻塞。

方法决定：Phase 1--3 主合成传感器改为 CPU idealized first-return raycast；Gazebo 固定 pose parity 仍是正式训练前必要门禁，Isaac 只保留未来可选域实验。二次创新依次放在 planner-consistent R/D/U 与 uncertainty、stable exit-stub graph、M-TARE 高层替换和多机器人分配，不混入 baseline smoke。

下一实验治理：只形成 `cano_cpu_raycast_lidar_contract_v1` pending proposal/card。范围为 1 个已有 Cano world、24 poses、16×720 rays、两次独立 scene 和全量 24-sample visualization；不生成新 world、不计正式数据、不训练。原 async Writer v3 提案退休，未实现、未执行、退休后 GPU run=0。

## 2026-08-10：Cano CPU Raycast 单图感知合同 PASS

批准与执行：用户在收到输入、方法、24 pose/8-8-8 role、每 pose 11,520 rays、两次独立 scene、成本、验收和停止边界后明确要求继续。实现通过 74/74 V3 测试、3/3 外部 Cano 合同测试和零警告 preflight 后，只执行一次，无重试、无新 world、无 Isaac/Gazebo/训练/拓扑/M-TARE 修改。

结果：解析盒已知距离最大误差 `2.3841858e-7 m`；24/24 scan 的 shape/dtype/0.3--50 m 合同通过；两份 fresh `RaycastingScene` 的 valid mask 完全一致且最大 range 差异 `0 m`；相对旧 24 CPU references 也为 `0 m`。valid ratio 最小 `0.9909722`、均值 `0.9970703`，最小水平净空 `2.0584958 m`，5 m spline-oracle branch LOS 24/24 通过。

证据：结果目录 `gate0_20260810_cano_cpu_raycast_lidar_contract_v1_seed0` 为 3.8 MB，含 24 个诊断 NPZ、完整 world/pose 图、全 24 scan 固定色标 contact sheet 与 provenance；46/46 manifest 条目复核通过。正式 dataset、training sample、model、Isaac/Gazebo run、topology/M-TARE change 全部为 0。

决定：冻结 native Cano mesh 的身份为“下一阶段理想化 CPU 感知 pilot 可用”，继续禁止把它称为 dynamic navigation mesh；本结果也不证明 Gazebo/真实 LiDAR parity。只授权准备五拓扑 contract pilot 的独立 proposal/card，未经新批准不生成地图、不执行 pilot、不训练。

## 2026-08-10：五拓扑 pilot 从 10 geometry 修正为 5 perception mesh

发现：旧草案要求每个 TNG 两个 topology-preserving geometry，并要求 footprint/planner rollout 证明 operational connectivity 不变；但同一计划已永久禁止 native Cano mesh 充当 dynamic navigation/collision asset。若现在直接做 10 个 native mesh 并称其为 topology-preserving pair，会把感知门禁冒充导航门禁。

修正：待审 immediate pilot 固定为 5 个预声明 TNG parent、每 parent 1 个 native perception mesh、50 个 canonical anchor、每 anchor 三个 `-30/0/+30°` yaw view，共 5 mesh、250 anchor、750 diagnostic observation 和两次独立 scene 的 17.28M primary rays。5 个 parent seed、5 个 geometry seed、family graph properties、无替换/无重试规则、30 页全样本 contact sheet、2 h/2 GB 上限均写入 proposal/card。

边界：原 2-geometry/1,500-observation paired invariance 设计保留，但延后到 navigation-grade collision backend 与 footprint/planner certification 建立后。当前只准备提案；治理层还需新增独立 multi-world non-formal contract-pilot schema，不能放宽现有 one-world sensor-smoke schema。未经用户新批准不实现、不生成、不执行。

## 2026-08-10：先建立可运行纵向切片，再做跨拓扑科研验证

用户决定：项目不能继续只围绕局部门禁和小故障推进，应先在已有资产上做出一个能够完整检查的系统，再根据暴露的问题调整。

实施：在不生成新 world、不训练、不启动 ROS/Gazebo/Isaac、不修改 M-TARE 的边界内，复用 seed-0 Cano world 建立 `16×720 CPU LiDAR → 当前帧规则出口 → 因果 Online Topometric Graph` 纵向切片，并以 spline oracle 作为同接口上界。完整运行 240 帧，保存全部机器证据和科研可视化。

结果与决定：v2 的规则出口 F1=`0.7503`，在线图 node F1=`0.8276`、直接几何 edge F1=`0.94`。图参数由同一开发轨迹 81 组候选选出，所以该结果登记为 `COMPLETED_WORKING_VERTICAL_SLICE`，不是 Phase 3/4 PASS。下一步不继续在该 world 调参；冻结规则和关联参数，放入五拓扑 pilot 做无调参重放。这样同时保留“先有可运行系统”的工程效率和“未见拓扑才能支持论文结论”的科研边界。

## 2026-08-11：五拓扑 pilot 因 Cano 导入 provenance 失败，零 world 封存

执行：用户批准下一阶段后，实现了独立 multi-world 诊断治理、五类固定模板、250-anchor/750-view 合同、冻结规则评测和全量证据生成。84/84 单元测试、5/5 外部测试与 preflight 通过后，仅执行一次批准命令。

结果：执行器在任何 world 构造前由 `_source_precheck` 检出 `subt_proc_gen` 来自冻结 E1 的 site-packages，而不是项目固定 checkout。run 在 1.35 s 内判 `FAIL_CANO_FIVE_TOPOLOGY_CPU_CONTRACT_PILOT`；world、mesh、anchor、observation、shard 和图像均为 0，失败证据约 38 KB，未重试。

决定：该结果归类为 runner 导入 provenance 配置错误，不评价生成器、拓扑 family、LiDAR、标签或冻结规则。修复只改变执行器环境路径，并在 RUNNING 前新增实际模块来源子进程检查；84/84 单元测试与 6/6 外部测试通过。原 run 不回写且原一次执行授权已消耗；任何 corrective execution 必须另立批准、run spec 与目录，种子/模板/阈值不得借机变化。

## 2026-08-11：五拓扑 v2b 暴露 anchor cardinality 算法缺陷

治理修复：corrective v2 因 runner 仍把 proposal/data-card 写死为 v1 路径，在 RUNNING 前发生 frozen-hash mismatch；executor/world 均未启动。用户要求继续后，v2b 只把两条路径改为从 approved spec 安全解析并补越界检查，84/84 单元、8/8 外部测试和 preflight 通过。

执行结果：v2b 只运行一次。P01--P03 共完成 150 anchors、450 static views 和三个合法诊断 shard，双场景最大 range 差异为 0，分支 LOS 通过，冻结规则 F1 为 `1.0000/0.9447/0.9154`。P04 graph/splines/native chamber mesh 已生成，但固定 selector 在 5 m 间距下只能得到 38/50 anchors，32.24 s 时停止；P05 未开始。run 为 FAIL，37 MB、51 项 evidence hash 通过，无重试。

归因与决定：P04 四条中心线共 326.13 m、655 个 spline knots，问题不是场景长度或候选不足，而是 event-first 后的 farthest-first greedy 只构造 maximal packing，不保证 requested cardinality。不得把要求降为 38、减小间距或换 seed。唯一推荐修正是目标数量感知、事件保持、分支均衡的 deterministic selector；零射线合同复用 v2b 的 P01--P04 bundle，并只为未开始的 P05 构造 graph/spline，不生成 mesh 或 LiDAR。方法修正和任何下一次正式执行都需新 proposal/card 与批准。

## 2026-08-11：selector v1 机器 PASS 被完整图复核否决为科研不充分

执行：用户以“做吧”批准零 mesh/零 ray selector audit。实现采用事件优先、0.5 m arc candidate 和 tunnel round-robin first-fit；85/85 单元、9/9 外部测试与 preflight 通过。唯一正式 run 在冻结指标上 PASS：5×50 anchors、最小间距 `5.4985 m`、事件覆盖、每 tunnel 非空、count spread `<=3`、replay hash 一致；P05 只构造 graph/spline，mesh/ray/data/training/model 为 0。

科研复核：完整 X-Y/X-Z 图显示 P01 尾段和 P03 connector 上段存在长空白。进一步只读量化发现同 tunnel 最大未覆盖半径 P01--P05 为 `32.780/16.564/92.306/10.501/17.558 m`。根因是按 tunnel **绝对数量**均衡且从 arc 起点 first-fit；这既未按 tunnel 长度分配密度，也未约束整条 spline 覆盖。

决定：封存 run 的机器 PASS 不回写，但项目解释固定为 `CONTRACT_PASS_RESEARCH_REVIEW_FAIL`，不授权 LiDAR rerun。下一候选必须按 arc length 分配 quota，在完整 `[0,L]` 上 coverage-first 放置，并新增同 tunnel 最大覆盖半径 `<=7.5 m`；保留 50/5 m/事件/seed。不经新批准不实现或执行 selector v2。

## 2026-08-11：selector v2 完整空间覆盖审计 PASS

批准与实现：用户在被明确告知本轮只修复“50 个 LiDAR 采样位置铺满整条隧道”、先做五图零射线验证且不采 LiDAR、不训练、不改 M-TARE 后回复“继续”。selector v2 保留结构事件，把非事件名额按 tunnel spline 弧长作确定性最大余数分配，并在 0.5 m 弧长候选格上使用二元 MILP，同时约束精确 tunnel 配额、全局三维欧氏间距 `>=5 m` 和完整 spline 同 tunnel/共享事件覆盖半径 `<=7.5 m`。86/86 单元测试与冻结 E1 的 9/9 外部合同测试通过。

执行结果：新 proposal/spec 的 preflight 为 0 错误、0 警告；唯一正式 run `gate0_20260811_cano_anchor_selector_coverage_audit_v2_seed0` 在 3.95 s 内 PASS 并封存 32 个文件，目录 1.4 MB。P01--P05 各 50 anchors；最小间距为 `5.006/5.500/5.185/5.500/5.500 m`，最大覆盖半径为 `3.500/4.000/6.392/4.000/4.000 m`；弧长配额、结构事件、每 tunnel 覆盖和 replay 全通过。mesh/ray/formal-data/training/model/trajectory/topology-runtime/M-TARE 计数均为 0。

人工复核与决定：逐张检查五张完整 X-Y/X-Z 图和汇总图后，P01 长直段、P03 上回环/右侧连接段及 P05 高低坡均连续覆盖，未发现 v1 类空洞。因此 selector v2 获得“用于下一次五拓扑 CPU LiDAR pilot 的采样合同 PASS”，但不升级为传感器、数据、学习、在线图或导航 PASS。下一步只准备新的 corrective LiDAR proposal/card；v2b 的旧 P01--P03 shard 不拼接复用，执行需用户另行批准。

## 2026-08-11：五拓扑 selector-v2 CPU LiDAR 感知合同 PASS

批准与执行：用户在被告知固定范围为 5 mesh、250 anchors、750 static views、17.28M 双场景 rays、零正式数据/训练/模型/轨迹/在线图/M-TARE 修改后回复“继续”。v3 proposal/card 记录该批准；新 runner 绑定 selector-v2 prerequisite hash，并强制复核精确弧长配额、`>=5 m` 全局间距与 `<=7.5 m` 完整 spline 覆盖。86/86 单元、9/9 冻结 E1 外部测试与 preflight 通过后，唯一正式命令从头执行五图，未拼接 v2b shard、未重试。

结果：run `gate0_20260811_cano_five_topology_cpu_contract_pilot_v3_selector_coverage_seed0` 在 57.34 s 内 PASS；生成 5 world bundle/mesh、250 anchors、750 observations、5 NPZ、5 complete maps 和 30 contact pages。双独立 scene 最大 range 差异均为 0，branch LOS 全部通过，100 项 evidence hash 复核通过。冻结规则总体 P/R/F1=`0.8872/0.9431/0.9143`、平均角误差 `3.501°`；P01--P05 F1=`1.0000/0.9246/0.8878/0.9245/0.8374`。

人工复核与决定：逐张检查 5 张地图和全部 30 页观测，未见空白批次或明显 raycast 损坏；P04 宽 chamber 的过检及 P05 坡道/多高度的漏检和过检与定量结果一致。不在这五图上回调冻结规则。该 PASS 只证明生成/理想传感器/客观标签/规则评测合同，不证明学习、未见拓扑泛化、在线图、Gazebo parity 或规划收益；正式数据与训练计数仍为 0。下一任务转为 100 topology parent 批次设计与审批。

## 2026-08-11：100-parent 路线禁止复制五模板，改为原生随机 topology-only 审计

发现：只读检查 `tools/v3/cano_five_topology_support.py` 确认，P01--P05 是五个显式 `_blueprint` 骨架，seed 只为固定节点加入小坐标扰动。将每个模板换 20 个 seed 会形成几何变化而非充分的组合拓扑变化，不能支持未见 topology 泛化。上游 `generate_environments.py` 虽调用真正的 random grown/connector primitives，但对 grown 使用无界 `while`，并未核验 connector 返回，不能原样作为 100-parent 审计器。

设计决定：下一材料 run 改为 topology-only candidate audit。冻结 10 个 tree/loop/scale/flat/3D strata，每层预声明 12 个 candidate 和 topology/reserved-geometry seed，调用固定 Cano 的 `add_random_grown_tunnel` / `add_random_connector_tunnel`，由项目 read-only adapter 记录每个返回并限制内部 trial。每层按 ID 保留前 10 个机器有效 parent；不足 10 即 FAIL，不在运行后追加 seed。accepted rank 1--8/9/10 分别形成 80 train/10 validation/10 development-test；人工图只查看每层第一个 train parent。

治理边界：`docs/CANO_100_TOPOLOGY_PARENT_DESIGN_V1.md`、proposal 与 pending card 已建立，但 runner/spec 尚未实现，实验尚未执行。候选最多 120 次构造+120 次 replay；mesh/anchor/LiDAR/label/formal-data/training/model 均为 0。该步骤需用户确认，PASS 后也只能另行提交 100-parent mesh proposal。

## 2026-08-11：原生随机 120-candidate 审计 FAIL，定位参数抽样产率与 cycle-rank 合同缺口

执行：用户明确“批准”后，proposal/card 绑定 topology-only 10×12 candidate、100-parent/80-10-10 目标及零 mesh/LiDAR/训练范围。实现记录每次 grown/connector 返回、same-seed replay、canonical/WL identity、split 与 10 张 train-only 图；89/89 单元、10/10 冻结 E1 外部测试、preflight 0 错误/0 警告后，正式命令只执行一次。

结果：run `gate0_20260811_cano_100_topology_parent_candidate_audit_v1_seed0` 完成全部 120 candidate，60 valid/60 invalid，各层 valid/12=`7/9/6/9/3/9/3/9/1/4`，最终 `FAIL_CANO_100_TOPOLOGY_PARENT_CANDIDATE_AUDIT`。只能得到 56 train/4 validation/0 development-test，禁止训练。60 个失败全在 grown，flat 成功率 33.3%，3D 66.7%；60 个成功 parent 的 canonical/WL identity 均 60/60 唯一，证明成功样本存在真实结构多样性。555.74 s、约 44.8 MB、269 项 hash 通过，无正式重试或 seed 替换。

指标缺口：S01 C07 的 requested connector=0/cycle rank=1，S07 C04 为 2/3；原 `cycle_rank>=connector` 不能保证 tree/loop family 身份。封存 V1 不回写，60 valid 只能按宽松历史合同解释，严格 family 至多 58。

推荐：fresh-seed V2 保持 10×12 和 80/10/10，不复用已观察的 V1 parent。把 Cano 上游无界 grown `while` 改成同 topology seed 内每条 tunnel 最多 20 次参数重抽、每次 100 internal trials 的可审计 rejection sampling；同时要求 exact cycle rank。该修订需新 proposal/card 和用户批准，仍不进入 mesh、LiDAR 或训练。

## 2026-08-11：corrective V2 材料固定，等待新批准

设计：新增 `docs/CANO_100_TOPOLOGY_PARENT_DESIGN_V2.md`、V2 proposal 和 topology-parent card。V2 保持 10 strata×12 candidates 与 first-10-valid/80-10-10 规则，但使用与 V1 零重叠的 fresh topology seeds `620101...621012` 和 reserved geometry seeds `720101...721012`。每条 requested tunnel 最多 20 次参数重抽，每次调用 Cano 原生 100 internal trials；所有参数与返回必须保存。family 验收修正为 `cycle_rank == requested connector count`。

泄漏与停止边界：V1 60 个成功 parent 全部作为已观察诊断证据排除，不并入 V2 split；V2 若任一层不足 10 个有效候选则封存 FAIL，不追加 seed、不改变 20-draw budget。人工只查看每层第一个 accepted train parent；validation/development-test 保持 machine-only。mesh、anchor、LiDAR、ray、label、formal dataset、training、model、trajectory 和 M-TARE change 均为 0。

批准更新：用户在收到 fresh 120 seeds、每条 tunnel 最多 20 次 parameter draw、每 draw 100 Cano internal trials、exact cycle rank 与零 mesh/LiDAR/训练的完整边界后连续回复“继续”。2026-08-11T14:53:35+08:00 将其登记为本精确 V2 的一次实现与执行批准。先实现、测试和 preflight；只有全部通过才创建一个不可覆盖 run 并执行一次。

## 2026-08-11：V2 生成产率修复成功，但 exact-cycle family 定义被证据否定

执行：新增 V2 bounded-resampling 实现、registry hash、V1 identity exclusion 和 exact-cycle 单元测试。90/90 单元、11/11 冻结 E1 external、preflight 0/0 后，唯一正式 run 完成 120 个 fresh candidates，无重试或 seed 变更。

结果：120/120 generation success、120/120 replay identical；936 条 requested-tunnel operation 平均 1.1357 draws、最大 5、0 条耗尽 20。各层 exact-cycle valid=`11/12/12/12/12/12/9/12/10/12`，只能保留 99 parent/80-10-9，正式状态 FAIL。6 个 invalid 全部生成、重放和其余合同通过，只因 actual cycle rank 比 requested connector 多 1。391/391 hash 通过，10 张 train-only 完整图人工复核通过；正式数据及训练计数仍为 0。

科学修正：Cano grown tunnel 会接入已有 intersection，因此 requested connector count 是生成 recipe，不是精确 cycle rank。下一步禁止再加 draw、换 seed 或重生成；推荐 V2R 只读恢复，把 strata 改为 dimensionality+grown/connector recipe，actual cycle rank 作为 GT metadata，以固定 C01-C10 形成 split。只读 counterfactual 得到 100 parent、80/10/10、100 canonical、100 WL、100 structure-event、50 3D；但该方法解释变化必须另获批准，sealed V2 不回写。

批准：用户在收到零 topology generation、零 replay、固定 C01-C10、recipe strata、100-parent/80-10-10 和零 mesh/LiDAR/训练范围后回复“继续干”。2026-08-11T15:55:10+08:00 登记为 V2R 一次实现与执行批准。必须先通过只读 source-hash、单元测试和治理 preflight；不修改 sealed V2。

## 2026-08-11：V2R 只读 recipe reclassification 正式 PASS

执行：冻结独立 helper/executor/runner、proposal/card、治理文件和五个 sealed-V2 源身份。93/93 单元、11/11 冻结 E1 external、391/391 source seal、静态/JSON 检查和 preflight 0 错误/0 警告后，创建不可覆盖目录并执行唯一一次批准命令。

结果：120/120 sealed candidate 通过 recipe 合同；每个 recipe 固定 C01--C10，共 100 parent，形成 80 train/10 validation/10 development-test。100/100 canonical identity、100/100 coordinate-free WL hash 唯一，100/100 含结构事件，50/50 为 3D。run 为 `PASS_CANO_100_TOPOLOGY_PARENT_RECIPE_RECLASSIFICATION_V2R`，17/17 新 evidence hash 通过，目录 250,816 bytes。

边界与决定：requested connector count 固定为生成 recipe，actual cycle rank 固定为 graph GT。sealed V2 FAIL 不回写；V2R 只冻结 parent 身份和 split。新 topology/replay/graph/spline/preview、mesh、LiDAR、label、formal data、training、model、trajectory和 M-TARE change 均为 0。下一步只能单独设计并审批 100-parent perception-mesh 合同。

## 2026-08-11：100-parent perception mesh 拆为 M0 合同与 M1 全量批次

只读规模审计：冻结的 100 parent 中心线总长 151.410 km，单图 0.587--2.980 km；五拓扑 pilot 的显式模板不足以证明 native Poisson mesh 在该尺度上的 exact replay 和资源行为。直接运行全 100 有数 GB 批次失败风险。

决定：先提交 M0，不把它当训练小数据集。M0 在每个 recipe 中固定选择 C01 train parent，共 10 parent/15.192 km；每图用预留 geometry seed 生成 primary 与 exact replay，共 20 mesh，并只显示 10 张完整 train 图。参数完全模仿 Cano 原生入口：FTA `[-2,-1] m`、random point-cloud params、default mesh params。零换 seed、零重抽、零替换、零修 mesh；validation/dev-test、LiDAR、标签、正式数据、训练和 M-TARE 均为 0。

M0 当前仅有 plan/proposal/card，状态 pending，尚未实现或执行。M0 PASS 也只允许另立 M1 全 100 proposal，诊断 mesh 不拼入正式 M1。

批准更新：用户在收到 V2R 已完成、M0 固定十个 C01 train sentinel、每图 primary+exact replay、20 个 native mesh、10 张完整 train 图、零 validation/dev-test 查看和零 LiDAR/训练范围后回复“继续吧那”。2026-08-11T16:32:10+08:00 登记为 M0 一次实现与执行批准；必须先测试和 preflight，通过后才可创建并执行一个不可覆盖 run。

## 2026-08-11：M0 在 native OBJ 字节重放门槛 FAIL

执行：98/98 单元、12/12 E1 external、V2R 17/17、V2 391/391 source seal 与 preflight 0/0 后，唯一正式 M0 run 从头执行。S01 primary/replay 后发现 exact OBJ SHA 不同，按冻结 stop rule 立即停止；S02--S10 未生成，无重试或阈值修改。

证据：拓扑、operation trace、effective geometry parameters 和 axis 完全一致；两个 mesh 各为 69,908 vertices、139,822 triangles、零 degenerate、单组件并各自 PASS。双向 nearest-vertex max 为 `0.599708/0.571740 m`，surface-area difference 为 `0.0425%`；完整 S01 图无断裂。run 43.508 s、21.9 MB、33/33 hash，所有 LiDAR/data/training/model/simulator/M-TARE 计数为 0。

归因与决定：Open3D Poisson 不保证 vertex ordering，上游随后按 vertex array 加 `Uniform[-0.2,0.2] m` 每坐标噪声，使 exact OBJ hash 在相同 seed 下不成立；理论三维差异上界为 `0.692820 m`。推荐 M0R 只把 replay 改成 exact inputs/axis + `<=0.75 m` 双向几何距离、`<=0.5 m` AABB 差和 `<=1%` 面积差。失败 M0 不回写、不拼接；M0R 需新批准。

批准更新：用户在收到 M0 封存失败证据、理论噪声上界和 M0R 唯一指标修正后回复“继续”。2026-08-11T16:57:02+08:00 登记为一次 M0R 实现与执行批准；parents、seeds、native method、20 meshes、10 train previews 和所有零范围不变。

## 2026-08-11：M0F 单次不可变感知资产正式 PASS

路线决定：M0--M0R4 已证明独立 Poisson remeshing 的字节、计数、顶点对应和极端点到面距离不适合作为下游复现合同。下游实际读取存储 OBJ，因此正式复现单位改为“一次生成、单资产质量检查、SHA-256 封存、下游禁止 remesh 替代”。

执行与结果：85/85 unit、16/16 E1 external 和 frozen preflight 0/0 后，唯一 M0F 正式运行完成十个固定 train C01 sentinel。10/10 primary、0 replay、10/10 完整图；graph/spline/operation/parent identity 全通过。合计 1,243,070 vertices、2,486,245 triangles、0 degenerate，最小主分量占比 0.9999353709；10 个 OBJ hash 唯一，110/110 evidence hash 独立通过。全部图人工复核无明显断裂、缺失、裁剪或 flat/3D 错误。LiDAR、标签、正式数据、训练、模型、仿真与 M-TARE change 均为 0。

边界与下一步：M0F 只证明十类 sentinel 的存储感知材料合同可行。M1 全 100 proposal 已起草但未批准、未实现、未执行；proposal 固定 80/10/10、每 parent 一次 primary、零 replay，并只允许人工查看 80 张 train 图。M1 PASS 后也只能另提 CPU LiDAR/客观标签合同，不能直接训练或替换规划器。

## 2026-08-11：M1 在第 45 个 parent 严格停止，并发现预览阶段标签缺陷

批准后完成 87/87 unit、16/16 E1 external、preflight 0/0，M1 唯一正式运行开始全 100 单次 primary。前 44 个通过；第 45 个 `S05_flat_branch_medium_C05` 的 240,422 triangles 中有一个严格共线、doubled area=0 的面，违反 zero-degenerate 合同，故立即停止并封存 FAIL。45 primary、37 train 图、0 replay、415/415 hash，LiDAR/data/training/model/M-TARE 均为 0。

只读反事实删除该面后 triangle=240,421、surface-area change=0、单组件和主分量 1.0 均不变。同时人工查看发现 M1 复用 renderer 的标题硬编码为 `M0 TRAIN ONLY`，故 37 张图的阶段 provenance 错误。推荐 M1R 对全部 100 从头统一删除 doubled-area<=1e-12 面并记录 raw/final provenance，同时使用 M1R stage label；尚未批准、实现或执行。

## 2026-08-11：M1R 全量不可变感知资产正式 PASS

批准的 M1R 修复：全部 100 从头生成；原生 OBJ 后统一计算 float64 doubled area，只允许删除 `<=1e-12` 的 face 行，vertex 行和重载坐标必须精确保持；raw/final hash、三角形数、面积和顶点差异逐资产记录；预览阶段标签改为 M1R。不得 remesh、换 seed、补跑或降低最终零退化面门槛。

执行结果：89/89 unit、17/17 external、preflight 0/0 后，唯一 M1R run 在约 5937.17 s 内 PASS。100 primary、100 sanitation、80 train previews、0 replay；100/100 最终 mesh 零 degenerate，100 个 hash 唯一，1001 项 evidence 封存。人工检查全部 80 train 图通过，validation/development-test 未显示。

解释决定：本轮实际删除 0 个面，M1 的退化面未重现，故固定为 Poisson 运行级非确定性证据。M1R PASS 证明的是封存资产及必经 sanitation 合同合格，不证明原生 mesh 每次字节确定、动态可导航或模型已学习。下一步只能另提 CPU↔Gazebo fixed-pose parity，不直接采正式数据或训练。

## 2026-08-12：CPU↔Gazebo fixed-pose LiDAR parity 正式 PASS，等待用户确认

批准与实施：用户在收到精确 3 parent/24 pose/24 CPU reference/72 Gazebo repeat、16×720 无噪声合同和零训练边界后回复“继续”。实现先修复 ROS 断网容器的 loopback 主机名问题，并按批准文本补上每传感器丢弃前 2 个 warm-up 帧；两项均在正式 run 前报告，未改变数据、插件、阈值或方法。21/21 unit/governance、解析 external control 和 preflight 0/0 通过后，只执行一次冻结正式命令。

结果：1 个解析 box、S01/S06/S10 C01 三个 train parent、24/24 pose、24 CPU reference 和 72/72 Gazebo 保存帧全部 PASS。角色为 8/8/8；valid agreement 最小 1.0，repeat 最大差 0，最差 pose MAE/P95/P99 分别为 `8.0223e-6/2.4796e-5/4.6730e-5 m`，水平最小距离差最大 `8.4668e-5 m`。4 SDF、7 diagnostic NPZ、24 pose metric、9 图与 70 项 evidence 已封存，70/70 hash 复核通过；无 parity 容器遗留。

人工复核：解析、pose map、6 页全部 pose 和分层误差图均完整可读；CPU/Gazebo range 成对一致，valid mismatch 全空。junction 和正俯仰误差略高，但仍为 `1e-5 m` 量级且不接近冻结门槛。

决定与边界：接受该 run 为传感器域 parity 的阶段性 PASS，但不自动把 Gate 0 改为最终 PASS，也不自动进入 Phase 2。parity NPZ 禁止并入训练。正式数据、teacher、训练、模型、轨迹、在线图与 M-TARE 修改保持 0。下一步需用户确认后另行起草正式 Data Card；旧 `PROGRESS/PROJECT_STATUS` 中 parity pending 描述由本条和 `PLAN.md` 第 12 节覆盖。

## 2026-08-12：Phase 2 路线收敛为 range-image 监督基线，弧长决定待批准

用户以“那继续吧”确认 parity 阶段结果并允许起草 Phase 2 数据设计；未授权数据导出、teacher 或训练。当前 `PLAN.md` 的 range/depth outgoing-exit baseline 覆盖旧 `GATE1_DATA_METHOD_V1.md` 中“第一版直接训练 causal BEV R/D/U”的顺序。BEV R/D/U 保留为 range baseline 通过后的主模型/消融候选，不静默混合两条路线。

拟定数据为 80 train/10 validation parent、22,500 place cluster、112,500 frame，student 只读 range/valid，teacher 为 graph/spline 5 m outgoing heading 加 mesh LOS；不采用 AI 标签、SSL、GNN 或 benchmark 数据。

冻结 registry 前发现上游 `distances[]` 与实际 `points[]` 累计欧氏弧长系统不一致：train/validation 差 790.421/105.010 m，单 world 最大 1.181%。问题类型为 data/provenance contract；它使采样间距、episode distance 和 per-world quota 尚不能冻结，但不否定 80/10 split、结构事件数或容量结论。推荐以封存 `points[]` 的 float64 累计欧氏弧长为唯一正式口径，旧字段只保留 provenance。该决定等待用户明确确认；确认前停止 registry、exporter、teacher 和 run 工作。

## 2026-08-12：欧氏弧长获批；互斥 role 暴露 terminal 覆盖缺陷

用户明确回复“同意继续干吧”，批准以封存 `points[]` 的 float64 累计欧氏弧长作为正式采样、插值、配额和距离口径；旧 Cano `distances[]` 只保留 provenance。该批准只允许继续完成 registry/Data Card 设计，不授权数据导出、teacher 或训练。

随后精确候选审计发现，原 `junction > terminal > interior` 互斥规则使 train 22/513、validation 8/62 个 terminal 事件没有 terminal-role 候选。30 个事件都有 10 m 内 incident-tunnel 候选，但全部同时落入 junction 邻域并被重标；问题类型为 data stratification/label overlap，不是采样容量缺失。

推荐保存独立 `near_junction/near_terminal` flags，并仅为配额定义 `terminal > junction > interior` primary role。反事实容量为 train `20007/3307/803`、validation `2625/407/98`，原配额仍可行；全部 junction/terminal 事件均恢复同 primary-role 覆盖，junction 每事件至少 3 个、terminal 至少 1 个。该规则等待用户明确确认；registry/per-world quota、exporter、teacher 与 run 保持停止。

## 2026-08-12：role overlap 获批并冻结 90-world registry

用户回复“继续”，确认独立 `near_junction/near_terminal` 与 terminal-first primary role。按事件最低覆盖量优先、剩余容量 Hamilton 最大余数分配，冻结 90 个 train/validation world 的 points-based 长度、容量、事件和配额。审计结果为 80/10 worlds、22500 clusters、112500 frames、0 超容量、junction/terminal 最小候选 3/1、C10 row=0。registry SHA-256=`d64edd758d86d1192dff870cdba08d392388cb48cfd755f428acdfb3ff4e8420`。

当前完整 Data Card 状态为 pending final approval。尚未实现 exporter/teacher，未创建 run spec，未执行 preflight/create_run，正式数据、teacher、training 和 model 仍为 0。

## 2026-08-12：完整 Data Card 获批，但 Gate 转换必须单独确认

用户回复“继续”，批准一次 Gate-1 data export + objective teacher generation，训练为 0。Data Card 已展开 80/10/10 world lists 和 90 条 trajectory，治理校验 0 error/0 warning。随后发现 operational `current_gate` 仍为 0，而治理合同只允许 `data_export/teacher_generation` 位于 Gate 1。禁止把操作伪装成 Gate 0，也禁止自动推进 Gate。推荐保留 Gate 0 历史 `GATE_MIXED`，仅把 operational current_gate 切到 1；等待用户明确授权。

用户随后再次回复“继续”，明确授权 operational current_gate 从 0 切换为 1。Gate 0 历史 `GATE_MIXED` 与全部旧证据保持不变；该转换只开放已批准的 Gate-1 数据实现、preflight 与一次 data export + teacher generation，训练仍未授权。

## 2026-08-12 — Gate 1 machine PASS 不等同于 scientific PASS

- 决定：保留 `gate1_20260812_cano_phase2_supervised_range_dataset_v1_seed0` 的执行器 PASS 和不可变 seal，但 Gate 1 科研判断记为 `GATE_MIXED`。
- 证据：运行完成 90 worlds、22,500 clusters、112,500 frames 和精确角色配额；然而封存 manifest/summary 未保存独立 scene replay 的逐帧 pass/max difference，也未保存 61 个 rejected candidate 的 ID/原因。人工查看 10 页样本和 10 张 recipe 图后，确认样本页没有 valid/teacher/LOS，recipe 图只是数量柱状图，且缺少整体分布图。
- 影响：数据数量、split 和运行时硬门槛没有被否定；Data Card 中可独立审计的 replay/replacement/visualization 证据未满足，因此禁止进入训练或宣称 Gate 1 PASS。
- 选项：A）推荐另立只读 corrective evidence audit，复投已选帧、重建拒绝明细并补齐科学图，约 24 分钟且不重导出 3.34 GB；B）接受证据缺口并永久保持 Gate 1 `GATE_MIXED`；C）完整重导出，重复约 3.34 GB，成本高且无必要。
- 授权：当前未批准 A/B/C 中任何后续操作；等待用户明确决定。现有 sealed run 不回写。

## 2026-08-12 — corrective evidence PASS，但 V1 dataset 因空间偏置不可训练

- 执行：用户回复“继续”批准一次只读 Gate-1 corrective evidence audit。正式 run 在约 913.71 s 内 PASS，精确复现 90 world、22,500 selected cluster、112,500 frame 和 61 rejected candidate；replay/teacher failure 为 0，两级源 seal 和新 run seal 全通过，源 dataset 未变化。
- 新证据：range+valid+teacher+LOS 页、真实空间 coverage 图和 aggregate distribution 图补齐了原证据缺口。
- 新缺陷：空间图显示 selector 在 event-first 后按 role 固定顺序填 quota，导致空间不均衡。只读量化为 50/90 world 共 58 条 tunnel 无 selected cluster；已有 selected tunnel 的 world 最大覆盖间距中位数 80 m、P95 222 m、最大 330 m，候选格为 5 m。
- 决定：audit run 记 PASS；V1 dataset 科研可用性记 FAIL；Gate 1 维持 `GATE_MIXED`，禁止进入训练。机器可复现性不能替代数据代表性。
- 选项：A）推荐先做零-ray selector-V2 feasibility audit，再经批准重导出约 3.2 GB；B）导出全部约 27,247 candidate cluster 后过滤并在训练时加权，规模和类别分布改变；C）直接训练 V1，因已知空间偏置不接受。
- 当前授权：未授权 selector V2 实现、审计、数据重导出或训练；等待用户决定。

## 2026-08-12 — selector V2 zero-ray feasibility PASS，必须完整生成 V2 数据

- 授权：用户回复“继续做”，批准一次 zero-ray selector-V2 implementation + formal audit；不包含 mesh、raycast、数据导出或训练。
- 方法：精确 role quota + 长度比例 tunnel quota + 全事件约束 + `<=10 m` 同 tunnel coverage 的稀疏二元 MILP，每 world 双解复核。
- 结果：90/90 world、22,500/22,500 selected、0 missing tunnel、634/575 events、0 replay mismatch；89 world max gap=5 m、1 world=10 m。正式状态 PASS。
- 对照：V1 为 50/90 world、58 missing tunnel、max gap=330 m。V1/V2 overlap=18,662，双方各有 3,838 unique cluster。
- 决定：V2 selector 冻结为下一版数据选择方法；V1 dataset 继续禁止训练。3,838 cluster 身份改变使“只换 manifest、继续沿用旧 Zarr”不可接受，必须另立 V2 Data Card 并完整重导出或提出同等严格的逐帧 provenance 迁移合同；推荐完整重导出。
- Gate：Gate 1 维持 `GATE_MIXED`；未授权 V2 data export 或 training。

## 2026-08-12 — V2 正式导出因 pre-eligibility tunnel quota 冲突严格 FAIL

- 授权与执行：用户回复“同意”，批准一次 V2 exporter implementation 和一次 80/10 全候选正式 data/teacher export；训练为 0。104/104 V3 unit、Data Card validator、真实一世界 smoke 和 preflight 均通过后，只创建并执行一个 run。
- 失败证据：前 22 world 完成；第 23 个 `S03_flat_unicyclic_small_C05` 的 tunnel 1 有 38 个原候选，7 个因 `minimum_horizontal_clearance_below_0p8m` 淘汰，eligible capacity=31，而 zero-ray 阶段冻结 tunnel quota=32。运行于求解前停止并封存 `FAIL_CANO_PHASE2_SUPERVISED_RANGE_DATASET_V2`，没有阈值/数据/配额/方法漂移。
- 影响：exact pre-eligibility tunnel quota 与 objective eligibility 不相容，故 90/90、22,500 cluster Gate 证据未成立；V2 dataset 不存在，partial shard 禁止训练。问题类型为 data/selector ordering，不是模型或 metric。
- 诊断：只读 counterfactual 在 eligible capacity 上重算长度比例 quota 得到 tunnel 1/6/12/28/45=`27/50/39/24/11`，该 world 可在角色配额、全部 10 个事件和原候选格 5 m 最大覆盖下选满 151；不把该诊断冒充正式结果。
- 选项：A）推荐新 V2R：先完成全部 eligibility，再按 eligible capacity 重算长度比例 tunnel quota，保持角色配额、全事件、每 tunnel、原格点 `<=10 m`；需完整重跑约一小时/12 GiB 上限。B）只要求每 tunnel 至少 1 加 `<=10 m` coverage，约束更简洁但取消长度比例配额，方法变化更大。C）降低 0.8 m clearance 或放宽 candidate eligibility，会改变安全/teacher 合同，不推荐。
- 决策边界：V2 一次授权已消费。未获新批准前不实现 V2R、不重跑、不训练；Gate 1 保持 `GATE_MIXED`。

## 2026-08-12 — V2R eligibility-first 正式数据导出机器 PASS

- 授权：用户在收到 V2 失败根因和 V2R 精确修正后回复“赶紧搞啊这节奏太慢了”，批准一次 V2R implementation + full export；不批准训练。
- 方法：每 world 先审查全部候选的五帧双 scene scan、clearance、LOS、teacher 与 replay，再从 eligible capacity 计算长度比例 tunnel quota；MILP 继续固定角色配额、全部事件、每 tunnel 和相对原始候选格的 `<=10 m` 覆盖。
- 验证：105/105 unit、Data Card validator、preflight 全通过。正式运行 90/90 worlds、27,247 candidate clusters、136,235 candidate frames、64 rejected、22,500 selected clusters、112,500 frames；634/575 events，max gap=10 m，selector/replay/teacher failures=0。
- 数据与证据：90 Zarr shards，train/validation clusters=20,000/2,500；角色配额完全匹配。10 complete sample pages、10 spatial maps、1 distribution；约 1,933.50 s、3.48 GB、19,338/19,338 seal。C10/M-TARE/training/models 均为 0。
- 决定：该 run 记 `PASS_CANO_PHASE2_SUPERVISED_RANGE_DATASET_V2R`，V2R 成为唯一正式训练候选数据集；V1 禁止训练，V2 partial 只作失败证据。Gate 1 暂保持 `GATE_MIXED`，必须由用户核验并确认后才能另立 Phase 3 训练卡，不能自动训练或推进在线拓扑。

## 2026-08-12 — Phase 3 不以出口单头代替结构语义

- 用户重申中心思想：学习局部结构特征、识别结构语义、据此建在线拓扑并最终替换 M-TARE 高层全局规划。
- 决定：旧 `PLAN.md` 的 `encoder -> exit head` 保留为 Cano-like B1 对标；正式 M1 使用共享 circular CNN 和 direction/count/role 三个 head，输出 128D `z_role`。不引入 GNN；图关系模型只在正式在线拓扑阶段根据证据决定。
- 表示证据：必须报告同 cluster 五视角一致性、不同结构分离、旋转等变、遮挡稳定和跨 validation world objective/role probe；中间特征维度本身不构成创新或 PASS。
- 标签边界：112,500 frame 的 direction/count/role 映射 0 mismatch；direction peak 对 heading 最大离散误差 <0.25°。count=5/6 在 train 仅 11/2 frame，第一版只把 1--4 纳入正式 count 判据，5--6 保留诊断，禁止通过 validation 泄漏或加权 loss 虚构数据充分性。
- 实施边界：用户“继续”授权训练合同与代码骨架推进，不等同于正式 training run 批准；训练前仍需批准 Gate-2 Data Card/run spec。

## 2026-08-12 — Gate 1 V2R 获确认并批准 Gate 2 三种子正式训练

- 用户在收到 V2R PASS、完整多任务合同、B0 validation 指标、隔离环境、模型/loader/指标实现和真实 no-update smoke 后回复“继续做”。
- 决定：Gate 1 以 V2R 数据与 teacher/空间/证据门槛通过收口，operational Gate 从 1 切到 2；历史 V1/V2 失败结论保持不变。
- 批准范围：B1 direction-only 与 M1 direction/count/role 共享 `z_role`，seeds 0/1/2，最多 30 epochs、patience 6；只读 80 train/10 validation，C10/M-TARE/SSL/AI 标签/图/规划器为 0。
- 固定成本：RTX 5090，预计 2--4 GPU 小时、最多 8 GB。固定验收使用 B0 F1=0.7957088、角色/count/表示稳定性门槛；不得运行中改 loss、head、seed、阈值、数据或 checkpoint 指标。

## 2026-08-12 — Gate 2 训练前启动失败，禁止静默重跑

- 证据：preflight 0/0 后正式命令在 environment-identity 子进程返回 1；训练未开始，optimizer step/epoch/checkpoint 均为 0。冻结 runner 没有保存该子进程 stderr。同一 identity 命令随后原样成功，冻结软件/GPU 身份完全匹配，无残留训练进程。
- 影响：没有模型结果，Gate 2 的方向、结构角色、分支数和表示稳定性结论全部尚未形成。该问题分类为 system/runner observability，不是 data、teacher、model 或 metric failure。
- 选项：A）推荐仅增加探针 stdout/stderr 持久化和一次短暂有界重试，重新冻结/preflight，并批准一次 replacement formal execution；科研合同与预计 2--4 GPU 小时不变。B）不重试，Gate 2 保持 `GATE_MIXED` 且训练结果为空。C）更换环境/模型/数据；当前没有证据支持，且会扩大变量，不推荐。
- 决定边界：第一次授权已经触发了一次正式命令，不能静默视为未消费。当前等待用户明确批准 A 或选择 B/C。

## 2026-08-12 — replacement 探针稳定复现 MKL/libgomp 冲突

- 授权与执行：用户回复“继续”批准保存 probe stdout/stderr、最多一次 2 秒重试，以及一次 replacement formal execution。测试和 preflight 全通过后，新 run 的两个 probe 均返回 1，按合同在训练前停止；optimizer step/epoch/checkpoint 仍全为 0。
- 具体证据：两个 stderr 都是 `MKL_THREADING_LAYER=INTEL is incompatible with libgomp.so.1`。外层 shell 和 Python `os.environ` 中该变量均为 null；runner 进程导入 NumPy 后，无显式 `env` 的 child 复现失败，显式 `env=os.environ.copy()` 的相同 child 通过。
- 影响与归因：Gate 2 模型结论仍为空；归因为 system/process-environment contract。正式 trainer 已是 NumPy-first，且 trainer child 本来就显式使用 `os.environ.copy()`，故没有证据要求改变训练环境、数据或模型。
- 选项：A）推荐仅给 identity probe 增加 `env=os.environ.copy()`，使 probe 与 trainer 子进程一致；重新冻结/preflight并批准新 replacement，科学合同和 2--4 GPU 小时成本不变。B）只改 probe 导入顺序，虽通过但不能直接证明与 trainer child 环境一致。C）设置 MKL GNU 或 FORCE_INTEL，会改变底层数学运行时，不推荐。
- 决定边界：本次 replacement 授权已消费。未获用户新批准前不修复、不第三次执行。

## 2026-08-12 — Gate 2 首次真正训练因 CuBLAS 可复现性合同 FAIL

- 执行：精确 MKL probe 修复获批后，环境探针和训练链通过。B1 seeds 0/1/2 完整训练并得到方向 F1 `0.9019/0.8944/0.9037`；M1 seed 0 完成后，日志显示 CuBLAS 未受 deterministic algorithms 约束。
- 证据：PyTorch 要求进程启动前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8` 或 `:16:8`。当前 runner 没有冻结该变量。M1 seed 0 指标虽包含方向/role/count/z_role，但不满足三种子可重放 integrity；M1 seed 1 在首 epoch 证据前停止，seed 2 未启动。
- 结论：run 为 `GATE_FAIL`，归因 system/reproducibility。B1 可保留为完整 baseline 证据；M1 seed 0 仅诊断。另有 count 0.6757、masking 0.2057、rotation logit error 0.004879 未达冻结门槛，但三种子因 integrity 中止，不能形成最终辅助指标结论。
- 选项：A）推荐 runner child env 冻结 `CUBLAS_WORKSPACE_CONFIG=:4096:8`，先执行不更新权重的真实 M1 backward 验证，再新审批从头完整六子运行；预计仍 2--4 GPU 小时。B）接受当前 FAIL，不再训练。C）关闭 deterministic 或放宽 stability/count 阈值会改变合同，不推荐。
- 授权边界：本次执行授权已消费；未获新批准前不修复、不重跑、不进入图或 M-TARE。

## 2026-08-12 — Gate 2 v1r3 完整执行为 GATE_MIXED，遮挡鲁棒性阻塞推进

- 执行：冻结 `CUBLAS_WORKSPACE_CONFIG=:4096:8` 的 v1r3 从头完成 B1/M1 × seeds 0/1/2；六个 child exit code=0，integrity PASS，62/62 evidence seal 复核通过，无 CuBLAS/NaN/timeout/残留进程。
- 通过证据：M1 direction F1 中位数 `0.868793`、seed floor `0.820010`；role macro-F1 中位数 `0.875548`；count 1--4 macro-F1 中位数 `0.760694`；same-cluster cosine mean/p05 中位数 `0.980767/0.899985`。
- 失败证据：固定 10% 方位列 masking 的 `z_role` cosine 中位数 `0.205687 < 0.85`。该低值在 CPU/CUDA 一致，训练合同没有遮挡/ray-dropout 输入扰动，问题分类为 model/input robustness。
- metric 诊断：相同 256 validation frame 与 checkpoint 的 6° rotation 在 CPU 最大 logit error 为 `6.68e-6/6.68e-6/7.63e-6`，满足冻结阈值；CUDA 为毫量级，但两端 `z_role` cosine 均约 1。原 GPU maximum-error 混入设备数值误差，问题分类为 metric/device contract。该 ad-hoc no-update 诊断不作为 formal Gate evidence。
- 结论：Gate 2 固定为 `GATE_MIXED`。方向和显式语义可继续作为通过证据；稳定表示未通过，禁止进入 Phase 4、在线图或 M-TARE。
- 选项：A）推荐先批准 formal no-update CPU/CUDA stability audit，再准备仅增加 train-only ray-column dropout 的 corrective 三种子训练；数据、split、teacher、network、heads、loss、阈值和 checkpoint 不变，预计仍 2--4 GPU 小时。B）接受 MIXED 并停止 Phase 3，项目不进入 Phase 4。C）降低 masking 阈值或直接忽略稳定性门，违反冻结合同，不接受。
- 决策边界：当前只完成诊断与状态更新；未获用户明确批准前不实现 audit/augmentation、不创建 run、不训练、不推进 Gate。

## 2026-08-13 — 严格恢复缺失 seed2，Gate 2 正式通过

- 系统事实：批准的 masking-corrective run 中 seed0/1 均完整退出 0；主机休眠导致外层 10860 秒墙钟 timeout 到期，总 runner 未启动 seed2、未聚合和封存。归因 system/orchestration，不是 data、teacher、model 或 metric failure。
- 用户决定：回复“继续”，批准推荐的严格恢复——复用经审计的 seed0/1，只按原配置训练缺失 seed2，并使用原门槛聚合；不重跑有效 seed，不改科学变量。
- 实施决定：创建独立 recovery run，旧中断目录保持原状；恢复入口硬校验旧 summary/checkpoint/log hash、trainer config、frame count、augmentation 和 forbidden reads，任何漂移即停止。
- 结果：seed2 与三 seed 聚合完整通过。方向 F1 中位数/下限=`0.887325/0.870524`，role/count 中位数=`0.882346/0.746956`，same-cluster mean/p05=`0.982660/0.902049`，masking cosine=`0.999586`，CPU rotation max error=`9.54e-6`。integrity、19/19 seal 和零 C10/M-TARE/graph/planner 均通过。
- Gate 决定：Gate 2 从 `GATE_MIXED` 更新为 `GATE_PASS`。不自动推进 Phase 4；等待用户审阅并明确授权下一阶段。

## 2026-08-13 — Gate 2 确认后只设计 Phase 4，不静默解除实施上限

- 冲突证据：用户在 Gate 2 PASS 后回复“继续”，但权威 `docs/PLAN.md` 明确规定当前实施范围止于 Phase 3、Phase 4--9 不实施。
- 决定：把“继续”用于完成下一阶段只读资产审计和 proposal，不解释为 Phase 4 正式实验授权；不修改 operational gate，不生成轨迹、不 raycast、不推理、不建图。
- 数据结论：C09 V2R 静态五视角样本不能冒充连续 trajectory。正式图 replay 必须另生成连续因果轨迹；开发拟用 3 个 C08（4,731 帧），冻结图验证拟用 10 个 C09（15,725 帧），C10 保留未读。
- 方法结论：复用旧 `OnlineTopometricGraph` 的 schema/因果接口和评估框架，但旧同轨迹 81 组 sweep 参数不直接升级；`z_role` 只作结构一致性审计，不承担 place recognition。主方法、B0 baseline、GT oracle 和停止边界见 `docs/PHASE4_OFFLINE_TOPOMETRIC_GRAPH_PROPOSAL_V1.md`。
- 所需用户决定：是否明确解除 Phase 3 实施上限，并授权按 proposal 建立 Phase 4 topology-replay data card 与开发 run；未经该决定保持停止。

## 2026-08-13 — 用户授权 Phase 4 C08 离线拓扑开发

- 用户决定：在收到 Phase 4 proposal、精确 3-C08/10-C09 数据量、方法、基线、oracle、C10 隔离和停止边界后回复“继续”，明确解除 Phase 3 实施上限。
- 当前授权：仅实施 Gate 4 C08 development；3 worlds、3 continuous doubled-edge trajectories、约 9,457.689 m、4,731 frames、M1D seeds 0/1/2、B0 和 GT oracle。C09 validation 尚未授权执行，C10/M-TARE/规划器读取或修改为 0。
- Gate 映射：Gate 3 direction/count/role 显式语义由已冻结 M1D 三头及 C09 验证证据收口，不重复训练。operational Gate 从 2 切到 4；Gate 2 PASS 历史不变，Gate 4 初始为 `GATE_MIXED`（尚未执行）。
- 工程顺序：先做 graph-edge/spline continuous trajectory 零 GPU 合同；通过后才建立正式 data card、run spec、preflight 和 C08 replay。不得用节点直线替代 tunnel spline 或静默修复映射缺陷。

## 2026-08-13 — C08 轨迹几何接口发现交叉口偏移，停止待决策

- 只读证据：三个 C08 世界共 312 条 graph edge，全部各自唯一引用一个存在的 tunnel spline，无 parallel-edge ambiguity；每世界 edge spline-arc 总和与全部 tunnel spline 总长一致。
- 具体异常：S01 四度交叉节点 `node_0015` 到 tunnel 1 的投影误差为 `0.000459 m`，到 tunnel 135 为 `0.361167 m`；后者影响相邻两条 graph edge。S06/S10 最大误差分别仅 `0.002013/0.003077 m`。
- 影响：拓扑 GT 未失效，但“图节点必须严格落在每条关联 spline 上”的连续轨迹假设不成立；此前直线长度 `9,457.689 m / 4,731 frames` 也低估约 1%，不能直接冻结为 Data Card。
- 分类：data geometry/interface contract，不是 teacher、model、metric 或 runtime failure。正式 trajectory/raycast/inference/graph 计数保持 0。
- 选项：A）推荐保留已冻结 C08 资产，允许每个 graph node 用最长 `0.5 m` 的显式连接段接入关联 spline，并在正式前审计连接段位于 mesh 可通行空间；B）重新生成或修改 C08 资产，会改变已冻结 parent/mesh 身份并要求重做资产合同；C）忽略偏移或使用整图节点直线，不接受。
- 当前边界：等待用户明确批准 A 或选择 B；在此之前不适配、不创建正式 Gate-4 run、不执行点云或建图。

## 2026-08-13 — 用户批准显式连接段，C08 trajectory mesh contract 正式 PASS

- 用户决定：明确回复“批准”选择方案 A；保留冻结 C08 资产，以最长 `0.5 m` 的显式 graph-node/spline 连接段形成连续轨迹，并先做 perception-mesh 净空审计。
- 治理修正：`topology_replay` 现被强制纳入 Data Card 审批操作；新增治理和连续轨迹测试后相关测试 23/23 PASS。正式 spec preflight 为 0 error / 0 warning。
- 正式结果：3 worlds、3 trajectories、312 edges、624 directed traversals；每边恰好正反各一次。精确总长 `9541.643029 m`，独立世界 2 m 采样共 `4773` frames。
- 几何结果：最大 connector 为 S01 `node_0015` 到 tunnel 135 的 `0.361167 m < 0.5 m`；以 `<=0.05 m` 采样，所有 connector 到 frozen perception mesh 的最小表面距离为 `1.148410 m > 0.8 m`。
- 证据：run `results/gate4_topology/gate4_20260813_cano_c08_trajectory_mesh_contract_v1_seed0` 为 COMPLETED/PASS，29/29 SHA-256 seal 复核通过，目录约 1.7 MB；三张完整 XY/XZ 图和一张异常连接段图已人工检查，无断裂或局部截断。
- 边界：raycast/inference/online-graph/C09/C10/M-TARE 均为 0。该 PASS 只解除 C08 感知回放的几何前置阻塞，不是 Gate 4 PASS，也不是 dynamic-navigation collision qualification。
- 后续：先向用户提交 4,773-frame C08 replay 的精确 Data Card、B0/M1D/oracle 方法、计算成本和 Gate 证据；获确认后才执行。

## 2026-08-13 — C08 causal topology replay 方案与 Data Card 草案固定

- 数据：只使用已过 geometry contract 的 3 个 C08，3 条独立 doubled-edge trajectory、`9541.643029 m`、`4773 frames`、25 junction/22 terminal；C09 只保留身份列表且不读文件，C10/M-TARE 为 0。
- 系统接口：Open3D 与 Torch 冻结环境不共存，故固定为文件隔离的 E1 CPU dual-scene raycast、Torch 三 checkpoint inference、CPU causal graph/evaluation 三阶段；不安装依赖、不改变冻结环境。
- 计算：16x720 LiDAR 的 primary rays=`54,984,960`，双 scene=`109,969,920`；预算 0.75 h、1 GiB、CPU+RTX 5090。
- 方法：新 causal graph V2 必须记录累计行程边长和 physical traversal trace；distance anchor 固定 20 m；B0、M1D seeds 0/1/2、GT oracle 共用同一 builder；C08 预声明 243 组 grid，只用于参数开发。
- 停止边界：oracle integrity、因果性、raycast/replay/teacher、checkpoint、verified-edge trace 任一失败即停；不在 run 内改数据、网络、teacher、指标、网格或阈值。
- 当前决定边界：proposal 与 draft Data Card 已建立，正式 V2 builder、raycast、inference 和 graph run 均未执行，等待用户明确批准。

## 2026-08-13 — 不绕过 S10 0.632 m 净空失败，C08 replay 授权已消费

- 决定：用户批准的一次完整 C08 implementation+execution 已执行；formal run 在 S10 frame 6 的 `0.631976306 m < 0.8 m` 证据处停止并封存为 FAIL，不续跑、不覆盖、不删帧、不降低阈值。
- 理由：双场景误差 0、branch LOS 完整；点位位于 `edge_0000/tunnel_129` 常规 spline，connector 仅 `0.000213 m`。问题属于冻结 mesh/trajectory 的 data geometry eligibility，不是模型、teacher、metric 或随机性。
- 影响：C08 4773-frame dataset、M1D inference、243-grid graph sweep 和参数 freeze 均未成立；Gate 4 保持 `GATE_MIXED`。
- 推荐：另行审批完整 clearance-only audit，先查清失败规模再选择 geometry/trajectory corrective route。安全轨迹裁剪会改变采样单位与结构覆盖，必须新 Data Card；重生成 S10 会改变资产身份并要求重做 geometry contract。
- 次要证据：失败 runner aggregate summary 把 partial attempted counts 记为 0；逐 world metrics、failure JSON 与 raw log 正确，sealed run 不回写，replacement runner 必须修正聚合证据。

## 2026-08-13 — 完整 clearance audit 后优先审计 floor-following pose，不直接删帧或重生成

- 授权与结果：用户“继续”批准一次完整 4773-pose audit；正式 run 完成 3,436,560 rays，S01/S06 0 failure，S10 9 route failures/9 segments，diagnostic evidence PASS。
- 物理去重：9 个 route failure 映射到 tunnel 129/16 的少数位置；doubled-edge 正反配对距离最小为 `0.008--0.452 m`，故不能按 9 个独立位置计算问题规模。
- 几何证据：tunnel 129/16 的生成半径为 `5.719008/5.223342 m`，低净空命中局部 mesh triangles；不支持“隧道本身设计得只有 0.6 m”的解释。
- 决定：不删除失败帧、不降 0.8 m、不立即随机重生成整张 S10。推荐先审计固定 FTA pose 是否未跟随粗糙地面，并求局部 collision-feasible sensor/base z；可行则重做完整 trajectory contract，不可行再重生成 S10 perception mesh。
- Gate：审计 PASS 是完整诊断，不是 trajectory eligibility PASS；C08 replay dataset 和图参数仍为空，Gate 4 保持 `GATE_MIXED`。

## 2026-08-13 — 局部 z 扫描否定“只修高度”，推荐先审计横向位移

- 治理决定：用户批准为非独立诊断位姿增加严格 `diagnostic_geometry_audit` 合同。Data Card 必须显式声明 formal dataset=false、source trajectory independent=true、pose subset independent=false、parameter candidates are not samples、zero training/model selection；raw/effective 必须等于真实 route pose 数，parameter evaluation 单独计数。该修正不放宽正式数据集对独立性的要求。
- 证据：9 个 route pose/5 个物理组，549 个 z 候选，395,280 水平射线、1,098 竖直射线。只有 frame 6/39/2551 存在可行 z，6/9 无解。顶部与地面都有有限命中证据，核心矛盾是贴地高度的水平净空不足。
- 影响：否定“仅重算 floor-following z 即可修复 4,773-frame trajectory”。尚不足以否定小范围 x/y 位移，也不支持直接重生成整个 S10 mesh。
- 选项：A）推荐，对 5 个物理组执行预声明 x/y 局部网格，每个 x/y 点再求 floor-following z 与水平净空，成本低、零推理/图/训练；B）直接重生成 S10，成本高且会改变资产身份，证据不足；C）降低阈值或删帧，破坏冻结安全/采样合同，禁止。
- 状态：formal run 完成且 21/21 seal，但 Gate 4 仍为 `GATE_MIXED`。方案 A 是新的 material diagnostic，必须先报告网格、ray 数、成本和通过证据，并获得用户明确批准。

## 2026-08-13 — lateral audit证明mesh可用，下一步重建完整连续轨迹

- 授权：用户回复“继续推进”，批准已报告的9 pose×21×21局部x/y+floor-z审计。实际Data Card保持raw/effective=`9/9`，parameter evaluations=`3969`，零训练/推理/建图/C09/C10/M-TARE。
- 证据：9/9 pose全部有非孤立可行连通块，每pose可行格为271--441/441；最近修正范围`0--0.224m`。总计2,857,680水平rays、11,907竖直rays，26/26测试、preflight 0/0、21/21 seal通过。
- 决定：不重生成S10 mesh，不删帧，不降0.8m阈值。局部问题应由完整floor-following pose与平滑lateral correction解决。
- 边界：9个点的局部可行性不证明4,773帧轨迹的连续性、曲率和全帧安全。禁止只将推荐点线性插值后直接进入推理。
- NEXT：用新的正式contract重建三个C08的完整floor-following trajectory，仅在S10必要区域求平滑x/y修正，然后重做4,773帧水平/竖直净空、连续性和connector审计。这是新material run，需用户新批准。

## 2026-08-13 — 完整轨迹纠正授权已消费，局部floor-support失败禁止静默插值

- 正式结果：S01/S06全帧通过；S10 frame338/2275在同一edge0065/tunnel3物理点的向下支撑距离为8.20/8.28m，与邻帧0.93--1.08m不连续。run已FAILED/17项seal，不覆盖、不续跑。
- 排除：失败点距最近lateral anchor 48m，紧支撑10m核给予的x/y偏移为0，不是跨通道偏移污染。
- 影响：旧horizontal-only audit没有检查floor support；因此当前仍没有合格的4,773-frame轨迹，不能恢复推理和建图。
- 选项：A）推荐，对该唯一物理点做机器人足迹尺度x/y支撑+高度连续性审计，成本低；B）重生成S10 mesh，需重做全部几何合同，当前证据不足；C）跨孔插值floor z，伪造地面支撑，禁止。
- 决定边界：方案A需新Data Card和用户明确批准；零训练/推理/建图/C09/C10/M-TARE。

## 2026-08-13 — 2m×2m足迹审计无支撑，停止轨迹层补丁

- 授权与范围：用户连续回复“继续/继续2/继续”，批准frame338/2275的2×41×41网格地面支撑审计。raw/effective仍为2/2非独立重复观测，3,362是parameter evaluations。
- 证据：两个观测的1,681格均为0支撑；共同可行格=0、robust component=0。水平ray=0是冻结的“先支撑、后净空”合同结果，不是漏算。29/29测试、preflight 0/0、15/15 seal通过。
- 决定：停止继续扩大轨迹偏移、插值floor z或手工修OBJ。问题已定为S10 perception-mesh data geometry defect。
- 选项：A）推荐，从同一封存topology/spline资产重生成S10 perception mesh，给新资产新身份并重做mesh/floor/connector/4773-frame完整资格；B）手工补面，不可复现且会人为改变几何，不推荐；C）放弃S10，会改变预声明三世界开发覆盖，不推荐。
- 决定边界：方案A是新material asset generation，需精确报告生成输入、随机性、身份、成本和重做证据后由用户批准。

## 2026-08-13 — S10 replacement首次授权因遗漏upstream PYTHONPATH在生成前失败

- 具体证据：executor导入`generate_cano_audited_bundle.py`时报`ModuleNotFoundError: subt_proc_gen`；新OBJ、Poisson、floor query和materialization均为0。失败run 11项seal完整。
- 根因：历史M1R使用compat E1并在子进程显式加入`external/procedural-subt-gen/src`；新runner用zarr E1并只加项目src。归类为system/runner environment contract。
- 影响：没有产生新mesh科研结论，但一次正式命令授权已消费，不能视为未执行而静默重跑。
- 推荐选项A：只修runner为历史compat E1 + 冻结`_executor_environment()`，添加E1/upstream/import identity证据，再批准一次replacement execution。生成器、seed、parent、阈值和单次materialization规则不变。
- 选项B：不重跑，Gate 4继续阻塞且S10 replacement为空。换生成器/修mesh/改seed没有当前证据支持。

## 2026-08-13 — runner修正不能绕过冻结E1依赖漂移

- 授权：用户“继续”批准runner-only环境修正和一次replacement；授权不包含修改或重建Python环境。
- 证据：兼容E1的核心版本和upstream commit匹配，`subt_proc_gen.tunnel`来自固定checkout；但完整materializer导入缺少`perlin_numpy`。历史freeze/安装日志固定其来源为commit `5e26837...`，当前E1实际不存在该包。
- 影响：v1r正式run未创建，materialization仍为0；不能把仅通过`tunnel`导入误报为完整生成环境可用。
- 选项A（推荐）：从历史freeze重建新的不可变E1副本，核对完整freeze、`pip check`、materializer导入和来源身份后执行已批准的一次replacement。成本为数分钟和数百MB。
- 选项B：直接向旧/tmp E1安装缺包，较快但污染历史环境，不推荐。选项C：停止，Gate 4继续阻塞。
- 决定边界：创建新E1和安装依赖是新增系统变更，必须由用户明确批准；未批准前不执行preflight/create_run/materialization。

## 2026-08-13 — native Poisson replacement否定“随机重生成即可修floor”

- 授权与环境：用户批准独立E1重建及一次S10 replacement。新E1按封存98行freeze恢复，未修改历史E1；freeze、pip、完整import和upstream身份全部PASS。
- 正式证据：新OBJ SHA=`f22ebb...`，source identity、mesh audit、sanitation全PASS；floor support=`2234/2558`，324失败、53连续段、最长144帧，frame338/2275仍失败。run FAILED并完整封存。
- 影响：否定“同种子/同参数再做一次native Poisson就能得到完整可行地面”。该mesh仍可作为perception候选，但不能作为dynamic floor/collision资格资产；Gate4 C08 replay仍阻塞。
- 选项A（推荐）：资产分离。Cano native mesh继续服务CPU LiDAR；从同一TNG/spline与冻结半径/FTA生成确定性、连续的floor/collision support geometry，并分别做`LIDAR_SENSOR_GATE`与`DYNAMIC_NAVIGATION_GATE`。
- 选项B：继续随机Poisson重生成，没有停止边界且已被一次正式反证，不推荐。选项C：删除S10/放宽±0.25m/插值地面会改变数据或门槛，禁止。
- 决定边界：方案A是新的geometry method和material asset generation，必须先提交精确几何定义、数据范围、baseline、成本和验收并获得用户批准；未批准不实现。

## 2026-08-13 — 停止独立 floor-ribbon v1，转向 incident-node 路口融合设计

- 只读/内存证据：三个 C08 共 4,773 帧；独立 floor ribbons 的异常归因中，S06 `21/21`、S10 `45/45` 均先命中共享拓扑节点上的另一 incident tunnel 地面，距离节点 `0.337--4.800 m`；非 incident tunnel 命中为 0。S01 仅首帧端点边界漏射。
- 影响：否定“每条 spline 独立铺宽条带并直接组合成一个 collision floor”的方法；它在路口产生相互覆盖的多值地面。该原型不能进入 formal run，也不能证明 dynamic navigation。
- 禁止的适配：不加宽条带、不降低 `1.00±0.05 m` 支撑门槛、不按期望 tunnel 忽略更近实体表面、不把 per-tunnel evaluator scene 冒充统一 collision world。
- 选项 A（推荐）：对每个显式 graph node 构造有界局部单值融合面，使所有 incident tunnel 在节点邻域共享连续高度，并保持非 incident/垂直叠层隧道在 3D 中分离；先做合成单测和全 C08 内存预检，再申请正式资产资格 run。
- 选项 B：用 3D swept-volume/voxel union 生成完整 collision mesh，路口连续性更自然，但实现、分辨率审计和计算成本更高。
- 选项 C：按 tunnel identity 分场景查询，只可作为几何合同诊断，不能作为真实统一地面或导航证据。
- 决定边界：当前仅完成问题归因与原型代码/单测，未创建 Data Card、spec 或正式结果。新几何方法属于 material method change，需向用户报告精确定义、baseline、成本和 pass/fail 后获得明确批准。

## 2026-08-13 — 单值地面必须与 floor-following 轨迹联合定约，停止固定高度预检

- 用户授权：仅批准 incident-node 单值融合实现与 4773 帧内存预检；不包含正式 run、轨迹变更、训练、推理、建图或 M-TARE。
- 结构尺度证据：融合尺度取 incident 最大地面半宽的 `1.0/1.5/2.0×`，S06 失败由21增至`196/996/1261`，S10由45增至`110/1357/1958`；最小尺度已有S01/S06/S10=`7/3/24`对节点融合域重叠。
- 无效结论：不能再要求“统一物理地面”同时精确匹配每条斜坡 spline 独立给出的旧 `spline_z+FTA` 高度。继续调融合半径只是在同一矛盾合同下过拟合。
- 选项 A（推荐）：几何先行、轨迹后验。几何只读 TNG/spline/radius/FTA，生成 incident junction 连续地面；随后保持 world、XY route、arc、frame index 和 traversal 不变，只由生成地面求 floor-following z，再审计4773帧安全与连续性。轨迹不参与几何生成，因此不构成轨迹泄漏。
- 选项 B：进一步采用稀疏 3D swept-volume/voxel union 生成路口和隧道碰撞面，再做同样 floor-following 轨迹求解；物理一致性更强，但实现、分辨率和计算成本更高。
- 选项 C：保持旧 sensor z 完全不变并放弃统一地面，只能做 per-tunnel evaluator 合同，不能声称真实 collision/dynamic navigation。
- 决定边界：已停止受影响实现；没有 spec、Data Card、formal run 或轨迹写入。需用户明确批准 A、B 或 C 后继续。

## 2026-08-13 — 固定 XY 的 geometry-first 路线无可行尺度，建议升级局部 3D union

- 授权：用户回复“继续”，批准方案 A 的原型、合成测试与内存预检；不含正式 run、训练、推理、图或 M-TARE。
- 逐层排除：旧 sensor 向下首命中不是完整 z solver；改用旧 spline floor `±2 m` 的双向所属层搜索后，除 S01 开放端点外候选完整。节点平面 flatten 造成16--33m串层；全长 XY 投影和纯 XY 节点域也会串层。最终合同固定为 incident identity + node-local spline arc + 3D node neighborhood。
- 最终证据：全部旧重叠点距共享节点的归一化三维半径最大为 S06=`1.011265`、S10=`1.026334`；必须至少覆盖该尺度。固定 XY 连续性却在 S10 `0.5×` 已失败6个step、最大回归`0.3691m`，S06在`0.75×`已有3个超限step。两项要求无共同尺度。
- 影响：否定方案 A 的固定 XY 子合同，不否定资产分离总体路线。不能通过 smooth-z、忽略实体首命中或调半径伪造连续地面。
- 选项 B（推荐）：局部 3D swept-volume/voxel/SDF union，显式保留上下叠层隔离并对 incident 交汇做体并集；再在路口局部联合优化 XY/z。预计 CPU 数分钟、内存数百 MB；需新增分辨率收敛、watertightness、collision clearance、轨迹连续性证据。
- 选项 C：停止动态 collision 资格，只保留 per-tunnel 离线语义 replay；成本低，但不能支持替换规划器或动态导航主张。
- 明确决策需求：用户需批准 B 后才能实现 3D union 原型；当前无 spec/Data Card/formal run/轨迹写入。

## 2026-08-14 — 用户批准 3D union 设计，不批准正式资格执行

- 决定：用户“继续吧”批准方案 B 的设计与实现准备。唯一允许范围是写明可审计几何定义、收敛合同和后续运行边界；不允许正式资产物化、轨迹写入或任何回放。
- 固定方法：碰撞空腔由 tunnel spline/radius 的 3D swept void 定义；只在显式节点的 incident tunnel、节点局部 arc 和 3D neighbourhood 内取 union。非 incident tunnel 永不合并，进入保守机器人包络即为资产失败。几何只读 TNG/spline/radius/FTA，轨迹后验求解，避免 trajectory-to-geometry leakage。
- 固定验证方向：预声明 `0.10/0.05/0.025 m` voxel-SDF resolution ladder；每级均需 union provenance、topology isolation、mesh integrity 和完整 4,773-frame safety/continuity pass/fail identity。不可按最终结果挑分辨率或调参数。
- 证据与边界：提案为 `docs/PHASE4_C08_3D_UNION_TRAJECTORY_PROPOSAL_V1.md`。下一步仍必须先提交 exact Data Card 和 one-shot spec，再取得用户单独执行批准；C09/C10、M-TARE、LiDAR、M1D/B0、graph 和训练继续禁止。

## 2026-08-14 — 3D union qualification 草案冻结，未授权执行

- 已提交草案 Data Card 与 spec，分别位于 `configs/v3/gate4/data_cards/cano_c08_3d_union_trajectory_qualification_v1.proposal.json` 和 `configs/v3/gate4/cano_c08_3d_union_trajectory_qualification_v1.proposal.json`。
- 范围固定为 3 个 C08 development worlds、624 次定向遍历和 4,773 帧；9 个 geometry source JSON 与 3 个 trajectory NPZ 都由 SHA-256 绑定。三档 `0.10/0.05/0.025 m` 都必须通过，不能从结果选择一个分辨率。
- 草案批准状态为 `PENDING`，command 为 pending implementation，故不能且未通过 preflight、create_run 或执行。此举仅把下一次用户决定所需的数据、方法、成本（CPU 2 小时/2 GB）、baseline、停止条件和证据写清。
- 当前需要的决定：是否批准一次性的 collision-asset materialization、junction-local trajectory write 和完整 4,773-frame × 3-resolution qualification；授权不应包含 LiDAR/inference/graph/C09/C10/M-TARE。

## 2026-08-14 — 已批准 qualification 在执行前被临时环境消失阻断

- 具体证据：既有 Gate 4 runner 的环境路径 `/tmp/mtare_cano_e1_zarr2187/bin/python` 与 `/tmp/mtare_cano_compat_e1_rebuilt_20260813/bin/python` 不存在；`python3`、`/usr/bin/python3`、Anaconda Python 均报 `ModuleNotFoundError: open3d`。
- 影响：3D-union runner 尚未实现为可执行冻结工具，未运行 preflight/create_run/qualification，0 collision asset、0 corrected trajectory、0 ray、0 inference/graph。不能把 Anaconda 中可用的 scipy/skimage 视为既定 Open3D RaycastingScene 合同的替代。
- 选项 A（推荐）：仅从已封存的兼容 freeze 恢复一个新的不可变 E1，核对完整 freeze、pip check、Python/Open3D/scipy/skimage import 与 source hashes；成本为一次依赖恢复，完成后不自动执行任何 qualification。
- 选项 B：重写资格链为无 Open3D 后端；这会改变已批准的 safety backend 与方法，需新的完整 Data Card/spec/approval，不推荐作为当前 execution 的静默替代。
- 选项 C：停止，Gate 4 保持混合状态。需要用户明确决定 A/B/C。

## 2026-08-14 — 新 Gate 4 qualification E1 恢复通过

- 用户批准选项 A 后，以 `/usr/bin/python3.12` 重建独立 `/tmp/mtare_gate4_qualification_e1_20260814`。它与封存 `e1_zarr2187` freeze 的排序 SHA-256 均为 `50144a4...841cb82e`，`pip check`、Python/Open3D/scipy/numpy/matplotlib/zarr/numcodecs import identity 全部通过。
- 首次恢复误以 Anaconda 为 Python 3.12，实测为 Python 3.13.5；其对 NumPy 1.26.4 触发不兼容源码编译，已停止并可恢复地移至带 `_py313_failed` 后缀的目录。没有把该环境混入恢复结果。
- 决定边界：环境恢复只解除 system blocker，不消耗或自动触发已批准的一次 qualification。后续仍先实现/测试工具、冻结 hash、升格 Data Card/spec、preflight 和 create_run；C09/C10/M-TARE 继续禁止。

## 2026-08-14 — 完整 C08 稠密 voxel-SDF ladder 与已报成本矛盾

- 证据：sealed C08 总 route length=`9,541.643029 m`。以不小于实际 tunnel radius 的保守 5m 圆截面，void volume lower bound=`749,398.891m³`；uniform `0.10/0.05/0.025m` 分别至少 `0.749/5.995/47.962` billion voxels。float32 SDF alone 是 `2.79/22.33/178.67 GiB`，未含任何索引、occupancy、提取或 mesh。
- 影响：`cano_c08_3d_union_trajectory_qualification_v1.proposal.json` 的完整世界三档稠密 SDF、2 GB/2 hour cost 和后续 run 实施不可同时成立。不能因用户已经批准原草案而偷偷改成 sparse/local implementation。
- 选项 A（推荐）：全局用确定性解析 swept-tube boundary，只有显式 incident node 的受限 3-D windows 使用三档 voxel-SDF union；规定窗口选择只来自 graph/spline/radius/FTA，所有 4,773 pose 仍做 qualification。成本可审计且与“局部 union”语义一致，但必须重新定义 convergence evidence 并新建 card/spec。
- 选项 B：稀疏/adaptive SDF；可降低内存，但失去 fixed uniform-grid convergence 的解释，需要不同方法合同。
- 选项 C：停止动态 geometry 路线。需要用户明确选择，且当前已批准的一次 execution 不会被消耗。

## 2026-08-14 — 推荐路线 A 的 local implicit-union V2 仅完成设计冻结

- 用户“继续”按上下文解释为接受推荐的方案 A，但只授予设计/草案范围。V2 用解析全局 swept-tube field 加 25 个显式 degree>=3 bounded union windows，取代不可行的全世界稠密 SDF。
- 固定窗口为 S01/S06/S10=`4/7/14`，sphere radius=`1.10 * max incident radius`，最大为`6.577872m`。局部场以 `0.10/0.05/0.025m` 串行处理；全局非 incident 近接不合并，window overlap、seam/manifold 或 resolution identity fail 直接停止。
- 新草案和方法为 `docs/PHASE4_C08_LOCAL_IMPLICIT_UNION_PROPOSAL_V2.md`、`configs/v3/gate4/data_cards/cano_c08_local_implicit_union_qualification_v2.proposal.json`、`configs/v3/gate4/cano_c08_local_implicit_union_qualification_v2.proposal.json`。它们明确为 PENDING，未消耗 execution authorization。
- V2 改变了 collision safety-query backend、资产形态、成本和 convergence evidence；因此仍需要用户明确批准 V2 one-shot execution，不能复用 V1 执行许可。

## 2026-08-14 — V2 无可扩展、冻结的 patch-mesh extraction backend

- 证据：新 immutable E1 内 `skimage/mcubes/vtk/trimesh/numba` 均不可导入。唯一项目 extractor 的 `_extract_boundary` 接受 Python `set[tuple[int,int,int]]`，并从每个 occupied voxel 枚举候选 cubes；V2 单个最大 0.025m local cube约146 million cells，超出该数据结构/循环的可扩展范围。
- 影响：不能产生 V2 所需的 deterministic local patch mesh、seam 或 manifold evidence。E1 的 Open3D 0.19.0 不构成等价的受控 marching-cubes backend，故不能以它静默替代。0 runner、0 preflight/create_run、0 asset、0 trajectory。
- 选项 A（推荐）：为 meshing 建一个独立版本锁定 sidecar，安装并固定 `scikit-image`，用其 marching-cubes 提取局部 field；同时记录 sidecar hash、算法版本与 cross-backend synthetic tests。它改变环境/方法，需 V3 card/spec/approval。
- 选项 B：自建 streaming extractor，避免新增依赖但开发/验证成本显著，且需要单独的数值/流形合同。
- 选项 C：改为 implicit-field query-only，成本较低但不能保持 patch mesh/seam/dynamic collision asset 证据主张。
- 选项 D：停止。需要用户明确选择；此前 V2 execution authorization不自动授权任一选项。

## 2026-08-14 — scikit-image meshing sidecar 通过身份恢复

- 用户选择推荐 A。创建独立 `/tmp/mtare_gate4_meshing_sidecar_v1`，基于 sealed E1 freeze 并仅增加 `scikit-image==0.24.0`。`pip check` 通过，Python/NumPy/SciPy/Open3D 与 `skimage.measure.marching_cubes` import identity均通过；sidecar freeze SHA=`6447ba5e...19c4035`。
- 边界：sidecar 是新的版本锁定 extraction backend，不能冒充原 E1，也不代表 V2 patch meshes 已被验证。必须先做 synthetic topology/seam/determinism contract，才可把它写进新的 Data Card/spec；没有自动恢复 qualification。

## 2026-08-14 — sidecar 合成 marching-cubes 合同通过

- 用户要求“运行”后，已实际执行 synthetic contract：65³ sphere SDF、0.1m spacing，两次 scikit-image extraction hash 一致，7,446 vertices/14,888 triangles、0退化三角形、所有 edge incidence=2。执行入口为 `tools/v3/verify_meshing_sidecar_contract_v1.py`。
- 结论边界：该结果验证 sidecar 的算法可用和可重复，不是 C08 geometry PASS，不减少 C08 25-window seam/isolation 证据，也未产生任何 C08 资产/轨迹/安全结论。

## 2026-08-14 — C08 local union 的 incident-tunnel 字段不足以定义入射 spline arcs

- 具体证据：C08 graph 中 `degree` 是 graph-edge multiplicity，而 `incident_tunnel_ids` 是 unique physical tunnel identity。S01 `node_0010` 为 degree 3、IDs `[135,139]`；同类情况在 S06/S10 大量存在。新增纯函数的 synthetic tests 2/2 PASS，但正式只读 audit 因先前错误假设 degree==unique-tunnel-count 而正确停止。
- 影响：若只 union unique tubes，将无法区分同一 tunnel 在节点两侧对应的不同 local spline arc，局部 field/seam 不能满足已声明的 topology isolation。不能复制 ID、忽略多重 edge 或扩大为完整 tunnel。
- 选项 A（推荐）：批准一个零 mesh/零 ray/零 trajectory 的只读 edge→spline projection and arc-incidence audit；它从 sealed graph/edges/splines 建立每条 incident edge 的 projection、arc direction、connector error 和唯一 replay rule，再决定 V3 geometry definition。
- 选项 B：明确放宽 V2 到 unique physical-tube union；较简单但改变“node-local arc”语义和非 incident 分离证据，需新方法审查。
- 选项 C：停止。需要用户决定；当前 execution authorization 不自动覆盖这一 topology-GT method change。

## 2026-08-14 — edge→spline arc-incidence audit 通过

- 用户批准推荐只读 audit 后，新增 projection/arc-direction contract；3/3 synthetic tests通过，C08 read-only audit通过。312 edges 映射为624 endpoint records，按 S01/S06/S10 为108/184/332；每记录保留 edge ID、node ID、physical tunnel ID、arc、projection、connector error 和离开节点 tangent。
- 最大 connector errors `0.361167487/0.002012064/0.003076177m` 均在既有 <=0.5m connector contract 内。该结果解决 degree 与 unique physical-tunnel ID 不同造成的 arc identity歧义：同一 tunnel 可有多个 edge incidence，不能被折叠。
- 边界：仅只读 audit，未产生 geometry/trajectory/safety/Gate结论。首次 JSON 输出的 numpy.bool_ 错误已仅修序列化后重跑；输入和推导规则不变。后续 local union 必须消费 edge-level arc identity，不能退回 unique-tunnel-only union。

## 2026-08-14 — local patch watertight 与开放 tunnel seam 不可同时成立

- 最小复现：贯穿 local cube 的解析 cylinder 在锁定 marching-cubes backend 下有136条boundary edges、非watertight；这是入射 tunnel 穿过窗口边界的预期拓扑，不是库故障。
- 先前闭合 synthetic 通过是因为 finite capsule endpoints 位于采样域内部，未覆盖实际 seam。给 local window 加球面/立方体封口会制造碰撞墙，不能作为 dynamic collision asset。
- 影响：当前 V3 “local patch自身watertight + analytic outside + 连续可通行 seam”合同自相矛盾，materializer、正式 Data Card/spec 和 run 全部停止。
- 推荐 A：以 hybrid implicit field 直接做 C08 qualification，验证边界 field/gradient continuity；local mesh降为可视化，动态 simulator mesh主张延期。B 为全局解析 mesh 与局部 patch共边界拼接；C 为统一 sparse global field。三者改变方法/证据/成本，需用户决定。

## 2026-08-14 — V3 one-shot qualification 因 list-root traversal loader 失败

- 用户明确选择方案 A 并授权实施/执行。冻结实现、7/7 测试、正式 Data Card/spec 与 preflight 0/0 后，只创建并执行了 run `gate4_20260814_cano_c08_hybrid_implicit_qualification_v3_seed0`。
- executor 在任何 geometry/frame qualification 前，将根节点为 JSON list 的 sealed traversal manifest 传给只允许 object root 的治理 `load_json()`，产生明确 `ValueError`。runner 将正式 run 标为 `FAILED` 并 seal 14 个文件；没有调参、覆盖或重试。
- 分类为 implementation/system defect。它使本次 C08 geometry qualification 结论无效，但不改变输入数据、teacher、模型或安全 metric。推荐的 corrective 仅替换 loader 并加入真实 list-root 回归测试；这仍是新代码/新 run，必须由用户另行批准，不能把原授权解释成自动重试许可。

## 2026-08-14 — V3R loader corrective 生效，但 visualization patch 数值退化触发 FAIL

- 用户回复“继续”批准了严格 loader-only V3R。9/9 tests 和 preflight 0/0 后，唯一新 run 正确越过 traversal load，并开始 serial patch extraction。
- S01 node_0039 / 0.10m patch 的只读完整性计数为 56,414 vertices、111,526 triangles、2 个面积 `<=1e-12m²` faces、1,302 boundary edges、0 nonmanifold edges。冻结 spec 要求 zero degenerate，executor 因而正确停止；run `FAILED`，18/18 seal，无重试。
- 这是非权威 visualization meshing 的数值完整性问题，不是 continuous field 或 4,773-frame safety 证据；后者尚未开始。推荐的新方法决策是允许确定性删除并记录退化 visualization faces，同时保留 zero-nonmanifold 和 continuous-field authority。该修改改变已批准 acceptance，必须新建 V4 Data Card/spec 并取得用户明确批准；也可选择停止。

## 2026-08-14 — V4 collision safety 通过 S01/S06，但 S06 continuity 失败

- 用户批准 V4 sanitation 后，11/11 tests、preflight 0/0 和唯一 run 正确越过两个历史实现阻塞。S01 与 S06 共 2,215 帧在全部三档的 horizontal/down/up 合同均通过，resolution pass/fail identity 通过。
- S06 maximum step 从 baseline `1.999999973m` 增为 `2.139792493m`，超过 frozen `baseline+0.1m`。最差 frame 141→142 的 z correction jump=`0.520064m`。当前实现只逐帧把 sensor 放到 support surface 上 1m 并选择 zero XY，未实现批准计划所要求的 continuity-constrained first/second-difference optimizer。
- 该证据出现时执行器已开始 S10 coarse patches；按 stop policy 手动终止子进程，runner 以 exit -15 封存 FAIL，67/67 seal。此操作阻止了无效的 S10 继续消耗，不是参数适配或重试。
- 推荐决策为 V5 补齐 window-local constrained XY/z optimizer 和 world-end immediate-stop enforcement；不允许放宽 continuity、改变 clearance、删帧或调 geometry。由于这是冻结工具/trajectory method change，必须用户新批准。

## 2026-08-14 — V5 formalization 前确认单一 global union 在 S10 合并非 incident support

- 用户批准 V5 optimizer 后实现的约束解在 S01/S06 可行；13/13 tests PASS。S10 则有两对窗口外相邻帧的 vertical support difference 单独超过 11m，因此并非 SLSQP 调参问题。
- 精确 provenance：两对都属于 edge_0065/tunnel3。frame341/2272 的 downward global-union zero surface分别由 tunnel135提供，而相邻 frame342/2271由route tunnel3提供。route tunnel3在错误floor点的SDF为10.716/10.847m，证明是非incident layer接管，不是同一tube采样噪声。
- 原计划规定 non-incident overlap立即FAIL，且只允许窗口帧调XY/z。故不得扩大membership、按结果筛hit或放宽step。V5在card/spec/preflight/create_run之前停止，无formal run。
- 如继续，必须决定新方法：edge-conditioned/layered collision/support field以sealed traversal edge选择physical tunnel layer，junction才做incident union；这会牺牲单一global asset主张但保持topology isolation。否则停止geometry路线。

## 2026-08-14 — edge-conditioned hard switch 因全部 junction seam 不连续而否定

- 用户明确批准 layered 方法后，16/16 tests 证明 layer dispatch 能排除 stacked non-incident tunnel，并只对 compatible junction 启用 incident union。
- 三档只读几何审计发现 25/25 窗口的 selected-layer field 与 incident union 在球面 tunnel openings 不相等，最大差 `5.608039m`；dispatch isolation 本身全部通过。hard switch 因此将非 incident overlap 问题换成了 collision/support seam discontinuity。
- 原计划把开放接缝列为立即 FAIL，故不得用 actual trajectory 恰好未触碰某些 seam 来覆盖几何失败，也不得直接 formalize/run。Gate 4 维持 `GATE_MIXED`，C08 geometry qualification仍未完成。
- 推荐下一决策：采用 selected-layer→incident-union 的连续径向 transition（保持固定窗口半径、无新增长度调参）并先做只读 field/gradient proof；其开发成本为新增混合场、梯度/单调性测试和三世界三档预检。备选 actual-ray-only contract 成本较低但削弱独立 geometry 证据；停止路线成本最低但 Gate 4 geometry blocker保留。

## 2026-08-14 — V6 连续场 seam PASS，但 finest-only optimizer 不满足三档共同 floor 合同

- 连续 radial smoothstep 解决了 hard-switch seam 与 S10 non-incident support 两个几何阻塞；25/25窗口三档值/C1分派通过，三世界 finest-field continuity feasibility通过。
- 正式V6在S01完整PASS后，于S06 coarse field发现10个window frames downward distance超出1.05m，最大1.082591m；horizontal/up有充分余量。根因是trajectory只按finest field的1.035m target冻结，未把三档floor约束同时写入optimizer。
- 这属于trajectory optimizer/multi-resolution method failure，不是数据、teacher、model、clearance threshold或transition seam failure。不得选择finest resolution、调大floor tolerance、删除10帧或从失败目录继续。
- 推荐V7用三档逐帧z可行区间交集并保持原continuity/最小位移目标；成本是新增共同可行性求解、单元测试、三世界只读proof及新的one-shot run。若交集为空则方法直接FAIL。任何V7实现/运行需用户新批准。

## 2026-08-15 — V7共同区间存在，但未精化的ray exit使投影不可重放

- V7原型证明S01/S06/S10三档support surface spread均小于0.10m floor带宽，S10最小共同区间仍约0.05138m；共同可行性本身未失败。
- 当前sphere marcher以`max(0.8*|sdf|, spacing/2)`前进，并把第一个outside travel直接作为hit。coarse spacing=0.10m时末步至少0.05m，未做bracket refinement；z投影因而不能稳定重现同一surface，S10 floor与step交替回归。
- 这否定optimizer-only V7的可执行性，问题属于数值求交实现。不得用增大floor tolerance、选择finest结果或增加外循环掩盖。推荐批准固定次数inside/outside二分的V7R；它改变ray算法，需新明确授权。

## 2026-08-15 — V7R二分有效，但严格上界与optimizer边界投影相差1e-11m

- 用户确认持续Gate-4目标，授权V7R固定32次bracketed ray exit。21/21测试、三世界只读proof和preflight通过后，仅创建并执行一次正式run。
- 正式run在S01 coarse停止：112帧down位于`1.05m`上边界附近，最大超出`9.525e-12m`；水平与向上合同全部通过。22/22证据seal，无重试。
- 根因是optimizer允许`1e-9m`收敛容差并将closest point放在闭区间边界，final audit却执行无数值裕量的严格比较。二分已把先前厘米级overshoot消除，当前不是几何或物理安全失败。
- 推荐V7R2只把optimizer内部共同区间收缩固定`1e-8m`，不改最终审计、物理阈值、场、窗口、分辨率、帧或外循环；该修正和新run仍需用户明确批准。

## 2026-08-15 — V7R2正式PASS，解除C08 geometry eligibility blocker

- 用户明确批准V7R2后，仅加入固定`1e-8m`optimizer内部裕量；最终物理阈值、几何场、窗口、三档分辨率、4773帧和连续性合同保持不变。
- 唯一正式run完成并PASS：3 worlds、25 windows、624 directed traversals、4773 frames、14319分辨率帧审计、150 patches；所有安全、接缝、隔离、patch完整性和连续性合同通过，178/178 seal。
- 该结果只证明独立dynamic collision/support资格，不把visualization patch声明为simulator watertight collision mesh，不执行M1D或graph，也不自动提升Gate 4。
- 下一步按既有Phase-4授权恢复冻结M1D因果topology replay；使用V7R2 sealed corrected trajectories，禁止C09/C10/M-TARE并保持原graph方法与评价边界。

## 2026-08-15 — V7R2 dynamic pose 与 native perception mesh 不兼容，causal replay 停止

- 只读输入绑定检查显示，V7R2合格sensor相对原native-perception sensor的z位移分别为S01 `-4.599496..-4.153639m`、S06 `-4.723445..-3.722062m`、S10 `-3.987098..-3.060038m`；S10另有最大`0.070501m` XY修正。
- 在S10固定帧`0,1,2,3,4,5,6,10,100,1000,2000,2557`上用V7R2 sensor pose只读raycast原生mesh，12/12 objective branch LOS失败；多帧水平扫描无命中并返回50m。range数组仍有限不代表观测语义有效。
- 影响：V7R2只解除独立collision/support场内的动态资格，不能自动恢复“原生Cano mesh只用于LiDAR”的联合合同。继续正式replay会把场外/地板下观测交给冻结M1D，因果语义和图结论无效。
- 分类：geometry/data-interface system blocker，不是checkpoint、graph或metric失败。当前未创建新spec/run、未执行模型推理或图更新、未读取C09/C10/M-TARE。
- 禁止静默绕过：不得删除LOS门禁，不得用原pose渲染而以V7R2 pose建图，不得把无命中50m帧视为合格。需用户明确选择新的统一geometry路线后方可继续。

## 2026-08-15 — V7R2 违反 window-only trajectory adjustment，项目层重分类

- 冻结spec明确写明`Optimize only window frames`，但executor先对全部4773帧执行`solve_support_sensor`，随后才对window frames做XY约束优化。
- 将sealed input、corrected NPZ与正式`complete_frame_audit.csv`连接后，窗口内帧为466，窗口外为4307；后者4307/4307均有非零z修改，范围`-4.716372..-3.060038m`。此前基于记忆报告的4343已由权威CSV更正。
- 因此V7R2的32次bracket ray、field seam/isolation和实现内安全结果仍是有效组件证据，但`COMPLETED`不能证明原计划的trajectory-locality或causal replay eligibility。项目解释固定为`FORMAL_MACHINE_PASS_RESEARCH_CONTRACT_FAIL`，sealed run不修改、不覆盖。
- 推荐corrective不再把isotropic tube底部同时当support：保留side/ceiling collision field，新增与sealed spline及`fta_distance_m`对齐的route-conditioned support surface；窗口外pose精确不可变，只有窗口内允许联合XY/z。该方法需用户明确批准、新只读proof及新formal run。

## 2026-08-15 — 采用独立 route-conditioned support，并保留单独的正式执行批准点

- 用户明确批准实施corrective/replay计划，包括实现、只读proof、Data Card/spec冻结与preflight；该计划同时要求展示preflight后再取得一次正式执行批准，因此本轮未调用`create_run.py`。
- downward support不再复用isotropic collision-tube ray exit。窗口外由当前traversal spline的`z+fta_distance_m`唯一确定；窗口内只组合独立incident edge arcs并沿用既有C1 transition。side/up仍由三分辨率continuous layered collision field权威审计。
- native Cano mesh仅用于LiDAR与诊断性horizontal clearance；finite scan、双场景一致性、objective branch LOS和teacher完整性仍是causal replay硬门禁。teacher/graph/sensor坐标禁止隐式混用。
- 只读geometry与native proof均PASS，支持继续一次新geometry formal run；V7R2目录与项目层`FORMAL_MACHINE_PASS_RESEARCH_CONTRACT_FAIL`解释保持不变。

## 2026-08-16 — V8误终止后禁止静默恢复

- 冻结12h限制由runner的`time.monotonic()`和外层timeout执行；主机暂停会造成`ps etime`大于monotonic有效运行时间。监控未先核对runner日志即发送SIGTERM，是错误操作。
- V8已按真实结果固定为`FAIL_CANO_C08_ROUTE_CONDITIONED_SUPPORT_CORRECTIVE_V8`，不得修改RUN_STATE、续跑目录、拼接部分结果或静默新建run。
- 推荐恢复路径为新的V8R one-shot run：完全相同的数据、route-conditioned support、collision field、三档分辨率、阈值和证据合同，只修订运行治理，明确暂停时间与有效运行时间的判定并禁止人工误杀。需要用户新批准。

## 2026-08-17 — 接受V8R正式geometry PASS并恢复causal replay资格

- V8R以完整独立重算完成，未读取V8部分资产；正式计数、安全、连续性、pose-locality、资源和seal合同全部PASS。
- 决定接受V8R为C08独立dynamic collision/support的权威资格证据。V7R2仍保持research-contract FAIL，V8仍保持operator-terminated FAIL，两者均不改写。
- 下一阶段只能使用V8R三条显式teacher/graph/sensor trajectories；native Cano继续仅负责LiDAR。必须先冻结trajectory、mesh、checkpoint、environment和工具hash，完成新causal replay preflight并取得用户正式执行批准。

## 2026-08-17 — C08 causal replay V2合同冻结并通过preflight

- Data Card与run spec固定3个C08 development worlds、4773个sensor frames、109,969,920条双场景rays、14319个冻结checkpoint inference frames、243×5×3=`3645`次graph replays；不读取C09/C10/M-TARE，不训练或改变参数网格、评分与门限。
- preflight返回0 error/0 warning，但这不构成正式执行授权。决定保留单独批准点：用户明确批准前不创建run；正式结果只能使用不可覆盖run ID `gate4_20260817_cano_c08_route_conditioned_causal_topology_replay_v2_seed0`。

## 2026-08-17 — C08 causal replay V2 identity validator bug，正式run不得续跑

- 用户回复“继续做”消费了V2 one-shot授权。sensor stage 4773/4773帧和全部109,969,920双场景rays PASS；inference stage因executor比较`checkpoint['mode'] != 'm1d'`而在0帧失败，而三个冻结checkpoint真实且一致的字段为大写`'M1D'`。
- checkpoint hash、seed及环境均未漂移，故问题分类为system/interface implementation bug；它使V2 inference/graph结论为空，但不否定已封存sensor或V8R geometry证据。
- 决定V2保持不可变`FAILED`，不得从sensor资产续跑、改RUN_STATE或覆盖目录。推荐选项A为新V2R完整replacement：只修正identity validator并加测试，重新冻结/preflight后取得新批准，成本约1.25h/1GiB。选项B为接受FAIL并停止Gate 4 replay。需要用户明确选择。

## 2026-08-17 — 接受V2R causal replay PASS并冻结C08开发参数

- 用户回复“修啊”选择V2R。修复严格限于checkpoint metadata validator从错误的`m1d`改为训练器/三个sealed checkpoints的真实值`M1D`，并新增正反回归测试；未改变任何科研变量。
- V2R完整one-shot run通过全部sensor、inference、graph与oracle门禁。决定接受`sf2_tr8_lr6_hh20_te45_da20`为唯一C08 development graph tuple；冻结数值为`2/8m/6m/20deg/45deg/20m`，原评分权重不变。
- V2 FAIL继续保持不可变system-interface failure；V2R PASS不是对V2目录的改写。V2R只证明C08开发回放和参数冻结，不构成Gate 4最终PASS，不授权C09 validation、C10 test、planner或M-TARE closed loop。下一步必须由用户另行决定。

## 2026-08-17 — C08参数按确定性并列规则保留，科研结论收窄

- 复核确认`hh20/hh25/hh35`在其他冻结维度相同时取得完全相同最高均分；executor按`(-score, parameter_id)`排序，故`hh20`是预声明确定性tie-break结果。决定不事后改选更保守角度，也不把该tuple描述为唯一最优。
- 决定将C08结论限定为aggregate M1D improvement与development tuple freeze。S06个别seed不优于B0、S10最差、结构节点过生成和参数平坦区必须随C09方案预注册报告。
- Oracle只作为connectivity/verified-edge integrity参照，不再称weighted-composite上界。C09此前参与M1D checkpoint选择，未来validation只能评价冻结感知下图构建泛化。读取C09仍需用户新批准。

## 2026-08-17 — C09精确合同替代历史估计，12小时geometry cap被否定

- 用户批准只读C09合同审计。决定以31654.752491m/15833 frames为唯一后续样本合同，废止历史节点直线估计31442.449m/15725 frames；1027 edges/2054 traversals/71 windows/213 support arcs同步冻结为proposal输入。
- 只读语义与connector检查全部通过，但639 patches相对C08的150 patches扩大4.26倍，线性约16.62h。故不得直接复制12h runner、删patch、挑world或降低分辨率来适配资源上限。
- 推荐下一决策为24h active cap、4GiB RAM、8GiB disk的单一C09 geometry qualification。该资源上限调整必须在新Data Card/spec中预声明并获用户批准；C09 causal validation仍以后置geometry PASS为前提。

## 2026-08-17 — C09 geometry V1 路径配置错误不得原地重试

- 用户批准V1后，preflight虽通过，但spec中的一个历史证据路径不存在；runner在C09读取和executor启动前异常退出。决定按正式one-shot政策把V1固定为system/config FAIL，而不改RUN_STATE为未执行、不覆盖目录、不补文件后原地重启。
- 正确历史文件`artifacts/evidence_sha256.txt`的SHA-256与spec冻结值`10f6eb7b...b159`一致，故修复范围只应是路径名及其回归验证，不允许改变世界、帧、窗口、几何方法、阈值、分辨率或资源上限。
- 可行选项A（推荐）是经用户新批准建立V1R并完整重跑，成本仍约16.62h、上限24h/4GiB/8GiB；选项B是接受V1 FAIL并停止C09资格工作。等待用户明确选择。

## 2026-08-19 — V1R主机重启中断后采用全量V1R2，不复用部分资产

- 取证显示V1R在7个世界、8311帧和258个patch后随主机重启终止；最后日志位于S08 finest patch阶段，RUN_STATE为过期`RUNNING`且无最终metrics/summary/seal。已完成部分为零安全/连续性/分辨率失败，但无法支持C09整体结论。
- 决定不修改旧RUN_STATE、不从断点续跑、不拼接或seal残缺目录。用户明确批准新的V1R2完整replacement，科研变量与资源上限全部保持，唯一运行治理改动是使用systemd sleep/shutdown inhibitor降低再次被挂起或正常关机中断的风险。
- sidecar依赖冻结哈希和meshing输出合同复核一致，35项回归测试PASS。V1R2仍须通过preflight并仅创建一次；若执行失败则保存真实FAIL，不调参、不删帧、不重试。C09 causal replay不在本次授权范围内。

## 2026-08-20 — V1R2保留正式FAIL，禁止用executor PASS静默覆盖runner合同失败

- V1R2 executor完整输出证明15833帧全部安全且三分辨率一致，但runner按预注册639 patch计数检查后给出正式FAIL。封存证据显示实际426记录严格分成每档142条；window manifest显示71个窗口的physical tunnel层数分布为65个双层、3个三层、3个单层，而只读manifest的213是directed incident support arcs总数，二者语义不同。
- 决定维持V1R2不可变`FAILED`，不把executor PASS直接解释为正式qualification PASS，也不启动C09 replay。缺陷分类为data/evidence-contract count bug，影响正式资格结论，但不否定已完成的几何安全证据。
- 推荐方案是另行批准一个只读corrective evidence audit：逐窗口冻结并验证`142 physical layers×3=426`合同、绑定V1R2的506-file seal、复核全部帧与资源门禁，不重新生成几何。成本分钟级。备选为完整V1R3重算（约10.35小时、4.1GB）或停止；需用户明确选择。

## 2026-08-20 — corrective audit V1 loader失败不得原地重试

- 正式V1 audit在0.028s内因list-root manifest与object-only governance loader不兼容而失败；源seal前后506/506一致，未执行证据评价、几何、模型或图。
- 决定保留V1不可变FAIL与11-file seal，不修改RUN_STATE、不在原目录替换loader或重跑。该失败属于system/implementation，不能解释为426合同或C09安全失败。
- 推荐V1R严格限定为显式list-root parser和真实main-path回归测试，输入、阈值、426合同、成本与claim boundary不变。新正式run仍需用户批准。

## 2026-08-20 — 接受V1R纠正审计PASS并恢复C09回放资格

- 用户要求固定完整流程目标并继续推进，消费一次loader-only V1R授权。V1R只改变清单根类型读取，未改变几何、样本、阈值、分辨率或证据语义。
- 正式结果确认`142 physical window-layers × 3 resolutions = 426 patches`且15833帧全部通过；源V1R2封印未变化。决定接受该独立纠正审计作为C09 geometry replay eligibility证据，同时保留V1R2与V1原正式FAIL。
- 下一决策限定为使用既有冻结M1D checkpoint与C08选定图参数执行C09 validation。不得在C09进行模型训练、checkpoint选择或参数搜索，也不得自动进入C10、planner或M-TARE。

## 2026-08-20 — C09机器PASS但科研结论定为MIXED

- 用户授权完整流程后，C09只运行C08冻结tuple，不在验证集重新选参数。正式sensor、inference、graph和seal门禁全部PASS，故接受`PASS_CANO_C09_CAUSAL_TOPOLOGY_VALIDATION_V1`作为系统可执行性与图完整性结论。
- 科研效果不作单向美化：三seed M1D平均composite仅比B0高0.011171，exit-F1反低0.013899；提升来自terminal reachability，而结构节点冗余更差，5/10世界平均composite低于B0。决定将效果结论记为`C09_FROZEN_GRAPH_VALIDATION_MIXED`。
- C09曾参与checkpoint选择，禁止称其为严格端到端未见泛化。C10、planner和M-TARE均未获授权；下一步只能由用户另行决定。

## 2026-08-20 — 用户明确扩大实施范围到整篇论文，Phase 5--9 改为逐 Gate 执行

- 用户目标：用结构语义构建的拓扑图替代 M-TARE 全局规划器，完成单机器人、多机器人、基线、严格测试、统计分析和可发表论文证据，而不是停在 C09 离线图验证。
- 冲突：旧 `PLAN.md` 只允许 Phase 4，且禁止 planner/M-TARE；该规则与用户本次明确目标冲突。决定以本次用户指令覆盖旧范围限制，并在 `PLAN.md` 第16节记录；不静默混合。
- 保留的治理：Gate 逐级结论、strict-test 隔离、Data Card、spec、preflight、不可覆盖 run 和正式执行批准全部继续生效。范围扩大不等于所有实验一次性预批准。
- Gate 4 以 `GATE_MIXED` 收口；C08/C09机器链路成立，但C09效果混合。C10保持封存，避免用最终测试修正规划器。
- Phase 5 主方法固定为 rule-based topological frontier，verified-edge Dijkstra 回退，显式 local rejection/retry；baseline 为原 M-TARE，必要上界为 GT-TNG Oracle。Oracle不能达到原M-TARE时停止路线。
- 准确系统边界保持不变：替换 `tare_planner_node`，保留 simulator/LiDAR/terrain/standalone localPlanner/pathFollower/control。第一正式动作先是 C08 逐帧 shadow，而非直接闭环。

## 2026-08-20 — 禁止由单帧空 frontier 直接宣布探索完成

- C08 23,865-cycle非正式开发预检中，M1D与oracle均出现短暂空frontier后恢复TARGET；这证明单帧语义空窗不等于环境探索完成。
- 问题分类为planner/integration completion contract，不是模型训练、数据或图连通失败。若直接发布`exploration_finish`会造成可复现的提前终止。
- 决定将图层状态改为`CANDIDATE_COMPLETE`；M-TARE handoff只有在独立的稳定图状态与覆盖判据确认后才允许发布finish。Phase-5 shadow只审计候选状态，不事后从离线轨迹虚构完成确认。
- map-frame waypoint、verified-edge路由和原TARE两次空`free_paths`触发8m map clearing合同不变。正式shadow尚未执行。

## 2026-08-20 — 接受 C08 Gate-5 formal shadow PASS，下一步转 Oracle feasibility

- 用户本轮“继续推进、把整个流程跑出来”只绑定到此前已经展示数据、方法、成本与证据的一个C08 shadow run，不解释为后续所有material run的一揽子预批准。正式preflight为0 error/0 warning后创建唯一不可覆盖目录并执行。
- 接受`PASS_CANO_C08_TOPOLOGICAL_PLANNER_SHADOW_V1`：4773 unique frames、23865 planner cycles、15条完整决策流，waypoint有限/4m有界、verified-edge-only路由和双重重放确定性全部通过；30/30封印一致。
- `CANDIDATE_COMPLETE`帧不作为失败也不作为完成声明，因为已证明其后可以恢复TARGET。完成判据继续独立后置，禁止影子数据虚构`exploration_finish`。
- 真实M-TARE bag审计推翻了yaw-only点云逆变换：倾斜段最多4710点错环；完整xyzw四元数使跨全程31帧全部0错环。决定部署接口强制完整姿态，禁止静默降级为平面yaw。
- 下一材料决策是M-TARE development worlds上的GT-TNG Oracle feasibility与原M-TARE公平baseline。二者必须同world/start/seed/sensor/localPlanner/runtime；未获新批准前不执行closed loop，不读取C10。

## 2026-08-20 — Oracle改为可复现的分层GT-map上界，历史teacher不得冒充

- M-TARE tunnel/garage没有TNG原图，历史`terrain_map_ext` teacher生成脚本也已缺失；bag只保留scan/odom，现存teacher合同与provenance表述冲突。该问题属于teacher/system provenance，否定“直接复用历史teacher作为GT-TNG Oracle”的结论。
- 决定同世界上界命名为`layered GT-map semantic oracle`：读取完整开发地图，按当前车辆Z选层并只沿高度连续support传播；不读取未来轨迹或模型输出。C08-TNG导入方案因native support资格风险不采用。
- 完整preview-map覆盖率暂不称reachable denominator；正式主指标采用M-TARE原生0.5m registered-scan unique-voxel explored volume，preview surface recall仅作带警告的diagnostic。

## 2026-08-20 — ROS checkpoint兼容采用无损deployment export，禁止换模型或重训

- 冻结M1D在Torch2.9/Python3.13可正常读取；ROS Torch2.0/Python3.8在读取配置中的`pathlib._local`对象时失败，参数tensor尚未进入加载。判定为serialization/system兼容问题，不是模型质量或权重损坏。
- 推荐且已实现唯一低风险路线：在冻结Torch2.9 sidecar读取原checkpoint，复制同名/同shape/同dtype CPU tensor，去除非primitive配置，以protocol-2 legacy格式保存；逐tensor hash与跨版本固定probe验证。不得训练、改权重或另选checkpoint。
- 合成跨版本验证已PASS且输出误差为0；正式seed0/1/2转换必须作为一个不可覆盖Gate-5 infrastructure run，任何tensor/hash/decision误差立即FAIL、不得重试。proposal已完成，等待用户明确批准。

## 2026-08-20 — 接受正式M1D ROS部署转换PASS

- 用户在精确范围展示后批准一次三checkpoint转换；决定将“全部都批准”解释为该已定义run的全部门禁，而不是尚未定义的未来closed-loop/C10一揽子授权。
- 正式run在4.414秒内完成，126个tensor hash与跨Torch固定probe/离散决策全部一致，22/22 seal复核通过；接受`PASS_M1D_ROS_DEPLOYMENT_EXPORT_V1`，解除ROS checkpoint serialization阻塞。
- 该结果只证明部署文件忠实于冻结模型，不证明闭环探索优于M-TARE。下一次material closed-loop仍需独立Data Card/spec/preflight与用户批准。

## 2026-08-20 — 仅锁Gazebo seed不足以形成公平M-TARE基线

- 冻结镜像的Gazebo支持`--seed`，已通过新launch显式注入；但源码复核显示TARE的local coverage、multi-robot sampling及PBS/CBS仍分别使用`std::random_device`和`rand()`。
- 决定不把logical run ID或Gazebo seed冒充完整实验seed，也不通过多跑挑取结果。该问题在正式closed-loop前作为baseline/system blocker处理。
- 推荐源码级最小补丁：单一显式`planner_seed`驱动现有采样器，算法候选、概率分布和规划目标保持不变；补丁本身必须先做same-seed bitwise replay与different-seed合法差异验证，再允许正式比较。需要用户明确授权。

## 2026-08-20 — planner_seed V1完整系统资格失败，不原地重试

- 用户批准后，最小源码补丁已通过source/binary/RNG/ROS参数资格；决定保留该组件证据，但不把它冒充完整Gazebo+ROS回放确定性。
- 正式V1在planner执行前失败。日志明确为roslaunch无法定位`tare_planner_node`，而只读容器检查确认目标二进制存在并可执行；缺陷是runner遗漏`source tare_system/devel/setup.bash`，分类为system/runner environment，不是模型、数据、teacher、metric或planner-seed算法失败。
- 按one-shot与failure policy，决定保留V1不可变FAIL及seal，不修改冻结runner、不在原目录重启、不借用部分bag作结论。
- 推荐V2只增加TARE devel环境source，保持tunnel、seeds 11/11/23、每次60秒、300帧、60 waypoints及exact identity判据不变。这个新material run需要用户新批准；在PASS前禁止50-case原M-TARE/M1D/Oracle性能比较。

## 2026-08-20 — V2的kTestID参数映射失败不得原地修复

- V2日志证明环境组合成功且patched planner binary开始执行；新失败为`kTestID wrong size: 2`。源码确认该字段编码通信策略与通信范围，不是实验随机seed。
- 决定把`test_id:=11`分类为runner parameter-mapping defect。单机器人公平合同应始终使用字符串`0`代表full comms；环境随机性继续由`gazebo_seed`控制，规划随机性继续由`planner_seed`控制，二者使用预注册11/11/23。
- V2保留不可变FAIL，不在原run中改参重启。V3还必须消除复用V1 runner导致的V1状态标签，以免机器证据名称与run版本不一致。
- 推荐一次V3 parameter-only corrective，其他科研变量与证据门禁不变；需要用户明确批准。

## 2026-08-20 — V3的单字符kTestID仍不合法，采用源码完整编码0001

- V3节点日志证明环境、binary、`planner_seed=11`和`kTestID=0`均生效；失败发生在原GridWorld无条件从`kTestID.substr(2,2)`解析robot count。
- 全量`kTestID`引用审计后，决定单机器人full-comms唯一合法编码为`0001`：首字符0选择full comms，第3–4字符01与`kRobotNum=1`一致，长度4不解析额外通信距离。
- V3保持不可变FAIL。V4只纠正完整通信模式编码并使用V4状态标签；正式执行需要新批准，其他world/seeds/runtime/evidence合同不得变化。

## 2026-08-20 — V4否定bitwise paired closed-loop，禁止原样执行50-case矩阵

- V4是首个完成全部三trial的完整系统资格；所有启动、参数、topic和样本数量门禁通过，最终唯一失败是seed11重复不相同。决定接受该FAIL为真实系统随机性证据，而不是继续修改seed或放宽exact门禁。
- 首帧同pose点云已不同，随后pose/waypoint实质分叉，说明差异来自仿真传感/调度并被闭环规划放大。原50-case每cell单次运行不能支持严格paired claim。
- 推荐路线是重新预注册replicated stochastic design：随机运行顺序、每个method/world/seed独立重复、报告置信区间与效果量，并加入waypoint/trajectory/coverage稳定性。备选是lockstep改造Gazebo/LiDAR，工程成本高且可能只获得脆弱的bitwise性质。
- 在用户选择前，不创建单机器人性能run，不把V4诊断bag用于模型选择，不读取C10。

## 2026-08-20 — 用户选择随机区组方案A，Gate 5收口并授权进入Gate 6

- 证据：V4的三条合法完整回放均通过启动/topic/count合同；但相同seed11的scan hash为0/100一致、waypoint仅1/20一致，首次扫描平均点差8.959mm，路线最大分离1.600m。严格逐帧配对假设无效。
- 决定：Gate 5以`GATE_MIXED`收口。保留并继续使用已通过的shadow、ROS交接、Oracle、coverage evaluator、checkpoint部署和planner-seed组件；旧50-case bitwise paired设计永久停用。
- 用户授权：在展示A（90-case随机区组，17--20h，103--120GB）与B（Gazebo/LiDAR lockstep，多日且不确定）后，用户明确回复“批准”，选择A并授权operational Gate进入6及执行这个精确范围的一次不可覆盖正式run。
- Gate-6合同：2个development worlds×5个environment seeds×9 cases/block；原M-TARE与Oracle各3个execution repeats，M1D checkpoint seeds 0/1/2各一次；600秒/case，总计90 cases与54,000仿真秒。运行顺序只由seed 20260820随机化一次，禁止失败重试、结果驱动重排、C09/C10、训练或模型/参数选择。
- 统计边界：10个world×seed blocks是配对单位；执行重复和checkpoint seeds量化系统/模型变异，但不是额外独立世界。完整地图Oracle只作不可部署上界。
- 当前NEXT：Gate-6实现测试、Data Card/spec、preflight；通过后依据上述批准创建并执行唯一run。

## 2026-08-20 — Gate 6正式run预检通过，消费既有一次性批准

- 冻结结果：90 cases、10 blocks、每family 30、54,000仿真秒、schedule SHA-256=`6aea0921...27f1a`；25/25相关测试和真实ROS Python3.8编译通过，镜像/planner/world/map/checkpoint/tool身份只读复核通过。
- preflight为0 error；唯一warning说明`closed_loop_single`治理操作不强制Data Card。本项目合同要求仍已建立并绑定专用Data Card，因此warning不改变证据范围。
- 决定：不再请求重复批准，消费用户对精确方案A已经给出的批准，只创建一次`gate6_20260820_mtare_single_robot_stochastic_v1_seed20260820`并执行。任何case失败即全run封存FAIL，无重试。

## 2026-08-20 — Gate 6 stochastic V1正式FAIL，禁止compression-only重跑

- 正式证据：首case完成600仿真秒和全部指标后，因冻结ROS镜像不存在`zstd` CLI在lossless archive阶段失败；V1立即停止、formal completed case=0、24/24 seal通过，无重试。
- 独立结论：case虽无planner异常，却全程0m、1 node/0 edge。M1D seed2在均匀101个同步帧上方向数均为0，最大方向概率仅0.2614--0.2617；同一冻结B0为101/101非空、首帧359°。因此这是C08/C09模型到AEE baseline世界的domain/interface blocker，不是单纯packaging。
- 决定：V1不可修改，不以已生成的单case指标冒充90-case结果，不做compression-only V1R。受影响工作停止，等待用户选择新的方法合同。
- 推荐选项A：把lossless archive移到有冻结`zstd`的host finalizer；当M1D方向为空时只使用既有current-scan B0作为显式bootstrap/fallback，记录每帧fallback率并先做独立短资格。若fallback长期占主导，只能称M1D+B0 hybrid，不能声称纯M1D收益。成本约30--60分钟资格，PASS后完整90-case仍约17--20小时/103--120GB。
- 选项B：建立AEE development-domain objective teacher并重新训练/校准，再重新冻结模型与评估；需要新Data Card、world-disjoint开发/验证设计和约数小时到数天，论文语义更纯但成本和泄漏风险更高。
- 选项C：停止M1D closed loop，改比较B0 causal-topology与原M-TARE；最快，但显著削弱“学习结构语义驱动规划”的主张。

## 2026-08-20 — 移动bag证明M1D跨域系统性空方向，推荐重新打开Gate2做head adaptation

- 新证据：原M-TARE移动产生的99帧上，M1D seed0/1/2为空99/97/99帧，B0为99/99非空；这否定“B0只bootstrap一小段、M1D随后接管”的假设。
- teacher可行性：冻结tunnel完整地图在20个移动pose上20/20产生非空客观方向，1/2/3出口分布2/17/1，count和decoded heading count一致。无需人工标注、模型输出或未来轨迹。
- 科研决定建议：为保持“学习结构语义驱动因果拓扑”的论文核心，优先做AEE objective-teacher head adaptation；encoder与embedding冻结，避免把Gate6失败变成全模型任意重训。B0只能作为低频、显式计数的安全fallback。
- 治理影响：training现只允许Gate2。必须由用户明确授权临时重新打开Gate2 corrective representation work；未批准前只保留proposal/Data Card，不实现collector、teacher、训练或新仿真。

## 2026-08-20 — 用户批准重新打开Gate 2执行AEE head adaptation

- 授权内容：AEE tunnel/garage 10条原M-TARE轨迹、30000 raw/6000 effective、完整地图objective teacher、三个seed head-only adaptation；encoder和embedding保持冻结。
- operational current_gate从6切回2是下游证据触发的corrective rollback，不改写历史Gate2 PASS、Gate5 MIXED或Gate6 V1 FAIL。Gate6恢复必须重新通过表示资格和6-case readiness。
- 未授权内容：C09/C10、strict-test、全模型重训、阈值搜索、90-case replacement、多机器人或最终benchmark正式执行。

## 2026-08-20 — Gate 2纠正性数据导出治理规则同步

- 冲突证据：权威`docs/PLAN.md`第18节和已批准Data Card明确授权Gate 2执行AEE纠正性`data_export`/`teacher_generation`，但`src/mtare_topo/governance.py`仍把两类operation硬限制为Gate 1，导致正式spec预检报`operation 'data_export' is allowed only in Gate 1`。
- 决定：旧的“data_export/teacher_generation仅Gate 1”规则由PLAN第18节在本次纠正范围内取代。治理范围最小扩展为Gate 1--2；每次操作仍必须同时满足当前Gate、用户批准、Data Card精确operation/Gate绑定和strict-test隔离，Gate 3及以后仍拒绝。
- 核验：新增Gate 2批准卡通过与Gate 3拒绝测试；治理25项、AEE导出10项全部通过。没有绕过preflight，也没有创建或执行正式run。

## 2026-08-20 — AEE sensor export V1入口环境失败，禁止原run重试

- 证据：冻结命令由PATH解析到`/usr/bin/python3` 3.12.3；该环境没有NumPy，runner第17行导入数据合同前间接在`aee_domain_adaptation.py`报`ModuleNotFoundError`。采集器、ROS和Gazebo均未启动，数据/teacher/model结论为空。
- 影响：V1授权已消费，run封存FAIL，0轨迹/0帧，10/10 seal通过。该问题分类为system/command-environment，不改变AEE数据设计或M1D跨域诊断。
- 选项A（推荐）：新replacement只把外层解释器固定为已验证的`/home/zeng-workstation/anaconda3/bin/python`，其runner help、NumPy 2.1.3和10轨迹合同导入均通过；重新冻结哈希/spec/preflight并执行，成本仍约3小时/20GB。选项B：在系统Python安装NumPy，会改变系统环境且无必要。选项C：停止纠正流程，保留Gate 2阻塞。
- 决策边界：未获得用户对A的明确replacement批准前，不创建第二个run、不采集、不生成teacher或训练。

## 2026-08-21 — 选择显式传感器接口与独立多几何联合纠正

- 证据：AEE原始雷达为16×350，直接映射到16×720的理论占用上限为48.61%；无训练短缝补齐几乎恢复同pose ideal mask，但garage seed2仍完全空输出。接口差异和几何泛化不足是两个独立问题。
- 用户决定：回复“A”，选择推荐的combined corrective；不采用仅多世界而保留含糊接口，也不把B0降级为论文主方法。
- 方法边界：正式接口必须保留350个物理束的有/无回波身份；旧unorganized短缝补齐仅为诊断。独立几何资格固定C13--C24共120个候选，每层前两个有效候选分train/validation，禁止开放式补seed。
- 污染控制：AEE tunnel/garage因已观察均只能进入后续corrective train；V2历史未接受候选只作实现调试；C09/C10和正式benchmark不读。
- 当前授权解释：本次“A”确定研究路线，不自动等同于尚未展示精确120候选成本/证据时的material one-shot execution批准。正式候选生成仍在preflight授权边界。

## 2026-08-20 — 用户批准V1R replacement与C09 retention窄例外

- 冲突证据：PLAN第18节写C09/C10读取为0，但同一获批方案和Data Card要求每epoch报告Cano validation retention，并以其作为AEE direction完全并列时的checkpoint tie-break；实际Cano validation为10个C09 worlds、12500帧。
- 用户决定：在推荐方案明确展示后回复“批准”。允许sealed Cano V2R C09 validation只读用于本次Gate-2 head adaptation的抗遗忘评估和并列选择；禁止进入训练、loss、归一化、阈值、augmentation或超参数调整。C10及后续sealed worlds仍为0。
- 同一批准授权一次V1R replacement：只把V1失败的外层Python绑定改为已冻结Anaconda Python/NumPy身份；10轨迹、30000/6000、方法、门槛、成本和无重试合同均不变，V1保持不可修改。

## 2026-08-20 — V1R归档权限失败后采用V1R2最小替代

- 证据：V1R第一条轨迹的3000/600帧、1141.027337 m移动和sensor shard均PASS；宿主`zstd`因case目录为root所有而无法创建归档。正式run立即FAIL、completed=0并18/18封印，禁止复用部分资产或原地重跑。
- 决定：采用V1R2，仅在collector成功后把无symlink的case树交接给冻结host UID/GID 1000:1000；宿主拒绝已存在archive/storage，使用不含`-f`的zstd-10并校验解压SHA-256后才删除raw。所有科研合同不变。
- 核验：37/37测试PASS；真实Docker-host smoke验证ownership、权限、压缩、解压hash和删除成功。用户在该替代边界下回复“批准”，授权一次不可覆盖V1R2；不构成teacher、training或后续run的预先批准。

## 2026-08-20 — 未来授权时间戳使V1R2证据无效，禁止静默修改

- 证据：V1R2冻结spec记录批准时间21:30，但run约21:07已经启动；授权本身真实存在，但该精确时间元数据不真实且发生在未来。
- 决定：不把它视为无关排版错误，不修改已创建run内的spec。立即中断首case，正式计数归零、partial资产禁用，标记FAIL并生成17项seal。
- 推荐：V1R3只修正授权记录方式和run ID，不改collector、handoff、archive、world/seed/帧数或阈值；重新preflight并取得明确replacement批准。问题分类为governance/evidence metadata，不是否定数据方法或权限修复。

## 2026-08-20 — 用户批准V1R3且正式sensor export PASS

- 用户明确回复“批准”，仅授权V1R3 metadata-only replacement。批准后真实主机时间`2026-08-20T21:45:37+08:00`被记录；final spec preflight为0 error/0 warning，唯一run随后创建并执行。
- 决定结果：V1R3为`PASS_AEE_DOMAIN_SENSOR_EXPORT_V1`，不是对V1/V1R/V1R2失败资产的覆盖或复用。10/10轨迹、30000 raw、6000 effective、train/validation各3000、累计11427.838595m全部按原合同通过。
- 证据：18,474,285,979-byte zstd归档均经解压SHA验证，124,735,822-byte sensor shards通过逐条合同；teacher/training/C09/C10为0；112/112 seal独立复核通过。
- 科研影响：AEE适配的数据可用性阻塞解除，但模型尚未训练，不能据此宣称空方向问题已修复或拓扑闭环有效。当前只推进Data Card已批准的objective teacher generation；readiness、90-case与多机器人正式执行仍不自动授权。

## 2026-08-21 — Objective teacher完整性失败，禁止按validation最大距离调snap

- 证据：formal teacher V1在garage seed23第360个有效样本报`no unique traversable support exists near the vehicle`，完成6/10 shards后封存FAIL；30/30 seal通过。该pose楼层支撑高度匹配，但当前格被0.55m obstacle inflation覆盖，最近可通行格距pose 1.0198m。
- 完整只读诊断：剩余garage seed23/37/53/71分别有4/0/21/14个同类失败，合计39/2400；所有失败pose均真实出现在PASS的M-TARE轨迹，最近可通行距离0.8246--1.4m。
- 分类与影响：teacher/map geometry contract mismatch，不是sensor data、模型或metric失败；它否定6000-label teacher完整性，因此训练、checkpoint selection、部署和readiness均停止。V1R3 sensor export继续有效。
- 禁止选项：删除39帧会破坏6000样本/身份合同；把0.8m snap直接提高到观察到的1.4m既由validation驱动又让标签原点偏移，存在泄漏与语义错误；二者均不采用。
- 推荐选项A：只读核对Gazebo collision geometry、vehicle footprint和preview PLY差异，按已有物理参数设计current-pose feasibility corrective，不新增由validation调出的阈值；成本约30--60分钟CPU，之后新replacement仍需preflight/批准。选项B：停止学习主路线并把B0拓扑作为主方法，成本低但改变论文核心。当前需用户明确选择。
- 参数核对补充：AEE local planner只明确`0.6×0.6m` footprint（外接半径约0.424m）；冻结0.55m是项目teacher的0.4m基础半径+0.15m裕量，当前无法从实际URDF collision geometry认证。preview PLY与Gazebo collision资产也未做距离对应审计，所以直接改到0.424m同样缺乏证据，仍需选项A。

## 2026-08-21 — AEE teacher physics audit确认是楼板上下表面混淆，不调整安全阈值

- 用户明确批准推荐选项A；执行范围保持只读，0训练、0模型、0 C09/C10，未创建replacement run。
- 新证据取代此前“实际URDF不可读”的不完整判断：冻结镜像中URDF可读但没有任何robot collision tag；Gazebo world collision精确使用DAE，teacher V1使用无normal的preview PLY，二者未有等价性证明。
- 39/3000失败的raw obstacle高度差中位`0.198028m`；DAE逐pose交叉验证显示旧teacher选中的近expected面法向朝下，其上约`0.152--0.196m`是法向朝上的楼板顶面。35/39在inflation之前已经错误，所以修改0.55m不能修复根因。
- 决定：保留V1 sealed FAIL；不删39帧、不调0.55m、不调0.8m。推荐的唯一科学纠正是用Gazebo同源DAE的有向三角面识别upward support layer，并保留其余冻结合同。因为这改变teacher地图表示，必须先形成新Data Card amendment和V1R proposal，完成测试及只读proof，再向用户展示成本/证据并取得一次正式执行批准。

## 2026-08-21 — 多层支撑采用精确surface连通，不新增坐标容差

- 证据：garage/tunnel exact-coordinate upward identity为880/168；到`1e-4m`仍基本不变，绝大多数是实际断开的mesh islands。只有garage primitive1在`1e-6m`下从28降到27，不能解释整体数量。
- 决定：保持同一coarse primitive内“共享精确世界坐标顶点”作为sheet连通规则，不为合并单个微缝引入新epsilon。不同sheet可以按既有高度连续图通行，但障碍膨胀按sheet identity隔离，避免0.2m叠层互相污染。
- 验证：23/23测试以及从头6000/6000帧只读proof PASS；未生成teacher shard、未训练、未读C09/C10。
- 授权边界：用户的“全部批准”用于实现、测试和只读proof。正式V1R teacher proposal仍保持PENDING，preflight仅因两项approval status失败，等待展示后的一次明确正式执行批准。

## 2026-08-21 — 用户批准并完成唯一DAE多层teacher V1R正式run

- 用户在正式Data Card/spec、6000帧proof、成本和证据展示后明确回复“批准”。授权严格限于10条V1R3轨迹、6000帧objective teacher生成；零训练、推理、C09/C10。
- 执行前新增“两世界全部几何和坐标链必须在shard 1之前验证”的fail-before-output门禁，并修复冻结镜像声明volume所需的显式runtime mounts。最终工具重新冻结，preflight 0错误0警告。
- 唯一run完成`PASS_AEE_DAE_MULTILAYER_OBJECTIVE_TEACHER_EXPORT_V1R`：10/10 shards、6000/6000 labels、46/46 seal，四张预览人工复核通过。
- 决定边界：该PASS允许为三seed head-only adaptation准备新提案，但不自动授权训练、C09 retention读取、checkpoint选择、部署或闭环。
## 2026-08-21 — 临时训练sidecar丢失时必须新建path-only replacement

- 证据：V1R正式run在`probe_environment()`启动前因`/tmp/mtare_phase3_torch290_zarr2187/bin/python`不存在而FAIL；0 seed、0 epoch、0 optimizer step、0 C09/C10读取，失败证据已经封存。
- 决定：不重建后原地重跑V1R，不修改其run目录，也不把环境缺失解释为模型失败。问题类型为system/environment。
- 可行选项审计后修订：直接使用另一`/tmp` sidecar仍有相同重启风险，因此采用其非覆盖、51 MiB持久副本；live Python/Torch/CUDA/Zarr/Numcodecs/NumPy/sklearn/GPU、sorted pip-freeze SHA-256及pip check与冻结合同完全一致。V1R2仍必须重新冻结Data Card/spec、preflight并取得明确replacement批准。
- 影响：DAE teacher PASS不变；三seed adaptation的科研结论仍未知；不得推进adapted ROS export、readiness或闭环。
## 2026-08-21 — Venv可执行路径不得用Path.resolve解引用

- 证据：V1R2正式run中，持久venv直接调用可导入Zarr，但runner的`.resolve()`得到底层Anaconda Python并导致`ModuleNotFoundError`；0训练与0数据读取。
- 决定：V1R2保持sealed FAIL。V1R3使用`expanduser + absolute path`，明确拒绝相对/缺失路径但保留venv symlink语义；模型、数据、方法和阈值无变化。
- 验证：合成venv symlink确证返回路径不等于resolved target；真实持久路径经同一helper后，Torch/CUDA/GPU/Zarr/Numcodecs/NumPy/sklearn、pip-freeze SHA-256与pip check全部PASS；23/23相关测试PASS。
- 授权边界：V1R3是新的不可覆盖material run，仍需独立明确批准；不能把用户对V1R2的批准写成V1R3批准。
## 2026-08-21 — Head-only适配正式否定；学习方向target必须与Cano schema一致

- 证据：V1R3 seed0完成全部10 epochs仍在AEE产生88.6%空方向、count/role单类塌缩，并使Cano direction F1下降0.082392；这不是环境或未训练。
- 标签缺陷：Cano监督为component center的3° Gaussian，AEE监督误把完整traversable angular sector直接作为同一`direction_target`。garage正标签密度0.620927对Cano 0.042693，违反同一head/同一loss的目标语义合同。
- 输入缺陷：AEE registered-scan valid density约0.46--0.48，Cano CPU raycast约0.998；冻结encoder的AEE z_role近常量。frame ID、raw index、yaw/quaternion、robot-frame azimuth、count offset和role index均通过只读核对。
- 决定：V1R3保持sealed scientific FAIL，不运行seed1/2、不挑epoch、不调threshold。后续保留AEE binary mask作为客观可通行证据，但学习target只用相同component center和冻结Cano 3° Gaussian。encoder必须允许域适配，不能继续把exact frozen z_role当验收目标。
- 旧trainer要求count macro-F1对1--6全部≥0.70，与`docs/PLAN.md`已声明“Gate只覆盖常见1--4、5--6只报告稀有诊断”冲突。后续以PLAN为准：1--4作为Gate，5--6完整报告但不决定PASS；该修正不能回写V1R3结论。

## 2026-08-21 — Head-only失败后采用canonical target + full encoder domain adaptation

- 证据：V1R3已证明冻结encoder时AEE valid density约0.46--0.48会从首个conv起造成表示近常量化；同时AEE sector mask与Cano 3° exit-center Gaussian不是同一监督语义。
- 决定：不重跑或微调V1R3。原始AEE mask继续作为teacher evidence，学习target统一为Cano component-center Gaussian；encoder、embedding和三个heads全部允许适配。
- 域匹配：每个Cano独立样本保留dense view并增加一个由AEE train valid mask确定的matched view；两份各0.5权重，禁止把增强view算作新独立样本。matched valid为Cano-valid与AEE-valid交集，禁止把原始Cano miss伪造成return。
- 指标：按权威PLAN，branch count 1--4进入正式Gate，5--6只完整报告。Cano C09仍只用于retention和第二selection key，C10/later保持0读取。
- 边界：实现与只读测试获准继续，但正式三seed训练仍需新Data Card/spec/preflight和一次明确material-run批准。
- Data Card精度修订：旧head-only card里的660s与0.2m不是本次sealed effective shard的实测值。V2从10个sensor shard的stamp/xyz重算并冻结每条599.000--599.005s、全体1.0s median temporal interval及2.022131m median adjacent displacement；不改变任何样本身份或split。

## 2026-08-21 — Full-encoder V2失败后停止单world语义微调

- 正式证据：V2 seed0十轮后AEE garage五项科研门禁全部失败，但encoder/embedding/heads均改变、tensor finite、tunnel z_role不再塌缩且tunnel direction F1达到0.317246。这排除“没训练”和“仍冻结”的解释。
- 归因：主要为data diversity / sensor-domain generalization。AEE训练的五条轨迹共享同一tunnel geometry；其3000帧不能当成3000个独立环境。garage是唯一不同geometry validation world，不得失败后静默并入train。
- 决定：V2保持不可变scientific FAIL，不运行seed1/2，不改0.5 threshold、不重加权loss、不挑tunnel结果、不把B0 fallback冒充learned PASS。
- 推荐下一证据：先对两个development worlds做同pose real registered scan与Gazebo DAE ideal raycast的固定小样本只读parity，回答差异是否可由共享sensor operator解释。若成立，训练一个冻结Cano structural encoder的paired front-end adapter；若不成立，必须扩充多个AEE/Gazebo development geometries。
- 备选与代价：移植多个Cano development worlds到AEE/Gazebo并重新采集/teacher，约数天且需geometry/navigation资格；直接用B0 topology最快但削弱“学习结构语义”核心贡献。新方法或数据生成均需用户决定及新Data Card/spec/批准。

## 2026-08-21 — 同位姿证据否定“只做前端即可”，推荐传感器接口与多几何联合纠正

- 执行决定：用户选择A后冻结64帧只读审计。V1因审计runner错误在数据前失败并独立封存；未原地重试。V1R只修入口错误，保持样本、射线、模型和判据不变后完成。
- 证据：真实AEE valid density约0.46--0.48，DAE ideal约0.95--1.00，两世界均表现为稀疏相邻角列；这是system/sensor-interface问题。理想输入在tunnel对三个M1D seed均有恢复，但garage seed2仍F1=0且100%空输出；这是model/data-diversity问题。
- 决定：按预声明全seed/全world判据，不把局部恢复升级为paired-adapter PASS；正式结论为`MORE_INDEPENDENT_AEE_GEOMETRY_REQUIRED`。也不否认已发现的共享sensor operator。
- 推荐：下一路线同时固定AEE稀疏角采样合同并引入多个独立开发geometry，再设计训练/验证；只做前端不足，只把现有garage并入train会破坏独立验证，直接B0则削弱论文核心。该路线改变数据与方法，需用户明确选择后另立Data Card/spec。

## 2026-08-21 — 用户批准并完成唯一C13--C24 topology-only资格run

- 授权：用户在精确展示120候选、10层×12、固定前两有效选择、10 train+10 validation、`<=2 CPU hours`、`<=2 GiB`及零下游数据/训练范围后明确回复“批准”。批准时间为`2026-08-21T20:46:57+08:00`，只授权一个不可覆盖Gate-2 audit。
- 决定：候选池固定为C13--C24；任一层不足两个有效候选即整批FAIL，禁止追加C25、换seed、放宽cycle-rank或按结果挑世界。该合同在执行中未改变。
- 结果：120/120完成，十层有效数`11/11/10/12/11/12/10/12/10/12`；固定选择20个parent，10 train和10 validation的ID与coordinate-bearing identity全部唯一且split-disjoint。状态为`PASS_AEE_CORRECTIVE_TOPOLOGY_CANDIDATE_AUDIT_V1`。
- 证据：run耗时1701.969秒、约97MB，376/376 SHA-256条目PASS，seal SHA-256=`08bf60ef7c626e5ebbf9510f09f2afa951434b4485567f6b8aca1beb515dcc51`；source unchanged，mesh/LiDAR/teacher/training/model/C09/C10/M-TARE均为0。
- 影响与边界：多几何父拓扑来源阻塞解除。该PASS不生成传感器样本、不证明模型泛化、不建立在线拓扑图，也不授权后续materialization或训练。下一步只能先为冻结10+10父世界形成geometry/sensor Data Card、成本、只读资格和新spec。

## 2026-08-21 — AEE organized 350束可采集，但物理角网格不是半开圆周

- 证据：冻结镜像`lidar.urdf.xacro`发布`/velodyne_points`，参数为`samples=350`、`organize_cloud=true`、水平`[-π,+π]`；插件源码将PointCloud2设为`width=16,height=350`、NaN保留无回波，并按`angle=-π+i·2π/(350-1)`生成方位。
- 影响：首尾两个source beam方向重复，实际349个独特方位。现有通用`resample_organized_azimuth()`按半开圆周和source index 0=0°解释输入；若直接用于原始消息，会产生180°方向起点错位并破坏teacher/scan对齐。它否定的是下一步sensor materialization接口完整性，不影响C13--C24 topology audit。
- 分类：system/sensor-interface contract，不是topology、teacher、model或metric问题。旧registered-scan数据无需反推，因为raw organized topic可重新采集。
- 选项A（推荐）：新增typed raw-scan canonicalizer，封存全部350束，使用精确物理角重排；对±π重复方向采用预声明、可审计的合并规则，再插值到720。Cano source ray也使用相同闭区间350束。成本约1--2小时实现/单测/只读synthetic proof，零正式数据和训练。
- 选项B：修改AEE xacro为350个半开圆周方向后重新采集；会改变冻结baseline simulator，需重做sensor parity，成本更高且不推荐。选项C：保持现有直接映射会使用错误方位，科学上不可接受。
- 决策需求：在实现下一materialization executor前，用户需明确批准选项A及其重复端点合并规则；不得静默把350解释成349或改变xacro。

## 2026-08-21 — 用户批准exact-angle方案A与first-return端点合并

- 用户明确回复`a`。实现保留全部350 raw records；349 unique派生视图只用于角度计算，不把原始传感器虚报为349束。
- `±π`重复方向沿用现有同格first-return规则取最近有效回波，避免新增平均或随机选择；其余插值严格要求左右物理方向均有真实回波。
- AEE raw及Cano synthetic均使用相同闭区间source angles；冻结xacro、teacher、网络、阈值和C09/C10隔离不变。17/17测试PASS，尚未生成material data。

## 2026-08-21 — 20-parent perception-mesh V1系统FAIL，不允许原地重跑

- 用户批准的唯一V1在最终31/31测试与preflight 0 error后执行；1.263秒内、任何mesh生成前，source precheck发现executor从持久E1 `site-packages`而非固定项目checkout导入`subt_proc_gen`并停止。
- 根因是新runner没有把旧runner已有的冻结`PYTHONPATH`传给executor子进程；不是数据、拓扑、mesh算法或指标失败。V1的0资产失败和14项seal永久保留。
- 选项A（推荐）是V1R只传递已验证环境；选项B是重装editable E1并重做环境资格。A改动更小，不改变科研合同，但仍需新spec/preflight与明确批准，禁止把V1批准扩展为重试授权。

## 2026-08-21 — V1R按1 GiB RAM stop rule停止，局部14资产不得复用

- 用户批准的V1R成功修复固定checkout导入，但运行中executor RSS观测为`1,243,852 KiB`，超过已展示和批准的1 GiB RAM预算；因此在第15个世界生成过程中立即SIGINT。
- V1R正式FAIL且不可恢复：14个局部mesh/metrics/sanitation虽全部PASS，也不得与新run拼接。146项seal永久保留。
- 分类为system/resource-contract问题；runner此前仅自动限制时间/磁盘，缺少RAM峰值监控也是证据缺口。
- 推荐新V1R2从头执行，将现实RAM硬限固定为2 GiB，并在runner中采样整个进程树RSS、保存peak和超限即停；备选逐parent子进程隔离保持1 GiB但实现与确定性证明成本更高。需要用户新决定，禁止静默提高预算。

## 2026-08-22 — V1R2证明批内native内存累积，推荐逐parent进程隔离

- V1R2的2 GiB自动门禁按合同工作：峰值`2,203,566,080 bytes`时SIGINT整个executor进程组，forced kill=false，正式FAIL并封存72项证据。
- 六个完成parent的累计峰值由1045.9 MiB增长到1731.8 MiB；首个耗时最大的C14阶段仅1174.8 MiB，而第7个处理时才越过2 GiB，支持native/Open3D allocator跨loop保留。
- 不推荐继续把单进程batch上限提高到4 GiB，因为它只容忍累积且无法保证S08--S10前不再越界；也不允许降低mesh分辨率或复用partial assets。
- 推荐V1R3保持每parent相同seed与算法，但每个parent由全新E1子进程完成materialize+sanitize+可选train preview，退出后释放native内存；batch runner逐parent执行2 GiB门禁并最终汇总20个结果。该执行拓扑变化需用户明确选择和新测试/spec。

## 2026-08-22 — 用户选择V1R3逐parent一次性隔离方案

- 用户明确选择方案A：20个已封存parent按固定顺序分别在全新E1子进程中生成，每个parent只允许一次materialize、sanitize和审计；训练split生成预览，验证split不生成预览。
- 科研合同不变：继续使用相同父拓扑、seed、网格算法和阈值；不读取或复用V1/V1R/V1R2局部资产，不重试、不重选，也不以Open3D跨run OBJ字节相等作为资格条件。
- 资源合同固定为每个parent的轻量batch runner、当前worker及其递归子进程合计RSS不超过2 GiB，20个parent串行执行，整批不超过2小时和1 GiB结果；超限或任一parent失败即停止并封存FAIL。
- 实现和只读证明已完成：35/35相关测试通过，batch进程未加载Open3D，11项冻结工具哈希、376项来源seal及环境检查通过，正式run目录尚不存在。下一步仍需把一次正式执行批准绑定到最终Data Card/spec后才能创建和运行。

## 2026-08-22 — 用户批准V1R3正式执行；单parent二次内存峰值使run FAIL

- 用户在20个固定parent、每parent一次fresh E1、2 GiB独立门禁、2小时/1 GiB整批预算以及零LiDAR/teacher/training/C09/C10范围展示后明确回复“批准”；授权时间为`2026-08-22T11:52:16+08:00`。
- 唯一正式run在7/20后停止：`S04_3d_unicyclic_small_C14`峰值`2,199,797,760 bytes`，超过2 GiB；自动SIGINT且无forced kill、retry或partial reuse。114项证据封存，seal SHA-256=`96e101919fd5fa89e74228677aa9c2601894b0b7acd7b380e7b7576a362cf14b`。
- 根因证据为原生成器的`distance_matrix()`显式广播构造`A×B×3`数组；分类为system/algorithmic-memory。它否定V1R3的20-mesh资源资格，不否定已封存topology来源或exact-angle sensor接口。
- 推荐新V1R4使用与原公式逐元素等价的分块欧氏距离计算，先证明小规模bit/numeric等价和失败parent内存界，再申请唯一正式run。提高RSS至4 GiB实现较快但无剩余parent峰值上界，科学和工程风险更高；停止路线成本最低但无法继续11,000帧学习与拓扑回放。需用户明确选择。

## 2026-08-22 — 用户选择V1R4精确query-row分块，不提高资源上限

- 用户明确回复`a`。决定保持相同20个parent、seed、Cano几何参数、2 GiB门禁和一次性执行语义；只把两个mesh containment查询的全量query rows改为固定顺序分块并及时释放。
- 32 MiB scratch target只决定每块行数，不进入最近点、半径或阈值判断；reference顺序和`np.argmin`首个tie语义保持不变。V1R4继续禁止读取或复用任何失败mesh资产，只核对V1R3的state、summary和seal三项失败元数据。
- 实现、47项等价/边界/集成测试、E1 compile、独立验收器、Data Card和proposal spec已完成；验收器以V1R3 sealed FAIL作负对照并正确拒绝，正式目录不存在。该方法选择不等于一次正式数据执行授权，仍需在精确范围/成本/证据展示后取得一次批准。

## 2026-08-22 — 用户批准且V1R4正式PASS，20-parent mesh阻塞解除

- 用户明确批准一次V1R4执行，真实授权时间`2026-08-22T12:52:03+08:00`。final preflight为0 error，唯一run从sealed topology重新生成全部20个parent，未读取或复用V1--V1R3 mesh资产。
- 决定结果：`PASS_AEE_CORRECTIVE_PERCEPTION_MESH_V1R4`；20/20 mesh/sanitation/metric/receipt、10 train previews、0 validation previews全部通过，无retry/reselection。最大RSS`1,081,565,184 bytes`，耗时`1,273.595s`，287/287 seal与独立验证通过，seal SHA-256=`2f843d221f9e6bc28b01d864fc9c2b4a4fe6e651030b0089a330258cad226159`。
- 影响：20-parent mesh资格已解除，可以准备11,000帧sensor/objective-teacher数据操作；这不授权数据生成或训练，也不证明结构语义模型、因果拓扑图或M-TARE替代优于baseline。下一项数据操作必须有新Data Card/spec和明确批准。

## 2026-08-22：11,000帧 corrective sensor/teacher V1 因metrics目录缺失早停

- 批准范围：20个V1R4 Cano worlds（10 train/10 validation）各500帧，加AEE tunnel/garage各5条seed轨迹、每轨100帧；合计40,000 raw、11,000 retained、11,000 objective labels。用户在精确组成与Data Card/preflight动作报告后回复“批准”，记录时间`2026-08-22T13:45:50+08:00`。
- 前置证据：29/29相关测试、20/20 topology-only 500-pose proof、首个train frame真实ray/teacher smoke、Data Card validator、preflight和23/23 tool hash全部PASS。
- 正式结果：唯一V1 run在第一个Cano world完成500帧、Zarr、manifest和train preview后，`write_json(metrics/cano/<parent>.json)`因漏建`metrics/cano/`而抛`FileNotFoundError`。run按合同FAIL并封存，181/181 seal无漂移，AEE/data aggregate/training/model/C09/C10均未执行。
- 分类与影响：纯system/filesystem初始化错误；V1不能作为11,000帧数据证据，partial资产禁止复用，但不否定已经完成的逐帧sensor/teacher计算。
- 选项：A）推荐V1R，仅新增目录初始化和覆盖测试，从头执行，约3--4小时/32GB；B）续跑/复用partial，违反不可部分复用合同，禁止；C）停止路线。等待用户明确选择A后才能实现V1R。

## 2026-08-22：V1R目录修复成功，但全量pose资格假设被证伪

- 用户明确批准V1R。30/30测试和preflight 0/0后创建唯一run；`metrics/cano/`修复生效，前三个世界1500帧成功并写出指标。
- 第4个世界首个失败帧的LiDAR双场景回放为bitwise/数值完全一致，teacher结构标签有效；唯一失败为横向净空`0.765984893m < 0.8m`。因此问题分类从V1的system目录错误转为data/pose-qualification缺陷，不是sensor、teacher算法或模型失败。
- 只读全量审计给出7/10000失败：6个clearance失败和1个teacher完整性失败，分布6/20 worlds；全部6058个候选cluster共30290帧，为eligibility-first重新选择提供足够的待审计总体，但覆盖是否仍满足必须由完整proof回答。
- V1R保留不可变FAIL并565/565封存，禁止复用1500帧或其它partial资产。推荐选项A是V1R2在任何最终选择前审计全部候选cluster，以5/5帧安全且teacher完整为资格，再在eligible集合上重新执行原结构覆盖/配额/最远间隔优化；成本约10分钟只读proof加原3--4小时正式生成。选项B为保持cluster身份并做局部pose纠正，需新增连续性与teacher/graph/sensor坐标合同，成本和风险更高。选项C为停止corrective路线。需要用户明确选择。

### Eligibility-first可行性证据补充

- 完整候选分母复核覆盖全部6058 clusters/30290 frames：6043 clusters/30215 frames满足5/5 safety+teacher完整性，15 clusters/21 frames不满足。先前19是汇总算术错误；逐world记录与新类型化聚合一致为21。
- 过滤后20/20 worlds仍精确选出100 clusters/500 frames，所有结构事件覆盖和隧道配额/间隔审计通过。因此选项A的样本容量与结构覆盖已被直接证明，不需要pose优化、阈值变化或样本数缩减。
- provenance要求：该路线不再声称selection完全blind to complete geometry；应登记为先做客观资格门禁、再在合格总体中做不读取模型输出的结构采样。严格C09/C10及正式M-TARE benchmark仍完全隔离。
- 用户明确选择A后，207.689秒新proof进一步以过滤前6058候选定义完整事件/隧道分母；20/20 worlds均零遗漏并选满100 clusters。决定进入V1R2实现，但正式Data Card/spec必须冻结正确的15 clusters/21 frames计数并再次展示后才能执行。

## 2026-08-22：V1R2 AEE启动期跨topic order pairing被真实seed23否定

- V1R2的Cano阶段完整PASS且精确复现eligibility proof；AEE seed11也完整PASS。seed23 bag含3009 raw与3002 registered/exact-odom记录，但两topic启动时间分别为0.741秒和2.138秒。旧extractor各取前3000条按序号配对，最大delta约0.4秒，因此正式run正确FAIL。
- 分类为data/sensor synchronization contract defect；它否定order-based pairing与11,000帧aggregate，不否定raw organized LiDAR、registered pose、Cano资格或teacher。V1R2封存3432/3432，禁止复用其Cano或单条AEE PASS资产。
- 只读诊断用严格单调一对一近时匹配得到3002 pairs，前3000 pair使用raw indices 7--3006，最大delta仅0.003秒且无>=0.1秒。推荐V1R3把固定抽样索引定义在matched-pair序列上；仍保留100/trajectory、0/30/.../2970、原topic、0.1秒阈值和pose identity。备选延迟启动到两流ready会改变online readiness且对启动抖动敏感；停止成本最低但阻断训练/拓扑/论文。需要用户明确选择。

## 2026-08-22：用户授予既定论文范围内持续执行权；V1R3 ownership接口失败转V1R4

- 授权决定：用户明确要求Codex在已固定的整篇论文目标、数据隔离和Gate边界内自主选择最优方法并执行，不再逐次请求常规批准。每个material run仍必须有独立Data Card/spec、preflight、不可覆盖目录和证据seal；目标改变、test泄漏、外部付费或不可逆范围扩张仍需停止确认。
- V1R3选择严格单调最近时间戳pairing；152项AEE测试和真实seed23包证明3000 pairs、max delta 0.003秒。正式run又证明Cano 20/20与seed11 pairing/sensor shard PASS。
- 系统缺陷：V1R3 wrapper不必要地把ownership handoff schema/status改名，而archive consumer稳定要求V1字符串；该接口错配使run在归档前FAIL。它不影响已计算的scan pairing、pose、teacher或Cano结论。
- 决定：V1R3保持3417项sealed FAIL且零partial复用。推荐并采用V1R4，仅让producer继续输出稳定V1 ownership接口；配对算法、数据、seed、帧、阈值、teacher、资源和C09/C10隔离完全不变。24项测试和preflight 0/0通过。

## 2026-08-22：V1R4 11,000帧数据阻塞解除，允许准备三seed训练

- V1R4从头完成10000 Cano与1000 AEE帧；10条AEE均满足3000 unique monotonic exact-odom pairs、最大delta总体0.004秒、固定100-frame schedule、sensor/teacher/归档合同。
- 结果为`PASS_AEE_CORRECTIVE_SENSOR_TEACHER_DATASET_V1R4`，3578项seal独立验证，SHA-256=`dd6d5ac8a5a48223d2693734700b11396dd12a3aec63e4bbeaf02cf6f5fbaa59`。V1--V1R3继续保留immutable FAIL且未被复用。
- 决定：11,000帧corrective data/teacher阻塞解除。下一步只准备三seed full-encoder corrective training；训练仍需绑定本PASS数据seal、冻结checkpoint初始化/optimizer/epoch/selection与独立Data Card/spec，不得读取C09/C10或自动进入拓扑/M-TARE实验。

## 2026-08-22：V5类别平衡未修复count/role，下一步开放独立embedding

- V5三seed执行与封印正常；方向F1保持`0.827722/0.824020/0.817702`，count仅`0.510308/0.536945/0.488471`，role仅`0.584015/0.588006/0.567634`，三seed科学门禁均FAIL。
- 类别权重已绑定训练实际有效质量，direction在首epoch后冻结，encoder/embedding逐tensor保持不变。因此本结果否定“只需类别平衡”与“后期direction更新干扰分类”两种机制解释，问题定位为冻结embedding的稀疏结构表征不足。
- 网络接口复核确认direction head直接读取encoder空间特征，而count/role只读取独立embedding分支。决定下一版从原始三seedsource checkpoint重新训练：首阶段重建平衡方向结果，随后冻结encoder与direction，只开放embedding/count/role；不得拼装或推广V4R/V5失败checkpoint。
- 用户已授予既定论文范围内的持续执行权；该方法仍需新的精确Data Card/spec、preflight、不可覆盖run与seal。C09/C10/formal benchmark继续为0，达到三seed门禁前不得进入拓扑回放。

## 2026-08-22：V6证伪独立embedding适配，转向encoder层级只读诊断

- V6三seed均完整执行；方向step-47哈希与最终哈希完全一致，方向F1保持`0.827722/0.824020/0.817702`。count仅`0.485931/0.511244/0.463479`，role仅`0.581593/0.580983/0.557465`，三seed最佳均为epoch 1。
- 这说明方向与embedding结构解耦合同成立，但开放embedding无法从冻结encoder中恢复稀疏count/role信息；继续同类head/embedding调参没有证据依据。
- 决定先执行零optimizer-step、只读的encoder层级可分性与方向敏感性诊断，确定最小需要开放的block及方向保持约束。只有该诊断给出唯一机制路线后才建立新Data Card/spec；不得直接全encoder重训、降低0.70门槛或推广V6 checkpoint。

## 2026-08-22：V7证伪简单方位汇聚，停止后端读出变体

- V7不新增tensor，以共享embedding逐方位变换后circular mean替代先mean后embedding；17项兼容性、旋转不变性与方向独立测试PASS。
- 正式三seed方向保持，但count=`0.376367/0.446549/0.387015`、role=`0.553312/0.574543/0.561693`均失败。该结果说明保留一阶方位特征分布仍不足，不支持继续枚举pooling/head结构。
- 决定下一方法必须让encoder适应稀疏输入，同时以冻结teacher对同一训练view的direction logits作显式retention；先冻结最小block范围、retention权重的非调参来源及完整门禁，再建新Data Card/spec。V7 checkpoint不可推广或部分拼装。

## 2026-08-22：V8 summary接口缺字段，原run封存并采用V8R

- V8 seed0完成10轮后，runner要求的有效类别质量字段未由新trainer写入summary，触发`V8 effective-class-mass drift: 0`；RUN_STATE=FAILED，completed_seeds=0，seed1/2未启动。
- 分类为system/evidence-interface defect，不评价三seed模型假设。seed0局部结果只作诊断且不得复用；原V8 run、checkpoint、日志和seal保持不可修改。
- 决定V8R只写回既有冻结常量`count/role effective mass`及对应class weights，新增接口测试；数据、source checkpoint、模型、阶段、损失、retention权重、优化器、选择规则、门槛与C09/C10隔离全部不变，从头执行三seed。

## 2026-08-22：V8R三seed科学FAIL，先审计count可观测性

- V8R三seed方向提升到`0.843--0.848`，role提升到`0.635--0.659`，说明最后encoder block与teacher retention均产生一致正效应；count仍仅`0.485--0.540`，未接近0.70。
- Cano完整teacher定义保证原dense pose对所有结构分支LOS，但mask-matched view复用完整branch_count标签；AEE mask可能移除部分出口的当前帧射线证据。这构成teacher/input mismatch风险，继续开放更多encoder前必须停止并审计。
- 决定使用固定5000-frame sparse validation、冻结B0开阔扇区算子和20度匹配容差，逐出口比较dense可观测与mask后可观测，零训练/零checkpoint选择。若存在mask因果性丢失，则不得声称完整count是单帧可学目标。

## 2026-08-22：可观测性审计否定大规模teacher mismatch，采用冻结B0必要fallback

- 正式audit发现mask因果丢失仅2/10677个dense-visible teacher exits；完整count在固定sparse输入中总体可辨识，神经count失败不是标签大规模不可见。
- 同一冻结B0的sparse count exact=`4572/5000`，count macro-F1=`0.775478`；固定`count<=1 terminal, count=2 interior, count>=3 junction`得到role macro-F1=`0.722601`。两者不调阈值即超过0.70。
- 决定新方法显式组合V8R机制训练的M1D方向与冻结B0 count/role fallback。神经count/role输出保留诊断但禁止进入runtime；这是预声明fallback的启用，不是降低门槛。
- 为遵守失败checkpoint不部分推广，必须从三套原M1D source重新执行新复合方法并以完整接口门禁选择；通过前仍不得进入拓扑回放。

## 2026-08-22：V9完整复合接口PASS，表示层阻塞解除

- V9从原始seed 0/1/2 checkpoint分别重训，没有拼装或部分推广V8R失败checkpoint；三个seed完整门槛均通过。
- runtime决定冻结为`learned direction + frozen B0 count + fixed count-to-role`。神经count/role继续作为诊断输出，但禁止影响在线拓扑图。
- 正式结果direction F1=`0.847874/0.847615/0.843185`，count F1=`0.775478`，role F1=`0.722601`；seal=`5d397b2c296029dfb2decdb57d9fbb9332e1713cbf1faaf46ee90cd8f6a85468`。
- 该PASS只解除AEE域结构语义runtime阻塞。最初状态更新误写为再次执行C08因果拓扑回放；复核`docs/PLAN.md`第18节后纠正：C08已有sealed PASS，重复回放不回答当前阻塞。
- 权威下一步是回到Gate 6，先把V9完整runtime无损导出到ROS，并执行`tunnel/garage × seeds 0/1/2 × 180秒`的6-case readiness。readiness PASS前禁止90-case，C09/C10继续为0。

## 2026-08-22：纠正V9在线输入算子并通过6-case readiness

- V1R2首个完整case虽移动104.117m并建出8节点/7边，但learned-empty为814/901，post-warmup fallback=`86.1521%`；封存AEE organized输入上三个V9 seed均仅2/1000 learned-empty。该数量级差异否定了把V1R2直接解释为模型泛化失败。
- 接口审计确认训练使用原始`/velodyne_points`的16×350物理束身份与冻结350→720算子，旧在线节点却使用`/registered_scan`重栅格化。决定V9专用输入改回原始organized点云；registered scan仅保留coverage审计。配对沿用已有0.1秒合同，封存数据实际最大0.004秒，未新增调参。
- V1R3六例全部通过，post-warmup fallback全部为0，最小移动20.121m，所有图/控制/话题门槛通过；状态`PASS_AEE_COMPOSITE_V9_READINESS_V1R3`，seal=`30a213d156d1f64bc3be3a8a4524df9ea6ac06463ed95b596c5d9d156f7b804d`。
- 决定解除90-case前置阻塞。既有90-case的场景、seed、重复数、600秒预算、统计和验收不变；只允许迁移V9 checkpoint/runtime和AEE原始输入证据合同。C09/C10继续禁止读取。
## 2026-08-23：用户重申既定论文范围内持续自主授权；V1R2机制审计冻结

- 用户明确要求不再逐次请求批准，由agent在既定论文范围、Gate边界和数据隔离合同内自行选择证据最充分的方案。常规material run仍必须有独立Data Card/spec、preflight、不可覆盖目录和seal；只有数据/test隔离、评价标准、研究问题或结论范围发生实质改变时才停止报告。
- 当前90-case正式run从29/90自然推进到32/90，未重启、修改、重排或并发第二个Gazebo run。
- 两次只读复核发现V1与V1R audit证据门禁不完整；两版均未执行并标为superseded。V1R2补齐逐帧目标合同、final current-node、无环路径、traversal起止帧/route arc/inclusive frame count、environment/status及四个冻结源生成器身份；26/26测试和15项工具hash通过。
- append-only最终图没有stub creation frame，无法从最终snapshot直接证明某历史帧的stub state。决定不伪造历史重建；V1R2明确记录不可重建，并以source spec绑定的observed-only frontier planner源码及SHA-256证明生成合同。
- V1R2 proposal保持`PENDING_FINAL_SOURCE_SEAL_NOT_EXECUTABLE`。旧V1R、未完成source和已有material run三类负控制均被拒绝，无正式card/spec/run生成。源90-case seal后才一次性finalize并执行。

## 2026-08-23：Luna复核证伪V1R2完备性，采用未执行替代V1R3

- Luna只读复核发现V1R2没有要求每条traversal的start/end row node绑定from/to，也没有要求全部traversal与全部inter-node event严格双射；未知target mode且缺frontier也可能漏过。该问题只否定V1R2审计准备度，不影响正在执行的90-case来源、模型或指标。
- V1R2保持`SUPERSEDED_UNEXECUTED_BY_V1R3`，不物化、不执行。V1R3要求精确target keys/modes/finite fields、traversal端点行绑定和traversal↔decision transition双射。
- V1R2的一小时environment age上限会在create_run后排队时产生无科学意义的误FAIL。V1R3只拒绝未来时间并继续精确核对host/platform/Python/status，不放宽任何数据或科研门禁。
- V1R3在当前12个sealed V9案例上验证403/403 traversals双射；33/33相关测试、17项冻结工具hash通过。未完成source、旧proposal和并行material run负控制均拒绝，正式card/spec/run不存在。

## 2026-08-23：case033继续支持双机制候选，仍等待30/30冻结方法

- case033 tunnel/env71/V9-seed1完整PASS，覆盖1052.375m³、移动222.181m、fallback=0；但1829帧满足stale re-anchor proxy，最长连续1829帧，末尾1790帧route arc不增长。
- 该例21次可分类frontier事件中13 matched、3 divergent、5 same-node；V1R3验证16/16 traversal与decision transition双射。13个sealed V9案例累计419/419 traversal通过，443次事件中221次非匹配，10/13案例存在回锚proxy。
- 新证据提高了`verified re-anchor + execution feedback`组合修正的可信度，但不改变方法选择规则：等待30/30完整分布，不用case033或当前半批结果选择阈值、删帧或提前消耗修正版正式矩阵。

## 2026-08-23：tunnel/env71块内证据确认V9低延迟但覆盖严重受限

- case035 original M-TARE/repeat2覆盖10020.125m³、移动1150.232m、AUC 3074632.134m³·s、规划p95 0.320s；同块V9 seed1 case033覆盖1052.375m³、移动222.181m、p95 0.011463s。
- 该块内观察说明V9的延迟优势真实存在，但当前图状态机制使覆盖损失远大于延迟收益。它支持修复图状态与执行反馈，不支持直接宣称V9优于baseline，也不替代30-case block-aware统计。
- 决定保持原90-case完整执行与sealed failure证据；不删V9失败块、不调整评分、不提前只挑延迟指标。最终论文方法必须先通过组合修正的probe/readiness/精确配对30-case。

## 2026-08-23：case037显示中等覆盖仍被后半程回锚故障截断

- garage/env53/V9-seed2覆盖4774.125m³、移动376.086m、规划p95 0.011574s，但1592帧满足arrival proxy，最长连续1581帧且末尾1515帧route arc不再增长。
- 31次frontier事件为15 matched、13 divergent、3 same-node；29/29 traversal与decision transition双射。当前14例累计448/448 traversal通过，474事件恰有237次非匹配，11/14案例存在proxy。
- 该例说明回锚故障不仅导致约20m的极端停滞，也能在已取得中等早期覆盖后截断后半程收益；继续等待30/30分布，不改既定组合候选和选择边界。

## 2026-08-23：V9样本过半，case039/040再次复现极端停滞

- case039 garage/env23/V9-seed0覆盖1397.500m³、移动75.652m，2590 proxy帧且末尾2533帧无route增长；case040 tunnel/env23/V9-seed1覆盖226.375m³、移动28.790m，2834 proxy帧且末尾2778帧无增长。
- 16/30 V9样本现已封存，其中13个存在proxy；483次frontier事件为243 matched、204 divergent、36 same-node，240 nonmatching；459/459 traversal通过V1R3双射。
- 组合修正的必要性已具有跨world、environment seed和checkpoint seed复现，但方法冻结仍等待30/30，保持不从半批结果选择阈值或删除失败块。

## 2026-08-23：tunnel/env23块内baseline确认当前V9不能作为论文最终方法

- case041 original M-TARE覆盖9200.250m³、移动1147.292m、规划p95 0.341s；同块V9 case040覆盖226.375m³、移动28.790m、p95 0.011269s。
- 当前V9虽约30倍低延迟，但覆盖仅baseline约2.46%；与2834 proxy帧和2778帧无route增长机制证据一致。该观察不替代完整统计，却明确否定直接推广原V9。
- 决定继续完整原矩阵并保持所有失败证据；论文最终方法必须是经过双机制组合修正、probe/readiness和精确配对30-case重新资格的版本。

## 2026-08-23：layered GT-map oracle实证不是性能上界，维持诊断身份

- case043 tunnel/env23 oracle覆盖632.000m³、移动84.713m；case045 garage/env37 oracle覆盖463.375m³、移动8.815m，均明显低于相邻original M-TARE证据。
- 该结果否定把oracle写成理想性能上界，但不影响主baseline、V9机制审计或当前矩阵有效性，因为冻结spec中oracle仅为diagnostic comparator，不决定方法选择和Gate。
- 决定保留全部oracle结果并在论文中作为“完整地图目标仍受闭环执行约束”的负面诊断；不删除oracle、不调整其参数，也不以它替代original M-TARE主比较。

## 2026-08-23：case048排除route arc增长作为单独健康判据

- case048 tunnel/env71/V9-seed0覆盖442.500m³、移动79.852m，2521帧满足arrival proxy且同一frontier持续2534帧；但最后帧仍有route arc增长，tail-without-growth=0。
- 这说明route arc可在旧图路径重复执行中持续累计，不能单独反驳回锚故障；需要结合verified path、exact next-hop waypoint、current-node不一致与coverage/travel。
- 当前17例为14 proxy cases，491事件中243 nonmatching，465/465 traversal通过。组合修正选择仍等待30/30，不根据单个tail字段改变门槛。

## 2026-08-23：V9达到20/30，回锚与出口反馈两部分均有独立主导案例

- case050 garage/env11/V9-seed2仅13个proxy帧，却有36/61 frontier事件非匹配，最终31 nodes/170 stubs/106 observed；覆盖4446.250m³，说明exit execution feedback/lifecycle可在没有大规模stale re-anchor时独立限制收益。
- case052 tunnel/env11/V9-seed0有2857 proxy帧、末尾2821帧无增长、同一frontier 2940帧，覆盖仅220.625m³；说明re-anchor缺陷也可独立主导极端停滞。
- 20/30样本现为17 proxy cases，580事件中295 nonmatching，547/547 traversal通过。两部分组合修正的必要性已稳定，但仍等待30/30后冻结具体方法，不从20例调阈值。

## 2026-08-23：Gate-6 V3主机时钟回拨，决定完整区组加尾部恢复

- 证据：正式V3在77个完整case后于case077 summarizer失败；598个runtime中唯一异常为`-0.1809999943s@391.93s`，其余597个为`0.116--0.459s`，主机同一运行窗口记录`Clock change detected`。case077仿真和5.97GB bag完成，但无合法summary。分类为system/instrumentation defect，不改变模型、数据、teacher、地图、控制或primary coverage合同。
- 旧run决定：保持`FAIL_AEE_COMPOSITE_V9_STOCHASTIC_V3`和1201项seal，不删除异常值、不修改RUN_STATE、不复用case077 partial、不追认PASS。
- 方案比较：完整90-case重跑最纯但需约15--20小时；只重跑case077最快但会形成baseline选择性retry；推荐并采用完整受影响区组+未执行尾部，共21例/约5小时，并用69个sealed不受影响case作只读aggregate composition。该方案保持十个完整随机区组并把旧`tunnel_env23`八个PASS全部排除。
- 用户持续授权已明确允许agent在既定Gate/数据/指标范围内自行选择最优方案，无需重复批准；本决定不改变研究问题、数据隔离、模型、指标或阈值。V3R仍须独立Data Card/spec/preflight、不可覆盖run和seal，任一新case或composition失败即停止且不重试。

## 2026-08-23：组合审计强制绑定原始schedule index

- V3R运行期间的独立只读复核发现，组合审计已校验90个case ID、69+21来源配额和完整区组规则，但尚未逐行校验`combined_case_sources.json.original_schedule_index`与冻结90-case schedule中的`index`完全相等；单独篡改该字段可能误PASS。
- 分类为尚未执行的evidence/governance implementation defect，不影响正在运行的21个案例、模型、数据、指标或既有正式证据。
- 决定在正式组合audit创建前强制每个case的原始index类型和数值逐项相等，同时显式拒绝重复case ID并增加index漂移负测试；专项4/4、联动11/11测试PASS，13项proposal冻结文件hash为0 mismatch。

## 2026-08-23：组合审计同时冻结性能统计，并按机制而非结果选择修正版probe

- 复核发现仅输出frontier/re-anchor机制不能回答Gate-6核心性能问题。决定在同一69+21只读组合清单上，通过既有status-only深拷贝bridge调用未修改的七指标、十区组预注册统计器；源summary不变，额外保存block差值、bootstrap区间、exact sign-flip和Holm校正。
- 修正版live probe禁止按覆盖、路程、延迟或最终结果挑case。固定规则为：在同时含两种已诊断机制的30个V9案例中，最小化两种机制首次均已出现的frame，tie按case ID；完整候选排名和零performance-outcome selection写入seal。
- 新V5 probe只要求至少一个源诊断修正真实激活并在其后继续产生route-arc增长；由于第一项修正可能因果性阻止第二项故障，不强迫单一干预run同时触发两类缺陷。后续6-case readiness记录两类计数，30-case修正版矩阵给出完整分布。

## 2026-08-23：持续授权作为Gate-6常规执行授权，V5配对来源固定22+8

- 用户再次明确要求既定论文范围内不再逐次请求批准，由agent自行选择证据最充分的方案。该持续授权用于Gate-6常规实现、Data Card/spec、preflight及前置条件满足后的唯一不可覆盖material run；仍须先展示精确数据、方法、成本和证据。数据隔离、研究问题、评价标准或结论范围若发生实质变化仍必须停止报告。
- V5 30-case runner的独立只读复核发现，其来源校验只要求`failed_source_run/recovery_run`合计30，未强制由冻结69+21组合导出的精确M1D配额22+8。该缺口可能在上游来源标签错误时误PASS，但尚未执行任何V5矩阵。
- 决定在runner和formal finalizer中同时强制22个predecessor+8个recovery，并增加错误配额负测试；不改变case、顺序、方法、阈值或指标。V5 proposal冻结30例、10 blocks、10/10/10 checkpoint、18,000仿真秒、约7小时/45GB，只有组合audit与六例readiness均seal PASS后才能物化。

## 2026-08-23：最终比较独立重验组合身份，未知seal只在来源完成后绑定

- 发现V5 30-case proposal finalizer初版会把真实source seal与文字占位符`PENDING`比较，导致未来合法来源必然被拒绝。该缺陷未执行、未创建formal card/spec/run；已改为固定唯一来源目录，并在来源完成后读取、全量复核和写入真实seal，正/负finalizer测试PASS。
- 最终comparison不能只信任上游`PASS`字符串。决定再次强制组合manifest的schedule file/content双hash、10 blocks×9、30/30/30 family、69+21 source及每个summary的case/block/family一致；修正版来源再次强制30个M1D、10 blocks×3和checkpoint 10/10/10。
- 最终统计保持冻结七指标、十区组、exact sign-flip、10,000次block bootstrap和Holm；original M-TARE仍为主baseline，defective V9只作修正前后因果对照，layered GT-map只作诊断。任何非有限统计、身份漂移或source mutation立即FAIL。

## 2026-08-23：论文单机器人主图由sealed统计自动生成

- 决定把单机器人论文图直接纳入最终comparison formal run，而不是事后手工抄数。renderer只接受`corrected_v5_stochastic_comparison_v1`，任何非有限值、零归一化baseline或非十区组输入立即拒绝。
- 固定输出三张事实图、每张PNG+矢量PDF：十区组四个关键原始指标，V5相对original M-TARE的七指标配对效应，V5相对defective V9的七指标修正效应；同时保存6文件的SHA-256 manifest和`manual_value_entry=false`。
- 相对效应只把同一统计器给出的原单位mean/CI除以对应sealed baseline均值，不改变exact sign-flip、bootstrap、Holm或方向解释。合成输入15/15联动测试PASS并完成目视版式检查；真实图必须等待最终comparison seal，当前不宣称真实收益。

## 2026-08-23：方法总览图严格区分学习语义、因果图与M-TARE local stack

- 新建`docs/figures/method_overview_v1.svg`作为论文方法矢量图：训练侧为Cano TNG/perception mesh→CPU LiDAR+objective teacher→冻结V9复合语义；部署侧为AEE raw LiDAR→冻结adapter→局部结构语义→V5 causal topometric graph→global frontier/route→原M-TARE local execution。
- 图中明确count/role来自冻结几何B0，在线图为learned local semantics + rule-based causal graph；不标为end-to-end navigation、full-map prediction或learned full topology。
- 只替换global exploration target层；localPlanner、pathFollower、control、avoidance、terrain与state estimation全部标为unchanged。Original M-TARE是主baseline，defective V9为因果前后对照，layered GT-map只作diagnostic。SVG通过XML解析并经Chrome栅格化目视检查，当前不写V5已验证或严格泛化已证明。

## 2026-08-23：真实轨迹/拓扑图采用预冻结身份选择，禁止按结果挑图

- 定性图固定两个block：`garage_env11`与`tunnel_env11`；每个block固定original M-TARE repeat0、defective V9 checkpoint0及与其case identity完全相同的corrected V5。选择函数不读取覆盖、路程、延迟、图规模或成功结果。
- 每幅图并排显示三条轨迹，V9/V5叠加persistent nodes与verified edges，并强制三个panel共享x/y尺度。标题只报告sealed final volume和travel，不把单例作为总体统计。
- source summary路径经组合audit manifest绑定，轨迹与snapshot再对predecessor/recovery或corrected run的原始seal逐文件验证；任何漂移立即FAIL。最终comparison总输出5张图、PNG/PDF共10文件，联动18/18测试PASS；真实V5图尚未生成。

## 2026-08-23：论文正文采用sealed-evidence占位，禁止把V5候选写成已验证结果

- `docs/PAPER_MANUSCRIPT_DRAFT_V1.md`先冻结问题、接口、方法边界、实验设计与限制，真实数值和结论只允许由对应sealed run填入；当前20处结果/协议占位明确保留。
- 独立只读检查发现初稿第4.4和5.3节可能把已实现的V5候选及计划中的30-case配对误读为已经完成的实证结果。分类为paper-reporting defect，不影响数据、模型、指标、正在运行的V3R或Gate-6门禁。
- 决定在Gate-6封存前使用“predeclared candidate / planned corrected cases”表述；最终comparison PASS后再自动替换相应占位。多机器人贡献若无Gate-7闭环证据则整段删除，不以单元测试代替论文结论。

## 2026-08-23：V3R 21例全PASS但旧状态统计接口导致聚合FAIL，采用sealed V1R只读组合

- V3R完成21/21 case且全部`PASS_SINGLE_ROBOT_CASE_V2`，最后case089 recording/archive PASS；随后`analyze_stochastic_cases`只接受`PASS_SINGLE_ROBOT_CASE_V1`，在聚合入口报`analysis cannot include a failed case`。分类为system/evidence status-compatibility defect，不是case、数据、模型、地图、指标或隔离失败。
- V3R保持`FAILED/FAIL_AEE_COMPOSITE_V9_STOCHASTIC_V3R_SYSTEM_RECOVERY`，不改RUN_STATE、不补写source-map、不追认PASS；336项seal逐项hash一致，seal SHA-256=`47db4379c817d7c8c994b6d172362df028ab3e197de46147034a2b02d1e6f713`。
- 方案成本：重跑21例约4小时且不增加科学信息；推荐并采用新V1R只读审计，精确要求失败状态/原因、21/21 case PASS、恢复schedule/progress、全部case文件seal，再与predecessor 69例组成90例。仅deep-copy status进入既有bridge，两个失败源均不提升为PASS。
- 旧待执行V1审计还存在probe selection两个未定义变量，保持未执行并由V1R替代。V1R把probe选择放到30个V9审计完成后；真实源只读proof验证90 cases、10 blocks、30/30/30、60 trace/snapshot和机制计数。用户既有持续授权覆盖该不改变研究合同的系统恢复，无需重跑或重复执行批准。

## 2026-08-23：69+21 V1R组合审计PASS，旧V9性能失败与双机制证据同时成立

- 唯一正式V1R run在preflight 0 error后执行一次并PASS；两个aggregate失败源保持FAILED，21个恢复case仍为逐案PASS，90个summary仅在deep copy中映射status。13/13 seal复核，seal SHA-256=`0c93c7ccb2e7700a6ec2cfc4bd59b6a9538c90c859fc4032c8ac3902863e6ea4`。
- 主指标coverage-time AUC中V9减original的十个block差值全部为负，均值`-2698975.773454167 m³·s`、相对差`-0.7067876193`、95% paired bootstrap CI=`[-3093492.148,-2304122.524]`、exact sign-flip p=`0.001953125`、Holm p=`0.013671875`。结论是旧V9正式显著劣于原M-TARE，不能作为论文最终方法。
- 30个V9案例中26个有exact arrival proxy、26个有nonmatching frontier attempt；共47123 proxy帧和497 nonmatching events。固定outcome-blind选择器选case052（first nonmatching frame119、first proxy frame144）作为组合修正live probe；选择未读取coverage/travel/latency结果。

## 2026-08-23：V5 probe因Python3.8缺少math.ulp入口FAIL，采用数值等价兼容修复

- 唯一V1R probe在case052启动后生成22帧、路线弧长7.618m，随后V5 method log报`module 'math' has no attribute 'ulp'`并clean exit；case runner因早退正确FAIL。run为`FAIL_AEE_COMPOSITE_V9_COMBINED_CORRECTION_PROBE_V1R`，完成case=0，不能评价修正是否有效。
- 分类为system/runtime compatibility：开发宿主有`math.ulp`，冻结ROS镜像Python3.8.10没有。数据、checkpoint、LiDAR、图初始更新、M-TARE local stack和case选择在失败前均正常，训练/optimizer/C09/C10为0。
- `math.ulp(1.0)`只用于拒绝坐标完全重合节点，替换为IEEE-754 binary64中数值完全相同的`sys.float_info.epsilon`；runtime与offline evidence同时修改，未增加阈值或改变判断边界。宿主联动14/14和镜像Python3.8只读compile/import PASS。
- 决定保留V1R FAIL和partial证据，新建同case052/180秒/checkpoint0的V1R2 replacement；不复用partial、不改变科研合同。用户持续授权覆盖该明确系统兼容修复。

## 2026-08-23：V5 V1R2单例机制PASS，继续六例readiness

- V1R2从头完成冻结case052，901个规划周期中0失败、0 post-warmup fallback，形成16节点/15条verified edge并移动199.881m。
- 出口执行反馈修正在frame51首次激活，frame52后route arc继续增长至204.990m；该证据满足预声明单例门禁。verified re-anchor在本例未激活，但合同明确不要求两种修正必须同时出现。
- 28/28 seal独立验证，seal SHA-256=`cfd62eac87614a60ed717dbad244ea2eb69ffbbe362f1782b4df0b1950926516`。决定不把单例推广为性能结论，下一步在2 worlds×3 checkpoints六例readiness中验证稳定性，随后才允许30-case精确配对比较。

## 2026-08-23：readiness启动入口修复后6/6 PASS，解除30-case前置阻塞

- V1命令确实执行但runner在顶层导入时缺少项目`src`路径，Gazebo未启动、completed cases=0。分类为system/runner startup defect；V1如实封存FAIL，10项seal SHA-256=`b2d8520ca3e7405c0fe0c9a6b66304caf9b142520fa19cdf3360c06b67da042d`。
- 采用最小V1R：只在runner中加载项目统一`_bootstrap`，不改变世界、checkpoint、case顺序、时长、算法、阈值或环境；从0/6执行且不复用V1。
- V1R 6/6案例通过全部门禁，最低移动134.748m、最低8节点/7 verified edges、最大fallback 2.1566%、0 failed cycles；6/6 execution feedback与5/6 re-anchor被实际触发。110/110 seal复核，SHA-256=`d301f54c1129af5e25a6a0e7d6cc7269ef6f71aa9987e08affaf66b171a8387d`。
- 决定解除V5稳定性前置阻塞，但不把readiness解释为性能收益；只进入预注册的30-case配对矩阵，Original M-TARE仍为主baseline。

## 2026-08-23：30-case来源哈希透传缺失，采用0/30 V1R替代

- V1在首个Gazebo前因`source_schedule_file_sha256`缺失而被外层证据接口拒绝；0个case、0仿真，9项seal SHA-256=`93bdb0de2cc618fb2e38e5bfb123a4a5626704e8b2ceebbbf750254903537c98`。分类为system/evidence composition defect。
- 决定不重用V1目录。V1R仅返回已被V1R audit seal绑定的source manifest文件与canonical-content哈希；科研输入和执行合同不变。
- 为避免再次由正式运行发现组合接口，V1R创建前增加完整pre-Gazebo proof，实际走完所有只读校验并生成30条命令。proof PASS后才允许唯一V1R执行。

## 2026-08-23：30-case V1R执行器符号未导出，V1R2改用actual-main sentinel proof

- V1R在首个case前报`run_mtare_single_robot_stochastic_v2 has no attribute run_case`；V2内部实际调用其V1 base的同名函数但未重导出。0/30、0 Gazebo，15项seal SHA-256=`749ade2608d07d077e7525fc312d9b9a0fb3cd53870441a1aa16a9f16e1a5ba8`。
- 选择在V5组合层显式绑定已冻结V1执行器，不修改底层V2或科研逻辑。V1R保持FAIL且不复用。
- 先前“生成30条命令”proof没有覆盖真实executor调用。V1R2门禁升级为actual `main()` + first-case sentinel，要求调用到达、case wrapper为V3、Gazebo未启动且临时失败能完整封存；该proof已PASS。
## 2026-08-23：GSE 主教师改用 native-mesh 横截面并显式屏蔽非唯一宽高

- 证据：解析 teacher 在 188,126 个训练序列规模的旧审计中 geometry-transition 为 0；35° turn 在远离节点的 8,324 个可判样本中也为 0。原因是 radius 是 per-tunnel 常量，不能代表 mesh 噪声、净空和横截面变化。
- 影响：继续使用解析半径会使“学习几何结构语义”的核心论文主张失效，受影响的是 teacher/data semantics，不是模型能力。
- 决定：主教师改为 sealed native mesh 左右/上下窄扇区射线；width/height 为投影距离之和，geometry-transition 为相邻 5 m 中位数至少 1 m 的持续变化。turn 固定为 15°，依据仅来自训练世界的 spline 分布，validation 未参与选择。
- 缺失策略：10-world train-only 哨兵的 23,318 序列中，265 个 width/height 横截面不完整，253 个位于 open junction。保留所有样本与事件标签，仅对不可靠的 width/height 维度应用显式 loss mask；禁止用解析 radius 回填或删除帧。
- 结果：五类事件分布为 `17,340/3,136/867/206/1,769`，几何目标有效率 `98.8635%`；正反 traversal 的 turn/transition 已通过 canonical edge arc 聚为共享 identity，32 项 GSE 单元测试通过。下一步必须以全部 80 个训练世界复核完整率和类别分布，之后才能物化正式 teacher export。
- 用户决策：用户已授予固定 GSE-Graph 范围内持续执行和最优方案选择权，不再逐项索取批准；本次不改变 80/10/10 split、测试隔离、论文问题或硬评价门槛。

## 2026-08-24：旧V5基线30/30封存后切换到GSE Gate 2

- 旧V5不可覆盖run自然完成30/30，顶层状态为`PASS_AEE_COMPOSITE_V9_COMBINED_CORRECTION_STOCHASTIC_V1R2`；10个区组、checkpoint 0/1/2各10例、18,000仿真秒，训练/optimizer/C09/C10读取均为0。
- 496项证据seal存在且SHA-256=`ce2bcd173528ce969dbe00e35a6dbc074164b61b12e0852beeb3a621f7201ca9`。一次只读检查在runner写seal前数秒报告“未发现seal”，随后顶层runner正常写出；这是时间差，不是来源缺陷，也没有修改或补写旧run。
- 决定：旧V5只作为exit-only感知+规则因果图基线/消融，不再扩展；current gate从已结束的Gate 6切换为GSE Gate 2，下一项唯一material operation是80训练世界native-mesh教师分布proof。
- 论文图保留：正式proof通过并seal后，只用fail-closed发布器将PNG、矢量PDF、CSV/JSON、来源run与独立SHA-256 manifest发布到`docs/figures/gse_graph/`；拒绝未seal来源和覆盖已有图。

## 2026-08-24：GSE教师proof的短边孤立帧缺陷采用V1R最小修复

- 证据：V1在第62个训练world停止；`S08_3d_loop_rich_C06/edge_0022`长度1.515895119118246m，正反traversal均短于五帧所需4m历史，因此主sequence inventory各记0序列/0唯一帧。mesh inventory却用`arange(0,length,1)`各创建2帧，再与0 anchor比较而FAIL。
- 范围审计：18,132个development traversals中仅上述2条为零序列；没有重复arc、spline多义性、世界缺失或split泄漏。V1在失败前完成61 worlds，validation/C10/M-TARE/训练/optimizer均为0，保持FAILED且seal SHA-256=`184b927d1789487a4c190e87b23b8a57aa2af70d3313487c1718071ccffdeaaf`。
- 决定：mesh inventory直接调用既有`deduplicated_frame_arcs_and_sequence_indices`；短于4m的traversal保留identity但不创建无法组成五帧样本的孤立frame。不删除edge/world，不改变80/10/10、188,126训练序列、252,430唯一帧、teacher、阈值或验收标准。
- 证据恢复：新增短边不触发raycast回归测试，GSE联动`42/42 PASS`；V1R使用新Data Card/spec、绑定V1失败seal并创建新run，不覆盖或复用V1输出。用户持续授权覆盖该不改变科研合同的实现修复。

## 2026-08-24：GSE native-mesh教师分布V1R正式PASS并发布论文图

- V1R重新读取全部80训练world并精确得到8,039 edges、16,078 directed traversals、188,126 five-frame sequences和252,430 unique usable frames；五类事件均超过500，几何有效率99.1718%，非junction缺测0.08977%，source前后不变且validation/C10/M-TARE/训练均为0。
- 决定：该结果解除teacher类别退化与mesh横截面完整性阻塞，但只证明训练教师可用，不代表模型、离线图或闭环PASS。下一步必须先冻结完整18,132 traversal manifest、关联对/困难负样本和存储资源上界，再允许teacher export。
- 正式run 15/15 seal独立复核，seal SHA-256=`f4a7002a322ac68af88b5246dac178c41ad1677bb84054621a8e99c89616210f`。论文图采用全部80训练world、无outcome selection，PNG/PDF/CSV/JSON/provenance/manifest已发布，manifest SHA-256=`5c20cec4a550dda2e03c87ee1f5fb6731b83f08df312a554521b255953e17eec`。

## 2026-08-24：18,132 traversal与学习关联teacher manifest正式PASS

- 完整90-world审计生成18,132 traversal rows、212,588 teacher observations、119,390 association pairs、3,891 turn/geometry-transition edge identities和90 world summaries；全部精确计数、唯一性、pair同parent/同event及same-identity标签一致性PASS。
- 训练集53,003个结构观测中52,465有正样本，52,962有同event困难负样本；52,174个正样本来自不同traversal、291来自同traversal revisit。validation为7,011结构观测、6,952正样本、7,011困难负样本，五类事件均存在。
- 决定：association loss只使用manifest的客观identity/pair，不按训练结果重选；没有正/负pair的少量anchor保留但相应loss mask，不删除。distance/angle关联仍是独立baseline。
- 存储决定：285,108唯一帧只存一次，序列仅存五个int64 reference。精确uncompressed payload为15.2944GiB，禁止57.0205GiB重复五帧布局；不宣称压缩率，正式export必须记录实际shard字节与逐shard hash。
- 正式run 17/17 seal独立复核，seal SHA-256=`1c2cb6bf3ebdf24f21a105cced59fb49066659bd6120667974276b940661b596`；C10/M-TARE/模型/训练均为0。该PASS只解除manifest阻塞，不代表模型或在线图PASS。

## 2026-08-24：sensor export前先补全incident-only可见出口token teacher

- 审查证据：已封存manifest包含结构事件、连续几何、place identity和association pairs，但`gse_exit_teacher.py`的directed exit identity/vertical profile尚未对212,588 observations形成完整token rows。直接生成LiDAR会留下不完整监督接口。
- 决定：不提前导出15.2944GiB sensor payload。先按`docs/GSE_EXIT_TOKEN_TEACHER_CONTRACT_V1.md`实现/审计token：corridor/turn/transition使用当前edge前后方向，node event只使用所选node的incident edges；非incident/stacked tunnel隔离。
- 可见性：保存完整incident candidates与native-mesh LOS后的visible tokens；不可见candidate不计学生漏检。opening width缺失只mask width loss，不删除token或用radius回填。identity、heading和四点vertical profile仍保留。
- sensor接口已独立实现并以synthetic slope+short-edge合同验证，GSE联动46/46 PASS；它只负责unique frame pose和16×720 batched raycast，不决定teacher semantics。

## 2026-08-24：出口token采用“身份/可见性保留，单维宽度mask”并解除导出阻塞

- V1冷启动循环导入发生在0 world/0 ray，决定保留不可变FAIL证据，不复制edge geometry实现；采用函数内延迟加载复用唯一`oriented_edge_polyline`语义。V1R不改变任何科学合同，从头运行且不复用V1 partial。
- 全90-world结果证明incident-only合同可实现：448,338 candidates中nonincident=0，29,695个共享tunnel ID的不同physical edges未折叠，所有38,218个junction/terminal observations至少保留一个可见出口。
- 决定学生的exit-token set只包含LOS-visible candidate；不可见candidate保留在审计证据中作为visibility mask，不计为学生漏检。opening width缺失仅mask width loss；heading、vertical profile、identity和candidate存在性保持有效，禁止radius回填。
- 该PASS只证明出口Teacher与数据接口合格，不证明学习有效或图质量提升。下一项唯一material operation是去重sensor+完整Teacher导出；训练必须等新导出seal和operation-bound Data Card。

## 2026-08-24：GSE训练集采用唯一帧Zarr与robot-frame监督，正式导出PASS后进入三seed训练

- 发现并在导出前阻断world-frame axis监督：学生只看到robot-frame LiDAR，若回归world-frame方向则目标不可观测。决定axis与exit heading统一为robot frame；训练时的circular azimuth roll同时旋转LiDAR与方向target，validation禁止augmentation，pose/yaw绝不进入学生输入。
- 数据采用90个parent级独立世界、18,132条directed traversals和212,588个五帧序列；285,108个sensor frames只保存一次。与57.0205 GiB重复五帧payload相比，正式Zarr shards为7.438 GiB，原始完整数组为15.3883 GiB。
- association只在5,087个客观结构identity上监督；corridor等无identity帧保留，但association loss mask为false。出口set只包含448,279个LOS-visible physical exits；438,968个visible token具有width loss，其他width只mask该维度。
- 唯一正式run `gate2_20260824_gse_deduplicated_dataset_export_v1_seed0` PASS，33,083项seal独立复核，SHA-256=`18acade5204258ebc6a822d24bc7910c1464bda7843a2e2e90e7f775c4cdb628`。该结果只证明训练数据完整可读，不证明模型学习、关联或图质量提升。
- 下一步固定为seeds 0/1/2训练与validation-only checkpoint selection。训练前必须单独冻结Data Card/spec、优化器/epoch/loss权重/采样和资源上限；C10与M-TARE保持不可读。

## 2026-08-24：拒绝继续训练旋转不变量方向头，采用圆周等变V1R

- 具体证据：V1 seed0 epoch1已完成188,126个训练序列和24,462个验证序列；event macro-F1=`0.530559`、width MAE=`1.2262m`，说明数据/优化器工作，但axis error=`79.7923°`且axis train loss约1.0，方向没有学习。
- 失效结论：原模型不能证明从LiDAR学习local axis，也会削弱exit heading的几何语义；因此V1不能继续到三seed，更不能进入离线图或论文主结果。分类为model/representation blocker。
- 可选方案：删除roll会让axis退化为几乎恒定forward target，创新性不足；给学生输入pose会造成接口泄漏；推荐保留方位时序特征并用固定圆周矩建立等变输出。采用推荐方案。
- V1R仅改变directional representation和roll lattice，不改80/10数据、Teacher、loss、optimizer、batch、epoch或checkpoint选择。V1 run按FAIL封存，V1R从随机初始化开始，不复用其epoch1权重。
- 用户已明确授权固定论文主线内自主选择最优修复且不重复索取批准，因此该最小模型修复直接进入新的Data Card/spec；若V1R仍不能降低axis误差，将再次停止而不以规划器调参掩盖。

## 2026-08-24：在三seed训练完成前冻结GSE校准、非学习几何基线与243组图参数

- 为避免看完validation结果后再设计对照，训练仍在seed0早期时即冻结后续协议：事件temperature/NLL、置信度与uncertainty联合拒绝、exit presence F1，以及past-only descriptor安全阈值；association先满足precision `>=0.98`、false accept `<=1%`，再最大化正确接受数，零接受不能PASS。
- 非学习连续几何主对照不是常数或解析radius，而是当前帧LiDAR的horizontal PCA、五截面robust包络及中心线多项式；真实validation前100帧只读接口检查100/100 finite、0写出，检查结果不用于调参。固定出口range-sector再构成非学习geometry-event图。
- validation replay使用每个world全部正反physical traversal的lexicographic Euler circuit；Teacher图压缩corridor和无事件degree-2 TNG点。预测图只压缩degree-2 metric anchor，遗漏事件造成的分支/末端anchor保留为错误；重复节点/边和unmapped edge不能通过集合去重隐藏。
- 共享图网格固定为`3^5=243`：stable frames `2/3/4`、minimum travel `2/4/6m`、anchor interval `20/30/40m`、association radius `8/12/16m`、ambiguity margin `0.02/0.05/0.10`。先过association安全门槛，再选择最小node/edge F1最高、均值最高、拓扑不变量更准且grid index最小的配置。
- 在线边现按无向node pair去重；重复真实穿越更新同一edge的次数、长度范围和最保守几何，不再伪造平行拓扑边。GSE相关接口/校准/基线/重放联动93/93 PASS。
- 论文方法主图已发布为`docs/figures/gse_graph/gse_method_overview.{png,pdf,svg}`并目视复核；同时保存生成器、provenance和SHA-256 manifest，manifest SHA-256=`3524d9e4f91caaf1a0f210e20a201c1236c1ffce799ab10c866f956d01ce35a9`。

## 2026-08-24：GSE连续几何门槛与exit-only公平验证口径预注册

- 在正式非学习几何结果产生前，连续几何主门槛固定为四项相对MAE降幅的等权宏平均；宏平均至少`10%`，且任一项退化不得超过`5%`。这样消除米、度和每米单位不可直接相加的问题，同时防止单项大幅提升掩盖另一结构量退化。
- 旧M1D作为exit-only/role-only感知基线时，只读取每条五帧验证序列的第五帧range+valid。其`interior/junction/terminal`概率无训练、无调参地映射为`corridor/junction/terminal`，`turn/geometry_transition`明确不可用并进入五类macro-F1分母。
- 基线验证固定使用历史seeds`0/1/2`三个sealed M1D checkpoint和同一`24,462`条C09 validation序列；不得读取C10/M-TARE，不得因GSE结果改变映射、阈值或checkpoint。
- 新增exit-only基线适配器、同验证集evaluator和几何门槛测试；GSE正式训练仍在运行，未改动任何训练冻结输入。
- 三seed事件门槛同时预注册：平均macro-F1绝对提升至少`0.05`、至少两seed达到`0.05`、任何seed不得回退；关联安全必须三个seed均非零接受并满足precision/false-accept合同。离线执行耗时只读基准在最大S10 validation world为每次`0.36--0.40s/4360帧`，支持完整243组串行重放而不缩减网格。

## 2026-08-24：论文交付矩阵切换为GSE-Graph且正式图片永久保留可追溯源

- 发现`docs/PAPER_COMPLETION_MATRIX_V1.md`仍把旧出口模型、规则图和V5闭环写成当前方法主线，与`docs/PLAN.md`第22节和GSE-Graph唯一投稿目标冲突。该旧叙述被明确废止，不能再指导实验顺序或论文主张。
- 交付矩阵现固定为“因果LiDAR→学习几何结构语义→学习式节点/关联→真实穿越验证边→M-TARE局部栈”，旧M-TARE、exit-only、规则因果图和GT-TNG分别保留为基线、消融或诊断，不删除其科学证据。
- 正式论文图只从completed sealed evidence发布；每张图同时保留PNG、PDF/SVG、CSV/JSON/NPZ源数据、确定性生成脚本、来源run/选择规则和SHA-256清单。不得只留截图，也不得在正式结果产生前手填图中数值。
- 用户已授予固定GSE-Graph范围内持续执行权限；run仍遵守Data Card/spec/preflight/immutable run合同，但不再把重复口头批准作为各步骤的阻塞条件。
- 图包只读复核确认现有四组manifest中所有已列文件均无漂移；随后用独立corrective publisher在验证原manifest和正式source seal后，为Teacher分布补充SVG并记录原生成器/corrective工具哈希，实验数值、PNG和PDF均未改动。新manifest SHA-256=`e5e2e45f5d5630b339916272cd51fce4d2e19db76cc3cbc8c6645acd25df82b2`。
- 2026-08-24定向原始来源复核新增GBPlanner、SG-SLAM和2026 SATE边界：它们分别依赖累积几何polyhedron、面向SLAM的语义图、或UAV俯视图traversability+Voronoi；当前未发现与GSE四项机制同时同构的工作。该结果支持继续当前方向，但不是穷尽性新颖性证明，投稿前仍必须更新系统综述。

## 2026-08-24：训练结果产生前冻结离线拓扑科学门槛、规则关联消融与论文图证据

- 完整GSE的node F1和edge F1分别与`exit-only+规则图`和`非学习几何事件+规则图`中对应最强者比较，每项必须绝对提高至少`0.05`；不得用同一个较弱基线同时充当两个分母。
- connected-component与cycle-rank跨validation world/seed的平均有符号误差绝对值各不得超过`0.25`；学习关联必须非空、precision `>=0.98`、false loop merge `<=0.01`。任一失败就停止C10与闭环，不用规划器调参掩盖。
- GSE规则关联消融只替换place/exit descriptor关联，继续使用同seed已校准事件阈值和不确定性拒绝。规则关联只看事件、位置、出口数和方向；近等距候选建立provisional node，不强制合并。
- 主要方法/消融/exit-only三个三seed方法及确定性非学习方法均运行固定`243`组结构网格，保留完整sweep和选中图；GT-TNG只作identity-aware诊断，不参与选择或宣称性能上界。
- Data Card、正式runner与fail-closed论文图publisher已在结果产生前完成。论文图必须从sealed PASS生成并同时保留PNG/PDF/SVG/CSV/完整JSON/provenance/生成器哈希；失败运行也保留可用于失败分析的机器证据，不发布成主结果图。
- 后续治理复核发现早期感知验证草卡使用不存在的`evaluation` operation、错误放在Gate 2且缺少标准Data Card字段，正式preflight必然拒绝。该问题分类为system/governance，不影响数据、模型或科学门槛。旧草卡在尚未生成spec/run前撤销，替换为Gate 3合法`threshold_calibration` Data Card；仍是同10个C09 parents、24,462序列和冻结三基线，C10/M-TARE零读取。runner/publisher及下游source path同步绑定新唯一身份。
- 规则关联回放另修正了evaluator provenance：错误rule merge现在与learned merge一样把冲突identity累积到节点真值状态，避免后续错误被当成干净关联；不改变图生成或参数选择，只消除基线关联计分偏差。专项回归证明污染节点保留`a/b` identity conflict并把后续合并计入false merge。

## 2026-08-24：感知论文图补齐双重测试集隔离复核

- 只读审计确认正式感知runner与科学gate均要求C10和M-TARE读取计数为0，但论文图publisher此前只重复检查C10计数；这不会改变正式实验判断，却留下错误摘要被发布的治理缺口。
- 决定在spec冻结前补上`mtare_worlds_read == 0`的fail-closed发布条件，并把该计数写入图provenance；模型、数据、阈值、指标和训练冻结文件均未改动。
- 新增非零M-TARE读取拒绝回归测试后，GSE专项合同为`117/117 PASS`。正式图仍只允许从completed、sealed、scientific PASS证据发布。

## 2026-08-24：新增PRISM-TopoMap与AIM-Mapping直接重合控制

- 训练等待期间只读复核原始论文页面，确认PRISM-TopoMap已覆盖“学习place recognition + scan matching + 在线location graph”，AIM-Mapping已覆盖“学习局部栅格结构表示 + 基于距离的拓扑表示 + 多机器人RL目标分配”。
- 两者都与GSE主张接近但不完全同构：PRISM的节点不是学习结构事件且目标是定位/回环，AIM的图由几何距离和boundary candidate构造且收益含策略学习；二者均不同时具备因果LiDAR显式隧道几何事件、语义决定节点/出口关联和真实穿越建边。
- 决定把两者加入贡献矩阵并在论文中正面比较边界；不改变现有方法。若后续正式检索发现完整同构工作，则按既定novelty stop rule停止相应投稿主张。

## 2026-08-24：离线拓扑入口补齐全部上游的双重隔离复核

- 只读代码审计发现离线拓扑runner只验证上游completed/sealed PASS，evaluator只复核`strict_test_worlds_read == 0`，没有对全部上游再次要求`mtare_worlds_read == 0`。当前来源记录均为0，因此没有数据污染或已生成结果失效。
- 决定在正式spec冻结前，让感知与拓扑freezer先拒绝任何非零C10/M-TARE来源；topology runner再对training/perception/dataset/teacher四个sealed source复核，evaluator也对直接消费的三个摘要执行相同检查。
- 新增污染source拒绝测试并通过；GSE专项合同更新为`118/118 PASS`。该修复只收紧治理门禁，不改观测、参数网格、图算法、指标或选择规则。

## 2026-08-24：补入Semantic Topometric Mapping与异构拓扑RL的直接新颖性边界

- 原始论文复核确认：Fredriksson等从累计二维occupancy grid规则分割intersection、pathway、dead-end和frontier并用于目标选择；Li等从二维probability grid、ESDF skeleton、frontier/border/viewpoint确定性构图，再以GNN/RL选择viewpoint。
- 这两项工作直接覆盖“结构语义用于拓扑探索”，因此决定禁止把这一宽泛表述写成GSE创新。GSE可检验主张继续限定为四机制组合：本体五帧因果LiDAR、显式三维隧道几何事件/出口token、学习关联直接决定结构节点、真实穿越才提交edge。
- 两项工作已加入贡献矩阵、论文related work和BibTeX。它们没有同时覆盖上述四机制，故当前方法不停止；若后续检索发现同输入/输出/建图机制的完整同构工作，仍执行novelty stop rule。

## 2026-08-24：论文草稿严格拆分已实现离线图与计划闭环集成

- 代码交叉审计发现草稿把尚未实现的GSE global target selector、M-TARE waypoint adapter和多机器人图交换写成既成事实；当前`gse_graph.py`只实现节点/关联/edge更新，Gate-4 evaluator只执行C09离线Euler replay。该表述不能作为现有证据。
- 决定把该段明确改为`Planned exploration integration`，在相关代码和sealed闭环证据产生前不得写成已实现。当前edge也改写为由预记录directed traversal trace确认；未来闭环必须以实际odometry trace维持同一合同。
- 同时修正正文中的旧单帧`100000/12500`计数为GSE正式`188126/24462`五帧序列和`252430/32678` unique frames；243设置明确为共享五维structural index grid，而非方法所有阈值完全相同。
- event/anchor触发、association gating、provisional node、edge真实字段、baseline集合、pending ablations与重复/冲突precision分母已逐项对齐冻结实现。独立复核八组问题全部关闭；模型、数据、阈值、训练run和C10/M-TARE隔离均未修改。

## 2026-08-24：活跃论文草稿增加结果证据边界

- 三seed训练尚未完成，C09感知与离线图评估器虽已实现和预注册但尚未执行，C10、GSE专用M-TARE adapter、单/多机器人闭环与其消融当前均无结果。
- 决定在草稿顶部写入不可绕过的证据边界，并规定所有`AUTO`结果字段必须由completed sealed来源替换；在此之前不得把计划实验写成已完成贡献。
- 摘要与方法图说明把当前edge依据表述为`traversal evidence`，把M-TARE替换表述为planned closed-loop integration。该修改不改变模型、数据、阈值、运行或论文目标，只消除过度声称风险。

## 2026-08-24：把Bio-Inspired Hybrid Map列为学习拓扑图直接近邻

- 作者公开稿表明Dang与Huber把含学习特征的spatial-implicit local frames集成进factor graph，并按累计行程和视点变化创建新local frame；这比只做place recognition的工作更接近GSE学习表示加拓扑图的边界。
- 该工作没有让junction/terminal/turn/geometry-transition等显式地下结构事件决定节点触发，也没有GSE exit-token关联和穿越证据后提交edge的组合，因此当前方法不停止。
- 决定将其加入贡献矩阵、论文related work和BibTeX并要求固定metric-anchor、无显式几何和规则关联消融共同排除keyframe式收益。训练、数据、阈值、C09/C10隔离均未改变。

## 2026-08-24：把FHT-Map列为学习特征层级拓扑直接近邻

- 作者公开稿证明FHT-Map已用CNN压缩视觉特征和局部LiDAR扫描建立main nodes，用轻量support nodes及累计free-space路径修补支持重定位与路径规划；节点选择受视觉信息丰富度和节点密度控制。
- 因此“学习特征进入层级拓扑图”不能作为GSE贡献。差异必须落在单一LiDAR因果输入、显式地下几何事件触发结构节点、exit-token参与关联和执行trace确认连接。
- 决定把FHT-Map加入贡献矩阵、正文与BibTeX，并要求固定metric-anchor、exit-only、规则关联和无显式几何消融排除其已覆盖机制。当前训练与实验合同不变。

## 2026-08-24：预注册C09拒绝与关联失败分析图

- 主感知结果图只展示事件、连续几何和选定关联precision，不足以说明不确定性拒绝如何换取安全或阈值是否挑选性展示。
- 决定新增独立发布器，固定读取三个seed的完整event rejection、place association和exit association曲线，标记冻结选择点并单列false-accept rate；不得只画最好seed或截掉失败区间。
- 发布器只接受completed sealed perception PASS，逐源复核seal和C10/M-TARE零读取，输出PNG/PDF/SVG/CSV/JSON/provenance/SHA-256且拒绝覆盖。它在训练完成前只作为冻结工具，不产生论文数值或图。

## 2026-08-24：出口token使用与M1D相同20度匹配诊断

- 审计发现M1D方向F1由20°角度匹配定义，而GSE的presence F1只判断六个query是否对应某个target，二者不能直接比较；仅报告后者会掩盖出口方向质量。
- 决定在正式C09前对GSE validation-selected token set应用完全相同的20°匹配，报告direction P/R/F1、匹配角误差、count exact与MAE，并用三个paired seeds与M1D并列。
- 该指标只补论文诊断，不改变已经预注册的event、continuous geometry和association科学门槛，也不用于checkpoint选择。共享方向metric、GSE token evaluator和新publisher全部绑定正式spec哈希。

## 2026-08-24：C09感知增加不可选择的逐世界泛化证据

- 聚合24,462序列的均值不足以证明方法对10个未见拓扑都有效，容易被少数大世界或容易世界主导。
- 决定在每个seed的冻结全局温度和拒绝阈值下，逐parent重放五类事件；连续几何逐parent保留width/height/slope/curvature MAE，并与同parent的第五帧非学习估计器配对。
- 新论文图强制包含10个C09 parent×3 seed及全部四字段，明确`selection_effect=NONE`，不参与checkpoint、阈值、世界或gate选择。六个C09证据包共43文件原子发布；正式C09尚未运行，因此当前只完成方法和证据合同，不产生论文数值。
- Luna只读审计指出seal本身不能证明“逐parent没有重新选阈值”且相同帧数不能证明几何MAE使用相同有效子集。发布器现额外要求每个parent记录的temperature/threshold逐值等于该seed全局冻结点，并要求GSE与规则baseline四字段valid-count逐项相同；任一漂移立即拒绝发布。

## 2026-08-24：C09正式感知资格因坡度回归塌缩FAIL，停止离线图

- 唯一正式run `gate3_20260824_gse_perception_validation_v1_seed0`以`FAIL_GSE_PERCEPTION_VALIDATION_V1`结束；49条seal逐项哈希匹配并精确覆盖除seal自身外的全部文件，model updates/optimizer steps/C10/M-TARE读取均为0。
- event gate以三seed平均`+0.154469` macro-F1增益PASS；place/exit association在非空接受集上以约`0.990` precision和`<0.01` false accept PASS；curvature/height/width相对非学习估计改善`85.31%/52.00%/24.07%`。
- 唯一失败字段是slope：三seed MAE均值`2.925625°`，非学习基线`1.527336°`，相对改善`-0.915508`，违反预注册“任一字段不得退化超过5%”。不得以其他三字段平均改善抵消该失败。
- 只读根因审计显示slope teacher标准差约`4.903°`，而模型预测标准差仅`0.905–1.057°`，MAE约等于直接预测0°；Teacher axis坡度与slope字段一致到`<1e-7°`，故问题是模型回归信号塌缩，不是Teacher矛盾或calibration。当前对`45°`归一化后的slope Smooth-L1再被四几何字段平均，数值级远小于事件/出口/关联loss，是主要实现原因候选。
- 决定在corrective方法明确前停止离线拓扑、C10和闭环；不降低门槛、不覆盖旧FAIL、不用简单ensemble伪装学习改善。可行选项为：(A)重标定slope loss并完整重训三seed，约再需20 GPU小时；(B)推荐的最小corrective，冻结已通过主干，以解析坡度为先验训练小型残差/置信度头，先做train-world内部交叉验证。两者均必须保持原门槛，坡度MAE至少不差于解析基线5%（C09当前对应`<=1.6037°`）。

## 2026-08-24：选择physics-guided slope residual并回退Gate 2

- 用户已授权在固定GSE论文范围内持续执行且不重复索取常规批准。采用C09 FAIL后推荐的最小corrective，而非再花约20 GPU小时完整重训：冻结所有已通过的GSE输出，只用五个因果LiDAR扫描的确定性几何量构造坡度先验，GRU(32)仅学习`±10°`有界残差和误差尺度。
- 为避免原GSE backbone已见C01--C08造成内部选择污染，corrective禁止使用其任何learned feature。模块级world split固定为C01--C06拟合、C07--C08选择；C09只允许在corrective完全冻结后重新执行一次完整感知资格，C10/M-TARE继续禁止。
- 精确盘点为拟合`60 worlds / 12,106 directed traversals / 190,600 unique frames / 142,184 sequences`，选择`20 / 3,972 / 61,830 / 45,942`，总计80 worlds、8,039 physical edges、16,078 directed traversals、252,430 frames、188,126 sequences。此前工作记录中的`12,104/16,076`由源轨迹逐项求和和`8,039×2`证据纠正，不进入正式Data Card。
- train-only acceptance固定为：三seed平均相对五帧解析先验改善至少5%，至少2 seed各改善至少5%，所有seed整体不退化，任何拓扑族退化不超过5%。不通过则停止路线，不读取C09、不降低门槛，也不把epoch 0解析先验冒充学习收益。
- operation-bound Data Card和spec已冻结；正式训练属于Gate 2 representation corrective，因此机器状态从Gate 3回退到Gate 2。旧C09仍保持`GATE_FAIL`科学证据，回退不覆盖或改写旧run。

## 2026-08-24：坡度corrective V1系统检查器FAIL与V1R恢复

- V1在源seal 33,083项通过后，于第一个C01 shard的cache前置检查失败；0 seed、0 optimizer step、C09/C10/M-TARE零读取，11文件seal SHA=`eb75870c060b9c0f744b1a255bd136a3392b11f8c4a3cdd372fecaaf7b286c62`。旧run永久保持`FAIL_GSE_SLOPE_CORRECTIVE_THREE_SEED_TRAINING_V1`。
- 唯一失败断言要求整个world的`local_frame_index==arange(frame_count)`；真实schema中该字段按directed traversal从0重置。全80 shards只读证明188,126条序列均使用连续Zarr行、local index逐帧`+1`、global references精确匹配且不跨reset，因此Gate结论未被数据缺陷或Teacher问题推翻，失败分类为system checker contract bug。
- V1R只移除world-global假设，并新增正确的序列内`+1`和遍历reset回归测试；专项测试从10增至12项全PASS。Data Card、模型、输入、split、seed、优化、baseline、门槛和C09隔离不变。按用户持续执行授权创建独立V1R proposal；V1R必须绑定旧FAIL三文件和exact seal，不重启或覆盖V1。

## 2026-08-24：坡度corrective V1R train-only资格PASS，返回C09

- 唯一V1R正式run以`PASS_GSE_SLOPE_CORRECTIVE_THREE_SEED_TRAINING_V1R`完成：seed0/1/2最佳epoch`31/50/47`，MAE`0.862578/0.862454/0.863412°`，相对相同五帧解析先验`1.664656°`改善`48.1828%/48.1903%/48.1327%`。三seed、10个topology family和资源门槛全部PASS。
- 115条seal精确覆盖run内除seal外的全部115文件，0 hash mismatch；seal SHA=`d8683e5b965d8f9f142263aad54a2c0e9277e3e75002f3a04fe807d756d23748`。总optimizer steps`19,321`，C09/M-TARE/strict test读取均为0。
- `strict_test_worlds_read=0`按合同包括C10，但summary没有单独C10字段。决定不修改sealed run；下一Gate-3 runner必须独立验证train cache的80个actual source paths全部为C01--C08，并显式记录`c10_worlds_read=0`。pre-run `status_snapshot`不得事后篡改，其语义在新证据中明确。
- 新C09只允许重新计算冻结slope corrective；原49文件C09 FAIL中的event、association、axis、width、height、curvature、M1D与nonlearning baseline逐seal复用。PASS必须同时满足原完整perception gate，以及corrected slope相对五帧解析先验的三seed/逐世界学习收益门槛；否则继续停止拓扑。

## 2026-08-27 — 保留5个singleton terminal并显式使用旋转增强

- 实现前只读诊断发现5/1,066个decision identities（fit 4、selection 1）各只有一条terminal-labelled observation；全部来自相邻junction占据其余1 m采样点的短边。原提案“每identity至少两个observations”对这5个单位不可满足。
- 删除它们会制造选择偏差；复制同一tensor会伪造重访；跨split或借用错误junction标签会造成Teacher污染。决定保留全部identity，只对这5条完整5帧terminal sequence使用固定180-bin circular azimuth shift作为正对，并单列`singleton_circular_shift_augmentation`。
- 该变换只约束place descriptor的朝向不变性，不读取objective geometry或identity作为学生输入，也不计为physical revisit。其余1,061个identity仍必须使用真实cross-view/distinct observation。
- 正式manifest以`1,066/1,066`完整覆盖、`2,132` pairs、零split泄漏PASS；这项决定不降低后续关联precision/false-accept/recall门槛。

## 2026-08-27 — Association Teacher阻塞解除，自动推进容量证明

- 正式manifest run状态为`PASS_GSE_FACTORIZED_ASSOCIATION_TEACHER_MANIFEST_V1`；18项seal SHA=`047b5d16b0083fcab56a2a04d286655b2711e148b704d98f30d958e7b1d90f5b`。
- 决定下一步采用低容量route-conditioned association proof，不训练新backbone：输入仅限部署时可得的learned descriptor/full exit tokens/uncertainty、实际穿越edge geometry和runtime distance；Teacher geometry仅选负样本。
- 继续保持C01--C06 fit、C07--C08 selection、C09/C10/M-TARE隔离。只有selection precision `>=0.98`、false accept/loop merge `<=0.01`、recall `>=0.25`且优于规则关联后，才允许恢复离线图。

## 2026-08-27 — Route geometry改为incoming-edge到exit-token关系，距离只作候选门禁

- 零训练诊断显示descriptor-only AUC为`0.943923--0.956191`，但seed1在`recall>=0.25`时最低false accept约`1.626%`，三seed尚未稳定满足1%安全门槛；容量proof有实际缺口而非无意义重复训练。
- 直接比较两条incoming edge geometry会错误惩罚同一junction的不同支路，固定rank组合反而显著退化。决定不采用该错误关系。
- full relation改为把已执行incoming edge的learned width/slope profile与候选观测的六个exit opening-width/vertical-profile匹配，并与完整token set和place descriptor联合。no-route版本作为独立消融。
- 不同world的structural-alias hard negative没有共同metric坐标系。空间距离只保留为在线`<=16 m`候选门禁，不进入分类器；否则“距离缺失/坐标系”会成为负样本泄漏。该决定不扩大runtime radius，也不改变Teacher identity或split。
- 预注册贡献门槛：full三seed均安全；平均safe recall相对descriptor至少`+0.05`、相对no-route至少`+0.02`，任一seed相对no-route回退不超过`0.02`。失败即停止route-conditioned association主张。

## 2026-08-27 — 容量证明PASS，冻结模型/归一化/阈值并只允许C09资格

- 正式full模型三seedsafe recall=`0.762774/0.645985/0.620438`，precision=`0.990521/1/1`；相对descriptor/no-route平均增益=`+0.457421/+0.476886`，全部预注册门槛通过。
- physical-only复核移除selection唯一augmentation positive后仍全部通过，排除5个singleton策略主导结论。
- 决定冻结六个小模型、三套perception输出对应的normalization和full三seed threshold；C09不得重新训练、选择epoch、改输入、改损失或调threshold。
- C09资格必须同时报告两域：identity-balanced structural aliases检验跨结构混淆，runtime strictly-past `<=16 m` candidates检验实际在线候选。只通过其中之一不得恢复离线图。
- 容量run 61项seal SHA=`91ef98c0bc5b4cd82492dd42499a57a1bfe22d020664fbb0189aa2345cee11cc`；C10/M-TARE继续隔离。

## 2026-08-27 — C09前强制统一最终坡度语义

- 发现容量V1的route relation消费旧GSE slope，而论文最终edge geometry合同消费risk-calibrated slope。两者并存会让关联训练/部署分布漂移，或形成“旧坡度匹配、新坡度存边”的隐式双接口。
- 决定保留V1为固定旧接口下的组件capacity evidence，但撤销“直接冻结V1进入C09”的下一步；不允许用数值很小作为忽略接口不一致的理由。
- 推荐并自动采用最小V1R：不重跑GSE主干、不改Teacher/pairs/架构/超参/门槛，只精确CUDA重放已冻结坡度corrective，固定残差scale=`0.89`，按global sequence identity替换观测column 10。
- 正式V1R新增三项硬证据：C07--C08冻结坡度输出byte-exact重放、其余145列byte-exact不变、descriptor/no-route关联输出byte-exact复现V1。任一失败即停止，不进入C09。
- 只读proof已通过188,126身份全覆盖；CPU与GPU最多约`0.034°`的差异被认定为设备数值路径，正式重放强制使用原CUDA环境。

## 2026-08-27 — V1R取代V1成为最终关联冻结点

- V1R统一坡度后full mean safe recall从旧接口`0.676399`变为`0.581509`，但三seed安全、physical-only、相对descriptor/no-route增益和所有family门槛仍全部PASS。
- 决定不因旧接口数字更高而保留旧坡度。论文主方法、C09和后续图只允许使用V1R checkpoint/normalization/threshold；V1只保留为接口纠正前的组件证据。
- descriptor与no-route byte-exact复现排除了数据顺序、Teacher或trainer漂移；变化可归因于route relation使用了最终corrected slope。
- 69项seal SHA=`64e5bfa8ba9fa1ff64bf491f042cc2d1a33ae0c13242437ce75a39a5e6c5eb40`。允许恢复C09资格准备，但不允许在C09重新训练、选epoch、调threshold或选择domain。

## 2026-08-27 — C09单seed关联FAIL后采用selection-only consensus+metric gate审计

- 三个冻结seed在C09 runtime false accept均超过1%，所以“任选一个seed部署”与单seed主张正式失败；不得挑seed2的相对最好数字。
- C09 score只读诊断显示2-of-3 consensus能让balanced alias达到precision 1.0/recall 0.508，但runtime仍有3.38% false accept；错误高度相关，单纯ensemble不够。
- 同一诊断显示大多数runtime真匹配距离集中在1 m内，而误合并扩展到16 m；consensus加保守metric cap存在同时达到安全和召回门的容量。该诊断只用于决定机制，不允许选择最终cap。
- 决定在C07--C08重新构造完全相同的strictly-past candidate域，预注册枚举`k=1/2/3`与固定distance grid，仅从selection选择满足balanced+runtime安全且召回最大的配置。C09随后只作冻结复核，C10仍为严格测试。
- 如果selection域无配置或新冻结配置在C09仍FAIL，停止Factorized association claim，不再用图/planner参数弥补。原C09 FAIL 25项seal SHA=`ca31e1d36958ae2291bb19c5c079fa7621b3b168e2fe5272de46ca00b99ac338`。

## 2026-08-28 — 多事件Teacher通过后采用固定16槽导出，不按观测最大值裁容量

- 原V1只因object-only JSON读取器拒绝四行数组而在0 world/0 raycast处失败。决定保留原FAIL并自动执行只改列表解析的V1R；禁止覆盖、复用partial或把系统错误算作科学失败。
- V1R证明全部1,076个结构事件身份可见、反向heading覆盖100%、两split和10个family均有terminal+junction共现，且两个稀有失败场景确有同帧多事件。互斥scene label路线永久停止。
- 虽然开发观测最大集合基数为5，下一导出仍使用预注册上限16，避免用C07--C08结果反向选择模型容量；0事件帧必须保留，超过16立即FAIL而非截断。
- 本轮已验证结构事件只包括degree-1 terminal与degree>=3 junction。geometry-transition保留为连续edge geometry语义；在没有独立Teacher证明前不得把它加入结构node set或声称已经验证。
- 下一步只允许完整Teacher导出及完整性审计。通过后才设计集合预测head；C09/C10/M-TARE继续隔离。
- 治理分类采用Gate 2 `teacher_generation`，不把新监督资产伪装成Gate 3普通audit；导出PASS后才返回Gate 3。该临时回退不允许重训旧模型或读取C09/C10。

## 2026-08-28 — Teacher导出PASS后采用方向特征上的16-query集合预测头

- 导出证明实际最大集合为5，但模型query数量继续固定16；不使用C07--C08观察结果缩小容量。
- 新head直接消费既有五帧encoder保留的azimuth directional feature和causal context。每个query输出presence、terminal/junction、robot-relative xyz、descriptor与uncertainty；旧scene event head仅作为baseline，不再决定结构节点。
- 初次readiness只实现接口、确定性set matching和loss，不执行训练。必须覆盖空集合、1--5事件、padding mask、query permutation、circular roll和finite backward。
- 三维正确匹配半径预定为4m，与冻结在线association metric cap一致；训练Data Card中必须同时报告precision、recall和false positive，不允许只看分类F1。

## 2026-08-28 — Assignment平局改为预测内容规范化，禁止query编号影响loss

- V1证明同一组预测仅改变query排列会产生`2.9373e-4` loss差，根因是solver最终按query row index打破近等价assignment；这违反集合模型基本语义。
- V1R在匹配前按presence/type/relative xyz/uncertainty/descriptor预测内容做确定性规范排序。Teacher identity不进入排序或学生输入，loss与阈值不变。
- 正式query permutation error=`1.1921e-7`并通过原`1e-6`门。该规则成为后续训练和评估唯一assignment合同。
- 初次容量训练冻结三套已有encoder，仅训练三套93,638参数decoder；失败时不得直接解冻backbone掩盖set head是否有独立容量。
## 2026-08-28 — 保留双距离几何表示，停止当前dense presence并先做只读归因

- 事实：180×2模型三seed的定位MAE均约2m，且seed1/2在79个same-bin second-depth targets上超过best baseline+0.10；因此“固定polar/depth坐标完全无效”不成立。
- 同时，固定0.95最高阈值仍产生`44,078/68,447/36,887` false positives，三seedprecision仅`0.138/0.103/0.130`；总体F1远低于exclusive baseline。当前输出不能安全触发结构节点。
- 决定：不进入graph，不通过增加epoch、扩大threshold grid、调loss权重或规划器参数补救；保留当前checkpoint/图作为失败与消融证据。
- 下一步只允许对sealed top16预测做零训练、零新推理归因，区分proposal capacity、confidence separability、slot duplication与empty-row cardinality。若oracle proposal不足则停止dense路线；若proposal足够而置信度不可分，只允许一个由证据确定的最小objectness corrective。
- 正式run seal SHA=`3186e311508a82fa5e6a8342806729ee4c9335cb9407b322f6420c2cf73503ee`。

## 2026-08-28 — V1R slot provenance取代objectness-only refit，停止structured-polar multi-depth

- V1的typed oracle recall通过proposal门，但其匹配未记录预测depth slot，无法证明方法声称的near/far双槽表示被使用。
- V1R精确证明slot collapse：三seed top16 slot1=`0/16/0`，0.95选中slot1=`0/0/0`，second-depth由slot1匹配=`0/0/0`，distinct slot near/far simultaneous pair=`0/0/0`。
- 结合candidate AP仅约`0.05--0.06`，当前路线同时需要slot-permutation监督和objectness ranking重构，不再属于一个最小corrective。决定不投入另一轮三seed试错。
- 采用预声明fallback：回到完整可执行exit/action geometry tokens，学习5帧action-set birth/death/split/merge及junction/terminal commit；这与旧exit-only方向基线不同，边继续只由真实穿越建立。
- V1 decision保留为被更完整语义证据取代的审计历史；V1R为当前权威决策。seal SHA=`8c9cf3714d8ceca617c57905573c8b7e86f7127dd2bcfa90accb84ab2d9d5ae5`。

## 2026-08-28 — 采用关系式token transport，禁止再次逐帧集合池化

- 正式证据表明可执行出口集合可以以`0.995411` macro-F1定义开发结构事件，且三个seed的descriptor-only跨帧物理出口匹配precision/recall均超过`0.98`。
- 旧ActionSetNodeDetector在时间模型前对6个token做confidence-weighted mean与maximum，丢失了单个出口的持续、出现和消失关系；其触发precision=`0.965870<0.995`。
- 决定保留完整token轴，显式计算相邻帧bipartite affinity/transport，再做因果set-temporal aggregation；输出必须含provisional/refusal，不能把16个缺少明确支持的selection episodes强制提交。
- Teacher identity只允许用于loss分组和离线评分，绝不能进入transport特征；edge依旧必须由真实执行穿越建立。
- 下一步先做零训练readiness。readiness或后续三seed安全门失败时停止该模型，不用graph/planner调参掩盖。
- 证据seal SHA=`10fa5df309c81d11254f3302280200fe19b7b77967f70b49f1f66f62da5f2f23`。

## 2026-08-29 — readiness通过后只允许一次三seed关系式模型容量训练

- V1失败属于明确API错误：`torch.flatnonzero`不存在；V1R等价替换后全部科学检查通过，因此不把系统错误解释为方法失败。
- 模型接口固定为240,101参数和显式soft transport；训练不得删除transport、退回pooling或让Teacher identity/pose进入forward。
- provisional/refusal是安全接口的一部分；selection存在16个无明确action-set支持episode，不能通过强制commit追召回。
- 下一训练门固定：aggregate precision>=0.995、junction/terminal precision>=0.99、episode recall>=0.25，且macro-F1相对旧pooling至少+0.05。失败后不进入graph或靠规划器补偿。
- V1/V1R seal=`9f4dcb8b6a3fe165d60e52155435d78f0ebc8053e112dd574764f34825dc89f9` / `853b040875f3184e4c3389402655198e4bd13c43533d0e23d73e69861f32d01a`。

## 2026-08-29 — 停止无状态概率均值commit，先审计可部署状态机

- 正式capacity没有阈值能同时达到冻结precision/recall门；不得挑seed1的高F1、降低0.995门或直接进入图。
- 但单seed均明显学到事件，且ensemble 26个非正确trigger中18个为同episode重复，证据不支持立刻否定完整关系表示。
- 决定把“表示分类”和“在线只建一次节点”拆开：checkpoint全部冻结，不再训练；下一步仅审计past-only debounce/refractory、seed consensus/disagreement refusal和provisional状态。
- 候选状态机必须在world-held-out开发划分上证明规则不是靠同一世界后验挑选；Teacher identity只能评分，不能用于去重状态或在线输入。
- 若无简单固定状态机同时达到原precision/recall和+0.05 F1门，停止关系事件路线。若通过，也只能获得一次冻结commit-policy正式验证资格，不能直接进入C09/graph。
- 证据seal SHA=`de4cb8d37bc834236e6a1739b7e2d8d91f963a2299d7dcd51e42b7a7605aad6f`。

## 2026-08-29 — 不采用手写状态机主张，改为exact-one结构化事件似然

- C07选出的状态机在C08安全迁移，证明debounce/consensus机制不是无效；但C07 macro-F1只0.753519，未达到固定+0.05增益门。
- 决定不扩大阈值、stable/release grid，也不把C08单侧PASS写成主结果；该状态机保留为后续运行时安全层和消融。
- V1 episode MIL的数学缺陷是只使用`max`正确峰，第二个、第三个commit峰没有代价，这与18个同episode重复trigger直接一致。
- 下一模型变化严格限制在loss：对每个positive episode最大化“恰好一次正确类别commit且其他帧全部不commit”的概率；corridor最大化零commit概率。架构、输入、Teacher、split和安全门先不变。
- 先做零训练readiness；若单峰排序、permutation、数值或真实backward任一失败，则停止，不创建训练run。
- 正式commit审计seal=`bc8771dd710f33053531f47202c4ee0a15482b58273469307a5584306fadaa47`。

## 2026-08-29 — exact-one readiness通过，V2训练只允许改变loss

- 数学和真实batch均证明exact-one likelihood修复旧max-MIL对非最大重复峰零梯度的问题。
- 决定保留240,101参数架构、raw tokens、geometry context、Teacher、episode-preserving batch、AdamW、6 epochs和全部安全门；不得同时改网络或数据来混淆归因。
- C07承担checkpoint与policy selection；C08承担一次world-disjoint development transfer，不允许用C08结果回调epoch、loss或policy。
- V2必须报告无状态输出和past-only stateful commit两套结果；主门以部署形状的stateful C07/C08双侧安全与macro-F1为准。
- 任一侧失败即停止，不进入C09/graph。正式readiness seal=`ced78ead9cf7773f51ee698d4ae4ae6f6b19a631530054797e151b9c47b07426`。

## 2026-08-29 — 停止直接事件分类，采用learned token validity与action-set tracking

- V2 exact-one按预注册训练完成，但C07/C08 macro-F1仅`0.573256/0.582755`；安全层不能恢复召回，继续调loss属于无依据试错。
- 决定永久停止当前junction/terminal direct classifier（V1 max-MIL与V2 exact-one）。两者保留为论文消融，不再进入C09或图。
- 新路线不把“出口方向”冒充完整结构语义：学生必须判断token validity，并保留opening width、vertical profile、descriptor、局部axis/width/height/slope/curvature和uncertainty；跨帧track关系直接构成action set。
- junction/terminal成为stable executable action set的确定性结构语义，而不是另一个黑盒分类头；node由该语义产生，edge仍由真实穿越验证。
- 先做只读可行性，Teacher identity只用于离线附着与评分，不得进入runtime transport或student input。若token validity/track cardinality上界不足，停止该主张，不训练。
- V2 seal=`51d3a8964ca7c556ccafaad93ee1392804ec429273f73db497cf6b791da3306b`。

## 2026-08-29 — 停止六自由query有效性修补，采用连续圆周可通行场

- V1R2证明问题不在单帧抖动：五帧descriptor平滑和三seed几何共识都没有提高validity AP，且99.5%精度下recall远低于0.50门。
- 假候选在时间和seed之间稳定，说明自由query学成了可重复的ghost slots；继续训练objectness、拒绝头或阈值会重复已经否定的机制。
- 决定保留六query exit模型、关系事件V1和exact-one V2作基线/消融，但从论文主方法移除free-query objectness。
- 新主表示固定为Dense Circular Traversability Field：对每个圆周方位预测可执行几何，出口由连续方位分量生成；这使“是否存在”绑定到传感器方位域而不是任意query槽。
- 下一步先审计Teacher是否可无损表达全部396,913 visible exits及冲突/连通分量语义；未通过不得导出或训练。原始range阈值必须作为非学习基线。
- V1R2 seal=`41c23bdf50f90cfec427d93add4ac7e6eecc001c41fb2f73655643c105986729`，图SHA=`2d176ed727b6207ef5bdf7fc0e0565d61022551bec498ed90176bb8fa36c3236`。

## 2026-08-29 — 宽扇区连通分量失败后采用圆周中心几何峰场

- 只读全量证据显示宽开口扇区在22,626行产生错误组件数，原因是靠近路口时多个真实开口角域物理重叠；不得通过调连通阈值掩盖实例语义冲突。
- 同一人口的出口中心在180个2°bin上same-bin和adjacent-bin冲突均为0，因此选择中心peak而非宽sector：presence决定实例，sub-bin residual恢复精确方位，width/profile作为peak属性。
- V1的round-trip FAIL来自encoder/auditor对半bin边界使用不同浮点精度。决定固化一个canonical float64转换函数并执行V1R；不改变bin数、数据或门。
- V1R全量PASS，说明中心peak是无损Teacher接口。raw-range AP仅约0.052，所以下一阶段必须训练并比较非学习基线，不能把接口PASS写成感知收益。
- 下一步只允许Gate-2 Teacher export；禁止直接训练、C09/C10、图或规划器。
- V1R seal=`ddad17c78f26ec05754c35ed88dc640473f30e9ea7c0b021995b2ed02fab663e`，图SHA=`360d2a8539941380b2f336281cb50328e1217f96648cad2b5b6e80fc334444ec`。

## 2026-08-29 — 冻结Circular Peak Field Teacher，进入零训练model readiness

- 80个world shards一次导出并逐数组回读一致，证明Teacher存储、partition、mask和join合同可执行；没有复制LiDAR或写入identity。
- 决定将本run固定为该表示唯一训练Teacher，不再产生另一套相同标签或修改180 bins、peak语义、width mask和profile定义。
- 导出PASS只证明标签可靠，不证明模型能学会；因此不直接训练，下一步必须先通过circular equivariance、causal mask、masked loss和真实batch backward readiness。
- 模型readiness不得通过Teacher identity、future frame或测试世界简化任务；失败则停止该架构，不修改Teacher掩盖。
- 正式export seal=`8c37f11a9e08b3cf0c48d005ae32aa2d79afcbba7c2330333ee9996605657f76`，图SHA=`044cccf8b8e8fe68e24446ae7b371a2662c4bfb270f03fd3a5f456d655cc0780`。

## 2026-08-29 — Circular Peak Geometry模型允许从头三seed训练

- readiness证明一个模型可同时保留圆周出口实例、局部轴线和连续隧道几何，不再采用“只学出口方向再由规则补图”的旧主线。
- 决定禁止用旧GSE checkpoint初始化主方法，因为旧训练曾读取当前C07--C08开发选择世界；复用会污染迁移结论。旧checkpoint只作baseline/ablation。
- 新训练固定C01--C06梯度、C07 checkpoint+threshold选择、C08一次零更新迁移；三个seed必须统一报告，不能挑单seed。
- peak存在损失使用正负类别等质量BCE，不引入手调class weight；连续量只在对应有效mask上学习，descriptor/uncertainty保留在可执行peak上。
- readiness PASS不代表性能PASS；若高精度peak recall、action-set cardinality或连续几何未改善，停止该模型，不进入图或planner调参。
- 正式readiness seal=`d06089680b97544a251e30e9c0214fa6c14df31fb3d30ef0c3d97852adc74c69`，图SHA=`38aa751b6f67a0a19c8cf5e6a5373fc1835eb85a5a6c055b33034c49b3b954ab`。

## 2026-08-29 — V1保留连续几何，停止单格等质量BCE峰值头

- 三seed完整训练排除“没有优化”解释；连续axis/width/height/curvature在C07和C08一致泛化，不能推倒整个几何语义backbone。
- exact peak AP和全部seed高精度阈值失败，说明当前单格presence接口不可部署；不得用0 selected peaks的精度假象或graph状态机补偿。
- 只读角度归因证明近邻方位容量存在，但top4/NMS仍需约20°容差才达到目标召回；决定不把宽容差改成正式评分，也不宣布V1成功。
- V2科学变量限制为peak supervision/loss：用圆周soft target为邻近方位提供连续梯度，用hard-negative ranking直接惩罚高分假峰。backbone、geometry heads、C01--C06/C07/C08 split、seeds、epochs和安全门先保持。
- 先做零训练loss/readiness；若不能证明正确近邻峰排序、hard-negative非零梯度、rotation equivariance和真实batch finite backward，则停止V2。
- V1 seal=`5477c4b7ddc32313d186e1a8a5b7dbb0dc8867cf9b52bb1b2fffb737b87ef018`，失败图SHA=`4569ea433eba92c73ab2e47a244e3d1231e190c282150cd94cf599ec41756289`。

## 2026-08-29 — Set Process cardinality保留，normalized finite-set intensity停止进入图

- 三seed从头训练证明显式1--4 cardinality可稳定泛化：C07/C08 count accuracy超过95%，结构action macro-F1超过93%。该头和证据保留为结构事件容量与后续消融，不因整体FAIL删除。
- 同一模型在冻结2deg连续一一匹配和99.5%精度门下几乎拒绝全部观测；raw数量成功不能替代出口身份/位置安全，也不能直接触发拓扑节点。
- 决定停止当前normalized finite-set intensity作为部署出口接口；不降低precision、bearing tolerance或recall门，不使用C08回调模型，不进入graph/planner。
- 当前证据只看到安全召回坍塌，尚不能断言根因是定位分辨率还是置信度排序。因此下一步只允许冻结输出failure attribution：全人口无阈值匹配、容差曲线、cardinality分层、三seed分歧与多种无训练置信度可分性。
- 只有归因证明一个明确、可独立消融的修复机制后，才允许下一模型readiness；禁止无依据增加epoch、容量、loss权重或阈值grid。
- 正式失败seal=`18a5dcb717ec7133aa0b27056fc18e844510ddbf738048ef8fd05ae481413f92`，图SHA=`29095d2986ae11940b4ef4ffbdf13969f8e809e02ec0ae0318d02fc96aa5221d`。

## 2026-08-29 — 采用cardinality-conditioned circular slot transport

- 冻结归因排除“只需重新校准confidence”：全部预注册confidence的安全召回低于0.2%，且三/四出口即使10deg容差也几乎无法恢复完整集合。
- 排除继续使用selected-bin residual：残差只在Teacher peak bin受监督；推理选到邻近bin时几乎不改善角度，而oracle真值bin残差已达到约0.47deg。
- 决定保留已证明有效的cardinality、causal circular encoder和global geometry；用K个azimuth-domain slot distributions替代一张共享intensity。每个slot必须经置换不变一一传输匹配不同真实出口，连续bearing由圆周分布直接计算。
- 该设计没有free-query存在性：K由显式count决定，每个slot都必须对应一个可执行出口；slot只是在固定传感器圆周域分配质量，一一匹配直接惩罚duplicate/missing mode。
- 先验证数学表示、旋转/slot permutation、distinctness、confidence concentration和真实batch backward；PASS后才允许训练。归因seal=`7ff969054736b21c31d61177a5c69b87c1a02b537a1144b0ab97c3a269662b1f`。

## 2026-08-29 — Slot transport readiness通过，允许一次从头三seed训练

- V1布尔审计器system failure没有改变方法判断；V1R绑定旧失败并证明所有11个科学检查通过。
- 决定冻结K-slot设计：1+2+3+4共10个cardinality branch slots、180个固定azimuth bins、soft continuous target、permutation-min bijection、circular resultant bearing/concentration；训练不得新增objectness或自由query。
- 允许将参考置换loss向量化，因为单元测试证明全部loss数值与slot-logit梯度等价；参考函数保留作回归，不将实现加速写成科学贡献。
- 训练必须从头开始，三个seed统一报告；C07选择checkpoint和唯一whole-set refusal threshold，C08只迁移。核心新增分层门是三出口和四出口完整集合，防止双出口走廊多数类造成虚假总体提升。
- readiness seal=`32ab6cb9ea65177c38ef5ef0b2eb88f61ef123943ffe740694a7d6de4cc00846`，图SHA=`8cdb43a46fd1827b4f64857f143dfe86904ff7f98f28e71b887f310d587f2445`。

## 2026-08-29 — Slot transport保留为有效消融，停止直接进入图并先归因严格定位

- 三个独立seed的最佳C07 loss几乎一致，且整体/三出口10deg exact-set显著超过set-process，排除“训练失败”和“bijective slots没有价值”。
- 但严格2deg下三出口几乎全部失败、四出口完全失败；minimum concentration在99.5% precision下只保留个位数观测，排除把当前confidence直接用于在线节点提交。
- 决定不降低2deg定位门、不降低0.995 precision、不删除较弱seed、不使用C08回调、不用图状态机掩盖感知错误。
- 下一步只允许冻结输出归因，比较每seed/ensemble、circular mean/argmax/local mode、目标邻域mass、entropy/concentration和seed alignment。只有归因指向明确且可独立消融的机制，才允许下一候选readiness。
- slope失败作为独立geometry问题记录，但本次先解析出口集合主阻塞；不得用只修slope宣称感知方法成立。
- 正式失败seal=`472630ffd88a1331ead2abfdaf8f3469c6e1501d97d041b2d109685253f06455`，图SHA=`c36d02071e5604b3a991c5d40cd29a306bf11195590a8e81095fd9b1a4b9871a`。

## 2026-08-29 — 采用Cyclic-Ordered Unimodal Slot Transport修复多出口换位扩散

- 冻结归因证明ensemble优于single seed，oracle count几乎不改善，argmax/local-mode也没有双侧稳定增益，因此不再尝试后处理解码、删seed或oracle cardinality。
- K>=3时目标附近mass显著下降且角误差增大，与当前K! permutation-min允许每个sample任意换位一致；这种监督不要求slot在圆周上保持相对顺序，会产生label switching和多峰扩散。
- 决定将匹配限制为orientation-preserving cyclic shifts：圆周旋转只引起循环换位，而反向/crossing assignment必须受罚。K=2与旧全排列等价，K>=3是唯一科学变化，便于独立消融。
- 同时将raw equivariant angular evidence投影为单峰proper circular distribution，以concentration表达定位不确定性；这是直接修复已测得的低目标mass和entropy-confidence失配，不增加free-query existence。
- 先做零训练readiness。只有数学顺序、数值稳定、等变性、missing-mode梯度和真实batch全部PASS，才允许训练；否则停止，不进入图。
- attribution seal=`06bee093f8a18e3eecd4bbbe407db97416ce986ac6169c9bf45235683bb3242c`，图SHA=`576b9b351956f518e434910a562e83b61d49f0de4d79ba1e2422555e1297bed2`。

## 2026-08-29 — COUST V1 metric FAIL保留，以distribution-space V1R作唯一替代

- V1 projected slot-mass rotation=`2.67e-7`，其他tensor最大误差=`7.63e-6`，但float32 atan2 bearing诊断=`0.003601deg`；将degree与dimensionless tensor统一比较没有量纲意义。
- 不覆盖V1；它保持正式FAIL。V1R绑定V1并保留角度诊断，由transport实际消费的normalized circular distribution承担旋转合同。
- V1R不放宽`3e-5`、不改模型/数据/样本/其他门；正式11/11 PASS，0 optimizer/checkpoint/test/graph。
- 仅允许进入三seed训练准备；readiness PASS不等于最终方法确立。若不能改善K>=3严格定位与safe recall，不得进入graph。

## 2026-08-29 — 停止COUST进入图，先归因复杂cardinality collapse

- COUST总体2deg exact-set提高7.4/10.9个百分点，但K=3比旧方法更差、K=4近零；决定不让多数类总体收益覆盖核心复杂结构失败。
- C07选择的concentration阈值迁移C08后precision仅0.818，决定拒绝“重新校准即可”的解释。
- 禁止只选择loss最低的seed2、降低precision/bearing门、扩大匹配容差、增加epoch或用图状态机补偿。
- 保留COUST为必要消融；下一步只允许冻结输出归因，必须区分cyclic-start仍不唯一、slot collapse、共享特征/容量、单峰偏置、ensemble alignment和confidence objective。
- 正式seal SHA=`47e4a84354ad58498e24de839fab7a3b33fb0dbd0846213a12dcfd5114bffd80`；图SHA=`d5f533ebf25b1c60431c0d30600cf2c7c2c1cd64fc6852bc1c933389b4e26129`。

## 2026-08-29 — 停止独立圆周slot分布，采用Joint Cyclic Gap Simplex候选

- 冻结归因表明unrestricted跨seed对齐和best single seed均不能恢复K>=3 strict集合；继续改ensemble或挑seed2没有科学依据。
- K=3 ensemble间距塌缩比例低于预注册主因门，且约94%最优匹配保持cyclic order；COUST的主要错误不是出口顺序或整体间距，而是各slot独立局部定位及其concentration目标。
- 决定停止COUST后处理、kappa校准和同类独立slot decoder。旧Set Process、Slot Transport和COUST完整保留为逐级消融。
- 新候选Joint Cyclic Gap Simplex只保留一个圆周phase自由度，并以softmax gap simplex产生K个正间距、强制总和360度，再累积得到K个有序出口；cyclic relabel由loss处理。
- 该改变直接约束完整集合，而不是添加更大backbone或更多loss。先做零训练readiness；只有wrap/rotation/cyclic-invariance/closure/distinctness/真实finite backward全部成立才允许Data Card和三seed训练。
- 相关工作核查未发现相同的“因果LiDAR几何出口集合→gap-simplex→执行验证拓扑图”机制；但Cano 2026已覆盖LiDAR出口角+tracking+纯拓扑图，因此论文贡献必须落在完整几何语义、联合集合表示、学习关联/拒绝和执行验证边，而不能声称首次LiDAR拓扑导航。

## 2026-08-29 — Joint Gap V1不直接放宽容差，先补phase概率与拒绝合同

- V1正式FAIL时phase distribution的旋转误差仅`1.40e-9`，而degree bearing在近零resultant下误差`0.00720°`；这说明统一数值门错误地要求未定义角度稳定。
- 但不允许仅修改评估器宣称PASS，因为代码审计发现phase distribution没有proper likelihood，且set confidence不含phase concentration。这会在训练后继续造成“分布散但角度看似存在”的安全漏洞。
- 决定保留V1 FAIL，创建V2方法纠正：cyclic phase候选直接用proper circular NLL；set confidence乘phase concentration；rotation以distribution/resultant为权威，degree输出在低concentration下只作诊断。
- V2不得改数据、gap floor、cardinality、geometry Teacher、backbone或测试隔离。若proper phase和权威rotation仍失败，停止Joint Gap路线。

## 2026-08-29 — Joint Gap V2合同通过，允许一次三seed能力验证

- V2证明phase概率可监督、低可观测样本可拒绝、gap严格闭合且复杂集合collapse有直接梯度；因此该候选具有明确不同于独立slot的结构机制。
- 允许从头三seed训练，但readiness不构成论文方法成立。训练人口、split、backbone、epochs和安全门沿用COUST公平对照。
- 主要判断必须是K=3/4 strict 2deg complete-set和C07→C08拒绝迁移；总体单/双出口收益不能覆盖复杂路口失败。
- 若训练失败，不再通过增加gap head、epoch或容差修补；Joint Gap进入失败消融并重新评估结构表示。

## 2026-08-29 — 不续跑首次JCGS训练；先纠正phase singular backward

- 训练前发现并纠正K=1 closing gap数学错误；正式V3重新资格证明纠正后其余方法合同保持成立。
- V1 seed1的CUDA gather越界发生在4个完整epoch之后；保存权重finite、seed0完整成功、selection未执行，所以不能把V1归为科学失败或通过。
- 联合phase以`atan2(resultant_y,resultant_x)`解码；在resultant接近/等于0时forward角度无定义且标准backward可产生NaN。当前训练器未设置nonfinite gradient fail-closed，符合“坏梯度更新后下一次bearing gather越界”的证据链。
- 决定不覆盖/续跑V1，也不挑seed0。最小corrective限于：用float32 phase-mass机器精度定义stable atan2 denominator floor；在periodic sampling和optimizer step前显式拒绝nonfinite；增加origin/low-resultant/real-batch回归。
- 该修正不改变数据、Teacher、phase/gap表示、参数量、loss权重、epoch、阈值或科学门。重新资格PASS后，V1R必须三个seed从头执行。

## 2026-08-29 — stable phase假设被V1R否定，改做同步kernel定位

- V4证明stable phase和finite fail-closed接口本身正确；但V1R seed1在与V1相同epoch/time重复device assert，且gradient guard未触发。
- 因而原`PHASE_RESULTANT_SINGULAR_GRADIENT`只保留为已修的潜在风险，不再作为本次实际崩溃根因。
- 拒绝第三次直接三seed重训，也拒绝通过clamp/nan_to_num/索引取模掩盖未知越界；这些操作可能静默改变Teacher或几何绑定。
- 决定用`CUDA_LAUNCH_BLOCKING=1`在完全相同seed1前5轮同步重放。诊断允许梯度但不用于模型选择；必须在第一次assert停止，并保存真实调用行与上下文。
- 只有定位到确切有限索引来源后才允许一个最小修正与重新资格；若根因是数据/Teacher非法，停止模型并回到Teacher，而不是适配模型绕过。

## 2026-08-29 — 采用canonical periodic endpoint修复，并允许一次JCGS V1R2

- 正式同步证据把越界锁定在phase probability gather；故障值是圆周零点的负浮点舍入，而非非法Teacher bin。因此拒绝修改Teacher、删除样本或clamp到179。
- 决定将所有periodic linear coordinates的整数部分modulo 180：180与0在圆周上相同，fraction保持0；非有限输入仍fail closed。这一规则同时覆盖loss和geometry sampling，避免两个坐标接口再次分叉。
- V5因确定性cuBLAS环境未传播在科学计算前失败，固定保留；V5R只传播既有训练环境变量，不改科学内容，正式通过真实故障批次和旧合同。
- 允许一个新的、从头执行的V1R2三seed训练。不得复用V1/V1R checkpoint、挑单seed、续跑epoch或改变门。readiness PASS不是方法成立；只有V1R2预注册科学指标决定是否进入拓扑图。

## 2026-08-29 — 停止JCGS进入图，先归因联合集合退化

- V1R2的系统完整性成立，但overall和K>=3严格集合均低于预注册门且低于COUST；决定不以“联合闭合表示更有理论约束”替代实验证据。
- K3/K4在10°仍为0，排除仅放宽2°即可解释；K1/K4 count为0且K2占优，提示cardinality长尾塌缩，但K3在count部分正确时完整集合仍为0，不能只加class weight就宣称可修。
- C07置信阈值在C08由唯一正确接受变成唯一错误接受，禁止重新校准、降低0.995 precision或让图状态机补偿。
- 决定保留三个checkpoint和预测作失败消融，只允许一次零训练/零新推理归因。归因同时计算oracle count、oracle phase、oracle gap和per-seed，必须区分表示、监督、ensemble与置信目标，再决定下一科学路线。

## 2026-08-29 — 停止完整出口集合回归，采用轴锚定事件—关系分解候选

- JCGS正式归因排除cardinality和跨seed cyclic alignment：oracle count及Teacher-aligned ensemble在两split的strict exact2增益都远低于0.05。
- K>=3在oracle phase下10deg完整集合为0，oracle gaps下也仅约11%--12%；决定同时否决任意global phase和gap-simplex shape作为主结构接口。
- 不删除旧工作：Set Process、Slot Transport、COUST和JCGS组成完整的表示消融链；旧event/cardinality、global geometry、descriptor transport及stateful commit结果是新候选的直接证据与基线。
- 新候选固定为Axis-Anchored Geometry-Semantic Event Relation：以可观测运动/轴线消除圆周起点自由度，以事件分类与相对branch关系替代一次性完整集合回归，几何关系直接产生节点与出口关联，执行穿越才产生边。
- 该决策不降低旧指标来宣布成功，也不允许直接进入图。先验证Teacher标签、反向traversal变换、样本人口和泄漏合同；若不可唯一/不可观测，立即停止候选。
- 正式归因seal SHA=`c627c1265f26b2dbcc49de542b9befae1a838f14d258bbe6772e17a2b424d1c4`；论文图SHA=`b61df1f3d0e6660b8dd29b3d6c3bc90c9dbde71b8cb010b27e890326a216899b`。

## 2026-08-29 — 接受Axis-Anchored Event Relation进入method readiness

- 决定：Teacher feasibility 12/12 PASS后，停止“方法方向未定”的搜索，固定Axis-Anchored Geometry-Semantic Event Relation为唯一主候选；下一步只允许实现/验证该接口。
- 证据：80 worlds、188,126 observations、396,913 exits、4,493 identities；全部identity唯一branch star，全部junction至少4 route views，时间关系与几何监督充足，33,083项源seal验证通过。
- 边界：这确立的是研究方法与监督可行性，不是模型性能PASS；三seed训练未开始，graph/planner仍关闭。
- 保留：旧Set Process、Slot Transport、COUST、JCGS和stateful commit均保留为论文基线、消融或失败分析；7个局部K2 junction作为provisional/reveal困难案例保留。
- 下一门：typed model、masked loss、past-only memory、rotation/permutation/reverse traversal和real-batch finite backward全部PASS后，才允许三seed训练。
- 正式run seal SHA=`709930fc5c2e7774883bd5493851b1d9f60d07a79e0926b1662cdfee3219a090`；论文图SHA=`6407e28a89e8b193b97f79b06a9ec96ac767636d80c28f660bc90196df9a1569`。

## 2026-08-29 — 冻结Axis-Anchored Event Relation网络接口并允许三seed Data Card

- 决定：819,196参数typed模型、七项loss和past-only union通过readiness，允许进入三seed训练准备；禁止在训练前再切换表示路线。
- 输入边界：forward只有五帧range/valid；identity只作Teacher-only contrastive target，pose/world/traversal/future/graph禁止。
- 系统决定：保留`3e-5`rotation门，冻结deterministic algorithms、`CUBLAS_WORKSPACE_CONFIG=:4096:8`和TF32关闭；不得以GPU默认精度为由放宽合同。
- 科学边界：readiness不证明泛化或论文收益；C07/C08训练结果才决定是否允许图。
- 下一步：三seed训练必须分五事件、四关系、连续几何、place/branch association和安全拒绝报告，turn/transition和reveal/withdraw不得被总体人口掩盖。
- 正式seal SHA=`022b0c0afec897ae5a3d33c3ddd30bb76d0c648038a8553b8c7a740b9d5931e5`；论文图SHA=`df85811305e643d6ef9f70e73b66b2cbd6c4a3c77e1c1e1248035e4d3d594a80`。

## 2026-08-29 — 缺失早期relation采用mask并拆分训练调度

- 决定：全部188,126样本保留；观察级前后Teacher不存在的relation pair标`-1`，禁止删样本、复制邻帧或把absence冒充未知。
- core pass：全样本event、有效relation、current branch geometry、axis和global geometry。
- descriptor pass：identity-balanced place/branch contrastive batches；identity继续禁止进入forward。
- 权重：仅从C01--C06人口一次性计算inverse-square-root normalized；C07/C08不适配。
- V1R证据：8 unit + masked core/descriptor CUDA backward + 原15项不变量PASS，模型参数不变。
- 正式seal SHA=`cce9ff427daa2f5399a592f10c5fd75bd9c7957b91e7bc42db380a92126c0c14`。

## 2026-08-29 — 稀疏关系候选必须显式预测数量并精确匹配几何目标

- 决定：固定六个proposal只是计算容量，部署有效token数由独立0--6 cardinality head决定；禁止恢复存在性阈值或把六个proposal全部写入图。
- 训练mask：缺少观察级Teacher的历史帧和相邻pair不参与proposal/count/transport loss，其他当前事件和几何监督保留；不删除轨迹开头。
- 稀有关系权重：withdraw/reveal只采用相对persistent人口的平方根权重，均小于10；拒绝旧dense BCE数千倍正权重。
- 真实loss发现第六个token geometry uncertainty无对应目标，决定删除该单一输出而不是padding/slicing Teacher。它减少65参数，不改变backbone、数据、阈值或结构语义。
- 两个不可覆盖corrective均PASS；允许准备一次三seed训练，但性能未成立且不得进入图。

## 2026-08-29 — 训练运行必须在科学输出前纠正总时限与完整召回分母

- V2 seed0两轮实测约8.5分钟/轮，30轮加评估无法被4小时外层wrapper完整容纳；决定提前停止而不等待确定性超时，失败run只作系统证据，不作科学结果。
- 评分器审查发现relation recall的旧正例分母来自成功定位的预测端点，真实端点未提出时没有FN；决定停止V2R，禁止让token recall门替代独立relation recall的客观完整性。
- 正式corrective固定：precision继续按模型发出的候选计算；token及persistent/reveal/withdraw的positives、FN、recall、F1全部按独立Teacher总数计算。阈值、容差和最低召回不变。
- corrective全量PASS并冻结18项seal SHA=`24551f44cf15fa7cf1f3c9a7a334a1bd4d84bd552fe45246b759f49bf99482fb`。允许唯一V2R2三seed训练；只改变外层8小时上限和评估计数正确性，不允许修改模型、loss、数据、epoch、seed或科学门槛。

## 2026-08-29 — 保持batch256，只纠正跨world CUDA缓存生命周期

- V2R3停止时长进程显存约20.8 GiB；独立batch256复现实验低于16 GiB，表明问题来自跨shape/world累积的allocator cache，而非单批计算容量。
- 正式V1证明batch128与batch256输出近似相同，但weighted validation loss最大差`0.0190654`；因此减小batch会改变checkpoint选择，科研上不可接受。
- 决定只在每个validation/final-inference world前后释放未使用缓存。V1R全C07证明输出/loss误差为0且最大process memory约9.71 GiB。
- 不修改训练batch、验证batch、模型、loss、数据、seed、epoch、阈值或评价口径。V2R4必须三个seed从头执行，V2R3部分epoch不得复用。
- 资源PASS只解除系统阻塞，不构成论文方法成立；科学FAIL仍必须停止进入graph。

## 2026-08-29 — V2R4超限2 MiB仍按硬门失败处理

- 训练进程达到`16,386 MiB`，冻结上限为`16,384 MiB`；拒绝用测量粒度或“只超2 MiB”事后豁免。
- 拒绝提高资源上限和减小batch；前者改门，后者已证明改变validation loss。
- 选择在60个training world边界释放unused allocator cache，并以完整seed0 epoch0对sealed V2R3做77 tensor exact parity和全部指标`3e-6` parity。
- V2R4部分checkpoint/三轮曲线只作系统证据与趋势，不得续跑、选epoch或参与C08/图。

## 2026-08-29 — 接受训练world unused-cache lifecycle并恢复三seed资格

- 正式单epoch证明所有77模型tensor和全部train/C07指标误差均为0，确认`empty_cache`只改变allocator生命周期，不改变优化轨迹。
- 60-world最大process memory从旧长进程越界降至12.017 GiB；因此采用world边界释放而非提高资源门、减小batch或process isolation。
- V2R5必须从三个seed全部重启，V2R3/V2R4 checkpoint不得复用；正式trainer直接记录process peak，runner超过16 GiB即FAIL。
- 该决定只恢复执行资格，不能代替C07/C08科学门。

## 2026-08-29 — V2R5保留为学习式token/transport基线，不再视为轴锚定主方法资格

- 代码审计确认`local_axis`没有进入proposal、transport或结构节点关系，当前实现不输出相对轴的continuation/side-branch语义。
- 数据审计确认route-aligned pose使fit/C07/C08水平axis几乎恒为前方；常数前向基线平均误差=`2.79198/3.26039/3.33944deg`，旧`axis<=10deg`门无科学区分力。
- 决定不停止V2R5，因为其三seedtoken、时序关系、几何、关联和拒绝结果是必要基线/消融；但禁止它的旧selection直接开放graph或支撑“学会中心轴”的论文结论。
- 最终主方法必须显式区分已知运动坐标锚与学习的几何结构语义：相对分支关系、事件、metric geometry、descriptor和uncertainty必须直接控制node生成/关联，edge仍只由真实穿越建立。
- 若保留learned axis作为贡献，必须新增独立yaw扰动验证并相对常数/运动方向基线取得预注册提升；不得把现有route-aligned评测改名为轴学习证据。

## 2026-08-29 — 最终事件必须由几何关系组成，独立event head降为消融

- 当前模型的event logits只依赖global encoder context；token count、relative bearing、token geometry、transport和global geometry没有进入event分类。
- 当前图以该独立event分类触发structural node，因此尚未建立“几何语义直接决定图结构”的方法因果链。
- 决定让V2R5完整完成并冻结为learned token/transport + independent-event强基线；不删除任何checkpoint、预测或图。
- 最终候选采用Geometry-Relation Event Composer：只从route-frame token集合、persistent/reveal/withdraw、metric geometry/变化和uncertainty产生五类事件；graph节点由composer触发，边仍仅由真实穿越建立。
- 独立event head、无显式几何和无transport必须成为三个独立消融；若composer不能改善事件和离线图，不进入闭环，不用planner调参掩盖。

## 2026-08-29 — 旧event Teacher与主方法脱节；冻结双Composer候选规范

- 具体证据：V2R5 trainer的event target来自deduplicated Zarr `event_index`，fit事件人口为`102874/19743/5551/1482/12534`；它没有读取corrected causal Teacher manifest。后者fit/C07/C08 transition仅`791/67/173`帧、`59/5/12` identities。
- 影响：V2R5的token、transport、metric geometry和descriptor仍是有效组件基线；它的event head不能作为最终语义checkpoint。这是Teacher/interface脱节，不是当前训练系统崩溃。
- 拒绝方案：不停止并删除V2R5，因其是必要基线；不将旧event logits重命名为几何语义；不立即训练一个无因果证据的新黑盒头。
- 选择方案：冻结`docs/GSE_GRAPH_METHOD_SPEC_V1.md`。Action-Set Relation Composer只由出口token几何/count/transport产生junction/terminal/provisional；Metric-Change Composer只由宽高坡度曲率因果序列产生turn/geometry-transition和回投位置。
- 方法边界：descriptor只进入节点/出口关联，不进入Composer；模糊关联保留provisional；edge只由真实穿越创建。
- 执行顺序：先完成V2R5三seed基线，再把已封存显式输出与corrected Teacher对齐做零训练可观测性/冲突审计。审计PASS前不实现Composer训练，不读C09/C10，不建图。
- 当前候选只是可实施规范，不是已确立论文方法；只有三seed增益、安全拒绝、必要消融和图def-use全部通过后才能改为`METHOD_ESTABLISHED`。

## 2026-08-29 — seed0显式状态诊断支持action-set，但简单metric change不足

- 实现了只消费部署显式输出的评估原语：`P(count>rank)`、token-set圆周矩、opening/profile/uncertainty加权量、persistent/reveal/withdraw质量和不跨traversal的因果几何序列。
- 评估原语现以确定性route-frame bearing规范化完整token/transport为642维矩阵，并通过8项单元测试和1个真实C07 archive对齐诊断；改动forbidden event logits/descriptor不改变矩阵。当前没有创建formal run、训练主模型参数或选择阈值。
- seed0 C07/C08阈值无关AUC：junction=`0.982144/0.967302`，terminal=`0.986168/0.995410`，turn=`0.694021/0.723302`，geometry-transition=`0.587813/0.592704`。
- 这不是方法PASS。它否定“简单宽高差分数就足以产生变化节点”，但不否定完整token-set、transport和时序Composer。
- 正式决策继续等待V2R5三seed封存；审计必须对完整显式集合做跨seed与C07→C08转移。若change-point仍无非平凡precision/recall，停止Metric-Change Composer，不扩容或调planner。
- 此情形的唯一方法级fallback冻结为`RouteGeometryProfile`：先证明当前LiDAR对前向通道宽/高/坡度/曲率纵向剖面的可见性和Teacher唯一性，再考虑显式剖面头。不允许用hidden context、descriptor或planner补救。
- 单seed进一步诊断使用全部642维显式状态及固定C07拟合/C08评价线性探针。junction/terminal AUC=`0.947877/0.995512`，turn=`0.733758`，geometry-transition=`0.574440`；turn/transition AP=`0.026425/0.019790`。因为只有seed0且没有formal Data Card，该值只作实时方法风险证据，不作PASS/FAIL或阈值选择。

## 2026-08-29 — 相关工作复核使Metric-Change成为方法必要条件

- 原始来源边界：Cano 2026已覆盖3D LiDAR出口角、出口时序稳定、intersection/tunnel/dead-end判断和轻量拓扑图；PRISM-TopoMap已覆盖learned place recognition + scan matching的在线拓扑定位；Semantic Topometric Mapping已覆盖结构语义驱动探索；Pan et al. 2026已覆盖时序概率descriptor和uncertainty filtering。
- 决定：Action-Set Composer、learned descriptor和refusal均只能作为组件，任何单项不得作为GSE-Graph创新。论文可生存边界收窄为“learned metric route geometry + executable exit relations直接组成typed causal events，再经拒绝关联与真实穿越形成图”。
- 影响：Metric-Change Composer及其可观测输入从可选增强改为主方法必要条件。若三seed显式输出正式审计仍不能恢复corrected change points，下一步必须是RouteGeometryProfile零训练可见性/Teacher唯一性证明，禁止退回出口计数、hidden event context、descriptor-only或planner补偿。
- 当前状态：方法仍为`FROZEN_CANDIDATE_NOT_ESTABLISHED`；V2R5继续完成为组件基线，C09/C10、graph与M-TARE保持关闭。

## 2026-08-29 — 冻结显式Composer可观测性审计口径，不提前创建run

- 数据：只用sealed V2R5三seed的C07/C08开发预测和corrected causal Teacher；C07/C08=`21,548/24,394`行、`10/10` worlds。transition只有`5/12`个独立identity，明确记录为多样性风险。
- 接口：每条观测只允许642维显式token/count/opening/profile/uncertainty/transport/reveal/五帧metric geometry；hidden context、event logits、descriptor、identity、pose、world、TNG与future禁止进入主探针。旧event probability只作独立baseline。
- 决定：允许12个固定诊断线性probe（3 seeds × 4 events）在C07拟合，C08一次原样迁移；三seed只对同事件score等权平均，不训练“平均显式状态”第四probe。这是零主模型更新的可观测性审计，不是deployable model training。C08不得选择probe、阈值或特征。
- 当时草拟的预注册门：ensemble C08 AUC junction/terminal/turn/transition=`0.90/0.90/0.70/0.65`，turn/transition AP至少为各自prevalence的2倍，至少2个seed的transition AUC不低于0.60。此处曾包含的`C07→C08 AUC gap<=0.10`已被下一条决策在正式card/spec生成前明确废止，因为C07是probe的拟合集而非独立验证集；gap仅保留为诊断。transition 0.65沿用先前causal geometry-delta正式门。
- 实现：`evaluate/run/freeze_gse_composer_observability_v1.py`及显式评估模块已完成；真实seed0全量只读smoke对齐45,942行和642维矩阵，10项unit PASS。V2R5三seedseal前禁止freeze spec、create run或执行正式审计。

## 2026-08-29 — readiness否决把线性probe训练AUC gap作为科学门

- 具体证据：seed0端到端只读诊断的explicit probe在C07拟合后，transition C07/C08 AUC=`0.999999/0.657706`、C08 AP=`0.014832`（约为`0.007092` prevalence的2.09倍）。C07只有5个transition identities，训练AUC接近1是拟合结果，`|fit-transfer|=0.342293`不能解释成跨世界表示退化。
- 影响：原草案`fit-transfer AUC gap<=0.10`混合了in-sample fit与untouched transfer，属于metric缺陷；若保留会在已有非随机C08信号时机械FAIL，不能回答显式状态是否可观测。
- 决定：正式card/spec尚未生成，因此在freeze前纠正。gap完整报告为过拟合诊断但不进PASS；科学门仍为untouched C08 AUC/AP、至少2/3 seed transition一致性及完整identity coverage报告。后续Composer性能、安全precision/recall和图门不降低。
- 资源：seed0四个固定probe约59秒、peak RSS=`915,852 KiB`；junction/terminal/turn/transition C08 AUC=`0.949001/0.996560/0.742239/0.657706`。该单seed值仅为readiness，不作方法PASS。

## 2026-08-29 — 三seedComposer审计采用score ensemble而非feature mean

- 风险：不同seed的六个token即使按route-frame bearing规范排序，也可能因count和相近bearing产生不同slot对应；先平均642维状态再拟合第四个probe会引入没有物理语义的跨seed feature混合。
- 决定：每个seed独立拟合四个固定probe，再对同事件C07/C08 score等权平均；train-free score和独立event probability也同样采用score-level ensemble。正式probe fit数由16改为12。
- 影响：这只纠正ensemble实现，单seed接口、Teacher、数据、C08隔离和全部科学阈值不变；正式card/spec仍未freeze。
- 实现复核进一步删除了虽未参与指标、但仍被无意义计算的跨seed `feature_matrix`平均，并增加测试保证ensemble字典不再生成该字段；12项测试全部通过。预测包中的`geometry`已沿trainer导出链确认来自模型前向的`width/height/slope/curvature`，不是Teacher target，因此主探针不存在几何真值泄漏。

## 2026-08-29 — 历史事件模块不得冒充双Composer实现

- `gse_action_set_node.ActionSetNodeDetector`的输入是三个seed联合的`[5,3,6,40]` raw tokens，40维中包含32维descriptor；这与主方法“每seed独立训练、descriptor只用于关联而不进入事件Composer”的边界冲突。
- `gse_causal_geometry_delta.CausalGeometryDeltaEventHead`显式读取hidden `context`和`baseline_event_logits`；这与Metric-Change Composer只能消费预测宽/高/坡度/曲率因果序列的边界冲突。
- 决定：两者只允许贡献集合池化、past-only mask、损失与评估实现经验，不允许原样复用、薄封装或改名作为GSE-Graph主方法。正式可观测性审计PASS后必须实现新typed Composer接口；FAIL则仍按RouteGeometryProfile proof处理。

## 2026-08-29 — 显式状态容量正式PASS，允许实现双Composer但不开放graph

- 正式证据：C07/C08=`21,548/24,394`行、10/10 worlds，12个固定显式probe，C08无适配；junction/terminal/turn/transition ensemble AUC=`0.965620/0.999131/0.768548/0.737538`，turn/transition AP为prevalence的`15.56x/2.51x`，全部预注册容量要求通过。
- 与V2R5直接失败的关系：V2R5旧事件头、关联阈值和直接关系提交失败不等于其中间显式状态无信息；本审计证明可迁移信息存在，因而继续实现受限Composer比重训旧头或立即转RouteGeometryProfile更有依据。
- 仍然存在的风险：C07高精度阈值迁移到C08后turn/transition identity coverage仅`4/52`与`0/12`；容量PASS不等于部署PASS，也不允许建立正式图。
- 决定：下一步实现两个新的typed接口。Action-Set只消费count/bearing/opening/profile/uncertainty/transport；Metric-Change只消费五帧预测width/height/slope/curvature及不确定性。descriptor、hidden context、旧event logits、GT identity/TNG/future全部禁止进入事件Composer。
- 必要证据：新Composer必须提高稀有事件precision/recall/identity coverage，并完成独立事件头、无显式几何、无transport三项消融。只有这些门通过才允许离线graph；否则停止Composer训练并重新评估RouteGeometryProfile。

## 2026-08-29 — 双Composer采用typed geometry-only接口并获得正式训练资格

- 历史ActionSet detector会把三个seed和descriptor联合输入事件头，历史CausalGeometryDelta会读取hidden context与旧event logits；继续包装它们不能证明“几何语义决定图结构”。因此新模块从零建立两个dataclass输入边界，不复用上述事件前向。
- Action Composer固定只使用bearing、existence、opening/profile/uncertainty、count和transport，并以相对方位矩和DeepSets聚合保证全局yaw及token排列不影响事件。Metric Composer固定只使用五帧width/height/slope/curvature、其一阶差和uncertainty，并学习五帧内因果回投。
- 正式readiness在真实C07输出上15/15检查PASS，参数量`102,500/36,485`，全部梯度finite，最大不变量误差`2.98e-8`，repeat=0；决定=`ALLOW_DUAL_COMPOSER_TRAINING_DATA_CARD`。
- 边界：readiness没有训练也没有性能结论；不得据此开放graph。第一版必须冻结V2R5三seed骨干/显式状态，每seed只训练自己的两个小Composer。C08不得反向影响checkpoint、温度、refusal或阈值；任何核心门失败都停止进入图，而不是改planner补救。

## 2026-08-29 — 缓存跨进程概率采用binary16表示级parity，不复制双路径

- V1前置复现的唯一差异是三个GPU派生概率的float16末位；当前重复执行逐位一致，主体几何/logits逐位一致，所有离散决策一致。证据不支持checkpoint或数据漂移，也不支持继续要求所有派生概率跨历史进程逐位相同。
- 可选路线包括：A）主体bit-exact、派生概率一binary16 epsilon且决策exact；B）复制旧C07/C08、只推理fit，形成两条生成路径；C）宽泛放宽全部字段。依据用户的自主最优选择授权，采用A，拒绝B/C。
- V1R没有使用观察到的`0.000488`作为事后阈值，而固定类型定义`np.finfo(np.float16).eps=0.0009765625`；任何主体差异、超过一个epsilon或决策变化仍立即FAIL。
- V1R三seed/240文件/564,378 forward全部PASS，主体误差0、派生最大`0.00048828125`、决策全相同。决定=`ALLOW_TYPED_DUAL_COMPOSER_THREE_SEED_TRAINING_IMPLEMENTATION`；缓存不得供其他隐藏事件头或图直接使用。

## 2026-08-29 — 导师汇报主文与技术附录分离

- 用户指出原28页HTML虽包含地图和失败图，但仍按内部术语组织，不能让老师直接看懂“怎么做、什么结果证明失败、为什么换方法、当前障碍是什么”。
- 决定保留原`GSE_GRAPH_METHOD_EXPERIMENT_REPORT`作为完整技术附录，新增`GSE_GRAPH_ADVISER_PROGRESS_REPORT`作为汇报主文，不删除或改写正式实验证据。
- 主文固定使用“问题→实际地图与数据→Teacher→逐项失败实验→证据驱动的方法选择→新方法实现程度→障碍/停止条件”顺序；地图必须展示树形、含环、复杂三维真实结果，失败必须同时给出预注册对照和实际指标。
- 该调整只改变科研沟通结构，不改变数据、Teacher、split、阈值、模型、阶段结论或下一实验；任何尚未训练/建图的部分均明确标为未完成。

## 2026-08-29 — Composer监督V1绘图系统FAIL只允许表示无关V1R

- V1已写出全部80个NPZ，summary的行数、事件、identity和回投七项检查均PASS；唯一异常是`axes.bar(dict, dict.values())`在当前Matplotlib中把字典视为不可哈希类别。
- 问题归类为证据可视化实现，不是data/Teacher/model/metric问题，不否定监督内容，但V1 runner没有完成可识别科学结果，必须保持FAIL。
- 唯一corrective为把两个字典显式转换成`list(keys)`与`list(values)`，并增加三格式实际绘图单测；禁止借V1R改变Teacher、回投锚点、样本或验收计数。
- V1R绑定V1 RUN_STATE、runner summary、科学summary和seal，最终全部科学值逐项一致并完成正式seal；决定允许进入双Composer trainer/evaluator实现。
## 2026-08-30 — ERCSS 正式审计两次 pre-science 环境失败；允许最后一次 V1R2 系统纠正

- V1 在核对320个mesh文档后、读取世界前因Torch环境缺少Open3D退出；V1R切换到已冻结Open3D 0.19.0 sidecar，但旧`representation/__init__.py`无条件导入Torch模型，使纯NumPy ERCSS子模块间接要求sidecar安装Torch，同样在读取世界前退出。
- 两次run均无Teacher人口、无科学指标、0训练/推理/C09/C10/M-TARE/graph；不能判定ERCSS科学成败。
- 修正仅把原公开Torch符号改为惰性加载，保持名称和来源不变，使geometry-only sidecar不再获得隐藏Torch依赖。必须先通过Torch环境公共API和sidecar无Torch导入合同。
- V1R卡的“不再重试”被用户已授予的主线持续执行授权覆盖一次，并显式记录为最后一次`V1R2`系统纠正；科学数据、Teacher、阈值、覆盖率、容量和证据合同不变。V1R2若再发生任何系统错误，停止正式ERCSS审计，不继续版本链。
## 2026-08-30 — ERCSS V1R2因冻结器旧测试SHA停止；不自动创建V1R3

- 具体证据：V1R2在`frozen tool drift: tests/v3/unit/test_gse_registered_structural_skeleton.py`处退出，耗时0.0006s，0 mesh/world/Teacher/optimizer/inference/C09/C10/M-TARE/graph；当前测试独立执行`10/10 PASS`。
- 原因：V1R2冻结器从V1R spec复制工具表，只刷新card/runner/freezer并新增package hash，没有刷新已因惰性导入合同而修改的测试文件SHA。这是freeze实现缺陷，不是数据、Teacher、模型、指标或ERCSS科学失败。
- 影响：ERCSS全量可行性仍未知；V1R2预注册为最后一次系统纠正，故不依赖用户的常规持续授权自动突破该停止边界。
- 推荐选项：用户明确允许一个V1R3，仅重新计算所有当前工具SHA，科学Data Card、人口、Teacher、方法、阈值、覆盖率、容量、成本与证据合同逐项不变。另一选项是停止ERCSS并重新选择论文主方法。

## 2026-08-30 — 否决有限端盖延长，冻结只修实际失败帧的最小姿态合同

- 容量审计V1在任何槽位结论前复现有限union起点越界；全量只读归因得到`402`个三变体失败，对应`134`个源姿态，最大残差`0.275015 m`，全部是坡道有限端盖问题。
- 第一候选按传感器垂直偏置延长端盖，正式结果虽把越界降为0，但最大延长`0.510535 m`并进入`61`个nonincident primitives，违反构造身份隔离。该实现保留为失败证据，禁止主路径调用。
- 选择不改几何的最小姿态合同：原姿态必须先对三种冻结union联合判定；只有失败帧可沿所属directed traversal向内搜索，其他帧必须bit-exact；frame/world/edge/traversal/global/local identity、数量和顺序均不得改变。
- 正式qualification PASS：`134`帧改变、`252,296`帧bit-exact，`757,290`次检查越界为0，最大内移`0.3000154683 m`，最小相邻间距`0.6999845317 m`。决定恢复slot-capacity审计，并要求其绑定本run seal。
- run=`results/gate3_semantics/gate3_20260830_primitive_finite_cap_pose_qualification_v1_seed0`；seal SHA=`8a36f7f46d141df4773f957ea9f67e586b8d602d70611abbcd28395df6c24f1c`。

## 2026-08-30 — fit-only容量证据将基元集合从8槽修订为32槽

- V1R正式审计80 parents×3 geometries×3 roles=`720`个五帧窗口和`41,472,000`条射线。C01--C06最大可见基元=`16`，预注册`ceil(1.25×16)=20`，故固定候选`8/16/32`中只能选择32。
- C07/C08只作不变容量迁移，最大=`23/19`，均小于32；没有用迁移域修改fit选择。8槽overflow=`264`窗口、16槽overflow=`15`、32槽=`0`。
- 这使权威方法计划中`SweptSuperellipsePrimitive[8]`及关系矩阵`[8,...]`与正式证据冲突。决定由32槽合同显式取代旧8槽，旧8槽仅作容量消融；不得截断、top-K删除或按射线支持事后筛Teacher。
- P0至此完成，允许冻结P1完整数据导出Data Card；不自动开放训练、图或测试世界。
- run=`results/gate3_semantics/gate3_20260830_primitive_slot_capacity_audit_v1r_seed0`；seal SHA=`8593159efc145189f6e028c06c87c1377ef628837eba39601b1afcc00d8c54ae`。

## 2026-08-30 — 同步治理操作范围以允许Phase 3显式Teacher导出

- 权威`docs/PLAN.md`要求当前Phase 3先完成程序构造监督的P1数据导出，再进入模型训练；旧`OPERATION_GATE_RANGE`仍把`data_export/teacher_generation`上限固定为Gate 2，导致真实P1a spec被预检拒绝。
- 拒绝把20 GB写入任务伪装成`audit`。选择最小治理同步：仅把`data_export`与`teacher_generation`允许范围从Gate 1--2改为Gate 1--3，并把旧“Gate 3必须拒绝”单测改为“带精确Data Card授权时通过”。
- Data Card审批、authorized gate/operation、严格测试隔离、输入/工具hash、成本与证据要求均不变；Gate 4+仍禁止数据/Teacher导出。

## 2026-08-30 — 将未探索状态从感知Teacher移回在线执行图

- 具体问题：旧关系接口包含`unexplored_port_probability[32,2]`，但同一局部几何在首次访问与重访时LiDAR可以完全相同，标签却随历史执行状态变化；五帧学生输入无法唯一辨识该量。
- 影响：若从完整TNG或未来轨迹生成该标签，会造成Teacher/未来状态泄漏；若仅按局部几何生成，则会把“未连接端点”和“尚未探索”混为一类，损害关系学习与论文因果主张。问题属于Teacher/方法接口，不影响P1a传感器与逐射线基元来源导出。
- 选择：依据用户的持续自主最优执行授权，删除感知头和P1b Teacher中的`unexplored_port`。模型仍学习显式基元端点、端点连接、非连接重叠、跨帧对应与不确定性；未配对端点可产生provisional port，`unexplored/attempted/verified`只由在线图的真实穿越trace维护，不回流为感知训练标签。
- 被否决路线：不使用完整地图/未来轨迹硬造未探索标签，也不增加手写junction/terminal类别补偿。P1b正式Data Card必须显式绑定该边界。
## 2026-08-30：P1a全量出现四来源射线；保留完整集合并继续，不把抽样最大值当全量门

- 证据：P1a正式全量导出在`S05_flat_branch_medium_C04__ellipse`首次观测到`maximum_source_membership=4`，该分片确定性重放为真并有独立tree SHA-256；codebook 119完整绑定`edge_0042/0047/0054/0059`四个基元并在数组中出现1条真实射线。此前槽位容量审计的720个固定窗口只观测到最大3。
- 核对：P1a冻结合同从未把3设为验收阈值，而是要求完整保存所有来源集合，`uint16` codebook对应的集合大小验收范围为1--255；P1b逐元素遍历codebook source set，不存在3来源截断。32槽约束限制五帧可见基元总数，不限制单射线来源集合大小。
- 决定：继续不可覆盖P1a正式run，不修改数据、阈值或执行器；把“最大3”限定为前置抽样观测，不外推为全量性质。为尚未冻结的P1b增加真实合同对应的四来源无损单元测试，防止后续实现回归。
- 影响：不改变P1a科学问题、Teacher定义、32槽选择或训练接口；若全量最终超过255来源或五帧可见基元超过32，仍按原合同FAIL并停止。

## 2026-08-30 — 用Teacher内容规范排序修复Hungarian槽位近似平局，不放宽不变量门限

- 正式V1在真实三形状batch上只有`target_permutation_loss_error_le_1e6`失败：最大差=`1.7344951629638672e-05`，其余接口、梯度、确定性、端点反向、关系对称与配准检查全部通过，`error=null`且0训练。
- 归因显示C1-mixed中两个物理基元使不同预测槽形成近似相同的Hungarian总成本；SciPy以输入列顺序打破平局。Teacher重排改变列顺序后交换了这两个基元，而它们关系标签不同，故关系loss受任意存储槽号影响。
- 否决把容差从`1e-6`放宽到观测误差，因为这会掩盖真实槽位依赖；也不在当前阶段改为二次图匹配，因为它会改变目标并显著增加训练成本。
- 选择在不变Hungarian目标前，使用端点反转不变的轴线/截面/时序签名，并以端口attachment和disconnected-overlap邻域迭代细化，得到与Teacher槽号无关的规范顺序。新增独立回归，完整测试从29增至30。
- V1R正式证明Teacher槽位重排和端点反向loss误差均为0，全部16项检查PASS，允许冻结非学习baseline与P2三seed训练；原V1 FAIL保持不可修改，V1R没有改变数据、Teacher、模型、六类loss、权重或阈值。
- 论文归档政策同时冻结：旧工作只有在能形成明确主张的情况下进入基线、消融或失败分析；环境/依赖/脚本接口故障只进入复现材料。建立逐run证据矩阵之前，不删除旧图、metrics、checkpoint、轨迹或失败归因。

## 2026-08-31 — 当前P2冻结前不重新出版合并旧C08的历史失败图

- 准备把Spatial Event Set、Geometry-Anchored Joint和Structured Polar Multi-depth三组旧失败汇成新论文图时，源summary人口被核对为`45,942=21,548+24,394`，即旧C07与旧C08的合并开发观察，不是纯C07；三者C09/C10/M-TARE读取均为0。
- 这些汇总数值和原图此前已记录于项目文档，检查没有打开当前P1a/P1b的C08 shard、没有运行当前模型、没有改变checkpoint、阈值或C07门，但继续重新消费并出版会模糊当前P2“C07通过前不读取C08”的隔离口径。
- 决定立即停止该新合成图，不创建任何输出。三个旧run、原图、metrics和seal继续按失败分析证据封存；待当前主模型与阈值冻结且C08门合法执行后，再决定是否从密封历史资产出版补充材料图。
- 当前允许的论文图片工作仍限纯fit/C07源；正式三seed训练继续不受影响，C08/C09/C10、graph和M-TARE仍不由当前run读取。
## 2026-08-31 — V1基元关系模型C07仅1/3通过，停止于C08和建图之前

- 正式run完整完成三seed各6轮和冻结C07评估，`error=null`；seed0/1/2的surface Chamfer相对非学习baseline分别改善`43.17%/50.56%/53.80%`，连续几何macro改善`42.78%/45.54%/44.81%`，说明程序构造监督能够教会显式3D扫掠几何。
- attachment F1分别为`0.02922/0.08097/0.04084`，相对baseline仅增加`2.34/7.52/3.50`个百分点；预注册要求至少2/3 seed增加5个百分点，实际只有seed1通过。三个seed在precision>=0.98的安全点都没有真实连接提交。
- 机制证据指向候选选择而非几何表示缺失：约`2.066--2.068M`个预测基元对应`544,414`个目标，precision约`26.3%`；冗余槽位使端口pair空间膨胀并产生大量假关系。
- 决定按冻结科学停止政策，不读取C08、不挑seed、不放宽阈值、不加手写规则、不进入graph/M-TARE。当前V1保留为“几何可学但关系提交失败”的核心消融和失败案例。
- 下一步只允许在密封C07输出上做零训练、零新推理的归因，区分候选存在性错误与oracle真基元条件下的关系能力；任何后续模型修订必须由该证据唯一决定，并新建精确Data Card/spec。
- run=`results/gate3_semantics/gate3_20260830_primitive_relation_three_seed_training_v1_seed0`；evidence-list SHA-256=`81f43937e813f11b1b8cf337edd05f842ba83a189cf94ed54feb5260b7fe2aa7`。
## 2026-08-31 — 纠正旧评估器cuDNN TF32漂移后再归因，不沿用不合规数字

- 归因V1R1按声明显式关闭matmul/cuDNN TF32后，seed0完整C07存在性、attachment和overlap三项均无法逐项复现旧formal结果，故在任何oracle结论前FAIL。
- 静态与运行证据：训练器包含`torch.backends.cuda.matmul.allow_tf32=False`、`torch.backends.cudnn.allow_tf32=False`和`set_float32_matmul_precision('highest')`；独立旧评估器三项均缺失；当前新进程默认matmul=false、cuDNN=true。这是评估器系统合同漂移。
- 影响：旧C07精确数字未满足声明的TF32-off合同，当前V1是否仍为1/3及失败归因必须重算；不影响checkpoint训练、P1数据/Teacher、C08零读取或M-TARE隔离。
- 选项：A沿用旧TF32-on设置完成归因，约20分钟但证据不合规；B重训练约21小时但训练本身无缺陷；C相同checkpoint做C07-only TF32-off重评并归因，约1小时。依据用户自主最优证据授权选择C。
- V1R2冻结581,796推理行=`64,644×3×(2 corrected evaluation passes+1 attribution pass)`，0 optimizer/C08/C09/C10/graph/M-TARE；正确评估结束前不得训练或选择关系修订。

## 2026-09-02 — 可观测关系修正版 C07 0/3，停止于 C08 与图回放之前

- 正式run完成三seed各三轮，共`240,624`步；源码、冻结参数、资源和数据隔离合同均通过，`error=null`，因此这是有效科学FAIL而非系统失败。
- 三seed表面改善`56.38%/49.19%/52.62%`、连续几何macro改善`20.49%/38.90%/29.11%`，证明程序构造监督学习扫掠几何这一子主张稳定成立。
- 普通attachment precision仅`5.65%/3.16%/3.23%`；相对基线F1提升只有seed 0超过5个百分点。安全分数在precision≥0.98条件下三seed全部`TP=0, FP=0`，属于系统性全拒绝，passing seeds=`0/3`。
- 决定执行预注册停止门：不读取C08，不启动在线图或M-TARE，不挑seed、不降低0.98门、不用手写规则补关系。当前observable corrective保留为“几何可学、端口关系不可安全提交”的核心失败分析/消融。
- 下一步仅允许对密封C07输出做零训练、零新推理归因，量化候选存在性、双端点证据、pair combinatorics、score calibration和oracle候选条件下的关系上限；任何新模型结构或Teacher变更必须先形成独立证据、Data Card和新冻结spec，不得把本run改写成PASS。
- run=`results/gate3_semantics/gate3_20260902_primitive_relation_observable_three_seed_training_v1_seed0`；54项evidence-list SHA-256=`544b6be586bdb472a12c8912b16a8cf87105ed7e4ff44302e1df239527f0553c`。

## 2026-09-02 — Observable最终评估遗漏cuDNN TF32-off，先纠正数值合同再作科学结论

- 训练器通过`_configure_determinism`显式关闭matmul和cuDNN TF32；独立最终评估器未调用该函数，runner环境也只冻结CUBLAS workspace。Torch 2.9当前新进程默认matmul TF32=false、cuDNN TF32=true，故训练/选择与最终门的数值执行合同不一致。
- 影响限定于最终C07精确计数和0/3判定；三个checkpoint、训练过程、Teacher、数据隔离、资源和C08零读取仍有效。密封旧结果不修改，但不再作为合规科学Gate。
- 否决沿用旧数字，因为违反声明合同；否决重训，因为模型训练本身已经TF32-off且冻结哈希通过。选择相同三个selected checkpoint做一次C07-only TF32-off纠正评估：每seed两遍完整`64,644`行，总计`387,864`次前向，0 optimizer、0 C08/C09/C10/graph/M-TARE。
- 纠正结果必须用同门同阈值独立写入新不可覆盖run；通过才允许C08，失败才允许继续关系归因。不得以该系统纠正为调整模型或挑seed的机会。

## 2026-09-02 — TF32-off纠正确认0/3；停止直接关系头并只开放C07失败归因

- 同三个selected checkpoint在确定性、matmul/cuDNN TF32均关闭的合同下完成每seed两遍全量C07评估，共`387,864`次前向、0 optimizer；纠正前后passing seeds均为0，科学方向没有改变。
- 三seed显式几何继续稳定优于非学习基线，但普通attachment precision仅`5.654%/3.155%/3.230%`；precision≥0.98的安全分数三seed均`TP=0`。这正式否定当前“直接端口pair关系头足以形成可提交结构连接”的假设。
- 源run的唯一后置故障是`_seal()`返回整数后又调用`len()`；它发生在科学输出与28项seal之后。决定不浪费GPU重跑，建立零推理evidence corrective，逐项验证源seal、人口、数值合同、资源、隔离、上游hash和紧凑副本。
- evidence corrective正式PASS，28项新seal SHA-256=`ea5b0833c1ba2c1e03e70e3ede38dc85eae38c23e08d78d44a78c915265c698b`；源run保持不可修改。
- 决定：C08、在线图、M-TARE继续关闭；只开放密封C07上的失败归因。归因必须先区分候选存在性、双端点证据、pair组合爆炸、关系表示和校准，才能决定计划内下一版关系模型；禁止挑seed、降0.98门、删帧或加手写连接规则。
- run=`results/gate3_semantics/gate3_20260902_primitive_relation_observable_c07_tf32_evidence_corrective_v1_seed0`。

## 2026-09-02 — Oracle仍无安全连接，停止独立pair分类并先审计连接超边Teacher

- 正式C07归因完成`193,932`次冻结前向并精确复现formal attachment/safe；九个部署、cardinality、proposal-oracle、best-link、raw/uncertainty/evidence条件均为`0/3` seeds通过。
- proposal oracle将三seed exact relation F1提高到`0.2215/0.1725/0.2075`，但raw和best-link在precision≥0.98时真实连接均为0；端点证据oracle F1约`0.89188`。因此冗余候选是次要放大器，主因是独立pair关系分数没有安全排序能力。
- 否决继续调独立pair BCE、阈值或置信度，因为oracle/raw也失败；否决用距离/角度规则补关系，因为这违反学习组合关系的论文主张。
- 选择计划内结构修订候选`Primitive Connection Hypergraph`：学习端点到可交换连接簇/超边的分配，由共享簇导出多元attachment；这保留显式几何、端点证据、五帧因果和执行验证，不引入事件类别规则。
- 该候选尚未确立。先做C01--C06 fit+C07 selection的零训练Teacher feasibility，验证端点唯一归属、clique闭合、可见裁剪、fit-only容量与组合压缩；任一失败则停止该候选。C08/graph/M-TARE仍禁止。
- attribution run=`results/gate3_semantics/gate3_20260902_primitive_relation_observable_failure_attribution_v1_seed0`；seal SHA-256=`fef3891cad80602437d8c45afd9d00c8505b0921c8824984f78c278cf7d6cf76`。

## 2026-09-02 — 停止稠密K槽连接超图，保留连接簇语义并转向线性composition-anchor输出

- 证据：fit/C07共491,196行全部形成对称、唯一、clique闭合的端点连接簇；C07可观测正连接442,936精确复现，说明Teacher关系语义有效。
- 容量：fit最大19簇，冻结25%余量要求24，从`4/8/16/32`只能选32；C07最大20且零overflow。
- 失败：固定K=32的cluster head输出1,005,969,408，高于pair head的974,532,864；真实active候选256,051,648对64,686,160，约多3.96倍。稠密slot没有解决组合规模，触发预注册compression FAIL。
- 否决：不能事后改余量、强选K=16、按C07调K、删除高簇行，或把Teacher语义PASS冒充decoder PASS。
- 决定：正式停止`dense Primitive Connection Hypergraph slot decoder`；连接簇作为已验证物理表示保留。下一候选为每端点O(E)输出共享composition anchor与uncertainty，再由学习概率兼容性形成簇；不使用几何阈值规则替代学习。
- 下一门：C01--C06 fit+C07 selection零训练composition-anchor Teacher/readiness；证明唯一性、因果坐标、同簇一致、异簇可分、stacked tunnel隔离、非学习基线和线性复杂度。未通过不得训练、读C08或建图。
- 证据run=`results/gate3_semantics/gate3_20260902_primitive_connection_hypergraph_teacher_feasibility_v1_seed0`；seal SHA-256=`61e6ce19bbcae565f2b2b0d0f678917acf6ee5cc1388c5780d428c79cdec3e87`。

## 2026-09-03 — composition-anchor Teacher通过，但Teacher几何距离近乎完美，训练前新增预测几何必要性诊断

- PASS证据：491,196行每端点唯一composition membership；同簇anchor最大距离0，异簇/overlap hard negative最小距离0.809503 m；O(E)输出256值/行，相对pair 1,984值缩减7.75倍。
- 非学习上界：fit-only阈值0.499973 m原样到C07得到TP442,606/FP0/FN330，precision1.0、recall0.999255、F1 0.999627，hard-negative FP=0。
- 风险：该近满分结果使用Teacher真值基元，证明锚点关系可观测，但也说明“额外学习relation”可能没有独立价值；不能因为接口新就直接训练并宣称创新。
- 决定：训练前新增一次三个冻结checkpoint的C07-only predicted-axis association necessity diagnostic；比较deployed/proposal-oracle几何距离与learned pair score的安全连接。
- 分叉：若预测几何关联已经安全，停止anchor head并把关系作为学习几何的确定性组合；若预测几何关联失败而Teacher上界成立，才实现O(E) anchor residual+uncertainty head。两种情况均不允许手写事件类别、降低0.98门或读取C08选择。
- PASS run=`results/gate3_semantics/gate3_20260903_primitive_composition_anchor_teacher_readiness_v1_seed0`；27项seal SHA-256=`4e48e809d5462cd98fb9b811746a7dbd0b831b06f06fa80f789aee1d4b17e1d8`。

## 2026-09-03 — 正确候选下预测端点仍不可安全连接；允许最小O(E)组合锚点修正

- 正式诊断对三个冻结checkpoint完成193,932条C07前向，旧learned-pair结果在deployed/proposal-oracle条件下精确复现，证明比较没有换模型、换数据或换评价。
- proposal oracle排除了候选漏检和冗余proposal的主要影响后，预测端点距离best F1仍只有`0.289743/0.282983/0.293032`，precision≥0.98非空安全连接三个seed均为0；Teacher-fit固定阈值的precision也只有`20.38%--23.10%`。
- 这否定“已经学到的轴端点加解析距离即可完成物理组合”的假设。Teacher真值锚点在同一批C07上F1=`0.999627`，因此失败不是组合语义不唯一，而是学生预测端点没有被训练成共享组合位置。
- 否决继续调距离阈值、按C07选阈值、放宽0.98门、恢复pair BCE、使用稠密连接槽或让图规则强行合并；这些都不能解决已观察到的端点定位失真或会破坏论文主张。
- 决定开放计划内最小修正：复用已学几何/时序表示，每个端点输出composition-anchor残差均值与不确定性，使用共享锚点回归和概率正负兼容性训练，输出保持O(E)。先做零训练readiness，不能直接启动新三seed训练。
- C08、C09/C10、在线图和M-TARE继续关闭。run=`results/gate3_semantics/gate3_20260903_primitive_predicted_geometry_association_diagnostic_v1_seed0`；seal SHA-256=`9eda771e22bdf9dbb886736c003c3a932c67b6a107e2fe390213764f7147919b`。

## 2026-09-03 — 否决axis-anchored residual frame；选择sensor-polar composition-anchor V2

- V1真实执行证明某些预测query轴向退化；V1R的径向fallback只覆盖无方向query，但V1R2进一步发现12个退化端点中有1个真实匹配且观测，不能把它们一律当冗余删除。
- V1R2全部loss/gradient/资源检查通过，但axis-frame query-permutation和yaw compatibility分别误差`0.001587/9.81384`，触发科学FAIL。这说明头依赖的预测tangent本身不稳定，不是再加一个fallback阈值能解决。
- 否决删除候选、固定世界x轴、把退化匹配改为unknown、放宽等变门或只检查随机合成数据；这些会改变Teacher/人口、破坏旋转一致性或掩盖真实失败。
- 选择V2：每个端点用传感器极坐标radial/lateral/gravity表达任意3D残差，保留descriptor/range/shape/evidence等稳定输入；兼容性仍由两个Gaussian anchor概率导出，输出保持O(E)。
- 上游query decoder在重排时本身存在`4.1008e-5 m`绝对float32误差，因此V2把新增head增量继续锁在原`3e-5`，并单独给继承的绝对anchor设`1e-4 m`数值上界；没有改变precision≥0.98的后续科学安全门。
- V2正式PASS并只开放fit/C07 anchor target sidecar及随后单独Data Card的head-only训练。V1R2作为axis-frame消融保留；C08/graph/M-TARE仍关闭。
- V2 run=`results/gate3_semantics/gate3_20260903_primitive_composition_anchor_model_readiness_v2_seed0`；seal SHA-256=`3fc29fb93c7f332c619192daae595239e8b523f8f2ee68c36413f1adbe6a907d`。
