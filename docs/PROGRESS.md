# PROJECT PROGRESS

## 2026-09-05：首个GPU优先对照完成，三线已实际交付

- DONE：RTX5090原C01同预算坐标干预，180观察/900帧/1080步，30.3838s；39seal复核、11图全部保存。raw/mean坐标1.780/2.457m，但横向2.336/1.959m，属于指标取舍，不是整体几何通过。
- DONE：CPU端口更新和6DoF坐标接口，图相关70项合成测试；主代理含坐标消融共82项复核通过。真实配准/图接线尚未完成。
- DONE：局部结构教师草稿，明确多实例参照、GT仅loss/评分、局部开口证据缺口；0新标签/0结构训练。原180只有mixed，不能在此宣称跨几何配对实验完成。
- NEXT：区域/开口合成合同和原180/900监督pilot精确卡准备；并行图软件接线。标签有效后直接做GT/预测/无组合短实验，不等待几何完美；不因空闲GPU而启动未就绪长训练。详见`GSE_COORDINATE_CONTROL_V1.md`。

## 2026-09-05：用户要求GPU优先与多线并行，C02对照已完成

C02 raw/old坐标MAE2.031974/3.922121m，10父地图均改善，但横向2.643606m和4271剩余候选说明不是完整检测/结构成功。20.068515s、0optimizer、309tests、28seal；11图全部保存，见`GSE_HEAD_DEVELOPMENT_INFERENCE_V1.md`。

当前推进三线：GPU原C01同预算坐标对照独立card/runner；CPU当前区域/端口监督合同；CPU图端口更新/配准复用。用户新方案明确几何不必全达标才能做受控结构实验。最先就绪且通过预检的GPU任务优先执行，不空跑三seed。完整目标已建立为active；没有新训练已启动的声明。

## 2026-09-05：C02开发清单确定，准备独立冻结推理

预冻结按每图18个中点分位选样，实测10父地图/180观察/900唯一帧/1489片段/180有向穿越；仅341个整数元数据文件、0几何/扫描/模型，0.120676s、RSS69660672字节、13项seal通过。负值mask软件漏洞在真实读取前修复，145项卡/reader/完整执行测试通过。运行后源过滤增强单列新测试，原冻结文件字节保持不变。唯一NEXT：新独立冻结推理卡和执行器，raw小头/旧模型各180主+18重复，无训练，无C07--C10或图；不能复用元数据授权读取模型。

## 2026-09-05：缓存错配对照完成，raw并非固定候选库；尚未验证跨父地图

- DONE：固定父地图内shift9、180观察/1452目标、四种输出各correct/shuffled，0训练/模型/原始扫描，2.966s，RSS620572672字节。原分数/匹配复现，8份全量逐观察证据、SVG和22seal校验通过。
- RESULT：raw正确/错配MAE1.780/5.220m、横向2.336/6.531m、角16.846/34.271°，10父地图全部正确更好；不是固定候选库，但仍有4308多余候选未检测惩罚，未排除训练记忆/运动相关性。偏移版停止扩展不变。
- SYSTEM：V1因object-only读取器遇到列表在0评分失败；V1R只修类型接口，增加真正端到端合成测试；23新测试/446指定回归PASS。旧run、权重、论文图保留。
- NEXT：C02十父地图精确元数据与冻结raw小头开发检查采样/card准备，尚不读数组/推理；旧骨干见过C01--C06，不能叫整模型未见测试。C07--C10、事件/图/闭环未开放。详见`GSE_AXIS_OBSERVATION_DEPENDENCE_V1.md`。

## 2026-09-05：固定预算小训练已完成，停止扩展偏移分支

- DONE：同180观察/900帧/1452片段，两分支各540步，1080步，15.977s，骨干冻结及缓存复用精确。两份最终权重、全部32候选和逐观察指标保留。
- RESULT：无偏移raw坐标MAE1.779605m，slot偏移2.825677m，旧冻结模型同matcher3.611578m；偏移版在10/10父地图上更差，STOP_BEFORE_EXPANSION。FIT匹配分数不罚多余候选，不是识别或泛化成绩。
- SYSTEM：原run在最终绘图因parents列表被整数覆盖失败；独立补全V1环境目录初始化失败。两者保持FAILED。V1R只补证据17.372s，0训练/推理/原始数据，匹配和原判定复现，11图/22seal通过，423项回归通过。
- NEXT：缓存预测的观测依赖/布局检查规格与软件，固定同父地图错配对照，不先重训。事件/端口监督、C07--C10和闭环仍停止；详见`GSE_POINT_AXIS_PROBE_V1.md`。

## 2026-09-05：点级轴线读出已接到骨干适配器，软件验证完成

- DONE：35,460参数点级分支，原XYZ/相对token位置可区分；slot条件表面→轴线偏移解除纯表面凸包限制；部署forward无教师身份，骨干eval/no-grad不变。
- FOUND/RESOLVED：每点唯一offset可能与多源射线监督冲突；真实teacher生成前改为slot条件低秩offset，共享offset仅保留消融。未删点、未改单标签、未生成监督。
- EVIDENCE：唯一`gse_point_axis_software_v1`预检通过后单次执行，48/48、2.120s、RSS1018449920 bytes；完整57,600点/32slot随机骨干前后权重相同，新头梯度有限；11项seal全过。
- REGRESSION：封存后指定坐标诊断/治理/组合/图等196项回归通过，2.82s。
- LIMIT：算子SE3不等于神经网络等变；背景主导也可能输出坐标，支撑量/方差/有效点数不是已校准安全证据。0数据集帧/checkpoint/optimizer，非训练或科学PASS。
- NEXT：原180观察的独立几何小训练卡/spec/executor，slot偏移与无偏移raw同输入同预算，旧冻结几何作参考；先核对冻结seed0输入复用/现有教师及小批资源。完整事件/端口监督、C07--C10和闭环仍停止。详见`GSE_POINT_AXIS_READOUT_V1.md`。

## 2026-09-05：真实坐标对照确认旧读出的表达瓶颈，停止只换点池的简单修补

- DONE：原180观察/900帧/1452片段/4356控制点，正式坐标支持对照16.254s完成，0模型/optimizer/新标签。27项seal全部独立通过，约12 MiB证据及11张图保留。
- RESULT：均值池3153/4356控制点在凸包外；2688点均值池外但raw池有重构见证；465点两种池都无法以纯凸组合表示。10父地图都存在均值损失。2个未定项保留。
- LIMIT：这是位置表达能力，不是新模型89.33%准确率；全局重构可能混合不同隧道，不能传给部署学生。未证明465个raw外点都是不可观或教师错误，不能删除。
- TEST：新计算/卡/执行器68项测试及指定扩展回归381项PASS。检查发现coplanar hull三角面见证和卡重复身份风险，正式run前修复；旧run不变。
- NEXT：薄改几何读出的软件规格与合成验证。原XYZ必须有点级可区分归属；仅广播旧token权重仍等于均值或引入回波数偏置。需支持表面到凸包外轴线的估计并检查梯度/退化；新训练单立卡/spec，不增加数据或自动恢复长训练。Phase 3科学门未升级，事件监督/C07--C10/闭环仍关闭。

## 2026-09-05：分解证明布局也错，定位到可检验的坐标读出限制

- DONE：`gse_axis_error_decomposition_v1` 6.150s完成，同180行/1452匹配/原分数精确复现；0扫描/forward/optimizer/新标签。
- RESULT：横向中位2.263m、无向角37.933°、双向有限折线距离5.201m；当前节点所属分别2.704m/32.416°/5.528m，不是仅foreign问题。82.58%是沿轴残差平方占比，不是只有裁剪错的样本占比。
- UNKNOWN：教师方向6、预测方向22不可分辨，保留全部记录。积分误差界最大0.0772m；数值近似解释不了米级布局偏差。
- EVIDENCE：11张图、逐片段/父地图统计、26项seal全部通过；seal-list=`cf05614525e2b0107749647f839c7b721049187d6d07bf7fc45d913db50bef5b`。41项新诊断测试+4项池化反例；指定回归313/313通过。
- FOUND：当前16竖直线×4方位列坐标均值加softmax读出，在合成上下层反例中无法恢复z=±3m，后端logit梯度为0。它是读出限制证据，不是实际180样本原因已定。
- NEXT：同180原始/均值坐标池的可表达性比较，独立精确卡/spec，0模型/训练/新标签；先验证真实影响再薄改前端，组合头长训练、C07--C10及闭环继续停止。

## 2026-09-05：现有教师自洽，预测裁剪轴线误差需要分解

- DONE：唯一`gse_local_teacher_audit_v1`完成180观察/10父地图/100旧节点/900帧引用；1452片段，支持/时间可见性与全部可见构造连接一致。438源块哈希通过，0扫描/推理/optimizer/新标签。
- FOUND：180行都有当前节点之外片段；178行有角投影重叠，不能解释为物理重叠。新节点标签仍不能从degree、片段存在或旧端点掩码直接得到。
- RESULT：全匹配的轴点欧氏误差均值10.882m、坐标MAE4.780m、半轴MAE0.539m。当前节点所属片段均值13.143m，不能只怪foreign。该指标包含轴向裁剪误差，不是横向/节点误差；与旧阈值筛选后的几何指标不同人口。
- EVIDENCE：5.758s、RSS685658112 bytes、10张全部180行XY/XZ图、逐行/逐父指标、日志、24项seal全部复核；seal-list=`69f17706af3a3dc36f3cd01ca7e9ae5d64b8d60fe906270133f50212c09e4642`。42项新测试/指定扩展回归268项PASS。
- NEXT：同180行零训练分解方向、沿轴偏移、横向偏差及截取长度，先决定修前端还是监督接口。不把此诊断宣布为新方法FAIL/PASS，不重训、不删样本、不改阈值，C07--C10和闭环保持关闭。

## 2026-09-05：原 180 观察共同坐标字段正式恢复，旧缓存精确复现

- DONE：`gate3_20260905_gse_composition_field_recovery_v1_seed0` 单次完成，error=null；180 次主推理+18 次重复，900 个唯一 C01 源帧，0 optimizer/新标签/C07--C10。
- RESULT：旧 `180×64×44` 特征和 `180×64` 置信度逐元素一致；六项原始几何字段重复一致；模型状态 SHA 不变，全部参数冻结且无梯度。不是新模型精度或建图 PASS。
- COST：6.137s，host RSS 1899761664 bytes，GPU allocated/reserved 1109133824/2082471936 bytes，输出约 1.1 MiB；访问的 541 个源 chunk 均核对旧 seal。
- EVIDENCE：10 个原始六字段 NPZ、行/帧 manifest、日志、环境、卡/spec、RUN_STATE、23 项 seal。seal-list=`478e0744a7abaf5281342722a037b4aa1267f23b9b89377f5dd67415fed35481`，23/23 独立复核通过。
- TEST：字段恢复/卡/reader 与旧模块扩展回归 255/255 PASS，1.68s。
- NEXT：同 180 行的教师几何/组合/非 incident 歧义核对卡/spec，结合恢复的预测检查节点/端口目标及 UNKNOWN 边界；旧缓存缺字段阻塞已解除，但完整结构监督尚未通过，因此不训练。

## 2026-09-05：通道口射线证据和冻结骨干输入 reader 已实现

- DONE：双向有限射线跨截面正证据、UNKNOWN/重合候选拒绝、mesh 截面适配及精确五帧 reader。原数据/标签未修改，0 实际扫描解码/模型/optimizer。
- REVIEW：独立审查复现 float32 端盖误判、门面起点刚体变换翻转、SDF 与 mesh 端盖采样错位。三项均在真实运行前修正；新增多边形边界回归，不用理想椭圆扩大 mesh 开口。
- TEST：新增两模块 40 项合成测试通过；完整回归 147/147 PASS，1.31s。全部是软件证据，不是教师/模型精度结论。
- ENV：冻结 Torch 2.9.0+cu129 在主机设备访问模式实际运行 CUDA 标量成功；RTX 5090 D。此前不是驱动/硬件故障。
- NEXT：冻结原 180 观察的共同坐标字段恢复卡/spec；指定 seed0 冻结推理与旧缓存一致性核对，不训练、不生成未经验证的事件标签、不读 C07--C10。监督剩余边界见 `GSE_NODE_OBSERVABILITY_READINESS_V1.md`。

## 2026-09-05：同 180 观察核对完成，发现物理端点与节点可见性的接口错位

- DONE：限定 C01 的 10 个父地图、180 个五帧观察、100 个节点；实际确认 900 个唯一源帧，不是直接用窗口数相乘推断。仅解码元数据，0 原始扫描/模型/训练。
- EVIDENCE：库存 V1 在读取分片前因 JSON 列表读取器错误失败并封存；V1R 仅修解析器，5.103s 完成、350 项源 chunk 校验、14 项输出 seal 全部复核。旧 run 不覆盖。
- FOUND：40 个分叉观察都有全部 incident 基元，但旧物理端点支持仅 6 个观察完全满足，来自 3 个不同分叉。解析 T 形反例证明端点壁面支持不是看到支路的必要条件；既不将剩余 34 个删掉，也不直接把真实 degree 当新事件标签。
- IMPLEMENTED：只读 allowlist、源 chunk 哈希与缺块失败检查、完整合成执行器测试、新可见基元输入桥；针对性测试 82/82 PASS。原物理端点桥保留比较，0 新训练标签。
- CORRECTION：V1R summary 的 `support_is_necessary_not_sufficient_for_event_label=true` 是未经验证且不适用于新节点任务的解释字段；数值清单有效，该解释由本条与 DECISION_LOG 明确撤回，密封文件不改写。
- ENV：主机 RTX 5090 D / 32607 MiB 可用；此前限制来自沙箱设备访问，不是硬件失效。未因此自动启动训练。
- NEXT：在同 180 行范围设计并验证节点级可观测监督与字段恢复合同，区分几何片段、可见支路和物理端点。新 Data Card 前不解码扫描、不推理/生成标签；不进入 C07--C10 或闭环。

## 2026-09-05：首批组合/因果图软件证据已完成，56/56 PASS

- DONE：共同坐标组合头及结构一致性/未知掩码；首次节点、逐段真实轨迹、歧义保留与前缀一致内核。新旧共 56 单元/回归通过。
- RUN：`gate3_20260905_gse_composition_software_contract_v1_seed0` 一次完成，1.684s，约 582 MiB 子进程 RSS，11 项 seal 复核通过；0 数据集帧/训练/闭环，不是科学 Gate PASS。
- FOUND：180 缓存只有旧特征/置信度/描述/度数/行号，缺新输入与端口标签；度数人口 20/120/38/2。通用 reader 初始化会打开全部分片，补充读取须显式 C01 allowlist。
- NEXT：任务/行 allowlist 软件合同、同 180 行精确元数据与可观测标签核对、Data Card；在此之前不启动训练。完整说明见 `GSE_COMPOSITION_IMPLEMENTATION_STATUS_V1.md`。


## 2026-09-05：用户批准新主线，开始合成接口与因果图实现

- DONE：明确新方案替代等待批准状态；本地 Git 历史基线已建立，保留所有历史科研资产。
- DOING：共同坐标几何组合、事件/端口接口、首次节点确认和连续实际穿越状态机；仅合成单元测试。
- LIMIT：Torch 2.9.0+cu129 可导入，当前 CUDA 不可用；尚未开始训练。合成测试不代表学习成功或 Gate PASS。
- NEXT：完成合成验证后，核对原 180 观察缓存及端口标签，明确缺失信息，再冻结数据相关 run。


## 2026-09-04：图上下文联合方案已完成方法边界草案，避免重复旧Factorized失败

- 新草案=`docs/GSE_GRAPH_PRIMITIVE_CONDITIONED_HYPOTHESIS_GRAPH_PROPOSAL_V1.md`，状态为`DRAFT_REQUIRES_EXPLICIT_METHOD_BOUNDARY_DECISION`，未创建数据、训练或图run。
- 历史Factorized路线已逐层复核：关联容量和C09 consensus曾达balanced P/R=`1.0/0.5077`及runtime false accept=`0.9865%`，所以图条件关联不是重新发明；真正失败是新节点、loop merge和edge commit生命周期耦合。
- 旧回放中274个真节点仅唯一提交192个，13条真trace relation仅建2边；这与双独立traversal节点提交、same-traversal首末已提交trigger建边直接相关。
- 草案的核心改动：稳定五帧结构事件即建立local-confirmed node；只有与loop候选合并需严格图因子；edge由departure port到arrival node的连续真实执行闭合。
- `NEXT`：用户确认方法边界后，先做Teacher/Teacher、Teacher/当前基元、当前事件/Teacher和全当前四档零训练回放；oracle node/edge P/R必须为1才允许event-only tiny overfit。

## 2026-09-04：双读出corrective在8秒内仍FAIL；停止该学生接口

- V1R只因Data Card汇总world别名无法通过预检，0 run/0 training；V1R1仅修正为10个逐世界trajectory记录，同180观察/100节点/80正对/500步正式执行。
- V1R1 `COMPLETED`、`error=null`，但loss reduction=`0.56397<0.95`，几何RMSE=`0.10883>0.01`，degree accuracy=`0.97778<0.99`，precision>=0.99时非空关联为0。
- 关联子空已将不同节点距离中位数拉到`1.15288`，但最小仍为0；部分重复结构在局部基元观测下仍完全混淆。因此不能通过延长训练合理化。
- 运行用时`8.20s`，GPU allocated/reserved=`0.371/0.476 GB`，host在合同内，22项证据密封；seal-list SHA-256=`d19fb26fd7f14dc2c423e4ff89e62f4b7d6dd96640c986acc5895a22a2bbf803`。
- `STOP`：不做单seed/三seed的局部place-identity训练。`DECISION REQUIRED`：将主方法固定为学习几何结构决定节点/端口，再用候选描述+里程计/邻接/执行一致性联合关联。新方法规格冻结前C08、graph replay和M-TARE仍关闭。

## 2026-09-04：180观察tiny overfit在12秒内FAIL；长训练已拦截

- 唯一正式run `gate3_20260904_gse_structural_node_tiny_overfit_v1_seed0`已`COMPLETED`且`error=null`；180个C01五帧观察、100个节点、80个正对、500步及冻结骨干均按合同完成，C07--C10/graph/M-TARE为0。
- 节点度数准确率=`0.99444`，但总loss仅下降`0.69449`，几何RMSE=`0.14321`，precision>=`0.99`时安全关联为0，因此正式科学FAIL。
- 诊断显示冻结输入的同节点/不同节点距离中位数=`0.22881/0.22244`，不能用一个原始距离同时完成几何复原和地点身份区分。同tiny Teacher目标本身P/R=`1.0/0.9625`，所以不否定基元组合路线。
- run用时`12.24s`，峰值GPU allocated/reserved=`1.11/2.09 GB`、host=`2.35 GB`，24项证据密封；seal-list SHA-256=`57fc01e05cff81644cf0f840844eeed7ef610a4186dff703276537f512c05588`。
- `NEXT`：仅重设计节点读出层：几何按维归一化回归头+独立关联embedding，然后做同180/100、同500步corrective tiny overfit。通过前不做单seed长训。

## 2026-09-04：结构节点+16 m图一致性正式PASS；下一步tiny overfit

- 正式run `gate3_20260904_gse_structural_node_graph_consistency_attribution_v1_seed0`完整复现源机器PASS/研究合同FAIL，并保存全部53个C07 descriptor FP。
- descriptor-only C07为TP/FP=`2960/53`、P/R=`0.982410/0.933459`；固定16 m图候选后为`2960/0`、P/R=`1.0/0.933459`，1 m/节点最坏位置误差结果不变，十个family全部有TP。
- fit固定门后TP/FP=`18266/33`、precision=`0.998197`；53个C07 FP最近间距`19.1678 m`。有效人口已纠正为fit 36,312、C07 5,874，另记6条无五帧窗口短traversal。
- run耗时`88.98s`，0 optimizer/model inference/C08--C10/graph replay/M-TARE；seal-list SHA-256=`7ee483fe66954b676bfd4f52b81743d7838ca7b294b3a995c1929322390a3240`。
- `NEXT`：tiny overfit，只验证学生能否恢复节点级几何证据；数百步内不收敛即停，不直接启动三seed。

## 2026-09-04：节点证据88秒验证出强信号；机器PASS因漏门降级，转图一致性快速归因

- V1首先发现把view-dependent support ray count写入地点身份会使五帧C07接受数变成0；该失败run原样保留。V1R只把support移到不确定性，重新审计后五帧C07 precision/recall=`0.982410/0.933459`、TP/FP=`2960/53`，oracle=`1/1`，十个family全部有TP。
- V1R正式执行约`87.58s`，0 optimizer/model inference/C08--C10/graph/M-TARE；seal-list SHA-256=`12d81fe24342c7712d3ee99655f860fb0de45c12adb1f98b57b13cd9994e84a0`。
- 合同复核发现执行器漏判accepted-pair false fraction `<=1%`：实际`1.759%`；fit有效观察36,312也少于预期36,318，原因是6条短traversal没有五帧窗口。因此V1R是`FORMAL_MACHINE_PASS_RESEARCH_CONTRACT_FAIL`，不能开放训练。
- `NEXT`：先对密封结果做合同重评分，再对53个C07 FP执行零训练图一致性候选归因。目标是在不改变数据/精度门的前提下，用既有空间候选、相邻结构和执行轨迹约束把误接受降到1%以下；通过后才进入tiny overfit。

## 2026-09-04：端点关系度量三seed完整FAIL；启用节点级快速证伪漏斗

- 正式run `gate3_20260904_primitive_endpoint_relation_metric_three_seed_training_v1_seed0`已完成`30,699`次head-only更新，`RUN_STATE=COMPLETED`、`error=null`，三个seed、冻结backbone、输入和资源合同均完整。
- C07共`64,644`行、`442,936`个objective true pairs。seed0/1/2 best F1=`0.252722/0.163434/0.280575`，precision=`0.173568/0.118684/0.198213`；precision`>=0.98`时三个seed均为0 TP，complete-link也无非空安全簇，正式结果`0/3 FAIL`。
- 结论不是“完全没学到”：三seed均显著超过非学习F1=`0.006380`，但正负分数尾部高度重叠，数十万近邻/叠置非连接端点被误连。端点相似度只能保留为候选特征或消融，不能决定节点合并或建边。
- run耗时`31,180.728s`，最高GPU process约`14.87GB`、host RSS约`2.55GB`；61项证据文件和seal完整，seal-list SHA-256=`ab4f3ccf737ca7afb57e204002aab13ab9221cbdf339958fb7426fa4520d14d3`。C08/C09/C10/graph/M-TARE读取或执行均为0。
- 当前NEXT为零训练`Structural Node Evidence Funnel`：在C01--C06 `426,552`与C07 `64,644`既有序列上，将多端点、多帧几何组合聚合为节点级观测，比较单帧/五帧、端点metric/节点描述子、无拒绝/不确定性拒绝，并先证明precision>=`0.98`的非空节点关联与困难负例隔离。未通过前不启动新三seed训练。


## 2026-09-04：endpoint relation metric三seed正式训练已启动

- Data Card/run spec已冻结，preflight=`0 errors / 0 warnings`；22项训练/评估/聚类测试全部PASS。唯一run=`results/gate3_semantics/gate3_20260904_primitive_endpoint_relation_metric_three_seed_training_v1_seed0`现为`RUNNING`。
- 固定训练合同为seeds`0/1/2`各3轮、每轮fit=`426,552`序列、C07=`64,644`序列、每seed=`10,233`步/总计`30,699`步；head=`426,818`参数，backbone=`2,635,631`参数冻结。
- 当前seed0 epoch0正在正常运行，GPU进程约`13.63 GiB<16 GiB`，host约`2.18 GiB<4 GiB`；C08/C09/C10、graph和M-TARE仍为0。
- `NEXT`：自然完成三seed训练与冻结C07门；不根据中间loss改epoch、loss、数据、Teacher、阈值或seed。

## 2026-09-04：无槽端点关系度量readiness正式PASS；允许设计三seed训练

- 新接口为`64 endpoint features -> 64-D normalized relation embeddings -> symmetric calibrated pair score`，不预测32个任意槽，也不设置共享dustbin类。loss直接监督同composition compactness、不同composition separation、balanced pair logistic和disconnected-overlap rejection。
- 单阈值complete-link解码已经实现：只有两个簇之间全部cross-pair都超过阈值才合并，因此拒绝A-B、B-C高但A-C低的链式误合并；它是通用一致性约束，不添加junction/terminal等手写语义规则。
- 正式run=`results/gate3_semantics/gate3_20260904_primitive_endpoint_relation_metric_readiness_v1_seed0`在固定C01 batch128上17项测试、18项checks全部PASS。34/34个新head张量梯度有限非零，backbone梯度全缺失且state hash不变。
- head/backbone参数=`426,818/2,635,631`；重复误差0，primitive permutation embedding/pair误差=`1.27e-6/4.17e-7`，yaw最大误差=`5.96e-7`，pair score精确对称，complete-link重复逐位一致。
- CUDA allocated/reserved/process=`7,609,980,928/13,941,866,496/14,631,829,504 bytes`，host RSS=`2,061,764 KiB`；0 optimizer/checkpoint/C07-forward/C08/graph/M-TARE。16项seal逐项PASS，seal-list SHA-256=`ed55c036f3d0a7271002c15a778e940721085bb0a5ed07b095874c884e14ae87`。
- `NEXT`：冻结三seed训练/evaluator Data Card。C01--C06提供梯度，C07选择checkpoint和唯一pair threshold；必须至少2/3 seed同时满足F1增益、precision>=0.98非空、overlap FP和complete-link cluster安全门，才允许一次C08零适配。

## 2026-09-04：局部组合槽失败归因正式PASS；任意槽路线停止

- 正式run=`results/gate3_semantics/gate3_20260904_primitive_local_composition_slot_failure_attribution_v1_seed0`已`COMPLETED`、`error=null`，三个冻结seed合计完成`193,932`条C07前向，0 optimizer、0 C08/graph/M-TARE。源best/safe阈值、TP/FP/FN和候选数逐项精确复现。
- 真连接端点被完整argmax判为dustbin的比例为`72.12%/86.82%/90.49%`；联合margin在活跃真端点上的零率为`68.71%/86.56%/90.15%`。这解释了部署置信分数为何大面积为0。
- 问题不止是margin：去掉全部乘法、仅用同一best-slot时，三个seed的precision=`21.50%/17.89%/21.93%`，F1=`0.2403/0.2236/0.2687`；soft affinity的F1=`0.2482/0.2466/0.2830`，但安全非空只出现`1/3` seed。槽身份本身没有形成跨seed安全边界。
- 正式诊断=`SLOT_IDENTITY_AND_CONFIDENCE_NOT_CROSS_SEED_SAFE`。当前32任意槽+dustbin主接口停止；checkpoint、图和因子表保留为“任意离散槽导致dustbin主导与关系混淆”的论文失败消融。
- 运行峰值host RSS=`3,129,147,392 bytes<4 GiB`，结果约`0.94 MiB`，28项seal逐项PASS；seal-list SHA-256=`d87f601a7cfc2d529df26aba94d14565f41ee24e6160de20a835671fc2e18f69`。
- `NEXT`：零训练实现无任意槽编号的endpoint relation metric readiness。同Teacher clique做对比正例，不同clique/dustbin/stacked-overlap做困难负例；pair score须对称、cluster解码须保守传递且支持拒绝。先验证真实batch梯度、置换/yaw、资源和Teacher不进入forward，才能另立三seed训练。

## 2026-09-04：局部组合槽三seed正式训练完成但C07仅1/3通过；转入零训练因子归因

- 唯一正式run=`results/gate3_semantics/gate3_20260903_primitive_local_composition_slot_three_seed_training_v1_seed0`已`COMPLETED`、`error=null`。三个seed各训练`10,233`步，总计`30,699`步；fit/C07人口固定为`426,552/64,644`条五帧序列，全部冻结输入、资源、checkpoint和C08零读取合同通过。
- C07同一`442,936`个可观测真连接上，best-F1为`0.138322/0.141336/0.137033`，均高于非学习基线`0.006380`至少5个百分点；但precision不低于`0.98`时，seed0/1没有任何非空阈值，seed2只有`6`个true positive。因此通过seed=`1/3<2/3`，科学状态为FAIL，不能建图。
- 分数分布暴露明确异常：三个seed的endpoint confidence中位数和90分位数均为0；seed0/1的99分位也仅`0.02383/0.00734`，seed2安全阈值虽precision=`1.0`，却只产生`768`个非平凡预测簇，对比Teacher的`411,215`个簇。模型学到弱排序信号，但结构化拒绝几乎把全部连接压成0。
- 已核验共享`exact_ranked_selection`实现只在相同分数组的末端选阈值，不拆分tie；当前best-F1阈值0是可真实部署的“接收全部零分候选”，不是评估器虚增。因此主要问题仍在模型槽身份/置信因子，而不是阈值程序。
- 62项证据从仓库根目录逐项SHA-256复核通过；seal-list SHA-256=`f393c3c61e22c40546f450bfae9a3b4538b358b1829f3198b73d2db29600c39e`。一次在run目录内执行核验造成的missing仅为工作目录错误，不是证据损坏。
- `NEXT`：只用冻结checkpoint和完整C07做零训练因子归因，分别测量槽一致性、dustbin margin、slot presence、primitive existence、endpoint evidence、entropy及其组合的零值率、排序和precision>=0.98非空TP。不得重训、降门、挑seed、读C08或加图规则。

## 2026-09-03：32槽局部组合关系模型readiness正式PASS；进入三seed训练实现

- 新主接口已经实现为`64 endpoints -> 32 exchangeable composition slots + dustbin`：集合Transformer对端点建模，Hungarian匹配消除任意槽编号，同槽概率形成关系，endpoint evidence与分配熵用于不确定性拒绝；学生forward不含world/node/TNG/绝对pose/Teacher identity。
- 14项新模型/损失测试、7项Teacher测试和5项既有observable backbone测试共26项全部PASS。真实C01 batch128含`850`个连接簇、`1,770`个簇内端点、`99`个dustbin端点和`2,520`个disconnected-overlap上三角负pair；5项loss有限，`81/81`个新head参数张量梯度有限非零，冻结backbone梯度全部缺失。
- V1在真实forward/backward后生成slot permutation时，以CPU generator直接请求CUDA randperm而system FAIL；运行`3.87 s`、无科学结论，12项seal SHA=`ffbb1e02d095193d121bd92d2fb8b1a975dd7783c7e3b0992a522efb5cc0d047`。V1R只将同seed置换先在CPU产生再复制到CUDA，模型、数据、loss、batch和门槛不变。
- V1R全部22项formal checks PASS：head/backbone参数=`1,001,507/2,635,631`，repeat bit-exact；slot输出/关系最大置换误差=`1.64e-7/3.73e-9`，Hungarian-matched loss误差=`2.15e-6`，yaw误差=`2.38e-7`，safe score finite且exact symmetric，backbone state hash前后相同。
- batch128实测peak CUDA allocated/reserved/process=`7,612,281,344/13,950,255,104/14,640,218,112 bytes`，host RSS=`2,029,760 KiB`，均低于冻结16/4 GiB门。optimizer/checkpoint/C07-forward/C08/C09/C10/graph/M-TARE全部为0。
- 正式V1R run=`results/gate3_semantics/gate3_20260903_primitive_local_composition_slot_readiness_v1r_seed0`；16项seal逐项PASS，seal-list SHA-256=`45488881d3ef76f8d81aef6a54f71c7f116aac8ea9e57fd968412fc2a4ea463a`。
- `NEXT`：实现并冻结三seed head-only训练/evaluator。C01--C06提供梯度，C07按mean local-slot total loss选checkpoint并冻结单一结构化拒绝阈值；必须在同一`442,936`个可观测真连接人口上报告F1、precision>=0.98非空TP、overlap FP、槽使用/塌缩和不确定性。至少2/3 seed通过前不读C08、不建图。

## 2026-09-03：局部组合槽Teacher可行性正式PASS；允许进入关系模型readiness

- 唯一正式run=`results/gate3_semantics/gate3_20260903_local_composition_slot_teacher_feasibility_v1_seed0`已`COMPLETED`、`error=null`。零训练读取fit/C07共`491,196`条五帧Teacher序列，其中fit=`426,552`、C07=`64,644`；sensor、model forward、optimizer、C08/C09/C10、graph和M-TARE读取全部为0。
- 原始可观测连接关系可无损分解为交换不变的局部组合簇：attachment非对称行、非clique行、slot解码pair mismatch、disconnected-overlap误合并和重复推导差异全部为0。即“同槽表示同一局部连接事件”在全部开发Teacher上严格成立，不需要手写补边规则。
- fit每帧最大组合簇数=`18`，按冻结的25%余量需要`23`个槽，因此从候选`8/16/32`中选择固定容量`32`；C07最大=`19`，overflow行=`0`。单簇最大端点数为4，符合地下junction局部组合语义。
- 审计仅用`84.40 s`、峰值主机RSS=`123,375,616 bytes`、结果目录`860 KiB`。27项证据seal逐项校验通过，seal-list SHA-256=`ee6ddaecbcce9959602e11f976aaea4d9144de354f98557877ddd8567a01145c`。
- 人话结论：旧模型失败是因为要求它从局部LiDAR猜十几米外的绝对连接坐标；新的接口只要求端点在当前局部结构中选择“属于哪个连接组或不确定”，目标本身已被证明一致、容量足够且不会混合断开的重叠隧道。
- `NEXT`：实现并审计`endpoint -> 32 exchangeable composition slots/dustbin`的O(EK)关系头、集合置换不变loss、模糊分配拒绝和同槽解码；先做真实batch零训练readiness与资源/梯度/泄漏检查，通过后才能冻结三seed训练。C08、结构图和M-TARE继续关闭。

## 2026-09-03：组合锚点失败归因3/3一致；停止全局坐标回归，转向局部组合槽可行性

- 正式只读归因corrective run=`results/gate3_semantics/gate3_20260903_primitive_composition_anchor_failure_attribution_v1r2_seed0`已`COMPLETED`、`error=null`。三个冻结seed精确重现原C07 TP/FP/FN与F1，合计`193,932`次前向；optimizer、C08/C09/C10、graph和M-TARE读取全部为0。
- 八组受控替换证明：deployed安全真连接三个seed均为0；仅以预测anchor距离排序时为`0/0/1`；把坐标替换为构造Teacher anchor后，三个seed都在precision=`1.0`下保留全部`442,936`个可观测真连接，F1=`1.0`且overlap hard-negative FP=`0`。Teacher anchor再使用已学习Gaussian scale时也有`275,709/115,150/150,393`个安全真连接，故Teacher几何与基本尺度并非首要阻塞。
- 预测primitive端点到Teacher anchor的平均误差为`11.155/11.811/11.363 m`，composition head仅降到`10.532/11.263/10.754 m`，改善`5.58%/4.64%/5.36%`。原primitive backbone虽把非学习axis MAE从`11.975 m`降到约`5.04--5.46 m`、surface Chamfer从`17.315 m`降到`7.55--8.80 m`，但绝对精度仍远达不到连接所需的亚米级尺度。
- Teacher自身不是矛盾来源：构造Teacher primitive端点到共享anchor的C07中位残差仅`0.000395 m`、q99=`0.1773 m`，正连接anchor完全共点，最近不同cluster间隔`0.8095 m`。因此失败位于“从局部LiDAR恢复足够精确的端点/组合坐标”，而不是连接标签不一致。
- V1在0帧处因错误绑定不存在的`artifacts/anchors`路径系统FAIL；V1R修正路径后因显式改变cuDNN TF32/determinism而不能逐整数复现旧评估。V1R2显式冻结源评估实际状态（CUDA matmul TF32=false、cuDNN TF32=true、deterministic=false、float32 highest）后3/3精确复现。两次失败只作复现附录，不进入模型科学结论。
- V1R2保留90条task-seed分层记录、八条件指标和PNG/PDF/SVG；28项seal逐项校验通过，seal-list SHA-256=`4825b95e6b0c7c9fa15756e2f7681f8c0b1007351ea75490c5be1c6adb03a2f2`。正式diagnosis=`PREDICTED_COMPOSITION_ANCHOR_COORDINATES_ARE_PRIMARY_BOTTLENECK`，decision=`STOP_GLOBAL_ENDPOINT_ANCHOR_REGRESSION_AND_DESIGN_OBSERVABLE_LOCAL_COMPOSITION_INTERFACE`。
- 人话结论：模型确实学到了比规则基线更好的几何和关系排序，但“让每个端点猜共享连接点的绝对三维位置”太难，误差约十米，无法安全建边。下一版不再增加旧head训练轮数，也不退回手写距离连接。
- `NEXT`：只用fit(C01--C06)+C07 Teacher做零训练“局部组合槽”容量与无损性审计。把双端可观测attachment分解为交换不变的连接簇，检查每个端点唯一归属、簇是否为clique、最大簇数/大小、`8/16/32`槽容量、disconnected-overlap隔离和无损relation重建。审计PASS后才允许设计`endpoint -> composition slot/dustbin`的O(EK)学习接口；继续禁止C08、训练、图和planner。

## 2026-09-03：组合锚点三seed C07正式0/3科学FAIL；转入只读失败归因

- 三个seed均完成冻结的3轮/10,233步，总计`30,699`次optimizer update；fit/C07人口、checkpoint选择、冻结backbone和资源合同全部通过。原训练run只在最终评估首批因浮点连乘左右次序产生`2.98e-8`非对称而系统FAIL，训练资产没有失效。
- 唯一零训练评估corrective把端点证据先组成严格对称外积，再乘已经对称的anchor compatibility；17项测试、49项source seal、三个selected checkpoint及全部冻结输入通过。完整`3×64,644=193,932`行C07推理中，旧误差三个seed均为`2.98e-8`，修正分数全部finite、bit-exact symmetric、最大非对称为0。
- 三seed deployed attachment F1=`0.22550/0.20233/0.21629`，均明显高于同输入非学习基线`0.00638`，所以共享组合锚点包含真实关系信号；但最佳F1点precision只有`0.14599/0.13757/0.13291`，overlap hard-negative FP=`335,927/283,790/436,279`。
- 三个seed在precision不低于`0.98`时均无非空真连接；anchor MAE仅改善`5.58%/4.64%/5.36%`，也都低于10%门。因此C07通过seed=`0/3`，decision=`STOP_COMPOSITION_ANCHOR_BEFORE_C08_AND_GRAPH`。
- 人话结论：模型学会了把部分真连接排到前面，却没有把预测端点拉到同一物理组合位置，也不能可靠排除空间接近但不连通的上下层/并行隧道；这个版本不能安全产生拓扑边。几何基元、时序表示、三seed checkpoint和F1增益保留为正证据/消融，不能外推为完整方法成功。
- corrective run=`results/gate3_semantics/gate3_20260903_primitive_composition_anchor_c07_evaluation_corrective_v1_seed0`，系统PASS、模型科学FAIL、27项seal逐项复核，seal-list SHA-256=`2b9a91c62885148962cd3070bab68487a96776626704e5f0c1cfd5933de3af53`；0 optimizer、0 C08/C09/C10、0 graph、0 M-TARE。
- `NEXT`：只用三个冻结checkpoint和完整C07做零训练失败归因，分别替换Teacher anchor、Teacher proposal、端点证据、scale/compatibility及纯距离排序，并按anchor距离、junction degree、形状、坡度和disconnected-overlap分层。先判定瓶颈是锚点可观测性/回归、尺度校准、端点证据还是关系排序；归因前禁止新训练、C08、图和planner。

## 2026-09-03：组合锚点seed0/1完成；seed2已自动开始

- seed0按冻结合同完成3/3轮，每轮fit/C07精确为`426,552/64,644`行和`3,411/522`个batch，共`10,233`次optimizer step；8个新head张量训练、`2,635,631`个backbone参数冻结，C08/C09/C10、graph和M-TARE读取均为0。
- 三轮C07选择总loss依次为`1.968394/1.950769/1.934105`，anchor NLL为`2.593200/2.541887/2.529543`，relation loss为`0.529499/0.521199/0.516868`。固定选择器选择epoch2；这证明优化稳定且未见C07连续目标继续改善，不等于离散连接资格通过。
- 分项风险不能由总loss掩盖：compatibility项从`0.772008`改善到`0.700828`，但overlap hard-negative项从`0.286990`恶化到`0.332909`（`+16.00%`）。人话解释是共享锚点更容易把真连接拉近，但对“空间接近却物理不连通”的叠置隧道区分可能变差；冻结训练继续，最终必须由overlap FP与precision>=0.98非空门判定。
- `selected.pt`已生成，seed0总耗时`9,741.33 s`；峰值CUDA allocation/GPU进程/主机RSS分别为`7,649,243,136/14,742,978,560/2,694,041,600 bytes`，全部低于冻结资源上限，population、steps、frozen-state和selected-exact合同均PASS。
- `selected.pt`与`epoch_02.pt` SHA-256逐字节相同，均为`af37f87cae32ee475e1dd63a66f30eb0d161afadcd66f210d5f55c45dc97e1a5`；冻结backbone状态前后哈希也相同。
- 总控已自动进入seed1，当前训练进程正常、GPU进程约`13.6 GiB`，尚未完成首轮。正式科学结论仍必须等待三个seed的离散attachment F1、precision>=0.98非空TP和anchor MAE评估；至少2/3 seed通过前继续禁止C08和建图。
- seed1运行期间对正式spec绑定的36项输入证据和18项工具做了独立只读中途复核，共54项、`96,294,491 bytes`，SHA-256为`54/54`一致、零缺失、零漂移；未向正式run写入任何数据。
- seed1 epoch0已完整完成：fit/C07人口=`426,552/64,644`，batch=`3,411/522`，累计`3,411`次optimizer step，全部loss与梯度有限。C07 total/anchor-NLL/relation=`2.022705/2.602968/0.536726`；相对seed0同阶段分别高`2.76%/0.38%/1.36%`，主要差异来自uncertainty calibration高`5.24%`，目前属于正常随机种子差异而非系统失败。
- seed1 epoch0耗时`3,264.93 s`，CUDA allocation/GPU进程峰值=`7,649,055,744/14,638,120,960 bytes`，低于16 GiB门；checkpoint大小`10,899,105 bytes`，SHA-256=`a0da178f2f5d650c95fdc7dc1d73e33531e82551c03414c6255fc9b13e7847dc`。已自动进入epoch1，C08/graph/M-TARE继续为0。
- seed1 epoch1已完整完成，累计`6,822`步；C07 total/anchor-NLL/relation/compatibility由首轮`2.022705/2.602968/0.536726/0.778734`改善到`2.012991/2.599057/0.528747/0.741555`，相对改善`0.48%/0.15%/1.49%/4.77%`。但overlap hard-negative由`0.294719`恶化到`0.315938`（`+7.20%`），与seed0同方向，错误合并风险仍未解除。
- seed1 epoch1耗时`3,263.32 s`，CUDA allocation/GPU进程峰值=`7,649,055,744/14,738,784,256 bytes`，checkpoint SHA-256=`cad6789319f92142eff7914244445a0ee44b0c3f5c2819c8eb4e0f7a193c117f`。末轮已自动开始；当前只证明连续优化，没有任何离散安全门结论。
- seed1已完成3/3轮和`10,233`步，固定选择器保留epoch2。相对epoch0，C07 total/relation/compatibility/uncertainty改善约`0.79%/1.65%/6.71%/1.26%`，anchor NLL仅改善`0.09%`；overlap hard-negative从`0.294719`恶化到`0.329283`（`+11.73%`）。这证明第二个seed也稳定优化，但不证明它能安全排除近邻非连接隧道。
- seed1总耗时`9,794.05 s`；峰值CUDA/GPU进程/host RSS=`7,649,055,744/14,738,784,256/2,694,041,600 bytes`，9项训练合同全真。`selected.pt`与`epoch_02.pt` SHA-256均为`a30ceb79d15f502ebf1b48984e09d90f9ed9ccbb38afe5a1efe7e84354b641a2`，冻结骨干前后哈希一致。总控已自动进入seed2，最终离散门仍未执行。

## 2026-09-03：组合锚点关系头三seed正式训练已启动

- Data Card/spec preflight=`0 errors/0 warnings`，唯一不可覆盖run=`results/gate3_semantics/gate3_20260903_primitive_composition_anchor_three_seed_training_v1_seed0`已进入`RUNNING`；37/37冻结测试通过，seed0 epoch0正在执行真实optimizer updates。
- 正式范围固定为seed0/1/2各3轮、batch128、每seed10,233步、总30,699步；只训练22,278参数，冻结2,635,631参数。fit/C07每轮=`426,552/64,644`行，C08/graph/M-TARE仍关闭。
- 启动后GPU进程约13.63 GiB、host RSS约2.36 GiB，均在16/4 GiB门内。尚无完整epoch或C07科学结果；不得把`RUNNING`解释为模型PASS。
- seed0 epoch0已完整结束：fit/C07=`426,552/64,644`行、`3,411/522` batches，累计3,411 optimizer steps。训练/C07 total loss=`2.072937/1.968394`，C07 anchor NLL=`2.593200`、relation=`0.529499`；训练与未见C07尚无明显分叉，但离散98%精度非空门仍未评价。epoch耗时=`3,244.46 s`，CUDA/GPU进程峰值=`7,648,700,416/14,638,120,960 bytes`，checkpoint SHA-256=`5005a5a76ee00b5e3d2e1853eedf5731d107569b5da41f59f4fed414b16d830b`；已自动进入seed0 epoch1。
- seed0 epoch1已完整结束，累计6,822步；C07 total/anchor-NLL/relation由epoch0的`1.968394/2.593200/0.529499`改善到`1.950769/2.541887/0.521199`（分别`0.90%/1.98%/1.57%`），训练total改善`5.00%`到`1.969389`。uncertainty calibration轻微恶化`0.24%`，但总选择loss仍降低，当前选择器保留epoch1；尚无离散门结论。epoch耗时=`3,243.87 s`，GPU进程峰值=`14,742,978,560 bytes<16 GiB`，checkpoint SHA-256=`e31b5705b8c2eef2901a97a800edc1e9f9a1cab98141bf0d1f040b945149d65d`；已进入seed0 epoch2最后一轮。
- 等待epoch1期间已把论文主稿与真实最终接口同步：旧`O(E^2)`独立pairwise classifier明确降为失败消融；主方法改写为每端点sensor-polar三维共享composition anchor与scale、Gaussian对称兼容度、双端evidence安全分数和traversal-only edge。该编辑只整理现有代码/封存证据，不读取C08或改动正式训练工具。

## 2026-09-03：组合锚点关系头batch-128正式readiness PASS；进入三seed训练冻结

- 唯一正式run=`results/gate3_semantics/gate3_20260903_primitive_composition_anchor_training_readiness_v1_seed0`完成且`error=null`。33/33测试和17项正式检查全PASS；四层loader精确绑定fit/C07=`426,552/64,644`行、`180/30`任务，C07只核人口且模型前向为0。
- 真实fit batch-128前向、反向和无梯度重复验证了`2,635,631`个冻结backbone参数、`22,278`个可训练head参数和8个head张量。8/8梯度有限非零，冻结梯度全部缺失，anchor/scale/compatibility/loss重复误差全部为0。
- 峰值CUDA allocation=`7,659,378,688 bytes`、GPU进程显存=`14,633,926,656 bytes`、host RSS=`2,155,584 KiB`，均低于冻结上限；optimizer steps/checkpoint writes/C08/C09/C10/graph/M-TARE全部为0。
- 13项证据逐项SHA-256复核一致，seal-list SHA-256=`252ee90df3d50c6bfa24d2b06289dc0e00774ed5038d03ee3ba904cc5442afbc`。decision=`ALLOW_COMPOSITION_ANCHOR_HEAD_ONLY_THREE_SEED_TRAINING_CARD`；这只证明训练接口和资源可行，不是C07模型效果。
- NEXT冻结一个三seed正式训练/评价run：每seed只训练22,278参数、固定3轮和10,233步，按C07 mean anchor total loss选checkpoint，再以同一442,936个可观测正连接人口执行F1增益、precision>=0.98非空TP、anchor MAE和冻结backbone门。至少2/3 seed通过前继续禁止C08与建图。

## 2026-09-03：组合锚点训练目标全量物化PASS；下一步只训练22,278参数关系头

- 新sidecar模块把程序构造的每个端点共享锚点按P1b `primitive_index/primitive_mask`顺序转换到当前sensor坐标，输出`float32 [N,32,2,3]`；inactive槽精确为0，文件不包含node/primitive identity、world pose、TNG、未来帧或LiDAR副本。
- V1因绘图直方图错误地拒绝`>=80 m`目标而在第22个C07任务前停止；只完成21/210任务，未训练。真实失败样例最大值`80.3915 m`但相连端点锚点距离仍为0，故判定为system/visualization failure而非Teacher failure。V1R新增overflow记录，不改变任何目标或科学阈值。
- V1R正式完成210/210任务和491,196/491,196序列：fit/C07=`426,552/64,644`，真连接锚点检查=`3,301,484/525,077`，最大连接两端差=`0 m`，最大float32误差=`3.81465e-6 m`，确定性全重推、source sequence、inactive zero和输入前后hash全部精确。
- 输出仅`48,241,720 bytes`，run总目录约58 MiB；fit/C07真实最大锚点距离=`82.2213/80.3915 m`，80 m图外端点=`99/2`并被明确记录而非删除。22项测试、17项正式检查、2,259项seal全PASS；seal-list SHA-256=`2045f4555d7fbe069489cae2baf92f0d72a86765a0f3eb2f0c460996dd1cc6c0`。
- decision=`ALLOW_COMPOSITION_ANCHOR_HEAD_ONLY_THREE_SEED_TRAINING_SPEC`。NEXT冻结head-only三seedData Card/spec与训练/评价器；仍只用C01--C06/C07，至少2/3 seed达到同输入基线增益、高精度非空连接和几何不回归前不读C08、不建图。

## 2026-09-02：可观测关系的同输入非学习C07基线正式冻结

- 唯一正式run=`results/gate3_semantics/gate3_20260902_primitive_relation_observable_nonlearning_c07_v1_seed0`完整重跑10个C07父世界、30个配对几何任务和64,644条五帧序列；17/17测试、全部三类shard tree hash和442,936个可观测正连接人口精确通过，82,141个隐藏正连接从正/负评分人口排除。
- 鲁棒非学习基线的可观测attachment结果为TP=`1,507`、FP=`27,961`、FN=`441,429`、precision=`0.05114`、recall=`0.003402`、F1=`0.006380`。旧未mask F1=`0.005819`保留为标签策略消融，不再作为新模型主比较。
- 几何、primitive和时序方法未变；primitive F1=`0.298311`、target coverage=`0.175306`，surface Chamfer=`17.314694 m`。运行`1,606.27 s`、peak RSS=`628,152 KiB`，45项seal-list SHA-256=`01100d79bcc91182974259e25ef5d548227d783720202e7dd56a552f67fea58e`。
- decision=`FREEZE_OBSERVABLE_NONLEARNING_C07_BASELINE_FOR_P2`。三seed学习模型必须在完全相同的可观测pair人口上把attachment F1至少提高5个百分点，并在precision>=0.98时产生非空真连接。

## 2026-09-02：可观测关系模型正式readiness PASS；允许冻结三seed训练

- 新关系接口保留V2稀疏基数/最小描述长度目标，同时把P1b物理连接监督限制为“两端都有0.25 m射线支持”的pair；隐藏连接只从attachment loss移除，不改写为负例。部署侧新增从五帧学生特征预测的endpoint-evidence概率，不读取Teacher mask。
- 初始`v1` Data Card把validation声明为空，preflight因治理schema拒绝，未创建run、未执行模型。`v1r`只恢复`S01_flat_tree_small_C07`的validation split声明，实际C07读取仍为0；新preflight为0 error/0 warning。
- 唯一正式run=`results/gate3_semantics/gate3_20260902_primitive_relation_observable_readiness_v1r_seed0`已`COMPLETED`、`error=null`，42/42测试与18/18正式检查全PASS。模型共`2,635,631`参数/`197`参数张量，其中只允许`742,149`参数/`64`关系与端点证据张量训练；`1,893,482`参数保持冻结。
- 三个V2 selected checkpoint均精确迁移`133`个已证明的几何/时序状态张量并重置`64`个关系/evidence张量。真实C01五帧样本含11个active primitives、17个可观测端点、8个可观测正连接和3个隐藏正连接；切换隐藏正连接标签的全部loss误差=`0`，稀疏cardinality目标逐位一致。
- 单次真实batch backward中64/64可训练张量梯度有限非零，133个冻结参数张量均无梯度；query permutation误差=`2.2888e-5<=3e-5`，target permutation、endpoint reversal和重复确定性误差均为0。peak CUDA allocation=`141,050,368 bytes`，运行`3.65 s`，14项seal-list SHA-256=`57e3d5f991326262965b57bb942ee51518ea6c927cf67c31702e1dd0ddc27294`。
- decision=`ALLOW_OBSERVABLE_RELATION_THREE_SEED_TRAINING_SPEC`。下一步固定复用每个seed的几何/时序checkpoint，只训练关系/evidence层；训练与C07门通过前仍禁止C08、结构图和M-TARE。

## 2026-09-02：连接Teacher可观测性正式审计PASS；定位为可修正的窗口级隐藏标签

- 唯一正式run `results/gate3_semantics/gate3_20260902_primitive_attachment_teacher_observability_v1_seed0`在5/5单元测试与preflight 0/0后完成。它只读C01--C07的`70`个开发世界、`210`个配对几何任务、`659,940`个源帧、`491,196`条五帧序列和`3,826,561`个无向正连接；optimizer/model inference/C08/C09/C10/graph/M-TARE均为0。10项人口、隔离和诊断检查全PASS，运行`12.13 s`、peak RSS=`528,248 KiB`，17项seal-list SHA-256=`151cdff6b18755a8458172247cf4b1d4e7d317230a7f443b904d1c4c133ccacf`。
- 审计直接验证旧P1b标签规则缺少端点可见性条件：只要两个基元在五帧任意位置分别出现且完整构造图中共享节点，就生成attachment；没有要求被标注的两个连接端点附近存在射线支撑。
- 在继承Teacher LOS margin的`0.25 m`主判据下，fit/C07只有`84.2799%/84.3564%`正连接的两个端点都被观测。fit的`429,720`条只有单端支持、`89,277`条双端均无支持；C07对应`71,461/10,680`条。最大端点缺口p90为`1.575/1.375 m`，不是float或0.025 m采样误差。
- 但物理连接identity在至少一个窗口获得双端支持的fit/C07覆盖为`95.1675%/95.2066%`，同时越过预注册`0.95`可修正门。因此冻结诊断=`WINDOW_LEVEL_HIDDEN_ENDPOINT_SUPERVISION_CORRECTABLE_BY_OBSERVABILITY_MASK`，decision=`MATERIALIZE_OBSERVABLE_LOCAL_ATTACHMENT_EVIDENCE_TEACHER`。这不是宣告连接模型成功，而是证明五帧总体包含足够连接事件、旧窗口级监督却把约15.7%无完整证据正例强行加入loss。
- 下一步只创建小型可观测性sidecar：每个active端点记录`gap<=0.25 m`支持位；attachment loss仅在候选两端均被支持时有效，隐藏正例变成unknown而不是negative。保留现有P1a/P1b和几何checkpoint不复制传感器数据；新关系readiness和三seedC07门通过前仍禁止C08、图和M-TARE。
- 该sidecar现已正式PASS：run=`results/gate3_semantics/gate3_20260902_primitive_attachment_observability_sidecar_v1_seed0`，12项检查全真，精确物化`210`个任务和`491,196`条序列。fit/C07保留可观测正连接=`2,782,487/442,936`，隐藏正连接=`518,997/82,141`明确保持unknown；可观测无向candidate pair=`34,860,975/5,874,759`。数据只占`892,360 bytes`、运行`8.86 s`、peak RSS=`91,540 KiB`，2041项seal-list SHA-256=`52f379fe2f1fdbeca46b8278f9136bb40dc5bcd13217116f4f88e7e75a5060da`。
- 当前进入masked relation readiness：sidecar必须逐序列绑定P1b，端点mask必须随Hungarian slot和endpoint reversal完全同步，attachment loss/metric只能读取双端有效pair；模型还需提供部署时的端点证据概率，不能依赖Teacher mask。上述接口全部通过前不训练。

## 2026-09-02：V2稀疏端点关系三seed正式C07科学FAIL；未进入C08或建图

- 唯一正式run `results/gate3_semantics/gate3_20260901_primitive_relation_sparse_port_three_seed_training_v1r_seed0`已无系统错误完成：三个seed各7轮、总计`561,456`次optimizer step，训练/C07人口、checkpoint、独立host RSS、GPU进程显存和源码完整性全部通过；`RUN_STATE=COMPLETED`，67项证据SHA-256逐项复核一致，seal-list SHA-256=`74a834a1c5a07b42e74a803cb4c008200579ed34153fefec050d77c7bc5a5c82`。
- 冻结C07门结果为`0/3` seed通过。三seed表面Chamfer相对同输入非学习基线改善`56.37%/49.19%/52.60%`，连续几何macro改善`20.55%/38.92%/29.07%`，primitive F1增益均为正，target coverage=`0.873565/0.975904/0.923516`，时序对应accuracy=`0.799571/0.816778/0.874264`；因此五帧LiDAR恢复显式扫掠几何与跨帧身份得到跨seed正证据。
- 失败集中在物理端口关系。attachment F1=`0.072331/0.032406/0.055452`，相对基线绝对增益=`+6.65/+2.66/+4.96`个百分点，只有seed0达到`+5`点门；三个seed在precision不低于`0.98`时均为`TP=0, FP=0, recall=0`，空拒绝不能生成任何拓扑边。人话解释是模型已能识别隧道几何段并跨帧跟踪，却仍不能可靠区分“空间上看起来接近”和“物理上真正相连”。
- 按冻结停止政策，C08 rows read=`0`，C09/C10、离线图、在线图和M-TARE均为0；decision=`STOP_SPARSE_PORT_BEFORE_C08_AND_GRAPH`。不得降低阈值、挑seed、用规则连接或规划器补偿。
- 当前唯一允许工作改为C07-only只读关系失败归因：分别测量Teacher对齐基元、真实基元数约束、端点几何候选域和不确定性排序的precision-recall上界，判断瓶颈属于基元对齐、候选组合、关系表征还是校准。归因前不修改数据、Teacher、模型、门槛或读取C08；若学习关系在oracle条件下仍不能产生非空安全连接，则按计划停止当前主方法并重新评估论文研究问题。
- C07-only归因现已实现并启动唯一正式run `results/gate3_semantics/gate3_20260902_primitive_relation_sparse_port_failure_attribution_v1_seed0`。18项合成/指标/隔离测试PASS，Data Card/spec preflight=`0 errors/0 warnings`；固定读取`3×64,644=193,932`条推理行，比较deployed、Teacher-cardinality和proposal-oracle三种候选，以及independent、best-link-union、best-link-mutual三种只用冻结学习分数的解码。零optimizer、零checkpoint写、零C08/C09/C10/graph/M-TARE；预计1--1.5小时。
- 上述V1在第一批128行、任何完整C07结果前因`scatter_`的`[1,32]` rank源不自动广播到`[128,32]`而系统FAIL；0完整推理、0训练，14项证据seal SHA-256=`b435e60ed20f2d6bbbcf9998436301d6a44e45f9ffb9c0f75598de242f7a96b6`。V1R唯一修正为显式`expand_as(order)`，新增batch128回归后19/19测试与preflight 0/0通过；新不可覆盖run `...failure_attribution_v1r_seed0`已正常越过原阻塞，当前GPU进程约13.6 GiB<16 GiB，科学结果待完成。
- V1R已完成全部`193,932`条C07推理并生成三seed结果，但外层封存器在科学计算完成后构造required-file列表时因未定义`SEEDS`触发NameError，原run保持系统FAIL和23项seal。零推理证据corrective随后20/20测试、preflight 0/0并正式PASS：12项证据检查、25项新seal及全部上游输入hash复核一致，seal-list SHA-256=`cfab8a74cb329b05198503c92e70d68c6230b44023f8ddf2d757b2ed9e37afd7`。
- 正式归因诊断=`RELATION_SCORE_FAILS_EVEN_WITH_PROPOSAL_ORACLE`，decision=`STOP_CURRENT_RELATION_HEAD_AND_REASSESS_METHOD`。deployed/cardinality/proposal-oracle独立关系F1分别为seed0=`0.0725/0.0837/0.2009`、seed1=`0.0325/0.0364/0.1793`、seed2=`0.0555/0.0589/0.1630`；三seed在所有七个诊断条件下的安全通过数均为0。best-link约束、精确tie-safe阈值和risk adjustment均不能产生precision≥0.98的非空真连接；当前关系头不得再训练或进入C08/图。
- 下一步仅允许零训练可观测性复核：检查P1b attachment Teacher是否要求模型判断五帧射线支持之外或被遮挡的连接点，并按端点可见性、连接点射线来源、距离、junction degree和stacked overlap分层。如果大部分失败正例缺乏直接观测支持，重新定义“可观测局部连接证据→真实穿越提交边”的研究接口；若Teacher正例充分可见，则当前pairwise关系表示本身构成方法阻塞并需另立新计划。

## 2026-09-01：V2首个正式run因主机RSS证据缺失主动停止；V1R资源corrective已启动

- 在原V1正式seed0运行不足7分钟、epoch0尚未完成时，源码复核发现训练器字段`process_memory`实际来自`nvidia-smi`，记录的是GPU进程显存；原总控却又把它当作主机RAM证据。显存门存在，但Data Card要求的主机峰值RSS没有独立证明。
- 按资源失败政策主动停止。原run已封存为`FAILED/FAIL_SYSTEM_HOST_RSS_MONITOR_CONTRACT_V1`：`44/44`测试完成，`0`个完成epoch、`0`个checkpoint、C07模型评分`0`、C08/C09/C10/graph/M-TARE=`0`；部分optimizer步数不可精确恢复且全部丢弃。15项seal SHA-256=`dad4df30c712a08adcb23e0e1f737d6170f47423587f36583f9ef9d21f08e5c6`。
- V1R只增加外层Linux`VmHWM`轮询和`RUSAGE_CHILDREN`高水位，并把旧字段明确解释为GPU进程显存；数据、2,629,870参数模型、loss、7轮阶段、seeds、优化器、checkpoint选择、C07/C08科学门全部不变。解析、live process和强制超限终止负控制使正式测试增至`48/48 PASS`。
- V1R Data Card/spec preflight=`0 errors/0 warnings`；唯一run=`results/gate3_semantics/gate3_20260901_primitive_relation_sparse_port_three_seed_training_v1r_seed0`已`RUNNING`，全新seed0 epoch0正在训练。启动时主机RSS约`2.29 GiB`、GPU进程显存约`3.05 GiB`，C08仍关闭。
- seed0 epoch0几何预训练已完整结束：训练=`426,552 rows / 26,736 batches`，C07=`64,644 rows / 522 batches`，累计optimizer steps=`26,736`，全部六类loss与gradient有限；C07 total loss=`0.554646`。epoch耗时=`4,251.64 s`，PyTorch CUDA allocation峰值=`7,882,955,264 bytes`，GPU进程峰值=`16,137,584,640 bytes=15.028 GiB<16 GiB`，独立主机RSS峰值约`2.53 GiB`。`epoch_00.pt`=`24,983,267 bytes`、SHA-256=`6c290448b015b66df41ef99e6e1e0e1dec698cf0cacd36ba22476c8033a26b4e`；已自动进入epoch1稀疏集合/MDL预训练。这些是训练完整性证据，不是最终科学PASS，C08仍未读取。
- seed0 epoch1稀疏集合/MDL预训练已完整结束：训练/C07人口继续精确为`426,552/64,644`行和`26,736/522`个batch，累计optimizer steps=`53,472`，全部loss与gradient有限。C07 primitive-set loss由`0.172645`降到`0.148694`（`13.87%`），surface loss由`0.187028`降到`0.154280`（`17.51%`），total由`0.554646`降到`0.536579`；port-relation loss由`0.708403`暂升到`0.712880`，符合该轮未训练关系头的阶段边界，不能据此声称连接成功。epoch耗时=`4,214.84 s`，CUDA allocation峰值=`8,009,593,344 bytes`，GPU进程峰值=`16,125,001,728 bytes<16 GiB`，主机VmHWM仍约`2.53 GiB`。`epoch_01.pt` SHA-256=`ff250b9e889c9b508e633ead7f5c2a943264352ecf0e95e24786e333a5c862ea`；已自动进入epoch2端点关系预训练，C08仍未读取。
- seed0 epoch2端点关系预训练已完整结束：累计optimizer steps=`80,208`，人口/批次数不变且全部loss/gradient有限。C07 port-relation loss由`0.712880`降到`0.404838`（改善`43.21%`），证明关系目标可被优化；同时primitive-set由`0.148694`恶化到`0.974497`、surface由`0.154280`恶化到`0.400233`、uncertainty由`0.074643`恶化到`1.326323`，暴露专任务阶段的明显灾难性遗忘。因此当前checkpoint不能用于建图，后三轮联合训练必须同时恢复几何并保留关系收益，否则V2按科学门失败。epoch耗时=`4,341.05 s`，GPU进程峰值=`16,047,407,104 bytes<16 GiB`，主机VmHWM仍约`2.53 GiB`。`epoch_02.pt` SHA-256=`2789d133bed156d3a8c46ef26de04cc66e353cb55d5c18e9d35a1e00c1cd06c9`；已自动进入epoch3时序/不确定性预训练，C08仍未读取。
- seed0 epoch3时序/不确定性预训练已完整结束：累计optimizer steps=`106,944`，人口/批次数精确且全部loss/gradient有限。C07 temporal loss由`2.119195`降到`0.595634`（改善`71.89%`），uncertainty由`1.326323`降到`0.001411`（改善`99.89%`）；但port-relation从`0.404838`退化到`0.677038`（恶化`67.24%`），primitive-set=`0.374362`和surface=`0.429515`仍远差于epoch1。这证明各专任务可以单独学习，却会相互覆盖；当前checkpoint仍不可建图。epoch耗时=`4,367.25 s`，GPU进程峰值=`15,755,902,976 bytes<16 GiB`，主机VmHWM约`2.53 GiB`。`epoch_03.pt` SHA-256=`34dee5d1d78dc153ba7f9eb4d2f3a7efc7fbc40bb9677163c7f9c3d67b255343`；已进入epoch4首轮六任务等权联合训练，后三轮能否同时保住几何、关系、时序和不确定性是当前关键证据，C08仍未读取。
- seed0 epoch4首轮六任务联合训练已完整结束：累计optimizer steps=`133,680`，人口/批次精确且全部loss/gradient有限。相对epoch3，C07 primitive-set/surface/port-relation/temporal/total分别改善`62.87%/64.38%/27.71%/30.32%/40.89%`；primitive-set=`0.138990`与surface=`0.152979`已恢复并略优于epoch1最佳，port-relation=`0.489455`保留了大部分专训收益但仍差于epoch2的`0.404838`，ray-free-space由`0.046841`退化到`0.057157`，uncertainty仍低为`0.002311`。联合训练因此明显缓解任务覆盖，但尚未证明离散基元数、attachment F1或safe true-positive门通过。epoch耗时=`4,432.20 s`，GPU进程峰值=`15,611,199,488 bytes<16 GiB`，主机VmHWM约`2.53 GiB`。`epoch_04.pt` SHA-256=`9cce9ed4111157a33f1abbaf2dc283b876036637b95004dda31be2b21680c14f`；已自动进入epoch5第二轮联合训练，C08仍未读取。
- seed0 epoch5第二轮联合训练已完整结束：累计optimizer steps=`160,416`，人口/批次精确且全部loss/gradient有限。相对epoch4，surface/ray-free-space/uncertainty分别改善`0.48%/9.47%/15.18%`，primitive-set仅恶化`0.62%`而基本稳定；但port-relation/temporal/total分别恶化`12.59%/9.01%/7.44%`。因此首轮联合平衡没有单调稳定提升，epoch4仍是当前最低C07 total checkpoint；不得手动按单项挑epoch5。epoch耗时=`4,431.68 s`，GPU进程峰值=`15,795,748,864 bytes<16 GiB`，主机VmHWM约`2.53 GiB`。`epoch_05.pt` SHA-256=`64df7c37e638986bda737b667ad1832ba0749a33d1512f3e8ddeb7865ada3b0b`；已自动进入epoch6最后一轮联合训练，最终仍需离散门判定，C08未读取。
- seed0 epoch6最后一轮及完整七轮训练已结束：累计optimizer steps精确=`187,152`。训练集port-relation仍为`0.282486`，但C07 port-relation由`0.551087`暴涨到`1.603483`（恶化`190.97%`）、C07 total由`0.224889`升到`0.408044`（恶化`81.44%`），构成后期关系过拟合/泛化崩坏证据；不是资源或数值失败。冻结选择正确保留epoch4：`selected.pt`与`epoch_04.pt`逐字节同SHA-256=`9cce9ed4111157a33f1abbaf2dc283b876036637b95004dda31be2b21680c14f`，而`epoch_06.pt` SHA-256=`b2ddda7ab9e6cbd9a45cd37b7d1627d93661183ac4c95b4b5538dcf65567cb87`。seed0全程峰值CUDA allocation=`8,009,593,344 bytes`、GPU进程=`16,137,584,640 bytes<16 GiB`、独立host RSS=`2,716,803,072 bytes`，资源PASS。C08及后续仍为0；seed1已从全新初始化自动开始。
- seed1 epoch0几何预训练已完整结束：训练/C07=`426,552/64,644`行、累计`26,736`步，全部loss/gradient有限。C07 total=`0.576785`，比seed0同阶段高`3.99%`；surface=`0.180455`和port-relation=`0.692983`分别比seed0好`3.51%/2.18%`，temporal/ray较差但尚未进入专训。该差异属于正常初始化波动，没有异常掉队，也不能据此挑seed。epoch耗时=`4,255.59 s`，GPU进程峰值=`16,137,584,640 bytes<16 GiB`；`epoch_00.pt` SHA-256=`c5714581bbcaaec06c11382d7ba58c4a6d74a7b7adea2188cebaee4215fb19db`。已自动进入seed1 epoch1稀疏集合/MDL训练，C08仍未读取。
- seed1 epoch1稀疏集合/MDL阶段已完整结束：累计`53,472`步，训练/C07人口仍精确为`426,552/64,644`行，全部loss与gradient有限。C07 surface由`0.180455`降到`0.123593`（改善`31.51%`），total由`0.576785`降到`0.558409`（改善`3.19%`），uncertainty由`0.145471`降到`0.079976`；但primitive-set由`0.176960`升到`0.182963`（恶化`3.39%`），port-relation基本不变为`0.693380`。人话解释是这一阶段明显改善了表面解释和置信度，但还不能证明冗余基元已经减少；必须等离散检测门统一判断。epoch耗时=`4,217.94 s`，GPU进程峰值=`16,087,252,992 bytes<16 GiB`；`epoch_01.pt` SHA-256=`8267d59f891bac1d75207315c6d518afa69a3c8d4ff5e99e35c5da0c3d66d59d`。已自动进入seed1 epoch2端点关系预训练，C08仍未读取。
- seed1 epoch2端点关系阶段已完整结束：累计`80,208`步，训练/C07人口精确且全部loss/gradient有限。C07 port-relation由`0.693380`降到`0.336576`（改善`51.46%`，也优于seed0同阶段`0.404838`），说明第二个随机初始化同样可以学习端点物理连接；但primitive-set/surface/uncertainty恶化到`0.731278/0.509630/0.324431`，再次出现“学连接时忘几何”的任务覆盖。当前checkpoint不可建图，后续联合阶段必须恢复几何且保住关系。epoch耗时=`4,330.07 s`，GPU进程峰值=`15,875,440,640 bytes<16 GiB`；`epoch_02.pt` SHA-256=`dfd4d8fdc27001e3d41b8db029227036a0ae58533c8515093f52e5755ce69d39`。已自动进入seed1 epoch3时序/不确定性预训练，C08仍未读取。
- seed1 epoch3时序/不确定性阶段已完整结束：累计`106,944`步，人口/批次精确且全部loss/gradient有限。C07 temporal由`2.115281`降到`0.452121`（改善`78.63%`），uncertainty由`0.324431`降到`0.004602`（改善`98.58%`）；primitive-set/surface部分恢复到`0.374387/0.426948`，但port-relation由`0.336576`退化到`0.854829`（恶化`153.98%`）。这与seed0同方向，确认各专任务均可学但共享表示互相覆盖。epoch耗时=`4,365.14 s`，GPU进程峰值=`15,963,521,024 bytes<16 GiB`；`epoch_03.pt` SHA-256=`19e925a14fe48289e4367b8ce65f1d896a8bba787fc58d99b6e764f7aad83840`。已进入seed1 epoch4首轮六任务联合训练；它能否同时恢复几何和连接是当前关键证据，C08仍未读取。
- seed1 epoch4首轮联合训练已完整结束：累计`133,680`步。C07 primitive-set/surface/ray恢复到`0.171751/0.156818/0.055121`，temporal=`0.411639`，total由epoch3的`0.370005`改善到`0.330205`；但port-relation进一步恶化到`1.183414`，比epoch2专训最佳`0.336576`高`251.61%`，也远差于seed0首轮联合的`0.489455`。训练集port-relation仅`0.292223`，故这是未见C07上的关系泛化崩坏，不是关系目标没优化或系统失败。epoch耗时=`4,423.04 s`，GPU进程峰值=`15,613,296,640 bytes<16 GiB`；`epoch_04.pt` SHA-256=`a92a8b66cadd009639bafb97712a93710b3f38aec333cdd73b0a1b611528d351`。已自动进入seed1 epoch5第二轮联合训练；剩余两轮若不能恢复关系，该seed很可能不能通过最终离散门，C08仍未读取。
- seed1 epoch5第二轮联合训练已完整结束：累计`160,416`步。C07 port-relation由`1.183414`回落到`0.951534`（恢复`19.59%`），primitive-set/surface保持并小幅改善为`0.167342/0.150039`，total改善`10.44%`到`0.295728`；但关系仍比epoch2专训最佳高`182.71%`，尚未恢复到可泛化状态。训练集port-relation=`0.306476`，训练/未见域差距仍大。epoch耗时=`4,420.49 s`，GPU进程峰值=`15,795,748,864 bytes<16 GiB`；`epoch_05.pt` SHA-256=`aecc8a1afa10140883363d2b97fc9a71177a8770e20cb22ca7e198a66bdb0c1c`。已进入seed1 epoch6最后一轮联合训练；C08仍未读取。
- seed1 epoch6及完整七轮训练已结束：累计optimizer steps精确=`187,152`。最后一轮C07 port-relation由`0.951534`恢复到`0.573549`（改善`39.72%`），primitive-set/ray/uncertainty也改善到`0.144844/0.049330/0.001086`；surface/temporal则从`0.150039/0.444534`回退到`0.175111/0.577640`。C07 total降到全程最低`0.253593`，所以冻结选择器按规则选中epoch6；`selected.pt`与`epoch_06.pt`同SHA-256=`0ee826131ff5781f5598da437745d2d9d0e367b0acf773caecdac374c90b3168`。关系比专训最佳仍高`70.41%`，说明最终是多任务折中而非完全稳定恢复；离散科学门尚未执行。seed1峰值CUDA allocation=`8,009,683,968 bytes`、GPU进程=`16,137,584,640 bytes<16 GiB`、独立host RSS=`2,716,803,072 bytes`，资源PASS。seed2已从全新初始化自动开始，C08仍未读取。
- seed2 epoch0几何预训练已完整结束：训练/C07人口精确为`426,552/64,644`行和`26,736/522`个batch，累计optimizer steps=`26,736`，全部loss和gradient有限。C07 surface=`0.145497`，比seed0/seed1同阶段低`22.21%/19.37%`；但primitive-set=`0.213505`，比两者高`23.67%/20.65%`，total=`0.547025`则分别低`1.37%/5.16%`。这说明第三个初始化的底层表面恢复健康但基元参数仍有随机差异，不构成连接成功；port-relation仍为未专训的`0.704926`。epoch耗时=`4,244.72 s`，CUDA allocation峰值=`8,008,509,952 bytes`；`epoch_00.pt`大小`24,983,267 bytes`、SHA-256=`8bf4145ed0416b6dcd9ad2b4a4d2b8314d4b1c64c0f0a0c6548cebf75767b372`。已自动进入seed2 epoch1稀疏集合/MDL阶段，C08/C09/C10、graph和M-TARE仍未读取。
- seed2 epoch1稀疏集合/MDL阶段已完整结束：累计optimizer steps=`53,472`，人口/批次继续逐项精确，全部loss和gradient有限。相对epoch0，C07 primitive-set由`0.213505`降到`0.151808`（改善`28.90%`），surface由`0.145497`降到`0.142166`（改善`2.29%`），ray-free-space改善`29.79%`，total改善`1.91%`。稀疏约束因此没有用破坏表面换取集合变小，而是同时改善基元集合与几何解释。其C07 total=`0.536580`与seed0同阶段只差`0.00025%`，并比seed1低`3.91%`，属于跨初始化一致的健康结果；仍不能证明连接成功。epoch耗时=`4,211.24 s`，CUDA allocation峰值=`8,012,101,120 bytes`、GPU进程峰值=`16,104,030,208 bytes<16 GiB`；`epoch_01.pt`大小`24,983,267 bytes`、SHA-256=`00a1395ccb0bbb2fed40a8b0b0ca973c85cffe7cc1119f0d2d199bd93186bddf`。已自动进入seed2 epoch2端点关系专训，C08及后续仍关闭。
- seed2 epoch2端点关系专训已完整结束：累计optimizer steps=`80,208`，训练/C07人口仍精确为`426,552/64,644`行和`26,736/522`个batch，全部loss与gradient有限。C07 port-relation由`0.697169`降到`0.227114`（改善`67.42%`），三seed同阶段均能明显优化物理连接目标；但primitive-set/surface/uncertainty同时恶化到`0.950817/0.266606/0.214647`，total恶化`19.73%`。人话解释是第三个初始化也能单独学会连接线索，但专门强化连接会覆盖此前学到的形状，因此当前checkpoint仍不能建图；epoch3及后三轮联合训练必须恢复几何并保留连接收益。epoch耗时=`4,329.97 s`，CUDA allocation峰值=`8,012,101,120 bytes`、GPU进程峰值=`15,883,829,248 bytes<16 GiB`；`epoch_02.pt`大小`30,715,315 bytes`、SHA-256=`9ae9037beafc3a0426c9e4c133b68053a93e2006e51ab5c19c0577a84864b29a`。三seed总训练已完成`454,512/561,456=80.95%`步，seed2 epoch3时序/不确定性阶段正在运行；C08/C09/C10、graph和M-TARE仍关闭。
- seed2 epoch3时序/不确定性专训已完整结束：累计optimizer steps=`106,944`，人口/批次精确且全部loss与gradient有限。C07 temporal由`2.168090`降到`0.444712`（改善`79.49%`），uncertainty由`0.214647`降到`0.003221`（改善`98.50%`），并且时序结果略优于seed0/seed1同阶段的`0.595634/0.452121`；但port-relation由`0.227114`恶化到`0.977031`（`+330.19%`），surface/ray也恶化`34.80%/268.37%`。人话解释是模型确实能跨五帧认出同一结构并学会不确定性，但会覆盖刚学到的物理连接与部分几何；三个种子都呈现“单项可学、共享表示互相覆盖”的同一机制。epoch耗时=`4,359.75 s`，GPU进程峰值=`15,854,469,120 bytes<16 GiB`；`epoch_03.pt`大小`31,796,511 bytes`、SHA-256=`0feb0499dbfa7eca2b4e4de6acaef457a1829fdc15e4a27879b994b0003c4b2f`。三seed总进度为`481,248/561,456=85.71%`，已自动进入seed2 epoch4首轮六任务联合训练；它能否同时恢复几何和连接是当前唯一关键证据，C08及后续仍关闭。
- seed2 epoch4首轮六任务联合训练已完整结束：累计optimizer steps=`133,680`，人口/批次精确且全部loss与gradient有限。相对epoch3，C07 primitive-set/surface/ray/temporal分别改善`76.89%/53.24%/63.43%/13.41%`，几何和时序已明显恢复；但port-relation由`0.977031`进一步恶化到`1.395239`（`+42.80%`），而训练集仅为`0.275771`，C07/fit差距=`5.06x`。人话解释是联合训练能把形状和跨帧身份拉回，却没有让物理连接迁移到未见拓扑；当前不是训练不收敛，而是连接泛化崩坏，尚未实现四种能力共存。epoch耗时=`4,414.37 s`，GPU进程峰值=`15,619,588,096 bytes<16 GiB`；`epoch_04.pt`大小`31,796,511 bytes`、SHA-256=`fb84118298b3733d49590eb2f148be283e6f80bc624b85f4b7361b58c5a41f3f`。三seed总进度=`507,984/561,456=90.48%`，已进入seed2 epoch5第二轮联合训练；剩余两轮是冻结日程内恢复连接泛化的最后机会，C08及后续仍关闭。
- seed2 epoch5第二轮联合训练已完整结束：累计optimizer steps=`160,416`，人口/批次精确且全部loss与gradient有限。相对epoch4，C07 port-relation由`1.395239`回落到`0.929426`（恢复`33.39%`），primitive-set/surface/uncertainty也改善`14.43%/1.55%/38.05%`，total改善`20.90%`并成为seed2当前最低；但port仍为训练集`0.296667`的`3.13x`、比epoch2专训最低值高`309.23%`，ray恶化`80.58%`且temporal轻微回退`3.47%`。人话解释是连接泛化开始恢复，却仍没有与几何、自由空间、时序稳定共存；这不是彻底崩坏，也尚不是可建图结果。epoch耗时=`4,416.13 s`，GPU进程峰值=`15,795,748,864 bytes<16 GiB`；`epoch_05.pt`大小`31,796,511 bytes`、SHA-256=`3687bae95655af9e213950ba238b22eee23a8ef7fb8ca06a483acdd810bc76cf`。三seed总进度=`534,720/561,456=95.24%`，已自动进入seed2 epoch6最后一轮联合训练；完成后立即执行冻结C07离散门，C08及后续仍关闭。
- seed2 epoch6最后一轮及三seed完整训练已结束：seed2累计optimizer steps=`187,152`，三seed总计精确=`561,456`；人口/批次逐项精确且全部loss/gradient有限。相对epoch5，C07 primitive-set/surface/ray/uncertainty改善`4.13%/1.88%/56.28%/0.95%`，但port-relation由`0.929426`恶化到`1.200159`（`+29.13%`）、temporal恶化`2.23%`、total恶化`13.66%`。因此固定选择器正确保留epoch5；`selected.pt`与`epoch_05.pt`同SHA-256=`3687bae95655af9e213950ba238b22eee23a8ef7fb8ca06a483acdd810bc76cf`，`epoch_06.pt` SHA-256=`1ca3d065112e33341f9cae5328d9d18dd92399ea91ed5ad238f58e964cdf0ca0`。seed2峰值host RSS=`2,730,606,592 bytes<16 GiB`且监控未终止；三seed训练资源全部PASS。人话解释是继续联合训练没有稳定消除连接泛化摆动，不能再靠加epoch；总控已自动启动三seed冻结C07离散评估，C08及后续仍关闭。
- 冻结C07离散评估已完成seed0：选中epoch4在`64,644`行上通过表面、连续几何、attachment F1、primitive F1和coverage五项门，但安全非空提交失败，因此该seed整体FAIL。具体为surface改善=`56.37%`、geometry macro改善=`20.55%`、attachment F1=`0.072331`（比基线`0.005819`高`6.65`个百分点）、primitive F1=`0.599041`、coverage=`0.873565`、temporal accuracy=`0.799571`。风险调整attachment在precision=`1.0`时true positive=`0`、false positive=`0`、recall=`0`；空提交不能冒充安全成功。人话解释是V2终于让连接F1超过预注册增益门，证明它学到了更有用的连接排序；但没有任何真连接能在98%精度要求下可靠落边，所以仍不可在线建图。结果SHA-256=`4c62f04f60eaadef411f752c1b4e2a0bb804375e884b2eb470192c9b4d24be8e`；seed1评估正在运行，C08仍未实例化。
- 等待训练期间补充原始来源边界：Hydra/S-Graphs证明解析自由空间/墙面层级图已有成熟方案，TopoNet/DAGMapper证明联合几何—拓扑学习也不是单点首次。相关工作矩阵、正文和BibTeX已同步；当前主张进一步限定为“因果三维隧道基元及物理关系→拒绝合并→真实穿越确认edge→探索收益”。解析连通只能作基线/安全核验，不能在P2失败后替代学习关系补门；本次文献修订没有改变运行模型、数据、Teacher或阈值，启动源码哈希复核无漂移。
- `NEXT`：自然完成V1R seed0--2七轮训练和冻结C07门；不得复用原V1部分更新、修改阈值或提前读取C08。

## 2026-08-31：V2稀疏端点关系三seed正式训练已启动；seed0 epoch0运行中

- 独立Data Card和run spec已冻结，preflight=`0 errors / 0 warnings`。正式人口为C01--C06的`60`个父拓扑、`180`个配对几何任务、每轮`426,552`个五帧序列；C07为`10`个父拓扑、`30`个任务、`64,644`个选择序列；C08保持条件关闭。
- 训练合同为seed `0/1/2`各7轮：几何、稀疏集合/最小描述长度、端点关系、时序+不确定性、3轮等权联合；每seed=`187,152`步，总计=`561,456`步。V2参数=`2,629,870`，batch=`16`，AdamW `3e-4`，不改变V1数据和同输入基线。
- 正式评估新增不可空安全门：每个通过seed除原五项几何/关系/覆盖门外，还必须以`p(attachment)*(1-uncertainty)`在precision `>=0.98`时至少命中一个真实attachment；零提交不能冒充100% precision。
- 唯一run=`results/gate3_semantics/gate3_20260831_primitive_relation_sparse_port_three_seed_training_v1_seed0`已从`CREATED_NOT_EXECUTED`进入`RUNNING`。冻结测试=`44/44 PASS`，seed0 epoch0已开始，初始GPU进程占用约`3.1 GiB`；C08/C09/C10、graph和M-TARE读取均为0。
- `NEXT`：持续完成seed0--2并按冻结C07门判定。只有至少2/3 seed完整通过才实例化C08 loader；不提前建图、不改阈值、不挑seed。

## 2026-08-31：稀疏端点关系V2零训练readiness正式PASS；允许冻结三seed训练

- 独立V2保留V1的五帧LiDAR、相对里程计、32槽扫掠几何和六类loss，只在`primitive_set_parameters`内部增加校准BCE、精确cardinality与冗余描述长度，并以端点位置/外向切线/截面/形状/描述子关系Transformer替代V1简单全配对MLP；新增attachment/overlap uncertainty，不增加事件类别或手写连接阈值。
- 正式Data Card只读取C04同一源序列的三种配对几何行：`3` rows、`15` causal frames、`5` unique primitives、`24` directed attachment entries、`18` disconnected-overlap entries和`12` temporal dustbins；C07/C08/C09/C10、graph、M-TARE读取均为0。
- run=`results/gate3_semantics/gate3_20260831_primitive_relation_sparse_port_readiness_v1_seed0`已`COMPLETED`、`error=null`、42/42测试PASS。模型=`2,629,870`参数/`193`参数张量，真实batch六类loss有限，`193/193`梯度全部有限非零。
- 稀疏梯度合同PASS：3/32真槽的matched gradient最大值=`-0.001546`，29个冗余槽gradient最小值=`+0.008870`；梯度下降因此同时抬高真槽并压低多余槽，不再复现V1“几乎32槽全开”的目标方向。
- query permutation error=`1.907e-6`、target permutation和endpoint reversal loss error=`0`、端点相对几何yaw rotation error=`2.384e-7`、重复误差=`0`；attachment/overlap严格对称，关系不确定性均在`[0,1]`。
- 峰值CUDA allocation=`391,308,288 bytes`，运行耗时=`3.17 s`，结果=`264 KiB`；0 optimizer/checkpoint/selection/graph/test。13项seal逐项复算有效，seal-list SHA-256=`81c017d4943792e93a8bd2f18cb3ccaa0bf96ff68deac7f6fd901622bf8ff2af`。
- `NEXT`：冻结并执行V2三seed七轮训练：epoch0几何、epoch1稀疏集合/MDL、epoch2端点关系、epoch3时序+不确定性、epoch4--6六类等权联合。C01--C06每轮`426,552`行/`26,736`步，C07每轮`64,644`行/`522`批；C07最终仍按原五项门和2/3 seeds判断，C08保持关闭。

## 2026-08-31：V1R2完整归因PASS；V1关系架构被否决，进入稀疏端点关系readiness

- 唯一不可覆盖run=`results/gate3_semantics/gate3_20260831_primitive_relation_v1_failure_attribution_v1r2_seed0`已`COMPLETED`、`error=null`。它对三个冻结checkpoint各做两遍TF32全关闭C07重评，再各做一遍actual-vs-proposal-oracle归因，共`581,796`次推理、`0` optimizer step；C08/C09/C10、graph和M-TARE读取/执行均为0。
- 修正后的C07科学门仍只有seed1 PASS，seed0/seed2 FAIL，最终`1/3<2/3`。TF32纠正只带来万分量级数值变化，没有改变任何seed的PASS/FAIL，故旧结论方向成立，但以后以V1R2精确数字为准。
- 三seed几乎都把32个槽全部激活：预测基元=`2,066,261/2,066,463/2,067,973`，Teacher目标均为`544,414`；端点pair空间相对Teacher精确膨胀=`14.0369/14.0393/14.0595x`。
- 只用Teacher匹配mask移除错误候选、完全不修改冻结关系logit后，attachment F1由`0.02927/0.08101/0.04083`提高到`0.20493/0.25674/0.23265`。这证明候选过量是主要故障之一，也证明现有LiDAR、时序、几何编码和部分关系排序可复用。
- 但三seed在precision `>=0.98`时，proposal oracle下attachment真实提交仍全部为0；`oracle_gate_seeds=0/3`。因此仅修existence/cardinality不足，正式诊断=`RELATION_HEAD_FAILS_EVEN_WITH_PROPOSAL_ORACLE`，决策=`REQUIRE_SPARSE_PORT_RELATION_ARCHITECTURE_READINESS`。
- 人话结论：模型能恢复隧道形状，也含有一定“哪两段可能相连”的信息；但它先生成太多重复段，再用缺少端点方向/物理相对变换上下文的全配对关系头打分，无法把真连接以足够高的可靠度排到最前。当前V1不能建图，下一版必须同时稀疏化基元集合并增强端点关系表示，不能只调阈值。
- 运行耗时=`3,924.63 s`，峰值主机RSS=`2,248,824 KiB`、CUDA allocation=`7,618,206,208 bytes`，结果目录约`1 MiB`；32项seal全部复算一致，seal-list SHA-256=`ca2600041eb72b70d5ff0118a900526297cbc867b9ab83396e1c2a12be115b93`。失败归因图保存在`previews/primitive_relation_v1_failure_attribution.{png,pdf,svg}`并进入论文失败分析证据池。
- `NEXT`：实现并审计一个零训练的sparse-port relation architecture readiness。保留现有五帧LiDAR/里程计接口和扫掠几何输出，在同一`primitive_set_parameters`损失族内加入基数/最小描述长度约束，以端点切线、相对位移、截面、描述子和上下文关系Transformer替代简单全配对MLP，并新增关系不确定性；先通过解析关系、置换/端点反转/旋转、真实batch有限梯度和资源合同，禁止提前训练或读取C08。

## 2026-08-31：归因V1R1揭示旧评估器cuDNN TF32漂移；C07-only V1R2纠正运行中

- 归因V1的Data Card因audit没有训练集而被通用validator拒绝，0 run/0数据读取；V1R只增加`NONE_FROZEN_CHECKPOINT_AUDIT_ONLY`占位后preflight 0/0并执行唯一run。
- V1R在seed0完整C07前向后按“formal existence/attachment/overlap必须逐项复现”合同主动FAIL，运行`447.94 s`、峰值RSS=`2,242,756 KiB`，0 optimizer/C08/C09/C10/graph/M-TARE；seal-list SHA-256=`f04bccc0308729a773f8f59ed51cf9d8e1b4bdd4935ead71cf08533e51db0375`。
- 代码与环境归因证明原正式训练器明确关闭matmul/cuDNN TF32并设最高FP32精度，但独立正式评估器漏掉该设置；当前Torch默认`matmul TF32=False`、`cuDNN TF32=True`。归因器按声明关闭两者后无法复现旧formal计数，因此旧V1精确C07数字违反声明的评估数值合同。
- 该问题属于system/evaluator配置，不是数据、Teacher、checkpoint或几何表示失败。否决继续用cuDNN TF32-on归因，因为它会把已知不合规设置写进后续论文证据；也否决重新训练，因为checkpoint训练设置本身正确且成本约21小时。
- 唯一V1R2只用相同三个selected checkpoint和完整C07：先在TF32全关闭下做`387,864`次两遍评估前向，再做`193,932`次actual-vs-proposal-oracle归因，共`581,796`次推理；0 optimizer、0 C08、0图/M-TARE。测试=`11/11 PASS`、preflight=`0 errors/0 warnings`，唯一run=`gate3_20260831_primitive_relation_v1_failure_attribution_v1r2_seed0`正在执行。
- `NEXT`：等待V1R2先给出正确C07三seed门，再依据同数值设置下的oracle结果选择稀疏existence/set decoder或新的sparse-port relation architecture；不得提前训练。

## 2026-08-31：P2三种子C07正式科学FAIL；当前V1停止于C08与建图之前

- 唯一不可覆盖run=`results/gate3_semantics/gate3_20260830_primitive_relation_three_seed_training_v1_seed0`已`COMPLETED`，`error=null`。三seed各完成6轮、每轮精确`426,552`个fit序列与`64,644`个C07序列，总optimizer steps=`481,248`；正式运行耗时=`78,035.36 s`，C08/C09/C10、graph、M-TARE读取/执行均为0。
- 冻结C07门只有seed1 PASS，seed0/seed2 FAIL，passing seeds=`1/3<2/3`，故正式决定=`STOP_BEFORE_C08_AND_GRAPH`。C08的`73,182`个迁移序列保持未读，不能用迁移域选择模型或补救当前结论。
- 三seed都证明显式3D几何比同输入非学习拟合器更完整：sampled-surface Chamfer相对baseline改善=`43.17%/50.56%/53.80%`，连续几何macro改善=`42.78%/45.54%/44.81%`，primitive F1均约`0.4167`且比baseline增加约`11.84`个百分点，target coverage=`99.92%/99.92%/99.98%`。
- 失败集中在端口连接关系：attachment F1=`0.02922/0.08097/0.04084`，相对baseline的增益=`2.34/7.52/3.50`个百分点；只有seed1超过预注册`5`个百分点门。三seed在precision `>=0.98`的安全选择下均没有可提交的真实连接。
- 机制证据不是“几何基元表示不了地下结构”，而是32槽模型过量激活：每个seed约预测`2.066--2.068M`个候选，对应`544,414`个真实目标，primitive precision仅约`26.3%`。多余候选进入端口两两组合后放大为大量假attachment和overlap，使关系头无法同时取得可用recall与安全precision。
- 当前结论边界：支持“因果LiDAR可学习显式扫掠地下几何与跨帧对应”，否决“当前V1关系输出可安全形成在线结构图”。不允许挑seed、放宽阈值、读取C08、增加规则或用规划器补偿。
- 正式对比图保存在run的`previews/primitive_relation_comparison.{png,pdf,svg}`，证据清单SHA-256=`81f43937e813f11b1b8cf337edd05f842ba83a189cf94ed54feb5260b7fe2aa7`，metrics summary SHA-256=`f9ef0f0d3263cc954d06ce66931b9787f9e2d151dd8a67e087fbd3087489007f`。该结果作为论文几何可学性证据与关系失败消融永久保留。
- `NEXT`：只读归因当前密封C07输出，把误差分解为“候选存在性/槽位冗余”“oracle真基元条件下的attachment能力”“候选数量导致的pair组合爆炸”“三种几何/拓扑难度分层”。归因不训练、不推理、不读取C08；完成后才冻结证据支持的单一模型修订Data Card/spec。

## 2026-08-30：同输入非学习基元基线正式冻结；三seed训练成为唯一下一步

- 正式基线只读取C07的10个独立父世界、30个配对几何任务和64,644个五帧序列；输入与学习模型相同，只有五帧16×720 range/valid和相对里程计，C08/C09/C10、模型、图和M-TARE读取均为0。
- 30/30任务全部完成，输入tree SHA逐任务一致，RUN_STATE=`COMPLETED`、`error=null`。运行耗时=`1,634.09 s`，峰值RSS=`627,476 KiB`，45项证据seal SHA-256=`605df99e2c542642bdc45d9530a10c89df1512cde95651c97f1a822688a637f7`。
- 非学习方法具有很高的候选纯度但严重漏检：基元precision=`0.999906`、recall=`0.175306`、F1=`0.298311`；目标覆盖率仅`17.53%`。它能拟合局部明显表面，却不能恢复完整组合结构。
- 关系理解几乎失效：attachment F1=`0.005819`，disconnected-overlap F1=`0`；跨帧对应accuracy=`0.155550`。几何surface Chamfer=`17.3137 m`，width/height MAE=`3.4845/2.1689 m`。这些数字冻结为学习模型的同输入C07下界，而不是主方法结果。
- 历史工作按论文价值继续保留：M-TARE、Cano-like出口模型、非学习基元拟合和GT oracle进入正式对照；槽位、时序、关系、监督和不确定性进入消融；具有科学含义的失败进入失败分析；纯脚本/环境故障只进复现附录。
- 当前主方法论文骨架已从旧exit-only稿件中独立建立为`docs/PRIMITIVE_RELATION_STRUCTURAL_GRAPH_MANUSCRIPT_DRAFT_V1.md`，完整覆盖construction supervisor、显式扫掠基元、端口/重叠/时序关系、执行验证图、同输入基线、单/多机器人实验与消融；未完成结果全部保留`AUTO`证据占位符，禁止手填。对应完成矩阵为`docs/PRIMITIVE_RELATION_PAPER_COMPLETION_MATRIX_V1.md`。
- 投稿新颖性已按2026-08-31原始来源复核重写为`docs/PRIMITIVE_RELATION_NOVELTY_MATRIX_V1.md`。ArcPro已覆盖程序合成点云到建筑程序，StructureNet已覆盖部件几何+关系学习，Cano/PRISM/M-TARE分别覆盖地下出口拓扑、学习地点图和多机器人探索；因此当前主张严格收窄为“部分可见因果LiDAR的扫掠隧道基元关系 → 持久结构图 → traversal-only edge → 不改局部栈的探索收益”。尚未发现六项机制全部同构的工作，但必须由P2/P3/闭环证据成立，不能只靠文字差异。
- 论文图索引已增加active-method override：既有`gse_method_overview`明确降为旧事件/出口基线图且保持不可删除；当前主论文登记`primitive_relation_method_overview`、数据/Teacher、三seed/C07、因果图、单/多机器人和消融失败七类新bundle。每类仍要求PNG+PDF/SVG+机器源+确定性publisher+run/seal，未完成结果不得生成性能图。
- `NEXT`：生成并preflight三seed训练Data Card/spec；C01--C06精确426,552个训练序列、C07精确64,644个选择序列，只有C07门通过才允许一次读取C08的73,182个迁移序列。训练期间C09/C10、graph与M-TARE保持关闭。
- 三seed训练Data Card/spec现已冻结，preflight=`0 errors/0 warnings`，唯一run=`gate3_20260830_primitive_relation_three_seed_training_v1_seed0`已创建并进入`RUNNING`。冻结27项测试=`27/27 PASS`，seed0第1轮正在执行；当前约3.76 GiB GPU，尚无epoch/C07结果，不提前作性能判断。
- seed0 epoch0已完整结束：训练=`426,552 rows / 26,736 batches`、C07=`64,644 rows / 522 batches`、累计optimizer steps=`26,736`，精确符合冻结计数；六类训练/C07 loss及gradient norm全部finite，checkpoint=`epoch_00.pt`已写出。epoch用时=`4,244.52 s`，峰值CUDA allocation=`7,848,031,744 bytes`、峰值进程GPU memory=`16,104,030,208 bytes=14.998 GiB<16 GiB`。当前自动进入seed0 epoch1 relation stage；尚未读取C08。
- seed0 epoch1关系预训练也已完整结束：累计optimizer steps=`53,472`，训练/C07仍分别为`426,552/64,644`行且全部finite，checkpoint=`epoch_01.pt`已写出。C07 port-relation loss由`0.708257`降至`0.524387`（下降`25.96%`），总loss由`0.627985`降至`0.602372`；几何项在关系单任务阶段暂时回升，须由后续联合阶段恢复，当前不能据此宣称科学PASS。epoch用时=`4,216.08 s`，峰值进程GPU memory=`15,980,298,240 bytes=14.883 GiB<16 GiB`。当前自动进入seed0 epoch2 temporal/uncertainty stage，C08仍未读取。
- seed0 epoch2时序/不确定性预训练完整结束：累计optimizer steps=`80,208`，精确`426,552`训练行和`64,644/522`个C07行/batch，checkpoint=`epoch_02.pt`已写出，全部loss和gradient finite。C07 temporal-equivariance loss由epoch1的`2.099932`降至`0.454086`（下降`78.38%`），uncertainty loss由`0.177309`降至`0.002560`（下降`98.56%`），总loss由`0.602372`降至`0.395260`。关系、表面与ray项因当前单任务阶段回退，不能据此判模型整体PASS；后续epoch3--5必须在联合目标下恢复。epoch用时=`4,206.29 s`，峰值进程GPU memory=`16,036,921,344 bytes=14.936 GiB<16 GiB`。当前自动进入seed0 epoch3首轮联合训练，C08仍未读取。
- seed0 epoch3首轮联合训练完整结束：累计optimizer steps=`106,944`，训练/C07仍精确为`426,552/64,644`行和`26,736/522`个batch，checkpoint=`epoch_03.pt`已写出，全部训练与选择loss、gradient norm均finite。联合目标使C07总loss由epoch2的`0.395260`降至`0.300798`；surface/ray/port-relation分别由`0.369060/0.293585/0.822979`恢复到`0.165779/0.043213/0.575632`，temporal/uncertainty为`0.573748/0.000516`。epoch用时=`4,296.16 s`，峰值进程GPU memory=`15,556,673,536 bytes=14.489 GiB<16 GiB`。当前自动进入seed0 epoch4第二轮联合训练；这些是内部loss进展，最终是否超过非学习基线仍只由冻结的C07几何/关系指标决定，C08仍未读取。
- seed0 epoch4第二轮联合训练完整结束：累计optimizer steps=`133,680`，训练/C07人口仍精确为`426,552/64,644`行和`26,736/522`个batch；全部loss与gradient norm finite，`epoch_04.pt`=`23,807,103 bytes`、SHA-256=`3ea307e0431c6f9896fb3ed10fb848ce625cd8404eb779c4bb117177f53cbec2`。C07总loss由`0.300798`小幅降至`0.298724`，primitive/temporal分别改善`2.29%/14.60%`，但surface/port-relation分别回升`18.26%/8.93%`，因此只判训练完整且未失稳，不作科学PASS。epoch用时=`4,299.70 s`，峰值进程GPU memory=`15,690,891,264 bytes=14.616 GiB<16 GiB`。当前自动进入seed0 epoch5最后一轮联合训练，C08仍未读取。
- seed0六轮训练现已完整结束：累计optimizer steps=`160,416`，每轮训练/C07人口均精确为`426,552/64,644`行和`26,736/522`个batch，六类loss、总loss及gradient norm全部finite。最后一轮训练目标继续降到`0.215987`，但C07总loss由epoch4的`0.298724`回升至`0.313710`；正式选择器因此保留epoch4，`selected.pt`与`epoch_04.pt` SHA-256逐字节相同，均为`3ea307e0431c6f9896fb3ed10fb848ce625cd8404eb779c4bb117177f53cbec2`。seed0总耗时=`25,562.59 s`、峰值CUDA allocation=`7,936,714,752 bytes`、峰值进程GPU memory=`16,104,030,208 bytes=14.998 GiB<16 GiB`。这证明训练和C07选模合同正常，不代表冻结科学指标已通过；总控已自动启动seed1，C08/C09/C10、graph和M-TARE读取仍为0。
- seed1 epoch0几何预训练已完整结束：训练=`426,552 rows / 26,736 batches`、C07=`64,644 rows / 522 batches`、累计optimizer steps=`26,736`，全部六类loss、总loss与gradient norm finite。C07总loss=`0.696785`，与seed0同阶段不同但不允许据此选seed或修改冻结方案；`epoch_00.pt`=`22,320,635 bytes`、SHA-256=`5bbd5dc9a2d93221f938592fee50ac2e1d2a7cb10dd62989c2d2ca5a7b2cc591`。epoch用时=`4,225.29 s`，峰值CUDA allocation=`7,955,024,896 bytes`、峰值进程GPU memory=`16,101,933,056 bytes=14.996 GiB<16 GiB`。当前已自动进入seed1 epoch1关系预训练，C08/C09/C10、graph和M-TARE仍未读取。
- seed1 epoch1关系预训练已完整结束：累计optimizer steps=`53,472`，训练/C07人口仍精确为`426,552/64,644`行与`26,736/522`个batch，全部loss与gradient norm finite。C07 port-relation loss由`0.697757`降至`0.577823`（下降`17.19%`），总loss由`0.696785`降至`0.624677`；几何、ray和时序项在关系单任务阶段回升，后续联合训练必须恢复，当前不作科学PASS。`epoch_01.pt`=`23,003,383 bytes`、SHA-256=`5c897bbf5fdb450cc37988d462815fd292cbba9413d4e3700ac643f77d3a8435`，epoch用时=`4,212.53 s`，峰值进程GPU memory=`15,636,365,312 bytes=14.563 GiB<16 GiB`。当前已自动进入seed1 epoch2时序/不确定性预训练，C08/C09/C10、graph和M-TARE仍未读取。
- seed1 epoch2时序/不确定性预训练已完整结束：累计optimizer steps=`80,208`，训练/C07人口仍精确为`426,552/64,644`行与`26,736/522`个batch，全部loss与gradient norm finite。C07 temporal loss由epoch1的`2.086685`降至`0.437748`（下降`79.02%`），uncertainty由`0.218156`降至`0.012942`（下降`94.07%`），总loss降至`0.398938`；surface/port/ray按单任务阶段预期回升，须由联合训练恢复。`epoch_02.pt` SHA-256=`282e4bfa4363313146e1316af0238c271bbbbc1b32197e9f045d4601b27a123e`，epoch用时=`4,108.59 s`，峰值进程GPU memory=`15,948,840,960 bytes=14.854 GiB<16 GiB`。
- seed1 epoch3首轮联合训练已完整结束：累计optimizer steps=`106,944`；C07 surface/ray/port分别从epoch2的`0.447559/0.182095/0.859987`恢复到`0.170429/0.040498/0.540537`，总loss降至`0.295186`，temporal/uncertainty保持为`0.462163/0.001311`。`epoch_03.pt` SHA-256=`6cf00643b14a9e644880090f57de4d01f0dfdcb0e6eb41f1ca5bdf8884cc46cb`，epoch用时=`3,957.33 s`，峰值进程GPU memory=`15,495,856,128 bytes=14.431 GiB<16 GiB`。
- seed1 epoch4第二轮联合训练已完整结束：累计optimizer steps=`133,680`；C07总loss进一步降至当前最佳`0.291615`，surface/port/temporal=`0.149446/0.501851/0.385202`，全部训练与选择loss、gradient norm finite。`epoch_04.pt` SHA-256=`b83cc0d768f318e72a0ed186f8689401858325c2379111e32faabbca7096fb4e`，epoch用时=`3,961.74 s`，峰值进程GPU memory=`15,690,891,264 bytes=14.616 GiB<16 GiB`。当前自动执行seed1 epoch5最后一轮；这些仍是内部选择loss，科学PASS只由六轮结束后的冻结C07几何/关系指标决定，C08/C09/C10、graph和M-TARE仍未读取。
- seed1六轮训练现已完整结束：累计optimizer steps=`160,416`，每轮训练/C07人口均精确为`426,552/64,644`行和`26,736/522`个batch，全部loss与gradient norm finite。最后一轮C07总loss由`0.291615`小幅降至`0.289074`并被固定选择器选中；ray与primitive/temporal项改善，但surface与port-relation分别回升到`0.169851/0.556585`，显示多任务间仍有顾此失彼，不能据此宣称结构恢复超过基线。`selected.pt`与`epoch_05.pt` SHA-256均为`ab7768ea4eb6b5ec47095f0fa0df1761109b391810db476e336e715783ecada8`；总耗时=`24,721.29 s`，峰值进程GPU memory=`16,101,933,056 bytes=14.996 GiB<16 GiB`。总控已自动启动seed2；C08/C09/C10、graph和M-TARE读取仍为0。
- seed2 epoch0几何预训练已完整结束：训练=`426,552 rows / 26,736 batches`、C07=`64,644 rows / 522 batches`、累计optimizer steps=`26,736`，全部六类loss、总loss和gradient norm finite。C07总loss=`0.682251`，介于seed0/seed1同阶段的`0.627985/0.696785`之间；这说明早期收敛存在正常种子差异，但没有第三种子异常掉队。`epoch_00.pt`=`22,320,635 bytes`、SHA-256=`f6db9b609d2ec2fd24bf4593dea1176481486790f75c32751bf7ba564d997f95`；本轮用时=`4,187.12 s`，峰值进程GPU memory=`16,101,933,056 bytes=14.996 GiB<16 GiB`。当前已自动进入seed2 epoch1关系预训练；C08/C09/C10、graph和M-TARE读取仍为0。
- seed2 epoch1关系预训练已完整结束：累计optimizer steps=`53,472`，训练/C07人口继续精确为`426,552/64,644`行和`26,736/522`个batch，全部loss和gradient norm finite。C07 port-relation loss由`0.703490`降到`0.686872`，仅改善`2.36%`，弱于seed0/seed1同阶段约`25.96%/17.19%`；同时surface loss由`0.167282`升到`0.533279`。这说明seed2在关系单任务阶段只获得较弱连接信号并明显遗忘几何，是后续联合训练必须修复的具体风险，但尚非六轮后的科学失败。`epoch_01.pt`=`23,003,383 bytes`、SHA-256=`4130ff198553c93fc02db21128391146161254b95622c959d2eb72a89144dfce`；本轮用时=`4,156.06 s`，峰值进程GPU memory=`15,804,137,472 bytes=14.715 GiB<16 GiB`。当前已自动进入seed2 epoch2时序/不确定性预训练；C08/C09/C10、graph和M-TARE读取仍为0。
- seed2 epoch2时序/不确定性预训练已完整结束：累计optimizer steps=`80,208`，训练/C07人口仍精确为`426,552/64,644`行与`26,736/522`个batch，全部loss和gradient norm finite。C07 temporal loss由`2.105678`降到`0.628625`（改善`70.15%`），uncertainty由`0.103742`降到`0.002108`（改善`97.97%`），说明五帧对应与置信度学习有效；surface由上一轮`0.533279`回落到`0.363073`，但仍未恢复epoch0的`0.167282`，所以关系阶段造成的几何遗忘仍需后三轮联合训练修复。`epoch_02.pt`=`23,807,103 bytes`、SHA-256=`8753e4b7a8389cd03d14d3f32b1dfd4f4960209edba31aca9c9b5bdce7aaa761`；本轮用时=`4,206.75 s`，峰值进程GPU memory=`16,007,561,216 bytes=14.909 GiB<16 GiB`。当前已自动进入seed2 epoch3首轮联合训练；C08/C09/C10、graph和M-TARE读取仍为0。
- seed2 epoch3首轮联合训练已完整结束：累计optimizer steps=`106,944`，训练/C07人口继续精确为`426,552/64,644`行和`26,736/522`个batch，全部loss与gradient norm finite。C07 surface/ray/port-relation分别由epoch2的`0.363073/0.229778/0.669983`恢复到`0.158167/0.039639/0.565359`，temporal/uncertainty仍保持在`0.542759/0.000626`，总loss由`0.401465`降到`0.301643`。人话结论是：联合训练已经修复了关系阶段造成的几何遗忘，并且连接关系和时序能力没有因此崩塌；但这仍是内部loss，不等于已超过非学习基线。`epoch_03.pt`=`23,807,103 bytes`、SHA-256=`bd7bebc934ec211df8f550a45a399a798cfd4370c4ed1429619984951182f728`；本轮用时=`4,271.80 s`，峰值进程GPU memory=`15,535,702,016 bytes=14.469 GiB<16 GiB`。当前已自动进入seed2 epoch4第二轮联合训练；C08/C09/C10、graph和M-TARE读取仍为0。
- seed2 epoch4第二轮联合训练已完整结束：累计optimizer steps=`133,680`，训练/C07人口仍精确为`426,552/64,644`行和`26,736/522`个batch，全部loss与gradient norm finite。C07 port-relation/temporal由`0.565359/0.542759`小幅改善到`0.561436/0.542357`，但surface/ray/primitive-set由`0.158167/0.039639/0.503309`回升到`0.172958/0.043921/0.727826`，总loss由`0.301643`回升到`0.341573`。人话结论是：连接关系和五帧追踪仍然稳定，但模型第二轮联合训练后更难准确解出基元的具体形状参数，是模型多任务顾此失彼，不是数据或系统故障。固定选择器当前应保留更均衡的epoch3，最后一轮仍按预注册日程自然完成。`epoch_04.pt`=`23,807,103 bytes`、SHA-256=`4efa3d05eeebe6158e6af56da4dbc9164546076d3dbcf8780844a461bf02801a`；本轮用时=`4,311.65 s`，峰值进程GPU memory=`15,690,891,264 bytes=14.616 GiB<16 GiB`。当前已自动进入seed2 epoch5最后一轮；C08/C09/C10、graph和M-TARE读取仍为0。
- seed2 epoch5及整个三种子六轮训练现已完整结束：seed2累计optimizer steps=`160,416`，三种子总计`481,248`次；每轮训练/C07人口均精确为`426,552/64,644`行和`26,736/522`个batch，全部loss与gradient norm finite。seed2最后一轮相比epoch4改善了port/primitive/temporal，C07总loss由`0.341573`降到`0.314999`，但仍高于epoch3的`0.301643`，固定选择器正确保留epoch3。`epoch_05.pt` SHA-256=`56fd2d7d5ab197bc183a2d61588be039652da2e0b5c334ecde9c1bc6610694f4`，`selected.pt` SHA-256=`bd7bebc934ec211df8f550a45a399a798cfd4370c4ed1429619984951182f728`；seed2总用时=`25,418.28 s`，峰值进程GPU memory=`16,101,933,056 bytes=14.996 GiB<16 GiB`。人话结论是：训练完整、资源合规，但多任务顾此失彼仍是模型风险；是否真正超过非学习基线现由冻结评估器判定。评估器已自动启动C07门检，进程级只读核验未见C08打开；C09/C10、graph和M-TARE读取仍为0。
- C07冻结物理指标门检的seed0已完成并正式FAIL：surface Chamfer=`9.8386 m`，相对非学习基线改善`43.17%`；连续几何macro改善`42.78%`；primitive F1=`0.4167`，增加`11.84`个百分点；target coverage=`99.92%`，增加`82.39`个百分点；时序对应accuracy=`90.32%`。但attachment precision/recall/F1仅=`2.56%/3.40%/2.92%`，相对baseline只增加`2.34`个百分点，低于预注册`5`个百分点门；安全阈值下零提交。人话结论是：模型已经明显学会更完整地恢复几何基元，但因预测`2,066,264`个候选而真实基元只有`544,414`个，多余候选进入两两关系判断后产生`680,094`个错误连接；当前卡在“候选过多+关系判别不安全”，不是几何不可学。seed1/2继续按原合同评估，不改阈值、评分或checkpoint；C08仍未打开。
- 当前主方法Figure 1已完成为`primitive_relation_method_overview`：明确区分训练期construction-program supervisor与部署期五帧LiDAR输入，画出swept primitive set、physical relation set、persistent structural graph、traversal-only edge和不变M-TARE局部栈。PNG/PDF/SVG、结构化source、provenance、确定性publisher和manifest均已生成并逐项hash复核；manifest SHA-256=`d2d6a23fe2a44ddad17b8f131b3bfa0219310ebd8fd4df1dcf9d21da8b10b63d`，图中没有未完成实验数值。旧event/exit-token方法图继续只作基线证据。
- 当前数据/Teacher Figure 2也已完成为`primitive_relation_dataset_overview`：从P1a/P1b密封证据直接读取字典序首个fit父世界`S01...C01`、`c1_mixed`和global frame 0，展示真实生成拓扑/轴线、连续截面族、学生可见range图和训练期primitive source-set code。选择规则与模型结果无关；源NPZ、summary、provenance、publisher及PNG/PDF/SVG已逐项hash验证，manifest SHA-256=`a517298d533dabed8ae848435b1378f1168edb4efb8839ec81ff1b32e5d7b17b`，C09/C10读取为0。
- 同输入非学习C07基线已发布为正文Figure 3：正式PNG/PDF/SVG从密封baseline run逐字节复制，并新增机器可读指标、provenance、确定性publisher和SHA-256清单；manifest=`54e080ae5c40484ad9a684d6b8e3690d86a28f1881380a0bec64106cd4dd3772`，全部5项清单文件复核OK，C08/C09/C10读取为0。旧工作继续按“正式对比/单变量消融/科学失败/复现错误”四类保留，不以成败决定删除。

## 2026-08-30：P2基元关系模型readiness正式PASS；进入非学习基线与三seed训练冻结

- P2 V1首次正式run完成全部真实前后向但科学FAIL，唯一失败项是Teacher槽位重排不变量：C1-mixed中两个基元形成Hungarian近似平局，存储列顺序改变了匹配，最大单项loss差=`1.7344951629638672e-05 > 1e-6`。其余15项检查全部PASS，`error=null`、0 optimizer/checkpoint/graph/C07--C10/M-TARE；V1保持封存。
- V1R唯一修复是在不变Hungarian目标前按端点反转不变的基元几何签名及迭代端口/overlap关系签名规范排序；没有改变P1数据、Teacher、32槽、模型、六类loss、权重或阈值。新增规范排序回归后完整测试=`30/30 PASS`。
- V1R正式读取一个C04 fit源序列的三种配对几何，共15帧、每行5个基元、8个有向attachment、6个disconnected overlap和4个temporal dustbin；三行共享`source_global_sequence_index=188724`，独立统计单位仍为1。
- 正式V1R全部16项检查PASS：参数=`1,969,516`，六类loss及总loss finite，`141/141`参数梯度finite且nonzero，重复推理误差0，Teacher槽位重排/端点反向loss误差均0，query permutation误差=`1.9073e-06`，配准旋转误差=`7.6294e-06 m`，关系logits严格对称。
- 峰值CUDA allocation=`379,431,936 bytes`，0 optimizer/checkpoint selection/graph/C07--C10/M-TARE；RUN_STATE=`COMPLETED`、`error=null`。13项证据seal SHA-256=`39f72889a7333a6c9ee060ebf13a317cafb0a8ef1762aee9a21144f148e3968f`，决策=`ALLOW_P2_THREE_SEED_TRAINING_SPEC`。
- `NEXT`：先冻结同一因果扫描输入的非学习鲁棒超椭圆/轴线/端口关系baseline及其C01--C06计算预算；再提供完整训练Data Card，固定C01--C06训练、C07 checkpoint/阈值选择、C08一次零适配迁移和seeds 0/1/2。训练前不得读取C08结果选择模型，C09/C10和graph继续关闭。

## 2026-08-30：P1b五帧基元关系Teacher正式PASS；P1数据阶段闭合

- 初始V1在全量物化前由真实分片readiness停止：双源适配后pilot任务漏传`source_run`，触发`KeyError`；10/10单元测试PASS、正式Teacher分片为0。该run以系统FAIL封存，不覆盖、不续跑。
- V1R唯一修复是把sealed P1a corrective路径显式传给真实分片pilot，并把V1失败summary/RUN_STATE/seal绑定为输入；同一修正P1a真实分片双重复分别耗时`14.87/14.52 s`，完整tree SHA均为`7d1e4b719a486be546189334f5e57a4e9025578f4f203e4290a73ba3347e5769`。
- 正式V1R精确完成`80 worlds / 240 tasks / 757,290 source frames / 564,378 five-frame sequences / 24,117 primitives`，最大可见基元=`24<32`；10项正式检查全部PASS。
- 标签人口为directed endpoint attachments=`8,779,620`、disconnected overlap=`3,023,794`、temporal dustbin=`538,391`，证明连接、相似重叠困难负例和遮挡对应均非空。
- Teacher分片=`283,883,437 bytes`，远低于12 GiB；验证127,380项P1a seal，0 optimizer/inference/graph/C09/C10，RUN_STATE=`COMPLETED`、`error=null`、耗时=`1,118.67 s`。
- V1R证据seal共41,390项，SHA-256=`f629b511e9a945bbe15249e3bc212be57a01b1aa0118d0d7d228824aa25f7d47`；决策=`ALLOW_P2_PRIMITIVE_RELATION_MODEL_TRAINING`。
- `NEXT`：实现并正式审计底层基元关系模型readiness，只使用C01--C06真实batch和解析ellipse/rounded/taper/curve/T/Y/X/stacked合同，验证32槽、排列/端点反转、SE(3)配准、六项loss和有限梯度；PASS后才冻结三seed训练。

## 2026-08-30：P1a无损存储corrective正式PASS；进入P1b关系Teacher物化

- P1a源run保持不可修改的resource-only FAIL；唯一修正run已完成`240/240`任务、`80`个父世界、`757,290`帧、`8,723,980,800`条射线、`24,117`个基元和`402`次姿态修正，C09/C10、训练、推理与graph均为0。
- 2,640个数组逐chunk复读后与源dtype原始字节一致，11数组/分片inventory、shape、chunks、dtype、fill/order/filter、group attrs及codebook/construction全部保持；240个源tree和目标tree均有独立SHA-256。
- 数据分片从`22,395,827,779`压缩到`19,388,655,191 bytes`，比率=`0.865726`，满足冻结的`<=20 GiB`门；全部9项正式检查PASS，RUN_STATE=`COMPLETED`、`error=null`。
- 正式corrective运行时长=`1,835.52 s`，127,380项seal SHA-256=`79fd988ac8c205d74c93e4858b7b579571a06502e778f31634046b48791a0668`；决策为`ALLOW_P1B_RELATION_TEACHER_MATERIALIZATION_FROM_CORRECTED_P1A`。
- P1b冻结器的预期单元测试计数由错误的9修正为实际合同10；正式同解释器复核=`10/10 PASS`，不改变Teacher、数据、阈值或方法。
- `NEXT`：绑定corrective seal冻结并preflight唯一P1b run，物化精确`564,378`个五帧32槽几何/端口连接/非连接重叠/时间对应Teacher；P1b PASS前不训练、不建图、不读取C09/C10或M-TARE。

## 2026-08-30：P1a全量科学数据合同通过但容量门FAIL；bit-exact存储corrective已完成

- P1a正式不可覆盖run已完成并封存：精确`80 parents×3 geometries=240`分片、`757,290`帧、`8,723,980,800`条射线、`24,117`个变体基元、402次配对姿态修正；fit/c07/c08分片=`180/30/30`，有效return=`8,539,172,957`，ambiguous rays=`24,339,748`，最大完整来源数4。
- 12项正式检查中11项通过：world/task/partition/frame/ray/primitive/correction计数、原点、确定性复算、uint16 provenance、完整多来源和所有tree hash均PASS。唯一失败项为`result_bytes_le_20gib`：实际分片`22,395,827,779 bytes=20.857740 GiB`。runner正常退出、`error=null`、RUN_STATE=`COMPLETED`，124,740项source seal SHA-256=`f437992b2d9afead5a7860ef35bc73eb6aeddc8d039d84dff6d39dbec4d5cf93`；因此这是resource-only formal FAIL，不否定数据科学合同，也不允许把原run改写成PASS。
- storage corrective Data Card/spec补齐治理必填字段后preflight=`0 errors/0 warnings`；唯一不可覆盖run已完成，12 CPU workers逐块执行zstd9 byte-shuffle、目标复读raw-byte equality、源/目标tree和side-document一致性。内置测试`2/2 PASS`；源P1a保持不可修改。
- 存储corrective已完成实现准备但尚未冻结/执行：`primitive_relation_lossless_repack.py`仅把Zarr压缩器改为zstd9 byte-shuffle，保持array inventory、shape、chunk、dtype、fill/order/filter和attributes；每个目标chunk写后复读并按原始dtype字节比较。正式runner还要求源P1a恰好只有`result_bytes_le_20gib`一项失败、240个源tree不漂移、11数组/分片全部逐块bit-exact、codebook/construction逐字节相同及修正后`<=20 GiB`。合成测试`2/2 PASS`；完整真实`S01_flat_tree_small_C04__rounded_rectangle`分片的11/11数组与group attrs逐块bit-exact，大小由`36,516,469`降至`31,816,271` bytes，比例`0.871286`，临时输出已移至回收站。只有源P1a完成并证明resource-only FAIL时才允许冻结Data Card/spec和创建新不可覆盖run。
- S05首次在`S05_flat_branch_medium_C04__ellipse`观测到单射线4来源；codebook 119将`edge_0042/0047/0054/0059`完整绑定并实际出现1条射线。这是完整人口超出720个固定审计窗口的抽样最大值，不违反P1a冻结的1--255完整集合合同，也不影响32个五帧可见基元槽。P1b现有遍历逻辑无截断，新增四来源无损测试后完整准备期单元测试=`10/10 PASS`。
- P1b补齐部署可得的五帧相对里程计后，在完整真实`S01_flat_tree_small_C04__c1_mixed`分片上独立执行两次：`1,292`帧、`860`个序列、`54`个构造基元，完整输出tree SHA-256均为`7d1e4b719a486be546189334f5e57a4e9025578f4f203e4290a73ba3347e5769`，临时资产全部删除。860个窗口的相对量全部finite、当前帧平移/偏航逐位为0，输出schema不存在绝对sensor/axis/yaw数组；含四来源完整保留回归在内的准备期单元测试=`10/10 PASS`。
- P2泄漏安全读取接口已实现并`3/3 PASS`：`PrimitiveRelationTrainingShard`配对P1a/P1b的parent/partition/geometry与frame rows，学生对象只暴露`[5,2,16,720]`归一化range/valid及五帧当前传感器坐标相对里程计；绝对sensor/axis/yaw、world/TNG和primitive identity均不进入学生输入。构造identity只存在Teacher目标和审计层，任何配对漂移或越界索引fail-closed。该项只是模型readiness准备，未训练、未读C09/C10。
- 真实标签人口非空：最大可见基元=`17<32`，directed endpoint attachments=`14,616`、disconnected angular overlaps=`5,891`、temporal dustbin=`1,013`；证明P1b确实生成连接、空间重叠困难负例和遮挡对应，而非空壳索引。
- 修正Teacher接口：`unexplored`随机器人历史trace变化，不能由相同五帧LiDAR唯一辨识；不再作为感知标签。模型学习基元端点、连接、非连接重叠、时间对应与不确定性，在线图再用真实穿越维护`unexplored/attempted/verified`。
- 该阶段的`NEXT`已由上方正式corrective PASS取代。

## 2026-08-30：可见基元容量正式PASS并冻结32槽；P1数据导出开放

- V1R绑定最小姿态资格seal，正式覆盖80 parents×3 geometries=`240`任务、每任务junction/terminal/interior三类五帧窗口，共`720`窗口、`41,472,000`条全扫描射线。
- C01--C06 fit最大可见基元=`16`，预注册25%余量要求`20`槽，因此`8/16/32`中冻结`32`；C07/C08最大=`23/19`，不改变容量即可transfer。
- 8槽overflow=`264/720`，16槽仍overflow=`15/720`，32槽=`0`；单ray source membership最大3，uint16 codebook与全部多来源身份完整保留。
- 方法计划中的`[8]`关系张量正式修订为`[32]`；这是由fit-only全量证据触发的容量修正，不是模型结果驱动调参。旧8槽保留为必要容量消融。
- `NEXT`：冻结并执行P1三形状数据导出Data Card/spec；生成757,290帧、564,378个五帧序列、range/valid/provenance code、32槽可见基元Teacher及关系/时序监督。
- run=`results/gate3_semantics/gate3_20260830_primitive_slot_capacity_audit_v1r_seed0`，seal SHA=`8593159efc145189f6e028c06c87c1377ef628837eba39601b1afcc00d8c54ae`。

## 2026-08-30：有限端盖最小姿态修复正式PASS；恢复槽位容量审计

- 原始有限构造union在坡道端点复现`402`次三变体越界，对应`134/252,430`个源姿态；问题来自world-vertical传感器偏置越过有限端盖，不是截面形状或模型问题。
- 延长端盖方案正式科学FAIL：虽消除越界，但最大延长`0.510535 m`且产生`61`个nonincident overlap，已否决并保留为负证据，主路径不调用该实现。
- 采用最小姿态修复：仅对同时不满足三种冻结union的端点帧，沿原directed traversal做`0.025 m`步进、40次二分和`0.025 m`内侧裕量；不修改地图、基元或帧身份。
- 正式PASS：80 worlds、252,430源姿态、757,290三变体检查；修复前/后越界=`402/0`，恰好修正`134`帧，其余`252,296`帧逐元素bit-exact，最大内移=`0.300015 m`，最小相邻arc间距=`0.699985 m`。
- `NEXT`：以该正式姿态资格seal重跑fit-only visible-primitive slot capacity；C01--C06选`8/16/32`中满足25%余量的最小容量，C07/C08只作冻结迁移检查。
- run=`results/gate3_semantics/gate3_20260830_primitive_finite_cap_pose_qualification_v1_seed0`，seal SHA=`8a36f7f46d141df4773f957ea9f67e586b8d602d70611abbcd28395df6c24f1c`。

## 2026-08-30：P1真实五帧窗口发现8-slot容量不足；全量导出暂停并进入容量审计

- 新codebook/visible-target接口34项unit全部PASS；world-local `uint16` code无损保存每条射线的完整primitive source set，歧义不折叠。
- C01 C1-mixed三类真实五帧窗口共172,800 rays：interior/junction/terminal分别需要`7/12/9`个可见primitive slots；junction第12个基元仍有153条surface-hit支持，不是单个数值离群点。单ray来源集合最大cardinality=3。
- 该证据否定方法规范中固定`maximum 8 slots`作为无损Teacher容量；不得截断到8、按命中数事后删基元或继续全量导出。
- `NEXT`：对C01--C06 60个fit父世界×3几何×3固定角色窗口做无训练容量审计，按预注册25% headroom与`8/16/32`最小power-of-two规则选择容量；C07/C08仅作不适配transfer检查。若32仍溢出则停止固定槽decoder并改为可变集合接口。

## 2026-08-30：高速CSG基元身份射线正式PASS；P1训练数据导出已开放

- 正式证明只读`S01_flat_tree_small_C01`的55个C1-mixed扫掠基元，固定普通走廊/交叉口/端点各2姿态，共6,144条均匀抽样射线；C09/C10、模型、训练、graph和M-TARE均为0。
- 30项unit与ellipse/T-union/stacked/ambiguity四类解析合同全部PASS；独立闭合primitive meshes通过ordered multi-hit、解析union候选验证和三点路径检查跳过组合内部面，并完整保留多源歧义。
- 正式结果：valid agreement=`0.998698`、qualified coverage=`0.988607`、primitive identity agreement=`1.0`、range MAE/p95/p99/max=`0.004389/0.014404/0.024525/0.048278 m`，超过5cm的射线为0。
- spacing-padded AABB稀疏查询与全55基元查询的逐hit SHA-256完全相同；稀疏CSG相对解析逐步Teacher加速=`15.56x`。完整`8,723,980,800`条P1射线按2倍edge复杂度保护与32 workers外推=`18.25 h <72 h`。
- `NEXT`：冻结P1 80-parent三几何实现导出Data Card/spec与分片executor；严格保持757,290帧、564,378序列、16×720射线和primitive membership，不删帧、不折叠歧义。
- run=`results/gate3_semantics/gate3_20260830_csg_mesh_provenance_acceleration_v1_seed0`，seal SHA=`ea2da44432aea12497721f4fa630c687e889f16457d66d0e1ee3f1b6033538c9`。

## 2026-08-30：P0三形状80-parent inventory正式PASS；P1只被provenance吞吐阻塞

- 形状合同正式PASS：20项解析/unit合同通过，ellipse/rounded-rectangle/C1-mixed解析射线最大误差=`8.88e-16 m`，端点参数数值导数=`0.00018`；55个C01 edge identity与circle参数精确保留。
- 80-parent inventory的V1按1cm轴线重合合同在5个degree-4端口FAIL；归因证明偏置仅为半径的`3.0%--10.7%`，采用“实际spline endpoint + graph composition anchor、anchor必须在incident free-space内”的物理合同。
- V1R随后在8个stale self-intersection degree字段FAIL；4个world中声明degree均比实际edge incidence多1，且无缺失spline分支，cycle rank与edge列表一致。V1R2以edge incidence为关系真值并完整保留8条mismatch审计。
- V1R2正式PASS：80 parents、8,039 edges、240 paired realizations、24,117 variant primitives、757,290 planned frames、564,378 sequences；面积最大误差=`7.11e-14 m2`，预计同压缩Zarr=`8.77 GB`，native render=`3.61 h`。
- 当前Python逐步隐式provenance实测外推=`80.86 days`，不可扩展；P1不得启动。`NEXT`：实现独立闭合基元mesh的高速多命中CSG union ray backend，按occupancy transition选真正union exit并保留face/primitive identity；先解析+C01吞吐证明。
- inventory run=`results/gate3_semantics/gate3_20260830_geometry_variant_inventory_v1r2_seed0`，seal SHA=`7d31daf13b2403873426c0bba165bef913992699c7493364144c2f4a3aedcef4`。

## 2026-08-30：P0统一swept-superellipse形状合同正式PASS

- 同一连续参数族以`n=2`表达ellipse、`n=6--10`表达rounded rectangle，端点宽高和形状指数沿arc使用cubic smoothstep做C1变化；curve/slope/taper/T/Y/X/stacked/ambiguity全部纳入20项合同。
- 三种新几何实现只由parent/edge SHA-256决定，不看LiDAR结果：ellipse、rounded rectangle、C1 mixed；每个端点严格保持原radius circle面积。
- run=`results/gate3_semantics/gate3_20260830_swept_superellipse_contract_v1_seed0`，seal SHA=`04203685c0948f833d60c738a230e57fb331bd4b22ef9dc40d98cf04141ffef3`。

## 2026-08-30：P0逐射线基元来源修正正式PASS；进入形状泛化合同

- 唯一正式corrective只读取一个C01训练世界，在6个固定基元中点位姿生成`6×16×720=69,120`条Teacher射线；C07/C10、模型、训练、graph和M-TARE均为0。
- `67,676`条射线命中组合隧道表面：`62,311`条具有唯一edge-primitive来源，`5,365`条具有多基元来源并保留完整55维membership、primary identity=`-1`、训练mask=`0`；`1,444`条在50m内无命中。
- 两遍完整投射逐元素相同，10项unit全部PASS，membership cardinality零不一致，歧义标签零误分配。此前“Poisson后身份丢失”的阻塞已由双资产合同解除：Teacher使用保留构造身份的隐式几何，感知域仍可使用native Poisson mesh。
- 该PASS证明圆形C01资产的provenance链，不等于三种截面数据已具备。`NEXT`仍属于P0：实现shape-generic swept-superellipse合同与ellipse/rounded-rectangle/taper/curve/T/Y/X/stacked解析测试，并完成80个父世界的几何变体只读inventory/资源证明；通过后才进入P1生成。
- run=`results/gate3_semantics/gate3_20260830_primitive_provenance_corrective_v1_seed0`，seal SHA=`cbf9f214fad25b3d04e3c05b2b81e51c20d77e795a268007d83016d22554bb06`。

## 2026-08-30：P0构造监督可行性正式FAIL；结构可导出但表面来源已丢失

- 正式proof只读一个C01父世界和675帧schema，固定抽查6行；C07--C10、训练、推理、graph和M-TARE均为0，系统`error=null`。
- 55条graph edges唯一生成55个扫掠基元，56个节点组合覆盖全部110个端口，最大node-to-spline投影误差=`0.001668807 m`。
- Cano在统一Poisson重建时丢失逐面构造身份，旧LiDAR也没有triangle/surface/primitive hit ID；故不能无歧义生成完整Teacher，禁止用最近几何伪造标签。
- `NEXT`：实现pre-Poisson primitive/surface provenance和raycaster hit identity合同；先做解析场景与一个C01小型重生成proof，仍不训练。
- run=`results/gate3_semantics/gate3_20260830_primitive_construction_supervision_feasibility_v1_seed0`，seal SHA=`1a5bb883942d281eabda5e10a84bb59cfd57e3ff6c5f36bbb23e04bb09319832`。

## 2026-08-29：论文主线改为程序构造监督的几何基元关系结构图

- `CURRENT`：Phase 3方法与数据合同；完整权威计划为`docs/PRIMITIVE_RELATION_STRUCTURAL_GRAPH_PLAN_V1.md`。
- `QUESTION`：五帧因果LiDAR能否学习显式扫掠超椭圆基元、端口组合与跨帧变换关系，并由这些关系形成比出口规则图更准确的结构语义图谱。
- `DATA`：复用C01--C08 80个拓扑父世界、252,430帧和188,126个五帧序列；计划在父世界split不变的条件下增加圆角矩形和连续混合截面，正式生成前必须重新Data Card和preflight。
- `METHOD`：construction program supervisor + explicit swept primitives + learned endpoint relations + temporal correspondence + traversal-verified graph。
- `BASELINES`：非学习基元拟合、Cano-like出口规则图、原M-TARE和GT-TNG oracle；旧Composer/集合出口/ERCSS保留为失败证据与消融。
- `RETIRED`：不再执行ERCSS V1R3；旧停止规则和三次系统失败记录保持不可修改。
- `NEXT`：只实现零训练construction-supervision feasibility pilot，验证生成器能否无歧义导出基元、变换、组合、surface provenance和可见端口；范围限解析微场景和C01少量开发世界，C07--C10/训练/graph/M-TARE均为0。

## 2026-08-29：RouteGeometryProfile正式科学FAIL；停止双Composer路线并重构为配准骨架学习

- 唯一正式proof完整读取C01--C08 `80` worlds/`188,126` observations及全部turn/transition=`1,998/1,031`行、`392/76`物理identity；系统`error=null`、source unchanged、peak RSS=`680,564 KiB`，C09/C10/M-TARE/graph/model/training均为0。
- Teacher本身通过：`62,180`个正反向canonical pairs的width/height/slope/curvature p95差=`0.4981m/0.2788m/0.1545°/0.001274m^-1`，均在既有几何限内。失败不是Teacher歧义。
- 单帧前向0/5/10/15/20m五段LiDAR支持不足：fit/selection的turn identity coverage=`21.55%/29.47%`，transition=`76.27%/58.82%`，平均仅支持约`1.55--1.89/5`段；transition完整五段Teacher行均为0。冻结80% identity/三段支持门失败。
- 决定不缩短profile、不降低覆盖率、不重试Composer或用graph补偿。`GeometrySemanticState -> dual Composer -> single-frame forward RouteGeometryProfile`方法链正式停止，失败代码、模型和图保留为消融证据。
- 相关工作复核表明：Cano已覆盖出口方向与纯拓扑导航；STGPlanner/SphereMap/地下SER已覆盖从dense占据图或几何地图抽取骨架；CenterLineDet已覆盖序列传感器到vectorized centerline graph；PRISM-TopoMap已覆盖point-cloud/odometry地点图与学习匹配。因此“预测骨架”本身不能作为贡献。
- 新候选收窄为`Ego-Motion Registered Causal Structural Skeleton (ERCSS)`：用部署时已有的相对里程计把五帧LiDAR配准为当前局部点集，直接预测带宽高坡度曲率与不确定性的可见稀疏隧道骨架片段；全局节点由稳定骨架分叉/终止产生，edge仍只由真实穿越提交。差异是无需dense全局地图、不是出口计数，也不让地点descriptor决定结构。
- NEXT只允许零训练Teacher/表示可行性：核对全部五帧相对位姿、因果点云配准、可见spline骨架唯一性、分叉/终止identity覆盖、固定容量无overflow及禁止绝对pose/world/TNG进入学生。PASS后才做model readiness；FAIL则重新评估论文主题。
- proof run=`results/gate3_semantics/gate3_20260829_gse_route_geometry_profile_proof_v1_seed0`；18项seal SHA=`91c8f43a02dcc6e0e59273ed35dd92a090878c3818ea9a6cce942c2a1a6ea020`，图SHA=`900e0f60c4e3fb0fea919a24716c96ec699daef5c8fb9ba236a130d89263534f`。

## 2026-08-29：双Composer三种子正式科学FAIL；图继续关闭

- 唯一不可覆盖run完成seed 0/1/2各10轮事件训练和2轮拒绝训练，共`5,004`个optimizer steps；三个trainer与统一评测均无系统错误，正式状态为`FAIL_GSE_DUAL_COMPOSER_THREE_SEED_TRAINING_V1`、`error=null`。
- C08三seedfull macro-F1=`0.535298/0.524545/0.529239`，对应旧V2R5 corrected-Teacher baseline=`0.581537/0.597852/0.588838`，平均增益=`-0.059715`，违反至少`+0.05`的主门。ensemble full=`0.537115`，baseline=`0.595675`。
- ensemble的turn F1从baseline `0.179159`降至`0.129777`；geometry-transition虽由`0.005495`升至`0.047223`，但高精度identity coverage仍为0。full在C07找不到满足安全门的非空阈值，C08提交数为0；去掉refusal后C08 precision仅`0.328028`、false accept=`67.1972%`。
- 消融否定当前表示：no-metric ensemble macro-F1=`0.559359`，高于full；no-transport=`0.537107`，与full几乎相同。metric ablation虽逐seed改变结果，但方向不是完整方法收益；transport没有跨seed必要性。C09/C10/M-TARE/graph/planner读取与回放均为0。
- 结论分类为`CURRENT_GEOMETRY_SEMANTIC_STATE_LACKS_LOCALIZED_ROUTE_PROFILE_AND_TRANSPORT_CONTRIBUTION`，不是数据、Teacher、随机种子、数值或系统失败。禁止增加epoch、调C08阈值、删除Metric Composer或进入图补偿。
- NEXT唯一允许步骤：按`docs/GSE_GRAPH_METHOD_SPEC_V1.md`做零训练`RouteGeometryProfile`可见性/Teacher唯一性证明，检验因果LiDAR能否支持沿前向通道的宽、高、坡度、曲率纵向剖面；证明PASS后才允许新表示readiness，FAIL则停止当前GSE方法方向并重新评估贡献。
- run=`results/gate3_semantics/gate3_20260829_gse_dual_composer_three_seed_training_v1_seed0`；66项证据seal SHA=`e32d182d69d611a7389da30e7200b3434034bad8e8461b6edf1fd74173deada8`，论文图SHA=`020d059713dc9633e66aee765dcc4974008d8bba4d355bf88b32de5db8a478af`。

## 2026-08-29：双Composer因果监督包V1R正式PASS

- V1完整生成80个world监督包和科学summary，全部计数/因果检查PASS，但最后把Python字典直接传给Matplotlib分类轴，触发`TypeError: unhashable type: 'dict'`；正式V1封存为系统FAIL，0训练/推理/test/graph。
- V1R唯一修复是将绘图输入显式转换为key/value列表，并新增PNG/PDF/SVG真实输出单测；Teacher、cache、标签、回投规则、计数和阈值均未改变，V1失败summary/seal被绑定为输入。
- 正式V1R精确完成80 worlds/188,126 rows，split=`142,184/21,548/24,394`；事件=`150,964/26,608/7,525/1,998/1,031`，有效turn/transition回投帧=`1,174/410`，覆盖全部`392/76`个物理身份。
- run=`results/gate3_semantics/gate3_20260829_gse_composer_supervision_v1r_seed0`，`COMPLETED`、`error=null`、99项证据、seal SHA=`3c2b8511fb11b8d8c91784871be5c12ce13d26db737997e55c45165f7cf6ff54`；0 optimizer/inference/C09/C10/M-TARE/graph/planner。
- NEXT进入每seed独立双Composer trainer/evaluator与no-metric/no-transport/no-refusal必要消融实现；C01--C06拟合、C07选择、C08一次迁移，graph继续关闭。

## 2026-08-29：三种子geometry-only显式状态缓存正式PASS

- 首次V1在seed0前置C07复现时因`token_count_probability`跨进程float16差`7.63e-6`按“所有字段bit-exact”合同FAIL；0个fit world导出、0 optimizer/checkpoint/test/graph。只读归因证明同一当前进程重复前向全部字段bit-exact，历史差异只存在于CUDA派生概率：count/transport-row/reveal最大`7.63e-6/4.88e-4/1.91e-6`，全部argmax/reveal@0.5决策一致，8个主体字段误差0。
- V1R固定表示级corrective：主体字段仍要求bit-exact；仅三个派生binary16概率允许`<=np.finfo(float16).eps=0.0009765625`，同时全部离散决策必须相同。15项unit、Data Card/spec和preflight 0/0通过后执行唯一V1R。
- V1R完成3 seeds ×80 worlds=`240`个typed NPZ、每seed C01--C06/C07/C08=`142,184/21,548/24,394`行，共`564,378`次冻结forward。60个development seed-world比较主体最大误差0、派生最大误差`0.00048828125`、全部决策相同。
- 缓存只含global row、bearing/existence logits/opening/profile/uncertainty/count/transport/global geometry共11字段；没有event/context/place或token descriptor/pose/world/TNG/identity。总缓存473 MiB，peak GPU约4.25 GiB、peak RSS约2.14 GiB、总时长347.45s。
- run=`results/gate3_semantics/gate3_20260829_gse_explicit_composer_cache_export_v1r_seed0`，`COMPLETED`、`error=null`、265项seal SHA=`edc613469f91c15f87be39c87cc3416e544e989fe267263fe768668dfe65faed`；0 optimizer/checkpoint/C09/C10/M-TARE/graph/planner。NEXT进入三seed Composer trainer/evaluator和必要消融实现，感知骨干不再重复前向。

## 2026-08-29：类型化双Composer正式readiness PASS

- 新增独立`ActionSetRelationComposer`与`MetricChangeComposer`。前者只接收五帧出口bearing/存在概率/开口宽度/垂直轮廓/几何不确定性/count及previous→current/dustbin/reveal transport；后者只接收五帧预测width/height/slope/curvature及不确定性。两者的typed forward都不暴露encoder context、旧event logits、place/exit descriptor、pose、world、TNG、Teacher identity或未来帧。
- Action Composer=`102,500`参数，Metric Composer=`36,485`参数；正式run精确核对C07 10 worlds/21,548 rows，并用覆盖五事件的10个固定真实样本及22条必要历史预测执行完整forward/loss/backward。
- 15/15正式检查和6/6 unit全部PASS：token permutation/global rotation误差均`2.98e-8`，两个batch permutation误差`2.98e-8`，repeat误差均0；回投仅支持有效过去/当前帧，全部参数gradient存在且finite。
- 唯一run=`results/gate3_semantics/gate3_20260829_gse_typed_composer_readiness_v1_seed0`，`COMPLETED`、`error=null`、peak RSS=`711,088 KiB`、19项seal SHA=`2111d45d13377487acc026e8601375c91caa08e49f78de6f71a58ddb8b2c438d`；0 optimizer/checkpoint/C08/C09/C10/M-TARE/graph/planner。
- 该PASS只证明论文主方法的受限接口正确且可训练，不证明事件性能。NEXT唯一允许步骤：冻结三seed Composer训练Data Card/trainer/evaluator，V2R5三seed各自冻结，只训练对应小Composer；C01--C06训练、C07 checkpoint/校准/refusal、C08一次零适配，未达到方法规范全部科学门前graph继续关闭。

## 2026-08-29：当前权威状态——V2R5失败封存，显式结构可观测性通过

- 论文主线仍处于 Phase 3，单一研究问题是：仅由五帧因果 LiDAR 提取的出口关系与度量几何，能否可靠地产生路口、终点、转弯和几何变化事件。正式拓扑图、C09/C10 严格测试和 M-TARE 闭环仍未开放。
- V2R5 三个随机种子已完整训练，共 `36,690` 次参数更新；C07/C08 使用 `21,548/24,394` 帧，系统运行、数据隔离和证据封存均正常，但科学结论为 `FAIL`。C08 旧事件 macro-F1=`0.590305`，几何相对非学习估计器总体改善=`7.795%<10%`，节点关联 precision=`0.976744<0.98`、recall=`0.011257`、false merge=`2.3256%>1%`，因此不得进入建图。
- 随后的显式 Composer 可观测性审计只使用 token/transport/width/height/slope/curvature，不允许旧事件分数、隐藏特征、descriptor、位姿、TNG、未来帧或测试世界。C07 拟合、C08 一次零适配评价，共 `45,942` 帧、12 个固定诊断器。
- 审计结论为 `PASS_GSE_COMPOSER_OBSERVABILITY_V1`：C08 junction/terminal/turn/transition AUC=`0.965620/0.999131/0.768548/0.737538`，证明显式几何中存在结构信号，允许继续实现双 Composer；但高精度 identity 覆盖仅为 `73/77、69/69、4/52、0/12`，尚未证明可部署召回。
- NEXT 唯一允许步骤：实现并验证类型化 `Action-Set Relation Composer` 与 `Metric-Change Composer` readiness。第一版冻结 V2R5 backbone/token/transport/geometry，只训练小型 Composer；先证明接口无旁路、因果、确定、可反向传播，再冻结正式训练 Data Card。未达到 precision `>=0.98`、false accept `<=1%`、recall `>=0.25` 及事件/identity 改善前，不进入拓扑建图。
- 汇报材料已分为两层：19页导师主报告`docs/GSE_GRAPH_ADVISER_PROGRESS_REPORT.html/.pdf`按“地图与数据→Teacher→失败证据→方法转向→当前障碍”组织，并展示树形/单环/复杂三维三类真实地图；28页`docs/GSE_GRAPH_METHOD_EXPERIMENT_REPORT.html/.pdf`保留为技术附录。

## 2026-08-29：Sparse Circular Relation Transport V2 readiness正式PASS

- 新V2为`783,675`参数，五帧逐方位causal backbone为每帧产生最多6个圆周token；相邻帧previous→current/dustbin与current reveal logits显式表示persistent/withdraw/reveal，forward唯一输入仍为`scans`。
- 53个predecessor backbone state keys从Slot seed0 checkpoint逐键、逐shape、逐值无损加载；没有missing/shape mismatch/unexpected。该复用恢复已证明的五帧axis路径，不重新从单帧学习方向。
- 固定8条C01--C03观察/34个历史帧覆盖五事件及persistent/reveal/withdraw；proposal、transport row、reveal、column exclusivity和完整辅助loss均finite，全部trainable parameters有gradient，0 missing/nonfinite，最大绝对gradient=`1.11413`。
- 16项正式检查全部PASS：同一pair同时reveal+withdraw、real-to-real、dustbin、rotation、batch permutation、reverse involution、deterministic repeat、完整人口provenance与真实batch均成立。proposal/transport/axis rotation误差=`7.45e-7/2.38e-7/1.79e-7`，semantic permutation=`5.96e-7`，repeat/reverse=`0`。
- 正式run `error=null`、source unchanged、0 optimizer/checkpoint/C09/C10/M-TARE/graph/planner；NEXT允许冻结V2三seed训练Data Card、trainer与严格C07选择/C08零适配评估。readiness只证明接口可训练，不证明科学性能。
- run=`results/gate3_semantics/gate3_20260829_gse_sparse_circular_relation_transport_readiness_v1_seed0`；seal SHA=`293a0abda403697772f09f811127a26b78a30b1f463a6cc64f7703c136d793f6`，图SHA=`e304de6bdc3f544819dce51d83d1036f301ec1c867fdfb220098eb1c1d5c0e05`。

## 2026-08-29：V2失败归因正式PASS；下一接口固定为稀疏圆周关系transport

- 正式零训练归因逐项复现C07/C08 `21,548/24,394`条观察、`66,752/77,490`个有效关系pair和三通道正例人口；五项机制检查全部PASS，`error=null`、source unchanged，0 optimizer/inference/C09/C10/M-TARE/graph/planner。
- 轴向根因闭合：当前last-frame directional head为`82.7165°/83.0120°`，同人口五帧directional-temporal predecessor为`5.4982°/5.7835°`，差距超过70°。
- 稀有关系的fit正例率persistent/reveal/withdraw=`1.1588%/0.01312%/0.01365%`，对应dense BCE正权重=`43.15/3811.18/3664.03×`。C07/C08所有通道在precision `>=0.98`下recall均低于`0.25`，正式安全失败完全复现。
- 关系并非完全不可观测：positive-vs-empty pair AUC约`0.84--0.92`；但exact-bin AP中persistent约`0.133--0.138`、reveal/withdraw仅`0.006--0.009`。oracle-count下±8°匹配率persistent/reveal/withdraw约C07=`45.8%/38.1%/35.0%`、C08=`46.1%/40.4%/37.6%`，同时p90误差仍`77--89` bins。
- 决定=`ALLOW_SPARSE_CIRCULAR_RELATION_TRANSPORT_V2_READINESS`：不能只改阈值或继续dense BCE；V2必须恢复五帧逐方位融合，以稀疏关系token、圆周软支持、困难负样本和显式跨帧transport/空槽表示persistent/reveal/withdraw。先做模型readiness，不直接训练。
- 论文归因图PNG/PDF/SVG与CSV已保留。run=`results/gate3_semantics/gate3_20260829_gse_axis_anchored_v2_failure_attribution_v1_seed0`；seal SHA=`1cf6ca4c7f8d3c9c4fb3a49526a48a8fd7a688854526802ced9ae6b000f04d0e`，图SHA=`19100bef12934de9eb887d840167982530b7c00d786c0deef7abac5a5070b8f0`。

## 2026-08-29：Axis-Anchored三种子能力训练正式FAIL；停止密集关系场并进入V2机制归因

- 唯一不可覆盖run完成seed 0/1/2各10轮、每seed`12,230`步，总计`36,690`个optimizer steps；正式状态为`FAIL_GSE_AXIS_ANCHORED_EVENT_RELATION_THREE_SEED_TRAINING_V1`，`error=null`、source unchanged，C09/C10/M-TARE/graph/planner读取均为0。
- C07/C08五类event macro-F1=`0.662292/0.633742`，低于预注册`0.737904`；axis误差=`82.7165°/83.0120°`，未达到`<=10°`。三个seed训练轨迹一致，排除单seed偶然性和增加epoch的合理性。
- C07上不存在同时满足precision `>=0.98`与recall `>=0.25`的persistent/reveal/withdraw阈值，因此三个关系通道和current branch field全部拒绝，recall=`0`；combined association在C07仅接受28个正例、recall=`0.004586`，迁移C08后precision=`0.958333`且false merge=`0.041667`。
- 连续几何只部分可保留：width/height/curvature相对规则基线多数改善，但slope在C07/C08分别回归`113.5%/92.6%`，整体验收FAIL。结构拒绝的C07/C08 recall仅`0.027615/0.023882`。
- 代码与历史证据给出两项直接机制：当前axis只消费最后一帧方向特征，而已验证模型用五帧逐方位融合并达到约`5.5°`；reveal/withdraw正样本率仅约`0.013%`，密集平衡BCE把正样本放大约`3,600--3,800×`，造成高召回、海量误报和不可用概率场。
- 决定停止当前dense independent-BCE关系场，不进入图。NEXT只允许零训练V2机制归因：量化五帧方向融合对照、关系稀疏度/可分性、圆周软峰与跨帧transport的必要性；通过后才实现最小V2，复用已验证时序geometry backbone并以稀疏关系峰/显式匹配替换dense threshold field。
- 正式run=`results/gate3_semantics/gate3_20260829_gse_axis_anchored_event_relation_three_seed_training_v1_seed0`；证据清单SHA=`9e9e9f250e35a5fbc354a24dd57153067f7fd22100afc69f67be71b6eadbf5c9`。

## 2026-08-29：三通道关系表示完成全量资格，正式进入训练实现

- 全量预审发现旧四选一关系接口在同一2度方向格无法同时表示不同出口的“出现+消失”；fit/C07/C08分别有`202/20/33`个真实冲突，旧V1/V1R降为表示失败消融。
- V2把persistent/reveal/withdraw改为三个独立二元通道，不改五帧LiDAR输入、事件/几何/descriptor目标或世界划分；模型参数为`819,067`。
- 正式V2逐行重建80个C01--C08世界的`188,126`条关系Teacher，所有真实冲突均无损保留，零同通道冲突；15项unit和19项方法合同全部PASS。
- 正式运行0 optimizer/checkpoint、0 C09/C10/M-TARE/graph/planner，33,083项源seal复核通过，峰值RAM约`1.98 GiB`。
- NEXT：实现并冻结seeds 0/1/2训练Data Card/trainer/evaluator；C01--C06梯度，C07 checkpoint/拒绝阈值选择，C08一次零适配迁移。训练效果未证明前不进入拓扑图。

## 2026-08-29：因果圆周出口集合过程readiness V1R正式PASS

- 769,784参数模型复用causal circular backbone，仅新增1--4 cardinality；180-bin输出归一化为同一集合质量，不再独立阈值筛峰。
- V1唯一失败是`5.96e-8==0`浮点精确比较；V1R以`1e-6`修复，仍严于原`3e-5`合同，其他输入/检查不变。
- correct set NLL显著低于shift/duplicate/ghost，正确count显著低于错误count；wrap和相距2 bins的出口均能同时解码。
- 全量80 worlds/188,126 observations/396,913 exits、真实1--4 batch、旋转、排列、历史和finite backward全部PASS；0训练/test/graph。
- NEXT：冻结一次三seedset-process训练；用连续bearing、cardinality/action、拒绝precision/coverage及完整几何决定是否冻结感知方法。

## 2026-08-29：V2三种子完成但独立局部峰路线停止

- 3 seeds ×10 epochs=`34,110` updates全部完成，best epochs均8，系统、输入隔离和封存正常。
- strict AP从V1的`0.0690/0.0575`提升到C07/C08=`0.1105/0.0972`，但无0.995 precision threshold，节点生成仍不安全。
- axis/width/height/curvature继续泛化；slope约`3.25°/3.34°`失败。
- oracle出口数+top-K NMS的strict命中仍仅`25.9%/24.2%`；只加count head不够，离散peak定位也不稳。
- 决定停止独立local-peak分类，下一接口改为cardinality + continuous circular bearings + permutation-invariant set matching的因果出口集合过程；先做零训练readiness。

## 2026-08-29：柔性角度峰值与困难负样本V2资格正式PASS

- 冻结V1输出在C07/C08精确复现hard-peak AP=`0.069034/0.057519`；真出口最近峰偏差p90均为4个bin（8°），证明网络有角度信号，但单格监督脆弱。
- ±10°匹配AP达到`0.900297/0.898578`，然而top-4+NMS需约20°容差才有足够高精度召回，因此拒绝只改解码或放宽评分。
- 新loss固定为radius-4 circular soft target、focal heatmap和top hard-negative ranking。合成排序、旋转、中心/ghost梯度方向全部正确；3个冻结模型的真实batch均finite backward且明确压制ghost。
- 0训练、0 checkpoint、0新推理、0阈值选择，C09/C10/M-TARE/graph均未读取；17项seal独立验证PASS。
- NEXT：只替换presence loss做V2三seed训练；模型、数据、10 epochs、连续几何头和严格exact-bin评价不变。

## 2026-08-28：180×2 StructuredPolarMultiDepth readiness正式PASS

- 全量readiness精确读取C01--C08 `80` worlds、`188,126` observations、`131,424` tokens、`1,076` identities；fit/selection=`98,279/33,145`，cardinality 0--5逐项一致，C09/C10/M-TARE零读取，source unchanged、`error=null`。
- typed rasterizer把同azimuth事件按距离升序写入2个depth slots；`277`个slot1 tokens全部保留，overflow=`0`，全token最大round-trip误差=`2.93e-5m`，radial fraction最大=`0.9999995`，azimuth residual最大=`0.999981°`。
- 新encoder=`172,430`参数，dense layout=`180×2=360`，top-16不做neighbor NMS；NumPy/Torch support parity=`0`、range/azimuth bound PASS、finite backward、top-k unique、repeat与seed replay逐位相同。
- 20度rotation下dense logits误差=`7.08e-8`、dense xyz=`1.10e-5m`、top-16 xyz=`6.68e-6m`，bin/slot shift完全正确；所有16项检查PASS。11项seal SHA=`34a55cc0728de7a6835545a1732bcc7d135a71b7d3fee67eb6d918adf29a8736`。
- NEXT冻结三seed dense训练Data Card与trainer：C01--C06 direct slot supervision，C07--C08 checkpoint/threshold；与旧exclusive、free-query joint、frozen set和nonlearning同人口比较。未训练前不声称精度收益、不进入graph。

## 2026-08-28：Structured Polar Teacher可行性PASS；固定180×2多深度槽

- 正式零训练审计精确覆盖C01--C08 `80` worlds、`188,126` observations、`131,424` tokens和`1,076` identities；fit/selection=`98,279/33,145`，0推理/阈值选择，C09/C10/M-TARE零读取，source unchanged、`error=null`。
- 同一2度azimuth bin出现`277`对事件，±1 bin内共`801`对；其中同azimuth+同elevation band=`251`对，证明180单槽和180×4单槽都不能无损表示。单帧同azimuth及同azimuth-elevation最大occupancy均=`2`。
- 最小多事件角分离仅`0.00479°`；典型共线事件是近处junction和40m外terminal。全部token方位量化残差`<=0.99999°`，固定34度旋转bin mismatch=`0`，复用readiness证明unsupported=`0`、最小support margin=`2.29e-5m`。
- 决定=`MULTI_DEPTH_POLAR_SLOT_REQUIRED`：主接口固定`180 azimuth bins × 2 radial-depth slots`，同bin按距离排序；禁止neighbor NMS，候选从360 dense slots按confidence top-k。16项seal SHA=`8190da57f2e85ba2ed8beaebf7e6f99beb860d0862b9aa9ff4c381864b5b2505`，图SHA=`a42b42f9fd42ab556ed24807ba6c6b0978266bb34f5cd3e483c002e4b88a08fa`。
- NEXT实现typed rasterizer、dense encoder/head和零训练readiness：全token无overflow重建、旋转等变、range bound、top-k确定性、finite backward和无pose/TNG输入；PASS前不训练。

## 2026-08-28：自由query失败归因PASS；下一表示必须使用结构化polar proposal

- V1在完成全部匹配后因`uint64→bincount`输出类型错误system FAIL，0科学输出，11项seal SHA=`e5dd42844038e1568a4c34f1b2ea029bf630b338c1251b8989cf57b33014ec5e`；V1R只增加显式`int64`转换，数据、预测、阈值、匹配和决策门不变。
- V1R精确覆盖C07--C08 `20` worlds、`45,942` rows、`33,145` tokens和`275` identities；0训练/推理/阈值选择，C09/C10/M-TARE零读取，source unchanged、`error=null`。
- 4m同类型NMS只删除seed0/1/2=`72/71/2`个候选，F1仍约`0.086/0.093/0.069`。全部16 query不看confidence时typed oracle recall仅=`0.122824/0.125147/0.116217`，忽略type也仅=`0.123518/0.128556/0.120984`，三条corrective gate全部失败。
- `29,051/28,884/29,135`个目标没有任何4m内候选；最近同类型候选方位中位误差=`75.18°/64.23°/67.32°`，径向中位误差=`14.99/10.97/15.47m`。0--4m覆盖接近1，4m后立即塌缩，决定=`STRUCTURED_POLAR_PROPOSAL_REQUIRED`。
- V1R 15项seal SHA=`2b9e30d94929be05ec1bbd8753afb86f188af2c3ccefdd41d566e87e9f39d056`，论文诊断图SHA=`58dbbf48a9b0b2100cc8227fce29eb1e4db9761918753ba3808513de4c6bc537`。NEXT先审计Teacher在180个polar bins上的同bin/邻bin冲突和局部support，再实现dense polar objectness+局部残差接口；不直接训练。

## 2026-08-28：GeometryAnchored联合训练正式FAIL；位置可学但事件候选判别失败

- 唯一不可覆盖run完成seed 0/1/2各8轮、`9,096`步，总计`27,288`个全模型optimizer steps；C01--C06/C07--C08固定为`142,184/45,942`行和`98,279/33,145`个tokens，C09/C10/M-TARE零读取，source unchanged、`error=null`。
- 三seed precision/recall/F1分别为`0.0831/0.0889/0.0859`、`0.1389/0.0701/0.0932`、`0.1108/0.0496/0.0685`；macro-F1=`0.0944/0.0888/0.0755`，全部远低于旧互斥center baseline F1/macro-F1=`0.3901/0.4010`。
- 三seed成功匹配后的3D MAE=`2.440/2.231/2.512m`，均通过4m位置门；其余precision、recall、分类型recall、F1提升和multi-event recall门全部失败。结论是free-range锚定保留了一部分位置能力，但16个自由query的稀疏存在性/候选生成机制稳定失效。
- 三个trainer均return 0，统一评估return 2仅表示科学FAIL；31项seal复核零mismatch，SHA=`ded71ce5e58f1dff71bec150e3a6d5a32547e87e54733b71e425dd01b2f40c38`，论文失败图SHA=`600895c5f481a061a3ed7ed94ed82f6faca038b2dd4c611f35614219e896e697`。
- NEXT只读审计候选槽占用、重复/错类型/错位置比例、匹配assignment和置信度分布；不增加epoch、不改阈值、不进入图。根据证据在“结构化polar proposal+局部残差”与“显式cardinality/objectness监督”中只选择一个最小corrective。

## 2026-08-28：GeometryAnchored Spatial Event readiness V2正式PASS

- 全量只读证明精确覆盖C01--C08 `80` worlds、`188,126` unique observations、V2 `131,424` tokens和`1,076` identities；type/cardinality/fit-selection计数全部一致，C09/C10/M-TARE零读取，source unchanged、`error=null`。
- `131,424/131,424`个可观测事件全部位于五帧因果range、±1 polar bin局部包络和固定0.25m LOS margin内；unsupported=`0`，最小support margin=`2.29e-5m`，NumPy/Torch profile误差=`0`。
- `264,134`参数encoder只接受五帧range/valid；真实0--5事件finite backward，query permutation error=`1.19e-7`，18度rotation xyz error=`7.15e-7m`、invariant error=`3.81e-6`，seed replay逐位相同，预测距离严格小于anchor。
- 12项seal逐项零mismatch，SHA=`de4a734e64339d0f73bd3ee415f7337126d323574e26ee7e2f8a8e477349740d`。新encoder已解除接口/Teacher阻塞，但尚未训练，不能声称感知效果提升。
- NEXT冻结joint encoder+set三seedData Card和训练实现：C01--C06提供全部梯度，C07--C08只选checkpoint/threshold；主比较仍是旧互斥center、旧冻结encoder set和非学习几何。

## 2026-08-28：Observable Spatial Event Teacher V2正式PASS

- 80个C01--C08 Teacher shards全部由封存V1确定性转换；`188,126/188,126`条观察和`1,076/1,076`个结构identity保留，0训练/推理/归一化/阈值选择，C09/C10/M-TARE零读取，源文件未变化。
- 固定LiDAR垂直FOV `[-15°,15°]`精确删除`1,631`个不可观测token（terminal/junction=`75/1,556`），保留`131,424`个（`30,714/100,710`）；fit/selection=`98,279/33,145`。
- 新cardinality 0--5=`82,326/82,183/21,756/1,724/128/9`，所有零事件行保留；1,631行删除原因、原slot、identity、距离和仰俯角完整保存，不按模型误差选样本。
- 1,298项seal逐项零mismatch，SHA=`736261034852c779d26e2619735755ce1bbd8bc7a415dbc68257923e7cfa5772`；保留论文图PNG SHA=`1648cc4b0a291eaec27b7336872a5d6ba4d170b7597d26b1ecbfd806e5e0a0f6`。
- 治理状态返回Gate 3。NEXT把encoder anchor冻结为五帧局部free-range envelope+既有0.25m LOS margin，并在V2 Teacher上重做一次零训练readiness；PASS前不训练。

## 2026-08-28：GeometryAnchored encoder V1接口通过，但发现Teacher超出LiDAR垂直视场

- 唯一正式零训练run精确读取C01--C08 `80` worlds、`188,126` observations、`133,055` tokens和`1,076` identities；集合基数/类型/划分全部复现，C09/C10/M-TARE零读取，输入未变化，`error=null`。
- 新encoder固定`264,134`参数，只接受五帧`range_m/valid_mask`。真实0--5事件batch finite backward，query permutation error=`0`，18度旋转xyz误差=`4.77e-7m`、不变量误差=`1.91e-6`，seed replay逐位相同，预测距离严格不超过扫描anchor；模型接口本身全部PASS。
- 唯一FAIL项是Teacher/传感器可观测性：`641/133,055`标签超出当前polar-cell free range。后续只读归因确认采用Teacher固定`0.25m` LOS margin后仍有`639`项，其中`619`项目标仰俯角已超出16线LiDAR固定`[-15°,15°]`视场；五帧包络仍有`577`个视场外失败。
- 决定把问题分类为`TEACHER_SENSOR_VERTICAL_FOV_MISMATCH`，停止joint training。V1保持不可修改，12项seal SHA=`39610c5fb4317da031614e52760ea1aa86ccc893fb5664d4fb79c33b541be5ce`。
- NEXT生成Teacher V2：只按冻结LiDAR垂直视场删除`1,631`个不可观测tokens，不删任何frame，仍保留全部`1,076` identities；V2固定为`131,424` tokens（terminal/junction=`30,714/100,710`），fit/selection=`98,279/33,145`，cardinality 0--5=`82,326/82,183/21,756/1,724/128/9`。encoder anchor改用五帧局部free-range envelope与原0.25m LOS margin；重新readiness PASS前禁止训练。

## 2026-08-28：空间事件集合失败归因V1R正式PASS；必须训练坐标感知空间encoder

- 只读审计使用封存C07--C08 `45,942`行、`33,563`个目标和`6,153`个多事件帧；0训练、0推理、0阈值选择，C09/C10/M-TARE零读取。
- 固定4m同类型NMS只删除seed0/1/2=`0/28/260`个候选，占`0/0.1065%/0.9623%`；F1几乎不变，正式排除duplicate-query主因。
- 忽略置信度、允许全部16 query参与时，typed 4m oracle recall仅=`0.199684/0.255102/0.212764`，全部低于预注册`exclusive recall 0.262849 + 0.10`门；position-only oracle也仅=`0.253195/0.305336/0.277657`。
- 距离分层显示0--4m实际recall仍有`0.537/0.403/0.562`，32--40m仅`0.005/0.008/0.012`，40--50m近零。决定=`ENCODER_SPATIAL_REPRESENTATION_INSUFFICIENT_NEW_ENCODER_OBJECTIVE_REQUIRED`。
- V1科学结论正确但图顺序/标题和Python Infinity序列化不适合作论文证据；V1保持不可修改。V1R只修严格JSON与图，结论逐项相同。V1R 16项seal SHA=`2cf2bd47a0607ea5c22f5b563d8c99b49f45a57cb15c765ae06c64de553343aa`，论文图SHA=`1a798eea42ace02aaec1c70a3ef60b0e8c9c5d352044f05b77cf07cfc35a3ef3`。
- NEXT实现零训练`GeometryAnchoredSpatialEventEncoder` readiness：显式角度/高度坐标、五帧自由距离径向锚点、集合token与anchor-relative深度；验证旋转等变、range-bound、空/多事件、有限梯度和无pose/TNG输入后，才允许joint encoder+set训练Data Card。

## 2026-08-28：冻结旧编码器的空间事件集合容量正式FAIL

- 唯一正式run完成三seed，每个seed只训练`93,638`参数decoder `4,448`步，总计`13,344`步；三个五帧encoder均为0 optimizer step。C01--C06/C07--C08固定为`142,184/45,942`行，正式进程C09/C10/M-TARE零读取，输入前后哈希一致，`error=null`。
- 同类型且3D误差`<=4m`时，三seed precision/recall/F1分别为`0.142821/0.119209/0.129951`、`0.211947/0.166076/0.186228`、`0.161066/0.129667/0.143671`；multi-event recall=`0.103525/0.154335/0.117021`，全部未达到安全门。
- 成功匹配事件的定位MAE为`2.578/2.491/2.633m`，说明局部位置回归并非完全无效；主要失败是大量无法在4m内对应Teacher的重复/错位候选，而不是程序或数值故障。
- 旧互斥事件+单中心baseline为P/R/F1=`0.765334/0.262849/0.391306`、匹配定位MAE=`1.737m`，明显优于集合头；非学习解析几何baseline F1=`0.003171`。因此“学习结构”仍优于纯规则，但“冻结旧encoder只外挂set head”被正式否定。
- 42项seal复核零mismatch，seal SHA=`bdcbbf1552c6f592376d4f85912f18c90ddb6eea28568a0db1c7bdfafc404322`；论文图SHA=`fa340e856e59feece57db0e48b210ce9f9ad1f4726d3ee75722fabe7b26c8ef1`。已删除`26,161,231,418`字节可再生缓存，保留三模型、基线、曲线和图。
- NEXT只读做距离/基数/类型/重复query失败归因与去重上界；不重试调loss/阈值/query数、不解冻encoder、不进入图回放。先判断下一版应加入集合排斥/基数校准，还是训练真正保留远程空间结构的encoder。

## 2026-08-28：16-query SpatialEventSetDecoder readiness V1R PASS

- V1唯一失败为query编号参与assignment平局打破，permutation loss error=`2.9373e-4`；数据、旋转和finite backward均PASS。V1永久保留。
- V1R按预测内容规范排序后匹配，误差降至`1.1921e-7<=1e-6`；旋转误差=`8.2970e-5m<=5e-4m`。
- 真实80 shards、188,126 rows、133,055 tokens及0--5事件基数全部对齐；decoder=`93,638`参数、16 queries，finite backward。
- 10项seal零mismatch，SHA=`a9625fa3eac099fa2a63a259f55e98f5200aacc49df60bc921e9571f76a62c78`；0训练/推理/C09/C10/M-TARE。
- NEXT冻结三seeddecoder-only capacity Data Card：对应冻结encoder，C01--C06 fit、C07--C08 selection，4m同类型set matching，和旧互斥单中心/非学习几何baseline比较。

## 2026-08-28：16槽空间事件set Teacher正式导出PASS

- 80个C01--C08 Teacher shards、188,126条唯一因果观测和1,076个Teacher-only identities全部生成；逐世界candidate/visible/blocked/cardinality与LOS feasibility逐项相同。
- token总数=`133,055`，terminal/junction=`30,789/102,266`；零/单/多事件行=`81,069/83,093/23,964`，最大基数5，固定16槽零截断。
- 学生输入未导出，identity/TNG只留在Teacher；0训练、推理、normalization、threshold selection，C09/C10/M-TARE零读取。
- 1,634项seal零mismatch，seal SHA=`29d821e2c9935a7e9f24dc9aa56ed14c01131a397c922c5f6f47330c8caef930`；论文图SHA=`4f7278394a89dc28827adcba55b88efe6acb2e04c5a7b6eca77ed43c82e9d343`。
- 状态返回Gate 3。NEXT实现16-query方向等变set decoder、确定性匹配和空集/多事件loss readiness，测试通过后才冻结训练Data Card。

## 2026-08-28：空间多事件Teacher全量LOS可行性正式PASS

- 首次V1因JSON根类型接口错误在0 world/0 raycast处system FAIL并封存；V1R只修列表解析，科学合同完全不变。
- V1R精确覆盖C01--C08 `80` worlds、`16,078` directed traversals、`188,126` causal observations。50m候选/可见/遮挡pairs=`326,974/133,055/193,919`。
- `1,076/1,076`个terminal/junction objective identities全部可见且全部有反向heading观测；最大单帧事件数=`5<=16`，唯一性与泄漏门全部PASS。
- fit多事件/terminal-junction共现帧=`17,811/10,052`，selection=`6,153/3,282`；10个family在两split均有共现。四个稀有行objective terminal全覆盖，row `44299/110361`同帧同时见terminal+junction。
- 19项seal零mismatch，seal SHA=`c557e1d184697b57e7e3a103e32581ce5df43a319da5a129698b419d3d4117c2`；论文图SHA=`5bf7fe520446d780aa98781cfffc66af2c310f42d10f031b7ddb57988aa418b7`。0训练/推理/完整Teacher导出，C09/C10/M-TARE零读取。
- NEXT冻结空间事件set Teacher导出Data Card：固定16槽，不删零事件帧，只输出类型、robot-relative xyz、mask和Teacher-only identity；导出通过前不设计/训练新head。

## 2026-08-28：route token失败归因正式PASS；主接口转为空间多事件集合

- 正式只读run精确连接`188,126`条C01--C08观测、98个关系端点、40个含关系世界、2个selection低支持terminal及其4条Teacher行；0训练、0模型推理、0阈值选择、C09/C10/M-TARE零读取。
- 98个关系端点中96个与相反事件类型直接相连且欧氏距离`≤50m`；48个terminal端点全部在这个量程尺度内直接连接junction。这是潜在场景共现条件，不冒充native-mesh LOS；下一Teacher审计必须补LOS。
- `node_0012`三个seed结构mass均`>0.925`且稳定选择junction，同时真实terminal与degree-3 junction仅约`13.484m`直连，归因为`exclusive_teacher_conflict`。`node_0040`三个seed route-stop score均`>0.736`，但结构mass约`0.967/0.106/0.016`，归因为`seed_representation_instability`。
- 两个失败均唯一归因，预注册5项门全部PASS；推荐`spatial_multi_event_teacher_before_new_head`。不再训练互斥terminal/junction分类器，改为允许同一LiDAR序列输出多个带相对三维位置的事件token。
- 19项seal零mismatch，seal文件SHA=`352412794a5c5de684b983ffc92090f1e5d1e09ec84b331fca6a3e0d89241aa2`；论文图SHA=`1d136211db2c81189fcb7d6f32f68282e149cf33066ece26856e7292e229c11c`。NEXT先做C01--C08多事件Teacher LOS/基数/唯一性/覆盖可行性审计，不生成训练资产、不读C09/C10。

## 2026-08-28：route-conditioned最小学习头科学FAIL；总体事件略好但关键端点和图召回未改善

- 唯一正式run按冻结Data Card完成三seed各200步，共600个643参数residual optimizer steps；原ActionSetNodeDetector保持0步更新。C01--C06/C07--C08仍为`142,184/45,942`条观测、`72/26`个关系端点，C09/C10/M-TARE零读取。
- 固定0.97 ensemble把正确decision episodes从`928`提高到`931`，predicted triggers从`1,022`降到`1,021`，false triggers从`94`降到`90`；高支持和全部端点保持`21/23`与`22/26`，说明route-flow不是完全无效。
- 预注册核心门仍失败：两个selection低支持端点保持`0/2`，没有恢复至少一个关键关系端点。完整图进一步把selection正确节点从既有`193`降到`185`，node recall由`0.704380`降至`0.675182`；edge仍为`4/4`正确、precision=`1`、false loop=`0`。
- 结论是低维汇总route-flow可以改善普通episode，却不足以决定稀有结构节点；停止扩大该小head、增加步数或调0.97阈值。它与AUC审计一并保留为论文失败消融，不能写成主方法收益。
- 44项seal逐项复核零mismatch，seal文件SHA=`5130ca94d266288069c6003e18d930a7f9f0d38f62dc8d2d541fe27cd7d74583`；论文失败图SHA=`14d5d3d5a4de6f77246883543c28184fb31e8c36a6d900d43671398cdae2f69e`。运行外壳结束后误读重写的command文件，但追加命令均在导入或已存在目录检查处停止，seal复核证明未改动正式证据。
- 历史12帧causal detector已在C09以false trigger `2.464%>1%`失败，禁止作为“新方案”重复。NEXT只读审计低支持端点在训练前后每seed的structural/conditional residual、16维route-flow与原始六出口token，判断失败来自信息汇总损失还是objective事件Teacher与可观测动作语义不一致；之后只允许token-level关系头或Teacher语义修正二选一。

## 2026-08-28：route-conditioned exit-flow审计正式PASS；新event head获得直接容量证据

- 固定公式为`mean_seed((backward-forward)/(forward+backward+lateral))`，每个objective endpoint按Teacher窗口取最大值；公式在运行前冻结，0 learned weights、0 threshold selection、0模型推理/训练。
- C01--C06的72个关系端点中terminal/junction各36，identity-level AUC=`0.949074`；C07--C08的26个端点中terminal/junction=`12/14`，AUC=`0.934524`。terminal均值在两split均显著高于junction，全部`AUC>=0.80`门通过。
- 两个selection低支持terminal得分=`0.2750/0.7648`；后者高于fit junction中位数`0.3741`，正是旧模型structure mass仅`0.3632`且完全漏检的端点。因而固定route feature至少具备恢复一个低支持端点的开发容量。
- 正式run覆盖`188,126`条观测和98个关系端点，18项seal逐项验证PASS，seal SHA=`6b2c5709dd75f71dfa862b8c1efd1ec9332988548391a751422b8ed27fd3c660`；论文图PNG SHA=`f5c8b20a9faf7e0988040de39a6ad027532e9e4a333b6a376dbb98a80e275636`。
- 已实现零初始化`16→32→3` route-conditioned event residual，共643参数；输入为五帧forward/backward/lateral/stop flow的current/mean/std/change，输出1个structural与2个conditional residual。启用前逐元素复现旧logits，严格过去引用和token排列不变性通过；相关18项测试PASS。NEXT接入三seed trainer、固定0.97评估与完整图回放后冻结Data Card/spec；C09/C10/M-TARE继续禁止。

## 2026-08-28：低支持端点可观测性审计PASS；主因转为route-agnostic语义混叠

- 正式审计对C01--C08 `188,126`条观测执行三个冻结seed共`564,378`次前向，0 optimizer/model update/threshold selection，C09/C10/M-TARE零读取。98个relation endpoints、fit/selection低支持`10/2`和selection四个低支持行精确覆盖。
- 低支持机制为fit `3 correct + 0 wrong-event + 7 no-proposal`，selection `0 + 1 + 1`。selection的`node_0012`三行structure mass最高`0.9887`，但terminal conditional仅约`0.0003`；`node_0040`唯一行structure mass=`0.3632`。
- 冻结context/物理几何5-NN显示`node_0012`几乎全部近邻是junction，`node_0040`的物理近邻全部是corridor；因此不接受审计器的机械“直接修conditional head”建议。强制类别头会把局部可见几何相同的样本硬分开。
- 源TNG复核发现两terminal分别经`13.484 m/11.659 m`短边直接连接degree `3/4` junction，而LiDAR量程50m；到达dead-end仍能看到身后路口。当前无方向出口集合把已走过的身后分支当成当前junction证据。
- 原始learned token进一步显示`node_0040`三个seed forward confidence mass均为0、backward约`1.55--1.74`；`node_0012`也普遍backward高于forward。NEXT正式审计route-conditioned exit-flow score在全部98个关系端点上的identity-level terminal/junction可分性；通过后才实现显式incoming/outgoing关系头。
- 19项seal逐项验证PASS，seal SHA=`585c14105215c973365419726c59c926af3a5aff92bbca33e6f2e4aed9bb882f`；论文图PNG SHA=`d13014e805e4166c1473abdebf10c34a6a83c0e66ceb1a59912d54b8972fe83e`。

## 2026-08-28：identity均衡结构残差科学FAIL；低支持端点是两种不同失败机制

- 新增零初始化`Linear(128,1)`结构logit残差、identity-mean-of-episode损失、三seed trainer、固定阈值评估和完整图回放；相关10项测试PASS。正式数据仍为C01--C06 `142,184`条fit观测、C07--C08 `45,942`条selection观测、fit/selection关系端点`72/26`，C09/C10/M-TARE零读取。
- V1数据卡因治理字段缩写被preflight拒绝，未创建run；V1R在训练前被旧run-id硬编码拦截，0推理/训练；V1R2完成冻结context提取后因3类动作概率未嵌入5类结构接口而system FAIL，0 optimizer step。两处接口均保留独立失败证据，未复用partial输出。
- V1R3三seed各执行200步，共600个129参数residual optimizer steps，原ActionSetNodeDetector optimizer steps为0。每个seed的安全checkpoint都选择step 0：训练后的候选无法恢复低支持端点且会开始损失原正确episode。
- 固定0.97 ensemble结果与冻结baseline完全相同：低支持端点`0/2`、高支持`21/23`、全部端点`22/26`、正确decision episodes `928/1,136`、false triggers `94`。唯一失败门是“至少恢复1个低支持端点”；其余安全门全部PASS。
- 完整开发图仍安全但无增益：C07--C08为`193/193`正确节点、`4/4`正确边，node/edge P=`1/1`、R=`0.704380/0.307692`、false loop=`0`；fit为`596/596`节点和`13/13`边。44项seal逐项验证PASS，seal SHA=`7dc32bdead2ae6cd0f6057f6b2fb38f4e5fea3bc012a960fe89adb78ae8851b2`，论文失败图PNG SHA=`e5484af4f30f65fbe6eca5b42e2c8bb0c6e462c16dab00e2079b1b1ff3d3f4c4`。
- 两个selection低支持terminal的冻结概率揭示不同机制：`S04...node_0012`三行结构mass=`0.9558/0.9887/0.9864`，已有proposal但被错分为junction；`S07...node_0040`只有一行且结构mass=`0.3632`，完全缺proposal。仅调structural logit不可能纠正前者的事件类别，也没有证据证明后者在现有5帧表示中可分。
- NEXT冻结一次C01--C08只读低支持端点可观测性审计：分别检查conditional event margin、structural margin、六出口token/几何profile、五帧变化和同类困难负样本。审计后只允许选择“事件类别表示修正”或“时间/几何token修正”之一；禁止增加训练步、调0.97阈值或读取C09/C10。

## 2026-08-28：episode曝光审计PASS；corrective改为identity级均衡而非行过采样

- 对现有ActionSetNode cache和`EpisodePreservingBatchSampler`做正式只读审计，确认原loss已按连续decision episode做MIL，不能把问题错误描述为纯row-balanced loss。
- 98个relation endpoints对应fit `286/3,282`个endpoint/all decision episodes，selection为`110/1,136`。fit低支持10个identity每个严格1 episode，高支持36个每个6--8个、均值6.2222；selection同样为1对6--8、均值6.2857。
- 三项预注册门全部PASS：fit低支持全为1 episode、高/低identity曝光比`6.2222≥6`、selection不反转。0训练/推理/C09/C10/M-TARE，18项seal逐项验证PASS。
- corrective合同冻结为：保留原negative loss与all-episode MIL；新增`mean_identity(mean_identity_episodes(episode_MIL))`。禁止重复稀有行、改变0.97阈值或把Teacher identity作为推理输入。
- 正式seal SHA=`671b949aad8f380f20b5000b992cd0538fd7c27f1d2cbbe3cd7efc676e8c1d29`；论文图PNG SHA=`d3e7fdb139dcb885974d70f2eeca300f7df091f44bcf359afb359391342c43a2`。
- NEXT实现零初始化128→1 structural-logit residual：冻结原ActionSetNodeDetector及junction/terminal conditional head，仅用C01--C06训练identity均衡项；C07--C08要求低支持proposal恢复且false trigger、普通episode及最终图安全不回归。

## 2026-08-28：开发端点支持审计PASS；端点均衡corrective获得训练依据

- 正式只读审计复现C01--C06 `142,184`条观测、36条relation、72个不同端点、3,037个proposal和718个commit；C07--C08为`45,942`、13、26、1,022和234。0训练/推理/C09/C10/M-TARE。
- fit端点支持分层：`1--3`条Teacher行的10个端点proposal/commit/final recall=`0.30/0.20/0.20`；`11--30`条的13个端点为`1/1/1`；`31+`的36个端点为`1/0.8611/0.6111`。
- selection端点支持分层：`1--3`条的2个端点三项recall均为0；`11--30`条的9个端点proposal/final=`0.8889/0.6667`，`31+`的14个为`0.9286/0.5714`。
- 预注册判据全部通过：fit低支持数量`10≥8`；高支持减低支持proposal recall=`0.70≥0.10`；selection方向不反转且差=`0.9130`。因此允许一次identity/relation-endpoint-balanced corrective，不允许改变0.97部署阈值、图规则或安全门。
- 关系失败同时复现：fit=`12 proposal + 7 commit + 4 qualification + 13 recovered`；selection=`3 proposal + 1 event + 5 commit + 4 recovered`。训练首先解决proposal支持偏斜，commit/qualification保留为后续独立问题，禁止一个loss混改全部机制。
- 19项seal逐项验证PASS，seal SHA=`276cae555f04f02faf40d70f1b903d765bad03caa55fcfe662f713be891b6e35`；论文图PNG SHA=`2d56ae73794398dd0d0a8bfde272049012511973622341c0b541378e8f37396d`。
- NEXT审计现有ActionSetNodeDetector训练接口，冻结原row-balanced batch/loss，再增加train-only relation-endpoint identity-balanced batch；checkpoint选择只看C07--C08低支持proposal恢复与全局安全不回归。

## 2026-08-28：C09关系端点漏斗PASS；下一修正锁定为端点均衡学习/提交

- V1因把包含相同`24,462`个唯一global IDs但顺序不同的Teacher文件与冻结action archive要求逐行同序而system FAIL，0科学归因；V1R改为按`global_sequence_index`一对一canonical join，样本集合、模型、图和阈值均未改变。
- V1R精确复现`1,646`次proposal、`442`个raw hypotheses、`114`个committed hypotheses、`74`个qualified/final nodes、`6`条raw edges、`1`条final edge，并为8条relation的16个端点给出唯一因果阶段。0推理、0训练、0阈值选择、0 C10/M-TARE。
- 端点漏斗为`16 objective → 11 triggered/correct-event → 8 committed → 5 qualified/final`；互斥失败为`5 proposal missing + 3 association/commit missing + 3 qualification reject + 5 recovered`。
- 关系漏斗为`4 endpoint proposal missing + 2 endpoint not committed + 1 endpoint qualification reject + 1 recovered`。6条raw edge中1条是真实relation，另5条均连接同一objective identity的重复hypothesis；edge assembler不是首要阻塞。
- 5个无proposal端点中4个terminal只有`1--3`条Teacher行，另1个junction有45条Teacher行仍完全漏检；下一步不能按C09个例调参，而是在C01--C08只读审计relation-endpoint监督支持与当前proposal/commit召回，判断是否采用identity/endpoint-balanced学习。
- V1R 21项seal逐项验证PASS，seal SHA=`03e9740454977cbb3b4e30c833ef543a81a92d39af9176f2d6eaaf3d91f5b774`；论文图PNG SHA=`c69c51a934d9acf6c69a09c0b5244e80155e0fa22fdea3539db1cea32ccbb344`。

## 2026-08-28：C09冻结验证科学FAIL；缺口锁定为关系端点覆盖

- 唯一正式run使用C09的`10`个世界、`32,678`个唯一LiDAR帧、`24,462`条五帧因果观测、`2,054`条directed traversals和`1,027`条physical edges；图构建完成并冻结后才在独立进程读取Teacher评分。0训练、0 checkpoint/阈值选择、0 C10/M-TARE读取。
- 冻结方法输出`74/74`个正确结构节点，node P/R=`1.0/0.569231`；输出`1/1`条正确执行边，edge P/R=`1.0/0.125`；false loop merge=`0`。节点安全与召回通过，唯一失败项是edge recall低于固定`0.25`门槛，因此C09正式结论为科学FAIL。
- 只读复核表明原始回放仅形成6条边：1条是真实relation且被最终保留，其余5条分别为同一objective节点的重复/自关系或受错误事件污染，并非被端点资格规则误删的真边。8条objective relation中仅1条两端均恢复，3条只恢复一端，4条两端均未恢复；16个relation endpoints最终只恢复5个。
- 当前阻塞因此不是edge assembler或endpoint filter，而是少量关系关键端点在proposal、association/commit或qualification阶段未稳定进入图。禁止降低edge recall门、修改4m半径、用C09调参或读取C10。
- 正式seal SHA=`d790407c7ad8f325b6c9173e2e9377cd23fbccf861a49fe5996c5649e6f1d2ad`；论文失败图PNG SHA=`0cc0b074e2dccaf479ec6c4649a48e5da004e6724519036eab81f205f40d9a68`。
- NEXT只读追踪8条relation的16个端点，逐端点标注`未proposal / 未commit / qualification拒绝 / final匹配`，逐关系标注`raw edge / final edge`；据此判断下一轮应修模型触发、节点关联还是资格规则。该审计不训练、不适配C09。

## 2026-08-28：执行端点几何开发容量正式PASS；节点/边安全缺口关闭

- 唯一正式run读取C01--C08 `188,126`条因果观测和`252,430`个1m route frames；`16,078`条declared traversals中`16,076`条有帧，已知1.516m短边的正反2条zero-sequence traversal保留且未伪造endpoint。
- 方法冻结为：旧verifier/三seed metric union在线关联保持4m；提交后用学习event forward选择physical edge side；执行轨迹endpoint anchors最大spread必须`≤2×1m`，同edge两侧冲突立即拒绝；两edge partial junction仅在outward directions点积`>0`时有branch witness，三edge直接见证；0 angle/distance grid。
- C01--C06输出`598/598`正确节点和`13/13`正确边，node P/R=`1/0.755051`、edge P/R=`1/0.361111`；C07--C08输出`193/193`与`4/4`，node P/R=`1/0.704380`、edge P/R=`1/0.307692`。全部固定安全/召回门通过。
- C07--C08已用于机制确认但未选择numeric threshold，因此该结果只称development capacity，不称unseen validation。C09历史上已用于旧感知/旧图validation，也不能称全项目untouched；下一步只把冻结新机制在C09做validation，严格未见测试仍为C10。
- 22项seal逐项复核PASS，seal SHA=`39c583292d6e427207a37b2c6c65ea5895894203175bc0a2c27f73b2ecfc063e`；论文图PNG SHA=`4e62e279869897603f2531cda66c19b020900175b0b760c4ff1e14237e776b32`。0 optimizer/inference/C09/C10/M-TARE。
- NEXT冻结C09 endpoint-geometry validation接口：复用已有冻结三seed模型和C09 causal sensor/Teacher，只新增必要推理，不训练、不选阈值；若C09安全失败，停止并报告，不读C10。

## 2026-08-28：执行端点合并inventory PASS；节点安全已过，剩余阻塞精确收缩为3条fit错误边

- 正式run只读`188,126`条C01--C08因果观测和`16,078`条directed traversal；`2,933/1,013`个fit/selection方向不变edge-endpoint tokens对objective identity均为零歧义。去掉endpoint side后selection出现1个physical-edge alias，证明端点角色是必要字段。
- support-2 endpoint候选为fit `9 positive/0 negative`、selection `5/0`；没有困难负例，因此明确禁止训练duplicate verifier。执行端点identity作为确定性topological invariant保留。
- fit-only factorized hybrid在两edge junction上要求至少一个旧语义verifier内部pair，再做endpoint union：fit node P/R=`0.980510/0.825758`，selection=`1/0.773723`，节点门全部通过；selection edge P/R=`1/0.307692`，恢复`4/13`条relation并通过。
- 完整图仍FAIL：fit恢复`15/36`条relation但输出18条edge，其中3条错误，edge P=`0.833333<0.98`。这不是selection退化或duplicate问题；下一机制只允许审计3个edge instance的两端事件一致性、endpoint token、exit-token与place证据。
- 26项seal逐项验证PASS，seal SHA=`1c65d49e88944f12770d0f0761f0593dc51eec734e198f877b7c811e606a2bf4`；论文图PNG SHA=`ad5086edb2b27703daa443110ebeb86ad2e9b42467d9c987d5735a09cf1eb68e`。C09/C10/M-TARE仍冻结。

## 2026-08-28：部分incidence容量审计PASS；错误主因是重复节点而非假结构事件

- 正式只读inventory使用C01--C06拟合世界和C07--C08选择世界的冻结因果replay，检查`29,621`个既有候选pair、`3,037/1,022`个proposal和`718/234`个已提交hypothesis；0训练、0推理、0 C09/C10/M-TARE读取。
- `old verifier OR 三个空间seed均在4m内`可恢复`5,595`个旧关联器拒绝的metric pair；不改变4m半径。仅保留至少3条不同physical edge支撑时，fit图node P/R=`0.998264/0.726010`、edge P/R=`1/0.333333`，全部安全门通过；selection为node P/R=`1/0.682482`、edge P/R=`1/0.230769`，只差1条relation。
- 两条physical edge支撑的junction在fit有`84正/22负`、selection有`28正/7负`，Teacher数量和family/stratum覆盖通过预注册inventory门。正式run 20项seal复核PASS，seal SHA=`ff97aa4c537d7a2fc69f76091e5673cee1ba2f84351ca0702080303ef4c2a27c`；论文图SHA=`96d68f48d982ee317cc017a278229cfb821023d16a637d808dc9e85f2d4018b4`。
- 后续objective分解发现这些“负例”并非统一的假结构：fit 22个中15个是同一真实节点的重复提交、5个是中心超4m/一对一分配遗漏，真正假事件仅2个；selection 7个中5个是重复、2个是中心遗漏、0个假事件。禁止用该标签训练generic junction reliability head。
- NEXT构造提交后hypothesis级duplicate inventory：同world/同event，候选由共享physical incident edge产生，正例为同一objective结构，负例为不同结构；审计候选数量、困难负例、family覆盖及descriptor/exit-token一致性。若Teacher充分，才训练或冻结fail-closed duplicate verifier；edge assembler、4m在线关联和C09/C10/M-TARE继续冻结。

## 2026-08-28：客观三维图评分PASS；旧评分略高估而非低估，主阻塞重定为节点可靠性与端点恢复

- 评分器使用每个已提交hypothesis的冻结预测中心均值，在同world/同event内与274个objective TNG节点做`4.0 m`最大基数、最小距离一对一匹配；重复预测和未匹配预测仍为FP，edge只能经该映射计分，Teacher在replay完成后才读取。
- V1 proposal仅因缺少preflight顶层治理字段而未创建run；V1R在执行器前把stable global IDs误当稠密row IDs而system FAIL，0科学结论，13项seal。V1R2改为四份来源global ID逐元素一致，preflight 0 error后唯一执行并正式PASS。
- scalar-center图客观node P/R/F1=`0.854749/0.558394/0.675497`，edge P/R/F1=`0.25/0.076923/0.117647`，macro=`0.396572`；spatial-center图为node=`0.951691/0.718978/0.819127`、edge=`1/0.153846/0.266667`、macro=`0.542897`。
- 相对scalar图，spatial图客观node F1 `+0.143630`、edge F1 `+0.149020`、macro `+0.146325`；学习几何改善建图的核心证据加强。但spatial仍FAIL node precision `0.951691<0.98`和edge recall `0.153846<0.25`，禁止进入C09/C10/M-TARE。
- 207个spatial committed nodes中197个一对一正确、10个未匹配；其中6个对应真实identity但中心均值超4m、2个是一对一重复、2个为turn/corridor假结构。13条真实relation中只有2条两端都匹配，且这2条全部被edge assembler恢复；缺边仍完全由端点节点未恢复造成。
- 22/22 seal逐项复核PASS，seal SHA=`3af74b2d53344acefa0097759f9774f5203c9b46d74a4cc806882b2ccb1193f7`；论文图PNG SHA=`f4fcce168536ef9b86111c28bfced5cc05dba2329eed5483fe9feca27a80f805`，PDF/SVG/CSV/JSONL provenance均保留。
- NEXT先按新的objective-spatial scorer重做fit/selection causal funnel，分别量化假触发、重复提交、中心超界、关联拒绝和relation端点损失；在C01--C06上证明“3-seed metric一致性恢复 + uncertainty/event可靠性拒绝 + duplicate consolidation”是否存在安全容量，冻结后才允许C07--C08复核。不得训练于稀少runtime negatives，也不得改4m半径或edge assembler。

## 2026-08-28：trace-commit失败漏斗PASS；主阻塞转为metric-aware关联

- V1因把单一非decision identity误算为正确decision identity而在scorer复现处system FAIL，未写归因资产；13项seal SHA=`1a8218a68a8d7ec43568e46a285f2a925390314405637bd91a7204eeb132dd44`。
- V1R明确复现`198 correct unique + 6 duplicate excess + 2 empty identity + 1 non-decision identity = 207 committed`后正式PASS；274个identity、13条relation全部得到唯一因果stage，0 optimizer/model/C09/C10/M-TARE，22项seal SHA=`549a2b99a764da2bd898987ecfc264a943a79622c1cd73a4262f7630635e7643`。
- node funnel：`274 true → 258 proposed → 250 ≥2 traces → 238 within 4m → 207 accepted-pair-supported → 198 correct unique committed`。76个未恢复identity分解为proposal missing 16、single trace 7、outside-4m 12、association verifier reject 31、ambiguity/commit 10；最大阻塞是旧verifier。
- edge funnel：13条真实relation中3条endpoint proposal missing、8条endpoint uncommitted、2条recovered；`no common committed trace=0`、`edge assembly miss=0`，禁止继续改edge组装器。
- 论文图 `gse_trace_commit_failure_funnel.png` SHA=`3d457d7da6f7415b7d73c00efed748044fbd632e364aa0ac7cb927a202b76ff2` 已保留。
- NEXT先做C01--C08只读runtime metric-pair Teacher inventory：把新3D中心距离/seed不确定性变成显式部署特征，确认C01--C06同world不同identity的4m困难负样本数量与family覆盖，再决定一次metric-aware association训练。C09/C10/M-TARE继续禁止。

## 2026-08-28：空间中心显著改善离线图但未过安全门；转只读失败漏斗审计

- 图资格V1/V1R均在graph replay前被system checker拦截：V1是float32 mean/project线性恒等检查过严（实测`4.069e-5 m`）；V1R是不同GPU batch packing不可能bit-exact（最坏`1.788e-4 m`）。两次均0 graph conclusion、0 optimizer/C09/C10/M-TARE，seal SHA分别=`b41539a7fa0098fb7b73eb07cdd1f3d3a4b470539878213e2b4ef17cd430a198`与`ca6a41dfa315946e556e6e504f03c3d72fb74cc176b9516245dba3fb6c27e26a`。
- V1R2固定全`188,126`行global-order batch128并用`0.25 mm`复现界，三seed/ensemble全部复现；正式graph replay完成且`error=null`、source unchanged、0 optimizer/C09/C10/M-TARE，38项seal SHA=`47a92caa1c5c95881f4eafd031d2dae899cb9ba4b6b20ca5c59336a2ee486848`。
- C07--C08相对旧scalar-center图：node F1 `0.715232→0.823285`、正确唯一节点`162→198`，edge F1 `0.117647→0.266667`、edge precision=`1.0`，node-edge macro-F1 `0.416439→0.544976`（`+0.128536`）。相对即时ghost/规则baseline增益=`0.090714`，已过`0.05`门。
- 整体仍科学FAIL：node precision=`0.956522<0.98`、edge recall=`0.153846<0.25`；fit也未满足安全门。学习几何已明确改善建图，因此不停止GSE主线，但不允许调阈值/4m半径/commit规则。
- NEXT对V1R2 sealed replay做一次只读failure funnel：逐identity区分proposal miss、cross-traversal association split、commit support不足和edge assembly miss，并生成论文级前后对比图；C09/C10/M-TARE继续禁止。

## 2026-08-28：纵向结构定位纠正PASS；恢复完整离线拓扑审计

- 唯一正式run `gate3_20260828_gse_spatial_longitudinal_corrective_v1_seed0` 在80个C01--C08开发世界完成三seed训练：每seed `3,960`步、合计`11,880`步；冻结空间decoder/backbone均0步，C09/C10/M-TARE读取均为0，`error=null`。
- C07--C08集成结构中心relative error从`4.641362 m`降至`2.668592 m`（改善`42.50%`），within-4m从`0.523952`升至`0.802866`（增益`0.278915`），global center MAE从`3.752201 m`降至`2.221275 m`；七项预注册gate全部PASS。
- lateral/up MAE保持`0.209026/0.136350 m`且三个seed的transverse drift严格为0，证明收益来自预注册的纵向纠正而非重训已解决分量。
- 三个`2.526 GB`确定性临时cache精确复现V1后删除，共释放`7,578,175,002` bytes；35项seal SHA=`f6508db20f0a44f3dc1ecf4a9d0c0e2f6df6a6c2b1627005a4f8d13bf9d2094b`。
- NEXT只读重放C07--C08完整trace-commit graph：使用新三seed结构中心替换旧center输出，保持4 m关联半径、node/edge定义、阈值、因果历史和失败政策不变；先证明node split与edge F1改善，再决定是否允许C09。

## 2026-08-28：空间中心残差审计PASS；剩余阻塞锁定为junction纵向距离

- V1/V1R先后因`274 total / 272 pair-capable`口径和事件交集实现错误system FAIL；V1R2完整产出后又被`1e-12 m`复现检查拒绝，差异仅`1.28e-7 m`且来自float32 archive。三次错误均在独立run封存，不修改上游模型。
- V1R3用bit-exact within-4m检查和明确`1e-6 m`均值复现界正式PASS：`8,839 rows / 272 identities / 144,533 pairs`，0 optimizer/C09/C10/M-TARE；13项seal SHA=`95083e43216dce94889c24c9032182e47bbc4f6ca56a2163e85b68ae4a142e4c`。
- 反事实根因：`oracle longitudinal + predicted transverse`的within-4m=`0.999928`、relative distance=`0.284234 m`；`predicted longitudinal + oracle transverse`仍只有`0.615192`、`4.025258 m`。横向/高度已接近解决，剩余失败由纵向预测主导。
- 分层：完整spatial的junction within-4m=`0.389981`，terminal=`0.886363`；coordinate median/row medoid=`0.616593/0.616478`，均不如arithmetic ensemble `0.619923`；seed disagreement不是主要根因。
- 论文图 `spatial_center_residual_audit.png` SHA=`125dd4eaafa6a03d3f3fe7a657ff4e02c4ec4ee80d6ba0f0610503acb56b1e9d` 已保留。NEXT冻结现有transverse decoder，只训练消费同一空间context的bounded longitudinal residual；门槛、4m radius、数据和split不变。

## 2026-08-28：range-image空间中心已学到横向/高度，但4m一致性差0.403点

- V1三seed训练完整完成：每seed `3,960`步、合计`11,880`，原GSE encoder、action model和scalar forward均0步更新；三次`2.526 GB`临时缓存按合同删除，共释放`7.578 GB`。V1只在最终JSON写出时因`numpy.bool_`序列化错误system FAIL，35项seal SHA=`67fc5bd676b06ac7872f99e6421ab12f5d0503ef963536a7203054ca325aa473`。
- V1R只读资格run复用V1封存的三seed输出，新增训练步为0、`error=null`、C09/C10/M-TARE读取为0；13项seal SHA=`1aa2af8e25339f0b96a3cddb3cf560c06b86593984f046986db43314f340bc29`。
- 圆环空间decoder证明结构位置可学习：lateral MAE从`0.452628`降至`0.209026 m`，up MAE从`0.582302`降至`0.136350 m`，global center MAE从`3.752201`降至`3.515969 m`，relative error从`4.641362`降至`3.993150 m`（改善`13.97%`），三个seed全部改善。
- 唯一FAIL gate：within-4m fraction从`0.523952`升至`0.619923`，增益`0.095971<0.10`，差`0.004029`。不降低门槛、不运行图、不读取C09。
- NEXT做sealed C07--C08只读误差分解：按event/identity/纵横高分量定位4m边界外的剩余离群；只在根因支持时预注册一次更强的cross-traversal consistency训练，否则停止中心路线。

## 2026-08-28：局部3D小头V2仍FAIL；冻结action context容量耗尽，转range-image空间解码

- Vector V1因ensemble索引错误system FAIL封存；修正后V1R正式科学FAIL，三个seed均选择初始化epoch，relative/within-4m无改善。V1R 28项seal SHA=`86261304b309267bd7507ade2598558c56dcd191f7322242322a599c33d67149`。
- V2改用已改善12.30%的dual-batch scalar checkpoint并严格冻结hidden+forward，只训练独立lateral/up。正式run `error=null`、source unchanged、`11,880` transverse-head steps、0 backbone/C09/C10/M-TARE；28项seal SHA=`1a79af8ff44d992aced90a4f1826d831e3eb57edb75289041a01053227f5b1ce`。
- V2 relative improvement=`12.36%`、longitudinal MAE=`3.477373 m`、global-center MAE从`3.752201`降至`3.701380 m`，但lateral/up MAE仍等于零输出baseline `0.452628/0.582302 m`，within-4m增益仍`0.089455<0.10`。唯一未过门不变。
- 结论：冻结action-set pooled context不含可学习的横向/竖向事件定位；scalar/vector小头路线停止。NEXT直接消费圆环range-image空间feature map与route-aligned angular coordinates，训练专用局部3D中心decoder；现有小头作为“pooled context容量不足”消融。

## 2026-08-28：双批次corrective三项通过、一项差1.05点；按停止规则转3D局部中心

- V1R3三seed正式完成`11,880`小头steps，backbone=0、`error=null`、source unchanged、0 C09/C10/M-TARE；28项seal SHA=`e97120f6158a3e140b681be636fac2035acf21187a3dd98cf5015bb2d650d1a9`。
- ensemble MAE由`3.520097`改善到`3.477373 m`；identity-macro relative error由`4.589575`降至`4.025258 m`，改善`12.30%`；三个seed relative error均改善。这三项gate通过。
- within-4m fraction由`0.523952`升至`0.613407`，提高`8.95`个百分点，低于预注册`10`点，因此整体科学FAIL且没有启动图。
- 按停止规则不再调scalar loss/weight/epoch。NEXT利用现有route tangent构造机器人局部forward/lateral/up坐标，输出`3D event_center_vector`；hidden层从V1R3初始化，longitudinal输出继承scalar权重，横向/竖向零初始化。先做Teacher/接口和零泄漏证明，再训练一次；失败则停止当前中心回归路线。

## 2026-08-28：跨穿越配对监督方向有效但正式FAIL；改为保留原回归的双批次纠正

- 正式V1R2完成三seed `11,880`个小头optimizer steps，backbone steps=0、`error=null`、sources unchanged、0 C09/C10/M-TARE；28项seal SHA=`5d0c8ce49f37863e69aafd88933b1d9143785fb589799faab139b71e58a0693e`。
- C07--C08 identity-macro relative center error从`4.589575 m`降至`4.218949 m`，改善`8.08%`；同节点跨视角落入4m范围由`0.523952`升至`0.582877`，增益`5.89`个百分点。两项均改善但未达到预注册`10%/10点`。
- 单点MAE由`3.520097 m`退化到`3.821066 m`。根因是V1R2用identity-balanced pair batch同时承担direct loss，替代了V1R3的row-balanced event batch；优化为跨视角聚合牺牲了逐行距离。
- NEXT只允许一次有界纠正：从每seed V1R3 checkpoint初始化；每步独立使用原row-balanced direct batch和identity-balanced cross-traversal pair batch；epoch必须先满足不劣于对应V1R3 MAE，再最小化relative error。保持数据、头容量、loss 1:1、epochs和安全门不变。

## 2026-08-28：学习结构中心显著减少重复节点，但完整图仍FAIL；转向完整穿越边与延迟消歧

- event-center V1R3完成三seed小头训练，冻结LiDAR/action-set backbone，C07--C08中心偏移MAE从fit-only baseline `5.496263 m`降至`3.520097 m`，改善`35.95%`；三个seed及junction/terminal分项全部改善，正式PASS，31项seal SHA=`631dd0e6a985aeeeec7b620af81491ed296568b1269899756dab255b6baecedc`。
- 学习中心接入完整trace graph后，C07--C08提交节点由293降至179，node precision从`0.617747`升至`0.905028`，node F1升至`0.715232`，false loop merge保持0；但16个真实junction仍被拆分，edge precision/recall=`0.25/0.076923`，正式科学FAIL。21项seal SHA=`c19cf0865173c6d8baeafbb8ccb35fc648c44597c7a429d04c23093f533d275e`。
- 失败不是程序或泄漏：`error=null`、sources unchanged、0 optimizer/model update、0 C09/C10/M-TARE。16个重复结构中13个的最短预测中心距离仍超过4m；另外2个遇到多候选后按合同安全拒绝，没有错误loop merge。
- 初次客观中心诊断错误地把masked普通走廊行的零填充值当坐标，使假触发在世界原点聚集；由此得到的node P=`0.961390`与edge P=`0.75`无效。修正为“有效结构行用objective center、无效行保持部署预测”后，上界node P/R=`0.988095/0.908759`、edge P/R=`1.0/0.461538`、false loop=0，证明图机制在正确中心下具备安全容量。
- NEXT不扩大关联半径、不读取C09；把独立逐帧scalar offset改为同一traversal上的时序/运动一致回归，用已知1m弧长推进约束中心剩余距离连续变化，并保留三seed不确定性拒绝。C09/C10/M-TARE/闭环继续禁止。

## 2026-08-28：双层trace-commit首次离线图proof科学FAIL；定位为缺失结构中心回归

- 唯一正式C01--C08容量run完成、`error=null`、0 optimizer/model update、0 C09/C10/M-TARE。C01--C06从9个固定proposal阈值选择`0.98`，但没有配置满足图安全门。
- C07--C08完整GSE trace-commit将immediate ghost节点precision/F1从`0.421227/0.579247`提高到`0.617747/0.638448`，false loop merge=`0`；但293个提交节点只对应181个唯一正确节点，边precision/recall/F1=`0.75/0.230769/0.352941`，node-edge macro-F1=`0.495695`，仅比最强基线高`0.016882`，正式FAIL。
- 失败分解证明主问题是同一结构被拆为2--4个节点，而不是错误loop merge：当前图用sensor pose做4m候选，不同incident tunnel的观测位置无法统一到同一junction/terminal中心。
- 只读oracle-longitudinal可行性检查把同节点候选对纯度从`0.9704`提高到`0.9976`，提交节点precision/recall从`0.5854/0.7628`提高到`0.9838/0.8869`且0 merge。NEXT新增signed event-center offset Teacher与轻量回归头；C01--C06训练/选择、C07--C08验证，C09/C10继续封锁。
- 正式FAIL run 17项seal SHA=`6ee5f39e352746da9c0c90aea11df4fcafccbbcdfeaad5448c8df9851bd3e5e1`，旧模型保留为“无结构中心学习”消融。

## 2026-08-28：route-conditioned结构化出口解码容量FAIL；最终节点改为假设/提交双层图

- 零训练正式run扫描C01--C06固定`900`组：confidence 50档×incoming角3档×persistence 3档×seed consensus 2档。`0/900`满足拟合安全门，因此按合同没有选择配置，也没有读取/评价C07--C08结果，更没有读取C09/C10/M-TARE。
- 最佳最低召回诊断为confidence=`0.28`、incoming半角=`50°`、3帧持续、2-of-3；fit总P/false/R=`0.848541/0.151459/0.699878`，junction P/R=`0.845749/0.821794`，terminal P/R=`0.877637/0.281081`。失败属于token count/interface容量，不是验证域泛化。
- 正式run `error=null`、0 optimizer/inference，13项seal SHA=`e83035b6fdf038be9631ece11ece6e418a849881e18233db3c552df1b047604f`。不再训练或调规则以让当前exit tokens直接提交最终节点。
- 最新相关工作边界确认：NTS已用学习式可探索方向建立ghost nodes，传统multi-hypothesis topological SLAM已用执行验证拓扑假设；两者简单拼接不能作为创新。NEXT重构为双层GSE：学习几何语义产生带连续几何和exit relation的provisional hypothesis graph；route-conditioned association管理对应关系；只有执行轨迹一致的hypothesis才提交到verified graph。必须新增与NTS/传统hypothesis mapping的机制差异和独立消融。

## 2026-08-28：action-set黑盒节点分类容量FAIL；转向route-conditioned结构化出口解码

- 三seed action-set模型完成`39,996`步、原LiDAR/exit-token backbone更新为0。V1的三套训练均成功，但旧selector把“无安全阈值”抛成进程异常；V1因此按system FAIL封存，45项seal SHA=`6c9d36cc646e4189fa8c608b3bfc4a913885dfa17e4665cf00243859301e62b6`，不得覆盖。
- 零训练corrective保持同一C07--C08、1,001阈值、模型和门槛，正式科学FAIL且`error=null`。满足全部最低召回时的最高精度点为threshold=`0.995`：总precision/false/recall=`0.965870/0.034130/0.498239`，junction P/R=`0.951977/0.382086`，terminal P/R=`0.987069/0.901575`；十个family均有正确触发，但安全错误预算失败。
- junction/terminal identity coverage=`102/146=0.698630`和`121/128=0.945312`。这证明exit-token集合含强结构信号，尤其terminal，但把集合压成黑盒节点类别仍产生过多junction假节点。corrective 13项seal SHA=`9a45bba6c7e0da1ecda268357f0b9f7d2eefca3d3f1815fa2931b856958e927e`。
- 按停止规则不读C09、不降低门槛、不加graph stability或继续扩分类器。NEXT只允许C01--C08零训练route-conditioned结构化出口解码审计：显式扣除已穿越incoming action，再以剩余学习式exit tokens及五帧持续性产生junction/terminal；若仍无安全容量，停止当前node-proposal主张并重评论文核心。

## 2026-08-28：5帧decision-mass fallback在C07--C08容量FAIL；停止categorical node gate

- 正式两进程V1在selector启动前因环境checker把SciPy实际`1.15.3`写成`1.15.2`而system FAIL，threshold step和C09 read均为0；V1R只修正patch version，Data Card/preflight与20项测试全部PASS。
- V1R在C07--C08完整扫描固定1,001个`P(junction)+P(terminal)`阈值，但不存在同时满足aggregate precision>=`0.995`/recall>=`0.25`及两类precision>=`0.99`/recall>=`0.25`的非空配置。selector正式科学capacity FAIL，C09 applicator没有启动。
- 结合12帧C09 precision=`0.975359`与5帧selection无安全阈值，正式停止“旧五类event概率经阈值/episode压缩生成节点”的categorical路线；不得重选阈值、降低1%预算或加graph stability掩盖。
- NEXT改为action-set node proposal：节点直接由学习式exit tokens的数量、方向、opening width、vertical profile与因果稳定性产生；junction/terminal是可执行出口集合的结构，不再由事件名称间接决定。先做C01--C08只读容量proof，复用已训练token/association/geometry，不读C09/C10/M-TARE。

## 2026-08-28：12帧decision-node C09资格科学FAIL；自动进入预注册5帧专用门

- 新Factorized图接口只允许junction/terminal生成decision node；turn和geometry-transition仅保存为edge geometry，不再生成拓扑节点。纯部署历史由sequence/frame manifest构造，24,462个C09 observations、2,054 traversals和237,678个past reference cells与Teacher事后重建逐元素一致，future/Teacher inference inputs均为0。
- 正式V1在seed0空间编码后因软件错误停止：冻结`validation_outputs.npz`是确定性置乱顺序，而executor错误假设按global ID排序。V1不可覆盖地封存为system FAIL；新增global_sequence_index+parent_id双射对齐和2个回归测试，相关测试`19/19 PASS`。
- V2完整执行三seed`98,034`个空间编码帧和`73,386`个episode observations，零训练/阈值选择/C10/M-TARE。工程证据完整、`error=null`，但科学FAIL：487个decision triggers中475个正确，precision=`0.975359`、false fraction=`0.024641`、recall=`0.879630`；junction/terminal precision=`0.978836/0.963303`，未达到预注册`0.98`安全线。
- 有效能力保留：junction/terminal recall=`0.868545/0.921053`，identity coverage=`66/71=0.929577`和`54/59=0.915254`，证明结构信号很强但假节点预算仍不安全。正式图和机器源保留为论文失败分析。
- NEXT按Data Card预注册fallback执行C07--C08-only的5帧decision-mass gate：只用`P(junction)+P(terminal)`触发并压缩连续episode，冻结阈值后一次应用C09。禁止降低安全门、使用C09选择、调图参数或进入planner。

## 2026-08-27：C09单seed关联资格科学FAIL；图回放停止

- 正式C09双域run完整、`error=null`、source unchanged：balanced三seedprecision=`0.9891/0.9718/1.0`，runtime=`0.9706/0.9622/0.9839`，均未实现“所有seed错误接受不超过1%”。
- 运行时错误不是零负例假象：246个真实negative中，seed0/1/2分别接受`102/89/17`个；S08/S09是主要困难域。S02/S03零negative已标记为不可识别。
- 25/25 exact seal SHA=`ca31e1d...ac338`，图和机器源保留为论文失败分析；C10/M-TARE/optimizer/threshold selection均为0。当前禁止离线图、planner和闭环。
- 只读机制诊断排除了“选最好seed”与单纯投票；最小可行revision为C07--C08-only选择`k-of-3 consensus + metric distance cap`。C09只说明该机制可能有效，不参与具体参数选择；下一步先做selection-domain同构候选proof。

## 2026-08-27：最终坡度统一接口下关联容量正式PASS

- 唯一V1R run完成三seed共`564,378`次冻结坡度序列推理；每seed `188,126`条global IDs完整对齐，C07--C08 GPU重放byte-exact，只改146维观测column 10，其余145列byte-exact。
- descriptor与no-route selection输出逐字节复现V1。full统一接口三seedprecision=`1/0.995370/1`、false accept=`0/0.004630/0`、safe recall=`0.419708/0.784672/0.540146`，physical-only全部通过。
- full平均recall=`0.581509`，比descriptor/no-route高`0.362530/0.381995`。虽然低于旧坡度V1的`0.676399`，仍大幅通过预注册增益门，最终论文接口只采用V1R，不挑旧接口结果。
- 正式run `23.161s`、`1,239` small-head steps、323 MiB、69/69 exact seal，SHA=`64e5bfa8...c5eb40`；C09/C10/M-TARE/GSE-backbone运行均为0。下一步冻结C09 balanced+runtime双域只读资格。

## 2026-08-27：发现并隔离坡度接口漂移；统一接口V1R已完成实现和前置证明

- 容量V1使用旧GSE slope列，而最终已通过的感知合同使用五帧physics-guided corrective与固定风险比例`0.89`；full关联又明确消费slope-to-exit-vertical关系。直接进入C09会造成训练/部署接口不一致，因此已在任何C09模型运行前停止。
- C01--C08 frozen cache精确包含fit `142,184`、selection `45,942`，合计`188,126`个唯一sequence identities；与关联Teacher `188,126/188,126`同集合。三seed checkpoint在原CUDA路径对45,942条selection输出均byte-exact重放，CPU约`0.018--0.034°`差异已排除为正式路径。
- 新统一接口只按global sequence ID把最终坡度写入146维观测第10列；真实内存proof确认三seed均仅第10列变化，其余145列逐字节一致，输出范围有限且不超过`±45°`。新增纯函数与fail-closed测试，专项`9/9 PASS`。
- V1R runner/Data Card/spec freezer正在冻结：固定原1,584/548 pairs、full/no-route/descriptor、三seed训练与安全门槛；额外要求descriptor/no-route selection archives与V1逐字节一致。下一步为preflight及唯一不可覆盖V1R，C09/C10/M-TARE/图/规划仍为0。

## 2026-08-27：class-mass解析审计FAIL；排除简单类别加权并停止categorical detector路线

- 正式零训练audit精确复核fit/selection episode=`3,956/1,350`，其中transition=`114/34`，raw fit episode mass=`2.8817%`，equal-class权重倍率会达到`8.675x`；lag10 frozen geometry-risk AUC仍为`0.764515`。
- 关键反证是最终ensemble的解析logit gradient mass：transition=`28.2380%`、junction=`18.0855%`、terminal=`1.1837%`、turn=`52.4928%`。尽管transition样本少，其巨大误差已使当前raw objective给它的实际梯度质量高于junction；预注册的“transition gradient mass低于junction”检查FAIL。
- 若按类别强平衡，解析gradient share将变为transition=`72.3886%`、turn=`25.4180%`、junction=`1.7872%`、terminal=`0.4062%`，会把优化极端倾向两个未学会类别，不能作为有证据的修复。
- run `gate3_20260827_gse_causal_episode_class_mass_audit_v1_seed0`为科学FAIL、`error=null`、0 inference/optimizer/forbidden reads，12项seal SHA=`5b888c284f9d5c808c96dee4de6a7964ecaff02beedb4da0fa8f57da13e8953f`。按停止规则放弃categorical episode detector与equal-class V2，不再训练。
- NEXT回到论文方法设计：连续几何学习此前已证明约40%回归改善，因果长历史有可观测信号；评估以learned continuous geometry signature、因果segment/change-point和traversal-verified graph组合替代五类事件分类。实现前先做最新相关工作差异矩阵与sealed C01--C08离线可行性proof，禁止进入图/planner。

## 2026-08-27：12帧episode detector三种子正式科学FAIL；稳定触发成功但稀有类别被压制

- V1在首个optimizer step前因trainer把`np.float32`传给PyTorch `Tensor.to()`而system FAIL；seed0空间缓存完整、optimizer step=0、禁止数据读取=0。修正为`torch.float32`后，真实episode batch与含boundary target的batch均完成前向/反向，全部trainable gradient有限；V1R重新preflight PASS后独立执行。
- V1R完整执行三seed、每seed 6 epochs/13,332 steps，总计`39,996`个detector optimizer steps；冻结backbone optimizer steps=0，冻结编码合计`757,290`帧。耗时`656.90s`，source unchanged，`error=null`，35/35 seal复核PASS，seal SHA=`bf49586ddd57a5a78ee248a36ae7a5505eefbf44fa5cb492d6b5ff5b1b62e599`。
- runtime-shaped连续响应压缩有效：ensemble阈值`0.986`下产生1,051个trigger，structural precision=`0.991437`、episode recall=`0.771852`，junction/terminal identity coverage=`140/146`和`125/128`。
- 结构类别学习仍FAIL：ensemble frame macro-F1=`0.654245`，比旧directional baseline `0.687904`下降`0.033659`；三个seed全部回归。turn仅`6/95` identities，geometry-transition=`0/17`，未达到`0.737904`、turn 40%和change `7/17`门槛。
- 当前positive MIL直接对5,306个episode取平均，junction/terminal/turn/transition episode=`3,424/994/740/148`，transition只占positive loss约`2.79%`，junction约`64.53%`；输出向常见类别坍缩与此一致。NEXT先做sealed输出上的零训练class-loss/gradient-mass audit；仅在确认目标失衡后才允许一次不扩容的class-balanced episode MIL修订，图和planner继续禁止。

## 2026-08-27：12帧真实引用V1R正式PASS，开始方向保持的训练实现

- V1执行器已证明全部引用有效，但外层错误要求Teacher中出现完整`16,078`条traversal；实际为`16,076`条有观测traversal，另有`2`条客观过短、零五帧观测。V1按system/count-semantics FAIL封存，13/13 seal SHA=`679885f6c71ebf8111c90a2bc4cf0caebc8313cd0a76a008cd206b5ac4a8dbfc`。
- V1R只修正计数表达并正式PASS：`16,078 total = 16,076 observed + 2 zero-observation`，80 worlds、252,430 unique frames、188,126 observations和5,306 episodes不变。全部`1,818,662`个有效引用单元在对应parent shard内解析且traversal-local frame index连续。
- 12帧历史分布为长度5/6/7/8/9/10/11/12=`16,076/16,074/15,968/15,648/15,106/14,356/13,412/81,486`；没有重采ray或写sensor payload。四个引用/掩码/episode/event array digest已冻结。
- V1R耗时`16.46s`、峰值RSS=`1,104,568 KiB`，source unchanged，12/12 seal SHA=`125903986cd8db048423286a60433d9ab0aae67343e29184edee5c2f2ea9d124`。NEXT实现按seed串行的冻结空间特征缓存，保留36个环形方位bin和128D pooled feature，再训练12帧episode detector；可再生大缓存不进入长期保留包。

## 2026-08-27：因果事件监督单位审计PASS，冻结12帧episode检测路线

- 唯一正式只读run `gate3_20260827_gse_causal_event_supervision_audit_v1_seed0`精确覆盖C01--C08 `80 worlds / 16,078 directed traversals / 252,430 unique LiDAR frames / 188,126 causal observations`，得到`1,534`个结构identity和`5,306`段同traversal连续Teacher episode；零训练、零推理、零C09/C10/M-TARE读取。
- corrected change-point的`1,031`个最终标签中，`891`个发生在closed confirmation之前，只有`140`个落在确认帧；`621`个标签的回投边界已不在当前5帧历史内，仅`410`个仍包含边界。该证据确认当前“每个标签帧独立分类正确”的目标与“整段事件只需安全确认一次”的拓扑任务错位。
- `76`个change identities对应`152`个方向确认episode，`152/152`在确认时都有完整12帧过去历史；因此无需重采LiDAR，可直接用现有唯一帧引用构造masked 12-frame输入，不复制sensor payload。
- 正式run耗时`2.18s`、峰值RSS=`655,944 KiB`，source unchanged，18/18证据哈希PASS，seal SHA=`781ea53ce97cd3bd7a345600649a7e4c6d38fb144d22deac19ab4dafd6b8b7d9`；机制图PNG/PDF/SVG人工目视完整。
- NEXT：实现`proposal → past-only confirmation → one event trigger → boundary back-projection`接口；训练采用episode-level multiple-instance supervision，完整corridor episode作为困难负样本。旧5帧逐帧分类、delta decoder、最后块与线性风险读出全部保留为论文消融/失败分析，不再扩模。

## 2026-08-27：几何条件身份风险容量proof正式FAIL，停止分类器扩张

- 唯一正式run `gate3_20260827_gse_geometry_conditioned_identity_risk_capacity_v1_seed0`在C01--C06 `60 worlds / 142,184 observations`上拟合固定60维、六个过去距离的线性softmax，并在C07--C08 `20 worlds / 45,942 observations / 17 change identities`上按原合同评价；优化正常收敛，`error=null`。
- 选择域event macro-F1仅`0.610400`，未达到`0.737904`；geometry-transition recall虽为`0.920833`，precision仅`0.036070`，turn F1=`0.256124`。没有任何非空阈值能同时满足开放集precision/false-accept/recall合同，因此identity coverage拒绝计算，全部科学门槛FAIL。
- leave-family-out最差macro-F1为S05 `0.588974`，S09的geometry-transition F1=`0`，说明失败不是单一topology family异常。正式run耗时`33.02s`、峰值RSS=`1,793,968 KiB`，C09/C10/strict-test/M-TARE读取和上游模型推理均为0，source前后不变；16/16证据哈希复核PASS。
- 结论：较长历史确有几何信号，但逐帧低容量风险分类仍把大量普通走廊判成turn/transition。按预声明停止当前分类器扩张，不调阈值、不扩大backbone、不进入图或planner。NEXT仅做C01--C08只读的因果事件边界/Teacher监督单位审计，决定是否把逐帧目标改为事件episode级因果检测，或重建显式更长历史输入。

## 2026-08-27：因果几何风险冲突审计正式PASS，授权低容量多变量proof

- 唯一只读run `gate3_20260827_gse_causal_geometry_risk_conflict_audit_v1_seed0`精确复核C01--C06 `60 worlds / 142,184 observations`与C07--C08 `20 / 45,942 / 17 change identities`，固定执行2--12m全部11个lag，不选择winning lag。optimizer/model inference/C09/C10/M-TARE均为0，source前后完全一致。
- 冻结预测的transition-vs-corridor AUC从lag4 `0.680215`随更长历史上升到lag10 `0.764515`，7--11m峰值相对lag4增加`0.087342`；Teacher lag10 AUC=`0.851485`。这证明7--10.5m确认延迟对应的较长因果历史含有实质可迁移信号。
- 单一风险标量仍不可用：冻结预测在每个lag的fit-only 1% corridor阈值下均为`0/17`身份覆盖；lag10选择域101个高风险corridor中91个为普通corrected corridor，仅10个来自删除的旧transition，故不能靠恢复旧标签或删困难负样本解决。
- 置信度分解显示original GSE在240个变化帧上的median conditional change=`0.9868`、median structural=`0.6876`；冻结delta/最后块将structural压到`0.0117/0.0079`，直接解释为何联合训练改善回归却抹掉节点触发。
- 正式run耗时`19.53s`、峰值RSS=`1,207,728 KiB`，18/18 seal复核PASS，seal SHA=`56f6b82cec7526cb17bccf6cd2ffa3a688cf9295a972e013c58894dc0f5e6269`；PNG/PDF/SVG人工目视完整。NEXT按预声明执行一次低容量、identity-balanced、多变量几何风险capacity proof；若仍FAIL，停止分类器扩张并修改因果输入/Teacher。

## 2026-08-27：最后编码块有界微调正式FAIL，进入因果几何风险冲突审计

- 唯一正式run `gate3_20260827_gse_causal_geometry_delta_last_block_training_v1_seed0`完整执行三seed、6 epochs/seed和`52,236`次optimizer step，`error=null`、return code=`2`。三seed均只更新`encoder[-1]`与全新多任务头；27个允许梯度tensor、0个意外冻结梯度，所有更早参数digest逐seed不变，最后块digest逐seed发生变化。
- 连续几何仍成功：ensemble normalized delta MAE=`0.348158`，相对冻结基线改善`40.71%`。但结构事件ensemble macro-F1=`0.680096`，相对旧directional baseline变化`-0.007809`；安全门下change-point=`0/17`、turn=`23/95`，未达到`7/17`和40%身份覆盖门槛。seed0/1/2分别为change-point `0/0/1`、turn `3/15/6` identities，不能挑seed替代ensemble。
- 运行耗时`6,286.37s`、峰值RSS=`2,901,796 KiB`，C09/C10/M-TARE读取均为0。26/26证据文件从项目根独立哈希复核PASS，seal SHA=`27c9ec4c92e26b8eb0fe3b548b8fa4cb3cb45ae3b425dc1f29185ba70d403d53`。
- 结论：增加有限表征可塑性仍只能改善几何回归，不能把几何变化转成低风险结构节点事件；禁止继续扩大backbone微调或调图阈值。NEXT：执行已预声明、零训练的C01--C08因果几何风险冲突审计，比较2--12m历史、结构置信度分解和困难负样本；根据正式证据决定是否执行低容量几何条件身份风险proof或修订输入/Teacher。

## 2026-08-27：显式geometry-delta多任务学习正式FAIL，连续几何回归成功但事件触发未恢复

- 唯一正式run `gate3_20260827_gse_causal_geometry_delta_multitask_training_v1_seed0`完整执行三seed、6 epochs/seed和总计`52,236`次optimizer step；backbone update、C09/C10/M-TARE读取均为0，源证据未变化，`error=null`。26/26文件独立哈希复核PASS，seal SHA=`aed1d39e051b7a917a2996ff4db301a2993d100dfc58007925c8f1e790c347e4`。
- 连续结构学习有效：ensemble normalized delta MAE从冻结基线`0.587167`降至`0.355174`，改善`39.51%`；三个seed分别改善`43.21%/40.69%/40.12%`，全部超过预注册10%门槛。
- 结构事件仍失败：ensemble macro-F1=`0.683669`，相对旧directional baseline变化`-0.004235`；geometry-transition F1和17个change-point identity coverage仍为0。说明共享decoder能从冻结特征回归连续变化，但冻结表征没有把该变化组织成可安全触发节点的判别特征。
- NEXT：执行已预声明的单次有界fallback，只解冻`encoder[-1]`最后残差块并训练全新零初始化多任务头；其余encoder、GRU、directional temporal及所有旧任务head冻结。数据、Teacher、C01--C06/C07--C08 split、6 epochs和科学门槛不变，不读C09/C10/M-TARE，不调图或planner。
- 该结果已发布为论文Figure 8：四panel分别展示delta MAE、相对改善、event macro-F1和identity coverage。PNG/PDF/SVG、CSV/JSON、provenance与生成脚本完整保留，6/6清单文件独立哈希通过，manifest SHA=`e8e24f347774a5d6d2f42c6daa346aa2097f9a0b441fda88d42f1f78022f2948`。

## 2026-08-27：causal geometry-delta可观测性正式PASS，进入显式多任务表示

- 唯一正式run `gate3_20260827_gse_causal_geometry_delta_observability_v1_seed0`只读审计80个C01--C08 worlds、188,126 observations和122,765个两端几何有效的四步lag pairs；C01--C06/C07--C08有效pairs=`92,845/29,920`，optimizer、model inference、C09/C10/M-TARE均为0。
- Teacher宽高变化在C07--C08的transition-vs-corridor AUC=`0.742067`；冻结三seed的metric geometry先平均再取变化，AUC=`0.680215`，fit/selection绝对差=`0.008180`。预注册门槛`Teacher>=0.70`、`predicted>=0.65`、gap`<=0.05`全部通过，说明LiDAR表征有可迁移的连续结构变化信号。
- 该信号尚不能直接作为安全事件门：以C01--C06 corridor 99分位冻结阈值后，C07--C08 false positive约`1.065%`，change-point recall=`0`、identity coverage=`0/17`；Teacher标量也仅`1/17`。因此禁止用手工delta threshold，下一模型必须联合回归`Δwidth/height/slope/curvature`与结构事件，并保留不确定性拒绝。
- 正式run耗时`4.498s`、峰值RSS=`933,672 KiB`，18/18完整seal，seal SHA=`c75787626ecc09a27870276e86a172d6434081034f4704a1f5286d2afe52fa50`；PNG/PDF/SVG动机图与源JSON保存在run的`previews/`。NEXT：实现冻结五帧表征上的显式geometry-delta多任务decoder；若正式三seed训练仍不能跨世界改善事件/身份覆盖，再启用预声明的最后编码块有限微调，不调图或planner。

## 2026-08-27：corrected causal event head三种子科学FAIL，转向显式几何变化学习

- 唯一正式run `gate3_20260827_gse_corrected_causal_event_training_v1_seed0`完整执行三seed、36 epochs和36,000 head optimizer steps；backbone update=0，C09/C10/M-TARE读取=0，`error=null`。26/26完整seal独立复核PASS，seal SHA=`4ca5aa2581a170679d14b3681f840c79d3b0f9d2b59e5e7dcc1f2af57fe17e5e`。
- 三个best seed在C07--C08 macro-F1=`0.668039/0.667012/0.670414`；ensemble=`0.680608`，比旧directional corrected-label baseline `0.687904`下降`0.007296`。节点开放集precision/false/recall=`0.990108/0.009892/0.678061`通过安全与召回，但change-point correct-class identity coverage为`0/17`，turn仅`15/95`。
- 因此失败不是资源、随机seed、节点阈值或Teacher写入错误；冻结backbone+纯分类小头不能把新持久变化语义跨C01--C06→C07--C08泛化。不得重跑、加epoch、降低7/17门槛或调图参数。
- 只读机制诊断显示4m Teacher宽高差对transition-vs-corridor的selection AUC=`0.7421`，冻结三seed预测几何差均值AUC=`0.6789`；两者在1% corridor FPR下都只覆盖`1/17`与`0/17`身份。NEXT：预注册显式causal geometry-delta representation，用全部有效序列监督连续Δwidth/height/slope/curvature并联合事件分类；先做C01--C08-only可观测性/容量proof，不读C09/C10。

## 2026-08-27：corrected causal Teacher V1R正式PASS，进入identity-balanced学习

- 用户授权对既定论文范围内的实现分支自动采用证据最优方案；因此按推荐A执行V1R，不再要求把稳定global ID重编号。V1仍保持不可修改的checker FAIL。
- 正式V1R覆盖80个C01--C08 worlds、16,078 traversals和188,126 observations。22/22检查通过：源/输出稀疏索引顺序与SHA-256=`9081c80f...7f32`完全一致，索引唯一、严格递增、范围`0..208227`、9个gap blocks和20,102个合法保留槽全部精确。
- 最终Teacher为corridor/junction/terminal/turn/change-point=`150,964/26,608/7,525/1,998/1,031` observations；对应结构identity=`563/503/392/76`，总1,534。全部1,090 proof labels被消费，1,031应用、39 junction+20 terminal按优先级抑制；74,064 association pairs精确再生。
- C09/C10/M-TARE/model inference/training/optimizer均为0；17/17文件独立seal覆盖与哈希通过，seal SHA=`3f840e1165709d04aab80374739551931783045515f1ba95456107e4b5867f49`。NEXT：操作阶段返回Gate 3，只用C01--C06拟合、C07--C08选择，设计identity-balanced结构事件学习；C09仍不读取。

## 2026-08-26：corrected Teacher V1因错误的连续全局序号合同FAIL，数据与标签未失败

- 唯一正式run `gate2_20260826_gse_corrected_causal_teacher_manifest_v1_seed0`已不可修改地封存为`FAIL_GSE_CORRECTED_CAUSAL_TEACHER_MANIFEST_V1`。除一项外全部检查通过：80 worlds、16,078 traversals、188,126 observations、1,031 change-point labels、76 change-point identities、1,534 total identities和74,064 association pairs均精确匹配，C09/C10/M-TARE/model/training读取为0。
- 唯一失败合同错误要求train子集的`global_sequence_index`等于连续`0..188125`。源Teacher的索引本来覆盖全部212,588条train/validation/strict观测；筛出C01--C08 train后必须保留20,102个C09/C10保留槽，因此合法索引唯一且单调，但范围为`0..208227`并有间隙。无缺失、无重复、无字段漂移。
- 问题归类为system/metric checker，不是data、Teacher或model。推荐replacement V1R只把验收改为“逐元素保留sealed源索引集合且输出严格单调”，不重编号、不改任何标签/身份/pair/阈值；预计约1分钟CPU。按失败策略，V1不覆盖且在明确选择前不创建replacement。
- 下游只读影响审计确认该假设没有扩散：现有open-set、directional/rare-event训练、dataset export和topology replay均已把稀疏global ID映射到compact row，且冻结Data Card明确记录`0..208227`与20,102个合法空槽；错误只存在于本次V1 executor的一条验收表达式。

## 2026-08-26：方案A持久双向因果change-point Teacher完整proof正式PASS

- 用户明确选择A后，新增类型化持久双向Teacher、过去窗口边界回投、degree-2端点合并和fail-closed多义拒绝；17项相关合成/旧Teacher回归测试全部PASS。正式run `gate3_20260826_gse_causal_change_point_proof_v1_seed0`覆盖精确80个C01--C08 worlds、8,039 edges、16,078 traversals、252,430 full geometry frames和188,126 sequences，C09/C10/M-TARE/model/training读取均为0。
- 旧3,035个transition identities经1,010个单向持久episode、435个双向候选和108个双向唯一因果候选收敛为76个结构change-points：45个degree-2 endpoint nodes、31个interior-edge points。双向检测延迟`7.0--10.5m`、中位`8.0m`，节点位置使用回投边界而非检测pose。
- proof产生1,090条候选标签；逐帧应用既有junction/terminal优先后抑制59条，最终保留1,031条、76/76 identities不丢失，C01--C06/C07--C08 identity=`59/17`，每identity `6--23`条。超过既有每类500条训练容量门槛，但独立identity仍少，后续必须identity-balanced并报告identity coverage，禁止相邻帧冒充独立样本。
- 正式run 23/23全目录seal复核PASS，SHA=`54bdc8155c3113b554240c63f568257dc8209f35e9cd0e98c28b6b6daf7b5f60`。原run图b把全局500门槛画在逐family柱上，数值不变但表达易误读；原图保持immutable，论文版纠正图包已从sealed source发布，manifest=`95d29a693251637e9f03c7367be6db3c4e22d125b2fb1039bcd3af5818c42b79`。
- NEXT：冻结并生成corrected Teacher manifest——保留junction/terminal/turn，全部旧transition先恢复corridor，再只写入1,031条新因果标签和76个新identity；完整一致性与泄漏proof PASS后才允许三seed结构事件训练，C09/C10/M-TARE继续不读取。

## 2026-08-26：transition失败分析论文图包发布并同步正文

- 从5个完整sealed来源自动发布`gse_transition_teacher_audit`，同时保留PNG/PDF/SVG、CSV、source JSON、provenance与SHA-256；6个清单内文件逐哈希一致，manifest SHA=`800008a52a1a0fcda07063af34dda67aecf3b87d924d81b379af66735c39b055`。图包只绘制C07--C08统计，C10/M-TARE读取0、无手填结果。
- 四panel固定展示：结构identity构成、transition到physical-edge端点距离、canonical标注跨度、冻结event/292D residual/directional head三方法的四类identity coverage；人工目视无裁切、遮挡或图例歧义。
- `docs/GSE_GRAPH_MANUSCRIPT_DRAFT_V1.md`已加入development failure analysis和Figure 5；`docs/PAPER_COMPLETION_MATRIX_V1.md`已从过期的“训练中/待C09”状态更新为当前sealed训练、感知PASS组件、C09图FAIL及Teacher决策边界。NEXT仍需明确A/B；受影响Teacher/训练/图回放不推进。

## 2026-08-26：Directional Event Head正式FAIL；geometry-transition Teacher/metric缺陷需决策

- 唯一正式run `gate3_20260826_gse_directional_structural_event_training_v1_seed0`完整执行三seed/36,000次head update，backbone update=0，耗时`3368.40s`、峰值RSS=`2,765,292 KiB`、C09/C10/M-TARE正式读取=0；26/26完整seal SHA=`ea7507d53802126ea3da6141604a2f7812727718fbda4fea281db833151c7aab`，状态`FAIL_GSE_DIRECTIONAL_STRUCTURAL_EVENT_TRAINING_V1`且`error=null`。
- C07--C08 ensemble macro-F1=`0.695935`，节点precision/false/recall=`0.990073/0.009927/0.487987`；junction/terminal/turn/transition identity coverage=`0.972603/0.984375/0.189474/0.002688`。方向布局明显改善turn并保持安全，但transition仍不可用。
- Teacher只读审计发现744个selection transition identities中`90.59%`位于edge端点10m内、`55.24%`区间不超过2m、`19.22%`仅单向出现；transition占全部1113结构identity的`66.85%`。当前标签主要捕获短暂端点/mesh seam变化，且帧级open-set合同不适合因果延迟change-point。
- 受影响工作停止。详见`docs/GSE_GEOMETRY_TRANSITION_TEACHER_AUDIT_V1.md`。推荐A：重定义持久、双向一致的因果几何change-point并先做C01--C08只读proof；备选B为只保留edge几何属性，C继续扩大模型不推荐。NEXT等待明确研究决策，不进入C09/C10/闭环。

## 2026-08-26：排除 Teacher 不可观测假设；定位事件头空间平均缺陷

- C07--C08 只读审计确认，Teacher 的几何突变比较跨度为前后各 `5 m`，学生仍只输入当前和四个过去帧；但传感器为 `50 m` 全向 LiDAR，转弯/突变当前帧前向扇区至少一束达到 `5 m` 的比例为 `520/521` 与 `4310/4333`。冻结三seed逐帧召回又分别为 `90.21%/82.88%`，故没有证据把失败归类为未来帧泄漏或不可观测标签。
- 真正断点在高精度节点门：阈值 `0.988674` 下转弯只覆盖 `5/95` identities、突变只覆盖 `1/744`；两类结构分数中位数仅 `0.898658/0.701140`，而junction/terminal为 `0.995860/0.999878`。
- 源码核对确认`event_head`只消费全空间平均后的GRU context；逐方位五帧特征只进入axis与exit-token分支。正式rare-event residual corrective又证明压缩后的146D输出无法恢复该信息：macro-F1=`0.706287`、turn=`11/95`、transition=`0/744`，run seal SHA=`c5b58a612f22ebbe9ac86ca4021909133dcf4d5403a23baff318923ad5374c9c`。
- 结论与下一步记录于`docs/GSE_RARE_EVENT_OBSERVABILITY_AUDIT_V1.md`。NEXT：保持Teacher、数据和主干冻结，实现逐方位五帧Directional Structural Event Head，拆分binary node evidence与conditional event class；先用C01--C06拟合/C07--C08选择，PASS前不再读取C09。

## 2026-08-26：event-node融合校准PASS；C09证明turn/transition表征是主阻塞

- C01--C08-only正式run `gate4_20260826_gse_event_node_ensemble_calibration_v1_seed0`已`COMPLETED/PASS_GSE_EVENT_NODE_ENSEMBLE_CALIBRATION_V1`。45,942条C07--C08观测上，冻结三seed非corridor概率均值乘冻结node score，阈值=`0.9597587988776491`，accepted=`6,831`、TP/FP=`6,763/68`、precision/false/recall=`0.990045/0.009955/0.493902`，10 family全部通过；13/13 seal SHA=`509cbe125cf8ecfa3ba93370fd161e71155af0a66eaf475e78e7fd7b24e15ca1`，零训练/inference/C09/C10/M-TARE。
- 已实现结构episode粗状态、融合事件类别和软类别关联接口，27项图/回放/ensemble测试PASS。完整C09 81组只读审计仍为`0/81`安全：max node/edge F1=`0.277778/0.063847`，说明开发域frame recall增益没有解决唯一地点覆盖。
- C09逐类identity coverage定位为：junction=`92.96%`、terminal=`96.61%`、turn=`3.77%`、geometry-transition=`1.46%`。turn+transition占`464/594`个Teacher identity，故整体event-node identity coverage只有`22.05%`；这是当前图召回上限的直接原因。
- C01--C06/C07--C08监督容量充分：turn observations=`1,482/521`、identities=`297/95`；geometry-transition observations=`12,534/4,333`、identities=`2,291/744`。NEXT：冻结主干和现有pair verifier，设计train-only、身份/episode平衡的结构事件corrective head；先在C01--C06拟合、C07--C08选择证明turn/transition identity coverage与开放集精度，再允许新的C09验证。

## 2026-08-26：C09拓扑V4正式FAIL；事件episode修正仍不足以恢复召回

- V4正式run `gate4_20260826_gse_offline_topology_validation_v4_seed0`已封存为`FAIL_GSE_OFFLINE_TOPOLOGY_VALIDATION_V4`。10个C09 worlds、24,462 sequences、四方法各自预声明243组中，三个基线均完成；主方法243组没有association-safe配置。305/305证据哈希匹配，seal SHA=`cdb98789d8b98a1d15007f94f72379f1c0814c4fc8288ab59777cacbe9e44f8d`，零C10/M-TARE/model update。
- 失败后的实现级根因修正把结构事件改为episode：同一段连续junction/terminal/turn/transition只允许产生一个结构节点，稳定corridor后才复位；事件稳定性与高精度node matchability门分离。26项图与replay专项测试PASS。
- 新逻辑在完整C09上的81组只读可行性审计仍为`0/81`安全。最佳node/edge F1=`0.290576/0.054865`；最接近安全的配置为`95 correct / 2 false`，precision=`0.979381`、false-loop=`0.020619`，真实节点recall仅`0.186869`。连通分量偏差已为0、cycle signed bias约`0.20`，说明重复节点/环数问题改善，但感知事件召回仍是主瓶颈。
- NEXT：停止继续调图参数。只在C01--C08上评估并校准三seed结构事件融合/episode代表观测，先证明能提高真实事件召回且维持开放集精度；未形成选择域证据前不再建V5正式run，不读C10或启动闭环。

## 2026-08-26：三seed开放集节点门正式校准PASS

- 唯一run `gate4_20260826_gse_node_matchability_ensemble_calibration_v1_seed0`已`COMPLETED/PASS_GSE_NODE_MATCHABILITY_ENSEMBLE_CALIBRATION_V1`；13/13 seal完整覆盖PASS，SHA=`08fd27e4af2a0351db65f549fafa3c0356680a7407844d680027d4059c78b4c0`，耗时`3.015s`，C09/C10/M-TARE与optimizer/backbone update均为0。
- C07--C08选择集精确为`45,942` observations（`13,693` structural identity-valid / `32,249` open-set）。三seed冻结matchability sigmoid用float64等权均值，阈值=`0.982292910416921`，accepted=`4,827`、TP/FP=`4,779/48`、precision=`0.9900559354`、false accept=`0.0099440646`、recall=`0.3490104433`；10 family全部通过。
- C09只读grid177原型证明节点门方向有效：system precision从`0.96463`升至`0.98450`，false-loop从`0.03537`降至`0.01550`，false merges从48降至14，cycle bias从`7.3`降至`3.27`；但单配置仍未达到1%，不是正式PASS。
- NEXT：先完整补齐三个基线的C09 sweep，再运行带冻结节点门的243组完整GSE corrective；阈值、pair gate、16m、Teacher和科学门槛全部不变。

## 2026-08-26：C09拓扑V3正式FAIL，根因从pair关联收敛到结构事件开放集门

- 唯一正式run `gate4_20260826_gse_offline_topology_validation_v3_seed0`完成主方法`243/243`预注册配置后fail closed；运行`532.99s`、峰值RSS约`2.33 GiB`、零optimizer/model update、零C10/M-TARE读取。12/12完整seal复核PASS，seal SHA=`a13a837a72b3ed0bc142b1601c578fbb5c0bfd70241d76a708d5cc62d25b4d6e`。
- 243组中无一达到系统级association precision `>=0.98`且false-loop `<=0.01`。最高precision配置为grid177：`1,357`次merge中`1,309`正确、`48`错误，precision=`0.964628`、false-loop=`0.035372`；最大node F1=`0.289822`、最大edge F1=`0.079003`，cycle-rank signed bias仍为`5.47--11.6`，因此不得读取C10或启动闭环。
- 固定grid177的只读归因把48次false merge分为：8次真实结构identity错配、4次candidate无唯一identity、36次query Teacher为corridor但模型稳定触发structural event。若只看有唯一结构identity的pair，precision约`1309/(1309+8+4)=0.9909`；说明冻结exit-token verifier的结构节点配对能力成立，主要失效点是上游结构事件假阳性穿过matchability门。
- 接口实现同时修复了两个不改变科学语义的性能/一致性问题：typed adapter改为先切row再cast，避免对24,462行反复复制整档；图节点使用精确空间cell候选并冻结creation observation/xyz代表对。39项相关测试PASS，单S01候选证明为`1,474 sequences / 39,230 <=16m pairs`。
- NEXT：不调关联阈值、不删帧、不用planner掩盖。先冻结一个C01--C08-only的三seed结构事件开放集一致性门，并补齐三种基线的完整C09 sweep；只有新节点生成门在C09同时恢复系统关联安全、node/edge增益与图不变量后，才允许进入C10。

## 2026-08-26：distance-aware三seedensemble正式PASS，冻结唯一关联接口

- 唯一只读run `gate3_20260826_gse_distance_aware_ensemble_calibration_v1_seed0`已`COMPLETED/PASS_GSE_DISTANCE_AWARE_ENSEMBLE_CALIBRATION_V1`；13/13 seal PASS，SHA=`25be6e18f3ee29a1d55943c340513fbd5a0821cf2388ae5526dc659223d7aa62`，耗时`2.19s`，optimizer/backbone/inference/GPU/C09/C10/M-TARE均为0。
- 线上候选域精确为31,469 pairs（12,813 positive / 18,656 negative），排除13,903条`>16m`离线训练/诊断pair。三seed V2 score用float64等权均值，不学习权重、不挑seed。
- 冻结阈值=`0.9431912302970886`，accepted=`3,720`，TP/FP=`3,683/37`，precision=`0.9900537634`、false merge=`0.0099462366`、recall=`0.2874424413`。10个family全部非空，最低family precision为S03 `0.96949`，最低recall为S08 `0.19227`，均满足门槛。
- V1R和V2原run继续是sealed FAIL；新PASS只证明新的distance-aware ensemble接口在C07--C08 selection上合格。NEXT：把三seedexit-token verifier、16m候选gate与冻结阈值接入新的C09离线拓扑replay，C09只评价不再选阈值/权重/radius；C10/M-TARE继续0读取。

## 2026-08-26：exit-token V2三seedFAIL；发现线上16m候选域未进入阈值评估

- 唯一V2 run `gate3_20260826_gse_exit_token_association_corrective_v2_seed0`已封存为`FAIL_GSE_EXIT_TOKEN_ASSOCIATION_CORRECTIVE_V2`；40/40 seal PASS，SHA=`7864f8c4893d104fd9a711147b76d8ef09b59dcaa29e22497dd7079e9c6d5037`。三seed均为best epoch 2，共`1,608` verifier steps、backbone 0、C09/C10/M-TARE读取0。
- exit-token方法有一致增益但单seed仍不安全：约25% recall处seed0/1/2 precision=`0.983114/0.984425/0.976804`，均未满足错误合并`<=1%`；V2正式FAIL不改写。
- 新metric/interface缺陷证据：Data Card与`OpenSetAssociationContract`规定在线candidate distance `<=16m`，但阈值选择器未使用distance。45,372条selection pairs中只有31,469条在线eligible；13,903条域外pair含755 positive和13,148 negative，最大距离约62m，线上永不进入关联候选。
- 正确过滤后，单seed仍FAIL，但V2三seed确定性均值在全部10 family上得到precision=`0.990054`、false merge=`0.009946`、recall=`0.287442`，而V1R均值仍无合格阈值。这是selection-only只读证据，不得回写为V2 PASS。
- NEXT：实现显式distance-aware selector并新增只读immutable ensemble calibration audit。若新审计重放上述结果且seal PASS，冻结三seed均值与唯一阈值，再首次读取C09验证；不得跳过该审计或把单seedFAIL改名为PASS。

## 2026-08-26：open-set corrective V1R三seed科学FAIL，定位出口token信息丢失

- 唯一V1R正式run `gate3_20260826_gse_open_set_association_corrective_v1r_seed0`已完成并封存为`FAIL_GSE_OPEN_SET_ASSOCIATION_CORRECTIVE_V1R`；37/37 seal复核PASS，seal SHA=`dd4c0b601147d77d124897c392575c6fb27d2de7a775dd57cf7d4e61002081eb`。运行`2552.54s`、判别器optimizer steps=`1809`、backbone steps=`0`，C09/C10/M-TARE读取均为0。
- 稀疏global ID修复已被正式证明：`188,126`个C01--C08身份保留完整export的`0..208227`编号，含9个合法空洞和20,102个隔离C09身份；`135,232/45,372` pair精确复现。V1的失败因此确认是system checker defect，V1R结果才是科学结论。
- 三seed最佳epoch=`2/4/3`，selection balanced BCE=`0.347957/0.350172/0.330133`，但三者均无满足全部non-vacuous安全合同的阈值。在召回约25%处，precision分别约`0.975862/0.981040/0.980123`，错误合并比例约`2.4138%/1.8960%/1.9877%`，仍高于固定`1%`。
- post-hoc只读三seed均值诊断达到precision约`0.987959`、错误合并`1.2041%`、recall约`0.2540`，说明特征接近但仍不能以ensemble投机通过。false positives同时含different-identity与open-set，后者占主要部分，S03/S07/S09较集中。
- 实现审计发现V1只把六个出口压成5个总体统计，且完全遗漏已监督训练的32D `exit_descriptor`；因此它并未实现论文要求的exit-token matching，也不能区分具有相似总体几何但出口布局不同的位置。NEXT：保持冻结GSE和相同C01--C06/C07--C08 split，设计轻量、旋转不变、置换不变的出口token集合匹配与困难负样本校准；在新Data Card/spec前不读C09/C10、不运行图或planner。

## 2026-08-26：方案A验证器接口已实现，旧open-set容量统计口径不可复现

- 已实现冻结GSE输出后的`matchability + symmetric pair verifier`、严格对称pair特征和non-vacuous阈值选择；总体要求precision `>=0.98`、false accept `<=0.01`、recall `>=0.25`，每个topology family还必须有非空接受并满足precision `>=0.95`、recall `>=0.10`。10项专项测试PASS。
- 对sealed C01--C08数据按声明语义重放发现旧容量数字错误：同parent、真实3D sensor位置、严格过去、16m内，open-set pair实际为fit `57,066`、selection `18,111`，而非`79,478/25,360`。即使取消“严格过去”，数学上限也只有`71,277/22,843`，所以旧数字不可能符合原声明。
- 正确总pair应为既有`78,166/27,261`加open-set `57,066/18,111`，即fit `135,232`、selection `45,372`。样本量仍覆盖60/20 worlds与全部10个family，但正式Data Card不得静默沿用错误数字。
- 受影响训练尚未启动，C09/C10/M-TARE读取、模型更新和optimizer step均为0。NEXT：明确采用可复现的严格过去3D口径，或重新定义并审计另一候选距离语义；冻结前不得训练。

## 2026-08-25：open-set关联corrective train-only容量只读证明

- 不读取scan数值、模型输出、C09/C10或M-TARE，只读取sealed C01--C08 Zarr身份/事件/位置元数据与既有pair manifest。C01--C06 fit为`60 worlds / 142,184 sequences / 39,310 identity-valid / 102,874 open-set corridor / 3,380 identities`；C07--C08 selection为`20 / 45,942 / 13,693 / 32,249 / 1,113`，两部分identity交集为0。
- 现有identity pair为fit `38,897 positive + 39,269 different-identity hard negative = 78,166`，selection `13,568 + 13,693 = 27,261`。
- 对每个open-set query只取同parent、严格过去、最近的identity-valid候选，最大16m候选半径内fit/selection分别有`79,478/25,360`条；8m范围仍有`59,628/18,992`。全部10个topology family均非空。
- 因而推荐corrective可冻结为fit `157,644` pairs、selection `52,621` pairs：既有正负对加一条/查询的16m open-set负对。该容量证明只支持可行性，不构成训练授权或性能结论；模型、loss、阈值与实质性recall门槛仍等待明确方法决策。

## 2026-08-25：离线拓扑V2关联安全正式FAIL，定位开放集缺口

- 正式V2完成`gse_learned_association`全部243组后，没有任何配置满足`precision>=0.98/false-loop<=0.01/nonzero`；最佳precision=`0.514286`，故在第一方法选择点fail closed。旧run保留完整243-row sweep、日志、RUN_STATE和12项seal，SHA=`4883061636c365f7210b9c5ae2f3993ecd667327a69b6ec41a2a67049b6452b3`。
- 三seed最高precision配置共70 merges=`36 correct + 34 false`。34个false目标全部无稳定GT结构身份；32个查询无结构identity、2个查询有identity但目标由早前误触发产生。主问题是普通位置的结构事件误触发进入未做open-set训练的descriptor关联，不是坡度overlay、资源或真实节点间identity混淆。
- 单纯用C09开放集负样本提高descriptor threshold会把三seed正确接受压到`46/3/2`，召回不足1%，不采用这种近乎全拒绝的门槛投机。
- NEXT受方法决策阻塞：推荐冻结原GSE，新增C01--C06 fit/C07--C08 selection的open-set matchability+pair verifier corrective，明确训练无身份查询、困难负样本和non-vacuous recall；备选为完整重训关联head，或停止学习关联主张。C10/M-TARE继续0读取。

## 2026-08-25：风险校准感知V2正式PASS，离线拓扑V2前置合同完成

- 正式run `gate3_20260825_gse_risk_calibrated_perception_validation_v2_seed0`已`COMPLETED/PASS_GSE_RISK_CALIBRATED_PERCEPTION_VALIDATION_V2`；38项seal逐哈希与全目录覆盖PASS，seal SHA=`5194dd6c489a31bdab0998d1cec1a0df27b677668afd04b8c128aebf26e0478e`。
- C07--C08 maximin选择固定scale=`0.89`；C09坡度三seed均值MAE=`0.768638°`、相对五帧先验改善`47.042%`，S02最坏回归`4.319%`。事件、关联与四字段完整感知门槛全部PASS，C10/M-TARE读取和模型更新均为0。
- 两套论文图包已发布并目视通过：`gse_perception_validation` manifest SHA=`0369797fd55ba087500bd8757abf09cd667a698328436233fc54899475f9838b`；`gse_slope_risk_calibration` manifest SHA=`cbc3dd3801704167cae357ff9968add43dab3626d44ec40ee063856fa9829781`。
- 离线拓扑V2已实现风险校准坡度按唯一`global_sequence_index`对齐覆盖，保持事件、宽高、曲率和描述子不变；旧感知FAIL仅作为已通过组件来源，完整V2 PASS作为新资格来源。Data Card验证0错误/0警告，24项证据/回放/发布测试PASS。
- NEXT：冻结run spec并执行唯一24,300 replay正式run；完整GSE的node/edge F1必须各自领先最强可部署基线至少`0.05`，关联precision不低于`0.98`、false loop不超过`1%`，component/cycle-rank有符号偏差绝对值不超过`0.25`。

## 2026-08-25：C09全残差corrective真实逐世界FAIL，风险校准V2进入正式预检

- 唯一V1 C09 run封存为`FAIL_GSE_CORRECTED_PERCEPTION_VALIDATION_V1`：29/29 seal复核PASS，seal SHA=`63a903ba7f7c75891bf9665f359a7dd6637788ec541832456941d1c5ab0398c4`，零optimizer/model update、零C10/M-TARE读取。汇总器另有sealed schema字段名错误，旧run不修改。
- 技术输出证明三seed C09总体slope MAE=`0.759107/0.746825/0.742425°`，相对五帧先验总体改善约`47.70%/48.54%/48.85%`；但`S02_3d_tree_small_C09`三seed平均MAE=`1.086488°`，基线`0.958263°`，退化`13.381%`，违反固定逐世界最多5%退化合同。
- 根因归类为模型跨几何风险/过修正：S02-C01--C06/C07--C08先验MAE为`2.0649/2.4613°`，C09同类先验突然降至`0.9583°`；网络继续施加为训练分布学到的残差。现有error-scale与实际错误几乎不相关，不能用置信阈值补救。
- 拒绝降低逐世界门槛。新增一个共享的risk-calibrated residual：只在C07--C08的固定`0.00..1.00`、步长`0.01`网格上最大化最差parent三seed平均改善，再按均值和较小index确定性决胜。选择值为`0.89`，不读取C09。
- 只读原型在完整C09上得到slope三seed平均MAE=`0.768638°`、总体改善`47.042%`；S02退化收敛到`4.319%`，原event/association不变，四字段平均改善`52.763%`。18项专项测试PASS。
- 新V2 Data Card/spec冻结20个C07--C08 calibration worlds、45,942序列、10个C09 development-validation worlds、24,462序列、零C10/M-TARE读取。当前NEXT为清除已封存V1的stale running状态后重新preflight；尚未创建V2正式run。

## 2026-08-24：坡度corrective已预注册，回退Gate 2执行train-only三seed证明

- V1R正式完成并封存为`PASS_GSE_SLOPE_CORRECTIVE_THREE_SEED_TRAINING_V1R`：seed0/1/2最佳epoch=`31/50/47`，C07--C08 slope MAE=`0.862578/0.862454/0.863412°`，五帧解析先验固定为`1.664656°`，相对改善=`48.1828%/48.1903%/48.1327%`；三seed均值改善`48.1686%`。最弱拓扑族仍改善约`32.05%`，全部seed与全部10 family无退化。
- 总optimizer steps=`19,321`、GPU峰值约`126.1 MB`、host RSS峰值约`1.62M KiB`、运行`527.91s`、输出约`12.6 MiB`。115条seal与除seal自身外115个文件精确一致、0坏哈希，seal SHA=`d8683e5b965d8f9f142263aad54a2c0e9277e3e75002f3a04fe807d756d23748`。
- 训练证据的`strict_test_worlds_read=0`已包含C10，但summary未单列C10；不修改既有seal。下一C09 runner将验证cache manifest的80个actual source paths全部是C01--C08 train shards，并显式输出`c10_worlds_read=0`。`status_snapshot.json`按设计是pre-run snapshot，最终状态权威为RUN_STATE+metrics+seal。
- Gate 2 corrective已PASS，执行位置返回Gate 3。下一唯一操作是新的完整C09资格：旧事件/出口/关联/宽高/曲率证据逐seal复用，只对全部24,462序列计算冻结坡度；同时要求原完整门槛和learned-vs-five-frame-prior门槛均PASS。
- 首次正式V1在第一个C01 cache shard、任何训练前因检查器错误FAIL并封存：它把每条directed traversal都会重置的`local_frame_index`误当成world-global `arange`。11文件seal SHA=`eb75870c...`，0 seed、0 optimizer step、C09/C10/M-TARE零读取。全80-world只读审计证明188,126条序列的Zarr行连续、序列内local index每步`+1`且global reference一致，故这是system checker defect，不是数据缺陷。
- V1R只把错误的world-global断言改成正确的within-sequence因果断言，并新增“遍历间reset合法 / 序列跨reset失败”两项回归；专项测试现`12/12 PASS`。V1保持不可修改，V1R的数据、模型、seed、优化、baseline和科学门槛均不变。
- 这不是重启GSE主模型：V1R已通过的event、axis、width、height、curvature、place和exit输出全部冻结；corrective只替换失败的`slope_deg`并增加坡度误差尺度。
- 固定方法为“五帧独立解析几何 → 五帧坡度均值先验 → GRU(32)学习有界残差”。最终层零初始化，epoch 0与解析先验逐元素一致；corrective不得读取冻结GSE backbone特征，避免其已见C01--C08造成内部split污染。
- 新Data Card精确绑定C01--C06拟合`60 worlds / 12,106 directed traversals / 190,600 unique frames / 142,184 sequences`，C07--C08内部选择`20 / 3,972 / 61,830 / 45,942`；合计80 worlds、8,039 physical edges、16,078 directed traversals、252,430 unique frames、188,126 sequences。C09/C10/M-TARE保持零读取。
- 真实train shards逐项证明五帧引用`188,126/188,126`严格递增、范围合法并与global frame references一致；坡度Teacher mask全部有效。世界身份、geometry列、掩码、有限值和源seal均fail closed。先前口头记录`12,104/16,076`已由冻结轨迹清单与`8,039×2`纠正为`12,106/16,078`。
- 模型、缓存和训练接口已实现；10项专项测试、真实C01单序列Zarr索引检查、全80-world只读因果引用审计和CPU合成评估PASS。正式run只在mean三seed相对五帧解析先验提升`>=5%`、至少2 seed各提升`>=5%`、全部seed不退化且任何拓扑族退化不超过5%时PASS；否则停止且不读C09。
- 当前operation-bound Data Card：`configs/v3/gate2/data_cards/gse_slope_corrective_three_seed_training_v1.json`；正式spec：`configs/v3/gate2/gse_slope_corrective_three_seed_training_v1.json`。下一步只允许preflight后创建并执行这一个不可覆盖run。

## 2026-08-24：GSE V1R训练PASS，C09感知资格因坡度头退化正式FAIL

- 唯一C09 run `gate3_20260824_gse_perception_validation_v1_seed0`已执行并按预注册门槛封存为`FAIL_GSE_PERCEPTION_VALIDATION_V1`；49个文件保留，model updates/optimizer steps/C10/M-TARE读取均为0，未进入离线拓扑或C10。
- event gate PASS：三seed校准macro-F1=`0.660767/0.666286/0.674803`，相对M1D平均绝对提升`0.154469`；association gate PASS：place/exit均有非空接受集，precision约`0.990`，false accept/loop merge低于`0.01`。
- geometry macro gate的四字段平均相对改善为`0.174563`，其中curvature/height/width分别改善`85.31%/52.00%/24.07%`；但slope从非学习基线`1.527336°`退化到GSE三seed均值`2.925625°`，相对改善`-0.915508`，违反单字段不得退化超过5%的硬门槛，因此整体FAIL。
- 只读根因证据：slope teacher标准差`4.903°`，三模型预测标准差仅`0.905–1.057°`，MAE几乎等于直接预测0°的`2.914°`，说明坡度头发生近零塌缩。Teacher的axis坡度与slope字段MAE `<1e-7°`，标签不矛盾；解析基线与Teacher相关系数`0.874`，冻结学习头仅约`0.28–0.31`。
- 受影响工作已停止：不调低门槛、不发布FAIL为主结果图、不启动离线拓扑。建议的最小corrective是保留已通过的冻结主干，用train-only数据训练“解析坡度先验+小型学习残差/置信度头”；先做train-world交叉验证，不可直接用C09调正后宣称通过。

- 唯一正式run `gate2_20260824_gse_graph_three_seed_training_v1r_seed0` 已`COMPLETED/PASS_GSE_GRAPH_THREE_SEED_TRAINING_V1R`；seed0/1/2分别完成`30/27/30` epochs，最佳epoch=`24/21/24`，总optimizer steps=`256,310`。三个最佳checkpoint SHA-256为`36a826e3.../dcbe5369.../307a73ae...`，峰值GPU allocation均约`2.95 GB`，C10/M-TARE读取均为0。33文件完整seal验证PASS，seal SHA=`71014eba...`。
- 论文训练证据已原子发布到`docs/figures/gse_graph/gse_training_curves.{png,pdf,svg,csv}`，并保留summary/provenance/SHA-256；manifest SHA=`1df96c1c...`。三seed最佳checkpoint均值event macro-F1=`0.639301`、axis error=`5.242620°`、association raw top-1=`0.979191`；这只证明可学习性，不代替C09基线增益与安全关联门槛。
- seed1第5轮以validation total loss=`2.299374`成为当前总损失最佳点：event macro-F1=`0.636916`、axis mean error=`7.145348°`、association raw top-1=`0.977129`、exit-count exact=`0.132777`；width/height/slope/curvature MAE=`1.059035m/0.453681m/3.052441°/0.004800m^-1`。事件较第4轮提升约5.7个百分点，方向/关联/宽度继续改善；exit-count仍波动，不用于改变训练规则。
- seed1第6轮validation total loss=`2.316509`，未刷新第5轮checkpoint；event/axis/association=`0.591293/7.201718°/0.973964`，exit-count exact临时升至`0.293516`。该分化进一步证明不得按单项峰值拼接checkpoint，训练与early-stop继续使用预注册总验证损失。
- seed1第7轮以validation total loss=`2.230869`刷新checkpoint：event macro-F1=`0.588540`、axis mean error=`7.150757°`、association raw top-1=`0.975547`、exit-count exact=`0.085520`；width/height/slope/curvature MAE=`1.020273m/0.507435m/3.044379°/0.004991m^-1`。总损失改善但事件、出口和部分几何单项退化，多任务冲突风险仍必须由训练后冻结感知门槛判定。
- seed1第8轮继续刷新validation total loss至`2.084704`：event macro-F1=`0.618032`、axis mean error=`6.180382°`、association raw top-1=`0.975978`、exit-count exact=`0.132696`；width/height/slope/curvature MAE=`1.082285m/0.522543m/3.048722°/0.004750m^-1`。方向与事件恢复但宽高未同步改善，仍不得提前判断感知PASS。
- seed1第9轮的validation total loss进一步下降至`2.073825`，但event macro-F1回落至`0.578078`，axis error=`6.573775°`、association raw top-1=`0.976554`、exit-count exact=`0.106083`；该分化再次证明不能从不同轮次拼接单项指标。
- seed1第12轮将validation total loss降至`1.916131`：event macro-F1=`0.615186`、axis error=`5.582360°`、association raw top-1=`0.978711`，width/height/slope/curvature MAE=`0.870863m/0.405418m/3.050215°/0.004567m^-1`。这是运行中指标，不替代最终checkpoint或正式校准结果。
- seed1第13轮validation total loss回升至`2.006811`，未替换第12轮checkpoint；event/axis/association=`0.607266/6.057812°/0.978855`，width/height临时改善到`0.800540m/0.385905m`。不同单项峰值仍不拼接。
- seed1第14轮刷新validation total loss至`1.901209`：event macro-F1=`0.629513`、axis=`5.771539°`、association raw top-1=`0.977560`，exit-count exact/MAE=`0.341836/0.888725`；width/height/slope/curvature MAE=`0.865711m/0.400504m/2.974960°/0.004526m^-1`。关联仍需正式校准达到非空98% precision合同。
- seed1第15轮继续刷新validation total loss至`1.873178`：event macro-F1=`0.636291`、axis=`5.435632°`、association raw top-1=`0.980150`，首次越过98%参考线；width/height/slope/curvature MAE=`0.826974m/0.407775m/2.956920°/0.004576m^-1`。raw retrieval仍不替代正式accepted-association precision/false-accept门槛。
- seed1第16轮validation total loss回升至`1.927858`，未替换第15轮checkpoint；event/axis/association=`0.625322/5.507405°/0.978999`，exit-count exact/MAE=`0.152400/1.443627`。出口数量继续波动，不能从不同轮次拼接单项指标。
- seed1第17轮validation total loss=`1.953054`，仍未替换第15轮checkpoint；event macro-F1临时升至`0.659786`，但association raw top-1降至`0.977129`、width MAE升至`0.933920m`、exit-count exact降至`0.088014`。该轮进一步证明多任务单项峰值不能拼接。
- seed1第18轮validation total loss=`1.900204`，仍高于第15轮最佳`1.873178`；event/axis/association=`0.658129/5.432605°/0.980150`，exit-count exact恢复到`0.367182`，width/height MAE=`0.819433m/0.389003m`。整体较第17轮均衡但不满足预注册checkpoint替换条件。
- seed1第19轮validation total loss=`1.892686`，接近但仍未低于第15轮；event/axis/association=`0.614194/5.405001°/0.977848`，exit-count exact=`0.223408`。已连续4轮未刷新，仍由冻结`patience=6`自然决定是否在第21轮后早停，禁止人工提前终止。
- seed1第20轮validation total loss=`1.969796`，仍未刷新；event/axis/association=`0.625808/5.230308°/0.980150`，exit-count exact=`0.097825`。patience现为5/6；第21轮若仍无改善将按冻结规则自动早停并封存第15轮，不人工干预。
- seed1第21轮将validation total loss从第15轮的`1.873178`刷新到`1.779433`，patience重置并继续训练；event/axis/association=`0.621149/5.350217°/0.979287`，exit-count exact=`0.378138`。完整checkpoint SHA-256=`dcbe5369...`；该选择来自整体多任务损失改善，不按更高event单项轮次拼接。
- seed1第22轮validation total loss=`1.829008`，未替换第21轮；event macro-F1临时升到`0.665762`，axis/association=`5.483267°/0.979143`，width/height MAE=`0.799886m/0.376791m`。patience=1/6；事件单项峰值不用于替换完整checkpoint。
- seed1第23轮validation total loss=`1.871594`，仍未替换第21轮；event/axis/association=`0.639045/5.240878°/0.979143`，width/height MAE改善到`0.737174m/0.370622m`，但出口相关损失使整体未刷新。patience=2/6。
- seed1第24轮validation total loss=`1.934274`，仍未替换第21轮；event/axis/association=`0.637727/5.222836°/0.978711`，width/height MAE=`0.759324m/0.368474m`，exit-count exact=`0.121576`。patience=3/6。
- seed1第25轮validation total loss=`1.863588`，仍未替换第21轮；event/axis/association=`0.652013/5.189621°/0.979718`，exit-count exact=`0.223367`。patience=4/6。
- seed1第26轮validation total loss=`1.818347`，仍未替换第21轮；event/axis/association=`0.674656/5.046064°/0.978136`，width MAE=`0.722421m`、exit-count exact=`0.307089`。patience=5/6；单项较强但完整checkpoint仍由总损失决定。
- seed1第27轮validation total loss=`1.783899`，未低于第21轮`1.779433`，触发冻结`patience=6`早停。Seed1正式完成27 epochs/79,544 optimizer steps，best epoch21、checkpoint SHA=`dcbe5369...`，validation outputs与summary完整，C10/M-TARE读取0；outer runner已无人工干预启动seed2。
- seed2第1轮完成：validation total loss=`3.158622`、event macro-F1=`0.521960`、axis error=`9.622934°`、association raw top-1=`0.963320`，width/height MAE=`1.216857m/0.440727m`，exit-count exact=`0.234854`。全部loss有限、无NaN/OOM；第三个独立初始化的axis已从随机方向水平显著下降，圆周等变修正继续成立。
- seed2第2轮刷新validation total loss到`2.800334`；event macro-F1=`0.558857`、axis error=`8.636085°`、association raw top-1=`0.967060`。相对首轮三项同时改善，独立seed继续正常收敛；出口数量单项波动不改变完整checkpoint选择。
- seed2第3轮继续刷新validation total loss到`2.516426`，但event macro-F1=`0.546815`、axis error=`8.934481°`暂时回落，association raw top-1升至`0.970512`。完整checkpoint按总损失更新，不把第2轮事件/方向与第3轮其它头拼接。
- seed2第4轮进一步刷新validation total loss到`2.473619`，且event macro-F1升至`0.595437`、axis error降至`7.115294°`，association raw top-1=`0.970081`。结构事件与方向在同一完整checkpoint上共同改善，继续正常收敛。
- seed2第5轮继续刷新validation total loss到`2.444190`；event macro-F1=`0.631743`、axis error=`6.738311°`、association raw top-1=`0.972670`。事件、方向和地点检索在同一完整checkpoint上继续同步改善。
- seed2第6轮再次刷新validation total loss到`2.318364`；axis error降至`6.359645°`、association raw top-1升至`0.974108`，但event macro-F1回落到`0.550858`。完整checkpoint仍按预注册总损失更新，禁止把第5轮事件头与第6轮其它任务拼接。
- seed2第7轮继续刷新validation total loss到`2.206158`；event macro-F1恢复到`0.613994`、axis error降至`6.295280°`、association raw top-1升至`0.975115`，width/height MAE改善到`0.987149m/0.404135m`。这是seed2当前最均衡的完整checkpoint，仍需自然完成训练和正式校准。
- seed2第8轮再次刷新validation total loss到`2.125442`；event macro-F1升至`0.633661`、exit-count exact升至`0.355899`、association raw top-1=`0.975259`、width MAE=`0.974782m`，但axis error回升至`6.605238°`。完整checkpoint按总损失更新，方向单项回落原样保留。
- seed2第9轮validation total loss=`2.147576`，未替换第8轮checkpoint，patience=`1/6`；event macro-F1=`0.638935`、axis error=`6.208579°`，但association raw top-1降至`0.973820`、exit-count exact降至`0.270379`。较强事件/方向单项不用于拼接或覆盖完整最佳点。
- seed2第10轮显著刷新validation total loss到`1.977626`并重置patience；event macro-F1=`0.641389`、axis error=`6.004109°`、association raw top-1=`0.975259`、curvature MAE=`0.004431m^-1`。exit-count exact仍只有`0.229049`，出口集合稳定性继续作为正式C09风险保留。
- seed2第11轮validation total loss=`2.060034`，未替换第10轮checkpoint，patience=`1/6`；axis error继续改善到`5.834826°`，但event macro-F1回落到`0.587724`、association raw top-1降至`0.973677`。方向单项改善不用于覆盖第10轮完整最佳模型。
- seed2第12轮validation total loss=`1.991636`，仍略高于第10轮最佳`1.977626`，patience=`2/6`；association raw top-1升至`0.978136`、width MAE降至`0.924664m`，但event macro-F1仅`0.594186`、exit-count exact降至`0.168057`。第10轮继续作为完整checkpoint。
- seed2第13轮刷新validation total loss到`1.914080`并重置patience；event macro-F1=`0.630552`、axis error=`5.574589°`、association raw top-1=`0.976266`，width/height MAE改善到`0.852062m/0.378183m`。exit-count exact仍为`0.234977`，出口集合风险继续保留。
- seed2第14轮validation total loss=`1.987991`，未替换第13轮checkpoint，patience=`1/6`；event macro-F1=`0.637528`、axis error改善到`5.492206°`、association raw top-1=`0.977417`，但exit-count exact仍只有`0.239760`。单项方向/事件改善不覆盖完整最佳点。
- seed2第15轮validation total loss=`1.943111`，仍未替换第13轮checkpoint，patience=`2/6`；event macro-F1=`0.642152`、width/height MAE改善到`0.788725m/0.366961m`，但exit-count exact跌到`0.132573`、curvature MAE升至`0.005265m^-1`。多任务分化原样保留。
- seed2第16轮validation total loss=`1.992618`，未替换第13轮checkpoint，patience=`3/6`；event macro-F1升至`0.647999`，width/height/curvature MAE=`0.812686m/0.368715m/0.004362m^-1`，但exit-count exact仅`0.155874`。较强事件与连续几何仍不足以改善完整多任务选择值。
- seed2第17轮validation total loss=`1.943679`，仍未替换第13轮checkpoint，patience=`4/6`；exit-count exact恢复到`0.335950`、width MAE=`0.794678m`，但event macro-F1回落到`0.622285`、association raw top-1=`0.976266`。若后续两轮均不刷新将按冻结规则自然早停。
- seed2第18轮刷新validation total loss到`1.894117`并重置patience；axis error降至`5.395400°`，width/height MAE=`0.776217m/0.367451m`。event macro-F1仅`0.608687`、association raw top-1=`0.975834`、exit-count exact=`0.232647`，完整最佳改善但结构/关联风险仍需正式校准。
- seed2第19轮validation total loss=`1.908076`，略高于第18轮最佳，patience=`1/6`；association raw top-1升至`0.978136`、exit-count exact=`0.328469`、width MAE=`0.759492m`，但event macro-F1仅`0.618330`。完整checkpoint保持第18轮。
- seed2第20轮validation total loss=`1.987336`，仍未替换第18轮checkpoint，patience=`2/6`；event macro-F1=`0.643913`、axis error=`5.420641°`、association raw top-1=`0.977848`，width/height/slope/curvature MAE=`0.759705m/0.363049m/2.951366°/0.004322m^-1`，但exit-count exact仍仅`0.218216`。
- seed2第21轮validation total loss=`1.904648`，略高于第18轮最佳，patience=`3/6`；event macro-F1升至`0.651376`、axis error降至`5.179517°`、association raw top-1=`0.978855`，但exit-count exact仅`0.141771`。三项强单项仍不替换完整checkpoint。
- seed2第22轮刷新validation total loss到`1.864445`并重置patience；event macro-F1=`0.647545`、axis error=`5.170332°`、association raw top-1首次升至`0.981732`，width/height MAE=`0.786814m/0.365834m`。exit-count exact仍为`0.235672`，且raw retrieval不替代正式安全关联校准。
- seed2第23轮validation total loss=`1.909927`，未替换第22轮checkpoint，patience=`1/6`；event macro-F1升至`0.661990`、axis error=`5.147214°`、width MAE降至`0.729782m`，但association raw top-1回落到`0.977848`、exit-count exact=`0.240332`。强事件/几何单项不覆盖完整最佳模型。
- seed2第24轮将validation total loss刷新到`1.860350`并重置patience；event macro-F1=`0.667571`、axis error=`5.168762°`、association raw top-1=`0.978999`，width/height MAE=`0.719964m/0.363890m`，exit-count exact明显升至`0.386763`。该轮为seed2当前最均衡完整checkpoint，仍需自然完成训练与正式C09校准。
- seed2第25轮validation total loss=`1.908971`，未替换第24轮最佳checkpoint，patience=`1/6`；event/axis/association=`0.658767/5.283594°/0.978855`，exit-count exact回落到`0.272954`、width MAE回升到`0.776471m`。第24轮继续作为seed2完整最佳点。
- seed2第26轮validation total loss=`1.908415`，仍未替换第24轮最佳checkpoint，patience=`2/6`；axis error和width MAE单项改善到`5.044488°/0.703888m`，但exit-count exact降到`0.166013`，event/association=`0.664378/0.978567`。单项几何改善不覆盖更均衡的第24轮。
- seed2第27轮validation total loss=`1.888253`，仍未替换第24轮最佳checkpoint，patience=`3/6`；axis error首次降到`4.925422°`，width MAE=`0.705142m`、association raw top-1=`0.979862`，但exit-count exact仅`0.201374`，event macro-F1=`0.653898`。单项方向突破不替换完整checkpoint。
- seed2第28轮validation total loss=`1.907361`，仍未替换第24轮最佳checkpoint，patience=`4/6`；axis error继续降到`4.911525°`，但event macro-F1和association raw top-1回落到`0.642624/0.977273`，exit-count exact=`0.249285`。还剩最多两轮，训练规则不变。
- seed2第29轮validation total loss=`1.969113`，未替换第24轮最佳checkpoint，patience=`5/6`；event/axis/association=`0.656517/5.030770°/0.977992`，exit-count exact=`0.225452`。只剩第30轮；若不刷新将同时到达patience和训练上限。
- 当前最低validation total loss为第24轮`1.786166`：event macro-F1=`0.629182`、axis mean error=`5.208880°`、association raw top-1=`0.979287`；width/height/slope/curvature MAE=`0.769008m/0.376695m/2.903517°/0.004352m^-1`。多任务单项持续波动，最终仍严格按最低validation total loss选checkpoint，不按单项最好轮次拼接。
- exit-count exact accuracy在第24轮为`0.182814`，仍是主要科研风险；不得用训练中间结果改loss或checkpoint规则。最终由冻结的exit presence/token set、事件、连续几何与关联校准共同决定是否继续。
- 训练结束后的validation-only Data Card已冻结为10个C09 world、24,462条五帧序列、32,678个唯一帧；公平对照为历史M1D三seed第五帧推理和当前帧非学习几何估计。事件、几何、关联的科学门槛在正式结果前已预注册。
- 正式感知验证runner、校准、基线、汇总、论文图发布器均已实现。离线拓扑现也已实现完整GSE、GSE规则关联消融、exit-only规则图、非学习几何规则图、GT-TNG诊断和固定243组选择；科学门槛要求node/edge F1各自比对应最强基线至少高0.05，且关联和图不变量同时安全。
- 感知证据现补齐独立failure/rejection publisher：从sealed C09 PASS读取全部三seed事件拒绝、place与exit causal-association完整曲线，标记冻结选择点并报告错误接受率；输出PNG/PDF/SVG/CSV/JSON/provenance/manifest，任何曲线漂移、C10/M-TARE读取或覆盖均fail closed。publisher已绑定尚未冻结的正式spec工具哈希，专项联动测试`118/118 PASS`。
- 公平性审计发现旧M1D报告20°出口方向匹配F1/角误差/数量，而GSE原校准只报告query presence和独立assignment误差，不能直接比较。现于正式C09前为GSE增加相同20°匹配诊断，并新增三seed配对exit-token figure publisher，固定报告direction F1、matched angular error、count exact与count MAE；无匹配时角误差保留为缺失而不伪造0。该诊断不进入既有感知PASS门槛。两套共享方向metric及publisher均进入spec哈希，GSE专项`121/121 PASS`。
- uncertainty证据现与其真实训练定义对齐：它是几何残差尺度，但部署时也参与事件拒绝。新诊断固定对全部validation观测作10个等频bin，比较预测`u²`与训练同定义masked normalized Smooth-L1几何残差，并并列event error rate；它明确`selection_effect=NONE`，不改checkpoint、阈值或gate。独立publisher将保存三seed完整bin、PNG/PDF/SVG/CSV/JSON/provenance/manifest。
- 新增五类事件论文表发布器：强制保留M1D与GSE全部3个seed、5类事件的precision/recall/F1/support，输出30行原始CSV及聚合Markdown/LaTeX、机器源、provenance与SHA-256；只重放冻结选择点，禁止制表阶段重新挑阈值。正式表仍为0，必须等待sealed C09 PASS。
- 六套C09论文证据现由一个原子发布器统一提交：总体感知、拒绝分析、出口token、不确定性、逐世界泛化和五类事件表必须全部完成才写总索引与总SHA-256。Luna只读审计指出静态目标回滚不能覆盖意外额外文件；现改为发布前后目录快照差分，每个子包实际新增集合必须精确等于声明，异常时只删除本次全部新增文件，并由总发布器独立逐项复核C09 seal。正式C09尚未执行，当前没有生成结果图表。
- 新增逐C09世界诊断：完整保留10个未见topology parent×3个seed的冻结事件增益，以及4项连续几何相对非学习估计器的改进；全局温度/拒绝阈值原样重放，所有世界、seed和字段必须入图，`selection_effect=NONE`。Luna复核后发布层又增加逐parent阈值元数据等于全局冻结点、四字段paired valid-count完全相等两项硬门禁；发布包固定PNG/PDF/SVG/CSV/JSON/provenance/SHA-256，不能用容易世界或不同有效子集抬高均值。
- GSE相关测试`151/151 PASS`，其中感知总发布器覆盖精确43文件计数、错误gate、C10/M-TARE读取、source seal漂移和意外partial文件回滚；逐世界发布器另覆盖完整10×3人口、parent/逐seed帧数、selection-effect、冻结阈值、paired valid-count漂移、unsealed source与额外未封文件拒绝。更宽的可运行V3 unit最近一次为`806 PASS / 3个历史无关FAIL`，本轮未重复借此声称全仓PASS。
- 离线baseline文字合同已纠正：M1D和感知连续量基线只看第五帧；非学习几何事件图与GSE公平使用同一五帧过去历史，但逐帧只运行确定性几何、无学习descriptor/uncertainty。旧spec中的`current-frame geometry events`会误导为整个事件图仅单帧，现已修正；代码、数据和网格未变。
- 离线拓扑来源门禁经Luna审计后加强：每个source seal路径必须位于自身run目录，非空、无重复/自引用，并精确覆盖目录内除seal外的全部文件；evaluator与outer runner均独立检查四个来源的run identity、PASS状态及C10/M-TARE零读取。新增越界、空seal、漏列、identity和dataset直接调用负测。
- 新门禁已直接通过两个真实封存来源：去重数据集`33,083/33,083`文件、seal `18acade5...`，Teacher manifest `17/17`文件、seal `1c2cb6bf...`，全部哈希及精确目录覆盖PASS；分别耗时6.35s/0.08s。训练/perception尚未完成，未伪装为已验证来源。
- 每个离线方法现执行后重读并强制`parameter_sweep.jsonl=243`行，selected config/nodes/edges/decision/summary文件集合必须精确。四方法预计world-config-seed replays固定为`7290/7290/7290/2430=24300`，总观测更新`59,442,660`；通用sweep合成证明实际调用数、243行和选定文件集完全匹配。
- 离线拓扑PASS后的两套论文图也已绑定原子发布：定量总图与固定拓扑示例各7文件，加总索引/manifest共16文件；任一子图多写、少写或失败均回滚本次新增集合，既有论文图不动。正式离线结果尚未运行，当前没有生成这些图。
- 三seed训练PASS后的学习曲线也改由完整seal验证包装器发布：固定7文件必须精确齐全，子publisher异常或意外输出时回滚本次新增集合。正式训练仍在运行，当前尚未生成最终训练曲线。
- 感知Data Card已通过标准治理校验并改用合法Gate 3 threshold-calibration身份；离线拓扑Data Card、一次性runner和论文图publisher也已就绪。两个fail-closed spec freezer只在上游sealed PASS后绑定精确工具/输入哈希，当前上游未完成时均拒绝且0输出，可在阶段交接时避免人工停顿。正式图固定保留PNG、PDF、SVG、CSV、完整JSON、provenance和SHA-256。C10和M-TARE继续封闭，直到感知和243组离线图参数全部冻结。
- 离线结果除汇总柱状图外，还预注册固定`S01/S06/S10 C09 + seed0 + 全四方法`的拓扑示例图；显示objective虚线图、正确/错误预测边、未映射节点和高度着色。evaluator现保存每个Teacher结构节点的客观坐标，publisher只画压缩后的结构图，不把被正常抑制的metric anchor误标为错误。合成sealed证据已验证完整七文件图包可生成。
- 旧论文交付矩阵已改为GSE-Graph主线；旧出口模型与规则图保留为基线/消融。正式论文图固定保留PNG、PDF/SVG、机器可读源数据、生成器、provenance与SHA-256。
- 最新原始来源新颖性复核新增PRISM-TopoMap与AIM-Mapping：前者学习地点识别并维护在线location graph，后者学习栅格结构表示与多机策略，但均未同时采用因果LiDAR显式隧道几何事件、语义决定结构节点/出口关联和真实穿越建边。它们已加入直接相关工作矩阵，当前主张继续但投稿前仍须更新检索。
- 同日追加Semantic Topometric Mapping与2025 RA-L Heterogeneous Topological Graph Exploration：它们分别从累计二维occupancy grid规则分割结构区域、从二维grid/ESDF确定性生成异构节点后学习策略。论文已明确不再把“结构语义+拓扑探索”泛称为创新，核心边界收窄为因果LiDAR学习的三维隧道几何事件直接决定节点/关联且edge只由真实穿越确认。
- 2026年7月OVTG也已补入直接边界：它学习开放词汇视觉语义并在图上用GAT对齐，但节点仍由空间新奇度触发、edge仍由时序/邻近无碰撞关系建立。当前四机制组合主张未被同构覆盖，贡献矩阵、正文与BibTeX已同步，引用完整性PASS。
- 训练等待期间从冻结模型与Data Card补齐论文实现细节：803,058个可训练参数、128维GRU/descriptor、6-query 8-head出口set、任务权重/归一化、确定性pair-aware batch64、AdamW/BF16/gradient clip与三seed checkpoint选择。独立逐项复核后，event logits/softmax、cosine axis surrogate、归一化类权重、uncertainty variance和`1e-8`早期tie规则均已精确写明；六项问题全部关闭，参数量由模型代码实例化复核，正文引用键全部存在。
- 论文实验协议已从冻结C09 evaluator补齐：event temperature/NLL、联合confidence-uncertainty拒绝、exit presence F1、parent内past-only place/exit关联、非空98% precision/1% false-accept约束、第五帧M1D映射、当前帧PCA+五截面几何对照和原始/相对MAE门槛。没有读取正式结果、C10或M-TARE。
- 论文离线图段落完成独立代码审计并纠正过度主张：当前只实现C09离线event graph，GSE global selector/M-TARE adapter/多机交换明确为pending；edge由预记录Euler traversal trace确认。正文计数改为188126/24462五帧序列和252430/32678 unique frames，并精确写明event/anchor、association、edge字段、243 structural grid、两个主baseline及pending ablations。八组问题复核关闭。
- NEXT：让seed1/2自然完成并封存三seed；随后自动执行一次validation-only感知资格。只有感知门槛PASS才进入固定243组离线拓扑回放。

## 2026-08-24：GSE训练V1因方位不变量axis head停止，V1R圆周等变修正就绪

- V1正式run在seed0第1轮后主动停止并封存：event macro-F1从随机初始化`0.166545`升到`0.530559`，width/height/slope/curvature MAE均有限，但axis mean angular error仍为`79.7923°`，不能支持“学习局部结构轴”主张。
- 根因是axis head和exit attention读取全方位平均/无位置矩的特征；训练roll旋转robot-frame方向target时，原head在结构上保持旋转不变，因此目标不可学习。该问题属于model interface，不影响sealed dataset或Teacher。
- V1保持`FAIL_GSE_GRAPH_THREE_SEED_TRAINING_V1`，只完成seed0 epoch1，不复用partial checkpoint；15项seal SHA-256=`494cfedb9a68aa4f58552d9aa730f238781c65bf0d482b7cf5fd242ba0d58824`。
- V1R只做三项修正：保留每个azimuth的五帧时序特征；用圆周概率矩输出axis、用attention圆周矩输出exit heading；train-only roll限制为编码器精确4列stride lattice。数据、Teacher、mask、loss权重、归一化、optimizer、epoch、patience、batch和validation selection不变。
- 新增2° circular roll等变回归测试并PASS；完整GSE合同测试为66项PASS。修正后真实batch无更新proof仍为64样本、8个正association identities、全部loss/backward finite、2.68 GiB峰值显存、权重不变。
- NEXT：冻结V1R spec并从随机初始化执行seeds 0/1/2；第1轮首先核验axis是否真正下降，禁止用V1 partial初始化。

## 2026-08-24：GSE去重LiDAR与完整多任务Teacher正式导出PASS

- 唯一正式run完成90/90 development worlds：80 train、10 validation；精确保存285,108个唯一16×720 LiDAR帧、212,588个五帧因果序列和1,062,940个引用，没有把五帧payload重复物化。
- 数据接口同时封存robot-frame local axis、width/height/slope/curvature、五类结构事件、5,087个place association identities、17,616个directed exit identities和119,390个关联对。模型输入实际读取为`[5,2,16,720]`，不包含pose或yaw。
- 448,338个incident candidates对应448,279个LOS-visible student tokens，其中438,968个具有opening-width监督；缺失几何、association和width均由独立mask表达，不删除样本、不用解析半径回填。
- 共投射3,284,444,160条full-scan rays；原始数组16,523,011,520 bytes，压缩后90个Zarr shards为7,988,637,782 bytes。执行171.962秒，runner墙钟173.08秒，峰值RSS 1,455,200 KiB，远低于3小时/8 GiB/24 GiB合同。
- 33,083项seal逐文件独立复核PASS，seal SHA-256=`18acade5204258ebc6a822d24bc7910c1464bda7843a2e2e90e7f775c4cdb628`；train/validation首中末样本均由训练读取器成功重建，C10/M-TARE/模型推理/训练/optimizer均为0。
- 论文数据图`docs/figures/gse_graph/gse_dataset_overview.{png,pdf,svg}`已生成并目视检查，同时保存CSV、summary、provenance和SHA-256 manifest；publication manifest SHA-256=`2abdde55475699113122bf8d2337730604d6c2b690f13c2621a68fb8dbeb5c71`。
- NEXT：冻结三seed训练合同和validation-only选择规则，先做无更新真实batch读入/forward/backward证明，再执行seeds 0/1/2训练；严格C10与M-TARE仍禁止读取。

## 2026-08-24：incident-only出口几何Teacher正式审计PASS

- 新增类型化物理出口candidate、route representative、robot-relative heading、四点vertical profile、native-mesh opening width mask和first-return LOS；corridor/turn/transition只保留当前physical edge双向，junction/terminal只保留所选node incident edges。同tunnel不同physical edge禁止折叠。
- V1在任何world/ray之前因冷启动循环导入失败，保持`FAIL_GSE_EXIT_TOKEN_AUDIT_V1`；目录420 KiB、10项seal，seal SHA-256=`d9464ea65343c0be228e0e8ca7b87f2247c1322e7bfc7ac77f3177f2e9273c34`。分类为system/module dependency，不影响数据或Teacher语义。
- V1R只把`oriented_edge_polyline`改为函数内延迟导入并增加cold-process回归测试，数据、offset、LOS margin、mask、阈值及资源合同全部不变；14项专项测试和合计54项GSE合同测试在对应geometry/Torch sidecar中PASS。
- 正式V1R完成90/90 worlds、212,588 observations和448,338 physical candidates；9,415,098 native-mesh rays在211.155秒内完成。nonincident candidates=0，zero-visible junction/terminal observations=0，29,695个同tunnel多physical-edge观测完整保留。
- 448,279/448,338 candidates可见（99.98684%）；439,027/448,338 width有效（97.92322%）。59个不可见candidate和9,311个width-invalid candidate均按合同显式mask，没有删除观测、回填半径或改变身份。
- 13/13 seal独立复核PASS，seal SHA-256=`c2a3299933c2f6b564c4593aff39b1896fc5a5611e70f35b414f0684bb9c3a33`；C10/M-TARE/模型/训练/optimizer均为0。
- 论文图`docs/figures/gse_graph/gse_exit_token_audit.{png,pdf,svg}`已从sealed PASS自动生成并目视检查，同时保存CSV、summary、provenance和SHA-256 manifest；publication manifest SHA-256=`2e198874ad0e7838508f5c531fc649d5b24716f043b66befa7178f36eb5117a1`。
- NEXT：冻结并执行285,108 unique 16×720 LiDAR frames、212,588完整多任务Teacher和five-frame reference的去重数据导出；不允许重复存储五帧payload。

## 2026-08-23：GSE native-mesh 几何教师恢复五类结构事件

- sensor export pose/batching接口已实现：从sealed traversal manifest重建axis/sensor/yaw，短边保留identity但不产生孤立帧，按32帧batch生成冻结16×720 range/valid。exit token现新增typed incident-only candidate：node事件只保留selected node的physical edges，corridor保留current edge正反方向，同tunnel不同edge不折叠，stacked/nonincident edge排除；专项及GSE联动`48/48 PASS`。尚未提前导出15.3GiB不完整训练数据。新增`docs/GSE_EXIT_TOKEN_TEACHER_CONTRACT_V1.md`，NEXT为native-mesh LOS、opening-width mask、代表点与四点vertical profile全量审计。
- 完整development teacher manifest正式PASS：90 worlds/9,066 edges/18,132 directed traversals/212,588 observations/285,108 unique frames/1,062,940 references精确一致，两个1.515895m短traversal保留identity且为0序列。生成119,390 association pairs；train结构观测53,003，positive anchors 52,465（其中cross-traversal 52,174），hard-negative anchors 52,962；validation结构观测7,011，positive 6,952，hard-negative 7,011。17/17 seal独立复核，seal SHA-256=`1c2cb6bf3ebdf24f21a105cced59fb49066659bd6120667974276b940661b596`，结果173MiB，C10/M-TARE/推理/训练均为0。
- 去重sensor payload精确为`16,422,220,800 bytes=15.2944GiB`，五索引仅8,503,520 bytes；禁止的逐序列重复payload为`61,225,344,000 bytes=57.0205GiB`。NEXT改为冻结唯一帧分片、teacher label/index、逐shard hash与资源/失败合同后再执行sensor export。
- V1R正式proof在57.16秒内PASS并完成独立15/15 seal复核，seal SHA-256=`f4a7002a322ac68af88b5246dac178c41ad1677bb84054621a8e99c89616210f`。精确审计80 worlds/8,039 edges/16,078 directed traversals/188,126 sequences/252,430 unique frames；事件corridor/junction/terminal/turn/geometry-transition=`135,123/26,608/7,525/2,003/16,867`，width/height有效率`99.1718%`，非junction缺测率`0.08977%`，全部门槛PASS，validation/C10/M-TARE/推理/训练/optimizer均为0。
- 论文图已目视检查并通过fail-closed发布器写入`docs/figures/gse_graph/gse_teacher_distribution.{png,pdf,svg}`，同时保留CSV、summary JSON、source-run provenance和独立manifest；补齐SVG后的论文图manifest SHA-256=`e5e2e45f5d5630b339916272cd51fce4d2e19db76cc3cbc8c6645acd25df82b2`。V1失败图未发布。
- 首次80-world正式proof在第62个训练世界`S08_3d_loop_rich_C06`按合同FAIL并封存，seal SHA-256=`184b927d1789487a4c190e87b23b8a57aa2af70d3313487c1718071ccffdeaaf`。唯一根因是1.515895m短边`edge_0022`的正反traversal按五帧/4m历史合同应为0序列、0帧，但mesh inventory先创建了各2个未被任何序列引用的孤立帧；全部18,132 traversals中仅这2个为零序列。已改为直接复用`deduplicated_frame_arcs_and_sequence_indices`，不删edge/world、不改188,126训练序列、252,430唯一帧、teacher或阈值；短边专项及GSE联动测试`42/42 PASS`。V1R corrective Data Card/spec冻结15项hash零漂移并已按新不可覆盖run执行。
- 旧 exit-only + deterministic-graph V5基线已自然完成`30/30`，顶层状态`PASS_AEE_COMPOSITE_V9_COMBINED_CORRECTION_STOCHASTIC_V1R2`，18,000仿真秒、10个区组、三个checkpoint各10例，训练/optimizer/C09/C10均为0；496项seal SHA-256=`ce2bcd173528ce969dbe00e35a6dbc074164b61b12e0852beeb3a621f7201ca9`。该run只作为基线/消融，不升级为GSE主方法。

- 原解析教师在全部 development inventory 上只有 `20 turn / 0 geometry-transition`，原因是 per-tunnel radius 在同一隧道内恒定，无法表示真实 mesh 横截面变化；受影响的正式 teacher export/training 已停止，未生成样本或 checkpoint。
- 新增 route-frame native-mesh 教师：沿左右/上下各 5 条窄扇区射线取投影中位数，得到 width/height；相邻 5 m 横截面中位数变化至少 1 m 时标为 geometry-transition。turn 阈值由仅训练集的 spline 分布审计从无有效远离节点样本的 35° 固定为 15°；validation/C10/M-TARE 均零读取。
- 10 个训练 C01 世界（S01--S10 每类一个）只读真实 mesh 哨兵包含 `23,318` 个五帧序列，五类事件为 corridor/junction/terminal/turn/geometry-transition=`17,340/3,136/867/206/1,769`，证明事件类别不再退化。
- `23,053/23,318=98.8635%` 序列有完整 width/height；265 个缺失中 253 个是 open junction、12 个是 corridor mesh 缺口。合同改为显式逐维 mask：不删除样本、不用 tunnel radius 补造标签，event/axis/slope/curvature/association 仍监督。
- 新增 mesh geometry、batch raycast、transition profile、mesh teacher inventory，以及把正反 traversal 映射到同一 physical turn/transition 的 canonical event identity。place descriptor 与匹配后的 exit-token descriptor 现在都受监督式对比损失；在线关联联合检查 event、place descriptor、出口 heading/descriptor/opening width/vertical profile 与空间距离，困难负样本按同事件几何近邻确定性选择。exit token 教师已明确为 directed physical edge identity、mesh opening width 和沿出口 2.5/5/7.5/10 m 的四点高度变化。数据存储采用 285,108 个唯一帧加五索引序列，禁止把 1,062,940 次引用重复物化。GSE 与论文图发布合同相关测试当前 `41/41 PASS`。
- NEXT：冻结18,132 traversal manifest、关联identity/困难负样本计数与去重存储资源估计；随后建立正式teacher-export Data Card/spec。在完整manifest和资源上界封存前不生成15.3GiB传感器数据，也不训练。
- 80-world proof 的 Data Card/spec、CPU/Open3D executor、one-shot runner、100 MiB结果上限和论文图合同已冻结；11项工具/输入hash一致。旧run封存后current gate已切换到GSE Gate 2，正式proof会保留80-world JSONL、CSV、PNG、vector PDF、原始日志和seal；新增的 fail-closed 发布器只接受完整PASS且已seal的80-world来源，并把PNG/PDF、CSV/JSON、来源与独立hash manifest保留到论文图目录。

## 2026-08-23：论文主线切换为 GSE-Graph

- 唯一目标已锁定为：从 5 帧因果 LiDAR 学习显式几何结构语义，并由语义直接生成/关联在线 topometric graph；完成单/多机器人闭环、严格测试、消融、复现包和投稿 PDF。
- 旧 V5 30-case 不终止、不修改，当前作为 exit-only + deterministic graph 基线自然收尾；最新只读状态为 `22/30` 完成且逐例系统 PASS，第 23 例运行中。run-level summary/seal 尚不存在，不能提前写性能结论。
- 已确认可复用数据为 80 train / 10 validation / 10 strict-test topology parents；封存单帧 train/validation 为 `100000/12500` 帧。新 5 帧序列数量必须由 directed-traversal 只读 inventory 精确计算，C10/M-TARE 在冻结前零读取。
- 新主方法、对照、消融、门槛、清理和论文图保留合同已写入 `docs/GSE_GRAPH_RESEARCH_PLAN_V1.md`；相关工作差异写入 `docs/GSE_GRAPH_CONTRIBUTION_MATRIX_V1.md`。
- 当前 NEXT：在不干扰活跃旧 run 的前提下，实现 GSE 类型化 observation/token、5 帧模型输出合同和序列 inventory；随后冻结正式 Data Card。论文图统一保留 PNG + 矢量源 + 生成脚本 + 机器可读数据 + hash。

## 2026-08-23：GSE 第一批方法合同与序列 inventory 完成

- 新增类型化 `GeometricSemanticObservation` / `ExitGeometryToken`，覆盖五类结构事件、局部轴、宽高、坡度、曲率、place descriptor、exit token set 与 uncertainty。
- 新增 5 帧 causal range-image + GRU + exit-set attention 模型接口；输入严格为 `[B,5,2,16,720]`，没有 pose、GT identity、完整图或未来帧入口。
- 新增 objective spline/TNG teacher primitives；事件优先级和结构尺度在训练前固定。新增 uncertainty-aware GSE graph：学习事件触发节点，descriptor + exit set + spatial gate 关联；多义关联建立 provisional node，edge 仅由多帧正行程 physical trace 生成并保存几何属性。
- 只读 90-world inventory 精确得到 `9066 edges / 18132 directed traversals / 212588 five-frame sequences / 285108 unique frames / 1062940 referenced frames / 276028.677900m`，train/validation 序列为 `188126/24462`，C10/M-TARE 读取为0。
- 模型已补齐 event、axis、metric geometry、set-token matching、supervised association 与 uncertainty 联合损失；评价器已固定 event macro-F1、连续量 MAE、association precision/false merge、node/edge F1、连通分量与 cycle-rank 误差。21/21 新合同测试 PASS。设计 Data Card 已保存为 `configs/v3/gate2/data_cards/gse_graph_causal_sequence_design_v1.json`。它当前只授权 inventory 与实现，正式 teacher export/training 仍需完整 18132-record manifest、资源估计和 operation-bound spec。

更新时间：2026-08-23  
当前 Phase：`PHASE_6_90_CASE_SINGLE_ROBOT_MATRIX`  
当前结论：V5 30-case配对矩阵V1R2已完成18/30且全部系统PASS，第19例运行中。修正版持续建图和移动，但效果明显依赖场景：已完成车库案例覆盖最高5242.875m³，部分隧道案例仅478.000--1719.625m³并有高重复行驶。所有负结果原样保留，0训练/C09/C10、未调参；总体优劣等待30/30封印后的十区组配对统计。论文主线阶段汇报见`docs/STRUCTURAL_TOPOLOGY_PAPER_MAINLINE_PROGRESS.pdf`。

## 2026-08-23：V5六例readiness V1R正式PASS

- 首次V1 runner在任何Gazebo/case启动前因缺少`_bootstrap`导入报`ModuleNotFoundError: mtare_topo`；completed cases=0，10/10失败证据封存，seal SHA-256=`b2d8520ca3e7405c0fe0c9a6b66304caf9b142520fa19cdf3360c06b67da042d`。该系统入口失败不评价V5。
- V1R只增加项目统一`tools/v3/_bootstrap.py`初始化并从0/6开始；清空`PYTHONPATH`的真实宿主导入、15/15联动测试、失败源全seal和最终preflight均PASS，不复用V1资产。
- 正式V1R 6/6 PASS，共5406 trace cycles。每例节点`8--20`、verified edges`7--20`、移动`134.747973--274.716819m`；最大post-warmup fallback=`0.0215664`，全部0 failed planner cycles。
- 6/6触发frontier execution rejection并在修正后继续route growth；5/6同时触发verified re-anchor。训练/optimizer/C09/C10均为0。110/110 seal复核PASS，seal SHA-256=`d301f54c1129af5e25a6a0e7d6cc7269ef6f71aa9987e08affaf66b171a8387d`。
- readiness只证明跨两场景/三checkpoint可运行，不回答性能优劣。NEXT为冻结30个V5案例、10 blocks×3 checkpoints、每例600秒的精确配对矩阵。

## 2026-08-23：V5 30-case V1在0/30证据接口FAIL，V1R完整pre-entry proof PASS

- V1在2.949秒、任何Gazebo/case启动前因组合校验返回值缺少`source_schedule_file_sha256`而FAIL；completed cases=0、训练/C09/C10=0。9/9 seal复核PASS，seal SHA-256=`93bdb0de2cc618fb2e38e5bfb123a4a5626704e8b2ceebbbf750254903537c98`。
- V1R只把已经seal验证的`source_manifest.json`文件SHA与canonical-content SHA透传给外层schedule audit；没有改变30例身份、顺序、算法、阈值、时长或数据。16/16联动测试通过。
- 新增真实只读完整pre-Gazebo proof：16项工具、30 cases、10 blocks、checkpoint 10/10/10、source 22+8、audit/readiness/failed-V1 seals、Docker image、planner binary和30条case命令全部PASS；schedule file/content SHA分别为`9f28acd9...cbc8`/`68127f87...ec2d`。NEXT为唯一V1R正式执行。

## 2026-08-23：V5 30-case V1R在0/30 case-executor绑定FAIL，V1R2 actual-main proof PASS

- V1R越过全部来源/schedule检查后，在首个case执行入口因V2 archive adapter未重导出V1 `run_case`而FAIL；0/30、0 Gazebo、训练/C09/C10=0。15/15 seal复核PASS，seal SHA-256=`749ade2608d07d077e7525fc312d9b9a0fb3cd53870441a1aa16a9f16e1a5ba8`。
- V1R2只把冻结且已用于原90-case的`run_mtare_single_robot_stochastic_v1.run_case`显式绑定到V2 adapter，不改变任何case或科研逻辑。
- 强化proof实际调用V1R2 `main()`并到达首个`mtare-v9-v5-000` case executor，由sentinel在Docker/Gazebo前主动截断；临时失败被正常RUN_STATE+seal封存。`PASS_ACTUAL_MAIN_TO_FIRST_CASE_EXECUTOR_SENTINEL`证明完整执行调用链已连通。NEXT为唯一V1R2正式执行。

## 2026-08-23：V5 case052 V1R2 live probe正式PASS

- V1R2从frame zero重跑冻结的`tunnel/env11/checkpoint0` 180秒案例，不复用V1R的22帧partial；唯一代码差异是Python3.8数值等价兼容替换。21/21联动测试、最终preflight与32项冻结工具hash均PASS。
- 正式结果为`PASS_AEE_COMPOSITE_V9_COMBINED_CORRECTION_PROBE_V1R2`：901 trace cycles、16 graph nodes、15 verified edges、199.881216m移动、0 planner failure、0 post-warmup fallback。出口执行拒绝修正共4次，首次frame51，frame52即出现修正后route增长，最终route arc=204.989750m。
- 本例verified re-anchor计数为0，不构成失败：预声明合同只要求至少一种源诊断修正激活并在其后继续推进，因为第一种修正可能因果性阻止第二种故障出现。没有训练、optimizer、GT evaluator、C09或C10读取。
- 28/28 evidence seal独立复核PASS，seal SHA-256=`cfd62eac87614a60ed717dbad244ea2eb69ffbbe362f1782b4df0b1950926516`。该结果只证明单例机制可运行，不证明总体优于M-TARE；NEXT为六例readiness，再进入冻结30-case配对矩阵。

## 2026-08-23：V3R逐案21/21完成但聚合兼容FAIL，V1R只读组合审计已冻结

- 新增`docs/PAPER_MANUSCRIPT_DRAFT_V1.md`英文论文正文骨架，覆盖问题、方法、协议、结果、限制和复现；当前保留20处sealed-evidence占位，不手填未完成实验。只读审查确认标题层级、本地书目路径和8个citation key完整。已把V5回锚/出口反馈及30-case身份改为“冻结候选与计划验证”语气，Gate-6证据链封存前不宣称实证成功。
- V3R的21个恢复case全部为`PASS_SINGLE_ROBOT_CASE_V2`并完成lossless archive；case089原始M-TARE覆盖19849.875m³、移动1144.368m、规划p95 0.348s。聚合阶段在任何source-map/统计资产写出前因旧统计器只接受V1状态字符串而FAIL，精确原因=`analysis cannot include a failed case`。run保持`FAIL_AEE_COMPOSITE_V9_STOCHASTIC_V3R_SYSTEM_RECOVERY`，336项seal SHA-256=`47db4379c817d7c8c994b6d172362df028ab3e197de46147034a2b02d1e6f713`且逐项复核PASS。
- 推荐并采用零Gazebo V1R：不修改两个失败run，仅验证21个恢复case全树seal、恢复schedule/progress及69个未受影响 predecessor case；在内存deep copy中映射status后做冻结统计。只读预证明得到90 cases、10×9 blocks、30/30/30配额、60个V9 trace/snapshot全部PASS，机制事件matched/divergent/same-node=`403/414/83`，outcome-blind probe为case052。
- V1R正式run=`gate6_20260823_aee_composite_v9_composed_frontier_attempt_audit_v1r_seed20260820`在15.421秒内PASS；13/13 seal独立验证，seal SHA-256=`0c93c7ccb2e7700a6ec2cfc4bd59b6a9538c90c859fc4032c8ac3902863e6ea4`。26/30案例含回锚proxy、26/30含nonmatching attempt，总proxy帧47123、nonmatching事件497；case052在frame144前已同时出现两机制并被固定为probe。
- 首次V1R live probe在frame21后节点以`module 'math' has no attribute 'ulp'`退出；case runner把“method clean exit before 180s”正确判FAIL。正式完成case=0、duration=12.464秒、训练/C09/C10=0，partial bag和22帧trace仅作失败证据。根因为ROS镜像Python3.8兼容性，不是V5机制结论。
- `math.ulp(1.0)`已替换为数值完全相等的`sys.float_info.epsilon=2.220446049250313e-16`，同时修改runtime和offline audit以保持一致；宿主14/14联动PASS，冻结镜像Python3.8.10只读compile/import PASS。新run必须是V1R2并从头执行同一case052。
- V5 30-case runner继续只选择冻结90-case矩阵中的30个M1D案例，保持10 blocks、每block三checkpoint、600秒/例、18,000仿真秒和全部原始case identity/order；不训练、不读取C09/C10。
- 独立复核发现runner原先只限制来源标签集合，未强制30个M1D案例必须为22个predecessor+8个recovery。该未执行治理缺口已修正，并增加错误配额负测试；相关runner/comparison测试6/6 PASS。
- 新增V5 30-case Data Card/spec proposal与source-binding finalizer。18项冻结文件hash 0 mismatch；修正了未知seal占位符被错误当成真实hash比较的未执行实现缺陷，并以正/负finalizer测试覆盖。
- 新增最终V5-vs-original-M-TARE与V5-vs-defective-V9只读比较Data Card/spec proposal和双source finalizer。比较阶段独立重验schedule双hash、10×9 blocks、30/30/30 family、69+21 source、30个V5的10/10/10 checkpoint和逐summary manifest identity。
- comparison现直接从sealed统计生成3张定量论文图的PNG+矢量PDF，并按预冻结身份规则生成2张真实轨迹/拓扑图：固定garage_env11/tunnel_env11、original repeat0、V9/V5 checkpoint0，选择不读取任何performance outcome；每个trajectory/snapshot再次对源seal。总计5图/10文件，保存hash manifest。16项comparison冻结hash一致，完整联动18/18测试PASS，合成图已目视检查版式。
- 当前NEXT：让V3R自然完成并seal，自动执行69+21组合审计、source-selected probe和六例readiness；三者PASS后finalize/preflight/create并执行唯一V5 30-case run，再做只读配对统计。
- 最终comparison现自动验证120条coverage curve的来源seal，以固定0--600秒网格和十个block生成四方法覆盖曲线；同时导出三方法七指标、两组配对效应和V5机制计数的CSV/Markdown/LaTeX表。定量、定性、表格、coverage联动测试18/18 PASS，20/20 proposal工具hash一致，禁止手填论文结果数字。

## 2026-08-23：90-case在77/90因主机校时FAIL，采用完整区组+尾部恢复

- predecessor=`gate6_20260822_aee_composite_v9_stochastic_v3_seed20260820`已封存为`FAIL_AEE_COMPOSITE_V9_STOCHASTIC_V3`，progress为77个完整case，失败case为`077_tunnel_env23_original_mtare_repeat0`；1201项seal，seal SHA-256=`1c1e51e6a386351e9900edd262fdcb0411f61969f79c7d634579979cf48e4e09`。
- case077仿真与录包达到完整区间，raw bag为5,974,544,949 bytes。598个`/runtime`消息中仅1个为`-0.1809999943s`（bag stamp 391.93s），其余597个全部有限非负且范围`0.116--0.459s`；主机日志在该case运行中的12:57:13明确记录`Clock change detected`。分类为system clock/instrumentation failure，不是模型、地图、控制或覆盖失败。
- 原始M-TARE的计时器使用`std::chrono::high_resolution_clock`，该唯一回拨样本触发冻结finite-value门禁；不删除负值、不补写summary、不修改旧RUN_STATE，也不把旧run追认PASS。
- 为避免只选择性重跑失败baseline，V3R固定完整重跑`tunnel_env23`九例（原indices 016/018/020/029/030/040/041/043/077），再执行原计划078--089十二例，共21例/12,600仿真秒。组合时旧区组8个PASS全部排除，case077 partial完全禁用，仅只读复用另外69个sealed PASS案例。
- 已新增V3R runner、Data Card、run spec与3项选择/组合单元测试；专项4/4测试PASS。V3R已经通过preflight、创建并执行，当前9/21；预计<=5小时、<=35GB、串行CPU/Gazebo、零训练/GPU/C09/C10。

## 2026-08-23：90-case运行到76/90，frontier-attempt审计V1R3冻结待源seal

- 正式run未重启、未修改；当前70/90，case070已启动。case065 baseline覆盖15230.375m³、移动1137.771m、规划p95 0.339s；case066--068 oracle诊断分别仅移动7.568/5.460/7.119m，继续证明其不是性能上界。case069 V9覆盖3663.250m³、移动901.947m、规划p95 0.011382s，0 fallback/0 proxy；93次frontier事件为15 matched、76 divergent、2 same-node，91/91 traversal双射PASS，而现有ledger仅记1次clearing，强支持图事件执行反馈。
- V1初版缺少完整row/stub/traversal/run identity验证，V1R复核后仍缺少traversal起止区间、final current-node、path cycle及environment/status内容门禁；两版均未执行并明确superseded。
- Luna复核发现V1R2仍可能接受未与事件双射的traversal及未知target mode，且一小时run-age限制会误拒绝有效审计；V1R2保持未执行并被V1R3替代，不影响源90-case。
- V1R3要求所有target键和mode精确匹配冻结planner，所有traversal的起止行节点匹配from/to，并与全部inter-node decision events形成严格双射；不再设置任意run-age上限。当前13个sealed V9案例、419条traversal全部通过真实只读回放。
- V1R3 Data Card/spec proposal冻结90 summaries、30 traces、30 snapshots和0 raw bag/GT/map/training/C09/C10；17项工具哈希一致、33/33测试PASS。旧版finalizer、未完成source和并行material run三类负控制均正确拒绝，未生成正式card/spec或run目录。
- 用户再次确认既定论文范围内由agent自主选择最优方案，不再逐次请求批准；治理仍保留独立Data Card/spec、preflight、不可覆盖run与seal，只有结论范围、数据隔离或评价合同发生实质变化时才停止报告。

### case033新增机制证据

- tunnel/env71/V9-seed1完整case PASS：600秒覆盖`1052.375 m³`、移动`222.181 m`、AUC=`539917.377 m³·s`、规划p95=`0.011463 s`，fallback=`0`。
- 3001帧中graph-backtrack为1856；1829帧满足旧节点arrival proxy，最长连续1829帧，末尾1790帧route arc不再增长。最终13 nodes/48 stubs/20 observed，单节点stub-minus-branch-count最大5。
- 21次可分类frontier事件为13 matched、3 divergent、5 same-node；8次非匹配。当前13例合计443事件为222 matched、188 divergent、33 same-node，非匹配221；10/13存在回锚proxy。该部分样本继续只作机制定位，不提前选择修正。

### case034 oracle重复证据

- tunnel/env53/layered-oracle/repeat2完整PASS：覆盖`1226.625 m³`、移动`165.122 m`、AUC=`695485.344 m³·s`、规划p95=`0.106019 s`；recording、storage与lossless archive合同通过。该单例只进入冻结block统计，不作独立上界结论。

### case035原始M-TARE块内证据

- tunnel/env71/original-M-TARE/repeat2完整PASS：覆盖`10020.125 m³`、移动`1150.232 m`、AUC=`3074632.134 m³·s`、规划p95=`0.320000 s`。
- 同一environment block的V9 seed1 case033覆盖`1052.375 m³`且规划p95=`0.011463 s`。当前只读块内观察显示V9计算延迟显著更低但覆盖被图状态机制压制；这不是最终配对统计，却进一步说明必须完成并验证组合修正，不能推广原V9。

### case036--037新增证据

- case036 tunnel/env11/original-M-TARE/repeat1覆盖`8597.375 m³`、移动`1173.637 m`、AUC=`2745350.063 m³·s`、规划p95=`0.311000 s`，完整PASS。
- case037 garage/env53/V9-seed2覆盖`4774.125 m³`、移动`376.086 m`、AUC=`2498277.631 m³·s`、规划p95=`0.011574 s`。1592帧出现re-anchor proxy，最长连续1581帧，末尾1515帧route arc不增长。
- case037的31次frontier事件为15 matched、13 divergent、3 same-node；V1R3验证29/29 traversal双射。当前14例累计474事件恰为237 matched/237 nonmatching，11/14存在proxy，448/448 traversal通过。

### case038--040新增块与机制证据

- case038 garage/env53/original-M-TARE/repeat2覆盖`18470.125 m³`、移动`1142.307 m`、AUC=`5820427.759 m³·s`、规划p95=`0.335000 s`；同块V9 case037为`4774.125 m³`与`0.011574 s`，继续显示低延迟但显著覆盖损失。
- case039 garage/env23/V9-seed0仅覆盖`1397.500 m³`、移动`75.652 m`；2590 proxy帧、最长2590帧，末尾2533帧route arc不增长，同一出口被选2672帧。
- case040 tunnel/env23/V9-seed1仅覆盖`226.375 m³`、移动`28.790 m`；2834 proxy帧、末尾2778帧无route增长，同一出口被选2846帧。16例累计13 proxy cases、483 events中240 nonmatching、459/459 traversals通过。

### case041--042新增baseline证据

- case041 tunnel/env23/original-M-TARE/repeat2覆盖`9200.250 m³`、移动`1147.292 m`、AUC=`3032636.981 m³·s`、规划p95=`0.341000 s`；同块V9 case040为`226.375 m³`与`0.011269 s`，确认低延迟不能补偿图状态停滞造成的覆盖损失。
- case042 garage/env37/original-M-TARE/repeat2覆盖`11162.875 m³`、移动`1122.736 m`、AUC=`3823744.628 m³·s`、规划p95=`0.341000 s`，完整PASS。

### case043--045确认oracle不是性能上界

- case043 tunnel/env23/layered-oracle/repeat2仅覆盖`632.000 m³`、移动`84.713 m`、AUC=`369652.664 m³·s`、规划p95=`0.097518 s`。
- case044 tunnel/env37/original-M-TARE/repeat2覆盖`7807.000 m³`、移动`1119.903 m`、AUC=`2919392.989 m³·s`、规划p95=`0.338000 s`。
- case045 garage/env37/layered-oracle/repeat2仅覆盖`463.375 m³`、移动`8.815 m`、AUC=`274615.771 m³·s`、规划p95=`0.112310 s`。oracle从冻结设计起只作诊断，不影响primary baseline比较；论文中禁止称其为理想性能上界，须单独解释完整地图目标执行失败。

### case046--048新增证据

- case046 garage/env71/original-M-TARE/repeat1覆盖`15023.625 m³`、移动`1146.507 m`、AUC=`5474721.725 m³·s`、规划p95=`0.326000 s`；case047 garage/env11/original-M-TARE/repeat2覆盖`15050.625 m³`、移动`1142.161 m`、AUC=`4892769.988 m³·s`、p95=`0.316000 s`。
- case048 tunnel/env71/V9-seed0仅覆盖`442.500 m³`、移动`79.852 m`、p95=`0.010944 s`；2521 proxy帧且同一出口持续2534帧。其tail-without-route-growth为0，说明route arc持续增长仍可发生在旧图路径重复执行中，不能作为无故障判据。
- case048的8次frontier事件为5 matched、1 divergent、2 same-node；6/6 traversal双射。当前17例累计14 proxy cases、491事件中243 nonmatching、465/465 traversal通过。

### case049--052新增双机制证据

- case049 garage/env53/oracle/repeat1仅覆盖`460.500 m³`、移动`5.890 m`，继续支持oracle仅作诊断。
- case050 garage/env11/V9-seed2覆盖`4446.250 m³`、移动`652.404 m`；仅13 proxy帧，但61次frontier事件中36次非匹配，最终31 nodes/170 stubs/106 observed，证明exit feedback/lifecycle可独立主导损失。
- case051 tunnel/env11/V9-seed1覆盖`1184.875 m³`、移动`288.119 m`；755 proxy帧、末尾663帧无route增长，26事件中15 nonmatching。
- case052 tunnel/env11/V9-seed0仅覆盖`220.625 m³`、移动`24.261 m`；2857 proxy帧、末尾2821帧无增长，同一出口2940帧。20例累计17 proxy cases、580事件中295 nonmatching、547/547 traversal通过。

## 2026-08-23：90-case运行到28/90，出口执行结果证据实现

- case027 original M-TARE tunnel/env11/repeat2完整PASS并lossless归档：600秒最终覆盖`7229.0 m³`、移动`1138.836 m`、规划p95 `0.321 s`；第29例已自动启动，run未修改、未重排、未重试。
- 新增`topology_frontier_attempt_evidence.py`，只用相邻决策周期、最终图节点坐标和稳定stub identity，把活动frontier后的图事件分成`matched_verified_departure`、`divergent_verified_departure`和`same_node_loop_merge`；不读GT/map/raw bag，不声明调参阈值。
- case026的68次可分类事件为8 matched、50 divergent、10 same-node，非匹配比例`0.882353`；60次非匹配中58次目标连续作用路径至少4m，56次至少8m，全部10次same-node均超过8m，排除“目标刚切换一帧”的主要误归因。最常重复的node4/stub3为19 divergent、4 same-node、0 matched。轨迹活动范围仅约`31.82×7.48 m`，node0↔node4的verified edge累计37次traversal，因此“能移动但重复绕旧区”已由自身执行轨迹而非GT证明。
- 当前11例对照中，两个严重回锚停滞案例只有1次可分类frontier事件；两个健康garage案例有较大空间拓展和正常节点增长；case026独立暴露出口执行反馈缺失。相关re-anchor/lifecycle/attempt专项单元测试`12/12 PASS`。该不完整样本仍只用于机制设计，最终选择等待30/30分布。
- 独立正式audit proposal已冻结90 summaries、30 V9 traces、30 snapshots、0 bag/GT/training/C09/C10；9项工具/测试hash一致，runner/evidence/finalizer专项`7/7 PASS`。运行中源作为负对照被finalizer正确拒绝且没有物化正式card/spec；proposal preflight仅保留源run仍运行、最终spec不存在、授权待final source binding三项预期阻塞。

## 2026-08-23：90-case运行到27/90，新增exit-stub lifecycle机制审计

- 正式run=`gate6_20260822_aee_composite_v9_stochastic_v3_seed20260820`持续运行；27个case均完成V2 case合同和host lossless archive，当前第28例自动执行。原run不修改、不重排、不重试。
- case026 tunnel/env37/seed1无回锚proxy且route arc持续增长到605.454m，但仅覆盖351.375m³、冗余0.999825；3001帧均指向frontier，最常选stub占1383帧，最终11节点/47 stubs/26 observed，单节点stub数最多超当前branch count 6。这证明V4回锚不是充分修复；分类为topology/planner exit-stub lifecycle与目标效率问题。
- 已新增无阈值的frontier lifecycle evidence，逐case报告target mode、graph update、frontier频次/最长run和最终stub分布；与re-anchor audit共同覆盖全部30个sealed V9案例。整条回归57/57 PASS。
- 冻结runner最终存在已确认接口缺陷：case finalizer写`PASS_SINGLE_ROBOT_CASE_V2`，统计器只接受V1状态。决定保留最终系统FAIL与seal，再由新只读audit核验90个summary和来源seal，仅在深拷贝内存中映射status并调用冻结统计器。
- 已完成10个V9案例的机制审计：8例有重复旧节点arrival proxy；两个tunnel案例最长连续2953帧、路径尾部约2900帧不增长且最终仅约20m；两个proxy为0的garage案例移动785--926m。新增tunnel/env53/seed0案例移动279.514m并覆盖1891.875m³，但1827帧命中proxy且尾部1782帧route arc不增长。fallback近零，定位为graph state而非模型输出。该不完整样本只支持机制诊断，不支持最终优劣结论。
- V4加入显式verified-backtrack re-anchor：绑定上一周期路径、唯一verified edge、正trace、既有loop radius和严格距离关系；追加反向真实traversal并重置状态。probe、2×3 readiness、精确源schedule绑定的30-case runner、只读配对统计均已实现。兼容audit现额外逐一核验30个sealed V9 trace/snapshot并输出机理表；源seal出现后一次性物化正式card/spec的fail-closed finalizer也已实现，整条回归54/54 PASS。
- 完整案例校验覆盖七项finite metrics、recording audit、planner evidence、method identity与storage hashes；相关回归51/51 PASS，ROS Python3.8 compile PASS。V4 live结果仍为未知。

## 2026-08-22：V9原始传感器接口恢复，6-case readiness正式PASS

- V1R2在线使用`/registered_scan`重栅格化，首例出现814/901 learned-empty；这与封存organized AEE数据每seed仅2/1000 learned-empty不一致，判定为部署输入算子错配，不能作为模型科学失败。
- V1R3仅对V9恢复`/velodyne_points`固定16×350物理束解码与冻结350→720算子；位姿仍来自`/state_estimation_at_scan`，沿用既有0.1秒配对合同。旧模式、模型、checkpoint、阈值、场景、seed和控制栈不变。
- 6/6 case通过：移动距离范围`20.121--279.053 m`；节点范围`2--19`；verified edges范围`1--19`；每例901个非hold周期、881个post-warmup周期且fallback均为0；零失败周期。
- 正式状态`PASS_AEE_COMPOSITE_V9_READINESS_V1R3`，耗时`1272.508 s`，108项证据，目录3.2GB，seal SHA-256=`30a213d156d1f64bc3be3a8a4524df9ea6ac06463ed95b596c5d9d156f7b804d`；训练/optimizer/C09/C10读取均为0。
- readiness已解除90-case前置阻塞。下一步只迁移既有90-case runner/spec到V9 deployment及AEE V2 recording contract，不改变90-case设计、统计方法或门槛。

## 2026-08-22：V9复合结构语义接口三seed正式PASS

- 方法固定为：学习模型输出出口方向，冻结B0输出出口数量，固定`count<=1/2/>=3`映射terminal/interior/junction；神经count/role只保留诊断，不进入runtime。
- 三seed sparse direction F1=`0.847874/0.847615/0.843185`，相对各自source提升`0.212363/0.175776/0.174785`；count/role macro-F1固定为`0.775478/0.722601`。
- 三seed科学门槛、方向保持、冻结/更新tensor合同全部通过；C09/C10/formal benchmark读取均为0。
- 正式状态`PASS_AEE_CORRECTIVE_COMPOSITE_V9`，耗时`528.695 s`，65项证据封存，seal SHA-256=`5d397b2c296029dfb2decdb57d9fbb9332e1713cbf1faaf46ee90cd8f6a85468`。
- 该结果只解除结构语义runtime阻塞；C08因果建图已有sealed PASS，下一步按权威计划验证AEE闭环readiness，尚未宣称90-case规划替代PASS。

## 2026-08-22：count可观测性审计完成，冻结B0 fallback满足count/role门槛

- 5,000固定validation帧、10,677 teacher exits全部审计；dense B0可匹配8,374 exits，mask后保留8,372，仅2个出口/2帧发生mask因果丢失，无法解释神经count失败。
- dense/sparse B0 count exact分别`4642/4572`帧；sparse count 1--4 macro-F1=`0.775478`，固定count→role mapping macro-F1=`0.722601`，均超过既定0.70。
- 正式audit为`COMPLETED_AEE_COUNT_OBSERVABILITY_AUDIT_V1`，12项seal SHA-256=`f263d71f8f2b1e0b1a802e780bd12fc9111ae1d9159ca561e4e4cbce1ee47970`，零训练/模型推理/C09/C10。
- 决定使用预声明necessary fallback构成完整复合方法；新run必须从原source重新执行，V8R checkpoint不得被部分推广。

## 2026-08-22：V8R方向保持与角色改善有效，但count可观测性成为阻塞

- V8R三seed完整执行；sparse direction F1=`0.847874/0.847615/0.843185`，均超过V4R/V5--V7和B0；role F1提升到`0.653891/0.658635/0.634710`。
- count 1--4 macro-F1仍为`0.487912/0.539757/0.484828`，三seed整体FAIL。encoder.0--4与direction head保持，encoder.5/embedding/count/role按合同改变，retention机制有效。
- 运行耗时`513.704 s`；62项seal SHA-256=`1da86766b002e9f6fc2fa576fee60148a7ce49ca8d69d4281775e551f466c15d`并校验通过；C09/C10/formal benchmark读取均为0。
- 下一步停止盲目扩大encoder，先审计固定mask-matched验证中完整拓扑count与实际可见出口的可辨识性；若mask因果性移除了teacher出口证据，则修正teacher/任务定义而非增加模型容量。

## 2026-08-22：V8因summary质量字段遗漏系统FAIL，转V8R最小接口修复

- V8 seed0完成10 epochs后，trainer summary漏写冻结的`count_effective_mass/role_effective_mass`；runner在模型验收前检测到缺失并停止，seed1/2未运行。
- run为`FAIL_AEE_CORRECTIVE_ENCODER_RETENTION_V8`，completed_seeds=0、execution_failure=true、耗时`182.564 s`；证据清单SHA-256=`7f95696a15dac161700fc6b738d07992e411b9b85699a16e4c2bb754e0157b2a`并校验通过。
- seed0局部诊断方向/count/role=`0.847874/0.487912/0.653891`，不得作为三seed科学结论或checkpoint推广。V8R只补summary冻结质量与权重字段并从原source重新执行，科研变量不变。

## 2026-08-22：V7方位结构汇聚FAIL，后端读出路线停止

- 三seed方向F1仍为`0.827722/0.824020/0.817702`且step-47 tensor哈希保持；count F1=`0.376367/0.446549/0.387015`，role F1=`0.553312/0.574543/0.561693`。
- V7严格保持checkpoint tensor key/shape和encoder，采用共享pointwise embedding后再做circular mean，合成测试证明旋转不变；正式结果仍在三seed失败，因此不再继续同类pooling/head变体。
- 运行耗时`487.452 s`；62项证据清单SHA-256=`d9f86c494779b217d8da1a396adc21488d069878360878753da38d4db137b62b`并独立校验通过；C09/C10/formal benchmark读取均为0。

## 2026-08-22：V6开放embedding仍FAIL，阻塞定位到encoder特征

- 三seed各完成10 epochs/470 steps；方向F1=`0.827722/0.824020/0.817702`，step-47与最终direction tensor SHA-256逐seed完全一致。
- count 1--4 macro-F1=`0.485931/0.511244/0.463479`，role macro-F1=`0.581593/0.580983/0.557465`；三个最佳checkpoint均为epoch 1，继续更新embedding反而降低held-out sparse分类。
- encoder逐tensor identity、embedding与semantic heads改变、所有tensor有限；这排除了“只需开放独立embedding”的机制解释，说明count/role所需的稀疏结构信息未被冻结encoder充分编码。
- 运行耗时`489.143 s`；62项证据清单SHA-256=`9beedc9ca7083c7c0c9bb07561be81784c4372ffc5a0ac75e2032c8a61dde458`并独立校验通过；C09/C10/formal benchmark读取均为0。

## 2026-08-22：V5保持方向但证伪“仅类别平衡即可修复count/role”

- 三seed各完成10 epochs/470 steps，无执行故障；sparse direction F1=`0.827722/0.824020/0.817702`，与V4R已验证的epoch-1方向结果一致。
- count 1--4 macro-F1=`0.510308/0.536945/0.488471`，role macro-F1=`0.584015/0.588006/0.567634`，三seed均未达到0.70，因此状态为`FAIL_AEE_CORRECTIVE_STAGED_BALANCED_HEADS_V5`。
- encoder和embedding冻结、direction只更新首个epoch；类别权重按实际有效训练质量计算。结果排除了方向头继续干扰和类别频率未校正这两个解释，剩余阻塞定位为冻结embedding表征不足。
- 运行耗时`485.467 s`；证据清单SHA-256=`dda6774027638b921bac2c2fec5927705637c3560c84fbd6030d91aeab44b4a6`并独立校验通过；C09/C10/formal benchmark读取均为0。

## 2026-08-22：V4R方向修复成功、整体门禁仍FAIL

- V4首次运行在0 seed/0 optimizer step因旧runner硬编码Data Card status停止并封存；10项seal SHA-256=`5b7edc2b93bfcd631d367c11685343028832225bdca359f4f402bd504086abad`，无partial reuse。
- V4R三seed各完成10 epochs/470 steps。sparse direction F1=`0.827722/0.824020/0.817702`，相对source提升`0.192211/0.152181/0.149303`并全部超过B0=`0.780483`；dense direction变化为`+0.047954/+0.071442/+0.034609`。
- count 1--4 macro-F1=`0.510118/0.538351/0.480905`，role macro-F1=`0.582275/0.583770/0.557399`，seed2 sparse empty=`0.0552`；故三seed总门禁均FAIL。
- encoder/embedding逐tensor identity通过，semantic heads改变且数值有限；C09/C10/formal benchmark读取均为0。62项seal SHA-256=`2b578c623b9354f17b123392b005690f59b1d823dd018495ca51984e5d141b1c`并独立验证。
- 方向分量直接推导count/role的只读审计也失败：count F1仅`0.311--0.350`、role F1仅`0.454--0.489`，不得用规则替代分类门禁。

## 2026-08-22：AEE corrective full-encoder V3 三seed科学FAIL

- 正式运行：`results/gate2_representation/gate2_20260822_aee_corrective_full_encoder_v3_seed20260822`；三seed、各10 epochs、各470 optimizer steps全部完成，执行故障为0。
- held-out sparse Cano方向F1为`0.495606/0.516503/0.506227`，相对各自source下降`0.139905/0.155336/0.162172`；dense方向也下降`0.194937/0.209198/0.219696`。
- AEE train-fit方向F1仅`0.054795/0.022707/0.051533`，空输出率`0.917/0.969/0.931`；同一5000-frame sparse验证上的B0方向F1为`0.780483`。
- 只读标签审计：Cano/AEE direction target平均正质量分别为`0.044495/0.045349`，普通BCE全空常数解的logit分别约`-3.067/-3.047`，与高空输出一致。
- 运行耗时`519.435 s`、62项证据封印，seal SHA-256为`ba5eff2b0300c0e0334a66eb0ad7124cf3528ce6dab62e4fb7fbd687ec442454`并独立验证；C09/C10/formal benchmark读取均为0。禁止retry、阈值回调、删seed或推广任一checkpoint。

## DONE

- 用户最终执行方案 V1 已固定为 `docs/PLAN.md`，并写入根目录实施前读取合同；
- 项目边界收敛为 LiDAR outgoing branches → Topometric Graph → global target → M-TARE Local Planner；
- 当前实施上限固定为 Phase 3，禁止现在修改 M-TARE；
- 官方 Isaac Sim 6.0.1 Compatibility 和本地 USD import 已通过；
- clean-room `g000/g001` 完成 topology/mesh/USD/Isaac 接口预检和失败知识积累；
- Cano GitHub 仓库现在可公开访问；默认分支 `refactor`，候选 HEAD `b6c77621187404b4dfab1249c7a1b40f63ad9ab3`；
- 仓库页面确认存在 `snippet_2.py`、`tunnel.py`、`mesh_generation.py`。
- 已建立并通过治理预检的运行规格 `configs/v3/gate0/cano_source_executability_smoke_v1.json`；
- 已隔离 checkout `procedural-subt-gen@b6c77621187404b4dfab1249c7a1b40f63ad9ab3`，remote/HEAD 精确匹配，tracked 工作区运行前后均干净；
- 已保存 Python 3.12 隔离环境与完整 `pip freeze`，原脚本运行前后 SHA-256 均为 `869bc12e...f05f12`；
- 已完成一次且仅一次原样 `scripts/snippet_2.py` 执行，失败日志、最小复现、license/dependency 审计和机器 summary 已封存。
- 只读入口审计确认 `snippet_3.py` 负责固定拓扑上的 point-cloud/mesh 计算，`generate_environments.py` 才原生保存 `mesh.obj/axis.txt/fta_dist.txt/model.sdf`；
- 已完成冻结兼容性矩阵 run：E1（Python 3.12 / NumPy 1.26.4 / SciPy 1.12.0 / OpenCV-headless 4.8.1.78）通过 `pip check`，原 `snippet_2.py` 退出 0；
- 矩阵遵守 stop-on-first-pass，作者年代备选 E2 未创建、未执行；原脚本与第三方 tracked 工作区保持不变；
- 原始 E0 失败与 E1 成功均有完整 constraints、freeze、安装日志、命令、时长和机器 summary。
- 已按用户批准执行一次且仅一次原生 `generate_environments.py` 导出：1 个临时环境、请求 3 grown/1 connector，13.70 s 退出 0，输出四个原生文件共 9,851,416 bytes；
- 已生成实际完整导出 mesh/axis 的 X-Y、X-Z、Y-Z 三联诊断图；验证和科研可视化均有机器可读 provenance；
- E1 freeze、`pip check`、第三方 remote/commit/entrypoint hash 和 tracked clean 状态在运行前后全部一致；21 项原生运行证据 hash 校验通过。
- 已实现项目侧 read-only Cano adapter、严格 bundle validator 和运行封存器，不修改或复制第三方核心；
- 外部 E1 合同测试 3/3 与 V3 单元测试 41/41 通过；实验 preflight 通过；
- seed 0 唯一一次 adapter generation 中 3 grown + 1 connector 返回全部明确成功，无整图重试；
- 标准临时 bundle 已输出 graph/splines/per-tunnel axis/WorldInfo/metadata/SDF/原始 mesh：graph 为 52 nodes/52 edges、1 component、cycle rank 1；4 条 spline 端点最大误差 `5.69e-14`；1,042 条 interior axis 行冲突为 0；
- 首次 validator 仅因 `numpy.bool_` 不能 JSON 序列化而失败；旧日志与哈希、修正说明和 validation-only rerun 均已保留，没有第二次生成、seed/metric/threshold 变化；最终 27 项运行证据与 bundle 内文件 hash 全部通过。
- 已根据论文任务边界把几何资格拆成 topology-GT、LiDAR-sensor 和 dynamic-navigation 三类门禁；严格流形失败不再未经传感器证据直接替代 LiDAR 判定；
- 用户已批准仅使用现有 adapter seed-0 `world_000` 的 24 个分层静态 pose，运行 Isaac 16×720 range 与 5 m spline/360°标签 smoke；正式数据集计数仍为 0；
- 已决定使用零数据集 `sensor_smoke` 专用审批卡，避免为诊断样本伪造 train/validation/test split 或轨迹。
- 已实现固定 pose/5 m spline label/CPU raycast、Isaac GMO capture、RTX-CPU 比较和全样本可视化工具；新增治理与几何测试合计 19/19 通过；
- 已冻结并通过 preflight：1 world、24 poses、8/8/8 role、16×720、50 m 和全部逐样本阈值；
- 已实际生成确定性 24-pose manifest 和 24 个 CPU reference；24/24 净空、标签 branch count 和分支视线准备检查通过；
- 已保存 Isaac 6.0.1/GPU/RTX extension 成功启动证据、两次包装恢复记录和最终自定义配置注册失败证据；正式数据与训练计数保持 0。
- 已完成 Isaac Sim 6.0.1 只读 registry/API/schema 审计：`config=` 只接受官方 USD registry，`usd_path=` 是自定义 OmniLidar 资产的受支持入口；旧 JSON 中 16×720/10 Hz/50 m 参数均可映射，且必须显式写出 16 channels 与 channelId 1..16；
- 当时已形成 `docs/ISAAC_RTX_EXACT_PROFILE_CREATION_AUDIT_V1.md` 和单传感器 v1 规格；该规格后来经单独批准并已执行失败，历史审计本身的 GPU/Cano/LiDAR/data/training/model 计数仍为 0。
- 本轮 JSON/Phase 一致性检查通过，治理与 Cano sensor smoke 单元测试 19/19 通过；没有因测试环境缺少项目级 pytest 而临时安装依赖，改用本机已有 Anaconda pytest 执行。
- 已按批准范围执行且只执行一次 exact-profile v1 探针：0 Cano world、1 解析盒、1 sensor、1 pose；本地 `OmniLidar` USDA 创建成功，69 个 checker 参数通过，冻结属性 read-back 为 0 mismatch，推导 720 tick/scan 和 11,520 nominal rays/scan；
- v1 未再出现 `Config not found`，证明旧注册阻塞解除；但直接 `LidarSensor.get_data` 返回的 header 为 `0xF4AEEB00`，不等于 GMO magic `0x4E474D4F`，300 frame 内完整扫描为 0；
- 当时已完成 Isaac 6.0.1 自带 Writer 测试的只读审计并准备 v2；runner 修正为只依据完整 NPZ + PASS capture summary 计数，静态编译和相关测试 25/25 通过；v2 后来经单独批准并由下方结果取代“未运行”状态。
- 用户单独批准后，v2 新卡/规格、5 项 hash、镜像、范围和输出目录 preflight 无错误无警告，V3 单元测试 53/53 通过；唯一一次 GPU run 已完成并封存；
- v2 再次证明 local USDA、69 项 checker 和属性 read-back PASS；19 项证据 hash 全部通过，目录 160 KB，完整扫描、正式数据、标签、训练和模型仍为 0；
- 已只读核对固定镜像官方 standalone Writer 示例，v2 的注册/attach/timeline/update 顺序与官方一致；用户随后批准内置 `Example_Rotary` 仅作 runtime A/B control。新探针、runner 与测试已实现，58/58 全量测试、零警告 preflight、官方示例/镜像/代码 hash 全部通过；唯一一次 GPU control 已执行并封存。
- 官方 control 在创建 `Example_Rotary` 前置资产阶段失败：配置名需要远端 NVIDIA 6.0 `Example_Rotary.usda`，冻结容器禁网，镜像和主机无本地副本。未创建传感器、未 attach Writer、未进入 300 frame；13/13 证据 hash 通过，124 KB，无重试，正式数据/标签/训练/模型均为 0。
- 用户批准的单文件来源冻结已 PASS：精确 URL 1 request、HTTP 200、0 redirect，下载 15,137 bytes，SHA-256 `0812faf5...56c8c6`；断网 OpenUSD 25.11 识别默认 prim `Example_Rotary`/`OmniLidar`，sublayer/reference/composition/external-asset dependency 全部为 0；16/16 evidence hash 通过，无 GPU/Writer/data。
- 精确官方 local-USDA Writer control v2 已按批准只执行一次：sensor creation=1、Writer attach=1、同步 standalone updates=300；callback/valid GMO/positive element/complete scan 全部为 0，writer-drain timeout 为 true；15/15 evidence hash 通过，160 KB，无样本、图片、训练或模型，无重试。
- 已只读审计固定镜像内 NVIDIA `test_lidar_sensor.py`（SHA-256 `2f545cfe...ccbb`）：官方 Writer 单测是 `AsyncTestCase`，等待 viewport ready 并使用 `next_update_async()`；形成审计文档和未批准 v3 单变量提案，本次审计 GPU/数据/训练计数为 0。
- 用户批准将 Isaac 从 Phase 1--3 关键路径降级为未来可选传感器域实验；async Writer v3 提案已退休且未实现/未执行，历史失败证据不删除、不回写。
- 已固定“可运行可复现代码先用”的复用边界：Cano TNG/表面点云/native mesh、项目 CPU raycast 和 spline label 可直接复用；原论文 CNN 训练代码当前不可见，B1 只能做 adapted reproduction；本项目创新延后到 R/D/U、exit-stub graph、M-TARE 高层替换和多机器人分配。
- 当时已形成 CPU Raycast 主后端文档、单图 proposal 和 smoke card；范围为 1 个现有 world、24 poses、每 pose 11,520 rays、两次独立 CPU scene，不生成新 world、不计正式数据、不训练；该提案随后获批并由下方 PASS 结果取代 pending 状态。
- 用户批准后已实现、冻结并只执行一次 Cano CPU Raycast 单图合同；执行前 V3 测试 74/74、外部 Cano 合同 3/3、preflight 零错误零警告；无重试。
- 单图合同 24/24 PASS：解析盒最大误差 `2.38e-7 m`，双独立 scene 和旧参考最大 range 差异均为 `0 m`，valid ratio `0.990972--1.0`、均值 `0.997070`，最小水平净空 `2.0585 m`，分支 LOS 24/24。
- 结果目录包含 24 个诊断 NPZ、完整 world/pose 图和全部 24 扫描统一色标 contact sheet；46/46 evidence hash 通过，目录 3.8 MB；正式 dataset/训练/model/Isaac/Gazebo/topology/M-TARE 计数均为 0。
- 已按用户“先把正常东西做出来”的指令实现开发纵向切片：使用已有 seed-0 Cano world，不生成新地图、不训练、不接 ROS、不修改 M-TARE；沿 52-edge source graph 的双边遍历形成 1,676.46 m、240 帧因果轨迹，共执行 2,764,800 条 primary CPU rays。
- 新增透明的当前帧 LiDAR range-sector 出口规则 `src/mtare_topo/semantics/range_exit_baseline.py`，以及维护事件节点、物理移动边、回环合并和 unexplored/traversed exit stub 的 `src/mtare_topo/topology/online_topometric.py`；新增 4 个组件测试，全量测试从 74 增至 78 个。
- v1 保留原始弱结果；诊断确认严格 node-ID 映射低估边质量，同时默认事件 debounce 漏建路口。随后在**同一开发轨迹**上枚举 81 组图关联配置，冻结为 stable frames 1、minimum event travel 6 m、loop radius 4 m、branch merge 20°、turn event 35°，不得把该选择称为独立验证。
- v2 结果已封存于 `results/prototypes/cano_seed0_lidar_to_topometric_vertical_slice_v2_graph_refined/`：出口 P/R/F1=`0.7480/0.7525/0.7503`、平均角误差 `6.985°`；规则图 43 nodes/50 edges/14 unexplored stubs，对照 oracle 44/50/12；node P/R/F1=`0.8372/0.8182/0.8276`，直接几何 edge P/R/F1=`0.94/0.94/0.94`。
- v2 保存 240 帧 NPZ、规则/Oracle 图 JSON、逐帧指标、81 组 sweep、全地图汇总图、24 帧固定索引诊断图和 60 状态 GIF；13/13 evidence SHA-256 已复核。该 run 状态为 `COMPLETED_WORKING_VERTICAL_SLICE`，不是 Phase 3/4 PASS。
- 已实现独立 `sensor_contract_pilot` 治理卡、五类显式 Cano 拓扑模板、拓扑/几何种子分离、事件优先的 50-anchor 选择、客观局部分支标签、冻结规则重放、五张完整地图与 30 页全观测可视化执行器；五类模板中心线静态长度均不小于 300 m。
- 正式执行前 84/84 V3 单元测试、5/5 外部合同测试和零错误零警告 preflight 通过。失败后新增固定 checkout 导入身份测试，当前为 84/84 与 6/6 通过；只验证导入，没有构造或重跑正式 world。
- corrective v2 在正式 executor 启动前发现 runner 仍硬编码 v1 proposal/data-card，按 hash 门禁记录为 `PRECHECK_FAILED_NOT_EXECUTED`；world/mesh/anchor/observation 均为 0，旧目录未复用。
- 已把 proposal/data-card 改为从 approved spec 安全解析，并拒绝绝对路径、`..` 越界和缺失文件；84/84 单元、8/8 冻结 E1 外部测试及 v2b preflight 均通过。
- v2b 已按批准命令执行一次：固定 checkout 导入、审批、hash、依赖环境和前置单图 PASS 全部通过；运行 32.24 s 后在 P04 anchor 选择处停止，37 MB、51 项 SHA-256 证据全部复核通过，无重试。
- P01/P02/P03 已各完成 50 anchor × 3 view，三个诊断 shard 形状均为 `[150,16,720]` / `[150,720]`，双独立 scene 最大差异均为 0，分支 LOS 通过；冻结规则 F1 分别为 `1.0000/0.9447/0.9154`。这些是失败 run 的局部诊断，不是五拓扑 PASS 或正式数据。
- 已实现并正式执行一次零 mesh/零 ray 的 target-cardinality selector audit：85/85 单元、9/9 外部测试、preflight 0 错误/0 警告；5 图均为 50 anchors，最小间距 `5.4985 m`，确定性 replay 全一致，32 项 evidence hash 通过，目录 1.3 MB。
- 已逐张检查五张完整 X-Y/X-Z 图，而非只接受 JSON PASS；发现 earliest-feasible round-robin 在达到 50 后留下长 spline 空白。只读量化的同 tunnel 最大覆盖半径为 P01 `32.780`、P02 `16.564`、P03 `92.306`、P04 `10.501`、P05 `17.558 m`。复核记录为 `docs/ANCHOR_SELECTOR_VISUAL_REVIEW_V1.md`。
- 已实现 selector v2：结构事件先保留，非事件名额按 tunnel spline 弧长作确定性最大余数分配，并在 0.5 m 弧长格上用二元 MILP 同时约束精确配额、全局 5 m 间距和完整 spline 7.5 m 同 tunnel 覆盖；单元测试 86/86、冻结 E1 外部合同 9/9 通过。
- 用户批准的唯一一次 v2 零 mesh/零 ray 正式审计已封存：5 图各 50 anchors，弧长配额、事件覆盖、每 tunnel 覆盖和 replay 均通过；P01--P05 最小间距分别为 `5.006/5.500/5.185/5.500/5.500 m`，最大覆盖半径为 `3.500/4.000/6.392/4.000/4.000 m`；32 项文件封存，目录 1.4 MB。
- 已逐张人工核验 v2 的五张完整 X-Y/X-Z 图和汇总图；P01 长直段、P03 上回环/连接段与 P05 高低坡均连续覆盖，未发现 v1 类长空洞。方法、逐图指标和边界记录为 `docs/ANCHOR_SELECTOR_COVERAGE_AUDIT_V2.md`。
- 用户在收到 5 mesh、250 anchors、750 views、17.28M 双场景 rays、零训练和停止边界后，以“继续”批准 v3 corrective 五拓扑 CPU LiDAR pilot；proposal/card 已绑定该批准，旧失败 shard 未拼接复用。
- 新增 v3 executor/runner，强制复核 selector v2 的精确弧长配额、`>=5 m` 全局间距和 `<=7.5 m` 完整 spline 覆盖；legacy v1/v2/v2b 执行器未修改。执行前 `py_compile`、86/86 单元测试、9/9 冻结 E1 外部测试和 preflight 全通过。
- 唯一批准命令无重试完成并封存：5 world bundles、5 native perception meshes、250 anchors、750 observations、5 NPZ shards、5 complete maps、30 contact pages；双独立 scene 共 17.28M primary rays，100 项 evidence hash 全部复核通过，目录约 75 MB。
- 冻结规则总体 branch P/R/F1=`0.8872/0.9431/0.9143`、平均匹配角误差 `3.501°`、分支数完全正确率 `0.912`；逐 topology F1 为 P01 `1.0000`、P02 `0.9246`、P03 `0.8878`、P04 `0.9245`、P05 `0.8374`。
- 已人工检查 5 张完整地图与全部 30 张 contact page。未见空白批次或明显 raycast 损坏；P04 宽 chamber 的过检与 P05 坡道/多高度的漏检和过检与指标一致。复核见 `docs/FIVE_TOPOLOGY_CPU_LIDAR_CONTRACT_V3.md`；不据此回调冻结规则。
- 用户批准 topology-only 120-candidate audit 后，已实现原生 Cano grown/connector read-only adapter、坐标完整 canonical identity、去坐标 WL hash、parent-level split、全部返回记录和 train-only 可视化；flat 层显式把 grown vertical tendency/noise 归零，因为 Cano `flat` 标志只处理 connector。89/89 单元、10/10 冻结 E1 外部测试及 preflight 0 错误/0 警告通过。
- 正式 run `gate0_20260811_cano_100_topology_parent_candidate_audit_v1_seed0` 在 555.74 s 内完成全部 120 个预声明候选并按门槛封存为 FAIL；60 valid/60 invalid，全部失败均为 grown generation failure，没有追加 seed、降低标准或重试正式 run。269 项 evidence hash 全部复核，目录约 44.8 MB。
- 各层 valid/12 为 `7/9/6/9/3/9/3/9/1/4`；flat 合计 20/60，3D 合计 40/60。当前选择只能得到 56 train/4 validation/0 development-test，禁止作为正式 split 或训练数据。
- 60 个成功 parent 的 canonical identity 与 coordinate-free WL hash 都是 60/60 唯一，且 60/60 含 degree>=3 事件；有效结构有真实多样性。已检查 10 张预声明 train-only 完整图，flat z=0，3D 高度跨度与复杂度清晰，无空白或损坏。
- 发现 family metric 缺口：S01 C07 在 0 connector 下 cycle rank=1，S07 C04 在 2 connector 下 cycle rank=3。后续必须从 `>= requested connector` 改为 `== requested connector`；详细复核见 `docs/CANO_100_TOPOLOGY_PARENT_AUDIT_V1_REVIEW.md`。
- 用户在收到 V2 精确范围后连续回复“继续”，已登记为一次执行批准；新增 bounded parameter-resampling executor/runner、严格 cycle 测试和 registry/exclusion/hash 门禁。静态编译、90/90 单元测试、11/11 冻结 E1 外部测试及 preflight 0 错误/0 警告通过。
- 唯一 V2 正式 run 在 1845.62 s 内完成并封存 FAIL：120/120 generation success、120/120 replay identical、114/120 exact-cycle valid，各层为 `11/12/12/12/12/12/9/12/10/12`，只能形成 99 parent 和 80/10/9。
- 936 条 requested-tunnel operation 中 834 条 draw1 成功、102 条需要重抽，平均 draw `1.1357`、最大 `5`，0 条耗尽 20-draw budget。6 个 invalid 的唯一原因均为 extra natural cycle，不再存在 generation failure。
- V2 99 个 retained parent 的 canonical/WL identity 均 99/99 唯一，99/99 含 degree>=3 事件，50/50 3D 高度合同通过；391/391 evidence hash 复核，目录约 101.5 MB，无 mesh/LiDAR/label/formal-data/training/model/trajectory/M-TARE 变更。
- 已逐张检查 10 张 fixed rank-1 train-only 完整 X-Y/X-Z 图：无空白/断裂，5 flat 全 z=0，5 3D 有坡度/多高度；未查看 validation/development-test 地图。完整复核见 `docs/CANO_100_TOPOLOGY_PARENT_AUDIT_V2_REVIEW.md`。
- 只读 counterfactual 不修改 sealed V2：将 strata 改为 dimensionality+grown/connector recipe，保留 actual cycle rank 为 GT metadata，并固定 C01-C10 后得到 100 parent、80/10/10、100 canonical、100 WL、100 structure-event、50 3D。该结果只是待审批 V2R 的设计证据，不是正式 PASS。
- 用户批准的 V2R 已实现并只正式执行一次：93/93 单元、11/11 冻结 E1 external、391/391 sealed-V2 source hash 和 preflight 0/0 均先通过；run 状态 `COMPLETED`，结论 `PASS_CANO_100_TOPOLOGY_PARENT_RECIPE_RECLASSIFICATION_V2R`。
- V2R 固定读取 120 个 sealed candidate，不构造或重放 topology；120/120 recipe-valid，固定 C01--C10 得到 100 parent 和 80/10/10 parent-disjoint split。100 canonical、100 WL、100 structure-event 与 50 个 3D parent 全部通过。
- V2R 只产出 3 个 manifest、3 个 metrics、1 个日志和治理快照；PNG=0，目录 250,816 bytes，17/17 evidence hash 复核通过。mesh/LiDAR/label/formal-data/training/model/trajectory/M-TARE change 全为 0。
- 已完成 100-parent mesh 只读规模审计：中心线总长 `151410.377 m`，单图 `586.797--2980.408 m`，中位数 `1347.694 m`；100 个 reserved geometry seed 全部唯一。为避免直接启动数 GB 未验证批次，方案固定为 M0 先对每个 recipe 的 C01 train parent 做 10 primary+10 replay native mesh，M0 PASS 后才另批 M1 全 100。
- M0 方案、proposal 和 card 已建立但未批准、未实现、未执行；固定 10 个 train parent、15.192 km 中心线、20 次 mesh materialization、10 张完整 train 图、零 validation/dev-test 查看、零 LiDAR/标签/训练，预算 `<=1 h/1.5 GiB`。
- M0 经用户批准后已实现并只执行一次；98/98 单元、12/12 external、17/17 V2R、391/391 V2 和 preflight 0/0 先通过。run 在 S01 primary/replay 后按 exact OBJ hash stop rule FAIL，43.508 s、21.9 MB、33/33 evidence hash，无重试。
- S01 两 mesh 均为 69,908 vertices/139,822 triangles、零 degenerate、单组件、axis exact；双向 nearest-vertex max `0.599708/0.571740 m`，surface-area relative difference `0.0425%`。完整 train 图无空白/断裂。
- 根因是上游 Poisson vertex ordering 不稳定后再按数组顺序加 `Uniform[-0.2,0.2] m` 坐标噪声；三维理论差异上界 `0.692820 m`。已形成只改 replay metric 的 M0R proposal，尚未批准、实现或执行。

## FAILED

- 历史匿名 clone 曾因 GitHub 凭据提示失败；该事实已被“仓库现已公开可读”取代，但旧失败日志保留；
- g000 Isaac RTX headless PNG writer 失败；USD import 本身通过；
- g001 开发中四次候选分别因 CTG/曲线坡度、转弯半径和 genus 合同被否决，最终候选通过；
- 这些 clean-room 结果不构成 Cano Phase 1 PASS。
- Cano 原样 smoke 在第一条 grown tunnel 失败：`geometry.py:262` 将 shape `(N,1)` 的 `dist_array` 传给 SciPy 1.18.0 `splrep`，抛出 `TypeError: only 0-dimensional arrays can be converted to Python scalars`；
- 本次完成 topology/world/point cloud/mesh 数均为 0，没有图片可科学地保存；
- 原 `snippet_2.py` 只构建并显示 graph/spline，不生成或导出 point cloud、mesh、graph JSON；同时没有 seed 接口；
- 依赖声明不锁版本且漏掉脚本直接导入的 `cv2`，因此不能称为开箱即用或环境可复现。
- E1 的第 5 个 connector 耗尽 1000 次 trial 后失败；原脚本丢弃 `(False, None)` 返回并仍退出 0，所以不能从“进程成功”推出“随机 topology 成功”。
- 原生导出的 mesh 虽可读取且非空（61,421 vertices / 122,862 triangles），但分成 7 个组件，非 watertight、非 edge/vertex manifold、不可定向且 self-intersecting，不能作为合格导航/拓扑几何；
- `axis.txt` schema 为 9 列，但 3,888 条 tunnel axis 行的冲突率为 100%：972 个唯一坐标分别被复制到 4 个 tunnel ID，不能作为 per-tunnel GT；
- 原生入口仍不导出 topology graph、不保存生成器成功返回值、无 seed 控制，所以不能证明请求的 tunnel 数、连通性、graph-mesh 一致性或 deterministic replay。
- read-only adapter 已解决返回值、graph/spline 和 per-tunnel axis 的可审计性，但原 Cano Poisson mesh 仍失败：58,825 vertices/117,675 triangles，虽为单组件和 vertex-manifold，仍非 edge-manifold、非 watertight、不可定向且 self-intersecting；
- 先前“因 strict mesh gate 失败所以不能进入 Isaac”的表述已纠正：它直接否决动态导航资格，不直接替代静态传感器测试。
- 24-pose sensor smoke 最终失败：Isaac 报 `MTARE_VLP16_720_50M_V1` config not found，GMO magic number 无效，300 rendered frames 内无 complete scan；RTX capture=0；
- 因 RTX 数据为 0，RTX/CPU 数值比较、24 张 individual RTX 图与两张 contact sheet 均未执行，禁止用 CPU 图冒充完成结果。
- exact-profile v1 总结为 `FAIL_EXACT_PROFILE_CREATION_PROBE`：USDA/checker/read-back 已通过，但直接 GMO 轮询无完整扫描；封存 summary 中 `complete_diagnostic_scans=1` 是由 Kit 异常退出码被掩盖导致的汇总错误，权威物理计数为 0，封存目录不回写。
- Writer v2 仍为 `FAIL_EXACT_PROFILE_CREATION_PROBE`：300 frame 内 callback=0、zero-element=0，shutdown 出现 writer schedule drain timeout；未生成 NPZ/图片，扫描计数正确为 0，并按 stop rule 未重跑。
- 官方 runtime control 为 `FAIL_OFFICIAL_WRITER_RUNTIME_CONTROL_MISSING_EVIDENCE`：`config=Example_Rotary` 在 network-none 下无法访问 NVIDIA 6.0 远端资产根，故没有产生传感器/Writer runtime 证据；该失败不能归因于 headless Writer runtime 或 custom sensor/scene coupling。
- 精确官方 local-USDA Writer control v2 为 `FAIL_OFFICIAL_LOCAL_USDA_WRITER_RUNTIME_CONTROL`：资产、sensor creation、Writer attach 和 300 次同步 update 均真实建立，但 callback=0 且 shutdown 两次 drain timeout。封存分类不回写；结合后续官方源码审计，只能归因到同步 standalone execution path，不能越界声称 async Kit runtime 也失败。
- 五拓扑正式 run `gate0_20260811_cano_five_topology_cpu_contract_pilot_v1_seed0` 为 `FAIL_CANO_FIVE_TOPOLOGY_CPU_CONTRACT_PILOT`：1.35 s 内在首个 world 构造前触发 `imported_from_fixed_checkout=False`。实际 world/mesh/anchor/observation/shard/可视化均为 0；约 38 KB 失败证据已封存，无重试。该结果只说明 runner 导入 provenance 配置错误，不评价五类拓扑、LiDAR 或冻结规则质量。
- corrective v2 `gate0_20260811_cano_five_topology_cpu_contract_pilot_v2_import_fix_seed0` 为 `PRECHECK_FAILED_NOT_EXECUTED`：runner 的工具表仍指向 v1 提案/卡，冻结 v2 hash 不匹配；没有进入 RUNNING，所有物理生成计数为 0。
- v2b `gate0_20260811_cano_five_topology_cpu_contract_pilot_v2b_runner_paths_seed0` 为 `FAIL_CANO_FIVE_TOPOLOGY_CPU_CONTRACT_PILOT`：P01--P03 完成后，P04 的冻结 `select_canonical_anchors` 在 5 m 欧氏间距下只能追加到 38/50 并抛出 `ValueError`；P05 未开始。当前算法是事件优先后做最远点贪心，它得到的是不可继续追加的极大 packing，不保证达到目标 cardinality。P04 四条中心线合计 326.13 m、655 个 spline knots，因此不能把失败归因为地图长度或候选点不足。
- selector audit v1 的机器状态为 PASS，但科研复核 FAIL：按 tunnel 绝对 anchor 数量接近不等于按 arc length 密度均衡，P03 最长 connector 虽分到 11 个点，却有 `92.306 m` 同 tunnel 未覆盖半径。该 PASS 不得升级为“结构采样已正确”或五拓扑 LiDAR 重跑授权。

## BLOCKED

- Cano 仓库 `pyproject.toml` 有 MIT classifier，但固定 commit 中没有 LICENSE/LICENCE/COPYING/NOTICE 正文。继续阻塞第三方源码并入 `src/`、修改后发布、再分发和开源授权声明；
- 已找到可执行的冻结 E1 环境，但原仓库自身仍无 lockfile；只有本项目保存的 E1 constraints/freeze 能复建本次环境；
- read-only adapter、单图标准 bundle、5-topology contract 和 100-topology parent 80/10/10 split 已通过，但 100-parent perception mesh 合同与完整 Phase 1 定量验收尚未完成；
- seed-0 native mesh 已通过单图 CPU `LIDAR_SENSOR_GATE`，但仍被 collision/navigation 门禁否决；感知资格不得冒充机器人可行走资产；
- 冻结规则已有五类固定拓扑的静态 branch 诊断证据，但这五图不是严格未见 topology 测试；在线图仍只在一个开发 world 上运行，跨拓扑 node/edge 泛化尚无证据；
- CPU synthetic 与 Gazebo Classic fixed-pose parity 已建立；真实 LiDAR、噪声与运动域 parity 尚未建立；
- 当前正式数据集、LiDAR 样本、标签和模型均为 `NONE`。
- Phase 2 Data Card 尚未完成并批准，因此 100k 正式 LiDAR 数据、teacher 和训练仍被阻塞；五拓扑 v3 的 750 个观测只能作为诊断合同数据。

## NEXT

唯一下一科研任务：评审 `docs/CANO_100_PARENT_PERCEPTION_MESH_M0_REVIEW.md` 和 M0R proposal。parents/seeds/native method/20 meshes/10 train previews 全不变，只把不成立的 OBJ-byte equality 改成 exact input/axis 与上游噪声上界内的 geometric replay。未经新批准不实现、不重跑；仍不采 LiDAR、不生成正式样本、不训练。

下一任务明确不做：不直接采 100k LiDAR、不把五拓扑诊断 NPZ 并入训练、不启动 CNN/GNN 训练、不把静态 anchor 伪造成在线轨迹、不修改第三方核心、不修改 M-TARE。CPU↔Gazebo parity 和正式 Data Card 仍是正式样本生成前置门禁。

## 2026-08-11 M0F 收口更新

- replay 合同在 M0R4 后正式退休；采用与下游一致的“单次原生 mesh + 单资产质量 + SHA 封存”合同。
- M0F 唯一正式运行 PASS：10/10 train sentinel primary、0 replay、10/10 完整图，耗时约 591.9 s，run 约 192 MiB。
- 汇总为 1,243,070 vertices、2,486,245 triangles、0 degenerate；最差主分量占比 0.9999353709；10 个 OBJ hash 唯一，110/110 evidence hash 通过。
- 十张 X-Y/X-Z 图已逐张人工核验，无明显空白、裁剪、整段缺失、中心线越界或 flat/3D 维度错误。
- scope 仍为：LiDAR=0、label=0、formal dataset=0、training=0、model=0、simulator=0、M-TARE change=0。
- M1 全 100 immutable asset proposal/card 已起草，状态 pending approval；尚未实现、尚未执行。

当前 NEXT：用户评审 `configs/v3/gate0/cano_100_parent_perception_mesh_m1_immutable_assets.proposal.json`。未经批准不写 M1 executor、不创建正式 run，也不跳到 LiDAR/标签/训练。

## 2026-08-11 M1/M1R 全量感知资产收口

- M1 在第 45 个 parent `S05_flat_branch_medium_C05` 检出 1 个严格共线零面积面后按规则停止；45 个 primary、37 张错误标为 M0 的 train preview 和 415/415 hash 已封存，未重试或回写。
- M1R 对全部 100 parent 从头生成，统一执行 `float64 doubled_area <= 1e-12` 面审计，并把预览修正为 `M1R TRAIN ONLY`。
- 唯一正式 M1R run PASS：100 primary、100 sanitation records、0 replay；10 recipe×10、80/10/10 split 精确一致。
- 总计 `12,608,078` vertices、`25,217,511` triangles、0 degenerate；100/100 mesh 与 sanitation PASS，100 个 final OBJ hash 唯一，最差主分量占比 `0.9998534082580015`。
- 本次从头生成未重现 M1 的退化面，实际删除数为 0；这被记录为 Poisson 运行级非确定性，不作为降低质量门槛的理由。
- 80/80 train 完整图按十个 recipe 全部人工核验通过；validation/development-test 图为 0，未用于开发观察。
- 1001 项 run evidence 已封存；正式 LiDAR、label、dataset、training、model、trajectory、online graph 和 M-TARE change 仍全部为 0。

当前 NEXT：只设计固定 pose 的 CPU synthetic ↔ Gazebo LiDAR parity proposal/card，先确定可执行后端、坐标系、pose、匹配方法、阈值和完整可视化。未经用户审批不执行 parity，也不生成正式训练数据。

## 2026-08-12 CPU↔Gazebo fixed-pose parity 收口

- 批准范围：1 个解析 box、3 个 train parent（S01/S06/S10 C01）、24 个固定 pose、24 CPU reference、72 Gazebo 保存帧；每传感器丢弃前 2 帧，零正式数据/标签/训练/M-TARE 修改。
- 实现与回归：21/21 governance+parity unit tests、解析 Gazebo external control 与 frozen preflight 均通过；临时 E1 按既有 Python 3.12.3 / NumPy 1.26.4 / SciPy 1.12.0 / Open3D 0.19.0 / Matplotlib 3.11.1 重建，`pip check` 通过。
- 正式结果：`PASS_CANO_CPU_GAZEBO_FIXED_POSE_LIDAR_PARITY_V1`；24/24 pose、72/72 保存帧、8/8/8 role 全通过。最差 pose valid agreement=1.0、MAE=`8.0223e-6 m`、P95=`2.4796e-5 m`、P99=`4.6730e-5 m`，repeat 最大差=0。
- 证据：4 个 frozen SDF、7 个 diagnostic NPZ、24 份 pose metric、1 张解析图、1 张 pose map、6 张全覆盖 parity page、1 张分层误差图；目录约 7.77 MB，70/70 seal 复核通过，无遗留 parity 容器。
- 人工复核：CPU/Gazebo range 图逐 pose 一致，valid mismatch 全空；parent/role/elevation 分层趋势与机器指标一致。所有图仅为 train-parent parity diagnostic，不进入训练。

当前 NEXT：等待用户确认该 PASS。确认后只能起草正式 Data Card 并先说明样本单位、数量、输入、teacher、split/leakage、成本与可视化；不得自动采样或训练。

## 2026-08-12 Phase 2 Data Card 设计更新

- 用户以“那继续吧”确认 parity 阶段结果并允许起草数据方案；未批准数据导出、teacher 或训练。
- 方案固定 80 train/10 validation parent、20,000/2,500 place cluster、100,000/12,500 frame；C10 和 M-TARE 本操作读取为 0。
- 候选容量为 train 24,117 cluster、validation 3,130 cluster；train 1,076、validation 133 个结构事件均有候选覆盖。
- 第一版学生是 16×720 range/valid，teacher 是 5 m spline outgoing heading + complete mesh LOS；B0 是冻结几何规则，B1 才是后续单独批准的小 CNN。
- 数据口径问题：上游 `distances[]` 与 `points[]` 欧氏弧长在 train/validation 分别差 790.421/105.010 m，单 world 最大 1.181%。推荐以 `points[]` 累计弧长为唯一采样坐标，等待用户明确确认。

当前 NEXT：用户审批完整 Data Card。批准后实现 selector/teacher/exporter 与测试，冻结 run spec 并 preflight；训练仍不在授权范围。

## 2026-08-12 Gate 1 数据导出与科研复核

- 用户批准的唯一 Gate-1 data export + objective teacher run 已执行完成；90/90 worlds、22,500 clusters、112,500 frames/teacher labels、90 Zarr shards 均完成，C10/M-TARE 读取为 0，training/model 为 0。
- 执行器状态为 `PASS_CANO_PHASE2_SUPERVISED_RANGE_DATASET_V1`；运行约 1425.35 s，封存前约 3.34 GB，12,790/12,790 run evidence hash 复核通过。
- train/validation 的 interior/junction/terminal 配额分别精确达到 `16350/3000/650` 与 `2050/375/75`；61 个候选被确定性替换，未发生 quota exhaustion。
- 后置科研复核发现 machine PASS 的 acceptance spec 比批准 Data Card 的证据要求窄：未持久化 112,500 次 independent-scene replay 的逐帧结果/最大误差；未持久化 61 个 rejected candidate 的 ID/原因；样本页缺 valid/teacher/LOS；coverage 图只是数量柱状图；distribution 图缺失。
- 封存 run 不回写，现有数据不重导出。Gate 1 当前为 `GATE_MIXED`，不能进入训练。

当前 NEXT：等待用户决定是否批准一次独立 corrective evidence audit。推荐只读复用现有 dataset 与固定 mesh，复投并补齐审计字段、拒绝明细和科学可视化；预计约 24 分钟、额外不超过约 1 GiB、零训练。未经批准不实现或执行。

## 2026-08-12 Gate 1 corrective audit 与 selector 缺陷

- corrective audit 正式机器 PASS：90 worlds、22,500 clusters、112,500 replay records、61 rejected records、0 replay/teacher failures；两套新 scene 与 sealed range 最大差均为 0，selected ID/order 完全一致，634/575 junction/terminal events 覆盖。
- 原 dataset 12,790 项 seal 前后不变，M1R 1,001 项及新 run 126 项 seal 均通过；耗时约 913.71 s、约 62 MB、零训练/模型/C10/M-TARE。
- 10 complete sample pages、10 spatial coverage figures 和 1 distribution figure 已生成并抽查。正确空间图暴露 V1 selector 按固定候选顺序填 quota 的系统偏置。
- 量化：50/90 world 共 58 条 tunnel 完全无 selected cluster；train 46/80 world/54 tunnel，validation 4/10 world/4 tunnel。已有 selected tunnel 的 world-level 最大覆盖间距中位数 80 m、P95 222 m、最大 330 m，而候选格为 5 m。
- V1 dataset 科研可用性 FAIL，禁止训练；Gate 1 保持 `GATE_MIXED`。corrective audit PASS 不覆盖这个新数据缺陷。

当前 NEXT：等待用户批准零-ray selector-V2 feasibility audit。目标是在不改 80/10、22,500 clusters、角色配额和事件覆盖的情况下，验证每 tunnel 覆盖、按长度分配和同 tunnel 最远点填充能达到的覆盖半径；未经批准不实现、不重导出数据。

## 2026-08-12 selector V2 feasibility 收口

- 用户批准的唯一 zero-ray selector-V2 formal audit 已 PASS；126/126 unit regression、preflight 0/0 后执行一次。
- 固定 90 world、27,247 candidate、22,500 selected、80/10 split 和原角色配额；634 junction 与 575 terminal events 全覆盖。
- 0 world/0 tunnel 缺失；89 world 最大同 tunnel candidate→selected 弧长距离 5 m，1 world 为 10 m；双次 MILP replay 0 mismatch。
- V1 baseline 保持 50/90 world、58 missing tunnels、330 m max gap。V2 与 V1 overlap 18,662，替换 3,838 cluster（17.1%）。
- 正式 run 约 7.11 s、14 MB、10 张 train-only 空间图、112/112 seal；mesh/raycast/data export/training/model/C10/M-TARE 均为 0。
- selector feasibility PASS 不产生训练数据；V1 Zarr 继续禁止训练，Gate 1 保持 `GATE_MIXED`。

当前 NEXT：起草 V2 Data Card 与完整重导出 proposal，明确 80/10 worlds、22,500 clusters、112,500 frames、相同 teacher、约 3.2 GB/25--30 min，并把 replay/rejection/完整科研图直接纳入导出合同。未经用户批准不执行导出或训练。

V2 Data Card/proposal 已起草为 pending。它要求先审计全部 27,247 candidate cluster/136,235 frame，再在 eligible subset 上重新求解精确 V2 约束，最终导出 22,500 cluster/112,500 frame；禁止顺序替补。结构性 approval-probe 为 0 error/0 warning。当前 NEXT 改为用户审阅并明确批准该卡；批准前不实现 exporter、不 preflight、不导出、不训练。

## 2026-08-12 V2R 数据与 Phase 3 多任务训练准备

- V2 首次导出在第 23 world 因 pre-eligibility tunnel quota 超过 eligible capacity 严格 FAIL；失败证据封存，未降低 0.8 m clearance 或改 teacher。
- 获批 V2R 改为每 world 全候选 eligibility 完成后再按 eligible capacity 计算长度比例 tunnel quota。正式 run PASS：90 worlds、27,247 candidates、64 rejected、22,500 clusters、112,500 frames、90 Zarr、19,338/19,338 seal、约 1,933.5 s/3.48 GB；0 C10/M-TARE/training/model。
- 用户重申中心思想后，Phase 3 正式 M1 固定为 `range+valid -> circular CNN -> 128D z_role -> direction/count/role heads`。出口单 head 降为 B1 对标；不把它冒充完整结构语义。
- 标签对齐审计：22,500 cluster 均为 5 frames；112,500 frame/cluster ID 唯一；role/count/heading/Zarr target 0 mismatch；direction peak 与 heading 最大离散误差 `<0.25 deg`。
- 长尾边界：train branch count 5/6 只有 11/2 frames；正式 count 判据只覆盖 1--4，5--6 诊断，不通过加权 loss 或 validation 泄漏伪装充分性。
- B0 在完整 10 validation C09/12,500 frames 上冻结：P/R/F1=`0.79169/0.79977/0.79571`，平均角误差 `8.8399 deg`，count accuracy=`0.91808`；训练模型必须面对这一几何 baseline。
- 已实现 V2R read-only loader、circular CNN、multitask loss、classification/embedding/equivariance metrics；7/7 新测试通过。真实 8-frame GPU forward/loss/backward no-update smoke PASS：521,098 params，输出 `[8,720]/[8,6]/[8,3]/[8,128]`，权重不变、optimizer step=0、training/model=0。
- 训练环境 `/tmp/mtare_phase3_torch290_zarr2187`：PyTorch 2.9.0+cu129、torchvision 0.24.0+cu129、Zarr 2.18.7、RTX 5090；`pip check` PASS，freeze 已入库；不修改 E1 export 环境。

当前 NEXT：用户审阅并批准 `configs/v3/gate2/data_cards/cano_phase3_multitask_structural_semantics_v1.json` 的精确三 seed 训练范围，同时确认 operational Gate 从 1 切换到 2。批准前不创建 Gate-2 run、不执行 optimizer step、不生成 checkpoint；不建图、不改 M-TARE。

## 2026-08-12 Gate 2 v1r3 完整 deterministic 训练与稳定性诊断

- v1r3 完整执行六个子实验，全部 exit code=0；`RUN_STATE=COMPLETED`，执行完整性 PASS，62/62 evidence SHA-256 复核通过，结果约 86 MB，C10/M-TARE/graph/planner 读取或修改均为 0。
- B1 direction F1 为 `0.901932/0.894351/0.903747`。M1 direction F1 为 `0.820010/0.868793/0.873366`，中位数 `0.868793`，最低 seed 仍高于冻结下限；role macro-F1 中位数 `0.875548`，count 1--4 macro-F1 中位数 `0.760694`。
- 同 cluster cosine mean/p05 中位数为 `0.980767/0.899985`，通过；固定遮挡 cosine 中位数为 `0.205687`，失败，因此辅助表示总门失败并形成 `GATE_MIXED`。
- 只读 no-update 诊断表明 6° rotation 的 CPU 最大 logit error 为 `6.68e-6/6.68e-6/7.63e-6`，三 seed 均通过 `2e-5`；同一前向在 CUDA 为毫量级，说明原 GPU maximum-error 判据混入设备浮点误差。该诊断不是新 formal run，不替代后续冻结审计。
- 遮挡失败在 CPU/CUDA 一致，且训练没有 ray-column dropout 或遮挡一致性约束，归因 model/input-robustness，不是系统确定性问题。

当前 NEXT：等待用户明确选择是否批准 Gate-2 corrective 路线。推荐先建立 no-update stability audit 与 train-only ray-column dropout augmentation proposal：数据、80/10 split、teacher、模型、heads、loss、阈值和 checkpoint 规则不变；只修正等变性审计设备定义并增加部署合理的训练输入扰动。未经批准不实现、不训练、不进入 Phase 4。

## 2026-08-13 Gate 2 masking corrective 恢复完成并 GATE_PASS

- 原 corrective 的 seed0/seed1 均以 `exit_code=0` 完成，但主机休眠使外层 `timeout 10860s` 到期；旧 runner 在 seed2 和聚合前退出，旧 `RUN_STATE` 保留为 `RUNNING`，不回写掩盖系统中断。
- 用户批准严格恢复：只读校验旧 seed0/1 的 summary、checkpoint、日志、冻结参数、100000/12500 frame 和零 forbidden reads；在新 recovery run 中仅训练缺失 seed2，再按原门槛聚合。
- recovery run：`results/gate2_representation/gate2_20260813_cano_phase3_masking_corrective_recovery_seed2_v1_seed2`。seed2 完成 18 epochs，最佳 epoch 12；方向 F1=`0.887325`，role macro-F1=`0.882346`，count 1--4 macro-F1=`0.700751`，同 cluster mean/p05=`0.982129/0.900698`，遮挡 cosine=`0.999370`，CPU rotation max error=`9.54e-6`。
- 三 seed 聚合：方向 F1 中位数/seed floor=`0.887325/0.870524`，role macro-F1 中位数=`0.882346`，每类 recall 中位数=`0.962732/0.877867/0.952000`，count 1--4 中位数=`0.746956`，同 cluster mean/p05=`0.982660/0.902049`，遮挡 cosine 中位数=`0.999586`，CPU rotation max error=`9.54e-6`、z cosine 最低=`0.999999821`。
- 结论：方向和全部辅助表示合同通过，integrity PASS，strict-test/M-TARE/graph/planner reads/changes 均为 0；19/19 recovery evidence hashes 复核通过，正式 Gate 结果为 `GATE_PASS`。
- 边界：本结果只完成 Phase 3 局部结构特征/语义学习，不代表在线拓扑或规划收益。不得自动进入 Phase 4；下一步唯一事项是用户审阅并明确确认是否推进。

## 2026-08-13 Phase 4 离线拓扑图只读设计完成，尚未授权实施

- 用户在 Gate 2 PASS 后回复“继续”；该回复确认继续推进设计，但 `docs/PLAN.md` 仍把实施上限固定在 Phase 3，因此未静默启动 Phase 4 实验。
- 只读审计确认：V2R C09 的 12,500 帧是独立 place cluster 五视角集合，不是连续在线轨迹；旧 240 帧图原型只有一个同轨迹调参开发 world且使用规则出口；`z_role` 是结构作用表示，不是 place descriptor。
- 当时形成 `docs/PHASE4_OFFLINE_TOPOMETRIC_GRAPH_PROPOSAL_V1.md`，其中 C08/C09 里程和帧数使用节点直线距离作设计估计。后续正式 C08 geometry contract 已将 C08 修正为 4,773 帧/9,541.643 m；C09 的 15,725 帧/31,442.449 m 同属未审计历史估计，未来不得直接用于执行。C10/M-TARE 读取为 0。
- 主方法固定为 M1D direction/count/role/confidence + 时间持续性 + spatial/role/heading-set association，实际通过后才建 verified edge，未通过方向保留 exit stub。B0 几何规则为 baseline，GT semantic 输入同 graph builder 为 oracle；GNN/RL/learned node score/place recognition 均不启用。
- 当前 NEXT：用户审阅 proposal，并明确授权是否解除 Phase 3 实施上限。批准前不实现 trajectory generator/inference adapter/正式 graph builder，不 raycast、不推理、不建图、不读取 C10、不修改 M-TARE。

## 2026-08-13 Phase 4 C08 trajectory contract PASS，因果拓扑 replay 待批准

- 用户批准保留冻结 C08 资产并采用 `<=0.5 m` 显式 graph-node/spline connector；正式 geometry run 3 worlds/312 edges/624 directed traversals 全部 PASS，精确总长 `9541.643029 m`、2 m 独立世界采样 `4773 frames`。
- connector 最大 `0.361167 m`、对 frozen perception mesh 最小表面净空 `1.148410 m`；29/29 seal 复核，三张完整 XY/XZ 图和 S01 放大图人工检查无断裂。该结论不是动态 collision 资格。
- `topology_replay` 已加入强制 Data Card 治理；连续轨迹/治理组件测试 23/23，分环境全量单元 154/154 PASS。E1 没有 Torch、Torch 环境没有 Open3D，因此正式 replay 固定为文件隔离的 CPU raycast -> GPU inference -> CPU graph 三阶段。
- 已形成 `docs/PHASE4_C08_CAUSAL_TOPOLOGY_REPLAY_PROPOSAL_V1.md` 与 draft Data Card。精确范围：3 C08/3 trajectories/4773 frames/109,969,920 dual-scene rays；同一 B0、M1D seeds0/1/2、GT oracle 和 causal graph builder；243 组预声明 C08 graph grid；预算 0.75 h/1 GiB。
- 当前唯一 NEXT：用户明确批准或否决一次 C08 causal topology replay implementation + execution。未批准前不实现正式 V2 graph builder、不 raycast、不推理、不建图；C09/C10/M-TARE 保持 0。

## 2026-08-13 Phase 4 C08 causal replay 因 S10 局部净空不足 FAIL

- 用户明确批准完整 implementation + execution。C08 frame contract、causal graph V2、双场景 sensor stage、三冻结 checkpoint inference、243 参数 sweep/selection 和正式 runner 已实现；专项单元测试 `7/7`、Data Card、冻结 spec 和 preflight 全部通过。
- 唯一正式 run `gate4_20260813_cano_c08_causal_topology_replay_v1_seed0` 在 sensor stage 严格停止。S01 `801` 帧和 S06 `1414` 帧全部通过；S10 `frame_index=6`、`route_arc_m=12.0` 的双场景最大差为 `0`，branch count=2 且两条 LOS 均通过，但最小水平净空仅 `0.631976306 m < 0.8 m`。
- 失败点位于 `edge_0000 / tunnel_129` 的常规中心线段，axis xyz=`[-208.404237,-93.236766,-31.417293]`；该 traversal connector 最大仅 `0.000213 m`。旧 trajectory mesh contract 只审计 connector，S10 connector minimum=`1.171914 m`，没有审计完整 spline 水平净空，故旧 PASS 与本次 FAIL 不矛盾。
- 影响：4773-frame replay dataset 未成立；M1D inference=`0`、parameter-grid graph replay=`0`、frozen C08 graph tuple=`NONE`。失败 run 约 63 MB，283 项证据封存，不回写。
- 次要 evidence 缺陷：aggregate summary 因 sensor stage 未生成完成 summary，把 partial attempted frame/ray count写为 0；逐 world metrics、failure JSON 和 raw log 正确保留。这不改变数据资格失败，但 replacement runner 必须保存 partial attempted counts。
- 当前 NEXT：推荐先审批一次零推理、零建图的完整 `4773` pose clearance-only audit，量化全部不合格区段和净空分布，再决定重生成几何/中心线还是预声明安全轨迹。禁止只删 frame 6、降低 0.8 m、续跑失败 run 或进入 C09。Gate 4 保持 `GATE_MIXED`。

## 2026-08-13 C08 complete clearance-only audit PASS，定位为 S10 少数局部物理异常

- 用户回复“继续”批准推荐的完整审计。正式前已固定 3 worlds/3 trajectories/`9541.643029 m`/`4773 frames`、每帧 720 条 world-horizontal rays、0.8 m 阈值、零 inference/graph/training/C09/C10/M-TARE；专项测试 `9/9`、preflight 0/0。
- 唯一正式 run `gate4_20260813_cano_c08_complete_clearance_audit_v1_seed0` 完成 `4773/4773` frames 与 `3,436,560` rays，状态 `PASS_CANO_C08_COMPLETE_CLEARANCE_AUDIT_V1`。PASS 只表示诊断证据完整，不表示所有轨迹合格。
- S01=`801/801` pass，minimum=`1.552989 m`；S06=`1414/1414` pass，minimum=`1.415781 m`；S10=`2549/2558` pass，9 个 frame 失败，minimum=`0.605363 m`，pass rate=`99.6482%`。
- 9 个 route failure 只涉及 tunnel 129/16 和 edge 0000/0009/0018/0069。多数是 doubled-edge 正反遍历的同一物理位置重复：edge0000 最近配对 `0.364 m`、edge0018 `0.452 m`、edge0069 `0.008 m`。因此不是 9 个独立危险区域。
- 设计 tunnel radius 为 129=`5.719008 m`、16=`5.223342 m`；低净空命中少数局部三角面，不是设计隧道只有 0.6 m。当前 pose 使用 per-world 固定 FTA 偏移，尚未证明能跟随局部粗糙地面；不能据此决定直接重生成 mesh 或删除 frame。
- 证据：4773-row CSV、4 个 segment JSON、3 world + aggregate metrics、6 张完整图、27/27 seal；人工检查确认 S01/S06 全程高于阈值，S10 红点局部集中且与曲线低谷一致。run 约 2.1 MB，执行约 4.29 s。
- 当前 NEXT：等待用户批准一次只读 S10 局部 floor-following pose feasibility audit。只在已定位物理位置检查竖直地面剖面、可行 sensor/base z 区间和 0.8 m 水平净空；若局部 z 修正可行，则重建完整 collision-feasible trajectory contract；若不可行，才重生成 S10 perception mesh。Gate 4 保持 `GATE_MIXED`。

## 2026-08-13 S10 local floor-pose audit PASS，但 6/9 位姿不能只修 z

- 用户批准了诊断 Data Card 治理修正；33/33 相关测试与 preflight 0/0 通过。Data Card 正确记录 9 个 raw/effective pose，549 是 parameter evaluations，不是独立样本。
- 正式 run `gate4_20260813_cano_s10_local_floor_pose_feasibility_v1_seed0` 审计 9 个 pose/5 个物理组，每个 61 个 z offset，合计 395,280 水平和 1,098 竖直 rays。run COMPLETED/PASS，21/21 seal，零 inference/graph/training/C09/C10/M-TARE。
- frame 6/39/2551 有可行 z 区间；105/108/600/676/2507/2519 无可行 z。这 6 个点在贴地高度下的水平净空为 `0.7355/0.7677/0.7858/0.7887/0.6511/0.5814 m`，因此完整轨迹不能仅靠 z 修复。
- 当前 NEXT：等待用户批准一次只读局部 x/y/z feasibility audit。对 5 个物理组预声明小范围横向网格，每个 x/y 位置自动求 floor-following z，区分“中心线局部偏置”与“mesh 实际不可行”。未批准前不执行；Gate 4 保持 `GATE_MIXED`。

## 2026-08-13 S10 local lateral-pose audit PASS，mesh无需重生成

- 用户批准一次正式局部x/y+floor-z审计。已完成9 pose、3,969 parameter cells、2,857,680水平rays和11,907竖直rays；26/26测试与preflight 0/0通过。
- 唯一正式run `gate4_20260813_cano_s10_local_lateral_pose_feasibility_v1_seed0` COMPLETED/PASS，21/21 seal，约2.42s。9/9 pose都有大面积非孤立可行区，最近修正最大`0.224m`。
- 科研结论：不重生成S10 mesh；问题是旧固定FTA高度和少量局部中心线位姿。但本PASS只是局部诊断，不是完整trajectory eligibility PASS。
- 当前 NEXT：等待用户批准完整4,773-frame floor-following + smooth lateral corrective trajectory contract。新轨迹必须全帧重审水平/上/下净空、连续性、曲率和connector；未批准前不重建、不推理、不建图。Gate 4保持`GATE_MIXED`。

## 2026-08-13 complete trajectory corrective 因 S10 floor-support 孔洞 FAIL

- 用户批准完整4,773-frame纠正；28/28测试、preflight 0/0通过。唯一正式run中S01=801/801、S06=1414/1414通过。
- S10在frame338/2275地面求解时严格停止；两帧是同一edge0065/tunnel3物理位置，down hit=`8.20/8.28m`，而邻帧=`0.93--1.08m`。横向平滑偏移为0，问题是局部mesh floor support，不是纠正核污染。
- run `gate4_20260813_cano_c08_floor_following_trajectory_corrective_v1_seed0` 已FAILED并17项证据封存；推理/图/训练/C09/C10/M-TARE为0。
- 当前 NEXT：等待用户批准对该唯一物理位置做机器人足迹尺度局部x/y floor-support审计，要求非孤立支撑、前后帧高度连续和0.8m净空。禁止跨孔插值或静默重跑；Gate 4保持`GATE_MIXED`。

## 2026-08-13 S10 footprint floor-support audit PASS，结果为0支撑

- 正式run `gate4_20260813_cano_s10_floor_support_hole_audit_v1_seed0` 完成frame338/2275各`41×41`网格，共3,362条floor queries。两个观测均为0支撑、0共同可行格、0连通块。
- 严格逻辑是只有floor z落在`axis_z+FTA ±0.25m`内才发水平/顶部ray，因此horizontal rays=0是物理支撑先决失败，不是执行不完整。
- 29/29测试、preflight 0/0、run COMPLETED/PASS、15/15 seal；零训练/推理/图/C09/C10/M-TARE。PASS仅表示诊断完整。
- 当前 NEXT：等待用户批准S10 perception-mesh可追溯重生成和完整资格链。禁止手工补OBJ、扩大轨迹绕行、删除S10或直接恢复推理/建图。Gate 4保持`GATE_MIXED`。

## 2026-08-13 S10 replacement在生成前因runner环境合同 FAIL

- 用户批准单次replacement；43/43测试和preflight 0/0通过。正式run在导入`subt_proc_gen`时立即失败，生成/materalization/floor ray全为0，11项seal。
- 根因已确认：必须复用历史compat E1和`PYTHONPATH=external/procedural-subt-gen/src:src`；当前runner错用zarr E1并遗漏upstream src。
- 当前 NEXT：等待用户明确批准runner-only修正和一次replacement execution。数据、生成器、seed、parent、阈值和单次materialization规则不变；未批准不修不跑。Gate 4保持`GATE_MIXED`。

## 2026-08-13 runner修正审计发现冻结E1缺失perlin-numpy，未创建正式run

- 用户回复“继续”批准runner-only修正和一次replacement。已新增不覆盖旧失败证据的`v1r` runner、Data Card和spec；43/43单元测试通过。
- 固定checkout、upstream commit `b6c776...`、Python 3.12.3、numpy 1.26.4、scipy 1.12.0、Open3D 0.19.0和matplotlib 3.11.1身份均匹配历史合同。
- 但冻结E1导入`subt_proc_gen.mesh_generation`时失败：`ModuleNotFoundError: perlin_numpy`。历史freeze与安装日志明确要求commit `5e26837...`，当前E1中`pip show perlin-numpy`为不存在，证明/tmp环境已漂移。
- 正式run尚未创建，materialization/Poisson/OBJ/floor ray仍全部为0。当前NEXT是等待用户批准从冻结commit重建新的不可变E1副本；禁止直接污染旧E1或继续执行。

## 2026-08-13 不可变E1重建PASS，但S10原生replacement地面支撑FAIL

- 用户批准重建独立E1并继续。新环境`/tmp/mtare_cano_compat_e1_rebuilt_20260813`按历史98行freeze构建；标准化freeze逐行相等且SHA=`e917255c...`，`pip check`、完整materializer/perlin导入、upstream commit与clean audit均通过。旧E1未修改。
- 43/43单元、1/1冻结环境外部测试、preflight 0/0通过后，唯一正式run `gate4_20260813_cano_s10_perception_mesh_replacement_v1r_seed0`执行一次原生Poisson物化。新OBJ SHA=`f22ebb...`，442,280 triangles、33.8MB；graph/spline/parent/operation/effective-geometry身份全部相等，mesh audit和sanitation通过。
- 完整2558帧floor audit只有2234支持、324不支持，support fraction=`0.8733385`；frame338/2275仍失败。失败分布为53个连续段，最长144帧/286m（arc 894--1180m）。run按冻结2558/2558门槛FAILED，证据seal完整。
- 科研结论：单次同配方随机Poisson重物化不能修复S10完整地面支撑，问题是原生perception mesh方法对动态地面合同不充分，而不是runner或单个偶然OBJ。零训练/推理/图/C09/C10/M-TARE，无重试。
- 当前NEXT：评审并批准资产分离方案：保留原生Cano mesh用于LiDAR/perception；从同一TNG/spline生成独立、可追溯的floor/collision support geometry用于trajectory qualification。未批准前不实现、不生成、不恢复图回放。

## 2026-08-13 geometry-first/fixed-XY feasibility 否定，等待 3D union 决策

- 用户批准 geometry-first、trajectory-afterward 原型，但不批准正式 run。所有工作保持只读/内存预检，未写新轨迹。
- 双向 `±2 m` 所属层搜索确认 V1 独立条带可重新求 floor-following z，但重叠边界产生台阶；简单节点平面融合会造成 S06/S10 最大 `16--33 m` 错误跨层，已淘汰。
- 加入 incident identity、节点局部 spline arc 和三维节点距离后，跨层串扰消除。解析局部下包络仍显示：原重叠要完整覆盖至结构半宽 `1.011×`（S06）/`1.026×`（S10），但固定 XY 连续性在 S10 `0.5×` 已有6个超限步长、最大回归`0.369 m`，S06 `0.75×`起超限。
- 结论：固定 XY、只求 z 的方案没有同时满足重叠覆盖和连续性的尺度区间。当前唯一 NEXT 是用户决定是否切换到局部 3D swept-volume/voxel union，并允许在路口局部联合求 XY/z；未批准前不实现、不生成正式资产、不推理、不建图。

## 2026-08-14 3D swept-volume union 设计获批，尚未获执行许可

- 用户回复“继续吧”，批准 Gate 4 方案 B 的设计与实现准备，明确边界为不生成正式资产、不写轨迹、不执行回放。
- 新提案固定为 `docs/PHASE4_C08_3D_UNION_TRAJECTORY_PROPOSAL_V1.md`：C08 仅 S01/S06/S10、624 次定向遍历、4,773 个既有 2 m frame；从 TNG/spline/radius/FTA 先生成独立 collision/support asset，在显式 incident 节点局部做 3D void union，非 incident 近接为失败而非合并；随后只在节点局部联合优化 XY/z。
- 分辨率 ladder 固定为 `0.10/0.05/0.025 m`，要求 topology isolation、局部 mesh integrity 与 4,773-frame pass/fail identity 全部收敛。任何分辨率分歧、开放/非流形表面、非 incident overlap 或 clearance/continuity fail 都停止。
- 当前唯一 NEXT：实现准备完成后提交精确 Data Card 与 one-shot qualification spec；用户再次明确批准前，禁止 materialization、trajectory write、preflight/create_run、4773 帧 audit、M1D/B0 inference、graph、C09/C10 和 M-TARE。

## 2026-08-14 C08 3D union Data Card/spec 草案已提交，等待执行批准

- 已冻结为草案而非 run：`configs/v3/gate4/data_cards/cano_c08_3d_union_trajectory_qualification_v1.proposal.json` 与 `configs/v3/gate4/cano_c08_3d_union_trajectory_qualification_v1.proposal.json`。
- 草案绑定 9 个 TNG/spline/geometry JSON 和 3 个 sealed trajectory NPZ 的 SHA-256，精确限定 S01/S06/S10、4,773 原始 frame、三档 `0.10/0.05/0.025 m` 分辨率、CPU 估算 2 小时/2 GB。native perception OBJ 不进入新 collision geometry。
- 两个 JSON 均为 `PENDING`/`DRAFT_AWAITING_EXPLICIT_EXECUTION_APPROVAL`，不是可 preflight 的正式 spec，且 command 仍明确标记为 pending implementation。没有创建 run dir、没有执行 `preflight.py`/`create_run.py`、没有物化 mesh/trajectory/audit。
- 当前唯一 NEXT：用户审阅并明确批准上述一项 C08 one-shot qualification 的数据范围、方法、成本与全部 pass/fail 证据；批准后才实现 runner、冻结工具 hash、preflight 并创建一次正式 run。

## 2026-08-14 C08 3D union 执行前环境合同阻塞

- 用户已授权一次性资格执行；在实现/冻结工具 hash、preflight、create_run 或资产物化前，执行前环境检查发现历史 Gate 4 runner 指向的 `/tmp/mtare_cano_e1_zarr2187/bin/python` 与 `/tmp/mtare_cano_compat_e1_rebuilt_20260813/bin/python` 均已不存在。
- `python3`、`/usr/bin/python3` 与 `/home/zeng-workstation/anaconda3/bin/python` 都确认 `open3d` 不可导入；后者有 `scipy/skimage`，但不满足既定 Open3D RaycastingScene safety audit 合同。没有运行命令、结果目录、mesh、trajectory、ray、inference 或 graph 输出。
- 这是 system/environment identity 问题，不能静默用不同几何/raycast backend 或向现有 Anaconda 环境安装包来绕开。受影响结论是 C08 3D-union qualification 尚未开始；Gate 4 保持 `GATE_MIXED`。
- 当前唯一 NEXT：用户决定是否授权从已封存的兼容 freeze 新建独立、不可变的 Gate-4 qualification E1，固定 Open3D/scipy/skimage 版本并作 import/pip/source-hash 验证；该系统恢复不执行 qualification 本身，恢复通过后再重新冻结工具 hash、preflight 并使用本次已经批准的一次资格执行。

## 2026-08-14 Gate 4 qualification E1 恢复 PASS，资格实验尚未执行

- 用户批准后，已从 `configs/v3/gate1/environments/e1_zarr2187_pip_freeze.txt` 在 `/tmp/mtare_gate4_qualification_e1_20260814` 建立新的独立 E1。封存 freeze 与重建排序 freeze SHA 均为 `50144a4...841cb82e`，`pip check` 为 `No broken requirements found`。
- import identity 通过：Python `3.12.3`、NumPy `1.26.4`、SciPy `1.12.0`、Open3D `0.19.0`、Matplotlib `3.11.1`、Zarr `2.18.7`、Numcodecs `0.15.1`。身份记录位于 `configs/v3/gate4/environments/gate4_qualification_e1_20260814.json`。
- 首次误选到实际为 Python `3.13.5` 的 Anaconda，因其不兼容 sealed Python 3.12.3 freeze 已停止；部分目录移至 `/tmp/mtare_gate4_qualification_e1_20260814_py313_failed` 保存失败知识，未删除或混入新 E1。
- 没有 preflight/create_run、collision geometry、trajectory、ray、LiDAR/inference/graph 或 C09/C10/M-TARE 读取。当前唯一 NEXT：实现资格工具、冻结其 hash、将已批准草案改为正式 approval、preflight 后创建并执行一次 C08 3D-union qualification。

## 2026-08-14 稠密 3D-SDF resolution ladder 成本合同失败，停止实现

- 在实现前按已冻结总路线长 `9,541.643029 m` 和保守 5 m tunnel radius 计算，三条完整 C08 swept void 的体积下界为 `749,398.891 m³`。因此稠密 3-D grid 的体素下界为：`0.10m=749,398,891`、`0.05m=5,995,191,129`、`0.025m=47,961,529,029`。
- 仅连续 float32 SDF 不含网格索引、occupancy、marching-cubes 临时内存或 mesh，就分别需 `2.79/22.33/178.67 GiB`；与草案的 CPU 2 小时/2 GB 不相容。该问题是 method/system cost contract，不是数据、teacher、model 或 metric 问题。
- 已停止在 runner、tool hash、preflight/create_run 和 materialization 之前；0 asset、0 trajectory、0 ray、0 inference/graph。现有 Data Card/spec 的完整世界稠密 resolution ladder 不能用于正式执行，Gate 4保持`GATE_MIXED`。
- 可选路线：A（推荐）重写方法为解析 swept-tube 全局边界 + 仅显式 junction-local bounded SDF windows，并对每个窗口三档收敛；资格 ray 可仍覆盖 4,773 帧，但这改变 asset/convergence 定义，需新 Data Card/spec/approval。B 使用稀疏/自适应 SDF，成本较低但不能再声称 uniform `0.025m` convergence。C 停止动态 geometry 路线。当前唯一 NEXT 是用户选择 A/B/C。

## 2026-08-14 local implicit-union V2 草案已提交，尚未获执行许可

- 用户“继续”按推荐路线解释为只批准起草方案 A。新方法文档为 `docs/PHASE4_C08_LOCAL_IMPLICIT_UNION_PROPOSAL_V2.md`，新 Data Card/spec 草案为 `configs/v3/gate4/data_cards/cano_c08_local_implicit_union_qualification_v2.proposal.json` 与 `configs/v3/gate4/cano_c08_local_implicit_union_qualification_v2.proposal.json`。
- 只读拓扑核验固定 25 个 degree>=3 窗口：S01=4、S06=7、S10=14；最大 incident radius=`5.979884m`、最大局部球半径=`6.577872m`。全局是轨迹无关的解析 swept-tube field，三档 SDF 仅按窗口串行处理；非 incident overlap、窗口重叠或 seam/manifold failure 都是 FAIL。
- V2 估算为峰值 RAM 4 GiB、磁盘 12 GiB、CPU 12 小时，替代已否定的完整世界 2 GB/2 小时稠密 SDF。两个 JSON 均为 PENDING/DRAFT；没有 runner、asset、trajectory、preflight/create_run 或 qualification。
- 当前唯一 NEXT：用户确认 V2 的混合 safety-query backend、25-window 范围、成本和验收。确认后才可实现工具、冻结 hash、升格 card/spec 并请求/执行 one-shot qualification。

## 2026-08-14 V2 patch-mesh extraction backend 缺失，停止实现

- 在已恢复、freeze 一致的 E1 中确认 `skimage`、`mcubes`、`vtk`、`trimesh`、`numba` 全部不可用。项目唯一已有等值面提取器 `src/mtare_topo/data/worldgen/tunnel_mesh.py` 使用 Python `set` 保存每个 occupied voxel，并逐体素/逐 cube 枚举；其实现不适用于 V2 最大局部窗口约 146 million cells。
- 因此不能满足 V2 要求的 deterministic patch mesh、seam 和 manifold 证据，也不能假装以 Open3D 替代（Open3D 当前合同只提供 raycasting，未提供该 scale 的受控等值面提取）。未写 runner、未冻结 hash、未 preflight/create_run、未生成任何 asset 或 trajectory。
- 这是 system/backend + method-evidence blocker，不是数据/teacher/model/metric失败。当前 V2 execution authorization 不足以指定新 meshing backend；不得静默安装新库、重写为 C++/Numba，或删除 patch-mesh/seam criterion。
- 当前唯一 NEXT：用户选择 A（推荐：批准在独立、版本锁定的 meshing sidecar 环境安装 `scikit-image` 并以其 marching-cubes 作为 V3 方法/证据的一部分），B（批准实现并审计自有 streaming extractor，开发成本更高），C（将 V2 改为隐式场 query-only，放弃 patch-mesh dynamic asset claim），或 D（停止）。

## 2026-08-14 version-locked scikit-image meshing sidecar 恢复 PASS

- 用户“继续”按推荐选项 A 执行。独立 sidecar `/tmp/mtare_gate4_meshing_sidecar_v1` 从 E1 freeze 重建，并只额外固定 `scikit-image==0.24.0`；原 immutable E1 未修改。
- `pip check` 通过；Python `3.12.3`、NumPy `1.26.4`、SciPy `1.12.0`、Open3D `0.19.0`、scikit-image `0.24.0` 与 `skimage.measure.marching_cubes` import 均通过。sidecar sort-freeze SHA=`6447ba5e...19c4035`，身份记录为 `configs/v3/gate4/environments/gate4_meshing_sidecar_v1.json`。
- 环境恢复不等于 V2 geometry 通过：没有实现 V2 tool、无 mesh/trajectory/ray/preflight/create_run/inference/graph。当前唯一 NEXT：为 sidecar marching-cubes 写 synthetic topology/seam/reproducibility contract，成功后仍须冻结 V3 Data Card/spec 并取得新的 execution approval。

## 2026-08-14 sidecar synthetic marching-cubes contract 实际 PASS

- 已实际运行 `/tmp/mtare_gate4_meshing_sidecar_v1/bin/python tools/v3/verify_meshing_sidecar_contract_v1.py`。固定 65³、0.1m sphere SDF 的两次 `skimage.measure.marching_cubes` 提取 hash 完全相同（`6261a4d7...7bc111`），7,446 vertices、14,888 triangles、0 degenerate triangles，所有 mesh edge incidence=2。
- 该 PASS 只证明锁定 sidecar 在合成闭合场上具有确定性、无退化与闭合 surface；它不证明 C08 incident union window 的 seam、non-incident isolation 或 trajectory qualification。C08 asset/trajectory/ray/inference/graph 仍为0。
- 当前唯一 NEXT：实现并测试 C08 local implicit union tool（source membership、window overlap、patch-to-analytic seam、hybrid query）后，冻结新 spec/card 和工具 hash，preflight/create_run，再按用户“运行”授权执行一次正式 qualification。

## 2026-08-14 incident tunnel 与 graph-edge 语义不等，停止 C08 window 实现

- 新增的 window 单元测试 2/2 通过；C08 read-only audit 在 S01 `node_0010` 停止。证据表明该节点 `degree=3` 但 `incident_tunnel_ids=[135,139]`：degree 是图边数，incident_tunnel_ids 是去重的物理隧道 ID。一条物理隧道可在同一节点贡献两条 graph edge；S06/S10 也普遍有该模式。
- 影响：V2 只按 unique tunnel ID 做 local union 虽可得到物理 void，但不能确定每一条入射 graph edge 在 spline 上的 node-local arc/方向，因而不能满足“仅 node-local incident spline arcs”或证明 patch-to-analytic seam。不能静默删除 degree check、把 ID 复制为 degree 个，或用整个 tunnel 替代 arc。
- 这是 topology-GT/geometry-method interface 问题，不是模型/teacher/metric失败；0 SDF/mesh/trajectory/ray/inference/graph。当前唯一 NEXT：用户决定是否批准一个只读 edge→spline projection/arc-incidence audit 来定义唯一、可重放的 node-local arcs（推荐），或改写 V2 的几何语义/停止。

## 2026-08-14 C08 edge→spline arc-incidence read-only audit PASS

- 用户批准后，`edge_arc_incidence` synthetic tests 3/3 通过；只读执行器对 312 条 sealed C08 edges 产生精确 624 条 endpoint projection records。S01/S06/S10 分别为 108/184/332 条，所有记录 arc 有限且每 edge 固定映射到一个已有 physical tunnel spline。
- 最大 connector error 为 S01=`0.361167487m`、S06=`0.002012064m`、S10=`0.003076177m`；S01 值与既有 <=0.5m connector contract 相容。junction endpoint records 为 13/21/42，保留同一 physical tunnel 在节点两侧的不同 graph-edge away tangent。
- 本次输出序列化首次因 `numpy.bool_` 停止，修正为 Python bool 后在完全相同输入/规则下重跑；无 SDF、mesh、trajectory、ray、inference/graph，故这不是正式 Gate run 或参数适配。
- 当前唯一 NEXT：将这个 per-edge arc-incidence contract 纳入 local implicit union 的 V3 定义和合成 seam test；之后仍需新的卡/spec/execution approval，才可 materialize C08 geometry。

## 2026-08-14 local implicit union V3 arc contract drafted

- 新增 `docs/PHASE4_C08_LOCAL_IMPLICIT_UNION_PROPOSAL_V3.md`。V3 固定以 `(edge,node,tunnel,arc,projection,away tangent,connector error)` 定义 junction patch；同一 physical tunnel 的相反 graph-edge arcs 保留为独立 records，不能折叠。
- 下一步唯一工作是 V3 through-tunnel+branch synthetic seam contract；仍无 C08 asset、trajectory、ray、inference 或 graph，materialization须新 card/spec/approval。

## 2026-08-14 V3 patch seam/watertightness 合同矛盾，停止 materializer

- 实现前最小复现确认：一个半径 1m、贯穿 8m local cube 的解析 cylinder SDF 经锁定 `scikit-image==0.24.0` marching cubes 提取后得到 5,508 vertices、10,880 triangles、136 条 boundary edges，`watertight=false`。这正是所有入射 tunnel 穿过 junction-window 边界时的必然情况。
- 先前 V3 synthetic PASS 将 arc endpoints 放在 cube 内，实际验证的是闭合 finite capsules，不是 analytic-global ↔ local-patch seam。若用 window sphere/cube surface 封口可得到 watertight mesh，但封口会成为阻断 tunnel 的虚假 collision wall。
- 因此“仅物化 local patch”同时要求“patch 自身 watertight”与“对外保持 tunnel 连通”不可兼得。该问题否定当前 V3 materializer/evidence contract，不是数据、teacher、模型或指标问题。0 C08 SDF/mesh/trajectory/ray/preflight/create_run。
- 选项 A（推荐）：正式资格使用连续 hybrid implicit field，边界证据改为 field value/gradient continuity，不声明 local patch 自身 watertight；patch mesh只作可视化。未来 simulator collision mesh另立 Gate。选项 B：实现全局解析 tube mesh并与 local patch共享边界三角化，保留 watertight dynamic asset，但开发和验证成本显著增加。选项 C：统一 sparse global implicit meshing，改变成本和方法。需用户明确选择并重写 Data Card/spec。

## 2026-08-14 C08 hybrid implicit qualification V3 正式 run 因 traversal JSON loader FAIL

- 用户选择方案 A 并明确要求实施完整 V3 计划。已实现 typed 624 arc-endpoint contract、25-window seam/volume-isolation、三档连续场 qualification、75 个 visualization-only patches、逐帧 route identity 和一次性 runner；7/7 专项测试通过，正式 Data Card/spec 的 preflight 为 0 error/0 warning。
- 唯一正式 run `gate4_20260814_cano_c08_hybrid_implicit_qualification_v3_seed0` 在 0.271 s 后停止。executor 用治理层 `load_json()` 读取根节点为 list 的 sealed `S01..._traversals.json`，该函数只接受 object，抛出 `ValueError: JSON root must be an object`。runner 返回 `FAIL_CANO_C08_HYBRID_IMPLICIT_QUALIFICATION_V3`、`RUN_STATE=FAILED`，14/14 文件已 seal。
- 这是 implementation/system loader mismatch，不是 geometry/data/teacher/model/metric 结论。正式计数为 worlds/windows/frames=`0/0/0`，因此不能回答 4,773 帧安全问题，不能解除 C08 geometry blocker，也没有执行 LiDAR、推理、图更新、C09/C10 或 M-TARE。按 one-shot/no-retry 合同未修冻结工具、未创建第二个 run。
- 当前唯一 NEXT：用户决定是否授权一个新的 V3R corrective run。推荐只把 traversal 读取改为标准 `json.load()` 并增加真实 list-root fixture/test，其他数据、方法、阈值、sidecar 和验收完全不变；重新冻结工具 hash、另建 Data Card/spec/run ID 后只执行一次。也可选择停止该路线。未经明确批准不得修复后重跑。

## 2026-08-14 C08 hybrid implicit V3R 在 visualization patch 退化三角形处 FAIL

- 用户批准严格 loader-only corrective；专用 list-root loader 与直接读取 sealed S01 108-record traversal manifest 的回归测试加入后，专项测试 9/9、V3R preflight 0/0，通过新的 Data Card/spec 创建并执行了唯一 run `gate4_20260814_cano_c08_hybrid_implicit_qualification_v3r_seed0`。
- loader corrective 生效。正式 run 依次完成 S01 node_0010/node_0015/node_0027 的 0.10m patches；node_0039 patch 触发冻结的 zero-degenerate stop condition并停止。只读复核得到 node_0039=`56,414 vertices / 111,526 triangles / 2 degenerate / 1,302 boundary / 0 nonmanifold edges`。前三个 patch 均为 0 degenerate/0 nonmanifold；boundary edges 是方案 A 预期的 visualization clipping。
- runner 约 5.79s 后封存 `FAIL_CANO_C08_HYBRID_IMPLICIT_QUALIFICATION_V3R`，`RUN_STATE=FAILED`，18/18 seal 验证通过。正式 worlds/windows/frames 仍为 0/0/0，未开始 trajectory qualification，故 4,773 帧安全问题仍无答案；零推理、图、C09/C10、M-TARE。
- 这是 visualization meshing/numerical-integrity failure；它不证明 authoritative continuous field 不安全。推荐下一步为新 V4 合同：marching-cubes 后确定性删除面积 `<=1e-12m²` 的 visualization-only faces并记录数量，重新验证 0 nonmanifold；连续场、输入、25 windows、三档分辨率和安全阈值不变。替代方案是维持 zero-degenerate hard gate并停止本路线。任何新执行都需用户再次明确决定。

## 2026-08-14 C08 hybrid implicit V4 在 S06 trajectory continuity 处 FAIL

- 用户批准 visualization sanitation-only V4。确定性 face sanitation 和 sealed V3R node_0039 回归加入后，测试 11/11、preflight 0/0；正式 run `gate4_20260814_cano_c08_hybrid_implicit_qualification_v4_seed0` 越过此前 loader 与 patch-degeneracy 阻塞。
- S01 三档均 801/801 安全帧通过、resolution identity 与 continuity PASS：最小 horizontal=`3.012562m`、最大 floor error=`0.002244m`、最小 up=`9.712936m`。S06 三档也均 1414/1414 安全帧通过且 identity PASS：最小 horizontal=`2.170422m`、最大 floor error=`0.010400m`、最小 up=`9.024221m`。
- S06 continuity FAIL：baseline maximum step=`1.999999973m`，冻结上限=`2.099999973m`，corrected maximum step=`2.139792493m`。最大点在 frame 141→142；z correction 从 `-4.661210m` 跳到 `-4.141146m`，相邻 correction jump=`0.520064m`。当前 executor 只做逐帧 support-Z 与 zero-XY，没有实现原计划要求的、受 continuity 约束的一阶/二阶联合优化。
- 发现后按 stop condition 手动终止 executor；runner 封存 `FAIL_CANO_C08_HYBRID_IMPLICIT_QUALIFICATION_V4`，67/67 seal，约 3904.68s、305MiB。已完成并保存 S01/S06 共 2,215 帧三档审计与 47 patches（S01=12、S06=21、S10 coarse=14）；S10 尚无 frame audit。零推理、图、C09/C10、M-TARE，无重试。
- 分类为 trajectory-optimizer implementation/method failure，不是 collision-field safety failure；S01/S06 全部已审计帧的 clearance contract 实际通过。推荐新 V5：实现真正的 window-local constrained XY/z optimizer，按位移→一阶→二阶词典序，明确把 continuity 写入约束，并在每个 world 后立即执行 stop；不改 geometry、frames、阈值或分辨率。需用户新决定；也可停止路线。

## 2026-08-14 V5 optimizer 实现预检发现 S10 窗口外 non-incident support 合并，不创建 formal run

- 用户批准 V5 optimizer corrective。新增 deterministic window-local constrained support/XY solver，13/13 tests PASS。只读 S01/S06 feasibility PASS：S01 maximum step=`2.010913m`；S06 只需 maximum XY correction=`0.019826m` 即把 maximum step 压到冻结 `2.099999973m`，finest-field floor error=`0.046273m`。
- S10 feasibility 在 formal Data Card/spec/preflight/create_run 前停止。固定 25-window membership 下有 29 个 step violations，其中 frame 341→342 与 2271→2272 两对均为 window-outside/window-outside，support z 分别跳变 `11.037345m` 与 `11.170081m`；仅垂直差已大于 `2.099999990m` 上限，任何 window-local XY/z optimizer 均不可行。
- 两对都在 sealed `edge_0065 / tunnel_3` traversal。frame 341 的 global-union floor z=`-9.122926m`，zero surface由非 incident tunnel 135 提供（SDF=`0.001506m`），route tunnel 3 在该点 SDF=`10.716186m`；下一 frame 342 floor z=`1.911425m` 才由 tunnel 3 提供。反向 traversal frame 2271→2272 重复同一切换：floor `1.935993m`→`-9.231038m`，frame 2272由 tunnel135而非route tunnel3提供。
- 这是原计划明令 hard-fail 的 window-outside non-incident overlap/union，不是 optimizer 收敛问题。扩大窗口、按轨迹过滤 hit、合并 tunnel 或放宽 continuity 都会改变冻结方法，故 V5 未创建 card/spec/run，0 formal assets/rays。
- 当前方法结论：单一 global analytic tube union 不能作为 S10 topology-isolated support field。若继续，推荐新设计为 edge-conditioned/layered analytic field：在 geometry 阶段冻结每条 edge 的 physical-tunnel layer，并由保留的 traversal identity选择层；junction window仍只union incident edge arcs。该方法不再是单一global union，需新的 Data Card/spec与用户明确批准。替代方案是接受当前 geometry route FAIL并停止。

## 2026-08-14 layered hard-switch seam audit FAIL，不创建 formal run

- 用户批准推荐的 edge-conditioned/layered redesign 后，实现了逐点/逐 ray sealed tunnel layer dispatch：窗口外只查询 traversal physical tunnel，兼容窗口内查询 incident union，非兼容 layer 永不被窗口合并。新增 stacked-tunnel、compatible-window、ray-layer tests 后专项测试 `16/16 PASS`。
- 正式化前只读审计覆盖 S01/S06/S10 的 `4/7/14=25` 个窗口、`0.10/0.05/0.025m` 三档。所有 window semantics audits 精确 PASS，但每个世界、每档均为 `0/N` seam PASS；最大 selected-layer 与 incident-union 球面开口 field error 分别达到 S01=`5.422588m`、S06=`5.433231m`、S10=`5.608039m`，远超采样间隔容差。
- 该结果证明 hard switch 虽排除了 tunnel135 对 tunnel3 的窗口外 support 抢占，却在 junction 球面制造不连续 collision/support field，触发原计划的 open-seam hard FAIL。按 stop policy 未继续 S10 trajectory feasibility，未冻结 Data Card/spec，未 preflight/create_run，正式 frames/assets/rays 均为0。
- 当前唯一 NEXT：用户选择新方法。推荐固定窗口半径不变、用无新增长度超参的径向 smooth transition 令球面等于 selected layer、节点中心等于 incident union，并先做 field/gradient seam 与三档只读 proof；备选为降低成 actual-ray-only seam（证据更弱）或停止 geometry route。

## 2026-08-14 V6 continuous-layered 正式 qualification 因 coarse floor 不一致 FAIL

- 用户批准的无参数径向 cubic transition 已实现：窗口球面严格恢复 selected traversal layer，节点中心恢复 incident union，权重及球面一阶导数为0。专项测试 `18/18 PASS`；S01/S06/S10 的25个窗口在三档均为 seam/dispatch PASS，最大边界值误差 `2.89e-15m`、公式误差 `2.00e-14m`、non-incident error=0。
- 只读 finest feasibility 中 S01/S06/S10 最大步长=`2.000286/2.072088/2.099999990m`；S10 max XY correction=`0.070142m`，原 edge0065/tunnel3 两组11m窗口外跳变已消失。正式 V6 Data Card/spec 绑定3 worlds、312 edges、624 traversals、4773 frames、25 windows、50 window-layers、150 patches，preflight 0/0。
- 唯一正式 run `gate4_20260814_cano_c08_continuous_layered_qualification_v6_seed0` 完成 S01 全三档 `801/801` safety、continuity与resolution identity；24个 S01 patches 全通过。S06 0.10m 14 patches通过，但10个窗口帧 floor FAIL，maximum down=`1.082591m`；minimum horizontal/up=`2.874159/9.034661m` 均安全。失败来自只在0.025m将target down固定1.035m，0.10m同轨迹产生至多约0.0476m额外downward distance。
- executor按首个 safety failure停止；runner=`FAIL_CANO_C08_CONTINUOUS_LAYERED_QUALIFICATION_V6`、RUN_STATE=`FAILED`、55/55 seal，duration=`1686.66s`、pre-seal bytes=`252,622,233`。未进入S06中/细档或S10正式审计，无重试，无C09/C10/模型/图/M-TARE。
- 当前唯一 NEXT：用户决定是否批准V7三分辨率共同可行区间optimizer（推荐），或改解析centerline field/停止。V6目录和冻结工具不得修改或续跑。

## 2026-08-15 V7 optimizer-only 预检被 ray-exit overshoot 数值误差阻塞

- 用户批准推荐V7后，已实现三档逐帧support-z区间交集、closest-to-sealed-z最小位移投影、窗口内continuity XY与post-XY再次投影；新增共同区间PASS/empty-interval FAIL测试，专项总计`20/20 PASS`。
- 只读结果：S01/S06共同区间最小宽度=`0.0875/0.051579m`并通过；S10共同区间始终存在，最小宽度约`0.05138m`、最大三档surface spread约`0.04862m`。因此不是数据或三档几何无共同floor。
- S10仍不收敛：floor error在约`0.0500m`与`0.084--0.087m`间振荡，post-projection maximum step约`2.11686m`。代码审计确认`LayeredHybridField.ray_exit_distances`在首次`sdf>=0`时直接返回travel；0.10m场最小march step为0.05m，没有保存last-inside/first-outside bracket或二分，返回值随ray origin phase产生厘米级overshoot。
- 该问题分类为ray-distance numerical implementation，不否定continuous transition或共同区间方法。按stop policy未把ray solver静默改成二分；无V7 card/spec/preflight/create_run/asset。当前唯一NEXT是用户决定是否批准V7R fixed-iteration bracketed ray exit（推荐）或停止。

## 2026-08-15 V7R bracketed ray exit 正式 run 因审计边界浮点不一致 FAIL

- 用户确认持续目标后，V7R 加入固定32次 last-inside/first-outside 二分；专项测试`21/21 PASS`。三世界只读 proof 均一轮收敛，S10 maximum step=`2.096518403m`、maximum XY correction=`0.070501093m`，三档最大surface spread降至`0.000328076m`。
- 新Data Card/spec绑定原3 worlds、312 edges、624 traversals、4773 frames、25 windows与原安全阈值；preflight 0/0。唯一正式run `gate4_20260815_cano_c08_bracketed_ray_exit_qualification_v7r_seed0` 在S01 0.10m全帧审计按stop rule FAIL，`RUN_STATE=FAILED`、22/22 seal、39.46s，无重试。
- 只读归因确认112帧只因down上界失败：range=`[1.049770668506,1.050000000010]m`，最大仅比`1.05m`高`9.525e-12m`；minimum horizontal/up=`3.182715561/9.653976750m`，二者无失败。optimizer收敛判据允许`FLOOR_TOL+1e-9`，但final audit使用严格`<=FLOOR_TOL`，且closest-point投影把帧放在精确上边界。
- 分类为trajectory-optimizer/final-audit numerical-contract mismatch，不是geometry/data/teacher/model或物理安全失败。当前唯一NEXT：用户决定是否批准V7R2在共同support interval两端使用固定`1e-8m`内部数值裕量、保持最终审计与物理`[0.95,1.05]m`合同不变（推荐），或停止；不得修改/续跑V7R目录。

## 2026-08-15 V7R2 dynamic collision/support qualification 正式 PASS

- 用户明确回复“继续”批准V7R2。optimizer仅将三档共同support interval两端收缩固定`1e-8m`；最终horizontal>=0.8m、down in `[0.95,1.05]m`、up>=0.8m合同、场、窗口、分辨率、帧和连续性阈值均未改变。专项测试`22/22 PASS`，三世界完整只读proof通过。
- 唯一正式run `gate4_20260815_cano_c08_support_interval_margin_qualification_v7r2_seed0` 完成并返回`PASS_CANO_C08_SUPPORT_INTERVAL_MARGIN_QUALIFICATION_V7R2`。3 worlds、25 windows、624 arc endpoints、624 directed traversals、4773 frames、14319 frame-resolution audits、10,309,680 horizontal rays、42,957 vertical rays、150 patches全部通过。
- S01/S06/S10三档分别`801/1414/2558`全帧PASS；finest minimum horizontal=`3.183091/2.940001/1.836959m`，maximum floor error=`0.049999990000/0.049999990003/0.049999990003m`，minimum up=`9.654101/8.963620/9.134100m`。S10 corrected maximum step=`2.096518403m`，低于`2.099999990m`上限。
- 正式耗时`14417.84s`、结果约`1.5GiB`，RUN_STATE=`COMPLETED`，178/178 seal复核；C09/C10/M-TARE/model/inference/graph reads或updates均为0。该PASS仅解除C08 geometry eligibility，不代表Gate 4 PASS。
- 当前唯一NEXT：以V7R2三条sealed corrected trajectories恢复冻结M1D/B0/oracle因果sensor→semantics→online graph replay，保持既有C08 graph方法、参数网格和Gate-4边界；先核对原失败replay的runner/spec并新建不可覆盖corrective run。

## 2026-08-15 collision/perception pose compatibility 检查失败，causal replay 停止

- 正式回放前核对原sensor executor、V7R2 corrected trajectory和native mesh绑定。V7R2 sensor相对原native-perception sensor的z位移范围：S01=`-4.599496..-4.153639m`，S06=`-4.723445..-3.722062m`，S10=`-3.987098..-3.060038m`；S10最大XY修正=`0.070501m`。
- 用冻结S10帧`0,1,2,3,4,5,6,10,100,1000,2000,2557`执行无写入native-mesh raycast；12/12 objective branch LOS失败，多帧720条水平射线零命中并返回50m。有限range输出不能把场外/地板下观测变成有效LiDAR。
- 该问题属于geometry/data-interface system blocker：V7R2 geometry PASS仍有效，但“独立动态场 + native mesh感知”的联合causal contract不成立。0新formal run、0 M1D inference、0 graph update、0 C09/C10/M-TARE读取。
- 后续contract review进一步确认根因是全轨迹support relocation，而不是必须更换LiDAR资产；推荐路线已更新为分离collision boundary与route-conditioned support，并保持native perception pose/domain。

## 2026-08-15 V7R2 window-only contract review FAIL

- 冻结spec写明`Optimize only window frames`。实现却在window mask生效前对全部轨迹运行support solve。以正式`complete_frame_audit.csv`为权威，窗口内共466帧，窗口外4307帧；窗口外`4307/4307`均被修改z：S01=`721/721`、S06=`1286/1286`、S10=`2300/2300`，总范围`-4.716372..-3.060038m`。
- V7R2 sealed run和`RUN_STATE=COMPLETED`保持不可变；32次bracket ray、continuous field seam/isolation和实现内安全审计仍可复用。但原计划的window-local trajectory和causal replay eligibility未被证明，项目层状态固定为`FORMAL_MACHINE_PASS_RESEARCH_CONTRACT_FAIL`。
- 当前唯一NEXT：用户批准推荐corrective——保留layered side/ceiling collision field，downward support改为沿sealed route spline的`centerline_z + fta_distance_m`，窗口外pose严格不可变，仅窗口内允许联合XY/z。随后先做三世界完整只读proof，再走新Data Card/spec/唯一formal run。详见`docs/C08_V7R2_CONTRACT_REVIEW.md`。

## 2026-08-15 route-conditioned support corrective 已实现，只读双 proof PASS，等待正式执行批准

- 新增类型化`RouteConditionedSupportField`与独立incident edge support arcs。窗口外仅查询当前traversal tunnel spline的`z+fta_distance_m`；窗口内按冻结node membership只组合incident edge arcs并沿用既有C1 radial transition。teacher/graph/sensor三坐标已显式分离。
- 相关测试`33/33 PASS`。完整只读geometry proof覆盖三世界、三分辨率和4773帧：25 windows、312 edges、624 endpoints/traversals、466 window frames精确匹配；4307个窗口外graph pose逐元素不变；14319 frame-resolution audits全部安全且pass/fail一致。全局minimum horizontal=`4.955093m`、minimum up=`5.241558m`、maximum down error=`7.11e-15m`。
- native Cano双场景proof覆盖4773/4773帧：所有full scans finite、objective branch LOS完整、两CPU scenes最大差=0。S10有9帧旧native horizontal-clearance诊断失败，但独立dynamic collision/support审计均PASS，native clearance不再冒充安全权威。
- 重建Torch2.9.0+cu129/Zarr2.18.7 sidecar，freeze SHA-256与历史`fccb0fe6...23ae`一致；三个M1D checkpoint hashes也一致。新geometry Data Card/spec已冻结，preflight为0 error/0 warning。
- 尚未调用`create_run.py`，未创建正式run、未执行model inference或graph replay。当前唯一NEXT是用户批准一次不可覆盖geometry正式执行；其PASS后才可绑定corrected trajectory hashes并preflight causal replay。

## 2026-08-16 V8 formal run 被监控误判错误终止，正式结论 FAIL

- 用户明确批准后，preflight再次0 error/0 warning，创建并执行唯一run `gate4_20260815_cano_c08_route_conditioned_support_corrective_v8_seed0`。输入、工具和sidecar freeze核对通过。
- 进程跨主机/会话暂停后的`ps etime`显示约18h27m，但runner日志的冻结`time.monotonic()`有效运行时间仅`4328.326607s`（约72.14min），未超过12h cap。监控时错误地把日历经过时间解释为有效执行时间，并向仍合法运行的executor发送SIGTERM。
- runner因此以`executor_exit_code=-15`封存`FAIL_CANO_C08_ROUTE_CONDITIONED_SUPPORT_CORRECTIVE_V8`，`RUN_STATE=FAILED`，77/77 seal复核一致，pre-seal结果约525.7MB。该FAIL属于agent/operator process-control error，不是geometry/data/teacher/model/metric或安全失败。
- 终止前完成60/150 patches；S01正式三档801/801帧全部PASS、721个窗口外pose逐元素不变、0 XY correction。S06完成0.10/0.05m审计并进入0.025m patch；S10未开始。无模型推理、graph replay、C09/C10或M-TARE读取。
- V8不可续跑或覆盖。当前唯一NEXT：用户决定是否批准新的V8R one-shot operational recovery（推荐：方法、数据、阈值、工具不变，只使用新run ID重新完整执行，并禁止人工根据`ps etime`干预），或接受本次正式FAIL并停止geometry路线。

## 2026-08-17 V8R route-conditioned support formal qualification PASS

- 用户回复“继续”批准新的one-shot V8R operational recovery。几何方法、输入、窗口、阈值和分辨率不变；runner新增完整process-group清理并明确12h为monotonic-active cap。相关测试`33/33 PASS`，process-group termination smoke PASS，preflight 0 error/0 warning。
- 唯一正式run `gate4_20260816_cano_c08_route_conditioned_support_corrective_v8r_seed0`完成并返回`PASS_CANO_C08_ROUTE_CONDITIONED_SUPPORT_CORRECTIVE_V8R`。3 worlds、25 windows、624 arc endpoints/traversals、4773 frames、14319 frame-resolution audits和150 patches全部通过。
- S01/S06/S10分别`801/1414/2558`全帧PASS；minimum horizontal=`5.349942/5.000771/4.955093m`，maximum floor error=`1.11e-16/7.11e-15/7.11e-15m`，minimum up=`5.500462/5.241558/6.066669m`。三世界maximum XY correction均为0。
- 466个window frames保持资格合同，4307个window-exterior poses逐元素完全不变；三档pass/fail一致且continuity PASS。正式有效运行`14044.69s`，pre-seal约1.510GB，`RUN_STATE=COMPLETED`，178/178 seal复核无差异。
- 零inference、graph update、training、C09/C10和M-TARE读取。该PASS解除C08 geometry replay资格，但不代表Gate 4 PASS。当前唯一NEXT：用三条V8R trajectory SHA-256冻结新的causal replay Data Card/spec，preflight后取得一次正式replay批准。

## 2026-08-17 C08 causal topology replay V2 preflight PASS，等待正式执行批准

- 已冻结`cano_c08_route_conditioned_causal_topology_replay_v2` Data Card与run spec，并绑定V8R三条trajectory、三份native mesh、Torch/Zarr sidecar、三个M1D checkpoint及全部执行工具SHA-256。
- 精确执行范围为3个C08 development worlds、4773个唯一sensor frames、109,969,920条双场景full-scan rays、14319个冻结模型inference frames及243参数组×5方法×3世界=`3645`次causal graph replay；training/optimizer step、C09/C10和M-TARE读取均为0。
- `python3 tools/v3/preflight.py --spec configs/v3/gate4/cano_c08_route_conditioned_causal_topology_replay_v2.json`返回0 error/0 warning。正式run目录尚未创建，当前唯一NEXT是用户明确批准一次不可覆盖正式回放；批准前不得调用`create_run.py`或执行sensor/inference/graph stage。

## 2026-08-17 C08 causal topology replay V2 因 checkpoint mode 大小写合同错误 FAIL

- 用户在preflight与成本/证据展示后回复“继续做”，批准唯一正式run `gate4_20260817_cano_c08_route_conditioned_causal_topology_replay_v2_seed0`。run已创建、执行并不可变封存为`FAILED/FAIL_CANO_C08_ROUTE_CONDITIONED_CAUSAL_TOPOLOGY_REPLAY_V2`，539/539 SHA-256 seal复核一致。
- sensor stage完整PASS：3 worlds、4773/4773 frames、109,969,920 dual-scene full-scan rays；finite scan、teacher eligibility、objective branch LOS和dual-scene identity全部通过。S10的9帧native clearance仅按冻结合同记录为诊断。
- inference在seed 0、0 frame处停止。checkpoint SHA-256精确匹配`55f6602...f9fe`，且内部`seed=0`正确；checkpoint保存的`mode='M1D'`，executor却要求`mode == 'm1d'`，触发`checkpoint identity mismatch`。seed1/2只读核对同样保存大写`M1D`且hash/seed正确。
- 分类为system/interface validation bug，不否定geometry、sensor、checkpoint权重或teacher。正式结果为0 inference frames、0 graph replays、约144MB证据；禁止续跑/覆盖或静默修复。
- 当前唯一NEXT：用户决定是否批准推荐V2R replacement——只把身份检查改为冻结事实`mode == 'M1D'`（或等价显式大小写规范化），补单元测试、重新冻结工具/spec与preflight，再新建一个完整one-shot run；预计仍约1.25h/1GiB。备选为接受V2 FAIL并停止Gate 4 replay。

## 2026-08-17 C08 causal topology replay V2R formal PASS

- 用户回复“修啊”批准推荐V2R完整replacement。唯一修复为checkpoint身份合同严格匹配训练器冻结值`mode='M1D'`；错误小写及错误seed均被回归测试拒绝，replay/graph测试`10/10 PASS`，三个真实checkpoint metadata smoke全部PASS。数据、权重、模型、阈值、243参数网格和评分权重未改变。
- 新Data Card/spec冻结且preflight 0 error/0 warning；唯一run `gate4_20260817_cano_c08_route_conditioned_causal_topology_replay_v2r_seed0`完整重算，不读取V2中间资产，返回`PASS_CANO_C08_ROUTE_CONDITIONED_CAUSAL_TOPOLOGY_REPLAY_V2R`。
- sensor=`4773/4773`frames、`109,969,920` dual-scene rays全部PASS；inference=`14,319`frames、3 checkpoints、0 training/optimizer step；graph=`243×5×3=3645`replays，所有grid rows oracle integrity PASS，C09/C10/M-TARE reads均为0。
- 冻结参数ID=`sf2_tr8_lr6_hh20_te45_da20`：stable frames 2、minimum event travel 8m、loop merge radius 6m、branch heading merge 20deg、turn event 45deg、distance anchor 20m；selection score=`0.8262623347`。
- mean C08 composite为B0=`0.743541`、M1D seeds0/1/2=`0.847797/0.824948/0.806043`、oracle=`0.847775`；所有方法mean connectivity与verified-edge correctness均为1.0。结果约172.4MB、有效总耗时1174.31s，RUN_STATE=`COMPLETED`，583/583 seal复核一致。
- 该PASS只冻结C08 development参数，不代表Gate 4 PASS，也不授权C09/C10或planner/M-TARE。当前唯一NEXT是用户审阅V2R科研结果并决定是否另立严格C09 validation方案；不得自动推进。

## 2026-08-17 C08 V2R 科研复核完成：development PASS with caveats

- 只读复核3645行sweep与15张selected graphs，未新增ray/inference/replay，C09/C10/M-TARE读取为0。完整结论见`docs/PHASE4_C08_V2R_RESEARCH_REVIEW.md`。
- 冻结tuple均分`0.826262`，但heading merge=`20/25/35deg`三组完全并列最优，12组距最优不超过0.005；按world排名为S01/S06/S10=`3/48/51`。选择是预声明词典序确定性选择，不是统计唯一参数识别。
- 三seed M1D相对B0的mean composite/exit-F1绝对提升=`+0.082722/+0.157219`，但S06上seed1/2 exit-F1低于B0，seed2 composite低`0.006678`，故只允许aggregate-improvement结论。
- S10的M1D mean composite/exit-F1最低，为`0.792206/0.602592`；三世界均过生成结构节点。oracle exit-F1仍高`0.085622`，且oracle composite不是数学上界，禁止声称模型超过GT oracle。
- 科研判断固定为`C08_DEVELOPMENT_PASS_WITH_ROBUSTNESS_CAVEATS`，Gate 4保持`GATE_MIXED`。当前唯一NEXT：用户决定是否批准只读建立精确C09 continuous trajectory/geometry/sample-count contract并起草一次性validation Data Card；历史C09估计不得直接执行。

## 2026-08-17 C09 read-only continuous contract audit PASS

- 用户回复“继续啊”批准只读审计全部10个C09 validation worlds。只读原有graph/spline/native mesh/geometry，未生成trajectory/LiDAR/model/graph资产；C10/M-TARE读取为0。
- 精确合同为1023 nodes、1027 edges、2054 directed traversals、31654.752491m、15833 frames、71 degree>=3 windows、62 terminals、2054 arc-incidence records和213 incident support arcs。历史31442.449m/15725 frames分别低估212.303491m/108 frames，正式禁用。
- 所有edge正反各一次；max connector=`0.004336588m`，min connector-to-mesh distance=`0.884980679m`；1331 window frames/14502 exterior frames，window overlap/membership conflict/nonincident frame均为0，support finite且唯一。审计与manifest内部一致性PASS。
- C09 geometry qualification固定规模为47499 frame-resolution audits、34199280 horizontal rays、47499 up/support queries和639三分辨率patches。按C08正式耗时线性估算约16.62h，原12h cap不足。
- 当前唯一NEXT：用户决定是否批准推荐的单一不可覆盖C09 geometry qualification设计/实现/preflight/执行，预声明24h active cap、RAM<=4GiB、disk<=8GiB；数据、方法、0.10/0.05/0.025m、窗口、阈值和FAIL policy不变。未批准不生成正式trajectory或geometry证据。

## 2026-08-17 C09 geometry qualification V1 在输入预检阶段 FAIL

- 用户回复“继续吧那”批准一次24h/4GiB/8GiB的不可覆盖C09 geometry qualification。C09专用materializer/runner已实现，相关测试`27/27 PASS`；Data Card补充声明未读取的C08 development与C10 strict-test角色后，preflight为0 error/0 warning。
- 唯一run `gate4_20260817_cano_c09_route_conditioned_geometry_qualification_v1_seed0`启动后，在任何executor或C09 geometry计算之前因run spec把既有C08 seal误写为根目录`EVIDENCE_SEAL.sha256`而触发`FileNotFoundError`；实际文件为`artifacts/evidence_sha256.txt`，其冻结SHA-256值本身正确。
- run已固定为`FAILED/FAIL_CANO_C09_ROUTE_CONDITIONED_GEOMETRY_QUALIFICATION_V1`并以10条SHA-256记录封存；C09 worlds/frames read=`0/0`，geometry audits、LiDAR、inference、graph、training、C10和M-TARE均为0。该结果不评价C09几何或模型。
- 当前唯一NEXT：用户决定是否批准推荐V1R replacement，仅修正冻结输入路径并增加存在性回归检查，重新冻结Data Card/spec、preflight并创建一个新run；或接受V1 system/config FAIL并停止。不得修改或重启V1目录。

## 2026-08-19 — C09 geometry V1R 被主机重启中断，V1R2获批

- V1R完成7/10 worlds、8311/15833 frames和258/639 patches后，主机于2026-08-19 13:15附近异常重启；进程消失，RUN_STATE停留在过期的`RUNNING`，且不存在最终metrics、summary或seal。已完成世界未发现安全、连续性或三分辨率一致性失败，但部分结果不足以形成C09资格结论。
- 该问题分类为外部system interruption，不是geometry/data/teacher/model/metric失败。V1R目录保持原样，不续跑、不覆盖、不拼接、不封存为完整结果，也不复用其中任何patch或逐帧资产。
- 用户已批准全量V1R2 replacement：仍使用10 worlds、1027 edges、2054 traversals、71 windows、15833 frames（1331 window/14502 exterior）和639 patches；方法、阈值、三档分辨率及24h/4GiB/8GiB上限不变，并加入`sleep:shutdown`阻断。
- 重建sidecar后，冻结依赖哈希`6447ba5e...c4035`和meshing确定性合同均PASS，35项相关测试全部PASS。当前唯一NEXT是V1R2 preflight后创建并执行一个不可覆盖run；C09 inference和graph replay仍禁止自动启动。

## 2026-08-19 — C09 geometry V1R2 formal run started

- preflight为0 error/0 warning；按用户批准只创建一次`gate4_20260819_cano_c09_route_conditioned_geometry_qualification_v1r2_seed0`并从原冻结输入完整启动。
- RUN_STATE已为`RUNNING`，10个世界的trajectory/traversal/connector资产均重新生成，`systemd-inhibit`确认对sleep/shutdown生效；当前进入S01三分辨率patch提取。
- 唯一NEXT是持续核对15833 frames、639 patches、零安全/连续性/分辨率/pose-locality失败并在完成后验证seal。不得自动启动C09 inference或graph replay。

## 2026-08-20 — C09 geometry V1R2 executor PASS / formal runner FAIL（patch计数合同错误）

- V1R2完成10/10 worlds、15833/15833 frames、47499三分辨率逐帧审计和34199280条水平射线；安全、连续性、三档一致性、非incident隔离及14502个窗口外pose精确不变全部PASS，C10/M-TARE/model/graph读取或执行为0。
- executor按71个窗口的实际unique physical tunnel layers和3档分辨率正确生成426个patch。逐窗口层数分布为65个双层、3个三层、3个单层，总计`65×2+3×3+3×1=142` physical window-layers，故`142×3=426`。冻结只读合同此前把213条incident support arcs误当成213个physical layers，预注册成`213×3=639`；runner因此将完整计算标记为正式FAIL。
- RUN_STATE=`FAILED`，结果4.1GB，506/506 SHA-256 seal复核通过。该问题属于data/evidence-contract计数缺陷，不是geometry、trajectory、teacher、model或安全失败；V1R2不可修改、不可重标PASS，C09 replay资格仍未获得。
- 当前唯一NEXT需要用户决定：推荐新建只读corrective evidence audit，绑定V1R2 seal并证明426的物理层语义及复核全部既有证据（分钟级、无需重算）；备选完整V1R3重算（约10.35小时/4.1GB）或接受FAIL停止。不得自动启动C09 replay。

## 2026-08-20 — C09 corrective evidence audit V1 因 list-root loader 实现错误 FAIL

- 用户持续目标授权推进完整流程；真实manifest回归测试确认71 windows的physical-layer分布为65个双层、3个三层、3个单层，总计142，故正确patch合同为`142×3=426`。Data Card/spec与preflight 0/0完成后只创建并执行一次正式audit。
- executor在证据评价前把list-root `window_manifest.json`交给只允许object root的治理`load_json()`，立即抛出`ValueError`。运行0.028s，geometry generation、frame audit replay、LiDAR、inference、graph、training、C10和M-TARE均为0。
- V1 corrective audit已不可变标记`FAILED`，11/11 seal通过；V1R2源seal执行前后均为506/506、SHA=`722409d3...9f004`，源资产无修改。问题分类为system/implementation loader defect，不否定426语义或V1R2几何证据。
- 当前唯一NEXT需用户批准loader-only V1R replacement：仅对三个list-root manifest使用显式`json.loads`并增加真实main-path回归测试，其他Data Card、输入、426合同和只读边界不变；不得在V1目录重试或自动启动C09 replay。

## 2026-08-20 — C09 corrective evidence audit V1R formal PASS

- 用户要求继续推进完整流程，批准loader-only V1R replacement。唯一实现变化是保留JSON清单声明的根类型；真实main-path回归与相关测试`3/3 PASS`，preflight为0 error/0 warning。
- 唯一正式run `gate4_20260820_cano_c09_geometry_evidence_corrective_audit_v1r_seed0`返回`PASS_CANO_C09_GEOMETRY_EVIDENCE_CORRECTIVE_AUDIT_V1R`：10 worlds、71 windows、142 physical window-layers、426 three-resolution patches、15833/15833 frames和47499 resolution audits全部通过，失败帧为0。
- 1331个窗口帧与14502个窗口外帧计数一致；全部窗口外pose精确不变。V1R2源seal执行前后均为506/506、SHA=`722409d3...9f004`，未修改任何源资产；新run 15/15 seal通过。
- 该PASS恢复C09 causal replay资格，但不改写V1R2与V1的正式FAIL，也不代表Gate 4 PASS。当前唯一NEXT：冻结已训练M1D三个checkpoint、B0、GT oracle和C08选定参数，设计并预检一次C09 causal topology replay；不训练、不调参、不读取C10或M-TARE。

## 2026-08-20 — C09 causal topology validation V1 formal PASS，科研结论 MIXED

- 用户要求持续完成整条流程。新C09专用sensor/inference/graph runner严格固定10 worlds、15833 frames、三个既有M1D checkpoints和C08 tuple `sf2_tr8_lr6_hh20_te45_da20`；C09参数搜索为0。相关合同测试`14/14 PASS`，Torch sidecar重建freeze SHA仍为`fccb0fe...b23ae`，preflight 0/0。
- 唯一run `gate4_20260820_cano_c09_causal_topology_validation_v1_seed0`返回`PASS_CANO_C09_CAUSAL_TOPOLOGY_VALIDATION_V1`：15833 sensor frames、364792320 dual-scene rays、47499 inference frames、50/50 graph replays全部通过；0 training/optimizer step，C10/M-TARE读取为0。
- 所有方法在10个世界的connectivity和verified-edge correctness均为1.0。mean composite：B0=`0.805246`，M1D seed0/1/2=`0.816083/0.825419/0.807749`，oracle=`0.850069`。三seed M1D均值相对B0 composite `+0.011171`，但exit-F1为`-0.013899`；terminal reachability提高`+0.123228`，node redundancy quality下降`-0.041508`。
- 按世界的三seed平均composite有5/10低于B0（S03/S05/S06/S07/S08）；因此正式机器PASS不等于模型一致优于基线，科研结论固定为`C09_FROZEN_GRAPH_VALIDATION_MIXED`。
- 总运行217.41s、pre-seal约546.3MB，RUN_STATE=`COMPLETED`，1881/1881 seal复核通过。10页可汇报PDF已更新，包含C09实际B0/M1D/oracle拓扑图。当前唯一NEXT是用户审阅并决定是否另行授权C10严格测试；不得自动读取C10、运行planner或M-TARE。

## 2026-08-20 — 整篇论文目标恢复，Phase 5 正式实现开始

- 用户明确要求持续完成“结构语义→在线拓扑→替换 M-TARE 全局规划→单/多机器人实验→论文分析”整条流程；旧的 Phase-4-only 范围已在 `PLAN.md` 中标为被覆盖，但 Gate 与正式运行治理继续有效。
- 接口复核确认替换单元是整个 `tare_planner_node`，保留独立 `localPlanner`、`pathFollower`、terrain、避障、状态估计、传感器和控制；公共主输出仍为 map-frame `/way_point`。
- 已新增正式纯规则规划模块：只在 trace-verified graph edges 上做 Dijkstra；候选只来自 `observed` exit stub；效用显式分解为结构潜力、图路程和失败重试惩罚；目标选择和 tie-break 完全确定。
- 已新增不依赖 ROS 的 M-TARE handoff 合同，覆盖 map-frame waypoint、完成状态、逐周期 runtime 和原 TARE 的“两次空 free_paths 后发布 8m map clearing”恢复语义。
- 规划/接口、shadow/proposal与 causal graph 回归合计15/15 PASS；15 张真实 C08 B0/M1D/oracle 最终图均能被新规划器读取并产生目标，未沿预测但未验证的边导航。
- 当前这些是实现和只读接口证据，不是 shadow 或 closed-loop 实验结论。下一步唯一任务：实现并冻结 C08 逐帧 shadow replay Data Card/spec，精确为3 worlds、3独立轨迹、4,773 unique frames、5 methods、23,865 planner cycles；preflight 后展示成本与门禁，再申请一次正式执行批准。C09/C10/M-TARE closed-loop读取为0。
- 非正式全量内存预检完成23,865 cycles且双重重放一致，但发现M1D seed0/1/2在S06/S10分别有`24/10`、`24/44`、`10/0`帧短暂无frontier，oracle S01也有71帧；随后又恢复TARGET。故“一帧无frontier即发布exploration_finish”被否定。规划器现在只输出`CANDIDATE_COMPLETE`，handoff必须收到独立稳定完成确认才能发布finish；proposal冻结后相关测试15/15 PASS。该预检不冒充formal shadow结果。

## 2026-08-20 — Phase 5 C08 formal shadow PASS 与真实 ROS 输入合同通过

- 用户把本轮“继续推进、把整个流程跑出来”绑定为已展示范围内的一次 C08 Gate-5 shadow 执行批准。Data Card/spec正式冻结后，preflight为0 error/0 warning；唯一不可覆盖run为`gate5_20260820_cano_c08_topological_planner_shadow_v1_seed0`。
- 正式结果为`PASS_CANO_C08_TOPOLOGICAL_PLANNER_SHADOW_V1`：3 worlds、4773 unique frames、B0/M1D seeds0/1/2/oracle共23865 planner cycles全部完成；全部waypoint有限且不超过4m，远程route只使用trace-verified traversed edges，15条method/world流的第二次完整replay hash全部一致。
- 暂时无frontier的帧保持`CANDIDATE_COMPLETE`而没有发布完成：M1D seed0=`34`、seed1=`68`、seed2=`10`帧，oracle=`71`帧；其后恢复TARGET的历史证据仍成立。run不生成新LiDAR/推理/训练，不读C09/C10，不启动M-TARE closed loop；30/30封印复核通过。
- 新增`registered_scan -> 16x720`部署适配。历史external-cave bag含2909组精确同时间戳scan/odom；跨630.66s抽取31帧、位移覆盖82.12m。只用yaw时坡面帧最多4710点错环，确认必须使用完整odometry四元数；修正后31/31帧错环为0，缺失格严格为50m+invalid mask，ROS Python3.8导入通过，相关回归18/18 PASS。
- 当前唯一NEXT：准备M-TARE development worlds的GT-TNG Oracle可行性与原M-TARE公平基线合同；在新Data Card/spec/preflight和一次性批准前，不执行Gazebo closed loop，不读C10。
- 后续只读资产核对发现：M-TARE内置`tunnel/garage`只有完整simulation mesh/pointcloud与launch，没有Cano/TNG原始graph；因此在这些同world公平比较中不能直接称`GT-TNG Oracle`。推荐把上界严格改名并实现为“由完整开发世界地图冻结的GT topological oracle”，仍与原M-TARE同world/start/seed/localPlanner/runtime；备选是把C08 TNG mesh导入M-TARE，但其native floor/support已知不具备任意闭环轨迹资格，成本和失效风险显著更高。该Oracle来源变更需用户明确确认后才能冻结下一份Data Card。

## 2026-08-20 — M-TARE闭环前置实现完成，正式checkpoint转换待批

- 已实现真实ROS在线链：exact-time `/registered_scan`+`/state_estimation_at_scan`、完整四元数range raster、冻结M1D推理、因果图、topological frontier、`/way_point`/`/runtime`/`/map_clearing`输出；localPlanner/pathFollower/control保持原M-TARE。handoff V2精确复现每两次空`/free_paths`触发8m清图并重置，空语义帧发布当前位置hold而非沿用陈旧目标。
- 历史checkpoint在ROS Python3.8/Torch2.0加载失败，错误为`pathlib._local`模块缺失；根因是Torch2.9 checkpoint携带Python3.13 `Path`配置对象，属于serialization/system兼容，不是模型或权重失败。
- 已实现无损部署转换器：只保留42个CPU tensor和primitive metadata，legacy protocol-2序列化；合成checkpoint已由Python3.8.10/Torch2.0.1成功加载，42个tensor hash一致且Torch2.9/2.0固定probe四类输出最大误差均为0。
- M-TARE历史分层teacher的生成脚本缺失，现存合同又混用`terrain_map_ext`与complete-pointcloud表述，禁止直接复用为Oracle证据。已新建pose-conditioned layered GT-map oracle：按车辆Z选support layer、0.5m邻层连续性传播、stacked floor隔离、完整地图局部出口重建。6/6合成测试通过；tunnel/garage各25个均匀历史pose与旧teacher出口数一致41/50，MAE约0.22，且无零出口异常。
- 已实现method-independent coverage evaluator，严格复现M-TARE `visualizationTools.cpp`的0.5m registered-scan unique-voxel explored volume，并记录coverage-time AUC、路径长度和重复观测。完整preview-map surface recall明确只作diagnostic，不冒充已资格化reachable denominator。
- 相关集成回归35/35 PASS。正式转换proposal为`configs/v3/gate5/m1d_ros_deployment_export_v1.proposal.json`，模拟批准后的preflight为0 error，仅提示infrastructure operation不强制Data Card。范围为3 checkpoints、126 tensor hashes、0 dataset frames、0 training/optimizer、预计<5分钟/30MB；等待用户明确批准后才去除proposal并创建唯一run。

## 2026-08-20 — 三个M1D模型的ROS无损部署转换正式PASS

- 用户在完整输入、范围、成本和门禁展示后回复“批准”及“全部都批准”；该授权仅绑定本次三个冻结checkpoint的单一不可覆盖部署转换，不外推为未来闭环或C10授权。
- Data Card/spec正式冻结，preflight为0 error；唯一run `gate5_20260820_m1d_ros_deployment_export_v1_seed0`返回`PASS_M1D_ROS_DEPLOYMENT_EXPORT_V1`。
- seed 0/1/2各42个tensor、共126个tensor hash在源文件、部署文件及ROS Torch 2.0加载后全部一致；Torch 2.9与ROS Torch 2.0固定probe输出满足`<=1e-6`且方向符号、数量argmax和角色argmax决策一致。
- 运行4.414秒，训练/optimizer step均为0；不读取dataset、trajectory、world、C10或M-TARE观测，不执行图或机器人。三个ROS checkpoint及22项证据全部封存并独立复核通过。
- 当前唯一NEXT：完成固定Gazebo seed的公平启动合同、layered GT-map oracle ROS节点与三方法统一记录器的非材料实现/测试；之后再为tunnel/garage闭环冻结精确Data Card/spec并单独申请正式执行批准。

## 2026-08-20 — 固定Gazebo seed与Oracle ROS链完成，原TARE内部随机源仍阻塞公平闭环

- 新增`configs/v3/gate5/roslaunch/vehicle_simulator_seeded.launch`和`system_seeded.launch`，原system组件保持不变，仅向Gazebo注入`--seed`；后者明确不启动`tare_planner_node`。ROS Noetic在冻结镜像内完整展开成功。
- 新增`tools/v3/layered_gt_map_global_node_v1.py`：与M1D节点使用相同exact-time scan/odom cadence、因果图、规划器、handoff及`/way_point`等公共接口，仅将语义预测替换为完整地图的pose-conditioned layered oracle。ROS Python3.8导入通过，相关回归28/28 PASS。
- 公平性审计确认原TARE仍有三个有效随机路径：single-robot local coverage viewpoint采样使用`std::random_device`，multi-robot grid sampling也使用`std::random_device`，PBS/CBS冲突排序使用`rand()`；因此Gazebo seed PASS不等于完整baseline seed PASS。
- 问题分类为baseline/system reproducibility，不否定模型、Oracle或ROS接口，但阻止正式M-TARE/M1D/Oracle统计比较。推荐对冻结TARE源码增加单一`planner_seed`参数并将上述生成器统一绑定，保留算法分布；需用户明确批准该baseline可复现性补丁后实施和验证。
- 统一记录合同`closed_loop_recording_topics_v1.json`及其解析/审计器已实现，三方法强制记录同一组原始scan、双频odometry、waypoint、runtime、finish、free-path、local path、cmd、terrain和恢复证据，禁止下采样。源码核对发现原TARE完成话题实际为`/sensor_coverage_planner/exploration_finish`，M1D/Oracle已纠正到相同话题；闭环相关回归更新为30/30 PASS。
- 单机器人development矩阵proposal已精确展开：tunnel/garage两个冻结世界、5个环境seed、原M-TARE/三个M1D训练seed/Oracle共5个variant，严格配对为50个600秒case，总仿真时长30000秒。三个ROS checkpoint hash均复核一致，C09/C10为0；矩阵验证与闭环回归更新为32/32 PASS。proposal仍显式标记planner-seed blocker，未创建或执行正式run。

## 2026-08-20 — planner_seed补丁通过，完整系统资格V1因运行环境遗漏正式FAIL

- 用户批准最小baseline可复现性补丁。冻结TARE源码中的两个`std::random_device`入口与PBS/CBS两个`rand()`入口已统一绑定显式`~planner_seed`，候选、分布、代价、阈值和规划逻辑不变；派生镜像内旧随机入口为0。
- 同seed直接RNG probe字节一致、异seed不同；ROS缺失私有seed时节点以fatal拒绝，提供seed时保持运行。派生规划二进制SHA-256为`82c25f...3ad`，相关回归33/33 PASS。
- 基础镜像遗留的`/etc/localtime`文件型VOLUME会让Docker 28拒绝创建容器；已生成只删除该Config.VOLUMES元数据的runnable clone，文件系统/规划二进制不变。新镜像可启动，ID=`sha256:5cbede...748c`。
- 完整资格Data Card/spec固定`tunnel`、原M-TARE、seed `11/11/23`、每次60仿真秒、每次100组scan/pose与20个waypoint；preflight 0 error。唯一run `gate5_20260820_mtare_planner_seed_qualification_v1_seed11`按用户批准执行。
- V1在5.44s后正式FAIL：`seed11_a`的roslaunch找不到`tare_planner_node`。证据显示二进制存在且可执行，根因是trial runner只追加路径却没有`source tare_system/devel/setup.bash`，所以catkin的libexec索引未注册。planner未执行，研究数据、模型、训练、C09/C10读取均为0；RUN_STATE与seal保留不可修改。
- 当前唯一NEXT需要用户明确批准V2 runner-environment-only corrective：只增加source TARE devel环境，其他world/seeds/runtime/counts/判据/无重试合同全部不变。不得在V1目录重跑，也不得进入50-case性能比较。

## 2026-08-20 — planner_seed完整系统资格V2参数映射错误正式FAIL

- 用户批准V2后，修复使用`source tare_system/devel/setup.bash --extend`，环境门禁确认AEE的`vehicle_simulator`和TARE的`tare_planner_node`同时可见；9项闭环回归PASS，preflight 0 error。
- 唯一V2 run成功启动Gazebo、AEE与patched TARE binary，但planner随即报告`kTestID wrong size: 2`并中止。源码只读证据确认`kTestID`是通信模式字符串：首字符`0`代表full comms；它不是随机seed。
- runner错误地将trial seed 11映射到`test_id:=11`。正确单机器人合同应固定`test_id:='0'`，同时只让`gazebo_seed`与`planner_seed`取11/11/23。问题分类为system/runner parameter mapping，不是否定environment fix、seed patch、模型或拓扑方法。
- V2在6.63s后封存为FAIL，不原地改参数或重跑；其summary继承V1状态标签也是证据命名缺陷，真实run ID和RUN_STATE仍明确指向V2。
- 当前唯一NEXT需用户批准V3 parameter-mapping-only corrective：固定`test_id='0'`并使用V3自有状态标签，其余tunnel、三trial、60秒、300帧、60waypoints及无重试判据全部不变。正式PASS前禁止50-case性能比较。

## 2026-08-20 — V3参数映射纠正已实现并完成proposal

- V3仅在planner启动命令中将通信模式`test_id`固定为`0`；`planner_seed`保持11/11/23且有唯一性断言，非planner命令不重写。继续使用V2已验证的双catkin workspace `--extend`环境。
- V3使用独立runner与`PASS/FAIL_MTARE_PLANNER_SEED_QUALIFICATION_V3`状态，不继承V1标签。环境、参数映射、歧义拒绝及既有闭环合同合计16/16测试PASS。
- V3 Data Card/spec proposal已固定tunnel三trial、180仿真秒、300帧、60 waypoints、2GB/0.5h上限。内存模拟批准后的preflight为0 error，仅有infrastructure不强制Data Card的预期提示；未创建run、未执行仿真。
- 当前唯一NEXT是用户明确批准V3正式执行；批准前proposal保持PENDING。

## 2026-08-20 — V3完整kTestID格式遗漏正式FAIL，V4实现完成

- V3真实启动确认`kTestID=0`与`planner_seed=11`均正确到达patched planner；随后原源码在`GridWorld`构造中无条件执行`kTestID.substr(2,2)`并因长度1抛出`out_of_range`。V3在6.51s封存为正式FAIL。
- 全量源码审计确认`kTestID`只在GridWorld与multi-robot manager读取：首字符`0`为full comms，第3–4字符编码robot count；默认合法单机器人编码为`0001`。长度4不会进入额外通信距离解析。
- V4参数映射现固定完整`test_id='0001'`，Gazebo/planner seeds仍为11/11/23；非planner命令不改，seed字段歧义拒绝。V4状态标签独立，相关测试19/19 PASS。
- V1/V2/V3均保持不可变FAIL。当前唯一NEXT是冻结V4 Data Card/spec并在用户新批准后执行；不得将此前短bag用于结果分析。

## 2026-08-20 — V4完成真实三trial，但严格同seed回放不成立

- 用户批准V4后，preflight 0 error并创建唯一run。`test_id=0001`、双workspace、Gazebo/planner seed及所有ROS组件均通过；seed11_a/seed11_b/seed23三条60秒回放全部完成，分别产生295/295/293组同步帧，三条required-topic audit全部PASS。
- seed23与seed11不同，variation control PASS；但两个seed11的100行frame prefix完全相同为0/100，scan hash相同0/100、pose hash相同1/100，20个waypoint仅第一个相同。正式run按预注册exact identity门禁标记`FAIL_MTARE_PLANNER_SEED_QUALIFICATION_V4`，205.64s、约1.3GB bags并封印。
- 只读数值诊断显示第一帧pose完全相同，但点云平均点位差8.96mm，0.10m voxel Jaccard为0.893；随后100帧平均pose分离0.151m、最大0.241m，0.10m Jaccard均值0.516，waypoint最大分离1.600m。差异不只是header timestamp或字节排列。
- 科研结论：已锁定Gazebo与planner显式随机源后，Gazebo/LiDAR/ROS并行执行仍非bitwise deterministic，小扫描差异会被M-TARE放大为路线差异。原50-case“精确配对”矩阵不得原样执行。
- 当前需用户选择：A推荐，改为预注册的多重复随机统计设计，并新增规划稳定性指标；B继续工程化Gazebo/LiDAR lockstep确定性，成本高且不直接增加方法科研价值。C10继续封存。

- 资源设计已量化：V4 60秒bag为456.16MB；无损zstd-10压缩到120.02MB（26.3%，3.57s/98MB RAM）。新矩阵固定为2 worlds×5 env seeds，原M-TARE每seed 3 execution repeats、M1D三个checkpoint各1次、Oracle 3 repeats，共90个600秒case；预计54000仿真秒、17--20墙钟小时、完整无损bag约103--120GB，且已获用户批准。

## 2026-08-20 — 用户批准随机区组设计，Gate 5以MIXED收口并进入Gate 6

- 用户在精确看到方案A/B、90-case规模、17--20小时和103--120GB成本后回复“批准”，明确选择A；这构成从Gate 5进入Gate 6的用户授权，不是自动推进。
- Gate 5正式结论为`GATE_MIXED`：C08 shadow、ROS交接、Oracle、coverage evaluator、M1D ROS导出和planner-seed组件可继续使用；保留风险是完整仿真无法逐帧bitwise复现，旧50-case精确配对设计永久停用。
- Gate 6固定为2 worlds×5 environment seeds×9 cases/block=`90` cases，每case 600秒，总仿真54,000秒。三类family各30次；10个world×seed block是统计配对单位，repeat/checkpoint seed不冒充独立world。
- 方法为冻结M1D因果拓扑全局规划；baseline为原M-TARE；完整开发地图Oracle仅为不可部署上界。主指标是0.5m体素coverage-time AUC，同时报告final coverage、路程、冗余、规划耗时及终点/航点离散度。
- 当前唯一NEXT：完成Gate-6 runner合同测试，冻结Data Card/spec与工具哈希，更新机器状态并运行preflight。通过后按本次一次性批准只创建并执行一个不可覆盖run；失败不重排、不删除、不重试，C09/C10和训练保持0。

## 2026-08-20 — Gate 6随机闭环正式预检PASS

- Gate-6 Data Card、90-case matrix、run spec、三checkpoint、派生ROS镜像、planner binary、world/map和全部执行/统计工具已冻结。schedule SHA-256=`6aea0921...27f1a`。
- runner、matrix、bag summarizer、Python3.8兼容与统计合同相关测试`25/25 PASS`；真实派生镜像中的Python3.8四个入口编译PASS，frozen-input只读检查确认90 cases/10 blocks/54,000秒及约627GiB可用磁盘。
- `preflight.py`为0 error、1个预期warning：`closed_loop_single`不强制Data Card，但本run仍主动绑定并snapshot专用Gate-6卡。唯一run ID固定为`gate6_20260820_mtare_single_robot_stochastic_v1_seed20260820`。
- 用户对精确A方案的一次性批准已经存在。当前唯一NEXT是调用`create_run.py`一次并启动冻结命令；运行中只监控，不改方法/顺序/阈值，不重试失败case。

## 2026-08-20 — Gate 6唯一正式run已创建，准备执行

- `create_run.py`只调用一次，创建`results/gate6_single_robot/gate6_20260820_mtare_single_robot_stochastic_v1_seed20260820`；创建后状态为`CREATED_NOT_EXECUTED`，Data Card、spec、matrix snapshot、环境和命令均已保存。
- 冻结命令使用`systemd-inhibit`阻止sleep/shutdown并设30小时外层硬上限。项目机器状态已绑定该唯一running experiment；下一动作只允许启动和监控，不允许创建第二个run。

## 2026-08-20 — Gate 6 stochastic V1首case收尾FAIL，并暴露真实域启动阻塞

- 唯一V1 run按批准启动，第一个随机case为`tunnel/env37/M1D-seed2`。Gazebo、LiDAR、localPlanner/pathFollower、terrain和M1D节点完成600仿真秒、3000同步帧、3001因果周期，topic audit PASS且planner failed cycle=0。
- 正式直接失败发生在指标生成之后：派生ROS镜像没有`zstd` CLI，case runner在压缩1,623,822,719-byte raw bag时抛`FileNotFoundError`。run按no-retry合同停止为`FAIL_MTARE_SINGLE_ROBOT_STOCHASTIC_V1`，formal completed cases=0、24/24 seal复核通过、约1.6GB，C09/C10/training/optimizer均为0。
- 只读科学复核同时发现不能只修压缩：机器人600秒路程=0，图=1 node/0 edge，3001/3001周期均为`no_observed_exit_stub_remains`。首帧range adapter接收5264点、off-ring=0，但M1D方向最大概率仅0.26165，冻结0.5阈值下0方向；均匀101帧全部0方向。
- 同一101帧上既有冻结B0均检测到1个出口，首帧为359°，说明可行的current-scan几何启动方向存在；问题是C08/C09训练域到AEE真实baseline world的模型域/接口缺口，而非LiDAR栅格为空或ROS循环失败。
- 当前Affected work停止。压缩-only replacement无科学意义。需要用户在推荐的“host-side lossless archive + 明确记录的B0空方向bootstrap/fallback并先做短资格”与“新增AEE域teacher/训练后重新冻结模型”或“停止M1D、改做B0拓扑”之间作明确决定。

## 2026-08-20 — 移动域复核否定B0长期fallback作为论文主方法，AEE head adaptation方案形成

- 对已封存V4原M-TARE移动bag均匀抽取99帧并送入三个冻结checkpoint：seed0/1/2空方向为`99/99`、`97/99`、`99/99`；最大方向概率均值为`0.3020/0.4073/0.2556`。B0为99/99非空，count分布为1/2/3=`18/74/7`。
- 因此B0不会只是偶尔bootstrap，而会接管几乎全部方向语义；若直接采用，不能支撑“M1D结构语义驱动规划”的核心论文主张。推荐路线从hybrid-only调整为客观AEE开发域head adaptation，B0只保留低频安全保护。
- 完整tunnel development map teacher在同一移动bag抽查20帧上20/20非空，方向数1/2/3=`2/17/1`且teacher count与decoded directions逐帧一致，证明无需人工标签或未来轨迹即可生成客观适配标签。
- 已形成`docs/GATE6_AEE_DOMAIN_ADAPTATION_CORRECTIVE_PROPOSAL_V1.md`和可机器校验的pending Data Card。计划为tunnel train/garage validation、10 trajectories、30000 raw/6000 effective frames，冻结encoder与embedding、只适配三语义heads，随后6-case readiness再恢复90-case。
- 现有治理只允许training位于Gate2；不能把纠正训练伪装成Gate6 infrastructure。当前需要用户明确授权临时重新打开Gate2 corrective representation work及对应数据/teacher/training范围。

## 2026-08-20 — 用户批准AEE域适配并临时重新打开Gate 2

- 用户明确回复“批准”，授权精确的AEE数据导出、完整地图teacher、三个seed head-only adaptation和operational Gate 2纠正；不是对C09/C10、readiness、90-case或多机器人正式执行的预先授权。
- Data Card已从proposal转为approved：tunnel train、garage validation、5 seeds各一条、30000 raw/6000 effective；完整地图只作客观teacher，encoder/embedding冻结，三语义heads适配。
- 当前唯一NEXT：实现并测试collector/teacher/exporter与head-adaptation executor，冻结数据/teacher run spec并preflight。数据导出和teacher属于一个不可覆盖material run；训练另立run spec并在数据PASS后执行。

## 2026-08-20 — AEE sensor export预检通过并创建唯一run

- 修正跨轨迹frame ID为`trajectory_id:stamp`，并明确每5帧固定抽样不使用0.2m位移筛选；数据规模与批准范围不变。
- PLAN第18节允许Gate 2纠正性数据/teacher，而旧治理仍限Gate 1；最小同步为Gate 1--2，Data Card仍须精确绑定operation与Gate，Gate 3+拒绝。治理25项、AEE 10项PASS。
- `configs/v3/gate2/aee_domain_sensor_export_v1.json` preflight为0 error/0 warning；成本约3小时/20GB，唯一run已创建尚待命令启动。
- 当前NEXT：执行10条轨迹；任何topic/type/count/range/off-ring/50m移动/压缩哈希/证据失败立即封存FAIL，不重试。

## 2026-08-20 — AEE sensor export V1入口环境FAIL

- 正式命令在runner导入阶段报`ModuleNotFoundError: numpy`；根因是冻结命令使用无NumPy的`/usr/bin/python3`，不是采集、数据、teacher或模型问题。
- V1为0 trajectories/0 frames，已写入原始stderr、failure metrics和FAILED RUN_STATE，10/10 seal复核PASS；禁止在原run重试。
- 已只读确认Anaconda Python 3.13.5 + NumPy 2.1.3可加载runner及完整合同。当前唯一NEXT是等待用户批准只替换外层Python路径的一次replacement run；科学合同保持不变。

### V1R replacement proposal完成

- 新wrapper冻结解析后Python路径/二进制SHA、NumPy版本/init/core SHA，只设置新V1R RUN_ID并单次调用原V1 main；环境失败不会进入核心runner。
- Luna独立审查确认所有科研与采集合约不变；补充parent V1 spec hash、不覆盖声明、forward-once和pre-entry fail测试。
- 41/41相关测试PASS；proposal JSON有效，preflight仅报告`user_authorization.status`尚非APPROVED。0新run、0采集、0teacher、0training。

## 2026-08-20 — V1R获批、预检通过并创建唯一run

- 用户批准Python-only replacement及sealed Cano C09 validation窄例外；PLAN/Data Card已同步，C09只读用于retention/tie-break，C10与后续strict-test仍为0。
- V1R冻结Python 3.13.5、NumPy 2.1.3及Python/NumPy二进制哈希；65/65相关测试PASS，preflight 0 error/0 warning。
- 唯一`gate2_20260820_aee_domain_sensor_export_v1r_seed20260820`已创建。当前只允许执行/监控该run；V1保持sealed FAIL且不复用。

## 2026-08-20 — AEE sensor export V1R归档权限FAIL，V1R2获批

- V1R第一条轨迹的科研合同全部通过：3000 raw、600 fixed every-fifth effective frames、1141.027337 m移动、传感器shard审计PASS；但容器生成的case目录归root所有，宿主`zstd`不能创建归档，故formal completed trajectories=0并按无重试合同封存FAIL，18/18 seal复核通过。
- V1R2仅增加成功采集后的容器到宿主ownership handoff，并要求宿主归档不存在、压缩命令无force覆盖参数、解压SHA-256一致后才删除raw bag；世界、seed、轨迹、帧数、抽样和阈值均不变。
- 37/37相关测试通过；真实Docker到宿主合成演练确认1000:1000、目录775/文件664、宿主压缩/解压哈希/删除全部PASS。用户已回复“批准”。
- 当前唯一NEXT：V1R2 preflight通过后只创建并执行一个不可覆盖run；任何采集、handoff或archive失败立即封存且不重试。

## 2026-08-20 — V1R2因授权时间证据错误主动停止

- V1R2 preflight曾为0 error/0 warning并启动，但冻结spec把`approved_at`误写为21:30，晚于约21:07的真实启动时间。该未来时间戳会使正式授权证据自相矛盾。
- 发现后立即中断runner并停止Docker case；第一条轨迹尚未完成，formal completed=0、formal frames=0，约1.7 GiB partial bag明确不可复用。run写为FAILED并17/17 seal复核通过。
- 科研合同、权限handoff实现及其37项测试/真实smoke结果不受影响。当前NEXT是准备只纠正授权元数据与新run ID的V1R3 proposal；必须新preflight和用户明确replacement批准后才能创建。

### V1R3 proposal已到批准边界

- V1R3薄wrapper只设置新run ID并复用V1R2 hooks；新增启动前授权时间门禁，拒绝非APPROVED、非法ISO-8601、无时区和晚于启动的时间。
- 16/16 replacement/permission/authorization测试PASS。proposal preflight仅有3项预期授权错误（status、approved_by、approved_at），其余0 warning；0新run、0新采集。
- 需用户明确回复批准V1R3；收到后才记录当时实际时间、冻结final spec并再次preflight。

### AEE teacher与三seed适配执行链已补齐

- teacher executor会逐条验证6000个全局唯一frame ID、完整地图hash、方向连通分量/出口数/结构角色一致性和train/validation三角色覆盖，并生成可视化与seal。
- 新增三seed head-adaptation outer runner：核对sensor/teacher/Cano seal、三个原M1D checkpoint、Torch2.9/CUDA12.9环境；固定每epoch Cano/AEE各3000，顺序运行seed0/1/2并汇总八项资格门禁。
- encoder与embedding仍按字节hash和真实AEE probe双重冻结；C09只读12500帧用于retention，C10/later sealed worlds为0。相关44/44 CPU合同测试PASS；尚未生成teacher或执行任何optimizer step。

### Adapted模型到ROS闭环readiness链已实现

- checkpoint exporter现在显式区分并保留`M1D`与`M1D_AEE_HEAD_ADAPTED_V1`模式，复制tensor后做same-runtime bit-exact及Torch2.9↔ROS Torch2.0输出/离散决策一致性；在线ROS节点只接受这两个白名单模式。
- 新增显式empty-direction fallback：learned方向非空时绝不调用B0；为空时B0只读当前range image，方向/count/role替代被逐帧记录，learned `z_role`保持不变。snapshot报告总fallback及首20帧后fallback率。
- single-robot case支持成功后的1000:1000 ownership handoff；宿主finalizer拒绝覆盖，zstd-10解压SHA一致后才删除raw bag，修复Gate6 V1的容器缺zstd阻塞。
- 6-case readiness runner按spec接收2 worlds×3 adapted checkpoints，每case固定180秒，硬检移动≥5m、2 nodes、1 verified edge、1 non-hold target、0 failed cycle及post20 fallback≤5%。当前不替用户选择未规定的environment seed。
- 整条AEE/部署/readiness相关67/67测试PASS，ROS Python3.8只读编译PASS；0正式teacher、0 optimizer、0 readiness case、0 C10。

## 2026-08-20 — AEE sensor export V1R3正式PASS

- 用户明确批准metadata-only V1R3 replacement；真实批准后主机时间`2026-08-20T21:45:37+08:00`写入spec，preflight为0 error/0 warning，只创建并执行一个不可覆盖run。
- 正式run `results/gate2_representation/gate2_20260820_aee_domain_sensor_export_v1r3_seed20260820`完成10/10轨迹：30000 raw同步帧、固定每5帧取1帧形成6000 effective，tunnel train/garage validation各3000；累计移动`11427.838595 m`。
- 十条轨迹的topic/type、finite range、ring、shape、唯一frame ID、固定raw indices、单条移动>=50m、1000:1000 ownership handoff及zstd-10解压SHA回验全部PASS。归档`18474285979 bytes`，sensor shards `124735822 bytes`，最终目录约18 GiB。
- RUN_STATE=`COMPLETED`、overall status=`PASS_AEE_DOMAIN_SENSOR_EXPORT_V1`；teacher/training/C09/C10读取均为0。112/112 seal条目独立`sha256sum -c`通过；seal SHA-256=`95b5d7c29183040bffbbad0490e0182ac6e8050ae272c28fa5083a02d435485f`，manifest SHA-256=`7de0091cb13fc88166233f3f9d2a99b13e395df27657b276880001adc3f38a04`。
- 下游teacher、三seed head adaptation、adapted ROS export、fallback/readiness、90-case V2及共享因果图/多机器人Hungarian协调器已有86/86相关CPU测试PASS；多机器人三个纯Python模块通过ROS Python3.8编译，但没有把单元测试冒充正式闭环结果。
- 当前唯一NEXT：以V1R3 seal/manifest为只读输入冻结并预检已批准Data Card范围内的objective teacher run；不得训练、运行readiness或读取C09/C10，直到teacher正式PASS。

## 2026-08-21 — AEE objective teacher V1因garage当前pose被膨胀障碍覆盖而FAIL

- 绑定V1R3 seal/manifest、两张完整地图和冻结工具的teacher spec通过preflight（0 error/0 warning）及14项相关测试，只创建并执行一个正式run。
- V1在完成6/10 shards、3600/6000 labels后，于`06_garage_seed23`有效样本360/raw frame1800停止：完整地图存在正确楼层支撑，但冻结0.55m障碍膨胀使当前真实走过pose附近0.8m内无可通行seed。run按合同封存`FAIL_AEE_OBJECTIVE_TEACHER_EXPORT_V1`，training/model inference/C09/C10均为0；30/30 seal复核PASS，seal SHA-256=`6df00c5220fda7d33afc705db47d5fe9a6912097fa80cc3763aadb37971d9651`。
- 只读诊断定位失败frame ID=`06_garage_seed23:362010000000`、xyz=`[50.2167,61.9072,5.4474]m`；expected/selected support=`4.6974/4.7492m`，最近可通行点为`1.0198m`，说明不是错层或空地图。
- 对其余未处理garage shards做只读完整复核：seed23/37/53/71失败数=`4/0/21/14`，合计39/2400；连同已PASS seed11，garage validation总缺陷39/3000。全部为中心落入inflated obstacle，最近可通行距离`0.8246--1.4000m`，呈坡道/楼层重复结构，不是随机单帧。
- 影响：objective teacher完整性门禁失败，后续head adaptation、ROS export和readiness全部保持停止；V1R3传感数据PASS不受影响。禁止删除39帧、复用部分teacher资产、把snap半径按validation最大值调到1.4m或在原run重试。
- 推荐下一步需用户决定：先做只读physics-based teacher corrective audit，核对Gazebo真实vehicle/collision footprint与preview pointcloud在39个pose的差异，并只从既有物理合同定义current-pose feasibility correction；预计30--60分钟CPU、0训练/模型/C09/C10。通过后仍需新teacher replacement spec/preflight/批准。
- 独立只读参数核对：local planner明确vehicle length/width=`0.6/0.6m`，矩形外接半径约`0.424m`；teacher的`0.55m`来自项目配置`base_collision_radius=0.4 + safety_margin=0.15`，没有可读AEE URDF collision geometry证明它等于真实Gazebo footprint。preview pointcloud与simulation collision资产来源/用途分离，但尚无逐点距离审计；因此不能直接宣称0.55正确，也不能静默替换为0.424。

## 2026-08-21 — 用户批准的AEE teacher物理一致性只读审计完成

- 冻结镜像实际包含`vehicle_simulator/urdf/robot.urdf.xacro`（SHA-256=`013a0dbd...ee3aa`）。URDF只有visual、没有任何`<collision>`；visual base为`0.6×0.4×0.3m`，轮组外廓约`0.7×0.6m`。AEE local planner单独使用`0.6×0.6m`，所以旧记录中“URDF不可读”被本条证据取代，但`0.55m`仍不是Gazebo robot collision自动推导值。
- Gazebo garage world的collision与visual都精确引用`garage.dae`（SHA-256=`06c66867...ab2b4e`）；teacher V1则读取并列的`preview/pointcloud.ply`（SHA-256=`2d5de09c...957d18`）。环境包没有PLY生成脚本或DAE↔PLY一致性元数据，不能把二者当作已证明等价的资产。
- 对garage全部3000帧重算仍精确得到`0/4/0/21/14=39`个失败，分成多个连续坡道段。35/39中心格已经被判raw obstacle，另4/39仅被膨胀覆盖；AEE `0.6×0.6m`当前车体方框内均有3--9个此类raw obstacle格，说明根因早于0.55m inflation。
- 这些“障碍”的高度只比旧teacher所选support高`0.150766--0.258327m`（中位`0.198028m`）。精确DAE中心竖线审计显示每个失败pose附近都有成对的楼板面：最接近冻结expected support的面误差仅`-0.064996..+0.025031m`且法向朝下；其上`0.151726--0.195767m`还有法向朝上的真实楼板顶面。旧XYZ-only PLY oracle无法识别面方向，选到楼板下表面后把顶面当成障碍。
- 同帧LiDAR近水平最短有效距离为`1.508011--2.020856m`，与“中心存在0.15--0.26m障碍”的teacher判断矛盾。这是teacher support-layer/modeling缺陷，不是实际贴墙，也不是sensor、模型或metric失败。
- 审计结论：禁止缩小0.55m、增大0.8m snap或删帧。推荐V1R只把support来源改为Gazebo实际使用的DAE有向三角面：选择当前层法向朝上的可行支撑面，再沿既有0.5m邻接高度合同构造层；side obstacle、0.55m inflation、720方向、6000帧身份和全部完整性门禁保持不变。该方法变更需新Data Card amendment、实现/合成测试、只读6000帧proof、final spec/preflight和一次明确执行批准；当前仍为0训练、0模型、0 C09/C10。

## 2026-08-21 — DAE upward-support实现通过，proof暴露model.sdf外部pose缺失

- 用户批准仅跳过25个精确零面积三角形；实现不设置非零面积阈值，garage/tunnel分别记录23/2个source face。6个约`1--2e-11 m²`非零sliver保留并由固定0.2m栅格自然判定。
- DAE loader解析active visual scene及完整node transform；upward support与PLY obstacle分离。大坐标重心插值改为等价的face-local公式，并用`64×float64 epsilon`排除物理竖直面的数值z残差，不引入物理角度参数。冻结镜像内15/15相关测试PASS。
- garage历史首失败帧`06_garage_seed23` sample360已用新表示查询成功：support z=`4.875413m`、exit count=2。
- 6000帧只读proof在任何标签写入前，于tunnel首帧`[0,0,0.75]`停止。根因是DAE为model坐标，现有loader遗漏`tunnel/model.sdf`明确的model pose `(-22,92.5,0)`；garage为identity。
- 完整只读链审计确认两世界均无include override、link/collision pose、mesh scale、`relative_to`、frame、joint或nested model。应用唯一SDF变换后DAE与preview PLY bbox对齐。tunnel/garage model.sdf SHA-256分别为`29b3dd29...7fb3`与`c24b5627...3499e`。
- 当前NEXT：需用户明确批准把model.sdf纳入冻结输入并应用精确`model×link×collision×scale×DAE-node`变换；批准前不实现该输入合同、不恢复proof、不生成正式teacher或训练。

## 2026-08-21 — SDF坐标链通过，单层2.5D teacher在真实双高度坡道处停止

- 用户批准绑定并应用`model.sdf`精确变换。严格parser冻结world/include/model/link/collision/mesh URI与scale，拒绝`relative_to`、frame、joint、nested model和hash漂移；tunnel使用`(-22,92.5,0)`、garage为identity。相关测试从15项增至18/18 PASS。
- tunnel首帧由“无支撑”纠正为support=`-0.008070m`、2 exits；garage历史V1首失败帧纠正为support=`4.875413m`、2 exits。
- 6000帧只读proof无teacher shard写入，tunnel 3000/3000及garage前2285帧通过后，在`08_garage_seed53` sample485/raw2425停止。累计5285帧不得冒充完整PASS。
- 失败格点同时包含`10.427020m`水平面与`11.014537m`约3°坡面，二者为不同primitive/material、非重复面，高差`0.587517m`；expected support=`10.720778m`恰为中点，严格unique-nearest发生并列。歧义在车辆约3.4m外；车辆中心support唯一。
- 相邻轨迹从低水平层进入上升坡层；多层图诊断显示两候选都可从中心经既有`|dz|<=0.5m`邻接传播到达。问题是单层2.5D表示无法保存同XY双高度，不是DAE解析、坐标、零面积、无支撑或浮点epsilon问题。
- 剩余714帧只读tie扫描为0；当前已知全6000范围内仅此1帧触发严格并列，但禁止删帧、加epsilon或任意高/低tie-break。
- 推荐NEXT需用户批准：实现多层局部支撑图，逐层障碍/膨胀与layer-continuous 720方向传播；预计4--8小时实现测试，之后约12分钟重做完整proof。正式teacher/training仍未授权。

## 2026-08-21 — DAE多层局部支撑图完成6,000帧只读proof

- 方法：同一XY保留全部DAE upward support候选；从当前pose的唯一支撑节点开始，在8邻域内按既有`|dz|<=0.5m`传播；720条方向ray只沿连续层传播。PLY只提供支撑面上方的障碍证据，障碍膨胀按精确source-surface identity隔离。Gazebo world/model.sdf/mesh transform完整应用。
- 实现：新增精确世界坐标连通的DAE upward-sheet identity；重复vertex index但坐标完全相同的接缝合并，真实断开的叠层保持不同identity。没有增加距离epsilon或数据驱动阈值。23/23相关单元与合成测试PASS。
- surface审计：garage/tunnel分别有880/168个identity。把连通容差放宽到`1e-4m`仍基本不变，说明主要是实际断开的geometry islands；仅garage primitive1有少量约`1e-6m`缝，合并收益只有1个sheet，故不引入任意容差。
- proof结果：从第1帧重跑，10/10轨迹、6000/6000有效帧全部PASS，train tunnel=3000、validation garage=3000、unique frame IDs=6000，耗时2371.436秒。teacher shards=0、model inference=0、training=0、C09/C10 reads=0。三个历史失败点全部通过。
- 正式提案：新增V1R shard generator/runner、Data Card与run spec proposal。proposal preflight只有`user_authorization.status`和Data Card `approval.status`两项预期失败；无其他schema、Gate、路径或run目录错误。正式teacher生成成本预计40--60分钟CPU、<4GiB RAM、<1GB结果；仍需一次明确执行批准。

## 2026-08-21 — DAE多层objective teacher正式V1R PASS

- 用户在看到6000/6000只读proof、十轨迹数据、40--60分钟CPU成本、证据和失败政策后明确回复“批准”。批准时间、Data Card、spec、工具和镜像哈希冻结后，最终preflight为0 error/0 warning；只创建并执行一个不可覆盖run。
- runner在任何标签前核验V1R3的112项source seal、两世界全部PLY/DAE/world/model.sdf哈希和预期world-from-mesh矩阵。随后串行生成10/10 shards和6000/6000标签，无重试、删帧、调参或部分复用。
- 最终状态`PASS_AEE_DAE_MULTILAYER_OBJECTIVE_TEACHER_EXPORT_V1R`，耗时2385.476秒。split=`train 3000 / validation 3000`，unique frame IDs=6000；train exit 1--5=`195/2083/502/214/6`，validation exit 1--6=`1058/1014/323/454/114/37`。
- 两split均含interior/junction/terminal：train=`2083/722/195`，validation=`1014/928/1058`。training/model inference/C09/C10均为0。
- 独立复核46/46 evidence seal、10 shard哈希、shape/dtype、frame/raw-index、direction/count/role和DAE provenance全部PASS；4张coverage/role preview逐张非空且覆盖完整。该PASS只解除客观teacher阻塞，不代表模型已适配或闭环实验完成。
- NEXT：修订三seed head-only adaptation Data Card/spec，使其只读取本V1R teacher；展示训练成本与checkpoint/retention证据并取得独立批准，未批准前不训练或读取C09。
## 2026-08-21 — AEE head adaptation V1R在训练前因临时sidecar丢失FAIL

- 用户批准一次三seed head-only adaptation；新Data Card/spec绑定V1R3 sensor seal、DAE-multilayer teacher seal、Cano V2R seal、seed 0/1/2 checkpoints和固定训练门槛，preflight为0 error/0 warning。
- 唯一run `results/gate2_representation/gate2_20260821_aee_head_adaptation_v1r_seed20260820`在环境probe处停止：冻结可执行文件`/tmp/mtare_phase3_torch290_zarr2187/bin/python`不存在。运行15.927秒，completed seeds=0、optimizer steps=0、C09/C10/later reads=0，状态`FAIL_AEE_HEAD_ADAPTATION_V1R`，10个文件封存，seal SHA-256=`fd5f05f5002b6a767e96d7a0015547c218f80e2788001c2b4b05d508fd31d5db`。
- 分类为system/environment asset failure；不否定sensor、teacher、数据、模型或指标，也没有回答adaptation研究问题。V1R不可修改或重试。
- 只读核查发现现存`/tmp/mtare_gate4_torch290_cu129_zarr2187/bin/python`具备完全相同的Python 3.13.5、Torch 2.9.0+cu129、CUDA 12.9、Zarr 2.18.7、Numcodecs 0.15.1、NumPy 2.1.3、sklearn 1.6.1和RTX 5090 D身份；其排序pip freeze SHA-256仍为`fccb0fe6...23ae`。
- NEXT：只准备V1R2 path-only replacement并重新preflight；需要新的明确批准后才创建和执行，禁止复用V1R目录或部分证据。
- V1R2 proposal现已冻结：新runner把Python executable改为spec绑定，并在模型加载前同时核验精确版本和live sorted pip-freeze SHA-256。只读probe PASS；proposal preflight只有`user_authorization.status`与Data Card `approval.status`两项预期批准错误，0 warning。没有创建V1R2 run。
- 新增可执行路径传递、pip-freeze顺序无关/内容严格及pip-check fail-closed测试；head-adaptation runner、dataset与trainer共22/22项CPU测试PASS。为避免第二次`/tmp`丢失，51 MiB sidecar已非覆盖复制到用户持久目录；live版本、CUDA/GPU、pip-freeze hash与依赖完整性全部PASS，proposal工具/环境哈希随实现重新冻结。
## 2026-08-21 — AEE head adaptation V1R2因venv symlink被resolve而训练前FAIL

- 用户明确批准V1R2；22/22测试和最终preflight 0 error/0 warning后只创建并执行一个不可覆盖run。
- run在9.556秒后于环境probe停止：outer runner对持久venv的`bin/python`调用`Path.resolve()`，将其解引用为底层Anaconda Python，绕过venv本地`zarr`，报`ModuleNotFoundError: zarr`。
- V1R2为`FAIL_AEE_HEAD_ADAPTATION_V1R2`；completed seeds/epochs/optimizer steps/C09/C10/later reads均为0。10文件seal独立核验PASS，seal SHA-256=`65fbd9885b4b648692de8b6509fc0a99888d3d9b3a6fbff0b99f59bba60e10ec`，禁止原地重跑或部分复用。
- V1R3只把路径规范化改为“必须为绝对路径、不得解引用venv symlink”；合成symlink测试和真实持久sidecar全链probe均PASS，相关23/23测试PASS。proposal preflight只有两项预期approval错误、0 warning，尚未创建V1R3 run。
- 独立零数据pre-entry proof又按正式函数链验证：venv调用路径保留且与resolved base不同、完整环境identity PASS、live pip-freeze hash一致、pip check PASS、child trainer argv首项仍为venv路径；dataset/checkpoint/C09/C10/optimizer/file write全部为0。该proof工具已冻结进V1R3 spec。
## 2026-08-21 — Head-only V1R3正式训练科学FAIL，冻结encoder路线被否定

- V1R3在23/23测试、零数据preentry proof及preflight 0/0后唯一执行。seed0完成10 epochs、470 optimizer steps；因资格失败返回2，seed1/2未启动。run状态`FAIL_AEE_HEAD_ADAPTATION_V1R3`，C10/later reads=0。
- AEE结果：direction F1=`0.029503` vs B0=`0.301158`，empty=`0.886`，count macro-F1=`0.084205`，role macro-F1=`0.157502`；count全部预测2 exits，role全部预测junction。Cano direction F1从`0.889860`降至`0.807468`，超过0.02保留界。
- 保持项：encoder/embedding tensor identity=true，fixed z_role probe差异=0，三个head确实改变。说明执行合同生效，但“冻结表示只改head”本身不成立。
- 输入合同诊断：AEE train/validation valid density=`0.461706/0.482740`，Cano train=`0.997619`；source encoder对AEE的z_role centroid norm=`0.99994`且mean feature std=`0.000826`，对Cano分别`0.85662/0.04485`，表示从首个conv起即近常量化。
- 标签合同诊断：Cano使用出口中心`3°` Gaussian，方向target均值`0.042693`；AEE V1R shard保存整片traversable sector，train/validation均值=`0.179390/0.620927`。二者不能直接进入同一BCE head。
- 已实现显式`cano_gaussian_component_centers`编码，原始mask仍保留用于exit component/几何证据；相关25/25测试PASS。6000/6000只读proof确认component count恒等，canonical target均值train/validation=`0.047020/0.046394`，与Cano标签尺度一致；0写文件、0训练、0 C09/C10。
- NEXT：设计完整encoder域适配和AEE-mask-matched Cano训练增强，并把count Gate恢复为PLAN预声明的常见1--4类；先新Data Card/spec/preflight，不得直接重跑V1R3。

## 2026-08-21 — AEE encoder-domain-adaptation V2 proposal ready

- DONE：保留6000条raw AEE teacher mask，并用相同component center与Cano冻结3° Gaussian生成学习target。
- DONE：实现全encoder/embedding/heads可训练合同和确定性AEE-mask-matched Cano views；每epoch独立样本3000 Cano + 3000 AEE，额外3000 Cano增强view不计独立样本，按0.5/0.5/1.0保持域权重平衡。
- DONE：count Gate恢复为PLAN规定的branch 1--4，5--6只作完整diagnostic；validation无增强，C10/later读取为0。
- DONE：39/39相关CPU测试与py_compile PASS；proposal preflight除8项待授权字段外0错误、0警告，0正式run、0训练。
- DONE：从sealed 6000帧重算轨迹口径，纠正旧估计为599.000--599.005s和2.022131m median effective-frame displacement；零数据pre-entry proof核验15项工具、精确GPU sidecar、pip与child entry，0数据/checkpoint/训练/C09/C10/写文件。
- NEXT：等待用户对展示的V2 Data Card/spec给出一次明确执行批准；随后finalize、preflight 0/0并执行唯一三seed run。

## 2026-08-21 — AEE encoder-domain-adaptation V2 formal scientific FAIL

- 用户批准、41/41测试和preflight 0/0后，唯一run完成seed0全部10 epochs/470 optimizer steps；seed1/2因seed0未达科研门禁未启动。
- AEE garage：direction F1 `0.017102`（B0 `0.301158`）、empty `0.972`、count 1--4 F1 `0.281081`、role F1 `0.362580`。Cano retention `0.889860 -> 0.735424`。finite/model-change合同PASS，效果与retention合同FAIL。
- RUN_STATE=`FAILED`，26/26 seal通过，seal SHA-256=`4d0ff2349b8f27b19547a3e8ebf6abec14bcb5b6693bd8872120886fc215a348`，约11MiB、168.711s、C10/later 0。
- 只读train/validation诊断显示V2在AEE tunnel达到direction F1 `0.317246`、role F1 `0.694783`并恢复z_role feature variation，但在garage几乎不泛化；一个AEE train world是当前数据多样性阻塞。
- NEXT：停止继续换loss/阈值或重跑V2。等待用户在推荐的同pose sensor-operator parity + paired front-end adapter、扩充AEE/Gazebo开发world、或B0降级路线之间作决定。

## 2026-08-21 — AEE same-pose sensor-operator audit completed

- DONE：V1 pre-execution runner错误已独立封存；V1R完成tunnel/garage各32帧、全部10条轨迹、737,280 rays、128 B0和384 M1D inference，14-file seal，0 training/C09/C10。
- RESULT：AEE真实scan在两世界均约半密度，确认共享sensor-grid差异；DAE理想scan让tunnel三seed恢复，却未让garage seed2恢复。
- CONCLUSION：paired front-end alone不足，`MORE_INDEPENDENT_AEE_GEOMETRY_REQUIRED`。传感器接口与geometry diversity两项都必须在后续方案中显式处理。
- NEXT：等待用户选择推荐的combined corrective、仅multi-world corrective或B0降级；选择前不训练、不读取C09/C10、不恢复Gate-6闭环。

## 2026-08-21 — 用户选择联合纠正方案A，topology-only资格到正式批准边界

- 原始AEE配置明确为VLP-16、350水平samples，目标模型栅格为16×720；`350/720=48.61%`解释两世界约半密度。64帧无训练短缝诊断把tunnel/garage valid mask与理想scan agreement提高到`0.99708/0.99975`，但garage seed2仍F1=0、empty=1，故正式路线必须同时处理接口和几何多样性。
- 正式接口不复用诊断性短缝补齐：新增组织化16×350源束重采样，只有左右物理束均有真实回波时才插值，真实无回波保持无效；源range/mask必须与16×720输出同时保存。接口与候选选择共17/17 CPU测试PASS。
- 权威Gate-0 V2复核取代误读的旧V1统计：V2 C11/C12为19/20有效，C12十类有效，但S07-C12和S09-C12已经进入旧split；剩余15个未接受候选只覆盖8类，不能组成完整独立验证。
- 新合同冻结C13--C24：10 strata×12=120候选，每类取前两个有效且identity唯一的世界，得到10 train+10 validation；任一类不足两个整批FAIL，不追加C25。当前只实现registry/selector/executor/runner，不生成候选。
- 历史`/tmp` E1已丢失。新持久E1按真正Gate-0 98行freeze重建，标准化freeze SHA双方均为`daa45423...e1f`、`pip check`PASS、upstream commit=`b6c776...`；S01-C01与S10-C10 canonical identity均与sealed V2逐字节相同。
- proposal、Data Card和run-spec proposal已准备。零数据preflight唯一4项错误均为待正式user authorization字段，另有1条“audit不强制Data Card但仍snapshot”预期warning；正式run目录不存在，C13/mesh/ray/teacher/training/C09/C10/M-TARE计数全0。
- NEXT：展示精确范围、约2小时CPU/2GB和证据后取得一次正式执行批准；批准后冻结最终hash、preflight 0 error并只创建一个不可覆盖topology-only run。

## 2026-08-21 — C13--C24 topology-only候选资格正式PASS

- 用户在看到精确的120候选、10类×12、每类固定选择前两个有效候选、10 train+10 validation、`<=2 CPU hours`、`<=2 GiB`以及零mesh/LiDAR/teacher/training/C09/C10范围后明确回复“批准”；授权只绑定这一个不可覆盖run。
- 授权时间、proposal、Data Card、run spec、120-slot registry、Cano E1、源提交、历史证据和全部工具hash冻结后，21/21相关测试PASS；最终preflight为0 error，唯一warning仅说明audit操作不强制Data Card但仍会封存快照。
- 唯一run `results/gate2_representation/gate2_20260821_aee_corrective_topology_candidate_audit_v1_seed20260821/`在1701.969秒内完成120/120候选；每个成功候选均同seed精确复现，源checkout运行前后不变。
- 十层有效数为`11/11/10/12/11/12/10/12/10/12`，全部不少于2；固定规则选出20个唯一父世界，其中10 train、10 validation，parent ID和坐标拓扑identity均20/20唯一且split-disjoint。S05因C13无效而固定选择C14/C15，其余层选择C13/C14。
- RUN_STATE=`COMPLETED`、状态=`PASS_AEE_CORRECTIVE_TOPOLOGY_CANDIDATE_AUDIT_V1`；120个逐候选metrics、20-parent/split manifest和完整图/样条证据已保存。376/376 seal从项目根目录复核PASS，seal SHA-256=`08bf60ef7c626e5ebbf9510f09f2afa951434b4485567f6b8aca1beb515dcc51`，目录约97MB。
- 范围审计仍为0 mesh、0 LiDAR、0 teacher、0 training、0 model、0 C09/C10和0 M-TARE变更。该PASS只解除多几何拓扑来源阻塞，不代表传感器数据、模型泛化、在线拓扑图或规划替换已经完成。
- NEXT：以冻结的10+10父世界为唯一来源，准备geometry/sensor materialization与采样Data Card/spec；先只做实现、成本和只读资格，不得把本次批准扩展为新数据生成或训练授权。

## 2026-08-21 — 下一阶段只读接口审计发现AEE 350束角度网格未完整定义

- 冻结AEE镜像确认原始topic为`/velodyne_points`，`organize_cloud=true`；插件输出PointCloud2 `width=16`、`height=350`，按azimuth-major/ring-minor排列，并为无回波保留NaN。因此重新采集原始束身份在系统上可行，旧`/registered_scan`不必被错误反推。
- 但xacro把水平角冻结为`min=-π,max=+π,samples=350`，插件用`i/(rangeCount-1)`计算角度。首尾两束方向重复，只有349个独特方位；现有`resample_organized_azimuth()`假设输入末维是从0开始的半开圆周均匀网格，不能直接消费插件原始顺序。
- 若直接使用，source index 0会被当成0°而实际为-180°，同时错误忽略重复端点；这会破坏方向teacher对齐。问题分类为sensor-interface contract，不影响已封存的topology-only PASS，也不读取C09/C10。
- 受影响工作已在新materialization executor写入前停止。推荐增加typed AEE Gazebo organized-scan canonicalizer：保留全部350 raw beam provenance，按精确物理角重排，并以预声明规则合并±π重复方向后再插值到720；Cano 16×350射线必须使用同一物理角合同。该方法选择需用户明确确认后实现和只读测试。

## 2026-08-21 — 用户批准方案A，exact-angle 350→720接口实现PASS

- 用户回复`a`批准推荐方案A。重复端点规则固定为项目已有first-return语义：两条均有效取较近回波，仅一条有效保留该回波，均无效保持无效；冻结AEE xacro不修改。
- 新增精确`[-π,+π]` 16×350 source ray生成、350 raw→349 unique canonicalization、物理角bracketing和720列插值；只有左右真实方向均有效时才生成中间列。
- 新增organized PointCloud2纯函数合同：严格`[350,16,3]`与ring `0..15`顺序、全NaN no-return、raw 0.1--130m provenance和model 0.3--50m view分离，输出统一`[16,350]`。
- 17/17接口测试与py_compile PASS；零mesh、零正式ray、零teacher、零training、零C09/C10。下一步恢复20-parent perception-mesh资格实现，并把mesh run与后续11,000-frame数据导出拆开，避免mesh失败污染正式数据。

## 2026-08-21 — 20-parent corrective perception-mesh executor已实现

- 新executor严格读取sealed selected-parent manifest，把20个C13--C15父拓扑映射到原生Cano重建；每个parent只生成一个primary mesh，geometry seed保持冻结。
- 复用已验证的M1R sanitation：只删除精确零面积面并逐资产执行axis/spline、连通分量、流形风险和source identity审计；禁止remeshing替代。
- 输出合同固定20 meshes、20 sanitation、20 metrics和10张corrective-train完整XY/XZ图；validation 10 worlds不渲染，只自动检查。LiDAR/teacher/formal sample/training/model/C09/C10/M-TARE均为0。
- exact-angle接口与mesh zero-material合同合计19/19测试PASS，E1 py_compile PASS。尚未创建proposal/spec或正式run；下一步补齐runner、Data Card、成本和preflight。

## 2026-08-21 — 20-parent perception-mesh V1在首个资产前因子进程导入路径FAIL

- 用户在看到精确20-parent、10 train+10 validation、20 mesh、10张train-only预览、`<=2 CPU hours`、`<=1 GiB`和零LiDAR/teacher/training/C09/C10范围后明确回复“批准”；31/31测试及最终preflight 0 error后只创建并执行一个不可覆盖run。
- V1运行1.263秒即由source precheck停止，原因是runner启动executor时未传递冻结checkout `PYTHONPATH`；实际`subt_proc_gen`来自持久E1的`site-packages`，而旧合同要求来自`external/procedural-subt-gen/src`。
- RUN_STATE=`FAILED`，状态=`FAIL_AEE_CORRECTIVE_PERCEPTION_MESH_V1`；mesh/sanitation/metric/preview均为0，LiDAR/teacher/training/model/C09/C10/M-TARE也为0。14/14 evidence seal复核PASS，seal SHA-256=`5f8d79a31ccc6b8806be0f3012269574379402723ba0eb99a9b66a79833d307c`。
- 分类为system/runner environment propagation failure，不否定sealed 20-parent topology、mesh方法或sensor合同。V1不可修改或重跑。
- NEXT：推荐V1R仅给executor传入旧runner已验证的冻结环境路径；不重装E1、不改数据/mesh算法/seed/阈值/范围。需新的明确选择与执行批准后才能建立V1R。

## 2026-08-21 — 用户选择V1R方案A，已准备到正式执行批准边界

- 用户明确回复`A`，选择只把旧runner已验证的冻结`_environment()`传给executor子进程；不重装E1，不改20-parent范围、mesh算法、seed、阈值或资源上限。
- 新V1R runner、proposal、Data Card和run-spec proposal已实现；32/32相关测试与E1 py_compile PASS。
- 与正式子进程相同的零资产proof确认`subt_proc_gen`来自固定checkout，commit=`b6c776...`、entrypoint hash、remote和tracked-clean全部PASS；9项工具/配置hash、376/376 topology seal和live/historical环境freeze均一致。
- proposal preflight只有4项待正式user authorization错误，另有1条infrastructure不强制Data Card但仍snapshot的提示；V1R目录不存在，mesh/ray/teacher/training/C09/C10均为0。
- NEXT：展示精确20-parent、`<=2 CPU hours`、`<=1 GiB`和证据后取得一次V1R正式执行批准；批准前不得create_run。

## 2026-08-21 — 20-parent perception-mesh V1R因RAM合同超限人工停止并封存FAIL

- 用户在精确V1R范围和最终preflight后明确批准；32/32测试及preflight 0 error后只创建并执行一个不可覆盖run。冻结checkout路径修正生效，V1的启动错误未复现。
- 运行到约13分钟时，executor RSS观测为`1,243,852 KiB`（约1.19 GiB），超过批准时声明的1 GiB RAM上限。按resource stop rule立即SIGINT，不允许为追求20/20而继续。
- 最终RUN_STATE=`FAILED`、状态=`FAIL_AEE_CORRECTIVE_PERCEPTION_MESH_V1R`、executor exit=`-2`、duration=`791.408s`。已写14 mesh、14 sanitation、14 metrics和7 train previews；14项局部identity/mesh/sanitation全部PASS，但不得拼接或升级为正式20-parent结论。
- 结果约208MB，146/146 evidence seal复核PASS，seal SHA-256=`10286494fc973530811f730fbe6914dac1dea940c2a214603b93f2830ebda608`；LiDAR/teacher/training/model/C09/C10/M-TARE均为0。
- 分类为resource-contract/system monitoring问题，不否定mesh质量。推荐V1R2把RAM硬上限改为2 GiB并由runner实时测量/记录peak RSS，从头生成20个世界；备选是每parent隔离子进程以尝试保持1 GiB，但改动更大且单世界峰值仍需证明。需用户明确选择。

## 2026-08-22 — 用户选择2 GiB硬监控方案A，V1R2准备到正式批准边界

- 用户回复`a`选择推荐方案：不改mesh/data合同，只把runner+executor递归进程树RSS硬上限冻结为2 GiB，每0.25秒由冻结psutil 7.2.2采样；超限SIGINT并FAIL。
- 新V1R2 runner保存完整RSS JSONL trace和resource summary，把RSS/time/disk共同纳入最终PASS；V1/V1R目录不读取、不恢复、不拼接。
- 合成PASS与1-byte限额FAIL测试均通过；全部相关测试33/33与E1 py_compile PASS。9项工具/配置hash、376/376 topology seal、live/historical freeze、pip check和upstream clean均PASS。
- proposal preflight只有4项待正式user authorization错误和1条infrastructure Data Card snapshot提示；V1R2目录不存在，mesh/ray/teacher/training/C09/C10均为0。
- NEXT：取得一次精确V1R2执行批准，批准后重冻hash、preflight 0 error并仅执行一个从头20-parent run。

## 2026-08-22 — V1R2自动RSS门禁在6/20后触发，正式FAIL

- 用户在33/33测试、全部来源/环境proof和精确2 GiB进程树RSS合同后批准唯一V1R2；最终preflight 0 error后从sealed topology从头执行，未读取V1/V1R资产。
- psutil每0.25秒采样runner及递归descendants；第1473个样本前峰值达到`2,203,566,080 bytes`=`2101.48 MiB`，超过`2,147,483,648 bytes`硬限，runner自动对executor进程组SIGINT，无forced kill。
- RUN_STATE=`FAILED`，状态=`FAIL_AEE_CORRECTIVE_PERCEPTION_MESH_V1R2`，duration=`373.957s`，完成6 mesh/6 sanitation/6 metrics/3 train previews；局部结果不得复用。结果约64MB，72/72 seal PASS，seal SHA-256=`6b6abd92fcc39bcba0f8f382399bbe728a5075c59200fbcd621a06ca5a6638b4`。
- trace按parent完成时间的累计峰值为`1045.9,1174.8,1270.3,1731.8,1731.8,1731.8 MiB`，随后第7个处理期间越过2 GiB；首个大C14单独阶段只有1174.8 MiB。这支持跨parent native/Open3D allocator内存保留，而非单parent天然超过2 GiB。
- 分类为system/batch-process memory accumulation，不否定6个局部mesh质量或20-parent方法。推荐V1R3每parent用全新E1子进程、每parent独立2 GiB RSS门禁，batch runner只汇总，从而由进程退出释放native内存；不调mesh参数、不复用旧资产。需用户明确批准方法后实现。

## 2026-08-22 — 用户选择V1R3逐parent隔离，准备到正式执行批准边界

- 用户回复`A`选择每parent exactly one fresh E1 child；每个worker只生成一个native mesh、sanitation和可选train preview，完成即退出释放native/Open3D内存。每parent的轻量runner+active worker进程树独立2 GiB/0.25s门禁。
- 新batch runner不导入Open3D；固定顺序启动20 worker，逐parent保存log、RSS trace/resource summary、receipt和metric，任一失败即停。最终仅20/20时生成PASS summary/manifest；V1/V1R/V1R2 read/reuse计数冻结为0。
- Open3D跨run OBJ hash不作为资格标准；每parent在本run只生成一次并立即冻结，禁止retry/reselection。六个历史共同parent只读比较确认topology/spline/geometry identity 6/6相等、两侧audit均PASS，但OBJ hash不同，支持该合同。
- 35/35相关测试和E1 py_compile PASS；batch不加载Open3D、错误parent identity生成前拒绝。11项工具/配置hash、376/376 source seal、environment/upstream proof均PASS。
- proposal preflight只有4项正式authorization待填和1条Data Card snapshot提示；V1R3目录不存在、0新mesh/ray/teacher/training/C09/C10。
- NEXT：取得一次精确V1R3执行批准，批准后最终重冻hash、preflight 0 error并执行唯一20-worker run。

## 2026-08-22 — V1R3隔离跨parent内存后暴露单parent二次内存峰值，正式FAIL

- 用户在精确20-parent、10+10 split、每parent 2 GiB、整批2小时/1 GiB和零下游范围展示后明确回复“批准”；授权时间`2026-08-22T11:52:16+08:00`。final preflight为0 error、1条Data Card snapshot提示，只创建并执行一个不可覆盖run。
- V1R3完成固定顺序前7个parent；第8个`S04_3d_unicyclic_small_C14`在7.121秒内达到`2,199,797,760 bytes`，超过`2,147,483,648 bytes`门禁，自动SIGINT，forced kill=false。整批耗时401.972秒，7 meshes/7 sanitation/7 metrics/4 train previews，114/114 seal，seal SHA-256=`96e101919fd5fa89e74228677aa9c2601894b0b7acd7b380e7b7576a362cf14b`。
- stack精确落在原Cano generator的`points_inside_of_tunnel_section -> distance_matrix(points,tunnel_points)`；实现一次性构造`A×B×3`浮点临时数组。trace在约2秒内从1.10--1.88 GiB波动升至2.049 GiB以上，因此这是单parent的system/algorithmic-memory问题，不是跨parent残留、数据漂移、mesh审计或模型问题。
- V1R3保持正式FAIL且7个局部资产不得复用。推荐V1R4把全量欧氏距离矩阵改为严格等价的固定分块计算，保持最近点、半径判断和几何输出语义不变，并先做逐元素等价与该parent只读资源proof；备选提高RSS上限但没有剩余19个parent上界，或停止corrective路线。需用户新选择，禁止自动重试或静默提高预算。

## 2026-08-22 — 用户选择V1R4精确分块方案，已到正式执行批准边界

- 用户回复`a`选择不提高RSS上限，改为同数学语义的query-row分块。新实现只处理并释放已完成的距离矩阵行，保留所有reference顺序、`np.argmin`首个等距点、原欧氏距离算术和严格`<`阈值；固定32 MiB只是临时内存目标，不参与几何判断。
- 同一mesh阶段的两个全矩阵热点`points_inside_of_tunnel_section`与`ids_points_inside_ptcl_sphere`均由worker生命周期内的scoped patch覆盖；退出后恢复原函数。拓扑生成阶段其它矩阵路径不在本次范围，且20个sealed topology来源已在原实现下全部通过。
- float32/float64、随机点、重复最近点、严格半径边界、带/不带axis vector、球面查询、不同chunk规模、补丁安装/恢复、错误parent fail-closed及runner轻量合同在内的47/47相关测试PASS，E1 compile/integration PASS。独立只读验收器以sealed V1R3 FAIL作负对照，正确拒绝run ID、7/20、超内存、旧receipt和预览不完整。
- V1R4 Data Card、proposal和run-spec proposal冻结相同20 parent、10+10 split、每parent 2 GiB、整批2小时/1 GiB、0 downstream和0 failed-mesh asset reuse。proposal preflight只剩4项正式authorization字段及1条Data Card snapshot提示；正式目录不存在。
- NEXT：取得一次精确V1R4正式执行批准；批准后更新真实时间和hash、final preflight 0 error并只创建/执行一个新run。

## 2026-08-22 — V1R4正式20-parent感知网格资格PASS

- 用户在V1R4精确20-parent、2 GiB、2小时/1 GiB、零下游和失败资产零复用范围展示后明确回复“批准”；授权时间`2026-08-22T12:52:03+08:00`。47/47测试与final preflight 0 error后只创建和执行一个不可覆盖run。
- 正式结果为`PASS_AEE_CORRECTIVE_PERCEPTION_MESH_V1R4`：20/20固定顺序worker、20 meshes、20 sanitation、20 metrics、20 receipts、10 train previews、0 validation previews；全部identity/mesh/sanitation/memory/source/exclusion/no-retry checks为true。
- 整批耗时`1273.5953265190183s`，结果封存前`390,694,331 bytes`；最大单parent进程树RSS`1,081,565,184 bytes`，低于2 GiB。V1R3失败parent `S04_3d_unicyclic_small_C14`在V1R4为21.110秒、约689.70 MiB并完整PASS。
- 287/287 evidence seal和独立验收均PASS，seal SHA-256=`2f843d221f9e6bc28b01d864fc9c2b4a4fe6e651030b0089a330258cad226159`。人工只看10张train完整XY/XZ预览，全部非空、无截断且mesh/axis对齐；validation预览保持0。
- 结论边界：解除20-parent感知mesh阻塞，不等于sensor/teacher、模型或拓扑回放PASS。NEXT只能冻结11,000帧sensor/objective-teacher Data Card、成本与spec；生成前需新批准，仍禁止C09/C10和训练。

## 2026-08-22：11,000帧 corrective sensor/teacher V1 系统性早停 FAIL

- 新增organized `16×350`保存、确定性`350→720`、100帧/轨迹固定索引、Cano结构事件/隧道覆盖采样、Cano双场景ray/teacher执行器、AEE collector/teacher和正式runner；相关测试29/29 PASS。
- 20个Cano世界的topology-only pose proof均得到100 clusters/500 frames，合计10,000；第一train世界真实frame smoke的scan、teacher、LOS和2.786 m净空PASS。
- Data Card/spec冻结后preflight 0错误/0警告，23个冻结工具哈希一致；唯一正式run为`gate2_20260822_aee_corrective_sensor_teacher_dataset_v1_seed20260822`。
- V1首个世界500帧全部计算并写入Zarr/manifest/preview后，在写`metrics/cano/S01_flat_tree_small_C13.json`时因父目录未创建触发`FileNotFoundError`。状态`FAIL_AEE_CORRECTIVE_SENSOR_TEACHER_DATASET_V1`，耗时9.740秒，181/181 seal，SHA-256=`1d41b86ec2dc34ce25bb24e4ba9b05d3160f9ce219245cf8325eee3b18570657`。
- 这是system目录初始化缺陷，不否定sensor、teacher或数据方法；AEE轨迹、训练、模型、C09/C10均为0。旧partial资产禁止复用。推荐V1R只显式创建`metrics/cano/`并增加测试，然后从第一个世界全量重跑；等待用户明确选择A。

## 2026-08-22：11,000帧 corrective sensor/teacher V1R 揭示姿态资格缺陷并正式FAIL

- 用户批准推荐方案A后，V1R只增加`metrics/cano/`初始化和薄入口；30/30测试、sidecar compile与final preflight 0错误/0警告PASS。唯一run从头执行且未读取V1 partial资产。
- 目录修复有效：前三个Cano世界1500帧和3份per-world metrics完整写出。第4个世界`S02_3d_tree_small_C14_t0003_k00029`第4帧的双场景raw/model回放精确一致、teacher标签和两分支LOS均有效，但横向净空仅`0.765984893m`，低于冻结`0.8m`，因此23.264秒时按合同停止。
- V1R状态`FAIL_AEE_CORRECTIVE_SENSOR_TEACHER_DATASET_V1R`，70MB，565/565 seal复核PASS，seal SHA-256=`a35626dd9499c3eea05021901580d549fe9eebbba381a57f214431b57e0a3b9a`；AEE、训练、模型、C09/C10全部为0，partial资产禁止复用。
- 全20-world固定pose的只读审计确认10000帧中7帧不合格、涉及6个世界：6帧横向净空不足（最小`0.689469814m`），另1帧净空`1.624943137m`但teacher完整性失败。当前选择器只做topology-only选择，因此原先单帧smoke没有证明全量姿态可用。
- 当前唯一NEXT需用户决定：推荐V1R2先对全部6058个候选cluster/30290帧做独立资格审计，再仅从5帧全部安全且teacher完整的cluster中按原事件/隧道/最远间隔目标全局选择100个/世界；重新证明覆盖后才冻结新Data Card/spec并从头执行。备选是做局部pose优化（实现与验证成本更高）或停止；禁止删7帧、降0.8m门槛或在V1R重试。

### V1R2 eligibility-first路线只读可行性proof

- 初版只读汇总曾把逐world失败帧误加为19；完整候选分母复核以类型化资格记录重算为21。6043个cluster的5/5帧全部安全且teacher完整，15个cluster/21帧被资格门禁拒绝。
- 在各world的eligible集合上重新运行原事件/隧道配额/最远间隔选择，20/20均得到精确100 clusters/500 frames，合计2000/10000；全部junction/terminal events覆盖，最大同隧道间隔仍逐world通过。
- 可行性结论是V1R2无需降阈值、删帧或移动pose。实现必须把旧selector audit中硬编码的`selection_blind_to_observations=true`更正为显式两阶段provenance：geometry/teacher eligibility先于最终结构选择，但不得使用模型输出、训练结果或C09/C10。该方法变化仍等待用户明确选择。
- 用户随后选择A。类型化完整候选分母proof耗时207.689秒：20/20 worlds、6058 candidates、6043 eligible、15 ineligible clusters/21 ineligible frames、2000 selected/10000 selected frames全部通过；过滤前定义的全部原始junction/terminal事件和隧道均零遗漏，0写入/训练/模型/C09/C10。

### V1R2实现与proposal已到正式批准边界

- 类型化frame/cluster eligibility要求5/5原子通过；qualified selector精确绑定6058候选identity，以过滤前事件/隧道为分母，结构评分阶段不读scan/teacher数值/模型，且明确不再声称最终selection对完整几何blind。
- 新V1R2 executor先审计30290姿态，再选2000 clusters，随后独立重算选中10000帧并要求qualification逐元素重放；保存6058-row provenance。runner同时核验精确聚合，并以进程树RSS和Docker cgroup执行4 GiB门禁。
- 37/37相关测试、Anaconda/E1 compile和207.689秒全量只读proof PASS。V1R2 Data Card/proposal/spec冻结20 Cano worlds、10 AEE trajectories、11000 retained/labels、4h/4GiB/40GiB、0 GPU/训练/C09/C10以及V1/V1R零partial复用。
- proposal preflight仅有`user_authorization.status`与Data Card `approval.status`两项预期错误，0 warning；正式目录不存在。NEXT是展示精确范围/成本/证据并取得一次正式执行批准，然后最终重冻approval/hash、preflight 0/0、create_run并执行一次。

## 2026-08-22：V1R2 Cano全量PASS，AEE启动期order pairing缺陷使正式run FAIL

- 用户批准精确V1R2后，37/37测试、26/26 frozen hashes及final preflight 0/0 PASS；唯一正式run从头执行且未读取V1/V1R partial资产。
- Cano阶段正式PASS：20/20 worlds、6058 candidates、6043 eligible、15 ineligible clusters/21 frames、2000 selected clusters、10000 frames和完整原始事件/隧道覆盖一致；duration 349.851秒，peak process-tree RSS 759226368 bytes。
- AEE第1条`tunnel seed11`完整PASS：3000 raw、100 retained、1152.335281m、sensor/teacher、1.942GB zstd archive及资源门禁均通过。第2条`tunnel seed23`采满约6.126GB bag后，在extract阶段因raw/registered order delta>=0.1秒停止；后续8条未启动。
- 只读取证确认bag有3009 raw、3002 registered且全部有exact-stamp odometry；raw从0.741秒开始而registered从2.138秒开始。旧代码从waypoint-ready时刻各取前3000条按序号配对，导致启动期错位。唯一单调近时匹配得到3002 pairs，前3000的raw indices=7--3006，max/p99/median delta=`0.003/0.003/0.001s`、>=0.1为0。
- V1R2整体状态`FAIL_AEE_CORRECTIVE_SENSOR_TEACHER_DATASET_V1R2`，8.0GB，3432/3432 seal独立复核PASS，seal SHA-256=`067c84ae60c3470e30cef50dc336c00674459ea1642e3572df4cc77898f6de6b`；partial Cano/AEE资产禁止复用，训练/模型/C09/C10为0。
- 问题分类为data/sensor synchronization contract，不否定Cano eligibility、AEE raw内容或teacher。推荐V1R3在完整bag中先构造唯一、严格单调、delta<0.1秒的raw↔registered exact-odom pairs，再以pair序列定义3000 raw frames并固定取pair indices `0,30,...,2970`；备选是延迟collection start直到两流ready（改启动语义且更脆弱）或停止。需用户明确选择。

## 2026-08-22：V1R3配对科学合同通过，但ownership接口改名使正式run FAIL

- 用户把既定论文目标内的方法与执行选择持续授权给Codex。V1R3冻结为严格单调最近时间戳匹配：exact-odom registered依次匹配未使用且不倒退的raw，delta严格小于0.1秒，首3000 pairs再固定取`0,30,...,2970`；世界、seed、帧数、teacher和阈值不变。
- 152项AEE回归测试PASS；真实V1R2 seed23失败bag用生产函数只读重放得到3000 pairs、raw indices 7--3006、max delta 0.003秒。ROS Python3.8容器导入PASS。
- V1R3正式run中Cano 20/20、10000帧PASS；AEE seed11得到3000 pairs、100帧、max delta 0.003秒、1148.690825m且sensor shard审计PASS。随后archive consumer拒绝wrapper新写的`PASS_..._V1R3` ownership状态；后续9条未启动。
- V1R3为immutable `FAIL_AEE_CORRECTIVE_SENSOR_TEACHER_DATASET_V1R3`，duration 991.679秒，3417项seal文件SHA-256=`2236082aaa2c606cc4fc4f17e85fc888c2446c58f3d6ace401503031ca6d3ac9`；partial资产禁止复用，训练/模型/C09/C10为0。
- 分类为system/interface defect，不否定配对或数据。V1R4只恢复稳定V1 ownership schema/status，保持全部科学合同；24项专项测试和preflight 0/0 PASS。NEXT为一个新的从头不可覆盖V1R4。

## 2026-08-22：V1R4完整11,000帧corrective dataset正式PASS

- V1R4从零执行，未读取V1--V1R3 partial资产。Cano阶段20/20 worlds、6058 candidates、6043 eligible、15 ineligible clusters/21 frames、2000 selected clusters和10000 frames全部PASS。
- AEE tunnel/garage各5条seed轨迹全部一次通过：每条冻结3000个严格单调exact-odom pair、固定保留100帧；10条最大pair delta总体上限0.004秒，最短轨迹1136.763062m。1000个frame ID唯一，sensor shard、teacher和角色覆盖全部通过。
- 全run为`PASS_AEE_CORRECTIVE_SENSOR_TEACHER_DATASET_V1R4`：11000 scans、11000 labels、6000 train、5000 validation；duration 7701.050秒，result 20,766,092,162 bytes，10个原始bag归档20,247,980,904 bytes，peak process-tree RSS 729,878,528 bytes。
- 3578/3578 seal独立复核PASS，seal SHA-256=`dd6d5ac8a5a48223d2693734700b11396dd12a3aec63e4bbeaf02cf6f5fbaa59`；training/model/M-TARE changes/C09/C10 reads均为0。NEXT是为三seed full-encoder corrective training冻结独立Data Card/spec和验收合同。
# 2026-08-27：普通状态变化点正式FAIL，主方法改为Factorized GSE-Graph

- 相关工作矩阵已冻结于`docs/GSE_ACTION_CONDITIONED_GEOMETRY_STATE_REVIEW_V1.md`：Cano已覆盖地下出口/纯拓扑，2012年工作已覆盖结构描述符+change-point，2025年道路工作已覆盖intersection layered topology。普通变化点不能作为创新。
- 新增零训练实现`gse_action_conditioned_state.py`及6项专项测试；正式run完整读取C01--C06 `142,184`和C07--C08 `45,942`冻结观测，C09/C10/M-TARE、推理和optimizer均为0。18项seal SHA=`e77543a2eda4d55403cc52f09aaa06189e7cf53d625adc13db36227fbe3a9692`，系统error=null。
- 科学结果FAIL：combined precision=`0.552632`、episode recall=`0.031111`、相对更强单项基线=`-0.005185`；turn/geometry-transition只覆盖`2/95`和`1/17` identities。只有`39,128/188,126`观测满足12-observation双块历史，且5维exit summary丢弃完整token关系。
- 决定不重跑、不调阈值、不训练普通change detector。新权威方法为Factorized GSE-Graph：junction/terminal作为decision nodes；turn和连续宽高/坡度/曲率进入trace-verified edge geometry profile。现有junction/terminal覆盖`140/146`、`125/128`和连续几何平均改善`52.763%`作为组件证据。
- 当前NEXT：只读构造C01--C08 decision-node/full-token/incident-edge-profile关联inventory，验证route-conditioned signature是否有足够正对、困难负对和因果profile覆盖；通过后才允许低容量association capacity proof。C09/C10/M-TARE和离线图继续0读取。

## 2026-08-27：Factorized association inventory发现pair Teacher失衡，停止直接训练

- 正式inventory系统正常、source unchanged、零optimizer/inference/C09/C10/M-TARE；17项seal SHA=`fc7781239b3e947ddc193f85ef9fec14c1ba1fae8fb96f2df02930cdcbe2cbeb`。
- 数据正面证据：fit/selection decision identities=`792/274`；至少5帧的causal inbound profile覆盖=`777/792`和`269/274`，均约`98.1%`；三个seed完整exit token字段全量有限；selection有`3,772`个跨physical-edge positive pairs。
- 决定性缺陷：旧pair cache过滤到decision节点后，fit=`23,359 positive / 27 negative`，selection=`8,083 / 71`；fit七个family、selection六个family为零negative。故当前Teacher无法训练或按family验证开放集decision association。
- 问题分类为data/Teacher sample-unit，不是否定Factorized方法。已停止capacity training；推荐保持runtime 16m候选域，新增identity-balanced structural-alias Teacher，每identity一个跨视角positive和一个同event/degree、objective geometry最近的不同identity negative。
- 当前NEXT：先冻结`docs/GSE_FACTORIZED_ASSOCIATION_TEACHER_PROPOSAL_V1.md`对应的逐identity manifest proof/Data Card；不扩大radius、不复制少量负对、不读取C09/C10/M-TARE。

## 2026-08-27 — Identity-balanced association Teacher manifest正式PASS

- 只读实现阶段发现5个短terminal identity各只有一条terminal观测；它们都具备完整5帧因果历史。未删除身份，也未把重复张量伪装成重访；只对这5个identity使用固定`180/720` circular-shift augmentation并单独计数。其余`1,061`个identity仍使用真实跨edge、reverse-view或distinct-observation positive。
- 11/11专项测试PASS；正式Data Card/spec/preflight通过后仅创建一个不可覆盖run：`results/gate3_semantics/gate3_20260827_gse_factorized_association_teacher_manifest_v1_seed0`。
- 正式结果`PASS_GSE_FACTORIZED_ASSOCIATION_TEACHER_MANIFEST_V1`：fit/selection=`792/274` identities，manifest=`1,066` units，pairs=`2,132`；positive kind=`563/491/7/5`，所有hard negative均为不同identity、相同split/event/degree且距离有限；10个family在两split均非空。
- 完整历史mask仍精确为fit `15`、selection `5`，没有删样本。duration=`4.589s`，peak RSS=`807,996 KiB`，run大小约`1.9 MiB`；optimizer/inference/C09/C10/strict/M-TARE均为0。
- 18项seal精确覆盖并复核PASS，SHA-256=`047b5d16b0083fcab56a2a04d286655b2711e148b704d98f30d958e7b1d90f5b`。PNG/PDF/SVG及source JSON已保留供论文Teacher/失败分析图使用。
- NEXT：冻结route-conditioned association capacity proof。学生只允许读取learned place descriptor、完整exit tokens、部署时可得不确定性、已穿越incident-edge geometry和runtime spatial distance；Teacher objective geometry、identity/world/parent/event标签禁止进入学生输入。未通过`P>=0.98 / false<=0.01 / recall>=0.25`前不进入C09、离线图或闭环。

## 2026-08-27 — Route-conditioned association capacity正式PASS

- 正式run=`results/gate3_semantics/gate3_20260827_gse_factorized_association_capacity_v1_seed0`；三seed各训练一个`8,017`参数full和`6,673`参数no-route模型，perception backbone/outputs完全冻结。
- full三seedsafe recall=`0.762774/0.645985/0.620438`，precision=`0.990521/1/1`，false accept=`0.009479/0/0`；删除selection中唯一augmentation positive后的physical-only子集仍全部满足总体及十family门槛。
- mean safe recall：descriptor-only=`0.218978`、no-route=`0.199513`、full=`0.676399`；full gain=`+0.457421/+0.476886`，全部贡献门槛PASS。seed1的descriptor/no-route均无安全阈值，而full达到`0.645985` recall且零false accept。
- 总small-head optimizer steps=`1,239`，backbone optimizer/inference/C09/C10/strict/M-TARE=`0`；duration=`16.071s`，peak RSS=`1,624,280 KiB`，run约`1.5 MiB`。
- 61项seal独立复核PASS，SHA-256=`91ef98c0bc5b4cd82492dd42499a57a1bfe22d020664fbb0189aa2345cee11cc`。论文图PNG/PDF/SVG/source JSON已保留。
- NEXT：冻结一次C09-only Factorized association qualification，模型/normalization/threshold只读；同时验证identity-balanced困难负样本和真实runtime严格过去16 m候选。C10、离线图和M-TARE仍禁止。

## 2026-08-28 — selection-only consensus+metric corrective正式PASS

- 新增类型化选择器、C09 applicator和两进程outer runner。进程1没有C09输入参数，只在C07--C08重建`8,839` queries / `9,380` pairs并选择；其退出且`c09_worlds_read=0`后，进程2才允许读取旧C09 FAIL的封存scores。
- 初次临时冒烟被计数门禁拦截：实现误把C01--C08全部行送入runtime构造。正式run创建前已修正为C07--C08局部候选后映射回188,126行冻结特征；不改数据、方法、grid或阈值。相关19项测试PASS。
- C07--C08固定grid为votes=`1/2/3`、distance=`0.5:0.5:16.0 m`；使用预注册开发裕量后8个配置合格，确定性规则选`2-of-3 + 4.0 m`。balanced precision/recall=`1.0/0.580292`，runtime precision/false/recall=`0.995027/0.004973/0.607356`。
- C09一次应用PASS：balanced precision/recall=`1.0/0.507692`，physical-only precision/recall=`1.0/0.503876`；runtime precision/false/recall=`0.990135/0.009865/0.558281`，全部aggregate/family/identifiability检查通过。
- 正式run=`results/gate3_semantics/gate3_20260828_gse_factorized_consensus_metric_corrective_v1_seed0`；duration=`31.744s`、peak RSS=`1,563,052 KiB`、source unchanged、C10/M-TARE/optimizer/model update=`0`。22/22 exact seal PASS，SHA=`295004061d1a0ef5e83f29e483625573516d0571b1e038bfc6656b640c67c714`。
- 正式PNG/PDF/SVG/source图已目视通过并保留。NEXT：在Phase 3内重新定义并冻结Factorized离线图资格，禁止复用旧五类节点grid；C10与闭环继续隔离。
## 2026-08-28：StructuredPolarMultiDepth三seed容量正式FAIL

- `DONE`：正式run完成3 seeds × 8 epochs × 9,096 steps=`27,288` optimizer steps；训练、评估、source-integrity和33项seal系统均PASS，`error=null`，C09/C10/M-TARE读取为0。
- `RESULT`：P/R/F1分别为`0.137586/0.212159/0.166922`、`0.103357/0.238045/0.144133`、`0.129531/0.165606/0.145364`，均显著低于best baseline F1=`0.390133`；定位MAE约`2.01m`是唯一三seed共同通过项。
- `CORRECTED`：同方位第二深度79目标recall为`0.088608/0.506329/0.240506`，但V1R证明这些匹配全部来自slot0，不能作为双槽学习收益；false positives=`44,078/68,447/36,887`。
- `FAILED`：Gate 3保持`GATE_FAIL`，分类为`DENSE_PRESENCE_OBJECTNESS_FAILURE_WITH_PARTIAL_MULTI_DEPTH_GEOMETRY_GAIN`；不得进入图、C09/C10或闭环。
- `EVIDENCE`：run=`results/gate3_semantics/gate3_20260828_gse_structured_polar_multidepth_capacity_v1_seed0`；seal SHA=`3186e311508a82fa5e6a8342806729ee4c9335cb9407b322f6420c2cf73503ee`；图SHA=`a16582801d99e9dd6e500061216e8fd797dd4fcb80bb32d3c812c738ebe85ac6`。
- `NEXT`：仅做sealed top16输出的零训练、零新推理objectness/proposal归因；根据预注册oracle/separability证据决定最小corrective或停止dense event路线。

## 2026-08-28：Structured-polar slot语义V1R决定停止该路线

- `V1 PASS`：typed top16 oracle recall=`0.674672/0.685352/0.619038`，但candidate AP=`0.060851/0.055909/0.051180`，one-per-bin和Teacher-cardinality上界均不能超过baseline。
- `V1 DEFECT`：oracle未保留预测depth-slot provenance，objectness-only refit结论不充分；V1 evidence保留，不修改。
- `V1R PASS`：top16 slot1候选=`0/16/0`，正式选中=`0/0/0`；79个second-depth targets由slot1匹配=`0/0/0`，distinct-slot near/far pair=`0/0/0`。
- `DECISION`：`STOP_STRUCTURED_POLAR_MULTIDEPTH_ROUTE_USE_EXECUTABLE_EXIT_TOKENS`。保留全部失败图作消融，不再重训双槽。
- `NEXT`：C01--C08只读full exit/action geometry token causal-event Teacher feasibility；0 C09/C10/M-TARE/graph。
- `EVIDENCE`：V1/V1R seal SHA=`d0ad99d0dfeeb57d6ad7c4d947d286623d46a92b0e415bef2129f71abd6ac925` / `8c9cf3714d8ceca617c57905573c8b7e86f7127dd2bcfa90accb84ab2d9d5ae5`。

## 2026-08-28：完整exit/action token因果transport feasibility PASS

- `DONE`：80个C01--C08 shard tree全部复核，人口精确为252,430 frames、188,126 observations、396,913 visible exit tokens、130,080/41,970 fit/selection相邻对。
- `TEACHER`：C07--C08可执行出口集合decision macro-F1=`0.995411`，decision episode支持覆盖=`1120/1136=0.985915`。
- `REPRESENTATION`：descriptor-only跨帧identity precision/recall三seed=`0.981889/0.988804`、`0.983521/0.990447`、`0.984711/0.991645`，证明结构token本身可追踪。
- `BASELINE`：旧mean/max pooling模型macro-F1/precision/episode recall=`0.743847/0.965870/0.498239`；固定5D变化点仅`0.552632/0.031111` precision/recall。
- `DECISION`：`IMPLEMENT_RELATIONAL_EXIT_TOKEN_TRANSPORT_EVENT_MODEL`。下一步只做保留token correspondence矩阵、uncertainty/refusal和episode commit的零训练readiness；边继续只在真实穿越后建立。
- `ISOLATION`：0 optimizer、新推理、threshold selection、graph、C09/C10/M-TARE；source unchanged；17项seal SHA=`10fa5df309c81d11254f3302280200fe19b7b77967f70b49f1f66f62da5f2f23`，图SHA=`86a7a8aff0b62f1f985155a413f697439fcd12a684c05aff9554fcb972089b5a`。

## 2026-08-29：Relational exit-token transport readiness V1R PASS

- V1因不存在的`torch.flatnonzero`在loss前system FAIL，seal=`9f4dcb8b...`；模型与科学门均未执行。
- V1R仅使用等价`torch.nonzero(...).flatten()`；专项/治理测试30项PASS，原FAIL被输入hash绑定。
- 模型参数=`240,101`；全量188,126 causal references违规0。token permutation误差最高`5.96e-8`，seed/masked/repeat误差0。
- transport sum error=`1.1921e-7`、std=`0.010695`；真实8行episode loss=`3.583166`且finite backward，typed provisional/refusal合同PASS。
- 决策=`ALLOW_RELATIONAL_EXIT_TRANSPORT_THREE_SEED_DATA_CARD`；下一步冻结C01--C06训练/C07--C08选择合同，继续禁止graph/C09/C10/M-TARE。
- V1R 17项seal SHA=`853b040875f3184e4c3389402655198e4bd13c43533d0e23d73e69861f32d01a`，图SHA=`dd9942c61ddf96164f9526e989d7239babb08ec6bd51b10f76f7fc8a5ca94b50`。

## 2026-08-29：Relational exit-token transport三seedcapacity科学FAIL

- `DONE`：3 seeds × 6 epochs × 6,666 steps=`19,998` optimizer steps；训练/评估/source/seal系统正常，C09/C10/M-TARE/graph/planner零读取。
- `SIGNAL`：单seed macro-F1=`0.827639/0.891699/0.838536`，证明模型学到了junction/terminal关系信号，但precision仅`0.869691/0.913255/0.732331`且seed校准不稳。
- `FAIL`：ensemble最佳可用precision/recall/macro-F1=`0.946612/0.405810/0.664556`，低于pooling baseline `0.743847`；0.995/0.99 safety门和+0.05增益门均失败。
- `ATTRIBUTION`：487 triggers中461正确；18个同episode重复、5个corridor误报、3个类型错判。主要缺口是无状态commit造成重复，以及缺少seed分歧拒绝，不是训练未运行或数据泄漏。
- `NEXT`：只读冻结输出，预注册past-only稳定/去重/consensus refusal状态机并做world-held-out可行性；不重训、不降门、不进图。
- `EVIDENCE`：run=`results/gate3_semantics/gate3_20260829_gse_relational_exit_transport_training_v1_seed0`；31项seal SHA=`de4cb8d37bc834236e6a1739b7e2d8d91f963a2299d7dcd51e42b7a7605aad6f`。

## 2026-08-29：Past-only commit-policy feasibility正式FAIL

- `DONE`：C07枚举336/类固定策略，C08只应用一次；0训练、新推理、C09/C10/M-TARE/graph。
- `SAFETY`：选中2-of-3、3帧稳定策略后，C07/C08 precision=`0.996241/0.997093`、duplicate=`0/0`，安全接口有效。
- `FAIL`：C07/C08 macro-F1=`0.753519/0.796339`；C07低于要求`0.793847`，不可用C08单侧PASS替代。
- `ROOT CAUSE`：旧MIL只奖励episode最大峰，不惩罚第二个峰；手写稳定规则提高安全但损失junction recall。
- `DECISION`：不扩大状态grid。下一步实现零训练exact-one structured episode likelihood readiness；只在数学合同和真实finite backward通过后考虑V2训练。
- `EVIDENCE`：run=`results/gate3_semantics/gate3_20260829_gse_relational_commit_policy_feasibility_v1_seed0`；16项seal SHA=`bc8771dd710f33053531f47202c4ee0a15482b58273469307a5584306fadaa47`，图SHA=`ab5b993bc1fb38622d8183242817a6267005f2dc2ce1df9dd6cc33b9f16658fa`。

## 2026-08-29：Structured exact-one event loss readiness PASS

- `SEMANTICS`：one/zero/two/wrong peak loss=`0.049946/3.546859/3.264189/4.596299`，正确单峰排序全部PASS。
- `ROOT FIX`：旧max-MIL对第二峰梯度=`0`，exact-one梯度=`0.831963`；新loss明确惩罚重复节点提交。
- `NUMERICS`：episode permutation error=0，extreme logits finite；真实128行、3个完整episode batch loss=`2.827224`并finite backward。
- `ISOLATION`：240,101参数不变，0 optimizer/trained inference/checkpoint/threshold/C09/C10/M-TARE/graph。
- `DECISION`：允许V2三seedData Card；C07选checkpoint/policy，C08一次迁移验证，两侧必须保持原安全和F1门。
- `EVIDENCE`：run=`results/gate3_semantics/gate3_20260829_gse_structured_exact_one_event_loss_readiness_v1_seed0`；15项seal SHA=`ced78ead9cf7773f51ee698d4ae4ae6f6b19a631530054797e151b9c47b07426`，图SHA=`16a79c5554d519426e002fd9ef0469690853f844ee2402781667c603c0cfb0ca`。

## 2026-08-29：Structured exact-one三seedcapacity正式FAIL

- `DONE`：3 seeds × 6 epochs=`19,998` steps；C07-only checkpoint/policy、C08零更新，系统和隔离全部正常。
- `C07`：P/R/F1=`1.0/0.332083/0.573256`，junction recall=`0.264423`。
- `C08`：P/R/F1=`0.990783/0.356551/0.582755`，安全和F1均失败。
- `CONCLUSION`：exact-one消除重复但过度拒绝，显著差于V1 stateful F1=`0.753519/0.796339`；停止直接事件分类路线。
- `NEXT`：只读审计learned exit-token validity + descriptor tracking + stable action set；不训练新头、不进测试或图。
- `EVIDENCE`：run=`results/gate3_semantics/gate3_20260829_gse_structured_exact_one_event_training_v1_seed0`；32项seal SHA=`51d3a8964ca7c556ccafaad93ee1392804ec429273f73db497cf6b791da3306b`，图SHA=`3b2c99f55e89d375f03ae30f2a202a4e66955859702a75895733084749f06156`。

## 2026-08-29：Frozen free-query token validity/action-track V1R2正式科学FAIL

- `SYSTEM HISTORY`：V1在preflight因Data Card validation world字符串停止，零run；V1R因inner/outer summary读取接口错误在指标前system FAIL。V1R2只修summary读取并绑定前序失败，7项相关测试PASS。
- `DONE`：80 worlds、252,430 raw frames、188,126 causal observations；每seed 1,128,756 slots，positive/negative=`396,913/731,843`；C07/C08=`21,548/24,394` observations。
- `RETAINED SIGNAL`：Teacher action-set macro-F1=`0.995411`，descriptor transport最小P/R=`0.981889/0.988804`，所以物理结构容量和已知真出口的身份跟踪仍有效。
- `FAIL`：precision>=0.995时，单帧validity recall=`0.093888/0.000010/0.006379`；五帧/三seed最佳=`0.092435/0.000423/0.010820`，AP gain全部为负。
- `ROOT CAUSE`：六自由query产生跨帧、跨seed稳定的ghost slots；descriptor persistence只能跟踪槽位，不能证明槽位对应真实可执行出口。
- `DECISION`：停止frozen free-query token validity路线，不训练新objectness/refusal头。下一步只做180-bin Dense Circular Traversability Field Teacher可行性；0训练、C09/C10/M-TARE/graph/planner。
- `EVIDENCE`：run=`results/gate3_semantics/gate3_20260829_gse_token_validity_action_track_feasibility_v1r2_seed0`；17/17 seal SHA=`41c23bdf50f90cfec427d93add4ac7e6eecc001c41fb2f73655643c105986729`；图SHA=`2d176ed727b6207ef5bdf7fc0e0565d61022551bec498ed90176bb8fa36c3236`。

## 2026-08-29：Circular executable-geometry peak field V1R正式PASS

- `REPRESENTATION CORRECTION`：宽扇区连通分量在`22,626/188,126=12.027%`观测合并/拆分出口，正式停止；改为180个2°bin的中心peak+sub-bin residual+width/mask+vertical profile。
- `V1 SYSTEM FAIL`：357°半bin边界被encoder float64和audit float32分到相邻bin，导致评估器读取错误bin；其余门PASS但无科学结论。
- `V1R PASS`：共享canonical转换后396,913 exits same/adjacent-bin collision=`0/0`；heading/rotation误差=`2.98e-8°/5.33e-6°`，width/profile误差0，slot permutation和五帧future violation均0。
- `BASELINE`：raw-range envelope fit/selection AP=`0.052647/0.051493`，在precision>=0.995时recall=`0/0`；表示可行不等于非学习规则已解决任务。
- `ISOLATION`：80 worlds、252,430 frames、188,126 observations；0 Teacher export/optimizer/new inference/C09/C10/M-TARE/graph，source unchanged、`error=null`。
- `DECISION`：允许一次Gate-2 Teacher export Data Card；导出后仍需模型readiness，不能直接训练或建图。
- `EVIDENCE`：run=`results/gate3_semantics/gate3_20260829_gse_circular_exit_geometry_field_feasibility_v1r_seed0`；17/17 seal SHA=`ddad17c78f26ec05754c35ed88dc640473f30e9ea7c0b021995b2ed02fab663e`；图SHA=`360d2a8539941380b2f336281cb50328e1217f96648cad2b5b6e80fc334444ec`。

## 2026-08-29：Circular peak-field Teacher export正式PASS

- `DONE`：80 world shards、188,126 observations、396,913 peaks；fit/selection=`142,184/45,942` observations和`299,872/97,041` peaks。
- `CONTENT`：仅180-bin presence/residual/width-mask/profile、count和因果references；无LiDAR、pose、identity、future或测试世界。
- `INTEGRITY`：80 shards逐数组回读一致，heading误差=`2.98e-8°`，source unchanged；2,949条seal独立复核PASS。
- `SIZE`：Teacher数据`22.243 MiB`，完整run 36 MiB。
- `DECISION`：允许零训练model readiness；仍禁止直接训练、C09/C10/M-TARE/graph/planner。
- `NEXT`：验证五帧circular encoder + 180-bin peak heads的等变性、掩码loss、因果输入和真实batch finite backward。
- `EVIDENCE`：run=`results/gate2_representation/gate2_20260829_gse_circular_exit_geometry_field_teacher_export_v1_seed0`；seal SHA=`8c37f11a9e08b3cf0c48d005ae32aa2d79afcbba7c2330333ee9996605657f76`；图SHA=`044cccf8b8e8fe68e24446ae7b371a2662c4bfb270f03fd3a5f456d655cc0780`。

## 2026-08-29：Circular peak geometry model readiness正式PASS

- `MODEL`：769,268参数；五帧因果LiDAR输入，180-bin peak+geometry+descriptor+uncertainty，并同时输出axis/width/height/slope/curvature；free-query参数=0。
- `REAL DATA`：全量188,126 references/join无误；8行真实batch覆盖1--4 exits、wrap和invalid width，共20 peaks。
- `BACKWARD`：所有输出/loss/gradient finite；无效width与nonpeak profile对loss影响精确为0。
- `EQUIVARIANCE`：dense/axis/global rotation error=`9.54e-7/9.48e-6/3.34e-6`，batch error=`9.54e-7`，repeat exact。
- `CAUSALITY`：五帧references全部past-only；最老帧变化使logit变化0.0333，时间路径有效。
- `DECISION`：允许三seed训练Data Card；C01--C06梯度、C07选择、C08一次迁移，旧C07/C08 checkpoint不得初始化主方法。
- `EVIDENCE`：run=`results/gate3_semantics/gate3_20260829_gse_circular_peak_geometry_model_readiness_v1_seed0`；seal SHA=`d06089680b97544a251e30e9c0214fa6c14df31fb3d30ef0c3d97852adc74c69`；图SHA=`38aa751b6f67a0a19c8cf5e6a5373fc1835eb85a5a6c055b33034c49b3b954ab`。

## 2026-08-29：Circular peak geometry V1三seed训练正式FAIL

- `DONE`：3×10 epochs，34,110 steps；best epochs=`8/9/8`，系统正常、C08零选模、测试/graph/planner零读取。
- `LEARNED`：C07/C08 axis=`4.53°/4.71°`，width=`0.721/0.735 m`，height=`0.400/0.541 m`，curvature=`0.00482/0.00470 per m`；连续结构学习有效。
- `FAIL`：slope=`3.14°/3.23°`；exact 2° local-peak AP=`0.0690/0.0575`，无0.995 precision threshold，安全recall/action F1均0。
- `ATTRIBUTION`：预测峰距Teacher median/p90/p99=`2°/8°/16°`；±10° AP约0.90，但每帧约19.7假峰。top4+NMS需约20°容差才安全recall>0.5，不能只改评分。
- `NEXT`：正式零训练soft angular target + hard-negative ranking readiness；只改peak target/loss，保留有效backbone/geometry heads和split。
- `EVIDENCE`：run=`results/gate3_semantics/gate3_20260829_gse_circular_peak_geometry_three_seed_training_v1_seed0`；seal SHA=`5477c4b7ddc32313d186e1a8a5b7dbb0dc8867cf9b52bb1b2fffb737b87ef018`；图SHA=`4569ea433eba92c73ab2e47a244e3d1231e190c282150cd94cf599ec41756289`。

## 2026-08-29：Causal Circular Exit Set Process三seed训练正式FAIL

- `DONE`：3×10 epochs，34,110 steps；best epochs=`9/8/8`，系统正常、C08零选模、C09/C10/M-TARE/graph/planner零读取。
- `LEARNED`：C07/C08出口数量准确率=`0.967143/0.959867`，raw结构macro-F1=`0.946488/0.938677`；结构类别学习成立。
- `FAIL`：在precision>=0.995时C07/C08仅接受`10/8`个观测，exit recall=`0.000440/0.000310`，完整集合覆盖和deployed action F1近零；slope=`3.412/3.491deg`仍失败。
- `INTERPRETATION`：极少数接受出口的bearing/width/profile很准，但样本太少；当前证据不能区分“多数出口定位不准”和“定位可用但置信度排序失效”。
- `NEXT`：只读冻结三seed预测做全人口角误差、容差曲线、cardinality分层与confidence separability归因；0训练/新推理/test/graph/planner。
- `EVIDENCE`：run=`results/gate3_semantics/gate3_20260829_gse_circular_exit_set_process_three_seed_training_v1_seed0`；88项seal SHA=`18a5dcb717ec7133aa0b27056fc18e844510ddbf738048ef8fd05ae481413f92`；图SHA=`29095d2986ae11940b4ef4ffbdf13969f8e809e02ec0ae0318d02fc96aa5221d`。

## 2026-08-29：Set Process冻结输出归因正式PASS

- `LOCALIZATION`：2deg exact-set=`0.2504/0.2258`，10deg=`0.6553/0.6520`；近邻信号存在但正式精度不足。
- `MULTI-EXIT`：三出口10deg exact-set仅`0.0479/0.0521`，四出口近零；单一强度场在复杂结构发生模式漏失。
- `COUPLING`：selected-bin residual median改善约0，而oracle Teacher-bin residual mean error仅`0.469/0.478deg`；训练/解码位置错位。
- `CONFIDENCE`：10种冻结分数在0.995 precision下最佳C07/C08 exact-set recall=`0.001854/0.001453`，后处理不可修复。
- `DECISION`：实现Cardinality-Conditioned Circular Slot Transport readiness；K个圆周slot一一匹配K个出口，直接连续bearing，无free-query existence。
- `EVIDENCE`：run=`results/gate3_semantics/gate3_20260829_gse_circular_exit_set_process_failure_attribution_v1_seed0`；18项seal SHA=`7ff969054736b21c31d61177a5c69b87c1a02b537a1144b0ab97c3a269662b1f`；图SHA=`892dd013e9ce907850f2ddbfad2123a31b93202ab0e9c53a9e1fe7d5411bc4d6`。

## 2026-08-29：Cardinality-Conditioned Circular Slot Transport readiness V1R PASS

- `V1 SYSTEM`：布尔valid-mask减法在科学结果前失败；V1封存。V1R只改logical mismatch并新增回归测试。
- `MODEL`：787,328参数、0 query/objectness；K由count决定，K个azimuth slots必须一一匹配K个出口。
- `BIJECTION`：correct/duplicate loss=`4.000/9.750`，缺失mode梯度=`-0.09375`；重复槽有直接纠正信号。
- `CONFIDENCE`：sharp/uniform concentration=`0.99998/~0`；confidence与定位集中度绑定。
- `REAL/SYMMETRY`：真实1--4出口batch finite；mass/resultant/count/geometry/slot/batch误差均约1e-6或更低，repeat=0，history连通。
- `NEXT`：冻结三seed训练；必须按cardinality分层报告，三/四出口失败不能被总体平均掩盖。
- `EVIDENCE`：V1R seal SHA=`32ab6cb9ea65177c38ef5ef0b2eb88f61ef123943ffe740694a7d6de4cc00846`；图SHA=`8cdb43a46fd1827b4f64857f143dfe86904ff7f98f28e71b887f310d587f2445`。

## 2026-08-29：Cardinality-Conditioned Circular Slot Transport三seed正式FAIL

- `SYSTEM PASS`：3 seeds × 10 epochs=`34,110` steps；best epochs=`8/8/9`，best C07 losses=`2.334300/2.333513/2.328594`；source unchanged、error=null、C08零选模、C09/C10/M-TARE/graph/planner零读取。
- `REAL GAIN`：C07/C08整体exact-set 2deg=`0.3413/0.2778`、10deg=`0.8380/0.8204`；相对set-process均提高。三出口10deg=`0.3446/0.2853`，证明bijective slot减少模式漏失。
- `FAIL`：三出口2deg=`0.0130/0.0071`，四出口2deg=`0/0`；99.5% precision下只接受`5/4`帧，exit recall=`0.000220/0.000155`；slope=`3.393/3.483deg`。
- `DECISION`：候选保留为粗粒度结构表示与消融，但不确立为最终方法，不进入图。下一步只读归因mean/mode、多峰、seed alignment与confidence separability。
- `EVIDENCE`：run=`results/gate3_semantics/gate3_20260829_gse_cardinality_conditioned_circular_slot_transport_three_seed_training_v1_seed0`；88项seal SHA=`472630ffd88a1331ead2abfdaf8f3469c6e1501d97d041b2d109685253f06455`；图SHA=`c36d02071e5604b3a991c5d40cd29a306bf11195590a8e81095fd9b1a4b9871a`。

## 2026-08-29：Slot Transport冻结失败归因正式PASS

- `EXCLUDED`：best local mode=`0.3238/0.2858`、best single seed=`0.2977/0.2704`、oracle count=`0.3440/0.2790`，均不能解释/修复ensemble mean=`0.3413/0.2778`。
- `ROOT CAUSE`：三出口bearing p50=`5.69/6.45deg`、target-two-bin mass p50=`0.131/0.124`；四出口=`8.47/10.44deg`和`0.088/0.063`。slot distribution随cardinality扩散。
- `CONFIDENCE`：best negative-entropy AUC=`0.8068`，但99.5% precision recall仅`0.00464/0.00310`，后处理不足。
- `DECISION`：下一candidate为Cyclic-Ordered Unimodal Slot Transport；只允许圆周顺序保持的cyclic matching，并将slot投影为proper unimodal circular distribution。
- `NEXT`：零训练readiness；反向/crossing受罚、rotation/cyclic permutation、concentration数值、duplicate gradient、真实1--4 finite backward。
- `EVIDENCE`：run seal SHA=`06bee093f8a18e3eecd4bbbe407db97416ce986ac6169c9bf45235683bb3242c`；图SHA=`576b9b351956f518e434910a562e83b61d49f0de4d79ba1e2422555e1297bed2`。

## 2026-08-29：COUST readiness V1R PASS，进入训练准备

- 实现788,618参数COUST：K出口只做圆周顺序保持的循环匹配，raw angular evidence投影为单峰离散von-Mises，concentration表达定位不确定性；无free query/objectness。
- V1因把float32 atan2的`0.003601deg`诊断混入dimensionless `3e-5`门而FAIL；实际slot probability rotation error=`2.67e-7`。失败run和17项seal保留。
- V1R不改模型、数据或`3e-5`阈值，只以模型实际消费的概率分布为旋转权威量；80 worlds、188,126 observations、396,913 exits与真实1--4出口batch全覆盖，11/11检查PASS，authoritative最大误差=`7.63e-6`。
- 模型与replacement检查7/7测试PASS；两个正式run各17项seal复核无漂移；0 optimizer/checkpoint/threshold/C09/C10/M-TARE/graph/planner。
- 方法尚未由性能确立。唯一NEXT：冻结COUST三seed训练Data Card/spec并执行；必须直接超过旧slot transport的K>=3严格定位和安全拒绝，否则停止。

## 2026-08-29：COUST三seed正式训练科学FAIL，转冻结输出归因

- 完成3 seeds ×10 epochs=`34,110` optimizer steps；best epochs均8，C07 loss=`3.2033/3.2575/2.1209`，784,266 development inference observations，source unchanged、`error=null`。
- 总体2deg exact-set相对旧Slot Transport提高`+0.0743/+0.1092`到C07/C08=`0.4156/0.3870`，raw count accuracy=`0.953/0.947`，证明简单结构有真实收益。
- 三出口2deg反而降到`0.00240/0.00179`，四出口=`0/0.00433`；总体改善被单/双出口多数类驱动，核心复杂结构假设失败。
- C07安全阈值只接受56观测，recall=`0.00246`；迁移C08 precision=`0.818`、recall=`0.00419`，concentration不可安全迁移。slope=`3.435/3.514deg`仍失败。
- 决策停止COUST进入graph；不得挑seed2或改门。下一唯一任务是只读冻结预测归因K>=3 slot collapse/order/start/ensemble/kappa，0新训练/推理/test/graph。
- 88项seal复核PASS；seal SHA=`47e4a84354ad58498e24de839fab7a3b33fb0dbd0846213a12dcfd5114bffd80`；图SHA=`d5f533ebf25b1c60431c0d30600cf2c7c2c1cd64fc6852bc1c933389b4e26129`。

## 2026-08-29：COUST复杂cardinality冻结归因正式PASS

- `DONE`：C07/C08=`21,548/24,394` observations、`45,504/51,537` exits、3 seeds；0训练/新推理/test/graph/planner，source unchanged。
- `EXCLUDED`：unrestricted ensemble对K=3 strict exact-set只提高`0.00068/0.00179`；最好single seed仅`0.00411/0.00565`，seed alignment和挑seed均不可修复。
- `SPACING`：ensemble K=3 gap-collapse比例=`0.178/0.173`，低于0.25门；约94% unrestricted最优assignment本来就是cyclic，粗间距不是主要缺失。
- `CONFIDENCE`：K=3 concentration AUC=`0.202/0.434`、kappa AUC=`0.350/0.399`；复杂结构置信度方向错误或近随机。
- `CONCLUSION`：根因=`COUST_K_GE_3_LOCALIZATION_AND_CONFIDENCE_OBJECTIVE_FAILURE`；停止独立slot distribution路线。
- `NEXT`：实现Joint Cyclic Gap Simplex零训练readiness：phase+K个正gap联合闭合圆周、cyclic-invariant loss、真实1--4 finite backward；不训练、不进测试/图。
- `EVIDENCE`：run=`results/gate3_semantics/gate3_20260829_gse_coust_complex_cardinality_failure_attribution_v1_seed0`；16项seal独立复核PASS，SHA=`bc6d05a03a2cd48d88e3f3a498a6286d6b4d670b1c79be5f8ae58102c3ff866c`。

## 2026-08-29：Joint Cyclic Gap Simplex readiness V1方法合同FAIL

- `PASS`：完整人口/join、1--4 cardinality、positive 360° closure、collapsed-gap梯度、proper scale、789,650参数真实8行finite backward、历史连通和零禁用操作。
- `FAIL`：phase mass rotation=`1.40e-9`，但近零resultant使degree bearing=`0.00720°`和sampled geometry=`1.97e-4`不满足统一`3e-5`门。
- `CONTRACT DEFECT`：V1没有proper phase likelihood，whole-set confidence也未绑定phase concentration；低可观测phase不能安全拒绝。
- `DECISION`：V1不训练。V2只增加proper phase NLL、phase-concentration refusal和distribution/resultant rotation authority；其他人口/结构/隔离不变。
- `EVIDENCE`：run=`results/gate3_semantics/gate3_20260829_gse_joint_cyclic_gap_simplex_readiness_v1_seed0`；17项seal SHA=`e2e636523e49ef0db711c8232a5f050a9740def9e5ab924bf8015d510497a942`。

## 2026-08-29：Joint Cyclic Gap Simplex V2 readiness正式PASS

- `DONE`：80 worlds、188,126 observations、396,913 exits、789,650参数；11/11检查PASS。
- `PHASE`：correct/uniform/wrong proper NLL=`0.000355/5.192957/16.000355`，真值梯度正确；低phase concentration绑定whole-set refusal。
- `JOINT SET`：K=1--4正gap closure误差最大`1.19e-7`；collapsed gap有`0.001787`纠正梯度。
- `REAL/SYMMETRY`：真实1--4 finite backward；phase mass/resultant/gap/batch权威误差均`<=9.54e-7`，repeat=0。
- `NEXT`：冻结一次三seed训练；K>=3 strict 2deg恢复和安全拒绝是确立方法的硬门。
- `EVIDENCE`：run=`results/gate3_semantics/gate3_20260829_gse_joint_cyclic_gap_simplex_readiness_v2_seed0`；seal SHA=`42c335f0ebdb7d0c97fef67746513d1d96bc28ba2861c5117ce482080f002b14`。

## 2026-08-29：JCGS singleton重新资格PASS；首次三seed训练系统FAIL

- `CORRECTION`：K=1 target closing gap由错误的0度改为360度；7项unit tests PASS。
- `READINESS V3`：相同80 worlds/188,126 observations/396,913 exits重新执行，singleton=`360.0deg`且V2全部11项科学检查保持PASS；0 optimizer/checkpoint/test/graph/planner。18项seal SHA=`b9e896962e0288b817c3e8838e7c7fe0a215a212cbe2b74ad9b8db70203cf605`。
- `TRAINING V1 SYSTEM FAIL`：seed0完整10 epochs/11,370 steps并导出45,942 development predictions；seed1在epoch4发生CUDA scatter/gather index out-of-bounds，seed2与统一selection未运行。两个保存checkpoint全部finite，故没有科学PASS/FAIL结论。
- `ISOLATION`：C08 checkpoint observations=0；C09/C10/M-TARE/graph/planner=0；失败run不可覆盖，36项seal SHA=`d7fe9281a168a1d6ca30b7a48268e598868b4797041410814b58e5a7eb1fe499`。
- `ROOT HYPOTHESIS`：低resultant phase的`atan2`允许singular/NaN backward，nonfinite梯度进入optimizer后使下一forward bearing索引非法；需稳定梯度和显式finite fail-closed证明。
- `NEXT`：实现stable circular phase angle、finite bearing/gradient regression与修正readiness；PASS后从头执行V1R三个seed，不续跑V1、不挑seed0。

## 2026-08-29：stable-phase V4 PASS，但V1R重复同一CUDA越界

- `V4 PASS`：stable atan2 backward、nonfinite bearing pre-gather和nonfinite gradient pre-step检查全部unit/真实batch通过；18项seal SHA=`4e649135b383d3e3fc47999daf034d75f7f4e379ac24fe2c2bc417eab2632559`。
- `V1R`：seed0完整10 epochs/11,370 steps、best epoch=8、C07 loss=`-2.197774`、45,942 predictions；seed1仍在epoch4触发相同CUDA gather out-of-bounds；seed2/selection=0。
- `UPDATED ATTRIBUTION`：finite-gradient guard未先触发，stable phase未改变失败时刻；根因不再归为phase singular backward，而是尚未同步定位的finite index boundary/kernel错误。
- `NO SCIENCE`：V1R无统一C07/C08指标，不得挑seed0；C09/C10/M-TARE/graph/planner=0。36项seal SHA=`a5f169e51b2ecd08341dfcbc63e45af1da2ff6692c2a3e2a0d0373ad6f9ad91c`。
- `NEXT`：只跑seed1最多5 epochs且`CUDA_LAUNCH_BLOCKING=1`的不可覆盖诊断，获得真实kernel调用行和索引范围后再修；不再盲目三seed重训。

## 2026-08-29：JCGS真实CUDA越界归因完成；周期端点V5R重新资格PASS

- `ATTRIBUTION`：同步CUDA在seed1 epoch4、5,673个成功step后复现；真实失败行是phase-log-mass gather，批次=`S10_3d_complex_C01/batch9/shift360`，样本global index=`180664`。
- `ROOT CAUSE`：角度`-2.842170943040401e-14°`在浮点圆周remainder中得到连续bin `180.0`；旧代码把它直接作为索引。Teacher bins为0--179且输出finite，排除Teacher污染与phase NaN。
- `FIX`：canonical periodic indexing将圆周端点180映射到0，不做clamp；模型sampling与phase loss共享同一转换。12/12 unit tests PASS。
- `V5 SYSTEM FAIL`：遗漏`CUBLAS_WORKSPACE_CONFIG`使确定性CUDA在loss前停止；0 optimizer/checkpoint，seal=`8e7527c7258f810471e2effa8e8efc18523a3fe092413e5d77f8b09c938c798a`。
- `V5R PASS`：只恢复训练环境变量；真实128行CUDA forward/backward与梯度finite，raw/canonical OOB=`1/0`，prior readiness全PASS，20项证据，seal=`14a249d19860be0c6c986a1ba66e606b8e633da2d7c3da187e9b085b031ef152`。
- `NO SCIENCE YET`：这仍只证明候选可训练。唯一NEXT为从头V1R2三seed训练；其K>=3严格集合与安全拒绝结果决定方法是否确立。

## 2026-08-29：JCGS V1R2三seed完整，但科学结果正式FAIL

- `SYSTEM PASS`：3×10 epochs、34,110 steps，best epochs=`8/8/7`；三seed全部越过旧CUDA故障，784,266 inference observations，error=null，C08零选模、测试/graph/planner零读取。
- `OVERALL FAIL`：C07/C08 strict 2deg exact-set=`0.2008/0.1572`，比COUST低`0.2148/0.2298`。
- `COMPLEX FAIL`：K3/K4在2/4/10deg complete-set全部0；K1/K4 count accuracy也为0，模型退化为双出口多数模式。
- `REFUSAL FAIL`：C07只接受1帧才达到precision 1，recall=`4.40e-5`；迁移C08仍接受1帧但错误。deployed macro-F1近0。
- `GEOMETRY FAIL`：axis=`9.75°/10.52°`，slope=`3.37°/3.46°`；其余宽高曲率通过但不足以确立方法。
- `DECISION`：停止JCGS进入图；下一任务只读冻结输出归因count/phase/gap/alignment/confidence，不训练、不新推理、不读测试。
- `EVIDENCE`：run 126 MiB，89项seal SHA=`2290a4262da57a35d9755368913f8f0c8fa6a5614934a1c282648aa1d560533b`。

## 2026-08-29：JCGS冻结归因完成，主方法改为轴锚定事件—关系分解

- `DONE`：C07/C08=`21,548/24,394` observations、`45,504/51,537` exits、3 seeds；formal exact2逐位复现，5项unit tests PASS，RAM=`776,244 KiB`，source unchanged。
- `EXCLUDED`：oracle count只提高exact2=`0.01211/0.01099`；Teacher-aligned seed ensemble再提高=`0.000046/0`，数量与seed对齐均非主因。
- `ROOT`：K>=3 oracle-phase exact10=`0/0`，oracle-gap exact10=`0.11268/0.12027`；phase和gap shape同时失败，JCGS完整集合回归路线终止。
- `RETAIN`：五帧circular encoder、结构event/cardinality、width/height/curvature、旧exit descriptor transport和exact-once commit证据；全部旧出口表示保留为消融。
- `NEXT METHOD`：Axis-Anchored Geometry-Semantic Event Relation；用运动/局部轴线固定参考系，学习事件类型和相对continuation/branch关系，而不是任意phase下的一次性完整集合。
- `NEXT`：先做C01--C08只读Teacher feasibility，验证轴锚可观测性、关系唯一性、反向traversal闭合、事件人口和无identity泄漏；0训练/新推理/test/graph。
- `EVIDENCE`：run=`results/gate3_semantics/gate3_20260829_gse_joint_cyclic_gap_simplex_failure_attribution_v1_seed0`；18项seal SHA=`c627c1265f26b2dbcc49de542b9befae1a838f14d258bbe6772e17a2b424d1c4`；图SHA=`b61df1f3d0e6660b8dd29b3d6c3bc90c9dbde71b8cb010b27e890326a216899b`。

## 2026-08-29：轴锚定事件—关系Teacher feasibility正式PASS

- 精确审计80 worlds、252,430 raw frames、188,126 observations、396,913出口和4,493个结构identity；完整数据源33,083项seal重验PASS。
- sensor-forward route axis因果可得；全部identity有唯一完整branch star，全部563个junction均有至少4个route views，K3/K4 junction占比fit/selection均>95%。
- fit/selection有`2814/995` reveal和`3032/1080` withdraw，五事件与width/height/slope/curvature监督充足；最小出口分离`5.0339deg`。
- 旧descriptor transport P/R均>0.98可作为表征证据；旧commit高精度但低召回，只作基线。7个局部K2 junction保留为provisional案例。
- `CURRENT`：方法方向已确立为Axis-Anchored Geometry-Semantic Event Relation，但网络实现和性能未确立，不能建图。
- `NEXT`：实现typed model/loss/memory并做零训练readiness；验证rotation/permutation/reverse traversal、identity不进forward、真实batch finite backward。PASS后才冻结三seed训练。
- `EVIDENCE`：19项seal SHA=`709930fc5c2e7774883bd5493851b1d9f60d07a79e0926b1662cdfee3219a090`；图SHA=`6407e28a89e8b193b97f79b06a9ec96ac767636d80c28f660bc90196df9a1569`。

## 2026-08-29：Axis-Anchored Event Relation method readiness正式PASS

- 新模型819,196参数，forward只接受五帧LiDAR；输出五事件、四类逐方位时间关系、分支/整体几何、两类descriptor和uncertainty。
- 固定7个真实C01 observations/29 unique frames覆盖五事件、四关系及place/branch正负pair；7项unit与15项formal checks全PASS，真实loss/gradient finite。
- rotation relation/event=`2.68e-7/2.98e-8`，batch permutation=`2.98e-8`，reverse/repeat=`0`；源33,083项seal重验PASS。
- 默认TF32曾使rotation=`9.64e-5`失败；保持原`3e-5`门并关闭TF32、启用确定性算法后通过，后续训练环境必须冻结该设置。
- `CURRENT`：方法和网络合同已经确立，科学性能尚未确立；仍不能建图。
- `NEXT`：三seed训练Data Card/trainer/evaluator，C01--C06梯度、C07选择、C08一次迁移；分事件/关系/几何/关联/拒绝报告。
- `EVIDENCE`：18项seal SHA=`022b0c0afec897ae5a3d33c3ddd30bb76d0c648038a8553b8c7a740b9d5931e5`；图SHA=`df85811305e643d6ef9f70e73b66b2cbd6c4a3c77e1c1e1248035e4d3d594a80`。

## 2026-08-29：masked early-relation训练合同V1R正式PASS

- 保留全部188,126样本；只有缺观察级Teacher的早期relation pair使用`-1` mask，事件/当前分支/几何继续监督，不删帧、不造标签。
- 有效pair positions fit/C07/C08=`448152/66752/77490`；fit relation bins=`79711009/934760/10583/11008`。
- event/relation inverse-sqrt权重仅由C01--C06冻结；C07/C08禁止重算。core与identity-balanced descriptor loss分开调度。
- 同一7-row真实batch的masked core/descriptor CUDA backward finite，8项unit和原15项不变量全PASS。
- `NEXT`：实现完整Teacher rasterizer、全人口core scheduler、identity-balanced sampler、三seed trainer和C07/C08 evaluator。
- `EVIDENCE`：18项seal SHA=`cce9ff427daa2f5399a592f10c5fd75bd9c7957b91e7bc42db380a92126c0c14`。

## 2026-08-29：显式数量与真实几何训练接口纠正PASS

- 六个proposal不再隐式等于六个出口；新增`Linear(128,7)`显式预测每帧0--6个结构分支，参数增加903。15项正式检查PASS，未来帧对过去count影响为0，rotation/batch误差=`3.58e-7/2.53e-7`，53个旧骨干键逐值兼容。
- 实现完整训练loss时，真实batch揭示vertical profile为4维而旧token uncertainty错误输出6维。未启动训练；删除唯一无目标通道，使geometry head减少65参数，总模型=`784,513`。
- 修正后8条真实观察的proposal/count/transport/event/token geometry/global geometry/uncertainty/place与branch descriptor全部loss有限；全部trainable gradient存在、有限且非零。15项unit及11项formal shape/因果/旋转/排列/隔离检查PASS。
- 当前方法仍只是可训练候选，科学性能未确立。`NEXT`：冻结三seed训练Data Card/spec；C01--C06梯度、C07选择、C08一次零适配，禁止C09/C10/M-TARE/graph。
- 证据：count corrective seal SHA=`d446569670c5aca364d79ecd0db9b3715548bd2dc8f975a7eabbe62975d69a0d`；geometry corrective run=`results/gate3_semantics/gate3_20260829_gse_sparse_relation_geometry_shape_corrective_v1_seed0`。

## 2026-08-29：完整Teacher关系召回纠正PASS，V2R2训练运行中

- `SYSTEM FAIL V2`：seed0两轮实测暴露整个三seedrun的4小时外层时限不足；在checkpoint/C08/科学结论前主动停止，13项seal SHA=`83e901ef0f0380c9aefa477912c00c62c041a278dd6707c85d458e41a3f68b72`。
- `SYSTEM FAIL V2R`：训练前只读评分器审查发现遗漏关系端点没有进入FN，可能虚高relation recall；零epoch/零checkpoint停止并封存。
- `CORRECTIVE PASS`：80 worlds、188,126 observations、三split有效pair=`448152/66752/77490`；完整Teacher persistent/reveal/withdraw人口全部独立计数，missing-endpoint示例由错误recall 1.0纠正为0.4/3 FN；21 unit +8 formal checks PASS。
- `EVIDENCE`：corrective run=`results/gate3_semantics/gate3_20260829_gse_sparse_relation_objective_recall_corrective_v1_seed0`；18项seal SHA=`24551f44cf15fa7cf1f3c9a7a334a1bd4d84bd552fe45246b759f49bf99482fb`；图SHA=`c07a287442bd72e07132df8117afdf0b64e1299df7158416282fe06534bbab27`。
- `RUNNING`：V2R2三seed正式训练；只修正8小时外层时限和完整Teacher召回分母，模型/数据/loss/seed/10 epochs/阈值不变。完成前不得进入graph/planner。

## 2026-08-29：Sparse Relation验证显存corrective PASS，准备V2R4

- `SYSTEM HISTORY`：V2R2在训练前因structural-refusal口径停止；V2R3在seed0首轮验证被资源监控停止，统一三seed科学结果仍为0。
- `REJECTED`：batch128虽降低显存，但validation loss最大改变`0.0190654`，会改变checkpoint选择，禁止采用。
- `PASS`：保持batch256，按C07 world释放unused CUDA cache；10/10 worlds、21,548/21,548 observations完成，输出/loss误差均为0。
- `RESOURCE`：最大allocated/reserved/process=`4.286/9.057/9.709 GiB`，均低于16 GiB；25/25 tests PASS。
- `EVIDENCE`：run=`results/gate3_semantics/gate3_20260829_gse_sparse_relation_evaluation_cache_resource_corrective_v1r_seed0`，21项seal SHA=`9aac7577efa75b235364fc738b26a3a24e958f6d2e0f4f65380f733b246f13f0`；PNG/PDF/SVG论文候选图保留。
- `NEXT`：冻结并执行V2R4三个seed；方法性能仍未确立，graph/planner/C09/C10/M-TARE继续为0。

## 2026-08-29：V2R4按16 GiB硬门停止，训练world corrective准备完成

- `SYSTEM FAIL`：seed0完成epoch0--2后，训练阶段process memory观测到`16,386 MiB`，超16 GiB两MiB；立即停止，seed1/2和selection为0。
- `TREND ONLY`：C07 event macro-F1到`0.6119`、axis到`6.391deg`、count到`0.9579`，turn/transition F1=`0.2416/0.1164`；不得作科学结论。
- `SEAL`：V2R4 13项seal SHA=`d83d3c436346a911e6d7e30cc9e9ee84d7b2a4a970a9331ee987525ddbb761e6`。
- `READY`：训练world cache corrective代码、Data Card、26项tests已PASS；下一步正式preflight并执行seed0单epoch权重/指标/显存parity。

## 2026-08-29：训练world cache corrective正式PASS，V2R5准备完成

- `EXACT`：seed0 epoch0的77/77模型tensor与sealed V2R3逐元素相同；全部train/C07指标最大误差=0。
- `RESOURCE`：60 worlds最大allocated/reserved/process=`7.383/11.363/12.017 GiB`，120次边界调用完整。
- `EVIDENCE`：run=`results/gate3_semantics/gate3_20260829_gse_sparse_relation_training_cache_resource_corrective_v1_seed0`；20项seal SHA=`ee6e2ce6c0b74f75795c0a6cd84cee134c465c39d988ab9e9f5f5d3f69329f93`；图三格式保留。
- `READY`：正式trainer/runner加入process-memory硬检查，27项tests PASS。下一步V2R5三个seed从头训练；方法性能仍未知。

## 2026-08-29：V2R5运行中发现轴语义与主方法接口缺口

- `RUNNING`：V2R5继续作为learned exit-token + persistent/reveal/withdraw transport基线完成三个seed；当前不得进入graph。
- `CODE EVIDENCE`：`local_axis`只被预测和单独监督，没有参与token坐标化、continuation/side-branch结构关系或节点触发。
- `DATA EVIDENCE`：fit/C07/C08=`142184/21548/24394`行的水平axis几乎恒为robot-forward，最大水平偏差约`7.4e-14deg`。
- `BASELINE`：恒定`[1,0,0]`不看LiDAR的平均axis误差=`2.79198/3.26039/3.33944deg`，优于当前模型曲线；旧`<=10deg`门不能证明中心轴学习。
- `DECISION`：V2R5只可成为论文基线/消融；完成后先冻结真正route-frame结构关系接口，再决定主方法训练，禁止旧selection直接开放graph。

## 2026-08-29：确认当前图由独立事件头触发，而非由几何语义组成

- `MODEL`：当前event logits仅来自global context；token count/bearing/geometry、transport和global metric geometry均未进入event head。
- `GRAPH`：结构节点由`observation.event`触发；几何字段主要被记录为trace/edge属性，未决定当前节点类型。
- `CONCLUSION`：V2R5属于多任务出口/事件/关系基线，不满足论文核心因果链。
- `NEXT AFTER V2R5`：冻结Geometry-Relation Event Composer，让五类事件由route-frame token集合、时序关系和几何变化组成；独立event head作为消融，PASS前不建主方法图。
- `DESIGN`：可执行接口、禁止输入、冻结backbone训练策略、六个对照和感知门已写入`docs/GSE_GEOMETRY_RELATION_EVENT_COMPOSER_PROPOSAL_V1.md`；状态仅为design，V2R5 seal前不执行。
- `TEACHER RISK`：turn使用前后局部spline，geometry-transition使用前后各5 m mesh宽高差；完整地图Teacher合法但可能与past-only显式输入时序不匹配。Composer训练前新增C01--C06零训练可观测性门，禁止用future/identity修复。

## 2026-08-29：双Composer候选方法规范冻结，但尚未科学确立

- `TEACHER AUDIT`：V2R5的独立event head仍使用旧transition mask，fit为12,534帧；corrected causal Teacher为fit/C07/C08=`791/67/173`帧、`59/5/12` identities。
- `METHOD SPEC`：新增`docs/GSE_GRAPH_METHOD_SPEC_V1.md`。junction/terminal由Action-Set Relation Composer产生；turn/geometry-transition由Metric-Change Composer及因果回投产生；descriptor只用关联，边只由真实穿越建立。
- `STATUS`：这是已冻结的可实施候选，不是`METHOD_ESTABLISHED`。只有corrected-Teacher可观测性、readiness、三seed增益和消融通过才能确立。
- `RUNNING`：V2R5 seed0已完成，seed1运行至epoch2，seed2未启动；无异常，约11.2 GiB GPU进程显存，低于16 GiB硬门。
- `NEXT`：V2R5封存后仅对显式token/transport/metric geometry与corrected Teacher做零训练容量/冲突审计；不读C09/C10，不建图。

### 单seed非正式接口诊断

- 新增纯评估模块`src/mtare_topo/evaluation/gse_composer_observability.py`，只接收count probability、token bearing/opening/profile/geometry uncertainty、transport/reveal和因果metric geometry；显式排除hidden context、event logits、descriptor、identity和pose。8项单元测试PASS，真实C07 archive对齐PASS。
- 只使用V2R5 seed0已有C07/C08 archive的非正式、无训练AUC诊断为：junction=`0.9821/0.9673`，terminal=`0.9862/0.9954`，turn=`0.6940/0.7233`，加入token opening/profile后geometry-transition=`0.5878/0.5927`。
- 结论仅限于：action-set决策有强显式信号，turn有中等信号，简单metric-delta不足以支持change-point。该结果不是三seed formal PASS，不允许据此定阈值或进图。
- 正式审计必须使用全量token-set/transport而非均值摘要；若三seed显式全集仍不能恢复corrected change-point，停止Metric-Change Composer。
- 进一步将全部显式状态规范化为642维（5帧count、6个token的bearing/width/profile/uncertainty、排序后完整transport/reveal、因果metric history/mask），用C07固定线性探针拟合、C08评价。seed0 C08 AUC/AP：junction=`0.9479/0.8788`，terminal=`0.9955/0.9265`，turn=`0.7338/0.0264`，geometry-transition=`0.5744/0.0198`。
- 这进一步证明当前显式状态对action-set强、对change-point弱，但仍为单seed非正式诊断。不能据此提前冻结或否决Metric-Change；必须等seed1/2封存后使用同一审计。

## 2026-08-29：创新复核确认最终方法仍未确立，并收窄唯一可行主张

- Cano原文已覆盖3D LiDAR出口角、出口跟踪、intersection/tunnel/dead-end判断与轻量拓扑图；PRISM-TopoMap覆盖学习式在线地点关联；Semantic Topometric Mapping覆盖结构语义探索；2026 Sequential Probabilistic Descriptor覆盖时序不确定描述子与高风险匹配过滤。
- 因此Action-Set、descriptor或uncertainty refusal单独通过都只能成为基线/组件。GSE-Graph必须证明“因果LiDAR学习路线连续度量几何 + 可执行出口关系 -> typed event -> 拒绝关联 -> traversal-verified edge”的完整因果链。
- `Metric-Change Composer`现为方法必要条件。当前seed0的完整显式状态对corrected geometry-transition仍弱，正式结论继续等待V2R5三seed封存。
- V2R5 seed1已完成epoch0--5共6/10轮，训练进程正常，约10.8 GiB GPU显存，低于16 GiB门；seed2尚未启动。
- `NEXT`不变：完成并封存V2R5三seed，然后创建corrected-Teacher正式显式可观测性审计。若三seedchange-point仍失败，先做RouteGeometryProfile可见性/Teacher proof，不建图、不读C09/C10。

## 2026-08-29：Composer正式可观测性审计实现就绪，等待V2R5 seal

- 新增确定性显式状态评估：完整token/count/opening/profile/uncertainty/transport/reveal和五帧metric geometry规范化为642维；hidden context、event logits和descriptor无法进入主探针。
- 新增C07-only阈值选择与C08原样transfer指标，含ROC-AUC、AP、precision/recall/FPR和结构identity coverage；tie score不能按标签拆分。10/10 unit PASS。
- 正式evaluator、runner和post-seal freezer已实现并compile PASS。真实seed0的20个C07/C08 archive只读smoke精确对齐45,942行、20 worlds和全部corrected event counts，矩阵=`45,942×642 float32`。
- C07 transition仅5 identities、C08为12，故正式审计使用阈值无关AUC/AP和identity coverage，不在此冻结部署阈值。12个固定线性probe（3 seeds × 4 events）只作C07→C08可观测性诊断，主模型optimizer/checkpoint update仍为0。
- V2R5 seed1已完成epoch0--7共8/10轮；当前进程正常、GPU约9.5 GiB，seed2待启动。三seedseal前不创建正式审计run。

### readiness metric corrective

- seed0四probe端到端诊断约59秒、峰值RSS约0.874 GiB，程序完整PASS。C08 explicit AUC为junction/terminal/turn/transition=`0.9490/0.9966/0.7422/0.6577`；transition AP=`0.01483`、identity coverage在C07选出的98% precision阈值下为`0/12`。这些仍是单seed非正式结果。
- 诊断发现草案中的`C07 fit AUC→C08 AUC gap<=0.10`无效：probe本身在C07拟合，训练AUC自然接近1，gap混合过拟合与迁移。正式spec尚未freeze，已在此前移除该PASS项并保留完整gap诊断。
- 正式门仍要求untouched C08 AUC、AP/prevalence和至少2/3 seed一致性；identity coverage完整报告，后续方法的98% precision/recall门不变。
- ensemble接口进一步纠正为每seed独立probe后对event score等权平均，不平均跨seed的642维token状态；避免slot对应差异产生伪feature。正式card/spec仍未freeze。
# 2026-08-29：ERCSS 相对配准与 Teacher 基础接口完成

- 已实现五帧 LiDAR 到当前机器人坐标的相对配准，返回值不保存绝对 pose/world/traversal/identity；全局平移旋转不变、禁止未来帧、确定性体素去重均有测试。
- 已实现物理 edge 独立采样、native-mesh 因果可见性、不可见间隙隔离、ego-connected 分量和类型化骨架目标。共享图节点按 node identity 连接，同 tunnel 的不同 edge 不折叠。
- 13项相关单元测试全部通过。当前尚未生成全量 Teacher、尚未训练或推理；科学可行性仍未确立。
- NEXT唯一允许步骤：冻结 Data Card/run spec，preflight 后执行一次不可覆盖的 C01--C08 零训练 Teacher/表示可行性审计；C09/C10/M-TARE/graph/planner 继续为0。
# 2026-08-30：ERCSS 全量审计尚未开始；三次 pre-science 系统停止后等待治理决定

- V1在核对320个C01--C08 mesh文档后，因Torch环境缺少Open3D在世界读取前停止；V1R切到冻结Open3D sidecar后，因旧representation包初始化隐式导入Torch而在世界读取前停止。
- 已把representation公开Torch API改为惰性加载；Open3D sidecar无Torch导入合同和Torch环境公开API均通过，ERCSS专项单元测试`10/10`通过。
- V1R2冻结器错误继承了修改前的单元测试SHA，runner在第一个工具哈希检查处停止；0 mesh核对、0 world、0 Teacher、0训练/推理/C09/C10/M-TARE/graph。没有形成ERCSS科学结论。
- 按预注册最终停止规则，不自动创建V1R3。当前NEXT需要明确治理决定：推荐仅刷新全部工具SHA并允许一个V1R3，科学Data Card、人口、Teacher、阈值、覆盖率和容量合同不得变化；否则停止ERCSS候选。
# 2026-08-30：V1R3纠正已预备但未获授权、未冻结、未执行

- 新增V1R3冻结器和runner草案；实际冻结必须显式提供`USER_EXPLICITLY_ALLOWED_ERCSS_V1R3`，否则立即拒绝。
- `--verify-only`自检PASS且写入0文件，确认V1R2真正阻塞的测试SHA从`408e5000...`变为当前`dfcd72c...`；V1R3将从每个当前工具路径重新计算全部12项SHA，不继承旧digest。
- 当前不存在V1R3 Data Card、run spec或run目录，未读取数据。正式执行仍等待用户明确覆盖V1R2停止规则。

# 2026-09-02：可观测基元关系TF32-off严格C07正式FAIL，进入关系失败归因

- `DONE`：相同三个冻结checkpoint在训练一致的TF32-off合同下完成`387,864`次C07前向；每seed=`64,644`行×2遍，0 optimizer、0 C08/C09/C10/graph/M-TARE。
- `RESULT`：surface改善=`56.376%/49.189%/52.600%`，geometry macro改善=`20.514%/38.906%/29.045%`；attachment precision=`5.654%/3.155%/3.230%`，安全连接TP=`0/0/0`，passing seeds=`0/3`。
- `SYSTEM CORRECTIVE`：源run在完成科学输出和28项seal后因`len(int)`后置报错；新的零推理证据run验证28/28源hash并形成独立28项seal，状态COMPLETED、error=null。
- `FAILED`：当前直接端口关系头无法产生precision≥0.98且非零的物理连接；不得进入C08、在线图或M-TARE。
- `NEXT`：只在密封C07/checkpoint上做正式失败归因，分别测候选存在性、双端点证据、pair组合空间、oracle候选下关系能力和置信度校准。归因完成前不设计新模型、不改变Teacher/阈值。
- `EVIDENCE`：`results/gate3_semantics/gate3_20260902_primitive_relation_observable_c07_tf32_evidence_corrective_v1_seed0`；seal SHA-256=`ea5b0833c1ba2c1e03e70e3ede38dc85eae38c23e08d78d44a78c915265c698b`。

# 2026-09-02：直接pair关系头归因完成，下一步连接超边Teacher可行性

- `DONE`：3 seeds×64,644 C07 rows=`193,932`次冻结前向；25项tests、formal attachment/safe逐项复现、TF32-off、资源和隔离全部PASS。
- `MECHANISM`：部署平均槽位=`16.14/29.01/23.27`且冗余严重；proposal oracle F1=`0.2215/0.1725/0.2075`，但raw/best-link安全TP三seed均0；端点证据oracle F1≈`0.89188`。
- `FAILED`：当前独立endpoint-pair relation score即使在正确候选下也无法安全排序，正式决定停止该head；不允许重加权、降阈值或手写规则绕过。
- `NEXT`：零训练审计`Primitive Connection Hypergraph` Teacher：C01--C06 fit和C07 selection全部observable attachment是否形成无冲突连接簇、每端点唯一归属、裁剪后clique闭合、fit-only容量和C07 overflow。0新模型推理、0 C08/C09/C10/graph/M-TARE。
- `EVIDENCE`：`results/gate3_semantics/gate3_20260902_primitive_relation_observable_failure_attribution_v1_seed0`；27项seal SHA-256=`fef3891cad80602437d8c45afd9d00c8505b0921c8824984f78c278cf7d6cf76`。

# 2026-09-02：连接簇Teacher PASS、稠密K槽decoder科学FAIL，composition-anchor readiness下一步

- `DONE`：C01--C06/C07共70 parents、210配对shards、491,196行零训练审计；全部连接关系为对称clique，每端点唯一归属，C07可观测正连接442,936精确复现。
- `CAPACITY`：fit最大19簇，25%余量要求24，冻结K=32；C07最大20，fit/C07 overflow均0。
- `FAILED`：K=32固定cluster输出1.006B，高于pair输出0.975B；active cluster候选256.05M约为pair 64.69M的3.96倍，未实现预注册组合压缩。
- `DECISION`：停止稠密global-slot hypergraph decoder，不改余量、不挑K=16；保留已证明的连接簇Teacher语义。
- `NEXT`：零训练`Learned Composition Anchor Field` Teacher/readiness：每端点唯一共享锚点、当前传感器坐标因果变换、同簇一致、异簇/stacked可分、fit-only校准边界和O(E)输出；0 C08/C09/C10/model/graph/M-TARE。
- `EVIDENCE`：`results/gate3_semantics/gate3_20260902_primitive_connection_hypergraph_teacher_feasibility_v1_seed0`；23项seal SHA-256=`61e6ce19bbcae565f2b2b0d0f678917acf6ee5cc1388c5780d428c79cdec3e87`。

# 2026-09-03：composition-anchor Teacher readiness PASS，预测几何必要性诊断NEXT

- `PASS`：70 parents/210 tasks/491,196 fit+C07 rows；每端点唯一anchor，同簇距离0，异簇与overlap hard negative最小0.809503 m。
- `BASELINE`：fit-only 0.499973 m阈值迁移C07得到precision1、recall0.999255、F1 0.999627，TP442,606/FP0/FN330，hard-negative FP0。
- `COMPLEXITY`：每行256个anchor xyz+uncertainty输出，对比1,984 pair scores缩减7.75倍。
- `RESOURCE/ISOLATION`：77.11s，peak RSS 2,508,700 KiB；0 range/model/optimizer/C08/C09/C10/graph/M-TARE，source unchanged。
- `RISK`：Teacher真值几何的距离基线近满分，额外relation head是否必要尚未证明。
- `NEXT`：三个冻结checkpoint C07-only predicted-axis association diagnostic；deployed与proposal-oracle分别比较几何距离和learned pair safe selection。0 optimizer/C08/graph/M-TARE。
- `EVIDENCE`：`results/gate3_semantics/gate3_20260903_primitive_composition_anchor_teacher_readiness_v1_seed0`；27项seal SHA-256=`4e48e809d5462cd98fb9b811746a7dbd0b831b06f06fa80f789aee1d4b17e1d8`。

# 2026-09-03：预测端点几何诊断完成，O(E) anchor residual/uncertainty readiness获准

- `DONE`：三seed×64,644 C07 rows=`193,932`次冻结前向；13项tests、旧learned-pair指标复现、TF32-off、资源和隔离全部PASS。
- `RESULT`：proposal-oracle predicted-geometry best F1=`0.289743/0.282983/0.293032`，deployed best F1=`0.041448/0.019698/0.025185`；两种掩码在precision≥0.98时三个seed安全TP全部为0。
- `MECHANISM`：Teacher-fit `0.499973 m`阈值用于proposal-oracle预测端点后precision仅`0.222877/0.203807/0.230984`，说明真实组合锚点可分，但预测轴端点误差和错误聚集使距离无法安全关联；候选正确也不能解决。
- `DECISION`：正式停止直接预测端点距离构图；按预注册分叉开放每端点`O(E)`共享composition-anchor残差+不确定性head readiness。继续禁止独立pair分类、稠密K槽、手写连接规则和C08读取。
- `NEXT`：实现typed anchor residual/uncertainty输出、概率兼容性loss与零训练readiness，验证等变/置换/反向、正负梯度、真实batch有限反传和identity-free forward；PASS前不训练、不建图。
- `EVIDENCE`：`results/gate3_semantics/gate3_20260903_primitive_predicted_geometry_association_diagnostic_v1_seed0`；28项seal SHA-256=`9eda771e22bdf9dbb886736c003c3a932c67b6a107e2fe390213764f7147919b`。

# 2026-09-03：axis-frame readiness科学FAIL，sensor-polar O(E) anchor V2 PASS

- `V1/V1R`：V1在真实冗余query的退化轴上主动失败；V1R修正数学计算完成后仅图源`.int()`后置失败。两次保持0训练/C07/C08并封存。
- `V1R2 SCIENTIFIC FAIL`：真实行有12个<0.1 mm退化轴端点，其中1个匹配且观测；axis-frame query/yaw contracts失败，证明预测轴不能作为总定义修正坐标。seal SHA-256=`3095f6ff7f511cd02eb134bc81359029b6d8720a0a36b42c5eb5076b78dc116b`。
- `V2 METHOD`：使用current-sensor polar radial/lateral/gravity frame，feature不再读取tangent/segment/bend；Gaussian compatibility温度低斜率初始化但保持可学习。无候选删除、身份输入、事件规则或阈值放宽。
- `V2 PASS`：20/20 tests、17/17 checks；真实行8 positive/6 overlap-hard-negative，观测degenerate-axis=1但radial-degenerate=0；22,278 trainable params、256 outputs、7.75x reduction、全部新梯度有限非零、backbone梯度0。
- `EQUIVARIANCE`：incremental query error=`1.5259e-5`，yaw anchor/scale/compatibility=`3.8147e-6/5.9605e-8/9.5367e-6`，继承上游absolute anchor=`4.1008e-5 m`，全部过门。
- `NEXT`：物化fit/C07 composition-anchor target sidecar；0 model/optimizer/C08/graph/M-TARE。PASS后才允许head-only三seed训练Data Card/spec。
- `EVIDENCE`：`results/gate3_semantics/gate3_20260903_primitive_composition_anchor_model_readiness_v2_seed0`；17项seal SHA-256=`3fc29fb93c7f332c619192daae595239e8b523f8f2ee68c36413f1adbe6a907d`。
