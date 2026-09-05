# 基元关系结构图论文完成矩阵 V1

状态：`ACTIVE_NOT_COMPLETE`
主方法：程序构造监督的五帧几何基元关系模型与执行验证拓扑图。

| 论文证据 | 当前状态 | 已有权威证据 | 完成条件 |
|---|---|---|---|
| 创新与方法边界 | 已冻结，投稿前需复核文献 | `docs/PRIMITIVE_RELATION_STRUCTURAL_GRAPH_PLAN_V1.md` | 相关工作矩阵逐项对应缺口、差异、消融和闭环收益 |
| 三形状程序数据 | PASS | 80父世界、240任务、757,290帧、564,378个五帧序列 | 已完成；严格测试和真实数据仍隔离 |
| 构造Teacher | PASS | 24,117基元、8,779,620连接、3,023,794断开重叠、538,391 dustbin | 已完成并密封 |
| 同输入非学习基线 | PASS | 原完整基元基线C07 primitive F1 0.2983、coverage 0.1753；当前双端可观测同输入attachment基线F1=`0.006380`，64,644行/442,936正例已密封 | 已完成并保留PNG/PDF/SVG及机器可读summary |
| 三种子学习模型 | COMPOSITION_ANCHOR_TRAINING_RUNNING | 旧直接pair-relation三seed在合规TF32-off C07为0/3且安全TP全0，已封存为失败消融；归因证明端点证据oracle F1约0.8919，但pair排序失败。新sensor-polar `O(E)`共享composition-anchor头为22,278参数，Teacher sidecar/readiness均PASS；当前唯一三seedrun中seed0已完成3轮/10,233步、选择epoch2，seed1运行中，C08为0 | seed0/1/2全部完成后由冻结评估器检查attachment F1增益>=0.05、precision>=0.98且TP>0、anchor MAE改善>=10%、骨干冻结；至少2/3同时通过才允许C08 |
| C08零适配迁移 | NOT_OPENED_CURRENT_COMPOSITION_ANCHOR_RUN | 当前正式run的C08 rows read=0，数据隔离和54项输入/工具哈希复核PASS | 仅当前方法在C07达到2/3后读取一次；若C07 FAIL则保持关闭并进入冻结失败归因 |
| 离线结构语义图 | LOCKED | 旧规则图只作基线/失败分析 | 感知通过后完成同轨迹因果回放、图指标和错误回环审计 |
| 严格未见与真实离线 | LOCKED | C10和正式测试禁止读取 | 模型、阈值和图参数冻结后一次执行；真实site按完整地点隔离 |
| 单机器人闭环 | PENDING | M-TARE接口和历史基线可复用 | 同世界/seed/传感器/局部规划合同下完成公平配对 |
| 2/3/4机器人闭环 | PENDING | 图增量共享与分配接口待主方法资格 | 完成效率、冗余、冲突、通信与失败统计 |
| 消融 | PENDING | 槽位容量和旧科学失败已有可复用证据 | 完成单帧、监督、重建、关系、时序、拒绝、edge几何等单变量实验 |
| 论文正文 | 骨架已建立 | `docs/PRIMITIVE_RELATION_STRUCTURAL_GRAPH_MANUSCRIPT_DRAFT_V1.md` | 全部AUTO占位符绑定sealed结果 |
| 图表与复现包 | PARTIAL | 当前主方法图、数据/Teacher地图图、同输入非学习C07基线图、V1三seed失败对比图及V1候选/关系归因图均已完成并密封；10组历史失败图已保留 | 主表、预测拓扑图、曲线、消融、失败案例和其余manifest齐全 |
| 投稿PDF | PENDING | 无 | 正文、补充材料、引用、匿名化与最终证据审计全部通过 |

旧工作角色由`docs/PAPER_EVIDENCE_RETENTION_MATRIX.md`控制。旧数字只有在输入与执行合同一致时才能进入主表；否则只用于方法动机、消融或失败分析。

## 完整故事验收合同

论文必须用同一条可审计因果链连接构造Teacher、基元关系学习、在线结构图、真实穿越边和单/多机器人探索。单个组件PASS、工程跑通或一组负结果均不能替代完整方法结论。当前模型若科学FAIL，保留为关键消融并完成归因，但论文目标保持未完成；只有证据驱动的后续主方法重新通过感知、图和闭环门，才允许填充正文结论和投稿PDF。
