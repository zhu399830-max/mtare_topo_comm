# 失败知识库

本文件把旧项目已经暴露的问题变成 V3 的硬检查，避免再通过“补网络、loss、阈值”重复试错。

## F01：帧数被误当作数据规模

- 现象：有限 world 和轨迹被切成大量相邻 `.npz`，表面样本数较多，独立结构与空间覆盖仍小。
- 影响：随机切帧会产生高度相关的 train/validation，指标虚高，无法证明跨 world 表示能力。
- V3 防线：data card 同时报 world、独立轨迹、移动距离、空间覆盖、有效结构事件和采样间距；按 world/trajectory 切分。

## F02：训练/验证场景与研究目标错位

- 现象：forest、campus、generic indoor 等结果混入地下结构语义论证。
- 影响：只能说明某些输入或代码能运行，不能支持地下 tunnel/cave/mine 的 H1--H3。
- V3 防线：Gate 0 先冻结地下 dev/test benchmark；非地下数据若使用，必须单列辅助目的，不能代替地下验证。

## F03：所谓 unseen 不等于严格 test

- 现象：历史开发已经查看或调试过多个 `unseen` world。
- 影响：这些 world 已被开发过程污染，不能承担最终无偏性能声明。
- V3 防线：冻结新的 world-disjoint strict test；禁止进入 SSL、统计、teacher/阈值标定、增强调节和 checkpoint 选择。

## F04：teacher 与部署 planner 不一致

- 现象：从静态几何或不完整 free-space 规则生成的方向/出口标签，不一定等于 local planner 实际可执行性。
- 影响：监督标签可以自洽但对规划无效，模型再准也不能改善闭环。
- V3 防线：Gate 1 主路线采用 planner-consistent teacher，保存 valid mask，并抽样做 teacher-planner consistency。

## F05：累积点云被当作 free/unknown 证据

- 现象：没有真实 ray origin 的 KeyedScan 可能被用于推断 free-space。
- 影响：遮挡后方和未观测区域会被错误标注，破坏出口与可通性语义。
- V3 防线：只有真实 ray origin 才启用 ray-based free/unknown；否则只构建 observed surface 和观测有效性。

## F06：结构角色、地点身份和节点分数混在一起

- 现象：embedding、canonical role、node score、topological event 被联合优化或互相代替。
- 影响：无法判断失败来自表示、语义还是节点规则，也容易学到 world identity。
- V3 防线：Gate 2 只验证稳定表示；Gate 3 分开 directional semantics、`z_role` 与未来的 `z_place`；Gate 4 先用规则节点 baseline。

## F07：闭环能跑不等于方法有效

- 现象：已有 shadow/closed-loop 中存在无运动、日志缺失、planner 拒绝或 baseline 条件不一致。
- 影响：coverage/path 对比无效，不能归因于拓扑方法。
- V3 防线：Gate 0 冻结接口和 INVALID 规则；Gate 5 先 shadow；Gate 6 才允许单机接管全局规划。

## F08：可视化替代统计证据

- 现象：挑选 topology 图或局部样例展示，但没有 split、样本 ID、单位和配套机器指标。
- 影响：无法复算，容易产生选择性展示。
- V3 防线：preview 必须有 provenance，关键结论依赖全量机器可读指标、分布和失败案例。

## F09：失败后静默改变问题

- 现象：通过换模型、加 loss、改阈值或换数据继续运行，研究问题和比较条件逐步漂移。
- 影响：实验之间不可比较，无法形成可信因果链路。
- V3 防线：每 Gate 一个主方法、一个 baseline、一个必要备选；发现缺陷立即停工反馈，变更写入决策日志并重新审批。

## 重新验证原则

旧资产能够缩短实现时间，但不能缩短证据链。每项复用必须回答：输入是否因果、坐标是否一致、数据是否获批、test 是否隔离、指标能否复算、它支持哪个 Gate 的哪条结论。
