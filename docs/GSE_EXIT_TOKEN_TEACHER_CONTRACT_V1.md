# GSE-Graph Exit Token Teacher Contract V1

状态：`IMPLEMENTATION_PROPOSAL_BEFORE_FORMAL_AUDIT`

## 目的

为每个五帧因果观测提供“当前 LiDAR 中客观可见的可通行出口集合”，使模型学习出口数量、方向、开口宽度、垂直剖面和跨重访 identity。完整 TNG、spline 和 mesh 只生成 teacher，不进入学生输入。

## 候选出口

- 普通 corridor：当前 physical edge 的两个方向各为一个候选；identity 分别为该 edge 的正/反有向端点。
- junction/terminal：使用结构事件判定所选的唯一最近 node，仅枚举该 node 的 incident physical edges；每条 edge 从 node 指向另一端形成一个候选。
- turn/geometry-transition：仍使用当前 physical edge 的前/后两个方向。
- 同 tunnel ID 的不同 physical edges 不合并；非 incident、stacked 或平行 tunnel 永不成为该结构节点的候选。
- 若一个观测同时落入两个 node 窗口，沿用已冻结事件教师的最近距离、node ID 次序，不另加参数。

## 每个 token 的客观量

- `identity`：`parent:edge:from_node->to_node`。
- `heading_robot_deg`：从当前 sensor pose 指向该出口方向的水平角，相对当前 traversal yaw。
- `opening_width_m`：沿候选 edge 离 node/当前点 2.5 m 处的 native-mesh route-frame 横截面宽度；不足2.5 m时夹到edge端点，但短于4 m且无因果样本的edge仍只作incident exit。
- `vertical_profile_m`：沿候选有向 spline 在2.5/5/7.5/10 m处相对起点的中心线高度变化。
- `confidence`不是 teacher 常数；模型置信度后续在validation上校准。

## 可见性与缺失

- 从当前 sealed sensor origin向2.5 m出口代表点做native-mesh first-return LOS；命中距离不得早于代表点超过既有0.25 m LOS margin。
- 学生 token 集只监督当前可见候选；完整 incident 集同时保存在audit字段，用于报告“结构已知但当前不可见”，禁止把不可见分支当模型漏检。
- opening width需左右扇区各至少3/5条有效射线；不完整时token保留identity/heading/vertical profile，width loss单独mask，不以解析radius回填。
- 若结构事件为junction/terminal但没有任何可见candidate，标记teacher-observability failure并停止正式export；不得删除该帧或降低可见性门槛。

## 正式审计要求

- 范围固定为80 train + 10 validation worlds、18,132 traversals、212,588 observations；C10和M-TARE零读取。
- 输出每帧incident/visible token数、identity、LOS、width mask、vertical profile和按event/split分布。
- 验证token identity唯一、heading有限、incident-only、正反edge对称、非incident stacked tunnel隔离和重复执行确定性。
- 在看到分布前不设置基于结果的过滤阈值；任何不可观测结构事件、identity冲突或非incident token均FAIL并报告。

## 与sensor export的顺序

出口token审计PASS并seal后，才允许执行唯一帧sensor export。export固定使用`src/mtare_topo/data/gse_sensor_export.py`重建poses并按32帧batch cast冻结16×720扫描；每个unique frame只存一次，sequence只存五个global int64 references。
