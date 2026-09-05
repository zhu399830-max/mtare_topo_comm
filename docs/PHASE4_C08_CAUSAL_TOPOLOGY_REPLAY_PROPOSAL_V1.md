# Phase 4 C08 Causal Topology Replay Proposal V1

状态：`PROPOSAL_AND_DATA_CARD_DRAFT_AWAITING_USER_APPROVAL`  
日期：2026-08-13  
几何前置：`PASS_CANO_C08_TRAJECTORY_MESH_CONTRACT_V1`

## 1. 本轮只回答什么

冻结 M1D 的局部 direction/count/role/z_role 输出，能否在三个 C08 连续因果轨迹上形成
稳定 structural node、实际通过后才确认的 edge，以及仍未通过的 exit stub，并据此冻结一组
仅供后续 C09 使用的图参数？

本轮不控制机器人、不选择探索目标、不修改 M-TARE、不读 C09/C10。

## 2. 精确数据卡

| world | GT nodes/edges | trajectory | frames @ 2m | structural events |
|---|---:|---:|---:|---:|
| S01_flat_tree_small_C08 | 54/54 | 1601.266 m | 801 | 4 junction + 5 terminal |
| S06_3d_branch_medium_C08 | 92/92 | 2826.012 m | 1414 | 7 junction + 7 terminal |
| S10_3d_complex_C08 | 164/166 | 5114.364 m | 2558 | 14 junction + 10 terminal |
| total | 310/312 | 9541.643 m | 4773 | 25 junction + 22 terminal |

独立采样单位是 3 个 topology parent 和 3 条完整 trajectory，不把 4773 个相邻帧称为独立
world/place。每条 GT edge 恰好正反各通过一次。LiDAR 固定为 16×720、0.3--50m、无噪声；
student 输入只有 normalized range + valid mask。pose 只用于世界坐标锚定和实际行驶距离。

CPU/Open3D 生成 primary 与独立 replay scene，共 `109,969,920` 条射线；只保存一份通过
determinism 审计的 range/valid。未压缩 range+mask 约 0.256 GiB，整个结果预计不超过 1 GiB。

## 3. 固定三阶段环境接口

1. E1 Python 3.12 / Open3D 0.19：按已封存 trajectory 生成 4773 帧 LiDAR、objective
   direction/role/count oracle 和双 scene 一致性证据；
2. PyTorch 2.9.0+cu129：只读 range+valid，分别加载 seed0/1/2 best checkpoint，保存每帧
   direction/count/role logits 和 128D z_role；不训练、不选择 checkpoint；
3. CPU 图阶段：B0、每个 M1D seed、GT oracle 都输入同一个因果 graph builder；完整 GT
   只在 oracle 和离线评价中出现，不进入 B0/M1D 或主图更新。

三个 best checkpoint SHA-256 固定为：

- seed0 `55f6602fb7fd74e3a44c3a4697a7c4be4d31469f46348f631dc606612125f9fe`；
- seed1 `3822d27af6d5fb0ef920da3e310927d42c5a112c5d671b4cbdc81d9db251f8df`；
- seed2 `20b1e9c8198b06473b68e31f4d8a3eb50eb5f2fc1b7aa577aa1aac92ba35346e`。

## 4. 方法、baseline、oracle

主方法每帧只使用当前语义输出、当前 pose、累计行程和历史图状态：方向 component 以冻结
probability 0.5 解码；role/count 用 argmax；结构事件需持续稳定；相邻结构节点之间超过固定
距离时加入 distance anchor；edge 只在物理轨迹已经从一个节点走到另一个节点后建立，长度
记录累计行程而非节点直线距离。回环关联必须同时满足空间、role 和 heading-set；z_role 只作
一致性审计/拒绝明显冲突，不单独触发地点合并。

- B0：冻结 `RangeExitBaseline`，同轨迹、同 graph builder；
- M1D：三个 checkpoint 各自完整 replay，报告中位数和最差 seed/world；
- Oracle：GT direction/role/count 输入同 graph builder，检验图接口上界。

旧 `OnlineTopometricGraph` 只复用 schema 思想，不直接作为正式实现：它用节点欧氏距离代替累计
行程、不能正确表达多条物理路径，也没有显式 confidence/role logits/distance anchor，因此必须
先写 V2 causal builder 和 contract test，旧 240-frame sweep 参数不得直接继承。

## 5. 参数开发与停止规则

图参数只在三个 C08 上从预声明有限网格选择：stable frames `{1,2,3}`、minimum event travel
`{4,6,8}m`、loop radius `{3,4,6}m`、heading-set tolerance `{20,25,35}deg`、turn event
`{25,35,45}deg`。distance-anchor 间隔先固定 `20m`，不与结构事件竞争。候选共 243 组。

选择先要求 oracle 无假 verified edge、保持一个 GT connected component、每条输出 edge 都有
对应实际行驶 trace；不满足则整轮停止。通过完整性后，以 exit preservation、connectivity、
node redundancy、path distortion 和 temporal churn 的预声明归一化综合分排序，并用较少节点、
较大稳定帧、较接近旧候选中心作确定性 tie-break。C08 结果只冻结参数，不宣称 unseen 泛化。

必须停止并报告：任一 raycast/teacher/replay 不一致；checkpoint 或阈值漂移；graph builder 使用
未来帧/GT；oracle 接口失败；verified edge 未有物理 trace；需要修改数据、模型、teacher、指标
或候选网格。不得在 run 内修补。

## 6. 成本、输出与本轮 PASS 含义

- 预计 CPU raycast 5--15 分钟，三 seed GPU inference 5--15 分钟，图 sweep 5--20 分钟；
- 预算 `0.75 h / 1 GiB / RTX 5090 + CPU`；
- 保存 trajectory、range/valid manifest、双 scene replay 指标、逐帧 oracle/B0/M1D outputs、
  每个候选图指标、冻结参数、最终 nodes/edges/stubs、decision trace、path metrics、失败原因、
  三张完整地图叠图、churn 曲线、Pareto 图、raw logs、environment 和 seal；
- run PASS 只表示执行完整且 oracle/因果/证据合同通过，并能冻结 C08 参数；Gate 4 仍保持
  `GATE_MIXED`，C09 未执行，不自动进入 M-TARE。

当前等待用户批准：允许实现 V2 causal builder，并按上述 Data Card 运行一次 4773-frame C08
development replay。批准不包含 C09/C10、规划器、closed-loop 或 M-TARE。
