# Cano 100 Topology Parent Corrective Design V2

状态：`APPROVED_FOR_ONE_EXECUTION_IMPLEMENTED_PENDING_PREFLIGHT`  
日期：2026-08-11  
前置失败：`gate0_20260811_cano_100_topology_parent_candidate_audit_v1_seed0`

用户批准：2026-08-11T14:53:35+08:00。用户在收到 fresh 120 seeds、20 parameter draws、每 draw 100 Cano internal trials、exact cycle rank 和零 mesh/LiDAR/训练边界后连续回复“继续”。

## 1. 现在要解决的唯一问题

V1 在 10 个结构层中预声明并完整尝试了 120 个候选，但只有 60 个候选有效，无法形成 100 个 topology parent 和 80/10/10 split。60 个失败全部发生在 grown tunnel generation。

V1 对每条 requested tunnel 只随机抽一次生成参数，再让 Cano 在这组固定参数下做最多 100 次内部位置/方向试验。如果这组长度、曲率和方向参数在当前网络中不可行，该 candidate 直接失败。Cano 上游批量入口则会在失败后重新抽参数，直到成功；V1 比上游方法更苛刻。

V2 只修正这个生成合同，并修正已经发现的 family 判据缺口。它仍是 topology-only 审计，不生成 mesh、LiDAR 或标签，不训练模型。

## 2. 为什么不直接沿用 V1 的 60 张图

- V1 结果已经被人工和机器查看，不能再把其中候选包装成新的 unseen validation/test。
- V1 各层分布严重不齐，尤其 S09 只有 1/12；把 60 张旧图和新图拼接会使 split 规则依赖已经观察到的失败结果。
- 两个 V1 nominal-valid parent 不满足严格 family identity：S01 C07 在 0 connector 下 cycle rank=1，S07 C04 在 2 connector 下 cycle rank=3。

因此 V2 使用一套全新的、执行前固定的 120 个 candidate seeds。V1 只作为生成器诊断证据，不计入正式 parent split。

## 3. 冻结的候选结构层

| Stratum | 维度 | Grown | Connector | 目标 family | Candidates | Retain |
|---|---:|---:|---:|---|---:|---:|
| S01_flat_tree_small | flat | 4 | 0 | small tree | 12 | 10 |
| S02_3d_tree_small | 3D | 4 | 0 | small slope tree | 12 | 10 |
| S03_flat_unicyclic_small | flat | 4 | 1 | single loop | 12 | 10 |
| S04_3d_unicyclic_small | 3D | 4 | 1 | 3D single loop | 12 | 10 |
| S05_flat_branch_medium | flat | 6 | 1 | medium branch | 12 | 10 |
| S06_3d_branch_medium | 3D | 6 | 1 | medium slope branch | 12 | 10 |
| S07_flat_loop_rich | flat | 8 | 2 | loop rich | 12 | 10 |
| S08_3d_loop_rich | 3D | 8 | 2 | 3D loop rich | 12 | 10 |
| S09_flat_complex | flat | 10 | 3 | complex flat network | 12 | 10 |
| S10_3d_complex | 3D | 10 | 3 | complex multiheight network | 12 | 10 |

Candidate ID 仍为 `Sxx_<name>_C01...C12`。精确 topology seed 公式为 `620000 + 100 × stratum_index + candidate_index`，得到 `620101...621012`；reserved geometry seed 为相同索引的 `720101...721012`。公式和完整展开注册表必须在执行规格中冻结并在 preflight 检查 120/120 唯一，运行后不得追加或替换。

## 4. V2 生成方法

每个 candidate 仅初始化一次空 TunnelNetwork，并使用一个 candidate topology seed 控制确定性随机流。对每条 requested grown 或 connector：

1. 最多进行 20 次 parameter draw；
2. 每次 draw 重新抽取 Cano 原生 generation parameters；
3. 每次用这组参数调用一次原生 `add_random_grown_tunnel` 或 `add_random_connector_tunnel`；
4. 每次 API call 保持 Cano 的 100 次 internal placement trials；
5. 首次返回成功即保留该 tunnel 并进入下一条 requested tunnel；
6. 20 次 parameter draw 全失败则 candidate invalid；
7. 保存全部 draw 参数、API 返回、成功 draw index 和失败位置。

这属于有界、确定性的 rejection sampling：它模仿上游“失败后重抽参数”的意图，但把无界 `while` 改为可审计的 20 次上限。它不增加 candidate seed，也不根据结果搜索 seed。

Flat strata 除设置 Cano `TunnelNetworkParams.flat=true` 外，继续显式将 grown `vertical_tendency` 和 `vertical_noise` 归零，因为 Cano 的 `flat` 标志只约束 connector；第三方源码不修改。

## 5. 单 parent 验收

每个有效 parent 必须同时满足：

- requested grown 和 connector 数全部实际成功，所有返回已记录；
- graph 单连通、graph/spline 数值有限、schema 合法；
- `cycle_rank == requested_connector_count`，不再接受额外隐式环；
- 中心线总长至少 300 m；
- flat 高度跨度不超过 `1e-6 m`，3D 高度跨度至少 5 m；
- 同 seed 从空网络完整 replay 后 canonical graph/spline hash 完全一致；
- canonical identity 不与 P01--P05、V1 60 parents、M-TARE benchmark identities 或本批已接收 parent 相同。

## 6. 批次验收和 split

- 120 个候选全部执行并保留结果；
- 每层至少 10 个有效，按 candidate ID 顺序保留前 10 个；不足 10 则整个 V2 FAIL，不追加 seed；
- 每层 accepted rank 1--8 为 train，rank 9 为 validation，rank 10 为 development-test；总计 80/10/10；
- 100 个 canonical graph/spline identities 全部唯一；
- coordinate-free WL hash 总数至少 40，且每层至少 3；
- 至少 70/100 parent 含 degree>=3 结构事件；
- 50 个 3D parent 全部满足高度跨度；
- parent ID 和 canonical identity 在三个 split 间严格不交叉。

如果 V2 仍 FAIL，停止继续修改随机预算；转向重新设计 flat complex 的空间尺度/碰撞约束，或引入另一公开程序化生成器，并先提交新方案。

## 7. 可视化与泄漏控制

只渲染并人工检查 10 张完整 graph/spline 图：每层第一个 accepted train parent。Validation 和 development-test 只做机器检查，不人工挑图。图中必须显示完整 X-Y 和 X-Z、节点、边、family、长度、cycle rank、vertical span、seed 和 provenance。

所有未来 mesh、轨迹和 LiDAR descendants 必须继承 parent split。M-TARE benchmark worlds 不参与生成参数、阈值、归一化或 checkpoint 选择。

## 8. 明确不做的事

本次上限为 120 个 topology constructions 加至多 100 个 accepted same-seed replays。以下计数必须保持 0：mesh、anchor、LiDAR observation、ray、teacher label、正式 dataset sample、trajectory、model、training sample、M-TARE change。

预计 CPU wall time不超过 2 小时，证据目录不超过 2 GB。V2 PASS 只表示 100 个 topology parent 和 split 可以冻结；下一步 perception mesh 仍需另立方案并审批。
