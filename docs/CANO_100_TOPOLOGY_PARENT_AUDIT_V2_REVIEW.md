# Cano 100 Topology Parent Audit V2 Review

状态：`V2_SEALED_FAIL_V2R_RECLASSIFICATION_RECOVERY_PASS`  
日期：2026-08-11  
证据：`results/gate0_baseline/gate0_20260811_cano_100_topology_parent_candidate_audit_v2_bounded_resampling_seed0/`

## 结论

V2 按冻结的 exact-cycle family 合同 FAIL，但 bounded parameter resampling 本身成功解决了 V1 的生成产率问题。

- 120/120 candidates 完整生成成功；
- 120/120 same-seed replay 完全一致；
- 114/120 通过 exact-cycle 合同；
- 各层 valid/12 为 `11/12, 12/12, 12/12, 12/12, 12/12, 12/12, 9/12, 12/12, 10/12, 12/12`；
- S07 只有 9 个，因此只能保留 99 parents，split 为 80/10/9；
- 99/99 canonical identities 和 99/99 coordinate-free WL hashes 唯一；
- 391/391 evidence SHA-256 复核通过；
- 运行 1845.62 s，封存前 101,458,639 bytes；
- mesh、LiDAR、label、formal dataset、training、model、trajectory、M-TARE change 均为 0。

## 参数重抽的实际效果

120 个 candidate 共执行 936 条 requested-tunnel operations。834 条第一次 draw 就成功，102 条需要重抽；draw count 分布为：

| Draw count | Operations |
|---:|---:|
| 1 | 834 |
| 2 | 82 |
| 3 | 16 |
| 4 | 3 |
| 5 | 1 |

平均 draw count 为 1.1357，最大仅 5，从未触发上限 20。V1 的 60 个 generation failure 在 V2 中降为 0。S05 flat-medium 从 V1 的 3/12 提升到 12/12，S09 flat-complex 从 1/12 提升到生成成功 12/12。

因此不应继续增加 draw budget，也不应再换 seed。bounded resampling 已经充分工作。

## V2 为什么仍失败

6 个 candidate 的唯一失败理由都是 `exact_connector_cycle_rank`：

| Candidate | Requested connectors | Actual cycle rank |
|---|---:|---:|
| S01 C08 | 0 | 1 |
| S07 C02 | 2 | 3 |
| S07 C04 | 2 | 3 |
| S07 C07 | 2 | 3 |
| S09 C10 | 3 | 4 |
| S09 C11 | 3 | 4 |

Cano grown tunnel 可能接入已有的中间 intersection node，从而自然形成额外 graph cycle。Connector count 描述的是生成器调用配方，不是最终图的 cycle rank。`cycle_rank == requested_connector_count` 因而不是正确的 family 身份定义。

## 完整图人工复核

已逐张检查 10 张固定 rank-1 train-only X-Y/X-Z 图：

- 所有图可读、完整，无空白或断裂；
- S01/S03/S05/S07/S09 的 X-Z 全部 z=0；
- S02/S04/S06/S08/S10 有明确坡度和多高度；
- grown/connector 数增加时，中心线规模、结构事件和闭环复杂度总体增加；
- X-Y 曲线视觉上绕回不等于 graph cycle，必须以 graph metric 为准。

Validation 和 development-test 没有进行人工地图查看。

## 推荐的 V2R 只读恢复

不重新生成 topology，不增加 seed，不修改 sealed V2。对现有 120 个 V2 candidate 做一次新的 validation-only reclassification：

1. 将 strata 从 tree/unicyclic 等 family 名称改成配方名称，例如 `flat_g4_c0`、`3d_g8_c2`；
2. candidate 必须生成成功、replay 一致、请求 tunnel 数正确、单连通、有限、长度和高度合同通过；
3. 要求 `actual_cycle_rank >= requested_connector_count` 只作为生成一致性检查；
4. `actual_cycle_rank`、terminal count、degree>=3 count 等真实 graph properties 全部进入 metadata，后续用于分层统计，不作为本轮拒绝理由；
5. 每个 recipe stratum 固定取 C01--C10，形成 80 train/10 validation/10 development-test；
6. 不渲染新图，不打开 validation/development-test 地图，不生成 mesh/LiDAR/labels，不训练。

只读反事实已证明该规则会得到 100 parents、80/10/10、100 canonical identities、100 WL hashes、100 structural-event parents 和 50 个合格 3D parents。

## V2R 正式恢复结果

用户批准后，独立 V2R runner 已从 sealed V2 正式执行一次，只读生成新的 manifest、metrics 和 evidence hash。执行前 93/93 单元测试、11/11 冻结 E1 external、391/391 source seal 与 preflight 0/0 通过。

结果目录：`results/gate0_baseline/gate0_20260811_cano_100_topology_parent_recipe_reclassification_v2r_seed0/`。

- `RUN_STATE=COMPLETED`，overall status 为 `PASS_CANO_100_TOPOLOGY_PARENT_RECIPE_RECLASSIFICATION_V2R`；
- 120/120 candidate recipe-valid；
- 固定选择 100 parent，split 为 80/10/10；
- 100 canonical、100 coordinate-free WL、100 structure-event、50 3D；
- 3 个 manifest、3 个 metrics、0 张 PNG，目录 250,816 bytes；
- 17/17 新 evidence hash 和 391/391 V2 source evidence hash 全部通过；
- 新 topology/replay/graph/spline、mesh、LiDAR、label、formal data、training、model、trajectory、M-TARE change 全为 0。

因此 V2 的 FAIL 保持原样；V2R 作为独立证据正式冻结 100-parent 身份和 80/10/10 split。下一步 mesh 必须另立 proposal，不能把本结果解释为完整 Phase 1 或数据集 PASS。
