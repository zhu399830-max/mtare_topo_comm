# Phase 4 C08 V2R Research Review

日期：2026-08-17

## 审查范围

本审查只读取已经封存的 C08 V2R evidence，不执行新 raycast、inference、graph replay、training 或参数选择，不读取 C09/C10/M-TARE。权威输入为：

- `gate4_20260817_cano_c08_route_conditioned_causal_topology_replay_v2r_seed0`
- 3 个 C08 development worlds、4,773 个 sensor frames
- 243 参数组 × 5 方法 × 3 worlds = 3,645 行 sweep
- B0、M1D seeds 0/1/2 与 GT oracle 的 15 个 selected graphs

## 机器合同结论

V2R 保持正式 PASS。4,773 帧、109,969,920 条双场景 rays、14,319 个 inference frames、3,645 次 graph replay 和所有 oracle-integrity rows 均通过；training/optimizer step、C09/C10/M-TARE reads 均为 0；583/583 seal 一致。

该结论证明 C08 development replay 与证据链完整，不等于 Gate 4 PASS，也不证明 untouched-topology 泛化或规划收益。

## 冻结参数的稳定性

冻结参数为 `sf2_tr8_lr6_hh20_te45_da20`，C08 三 seed × 三 world 的 mean composite 为 `0.8262623347`，总体标准差 `0.0340769`，最差单元为 `0.7807416`。

参数选择是确定性的，但不是统计上唯一：heading merge 为 `20/25/35 deg` 的三组配置取得完全相同的最高均分；实现按 `(-score, parameter_id)` 排序，因此词典序选择 `hh20`。共有 12 组参数距最优不超过 `0.005`。C08 的强证据主要支持 stable frames=`2`、event travel=`8 m` 和 loop radius=`6 m`；对 heading merge 角度的辨识力较弱。

按 seed 看，冻结参数排名/距该 seed 最优差值为：

- seed0：并列最优，差值 `0`
- seed1：第 12，差值 `0.014661`
- seed2：第 9，差值 `0.008333`

按 world 对三 seed 求均值，冻结参数在 S01/S06/S10 分别排第 `3/48/51`，距各 world 最优为 `0/0.031120/0.020118`。它是三世界聚合折中，不应描述为每个世界的局部最优。

## 方法效果

冻结参数下，三 seed M1D 的整体 mean composite=`0.826262`、exit F1=`0.654952`；B0 分别为 `0.743541/0.497732`。绝对提升为 `+0.082722/+0.157219`，相对提升约 `11.1%/31.6%`。

三 seed composite 为 `0.847797/0.824948/0.806043`，存在 `0.041754` 的 seed range。M1D 并非逐 seed、逐 world 支配 B0：S06 上 seed1 的 exit F1 低 `0.047619`，seed2 的 exit F1/composite 分别低 `0.035088/0.006678`。因此科研结论只能是 aggregate improvement，不能声明 uniformly better。

世界难度随结构复杂度增加：

| world | M1D mean composite | M1D mean exit F1 | predicted structural nodes | truth events |
|---|---:|---:|---:|---:|
| S01 | 0.862266 | 0.704646 | 13.67 | 9 |
| S06 | 0.824315 | 0.657616 | 24.67 | 14 |
| S10 | 0.792206 | 0.602592 | 38.00 | 24 |

S10 是明确最差世界；三个世界均有结构节点过生成，尤其 S06。connectivity 和 verified-edge correctness 对全部方法均为 `1.0`，说明移动边合同稳定，但这两个指标在当前 C08 上缺少区分度，不能单独支撑语义优势。

## Oracle 边界

Oracle mean exit F1=`0.740573`，比三 seed M1D 均值高 `0.085622`，说明结构语义仍有清晰提升空间。Oracle composite=`0.847775`，而 seed0 composite=`0.847797`，seed0 高约 `0.000021`。原因是 composite 同时奖励 terminal reachability、冗余、路径和 churn；oracle 只保证 evaluator-integrity，不是该加权 composite 的数学上界。

因此后续不得写“模型综合性能超过 oracle”，也不得把 oracle composite 当作可达上界。V2R 的预声明硬门只要求 oracle connectivity 与 verified-edge correctness 为 1.0，该现象不推翻正式 PASS，但限制指标解释。

## 科研判断与下一步

审查结论为 `C08_DEVELOPMENT_PASS_WITH_ROBUSTNESS_CAVEATS`：C08 足以冻结一个确定性 development tuple，并支持 M1D 相对 B0 的聚合改进；它不足以给出唯一参数识别、逐世界优势或 unseen-topology 结论。

推荐下一步是另立一次严格 C09 validation proposal，并保持：

1. 参数 `sf2_tr8_lr6_hh20_te45_da20`、模型、评分和所有阈值完全冻结；
2. C09 只执行一次，不按结果回调参数或选择 seed；
3. 报告三 seed 中位数、最差世界、S10 类复杂世界退化和节点过生成；
4. 明确 C09 已参与 M1D checkpoint selection，只能评价冻结感知下的图构建泛化，不能声称严格端到端 unseen generalization；
5. 历史 C09 里程/帧数是未审计估计，正式 Data Card 前必须另获批准，只读建立连续 trajectory、geometry 与 exact sample-count contract；
6. C10、planner 和 M-TARE 继续禁止。

本审查不授权读取 C09。Gate 4 保持 `GATE_MIXED`。
