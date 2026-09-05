# Anchor Selector Coverage Audit V2

日期：2026-08-11  
正式结果：`PASS_CANO_ANCHOR_SELECTOR_COVERAGE_AUDIT_V2`  
证据目录：`results/gate0_baseline/gate0_20260811_cano_anchor_selector_coverage_audit_v2_seed0/`

## 本轮到底验证什么

本轮只验证五类固定地下拓扑中 50 个 LiDAR 候选采样位姿是否沿完整隧道合理铺开。它没有生成新 mesh，没有发射射线，没有产生训练样本，没有训练模型，也没有修改在线拓扑图或 M-TARE。

选择器保留端点、路口及主要转弯等结构事件；其余名额按每条 tunnel 的 spline 弧长比例分配。随后在 0.5 m 弧长候选格上求解确定性二元混合整数规划，同时满足：每图恰好 50 点、全局三维欧氏间距至少 5 m、每条 tunnel 的非事件点数量等于弧长配额、完整 spline 上任一 0.5 m 探针到同 tunnel anchor 或共享结构事件不超过 7.5 m。

## 结果

| 拓扑 | anchors | 最小全局间距 (m) | 最大同隧道覆盖半径 (m) | 结论 |
|---|---:|---:|---:|---|
| P01 straight-turn | 50 | 5.006 | 3.500 | PASS |
| P02 branch-deadend | 50 | 5.500 | 4.000 | PASS |
| P03 loop-bottleneck | 50 | 5.185 | 6.392 | PASS |
| P04 chamber-multiexit | 50 | 5.500 | 4.000 | PASS |
| P05 slope-multiheight | 50 | 5.500 | 4.000 | PASS |

五图的弧长配额均精确满足，结构事件覆盖、全 tunnel 覆盖及确定性 replay 均通过。总体最小间距为 `5.006145889 m`，最坏覆盖半径为 `6.391727627 m`，分别满足 `>=5 m` 和 `<=7.5 m`。

## 完整图人工复核

已逐张检查全部五张 X-Y/X-Z 完整图和汇总图。P01 长直段与转弯末端连续覆盖；P03 下支路、上回环和右侧连接段全部有采样点；P05 的正负高差坡段完整覆盖。未再出现 v1 中 P01 `32.780 m`、P03 `92.306 m` 的长区间空洞。

可直接查看：

- `previews/P01_straight_turn_complete_anchor_map.png`
- `previews/P02_branch_deadend_complete_anchor_map.png`
- `previews/P03_loop_bottleneck_complete_anchor_map.png`
- `previews/P04_chamber_multiexit_complete_anchor_map.png`
- `previews/P05_slope_multiheight_complete_anchor_map.png`
- `previews/five_topology_spacing_coverage_summary.png`

## 科研边界与下一步

该 PASS 只解除“五拓扑 LiDAR pilot 的采样位置不完整”这一阻塞。它不证明 LiDAR 有效、结构语义可学习、模型能泛化、在线拓扑正确或探索性能提高。

下一步应使用相同五类拓扑、相同 mesh/sensor/teacher/冻结规则和 selector v2，另立一次 corrective 五拓扑 CPU LiDAR pilot。此前 v2b 的 P01--P03 观测只保留为失败 run 诊断，不混入新结果。正式执行前必须向用户明确 5 mesh、250 anchors、750 views、17.28M primary rays、预计时间/磁盘、验收和停止条件，并取得新的执行批准。
