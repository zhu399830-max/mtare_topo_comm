# 五拓扑 CPU LiDAR 感知合同 V3 复核

更新时间：2026-08-11

## 本轮回答的问题

使用已经通过完整空间覆盖审计的 selector v2，在不更换 topology、seed、mesh、标签或冻结规则参数的条件下，五类 Cano 地下拓扑能否完整生成静态 CPU LiDAR 观测、客观 outgoing-branch 标签，并得到可复核的规则基线结果。

结论：`PASS_CANO_FIVE_TOPOLOGY_CPU_CONTRACT_PILOT`。

证据目录：`results/gate0_baseline/gate0_20260811_cano_five_topology_cpu_contract_pilot_v3_selector_coverage_seed0/`。

## 实际执行链路

```text
5 个预声明 Cano topology parent
→ graph / splines / native perception mesh
→ selector v2：结构事件 + 按弧长配额 + 5 m 间距 + 7.5 m 完整覆盖
→ 250 anchors × 每点 3 个固定 yaw = 750 个静态观测
→ Open3D CPU first-return raycast，16×720，0.3--50 m
→ 完整 spline/TNG 只生成 360° outgoing-exit teacher label
→ 冻结 range-sector rule 预测
→ branch precision / recall / F1 / angular error
```

学生可见输入只有 `student_range_m` 和 `student_valid_mask`；`teacher_exit_label_720` 是监督目标。完整 graph、spline 和 mesh 不进入学生输入。

## 数据量与文件合同

| 项目 | 实际数量 |
|---|---:|
| topology parent / native perception mesh | 5 / 5 |
| canonical anchor | 250 |
| 静态观测 | 750 |
| 每观测射线 | 11,520 |
| 两个独立 scene 的总 primary rays | 17,280,000 |
| NPZ shard | 5 |
| 完整地图 / contact page | 5 / 30 |
| 正式数据集样本 / 训练样本 / 模型 | 0 / 0 / 0 |

每个 shard 只含：

- `student_range_m`: `[150,16,720]`, `float32`；
- `student_valid_mask`: `[150,16,720]`, `uint8`；
- `teacher_exit_label_720`: `[150,720]`, `float32`。

这 750 个观测是合同诊断数据，不属于 train/validation/test，不能拿来宣称数据集已完成或模型已训练。

## 定量结果

| topology | Precision | Recall | F1 | 平均角误差 | 分支数完全正确 |
|---|---:|---:|---:|---:|---:|
| P01 straight-turn | 1.0000 | 1.0000 | 1.0000 | 2.542° | 1.00 |
| P02 branch-deadend | 0.9109 | 0.9388 | 0.9246 | 2.284° | 0.94 |
| P03 loop-bottleneck | 0.8667 | 0.9100 | 0.8878 | 4.819° | 0.90 |
| P04 chamber-multiexit | 0.8596 | 1.0000 | 0.9245 | 4.328° | 0.84 |
| P05 slope-multiheight | 0.8095 | 0.8673 | 0.8374 | 3.561° | 0.88 |
| 总体 | 0.8872 | 0.9431 | 0.9143 | 3.501° | 0.912 |

五个 parent 的双独立 RaycastingScene 最大 range 差异均为 `0 m`，branch LOS 全部通过。selector v2 的总体最小 anchor 间距为 `5.0061 m`，最坏完整 spline 覆盖半径为 `6.3917 m`，满足冻结的 `>=5 m` 与 `<=7.5 m` 合同。

## 人工可视化复核

已逐张检查 5 张完整 X-Y/X-Z 地图和全部 30 张 contact page，覆盖全部 750 个观测：

- 五张地图都显示完整 mesh、中心线和 anchor；P03 回环、P04 多出口 chamber、P05 高低坡段均被覆盖；
- 30 页均正常渲染，没有空白批次或明显传感器损坏；
- P04 的宽 chamber 和多出口附近出现额外规则预测，与其较低 precision 和 `0.84` 分支数完全正确率一致；
- P05 的斜坡、多高度和多分支附近出现漏检、过检和角度偏移，与最低 F1 `0.8374` 一致。

因此可视化支持机器指标，不支持为了提高结果继续在这五张图上调冻结规则。P04/P05 应作为后续学习方法要解决的已知结构困难，但不能反复用作模型选择测试集。

## 结论边界与下一步

本轮证明了五类拓扑的生成、理想化 LiDAR、客观标签、样本存储和规则评测链路可工作。它没有证明：

- 已学到结构特征或结构语义；
- 规则或未来模型能在未见 topology 泛化；
- 静态 anchors 能直接组成因果在线拓扑图；
- native perception mesh 可供机器人碰撞导航；
- CPU synthetic 与 Gazebo/真实 LiDAR 一致；
- 已替换或优于 M-TARE 高层规划器。

按 `docs/PLAN.md`，下一科研任务进入 Phase 1 的 100-topology parent 数据源设计：先冻结 topology family、参数范围、seed、验收、失败处理和 parent-level `80/10/10` 隔离，再经审批批量生成 graph/spline/mesh。正式 100k LiDAR 样本仍必须等待固定 pose 的 CPU↔Gazebo parity 与独立 Data Card；当前不训练 CNN/GNN，也不修改 M-TARE。
