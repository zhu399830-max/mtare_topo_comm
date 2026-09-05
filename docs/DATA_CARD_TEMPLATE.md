# V3 Data Card 模板

本模板用于“先提案、再确认”，不是当前数据集。当前 V3 数据集仍为 `NONE`。Gate 0 未通过前不得据此标注、导出数据或训练。

## 人读摘要

提交审批时先用表格给出：

| 项目 | 必须回答 |
|---|---|
| 目的 | 对应哪个 Gate、研究问题和 operation |
| 原始来源 | rosbag/dataset/world、路径、许可用途、topic |
| 点云事实 | scan/point 时间戳、pose、真实 ray origin、强度及缺失情况 |
| 环境 | train、validation、strict test 的 world 名称 |
| 轨迹 | 每个 world 的独立轨迹、起点/seed、时长、距离、空间覆盖 |
| 样本 | 原始帧、空间/时间采样规则、有效样本、有效结构事件 |
| 结构覆盖 | junction、branch、dead end、width change、slope、多高度等 |
| 输入 | 在线可得通道、坐标、范围、分辨率、历史因果窗口 |
| Teacher | 来源、输出、valid mask、与 local planner 一致性检查 |
| AI 标注 | annotation bundle、labeler/version、prompt、schema、confidence/abstain、冲突规则 |
| 人工金标 | 分层数量、复核者、双标/仲裁、稀有与冲突样本处理 |
| Split | world/trajectory disjoint、历史污染审计、test 隔离 |
| 成本 | 预计磁盘、导出时间、训练时间和算力 |
| 批准 | 用户确认记录、允许的 operation 和 Gate |

## 机器字段

`tools/v3/validate_manifest.py` 读取 `v3_data_card_v1` JSON。顶层必须包含：

```text
schema_version, card_id, purpose, approval, source, worlds,
trajectories, sampling, teacher, split, leakage_audit, estimated_cost
```

其中：

- `approval.status` 必须是 `APPROVED`，并列出 `authorized_operations`、`authorized_gates` 和 `confirmation_reference`；
- `worlds` 分列 train、validation、ssl、normalization、teacher_calibration、threshold_calibration、augmentation_tuning、checkpoint_selection、strict_test；
- 每条 trajectory 必须记录 id、world、split、independent、duration_s、distance_m、spatial_coverage_m；
- `sampling` 必须同时记录 raw frame、effective sample、effective structure event 和空间采样规则；
- `leakage_audit` 必须逐项确认 strict test 未进入监督、SSL、统计、标定、增强调节和 checkpoint 选择。

若 operation 是 `annotation_pilot` 或 `ai_annotation`，还必须包含 `annotation`：

```text
labeler_name, labeler_version, prompt_reference,
input_bundle_contract, output_schema,
planned_ai_sample_count, planned_human_gold_count,
conflict_policy, abstain_policy,
strict_test_excluded=true, raw_responses_preserved=true
```

AI 辅助标签必须与 planner/map 客观标签分字段保存；不能通过人工或 AI 覆盖后丢失原始 teacher。每个样本还应保留 prompt hash、原始响应、逐字段 confidence、审计状态和最终采纳原因。

机器校验通过不等于数据充分。数据是否足以进入 Gate 2，仍需结合地下 world 数、独立轨迹、空间覆盖、结构事件分布和 teacher-planner consistency，由用户审查 Gate 1 证据后决定。
