# 旧结果资产地图

旧结果按用途归档，不因文件名中的 `unseen`、`validation`、`fixed` 或已有 checkpoint 自动成为 V3 证据。

| 旧结果组 | V3 定位 | 可保留信息 | 禁止用途 |
|---|---|---|---|
| `results/structural_dataset_v3`、`structural_dataset_v4_multienv` | FAILURE_EVIDENCE | 旧输入、采样与规模问题 | V3 训练、归一化、SSL、验证 |
| `results/topological_semantic_dataset_v3_*`、`v4_*`、`v5_*` | FAILURE_EVIDENCE | teacher/split/role-event 缺陷与审计过程 | V3 监督、阈值、模型选择 |
| `results/mtare_transfer_inputs`、`mtare_transfer_recordings` | RAW_ASSET_CANDIDATE | 追溯 rosbag/world/topic；Gate 1 可重新审计原始记录 | 直接复用旧 `.npz` |
| `results/structural_*feasibility`、`structural_multienv_training` | LEGACY_BASELINE | 结构表示可行性和失败案例 | 声称 Gate 2 通过 |
| `results/topological_semantic_model*`、`semantic_bottleneck_role` | LEGACY_BASELINE / FAILURE_EVIDENCE | 旧训练曲线、方向/role 失败、checkpoint 追溯 | V3 checkpoint 初始化、选择或性能结论 |
| `results/offline_topology_node_*`、`topological_role_event_audit_*` | LEGACY_BASELINE | 节点/图指标实现与失败案例 | 声称 Gate 4 通过 |
| `results/semantic_topology_cmu_sim` | FAILURE_EVIDENCE / RUNTIME_ASSET | topic、日志、snapshot、闭环异常 | 公平 baseline 或 Gate 5/6 通过证据 |
| `results/subtgraph_*`、`external_cave_frozen_comparison` | HISTORICAL_SYSTEM_RUN | 外部 world 与执行反馈故障知识 | 自动列为严格 test 或最终性能比较 |
| `results/v3_gate_audit` | V3_ACTIVE | 初始化审计和 clean-room 边界 | 代替 Gate 0 baseline 实验 |
| `results/gate0_baseline` 到 `gate8_final` | V3_ACTIVE | 通过治理流程产生的新证据 | 跨 Gate 混放或覆盖旧 run |

## 资产读取顺序

先读 summary/manifest/config，再看 metrics 和失败日志，最后才看 preview。旧 `.npz` 只能检查其结构和缺陷，不能复制进新数据集。旧 `.pt` 只能追溯架构和失败，不得载入 V3 正式训练。
