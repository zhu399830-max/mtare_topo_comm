# 旧代码资产地图

本表是迁移索引，不是复用许可。`REUSE_CANDIDATE` 必须在对应 Gate 重新通过 contract、因果性、坐标和 replay 测试。

| 旧路径 | 现有作用 | V3 定位 | 计划归属 | 重新使用前必须验证 |
|---|---|---|---|---|
| `learning/local_structural_map/schema.py` | StandardFrame/样本结构 | REUSE_CANDIDATE | `src/mtare_topo/common`、`data` | 字段、单位、坐标系、在线可得性 |
| `learning/local_structural_map/geometry.py` | 坐标与几何变换 | REUSE_CANDIDATE | `common` | 变换方向、yaw、时间同步单测 |
| `learning/local_structural_map/builder.py` | 局部结构输入构建 | REUSE_CANDIDATE | `data` | 因果输入、terrain-relative 通道、未知区定义 |
| `learning/local_structural_map/raycast.py` | ray/free-space 工具 | REUSE_CANDIDATE | `data`、`teacher` | 必须存在真实 ray origin；不能由累积 KeyedScan 伪造 free-space |
| `learning/local_structural_map/datasets/*.py` | LAMP/M-TARE/SubT 读取器 | REUSE_CANDIDATE | `data` | 原始来源、许可、topic、轨迹独立性和时间戳 |
| `learning/local_structural_map/runtime/mtare_runtime.py` | 在线 M-TARE 适配 | REUSE_CANDIDATE | `integration` | 离线/在线 parity、延迟、缺帧行为 |
| `learning/local_structural_map/tests/` | 旧 contract/replay 测试 | REUSE_CANDIDATE | `tests/v3/contracts,replay` | 在 V3 schema 下重新运行，不能原样视为通过 |
| `learning/structural_learning/surface_evidence.py` | 表面证据构建 | REUSE_CANDIDATE | `data` | 与部署观测一致、无未来信息 |
| `learning/structural_learning/topological_supervision.py` | 旧结构监督 | FAILURE_EVIDENCE / REUSE_CANDIDATE | `teacher` | 与 local planner 一致性和 valid mask |
| `learning/structural_learning/topological_metrics.py` | 图与语义指标 | REUSE_CANDIDATE | `evaluation` | 指标定义、单位、聚合和边界样例 |
| `learning/structural_learning/tools/render_topology_graph.py` | 图渲染 | REUSE_CANDIDATE | `evaluation` | 图数据 provenance、坐标和颜色图例 |
| `learning/structural_learning/tools/semantic_topology_global_node.py` | 在线语义拓扑/规划原型 | LEGACY_BASELINE | `topology/planning/integration` | 必须拆分职责；Gate 4--6 分别验证 |
| `learning/structural_learning/*model*.py` 与训练脚本 | 旧 bottleneck/semantic 模型 | LEGACY_BASELINE | `representation/semantics` | 新 data card、split、teacher 和模型选择协议 |
| `learning/structural_learning/tools/export_*` | 旧数据导出器 | FAILURE_EVIDENCE | `data/teacher` | 不得直接导出 V3；先冻结 Gate 1 contract |
| `scripts/record_mtare_transfer_world.sh` | world 录包 | REUSE_CANDIDATE | `integration` | world/start/seed/runtime、topic 完整性和有效运动 |
| `scripts/run_semantic_topology_cmu_sim.sh` | 仿真启动原型 | REUSE_CANDIDATE | `integration` | 控制边界、baseline parity、异常退出和日志完整性 |
| `configs/learning/` | 旧实验配置 | LEGACY_BASELINE | 无直接迁移 | 只用于复现实验和追溯，不作为 V3 默认值 |
| `contracts/*.json` | 旧 teacher/任务定义 | FAILURE_EVIDENCE | `teacher/topology` | 重新定义后版本化，禁止暗中沿用标签 |
| `PROJECT_PIPELINE_V2_CN.md` | 旧流程说明 | LEGACY_BASELINE | `docs` 参考 | 与 MASTER PLAN V3 冲突时以 V3 为准 |

## 暂不删除

`learning/`、`configs/learning/`、`contracts/`、旧启动脚本和旧 Docker 环境全部保留。只有新模块成为唯一入口、引用扫描为零、历史复现信息已保留并得到用户批准后，才能列入实际删除清单。
