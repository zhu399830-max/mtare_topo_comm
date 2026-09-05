# V3 仓库结构与迁移规则

实施前第一顺序固定为读取 `docs/PLAN.md` 和 `docs/PROGRESS.md`。它们分别给出当前权威路线和唯一下一任务；本结构文件不得被用来绕过 Phase 边界。

本文件规定 MASTER PLAN V3 的正式代码、配置、测试和结果应放在哪里。旧目录暂不移动、不删除；只有通过当前 Gate 的 contract 和测试后，代码才迁入 `src/mtare_topo/`。

## 1. 正式结构

```text
docs/                       项目总纲、状态、决策、实验规程
src/mtare_topo/
  common/                   坐标、时间戳、标识、通用类型
  data/                     原始数据读取、StandardFrame、输入构建
  teacher/                  teacher、有效 mask、一致性审计
  representation/           Gate 2 表示学习
  semantics/                Gate 3 方向语义与结构角色
  topology/                 Gate 4 增量图、节点、边、exit stub
  planning/                 Gate 6/7 全局选择与任务分配
  integration/              ROS/M-TARE 适配与 shadow/closed-loop
  evaluation/               指标、统计、可视化和失败分类
configs/v3/gate0..gate8/    经审批的 Gate 配置；不得跨 Gate 混放
tools/v3/                   实验预检、运行目录、manifest 和状态工具
tests/v3/
  unit/                     纯函数和局部行为
  contracts/                schema、坐标、topic、输入输出 contract
  replay/                   离线/在线一致性和 deterministic replay
  integration/              M-TARE/ROS 集成
results/gate0_baseline..    V3 正式证据；每次运行一个独立目录
results/prototypes/         明确标注的开发纵向切片；不能计作 Gate PASS
legacy/                     旧代码、旧结果和失败知识的索引，不存副本
```

当前 Gate 3 的正式方法细化为 `docs/PRIMITIVE_RELATION_STRUCTURAL_GRAPH_PLAN_V1.md`。它取代旧的事件分类/Composer 主线；`docs/SUPERVISED_STRUCTURE_LEARNING_V1.md` 仅保留通用数据隔离和历史监督学习治理要求。任何输入、teacher、导出或训练实现开始前，必须先读取 `docs/PLAN.md`、`docs/PROGRESS.md` 和新的方法计划。

## 2. 当前事实

- 当前处于 Gate 3 方法重建阶段，尚未形成新方法的 Gate 结论。
- 当前唯一任务是构造监督可行性验证：确认程序化地图能否为可见 LiDAR 返回提供唯一、无泄漏的几何基元、变换、组合关系和表面来源标签。
- 新方法尚未导出数据、生成 teacher、训练模型或建立图；旧 checkpoint 仅作为历史基线或失败证据。
- `learning/`、`configs/learning/`、`contracts/` 和旧 `results/` 仍是历史资产，不因目录重构自动升级为 V3 实现。
- 当前已有一个不训练的开发纵向切片：`range_exit_baseline.py` 与 `online_topometric.py` 已进入 `src/mtare_topo/` 并有单元测试；它不代表 Phase 3/4 PASS。阅读与复现入口为 `docs/WORKING_VERTICAL_SLICE_V1.md`。

## 3. 代码迁移规则

采用逐模块迁移，不做一次性重写：

1. 先在当前 Gate 写清输入、输出、坐标系、时间因果性和验收指标。
2. 从旧代码中选最小可复用实现，补 provenance 和测试。
3. 通过 contract/replay 测试后迁入对应 `src/mtare_topo/` 子包。
4. 新代码成为唯一正式入口后，旧入口标为 `LEGACY_REFERENCE`。
5. 只有确认无调用、结果已归档且用户明确同意，才删除旧文件。

任何迁移都不得顺手改变数据 split、teacher、指标、阈值或实验结论。

## 4. 状态标签

- `V3_ACTIVE`：当前 Gate 正式使用并有测试/证据。
- `REUSE_CANDIDATE`：可能复用，但尚未通过 V3 contract。
- `LEGACY_BASELINE`：可作明确标注的历史对照，不能支持 V3 结论。
- `FAILURE_EVIDENCE`：用于保留失败模式和防止重复踩坑。
- `RETIRE_CANDIDATE`：疑似冗余，未获批准前不得删除。

具体映射见 `legacy/CODE_ASSET_MAP.md` 和 `legacy/RESULT_ASSET_MAP.md`。

## 5. 结果目录规则

标准运行 ID：

```text
gate{N}_YYYYMMDD_<question-or-task>_<method>_seed<seed>
```

每个运行目录至少包含：

```text
config/      run spec、配置快照、data card/manifest
logs/        stdout、stderr、系统或 ROS 日志
metrics/     原始指标与机器可读 summary
previews/    只回答研究问题的图，并附 provenance
artifacts/   checkpoint、trajectory、graph、decision trace 等
```

运行不得直接写入旧结果目录，也不得覆盖已有 run。
