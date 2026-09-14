# 完整地图与数据集审计（Gate 0）

## 先说结论

- 当前 V3 正式数据集：**尚未建立（NONE）**。Gate 0 只做接口、地图和旧证据审计，没有批准生成新训练样本。
- 旧闭环语义模型的实际训练数据：`results/topological_semantic_dataset_v4_supervision_fixed`，共 4849 个局部 `.npz`。
- 真正进入旧模型训练集的场景只有 `tunnel + garage`；`forest` 是验证，`campus + indoor` 是测试。
- 这些 `.npz` 是四条训练轨迹沿线截取的局部观测窗口，不是 3024 张独立地图，更不是 3024 个独立地下环境。
- 项目里另有 5566 个 LAMP 结构表征样本，但属于更早的另一条结构学习流水线，并未训练上述旧闭环语义模型。

## 当前拟议实验地图

完整源几何总图：`current_benchmark_complete_source_maps.png`

| 用途 | 地图 |
|---|---|
| 开发 | `tunnel`, `unseen_mine`, `external_cave` |
| 回归开发 | `subtgraph_operational_01`, `subtgraph_operational_02` |
| 接口冒烟，不作地下结论 | `garage` |
| V3 严格未见测试 | 暂无；仍需隔离新增至少两个不同几何族地下世界 |

图中展示的是 `.world/.dae/.obj` 的完整范围，红星是当前配置起点。它不是历史轨迹图，也不是机器人扫到哪里才出现哪里的累计点云。

## 旧语义数据拆分

| split | 数量 | 世界 |
|---|---:|---|
| train | 3024 | tunnel 1545；garage 1479 |
| val | 356 | forest 356 |
| test | 1469 | campus 694；indoor 775 |

训练轨迹只有四条：`tunnel:traj01`, `tunnel:traj02`, `garage:traj01`, `garage:traj02`。因此样本数量看似为 3024，几何多样性仍只有两个世界，而且同一轨迹相邻样本高度相关。

## `.npz` 是什么

`.npz` 是 NumPy 的压缩归档格式，一个文件可以保存多个命名数组。旧语义数据的代表性样本包含：

- 当前局部输入 `student_input_current`: `4×100×100`；
- 8 帧历史 `student_input_history`: `8×4×100×100`；
- 32 个方向的可通行性、可达距离、可达面积和出口监督；
- 出口中心/宽度/长度；
- 连续 openness、bottleneck、main-path-continuity 分数；
- 64 维 canonical/topological role 教师目标；
- 位姿、时间戳、世界、轨迹、split 和数据契约字段。

所以“监督标签”来自离线教师规则/几何处理结果，并非人工逐帧真值。V3 如果改为 AI 辅助人工标注有监督学习，必须重新定义标签本体、标注输入、复核协议和世界隔离，不能直接把这些旧 teacher 标签当作人工真值。

## 文件位置

- 完整地图清单：`complete_source_map_manifest.json`
- 数据集机器可读清单：`dataset_inventory.json`
- 旧语义数据：`results/topological_semantic_dataset_v4_supervision_fixed/`
- 旧部署权重：`results/semantic_bottleneck_role/20260807_polar_semantic_v5_final/semantic_bottleneck_underground_deployment_epoch70.pt`
- 旧部署声明：`results/semantic_bottleneck_role/20260808_underground_crossval_v1/deployment_manifest.json`
- 旧 LAMP 结构数据：`results/structural_dataset_v4_multienv/`

## 当前暴露的问题

1. 旧部署模型的训练几何只有两个，而且 `garage` 不是地下环境。
2. 旧地下交叉验证只是同一 `tunnel/garage` 世界里的轨迹互换，不是世界级隔离验证。
3. `forest/campus/indoor` 没有增强训练，只能说明做过域外验证/测试；它们也不适合作为地下任务主证据。
4. 所有现有地下候选地图都已被历史开发或结果观察污染，因此没有严格未见测试集。
5. SubTGraph 两个 OBJ 的完整横向跨度达到公里量级，而当前起点配置在 `(0, 40)`；这与历史 no-motion/spawn 问题一致，必须先做单位、连通域和可达面审计，不能直接纳入正式公平实验。
