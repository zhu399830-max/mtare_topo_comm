# Cano 100 Parent Perception Mesh M1R Review

状态：`PASS_CANO_100_PARENT_PERCEPTION_MESH_M1R`  
日期：2026-08-11  
正式运行：`results/gate0_baseline/gate0_20260811_cano_100_parent_perception_mesh_m1r_sanitized_assets_seed0`

## 1. 本阶段做了什么

M1R 从冻结的 V2R parent 身份重新物化全部 100 个原生 Cano 感知网格，并在每个原生 OBJ 写出后统一执行确定性退化面审计。审计谓词固定为 `float64 doubled_area <= 1e-12`；只有命中谓词的 OBJ face 行允许删除，所有 vertex 行及重载后的顶点坐标必须逐值不变。最终 OBJ 作为不可变下游输入，禁止重新 meshing 替代。

这一步只建立感知场景资产，不采集 LiDAR、不生成监督标签、不建立正式数据集、不训练模型，也不修改在线拓扑图或 M-TARE。

## 2. 正式结果

- 正式状态 PASS；executor 约 `5935.73 s`，runner 约 `5937.17 s`。
- parent 与 split：100 个，10 个 recipe 各 10 个；`80 train / 10 validation / 10 development-test`。
- 资产：100 个 primary OBJ、100 个 sanitation record、0 replay；100/100 source identity、单资产质量和 sanitation 合同通过。
- 总规模：`12,608,078` vertices、`25,217,511` triangles、`1,887,494,332` OBJ bytes。
- 最终退化面总数为 0；最差 largest-component fraction 为 `0.9998534082580015`；最大 component 数为 12，39 个资产含多个 component，但均通过冻结的 `>=0.999` 主分量门槛。
- 100 个 final OBJ hash 全部唯一；100/100 顶点数量和坐标精确保持；100/100 raw hash 与 final hash 相同。
- 80 张 train-only 完整图已生成；validation 与 development-test 各 0 张，未泄漏到人工开发审阅。
- run 封存清单共 1001 项；runner 记录 V2、V2R、M0F、M1 四个上游封存源均为 0 mismatch。
- scope 保持为零：anchors、LiDAR observations、teacher labels、formal dataset samples、training samples、models、trajectories、Gazebo/Isaac runs、M-TARE changes。

## 3. 为什么这次没有删除面仍然算有效修复

上一轮 M1 在 `S05_flat_branch_medium_C05` 产生了 1 个严格共线、零面积三角面，因此按冻结规则停止。M1R 对全部 100 个 parent 从头生成；本次 Poisson 运行没有重现该面，100 个 sanitation record 的 `removed_triangle_count` 均为 0。

这说明原生 Poisson 物化存在运行级非确定性，不能保证相同 parent 每次都出现相同退化面。M1R 没有据此降低门槛：它把统一 sanitation/audit 放到每一次物化的必经路径上，若未来再次出现满足谓词的面，只删除对应 face 行并证明顶点精确不变；若出现谓词外质量失败，仍必须停止。此次 PASS 的含义是“本次封存的 100 个最终资产全部合格”，不是“原生生成器已变为确定性”。

## 4. 人工可视化核验

按 recipe 检查了全部 80 张 train 完整 X-Y/X-Z 图。R01--R10 均未发现空白、画面裁剪、整条隧道缺失、中心线明显越出网格包络或阶段标题错误；flat recipe 保持近似零高差，3D recipe 显示明确坡度和多高度变化。R08--R10 的富环、富分支和复杂交汇处也未见网格脱落。

图中灰色为 mesh 确定性采样、蓝色为 grown tunnel、橙色为 connector、红叉为 degree >= 3 事件。标题和 provenance 均为 `M1R TRAIN ONLY`。没有渲染或人工查看 validation/development-test 图。

## 5. 科学结论与边界

Phase 1 的 100-parent 程序化拓扑源、80/10/10 parent-level split 和对应不可变感知网格资产已经建立。它们覆盖 flat/3D、不同 grown/connector recipe 及实际 cycle-rank 变化，可作为后续理想 CPU LiDAR 的场景材料。

尚未完成的是研究方法主体：当前没有正式点云样本、结构监督标签、CNN/GNN、结构语义特征、在线 topometric graph 泛化证据、M-TARE 高层替换或探索收益。M1R PASS 不能被表述为“模型已经训练好”或“拓扑规划已经完成”。这些网格也没有取得动态 collision/navigation 资格。

## 6. 下一步唯一建议

先提交并审批固定 pose 的 CPU synthetic 与 Gazebo LiDAR parity 合同。该合同只验证同一冻结几何、同一 pose、同一 16×720 射线定义在两个后端的坐标系、有效射线和距离分布是否可对齐；仍不生成正式训练集。

只有 parity PASS 且正式 Data Card 另获批准后，才对 80 个 train parent 生成 causal/student input 与客观 outgoing-branch 标签。validation 只用于冻结指标，development-test 保持封存；不得把原 M-TARE benchmark 地图加入训练或调参。
