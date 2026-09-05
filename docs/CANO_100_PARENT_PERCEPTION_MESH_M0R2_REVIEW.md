# Cano 100 Parent Perception Mesh M0R2 Review

状态：`M0R2_SEALED_FAIL_M0R3_POINT_TO_SURFACE_PROPOSAL_PENDING_APPROVAL`  
日期：2026-08-11  
证据：`results/gate0_baseline/gate0_20260811_cano_100_parent_perception_mesh_contract_m0r2_triangle_discretization_seed0/`

## 结论

M0R2 在固定 train sentinel `S08_3d_loop_rich_C01` 按冻结规则停止并封存 FAIL。S01--S07 全部通过；S08 的 triangle discretization、exact topology/input/axis/vertex count、两个单体 mesh、AABB 和 surface area 均通过，唯一失败项是 replay vertex 到最近 primary vertex 的 maximum distance=`0.844986 m`，超过 `0.75 m`。

失败不是地图、网格连续性或 0.01% triangle-count 修订的问题，而是“vertex-to-vertex maximum”仍混入了两次 Poisson 三角化采样位置不同造成的对应误差。科学上应比较点到对方三角形表面的距离。

## 正式执行事实

- 实现后 79/79 V3 unit、14/14 E1 external 通过；
- preflight 为 0 errors、0 warnings；
- 正式执行 `692.344 s`，封存前 `273,706,271 bytes`；
- 完成 S01--S08 primary/replay 共 16 个 native mesh、8 份 parent metrics 和 8 张完整 train 图；
- S09/S10 未生成；没有 retry、seed/parent 替换、修复或阈值修改；
- runner 状态为 `FAIL_CANO_100_PARENT_PERCEPTION_MESH_CONTRACT_M0R2`；
- 146/146 evidence SHA-256 复核通过；
- validation/development-test、LiDAR、label、formal dataset、training、model、trajectory、simulator 和 M-TARE change 均为 0。

## S08 机器证据

| Metric | Primary→Replay | Replay→Primary | Result |
|---|---:|---:|---|
| nearest-vertex maximum | 0.576404 m | 0.844986 m | reverse FAIL |
| nearest-vertex p99 | 0.431034 m | 0.429871 m | recorded |
| nearest-vertex p99.9 | 0.491219 m | 0.492261 m | recorded |
| nearest-vertex p99.99 | 0.529466 m | 0.563481 m | recorded |
| vertices >0.75 m | 0 | 3/185,205 | 0.001620% |

三个超限 replay vertices 不是孤立点：triangle degree 为 7、8、8，主要聚集在 `(8,-8,-1) m` 附近的局部连接表面。S08 两次 triangle count 为 `370418/370416`，相对差 `0.0005399%`，通过 0.01% 合同；AABB 最大差 `0.012009 m`；surface-area 相对差 `0.050641%`；两个 mesh 均为一个 triangle component、零 degenerate triangle。

## 完整图核验

S01--S08 的完整 X-Y/X-Z 图全部逐张检查：

- gray mesh 连续覆盖全部 blue grown 和 orange connector；
- flat/3D 高度关系正确；
- tree、unicyclic、branch-medium、loop-rich 的分支和环结构完整；
- S08 多环、连接段和约 `-25 m` 到 `+45 m` 高差正常；
- 没有可见断裂、空白、尖刺、孤立大组件或坐标轴误用。

## 只读 point-to-surface 复算

使用 Open3D `RaycastingScene.compute_distance`，把每个 primary vertex 查询到 replay triangle surface，反向同样计算；不修改或重写任何封存资产。

S01--S08 的双向 point-to-surface maximum 全部在 `0.442259--0.510173 m`。S08 为：

- primary vertices → replay surface：`0.494647 m`；
- replay vertices → primary surface：`0.502614 m`；
- 两个方向均没有点超过 `0.75 m`。

这证明 S08 的 `0.844986 m` 是 vertex sampling correspondence artifact：附近没有足够接近的 primary vertex，但存在距离约 `0.503 m` 的 primary triangle surface。

## 推荐 M0R3（尚未批准）

保持 M0R2 的十个 train parent、seed、native method、20 个 mesh、10 张完整图、triangle-count 0.01% 条件、资源上限和所有零范围不变。只替换一个错误指标：

- 移除 bidirectional nearest-vertex maximum 的 PASS/FAIL 作用；
- 改为 bidirectional vertex-to-opposite-triangle-surface maximum `<=0.75 m`；
- nearest-vertex mean/p99/p99.9/p99.99/maximum 继续记录，仅作为诊断；
- exact graph/spline/operation trace/effective parameters/axis/vertex count 全部保留；
- individual mesh、triangle-count relative difference、AABB、surface area 和 OBJ hash 规则全部保留。

`0.75 m` 阈值没有改变，仍来自 post-Poisson `Uniform[-0.2,0.2] m` 每坐标噪声的 paired 3D 理论上界 `sqrt(3)*0.4=0.692820 m` 加 `0.057180 m` 数值余量。修订的是被比较的几何对象：surface mesh 应与 surface 比较，不能把不同三角化的 vertex sampling 当作表面本身。

M0R3 必须另获用户批准后才能实现和执行。M0R2 失败目录保持不可变且不拼接复用。
