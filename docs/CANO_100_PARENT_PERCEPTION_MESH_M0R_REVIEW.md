# Cano 100 Parent Perception Mesh M0R Review

状态：`M0R_SEALED_FAIL_M0R2_TRIANGLE_DISCRETIZATION_PROPOSAL_PENDING_APPROVAL`  
日期：2026-08-11  
证据：`results/gate0_baseline/gate0_20260811_cano_100_parent_perception_mesh_contract_m0r_geometric_replay_seed0/`

## 结论

M0R 在第二个固定 train sentinel `S02_3d_tree_small_C01` 按冻结规则停止并封存 FAIL。S01 通过全部修正后的几何重放指标；S02 的 exact topology/input/axis、两个单体 mesh 质量以及全部距离、边界和面积指标均通过，唯一失败项是 primary/replay triangle count 相差 2。

这次失败说明“Poisson 重建的三角面数必须逐个相等”仍是实现级而非几何级重放条件。它不表示 S02 地图断裂、三维结构错误或感知网格不可用。没有修改阈值、换 seed、重试、继续生成 S03--S10，或读取 validation/development-test。

## 正式执行事实

- 执行前 100/100 单元测试、13/13 冻结 E1 external、M0 33/33、V2R 17/17、V2 391/391 source seal 和 preflight 通过；
- 正式执行用时 `78.462 s`，封存前 `46,309,346 bytes`；
- 只生成 S01/S02 primary+replay 共 4 个 native mesh 和 2 张完整 train 图；
- runner 状态为 `FAIL_CANO_100_PARENT_PERCEPTION_MESH_CONTRACT_M0R`；
- 50/50 evidence SHA-256 复核通过；
- LiDAR、label、formal dataset、training、model、trajectory、simulator 和 M-TARE change 均为 0。

## S02 唯一失败项

| 项目 | Primary | Replay | 结果 |
|---|---:|---:|---|
| vertex count | 78,650 | 78,650 | exact PASS |
| triangle count | 157,298 | 157,296 | exact FAIL |
| triangle count absolute difference | 2 | — | 0.001271% |
| surface area | 22,023.3213 m² | 22,001.1608 m² | relative difference 0.100623%, PASS |
| nearest-vertex maximum | 0.692104 m | 0.554699 m | both <=0.75 m, PASS |
| nearest-vertex p99 | 0.429474 m | 0.427670 m | recorded |

AABB 端点最大坐标差为 `0.084740 m`，低于 `0.5 m`。两份 mesh 都是一个 triangle component、largest-component fraction=`1.0`、零 degenerate triangle，并通过全部 axis/spline/AABB 可读性检查。

## 可视化核验

- S01 完整 X-Y/X-Z 图为连续平面分支隧道，mesh 覆盖全部中心线；
- S02 完整 X-Y/X-Z 图显示约 `-20 m` 到 `+22 m` 的三维高差及多个分支事件；
- 两图均无空白、断裂、孤立大组件或坐标轴误用；灰色 mesh、蓝色 grown centerline、橙色 connector 和红色结构事件位置一致。

## 原因判断

M0 已证明 Open3D Poisson 输出的顶点顺序不是字节稳定的。M0R 进一步表明其面三角化也可能有极小离散差异：在完全相同的 graph、spline、operation trace、effective geometry parameters 和 exact axis 下，S02 仅相差 2/157,298 个三角面，而双向几何距离、AABB、表面积及单体网格质量同时通过。

因此 triangle-count exact equality 不应作为 native Poisson 感知网格的物理复现定义。这里的目标是封存可供同一 CPU LiDAR 管线使用的几何等价 primary mesh，不是证明上游 Poisson 的内部三角化逐面确定。

## 推荐 M0R2（尚未批准）

保持 M0R 的 parent、seed、上游方法、20 个 mesh、10 张 train 图、资源限制和所有零范围不变。只修改一个条件：

- 保留 exact graph/spline/operation trace/effective geometry parameters/axis；
- 保留 exact vertex count；
- 把 exact triangle count 改为 `relative difference <=0.0001`（0.01%）；
- 保留双向 nearest-vertex maximum `<=0.75 m`、AABB endpoint `<=0.5 m`、surface-area relative difference `<=1%` 和全部单体 mesh 质量条件；
- 继续记录 OBJ hash 和 triangle count，不要求二者相等；
- 若 vertex count 不等或上述任一条件失败，仍立即停止，不再现场放宽。

0.01% 是一个预先冻结的细粒度离散容差：对 S02 规模最多允许约 15 个面的数量差，明显严于 1% 面积合同；当前观察值为 0.001271%。该修订只使用 train S01/S02 做基础设施合同校准，不读取 validation/development-test，不构成测试泄漏。

M0R2 必须得到用户单独批准后才能实现和执行。M0R 失败目录保持不可变，不作为 M0R2 输出拼接复用。
