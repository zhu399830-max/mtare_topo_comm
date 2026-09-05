# Cano 100 Parent Perception Mesh M0R4 Review

状态：`M0R4_SEALED_FAIL_REPLAY_ROUTE_RETIREMENT_RECOMMENDED`  
日期：2026-08-11  
证据：`results/gate0_baseline/gate0_20260811_cano_100_parent_perception_mesh_contract_m0r4_unified_discretization_seed0/`

## 结论

M0R4 在固定 train S08 按冻结规则停止并封存 FAIL。S01--S07 通过；S08 唯一失败项是 primary-vertex-to-replay-surface maximum=`0.801825 m`，超过 `0.75 m`。反向 maximum=`0.452970 m`，所有 count、input、axis、mesh quality、AABB 和 area 条件通过。

不建议继续增加 M0R5 percentile/maximum 规则。连续 M0--M0R4 已充分证明：Cano/Open3D Poisson 的线程顺序、三角化、vertex sampling 和后置 vertex-order-dependent noise 不能提供 seed-level exact reconstruction。继续约束两次独立 Poisson 输出，不是下游 CPU LiDAR 所需的科研问题。

正式下游会读取一次生成并 SHA-256 封存的 OBJ，不会在训练或评测时重新 meshing。因此正确的复现单位应是 immutable perception-mesh asset，而不是再次运行非确定上游后要求近似得到同一 surface。

## 正式事实

- 83/83 unit、16/16 E1 external、preflight 0/0 通过；
- 执行 `690.162 s`，封存前 `273,704,826 bytes`；
- 16 个 mesh、8 份 metrics、8 张完整 train 图；S09/S10 未生成；
- 146/146 evidence SHA-256 复核通过；
- 零 validation/test read、LiDAR、label、dataset、training、model、trajectory、simulator 和 M-TARE change。

## S08 失败性质

- point-to-surface p99=`0.294 m`，p99.9=`0.365 m`，p99.99=`0.416 m`；
- 只有 1/185,205 个 primary vertex 超过 0.75 m，占 `0.000540%`；
- 该点 distance=`0.801825 m`，triangle degree=4；
- vertex counts `185205/185205`；triangle counts `370416/370418`，差异 `0.000540%`；
- AABB difference=`0.031998 m`，area difference=`0.091430%`；
- 两个 mesh 各自全部质量检查通过，完整图无断裂或孤立大组件。

若仅为让 replay PASS，可以把 maximum 改为 robust percentile；但这会继续优化一个不会被下游使用的二次重建行为。科研资源应转向 LiDAR 感知合同和结构语义任务。

## 推荐 M0F：single-pass immutable asset contract

从头生成同样十个固定 train sentinel，但每个只生成一个 primary native mesh：

1. graph、spline、operation trace、effective parameters、reserved seed 和 axis 与 source exact；
2. 每个 mesh 独立通过 nonempty、finite、index、zero-degenerate、component>=0.999、axis coverage 和 spline-in-AABB 合同；
3. 生成十张完整 X-Y/X-Z train 图并全部人工核验；
4. 记录 vertex/triangle/component/AABB/area/manifold diagnostics；
5. 每个 OBJ、axis、graph、spline、parameter document 以 SHA-256 封存；
6. 后续 CPU LiDAR 和 M1 只能读取封存资产，禁止重新 meshing 替换；
7. 不再生成 replay mesh，不再声称上游 meshing 可按 seed bitwise 或 surface-level 重现。

M0F 仍不生成 LiDAR、标签、正式数据或模型。PASS 后只允许提出 M1：为固定 100 parents 各生成并封存一个 primary mesh。这样 M0F 为 10 个网格而不是 20 个，M1 为 100 个而不是 200 个，直接减少约一半 meshing 时间和磁盘。

这不是放松每个正式资产的质量；它删除的是与实际系统无关、且已经证明不成立的“独立重建等价”假设。
