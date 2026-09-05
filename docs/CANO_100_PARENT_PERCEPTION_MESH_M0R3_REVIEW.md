# Cano 100 Parent Perception Mesh M0R3 Review

状态：`M0R3_SEALED_FAIL_M0R4_UNIFIED_DISCRETIZATION_PROPOSAL_PENDING_APPROVAL`  
日期：2026-08-11  
证据：`results/gate0_baseline/gate0_20260811_cano_100_parent_perception_mesh_contract_m0r3_point_to_surface_seed0/`

## 结论

M0R3 在固定 train sentinel `S07_flat_loop_rich_C01` 按冻结规则停止并封存 FAIL。S01--S06 全部通过；S07 唯一失败项是 exact vertex count：primary/replay 为 `156110/156109`，相差 1 个，relative difference=`0.0006406%`。

point-to-surface 修订本身通过：S07 双向 maximum=`0.462879/0.511900 m`，均低于未改变的 `0.75 m`。triangle count 为 `312234/312232`，relative difference=`0.0006405%`，通过 0.01%。输入、axis、两个单体 mesh、AABB 和面积全部通过。

这表明 native Poisson 的 vertex count 与 triangle count 都属于非确定的离散采样结果。要求其中一个 exact、另一个 bounded 在方法上不一致。几何重放合同应统一为：生成输入 exact，离散数量 bounded，实际 surface geometry bounded。

## 正式执行事实

- 81/81 unit、15/15 E1 external 和 preflight 0/0 通过；
- 正式执行 `541.012 s`，封存前 `214,199,518 bytes`；
- 完成 S01--S07 共 14 个 native mesh、7 份 metrics 和 7 张完整 train 图；
- S08--S10 未生成，无 retry、replacement、repair 或阈值修改；
- 130/130 evidence SHA-256 复核通过；
- LiDAR、label、formal dataset、training、model、trajectory、simulator 和 M-TARE change 均为 0。

## S07 核验

- 完整 X-Y/X-Z 图连续覆盖全部中心线、连接段和结构事件，无断裂、空白或坐标错误；
- primary/replay 都是一个 triangle component，largest fraction=`1.0`；
- 两者均零 degenerate triangle，全部 axis/spline/AABB 单体检查通过；
- AABB endpoint maximum difference=`0.092010 m`；
- surface-area relative difference=`0.013279%`；
- nearest-vertex maximum=`0.598829/0.600387 m`，仅作诊断；
- point-to-surface maximum=`0.462879/0.511900 m`，正式 PASS。

## 推荐 M0R4（尚未批准）

保持 M0R3 的十个 train parent、seed、native method、point-to-surface `0.75 m`、triangle-count `0.01%`、其他质量条件、资源和所有零范围不变。只把：

- `vertex_count_equal` 改为 `vertex_count_relative_difference <=0.0001`（0.01%）。

形成统一且完整的 replay 定义：

1. graph/spline/operation trace/effective parameters/axis exact；
2. vertex 和 triangle count 各自 relative difference <=0.01%；
3. 双向 vertex-to-opposite-triangle-surface maximum <=0.75 m；
4. AABB endpoint <=0.5 m、surface area difference <=1%；
5. 每个 mesh 独立通过 component、degenerate、axis 和 spline 可读性合同；
6. OBJ hash、nearest-vertex statistics 和实际 counts 全部记录，但不作为 identity。

0.01% 与已经冻结的 triangle discretization tolerance 相同，不新增另一套阈值；S07 观察值为 0.0006406%。该校准仍只使用 train sentinel，不读取 validation/development-test。

M0R4 必须另获批准并从头执行；M0R3 失败资产不拼接复用。
