# Cano 100 Parent Perception Mesh M0 Review

状态：`M0_SEALED_FAIL_M0R_SUBSEQUENTLY_EXECUTED_AND_FAILED`  
日期：2026-08-11  
证据：`results/gate0_baseline/gate0_20260811_cano_100_parent_perception_mesh_contract_m0_seed0/`

## 结论

M0 在第一个固定 train sentinel `S01_flat_tree_small_C01` 的 exact OBJ SHA-256 replay 门槛处按计划停止并封存 FAIL。没有继续生成 S02--S10，没有换 seed、补 parent、修改阈值或重试。

这次失败否定的是“Cano/Open3D native Poisson mesh 可以按 seed 得到字节完全相同 OBJ”的假设，不是否定 S01 topology、geometry 参数、axis 或 perception-mesh 可读性。

## 正式执行事实

- 执行前 98/98 单元、12/12 冻结 E1 external、17/17 V2R source、391/391 V2 source 和 preflight 0/0 通过；
- 正式 run 用时 `43.508 s`，封存前 `21,916,055 bytes`；
- 只生成 S01 primary/replay 两个 mesh 和一张完整 train 图；
- runner 状态为 `FAIL_CANO_100_PARENT_PERCEPTION_MESH_CONTRACT_M0`；
- 33/33 evidence SHA-256 复核通过；
- validation/development-test、LiDAR、label、formal data、training、model、trajectory、simulator 和 M-TARE change 均为 0。

## 一致的部分

Primary 和 replay 的以下内容完全一致：

- graph identity；
- spline identity；
- bounded topology operation trace；
- reserved geometry seed；
- 所有 effective tunnel/intersection/mesh 参数；
- axis 数组逐元素完全一致；
- vertex count=`69908`；
- triangle count=`139822`；
- 单 triangle component，largest-component fraction=`1.0`；
- 零 degenerate triangle；
- 所有 source tunnel 均有合法 axis；
- splines 均在 mesh AABB+10 m 内；
- 两个 mesh 各自均通过 M0 的 readability/axis/component 合同。

完整 S01 X-Y/X-Z 图显示 mesh 沿全部中心线连续覆盖，没有空白、断裂或孤立大组件；flat 高度关系正确。

## 不一致的部分

Primary OBJ 为 `10,179,377 bytes`，replay 为 `10,179,135 bytes`，SHA-256 不同。直接 vertex/triangle 数组顺序也不同，因此不能用逐索引坐标差解释几何误差。

把两个 vertex set 做双向最近邻后：

| Metric | Primary→Replay | Replay→Primary |
|---|---:|---:|
| mean nearest vertex distance | 0.217879 m | 0.217737 m |
| p95 | 0.378225 m | 0.378049 m |
| p99 | 0.434952 m | 0.434579 m |
| maximum | 0.599708 m | 0.571740 m |

AABB 端点最大坐标差为 `0.110840 m`；表面积为 `20813.1701/20822.0125 m²`，相对差 `0.00042485`（0.0425%）。

## 根因

上游 `TunnelNetworkMeshGenerator.compute_all()` 先运行 Open3D Poisson reconstruction，再调用 `add_noise_to_mesh()`，对当前 vertex array 的每个坐标独立加入 `Uniform[-0.2,0.2] m` 噪声。Poisson 的并行内部顺序不能保证相同 vertex ordering；即便 RNG 和参数相同，后置噪声也会被分配给不同顶点。

两次噪声分配对同一空间位置的理论坐标差上界为 `0.4 m/axis`，三维欧氏上界为 `sqrt(3)*0.4=0.692820 m`。本次双向最近顶点最大值 `0.599708 m` 落在该方法上界内。因此 exact OBJ hash 不是 native baseline 的正确 replay 指标。

## 推荐 M0R

保持 parent、seed、生成方法、数量、资源和零数据/训练范围不变，只把 replay 判据从“OBJ 字节相同”改为“输入完全相同且输出在上游噪声上界内几何等价”：

- graph/spline/operation trace/effective geometry parameters/axis 继续要求 exact；
- primary/replay 各自继续通过全部 mesh 质量合同；
- vertex count、triangle count 要求相等；
- 双向 nearest-vertex maximum `<=0.75 m`；
- AABB 各坐标端点最大差 `<=0.5 m`；
- surface-area relative difference `<=1%`；
- OBJ SHA-256 继续记录，但不要求相等；
- M1 一旦生成，每张 primary OBJ 以封存 SHA-256 作为后续 LiDAR 的不可变正式资产，不再依赖重新 meshing 获得同一个文件。

`0.75 m` 来自上游三维噪声理论上界 `0.692820 m` 加 `0.057180 m` 数值余量，不是根据 validation/test 调参。本次只观察 train S01，validation/development-test 仍未读取或渲染。

M0R 必须另获用户批准并从头生成同样十个 train sentinel；失败 M0 不回写、不拼接复用。
