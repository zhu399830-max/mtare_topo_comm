# Cano 100 Parent Perception Mesh Plan V1

状态：`M0_PENDING_USER_APPROVAL_NOT_IMPLEMENTED_NOT_EXECUTED`  
日期：2026-08-11  
冻结上游：`gate0_20260811_cano_100_topology_parent_recipe_reclassification_v2r_seed0`

## 1. 现在已有的输入

V2R 已冻结 100 个互异 topology parent 及 80 train / 10 validation / 10 development-test split。100 个 parent 的中心线总长为 `151410.377 m`，单图范围 `586.797--2980.408 m`，中位数 `1347.694 m`。每个 parent 已在 topology 生成前预留唯一 geometry seed，范围为 `720101--721010`。

这些输入只有 graph、splines、生成 trace 和 split manifest。当前仍没有 100-parent mesh、LiDAR 样本、监督标签或模型。

## 2. 为什么先做 M0，而不是直接生成 100 张

五拓扑 pilot 已证明 Cano native mesh 可用于静态 CPU raycast，但那五张图是显式模板，长度和参数不覆盖当前随机 parent 的 `0.59--2.98 km` 尺度。Cano native Poisson mesh 还已知非 watertight、非 manifold、不可定向，不能作为导航碰撞资产。

因此把 100-parent mesh 分为两个材料阶段：

1. `M0`：只用每个 recipe 的 C01 train parent，共 10 张，验证长随机网络的 mesh 生成、重放、资源和可视化合同；
2. `M1`：M0 经人工与机器复核 PASS 后，另行批准并从头生成全部 100 张正式 perception mesh。

M0 不是训练小数据集，不产生学习样本。它用于防止未经验证就启动预计数 GB 的正式材料生成。M0 结果不得拼入未来正式数据集，M1 不直接复用 M0 mesh。

## 3. M0 的精确输入

固定 10 个 train-only sentinel，禁止根据 mesh 结果换 parent：

| Parent | Recipe | Geometry seed | Centerline m |
|---|---|---:|---:|
| S01_flat_tree_small_C01 | flat g4 c0 | 720101 | 820.267 |
| S02_3d_tree_small_C01 | 3D g4 c0 | 720201 | 845.946 |
| S03_flat_unicyclic_small_C01 | flat g4 c1 | 720301 | 969.063 |
| S04_3d_unicyclic_small_C01 | 3D g4 c1 | 720401 | 964.948 |
| S05_flat_branch_medium_C01 | flat g6 c1 | 720501 | 1392.538 |
| S06_3d_branch_medium_C01 | 3D g6 c1 | 720601 | 1132.858 |
| S07_flat_loop_rich_C01 | flat g8 c2 | 720701 | 2174.547 |
| S08_3d_loop_rich_C01 | 3D g8 c2 | 720801 | 2077.994 |
| S09_flat_complex_C01 | flat g10 c3 | 720901 | 2945.773 |
| S10_3d_complex_C01 | 3D g10 c3 | 721001 | 1868.555 |

合计中心线 `15192.489 m`。Validation 和 development-test parent 不读取 mesh、不渲染、不人工查看。

## 4. M0 方法

对每个 sentinel 做 primary 和 exact replay，共 20 次独立 materialization：

1. 从 sealed V2 trace 按原 topology seed、draw 顺序和冻结 Cano API 重建 network；
2. 重建后的 graph/splines identity 必须与 V2R source identity 完全一致，否则在 meshing 前停止；
3. 切换到该 parent 的 reserved geometry seed；
4. 完全模仿 Cano `generate_environments.py`：先从 `[-2,-1] m` 抽一个 FTA distance，使用 `TunnelNetworkPtClGenParams.random()` 和 `TunnelNetworkMeshGenParams.from_defaults()`；
5. 只调用一次 `compute_all()` 和一次 `save_mesh()`，不得重抽参数、换 seed、修 mesh、删 component 或 keep-best；
6. 保存所有实际 tunnel/intersection point-cloud 参数、mesh 参数、axis、graph/splines identity、OBJ 和资源记录；
7. primary/replay 使用同样输入，比较 graph、spline、effective geometry parameter 和 mesh SHA-256。

该身份是 `native Cano perception mesh`，只服务后续理想化传感器。非 watertight/manifold/self-intersection 风险必须逐图记录，不能被解释成 navigation-grade collision mesh。

## 5. M0 机器验收

每个 primary/replay 必须满足：

- frozen V2R、V2 source seal 和固定 Cano checkout/hash 一致；
- graph/spline canonical identity 与冻结 parent 完全相同；
- OBJ 可读，vertex/triangle 非空且全部有限，triangle index 合法，零面积 triangle 为 0；
- 每个 source tunnel 在 axis 记录中出现，axis 坐标/方向/radius 有限且 radius 为正；
- spline 样本均落入带 10 m margin 的 mesh AABB；
- 最大 triangle component 占比至少 `0.999`；非 manifold、watertight、orientable 和 component 数只如实报告，不据此冒充导航资格；
- primary/replay 的 effective geometry parameters 与 mesh SHA-256 完全一致；
- 10/10 parent 全部通过；任一失败则 M0 FAIL，不换 seed、不补图、不降低门槛。

## 6. 可视化与人工核验

只生成 10 张 primary train parent 完整图，每张固定显示：完整 X-Y、完整 X-Z、graph/splines、mesh surface 的确定性等距下采样、节点/结构事件、尺寸和 provenance。禁止裁剪成局部最好案例，禁止渲染 validation/development-test。

人工逐图核验空白、断裂、孤立大组件、中心线越出表面范围、flat/3D 高度错误和明显的 Poisson 伪封口。机器 PASS 但完整图失败时，项目结论仍为科研 FAIL。

## 7. 精确范围与成本

- train sentinel parents：10；
- same-seed topology reconstructions：20；
- native mesh materializations：20（10 primary + 10 replay）；
- train-only complete previews：10；
- validation/development-test parents read or rendered：0；
- anchors、LiDAR observations、teacher labels、formal dataset samples、training samples、models、trajectories、Gazebo/Isaac/M-TARE changes：全部 0；
- 预计冻结 E1 CPU `<=1 h`，结果 `<=1.5 GiB`。

M0 PASS 只授权提交 M1 的 100-parent mesh 方案；不会自动生成剩余 90 张，不进入 LiDAR 或训练。

## 8. 需要用户核验的决定

是否批准按上述固定 10 个 train sentinel、每图 primary+replay、原生随机 geometry 参数、零重试和零 LiDAR/训练范围，实现并执行一次 M0。批准后才编写 executor/runner、测试和正式 run spec。
