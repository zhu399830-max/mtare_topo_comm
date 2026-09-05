# Cano CPU–Gazebo Fixed-Pose LiDAR Parity V1

状态：`APPROVED_FOR_IMPLEMENTATION_AND_ONE_EXECUTION`  
日期：2026-08-11

## 1. 当前要回答的问题

M1R 已经封存 100 个程序化地下网络的不可变感知 OBJ，但正式训练数据仍不能生成。当前缺口是：Open3D CPU idealized raycast 产生的 16×720 range image，与最终 M-TARE/Gazebo 运行域中的 LiDAR，在相同三角网格、相同位姿和相同射线定义下是否足够一致。

本实验只验证传感器域映射。它不验证模型、结构语义、在线拓扑图、机器人动力学或探索性能。

## 2. 为什么选 Gazebo Classic 而不是 Isaac

- M-TARE 现有运行链在 ROS Noetic + Gazebo Classic 域中，正式高层规划替换最终也必须回到该域比较。
- 宿主机没有 Gazebo 命令，但固定镜像 `mtare-semantic-runtime:local`（image ID `sha256:9819eea...e78f9a`）内有 Gazebo 11.10.2、ROS Noetic、CPU ray sensor 和 `libgazebo_ros_velodyne_laser.so`。
- 插件支持 organized `sensor_msgs/PointCloud2`，含 xyz/ring，能还原 16×720 raster；CPU `ray` 后端无需 GPU。
- Isaac 的同步 Writer 路线已退出关键路径，失败证据继续保留，不在本实验重开。

三个位于旧实验中的活动容器只用于只读环境核查；正式实验必须新建隔离容器，不复用、不停止、不修改旧容器。

## 3. 固定输入

只读使用 M1R 的三个 train parent，不读取 validation 或 development-test：

| parent | 作用 | topology/geometry seed | final OBJ SHA-256 |
|---|---|---|---|
| `S01_flat_tree_small_C01` | 平面、较简单 tree | `620101 / 720101` | `d98801e6...c34` |
| `S06_3d_branch_medium_C01` | 三维、中等分支 | `620601 / 720601` | `dcd5ea38...274` |
| `S10_3d_complex_C01` | 三维、富环复杂 | `621001 / 721001` | `407a7505...87b` |

OBJ 以 `<mesh><uri>` 作为静态 collision geometry，scale=`1 1 1`、world pose=`0 0 0 0 0 0`。Gazebo 与 Open3D 必须读取同一个封存 OBJ 字节；不允许转换、简化、修复、重新 meshing 或复制成另一几何版本。

## 4. 位姿如何选

每个 parent 固定 8 个位姿，共 24 个。全部根据 graph/spline/FTA 在看到任何 LiDAR 前确定，传感器高度为 `axis_z + fta_distance + 1.0 m`，yaw 取局部 spline tangent。

- S01：3 tunnel-interior、2 junction-transition、3 terminal-approach；
- S06：3 tunnel-interior、3 junction-transition、2 terminal-approach；
- S10：2 tunnel-interior、3 junction-transition、3 terminal-approach；
- 总计：8 interior、8 junction、8 terminal。

interior 候选须离 degree!=2 事件至少 12 m，已选位姿间至少 20 m；junction 按 node ID 选择 degree>=3 中心或固定 1 m 入射支路偏移；terminal 按 node ID 在端点向内固定 3 m。候选不足即停止，不允许看扫描后换位姿。

## 5. 两个传感器的同一合同

固定 raster：16 elevations `[-15,-13,...,15]°`，720 azimuths `[0,0.5,...,359.5]°`，near/far=`0.3/50.0 m`，first return，range resolution=`0.001 m`，无噪声、无 dropout、无 motion distortion。

Gazebo 使用 `sensor type="ray"` 和 CPU `libgazebo_ros_velodyne_laser.so`，`organize_cloud=true`。SDF horizontal min/max 必须为 `0°/359.5°`，因为插件按 `i*(max-min)/(719)` 还原角度；直接写 `-π/+π` 会产生 719 个间隔并重复首尾方向，不能与 CPU 的 0.5° grid 对齐。vertical min/max 为 `-15°/+15°`。

每个位姿先丢弃 warm-up scans，再保存 3 个连续静态 Gazebo scans。CPU 保存 1 个 reference。Gazebo NaN 还原为 `range=50 m, valid=0`，有限点按 ring 与由 atan2 得到的 0.5° bin 重排；重复 bin 或缺失维度直接失败。

## 6. 先做解析控制，再测 Cano

同一次批准运行先启动一个封闭 axis-aligned box 控制世界。CPU 和 Gazebo 都与解析 slab-intersection 距离比较，用于验证单位、轴方向、角度顺序、ring 顺序、near/far 和 raster reconstruction。控制失败时不加载 Cano OBJ，直接封存 FAIL。

控制通过后按 S01→S06→S10 顺序，各在独立 Gazebo world/session 中测试 8 个固定 pose。任一 parent/pose 失败立即停止，不跳过、不换图、不改阈值。

## 7. 冻结指标与门槛

每个 pose 都必须满足：

- Gazebo 三次静态 repeat 的 valid mask 完全一致，共同有效射线最大 range 差 `<=0.0011 m`；
- CPU/Gazebo raster shape、frame、ring、azimuth/elevation mapping 全部通过；
- CPU/Gazebo valid-mask agreement `>=0.98`；
- 共同有效射线 range MAE `<=0.05 m`、absolute error P95 `<=0.10 m`、P99 `<=0.25 m`；
- 水平最小有效距离差 `<=0.10 m`；
- 不允许只用全局平均掩盖某一 pose 失败。

同时保存 direct-bin error、±1 azimuth neighbour 的 edge diagnostic、按 elevation/role/parent 的误差分布，但 neighbour metric 只解释三角边界差异，不替代 direct-bin PASS。

这些阈值在看到 Cano Gazebo scan 前冻结。若失败，先按 frame/angle、mesh import、ray backend、非流形边界四类归因；不得直接把阈值放宽或添加噪声训练。

## 8. 输出与可视化

- 1 张完整三-parent pose map；
- 1 张解析 control 的 CPU/Gazebo/analytic 对照图；
- 6 张固定页面，每页 4 个 pose，完整覆盖 24 个 pose；每个 panel 显示 CPU range、Gazebo range、valid mismatch 和 absolute error；
- parent/role/elevation 的误差统计图与全部机器可读 JSON/diagnostic NPZ；
- SDF、sensor description、container/image/plugin hash、命令、日志、RUN_STATE 和 evidence SHA-256。

所有 NPZ 仅为 parity diagnostic，不进入正式数据集。

## 9. 明确不做

正式数据样本=0、训练标签=0、训练样本=0、模型=0、trajectory=0、在线图=0、M-TARE change=0。不开 Gazebo GUI，不运行机器人，不使用 benchmark world，不查看 validation/development-test，不调 teacher/range rule，不训练 CNN/GNN。

## 10. PASS 后才允许什么

PASS 只允许另行提交正式 Data Card：从 80 个 train parent 生成因果 LiDAR student input 与 graph/spline objective outgoing-branch label；validation 只作冻结模型选择，development-test 保持封存。正式数据卡必须重新说明 place、trajectory、样本数、Zarr/manifest 格式和可视化，不能把本 parity 的 24 对扫描并入训练。
