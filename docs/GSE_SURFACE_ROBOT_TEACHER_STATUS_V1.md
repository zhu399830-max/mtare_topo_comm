# 面片教师：机器人合同核实与三态软件边界

日期：2026-09-07。对应 Phase 3 的问题：相同五帧观测下，面片关系能否改善开口、稳定结构锚点和穿越连接。这里只完成配置核实与合成软件，**没有新增真实标签、扫描读取、训练或研究成绩**。

## 核实结论及立即采用的备案

找到了旧部署容器中的机器人描述和实际规划器配置，不再重复寻找同一批参数。能够确定规划器如何使用参数；**无法据此绑定真实三维碰撞包络、最大安全坡度、最大可跨台阶及现有采样位姿对应的地面支撑**。

按照已批准执行规格的第一项备案：停止真实 physical teacher 生成，继续面片、可见开口和软件。此时不得宣传地面可通行、完整根可达或者安全行驶成功。根代理已收到证据并采用此备案；不需要重复常规批准。这不是宣布现有数据不能学习结构，也不是要求重造所有地图。

## 读取范围与可复核来源

只读容器 `semantic_topology_garage_30834`，查询时状态 `Exited (137) 2 weeks ago`，镜像 `mtare-semantic-runtime:local`。没有启动容器、ROS、Gazebo 或仿真。以下均为机器人配置、URDF 和算法源码，不是 world、benchmark、LiDAR 或 checkpoint。

容器路径前缀：

```text
/home/docker-user/mtare/autonomous_exploration_development_environment/src/
```

SHA-256 对应 `docker cp <container>:<file> - | tar -xOf -` 解包后的原文件字节；没有将 tar 容器字节误当源文件哈希。此次没有把这些源码复制成新数据资产。

|前缀后的文件|SHA-256|
|---|---|
|`local_planner/launch/local_planner.launch`|`954db01229f021a93213bc6fe491ba58a07a0bc5d8f484288eed8adda17afbbc`|
|`local_planner/src/localPlanner.cpp`|`0d6b70d32433be5a22f1f34a048c854ebe7169a289c1064e2858c61c308110d6`|
|`vehicle_simulator/urdf/robot.urdf.xacro`|`013a0dbdd5dbb2c18c543215d53c3b8e1c3979d12fa3b108d8c7d269597ee3aa`|
|`vehicle_simulator/src/vehicleSimulator.cpp`|`ed906e39762379074c2fbc4ac8a597403181315626d993fcfa43b6eb4f3c028e`|
|`terrain_analysis/launch/terrain_analysis.launch`|`8b982e2c38ba3d8c5f0110f3bc09be2bb053ecc0a3976c82110d25951cfb7553`|
|`terrain_analysis/src/terrainAnalysis.cpp`|`656eca7761a0728681abc41f9e5d63d65e55445bc7791d94a59e8d4815a13253`|

这些证据绑定该停止容器的文件，不冒充仍在运行的 ROS 参数服务器状态，亦未证明其他容器内容完全相同。

## 参数确切含义

|来源与行号|已核实内容|不能推出的结论|
|---|---|---|
|`robot.urdf.xacro:16–21,31`|车身 **visual** box 为 `0.6×0.4×0.3 m`；相对 `base_origin` 下降 `0.5 m`。全文件各 link 有 visual/inertial，没有 `<collision>`。轮子和支架也是 visual。|不能把车身视觉盒或者加上轮子后的视觉外包络直接当已验证碰撞体。|
|`local_planner.launch:16–17,25`；`localPlanner.cpp:721–722,766–767`|规划配置长宽 `0.6×0.6 m`。源码用它们计算旋转障碍区域；但 `checkRotObstacle=false`。|不是完整三维 swept-body 检查，也不是URDF碰撞包络。前进候选还依赖路径体素对应表；本次未把该表或配置推导为新的安全证明。|
|`local_planner.launch:27–28,32`；`localPlanner.cpp:193,754–758,793`|地形点强度高于 `0.15 m` 计阻挡，`0.1 m` 用于地形代价条件，路径阻挡点阈值是2。|障碍分类高度不能等同最大可跨台阶；点数不足也不证明路径安全。|
|仓库 `configs/v3/gate5/roslaunch/vehicle_simulator_seeded.launch:6,24,48–49,55,73`；`vehicleSimulator.cpp:284–287,384,405–419`|启动配置 `vehicleHeight=.75 m`，源码 `vehicleZ=terrainZ+vehicleHeight`，输出 sensor 原点位置。`maxIncl=30°` 只用于地面拟合斜率超限时将 `terrainValid=false`。|`.75 m` 不是车身高；`30°` 不是车辆最大安全坡度，拟合失败也不等于车辆执行停机。|
|`local_planner.launch:77–78`|`useInclToStop=false`，虽配置 `inclThre=45°`，相关倾斜停机选项未启用。|不能把未启用停机阈值当物理爬坡保证。|
|`terrain_analysis.launch:10,20,22–23`；`terrainAnalysis.cpp:562–572,581–625`|`considerDrop=true` 使相对平面高度取绝对值；实际返回地形点强度是这个高度。`vehicleHeight=1.5 m` 在此模块是保留高度差上限；最少10点；`noDataObstacle=false`。|`1.5 m` 不是另一套车身尺寸；没有数据不能据此自动标障碍。这里的二维地形分析也不自动解决地下堆叠多层归属。|

旧 `configs/learning/topological_semantic_teacher_v1.yaml` 中的碰撞半径、安全裕量等仅是历史教师配置。没有用它补齐当前部署缺口，也没有用 TNG 连接关系替代通行验证。

恢复真实物理标签必须有可追溯的包络/安全裕量、支撑与接地模型、坡度/台阶能力，以及传感器到地面参考的有效绑定；本次不猜数、不新增真实参数。若只能提供规划器接受策略，可以另行明确标成“规划器策略代理”，不能偷换为物理通过。

## 新增纯内存教师融合内核

文件：`src/mtare_topo/teacher/gse_surface_teacher_v1.py`。

```python
target = fuse_ground_path(
    reference=PhysicalReference(path, physical_status, robot_contract, evidence_id),
    observed=FiveFrameSupport(...),
    allow_synthetic=False,
)
```

- `GroundRobotContract` 没有科学数值默认值，要求完整包络、坡度/台阶能力及五项来源绑定；缺合同输出未知。合成值必须显式 `SYNTHETIC_ONLY`，且默认不能输出确定部署标签。
- `GroundPath` 保存有序地面接触状态、层和显式跨层支持关系，不对3D空气做搜索。不根据相同XY、相邻点或TNG身份创建通行边。
- `PhysicalReference` 与 `FiveFrameSupport` 分开。后者严格绑定5个递增因果帧、支持状态、观察证据来源与阻挡类型；不得出现未来帧。
- 正例要求同一条路径的全部地面/转换获得观察支持、跨层转换显式有支撑、物理参考确认通过。
- 负例要求观察到碰撞、断地、坡度或台阶违例，并由物理参考确认。搜索失败、未观测地面及隐藏地图障碍不构成负例。参考和观察冲突保留未知。
- **此负例只对该候选路径成立**：另一条支路可能可走，不能升级为整个开口不可达、端墙、尽头事件、全局无连接或删除探索目标。完整根不可达需要另外的充分阻断证据，本内核没有虚构它。

这里消费的是上游已经形成的证据对象，不伪装已实现“射线→可靠地面”“mesh→连续碰撞检查”或真实开口教师。来源字符串是生产方声明，不是自动验证物理真值或身份的机制。未知物理合同不会阻止纯可见开口接口继续开发。

## 合成结果和唯一下一步

`tests/v3/unit/test_gse_surface_teacher_v1.py`：42项通过，0.15秒；完全内存伪场景、零真实样本/权重/优化器步骤。覆盖三态9种组合、缺合同、合成隔离、观测不变时隐藏地图变化、四种有证据障碍、同XY堆叠、无地面/转换支持、不同路径错配、阻挡支路与可走替代支路并存、因果帧及类型检查。

这只证明融合合同的软件行为，不是三维碰撞/射线积分的几何验收，更不是方法有效性。当前唯一可继续的教师方向是同五帧**可见开口及其观察支持**；地面可通行字段保持未知。面片与图软件可并行推进；实际数据读取/教师导出仍由新精确卡和不可覆盖运行约束，不因这份文档自动开始。
