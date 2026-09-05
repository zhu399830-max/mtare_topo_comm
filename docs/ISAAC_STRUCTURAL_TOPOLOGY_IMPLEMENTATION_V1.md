# Isaac 地下结构语义拓扑实施蓝图 V1

状态：`SUPERSEDED_ON_CRITICAL_PATH_RETAINED_AS_HISTORICAL_REFERENCE`  
日期：2026-08-10  
当前 Gate：Gate 0，`GATE_MIXED`  
实施授权：无。本文件不授权下载第三方代码、启动 Isaac/Gazebo、生成数据或训练。2026-08-10 用户决定不再让 Isaac 阻塞主线；Phase 1--3 的当前传感器后端以 `docs/CPU_RAYCAST_LIDAR_BACKEND_V1.md` 和 `docs/PLAN.md` 为准。本文件中的 Isaac 设计只保留为历史与未来可选域实验参考。

## 1. 固定系统路线

```text
被许可的论文生成器，或按论文独立复现的 graph-to-mesh
  -> topology graph + centerline + common collision/render mesh
  -> Isaac USD stage + canonical LiDAR
  -> raw causal episodes + oracle geometry
  ├─ B1: Cano-like range-image exit CNN
  └─ M1: causal BEV -> R(theta), D(theta), U(theta)
       -> deterministic structural semantics
       -> persistent node/edge/exit-stub graph
       -> graph frontier selection and multi-robot allocation
       -> /way_point
       -> retained M-TARE localPlanner/pathFollower/control
```

第一篇程序化生成工作提供场景生成思想；Cano 等人的出口 CNN 是必须复现的最近 baseline；本项目的研究对象是 planner-consistent 局部结构事实如何形成稳定图并改善 M-TARE 高层探索。程序化生成器、感知、建图和规划必须分层，不做一个无法归因的端到端网络。

## 2. 第三方生成器获取决策

### 2.1 当前已核实事实

- 2022 年论文公开指向 `github.com/LorenzoCanoAn/gzb-subt-env-proc-gen`，描述的是用 DARPA SubT tiles 组装 Gazebo world 的生成器；
- 2024 年 IROS 工作描述的是更通用的 graph-to-mesh 工具，生成 mesh 后可导入目标仿真器；
- 2026 年 JFR 感知工作使用的是 2024 工具生成的 50 个程序化环境；
- 2024 工具的独立公开仓库仍未定位；2022 论文仓库当前不能匿名克隆，公开检索也没有找到可信 fork/归档，因此 commit、LICENSE 和依赖无法核实，G1 复用被阻断。

来源：

- 2022 Gazebo 生成论文：https://doi.org/10.1007/978-3-031-21062-4_26
- 2024 graph-to-mesh 论文：https://doi.org/10.1109/IROS58592.2024.10801552
- 2026 感知与拓扑导航：https://doi.org/10.1002/rob.70157

### 2.2 允许的两条实现路径

`G1_LICENSED_REUSE`：只有取得仓库、固定 commit、保存 LICENSE、确认依赖和论文引用要求后，才可将第三方代码放在 `external/`，通过 adapter 调用；不修改来源历史，不把第三方代码声称为本项目贡献。

`G2_INDEPENDENT_REIMPLEMENTATION`：若新版代码未公开或许可证不明确，则只依据论文公开算法独立实现 graph-to-mesh。实现必须记录参考段落、独立设计决策、测试和差异，不能复制来源不明代码。推荐将此作为当前默认准备路线。

任何许可证不明确、仅在论文中出现但拿不到代码、或仅能非研究用途使用的情况都阻断 `G1`。用户明确说“使用别人生成器”不等于自动获得软件许可。

## 3. 生成器统一输出合同

无论 G1 还是 G2，后续模块只接收以下中立 bundle：

```text
world_<id>/
  topology.json
  centerlines.parquet
  render_mesh.obj|ply
  collision_mesh.obj|ply
  navigable_surface.ply
  gazebo/world.sdf
  isaac/stage.usd
  generation.json
  hashes.json
```

`topology.json`：node/edge ID、degree、边长、中心线、预期出口、loop/dead-end/chamber 等生成事实。`generation.json`：seed、生成器 commit/version、参数、单位、坐标系和导出器版本。

生成必须以 TNG 为隐藏父对象，并分离 `topology_seed/geometry_seed/clutter_seed/trajectory_seed/sensor_seed`。同一 TNG 的全部后代必须位于同一 split。每个 TNG 至少提供两个通过 robot-footprint connectivity 与 planner rollout 认证的 topology-preserving geometry variant；改变可通连接的 geometry 禁止作为不变性正样本。完整定义见 `docs/TNG_COUNTERFACTUAL_DATA_CONTRACT_V1.md`。

统一约束：米、右手坐标、Z-up；所有转换写显式 4×4 matrix。render 和 collision mesh 必须共享同一中心线与拓扑 hash。Isaac USD 与 Gazebo SDF 的 canonical geometry hash 必须一致，视觉材质可不同但不可改变 collision。

生成器 validator 必须检查：

- seed 重放的图、中心线和 mesh hash 一致；
- 无 NaN/Inf、零面积面、严重自交和未解释的非流形区域；
- 每条 graph edge 对应一个连续可通 tunnel component；
- node degree 与局部几何出口数量一致；
- robot footprint 膨胀后仍存在有效 navigable path；
- collision/render/teacher 使用同一尺度和 frame；
- Isaac 与 Gazebo 的固定 landmark 坐标误差小于批准阈值。

## 4. Isaac 接入层

项目当前没有 Isaac 代码。第一版新增实现应位于：

```text
src/mtare_topo/data/worldgen/
src/mtare_topo/data/isaac/
tools/v3/worldgen/
configs/v3/gate1/worldgen/
tests/v3/worldgen/
```

### 4.1 两种采集模式

`STATIC_POSE`：复现 Cano baseline。沿中心线选 place，加入受约束的 lateral/yaw/roll/pitch 扰动，采一帧 LiDAR；用于 range-image、单帧 BEV 和标签对齐。

`CAUSAL_EPISODE`：训练我们的模型。沿同一可执行轨迹按冻结频率采连续 scan、pose、ray origin 和 timestamp。历史必须来自真实先后时序，禁止把互不相关的随机静态 pose 拼成“4 s 历史”。

### 4.2 传感器合同

先审计并冻结保留的 M-TARE LiDAR 参数，再配置 Isaac RTX LiDAR：rings、vertical angles、horizontal resolution、rotation rate、range、minimum range、noise、return policy 和 sensor frame 必须进入 hash。

Cano baseline 的论文输入是 VLP-16 风格 `16×720`、50 m range image。若冻结的 M-TARE sensor 与之相同，直接复现；若不同，B1 必须从同一 canonical LiDAR 栅格化并明确称为 `Cano-like adapted baseline`，不能为了复制论文而让 B1 和 M1 使用不同传感器。

Isaac-Gazebo sensor parity pilot 至少在 20 个固定 pose 比较：有效点数、每 ring range 分布、最近障碍距离、自由射线、坐标变换和固定 landmark。系统差异超过预设阈值时停止批量生成。

### 4.3 原始数据与容量

原始数据按 episode-level Zarr 保存点云、offset、pose、ray origin、timestamp 和 provenance。派生 range image 小，可缓存；`16×160×160` float16 BEV 约 0.78 MiB/observation，50 万条约 381 GiB，禁止默认保存全量未压缩 BEV。首选训练时从 raw episode 确定性构建，或保存可重建的压缩 shard cache。

## 5. 数据规模和 split

### 5.1 Contract pilot

5 个 TNG topology parent，每个 2 个认证 geometry variant，共 10 个 mesh world；每个 TNG 50 个 matched canonical anchor，每个 geometry-specific anchor 3 个 view，共 1,500 observation。只检查生成、传感器、标签、配对、重放和可视化，不训练。

### 5.2 正式容量上限

| Split | TNG parent | Mesh/TNG | Anchor/TNG | View/geometry-anchor | Observation |
|---|---:|---:|---:|---:|---:|
| `SYN_TRAIN` | 80 | 2 | 500 | 5 | 400,000 |
| `SYN_VAL` | 10 | 2 | 500 | 5 | 50,000 |
| `SYN_DEV_TEST` | 10 | 2 | 500 | 5 | 50,000 |
| 总计 | 100 | 200 mesh | 50,000 canonical anchor | — | 500,000 |

正式训练采用 20、40、80 个 train TNG parent 的嵌套学习曲线，对应约 10 万、20 万和 40 万 train observation。增加数据时增加 topology family/topology seed，不能优先增加同 TNG 的相邻帧或 geometry 数量。

另外预留至少两个 sealed topology family，每 family 候选 10 个 world，不计入 50 万开发数据。M-TARE 官方五图和项目现有 underground benchmark 都不进入这些 split。

## 6. B1：对标论文复现

输入是同一 canonical LiDAR 的 range image；目标是在 360 个角度 bin 的真实出口方向放 Gaussian peak。保存论文原设定复现配置：Adam、初始学习率 `4e-5`、epoch decay `0.999`、MSE、batch 64、最多 512 epochs。由于我们的 split 是 world-disjoint，不能照搬论文可能不同的 split 结论。

B1 必须先通过：

- 256 样本 overfit；
- 旋转输入对应 circular shift；
- held-out world 出口 angle error、precision/recall；
- 参数量、推理频率和失败案例复现；
- 不接拓扑图时只评价感知，不能用规划结果掩盖感知失败。

## 7. M1：结构语义主模型

### 7.1 第一版只训练 R/D/U

```text
current 8-channel BEV + causal 8-channel BEV
  -> lightweight ResNet/CNN
  -> polar aggregation over 32 directions
  -> R[32], D[32], log_variance[32]
```

- `R`：footprint/planner-consistent traversability；
- `D`：沿有效方向可达距离，按冻结最大距离归一化；
- `U`：aleatoric confidence/calibration；
- `E`：从 R/D/U 经 circular grouping 确定性提取，不另设竞争 head；
- `G_local`：第一版只作 oracle probe，不进入主 loss。真实 traverse 后建立 edge，比让网络猜隐藏连接更可靠。

候选 loss：masked focal/BCE for R、masked Smooth-L1 for D、heteroscedastic NLL/calibration for U。第一轮不加 AI role、contrastive、GRU、node score、waypoint 或 planning loss。

首个训练 run 的候选 optimizer 为 AdamW、`lr=3e-4`、`weight_decay=1e-4`、effective batch 32、mixed precision、cosine schedule、最多 100 epochs、world-disjoint validation early stop。运行前必须根据实际 GPU 内存写获批 run spec；只能通过 gradient accumulation 改物理 batch，保持 effective batch。

### 7.2 结构语义由冻结解释器产生

连续出口 component 给出 bearing、width、reachable distance 和 confidence。多帧持续后，解释器产生：

- `corridor`：两个近似相对且稳定的出口；
- `turn`：两个稳定但非相对出口；
- `branch`：三个以上稳定出口；
- `dead_end`：进入后仅保留来向出口且局部空间终止；
- `chamber`：宽自由区域与多出口共同满足；
- `bottleneck/width_transition`：沿行进方向 clearance 显著变化；
- `slope/multi_height`：terrain-relative elevation 证据支持。

所有阈值在 `SYN_VAL` 冻结。单帧类别不能直接创建 node，unknown/ambiguous 必须保留。

## 8. 第一版拓扑状态机，不训练图网络

节点状态：`CANDIDATE -> TENTATIVE -> CONFIRMED -> MERGED|RETIRED`。只有持续结构事件、与上一 anchor 的距离和 pose/geometry consistency 同时满足，才确认 structural node；稳定长通道使用 anchor 保持边长受控。

Exit stub 状态沿用：`unobserved -> observed -> attempted -> traversed | temporarily_failed | blocked`。stub 保存 node ID、global bearing、width、distance、confidence、观测次数和最后尝试原因。

机器人实际从 node A 的 stub 行驶并确认到达 node B 后才建立 verified edge。边保存长度、clearance、slope、travel time、失败次数和双向状态。仅因两个节点 role 相似不得 merge；loop edge 必须经过 pose proximity、bearing、局部点云配准或执行轨迹验证。

## 9. 高层规划替换

候选目标仅来自未完成 exit stub。第一版效用：

```text
utility = information_potential
          - lambda_d * graph_travel_cost
          - lambda_r * retry_or_revisit_penalty
          - lambda_u * structural_uncertainty
```

在 confirmed graph 上用 Dijkstra/A* 到目标 node；将目标 stub 转为 local planner 可接受的 map-frame `/way_point`。必须复现原全局节点的 `/free_paths` execution feedback、`/map_clearing` 恢复、completion 和 runtime 契约，不能只发布 waypoint。

多机器人在单机通过后加入。第一版共享增量 node/edge/stub 状态，以中央 Hungarian 分配；断联时 lease 到期后局部选择，重连再做 ID/geometry conflict resolution。

## 10. 逐级验证与停止条件

| 阶段 | 只回答 | 必须证据 | 失败后处理 |
|---|---|---|---|
| V0 Generator | graph 与 mesh 是否一致 | seed replay、degree/edge/path、mesh/collision preview | 修生成器，不采数据 |
| V1 Import | Isaac/Gazebo 是否同源 | hash、landmark、collision parity | 修导出，不采数据 |
| V2 Sensor | LiDAR 是否等价 | 20 pose ring/range/free-ray 对比 | 修 sensor，不扩批量 |
| V3 Teacher | R/D 是否符合 local planner | 分层人工/rollout audit、valid mask | 修 teacher，不训练 |
| V4 B1 | 论文感知能否复现 | angle PR、rotation、held-out world | 定位数据/实现差异 |
| V5 M1 | 结构事实是否更好 | R F1、D MAE、ECE、exit PR | 只修当前感知层 |
| V6 Semantics | 显式角色是否稳定 | event F1、coverage/rotation persistence | 修解释器，不建图 |
| V7 Topology | 图是否正确 | exit preservation、connectivity、redundancy、distortion | 修状态机，不规划 |
| V8 Shadow | 在线是否稳定实时 | latency、graph growth、target stability、contract logs | 不接管控制 |
| V9 Closed loop | 是否优于 M-TARE | paired coverage-time、AUC、path/revisit/stuck | 不进入多机 |
| V10 Multi-robot | 协同是否增益 | team coverage、redundancy、bytes、conflicts | 修分配层 |

科研可视化只保存能核验对应阶段的固定图：world graph/mesh overlay、固定 pose sensor parity、raw/range/BEV/teacher panel、rotation/coverage matched pair、semantic timeline、node-edge-stub overlay、planner decision trace 和 paired coverage curve。每图必须绑定机器指标与样本 ID。

## 11. 当前能力审计与阻塞项

能力审计和镜像准备已执行；没有启动 Compatibility Checker、生成 world/数据或训练模型。当前结果如下：

- 主机已检测到 NVIDIA GeForce RTX 5090 D，显存约 32 GB，驱动版本 580.173.02；
- Docker GPU 透传已在短时本地容器中通过；
- NVIDIA 官方 `nvcr.io/nvidia/isaac-sim:6.0.1` 已拉取，digest 为 `sha256:783444c706538aa76cf5126e911ddc5e618779e6105305ad4af4260362a30aa9`；
- 项目、本机常见安装路径和用户目录中均未检测到 `isaac-sim.sh`、Isaac Lab 或 Isaac Sim 安装目录；
- Compatibility Checker 的启动申请因审批服务断线在进程创建前被拒绝两次，所以结果仍为 `UNKNOWN`，不能认定 Isaac 可运行；
- 批量无界面采集继续采用该固定 headless container，不切换版本。

当前仍有以下阻塞项：

1. 2024 程序化生成器的公开代码仓库尚未定位；2022 论文仓库当前不能匿名访问且无可信归档，当前只能联系作者或按论文独立复现；
2. Isaac Compatibility Checker 尚未实际执行；
3. M-TARE canonical LiDAR 参数、seed 和 planner horizon 未冻结；
4. Isaac-Gazebo common geometry converter 尚不存在；
5. Gate 0 coverage、recorder 和有效 baseline 仍未完成。

因此下一步是执行已下载镜像的 Compatibility Checker，并选择“作者提供合法代码”或“按论文独立复现 TNG-to-mesh”。两者通过后才做一个 TNG/两个 geometry variant 的 import/sensor smoke；不能直接开始 5-TNG pilot。

官方安装依据：

- Workstation：<https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_workstation.html>
- Container：<https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_container.html>
