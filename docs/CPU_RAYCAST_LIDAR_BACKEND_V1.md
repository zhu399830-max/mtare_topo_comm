# CPU Raycast LiDAR 主数据后端 V1

状态：`AUTHORITATIVE_ROUTE_SINGLE_WORLD_PASS_FIVE_TOPOLOGY_PROPOSAL_PENDING`  
日期：2026-08-10  
适用范围：Phase 1--3 的程序化感知数据；不授权运行、批量或训练。

## 1. 路线决定

用户确认停止让 Isaac 阻塞主线。Isaac 的历史失败继续作为 `FAILURE_EVIDENCE` 保存，未完成的 async Writer v3 提案退休且不执行。Isaac 降级为未来可选的传感器域实验，不再是 Phase 1--3 前置条件。

新的主链路为：

```text
Cano procedural TNG / graph / splines
  -> audited perception mesh
  -> Open3D CPU RaycastingScene
  -> causal 16x720 range image + valid mask + ray origin
  -> objective spline/TNG outgoing-branch supervision
  -> Cano-like small CNN/ResNet18 baseline
  -> unseen-topology structural evaluation
  -> later Online Topometric Graph
  -> later M-TARE global-planner replacement
```

Gazebo 仍承担最终 M-TARE 同条件闭环测试。正式训练之前必须另做固定 pose 的 CPU synthetic ↔ Gazebo LiDAR parity；真实部署前还必须验证真实传感器域。CPU PASS 不能冒充 Gazebo 或真实 LiDAR PASS。

## 2. 为什么不是直接使用 Cano 表面点云

Cano 生成器的 point cloud 是为构造隧道表面与 Poisson mesh 服务的完整环境表面，不是机器人在某一时刻可见的局部 LiDAR。把完整表面点云直接输入模型会泄漏遮挡后结构和全局信息。

因此学生输入必须从明确的机器人位姿和 ray origin 发射射线，只保留第一返回与未命中 mask。完整 mesh、TNG 和 splines 只允许用于离线 teacher、质量审计和 provenance。

### 2.1 直接复用与 adapted reproduction 的边界

| 能力 | 当前证据 | 身份 |
|---|---|---|
| Cano TNG、表面点云、native Poisson mesh | 固定 commit 与 E1 环境已运行，项目 read-only adapter 可导出 graph/splines/mesh | 直接复用 baseline 工具 |
| 16×720 CPU first-return LiDAR | 项目已有 Open3D 0.19 实现与 24 个诊断 reference | 直接复用项目基础设施，需新合同 |
| 5 m spline outgoing-branch 标签 | 现有 deterministic 实现与单元测试 | 直接复用 objective teacher |
| 论文出口 CNN 训练程序 | 当前 generator checkout 未找到训练代码 | 不得声称直接代码复现 |
| 小 CNN/ResNet18 range-image exit baseline | 可按论文输入输出与训练设定实现 | `Cano-like adapted reproduction` |
| planner-consistent R/D/U、stable exit-stub graph、M-TARE global replacement、多机器人分配 | 原 baseline 不提供 | 本项目二次创新候选，必须逐 Gate 验证 |

许可证边界仍有效：upstream `pyproject.toml` 有 MIT classifier，但固定 commit 没有 LICENSE 正文。当前只允许本地研究运行和 adapter 调用；第三方源码复制、修改后发布和再分发继续阻塞。

## 3. 第一版传感器合同

```text
backend: Open3D 0.19.0 CPU RaycastingScene
scan: 16 elevations × 720 azimuth columns = 11,520 rays
elevation: -15° ... +15°, step 2°
azimuth: 0° ... 359.5°, step 0.5°
range: 0.3--50.0 m
return: first hit only
frame: right-handed robot frame, x forward, y left, z up
output: range_m[16,720], valid_mask[16,720], local_directions, ray_origin, pose/yaw metadata
noise: none in the contract baseline
```

未命中统一写 `range_m=50.0, valid_mask=0`，不得与真实 50 m 命中混淆。有效 range 必须有限且落在合同范围内。第一版不模拟 intensity、rolling scan、motion distortion、material reflectance、multi-return 或随机 dropout；这些只能在 CPU↔Gazebo parity 后作为单独 domain-randomization 消融加入。

## 4. 客观标签

当前 baseline 标签不是 AI 标注。对 pose 对应的轴线 anchor，以 5 m 三维球与所有冻结 spline 求交，合并 8° 内重复方向，转换到机器人坐标；在 720 个方位 bin 上生成 sigma=3° 的 circular Gaussian peak。

保存离散 heading、branch count、原始交点和 720-bin label。graph/splines 不能进入学生输入。AI 高层结构属性仍是 objective baseline 通过后的可选辅助消融。

## 5. 几何资产边界

现有 Cano native Poisson mesh 已知非 edge-manifold、非 watertight、不可定向且 self-intersecting，因此永久禁止把它称为 navigation-grade collision asset。

但 Phase 计划已经把资格拆为：

- `TOPOLOGY_GT_GATE`：Cano graph/splines 已通过单图审计；
- `LIDAR_SENSOR_GATE`：由本 CPU raycast 合同单独判断；
- `DYNAMIC_NAVIGATION_GATE`：Cano native mesh 已失败，未来 Gazebo world 必须使用通过审计的 project navigation-grade geometry 或其他合格 collision backend。

如果 native mesh 的 CPU LiDAR 合同失败，下一备选不是降阈值，而是单独实现 `Cano graph/splines -> project navigation-grade geometry`，保持 pose、sensor 和 teacher 不变做 A/B。

## 6. 单图合同实验

提案：`configs/v3/gate0/cano_cpu_raycast_lidar_contract_v1.proposal.json`  
审批卡：`configs/v3/gate0/data_cards/cano_cpu_raycast_lidar_contract_v1.json`

只使用已有 `cano_audited_adapter_seed0_world_000`，不生成地图。24 poses 固定为 8 tunnel interior、8 junction transition、8 terminal approach。每轮 276,480 rays，使用两次独立 scene 构建验证确定性。

旧失败 run 中已有 24 个 CPU reference：valid ratio 为 0.990972--1.0，均值 0.997070，最小水平净空 2.058 m，24/24 分支 LOS 通过。这些只作为只读回归 baseline，不升级为 V3 数据。

实验必须展示完整 world pose map 和全部 24 张同尺度 range/label 图；不能选最好看的样本。PASS 只授权另行审批的 5-topology contract pilot，不授权训练。

执行结果为 `PASS_CANO_CPU_RAYCAST_LIDAR_CONTRACT`：24/24 pose、双独立 scene、旧参考回归、range/shape、净空与 branch LOS 全部通过；46/46 封存 hash 通过。结果目录为 `results/gate0_baseline/gate0_20260810_cano_cpu_raycast_lidar_contract_v1_seed0/`。该 PASS 仍不改变 native mesh 的非导航身份。

## 7. 下一步五拓扑感知合同修正

待审提案：`configs/v3/gate0/cano_five_topology_cpu_contract_pilot_v1.proposal.json`。

旧草案要求 5 个 TNG 每个 2 个 topology-preserving geometry，并用 footprint/planner rollout 认证。但当前 native mesh 不能作为 collision/navigation asset，因此现在无法诚实完成这种 operational-connectivity 认证。即时 pilot 修正为 5 个 TNG parent、每个 1 个 native perception mesh、每个 50 个 canonical anchor、每 anchor 3 个固定 yaw view，共 5 mesh、250 anchor、750 个诊断 observation；不训练。

两 geometry 的 paired invariance pilot 不删除，但延后到 navigation-grade collision backend 建立后。当前五拓扑提案仍为 `PENDING_USER_APPROVAL_NOT_IMPLEMENTED_NOT_EXECUTED`，不授权生成。
