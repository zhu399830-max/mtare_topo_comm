# Phase 2 有监督 Range-Image 数据方案 V1

状态：`PENDING_USER_APPROVAL_DESIGN_ONLY`  
日期：2026-08-12  
当前 Gate：Gate 0 `GATE_MIXED`；本文件不授权导出、teacher 生成或训练。

## 1. 唯一研究问题

在不使用 M-TARE benchmark、完整地图或未来信息作为学生输入的前提下，现有 80 个程序化训练拓扑能否生成数量充分、结构事件完整、teacher 可审计的第一版监督数据，用于训练：

```text
16×720 local LiDAR range + valid mask
                 -> local outgoing traversable branch directions
```

第一版不训练 BEV R/D/U、不使用 GNN、不做对比学习、不使用 AI 标签。旧 `GATE1_DATA_METHOD_V1.md` 中把 causal BEV R/D/U 作为第一主模型的规则，由当前权威 `PLAN.md` 第 6--7 节覆盖：先完成 Cano-like range-image exit baseline，BEV R/D/U 只保留为 baseline 通过后的主模型/消融候选。

## 2. 数据来源与严格隔离

唯一来源是封存 M1R：

```text
results/gate0_baseline/
  gate0_20260811_cano_100_parent_perception_mesh_m1r_sanitized_assets_seed0/
```

- train：`S01...S10` 每个 recipe 的 `C01--C08`，共 80 parent；
- validation：`S01...S10` 每个 recipe 的 `C09`，共 10 parent；
- sealed development-test：`S01...S10` 每个 recipe 的 `C10`，共 10 parent，本轮读取计数为 0；
- M-TARE benchmark、LAMP、SubT、历史 NPZ 和五拓扑 diagnostic shard 全部不进入本数据；
- parity 的 7 个 NPZ 永久保持 diagnostic-only，不拼入训练。

只读容量审计得到：

| split | parent | spline长度 | tunnel | junction | terminal | cycle | 3D world |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 80 | 122.177 km | 624 | 563 | 513 | 116 | 40 |
| validation | 10 | 15.826 km | 78 | 71 | 62 | 14 | 5 |
| sealed development-test | 10 | 14.409 km | 78 | 71 | 65 | 15 | 5 |

独立泛化单位是完整 topology parent，不是 frame、Zarr shard、yaw 或 place cluster。

## 3. 采样单位与精确数量

每条 spline 从 arc=`0.5 m` 开始按 1 m 取样；连续五个位置组成一个不重叠的 5 m `place_cluster`。每簇只有一个 tunnel、一个 cluster ID 和五个 frame，frame 的 yaw 固定为当前位置 spline tangent。不同 yaw 对 360° LiDAR 只是循环列移，不重新 raycast、不计作新样本；训练时允许对输入和标签做同量 circular shift，但增强不增加 raw/effective sample count。

### 3.1 必须先确认的弧长口径

冻结 spline 同时含 `distances[]` 和 `points[]`。只读审计发现二者并不等长：`distances[]` 是 Cano `spline.discretize(0.5)` 返回的参数距离，而现有位置插值代码按 `points[]` 相邻点的累计欧氏距离工作。精确差异为：

| split | 旧 `distances[]` 合计 | `points[]` 欧氏弧长 | 差值 | 相对差 |
|---|---:|---:|---:|---:|
| train | 121,386.151 m | 122,176.571 m | 790.421 m | 0.651% |
| validation | 15,721.222 m | 15,826.232 m | 105.010 m | 0.668% |
| sealed development-test（只读历史清单审计） | 14,303.004 m | 14,409.204 m | 106.200 m | 0.742% |

单 world 相对差范围为 `0.227%--1.181%`。这不影响 parent split、结构事件数或候选容量是否足够，但会改变“1 m 间隔”、episode 距离和 per-world quota 的精确含义。

推荐并等待用户明确确认：正式采样统一重算封存 `points[]` 的 float64 累计欧氏弧长；`distances[]` 只作上游 provenance，不参与采样、配额或距离报告。理由是 mesh、pose、raycast 和 teacher 均位于 `points[]` 的实际世界坐标中。若选择旧 `distances[]`，必须另写参数距离插值器并重新做容量审计。确认前不冻结 90-world registry、不实现 exporter、不生成 teacher。

每个 parent 形成一个 `segmented_centerline_episode`：各 tunnel segment 有明确 reset 标志，不伪装成动态可执行机器人轨迹。距离与空间覆盖按唯一 spline 弧长报告；它只服务静态感知，不声称通过 `DYNAMIC_NAVIGATION_GATE`。

候选容量（未生成 LiDAR）：train 24,117 cluster/120,585 frame，validation 3,130 cluster/15,650 frame。固定选择目标：

| split | interior cluster | junction cluster | terminal cluster | place cluster | frame |
|---|---:|---:|---:|---:|---:|
| train | 16,350 | 3,000 | 650 | 20,000 | 100,000 |
| validation | 2,050 | 375 | 75 | 2,500 | 12,500 |
| development-test | 0 | 0 | 0 | 0 | 0 |
| 合计 | 18,400 | 3,375 | 725 | 22,500 | 112,500 |

role 只作分层，不进入学生输入。原草案使用互斥的 `junction > terminal > interior` 优先级；精确事件审计发现这会把同时邻近 junction 与 terminal 的候选全部标成 junction，导致 train `22/513`、validation `8/62` 个 terminal 事件没有 terminal-role 候选。30 个事件均有空间候选，问题完全由互斥优先级造成，不是隧道长度或采样格缺失。

用户已经确认：保存两个非互斥审计标签 `near_junction` 与 `near_terminal`；为精确配额另定义 `primary_role = terminal > junction > interior`。其中 junction 邻近为 cluster center 到任一 degree>=3 node 的欧氏距离 `<=10 m`，terminal 邻近还要求该 tunnel 在 degree=1 node 的 `incident_tunnel_ids` 中。该规则得到候选容量：train `20,007/3,307/803`，validation `2,625/407/98`（interior/junction/terminal），仍能满足原配额。train 563/513 与 validation 71/62 个 junction/terminal 事件均有同 primary-role 候选；每个 junction 至少 3 个、terminal 至少 1 个。

90 个 world 的 points-based 长度、容量、事件数、事件最小候选数和 per-role quota 已冻结于 `configs/v3/gate1/manifests/cano_phase2_world_registry_v1.json`，SHA-256 为 `d64edd758d86d1192dff870cdba08d392388cb48cfd755f428acdfb3ff4e8420`。机器复核确认 80/10、90 个唯一 ID、全部配额闭合、0 个 quota 超容量、0 个 C10 row。

所有候选在扫描前排序并封存。质量失败只能从同 world、同 role 的预声明候选池取下一个；不得添加新位置、换 world、换 seed或降低门槛。配额不足即整次 run FAIL。

## 4. Student 输入

原始存储：

```text
range_m[N,16,720]       float32
valid_mask[N,16,720]    uint8
pose_world_sensor[N,7]  float64      # provenance only
ray_origin_world[N,3]   float64      # provenance/审计 only
place_cluster_id[N]
frame_in_cluster[N]     0..4
```

模型只能读取 `range_m/50.0` 与 `valid_mask` 两个通道；坐标系为右手 robot frame，x forward、y left、z up。范围 0.3--50 m，16 elevations `[-15,-13,...,15]°`，720 azimuth `[0,0.5,...,359.5]°`，first return，无噪声。`world_id`、split、pose、绝对 xyz、TNG、graph、spline、event role 和未来信息禁止进入 student tensor。

第一版不缓存 BEV。若 range baseline PASS 后进入 BEV，必须另立 schema、Data Card 和消融，不能静默改本数据输入。

## 5. Objective teacher

teacher 不是 AI 标注。对每个 axis anchor：

1. 用半径 5 m 三维球与封存 spline 求交；
2. 合并 8° 内重复 heading；
3. 变换到 robot frame；
4. 生成 `heading_deg[]`、`branch_count` 和 sigma=3° 的 `exit_target[720]`；
5. 对每个代表性 5 m branch point 做 mesh ray LOS；碰撞必须不早于 target distance−0.25 m；
6. 保存 oracle raw intersection、LOS、teacher version/hash，但只把通过 LOS 的完整样本放入 loss。

任何不可观测分支都不能填成负类。本 V1 为保持标签简单，要求一个被选 cluster 的五个 frame 全部通过 branch LOS、水平最小净空 `>=0.8 m`、`branch_count` 在 `[1,8]`；否则整个 cluster 失效并按冻结候选顺序替补。若未来希望保留部分不可观测分支并用 direction mask，必须升级 teacher schema，不在本 run 临时修改。

## 6. Baseline 与第一模型

- B0：冻结的 range-sector geometry/rule 出口检测器；
- B1：小 CNN/ResNet18 量级的 Cano-like adapted range-image exit predictor；
- random/circular prior：用于确认模型不是只学到“通常有两个方向”的类别先验。

本数据 run 不训练上述模型。数据 PASS 后另立 Phase 3 训练 proposal，固定 place-cluster-aware sampler，避免同簇五帧在一个 batch 中伪装成五个独立 place。

## 7. 存储与证据

正式存储为 split/world sharded Zarr；`manifest.jsonl` 每帧一行，`place_clusters.jsonl` 每簇一行，`world_manifest.json` 每 world 一条。`.npz` 只允许少量 debug，不作为正式数据。

预览只从 train 固定生成：

- 10 张 recipe coverage 图；
- 10 页分层样本，每页固定 10 个 train cluster；
- 每个 panel 同时显示 range、valid、teacher heading/target、role 和 LOS；
- role/branch-count/elevation/range/valid-ratio 分布图及机器 JSON；
- validation 和 sealed development-test 预览均为 0。

预计未压缩核心数组约 6.84 GiB，Zarr、manifest、teacher、日志和预览总预算 `<=12 GiB`；冻结 E1 CPU 预计 `<=3 h`。不使用 GPU、不启动 Gazebo/Isaac、不训练。

## 8. PASS/FAIL

PASS 必须同时满足：

- M1R 1001 项与 parity 70 项 source seal 无差异；
- train/validation parent 精确为 80/10，C10 与 M-TARE benchmark 读取为 0；
- 22,500 cluster、112,500 frame、角色配额精确；
- train 1,076 和 validation 133 个真实结构事件全部至少覆盖一次；
- 每个 cluster 五帧、1 m 间隔、无跨 tunnel、无重复 frame identity；
- 112,500/112,500 scan shape/dtype/range、净空、teacher、LOS 和独立 scene replay 全通过；
- student tensor 泄漏审计为 0；
- train-only 预览完整、validation/test 预览为 0；
- 运行时间、磁盘、manifest、日志、summary、RUN_STATE 和 SHA seal 完整。

任一项失败即封存 FAIL；不补 world、不改 role 配额、不放宽 LOS/净空、不把失败样本 mask 后凑数。PASS 只证明数据合同合格，不证明 CNN 已学会结构语义。

## 9. 当前审批点

本方案、90-world registry 和完整 Data Card 现在请求最终数据执行审批。批准后才允许：实现 selector/teacher/exporter、补 contract tests、冻结 Gate 1 run spec、preflight，并在再次展示冻结命令与成本后执行一次正式数据导出。训练仍需第二次独立批准。
