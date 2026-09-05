# TNG 程序化地下数据与反事实实验合同 V1

状态：`DRAFT_FOR_USER_REVIEW`  
日期：2026-08-10  
适用范围：Gate 0 生成器准备、Gate 1 数据合同、Gate 2 结构表示消融  
执行授权：无。本文件不授权启动 CPU/Gazebo/Isaac 采集、生成 world、导出数据或训练。

## 1. 吸收论文的准确位置

Cano、Tardioli、Mosteo 的 IROS 2024 工作从 Tunnel Network Graph（TNG）开始，通过定制 meshing 生成遵循图结构的地下 mesh，再导入目标模拟器。它在本项目中是数据生成基础设施，不是结构语义模型、拓扑建图算法或探索规划 baseline。

来源：<https://doi.org/10.1109/IROS58592.2024.10801552>。论文事实与本项目推导必须分开记录；下文的 teacher、paired experiment、M-TARE 接入均是本项目设计，不归功于原论文。

## 2. 固定生成因果链

```text
topology_family
  -> topology_seed
  -> TNG = nodes + tunnel graphs + shared intersection nodes
  -> centerline / 3D spline
  -> geometry_seed
  -> cross section + radius + curvature + floor + surface noise
  -> intersection cleanup + watertight/collision meshing
  -> audited native perception mesh + qualified Gazebo collision mesh
  -> CPU Raycast LiDAR + later Gazebo parity
  -> sensor_seed / trajectory_seed
  -> LiDAR observation
```

随机源必须分开，禁止只保存一个无法解释的 `seed`：

- `topology_seed`：决定节点、边、degree、cycle、dead-end 和三维连接；
- `geometry_seed`：在固定 TNG 下决定半径、曲率细节、粗糙度、地面与材质；
- `clutter_seed`：决定不改变永久 TNG 的局部障碍与遮挡；
- `trajectory_seed`：决定可执行轨迹与起点；
- `sensor_seed`：决定传感器噪声与 dropout。

任何阶段失败都必须能只重放该阶段，同时保持上游 hash 不变。

## 3. TNG 与几何的独立控制

TNG primitive 首版包含论文描述的 Randomly Grown Tunnel Graph（RGTG）与 Connector Tunnel Graph（CTG）。RGTG 控制水平/垂直 tendency、noise、基础段长和段长噪声；CTG 在已有节点间增加连接，形成 cycle 和网状结构。实现还必须输出 collision rejection 记录，防止几何交叉与图连接不一致。

每个 TNG 保存至少：

- `tng_id`、`topology_family_id`、`topology_seed`；
- node/edge/TG/intersection ID、三维位置、degree、edge length；
- connected components、cycle rank、dead-end 数、branch-degree histogram；
- slope/vertical-span 分布、最短路径和中心线；
- 生成器版本、参数、父 hash 和验证状态。

几何变体保存 `geometry_variant_id` 与单独 hash。允许变化：隧道半径、截面、曲率细节、Perlin/表面噪声、地面粗糙度、材质和不改变永久连接的遮挡。不得把“同 TNG”直接等同于“同机器人结构”。

## 4. 拓扑保持认证

同一 TNG 的两个 mesh 只有同时通过下列检查，才是 `TOPOLOGY_PRESERVING_PAIR`：

1. canonical TNG hash 相同；
2. graph edge 与连续 tunnel component 一一对应；
3. robot footprint 膨胀后的 operational connectivity 相同；
4. node degree、可达 exit set、cycle rank 和 dead-end set 相同；
5. planner rollout 没有因宽度、坡度、障碍或碰撞 mesh 产生永久断边；
6. 对应 anchor 的传感器姿态、coverage 和观测范围可匹配。

若窄化、坍塌、坡度或障碍改变可通行性，则它是 `CONNECTIVITY_CHANGING_VARIANT`，只能作为 hard negative、鲁棒性测试或动态状态样本，禁止作为不变性正样本。

## 5. 真值、可观测投影与防泄漏

生成器拥有完整 TNG、mesh、axis、node、edge 和 intersection 真值；部署机器人只能看到当前及因果历史 LiDAR。数据导出必须并行保存三层对象：

```text
oracle_global_tng             # 仅 provenance、teacher 和评价
observable_ego_graph_target   # 经射线/范围/遮挡/coverage mask 后的局部投影
student_observation           # LiDAR 派生 current + causal BEV
```

完整 TNG、绝对位置、`tng_id`、geometry variant ID、未来轨迹和不可见连接禁止进入 student tensor。不可观测连接标为 `UNKNOWN/MASKED`，不能当负类。第一版 M1 仍只训练 planner-consistent `R/D/U`；`observable_ego_graph_target` 与 `G_local` 先作为 oracle probe，不能绕过逐层验证直接变成端到端图预测。

## 6. 配对实验

数据 manifest 必须支持以下配对：

- `P_TOPO_SAME_GEOM_DIFF`：同一局部 TNG、不同且已认证的几何变体；
- `P_TOPO_DIFF_VISIBLE_MATCHED`：不同局部 TNG、相似可见几何和相近 coverage；
- `P_COVERAGE_DIFF`：同一 geometry/place、不同时序 coverage；
- `P_ROTATION`：同一样本的可验证 circular equivariance；
- `P_CONNECTIVITY_CHANGED`：父 TNG 相同但 operational connectivity 被明确改变。

必须先用纯监督 M1 评价这些 paired benchmarks。只有 M1 稳定且独立 run/data card 获批，`P_TOPO_SAME_GEOM_DIFF` 才可进入 A1 consistency/contrastive auxiliary。不得用同 TNG ID 作为模型输入或批次捷径。

## 7. Split 原子性与容量口径

split 的最小原子是 `topology_parent`。同一 TNG 的所有 geometry、clutter、trajectory、pose、AI/人工审阅和派生缓存必须进入同一 split。先分配 TNG，再生成 geometry；禁止先生成 mesh 后随机按 observation 切分。

立即执行的 5-TNG perception contract pilot（2026-08-10 单图 CPU PASS 后修订）为：

- 5 个 topology parent；
- 每个 TNG 1 个 native perception mesh，共 5 个 mesh world；
- 每个 TNG 50 个 matched canonical anchor；
- 每个 anchor 3 个固定 yaw view；
- `5 × 50 × 1 × 3 = 750` diagnostic observation；不训练。

修订原因：native Cano mesh 已被动态导航门禁否决，现阶段不能用 footprint/planner rollout 认证两个 native geometry 的 operational connectivity 相同。旧的 2-geometry/1,500-observation 设计保留为后续 paired invariance pilot，但只有 navigation-grade collision backend 和 planner certification 建立后才能单独审批。正式容量设计中的每 TNG 至少两个 geometry variant 同样依赖该前置条件。

正式容量上限保持 500,000 observation，但报告口径改为：

- 80/10/10 个 train/validation/development-test TNG parent；
- 每个 TNG 至少 2 个认证 geometry variant，共 160/20/20 个 mesh realization；
- 每个 TNG 500 个 matched canonical anchor；
- 每个 geometry-specific anchor 5 个 sensor view；
- `100 × 500 × 2 × 5 = 500,000` observation。

因此正式设计是 100 个独立拓扑、200 个 mesh、50,000 个 canonical anchor、100,000 个 geometry-specific place instance 和 500,000 个 observation，不能只报最后一个数字。20/40/80 学习曲线增加的是 train TNG parent，不是同一 TNG 的更多帧。

## 8. 拓扑族与域外测试

开发数据至少分层覆盖：tree-like、single-cycle、multi-cycle/network、branch-heavy、dead-end-heavy、long-corridor、3D-slope/multi-level。每族报告 node/edge、cycle rank、degree、edge length、vertical span 和 operational-width 分布，不能只用名称声明覆盖。

TNG 管状先验对矿井和人工隧道强，对天然洞穴、大 chamber、坍塌、柱体、复杂障碍和非管状空间弱。因此：

- 程序 TNG 数据用于监督训练、学习曲线和受控消融；
- 经过 provenance/许可审计的 LAMP/SubT 完整 site 用于离线域验证或另行批准的适配；
- M-TARE 原地图只用于系统冻结后的公平 benchmark；
- 至少一个 sealed strict-test family 必须是非管状或超出生成器分布的地下环境。

## 9. Gate 验收新增项

V0 Generator 在原检查外新增：TNG/geometry seed 解耦、topology-parent split、paired certification、拓扑统计覆盖和 hidden-TNG leakage test。V5 M1 新增 paired evaluation：同拓扑跨几何稳定性、相似可见几何下不同拓扑可分性，以及 connectivity-changing variant 的敏感性。

这些指标只能支持“结构事实更稳定”的结论；只有后续 V7 图指标和 V9 闭环探索同时改善，才能支持“结构语义拓扑改善规划”。
