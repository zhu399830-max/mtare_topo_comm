# Cano 100 Topology Parent 数据源设计 V1

更新时间：2026-08-11  
状态：`PENDING_USER_REVIEW_NOT_IMPLEMENTED_NOT_EXECUTED`

## 1. 为什么不能直接复制五拓扑 pilot

五拓扑 pilot 的 P01--P05 是五个项目侧显式骨架，`topology_seed` 只对既定节点坐标施加小扰动。把每类换 20 个 seed 会得到 100 个几何略有变化、组合骨架仍高度重复的 parent，无法证明结构特征能在未见 topology 泛化。

100-parent 阶段改用固定 checkout 中 Cano 原生的：

```text
TunnelNetwork.add_random_grown_tunnel
TunnelNetwork.add_random_connector_tunnel
```

项目只写 read-only audited adapter，负责固定 RNG、检查每次返回、导出 graph/splines、重放 hash、统计失败和隔离 split；不修改第三方核心。

## 2. 分两步做，不直接烧 100 张 mesh

### Step A：120 candidate → 100 topology parent

本轮只生成 graph/splines，不生成 mesh、anchor、LiDAR、label 或模型。先回答“随机生成器能否提供足够、可重放、真正不同的结构”。

固定 10 个结构层，每层预声明 12 个候选，按候选 ID 顺序保留前 10 个通过机器合同的 parent。每层不足 10 个则整轮 FAIL，不在运行后追加 seed。

| stratum | 维度 | grown | connector | 目标结构 | candidates / retained |
|---|---|---:|---:|---|---:|
| S01 | flat | 4 | 0 | 小型树 | 12 / 10 |
| S02 | 3D | 4 | 0 | 小型坡道树 | 12 / 10 |
| S03 | flat | 4 | 1 | 小型单环 | 12 / 10 |
| S04 | 3D | 4 | 1 | 3D 单环 | 12 / 10 |
| S05 | flat | 6 | 1 | 中型分支 | 12 / 10 |
| S06 | 3D | 6 | 1 | 中型坡道分支 | 12 / 10 |
| S07 | flat | 8 | 2 | 多环网络 | 12 / 10 |
| S08 | 3D | 8 | 2 | 3D 多环网络 | 12 / 10 |
| S09 | flat | 10 | 3 | 复杂平面网络 | 12 / 10 |
| S10 | 3D | 10 | 3 | 复杂多高度网络 | 12 / 10 |

Cano 参数冻结为：grown distance `100--300 m`，collision distance `10 m`，intersection minimum distance `30 m`，minimum intersection angle `30°`，maximum inclination `30°`。每条 grown/connector 只有一次 API 调用和有限内部 trial budget；返回失败即记录 candidate failure，不用无限 `while`，也不忽略 connector 返回值。

### Step B：100 accepted parent → perception mesh

只有 Step A 的结构数量、重放和多样性评审 PASS 后，才另立 proposal 给 100 个 accepted parent 生成 native perception mesh。mesh 失败不能从 Step A 的 20 个 spare 中偷偷补位；应报告 geometry failure 并决定是修 geometry contract 还是取消该 parent。Step B 不在本设计批准范围内。

## 3. 固定 split 与泄漏控制

每个 stratum 的前 10 个机器有效 parent 按 accepted rank 分配：

- rank 1--8：train，共 80；
- rank 9：validation，共 10；
- rank 10：development-test，共 10。

split 原子是 topology parent。未来同一 parent 的全部 geometry、trajectory、anchor 和 LiDAR 必须留在同一 split。M-TARE 官方比较地图、项目地下 benchmark 和 sealed strict test 不参与候选生成、参数选择、阈值、归一化或训练。

本轮 10 张人工复核图只取每个 stratum 的第一个 train parent；validation/test 只运行机器合同并封存，不做人眼开发复核，防止 test 反馈进入生成器调整。

## 4. Step A 验收

单 candidate 有效条件：

- 请求的 grown/connector 数均真实成功并保留，所有返回值完整记录；
- graph 为单连通分量，graph/spline 均有限且 schema 合法；
- connector stratum 的 cycle rank 不小于请求 connector 数；
- 总 centerline length 不小于 `300 m`；
- flat stratum 的 vertical span 不大于 `1e-6 m`；3D stratum不小于 `5 m`；
- 同 seed graph/spline replay hash 完全一致；
- 与五拓扑 pilot、M-TARE benchmark 和本轮其他 accepted parent 的 canonical identity 不重复。

批次 PASS 条件：

- 每层至少 10/12 candidate 有效，固定规则得到正好 100 accepted parent；
- 80/10/10 parent IDs 与 canonical hashes 全部互斥；
- 100 个 canonical graph/spline identities 全部唯一；
- coordinate-free Weisfeiler--Lehman graph hash 至少 40 个，且每层至少 3 个；
- 至少 70/100 parent 含 degree `>=3` 的结构事件；
- 所有 50 个 3D parent 满足 vertical-span 条件；
- 保存全部 120 candidate 的成功/失败原因、结构分布、replay、split manifest 和 10 张 train-only 完整图。

若 WL 多样性或高阶分支覆盖不足，这说明 Cano 随机配方不能直接满足数据需求。停止在 topology source，考虑新增预声明高阶路口/chamber recipe；不能靠看 test、换 seed 或转去训练补救。

## 5. 本轮精确数据和成本

- 候选上限：120 topology constructions + 120 same-seed graph-only replays；
- 保留目标：100 topology parents，80/10/10；
- mesh / anchor / LiDAR observation / ray / teacher label / training sample / model：全部 0；
- 预计：冻结 E1 CPU，最长 2 小时、2 GB；
- 输出位置：新的不可覆盖 `results/gate0_baseline/gate0_20260811_cano_100_topology_parent_candidate_audit_v1_seed0/`；
- 当前只形成设计、proposal 和 pending card，尚未实现 runner，尚未执行。

## 6. 用户需要核验的决定

需要确认的是：是否同意用“10 层 × 12 个预声明 candidate → 每层前 10 个机器有效 parent”的方式，先完成 topology-only 审计。批准后才实现 adapter/runner、测试、preflight，并在执行正式命令前再次报告实际命令与成本；本批准不覆盖 mesh、LiDAR 或训练。
