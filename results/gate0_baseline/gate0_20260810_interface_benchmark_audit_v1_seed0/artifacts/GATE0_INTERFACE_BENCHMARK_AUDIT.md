# Gate 0 接口与地下 Benchmark 审计（待用户确认）

本轮结论不是“Gate 0 已通过”，而是把下一步真正要解决的问题定位清楚了：全局替换边界已有源码证据，但严格测试世界、随机种子、统一覆盖率和有效重复基线仍不成立，因此当前状态保持 `GATE_MIXED`，没有启动仿真、数据构建或训练。

## 1. 到底替换什么

正式替换单元应是 ROS 节点 `tare_planner_node`，不是整个 M-TARE/CMU 导航栈。该节点内部的滚动地图、keypose/grid、frontier/viewpoint、全局路线与目标选择都被替换；仿真、注册点云、扫描时刻位姿、地形分析、独立的 `localPlanner`、`pathFollower` 和 `/cmd_vel` 控制全部保留。

必须特别区分两个名字相近的模块：M-TARE 源码内部的 `local_coverage_planner` 属于 `tare_planner_node`，会随全局节点一起被替换；独立 ROS 包 `local_planner` 的 `localPlanner` 才是后续公平比较中保留的局部避障器。

闭环主链是：

```text
/registered_scan + /state_estimation_at_scan
        + causal terrain / execution feedback
                         |
                         v
       V3 semantic-topological global planner
                         |
                  /way_point (map)
                         |
                         v
      retained localPlanner + /terrain_map
                         |
                   /path (vehicle)
                         |
                         v
              retained pathFollower
                         |
                    /cmd_vel
```

旧 semantic node 只接了点云和位姿并发布 `/way_point`。它没有接 `/free_paths`，也没有复现原 TARE 的 `/map_clearing` 恢复逻辑、`exploration_finish` 和 `/runtime`。因此过去“网络预测方向—直接发 waypoint”的实现并没有完整替换原全局节点的运行契约，这很可能参与造成卡死、重复目标和结果无法解释。

详细字段、源码哈希和证据见 `interface_contract_draft.json`。

## 2. 世界怎么分

建议把 `tunnel`、`unseen_mine`、`external_cave` 作为三个主开发世界；把 SubTGraph operational 01/02 作为修好 spawn/基线后加入的开发回归世界；`garage` 只做接口 smoke，不进入地下主结论。`indoor`、`campus`、`forest` 从地下 benchmark 排除。

现有世界中严格测试世界数量是 **0**。原因不是它们地下特征不够，而是旧模型和规划器开发已经看过这些世界的轨迹、图、指标或失败结果。历史 protocol 中的 `unseen`、`blind`、`training forbidden` 只能说明当时的约束，不能在结果被反复用于修复后恢复成 V3 strict test。

所以必须新增两个真正隔离的地下世界槽位。可以采购/引入新的公开世界，也可以独立程序生成，但在冻结 hash 后不能被训练、SSL、归一化、teacher/阈值、增强或 checkpoint 选择接触。

详细污染证据见 `world_inventory.json`。

## 3. 旧 baseline 能不能用

- tunnel 180 s、unseen mine v3 180 s、external cave 600 s 有明确运动，可作为接口和指标开发的 `VALID_CANDIDATE`。
- garage v1 和 unseen mine v1 不完整，判为 `INVALID`。
- SubTGraph operational 01 的原 TARE 600 s 路径为 0 m，判为 `INVALID_NO_MOTION`，不能把它当成 TARE 性能差。
- SubTGraph operational 02 没有原 TARE baseline。
- 其余 bag 因缺少统一指标、完整接口日志或 seed，只能判为 `INSUFFICIENT_EVIDENCE`。

没有任何旧 run 可以直接进入最终 mean/std。正式协议拟定为每方法、每世界 5 次、600 s，但现在仍不能运行，因为 seed 尚未可控。

## 4. 新发现的 seed 问题

`vehicle_simulator.launch` 没有 seed 参数；LiDAR 插件使用非零高斯噪声和 `rand()`，TARE 中也存在 `std::random_device`/`rand()`。所以旧配置中的 `UNCONFIRMED_SIMULATION_RUNTIME_RANDOMNESS` 不是文档小问题，而是公平重复实验的阻塞项。

下一步必须先实现一个覆盖 Gazebo、LiDAR 和规划器随机源的统一 seed，并验证相同 seed 可复现、不同 seed 确实产生合法扰动。只有之后才能填写 5 个具体 seed，不能把 run 编号伪装成 seed。

## 5. 覆盖率为什么还不能冻结

旧 `analyze_exploration_bag.py` 能计算 2 m 轨迹格子和已观测 scan 格子，但没有世界可达表面的分母。它适合诊断运动、重访和观察规模，不足以成为跨世界主覆盖率。

建议主指标改为“已直接观测的 0.5 m 参考表面体素 / 该世界可达参考表面体素”，报告 600 s coverage-time 曲线及归一化 AUC。这个定义仍是草案：参考表面和 reachable mask 生成器尚未实现，必须先在开发世界可视化核验，确认不会把封闭墙背面、不可达洞室或世界外网格算入分母。

## 6. 下一步执行顺序

1. 用户确认或修改本轮的替换边界、开发世界划分和“新增两个 strict test”的原则。
2. 只实现 Gate-0 公共基础设施：确定性 seed、完整 recorder/health/INVALID 判定、参考 coverage 生成与可视化核验、SubTGraph spawn/no-motion 诊断。
3. 把三个开发世界和两个新 strict-test world 的 hash、spawn、seed、600 s、5 次重复、指标实现冻结成正式 benchmark。
4. 用户再次批准后，才运行原 M-TARE 重复 baseline。
5. baseline 有效后才讨论 Gate 0 是否 PASS；之后 Gate 1 才提交原始点云数据卡，绝不回到旧 `.npz` 上继续补模型。

本轮还发现三个旧 `semantic_topology_garage` 容器已运行约两天，以及项目根 `.git` 为空目录。前者在任何新仿真前必须隔离/停止，后者意味着当前根项目无法可靠追踪变更；本轮都没有擅自修改。
