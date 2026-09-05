# Gate 7 多机器人闭环准备清单 V1

状态：`DESIGN_ONLY_BLOCKED_BY_GATE6_RESULT`  
日期：2026-08-23

## 当前已实现

- `SharedCausalTopometricGraph`：合并各机器人最新因果图，只接受 trace-verified edges；
- `allocate_shared_frontiers`：确定性的 one-to-one Hungarian 分配；
- `MultiRobotCoordinatorRuntime`：lease、retry、完成状态和通信字节计数；
- 纯 Python 单元测试与 Python 3.8 静态兼容；2026-08-23 当前工作树重新执行 shared graph、allocation、
  coordinator 三组测试为 19/19 PASS。

这些证据只证明算法骨架，不能证明多机器人 ROS/Gazebo 闭环或探索收益。

## Gate 7 正式执行前必须补齐

1. namespaced ROS adapter：每机器人独立 scan、pose、free-path、waypoint、runtime 和 graph snapshot；
2. coordinator transport：snapshot revision/timestamp、重复/乱序包、断联、重连和通信字节；
3. execution feedback：到达、local-planner rejection、失败重试、lease expiry 和 frontier 完成；
4. waypoint handoff：把 shared frontier 转为对应机器人坐标系下有限、可达的 `PointStamped`；
5. planner isolation：证明原 M-TARE global coordination 已关闭，只保留 local planner/path follower；
6. conflict safety：窄通道会车、交叉路径、抢占、inactive robot 和错误图融合时的 fail-safe；
7. 统一 runner/evaluator：1/2/3/4 robots，原 M-TARE、independent-nearest、shared-topology Hungarian 和诊断 Oracle；
8. 正式证据：coverage-time、完成时间、团队/单机器人路径、重复覆盖、目标冲突、通信字节、延迟、成功率、
   per-robot trajectory、nodes/edges/exit states、assignment trace、failure reason、RUN_STATE 和 SHA-256 seal。

## 固定进入顺序

Gate 6 给出正式结论后才允许实施：

```text
ROS namespace + feedback 单测
→ 两机器人短 Gazebo smoke
→ 断联/重连与窄通道安全资格
→ 1/2/3/4 robot development matrix
→ 参数冻结
→ sealed strict-test matrix
```

不得把当前 CPU 单测写成多机器人实验，不得同时运行原 M-TARE 与本方法两套全局协调，也不得在严格测试上
选择分配权重、lease 时长或图融合阈值。
