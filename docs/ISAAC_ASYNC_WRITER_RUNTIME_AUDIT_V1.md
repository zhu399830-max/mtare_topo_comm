# Isaac Writer 异步执行模型审计 V1

日期：2026-08-10  
性质：只读源码审计；GPU、仿真、点云样本、数据、标签、训练、模型均为 0。

## 审计问题

`gate0_20260810_isaac_official_local_usda_writer_control_v2_seed0` 已证明：精确官方本地 USDA 创建成功、Writer attach 成功、同步 standalone 更新 300 frame，但 Writer callback 为 0，shutdown 出现两次 writer-drain timeout。

需要判断这个结果能否直接推出“本机整个 headless Replicator runtime 失效”。

## NVIDIA 主证据

固定镜像：`nvcr.io/nvidia/isaac-sim:6.0.1@sha256:783444c706538aa76cf5126e911ddc5e618779e6105305ad4af4260362a30aa9`

固定文件：`/isaac-sim/exts/isaacsim.sensors.experimental.rtx/isaacsim/sensors/experimental/rtx/tests/test_lidar_sensor.py`

SHA-256：`2f545cfeb3cad87f15f5660f5f3d4179eaf95e179954c4977f9497f7d259ccbb`

关键实现：

- line 59：测试类继承 `omni.kit.test.AsyncTestCase`；
- lines 252--253：`create_new_stage_async()` 后显式等待 `ViewportManager.wait_for_viewport_async()`；
- lines 282--286：通过 `LidarSensor.attach_writer(...)` 挂载 GMO Writer；
- lines 290--293：timeline play 后，每帧使用 `await omni.kit.app.get_app().next_update_async()`；
- lines 262--263：停止 timeline 后再执行一次异步 app update 作为 flush。

该生命周期与 v2 采用的同步 standalone `simulation_app.update()` 不等价。此前核对的官方 `inspect_lidar_gmo.py` 确实支持 v2 的同步写法，但 NVIDIA 实际自动化 Writer 单测提供了另一条尚未在本机验证的执行路径。

## 结论边界

v2 的封存 summary 与其预先声明的 `HOST_OR_HEADLESS_REPLICATOR_RUNTIME_BLOCKED` 分类不回写。后验跨证据解释必须更窄：

```text
已证明：同步 standalone + exact official local USDA + official four-cube scene 下，Writer callback=0。
未证明：NVIDIA 异步 Kit test lifecycle 下 callback 也为 0。
因此未证明：整个 host/headless Replicator runtime 普遍失效。
```

这不是通过事后改阈值挽救失败，而是修正因官方测试执行模型差异造成的归因范围。

## 推荐的唯一下一实验

执行一个单变量 A/B：保持 v2 的镜像、断网、官方 USDA 字节、四 cube scene、sensor pose、Writer、300 updates 和零数据边界不变，只把同步 standalone lifecycle 替换为 NVIDIA 单测采用的 async Kit lifecycle。

提案：`configs/v3/gate0/isaac_official_async_writer_control_v3.proposal.json`。

后续状态：用户决定不再让 Isaac 阻塞主线，本提案已更新为 `RETIRED_BY_USER_ROUTE_DECISION_NOT_IMPLEMENTED_NOT_EXECUTED`。它未实现、未建运行目录、未启动 GPU；本审计继续作为未来可选传感器域排障参考。
