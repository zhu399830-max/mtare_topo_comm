# Isaac Sim 6.0.1 GMO 输出失败审计 V1

日期：2026-08-10  
状态：`CUSTOM_USDA_VALID / DIRECT_GET_DATA_FAILED / WRITER_V2_CALLBACK_ZERO_FAILED`

## 1. 这一步为什么做

本项目只有先证明冻结的 16×720、10 Hz、0.3--50 m 自定义 RTX LiDAR 能稳定产生一圈完整扫描，后续 Cano mesh 射线验证、客观标签、正式数据集和训练才有合法输入。这个探针不为“凑实验”，而是隔离传感器基础设施，防止把读取接口失败误判为地下场景或学习方法失败。

## 2. 已执行探针与真实结果

证据目录：`results/gate0_baseline/gate0_20260810_cano_exact_profile_creation_probe_v1_seed0/`。

固定范围为 0 个 Cano world、1 个解析封闭盒、1 个 sensor、1 个 pose；正式 dataset、label、training、model、topology 和 M-TARE 修改均为 0。

- `Lidar.create(usd_path=...)` 成功创建 `/World/ProbeLidar`，类型为 `OmniLidar`；
- 69 个 SensorChecker 参数通过；
- 冻结属性逐项回读 0 mismatch，推导为 720 tick/scan、11,520 nominal rays/scan；
- 旧的 `Config not found` 不再出现，证明本地 USDA 已解决 config-name 注册阻塞；
- 直接调用 `LidarSensor.get_data("generic-model-output")` 时，300 个 render frame 均得到非法 GMO header；实际首字为 `0xF4AEEB00`，而官方 GMO magic 应为 `0x4E474D4F`；
- 未生成 `complete_scan.npz`、GMO summary 或 range image，故完整诊断扫描数严格为 0。

最终结论为 `FAIL_EXACT_PROFILE_CREATION_PROBE`。它否决的是当前直接轮询输出路径，不否决 USDA 参数、Cano mesh 射线或 RTX Lidar Core 本身。

## 3. 发现的证据汇总错误

Isaac 脚本已经写出 `capture_failure.json`，但 Kit shutdown 将容器退出码表现为 0。v1 外层封装错误地用退出码推算 `complete_diagnostic_scans=1`。该字段不具备物理真实性；缺失的完整扫描文件、失败 JSON 和最终 FAIL 状态共同证明真实数量为 0。

封存的 v1 目录不回写、不覆盖。后续 runner 已改为只有同时存在完整 NPZ 与状态为 `PASS_EXACT_PROFILE_COMPLETE_SCAN` 的 capture summary 时才计数 1，不能再由进程退出码冒充扫描成功。

## 4. 官方实现对下一步的约束

固定镜像 `nvcr.io/nvidia/isaac-sim:6.0.1@sha256:783444...30aa9` 自带测试 `test_lidar_sensor.py` 使用 Replicator `Writer` 的 `renderProduct` callback 获取 `GenericModelOutput`。官方 Writer 在回调数据中提取 `GenericModelOutput` 后再调用 `parse_generic_model_output_data`，而不是依赖本次失败的同步直接轮询。

因此 v2 只允许改变输出接线：

```text
同一 local USDA + 同一解析盒 + 同一 sensor/pose/profile/backend
→ LidarSensor render product
→ Replicator Writer callback
→ GenericModelOutput parser
→ 首个 complete scan
```

不得改变 sensor 参数、scene、GPU backend、帧上限或验收阈值，也不得换内置型号。Writer v2 代码通过静态编译与单元测试后已获得单独批准并执行一次；结果见下节。

## 5. Writer v2 实际结果

证据目录：`results/gate0_baseline/gate0_20260810_cano_exact_profile_writer_probe_v2_seed0/`。

- preflight 无错误无警告，53/53 V3 单元测试通过；
- local USDA、69 项 SensorChecker 和零差异属性回读再次 PASS；
- 300 个 render frame 内 Writer callback 次数为 0，zero-element 次数也为 0，说明失败发生在 writer 调度/回调之前，不是回调内解析到空点云；
- shutdown 报两次 `Timed out while waiting for pending Replicator writer schedules to drain`；
- `complete_scan.npz`、GMO/capture summary 和 range image 均未生成，正确完整扫描计数为 0；
- run 状态为 `FAIL_EXACT_PROFILE_CREATION_PROBE`，19 项封存证据 hash 全部通过，目录 160 KB；未重跑、未 fallback。

固定镜像自带 standalone `inspect_lidar_gmo.py` 的 Writer 注册、`renderProduct`、时间线和 `simulation_app.update()` 顺序与 v2 一致，源码 SHA-256 为 `fb6ab0...89d46`。因此当前不能简单归因为项目没有照官方 API 写。剩余原因至少包含：本机 headless Replicator writer 调度异常，或自定义 sensor/scene 与 render product 的耦合异常；现有证据不能在两者间继续细分。

## 6. 下一次材料实验的 PASS/FAIL

下一次不应继续修改或重跑 custom probe。推荐先做一个隔离的官方健康对照：固定镜像内置 `Example_Rotary`、官方简单 cube scene、同一 Writer 模式、最多 300 frame，只统计 callback 与合法 GMO。

1. 若官方 control 仍为 callback=0，则阻塞归入本机/headless Replicator runtime，不再继续 custom profile 调参；
2. 若官方 control 有合法 callback/GMO，则阻塞收敛到 custom USDA 或 scene/render-product coupling，之后必须另立单因素审计；
3. 内置 control 不能替代项目传感器、不能进入数据集、不能支持 Cano 或学习结论；
4. 当前不需要点云图，因为研究问题是 writer 调度健康；日志和 callback/GMO 计数才是直接证据。

提案位于 `configs/v3/gate0/isaac_official_lidar_writer_control_v1.proposal.json`，当前未批准、未实现、未执行。任一结果都不自动恢复 Cano 24 poses，不创建正式数据集，不训练模型。
