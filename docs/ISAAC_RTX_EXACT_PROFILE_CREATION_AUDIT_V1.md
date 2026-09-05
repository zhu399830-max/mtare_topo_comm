# Isaac Sim 6.0.1 自定义 RTX LiDAR 创建路径审计 V1

日期：2026-08-10  
状态：`READ_ONLY_AUDIT_PASS / V1_MATERIAL_PROBE_EXECUTED_FAILED / SUPERSEDED_BY_GMO_WRITER_AUDIT_V1`  
当前 Phase：`PHASE_1_CANO_SENSOR_GATE_REPLICATOR_RUNTIME_CONTROL_PENDING`  

## 1. 本轮只回答什么

本轮不生成 world、不采 Cano 数据、不训练模型、不运行 GPU，只回答：冻结的 `MTARE_VLP16_720_50M_V1` 为什么在 Isaac Sim 6.0.1 中创建失败，以及保持完全相同 profile 的官方最小修复路径是什么。

正式 dataset、train、validation、test、LiDAR sample 和 model 数量均为 0。

## 2. 确定结论

Isaac Sim 6.0.1 的新 RTX API 将 `Lidar.create(config=...)` 中的 `config` 解析为 `SUPPORTED_LIDAR_CONFIGS` 中的 NVIDIA/OEM USD 资产名。它不会按文件名动态发现任意自定义 JSON。旧命令 `IsaacSensorCreateRtxLidar` 在 6.0.1 中也优先用同一 USD 白名单解析 `config`，解析不到后才落到已弃用的 camera/JSON 路径。

因此，先前失败的直接原因不是冻结参数非法，而是把旧式 JSON 文件名当成了新式注册表配置名。把 JSON 挂载到 profile 目录不等于加入 `SUPPORTED_LIDAR_CONFIGS`。

6.0.1 已明确提供自定义资产入口：

```python
Lidar.create(
    path="/World/DiagnosticLidar",
    usd_path="/workspace/configs/v3/gate0/sensors/mtare_vlp16_720_50m_v1.usda",
    accumulate_outputs=True,
    aux_output_level="BASIC",
    tick_rate=10.0,
)
```

推荐采用本地 `.usda` 直载，不修改 NVIDIA 的 `SUPPORTED_LIDAR_CONFIGS`，不修改镜像内扩展，不切换近似内置型号，也不再依赖已弃用 JSON/camera 创建路径。这仍然是同一个 RTX Lidar Core backend，仅改变 profile 的合法载入载体。

## 3. 冻结参数如何映射到 USD

目标 prim 必须是 `OmniLidar`，应用 `OmniSensorGenericLidarCoreAPI`。单个 `s001` emitter state 由该 Core API 自动包含，但数组值必须显式覆盖。

| 冻结含义 | 旧 JSON | USD 属性 | 固定值 |
|---|---|---|---:|
| 旋转扫描 | `scanType` | `omni:sensor:Core:scanType` | `ROTARY` |
| 旋转方向 | `rotationDirection` | `omni:sensor:Core:rotationDirection` | `CCW` |
| 完整扫描频率 | `scanRateBaseHz` | `omni:sensor:Core:scanRateBaseHz` | 10 Hz |
| 每圈方位列数 | `patternFiringRateHz / scanRateBaseHz` | `omni:sensor:Core:patternFiringRateHz` | 7200 / 10 = 720 |
| 垂直通道 | `numberOfEmitters` | `numberOfEmitters` 与 `numberOfChannels` | 16 / 16 |
| 垂直角 | `emitterStates[0].elevationDeg` | `emitterState:s001:elevationDeg` | -15° 到 15°，步长 2° |
| 发射器方位偏置 | `azimuthDeg` | `emitterState:s001:azimuthDeg` | 16 个 0° |
| 发射时差 | `fireTimeNs` | `emitterState:s001:fireTimeNs` | 16 个 0 ns |
| 通道映射 | 旧 JSON 隐式 | `emitterState:s001:channelId` | 1 到 16 |
| 量程 | `nearRangeM/farRangeM` | 同名 USD Core 属性 | 0.3 m / 50 m |
| 回波数 | `maxReturns` | 同名 USD Core 属性 | 1 |
| 射线模型 | `rayType` | 同名 USD Core 属性 | `IDEALIZED` |
| 距离分辨率/精度 | `rangeResolutionM/rangeAccuracyM` | 同名 USD Core 属性 | 0.001 m / 0 m |
| 误差 | 四个 azimuth/elevation error 字段 | 同名 USD Core 属性 | 全 0 |
| 输出坐标系 | 采集器运行时参数 | `outputFrameOfReference` | `SENSOR` |

旧 JSON 的 `minReflectanceRange` 在 USD schema 中名为 `minReflectionRangeM`，`wavelengthNm` 在 USD schema 中名为 `waveLengthNm`。`emitterStateCount=1` 在新 USD schema 中不再是独立属性，而由唯一的 `s001` emitter-state API 表达。`numberOfChannels=16` 和 `channelId=[1..16]` 必须显式写出，不能继承 schema 的 128-channel 默认值。

## 4. 证据来源

审计对象固定为 Docker image：

- tag：`nvcr.io/nvidia/isaac-sim:6.0.1`
- image id：`sha256:2d4ebfef053f740b634e30723e38ed736255f677553425ef35552437d74c3b21`
- repo digest：`sha256:783444c706538aa76cf5126e911ddc5e618779e6105305ad4af4260362a30aa9`

镜像内源码事实：

- `experimental/rtx/impl/lidar.py` 明确说明 `config` 来自 `SUPPORTED_LIDAR_CONFIGS`，同时公开 `usd_path`；SHA-256 `dbb587...fd974`。
- `experimental/rtx/impl/_sensor_base.py` 的 `_create_from_usd` 直接引用本地/远端 USD，并查找 `OmniLidar` prim；SHA-256 `49de5e...a99f0`。
- `experimental/rtx/impl/rtx_lidar_configs.py` 注册的是 `/Isaac/Sensors/.../*.usd[a]` 官方资产；SHA-256 `dac2ae...f2e3`。
- NVIDIA 自测 `test_rtx_lidar_configs.py` 使用 `Lidar.create` 后调用 `SensorCheckerUtil.validateParams`；SHA-256 `5bd01d...1f70`。
- deprecated `impl/commands.py` 也先遍历 USD supported-config registry；SHA-256 `b4d2db...9d11`。
- Lidar schema 文档说明 10 Hz 与 7200 pattern firing rate 对应每圈 720 个 tick，并定义 emitter-state 数组；SHA-256 `4a4e26...8b61`。
- 生成 schema 定义全部 Core 属性、类型与默认值；SHA-256 `08f5b2...e144`。
- 当前冻结 JSON SHA-256：`d9b45461b4f113c25fd5a2226bef50dd83d0fe2e3b1407cd93da9ff7798688fe`。

## 5. 推荐的独立创建探针

探针不是 24-pose smoke，也不是数据集。它只创建 1 个自定义 USD LiDAR，在一个程序化封闭测试盒中取得 1 个完整 RTX scan。10 Hz 完整 scan 可能跨多个 render frame，因此不能把“1 个 scan”误写成“1 个 render frame”。

输入和数量固定为：

- world：0 个 Cano world；1 个仅供传感器自检的解析几何封闭盒；
- sensor：1 个；pose：1 个；完整 scan：1 个；
- nominal rays：16 × 720 = 11,520；实际有效 hit 数由场景遮挡与 RTX 输出规则决定，不预设为 11,520；
- dataset/train/validation/test/model：全部 0；
- backend：Isaac Sim 6.0.1 RTX Lidar Core；不允许 CPU 替代；
- profile：与冻结 JSON 完全一致；不允许内置型号 fallback。

探针 PASS 必须同时满足：

1. `Lidar.create(usd_path=...)` 返回单个有效 `OmniLidar` prim；
2. `SensorCheckerUtil.validateParams` 返回无错误且 validated parameter 数大于 0；
3. 对冻结属性逐项 read-back，类型、数组长度和值全部一致；
4. `numberOfEmitters=16`、`numberOfChannels=16`、`channelId=1..16`；
5. `scanRateBaseHz=10`、`patternFiringRateHz=7200`，解析得到 720 tick/scan；
6. 300 render frame 上限内得到且只保存第 1 个 complete scan，GMO magic/header 合法，point count 大于 0；
7. 日志中不得出现 `Config not found`、invalid GMO 或静默 fallback；
8. 保存完整日志、USD/hash、read-back JSON、checker JSON、GMO summary 和一张仅用于确认盒体命中的 range-image 图。

任一项失败即停止。失败后不得自动改 profile、换内置 LiDAR、改 backend 或恢复 24 poses。

## 6. 后续边界

本审计随后对应的 v1 材料探针已经执行：local USDA 创建、SensorChecker 和精确属性回读通过，但直接 GMO 读取失败，完整扫描为 0。v1 规格与授权不得重用。最新阻塞、证据偏差和 Writer-callback v2 方案见 `docs/ISAAC_RTX_GMO_WRITER_AUDIT_V1.md`。

探针 PASS 后也不能自动恢复 24 poses。恢复时仍使用原 seed-0 world、原 24 poses、原标签、原阈值和同一 USD profile，并再次提交执行确认。

## 7. 本轮验证

- 两个新增/更新的状态 JSON 均通过 `jq` 语法校验；
- `PLAN.md`、`PROGRESS.md` 与 `results/project_status.json` 的当前 Phase 一致；
- 使用本机已有 Anaconda pytest 环境运行 `test_governance.py` 与 `test_cano_sensor_smoke.py`，19/19 通过；
- 本节记录的是审计当时的验证边界；后续 v1 GPU 探针已单独执行并封存，不改变本审计的只读证据来源。
