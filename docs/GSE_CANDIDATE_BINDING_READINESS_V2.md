# 结构候选的生产身份绑定：最薄接口设计 V2

2026-09-05。**只完成代码审查与设计，尚未实现本文件的新生产入口。**

当前仍是 Phase 3 的几何组合与局部结构验证。本项只解决系统接线：候选如何证明来自所声明的五帧输入及同一次计算，不改变模型、标签、阈值或研究验收，不启动图回放。

本轮读取源码和既有规格，运行已有候选转换的 30 项合成测试：全部通过，0.64 秒。真实世界、扫描、缓存、teacher、权重读取均为 0；新训练和 GPU 操作为 0。唯一新增产物是本文档，未修改冻结源码或运行目录。

## 1. 已经做对的部分和真实缺口

`src/mtare_topo/topology/gse_region_candidates.py::region_candidates_from_prediction()` 已经能：

- 保留全部 32 轴对应的 64 个 query／方向及原始分数；不筛选、不建节点或边。
- 检查数值形状、dtype、device、支持 mask、时间格式及外参 frame 名称。
- 正确把位置按 `R*p+t`、方向按 `R*d` 转为机器人坐标。
- 保留未知宽高和数值不支持方向，输出复制后的不可变 CPU 值。

但它独立接收 `prediction`、`axes`、`stamp`，无法证明这三者在生产时属于同一观察。

既有测试 `test_matching_numeric_masks_do_not_certify_forward_identity_or_prediction_time` 是直接反例：原预测不变，将前 31 槽轴线整体平移 10 m、时间戳从 3 改为 9，支持 mask 仍相同，转换成功。**该测试证明当前转换器的能力边界，不证明生产身份已通过。**

仅添加一个可由调用方随意填写的 `input_sha256` 字符串不能修复它；把错配数据一起计算哈希，也只是封存了一份错配。

## 2. 建议只增加一个拥有输入和计算过程的入口

推荐新独立模块，不修改当前转换器或模型类。接口草案：

```python
produce_bound_region_candidates(
    observation: OwnedCausalObservation,
    runtime: VerifiedRegionRuntime,
    robot_from_sensor: SensorToRobotExtrinsic,
) -> BoundRegionCandidateBatch
```

三个类型名称均为设计，当前仓库没有对应实现。这个入口不接收外部 `RegionPrediction`、另一路 axes 或可替换的输出时间戳，也不接收教师答案。

内部顺序固定：

```text
拥有所有权的五帧输入快照
→ 已核验且排他使用的轴线前端 forward
→ 直接取这一次输出的全部 32 条轴线
→ tokens_from_axes → 同一次结构头 forward
→ 现有 region_candidates_from_prediction
→ 候选批次与生产凭据一起返回
```

由当前五帧包的最后一帧生成 `ObservationStamp`，不允许调用方重新命名时间。轴线只生成一次，结构头与候选转换使用同一个拥有所有权的轴线结果；不能在两步之间重新取“最新轴线”。同步的单观察入口优先，第一版不做异步多观察 batch，以减少行号串线风险。

这不是再实现网络：直接复用 `FrozenBackbonePointAxisAdapter.forward(range_valid, relative_translation_current_sensor_m, relative_yaw_current_sensor_deg)`，取 `.votes.axis_control_m`，再调用已有 `RegionQueryHead`。实际前端必须事先按指定冻结分支组装；不能默认使用新随机 readout、默认 offset 分支或未经核验的旧几何字段。本文不选择任何部署权重。

## 3. 输入包必须来自接收端，而不是事后补身份

`OwnedCausalObservation` 最少绑定：

| 内容 | 要求 |
|---|---|
| 流身份 | robot／sensor stream 与 session；仅部署观测身份，不使用构造 node、edge、traversal ID |
| 五帧身份 | 每帧原始序号与采集时间、共同 clock domain；顺序严格因果，当前时间来自末帧 |
| 模型实际输入 | 原 `range_valid` 和两项因果相对运动张量；使用原模型字段与 shape，不替换表示 |
| 坐标与运动来源 | 传感器／机器人 frame、当前传感器坐标约定、部署运动来源引用；不能传 GT 地图位姿冒充部署定位 |
| 内容快照 | 在接收边界取得独占或复制的张量快照；标识与输入一起交接，禁止生产过程中被上游复用缓冲区覆盖 |

时间戳是否取扫描起点、终点或中点必须沿用采集接口的明确合同；不同定义不能仅凭同一个浮点数视为同步。这里不发明一个可调“允许时间差”。若现有接收端没有共同时间约定，绑定未完成，不静默最近邻匹配。

不能仅靠类型或 Python 私有构造保证物理来源可信。责任边界是可信传感器接收端与排他生产器：本接口防止系统中意外串帧、错误复用和内容漂移，不宣称能识别上游恶意伪造。

## 4. 生产凭据要记录什么

`BoundRegionCandidateBatch` 包装现有 `RegionCandidateBatch`，另携不可变凭据；不新增图决策。

- 当前 stream/session、五帧身份、当前观察时间、坐标约定。
- 实际使用输入快照的摘要；包含字段名、shape、dtype、明确字节序与连续字节，不能只哈希数值字符串或将 dtype 转换后忽略差异。
- 本次轴线输出和完整 `RegionPrediction` 的摘要；支持 mask 同样纳入，不能只哈希有效子集。
- 已验证模型 bundle/session 身份、轴线前端与结构头的源码／配置／权重摘要、实际 `use_relations` 模式及预处理版本。
- 实际外参矩阵及来源／版本摘要；不以可变路径名代替具体标定内容。
- 同次生产调用的 observation receipt 标识；重新交付同一输入的计算记录不等于新采集帧，不能用于增加跟踪稳定计数。

摘要应从**随后真正被消费的不可变快照**计算，避免“先哈希 A、后读取已被改成 B 的数组”。`torch.inference_mode()` 不保证权重不会被另一线程更新；生产器必须拥有冻结且排他使用的模型 bundle，不能与训练优化器共享对象。

避免每帧反复读取大 checkpoint：模型在建立已验证 runtime 时核验内容，运行期间禁止训练与并发修改，记录该 runtime 的确定身份。若暂时不能保证对象所有权，就不能声称运行时权重来源已被绑定；单写一个 checkpoint SHA 字符串不够。

候选 receipt 只能证明这份输入经这个生产器生成该输出，不是模型准确性、候选唯一性或可通行性的证书。

## 5. 可复用与不能冒用的现有入口

| 位置 | 如何复用 | 不能据此宣称什么 |
|---|---|---|
| `gse_point_axis_readout.FrozenBackbonePointAxisAdapter.forward` | 已有因果 LiDAR／运动到全轴线计算链 | 当前数值 forward 不自带采集帧身份或模型部署凭据 |
| `gse_region_queries.tokens_from_axes`、`RegionQueryHead.forward` | 在同一生产调用中直接串接，无教师参数 | mask 相等不能证明调用身份相同 |
| `gse_region_candidates.region_candidates_from_prediction` | 原样作为最后的无决策坐标转换 | 不是独立输入来源校验器 |
| `gse_registration_binding.RobotFrameScan`、`DeploymentPose6D` | 参考显式时间与坐标合同；后续各消费链需共享当前采集身份 | 当前注册扫描只有时间／点云摘要，并未与五帧模型包建立公共接收端绑定 |
| `gse_partial_structure_cache.load_partial_structure_cache` | 现有离线源哈希、精确 frame／row 对应可用于科研复现 | 依赖封存 teacher-audit 的离线研究桥不能搬进部署 forward，也不能替代在线采集身份 |

## 6. 实施前要固定的最小合成验收

只用合成张量、假轴线 producer 与真实随机初始化结构头，不读权重；其中至少一项真实执行 `tokens_from_axes → RegionQueryHead → 候选转换`，不全部用 mock 代替。

1. **同掩码异内容：**两个全非退化但不同内容观察交错送入，各自 receipt、轴线、预测和候选对应正确；生产接口不存在外部注入另一份 prediction 的参数。
2. **因果帧包：**错序、重复序号、未来帧、跨 stream/session、clock domain 不一致明确拒绝；当前 stamp 精确来自末帧，不得事后改成另一时刻。
3. **缓冲区复用：**取得快照后改变调用方原张量，生产结果仍对应所拥有快照；若无法保证拥有权则拒绝，不发布“已绑定”结果。
4. **实际调用链：**每包轴线前端一次、结构头一次；头实际收到的轴线与转换消费的轴线相同；receipt 来自这条执行链，不是另传哈希。
5. **保留全候选：**完整 32 轴／64 query，包括低分、未知方向和缺失尺寸；forward 无 teacher mask／真实 ID／loss matches 接口。
6. **外参与坐标：**非零 roll／pitch／yaw 与平移；frame 或标定版本不一致拒绝；中心带平移、方向不带平移，不能把多候选中心都替换为机器人位姿。
7. **运行时状态：**训练态／未核验 bundle／并发权重更新不允许生产；模型身份相同但模式配置不同不能共享同一 runtime receipt。
8. **重送与前缀：**相同采集包重复交付保留相同采集身份，不能伪装成新观察；只运行序列前缀与完整序列运行到同一点，先前 receipt 不变。
9. **异常原子性：**轴线前端或结构头报错，不返回半份候选；外部不能将上一次成功预测补到本次失败包。

其中生产入口、接收端输入包、模型 runtime 与这些新增测试均尚未实现。已有 30 项候选转换测试仅支持第 5／6 项中的数值子合同及现有缺口反例，不能算整套生产绑定通过。

## 7. 绑定之后仍然不能直接建图

这一步只回答“输出属于哪个输入”。后续仍缺：部署置信度校准、候选去重／跨帧实例跟踪、真实端口确认及未知尺寸处理、结构中心与机器人轨迹分开记录、同帧多实例图接口、端口关联及实际穿越边接线。

因此绑定完成也不能宣称模型可靠、结构节点已提交或图已建立；旧图不因 receipt 完成而自动开放。当前最优先的研究问题仍是已发现的训练／评分匹配失配及成员漏报，本文不转移主线去做真实回放。
