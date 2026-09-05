# 几何组合主线：首批实现记录

日期：2026-09-05。方法依据：`GSE_GRAPH_COMPOSITION_EXECUTION_PLAN_V1.md`。

## 已完成且可复查

- 本地 Git 基线 `1fc1d56` 保存实施前 2569 个源代码/配置/文档文件；未发布远端。大数据、权重和论文位图保留原处，不进入 Git 大文件提交。
- `representation/gse_composition.py`：共同坐标相对几何特征、小型组合头、端口归属、显式未知掩码、冻结基元输出适配器、只作用于共同可见结构的跨实现一致性损失。事件是 corridor/junction/terminal，不训练全局地点身份。
- `topology/gse_causal_graph.py`：首次稳定观测即确认节点、显式 departure、逐段连续轨迹闭合、拒绝不确定合并但保留独立节点、不可变快照和时间顺序检查。
- 测试覆盖共同 yaw、置换、角度变化、上下层、无支持、未知标签、结构配对、梯度、首次节点、连续中间边、回环、歧义、断裂轨迹及前缀一致性；连同旧模块回归共 **56/56 PASS**。

## 唯一软件证据运行

`results/gate3_semantics/gate3_20260905_gse_composition_software_contract_v1_seed0`

- preflight：0 errors / 0 warnings；只执行一次，无覆盖/重试。
- status：SOFTWARE_CONTRACT_PASS；耗时 1.684 s；子进程峰值 RSS 610095104 bytes。
- 0 数据集世界、0 数据集帧、0 optimizer steps、0 M-TARE。
- 原始日志、JUnit XML、环境、spec、工具哈希、metrics、RUN_STATE 和 11 项证据 seal 均保存；seal 独立复核全通过。
- seal-list SHA-256：`1d25d2b74904ec8df8f28082d55d725752cd198834bba5c166511a3294d3c533`。

这不是学习精度、真实地图建图、正式图资格或 Gate PASS。配准器、端口跨重访对应、实际传感器/执行适配、校准阈值、局部几何回归与正式训练仍未接入；内核接收的验证布尔证据不能代替真正的配准实现。

## 180 观察缓存检查：不能直接开始新训练

只检查既有密封 NPZ 的成员目录和 JSON 清单，未加载地图/传感器数组。

旧缓存五项：endpoint_features、endpoint_confidence、target_descriptor、target_degree、source_global_sequence_index。没有完整共同坐标 XYZ/方向，也没有逐端口归属/有效性标签，不能从旧局部特征反演并冒充真实预测。

清单：10 个 C01 mixed 任务，共 180 观察；度数 1/2/3/4 分别 20/120/38/2。二度节点占 2/3，必须用分类别/宏平均指标，不能以总准确率衡量结构识别。度数标签也不能直接冒充经过可观测性验证的事件标签。

旧通用 loader 构造时扫描并打开根目录的全部分片，即使随后只选择 C01 行。新补充导出必须用显式任务 allowlist 初始化，只打开这十个任务，不能复用该初始化路径然后声称其他世界 0 访问。

环境：冻结 sidecar 可导入 Torch 2.9.0+cu129 / NumPy 2.1.3 / pytest 8.3.4。最初沙箱内 CUDA 查询失败；随后主机只读查询确认 RTX 5090 D、32607 MiB 正常。这是设备访问限制，不是硬件故障；尚未开始新训练。

## 同 180 行核对已经完成

`gate3_20260905_gse_composition_inventory_v1r_seed0`：同 10 个 C01 mixed 父地图、180 个观察、100 个节点、900 个唯一源帧，唯一帧数从实际元数据逐项确认。静态距离采样没有采集时钟，duration 为 null；没有伪造运行时长或独立样本量。

只读 reader 只打开清单任务/行所需元数据；350 个实际源 chunk 按旧 seal 校验。未解码 LiDAR range/valid、未读 checkpoint、未生成新训练标签。用时 5.103 s、RSS 149938176 bytes。14 项输出 seal 已复核；seal-list SHA-256：`e532f9366ff5dc129c6112fafcd33b9c74057adfd6b58b45d37b1ab34e5a7213`。

第一次 V1 因 JSON 对象读取器不能接收列表而在分片读取前失败，保持失败结果不变。V1R 仅修读取器，另加完整合成执行到 seal 的回归测试，不改变样本或阈值。

| 真实 degree（仅核对） | 观察数 | 全部旧物理端点满足支持的观察数 |
| --- | ---: | ---: |
| 1 | 20 | 20 |
| 2 | 120 | 120 |
| 3 | 38 | 6 |
| 4 | 2 | 0 |

关键解释：40 个分叉观察均含全部 incident 基元的可见几何片段；只有 6 个完全满足旧端点支持，来自 3 个节点。两种“可见”不是同一含义。T 形解析反例中，激光已经能照进支路，但交汇内部本来没有墙面回波，所以不能要求回波抵达构造端点才能承认看到了通道。

因此，密封 summary 中 `support_is_necessary_not_sufficient_for_event_label=true` 的节点任务解释被明确撤回，不修改密封文件。数值统计有效，但不能把其余 34 个观察删掉，不能直接从 degree 生成可观测标签，也不能据反例宣称全部真实三维样本都有完整监督。

## 新增代码与验证

- `data/gse_scoped_inventory.py`：精确任务/行 allowlist、只读键范围、源哈希、缺块/越界/符号链接逃逸拒绝。
- `governance_inventory.py`：仅元数据 audit 可用的精确库存卡；不放宽训练和导出卡。
- `visible_geometry_input_from_prediction`：保留预测可见基元的共同坐标几何，不依赖物理端点分类分数；旧桥保留作比较。它不等于已经验证了端口或连接。
- 库存端到端、治理回归、输入桥、解析反例及组合/因果图针对性测试 **82/82 PASS**；扩大到旧学生读出/关联/图漏斗回归后 **107/107 PASS**，1.11 s。这是本地测试，不混称为原 56 项密封 run 的内容。

## 唯一下一步

明确并测试节点级“可观测结构”监督及同 180 行字段恢复合同：使用可见几何与局部自由空间证据，而不是未经验证迁移旧端点掩码。实际读取扫描、导出预测或生成新标签前分别冻结精确 Data Card/spec；当前库存卡不授权这些操作。完成接口可学性证据之前不扩大训练，不改变人口/阈值，不读 C07--C10 或启动闭环。
