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

环境：冻结 sidecar 可导入 Torch 2.9.0+cu129 / NumPy 2.1.3 / pytest 8.3.4；CUDA 当前不可用。CPU 单元检查正常；未启动新训练或将旧 CUDA 训练结果冒充 CPU 重现。

## 唯一下一步

实现显式任务/行 allowlist 的读取合同，并从元数据核实五帧唯一源帧数、轨迹实际距离/时间和端点可见性；不得把 180×5=900 个窗口位置当成 900 个唯一原始帧，也不得伪造独立轨迹时长。随后冻结相同 180 行的字段补充与教师有效性 Data Card/spec。若缓存/标签有问题，停受影响训练，保留软件工作并报告原因；不放宽验收线。
