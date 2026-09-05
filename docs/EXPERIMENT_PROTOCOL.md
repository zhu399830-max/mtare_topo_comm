# V3 实验执行与证据规程

本规程适用于数据导出、teacher 审计、训练、离线回放、shadow、closed-loop 和 benchmark。它与 `AGENTS.md`、`MASTER_PLAN_V3.md` 一起构成每次工作的强制入口。

## 1. 开工前必须先向用户说明

在任何可能产生研究结论或消耗计算资源的动作前，先报告：

```text
当前 Gate 与唯一研究问题
本次动作及为什么现在做
使用的数据源、world、轨迹、split、样本口径
主方法、baseline、必要备选（若已获批准）
将读取和写入的目录
预计时间、磁盘和计算成本
验收条件、停止条件和拟保存的证据
```

报告不能只说“跑一下看看”。标注 pilot、AI 批量标注、数据导出、teacher 生成、SSL 和训练都必须提交 data card，并得到用户明确确认。

## 2. Preflight 硬检查

运行前执行：

```bash
python3 tools/v3/preflight.py --spec configs/v3/gateN/<run>.json
```

以下任一情况必须阻断：

- run Gate 与 `results/project_status.json` 当前 Gate 不一致；
- run spec 没有与本次范围绑定的用户明确批准记录；
- 已有运行被标记为正在进行；
- 运行目录已存在，存在覆盖风险；
- 标注 pilot、AI 标注、数据导出、teacher 生成、SSL、训练、标定或 checkpoint 选择缺少已批准 data card；
- train/dev/SSL/统计/调参 world 与 strict test world 重叠；
- data card 缺 world、独立轨迹、空间覆盖、有效样本、teacher、split 或成本信息；
- data card 的用户批准没有明确绑定允许的 operation、Gate 和确认记录；
- spec 没有研究问题、方法、baseline、命令或验收条件。

Preflight 通过只表示治理字段完整，不表示数据科学合理或 Gate 已通过。

## 3. 运行与证据

用下列命令只创建标准证据目录和快照，不会执行实验命令：

```bash
python3 tools/v3/create_run.py --spec configs/v3/gateN/<run>.json
```

随后在用户知情的前提下执行 spec 中冻结的命令。禁止边运行边静默改 config。必须记录：

- 完整命令、seed、环境信息和源配置快照；
- 输入 manifest/data card；
- 原始日志和失败原因；
- 原始指标、聚合方法和 summary；
- 对应 Gate 要求的 artifact；
- 所有偏差、INVALID run 和人工干预。

## 4. 数据规模的报告方式

数据规模必须同时按下列维度报告，不能以 `.npz` 文件数量或相邻帧数代替：

- 独立 world 数及名称；
- 每个 world 的独立 trajectory 数；
- 每条 trajectory 的时长、移动距离、空间覆盖；
- 原始帧数、空间/时间采样规则、有效样本数；
- 交叉口、分岔、死端、宽窄变化、坡地、多高度等有效结构事件；
- 各 split 的 world-disjoint 审计和历史污染审计。

`.npz` 只是 NumPy 的压缩容器，可装多个数组和元数据；一个文件不天然等于一个独立样本，更不等于一条独立轨迹或一个独立环境。

## 5. 可视化规则

只制作能够支持或反驳当前假设的图。每张图必须配套机器可读数据，并标明：

```text
Gate / experiment ID / question
split / world / trajectory / sample ID
方法与 baseline
坐标系、单位、颜色含义
该图支持或反驳什么，以及不能说明什么
```

优先级：数据和 teacher 叠图 > 失败案例 > 指标分布/coverage-time > 定性展示。不得只挑最好看的样本，不得对 strict test 反复看图调参。

## 6. 问题与停止报告

发现问题后，先停受影响动作并报告：

```text
观察到的事实和证据路径
影响的 Gate、实验和已有结论
归因：数据 / teacher / split / 实现 / 系统 / 指标 / 未知
可选方案及各自成本
推荐方案和理由
需要用户决定的事项
```

不能以改阈值、补 loss、换网络、换数据或重跑来绕过问题。

## 7. 阶段结束报告

每次阶段结束必须给出：完成了什么、没做什么、生成的文件、测试结果、发现的问题、当前 Gate 是否变化、下一步唯一任务。Gate 状态只能是 `GATE_PASS`、`GATE_MIXED` 或 `GATE_FAIL`；即使 PASS，也必须由用户确认后才能推进。
