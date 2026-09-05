# AEE 结构语义域适配与 Gate 6 恢复方案 V1

## 结论

Gate 6 V1 证明当前冻结 M1D 不能直接用于 M-TARE 的 AEE 传感域。问题不是模型文件不能加载，而是方向头在真实移动扫描上几乎始终不给出口：三个 checkpoint 在 99 个移动帧上的空方向分别为 `99/99`、`97/99`、`99/99`；冻结 B0 为 `99/99` 非空，完整地图 teacher 在抽查 20 帧上也为 `20/20` 非空且给出 1--3 个出口。

因此不能只补 `zstd`，也不能把 B0 长期接管后仍称为“结构语义模型替换 M-TARE”。推荐重新打开表示学习纠正环节，做一次受控的 AEE 开发域 head adaptation；通过后依次重做短闭环资格和 Gate 6 统计实验。

## 科研问题

在保持已学习 encoder 与 `z_role` 表示不变时，使用客观完整地图 teacher 只适配 direction/count/role heads，能否消除 Cano→AEE 的输出域偏移，同时保留原 Cano validation 能力，并让因果拓扑规划器在真实 AEE 闭环中主要依赖学习方向而不是 B0？

## 数据合同

- 开发训练世界：AEE `tunnel`，环境 seeds `11/23/37/53/71`。
- 开发验证世界：AEE `garage`，环境 seeds `11/23/37/53/71`。
- 每条轨迹由原 M-TARE 控制，固定起点，收集恰好 3,000 个同步 `registered_scan + state_estimation_at_scan` 帧；最多 660 仿真秒，少于 3,000 帧或移动不足 50 m 即整次导出 FAIL。
- 共 10 条轨迹、30,000 raw frames。每 5 帧保留 1 个训练/验证样本，共 6,000 effective frames：tunnel 3,000、garage 3,000。
- 每个样本保存 16×720 range/valid、pose、时间、trajectory/seed/frame identity；原始 bag 使用主机冻结 `zstd-10` 无损压缩并验证解压 SHA-256。
- teacher 只读取对应 development world 的冻结完整点云，通过现有 layered height-conditioned query 生成 720方向、count和role；不读取未来轨迹、M1D输出或规划结果。
- sealed Cano V2R C09 validation只允许只读用于每epoch抗遗忘评估，以及AEE direction完全并列时的checkpoint tie-break；禁止进入训练、loss、归一化、阈值和超参数调整。C10及后续sealed benchmark worlds全部禁止读取。tunnel/garage在此后只能称 development，不能作为最终泛化测试。

## 模型纠正

- 分别从三个冻结 M1D checkpoint seeds 0/1/2 初始化。
- 冻结整个 encoder 和 embedding head，保持 `z_role` 表示不变；只训练 direction head、count head、role head。
- 每个 batch 固定一半来自原 Cano V2R train、一半来自 AEE tunnel，防止只适应单一 AEE world。
- 保持原 multitask loss 和原 augmentation，不新增结构标签或可学习模块。
- 最多 10 epochs，AdamW，learning rate `1e-4`（原训练 `3e-4` 的预注册 fine-tune rate），weight decay `1e-4`，batch 128，seeds 0/1/2；不搜索超参数。
- checkpoint 选择只看 AEE garage direction F1，tie-break依次为 Cano validation direction F1、AEE garage role F1、较早epoch。

## 表示资格门禁

三 seed 中位数与每 seed都必须报告。最低门禁：

- AEE garage direction F1 不低于同样本冻结 B0；
- AEE garage 空方向帧比例不高于 5%；
- AEE garage count macro-F1 和 role macro-F1 均不低于 0.70；
- 原 Cano validation direction F1 相对各自冻结 M1D 降低不超过 0.02；
- encoder 与 embedding-head tensors逐字节不变，`z_role` 对固定 probe 最大绝对差为 0；
- C09只读帧数必须精确报告且仅用于上述retention/tie-break；0 C10、0后续strict-test、0非预注册训练/选择。

任何门禁失败则停止，不进入闭环。

## 闭环恢复资格

适配通过后只运行 6 个 readiness cases：2 worlds × 3 adapted checkpoints，每个 180 仿真秒。B0 仅在学习方向为空时提供显式安全 fallback，并逐帧记录。

每个 case 必须满足：

- required topics 全部健康，0 failed planner cycle；
- 机器人移动至少 5 m，至少产生 2 nodes、1 verified edge和1个非hold waypoint；
- 首 20 帧之后 learned-empty/B0-fallback 比例不高于 5%；
- bag主机侧无损压缩与SHA校验通过。

若 fallback 超过 5%，方法不能称为 M1D-primary，停止完整90-case实验。

## Gate 6 正式比较

readiness PASS 后，重新冻结新的90-case matrix；原 M-TARE和Oracle合同不变，M1D换为三个adapted checkpoint。统计仍以10个world×environment-seed block为单位，主指标仍是0.5m coverage-time AUC。V1保持不可修改FAIL，不复用其case作为正式结果。

## 成本与治理

- 数据与teacher：约2--3小时，预计10--20GB。
- 三seed head adaptation：约1--2 GPU小时。
- 6-case readiness：约30--60分钟，预计2--6GB。
- 通过后新90-case：约17--20小时，预计103--120GB。

训练操作按现有治理只允许 Gate 2。推荐由用户明确授权“Gate 6发现域失败后，临时重新打开 Gate 2 corrective representation work”；训练PASS后再按顺序恢复Gate 4/5接口复核与Gate 6。不得把训练伪装成Gate 6 infrastructure。

## 需要的明确决定

批准本方案只授权实现数据collector/teacher/head-adaptation/readiness工具、测试和proposal preflight。正式数据导出、训练、readiness及新的90-case仍分别按Data Card/spec治理；若希望一次授权前两项，应明确写明“批准AEE域适配数据导出和三seed head adaptation”。
