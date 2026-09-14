# 下一次比较不能只重复旧训练

`python3 tools/v3/audit_relation_comparison_history.py`已实际运行，读取3个明确开发run的summary及首轮冻结卡，输出逐文件SHA。没有扫描/模型权重读取，没有训练，没有完整seal重验。

| 历史运行 | 实际更新 | 原合同结果 |
|---|---:|---|
| gse_synthetic_fit_v2_seed0 | A/B/C各500 | 三组拟合均未通过 |
| gse_synthetic_window_fit_v1_seed0 | A/B/C各500 | 三组拟合均未通过 |
| gse_block_representation_fit_v2_seed0 | r0/r1/r2各500 | 三组拟合均未通过 |

这些是不同接口/评分合同，不能拼成同一排名。它们也不覆盖项目所有历史方法，不能由此宣布几何学习原理失败。

首轮冻结卡的unknown_fields为dimensions、reachability、membership。卡SHA为4e85c117702f1200617b3b443c14497794ef94af2e5f61feb046788053fae4aa。旧surface模型输出这些字段不等于它们得到训练，尤其不能展示随机归属或可靠性头为模型成果。

已有observed_targets支持逐开口—锚点true/false/null，null进入无效掩码；物理可达监督在此adapter始终关闭。已有training_join绑定同源观测、缓存与teacher，仅验证对齐，不证明标签语义。新C0控制能公平隔离显式关系属性，却不能补齐缺失目标，或自动解除旧拟合失败停止条件。

严格区别：即使不直接监督关系，模型也可能通过位置任务间接学习关系；所以不能说“没有membership标签就不可能测试几何收益”。但已有完整C未通过小样本拟合、且旧失败已用完预定修正，单加C0重跑没有足够的新依据。

下一先列出并读取已有C01—C07标签冻结清单、汇总元数据，核对实际支持开口/端点关系的人口、未知比例、开发分区和来源。存在即复用；不存在就明确缺项，不将构造身份硬转实例标签。精确新读取范围冻结前不加载数据载荷。本轮不启动新训练、不重写原结果，论文图片全部保留。
