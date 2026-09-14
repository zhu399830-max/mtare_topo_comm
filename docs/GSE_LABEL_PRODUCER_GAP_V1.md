# 当前真正缺少的是标签生产器，不是更多诊断运行

代码核对得到明确事实：

- `gse_terminal_reference_v2.py`保存端面见证，但输出固定`semantic_label=None`、`complete_region=False`。
- `gse_window_opening_proposals_v1.py`固定`width_m=None, height_m=None, anchor=None, membership=None`及`training_eligible=False`；reference_position只是源几何参考，不是已认证开口标签。
- `gse_directed_interface_evidence_v1.py`固定`qualified_labels=0, complete_region=False`，仅解释进出证据。
- `gse_surface_target_adapter_v1.py`能接收独立锚点/开口/归属/评分区域，但只验证显式记录，不从诊断证据生产记录，也不认证完整性。
- `gse_surface_review_v1.py`能导入实际标注，但不存在可冒称人工复核的真实记录。

所以，扩大现有diagnose_observation的调用次数不会自动生成任何合格标签。这是未完成的软件与监督定义接线，不是数据已经证明方法失败，也不能把全部责任推给未知回波。已有6036观察、383/31/29端面支持与真实路口接口证据保留复用。

## 下一实施约束

暂停新的端面/来源计数/接口数量批量诊断，直到独立标签生产器能在合成例上输出实际非空目标和已知负例/未知区分。不得把上述None/False常量改成True来绕过资格。

生产器要连接三个不同层次：

1. 候选物理参考与观测证据，仅teacher路径可见；
2. 独立锚点、局部开口、明确归属的目标记录；参考切割端点不直接成为开口；
3. 有依据的可评分区域；隐藏结构未知，完整区域内额外预测计误检。

必须以固定的直道、T路口、近邻双路口、分层、尽头和遮挡反例检验：实际输出正确目标而非永远空列表；同一可见观测不因隐藏参考改变确定标签；未知不强行标负。软件测试与现有真实证据都不能自动认证人工参与。

当前没有足够证据直接宣布已有参考位置/所有区域成为合格监督，因此本次不生成真实标签或启动训练。此发现改变下一动作：完成生产器定义与实现，而不是继续创建诊断run或准备GPU特征。完整目标不缩小、不标完成。
