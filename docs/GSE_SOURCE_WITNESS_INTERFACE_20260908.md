# 来源证据接口：保留扫描，不把来源候选当确定标签

当前只完成软件接口，没有执行teacher、生成标签或训练。固定后续对象仍为原12观察、60帧，不新增人口。

`teacher/source_witness_policy.py`输入逐射线有效性、报告来源集合、候选表面集合、几何不确定性和独立认证标记。输出分开保存：

- diagnostic：报告与候选一致、双方单一且没有显式不确定性，只能作诊断。
- certified：还必须有独立认证。现有来源审计没有提供这项认证，所以不能把其余射线自动当确定监督。
- 未许可来源返回owner=-1，但不修改原始距离、valid或来源码。调用者必须显式选择模式。

19项联合测试通过（0.24秒），包括本接口4项与既有来源数值参考15项。接口不自行证明扫描行号、来源列身份或上游证书真实性；后续适配必须绑定封存清单。

## 必须接线的现有消费者

| 路径 | 当前如何使用来源 | 接线要求 |
|---|---|---|
| directed_interface_evidence_v1 | entering要求单一来源；leaving由交点方向判断 | 仅来源依赖证据用新mask，不能一律删除leaving |
| interior_branch_evidence_v1 | return_sources等于指定单一来源 | 独立来源许可后才计入内部支路见证 |
| observed_operand_entries_v1，经lateral_branch_evidence_v1 | 单一来源后检查实际进入交点 | 在进入筛选前应用许可，不改原始射线 |
| window_opening_diagnostic_v1及继承路径 | 单元素source_sets映射为owner | 用显式模式的owner，未知保留-1 |
| terminal_reference_v1/v2，经terminal_diagnostic | cap见证加单一来源 | 来源未许可的cap保留为未知证据而非确定尽头 |

另外junction_interface_diagnostic保存原reported来源，仅是原始记录；supplement_support和supported_construction_teacher也读取来源集合，不可把新审计自动用于这些历史生产器。尚未宣称所有传递调用链已完成适配或认证。

## 下一步

在上述既有消费者加入显式、可追踪的来源许可接线，旧无新字段路径保持历史行为；新诊断入口不得回落旧来源假定。完成端到端未知传播测试后，才绑定原12观察的部分监督诊断规格。该诊断必须明确非完整检测监督、非训练资格；不为继续训练而伪造证书或完整背景。
