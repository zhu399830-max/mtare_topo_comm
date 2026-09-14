# 训练前缺口：复用已有负监督和旧校准资产

本轮只读源码和哈希绑定元数据，真实扫描/新标签/训练均0。

## 不是“完全没有负监督”

旧block_relation_train_v1调用development_corrective_training_v1.train_update，后者经conditional_branch_selection调用conditional_anchor_branch_loss。最终负例查询由gse_reference_exclusion_binding_v1提供；它不是一条未使用的孤立函数。

条件负例要求：查询位置有观测支持、完整构造参考清单存在、查询距所有参考超过既定4米匹配半径。参考包含未获正标签和ROI外的degree1/degree>=3节点。匹配正例不会同时被处罚为负例；未知位置不产生负梯度。教师信息仅进入训练损失，不能进入部署前向。

这只排除“不能匹配任何构造参考”的预测，不证明所有物理几何事件已标全。不能把自由空间本身当作无路口证明，也不能把anchor_region_complete改为true来扩大负例。

源码依据：

- src/mtare_topo/representation/development_corrective_training_v1.py
- src/mtare_topo/representation/conditional_branch_selection_v1.py
- src/mtare_topo/representation/conditional_anchor_branch_loss.py
- src/mtare_topo/teacher/gse_reference_exclusion_binding_v1.py
- src/mtare_topo/teacher/gse_reference_exclusion_v1.py

## 校准缺口首先是所选路线覆盖

多位置清单复用每父地图既定hash rank0的一条路线，并展开所有位置；它不是遍历所有路口。封存target_manifest确认校准180项路口目标为0。

另外已有旧307人口中的27项校准记录，分支出现87次：

| C07父地图 | 观察 |
|---|---:|
| S01_flat_tree_small_C07 | 3 |
| S02_3d_tree_small_C07 | 3 |
| S07_flat_loop_rich_C07 | 13 |
| S09_flat_complex_C07 | 8 |

这27项与rank0校准180按task、source_sequence_id、五帧来源匹配，精确交集为0。87是清单中的分支出现次数，不是87个独立出口，也不是新V8有效标签。

来源saved_branch_binding_scope_v1.json SHA f859cb8436974d63350dda8856a50fee68b7921c410cbbaaf06540e2482b485a；解码内容SHA 9965eeac86b827266b76b0318c3bf99f3dee39641cba47ccab3a80c8b25b2bcb。旧2259运行seal为2daea1053cc2f6b00f132bc21496ba77b7091e092e2a882cb771aa6d1ff06d27，目标清单按seal核验。

## 下一步

先检查旧27项的独立结构实体、标签版本、原始来源和现有输入缓存，确定能复用什么及真正缺项。只做已有资产核验，不直接更换当前校准集或把旧标签冒充V8；任何新数据操作另行精确绑定。

保持未知掩码与原指标。当前部分标签合同仍不能支持完整检测F1，负监督实现存在也不等于已有校准人口充分。不要再为“没有完整窗口”重造一套负例算法。
