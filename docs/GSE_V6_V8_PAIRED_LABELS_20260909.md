# 新旧标签逐来源比较：增加72项，但有1项退回未知

本次只读141个同源观察的封存结果，没有生成标签或训练。逐项结果及原文件SHA保存在figures/gse_graph/v8_saved_paired_audit_20260909.json；脚本tools/v3/v8_paired_saved_audit_v1.py，会话18038退出码0。

## 已确认结果

| 旧版→新版 | 观察数 |
|---|---:|
| 有路口→有路口 | 12 |
| 无路口→有路口 | 72 |
| 有路口→无路口 | 1 |
| 无路口→无路口 | 56 |

这是标签可用性变化，不是准确率。V8同时包含侧向证据、几何场和精度策略变化，不能将收益单独归因某一组件。人口为旧正例所在任务的多位置观察，不是无偏评价。

共同保留的12项同身份锚点局部坐标最大差值为0米，说明没有修改这些构造参考位置；这不是跨帧世界坐标验证，也不是学生定位精度。

## 为什么还有57项无锚点

54项的原始证据记录nearby_junction_references为0，即当前参考范围内没有路口候选；不能直接记为漏检，也不能未经完整可评分区域核验就当背景负例。

另外3项存在候选，但方向证据不足：
- S03/C04 rounded_rectangle，source29340，弧长4米，node_0059：旧正例退回未知。
- S08/C01 ellipse，source116866，弧长10米，node_0088：新旧均无锚点。
- S10/C04 ellipse，source192654，弧长5米，node_0164：新旧均无锚点。

三项均明确保留未知，不强行补标签。新增数量不能掩盖第一项退化，下一步只读其源证据和归档源码，区分旧错误正例与新证据误拒绝，不先改教师或重训。

## 各任务完整统计

| 任务 | 观察 | 旧路口标签 | 新路口标签 |
|---|---:|---:|---:|
| S02_3d_tree_small_C05__c1_mixed | 6 | 1 | 6 |
| S02_3d_tree_small_C05__ellipse | 6 | 1 | 6 |
| S02_3d_tree_small_C05__rounded_rectangle | 6 | 1 | 6 |
| S03_flat_unicyclic_small_C04__rounded_rectangle | 18 | 1 | 5 |
| S06_3d_branch_medium_C02__c1_mixed | 8 | 1 | 7 |
| S06_3d_branch_medium_C02__ellipse | 8 | 1 | 7 |
| S06_3d_branch_medium_C02__rounded_rectangle | 8 | 1 | 7 |
| S08_3d_loop_rich_C01__c1_mixed | 20 | 1 | 7 |
| S08_3d_loop_rich_C01__ellipse | 20 | 1 | 6 |
| S08_3d_loop_rich_C01__rounded_rectangle | 20 | 1 | 7 |
| S10_3d_complex_C04__c1_mixed | 7 | 1 | 7 |
| S10_3d_complex_C04__ellipse | 7 | 1 | 6 |
| S10_3d_complex_C04__rounded_rectangle | 7 | 1 | 7 |

## 尚未完成的结论

没有证明标签完整、所有新增72项正确、普通背景可评分或校准人口充分。下一步优先追查3个有候选未知项，特别是唯一旧正例退化；已有完整输出直接复用，不重算141。Phase3不推进，模型训练仍为0。

