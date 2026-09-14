# 条件负监督所需网格：精确缺口与复用实现

已有material_manifest的30观察与本次307观察按task/sequence/五帧匹配，交集为0。旧单例join probe构造过网格，但只保存诊断摘要，不是307份持久网格缓存。因此不能直接把已有网格数量当本次资源已齐。

复用方案不更换观测规则：原student扫描/相对运动 → 原CPU project_input → 原0.25米80×80×80网格 → 原bound_reference_exclusion核验。新增development_reference_grid只封装这条既有路径，不提取重复学生patch，不重新调用encoder。2测试0.76秒通过，CPU原网格source/content hash和state逐元素相同，错观察绑定拒绝。尚未真实批量执行。

精确准备清单：configs/v3/gate3/development_reference_grid_preparation_v1.json（DRAFT，不是执行许可）。

| 口径 | 数量 |
|---|---:|
| 需要网格的观察 | 307 |
| 五帧使用次数 / 去重变体源帧 | 1535 / 1529 |
| 需读取任务 | 156 |
| 原按任务读取器解码观察 | 2184 |
| 仅解码带入、不作为目标的观察 | 1877 |
| 解码观察涉及去重变体帧 | 10914 |
| sensor数组 | 468 |
| sensor解码总字节（不是峰值） | 1005566784 |

卡引用及307选择绑定均记录hash；数组/chunk/input/hash继承原完整scope，只允许156任务。原读取器逐任务检查全部输入，因此不能把读取范围只写成307。只对选择绑定里的观察生成网格，其余不入训练/评分。

拟定CPU上限2小时、RAM4GiB、结果2GiB、零GPU，不是完成时间承诺。旧30观察material耗时134.336秒只能提供粗略量级，不能作为本人口最坏吞吐证明。下一按该精确范围建立数据导出卡、验证器和单次executor，freeze/preflight后生成307观测网格；不再检查网格是否存在，不静默重定义背景。训练仍未启动，Phase3未通过。
