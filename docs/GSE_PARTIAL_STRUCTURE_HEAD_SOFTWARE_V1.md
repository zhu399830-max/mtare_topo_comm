# 缓存结构小头：软件接线已完成，实际训练尚未运行

2026-09-05。复用原180观察的冻结轴线缓存及新支持标签，避免再次运行900帧扫描和骨干。下一实验只检验中心/部分成员的可学习性，不替代完整三分类、检测或图验收。

## 已实现

1. `gse_partial_structure_cache.py`：显式五文件路径/SHA读取。原sample_manifest只有task/row/count，必须借已封存observation_audit的sourceID和五帧编号绑定新teacher。该身份是原producer证据链继承，不假称缓存自己带帧ID。全部32预测槽逐值保留；GT mask仅在loss，原GT padding必须零。
2. `gse_region_target_bridge.py`：仅loss侧按三个控制点欧氏距离均值做GT/预测轴线匹配，再确认正反向与全局对应唯一性。非唯一/机器精度平局UNKNOWN，不借真实节点ID或关系破平局。该新成本在冻结前选择以保证共同旋转下的几何对应；旧L1评分器和历史指标不动。“唯一”不是几何预测准确或地点关联可靠的证明。
3. `gse_partial_structure_training.py`：显式seed/步数/batch/lr/device/hidden，三分支相同初始权重和样本顺序，仅更新小头。GT、预测、同预测去显式关系各自loss侧匹配，后两者输入和目标完全一致。保存全部loss、分母、匹配、梯度和更新次数；不内置科学PASS判据。

合并回归292 PASS/3.11s（含本轮教师、方向桥、缓存、小头、持久节点坐标及原区域头）。新增集成测试把合成teacher经方向桥接入真实三分支forward/loss/backward；纯合成，未读真实缓存/标签/权重，尚未启动新的真实训练。

## 下一步

先以新精确卡绑定五文件与原producer规格/版本，执行一次缓存关联和loss目标转换，记录方向对应后的真实正负分母、未知/未匹配数及每父地图账本。该导出不重渲染、不跑骨干、不训练。然后以实际分母冻结短GPU小头预算与评价，不另要用户重复批准。

当前数据只有6个独立交汇实例、0有效终点事件，所有label_complete=false。未匹配query不能被标成背景；不得把忽略未知后的precision包装成完整检测precision，不报告无终点的三类macro-F1为完整验收。GT与预测有效候选数量可能不同，不称单纯几何噪声消融。mixed-only没有跨形状配对，暂不运行一致性消融。
