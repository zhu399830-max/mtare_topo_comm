# 原始观测与支路核心复核

全部12例均属于三个训练父地图，不是独立测试。observation_00.png至observation_11.png为距离图和10米内观测；full_00.png至full_11.png为同五帧完整有效回波。红色圆仅表示10米裁剪，不是墙、出口或真值。

artifacts/review.json保存24个AI提出的表面核心及逐例理由；ai_core_*.npz保存原始射线ID、核心提议和未知掩码。它们不是模型预测，不是独立人工金标，也不代表完整支路分割。

metrics/ideal_grouping.json与pair_correction.json分别记录旧/修正射线配对下的理想核心分组。修正后12例各两个核心组、无错合并或已标射线遗留；这是给定局部AI参考后的后端检查，不是模型准确率。
