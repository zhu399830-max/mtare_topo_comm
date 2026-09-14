# 剩余91条参考核验与7条未知定位

执行check_double_axis_reference_native.py，原合同和旧360脚本SHA检查通过，会话13661 exit0。原91条回退原因分布准确复现：61重复/多壳、27非交替/缺失、3未闭合。84条参考返回与独立解析一致，最大误差3.051757815342171e-6米；7条未知，无超过原1e-5米对照线的返回矛盾。耗时39.384973秒。加原269候选共353/360一致，不是全人口通过。

逐条原始输出：[JSONL](evidence/double_axis_reference_20260908.jsonl)。零封存扫描载荷、零训练标签、零优化器更新。六轴方向不是完整LiDAR扫描分布。

进一步只对7未知追踪既有interval_winding_exit的实际返回分支，未改算法。trace_double_axis_unknowns.py会话16922 exit0：4条来自oriented_winding抛ValueError，2条原精度/float32占据不一致，1条退出来源不在同t原生交点来源中。详细原生距离、operand与拒绝代码：[追踪JSONL](evidence/double_axis_unknown_trace_20260908.jsonl)。分支定位不自动证明三角面退化或舍入是完整根因。

下一检查这7条命中三角面是否共享真实平面，并对比高精度交点。不重新添加距离分组阈值，不改原数值门槛；尚未授权将未知填为确定标签。旧失败及论文图片保留。
