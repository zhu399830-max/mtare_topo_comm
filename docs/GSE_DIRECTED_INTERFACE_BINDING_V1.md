# 真实源轴向与量化射线接线

新增`gse_directed_interface_binding_v1.py`：按node＋endpoint定位实际源轴，使用axis_start_index后第一个真实轴段。辅助connector不能参与方向判断；同源相反incidence独立保留，缺轴/重复ID/退化轴拒绝处理。

`interpret_bound_result`复用原teacher对齐校验与normalize64→cast32射线打包，逐交点校验源接口身份、原五帧行号、float32参数、量化射线重建XYZ及原10m ROI，再调用进出解释器。它不重算网格交点，不改变原有序诊断，不把解释结果升为语义标签。

轴向接线6项测试加原方向解释4项，共10通过（0.19秒）。目前测试范围为轴向/解释组件；完整bundle解释虽已实现，尚未用封存真实交点验证，不宣称真实接线成功。新标签、训练和真实新诊断0。

已定位历史配置`configs/v3/gate3/gse_surface_junction_interfaces_v1.json`，后续先绑定其中已保存的原始交点及其精确源材料，对共同样本解释，不必重新运行整个人口的网格求交。真实几何支持仍不等于完整评分区域，不能用纯组件测试替代研究验收。
