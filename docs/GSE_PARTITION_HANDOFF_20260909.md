# 原生超点到训练端的缓存交接已实现

原生SPG在Python3.12环境，训练端在Torch Python3.13环境。本轮新增NumPy-only的development_partition_handoff，不安装依赖到另一环境，也不把R2缺失回退为R1。

生产侧`produce_partitions`只消费绑定hash的原compact几何，保留同一10米ROI及原射线索引，调用既有体素与SPG实现，输出内存NPZ字节。缓存同时包含两种assignment、点、索引及闭合来源header，无teacher目标、无模型推理。实际文件写入和正式运行证据由待接入的executor负责，本轮没有导出307分组。

消费侧`decode_partitions`逐项检查payload hash、观察/缓存身份、准确坐标和索引（不是近似最近点匹配）、两种分组覆盖与4096容量。`bind_partition_payload`再以实际compact.cache_sha256及ObservationBinding核对来源，共享已有上下文聚合，返回R0/R1/R2三组。数据读取器新增保留原cache hash，防止仅靠同形状数组误接上下文。

14联合测试0.77秒通过，覆盖同源往返、错文件/帧身份/缓存/坐标/射线拒绝、缺表示拒绝及三组上下文接线。生产函数测试注入分组器，只验证消费ROI和封装；不是重新执行真实SPG算法。原生真实样本分组结果仍以先前probe为依据，不把软件测试冒充307人口提取。

下一将该生产/消费体直接纳入正式执行器与精确运行卡：固定307源缓存/构造/朝向/目标和所需观测网格的hash与环境，流式提取表示并接已保存部分目标。禁止为省事把未准备的观测网格设为完整背景、让R2使用R1分组或在冻结后修改代码。当前无训练、无新标签、无图收益，Phase3未通过。
