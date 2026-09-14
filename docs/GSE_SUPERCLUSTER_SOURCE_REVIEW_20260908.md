# 官方SuperCluster：实际机制与适配边界

已获取官方仓库至`build/gse_supercluster_reference_v1`，commit `eb959b61226f60e0c037cc105c56d51318650e8a`，工作树干净，MIT许可证保留。未安装依赖、未运行模型、未读取研究数据。网页master与本地固定提交行号不同，以本地为执行依据。

官方：[仓库](https://github.com/drprojects/superpoint_transformer)，[固定模型源码](https://github.com/drprojects/superpoint_transformer/blob/eb959b61226f60e0c037cc105c56d51318650e8a/src/models/panoptic.py)。

## 实际训练链，而非只借一个分块器

`panoptic.py`提取超点特征，预测语义与邻边亲和度，再由独立分区器聚合对象。邻边特征使用端点特征差的绝对值与平均值，保证交换端点时对称。当前固定版本的节点偏移头与损失被注释，不能把其docstring中旧偏移描述当成实际训练。分区使用已有超点位置，不靠自由查询猜路口坐标。

`data/instance.py::instance_graph`需要每个超点与真实实例的重叠计数。平滑亲和度根据两端对彼此主实例的重叠比例平均计算；它不是几何距离阈值。缺实例标签时，`transforms/instance.py`保存候选邻边但令亲和度目标为None，不凭邻近生成真值。`output_panoptic.py`在loss前处理void，但其void语义须逐条件核对，不能自动等价于我们的部分观测未知合同。

## 与地下任务的对应和缺口

可参考其“先预测局部归属关系，再聚合”的成熟机制，避免同时自由预测大量节点位置。它不是地下拓扑算法：对象实例≠交汇结构，分区边≠通行边，超点重心≠路口中心。我们的多来源回波也不是天然互斥实例分区。不得把源基元编号直接替换为路口身份来凑齐监督。

因此先运行官方算法的合成接口参考，再按现有来源建立目标适配反例；不得直接在45例锚点标签上假称复现SuperCluster。最终创新仍需几何组合对结构与图的增益，而非采用该网络。

## 环境与执行顺序

现有训练解释器缺torch_geometric、torch_scatter、torch_cluster、pytorch_lightning、hydra、pgeof及FRNN等已检查模块。官方安装脚本指定Python3.8/Torch2.2/CUDA11.8或12.1；不覆盖现有Torch2.9环境，也不直接运行该脚本。此为依赖条件，尚未证明兼容失败。

下一检查固定源码模块依赖闭包，选择最小官方亲和度/分区合成测试的隔离环境，保持算法原实现及许可证；若需要适配，逐项标明差异。已有同邻居关系属性对照仍保留，但不能代替官方参考链路验收。此步骤替代立即扩展自制消息模块的优先级，响应用户明确要求先借鉴并跑通参考实现。

## 核对文件哈希

- models/panoptic.py: `aba3889ade1c7fdd57862d3e57ac17242af7ea766c8969e0c56aeefae9b6b394`
- data/instance.py: `9c5ecd20112ebd406b8eba29fe95a10f171a85f88c8c6f8902af2b7b6999031b`
- transforms/instance.py: `0ae231d604255744d0585a97dd41a7d90bd6bb48622686c0a1871910390c337b`
- utils/output_panoptic.py: `82904234bf16d9756fa0b9b6c3707e233039e3875db5e4fade7d91cd24c65629`
- install.sh: `531bbc2e3d7d1bec479f6e1378eac7b23c3cfd86b6fde064e7c263825b93bfde`
