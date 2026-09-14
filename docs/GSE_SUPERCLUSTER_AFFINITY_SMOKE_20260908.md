# 官方亲和度模块：首次实际执行

命令：`env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 build/gse_supercluster_env_v1/bin/python tools/v3/check_supercluster_affinity.py`。

固定commit eb959b61226f60e0c037cc105c56d51318650e8a，源码工作树无修改。加载器使用官方AST原定义，包含完整InstanceData、CSRData、TensorHolderMixIn和实际图工具，不重写major、亲和度或scatter算法。仅绕开顶层src初始化对全套训练/可视化的导入；这不是完整官方入口复现。加载器首次两次运行因类注解Dict/h5py未预加载而失败，改为预加载该模块的外部依赖后正常执行；上游算法未修改。

## 已知答案与结果

三个块分别含：A对象4点；A对象3点+B对象1点；B对象4点。候选邻边包含反向、重复和自环，官方to_trimmed保留0—1、1—2。官方平滑目标分别0.875、0.125；硬多数归属目标分别1、0，与手算一致。反向输入、空图、非连续对象编号替换均通过。

这验证混合块不会被直接当作纯块，且块编号/对象编号没有物理语义。仍需注意：来源多义的同一个回波与多个不同点分别属于不同对象不是同一件事，不能把多重来源重复计数塞进该实例分区监督。

本次没有调用神经网络，没有训练、真实地图读取、聚类分区或地下结构预测。下一运行官方InstancePartitioner及真实cut-pursuit依赖的已知合成图；通过后才能检查完整实例机制，不应把本次亲和度目标计算冒充学会关系。

执行输出包含6个实际读取源码的SHA和按需定义清单，可由命令重现。关键instance.py SHA为9c5ecd20112ebd406b8eba29fe95a10f171a85f88c8c6f8902af2b7b6999031b，CSR为be8ed67948b2857126515f1751b54aec83dcc8c0b55b03eb22dcc4365dfed1a4。
