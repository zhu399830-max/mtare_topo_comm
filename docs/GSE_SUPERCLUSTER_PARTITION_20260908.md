# 官方聚类运行记录

命令：`env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 build/gse_supercluster_env_v1/bin/python tools/v3/check_supercluster_partition.py`。最终会话27680，exit0；状态为PARTITION_DIAGNOSTIC_WITH_TWO_EDGE_COMPATIBILITY_GAP，不是完整PASS。

固定官方提交eb959b61226f60e0c037cc105c56d51318650e8a，算法未修改。使用此前按需源码加载器，非完整官方训练入口。

三块位置x=0、0.1、2米，大小各4，同语义；关系0.875、0.125。regularization=0.01，x_weight=1，p_weight=1，单线程。这些是合成软件参数，不是地下开发校准。

原始2条边在官方调用grid_graph时触发2×2布局TypeError；首次grid-graph安装也因发行名不一致失败，随后按官方install.sh安装pygrid-graph0.0.4成功。另有pycut-pursuit0.1.4与hydra-core1.3.6，只装独立venv。

保留两边报错检查。独立诊断增加0—2零亲和度边，logit为负无穷、sigmoid为0，官方目标中边代价为0。此诊断输出[0,0,1]；两条非零边提高到0.9999后输出[0,0,0]；三次重复与节点重编号一致。不能把它当作原始接口已修复，也不能用于静默修改研究候选图。

实际调用官方InstancePartitioner、instance_cut_pursuit、_instance_cut_pursuit、scatter_mean_weighted及get_stuff_mask。脚本输出源码SHA、依赖版本与结果。无研究数据、无神经网络预测、0训练更新。下一对布局和小图边数问题做明确适配与反例检查，再审查地下任务监督，不能从软件测试推进到论文成功。
