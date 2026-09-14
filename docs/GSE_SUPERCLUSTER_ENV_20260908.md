# 官方参考运行环境进度

独立venv：`build/gse_supercluster_env_v1`。使用system-site-packages复用已有Python3.13/Torch2.9.0+cu129/NumPy等基础库，新增包仅写入该venv，不修改原训练环境。尚非完整冻结、自包含的正式实验环境。

已安装PyG2.7.0、torch-scatter2.1.2+pt29cpu及xxhash4.0.1；使用CPU二进制wheel，未编译GPU扩展。实际验证scatter_sum([1,2,3],[0,0,1])=[3,3]，梯度全1，consecutive_cluster([7,7,2])=[1,1,0]。原训练解释器仍无法找到torch_scatter和torch_geometric，隔离检查通过。

这是依赖算子检查，不是SuperCluster模型复现。官方InstanceData还依赖CSR、TensorHolder及图/稀疏工具；顶层src和utils初始化会导入额外训练与可视化依赖。下一最小导入闭包必须保持官方实际算法，不能伪造scatter、major、instance_graph或用几何阈值代替亲和度目标。若采用绕开顶层初始化的加载器，记录该差异，不声称端到端官方入口已运行。

CPU分区另需pycut-pursuit/grid_graph等，当前未安装。没有执行install.sh，没有使用地下数据，没有训练或新教师。源码仍固定于上一条记录的官方commit。
