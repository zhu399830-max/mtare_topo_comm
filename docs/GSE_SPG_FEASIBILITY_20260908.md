# 经典超点几何分块：实现边界和隔离依赖

本轮零地图读取、零标签、零训练、零安装。只读官方源码、Git引用及本机依赖。结论是有可复用的独立几何分块接口，但本机尚未满足编译依赖，不能宣称已经运行。

## 官方算法中只取哪部分

XYZ → 45邻居局部几何特征 → 10邻居图 → cut-pursuit分块 → components / in_component。沿用原合同voxel=.03、reg=.1、edge_weight参数1及verticality倍权2；无需RGB、标签或旧PyTorch训练栈。[官方分块脚本](https://github.com/loicland/superpoint_graph/blob/0209777339327c9b327b6947af6c89b20bb45981/partition/partition.py)。

原程序随后调用`compute_sp_graph`进行Delaunay图构造；本项目不运行该部分。几何邻居只用于分块，不代表地面机器人可以穿越。[官方图实现](https://github.com/loicland/superpoint_graph/blob/0209777339327c9b327b6947af6c89b20bb45981/partition/graphs.py)。

原作者说明分块具有随机性，故必须记录并实际验证重复运行和点序变化，不预先宣布确定性。[官方说明](https://github.com/loicland/superpoint_graph)。

## 已解析的确切源码版本

- SPG `ssp+spg`：`0209777339327c9b327b6947af6c89b20bb45981`，通过`git ls-remote`核对。
- 该commit树中的`partition/cut-pursuit`子模块：`e9501d158c94c286b3cd84563f0636268886fa50`，通过GitHub Git tree API核对。
- cut-pursuit当前HEAD为`ba155e45754057cb2b150f439f2379298b52fa69`，与上述gitlink不同，**不混用当前HEAD**。

本次只解析版本，尚未下载、编译或冻结二进制。网页分支源码用于机制检查，后续以确切commit源码为准。

## 依赖与成本边界

本机有cmake、g++、pkg-config、系统Python3.12头文件。所检查的系统及Anaconda标准路径没有Boost Python和Eigen头文件，系统指定路径也没有Boost Python/NumPy 3.12库；冻结训练解释器此前检查不到libcp/libply_c。不能将其表述为全盘不存在依赖。

官方`libply_c`需要Eigen、Boost、Python与NumPy开发接口；固定子模块的CMake还按Python版本解析Boost NumPy库。[几何特征构建文件](https://github.com/loicland/superpoint_graph/blob/0209777339327c9b327b6947af6c89b20bb45981/partition/ply_c/CMakeLists.txt)、[固定cut-pursuit构建文件](https://github.com/loicland/cut-pursuit/blob/e9501d158c94c286b3cd84563f0636268886fa50/src/CMakeLists.txt)。

不能把旧文档的Python3.6链接命令直接用于当前3.13训练解释器。下一执行独立前缀内的源码与依赖获取、构建及纯合成软件烟测；不sudo安装、不修改系统或冻结训练环境。提前记录包版本、SHA和命令；若旧接口不兼容，记录具体编译错误后退出本候选，不做无限移植。

烟测只验证：全部输入点有分块对应，组件索引范围合法，空/退化输入明确处理，能保留原点及五帧来源映射，记录重复执行差异和资源。随机性不能靠挑一次结果掩盖。它不是感知成绩或训练资格。

只有独立构建与映射合同通过，才冻结同已有观察的表示提取对照；当前仍不训练，仍不复活失败的集合读出。超点只是待比较的几何表示，原论文问题与图收益验收不改变。
