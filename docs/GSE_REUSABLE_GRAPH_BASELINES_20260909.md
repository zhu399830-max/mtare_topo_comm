# 可复用建图代码：实际接口核对

2026-09-09官方非空骨架合成输出已保存：离散全已知T形、0.25m网格、2793自由体素逐项检查最近障碍parent与distance一致；原generateSkeleton+generateSparseGraph输出19节点25边、1连通分量、cycle rank7。图/原连接保存在docs/figures/gse_graph/official_skeleton_synthetic_t_20260909/graph.json及graph.svg。是空间规划骨架，不是1路口3分支的语义图；不能按degree直接评分语义路口或把边当已穿越。构建36690 exit0；无真实LiDAR/训练。下一核对原生图末端/角部附加分支与语义评价映射，明确算法原生结果和任何压缩适配分开，不调参凑树、不宣称真实基线通过。

2026-09-09官方骨架独立CPU构建完成：voxblox c8066b0、minkindr564f126与既定skeleton，复用Eigen，11个Ubuntu包仅解包到build/gse_voxblox_cpu_v1/prefix；未系统安装/改训练环境。自有CMake编译原核心，生成副本仅把序列化tmp_byte_offset由uint32改uint64兼容API，上游不改；补齐生成头目录。build86644 exit0，空ESDF调用generateSkeleton exit0。仅链接检查，0节点非研究效果。完整hash/补丁清单voxblox_standalone_build_v1.json。下一构造非空合成ESDF并验证distance/parent一致，再检查官方骨架恢复，暂不读取307或训练。

2026-09-09官方骨架源码已取得并固定：third_party/mav_voxblox_planning_reference提交a90260baaa07cd046847d7a237ab17e2947dbce7，BSD-3-Clause原许可保留，约1.6MiB。已读真实CMake/package与核心源码；7核心cpp未直接依赖ROS，示例程序依赖voxblox_ros，独立构建可行性待编译。算法使用ESDF observed/distance/fixed/parent，不能三值grid伪造。cmake/g++可用，系统开发包未检测到；未安装/编译/读数据。源固定清单configs/v3/environments/voxblox_skeleton_source_v1.json。下一固定voxblox核心依赖并准备隔离最小构建，仅合成输入，不改上游算法或训练环境。

## 本轮决定

不继续扩展“旧出口轴段必须在路口相交”的适配，也不把教师骨架当非学习预测。先核对官方 Voxblox skeleton 的独立构建和合成输入接口；不能直接拿已有三值观测网格冒充ESDF。PRISM保留为后续连续地点图比较，不作为本轮局部路口检测器。

本轮只读项目源码与官方网页，真实扫描、教师数据、checkpoint读取均0；无新训练、无正式run、未安装第三方依赖。这是接口选择证据，不是基线跑通或方法成功。

## 已有组件究竟是什么

| 组件 | 实际输入/输出 | 可复用部分与限制 |
|---|---|---|
| 本地 registered structural skeleton | representation文件定义类型与因果点云注册；teacher文件从graph/splines采样骨架 | 复用坐标注册；不能将真值骨架作为LiDAR基线 |
| 已保存 observed grid | state、free_frame_bits、occupied_frame_bits | 已观测自由/表面/未知证据；没有ESDF距离值，不证明支撑或可通行 |
| 旧出口＋局部轴线 | 水平长距离扇区、表面拟合曲线 | 保留方向/拟合诊断；不能把轴段端点或求交失败改写成真值路口 |

源码依据：`src/mtare_topo/teacher/gse_registered_structural_skeleton_teacher.py`的sample_physical_skeleton读取graph、splines并使用edge_arc_incidence；`src/mtare_topo/representation/gse_registered_structural_skeleton.py`并未实现从观测生成骨架；`src/mtare_topo/data/development_grid_export.py`的ARRAYS只有三种状态证据数组。

## 官方实现，不只是论文名称

**Voxblox skeleton**：官方仓库提供从ESDF构建骨架和稀疏图的模块。公开头文件的SkeletonGenerator接收Layer<EsdfVoxel>，提供generateSkeleton、generateSparseGraph及图访问接口。适合检查自由空间结构图；其图边是地图中的候选连接，不能直接称为机器人已穿越边。原系统面向飞行器，不能用其空间连通代替地面机器人支撑约束。

来源：[官方仓库](https://github.com/ethz-asl/mav_voxblox_planning)、[实际类接口](https://raw.githubusercontent.com/ethz-asl/mav_voxblox_planning/master/voxblox_skeleton/include/voxblox_skeleton/skeleton_generator.h)。本轮成功读取README和头文件；CMake及实现文件网页请求失败，尚未核实独立编译成本，也没有声称已编译。

**PRISM-TopoMap**：官方输入为里程计、点云及可选图像；包含地点识别与扫描匹配，并提供点云权重。节点增加包含扫描重叠和最大边长条件，不等于交汇中心。用于连续地点图/关联参考，需要明确传感器模式、权重、轨迹与评价对象，不能将地点节点与构造路口直接混评分。

来源：[官方README](https://raw.githubusercontent.com/KirillMouraviev/PRISM-TopoMap/main/README.md)。没有下载外部权重、数据或启动ROS。其README列出的GT pose选项必须在部署路径禁用；已知位姿诊断则需与主方法统一说明。

## 不把基线接入变成新的长期支线

下一动作限定为核对官方骨架模块的固定版本、许可证、构建依赖和合成ESDF入口。若不能独立复用，报告具体缺口，不重写一个新骨架算法冒充复现，也不安装整套探索系统来绕过接口问题。正式真实数据执行仍需绑定原C01–C07的307输入和新运行合同。

同时必须保持论文结论边界：当前BlockStructureReadout读取点块embedding、冻结context、中心、范围、退化标记，通过共享Transformer解码。已运行的R0/R1/R2主要比较分组；没有输入显式超点图边属性，也没有独立关系移除对照。注意力可能学习关系，但现有结果不足以宣称显式几何组合的独立贡献。源码：`src/mtare_topo/representation/gse_block_structure_readout.py`。后续基线完善不能代替该必要消融。
