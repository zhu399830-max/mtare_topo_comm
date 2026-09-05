# 地下结构语义与拓扑探索论文学习路线

日期：2026-08-10

这份阅读顺序围绕本项目的实际决策，不是泛泛列论文。先读最接近的方法，明确哪些不能再当作创新；再读全局探索、图规划、表示学习和数据集。

## 1. 最先读：与当前想法最接近

### Cano, Tardioli, Mosteo, 2026

论文：[Autonomous Navigation in Large-Scale Underground Environments Based on a Purely Topological Understanding of Tunnel Networks](https://onlinelibrary.wiley.com/doi/full/10.1002/rob.70157)

它把 VLP-16 的 3D LiDAR 转成 `16 × 720` depth image，由轻量 2D CNN 输出 360 个角度的出口响应，再通过 tracking/hysteresis 形成纯拓扑导航。训练集来自 50 个程序化地下环境，每个 10,000 样本，共 500,000 样本；标签由生成器中心线方向构成。

学习重点：

- 为什么他们用 range/depth image，而我们选择 terrain-relative BEV；
- 他们的标签是中心线出口方向，我们要做 planner-consistent R/D/G_local；
- 他们已经覆盖“合成地下 LiDAR + CNN 出口 + 拓扑图”，本项目必须在部分观测、不确定性、M-TARE 全局替换和多机器人上形成差异。

相关生成器：[Procedural Generation of Tunnel Networks for Unsupervised Training and Testing in Underground Applications](https://zaguan.unizar.es/record/148626/files)

## 2. 看清要替换的全局探索基线

### TARE

论文页：[TARE: A Hierarchical Framework for Efficiently Exploring Complex 3D Environments](https://publications.ri.cmu.edu/tare-a-hierarchical-framework-for-efficiently-exploring-complex-3d-environments)

学习重点：局部/全局层如何分工、coverage 与 frontier 如何定义、我们替换的是全局表示与目标选择而不是 local planner。

### Graph-based subterranean exploration

论文页：[Graph-based subterranean exploration path planning using aerial and legged robots](https://www.research-collection.ethz.ch/entities/publication/076e02d3-f139-4bd1-ac62-9fe87bd0af3c)

学习重点：稀疏图、frontier/dead-end、局部与全局图规划如何配合。它能帮助判断我们的结构语义图是否真正改善规划，而不是只换数据结构。

## 3. 理解对比学习能做什么、不能做什么

### PointContrast

论文：[PointContrast: Unsupervised Pre-training for 3D Point Cloud Understanding](https://arxiv.org/abs/2007.10985)

学习重点：对比预训练可提升下游 3D 表示与跨域迁移，但它本质上是 pretraining；它不会自动给出机器人 footprint 下的可通性、可达距离和隐藏连接真值。

### 合成监督 + 未标注真实域适配

论文：[Self-supervised domain adaptation for 3D semantic segmentation using synthetic data](https://link.springer.com/article/10.1007/s10514-024-10158-4)

学习重点：合成标注数据做主监督、真实未标注数据做 domain adaptation，是本项目以后评估 LAMP/SubT SSL 的更合理范式；仍必须与 synthetic-only baseline 消融。

### 轨迹经验自监督 traversability

论文：[Self-Supervised Learning of LiDAR Traversability for Off-road Autonomous Driving](https://arxiv.org/abs/2208.01329)

学习重点：机器人走过的轨迹可形成正样本，但“没走过”不等于不可通，不能替代完整地图上的出口连接 teacher。

## 4. 地下结构识别的几何 baseline

论文：[Unsupervised Junction Recognition in Underground Environments](https://arxiv.org/abs/2006.04225)

它使用 2D 点云和谱聚类识别地下 junction。学习重点：先建立不学习的几何/聚类 baseline，才能证明神经表示带来真实增益。

## 5. 真实地下数据集

### LAMP 2.0

论文：[LAMP 2.0: A Robust Multi-Robot SLAM System for Operation in Challenging Large-Scale Underground Environments](https://arxiv.org/abs/2205.13135)

用途：真实地下域的未标注预训练、输入鲁棒性或完整 site 测试，前提是逐 site 审计原始 scan、pose、ray origin 和时间同步。现有工程提取出的 LAMP KeyedScan/pose 配对不能直接当作 oracle teacher 数据。

### SubT-MRS

论文：[SubT-MRS Dataset](https://arxiv.org/abs/2307.07607)

它包含 30 多个地下场景和多模态传感器。用途同样是跨域与完整站点测试，不是拿测试站点参与 SSL 后再声称 unseen。

## 6. Isaac 只解决哪一层

- [Isaac Sim Synthetic Data Generation](https://docs.isaacsim.omniverse.nvidia.com/latest/synthetic_data_generation/index.html)：传感器数据采集、Replicator 与随机化。
- [Isaac Sim RTX LiDAR](https://docs.isaacsim.omniverse.nvidia.com/6.0.1/sensors/isaacsim_sensors_rtx_lidar.html)：LiDAR 配置与输出。
- [Isaac Lab Terrain Generator API](https://isaac-sim.github.io/IsaacLab/develop/source/api/lab/isaaclab.terrains.html)：seeded terrain/trimesh 生成接口。

Isaac 能批量渲染和随机化，但标准 terrain generator 不等于地下拓扑 grammar。我们仍需自己实现“结构图 -> 隧道/洞穴几何 -> collision/free space -> 同源 Isaac/Gazebo 导出”。

## 7. 建议学习顺序

1. Cano 2026：用一页纸写出其输入、标签、网络、数据规模、拓扑输出和局限。
2. TARE：画出原系统 global planner 到 `/way_point`、local planner、controller 的边界。
3. 本项目 `docs/GATE1_DATA_METHOD_V1.md`：逐项对照我们为何不同。
4. PointContrast 与 synthetic-to-real：理解 SSL 只是增强项。
5. LAMP/SubT：学习按完整 site 隔离，而不是按帧随机切分。

读完前三项后，应该能回答一句话：本项目不是训练“地下场景分类器”，而是从在线残缺 LiDAR 估计对规划有用的局部结构事实，并用它稳定建图和替换 M-TARE 的全局决策。

