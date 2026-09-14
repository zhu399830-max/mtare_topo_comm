# 结构语义模型：训练、指标和当前问题（小白版）

## 一句话结论

模型已经能从实时局部表面地图预测方向、出口和开阔度等结构事实，并据此生成旋转不敏感的 64 维结构角色；它明显削弱了“看到的地图越多，特征就越像”的 coverage 捷径。但是，完整地下探索还没有成功超过 TARE。当前主要故障位于结构出口到局部运动的执行闭环，次要问题是未见几何上的方向预测误差。

正式判定是：`SEMANTIC_BOTTLENECK_ROLE_MIXED`。

## 模型到底学什么

输入不是环境类别标签，也不是预先规定的“这是 T 形路口”分类。输入是当前局部表面栅格和已经变换到当前机器人坐标系的历史局部表面。网络先预测 32 个方向上的可通行性、可达距离、可达面积、出口扇区，以及出口数、开阔度、瓶颈程度和主路连续性。

随后固定 canonicalizer 只读取这些显式结构事实，生成 64 维 canonical structural role。它不读取 encoder latent、世界名称、轨迹、绝对位置和 surface cell count。因此这是“先发现结构事实，再形成结构语义”，不是环境分类器。

代码证据：

- `Direct.forward` 是对照组：`encoder latent -> role`。
- `Semantic.forward` 是本方法：`encoder -> explicit heads -> role_module(q)`。
- `Role` 没有参数，也没有 encoder 引用。
- 打乱显式语义后 topology Spearman 从 0.248 降到约 0；全置零后角色退化，证明没有 latent 旁路。

## 数据划分和 forest 的含义

- 训练：tunnel + garage。
- 验证：forest，只用于在训练过程中选 checkpoint。
- 最终测试：campus + indoor，在 checkpoint 冻结后各评价一次。

`forest` 不是项目的目标场景，也不是用森林训练地下模型。它只是一个隔离开的验证集合，用来阻止我们根据训练集表现挑模型。真正的地下外部验证另外使用完全未参与训练的 Cave World 和 SubTGraph Operational 01/02。

## 训练轮数

Direct-role 对照训练记录为 31 轮。Semantic-bottleneck 共训练 70 轮；离线 forest 指标选择出的最佳 canonical checkpoint 是第 34 轮，在线地下部署另保存了第 70 轮模型。不是只训练一轮；`config.json` 中的 `epochs: 1` 是最后一次 resume 调用的增量参数，不是累计轮数，累计轮数以两个 `*_training.json` 为准。

训练时使用 coverage/方向数量分层重采样和可见性 dropout，降低观测面积与出口数量的伪相关。动态结构分支没有训练。

## 指标怎么读

- Topology Spearman：角色距离和真实结构差异的排序一致程度，越高越好。
- Coverage Spearman：角色距离是否被“看到了多少地图”支配，目标是低于 topology。
- Positive/hard-negative 排序：拓扑相同但覆盖不同的样本应该近；覆盖相似但拓扑不同的样本应该远。成功率越高越好。
- Top-K 检索：用一个位置的角色检索空间上分离但结构相近的位置，命中前 1/5/10 的比例。
- Rotation distance：同一结构换机器人朝向后的角色距离，越接近 0 越好。
- Nuisance probe：从角色反推 coverage、surface cell 等干扰量是否困难，同时反推出口、开阔度等结构量是否更容易。

## 离线核心结果

| 项目 | Direct role | Semantic bottleneck | 解释 |
|---|---:|---:|---|
| Forest topology Spearman | 0.229 | 0.248 | 结构相关提高 |
| Forest coverage Spearman | 0.183 | 0.038 | coverage 捷径明显削弱 |
| Pair 排序成功率 | 0.648 | 0.705 | 正负结构关系更正确 |
| Top-1 / Top-5 / Top-10 | 0.006 / 0.028 / 0.059 | 0.014 / 0.059 / 0.101 | 明显高于随机 0.003 / 0.014 / 0.029 |
| 旋转距离 | 0.000162 | 0.000066 | canonical role 更稳定 |

Teacher 显式事实能确定当前 role：teacher oracle 重建误差为 0，topology Spearman 为 1.0。Student 和 teacher 的 role cosine error 差距约为 0.081。把 student direction 单独替换成 teacher direction，误差改善 0.069，是最大的误差来源；替换 reachable distance 改善 0.010。

Indoor 泛化较好：semantic topology 0.697、coverage 0.322、pair 排序 0.911。Campus 较弱：topology 0.158、coverage 0.287、pair 排序 0.488。这也是结果只能判 MIXED 的原因之一。

## 在线地下仿真说明了什么

外部 Cave World 的同起点 600 秒对照中，本方法走了 776 m，却只覆盖 62 个 2 m 网格，重复访问率 89.7%；TARE 覆盖 229 个网格，重复访问率 61.8%。因此不能宣称本方法探索更快。该实验暴露的是大量回访和执行闭环问题。

SubTGraph Operational 01 的工程回归中，加入不可执行出口反馈后，相同 180 秒覆盖从 24 个网格提高到 68 个，最大半径从 18.4 m 提高到 29.9 m，说明问题定位是有效的。但新的 Operational 02 盲测仍在部分前进后卡住：300 秒只形成 10 个节点、9 条边、24 个网格。

代码检查发现，旧 watchdog 用“从目标发布以来累计移动量”判断失败。机器人只要先移动超过 0.5 m，随后永久卡住也不会拒绝目标。现在已改成“距目标的最近有效下降 + 8 秒无进展”并加入回归测试。修复后在同一 Operational 02、同一 180 秒窗口中，覆盖由 24 增至 73 个 2 m 网格，重复访问率由 86.7% 降至 59.4%，观测表面由 2707 增至 4702 个 1 m 网格，并建立 18 个节点、24 条边，正确拒绝了 6 个无进展目标。这证明该 bug 是因果因素；但 59.4% 的回访仍偏高，也还不是完整地图，所以不能把它写成最终探索成功。

此外，SubTGraph Operational 01 上原始 TARE 因缺少该外部 world 的边界/配置，启动后直接 Return home，不能作为公平 TARE 分数。当前公平的 TARE 对照只有 Cave World。

## 可以放入论文/汇报的图

1. `previews/fig2_topology_vs_coverage.png`：核心表示结果，结构相关与 coverage 干扰对比。
2. `previews/fig3_forest_retrieval.png`：跨位置 Top-K 检索。
3. `previews/fig4_pair_separation.png`：positive/hard-negative 分离。
4. `previews/fig5_dependency_audit.png`：语义瓶颈依赖审计。
5. `previews/fig6_error_propagation.png`：哪种显式语义误差最影响角色。
6. `../../external_cave_frozen_comparison/20260808_600s_v1/external_cave_trajectory_topology.png`：Cave World 中本方法与 TARE 的轨迹和拓扑图。
7. `../../subtgraph_execution_feedback_comparison/20260808_180s_v1/before_after_metrics.png`：执行反馈修复前后。
8. `../../subtgraph_operational_02_blind/20260808_300s_v1/topology_on_accumulated_map.png`：第二张未见地下图上的盲测拓扑。
9. `../../subtgraph_operational_02_recent_progress/20260808_180s_v2/before_after_metrics.png`：Operational 02 最近进展 watchdog 的公平前后对比。
10. `../../subtgraph_operational_02_recent_progress/20260808_180s_v2/topology_on_accumulated_map.png`：修复后 18 节点、24 边的地图坐标拓扑。

## 怎样复现

从项目根目录执行：

```bash
cd /home/zeng-workstation/mtare_topo_comm

# 从头训练两个版本；新建输出目录，避免覆盖冻结结果
python3 learning/structural_learning/tools/run_semantic_bottleneck_role.py \
  --dataset-root results/topological_semantic_dataset_v4_supervision_fixed \
  --out results/semantic_bottleneck_role/reproduction_70e \
  --epochs 70 --phase all --seed 20260807

# 生成论文图
python3 learning/structural_learning/tools/make_semantic_role_figures.py \
  --run results/semantic_bottleneck_role/20260807_polar_semantic_v5_final

# 验证最近进展 watchdog 修复
python3 -m pytest -q \
  learning/structural_learning/tests/test_semantic_topology_watchdog.py

# 无 Gazebo GUI，运行新的外部地下 Operational 02 闭环测试
RVIZ=false GAZEBO_GUI=false \
  bash scripts/run_semantic_topology_cmu_sim.sh \
  subtgraph_operational_02 closed_loop reproduce_op02_600s 600

# 如果本机显示环境正常，只打开 RViz，不开容易崩溃的 Gazebo GUI
RVIZ=true GAZEBO_GUI=false KEEP_CONTAINER=1 \
  bash scripts/run_semantic_topology_cmu_sim.sh \
  subtgraph_operational_02 closed_loop inspect_op02_rviz 600
```

RViz 中 `/semantic_topology/markers` 的绿色/红色球是实际地图坐标中的拓扑节点，蓝线是机器人真正连接出的图边；不是把特征向量画在坐标系里。

## 下一步唯一建议

冻结当前表示模型，不再增加训练轮数或换网络。把全局语义 waypoint 与局部规划器之间改成明确的目标状态机：`accepted / progressing / reached / rejected`，以局部规划器状态和最近距离下降共同消费或封锁出口。随后只在全新 SubTGraph 地图上做一次 600 秒盲测；成功标准是不会长期卡在同一目标，并显著降低重复访问率。只有该闭环通过后才值得继续比较完整覆盖时间和通信量。
