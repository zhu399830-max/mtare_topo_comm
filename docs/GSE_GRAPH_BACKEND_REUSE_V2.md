# GSE 图后端复用与端口更新 V2

2026-09-05。当前是 Phase 3 的独立 CPU 软件支线，研究问题仍为学习几何组合是否改善结构建图。本次读取 0 个世界资产、0 个数据集帧、0 个模型权重；没有正式图回放、训练或科学 PASS。旧冻结源文件和历史运行未修改。

## 已有实现实际做到什么

`src/mtare_topo/topology/gse_causal_graph.py` 的 `CausalStructuralGraph.update` 已能在首次稳定事件建立节点；`depart` 加连续 `PoseEstimate` 可将连续经过的节点分别连边；`snapshot` 支持前缀回放验证。

有三个具体缺口，不能继续描述成完整建图后端：

1. `ports` 和 `port_states` 只在新节点时写入。`verified_revisit` 只选已有节点；`event_pending` 提前返回，同一节点的新端口和几何都没有增量更新。
2. `AssociationEvidence.accepted` 只对调用方传入的四个布尔量求逻辑与。它没有点云输入、估计变换、残差、对应点数或退化检查；并没有执行真实配准。
3. 旧 `PoseEstimate` 只有 xyz 和位置对角协方差，没有姿态；`StructuralPort.direction_robot` 又位于机器人坐标。跨朝向重访直接比较或覆盖该方向会混坐标。现已新增独立 6DoF frame adapter，但尚未替换或接线旧图。

## 可复用入口与不能冒充的功能

| 入口 | 实际输入与能力 | 复用边界 |
|---|---|---|
| `CausalStructuralGraph.update/depart` | 结构观测、部署位置、外部关联证据、连续轨迹 | 保留状态机，不宣称已有真实配准或端口更新 |
| `primitive_relation_nonlearning._register_points` | range/valid、已给定五帧平移和 yaw；将回波转到当前坐标 | 是已知运动下的点变换，不是从点云估计运动的 ICP |
| `primitive_predicted_geometry_association_slice` | 预测基元端点、评分对齐目标与掩码；计算距离和学习关系诊断 | 是带评分教师的离线诊断，不可直接作为部署配准或候选接口 |
| 新 `gse_port_updates.update_ports` | 已转节点坐标的端口、图内端口表、逐观察显式候选证据 | 纯函数端口生命周期；尚未接入旧状态机 |
| 新 `gse_graph_frames.ports_in_node_frame` | 同参考系的节点锚点/当前部署 R、t、完整 6×6 协方差及同步结构观测 | 将机器人方向转持久节点方向；不估计配准、不产生端口位置 |

在 `src`、`tools/v3`、`tests/v3`、`integration`、`external` 的 Python/C++ 源文件范围，搜索 `registration_icp`、`IterativeClosestPoint`、GICP、point-to-plane 等入口未找到可直接调用的点云配准实现；对 integration/external 也执行了不受 ignore 规则影响的同范围复查。该结论不涵盖系统全局安装、仓库外 ROS workspace 或未登记第三方代码，不能据此称整个机器不存在配准库。

主 agent 另以安装元数据核实 `/home/zeng-workstation/.local/share/mtare_topo_comm/envs/cano_e1_topology_v1/bin/python` 环境已有 Open3D 0.19.0 / NumPy 1.26.4，可作为下一步成熟 CPU 配准库候选；安装存在不等于 adapter 或配准实验已完成。后续只需薄适配，不重新实现 ICP，也不把上面的已知位姿点变换包装成估计器。薄适配至少应返回估计 SE(3)、对应数量、双向残差、重叠与退化状态，并保留失败原因；阈值需来自新实验合同，本文不新增科学阈值。

## 新纯函数端口更新合同

`PortGeometry.direction_node` 明确在持久节点坐标；调用方先依据部署姿态转换，函数不接受或推测原机器人坐标方向。`ObservedPort.observation_local_id` 和 `GraphPort.graph_local_id` 是不同命名空间，数值相等不表示匹配。

`PortMatchEvidence` 绑定节点、时间和观察端口，提供全部候选图端口及几何验证结果。它仍是**证据消费接口**，不是证据生成器；无教师字段也不能防止调用方偷偷用教师构造对应，因此部署接线仍需追踪输入来源。

- 每个观察必须恰有一条证据；漏证据不等于无匹配。
- 单候选、该候选未被任何其他观察竞争、且几何验证通过，才能更新几何。
- 候选为空表示调用方已经完成搜索但未匹配，分配新的图内 ID。
- 多候选、竞争候选或验证失败均保留 pending 观察，不合并，也不伪造重复确认端口。
- 新端口为 observed；更新已有端口不得重置 attempted、traversed、temporarily_failed 或 failed。感知更新不负责改变执行状态。
- 未再次看到的端口保留，不把遮挡当成端口消失；最新验证几何替换旧几何，不引入未经校准的平均权重。
- 所有输入不可变，输出新表与决策；严格递增时间、图内 ID 存在性和证据绑定在任何输出前验证。

这个模块不建节点、不合并节点、不提交边；全部测试都是软件合成样例。

## 六自由度坐标适配

新增 `gse_graph_frames.py`：`DeploymentPose6D` 必须显式传入局部到共同部署参考系的 R、t、时间戳与 6×6 协方差；拒绝反射、尺度、非有限数据及非对称/非半正定协方差。协方差顺序为局部切空间 `[tx,ty,tz,rx,ry,rz]`（米/弧度），仅验证并原样保存，不把旧位置对角协方差补零冒充姿态不确定性，不宣称已做传播或校准。

节点坐标锚定其记录位姿，`node_from_robot` 计算 `inverse(T_ref_node) @ T_ref_robot`；`ports_in_node_frame` 只用相对旋转作用方向，宽高、置信度及观察局部 ID 保持。机器人姿态必须与观测时间完全一致；节点锚点可以早于当前时刻但不得来自未来；两个位姿必须处于同一参考系。数组输入复制为不可变 tuple，不保留可写别名。

旧 `StructuralPort` 没有端口位置，因此本 adapter 不从机器人原点伪造端口位置。后续模型若输出端口位置，需在新的带位置接口中显式变换。当前也不负责更新图位姿、确认关联或传播协方差。

## 实测与剩余接线

命令：

```bash
env PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python \
  -m pytest -q tests/v3/unit/test_gse_graph_frames.py tests/v3/unit/test_gse_port_updates.py tests/v3/unit/test_gse_causal_graph.py
```

首批端口结果：46 passed in 0.10 s，包括 32 项新端口测试及 14 项旧图回归。补 frame adapter 后：**70 passed in 0.16 s**，新增 24 项坐标测试。覆盖执行状态保持、双向唯一候选、同图端口竞争、模糊/失败留存、输入顺序不变性、前缀不变性、空观察、错误时间/节点/候选、真假布尔、非法几何，以及任意共同 SE(3) 不变、同位姿、旋转重访、roll/pitch、节点坐标平移、反射拒绝、6×6 协方差与旧 3DoF 输入拒绝。未使用 GPU。

剩余工作顺序：从真实部署接口接收完整姿态/协方差并接入已完成 frame adapter → 基于已有 Open3D 环境的真实局部点云配准 adapter 和失败测试 → 端口候选与唯一性证据生成 → 在新版本图 adapter 中接入端口表及 pending 生命周期 → 合成连续回放验证首次节点、停留、重访、探索状态与边的完整链路。旧冻结图必须保留，不能直接修改而造成历史源哈希漂移。

只有后续实际接线和模型/轨迹回放证明图收益，才属于论文方法证据；本次只是补齐一个可独立验证的软件缺口。
