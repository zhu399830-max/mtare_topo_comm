# 可运行纵向链路 V1：怎么读、怎么核验、怎么重跑

状态：`DEVELOPMENT_PROTOTYPE / NOT_UNSEEN_TOPOLOGY_EVIDENCE`

## 1. 它现在做了什么

```text
已有 Cano seed-0 world
→ 沿完整 source graph 构造 240 帧因果遍历
→ 每帧 16×720 Open3D CPU first-return LiDAR
→ 当前帧 range-sector 规则预测 outgoing branches
→ OnlineTopometricGraph 增量生成节点、物理移动边和 exit stubs
```

完整 spline/graph 不进入 LiDAR 规则，只用于生成客观 branch label、oracle 上界和最终评测。当前没有神经网络、checkpoint、ROS、闭环或 M-TARE 修改。

建议按以下顺序读代码：

1. `src/mtare_topo/semantics/range_exit_baseline.py`：一帧 range image 如何变成出口方向；
2. `src/mtare_topo/topology/online_topometric.py`：出口事件如何增量形成节点、边和 unexplored/traversed stub；
3. `tools/v3/run_cano_lidar_topometric_vertical_slice.py`：轨迹、raycast、oracle、指标、图和可视化如何串起来；
4. `tests/v3/unit/test_vertical_slice_components.py`：最小行为合同。

## 2. 当前结果怎么读

权威目录：

```text
results/prototypes/cano_seed0_lidar_to_topometric_vertical_slice_v2_graph_refined/
  metrics/summary.json                 总指标与结论边界
  metrics/per_frame.json               240 帧逐帧出口、角色和匹配记录
  metrics/association_sweep.json       同轨迹 81 组图参数选择全过程
  artifacts/trajectory_scans.npz       range/valid/label/pose/yaw 数组
  artifacts/lidar_rule_online_graph.json
  artifacts/oracle_online_graph.json
  artifacts/route_manifest.json
  previews/vertical_slice_summary.png  完整地图、轨迹、两张最终图和时序统计
  previews/branch_diagnostic_24_frames.png
  previews/online_topometric_replay.gif
  artifacts/evidence_sha256.txt
```

`trajectory_scans.npz` 是多个 NumPy 数组的压缩容器，不是 JSON。这里包含：

- `range_m`: `(240,16,720)`，米；
- `valid_mask`: `(240,16,720)`；
- `label_720`: `(240,720)`；
- `sensor_xyz_m`, `axis_xyz_m`: `(240,3)`；
- `yaw_deg`: `(240,)`。

读取示例：

```bash
/tmp/mtare_cano_compat_e1_py312_np126_sp112/bin/python - <<'PY'
import numpy as np
p = "results/prototypes/cano_seed0_lidar_to_topometric_vertical_slice_v2_graph_refined/artifacts/trajectory_scans.npz"
with np.load(p) as d:
    for key in d.files:
        print(key, d[key].shape, d[key].dtype)
PY
```

## 3. 当前数字与边界

- 240 帧、1,676.46 m、2,764,800 条 primary rays；
- branch precision/recall/F1：`0.7480/0.7525/0.7503`；
- 平均匹配角误差：`6.985°`；
- rule graph：43 nodes / 50 edges / 14 unexplored stubs；
- oracle graph：44 nodes / 50 edges / 12 unexplored stubs；
- node F1：`0.8276`；直接几何 edge F1：`0.94`。

这些数字只属于一个已经检查过的开发 world。图关联参数是在同一轨迹的 81 组候选中选择的，不能作为 unseen-topology 泛化结果。branch F1=`0.7503` 也说明规则感知仍有明显漏检和误检；后续学习模型要优先解决这一层，而不是继续在单图上调图参数。

## 4. 核验与重跑

项目单元测试：

```bash
/home/zeng-workstation/anaconda3/bin/python -m pytest tests/v3/unit -q
```

Cano 外部合同测试使用冻结 E1，不向默认环境安装依赖：

```bash
/tmp/mtare_cano_compat_e1_py312_np126_sp112/bin/python \
  -m unittest tests.v3.external.test_cano_adapter_contract -v
```

核验 v2 已封存文件：

```bash
sha256sum -c \
  results/prototypes/cano_seed0_lidar_to_topometric_vertical_slice_v2_graph_refined/artifacts/evidence_sha256.txt
```

如需重跑，必须给一个**新的输出目录**；脚本会拒绝覆盖 v2：

```bash
/tmp/mtare_cano_compat_e1_py312_np126_sp112/bin/python \
  tools/v3/run_cano_lidar_topometric_vertical_slice.py \
  --output results/prototypes/manual_replay_YYYYMMDD
```

该重跑大约执行 276 万条 CPU rays 并生成 PNG/GIF。它只验证确定性复现，不得通过反复重跑挑选更好结果。

## 5. 下一步

冻结 v2 的规则和图参数，在 5 个新 topology 上无调参重放。逐 topology 报告 branch/node/geometric-edge 指标后，才能判断：规则是否足够当工程 baseline、训练数据要覆盖哪些失败类型、以及小 CNN/ResNet18 exit head 是否真正优于规则。
