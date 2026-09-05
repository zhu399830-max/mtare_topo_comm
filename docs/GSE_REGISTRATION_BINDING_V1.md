# 配准候选的时间、位姿与图节点绑定 V1

2026-09-05，Phase 3 CPU 软件支线。只做合成测试，0 真实点云/地图/模型读取、0 GPU、0 正式图运行。旧图、旧位姿接口及已交付的 ICP 模块均未修改。

新增 `gse_registration_binding.py`，把已实现的配准从“任意点云＋任意初值”接到部署可得的时序输入：

- `RobotFrameScan`：有时间戳的机器人局部坐标点云，米；有限性校验、不可变复制及规范字节 SHA-256。原始传感器坐标必须先按标定外参转换，不能直接混入机器人坐标。
- `RegistrationCandidate`：图内候选节点 ID、历史点云及对应完整 `DeploymentPose6D`；历史 scan/pose 时间必须一致。
- `register_candidate`：当前 scan/pose 时间一致，历史不晚于当前，两 pose 使用同一部署参考系；初值唯一来自 `inverse(T_ref_history) @ T_ref_current`，没有外部 initial 或 teacher transform 参数，然后实际调用已有 `register_local_clouds`。
- `BoundRegistrationEvidence`：记录节点、两扫描时间、点云哈希、完整输入姿态、初值、配置及配准证据，不输出节点合并决定。
- `register_candidates`：先验证全部候选，再依次返回每个候选结果。没有只取第一个通过者、分数排名或自动 winner；多个候选均通过时全部保留，歧义交给后续端口/轨迹联合验证。

重要坐标边界：historical pose 是**历史点云记录时的机器人姿态**，不保证等于持久图节点锚点。估计变换把当前机器人坐标转到历史点云机器人坐标；后续写图必须再显式组合历史点云到节点锚点的变换，不能把二者静默当成同一帧。

接口命名和字段不能证明调用方未伪造部署位姿或图内候选 ID。真实接线还需要候选来自在线图、位姿来自部署估计、点云为因果历史的来源检查；本模块不允许的只是直接传任意 GT 初值捷径。

## 验证

在既有 Open3D 0.19.0 CPU 环境运行：

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/zeng-workstation/.local/share/mtare_topo_comm/envs/cano_e1_topology_v1/bin/python \
  -m unittest discover -s tests/v3/unit -p 'test_gse_registration*.py' -v
```

实测：**35 tests 全通过，1.058 s，0 skip**。其中旧 ICP 22 项、新绑定 13 项；新绑定包含 12 项纯函数/mock 接线验证及 1 项实际 Open3D 合成集成。

集成样例：546 点非对称三维场景，真实旋转 0.03 rad，平移 `(0.07,-0.05,0.04) m`；部署初值仅估计 x 方向 0.02 m 且旋转为 identity，并不提供正确变换。经绑定调用实际 ICP 后双向检查通过，估计变换与合成真值元素误差 ≤1e-6；真值只用于造场景和事后断言。

其他测试检查：共同参考系与坐标变换方向、当前过期/未来 pose、历史未来帧、历史 scan/pose 不同步、后置坏候选使整个批次在 ICP 前拒绝、两候选均通过仍无 winner、失败结果不删除、输入复制与哈希、空候选、重复 ID、接口不存在任意初值或教师参数。

Torch 环境的组合回归为 **91 passed / 14 skipped / 0.22 s**；因该环境无 Open3D，14 个实际后端相关测试不在此执行，应引用上方专用 CPU 环境的真实执行证据，不能将 skip 计成通过。

## 尚未完成

这只是“部署扫描/位姿 → 真实配准证据”的窄接线。新 query head 未接入图；节点检索、传感器外参、历史扫描缓存、port matcher、轨迹上下文、多候选唯一性和最终图 adapter 仍未完整接通。只有几何候选通过不能自动合并节点，也不能据此宣称图质量或论文方法已通过。
