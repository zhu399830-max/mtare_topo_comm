# 将配准结果接到持久节点坐标 V1

2026-09-05，Phase 3 CPU 合成软件支线；0 真实地图/帧/模型/GPU，0 正式 run。旧冻结图、位姿、配准和绑定模块均未修改。

## 解决的具体缺口

前一配准返回的变换将**当前机器人坐标映射到历史扫描时的机器人坐标**。历史扫描机器人姿态通常不是持久节点锚点；直接把该变换称为“当前到节点”会造成系统坐标错误。

新增 `gse_registered_node_frame.py` 的 `registered_node_frame`，输入既有 `BoundRegistrationEvidence`、显式图内节点 ID 和持久节点锚点 `DeploymentPose6D`。唯一计算为：

```text
T_node_current = inverse(T_reference_node)
                 × T_reference_history
                 × estimated_T_history_current
```

最后一项必须是已接受配准的实际估计值，不能换成原部署初值。历史点云姿态与节点锚点明确独立；锚点可以建立于历史扫描之后，但不得晚于当前观测，两个时刻不被强行改成相同。

接口拒绝：节点 ID 不符；绑定扫描/位姿时间不一致；历史扫描或节点锚点来自未来；参考系不一致；配准失败、未知变换、矛盾拒绝原因/运行错误或非 SE(3) 估计。任何拒绝都不回退到旧位姿，不更新图。

输出 `RegisteredNodeFrame` 包含节点 ID、当前时间、估计变换、锚点以及完整原绑定证据；**不生成协方差、不校准不确定性、不确认全局地点身份**。输入的部署协方差仅保留在原位姿 provenance 内，不能当作已传播的配准协方差。

可选 `registered_ports_in_node_frame` 将结构端口方向旋转到上述候选节点坐标，保留全部局部 ID、宽高和置信度，要求观测时间一致，并复核 frame 未被改写。原端口没有位置，不伪造位置；它也不进行端口匹配、候选排序或节点合并。

同一观察的两份配准候选即使均通过，也输出两份独立 frame 假设，不能用此模块选择 winner。`BoundRegistrationEvidence` 是受类型和一致性检查的证据对象，不是防伪凭证；部署接线仍需保证它确实由已验证的输入链产生。

## 验证与限制

新增 14 项纯合成测试，验证非零 roll/pitch/yaw 与平移乘积、节点锚点与历史扫描不同、拒绝未来/错 ID/错参考系、失败/未知/非 SE(3) 注册不得 fallback、端口方向不加平移、时间绑定、篡改 frame 拒绝、两个通过候选保留歧义、不制造协方差及确定性。

Torch 环境联合配准/绑定/frame/port/旧图回归：**105 passed / 14 skipped / 0.26 s**。14 个 skip 是该环境无 Open3D 的后端相关测试，真实后端执行证据见 `GSE_REGISTRATION_BINDING_V1.md` 的专用 CPU 35 项全通过，不计作本次新增方法实验。

在既有 Open3D sidecar 使用标准库 unittest 独立执行本文件，**14 tests 全通过，0.015 s**；这些是 frame 消费测试，使用明确标为 `synthetic-frame-only` 的合成配准证据，不声称本轮又运行 ICP 或真实数据。

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/zeng-workstation/.local/share/mtare_topo_comm/envs/cano_e1_topology_v1/bin/python \
  -m unittest discover -s tests/v3/unit -p test_gse_registered_node_frame.py -v
```

下一步仍为部署端口候选/对应与轨迹联合证据，以及新图 adapter 对首次节点、重访和探索端口状态的完整接线。新 query head 尚未接图；本次坐标乘积正确不等于图质量或论文方法成功。
