# Cano 100 Parent Perception Mesh M0F Review

状态：`PASS_CANO_100_PARENT_PERCEPTION_MESH_CONTRACT_M0F`  
日期：2026-08-11  
正式运行：`results/gate0_baseline/gate0_20260811_cano_100_parent_perception_mesh_contract_m0f_immutable_assets_seed0`

## 1. 本阶段到底做了什么

M0F 不是数据集训练，也不是规划器实验。它只回答一个材料问题：冻结的 Cano 程序化地下网络能否为十种 recipe 各生成一个可读取、与上游 graph/spline 完全对应、质量合格并可用 SHA-256 固定的原生感知网格。

固定输入是 V2R 的十个 train C01 sentinel，覆盖 flat/3D、tree、unicyclic、branch、loop-rich 和 complex。每个 parent 只生成一次 primary OBJ；没有 replay、补图、换 seed、重抽参数或失败后改门槛。下游必须读取已封存 OBJ，禁止重新 meshing 替代。

## 2. 正式结果

- 正式状态：PASS；executor exit code 0；运行约 590.6 s，runner 总耗时约 591.9 s。
- 原生 primary mesh：10/10；replay mesh：0；完整 X-Y/X-Z 图：10/10。
- source identity：graph、splines、operation trace、parent identity 全部通过。
- OBJ：10 个 SHA-256 全部唯一；110/110 个 evidence hash 独立复核一致。
- 总规模：1,243,070 vertices、2,486,245 triangles、185,916,640 OBJ bytes；整个 run 约 192 MiB。
- 退化 triangle：0；最小 largest-component fraction 为 0.9999353709。
- S04、S06、S08、S09 存在 2/5/2/3 个 triangle components，但额外 component 只占极小比例，全部超过冻结的 0.999 主分量门槛；不得把这些 mesh 宣称为 watertight、manifold 或 navigation collision asset。
- 所有 scope 计数保持为零：LiDAR、anchor、teacher label、formal dataset、training sample、model、trajectory、Gazebo/Isaac、M-TARE change。

## 3. 人工图像核验

十张完整图逐张查看，没有发现空白、裁剪、明显断裂、中心线越出网格覆盖、flat/3D 维度错误或缺失整条隧道。S09 平面复杂网络保持近零高差；S10 三维复杂网络显示约 53.65 m 高差。完整图位于：

`results/gate0_baseline/gate0_20260811_cano_100_parent_perception_mesh_contract_m0f_immutable_assets_seed0/previews/train_complete_maps/`

图中灰色是 mesh 确定性采样，蓝色是 grown tunnel，橙色是 connector，红叉是 degree >= 3 结构事件。它证明的是“材料覆盖和结构对应”，不是传感器可识别率。

## 4. 科学结论与边界

结论成立：单次生成后把 OBJ 当作不可变资产，是当前 Cano/Open3D 管线可审计、与真实下游消费方式一致的复现单位。M0--M0R4 已说明“重新生成同一 Poisson mesh 必须逐点相同”不是有效合同，因此 M1 不再产生 replay mesh。

尚未成立：没有 LiDAR 输入、没有监督标签、没有样本 NPZ、没有 CNN/GNN、没有在线结构语义、没有拓扑建图和 M-TARE 高层规划替换。M0F 只解除“100 个 parent 能否开始物化 perception mesh”的前置风险。

## 5. 下一步唯一建议

提交 M1 全 100 parent immutable perception asset proposal：固定 80 train / 10 validation / 10 development-test、每个 parent 一次 primary、零 replay、同一 source/quality/hash 合同。只人工查看 80 个 train 完整图；validation 与 development-test 只做冻结的自动质量核验并封存，避免用其视觉结果指导开发。

M1 仍不是正式学习数据集。M1 PASS 后才可另提 CPU LiDAR/客观监督标签的数据合同；不能从本次 PASS 直接跳到训练或修改 M-TARE。
