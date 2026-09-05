# Phase 3 多任务结构特征与结构语义合同 V1

状态：`PROPOSED_NOT_TRAINED`  
数据：sealed Cano V2R（80 train / 10 validation；C10 与 M-TARE 隔离）

## 研究问题

仅由局部 `16x720 range_m + valid_mask`，能否在未见 topology parent 上学习
稳定的结构作用表示 `z_role`，同时恢复可通行分支方向、分支数量和
`interior/junction/terminal` 结构角色，为后续在线 exit-stub 拓扑提供可靠观测？

## 输入与目标

- 输入通道 0：`clip(range_m, 0.3, 50)/50`；无效像素保持 1.0。
- 输入通道 1：`valid_mask`，取值 0/1。
- 禁止输入 pose、yaw、absolute xyz、parent/split/role ID、tunnel、graph、spline、oracle 或未来帧。
- 方向目标：720-bin `exit_target`；客观 spline/LOS teacher。
- 数量目标：`branch_count`；类别 1--6，训练内部映射为 0--5。
- 角色目标：cluster 共享的 `interior/junction/terminal`。

## 模型与对照

- B0：冻结 `RangeExitBaseline`，不调参。
- B1：与 M1 同 encoder，只保留 direction head；Cano-like adapted reproduction。
- M1：2-channel circular range-image CNN，azimuth 维 circular padding，ring 维普通 padding；
  azimuth 仅下采样到 180 列（2° latent resolution）后恢复 720 bins，避免 45 列/8°瓶颈；
  共享 encoder 输出 128D L2-normalized `z_role`，接 direction/count/role heads。
- 第一版不使用 GNN、Transformer、SSL、AI 标签、点云网络或 BEV 重建。

## Loss

主任务为方向语义；数量和角色为辅助监督：

```text
L = L_direction + 0.25 * L_count + 0.25 * L_role
L_direction = BCEWithLogits（正负不做 validation 调权）
L_count = CrossEntropy
L_role = class-balanced CrossEntropy（权重仅由 train cluster 统计冻结）
```

不在第一次正式 run 中搜索 loss weight、阈值或模型宽度。方向 component 的 circular
grouping 和检测阈值只能由 train-only 内部 calibration 或另行批准的 validation
threshold calibration 冻结；第一训练 run 先保存 logits/targets，以固定默认 0.5
报告，并明确它不是最终 planner threshold。

## Split 与抽样

- train：80 world、20,000 clusters、100,000 frames。
- validation/checkpoint selection：10 unseen parent、2,500 clusters、12,500 frames。
- strict test：10 个 C10 parent，训练、归一化、增强、阈值和 checkpoint 选择均读取 0。
- mini-batch 以 cluster 的五视角为连续基本分组并确定性打乱；保留自然数据分布，role
  不做重复采样，仅在 role CE 中使用 train-cluster 统计得到的 class weight。epoch 报告实际
  unique frame/cluster coverage。
- normalization 常数 50 m 来自传感器合同，不从 validation/test 估计。

## 表示与语义指标

- Direction：component precision/recall/F1、matched angular error、branch-count consistency。
- Role：macro-F1、每类 P/R/F1、confusion matrix；以 cluster 聚合五视角 logits 为主，frame 指标为辅。
- Count：1--4 macro-F1/accuracy 和逐类指标；5--6 只作稀有诊断。
- `z_role`：同 cluster 五视角 cosine、一致旋转前后 feature/output 等变、固定遮挡/稀疏扰动稳定性；
  frozen embedding 的 train-only fit / validation-world linear probe 与 nearest-neighbor role retrieval。
- 工程：参数量、单帧 batch-1 latency、吞吐、峰值显存。

## 第一轮训练预算与停止条件

- 3 个固定 seed：0/1/2；每 seed 最多 30 epochs，early stopping patience=6，选择 validation
  direction F1 为主、role macro-F1 为并列审计，不以单一 train loss 选 checkpoint。
- RTX 5090 32 GB；预计总 GPU 2--4 小时、checkpoint/日志/预测不超过 8 GB。
- 新建隔离训练环境，固定 PyTorch 2.9.0+cu129、torchvision 0.24.0+cu129、Zarr 2.18.7；
  不修改 E1 数据导出环境。
- 任一 source seal、split、label alignment、非有限 loss、跨 seed 崩溃、validation 泄漏或证据缺失立即停止。
- M1 若没有同时优于随机/多数类、没有稳定 `z_role` 证据，或 direction 明显不及 B0，结论为 FAIL/MIXED；
  不通过加 head、换 GNN、看 C10 或修改 M-TARE 规避。

## 输出证据

每 seed 保存冻结 config、command、environment、source/data seal、raw log、epoch metrics、best/last
checkpoint、validation logits/targets、角色混淆矩阵、direction PR/F1/角误差、count 分层、embedding
稳定性和失败案例。只画完整、固定规则的 train/validation 科研图；不查看 C10/M-TARE。
