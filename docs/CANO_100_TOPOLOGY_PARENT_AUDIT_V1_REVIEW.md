# Cano 100 Topology Parent Candidate Audit V1 复核

更新时间：2026-08-11  
正式状态：`FAIL_CANO_100_TOPOLOGY_PARENT_CANDIDATE_AUDIT`

证据目录：`results/gate0_baseline/gate0_20260811_cano_100_topology_parent_candidate_audit_v1_seed0/`

## 本轮做了什么

本轮不再复制 P01--P05 五个显式模板，而是直接调用固定 Cano checkout 的原生 `add_random_grown_tunnel` 与 `add_random_connector_tunnel`。固定 10 个 tree/loop/scale/flat/3D stratum，每层 12 个预声明 seed；每条请求隧道只有一次参数抽样和一次最多 100 internal trials 的 API 调用。成功 candidate 用同 seed 完整 replay，随后按每层前 10 个有效 parent 形成目标 80/10/10 split。

实际执行 120 个 primary candidate；60 个成功 candidate 均完成同 seed replay。没有 mesh、anchor、LiDAR、teacher label、训练或模型。

## 结果

| stratum | requested grown/connector | valid / 12 |
|---|---:|---:|
| S01 flat tree small | 4 / 0 | 7 |
| S02 3D tree small | 4 / 0 | 9 |
| S03 flat unicyclic small | 4 / 1 | 6 |
| S04 3D unicyclic small | 4 / 1 | 9 |
| S05 flat branch medium | 6 / 1 | 3 |
| S06 3D branch medium | 6 / 1 | 9 |
| S07 flat loop rich | 8 / 2 | 3 |
| S08 3D loop rich | 8 / 2 | 9 |
| S09 flat complex | 10 / 3 | 1 |
| S10 3D complex | 10 / 3 | 4 |

总计 `60/120` 通过当前机器合同。flat 为 `20/60=33.3%`，3D 为 `40/60=66.7%`。当前选择规则只能形成 `56 train / 4 validation / 0 development-test`，不能形成 100 parent 或正式 split，因此整轮 FAIL，60 个成功 parent 也不得转成训练集。

有价值的正结果是：60 个成功 parent 的 coordinate-bearing canonical identity 为 `60/60` 唯一，coordinate-free WL graph hash 也为 `60/60` 唯一；60/60 含 degree `>=3` 结构事件。说明 Cano 原生随机生成在成功时确实能产生结构多样性，问题是生成产率与 family 合同，不是仍在复制五模板。

## 失败根因

60 个失败全部发生在 grown tunnel，已到达 connector 的 candidate 没有 connector 失败。当前 adapter 对每条 grown 只抽一套 `GrownTunnelGenerationParams`；上游 API 的 100 trials 只在该固定参数下尝试不同起点/方向。如果该参数组合在当前网络里持续违反自碰撞、隧道碰撞、倾角或交角条件，整个 candidate 就停止。

固定 Cano 的 `generate_environments.py` 实际在 grown 失败后会重新抽参数并继续 `while not result`；它没有边界，不适合直接审计，但说明 V1 的“一条隧道只允许一次参数抽样”比原生成逻辑更严格。随网络规模增大，二维 flat 空间更快拥挤，因此 S05/S07/S09 下降到 `3/3/1`，而对应 3D 层为 `9/9/4`。

## 新发现的指标缺口

V1 只要求 `cycle_rank >= requested connector count`。这会漏掉 grown tunnel 意外与已有节点重合形成的额外闭环：

- `S01_flat_tree_small_C07` 请求 0 connector，但 cycle rank=`1`；
- `S07_flat_loop_rich_C04` 请求 2 connector，但 cycle rank=`3`。

因此 60 valid 是按封存 V1 合同得到的宽松计数；严格按 stratum family 至多 58 个。封存结果不回写，后续合同必须要求 `cycle_rank == requested connector count`。

## 可视化复核

已检查每层 accepted-rank-1 train parent 的 10 张完整 X-Y/X-Z 图：

- 10 张均正常渲染，无空白或 schema 损坏；
- 五个 flat parent 的 X-Z 均严格位于 `z=0`；
- 五个 3D parent 有清晰坡度和多高度层，平均 vertical span 从约 33 m 到 90 m；
- 随 grown/connector 数增加，分支、回环、覆盖范围和 centerline length 明显增加；S09/S10 的有效图达到约 2.6--2.8 km centerline；
- 图中部分 grown spline 视觉上绕成近环，但是否形成拓扑闭环必须以 graph cycle rank 为准，不能凭图片命名 family。

这些图支持“有效 parent 具有结构变化”，不支持“V1 已获得 100 parent”。validation/development-test 没有人工作为开发样本查看。

## 推荐 corrective V2

V2 不复用 V1 的 60 个 parent 作为正式 split，使用全新预声明的 120 个 seed，避免把已观察的开发候选伪装成未见 test。保持 10×12 strata 和 80/10/10 目标，做两个明确修正：

1. 同一 topology seed 内，每条 requested grown/connector 最多允许 20 次**参数重抽**；每次仍调用原生 API 的 100 internal trials。所有参数抽样、返回和失败均记录。达到 20 次仍失败则 candidate 无效；不更换 topology seed。
2. family 合同改为 `cycle_rank == requested connector count`，拒绝 grown 造成的额外闭环。

该方法是把上游无界 `while` 改成确定性、有上限、可重放的参数 rejection sampling，不是运行后补 seed。V2 仍只做 graph/spline，不生成 mesh/LiDAR/标签或训练；需要新 proposal/card 与用户批准。
