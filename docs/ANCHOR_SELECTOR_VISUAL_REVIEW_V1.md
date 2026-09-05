# Anchor Selector V1 科研复核

日期：2026-08-11  
对应运行：`gate0_20260811_cano_anchor_selector_zero_raycast_audit_v1_seed0`  
机器合同：`PASS_CANO_ANCHOR_SELECTOR_ZERO_RAYCAST_AUDIT`  
科研复核：`FAIL_INCOMPLETE_FULL_SPLINE_COVERAGE_METRIC`

## 1. 机器合同通过了什么

- 五个固定 topology 均输出恰好 50 个 anchors，总数 250；
- 全局最小两点欧氏距离为 `5.4985095 m`，高于冻结的 `5 m`；
- 结构事件候选均被事件 anchor 精确覆盖；
- 每个 tunnel 至少有一个 anchor；
- 同一输入的第二次选择 hash 完全一致；
- P01--P04 只读复用封存 graph/spline，P05 只构造一次 graph/spline；mesh、ray、正式数据、训练和模型均为 0。

## 2. 可视化为什么否决“采样已完成”

V1 用 tunnel 轮询控制**数量差**，但每条 spline 内部总是从 arc 起点向后选择最早可行点。达到全局 50 个后提前停止，因此长 tunnel 的后半段可能完全没有普通 anchor。P01 尾段与 P03 connector 上段在完整图中有明显空白。

对每条 spline knot 计算“到同一 source tunnel 最近 anchor 的欧氏距离”，得到最大未覆盖半径：

| Parent | 每 tunnel 最大未覆盖距离（m） | Parent 最大值（m） |
|---|---|---:|
| P01 | `1: 32.780` | 32.780 |
| P02 | `1: 11.555, 2: 16.326, 3: 16.564` | 16.564 |
| P03 | `1: 5.848, 2: 14.802, 3: 9.322, 4: 92.306` | 92.306 |
| P04 | `1: 10.064, 2: 7.503, 3: 10.501, 4: 9.736` | 10.501 |
| P05 | `1: 11.758, 2: 16.559, 3: 17.558` | 17.558 |

所以原机器指标中的 `selected-per-tunnel spread <= 3` 不是正确的空间均衡定义。不同 tunnel 长度不同，绝对数量接近反而会让长 connector 严重欠采样。该 PASS 只能证明 cardinality/spacing/replay，不足以证明全结构覆盖，也不授权 LiDAR rerun。

## 3. 推荐 V2 方法

1. 保留 terminal、junction、turn 事件 anchor 和全局 5 m 最小间距；
2. 按每条 tunnel 的 arc length 分配目标配额，不再要求绝对数量差不超过 3；
3. 在每条 tunnel 的完整 `[0, L]` 区间做等弧长/coverage-first 放置，而不是从一端 first-fit；
4. 对交叉口附近的跨 tunnel 冲突做确定性 phase shift 或有限回溯；
5. 新增核心验收：每条 spline 到**同 tunnel** anchor 的最大覆盖半径不超过 `7.5 m`，同时保留恰好 50、全局间距不低于 5 m、结构事件覆盖和 replay 一致。

V2 必须另立 proposal/run；不得回写或重新解释已封存 V1。
