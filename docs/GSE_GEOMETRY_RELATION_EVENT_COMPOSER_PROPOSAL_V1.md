# GSE Geometry-Relation Event Composer 方案 V1

状态：`SUPERSEDED_BY_GSE_GRAPH_METHOD_SPEC_V1`

> 2026-08-29：本文的单一五类 Composer 已被
> `docs/GSE_GRAPH_METHOD_SPEC_V1.md` 取代。新规范使用 corrected causal
> change-point Teacher，并将 junction/terminal 的 action-set relation 与
> turn/geometry-transition 的 metric change 分开建模。本文仅保留为
> 方法收敛过程记录，不得据此创建新训练。

## 要回答的问题

从五帧 LiDAR 学到的出口布局、出口几何、跨帧关系和连续通道几何，能否比独立 CNN 事件分类头更可靠地识别 `junction / terminal / turn / geometry-transition`，并让这些可解释量真正决定在线拓扑节点？

V2R5 完整三 seed 运行是固定基线，不属于本方案结果。它完成并封存前，本方案不训练、不选择阈值、不读取 C09/C10、不建图。

## 方法因果链

```text
5-frame causal LiDAR
  -> frozen circular directional encoder
  -> count-conditioned exit tokens
       heading in robot route frame
       opening width + vertical profile
       geometry uncertainty
  -> persistent / reveal / withdraw transport
  -> causal width / height / slope / curvature sequence
  -> Geometry-Relation Event Composer
  -> five structural-event probabilities + calibrated refusal
  -> structural node trigger and learned node/exit association
       (place/exit descriptors enter association here, never the Composer)
  -> edge only after physical traversal
```

机器人前向只定义已知 route-frame 坐标，不作为学习贡献。`local_axis` 若保留，只是辅助输出；它必须在独立 yaw 扰动验证中优于常数前向基线，现有 route-aligned `axis<=10deg` 不再作为科学证据。

## Composer 的唯一允许输入

每个样本只允许使用下列部署时可得的模型输出：

- 五帧 `token_count_probability`；
- 五帧有效 token 的连续相对方位、opening width、vertical profile 和 geometry uncertainty；
- 四个相邻帧的 persistent/reveal/withdraw 概率及 dustbin 质量；
- 五帧因果 width、height、slope、curvature 及其一阶变化；
- 历史有效 mask。

禁止输入：encoder context、任意隐藏 feature map、place/exit descriptor、原独立 event logits、原 observation uncertainty、GT event/exit/node identity、pose、world、traversal、TNG、未来帧、graph state 和 planner state。descriptor只允许用于后续节点/出口关联；当前observation uncertainty由旧event head的正确率目标训练，也不得成为旁路。这样可以直接审计事件是否确实由几何关系组成，而不是从隐藏表示绕过。

## 最小实现

1. 复用 V2R5 的五帧 backbone、token、transport 和 geometry heads。
2. 同一个 geometry head 逐帧作用于 causal `frame_context`，得到五帧连续几何序列；不新增几何参数。
3. 第`k`个NMS token的连续有效权重固定为`P(count > k)`，不使用hard count或存在阈值；对每帧token set做permutation-invariant汇聚，保留期望数量、圆周一/二阶矩、opening/profile的有效权重统计和geometry uncertainty统计。
4. 对 transport 保留三类期望数量、接受质量、dustbin 质量和跨帧变化。
5. 将上述显式量及连续几何差分输入小型MLP composer，输出五类事件。拒绝分数只由composer event probability/entropy和独立composer calibration产生，不复用旧event uncertainty。
6. 原 `context -> event` 头原样保留，只作为同 backbone、同数据、同 seed 的独立事件基线。

第一版冻结 backbone 与全部几何/token/transport heads，只训练 composer，避免事件 loss 反向把“可解释中间量”变成未经验证的隐藏通道。Composer参数输入维度、层数和总量必须在readiness中精确冻结；不得把descriptor或context扩维为隐式旁路。只有冻结 composer 证明几何关系具有增益后，才允许另立端到端消融。

## 固定数据和选择

- C01--C06：60 worlds，142,184 observations，仅训练 composer；
- C07：10 worlds，21,548 observations，选择 checkpoint、温度和拒绝阈值；
- C08：10 worlds，24,394 observations，一次零适配迁移；
- seeds 固定 `0/1/2`，每个 composer 绑定对应 V2R5 seed checkpoint；
- C09/C10、M-TARE、graph 和 planner 在感知门通过前保持零读取。

## 训练前 Teacher—输入可观测性门

现有Teacher允许读取完整客观几何：`turn`由当前点前后局部spline的heading change定义，`geometry-transition`由当前点前后各5 m的native-mesh width/height中位差定义。这不是学生输入泄漏，但可能让事件时间点超出只看当前/过去显式输出的可观测范围。

因此在任何composer optimizer step前，必须只用C01--C06完成零训练proof：

- junction/terminal 标签与past-only count/token/reveal/withdraw的可分人口；
- turn 标签与允许的bearing变化、curvature、vertical profile之间的可分性；
- geometry-transition标签与五帧width/height、exit opening/profile及其变化之间的可分性；
- 标签相同但允许特征互相矛盾、以及允许特征近同但标签不同的冲突人口；
- 禁止把future geometry、route arc、node/edge identity或独立event logits加入feature来修复冲突。

若oracle显式输入仍不能使turn/transition相对多数类获得非平凡precision/recall，判定为Teacher—输入时序不匹配，停止composer并重新评估事件时间语义；不得先训练、更换阈值或读取C07/C08绕过。

## 必须比较的对象

1. V2R5 独立 `context -> event` 头；
2. 仅 count + bearing 的 Cano-like composer；
3. 完整 Geometry-Relation Event Composer；
4. 完整方法去掉 metric geometry；
5. 完整方法去掉 transport；
6. 完整方法改为单帧。

这些对照共同回答：收益究竟来自普通出口数量、五帧时序，还是显式几何结构关系。

## 进入图之前的证据

- 三 seed 平均 event macro-F1 相对 V2R5 独立头绝对提高至少 `0.05`，至少 `2/3` seed 达到该提升，任何 seed 不得低于其 V2R5 对应结果；
- C07 和 C08 的 `turn`、`geometry-transition` F1 都必须分别高于 V2R5，不允许只靠 corridor/junction/terminal 多数类提高总体；
- uncertainty refusal 在 precision `>=0.98`、false accept `<=1%` 时 recall `>=0.25`，阈值只在 C07 选择；
- 去掉 metric geometry 或 transport 必须分别造成可重复的事件或后续 graph 退化；若完整方法与消融无差异，则不得声称几何/关系贡献；
- deterministic yaw-shift C07 轴审计必须与未旋转输出等变，并显著优于恒定前向基线，否则从论文贡献中删除 learned-axis 主张；
- 无 GT identity、隐藏 context、未来帧或测试世界泄漏。

任一核心门失败，停止 composer，不进入主方法图；V2R5继续作为论文基线和失败机制证据。通过后才冻结类型化 observation adapter，让 composer event、token geometry、descriptor 和 uncertainty 实际进入节点触发与关联，再做离线 graph 消融。
