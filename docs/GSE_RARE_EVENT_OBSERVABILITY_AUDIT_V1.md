# GSE-Graph 稀有结构事件可观测性审计 V1

日期：2026-08-26  
状态：`READ_ONLY_DIAGNOSTIC_COMPLETE`

## 问题

确认 C07--C08 中转弯和几何截面突变不能通过高精度节点生成门，究竟来自
Teacher 与因果输入错位，还是当前模型丢失了必要的空间结构信息。

本审计只读取既有 C01--C08 数据、冻结输出和源代码；没有修改 Teacher、数据、
checkpoint 或阈值，没有读取 C09、C10 或 M-TARE。

## 证据

1. 几何突变 Teacher 在当前位置前后各取 `5 m` 的宽度/高度中位数，变化超过
   `1 m` 时标注突变。该 Teacher 使用完整 mesh 产生监督标签，但未来扫描不是模型
   输入。
2. 学生输入是当前帧加四个过去帧的 50 m 全向 LiDAR。C07--C08 的当前帧前向
   `±5°`、垂直 `±5°` 扇区中：
   - 转弯帧 `520/521 = 99.81%` 至少有一束光线达到 `5 m`；
   - 几何突变帧 `4310/4333 = 99.47%` 至少有一束光线达到 `5 m`；
   - 两类逐世界最低比例分别为 `98.72%` 和 `96.83%`。
3. 冻结三 seed 平均事件分类器在 C07--C08 上已经得到转弯召回 `90.21%`、
   几何突变召回 `82.88%`。因此这两类不是从输入中完全不可观测。
4. 问题发生在节点所需的高精度置信度：全局 1% 错误接受合同选择阈值
   `0.988674` 后，95 个转弯 identity 只覆盖 5 个，744 个突变 identity 只覆盖 1 个。
   转弯帧结构分数中位数为 `0.898658`，突变为 `0.701140`；路口和尽头分别为
   `0.995860` 和 `0.999878`。这说明统一节点分数没有校准不同结构事件的证据强度。
5. `GeometrySemanticEventNet` 的 `event_head` 只读取经过 elevation/azimuth 全局均值
   和 GRU 得到的 `context`。保留 azimuth 布局的 `directional` 特征只进入中心轴与
   exit-token 分支，没有进入事件分类。转弯和截面变化所需的方向布局在事件头之前
   被空间平均。
6. 冻结 146D 输出上的 292-128-64-5 residual corrective 已正式失败：event
   macro-F1=`0.706287`，转弯 identity coverage=`11/95`，突变=`0/744`。因此只在
   已压缩输出后做类别重加权不能恢复丢失的空间证据。

## 结论

当前没有证据把失败归类为 Teacher 泄漏或数据不可观测。主阻塞属于
`model / spatial-temporal event representation`：事件头过早丢弃方位结构，同时把
“是否值得生成节点”和“属于哪种结构事件”压在同一个五分类 softmax 中。

因此不重建 C01--C08 Teacher，不移动现有标签，不降低 1% 错误接受门槛，也不继续
调整图参数。

## 下一方法

新增一个冻结主干后的 **Directional Structural Event Head**：

- 从现有 encoder 取得五帧逐方位特征，显式构造当前、过去汇总及其变化；
- 用圆周等变卷积与集合池化保留方向布局；
- 独立输出 binary structural evidence 和 conditional event class；
- binary head 使用 corridor 困难负样本与结构 identity 平衡；
- conditional head 只在结构样本上区分 junction、terminal、turn、transition；
- backbone、连续几何头、exit tokens、place descriptor 和 pair verifier全部冻结；
- C01--C06 拟合，C07--C08 选择；通过前不再读取 C09。

通过条件保持不变：aggregate precision `>=0.98`、false accept `<=0.01`、recall
`>=0.40`，转弯和突变 correct-class identity coverage 均 `>=0.40`，路口和尽头均
`>=0.90`，且十个 topology family 全部非空并满足原 family 门槛。

