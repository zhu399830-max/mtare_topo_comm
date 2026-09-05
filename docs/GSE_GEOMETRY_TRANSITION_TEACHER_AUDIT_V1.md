# GSE-Graph 几何突变 Teacher 审计 V1

日期：2026-08-26  
状态：`TEACHER_METRIC_DEFECT_REQUIRES_RESEARCH_DECISION`

## 触发证据

Directional Structural Event Head 的唯一正式训练已经完整执行并封存为科学 FAIL：

- 三个 seed、36,000 次新头 optimizer step，backbone optimizer step 为 0；
- C07--C08 ensemble event macro-F1=`0.695935`；
- 节点 precision/false/recall=`0.990073/0.009927/0.487987`；
- junction/terminal/turn/geometry-transition identity coverage=
  `0.972603/0.984375/0.189474/0.002688`；
- 运行 `3368.40 s`，峰值 RSS=`2,765,292 KiB`；
- C09/C10/M-TARE正式训练读取为0，26/26完整seal SHA-256=
  `ea7507d53802126ea3da6141604a2f7812727718fbda4fea281db833151c7aab`。

这证明保留五帧方位布局能把转弯覆盖从原来的约5%提高到约19%，并保持开放集
安全，但仍不能把几何突变变成可靠节点。

## Teacher 结构审计

对 C07--C08 的 sealed Teacher identity manifest 做只读统计：

- 几何突变 identity=`744`，平均每世界`37.2`个，占选择域全部`1,113`个结构
  identity 的`66.85%`；
- `674/744=90.59%`的中心在物理 edge 端点`10 m`以内，`735/744=98.79%`
  在`12 m`以内，全部在`15 m`以内；
- 标注区间中位长度仅`2 m`；`411/744=55.24%`不超过`2 m`，`659/744=88.58%`
  不超过`5 m`；
- `143/744=19.22%`只在一个 traversal 方向出现，违反同一物理结构应具有双向
  一致证据的预期；
- 620条物理edge出现突变，117条edge有多个独立突变identity，单edge最多3个。

当前 Teacher 是前后各5 m宽高的阈值比较，但这些统计表明它主要捕获edge端点附近
短暂的native-mesh截面波动或generator接缝，而不是稳定、方向一致、值得生成独立
拓扑节点的几何变化。把邻近的corridor帧全部当作open-set负例，还会惩罚合法的延迟
检测，与在线因果change-point的语义不一致。

## 影响

问题分类为`teacher + metric`，同时暴露原事件头的空间表示缺陷。它使当前
geometry-transition节点的训练门、identity coverage和C09图召回结论不能用于判断
“学习几何是否能改善建图”。路口、尽头、转弯、连续几何、exit token、place
association、traversal-only edge和已封存基线证据不受影响。

不得继续做以下操作：用当前transition标签重训更大模型、降低1%错误接受门、把
邻近corridor静默改成正例、继续扫图参数或进入C09/C10/闭环。

## 可行方案

### A（推荐）：持久、双向一致的因果几何 change-point

- 在每条physical edge的canonical arc上从mesh宽高profile构造持久变化段；
- 要求正反traversal得到同一物理change-point；单向、短暂和非持久波动拒绝；
- 端点附近变化优先归并到已有TNG degree-2几何节点，避免generator seam重复造点；
- 在线只用已观测历史执行change-point检测，允许检测延迟，并用已走过轨迹回投
  物理边界位置；
- 新Teacher先做C01--C08只读容量/一致性proof，再决定是否重导受影响标签和训练；
- 保留geometry-transition作为论文结构语义节点，最符合GSE-Graph主张。

成本：中等。LiDAR、world、mesh和绝大多数Teacher字段可复用；需要重建transition
identity/event标签、更新评价合同，并训练新的event head。预计proof为CPU小时级，
正式标签重建/训练为GPU数小时级。

### B：取消geometry-transition节点，仅作为edge几何属性

宽度、净空、坡度、曲率沿真实穿越累积在edge上，只保留junction、terminal和turn
结构节点。实现成本较低、稳定性更高，但削弱“几何变化直接生成节点”的论文贡献，
需要重写贡献与消融。

### C：保持当前Teacher继续扩大模型

不推荐。压缩输出corrective与方位时序corrective已用两种信息接口正式失败，而且
Teacher存在短暂、端点集中和单向不一致证据；继续扩大模型不能修复监督语义。

## 推荐决策

选择A。它不推倒现有工作，能保留论文核心创新，同时把“几何突变”从帧级类别改成
真正适合在线拓扑的因果结构事件。只有新的只读Teacher proof证明数量、持续性、
双向一致性和图稀疏性合理后，才允许重导标签和训练。

