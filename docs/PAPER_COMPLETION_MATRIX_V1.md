# GSE-Graph 论文完整交付证据矩阵 V1

状态：`ACTIVE_NOT_COMPLETE`  
更新时间：2026-08-27  
唯一目标：完成并投稿“从因果 LiDAR 学习几何结构语义，并由语义直接建立在线拓扑图”的 GSE-Graph 论文。

## 1. 方法主线

```text
五帧因果 LiDAR
→ 中心轴、出口布局、宽度、净空、坡度、曲率、结构事件和不确定性
→ 学习式节点生成与关联
→ 真实穿越验证的稀疏 topometric graph
→ 全局探索目标
→ 原 M-TARE 局部规划、避障和控制
```

模型输出必须直接影响节点生成、节点/出口关联和拒绝合并；图的边只能由真实穿越建立。旧出口模型和规则图仅作为对照，不能再被表述为主方法。

## 2. 论文证据状态

| 环节 | 必须回答的问题 | 当前状态 | 已有证据 / 下一项工作 |
|---|---|---|---|
| 创新边界 | 与 Cano、出口识别、place recognition、frontier policy 和 M-TARE 有何本质差异 | 已建立 | `docs/GSE_GRAPH_CONTRIBUTION_MATRIX_V1.md`；投稿前更新相关工作检索 |
| 数据与 Teacher | 是否学习了真实结构量，且 train/validation/test 拓扑隔离 | 持久双向change-point proof与corrected manifest均PASS | 80个C01--C08 worlds、188,126条观测；旧3,035个transition identities收敛为76个稳定change-points，应用事件优先级后保留1,031条标签，fit/selection identity=`59/17`。corrected manifest seal=`3f840e...7f49`；C09/C10/M-TARE读取0 |
| 几何语义模型 | 结构事件、连续几何、中心轴、出口 token 和描述子能否共同学习 | 连续delta学习PASS组件；离散change-point仍FAIL；有界fallback运行中 | 冻结表征多任务把selection normalized delta MAE从`0.587167`降至`0.355174`（改善`39.51%`），但event macro-F1=`0.683669`、change-point=`0/17`。当前只微调最后残差块+全新头，其他主干冻结；旧失败头不复用 |
| 感知资格 | 是否显著优于旧出口模型和非学习几何估计 | 连续几何/关联通过；结构事件尚未取得资格 | 风险校准感知、坡度与exit-token关联已有PASS组件；新delta回归也明显改善，但可靠节点触发未通过。最后编码块正式三seed实验通过前禁止把组件结果写成完整感知PASS |
| 离线拓扑 | 学习语义是否真正改善节点、边、连通分量和环结构 | C09 V3/V4正式FAIL | 243组完整GSE回放没有association-safe且高召回配置；V4完整seal=`cdb987...e44f8d`。junction/terminal覆盖约93%/97%，turn/transition约4%/1%，直接限制图召回；禁止用图参数或planner掩盖 |
| 严格未见拓扑 | 冻结方法在 C10 是否保持性能 | 禁止读取，待全部验证参数冻结 | 一次性 strict-test，不用于训练、阈值或 checkpoint 选择 |
| 单机器人闭环 | GSE 图是否改善覆盖效率、冗余和失败率 | 待离线图通过 | 与原 M-TARE 配对比较；保留原 local planner/controller |
| 多机器人闭环 | 2/3/4 机器人共享稀疏结构图是否降低重复探索和通信 | 待单机器人通过 | 报告 coverage-time、路径冗余、目标冲突、通信字节、延迟和失败 |
| 消融 | 收益来自几何、时间、学习关联、不确定性还是 edge 属性 | 接口规划完成，实验未运行 | 五项固定消融：无显式几何、单帧、距离/角度关联、无拒绝、无 edge 几何 |
| 论文与复现 | 每个结论能否从图表追溯到密封结果 | 正文骨架与9组图包已完成 | transition失败分析manifest=`800008...9b055`；持久双向Teacher proof论文图manifest=`95d29a69...2b79`，含优先级整合后的1,031标签。最终拓扑、C10和闭环图待合格方法结果 |

## 3. 已保留的旧工作

- 原始 M-TARE：论文主基线。
- 旧出口模型：Cano-like 感知基线。
- 规则因果图：无学习几何语义建图消融。
- GT-TNG：诊断上界，不称为理论上界。
- world、TNG、spline、mesh、LiDAR、teacher、Gazebo 和轨迹验证工具：继续复用。
- 已完成旧闭环结果：只作为基线、失败分析或消融证据，不再扩展旧方法。

## 4. 固定执行顺序

1. ~~明确选择持久双向change-point（A）或edge属性路线（B）。~~ 已选A。
2. ~~对新transition语义做C01--C08只读容量、双向一致性、稀疏性和因果检测proof。~~ 已PASS。
3. ~~重建受影响Teacher。~~ 已PASS；完成最后编码块有界多任务训练并据固定门槛冻结或停止表示路线。
4. 重跑C09离线拓扑并冻结唯一图参数；失败则停止该主张。
5. 只读执行 C10 严格未见拓扑测试。
6. 执行单机器人和 2/3/4 机器人 M-TARE 闭环比较及固定消融。
7. 自动生成图表，完成正文、补充材料、复现包和投稿 PDF。

若学习几何不能改善感知与离线拓扑，停止受影响的闭环路线，不通过规划器调参掩盖失败。

## 5. 论文图片保留合同

正式图片统一保存在 `docs/figures/gse_graph/`。每张图必须同时保留：

- 可直接放论文的 PNG；
- PDF/SVG 矢量版本；
- CSV/JSON/NPZ 源数据；
- 确定性生成脚本；
- 来源 run、样本选择规则和 SHA-256 清单。

当前已保留方法、数据、Teacher、exit token、训练、感知、坡度风险校准、transition失败分析、持久双向Teacher proof和geometry-delta多任务失败分析共10组图包。新delta图包含PNG/PDF/SVG、CSV/JSON与provenance，6个清单内文件逐哈希一致，manifest SHA=`e8e24f347774a5d6d2f42c6daa346aa2097f9a0b441fda88d42f1f78022f2948`。最终离线拓扑、单/多机器人coverage-time和典型图将在对应正式结果封存后生成，禁止手填数值或不可追溯截图。

## 6. 完成定义

只有以下内容全部形成可复现证据，目标才完成：GSE-Graph 方法和冻结模型、严格未见拓扑结果、单/多机器人闭环、必要基线和消融、全部论文图表、正文、补充材料、复现包与投稿 PDF。
