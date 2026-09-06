# C01—C07身份索引库存：一次受控只读审计

2026-09-06。状态：数据卡与运行规格已准备，**尚未创建或执行run**。最终命令/源码哈希须由主代理在全部实现交接后统一冻结；当前spec的`source_sha256`为空并显式标记草稿，不能据此直接执行。

## 本次只回答库存问题

现有70父地图、三种几何变体的身份索引是否对应一致？哪些已记录的有向穿越包含至少21个连续决策？真实帧、序列及可枚举区间各有多少？

本次不决定哪48个结构片段进入研究，不提名junction/terminal/corridor/alias，不生成图像、不做人工标注、不生成teacher、不运行模型或训练。下游盲看包必须等待真实库存、提名规则和新的精确导出范围。

## 数据边界

卡：`configs/v3/gate3/data_cards/gse_review_source_inventory_v1.json`。规格：`configs/v3/gate3/gse_review_source_inventory_v1.json`。完整70父名、210个task逐项列入卡，不能按当前有样本的父图临时生成分区。

- C01—C06：60个父地图，P1a/P1b目录`fit`。
- C07：10个父地图，目录`c07`；仍属开发范围，不称整模型严格未见。
- 三变体：`ellipse`、`rounded_rectangle`、`c1_mixed`。
- 两个来源为既有P1a无损存储修正版与P1b V1R，精确根和两个封存摘要均绑定在卡的`scope.source_roots/source_seals`。
- 禁止读取C08—C10资产、完整世界、构造、旧模型、预测或checkpoint；原180卡不提供本次权限。

来源定位与字段依据见[数据入口说明](GSE_REVIEW_SOURCE_MAP_V1.md)。这次不读取全量task/traversal manifest；从明确允许的shard获得身份人口，不借库存动作打开其他分区或几何附件。

## 只读七类数组

| 来源 | 允许字段 |
|---|---|
| P1a | `global_frame_index`、`local_frame_index`、`traversal_index`、`route_arc_m` |
| P1b | `source_global_sequence_index`、`variant_global_sequence_index`、`frame_row` |

另外只允许对应shard的`.zgroup`、`.zattrs`及上述数组的`.zarray`；不默认允许数组`.zattrs`、`.zmetadata`、任意目录遍历或其他array。允许的身份数组chunk必须按seal和精确字段筛选后再resolve。读取字节、字段和摘要逐项入账。

`range_m`、`valid_mask`、来源码、绝对pose、轴线、mask、几何监督和构造JSON均不读取。这里的`route_arc_m`是实际有向行程索引，不是传感器绝对位姿，也没有秒数意义。

## 已知预期与未知实际分开

卡中的预期核对值：70父、210task、163732个逻辑五帧序列、491196个变体观察。后两数是沿用原配置的人口检查，**不是本次已测结果**；不一致即停止并说明，不能改期待数凑通过。

原始去重帧数、活动穿越数、可枚举21决策区间数均为null，待实际索引计算。结构事件数仍为null且本次不会推导；时间`duration_s=null`，禁止填假时长。

三变体source sequence身份必须一致；`variant_global_sequence_index=source_id+variant_index×188126`。**188126只是原变体偏移步长/历史population常量，不是source ID上界**；原身份可稀疏，不能据此拒绝大于该数的合法source ID。

在每个父shard内部核对P1a原全局帧连续、local frame和traversal查表、实际route arc、五帧同一traversal且最后帧为当前决策、P1b完整滚动窗口及三变体对应。不能把多个父图拼起来要求global frame无间隙。

穿越统计只能叫 **`with_frames`/`active` 已记录穿越**。原生成时零序列且零帧的穿越可能不进入P1a attrs；本次不读原`traversal_manifest`，不能宣称核验了完整构造的全部directed traversals。索引的完整性仅相对于当前shard已记录帧人口。

少于21决策的活动穿越正常记录缺额，不当作软件失败、不重复补足。21决策按原五帧滑窗通常引用25帧，仍逐行计算真实引用与去重；连续索引不等于已证明物理执行连续，输出保留`continuity_requires_source_audit=true`。

## 为什么需要窄审计卡

通用`v3_data_card_v1`强制每条轨迹有正数时长/距离/覆盖，原帧数与有效结构事件数为正整数；本次要测的正是未知库存，而且没有采集时钟与结构标签。不能填1或虚构轨迹通过检查。

因此仅新增`v3_identity_inventory_card_v1`，只接`audit`、Gate3、固定70父/210task/两seal/七字段及null未知值。通用训练卡规则不改，也不借这个审计schema获得训练、teacher、潜在标签、图片或标注权限。批准来源为用户完整双通路计划和既有持续范围授权，不虚构新的对话。

## 成本、产物与执行顺序

CPU单次总上限600秒、主存4GiB、输出0.25GB、GPU使用0。无扫描解码、骨干/模型、优化步骤、结构选择与标注。

保存每父全部活动traversal区间及精确source/frame/arc索引、短穿越账本、实测汇总、字段/文件读取哈希、原始日志、环境、冻结配置与命令、RUN_STATE和SHA-256 seal。没有48候选名，也没有新标签。

所有作者交接停止后：主代理补全最终工具/命令哈希并移除草稿状态，执行preflight，创建唯一不可覆盖run，再执行一次reader/runner。任一源漂移、越界访问、身份错误、预期人口不符或资源超限均失败并封存；不覆盖、不在原run重试。

准备完成不等于真实库存完成，更不等于几何结构方法或论文通过。本轮结束时实际读取和结果由正式run另行记录，不能用此说明预填成功。
