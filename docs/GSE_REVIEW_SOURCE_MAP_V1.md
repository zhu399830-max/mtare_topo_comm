# 48片段人工复核：现有数据入口与尚缺内容

2026-09-06。**本次只核查源码、既有配置和文档，没有打开新的真实Zarr数组、扫描、构造JSON或权重，没有执行选样/导出。** 下文的路径和字段是生产代码与冻结配置所声明的入口，不是对当前磁盘全部资产再次验真的结论。48片段、3024观察仍是目标；真实候选人口及标签尚未取得。

## 1. 应从哪一份资产读取

以下路径均相对仓库根，缩写只用于本文，不是可直接传给reader的路径。

```text
P1A = results/gate3_semantics/gate3_20260830_primitive_relation_p1a_lossless_storage_corrective_v1_seed0
P1B = results/gate3_semantics/gate3_20260830_primitive_relation_p1b_teacher_materialization_v1r_seed0
task = {parent_id}__{variant}
partition = fit（C01—C06）或 c07（C07）
```

| 内容 | 精确路径模板 | 来源依据 |
|---|---|---|
| LiDAR与逐帧身份 | `P1A/artifacts/dataset/{partition}/{task}.zarr` | P1a无损重压缩生产器 |
| 扫描来源码字典 | `P1A/artifacts/codebooks/{partition}/{task}.json` | 原P1a附件，无损版保留 |
| 完整构造参考 | `P1A/artifacts/constructions/{partition}/{task}.json` | 原P1a附件，无损版保留 |
| 五帧索引、几何监督、相对运动 | `P1B/artifacts/teacher/{partition}/{task}.zarr` | P1b V1R |
| 完整任务库存 | `P1A/artifacts/task_manifest.json`、`P1B/artifacts/task_manifest.json` | 任务身份、帧/序列量及shard哈希；需新受控元数据读取 |
| 原有向穿越清单 | `P1A/artifacts/traversal_manifest.jsonl` | 从原P1a复制保留 |
| 已采用的姿态修正记录 | `P1A/artifacts/frame_pose_corrections.json` | 不能另行重建未修正pose冒充当前输入 |
| 文件封存索引 | 两个run各自的 `artifacts/evidence_sha256.txt` | 新卡先绑定索引，再精确选择允许的文件 |

`configs/v3/gate3/primitive_relation_p1b_teacher_materialization_v1r.json` 的 `source_p1a_run` 明确指向上述无损版；不能误用首次P1a导出的旧存储目录。已有 `gse_supported_construction_teacher_v1r` 卡同样使用这两个根，并记录：

- P1A seal SHA：`79fd988ac8c205d74c93e4858b7b579571a06502e778f31634046b48791a0668`。
- P1B seal SHA：`f629b511e9a945bbe15249e3bc212be57a01b1aa0118d0d7d228824aa25f7d47`。

以上是**既有配置中的摘要值，本轮未重新读取索引或核验全部payload**。旧原180卡只允许原C01 `c1_mixed` 子集，不能授权新C01—C07三变体读取。新reader不能调用旧全量生产器或全树seal验证去顺带打开其他分区。

## 2. 三变体与完整父地图配置

生产枚举 `GeometryRealization` 精确取值为：

| 变体名 | 生产索引 | 含义 |
|---|---:|---|
| `ellipse` | 0 | 椭圆截面 |
| `rounded_rectangle` | 1 | 圆角矩形型超椭圆截面 |
| `c1_mixed` | 2 | 两端截面族变化的混合实现 |

三者复用父拓扑与基元身份，但不是三份独立地图，也不是新的A/B/C模型名。截面参数按父图/基元身份确定，不按模型分数生成。依据：[geometry_variant_contract.py](../src/mtare_topo/teacher/geometry_variant_contract.py)。

既有P1b数据卡的配置清单中，C01—C07共有70个不同父名：下面十个名称前缀分别加 `_C01` 至 `_C07`。这是**配置库存**，尚须同受控task manifest核对真实完整性。

```text
S01_flat_tree_small
S02_3d_tree_small
S03_flat_unicyclic_small
S04_3d_unicyclic_small
S05_flat_branch_medium
S06_3d_branch_medium
S07_flat_loop_rich
S08_3d_loop_rich
S09_flat_complex
S10_3d_complex
```

例如 `S01_flat_tree_small_C07__ellipse` 的sensor目录在 `dataset/c07/`，teacher目录在 `teacher/c07/`。C07的十父必须全部进入父库存哈希分区，然后前半校准、后半开发；不能只对恰好有候选的父图分半。旧数据卡把更多父图列为历史训练来源，其分类不能覆盖新PLAN的C01—C06拟合/C07父图拆分政策。C07也不称整个模型严格未见。

现有总任务manifest/封存索引还覆盖历史其他分区。新卡可以绑定共享元数据文件的哈希，但必须明确共享文件的元数据范围，解析后先筛允许的C01—C07父图/partition/task，再解析或resolve对应资产；不得打开C08及之后的shard、构造或模型产物。

## 3. 连续片段的身份如何接起来

### P1a：一行是一帧

schema：`primitive_relation_p1a_sensor_shard_v1`。`.zattrs` 有 `parent_id`、`partition`、`geometry_realization`、`sensor_shape=[16,720]`、`maximum_range_m=50.0`、`student_pose_input_forbidden=true` 和 `traversal_ids`。

| 字段 | 生产形状/用途 | 本轮后续读取等级 |
|---|---|---|
| `global_frame_index` | `[F]`，原全局帧身份 | 受控索引元数据 |
| `local_frame_index` | `[F]`，当前traversal内帧序号 | 受控索引元数据 |
| `traversal_index` | `[F] int32`，索引到attrs的唯一traversal列表 | 受控索引元数据 |
| `route_arc_m` | `[F] float64`，当前有向穿越的实际弧长 | 受控空间元数据，不是时间 |
| `range_m` | `[F,16,720] float32` | 真正扫描payload，精确片段卡后才读 |
| `valid_mask` | `[F,16,720] uint8` | 同上，必须核二值性 |
| `primitive_membership_code` | `[F,16,720] uint16` | Teacher参考，禁止学生/盲看输入 |
| `axis_xyz_m`、`sensor_xyz_m`、`tangent_world_xyz`、`yaw_deg` | `[F,3]`或`[F]` | 真值坐标/姿态，仅按新合同用于参考或已知位姿图诊断 |

实际frame边界从指定数组 `.zarray` 的shape核验，不假定固定每世界帧量；无损重压缩版的chunk/compressor必须读其当前 `.zarray`，不能硬编码首次导出的chunk。

### P1b：一行是五帧观察

schema：`primitive_relation_p1b_teacher_shard_v1`。attrs有相同parent/partition/variant，`maximum_slots=32`、`window_frames=5`、`student_identity_input_forbidden=true`。

| 字段 | 形状 | 精确语义 |
|---|---|---|
| `source_global_sequence_index` | `[Q] int64` | 同一原观察的身份，三变体应逐项相同 |
| `variant_global_sequence_index` | `[Q] int64` | 生产规则为source ID加`realization_index×188126`，三变体不应拿此字段要求相同 |
| `frame_row` | `[Q,5] int32` | 指向配对P1a shard的局部行；最后一行是当前决策帧 |
| attrs `traversal_ids` | 长度`Q`的列表 | **每个序列一项**，不同于P1a的唯一ID查表 |
| `relative_translation_current_sensor_m` | `[Q,5,3] float32` | 五帧到当前传感器坐标的相对平移 |
| `relative_yaw_current_sensor_deg` | `[Q,5] float32` | 五帧到当前传感器坐标的相对yaw；当前帧为0 |

生产规则见 [primitive_relation_sequences.py](../src/mtare_topo/data/primitive_relation_sequences.py)：按原traversal的 `global_sequence_offset` 排序，`source_id=offset+local_sequence`；五帧为 `global_frame_offset+local_sequence+[0,1,2,3,4]`，再换成shard局部行。原manifest还包含 `parent_id`、`traversal_id`、`edge_id`、`from_node_id`、`to_node_id`、`sequence_count`、`unique_frame_count`、`global_frame_offset`、`global_sequence_offset`。有序序列数非零时，生产合同要求该traversal原帧数=序列数+4。

因此21个连续且同traversal的决策，按该生产合同应引用25帧/变体，而不是105独立帧；最终仍须逐行核验实际引用和跨片段去重。21步不等于21秒，也不能只检查行号连续就宣称物理轨迹已连续。应从P1a实际 `route_arc_m[frame_row[:,-1]]` 核定步距；姿态修正使边界不宜直接假定整齐1米。

新采样核心的 `frame_traversal_ids` 可通过P1a `traversal_index→attrs.traversal_ids` 恢复，必须同P1b每序列traversal一致；`decision_frame_rows=frame_row[:,-1]`。三变体按source ID连接，不按最近空间位置、预测相似度或数组长度猜配对。

## 4. 几何与构造监督有哪些，缺哪些

P1b已有的32槽字段可原样作为受控loss/参考来源：

```text
primitive_index [Q,32] int32；primitive_mask [Q,32] uint8
axis_control_current_sensor_m [Q,32,3,3] float32
endpoint_half_axes_m [Q,32,2,2] float32
endpoint_shape_exponent [Q,32,2] float32
support_ray_count [Q,32] int32
temporal_visibility [Q,5,32] uint8
frame_primitive_index [Q,5,32] int32
temporal_destination [Q,5,32] int8
endpoint_neighbor [Q,32,2,3] int8
disconnected_overlap_packed [Q,32,4] uint8
```

它们不是新的独立结构实例/完整开口/存在背景标签。`endpoint_neighbor`是构造端点关系，不能把裁剪轴端直接当可见开口；`disconnected_overlap_packed`是小端bit序打包的32×32角投影重叠，不等于物理碰撞，也不是已经审查过的误合并场景分类。

构造JSON schema为 `primitive_relation_realized_construction_v1`，内含 `base_construction`（`primitive_construction_graph_v1`）和 `realized_primitives`：

- `base_construction.primitives[]`：`primitive_id`、`source_edge_id`、`source_tunnel_id`、轴线、半径、两个endpoint；endpoint有 `node_id`、`endpoint_index`、位置及composition anchor。
- `base_construction.composition_operations[]`：`node_id`、`degree`、`anchor_xyz_m`、`member_endpoints`；degree按实际edge incidence计，不应改用历史declared degree。
- `realized_primitives[]`：与base严格同顺序的 `primitive_id`、`centerline_xyz_m`、`endpoint_half_axes_m`、`endpoint_shape_exponent`。
- codebook schema为 `primitive_membership_codebook_v1`，包含 `primitive_ids` 和 `source_sets`；code0为空集合，非零码保留所有来源，不能擅自挑一个来源。

依据：[构造生产器](../src/mtare_topo/teacher/primitive_construction_supervisor.py)、[构造加载器](../src/mtare_topo/data/primitive_relation_materialization.py)、[关系存储](../src/mtare_topo/data/primitive_relation_storage.py)。后续仍要严格验证整数类型，不能沿用旧宽松`int(value)`读取把浮点身份截断。

## 5. 四个分层不是现成字段

| 目标分层 | 可以复用的提名依据 | 仍需明确、实现或人工复核的部分 |
|---|---|---|
| junction | 构造composition degree≥3、node ID及anchor | 从所有相关traversal定位21步片段、是否真正可辨；同node跨traversal去重 |
| terminal | degree=1的构造端与对应traversal | 是否有可见终端/cap证据；隐藏终点只能UNKNOWN，不能因构造degree强贴可见终点 |
| corridor | 普通edge内部或degree=2构造位置可提名 | 内部片段的稳定独立ID、与事件区的边界和窗口定位规则尚未实现；普通走廊不是自动完整背景 |
| alias | 非连接角投影重叠、平行/分层/重复构造可提供线索 | 无现成`alias`标签；需不依赖预测分数的确定性提名与规范化独立身份，且不能把角重叠当物理相交 |

**当前阻塞不是找不到扫描，而是尚无从这些源字段生成完整四层候选的已冻结操作规则及真实核对结果。** 例如“距离anchor多近”“普通edge取哪段”“alias两结构怎样合为一个独立case”不能在看过效果后临时改，也不能用同一结构同时凑两个分层配额。这不否定采样核心；它明确了caller还缺的元数据派生步骤。

盲看导出必须只显示当前与过去range/valid恢复的点云及允许的相对运动；构造、degree、primitive/source IDs、GT mask和未来帧先隔离。原始点坐标按已有LiDAR方向函数从range恢复，无需新增扫描或读取世界mesh。人工先保存盲看提交，再揭示构造；揭示后不能把隐藏结构回填为可见。分层提名只能用于抽样，不能直接成为模型训练答案。

## 6. 最小下一入口及停止条件

1. 独立元数据卡锁定70个允许父名、三变体、两个run/封存索引和精确允许字段；先共享库存元数据，再指定shard的 `.zattrs`/`.zarray` 及索引chunk。只生成真实库存/候选账本，不读扫描。
2. 用现有构造与上述规则形成四层提名，需要构造字段时由同一明确卡或后续独立卡授权；未冻结提名语义前不声称自动得到48。
3. 将源核查得到的严格 `ParentPopulation`、`CandidateSegment` 交给 [确定性采样核心](../src/mtare_topo/data/gse_review_sampling_v1.py)。不足如实报告，不改配额、不重复结构、不加相邻片段凑数。
4. 真正选中后冻结准确行、去重帧、空间间距及三变体对应，另以精确扫描/复核范围导出盲看材料。人工作业未完成，不开A/B/C正式训练。

必须保留身份/帧绑定检查。旧 `PrimitiveRelationTrainingShard` 初始化会全读 `frame_row[:]`；旧scoped readers又硬编码原C01/`fit`/`c1_mixed`范围，均不能未经新边界直接当48片段reader。新受控reader应先按明确task/字段筛seal条目再resolve，禁止扫描整个results或开启所有Zarr字段。若三变体缺失、traversal跨段、序列/帧身份漂移、同结构重复分层或有效人口不足，停止受影响选择并输出缺口。

本轮产物只有本文：**新真实数组读取0、候选账本0、人工标签0、训练0。**
