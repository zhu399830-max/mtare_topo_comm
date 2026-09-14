# 原生候选几何图入口：运行合同草案

## 正式冻结参数记录（2026-09-09）

以原生入口与上游头文件/构造函数的源码哈希共同绑定，不按结果调参。

- TSDF：Simple，voxel=0.25m，block=16，truncation=1m，max_weight=10000，carving=true，min/max_ray=0.1/50m，const_weight=false，allow_clear=true，weight_dropoff=true，sparsity_compensation=false/factor1，threads=1，order=mixed。
- TSDF未启用算法的保留配置：anti_grazing=false，start_voxel_subsampling_factor=2，max_consecutive_ray_collisions=2，clear_checks_every_n_frames=1，max_integration_time_s为float最大值；不是本轮Simple积分器的额外过滤。
- ESDF：full_euclidean=true，max/min/default_distance=2/0.2/2m，min_diff=0.001m，min_weight=1e-6，num_buckets=20，multi_queue=false，occupied_crust=false。clear/occupied_sphere_radius=1.5/5m保留原默认，但不调用机器人清空球接口。
- Skeleton：min_separation_angle=0.785rad，generate_by_layer_neighbors=false，num_neighbors_for_edge=18，check_edges_on_construction=false，vertex_pruning_radius=0.35m，min_gvd_distance=0.4m，cleanup=kSimplify。
- 边诊断：每0.125m以内采样，所在体素查值，未知与低于0.4m分别计数；不删除或替换边，不是连续安全证明。
- 人口执行：父进程8GiB地址空间、单子进程4GiB地址空间，CPU120秒/墙钟150秒/单文件16MiB；全批43200秒、结果12GiB、零GPU。单例失败即停止封存，不重试。

本次仍以56父地图为统计单位，不把307个观察称为307个独立地点。原容器中2184个观察会被原NPZ校验器解码验证，只有清单307行进入算法；不增加1877个附带行的输出或训练用途。时间/空间间隔及五帧来源沿用清单逐行frame_rows和原轨迹记录，不重新采样，也不声称固定1米间距。教师仅保留历史元数据身份核对，不读取构造几何或标签进入算法。

入口已编译：tools/v3/native_voxblox/observation_graph.cpp。标准输入GSE_RANGE_V1，恰好5帧，每帧相对位置xyz、相对yaw度和16×720个range_m/valid对。当前帧变换必须为零，回波0–50米，valid仅0/1；拒绝多余字段或截断包。原归一化张量由native_voxblox_packet编码，来源身份由外层清单管理，不进入原生前向。空观测允许返回空图。

预期数据只用既定307观察(250/27/30)，56父地图、156容器、1529去重变体帧；完整容器2184观察中其余1877不参与输出。沿用development_sensor_scope固定元数据SHA17c94619c34a60861bdc5507ae010749eeb5411234db0cc8a8a1d2235f03a334。此文不是正式数据卡，真实输入数组尚未读取/验证。

TSDF/ESDF/骨架配置继承合成入口，不按开发成绩选择：0.25米体素，16体素块，TSDF截断1米、量程50米、单线程；ESDF full_euclidean=true、无occupied crust、不调用清空球，其余固定原默认。需在正式规格完整记录默认字段和上游版本，不能只记录覆盖项。低于原min_ray_length=0.1米的有效回波单独计数，不能隐藏算法过滤行为。

每个观察独立构图，保留原生所有节点/边与边采样诊断，不跨独立片段连边。它是候选几何图，既不是持久语义节点图，也不是已穿越/安全边图；不得直接搬到学习路口表中宣称同任务优劣。正式执行前冻结资源上限、超限即停止、每观察原始输出、输入哈希、参数、耗时和错误原因。不得读取C08–C10或任何新checkpoint。

已执行软件检查check_native_voxblox_packet.py：空观测0节点；错误魔数、帧数、截断包、附加字段共4类拒绝；五帧各11520个合成定距回波总57600，原生2节点1边，TSDF支持检查通过。定距回波并非研究地图，不作结构正确结论。
