# 新增2676观察输入已导出

独立运行：`results/gate3_semantics/gate3_20260907_gse_supplement_input_v1_seed20260906`。

精确卡/spec和源码环境冻结后，53相关回归及preflight通过。create_run一次、执行一次，终态COMPLETED。固定210任务/2676五帧观察、13374去重变体源帧；fit2310、calibration189、development177。原3360输入和特征完全未重导或重算。

实测21.512秒，峰值主机277,495,808字节，输出564,219,036字节，零GPU/训练更新/标签。累计原始块解码上界约3.55GB不等于峰值内存。输出NPZ仅含range、valid、relative translation/yaw、frame rows、source sequence IDs；没有绝对pose、构造或teacher目标进入数值输入。

所有223项seal重新SHA核验通过，seal SHA-256：`b49727c02eb7265e40411e784c86c079fea4430b7464e51f68bf92c9d73c92a8`。独立重新打开210个NPZ，检查六字段、人口形状和全部五帧/序列与清单一致；210行原始JSONL可解析。读取器运行时检查原数组数值/来源/相对运动合同，结果另保存input_manifest、来源哈希、源码快照、环境/命令、summary、RUN_STATE。

合并人口6036现已具备输入来源：原3360包加新2676包。input_manifest保留全部combined_requests及来源/采样角色，原清单不改写。2676不是2676独立实体；五帧历史重叠和三变体仍按父地图统计。

下一主线回到实际观测标签。复用原面片、射线证据、局部开口和端面支持程序，绑定新增清单所需来源；核实结构锚点、开口与未知区域，不要求残缺扫描完整逆问题唯一，也不把926构造候选直接当标签。停止重复输入导出/位置采样/旧公共特征提取。物理可达标签缺机器人合同的备案保持暂停。本次为输入准备成功，不是模型或建图科学PASS。
