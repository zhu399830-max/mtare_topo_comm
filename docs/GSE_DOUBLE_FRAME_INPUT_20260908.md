# 首个完整扫描诊断：输入接口已核对

本项只为完整扫描诊断准备输入，不是扫描正确性通过，也不是训练数据导出。

- 固定 `double_junction__circle__view0` 第0帧，位置 `[-4.4, 0, 0.04]` 米，yaw为0。
- 使用现有 `lidar_local_directions` 与 `world_directions`，保持原float32角度生成、随后转float64再旋转的路径；16行、720列，11520条不同方向。
- 独立解析距离采用与精确诊断相同的单位方向，避免射线参数与米制距离不一致。
- 只有1个声明、1帧，不算1个五帧训练观察；没有读取真实世界载荷、没有标签和训练。
- 原网格参数0.05米/64角段、原误差线1e-5米不变。未知不转成最大量程；数值比较与传感器有效距离分别记录。

代码：`src/mtare_topo/evaluation/double_frame_diagnostic.py`。
输入清单：`configs/v3/gate3/double_frame_input_v1.json`，包含声明、实际方向、单位方向和独立解析距离的SHA-256。

验证：CPU测试环境执行 `test_double_frame_diagnostic.py` 与 `test_exact_reference_diagnostic.py`，7 passed in 0.28s。随后使用 `cano_e1_topology_v1` 原生环境只计算声明/方向/解析清单，exit0；没有执行网格求交。

下一步：以此精确人口完成单帧executor、环境及工具冻结、运行规格和预检，之后只创建一个不可覆盖的正确性/吞吐诊断run。完整扫描结果仍未知；不扩大到全矩阵或启动训练。
