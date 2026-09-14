# 双路口360固定轴向射线诊断

原12双路口声明：圆/椭圆/圆角矩形各四视点，每声明五历史位置，每位置正负XYZ六方向。共60位置/360射线，不是实际LiDAR角度分布，也不是360独立环境。

执行前合同：configs/v3/gate3/double_axis_software_contract_v1.json，SHA ff778aa7c549c778147647b4ed219ee4456ae0f46d6089376342e627767a97ef。脚本及五个关联文件共六文件哈希运行前后核对，非完整环境封存。原native数值距离对照线1e-5米，未根据结果调整。

命令：env PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zeng-workstation/.local/share/mtare_topo_comm/envs/cano_e1_topology_v1/bin/python tools/v3/check_double_axis_candidates_native.py

会话64144 exit0，5.494668秒。269候选与独立棱柱解析一致，最大距离误差2.6702880866480427e-6米。91待参考：61重复或多壳、27非交替或缺失、3末尾未闭合；不把这些拒绝计作正确返回。输出逐声明计数与误差，无超过既定对照线的候选。

下一同91射线调用既有慢参考，列可恢复、未知和矛盾；不另选方向或改变容差。此次只做软件声明诊断，零封存扫描载荷、零标签、零训练。不宣称全扫描修复或方法创新成立。
