# 全射线区间验证 V2

## 后续原生验证

已有cano_e1_topology_v1环境缺pytest，测试入口未执行。未安装新依赖，新增不依赖pytest的tools/v3/check_interval_v2_native.py，用Open3D原生盒体与原生list_intersections执行四个案例，exit0。Open3D0.19.0/NumPy1.26.4：两个小间隙旧返回29.999998093米、V2返回1.999999881米；2厘米间隙二者近墙，真实重叠二者远墙。核心耗时0.016897秒、进程峰值314872KiB，不含导入计时，不能外推复杂网格吞吐。原pytest测试保留，未谎称在此环境运行。

命令：env PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zeng-workstation/.local/share/mtare_topo_comm/envs/cano_e1_topology_v1/bin/python tools/v3/check_interval_v2_native.py

此后下一步是原双路口固定异常射线的原生网格检查，不是全矩阵重渲染。

实现：src/mtare_topo/teacher/csg_interval_raycaster_v2.py。复用既有场景构造与网格占据区间验证，对所有射线执行，不只补缺失返回，不回退旧交点分组。它是慢速数值参考，尚未取得数据导出资格。

联合测试17 passed、1 skipped in 0.31s，exit0。覆盖三个正间隙、真实重叠、量程限制、来源非法、缺失交点、实际V2批处理及旧分组反例。批处理使用明确合成intersection IO，不是原生Open3D运行。首次三个失败为测试引用不存在的source_ids字段，修正成已有source_primitive_ids；算法和阈值未改。

测试命令：env PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 build/gse_supercluster_env_v1/bin/python -m pytest -q tests/v3/unit/test_csg_interval_raycaster_v2.py tests/v3/unit/test_mesh_interval_exit.py tests/v3/unit/test_mesh_winding_diagnostic.py tests/v3/unit/test_csg_group_gap_reproduction.py tests/v3/unit/test_gse_double_polygon_gap.py

Open3D测试因依赖不可导入跳过。下一核对已有sidecar，进行固定单射线原生验证和成本测量。未读新研究载荷、未训练、未改旧run。交点完整性、原12例异常与全矩阵资格仍未解决，不是模型或论文成绩。
