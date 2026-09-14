# 同一网格的起点—候选—参考组合诊断

BoundRayDiagnostic持有网格不可变副本，原生场景、缓存检查和慢区间参考都绑定这份几何。每次比较保留起点状态、快速状态、回退原因、参考距离和来源一致性，不提供生产导出入口。

实际命令：env PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zeng-workstation/.local/share/mtare_topo_comm/envs/cano_e1_topology_v1/bin/python tools/v3/check_bound_ray_diagnostic_native.py

exit0。8原生盒体中，gap/overlap/separated/coincident_sources四候选与参考的距离和来源一致；contact/internal_shell/duplicate_shell/surface_origin四例请求参考。最后一例参考仍未知，未填距离。逐例外部网格修改后再次调用，结果保持相同。原失败诊断脚本未改写。

仍然只是软件组合证据，不能证明任意网格/射线的原生交点完整性。下一转到原12双路口声明的60历史位置，以六个固定轴向共360射线检查候选覆盖率和独立解析一致性；先冻结软件范围，零封存扫描读取、零标签导出、零训练。不继续堆叠盒体测试。
