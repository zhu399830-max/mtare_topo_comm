# 请求完成后释放结果：软件验证

本次只修执行器对象生命周期，不改标签算法、数据、阈值、射线批量或内存上限。旧运行及原worker保持不变。

新入口为tools/v3/request_scoped_teacher_worker_v1.py及对应client。每次计算放在独立函数作用域中，只返回序列化字节和成功标记；发送后释放响应。教师仍从冻结V8归档导入，运行时核对项目模块来源。

## 已完成

联合测试16项，1.08秒，退出码0。包含：

- 成功响应与旧格式的gzip字节一致。
- 连续三次请求，下一计算开始前上一输入和raw弱引用已失效，无需显式GC。
- diagnose及produce分别失败时对象释放，错误响应保留原因。
- 失败响应后不消费下一请求。
- 原3GiB地址空间限制下，同一实际子进程连续两次合成echo及正常退出。
- 原传输和V8人口摘要回归。

命令：

```sh
env PYTHONPATH=src MALLOC_ARENA_MAX=2 /home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python -m pytest -q tests/v3/unit/test_request_scoped_teacher_worker_v1.py tests/v3/unit/test_teacher_bundle_wire_v1.py tests/v3/unit/test_v8_multiview_population_v1.py
```

真实地图观察、真实教师计算、训练更新均为0。合成测试证明引用释放及协议行为，不证明真实场景峰值内存已满足上限，也不证明标签质量。

## 下一项限定验证

准备独立规格复用已绑定的source192653/192654两位置，唯一变更为请求作用域释放；不混入分批、GC或增大资源上限。前例必须完整raw/targets与封存参考一致，后例是否越过原故障独立记录。规格、数据卡与源码冻结并预检前不执行。旧两项失败保留，不能把这次软件通过当作重跑141或训练许可。
