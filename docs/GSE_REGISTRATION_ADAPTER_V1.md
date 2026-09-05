# 真实局部点云配准薄适配 V1（仅 CPU 软件合成验证）

2026-09-05。当前 Phase 3 图软件并行支线。0 世界文件、0 数据集帧、0 模型权重、0 训练、0 正式图回放；未修改旧冻结图/位姿/端口源。这里的“真实配准”指实际调用算法估计变换，而不是输入 `registration_passed=True`，不指已经在真实地下数据验证。

## 做了什么

新增 `src/mtare_topo/topology/gse_registration.py` 的 `register_local_clouds`：输入当前 source 与历史 target 的本地点云（米）、部署提供的初始 `T_target_source` 和完整 `RegistrationConfig`。它调用已有 Open3D 0.19.0 的 point-to-plane ICP，从点云估计变换，不读取身份、地图或真值对应。

采用官方 `registration_icp`、`TransformationEstimationPointToPlane`、`ICPConvergenceCriteria`、法向估计及 `evaluate_registration`。Open3D 定义 source→target 变换；fitness 为对应数除以 source 点数。[官方 ICP API](https://www.open3d.org/docs/release/python_api/open3d.pipelines.registration.registration_icp.html)、[结果字段](https://www.open3d.org/docs/release/python_api/open3d.pipelines.registration.RegistrationResult.html)、[法向 API](https://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html)

没有自写 ICP，没有全局检索、RANSAC 重试或静默退回点到点。仅当调用方显式给出非零 `point_to_point_bootstrap_iterations` 时，先执行指定次数的 point-to-point 初始化，再运行 point-to-plane；请求值保存在结果中。

所有数据相关参数必须由调用方明确提供：对应距离、法向邻域及质量、点覆盖率、迭代/收敛条件、bootstrap 次数、双向最少对应与 overlap/RMSE 条件、信息秩相对特征值阈值和 condition 上限。代码没有科学默认值；测试中的数值只是合成 fixture，不能直接当开发校准完成。

## 证据与拒绝机制

1. 非有限/形状错误点云、非 SE(3) 初始变换、反射/尺度及不完整配置在配准前拒绝，不删 NaN 修成可运行输入。
2. 法向由点云估计；局部支持不足、近似共线或表面方差过高的点被显式排除，完整保留 source/target 原索引。原输入点数仍为本 adapter overlap 分母，同时另报 Open3D 过滤后分母的 fitness，避免少留点虚增覆盖。
3. 单次 ICP 后对估计变换作正向评估，并对其逆变换作反向评估；不是分别优化两个变换，也不是仅验证一边重叠。
4. 保存估计 SE(3)、两向对应数、两种 fitness、欧氏 RMSE、point-to-plane RMSE、信息谱/秩/condition、明确拒绝原因。
5. 无对应时，RMSE 记为 `None` 并拒绝。不能把 Open3D 空对应情况下的零值当作完美配准。
6. 两个方向必须满足显式 overlap、对应数、RMSE 和完整六自由度局部约束。平面秩 3、开放直走廊秩 5 即使残差为零也拒绝；局部退化不能自动合并节点。

信息诊断使用 point-to-plane Jacobian。官方 0.19.0 源码计算旋转项 `p × n` 与平移项 `n`；本模块将点居中并按 RMS 半径尺度归一，构成参数顺序为“平移/半径、旋转”的无量纲矩阵 `JᵀJ/N`，从而避免参考原点与米/毫米尺度改变条件数。[官方实现](https://raw.githubusercontent.com/isl-org/Open3D/v0.19.0/cpp/open3d/pipelines/registration/TransformationEstimation.cpp)

**这不是标定后的位姿协方差，也不是地点唯一性证书。** 满秩只说明当前对应下的局部约束，不排除重复房间、相似路口或错误局部极小值。`accepted` 只是这次局部几何检查通过，不能单独作为图节点合并许可。

## 测试实测与系统问题

环境：已有 `cano_e1_topology_v1`，Python 3.12.3、Open3D 0.19.0、NumPy 1.26.4。该环境没有 pytest，使用 Python 标准库 unittest，不安装额外包。

首次 22 项测试中，9 项纯函数通过、13 项配准测试在 import 处失败：Open3D 检测到 GPU 后先加载 CUDA pybind，再加载 CPU pybind 出现类型重复注册。安装的官方初始化源码明确避免双加载。这是环境选择错误，发生在任何 ICP 之前；不是算法或科学实验失败。

修正为启动独立 CPU 进程时设置 `CUDA_VISIBLE_DEVICES=''`，使用 Open3D 唯一选中的绑定，并在 adapter 检查 `__DEVICE_API__ == 'cpu'`。不在进程运行中修改设备环境，也不混载两个 pybind。启动时的“无可用 CUDA”ImportWarning 是刻意屏蔽 GPU 的结果，不是 GPU 硬件故障。

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/zeng-workstation/.local/share/mtare_topo_comm/envs/cano_e1_topology_v1/bin/python \
  -m unittest discover -s tests/v3/unit -p test_gse_registration.py -v
```

结果：**22 tests，1.066 s，全部通过、0 skip**。其中 13 项使用实际 CPU Open3D（后端错误测试显式 mock 错误，其他配准实际执行），9 项验证输入、配置与信息矩阵纯函数。

主要合成结果：546 点非对称三维平面组合，从 identity 初始化估计非零旋转和平移；已知合成变换只用于生成测试对及事后断言，不作为配准初值或对应。两向均 546 对应，fitness 1.0；变换元素最大误差 `1.11e-16`，正向 RMSE `2.01e-16 m`，信息秩 6、condition 99.09。另加固定噪声测试，以 `0.001` 的元素容差验证估计而非精确复制。

覆盖：无重叠、单平面、开放直廊、单向重叠高而反向低、邻域支持不足、共线退化、显式 condition 拒绝、显式 bootstrap、输入不变、单线程重复一致、后端失败、错误 SE(3)/NaN、尺度及原点归一不变。

主 Torch 环境没有 Open3D，因此组合回归结果为 **79 passed / 13 skipped / 0.19 s**（包括原 70 项 frame/port/graph 与 9 项纯函数）；13 项实际配准的证据只能引用上方专用 sidecar 的全通过结果，不能把 skip 当通过。

## 下一步接线与边界

已有 `gse_graph_frames` 负责部署位姿与节点坐标，`gse_port_updates` 消费唯一对应证据；新增模块现在提供实际扫描配准证据。仍需在新版本 graph adapter 中：

- 绑定 source/target 点云、时间戳、候选节点及配置来源，限制只用因果历史；
- 由部署估计给初值，在图上保留配准失败及局部退化，不能用真值位姿替代；
- 用估计变换校验端口对应与执行轨迹，并完成多候选唯一性裁决；
- 将通过的完整证据转为图更新，接入端口增量和 pending 生命周期；
- 执行首次访问、重访、歧义和真实穿越的完整合成回放，之后再按独立合同读取真实数据。

没有本地地图缓存、候选检索、跨进程传输、正式阈值校准、真实点云实验或新图闭环成功的声明。配准是复用的可靠性机制，不作为论文的新算法贡献。
