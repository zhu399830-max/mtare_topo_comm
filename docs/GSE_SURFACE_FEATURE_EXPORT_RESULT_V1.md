# 公共五帧特征已完成真实提取

2026-09-07。一次不可覆盖运行已完成，不是准备状态。Phase 3 的面片关系建图研究问题保持不变，本次没有训练新模型。

运行：`results/gate3_semantics/gate3_20260907_gse_surface_features_v1_seed0`。

- C01–C07的70父地图、210输入文件、3360个五帧观察全部处理；源帧16800，拟合/校准/开发观察2880/240/240。
- 使用原observable seed0、第2轮冻结权重，只调用编码器 `_memory`。不运行旧连接预测头、不读构造标签、不重新选择权重。
- 保留当前传感器系全部57600原点槽的XYZ和有效mask，保存900×128公共特征。不保存展开后的57600×128特征；索引由固定传感器布局重建。
- 153.057秒完成；主机峰值1,709,981,696字节，CUDA allocator峰值保留98,566,144字节。该CUDA数不含驱动/context，不能称为整卡占用。整个结果目录3,533,302,883字节，约3.29GiB。
- 3360条来源摘要全部唯一，全部记录同一编码器状态摘要；217个来源文件包含6元数据、210输入和1权重。结束时输入摘要和模型状态再次核对。

正式预检无错误/警告，运行前56项相关测试通过（2.39秒）。执行期间出现Torch旧TF32设置接口弃用提示，未改变其关闭设置，运行成功；没有为消除此提示重跑。

运行后的3372个seal条目全部SHA核验通过。seal SHA：`31bbe090558dc1c812bbf4aa6783cc15fad4618cbc095898128fede0b6f90698`。

编码器状态SHA：`200f5c2fbe66d68961cf2aea06e21f747cb5b8d419536008491bb58d3641a8cb`。文件权重SHA与张量状态SHA是不同对象，均保留。

主要证据：`config/run_spec.json`、`config/data_card.json`、`config/environment.json`、`logs/observations.jsonl`、`artifacts/feature_manifest.json`、`artifacts/features/`、`metrics/summary.json`、`RUN_STATE.json`、`artifacts/evidence_sha256.txt`。

意义：A/B/C后续不需要重复运行旧编码器，可直接复用同一份真实公共特征。它不证明面片关系有效，也未解决标签质量问题。新标签0、优化器更新0、科学门未升级。下一主线回到可训练的观测标签与固定小样本拟合前置条件，不继续新增编码器或中心拟合诊断。旧权重、失败结果及论文图片全部保留。
