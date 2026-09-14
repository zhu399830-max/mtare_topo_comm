# 951条已有支路方向全部接通

这是已有数据的只读绑定检查，不是新教师、标签导出、模型训练或图实验。

| 分区 | 含路口观察 | 成功绑定支路 |
|---|---:|---:|
| 拟合 | 250 | 774 |
| 校准 | 27 | 87 |
| 开发评价 | 30 | 90 |
| 合计 | 307 | 951 |

原构造文件156份、原朝向数组156份。方向由已有`construction_incident_paths`和`bind_axes`读取原轴线，按对应观察的原yaw转换到当前传感器坐标；不使用窗口开口方向或学生预测猜方向。`read_saved_junction_branches`将末尾尽头参考保留在原记录中，不误当支路。

每次读取构造、目标、原接口和朝向数据均检查清单hash；核对原source task/sequence/五帧，构造canonical hash、接口归属、支路数量和非空见证。全部307观察/951支路通过；未知入口坐标保持None，training_eligible保持False。此结果不证明支路全集完整、入口宽高可得或方向已被模型学会。

## 复现及清单

- `tools/v3/prepare_saved_branch_binding_scope.py`：从原2676记录中取全部307个已有路口记录，不依据模型分数；核对全部307条V3→V2→V1引用链。
- `configs/v3/gate3/saved_branch_binding_scope_v1.json`：无损gzip/base64 JSON封装，解码后623338字节、1145个精确文件引用；朝向解码上界4480720字节。它是读取清单，不是训练授权。
- 解码内容SHA-256：`9965eeac86b827266b76b0318c3bf99f3dee39641cba47ccab3a80c8b25b2bcb`。
- `tools/v3/check_saved_branch_directions.py`：实际全量检查，会话33278正常结束。Zarr通过只含绑定key的内存store读取，不遍历其他数据。无C08—C10。

```sh
PYTHONPATH=src /home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python tools/v3/check_saved_branch_directions.py
```

原始明文清单在工具返回时被截断，未作为有效清单保存；只读重新编译采用无损压缩传输，解码hash固定后才执行绑定。没有重跑旧实验、覆盖原run或修改数据。

## 下一项不再查方向

方向接口至此完成，不继续追加同类方向微测试。下一把307记录与已有公共观测特征按task/sequence/frame进行元数据接线，复用已核实的条件位置负例机制；逐条统计已见支路和原参考接口全集差别，以显式未知掩码界定能评分的支路范围。先固定可用监督/评价合同，再按数据卡进行一次正式最小配对诊断，不能将本轮只读结果冒充训练资格。

307是原人口中的路口部分，不是重新选出的完整训练集；尽头和原背景必须保留在最终配对设计中。完整结构感知→图→探索目标和开发隔离不变，Phase3未通过。
