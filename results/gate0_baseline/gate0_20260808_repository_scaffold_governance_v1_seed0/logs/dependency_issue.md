# 测试框架问题记录

- 事实：系统 `/usr/bin/python3` 未安装 `pytest`，首次测试命令返回 `No module named pytest`。
- 影响：只影响测试执行器，不影响数据、方法、Gate 或任何研究结论。
- 选择：未安装新依赖，改用 Python 标准库 `unittest`，保留相同测试目标并增加审批绑定与运行互斥测试。
- 结果：最终 10 项测试全部通过。
