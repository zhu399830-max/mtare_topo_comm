# V3 实验治理工具

这些工具只管理实验边界和证据，不实现研究算法。

## 使用顺序

1. 在 `configs/v3/gateN/` 写 run spec，并把 `user_authorization` 保持为 `DRAFT`。
2. 向用户说明 Gate、数据、方法、baseline、成本、输出和验收条件。
3. 用户确认后，将批准范围和确认记录写入 spec；数据操作还需填写已批准 data card。
4. 执行 `python3 tools/v3/preflight.py --spec <spec>`。
5. 通过后执行 `python3 tools/v3/create_run.py --spec <spec>`。
6. 检查结果目录 `config/command.txt`，再执行冻结命令。
7. 保存 log、metrics、summary 和必要 artifact，最后更新项目状态。

## 工具边界

- `preflight.py`：只读检查 Gate、用户批准、运行互斥、路径、data card、AI annotation contract 和 strict-test 泄漏。
- `create_run.py`：创建不可覆盖的标准结果目录并快照配置；绝不执行命令。
- `validate_manifest.py`：单独校验 data card。
- `update_status.py`：只更新运行状态、最近结果、阻塞和下一步；没有 Gate 变更参数。

工具使用 Python 标准库，不要求训练环境或 ROS 环境。治理测试命令：

```bash
python3 -m unittest discover -s tests/v3/unit -p 'test_*.py' -v
```
