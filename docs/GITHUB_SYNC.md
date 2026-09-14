# GitHub 同步范围

用户已要求代为创建并上传到 `zhu399830-max` 账户。目标为私有仓库 `mtare_topo_comm`，默认推送 `main`；不会修改账户中的 `hello-agents` 仓库，不强制推送，不改写当前 Git 历史。

提交邮箱：`zhu399830@gmail.com`。已通过 GitHub 官方设备登录确认账号 `zhu399830-max`；凭据由系统 keyring 保存。邮箱和账户主页本身不构成登录凭证，任何密码、令牌或本机凭据文件均不应进入 Git。

## 已核实的上传结果

私有仓库：https://github.com/zhu399830-max/mtare_topo_comm 。默认分支为 `main`。

首次同步快照 `9fe93cc1e8886d3572a87021753a0a9344322aeb` 已成功推送，共5492个文件。远程分支提交与本地一致，GitHub 文件树 SHA `0fdccb637a868f8ee0f180f82e2e7d8845129544` 与本地一致；另行核实 `docs/PHASE4_STAGE_RESULTS_REPORT.pdf` 远程存在。本回执在后续独立提交中保存，不改变实验结论。

## 本次纳入

当前源码、工具、测试、配置、说明文档与必要元数据；额外解除 `docs/` 内论文 PNG/JPG/JPEG/PDF 的忽略。同步前盘点该类图片/报告为181个文件、58,133,916字节。该数量是上传准备清单，不是新增实验或实验样本数。

## 不自动纳入

- `results/` 的原始运行（仅保留已有 `results/project_status.json` 例外）。
- 模型、NumPy/Zarr 数据、ROS bag、视频、构建产物及本地环境。
- `external/` 与 `third_party/` 的独立 Git 工作树；上游地址、提交与脏状态另见 `EXTERNAL_DEPENDENCIES_GIT.json`。其中本地生成数据不能靠重新 clone 上游恢复。
- `docs/exports/gse_direction_task_portable_20260912.zip`：134,569,619字节，超过普通 GitHub 单文件上传范围；保留在本地，不写入本次提交。不会因为排除它而删除文件或重写旧对象。
- 本机授权、密钥和环境私密配置。

此项是代码与论文材料同步，**不是模型、数据及所有实验结果的完整异地备份**。远程创建、push及远程提交SHA比对成功之前，不能登记为“已上传”；实际状态保存在 `results/project_status.json` 的 `repository_sync` 字段。
