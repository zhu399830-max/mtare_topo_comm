# GSE-Graph / M-TARE structural exploration research

地下环境几何结构表示、拓扑任务维护与 M-TARE 集成研究。此仓库保存当前代码、配置、开发记录、测试和 `docs/` 中的论文图片与 PDF。

## 当前状态

这是研究开发仓库，不是已验证成功的完整方法。输入、冻结模型推理、局部点云配准和原生执行接线已有独立证据；学习特征影响真实探索决策及相对基线收益尚未完成验证。当前权威状态见 [docs/PLAN.md](docs/PLAN.md)、[docs/PROGRESS.md](docs/PROGRESS.md) 和 [results/project_status.json](results/project_status.json)。历史失败结论保留，不因上传 GitHub 改判。

## 入口

- [中文研究复盘（下载 HTML 后打开，内嵌 19 张实验图）](docs/GSE_RESEARCH_REVIEW_20260914.html)
- [实验结果归档与恢复说明](docs/github_evidence_20260914/README.md)
- [大型结果与模型附件](https://github.com/zhu399830-max/mtare_topo_comm/releases/tag/research-evidence-20260914)

- `src/mtare_topo/`：数据、几何表示、结构、拓扑和集成代码。
- `tools/v3/`、`configs/v3/`、`tests/v3/`：运行工具、冻结规格与测试。
- `integration/native_structure_bridge/`：原生 M-TARE 桥接及说明。
- `docs/figures/`：阶段结果图及对应说明；图中成功软件检查不等于论文方法成功。
- [docs/GSE_LEARNED_FRONTEND_INTEGRATION.md](docs/GSE_LEARNED_FRONTEND_INTEGRATION.md)：最新系统接线与尚未解决的问题。
- [docs/GITHUB_SYNC.md](docs/GITHUB_SYNC.md)：本次同步范围与未上传资产。

复现实验前须遵守 `AGENTS.md` 和当前计划中的数据隔离、环境及运行约束。仓库内含历史绝对路径和多个独立环境，不能保证换一台机器后仅凭 clone 即可运行所有实验。

## 不包含完整数据备份

本次显式整理结果图片、小型指标与记录到 Git，大型文本结果和模型权重另放 Release。确切范围及上传校验以归档清单和回执为准。ROS bag、数组数据集、环境与外部依赖工作树不包含在完整备份中。原文件仍留在本机；路径和哈希引用不代表对应资产已经上传。
