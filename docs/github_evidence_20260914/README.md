# 研究结果归档 · 2026-09-14

此归档保留成功、失败和未完成的研究证据，不改变任何历史验收结论。没有新训练、仿真或本地资产删除。

## 从哪里看

- [中文研究复盘](../GSE_RESEARCH_REVIEW_20260914.html)：下载后用浏览器打开，19 张原实验图已内嵌，可离线阅读。
- [原始结果目录](../../results/)：图片、PDF、小型指标、日志和运行说明，保持原路径。
- [Release 附件](https://github.com/zhu399830-max/mtare_topo_comm/releases/tag/research-evidence-20260914)：43 个独立 ZIP，共 5,283,597,379 字节，包含大型结果明细及 .pt 权重。
- `included.jsonl`：34,827 个已选择文件的原路径、SHA-256、Git/附件位置；重复文件可通过记录恢复。
- `not_uploaded.jsonl`：未上传的 462,276 个文件，约 423.6 GB，逐项记录大小及原因；未读取这些原始数据内容，也未为它们生成内容哈希。
- `summary.json` 是打包时快照，不是上传成功证明；`upload_receipt.json` 在远端附件大小及 SHA-256 全部通过后生成。

27,682 个结果文件直接进入 Git；7,145 个进入附件，去除 982 个内容完全相同的副本后打包。代码、配置和原有 docs 另由 Git 保存。原始扫描、NumPy/Zarr 数据、mesh、外部依赖及环境不在完整备份范围内，不能宣称整机已备份。私有仓库链接需要账号访问权限。

## 恢复附件

在克隆仓库后，下载全部附件到单独目录：

```sh
gh release download research-evidence-20260914 --repo zhu399830-max/mtare_topo_comm --dir /your/evidence-assets
cd /your/evidence-assets
sha256sum -c SHA256SUMS.txt
```

再于仓库中运行：

```sh
python3 tools/v3/restore_github_evidence.py --assets /your/evidence-assets --destination /your/mtare_topo_comm
```

恢复工具按清单校验内容，不覆盖不同内容的已有文件，也不会加载执行模型权重。不同运行使用不同环境，恢复文件不等于全部实验可以直接运行。
