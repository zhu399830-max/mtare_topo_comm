# 仓库状态检查

- 根目录 `/home/zeng-workstation/mtare_topo_comm/.git` 存在，但为空。
- `git status --short` 返回 `fatal: not a git repository`。
- `external/SubTGraph/.git` 与 `external/gazebo_cave_world/.git` 是独立的外部仓库；它们不能为项目根目录提供变更追踪。
- 本轮没有初始化、修复或写入任何 Git 元数据。

影响：当前重构只能依赖文件清单和结果证据回溯，无法使用根仓库 commit、diff 或 branch 保护。进入旧代码实质迁移前，建议由用户决定是否在根目录初始化正式 Git 仓库，以及外部仓库采用 submodule、subtree 还是保留独立目录。
