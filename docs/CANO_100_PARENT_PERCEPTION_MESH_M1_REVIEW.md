# Cano 100 Parent Perception Mesh M1 Review

状态：`FAIL_CANO_100_PARENT_PERCEPTION_MESH_M1`  
日期：2026-08-11  
正式运行：`results/gate0_baseline/gate0_20260811_cano_100_parent_perception_mesh_m1_immutable_assets_seed0`

## 1. 正式结果

M1 在 manifest 第 45 个 parent `S05_flat_branch_medium_C05` 按冻结 stop rule 停止。执行约 1129.69 s，产出 45 个 primary mesh、45 份 parent metrics、37 张 train 完整图、0 replay，结果约 504 MiB；415/415 evidence hash 复核通过。C06--C10 与 R06--R10 没有开始，无重试、换 seed、替换 parent、阈值修改或失败后继续。

前 44 个资产全部通过。失败 parent 的 graph、splines、operation trace、V2R parent identity、axis coverage、有限性、索引、AABB、组件和 source tunnel 检查全部通过；120,208 vertices、240,422 triangles、单组件、largest-component fraction 1.0。唯一失败项为 `degenerate_triangle_count=1`，冻结合同要求严格为 0。

## 2. 退化面的精确证据

失败 triangle index 为 140706，vertex indices 为 `[16280, 16278, 16279]`。三个点的 y/z 完全相同，x 分别约为 `-26.688900/-26.160700/-26.304701 m`，因此三点严格共线，doubled area 为 0。三边约 `0.528200/0.144001/0.384199 m`。其余 triangle 的最小非零 doubled area 为约 `1.5409e-05`，不存在一组接近阈值的模糊面。

只读内存反事实删除该面后：triangle 变为 240,421，组件仍为 1，largest-component fraction 仍为 1.0，surface area 从 `33925.77670537723` 到 `33925.77670537723 m²`，变化为 0。该反事实没有写回 sealed OBJ。

## 3. 第二个独立缺陷：预览 provenance

M1 executor 复用了 M0 `_render_complete_train_map`，而该 helper 把标题硬编码为 `M0 TRAIN ONLY`。因此已生成的 37 张图虽然 parent、recipe、topology seed、geometry seed 和几何内容来自 M1，却错误标注阶段为 M0。这个问题不影响网格，但违反阶段性科研可视化必须正确标识 provenance 的要求。失败样本图的几何覆盖没有明显断裂，但本批所有预览均不能作为最终 M1 图证据。

## 4. 影响边界

这不是 topology 数据错误，也不是结构类别缺失；它是 native Poisson mesh 中一个零表面积面与 M0 标题复用造成的材料/证据合同问题。正式 LiDAR、标签、NPZ、训练、模型、仿真和 M-TARE change 仍全部为 0。45 个失败批资产只保留为诊断证据，不得拼入未来正式 M1R 或训练集。

## 5. 推荐纠正路线 M1R

对全部 100 parent 从头执行统一、预先声明的 perception-only sanitation：在 native materialization 后、最终 OBJ 封存前，以 float64 计算每个 triangle 的 doubled area，确定性删除 `<=1e-12` 的零面积面；记录 raw OBJ SHA-256、删除数、清理前后 triangle 数和面积差。最终资产仍必须满足 M0F/M1 的全部 source/quality 合同，包括清理后退化面严格为 0。该操作不补面、不移动 vertex、不重采样、不换 seed、不做 keep-best。

同时让 renderer 接收明确 stage label，M1R train 图必须写 `M1R TRAIN ONLY`；validation/development-test 仍不渲染、不人工检查。

不推荐直接把“允许 1 个退化面”写进阈值，因为这会把无效 primitive 留给后续 raycasting，并弱化资产合同。也不建议从第 46 个继续：那会把未清理和已清理资产混入同一正式批次。
