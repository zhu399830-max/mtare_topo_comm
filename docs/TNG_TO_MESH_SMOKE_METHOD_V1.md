# TNG 到地下隧道 Mesh 单图烟雾测试方法 V1

状态：`SINGLE_WORLD_IMPORT_PASS_AWAITING_REVIEW`  
日期：2026-08-10  
当前范围：一张 mesh-aware TNG、一个 geometry variant、一次 Isaac headless import；禁止批量世界、传感器采集、标签、数据集和训练。

## 1. 本阶段回答的问题

> 采用 Cano、Tardioli、Mosteo IROS 2024 的 TNG-first 思想，能否将一个隐藏拓扑母图独立转换为拓扑一致、封闭、可碰撞并可被 Isaac 打开的地下隧道 mesh？

论文公开事实是“从地下环境 graph 表示开始，定制 meshing，使隧道遵循 graph，再导入目标模拟器”。原代码不可获得且许可证不可核验，因此本实现不复制第三方代码，只复现因果顺序和接口思想。

## 2. 本次独立实现与论文思想的对应关系

```text
论文思想                         本次 clean-room smoke
TNG 决定连通关系                 已实现 RGTG + CTG topology parent
中心线驱动隧道                  每条 graph edge 是一条 3-D centerline segment
交叉口由相交隧道形成             所有 edge 的截面体素做集合并集
定制 mesh                        对 free-space voxel union 提取封闭边界
导入任意仿真器                  同一三角网格导出 OBJ 与 USD
```

本次不用来源不明的 Poisson/graph-to-mesh 代码。体素并集的优点是岔口自动融合、结果确定、容易验证 watertight；缺点是表面有离散台阶。它是 V0/V1 可行性原型，不是最终论文级视觉几何。

## 3. 为什么必须重新生成 mesh-aware TNG

抽象图 v2 `tng_762e384fdc6b6fa0` 的采样非相邻边净距为 3.451 m。若隧道最大半宽约 2.75 m，两条非相邻隧道需要至少：

```text
2 × 2.75 m + 1.0 m geometry guard = 6.5 m
```

否则体积会意外融合，产生 TNG 没有的连接。v2 仍通过抽象图合同，但禁止用于本次 5 m 级宽度的 mesh。新 TNG 使用 6.5 m 的边—边和节点—边门槛；不能为了保留旧 hash 缩小或忽略几何冲突。

## 4. 冻结参数

### 4.1 拓扑参数

- 24 nodes；3 CTG connectors；cycle rank 3；
- base segment 16 m，长度噪声 2 m；
- 最大 degree 4；branch probability 0.30；
- 最大绝对坡度 0.22 rad（约 12.6°）；
- 非相邻 edge clearance 与 node-edge clearance 均至少 6.5 m；
- topology seed 与 geometry seed 独立。

### 4.2 geometry variant `g000`

- right-handed、Z-up、metre；
- nominal lateral half-width 2.50 m，每 edge 由 geometry seed 确定 ±0.20 m 变化；
- nominal vertical radius 2.50 m，每 edge ±0.15 m；
- floor clipping depth 1.50 m，形成约 4.0 m 内部净高；
- voxel size 0.50 m；
- 交叉口使用所有 edge free-space 的集合并集；
- 边界使用固定全局对角线的 marching tetrahedra 提取，避免直接体素方块表面的非流形棱/角歧义；
- 无 clutter、坍塌、材质随机化和 surface noise。

## 5. Mesh 与拓扑验收

生成后必须同时满足：

1. 固定 topology/geometry seeds 重放得到同一 topology hash、mesh hash；
2. free-space voxel 只有一个 6-neighbour connected component；
3. 三角形无重复顶点、零面积、NaN 或 Inf；
4. 每条无向 mesh edge 恰好属于两个三角形，即 watertight 2-manifold；
5. mesh connected component 为 1；
6. mesh genus 等于 TNG cycle rank；不允许近邻隧道融合形成额外 handle；
7. 最大中心线坡度不超过 0.22 rad；
8. 每条 TNG edge 的中心线样本和 floor sample 均落在预期 free-space 中；
9. OBJ 与 USD 使用同一 vertex/index hash；
10. Isaac headless 打开 USD 后能找到一个 Mesh prim、有效 extent、triangle count 和 collision API。

这些检查只证明单图生成/导入链可工作。体素中心线分析不等于机器人 footprint rollout，后者仍是两个 geometry variant 认证前的必要门槛。

## 6. 本轮必须保存

- mesh-aware topology JSON；
- geometry config、seed bundle、生成器版本与 hash；
- render/collision OBJ；
- Isaac USD；
- mesh validation JSON；
- graph/mesh 完整俯视、立面与三维预览；
- Isaac import stdout/stderr 与机器可读结果；
- 所有失败尝试及否决原因。

## 7. 停止边界

单图 USD import 通过后立即停在用户核验点。未经新的 data card 与阶段确认，不生成第二个 geometry variant，不采 LiDAR，不构建 5-TNG pilot，不训练任何模型。

## 8. 实际结果

- mesh-aware parent：`tng_3eca286d5e4c4cd3`，24 nodes、26 edges、cycle rank 3、采样净距 6.894 m；
- geometry `g000`：111,864 vertices、223,736 triangles、0 degenerate face、单一 watertight 2-manifold、genus 3，重放 mesh hash 一致；
- USD v1：真实 Isaac 导入失败，定位为序列化数组分块缺逗号，失败文件和日志保留；
- USD v2：修复并加回归测试后，在官方 Isaac Sim 6.0.1 中导入 `PASS`，完成 10 次 update，并确认 extent、metre/Z-up 和碰撞 API；
- Isaac RTX preview：`FAIL`，headless renderer 无法推进 scheduled frame，不能声称已取得仿真截图；
- 审核图：从验收 OBJ 全部表面顶点生成三视图，属于 offline actual-mesh projection；
- 测试：40/40；数据观察数 0；训练模型数 0。

复现命令见 `docs/RUN_SINGLE_TNG_MESH_SMOKE.md`。
