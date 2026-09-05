# Navigation-grade 地下几何 V1

状态：`SINGLE_WORLD_OFFLINE_AND_ISAAC_IMPORT_PASS`  
范围：一张 TNG、一个 `g001_navigation_grade`；没有动态机器人、LiDAR、数据集或训练。

## 1. 为什么 g000 不能批量使用

`g000` 只证明 topology→mesh→USD→Isaac import 链路。它使用直线胶囊体并集，没有曲率、机器人实体和局部坡度验收。进一步审计还发现历史 TNG 生成器只限制 RGTG 树边坡度，CTG connector 漏掉坡度限制；历史 parent 因此保留为导入烟雾证据，但禁止作为 navigation-grade 数据母图。

## 2. g001 方法

1. CTG 与 RGTG 使用同一 `max_abs_pitch_rad` 合同；
2. degree-2 节点使用相邻边方向形成连续切线，edge 内采用 cubic Hermite 曲线；
3. 轻微横向曲率在端点位置和切线处归零；
4. 高程按累计水平弧长分配，避免参数化制造局部陡坡；
5. 截面沿曲线路径扫掠，地面采用 global-Z floor profile；
6. degree≥3 节点使用 1.15 倍局部洞室填平交汇孔洞；
7. free-space union 由 0.4 m voxel 与固定 marching tetrahedra 提取单一封闭表面；
8. USD 同时保存 collision Mesh 和只用于审计的 26 条 NavigationCenterlines。

## 3. 机器人验收合同

- 机器人本体半径：0.60 m；安全边界：0.20 m；探针半径：0.80 m；
- 高度：1.50 m；底部离地：0.25 m；
- 最大路径坡度：0.22 rad；
- 最小路径转弯半径：2.50 m；
- 对每个路径采样点，在底部/中部/顶部检查中心与 12 个圆周点；
- 全部实体探针必须落在同一 free-space voxel component；
- 非相邻曲线路径净距必须满足最大截面、洞室和 1.0 m geometry guard；
- mesh 必须 watertight、单组件、零退化面，且 genus 等于 TNG cycle rank。

这仍是几何静态认证，不等于动力学 rollout。轮胎/底盘与三角地面的接触稳定性必须在下一阶段单独验证。

## 4. 结果

- parent：`tng_84998d00587e03dc`；24 nodes、26 edges、cycle rank 3；
- 最大原始边坡度：0.180190890 rad；
- 最大曲线路径坡度：0.180190773 rad；
- 最小转弯半径：2.837491 m；
- 非相邻路径最小净距：6.856996 m，要求 6.405 m；
- 机器人探针：43,953，失败 0；
- mesh：159,784 vertices、319,576 triangles、watertight、genus 3；
- replay mesh/OBJ/USD hash：一致；
- Isaac Sim 6.0.1：导入 PASS，26 curves、1,127 curve points、碰撞 API 均可读取。

## 5. 失败证据

四次被否决的结果分别为 CTG/曲线超坡度、转弯半径不足和额外 genus。阈值没有降低，记录位于 `logs/g001_navigation_grade_failures.json`。

## 6. 下一停止点

下一步只允许在这张 `g001` 中加入测试机器人，执行静态落地、轮地接触、沿中心线低速 rollout 和碰撞/卡死检查。该项通过后才允许一个内部 LiDAR smoke；仍不批量生成地图。
