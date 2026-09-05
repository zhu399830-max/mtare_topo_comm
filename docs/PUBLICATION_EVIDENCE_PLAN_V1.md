# 论文贡献与期刊证据计划 V1

状态：`HISTORICAL_ASPIRATIONAL_PLAN_CURRENT_EVIDENCE_TRACKED_SEPARATELY`  
日期：2026-08-10

> 2026-08-23 说明：本文件保留早期投稿目标和理想证据上限。当前已实现方法、已证明结论与剩余实验
> 以 `docs/PAPER_COMPLETION_MATRIX_V1.md` 为准。尤其不得把本文件中的 R/D/U 或真实地下闭环要求
> 写成已经实现或已经通过；当前正式方法是 learned exit direction + frozen geometry count/role、persistent
> causal topology 和 M-TARE global-planner replacement。

## 1. 能否发好期刊

有可能，但“使用已有生成器 + 模仿已有 CNN + 接 M-TARE”本身只是系统集成，不足以形成好期刊论文。2026 年 JFR 论文已经覆盖合成地下 LiDAR、出口 CNN、exit tracking、纯拓扑导航与真实隧道 zero-shot。我们的论文必须用完整因果证据证明新的研究问题。

建议中心命题：

> 在因果残缺 LiDAR 下，planner-consistent 且带不确定性的局部结构语义，能否构建比几何 keyframe 和出口角度检测更稳定的 exit-stub 拓扑图，并在保留同一 local planner 时提高单/多机器人地下探索效率？

## 2. 必须成立的贡献

1. `Perception`：不是只检测 tunnel bearing，而是预测与执行 footprint/local planner 一致的 R/D/U，并证明跨程序化世界和真实地下域泛化；
2. `Topology`：提出可解释的 persistent structural node、verified edge 和 lifecycle-aware exit stub，报告 preservation/connectivity/distortion，而非只展示一张图；
3. `Planning`：在相同 M-TARE sensor/local planner/control 下替换高层表示和目标选择，用 paired seeds 归因探索提升；
4. `Multi-robot`：证明图共享和 exit 分配减少重复探索与通信负担；若没有这项，论文贡献应降为单机器人版本；
5. `Reproducibility`：发布合法可用的生成适配器、split manifest、模型配置、评价代码和足够的数据 provenance。

任意一项没有实验支撑，都不能写成 contribution。

## 3. 最小对照与消融

- `M-TARE original`：原全局规划；
- `B0 geometry-only`：相同 partial BEV 的规则出口；
- `B1 Cano-like`：range-image 360-degree exit CNN；
- `M1 R/D/U`：主感知；
- `T0 distance/keyframe graph`：无结构语义图；
- `T1 bearing-only topology`：B1 + 相同 topology builder；
- `T2 structural topology`：M1 + topology builder；
- `P0 M-TARE frontier` 对 `P1 graph exit-stub planning`；
- 多机器人 `independent/nearest`, `M-TARE coordination`, `Hungarian graph allocation`。

核心消融：无 causal history、无 uncertainty、无 persistence、无 exit lifecycle、无 semantic role、无 multi-robot communication term。禁止同时改感知、图和 planner 后只报一个最终数字。

## 4. 数据与实验最低门槛

### 仿真

- 100 个独立 TNG topology parent 的开发数据学习曲线，上限 50 万 observation；至少生成 200 个 mesh realization，并报告 topology-parent 原子 split；
- same-TNG/different-certified-geometry paired benchmark，以及 different-TNG/matched-visible-geometry hard-negative benchmark；
- M-TARE 官方五图 parity table，但非地下图只作补充；
- 项目现有 5 个地下开发/回归图修复后做系统开发；
- 至少两个 sealed topology family，每 family 候选 10 world；
- 每个正式 closed-loop 条件至少 5 个 paired seed，报告每次 run、均值/标准差、median/IQR 和 bootstrap CI；
- 失败/无运动必须按冻结规则区分 invalid system run 与 valid poor planner outcome。

### 真实世界

仅用 LAMP/SubT 离线 bag 可以支持感知泛化，但不足以独立支撑“改善探索”的 field claim。若目标是 JFR 或 T-RO，建议至少：

- 两个几何族不同、训练未接触的真实地下 site；
- 每个 site 至少 3--5 次闭环运行或可复现实机任务；
- 原方法/主方法共享 robot、sensor、local planner 和任务；
- 保存结构语义、图、decision trace、coverage 和失败视频；
- 明确安全员干预、定位失败和通信中断。

没有实地条件时，最现实的路线是 RA-L/强会议级别，而不是把纯仿真包装成 field robotics。

## 5. 期刊定位

### IEEE Robotics and Automation Letters（最现实的第一目标）

RA-L 面向及时、简洁的创新机器人研究和应用结果，篇幅短，适合一条聚焦贡献。满足条件：冻结主方法，强仿真、严格 unseen、至少一组真实回放或实机演示，且只讲“结构语义拓扑改善地下探索”这一条主线。

官方范围：https://www.ieee-ras.org/publications/ieee-robotics-and-automation-letters/

### Journal of Field Robotics（最匹配，但必须有 field 证据）

JFR 聚焦非结构化、动态环境中的有影响力 field robotics，明确覆盖 mining 与 search and rescue。由于最近对标论文本身已发 JFR，我们需要真实地下闭环、多场景失败分析和系统完整性，而不是只提高出口检测准确率。

官方范围：https://onlinelibrary.wiley.com/journal/15564967

### IEEE Transactions on Robotics（冲刺目标）

T-RO 要求对机器人领域 state of the art 的 major advance，强调未知、不可直接感知的非结构环境。仅组合已有模块不够；需要更一般的部分观测结构图理论/算法、显著跨域与多机器人结果、系统性比较和真实部署。

官方范围：https://www.ieee-ras.org/publications/t-ro/

## 6. 推荐投稿策略

当前推荐目标不是预先承诺 T-RO，而是按证据升级：

```text
Gate 3--4 仅感知/离线图通过
  -> 还不投稿完整系统论文

Gate 6 单机严格仿真 + 一组真实验证通过
  -> RA-L 候选

Gate 7 多机器人 + 两个真实地下 site + 完整系统证据
  -> JFR 主目标

若形成可泛化算法贡献并显著超过强基线
  -> 再评估 T-RO
```

投稿前必须做一次独立 novelty review，更新到当时最新工作；当前文献判断不能替代未来投稿时检索。

## 7. 论文故事线

建议题目方向：`Planner-Consistent Structural Semantics for Persistent Topological Exploration in Underground Environments`。

文章结构：问题与 M-TARE 接口边界；客观合成数据和 R/D/U；结构语义与 exit-stub 图；单/多机器人图规划；因果消融；仿真 parity/strict test；真实地下部署；局限和失败。

不把“我们用了 Isaac”“我们生成 50 万数据”写成贡献；它们是可复现基础设施。生成器必须完整引用原作者，合法复用部分单列 third-party attribution。
