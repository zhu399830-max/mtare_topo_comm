# Phase 4 C09 Read-only Continuous Contract Audit V1

日期：2026-08-17

## 结论

十个 C09 validation worlds 的连续 route、graph/spline、junction-window 与 route-conditioned support 语义预检全部通过。机器清单见 `configs/v3/gate4/cano_c09_readonly_continuous_contract_manifest_v1.json`。

本轮只读原有 C09 `graph.json/splines.json/mesh.obj/geometry_parameters.json`，没有生成 trajectory、LiDAR、模型输出或 graph replay；C10/M-TARE 读取为 0。首次语义统计仅在 stdout JSON 编码阶段因 `numpy.bool_` 失败，未产生资产或改变计算；显式转为原生 `bool` 后同一只读计算完成。

## 精确数据合同

- 10 个 topology parents / 10 条独立 doubled-edge trajectories
- 1,023 nodes、1,027 edges、2,054 directed traversals
- 31,654.752491 m route length
- 固定 2 m 采样得到 15,833 frames
- 71 个 degree>=3 windows、62 个 terminal nodes
- 2,054 arc-incidence records、213 incident support arcs
- 1,331 window frames、14,502 window-exterior frames

所有 edge 恰好正反各一次；最大 connector=`0.004336588 m < 0.5 m`；最小 connector-to-native-mesh surface distance=`0.884980679 m >= 0.8 m`。窗口 overlap、membership conflict、non-incident window frame 均为 0，全部 support query finite 且唯一。

历史 `31,442.449 m / 15,725 frames` 是节点直线距离估计，分别少 `212.303491 m / 108 frames`，正式方案禁止沿用。

## 后续 geometry qualification 的固定规模

若继续采用 C08 V8R 相同三分辨率合同，C09 需要：

- 47,499 frame-resolution audits
- 34,199,280 horizontal collision rays
- 47,499 upward collision rays与47,499 support evaluations
- 639 window-layer-resolution patches

C08 的150 patches正式运行约3.90小时；线性缩放到639 patches约16.62小时。原12小时cap不足，因此不能直接复制C08 runner。推荐新的单一不可覆盖geometry run预声明24小时active cap、峰值RAM<=4GiB、disk<=8GiB；方法、分辨率、半径、阈值和failure policy保持不变。备选是停止C09，不建议把十个world拆成可选择性重试的独立正式结论，也不建议删patch或降低分辨率。

## geometry PASS 后的冻结 validation 规模

只有C09 geometry qualification正式PASS后，才可另立causal validation：

- 15,833 unique sensor frames
- 364,792,320 dual-scene full-scan rays
- 47,499 M1D inference frames
- 冻结tuple × 5 methods × 10 worlds = 50 graph replays
- parameter sweep、threshold tuning、seed selection均为0

C09已经参与M1D checkpoint选择，因此只能评价“冻结感知条件下的graph-builder泛化”，不能宣称严格端到端unseen泛化。C10、planner与M-TARE继续禁止。

## 当前边界

审计状态为`PASS_CANO_C09_READONLY_CONTINUOUS_CONTRACT_AUDIT_V1`，但这不是geometry qualification PASS或Gate 4 PASS。下一步需要用户决定是否批准按上述24小时/8GiB预算实施、preflight并执行一次完整C09 route-conditioned geometry qualification。
