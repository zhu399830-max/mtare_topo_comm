#!/usr/bin/env python3
"""Render the sealed Phase-4 stage results as a presentation-ready PDF."""

from __future__ import annotations

from pathlib import Path
from textwrap import wrap
import json

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "docs" / "PHASE4_STAGE_RESULTS_REPORT.pdf"

DATA_IMAGE = ROOT / "results/gate1_data/gate1_20260812_cano_phase2_supervised_range_dataset_v2r_seed0/previews/train_complete_samples/page_06_S06.png"
MODEL_IMAGE = ROOT / "results/gate2_representation/gate2_20260813_cano_phase3_masking_corrective_recovery_seed2_v1_seed2/previews/v1r3_vs_m1d_corrective.png"
GEOMETRY_IMAGE = ROOT / "results/gate4_topology/gate4_20260816_cano_c08_route_conditioned_support_corrective_v8r_seed0/previews/S10_3d_complex_C08_complete_qualification.png"
TRAJECTORY_IMAGE = ROOT / "results/gate4_topology/gate4_20260817_cano_c08_route_conditioned_causal_topology_replay_v2r_seed0/previews/S10_3d_complex_C08_objective_role_trajectory.png"
C09_RUN = ROOT / "results/gate4_topology/gate4_20260820_cano_c09_causal_topology_validation_v1_seed0"
C09_TRAJECTORY_IMAGE = C09_RUN / "previews/S10_3d_complex_C09_objective_role_trajectory.png"

FONT_REGULAR = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
FP = FontProperties(fname=FONT_REGULAR)
FP_BOLD = FontProperties(fname=FONT_BOLD)

NAVY = "#13263A"
BLUE = "#2D6CDF"
PALE_BLUE = "#EAF1FF"
ORANGE = "#E9822B"
PALE_ORANGE = "#FFF1E6"
GREEN = "#198754"
PALE_GREEN = "#E9F6EF"
RED = "#C64242"
GRAY = "#5F6B76"
LIGHT = "#F4F6F8"
WHITE = "#FFFFFF"


def new_page(title: str, section: str, page_no: int):
    fig = plt.figure(figsize=(13.333, 7.5), facecolor=WHITE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_autoscale_on(False)
    ax.text(0.055, 0.935, section, fontproperties=FP_BOLD, fontsize=10, color=BLUE, va="top")
    ax.text(0.055, 0.885, title, fontproperties=FP_BOLD, fontsize=24, color=NAVY, va="top")
    ax.plot([0.055, 0.945], [0.83, 0.83], color="#DDE3EA", linewidth=1)
    ax.text(0.055, 0.035, "地下结构语义与因果拓扑图｜阶段实验结果", fontproperties=FP, fontsize=8, color=GRAY)
    ax.text(0.945, 0.035, str(page_no), fontproperties=FP, fontsize=8, color=GRAY, ha="right")
    return fig, ax


def rounded_box(ax, xy, width, height, facecolor=LIGHT, edgecolor="none", radius=0.018):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    return patch


def paragraph(ax, x, y, text, width=50, fontsize=13, color=NAVY, line_height=0.042, bold=False):
    lines = []
    for part in text.split("\n"):
        lines.extend(wrap(part, width=width, break_long_words=False) or [""])
    prop = FP_BOLD if bold else FP
    for i, line in enumerate(lines):
        ax.text(x, y - i * line_height, line, fontproperties=prop, fontsize=fontsize, color=color, va="top")
    return y - len(lines) * line_height


def place_image(fig, path: Path, rect):
    image = Image.open(path).convert("RGB")
    ax = fig.add_axes(rect)
    ax.imshow(image)
    ax.set_axis_off()
    return ax


def save(pdf, fig):
    pdf.savefig(fig, facecolor=fig.get_facecolor(), bbox_inches=None)
    plt.close(fig)


def cover(pdf):
    fig = plt.figure(figsize=(13.333, 7.5), facecolor=NAVY)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.add_patch(FancyBboxPatch((0.055, 0.12), 0.015, 0.72, boxstyle="round,pad=0,rounding_size=0.005", facecolor=ORANGE, edgecolor="none", transform=ax.transAxes))
    ax.text(0.10, 0.77, "地下结构语义与因果拓扑图", fontproperties=FP_BOLD, fontsize=31, color=WHITE, va="top")
    ax.text(0.10, 0.67, "阶段实验结果与当前边界", fontproperties=FP_BOLD, fontsize=25, color="#BFD3FF", va="top")
    ax.text(0.10, 0.51, "从全周 LiDAR 学习局部结构语义，\n再按时间顺序建立可探索的拓扑图", fontproperties=FP, fontsize=17, color=WHITE, va="top", linespacing=1.55)
    rounded_box(ax, (0.10, 0.255), 0.75, 0.12, facecolor="#203A55")
    ax.text(0.125, 0.335, "阶段结论", fontproperties=FP_BOLD, fontsize=11, color="#9FC0FF", va="top")
    ax.text(0.125, 0.295, "模型已训练完成；C08 开发与 C09 冻结参数验证的 LiDAR → 结构语义 → 因果拓扑图链路均已跑通。", fontproperties=FP_BOLD, fontsize=14, color=WHITE, va="top")
    ax.text(0.10, 0.115, "2026-08-20｜Gate 4 当前状态：GATE_MIXED", fontproperties=FP, fontsize=10, color="#B8C5D1")
    save(pdf, fig)


def overview(pdf):
    fig, ax = new_page("我们完成的是一条可运行的结构建图链路", "01  工作目标", 2)
    paragraph(ax, 0.06, 0.78, "目标不是给单帧贴类别，而是让机器人从当前 LiDAR 判断可通方向、分支和结构角色，并把连续观测变成可用于探索的拓扑图。", width=66, fontsize=14)
    labels = [
        ("地下世界", "中心线与客观结构"),
        ("全周 LiDAR", "16×720 距离与有效位"),
        ("M1D 模型", "方向、分支数、结构角色"),
        ("因果更新", "持续性与空间关联"),
        ("拓扑图", "节点、已验证边、未探索出口"),
    ]
    xs = [0.055, 0.238, 0.421, 0.604, 0.787]
    for i, ((head, sub), x) in enumerate(zip(labels, xs)):
        rounded_box(ax, (x, 0.49), 0.15, 0.16, facecolor=PALE_BLUE if i not in (2, 4) else PALE_ORANGE)
        ax.text(x + 0.075, 0.595, head, fontproperties=FP_BOLD, fontsize=14, color=NAVY, ha="center")
        ax.text(x + 0.075, 0.535, sub, fontproperties=FP, fontsize=9.5, color=GRAY, ha="center")
        if i < 4:
            ax.annotate("", xy=(x + 0.178, 0.57), xytext=(x + 0.154, 0.57), arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.8), xycoords=ax.transAxes)
    rounded_box(ax, (0.055, 0.20), 0.42, 0.16, facecolor=PALE_GREEN)
    ax.text(0.08, 0.315, "在线输入", fontproperties=FP_BOLD, fontsize=13, color=GREEN)
    ax.text(0.08, 0.265, "只使用当前 LiDAR 距离和有效位；不读取完整地图、\n真实拓扑图或未来帧。", fontproperties=FP, fontsize=12, color=NAVY, va="top")
    rounded_box(ax, (0.525, 0.20), 0.42, 0.16, facecolor=PALE_ORANGE)
    ax.text(0.55, 0.315, "离线信息", fontproperties=FP_BOLD, fontsize=13, color=ORANGE)
    ax.text(0.55, 0.265, "完整地图只负责产生客观标签、验证传感器位姿\n和评价最后建立的图。", fontproperties=FP, fontsize=12, color=NAVY, va="top")
    save(pdf, fig)


def data_page(pdf):
    fig, ax = new_page("模型看到的数据是什么样", "02  数据与客观标签", 3)
    rounded_box(ax, (0.055, 0.71), 0.89, 0.09, facecolor=LIGHT)
    ax.text(0.075, 0.765, "90 个程序化地下世界｜80 训练 + 10 验证｜22,500 个空间位置｜112,500 帧 LiDAR", fontproperties=FP_BOLD, fontsize=13, color=NAVY, va="center")
    place_image(fig, DATA_IMAGE, [0.055, 0.12, 0.66, 0.55])
    rounded_box(ax, (0.745, 0.47), 0.20, 0.19, facecolor=PALE_BLUE)
    ax.text(0.765, 0.625, "怎么看这张图", fontproperties=FP_BOLD, fontsize=12, color=BLUE, va="top")
    ax.text(0.765, 0.575, "横轴：360° 水平方向\n纵轴：16 条激光线\n颜色：首个障碍物距离\n白线：正确出口方向", fontproperties=FP, fontsize=11, color=NAVY, va="top", linespacing=1.45)
    rounded_box(ax, (0.745, 0.20), 0.20, 0.19, facecolor=PALE_GREEN)
    ax.text(0.765, 0.355, "数据隔离", fontproperties=FP_BOLD, fontsize=12, color=GREEN, va="top")
    ax.text(0.765, 0.305, "训练和验证按世界分开。\n同一地图和相邻帧不会跨 split，\n避免把记忆地图当成泛化。", fontproperties=FP, fontsize=11, color=NAVY, va="top", linespacing=1.45)
    ax.text(0.055, 0.085, "图 1　S06 训练数据页：普通通道、路口与末端使用同一种 LiDAR 输入和客观出口标签。", fontproperties=FP, fontsize=9, color=GRAY)
    save(pdf, fig)


def model_page(pdf):
    fig, ax = new_page("M1D 同时学习方向、分支数量和结构角色", "03  结构语义模型", 4)
    place_image(fig, MODEL_IMAGE, [0.055, 0.35, 0.62, 0.40])
    rounded_box(ax, (0.70, 0.54), 0.245, 0.20, facecolor=PALE_BLUE)
    ax.text(0.72, 0.705, "模型设计", fontproperties=FP_BOLD, fontsize=12, color=BLUE, va="top")
    ax.text(0.72, 0.66, "环形卷积处理 360° 输入，避免首尾方向断开。\n共享编码器输出可通方向、分支数量、结构角色和 128 维结构表示。", fontproperties=FP, fontsize=10.5, color=NAVY, va="top", linespacing=1.4)
    rounded_box(ax, (0.70, 0.35), 0.245, 0.14, facecolor=PALE_ORANGE)
    ax.text(0.72, 0.455, "关键修正", fontproperties=FP_BOLD, fontsize=12, color=ORANGE, va="top")
    ax.text(0.72, 0.41, "训练时遮挡部分射线列，使残缺观测下的结构表示仍保持一致。", fontproperties=FP, fontsize=10.5, color=NAVY, va="top")
    metrics = [("方向 F1", "0.887", "规则基线 0.796"), ("结构角色 F1", "0.882", "三类角色"), ("分支数量 F1", "0.747", "只统计 1–4"), ("遮挡一致性", "0.9996", "修正前约 0.21")]
    for i, (name, value, note) in enumerate(metrics):
        x = 0.055 + i * 0.225
        rounded_box(ax, (x, 0.14), 0.20, 0.13, facecolor=LIGHT)
        ax.text(x + 0.018, 0.235, name, fontproperties=FP_BOLD, fontsize=10.5, color=GRAY, va="top")
        ax.text(x + 0.018, 0.195, value, fontproperties=FP_BOLD, fontsize=20, color=NAVY, va="top")
        ax.text(x + 0.105, 0.18, note, fontproperties=FP, fontsize=8.5, color=GRAY, va="top")
    ax.text(0.055, 0.305, "图 2　橙色为遮挡修正后的 M1D。方向与角色性能保持稳定，遮挡一致性显著恢复。", fontproperties=FP, fontsize=9, color=GRAY)
    save(pdf, fig)


def geometry_page(pdf):
    fig, ax = new_page("先保证 LiDAR 位姿物理有效，再谈模型和拓扑", "04  轨迹几何资格", 5)
    place_image(fig, GEOMETRY_IMAGE, [0.055, 0.28, 0.89, 0.48])
    ax.text(0.055, 0.245, "图 3　S10 复杂三维场景：左为俯视轨迹，中为高度变化，右为整条轨迹的水平、向下和向上安全距离。", fontproperties=FP, fontsize=9, color=GRAY)
    items = [
        ("4,773 / 4,773", "C08 全部轨迹帧通过"),
        ("4,307 / 4,307", "路口窗口外位姿完全不变"),
        ("3 档分辨率", "0.10 / 0.05 / 0.025 m 结论一致"),
        ("用途分离", "原生网格生成 LiDAR；独立场负责安全资格"),
    ]
    for i, (value, label) in enumerate(items):
        x = 0.055 + i * 0.225
        rounded_box(ax, (x, 0.085), 0.20, 0.11, facecolor=PALE_GREEN if i < 3 else PALE_BLUE)
        ax.text(x + 0.018, 0.165, value, fontproperties=FP_BOLD, fontsize=14, color=GREEN if i < 3 else BLUE, va="top")
        ax.text(x + 0.018, 0.12, label, fontproperties=FP, fontsize=9.5, color=NAVY, va="top")
    save(pdf, fig)


def graph_page(pdf):
    fig, ax = new_page("拓扑图严格按照轨迹顺序逐帧建立", "05  因果拓扑回放", 6)
    place_image(fig, TRAJECTORY_IMAGE, [0.055, 0.18, 0.64, 0.57])
    ax.text(0.055, 0.145, "图 4　S10 完整回放轨迹：蓝色普通通道、橙色路口、红色末端，覆盖回环、坡道、多高度和分支。", fontproperties=FP, fontsize=9, color=GRAY)
    steps = [
        ("1", "稳定结构事件", "连续看到分支或结构变化时创建节点"),
        ("2", "实际通过后连边", "机器人走过后才形成已验证边"),
        ("3", "保存未探索出口", "看见但没走过的方向保留为 exit stub"),
        ("4", "控制图规模", "长直通道只保留必要距离锚点"),
        ("5", "回访位置合并", "综合距离、角色和出口方向进行关联"),
    ]
    for i, (n, head, body) in enumerate(steps):
        y = 0.70 - i * 0.115
        ax.text(0.735, y, n, fontproperties=FP_BOLD, fontsize=13, color=WHITE, ha="center", va="center", bbox=dict(boxstyle="circle,pad=0.35", facecolor=BLUE, edgecolor="none"))
        ax.text(0.77, y + 0.018, head, fontproperties=FP_BOLD, fontsize=11, color=NAVY, va="center")
        ax.text(0.77, y - 0.026, body, fontproperties=FP, fontsize=9.5, color=GRAY, va="center")
    save(pdf, fig)


def results_page(pdf):
    fig, ax = new_page("学到的结构语义改善了拓扑图质量", "06  C08 开发结果", 7)
    chart = fig.add_axes([0.08, 0.25, 0.46, 0.48])
    categories = ["拓扑综合分数", "出口 F1"]
    baseline = [0.7435407, 0.4977324]
    m1d = [0.8262623, 0.6549515]
    x = [0, 1]
    chart.bar([v - 0.18 for v in x], baseline, width=0.34, color="#9AA6B2", label="几何规则基线")
    chart.bar([v + 0.18 for v in x], m1d, width=0.34, color=BLUE, label="M1D 三次训练平均")
    chart.set_ylim(0, 1.0); chart.set_xticks(x, categories, fontproperties=FP, fontsize=11)
    chart.tick_params(axis="y", labelsize=9); chart.grid(axis="y", alpha=0.2); chart.set_axisbelow(True)
    chart.spines[["top", "right"]].set_visible(False)
    chart.legend(prop=FP, frameon=False, loc="upper right")
    for pos, val in zip([v - 0.18 for v in x], baseline): chart.text(pos, val + 0.025, f"{val:.3f}", ha="center", fontproperties=FP_BOLD, fontsize=10, color=GRAY)
    for pos, val in zip([v + 0.18 for v in x], m1d): chart.text(pos, val + 0.025, f"{val:.3f}", ha="center", fontproperties=FP_BOLD, fontsize=10, color=BLUE)
    rounded_box(ax, (0.60, 0.57), 0.345, 0.17, facecolor=PALE_GREEN)
    ax.text(0.625, 0.70, "主要提升", fontproperties=FP_BOLD, fontsize=12, color=GREEN, va="top")
    ax.text(0.625, 0.645, "拓扑综合分数  +0.083\n出口 F1              +0.157", fontproperties=FP_BOLD, fontsize=16, color=NAVY, va="top", linespacing=1.55)
    rounded_box(ax, (0.60, 0.36), 0.345, 0.15, facecolor=PALE_BLUE)
    ax.text(0.625, 0.475, "没有被破坏的基本性质", fontproperties=FP_BOLD, fontsize=12, color=BLUE, va="top")
    ax.text(0.625, 0.42, "图连通率 1.000\n已验证边正确率 1.000", fontproperties=FP_BOLD, fontsize=14, color=NAVY, va="top", linespacing=1.5)
    rounded_box(ax, (0.60, 0.15), 0.345, 0.15, facecolor=PALE_ORANGE)
    ax.text(0.625, 0.265, "必须保留的限制", fontproperties=FP_BOLD, fontsize=12, color=ORANGE, va="top")
    ax.text(0.625, 0.215, "参数存在近似并列；S10 最弱；并非每个 seed / world 都优于基线。", fontproperties=FP, fontsize=10.5, color=NAVY, va="top")
    ax.text(0.08, 0.165, "4,773 个唯一传感器帧｜3 个冻结模型｜14,319 次推理｜3,645 次因果图回放", fontproperties=FP_BOLD, fontsize=10.5, color=GRAY)
    save(pdf, fig)


def c09_results_page(pdf):
    fig, ax = new_page("C09 冻结参数验证：完整流程通过，提升并不均匀", "07  C09 验证结果", 8)
    place_image(fig, C09_TRAJECTORY_IMAGE, [0.055, 0.38, 0.55, 0.38])
    ax.text(0.055, 0.345, "图 5　C09 最复杂三维世界的完整客观角色轨迹。", fontproperties=FP, fontsize=9, color=GRAY)
    chart = fig.add_axes([0.65, 0.42, 0.29, 0.31])
    labels = ["B0", "M1D-0", "M1D-1", "M1D-2", "Oracle"]
    composite = [0.805246, 0.816083, 0.825419, 0.807749, 0.850069]
    colors = ["#9AA6B2", BLUE, BLUE, BLUE, ORANGE]
    bars = chart.bar(range(5), composite, color=colors)
    chart.set_ylim(0.75, 0.87); chart.set_xticks(range(5), labels, rotation=25, fontproperties=FP, fontsize=8)
    chart.tick_params(axis="y", labelsize=8); chart.grid(axis="y", alpha=.2); chart.set_axisbelow(True)
    chart.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, composite): chart.text(bar.get_x()+bar.get_width()/2, value+.002, f"{value:.3f}", ha="center", fontsize=7, color=NAVY)
    ax.text(0.65, 0.755, "10 个世界平均综合分", fontproperties=FP_BOLD, fontsize=11, color=NAVY)
    facts = [
        ("15,833", "唯一 LiDAR 帧"), ("47,499", "冻结模型推理帧"),
        ("50 / 50", "固定参数拓扑回放"), ("1,881 / 1,881", "封印文件复核通过"),
    ]
    for i, (value, label) in enumerate(facts):
        x = 0.055 + i * 0.225
        rounded_box(ax, (x, 0.17), 0.20, 0.11, facecolor=PALE_GREEN)
        ax.text(x+.018, 0.247, value, fontproperties=FP_BOLD, fontsize=15, color=GREEN, va="top")
        ax.text(x+.018, 0.205, label, fontproperties=FP, fontsize=9.5, color=NAVY, va="top")
    ax.text(0.055, 0.115, "M1D seed1 的综合分最高（0.825），但三 seed 平均出口 F1 与 B0 接近，说明跨世界收益存在 seed 和场景差异。", fontproperties=FP_BOLD, fontsize=11, color=NAVY)
    ax.text(0.055, 0.075, "C09 曾参与 checkpoint 选择，因此这里只证明冻结图参数下的验证结果，不是严格端到端未见测试；C10 仍未读取。", fontproperties=FP, fontsize=9.5, color=RED)
    save(pdf, fig)


def c09_topology_page(pdf):
    fig, ax = new_page("实际生成的拓扑图：S10 三种语义输入对比", "08  C09 拓扑结果图", 9)
    methods = [("b0", "规则基线 B0"), ("m1d_seed1", "M1D seed1"), ("oracle", "客观语义 Oracle")]
    summary = json.loads((C09_RUN / "metrics/graph_summary.json").read_text(encoding="utf-8"))
    by_key = {(row["method"], row["world"]): row for row in summary["validation_metrics"]}
    world = "S10_3d_complex_C09"
    for index, (method, title) in enumerate(methods):
        graph = json.loads((C09_RUN / f"artifacts/selected_graphs/{method}_{world}_graph.json").read_text(encoding="utf-8"))
        pane = fig.add_axes([0.055 + index * 0.30, 0.29, 0.27, 0.43])
        nodes = {int(node["id"]): node for node in graph["nodes"]}
        for edge in graph["edges"]:
            a, b = nodes[int(edge["from"])]["xyz_m"], nodes[int(edge["to"])]["xyz_m"]
            pane.plot([a[0], b[0]], [a[1], b[1]], color="#AAB5C0", lw=.45, alpha=.8)
        for kind, color, size in (("anchor", "#6F7D89", 5), ("structural", ORANGE, 15)):
            subset = [node for node in graph["nodes"] if node["node_kind"] == kind]
            pane.scatter([n["xyz_m"][0] for n in subset], [n["xyz_m"][1] for n in subset], s=size, c=color, edgecolors="none", zorder=3)
        pane.set_aspect("equal", adjustable="datalim"); pane.set_axis_off(); pane.set_title(title, fontproperties=FP_BOLD, fontsize=12, color=NAVY)
        metric = by_key[(method, world)]
        pane.text(.5, -.09, f"结构节点 {int(metric['predicted_structural_nodes'])}｜出口 F1 {metric['exit_f1']:.3f}\n连通率 {metric['connectivity']:.3f}｜验证边正确率 {metric['verified_edge_correctness']:.3f}", transform=pane.transAxes, ha="center", va="top", fontproperties=FP, fontsize=8, color=GRAY)
    ax.text(0.055, 0.15, "灰点为距离锚点，橙点为识别出的路口/末端结构节点，线为机器人实际走过后才建立的已验证边。", fontproperties=FP, fontsize=10, color=NAVY)
    ax.text(0.055, 0.105, "三张图都保持全局连通和 100% 的已验证边物理轨迹证据；差异主要来自结构节点数量和出口事件的精确率/召回率。", fontproperties=FP_BOLD, fontsize=10.5, color=NAVY)
    save(pdf, fig)


def status_page(pdf):
    fig, ax = new_page("当前结论与下一步", "09  汇报边界", 10)
    columns = [
        (0.055, PALE_GREEN, GREEN, "已经完成", ["正式数据与三次 M1D 训练", "C08 开发参数选择", "C09 15,833 帧几何资格", "C09 推理与 50 次因果建图"]),
        (0.365, PALE_BLUE, BLUE, "已经证明", ["完整流程可执行并封存", "所有图保持连通", "已验证边均有物理轨迹", "M1D seed1 综合分优于 B0"]),
        (0.675, PALE_ORANGE, ORANGE, "尚不能宣称", ["M1D 每个 seed 都优于 B0", "C10 严格未见测试结果", "M-TARE 闭环替换收益", "多机器人探索收益"]),
    ]
    for x, bg, color, title, items in columns:
        rounded_box(ax, (x, 0.31), 0.27, 0.45, facecolor=bg)
        ax.text(x + 0.025, 0.71, title, fontproperties=FP_BOLD, fontsize=16, color=color, va="top")
        for i, item in enumerate(items):
            y = 0.635 - i * 0.075
            ax.text(x + 0.025, y, "●", fontproperties=FP, fontsize=9, color=color, va="center")
            ax.text(x + 0.052, y, item, fontproperties=FP, fontsize=11, color=NAVY, va="center")
    rounded_box(ax, (0.055, 0.12), 0.89, 0.12, facecolor=NAVY)
    ax.text(0.075, 0.205, "现阶段最准确的结论", fontproperties=FP_BOLD, fontsize=11, color="#9FC0FF", va="top")
    ax.text(0.075, 0.158, "C08 开发与 C09 冻结参数回放均已跑通；C09 上 M1D 最佳 seed 优于 B0，但平均收益不均匀。下一项独立研究决策是是否授权 C10 严格测试。", fontproperties=FP_BOLD, fontsize=12.5, color=WHITE, va="top")
    save(pdf, fig)


def main() -> None:
    required = [DATA_IMAGE, MODEL_IMAGE, GEOMETRY_IMAGE, TRAJECTORY_IMAGE, C09_TRAJECTORY_IMAGE, Path(FONT_REGULAR), Path(FONT_BOLD)]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing report inputs: {missing}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUTPUT, metadata={"Title": "地下结构语义与因果拓扑图：阶段实验结果", "Author": "mtare_topo_comm"}) as pdf:
        cover(pdf)
        overview(pdf)
        data_page(pdf)
        model_page(pdf)
        geometry_page(pdf)
        graph_page(pdf)
        results_page(pdf)
        c09_results_page(pdf)
        c09_topology_page(pdf)
        status_page(pdf)
    print(OUTPUT)


if __name__ == "__main__":
    main()
