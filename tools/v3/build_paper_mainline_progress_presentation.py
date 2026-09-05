#!/home/zeng-workstation/anaconda3/bin/python
"""Build a Chinese paper-mainline progress report as a slide-style PDF.

The deck intentionally avoids internal Gate terminology and reports only sealed
results plus an explicitly labelled interim view of the active 30-case run.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "docs/presentation_assets/paper_mainline_progress"
PDF_PATH = ROOT / "docs/STRUCTURAL_TOPOLOGY_PAPER_MAINLINE_PROGRESS.pdf"
FONT_REGULAR = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"

W, H = 1920, 1080
BG = "#F7F8FA"
NAVY = "#102A43"
TEXT = "#243B53"
MUTED = "#627D98"
TEAL = "#0E8A8A"
TEAL_LIGHT = "#DDF4F2"
BLUE = "#3B82F6"
BLUE_LIGHT = "#E8F1FF"
ORANGE = "#F59E0B"
ORANGE_LIGHT = "#FFF2D8"
RED = "#D64545"
RED_LIGHT = "#FDE8E7"
GREEN = "#2F9E62"
GREEN_LIGHT = "#E4F5EB"
WHITE = "#FFFFFF"
LINE = "#D9E2EC"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REGULAR, size)


def wrap(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont, width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        if not paragraph:
            lines.append("")
            continue
        current = ""
        for ch in paragraph:
            candidate = current + ch
            if current and draw.textbbox((0, 0), candidate, font=fnt)[2] > width:
                lines.append(current.rstrip())
                current = ch.lstrip()
            else:
                current = candidate
        if current:
            lines.append(current.rstrip())
    return lines


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    fnt: ImageFont.FreeTypeFont,
    fill: str,
    width: int,
    line_gap: int = 10,
    max_lines: int | None = None,
) -> int:
    lines = wrap(draw, text, fnt, width)
    if max_lines is not None:
        lines = lines[:max_lines]
    x, y = xy
    step = fnt.size + line_gap
    for line in lines:
        draw.text((x, y), line, font=fnt, fill=fill)
        y += step
    return y


def canvas(title: str, kicker: str, page: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((66, 42, 358, 88), radius=20, fill=TEAL_LIGHT)
    d.text((88, 50), kicker, font=font(24, True), fill=TEAL)
    d.text((68, 116), title, font=font(54, True), fill=NAVY)
    d.line((68, 193, 1852, 193), fill=LINE, width=3)
    d.text((68, 1030), "学习式局部出口语义拓扑探索 · 论文主线阶段总结", font=font(20), fill=MUTED)
    d.text((1785, 1030), f"{page:02d}", font=font(20, True), fill=MUTED)
    return img, d


def card(d: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str, body: str,
         accent: str = TEAL, fill: str = WHITE, title_size: int = 31, body_size: int = 25) -> None:
    x1, y1, x2, y2 = box
    d.rounded_rectangle(box, radius=24, fill=fill, outline=LINE, width=2)
    d.rounded_rectangle((x1, y1, x1 + 12, y2), radius=6, fill=accent)
    d.text((x1 + 36, y1 + 28), title, font=font(title_size, True), fill=NAVY)
    draw_wrapped(d, (x1 + 36, y1 + 86), body, font(body_size), TEXT, x2 - x1 - 70, 11)


def metric(d: ImageDraw.ImageDraw, box: tuple[int, int, int, int], value: str, label: str,
           color: str = TEAL, fill: str = WHITE) -> None:
    x1, y1, x2, y2 = box
    d.rounded_rectangle(box, radius=22, fill=fill, outline=LINE, width=2)
    d.text((x1 + 28, y1 + 28), value, font=font(48, True), fill=color)
    draw_wrapped(d, (x1 + 28, y1 + 94), label, font(23), TEXT, x2 - x1 - 56, 8)


def bullets(d: ImageDraw.ImageDraw, xy: tuple[int, int], items: Iterable[str], width: int,
            size: int = 29, gap: int = 26, color: str = TEXT, bullet_color: str = TEAL) -> int:
    x, y = xy
    fnt = font(size)
    for item in items:
        d.ellipse((x, y + 13, x + 15, y + 28), fill=bullet_color)
        lines = wrap(d, item, fnt, width - 40)
        for i, line in enumerate(lines):
            d.text((x + 34, y + i * (size + 10)), line, font=fnt, fill=color)
        y += max(1, len(lines)) * (size + 10) + gap
    return y


def paste_contain(base: Image.Image, source: Path, box: tuple[int, int, int, int],
                  background: str = WHITE) -> None:
    x1, y1, x2, y2 = box
    panel = Image.new("RGB", (x2 - x1, y2 - y1), background)
    im = Image.open(source).convert("RGB")
    scale = min(panel.width / im.width, panel.height / im.height)
    resized = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.Resampling.LANCZOS)
    panel.paste(resized, ((panel.width - resized.width) // 2, (panel.height - resized.height) // 2))
    base.paste(panel, (x1, y1))


def add_arrow(d: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], color: str = TEAL) -> None:
    d.line((*start, *end), fill=color, width=8)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    length = 24
    for delta in (2.55, -2.55):
        p = (end[0] + length * math.cos(angle + delta), end[1] + length * math.sin(angle + delta))
        d.line((*end, *p), fill=color, width=8)


def load_current_cases() -> list[dict]:
    run = ROOT / "results/gate6_single_robot/gate6_20260823_aee_composite_v9_combined_correction_stochastic_v1r2_seed20260820"
    rows = []
    for path in sorted((run / "artifacts/cases").glob("*/summary.json")):
        item = json.loads(path.read_text())
        if item.get("status") != "PASS_SINGLE_ROBOT_CASE_V2":
            continue
        case, metrics = item["case"], item["metrics"]
        audit = item["combined_correction_mechanism_audit"]
        rows.append({
            "case_id": case["case_id"], "world": case["world"],
            "environment_seed": case["environment_seed"], "checkpoint_seed": case["checkpoint_seed"],
            "coverage": metrics["final_explored_volume_m3"],
            "travel": metrics["traveling_distance_m"],
            "efficiency": metrics["final_volume_per_travel_meter_m2"],
            "reanchors": audit["verified_reanchor_count"],
            "rejections": audit["frontier_execution_rejection_count"],
        })
    return rows


def build_slides(rows: list[dict]) -> list[Image.Image]:
    slides: list[Image.Image] = []

    # 1. Cover
    img = Image.new("RGB", (W, H), NAVY)
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((84, 88, 420, 142), radius=24, fill=TEAL)
    d.text((112, 98), "论文主线阶段汇报", font=font(27, True), fill=WHITE)
    d.text((84, 242), "从局部出口语义到在线拓扑探索", font=font(70, True), fill=WHITE)
    d.text((84, 344), "学习哪里可能有路，再用因果图决定下一步", font=font(48), fill="#C9E8E6")
    d.rounded_rectangle((84, 502, 1836, 810), radius=34, fill="#173F5F")
    bullets(d, (132, 548), [
        "从 LiDAR 学习局部可通出口方向，而不是预测整张地图或探索价值",
        "以真实运动轨迹验证节点和边，用执行反馈维护未探索出口",
        "只替换全局表示与目标选择，保留 M-TARE 的局部规划和控制",
    ], 1600, size=34, gap=24, color=WHITE, bullet_color="#7BD4CC")
    d.text((84, 960), f"正式闭环进展：{len(rows)}/30 组完成 · 2026-08-23", font=font(29, True), fill="#A8DADC")
    slides.append(img)

    # 2. Research problem
    img, d = canvas("论文要回答的核心问题", "01 · 研究问题", 2)
    d.text((70, 230), "地下探索真正困难的不是“看见一条路”，而是长期保持正确的全局状态。", font=font(34, True), fill=TEXT)
    card(d, (70, 322, 620, 820), "感知别名", "长走廊外观相似，单帧几何很难区分“新位置”和“回到旧节点”。", ORANGE, ORANGE_LIGHT)
    card(d, (685, 322, 1235, 820), "拓扑状态", "看见出口不等于已经走过；必须区分待探索方向、真实连接边和失败尝试。", TEAL, TEAL_LIGHT)
    card(d, (1300, 322, 1850, 820), "公平比较", "如果同时更换局部规划、控制或地图，就无法判断收益来自出口语义和图状态还是其他模块。", BLUE, BLUE_LIGHT)
    d.rounded_rectangle((180, 865, 1740, 970), radius=26, fill=NAVY)
    d.text((250, 892), "研究假设：更紧凑、可解释、执行一致的结构拓扑图，能改善全局探索决策。", font=font(34, True), fill=WHITE)
    slides.append(img)

    # 3. Method pipeline
    img, d = canvas("核心方法：观测、记忆、决策、反馈形成闭环", "02 · 方法总览", 3)
    labels = [
        ("实时 LiDAR", "16线全周距离"),
        ("局部结构输入", "学习出口方向 + 几何数量/角色"),
        ("因果拓扑图", "节点 + 验证边 + 出口"),
        ("全局目标", "图前沿 + 最短路径"),
        ("原局部栈", "避障 + 跟踪 + 控制"),
    ]
    x_positions = [70, 430, 790, 1150, 1510]
    for i, ((title, body), x) in enumerate(zip(labels, x_positions)):
        fill = TEAL_LIGHT if i in (1, 2, 3) else BLUE_LIGHT
        accent = TEAL if i in (1, 2, 3) else BLUE
        d.rounded_rectangle((x, 330, x + 300, 600), radius=28, fill=fill, outline=accent, width=4)
        d.text((x + 28, 375), title, font=font(34, True), fill=NAVY)
        draw_wrapped(d, (x + 28, 455), body, font(25), TEXT, 244, 10)
        if i < len(labels) - 1:
            add_arrow(d, (x + 305, 465), (x_positions[i + 1] - 12, 465), accent)
    add_arrow(d, (1660, 640), (935, 785), ORANGE)
    add_arrow(d, (935, 785), (935, 620), ORANGE)
    d.rounded_rectangle((550, 740, 1320, 920), radius=26, fill=ORANGE_LIGHT, outline=ORANGE, width=3)
    d.text((602, 777), "执行反馈", font=font(34, True), fill=NAVY)
    d.text((602, 842), "真实抵达旧节点 → 回锚；走错出口/绕回原点 → 记录失败并重选", font=font(27), fill=TEXT)
    slides.append(img)

    # 4. Learning
    img, d = canvas("学习式局部出口语义：从 LiDAR 提取可通方向", "03 · 学习模型", 4)
    sample = ROOT / "results/gate1_data/gate1_20260812_cano_phase2_supervised_range_dataset_v2r_seed0/previews/train_complete_samples/page_06_S06.png"
    d.rounded_rectangle((70, 245, 1050, 870), radius=24, fill=WHITE, outline=LINE, width=2)
    paste_contain(img, sample, (92, 270, 1028, 820))
    d.text((102, 830), "实际训练样本：彩色为 LiDAR 距离，白线为离线出口标签", font=font(22), fill=MUTED)
    metric(d, (1100, 260, 1840, 425), "0.843–0.848", "三个冻结模型的出口方向 F1", TEAL, TEAL_LIGHT)
    metric(d, (1100, 458, 1460, 650), "0.775", "分支数量 F1", BLUE, BLUE_LIGHT)
    metric(d, (1480, 458, 1840, 650), "0.723", "结构角色 F1", ORANGE, ORANGE_LIGHT)
    card(d, (1100, 685, 1840, 900), "边界说清楚", "学习分支负责出口方向；数量和角色仍由冻结几何分支提供。我们不宣称端到端理解完整地下地图。", RED, RED_LIGHT, 29, 23)
    slides.append(img)

    # 5. Causal graph
    img, d = canvas("在线拓扑图：边必须由真实运动验证", "04 · 因果建图", 5)
    graph_img = ROOT / "results/gate4_topology/gate4_20260817_cano_c08_route_conditioned_causal_topology_replay_v2r_seed0/previews/S10_3d_complex_C08_objective_role_trajectory.png"
    d.rounded_rectangle((70, 240, 1190, 905), radius=24, fill=WHITE, outline=LINE, width=2)
    paste_contain(img, graph_img, (88, 258, 1172, 887))
    bullets(d, (1245, 275), [
        "稳定的结构事件形成节点",
        "机器人真实走过后才建立连接边",
        "看见但未走过的分支保留为待探索出口",
        "回到旧位置时合并，而不是重复建图",
        "完整地图只用于离线评分，不参与在线更新",
    ], 590, size=28, gap=25)
    slides.append(img)

    # 6. Controlled replacement
    img, d = canvas("公平替换：只改变全局层", "05 · 与 M-TARE 的接口", 6)
    d.rounded_rectangle((80, 255, 900, 820), radius=30, fill=WHITE, outline=LINE, width=3)
    d.text((125, 292), "原版 M-TARE", font=font(39, True), fill=NAVY)
    card(d, (130, 385, 850, 535), "全局层", "粗粒度度量地图 + 原全局目标选择", BLUE, BLUE_LIGHT, 29, 23)
    card(d, (130, 580, 850, 730), "局部执行层", "局部规划、避障、跟踪、控制", TEAL, TEAL_LIGHT, 29, 23)
    d.rounded_rectangle((1020, 255, 1840, 820), radius=30, fill=WHITE, outline=TEAL, width=5)
    d.text((1065, 292), "我们的方法", font=font(39, True), fill=NAVY)
    card(d, (1070, 385, 1790, 535), "全局层（替换）", "出口语义驱动的因果拓扑图 + 图前沿目标选择", ORANGE, ORANGE_LIGHT, 29, 23)
    card(d, (1070, 580, 1790, 730), "局部执行层（不变）", "同一套局部规划、避障、跟踪、控制", TEAL, TEAL_LIGHT, 29, 23)
    d.rounded_rectangle((220, 865, 1700, 955), radius=22, fill=NAVY)
    d.text((296, 889), "因此闭环差异主要来自：全局表示是否更紧凑，以及拓扑状态是否与真实执行一致。", font=font(31, True), fill=WHITE)
    slides.append(img)

    # 7. Offline evidence
    img, d = canvas("离线因果回放：学习式出口方向改善图表达", "06 · 早期证据", 7)
    metric(d, (75, 270, 475, 470), "4,773", "唯一 LiDAR 帧", TEAL, TEAL_LIGHT)
    metric(d, (510, 270, 910, 470), "14,319", "冻结模型推理帧", BLUE, BLUE_LIGHT)
    metric(d, (945, 270, 1345, 470), "3,645", "固定参数图回放", ORANGE, ORANGE_LIGHT)
    metric(d, (1380, 270, 1845, 470), "1.000", "连通率与验证边正确率", GREEN, GREEN_LIGHT)
    d.text((78, 560), "学习出口方向 vs. 单帧几何出口规则", font=font(33, True), fill=NAVY)
    # two horizontal comparisons
    for y, label, b0, m1d in [(665, "拓扑综合分", 0.744, 0.826), (790, "出口 F1", 0.498, 0.655)]:
        d.text((80, y), label, font=font(28, True), fill=TEXT)
        x0, maxw = 380, 1180
        d.rounded_rectangle((x0, y + 2, x0 + int(maxw * b0), y + 42), radius=18, fill="#AFC8E8")
        d.rounded_rectangle((x0, y + 52, x0 + int(maxw * m1d), y + 92), radius=18, fill=TEAL)
        d.text((x0 + int(maxw * b0) + 16, y - 2), f"几何规则 {b0:.3f}", font=font(23), fill=MUTED)
        d.text((x0 + int(maxw * m1d) + 16, y + 47), f"学习语义 {m1d:.3f}", font=font(23, True), fill=TEAL)
    d.text((80, 935), "说明：这是开发回放中的图表达证据，不等同于闭环探索优于 M-TARE。", font=font(24), fill=RED)
    slides.append(img)

    # 8. Failure diagnosis
    img, d = canvas("旧图规划器为什么失败：不是模型没输出，而是图状态没跟上", "07 · 问题定位", 8)
    metric(d, (75, 255, 570, 480), "−70.7%", "旧图规划器相对原版 M-TARE 的覆盖时间积分", RED, RED_LIGHT)
    metric(d, (605, 255, 1100, 480), "47,123", "返回旧节点但图状态未同步的帧", ORANGE, ORANGE_LIGHT)
    metric(d, (1135, 255, 1630, 480), "497", "实际离开方向与目标出口不一致事件", BLUE, BLUE_LIGHT)
    d.rounded_rectangle((1660, 255, 1845, 480), radius=22, fill=NAVY)
    d.text((1692, 285), "10/10", font=font(42, True), fill=WHITE)
    d.text((1690, 355), "区组均差", font=font(23, True), fill="#C9E8E6")
    card(d, (80, 570, 890, 875), "问题一：回到旧节点却没回锚", "机器人已经沿验证边返回，但 current node 仍停留在旧身份，随后全局目标从错误位置继续计算。", ORANGE, ORANGE_LIGHT)
    card(d, (1010, 570, 1820, 875), "问题二：走错出口仍重复选择", "真实离开方向与目标不一致、甚至绕回同一节点时，目标出口没有及时退出候选集合。", BLUE, BLUE_LIGHT)
    d.text((80, 930), "统计证据：十区组精确符号翻转 p=0.00195；坏结果被完整保留，作为因果修正基线。", font=font(24), fill=MUTED)
    slides.append(img)

    # 9. Corrections
    img, d = canvas("组合修正：让图状态服从真实执行", "08 · 方法改进", 9)
    card(d, (80, 270, 900, 690), "修正 A · 可靠回锚", "当机器人沿已经验证的边返回旧节点，并满足冻结的轨迹与距离证据时，确定性更新当前节点身份。\n\n作用：避免“机器人回来了，但图还以为它在别处”。", TEAL, TEAL_LIGHT, 36, 28)
    card(d, (1020, 270, 1840, 690), "修正 B · 出口执行反馈", "当实际离开方向不匹配目标出口，或机器人绕回同一节点时，把失败写入既有重试记录并重新选择。\n\n作用：避免同一错误出口被长期重复选择。", ORANGE, ORANGE_LIGHT, 36, 28)
    d.rounded_rectangle((210, 775, 1710, 930), radius=28, fill=NAVY)
    d.text((275, 808), "没有新增学习参数 · 没有读取完整地图 · 没有降低评价门槛", font=font(35, True), fill=WHITE)
    d.text((405, 872), "修正的是在线图状态机，而不是事后挑选更好的模型或案例。", font=font(29), fill="#C9E8E6")
    slides.append(img)

    # 10. Experiment design
    img, d = canvas("闭环实验如何保证公平", "09 · 实验设计", 10)
    metric(d, (80, 255, 470, 450), "2", "地下世界：隧道 / 车库", TEAL, TEAL_LIGHT)
    metric(d, (505, 255, 895, 450), "5", "环境随机种子", BLUE, BLUE_LIGHT)
    metric(d, (930, 255, 1320, 450), "3", "冻结模型 checkpoint", ORANGE, ORANGE_LIGHT)
    metric(d, (1355, 255, 1845, 450), "600 s", "每个案例相同预算", GREEN, GREEN_LIGHT)
    d.text((82, 545), "主比较", font=font(31, True), fill=NAVY)
    card(d, (80, 600, 500, 860), "原版 M-TARE", "论文性能主基线", BLUE, BLUE_LIGHT, 29, 23)
    card(d, (530, 600, 950, 860), "修正版拓扑图", "论文主方法", TEAL, TEAL_LIGHT, 29, 23)
    card(d, (980, 600, 1400, 860), "旧缺陷版本", "解释修正来源", ORANGE, ORANGE_LIGHT, 29, 23)
    card(d, (1430, 600, 1850, 860), "完整地图诊断", "只作诊断，不称上界", RED, RED_LIGHT, 29, 23)
    d.text((82, 920), "统计单位是十个“世界×环境种子”区组；checkpoint 不是额外独立世界。", font=font(26), fill=MUTED)
    slides.append(img)

    # 11. Interim results
    n = len(rows)
    img, d = canvas(f"当前闭环结果：{n}/30 组完成，效果存在明显场景差异", "10 · 正式实验进展", 11)
    left, top, right, bottom = 95, 315, 1835, 785
    d.line((left, bottom, right, bottom), fill=LINE, width=3)
    max_cov = max(r["coverage"] for r in rows) if rows else 1.0
    bw = max(18, int((right - left) / max(1, len(rows)) * 0.68))
    step = (right - left) / max(1, len(rows))
    for i, row in enumerate(rows):
        x = left + i * step + (step - bw) / 2
        h = (bottom - top) * row["coverage"] / max_cov
        color = BLUE if row["world"] == "garage" else ORANGE
        d.rounded_rectangle((int(x), int(bottom - h), int(x + bw), bottom), radius=7, fill=color)
        label = ("G" if row["world"] == "garage" else "T") + str(row["environment_seed"])
        d.text((int(x) - 3, bottom + 15), label, font=font(17, True), fill=MUTED)
    for frac in (0.25, 0.5, 0.75, 1.0):
        y = bottom - (bottom - top) * frac
        d.line((left, y, right, y), fill="#E7EDF3", width=2)
        d.text((25, y - 12), f"{max_cov*frac:.0f}", font=font(18), fill=MUTED)
    d.text((95, 260), "最终覆盖体积（m³）", font=font(26, True), fill=TEXT)
    d.rectangle((1450, 245, 1480, 275), fill=BLUE); d.text((1495, 243), "车库", font=font(22), fill=TEXT)
    d.rectangle((1600, 245, 1630, 275), fill=ORANGE); d.text((1645, 243), "隧道", font=font(22), fill=TEXT)
    if rows:
        cov_min, cov_max = min(r["coverage"] for r in rows), max(r["coverage"] for r in rows)
        total_reanchor = sum(r["reanchors"] for r in rows)
        total_reject = sum(r["rejections"] for r in rows)
        metric(d, (110, 845, 520, 985), f"{cov_min:.0f}–{cov_max:.0f}", "当前覆盖范围（m³）", TEAL, TEAL_LIGHT)
        metric(d, (560, 845, 970, 985), str(total_reanchor), "可靠回锚累计触发", BLUE, BLUE_LIGHT)
        metric(d, (1010, 845, 1420, 985), str(total_reject), "错误出口拒绝累计触发", ORANGE, ORANGE_LIGHT)
        metric(d, (1460, 845, 1840, 985), f"{n}/30", "完整归档案例", GREEN, GREEN_LIGHT)
    slides.append(img)

    # 12. Interpretation
    img, d = canvas("目前已经证明什么，还没有证明什么", "11 · 阶段结论", 12)
    d.rounded_rectangle((80, 255, 900, 900), radius=30, fill=GREEN_LIGHT, outline=GREEN, width=3)
    d.text((130, 300), "已经证明", font=font(41, True), fill=GREEN)
    bullets(d, (135, 400), [
        "局部出口方向模型已经训练并冻结",
        "拓扑图能够按真实观测在线建立",
        "已经替换 M-TARE 全局目标层并驱动车辆",
        "两类旧图状态故障已被执行反馈修正",
        "坏结果和工程失败均被完整保留",
    ], 690, size=28, gap=22, bullet_color=GREEN)
    d.rounded_rectangle((1020, 255, 1840, 900), radius=30, fill=RED_LIGHT, outline=RED, width=3)
    d.text((1070, 300), "尚未证明", font=font(41, True), fill=RED)
    bullets(d, (1075, 400), [
        "整体探索效率已经优于原版 M-TARE",
        "在所有隧道结构中都能减少重复绕行",
        "多机器人共享图一定带来收益",
        "开发结果能够直接代表严格未见世界",
        "当前方法达到端到端结构理解",
    ], 690, size=28, gap=22, bullet_color=RED)
    slides.append(img)

    # 13. Roadmap and contribution
    img, d = canvas("从当前实验到论文完成", "12 · 下一步与论文贡献", 13)
    steps = [
        ("完成 30/30", "封存剩余闭环案例", TEAL),
        ("配对统计", "效果量、置信区间、失败场景", BLUE),
        ("多机器人", "共享图、冲突、通信与分配", ORANGE),
        ("严格测试", "冻结参数后一次读取未见世界", GREEN),
        ("论文交付", "图表、正文、补充材料与复现", RED),
    ]
    xs = [75, 430, 785, 1140, 1495]
    for i, ((name, body, color), x) in enumerate(zip(steps, xs)):
        d.ellipse((x + 102, 285, x + 178, 361), fill=color)
        d.text((x + 127, 300), str(i + 1), font=font(28, True), fill=WHITE)
        if i < len(steps) - 1:
            add_arrow(d, (x + 185, 323), (xs[i + 1] + 94, 323), LINE)
        d.rounded_rectangle((x, 405, x + 300, 640), radius=24, fill=WHITE, outline=color, width=3)
        d.text((x + 28, 442), name, font=font(31, True), fill=NAVY)
        draw_wrapped(d, (x + 28, 510), body, font(23), TEXT, 244, 9)
    d.rounded_rectangle((150, 735, 1770, 940), radius=34, fill=NAVY)
    d.text((215, 775), "论文核心贡献", font=font(36, True), fill="#7BD4CC")
    d.text((215, 842), "把学习式局部出口方向变成可执行、可纠错、可验证的因果拓扑状态，", font=font(31, True), fill=WHITE)
    d.text((215, 892), "并在不改变局部运动栈的条件下检验它是否值得替换 M-TARE 全局层。", font=font(31, True), fill=WHITE)
    slides.append(img)

    return slides


def save_outputs(slides: list[Image.Image]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    slide_paths = []
    for index, slide in enumerate(slides, 1):
        path = OUT_DIR / f"slide_{index:02d}.png"
        slide.save(path, quality=95)
        slide_paths.append(path)

    slides[0].save(
        PDF_PATH, "PDF", resolution=150.0, save_all=True,
        append_images=slides[1:], quality=92,
    )

    manifest = {
        "schema_version": "paper_mainline_progress_presentation_v1",
        "slide_count": len(slides),
        "completed_closed_loop_case_count_at_render": len(load_current_cases()),
        "pdf": PDF_PATH.relative_to(ROOT).as_posix(),
        "slide_images": [p.relative_to(ROOT).as_posix() for p in slide_paths],
        "claim_boundary": "Interim paper-mainline progress deck; final closed-loop superiority awaits the sealed 30-case paired analysis.",
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


def main() -> int:
    rows = load_current_cases()
    if not rows:
        raise RuntimeError("no completed corrected closed-loop cases")
    slides = build_slides(rows)
    save_outputs(slides)
    print(PDF_PATH.relative_to(ROOT))
    print(f"slides={len(slides)} cases={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
