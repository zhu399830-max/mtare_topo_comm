#!/usr/bin/env python3
"""Visualize only the sealed native capture/finalize artifacts; no model calls.

Outputs are restricted to a new, dedicated docs/figures directory. Original
runs, source manifests, records and tokens are read-only. Explicit wording-only
refreshes require unchanged source hashes and numerical counts. The figures
are descriptive evidence, not another run.
"""
from __future__ import annotations

from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.collections import LineCollection
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CAPTURE = ROOT / "results/gate6_single_robot/gate6_20260913_gse_native_structure_capture_v1_seed11"
CASE = CAPTURE / "artifacts/cases/tunnel_seed11_native_shadow_capture"
TOKENS = ROOT / "results/gate6_single_robot/gate6_20260913_gse_native_structure_finalize_v1_seed11/artifacts/frozen_tokens"
OUTPUT = ROOT / "docs/figures/gse_native_structure_capture_v1"
FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
SUCCESS, REJECT, INK = "#087f70", "#c15034", "#18334e"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path):
    return str(path.relative_to(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-derived-wording", action="store_true",
        help="Refresh only existing derived products after verifying identical inputs and counts")
    args = parser.parse_args()
    previous = None
    if OUTPUT.exists() and not args.refresh_derived_wording:
        raise FileExistsError(f"Refusing to overwrite existing visualization assets: {OUTPUT}")
    if args.refresh_derived_wording:
        previous = json.loads((OUTPUT / "sources_and_claims.json").read_text())
    sources = {}

    def record_source(path):
        sources[relative(path)] = dict(sha256=sha256(path), bytes=path.stat().st_size)

    def read_json(path):
        record_source(path)
        return json.loads(path.read_text())

    def read_jsonl(path):
        record_source(path)
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    trajectory = read_jsonl(CASE / "evidence/trajectory.jsonl")
    coverage = read_jsonl(CASE / "evidence/coverage_curve.jsonl")
    sidecar = read_json(CASE / "planner/advice_sidecar/summary.json")
    registry_path = CASE / "planner/advice_sidecar/native_region_tasks.json"
    registry = read_json(registry_path)
    if sources[relative(registry_path)]["sha256"] != sidecar["native_region_registry_sha256"]:
        raise ValueError("Existing native registry SHA does not match its saved sidecar summary")
    token_summary = read_json(TOKENS / "summary.json")
    token_hashes = read_json(TOKENS / "files_sha256.json")
    expected_records = [TOKENS / f"record_{index:06d}.json" for index in range(120)]
    if sorted(TOKENS.glob("record_*.json")) != expected_records:
        raise ValueError("Expected the exact fixed 120-record population, no selection")
    windows = []
    for index, path in enumerate(expected_records):
        record = read_json(path)
        if sources[relative(path)]["sha256"] != token_hashes[path.name]:
            raise ValueError(f"Saved record SHA mismatch: {path.name}")
        if record["window_id"] != f"second_{index:03d}":
            raise ValueError("Fixed window ordering changed")
        frames = record["source"]["frames"]
        if len(frames) != 5 or any(a["stamp_ns"] >= b["stamp_ns"] for a, b in zip(frames, frames[1:])):
            raise ValueError("Each displayed record must bind exactly five chronological source frames")
        passed = record["status"] == "PROCESSED"
        if not passed and (record["status"] != "REJECTED_RECORD" or not record["reason"].startswith("INCOMPATIBLE_FULL_ROTATION:")):
            raise ValueError("Unexpected result category; do not fold another failure into pose rejection")
        token_info = None
        if passed:
            if record.get("model_forward") is not True:
                raise ValueError("Processed record lacks its saved model execution declaration")
            token_path = TOKENS / record["token_ref"]["path"]
            if token_path.parent != TOKENS:
                raise ValueError("Unexpected token reference outside the authorized artifact directory")
            record_source(token_path)
            observed_hash = sources[relative(token_path)]["sha256"]
            if observed_hash != record["token_ref"]["sha256"] or observed_hash != token_hashes[token_path.name]:
                raise ValueError("Saved token SHA mismatch")
            token_info = dict(path=relative(token_path), sha256=observed_hash)
        elif record.get("model_forward") is not False:
            raise ValueError("Rejected window must not be labelled as a successful model forward")
        windows.append(dict(index=index, window_id=record["window_id"], status=record["status"],
            recorded_producer_reason=record.get("reason"), current_stamp_ns=frames[-1]["stamp_ns"],
            source_frame_keys=[frame["source_key"] for frame in frames],
            record_path=relative(path), record_sha256=sources[relative(path)]["sha256"], token=token_info))
    count = Counter(row["status"] for row in windows)
    if count != {"PROCESSED": 14, "REJECTED_RECORD": 106}:
        raise ValueError(f"Frozen population counts changed: {count}")
    if (token_summary["processed_windows"], token_summary["rejected_windows"], token_summary["recorded_windows"]) != (14, 106, 120):
        raise ValueError("Worker summary and all-record evidence disagree")
    if len(trajectory) != 600 or len(coverage) != 600:
        raise ValueError("Expected all 600 existing synchronized samples")
    elapsed = np.asarray([row["elapsed_sec"] for row in trajectory])
    stamps = np.asarray([row["stamp_sec"] for row in trajectory])
    xyz = np.asarray([row["xyz_m"] for row in trajectory])
    cover_t = np.asarray([row["elapsed_sec"] for row in coverage])
    voxels = np.asarray([row["explored_voxels"] for row in coverage])
    volumes = np.asarray([row["explored_volume_m3"] for row in coverage])
    if not (np.isfinite(xyz).all() and np.all(np.diff(elapsed) > 0) and np.array_equal(elapsed, cover_t)):
        raise ValueError("Unexpected trajectory/coverage synchronization or nonfinite coordinates")
    if not np.all(np.diff(voxels) >= 0) or not np.allclose(volumes, voxels * 0.125, rtol=0, atol=1e-9):
        raise ValueError("Observed voxel counts/0.5-m voxel volumes do not match saved evidence")
    capture_start_sec = float(stamps[0] - elapsed[0])
    if not np.allclose(stamps - elapsed, capture_start_sec, atol=1e-9, rtol=0):
        raise ValueError("Recorded elapsed-time origin changed")
    window_t = np.asarray([row["current_stamp_ns"] * 1e-9 - capture_start_sec for row in windows])
    if np.any(np.diff(window_t) <= 0):
        raise ValueError("Window timestamps are not chronological")
    for row, time_sec in zip(windows, window_t):
        row["capture_elapsed_sec"] = float(time_sec)
    if sidecar["counts"]["identity_fallback"] != sidecar["counts"]["candidates"] or sidecar["counts"]["cache_applied"] != 0:
        raise ValueError("Do not label a modified policy run as an all-identity native trajectory")
    path_metric = float(coverage[-1]["traveling_distance_m"])
    path_polyline = float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())
    counts = dict(independent_worlds=1, world="tunnel", seed=11, independent_trajectories=1,
        trajectory_samples=len(trajectory), coverage_samples=len(coverage),
        recorded_sample_elapsed_range_sec=[float(elapsed[0]), float(elapsed[-1])],
        saved_traveling_distance_m=path_metric, raw_trajectory_polyline_length_m=path_polyline,
        final_observed_voxels=int(voxels[-1]), final_observed_voxel_volume_m3=float(volumes[-1]),
        voxel_edge_m=0.5, native_snapshots=sidecar["counts"]["candidates"],
        identity_advice=sidecar["counts"]["identity_fallback"], native_tasks=len(registry["tasks"]),
        direction_tasks=registry["direction_task_count"], confirmed_traversals=len(registry["confirmed_traversals"]),
        registry_rejected=sidecar["counts"]["registry_rejected"],
        token_windows=120, token_processed=14, pose_rejected=106, token_artifacts_verified=14,
        new_model_forwards_in_visualization=0, new_training_steps=0)
    if not FONT.is_file():
        raise FileNotFoundError("Expected installed Chinese font, no font replacement/download")
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=str(FONT)).get_name(),
        "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
        "axes.spines.top": False, "axes.spines.right": False, "axes.unicode_minus": False,
        "axes.edgecolor": "#b8c5d3", "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK, "pdf.fonttype": 42, "ps.fonttype": 42})
    if previous is not None and (previous["inputs"] != sources or previous["counts"] != counts):
        raise ValueError("Wording-only refresh requires the exact prior source hashes and numerical counts")
    OUTPUT.mkdir(parents=True, exist_ok=previous is not None)
    outputs = []

    def save(fig, name):
        for suffix in ("png", "pdf"):
            path = OUTPUT / f"{name}.{suffix}"
            fig.savefig(path, dpi=180, facecolor="white")
            outputs.append(path)
        plt.close(fig)

    figure = plt.figure(figsize=(14, 9))
    grid = figure.add_gridspec(2, 2, left=.07, right=.97, top=.82, bottom=.14,
        width_ratios=[1.12, 1], hspace=.40, wspace=.29)
    xy_axis = figure.add_subplot(grid[:, 0])
    xz_axis = figure.add_subplot(grid[0, 1])
    coverage_axis = figure.add_subplot(grid[1, 1])
    figure.text(.07, .95, "实际补录成果：原生 M-TARE 轨迹与观测增长", fontsize=21, weight="bold")
    figure.text(.07, .902, f"tunnel · seed 11 · 1 条开发轨迹 · 600 条同步采样 · 原指标路径 {path_metric:.3f} m", fontsize=12)
    figure.text(.07, .86, "113 / 113 次建议均保持原路线；本图不是学习模型路线或对照优势。", color="#6a7785", fontsize=11)
    for axis, second, title in ((xy_axis, 1, "A  实际 XY 轨迹（坐标等比例）"),
                                (xz_axis, 2, "B  实际 XZ 轨迹（纵轴独立缩放）")):
        points = xyz[:, [0, second]]
        segments = np.stack((points[:-1], points[1:]), axis=1)
        line = LineCollection(segments, cmap="viridis", norm=plt.Normalize(0, 120), linewidth=2.5)
        line.set_array(elapsed[:-1])
        axis.add_collection(line)
        axis.autoscale()
        axis.scatter(*points[0], color="white", edgecolor=INK, s=68, linewidth=1.8, zorder=4, label="起点")
        axis.scatter(*points[-1], color=INK, marker="s", s=52, zorder=4, label="末点")
        axis.set(title=title, xlabel="X（m）", ylabel=f"{'Y' if second == 1 else 'Z'}（m）")
        axis.grid(alpha=.25)
    xy_axis.set_aspect("equal", adjustable="box")
    xy_axis.legend(loc="best", fontsize=10)
    colorbar = figure.colorbar(line, ax=xy_axis, location="bottom", fraction=.045, pad=.11, aspect=30)
    colorbar.set_label("实际记录时间（s）；线色只表示时间")
    coverage_axis.plot(cover_t, voxels, color=SUCCESS, linewidth=2.4)
    coverage_axis.fill_between(cover_t, voxels, color=SUCCESS, alpha=.08)
    coverage_axis.set(title="C  累计观测体素（不是完整地图覆盖率）", xlabel="实际记录时间（s）",
        ylabel="累计观测体素数（个）", xlim=(0, 120), ylim=(0, float(voxels[-1]) * 1.22))
    coverage_axis.grid(alpha=.25)
    coverage_axis.text(.96, .92, f"{voxels[-1]:,} 个体素\n{volumes[-1]:,.2f} m³ 体素体积", transform=coverage_axis.transAxes,
        ha="right", va="top", fontsize=12, color=SUCCESS)
    figure.text(.07, .064, "边界：0.5 m 观测体素代理，不代表可通行自由空间、整图覆盖百分比或完整探索完成。", fontsize=11)
    figure.text(.07, .032, "来源：capture_v1 已保存 trajectory / coverage_curve；仅重新绘图，未仿真、训练或调用模型。", fontsize=10, color="#6a7785")
    save(figure, "actual_execution")

    processed = np.array([row["status"] == "PROCESSED" for row in windows])
    figure, axis = plt.subplots(figsize=(14, 6))
    figure.subplots_adjust(left=.12, right=.97, top=.70, bottom=.31)
    figure.text(.07, .92, "冻结 token 的真实结果：全部 120 个固定时间窗", fontsize=21, weight="bold")
    figure.text(.07, .838, "14 个完成 token 提取", color=SUCCESS, fontsize=18, weight="bold")
    figure.text(.42, .838, "106 个因完整旋转不兼容被拒绝", color=REJECT, fontsize=18, weight="bold")
    figure.text(.07, .776, "每点对应一个原五帧时间窗；按真实当前帧时间排列，不筛选成功样本。", fontsize=11, color="#6a7785")
    axis.scatter(window_t[processed], np.ones(processed.sum()), s=46, color=SUCCESS, marker="o", zorder=3)
    axis.scatter(window_t[~processed], np.zeros((~processed).sum()), s=36, color=REJECT, marker="x", linewidth=1.5, zorder=3)
    axis.set(xlim=(-1, 121), ylim=(-.45, 1.45), xlabel="当前帧相对补录起点时间（s）",
        yticks=[0, 1], yticklabels=["姿态拒绝", "token 已保存"])
    axis.set_xticks(np.arange(0, 121, 10))
    axis.grid(axis="x", alpha=.23)
    axis.axhline(0, color=REJECT, alpha=.13)
    axis.axhline(1, color=SUCCESS, alpha=.13)
    figure.text(.07, .188, "拒绝原因：本轮冻结接入层只传水平转动；三维姿态不兼容时保留拒绝，未压平姿态。", fontsize=11)
    figure.text(.07, .129, "成功仅指特征产物存在且 SHA 已核验，不代表方向识别、配准成立、任务对应正确或学习接管控制。", fontsize=11)
    figure.text(.07, .069, "两组来源分别绘制：120 个模型窗 ≠ 113 个 native 五帧快照；本图未做 epoch 回填或伪造路线关联。", fontsize=11)
    figure.text(.07, .030, "来源：finalize_v1 的 120 份 record 与 14 份已有 token；颜色仅表示执行状态，不是语义类别。", fontsize=10, color="#6a7785")
    save(figure, "token_window_outcomes")

    # Recheck all consumed source bytes after plotting; no sealed artifact may change.
    for path, description in sources.items():
        if sha256(ROOT / path) != description["sha256"]:
            raise ValueError(f"Input changed while drawing: {path}")
    provenance = dict(schema_version="native_structure_capture_visualization_v1",
        purpose="read_only_visualization_of_existing_capture_and_finalize_evidence",
        script=dict(path=relative(Path(__file__)), sha256=sha256(Path(__file__))),
        counts=counts, windows=windows,
        rejection_cause_explanation="本轮 FrozenDualPathEncoderAdapterV1 冻结接入层只传水平转动；三维姿态不兼容时保留拒绝，未压平姿态。底层 register/_memory 已支持完整旋转，不能将接入层漏传归因为底层编码器缺少 fullR 能力。",
        recorded_producer_reason_note="windows.recorded_producer_reason 原样保留 sealed record 中的历史错误串；该措辞不作为底层编码器能力边界的判定。",
        native_execution_state_counts=dict(Counter(row["execution_state"] for row in registry["tasks"])),
        claim_boundaries=["Native M-TARE controlled this recorded trajectory; all 113 advice messages were identity fallback.",
            "Observed voxel volume is not complete-map coverage percentage, free space or completion proof.",
            "14 token outputs do not establish directional semantics, registration, identity or planning benefit.",
            "106 windows were rejected by this frozen adapter's yaw-only handoff; the underlying register/_memory supports full rotation. No pose flattening or new forwards.",
            "120 model windows are not treated as the 113 native snapshots; no correspondence is fabricated.",
            "Native grid regions and actual pose anchors are not inferred channel branches or verified traversal edges.",
            "Original capture failure, finalized outputs and sealed run evidence were not modified."],
        point_cloud_visualization=False, source_bytes_unchanged_after_plot=True,
        inputs=sources, products={relative(path): dict(sha256=sha256(path), bytes=path.stat().st_size) for path in outputs})
    manifest = OUTPUT / "sources_and_claims.json"
    with manifest.open("w" if previous is not None else "x") as stream:
        json.dump(provenance, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(output_directory=relative(OUTPUT), figures=[relative(path) for path in outputs],
        provenance=relative(manifest), sources_verified=len(sources), counts=counts), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
