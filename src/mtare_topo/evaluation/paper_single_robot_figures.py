"""Deterministic paper figures for the sealed Gate-6 single-robot analysis."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mtare_topo.evaluation.stochastic_closed_loop import METRICS


FAMILY_LABELS = {
    "original_mtare": "Original M-TARE",
    "m1d_topology": "V5 structural topology",
    "layered_gt_map_oracle": "GT-map diagnostic",
}
FAMILY_COLORS = {
    "original_mtare": "#4C566A",
    "m1d_topology": "#0072B2",
    "layered_gt_map_oracle": "#CC79A7",
}
PANEL_METRICS = (
    ("coverage_time_auc_m3_s", "Coverage-time AUC", "m³·s"),
    ("final_explored_volume_m3", "Final explored volume", "m³"),
    ("final_volume_per_travel_meter_m2", "Volume per travel", "m²"),
    ("planner_runtime_p95_sec", "Planner latency p95", "s"),
)
METRIC_LABELS = {
    "coverage_time_auc_m3_s": "Coverage-time AUC",
    "final_explored_volume_m3": "Final volume",
    "mean_explored_volume_m3": "Mean volume",
    "traveling_distance_m": "Travel distance",
    "cumulative_point_redundancy": "Point redundancy",
    "final_volume_per_travel_meter_m2": "Volume / travel",
    "planner_runtime_p95_sec": "Planner latency p95",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite paper figure value: {name}")
    return result


def _save(figure: Any, stem: Path) -> list[Path]:
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    figure.savefig(
        png,
        dpi=220,
        bbox_inches="tight",
        metadata={"Software": "mtare_topo Gate-6 deterministic renderer"},
    )
    figure.savefig(
        pdf,
        bbox_inches="tight",
        metadata={
            "Creator": "mtare_topo Gate-6 deterministic renderer",
            "Producer": "matplotlib",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(figure)
    return [png, pdf]


def _block_performance(analysis: Mapping[str, Any], output_dir: Path) -> list[Path]:
    main = analysis["corrected_main_analysis"]
    records = main.get("block_records")
    if not isinstance(records, list) or len(records) != 10:
        raise ValueError("paper block figure requires exactly ten block records")
    block_ids = [str(record["block_id"]) for record in records]
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.7))
    for axis, (metric, title, unit) in zip(axes.ravel(), PANEL_METRICS):
        for family in FAMILY_LABELS:
            values = [
                _finite(
                    record["families"][family]["metric_means"][metric],
                    name=f"{record['block_id']}:{family}:{metric}",
                )
                for record in records
            ]
            axis.plot(
                range(len(records)), values, marker="o", markersize=4.5,
                linewidth=1.4, color=FAMILY_COLORS[family], label=FAMILY_LABELS[family],
            )
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.set_xticks(range(len(records)), block_ids, rotation=38, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=0.25)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    figure.subplots_adjust(top=0.84, bottom=0.13, hspace=0.52, wspace=0.22)
    figure.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.925),
        ncol=3, frameon=False,
    )
    figure.suptitle(
        "Single-robot randomized blocks (three repeats/checkpoints averaged)", y=0.985
    )
    return _save(figure, output_dir / "single_robot_block_performance")


def _relative_effect_rows(
    metrics: Mapping[str, Any], baseline_means: Mapping[str, float], *, prefix: str
) -> tuple[list[str], list[float], list[float], list[float]]:
    labels, means, lower, upper = [], [], [], []
    for metric in METRICS:
        value = metrics[metric]
        baseline = _finite(baseline_means[metric], name=f"{prefix}:{metric}:baseline")
        if baseline == 0.0:
            raise ValueError(f"zero paper figure baseline: {prefix}:{metric}")
        mean = _finite(value["mean_difference"], name=f"{prefix}:{metric}:mean") / baseline
        ci = value["paired_bootstrap_95_ci"]
        lo = _finite(ci[0], name=f"{prefix}:{metric}:ci0") / baseline
        hi = _finite(ci[1], name=f"{prefix}:{metric}:ci1") / baseline
        labels.append(METRIC_LABELS[metric])
        means.append(mean * 100.0)
        lower.append((mean - lo) * 100.0)
        upper.append((hi - mean) * 100.0)
    return labels, means, lower, upper


def _effect_figure(
    labels: list[str], means: list[float], lower: list[float], upper: list[float],
    *, title: str, output: Path, color: str,
) -> list[Path]:
    figure, axis = plt.subplots(figsize=(9.0, 5.7), constrained_layout=True)
    positions = list(range(len(labels)))
    axis.errorbar(
        means, positions, xerr=[lower, upper], fmt="o", markersize=6,
        color=color, ecolor=color, elinewidth=1.6, capsize=3,
    )
    axis.axvline(0.0, color="#222222", linewidth=1.0, linestyle="--")
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlabel("Relative paired difference (%)")
    axis.set_title(title)
    axis.grid(axis="x", alpha=0.25)
    return _save(figure, output)


def render_single_robot_paper_figures(
    analysis: Mapping[str, Any], output_dir: Path
) -> dict[str, Any]:
    """Render six sealed files (PNG/PDF for three factual figure panels)."""

    if analysis.get("schema_version") != "corrected_v5_stochastic_comparison_v1":
        raise ValueError("paper renderer requires corrected V5 comparison evidence")
    output_dir.mkdir(parents=True, exist_ok=False)
    files = _block_performance(analysis, output_dir)

    main = analysis["corrected_main_analysis"]
    original_means = {
        metric: _finite(
            main["family_summaries"]["original_mtare"][metric]["mean"],
            name=f"original:{metric}",
        )
        for metric in METRICS
    }
    labels, means, lower, upper = _relative_effect_rows(
        main["m1d_comparisons"], original_means, prefix="v5_minus_original"
    )
    files.extend(_effect_figure(
        labels, means, lower, upper,
        title="V5 structural topology vs. original M-TARE (10 paired blocks)",
        output=output_dir / "v5_vs_original_paired_effects", color=FAMILY_COLORS["m1d_topology"],
    ))

    corrective = analysis["v5_vs_defective_v9"]["metrics"]
    defective_means = {
        metric: _finite(corrective[metric]["defective_v9_mean"], name=f"defective:{metric}")
        for metric in METRICS
    }
    labels, means, lower, upper = _relative_effect_rows(
        corrective, defective_means, prefix="v5_minus_defective_v9"
    )
    files.extend(_effect_figure(
        labels, means, lower, upper,
        title="Combined graph correction: V5 vs. defective V9 (10 paired blocks)",
        output=output_dir / "v5_vs_defective_v9_paired_effects", color="#D55E00",
    ))

    rows = []
    for path in sorted(files):
        rows.append({
            "path": path.name,
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        })
    return {
        "schema_version": "gate6_single_robot_paper_figure_manifest_v1",
        "figure_count": 3,
        "file_count": 6,
        "source_schema_version": analysis["schema_version"],
        "manual_value_entry": False,
        "files": rows,
    }


__all__ = ["render_single_robot_paper_figures"]
