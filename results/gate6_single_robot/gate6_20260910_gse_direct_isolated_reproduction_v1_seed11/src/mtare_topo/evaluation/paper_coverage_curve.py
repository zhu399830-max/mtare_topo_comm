"""Deterministic block-aggregated coverage-time paper figure."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mtare_topo.evaluation.stochastic_closed_loop import paired_bootstrap_ci


CURVE_FAMILIES = (
    "original_mtare",
    "defective_v9",
    "corrected_v5",
    "layered_gt_map_oracle",
)
CURVE_LABELS = {
    "original_mtare": "Original M-TARE",
    "defective_v9": "Defective V9",
    "corrected_v5": "V5 structural topology",
    "layered_gt_map_oracle": "GT-map diagnostic",
}
CURVE_COLORS = {
    "original_mtare": "#4C566A",
    "defective_v9": "#D55E00",
    "corrected_v5": "#0072B2",
    "layered_gt_map_oracle": "#CC79A7",
}
TIME_GRID_SEC = tuple(float(value) for value in range(0, 601, 10))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite coverage curve value: {name}")
    return result


def load_coverage_curve(path: Path, *, expected_samples: int) -> list[tuple[float, float]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            item = json.loads(line)
            elapsed = _finite(item["elapsed_sec"], name=f"{path}:{line_number}:time")
            volume = _finite(
                item["explored_volume_m3"], name=f"{path}:{line_number}:volume"
            )
            if elapsed < 0.0 or volume < 0.0:
                raise ValueError(f"negative coverage curve value: {path}:{line_number}")
            if rows and (elapsed <= rows[-1][0] or volume < rows[-1][1] - 1e-9):
                raise ValueError(f"non-monotonic coverage curve: {path}:{line_number}")
            rows.append((elapsed, volume))
    if len(rows) != expected_samples or not rows or rows[-1][0] > 600.5:
        raise ValueError(f"coverage curve sample contract mismatch: {path}")
    return rows


def _resample(samples: Sequence[tuple[float, float]]) -> list[float]:
    result, index, current = [], 0, 0.0
    for target in TIME_GRID_SEC:
        while index < len(samples) and samples[index][0] <= target + 1e-9:
            current = samples[index][1]
            index += 1
        if target == TIME_GRID_SEC[-1]:
            current = samples[-1][1]
        result.append(current)
    return result


def aggregate_coverage_curves(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    quotas = Counter(str(record["curve_family"]) for record in records)
    if len(records) != 120 or quotas != Counter({family: 30 for family in CURVE_FAMILIES}):
        raise ValueError("paper coverage curves require 30 cases for each of four families")
    grouped: dict[str, dict[str, list[list[float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    case_rows = []
    seen = set()
    for record in records:
        family = str(record["curve_family"])
        block = str(record["block_id"])
        identity = (family, str(record["case_id"]))
        if identity in seen:
            raise ValueError("duplicate paper coverage curve identity")
        seen.add(identity)
        values = _resample(record["samples"])
        grouped[family][block].append(values)
        for elapsed, volume in zip(TIME_GRID_SEC, values):
            case_rows.append({
                "curve_family": family,
                "case_id": str(record["case_id"]),
                "block_id": block,
                "elapsed_sec": elapsed,
                "explored_volume_m3": volume,
            })
    aggregate_rows = []
    for family in CURVE_FAMILIES:
        blocks = grouped[family]
        if len(blocks) != 10 or any(len(values) != 3 for values in blocks.values()):
            raise ValueError(f"paper coverage curve blocks are incomplete: {family}")
        block_means = [
            [sum(values[repeat][time] for repeat in range(3)) / 3.0 for time in range(len(TIME_GRID_SEC))]
            for _, values in sorted(blocks.items())
        ]
        for time_index, elapsed in enumerate(TIME_GRID_SEC):
            values = [row[time_index] for row in block_means]
            lower, upper = paired_bootstrap_ci(values)
            aggregate_rows.append({
                "curve_family": family,
                "elapsed_sec": elapsed,
                "block_mean_explored_volume_m3": sum(values) / len(values),
                "block_bootstrap_95_ci_low_m3": lower,
                "block_bootstrap_95_ci_high_m3": upper,
            })
    return {
        "schema_version": "gate6_single_robot_coverage_curve_analysis_v1",
        "case_count": 120,
        "family_case_counts": dict(sorted(quotas.items())),
        "block_count": 10,
        "repeats_or_checkpoints_per_block": 3,
        "time_grid_sec": list(TIME_GRID_SEC),
        "pointwise_interval": "10000-sample bootstrap over ten block means; descriptive, not simultaneous",
        "case_rows": case_rows,
        "aggregate_rows": aggregate_rows,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def render_coverage_curve(
    records: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    required = (
        "source_audit_run", "source_audit_seal_sha256",
        "corrected_v5_run", "corrected_v5_seal_sha256",
        "verified_curve_files",
    )
    if any(not provenance.get(field) for field in required):
        raise ValueError("paper coverage curve provenance is incomplete")
    analysis = aggregate_coverage_curves(records)
    output_dir.mkdir(parents=True, exist_ok=False)
    case_csv = output_dir / "coverage_time_case_curves.csv"
    aggregate_csv = output_dir / "coverage_time_block_aggregate.csv"
    analysis_json = output_dir / "coverage_time_block_aggregate.json"
    _write_csv(
        case_csv,
        analysis["case_rows"],
        ("curve_family", "case_id", "block_id", "elapsed_sec", "explored_volume_m3"),
    )
    _write_csv(
        aggregate_csv,
        analysis["aggregate_rows"],
        (
            "curve_family", "elapsed_sec", "block_mean_explored_volume_m3",
            "block_bootstrap_95_ci_low_m3", "block_bootstrap_95_ci_high_m3",
        ),
    )
    analysis_json.write_text(
        json.dumps(
            {key: value for key, value in analysis.items() if key != "case_rows"},
            indent=2, sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    figure, axis = plt.subplots(figsize=(9.4, 6.0), constrained_layout=True)
    for family in CURVE_FAMILIES:
        rows = [row for row in analysis["aggregate_rows"] if row["curve_family"] == family]
        x = [row["elapsed_sec"] for row in rows]
        mean = [row["block_mean_explored_volume_m3"] for row in rows]
        lower = [row["block_bootstrap_95_ci_low_m3"] for row in rows]
        upper = [row["block_bootstrap_95_ci_high_m3"] for row in rows]
        axis.plot(x, mean, color=CURVE_COLORS[family], linewidth=2.0, label=CURVE_LABELS[family])
        axis.fill_between(x, lower, upper, color=CURVE_COLORS[family], alpha=0.12, linewidth=0)
    axis.set_xlabel("Exploration time (s)")
    axis.set_ylabel("Explored volume (m³)")
    axis.set_title("Single-robot coverage over time (10 randomized blocks)")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    png = output_dir / "coverage_time_curve.png"
    pdf = output_dir / "coverage_time_curve.pdf"
    figure.savefig(png, dpi=220, bbox_inches="tight", metadata={"Software": "mtare_topo deterministic renderer"})
    figure.savefig(pdf, bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    plt.close(figure)

    files = [case_csv, aggregate_csv, analysis_json, png, pdf]
    return {
        "schema_version": "gate6_single_robot_coverage_curve_manifest_v1",
        "figure_count": 1,
        "file_count": len(files),
        "manual_value_entry": False,
        "case_count": 120,
        "block_count": 10,
        "source_provenance": dict(provenance),
        "files": [
            {"path": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in sorted(files)
        ],
    }


__all__ = [
    "CURVE_FAMILIES",
    "TIME_GRID_SEC",
    "aggregate_coverage_curves",
    "load_coverage_curve",
    "render_coverage_curve",
]
