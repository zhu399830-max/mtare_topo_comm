#!/usr/bin/env python3
"""Audit whether frozen five-frame GSE features expose causal geometry change."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mtare_topo.evaluation.gse_causal_geometry_delta import (
    fixed_negative_quantile_threshold,
    signed_lag_delta,
    summarize_binary_score,
)
from mtare_topo.governance import write_json


PASS_STATUS = "PASS_GSE_CAUSAL_GEOMETRY_DELTA_OBSERVABILITY_V1"
FAIL_STATUS = "FAIL_GSE_CAUSAL_GEOMETRY_DELTA_OBSERVABILITY_V1"
GEOMETRY_NAMES = ("width_m", "height_m", "slope_deg", "curvature_per_m")
FEATURE_SCALE = np.asarray((30.0, 30.0, 45.0, 0.1), dtype=np.float64)
EVENTS = ("corridor", "junction", "terminal", "turn", "geometry_transition")


def _read_teacher(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            missing = {
                "global_sequence_index",
                "parent_id",
                "traversal_id",
                "sequence_index",
                "event",
                "identity",
                *GEOMETRY_NAMES,
            } - row.keys()
            if missing:
                raise RuntimeError(f"Teacher row {line_number} misses {sorted(missing)}")
            rows.append(row)
    return rows


def _lag_pairs(rows: list[dict], lag: int = 4) -> tuple[np.ndarray, np.ndarray]:
    lookup: dict[tuple[str, int], int] = {}
    for index, row in enumerate(rows):
        key = (str(row["traversal_id"]), int(row["sequence_index"]))
        if key in lookup:
            raise RuntimeError(f"duplicate traversal sequence key: {key}")
        lookup[key] = index
    current: list[int] = []
    past: list[int] = []
    for index, row in enumerate(rows):
        previous = lookup.get((str(row["traversal_id"]), int(row["sequence_index"]) - lag))
        if previous is not None:
            current.append(index)
            past.append(previous)
    return np.asarray(current, dtype=np.int64), np.asarray(past, dtype=np.int64)


def _score_summary(
    score: np.ndarray,
    events: np.ndarray,
    partition: np.ndarray,
    code: int,
) -> dict:
    eligible = (partition == code) & np.isin(events, ("corridor", "geometry_transition"))
    positive = events[eligible] == "geometry_transition"
    return summarize_binary_score(score[eligible], positive).to_dict()


def _write_figure(
    output: Path,
    selection_events: np.ndarray,
    teacher_score: np.ndarray,
    predicted_score: np.ndarray,
    auc_rows: list[dict],
    mae_rows: list[dict],
    low_fpr_rows: list[dict],
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11.4, 7.6), constrained_layout=True)
    corridor = selection_events == "corridor"
    transition = selection_events == "geometry_transition"
    bins = np.linspace(0.0, float(np.quantile(np.r_[teacher_score, predicted_score], 0.995)), 50)
    bins[0] = 0.0
    for score, title, axis in (
        (teacher_score, "Teacher geometry change", axes[0, 0]),
        (predicted_score, "Frozen GSE geometry change", axes[0, 1]),
    ):
        axis.hist(score[corridor], bins=bins, density=True, alpha=0.65, label="corridor")
        axis.hist(score[transition], bins=bins, density=True, alpha=0.65, label="transition")
        axis.set_title(title)
        axis.set_xlabel("max |Δwidth, Δheight| (m)")
        axis.set_ylabel("density")
        axis.legend(frameon=False)
    labels = [row["source"] for row in auc_rows]
    axes[1, 0].bar(labels, [row["selection_auc"] for row in auc_rows], color="#3478b8")
    axes[1, 0].axhline(0.65, color="#b53a3a", linestyle="--", linewidth=1.2, label="feasibility gate")
    axes[1, 0].set_ylim(0.5, 0.8)
    axes[1, 0].set_ylabel("transition-vs-corridor ROC-AUC")
    axes[1, 0].set_title("C07–C08 causal observability")
    axes[1, 0].tick_params(axis="x", rotation=25)
    axes[1, 0].legend(frameon=False)
    sources = sorted({row["source"] for row in mae_rows})
    x = np.arange(len(GEOMETRY_NAMES))
    width = 0.8 / max(1, len(sources))
    for offset, source in enumerate(sources):
        values = [next(row["selection_mae"] for row in mae_rows if row["source"] == source and row["component"] == name) for name in GEOMETRY_NAMES]
        axes[1, 1].bar(x + (offset - (len(sources) - 1) / 2) * width, values, width, label=source)
    axes[1, 1].set_xticks(x, ("width", "height", "slope", "curvature"))
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_ylabel("absolute delta error (native unit)")
    axes[1, 1].set_title(
        "Frozen prediction error; 1% FPR transition coverage "
        + ", ".join(f"{row['source']}={row['identity_covered']}/{row['identity_total']}" for row in low_fpr_rows)
    )
    axes[1, 1].legend(frameon=False, fontsize=8)
    figure.suptitle("Why GSE-Graph needs explicit causal geometry-delta learning", fontsize=14)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output.with_suffix(f".{suffix}"), dpi=220 if suffix == "png" else None)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--pair-cache", required=True, type=Path)
    parser.add_argument("--feature", action="append", required=True, type=Path)
    args = parser.parse_args()
    if len(args.feature) != 3:
        raise RuntimeError("exactly three frozen feature arrays are required")

    run_dir = args.run_dir.resolve()
    artifacts = run_dir / "artifacts"
    previews = run_dir / "previews"
    metrics_dir = run_dir / "metrics"
    artifacts.mkdir(parents=True, exist_ok=True)
    previews.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_teacher(args.teacher)
    if len(rows) != 188126:
        raise RuntimeError(f"Teacher population drift: {len(rows)}")
    with np.load(args.pair_cache, allow_pickle=False) as archive:
        global_indices = np.asarray(archive["compact_to_global_sequence_index"], dtype=np.int64)
        partition = np.asarray(archive["partition_code"], dtype=np.uint8)
        parent_ids = np.asarray(archive["parent_id"])
    if tuple(np.bincount(partition, minlength=2)) != (142184, 45942):
        raise RuntimeError("fit/selection partition drift")
    teacher_global = np.asarray([int(row["global_sequence_index"]) for row in rows], dtype=np.int64)
    teacher_parent = np.asarray([str(row["parent_id"]) for row in rows])
    if not np.array_equal(global_indices, teacher_global) or not np.array_equal(parent_ids, teacher_parent):
        raise RuntimeError("Teacher/cache compact-row identity mismatch")
    if any(str(value).endswith(("_C09", "_C10")) for value in parent_ids):
        raise RuntimeError("forbidden C09/C10 parent in development audit")

    events_all = np.asarray([str(row["event"]) for row in rows])
    if not set(events_all).issubset(EVENTS):
        raise RuntimeError("unknown corrected Teacher event")
    geometry = np.asarray(
        [
            [float(row[name]) if row[name] is not None else np.nan for name in GEOMETRY_NAMES]
            for row in rows
        ],
        dtype=np.float64,
    )
    geometry_valid = np.asarray(
        [bool(row.get("geometry_valid")) and np.all(np.isfinite(geometry[index])) for index, row in enumerate(rows)],
        dtype=np.bool_,
    )
    current_all, past_all = _lag_pairs(rows, lag=4)
    valid_pair = geometry_valid[current_all] & geometry_valid[past_all]
    current = current_all[valid_pair]
    past = past_all[valid_pair]
    if len(current) == 0 or np.any(partition[current] != partition[past]):
        raise RuntimeError("invalid causal lag population")
    events = events_all[current]
    split = partition[current]
    identities = np.asarray([str(rows[index]["identity"]) for index in current])
    teacher_delta = signed_lag_delta(geometry[current], geometry[past])
    teacher_score = np.max(np.abs(teacher_delta[:, :2]), axis=1)

    feature_delta: list[np.ndarray] = []
    for feature_path in args.feature:
        feature = np.load(feature_path, mmap_mode="r")
        if feature.shape != (188126, 146) or feature.dtype != np.float32:
            raise RuntimeError(f"frozen observation feature drift: {feature_path}")
        predicted_geometry = np.asarray(feature[:, 8:12], dtype=np.float64) * FEATURE_SCALE
        feature_delta.append(signed_lag_delta(predicted_geometry[current], predicted_geometry[past]))
    predicted_mean_delta = np.mean(np.stack(feature_delta, axis=0), axis=0)

    score_by_source = {"Teacher": teacher_score}
    for seed, delta in enumerate(feature_delta):
        score_by_source[f"seed{seed}"] = np.max(np.abs(delta[:, :2]), axis=1)
    score_by_source["3-seed mean"] = np.max(np.abs(predicted_mean_delta[:, :2]), axis=1)

    auc_rows: list[dict] = []
    low_fpr_rows: list[dict] = []
    for source, score in score_by_source.items():
        fit = _score_summary(score, events, split, 0)
        selection = _score_summary(score, events, split, 1)
        auc_rows.append(
            {
                "source": source,
                "fit_auc": fit["roc_auc"],
                "selection_auc": selection["roc_auc"],
                "absolute_auc_gap": abs(fit["roc_auc"] - selection["roc_auc"]),
                "fit_positive": fit["positive_count"],
                "fit_negative": fit["negative_count"],
                "selection_positive": selection["positive_count"],
                "selection_negative": selection["negative_count"],
            }
        )
        fit_corridor = (split == 0) & (events == "corridor")
        selection_corridor = (split == 1) & (events == "corridor")
        selection_transition = (split == 1) & (events == "geometry_transition")
        threshold = fixed_negative_quantile_threshold(score[fit_corridor], false_positive_rate=0.01)
        accepted_transition = selection_transition & (score >= threshold)
        transition_identities = set(identities[selection_transition])
        covered = set(identities[accepted_transition])
        low_fpr_rows.append(
            {
                "source": source,
                "fit_threshold": threshold,
                "selection_corridor_false_positive_rate": float(np.mean(score[selection_corridor] >= threshold)),
                "selection_transition_recall": float(np.mean(score[selection_transition] >= threshold)),
                "identity_covered": len(covered),
                "identity_total": len(transition_identities),
            }
        )

    mae_rows: list[dict] = []
    for seed, delta in [(f"seed{index}", value) for index, value in enumerate(feature_delta)] + [("3-seed mean", predicted_mean_delta)]:
        absolute_error = np.abs(delta - teacher_delta)
        for component_index, component in enumerate(GEOMETRY_NAMES):
            mae_rows.append(
                {
                    "source": seed,
                    "component": component,
                    "fit_mae": float(np.mean(absolute_error[split == 0, component_index])),
                    "selection_mae": float(np.mean(absolute_error[split == 1, component_index])),
                }
            )

    def write_csv(name: str, data: list[dict]) -> None:
        with (artifacts / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)

    write_csv("auc_summary.csv", auc_rows)
    write_csv("delta_mae.csv", mae_rows)
    write_csv("fit_threshold_selection_transfer.csv", low_fpr_rows)
    selected = split == 1
    _write_figure(
        previews / "gse_causal_geometry_delta_observability",
        events[selected],
        teacher_score[selected],
        score_by_source["3-seed mean"][selected],
        auc_rows,
        mae_rows,
        low_fpr_rows,
    )

    mean_auc = next(row for row in auc_rows if row["source"] == "3-seed mean")
    teacher_auc = next(row for row in auc_rows if row["source"] == "Teacher")
    exact_worlds = len(set(parent_ids.tolist())) == 80
    exact_transition_identities = len(set(identities[(split == 1) & (events == "geometry_transition")])) == 17
    checks = {
        "exact_80_development_worlds": exact_worlds,
        "exact_188126_observations": len(rows) == 188126,
        "exact_fit_selection_populations": tuple(np.bincount(partition, minlength=2)) == (142184, 45942),
        "exact_17_selection_transition_identities": exact_transition_identities,
        "teacher_selection_auc_at_least_0_70": teacher_auc["selection_auc"] >= 0.70,
        "predicted_mean_selection_auc_at_least_0_65": mean_auc["selection_auc"] >= 0.65,
        "predicted_mean_fit_selection_gap_at_most_0_05": mean_auc["absolute_auc_gap"] <= 0.05,
        "all_values_finite": bool(
            np.all(np.isfinite(teacher_delta))
            and np.all(np.isfinite(predicted_mean_delta))
            and all(np.isfinite(list(row.values())[1:]).all() for row in auc_rows)
        ),
        "zero_c09_c10_mtare": True,
        "zero_training_or_model_update": True,
    }
    passed = all(checks.values())
    summary = {
        "schema_version": "gse_causal_geometry_delta_observability_v1",
        "overall_status": PASS_STATUS if passed else FAIL_STATUS,
        "scientific_pass": passed,
        "research_conclusion": (
            "Frozen five-frame GSE geometry predictions contain cross-world causal change signal, so an explicit geometry-delta multitask decoder is justified. The 1% FPR diagnostic remains insufficient and is not a pass criterion."
            if passed
            else "Frozen five-frame geometry change does not meet the pre-registered observability/cross-world stability requirements; do not train the proposed delta decoder."
        ),
        "observations": len(rows),
        "worlds": len(set(parent_ids.tolist())),
        "fit_observations": int(np.sum(partition == 0)),
        "selection_observations": int(np.sum(partition == 1)),
        "causal_lag_steps": 4,
        "valid_lag_pairs": len(current),
        "all_lag_pairs_before_geometry_validity": len(current_all),
        "geometry_invalid_lag_pairs_excluded": int(len(current_all) - len(current)),
        "fit_valid_lag_pairs": int(np.sum(split == 0)),
        "selection_valid_lag_pairs": int(np.sum(split == 1)),
        "auc": auc_rows,
        "delta_mae": mae_rows,
        "low_fpr_diagnostic": low_fpr_rows,
        "checks": checks,
        "optimizer_steps": 0,
        "model_inference_frames": 0,
        "c09_worlds_read": 0,
        "strict_test_worlds_read": 0,
        "mtare_worlds_read": 0,
    }
    write_json(metrics_dir / "observability_summary.json", summary)
    write_json(
        previews / "gse_causal_geometry_delta_observability_source.json",
        {
            "figure": "gse_causal_geometry_delta_observability",
            "sources": [str(args.teacher), str(args.pair_cache), *map(str, args.feature)],
            "statistics": summary,
            "paper_use": "Development failure-to-method motivation only; not a strict-test result.",
        },
    )
    print(json.dumps(summary, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
