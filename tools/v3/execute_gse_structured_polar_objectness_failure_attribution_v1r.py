#!/usr/bin/env python3
"""Correct V1 attribution with explicit predicted depth-slot provenance."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import PROJECT_ROOT  # noqa: F401
from evaluate_gse_structured_polar_multidepth_capacity_v1 import _second_depth_target_mask
from execute_gse_structured_polar_objectness_failure_attribution_v1 import _load_predictions
from mtare_topo.data.gse_observable_spatial_event_dataset import load_observable_spatial_event_teacher
from mtare_topo.evaluation.gse_spatial_event_set_metrics import _maximum_valid_matching


PASS = "PASS_GSE_STRUCTURED_POLAR_OBJECTNESS_FAILURE_ATTRIBUTION_V1R"


def _target_bins(targets) -> np.ndarray:
    xyz = targets["event_relative_xyz_m"]
    bearing = np.degrees(np.mod(np.arctan2(xyz[..., 1], xyz[..., 0]), 2.0 * np.pi))
    return np.mod(np.floor((bearing + 0.25) / 2.0).astype(np.int64), 180)


def _target_to_prediction(predictions, targets) -> np.ndarray:
    mapping = np.full(targets["event_mask"].shape, -1, dtype=np.int16)
    for row in range(len(mapping)):
        chosen = np.arange(predictions["confidence"].shape[1], dtype=np.int64)
        chosen = np.asarray(sorted(chosen.tolist(), key=lambda index: (int(predictions["event_type"][row, index]), *tuple(float(value) for value in predictions["relative_xyz_m"][row, index]), -float(predictions["confidence"][row, index]))), dtype=np.int64)
        active = np.flatnonzero(targets["event_mask"][row])
        matches = _maximum_valid_matching(
            predictions["event_type"][row, chosen],
            predictions["relative_xyz_m"][row, chosen],
            targets["event_type_index"][row, active],
            targets["event_relative_xyz_m"][row, active],
            maximum_error_m=4.0,
        )
        for prediction_local, target_local, _ in matches:
            mapping[row, active[target_local]] = int(chosen[prediction_local])
    return mapping


def _quantiles(values: np.ndarray) -> dict[str, float | None]:
    if not len(values):
        return {"minimum": None, "median": None, "maximum": None}
    return {"minimum": float(np.min(values)), "median": float(np.median(values)), "maximum": float(np.max(values))}


def _seed_record(predictions, targets, second_depth, target_bins) -> dict[str, object]:
    mapping = _target_to_prediction(predictions, targets)
    matched = mapping >= 0
    prediction_slots = np.full_like(mapping, -1, dtype=np.int8)
    rows, columns = np.nonzero(matched)
    prediction_slots[rows, columns] = predictions["depth_slot"][rows, mapping[rows, columns]]
    second_matched = matched & second_depth
    second_counts = {str(slot): int(np.sum(second_matched & (prediction_slots == slot))) for slot in (0, 1)}
    pair_rows = np.flatnonzero(second_depth.any(axis=1))
    both_matched = 0
    distinct_slot = 0
    both_slot0 = 0
    for row in pair_rows:
        far = int(np.flatnonzero(second_depth[row])[0])
        active = np.flatnonzero(targets["event_mask"][row])
        near_candidates = active[(target_bins[row, active] == target_bins[row, far]) & (active != far)]
        if len(near_candidates) != 1:
            raise RuntimeError("same-bin pair identity drift")
        near = int(near_candidates[0])
        if matched[row, far] and matched[row, near]:
            both_matched += 1
            slots = {int(prediction_slots[row, far]), int(prediction_slots[row, near])}
            distinct_slot += int(slots == {0, 1})
            both_slot0 += int(slots == {0})
    top16_slot = {str(slot): int(np.sum(predictions["depth_slot"] == slot)) for slot in (0, 1)}
    selected = predictions["confidence"] >= 0.95
    selected_slot = {str(slot): int(np.sum(selected & (predictions["depth_slot"] == slot))) for slot in (0, 1)}
    slot1_confidence = predictions["confidence"][predictions["depth_slot"] == 1]
    return {
        "top16_candidates_by_depth_slot": top16_slot,
        "selected_candidates_by_depth_slot": selected_slot,
        "top16_slot1_confidence": _quantiles(slot1_confidence.astype(np.float64)),
        "all_typed_matches_by_prediction_depth_slot": {str(slot): int(np.sum(matched & (prediction_slots == slot))) for slot in (0, 1)},
        "second_depth_targets": int(second_depth.sum()),
        "matched_second_depth_by_prediction_slot": second_counts,
        "matched_second_depth_total": int(second_matched.sum()),
        "same_bin_pair_rows": len(pair_rows),
        "same_bin_pairs_both_targets_matched": both_matched,
        "same_bin_pairs_matched_with_distinct_prediction_slots": distinct_slot,
        "same_bin_pairs_matched_with_slot0_only": both_slot0,
        "slot1_second_depth_utilized": second_counts["1"] > 0,
        "distinct_slot_pair_utilized": distinct_slot > 0,
    }


def _plot(output: Path, records: list[dict], decision: str) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.3), constrained_layout=True)
    x = np.arange(3)
    slot0 = [r["top16_candidates_by_depth_slot"]["0"] for r in records]
    slot1 = [r["top16_candidates_by_depth_slot"]["1"] for r in records]
    axes[0].bar(x - 0.18, slot0, 0.36, label="slot0")
    axes[0].bar(x + 0.18, slot1, 0.36, label="slot1")
    axes[0].set_yscale("symlog", linthresh=1)
    axes[0].set_xticks(x, ["seed 0", "seed 1", "seed 2"])
    axes[0].set_ylabel("top16 candidate count (symlog)")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    matched0 = [r["matched_second_depth_by_prediction_slot"]["0"] for r in records]
    matched1 = [r["matched_second_depth_by_prediction_slot"]["1"] for r in records]
    axes[1].bar(x, matched0, label="matched by slot0", color="#f08c00")
    axes[1].bar(x, matched1, bottom=matched0, label="matched by slot1", color="#1864ab")
    axes[1].set_xticks(x, ["seed 0", "seed 1", "seed 2"])
    axes[1].set_ylabel("79 second-depth targets")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)

    pair0 = [r["same_bin_pairs_matched_with_slot0_only"] for r in records]
    pair01 = [r["same_bin_pairs_matched_with_distinct_prediction_slots"] for r in records]
    axes[2].bar(x, pair0, label="both via slot0", color="#e67700")
    axes[2].bar(x, pair01, bottom=pair0, label="distinct slot0+slot1", color="#2b8a3e")
    axes[2].set_xticks(x, ["seed 0", "seed 1", "seed 2"])
    axes[2].set_ylabel("same-bin near/far pairs jointly matched")
    axes[2].legend(fontsize=8)
    axes[2].grid(axis="y", alpha=0.25)
    figure.suptitle(f"Structured-polar depth-slot utilization: {decision}")
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output / f"gse_structured_polar_objectness_failure_attribution_v1r.{suffix}", dpi=220)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-root", required=True, type=Path)
    parser.add_argument("--capacity-run", required=True, type=Path)
    parser.add_argument("--v1-attribution-run", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=False)
    teacher = load_observable_spatial_event_teacher(args.teacher_root.resolve())
    rows = teacher.selection_rows; targets = teacher.targets(rows); global_index = teacher.global_sequence_index[rows]
    second_depth = _second_depth_target_mask(targets); target_bins = _target_bins(targets)
    if int(second_depth.sum()) != 79:
        raise RuntimeError("second-depth population drift")
    v1 = json.loads((args.v1_attribution_run / "artifacts/audit/summary.json").read_text(encoding="utf-8"))
    if v1.get("status") != "PASS_GSE_STRUCTURED_POLAR_OBJECTNESS_FAILURE_ATTRIBUTION_V1" or v1.get("decision") != "FROZEN_GEOMETRY_OBJECTNESS_REFIT_REQUIRED":
        raise RuntimeError("V1 attribution source drift")
    records = []
    for seed in (0, 1, 2):
        predictions = _load_predictions(args.capacity_run / f"artifacts/models/seed{seed}/selection_outputs.npz", rows, global_index)
        records.append({"seed": seed, **_seed_record(predictions, targets, second_depth, target_bins)})
    semantic_gate = all(r["slot1_second_depth_utilized"] and r["distinct_slot_pair_utilized"] for r in records)
    decision = "KEEP_FROZEN_GEOMETRY_OBJECTNESS_REFIT" if semantic_gate else "STOP_STRUCTURED_POLAR_MULTIDEPTH_ROUTE_USE_EXECUTABLE_EXIT_TOKENS"
    summary = {
        "schema_version": "gse_structured_polar_objectness_failure_attribution_v1r",
        "status": PASS,
        "decision": decision,
        "v1_decision": v1["decision"],
        "v1_decision_superseded": decision != "KEEP_FROZEN_GEOMETRY_OBJECTNESS_REFIT",
        "semantic_depth_slot_gate": semantic_gate,
        "population": {"worlds": 20, "observations": len(rows), "target_tokens": int(targets["event_mask"].sum()), "second_depth_targets": int(second_depth.sum()), "seed_archives": 3},
        "seed_records": records,
        "optimizer_steps": 0,
        "model_inference_frames": 0,
        "threshold_selection_steps": 0,
        "duration_seconds": time.monotonic() - started,
        "c09_worlds_read": 0,
        "c10_worlds_read": 0,
        "mtare_worlds_read": 0,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "figure_source.json").write_text(json.dumps({"schema_version": "gse_structured_polar_objectness_failure_attribution_v1r_figure_source_v1", "summary": summary}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fields = ("seed", "top16_slot0", "top16_slot1", "selected_slot0", "selected_slot1", "matched_second_by_slot0", "matched_second_by_slot1", "pairs_both_matched", "pairs_distinct_slots", "pairs_slot0_only")
    with (output / "depth_slot_utilization.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for r in records:
            writer.writerow({"seed": r["seed"], "top16_slot0": r["top16_candidates_by_depth_slot"]["0"], "top16_slot1": r["top16_candidates_by_depth_slot"]["1"], "selected_slot0": r["selected_candidates_by_depth_slot"]["0"], "selected_slot1": r["selected_candidates_by_depth_slot"]["1"], "matched_second_by_slot0": r["matched_second_depth_by_prediction_slot"]["0"], "matched_second_by_slot1": r["matched_second_depth_by_prediction_slot"]["1"], "pairs_both_matched": r["same_bin_pairs_both_targets_matched"], "pairs_distinct_slots": r["same_bin_pairs_matched_with_distinct_prediction_slots"], "pairs_slot0_only": r["same_bin_pairs_matched_with_slot0_only"]})
    _plot(output, records, decision)
    print(json.dumps({"status": PASS, "decision": decision, "semantic_depth_slot_gate": semantic_gate}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
