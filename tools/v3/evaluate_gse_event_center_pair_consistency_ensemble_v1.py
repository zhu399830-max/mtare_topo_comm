#!/usr/bin/env python3
"""Evaluate three cross-traversal event-center heads on C07--C08."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from train_gse_event_center_pair_consistency_v1 import _cross_view_metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    for seed in range(3): parser.add_argument(f"--seed{seed}", required=True, type=Path)
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--action-cache", required=True, type=Path)
    parser.add_argument("--baseline-projection", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=False)
    predictions, seed_summaries, reference = [], {}, None
    for seed in range(3):
        path = getattr(args, f"seed{seed}").resolve()
        with np.load(path / "selection_outputs.npz", allow_pickle=False) as archive:
            current = {key: archive[key] for key in archive.files}
        if reference is None: reference = current
        elif not all(np.array_equal(current[key], reference[key]) for key in ("observation_row", "global_sequence_index", "target_offset_m", "event_index")):
            raise RuntimeError("pair-consistency seed selection identity drift")
        predictions.append(current["predicted_offset_m"].astype(np.float64))
        seed_summaries[str(seed)] = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    assert reference is not None
    with np.load(args.teacher.resolve(), allow_pickle=False) as archive:
        global_index = archive["global_sequence_index"]
        identity = archive["identity"].astype(str); tangent = archive["route_tangent_xyz"].astype(np.float32)
        target_all = archive["signed_center_offset_m"].astype(np.float32)
        oracle = archive["oracle_longitudinal_center_xyz_m"].astype(np.float32)
    if not np.array_equal(global_index[reference["observation_row"]], reference["global_sequence_index"]):
        raise RuntimeError("pair-consistency teacher identity drift")
    traversal = np.load(args.action_cache / "traversal_id.npy").astype(str)
    sensor = oracle - target_all[:, None] * tangent
    rows = reference["observation_row"].astype(np.int64)
    ensemble = np.mean(np.stack(predictions), axis=0)
    target = reference["target_offset_m"].astype(np.float64)
    ensemble_cross = _cross_view_metrics(ensemble, rows, identity, traversal, sensor, tangent, target_all)
    with np.load(args.baseline_projection.resolve(), allow_pickle=False) as archive:
        if not np.array_equal(archive["global_sequence_index"], global_index):
            raise RuntimeError("pair-consistency baseline identity drift")
        baseline_prediction = archive["predicted_offset_m"][rows].astype(np.float64)
    baseline_cross = _cross_view_metrics(baseline_prediction, rows, identity, traversal, sensor, tangent, target_all)
    baseline_mae = float(np.mean(np.abs(baseline_prediction - target)))
    mae = float(np.mean(np.abs(ensemble - target)))
    relative_improvement = 1.0 - ensemble_cross["identity_macro_relative_vector_error_m"] / baseline_cross["identity_macro_relative_vector_error_m"]
    within_gain = ensemble_cross["identity_macro_within_4m_fraction"] - baseline_cross["identity_macro_within_4m_fraction"]
    gates = {
        "relative_vector_error_improves_by_0p10": relative_improvement >= .10,
        "within_4m_fraction_gains_0p10": within_gain >= .10,
        "ensemble_mae_does_not_regress": mae <= baseline_mae,
        "all_seeds_relative_error_below_baseline": all(
            value["selection_cross_view"]["identity_macro_relative_vector_error_m"]
            < baseline_cross["identity_macro_relative_vector_error_m"] for value in seed_summaries.values()
        ),
    }
    passed = all(gates.values())
    summary = {
        "schema_version": "gse_event_center_pair_consistency_ensemble_v1",
        "status": "PASS_GSE_EVENT_CENTER_PAIR_CONSISTENCY_ENSEMBLE_V1" if passed else "FAIL_GSE_EVENT_CENTER_PAIR_CONSISTENCY_ENSEMBLE_V1",
        "selection_rows": len(rows), "selection_mae_m": mae, "baseline_mae_m": baseline_mae,
        "ensemble_cross_view": ensemble_cross, "baseline_cross_view": baseline_cross,
        "relative_vector_error_improvement": relative_improvement, "within_4m_fraction_gain": within_gain,
        "seeds": seed_summaries, "gates": gates,
        "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0,
    }
    np.savez_compressed(
        args.output_dir / "ensemble_selection_outputs.npz",
        observation_row=rows, global_sequence_index=reference["global_sequence_index"],
        predicted_offset_m=ensemble.astype(np.float32), target_offset_m=target.astype(np.float32),
        event_index=reference["event_index"].astype(np.int8),
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True)); return 0 if passed else 2


if __name__ == "__main__": raise SystemExit(main())
