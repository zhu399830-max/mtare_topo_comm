#!/usr/bin/env python3
"""Evaluate the three-seed height-aware spatial event-center ensemble."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from train_gse_spatial_event_center_v1 import _cross_metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    for seed in range(3):
        parser.add_argument(f"--seed{seed}", required=True, type=Path)
    parser.add_argument("--baseline-projection", required=True, type=Path)
    parser.add_argument("--action-cache", required=True, type=Path)
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    references = None
    vectors = []
    centers = []
    seed_summaries = {}
    for seed in range(3):
        path = getattr(args, f"seed{seed}").resolve()
        with np.load(path / "selection_outputs.npz", allow_pickle=False) as archive:
            current = {key: archive[key] for key in archive.files}
        if references is None:
            references = current
        elif not all(np.array_equal(current[key], references[key]) for key in (
            "observation_row", "global_sequence_index", "target_local_vector_m", "target_center_xyz_m",
        )):
            raise RuntimeError("spatial event-center seed selection identity drift")
        vectors.append(current["predicted_local_vector_m"].astype(np.float64))
        centers.append(current["predicted_center_xyz_m"].astype(np.float64))
        seed_summaries[str(seed)] = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    assert references is not None
    rows = references["observation_row"].astype(np.int64)
    target_vector = references["target_local_vector_m"].astype(np.float64)
    target_center = references["target_center_xyz_m"].astype(np.float64)
    ensemble_vector = np.mean(np.stack(vectors), axis=0)
    ensemble_center = np.mean(np.stack(centers), axis=0)

    with np.load(args.baseline_projection.resolve(), allow_pickle=False) as archive:
        if not np.array_equal(archive["global_sequence_index"][rows], references["global_sequence_index"]):
            raise RuntimeError("spatial event-center baseline identity drift")
        baseline_center = archive["projected_center_xyz_m"][rows].astype(np.float64)
        baseline_longitudinal = archive["predicted_offset_m"][rows].astype(np.float64)
    traversal = np.load(args.action_cache.resolve() / "traversal_id.npy").astype(str)
    with np.load(args.teacher.resolve(), allow_pickle=False) as archive:
        identity = archive["identity"].astype(str)
    ensemble_cross = _cross_metrics(ensemble_center, rows, identity, traversal, target_center)
    baseline_cross = _cross_metrics(baseline_center, rows, identity, traversal, target_center)
    ensemble_center_mae = float(np.mean(np.linalg.norm(ensemble_center - target_center, axis=1)))
    baseline_center_mae = float(np.mean(np.linalg.norm(baseline_center - target_center, axis=1)))
    component_mae = np.mean(np.abs(ensemble_vector - target_vector), axis=0)
    baseline_component_mae = np.asarray([
        np.mean(np.abs(baseline_longitudinal - target_vector[:, 0])),
        np.mean(np.abs(target_vector[:, 1])),
        np.mean(np.abs(target_vector[:, 2])),
    ])
    relative_improvement = 1.0 - (
        ensemble_cross["identity_macro_relative_vector_error_m"]
        / baseline_cross["identity_macro_relative_vector_error_m"]
    )
    within_gain = (
        ensemble_cross["identity_macro_within_4m_fraction"]
        - baseline_cross["identity_macro_within_4m_fraction"]
    )
    gates = {
        "relative_vector_error_improves_by_0p10": bool(relative_improvement >= 0.10),
        "within_4m_fraction_gains_0p10": bool(within_gain >= 0.10),
        "longitudinal_mae_does_not_regress": bool(component_mae[0] <= baseline_component_mae[0] + 1e-6),
        "lateral_mae_improves_zero_output": bool(component_mae[1] < baseline_component_mae[1]),
        "vertical_mae_improves_zero_output": bool(component_mae[2] < baseline_component_mae[2]),
        "global_center_mae_improves_scalar_projection": bool(ensemble_center_mae < baseline_center_mae),
        "all_seeds_relative_error_below_baseline": bool(all(
            value["selection_cross_view"]["identity_macro_relative_vector_error_m"]
            < baseline_cross["identity_macro_relative_vector_error_m"]
            for value in seed_summaries.values()
        )),
    }
    passed = all(gates.values())
    summary = {
        "schema_version": "gse_spatial_event_center_ensemble_v1",
        "status": (
            "PASS_GSE_SPATIAL_EVENT_CENTER_ENSEMBLE_V1"
            if passed else "FAIL_GSE_SPATIAL_EVENT_CENTER_ENSEMBLE_V1"
        ),
        "selection_rows": len(rows),
        "ensemble_cross_view": ensemble_cross,
        "baseline_cross_view": baseline_cross,
        "relative_vector_error_improvement": relative_improvement,
        "within_4m_fraction_gain": within_gain,
        "ensemble_global_center_mae_m": ensemble_center_mae,
        "baseline_global_center_mae_m": baseline_center_mae,
        "ensemble_component_mae_m": component_mae.tolist(),
        "baseline_component_mae_m": baseline_component_mae.tolist(),
        "seeds": seed_summaries,
        "gates": gates,
        "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0,
    }
    np.savez_compressed(
        args.output_dir / "ensemble_selection_outputs.npz",
        observation_row=rows,
        global_sequence_index=references["global_sequence_index"],
        predicted_local_vector_m=ensemble_vector.astype(np.float32),
        predicted_center_xyz_m=ensemble_center.astype(np.float32),
        target_local_vector_m=target_vector.astype(np.float32),
        target_center_xyz_m=target_center.astype(np.float32),
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
