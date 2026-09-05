#!/usr/bin/env python3
"""Select C07--C08 maximin residual scale and apply it to frozen C09 outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json
from mtare_topo.representation.gse_slope_risk_calibration import (
    apply_residual_scale,
    select_maximin_residual_scale,
)


PASS_STATUS = "PASS_GSE_SLOPE_RISK_CALIBRATED_C09_EVALUATION_V2"
FAMILIES = (
    "S01_flat_tree_small",
    "S02_3d_tree_small",
    "S03_flat_unicyclic_small",
    "S04_3d_unicyclic_small",
    "S05_flat_branch_medium",
    "S06_3d_branch_medium",
    "S07_flat_loop_rich",
    "S08_3d_loop_rich",
    "S09_flat_complex",
    "S10_3d_complex",
)
EXPECTED_SELECTION_PARENTS = frozenset(
    f"{family}_C{code:02d}" for family in FAMILIES for code in (7, 8)
)
EXPECTED_C09_PARENTS = frozenset(f"{family}_C09" for family in FAMILIES)


def _load_archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _check_identity(rows: list[dict[str, np.ndarray]], count: int, parents: frozenset[str]) -> None:
    reference_indices = None
    reference_parents = None
    for row in rows:
        indices = np.asarray(row.get("global_sequence_index"), dtype=np.int64)
        parent_id = np.asarray(row.get("parent_id")).astype(str)
        if (
            indices.shape != (count,)
            or parent_id.shape != (count,)
            or len(np.unique(indices)) != count
            or set(parent_id.tolist()) != set(parents)
            or np.asarray(row.get("target_slope_deg")).shape != (count,)
            or np.asarray(row.get("five_frame_prior_slope_deg")).shape != (count,)
            or np.asarray(row.get("corrected_slope_deg")).shape != (count,)
        ):
            raise RuntimeError("slope risk-calibration identity or count drift")
        if reference_indices is not None and (
            not np.array_equal(indices, reference_indices)
            or not np.array_equal(parent_id, reference_parents)
        ):
            raise RuntimeError("slope risk-calibration seed alignment drift")
        reference_indices = indices
        reference_parents = parent_id


def _metrics(
    target: np.ndarray,
    current: np.ndarray,
    prior: np.ndarray,
    corrected: np.ndarray,
    parents: np.ndarray,
) -> dict[str, object]:
    current_error = np.abs(current - target)
    prior_error = np.abs(prior - target)
    corrected_error = np.abs(corrected - target)
    per_world = []
    for parent in sorted(set(parents.tolist())):
        mask = parents == parent
        prior_mae = float(prior_error[mask].mean())
        corrected_mae = float(corrected_error[mask].mean())
        per_world.append(
            {
                "parent_id": parent,
                "sequences": int(mask.sum()),
                "current_mae_deg": float(current_error[mask].mean()),
                "five_frame_prior_mae_deg": prior_mae,
                "corrected_mae_deg": corrected_mae,
                "relative_improvement_over_five_frame_prior": (prior_mae - corrected_mae) / prior_mae,
            }
        )
    prior_mae = float(prior_error.mean())
    corrected_mae = float(corrected_error.mean())
    return {
        "sequences": int(len(target)),
        "current_frame_mae_deg": float(current_error.mean()),
        "five_frame_prior_mae_deg": prior_mae,
        "corrected_mae_deg": corrected_mae,
        "relative_improvement_over_five_frame_prior": (prior_mae - corrected_mae) / prior_mae,
        "per_world": per_world,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corrective-run", required=True, type=Path)
    parser.add_argument("--raw-c09-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    corrective_run = args.corrective_run.resolve()
    raw_c09_dir = args.raw_c09_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    selection = [
        _load_archive(corrective_run / f"artifacts/models/seed{seed}/selection_outputs.npz")
        for seed in (0, 1, 2)
    ]
    _check_identity(selection, 45_942, EXPECTED_SELECTION_PARENTS)
    calibration = select_maximin_residual_scale(selection, grid_denominator=100)
    calibration_row = {
        "schema_version": "gse_slope_residual_risk_calibration_v2",
        "selection_worlds": 20,
        "selection_sequences": 45_942,
        "selection_codes": ["C07", "C08"],
        "selection_rule": "maximize worst parent three-seed-mean relative improvement; tie mean improvement; tie smaller grid index",
        "c09_worlds_read_during_selection": 0,
        "c10_worlds_read": 0,
        "mtare_worlds_read": 0,
        **calibration.to_dict(),
    }
    write_json(output_dir / "risk_calibration.json", calibration_row)

    raw = [_load_archive(raw_c09_dir / f"seed{seed}_outputs.npz") for seed in (0, 1, 2)]
    _check_identity(raw, 24_462, EXPECTED_C09_PARENTS)
    raw_summary = load_json(raw_c09_dir / "summary.json")
    if raw_summary.get("overall_status") != "PASS_GSE_SLOPE_CORRECTIVE_C09_EVALUATION_V1":
        raise RuntimeError("raw C09 corrective evaluation is not a technical PASS")
    seed_rows = []
    for seed, row in enumerate(raw):
        target = np.asarray(row["target_slope_deg"], dtype=np.float32)
        current = np.asarray(row["current_slope_deg"], dtype=np.float32)
        prior = np.asarray(row["five_frame_prior_slope_deg"], dtype=np.float32)
        calibrated = apply_residual_scale(prior, row["corrected_slope_deg"], calibration.residual_scale)
        parents = np.asarray(row["parent_id"]).astype(str)
        metrics = _metrics(target, current, prior, calibrated, parents)
        metrics.update(
            {
                "seed": seed,
                "checkpoint_epoch": int(raw_summary["seeds"][seed]["checkpoint_epoch"]),
                "residual_scale": calibration.residual_scale,
                "raw_full_residual_mae_deg": float(raw_summary["seeds"][seed]["corrected_mae_deg"]),
            }
        )
        seed_rows.append(metrics)
        np.savez_compressed(
            output_dir / f"seed{seed}_outputs.npz",
            global_sequence_index=row["global_sequence_index"],
            parent_id=parents,
            target_slope_deg=target,
            current_slope_deg=current,
            five_frame_prior_slope_deg=prior,
            raw_full_residual_slope_deg=row["corrected_slope_deg"],
            corrected_slope_deg=calibrated,
            predicted_error_scale_deg=row["predicted_error_scale_deg"],
            residual_scale=np.asarray(calibration.residual_scale, dtype=np.float32),
        )
        write_json(output_dir / f"seed{seed}_metrics.json", metrics)

    summary = {
        "schema_version": "gse_slope_risk_calibrated_c09_evaluation_v2",
        "overall_status": PASS_STATUS,
        "validation_worlds": 10,
        "validation_sequences": 24_462,
        "validation_unique_frames": 32_678,
        "seeds": seed_rows,
        "world_evidence": raw_summary["world_evidence"],
        "risk_calibration": calibration_row,
        "raw_corrective_summary": str((raw_c09_dir / "summary.json").relative_to(PROJECT_ROOT)),
        "optimizer_steps": 0,
        "model_updates": 0,
        "c10_worlds_read": 0,
        "strict_test_worlds_read": 0,
        "mtare_worlds_read": 0,
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps({"overall_status": PASS_STATUS, "residual_scale": calibration.residual_scale}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
