#!/usr/bin/env python3
"""Evaluate the frozen non-learning range-geometry baseline on validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.cano_sensor_smoke import MAX_RANGE_M
from mtare_topo.data.gse_training_dataset import GSESequenceDataset
from mtare_topo.governance import load_json, write_json
from mtare_topo.semantics.range_geometry_baseline import RangeGeometryBaseline


EXPECTED_DATASET_ID = "gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
EXPECTED_DATASET_STATUS = "PASS_GSE_DEDUPLICATED_DATASET_EXPORT_V1"
PASS_STATUS = "PASS_GSE_NONLEARNING_GEOMETRY_VALIDATION_V1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate(dataset_run: Path, output_dir: Path) -> dict:
    root = PROJECT_ROOT.resolve()
    dataset_run = dataset_run.resolve()
    output_dir = output_dir.resolve()
    dataset_run.relative_to(root)
    output_dir.relative_to(root)
    if dataset_run.name != EXPECTED_DATASET_ID:
        raise RuntimeError("unexpected GSE dataset source")
    state = load_json(dataset_run / "RUN_STATE.json")
    source_summary = load_json(dataset_run / "metrics/summary.json")
    if (
        state.get("state") != "COMPLETED"
        or state.get("overall_status") != EXPECTED_DATASET_STATUS
        or source_summary.get("overall_status") != EXPECTED_DATASET_STATUS
        or source_summary.get("split_totals", {}).get("validation", {}).get("sequences") != 24462
        or source_summary.get("strict_test_worlds_read") != 0
        or source_summary.get("mtare_worlds_read") != 0
    ):
        raise RuntimeError("sealed GSE dataset is not the required development PASS")
    if output_dir.exists():
        raise RuntimeError("non-learning geometry output already exists")
    output_dir.mkdir(parents=True)

    baseline = RangeGeometryBaseline()
    dataset = GSESequenceDataset(dataset_run, "validation", augment_azimuth=False)
    predicted = np.empty((len(dataset), 4), dtype=np.float32)
    target = np.empty_like(predicted)
    valid = np.empty((len(dataset), 4), dtype=np.uint8)
    axis_error = np.empty(len(dataset), dtype=np.float32)
    global_indices = np.empty(len(dataset), dtype=np.int64)
    observation_ids = np.empty(len(dataset), dtype="U128")
    parent_ids = np.empty(len(dataset), dtype="U64")
    started = time.monotonic()
    for index in range(len(dataset)):
        sample = dataset[index]
        result = baseline.predict(
            sample["student"][-1, 0] * MAX_RANGE_M,
            sample["student"][-1, 1],
        )
        predicted[index] = (
            result["width_m"],
            result["height_m"],
            result["slope_deg"],
            result["curvature_per_m"],
        )
        target[index] = (
            sample["targets"]["width_m"],
            sample["targets"]["height_m"],
            sample["targets"]["slope_deg"],
            sample["targets"]["curvature_per_m"],
        )
        valid[index] = sample["targets"]["geometry_valid_mask"]
        estimate_axis = np.asarray(result["local_axis"], dtype=np.float64)
        teacher_axis = np.asarray(sample["targets"]["local_axis"], dtype=np.float64)
        cosine = float(np.clip(estimate_axis @ teacher_axis, -1.0, 1.0))
        axis_error[index] = math.degrees(math.acos(cosine))
        global_indices[index] = int(sample["global_sequence_index"])
        observation_ids[index] = str(sample["observation_id"])
        parent_ids[index] = str(sample["parent_id"])
    if len(dataset) != 24462 or len(np.unique(global_indices)) != len(dataset) or len(np.unique(parent_ids)) != 10:
        raise RuntimeError("validation baseline population drift")
    if not np.all(np.isfinite(predicted)) or not np.all(np.isfinite(axis_error)):
        raise RuntimeError("non-learning geometry baseline produced non-finite outputs")
    absolute = np.abs(predicted - target)
    counts = valid.sum(axis=0)
    if np.any(counts == 0):
        raise RuntimeError("non-learning baseline has an empty geometry metric")
    names = ("width_m", "height_m", "slope_deg", "curvature_per_m")
    geometry_mae = {
        name: float((absolute[:, index] * valid[:, index]).sum() / counts[index])
        for index, name in enumerate(names)
    }
    per_parent = []
    for parent_id in sorted(np.unique(parent_ids).tolist()):
        selected = parent_ids == parent_id
        parent_valid = valid[selected]
        parent_counts = parent_valid.sum(axis=0)
        if not selected.any() or np.any(parent_counts == 0):
            raise RuntimeError(f"non-learning geometry parent has an empty metric: {parent_id}")
        parent_absolute = absolute[selected]
        per_parent.append(
            {
                "parent_id": parent_id,
                "frames": int(selected.sum()),
                "geometry_mae": {
                    name: float(
                        (parent_absolute[:, field] * parent_valid[:, field]).sum()
                        / parent_counts[field]
                    )
                    for field, name in enumerate(names)
                },
                "geometry_valid_count": {
                    name: int(parent_counts[field]) for field, name in enumerate(names)
                },
                "axis": {
                    "mean_angular_error_deg": float(axis_error[selected].mean()),
                    "p95_angular_error_deg": float(np.quantile(axis_error[selected], 0.95)),
                },
            }
        )
    if len(per_parent) != 10 or sum(row["frames"] for row in per_parent) != len(dataset):
        raise RuntimeError("non-learning per-parent diagnostics do not cover C09")
    np.savez_compressed(
        output_dir / "validation_predictions.npz",
        predicted_geometry=predicted,
        target_geometry=target,
        geometry_valid_mask=valid,
        axis_error_deg=axis_error,
        global_sequence_index=global_indices,
        observation_id=observation_ids,
        parent_id=parent_ids,
    )
    summary = {
        "schema_version": "gse_nonlearning_geometry_validation_v1",
        "overall_status": PASS_STATUS,
        "method": "single-current-scan horizontal PCA plus robust five-section cross-section/centreline polynomial estimator",
        "config": baseline.config.to_dict(),
        "validation_frames": len(dataset),
        "validation_worlds": int(len(np.unique(parent_ids))),
        "geometry_mae": geometry_mae,
        "geometry_valid_count": {name: int(counts[index]) for index, name in enumerate(names)},
        "per_parent_diagnostic": {
            "selection_effect": "NONE_ALL_TEN_VALIDATION_PARENTS_AND_ALL_GEOMETRY_FIELDS",
            "parents": per_parent,
        },
        "axis": {
            "mean_angular_error_deg": float(axis_error.mean()),
            "p95_angular_error_deg": float(np.quantile(axis_error, 0.95)),
        },
        "duration_seconds": time.monotonic() - started,
        "source_dataset_run": str(dataset_run.relative_to(root)),
        "source_dataset_seal_sha256": _sha256(dataset_run / "artifacts/evidence_sha256.txt"),
        "strict_test_worlds_read": 0,
        "mtare_worlds_read": 0,
        "model_inference_frames": 0,
        "optimizer_steps": 0,
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-run", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.dataset_run, args.output_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
