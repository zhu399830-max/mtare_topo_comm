#!/usr/bin/env python3
"""Combine three all-row spatial-longitudinal center predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from _bootstrap import PROJECT_ROOT  # noqa: F401
from mtare_topo.evaluation.gse_spatial_center_projection import combine_seed_projections


EXPECTED = 188_126


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    for seed in range(3):
        parser.add_argument(f"--seed{seed}", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    reference = None
    vectors = []
    source_sha = {}
    for seed in range(3):
        path = getattr(args, f"seed{seed}").resolve()
        with np.load(path, allow_pickle=False) as archive:
            if str(archive["schema_version"].item()) != "gse_spatial_longitudinal_center_seed_all_v1":
                raise RuntimeError(f"spatial-center seed{seed} schema drift")
            if int(archive["seed"].item()) != seed:
                raise RuntimeError(f"spatial-center seed{seed} identity drift")
            current = {
                key: archive[key] for key in (
                    "global_sequence_index", "sensor_xyz_m", "route_local_basis",
                    "tangent_valid", "predicted_local_vector_m",
                )
            }
        if reference is None:
            reference = current
        elif not all(np.array_equal(current[key], reference[key]) for key in (
            "global_sequence_index", "sensor_xyz_m", "route_local_basis", "tangent_valid",
        )):
            raise RuntimeError("spatial-center seed population drift")
        vectors.append(current["predicted_local_vector_m"])
        source_sha[str(seed)] = _sha(path)
    assert reference is not None
    if reference["global_sequence_index"].shape != (EXPECTED,):
        raise RuntimeError("spatial-center ensemble row-count drift")
    combined = combine_seed_projections(
        np.stack(vectors), reference["sensor_xyz_m"], reference["route_local_basis"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        schema_version=np.asarray("gse_spatial_longitudinal_center_ensemble_all_v1"),
        global_sequence_index=reference["global_sequence_index"].astype(np.int64),
        seed_local_vector_m=np.stack(vectors).astype(np.float32),
        seed_center_xyz_m=combined["seed_center_xyz_m"],
        predicted_local_vector_m=combined["predicted_local_vector_m"],
        projected_center_xyz_m=combined["projected_center_xyz_m"],
        offset_std_m=combined["position_uncertainty_m"],
        sensor_xyz_m=reference["sensor_xyz_m"].astype(np.float32),
        route_local_basis=reference["route_local_basis"].astype(np.float32),
        tangent_valid=reference["tangent_valid"],
    )
    uncertainty = combined["position_uncertainty_m"]
    summary = {
        "schema_version": "gse_spatial_longitudinal_center_ensemble_all_v1",
        "rows": EXPECTED,
        "seed_source_sha256": source_sha,
        "output_sha256": _sha(args.output),
        "position_uncertainty_mean_m": float(np.mean(uncertainty)),
        "position_uncertainty_p95_m": float(np.percentile(uncertainty, 95)),
        "linear_identity_max_abs_m": float(combined["linear_identity_max_abs_m"]),
        "linear_identity_tolerance_m": 1e-4,
        "degenerate_route_rows": int(np.sum(~reference["tangent_valid"])),
        "optimizer_steps": 0,
        "model_updates": 0,
        "c09_worlds_read": 0,
        "c10_worlds_read": 0,
        "mtare_worlds_read": 0,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
