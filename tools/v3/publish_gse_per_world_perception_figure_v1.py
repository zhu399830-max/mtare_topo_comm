#!/usr/bin/env python3
"""Publish all-parent C09 perception diagnostics from one sealed PASS run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import PROJECT_ROOT
from mtare_topo.evaluation.gse_evidence_integrity import verify_complete_run_seal
from mtare_topo.governance import load_json, write_json


EXPECTED_RUN_ID = "gate3_20260824_gse_perception_validation_v1_seed0"
EXPECTED_STATUS = "PASS_GSE_PERCEPTION_VALIDATION_V1"
FIGURE_ID = "gse_per_world_perception"
FIELDS = ("width_m", "height_m", "slope_deg", "curvature_per_m")
EXPECTED_GSE_SELECTION_EFFECT = (
    "NONE_REPLAY_OF_GLOBAL_FROZEN_EVENT_POINT_AND_ALL_GEOMETRY_FIELDS"
)
EXPECTED_EVENT_SELECTION_EFFECT = "NONE_REPLAY_OF_FROZEN_EVENT_POINT"
EXPECTED_BASELINE_SELECTION_EFFECT = "NONE_ALL_TEN_VALIDATION_PARENTS_AND_ALL_GEOMETRY_FIELDS"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _indexed(records: list[dict], *, context: str) -> dict[str, dict]:
    indexed = {str(record["parent_id"]): record for record in records}
    if len(indexed) != len(records):
        raise RuntimeError(f"duplicate parent in {context}")
    return indexed


def _collect(run_dir: Path) -> dict:
    calibration = [load_json(run_dir / f"artifacts/calibration/seed{seed}_summary.json") for seed in range(3)]
    exit_only = load_json(run_dir / "artifacts/exit_only_baseline/summary.json")
    geometry = load_json(run_dir / "artifacts/nonlearning_geometry/summary.json")
    if [int(row["seed"]) for row in exit_only.get("per_seed", [])] != [0, 1, 2]:
        raise RuntimeError("per-world figure requires exact M1D seeds 0/1/2")
    gse_by_seed = [
        _indexed(result["per_parent_diagnostic"]["parents"], context=f"GSE seed {seed}")
        for seed, result in enumerate(calibration)
    ]
    m1d_by_seed = [
        _indexed(result["per_parent_event"], context=f"M1D seed {seed}")
        for seed, result in enumerate(exit_only["per_seed"])
    ]
    baseline = _indexed(geometry["per_parent_diagnostic"]["parents"], context="geometry baseline")
    if geometry["per_parent_diagnostic"].get("selection_effect") != EXPECTED_BASELINE_SELECTION_EFFECT:
        raise RuntimeError("non-learning per-parent selection-effect contract drift")
    frozen_points = []
    for seed, result in enumerate(calibration):
        diagnostic = result["per_parent_diagnostic"]
        if diagnostic.get("selection_effect") != EXPECTED_GSE_SELECTION_EFFECT:
            raise RuntimeError(f"GSE seed {seed} per-parent selection-effect contract drift")
        frozen_points.append(
            {
                "seed": seed,
                "temperature": float(result["event"]["temperature"]["temperature"]),
                "threshold": float(result["event"]["rejection_selection"]["threshold"]),
            }
        )
    parent_sets = [set(baseline), *(set(value) for value in gse_by_seed), *(set(value) for value in m1d_by_seed)]
    if len(baseline) != 10 or any(value != parent_sets[0] for value in parent_sets[1:]):
        raise RuntimeError("per-world sources do not contain the same exact ten C09 parents")

    rows = []
    for parent_id in sorted(baseline):
        baseline_frames = int(baseline[parent_id]["frames"])
        for seed in range(3):
            gse = gse_by_seed[seed][parent_id]
            m1d = m1d_by_seed[seed][parent_id]
            if int(gse["frames"]) != baseline_frames or int(m1d["frames"]) != baseline_frames:
                raise RuntimeError(f"per-world frame population mismatch: {parent_id}")
            event = gse["event"]
            frozen = frozen_points[seed]
            if (
                event.get("selection_effect") != EXPECTED_EVENT_SELECTION_EFFECT
                or float(event["temperature"]) != frozen["temperature"]
                or float(event["threshold"]) != frozen["threshold"]
            ):
                raise RuntimeError(f"per-world event point differs from the global frozen point: {parent_id}/seed{seed}")
            field_improvements = {}
            field_valid_counts = {}
            for field in FIELDS:
                reference = float(baseline[parent_id]["geometry_mae"][field])
                learned = float(gse["geometry_mae"][field])
                reference_count = int(baseline[parent_id]["geometry_valid_count"][field])
                learned_count = int(gse["geometry_valid_count"][field])
                if not np.isfinite(reference) or reference <= 0.0 or not np.isfinite(learned):
                    raise RuntimeError(f"invalid per-world geometry metric: {parent_id}/{field}")
                if reference_count <= 0 or learned_count != reference_count:
                    raise RuntimeError(f"paired geometry valid-count mismatch: {parent_id}/{field}/seed{seed}")
                field_improvements[field] = float((reference - learned) / reference)
                field_valid_counts[field] = reference_count
            rows.append(
                {
                    "parent_id": parent_id,
                    "seed": seed,
                    "frames": baseline_frames,
                    "gse_event_macro_f1": float(gse["event"]["macro_f1"]),
                    "m1d_event_macro_f1": float(m1d["event"]["macro_f1"]),
                    "event_macro_f1_gain": float(
                        gse["event"]["macro_f1"] - m1d["event"]["macro_f1"]
                    ),
                    "geometry_relative_improvement_macro": float(np.mean(list(field_improvements.values()))),
                    "geometry_relative_improvement": field_improvements,
                    "geometry_valid_count": field_valid_counts,
                }
            )
    if len(rows) != 30 or sum(row["frames"] for row in rows) != 3 * 24462:
        raise RuntimeError("per-world figure does not cover 10 parents x 3 seeds x 24,462 frames")
    return {
        "schema_version": "gse_per_world_perception_diagnostic_v1",
        "selection_effect": "NONE_ALL_TEN_C09_PARENTS_ALL_THREE_PREDECLARED_SEEDS_ALL_FOUR_GEOMETRY_FIELDS",
        "parents": sorted(baseline),
        "seeds": [0, 1, 2],
        "geometry_fields": list(FIELDS),
        "frozen_event_points": frozen_points,
        "rows": rows,
        "strict_test_worlds_read": 0,
        "mtare_worlds_read": 0,
    }


def publish(run_dir: Path, destination: Path) -> dict:
    root = PROJECT_ROOT.resolve()
    run_dir = run_dir.resolve()
    destination = destination.resolve()
    run_dir.relative_to(root)
    destination.relative_to(root)
    state = load_json(run_dir / "RUN_STATE.json")
    summary = load_json(run_dir / "metrics/summary.json")
    gate = load_json(run_dir / "metrics/perception_gate.json")
    if (
        run_dir.name != EXPECTED_RUN_ID
        or state.get("state") != "COMPLETED"
        or state.get("overall_status") != EXPECTED_STATUS
        or summary.get("overall_status") != EXPECTED_STATUS
        or gate.get("passed") is not True
        or gate.get("strict_test_worlds_read") != 0
        or gate.get("mtare_worlds_read") != 0
    ):
        raise RuntimeError("per-world perception figure requires the sealed C09 PASS")
    verify_complete_run_seal(root, run_dir, EXPECTED_STATUS)
    sources = [
        run_dir / "RUN_STATE.json",
        run_dir / "metrics/summary.json",
        run_dir / "metrics/perception_gate.json",
        run_dir / "artifacts/exit_only_baseline/summary.json",
        run_dir / "artifacts/nonlearning_geometry/summary.json",
        *(run_dir / f"artifacts/calibration/seed{seed}_summary.json" for seed in range(3)),
    ]
    seal = run_dir / "artifacts/evidence_sha256.txt"
    sealed = {}
    for line in seal.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        sealed[relative] = expected
    for path in sources:
        relative = str(path.relative_to(root))
        if sealed.get(relative) != _sha256(path):
            raise RuntimeError(f"per-world source is absent from the run seal: {relative}")

    suffixes = (".png", ".pdf", ".svg", ".csv", "_source.json", "_provenance.json", "_sha256.txt")
    targets = [destination / f"{FIGURE_ID}{suffix}" for suffix in suffixes]
    if any(path.exists() for path in targets):
        raise RuntimeError("per-world figure destination exists; refusing overwrite")
    destination.mkdir(parents=True, exist_ok=True)
    diagnostic = _collect(run_dir)
    rows = diagnostic["rows"]

    with (destination / f"{FIGURE_ID}.csv").open("w", encoding="utf-8", newline="") as stream:
        fieldnames = (
            "parent_id",
            "seed",
            "frames",
            "gse_event_macro_f1",
            "m1d_event_macro_f1",
            "event_macro_f1_gain",
            "geometry_relative_improvement_macro",
            *(f"{field}_relative_improvement" for field in FIELDS),
            *(f"{field}_valid_count" for field in FIELDS),
        )
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{name: row[name] for name in fieldnames[:7]},
                    **{
                        f"{field}_relative_improvement": row["geometry_relative_improvement"][field]
                        for field in FIELDS
                    },
                    **{
                        f"{field}_valid_count": row["geometry_valid_count"][field]
                        for field in FIELDS
                    },
                }
            )

    parents = diagnostic["parents"]
    event = np.asarray(
        [[next(row["event_macro_f1_gain"] for row in rows if row["parent_id"] == parent and row["seed"] == seed) for seed in range(3)] for parent in parents]
    )
    geometry = np.asarray(
        [[next(row["geometry_relative_improvement_macro"] for row in rows if row["parent_id"] == parent and row["seed"] == seed) for seed in range(3)] for parent in parents]
    )
    bound = max(0.05, float(np.max(np.abs(np.concatenate((event.ravel(), geometry.ravel()))))))
    figure, axes = plt.subplots(1, 2, figsize=(8.0, 5.2), constrained_layout=True)
    labels = [parent.replace("_flat_tree_small", "") for parent in parents]
    for axis, values, title, unit in (
        (axes[0], event, "Structural-event gain over exit-only", "Δ macro-F1"),
        (axes[1], geometry, "Geometry gain over deterministic", "relative MAE improvement"),
    ):
        image = axis.imshow(values, cmap="RdBu", vmin=-bound, vmax=bound, aspect="auto")
        axis.set_xticks(range(3), ["seed 0", "seed 1", "seed 2"])
        axis.set_yticks(range(10), labels)
        axis.set_title(title, fontsize=10)
        for row_index in range(10):
            for seed in range(3):
                value = values[row_index, seed]
                axis.text(seed, row_index, f"{value:+.2f}", ha="center", va="center", fontsize=7)
        colorbar = figure.colorbar(image, ax=axis, shrink=0.84)
        colorbar.set_label(unit, fontsize=8)
    axes[0].set_ylabel("Unseen C09 topology parent")
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(destination / f"{FIGURE_ID}.{suffix}", dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    write_json(destination / f"{FIGURE_ID}_source.json", diagnostic)
    write_json(
        destination / f"{FIGURE_ID}_provenance.json",
        {
            "schema_version": "gse_paper_figure_provenance_v1",
            "figure_id": FIGURE_ID,
            "source_run": str(run_dir.relative_to(root)),
            "source_seal": str(seal.relative_to(root)),
            "source_seal_sha256": _sha256(seal),
            "selection_rule": diagnostic["selection_effect"],
            "strict_test_worlds_read": 0,
            "mtare_worlds_read": 0,
            "manual_value_entry": False,
            "generator": str(Path(__file__).resolve().relative_to(root)),
            "generator_sha256": _sha256(Path(__file__).resolve()),
            "source_files": {str(path.relative_to(run_dir)): _sha256(path) for path in sources},
        },
    )
    manifest = destination / f"{FIGURE_ID}_sha256.txt"
    manifest.write_text(
        "".join(f"{_sha256(path)}  {path.relative_to(root)}\n" for path in sorted(targets) if path != manifest),
        encoding="utf-8",
    )
    return {"figure_id": FIGURE_ID, "published_files": len(targets), "manifest_sha256": _sha256(manifest)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--destination", type=Path, default=PROJECT_ROOT / "docs/figures/gse_graph")
    args = parser.parse_args()
    print(json.dumps(publish(args.run_dir, args.destination), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
