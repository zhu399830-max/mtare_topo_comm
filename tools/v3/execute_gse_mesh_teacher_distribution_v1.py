#!/usr/bin/env python3
"""Formal train-only native-mesh teacher distribution proof for GSE-Graph."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.gse_mesh_teacher_inventory import world_mesh_teacher_inventory
from mtare_topo.governance import load_json, write_json


EVENT_NAMES = ("corridor", "junction", "terminal", "turn", "geometry_transition")
EXPECTED_TRAIN_WORLDS = 80
EXPECTED_TRAIN_EDGES = 8039
EXPECTED_TRAIN_DIRECTED_TRAVERSALS = 16078
EXPECTED_TRAIN_SEQUENCES = 188126
EXPECTED_TRAIN_UNIQUE_FRAMES = 252430
MINIMUM_EVENT_COUNT = 500
MINIMUM_GEOMETRY_VALID_FRACTION = 0.98
MAXIMUM_NON_JUNCTION_MISSING_FRACTION = 0.005


def _scene(mesh_path: Path) -> o3d.t.geometry.RaycastingScene:
    mesh = o3d.io.read_triangle_mesh(str(mesh_path), enable_post_processing=False)
    if not mesh.has_vertices() or not mesh.has_triangles():
        raise RuntimeError(f"empty mesh: {mesh_path}")
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene


def _batch_cast(scene: o3d.t.geometry.RaycastingScene):
    def cast(origins: np.ndarray, directions: np.ndarray) -> np.ndarray:
        expanded = np.broadcast_to(origins[:, None, :], directions.shape)
        rays = np.concatenate((expanded, directions), axis=2).astype(np.float32, copy=False)
        return (
            scene.cast_rays(o3d.core.Tensor(rays.reshape(-1, 6)))["t_hit"]
            .numpy()
            .reshape(directions.shape[:2])
        )

    return cast


def _plot_distribution(summary: dict, png_path: Path, pdf_path: Path) -> None:
    counts = summary["event_counts"]
    missing = summary["missing_continuous_geometry_by_event"]
    labels = ["corridor", "junction", "terminal", "turn", "geometry\ntransition"]
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2"]
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    axes[0].bar(labels, [counts[name] for name in EVENT_NAMES], color=colors)
    axes[0].set_ylabel("five-frame training sequences")
    axes[0].set_title("GSE structural-event teacher distribution")
    axes[0].tick_params(axis="x", labelrotation=18)
    for index, name in enumerate(EVENT_NAMES):
        axes[0].text(index, counts[name], f"{counts[name]:,}", ha="center", va="bottom", fontsize=8)

    axes[1].bar(labels, [missing[name] for name in EVENT_NAMES], color=colors)
    axes[1].set_ylabel("masked width/height targets")
    axes[1].set_title("Non-unique or incomplete mesh cross-sections")
    axes[1].tick_params(axis="x", labelrotation=18)
    for index, name in enumerate(EVENT_NAMES):
        axes[1].text(index, missing[name], f"{missing[name]:,}", ha="center", va="bottom", fontsize=8)
    figure.suptitle(
        f"Train-only native-mesh teacher proof — geometry valid {summary['geometry_valid_fraction']:.2%}"
    )
    figure.savefig(png_path, dpi=180)
    figure.savefig(pdf_path)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--mesh-root", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    registry_path = args.registry.resolve()
    mesh_root = args.mesh_root.resolve()
    started = time.monotonic()
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (run_dir / "previews").mkdir(parents=True, exist_ok=True)

    registry = load_json(registry_path)
    schema = registry["sampling_contract"]["row_schema"]
    rows = [dict(zip(schema, row, strict=True)) if isinstance(row, list) else dict(row) for row in registry["rows"]]
    train_rows = sorted((row for row in rows if row["split"] == "train"), key=lambda row: row["parent_id"])
    if len(train_rows) != EXPECTED_TRAIN_WORLDS:
        raise RuntimeError(f"expected {EXPECTED_TRAIN_WORLDS} train worlds, found {len(train_rows)}")
    if any(str(row["parent_id"]).endswith("_C09") or str(row["parent_id"]).endswith("_C10") for row in train_rows):
        raise RuntimeError("validation or strict-test parent leaked into train-only audit")

    worlds: list[dict] = []
    event_counts: Counter[str] = Counter()
    missing_counts: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    world_stream = (run_dir / "artifacts/world_teacher_inventory.jsonl").open("w", encoding="utf-8")
    for index, row in enumerate(train_rows, start=1):
        parent_id = str(row["parent_id"])
        primary = mesh_root / parent_id / "primary"
        scene = _scene(primary / "mesh.obj")
        result = world_mesh_teacher_inventory(
            parent_id=parent_id,
            split="train",
            graph=load_json(primary / "graph.json"),
            spline_document=load_json(primary / "splines.json"),
            geometry_parameters=load_json(primary / "geometry_parameters.json"),
            cast_distances=_batch_cast(scene),
        )
        worlds.append(result)
        world_stream.write(json.dumps(result, separators=(",", ":"), sort_keys=True) + "\n")
        event_counts.update(result["event_counts"])
        missing_counts.update(result["missing_continuous_geometry_by_event"])
        for key in (
            "edge_count",
            "directed_traversal_count",
            "sequence_count",
            "labelled_sequence_count",
            "continuous_geometry_target_count",
            "missing_continuous_geometry_target_count",
            "complete_mesh_frame_count",
            "incomplete_mesh_frame_count",
            "transition_candidate_count",
        ):
            totals[key] += int(result[key])
        print(
            json.dumps(
                {
                    "world": parent_id,
                    "index": index,
                    "of": len(train_rows),
                    "sequences": result["sequence_count"],
                    "geometry_targets": result["continuous_geometry_target_count"],
                    "events": result["event_counts"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    world_stream.close()

    geometry_valid_fraction = totals["continuous_geometry_target_count"] / totals["sequence_count"]
    non_junction_missing = sum(
        missing_counts[name] for name in EVENT_NAMES if name != "junction"
    )
    non_junction_sequences = totals["sequence_count"] - event_counts["junction"]
    non_junction_missing_fraction = non_junction_missing / non_junction_sequences
    checks = {
        "exact_train_world_count": len(worlds) == EXPECTED_TRAIN_WORLDS,
        "exact_train_edge_count": totals["edge_count"] == EXPECTED_TRAIN_EDGES,
        "exact_train_directed_traversal_count": (
            totals["directed_traversal_count"] == EXPECTED_TRAIN_DIRECTED_TRAVERSALS
        ),
        "exact_train_sequence_count": totals["sequence_count"] == EXPECTED_TRAIN_SEQUENCES,
        "exact_train_unique_frame_count": (
            totals["complete_mesh_frame_count"] + totals["incomplete_mesh_frame_count"]
            == EXPECTED_TRAIN_UNIQUE_FRAMES
        ),
        "all_sequences_have_event_label": totals["labelled_sequence_count"] == totals["sequence_count"],
        "all_five_event_classes_have_at_least_500_train_sequences": all(
            event_counts[name] >= MINIMUM_EVENT_COUNT for name in EVENT_NAMES
        ),
        "geometry_valid_fraction_at_least_0p98": geometry_valid_fraction >= MINIMUM_GEOMETRY_VALID_FRACTION,
        "non_junction_missing_fraction_at_most_0p005": (
            non_junction_missing_fraction <= MAXIMUM_NON_JUNCTION_MISSING_FRACTION
        ),
        "zero_validation_strict_test_mtare_reads": True,
    }
    passed = all(checks.values())
    summary = {
        "schema_version": "gse_mesh_teacher_distribution_v1",
        "overall_status": "PASS_GSE_MESH_TEACHER_DISTRIBUTION_V1" if passed else "FAIL_GSE_MESH_TEACHER_DISTRIBUTION_V1",
        "split": "train",
        "train_worlds_read": len(worlds),
        "validation_worlds_read": 0,
        "strict_test_worlds_read": 0,
        "mtare_worlds_read": 0,
        "totals": dict(totals),
        "event_counts": {name: int(event_counts[name]) for name in EVENT_NAMES},
        "missing_continuous_geometry_by_event": {
            name: int(missing_counts[name]) for name in EVENT_NAMES
        },
        "geometry_valid_fraction": geometry_valid_fraction,
        "non_junction_missing_fraction": non_junction_missing_fraction,
        "acceptance_thresholds": {
            "minimum_event_count": MINIMUM_EVENT_COUNT,
            "minimum_geometry_valid_fraction": MINIMUM_GEOMETRY_VALID_FRACTION,
            "maximum_non_junction_missing_fraction": MAXIMUM_NON_JUNCTION_MISSING_FRACTION,
        },
        "checks": checks,
        "duration_seconds": time.monotonic() - started,
    }
    with (run_dir / "artifacts/event_distribution.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("event", "sequence_count", "masked_width_height_count"))
        for name in EVENT_NAMES:
            writer.writerow((name, event_counts[name], missing_counts[name]))
    write_json(run_dir / "metrics/summary.json", summary)
    _plot_distribution(
        summary,
        run_dir / "previews/gse_teacher_distribution.png",
        run_dir / "previews/gse_teacher_distribution.pdf",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not passed:
        raise RuntimeError("GSE native-mesh teacher distribution proof failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
