#!/usr/bin/env python3
"""Execute the five-topology zero-training Cano CPU sensor contract pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import traceback
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from _bootstrap import PROJECT_ROOT
from cano_five_topology_support import (
    _per_tunnel_axis,
    _source_precheck,
    build_mesh_generator,
    build_parent,
    canonical_exports,
    topology_identity_preview,
)
from mtare_topo.data.cano_contract_pilot import (
    PARENT_REGISTRY,
    VIEW_OFFSETS_DEG,
    canonical_document_hash,
    graph_family_metrics,
    match_headings,
    select_canonical_anchors,
    spline_arrays,
)
from mtare_topo.data.cano_sensor_smoke import (
    AZIMUTH_COLUMNS,
    ELEVATION_DEG,
    MAX_RANGE_M,
    NEAR_RANGE_M,
    lidar_local_directions,
    structural_label,
    world_directions,
)
from mtare_topo.governance import write_json
from mtare_topo.semantics.range_exit_baseline import RangeExitBaseline


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_hash(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _scene(mesh: o3d.geometry.TriangleMesh) -> o3d.t.geometry.RaycastingScene:
    result = o3d.t.geometry.RaycastingScene()
    result.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return result


def _cast(
    scene: o3d.t.geometry.RaycastingScene,
    origin: np.ndarray,
    directions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    origins = np.broadcast_to(np.asarray(origin, dtype=np.float32), directions.shape)
    rays = np.concatenate((origins, np.asarray(directions, dtype=np.float32)), axis=-1)
    raw = scene.cast_rays(o3d.core.Tensor(rays.reshape(-1, 6)))["t_hit"].numpy()
    raw = raw.reshape(directions.shape[:-1])
    valid = np.isfinite(raw) & (raw >= NEAR_RANGE_M) & (raw <= MAX_RANGE_M)
    return np.where(valid, raw, MAX_RANGE_M).astype(np.float32), valid.astype(np.uint8)


def _validate_scan(range_m: np.ndarray, valid_mask: np.ndarray) -> None:
    expected_shape = (len(ELEVATION_DEG), AZIMUTH_COLUMNS)
    if range_m.shape != expected_shape or valid_mask.shape != expected_shape:
        raise RuntimeError(
            f"scan shape contract failed: {range_m.shape}/{valid_mask.shape} != {expected_shape}"
        )
    if range_m.dtype != np.float32 or valid_mask.dtype != np.uint8:
        raise RuntimeError(
            f"scan dtype contract failed: {range_m.dtype}/{valid_mask.dtype}"
        )
    valid = valid_mask.astype(bool)
    if np.any(~np.isfinite(range_m)):
        raise RuntimeError("scan contains non-finite canonical ranges")
    if np.any((range_m[valid] < NEAR_RANGE_M) | (range_m[valid] > MAX_RANGE_M)):
        raise RuntimeError("valid scan range is outside the frozen near/far contract")
    if np.any(range_m[~valid] != MAX_RANGE_M):
        raise RuntimeError("invalid scan cells are not exactly the frozen far range")


def _horizontal_directions() -> np.ndarray:
    azimuth = np.radians(np.arange(AZIMUTH_COLUMNS, dtype=np.float32) * 0.5)
    return np.stack(
        (np.cos(azimuth), np.sin(azimuth), np.zeros(AZIMUTH_COLUMNS)), axis=-1
    ).astype(np.float32)


def _representative_branch_points(label: dict[str, Any]) -> list[np.ndarray]:
    points = []
    for heading in label["headings_world_deg"]:
        selected = min(
            label["raw_intersections"],
            key=lambda item: abs(
                (float(item["heading_world_deg"]) - float(heading) + 180.0) % 360.0
                - 180.0
            ),
        )
        points.append(np.asarray(selected["xyz"], dtype=np.float64))
    return points


def _plot_world(
    mesh: o3d.geometry.TriangleMesh,
    splines_document: dict[str, Any],
    anchors: list[dict[str, Any]],
    parent_id: str,
    destination: Path,
) -> None:
    vertices = np.asarray(mesh.vertices)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    for axis, dimensions, labels in (
        (axes[0], (0, 1), ("X [m]", "Y [m]")),
        (axes[1], (0, 2), ("X [m]", "Z [m]")),
    ):
        axis.scatter(
            vertices[:, dimensions[0]],
            vertices[:, dimensions[1]],
            s=0.1,
            c="#888888",
            alpha=0.15,
            rasterized=True,
        )
        for tunnel in splines_document["tunnels"]:
            points = np.asarray(tunnel["points"], dtype=np.float64)
            axis.plot(
                points[:, dimensions[0]],
                points[:, dimensions[1]],
                color="#202020",
                linewidth=1.0,
            )
        anchor_points = np.asarray([item["axis_xyz_m"] for item in anchors])
        roles = [item["objective_structure"]["topological_role"] for item in anchors]
        colors = {"terminal": "#D55E00", "corridor": "#0072B2", "junction": "#009E73"}
        for role in colors:
            mask = np.asarray([value == role for value in roles])
            if np.any(mask):
                axis.scatter(
                    anchor_points[mask, dimensions[0]],
                    anchor_points[mask, dimensions[1]],
                    s=22,
                    c=colors[role],
                    label=role,
                    edgecolors="white",
                    linewidths=0.25,
                )
        axis.set_xlabel(labels[0])
        axis.set_ylabel(labels[1])
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, linewidth=0.25, alpha=0.35)
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle(f"{parent_id} | complete native perception mesh, splines and all 50 anchors")
    fig.savefig(destination, dpi=170)
    plt.close(fig)


def _plot_contact_page(
    records: list[dict[str, Any]],
    ranges: np.ndarray,
    page_index: int,
    destination: Path,
) -> None:
    fig, axes = plt.subplots(5, 5, figsize=(18, 13), constrained_layout=True)
    for local_index, axis in enumerate(axes.flat):
        record = records[local_index]
        image = ranges[local_index]
        axis.imshow(image, origin="lower", aspect="auto", vmin=NEAR_RANGE_M, vmax=MAX_RANGE_M, cmap="viridis")
        for heading in record["truth_headings_robot_deg"]:
            axis.axvline(float(heading) * 2.0, color="#00FF66", linewidth=0.8)
        for heading in record["predicted_headings_robot_deg"]:
            axis.axvline(float(heading) * 2.0, color="#FF00CC", linewidth=0.65, linestyle="--")
        axis.set_title(
            f"{record['parent_id']} {record['anchor_id']} {record['view_offset_deg']:+.0f}°\n"
            f"GT/P={record['truth_branch_count']}/{record['predicted_branch_count']}",
            fontsize=6.2,
        )
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle(
        f"Five-topology contract page {page_index + 1:02d}/30 | green=objective branch, magenta=frozen rule",
        fontsize=12,
    )
    fig.savefig(destination, dpi=125)
    plt.close(fig)


def _plot_summary(parent_metrics: list[dict[str, Any]], destination: Path) -> None:
    names = [item["parent_id"].replace("P0", "P") for item in parent_metrics]
    precision = [item["branch_metrics"]["precision"] for item in parent_metrics]
    recall = [item["branch_metrics"]["recall"] for item in parent_metrics]
    f1 = [item["branch_metrics"]["f1"] for item in parent_metrics]
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    axes[0].bar(x - 0.25, precision, 0.25, label="precision")
    axes[0].bar(x, recall, 0.25, label="recall")
    axes[0].bar(x + 0.25, f1, 0.25, label="F1")
    axes[0].set_xticks(x, names, rotation=20, ha="right")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("branch metric @20°")
    axes[0].legend()
    axes[0].grid(True, axis="y", linewidth=0.25, alpha=0.4)
    role_names = ("terminal", "corridor", "junction")
    bottoms = np.zeros(len(names))
    colors = ("#D55E00", "#0072B2", "#009E73")
    for role, color in zip(role_names, colors):
        values = np.asarray([item["objective_structure_counts"].get(role, 0) for item in parent_metrics])
        axes[1].bar(x, values, bottom=bottoms, label=role, color=color)
        bottoms += values
    axes[1].set_xticks(x, names, rotation=20, ha="right")
    axes[1].set_ylabel("canonical anchors (50 per parent)")
    axes[1].legend()
    axes[1].grid(True, axis="y", linewidth=0.25, alpha=0.4)
    fig.suptitle("Frozen single-world range rule replayed without tuning on five new topology families")
    fig.savefig(destination, dpi=170)
    plt.close(fig)


def _aggregate_branch(records: list[dict[str, Any]]) -> dict[str, Any]:
    matched = sum(int(item["match"]["matched"]) for item in records)
    predicted = sum(int(item["match"]["predicted"]) for item in records)
    truth = sum(int(item["match"]["truth"]) for item in records)
    errors = [value for item in records for value in item["match"]["angular_errors_deg"]]
    precision = matched / max(1, predicted)
    recall = matched / max(1, truth)
    return {
        "matched": matched,
        "predicted": predicted,
        "truth": truth,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(1e-12, precision + recall),
        "mean_matched_angular_error_deg": float(np.mean(errors)) if errors else None,
        "exact_branch_count_fraction": float(np.mean([item["truth_branch_count"] == item["predicted_branch_count"] for item in records])),
    }


def execute(run_dir: Path) -> dict[str, Any]:
    source_before = _source_precheck()
    worlds_root = run_dir / "artifacts/worlds"
    shards_root = run_dir / "artifacts/shards"
    maps_root = run_dir / "previews/world_maps"
    pages_root = run_dir / "previews/contact_pages"
    for path in (worlds_root, shards_root, maps_root, pages_root):
        path.mkdir(parents=True, exist_ok=False)

    baseline = RangeExitBaseline()
    local_directions = lidar_local_directions()
    horizontal = _horizontal_directions()
    all_observation_records: list[dict[str, Any]] = []
    all_contact_ranges: list[np.ndarray] = []
    parent_metrics: list[dict[str, Any]] = []
    replay_records = []

    for registry in PARENT_REGISTRY:
        parent_id = registry["parent_id"]
        world_dir = worlds_root / parent_id
        world_dir.mkdir()
        network, nodes, generation = build_parent(parent_id, registry["topology_seed"])
        graph, splines_document = canonical_exports(network)
        graph_hash = canonical_document_hash(graph)
        spline_hash = canonical_document_hash(splines_document)
        family = graph_family_metrics(parent_id, graph, splines_document)
        if not family["passed"]:
            raise RuntimeError(f"{parent_id}: graph family contract failed: {family}")

        replay_network, replay_nodes, replay_generation = build_parent(
            parent_id, registry["topology_seed"]
        )
        replay_graph, replay_splines = canonical_exports(replay_network)
        replay_graph_hash = canonical_document_hash(replay_graph)
        replay_spline_hash = canonical_document_hash(replay_splines)
        replay_match = graph_hash == replay_graph_hash and spline_hash == replay_spline_hash
        if not replay_match:
            raise RuntimeError(f"{parent_id}: topology replay hash mismatch")
        _, alternate_geometry = build_mesh_generator(
            replay_network, replay_nodes, parent_id, registry["geometry_seed"] + 1
        )
        after_alt_graph, after_alt_splines = canonical_exports(replay_network)
        geometry_seed_separated = (
            canonical_document_hash(after_alt_graph) == replay_graph_hash
            and canonical_document_hash(after_alt_splines) == replay_spline_hash
        )
        topology_seed_sensitive = topology_identity_preview(
            parent_id, registry["topology_seed"]
        ) != topology_identity_preview(parent_id, registry["topology_seed"] + 1)
        if not geometry_seed_separated or not topology_seed_sensitive:
            raise RuntimeError(f"{parent_id}: seed-separation control failed")
        replay_records.append(
            {
                "parent_id": parent_id,
                "primary_graph_sha256": graph_hash,
                "replay_graph_sha256": replay_graph_hash,
                "primary_spline_sha256": spline_hash,
                "replay_spline_sha256": replay_spline_hash,
                "replay_match": replay_match,
                "geometry_seed_separated": geometry_seed_separated,
                "topology_seed_sensitive": topology_seed_sensitive,
                "alternate_geometry_seed_without_meshing": alternate_geometry["geometry_seed"],
                "primary_generation": generation,
                "replay_generation": replay_generation,
            }
        )

        mesh_generator, geometry = build_mesh_generator(
            network, nodes, parent_id, registry["geometry_seed"]
        )
        mesh_generator.compute_all()
        mesh_path = world_dir / "mesh.obj"
        mesh_generator.save_mesh(str(mesh_path))
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        if len(vertices) == 0 or len(triangles) == 0:
            raise RuntimeError(f"{parent_id}: native perception mesh is empty")
        chamber_points = (
            len(mesh_generator.ps_of_intersection(nodes["center"]))
            if parent_id == "P04_chamber_multiexit"
            else 0
        )
        if parent_id == "P04_chamber_multiexit" and chamber_points <= 0:
            raise RuntimeError("P04_chamber_multiexit: chamber surface is empty")

        np.savetxt(world_dir / "axis.txt", _per_tunnel_axis(mesh_generator), fmt="%.9g")
        np.savetxt(world_dir / "fta_dist.txt", np.asarray([geometry["fta_distance_m"]]), fmt="%.17g")
        write_json(world_dir / "graph.json", graph)
        write_json(world_dir / "splines.json", splines_document)
        anchors = select_canonical_anchors(
            graph, splines_document, geometry["fta_distance_m"]
        )
        if len(anchors) != 50:
            raise RuntimeError(f"{parent_id}: expected 50 anchors, got {len(anchors)}")
        write_json(world_dir / "anchors.json", {"parent_id": parent_id, "anchors": anchors})

        scene_first = _scene(mesh)
        scene_second = _scene(mesh)
        splines = spline_arrays(splines_document)
        ranges = np.empty((150, 16, 720), dtype=np.float32)
        valid_masks = np.empty((150, 16, 720), dtype=np.uint8)
        labels = np.empty((150, 720), dtype=np.float32)
        parent_records: list[dict[str, Any]] = []
        parent_clearance = []
        parent_branch_los = []
        observation_index = 0
        for anchor in anchors:
            origin = np.asarray(anchor["sensor_xyz_m"], dtype=np.float64)
            clearance_range, clearance_valid = _cast(scene_first, origin, horizontal)
            hits = clearance_range[clearance_valid.astype(bool)]
            minimum_clearance = float(np.min(hits)) if hits.size else MAX_RANGE_M
            parent_clearance.append(minimum_clearance)
            for view_offset in VIEW_OFFSETS_DEG:
                yaw = float((anchor["base_yaw_deg"] + view_offset) % 360.0)
                label = structural_label(anchor["axis_xyz_m"], yaw, splines)
                directions = world_directions(local_directions, yaw)
                first_range, first_valid = _cast(scene_first, origin, directions)
                second_range, second_valid = _cast(scene_second, origin, directions)
                _validate_scan(first_range, first_valid)
                _validate_scan(second_range, second_valid)
                common = first_valid.astype(bool) & second_valid.astype(bool)
                maximum_difference = float(np.max(np.abs(first_range[common] - second_range[common]))) if np.any(common) else 0.0
                masks_equal = bool(np.array_equal(first_valid, second_valid))
                if not masks_equal or maximum_difference > 1e-6:
                    raise RuntimeError(f"{parent_id}/{anchor['anchor_id']}: independent scene mismatch")
                prediction = baseline.predict(first_range, first_valid, ELEVATION_DEG)
                match = match_headings(
                    prediction["headings_robot_deg"], label["headings_robot_deg"]
                )
                branch_los = []
                for target in _representative_branch_points(label):
                    vector = target - origin
                    target_distance = float(np.linalg.norm(vector))
                    branch_range, branch_valid = _cast(
                        scene_first,
                        origin,
                        (vector / target_distance).reshape(1, 3).astype(np.float32),
                    )
                    clear = bool((not branch_valid[0]) or float(branch_range[0]) >= target_distance - 0.25)
                    branch_los.append(clear)
                    parent_branch_los.append(clear)
                ranges[observation_index] = first_range
                valid_masks[observation_index] = first_valid
                labels[observation_index] = np.asarray(label["label_720"], dtype=np.float32)
                record = {
                    "observation_id": f"{parent_id}_{anchor['anchor_id']}_view_{view_offset:+.0f}",
                    "parent_id": parent_id,
                    "anchor_id": anchor["anchor_id"],
                    "view_offset_deg": view_offset,
                    "yaw_deg": yaw,
                    "axis_xyz_m": anchor["axis_xyz_m"],
                    "sensor_xyz_m": anchor["sensor_xyz_m"],
                    "objective_structure": anchor["objective_structure"],
                    "truth_headings_robot_deg": label["headings_robot_deg"],
                    "truth_branch_count": int(label["branch_count"]),
                    "predicted_headings_robot_deg": prediction["headings_robot_deg"],
                    "predicted_branch_count": int(prediction["branch_count"]),
                    "rule_threshold_m": float(prediction["threshold_m"]),
                    "valid_ratio": float(np.mean(first_valid)),
                    "minimum_horizontal_clearance_m": minimum_clearance,
                    "independent_scene_masks_equal": masks_equal,
                    "independent_scene_maximum_valid_range_difference_m": maximum_difference,
                    "branch_los_all_clear": bool(all(branch_los)),
                    "match": match,
                }
                parent_records.append(record)
                all_observation_records.append(record)
                all_contact_ranges.append(first_range)
                observation_index += 1

        if min(parent_clearance) < 0.8:
            raise RuntimeError(f"{parent_id}: minimum clearance {min(parent_clearance):.6f} m < 0.8 m")
        if not all(parent_branch_los):
            raise RuntimeError(f"{parent_id}: at least one objective branch LOS failed")
        shard_arrays = {
            "student_range_m": ranges,
            "student_valid_mask": valid_masks,
            "teacher_exit_label_720": labels,
        }
        shard_path = shards_root / f"{parent_id}.npz"
        np.savez_compressed(shard_path, **shard_arrays)
        role_counts = {
            role: sum(anchor["objective_structure"]["topological_role"] == role for anchor in anchors)
            for role in ("terminal", "corridor", "junction")
        }
        metric = {
            "parent_id": parent_id,
            "topology_seed": registry["topology_seed"],
            "geometry_seed": registry["geometry_seed"],
            "graph_family": family,
            "mesh": {
                "vertices": len(vertices),
                "triangles": len(triangles),
                "chamber_surface_points": chamber_points,
                "known_identity": "native Cano perception mesh; not navigation-qualified",
            },
            "anchors": 50,
            "observations": 150,
            "objective_structure_counts": role_counts,
            "branch_metrics": _aggregate_branch(parent_records),
            "minimum_horizontal_clearance_m": min(parent_clearance),
            "branch_los_passed": bool(all(parent_branch_los)),
            "independent_scene_maximum_valid_range_difference_m": max(item["independent_scene_maximum_valid_range_difference_m"] for item in parent_records),
            "shard": {
                "path": str(shard_path.relative_to(PROJECT_ROOT)),
                "sha256": _sha256(shard_path),
                "array_hash": _array_hash(shard_arrays),
                "student_fields": ["student_range_m", "student_valid_mask"],
                "teacher_fields": ["teacher_exit_label_720"],
                "forbidden_student_fields_absent": ["parent_id", "world_pose", "graph", "splines"],
            },
            "geometry_parameters": geometry,
        }
        parent_metrics.append(metric)
        write_json(run_dir / "metrics" / f"{parent_id}.json", {"metrics": metric, "observations": parent_records})
        write_json(
            world_dir / "metadata.json",
            {
                "parent_id": parent_id,
                "role": "DIAGNOSTIC_CONTRACT_WORLD_NOT_FORMAL_DATASET",
                "generation": generation,
                "geometry": geometry,
                "graph_sha256": graph_hash,
                "splines_sha256": spline_hash,
                "mesh_sha256": _sha256(mesh_path),
                "formal_dataset_worlds": 0,
                "training_samples": 0,
            },
        )
        _plot_world(
            mesh,
            splines_document,
            anchors,
            parent_id,
            maps_root / f"{parent_id}_complete_map.png",
        )

    write_json(run_dir / "metrics/replay_and_seed_separation.json", {"parents": replay_records})
    write_json(
        run_dir / "artifacts/observation_manifest.json",
        {
            "schema_version": "cano_five_topology_diagnostic_observation_manifest_v1",
            "ordering": "parent registry order, anchor_000..049, yaw offsets -30/0/+30",
            "formal_dataset": False,
            "observations": all_observation_records,
        },
    )
    contact_ranges = np.asarray(all_contact_ranges, dtype=np.float32)
    for page_index in range(30):
        start = page_index * 25
        _plot_contact_page(
            all_observation_records[start : start + 25],
            contact_ranges[start : start + 25],
            page_index,
            pages_root / f"page_{page_index + 1:02d}_of_30.png",
        )
    _plot_summary(parent_metrics, run_dir / "previews/five_topology_branch_summary.png")
    aggregate = _aggregate_branch(all_observation_records)
    source_after = _source_precheck()
    summary = {
        "schema_version": "cano_five_topology_cpu_contract_pilot_summary_v1",
        "overall_status": "PASS_CANO_FIVE_TOPOLOGY_CPU_CONTRACT_PILOT",
        "scope": {
            "topology_parents": 5,
            "native_perception_meshes": 5,
            "canonical_anchors": 250,
            "diagnostic_observations": 750,
            "rays_per_observation": 11520,
            "independent_scene_passes": 2,
            "total_primary_rays": 17280000,
            "formal_dataset_worlds": 0,
            "formal_dataset_samples": 0,
            "labels_for_training": 0,
            "training_samples": 0,
            "models": 0,
            "ai_labels": 0,
            "isaac_runs": 0,
            "gazebo_runs": 0,
            "topology_runtime_changes": 0,
            "mtare_changes": 0,
            "trajectories": 0,
            "online_graph_metrics": "NOT_APPLICABLE_STATIC_ANCHORS_HAVE_NO_CAUSAL_MOVEMENT_EDGES",
        },
        "aggregate_frozen_rule_branch_metrics": aggregate,
        "parents": parent_metrics,
        "source_unchanged": source_before == source_after,
        "source_before": source_before,
        "source_after": source_after,
        "complete_visuals": {"world_maps": 5, "contact_pages": 30, "observations_represented": 750},
        "claim_boundary": [
            "This is a zero-training sensor/teacher contract pilot, not a formal dataset.",
            "The frozen range rule is evaluated without retuning; it is not a learned structural model.",
            "The 750 observations are static canonical views, not causal trajectories; node/edge online-graph metrics are intentionally not fabricated.",
            "Native Cano meshes remain disqualified for dynamic navigation even when the perception contract passes.",
            "No M-TARE benchmark world, model training, Isaac, Gazebo or M-TARE modification is involved.",
        ],
    }
    write_json(run_dir / "metrics/summary.json", summary)
    write_json(
        run_dir / "previews/provenance.json",
        {
            "world_maps": "All five complete native meshes, full splines and every canonical anchor; no crop or best-case selection.",
            "contact_pages": "Thirty deterministic 25-observation pages cover all 750 scans exactly once.",
            "colors": "green=objective branch heading; magenta=frozen range-rule branch heading",
            "units": "positions/ranges metres; headings degrees",
            "limitations": summary["claim_boundary"],
        },
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        summary = execute(args.run_dir.resolve())
    except Exception as exc:
        write_json(
            args.run_dir.resolve() / "metrics/executor_failure.json",
            {
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
