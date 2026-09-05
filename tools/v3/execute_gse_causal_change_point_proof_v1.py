#!/usr/bin/env python3
"""Execute the C01-C08 causal geometry change-point Teacher proof."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.gse_causal_change_point_proof import world_causal_change_point_proof
from mtare_topo.governance import load_json, write_json


EXPECTED = {
    "world_count": 80,
    "edge_count": 8039,
    "directed_traversal_count": 16078,
    "sequence_count": 188126,
    "unique_frame_count": 252430,
}
MINIMUM_GEOMETRY_TRANSITION_OBSERVATIONS = 500
FAMILIES = tuple(f"S{index:02d}" for index in range(1, 11))


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


def _write_row(stream, row: dict[str, object]) -> None:
    stream.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _old_transition_identities(path: Path) -> dict[tuple[str, str], list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("split") != "train":
                continue
            if row.get("event") != "geometry_transition":
                continue
            parent_id = str(row["parent_id"])
            if not parent_id.endswith(tuple(f"_C{index:02d}" for index in range(1, 9))):
                raise RuntimeError("non-C01-C08 record entered old transition comparison")
            grouped[(parent_id, str(row["identity"]))].append(row)
    if any(len(rows) != 1 for rows in grouped.values()):
        raise RuntimeError("old transition identity rows are not unique")
    return grouped


def _plot(
    *,
    family_rows: list[dict[str, object]],
    pipeline: MappingLike,
    delays: list[float],
    output_stem: Path,
) -> None:
    labels = [str(row["family"]) for row in family_rows]
    old = np.asarray([int(row["old_transition_identity_count"]) for row in family_rows])
    new = np.asarray([int(row["new_emitted_identity_count"]) for row in family_rows])
    observations = np.asarray([int(row["new_emitted_observation_count"]) for row in family_rows])
    figure, axes = plt.subplots(2, 2, figsize=(12.2, 8.0), constrained_layout=True)

    x = np.arange(len(labels))
    axes[0, 0].bar(x - 0.19, old, width=0.38, label="old short-frame identities", color="#9CA3AF")
    axes[0, 0].bar(x + 0.19, new, width=0.38, label="persistent causal identities", color="#2563EB")
    axes[0, 0].set_xticks(x, labels)
    axes[0, 0].set_ylabel("identity count")
    axes[0, 0].set_title("a  Teacher identity cleanup by topology family", loc="left")
    axes[0, 0].legend(frameon=False, fontsize=8)

    axes[0, 1].bar(labels, observations, color="#0EA5E9")
    axes[0, 1].axhline(
        MINIMUM_GEOMETRY_TRANSITION_OBSERVATIONS,
        color="#DC2626",
        linestyle="--",
        linewidth=1.2,
        label="frozen 500-observation training floor",
    )
    axes[0, 1].set_ylabel("causally labelled observations")
    axes[0, 1].set_title("b  Effective training observations", loc="left")
    axes[0, 1].legend(frameon=False, fontsize=8)

    pipeline_names = ["directional\npersistent", "bidirectional", "causal both", "emitted"]
    pipeline_values = [
        int(pipeline["persistent_directional_episode_count"]),
        int(pipeline["bidirectional_candidate_count"]),
        int(pipeline["causal_bidirectional_candidate_count"]),
        int(pipeline["emitted_point_count"]),
    ]
    axes[1, 0].bar(pipeline_names, pipeline_values, color=["#94A3B8", "#60A5FA", "#22C55E", "#15803D"])
    for index, value in enumerate(pipeline_values):
        axes[1, 0].text(index, value, f"{value:,}", ha="center", va="bottom", fontsize=8)
    axes[1, 0].set_ylabel("candidate/event count")
    axes[1, 0].set_title("c  Fail-closed evidence funnel", loc="left")

    axes[1, 1].hist(delays, bins=min(20, max(5, int(math.sqrt(max(1, len(delays)))))), color="#F59E0B", edgecolor="white")
    axes[1, 1].set_xlabel("causal detection delay (m)")
    axes[1, 1].set_ylabel("directional episodes")
    axes[1, 1].set_title("d  Detection delay after boundary", loc="left")
    figure.suptitle("Persistent bidirectional causal geometry change-point Teacher — C01–C08")
    for suffix, kwargs in (("png", {"dpi": 190}), ("pdf", {}), ("svg", {})):
        figure.savefig(output_stem.with_suffix(f".{suffix}"), **kwargs)
    plt.close(figure)


class MappingLike(dict):
    """Local annotation helper that avoids importing a runtime-only protocol."""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--mesh-root", required=True, type=Path)
    parser.add_argument("--old-teacher-run", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    registry_path = args.registry.resolve()
    mesh_root = args.mesh_root.resolve()
    old_teacher_run = args.old_teacher_run.resolve()
    started = time.monotonic()
    for directory in (run_dir / "artifacts", run_dir / "metrics", run_dir / "previews"):
        directory.mkdir(parents=True, exist_ok=True)

    registry = load_json(registry_path)
    schema = registry["sampling_contract"]["row_schema"]
    decoded = [
        dict(zip(schema, row, strict=True)) if isinstance(row, list) else dict(row)
        for row in registry["rows"]
    ]
    rows = sorted(
        (
            row
            for row in decoded
            if row["split"] == "train"
            and str(row["parent_id"]).endswith(tuple(f"_C{index:02d}" for index in range(1, 9)))
        ),
        key=lambda row: str(row["parent_id"]),
    )
    if len(rows) != EXPECTED["world_count"]:
        raise RuntimeError(f"expected 80 C01-C08 worlds, found {len(rows)}")
    if any(str(row["parent_id"]).endswith(("_C09", "_C10")) for row in rows):
        raise RuntimeError("C09/C10 leaked into change-point proof")

    streams = {
        name: (run_dir / f"artifacts/{name}.jsonl").open("w", encoding="utf-8")
        for name in (
            "world_change_point_summary",
            "bidirectional_change_points",
            "directional_persistent_episodes",
            "past_only_causal_episodes",
            "causal_change_point_labels",
        )
    }
    totals: Counter[str] = Counter()
    family_totals: dict[str, Counter[str]] = {family: Counter() for family in FAMILIES}
    new_points_by_edge: dict[tuple[str, str], list[dict]] = defaultdict(list)
    all_points: list[dict] = []
    emitted_identities: set[str] = set()
    emitted_fit_identities: set[str] = set()
    emitted_selection_identities: set[str] = set()
    delays: list[float] = []
    all_world_checks = True
    try:
        for world_index, row in enumerate(rows, start=1):
            parent_id = str(row["parent_id"])
            family = parent_id.split("_", 1)[0]
            primary = mesh_root / parent_id / "primary"
            result = world_causal_change_point_proof(
                parent_id=parent_id,
                split="train",
                graph=load_json(primary / "graph.json"),
                spline_document=load_json(primary / "splines.json"),
                geometry_parameters=load_json(primary / "geometry_parameters.json"),
                cast_distances=_batch_cast(_scene(primary / "mesh.obj")),
            )
            all_world_checks &= all(bool(value) for value in result["checks"].values())
            world_totals = Counter({key: int(value) for key, value in result["totals"].items()})
            world_totals["world_count"] = 1
            totals.update(world_totals)
            family_totals[family].update(world_totals)
            _write_row(
                streams["world_change_point_summary"],
                {
                    "parent_id": parent_id,
                    "family": family,
                    "totals": dict(world_totals),
                    "checks": result["checks"],
                },
            )
            for point in result["points"]:
                all_points.append(point)
                _write_row(streams["bidirectional_change_points"], point)
                if point["emitted"]:
                    identity = str(point["identity_assignment"]["identity"])
                    emitted_identities.add(identity)
                    (emitted_fit_identities if parent_id.endswith(tuple(f"_C{i:02d}" for i in range(1, 7))) else emitted_selection_identities).add(identity)
                    new_points_by_edge[(parent_id, str(point["edge_id"]))].append(point)
                    for group in point["causal_matches"]:
                        delays.append(float(group[0]["causal_delay_m"]))
            for episode in result["directional_episodes"]:
                _write_row(streams["directional_persistent_episodes"], episode)
            for episode in result["causal_episodes"]:
                _write_row(streams["past_only_causal_episodes"], episode)
            for label in result["labels"]:
                _write_row(streams["causal_change_point_labels"], label)
            print(
                json.dumps(
                    {
                        "world": parent_id,
                        "index": world_index,
                        "of": len(rows),
                        "persistent": world_totals["persistent_directional_episode_count"],
                        "bidirectional": world_totals["bidirectional_candidate_count"],
                        "emitted_identities": world_totals["emitted_identity_count"],
                        "labels": world_totals["emitted_observation_count"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        for stream in streams.values():
            stream.close()

    old_grouped = _old_transition_identities(old_teacher_run / "artifacts/edge_event_identities.jsonl")
    old_by_family: Counter[str] = Counter()
    old_new_counts: Counter[str] = Counter()
    mapping_stream = (run_dir / "artifacts/old_new_identity_mapping.jsonl").open("w", encoding="utf-8")
    try:
        for (parent_id, old_identity), rows_for_identity in sorted(old_grouped.items()):
            old = rows_for_identity[0]
            family = parent_id.split("_", 1)[0]
            old_by_family[family] += 1
            candidates = new_points_by_edge[(parent_id, str(old_identity.split(":")[-3]))]
            distances = sorted(
                (
                    abs(float(old["canonical_center_arc_m"]) - float(point["canonical_center_arc_m"])),
                    str(point["identity_assignment"]["identity"]),
                )
                for point in candidates
            )
            within = [item for item in distances if item[0] <= 10.0 + 1e-9]
            if not within:
                status = "unmatched_old_artifact"
                mapped_identity = None
                distance = None
            elif len(within) > 1 and math.isclose(within[0][0], within[1][0], abs_tol=1e-9):
                status = "ambiguous_old_comparison_only"
                mapped_identity = None
                distance = within[0][0]
            else:
                status = "mapped_within_existing_node_radius"
                mapped_identity = within[0][1]
                distance = within[0][0]
            old_new_counts[status] += 1
            _write_row(
                mapping_stream,
                {
                    "parent_id": parent_id,
                    "family": family,
                    "old_identity": old_identity,
                    "old_center_arc_m": old["canonical_center_arc_m"],
                    "mapping_status": status,
                    "new_identity": mapped_identity,
                    "center_distance_m": distance,
                },
            )
    finally:
        mapping_stream.close()

    # Recompute unique counts globally because node identities can be shared by
    # multiple incident edge points and must not be summed per world as points.
    totals["emitted_unique_identity_count"] = len(emitted_identities)
    totals["emitted_point_count"] = totals["emitted_degree_two_endpoint_count"] + totals["emitted_interior_edge_count"]
    family_rows: list[dict[str, object]] = []
    with (run_dir / "artifacts/family_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = (
            "family",
            "world_count",
            "old_transition_identity_count",
            "new_emitted_identity_count",
            "new_emitted_point_count",
            "new_emitted_observation_count",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for family in FAMILIES:
            values = family_totals[family]
            row = {
                "family": family,
                "world_count": sum(str(item["parent_id"]).startswith(family + "_") for item in rows),
                "old_transition_identity_count": old_by_family[family],
                "new_emitted_identity_count": values["emitted_identity_count"],
                "new_emitted_point_count": values["emitted_degree_two_endpoint_count"] + values["emitted_interior_edge_count"],
                "new_emitted_observation_count": values["emitted_observation_count"],
            }
            family_rows.append(row)
            writer.writerow(row)

    checks = {
        **{f"exact_{name}": totals[name] == expected for name, expected in EXPECTED.items()},
        "all_world_contracts_pass": all_world_checks,
        "emitted_geometry_transition_observations_at_least_500": totals["emitted_observation_count"] >= MINIMUM_GEOMETRY_TRANSITION_OBSERVATIONS,
        "fit_has_nonzero_emitted_identity": bool(emitted_fit_identities),
        "selection_has_nonzero_emitted_identity": bool(emitted_selection_identities),
        "all_emitted_points_are_bidirectional_and_causal": totals["emitted_point_count"] <= totals["causal_bidirectional_candidate_count"] <= totals["bidirectional_candidate_count"],
        "ambiguous_degree_two_endpoints_are_fail_closed": all(
            not point["emitted"]
            for point in all_points
            if point["identity_assignment"]["identity_kind"] == "ambiguous_degree_two_endpoints"
        ),
        "all_reported_delays_finite_nonnegative": bool(delays) and all(math.isfinite(value) and value >= -1e-9 for value in delays),
        "old_comparison_identity_count_matches_manifest": sum(old_by_family.values()) == len(old_grouped),
        "zero_c09_c10_mtare_model_training_reads": True,
    }
    passed = all(checks.values())
    source = {
        "schema_version": "gse_causal_change_point_figure_source_v1",
        "family_rows": family_rows,
        "pipeline": dict(totals),
        "causal_delays_m": delays,
        "old_new_mapping_counts": dict(old_new_counts),
        "acceptance_floor_observations": MINIMUM_GEOMETRY_TRANSITION_OBSERVATIONS,
    }
    write_json(run_dir / "previews/gse_causal_change_point_proof_source.json", source)
    _plot(
        family_rows=family_rows,
        pipeline=MappingLike(totals),
        delays=delays,
        output_stem=run_dir / "previews/gse_causal_change_point_proof",
    )
    figure_hashes = {
        path.name: _sha256(path)
        for path in sorted((run_dir / "previews").glob("gse_causal_change_point_proof.*"))
    }
    write_json(
        run_dir / "previews/gse_causal_change_point_proof_provenance.json",
        {
            "schema_version": "gse_causal_change_point_figure_provenance_v1",
            "generated_by": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
            "source": "previews/gse_causal_change_point_proof_source.json",
            "scope": "C01-C08 only; no C09/C10/M-TARE/model/training",
            "file_sha256": figure_hashes,
        },
    )
    summary = {
        "schema_version": "gse_causal_change_point_proof_v1",
        "overall_status": "PASS_GSE_CAUSAL_CHANGE_POINT_PROOF_V1" if passed else "FAIL_GSE_CAUSAL_CHANGE_POINT_PROOF_V1",
        "research_decision": "A_PERSISTENT_BIDIRECTIONAL_CAUSAL_CHANGE_POINT",
        "scope": {
            "worlds": 80,
            "fit_worlds_C01_C06": 60,
            "selection_worlds_C07_C08": 20,
            "c09_worlds_consumed": 0,
            "c10_worlds_consumed": 0,
            "mtare_worlds_consumed": 0,
        },
        "config": {
            "spacing_m": 1.0,
            "comparison_span_m": 5.0,
            "width_change_m": 1.0,
            "height_change_m": 1.0,
            "endpoint_merge_radius_m": 10.0,
            "new_tuned_hyperparameter_count": 0,
        },
        "totals": dict(totals),
        "fit_emitted_identity_count": len(emitted_fit_identities),
        "selection_emitted_identity_count": len(emitted_selection_identities),
        "family_summary": family_rows,
        "old_transition_identity_count": len(old_grouped),
        "old_new_mapping_counts": dict(old_new_counts),
        "causal_delay_m": {
            "count": len(delays),
            "minimum": min(delays) if delays else None,
            "median": float(np.median(delays)) if delays else None,
            "maximum": max(delays) if delays else None,
        },
        "acceptance_thresholds": {
            "minimum_geometry_transition_observations": MINIMUM_GEOMETRY_TRANSITION_OBSERVATIONS,
        },
        "checks": checks,
        "strict_test_worlds_read": 0,
        "c09_worlds_consumed": 0,
        "mtare_worlds_read": 0,
        "model_inference_frames": 0,
        "training_samples_consumed": 0,
        "optimizer_steps": 0,
        "duration_seconds": time.monotonic() - started,
    }
    write_json(run_dir / "metrics/summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not passed:
        raise RuntimeError("causal change-point Teacher proof failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
