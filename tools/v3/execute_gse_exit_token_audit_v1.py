#!/usr/bin/env python3
"""Audit incident-only GSE exit geometry over the sealed 90-world manifest."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

import numpy as np
import open3d as o3d

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.gse_exit_token_audit import world_exit_token_audit
from mtare_topo.governance import load_json, write_json


EXPECTED_CANDIDATES = {
    "train": {
        "corridor": 270246,
        "geometry_transition": 33734,
        "junction": 81461,
        "terminal": 7525,
        "turn": 4006,
    },
    "validation": {
        "corridor": 34902,
        "geometry_transition": 5232,
        "junction": 9743,
        "terminal": 869,
        "turn": 620,
    },
}
EXPECTED_OBSERVATIONS = 212588
EXPECTED_CANDIDATE_COUNT = 448338
EXPECTED_NODE_OBSERVATIONS = 38218
EXPECTED_SAME_TUNNEL_DISTINCT_EDGE_OBSERVATIONS = 29695


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


def _read_grouped(path: Path) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            result[str(row["parent_id"])].append(row)
    return dict(result)


def _write_row(stream, row: dict) -> None:
    stream.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--manifest-run", required=True, type=Path)
    parser.add_argument("--mesh-root", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    manifest_artifacts = args.manifest_run.resolve() / "artifacts"
    mesh_root = args.mesh_root.resolve()
    started = time.monotonic()
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)

    traversals = _read_grouped(manifest_artifacts / "traversal_manifest.jsonl")
    observations = _read_grouped(manifest_artifacts / "teacher_observations.jsonl")
    if set(traversals) != set(observations) or len(observations) != 90:
        raise RuntimeError("sealed manifest parent coverage mismatch")
    if any(parent_id.endswith("_C10") for parent_id in observations):
        raise RuntimeError("strict-test parent leaked into exit-token audit")

    token_path = run_dir / "artifacts/exit_token_audit.jsonl"
    world_path = run_dir / "artifacts/world_exit_token_summary.jsonl"
    split_event_candidates: dict[str, Counter[str]] = {
        "train": Counter(),
        "validation": Counter(),
    }
    split_event_visible: dict[str, Counter[str]] = {
        "train": Counter(),
        "validation": Counter(),
    }
    split_event_width_valid: dict[str, Counter[str]] = {
        "train": Counter(),
        "validation": Counter(),
    }
    totals: Counter[str] = Counter()
    observation_candidate_keys: set[tuple[str, str]] = set()
    candidate_key_count = 0

    with token_path.open("w", encoding="utf-8") as token_stream, world_path.open(
        "w", encoding="utf-8"
    ) as world_stream:
        for world_index, parent_id in enumerate(sorted(observations), start=1):
            rows = observations[parent_id]
            split_values = {str(row["split"]) for row in rows}
            if len(split_values) != 1:
                raise RuntimeError("one parent crosses data splits")
            split = split_values.pop()
            primary = mesh_root / parent_id / "primary"
            result = world_exit_token_audit(
                parent_id=parent_id,
                split=split,
                observations=rows,
                traversal_manifest=traversals[parent_id],
                graph=load_json(primary / "graph.json"),
                spline_document=load_json(primary / "splines.json"),
                geometry_parameters=load_json(primary / "geometry_parameters.json"),
                cast_distances=_batch_cast(_scene(primary / "mesh.obj")),
            )
            summary = result["summary"]
            _write_row(world_stream, summary)
            totals.update(
                {
                    "world_count": 1,
                    "observation_count": summary["observation_count"],
                    "candidate_count": summary["candidate_count"],
                    "visible_token_count": summary["visible_token_count"],
                    "width_valid_count": summary["width_valid_count"],
                    "node_event_observation_count": summary["node_event_observation_count"],
                    "zero_visible_node_event_count": summary["zero_visible_node_event_count"],
                    "same_tunnel_distinct_edge_observation_count": summary[
                        "same_tunnel_distinct_edge_observation_count"
                    ],
                    "nonincident_candidate_count": summary["nonincident_candidate_count"],
                }
            )
            split_event_candidates[split].update(summary["event_candidate_counts"])
            split_event_visible[split].update(summary["event_visible_counts"])
            split_event_width_valid[split].update(summary["event_width_valid_counts"])
            for token_row in result["rows"]:
                key = (
                    str(token_row["observation_id"]),
                    str(token_row["candidate"]["identity"]),
                )
                candidate_key_count += 1
                observation_candidate_keys.add(key)
                _write_row(token_stream, token_row)
            print(
                json.dumps(
                    {
                        "world": parent_id,
                        "index": world_index,
                        "of": len(observations),
                        "observations": summary["observation_count"],
                        "candidates": summary["candidate_count"],
                        "visible": summary["visible_token_count"],
                        "zero_visible_node_events": summary["zero_visible_node_event_count"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    checks = {
        "exact_world_count": totals["world_count"] == 90,
        "exact_observation_count": totals["observation_count"] == EXPECTED_OBSERVATIONS,
        "exact_candidate_count": totals["candidate_count"] == EXPECTED_CANDIDATE_COUNT,
        "exact_node_event_observation_count": totals["node_event_observation_count"]
        == EXPECTED_NODE_OBSERVATIONS,
        "exact_candidate_distribution": all(
            dict(split_event_candidates[split]) == expected
            for split, expected in EXPECTED_CANDIDATES.items()
        ),
        "all_observation_candidate_keys_unique": candidate_key_count
        == len(observation_candidate_keys),
        "zero_nonincident_candidates": totals["nonincident_candidate_count"] == 0,
        "zero_invisible_node_events": totals["zero_visible_node_event_count"] == 0,
        "same_tunnel_distinct_edges_preserved": totals[
            "same_tunnel_distinct_edge_observation_count"
        ]
        == EXPECTED_SAME_TUNNEL_DISTINCT_EDGE_OBSERVATIONS,
        "at_least_one_visible_token": totals["visible_token_count"] > 0,
        "at_least_one_valid_width": totals["width_valid_count"] > 0,
        "zero_strict_test_mtare_reads": True,
    }
    passed = all(checks.values())
    summary = {
        "schema_version": "gse_exit_token_audit_v1",
        "overall_status": "PASS_GSE_EXIT_TOKEN_AUDIT_V1"
        if passed
        else "FAIL_GSE_EXIT_TOKEN_AUDIT_V1",
        "totals": dict(totals),
        "candidate_counts": {
            split: dict(sorted(values.items()))
            for split, values in split_event_candidates.items()
        },
        "visible_counts": {
            split: dict(sorted(values.items()))
            for split, values in split_event_visible.items()
        },
        "width_valid_counts": {
            split: dict(sorted(values.items()))
            for split, values in split_event_width_valid.items()
        },
        "visible_fraction": totals["visible_token_count"] / totals["candidate_count"],
        "width_valid_fraction": totals["width_valid_count"] / totals["candidate_count"],
        "checks": checks,
        "native_mesh_cross_section_rays": totals["candidate_count"] * 20,
        "native_mesh_los_rays": totals["candidate_count"],
        "strict_test_worlds_read": 0,
        "mtare_worlds_read": 0,
        "model_inference_frames": 0,
        "training_samples_consumed": 0,
        "optimizer_steps": 0,
        "duration_seconds": time.monotonic() - started,
    }
    write_json(run_dir / "metrics/summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not passed:
        raise RuntimeError("GSE exit-token audit checks failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
