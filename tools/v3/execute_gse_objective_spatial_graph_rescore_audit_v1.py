#!/usr/bin/env python3
"""Rescore two sealed GSE graphs against objective 3D TNG node centres."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from _bootstrap import PROJECT_ROOT  # noqa: F401
from mtare_topo.evaluation.gse_objective_spatial_graph_score import (
    ObjectiveGraphNode,
    PredictedGraphNode,
    score_objective_spatial_graph,
)
from mtare_topo.evaluation.gse_trace_commit_failure_funnel import true_relations


EXPECTED_ALL = 188_126
EXPECTED_SELECTION = 45_942
EXPECTED_OBJECTIVE_NODES = 274
EXPECTED_RELATIONS = 13


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_jsonl(path: Path, values: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, sort_keys=True) + "\n")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _world_from_identity(identity: str) -> str:
    marker = ":node:"
    if marker not in identity:
        raise RuntimeError(f"objective identity has no world prefix: {identity}")
    return identity.split(marker, 1)[0]


def _objective_nodes(teacher_npz: Path) -> tuple[list[ObjectiveGraphNode], np.ndarray, np.ndarray]:
    with np.load(teacher_npz, allow_pickle=False) as archive:
        global_index = archive["global_sequence_index"].astype(np.int64)
        partition = archive["partition_code"].astype(np.uint8)
        valid = archive["valid_mask"].astype(np.bool_)
        identity = archive["identity"].astype(str)
        event = archive["event"].astype(str)
        center = archive["objective_center_xyz_m"].astype(np.float64)
    if (
        len(global_index) != EXPECTED_ALL
        or len(np.unique(global_index)) != EXPECTED_ALL
        or np.any(np.diff(global_index) <= 0)
        or partition.shape != (EXPECTED_ALL,)
    ):
        raise RuntimeError("objective spatial scorer Teacher alignment drift")
    selection = partition == 1
    if int(selection.sum()) != EXPECTED_SELECTION:
        raise RuntimeError("objective spatial scorer selection population drift")
    decision = selection & valid & np.isin(event, ("junction", "terminal"))
    nodes = []
    for name in sorted(set(identity[decision].tolist())):
        rows = np.flatnonzero(decision & (identity == name))
        events = set(event[rows].tolist())
        centres = center[rows]
        if len(events) != 1 or not np.all(centres == centres[0]):
            raise RuntimeError(f"objective node Teacher is non-unique: {name}")
        nodes.append(ObjectiveGraphNode(
            identity=name,
            world=_world_from_identity(name),
            event=next(iter(events)),
            xyz_m=tuple(float(value) for value in centres[0]),
        ))
    if len(nodes) != EXPECTED_OBJECTIVE_NODES:
        raise RuntimeError("objective graph node count drift")
    return nodes, selection, global_index


def _relations(
    teacher_jsonl: Path, selection: np.ndarray, expected_global_index: np.ndarray,
) -> set[tuple[str, str]]:
    rows = _read_jsonl(teacher_jsonl)
    if len(rows) != EXPECTED_ALL:
        raise RuntimeError("objective graph relation Teacher population drift")
    observed_global_index = np.asarray(
        [int(value["global_sequence_index"]) for value in rows], dtype=np.int64,
    )
    if not np.array_equal(observed_global_index, expected_global_index):
        raise RuntimeError("objective graph relation Teacher alignment drift")
    relation_map = true_relations(
        np.flatnonzero(selection),
        [str(value["traversal_id"]) for value in rows],
        [int(value["sequence_index"]) for value in rows],
        [None if value.get("identity") is None else str(value["identity"]) for value in rows],
        [str(value["event"]) for value in rows],
    )
    relations = {tuple(sorted(value)) for value in relation_map}
    if len(relations) != EXPECTED_RELATIONS:
        raise RuntimeError("objective graph relation count drift")
    return relations


def _score_run(
    run: Path,
    projection: Path,
    center_key: str,
    objective_nodes: list[ObjectiveGraphNode],
    objective_relations: set[tuple[str, str]],
    expected_global_index: np.ndarray,
) -> tuple[dict, list[dict], list[dict]]:
    nodes_raw = _read_jsonl(run / "artifacts/replay/verified_nodes.jsonl")
    edges = _read_jsonl(run / "artifacts/replay/verified_edges.jsonl")
    with np.load(projection, allow_pickle=False) as archive:
        global_index = archive["global_sequence_index"].astype(np.int64)
        center = archive[center_key].astype(np.float64)
    if len(center) != EXPECTED_ALL or center.shape != (EXPECTED_ALL, 3):
        raise RuntimeError(f"predicted centre population drift: {run.name}")
    if not np.array_equal(global_index, expected_global_index):
        raise RuntimeError(f"predicted centre row alignment drift: {run.name}")
    predicted = []
    node_rows = []
    for node in nodes_raw:
        evidence = np.asarray([int(value) for value in node["evidence_rows"]], dtype=np.int64)
        if not len(evidence) or np.any(evidence < 0) or np.any(evidence >= EXPECTED_ALL):
            raise RuntimeError("committed node evidence row drift")
        xyz = np.mean(center[evidence], axis=0, dtype=np.float64)
        value = PredictedGraphNode(
            hypothesis_id=int(node["hypothesis_id"]),
            world=str(node["world"]), event=str(node["event"]),
            xyz_m=tuple(float(item) for item in xyz),
        )
        predicted.append(value)
        node_rows.append({
            "hypothesis_id": value.hypothesis_id, "world": value.world,
            "event": value.event, "evidence_rows": evidence.tolist(),
            "predicted_xyz_m": list(value.xyz_m),
        })
    legacy = json.loads((run / "artifacts/replay/summary.json").read_text(encoding="utf-8"))["selection"]["gse_trace_commit"]
    score = score_objective_spatial_graph(
        predicted_nodes=predicted, predicted_edges=edges,
        objective_nodes=objective_nodes, objective_relations=objective_relations,
        legacy_false_loop_merges=int(legacy["false_loop_merges"]),
        distance_cap_m=4.0,
    )
    mapping = score["matching"]["mapping"]
    match_by_hypothesis = {
        int(value["hypothesis_id"]): value for value in score["matching"]["matches"]
    }
    for value in node_rows:
        match = match_by_hypothesis.get(int(value["hypothesis_id"]))
        value.update({
            "matched": match is not None,
            "objective_identity": None if match is None else match["objective_identity"],
            "match_distance_m": None if match is None else match["distance_m"],
        })
    edge_rows = []
    for edge in edges:
        left = mapping.get(int(edge["from_hypothesis"]))
        right = mapping.get(int(edge["to_hypothesis"]))
        relation = tuple(sorted((left, right))) if left is not None and right is not None and left != right else None
        edge_rows.append({
            **edge, "mapped_left_identity": left, "mapped_right_identity": right,
            "correct_objective_relation": bool(relation in objective_relations) if relation else False,
        })
    score["legacy_exact_evidence_row_score"] = legacy
    score["projection_sha256"] = _sha(projection)
    score["replay_seal_sha256"] = _sha(run / "artifacts/evidence_sha256.txt")
    return score, node_rows, edge_rows


def _science_gates(score: dict, stronger_baseline_macro_f1: float | None = None) -> dict[str, bool]:
    gates = {
        "node_precision_at_least_0_98": bool(score["node_precision"] >= 0.98),
        "node_recall_at_least_0_25": bool(score["node_recall"] >= 0.25),
        "edge_precision_at_least_0_98": bool(score["edge_precision"] >= 0.98),
        "edge_recall_at_least_0_25": bool(score["edge_recall"] >= 0.25),
        "false_loop_at_most_0_01": bool(score["false_loop_merge_fraction"] <= 0.01),
    }
    if stronger_baseline_macro_f1 is not None:
        gates["macro_f1_gain_at_least_0_05"] = bool(
            score["node_edge_macro_f1"] - stronger_baseline_macro_f1 >= 0.05
        )
    gates["all_passed"] = all(gates.values())
    return gates


def _figure(output: Path, predecessor: dict, spatial: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ("node_precision", "node_recall", "node_f1", "edge_precision", "edge_recall", "edge_f1")
    labels = ("Node P", "Node R", "Node F1", "Edge P", "Edge R", "Edge F1")
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(1, 3, figsize=(12.3, 3.9), constrained_layout=True)
    x = np.arange(len(names)); width = 0.36
    for axis, score, title in zip(axes[:2], (predecessor, spatial), ("Scalar center graph", "Spatial 3D center graph"), strict=True):
        old = [score["legacy_exact_evidence_row_score"][name] for name in names]
        new = [score[name] for name in names]
        axis.bar(x - width / 2, old, width, label="Exact trigger-row identity", color="#9AA0A6")
        axis.bar(x + width / 2, new, width, label="Objective 3D node match", color="#2878B5")
        axis.set_xticks(x, labels, rotation=25, ha="right")
        axis.set_ylim(0, 1.08); axis.set_title(title); axis.set_ylabel("Score")
        axis.axhline(0.98, color="#D55E00", linestyle="--", linewidth=1, alpha=.7)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    distances = np.asarray([value["distance_m"] for value in spatial["matching"]["matches"]], dtype=np.float64)
    axes[2].hist(distances, bins=np.linspace(0, 4, 21), color="#2A9D8F", edgecolor="white")
    axes[2].axvline(4.0, color="#D55E00", linestyle="--", linewidth=1.2, label="Frozen 4 m cap")
    axes[2].set_xlabel("Matched node-center error (m)")
    axes[2].set_ylabel("Committed nodes")
    axes[2].set_title("Spatial graph localization")
    axes[2].legend(frameon=False, fontsize=8)
    figure.suptitle("GSE-Graph evaluation: trigger timing versus objective structure location", fontsize=12.5)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output / f"gse_objective_spatial_graph_rescore.{suffix}", dpi=240 if suffix == "png" else None)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predecessor-run", required=True, type=Path)
    parser.add_argument("--spatial-run", required=True, type=Path)
    parser.add_argument("--objective-teacher", required=True, type=Path)
    parser.add_argument("--observation-teacher", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    objective_nodes, selection, global_index = _objective_nodes(args.objective_teacher.resolve())
    objective_relations = _relations(args.observation_teacher.resolve(), selection, global_index)

    predecessor_run = args.predecessor_run.resolve()
    spatial_run = args.spatial_run.resolve()
    predecessor, predecessor_nodes, predecessor_edges = _score_run(
        predecessor_run,
        predecessor_run / "artifacts/projection/event_center_projection.npz",
        "projected_center_xyz_m", objective_nodes, objective_relations, global_index,
    )
    spatial, spatial_nodes, spatial_edges = _score_run(
        spatial_run,
        spatial_run / "artifacts/projection/spatial_center_ensemble_all_rows.npz",
        "projected_center_xyz_m", objective_nodes, objective_relations, global_index,
    )
    baseline_summary = json.loads((spatial_run / "artifacts/replay/summary.json").read_text(encoding="utf-8"))
    stronger_baseline = max(
        float(baseline_summary["selection"]["learned_ghost_immediate_commit"]["node_edge_macro_f1"]),
        float(baseline_summary["selection"]["structured_rule_trace_commit"]["node_edge_macro_f1"]),
    )
    predecessor_gates = _science_gates(predecessor, stronger_baseline)
    spatial_gates = _science_gates(spatial, stronger_baseline)
    summary = {
        "schema_version": "gse_objective_spatial_graph_rescore_audit_v1",
        "status": "PASS_GSE_OBJECTIVE_SPATIAL_GRAPH_RESCORE_AUDIT_V1",
        "question": "Does objective 3D one-to-one node matching change the scientific conclusion of the sealed trace-commit graphs?",
        "scoring_policy": {
            "runtime_teacher_access": 0,
            "predicted_node_location": "arithmetic mean of frozen evidence-row predicted centers",
            "matching": "maximum-cardinality then minimum-distance one-to-one assignment within identical world and event",
            "distance_cap_m": 4.0,
            "duplicates": "unmatched false positives",
            "verified_edges": "scored only through the frozen node assignment",
            "legacy_score_retained": True,
        },
        "population": {
            "all_observations": EXPECTED_ALL, "selection_observations": EXPECTED_SELECTION,
            "worlds": 20, "objective_nodes": len(objective_nodes),
            "objective_relations": len(objective_relations),
        },
        "predecessor": predecessor, "spatial": spatial,
        "science_gates": {"predecessor": predecessor_gates, "spatial": spatial_gates},
        "metric_delta_spatial_minus_predecessor": {
            name: float(spatial[name] - predecessor[name])
            for name in ("node_precision", "node_recall", "node_f1", "edge_precision", "edge_recall", "edge_f1", "node_edge_macro_f1")
        },
        "stronger_rule_or_immediate_baseline_macro_f1": stronger_baseline,
        "sources": {
            "objective_teacher_sha256": _sha(args.objective_teacher.resolve()),
            "observation_teacher_sha256": _sha(args.observation_teacher.resolve()),
        },
        "optimizer_steps": 0, "model_inference_frames": 0, "model_updates": 0,
        "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0,
        "duration_seconds": time.monotonic() - started,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_jsonl(output / "predecessor_node_matches.jsonl", predecessor_nodes)
    _write_jsonl(output / "spatial_node_matches.jsonl", spatial_nodes)
    _write_jsonl(output / "predecessor_edge_matches.jsonl", predecessor_edges)
    _write_jsonl(output / "spatial_edge_matches.jsonl", spatial_edges)
    with (output / "metric_comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("method", "scorer", "node_precision", "node_recall", "node_f1", "edge_precision", "edge_recall", "edge_f1", "macro_f1"))
        for method, score in (("scalar", predecessor), ("spatial", spatial)):
            for scorer, values in (("legacy_exact_row", score["legacy_exact_evidence_row_score"]), ("objective_3d", score)):
                writer.writerow((method, scorer, *(values[name] for name in ("node_precision", "node_recall", "node_f1", "edge_precision", "edge_recall", "edge_f1", "node_edge_macro_f1"))))
    _figure(output, predecessor, spatial)
    figure_source = {
        "schema_version": "gse_objective_spatial_graph_rescore_figure_source_v1",
        "summary_sha256": _sha(output / "summary.json"),
        "metric_comparison_sha256": _sha(output / "metric_comparison.csv"),
        "figure_png_sha256": _sha(output / "gse_objective_spatial_graph_rescore.png"),
    }
    (output / "figure_source.json").write_text(json.dumps(figure_source, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
