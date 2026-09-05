#!/usr/bin/env python3
"""Posthoc Teacher evaluation for a frozen teacher-free C09 GSE graph."""

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

from mtare_topo.evaluation.gse_objective_spatial_graph_score import (
    ObjectiveGraphNode,
    PredictedGraphNode,
    score_objective_spatial_graph,
)
from mtare_topo.evaluation.gse_trace_commit_failure_funnel import true_relations


PASS = "PASS_GSE_ENDPOINT_GEOMETRY_C09_VALIDATION_V1"
FAIL = "FAIL_GSE_ENDPOINT_GEOMETRY_C09_VALIDATION_V1"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _objective_nodes(teacher: list[dict], parent_manifest: Path) -> list[ObjectiveGraphNode]:
    observed = {}
    for row in teacher:
        event = str(row["event"])
        identity = row.get("identity")
        if event not in ("junction", "terminal") or identity is None:
            continue
        identity = str(identity)
        previous = observed.setdefault(identity, event)
        if previous != event:
            raise RuntimeError(f"C09 objective event identity drift: {identity}")
    manifest = json.loads(parent_manifest.read_text(encoding="utf-8"))
    sources = {
        str(row["parent_id"]): Path(str(row["source_graph"]))
        for row in manifest["parents"] if str(row.get("parent_id", "")).endswith("_C09")
    }
    if len(sources) != 10:
        raise RuntimeError("C09 objective parent manifest drift")
    graph_nodes = {}
    for world, relative in sorted(sources.items()):
        graph = json.loads((Path.cwd() / relative).read_text(encoding="utf-8"))
        for node in graph["nodes"]:
            identifier = f"{world}:node:{node['id']}"
            degree = int(node["degree"])
            event = "junction" if degree >= 3 else "terminal" if degree == 1 else "other"
            graph_nodes[identifier] = (event, tuple(float(value) for value in node["xyz"]))
    missing = sorted(set(observed) - set(graph_nodes))
    if missing:
        raise RuntimeError(f"C09 objective identities absent from TNG: {missing[:3]}")
    result = []
    for identity, event in sorted(observed.items()):
        graph_event, xyz = graph_nodes[identity]
        if graph_event != event:
            raise RuntimeError(f"C09 Teacher/TNG event mismatch: {identity}")
        result.append(ObjectiveGraphNode(
            identity, identity.split(":node:", 1)[0], event, xyz,
        ))
    return result


def _plot(output: Path, score: dict, development: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), constrained_layout=True)
    labels = ("Nodes", "Edges")
    x = np.arange(2)
    width = .34
    validation_precision = [score["node_precision"], score["edge_precision"]]
    validation_recall = [score["node_recall"], score["edge_recall"]]
    development_precision = [
        development["scores"]["selection"]["node_precision"],
        development["scores"]["selection"]["edge_precision"],
    ]
    development_recall = [
        development["scores"]["selection"]["node_recall"],
        development["scores"]["selection"]["edge_recall"],
    ]
    for axis, current, reference, title, gate, letter in (
        (axes[0], validation_precision, development_precision, "False-structure safety", .98, "A"),
        (axes[1], validation_recall, development_recall, "Recovered topology", .25, "B"),
    ):
        axis.bar(x - width / 2, reference, width, color="#72B7B2", label="C07–C08 development")
        axis.bar(x + width / 2, current, width, color="#4C78A8", label="C09 frozen validation")
        axis.axhline(gate, color="#D62728", linestyle="--", linewidth=1.2, label=f"gate {gate:.2f}")
        axis.set_xticks(x, labels); axis.set_ylim(0, 1.03); axis.set_title(title)
        axis.grid(axis="y", color="#E6E6E6", linewidth=.7); axis.set_axisbelow(True)
        axis.text(-.10, 1.04, letter, transform=axis.transAxes, fontweight="bold", fontsize=12)
    axes[0].set_ylabel("Precision"); axes[1].set_ylabel("Recall")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    axes[1].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("GSE-Graph: frozen execution-endpoint geometry on C09")
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"gse_endpoint_geometry_c09.{suffix}", dpi=240 if suffix == "png" else None)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-graph-dir", required=True, type=Path)
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--parent-manifest", required=True, type=Path)
    parser.add_argument("--development-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=False)
    graph_dir = args.frozen_graph_dir.resolve()
    inference_manifest_path = graph_dir / "inference_manifest.json"
    graph_hashes_before = {
        name: _sha(graph_dir / name) for name in (
            "inference_manifest.json", "verified_nodes.jsonl", "verified_edges.jsonl",
            "hypothesis_qualification.jsonl", "decision_trace.jsonl",
            "action_ensemble.npz", "spatial_center_ensemble.npz",
        )
    }
    inference = json.loads(inference_manifest_path.read_text(encoding="utf-8"))
    if (
        inference.get("status") != "PASS_TEACHER_FREE_C09_GRAPH_FREEZE_V1"
        or inference.get("teacher_inputs_read") != 0
        or inference.get("teacher_identity_inputs_read") != 0
    ):
        raise RuntimeError("C09 graph was not frozen teacher-free")
    teacher = [
        row for row in _read_jsonl(args.teacher.resolve())
        if str(row.get("split")) == "validation" and str(row.get("parent_id", "")).endswith("_C09")
    ]
    teacher.sort(key=lambda row: int(row["global_sequence_index"]))
    if len(teacher) != 24_462:
        raise RuntimeError("C09 posthoc Teacher population drift")
    objective = _objective_nodes(teacher, args.parent_manifest.resolve())
    traversal = np.asarray([str(row["traversal_id"]) for row in teacher])
    sequence = np.asarray([int(row["sequence_index"]) for row in teacher], dtype=np.int64)
    identity = [None if row.get("identity") is None else str(row["identity"]) for row in teacher]
    event = [str(row["event"]) for row in teacher]
    relations_with_trace = true_relations(
        range(len(teacher)), traversal, sequence, identity, event,
    )
    relations = set(relations_with_trace)
    nodes = _read_jsonl(graph_dir / "verified_nodes.jsonl")
    edges = _read_jsonl(graph_dir / "verified_edges.jsonl")
    predicted = [PredictedGraphNode(
        int(row["hypothesis_id"]), str(row["world"]), str(row["event"]),
        tuple(float(value) for value in row["xyz_m"]),
    ) for row in nodes]
    score = score_objective_spatial_graph(
        predicted_nodes=predicted, predicted_edges=edges,
        objective_nodes=objective, objective_relations=relations, distance_cap_m=4.0,
    )
    gates = {
        "node_precision_at_least_0p98": score["node_precision"] >= .98,
        "node_recall_at_least_0p25": score["node_recall"] >= .25,
        "edge_precision_at_least_0p98": score["edge_precision"] >= .98,
        "edge_recall_at_least_0p25": score["edge_recall"] >= .25,
        "false_loop_merge_at_most_0p01": score["false_loop_merge_fraction"] <= .01,
    }
    passed = all(gates.values())
    result = {
        "schema_version": "gse_endpoint_geometry_c09_validation_v1",
        "overall_status": PASS if passed else FAIL, "scientific_pass": passed,
        "question": "Does the frozen geometry-semantic execution-endpoint graph retain node/edge safety and recall on C09?",
        "score": score, "gates": gates, "objective_nodes": len(objective),
        "objective_relations": len(relations), "teacher_rows_read_posthoc": len(teacher),
        "graph_frozen_before_teacher": True, "graph_hashes_before_teacher": graph_hashes_before,
        "optimizer_steps": 0, "model_updates": 0, "checkpoint_selection_steps": 0,
        "threshold_selection_steps": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0,
    }
    (output / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output / "scores.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(("object", "precision", "recall", "f1"))
        writer.writerow(("node", score["node_precision"], score["node_recall"], score["node_f1"]))
        writer.writerow(("edge", score["edge_precision"], score["edge_recall"], score["edge_f1"]))
    development = json.loads(args.development_summary.resolve().read_text(encoding="utf-8"))
    _plot(output, score, development)
    figure_source = {
        "schema_version": "gse_endpoint_geometry_c09_figure_source_v1",
        "validation_score": score,
        "development_selection_score": development["scores"]["selection"],
        "gates": {"precision": .98, "recall": .25, "false_loop_merge": .01},
    }
    (output / "figure_source.json").write_text(
        json.dumps(figure_source, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    graph_hashes_after = {name: _sha(graph_dir / name) for name in graph_hashes_before}
    if graph_hashes_before != graph_hashes_after:
        raise RuntimeError("C09 graph changed during posthoc Teacher evaluation")
    print(json.dumps(result, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
