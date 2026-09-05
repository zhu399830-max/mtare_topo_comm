#!/usr/bin/env python3
"""Read-only causal loss-funnel audit for the spatial trace-commit graph."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from _bootstrap import PROJECT_ROOT  # noqa: F401
from mtare_topo.evaluation.gse_trace_commit_failure_funnel import (
    classify_identity_failure,
    classify_relation_failure,
    cross_trace_pair_state,
    true_relations,
)


EXPECTED = 188_126


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bar_labels(axis, bars, *, digits: int = 3) -> None:
    for bar in bars:
        value = float(bar.get_height())
        label = f"{value:.{digits}f}" if not value.is_integer() else str(int(value))
        axis.text(
            bar.get_x() + bar.get_width() / 2, value,
            label, ha="center", va="bottom", fontsize=8,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-run", required=True, type=Path)
    parser.add_argument("--predecessor-run", required=True, type=Path)
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--pair-cache", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    current = args.current_run.resolve()
    predecessor = args.predecessor_run.resolve()

    teacher_rows = _read_jsonl(args.teacher.resolve())
    if len(teacher_rows) != EXPECTED:
        raise RuntimeError("failure-funnel teacher population drift")
    identity = [None if value.get("identity") is None else str(value["identity"]) for value in teacher_rows]
    event = [str(value["event"]) for value in teacher_rows]
    traversal = np.asarray([str(value["traversal_id"]) for value in teacher_rows])
    sequence = np.asarray([int(value["sequence_index"]) for value in teacher_rows], dtype=np.int64)
    with np.load(args.pair_cache.resolve(), allow_pickle=False) as archive:
        partition = archive["partition_code"].astype(np.uint8)
        association_valid = archive["association_valid"].astype(np.bool_)
        if len(partition) != EXPECTED:
            raise RuntimeError("failure-funnel pair population drift")
    with np.load(current / "artifacts/projection/spatial_center_ensemble_all_rows.npz", allow_pickle=False) as archive:
        center = archive["projected_center_xyz_m"].astype(np.float64)
    with np.load(current / "artifacts/replay/association_pairs.npz", allow_pickle=False) as archive:
        left = archive["left"].astype(np.int64)
        right = archive["right"].astype(np.int64)
        accepted = archive["accepted"].astype(np.bool_)
    accepted_pairs = {
        (int(l), int(r)): bool(value)
        for l, r, value in zip(left, right, accepted, strict=True)
    }

    decisions = _read_jsonl(current / "artifacts/replay/decision_trace.jsonl")
    proposal_rows = sorted({int(value["row"]) for value in decisions})
    selection_rows = np.flatnonzero(partition == 1)
    if len(selection_rows) != 45_942 or len(proposal_rows) != 1_022:
        raise RuntimeError("failure-funnel proposal/split drift")
    proposal_rows_by_identity: dict[str, list[int]] = defaultdict(list)
    for row in proposal_rows:
        name = identity[row]
        if name is not None and event[row] in ("junction", "terminal"):
            proposal_rows_by_identity[name].append(row)

    committed_nodes = _read_jsonl(current / "artifacts/replay/verified_nodes.jsonl")
    committed_hypotheses_by_identity: dict[str, list[int]] = defaultdict(list)
    committed_rows_by_identity_trace: dict[tuple[str, str], list[int]] = defaultdict(list)
    hypothesis_identity: dict[int, str | None] = {}
    false_hypotheses = []
    mixed_hypotheses = []
    for node in committed_nodes:
        hypothesis_id = int(node["hypothesis_id"])
        rows = [int(value) for value in node["evidence_rows"]]
        names = {identity[row] for row in rows if identity[row] is not None}
        name = next(iter(names)) if len(names) == 1 else None
        hypothesis_identity[hypothesis_id] = name
        if len(names) > 1:
            mixed_hypotheses.append(hypothesis_id)
        elif name is None:
            false_hypotheses.append(hypothesis_id)
        else:
            committed_hypotheses_by_identity[name].append(hypothesis_id)
            for row in rows:
                if identity[row] == name:
                    committed_rows_by_identity_trace[(name, str(traversal[row]))].append(row)

    true_identity_event = {}
    for row in selection_rows:
        name = identity[int(row)]
        if name is not None and event[int(row)] in ("junction", "terminal"):
            old = true_identity_event.setdefault(name, event[int(row)])
            if old != event[int(row)]:
                raise RuntimeError("failure-funnel identity event drift")
    if len(true_identity_event) != 274:
        raise RuntimeError("failure-funnel true identity count drift")

    identity_records = []
    for name in sorted(true_identity_event):
        rows = proposal_rows_by_identity.get(name, [])
        pair_state = cross_trace_pair_state(
            rows, traversal, center, association_valid, accepted_pairs,
        )
        hypotheses = committed_hypotheses_by_identity.get(name, [])
        stage = classify_identity_failure(rows, pair_state, hypotheses)
        identity_records.append({
            "identity": name,
            "event": true_identity_event[name],
            "proposal_rows": len(rows),
            "proposal_traversals": pair_state["distinct_traversals"],
            "cross_trace_pairs": pair_state["cross_trace_pairs"],
            "minimum_cross_trace_distance_m": pair_state["minimum_distance_m"],
            "has_within_4m_pair": pair_state["has_within_radius_pair"],
            "accepted_pair_count": pair_state["accepted_pair_count"],
            "committed_hypotheses": len(hypotheses),
            "stage": stage,
        })
    identity_stage = Counter(value["stage"] for value in identity_records)
    event_stage = {
        name: Counter(value["stage"] for value in identity_records if value["event"] == name)
        for name in ("junction", "terminal")
    }

    relations = true_relations(selection_rows, traversal, sequence, identity, event)
    if len(relations) != 13:
        raise RuntimeError("failure-funnel true relation count drift")
    edges = _read_jsonl(current / "artifacts/replay/verified_edges.jsonl")
    predicted_relations = set()
    for edge_value in edges:
        left_name = hypothesis_identity.get(int(edge_value["from_hypothesis"]))
        right_name = hypothesis_identity.get(int(edge_value["to_hypothesis"]))
        if left_name is not None and right_name is not None and left_name != right_name:
            predicted_relations.add(tuple(sorted((left_name, right_name))))
    relation_records = []
    for relation, traces in sorted(relations.items()):
        stage = classify_relation_failure(
            relation, traces, proposal_rows_by_identity,
            committed_hypotheses_by_identity, committed_rows_by_identity_trace,
            predicted_relations,
        )
        relation_records.append({
            "left_identity": relation[0], "right_identity": relation[1],
            "supporting_traversals": len(traces), "stage": stage,
        })
    relation_stage = Counter(value["stage"] for value in relation_records)

    current_summary = json.loads((current / "artifacts/replay/summary.json").read_text(encoding="utf-8"))
    predecessor_summary = json.loads((predecessor / "artifacts/replay/summary.json").read_text(encoding="utf-8"))
    new_metrics = current_summary["selection"]["gse_trace_commit"]
    old_metrics = predecessor_summary["selection"]["gse_trace_commit"]
    duplicate_excess = sum(max(0, len(value) - 1) for value in committed_hypotheses_by_identity.values())
    correct_identity_names = set(committed_hypotheses_by_identity) & set(true_identity_event)
    correct_unique = len(correct_identity_names)
    non_target_identity_hypotheses = sum(
        len(value) for name, value in committed_hypotheses_by_identity.items()
        if name not in true_identity_event
    )
    if (
        len(committed_nodes) != new_metrics["committed_nodes"]
        or correct_unique != new_metrics["correct_unique_nodes"]
        or len(mixed_hypotheses) != new_metrics["false_loop_merges"]
        or correct_unique + non_target_identity_hypotheses + duplicate_excess + len(false_hypotheses) + len(mixed_hypotheses) != len(committed_nodes)
        or len(predicted_relations & set(relations)) != new_metrics["correct_unique_edges"]
    ):
        raise RuntimeError("failure-funnel scorer reproduction drift")

    summary = {
        "schema_version": "gse_trace_commit_failure_funnel_audit_v1",
        "status": "PASS_GSE_TRACE_COMMIT_FAILURE_FUNNEL_AUDIT_V1",
        "question": "Where are true structure nodes and traversed relations lost after spatial-center qualification?",
        "population": {
            "selection_observations": len(selection_rows),
            "proposal_rows": len(proposal_rows),
            "true_node_identities": len(true_identity_event),
            "true_trace_relations": len(relations),
            "committed_nodes": len(committed_nodes),
            "verified_edges": len(edges),
        },
        "node_funnel": {
            "stage_counts": dict(identity_stage),
            "event_stage_counts": {name: dict(value) for name, value in event_stage.items()},
            "correct_unique_identities": correct_unique,
            "duplicate_committed_excess": duplicate_excess,
            "false_empty_identity_hypotheses": len(false_hypotheses),
            "non_target_identity_hypotheses": non_target_identity_hypotheses,
            "mixed_identity_hypotheses": len(mixed_hypotheses),
        },
        "edge_funnel": {
            "stage_counts": dict(relation_stage),
            "predicted_relations": len(predicted_relations),
            "correct_relations": len(predicted_relations & set(relations)),
        },
        "metric_change": {
            name: {"predecessor": float(old_metrics[name]), "spatial": float(new_metrics[name]), "delta": float(new_metrics[name] - old_metrics[name])}
            for name in ("node_precision", "node_recall", "node_f1", "edge_precision", "edge_recall", "edge_f1", "node_edge_macro_f1")
        },
        "sources": {
            "current_run": str(current), "current_seal_sha256": _sha(current / "artifacts/evidence_sha256.txt"),
            "predecessor_run": str(predecessor), "predecessor_seal_sha256": _sha(predecessor / "artifacts/evidence_sha256.txt"),
            "teacher_sha256": _sha(args.teacher.resolve()), "pair_cache_sha256": _sha(args.pair_cache.resolve()),
        },
        "optimizer_steps": 0, "model_updates": 0,
        "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0,
        "duration_seconds": time.monotonic() - started,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output / "identity_audit.jsonl").open("w", encoding="utf-8") as stream:
        for value in identity_records:
            stream.write(json.dumps(value, sort_keys=True) + "\n")
    with (output / "relation_audit.jsonl").open("w", encoding="utf-8") as stream:
        for value in relation_records:
            stream.write(json.dumps(value, sort_keys=True) + "\n")

    with (output / "metric_comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(("metric", "predecessor", "spatial", "delta"))
        for name, value in summary["metric_change"].items():
            writer.writerow((name, value["predecessor"], value["spatial"], value["delta"]))
    with (output / "node_funnel.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(("stage", "count"))
        for name, value in sorted(identity_stage.items()): writer.writerow((name, value))
    with (output / "edge_funnel.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(("stage", "count"))
        for name, value in sorted(relation_stage.items()): writer.writerow((name, value))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 7.2), constrained_layout=True)
    metrics = ("node_precision", "node_recall", "node_f1", "edge_precision", "edge_recall", "edge_f1")
    x = np.arange(len(metrics)); width = 0.36
    old_values = [old_metrics[name] for name in metrics]; new_values = [new_metrics[name] for name in metrics]
    bars0 = axes[0, 0].bar(x - width / 2, old_values, width, label="Scalar center", color="#9AA0A6")
    bars1 = axes[0, 0].bar(x + width / 2, new_values, width, label="Spatial 3D center", color="#2878B5")
    axes[0, 0].set_xticks(x, ["Node P", "Node R", "Node F1", "Edge P", "Edge R", "Edge F1"])
    axes[0, 0].set_ylim(0, 1.13); axes[0, 0].set_ylabel("Score"); axes[0, 0].set_title("Graph quality on unseen C07–C08")
    axes[0, 0].legend(frameon=False, ncol=2); _bar_labels(axes[0, 0], bars0); _bar_labels(axes[0, 0], bars1)

    funnel_labels = ("True", "Proposed", "≥2 traces", "Within 4 m", "Accepted pair", "Committed")
    funnel_values = [
        len(identity_records),
        sum(bool(value["proposal_rows"]) for value in identity_records),
        sum(value["proposal_traversals"] >= 2 for value in identity_records),
        sum(bool(value["has_within_4m_pair"]) for value in identity_records),
        sum(value["accepted_pair_count"] > 0 for value in identity_records),
        sum(value["committed_hypotheses"] > 0 for value in identity_records),
    ]
    bars = axes[0, 1].bar(np.arange(len(funnel_values)), funnel_values, color="#2A9D8F")
    axes[0, 1].set_xticks(np.arange(len(funnel_values)), funnel_labels, rotation=20, ha="right")
    axes[0, 1].set_ylabel("True node identities"); axes[0, 1].set_title("Causal node survival funnel"); _bar_labels(axes[0, 1], bars, digits=0)

    missed_order = ("proposal_missing", "insufficient_traversal_support", "center_outside_4m", "association_verifier_reject", "ambiguity_or_commit_logic")
    missed_labels = ("No proposal", "One trace", "Outside 4 m", "Verifier reject", "Ambiguity/commit")
    missed_values = [identity_stage.get(name, 0) for name in missed_order]
    bars = axes[1, 0].bar(np.arange(len(missed_values)), missed_values, color="#E76F51")
    axes[1, 0].set_xticks(np.arange(len(missed_values)), missed_labels, rotation=20, ha="right")
    axes[1, 0].set_ylabel("Missed true identities"); axes[1, 0].set_title("Why true nodes are still lost"); _bar_labels(axes[1, 0], bars, digits=0)

    edge_order = ("endpoint_proposal_missing", "endpoint_not_committed", "no_common_committed_trace", "edge_assembly_miss", "recovered")
    edge_labels = ("Proposal missing", "Endpoint uncommitted", "No shared trace", "Assembly miss", "Recovered")
    edge_values = [relation_stage.get(name, 0) for name in edge_order]
    colors = ["#E76F51", "#F4A261", "#E9C46A", "#8D99AE", "#2A9D8F"]
    bars = axes[1, 1].bar(np.arange(len(edge_values)), edge_values, color=colors)
    axes[1, 1].set_xticks(np.arange(len(edge_values)), edge_labels, rotation=20, ha="right")
    axes[1, 1].set_ylabel("True trace relations"); axes[1, 1].set_title("Why traversed edges are still lost"); _bar_labels(axes[1, 1], bars, digits=0)
    figure.suptitle("GSE-Graph spatial center: improvement and remaining causal bottleneck", fontsize=13)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output / f"gse_trace_commit_failure_funnel.{suffix}", dpi=220 if suffix == "png" else None)
    plt.close(figure)
    provenance = {
        "schema_version": "gse_trace_commit_failure_funnel_figure_source_v1",
        "summary_sha256": _sha(output / "summary.json"),
        "identity_audit_sha256": _sha(output / "identity_audit.jsonl"),
        "relation_audit_sha256": _sha(output / "relation_audit.jsonl"),
        "figure_png_sha256": _sha(output / "gse_trace_commit_failure_funnel.png"),
    }
    (output / "figure_source.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
