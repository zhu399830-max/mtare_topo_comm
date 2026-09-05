#!/usr/bin/env python3
"""Read-only causal funnel for the eight frozen C09 graph relations."""

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
from evaluate_gse_endpoint_geometry_c09_v1 import _objective_nodes
from mtare_topo.evaluation.gse_causal_episode_metrics import extract_decision_mass_triggers
from mtare_topo.evaluation.gse_trace_commit_failure_funnel import true_relations


EXPECTED_OBSERVATIONS = 24_462
EXPECTED_TRIGGERS = 1_646
EXPECTED_HYPOTHESES = 442
EXPECTED_COMMITTED = 114
EXPECTED_QUALIFIED = 74
EXPECTED_RELATIONS = 8
EXPECTED_ENDPOINTS = 16


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def _resolved(identifier: int, merged: dict[int, int]) -> int:
    seen = set()
    while identifier in merged:
        if identifier in seen:
            raise RuntimeError("deferred hypothesis merge cycle")
        seen.add(identifier)
        identifier = merged[identifier]
    return identifier


def _raw_edges(
    primary: list[dict], row_to_hypothesis: dict[int, int], committed: set[int],
    traversal: np.ndarray, sequence: np.ndarray,
) -> list[dict]:
    by_trace: dict[str, list[int]] = defaultdict(list)
    for decision in primary:
        by_trace[str(traversal[int(decision["row"])])].append(int(decision["row"]))
    edge_keys = set()
    edges = []
    for trace, rows in sorted(by_trace.items()):
        rows.sort(key=lambda row: (int(sequence[row]), row))
        endpoints = []
        for row in rows:
            hypothesis = row_to_hypothesis[row]
            if hypothesis in committed and (not endpoints or endpoints[-1] != hypothesis):
                endpoints.append(hypothesis)
        if len(endpoints) < 2 or endpoints[0] == endpoints[-1]:
            continue
        key = tuple(sorted((endpoints[0], endpoints[-1])))
        if key in edge_keys:
            continue
        edge_keys.add(key)
        edges.append({
            "raw_edge_id": len(edges), "from_hypothesis": endpoints[0],
            "to_hypothesis": endpoints[-1], "trace_id": trace,
        })
    return edges


def _plot(output: Path, summary: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.1), constrained_layout=True)
    endpoint = summary["endpoint_funnel"]
    labels = ("Objective", "Triggered", "Correct event", "Committed", "Qualified", "Final graph")
    values = [endpoint[name] for name in (
        "objective", "any_proposal", "correct_event_proposal", "committed",
        "qualified", "final_recovered",
    )]
    bars = axes[0].bar(np.arange(len(values)), values, color="#2878B5")
    axes[0].set_xticks(np.arange(len(values)), labels, rotation=22, ha="right")
    axes[0].set_ylabel("Relation endpoints")
    axes[0].set_title("A  Causal survival of relation endpoints")
    axes[0].set_ylim(0, EXPECTED_ENDPOINTS + 2)
    for bar, value in zip(bars, values, strict=True):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + .25, str(value), ha="center")

    order = (
        "endpoint_proposal_missing", "endpoint_event_misclassified",
        "endpoint_not_committed", "endpoint_qualification_reject",
        "endpoint_final_match_missing", "raw_edge_missing", "recovered",
    )
    short = ("No trigger", "Wrong event", "Not committed", "Qualification", "Final match", "No raw edge", "Recovered")
    counts = summary["relation_stage_counts"]
    values = [int(counts.get(name, 0)) for name in order]
    colors = ["#E76F51", "#F4A261", "#E9C46A", "#B8A65A", "#8D99AE", "#6C757D", "#2A9D8F"]
    bars = axes[1].bar(np.arange(len(values)), values, color=colors)
    axes[1].set_xticks(np.arange(len(values)), short, rotation=22, ha="right")
    axes[1].set_ylabel("Objective relations")
    axes[1].set_title("B  First causal bottleneck per relation")
    axes[1].set_ylim(0, EXPECTED_RELATIONS + 1)
    for bar, value in zip(bars, values, strict=True):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + .12, str(value), ha="center")
    figure.suptitle("GSE-Graph C09: why safe nodes do not yet recover enough topology", fontsize=13)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output / f"gse_c09_relation_endpoint_funnel.{suffix}", dpi=240 if suffix == "png" else None)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-run", required=True, type=Path)
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--parent-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    validation = args.validation_run.resolve()
    graph = validation / "artifacts/teacher_free_graph"
    evaluation = validation / "artifacts/evaluation"

    teacher_source = [
        row for row in _read_jsonl(args.teacher.resolve())
        if str(row.get("split")) == "validation" and str(row.get("parent_id", "")).endswith("_C09")
    ]
    if len(teacher_source) != EXPECTED_OBSERVATIONS:
        raise RuntimeError("C09 endpoint-funnel Teacher population drift")
    with np.load(graph / "action_ensemble.npz", allow_pickle=False) as archive:
        global_index = archive["global_sequence_index"].astype(np.int64)
        probability = archive["probability"].astype(np.float64)
        uncertainty = archive["uncertainty"].astype(np.float64)
    teacher_by_index = {int(row["global_sequence_index"]): row for row in teacher_source}
    if (
        len(teacher_by_index) != EXPECTED_OBSERVATIONS
        or len(np.unique(global_index)) != EXPECTED_OBSERVATIONS
        or set(global_index.tolist()) != set(teacher_by_index)
        or probability.shape != (EXPECTED_OBSERVATIONS, 5)
    ):
        raise RuntimeError("C09 endpoint-funnel action/Teacher alignment drift")
    # The validation model archive uses a deterministic shuffled canonical order;
    # align Teacher by stable global ID instead of assuming file-row order.
    teacher = [teacher_by_index[int(value)] for value in global_index]
    traversal = np.asarray([str(row["traversal_id"]) for row in teacher])
    sequence = np.asarray([int(row["sequence_index"]) for row in teacher], dtype=np.int64)
    teacher_event = np.asarray([str(row["event"]) for row in teacher])
    teacher_identity = np.asarray([
        "" if row.get("identity") is None else str(row["identity"]) for row in teacher
    ])
    triggers = extract_decision_mass_triggers(
        probability, traversal, sequence, uncertainty, decision_threshold=.97,
    )
    predicted_event = {int(value.row): {1: "junction", 2: "terminal"}[int(value.predicted_event_index)] for value in triggers}

    decisions = _read_jsonl(graph / "decision_trace.jsonl")
    primary = [row for row in decisions if row["action"] != "deferred_associate"]
    deferred = [row for row in decisions if row["action"] == "deferred_associate"]
    if (
        len(primary) != EXPECTED_TRIGGERS
        or {int(value.row) for value in triggers} != {int(row["row"]) for row in primary}
    ):
        raise RuntimeError("C09 endpoint-funnel proposal replay drift")
    merged = {int(row["merged_hypothesis_id"]): int(row["hypothesis_id"]) for row in deferred}
    row_to_hypothesis = {
        int(row["row"]): _resolved(int(row["hypothesis_id"]), merged) for row in primary
    }
    if len(set(int(row["hypothesis_id"]) for row in primary)) != EXPECTED_HYPOTHESES:
        raise RuntimeError("C09 endpoint-funnel hypothesis population drift")
    evidence_by_hypothesis: dict[int, list[int]] = defaultdict(list)
    for row, hypothesis in row_to_hypothesis.items():
        evidence_by_hypothesis[hypothesis].append(row)

    qualification = _read_jsonl(graph / "hypothesis_qualification.jsonl")
    qualification_by_id = {int(row["hypothesis_id"]): row for row in qualification}
    committed = set(qualification_by_id)
    allowed = {identifier for identifier, row in qualification_by_id.items() if bool(row["allowed"])}
    nodes = _read_jsonl(graph / "verified_nodes.jsonl")
    edges = _read_jsonl(graph / "verified_edges.jsonl")
    if len(committed) != EXPECTED_COMMITTED or len(allowed) != EXPECTED_QUALIFIED or len(nodes) != EXPECTED_QUALIFIED:
        raise RuntimeError("C09 endpoint-funnel qualification population drift")

    raw_edges = _raw_edges(primary, row_to_hypothesis, committed, traversal, sequence)
    if len(raw_edges) != 6 or len(edges) != 1:
        raise RuntimeError("C09 endpoint-funnel raw/final edge reproduction drift")

    objective = _objective_nodes(teacher, args.parent_manifest.resolve())
    relations_with_trace = true_relations(
        range(len(teacher)), traversal, sequence,
        [None if not value else value for value in teacher_identity.tolist()],
        teacher_event.tolist(),
    )
    relations = set(relations_with_trace)
    endpoints = sorted({identity for relation in relations for identity in relation})
    if len(relations) != EXPECTED_RELATIONS or len(endpoints) != EXPECTED_ENDPOINTS:
        raise RuntimeError("C09 endpoint-funnel objective relation population drift")
    objective_event = {node.identity: node.event for node in objective}

    evaluation_summary = json.loads((evaluation / "summary.json").read_text(encoding="utf-8"))
    final_mapping = {
        int(key): str(value) for key, value in evaluation_summary["score"]["matching"]["mapping"].items()
    }
    final_by_identity = {identity: hypothesis for hypothesis, identity in final_mapping.items()}
    if len(final_mapping) != 74:
        raise RuntimeError("C09 endpoint-funnel final spatial matching drift")

    hypothesis_event = {}
    for hypothesis, rows in evidence_by_hypothesis.items():
        events = {predicted_event[row] for row in rows}
        if len(events) != 1:
            raise RuntimeError("C09 replay hypothesis mixes predicted event classes")
        hypothesis_event[hypothesis] = next(iter(events))

    endpoint_records = []
    for identity in endpoints:
        event = objective_event[identity]
        teacher_rows = np.flatnonzero((teacher_identity == identity) & (teacher_event == event)).tolist()
        proposal_rows = [row for row in teacher_rows if row in predicted_event]
        correct_rows = [row for row in proposal_rows if predicted_event[row] == event]
        committed_h = sorted({
            row_to_hypothesis[row] for row in correct_rows
            if row_to_hypothesis[row] in committed and hypothesis_event[row_to_hypothesis[row]] == event
        })
        qualified_h = sorted(set(committed_h) & allowed)
        final_h = final_by_identity.get(identity)
        if final_h is not None:
            stage = "final_recovered"
        elif qualified_h:
            stage = "final_spatial_match_missing"
        elif committed_h:
            stage = "qualification_reject"
        elif correct_rows:
            stage = "association_or_commit_missing"
        elif proposal_rows:
            stage = "event_misclassified"
        else:
            stage = "proposal_missing"
        rejection_reasons = sorted({
            reason
            for hypothesis in committed_h
            for reason, key in (
                ("factorized", "factorized_allowed"), ("anchor", "anchor_allowed"),
                ("branch", "branch_witness"),
            )
            if not bool(qualification_by_id[hypothesis][key])
        })
        endpoint_records.append({
            "identity": identity, "world": identity.split(":node:", 1)[0], "event": event,
            "teacher_rows": len(teacher_rows), "proposal_rows": len(proposal_rows),
            "correct_event_proposal_rows": len(correct_rows),
            "committed_hypothesis_ids": committed_h,
            "qualified_hypothesis_ids": qualified_h,
            "final_hypothesis_id": final_h, "qualification_rejection_reasons": rejection_reasons,
            "stage": stage,
        })
    endpoint_by_identity = {row["identity"]: row for row in endpoint_records}
    endpoint_stage = Counter(row["stage"] for row in endpoint_records)

    raw_edge_records = []
    raw_relations = set()
    for edge in raw_edges:
        identities = []
        nondecision = []
        for key in ("from_hypothesis", "to_hypothesis"):
            rows = evidence_by_hypothesis[int(edge[key])]
            names = sorted({
                teacher_identity[row] for row in rows
                if teacher_identity[row] and teacher_event[row] in ("junction", "terminal")
            })
            other = sorted({
                teacher_identity[row] for row in rows
                if teacher_identity[row] and teacher_event[row] not in ("junction", "terminal")
            })
            identities.append(names); nondecision.append(other)
        relation = None
        kind = "ambiguous_or_empty"
        if len(identities[0]) == len(identities[1]) == 1:
            if identities[0][0] == identities[1][0]:
                kind = "same_objective_identity"
            else:
                relation = tuple(sorted((identities[0][0], identities[1][0])))
                kind = "true_relation" if relation in relations else "false_relation"
                if kind == "true_relation":
                    raw_relations.add(relation)
        raw_edge_records.append({
            **edge, "endpoint_decision_identities": identities,
            "endpoint_nondecision_identities": nondecision,
            "objective_relation": list(relation) if relation else None, "kind": kind,
        })

    final_relations = set()
    for edge in edges:
        left = final_mapping.get(int(edge["from_hypothesis"]))
        right = final_mapping.get(int(edge["to_hypothesis"]))
        if left is not None and right is not None and left != right:
            relation = tuple(sorted((left, right)))
            if relation in relations:
                final_relations.add(relation)
    if len(raw_relations) != 1 or len(final_relations) != 1 or raw_relations != final_relations:
        raise RuntimeError("C09 endpoint-funnel true edge reproduction drift")

    relation_records = []
    for relation in sorted(relations):
        stages = [endpoint_by_identity[value]["stage"] for value in relation]
        if relation in final_relations:
            stage = "recovered"
        elif "proposal_missing" in stages:
            stage = "endpoint_proposal_missing"
        elif "event_misclassified" in stages:
            stage = "endpoint_event_misclassified"
        elif "association_or_commit_missing" in stages:
            stage = "endpoint_not_committed"
        elif "qualification_reject" in stages:
            stage = "endpoint_qualification_reject"
        elif "final_spatial_match_missing" in stages:
            stage = "endpoint_final_match_missing"
        elif relation not in raw_relations:
            stage = "raw_edge_missing"
        else:
            raise RuntimeError("C09 relation has no unique causal stage")
        relation_records.append({
            "left_identity": relation[0], "right_identity": relation[1],
            "left_endpoint_stage": stages[0], "right_endpoint_stage": stages[1],
            "supporting_traversals": sorted(relations_with_trace[relation]),
            "raw_edge_present": relation in raw_relations,
            "final_edge_present": relation in final_relations, "stage": stage,
        })
    relation_stage = Counter(row["stage"] for row in relation_records)
    endpoint_funnel = {
        "objective": len(endpoint_records),
        "any_proposal": sum(row["proposal_rows"] > 0 for row in endpoint_records),
        "correct_event_proposal": sum(row["correct_event_proposal_rows"] > 0 for row in endpoint_records),
        "committed": sum(bool(row["committed_hypothesis_ids"]) for row in endpoint_records),
        "qualified": sum(bool(row["qualified_hypothesis_ids"]) for row in endpoint_records),
        "final_recovered": sum(row["final_hypothesis_id"] is not None for row in endpoint_records),
    }
    if endpoint_funnel["final_recovered"] != 5 or relation_stage.get("recovered", 0) != 1:
        raise RuntimeError("C09 endpoint-funnel known final result drift")

    summary = {
        "schema_version": "gse_c09_relation_endpoint_failure_funnel_v1",
        "status": "PASS_GSE_C09_RELATION_ENDPOINT_FAILURE_FUNNEL_V1",
        "question": "At which causal stage are the seven missing C09 relations lost?",
        "population": {
            "worlds": 10, "observations": len(teacher), "proposal_triggers": len(primary),
            "raw_hypotheses": EXPECTED_HYPOTHESES, "raw_committed_hypotheses": len(committed),
            "qualified_hypotheses": len(allowed), "final_nodes": len(nodes),
            "raw_edges": len(raw_edges), "final_edges": len(edges),
            "objective_relations": len(relations), "relation_endpoints": len(endpoints),
        },
        "endpoint_funnel": endpoint_funnel,
        "endpoint_stage_counts": dict(endpoint_stage),
        "relation_stage_counts": dict(relation_stage),
        "raw_edge_kinds": dict(Counter(row["kind"] for row in raw_edge_records)),
        "conclusion": "The frozen edge assembler is not the primary blocker; relation endpoint survival must be corrected on development worlds before another validation route.",
        "sources": {
            "validation_run": str(validation.relative_to(PROJECT_ROOT)),
            "validation_seal_sha256": _sha(validation / "artifacts/evidence_sha256.txt"),
            "teacher_sha256": _sha(args.teacher.resolve()),
            "parent_manifest_sha256": _sha(args.parent_manifest.resolve()),
        },
        "optimizer_steps": 0, "model_updates": 0, "model_inference_frames": 0,
        "threshold_selection_steps": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0,
        "duration_seconds": time.monotonic() - started,
    }
    _write_jsonl(output / "endpoint_audit.jsonl", endpoint_records)
    _write_jsonl(output / "relation_audit.jsonl", relation_records)
    _write_jsonl(output / "raw_edge_audit.jsonl", raw_edge_records)
    with (output / "endpoint_funnel.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(("stage", "surviving_endpoints"))
        for key, value in endpoint_funnel.items(): writer.writerow((key, value))
    with (output / "relation_stage.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(("stage", "relations"))
        for key, value in sorted(relation_stage.items()): writer.writerow((key, value))
    _plot(output, summary)
    summary["figure_sha256"] = _sha(output / "gse_c09_relation_endpoint_funnel.png")
    (output / "figure_source.json").write_text(
        json.dumps({
            "schema_version": "gse_c09_relation_endpoint_funnel_figure_source_v1",
            "endpoint_funnel": endpoint_funnel,
            "relation_stage_counts": dict(relation_stage),
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
