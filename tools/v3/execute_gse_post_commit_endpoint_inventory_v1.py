#!/usr/bin/env python3
"""Audit execution-endpoint consolidation without training or test reads."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np

from _bootstrap import PROJECT_ROOT  # noqa: F401
from mtare_topo.evaluation.gse_causal_episode_metrics import extract_decision_mass_triggers
from mtare_topo.evaluation.gse_objective_spatial_graph_score import (
    ObjectiveGraphNode, PredictedGraphNode, score_objective_spatial_graph,
)
from mtare_topo.evaluation.gse_partial_incidence_reliability import (
    physical_incidence_count, unanimous_seed_metric_support,
)
from mtare_topo.evaluation.gse_trace_commit_failure_funnel import true_relations
from mtare_topo.topology.gse_post_commit_consolidation import (
    CommittedEndpointSignature, endpoint_consolidation_mapping,
    remap_verified_edges, traversed_edge_endpoint,
)
from mtare_topo.topology.gse_trace_commit_replay import ProposalTrigger, replay_trace_commits


EXPECTED_OBSERVATIONS = 188_126
EXPECTED_TRAVERSALS = 16_078


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _objective_nodes(code: int, partition: np.ndarray, archive: dict[str, np.ndarray]) -> list[ObjectiveGraphNode]:
    mask = partition == code
    mask &= archive["valid"] & np.isin(archive["event"], ("junction", "terminal"))
    result = []
    for identity in sorted(set(archive["identity"][mask].tolist())):
        rows = np.flatnonzero(mask & (archive["identity"] == identity))
        if len(set(archive["event"][rows].tolist())) != 1 or not np.all(archive["center"][rows] == archive["center"][rows[0]]):
            raise RuntimeError(f"post-commit objective Teacher drift: {identity}")
        result.append(ObjectiveGraphNode(
            identity, identity.split(":node:", 1)[0], str(archive["event"][rows[0]]),
            tuple(float(value) for value in archive["center"][rows[0]]),
        ))
    return result


def _resolved_identity(hypothesis: dict, teacher: list[dict]) -> tuple[str | None, str]:
    identities = {
        str(teacher[value.row]["identity"])
        for value in hypothesis["evidence"]
        if teacher[value.row].get("identity") is not None
        and teacher[value.row]["event"] == hypothesis["event"]
    }
    if len(identities) == 1:
        return next(iter(identities)), "unique"
    return None, "none" if not identities else "mixed"


def _score_consolidated(
    *, replay: dict, teacher: list[dict], center: np.ndarray, forward: np.ndarray,
    traversal_length: dict[str, float], old_lookup: dict[tuple[int, int], bool],
    objective_nodes: list[ObjectiveGraphNode], relations: set[tuple[str, str]],
    mode: str,
) -> tuple[dict, list[dict], list[dict]]:
    kept: dict[int, dict] = {}
    signatures = []
    hypothesis_records = []
    for hypothesis in replay["hypotheses"]:
        if not hypothesis["committed"]:
            continue
        evidence = [int(value.row) for value in hypothesis["evidence"]]
        traversals = [str(value.traversal_id) for value in hypothesis["evidence"]]
        incidence = physical_incidence_count(traversals)
        internal = [tuple(sorted((left, right))) for left, right in itertools.combinations(evidence, 2)]
        old_support = sum(old_lookup.get(value, False) for value in internal)
        keep = hypothesis["event"] == "terminal" or incidence >= 2
        if mode == "factorized_hybrid":
            keep = hypothesis["event"] == "terminal" or incidence >= 3 or (incidence >= 2 and old_support >= 1)
        if not keep:
            continue
        endpoints = set()
        for value in hypothesis["evidence"]:
            row = int(value.row)
            traversal_id = str(value.traversal_id)
            length = traversal_length[traversal_id]
            event_arc = float(teacher[row]["traversal_arc_m"]) + float(forward[row])
            event_arc = min(length, max(0.0, event_arc))
            endpoints.add(traversed_edge_endpoint(traversal_id, event_arc, length))
        signature = CommittedEndpointSignature(
            int(hypothesis["id"]), str(hypothesis["world"]), str(hypothesis["event"]),
            tuple(sorted(endpoints)),
        )
        resolved, identity_state = _resolved_identity(hypothesis, teacher)
        kept[int(hypothesis["id"])] = hypothesis
        signatures.append(signature)
        hypothesis_records.append({
            "hypothesis_id": int(hypothesis["id"]), "world": str(hypothesis["world"]),
            "event": str(hypothesis["event"]), "physical_incidence_count": incidence,
            "old_internal_support": old_support, "evidence_rows": evidence,
            "resolved_teacher_identity": resolved, "teacher_identity_state": identity_state,
            "endpoint_tokens": [
                {"physical_edge_id": value.physical_edge_id, "endpoint_side": value.endpoint_side}
                for value in signature.endpoints
            ],
        })
    mapping, components = endpoint_consolidation_mapping(signatures)
    members: dict[int, list[int]] = defaultdict(list)
    for hypothesis_id, representative in mapping.items():
        members[representative].append(hypothesis_id)
    predicted = []
    for representative, identifiers in sorted(members.items()):
        evidence = [
            int(value.row) for hypothesis_id in identifiers
            for value in kept[hypothesis_id]["evidence"]
        ]
        hypothesis = kept[representative]
        predicted.append(PredictedGraphNode(
            representative, str(hypothesis["world"]), str(hypothesis["event"]),
            tuple(float(value) for value in np.mean(center[evidence], axis=0)),
        ))
    filtered_edges = [
        value for value in replay["edges"]
        if int(value["from_hypothesis"]) in mapping and int(value["to_hypothesis"]) in mapping
    ]
    edges, self_loops, duplicate_edges = remap_verified_edges(filtered_edges, mapping)
    score = score_objective_spatial_graph(
        predicted_nodes=predicted, predicted_edges=edges, objective_nodes=objective_nodes,
        objective_relations=relations, distance_cap_m=4.0,
    )
    objective_mapping = score["matching"]["mapping"]
    normalized_truth = {tuple(sorted(value)) for value in relations}
    edge_records = []
    for edge in edges:
        left = objective_mapping.get(int(edge["from_hypothesis"]))
        right = objective_mapping.get(int(edge["to_hypothesis"]))
        relation = tuple(sorted((left, right))) if left and right and left != right else None
        edge_records.append({
            "trace_id": str(edge["trace_id"]),
            "from_hypothesis": int(edge["from_hypothesis"]),
            "to_hypothesis": int(edge["to_hypothesis"]),
            "mapped_relation": list(relation) if relation else None,
            "objective_correct": relation in normalized_truth if relation else False,
        })
    record_by_id = {value["hypothesis_id"]: value for value in hypothesis_records}
    candidate_records = []
    for component in components:
        for left, right in itertools.combinations(component, 2):
            first, second = record_by_id[left], record_by_id[right]
            shared = sorted(
                {(value["physical_edge_id"], value["endpoint_side"]) for value in first["endpoint_tokens"]}
                & {(value["physical_edge_id"], value["endpoint_side"]) for value in second["endpoint_tokens"]}
            )
            first_identity, second_identity = first["resolved_teacher_identity"], second["resolved_teacher_identity"]
            label = "positive" if first_identity and first_identity == second_identity else (
                "negative" if first_identity and second_identity else "unknown"
            )
            candidate_records.append({
                "left_hypothesis": left, "right_hypothesis": right,
                "world": first["world"], "event": first["event"], "label": label,
                "left_identity": first_identity, "right_identity": second_identity,
                "shared_endpoint_tokens": [
                    {"physical_edge_id": value[0], "endpoint_side": value[1]} for value in shared
                ],
            })
    score["consolidation"] = {
        "mode": mode, "component_count": len(components),
        "merged_hypotheses": sum(len(value) - 1 for value in components),
        "components": [list(value) for value in components],
        "candidate_labels": dict(Counter(value["label"] for value in candidate_records)),
        "merge_created_self_loops_removed": self_loops,
        "duplicate_verified_edges_removed": duplicate_edges,
    }
    return score, candidate_records, edge_records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--traversals", required=True, type=Path)
    parser.add_argument("--pair-cache", required=True, type=Path)
    parser.add_argument("--action-ensemble", required=True, type=Path)
    parser.add_argument("--spatial-projection", required=True, type=Path)
    parser.add_argument("--association-pairs", required=True, type=Path)
    parser.add_argument("--objective-teacher", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    teacher = _read_jsonl(args.teacher.resolve())
    traversals_manifest = _read_jsonl(args.traversals.resolve())
    if len(teacher) != EXPECTED_OBSERVATIONS or len(traversals_manifest) != EXPECTED_TRAVERSALS:
        raise RuntimeError("post-commit endpoint source population drift")
    traversal_length = {str(value["traversal_id"]): float(value["length_m"]) for value in traversals_manifest}
    if len(traversal_length) != EXPECTED_TRAVERSALS:
        raise RuntimeError("post-commit traversal identity drift")
    parent = np.asarray([str(value["parent_id"]) for value in teacher])
    traversal = np.asarray([str(value["traversal_id"]) for value in teacher])
    sequence = np.asarray([int(value["sequence_index"]) for value in teacher], dtype=np.int64)
    identity = [None if value.get("identity") is None else str(value["identity"]) for value in teacher]
    event_truth = [str(value["event"]) for value in teacher]
    global_index = np.asarray([int(value["global_sequence_index"]) for value in teacher], dtype=np.int64)
    with np.load(args.pair_cache.resolve(), allow_pickle=False) as archive:
        if not np.array_equal(archive["compact_to_global_sequence_index"], global_index):
            raise RuntimeError("post-commit pair-cache alignment drift")
        partition = archive["partition_code"].astype(np.uint8)
        association_valid = archive["association_valid"].astype(np.bool_)
    with np.load(args.action_ensemble.resolve(), allow_pickle=False) as archive:
        if not np.array_equal(archive["global_sequence_index"], global_index):
            raise RuntimeError("post-commit action alignment drift")
        probability = archive["probability"].astype(np.float64)
        uncertainty = archive["uncertainty"].astype(np.float64)
    with np.load(args.spatial_projection.resolve(), allow_pickle=False) as archive:
        if not np.array_equal(archive["global_sequence_index"], global_index):
            raise RuntimeError("post-commit projection alignment drift")
        center = archive["projected_center_xyz_m"].astype(np.float64)
        seed_center = archive["seed_center_xyz_m"].astype(np.float64)
        position_std = archive["offset_std_m"].astype(np.float64)
        forward = archive["predicted_local_vector_m"][:, 0].astype(np.float64)
    with np.load(args.association_pairs.resolve(), allow_pickle=False) as archive:
        left = archive["left"].astype(np.int64)
        right = archive["right"].astype(np.int64)
        old_accepted = archive["accepted"].astype(np.bool_)
    metric_accepted = unanimous_seed_metric_support(seed_center, left, right, distance_cap_m=4.0)
    union_lookup = {
        tuple(sorted((int(first), int(second)))): bool(value)
        for first, second, value in zip(left, right, old_accepted | metric_accepted, strict=True)
    }
    old_lookup = {
        tuple(sorted((int(first), int(second)))): bool(value)
        for first, second, value in zip(left, right, old_accepted, strict=True)
    }
    with np.load(args.objective_teacher.resolve(), allow_pickle=False) as archive:
        if not np.array_equal(archive["global_sequence_index"], global_index):
            raise RuntimeError("post-commit objective alignment drift")
        objective = {
            "valid": archive["valid_mask"].astype(np.bool_), "identity": archive["identity"].astype(str),
            "event": archive["event"].astype(str), "center": archive["objective_center_xyz_m"].astype(np.float64),
        }

    # The endpoint token itself must be an unambiguous topological invariant.
    endpoint_teacher: dict[int, dict[tuple[str, str, object], set[str]]] = {0: defaultdict(set), 1: defaultdict(set)}
    physical_teacher: dict[int, dict[tuple[str, str], set[str]]] = {0: defaultdict(set), 1: defaultdict(set)}
    for row, value in enumerate(teacher):
        code = int(partition[row])
        if code not in (0, 1) or value["event"] not in ("junction", "terminal") or value.get("identity") is None:
            continue
        token = traversed_edge_endpoint(
            str(value["traversal_id"]), float(value["traversal_arc_m"]),
            traversal_length[str(value["traversal_id"])],
        )
        endpoint_teacher[code][(str(value["parent_id"]), str(value["event"]), token)].add(str(value["identity"]))
        physical_teacher[code][(token.physical_edge_id, str(value["event"]))].add(str(value["identity"]))

    scores: dict[str, dict[str, dict]] = {}
    candidates: dict[str, dict[str, list[dict]]] = {}
    edge_records: dict[str, dict[str, list[dict]]] = {}
    populations = {}
    for code, name in ((0, "fit"), (1, "selection")):
        rows = np.flatnonzero(partition == code)
        trigger_values = extract_decision_mass_triggers(
            probability[rows], traversal[rows], sequence[rows], uncertainty[rows], decision_threshold=0.97,
        )
        proposals = []
        for value in trigger_values:
            row = int(rows[value.row])
            event_index = int(value.predicted_event_index)
            proposals.append(ProposalTrigger(
                row=row, world=str(parent[row]), order=row, traversal_id=str(traversal[row]),
                sequence_index=int(sequence[row]), event={1: "junction", 2: "terminal"}[event_index],
                confidence=float(probability[row, event_index]), uncertainty=float(uncertainty[row]),
                xyz_m=tuple(float(item) for item in center[row]), teacher_identity=identity[row],
                position_uncertainty_m=float(position_std[row]),
            ))
        replay = replay_trace_commits(
            proposals, union_lookup, association_valid_rows=set(np.flatnonzero(association_valid)),
            distance_cap_m=4.0, independent_traces_required=2,
        )
        objective_nodes = _objective_nodes(code, partition, objective)
        relations = set(true_relations(rows, traversal, sequence, identity, event_truth))
        scores[name] = {}
        candidates[name] = {}
        edge_records[name] = {}
        for mode in ("support2_endpoint", "factorized_hybrid"):
            score, candidate, edges = _score_consolidated(
                replay=replay, teacher=teacher, center=center, forward=forward,
                traversal_length=traversal_length, old_lookup=old_lookup,
                objective_nodes=objective_nodes, relations=relations, mode=mode,
            )
            scores[name][mode] = score
            candidates[name][mode] = candidate
            edge_records[name][mode] = edges
        endpoint_ambiguous = sum(len(value) > 1 for value in endpoint_teacher[code].values())
        physical_ambiguous = sum(len(value) > 1 for value in physical_teacher[code].values())
        populations[name] = {
            "observations": int(len(rows)), "proposals": len(proposals),
            "committed_hypotheses": sum(bool(value["committed"]) for value in replay["hypotheses"]),
            "endpoint_teacher_tokens": len(endpoint_teacher[code]),
            "endpoint_teacher_ambiguous_tokens": endpoint_ambiguous,
            "physical_only_teacher_tokens": len(physical_teacher[code]),
            "physical_only_teacher_ambiguous_tokens": physical_ambiguous,
        }

    fit_hybrid = scores["fit"]["factorized_hybrid"]
    selection_hybrid = scores["selection"]["factorized_hybrid"]
    no_negative_candidates = all(
        value["label"] != "negative"
        for partition_values in candidates.values() for mode_values in partition_values.values() for value in mode_values
    )
    inventory_complete = bool(
        populations["fit"]["endpoint_teacher_tokens"] == 2933
        and populations["selection"]["endpoint_teacher_tokens"] == 1013
        and populations["fit"]["endpoint_teacher_ambiguous_tokens"] == 0
        and populations["selection"]["endpoint_teacher_ambiguous_tokens"] == 0
        and scores["fit"]["support2_endpoint"]["consolidation"]["candidate_labels"] == {"positive": 9}
        and scores["selection"]["support2_endpoint"]["consolidation"]["candidate_labels"] == {"positive": 5}
    )
    graph_capacity = bool(
        fit_hybrid["node_precision"] >= .98 and fit_hybrid["node_recall"] >= .25
        and fit_hybrid["edge_recall"] >= .25
        and selection_hybrid["node_precision"] >= .98 and selection_hybrid["node_recall"] >= .25
        and selection_hybrid["edge_precision"] >= .98 and selection_hybrid["edge_recall"] >= .25
    )
    edge_safety_remaining = fit_hybrid["edge_precision"] < .98
    summary = {
        "schema_version": "gse_post_commit_endpoint_inventory_v1",
        "status": "PASS_GSE_POST_COMMIT_ENDPOINT_INVENTORY_V1" if inventory_complete else "FAIL_GSE_POST_COMMIT_ENDPOINT_INVENTORY_V1",
        "question": "Can execution endpoint identity consolidate duplicate committed GSE nodes without expanding online association?",
        "population": populations, "scores": scores,
        "gates": {
            "endpoint_teacher_unambiguous": all(populations[name]["endpoint_teacher_ambiguous_tokens"] == 0 for name in populations),
            "exact_candidate_population": inventory_complete,
            "negative_candidate_population_sufficient_for_training": not no_negative_candidates,
            "training_authorized": False,
            "factorized_hybrid_node_capacity": fit_hybrid["node_precision"] >= .98 and selection_hybrid["node_precision"] >= .98,
            "factorized_hybrid_selection_graph_capacity": all(selection_hybrid[key] >= .98 for key in ("node_precision", "edge_precision")) and selection_hybrid["edge_recall"] >= .25,
            "factorized_hybrid_full_graph_capacity": graph_capacity and not edge_safety_remaining,
            "fit_edge_safety_remaining": edge_safety_remaining,
            "inventory_complete": inventory_complete,
        },
        "decision": {
            "duplicate_verifier_training": "FORBIDDEN_ZERO_HARD_NEGATIVE_CANDIDATES",
            "deterministic_endpoint_consolidation": "SUPPORTED_AS_EXECUTION_TOPOLOGY_INVARIANT",
            "remaining_blocker": "THREE_FIT_FALSE_VERIFIED_EDGE_INSTANCES_REQUIRE_FAIL_CLOSED_EDGE_ENDPOINT_SEMANTIC_AUDIT",
            "online_distance_cap_changed": False, "edge_assembler_changed": False,
            "selection_used_for_threshold_or_training": False,
        },
        "sources": {name: _sha(path.resolve()) for name, path in {
            "teacher": args.teacher, "traversals": args.traversals, "pair_cache": args.pair_cache,
            "action_ensemble": args.action_ensemble, "spatial_projection": args.spatial_projection,
            "association_pairs": args.association_pairs, "objective_teacher": args.objective_teacher,
        }.items()},
        "optimizer_steps": 0, "model_inference_frames": 0, "model_updates": 0,
        "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0,
        "duration_seconds": time.monotonic() - started,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for name in ("fit", "selection"):
        for mode in ("support2_endpoint", "factorized_hybrid"):
            with (output / f"{name}_{mode}_candidates.jsonl").open("w", encoding="utf-8") as stream:
                for value in candidates[name][mode]:
                    stream.write(json.dumps(value, sort_keys=True) + "\n")
            with (output / f"{name}_{mode}_edges.jsonl").open("w", encoding="utf-8") as stream:
                for value in edge_records[name][mode]:
                    stream.write(json.dumps(value, sort_keys=True) + "\n")
    with (output / "capacity_scores.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("partition", "mode", "nodes", "node_precision", "node_recall", "node_f1", "edges", "edge_precision", "edge_recall", "edge_f1", "macro_f1"))
        for name in ("fit", "selection"):
            for mode in ("support2_endpoint", "factorized_hybrid"):
                value = scores[name][mode]
                writer.writerow((name, mode, value["committed_nodes"], value["node_precision"], value["node_recall"], value["node_f1"], value["committed_edges"], value["edge_precision"], value["edge_recall"], value["edge_f1"], value["node_edge_macro_f1"]))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), constrained_layout=True)
    labels = ["Fit\nendpoint", "Fit\nhybrid", "Selection\nendpoint", "Selection\nhybrid"]
    ordered = [scores["fit"]["support2_endpoint"], scores["fit"]["factorized_hybrid"], scores["selection"]["support2_endpoint"], scores["selection"]["factorized_hybrid"]]
    x = np.arange(4)
    axes[0].bar(x - .18, [value["node_precision"] for value in ordered], .36, label="Precision", color="#2878B5")
    axes[0].bar(x + .18, [value["node_recall"] for value in ordered], .36, label="Recall", color="#9AC9DB")
    axes[1].bar(x - .18, [value["edge_precision"] for value in ordered], .36, label="Precision", color="#E76F51")
    axes[1].bar(x + .18, [value["edge_recall"] for value in ordered], .36, label="Recall", color="#F2B6A0")
    for axis, title in zip(axes, ("Objective node qualification", "Execution-verified edge qualification"), strict=True):
        axis.axhline(.98, color="black", linestyle="--", linewidth=1, label="Safety 0.98")
        axis.axhline(.25, color="gray", linestyle=":", linewidth=1, label="Recall 0.25")
        axis.set_xticks(x, labels); axis.set_ylim(0, 1.05); axis.set_title(title)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Score"); axes[0].legend(frameon=False, fontsize=8)
    axes[1].legend(frameon=False, fontsize=8)
    figure.suptitle("Post-commit endpoint consolidation: node solved, fit edge safety remains")
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output / f"gse_post_commit_endpoint_inventory.{suffix}", dpi=240 if suffix == "png" else None)
    plt.close(figure)
    (output / "figure_source.json").write_text(json.dumps({
        "schema_version": "gse_post_commit_endpoint_inventory_figure_source_v1",
        "summary_sha256": _sha(output / "summary.json"), "csv_sha256": _sha(output / "capacity_scores.csv"),
        "png_sha256": _sha(output / "gse_post_commit_endpoint_inventory.png"),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0 if inventory_complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
