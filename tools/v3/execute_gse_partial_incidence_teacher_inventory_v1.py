#!/usr/bin/env python3
"""Inventory objective labels for partially observed two-edge junctions."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
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
from mtare_topo.topology.gse_trace_commit_replay import ProposalTrigger, replay_trace_commits


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


def _family(world: str) -> str:
    return world.rsplit("_C", 1)[0]


def _stratum(world: str) -> str:
    return world.rsplit("_C", 1)[1]


def _objective_nodes(
    code: int, partition: np.ndarray, archive: dict[str, np.ndarray],
) -> list[ObjectiveGraphNode]:
    valid = archive["valid"]
    identity = archive["identity"]
    event = archive["event"]
    center = archive["center"]
    mask = (partition == code) & valid & np.isin(event, ("junction", "terminal"))
    result = []
    for name in sorted(set(identity[mask].tolist())):
        rows = np.flatnonzero(mask & (identity == name))
        if len(set(event[rows].tolist())) != 1 or not np.all(center[rows] == center[rows[0]]):
            raise RuntimeError(f"partial-incidence objective Teacher drift: {name}")
        result.append(ObjectiveGraphNode(
            name, name.split(":node:", 1)[0], str(event[rows[0]]),
            tuple(float(value) for value in center[rows[0]]),
        ))
    return result


def _rule_score(
    replay: dict, center: np.ndarray, objective_nodes: list[ObjectiveGraphNode],
    relations: set[tuple[str, str]], junction_support: int,
) -> dict:
    predicted = []
    keep = set()
    for hypothesis in replay["hypotheses"]:
        if not hypothesis["committed"]:
            continue
        traversals = [str(value.traversal_id) for value in hypothesis["evidence"]]
        if hypothesis["event"] == "junction" and physical_incidence_count(traversals) < junction_support:
            continue
        rows = [int(value.row) for value in hypothesis["evidence"]]
        keep.add(int(hypothesis["id"]))
        predicted.append(PredictedGraphNode(
            int(hypothesis["id"]), str(hypothesis["world"]), str(hypothesis["event"]),
            tuple(float(value) for value in np.mean(center[rows], axis=0)),
        ))
    edges = [
        value for value in replay["edges"]
        if int(value["from_hypothesis"]) in keep and int(value["to_hypothesis"]) in keep
    ]
    return score_objective_spatial_graph(
        predicted_nodes=predicted, predicted_edges=edges,
        objective_nodes=objective_nodes, objective_relations=relations,
        distance_cap_m=4.0,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", required=True, type=Path)
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
    if len(teacher) != EXPECTED:
        raise RuntimeError("partial-incidence Teacher population drift")
    parent = np.asarray([str(value["parent_id"]) for value in teacher])
    traversal = np.asarray([str(value["traversal_id"]) for value in teacher])
    sequence = np.asarray([int(value["sequence_index"]) for value in teacher], dtype=np.int64)
    identity = [None if value.get("identity") is None else str(value["identity"]) for value in teacher]
    event_truth = [str(value["event"]) for value in teacher]
    global_index = np.asarray([int(value["global_sequence_index"]) for value in teacher], dtype=np.int64)
    with np.load(args.pair_cache.resolve(), allow_pickle=False) as archive:
        if not np.array_equal(archive["compact_to_global_sequence_index"], global_index):
            raise RuntimeError("partial-incidence pair-cache alignment drift")
        partition = archive["partition_code"].astype(np.uint8)
        association_valid = archive["association_valid"].astype(np.bool_)
    with np.load(args.action_ensemble.resolve(), allow_pickle=False) as archive:
        if not np.array_equal(archive["global_sequence_index"], global_index):
            raise RuntimeError("partial-incidence action alignment drift")
        probability = archive["probability"].astype(np.float64)
        uncertainty = archive["uncertainty"].astype(np.float64)
    with np.load(args.spatial_projection.resolve(), allow_pickle=False) as archive:
        if not np.array_equal(archive["global_sequence_index"], global_index):
            raise RuntimeError("partial-incidence projection alignment drift")
        center = archive["projected_center_xyz_m"].astype(np.float64)
        seed_center = archive["seed_center_xyz_m"].astype(np.float64)
        position_std = archive["offset_std_m"].astype(np.float64)
    with np.load(args.association_pairs.resolve(), allow_pickle=False) as archive:
        left = archive["left"].astype(np.int64)
        right = archive["right"].astype(np.int64)
        old_accepted = archive["accepted"].astype(np.bool_)
    metric_accepted = unanimous_seed_metric_support(seed_center, left, right, distance_cap_m=4.0)
    union_accepted = old_accepted | metric_accepted
    accepted = {
        (int(first), int(second)): bool(value)
        for first, second, value in zip(left, right, union_accepted, strict=True)
    }
    old_lookup = {
        (int(first), int(second)): bool(value)
        for first, second, value in zip(left, right, old_accepted, strict=True)
    }
    metric_lookup = {
        (int(first), int(second)): bool(value)
        for first, second, value in zip(left, right, metric_accepted, strict=True)
    }
    with np.load(args.objective_teacher.resolve(), allow_pickle=False) as archive:
        if not np.array_equal(archive["global_sequence_index"], global_index):
            raise RuntimeError("partial-incidence objective alignment drift")
        objective = {
            "valid": archive["valid_mask"].astype(np.bool_),
            "identity": archive["identity"].astype(str),
            "event": archive["event"].astype(str),
            "center": archive["objective_center_xyz_m"].astype(np.float64),
        }

    manifests: dict[int, list[dict]] = {}
    scores: dict[int, dict[str, dict]] = {}
    populations = {}
    for code in (0, 1):
        rows = np.flatnonzero(partition == code)
        trigger_values = extract_decision_mass_triggers(
            probability[rows], traversal[rows], sequence[rows], uncertainty[rows],
            decision_threshold=0.97,
        )
        event_names = {1: "junction", 2: "terminal"}
        proposals = []
        for value in trigger_values:
            row = int(rows[value.row])
            predicted_event = int(value.predicted_event_index)
            proposals.append(ProposalTrigger(
                row=row, world=str(parent[row]), order=row,
                traversal_id=str(traversal[row]), sequence_index=int(sequence[row]),
                event=event_names[predicted_event],
                confidence=float(probability[row, predicted_event]),
                uncertainty=float(uncertainty[row]), xyz_m=tuple(float(item) for item in center[row]),
                teacher_identity=identity[row], position_uncertainty_m=float(position_std[row]),
            ))
        replay = replay_trace_commits(
            proposals, accepted, association_valid_rows=set(np.flatnonzero(association_valid)),
            distance_cap_m=4.0, independent_traces_required=2,
        )
        objective_nodes = _objective_nodes(code, partition, objective)
        relations = set(true_relations(rows, traversal, sequence, identity, event_truth))
        scores[code] = {
            f"junction_support_{support}": _rule_score(
                replay, center, objective_nodes, relations, support,
            ) for support in (1, 2, 3, 4)
        }
        base = scores[code]["junction_support_1"]
        matched = {int(value) for value in base["matching"]["mapping"]}
        records = []
        for hypothesis in replay["hypotheses"]:
            if not hypothesis["committed"]:
                continue
            hypothesis_id = int(hypothesis["id"])
            evidence = np.asarray([int(value.row) for value in hypothesis["evidence"]], dtype=np.int64)
            traversals = sorted({str(value.traversal_id) for value in hypothesis["evidence"]})
            event_index = 1 if hypothesis["event"] == "junction" else 2
            seed_means = np.mean(seed_center[:, evidence], axis=1)
            disagreement = max(
                float(np.linalg.norm(seed_means[first] - seed_means[second]))
                for first in range(3) for second in range(first + 1, 3)
            )
            internal_pairs = [tuple(sorted((int(a), int(b)))) for i, a in enumerate(evidence) for b in evidence[i + 1:]]
            records.append({
                "partition": "fit" if code == 0 else "selection",
                "hypothesis_id": hypothesis_id, "world": str(hypothesis["world"]),
                "family": _family(str(hypothesis["world"])), "stratum": _stratum(str(hypothesis["world"])),
                "event": str(hypothesis["event"]), "objective_match": hypothesis_id in matched,
                "objective_identity": base["matching"]["mapping"].get(hypothesis_id),
                "physical_incidence_count": physical_incidence_count(traversals),
                "evidence_rows": evidence.tolist(), "traversal_ids": traversals,
                "evidence_count": len(evidence),
                "event_confidence_min": float(np.min(probability[evidence, event_index])),
                "event_confidence_mean": float(np.mean(probability[evidence, event_index])),
                "event_uncertainty_max": float(np.max(uncertainty[evidence])),
                "event_uncertainty_mean": float(np.mean(uncertainty[evidence])),
                "position_std_max_m": float(np.max(position_std[evidence])),
                "position_std_mean_m": float(np.mean(position_std[evidence])),
                "seed_center_disagreement_max_m": disagreement,
                "evidence_center_spread_max_m": float(np.max(np.linalg.norm(center[evidence] - np.mean(center[evidence], axis=0), axis=1))),
                "old_verifier_internal_pairs": sum(old_lookup.get(value, False) for value in internal_pairs),
                "metric_internal_pairs": sum(metric_lookup.get(value, False) for value in internal_pairs),
            })
        manifests[code] = records
        two = [value for value in records if value["event"] == "junction" and value["physical_incidence_count"] == 2]
        negatives = [value for value in two if not value["objective_match"]]
        populations["fit" if code == 0 else "selection"] = {
            "observations": int(len(rows)), "proposal_triggers": len(proposals),
            "committed_hypotheses": len(records),
            "two_edge_junctions": len(two),
            "two_edge_positive": sum(value["objective_match"] for value in two),
            "two_edge_negative": len(negatives),
            "two_edge_negative_families": len({value["family"] for value in negatives}),
            "two_edge_negative_strata": len({value["stratum"] for value in negatives}),
            "two_edge_negative_by_family": dict(Counter(value["family"] for value in negatives)),
            "two_edge_negative_by_stratum": dict(Counter(value["stratum"] for value in negatives)),
        }

    fit = populations["fit"]
    teacher_sufficient = bool(
        fit["two_edge_junctions"] >= 100 and fit["two_edge_positive"] >= 80
        and fit["two_edge_negative"] >= 20 and fit["two_edge_negative_families"] >= 8
        and fit["two_edge_negative_strata"] >= 5
    )
    fit_rule = scores[0]["junction_support_3"]
    selection_rule = scores[1]["junction_support_3"]
    fit_rule_safe = bool(
        fit_rule["node_precision"] >= .98 and fit_rule["node_recall"] >= .25
        and fit_rule["edge_precision"] >= .98 and fit_rule["edge_recall"] >= .25
    )
    summary = {
        "schema_version": "gse_partial_incidence_teacher_inventory_v1",
        "status": "PASS_GSE_PARTIAL_INCIDENCE_TEACHER_INVENTORY_V1" if teacher_sufficient and fit_rule_safe else "FAIL_GSE_PARTIAL_INCIDENCE_TEACHER_INVENTORY_V1",
        "question": "Is there sufficient fit-only evidence to learn reliability only for partially observed two-edge junctions?",
        "association": {
            "candidate_pairs": len(left), "old_accepted_pairs": int(np.sum(old_accepted)),
            "unanimous_metric_pairs": int(np.sum(metric_accepted)),
            "metric_only_rescued_pairs": int(np.sum(metric_accepted & ~old_accepted)),
            "union_contract": "old_verifier OR all three learned centers within unchanged 4m cap",
        },
        "population": populations,
        "incidence_rule_scores": {"fit": scores[0], "selection": scores[1]},
        "gates": {
            "fit_two_edge_total_at_least_100": fit["two_edge_junctions"] >= 100,
            "fit_two_edge_positive_at_least_80": fit["two_edge_positive"] >= 80,
            "fit_two_edge_negative_at_least_20": fit["two_edge_negative"] >= 20,
            "fit_negative_families_at_least_8": fit["two_edge_negative_families"] >= 8,
            "fit_negative_strata_at_least_5": fit["two_edge_negative_strata"] >= 5,
            "three_edge_rule_fit_science_safe": fit_rule_safe,
            "teacher_sufficient": teacher_sufficient,
            "all_passed": teacher_sufficient and fit_rule_safe,
        },
        "selection_boundary": {
            "three_edge_rule_node_precision": selection_rule["node_precision"],
            "three_edge_rule_edge_precision": selection_rule["edge_precision"],
            "three_edge_rule_edge_recall": selection_rule["edge_recall"],
            "remaining_relation_gap": max(0, 4 - int(selection_rule["correct_unique_edges"])),
            "selection_used_for_training_or_thresholds": False,
        },
        "sources": {
            "teacher_sha256": _sha(args.teacher.resolve()), "pair_cache_sha256": _sha(args.pair_cache.resolve()),
            "action_sha256": _sha(args.action_ensemble.resolve()), "projection_sha256": _sha(args.spatial_projection.resolve()),
            "association_pairs_sha256": _sha(args.association_pairs.resolve()), "objective_teacher_sha256": _sha(args.objective_teacher.resolve()),
        },
        "optimizer_steps": 0, "model_inference_frames": 0, "model_updates": 0,
        "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0,
        "duration_seconds": time.monotonic() - started,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for code, name in ((0, "fit"), (1, "selection")):
        with (output / f"{name}_hypotheses.jsonl").open("w", encoding="utf-8") as stream:
            for value in manifests[code]:
                stream.write(json.dumps(value, sort_keys=True) + "\n")
    with (output / "incidence_rule_scores.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("partition", "junction_support", "node_precision", "node_recall", "node_f1", "edge_precision", "edge_recall", "edge_f1", "macro_f1"))
        for code, name in ((0, "fit"), (1, "selection")):
            for support in (1, 2, 3, 4):
                value = scores[code][f"junction_support_{support}"]
                writer.writerow((name, support, *(value[key] for key in ("node_precision", "node_recall", "node_f1", "edge_precision", "edge_recall", "edge_f1", "node_edge_macro_f1"))))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), constrained_layout=True)
    x = np.arange(1, 5)
    for code, name, color in ((0, "Fit C01–C06", "#2878B5"), (1, "Selection C07–C08", "#E76F51")):
        axes[0].plot(x, [scores[code][f"junction_support_{k}"]["node_precision"] for k in x], marker="o", label=name, color=color)
        axes[1].plot(x, [scores[code][f"junction_support_{k}"]["edge_recall"] for k in x], marker="o", label=name, color=color)
    axes[0].axhline(.98, color="black", linestyle="--", linewidth=1); axes[0].set_title("Node safety from route-diverse support")
    axes[0].set_ylabel("Objective node precision"); axes[0].set_ylim(.9, 1.01)
    axes[1].axhline(.25, color="black", linestyle="--", linewidth=1); axes[1].set_title("Verified relation recovery")
    axes[1].set_ylabel("Objective edge recall"); axes[1].set_ylim(0, .48)
    for axis in axes:
        axis.set_xlabel("Required physical edges for a junction"); axis.set_xticks(x); axis.legend(frameon=False)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("GSE-Graph partial-incidence reliability boundary")
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output / f"gse_partial_incidence_inventory.{suffix}", dpi=240 if suffix == "png" else None)
    plt.close(figure)
    provenance = {
        "schema_version": "gse_partial_incidence_inventory_figure_source_v1",
        "summary_sha256": _sha(output / "summary.json"),
        "csv_sha256": _sha(output / "incidence_rule_scores.csv"),
        "figure_png_sha256": _sha(output / "gse_partial_incidence_inventory.png"),
    }
    (output / "figure_source.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["gates"]["all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
