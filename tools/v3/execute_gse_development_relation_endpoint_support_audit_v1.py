#!/usr/bin/env python3
"""C01-C08 read-only audit of relation-endpoint support and survival."""

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
from mtare_topo.evaluation.gse_causal_episode_metrics import extract_decision_mass_triggers
from mtare_topo.evaluation.gse_partial_incidence_reliability import unanimous_seed_metric_support
from mtare_topo.evaluation.gse_trace_commit_failure_funnel import true_relations
from mtare_topo.topology.gse_trace_commit_replay import ProposalTrigger, replay_trace_commits


EXPECTED_OBSERVATIONS = 188_126
EXPECTED = {
    "fit": {"code": 0, "observations": 142_184, "proposals": 3_037, "committed": 718, "relations": 36, "endpoints": 72},
    "selection": {"code": 1, "observations": 45_942, "proposals": 1_022, "committed": 234, "relations": 13, "endpoints": 26},
}


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


def _support_bin(value: int) -> str:
    if value <= 3:
        return "1-3"
    if value <= 10:
        return "4-10"
    if value <= 30:
        return "11-30"
    return "31+"


def _rate(rows: list[dict], key: str) -> float:
    return sum(bool(row[key]) for row in rows) / len(rows) if rows else 0.0


def _plot(output: Path, summary: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 7.2), constrained_layout=True)
    bins = ("1-3", "4-10", "11-30", "31+")
    x = np.arange(len(bins)); width = .36
    for axis, partition, letter in ((axes[0, 0], "fit", "A"), (axes[0, 1], "selection", "B")):
        stats = summary["support_bin_stats"][partition]
        count = [stats.get(name, {}).get("endpoints", 0) for name in bins]
        proposal = [stats.get(name, {}).get("proposal_recall", 0.0) for name in bins]
        final = [stats.get(name, {}).get("final_recall", 0.0) for name in bins]
        axis2 = axis.twinx()
        axis.bar(x, count, width=.68, color="#D9E6F2", label="Endpoints")
        axis2.plot(x, proposal, marker="o", color="#E76F51", label="Proposal recall")
        axis2.plot(x, final, marker="s", color="#2878B5", label="Final-node recall")
        axis.set_xticks(x, bins); axis.set_xlabel("Teacher rows per endpoint")
        axis.set_ylabel("Endpoint count"); axis2.set_ylabel("Recall"); axis2.set_ylim(0, 1.05)
        axis.set_title(f"{letter}  {partition.capitalize()} support and survival")
        handles, labels = axis.get_legend_handles_labels(); h2, l2 = axis2.get_legend_handles_labels()
        axis2.legend(handles + h2, labels + l2, frameon=False, fontsize=8, loc="lower right")

    stage_order = ("proposal_missing", "event_misclassified", "association_or_commit_missing", "qualification_reject", "final_recovered")
    stage_labels = ("No proposal", "Wrong event", "Not committed", "Qualification", "Recovered")
    for axis, partition, letter in ((axes[1, 0], "fit", "C"), (axes[1, 1], "selection", "D")):
        counts = summary["endpoint_stage_counts"][partition]
        values = [counts.get(name, 0) for name in stage_order]
        bars = axis.bar(np.arange(len(values)), values, color=["#E76F51", "#F4A261", "#E9C46A", "#8D99AE", "#2A9D8F"])
        axis.set_xticks(np.arange(len(values)), stage_labels, rotation=20, ha="right")
        axis.set_ylabel("Relation endpoints"); axis.set_title(f"{letter}  {partition.capitalize()} endpoint stages")
        for bar, value in zip(bars, values, strict=True):
            axis.text(bar.get_x() + bar.get_width() / 2, value + .3, str(value), ha="center")
    figure.suptitle("GSE-Graph development audit: supervision support versus topology-endpoint survival", fontsize=13)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output / f"gse_development_relation_endpoint_support.{suffix}", dpi=240 if suffix == "png" else None)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("teacher", "pair_cache", "action_ensemble", "spatial_projection", "association_pairs", "capacity_run", "output_dir"):
        parser.add_argument(f"--{name.replace('_', '-')}", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=False)
    teacher = _read_jsonl(args.teacher.resolve())
    if len(teacher) != EXPECTED_OBSERVATIONS:
        raise RuntimeError("development endpoint-support Teacher population drift")
    parent = np.asarray([str(row["parent_id"]) for row in teacher])
    traversal = np.asarray([str(row["traversal_id"]) for row in teacher])
    sequence = np.asarray([int(row["sequence_index"]) for row in teacher], dtype=np.int64)
    identity = np.asarray(["" if row.get("identity") is None else str(row["identity"]) for row in teacher])
    event_truth = np.asarray([str(row["event"]) for row in teacher])
    global_index = np.asarray([int(row["global_sequence_index"]) for row in teacher], dtype=np.int64)
    with np.load(args.pair_cache.resolve(), allow_pickle=False) as archive:
        if not np.array_equal(archive["compact_to_global_sequence_index"], global_index):
            raise RuntimeError("development endpoint-support pair-cache alignment drift")
        partition = archive["partition_code"].astype(np.uint8)
        association_valid = archive["association_valid"].astype(np.bool_)
    with np.load(args.action_ensemble.resolve(), allow_pickle=False) as archive:
        if not np.array_equal(archive["global_sequence_index"], global_index):
            raise RuntimeError("development endpoint-support action alignment drift")
        probability = archive["probability"].astype(np.float64)
        uncertainty = archive["uncertainty"].astype(np.float64)
    with np.load(args.spatial_projection.resolve(), allow_pickle=False) as archive:
        if not np.array_equal(archive["global_sequence_index"], global_index):
            raise RuntimeError("development endpoint-support projection alignment drift")
        center = archive["projected_center_xyz_m"].astype(np.float64)
        seed_center = archive["seed_center_xyz_m"].astype(np.float64)
        position_std = archive["offset_std_m"].astype(np.float64)
    with np.load(args.association_pairs.resolve(), allow_pickle=False) as archive:
        left = archive["left"].astype(np.int64); right = archive["right"].astype(np.int64)
        old_accepted = archive["accepted"].astype(np.bool_)
    metric = unanimous_seed_metric_support(seed_center, left, right, distance_cap_m=4.0)
    accepted_lookup = {
        tuple(sorted((int(a), int(b)))): bool(value)
        for a, b, value in zip(left, right, old_accepted | metric, strict=True)
    }

    capacity = args.capacity_run.resolve()
    capacity_summary = json.loads((capacity / "artifacts/capacity/summary.json").read_text(encoding="utf-8"))
    if capacity_summary.get("status") != "PASS_GSE_ENDPOINT_GEOMETRY_CAPACITY_V1":
        raise RuntimeError("development endpoint-support capacity source drift")
    endpoint_records = []
    relation_records = []
    support_stats = {}; endpoint_stage_counts = {}; relation_stage_counts = {}
    partition_population = {}
    for name, expected in EXPECTED.items():
        rows = np.flatnonzero(partition == expected["code"])
        triggers = []
        for value in extract_decision_mass_triggers(
            probability[rows], traversal[rows], sequence[rows], uncertainty[rows], decision_threshold=.97,
        ):
            row = int(rows[int(value.row)]); event_index = int(value.predicted_event_index)
            triggers.append(ProposalTrigger(
                row=row, world=str(parent[row]), order=row, traversal_id=str(traversal[row]),
                sequence_index=int(sequence[row]), event={1: "junction", 2: "terminal"}[event_index],
                confidence=float(probability[row, event_index]), uncertainty=float(uncertainty[row]),
                xyz_m=tuple(float(item) for item in center[row]),
                teacher_identity=None if not identity[row] else str(identity[row]),
                position_uncertainty_m=float(position_std[row]),
            ))
        replay = replay_trace_commits(
            triggers, accepted_lookup, association_valid_rows=set(np.flatnonzero(association_valid)),
            distance_cap_m=4.0, independent_traces_required=2,
        )
        committed_hypotheses = [value for value in replay["hypotheses"] if value["committed"]]
        if len(rows) != expected["observations"] or len(triggers) != expected["proposals"] or len(committed_hypotheses) != expected["committed"]:
            raise RuntimeError(f"development endpoint-support replay population drift: {name}")
        qualification = _read_jsonl(capacity / f"artifacts/capacity/{name}_hypothesis_qualification.jsonl")
        qualification_by_id = {int(row["hypothesis_id"]): row for row in qualification}
        if set(qualification_by_id) != {int(value["id"]) for value in committed_hypotheses}:
            raise RuntimeError(f"development endpoint-support qualification alignment drift: {name}")
        allowed = {identifier for identifier, row in qualification_by_id.items() if bool(row["allowed"])}
        final_mapping = {
            int(key): str(value) for key, value in capacity_summary["scores"][name]["matching"]["mapping"].items()
        }
        final_by_identity = {value: key for key, value in final_mapping.items()}
        proposal_by_identity: dict[str, list[ProposalTrigger]] = defaultdict(list)
        for trigger in triggers:
            if trigger.teacher_identity is not None and event_truth[trigger.row] in ("junction", "terminal"):
                proposal_by_identity[str(trigger.teacher_identity)].append(trigger)
        committed_by_identity: dict[str, set[int]] = defaultdict(set)
        for hypothesis in committed_hypotheses:
            names = {
                str(value.teacher_identity) for value in hypothesis["evidence"]
                if value.teacher_identity is not None and event_truth[int(value.row)] in ("junction", "terminal")
            }
            for value in names:
                committed_by_identity[value].add(int(hypothesis["id"]))
        relations_with_trace = true_relations(
            rows, traversal, sequence, [None if not value else value for value in identity.tolist()], event_truth.tolist(),
        )
        relations = set(relations_with_trace)
        endpoints = sorted({value for relation in relations for value in relation})
        if len(relations) != expected["relations"] or len(endpoints) != expected["endpoints"]:
            raise RuntimeError(f"development endpoint-support objective population drift: {name}")
        current_endpoints = []
        for endpoint in endpoints:
            truth_rows = np.flatnonzero((partition == expected["code"]) & (identity == endpoint) & np.isin(event_truth, ("junction", "terminal"))).tolist()
            events = {str(event_truth[row]) for row in truth_rows}
            if len(events) != 1:
                raise RuntimeError(f"development endpoint-support event identity drift: {endpoint}")
            event = next(iter(events)); proposals = proposal_by_identity.get(endpoint, [])
            correct = [value for value in proposals if value.event == event]
            committed_ids = sorted(
                identifier for identifier in committed_by_identity.get(endpoint, set())
                if str(next(value for value in committed_hypotheses if int(value["id"]) == identifier)["event"]) == event
            )
            qualified_ids = sorted(set(committed_ids) & allowed)
            final = endpoint in final_by_identity
            if final: stage = "final_recovered"
            elif qualified_ids: stage = "final_spatial_match_missing"
            elif committed_ids: stage = "qualification_reject"
            elif correct: stage = "association_or_commit_missing"
            elif proposals: stage = "event_misclassified"
            else: stage = "proposal_missing"
            record = {
                "partition": name, "identity": endpoint, "world": endpoint.split(":node:", 1)[0],
                "family": endpoint.split("_C", 1)[0].split("_", 1)[1], "event": event,
                "teacher_rows": len(truth_rows), "support_bin": _support_bin(len(truth_rows)),
                "any_proposal": bool(proposals), "correct_event_proposal": bool(correct),
                "committed": bool(committed_ids), "qualified": bool(qualified_ids), "final_recovered": final,
                "proposal_rows": len(proposals), "correct_event_proposal_rows": len(correct),
                "committed_hypothesis_ids": committed_ids, "qualified_hypothesis_ids": qualified_ids,
                "final_hypothesis_id": final_by_identity.get(endpoint), "stage": stage,
            }
            endpoint_records.append(record); current_endpoints.append(record)
        stage_counts = Counter(row["stage"] for row in current_endpoints)
        endpoint_stage_counts[name] = dict(stage_counts)
        bin_stats = {}
        for support_bin in ("1-3", "4-10", "11-30", "31+"):
            selected = [row for row in current_endpoints if row["support_bin"] == support_bin]
            if selected:
                bin_stats[support_bin] = {
                    "endpoints": len(selected), "proposal_recall": _rate(selected, "correct_event_proposal"),
                    "commit_recall": _rate(selected, "committed"), "qualified_recall": _rate(selected, "qualified"),
                    "final_recall": _rate(selected, "final_recovered"),
                }
        support_stats[name] = bin_stats
        endpoint_by_identity = {row["identity"]: row for row in current_endpoints}
        raw_relations = set()
        hypothesis_identity = {}
        for hypothesis in committed_hypotheses:
            names = {
                str(value.teacher_identity) for value in hypothesis["evidence"]
                if value.teacher_identity is not None and event_truth[int(value.row)] in ("junction", "terminal")
            }
            hypothesis_identity[int(hypothesis["id"])] = next(iter(names)) if len(names) == 1 else None
        for edge in replay["edges"]:
            first = hypothesis_identity.get(int(edge["from_hypothesis"])); second = hypothesis_identity.get(int(edge["to_hypothesis"]))
            if first and second and first != second:
                relation = tuple(sorted((first, second)))
                if relation in relations: raw_relations.add(relation)
        final_relations = set()
        for row in _read_jsonl(capacity / f"artifacts/capacity/{name}_edge_audit.jsonl"):
            if row["objective_correct"] and row["mapped_relation"]:
                final_relations.add(tuple(sorted(map(str, row["mapped_relation"]))))
        stages = Counter()
        for relation in sorted(relations):
            endpoint_stages = [endpoint_by_identity[value]["stage"] for value in relation]
            if relation in final_relations: stage = "recovered"
            elif "proposal_missing" in endpoint_stages: stage = "endpoint_proposal_missing"
            elif "event_misclassified" in endpoint_stages: stage = "endpoint_event_misclassified"
            elif "association_or_commit_missing" in endpoint_stages: stage = "endpoint_not_committed"
            elif "qualification_reject" in endpoint_stages: stage = "endpoint_qualification_reject"
            elif "final_spatial_match_missing" in endpoint_stages: stage = "endpoint_final_match_missing"
            elif relation not in raw_relations: stage = "raw_edge_missing"
            else: raise RuntimeError("development relation lacks a unique causal stage")
            stages[stage] += 1
            relation_records.append({
                "partition": name, "left_identity": relation[0], "right_identity": relation[1],
                "left_endpoint_stage": endpoint_stages[0], "right_endpoint_stage": endpoint_stages[1],
                "supporting_traversals": sorted(relations_with_trace[relation]),
                "raw_edge_present": relation in raw_relations, "final_edge_present": relation in final_relations,
                "stage": stage,
            })
        relation_stage_counts[name] = dict(stages)
        partition_population[name] = {
            "observations": len(rows), "proposals": len(triggers), "committed_hypotheses": len(committed_hypotheses),
            "qualified_hypotheses": len(allowed), "objective_relations": len(relations), "relation_endpoints": len(endpoints),
            "raw_true_relations": len(raw_relations), "final_true_relations": len(final_relations),
        }

    fit_low = support_stats["fit"].get("1-3", {})
    fit_high_rows = [row for row in endpoint_records if row["partition"] == "fit" and row["teacher_rows"] >= 11]
    selection_low = support_stats["selection"].get("1-3", {})
    selection_high_rows = [row for row in endpoint_records if row["partition"] == "selection" and row["teacher_rows"] >= 11]
    fit_gap = _rate(fit_high_rows, "correct_event_proposal") - float(fit_low.get("proposal_recall", 0.0))
    selection_gap = _rate(selection_high_rows, "correct_event_proposal") - float(selection_low.get("proposal_recall", 0.0))
    decision = {
        "fit_low_support_count_at_least_8": int(fit_low.get("endpoints", 0)) >= 8,
        "fit_high_minus_low_proposal_recall_at_least_0p10": fit_gap >= .10,
        "selection_nonreversal": selection_gap >= 0.0,
        "fit_proposal_recall_gap": fit_gap, "selection_proposal_recall_gap": selection_gap,
    }
    decision["identity_balanced_corrective_justified"] = all((
        decision["fit_low_support_count_at_least_8"],
        decision["fit_high_minus_low_proposal_recall_at_least_0p10"],
        decision["selection_nonreversal"],
    ))
    summary = {
        "schema_version": "gse_development_relation_endpoint_support_audit_v1",
        "status": "PASS_GSE_DEVELOPMENT_RELATION_ENDPOINT_SUPPORT_AUDIT_V1",
        "question": "Does per-identity supervision scarcity explain relation-endpoint proposal/commit loss on C01-C08?",
        "population": partition_population, "support_bin_stats": support_stats,
        "endpoint_stage_counts": endpoint_stage_counts, "relation_stage_counts": relation_stage_counts,
        "pre_registered_decision": decision,
        "sources": {name: _sha(path.resolve()) for name, path in {
            "teacher": args.teacher, "pair_cache": args.pair_cache, "action_ensemble": args.action_ensemble,
            "spatial_projection": args.spatial_projection, "association_pairs": args.association_pairs,
            "capacity_seal": capacity / "artifacts/evidence_sha256.txt",
        }.items()},
        "optimizer_steps": 0, "model_updates": 0, "model_inference_frames": 0,
        "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0,
        "duration_seconds": time.monotonic() - started,
    }
    _write_jsonl(output / "endpoint_support_audit.jsonl", endpoint_records)
    _write_jsonl(output / "relation_audit.jsonl", relation_records)
    with (output / "support_bin_stats.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(("partition", "support_bin", "endpoints", "proposal_recall", "commit_recall", "qualified_recall", "final_recall"))
        for name in ("fit", "selection"):
            for support_bin, values in support_stats[name].items():
                writer.writerow((name, support_bin, *(values[key] for key in ("endpoints", "proposal_recall", "commit_recall", "qualified_recall", "final_recall"))))
    _plot(output, summary)
    summary["figure_sha256"] = _sha(output / "gse_development_relation_endpoint_support.png")
    (output / "figure_source.json").write_text(json.dumps({
        "schema_version": "gse_development_relation_endpoint_support_figure_source_v1",
        "support_bin_stats": support_stats, "endpoint_stage_counts": endpoint_stage_counts,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
