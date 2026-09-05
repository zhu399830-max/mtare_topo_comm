#!/usr/bin/env python3
"""Evaluate frozen C09 causal decision-node outputs after inference is complete."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mtare_topo.evaluation.gse_causal_episode_metrics import (
    evaluate_decision_event_triggers,
    extract_causal_event_triggers,
)
from mtare_topo.representation.gse_causal_episode_detector import (
    materialize_causal_episode_references,
)
from mtare_topo.semantics.geometric_semantics import EVENT_NAMES


PASS_STATUS = "PASS_GSE_FACTORIZED_CAUSAL_NODE_C09_V1"
FAIL_STATUS = "FAIL_GSE_FACTORIZED_CAUSAL_NODE_C09_V1"
STRUCTURAL_THRESHOLD = 0.986
DECISION_EVENTS = ("junction", "terminal")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _identity_coverage(probability, rows, bank, traversal, sequence, boundary, uncertainty):
    triggers = extract_causal_event_triggers(
        probability, traversal, sequence, boundary, uncertainty,
        structural_threshold=STRUCTURAL_THRESHOLD,
    )
    decision_indices = {EVENT_NAMES.index(name) for name in DECISION_EVENTS}
    matched_episode: set[int] = set()
    correct_identity = {index: set() for index in decision_indices}
    for trigger in triggers:
        if trigger.predicted_event_index not in decision_indices:
            continue
        episode = int(bank.episode_id[trigger.row])
        if episode < 0 or episode in matched_episode:
            continue
        matched_episode.add(episode)
        if trigger.predicted_event_index == int(bank.event_index[trigger.row]):
            identity = rows[trigger.row].get("identity")
            if identity is not None:
                correct_identity[trigger.predicted_event_index].add(str(identity))
    result = {}
    for index in sorted(decision_indices):
        name = EVENT_NAMES[index]
        truth = {
            str(row["identity"]) for row, event in zip(rows, bank.event_index, strict=True)
            if int(event) == index and row.get("identity") is not None
        }
        correct = correct_identity[index]
        result[name] = {
            "correct_identities": len(correct),
            "true_identities": len(truth),
            "coverage": len(correct) / len(truth) if truth else 0.0,
        }
    return result


def _selection_decision_metrics(summary: dict) -> dict:
    per_event = summary["new_ensemble_trigger_metrics"]["per_event"]
    selected = {name: per_event[name] for name in DECISION_EVENTS}
    correct = sum(int(value["correct_unique_episodes"]) for value in selected.values())
    predicted = sum(int(value["predicted_triggers"]) for value in selected.values())
    truth = sum(int(value["true_episodes"]) for value in selected.values())
    return {
        "predicted_decision_triggers": predicted,
        "true_decision_episodes": truth,
        "correctly_classified_unique_decision_episodes": correct,
        "decision_trigger_precision": correct / predicted,
        "decision_episode_recall": correct / truth,
        "per_event": selected,
    }


def _figure(output: Path, selection: dict, validation: dict, identity: dict) -> None:
    import matplotlib.pyplot as plt

    source = {
        "selection": selection,
        "c09_validation": validation,
        "c09_identity_coverage": identity,
        "structural_threshold": STRUCTURAL_THRESHOLD,
    }
    (output / "gse_factorized_causal_nodes_c09_source.json").write_text(
        json.dumps(source, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    names = ["Aggregate", "Junction", "Terminal"]
    selection_precision = [
        selection["decision_trigger_precision"],
        selection["per_event"]["junction"]["precision"],
        selection["per_event"]["terminal"]["precision"],
    ]
    validation_precision = [
        validation["decision_trigger_precision"],
        validation["per_event"]["junction"]["precision"],
        validation["per_event"]["terminal"]["precision"],
    ]
    selection_recall = [
        selection["decision_episode_recall"],
        selection["per_event"]["junction"]["recall"],
        selection["per_event"]["terminal"]["recall"],
    ]
    validation_recall = [
        validation["decision_episode_recall"],
        validation["per_event"]["junction"]["recall"],
        validation["per_event"]["terminal"]["recall"],
    ]
    x = np.arange(3)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    for axis, left, right, title in (
        (axes[0], selection_precision, validation_precision, "Decision-trigger precision"),
        (axes[1], selection_recall, validation_recall, "Decision-episode recall"),
    ):
        axis.bar(x - .18, left, .36, label="C07-C08 selection", color="#4C78A8")
        axis.bar(x + .18, right, .36, label="C09 validation", color="#F58518")
        axis.set_xticks(x, names)
        axis.set_ylim(0.0, 1.05)
        axis.set_ylabel("Score")
        axis.set_title(title)
        axis.grid(axis="y", alpha=.25)
    axes[0].legend(frameon=False, loc="lower left")
    fig.suptitle("Past-only learned junction/terminal node generation")
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"gse_factorized_causal_nodes_c09.{suffix}", dpi=220)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference-dir", required=True, type=Path)
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--selection-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise RuntimeError("C09 causal node evaluation output already exists")
    output.mkdir(parents=True)
    inference = args.inference_dir.resolve()
    manifest = json.loads((inference / "inference_manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("teacher_inputs_read") != 0
        or manifest.get("teacher_identity_inputs_read") != 0
        or manifest.get("future_reference_cells") != 0
        or manifest.get("causal_observations") != 24_462
        or manifest.get("unique_lidar_frames") != 32_678
        or manifest.get("structural_threshold") != STRUCTURAL_THRESHOLD
    ):
        raise RuntimeError("C09 inference/Teacher separation contract drift")
    with np.load(inference / "ensemble_deployment_outputs.npz", allow_pickle=False) as archive:
        global_index = archive["global_sequence_index"].astype(np.int64)
        parent = archive["parent_id"].astype(str)
        observation = archive["observation_id"].astype(str)
        traversal = archive["traversal_id"].astype(str)
        sequence = archive["sequence_index"].astype(np.int64)
        probability = archive["probability"].astype(np.float64)
        boundary = archive["boundary_offset_m"].astype(np.float64)
        uncertainty = archive["uncertainty"].astype(np.float64)
    teacher_rows = [
        row for row in _read_jsonl(args.teacher.resolve())
        if str(row["parent_id"]).endswith("_C09")
    ]
    teacher_rows.sort(key=lambda row: int(row["global_sequence_index"]))
    if (
        len(teacher_rows) != 24_462
        or not np.array_equal(global_index, [int(row["global_sequence_index"]) for row in teacher_rows])
        or not np.array_equal(parent, [str(row["parent_id"]) for row in teacher_rows])
        or not np.array_equal(observation, [str(row["observation_id"]) for row in teacher_rows])
        or not np.array_equal(traversal, [str(row["traversal_id"]) for row in teacher_rows])
        or not np.array_equal(sequence, [int(row["sequence_index"]) for row in teacher_rows])
    ):
        raise RuntimeError("C09 deployment output and evaluation Teacher are not bijective")
    bank = materialize_causal_episode_references(teacher_rows)
    metrics = evaluate_decision_event_triggers(
        probability, bank.event_index, bank.episode_id, traversal, sequence,
        boundary, uncertainty, structural_threshold=STRUCTURAL_THRESHOLD,
    )
    identity = _identity_coverage(
        probability, teacher_rows, bank, traversal, sequence, boundary, uncertainty
    )
    selection_summary = json.loads(args.selection_summary.resolve().read_text(encoding="utf-8"))
    if selection_summary["new_ensemble_trigger_metrics"]["structural_threshold"] != STRUCTURAL_THRESHOLD:
        raise RuntimeError("frozen causal episode threshold drift")
    selection = _selection_decision_metrics(selection_summary)
    requirements = {
        "decision_triggers_nonvacuous": metrics["predicted_decision_triggers"] > 0,
        "aggregate_precision_at_least_0p98": metrics["decision_trigger_precision"] >= .98,
        "aggregate_false_fraction_at_most_0p01": metrics["false_decision_trigger_fraction"] <= .01,
        "aggregate_recall_at_least_0p60": metrics["decision_episode_recall"] >= .60,
        "junction_precision_at_least_0p98": metrics["per_event"]["junction"]["precision"] >= .98,
        "junction_recall_at_least_0p50": metrics["per_event"]["junction"]["recall"] >= .50,
        "terminal_precision_at_least_0p98": metrics["per_event"]["terminal"]["precision"] >= .98,
        "terminal_recall_at_least_0p75": metrics["per_event"]["terminal"]["recall"] >= .75,
        "junction_identity_coverage_at_least_0p80": identity["junction"]["coverage"] >= .80,
        "terminal_identity_coverage_at_least_0p80": identity["terminal"]["coverage"] >= .80,
    }
    passed = all(requirements.values())
    result = {
        "schema_version": "gse_factorized_causal_node_c09_evaluation_v1",
        "overall_status": PASS_STATUS if passed else FAIL_STATUS,
        "scientific_pass": passed,
        "question": "Do frozen past-only causal episodes generate safe junction/terminal nodes on all C09 worlds?",
        "worlds": 10,
        "causal_observations": 24_462,
        "true_junction_episodes": 426,
        "true_terminal_episodes": 114,
        "true_junction_identities": 71,
        "true_terminal_identities": 59,
        "selection_decision_metrics": selection,
        "c09_decision_metrics": metrics,
        "c09_identity_coverage": identity,
        "requirements": requirements,
        "threshold_selection_steps": 0,
        "optimizer_steps": 0,
        "model_updates": 0,
        "inference_teacher_inputs_read": int(manifest["teacher_inputs_read"]),
        "evaluation_teacher_rows_read": len(teacher_rows),
        "c10_worlds_read": 0,
        "strict_test_worlds_read": 0,
        "mtare_worlds_read": 0,
    }
    (output / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _figure(output, selection, metrics, identity)
    print(json.dumps(result, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
