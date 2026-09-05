#!/usr/bin/env python3
"""C01--C08 zero-training capacity audit for structured learned exits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from mtare_topo.representation.gse_causal_episode_detector import materialize_causal_episode_references
from mtare_topo.representation.gse_route_conditioned_node import (
    RouteConditionedNodeConfig,
    evaluate_structured_decision_events,
    route_conditioned_action_scores,
    structured_decision_events,
)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _compact_episode(source: np.ndarray) -> np.ndarray:
    unique = np.unique(source[source >= 0])
    lookup = {int(value): index for index, value in enumerate(unique.tolist())}
    return np.asarray([lookup.get(int(value), -1) for value in source], dtype=np.int64)


def _gates(metrics: dict, *, fit: bool) -> dict[str, bool]:
    overall_precision = .997 if fit else .995
    per_precision = .995 if fit else .99
    false_limit = .003 if fit else .005
    return {
        "aggregate_precision": metrics["decision_trigger_precision"] >= overall_precision,
        "aggregate_false": metrics["false_decision_trigger_fraction"] <= false_limit,
        "aggregate_recall": metrics["decision_episode_recall"] >= .25,
        "junction_precision": metrics["per_event"]["junction"]["precision"] >= per_precision,
        "junction_recall": metrics["per_event"]["junction"]["recall"] >= .25,
        "terminal_precision": metrics["per_event"]["terminal"]["precision"] >= per_precision,
        "terminal_recall": metrics["per_event"]["terminal"]["recall"] >= .25,
    }


def _record(config: RouteConditionedNodeConfig, metrics: dict) -> dict:
    clean = {key: value for key, value in metrics.items() if key != "trigger_rows"}
    return {"config": config.__dict__, "metrics": clean}


def _identity_coverage(metrics: dict, predicted: np.ndarray, target: np.ndarray, identity: np.ndarray) -> dict:
    trigger_rows = metrics["trigger_rows"]
    result = {}
    for event, name in ((1, "junction"), (2, "terminal")):
        truth = set(identity[target == event].tolist())
        correct = {str(identity[row]) for row in trigger_rows if int(target[row]) == event and int(predicted[row]) == event}
        result[name] = {"true_identities": len(truth), "correct_trigger_identities": len(correct), "coverage": len(correct) / len(truth) if truth else 0.0}
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--pair-cache", required=True, type=Path)
    parser.add_argument("--seed0", required=True, type=Path)
    parser.add_argument("--seed1", required=True, type=Path)
    parser.add_argument("--seed2", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    if args.output_dir.exists():
        raise RuntimeError("structured node output exists; overwrite is forbidden")
    args.output_dir.mkdir(parents=True)
    rows = _read_jsonl(args.teacher.resolve())
    if len(rows) != 188126:
        raise RuntimeError("structured node Teacher population drift")
    global_ids = np.asarray([int(row["global_sequence_index"]) for row in rows], dtype=np.int64)
    traversal = np.asarray([str(row["traversal_id"]) for row in rows])
    sequence = np.asarray([int(row["sequence_index"]) for row in rows], dtype=np.int64)
    identity = np.asarray([str(row["identity"]) for row in rows])
    parent = np.asarray([str(row["parent_id"]) for row in rows])
    bank = materialize_causal_episode_references(rows)
    target = np.where(np.isin(bank.event_index, (1, 2)), bank.event_index, 0).astype(np.int8)
    source_episode = np.where(target > 0, bank.episode_id, -1)
    with np.load(args.pair_cache.resolve(), allow_pickle=False) as archive:
        if not np.array_equal(archive["compact_to_global_sequence_index"], global_ids):
            raise RuntimeError("structured node compact identity drift")
        partition = archive["partition_code"].astype(np.uint8)
    if int(np.sum(partition == 0)) != 142184 or int(np.sum(partition == 1)) != 45942:
        raise RuntimeError("structured node split count drift")
    # Frozen sensor poses follow traversal tangent, so the executed incoming
    # action is exactly behind the robot in the current [sin,cos] frame.
    incoming_heading = np.tile(np.asarray([0.0, -1.0]), (len(rows), 1))
    token_paths = (args.seed0, args.seed1, args.seed2)
    token = {}
    for seed, path in enumerate(token_paths):
        with np.load(path.resolve(), allow_pickle=False) as archive:
            if not np.array_equal(archive["global_sequence_index"], global_ids):
                raise RuntimeError(f"structured node seed{seed} identity drift")
            token[seed] = (
                np.asarray(archive["exit_confidence"], dtype=np.float32),
                np.asarray(archive["exit_heading_unit"], dtype=np.float32),
            )
    scores_by_angle = {}
    for angle in (20.0, 35.0, 50.0):
        scores_by_angle[angle] = {
            seed: route_conditioned_action_scores(
                token[seed][0], token[seed][1], incoming_heading,
                incoming_half_angle_deg=angle,
            )
            for seed in (0, 1, 2)
        }
    fit_rows = np.flatnonzero(partition == 0)
    selection_rows = np.flatnonzero(partition == 1)
    fit_episode = _compact_episode(source_episode[fit_rows])
    selection_episode = _compact_episode(source_episode[selection_rows])
    grid = []
    passing = []
    for threshold in np.linspace(.02, 1.0, 50):
        for angle in (20.0, 35.0, 50.0):
            fit_scores = {seed: (values[0][fit_rows], values[1][fit_rows]) for seed, values in scores_by_angle[angle].items()}
            for persistence in (1, 2, 3):
                for consensus in (2, 3):
                    config = RouteConditionedNodeConfig(float(threshold), angle, persistence, consensus)
                    predicted = structured_decision_events(
                        fit_scores, traversal[fit_rows], sequence[fit_rows], config
                    )
                    metrics = evaluate_structured_decision_events(
                        predicted, target[fit_rows], fit_episode,
                        traversal[fit_rows], sequence[fit_rows],
                    )
                    gates = _gates(metrics, fit=True)
                    record = _record(config, metrics)
                    record["gates"] = gates
                    grid.append(record)
                    if all(gates.values()):
                        passing.append((config, metrics))
    with (args.output_dir / "fit_parameter_grid.jsonl").open("w", encoding="utf-8") as stream:
        for record in grid:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    selected_config = None
    selection_metrics = None
    selection_gates = None
    selection_families = {}
    identity_coverage = {}
    if passing:
        selected_config, fit_metrics = max(
            passing,
            key=lambda value: (
                value[1]["decision_episode_recall"],
                min(value[1]["per_event"]["junction"]["recall"], value[1]["per_event"]["terminal"]["recall"]),
                value[1]["decision_trigger_precision"],
                -value[0].confidence_threshold, -value[0].incoming_half_angle_deg,
                -value[0].persistence_observations, -value[0].seed_consensus,
            ),
        )
        selection_scores = {seed: (values[0][selection_rows], values[1][selection_rows]) for seed, values in scores_by_angle[selected_config.incoming_half_angle_deg].items()}
        selection_predicted = structured_decision_events(
            selection_scores, traversal[selection_rows], sequence[selection_rows], selected_config
        )
        selection_metrics = evaluate_structured_decision_events(
            selection_predicted, target[selection_rows], selection_episode,
            traversal[selection_rows], sequence[selection_rows],
        )
        selection_gates = _gates(selection_metrics, fit=False)
        identity_coverage = _identity_coverage(
            selection_metrics, selection_predicted, target[selection_rows], identity[selection_rows]
        )
        for family in sorted(set(value[:3] for value in parent[selection_rows].tolist())):
            mask = np.asarray([value.startswith(family + "_") for value in parent[selection_rows]], dtype=np.bool_)
            family_episode = _compact_episode(source_episode[selection_rows][mask])
            family_metrics = evaluate_structured_decision_events(
                selection_predicted[mask], target[selection_rows][mask], family_episode,
                traversal[selection_rows][mask], sequence[selection_rows][mask],
            )
            selection_families[family] = _record(selected_config, family_metrics)["metrics"]
        selection_gates["all_ten_families_have_correct_trigger"] = len(selection_families) == 10 and all(value["correctly_classified_unique_decision_episodes"] > 0 for value in selection_families.values())
    diagnostic = max(
        grid,
        key=lambda record: (
            min(record["metrics"]["decision_trigger_precision"], record["metrics"]["per_event"]["junction"]["precision"], record["metrics"]["per_event"]["terminal"]["precision"])
            if record["metrics"]["decision_episode_recall"] >= .25 and record["metrics"]["per_event"]["junction"]["recall"] >= .25 and record["metrics"]["per_event"]["terminal"]["recall"] >= .25 else -1.0,
            record["metrics"]["decision_episode_recall"],
        ),
    )
    passed = bool(passing) and all(selection_gates.values())
    summary = {
        "schema_version": "gse_route_conditioned_structured_node_capacity_v1",
        "status": "PASS_GSE_ROUTE_CONDITIONED_STRUCTURED_NODE_CAPACITY_V1" if passed else "FAIL_GSE_ROUTE_CONDITIONED_STRUCTURED_NODE_CAPACITY_V1",
        "grid_configurations": len(grid), "fit_passing_configurations": len(passing),
        "fit_best_minimum_recall_diagnostic": diagnostic,
        "selected_config": selected_config.__dict__ if selected_config else None,
        "selected_fit_metrics": _record(selected_config, fit_metrics)["metrics"] if selected_config else None,
        "selection_metrics": _record(selected_config, selection_metrics)["metrics"] if selected_config else None,
        "selection_gates": selection_gates,
        "selection_per_family": selection_families,
        "selection_identity_coverage": identity_coverage,
        "fit_observations": len(fit_rows), "selection_observations": len(selection_rows),
        "optimizer_steps": 0, "model_inference_frames": 0,
        "duration_seconds": time.monotonic() - started,
        "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
