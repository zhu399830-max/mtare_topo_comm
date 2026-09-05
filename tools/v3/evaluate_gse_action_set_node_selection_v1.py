#!/usr/bin/env python3
"""Freeze the three-seed action-set node gate on C07--C08 only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mtare_topo.evaluation.gse_causal_episode_metrics import (
    evaluate_decision_mass_triggers,
    extract_decision_mass_triggers,
    select_decision_mass_threshold,
)


def _five_class(probability: np.ndarray) -> np.ndarray:
    values = np.asarray(probability, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not np.all(np.isfinite(values)):
        raise ValueError("action-set probability contract drift")
    result = np.zeros((len(values), 5), dtype=np.float64)
    result[:, :3] = values
    if np.any(result < 0.0) or not np.allclose(result.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("action-set probability simplex drift")
    return result


def _identity_coverage(
    probability: np.ndarray,
    target: np.ndarray,
    episode: np.ndarray,
    traversal: np.ndarray,
    sequence: np.ndarray,
    identity: np.ndarray,
    threshold: float,
) -> dict:
    uncertainty = -np.sum(
        probability * np.log(np.clip(probability, 1e-8, 1.0)), axis=1
    ) / np.log(5.0)
    triggers = extract_decision_mass_triggers(
        probability, traversal, sequence, uncertainty, decision_threshold=threshold
    )
    result = {}
    for event, name in ((1, "junction"), (2, "terminal")):
        truth = set(identity[target == event].tolist())
        correct = {
            str(identity[trigger.row]) for trigger in triggers
            if trigger.predicted_event_index == event
            and int(target[trigger.row]) == event and int(episode[trigger.row]) >= 0
        }
        result[name] = {
            "true_identities": len(truth),
            "correct_trigger_identities": len(correct),
            "coverage": len(correct) / len(truth) if truth else 0.0,
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--seed0", required=True, type=Path)
    parser.add_argument("--seed1", required=True, type=Path)
    parser.add_argument("--seed2", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise RuntimeError("action-set selection output exists; overwrite is forbidden")
    args.output_dir.mkdir(parents=True)

    partition = np.load(args.cache_dir / "partition_code.npy")
    selection_rows = np.flatnonzero(partition == 1)
    traversal_all = np.load(args.cache_dir / "traversal_id.npy", allow_pickle=False)
    sequence_all = np.load(args.cache_dir / "sequence_index.npy")
    identity_all = np.load(args.cache_dir / "identity.npy", allow_pickle=False)
    parent_all = np.load(args.cache_dir / "parent_id.npy", allow_pickle=False)
    expected_target = np.load(args.cache_dir / "decision_target.npy")[selection_rows]
    source_episode = np.load(args.cache_dir / "decision_episode_id.npy")[selection_rows]
    unique_episode = np.unique(source_episode[source_episode >= 0])
    remap = {int(value): index for index, value in enumerate(unique_episode.tolist())}
    episode = np.asarray([remap.get(int(value), -1) for value in source_episode], dtype=np.int64)
    traversal = traversal_all[selection_rows].astype(str)
    sequence = sequence_all[selection_rows].astype(np.int64)
    identity = identity_all[selection_rows].astype(str)
    parent = parent_all[selection_rows].astype(str)

    probabilities = []
    seed_metrics = {}
    for seed, path in enumerate((args.seed0, args.seed1, args.seed2)):
        with np.load(path.resolve(), allow_pickle=False) as archive:
            if (
                not np.array_equal(archive["observation_row"], selection_rows)
                or not np.array_equal(archive["decision_target"], expected_target)
                or not np.array_equal(archive["decision_episode_id"], episode)
            ):
                raise RuntimeError(f"seed{seed} action-set selection identity drift")
            probabilities.append(np.asarray(archive["probability"], dtype=np.float64))
    ensemble_three = np.mean(np.stack(probabilities), axis=0)
    ensemble = _five_class(ensemble_three)
    uncertainty = -np.sum(ensemble * np.log(np.clip(ensemble, 1e-8, 1.0)), axis=1) / np.log(5.0)
    threshold_error = None
    try:
        selected = select_decision_mass_threshold(
            ensemble, expected_target, episode, traversal, sequence, uncertainty,
            minimum_precision=.995, minimum_per_event_precision=.99,
            minimum_recall=.25, minimum_per_event_recall=.25,
        )
    except RuntimeError as exc:
        threshold_error = str(exc)
        diagnostics = []
        for candidate in np.linspace(0.0, 1.0, 1001):
            metrics = evaluate_decision_mass_triggers(
                ensemble, expected_target, episode, traversal, sequence, uncertainty,
                decision_threshold=float(candidate),
            )
            per_event = metrics["per_event"]
            if (
                metrics["predicted_decision_triggers"] > 0
                and metrics["decision_episode_recall"] >= .25
                and per_event["junction"]["recall"] >= .25
                and per_event["terminal"]["recall"] >= .25
            ):
                diagnostics.append(metrics)
        if diagnostics:
            selected = max(
                diagnostics,
                key=lambda value: (
                    min(value["decision_trigger_precision"], value["per_event"]["junction"]["precision"], value["per_event"]["terminal"]["precision"]),
                    value["decision_trigger_precision"], value["decision_episode_recall"],
                    -value["decision_threshold"],
                ),
            )
        else:
            raise RuntimeError("no action-set threshold even satisfies the minimum recall diagnostics")
    threshold = float(selected["decision_threshold"])
    for seed, values in enumerate(probabilities):
        five = _five_class(values)
        seed_uncertainty = -np.sum(
            five * np.log(np.clip(five, 1e-8, 1.0)), axis=1
        ) / np.log(5.0)
        seed_metrics[str(seed)] = evaluate_decision_mass_triggers(
            five, expected_target, episode, traversal, sequence, seed_uncertainty,
            decision_threshold=threshold,
        )

    per_family = {}
    for family in sorted(set(value[:3] for value in parent.tolist())):
        mask = np.asarray([value.startswith(family + "_") for value in parent], dtype=np.bool_)
        per_family[family] = evaluate_decision_mass_triggers(
            ensemble[mask], expected_target[mask], episode[mask], traversal[mask], sequence[mask],
            uncertainty[mask], decision_threshold=threshold,
        )
    identity_coverage = _identity_coverage(
        ensemble, expected_target, episode, traversal, sequence, identity, threshold
    )
    gates = {
        "aggregate_precision_at_least_0p995": selected["decision_trigger_precision"] >= .995,
        "aggregate_false_fraction_at_most_0p005": selected["false_decision_trigger_fraction"] <= .005,
        "aggregate_recall_at_least_0p25": selected["decision_episode_recall"] >= .25,
        "junction_precision_at_least_0p99": selected["per_event"]["junction"]["precision"] >= .99,
        "junction_recall_at_least_0p25": selected["per_event"]["junction"]["recall"] >= .25,
        "terminal_precision_at_least_0p99": selected["per_event"]["terminal"]["precision"] >= .99,
        "terminal_recall_at_least_0p25": selected["per_event"]["terminal"]["recall"] >= .25,
        "all_ten_families_have_a_correct_trigger": len(per_family) == 10 and all(
            value["correctly_classified_unique_decision_episodes"] > 0 for value in per_family.values()
        ),
    }
    passed = threshold_error is None and all(gates.values())
    summary = {
        "schema_version": "gse_action_set_node_selection_v1",
        "status": "PASS_GSE_ACTION_SET_NODE_SELECTION_V1" if passed else "FAIL_GSE_ACTION_SET_NODE_SELECTION_V1",
        "selection_observations": len(selection_rows),
        "selection_worlds": len(np.unique(parent)),
        "ensemble": selected,
        "threshold_selection_error": threshold_error,
        "seed_at_ensemble_threshold": seed_metrics,
        "identity_coverage_diagnostic": identity_coverage,
        "per_family": per_family,
        "gates": gates,
        "threshold_selection_rows": len(selection_rows),
        "c09_worlds_read": 0,
        "c10_worlds_read": 0,
        "mtare_worlds_read": 0,
    }
    np.savez_compressed(
        args.output_dir / "selection_ensemble_outputs.npz",
        observation_row=selection_rows,
        probability=ensemble.astype(np.float32),
        uncertainty=uncertainty.astype(np.float32),
        decision_target=expected_target,
        decision_episode_id=episode,
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
