#!/usr/bin/env python3
"""Evaluate the frozen three-seed causal episode ensemble on C07-C08 only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mtare_topo.evaluation.gse_causal_episode_metrics import (
    extract_causal_event_triggers,
    select_structural_threshold,
)
from mtare_topo.representation.gse_corrected_causal_event import (
    OLD_DIRECTIONAL_ENSEMBLE_MACRO_F1,
    REQUIRED_EVENT_MACRO_F1,
)
from mtare_topo.semantics.geometric_semantics import EVENT_NAMES


PASS_STATUS = "PASS_GSE_CAUSAL_EPISODE_ENSEMBLE_V1"
FAIL_STATUS = "FAIL_GSE_CAUSAL_EPISODE_ENSEMBLE_V1"


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _frame_metrics(probability: np.ndarray, target: np.ndarray) -> dict:
    predicted = np.argmax(probability, axis=1)
    per_event = {}
    f1_values = []
    for index, name in enumerate(EVENT_NAMES):
        correct = int(np.sum((predicted == index) & (target == index)))
        predicted_count = int(np.sum(predicted == index))
        target_count = int(np.sum(target == index))
        precision = correct / predicted_count if predicted_count else 0.0
        recall = correct / target_count if target_count else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_event[name] = {
            "precision": precision, "recall": recall, "f1": f1,
            "target_rows": target_count, "predicted_rows": predicted_count,
        }
    return {
        "accuracy": float(np.mean(predicted == target)),
        "macro_f1": float(np.mean(f1_values)),
        "per_event": per_event,
    }


def _identity_coverage(
    probability: np.ndarray,
    target: np.ndarray,
    episode_id: np.ndarray,
    identity: np.ndarray,
    traversal: np.ndarray,
    sequence: np.ndarray,
    boundary: np.ndarray,
    uncertainty: np.ndarray,
    threshold: float,
) -> dict:
    triggers = extract_causal_event_triggers(
        probability, traversal, sequence, boundary, uncertainty,
        structural_threshold=threshold,
    )
    matched_episode: set[int] = set()
    correct_identity = {index: set() for index in range(1, len(EVENT_NAMES))}
    for trigger in triggers:
        row = trigger.row
        episode = int(episode_id[row])
        if episode < 0 or episode in matched_episode:
            continue
        matched_episode.add(episode)
        if trigger.predicted_event_index == int(target[row]):
            correct_identity[int(target[row])].add(str(identity[row]))
    result = {}
    for event in range(1, len(EVENT_NAMES)):
        truth = set(identity[target == event].tolist())
        truth.discard("")
        correct = correct_identity[event]
        result[EVENT_NAMES[event]] = {
            "correct_identities": len(correct),
            "true_identities": len(truth),
            "coverage": len(correct) / len(truth) if truth else 0.0,
        }
    return result


def _threshold_metrics(
    probability, target, episode, traversal, sequence, boundary, uncertainty,
) -> tuple[dict | None, str | None]:
    try:
        metrics = select_structural_threshold(
            probability, target, episode, traversal, sequence, boundary, uncertainty,
            minimum_precision=.99,
        )
        return metrics, None
    except RuntimeError as exc:
        return None, str(exc)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-root", required=True, type=Path)
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--old-directional-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise RuntimeError("ensemble output already exists; overwrite is forbidden")
    args.output_dir.mkdir(parents=True)
    teacher = _read_jsonl(args.teacher.resolve())
    if len(teacher) != 188126:
        raise RuntimeError("ensemble Teacher count drift")
    global_all = np.asarray([int(row["global_sequence_index"]) for row in teacher], dtype=np.int64)
    traversal_all = np.asarray([str(row["traversal_id"]) for row in teacher])
    sequence_all = np.asarray([int(row["sequence_index"]) for row in teacher], dtype=np.int64)
    identity_all = np.asarray([
        "" if row.get("identity") is None else str(row["identity"]) for row in teacher
    ])

    seed_probabilities = []
    seed_boundaries = []
    seed_uncertainties = []
    seed_frame_metrics = {}
    rows = None
    target = None
    episode = None
    boundary = None
    uncertainty = None
    for seed in (0, 1, 2):
        with np.load(
            args.training_root / f"seed{seed}/selection_outputs.npz", allow_pickle=False
        ) as archive:
            current_rows = archive["observation_row"].astype(np.int64)
            current_probability = archive["probability"].astype(np.float64)
            current_target = archive["event_index"].astype(np.int64)
            current_episode = archive["episode_id"].astype(np.int64)
            current_boundary = archive["boundary_offset_m"].astype(np.float64)
            current_uncertainty = archive["uncertainty"].astype(np.float64)
        if rows is None:
            rows, target, episode = current_rows, current_target, current_episode
            boundary, uncertainty = current_boundary, current_uncertainty
        elif (
            not np.array_equal(rows, current_rows)
            or not np.array_equal(target, current_target)
            or not np.array_equal(episode, current_episode)
        ):
            raise RuntimeError("three-seed selection identity alignment drift")
        seed_probabilities.append(current_probability)
        seed_boundaries.append(current_boundary)
        seed_uncertainties.append(current_uncertainty)
        seed_frame_metrics[str(seed)] = _frame_metrics(current_probability, current_target)
    assert rows is not None and target is not None and episode is not None
    assert boundary is not None and uncertainty is not None
    if len(rows) != 45942:
        raise RuntimeError("ensemble selection count drift")
    global_index = global_all[rows]
    traversal = traversal_all[rows]
    sequence = sequence_all[rows]
    identity = identity_all[rows]
    ensemble_probability = np.mean(np.stack(seed_probabilities), axis=0)
    boundary = np.mean(np.stack(seed_boundaries), axis=0)
    uncertainty = np.mean(np.stack(seed_uncertainties), axis=0)
    ensemble_frame = _frame_metrics(ensemble_probability, target)
    ensemble_trigger, ensemble_threshold_error = _threshold_metrics(
        ensemble_probability, target, episode, traversal, sequence, boundary, uncertainty
    )

    old_probabilities = []
    old_seed_frame = {}
    for seed in (0, 1, 2):
        with np.load(
            args.old_directional_root / f"seed{seed}/selection_outputs.npz", allow_pickle=False
        ) as archive:
            old_global = archive["global_sequence_index"].astype(np.int64)
            old_probability = archive["probability"].astype(np.float64)
        if not np.array_equal(old_global, global_index):
            raise RuntimeError("old directional baseline population drift")
        old_probabilities.append(old_probability)
        old_seed_frame[str(seed)] = _frame_metrics(old_probability, target)
    with np.load(
        args.old_directional_root / "ensemble_selection_outputs.npz", allow_pickle=False
    ) as archive:
        old_global = archive["global_sequence_index"].astype(np.int64)
        old_ensemble = archive["probability"].astype(np.float64)
    if not np.array_equal(old_global, global_index):
        raise RuntimeError("old directional ensemble identity drift")
    old_frame = _frame_metrics(old_ensemble, target)
    if not np.isclose(old_frame["macro_f1"], OLD_DIRECTIONAL_ENSEMBLE_MACRO_F1, atol=1e-12):
        raise RuntimeError("sealed old directional macro-F1 drift")
    old_trigger, old_threshold_error = _threshold_metrics(
        old_ensemble, target, episode, traversal, sequence,
        np.zeros(len(rows)), np.zeros(len(rows)),
    )

    identity_coverage = None
    if ensemble_trigger is not None:
        identity_coverage = _identity_coverage(
            ensemble_probability, target, episode, identity, traversal, sequence,
            boundary, uncertainty, ensemble_trigger["structural_threshold"],
        )
    gain = ensemble_frame["macro_f1"] - old_frame["macro_f1"]
    seed_gains = {
        str(seed): seed_frame_metrics[str(seed)]["macro_f1"] - old_seed_frame[str(seed)]["macro_f1"]
        for seed in (0, 1, 2)
    }
    requirements = {
        "ensemble_frame_macro_f1_at_least_0p737904": ensemble_frame["macro_f1"] >= REQUIRED_EVENT_MACRO_F1,
        "ensemble_gain_at_least_0p05": gain >= .05,
        "at_least_two_seed_gains_at_least_0p05": sum(value >= .05 for value in seed_gains.values()) >= 2,
        "no_seed_regression": all(value >= 0.0 for value in seed_gains.values()),
        "nonvacuous_trigger_threshold": ensemble_trigger is not None,
        "structural_trigger_precision_at_least_0p98": bool(ensemble_trigger and ensemble_trigger["structural_trigger_precision"] >= .98),
        "false_trigger_fraction_at_most_0p01": bool(ensemble_trigger and 1.0 - ensemble_trigger["structural_trigger_precision"] <= .01),
        "structural_episode_recall_at_least_0p40": bool(ensemble_trigger and ensemble_trigger["structural_episode_recall"] >= .40),
        "junction_identity_coverage_at_least_0p90": bool(identity_coverage and identity_coverage["junction"]["coverage"] >= .90),
        "terminal_identity_coverage_at_least_0p90": bool(identity_coverage and identity_coverage["terminal"]["coverage"] >= .90),
        "turn_identity_coverage_at_least_0p40": bool(identity_coverage and identity_coverage["turn"]["coverage"] >= .40),
        "transition_correct_identities_at_least_7_of_17": bool(identity_coverage and identity_coverage["geometry_transition"]["correct_identities"] >= 7 and identity_coverage["geometry_transition"]["true_identities"] == 17),
    }
    passed = all(requirements.values())
    result = {
        "schema_version": "gse_causal_episode_ensemble_evaluation_v1",
        "overall_status": PASS_STATUS if passed else FAIL_STATUS,
        "scientific_pass": passed,
        "selection_observations": len(rows),
        "new_ensemble_frame_metrics": ensemble_frame,
        "old_directional_frame_metrics": old_frame,
        "ensemble_frame_macro_f1_gain": gain,
        "seed_frame_metrics": seed_frame_metrics,
        "old_seed_frame_metrics": old_seed_frame,
        "seed_macro_f1_gains": seed_gains,
        "new_ensemble_trigger_metrics": ensemble_trigger,
        "new_ensemble_threshold_error": ensemble_threshold_error,
        "old_directional_trigger_metrics": old_trigger,
        "old_directional_threshold_error": old_threshold_error,
        "identity_coverage": identity_coverage,
        "requirements": requirements,
        "strict_test_worlds_read": 0,
        "c09_worlds_read": 0,
        "mtare_worlds_read": 0,
    }
    np.savez_compressed(
        args.output_dir / "ensemble_selection_outputs.npz",
        observation_row=rows,
        global_sequence_index=global_index,
        probability=ensemble_probability.astype(np.float32),
        event_index=target,
        episode_id=episode,
        boundary_offset_m=boundary.astype(np.float32),
        uncertainty=uncertainty.astype(np.float32),
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
