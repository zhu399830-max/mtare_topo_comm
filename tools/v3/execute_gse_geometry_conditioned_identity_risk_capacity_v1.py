#!/usr/bin/env python3
"""Fit/evaluate the bounded C01-C08 geometry-conditioned event-risk proof."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from mtare_topo.governance import write_json
from mtare_topo.representation.gse_corrected_causal_event import corrected_causal_event_gate
from mtare_topo.representation.gse_geometry_conditioned_risk import (
    DEFAULT_CAUSAL_LAGS,
    EVENT_NAMES,
    causal_geometry_risk_features,
    fit_geometry_conditioned_risk_readout,
    identity_balanced_event_weights,
)
from mtare_topo.representation.gse_rare_event_corrective import (
    evaluate_rare_event_corrective,
)


PASS_STATUS = "PASS_GSE_GEOMETRY_CONDITIONED_IDENTITY_RISK_CAPACITY_V1"
FAIL_STATUS = "FAIL_GSE_GEOMETRY_CONDITIONED_IDENTITY_RISK_CAPACITY_V1"
FEATURE_SCALE = np.asarray((30.0, 30.0, 45.0, 0.1), dtype=np.float64)
EVENT_INDEX = {name: index for index, name in enumerate(EVENT_NAMES)}
L2_STRENGTH = 1e-3
MAXIMUM_ITERATIONS = 500


def _read_teacher(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if len(rows) != 188126:
        raise RuntimeError(f"corrected Teacher population drift: {len(rows)}")
    return rows


def _factor_identity(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(
        ["" if row.get("identity") is None else str(row["identity"]) for row in rows],
        dtype=str,
    )
    unique = sorted(set(values[values != ""].tolist()))
    code = {value: index for index, value in enumerate(unique)}
    encoded = np.asarray([code.get(value, -1) for value in values], dtype=np.int64)
    return values, encoded


def _argmax_metrics(probability: np.ndarray, truth: np.ndarray) -> dict:
    predicted = np.argmax(probability, axis=1)
    matrix = np.zeros((len(EVENT_NAMES), len(EVENT_NAMES)), dtype=np.int64)
    np.add.at(matrix, (truth, predicted), 1)
    per_class = {}
    f1 = []
    for index, name in enumerate(EVENT_NAMES):
        tp = int(matrix[index, index])
        actual = int(matrix[index].sum())
        predicted_count = int(matrix[:, index].sum())
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / actual if actual else 0.0
        value = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {"precision": precision, "recall": recall, "f1": value, "support": actual}
        f1.append(value)
    return {"macro_f1": float(np.mean(f1)), "per_class": per_class, "confusion": matrix.tolist()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--pair-cache", required=True, type=Path)
    parser.add_argument("--feature", action="append", required=True, type=Path)
    args = parser.parse_args()
    if len(args.feature) != 3:
        raise RuntimeError("identity-risk capacity proof requires exactly three frozen feature arrays")
    run_dir = args.run_dir.resolve()
    artifacts = run_dir / "artifacts"
    metrics_dir = run_dir / "metrics"
    for directory in (artifacts, metrics_dir):
        directory.mkdir(parents=True, exist_ok=True)

    rows = _read_teacher(args.teacher)
    with np.load(args.pair_cache, allow_pickle=False) as archive:
        global_index = np.asarray(archive["compact_to_global_sequence_index"], dtype=np.int64)
        partition = np.asarray(archive["partition_code"], dtype=np.uint8)
        parent_cache = np.asarray(archive["parent_id"], dtype=str)
    teacher_global = np.asarray([int(row["global_sequence_index"]) for row in rows])
    parent = np.asarray([str(row["parent_id"]) for row in rows])
    if (
        not np.array_equal(global_index, teacher_global)
        or not np.array_equal(parent_cache, parent)
        or tuple(np.bincount(partition, minlength=2)) != (142184, 45942)
        or any(value.endswith(("_C09", "_C10")) for value in parent)
    ):
        raise RuntimeError("identity-risk compact population/split drift")
    event = np.asarray([EVENT_INDEX[str(row["event"])] for row in rows], dtype=np.int64)
    identity_text, identity = _factor_identity(rows)
    if np.any((event == 0) != (identity < 0)):
        raise RuntimeError("corrected event/identity contract drift")
    family = np.asarray([value.split("_", 1)[0] for value in parent], dtype=str)
    if sorted(set(family.tolist())) != [f"S{index:02d}" for index in range(1, 11)]:
        raise RuntimeError("identity-risk topology-family drift")
    traversal = np.asarray([str(row["traversal_id"]) for row in rows], dtype=str)
    sequence = np.asarray([int(row["sequence_index"]) for row in rows], dtype=np.int64)
    geometry_valid = np.asarray([bool(row.get("geometry_valid")) for row in rows], dtype=np.bool_)

    geometry_seed = []
    probability_seed = []
    uncertainty_seed = []
    for path in args.feature:
        feature = np.load(path, mmap_mode="r")
        if feature.shape != (188126, 146) or feature.dtype != np.float32:
            raise RuntimeError(f"identity-risk frozen feature drift: {path}")
        probability = np.asarray(feature[:, :5], dtype=np.float64)
        probability /= probability.sum(axis=1, keepdims=True)
        geometry_seed.append(np.asarray(feature[:, 8:12], dtype=np.float64) * FEATURE_SCALE)
        probability_seed.append(probability)
        uncertainty_seed.append(np.asarray(feature[:, 140], dtype=np.float64))
    geometry = np.mean(np.stack(geometry_seed), axis=0)
    baseline_probability = np.mean(np.stack(probability_seed), axis=0)
    baseline_probability /= baseline_probability.sum(axis=1, keepdims=True)
    uncertainty = np.mean(np.stack(uncertainty_seed), axis=0)
    features = causal_geometry_risk_features(
        geometry,
        uncertainty,
        baseline_probability,
        traversal,
        sequence,
        partition,
        geometry_valid,
        lags=DEFAULT_CAUSAL_LAGS,
    )
    fit = partition == 0
    selection = partition == 1
    weights = identity_balanced_event_weights(event, identity_text, parent, fit)
    readout, fit_summary = fit_geometry_conditioned_risk_readout(
        features,
        event,
        weights,
        l2_strength=L2_STRENGTH,
        maximum_iterations=MAXIMUM_ITERATIONS,
    )
    probability = readout.predict_probability(features.values)
    selection_metrics = evaluate_rare_event_corrective(
        probability[selection], event[selection], identity[selection], family[selection]
    )
    baseline_metrics = evaluate_rare_event_corrective(
        baseline_probability[selection], event[selection], identity[selection], family[selection]
    )
    gate = corrected_causal_event_gate(selection_metrics)

    leave_family_out = []
    for held_family in sorted(set(family.tolist())):
        diagnostic_fit = fit & (family != held_family)
        diagnostic_weights = identity_balanced_event_weights(
            event, identity_text, parent, diagnostic_fit
        )
        diagnostic_readout, diagnostic_summary = fit_geometry_conditioned_risk_readout(
            features,
            event,
            diagnostic_weights,
            l2_strength=L2_STRENGTH,
            maximum_iterations=MAXIMUM_ITERATIONS,
        )
        held = fit & (family == held_family)
        held_metrics = _argmax_metrics(
            diagnostic_readout.predict_probability(features.values[held]), event[held]
        )
        leave_family_out.append(
            {
                "held_family": held_family,
                "held_observations": int(held.sum()),
                "macro_f1": held_metrics["macro_f1"],
                "junction_f1": held_metrics["per_class"]["junction"]["f1"],
                "terminal_f1": held_metrics["per_class"]["terminal"]["f1"],
                "turn_f1": held_metrics["per_class"]["turn"]["f1"],
                "geometry_transition_f1": held_metrics["per_class"]["geometry_transition"]["f1"],
                "optimizer_iterations": diagnostic_summary.optimizer_iterations,
            }
        )

    np.savez_compressed(
        artifacts / "geometry_conditioned_risk_readout.npz",
        feature_names=np.asarray(features.names, dtype=str),
        mean=readout.standardizer.mean,
        scale=readout.standardizer.scale,
        coefficient=readout.coefficient,
        intercept=readout.intercept,
        lags=np.asarray(DEFAULT_CAUSAL_LAGS, dtype=np.int64),
        l2_strength=np.asarray(L2_STRENGTH, dtype=np.float64),
    )
    np.savez_compressed(
        artifacts / "selection_outputs.npz",
        global_sequence_index=global_index[selection],
        probability=probability[selection].astype(np.float32),
        baseline_probability=baseline_probability[selection].astype(np.float32),
        event=event[selection].astype(np.int8),
        identity=identity[selection],
        family=family[selection],
    )
    np.savez_compressed(
        artifacts / "fit_contract.npz",
        global_sequence_index=global_index[fit],
        sample_weight=weights[fit],
        event=event[fit].astype(np.int8),
        feature_names=np.asarray(features.names, dtype=str),
    )
    with (artifacts / "leave_family_out.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(leave_family_out[0]))
        writer.writeheader()
        writer.writerows(leave_family_out)

    overall = PASS_STATUS if gate["passed"] else FAIL_STATUS
    result = {
        "schema_version": "gse_geometry_conditioned_identity_risk_capacity_v1",
        "overall_status": overall,
        "scientific_pass": bool(gate["passed"]),
        "method": "deterministic identity-balanced linear softmax over six fixed causal geometry lags, frozen conditional event evidence and uncertainty",
        "feature_count": len(features.names),
        "lags": list(DEFAULT_CAUSAL_LAGS),
        "l2_strength": L2_STRENGTH,
        "fit_worlds": 60,
        "fit_observations": int(fit.sum()),
        "selection_worlds": 20,
        "selection_observations": int(selection.sum()),
        "selection_change_identities": len(set(identity[selection & (event == 4)].tolist())),
        "fit_summary": fit_summary.to_dict(),
        "baseline_selection_metrics": baseline_metrics,
        "selection_metrics": selection_metrics,
        "gate": gate,
        "leave_family_out": leave_family_out,
        "optimizer_iterations": fit_summary.optimizer_iterations + sum(row["optimizer_iterations"] for row in leave_family_out),
        "risk_readout_scored_observations": int(len(rows) + fit.sum()),
        "upstream_model_inference_frames": 0,
        "c09_worlds_read": 0,
        "strict_test_worlds_read": 0,
        "mtare_worlds_read": 0,
    }
    write_json(metrics_dir / "capacity_summary.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if gate["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
