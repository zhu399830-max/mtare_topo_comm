#!/usr/bin/env python3
"""Apply the frozen C07-C08 five-frame decision gate once to C09."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import softmax

from mtare_topo.evaluation.gse_causal_episode_metrics import (
    evaluate_decision_mass_triggers,
)
from mtare_topo.representation.gse_causal_episode_detector import (
    materialize_causal_episode_references,
)
from mtare_topo.representation.gse_causal_episode_runtime import (
    align_baseline_event_logits,
)
from select_gse_five_frame_decision_gate_v1 import _identity_coverage


PASS_STATUS = "PASS_GSE_FIVE_FRAME_DECISION_GATE_C09_V1"
FAIL_STATUS = "FAIL_GSE_FIVE_FRAME_DECISION_GATE_C09_V1"


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _figure(output: Path, selection: dict, validation: dict, selection_id: dict, validation_id: dict):
    import matplotlib.pyplot as plt

    source = {
        "selection_metrics": selection,
        "c09_metrics": validation,
        "selection_identity_coverage": selection_id,
        "c09_identity_coverage": validation_id,
    }
    (output / "gse_five_frame_decision_gate_c09_source.json").write_text(
        json.dumps(source, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    names = ["Aggregate", "Junction", "Terminal"]
    x = np.arange(3)
    selection_p = [selection["decision_trigger_precision"], *[
        selection["per_event"][name]["precision"] for name in ("junction", "terminal")
    ]]
    validation_p = [validation["decision_trigger_precision"], *[
        validation["per_event"][name]["precision"] for name in ("junction", "terminal")
    ]]
    selection_r = [selection["decision_episode_recall"], *[
        selection["per_event"][name]["recall"] for name in ("junction", "terminal")
    ]]
    validation_r = [validation["decision_episode_recall"], *[
        validation["per_event"][name]["recall"] for name in ("junction", "terminal")
    ]]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), constrained_layout=True)
    for axis, left, right, title in (
        (axes[0], selection_p, validation_p, "Decision-trigger precision"),
        (axes[1], selection_r, validation_r, "Decision-episode recall"),
    ):
        axis.bar(x - .18, left, .36, color="#4C78A8", label="C07-C08 selection")
        axis.bar(x + .18, right, .36, color="#F58518", label="C09 validation")
        axis.set_xticks(x, names); axis.set_ylim(0, 1.05); axis.set_ylabel("Score")
        axis.set_title(title); axis.grid(axis="y", alpha=.25)
    axes[0].legend(frameon=False, loc="lower left")
    fig.suptitle("Five-frame junction/terminal decision gate")
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"gse_five_frame_decision_gate_c09.{suffix}", dpi=220)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--training-run", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise RuntimeError("five-frame C09 application output already exists")
    output.mkdir(parents=True)
    calibration = json.loads(args.calibration.resolve().read_text(encoding="utf-8"))
    if (
        calibration.get("c09_worlds_read") != 0
        or calibration.get("selection_worlds") != 20
        or calibration.get("selection_observations") != 45_942
        or calibration.get("threshold_grid_points") != 1001
    ):
        raise RuntimeError("five-frame decision calibration drift")
    threshold = float(calibration["decision_threshold"])
    rows = [
        row for row in _read_jsonl(args.teacher.resolve())
        if str(row["parent_id"]).endswith("_C09")
    ]
    rows.sort(key=lambda row: int(row["global_sequence_index"]))
    if len(rows) != 24_462 or len({str(row["parent_id"]) for row in rows}) != 10:
        raise RuntimeError("five-frame C09 Teacher population drift")
    global_index = np.asarray([int(row["global_sequence_index"]) for row in rows], dtype=np.int64)
    parent = np.asarray([str(row["parent_id"]) for row in rows])
    seed_probability = []
    training = args.training_run.resolve()
    for seed in range(3):
        with np.load(
            training / f"artifacts/models/seed{seed}/validation_outputs.npz",
            allow_pickle=False,
        ) as archive:
            logits = align_baseline_event_logits(
                archive["global_sequence_index"], archive["parent_id"], archive["event_logits"],
                global_index, parent,
            )
        seed_probability.append(softmax(logits.astype(np.float64), axis=1))
    ensemble = np.mean(np.stack(seed_probability), axis=0)
    bank = materialize_causal_episode_references(rows)
    traversal = np.asarray([str(row["traversal_id"]) for row in rows])
    sequence = np.asarray([int(row["sequence_index"]) for row in rows], dtype=np.int64)
    metrics = evaluate_decision_mass_triggers(
        ensemble, bank.event_index, bank.episode_id, traversal, sequence,
        np.zeros(len(rows)), decision_threshold=threshold,
    )
    identity = _identity_coverage(ensemble, rows, bank, threshold)
    requirements = {
        "nonvacuous": metrics["predicted_decision_triggers"] > 0,
        "aggregate_precision_at_least_0p98": metrics["decision_trigger_precision"] >= .98,
        "aggregate_false_at_most_0p01": metrics["false_decision_trigger_fraction"] <= .01,
        "aggregate_recall_at_least_0p40": metrics["decision_episode_recall"] >= .40,
        "junction_precision_at_least_0p98": metrics["per_event"]["junction"]["precision"] >= .98,
        "junction_recall_at_least_0p40": metrics["per_event"]["junction"]["recall"] >= .40,
        "terminal_precision_at_least_0p98": metrics["per_event"]["terminal"]["precision"] >= .98,
        "terminal_recall_at_least_0p60": metrics["per_event"]["terminal"]["recall"] >= .60,
        "junction_identity_coverage_at_least_0p80": identity["junction"]["coverage"] >= .80,
        "terminal_identity_coverage_at_least_0p80": identity["terminal"]["coverage"] >= .80,
    }
    passed = all(requirements.values())
    result = {
        "schema_version": "gse_five_frame_decision_gate_c09_application_v1",
        "overall_status": PASS_STATUS if passed else FAIL_STATUS,
        "scientific_pass": passed, "decision_threshold": threshold,
        "selection_metrics": calibration["selection_metrics"],
        "selection_identity_coverage": calibration["selection_identity_coverage"],
        "c09_metrics": metrics, "c09_identity_coverage": identity,
        "requirements": requirements,
        "selection_process_c09_worlds_read": int(calibration["c09_worlds_read"]),
        "c09_worlds": 10, "c09_observations": 24_462,
        "model_inference_frames": 0, "optimizer_steps": 0, "model_updates": 0,
        "threshold_selection_steps_on_c09": 0, "c10_worlds_read": 0,
        "strict_test_worlds_read": 0, "mtare_worlds_read": 0,
    }
    np.savez_compressed(
        output / "c09_outputs.npz", global_sequence_index=global_index,
        probability=ensemble.astype(np.float32), event_index=bank.event_index,
        episode_id=bank.episode_id,
    )
    (output / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _figure(
        output, calibration["selection_metrics"], metrics,
        calibration["selection_identity_coverage"], identity,
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
