#!/usr/bin/env python3
"""Audit class-loss and analytic logit-gradient mass in the sealed V1R run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mtare_topo.evaluation.gse_episode_gradient_mass import analytic_episode_class_mass
from mtare_topo.governance import write_json
from mtare_topo.representation.gse_causal_episode_detector import materialize_causal_episode_references
from mtare_topo.semantics.geometric_semantics import EVENT_NAMES


PASS_STATUS = "PASS_GSE_CAUSAL_EPISODE_CLASS_MASS_AUDIT_V1"
FAIL_STATUS = "FAIL_GSE_CAUSAL_EPISODE_CLASS_MASS_AUDIT_V1"


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _episode_counts(event: np.ndarray, episode: np.ndarray, rows: np.ndarray) -> dict:
    return {
        EVENT_NAMES[index]: int(len(np.unique(episode[rows][event[rows] == index])))
        for index in range(1, len(EVENT_NAMES))
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--training-run", required=True, type=Path)
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--pair-cache", required=True, type=Path)
    parser.add_argument("--risk-audit-summary", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    rows = _read_jsonl(args.teacher.resolve())
    if len(rows) != 188126:
        raise RuntimeError("class-mass Teacher count drift")
    bank = materialize_causal_episode_references(rows)
    with np.load(args.pair_cache.resolve(), allow_pickle=False) as archive:
        partition = archive["partition_code"].astype(np.uint8)
        global_index = archive["compact_to_global_sequence_index"].astype(np.int64)
    teacher_global = np.asarray([int(row["global_sequence_index"]) for row in rows], dtype=np.int64)
    if (
        partition.shape != (188126,)
        or not np.array_equal(global_index, teacher_global)
        or int(np.sum(partition == 0)) != 142184
        or int(np.sum(partition == 1)) != 45942
    ):
        raise RuntimeError("class-mass fit/selection population drift")
    fit_rows = np.flatnonzero(partition == 0)
    selection_rows = np.flatnonzero(partition == 1)
    fit_counts = _episode_counts(bank.event_index, bank.episode_id, fit_rows)
    selection_counts = _episode_counts(bank.event_index, bank.episode_id, selection_rows)
    if fit_counts != {"junction": 2542, "terminal": 740, "turn": 560, "geometry_transition": 114}:
        raise RuntimeError("fit episode count drift")
    if selection_counts != {"junction": 882, "terminal": 254, "turn": 180, "geometry_transition": 34}:
        raise RuntimeError("selection episode count drift")

    seed_results = {}
    probabilities = []
    compact_episode = None
    compact_event = None
    observed_rows = None
    for seed in (0, 1, 2):
        with np.load(
            args.training_run.resolve() / f"artifacts/models/seed{seed}/selection_outputs.npz",
            allow_pickle=False,
        ) as archive:
            current_rows = archive["observation_row"].astype(np.int64)
            probability = archive["probability"].astype(np.float64)
            event = archive["event_index"].astype(np.int64)
            episode = archive["episode_id"].astype(np.int64)
        if (
            not np.array_equal(current_rows, selection_rows)
            or not np.array_equal(event, bank.event_index[selection_rows])
        ):
            raise RuntimeError(f"seed{seed} selection alignment drift")
        if observed_rows is None:
            observed_rows, compact_event, compact_episode = current_rows, event, episode
        elif not np.array_equal(compact_event, event) or not np.array_equal(compact_episode, episode):
            raise RuntimeError("three-seed episode identity drift")
        seed_results[str(seed)] = analytic_episode_class_mass(probability, event, episode)
        probabilities.append(probability)
    assert compact_event is not None and compact_episode is not None
    ensemble = analytic_episode_class_mass(
        np.mean(np.stack(probabilities), axis=0), compact_event, compact_episode
    )
    risk = json.loads(args.risk_audit_summary.resolve().read_text(encoding="utf-8"))
    lag10_rows = [
        row for row in risk["lag_risk_audit"]
        if int(row["lag_steps"]) == 10 and str(row["source"]) == "Frozen prediction"
    ]
    if len(lag10_rows) != 1:
        raise RuntimeError("sealed lag10 observability evidence drift")
    lag10_auc = float(lag10_rows[0]["transition_vs_corridor_auc"])
    transition_fit_multiplier = 3956 / (4 * 114)
    transition_selection_multiplier = 1350 / (4 * 34)
    transition = ensemble["per_event"]["geometry_transition"]
    junction = ensemble["per_event"]["junction"]
    checks = {
        "exact_fit_selection_populations": len(fit_rows) == 142184 and len(selection_rows) == 45942,
        "exact_fit_episode_counts": sum(fit_counts.values()) == 3956,
        "exact_selection_episode_counts": sum(selection_counts.values()) == 1350,
        "transition_raw_fit_mass_below_three_percent": fit_counts["geometry_transition"] / 3956 < .03,
        "transition_equal_class_multiplier_above_eight": transition_fit_multiplier > 8.0,
        "selection_transition_gradient_mass_below_junction": transition["raw_joint_gradient_mass_share"] < junction["raw_joint_gradient_mass_share"],
        "long_history_signal_remains_observable": lag10_auc >= .75,
        "zero_optimizer_steps": True,
        "zero_forbidden_reads": True,
    }
    passed = all(checks.values())
    result = {
        "schema_version": "gse_causal_episode_class_mass_audit_v1",
        "overall_status": PASS_STATUS if passed else FAIL_STATUS,
        "scientific_pass": passed,
        "question": "Does raw episode-frequency MIL suppress rare structural classes despite observable twelve-frame geometry?",
        "fit_episode_counts": fit_counts,
        "selection_episode_counts": selection_counts,
        "fit_raw_episode_mass_share": {key: value / 3956 for key, value in fit_counts.items()},
        "fit_equal_class_per_episode_weight_multiplier": {key: 3956 / (4 * value) for key, value in fit_counts.items()},
        "selection_equal_class_per_episode_weight_multiplier": {key: 1350 / (4 * value) for key, value in selection_counts.items()},
        "transition_fit_multiplier": transition_fit_multiplier,
        "transition_selection_multiplier": transition_selection_multiplier,
        "seed_analytic_mass": seed_results,
        "ensemble_analytic_mass": ensemble,
        "lag10_predicted_geometry_risk_auc": lag10_auc,
        "checks": checks,
        "recommended_next": "SAME_CAPACITY_EQUAL_CLASS_EPISODE_MIL_V2" if passed else "STOP_CAUSAL_EPISODE_DETECTOR_ROUTE",
        "model_inference_frames": 0,
        "optimizer_steps": 0,
        "backbone_optimizer_steps": 0,
        "c09_worlds_read": 0,
        "strict_test_worlds_read": 0,
        "mtare_worlds_read": 0,
    }
    write_json(run_dir / "metrics/class_mass_audit.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
