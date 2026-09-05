#!/usr/bin/env python3
"""Evaluate the frozen three-seed event-center offset ensemble on C07--C08."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    for seed in range(3):
        parser.add_argument(f"--seed{seed}", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    predictions = []
    reference = None
    seed_summaries = {}
    for seed in range(3):
        path = getattr(args, f"seed{seed}").resolve()
        with np.load(path / "selection_outputs.npz", allow_pickle=False) as archive:
            current = {key: archive[key] for key in archive.files}
        if reference is None:
            reference = current
        elif not all(np.array_equal(current[key], reference[key]) for key in ("observation_row", "global_sequence_index", "target_offset_m", "event_index")):
            raise RuntimeError("event-center seed selection identity drift")
        predictions.append(current["predicted_offset_m"].astype(np.float64))
        seed_summaries[str(seed)] = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    assert reference is not None
    ensemble = np.mean(np.stack(predictions), axis=0)
    target = reference["target_offset_m"].astype(np.float64)
    event = reference["event_index"].astype(np.int64)
    baseline = float(seed_summaries["0"]["selection_baseline_mae_m"])
    if any(abs(float(value["selection_baseline_mae_m"]) - baseline) > 1e-12 for value in seed_summaries.values()):
        raise RuntimeError("event-center baseline differs across seeds")
    mae = float(np.mean(np.abs(ensemble - target)))
    per_event = {}
    for index, name in ((1, "junction"), (2, "terminal")):
        mask = event == index
        per_event[name] = {"rows": int(np.sum(mask)), "mae_m": float(np.mean(np.abs(ensemble[mask] - target[mask])))}
    gates = {
        "aggregate_mae_improves_baseline_by_0p10": 1.0 - mae / baseline >= .10,
        "junction_mae_below_baseline": per_event["junction"]["mae_m"] < baseline,
        "terminal_mae_below_baseline": per_event["terminal"]["mae_m"] < baseline,
        "all_three_seeds_improve_baseline": all(float(value["selection_mae_m"]) < baseline for value in seed_summaries.values()),
    }
    passed = all(gates.values())
    summary = {
        "schema_version": "gse_event_center_offset_ensemble_v1",
        "status": "PASS_GSE_EVENT_CENTER_OFFSET_ENSEMBLE_V1" if passed else "FAIL_GSE_EVENT_CENTER_OFFSET_ENSEMBLE_V1",
        "selection_rows": len(target), "selection_mae_m": mae,
        "selection_baseline_mae_m": baseline, "relative_mae_improvement": 1.0 - mae / baseline,
        "per_event": per_event, "seeds": seed_summaries, "gates": gates,
        "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0,
    }
    np.savez_compressed(
        args.output_dir / "ensemble_selection_outputs.npz",
        observation_row=reference["observation_row"],
        global_sequence_index=reference["global_sequence_index"],
        predicted_offset_m=ensemble.astype(np.float32), target_offset_m=target.astype(np.float32),
        event_index=event.astype(np.int8),
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
