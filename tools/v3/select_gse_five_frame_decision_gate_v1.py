#!/usr/bin/env python3
"""Select a junction/terminal decision-mass gate on C07-C08 only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mtare_topo.evaluation.gse_causal_episode_metrics import (
    extract_decision_mass_triggers,
    select_decision_mass_threshold,
)
from mtare_topo.representation.gse_causal_episode_detector import (
    materialize_causal_episode_references,
)
from mtare_topo.semantics.geometric_semantics import EVENT_NAMES


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _identity_coverage(probability, rows, bank, threshold):
    traversal = np.asarray([str(row["traversal_id"]) for row in rows])
    sequence = np.asarray([int(row["sequence_index"]) for row in rows], dtype=np.int64)
    triggers = extract_decision_mass_triggers(
        probability, traversal, sequence, np.zeros(len(rows)), decision_threshold=threshold
    )
    result = {}
    for name in ("junction", "terminal"):
        event = EVENT_NAMES.index(name)
        truth = {str(row["identity"]) for row in rows if row["event"] == name}
        correct = set()
        seen = set()
        for trigger in triggers:
            episode = int(bank.episode_id[trigger.row])
            if episode < 0 or episode in seen:
                continue
            seen.add(episode)
            if trigger.predicted_event_index == int(bank.event_index[trigger.row]) == event:
                correct.add(str(rows[trigger.row]["identity"]))
        result[name] = {
            "correct_identities": len(correct), "true_identities": len(truth),
            "coverage": len(correct) / len(truth),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--pair-cache", required=True, type=Path)
    for seed in range(3):
        parser.add_argument(f"--observation{seed}", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise RuntimeError("five-frame decision selection output already exists")
    output.mkdir(parents=True)
    rows = _read_jsonl(args.teacher.resolve())
    if len(rows) != 188_126 or any(str(row["parent_id"]).endswith(("_C09", "_C10")) for row in rows):
        raise RuntimeError("decision selector Teacher population drift")
    with np.load(args.pair_cache.resolve(), allow_pickle=False) as archive:
        global_index = archive["compact_to_global_sequence_index"].astype(np.int64)
        parent = archive["parent_id"].astype(str)
        partition = archive["partition_code"].astype(np.uint8)
    teacher_global = np.asarray([int(row["global_sequence_index"]) for row in rows], dtype=np.int64)
    teacher_parent = np.asarray([str(row["parent_id"]) for row in rows])
    selection_index = np.flatnonzero(partition == 1)
    if (
        not np.array_equal(global_index, teacher_global)
        or not np.array_equal(parent, teacher_parent)
        or len(selection_index) != 45_942
        or set(partition.tolist()) != {0, 1}
    ):
        raise RuntimeError("decision selector population alignment drift")
    selection_rows = [rows[int(index)] for index in selection_index]
    bank = materialize_causal_episode_references(selection_rows)
    seed_probability = []
    for seed in range(3):
        observation = np.load(getattr(args, f"observation{seed}").resolve(), mmap_mode="r")
        if observation.shape != (188_126, 146) or observation.dtype != np.float32:
            raise RuntimeError(f"seed{seed} unified observation drift")
        probability = np.asarray(observation[selection_index, :5], dtype=np.float64)
        if not np.allclose(probability.sum(axis=1), 1.0, rtol=0.0, atol=1e-5):
            raise RuntimeError(f"seed{seed} event probability drift")
        seed_probability.append(probability)
    ensemble = np.mean(np.stack(seed_probability), axis=0)
    traversal = np.asarray([str(row["traversal_id"]) for row in selection_rows])
    sequence = np.asarray([int(row["sequence_index"]) for row in selection_rows], dtype=np.int64)
    metrics = select_decision_mass_threshold(
        ensemble, bank.event_index, bank.episode_id, traversal, sequence,
        np.zeros(len(selection_rows)), minimum_precision=.995,
        minimum_per_event_precision=.99, minimum_recall=.25,
        minimum_per_event_recall=.25,
    )
    identity = _identity_coverage(ensemble, selection_rows, bank, metrics["decision_threshold"])
    true_junction = int(np.sum([
        int(np.unique(bank.event_index[bank.episode_id == episode]).item()) == EVENT_NAMES.index("junction")
        for episode in np.unique(bank.episode_id[bank.episode_id >= 0])
    ]))
    true_terminal = int(np.sum([
        int(np.unique(bank.event_index[bank.episode_id == episode]).item()) == EVENT_NAMES.index("terminal")
        for episode in np.unique(bank.episode_id[bank.episode_id >= 0])
    ]))
    if (
        true_junction != 882 or true_terminal != 254
        or identity["junction"]["true_identities"] != 146
        or identity["terminal"]["true_identities"] != 128
    ):
        raise RuntimeError("decision selector episode/identity count drift")
    result = {
        "schema_version": "gse_five_frame_decision_gate_selection_v1",
        "decision_threshold": metrics["decision_threshold"],
        "selection_metrics": metrics,
        "selection_identity_coverage": identity,
        "selection_worlds": 20, "selection_observations": 45_942,
        "junction_episodes": true_junction, "terminal_episodes": true_terminal,
        "threshold_grid_points": 1001,
        "model_inference_frames": 0, "optimizer_steps": 0, "model_updates": 0,
        "c09_worlds_read": 0, "c10_worlds_read": 0,
        "strict_test_worlds_read": 0, "mtare_worlds_read": 0,
    }
    np.savez_compressed(
        output / "selection_outputs.npz",
        global_sequence_index=global_index[selection_index],
        probability=ensemble.astype(np.float32),
        event_index=bank.event_index, episode_id=bank.episode_id,
    )
    (output / "calibration.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
