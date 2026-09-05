#!/usr/bin/env python3
"""Audit whether a fixed past-only commit state can rescue frozen outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mtare_topo.evaluation.gse_causal_episode_metrics import extract_decision_mass_triggers
from mtare_topo.evaluation.gse_relational_commit_policy import (
    EventCommitPolicy,
    evaluate_event_commits,
    extract_event_commits,
    merge_event_commits,
)


PASS = "PASS_GSE_RELATIONAL_COMMIT_POLICY_FEASIBILITY_V1"
FAIL = "FAIL_GSE_RELATIONAL_COMMIT_POLICY_FEASIBILITY_V1"
TARGET_MACRO_F1 = 0.793847137320708
THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995, 0.997, 0.999)


def _policy_grid(event: int) -> tuple[EventCommitPolicy, ...]:
    return tuple(
        EventCommitPolicy(event, threshold, votes, stable, release)
        for threshold in THRESHOLDS
        for votes in (2, 3)
        for stable in (1, 2, 3)
        for release in (2, 3, 5, 8)
    )


def _passes_safety(metrics: dict) -> bool:
    return bool(
        metrics["precision"] >= 0.995
        and metrics["false_fraction"] <= 0.005
        and metrics["recall"] >= 0.25
        and all(row["precision"] >= 0.99 and row["recall"] >= 0.25 for row in metrics["per_event"].values())
    )


def _current_failure_decomposition(
    probability: np.ndarray,
    target: np.ndarray,
    episode: np.ndarray,
    traversal: np.ndarray,
    sequence: np.ndarray,
) -> dict:
    five = np.zeros((len(target), 5), dtype=np.float64)
    five[:, :3] = probability
    triggers = extract_decision_mass_triggers(
        five, traversal, sequence, np.zeros(len(target)), decision_threshold=0.985
    )
    true_episode = {
        int(value): int(np.unique(target[episode == value]).item())
        for value in np.unique(episode[episode >= 0])
    }
    matched: set[int] = set()
    result = {"correct": 0, "duplicate_same_episode": 0, "corridor_false": 0, "wrong_event": 0}
    for trigger in triggers:
        identity = int(episode[trigger.row])
        if identity not in true_episode:
            result["corridor_false"] += 1
        elif identity in matched:
            result["duplicate_same_episode"] += 1
        else:
            matched.add(identity)
            if trigger.predicted_event_index == true_episode[identity]:
                result["correct"] += 1
            else:
                result["wrong_event"] += 1
    result["total"] = len(triggers)
    return result


def _plot(output: Path, summary: dict) -> None:
    selected = summary["selected_policy"]
    stateless = summary["stateless_ensemble"]
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 3.9), constrained_layout=True)
    labels = ["Stateless\nC07+C08", "Stateful\nC07", "Stateful\nC08"]
    precision = [stateless["precision"], selected["c07"]["precision"], selected["c08"]["precision"]]
    macro = [stateless["macro_f1"], selected["c07"]["macro_f1"], selected["c08"]["macro_f1"]]
    x = np.arange(3)
    axes[0].bar(x - 0.18, precision, 0.36, label="precision", color="#59a14f")
    axes[0].bar(x + 0.18, macro, 0.36, label="macro-F1", color="#4e79a7")
    axes[0].axhline(0.995, color="#b22222", linestyle="--", linewidth=1.0, label="precision gate")
    axes[0].axhline(TARGET_MACRO_F1, color="#7b3294", linestyle=":", linewidth=1.2, label="F1 gate")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0.0, 1.03)
    axes[0].set_title("A  Commit-policy transfer")
    axes[0].legend(frameon=False, fontsize=7)
    failure = summary["stateless_failure_decomposition"]
    names = ["Duplicate", "Corridor false", "Wrong type"]
    values = [failure["duplicate_same_episode"], failure["corridor_false"], failure["wrong_event"]]
    axes[1].bar(names, values, color=["#f28e2b", "#e15759", "#b07aa1"])
    axes[1].set_ylabel("Non-correct triggers")
    axes[1].set_title("B  Stateless failure attribution")
    for index, value in enumerate(values):
        axes[1].text(index, value + 0.25, str(value), ha="center", fontsize=9)
    for axis in axes:
        axis.grid(axis="y", alpha=0.22)
        axis.set_axisbelow(True)
    figure.suptitle("Frozen relational outputs: causal commit state is helpful but insufficient")
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output / f"gse_relational_commit_policy_feasibility_v1.{suffix}", dpi=220)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--seed0", required=True, type=Path)
    parser.add_argument("--seed1", required=True, type=Path)
    parser.add_argument("--seed2", required=True, type=Path)
    parser.add_argument("--training-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise RuntimeError("commit feasibility output exists; overwrite is forbidden")
    args.output_dir.mkdir(parents=True)
    partition = np.load(args.cache_dir / "partition_code.npy")
    rows = np.flatnonzero(partition == 1)
    target = np.load(args.cache_dir / "decision_target.npy")[rows].astype(np.int64)
    source_episode = np.load(args.cache_dir / "decision_episode_id.npy")[rows].astype(np.int64)
    unique_episode = np.unique(source_episode[source_episode >= 0])
    remap = {int(value): index for index, value in enumerate(unique_episode.tolist())}
    episode = np.asarray([remap.get(int(value), -1) for value in source_episode], dtype=np.int64)
    traversal = np.load(args.cache_dir / "traversal_id.npy")[rows].astype(str)
    sequence = np.load(args.cache_dir / "sequence_index.npy")[rows].astype(np.int64)
    parent = np.load(args.cache_dir / "parent_id.npy")[rows].astype(str)
    seed_probability = []
    for seed, path in enumerate((args.seed0, args.seed1, args.seed2)):
        with np.load(path.resolve(), allow_pickle=False) as archive:
            if (
                not np.array_equal(archive["observation_row"], rows)
                or not np.array_equal(archive["decision_target"], target)
                or not np.array_equal(archive["decision_episode_id"], episode)
            ):
                raise RuntimeError(f"seed{seed} commit-feasibility identity drift")
            seed_probability.append(np.asarray(archive["probability"], dtype=np.float64))
    probability = np.stack(seed_probability)
    if probability.shape != (3, 45_942, 3):
        raise RuntimeError("commit-feasibility population drift")
    c07 = np.char.endswith(parent, "C07")
    c08 = np.char.endswith(parent, "C08")
    if int(c07.sum()) != 21_548 or int(c08.sum()) != 24_394:
        raise RuntimeError("C07/C08 world-transfer population drift")

    candidates: dict[int, list[tuple[EventCommitPolicy, tuple, dict]]] = {1: [], 2: []}
    csv_rows = []
    for event in (1, 2):
        for policy in _policy_grid(event):
            commits = extract_event_commits(probability, traversal, sequence, c07, policy)
            metrics = evaluate_event_commits(commits, target, episode, c07)
            row = metrics["per_event"]["junction" if event == 1 else "terminal"]
            candidates[event].append((policy, commits, row))
            csv_rows.append({
                **policy.to_dict(),
                "c07_precision": row["precision"], "c07_recall": row["recall"],
                "c07_f1": row["f1"], "c07_predicted_commits": row["predicted_commits"],
            })
    safe = {
        event: [item for item in candidates[event] if item[2]["precision"] >= 0.99 and item[2]["recall"] >= 0.25]
        for event in (1, 2)
    }
    if not all(safe.values()):
        raise RuntimeError("no per-event commit policy satisfies even the frozen class safety margins")
    joint = []
    for junction in safe[1]:
        for terminal in safe[2]:
            commits = merge_event_commits(junction[1], terminal[1])
            metrics = evaluate_event_commits(commits, target, episode, c07)
            if _passes_safety(metrics):
                joint.append((metrics, junction[0], terminal[0]))
    if not joint:
        raise RuntimeError("no joint C07 commit policy satisfies frozen safety margins")
    selected_c07, junction_policy, terminal_policy = max(
        joint,
        key=lambda item: (
            item[0]["macro_f1"], item[0]["recall"], item[0]["precision"],
            -item[0]["predicted_commits"], tuple(-value for value in (
                item[1].probability_threshold, item[1].consensus_votes, item[1].stable_frames, item[1].release_frames,
                item[2].probability_threshold, item[2].consensus_votes, item[2].stable_frames, item[2].release_frames,
            )),
        ),
    )
    c08_commits = merge_event_commits(
        extract_event_commits(probability, traversal, sequence, c08, junction_policy),
        extract_event_commits(probability, traversal, sequence, c08, terminal_policy),
    )
    selected_c08 = evaluate_event_commits(c08_commits, target, episode, c08)
    training = json.loads(args.training_summary.read_text(encoding="utf-8"))
    stateless_metrics = training["selection"]["ensemble"]
    stateless = {
        "precision": stateless_metrics["decision_trigger_precision"],
        "recall": stateless_metrics["decision_episode_recall"],
        "macro_f1": stateless_metrics["decision_episode_macro_f1"],
        "threshold": stateless_metrics["decision_threshold"],
    }
    decomposition = _current_failure_decomposition(
        probability.mean(axis=0), target, episode, traversal, sequence
    )
    gates = {
        "c07_safety": _passes_safety(selected_c07),
        "c07_macro_f1_gain_at_least_0p05": selected_c07["macro_f1"] >= TARGET_MACRO_F1,
        "c08_safety": _passes_safety(selected_c08),
        "c08_macro_f1_gain_at_least_0p05": selected_c08["macro_f1"] >= TARGET_MACRO_F1,
        "stateless_decomposition_exact": decomposition == {
            "correct": 461, "duplicate_same_episode": 18,
            "corridor_false": 5, "wrong_event": 3, "total": 487,
        },
        "zero_training_inference_test_graph": True,
    }
    passed = all(gates.values())
    summary = {
        "schema_version": "gse_relational_commit_policy_feasibility_v1",
        "status": PASS if passed else FAIL,
        "scientific_pass": passed,
        "decision": "ALLOW_FROZEN_COMMIT_POLICY_FORMALIZATION" if passed else "STATE_MACHINE_INSUFFICIENT_REQUIRE_STRUCTURED_ONE_COMMIT_LEARNING_OR_STOP",
        "population": {
            "selection_observations": len(rows), "selection_worlds": len(np.unique(parent)),
            "c07_observations": int(c07.sum()), "c08_observations": int(c08.sum()),
            "c07_decision_episodes": selected_c07["true_decision_episodes"],
            "c08_decision_episodes": selected_c08["true_decision_episodes"],
        },
        "grid": {
            "thresholds": list(THRESHOLDS), "consensus_votes": [2, 3],
            "stable_frames": [1, 2, 3], "release_frames": [2, 3, 5, 8],
            "candidates_per_event": len(_policy_grid(1)),
            "safe_c07_candidates": {"junction": len(safe[1]), "terminal": len(safe[2]), "joint": len(joint)},
        },
        "selected_policy": {
            "selection_rule": "C07 safety first, then maximum macro-F1/recall/precision; one immutable application to C08",
            "junction": junction_policy.to_dict(), "terminal": terminal_policy.to_dict(),
            "c07": selected_c07, "c08": selected_c08,
        },
        "stateless_ensemble": stateless,
        "stateless_failure_decomposition": decomposition,
        "pooled_baseline_macro_f1": 0.743847137320708,
        "required_macro_f1": TARGET_MACRO_F1,
        "gates": gates,
        "optimizer_steps": 0, "model_inference_frames": 0, "threshold_training_rows": 0,
        "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0, "graph_replays": 0,
    }
    with (args.output_dir / "candidate_grid.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "figure_source.json").write_text(json.dumps({"schema_version": "gse_relational_commit_policy_figure_source_v1", "summary": summary}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _plot(args.output_dir, summary)
    print(json.dumps({"status": summary["status"], "decision": summary["decision"], "selected_policy": summary["selected_policy"], "gates": gates}, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
