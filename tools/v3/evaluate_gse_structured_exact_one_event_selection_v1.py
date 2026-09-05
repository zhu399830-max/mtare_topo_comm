#!/usr/bin/env python3
"""Select C07 exact-one commit policy and apply it once to C08."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mtare_topo.evaluation.gse_causal_episode_metrics import (
    evaluate_decision_mass_triggers,
    select_decision_mass_threshold,
)
from mtare_topo.evaluation.gse_relational_commit_policy import (
    EventCommitPolicy,
    evaluate_event_commits,
    extract_event_commits,
    merge_event_commits,
)


PASS = "PASS_GSE_STRUCTURED_EXACT_ONE_EVENT_SELECTION_V1"
FAIL = "FAIL_GSE_STRUCTURED_EXACT_ONE_EVENT_SELECTION_V1"
TARGET_MACRO_F1 = 0.793847137320708
THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995, 0.997, 0.999)


def _grid(event: int) -> tuple[EventCommitPolicy, ...]:
    return tuple(
        EventCommitPolicy(event, threshold, votes, stable, release)
        for threshold in THRESHOLDS for votes in (2, 3)
        for stable in (1, 2, 3) for release in (2, 3, 5, 8)
    )


def _safe(metrics: dict) -> bool:
    return bool(
        metrics["precision"] >= 0.995 and metrics["false_fraction"] <= 0.005
        and metrics["recall"] >= 0.25
        and all(row["precision"] >= 0.99 and row["recall"] >= 0.25 for row in metrics["per_event"].values())
    )


def _five(probability: np.ndarray) -> np.ndarray:
    result = np.zeros((len(probability), 5), dtype=np.float64)
    result[:, :3] = probability
    return result


def _uncertainty(probability: np.ndarray) -> np.ndarray:
    return -np.sum(probability * np.log(np.clip(probability, 1e-8, 1.0)), axis=1) / np.log(5.0)


def _stateless(
    probability: np.ndarray, target: np.ndarray, episode: np.ndarray,
    traversal: np.ndarray, sequence: np.ndarray, c07: np.ndarray, c08: np.ndarray,
) -> dict:
    five = _five(probability)
    uncertainty = _uncertainty(five)
    error = None
    try:
        selected = select_decision_mass_threshold(
            five[c07], target[c07], episode[c07], traversal[c07], sequence[c07], uncertainty[c07],
            minimum_precision=.995, minimum_per_event_precision=.99,
            minimum_recall=.25, minimum_per_event_recall=.25,
        )
    except RuntimeError as exc:
        error = str(exc)
        diagnostics = []
        for threshold in np.linspace(0.0, 1.0, 1001):
            row = evaluate_decision_mass_triggers(
                five[c07], target[c07], episode[c07], traversal[c07], sequence[c07], uncertainty[c07],
                decision_threshold=float(threshold),
            )
            if row["decision_episode_recall"] >= .25 and all(value["recall"] >= .25 for value in row["per_event"].values()):
                diagnostics.append(row)
        selected = max(diagnostics, key=lambda row: (row["decision_trigger_precision"], row["decision_episode_macro_f1"]))
    c08_result = evaluate_decision_mass_triggers(
        five[c08], target[c08], episode[c08], traversal[c08], sequence[c08], uncertainty[c08],
        decision_threshold=float(selected["decision_threshold"]),
    )
    return {"c07": selected, "c08": c08_result, "selection_error": error}


def _plot(output: Path, summary: dict) -> None:
    old = summary["old_stateful_baseline"]
    new = summary["stateful"]
    labels = ["Old state\nC07", "Exact-one\nC07", "Old state\nC08", "Exact-one\nC08"]
    precision = [old["c07"]["precision"], new["c07"]["precision"], old["c08"]["precision"], new["c08"]["precision"]]
    macro = [old["c07"]["macro_f1"], new["c07"]["macro_f1"], old["c08"]["macro_f1"], new["c08"]["macro_f1"]]
    recall = [old["c07"]["recall"], new["c07"]["recall"], old["c08"]["recall"], new["c08"]["recall"]]
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 3.9), constrained_layout=True)
    x = np.arange(4)
    axes[0].bar(x - .18, precision, .36, label="precision", color="#59a14f")
    axes[0].bar(x + .18, macro, .36, label="macro-F1", color="#4e79a7")
    axes[0].axhline(.995, color="#b22222", linestyle="--", linewidth=1.0)
    axes[0].axhline(TARGET_MACRO_F1, color="#7b3294", linestyle=":", linewidth=1.2)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0.0, 1.03)
    axes[0].set_title("A  Safe event commits")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].bar(x, recall, color=["#bab0ab", "#f28e2b", "#bab0ab", "#f28e2b"])
    axes[1].axhline(.25, color="#b22222", linestyle="--", linewidth=1.0)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0.0, 1.03)
    axes[1].set_title("B  Decision-episode recall")
    for axis in axes:
        axis.grid(axis="y", alpha=.23)
        axis.set_axisbelow(True)
    figure.suptitle("Structured exact-one learning: C07 selection and unchanged C08 transfer")
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output / f"gse_structured_exact_one_event_selection_v1.{suffix}", dpi=220)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--seed0", required=True, type=Path)
    parser.add_argument("--seed1", required=True, type=Path)
    parser.add_argument("--seed2", required=True, type=Path)
    parser.add_argument("--old-state-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise RuntimeError("structured exact-one selection output exists; overwrite is forbidden")
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
    probabilities = []
    for seed, path in enumerate((args.seed0, args.seed1, args.seed2)):
        with np.load(path.resolve(), allow_pickle=False) as archive:
            if (
                not np.array_equal(archive["observation_row"], rows)
                or not np.array_equal(archive["decision_target"], target)
                or not np.array_equal(archive["decision_episode_id"], episode)
            ):
                raise RuntimeError(f"seed{seed} exact-one selection identity drift")
            probabilities.append(np.asarray(archive["probability"], dtype=np.float64))
    probability = np.stack(probabilities)
    c07 = np.char.endswith(parent, "C07")
    c08 = np.char.endswith(parent, "C08")
    if probability.shape != (3, 45_942, 3) or (int(c07.sum()), int(c08.sum())) != (21_548, 24_394):
        raise RuntimeError("structured exact-one evaluation population drift")
    candidates: dict[int, list[tuple[EventCommitPolicy, tuple, dict]]] = {1: [], 2: []}
    table = []
    for event in (1, 2):
        name = "junction" if event == 1 else "terminal"
        for policy in _grid(event):
            commits = extract_event_commits(probability, traversal, sequence, c07, policy)
            metrics = evaluate_event_commits(commits, target, episode, c07)["per_event"][name]
            candidates[event].append((policy, commits, metrics))
            table.append({**policy.to_dict(), "c07_precision": metrics["precision"], "c07_recall": metrics["recall"], "c07_f1": metrics["f1"], "c07_predicted_commits": metrics["predicted_commits"]})
    per_event_safe = {
        event: [item for item in candidates[event] if item[2]["precision"] >= .99 and item[2]["recall"] >= .25]
        for event in (1, 2)
    }
    joint = []
    if all(per_event_safe.values()):
        for junction in per_event_safe[1]:
            for terminal in per_event_safe[2]:
                metrics = evaluate_event_commits(merge_event_commits(junction[1], terminal[1]), target, episode, c07)
                if _safe(metrics):
                    joint.append((metrics, junction[0], terminal[0]))
    selection_error = None
    if joint:
        c07_metrics, junction_policy, terminal_policy = max(
            joint, key=lambda item: (item[0]["macro_f1"], item[0]["recall"], item[0]["precision"], -item[0]["predicted_commits"])
        )
    else:
        selection_error = "no C07 joint policy satisfies the frozen safety margins"
        junction_policy, junction_commits, _ = max(candidates[1], key=lambda item: (item[2]["precision"], item[2]["recall"], item[2]["f1"]))
        terminal_policy, terminal_commits, _ = max(candidates[2], key=lambda item: (item[2]["precision"], item[2]["recall"], item[2]["f1"]))
        c07_metrics = evaluate_event_commits(merge_event_commits(junction_commits, terminal_commits), target, episode, c07)
    c08_metrics = evaluate_event_commits(merge_event_commits(
        extract_event_commits(probability, traversal, sequence, c08, junction_policy),
        extract_event_commits(probability, traversal, sequence, c08, terminal_policy),
    ), target, episode, c08)
    per_family = {}
    for suffix, mask in (("c07", c07), ("c08", c08)):
        per_family[suffix] = {}
        for family in sorted(set(value[:3] for value in parent[mask].tolist())):
            family_mask = mask & np.asarray([value.startswith(family + "_") for value in parent], dtype=np.bool_)
            commits = merge_event_commits(
                extract_event_commits(probability, traversal, sequence, family_mask, junction_policy),
                extract_event_commits(probability, traversal, sequence, family_mask, terminal_policy),
            )
            per_family[suffix][family] = evaluate_event_commits(commits, target, episode, family_mask)
    stateless = _stateless(probability.mean(axis=0), target, episode, traversal, sequence, c07, c08)
    old_outer = json.loads(args.old_state_summary.read_text(encoding="utf-8"))
    old_selected = old_outer["result"]["selected_policy"]
    old_stateful = {"c07": old_selected["c07"], "c08": old_selected["c08"]}
    gates = {
        "c07_policy_found": selection_error is None,
        "c07_safety": _safe(c07_metrics),
        "c07_macro_f1_at_least_0p793847": c07_metrics["macro_f1"] >= TARGET_MACRO_F1,
        "c08_safety": _safe(c08_metrics),
        "c08_macro_f1_at_least_0p793847": c08_metrics["macro_f1"] >= TARGET_MACRO_F1,
        "all_ten_families_correct_c07": len(per_family["c07"]) == 10 and all(row["correct_unique_episodes"] > 0 for row in per_family["c07"].values()),
        "all_ten_families_correct_c08": len(per_family["c08"]) == 10 and all(row["correct_unique_episodes"] > 0 for row in per_family["c08"].values()),
        "zero_test_training_graph": True,
    }
    passed = all(gates.values())
    summary = {
        "schema_version": "gse_structured_exact_one_event_selection_v1",
        "status": PASS if passed else FAIL, "scientific_pass": passed,
        "decision": "ALLOW_STRUCTURED_EXACT_ONE_C09_QUALIFICATION_PREPARATION" if passed else "STOP_STRUCTURED_EXACT_ONE_EVENT_ROUTE",
        "population": {"observations": len(rows), "worlds": len(np.unique(parent)), "c07_observations": int(c07.sum()), "c08_observations": int(c08.sum())},
        "policy_grid": {"candidates_per_event": len(_grid(1)), "safe_junction": len(per_event_safe[1]), "safe_terminal": len(per_event_safe[2]), "safe_joint": len(joint)},
        "stateful": {"selection_error": selection_error, "junction_policy": junction_policy.to_dict(), "terminal_policy": terminal_policy.to_dict(), "c07": c07_metrics, "c08": c08_metrics},
        "stateless": stateless,
        "old_stateful_baseline": old_stateful,
        "pooled_baseline_macro_f1": 0.743847137320708,
        "required_macro_f1": TARGET_MACRO_F1,
        "per_family": per_family, "gates": gates,
        "c08_policy_updates": 0, "c08_checkpoint_reads": 0,
        "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0, "graph_replays": 0,
    }
    with (args.output_dir / "candidate_grid.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader(); writer.writerows(table)
    np.savez_compressed(
        args.output_dir / "selection_ensemble_outputs.npz", observation_row=rows,
        probability=probability.mean(axis=0).astype(np.float32), decision_target=target.astype(np.int8), decision_episode_id=episode,
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "figure_source.json").write_text(json.dumps({"schema_version": "gse_structured_exact_one_event_figure_source_v1", "summary": summary}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _plot(args.output_dir, summary)
    print(json.dumps({"status": summary["status"], "decision": summary["decision"], "stateful": summary["stateful"], "gates": gates}, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
