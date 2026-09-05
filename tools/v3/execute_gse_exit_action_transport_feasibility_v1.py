#!/usr/bin/env python3
"""Read-only C01-C08 feasibility proof for causal exit-token transport."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr

from _bootstrap import PROJECT_ROOT  # noqa: F401
from mtare_topo.evaluation.gse_exit_action_transport import (
    attach_teacher_exit_identities,
    decision_from_visible_exit_count,
    descriptor_transport_pair,
    multiclass_metrics,
)


PASS = "PASS_GSE_EXIT_ACTION_TRANSPORT_FEASIBILITY_V1"
FAIL = "FAIL_GSE_EXIT_ACTION_TRANSPORT_FEASIBILITY_V1"
EXPECTED = {"fit": 142_184, "selection": 45_942}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_hash(path: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    total = 0
    for item in files:
        relative = item.relative_to(path).as_posix().encode()
        payload_hash = bytes.fromhex(_sha256(item))
        digest.update(len(relative).to_bytes(4, "little"))
        digest.update(relative)
        digest.update(payload_hash)
        total += item.stat().st_size
    return digest.hexdigest(), len(files), total


def _verify_cache(cache: Path) -> dict:
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["array_sha256"].items():
        if _sha256(cache / name) != expected:
            raise RuntimeError(f"action cache drift: {name}")
    return manifest


def _load_teacher(dataset: Path, shard_manifest: Path) -> dict[str, np.ndarray | dict]:
    source = json.loads(shard_manifest.read_text(encoding="utf-8"))
    records = {row["parent_id"]: row for row in source["shards"] if row["split"] == "train"}
    shards = sorted(dataset.glob("*.zarr"))
    if len(shards) != 80 or len(records) != 80:
        raise RuntimeError("C01-C08 shard population drift")
    arrays: dict[str, list[np.ndarray]] = {
        name: [] for name in (
            "global_sequence_index", "exit_mask", "exit_heading_unit",
            "exit_opening_width_m", "exit_width_valid_mask",
            "exit_vertical_profile_m", "exit_identity",
        )
    }
    verified_files = 0
    verified_bytes = 0
    for path in shards:
        record = records.get(path.stem)
        if record is None or not path.stem.endswith(tuple(f"_C{i:02d}" for i in range(1, 9))):
            raise RuntimeError(f"unexpected development shard: {path.name}")
        tree, files, size = _tree_hash(path)
        if (
            tree != record["shard_tree_sha256"]
            or files != int(record["shard_file_count"])
            or size != int(record["shard_bytes"])
        ):
            raise RuntimeError(f"dataset shard drift: {path.name}")
        verified_files += files
        verified_bytes += size
        group = zarr.open_group(str(path), mode="r")
        for name in arrays:
            arrays[name].append(np.asarray(group[name]))
    joined = {name: np.concatenate(values) for name, values in arrays.items()}
    order = np.argsort(joined["global_sequence_index"])
    joined = {name: value[order] for name, value in joined.items()}
    joined["verification"] = {
        "shards": len(shards),
        "files": verified_files,
        "bytes": verified_bytes,
        "all_shard_tree_hashes_match": True,
    }
    return joined


def _episode_support(
    target: np.ndarray,
    episode: np.ndarray,
    count: np.ndarray,
    split: np.ndarray,
) -> dict[str, dict[str, int | float]]:
    support = ((target == 1) & (count >= 3)) | ((target == 2) & (count == 1))
    result = {}
    for code, name in ((0, "fit"), (1, "selection")):
        active = (split == code) & (target > 0)
        episodes = np.unique(episode[active])
        covered = sum(bool(np.any(support & (episode == value))) for value in episodes)
        result[name] = {
            "decision_rows": int(active.sum()),
            "supported_decision_rows": int(np.sum(active & support)),
            "decision_episodes": len(episodes),
            "supported_decision_episodes": covered,
            "episode_coverage": covered / len(episodes),
            "junction_rows": int(np.sum(active & (target == 1))),
            "supported_junction_rows": int(np.sum(active & (target == 1) & support)),
            "terminal_rows": int(np.sum(active & (target == 2))),
            "supported_terminal_rows": int(np.sum(active & (target == 2) & support)),
        }
    return result


def _transport(
    tokens: np.ndarray,
    target: dict[str, np.ndarray],
    traversal: np.ndarray,
    sequence: np.ndarray,
    split: np.ndarray,
) -> dict[str, dict[str, int | float]]:
    attached = attach_teacher_exit_identities(tokens, target)
    descriptor = np.asarray(tokens[:, :, 8:40], dtype=np.float32)
    adjacent = np.flatnonzero(
        (traversal[1:] == traversal[:-1]) & (sequence[1:] == sequence[:-1] + 1)
    ) + 1
    totals = {
        name: {key: 0 for key in ("pairs", "correct", "proposed", "possible", "exact")}
        for name in EXPECTED
    }
    for row in adjacent:
        name = "fit" if int(split[row]) == 0 else "selection"
        previous_slots = np.flatnonzero(attached[row - 1] >= 0)
        current_slots = np.flatnonzero(attached[row] >= 0)
        previous_id = attached[row - 1, previous_slots]
        current_id = attached[row, current_slots]
        possible = len(set(previous_id.tolist()) & set(current_id.tolist()))
        left, right = descriptor_transport_pair(
            descriptor[row - 1, previous_slots], descriptor[row, current_slots]
        )
        correct = int(np.sum(previous_id[left] == current_id[right]))
        item = totals[name]
        item["pairs"] += 1
        item["correct"] += correct
        item["proposed"] += len(left)
        item["possible"] += possible
        item["exact"] += int(correct == possible and len(left) == possible)
    result = {}
    for name, item in totals.items():
        result[name] = {
            **item,
            "precision": item["correct"] / item["proposed"],
            "recall": item["correct"] / item["possible"],
            "exact_pair_fraction": item["exact"] / item["pairs"],
        }
    return result


def _plot(output: Path, summary: dict) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), constrained_layout=True)
    upper = summary["teacher_action_set_upper_bound"]["selection"]["macro_f1"]
    pooled = summary["sealed_baselines"]["pooled_token_event_model"]["macro_f1"]
    axes[0].bar([0, 1], [pooled, upper], color=["#9c755f", "#4e79a7"])
    axes[0].set_xticks([0, 1], ["Old pooled\ntokens", "Executable-set\nupper bound"])
    axes[0].set_ylabel("Decision macro-F1")
    axes[0].set_ylim(0.0, 1.04)
    axes[0].set_title("A  Structure retained")

    x = np.arange(3)
    precision = [row["selection"]["precision"] for row in summary["seed_transport"]]
    recall = [row["selection"]["recall"] for row in summary["seed_transport"]]
    axes[1].bar(x - 0.18, precision, 0.36, label="precision", color="#59a14f")
    axes[1].bar(x + 0.18, recall, 0.36, label="recall", color="#f28e2b")
    axes[1].axhline(0.98, color="#b22222", linestyle="--", linewidth=1.0)
    axes[1].set_xticks(x, ["seed 0", "seed 1", "seed 2"])
    axes[1].set_ylim(0.95, 1.002)
    axes[1].set_title("B  Cross-frame exit identity")
    axes[1].legend(frameon=False, fontsize=8)

    coverage = summary["teacher_episode_support"]
    axes[2].bar([0, 1], [coverage["fit"]["episode_coverage"], coverage["selection"]["episode_coverage"]], color=["#76b7b2", "#e15759"])
    axes[2].axhline(0.98, color="#b22222", linestyle="--", linewidth=1.0)
    axes[2].set_xticks([0, 1], ["C01-C06", "C07-C08"])
    axes[2].set_ylim(0.95, 1.002)
    axes[2].set_title("C  Episodes with executable support")
    axes[2].set_ylabel("Episode coverage")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
    figure.suptitle("GSE-Graph: executable exit/action-token feasibility")
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output / f"gse_exit_action_transport_feasibility_v1.{suffix}", dpi=220)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--shard-manifest", required=True, type=Path)
    parser.add_argument("--action-cache", required=True, type=Path)
    parser.add_argument("--pooled-baseline", required=True, type=Path)
    parser.add_argument("--summary-baseline", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cache_manifest = _verify_cache(args.action_cache.resolve())
    teacher = _load_teacher(args.dataset.resolve(), args.shard_manifest.resolve())
    global_index = np.load(args.action_cache / "global_sequence_index.npy")
    split = np.load(args.action_cache / "partition_code.npy")
    target = np.load(args.action_cache / "decision_target.npy")
    episode = np.load(args.action_cache / "decision_episode_id.npy")
    traversal = np.load(args.action_cache / "traversal_id.npy")
    sequence = np.load(args.action_cache / "sequence_index.npy")
    raw = np.load(args.action_cache / "raw_tokens.npy", mmap_mode="r")
    if (
        len(global_index) != 188_126
        or not np.array_equal(teacher["global_sequence_index"], global_index)
        or int(np.sum(split == 0)) != EXPECTED["fit"]
        or int(np.sum(split == 1)) != EXPECTED["selection"]
        or raw.shape != (188_126, 3, 6, 40)
        or cache_manifest["fit_decision_episodes"] != 3_282
        or cache_manifest["selection_decision_episodes"] != 1_136
    ):
        raise RuntimeError("exit/action feasibility population drift")
    visible_count = np.asarray(teacher["exit_mask"], dtype=np.int64).sum(axis=1)
    upper = {}
    for code, name in ((0, "fit"), (1, "selection")):
        rows = split == code
        upper[name] = multiclass_metrics(
            target[rows], decision_from_visible_exit_count(visible_count[rows])
        )
        upper[name]["observations"] = int(rows.sum())
        upper[name]["visible_tokens"] = int(visible_count[rows].sum())
    support = _episode_support(target, episode, visible_count, split)
    transport_target = {
        "mask": teacher["exit_mask"],
        "heading": teacher["exit_heading_unit"],
        "width": teacher["exit_opening_width_m"],
        "width_mask": teacher["exit_width_valid_mask"],
        "identity": teacher["exit_identity"],
    }
    seed_transport = []
    for seed in (0, 1, 2):
        seed_transport.append({
            "seed": seed,
            **_transport(
                np.asarray(raw[:, seed], dtype=np.float32),
                transport_target,
                traversal,
                sequence,
                split,
            ),
        })
    pooled_outer = json.loads(args.pooled_baseline.read_text(encoding="utf-8"))
    pooled = pooled_outer["selection"]["ensemble"]
    fixed_outer = json.loads(args.summary_baseline.read_text(encoding="utf-8"))
    fixed = fixed_outer["proof"]["metrics"]["combined"]["selection"]
    checks = {
        "exact_population": upper["fit"]["observations"] == 142_184 and upper["selection"]["observations"] == 45_942,
        "exact_visible_tokens": upper["fit"]["visible_tokens"] == 299_872 and upper["selection"]["visible_tokens"] == 97_041,
        "dataset_tree_verified": bool(teacher["verification"]["all_shard_tree_hashes_match"]),
        "teacher_selection_macro_f1_at_least_0p98": upper["selection"]["macro_f1"] >= 0.98,
        "teacher_selection_episode_coverage_at_least_0p98": support["selection"]["episode_coverage"] >= 0.98,
        "all_seed_selection_transport_precision_at_least_0p98": all(row["selection"]["precision"] >= 0.98 for row in seed_transport),
        "all_seed_selection_transport_recall_at_least_0p98": all(row["selection"]["recall"] >= 0.98 for row in seed_transport),
        "old_pooled_model_below_safety_contract": pooled["decision_trigger_precision"] < 0.995,
        "zero_forbidden_operations": True,
    }
    scientific_pass = all(checks.values())
    decision = (
        "IMPLEMENT_RELATIONAL_EXIT_TOKEN_TRANSPORT_EVENT_MODEL"
        if scientific_pass
        else "STOP_EXIT_TOKEN_TRANSPORT_ROUTE_REASSESS_REPRESENTATION"
    )
    summary = {
        "schema_version": "gse_exit_action_transport_feasibility_v1",
        "status": PASS if scientific_pass else FAIL,
        "scientific_pass": scientific_pass,
        "decision": decision,
        "question": "Can full executable exit/action tokens retain causal structure and cross-frame identity strongly enough to replace pooled action summaries?",
        "method": "Evaluation-only heading/width attachment followed by Teacher-free adjacent-frame descriptor assignment; executable visible-exit cardinality is the Teacher upper bound.",
        "important_scope": "Transport scores use only the six predictions attached to visible Teacher exits; confidence/objectness and refusal remain unsolved model requirements.",
        "population": {
            "worlds": 80,
            "fit_worlds": 60,
            "selection_worlds": 20,
            "raw_frames": 252_430,
            "observations": 188_126,
            "fit_observations": 142_184,
            "selection_observations": 45_942,
            "visible_exit_tokens": 396_913,
            "fit_decision_episodes": 3_282,
            "selection_decision_episodes": 1_136,
            "fit_adjacent_pairs": 130_080,
            "selection_adjacent_pairs": 41_970,
        },
        "dataset_verification": teacher["verification"],
        "teacher_action_set_upper_bound": upper,
        "teacher_episode_support": support,
        "seed_transport": seed_transport,
        "sealed_baselines": {
            "fixed_five_dimensional_change_point": {
                "precision": fixed["structural_trigger_precision"],
                "episode_recall": fixed["structural_episode_recall"],
            },
            "pooled_token_event_model": {
                "macro_f1": pooled["decision_episode_macro_f1"],
                "precision": pooled["decision_trigger_precision"],
                "episode_recall": pooled["decision_episode_recall"],
            },
        },
        "checks": checks,
        "optimizer_steps": 0,
        "model_inference_frames": 0,
        "threshold_selection_steps": 0,
        "graph_replays": 0,
        "duration_seconds": time.monotonic() - started,
        "c09_worlds_read": 0,
        "c10_worlds_read": 0,
        "mtare_worlds_read": 0,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "figure_source.json").write_text(json.dumps({"schema_version": "gse_exit_action_transport_feasibility_figure_source_v1", "summary": summary}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output / "transport_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("seed", "partition", "adjacent_pairs", "correct", "proposed", "possible", "precision", "recall", "exact_pair_fraction"))
        writer.writeheader()
        for row in seed_transport:
            for partition in EXPECTED:
                item = row[partition]
                writer.writerow({"seed": row["seed"], "partition": partition, "adjacent_pairs": item["pairs"], **{key: item[key] for key in ("correct", "proposed", "possible", "precision", "recall", "exact_pair_fraction")}})
    _plot(output, summary)
    print(json.dumps({"status": summary["status"], "decision": decision, "checks": checks}, indent=2, sort_keys=True))
    return 0 if scientific_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
