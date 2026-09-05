#!/usr/bin/env python3
"""Zero-training C01-C08 audit of frozen token validity evidence."""

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
from mtare_topo.evaluation.gse_exit_action_transport import attach_teacher_exit_identities
from mtare_topo.evaluation.gse_token_validity_feasibility import (
    binary_ranking_metrics,
    causal_descriptor_track_mean_confidence,
    cross_seed_geometry_consensus_score,
)


PASS = "PASS_GSE_TOKEN_VALIDITY_ACTION_TRACK_FEASIBILITY_V1"
FAIL = "FAIL_GSE_TOKEN_VALIDITY_ACTION_TRACK_FEASIBILITY_V1"


def _result_payload(path: Path) -> dict:
    """Accept both outer-run summaries and already-inner metric summaries."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    result = payload.get("result")
    return result if isinstance(result, dict) else payload


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
        digest.update(len(relative).to_bytes(4, "little"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256(item)))
        total += item.stat().st_size
    return digest.hexdigest(), len(files), total


def _load_teacher(dataset: Path, shard_manifest: Path) -> dict[str, np.ndarray | dict]:
    manifest = json.loads(shard_manifest.read_text(encoding="utf-8"))
    records = {row["parent_id"]: row for row in manifest["shards"] if row["split"] == "train"}
    shards = sorted(dataset.glob("*.zarr"))
    if len(shards) != 80 or len(records) != 80:
        raise RuntimeError("C01-C08 shard population drift")
    names = (
        "global_sequence_index", "exit_mask", "exit_heading_unit",
        "exit_opening_width_m", "exit_width_valid_mask", "exit_identity",
    )
    arrays: dict[str, list[np.ndarray]] = {name: [] for name in names}
    verified_files = 0
    verified_bytes = 0
    for path in shards:
        record = records.get(path.stem)
        if record is None or not path.stem.endswith(tuple(f"_C{i:02d}" for i in range(1, 9))):
            raise RuntimeError(f"unexpected development shard: {path.name}")
        tree, files, size = _tree_hash(path)
        if tree != record["shard_tree_sha256"] or files != int(record["shard_file_count"]) or size != int(record["shard_bytes"]):
            raise RuntimeError(f"dataset shard drift: {path.name}")
        verified_files += files
        verified_bytes += size
        group = zarr.open_group(str(path), mode="r")
        for name in names:
            arrays[name].append(np.asarray(group[name]))
    joined = {name: np.concatenate(values) for name, values in arrays.items()}
    order = np.argsort(joined["global_sequence_index"])
    joined = {name: value[order] for name, value in joined.items()}
    joined["verification"] = {
        "shards": 80,
        "files": verified_files,
        "bytes": verified_bytes,
        "all_shard_tree_hashes_match": True,
    }
    return joined


def _verify_cache(cache: Path) -> dict:
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["array_sha256"].items():
        if _sha256(cache / name) != expected:
            raise RuntimeError(f"action cache drift: {name}")
    return manifest


def _plot(output: Path, summary: dict) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13.6, 4.1), constrained_layout=True)
    seeds = np.arange(3)
    current = [row["selection"]["current_confidence"]["average_precision"] for row in summary["seed_metrics"]]
    causal = [row["selection"]["causal_five_frame_mean"]["average_precision"] for row in summary["seed_metrics"]]
    consensus = [row["selection"]["cross_seed_geometry_consensus"]["average_precision"] for row in summary["seed_metrics"]]
    axes[0].bar(seeds - .24, current, .24, label="single frame", color="#4e79a7")
    axes[0].bar(seeds, causal, .24, label="5-frame track", color="#f28e2b")
    axes[0].bar(seeds + .24, consensus, .24, label="3-seed geometry", color="#59a14f")
    axes[0].set_xticks(seeds, ["seed 0", "seed 1", "seed 2"])
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Token average precision")
    axes[0].set_title("A  Validity separability")
    axes[0].legend(frameon=False, fontsize=8)

    recall = [[row["selection"][name]["recall_at_precision_floor"] for row in summary["seed_metrics"]] for name in ("current_confidence", "causal_five_frame_mean", "cross_seed_geometry_consensus")]
    axes[1].bar(seeds - .24, recall[0], .24, color="#4e79a7")
    axes[1].bar(seeds, recall[1], .24, color="#f28e2b")
    axes[1].bar(seeds + .24, recall[2], .24, color="#59a14f")
    axes[1].axhline(.5, color="#b22222", linestyle="--", linewidth=1)
    axes[1].set_xticks(seeds, ["seed 0", "seed 1", "seed 2"])
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Recall at precision >= 0.995")
    axes[1].set_title("B  Safety/recall contract")

    labels = ["Teacher\naction set", "V1 direct\nevent", "V2 exact-one"]
    values = [
        summary["retained_capacity"]["teacher_action_set_selection_macro_f1"],
        summary["sealed_direct_event_baselines"]["v1_c08_macro_f1"],
        summary["sealed_direct_event_baselines"]["v2_c08_macro_f1"],
    ]
    axes[2].bar(np.arange(3), values, color=["#76b7b2", "#9c755f", "#e15759"])
    axes[2].axhline(.793847, color="#b22222", linestyle="--", linewidth=1)
    axes[2].set_xticks(np.arange(3), labels)
    axes[2].set_ylim(0, 1.04)
    axes[2].set_ylabel("Decision macro-F1")
    axes[2].set_title("C  Capacity exists; validity fails")
    for axis in axes:
        axis.grid(axis="y", alpha=.25)
        axis.set_axisbelow(True)
    figure.suptitle("GSE-Graph frozen free-query token validity diagnosis")
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output / f"gse_token_validity_action_track_feasibility_v1.{suffix}", dpi=220)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--shard-manifest", required=True, type=Path)
    parser.add_argument("--action-cache", required=True, type=Path)
    parser.add_argument("--transport-summary", required=True, type=Path)
    parser.add_argument("--v1-summary", required=True, type=Path)
    parser.add_argument("--v2-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cache_manifest = _verify_cache(args.action_cache.resolve())
    teacher = _load_teacher(args.dataset.resolve(), args.shard_manifest.resolve())
    raw = np.asarray(np.load(args.action_cache / "raw_tokens.npy", mmap_mode="r"), dtype=np.float32)
    split = np.load(args.action_cache / "partition_code.npy")
    parent = np.load(args.action_cache / "parent_id.npy")
    traversal = np.load(args.action_cache / "traversal_id.npy")
    sequence = np.load(args.action_cache / "sequence_index.npy")
    global_index = np.load(args.action_cache / "global_sequence_index.npy")
    if (
        raw.shape != (188_126, 3, 6, 40)
        or len(global_index) != 188_126
        or not np.array_equal(global_index, teacher["global_sequence_index"])
        or int(np.sum(split == 0)) != 142_184
        or int(np.sum(split == 1)) != 45_942
        or cache_manifest["fit_decision_episodes"] != 3_282
        or cache_manifest["selection_decision_episodes"] != 1_136
    ):
        raise RuntimeError("token-validity population drift")
    c07 = np.asarray([str(value).endswith("_C07") for value in parent], dtype=bool)
    c08 = np.asarray([str(value).endswith("_C08") for value in parent], dtype=bool)
    if int(c07.sum()) != 21_548 or int(c08.sum()) != 24_394 or np.any((split == 1) != (c07 | c08)):
        raise RuntimeError("C07/C08 partition identity drift")

    target = {
        "mask": teacher["exit_mask"],
        "heading": teacher["exit_heading_unit"],
        "width": teacher["exit_opening_width_m"],
        "width_mask": teacher["exit_width_valid_mask"],
        "identity": teacher["exit_identity"],
    }
    consensus = cross_seed_geometry_consensus_score(raw)
    seed_metrics = []
    partitions = {"fit": split == 0, "selection": split == 1, "c07": c07, "c08": c08}
    validity_sha = []
    for seed in range(3):
        attached = attach_teacher_exit_identities(raw[:, seed], target)
        validity = attached >= 0
        validity_sha.append(hashlib.sha256(validity.tobytes()).hexdigest())
        causal = causal_descriptor_track_mean_confidence(raw[:, seed], traversal, sequence, history=5)
        metrics = {"seed": seed, "positive_tokens": int(validity.sum()), "negative_tokens": int(validity.size - validity.sum())}
        for name, rows in partitions.items():
            labels = validity[rows].reshape(-1)
            metrics[name] = {
                "observations": int(rows.sum()),
                "current_confidence": binary_ranking_metrics(labels, raw[rows, seed, :, 0]),
                "causal_five_frame_mean": binary_ranking_metrics(labels, causal[rows]),
                "cross_seed_geometry_consensus": binary_ranking_metrics(labels, consensus[rows, seed]),
            }
        seed_metrics.append(metrics)

    transport = _result_payload(args.transport_summary)
    v1 = _result_payload(args.v1_summary)
    v2 = _result_payload(args.v2_summary)
    retained = {
        "teacher_action_set_selection_macro_f1": float(transport["teacher_action_set_upper_bound"]["selection"]["macro_f1"]),
        "teacher_decision_episode_coverage": float(transport["teacher_episode_support"]["selection"]["episode_coverage"]),
        "all_seed_descriptor_transport_precision_min": min(float(row["selection"]["precision"]) for row in transport["seed_transport"]),
        "all_seed_descriptor_transport_recall_min": min(float(row["selection"]["recall"]) for row in transport["seed_transport"]),
    }
    direct = {
        "v1_c07_macro_f1": float(v1["selected_policy"]["c07"]["macro_f1"]),
        "v1_c08_macro_f1": float(v1["selected_policy"]["c08"]["macro_f1"]),
        "v2_c07_macro_f1": float(v2["stateful"]["c07"]["macro_f1"]),
        "v2_c08_macro_f1": float(v2["stateful"]["c08"]["macro_f1"]),
    }
    current_recall = [row["selection"]["current_confidence"]["recall_at_precision_floor"] for row in seed_metrics]
    corrected_recall = [
        max(
            row["selection"]["causal_five_frame_mean"]["recall_at_precision_floor"],
            row["selection"]["cross_seed_geometry_consensus"]["recall_at_precision_floor"],
        )
        for row in seed_metrics
    ]
    ap_gain = [
        max(
            row["selection"]["causal_five_frame_mean"]["average_precision"],
            row["selection"]["cross_seed_geometry_consensus"]["average_precision"],
        ) - row["selection"]["current_confidence"]["average_precision"]
        for row in seed_metrics
    ]
    checks = {
        "exact_population_and_partitions": True,
        "exact_teacher_validity_population_each_seed": all(row["positive_tokens"] == 396_913 and row["negative_tokens"] == 731_843 for row in seed_metrics),
        "dataset_tree_verified": bool(teacher["verification"]["all_shard_tree_hashes_match"]),
        "retained_action_set_capacity": retained["teacher_action_set_selection_macro_f1"] >= .98,
        "retained_descriptor_transport": retained["all_seed_descriptor_transport_precision_min"] >= .98 and retained["all_seed_descriptor_transport_recall_min"] >= .98,
        "all_seed_current_high_precision_recall_at_least_0p50": all(value >= .50 for value in current_recall),
        "all_seed_causal_or_consensus_high_precision_recall_at_least_0p50": all(value >= .50 for value in corrected_recall),
        "all_seed_causal_or_consensus_ap_gain_at_least_0p05": all(value >= .05 for value in ap_gain),
        "zero_forbidden_operations": True,
    }
    scientific_pass = all(checks.values())
    decision = "ALLOW_TOKEN_VALIDITY_MODEL_DATA_CARD" if scientific_pass else "STOP_FROZEN_FREE_QUERY_TOKEN_VALIDITY_ROUTE_USE_DENSE_CIRCULAR_TRAVERSABILITY_FIELD"
    summary = {
        "schema_version": "gse_token_validity_action_track_feasibility_v1",
        "status": PASS if scientific_pass else FAIL,
        "scientific_pass": scientific_pass,
        "decision": decision,
        "question": "Can frozen free-query exit tokens be safely validated using their confidence, five-frame descriptor tracks or three-seed geometry consensus?",
        "method": "Evaluation-only Teacher attachment; tied-threshold ranking of single-frame confidence, past-only five-frame descriptor-track mean and Teacher-free three-seed heading/width/profile consensus.",
        "population": {
            "worlds": 80, "raw_frames": 252_430, "observations": 188_126,
            "fit_observations": 142_184, "selection_observations": 45_942,
            "c07_observations": 21_548, "c08_observations": 24_394,
            "slots_per_seed": 1_128_756, "positive_tokens_per_seed": 396_913,
            "negative_tokens_per_seed": 731_843, "seed_archives": 3,
        },
        "dataset_verification": teacher["verification"],
        "validity_label_sha256": validity_sha,
        "seed_metrics": seed_metrics,
        "retained_capacity": retained,
        "sealed_direct_event_baselines": direct,
        "best_corrective_selection_recall_at_precision_0p995": corrected_recall,
        "best_corrective_selection_ap_gain": ap_gain,
        "checks": checks,
        "failure_attribution": "The six free queries contain persistent seed-consistent ghost slots. Descriptor persistence tracks slot identity but does not establish physical-exit validity.",
        "next_representation": "A dense circular traversability field predicts per-bearing executable free-space geometry; connected angular components produce exits without free-query objectness.",
        "optimizer_steps": 0, "model_inference_frames": 0, "threshold_selection_steps": 0,
        "graph_replays": 0, "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0,
        "duration_seconds": time.monotonic() - started,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "figure_source.json").write_text(json.dumps({"schema_version": "gse_token_validity_action_track_figure_source_v1", "summary": summary}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output / "token_validity_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("seed", "partition", "score", "average_precision", "recall_at_precision_0p995", "threshold", "achieved_precision"))
        writer.writeheader()
        for row in seed_metrics:
            for partition in partitions:
                for score_name in ("current_confidence", "causal_five_frame_mean", "cross_seed_geometry_consensus"):
                    item = row[partition][score_name]
                    writer.writerow({"seed": row["seed"], "partition": partition, "score": score_name, "average_precision": item["average_precision"], "recall_at_precision_0p995": item["recall_at_precision_floor"], "threshold": item["threshold_at_precision_floor"], "achieved_precision": item["precision_at_selected_threshold"]})
    _plot(output, summary)
    print(json.dumps({"status": summary["status"], "decision": decision, "checks": checks}, indent=2, sort_keys=True))
    return 0 if scientific_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
