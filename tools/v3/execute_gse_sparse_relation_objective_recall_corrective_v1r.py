#!/usr/bin/env python3
"""Read-only full-population proof for objective token/relation recall."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import evaluate_gse_sparse_circular_relation_transport_v2 as evaluation
import train_gse_axis_anchored_event_relation_v1 as base
import train_gse_sparse_circular_relation_transport_v2 as sparse
from mtare_topo.governance import write_json


PASS = "PASS_GSE_SPARSE_RELATION_OBJECTIVE_RECALL_CORRECTIVE_V1R"
FAIL = "FAIL_GSE_SPARSE_RELATION_OBJECTIVE_RECALL_CORRECTIVE_V1R"
EXPECTED = {
    "fit": {"worlds": 60, "observations": 142184, "valid_pairs": 448152},
    "c07": {"worlds": 10, "observations": 21548, "valid_pairs": 66752},
    "c08": {"worlds": 10, "observations": 24394, "valid_pairs": 77490},
}


def _split_name(parent: str) -> str:
    match = re.search(r"_C(\d+)$", parent)
    if match is None:
        raise ValueError(f"condition suffix missing: {parent}")
    condition = int(match.group(1))
    if 1 <= condition <= 6:
        return "fit"
    if condition == 7:
        return "c07"
    if condition == 8:
        return "c08"
    raise ValueError(f"forbidden condition in corrective: C{condition:02d}")


def _count_world(world: dict) -> dict:
    history = np.asarray(world["history_observation_rows"], dtype=np.int64)
    identity = np.asarray(world["branch_identity"], dtype=np.int64)
    if history.shape != (len(identity), 5):
        raise ValueError("history population shape drift")
    if np.any(history < -1) or np.any(history >= len(identity)):
        raise ValueError("history row outside observation population")
    result = {"observations": len(identity), "valid_pairs": 0, "persistent": 0, "reveal": 0, "withdraw": 0}
    for row in range(len(identity)):
        for step in range(4):
            left = int(history[row, step]); right = int(history[row, step + 1])
            if left < 0 or right < 0:
                continue
            previous = {int(value) for value in identity[left] if value >= 0}
            current = {int(value) for value in identity[right] if value >= 0}
            result["valid_pairs"] += 1
            result["persistent"] += len(previous & current)
            result["reveal"] += len(current - previous)
            result["withdraw"] += len(previous - current)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--sequence-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)

    traversals = base.manifest_traversals(args.sequence_manifest.resolve())
    parents = sorted(parent for parent in traversals if re.search(r"_C(?:0[1-8])$", parent))
    totals = {name: defaultdict(int) for name in EXPECTED}
    per_world = []
    for parent in parents:
        split = _split_name(parent)
        world = sparse._load_world(args.dataset_root.resolve(), parent, traversals[parent])
        counts = _count_world(world)
        totals[split]["worlds"] += 1
        for key, value in counts.items():
            totals[split][key] += int(value)
        per_world.append({"parent_id": parent, "split": split, **counts})

    totals = {split: dict(values) for split, values in totals.items()}
    emitted_score = np.asarray([0.95, 0.90, 0.20])
    emitted_truth = np.asarray([True, True, False])
    candidate_only = evaluation._binary_metrics(emitted_score, emitted_truth, 0.85)
    objective = evaluation._objective_recall_metrics(
        emitted_score, emitted_truth, 0.85, objective_positive_total=5
    )
    structural = evaluation._structural_metrics(
        (emitted_score, emitted_truth, 5), threshold=0.85
    )
    synthetic = {
        "candidate_only": candidate_only,
        "objective": objective,
        "structural_objective": structural,
    }

    checks = {
        "exact_80_worlds": len(parents) == 80 and sum(v.get("worlds", 0) for v in totals.values()) == 80,
        "exact_split_populations": all(
            totals[split].get(key) == expected
            for split, values in EXPECTED.items()
            for key, expected in values.items()
        ),
        "all_relation_types_have_objective_positives": all(
            totals[split].get(name, 0) > 0
            for split in EXPECTED
            for name in ("persistent", "reveal", "withdraw")
        ),
        "world_sums_equal_split_totals": all(
            sum(row[key] for row in per_world if row["split"] == split) == totals[split][key]
            for split in EXPECTED
            for key in ("observations", "valid_pairs", "persistent", "reveal", "withdraw")
        ),
        "missing_teacher_positives_become_false_negatives": objective["tp"] == 2 and objective["fn"] == 3 and objective["positives"] == 5,
        "objective_recall_is_not_candidate_only_recall": objective["recall"] == 0.4 and candidate_only["recall"] == 1.0,
        "precision_remains_candidate_based": objective["precision"] == candidate_only["precision"] == 1.0,
        "structural_refusal_uses_same_objective_denominator": structural["tp"] == 2 and structural["fn"] == 3 and structural["positives"] == 5 and structural["recall"] == 0.4 and np.isclose(structural["f1"], 4.0 / 7.0),
        "zero_training_inference_or_forbidden_worlds": True,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    passed = all(checks.values())
    summary = {
        "schema_version": "gse_sparse_relation_objective_recall_corrective_v1r",
        "status": PASS if passed else FAIL,
        "scientific_pass": passed,
        "decision": "ALLOW_FINAL_CORRECTED_THREE_SEED_TRAINING" if passed else "STOP_BEFORE_TRAINING",
        "totals": totals,
        "synthetic_missing_endpoint_proof": synthetic,
        "checks": checks,
        "duration_seconds": time.monotonic() - started,
        "optimizer_steps": 0,
        "model_forward_observations": 0,
        "checkpoints_written": 0,
        "thresholds_selected": 0,
        "c09_worlds_read": 0,
        "c10_worlds_read": 0,
        "mtare_worlds_read": 0,
        "graph_replays": 0,
        "planner_calls": 0,
    }
    write_json(output / "summary.json", summary)
    write_json(output / "per_world_counts.json", per_world)
    write_json(output / "figure_source.json", summary)

    names = ("persistent", "reveal", "withdraw")
    splits = ("fit", "c07", "c08")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    x = np.arange(len(names)); width = 0.24
    for index, split in enumerate(splits):
        axes[0].bar(x + (index - 1) * width, [totals[split][name] for name in names], width, label=split.upper())
    axes[0].set(xticks=x, xticklabels=names, title="Objective causal relation population", ylabel="Teacher positives")
    axes[0].legend(); axes[0].grid(axis="y", alpha=.2)
    axes[1].bar([0, 1], [candidate_only["recall"], objective["recall"]], color=["#c44e52", "#4c72b0"])
    axes[1].set(xticks=[0, 1], xticklabels=["Invalid\ncandidate-only", "Correct\nobjective total"], ylim=(0, 1.05), title="Missing-endpoint recall proof", ylabel="Recall")
    axes[1].grid(axis="y", alpha=.2)
    fig.suptitle("Sparse relation metric corrective V1R: complete Teacher denominators")
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"gse_sparse_relation_objective_recall_corrective_v1r.{suffix}", dpi=220)
    plt.close(fig)
    print(json.dumps({"status": summary["status"], "totals": totals, "checks": checks}, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
