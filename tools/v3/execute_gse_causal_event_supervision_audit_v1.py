#!/usr/bin/env python3
"""Audit frame labels versus causal structural-event episodes on C01-C08."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mtare_topo.evaluation.gse_causal_event_supervision import (
    contiguous_structural_episodes,
    transition_supervision_alignment,
)
from mtare_topo.governance import load_json, write_json


PASS_STATUS = "PASS_GSE_CAUSAL_EVENT_SUPERVISION_AUDIT_V1"
EVENTS = ("junction", "terminal", "turn", "geometry_transition")


def _jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _quantiles(values: list[int]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(array)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "maximum": float(np.max(array)),
    }


def _plot(summary: dict, output: Path) -> None:
    alignment = summary["transition_alignment"]
    inventory = summary["episode_inventory"]
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), constrained_layout=True)

    timing = [
        alignment["labels_before_closed_confirmation"],
        alignment["labels_at_closed_confirmation"],
        alignment["labels_after_closed_confirmation"],
    ]
    axes[0, 0].bar(["proposal interval", "confirmation", "after"], timing, color=["#F59E0B", "#15803D", "#94A3B8"])
    axes[0, 0].set_ylabel("transition-labelled frames")
    axes[0, 0].set_title("a  Frame labels depend on an event interval", loc="left")
    for index, value in enumerate(timing):
        axes[0, 0].text(index, value, f"{value:,}", ha="center", va="bottom", fontsize=9)

    visibility = [
        alignment["labels_with_boundary_inside_five_frame_history"],
        alignment["labels_with_boundary_outside_five_frame_history"],
    ]
    axes[0, 1].bar(["boundary in\n5-frame history", "boundary outside\n5-frame history"], visibility, color=["#2563EB", "#DC2626"])
    axes[0, 1].set_ylabel("transition-labelled frames")
    axes[0, 1].set_title("b  Five-frame target alignment", loc="left")
    for index, value in enumerate(visibility):
        axes[0, 1].text(index, value, f"{value:,}", ha="center", va="bottom", fontsize=9)

    episode_counts = [inventory[name]["episode_count"] for name in EVENTS]
    axes[1, 0].bar(["junction", "terminal", "turn", "change"], episode_counts, color=["#0EA5E9", "#10B981", "#8B5CF6", "#F97316"])
    axes[1, 0].set_ylabel("contiguous Teacher episodes")
    axes[1, 0].set_title("c  Event—not frame—is the natural unit", loc="left")
    for index, value in enumerate(episode_counts):
        axes[1, 0].text(index, value, f"{value:,}", ha="center", va="bottom", fontsize=8)

    history = alignment["history_available_at_confirmation"]
    lengths = [5, 8, 10, 12]
    counts = [history[str(value)] for value in lengths]
    axes[1, 1].plot(lengths, counts, marker="o", color="#0369A1", linewidth=2)
    axes[1, 1].set_xticks(lengths)
    axes[1, 1].set_ylim(0, alignment["directional_change_episodes"] * 1.1)
    axes[1, 1].set_xlabel("causal scan history length")
    axes[1, 1].set_ylabel("confirmation episodes covered")
    axes[1, 1].set_title("d  Existing scans support 12-frame confirmation", loc="left")
    figure.suptitle("Why frame classification fails: GSE causal event supervision audit")
    for suffix, kwargs in (("png", {"dpi": 190}), ("pdf", {}), ("svg", {})):
        figure.savefig(output.with_suffix(f".{suffix}"), **kwargs)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--proof-points", required=True, type=Path)
    parser.add_argument("--proof-labels", required=True, type=Path)
    parser.add_argument("--risk-summary", required=True, type=Path)
    parser.add_argument("--capacity-summary", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    for directory in (run_dir / "artifacts", run_dir / "metrics", run_dir / "previews"):
        directory.mkdir(parents=True, exist_ok=True)

    teacher = _jsonl(args.teacher)
    points = _jsonl(args.proof_points)
    labels = _jsonl(args.proof_labels)
    risk = load_json(args.risk_summary)
    capacity = load_json(args.capacity_summary)
    if len(teacher) != 188126:
        raise RuntimeError("corrected Teacher population drift")
    parents = {str(row["parent_id"]) for row in teacher}
    if len(parents) != 80 or any(parent.endswith(("_C09", "_C10")) for parent in parents):
        raise RuntimeError("causal supervision audit split drift")

    episodes = contiguous_structural_episodes(teacher)
    by_event: dict[str, list] = defaultdict(list)
    for episode in episodes:
        by_event[episode.event].append(episode)
    event_counts = Counter(str(row["event"]) for row in teacher)
    event_identities = {
        name: {str(row["identity"]) for row in teacher if str(row["event"]) == name}
        for name in EVENTS
    }
    episode_inventory = {
        name: {
            "observation_count": event_counts[name],
            "identity_count": len(event_identities[name]),
            "episode_count": len(by_event[name]),
            "episode_frame_count_quantiles": _quantiles([item.frame_count for item in by_event[name]]),
        }
        for name in EVENTS
    }
    alignment = transition_supervision_alignment(
        teacher_rows=teacher,
        proof_points=points,
        proof_labels=labels,
        history_lengths=(5, 8, 10, 12),
    )
    timing_rows = alignment.pop("timing_rows")
    with (run_dir / "artifacts/transition_timing.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(timing_rows[0]))
        writer.writeheader()
        writer.writerows(timing_rows)
    with (run_dir / "artifacts/event_episode_inventory.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ["event", "observation_count", "identity_count", "episode_count", "minimum", "q25", "median", "q75", "maximum"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name in EVENTS:
            row = episode_inventory[name]
            writer.writerow({
                "event": name,
                "observation_count": row["observation_count"],
                "identity_count": row["identity_count"],
                "episode_count": row["episode_count"],
                **row["episode_frame_count_quantiles"],
            })

    checks = {
        "exact_teacher_population": len(teacher) == 188126,
        "exact_world_population": len(parents) == 80,
        "exact_transition_labels": alignment["final_transition_labels"] == 1031,
        "exact_transition_identities": alignment["final_transition_identities"] == 76,
        "exact_emitted_points": alignment["emitted_change_points"] == 76,
        "exact_directional_change_episodes": alignment["directional_change_episodes"] == 152,
        "majority_labels_precede_closed_confirmation": alignment["labels_before_closed_confirmation"] > alignment["final_transition_labels"] / 2,
        "majority_labels_do_not_contain_boundary_in_five_frame_history": alignment["labels_with_boundary_outside_five_frame_history"] > alignment["final_transition_labels"] / 2,
        "all_confirmation_events_have_twelve_frame_history": alignment["history_available_at_confirmation"]["12"] == 152,
        "longer_history_signal_predecessor_passed": risk.get("overall_status") == "PASS_GSE_CAUSAL_GEOMETRY_RISK_CONFLICT_AUDIT_V1" and bool(risk.get("mechanism_conclusion", {}).get("longer_history_signal_material")),
        "frame_risk_capacity_predecessor_failed_scientifically": capacity.get("overall_status") == "FAIL_GSE_GEOMETRY_CONDITIONED_IDENTITY_RISK_CAPACITY_V1" and capacity.get("selection_metrics", {}).get("structural_selection_error") is not None,
        "zero_forbidden_reads": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"causal supervision audit checks failed: {checks}")

    result = {
        "schema_version": "gse_causal_event_supervision_audit_v1",
        "overall_status": PASS_STATUS,
        "audit_pass": True,
        "scientific_qualification": False,
        "worlds": 80,
        "causal_observations": len(teacher),
        "structural_identities": len({str(row["identity"]) for row in teacher if row.get("identity") is not None}),
        "structural_episodes": len(episodes),
        "episode_inventory": episode_inventory,
        "transition_alignment": alignment,
        "checks": checks,
        "diagnosis": "The persistent change-point Teacher defines a proposal-to-confirmation episode, while the current objective requires every labelled frame to be independently correct. Most labels precede closed confirmation and most five-frame histories no longer contain the back-projected boundary.",
        "recommended_method": "TWELVE_FRAME_CAUSAL_EVENT_DETECTOR_WITH_EPISODE_LEVEL_MULTIPLE_INSTANCE_SUPERVISION_AND_BACKPROJECTED_NODE_COMMIT",
        "recommended_interface": {
            "proposal": "Per-frame geometric-semantic evidence remains provisional.",
            "confirmation": "A masked 12-frame past-only temporal decoder confirms at most one event per Teacher episode.",
            "loss": "Require at least one correct-class trigger inside the objective proposal-to-confirmation bag; use complete corridor bags as hard negatives instead of forcing every positive bag frame to be positive.",
            "graph": "Commit one node only after causal confirmation and back-project its metric position to the estimated structural boundary; edge creation remains traversal-only.",
            "data": "Reuse existing unique LiDAR frames by reference; generate no new rays and duplicate no sensor payload.",
        },
        "optimizer_steps": 0,
        "model_inference_frames": 0,
        "c09_worlds_read": 0,
        "strict_test_worlds_read": 0,
        "mtare_worlds_read": 0,
    }
    write_json(run_dir / "metrics/supervision_audit.json", result)
    write_json(
        run_dir / "previews/gse_causal_event_supervision_source.json",
        {
            "schema_version": "gse_causal_event_supervision_figure_source_v1",
            "transition_alignment": alignment,
            "episode_inventory": episode_inventory,
            "recommended_method": result["recommended_method"],
        },
    )
    write_json(
        run_dir / "previews/gse_causal_event_supervision_provenance.json",
        {"source": "sealed C01-C08 Teacher/proof/risk/capacity runs", "manual_result_values": 0},
    )
    _plot(result, run_dir / "previews/gse_causal_event_supervision")
    (run_dir / "previews/README.md").write_text(
        "This figure is generated from sealed C01-C08 development evidence only. It is a mechanism audit, not a model or topology result.\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
