#!/usr/bin/env python3
"""Audit the Teacher contract for axis-anchored temporal event relations."""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr

from mtare_topo.governance import load_json, write_json
from mtare_topo.semantics.geometric_semantics import EVENT_NAMES


PASS = "PASS_GSE_AXIS_ANCHORED_EVENT_RELATION_TEACHER_FEASIBILITY_V1"
FAIL = "FAIL_GSE_AXIS_ANCHORED_EVENT_RELATION_TEACHER_FEASIBILITY_V1"
EVENT_INDEX = {name: index for index, name in enumerate(EVENT_NAMES)}


def signed_bearing_deg(heading_unit: np.ndarray) -> np.ndarray:
    heading = np.asarray(heading_unit, dtype=np.float64)
    if heading.shape[-1] != 2 or not np.all(np.isfinite(heading)):
        raise ValueError("heading units must be finite [...,2]")
    return np.degrees(np.arctan2(heading[..., 0], heading[..., 1]))


def minimum_circular_separation_deg(bearing: np.ndarray) -> float:
    values = np.sort(np.remainder(np.asarray(bearing, dtype=np.float64), 360.0))
    if len(values) < 2:
        return 360.0
    gaps = np.diff(np.r_[values, values[0] + 360.0])
    return float(gaps.min())


def _partition(parent: str) -> str:
    match = re.search(r"_C(\d+)$", parent)
    if match is None or not 1 <= int(match.group(1)) <= 8:
        raise ValueError(f"parent outside C01-C08: {parent}")
    return "fit" if int(match.group(1)) <= 6 else "selection"


def _sequence_manifest(path: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            parent = str(record["parent_id"])
            match = re.search(r"_C(\d+)$", parent)
            if str(record["split"]) != "train" or match is None or int(match.group(1)) > 8:
                continue
            rows = result.setdefault(parent, [])
            if int(record["world_sequence_row"]) != len(rows):
                raise RuntimeError(f"non-contiguous sequence manifest: {parent}")
            rows.append(str(record["traversal_id"]))
    return result


def _identity_record() -> dict:
    return {"partition": None, "event": None, "observations": 0, "full_rows": 0, "traversals": set(), "stars": set()}


def _audit(dataset_root: Path, sequence_manifest: Path) -> tuple[dict, list[dict], dict]:
    traversal_by_parent = _sequence_manifest(sequence_manifest)
    if len(traversal_by_parent) != 80:
        raise RuntimeError(f"expected 80 C01-C08 worlds, got {len(traversal_by_parent)}")
    observations = Counter(); visible_exits = Counter(); raw_frames = Counter()
    event_observations = {part: np.zeros(len(EVENT_NAMES), dtype=np.int64) for part in ("fit", "selection")}
    visible_cardinality = {part: np.zeros(7, dtype=np.int64) for part in ("fit", "selection")}
    geometry_valid = {part: np.zeros(4, dtype=np.int64) for part in ("fit", "selection")}
    exit_width_valid = Counter()
    identities: dict[int, dict] = defaultdict(_identity_record)
    temporal = {part: Counter() for part in ("fit", "selection")}
    maximum_anchor_lateral = 0.0; minimum_anchor_forward = math.inf
    minimum_exit_separation = 360.0
    example = None
    for parent in sorted(traversal_by_parent):
        part = _partition(parent)
        group = zarr.open_group(str(dataset_root / f"{parent}.zarr"), mode="r")
        traversal = np.asarray(traversal_by_parent[parent])
        event = np.asarray(group["event_index"][:], dtype=np.int64)
        mask = np.asarray(group["exit_mask"][:], dtype=bool)
        heading = np.asarray(group["exit_heading_unit"][:], dtype=np.float64)
        exit_identity = np.asarray(group["exit_identity"][:], dtype=np.int64)
        association_identity = np.asarray(group["association_identity"][:], dtype=np.int64)
        association_valid = np.asarray(group["association_valid_mask"][:], dtype=bool)
        invisible = np.asarray(group["exit_invisible_count"][:], dtype=np.int64)
        axis = np.asarray(group["local_axis_robot"][:], dtype=np.float64)
        geometry_mask = np.asarray(group["geometry_valid_mask"][:], dtype=bool)
        width_mask = np.asarray(group["exit_width_valid_mask"][:], dtype=bool)
        if len(traversal) != len(event) or mask.shape != (len(event), 6) or heading.shape != (len(event), 6, 2):
            raise RuntimeError(f"dataset/manifest shape drift: {parent}")
        observations[part] += len(event); visible_exits[part] += int(mask.sum()); raw_frames[part] += len(group["global_frame_index"])
        event_observations[part] += np.bincount(event, minlength=len(EVENT_NAMES))
        visible_cardinality[part] += np.bincount(mask.sum(axis=1), minlength=7)
        geometry_valid[part] += geometry_mask.sum(axis=0); exit_width_valid[part] += int(width_mask.sum())
        maximum_anchor_lateral = max(maximum_anchor_lateral, float(np.abs(axis[:, 1]).max()))
        minimum_anchor_forward = min(minimum_anchor_forward, float(axis[:, 0].min()))
        for row in range(len(event)):
            bearings = signed_bearing_deg(heading[row, mask[row]])
            minimum_exit_separation = min(minimum_exit_separation, minimum_circular_separation_deg(bearings))
            if association_valid[row]:
                key = int(association_identity[row]); record = identities[key]
                if record["partition"] not in (None, part) or record["event"] not in (None, int(event[row])):
                    raise RuntimeError(f"association identity crosses partition or event: {key}")
                record["partition"] = part; record["event"] = int(event[row]); record["observations"] += 1
                record["traversals"].add(str(traversal[row]))
                if invisible[row] == 0:
                    star = tuple(sorted(int(value) for value in exit_identity[row, mask[row]]))
                    record["full_rows"] += 1; record["stars"].add(star)
                    if example is None and part == "selection" and int(event[row]) == EVENT_INDEX["junction"] and len(star) == 4:
                        example = {"parent_id": parent, "association_identity": key, "bearings_deg": bearings.tolist(), "exit_identity_teacher_only": list(star)}
        for row in range(1, len(event)):
            if traversal[row] != traversal[row - 1]:
                continue
            previous = set(int(value) for value in exit_identity[row - 1, mask[row - 1]])
            current = set(int(value) for value in exit_identity[row, mask[row]])
            temporal[part]["adjacent_pairs"] += 1
            temporal[part]["persistent_tokens"] += len(previous & current)
            temporal[part]["revealed_tokens"] += len(current - previous)
            temporal[part]["withdrawn_tokens"] += len(previous - current)
            temporal[part]["changed_pairs"] += previous != current
    identity_rows = []
    for identity, record in sorted(identities.items()):
        stars = record["stars"]
        canonical_count = len(next(iter(stars))) if len(stars) == 1 else -1
        identity_rows.append({
            "association_identity_teacher_only": identity, "partition": record["partition"],
            "event": EVENT_NAMES[record["event"]], "observations": record["observations"],
            "distinct_traversals": len(record["traversals"]), "full_visibility_rows": record["full_rows"],
            "star_variants": len(stars), "canonical_exit_count": canonical_count,
        })
    population = {
        "worlds": len(traversal_by_parent), "fit_worlds": 60, "selection_worlds": 20,
        "observations": int(sum(observations.values())), "fit_observations": int(observations["fit"]), "selection_observations": int(observations["selection"]),
        "raw_frames": int(sum(raw_frames.values())), "visible_exit_tokens": int(sum(visible_exits.values())),
        "fit_visible_exit_tokens": int(visible_exits["fit"]), "selection_visible_exit_tokens": int(visible_exits["selection"]),
        "association_identities": len(identity_rows),
    }
    metrics = {
        "population": population,
        "event_observations": {part: {name: int(event_observations[part][index]) for index, name in enumerate(EVENT_NAMES)} for part in ("fit", "selection")},
        "visible_cardinality": {part: {str(index): int(value) for index, value in enumerate(visible_cardinality[part]) if value} for part in ("fit", "selection")},
        "geometry_valid": {part: {name: int(geometry_valid[part][index]) for index, name in enumerate(("width", "height", "slope", "curvature"))} for part in ("fit", "selection")},
        "exit_width_valid": dict(exit_width_valid),
        "axis_anchor": {"maximum_absolute_lateral_component": maximum_anchor_lateral, "minimum_forward_component": minimum_anchor_forward},
        "minimum_visible_exit_separation_deg": minimum_exit_separation,
        "temporal_relations": {part: dict(values) for part, values in temporal.items()},
        "example_relation_star": example,
    }
    return metrics, identity_rows, {"event_observations": event_observations, "visible_cardinality": visible_cardinality}


def _identity_summary(identity_rows: list[dict]) -> dict:
    result = {}
    for partition in ("fit", "selection"):
        result[partition] = {}
        for event in EVENT_NAMES[1:]:
            rows = [row for row in identity_rows if row["partition"] == partition and row["event"] == event]
            cardinality = Counter(row["canonical_exit_count"] for row in rows)
            result[partition][event] = {
                "identities": len(rows), "multi_traversal_identities": sum(row["distinct_traversals"] >= 2 for row in rows),
                "minimum_traversals": min((row["distinct_traversals"] for row in rows), default=0),
                "minimum_full_visibility_rows": min((row["full_visibility_rows"] for row in rows), default=0),
                "exact_canonical_star_identities": sum(row["star_variants"] == 1 for row in rows),
                "canonical_exit_count": {str(key): value for key, value in sorted(cardinality.items())},
            }
    return result


def _plot(output: Path, summary: dict) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(15.2, 8.4), constrained_layout=True)
    names = list(EVENT_NAMES)
    x = np.arange(len(names))
    for offset, part in enumerate(("fit", "selection")):
        axes[0, 0].bar(x + (offset - .5) * .36, [summary["teacher"]["event_observations"][part][name] for name in names], .36, label=part)
    axes[0, 0].set_yscale("log"); axes[0, 0].set_xticks(x, [name.replace("geometry_", "geom. ") for name in names], rotation=25, ha="right"); axes[0, 0].set_title("A  Event supervision population"); axes[0, 0].legend(frameon=False)
    semantic = names[1:]
    for offset, part in enumerate(("fit", "selection")):
        axes[0, 1].bar(np.arange(4) + (offset - .5) * .36, [summary["identities"][part][name]["identities"] for name in semantic], .36, label=part)
    axes[0, 1].set_yscale("log"); axes[0, 1].set_xticks(range(4), [name.replace("geometry_", "geom. ") for name in semantic], rotation=20, ha="right"); axes[0, 1].set_title("B  Independent event identities")
    width = .22
    for index, count in enumerate((2, 3, 4)):
        axes[0, 2].bar(np.arange(2) + (index - 1) * width, [summary["identities"][part]["junction"]["canonical_exit_count"].get(str(count), 0) for part in ("fit", "selection")], width, label=f"{count} exits")
    axes[0, 2].set_xticks((0, 1), ("fit", "selection")); axes[0, 2].set_title("C  Junction relation-star size"); axes[0, 2].legend(frameon=False)
    temporal_names = ("persistent_tokens", "revealed_tokens", "withdrawn_tokens")
    for offset, part in enumerate(("fit", "selection")):
        axes[1, 0].bar(np.arange(3) + (offset - .5) * .36, [summary["teacher"]["temporal_relations"][part][name] for name in temporal_names], .36, label=part)
    axes[1, 0].set_yscale("log"); axes[1, 0].set_xticks(range(3), ("persistent", "reveal", "withdraw")); axes[1, 0].set_title("D  Causal token relations")
    prior = summary["prior_evidence"]
    labels = ("transport P", "transport R", "commit P", "commit R")
    fit_values = (prior["descriptor_transport"]["fit_min_precision"], prior["descriptor_transport"]["fit_min_recall"], prior["stateful_commit"]["c07_precision"], prior["stateful_commit"]["c07_recall"])
    selection_values = (prior["descriptor_transport"]["selection_min_precision"], prior["descriptor_transport"]["selection_min_recall"], prior["stateful_commit"]["c08_precision"], prior["stateful_commit"]["c08_recall"])
    axes[1, 1].bar(np.arange(4) - .18, fit_values, .36, label="fit/C07"); axes[1, 1].bar(np.arange(4) + .18, selection_values, .36, label="selection/C08"); axes[1, 1].set_xticks(range(4), labels, rotation=20); axes[1, 1].set_ylim(0, 1.03); axes[1, 1].set_title("E  Reusable evidence, not new result"); axes[1, 1].legend(frameon=False)
    example = summary["teacher"]["example_relation_star"]
    if example is not None:
        angle = np.deg2rad(example["bearings_deg"]); axes[1, 2].scatter(np.sin(angle), np.cos(angle), s=90, color="#e15759")
        for index, value in enumerate(example["bearings_deg"]): axes[1, 2].text(np.sin(angle[index]) + .04, np.cos(angle[index]) + .04, f"{value:.1f}°", fontsize=8)
    axes[1, 2].arrow(0, 0, 0, .72, width=.015, color="#4e79a7", length_includes_head=True); axes[1, 2].text(.04, .58, "route axis", color="#4e79a7"); axes[1, 2].set(xlim=(-1.15, 1.15), ylim=(-1.15, 1.15), aspect="equal", title="F  Axis-anchored K4 relation Teacher"); axes[1, 2].axis("off")
    for axis in axes.flat[:5]: axis.grid(axis="y", alpha=.2); axis.set_axisbelow(True)
    figure.suptitle("GSE-Graph axis-anchored event–relation Teacher feasibility")
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output / f"gse_axis_anchored_event_relation_teacher_feasibility_v1.{suffix}", dpi=220)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--sequence-manifest", required=True, type=Path)
    parser.add_argument("--exit-action-summary", required=True, type=Path)
    parser.add_argument("--commit-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=False); started = time.monotonic()
    teacher, identity_rows, _ = _audit(args.dataset_root.resolve(), args.sequence_manifest.resolve())
    identity_summary = _identity_summary(identity_rows)
    exit_action = load_json(args.exit_action_summary.resolve())["result"]
    commit = load_json(args.commit_summary.resolve())
    descriptor = exit_action["seed_transport"]
    prior = {
        "descriptor_transport": {
            "fit_min_precision": min(row["fit"]["precision"] for row in descriptor), "fit_min_recall": min(row["fit"]["recall"] for row in descriptor),
            "selection_min_precision": min(row["selection"]["precision"] for row in descriptor), "selection_min_recall": min(row["selection"]["recall"] for row in descriptor),
            "scope": exit_action["important_scope"],
        },
        "stateful_commit": {
            "c07_precision": commit["selected_policy"]["c07"]["precision"], "c07_recall": commit["selected_policy"]["c07"]["recall"],
            "c08_precision": commit["selected_policy"]["c08"]["precision"], "c08_recall": commit["selected_policy"]["c08"]["recall"],
            "c07_macro_f1": commit["selected_policy"]["c07"]["macro_f1"], "c08_macro_f1": commit["selected_policy"]["c08"]["macro_f1"],
            "scope": "Old hand-written commit baseline is reusable evidence/ablation only; it did not satisfy the prior macro-F1 gain gate.",
        },
    }
    expected = {"worlds": 80, "fit_observations": 142184, "selection_observations": 45942, "observations": 188126, "visible_exit_tokens": 396913, "association_identities": 4493}
    junction_fraction = {part: sum(identity_summary[part]["junction"]["canonical_exit_count"].get(str(k), 0) for k in (3, 4)) / identity_summary[part]["junction"]["identities"] for part in ("fit", "selection")}
    checks = {
        "exact_population": all(teacher["population"][name] == value for name, value in expected.items()),
        "route_axis_is_causally_fixed_in_sensor_frame": teacher["axis_anchor"]["maximum_absolute_lateral_component"] <= 1e-6 and teacher["axis_anchor"]["minimum_forward_component"] > 0.85,
        "all_event_identities_have_exact_full_star": all(row["star_variants"] == 1 and row["full_visibility_rows"] >= 1 for row in identity_rows),
        "all_junction_identities_have_at_least_four_route_views": all(identity_summary[part]["junction"]["minimum_traversals"] >= 4 for part in ("fit", "selection")),
        "junction_k3_k4_identity_fraction_at_least_0p95": all(value >= .95 for value in junction_fraction.values()),
        "selection_has_at_least_50_identities_per_structural_event": all(identity_summary["selection"][name]["identities"] >= 50 for name in EVENT_NAMES[1:]),
        "visible_exit_centers_separated_by_more_than_2deg": teacher["minimum_visible_exit_separation_deg"] > 2.0,
        "fit_and_selection_have_reveal_withdraw_supervision": all(teacher["temporal_relations"][part][name] >= threshold for part, threshold in (("fit", 1000), ("selection", 500)) for name in ("revealed_tokens", "withdrawn_tokens")),
        "global_geometry_masks_cover_at_least_0p98": all(teacher["geometry_valid"][part][name] / teacher["population"][f"{part}_observations"] >= .98 for part in ("fit", "selection") for name in ("width", "height", "slope", "curvature")),
        "prior_descriptor_transport_precision_recall_at_least_0p98": min(prior["descriptor_transport"][name] for name in ("fit_min_precision", "fit_min_recall", "selection_min_precision", "selection_min_recall")) >= .98,
        "prior_stateful_commit_is_safe_but_not_claimed_as_main": prior["stateful_commit"]["c07_precision"] >= .995 and prior["stateful_commit"]["c08_precision"] >= .995 and prior["stateful_commit"]["c07_recall"] >= .25 and prior["stateful_commit"]["c08_recall"] >= .25,
        "zero_training_inference_test_graph": True,
    }
    scientific_pass = all(checks.values())
    summary = {
        "schema_version": "gse_axis_anchored_event_relation_teacher_feasibility_v1",
        "status": PASS if scientific_pass else FAIL, "scientific_pass": scientific_pass,
        "decision": "ALLOW_AXIS_ANCHORED_EVENT_RELATION_METHOD_READINESS" if scientific_pass else "STOP_AXIS_ANCHORED_EVENT_RELATION_METHOD",
        "question": "Can one-shot complete-set regression be replaced by a causally anchored event episode with persistent/reveal/withdraw exit relations and execution-verified graph updates?",
        "method_contract": {
            "runtime_inputs": "five causal LiDAR frames, sensor-forward route axis/odometry, and past-only token memory",
            "learned_outputs": "five event probabilities, local geometry, persistent/reveal/withdraw branch relations, branch geometry/descriptors, uncertainty",
            "teacher_only": "association and directed-exit identities are used only for MIL/contrastive relation supervision and never enter forward",
            "graph_rule": "stable event commits a node; only physical traversal commits an edge",
            "difference_from_old_event_model": "explicit axis-relative pairwise branch geometry plus episode branch-union memory; no mean/max pooling and no one-shot exact complete-set dependency",
        },
        "teacher": teacher, "identities": identity_summary, "junction_k3_k4_identity_fraction": junction_fraction,
        "prior_evidence": prior, "checks": checks, "duration_seconds": time.monotonic() - started,
        "optimizer_steps": 0, "new_model_inference_observations": 0, "checkpoint_selection_observations": 0,
        "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0, "graph_replays": 0, "planner_calls": 0,
    }
    write_json(output / "summary.json", summary); write_json(output / "figure_source.json", summary)
    with (output / "identity_contract.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(identity_rows[0])); writer.writeheader(); writer.writerows(identity_rows)
    with (output / "event_population.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream); writer.writerow(("partition", "event", "observations", "identities"))
        for part in ("fit", "selection"):
            for name in EVENT_NAMES:
                writer.writerow((part, name, teacher["event_observations"][part][name], 0 if name == "corridor" else identity_summary[part][name]["identities"]))
    _plot(output, summary)
    print(json.dumps({"status": summary["status"], "decision": summary["decision"], "checks": checks}, indent=2, sort_keys=True))
    return 0 if scientific_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
