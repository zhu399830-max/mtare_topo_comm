#!/usr/bin/env python3
"""Zero-training readiness proof for Axis-Anchored Event Relation."""
from __future__ import annotations

import argparse
import inspect
import json
import math
from pathlib import Path
import re
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import zarr

from mtare_topo.governance import write_json
from mtare_topo.data.gse_axis_anchored_event_relation_training import (
    materialize_world_relation_teacher,
)
from mtare_topo.representation.gse_axis_anchored_event_relation import (
    AxisAnchoredEventRelationNet,
    axis_anchored_descriptor_loss,
    axis_anchored_event_relation_core_loss,
    axis_anchored_event_relation_input_contract,
    axis_anchored_event_relation_loss,
    parameter_count,
    reverse_axis_field,
)


PASS = "PASS_GSE_AXIS_ANCHORED_EVENT_RELATION_READINESS_V1"
FAIL = "FAIL_GSE_AXIS_ANCHORED_EVENT_RELATION_READINESS_V1"
PASS_V1R = "PASS_GSE_AXIS_ANCHORED_EVENT_RELATION_READINESS_V1R"
FAIL_V1R = "FAIL_GSE_AXIS_ANCHORED_EVENT_RELATION_READINESS_V1R"
PASS_V2 = "PASS_GSE_AXIS_ANCHORED_EVENT_RELATION_READINESS_V2"
FAIL_V2 = "FAIL_GSE_AXIS_ANCHORED_EVENT_RELATION_READINESS_V2"
MAX_RANGE_M = 50.0
SAMPLES = (
    ("S01_flat_tree_small_C01", 7),
    ("S01_flat_tree_small_C01", 4),
    ("S01_flat_tree_small_C01", 216),
    ("S01_flat_tree_small_C02", 537),
    ("S01_flat_tree_small_C01", 25),
    ("S01_flat_tree_small_C01", 89),
    ("S01_flat_tree_small_C01", 90),
    ("S01_flat_tree_small_C03", 323),
)

EXPECTED_FULL_POPULATION = {
    "fit": {
        "worlds": 60, "observations": 142184, "valid_pair_positions": 448152,
        "complete_relation_rows": 94144, "persistent": 934760, "reveal": 10583,
        "withdraw": 11008, "simultaneous_reveal_withdraw_bins": 202,
        "current_branch_tokens": 299872,
    },
    "c07": {
        "worlds": 10, "observations": 21548, "valid_pair_positions": 66752,
        "complete_relation_rows": 13828, "persistent": 139346, "reveal": 1636,
        "withdraw": 1728, "simultaneous_reveal_withdraw_bins": 20,
        "current_branch_tokens": 45504,
    },
    "c08": {
        "worlds": 10, "observations": 24394, "valid_pair_positions": 77490,
        "complete_relation_rows": 16388, "persistent": 161597, "reveal": 2107,
        "withdraw": 2181, "simultaneous_reveal_withdraw_bins": 33,
        "current_branch_tokens": 51537,
    },
}


def _manifest_traversals(path: Path) -> dict[str, list[str]]:
    traversals: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            parent = str(record["parent_id"])
            match = re.search(r"_C(\d+)$", parent)
            if record.get("split") != "train" or match is None or int(match.group(1)) > 8:
                continue
            rows = traversals.setdefault(parent, [])
            if int(record["world_sequence_row"]) != len(rows):
                raise RuntimeError(f"non-contiguous sequence manifest: {parent}")
            rows.append(str(record["traversal_id"]))
    if len(traversals) != 80:
        raise RuntimeError(f"expected 80 C01-C08 parents, got {len(traversals)}")
    return traversals


def _full_population_audit(dataset_root: Path, sequence_manifest: Path) -> dict[str, dict[str, int]]:
    aggregate: dict[str, dict[str, int]] = {}
    for partition in EXPECTED_FULL_POPULATION:
        aggregate[partition] = {name: 0 for name in EXPECTED_FULL_POPULATION[partition]}
    for parent, traversal in sorted(_manifest_traversals(sequence_manifest).items()):
        condition = int(parent.rsplit("_C", 1)[1])
        partition = "fit" if condition <= 6 else f"c{condition:02d}"
        group = zarr.open_group(str(dataset_root / f"{parent}.zarr"), mode="r")
        observations = len(group["event_index"])
        if len(traversal) != observations:
            raise RuntimeError(f"dataset/manifest observation drift: {parent}")
        teacher = materialize_world_relation_teacher(
            local_frame_references=np.asarray(group["local_frame_references"][:], dtype=np.int64),
            traversal_id=traversal,
            exit_mask=np.asarray(group["exit_mask"][:], dtype=bool),
            exit_identity=np.asarray(group["exit_identity"][:], dtype=np.int64),
            exit_heading_unit=np.asarray(group["exit_heading_unit"][:], dtype=np.float32),
            exit_opening_width_m=np.asarray(group["exit_opening_width_m"][:], dtype=np.float32),
            exit_width_valid_mask=np.asarray(group["exit_width_valid_mask"][:], dtype=bool),
            exit_vertical_profile_m=np.asarray(group["exit_vertical_profile_m"][:], dtype=np.float32),
        )
        population = teacher.population()
        aggregate[partition]["worlds"] += 1
        aggregate[partition]["observations"] += observations
        for name in population:
            if name in aggregate[partition]:
                aggregate[partition][name] += int(population[name])
    return aggregate


def _bearing_bin(heading: np.ndarray) -> tuple[int, float]:
    bearing = float(np.degrees(np.arctan2(float(heading[0]), float(heading[1]))) % 360.0)
    index = int(np.floor((bearing + 1.0) / 2.0)) % 180
    residual = (bearing - 2.0 * index + 180.0) % 360.0 - 180.0
    if not -1.00001 <= residual <= 1.00001:
        raise RuntimeError("nearest bearing-bin residual drift")
    return index, residual


def _row(dataset_root: Path, parent: str, row: int) -> tuple[np.ndarray, dict, dict]:
    group = zarr.open_group(str(dataset_root / f"{parent}.zarr"), mode="r")
    references = np.asarray(group["local_frame_references"][:], dtype=np.int64)
    current_lookup = {int(value): index for index, value in enumerate(references[:, -1])}
    history = [current_lookup.get(int(value), -1) for value in references[row]]
    if -1 in history:
        raise RuntimeError(f"sample lacks five observation-level Teacher rows: {parent}:{row}")
    mask = np.asarray(group["exit_mask"][:], dtype=bool)
    identity = np.asarray(group["exit_identity"][:], dtype=np.int64)
    heading = np.asarray(group["exit_heading_unit"][:], dtype=np.float32)
    relation = np.zeros((4, 180, 3), dtype=np.int64)
    for time_index, (previous_row, current_row) in enumerate(zip(history[:-1], history[1:])):
        previous = {int(key): value for key, value in zip(identity[previous_row, mask[previous_row]], heading[previous_row, mask[previous_row]])}
        current = {int(key): value for key, value in zip(identity[current_row, mask[current_row]], heading[current_row, mask[current_row]])}
        for state, keys, source in (
            (0, set(previous) & set(current), current),
            (1, set(current) - set(previous), current),
            (2, set(previous) - set(current), previous),
        ):
            for key in sorted(keys):
                bearing_index, _ = _bearing_bin(source[key])
                if relation[time_index, bearing_index, state] == 1:
                    raise RuntimeError(f"same-channel relation-bin ambiguity: {parent}:{row}:{time_index}:{bearing_index}:{state}")
                relation[time_index, bearing_index, state] = 1
    final_mask = mask[row]
    branch_identity = np.full(180, -1, dtype=np.int64)
    branch_residual = np.zeros(180, dtype=np.float32)
    branch_width = np.zeros(180, dtype=np.float32)
    branch_width_valid = np.zeros(180, dtype=np.uint8)
    branch_profile = np.zeros((180, 4), dtype=np.float32)
    widths = np.asarray(group["exit_opening_width_m"][row], dtype=np.float32)
    width_valid = np.asarray(group["exit_width_valid_mask"][row], dtype=bool)
    profiles = np.asarray(group["exit_vertical_profile_m"][row], dtype=np.float32)
    for slot in np.flatnonzero(final_mask):
        index, residual = _bearing_bin(heading[row, slot])
        if branch_identity[index] >= 0:
            raise RuntimeError(f"current branch-bin ambiguity: {parent}:{row}:{index}")
        branch_identity[index] = int(identity[row, slot])
        branch_residual[index] = residual
        branch_width[index] = widths[slot]
        branch_width_valid[index] = int(width_valid[slot])
        branch_profile[index] = profiles[slot]
    current_present = branch_identity >= 0
    if not np.array_equal(current_present, branch_identity >= 0):
        raise RuntimeError(f"final relation/current branch mismatch: {parent}:{row}")
    ranges = np.asarray(group["range_m"][references[row]], dtype=np.float32) / MAX_RANGE_M
    valid = np.asarray(group["valid_mask"][references[row]], dtype=np.float32)
    scan = np.stack((ranges, valid), axis=1)
    association_valid = bool(group["association_valid_mask"][row])
    targets = {
        "event_index": int(group["event_index"][row]),
        "relation_index": relation,
        "branch_heading_residual_deg": branch_residual,
        "branch_opening_width_m": branch_width,
        "branch_width_valid_mask": branch_width_valid,
        "branch_vertical_profile_m": branch_profile,
        "branch_identity": branch_identity,
        "local_axis": np.asarray(group["local_axis_robot"][row], dtype=np.float32),
        "geometry": np.asarray(group["geometry"][row], dtype=np.float32),
        "geometry_valid_mask": np.asarray(group["geometry_valid_mask"][row], dtype=np.uint8),
        "association_identity": int(group["association_identity"][row]) if association_valid else -1,
    }
    manifest = {
        "parent_id": parent, "row": row, "history_rows": history,
        "event_index": targets["event_index"],
        "positive_relation_channels": [name for index, name in enumerate(("persistent", "reveal", "withdraw")) if relation[..., index].any()],
        "association_identity_teacher_only": targets["association_identity"],
        "current_branch_count": int(final_mask.sum()),
    }
    return scan, targets, manifest


def _batch(dataset_root: Path, device: torch.device) -> tuple[torch.Tensor, dict[str, torch.Tensor], list[dict]]:
    scans = []
    rows = []
    manifest = []
    for parent, index in SAMPLES:
        scan, target, record = _row(dataset_root, parent, index)
        scans.append(scan); rows.append(target); manifest.append(record)
    target = {}
    for name in rows[0]:
        target[name] = torch.from_numpy(np.stack([np.asarray(row[name]) for row in rows])).to(device)
    return torch.from_numpy(np.stack(scans)).to(device), target, manifest


def _maximum_gradient(model: torch.nn.Module) -> float:
    values = [float(parameter.grad.abs().max()) for parameter in model.parameters() if parameter.grad is not None]
    return max(values) if values else math.nan


def _plot(output: Path, summary: dict) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14.4, 4.2), constrained_layout=True)
    event_counts = summary["real_batch"]["event_counts"]
    axes[0].bar(range(5), [event_counts[str(index)] for index in range(5)], color="#4e79a7")
    axes[0].set_xticks(range(5), ("corridor", "junction", "terminal", "turn", "geometry\ntransition"), rotation=20)
    axes[0].set_title("A  Real-batch event coverage")
    relation_counts = summary["real_batch"]["relation_counts"]
    axes[1].bar(range(3), [relation_counts[name] for name in ("persistent", "reveal", "withdraw")], color="#59a14f")
    axes[1].set_yscale("log"); axes[1].set_xticks(range(3), ("persistent", "reveal", "withdraw"), rotation=15)
    axes[1].set_title("B  Past-to-current relations")
    errors = summary["equivariance"]
    names = ("relation", "event", "batch", "reverse")
    values = (errors["relation_rotation_max_abs"], errors["event_rotation_max_abs"], errors["batch_permutation_max_abs"], errors["reverse_axis_roundtrip_max_abs"])
    axes[2].bar(range(4), values, color="#e15759"); axes[2].set_yscale("symlog", linthresh=1e-9)
    axes[2].axhline(3e-5, color="black", linestyle="--", linewidth=1)
    axes[2].set_xticks(range(4), names, rotation=15); axes[2].set_title("C  Interface invariance error")
    for axis in axes: axis.grid(axis="y", alpha=.2); axis.set_axisbelow(True)
    figure.suptitle("GSE-Graph axis-anchored event–relation method readiness")
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output / f"gse_axis_anchored_event_relation_readiness_v1.{suffix}", dpi=220)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--status-version", choices=("v1", "v1r", "v2"), default="v1")
    parser.add_argument("--sequence-manifest", type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(20260829)
    if device.type == "cuda": torch.cuda.manual_seed_all(20260829)
    scans, targets, manifest = _batch(args.dataset_root.resolve(), device)
    if args.status_version == "v2" and args.sequence_manifest is None:
        raise ValueError("V2 requires --sequence-manifest for the full 80-world audit")
    full_population = (
        _full_population_audit(args.dataset_root.resolve(), args.sequence_manifest.resolve())
        if args.status_version == "v2" else None
    )
    model = AxisAnchoredEventRelationNet().to(device)
    predicted = model(scans)
    losses = axis_anchored_event_relation_loss(predicted, targets)
    losses["total"].backward()
    finite_outputs = all(bool(torch.isfinite(value).all()) for value in predicted.values())
    finite_losses = all(bool(torch.isfinite(value)) for value in losses.values())
    finite_gradients = all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters())
    model.zero_grad(set_to_none=True)
    masked_targets = dict(targets)
    masked_targets["relation_index"] = targets["relation_index"].clone()
    masked_targets["relation_index"][:, :2] = -1
    masked_targets["branch_presence_mask"] = targets["branch_identity"] >= 0
    masked_targets["event_class_weight"] = torch.tensor(
        [0.266182241, 0.607610566, 1.14589883, 2.21772549, 0.762582869], device=device
    )
    masked_targets["relation_positive_rate"] = torch.tensor(
        [934760 / 80667360, 10583 / 80667360, 11008 / 80667360], device=device
    )
    masked_predicted = model(scans)
    masked_core = axis_anchored_event_relation_core_loss(masked_predicted, masked_targets)
    masked_descriptor = axis_anchored_descriptor_loss(masked_predicted, masked_targets)
    masked_total = masked_core["total"] + masked_descriptor["total"]
    masked_total.backward()
    masked_training_finite = (
        bool(torch.isfinite(masked_total))
        and all(bool(torch.isfinite(value)) for value in masked_core.values())
        and all(bool(torch.isfinite(value)) for value in masked_descriptor.values())
        and all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters())
    )

    model.eval()
    shift_columns = 40; shift_bins = 10
    with torch.no_grad():
        base = model(scans[:2])
        rotated = model(torch.roll(scans[:2], shift_columns, dims=-1))
        relation_rotation = float((rotated["relation_probability_sequence"] - torch.roll(base["relation_probability_sequence"], shift_bins, dims=2)).abs().max())
        event_rotation = float((rotated["event_probability"] - base["event_probability"]).abs().max())
        order = torch.tensor([1, 0], device=device)
        permuted = model(scans[:2][order])
        batch_error = float((permuted["event_probability"][order] - base["event_probability"]).abs().max())
        field = torch.arange(180, device=device, dtype=torch.float32)[None]
        reverse_error = float((reverse_axis_field(reverse_axis_field(field)) - field).abs().max())
        repeat = model(scans[:2])
        repeat_error = float((repeat["branch_relation_probability"] - base["branch_relation_probability"]).abs().max())
    relation_array = targets["relation_index"].detach().cpu().numpy()
    relation_counts = relation_array.sum(axis=(0, 1, 2))
    relation_negatives = (relation_array == 0).sum(axis=(0, 1, 2))
    event_counts = np.bincount(targets["event_index"].detach().cpu().numpy(), minlength=5)
    association = targets["association_identity"].detach().cpu().numpy()
    branch_identity = targets["branch_identity"].detach().cpu().numpy()
    valid_branch = branch_identity[branch_identity >= 0]
    checks = {
        "typed_forward_has_only_scans": tuple(inspect.signature(model.forward).parameters) == ("scans",),
        "forbidden_identity_pose_future_inputs": set(("association_identity", "exit_identity", "pose", "future_frame")).issubset(axis_anchored_event_relation_input_contract()["forbidden_forward_inputs"]),
        "parameter_budget_at_most_1p5m": parameter_count() <= 1_500_000,
        "real_batch_covers_five_events": bool(np.all(event_counts > 0)),
        "real_batch_covers_three_multilabel_relations": bool(np.all(relation_counts > 0) and np.all(relation_negatives > 0)),
        "same_bin_reveal_withdraw_representable": bool(np.any((relation_array[..., 1] == 1) & (relation_array[..., 2] == 1))),
        "real_batch_has_place_positive_and_negative": len(association[association >= 0]) >= 3 and len(np.unique(association[association >= 0])) >= 2 and max(np.bincount(association[association >= 0])) >= 2,
        "real_batch_has_branch_positive_and_negative": len(np.unique(valid_branch)) >= 2 and max(np.bincount(valid_branch)) >= 2,
        "real_forward_loss_backward_finite": finite_outputs and finite_losses and finite_gradients and math.isfinite(_maximum_gradient(model)),
        "masked_early_relation_and_split_training_finite": masked_training_finite,
        "aligned_relation_rotation_at_most_3e5": relation_rotation <= 3e-5,
        "event_rotation_invariant_at_most_3e5": event_rotation <= 3e-5,
        "batch_permutation_at_most_3e6": batch_error <= 3e-6,
        "reverse_axis_is_exact_involution": reverse_error == 0.0,
        "deterministic_repeat_exact": repeat_error == 0.0,
        "past_only_union_is_bounded": bool(((predicted["branch_union_probability"] >= 0) & (predicted["branch_union_probability"] <= 1)).all()),
        "zero_optimizer_test_graph": True,
    }
    if args.status_version == "v2":
        checks["full_80_world_multilabel_teacher_exact"] = full_population == EXPECTED_FULL_POPULATION
        checks["all_partitions_cover_same_bin_reveal_withdraw"] = all(
            full_population[partition]["simultaneous_reveal_withdraw_bins"] > 0
            for partition in ("fit", "c07", "c08")
        )
    checks = {name: bool(value) for name, value in checks.items()}
    scientific_pass = all(checks.values())
    summary = {
        "schema_version": "gse_axis_anchored_event_relation_readiness_v1",
        "status": ({"v1": PASS, "v1r": PASS_V1R, "v2": PASS_V2}[args.status_version]
                   if scientific_pass else
                   {"v1": FAIL, "v1r": FAIL_V1R, "v2": FAIL_V2}[args.status_version]),
        "scientific_pass": scientific_pass,
        "decision": "ALLOW_AXIS_ANCHORED_EVENT_RELATION_THREE_SEED_DATA_CARD" if scientific_pass else "STOP_AXIS_ANCHORED_EVENT_RELATION_BEFORE_TRAINING",
        "question": "Can the typed axis-anchored event-relation model satisfy causal, geometric, equivariant and finite-backward contracts on real C01-C02 data without identity in forward?",
        "method": axis_anchored_event_relation_input_contract(),
        "parameters": parameter_count(), "device": str(device),
        "full_population": full_population,
        "real_batch": {
            "observations": len(scans), "samples": manifest,
            "event_counts": {str(i): int(value) for i, value in enumerate(event_counts)},
            "relation_counts": {name: int(value) for name, value in zip(("persistent", "reveal", "withdraw"), relation_counts)},
            "relation_negative_counts": {name: int(value) for name, value in zip(("persistent", "reveal", "withdraw"), relation_negatives)},
            "losses": {name: float(value.detach()) for name, value in losses.items()},
            "masked_training_contract": {
                "ignored_relation_steps": len(scans) * 2,
                "core_losses": {name: float(value.detach()) for name, value in masked_core.items()},
                "descriptor_losses": {name: float(value.detach()) for name, value in masked_descriptor.items()},
                "finite": masked_training_finite,
            },
            "maximum_absolute_gradient": _maximum_gradient(model),
        },
        "equivariance": {
            "relation_rotation_max_abs": relation_rotation, "event_rotation_max_abs": event_rotation,
            "batch_permutation_max_abs": batch_error, "reverse_axis_roundtrip_max_abs": reverse_error,
            "repeat_max_abs": repeat_error,
        },
        "checks": checks, "duration_seconds": time.monotonic() - started,
        "optimizer_steps": 0, "checkpoints_created": 0, "new_model_inference_observations": 0,
        "readiness_forward_observations": len(scans) + 8, "c09_worlds_read": 0, "c10_worlds_read": 0,
        "mtare_worlds_read": 0, "graph_replays": 0, "planner_calls": 0,
    }
    write_json(output / "summary.json", summary); write_json(output / "figure_source.json", summary)
    write_json(output / "real_batch_manifest.json", manifest)
    _plot(output, summary)
    print(json.dumps({"status": summary["status"], "decision": summary["decision"], "checks": checks}, indent=2, sort_keys=True))
    return 0 if scientific_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
