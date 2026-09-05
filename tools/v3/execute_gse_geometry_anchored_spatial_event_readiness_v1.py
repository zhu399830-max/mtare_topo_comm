#!/usr/bin/env python3
"""Full zero-training readiness proof for the geometry-anchored event encoder."""
from __future__ import annotations

import argparse
from collections import Counter
import inspect
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import zarr

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import write_json
from mtare_topo.representation.gse_geometry_anchored_event import (
    GeometryAnchoredSpatialEventEncoder,
    geometry_anchored_input_contract,
)
from mtare_topo.representation.gse_spatial_event_set import (
    spatial_event_set_loss,
    validate_spatial_event_targets,
)


PASS = "PASS_GSE_GEOMETRY_ANCHORED_SPATIAL_EVENT_READINESS_V1"
FAIL = "FAIL_GSE_GEOMETRY_ANCHORED_SPATIAL_EVENT_READINESS_V1"
EXPECTED = {
    "worlds": 80,
    "fit_worlds": 60,
    "selection_worlds": 20,
    "observations": 188126,
    "tokens": 133055,
    "terminal": 30789,
    "junction": 102266,
    "identities": 1076,
    "parameters": 264134,
}


def _target_batch(samples: dict[int, dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        name: torch.cat([samples[count][name] for count in range(6)], dim=0)
        for name in samples[0]
    }


def _scan_batch(samples: dict[int, dict[str, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.cat([samples[count]["range_m"] for count in range(6)], dim=0),
        torch.cat([samples[count]["valid_mask"] for count in range(6)], dim=0),
    )


def _maximum_difference(
    first: dict[str, torch.Tensor], second: dict[str, torch.Tensor], names: tuple[str, ...]
) -> float:
    return max(float(torch.max(torch.abs(first[name] - second[name]))) for name in names)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-root", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    teacher_shards = sorted(args.teacher_root.resolve().glob("*/*.zarr"))
    counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    cardinality: Counter[int] = Counter()
    identity_values: set[int] = set()
    global_ids: list[np.ndarray] = []
    samples: dict[int, dict[str, torch.Tensor]] = {}
    unsupported: list[dict[str, object]] = []
    support_margins: list[np.ndarray] = []
    range_minimum = math.inf
    range_maximum = -math.inf
    invalid_not_maximum = 0

    for shard_index, teacher_path in enumerate(teacher_shards, 1):
        teacher = zarr.open_group(str(teacher_path), mode="r")
        parent = str(teacher.attrs["parent_id"])
        partition = str(teacher.attrs["partition"])
        source_path = args.source_root.resolve() / "train" / f"{parent}.zarr"
        source = zarr.open_group(str(source_path), mode="r")
        teacher_gid = np.asarray(teacher["global_sequence_index"][:], dtype=np.int64)
        source_gid = np.asarray(source["global_sequence_index"][:], dtype=np.int64)
        references = np.asarray(source["local_frame_references"][:], dtype=np.int64)
        if not np.array_equal(teacher_gid, source_gid):
            raise RuntimeError(f"Teacher/source global identity drift: {parent}")
        if references.shape != (len(teacher_gid), 5):
            raise RuntimeError(f"five-frame reference drift: {parent}")

        target = {
            "event_type_index": torch.from_numpy(
                np.asarray(teacher["event_type_index"][:], dtype=np.int64)
            ),
            "event_relative_xyz_m": torch.from_numpy(
                np.asarray(teacher["event_relative_xyz_m"][:], dtype=np.float32)
            ),
            "event_identity_index": torch.from_numpy(
                np.asarray(teacher["event_identity_index"][:], dtype=np.int64)
            ),
            "event_mask": torch.from_numpy(
                np.asarray(teacher["event_mask"][:], dtype=np.uint8)
            ),
        }
        validate_spatial_event_targets(target)
        mask = target["event_mask"].numpy().astype(bool)
        cards = mask.sum(axis=1)
        types = target["event_type_index"].numpy()
        identities = target["event_identity_index"].numpy()
        relative = target["event_relative_xyz_m"].numpy()
        if not np.array_equal(
            cards, np.asarray(teacher["set_cardinality"][:], dtype=np.int64)
        ):
            raise RuntimeError(f"Teacher cardinality drift: {parent}")

        current_indices = references[:, -1]
        current_range = np.asarray(source["range_m"].oindex[current_indices], dtype=np.float32)
        current_valid = np.asarray(source["valid_mask"].oindex[current_indices], dtype=np.uint8)
        if current_range.shape != (len(cards), 16, 720) or current_valid.shape != current_range.shape:
            raise RuntimeError(f"organized LiDAR shape drift: {parent}")
        range_minimum = min(range_minimum, float(current_range.min()))
        range_maximum = max(range_maximum, float(current_range.max()))
        invalid_not_maximum += int(np.count_nonzero(current_range[current_valid == 0] != 50.0))
        if (
            not np.all(np.isfinite(current_range))
            or np.any(current_range < 0.3 - 1e-5)
            or np.any(current_range > 50.0 + 1e-5)
            or np.any((current_valid != 0) & (current_valid != 1))
        ):
            raise RuntimeError(f"organized LiDAR value drift: {parent}")

        supported_range = np.where(current_valid.astype(bool), current_range, 50.0)
        free_profile = supported_range.reshape(len(cards), 16, 180, 4).max(axis=(1, 3))
        active_row, active_slot = np.nonzero(mask)
        active_xyz = relative[active_row, active_slot]
        distance = np.linalg.norm(active_xyz, axis=1)
        bearing_deg = np.degrees(np.arctan2(active_xyz[:, 1], active_xyz[:, 0])) % 360.0
        # Learned bin zero averages raw columns 0..3 and is centered at 0.75 deg.
        bearing_bin = np.floor((bearing_deg + 0.25) / 2.0).astype(np.int64) % 180
        anchor = free_profile[active_row, bearing_bin]
        margin = anchor - distance
        support_margins.append(margin.astype(np.float32))
        bad = np.flatnonzero(margin < -1e-3)
        for local_index in bad[: max(0, 32 - len(unsupported))]:
            row = int(active_row[local_index])
            slot = int(active_slot[local_index])
            unsupported.append(
                {
                    "parent_id": parent,
                    "global_sequence_index": int(teacher_gid[row]),
                    "slot": slot,
                    "distance_m": float(distance[local_index]),
                    "free_range_anchor_m": float(anchor[local_index]),
                    "bearing_deg": float(bearing_deg[local_index]),
                }
            )
        counts["unsupported_tokens"] += int(len(bad))

        for count in range(6):
            if count in samples:
                continue
            rows = np.flatnonzero(cards == count)
            if len(rows) == 0:
                continue
            row = int(rows[0])
            refs = references[row]
            sample_range = np.asarray(source["range_m"].oindex[refs], dtype=np.float32)
            sample_valid = np.asarray(source["valid_mask"].oindex[refs], dtype=np.uint8)
            samples[count] = {
                **{name: value[row : row + 1].clone() for name, value in target.items()},
                "range_m": torch.from_numpy(sample_range[None]),
                "valid_mask": torch.from_numpy(sample_valid[None]),
            }

        counts.update(
            {
                "worlds": 1,
                "observations": len(cards),
                "tokens": int(mask.sum()),
                f"{partition}_worlds": 1,
            }
        )
        cardinality.update(int(value) for value in cards)
        type_counts.update(
            {
                "terminal": int(np.sum(types[mask] == 0)),
                "junction": int(np.sum(types[mask] == 1)),
            }
        )
        identity_values.update(int(value) for value in identities[mask])
        global_ids.append(teacher_gid)
        print(
            json.dumps(
                {
                    "world": parent,
                    "index": shard_index,
                    "of": len(teacher_shards),
                    "observations": len(cards),
                    "tokens": int(mask.sum()),
                    "unsupported": int(len(bad)),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if set(samples) != set(range(6)):
        raise RuntimeError("real Teacher lacks cardinality 0 through 5")
    target_batch = _target_batch(samples)
    range_batch, valid_batch = _scan_batch(samples)

    torch.manual_seed(20260828)
    model = GeometryAnchoredSpatialEventEncoder()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    outputs = model(range_batch, valid_batch)
    losses = spatial_event_set_loss(outputs, target_batch)
    losses["total"].backward()
    finite_backward = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    all_output_finite = all(bool(torch.isfinite(value).all()) for value in outputs.values())
    physical_bound_error = float(
        torch.max(outputs["event_radial_distance_m"] - outputs["event_free_range_anchor_m"])
    )

    permutation = torch.tensor([7, 3, 15, 0, 9, 4, 2, 13, 5, 10, 1, 14, 6, 8, 12, 11])
    permuted = {
        name: value[:, permutation]
        if value.ndim >= 2 and value.shape[1] == 16
        else value
        for name, value in outputs.items()
    }
    permuted_loss = spatial_event_set_loss(permuted, target_batch)
    permutation_error = max(
        abs(float(losses[name].detach()) - float(permuted_loss[name].detach()))
        for name in ("total", "presence", "event_type", "position", "descriptor", "uncertainty")
    )

    model.eval()
    column_roll = 36
    with torch.no_grad():
        base = model(range_batch, valid_batch)
        rotated = model(
            torch.roll(range_batch, shifts=column_roll, dims=-1),
            torch.roll(valid_batch, shifts=column_roll, dims=-1),
        )
    angle = 2.0 * math.pi * column_roll / 720
    xyz = base["event_relative_xyz_m"]
    expected_xyz = torch.stack(
        (
            math.cos(angle) * xyz[..., 0] - math.sin(angle) * xyz[..., 1],
            math.sin(angle) * xyz[..., 0] + math.cos(angle) * xyz[..., 1],
            xyz[..., 2],
        ),
        dim=-1,
    )
    rotation_error = float(torch.max(torch.abs(rotated["event_relative_xyz_m"] - expected_xyz)))
    invariant_rotation_error = _maximum_difference(
        base,
        rotated,
        (
            "event_presence_logits",
            "event_type_logits",
            "event_descriptor",
            "event_uncertainty_m",
            "event_radial_fraction",
            "event_free_range_anchor_m",
        ),
    )

    torch.manual_seed(20260828)
    replay_model = GeometryAnchoredSpatialEventEncoder().eval()
    with torch.no_grad():
        replay = replay_model(range_batch, valid_batch)
    determinism_error = _maximum_difference(
        base,
        replay,
        (
            "event_presence_logits",
            "event_type_logits",
            "event_relative_xyz_m",
            "event_descriptor",
            "event_uncertainty_m",
        ),
    )

    all_global_ids = np.concatenate(global_ids)
    margins = np.concatenate(support_margins)
    contract = geometry_anchored_input_contract()
    forward_parameters = tuple(inspect.signature(model.forward).parameters)
    expected_cardinality = {0: 81069, 1: 83093, 2: 22076, 3: 1751, 4: 128, 5: 9}
    checks = {
        "exact_80_worlds": counts["worlds"] == EXPECTED["worlds"],
        "exact_60_20_split": counts["fit_worlds"] == 60 and counts["selection_worlds"] == 20,
        "exact_188126_unique_observations": counts["observations"] == 188126
        and len(np.unique(all_global_ids)) == 188126,
        "exact_133055_tokens": counts["tokens"] == 133055,
        "exact_type_and_identity_counts": dict(type_counts) == {"terminal": 30789, "junction": 102266}
        and len(identity_values) == 1076,
        "exact_cardinality_histogram": dict(cardinality) == expected_cardinality,
        "all_teacher_tokens_have_scan_free_range_support": counts["unsupported_tokens"] == 0,
        "source_range_and_invalid_contract": range_minimum >= 0.3 - 1e-5
        and range_maximum <= 50.0 + 1e-5
        and invalid_not_maximum == 0,
        "range_valid_only_forward_signature": forward_parameters == ("range_m", "valid_mask"),
        "forbidden_inputs_declared": set(contract["forbidden_inputs"])
        == {
            "world_pose",
            "sensor_pose",
            "route_identity",
            "traversal_identity",
            "TNG_identity",
            "future_frame",
            "teacher_feature",
        },
        "exact_264134_parameters": parameter_count == EXPECTED["parameters"],
        "real_cardinality_0_through_5_finite_backward": finite_backward and all_output_finite,
        "query_permutation_loss_error_at_most_1e_6": permutation_error <= 1e-6,
        "circular_rotation_xyz_error_at_most_1e_3_m": rotation_error <= 1e-3,
        "circular_rotation_invariants_error_at_most_2e_5": invariant_rotation_error <= 2e-5,
        "free_range_bound_error_nonpositive": physical_bound_error <= 1e-5,
        "seed_replay_bit_exact": determinism_error == 0.0,
        "zero_training_threshold_C09_C10_MTARE": True,
    }
    checks["all_passed"] = all(checks.values())
    status = PASS if checks["all_passed"] else FAIL
    support = {
        "schema_version": "gse_geometry_anchored_support_audit_v1",
        "tokens": int(len(margins)),
        "unsupported_tokens": int(counts["unsupported_tokens"]),
        "margin_m": {
            "minimum": float(margins.min()),
            "p01": float(np.quantile(margins, 0.01)),
            "median": float(np.median(margins)),
            "p99": float(np.quantile(margins, 0.99)),
            "maximum": float(margins.max()),
        },
        "unsupported_examples": unsupported,
        "definition": "Teacher event distance <= max current LiDAR support in its four-column polar cell.",
    }
    write_json(output_dir / "support_audit.json", support)
    summary = {
        "schema_version": "gse_geometry_anchored_spatial_event_readiness_v1",
        "status": status,
        "population": {
            "worlds": counts["worlds"],
            "fit_worlds": counts["fit_worlds"],
            "selection_worlds": counts["selection_worlds"],
            "observations": counts["observations"],
            "unique_global_sequence_indices": int(len(np.unique(all_global_ids))),
            "tokens": counts["tokens"],
            "event_identities": len(identity_values),
            "type_counts": dict(type_counts),
            "cardinality_histogram": dict(sorted(cardinality.items())),
        },
        "encoder": {
            "parameters": parameter_count,
            "queries": 16,
            "polar_bins": 180,
            "real_teacher_batch_cardinalities": list(range(6)),
            "input_contract": contract,
            "query_permutation_loss_error": permutation_error,
            "circular_rotation_xyz_max_abs_error_m": rotation_error,
            "circular_rotation_invariant_max_abs_error": invariant_rotation_error,
            "free_range_bound_max_signed_error_m": physical_bound_error,
            "seed_replay_max_abs_error": determinism_error,
            "finite_backward": finite_backward,
        },
        "scan_support": support,
        "checks": checks,
        "duration_seconds": time.monotonic() - started,
        "optimizer_steps": 0,
        "trained_model_inference_frames": 0,
        "threshold_selection_steps": 0,
        "random_initialization_forward_batches": 4,
        "c09_worlds_read": 0,
        "c10_worlds_read": 0,
        "mtare_worlds_read": 0,
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0 if status == PASS else 2


if __name__ == "__main__":
    raise SystemExit(main())
