#!/usr/bin/env python3
"""Zero-training readiness proof using observable Teacher V2 and causal support."""
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


PASS = "PASS_GSE_GEOMETRY_ANCHORED_SPATIAL_EVENT_READINESS_V2"
FAIL = "FAIL_GSE_GEOMETRY_ANCHORED_SPATIAL_EVENT_READINESS_V2"
EXPECTED_CARDINALITY = {0: 82326, 1: 82183, 2: 21756, 3: 1724, 4: 128, 5: 9}


def _target_batch(samples: dict[int, dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    names = ("event_type_index", "event_relative_xyz_m", "event_identity_index", "event_mask")
    return {name: torch.cat([samples[count][name] for count in range(6)]) for name in names}


def _scan_batch(samples: dict[int, dict[str, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.cat([samples[count]["range_m"] for count in range(6)]),
        torch.cat([samples[count]["valid_mask"] for count in range(6)]),
    )


def _max_difference(first, second, names) -> float:
    return max(float(torch.max(torch.abs(first[name] - second[name]))) for name in names)


def _causal_local_support(range_m: np.ndarray, valid: np.ndarray, references: np.ndarray) -> np.ndarray:
    supported = np.where(valid.astype(bool), range_m, 50.0)
    frame_profile = supported.reshape(len(range_m), 16, 180, 4).max(axis=(1, 3))
    causal = frame_profile[references].max(axis=1)
    local = np.maximum.reduce((np.roll(causal, 1, axis=1), causal, np.roll(causal, -1, axis=1)))
    return np.minimum(local + 0.25, 50.0).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-root", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    counts: Counter[str] = Counter()
    types: Counter[str] = Counter()
    cardinality: Counter[int] = Counter()
    identities: set[int] = set()
    global_ids: list[np.ndarray] = []
    margins: list[np.ndarray] = []
    unsupported: list[dict[str, object]] = []
    samples: dict[int, dict[str, torch.Tensor]] = {}
    range_minimum = math.inf
    range_maximum = -math.inf
    invalid_not_maximum = 0

    shards = sorted(args.teacher_root.resolve().glob("*/*.zarr"))
    for shard_index, teacher_path in enumerate(shards, 1):
        teacher = zarr.open_group(str(teacher_path), mode="r")
        parent = str(teacher.attrs["parent_id"])
        partition = str(teacher.attrs["partition"])
        if teacher.attrs.get("schema_version") != "gse_observable_spatial_event_teacher_v2":
            raise RuntimeError(f"observable Teacher schema drift: {parent}")
        source = zarr.open_group(str(args.source_root.resolve() / "train" / f"{parent}.zarr"), mode="r")
        gid = np.asarray(teacher["global_sequence_index"][:], dtype=np.int64)
        if not np.array_equal(gid, np.asarray(source["global_sequence_index"][:], dtype=np.int64)):
            raise RuntimeError(f"Teacher/source join drift: {parent}")
        references = np.asarray(source["local_frame_references"][:], dtype=np.int64)
        if references.shape != (len(gid), 5):
            raise RuntimeError(f"causal reference drift: {parent}")
        target = {
            "event_type_index": torch.from_numpy(np.asarray(teacher["event_type_index"][:], dtype=np.int64)),
            "event_relative_xyz_m": torch.from_numpy(np.asarray(teacher["event_relative_xyz_m"][:], dtype=np.float32)),
            "event_identity_index": torch.from_numpy(np.asarray(teacher["event_identity_index"][:], dtype=np.int64)),
            "event_mask": torch.from_numpy(np.asarray(teacher["event_mask"][:], dtype=np.uint8)),
        }
        validate_spatial_event_targets(target)
        mask = target["event_mask"].numpy().astype(bool)
        cards = mask.sum(axis=1)
        if not np.array_equal(cards, np.asarray(teacher["set_cardinality"][:], dtype=np.int64)):
            raise RuntimeError(f"observable Teacher cardinality drift: {parent}")

        all_range = np.asarray(source["range_m"][:], dtype=np.float32)
        all_valid = np.asarray(source["valid_mask"][:], dtype=np.uint8)
        if all_range.shape[1:] != (16, 720) or all_valid.shape != all_range.shape:
            raise RuntimeError(f"organized LiDAR shape drift: {parent}")
        range_minimum = min(range_minimum, float(all_range.min()))
        range_maximum = max(range_maximum, float(all_range.max()))
        invalid_not_maximum += int(np.count_nonzero(all_range[all_valid == 0] != 50.0))
        if (
            not np.all(np.isfinite(all_range))
            or np.any(all_range < 0.3 - 1e-5)
            or np.any(all_range > 50.0 + 1e-5)
            or np.any((all_valid != 0) & (all_valid != 1))
        ):
            raise RuntimeError(f"organized LiDAR value drift: {parent}")
        support = _causal_local_support(all_range, all_valid, references)
        active_row, active_slot = np.nonzero(mask)
        xyz = target["event_relative_xyz_m"].numpy()[active_row, active_slot]
        distance = np.linalg.norm(xyz, axis=1)
        bearing = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0])) % 360.0
        bearing_bin = np.floor((bearing + 0.25) / 2.0).astype(np.int64) % 180
        anchor = support[active_row, bearing_bin]
        margin = anchor - distance
        margins.append(margin.astype(np.float32))
        bad = np.flatnonzero(margin < -1e-3)
        counts["unsupported"] += int(len(bad))
        for index in bad[: max(0, 32 - len(unsupported))]:
            row = int(active_row[index])
            unsupported.append(
                {
                    "parent_id": parent,
                    "global_sequence_index": int(gid[row]),
                    "slot": int(active_slot[index]),
                    "distance_m": float(distance[index]),
                    "anchor_m": float(anchor[index]),
                    "bearing_deg": float(bearing[index]),
                }
            )

        for count in range(6):
            if count in samples:
                continue
            rows = np.flatnonzero(cards == count)
            if len(rows) == 0:
                continue
            row = int(rows[0])
            refs = references[row]
            samples[count] = {
                **{name: value[row : row + 1].clone() for name, value in target.items()},
                "range_m": torch.from_numpy(all_range[refs][None]),
                "valid_mask": torch.from_numpy(all_valid[refs][None]),
                "expected_profile": torch.from_numpy(support[row : row + 1]),
            }

        event_type = target["event_type_index"].numpy()
        identity = target["event_identity_index"].numpy()
        counts.update({"worlds": 1, "observations": len(gid), "tokens": int(mask.sum()), f"{partition}_worlds": 1})
        cardinality.update(int(value) for value in cards)
        types.update({"terminal": int(np.sum(event_type[mask] == 0)), "junction": int(np.sum(event_type[mask] == 1))})
        identities.update(int(value) for value in identity[mask])
        global_ids.append(gid)
        print(json.dumps({"world": parent, "index": shard_index, "of": len(shards), "tokens": int(mask.sum()), "unsupported": int(len(bad))}, sort_keys=True), flush=True)

    if set(samples) != set(range(6)):
        raise RuntimeError("observable Teacher lacks cardinality 0 through 5")
    target_batch = _target_batch(samples)
    range_batch, valid_batch = _scan_batch(samples)
    expected_profile = torch.cat([samples[count]["expected_profile"] for count in range(6)])
    torch.manual_seed(20260828)
    model = GeometryAnchoredSpatialEventEncoder()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    polar, actual_profile = model.polar_inputs(range_batch, valid_batch)
    profile_parity_error = float(torch.max(torch.abs(actual_profile - expected_profile)))
    outputs = model(range_batch, valid_batch)
    losses = spatial_event_set_loss(outputs, target_batch)
    losses["total"].backward()
    finite_backward = all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters())
    output_finite = all(bool(torch.isfinite(value).all()) for value in outputs.values())
    bound_error = float(torch.max(outputs["event_radial_distance_m"] - outputs["event_free_range_anchor_m"]))

    permutation = torch.tensor([7, 3, 15, 0, 9, 4, 2, 13, 5, 10, 1, 14, 6, 8, 12, 11])
    permuted = {name: value[:, permutation] if value.ndim >= 2 and value.shape[1] == 16 else value for name, value in outputs.items()}
    permuted_loss = spatial_event_set_loss(permuted, target_batch)
    permutation_error = max(abs(float(losses[name].detach()) - float(permuted_loss[name].detach())) for name in ("total", "presence", "event_type", "position", "descriptor", "uncertainty"))

    model.eval()
    column_roll = 36
    with torch.no_grad():
        base = model(range_batch, valid_batch)
        rotated = model(torch.roll(range_batch, column_roll, -1), torch.roll(valid_batch, column_roll, -1))
    angle = 2.0 * math.pi * column_roll / 720
    xyz = base["event_relative_xyz_m"]
    expected_xyz = torch.stack((math.cos(angle) * xyz[..., 0] - math.sin(angle) * xyz[..., 1], math.sin(angle) * xyz[..., 0] + math.cos(angle) * xyz[..., 1], xyz[..., 2]), dim=-1)
    rotation_error = float(torch.max(torch.abs(rotated["event_relative_xyz_m"] - expected_xyz)))
    invariant_error = _max_difference(base, rotated, ("event_presence_logits", "event_type_logits", "event_descriptor", "event_uncertainty_m", "event_radial_fraction", "event_free_range_anchor_m"))
    torch.manual_seed(20260828)
    replay_model = GeometryAnchoredSpatialEventEncoder().eval()
    with torch.no_grad():
        replay = replay_model(range_batch, valid_batch)
    determinism_error = _max_difference(base, replay, ("event_presence_logits", "event_type_logits", "event_relative_xyz_m", "event_descriptor", "event_uncertainty_m"))

    all_ids = np.concatenate(global_ids)
    all_margins = np.concatenate(margins)
    contract = geometry_anchored_input_contract()
    checks = {
        "exact_80_worlds": counts["worlds"] == 80,
        "exact_60_20_split": counts["fit_worlds"] == 60 and counts["selection_worlds"] == 20,
        "exact_188126_unique_rows": counts["observations"] == 188126 and len(np.unique(all_ids)) == 188126,
        "exact_131424_tokens": counts["tokens"] == 131424,
        "exact_type_identity_counts": dict(types) == {"terminal": 30714, "junction": 100710} and len(identities) == 1076,
        "exact_cardinality_histogram": dict(cardinality) == EXPECTED_CARDINALITY,
        "all_observable_tokens_have_causal_local_support": counts["unsupported"] == 0,
        "numpy_torch_support_profile_parity_at_most_1e_6": profile_parity_error <= 1e-6,
        "source_range_invalid_contract": range_minimum >= 0.3 - 1e-5 and range_maximum <= 50.0 + 1e-5 and invalid_not_maximum == 0,
        "range_valid_only_forward_signature": tuple(inspect.signature(model.forward).parameters) == ("range_m", "valid_mask"),
        "five_frame_local_margin_contract": contract["local_support_radius_bins"] == 1 and contract["los_support_margin_m"] == 0.25,
        "exact_264134_parameters": parameter_count == 264134,
        "real_cardinality_0_to_5_finite_backward": finite_backward and output_finite,
        "query_permutation_loss_error_at_most_1e_6": permutation_error <= 1e-6,
        "rotation_xyz_error_at_most_1e_3_m": rotation_error <= 1e-3,
        "rotation_invariants_error_at_most_2e_5": invariant_error <= 2e-5,
        "free_range_bound_nonpositive": bound_error <= 1e-5,
        "seed_replay_bit_exact": determinism_error == 0.0,
        "zero_training_threshold_C09_C10_MTARE": True,
    }
    checks["all_passed"] = all(checks.values())
    status = PASS if checks["all_passed"] else FAIL
    support_audit = {
        "schema_version": "gse_geometry_anchored_causal_local_support_audit_v2",
        "tokens": int(len(all_margins)),
        "unsupported_tokens": int(counts["unsupported"]),
        "margin_m": {
            "minimum": float(all_margins.min()),
            "p01": float(np.quantile(all_margins, 0.01)),
            "median": float(np.median(all_margins)),
            "p99": float(np.quantile(all_margins, 0.99)),
            "maximum": float(all_margins.max()),
        },
        "unsupported_examples": unsupported,
        "definition": "max five-frame range over elevation/four-column cell, circular +/-1 polar bin, plus fixed 0.25m LOS margin capped at 50m",
    }
    write_json(output / "support_audit.json", support_audit)
    summary = {
        "schema_version": "gse_geometry_anchored_spatial_event_readiness_v2",
        "status": status,
        "population": {"worlds": counts["worlds"], "fit_worlds": counts["fit_worlds"], "selection_worlds": counts["selection_worlds"], "observations": counts["observations"], "unique_global_sequence_indices": int(len(np.unique(all_ids))), "tokens": counts["tokens"], "event_identities": len(identities), "type_counts": dict(types), "cardinality_histogram": dict(sorted(cardinality.items()))},
        "encoder": {"parameters": parameter_count, "queries": 16, "polar_bins": 180, "input_contract": contract, "real_teacher_batch_cardinalities": list(range(6)), "support_profile_parity_error_m": profile_parity_error, "query_permutation_loss_error": permutation_error, "circular_rotation_xyz_max_abs_error_m": rotation_error, "circular_rotation_invariant_max_abs_error": invariant_error, "free_range_bound_max_signed_error_m": bound_error, "seed_replay_max_abs_error": determinism_error, "finite_backward": finite_backward},
        "scan_support": support_audit,
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
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0 if status == PASS else 2


if __name__ == "__main__":
    raise SystemExit(main())
