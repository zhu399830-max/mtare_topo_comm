#!/usr/bin/env python3
"""Zero-training full-population readiness for the 180x2 event field."""
from __future__ import annotations

from collections import Counter
import argparse
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
from mtare_topo.representation.gse_spatial_event_set import validate_spatial_event_targets
from mtare_topo.representation.gse_structured_polar_multidepth_event import (
    StructuredPolarMultiDepthEventEncoder,
    rasterize_structured_polar_targets,
    structured_polar_multidepth_input_contract,
    structured_polar_multidepth_loss,
)


PASS = "PASS_GSE_STRUCTURED_POLAR_MULTIDEPTH_READINESS_V1"
FAIL = "FAIL_GSE_STRUCTURED_POLAR_MULTIDEPTH_READINESS_V1"
EXPECTED_CARDINALITY = {0: 82326, 1: 82183, 2: 21756, 3: 1724, 4: 128, 5: 9}


def _support(range_m: np.ndarray, valid: np.ndarray, references: np.ndarray) -> np.ndarray:
    supported = np.where(valid.astype(bool), range_m, 50.0)
    frame_profile = supported.reshape(len(range_m), 16, 180, 4).max(axis=(1, 3))
    causal = frame_profile[references].max(axis=1)
    local = np.maximum.reduce((np.roll(causal, 1, axis=1), causal, np.roll(causal, -1, axis=1)))
    return np.minimum(local + 0.25, 50.0).astype(np.float32)


def _target(group) -> dict[str, torch.Tensor]:
    result = {
        "event_type_index": torch.from_numpy(np.asarray(group["event_type_index"][:], dtype=np.int64)),
        "event_relative_xyz_m": torch.from_numpy(np.asarray(group["event_relative_xyz_m"][:], dtype=np.float32)),
        "event_identity_index": torch.from_numpy(np.asarray(group["event_identity_index"][:], dtype=np.int64)),
        "event_mask": torch.from_numpy(np.asarray(group["event_mask"][:], dtype=np.uint8)),
    }
    validate_spatial_event_targets(result)
    return result


def _reconstruct(dense, support: torch.Tensor) -> torch.Tensor:
    bins = torch.arange(180, dtype=support.dtype, device=support.device)
    center = (bins + 0.375) * (2.0 * torch.pi / 180.0)
    azimuth = center[None, :, None] + dense["event_azimuth_residual_rad"]
    radial = dense["event_radial_fraction"] * support[:, :, None]
    elevation = dense["event_elevation_rad"]
    horizontal = radial * torch.cos(elevation)
    return torch.stack((horizontal * torch.cos(azimuth), horizontal * torch.sin(azimuth), radial * torch.sin(elevation)), dim=-1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-root", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    teacher_paths = sorted(args.teacher_root.resolve().glob("*/*.zarr"))
    if len(teacher_paths) != 80:
        raise RuntimeError("structured polar readiness requires 80 Teacher shards")
    counts: Counter[str] = Counter()
    cardinality: Counter[int] = Counter()
    identities: set[int] = set()
    global_ids: list[np.ndarray] = []
    maximum_roundtrip_error = 0.0
    maximum_radial_fraction = 0.0
    maximum_azimuth_residual_deg = 0.0
    slot1_tokens = 0
    overflow_rows = 0
    profile_minimum = math.inf
    profile_maximum = -math.inf
    samples: dict[str, dict[str, torch.Tensor]] = {}
    per_world: list[dict[str, object]] = []

    for index, teacher_path in enumerate(teacher_paths, 1):
        teacher = zarr.open_group(str(teacher_path), mode="r")
        parent = str(teacher.attrs["parent_id"])
        partition = str(teacher.attrs["partition"])
        source = zarr.open_group(str(args.source_root.resolve() / "train" / f"{parent}.zarr"), mode="r")
        gid = np.asarray(teacher["global_sequence_index"][:], dtype=np.int64)
        if not np.array_equal(gid, np.asarray(source["global_sequence_index"][:], dtype=np.int64)):
            raise RuntimeError(f"structured polar Teacher/source join drift: {parent}")
        targets = _target(teacher)
        cards = targets["event_mask"].sum(dim=1).numpy().astype(np.int64)
        if not np.array_equal(cards, np.asarray(teacher["set_cardinality"][:], dtype=np.int64)):
            raise RuntimeError(f"structured polar cardinality drift: {parent}")
        references = np.asarray(source["local_frame_references"][:], dtype=np.int64)
        all_range = np.asarray(source["range_m"][:], dtype=np.float32)
        all_valid = np.asarray(source["valid_mask"][:], dtype=np.uint8)
        profile = _support(all_range, all_valid, references)
        profile_minimum = min(profile_minimum, float(profile.min()))
        profile_maximum = max(profile_maximum, float(profile.max()))
        try:
            dense = rasterize_structured_polar_targets(targets, torch.from_numpy(profile))
        except ValueError as exc:
            overflow_rows += 1
            raise RuntimeError(f"structured polar rasterizer failed: {parent}: {exc}") from exc
        mask = dense["event_mask"]
        token_count = int(mask.sum())
        expected_tokens = int(targets["event_mask"].sum())
        if token_count != expected_tokens:
            raise RuntimeError(f"structured polar token loss: {parent}")
        reconstructed = _reconstruct(dense, torch.from_numpy(profile))
        if bool(mask.any()):
            maximum_roundtrip_error = max(maximum_roundtrip_error, float(torch.max(torch.abs(reconstructed[mask] - dense["event_relative_xyz_m"][mask]))))
            maximum_radial_fraction = max(maximum_radial_fraction, float(dense["event_radial_fraction"][mask].max()))
            maximum_azimuth_residual_deg = max(maximum_azimuth_residual_deg, float(torch.rad2deg(torch.abs(dense["event_azimuth_residual_rad"][mask])).max()))
        slot1_tokens += int(mask[:, :, 1].sum())
        if "same_bin" not in samples:
            duplicate = torch.nonzero(mask[:, :, 1].any(dim=1), as_tuple=False).flatten()
            if len(duplicate):
                row = int(duplicate[0])
                refs = references[row]
                samples["same_bin"] = {
                    **{name: value[row : row + 1].clone() for name, value in targets.items()},
                    "range_m": torch.from_numpy(all_range[refs][None]),
                    "valid_mask": torch.from_numpy(all_valid[refs][None]),
                    "profile": torch.from_numpy(profile[row : row + 1]),
                }
        for count in range(6):
            key = f"cardinality_{count}"
            if key in samples:
                continue
            rows = np.flatnonzero(cards == count)
            if len(rows):
                row = int(rows[0])
                refs = references[row]
                samples[key] = {
                    **{name: value[row : row + 1].clone() for name, value in targets.items()},
                    "range_m": torch.from_numpy(all_range[refs][None]),
                    "valid_mask": torch.from_numpy(all_valid[refs][None]),
                    "profile": torch.from_numpy(profile[row : row + 1]),
                }
        active = targets["event_mask"].bool()
        identities.update(int(value) for value in targets["event_identity_index"][active])
        counts.update({"worlds": 1, "observations": len(gid), "tokens": expected_tokens, f"{partition}_worlds": 1, f"{partition}_tokens": expected_tokens})
        cardinality.update(int(value) for value in cards)
        global_ids.append(gid)
        per_world.append({"parent_id": parent, "partition": partition, "observations": len(gid), "tokens": expected_tokens, "slot1_tokens": int(mask[:, :, 1].sum())})
        print(json.dumps({"world": parent, "index": index, "of": 80, "tokens": expected_tokens, "slot1": int(mask[:, :, 1].sum())}, sort_keys=True), flush=True)

    if set(samples) != {"same_bin", *(f"cardinality_{value}" for value in range(6))}:
        raise RuntimeError("structured polar readiness sample coverage drift")
    ordered_samples = [samples[f"cardinality_{value}"] for value in range(6)] + [samples["same_bin"]]
    range_batch = torch.cat([sample["range_m"] for sample in ordered_samples])
    valid_batch = torch.cat([sample["valid_mask"] for sample in ordered_samples])
    target_batch = {name: torch.cat([sample[name] for sample in ordered_samples]) for name in ("event_type_index", "event_relative_xyz_m", "event_identity_index", "event_mask")}
    expected_profile = torch.cat([sample["profile"] for sample in ordered_samples])
    torch.manual_seed(20260828)
    model = StructuredPolarMultiDepthEventEncoder()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    polar, actual_profile = model.polar_inputs(range_batch, valid_batch)
    profile_parity_error = float(torch.max(torch.abs(actual_profile - expected_profile)))
    dense_targets = rasterize_structured_polar_targets(target_batch, actual_profile.detach())
    outputs = model(range_batch, valid_batch)
    losses = structured_polar_multidepth_loss(outputs, dense_targets, presence_positive_weight=10.0)
    losses["total"].backward()
    finite_backward = all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters())
    finite_outputs = all(bool(torch.isfinite(value).all()) for value in outputs.values() if torch.is_floating_point(value))
    range_bound_error = float(torch.max(outputs["dense_event_radial_distance_m"] - outputs["free_range_profile_m"][:, :, None]))
    azimuth_bound_error = float(torch.max(torch.abs(outputs["dense_event_azimuth_residual_rad"]))) - math.pi / 180.0
    topk_unique = all(len(torch.unique(row)) == 16 for row in (outputs["event_azimuth_bin_index"] * 2 + outputs["event_depth_slot_index"]))

    model.eval()
    shift_columns = 40
    shift_bins = 10
    angle = 2.0 * math.pi * shift_columns / 720
    with torch.no_grad():
        base = model(range_batch, valid_batch)
        rotated = model(torch.roll(range_batch, shift_columns, -1), torch.roll(valid_batch, shift_columns, -1))
        repeated = model(range_batch, valid_batch)
    dense_logit_rotation_error = float(torch.max(torch.abs(rotated["dense_event_presence_logits"] - torch.roll(base["dense_event_presence_logits"], shift_bins, 1))))
    dense_xyz = base["dense_event_relative_xyz_m"]
    expected_dense_xyz = torch.stack((math.cos(angle) * dense_xyz[..., 0] - math.sin(angle) * dense_xyz[..., 1], math.sin(angle) * dense_xyz[..., 0] + math.cos(angle) * dense_xyz[..., 1], dense_xyz[..., 2]), dim=-1)
    dense_xyz_rotation_error = float(torch.max(torch.abs(rotated["dense_event_relative_xyz_m"] - torch.roll(expected_dense_xyz, shift_bins, 1))))
    top_xyz = base["event_relative_xyz_m"]
    expected_top_xyz = torch.stack((math.cos(angle) * top_xyz[..., 0] - math.sin(angle) * top_xyz[..., 1], math.sin(angle) * top_xyz[..., 0] + math.cos(angle) * top_xyz[..., 1], top_xyz[..., 2]), dim=-1)
    top_xyz_rotation_error = float(torch.max(torch.abs(rotated["event_relative_xyz_m"] - expected_top_xyz)))
    top_index_rotation_exact = bool(torch.equal(rotated["event_azimuth_bin_index"], (base["event_azimuth_bin_index"] + shift_bins) % 180) and torch.equal(rotated["event_depth_slot_index"], base["event_depth_slot_index"]))
    repeat_exact = all(torch.equal(base[name], repeated[name]) for name in ("dense_event_presence_logits", "dense_event_relative_xyz_m", "event_confidence", "event_relative_xyz_m", "event_azimuth_bin_index", "event_depth_slot_index"))
    torch.manual_seed(20260828)
    replay_model = StructuredPolarMultiDepthEventEncoder().eval()
    with torch.no_grad():
        replay = replay_model(range_batch, valid_batch)
    seed_replay_exact = all(torch.equal(base[name], replay[name]) for name in ("dense_event_presence_logits", "dense_event_relative_xyz_m", "event_confidence", "event_relative_xyz_m", "event_azimuth_bin_index", "event_depth_slot_index"))

    all_global = np.concatenate(global_ids)
    contract = structured_polar_multidepth_input_contract()
    checks = {
        "exact_population": counts["worlds"] == 80 and counts["observations"] == 188126 and len(np.unique(all_global)) == 188126 and counts["tokens"] == 131424 and len(identities) == 1076,
        "exact_split_and_cardinality": counts["fit_worlds"] == 60 and counts["selection_worlds"] == 20 and counts["fit_tokens"] == 98279 and counts["selection_tokens"] == 33145 and dict(cardinality) == EXPECTED_CARDINALITY,
        "all_tokens_roundtrip_without_overflow": overflow_rows == 0 and maximum_roundtrip_error <= 1e-4,
        "exact_277_second_depth_tokens": slot1_tokens == 277,
        "raster_targets_within_bounds": maximum_radial_fraction <= 1.0 + 1e-5 and maximum_azimuth_residual_deg <= 1.0 + 1e-5,
        "range_valid_only_signature": tuple(inspect.signature(model.forward).parameters) == ("range_m", "valid_mask"),
        "contract_180x2_top16_no_nms": contract["dense_layout"] == (180, 2) and contract["proposal_rule"] == "confidence_top16_over_360_without_neighbor_suppression",
        "exact_172430_parameters": parameter_count == 172430,
        "numpy_torch_profile_parity": profile_parity_error <= 1e-6,
        "finite_output_and_backward": finite_outputs and finite_backward,
        "prediction_range_and_azimuth_bounds": range_bound_error <= 1e-6 and azimuth_bound_error <= 1e-6,
        "top16_unique_and_repeat_exact": topk_unique and repeat_exact,
        "dense_rotation_equivariance": dense_logit_rotation_error <= 2e-5 and dense_xyz_rotation_error <= 2e-4,
        "top16_rotation_equivariance": top_xyz_rotation_error <= 2e-4 and top_index_rotation_exact,
        "seed_replay_exact": seed_replay_exact,
        "zero_training_threshold_C09_C10_MTARE": True,
    }
    checks["all_passed"] = all(checks.values())
    result = {
        "schema_version": "gse_structured_polar_multidepth_readiness_v1",
        "status": PASS if checks["all_passed"] else FAIL,
        "population": {"worlds": counts["worlds"], "fit_worlds": counts["fit_worlds"], "selection_worlds": counts["selection_worlds"], "observations": counts["observations"], "unique_global_sequence_indices": int(len(np.unique(all_global))), "tokens": counts["tokens"], "fit_tokens": counts["fit_tokens"], "selection_tokens": counts["selection_tokens"], "event_identities": len(identities), "cardinality": dict(sorted(cardinality.items())), "slot1_tokens": slot1_tokens},
        "rasterizer": {"overflow_rows": overflow_rows, "maximum_roundtrip_error_m": maximum_roundtrip_error, "maximum_radial_fraction": maximum_radial_fraction, "maximum_azimuth_residual_deg": maximum_azimuth_residual_deg, "profile_minimum_m": profile_minimum, "profile_maximum_m": profile_maximum},
        "model": {"parameters": parameter_count, "dense_slots": 360, "topk_events": 16, "profile_parity_error_m": profile_parity_error, "range_bound_max_signed_error_m": range_bound_error, "azimuth_bound_signed_error_rad": azimuth_bound_error, "dense_logit_rotation_error": dense_logit_rotation_error, "dense_xyz_rotation_error_m": dense_xyz_rotation_error, "top_xyz_rotation_error_m": top_xyz_rotation_error, "top_index_rotation_exact": top_index_rotation_exact, "topk_unique": topk_unique, "repeat_exact": repeat_exact, "seed_replay_exact": seed_replay_exact, "finite_backward": finite_backward},
        "contract": contract,
        "checks": checks,
        "per_world": per_world,
        "optimizer_steps": 0,
        "trained_model_inference_frames": 0,
        "threshold_selection_steps": 0,
        "duration_seconds": time.monotonic() - started,
        "c09_worlds_read": 0,
        "c10_worlds_read": 0,
        "mtare_worlds_read": 0,
    }
    write_json(output / "summary.json", result)
    print(json.dumps(result, indent=2))
    return 0 if checks["all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
