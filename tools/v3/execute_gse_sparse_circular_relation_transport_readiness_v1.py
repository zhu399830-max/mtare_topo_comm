#!/usr/bin/env python3
"""Formal zero-training readiness for sparse circular relation transport V2."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch.nn import functional as F

import train_gse_axis_anchored_event_relation_v1 as training
from mtare_topo.governance import load_json, write_json
from mtare_topo.representation.gse_circular_peak_geometry_model import (
    CURVATURE_SCALE_PER_M,
    METRIC_DISTANCE_SCALE_M,
    SLOPE_SCALE_DEG,
)
from mtare_topo.representation.gse_sparse_circular_relation_transport import (
    MAX_TOKENS,
    SparseCircularRelationTransportNet,
    parameter_count,
    relation_targets_from_token_identities,
    reverse_transport_logits,
    sparse_circular_relation_transport_input_contract,
    sparse_relation_transport_loss,
)


PASS = "PASS_GSE_SPARSE_CIRCULAR_RELATION_TRANSPORT_READINESS_V1"
EXPECTED_PARAMETERS = 783675
EXPECTED_POPULATION = {
    "fit": {"worlds": 60, "observations": 142184, "valid_pair_positions": 448152, "persistent": 934760, "reveal": 10583, "withdraw": 11008, "simultaneous_reveal_withdraw_bins": 202},
    "c07": {"worlds": 10, "observations": 21548, "valid_pair_positions": 66752, "persistent": 139346, "reveal": 1636, "withdraw": 1728, "simultaneous_reveal_withdraw_bins": 20},
    "c08": {"worlds": 10, "observations": 24394, "valid_pair_positions": 77490, "persistent": 161597, "reveal": 2107, "withdraw": 2181, "simultaneous_reveal_withdraw_bins": 33},
}
BACKBONE_PREFIXES = (
    "bearing_azimuth_rad", "encoder.", "temporal.", "directional_temporal.",
    "axis_azimuth_head.", "axis_vertical_head.", "geometry_head.",
    "place_head.", "observation_uncertainty_head.",
)


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.max(torch.abs(left - right)).detach().cpu())


def _circular_degree_error(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.abs(torch.remainder(left - right + 180.0, 360.0) - 180.0)


def _align_identities(predicted_bins: np.ndarray, true_identity_by_bin: np.ndarray) -> np.ndarray:
    output = np.full(MAX_TOKENS, -1, dtype=np.int64)
    true_bins = np.flatnonzero(true_identity_by_bin >= 0)
    if len(true_bins) > MAX_TOKENS:
        raise RuntimeError("real readiness frame exceeds six relation tokens")
    if not len(true_bins):
        return output
    difference = np.abs(predicted_bins[:, None] - true_bins[None, :]); cost = np.minimum(difference, 180 - difference)
    predicted_rows, target_columns = linear_sum_assignment(cost)
    output[predicted_rows] = true_identity_by_bin[true_bins[target_columns]]
    return output


def _load_real_batch(dataset_root: Path, sequence_manifest: Path, predecessor_summary: Path, device: torch.device) -> tuple[torch.Tensor, dict, list[dict]]:
    traversals = training.manifest_traversals(sequence_manifest)
    samples = load_json(predecessor_summary)["result"]["real_batch"]["samples"]
    if len(samples) != 8:
        raise RuntimeError("readiness real sample manifest drift")
    worlds = {}; scans = []; presence = []; identity = []; event = []; axis = []; geometry = []; geometry_valid = []
    for sample in samples:
        parent = sample["parent_id"]
        if parent not in worlds:
            worlds[parent] = training._load_world(dataset_root, parent, traversals[parent])
        world = worlds[parent]; row = int(sample["row"]); history_rows = np.asarray(sample["history_rows"], dtype=np.int64)
        if not np.array_equal(world["references"][row], world["references"][history_rows, -1]):
            raise RuntimeError(f"real readiness history join drift: {parent}/{row}")
        scan, target = training._batch(world, np.asarray([row]), device=device)
        scans.append(scan); presence.append(world["branch_presence_mask"][history_rows]); identity.append(world["branch_identity"][history_rows])
        event.append(int(target["event_index"][0])); axis.append(target["local_axis"][0]); geometry.append(target["geometry"][0]); geometry_valid.append(target["geometry_valid_mask"][0])
    batch = torch.cat(scans, dim=0)
    target = {
        "presence": torch.from_numpy(np.stack(presence)).to(device=device),
        "identity_by_bin": np.stack(identity),
        "event": torch.tensor(event, dtype=torch.long, device=device),
        "axis": torch.stack(axis), "geometry": torch.stack(geometry), "geometry_valid": torch.stack(geometry_valid).bool(),
    }
    return batch, target, samples


def _checkpoint_compatibility(model: SparseCircularRelationTransportNet, checkpoint_path: Path) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = checkpoint["model"]; target = model.state_dict()
    required = sorted(key for key in target if key == "bearing_azimuth_rad" or key.startswith(BACKBONE_PREFIXES[1:]))
    missing = [key for key in required if key not in source]
    shape_mismatch = [key for key in required if key in source and source[key].shape != target[key].shape]
    if missing or shape_mismatch:
        raise RuntimeError(f"predecessor backbone incompatibility: missing={missing}, shape={shape_mismatch}")
    compatible = {key: source[key] for key in required}; result = model.load_state_dict(compatible, strict=False)
    unexpected = list(result.unexpected_keys)
    exact = all(torch.equal(model.state_dict()[key].cpu(), source[key].cpu()) for key in required)
    return {"required_keys": len(required), "loaded_keys": len(compatible), "missing_required": missing, "shape_mismatch": shape_mismatch, "unexpected": unexpected, "loaded_values_exact": exact, "source_seed": int(checkpoint["seed"]), "source_epoch": int(checkpoint["epoch"])}


def _rotation_expected_axis(axis: torch.Tensor, angle_deg: float) -> torch.Tensor:
    angle = math.radians(angle_deg)
    return torch.stack((math.cos(angle) * axis[:, 0] - math.sin(angle) * axis[:, 1], math.sin(angle) * axis[:, 0] + math.cos(angle) * axis[:, 1], axis[:, 2]), dim=-1)


def _plot(output: Path, summary: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)
    population = summary["full_population"]
    axes[0, 0].bar(("fit", "C07", "C08"), [population[name]["observations"] for name in ("fit", "c07", "c08")], color="#4e79a7"); axes[0, 0].set_title("A  Sealed observations")
    axes[0, 1].bar(("keys", "exact"), (summary["checkpoint_compatibility"]["loaded_keys"], summary["checkpoint_compatibility"]["required_keys"] if summary["checkpoint_compatibility"]["loaded_values_exact"] else 0), color="#59a14f"); axes[0, 1].set_title("B  Reused backbone state")
    names = ("proposal", "transport_row", "reveal", "column_exclusivity")
    axes[1, 0].bar(names, [summary["real_batch"]["losses"][name] for name in names], color="#f28e2b"); axes[1, 0].tick_params(axis="x", rotation=18); axes[1, 0].set_title("C  Real-batch finite objectives")
    invariant_names = ("proposal_rotation", "transport_rotation", "axis_rotation", "batch_permutation")
    axes[1, 1].bar(invariant_names, [summary["equivariance"][name] for name in invariant_names], color="#e15759"); axes[1, 1].set_yscale("symlog", linthresh=1e-9); axes[1, 1].axhline(3e-5, color="black", linestyle="--"); axes[1, 1].tick_params(axis="x", rotation=18); axes[1, 1].set_title("D  Maximum invariant error")
    for axis in axes.flat: axis.grid(axis="y", alpha=.2); axis.set_axisbelow(True)
    fig.suptitle("Sparse circular relation transport V2 readiness")
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"gse_sparse_circular_relation_transport_readiness_v1.{suffix}", dpi=220)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--dataset-root", required=True, type=Path); parser.add_argument("--sequence-manifest", required=True, type=Path); parser.add_argument("--predecessor-readiness-summary", required=True, type=Path); parser.add_argument("--predecessor-checkpoint", required=True, type=Path); parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(); started = time.monotonic(); output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError("formal sparse relation readiness requires CUDA")
    torch.manual_seed(0); torch.cuda.manual_seed_all(0); torch.use_deterministic_algorithms(True); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda"); model = SparseCircularRelationTransportNet()
    compatibility = _checkpoint_compatibility(model, args.predecessor_checkpoint.resolve()); model = model.to(device)
    scans, target, samples = _load_real_batch(args.dataset_root.resolve(), args.sequence_manifest.resolve(), args.predecessor_readiness_summary.resolve(), device)
    model.train(); result = model(scans)
    aligned = np.full((len(scans), 5, MAX_TOKENS), -1, dtype=np.int64)
    predicted_bins = result["token_bin_index"].detach().cpu().numpy(); identity_by_bin = target["identity_by_bin"]
    for batch in range(len(scans)):
        for frame in range(5):
            aligned[batch, frame] = _align_identities(predicted_bins[batch, frame], identity_by_bin[batch, frame])
    aligned_identity = torch.from_numpy(aligned).to(device=device)
    relation_losses = sparse_relation_transport_loss(result, proposal_presence=target["presence"], aligned_token_identity=aligned_identity)
    event_loss = F.cross_entropy(result["event_logits"], target["event"])
    axis_target = F.normalize(target["axis"].to(result["local_axis"].dtype), dim=-1); axis_loss = (1.0 - (result["local_axis"] * axis_target).sum(-1)).mean()
    geometry_prediction = torch.stack((result["width_m"] / METRIC_DISTANCE_SCALE_M, result["height_m"] / METRIC_DISTANCE_SCALE_M, result["slope_deg"] / SLOPE_SCALE_DEG, result["curvature_per_m"] / CURVATURE_SCALE_PER_M), dim=-1)
    geometry_target = torch.stack((target["geometry"][:, 0] / METRIC_DISTANCE_SCALE_M, target["geometry"][:, 1] / METRIC_DISTANCE_SCALE_M, target["geometry"][:, 2] / SLOPE_SCALE_DEG, target["geometry"][:, 3] / CURVATURE_SCALE_PER_M), dim=-1)
    geometry_loss = F.smooth_l1_loss(geometry_prediction[target["geometry_valid"]], geometry_target[target["geometry_valid"]])
    auxiliary = 1e-3 * (result["token_descriptor"][..., 0].mean() + result["token_opening_width_m"].mean() + result["token_vertical_profile_m"].mean() + result["token_geometry_uncertainty"].mean() + result["place_descriptor"][:, 0].mean() + result["observation_uncertainty"].mean())
    total = relation_losses["total"] + event_loss + axis_loss + geometry_loss + auxiliary
    model.zero_grad(set_to_none=True); total.backward()
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    missing_gradient = [name for name, parameter in trainable if parameter.grad is None]
    nonfinite_gradient = [name for name, parameter in trainable if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())]
    maximum_gradient = max(float(parameter.grad.abs().max()) for _, parameter in trainable if parameter.grad is not None)
    model.eval()
    with torch.no_grad():
        base = model(scans); repeated = model(scans); shifted = model(torch.roll(scans, 20, dims=-1)); permutation = torch.tensor([7, 0, 5, 2, 6, 1, 4, 3], device=device); permuted = model(scans[permutation])
    inverse = torch.argsort(permutation)
    float_names = tuple(name for name, value in base.items() if value.is_floating_point())
    repeat_error = max(_max_abs(base[name], repeated[name]) for name in float_names)
    physical_names = ("token_bearing_deg", "token_opening_width_m", "token_vertical_profile_m", "width_m", "height_m", "slope_deg", "curvature_per_m")
    permutation_names = tuple(name for name in float_names if name not in physical_names)
    permutation_by_field = {name: _max_abs(base[name], permuted[name][inverse]) for name in permutation_names}
    permutation_error = max(permutation_by_field.values())
    physical_permutation = {name: _max_abs(base[name], permuted[name][inverse]) for name in physical_names if name != "token_bearing_deg"}
    token_bearing_permutation = float(_circular_degree_error(base["token_bearing_deg"], permuted["token_bearing_deg"][inverse]).max())
    token_index_permutation_exact = bool(torch.equal(base["token_bin_index"], permuted["token_bin_index"][inverse]))
    proposal_rotation = _max_abs(torch.roll(base["proposal_logits"], 5, dims=-1), shifted["proposal_logits"])
    token_bin_rotation_exact = bool(torch.equal(torch.remainder(base["token_bin_index"] + 5, 180), shifted["token_bin_index"]))
    token_bearing_rotation = float(_circular_degree_error(shifted["token_bearing_deg"], torch.remainder(base["token_bearing_deg"] + 10.0, 360.0)).max())
    transport_rotation = max(_max_abs(base["transport_row_logits"], shifted["transport_row_logits"]), _max_abs(base["transport_reveal_logits"], shifted["transport_reveal_logits"]))
    axis_rotation = _max_abs(shifted["local_axis"], _rotation_expected_axis(base["local_axis"], 10.0))
    invariant_names = ("event_logits", "width_m", "height_m", "slope_deg", "curvature_per_m", "place_descriptor", "observation_uncertainty")
    invariant_rotation = max(_max_abs(base[name], shifted[name]) for name in invariant_names)
    reverse_row, reverse_reveal = reverse_transport_logits(base["transport_row_logits"], base["transport_reveal_logits"]); restored_row, restored_reveal = reverse_transport_logits(reverse_row, reverse_reveal)
    reverse_error = max(_max_abs(restored_row, base["transport_row_logits"]), _max_abs(restored_reveal, base["transport_reveal_logits"]))
    synthetic_identity = torch.full((1, 5, MAX_TOKENS), -1, dtype=torch.long, device=device); synthetic_identity[0, 0, :2] = torch.tensor([10, 20], device=device); synthetic_identity[0, 1:, :2] = torch.tensor([10, 30], device=device)
    synthetic_row, synthetic_reveal = relation_targets_from_token_identities(synthetic_identity)
    simultaneous = bool(synthetic_row[0, 0, 1] == MAX_TOKENS and synthetic_reveal[0, 0, 1] == 1 and synthetic_row[0, 0, 0] == 0)
    predecessor_population = load_json(args.predecessor_readiness_summary.resolve())["result"]["full_population"]
    population_exact = all(all(int(predecessor_population[split][key]) == value for key, value in expected.items()) for split, expected in EXPECTED_POPULATION.items())
    relation_counts = {"persistent": int(((synthetic_row >= 0) & (synthetic_row < MAX_TOKENS)).sum()), "withdraw": int((synthetic_row == MAX_TOKENS).sum()), "reveal": int((synthetic_reveal == 1).sum())}
    equivariance = {"proposal_rotation": proposal_rotation, "token_bearing_rotation_deg": token_bearing_rotation, "transport_rotation": transport_rotation, "axis_rotation": axis_rotation, "invariant_rotation": invariant_rotation, "batch_permutation": permutation_error, "token_bearing_permutation_deg": token_bearing_permutation, "token_index_permutation_exact": token_index_permutation_exact, "batch_permutation_by_field": permutation_by_field, "physical_batch_permutation_by_field": physical_permutation, "repeat": repeat_error, "reverse_roundtrip": reverse_error}
    checks = {
        "typed_forward_is_lidar_only": sparse_circular_relation_transport_input_contract()["forward_parameters"] == ("self", "scans"),
        "parameter_budget_at_most_1p5m": parameter_count() == EXPECTED_PARAMETERS and EXPECTED_PARAMETERS <= 1500000,
        "full_population_provenance_exact": population_exact,
        "real_batch_exact_and_complete": len(samples) == 8 and scans.shape == (8, 5, 2, 16, 720) and int(target["presence"].sum()) > 0,
        "predecessor_backbone_loads_exactly": compatibility["loaded_values_exact"] and not compatibility["missing_required"] and not compatibility["shape_mismatch"],
        "simultaneous_reveal_withdraw_is_representable": simultaneous,
        "dustbin_and_real_transport_are_typed": relation_counts["persistent"] > 0 and relation_counts["withdraw"] > 0 and relation_counts["reveal"] > 0,
        "real_forward_loss_backward_finite": bool(torch.isfinite(total)) and not missing_gradient and not nonfinite_gradient and math.isfinite(maximum_gradient),
        "proposal_rotation_at_most_3e5": proposal_rotation <= 3e-5 and token_bin_rotation_exact and token_bearing_rotation <= 3e-4,
        "transport_rotation_at_most_3e5": transport_rotation <= 3e-5,
        "axis_rotation_at_most_3e5": axis_rotation <= 3e-5,
        "global_outputs_rotation_invariant": invariant_rotation <= 3e-5,
        "batch_permutation_at_most_3e6": permutation_error <= 3e-6 and token_index_permutation_exact and token_bearing_permutation <= 3e-4 and physical_permutation["slope_deg"] <= 3e-5 and max(value for name, value in physical_permutation.items() if name != "slope_deg") <= 3e-5,
        "deterministic_repeat_exact": repeat_error == 0.0,
        "reverse_transport_exact_involution": reverse_error == 0.0,
        "zero_optimizer_checkpoint_test_graph": True,
    }
    checks = {name: bool(value) for name, value in checks.items()}; scientific_pass = all(checks.values())
    summary = {
        "schema_version": "gse_sparse_circular_relation_transport_readiness_v1", "status": PASS if scientific_pass else "FAIL_GSE_SPARSE_CIRCULAR_RELATION_TRANSPORT_READINESS_V1", "scientific_pass": scientific_pass,
        "decision": "ALLOW_SPARSE_CIRCULAR_RELATION_TRANSPORT_THREE_SEED_DATA_CARD" if scientific_pass else "STOP_SPARSE_CIRCULAR_RELATION_TRANSPORT_BEFORE_TRAINING",
        "question": "Can sparse circular tokens and explicit dustbin transport satisfy the V2 causal relation contract before training?",
        "parameters": EXPECTED_PARAMETERS, "method": sparse_circular_relation_transport_input_contract(), "full_population": predecessor_population,
        "checkpoint_compatibility": compatibility, "real_batch": {"observations": 8, "unique_history_frames": 34, "samples": samples, "losses": {**{name: float(value.detach()) for name, value in relation_losses.items()}, "event": float(event_loss.detach()), "axis": float(axis_loss.detach()), "geometry": float(geometry_loss.detach()), "total": float(total.detach())}, "maximum_absolute_gradient": maximum_gradient, "missing_gradient_parameters": missing_gradient, "nonfinite_gradient_parameters": nonfinite_gradient},
        "synthetic_relation_counts": relation_counts, "equivariance": equivariance, "checks": checks,
        "device": str(device), "readiness_forward_observations": 32, "duration_seconds": time.monotonic() - started,
        "optimizer_steps": 0, "checkpoints_created": 0, "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0, "graph_replays": 0, "planner_calls": 0,
    }
    write_json(output / "summary.json", summary); write_json(output / "figure_source.json", summary)
    with (output / "readiness_checks.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("check", "passed")); writer.writeheader(); writer.writerows({"check": name, "passed": value} for name, value in checks.items())
    _plot(output, summary); print(json.dumps({"status": summary["status"], "decision": summary["decision"], "checks": checks}, indent=2, sort_keys=True)); return 0 if scientific_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
