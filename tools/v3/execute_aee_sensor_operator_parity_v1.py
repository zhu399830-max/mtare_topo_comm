#!/home/zeng-workstation/anaconda3/bin/python
"""Compare AEE real and same-pose ideal DAE scans with B0 and frozen M1D."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.aee_head_adaptation import binary_direction_headings
from mtare_topo.data.aee_sensor_operator_parity import evenly_spaced_indices, observation_parity_metrics
from mtare_topo.data.cano_sensor_smoke import ELEVATION_DEG
from mtare_topo.evaluation.phase3_semantic_metrics import decode_direction_components, match_headings
from mtare_topo.representation.phase3_structural_semantics import StructuralSemanticNet
from mtare_topo.semantics.range_exit_baseline import RangeExitBaseline


SENSOR_RUN = PROJECT_ROOT / "results/gate2_representation/gate2_20260820_aee_domain_sensor_export_v1r3_seed20260820"
TEACHER_RUN = PROJECT_ROOT / "results/gate2_representation/gate2_20260821_aee_dae_multilayer_objective_teacher_export_v1r_seed20260820"
CHECKPOINTS = {
    0: (PROJECT_ROOT / "results/gate2_representation/gate2_20260812_cano_phase3_masking_corrective_ray_dropout_v1r3_seed0/artifacts/models/m1d_seed0/best.pt", "55f6602fb7fd74e3a44c3a4697a7c4be4d31469f46348f631dc606612125f9fe"),
    1: (PROJECT_ROOT / "results/gate2_representation/gate2_20260812_cano_phase3_masking_corrective_ray_dropout_v1r3_seed0/artifacts/models/m1d_seed1/best.pt", "3822d27af6d5fb0ef920da3e310927d42c5a112c5d671b4cbdc81d9db251f8df"),
    2: (PROJECT_ROOT / "results/gate2_representation/gate2_20260813_cano_phase3_masking_corrective_recovery_seed2_v1_seed2/artifacts/models/m1d_seed2/best.pt", "20b1e9c8198b06473b68e31f4d8a3eb50eb5f2fc1b7aa577aa1aac92ba35346e"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def direction_metrics(predictions: list[list[float]], truth: list[tuple[float, ...]]) -> dict:
    matched = predicted = target = 0
    errors: list[float] = []
    for left, right in zip(predictions, truth, strict=True):
        m, p, t, e = match_headings(left, right, 20.0)
        matched += m; predicted += p; target += t; errors.extend(e)
    precision = matched / max(predicted, 1)
    recall = matched / max(target, 1)
    return {
        "matched": matched, "predicted": predicted, "truth": target,
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "empty_rate": float(np.mean([len(item) == 0 for item in predictions])),
        "mean_matched_angular_error_deg": float(np.mean(errors)) if errors else None,
    }


def load_world(world: str, ideal: dict[str, np.ndarray], manifest: dict) -> dict[str, np.ndarray | list]:
    selected = evenly_spaced_indices(3000, 32)
    world_records = [item for item in manifest["records"] if item["world"] == world]
    real_range, real_valid, truth, frame_ids, roles, counts = [], [], [], [], [], []
    for global_index in selected:
        trajectory_index, row = divmod(int(global_index), 600)
        record = world_records[trajectory_index]
        sensor_path = SENSOR_RUN / record["sensor_shard"]
        teacher_path = TEACHER_RUN / record["teacher_shard"]
        if sha256(sensor_path) != record["sensor_shard_sha256"] or sha256(teacher_path) != record["teacher_shard_sha256"]:
            raise RuntimeError(f"selected shard identity drift: {record['trajectory_id']}")
        with np.load(sensor_path, allow_pickle=False) as sensor, np.load(teacher_path, allow_pickle=False) as teacher:
            if str(sensor["frame_id"][row]) != str(teacher["frame_id"][row]):
                raise RuntimeError("sensor/teacher frame identity mismatch")
            real_range.append(sensor["range_m"][row]); real_valid.append(sensor["valid_mask"][row])
            truth.append(binary_direction_headings(teacher["direction_target"][row]))
            frame_ids.append(str(sensor["frame_id"][row])); roles.append(int(teacher["role_target"][row])); counts.append(int(teacher["count_target"][row]))
    if not np.array_equal(ideal[f"{world}_selected_global_index"], selected):
        raise RuntimeError("ideal scan selection drift")
    return {
        "real_range": np.stack(real_range), "real_valid": np.stack(real_valid),
        "ideal_range": ideal[f"{world}_range_m"], "ideal_valid": ideal[f"{world}_valid_mask"],
        "truth": truth, "frame_id": frame_ids, "role_target": np.asarray(roles), "count_target": np.asarray(counts),
    }


@torch.inference_mode()
def infer(model: StructuralSemanticNet, ranges: np.ndarray, valid: np.ndarray) -> dict[str, np.ndarray]:
    student = torch.from_numpy(np.stack((np.clip(ranges, 0.3, 50.0) / 50.0, valid.astype(np.float32)), axis=1).astype(np.float32))
    output = model(student)
    return {key: value.cpu().numpy() for key, value in output.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--ideal-scans", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve(); started = time.monotonic()
    manifest = json.loads((TEACHER_RUN / "artifacts/teacher_manifest.json").read_text())
    with np.load(args.ideal_scans, allow_pickle=False) as source:
        ideal = {key: np.asarray(source[key]) for key in source.files}
    worlds = {world: load_world(world, ideal, manifest) for world in ("tunnel", "garage")}
    b0 = RangeExitBaseline(); summary_worlds = {}; arrays = {}
    for world, data in worlds.items():
        parity = observation_parity_metrics(data["real_range"], data["real_valid"], data["ideal_range"], data["ideal_valid"])
        b0_metrics = {}
        for domain in ("real", "ideal"):
            predictions = [b0.predict(r, v, ELEVATION_DEG)["headings_robot_deg"] for r, v in zip(data[f"{domain}_range"], data[f"{domain}_valid"], strict=True)]
            b0_metrics[domain] = direction_metrics(predictions, data["truth"])
        row_real = np.mean(data["real_valid"], axis=(0, 2))
        row_ideal = np.mean(data["ideal_valid"], axis=(0, 2))
        summary_worlds[world] = {
            "frames": 32, "observation": parity.to_dict(), "b0": b0_metrics,
            "real_valid_fraction_by_elevation": row_real.tolist(),
            "ideal_valid_fraction_by_elevation": row_ideal.tolist(),
            "valid_fraction_delta_real_minus_ideal_by_elevation": (row_real - row_ideal).tolist(),
            "m1d": {},
        }
        for key in ("real_range", "real_valid", "ideal_range", "ideal_valid", "role_target", "count_target"):
            arrays[f"{world}_{key}"] = data[key]
        arrays[f"{world}_frame_id"] = np.asarray(data["frame_id"], dtype="U128")
    inference_frames = 0
    for seed, (checkpoint_path, expected_hash) in CHECKPOINTS.items():
        if sha256(checkpoint_path) != expected_hash:
            raise RuntimeError(f"M1D seed {seed} checkpoint drift")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("mode") != "M1D" or int(checkpoint.get("seed", -1)) != seed:
            raise RuntimeError(f"M1D seed {seed} identity drift")
        model = StructuralSemanticNet(); model.load_state_dict(checkpoint["model"], strict=True); model.eval()
        for world, data in worlds.items():
            domain_outputs = {}
            for domain in ("real", "ideal"):
                output = infer(model, data[f"{domain}_range"], data[f"{domain}_valid"]); inference_frames += 32
                predictions = [decode_direction_components(logits, 0.5) for logits in output["direction_logits"]]
                domain_outputs[domain] = output
                summary_worlds[world]["m1d"].setdefault(str(seed), {})[domain] = {
                    **direction_metrics(predictions, data["truth"]),
                    "role_accuracy": float(np.mean(output["role_logits"].argmax(1) == data["role_target"])),
                    "count_accuracy": float(np.mean(output["count_logits"].argmax(1) == data["count_target"])),
                }
                arrays[f"{world}_m1d_seed{seed}_{domain}_direction_logits"] = output["direction_logits"].astype(np.float32)
                arrays[f"{world}_m1d_seed{seed}_{domain}_z_role"] = output["z_role"].astype(np.float32)
            cosine = np.sum(domain_outputs["real"]["z_role"] * domain_outputs["ideal"]["z_role"], axis=1)
            summary_worlds[world]["m1d"][str(seed)]["real_ideal_z_cosine"] = {
                "mean": float(cosine.mean()), "p05": float(np.quantile(cosine, 0.05)), "minimum": float(cosine.min())
            }
    decision_checks = {}
    for world in ("tunnel", "garage"):
        seeds = summary_worlds[world]["m1d"]
        real_f1 = np.asarray([seeds[str(seed)]["real"]["f1"] for seed in range(3)])
        ideal_f1 = np.asarray([seeds[str(seed)]["ideal"]["f1"] for seed in range(3)])
        real_empty = np.asarray([seeds[str(seed)]["real"]["empty_rate"] for seed in range(3)])
        ideal_empty = np.asarray([seeds[str(seed)]["ideal"]["empty_rate"] for seed in range(3)])
        decision_checks[world] = {
            "all_three_seeds_ideal_f1_strictly_above_real": bool(np.all(ideal_f1 > real_f1)),
            "all_three_seeds_ideal_empty_strictly_below_real": bool(np.all(ideal_empty < real_empty)),
            "median_real_f1": float(np.median(real_f1)), "median_ideal_f1": float(np.median(ideal_f1)),
            "median_real_empty_rate": float(np.median(real_empty)), "median_ideal_empty_rate": float(np.median(ideal_empty)),
        }
    supported = all(all(value[key] for key in ("all_three_seeds_ideal_f1_strictly_above_real", "all_three_seeds_ideal_empty_strictly_below_real")) for value in decision_checks.values())
    conclusion = "PAIRED_FRONTEND_ADAPTER_SUPPORTED" if supported else "MORE_INDEPENDENT_AEE_GEOMETRY_REQUIRED"
    summary = {
        "schema_version": "aee_sensor_operator_parity_v1", "overall_status": "COMPLETED_AEE_SENSOR_OPERATOR_PARITY_V1",
        "conclusion": conclusion, "worlds": summary_worlds, "decision_checks": decision_checks,
        "unique_frames": 64, "b0_predictions": 128, "m1d_checkpoints": 3, "model_inference_frames": inference_frames,
        "training_steps": 0, "optimizer_steps": 0, "c09_reads": 0, "c10_reads": 0,
        "duration_seconds": time.monotonic() - started,
    }
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True); (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics/summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(run_dir / "artifacts/audit_arrays.npz", **arrays)
    (run_dir / "previews").mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(4, 2, figsize=(14, 8), constrained_layout=True)
    for column, world in enumerate(("tunnel", "garage")):
        data = worlds[world]; index = 16
        both = data["real_valid"][index].astype(bool) & data["ideal_valid"][index].astype(bool)
        absolute = np.where(both, np.abs(data["real_range"][index] - data["ideal_range"][index]), np.nan)
        panels = (
            (data["real_range"][index], "real range", 0.3, 50.0),
            (data["ideal_range"][index], "DAE ideal range", 0.3, 50.0),
            (data["real_valid"][index].astype(int) - data["ideal_valid"][index].astype(int), "valid: real - ideal", -1, 1),
            (absolute, "absolute error on both-valid", 0, 5),
        )
        for row, (image, label, lower, upper) in enumerate(panels):
            axes[row, column].imshow(image, aspect="auto", origin="lower", vmin=lower, vmax=upper)
            axes[row, column].set_title(f"{world}: {label}")
            axes[row, column].set_ylabel("elevation row")
            axes[row, column].set_xlabel("azimuth column")
    figure.savefig(run_dir / "previews/same_pose_real_vs_ideal.png", dpi=150)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for world, linestyle in (("tunnel", "-"), ("garage", "--")):
        axis.plot(ELEVATION_DEG, summary_worlds[world]["real_valid_fraction_by_elevation"], linestyle, label=f"{world} real")
        axis.plot(ELEVATION_DEG, summary_worlds[world]["ideal_valid_fraction_by_elevation"], linestyle, alpha=0.55, label=f"{world} ideal")
    axis.set_xlabel("elevation (deg)"); axis.set_ylabel("valid return fraction"); axis.set_ylim(0, 1.02); axis.grid(alpha=0.25); axis.legend()
    figure.savefig(run_dir / "previews/valid_fraction_by_elevation.png", dpi=150)
    plt.close(figure)
    print(json.dumps({"status": summary["overall_status"], "conclusion": conclusion, "inference_frames": inference_frames}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
