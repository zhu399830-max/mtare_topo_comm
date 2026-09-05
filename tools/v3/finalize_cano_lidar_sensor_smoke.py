#!/usr/bin/env python3
"""Compare all 24 RTX scans to CPU rays and render complete smoke evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.cano_sensor_smoke import lidar_local_directions
from mtare_topo.governance import load_json, write_json


MIN_VALID_RATIO = 0.70
MIN_HIT_MASK_IOU = 0.80
MAX_MEDIAN_MAE_M = 0.10
MAX_P95_MAE_M = 0.50
ROLE_COLORS = {
    "tunnel_interior": "#1f77b4",
    "junction_transition": "#d62728",
    "terminal_approach": "#2ca02c",
}


def _provenance(fig: plt.Figure, run_id: str, text: str) -> None:
    fig.text(
        0.01,
        0.005,
        f"{run_id} | diagnostic smoke, not dataset | {text}",
        fontsize=7,
        color="#333333",
    )


def _point_cloud_xy(range_m: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = lidar_local_directions() * range_m[..., None]
    mask = valid.astype(bool)
    return points[..., 0][mask], points[..., 1][mask]


def _render_individual(
    destination: Path,
    run_id: str,
    pose: dict,
    rtx_range: np.ndarray,
    rtx_valid: np.ndarray,
    cpu_range: np.ndarray,
    metrics: dict,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    image = axes[0].imshow(rtx_range, aspect="auto", vmin=0, vmax=50, cmap="viridis")
    axes[0].set_title("Isaac RTX range (16×720)")
    axes[0].set_xlabel("azimuth column (0.5°)")
    axes[0].set_ylabel("elevation row (-15°…15°)")
    fig.colorbar(image, ax=axes[0], label="range [m]", fraction=0.046)

    x, y = _point_cloud_xy(rtx_range, rtx_valid)
    axes[1].scatter(x, y, s=0.25, c=np.hypot(x, y), cmap="viridis", vmin=0, vmax=50)
    axes[1].scatter([0], [0], marker="x", c="red", s=40)
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].set_xlim(-50, 50)
    axes[1].set_ylim(-50, 50)
    axes[1].set_title("RTX returns, sensor-local XY")
    axes[1].set_xlabel("x [m]")
    axes[1].set_ylabel("y [m]")

    azimuth = np.arange(720) * 0.5
    label = np.asarray(pose["label"]["label_720"])
    axes[2].plot(azimuth, label, color="#d62728", linewidth=1.5, label="5 m spline target")
    cpu_profile = np.mean(cpu_range, axis=0) / 50.0
    axes[2].plot(azimuth, cpu_profile, color="#777777", linewidth=0.6, alpha=0.7, label="CPU mean range / 50")
    for heading in pose["label"]["headings_robot_deg"]:
        axes[2].axvline(heading, color="#d62728", linestyle=":", linewidth=0.8)
    axes[2].set_xlim(0, 360)
    axes[2].set_ylim(0, 1.05)
    axes[2].set_title(f"360° structural target ({pose['label']['branch_count']} branches)")
    axes[2].set_xlabel("robot-relative azimuth [deg]")
    axes[2].legend(fontsize=7)
    fig.suptitle(
        f"{pose['sample_id']} | {pose['role']} | valid={metrics['rtx_valid_ratio']:.3f} "
        f"IoU={metrics['hit_mask_iou']:.3f} p95={metrics['matched_range_abs_error_p95_m']:.3f} m"
    )
    _provenance(fig, run_id, "complete 24/24 individual review; miss cells encoded as 50 m")
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def _render_range_label_sheet(
    destination: Path, run_id: str, poses: list[dict], arrays: dict[str, tuple[np.ndarray, np.ndarray]]
) -> None:
    fig, axes = plt.subplots(6, 4, figsize=(16, 18), sharex=True, sharey=True)
    for axis, pose in zip(axes.flat, poses):
        range_m, _valid = arrays[pose["sample_id"]]
        profile = np.min(range_m, axis=0)
        label = np.asarray(pose["label"]["label_720"])
        azimuth = np.arange(720) * 0.5
        axis.plot(azimuth, profile / 50.0, color="#444444", linewidth=0.55)
        axis.fill_between(azimuth, 0, label, color=ROLE_COLORS[pose["role"]], alpha=0.35)
        axis.set_title(pose["sample_id"], fontsize=8)
        axis.set_xlim(0, 360)
        axis.set_ylim(0, 1.02)
        axis.grid(alpha=0.15)
    fig.suptitle("All 24 RTX minimum-range profiles + 5 m structural labels (no sample selection)")
    fig.supxlabel("robot-relative azimuth [deg]")
    fig.supylabel("range/50 and target confidence")
    _provenance(fig, run_id, "rows follow frozen pose manifest order; colors encode role")
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.97))
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def _render_point_cloud_sheet(
    destination: Path, run_id: str, poses: list[dict], arrays: dict[str, tuple[np.ndarray, np.ndarray]]
) -> None:
    fig, axes = plt.subplots(6, 4, figsize=(16, 18), sharex=True, sharey=True)
    for axis, pose in zip(axes.flat, poses):
        range_m, valid = arrays[pose["sample_id"]]
        x, y = _point_cloud_xy(range_m, valid)
        axis.scatter(x, y, s=0.08, color=ROLE_COLORS[pose["role"]], alpha=0.55)
        axis.scatter([0], [0], marker="x", color="black", s=12)
        for heading in pose["label"]["headings_robot_deg"]:
            radians = np.radians(heading)
            axis.plot([0, 5 * np.cos(radians)], [0, 5 * np.sin(radians)], color="red", linewidth=0.8)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlim(-50, 50)
        axis.set_ylim(-50, 50)
        axis.set_title(pose["sample_id"], fontsize=8)
    fig.suptitle("All 24 Isaac RTX point clouds in sensor-local XY + structural directions")
    fig.supxlabel("x [m]")
    fig.supylabel("y [m]")
    _provenance(fig, run_id, "all valid returns shown; red rays are spline labels, not predictions")
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.97))
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def _render_pose_map(
    destination: Path, run_id: str, world_dir: Path, poses: list[dict]
) -> None:
    splines = load_json(world_dir / "splines.json")
    fig, axis = plt.subplots(figsize=(12, 9))
    for item in splines["tunnels"]:
        points = np.asarray(item["points"])
        axis.plot(points[:, 0], points[:, 1], color="#777777", linewidth=2, alpha=0.8)
        midpoint = points[len(points) // 2]
        axis.text(midpoint[0], midpoint[1], f"T{item['tunnel_id']}", fontsize=8)
    for role, color in ROLE_COLORS.items():
        selected = [pose for pose in poses if pose["role"] == role]
        xy = np.asarray([pose["axis_xyz_m"][:2] for pose in selected])
        axis.scatter(xy[:, 0], xy[:, 1], s=42, color=color, label=f"{role} (n={len(selected)})")
        for pose in selected:
            x, y = pose["axis_xyz_m"][:2]
            yaw = np.radians(pose["yaw_deg"])
            axis.arrow(x, y, 3 * np.cos(yaw), 3 * np.sin(yaw), color=color, width=0.15)
    axis.set_aspect("equal", adjustable="box")
    axis.set_title("Frozen seed-0 Cano world: complete 24-pose diagnostic layout")
    axis.set_xlabel("cano_world x [m]")
    axis.set_ylabel("cano_world y [m]")
    axis.legend()
    axis.grid(alpha=0.2)
    _provenance(fig, run_id, "spline axes only; background is not a reconstructed occupancy map")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(destination, dpi=170)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--world-dir", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    world_dir = args.world_dir.resolve()
    manifest = load_json(run_dir / "artifacts/pose_manifest.json")
    capture = load_json(run_dir / "metrics/isaac_capture.json")
    poses = manifest["poses"]
    if capture.get("overall_status") != "PASS_CAPTURED_24":
        raise RuntimeError("finalization requires a complete 24-pose Isaac capture")

    sample_metrics = []
    rtx_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    individual_dir = run_dir / "previews/all_24_individual"
    individual_dir.mkdir(exist_ok=False)
    all_passed = True
    for pose in poses:
        sample_id = pose["sample_id"]
        with np.load(run_dir / "artifacts/rtx_capture" / f"{sample_id}.npz") as rtx:
            rtx_range = rtx["range_m"].copy()
            rtx_valid = rtx["valid_mask"].copy().astype(bool)
        with np.load(run_dir / "artifacts/cpu_reference" / f"{sample_id}.npz") as cpu:
            cpu_range = cpu["range_m"].copy()
            cpu_valid = cpu["valid_mask"].copy().astype(bool)
        shape_passed = rtx_range.shape == cpu_range.shape == (16, 720)
        finite_passed = bool(np.all(np.isfinite(rtx_range)) and np.all(np.isfinite(cpu_range)))
        intersection = int(np.sum(rtx_valid & cpu_valid))
        union = int(np.sum(rtx_valid | cpu_valid))
        iou = intersection / union if union else 1.0
        matched_error = np.abs(rtx_range[rtx_valid & cpu_valid] - cpu_range[rtx_valid & cpu_valid])
        median_error = float(np.median(matched_error)) if matched_error.size else float("inf")
        p95_error = float(np.percentile(matched_error, 95)) if matched_error.size else float("inf")
        valid_ratio = float(np.mean(rtx_valid))
        passed = bool(
            shape_passed
            and finite_passed
            and valid_ratio >= MIN_VALID_RATIO
            and iou >= MIN_HIT_MASK_IOU
            and median_error <= MAX_MEDIAN_MAE_M
            and p95_error <= MAX_P95_MAE_M
        )
        all_passed = all_passed and passed
        metrics = {
            "sample_id": sample_id,
            "role": pose["role"],
            "expected_branch_count": pose["expected_branch_count"],
            "observed_label_branch_count": pose["label"]["branch_count"],
            "shape": list(rtx_range.shape),
            "shape_passed": shape_passed,
            "finite_passed": finite_passed,
            "rtx_valid_ratio": valid_ratio,
            "cpu_valid_ratio": float(np.mean(cpu_valid)),
            "hit_mask_iou": iou,
            "matched_cell_count": int(matched_error.size),
            "matched_range_abs_error_median_m": median_error,
            "matched_range_abs_error_p95_m": p95_error,
            "passed": passed,
        }
        sample_metrics.append(metrics)
        rtx_arrays[sample_id] = (rtx_range, rtx_valid)
        _render_individual(
            individual_dir / f"{sample_id}.png",
            run_dir.name,
            pose,
            rtx_range,
            rtx_valid,
            cpu_range,
            metrics,
        )

    role_counts = {
        role: sum(pose["role"] == role for pose in poses) for role in ROLE_COLORS
    }
    label_counts_passed = role_counts == {
        "tunnel_interior": 8,
        "junction_transition": 8,
        "terminal_approach": 8,
    } and all(
        pose["label"]["branch_count"] == pose["expected_branch_count"] for pose in poses
    )
    all_passed = all_passed and label_counts_passed and len(poses) == 24
    _render_range_label_sheet(
        run_dir / "previews/all24_range_label_contact_sheet.png",
        run_dir.name,
        poses,
        rtx_arrays,
    )
    _render_point_cloud_sheet(
        run_dir / "previews/all24_pointcloud_contact_sheet.png",
        run_dir.name,
        poses,
        rtx_arrays,
    )
    _render_pose_map(
        run_dir / "previews/pose_map_full_world.png", run_dir.name, world_dir, poses
    )
    aggregate = {
        "minimum_rtx_valid_ratio": min(item["rtx_valid_ratio"] for item in sample_metrics),
        "minimum_hit_mask_iou": min(item["hit_mask_iou"] for item in sample_metrics),
        "maximum_sample_median_abs_error_m": max(
            item["matched_range_abs_error_median_m"] for item in sample_metrics
        ),
        "maximum_sample_p95_abs_error_m": max(
            item["matched_range_abs_error_p95_m"] for item in sample_metrics
        ),
    }
    write_json(
        run_dir / "metrics/rtx_cpu_comparison.json",
        {
            "schema_version": "cano_rtx_cpu_comparison_v1",
            "overall_status": "PASS" if all_passed else "FAIL",
            "frozen_acceptance": {
                "pose_count": 24,
                "shape": [16, 720],
                "minimum_rtx_valid_ratio_per_pose": MIN_VALID_RATIO,
                "minimum_hit_mask_iou_per_pose": MIN_HIT_MASK_IOU,
                "maximum_matched_range_median_abs_error_per_pose_m": MAX_MEDIAN_MAE_M,
                "maximum_matched_range_p95_abs_error_per_pose_m": MAX_P95_MAE_M,
            },
            "role_counts": role_counts,
            "label_counts_passed": label_counts_passed,
            "aggregate": aggregate,
            "samples": sample_metrics,
        },
    )
    print(json.dumps({"passed": all_passed, **aggregate}, indent=2))
    return 0 if all_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
