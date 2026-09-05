#!/usr/bin/env python3
"""Repeat P1b on the first completed real P1a shard without retaining assets."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np
import zarr

from _bootstrap import PROJECT_ROOT
from run_primitive_relation_p1b_teacher_materialization_v1 import _export_task


P1A = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1a_lossless_storage_corrective_v1_seed0"
TRAVERSALS = P1A / "artifacts/traversal_manifest.jsonl"


def main() -> None:
    rows = [json.loads(line) for line in (P1A / "logs/task_progress.jsonl").read_text().splitlines() if line.strip()]
    if not rows:
        raise RuntimeError("no complete P1a shard is available for the real P1b pilot")
    source = min(rows, key=lambda value: (int(value["frames"]), str(value["task_id"])))
    traversals = []
    with TRAVERSALS.open("r", encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            if value["parent_id"] == source["world"]: traversals.append(value)
    task = {
        **source,
        "source_run": str(P1A),
        "realization_index": {"ellipse": 0, "rounded_rectangle": 1, "c1_mixed": 2}[source["realization"]],
        "traversals": traversals,
    }
    results = []
    odometry_checks = []
    with tempfile.TemporaryDirectory(prefix="primitive_relation_p1b_real_pilot_") as temporary:
        root = Path(temporary)
        for repeat in range(2):
            task["output_root"] = str(root / f"repeat_{repeat}")
            results.append(_export_task(task))
            shard = zarr.open_group(
                str(root / f"repeat_{repeat}" / "teacher" / source["partition"] / f"{source['task_id']}.zarr"),
                mode="r",
            )
            translation = shard["relative_translation_current_sensor_m"][:]
            yaw = shard["relative_yaw_current_sensor_deg"][:]
            odometry_checks.append(
                np.isfinite(translation).all()
                and np.isfinite(yaw).all()
                and np.array_equal(translation[:, -1], np.zeros_like(translation[:, -1]))
                and np.array_equal(yaw[:, -1], np.zeros_like(yaw[:, -1]))
                and not ({"sensor_xyz_m", "axis_xyz_m", "yaw_deg"} & set(shard.array_keys()))
            )
    comparable = (
        "frames", "sequences", "primitive_count", "maximum_visible_primitives",
        "directed_attachment_labels", "undirected_disconnected_overlap_labels",
        "temporal_dustbin_labels", "first_target_digest", "shard_tree_sha256", "shard_bytes",
    )
    checks = {
        "real_p1a_shard_complete": source.get("deterministic_replay") is True,
        "five_frame_sequences_nonempty": results[0]["sequences"] > 0,
        "visible_capacity_le_32": results[0]["maximum_visible_primitives"] <= 32,
        "attachment_labels_nonempty": results[0]["directed_attachment_labels"] > 0,
        "repeat_exact": all(results[0][key] == results[1][key] for key in comparable),
        "relative_odometry_finite_current_zero_no_absolute_pose": all(odometry_checks),
        "temporary_assets_removed": not root.exists(),
    }
    payload = {
        "overall_status": "PASS_PRIMITIVE_RELATION_P1B_REAL_SHARD_CONTRACT" if all(checks.values()) else "FAIL_PRIMITIVE_RELATION_P1B_REAL_SHARD_CONTRACT",
        "checks": checks, "task_id": source["task_id"],
        "frames": results[0]["frames"], "sequences": results[0]["sequences"],
        "primitive_count": results[0]["primitive_count"], "maximum_visible_primitives": results[0]["maximum_visible_primitives"],
        "directed_attachment_labels": results[0]["directed_attachment_labels"],
        "undirected_disconnected_overlap_labels": results[0]["undirected_disconnected_overlap_labels"],
        "temporal_dustbin_labels": results[0]["temporal_dustbin_labels"],
        "repeat_tree_sha256": results[0]["shard_tree_sha256"],
        "repeat_duration_seconds": [value["duration_seconds"] for value in results],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not all(checks.values()): raise SystemExit(2)


if __name__ == "__main__":
    main()
