#!/usr/bin/env python3
"""Same180 raw-vs-mean coordinate support; no learned forward/checkpoint."""
from collections import Counter
import json

import numpy as np
import scipy
import torch

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.gse_scoped_model_input import ScopedCompositionModelReader
from mtare_topo.evaluation.gse_coordinate_support import coordinate_support_bounds
from mtare_topo.governance import build_run_id, write_json
from mtare_topo.representation.primitive_relation_model import register_causal_lidar_points, _token_xyz
import run_gse_local_teacher_audit_v1 as executor


KIND = "same180_raw_vs_mean_coordinate_support_v1"


def json_bound(result):
    return {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in result.items()}


def aggregate(rows):
    points = [p for row in rows for p in row["coordinate_support"]["control_points"]]
    output = {"observations": len(rows), "control_points": len(points)}
    for pool in ("mean", "raw"):
        output[pool] = {"statuses": dict(Counter(p[pool + "_status"] for p in points)),
            "lower_bound_m": executor.distribution([p[pool + "_lower_bound_m"] for p in points]),
            "upper_bound_m": executor.distribution([p[pool + "_upper_bound_m"] for p in points])}
    output["mean_excluded_raw_witness"] = sum(p["pooling_exclusion_with_raw_witness"] for p in points)
    output["both_excluded"] = sum(p["mean_status"] == p["raw_status"] == "OUTSIDE_CERTIFIED" for p in points)
    output["both_witness"] = sum(p["mean_status"] == p["raw_status"] == "WITHIN_TOLERANCE_WITNESS" for p in points)
    return output


class CoordinateDiagnostic:
    coordinate_audit = True
    summary_key = "coordinate_support_expressivity"

    def start(self, spec, card, read_sealed):
        if (spec.get("diagnostic") != KIND or card.get("diagnostic") != KIND
                or spec.get("wall_time_cap_s") != 600
                or spec.get("expected_scipy_version") != scipy.__version__
                or card.get("sensor_usage") != "existing_coordinate_support_only_no_model_or_new_labels"):
            raise ValueError("coordinate support scope drift")
        expected = {}
        for seal in spec["source_seals"]:
            for line in executor.contained(seal).read_text().splitlines():
                digest, path = line.split(None, 1); expected[str(executor.contained(path))] = digest
        self.reader = ScopedCompositionModelReader(executor.contained(spec["sensor_root"]),
            executor.contained(spec["teacher_root"]), card["selected_rows"], expected_sha256=expected)
        self.additional_reads = self.reader.opened
        self.sensor_frames_decoded = 0
        previous = read_sealed(executor.contained(spec["previous_observation_audit"]), lambda p: json.loads(p.read_text()))
        self.previous = {(r["task"], r["row_index"]): r for r in previous}
        selected_keys = {(r["task"], r["row_index"]) for r in card["selected_rows"]}
        if len(previous) != 180 or len(self.previous) != 180 or set(self.previous) != selected_keys:
            raise ValueError("previous exact180 inventory drift")
        self.run = executor.PROJECT_ROOT / "results/gate3_semantics" / build_run_id(spec)
        self.row_log = self.run / "logs/coordinate_support.jsonl"
        self.row_log.touch(exist_ok=False)

    def task(self, rows, teacher, prediction):
        task = rows[0]["task"]
        batch = self.reader.read_task(task)
        if not np.array_equal(batch.frame_rows, teacher["frame_row"]):
            raise ValueError("raw-coordinate/teacher frame-order drift")
        self.sensor_frames_decoded += len(np.unique(batch.frame_rows))
        for i, row in enumerate(rows):
            if row != self.previous.get((task, row["row_index"])):
                raise ValueError("previous scoring rows not exactly reproduced")
            student = batch.student[i]
            with torch.inference_mode():
                # These are deterministic geometric transforms, NOT a neural
                # network forward. Use the exact historical float32 arithmetic.
                coordinates, valid = register_causal_lidar_points(
                    torch.from_numpy(student.range_valid[None]),
                    torch.from_numpy(student.relative_translation_current_sensor_m[None]),
                    torch.from_numpy(student.relative_yaw_current_sensor_deg[None]))
                pooled, pooled_valid = _token_xyz(coordinates, valid)
            raw = coordinates[valid].numpy()
            mean = pooled[pooled_valid].numpy()
            slots = np.flatnonzero(teacher["primitive_mask"][i])
            queries = teacher["axis_control_current_sensor_m"][i, slots].reshape(-1, 3)
            mean_bounds, raw_bounds = coordinate_support_bounds(mean, queries), coordinate_support_bounds(raw, queries)
            tolerance = mean_bounds["numerical_bound_m"] + raw_bounds["numerical_bound_m"]
            if np.any(raw_bounds["lower_bound_m"] > mean_bounds["upper_bound_m"] + tolerance):
                raise ValueError("mean-coordinate convex support is not contained in original return support within numeric bound")
            points = []
            for j, query in enumerate(queries):
                item = {"teacher_slot_scoring_only": int(slots[j // 3]), "control_index": j % 3,
                    "query_current_sensor_m": query.tolist()}
                for pool, bounds in (("mean", mean_bounds), ("raw", raw_bounds)):
                    item.update({pool + "_status": bounds["status"][j],
                        pool + "_lower_bound_m": float(bounds["lower_bound_m"][j]),
                        pool + "_upper_bound_m": float(bounds["upper_bound_m"][j])})
                item["pooling_exclusion_with_raw_witness"] = (item["mean_status"] == "OUTSIDE_CERTIFIED"
                    and item["raw_status"] == "WITHIN_TOLERANCE_WITNESS")
                points.append(item)
            row["coordinate_support"] = {"control_points": points,
                "mean_support": json_bound(mean_bounds), "raw_support": json_bound(raw_bounds),
                "point_order": "raw: C-order valid indices of B1x5x16x720 registered points; mean: C-order valid B1x5x180 mean tokens"}
            with self.row_log.open("a") as log:
                progress = {"task": task, "row_index": row["row_index"], "raw_points": len(raw),
                    "mean_points": len(mean), "controls": len(queries),
                    "mean_excluded_raw_witness": sum(p["pooling_exclusion_with_raw_witness"] for p in points)}
                log.write(json.dumps(progress) + "\n")

    def summarize(self, rows, run):
        result = aggregate(rows)
        if result["control_points"] != 4356 or self.sensor_frames_decoded != 900:
            raise ValueError("coordinate support exact population drift")
        result.update({"sensor_frames_decoded": self.sensor_frames_decoded,
            "additional_input_files_read": len(self.additional_reads), "no_model_or_checkpoint": True,
            "no_optimizer_or_new_targets": True, "old_row_score_parity_exact": True,
            "interpretation": "Support is a geometric capacity bound only. Global convex combinations may mix unrelated tunnels; a witness is not a learned primitive or observable port.",
            "numeric_policy": "SVD/Qhull only propose candidates; support maxima recomputed on all original points, upper bound has nonnegative sum-one reconstruction; no jitter/retry. Bounds cover source-coordinate quantization/rounding, not LiDAR measurement uncertainty."})
        parents = [{"task": task, **aggregate([r for r in rows if r["task"] == task])} for task in sorted({r["task"] for r in rows})]
        write_json(run / "artifacts/coordinate_support_by_parent.json", parents)
        lines = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="410">',
            '<rect width="100%" height="100%" fill="white"/>',
            '<text x="25" y="30" font-size="20">C01 coordinate support: pooled exclusion with raw reconstruction witness</text>',
            '<text x="25" y="55" font-size="14">All 180 observations / 4356 controls. Capacity diagnosis, not learned accuracy.</text>']
        for i, parent in enumerate(parents):
            x, count, total = 40 + 95*i, parent["mean_excluded_raw_witness"], parent["control_points"]
            height = 260 * count / total
            lines.extend([f'<rect x="{x}" y="{340-height:.2f}" width="55" height="{height:.2f}" fill="#da7628"/>',
                f'<text x="{x}" y="362" font-size="14">{parent["task"][:3]}</text>',
                f'<text x="{x}" y="{330-height:.2f}" font-size="12">{count}/{total}</text>'])
        lines.extend(['<text x="25" y="393" font-size="13">Bar height: fraction (0 to 1). All bounds, ambiguous cases and witnesses preserved per control.</text>', '</svg>'])
        (run / "previews/coordinate_support_by_parent.svg").write_text("\n".join(lines))
        return result


if __name__ == "__main__":
    raise SystemExit(executor.main(diagnostic=CoordinateDiagnostic()))
