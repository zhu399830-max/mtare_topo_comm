#!/usr/bin/env python3
"""Decompose the exact sealed 180-row audit's scoring correspondences."""
import json
from pathlib import Path

import numpy as np

from _bootstrap import PROJECT_ROOT
from mtare_topo.evaluation.gse_axis_error_decomposition import decompose_axes
from mtare_topo.governance import write_json
import run_gse_local_teacher_audit_v1 as executor


KIND = "fixed_assignment_axis_error_decomposition_v1"
METRICS = ("point_mean_m", "coordinate_mae_m", "axial_rms_m", "transverse_rms_m",
    "total_rms_m", "direction_error_deg", "undirected_direction_error_deg",
    "teacher_length_m", "predicted_length_m", "length_ratio", "length_absolute_error_m",
    "sensor_vertical_rms_m", "pred_to_teacher_polyline_m", "teacher_to_pred_polyline_m", "polyline_symmetric_m",
    "quadrature_error_bound_m")


def aggregate(matches):
    result = {"matches": len(matches), "metrics": {key: executor.distribution([
        m[key] for m in matches if m[key] is not None]) for key in METRICS},
        "teacher_direction_unknown": sum(not m["teacher_direction_resolved"] for m in matches),
        "prediction_direction_unknown": sum(not m["prediction_direction_resolved"] for m in matches)}
    axial = sum(m["axial_squared_error_sum_m2"] for m in matches if m["axial_squared_error_sum_m2"] is not None)
    transverse = sum(m["transverse_squared_error_sum_m2"] for m in matches if m["transverse_squared_error_sum_m2"] is not None)
    result["axial_squared_fraction_resolved_only"] = axial / (axial + transverse) if axial + transverse > 0 else None
    result["metric_unknown_counts"] = {key: sum(m[key] is None for m in matches) for key in METRICS}
    return result


def scatter_svg(matches):
    """All resolved matches, not a chosen success/failure population."""
    valid = [m for m in matches if m["teacher_direction_resolved"]]
    limit = max([m[k] for m in valid for k in ("axial_rms_m", "transverse_rms_m")] + [1.])
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="850" height="740" viewBox="0 0 850 740">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="40" y="28" font-size="19">C01 / 180 observations: axis-control error decomposition</text>',
        '<text x="40" y="51" font-size="13">Orange: target-incident; gray: other visible fragments. Scoring identities only.</text>',
        '<text x="40" y="71" font-size="13">Frozen correspondence; dashed line is equal error, NOT an acceptance threshold.</text>',
        '<rect x="85" y="100" width="570" height="570" fill="none" stroke="#222"/>',
        '<path d="M85 670 L655 100" stroke="#aaa" stroke-dasharray="4 4"/>']
    for j in range(6):
        v, pos = limit * j / 5, 570 * j / 5
        parts.append(f'<text x="{85 + pos:.2f}" y="690" font-size="12" text-anchor="middle">{v:.1f}</text>')
        parts.append(f'<text x="75" y="{674 - pos:.2f}" font-size="12" text-anchor="end">{v:.1f}</text>')
    for m in sorted(valid, key=lambda m: m["target_incident_scoring_only"]):
        x, y = 85 + m["axial_rms_m"] / limit * 570, 670 - m["transverse_rms_m"] / limit * 570
        color = "#d76516" if m["target_incident_scoring_only"] else "#7b8793"
        parts.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="2.5" fill="{color}" fill-opacity=".55"/>')
    parts.extend(['<text x="330" y="719" font-size="16">Along teacher chord RMS (m)</text>',
        '<text transform="translate(22,505) rotate(-90)" font-size="16">Transverse to teacher chord RMS (m)</text>',
        f'<text x="680" y="120" font-size="13">Plotted: {len(valid)}</text>',
        f'<text x="680" y="143" font-size="13">Unknown: {len(matches)-len(valid)}</text>', '</svg>'])
    return "\n".join(parts)


class AxisDiagnostic:
    def start(self, spec, card, read_sealed):
        if (spec.get("diagnostic") != KIND or card.get("diagnostic") != KIND
                or spec.get("quadrature_samples") != 256
                or card.get("new_scoring_thresholds") is not False):
            raise ValueError("axis decomposition scope drift")
        previous = read_sealed(executor.contained(spec["previous_observation_audit"]), lambda p: json.loads(p.read_text()))
        self.previous = {(r["task"], r["row_index"]): r for r in previous}
        selected = {(r["task"], r["row_index"]) for r in card["selected_rows"]}
        if len(self.previous) != len(previous) or set(self.previous) != selected:
            raise ValueError("previous audit row selection drift")

    def task(self, rows, teacher, prediction):
        for i, row in enumerate(rows):
            previous = self.previous[(row["task"], row["row_index"])]
            # All old row metadata, incidence and float32 scores must reproduce
            # BEFORE adding diagnostics. No re-matching to improve the result.
            if row != previous:
                raise ValueError("old scoring correspondence/metrics did not reproduce exactly")
            group = next(c for c in row["all_visible_construction_groups_scoring_only"]
                if c["node_id_scoring_only"] == str(row["target_node_scoring_only"]))
            incident = {m["teacher_slot_scoring_only"] for m in group["present_members"]}
            diagnostics = []
            for match in row["oracle_geometry_matches_not_detections"]:
                pslot, tslot = match["prediction_slot"], match["teacher_slot_scoring_only"]
                truth = teacher["axis_control_current_sensor_m"][i, tslot]
                if match["reversed"]:
                    truth = truth[::-1]
                metrics = decompose_axes(prediction["axis_control_current_sensor_m"][i, pslot], truth)
                if not np.isclose(metrics["point_mean_m"], match["axis_control_point_mean_euclidean_m"], rtol=2e-7, atol=2e-6):
                    raise ValueError("float64 decomposition does not reproduce float32 point error within source precision")
                if metrics["teacher_direction_resolved"]:
                    identity = metrics["axial_rms_m"]**2 + metrics["transverse_rms_m"]**2
                    if not np.isclose(identity, metrics["total_rms_m"]**2, rtol=1e-12, atol=1e-12):
                        raise ValueError("orthogonal squared-error decomposition broken")
                diagnostics.append({**metrics, "prediction_slot": pslot, "teacher_slot_scoring_only": tslot,
                    "target_incident_scoring_only": tslot in incident,
                    "existence_probability_unchanged": match["existence_probability"]})
            row["axis_decomposition_fixed_assignment"] = diagnostics

    def summarize(self, rows, run):
        matches = [m for r in rows for m in r["axis_decomposition_fixed_assignment"]]
        if len(matches) != 1452:
            raise ValueError("fixed matched primitive count drift")
        result = {"all": aggregate(matches),
            "target_incident": aggregate([m for m in matches if m["target_incident_scoring_only"]]),
            "other_visible_fragments": aggregate([m for m in matches if not m["target_incident_scoring_only"]]),
            "old_row_and_score_parity_exact": True,
            "teacher_curve_definition": "Diagnostic two-segment interpolant of three supervised cropped controls, not the original map spline or a surface-distance benchmark.",
            "legacy_metric_comparison": "All fixed1452 assignments retained. Historical existence-gated metrics and optional endpoint-observed tie ordering are not reproduced; no new threshold or assignment selected.",
            "direction_unknown_policy": "Chord no longer than 2sqrt(3)*max input ULP is unresolved; keep row and report None, never silently delete or fill direction.",
            "quadrature_policy": "256 equal-arc midpoint samples per direction with analytic finite-segment distances. Symmetric mean approximation error <= (Lpred+Lteacher)/(4*256)."}
        parents = [{"task": task, **aggregate([m for r in rows if r["task"] == task
            for m in r["axis_decomposition_fixed_assignment"]])} for task in sorted({r["task"] for r in rows})]
        write_json(run / "artifacts/axis_decomposition_by_parent.json", parents)
        (run / "previews/axis_error_decomposition.svg").write_text(scatter_svg(matches))
        return result


if __name__ == "__main__":
    raise SystemExit(executor.main(diagnostic=AxisDiagnostic()))
