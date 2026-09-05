#!/usr/bin/env python3
"""One bounded existing-teacher/cache audit; no model execution or new labels."""
import argparse
import html
import json
from pathlib import Path
import platform
import resource
import signal
import time
import traceback

import numpy as np
import torch
import zarr

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.gse_local_teacher_audit import (
    FIELDS, PREDICTION_FIELDS, ScopedLocalTeacherReader,
    expected_construction_attachment, score_cached_geometry,
)
from mtare_topo.governance import build_run_id, load_json, write_json
from mtare_topo.governance_inventory import validate_scoped_inventory_card, validate_scoped_coordinate_audit_card
from run_gse_composition_inventory_v1 import sha


def contained(relative):
    path = (PROJECT_ROOT / relative).resolve()
    if not path.is_relative_to(PROJECT_ROOT):
        raise ValueError("input path escapes project")
    return path


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"n": 0, "mean": None, "median": None, "p90": None, "min": None, "max": None}
    return {"n": len(values), "mean": float(values.mean()), "median": float(np.median(values)),
        "p90": float(np.quantile(values, .9)), "min": float(values.min()), "max": float(values.max())}


def audit_task(records, teacher, prediction, construction):
    expected = expected_construction_attachment(teacher["primitive_index"], teacher["primitive_mask"], construction)
    if not np.array_equal(expected, teacher["endpoint_attachment"]):
        raise ValueError("stored relations differ from all visible construction incidence")
    lookup = {str(p["primitive_id"]): i for i, p in enumerate(construction["realized_primitives"])}
    groups = construction["base_construction"]["composition_operations"]
    if len({str(c["node_id"]) for c in groups}) != len(groups):
        raise ValueError("duplicate construction node")
    scores = score_cached_geometry(prediction, teacher)
    rows = []
    for b, record in enumerate(records):
        active = np.flatnonzero(teacher["primitive_mask"][b])
        slot_for = {int(teacher["primitive_index"][b, s]): int(s) for s in active}
        composition_inventory = []
        for c in groups:
            present = [{"teacher_slot_scoring_only": slot_for[lookup[str(m["primitive_id"])]],
                        "physical_endpoint_index_scoring_only": m["endpoint_index"]}
                       for m in c["member_endpoints"] if lookup[str(m["primitive_id"]) ] in slot_for]
            if present:
                composition_inventory.append({"node_id_scoring_only": str(c["node_id"]),
                    "construction_degree_not_event_label": c["degree"], "present_members": present,
                    "all_incident_fragments_present": len(present) == c["degree"],
                    "observable_event_validity": "NOT_ESTABLISHED_BY_FRAGMENT_MEMBERSHIP"})
        target = next((c for c in composition_inventory
            if c["node_id_scoring_only"] == str(record["target_node_scoring_only"])), None)
        if target is None or target["construction_degree_not_event_label"] != record["degree"]:
            raise ValueError("selected legacy node/degree absent or drifted")
        rows.append({**record, "frame_rows": teacher["frame_row"][b].tolist(),
            "visible_primitive_fragments": len(active),
            "incident_fragments_present": len(target["present_members"]),
            "foreign_fragments_present": len(active) - len(target["present_members"]),
            "all_target_incident_fragments_present": target["all_incident_fragments_present"],
            "construction_attachment_pairs": int(teacher["endpoint_attachment"][b].sum()) // 2,
            "angular_alias_pairs_not_physical_overlap": int(teacher["angular_overlap"][b].sum()) // 2,
            "all_visible_construction_groups_scoring_only": composition_inventory,
            "oracle_geometry_matches_not_detections": scores[b], "training_target_generated": False})
    return rows


def preview_svg(task, rows, teacher, prediction):
    """Every row in a task, both projections; vector evidence, no selection."""
    width, height = 1000, 90 + 210 * len(rows)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="20" y="24" font-size="17">{html.escape(task)}</text>',
        '<text x="20" y="47" font-size="13">Black: cropped teacher axes; orange: oracle-matched predictions; gray: all other slots.</text>',
        '<text x="20" y="67" font-size="13">Assignment is scoring-only, not accepted detection. Each XY/XZ panel has equal metric scale.</text>']
    for b, row in enumerate(rows):
        truth = teacher["axis_control_current_sensor_m"][b, teacher["primitive_mask"][b].astype(bool)]
        pred = prediction["axis_control_current_sensor_m"][b]
        matched = {m["prediction_slot"] for m in row["oracle_geometry_matches_not_detections"]}
        all_points = np.concatenate((truth.reshape(-1, 3), pred.reshape(-1, 3), np.zeros((1, 3))))
        for panel, axes in enumerate(((0, 1), (0, 2))):
            left, top = 15 + 495 * panel, 85 + 210 * b
            low, high = all_points[:, axes].min(axis=0), all_points[:, axes].max(axis=0)
            scale = min(465 / max(high[0] - low[0], 1), 163 / max(high[1] - low[1], 1))
            center = (low + high) / 2
            def point(p):
                q = (p[list(axes)] - center) * scale
                return f"{left + 240 + q[0]:.3f},{top + 113 - q[1]:.3f}"
            label = f'row={row["row_index"]} {"XY" if panel == 0 else "XZ"} | fragments={len(truth)}, GT degree={row["degree"]} (not event label)'
            parts.append(f'<rect x="{left}" y="{top}" width="480" height="200" fill="none" stroke="#bbb"/>')
            parts.append(f'<text x="{left + 6}" y="{top + 17}" font-size="11">{html.escape(label)}</text>')
            for slot, axis in enumerate(pred):
                color = "#dc6815" if slot in matched else "#d7dce0"
                parts.append(f'<polyline points="{" ".join(point(p) for p in axis)}" fill="none" stroke="{color}" stroke-width="1.2"/>')
            for axis in truth:
                parts.append(f'<polyline points="{" ".join(point(p) for p in axis)}" fill="none" stroke="#111" stroke-width="1.5"/>')
            origin = point(np.zeros(3)).split(",")
            parts.append(f'<circle cx="{origin[0]}" cy="{origin[1]}" r="3" fill="#176dc1"/>')
    return "\n".join(parts + ["</svg>"])


def main(*, diagnostic=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    spec, run = load_json(args.spec), args.run_dir.resolve()
    if (run != PROJECT_ROOT / "results/gate3_semantics" / build_run_id(spec)
            or load_json(run / "RUN_STATE.json")["state"] != "CREATED_NOT_EXECUTED"
            or load_json(run / "config/run_spec.json") != spec):
        raise RuntimeError("requires fresh run; no overwrite or retry")
    started, error, summary, accessed = time.monotonic(), None, {}, {}
    write_json(run / "RUN_STATE.json", {"state": "RUNNING", "run_id": run.name})
    def deadline(signum, frame):
        raise TimeoutError(f"audit exceeded frozen {wall_cap}s deadline")
    wall_cap = 600 if diagnostic is not None and getattr(diagnostic, "coordinate_audit", False) else 120
    old_handler = signal.signal(signal.SIGALRM, deadline); signal.alarm(wall_cap)
    try:
        card = load_json(contained(spec["data_card"]))
        validator = validate_scoped_coordinate_audit_card if wall_cap == 600 else validate_scoped_inventory_card
        validation = validator(card)
        if (not validation.passed or load_json(run / "config/data_card.json") != card
                or spec["operation"] != "audit" or card["observation_count"] != 180
                or card.get("existing_prediction_cache_read_only") is not True
                or card.get("allowed_teacher_fields") != sorted(FIELDS)
                or card.get("allowed_cache_fields") != list(PREDICTION_FIELDS)):
            raise ValueError(f"exact audit scope/card drift: {validation.errors}")
        frozen = {**card["sealed_sources"], **spec["source_sha256"]}
        for relative, digest in frozen.items():
            if sha(contained(relative)) != digest:
                raise ValueError(f"frozen input/tool drift: {relative}")
        versions = {"python": platform.python_version(), "numpy": np.__version__,
                    "torch": torch.__version__, "zarr": zarr.__version__}
        if versions != spec["expected_versions"]:
            raise ValueError(f"environment drift: {versions}")
        write_json(run / "config/execution_environment.json", {**versions, "device": "cpu", "platform": platform.platform()})
        records = json.loads(contained(spec["selection_manifest"]).read_text())
        selected = [{k: r[k] for k in ("task", "row_index", "source_global_sequence_index")} for r in records]
        if selected != card["selected_rows"] or len({r["task"] for r in records}) != 10:
            raise ValueError("exact 180 selection drift")
        seals = {}
        for seal_path in spec["source_seals"]:
            for line in contained(seal_path).read_text().splitlines():
                digest, relative = line.split(None, 1)
                path = str(contained(relative))
                if path in seals and seals[path] != digest:
                    raise ValueError("conflicting sealed source")
                seals[path] = digest
        def read_sealed(path, loader):
            digest = sha(path)
            if seals.get(str(path)) != digest:
                raise ValueError(f"unsealed input: {path}")
            accessed[str(path)] = digest
            return loader(path)
        reader = ScopedLocalTeacherReader(contained(spec["teacher_root"]), selected, expected_sha256=seals)
        manifests = read_sealed(contained(spec["prediction_manifest"]), lambda p: json.loads(p.read_text()))
        if (not isinstance(manifests, list) or len(manifests) != len(reader.selection)
                or {r["task"] for r in manifests} != set(reader.selection)):
            raise ValueError("prediction manifest task drift")
        if diagnostic is not None:
            diagnostic.start(spec, card, read_sealed)
        all_rows, tasks = [], []
        with (run / "logs/audit.log").open("x") as log:
            for task in reader.selection:
                records_task = [r for r in records if r["task"] == task]
                teacher = reader.read_task(task)
                manifest = next(r for r in manifests if r["task"] == task)
                if (manifest["source_sequence_indices"] != [r["source_global_sequence_index"] for r in records_task]
                        or not np.array_equal(manifest["frame_rows"], teacher["frame_row"])):
                    raise ValueError("cached prediction/teacher ordering drift")
                def load_prediction(path):
                    with np.load(path, allow_pickle=False) as archive:
                        if set(archive.files) != set(PREDICTION_FIELDS):
                            raise ValueError("prediction fields drift")
                        return {key: archive[key] for key in PREDICTION_FIELDS}
                prediction = read_sealed(contained(spec["prediction_root"] + "/" + task + ".npz"), load_prediction)
                construction = read_sealed(contained(spec["construction_root"] + "/" + task + ".json"), load_json)
                if construction["parent_id"] != task.split("__")[0] or construction["geometry_realization"] != "c1_mixed":
                    raise ValueError("construction identity drift")
                rows = audit_task(records_task, teacher, prediction, construction)
                if diagnostic is not None:
                    diagnostic.task(rows, teacher, prediction)
                all_rows.extend(rows)
                task_stats = {"task": task, "observations": len(rows),
                    "unique_frames": len(np.unique(teacher["frame_row"])),
                    "visible_fragments": sum(r["visible_primitive_fragments"] for r in rows),
                    "axis_point_error_m": distribution([m["axis_control_point_mean_euclidean_m"]
                        for r in rows for m in r["oracle_geometry_matches_not_detections"]])}
                tasks.append(task_stats)
                (run / "previews" / (task + ".svg")).write_text(preview_svg(task, rows, teacher, prediction))
                log.write(json.dumps(task_stats) + "\n"); log.flush()
        accessed.update(reader.opened)
        if diagnostic is not None:
            accessed.update(getattr(diagnostic, "additional_reads", {}))
        matches = [m for r in all_rows for m in r["oracle_geometry_matches_not_detections"]]
        summary = {"observations": len(all_rows), "parents": len(tasks),
            "unique_node_identities_scoring_only": len({(r["task"], str(r["target_node_scoring_only"])) for r in all_rows}),
            "unique_source_frames_referenced_not_decoded": sum(t["unique_frames"] for t in tasks),
            "visible_fragments": len(matches), "all_target_incident_fragments_present": sum(r["all_target_incident_fragments_present"] for r in all_rows),
            "observations_with_foreign_fragments": sum(r["foreign_fragments_present"] > 0 for r in all_rows),
            "observations_with_angular_alias": sum(r["angular_alias_pairs_not_physical_overlap"] > 0 for r in all_rows),
            "construction_relations_exact": True, "visible_support_consistent": True,
            "cached_geometry_error_oracle_alignment": {key: distribution([m[key] for m in matches]) for key in (
                "axis_coordinate_mae_m", "axis_control_point_mean_euclidean_m", "half_axes_mae_m",
                "shape_exponent_mae", "existence_probability")},
            "source_chunk_files_read": len(reader.opened), "sensor_frames_decoded": 0,
            "model_inference": 0, "optimizer_steps": 0, "new_targets": 0, "training_ready": False,
            "limitations": ["Angular alias is not physical overlap or proof of a hidden connection.",
                "Positive fragment membership does not certify visible junction/terminal or physical endpoint.",
                "Hungarian assignment uses teacher relations in tie ordering; scoring oracle, not deployment association.",
                "No detection threshold, calibration or uncertainty validity selected; errors are on fit-only tiny population."]}
        if (summary["observations"], summary["parents"], summary["unique_node_identities_scoring_only"],
                summary["unique_source_frames_referenced_not_decoded"]) != (180, 10, 100, 900):
            raise ValueError("population drift")
        if diagnostic is not None:
            key = getattr(diagnostic, "summary_key", "axis_error_decomposition")
            summary[key] = diagnostic.summarize(all_rows, run)
            summary["sensor_frames_decoded"] = getattr(diagnostic, "sensor_frames_decoded", 0)
        write_json(run / "artifacts/observation_audit.json", all_rows)
        write_json(run / "artifacts/parent_audit.json", tasks)
        write_json(run / "artifacts/source_reads_sha256.json", {str(Path(p).relative_to(PROJECT_ROOT)): h for p, h in sorted(accessed.items())})
        for path, digest in {**accessed, **{str(contained(p)): h for p, h in frozen.items()}}.items():
            if sha(Path(path)) != digest:
                raise ValueError("source/tool changed during audit")
        summary["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        if summary["peak_rss_bytes"] > 4 * 1024**3 or time.monotonic() - started > wall_cap:
            raise RuntimeError("audit resource cap exceeded")
    except Exception:
        error = traceback.format_exc(); (run / "logs/error.log").write_text(error)
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, old_handler)
    write_json(run / "metrics/summary.json", {"status": "AUDIT_COMPLETE" if error is None else "AUDIT_FAIL",
        "elapsed_s": time.monotonic() - started, "result": summary, "error": error, "scientific_gate_pass": False})
    write_json(run / "RUN_STATE.json", {"state": "COMPLETED" if error is None else "FAILED", "run_id": run.name, "error": error})
    seal = run / "artifacts/evidence_sha256.txt"
    seal.write_text("".join(f"{sha(p)}  {p.relative_to(PROJECT_ROOT)}\n" for p in sorted(run.rglob("*")) if p.is_file() and p != seal))
    print(json.dumps({"result": summary, "error": error, "seal_sha256": sha(seal)}))
    return int(error is not None)


if __name__ == "__main__":
    raise SystemExit(main())
