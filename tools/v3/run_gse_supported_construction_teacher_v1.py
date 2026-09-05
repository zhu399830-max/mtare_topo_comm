#!/usr/bin/env python3
"""One sealed partial construction-teacher pilot, with no model or rerender."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import platform
import resource
import signal
import time
import traceback

import numpy as np
import scipy
import torch
import zarr

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.gse_supported_teacher_reader import ScopedSupportedTeacherReader, selected_source_index
from mtare_topo.governance import build_run_id, load_json, write_json
from mtare_topo.governance_supported_teacher import validate_supported_teacher_card
from mtare_topo.teacher.gse_supported_construction_teacher import build_task_targets


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def contained(relative):
    path = (PROJECT_ROOT / relative).resolve()
    if not path.is_relative_to(PROJECT_ROOT):
        raise ValueError("path escapes project")
    return path


def preview_svg(task, rows, teacher):
    """All observations and all candidate centers, not hand-picked success views."""
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="{90 + 210 * len(rows)}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="15" y="22" font-size="16">{html.escape(task)}: construction-support targets</text>',
        '<text x="15" y="44" font-size="12">Gray axes: existing five-frame visible fragments. Green center: supported; gray cross: UNKNOWN.</text>',
        '<text x="15" y="65" font-size="12">Support interval is an observability proxy, NOT physical connectivity or terminal-cap certification.</text>']
    for row_index, row in enumerate(rows):
        axes = teacher["axis_control_current_sensor_m"][row_index, teacher["primitive_mask"][row_index].astype(bool)]
        centers = np.asarray([r["center_current_sensor_m"] for r in row["regions"]], dtype=float).reshape(-1, 3)
        points = np.concatenate((axes.reshape(-1, 3), centers, np.zeros((1, 3))))
        for panel, dimensions in enumerate(((0, 1), (0, 2))):
            left, top = 15 + panel * 495, 85 + row_index * 210
            projected = points[:, dimensions]
            low, high = projected.min(0), projected.max(0)
            center = (low + high) / 2
            scale = min(465 / max(high[0] - low[0], 1), 160 / max(high[1] - low[1], 1))
            def xy(point):
                q = (np.asarray(point)[list(dimensions)] - center) * scale
                return left + 240 + q[0], top + 115 - q[1]
            def polyline(axis):
                return " ".join(f"{x:.3f},{y:.3f}" for x, y in map(xy, axis))
            label = f'row={row["row_index"]} {"XY" if panel == 0 else "XZ"}; {sum(r["center_valid"] for r in row["regions"])} supported / {len(row["regions"])} candidate centers'
            parts.extend((f'<rect x="{left}" y="{top}" width="480" height="200" fill="none" stroke="#ccc"/>',
                f'<text x="{left + 5}" y="{top + 16}" font-size="11">{html.escape(label)}</text>'))
            for axis in axes:
                parts.append(f'<polyline points="{polyline(axis)}" fill="none" stroke="#9ca3af" stroke-width="1.5"/>')
            for region in row["regions"]:
                x, y = xy(region["center_current_sensor_m"])
                if region["center_valid"]:
                    parts.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="3.5" fill="#168447"/>')
                    label = region["event_target"] or "event?"
                    parts.append(f'<text x="{x + 4:.3f}" y="{y - 4:.3f}" font-size="8" fill="#116333">{html.escape(label)}</text>')
                else:
                    parts.append(f'<path d="M{x-3:.3f},{y-3:.3f}l6,6m-6,0l6,-6" stroke="#888"/>')
            x, y = xy(np.zeros(3))
            parts.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="2.5" fill="#176dc1"/>')
    return "\n".join(parts + ["</svg>"])


def aggregate(tasks, all_rows):
    numeric = ("observations", "unique_sensor_frames", "visible_fragments", "candidate_region_instances",
               "center_labels", "member_positive", "member_negative", "observations_with_nonincident_source_ambiguity")
    result = {key: sum(t["counts"][key] for t in tasks) for key in numeric}
    result["parents"] = len(tasks)
    result["events"] = {event: sum(t["counts"]["events"][event] for t in tasks) for event in ("corridor", "junction", "terminal")}
    result["independent_construction_centers_teacher_only"] = len({(r["task"], g["construction_node_id_teacher_only"])
        for r in all_rows for g in r["regions"] if g["center_valid"]})
    result["independent_event_instances_teacher_only"] = {event: len({(r["task"], g["construction_node_id_teacher_only"])
        for r in all_rows for g in r["regions"] if g["event_target"] == event}) for event in result["events"]}
    result["unknown_event_candidates"] = sum(not g["event_valid"] for r in all_rows for g in r["regions"])
    result["unknown_member_reasons"] = {}
    for row in all_rows:
        for group in row["regions"]:
            for member in group["members"]:
                for reason in member["unknown_reasons"]:
                    result["unknown_member_reasons"][reason] = result["unknown_member_reasons"].get(reason, 0) + 1
    return result


def validate_task_output(output, source):
    """Recount saved records; summary counters cannot conceal bad labels."""
    rows = output["rows"]
    if len(rows) != len(source["records"]):
        raise ValueError("target row population mismatch")
    groups = [g for r in rows for g in r["regions"]]
    if any(g["event_target"] == "terminal" for g in groups):
        raise ValueError("terminal target produced without source cap evidence")
    if any(g["event_valid"] != (g["event_target"] in ("corridor", "junction")) for g in groups):
        raise ValueError("event label/valid mask disagreement")
    if any(len(g["directional_member_valid"]) != 64 or len(g["directional_member_target"]) != 64 for g in groups):
        raise ValueError("member population must remain 64 directional tokens")
    actual = {"observations": len(rows), "unique_sensor_frames": len({f for r in rows for f in r["frame_rows"]}),
        "visible_fragments": sum(r["visible_fragments"] for r in rows), "candidate_region_instances": len(groups),
        "center_labels": sum(g["center_valid"] for g in groups),
        "events": {event: sum(g["event_target"] == event for g in groups) for event in ("corridor", "junction", "terminal")},
        "member_positive": sum(sum(v and t == 1 for v, t in zip(g["directional_member_valid"], g["directional_member_target"])) for g in groups),
        "member_negative": sum(sum(v and t == 0 for v, t in zip(g["directional_member_valid"], g["directional_member_target"])) for g in groups),
        "observations_with_nonincident_source_ambiguity": sum(bool(r["nonincident_source_pairs_teacher_only"]) for r in rows)}
    if actual != output["counts"]:
        raise ValueError("output counts differ from actual saved target records")
    if (not np.array_equal([r["frame_rows"] for r in rows], source["teacher"]["frame_row"])
            or actual["visible_fragments"] != int(source["teacher"]["primitive_mask"].sum())
            or actual["unique_sensor_frames"] != len(source["sensor_frame_rows"])):
        raise ValueError("saved targets differ from source frame/support population")


def execute(spec, run):
    """Fresh-run lifecycle; failed evidence is sealed and cannot be retried."""
    run = Path(run).resolve()
    if (run != PROJECT_ROOT / "results/gate3_semantics" / build_run_id(spec)
            or load_json(run / "RUN_STATE.json")["state"] != "CREATED_NOT_EXECUTED"
            or load_json(run / "config/run_spec.json") != spec):
        raise RuntimeError("requires exact fresh run; no overwrite or retry")
    started, error, summary = time.monotonic(), None, {}
    reader, frozen = None, {}
    write_json(run / "RUN_STATE.json", {"state": "RUNNING", "run_id": run.name})
    def deadline(signum, frame):
        raise TimeoutError("frozen 600-second CPU teacher deadline exceeded")
    old_handler = signal.signal(signal.SIGALRM, deadline)
    signal.alarm(600)
    try:
        card = load_json(contained(spec["data_card"]))
        report = validate_supported_teacher_card(card)
        if (not report.passed or load_json(run / "config/data_card.json") != card
                or spec["operation"] != "teacher_generation" or spec["wall_time_cap_s"] != 600
                or spec["native_geometry"] != card["native_geometry"]):
            raise ValueError(f"frozen pilot/card mismatch: {report.errors}")
        frozen = {**card["sealed_sources"], **spec["source_sha256"]}
        for relative, digest in frozen.items():
            if sha(contained(relative)) != digest:
                raise ValueError(f"frozen source/tool drift: {relative}")
        versions = {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                    "zarr": zarr.__version__, "scipy": scipy.__version__}
        if versions != spec["expected_versions"]:
            raise ValueError(f"environment drift: {versions}")
        write_json(run / "config/execution_environment.json", {**versions, "device": "cpu", "platform": platform.platform()})
        expected = selected_source_index(PROJECT_ROOT, card)
        reader = ScopedSupportedTeacherReader(PROJECT_ROOT, card, expected_sha256=expected)
        all_rows, tasks = [], []
        with (run / "logs/teacher.log").open("x") as log:
            for task in sorted(reader.selection):
                log.write(json.dumps({"task": task, "state": "READING", "elapsed_s": time.monotonic() - started}) + "\n"); log.flush()
                source = reader.read_task(task)
                output = build_task_targets(**{key: source[key] for key in
                    ("teacher", "sensor", "sensor_frame_rows", "construction", "codebook")})
                if (not output["old_teacher_all_fields_reconstructed_exactly"] or output["capacity_ready"]
                        or len(output["rows"]) != len(source["records"])):
                    raise ValueError("target/parity population contract drift")
                validate_task_output(output, source)
                for row, record in zip(output["rows"], source["records"]):
                    if row["source_global_sequence_index"] != record["source_global_sequence_index"]:
                        raise ValueError("new label ordering differs from frozen original selection")
                    row.update({"task": task, "row_index": record["row_index"]})
                write_json(run / "artifacts" / (task + ".json"), output)
                (run / "previews" / (task + ".svg")).write_text(preview_svg(task, output["rows"], source["teacher"]))
                task_summary = {"task": task, "counts": output["counts"], "old_teacher_exact": True}
                tasks.append(task_summary); all_rows.extend(output["rows"])
                log.write(json.dumps({**task_summary, "state": "TARGETS_WRITTEN", "elapsed_s": time.monotonic() - started}) + "\n"); log.flush()
                print(json.dumps({"task": task, "counts": output["counts"]}), flush=True)
                if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > 4 * 1024**3:
                    raise RuntimeError("4GiB peak host RAM cap exceeded")
                del source, output
        summary = aggregate(tasks, all_rows)
        expected_counts = spec["expected_counts"]
        for key, frozen_key in (("observations", "observations"), ("parents", "parents"),
                                ("unique_sensor_frames", "unique_frames"), ("visible_fragments", "visible_fragments")):
            if summary[key] != expected_counts[frozen_key]:
                raise ValueError(f"population drift: {key}={summary[key]}")
        if summary["events"]["terminal"] != 0:
            raise ValueError("terminal target produced without frozen cap evidence implementation")
        summary.update({"capacity_ready": False, "three_class_capacity_reason": "TERMINAL_CAP_EVIDENCE_NOT_IMPLEMENTED",
            "physical_connectivity_certified": False, "old_teacher_exact": True,
            "model_inference_frames": 0, "optimizer_steps": 0, "scan_rerenders": 0,
            "existing_ray_records_decoded": summary["unique_sensor_frames"] * 16 * 720,
            "source_files_opened": len(reader.opened), "interval_proxy_not_continuous_visibility_proof": True})
        write_json(run / "artifacts/observation_targets.json", all_rows)
        write_json(run / "artifacts/parent_counts.json", tasks)
        (run / "summary.md").write_text(
            "# Supported-construction partial teacher\n\n"
            f"Generated labels on {summary['observations']} original observations, {summary['parents']} C01 parents and {summary['unique_sensor_frames']} existing frames.\n\n"
            f"Supported center instances: {summary['center_labels']}; independent centers: {summary['independent_construction_centers_teacher_only']}. Event labels: {summary['events']}.\n\n"
            "This is construction truth filtered by a five-frame axis-support interval proxy. It is not a physical connectivity proof or a model score. All unknowns and all observation previews are retained.\n\n"
            "Terminal cap evidence is not implemented: terminal remains UNKNOWN and complete three-class capacity is NOT READY. No model inference, optimization, rerender or scientific Gate PASS.\n")
        for path, digest in {**reader.opened, **{str(contained(p)): h for p, h in frozen.items()}}.items():
            if sha(Path(path)) != digest:
                raise ValueError(f"source/tool changed during teacher generation: {path}")
        summary["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        summary["output_bytes_before_final_metadata"] = sum(p.stat().st_size for p in run.rglob("*") if p.is_file())
        if (summary["peak_rss_bytes"] > 4 * 1024**3 or time.monotonic() - started > 600
                or summary["output_bytes_before_final_metadata"] > 249_000_000):
            raise RuntimeError("frozen CPU/RAM/0.25GB result cap exceeded")
    except Exception:
        error = traceback.format_exc()
        (run / "logs/error.log").write_text(error)
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, old_handler)
        if reader is not None:
            write_json(run / "artifacts/source_reads_sha256.json", {str(Path(p).relative_to(PROJECT_ROOT)): digest
                for p, digest in sorted(reader.opened.items())})
    status = "SUPPORTED_CONSTRUCTION_PARTIAL_TEACHER_GENERATED" if error is None else "TEACHER_GENERATION_FAIL"
    write_json(run / "metrics/summary.json", {"status": status, "elapsed_s": time.monotonic() - started,
        "result": summary, "error": error, "scientific_gate_pass": False})
    write_json(run / "RUN_STATE.json", {"state": "COMPLETED" if error is None else "FAILED", "run_id": run.name, "error": error})
    seal = run / "artifacts/evidence_sha256.txt"
    seal.write_text("".join(f"{sha(p)}  {p.relative_to(PROJECT_ROOT)}\n" for p in sorted(run.rglob("*")) if p.is_file() and p != seal))
    print(json.dumps({"status": status, "result": summary, "error": error, "seal_sha256": sha(seal)}), flush=True)
    return int(error is not None)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    return execute(load_json(args.spec), args.run_dir)


if __name__ == "__main__":
    raise SystemExit(main())
