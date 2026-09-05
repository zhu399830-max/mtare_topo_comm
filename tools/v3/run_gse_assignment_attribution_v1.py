#!/usr/bin/env python3
"""Attribute cached region assignments; never load weights or execute a model."""
import argparse
from dataclasses import fields
import hashlib
import html
import json
import os
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
from mtare_topo.data.gse_assignment_inputs import load_assignment_inputs
from mtare_topo.evaluation.gse_assignment_attribution import diagnose_assignments
from mtare_topo.evaluation.gse_partial_structure import evaluate_partial_structure
from mtare_topo.governance import build_run_id, load_json, write_json
from mtare_topo.governance_assignment_attribution import validate_assignment_attribution_card
from mtare_topo.representation.gse_region_queries import RegionPrediction, RegionTargets


BRANCHES = ("gt_axes", "predicted_axes", "predicted_no_relations")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contained(relative):
    path = (PROJECT_ROOT / relative).resolve()
    if not path.is_relative_to(PROJECT_ROOT): raise ValueError("path outside project")
    return path


def sliced(value, indices):
    return type(value)(**{f.name: getattr(value, f.name)[indices] for f in fields(value)})


def preview(branch, task, indices, comparisons):
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="{85 + 190 * len(indices)}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="15" y="23" font-size="15">{html.escape(branch + " / " + task)}: fixed final snapshot, sensor coordinates (m)</text>',
        '<text x="15" y="45" font-size="12">Black: known center; blue: geometry assignment; orange: label-conditioned joint assignment.</text>',
        '<text x="15" y="64" font-size="12">Joint uses teacher answers: diagnosis only, NOT deployable detection or replacement scores.</text>']
    for panel_row, index in enumerate(indices):
        records = [r for r in comparisons if r["observation_index"] == index]
        points = [np.zeros(3)] + [np.asarray(r[k], dtype=float) for r in records
            for k in ("teacher_center_m", "geometry_center_m", "joint_center_m") if r[k] is not None]
        cloud = np.asarray(points)
        for panel, dims in enumerate(((0, 1), (0, 2))):
            left, top = 15 + panel * 495, 80 + panel_row * 190
            low, high = cloud[:, dims].min(0), cloud[:, dims].max(0)
            middle = (low + high) / 2
            scale = min(455 / max(high[0] - low[0], 1), 130 / max(high[1] - low[1], 1))
            def xy(point):
                q = (np.asarray(point)[list(dims)] - middle) * scale
                return left + 240 + q[0], top + 105 - q[1]
            parts.extend((f'<rect x="{left}" y="{top}" width="480" height="180" fill="none" stroke="#ccc"/>',
                f'<text x="{left+5}" y="{top+16}" font-size="11">observation={index}, {"XY" if panel == 0 else "XZ"}; known targets={len(records)}</text>'))
            for r in records:
                truth = r["teacher_center_m"]
                for key, color in (("geometry_center_m", "#2166ac"), ("joint_center_m", "#d95f02")):
                    if r[key] is None: continue
                    x, y = xy(r[key])
                    if truth is not None:
                        a, b = xy(truth)
                        parts.append(f'<line x1="{a:.3f}" y1="{b:.3f}" x2="{x:.3f}" y2="{y:.3f}" stroke="{color}" stroke-opacity="0.5"/>')
                    parts.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="3" fill="{color}"/>')
                if truth is not None:
                    x, y = xy(truth)
                    parts.append(f'<path d="M{x-3:.3f},{y-3:.3f}l6,6m-6,0l6,-6" stroke="black"/>')
    return "\n".join(parts + ["</svg>"])


def analyze_and_save(run, loaded):
    data = loaded["data"]
    manifest = data["manifest"]
    output = {}
    for branch in BRANCHES:
        key = "gt" if branch == "gt_axes" else "predicted"
        prediction, target = loaded["predictions"][branch], data["targets"][key]
        bridge = [r["target_transport"] for r in data["transport"][key]]
        original = loaded["training_summary"]["result"]["evaluation"]["final"][branch]
        check = evaluate_partial_structure(prediction, target, membership_threshold=.5,
            manifest=manifest, direction_bridge_ledger=bridge)
        if check.summary != original["aggregate"]:
            raise ValueError("original aggregate not exactly reproduced: " + branch)
        saved = loaded["saved_scoring"][branch]
        if (not torch.equal(check.scored_member_mask, saved["scored_member_mask"])
                or not torch.equal(check.unique_center_query, saved["unique_center_query"])):
            raise ValueError("original scoring masks/queries not reproduced: " + branch)
        tasks = sorted({r["task"] for r in manifest})
        if set(original["parents"]) != set(tasks): raise ValueError("parent population drift")
        for task in tasks:
            indices = [i for i, r in enumerate(manifest) if r["task"] == task]
            parent = evaluate_partial_structure(sliced(prediction, indices), sliced(target, indices),
                membership_threshold=.5, manifest=[manifest[i] for i in indices],
                direction_bridge_ledger=[bridge[i] for i in indices]).summary
            if parent != original["parents"][task]: raise ValueError("original parent not exactly reproduced: " + task)
        result = diagnose_assignments(prediction, target, manifest, data["transport"][key], loaded["last_epoch_batches"])
        if result["geometry_summary"] != original["aggregate"]: raise ValueError("diagnostic geometry drift")
        write_json(run / "artifacts" / (branch + "__attribution.json"), result)
        write_json(run / "artifacts" / (branch + "__junctions.json"), result["junction_table"])
        for task in tasks:
            indices = [i for i, r in enumerate(manifest) if r["task"] == task]
            (run / "previews" / (branch + "__" + task + ".svg")).write_text(
                preview(branch, task, indices, result["target_comparisons"]))
        # Keep detailed target rows in artifacts, not duplicated in summary.
        output[branch] = {k: v for k, v in result.items() if k not in ("target_comparisons", "junction_table")}
        output[branch]["junction_table_rows"] = len(result["junction_table"])
        record = {"branch": branch, "original_aggregate_masks_parents_reproduced": True,
            "target_rows": len(result["target_comparisons"]), "junction_rows": len(result["junction_table"])}
        with (run / "logs/diagnostic.jsonl").open("a") as log:
            log.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
    write_json(run / "artifacts/manifest.json", manifest)
    write_json(run / "artifacts/last_epoch_batches.json", loaded["last_epoch_batches"])
    return {"branches": output, "cached_prediction_observations": len(BRANCHES) * len(manifest),
        "model_inference": 0, "checkpoint_reads": 0, "optimizer_steps": 0, "new_sensor_frames": 0,
        "original_scores_reproduced": True, "joint_diagnostic_only": True}


def execute(spec, run):
    run = Path(run).resolve()
    if (run != PROJECT_ROOT / "results/gate3_semantics" / build_run_id(spec)
            or load_json(run / "RUN_STATE.json")["state"] != "CREATED_NOT_EXECUTED"
            or load_json(run / "config/run_spec.json") != spec):
        raise RuntimeError("requires exact fresh run; no overwrite/retry")
    started, result, error, reads = time.monotonic(), {}, None, {}
    write_json(run / "RUN_STATE.json", {"state": "RUNNING", "run_id": run.name})
    def deadline(signum, frame): raise TimeoutError("fixed300s attribution deadline exceeded")
    old_handler = signal.signal(signal.SIGALRM, deadline); signal.alarm(300)
    try:
        card = load_json(contained(spec["data_card"]))
        report = validate_assignment_attribution_card(card)
        if (not report.passed or spec["operation"] != "data_export" or spec["wall_time_cap_s"] != 300
                or load_json(run / "config/data_card.json") != card or spec["attribution"] != card["attribution"]):
            raise ValueError(f"frozen attribution scope drift: {report.errors}")
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
            raise ValueError("CPU-only attribution requires CUDA_VISIBLE_DEVICES empty from start")
        frozen = {**card["sealed_sources"], **spec["source_sha256"]}
        for path, checksum in frozen.items():
            if sha(contained(path)) != checksum: raise ValueError("source drift: " + path)
        versions = {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__, "zarr": zarr.__version__, "scipy": scipy.__version__}
        if versions != spec["expected_versions"]: raise ValueError("frozen environment drift")
        write_json(run / "config/execution_environment.json", {**versions, "platform": platform.platform(),
            "device": "cpu", "CUDA_VISIBLE_DEVICES": "", "model_execution": False})
        loaded = load_assignment_inputs(PROJECT_ROOT, card)
        reads = loaded["read_hashes"]
        result = analyze_and_save(run, loaded)
        if result["cached_prediction_observations"] != 540: raise ValueError("exact3x180 cache population required")
        result["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        if result["peak_rss_bytes"] > 4 * 1024**3 or time.monotonic() - started > 300:
            raise RuntimeError("frozen CPU resource budget exceeded")
        if sum(p.stat().st_size for p in run.rglob("*") if p.is_file()) > 490_000_000:
            raise RuntimeError("frozen output budget exceeded")
        for path, checksum in {**frozen, **reads}.items():
            if sha(contained(path)) != checksum: raise ValueError("source changed during attribution")
    except Exception:
        error = traceback.format_exc(); (run / "logs/error.log").write_text(error)
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, old_handler)
    write_json(run / "artifacts/source_reads_sha256.json", reads)
    status = "CACHED_ASSIGNMENT_ATTRIBUTION_COMPLETE" if error is None else "CACHED_ASSIGNMENT_ATTRIBUTION_FAIL"
    write_json(run / "metrics/summary.json", {"status": status, "elapsed_s": time.monotonic() - started,
        "result": result, "error": error, "scientific_gate_pass": False})
    write_json(run / "RUN_STATE.json", {"state": "COMPLETED" if error is None else "FAILED", "run_id": run.name, "error": error})
    seal = run / "artifacts/evidence_sha256.txt"
    seal.write_text("".join(f"{sha(p)}  {p.relative_to(PROJECT_ROOT)}\n" for p in sorted(run.rglob("*")) if p.is_file() and p != seal))
    print(json.dumps({"status": status, "error": error, "elapsed_s": time.monotonic() - started, "seal_sha256": sha(seal)}), flush=True)
    return int(error is not None)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    return execute(load_json(args.spec), args.run_dir)


if __name__ == "__main__": raise SystemExit(main())
