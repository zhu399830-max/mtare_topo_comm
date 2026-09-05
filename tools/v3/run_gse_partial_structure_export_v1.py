#!/usr/bin/env python3
"""One bounded cache/identity/partial-target conversion, no model execution."""
import argparse
from dataclasses import fields
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
from mtare_topo.data.gse_partial_structure_cache import load_partial_structure_cache
from mtare_topo.governance import build_run_id, load_json, write_json
from mtare_topo.governance_partial_structure import validate_partial_structure_export_card
from mtare_topo.representation.gse_region_queries import RegionTargets, tokens_from_axes
from mtare_topo.representation.gse_region_target_bridge import align_axis_directions, partial_region_targets
from run_gse_supported_construction_teacher_v1 import sha


def contained(relative):
    path = (PROJECT_ROOT / relative).resolve()
    if not path.is_relative_to(PROJECT_ROOT):
        raise ValueError("path outside project")
    return path


def original_counts(cache):
    groups = [g for row in cache.partial_targets for g in row["regions"]]
    return {"observations": len(cache.manifest), "parents": len({r["task"] for r in cache.manifest}),
        "unique_frames": len({(r["task"], f) for r in cache.manifest for f in r["frame_rows"]}),
        "visible_fragments": int(cache.loss_only_gt_mask.sum()),
        "center_labels": sum(g["center_valid"] for g in groups),
        "member_positive": sum(sum(v and t == 1 for v, t in zip(g["directional_member_valid"], g["directional_member_target"])) for g in groups),
        "member_negative": sum(sum(v and t == 0 for v, t in zip(g["directional_member_valid"], g["directional_member_target"])) for g in groups),
        "events": {event: sum(g["event_target"] == event for g in groups) for event in ("corridor", "junction", "terminal")}}


def convert_cache(cache):
    """No detector instantiation: only detached geometry/target tensor transforms."""
    gt = torch.tensor(cache.gt_axes)
    pred = torch.tensor(cache.prediction_axes)
    mask = torch.tensor(cache.loss_only_gt_mask)
    arrays = {"gt_axes": cache.gt_axes, "predicted_axes": cache.prediction_axes,
              "loss_only_gt_mask": cache.loss_only_gt_mask}
    per_observation, summaries = {}, {}
    for branch, axes in (("gt", gt), ("predicted", pred)):
        alignment = align_axis_directions(axes, gt, mask)
        bridge = partial_region_targets(cache.partial_targets, alignment, dtype=axes.dtype)
        tokens = tokens_from_axes(axes)
        target = bridge.targets
        for field in fields(RegionTargets):
            arrays[branch + "__" + field.name] = getattr(target, field.name).cpu().numpy()
        arrays[branch + "__teacher_direction"] = alignment.teacher_direction.cpu().numpy()
        arrays[branch + "__direction_valid"] = alignment.valid.cpu().numpy()
        arrays[branch + "__input_direction_valid"] = tokens.valid.cpu().numpy()
        records = []
        for i, identity in enumerate(cache.manifest):
            usable = target.member_valid[i] & tokens.valid[i][None]
            known = target.center_valid[i] | target.event_valid[i] | target.member_valid[i].any(-1)
            eligible = target.center_valid[i] | target.event_valid[i] | (usable & (target.members[i] == 1)).any(-1)
            records.append({**identity, "direction_alignment": alignment.ledger[i],
                "direction_reasons": alignment.reasons[i], "teacher_direction": alignment.teacher_direction[i].tolist(),
                "target_transport": bridge.ledger[i], "numeric_input_valid_queries": int(tokens.valid[i].sum()),
                "known_region_targets": int(known.sum()), "eligible_region_targets": int(eligible.sum()),
                "region_cardinality_unmatched_lower_bound": max(0, int(eligible.sum()) - int(tokens.valid[i].sum())),
                "usable_member_positive": int((usable & (target.members[i] == 1)).sum()),
                "usable_member_negative": int((usable & (target.members[i] == 0)).sum()),
                "unsupported_transported_members": int((target.member_valid[i] & ~tokens.valid[i][None]).sum())})
        keys = ("numeric_input_valid_queries", "known_region_targets", "eligible_region_targets",
                "region_cardinality_unmatched_lower_bound", "usable_member_positive", "usable_member_negative", "unsupported_transported_members")
        summary = {key: sum(r[key] for r in records) for key in keys}
        summary["direction_alignment"] = {key: sum(r["direction_alignment"][key] for r in records) for key in records[0]["direction_alignment"]}
        summary["target_transport"] = {key: sum(r["target_transport"][key] for r in records) for key in records[0]["target_transport"]}
        summary["max_valid_centers_per_observation"] = int(target.center_valid.sum(1).max())
        summary["all_labels_incomplete"] = not bool(target.label_complete.any())
        per_observation[branch], summaries[branch] = records, summary
    if not np.array_equal(arrays["gt__centers_m"], arrays["predicted__centers_m"]) or not np.array_equal(arrays["gt__event_valid"], arrays["predicted__event_valid"]):
        raise ValueError("geometry correspondence must not change region centers/events")
    return arrays, per_observation, summaries


def preview(records):
    tasks = sorted({r["task"] for r in records["gt"]})
    text = ['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="500">',
            '<rect width="100%" height="100%" fill="white"/>',
            '<text x="20" y="28" font-size="18">Partial targets: raw vs usable directional supervision (not model performance)</text>',
            '<text x="20" y="57" font-size="13">Parent | original positives / negatives | GT usable | prediction usable</text>']
    for i, task in enumerate(tasks):
        subset = {b: [r for r in rows if r["task"] == task] for b, rows in records.items()}
        original = [sum(r["target_transport"]["original_member_" + sign] for r in subset["gt"]) for sign in ("positive", "negative")]
        usable = {b: [sum(r["usable_member_" + sign] for r in rows) for sign in ("positive", "negative")] for b, rows in subset.items()}
        label = f"{task}: {original} | {usable['gt']} | {usable['predicted']}"
        text.append(f'<text x="20" y="{90 + i * 34}" font-size="13">{html.escape(label)}</text>')
    text.append('<text x="20" y="475" font-size="13">Unknown directions remain UNKNOWN. No new frames, backbone, optimization or scientific PASS.</text>')
    return "\n".join(text + ["</svg>"])


def execute(spec, run):
    run = Path(run).resolve()
    if (run != PROJECT_ROOT / "results/gate3_semantics" / build_run_id(spec)
            or load_json(run / "RUN_STATE.json")["state"] != "CREATED_NOT_EXECUTED"
            or load_json(run / "config/run_spec.json") != spec):
        raise RuntimeError("requires fresh exact run; no overwrite/retry")
    started, error, summary, reads = time.monotonic(), None, {}, {}
    write_json(run / "RUN_STATE.json", {"state": "RUNNING", "run_id": run.name})
    def deadline(signum, frame):
        raise TimeoutError("120s export deadline exceeded")
    previous_handler = signal.signal(signal.SIGALRM, deadline); signal.alarm(120)
    try:
        card = load_json(contained(spec["data_card"]))
        report = validate_partial_structure_export_card(card)
        if (not report.passed or spec["operation"] != "data_export" or spec["wall_time_cap_s"] != 120
                or load_json(run / "config/data_card.json") != card):
            raise ValueError(f"exact export scope drift: {report.errors}")
        frozen = {**card["sealed_sources"], **spec["source_sha256"]}
        for path, digest in frozen.items():
            if sha(contained(path)) != digest: raise ValueError(f"frozen source drift: {path}")
        versions = {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__, "zarr": zarr.__version__, "scipy": scipy.__version__}
        if versions != spec["expected_versions"]: raise ValueError("frozen environment drift")
        write_json(run / "config/execution_environment.json", {**versions, "device": "cpu", "platform": platform.platform()})
        cache = load_partial_structure_cache(project_root=PROJECT_ROOT, sources=card["sources"])
        reads = cache.read_hashes
        selection = [{k: row[k] for k in ("task", "row_index", "source_global_sequence_index")} for row in cache.manifest]
        if selection != card["selected_rows"]: raise ValueError("cache ordering differs from frozen selection")
        counts = original_counts(cache)
        for key, value in counts.items():
            if spec["expected_counts"].get(key) != value: raise ValueError(f"original count drift: {key}")
        arrays, records, effective = convert_cache(cache)
        np.savez_compressed(run / "artifacts/partial_training_inputs.npz", **arrays)
        write_json(run / "artifacts/manifest.json", list(cache.manifest))
        write_json(run / "artifacts/target_transport.json", records)
        write_json(run / "artifacts/identity_provenance.json", cache.provenance)
        (run / "previews/parent_target_transport.svg").write_text(preview(records))
        with (run / "logs/export.log").open("x") as log:
            for branch in records:
                for row in records[branch]: log.write(json.dumps({"branch": branch, **row}) + "\n")
        summary = {"original_counts": counts, "effective_counts": effective, "capacity_ready": False,
            "full_three_class_ready": False, "source_files_read": len(reads),
            "new_sensor_frames": 0, "model_inference": 0, "optimizer_steps": 0,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
        for path, digest in {**frozen, **reads}.items():
            if sha(contained(path)) != digest: raise ValueError("source changed during export")
        if (summary["peak_rss_bytes"] > 4 * 1024**3 or time.monotonic() - started > 120
                or sum(p.stat().st_size for p in run.rglob("*") if p.is_file()) > 249_000_000):
            raise RuntimeError("CPU/RAM/output budget exceeded")
    except Exception:
        error = traceback.format_exc(); (run / "logs/error.log").write_text(error)
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, previous_handler)
    write_json(run / "artifacts/source_reads_sha256.json", reads)
    status = "PARTIAL_STRUCTURE_INPUTS_EXPORTED" if error is None else "PARTIAL_STRUCTURE_EXPORT_FAIL"
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
