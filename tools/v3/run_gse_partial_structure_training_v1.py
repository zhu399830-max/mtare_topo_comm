#!/usr/bin/env python3
"""Fixed-budget three-head CUDA capacity diagnosis, not detector qualification."""
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
from mtare_topo.data.gse_partial_training_inputs import load_training_inputs
from mtare_topo.evaluation.gse_partial_structure import evaluate_partial_structure
from mtare_topo.governance import build_run_id, load_json, write_json
from mtare_topo.governance_partial_structure_training import validate_partial_structure_training_card
from mtare_topo.representation.gse_region_queries import RegionQueryHead, RegionPrediction, RegionTargets, tokens_from_axes
from mtare_topo.representation.gse_partial_structure_training import (
    BRANCHES, PartialStructureExample, PartialTrainingConfig, train_partial_structure,
)
from run_gse_supported_construction_teacher_v1 import sha


def contained(relative):
    path = (PROJECT_ROOT / relative).resolve()
    if not path.is_relative_to(PROJECT_ROOT): raise ValueError("path outside project")
    return path


def slice_target(target, selection):
    return RegionTargets(**{f.name: getattr(target, f.name)[selection] for f in fields(RegionTargets)})


def predict(head, axes, batch_size, device):
    head.eval(); pieces = []
    with torch.no_grad():
        for start in range(0, len(axes), batch_size):
            value = head(tokens_from_axes(axes[start:start + batch_size].to(device)))
            pieces.append({f.name: getattr(value, f.name).detach().cpu() for f in fields(RegionPrediction)})
    return RegionPrediction(**{f.name: torch.cat([p[f.name] for p in pieces]) for f in fields(RegionPrediction)})


def preview(branch, stage, task, indices, prediction, target):
    colors = ("#2166ac", "#d73027", "#e08214")
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="{90 + 190 * len(indices)}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="15" y="24" font-size="16">{html.escape(branch + " / " + stage + " / " + task)}</text>',
        '<text x="15" y="47" font-size="12">All valid proposals: blue corridor / red junction / orange terminal. Black crosses: known construction centers.</text>',
        '<text x="15" y="68" font-size="12">No GT filtering of proposals. Incomplete labels: this is candidate capacity, NOT autonomous detection.</text>']
    for row_number, row in enumerate(indices):
        valid = prediction.query_supported[row]
        pred = prediction.centers_m[row, valid].numpy()
        classes = prediction.event_logits[row, valid].argmax(-1).numpy()
        truth = target.centers_m[row, target.center_valid[row]].numpy()
        combined = np.concatenate((pred, truth, np.zeros((1, 3))))
        for panel, dims in enumerate(((0, 1), (0, 2))):
            left, top = 15 + 495 * panel, 85 + 190 * row_number
            low, high = combined[:, dims].min(0), combined[:, dims].max(0)
            middle = (low + high) / 2
            scale = min(460 / max(high[0] - low[0], 1), 135 / max(high[1] - low[1], 1))
            def xy(point):
                p = (point[list(dims)] - middle) * scale
                return left + 240 + p[0], top + 103 - p[1]
            parts.extend((f'<rect x="{left}" y="{top}" width="480" height="180" fill="none" stroke="#ccc"/>',
                f'<text x="{left+5}" y="{top+16}" font-size="11">observation={row} {"XY" if panel == 0 else "XZ"}; queries={len(pred)}, centers={len(truth)}</text>'))
            for point, label in zip(pred, classes):
                x, y = xy(point)
                parts.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="2.5" fill="{colors[label]}" fill-opacity="0.7"/>')
            for point in truth:
                x, y = xy(point)
                parts.append(f'<path d="M{x-4:.3f},{y-4:.3f}l8,8m-8,0l8,-8" stroke="black" stroke-width="1.5"/>')
    return "\n".join(parts + ["</svg>"])


def evaluate_and_save(run, branch, stage, head, data, config, threshold):
    key = "gt" if branch == "gt_axes" else "predicted"
    axes, target = data["axes"][key], data["targets"][key]
    prediction = predict(head, axes, config.batch_size, config.device)
    bridge = [r["target_transport"] for r in data["transport"][key]]
    evaluated = evaluate_partial_structure(prediction, target, membership_threshold=threshold,
        manifest=data["manifest"], direction_bridge_ledger=bridge)
    prefix = branch + "__" + stage
    np.savez_compressed(run / "artifacts" / (prefix + "__predictions.npz"),
        **{f.name: getattr(prediction, f.name).numpy() for f in fields(RegionPrediction)},
        scored_member_mask=evaluated.scored_member_mask.numpy(), unique_center_query=evaluated.unique_center_query.numpy())
    write_json(run / "artifacts" / (prefix + "__observations.json"), list(evaluated.observations))
    parents = {}
    for task in sorted({r["task"] for r in data["manifest"]}):
        indices = [i for i, r in enumerate(data["manifest"]) if r["task"] == task]
        selection = torch.tensor(indices)
        subset = RegionPrediction(**{f.name: getattr(prediction, f.name)[selection] for f in fields(RegionPrediction)})
        parents[task] = evaluate_partial_structure(subset, slice_target(target, selection), membership_threshold=threshold,
            manifest=[data["manifest"][i] for i in indices], direction_bridge_ledger=[bridge[i] for i in indices]).summary
        (run / "previews" / (prefix + "__" + task + ".svg")).write_text(preview(branch, stage, task, indices, prediction, target))
    write_json(run / "metrics" / (prefix + ".json"), {"aggregate": evaluated.summary, "parents": parents})
    return {"aggregate": evaluated.summary, "parents": parents}


def run_training(run, data, settings, evaluation, *, step_callback=None):
    config = PartialTrainingConfig(settings["seed"], settings["steps_per_branch"], settings["batch_size"], settings["lr"], settings["device"])
    hidden = settings["hidden"]
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(config.seed)
        initial = RegionQueryHead(hidden=hidden)
    if sum(p.numel() for p in initial.parameters()) != settings["parameters_per_head"]:
        raise ValueError("head parameter count drift")
    initial_state = {k: v.detach().clone() for k, v in initial.state_dict().items()}
    torch.save(initial_state, run / "artifacts/shared_initial_state.pt")
    outputs = {stage: {} for stage in ("initial", "final")}
    for branch in BRANCHES:
        initial.use_relations = settings["use_relations"][branch]
        outputs["initial"][branch] = evaluate_and_save(run, branch, "initial", initial.to(config.device), data, config, evaluation["member_threshold"])
    del initial
    examples = {}
    for key in ("gt", "predicted"):
        examples[key] = [PartialStructureExample(data["axes"][key][i:i+1], slice_target(data["targets"][key], slice(i,i+1)))
            for i in range(len(data["manifest"]))]
    trained = train_partial_structure({"gt_axes": examples["gt"], "predicted_axes": examples["predicted"],
        "predicted_no_relations": examples["predicted"]}, config, hidden=hidden, on_step=step_callback)
    if not all(torch.equal(trained.initial_state[k], v) for k, v in initial_state.items()):
        raise ValueError("training initialization differs from independently evaluated initial state")
    for branch in BRANCHES:
        history = trained.history[branch]
        if len(history) != config.steps or [r["sample_indices"] for r in history] != trained.schedule:
            raise ValueError("paired update budget/schedule drift")
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in trained.heads[branch].state_dict().items()},
            "branch": branch, "settings": settings, "step": config.steps, "use_relations": trained.heads[branch].use_relations},
            run / "artifacts" / (branch + "__final.pt"))
        outputs["final"][branch] = evaluate_and_save(run, branch, "final", trained.heads[branch], data, config, evaluation["member_threshold"])
    write_json(run / "artifacts/shared_schedule.json", trained.schedule)
    write_json(run / "artifacts/training_history.json", trained.history)
    write_json(run / "artifacts/manifest.json", data["manifest"])
    return {"evaluation": outputs, "optimizer_steps": sum(len(v) for v in trained.history.values()),
        "head_inference_windows": 2 * len(BRANCHES) * len(data["manifest"]), "backbone_windows": 0,
        "new_sensor_frames": 0, "same_initial_state_verified": True, "same_schedule_verified": True}


def execute(spec, run):
    run = Path(run).resolve()
    if (run != PROJECT_ROOT / "results/gate3_semantics" / build_run_id(spec)
            or load_json(run / "RUN_STATE.json")["state"] != "CREATED_NOT_EXECUTED"
            or load_json(run / "config/run_spec.json") != spec):
        raise RuntimeError("requires exact fresh run; no overwrite/retry")
    started, error, result, reads = time.monotonic(), None, {}, {}
    write_json(run / "RUN_STATE.json", {"state": "RUNNING", "run_id": run.name})
    def deadline(signum, frame): raise TimeoutError("fixed1800s training deadline exceeded")
    old_handler = signal.signal(signal.SIGALRM, deadline); signal.alarm(1800)
    try:
        card = load_json(contained(spec["data_card"]))
        report = validate_partial_structure_training_card(card)
        if (not report.passed or spec["operation"] != "training" or spec["wall_time_cap_s"] != 1800
                or load_json(run / "config/data_card.json") != card or spec["training"] != card["training"]
                or spec["evaluation"] != card["evaluation"]):
            raise ValueError(f"frozen head training scope drift: {report.errors}")
        frozen = {**card["sealed_sources"], **spec["source_sha256"]}
        for path, digest in frozen.items():
            if sha(contained(path)) != digest: raise ValueError(f"frozen source drift: {path}")
        versions = {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__, "zarr": zarr.__version__, "scipy": scipy.__version__}
        if versions != spec["expected_versions"]: raise ValueError("frozen environment drift")
        if not torch.cuda.is_available(): raise RuntimeError("required CUDA device unavailable; no CPU fallback")
        torch.use_deterministic_algorithms(True); torch.cuda.reset_peak_memory_stats()
        write_json(run / "config/execution_environment.json", {**versions, "platform": platform.platform(),
            "device": card["training"]["device"], "gpu_name": torch.cuda.get_device_name(0), "deterministic_algorithms": True})
        data = load_training_inputs(PROJECT_ROOT, card)
        reads = data["read_hashes"]
        with (run / "logs/training.jsonl").open("x") as log:
            def callback(branch, record):
                log.write(json.dumps({"branch": branch, "elapsed_s": time.monotonic() - started, **record}) + "\n"); log.flush()
                if record["step"] == 1 or record["step"] % 30 == 0:
                    print(json.dumps({"branch": branch, "step": record["step"], "loss": record["loss"]["total"], "elapsed_s": time.monotonic() - started}), flush=True)
                if (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > 4 * 1024**3
                        or torch.cuda.max_memory_reserved() > 4 * 1024**3):
                    raise RuntimeError("host/GPU4GiB budget exceeded")
            result = run_training(run, data, card["training"], card["evaluation"], step_callback=callback)
        for key in ("optimizer_steps", "head_inference_windows", "backbone_windows", "new_sensor_frames"):
            if result[key] != spec["expected_counts"][key]: raise ValueError(f"execution population drift: {key}")
        result["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        result["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["peak_gpu_reserved_bytes"] = torch.cuda.max_memory_reserved()
        if (result["peak_rss_bytes"] > 4 * 1024**3 or result["peak_gpu_reserved_bytes"] > 4 * 1024**3
                or time.monotonic() - started > 1800 or sum(p.stat().st_size for p in run.rglob("*") if p.is_file()) > 499_000_000):
            raise RuntimeError("frozen resource limit exceeded")
        for path, digest in {**frozen, **reads}.items():
            if sha(contained(path)) != digest: raise ValueError("source changed during training")
    except Exception:
        error = traceback.format_exc(); (run / "logs/error.log").write_text(error)
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, old_handler)
    write_json(run / "artifacts/source_reads_sha256.json", reads)
    status = "PARTIAL_STRUCTURE_FIXED_BUDGET_COMPLETE" if error is None else "PARTIAL_STRUCTURE_TRAINING_FAIL"
    write_json(run / "metrics/summary.json", {"status": status, "elapsed_s": time.monotonic() - started,
        "result": result, "error": error, "scientific_gate_pass": False, "full_three_class_ready": False})
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


if __name__ == "__main__":
    raise SystemExit(main())
