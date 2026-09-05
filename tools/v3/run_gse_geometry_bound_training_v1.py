#!/usr/bin/env python3
"""Single-variable geometry-bound head training; preserve prior joint baseline."""
import argparse
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
from mtare_topo.data.gse_partial_training_inputs import load_training_inputs
from mtare_topo.governance import build_run_id, load_json, write_json
from mtare_topo.governance_geometry_bound import validate_geometry_bound_training_card
from mtare_topo.representation.gse_region_queries import RegionQueryHead
from mtare_topo.representation.gse_partial_structure_training import BRANCHES, PartialStructureExample, PartialTrainingConfig
from mtare_topo.representation.gse_geometry_bound_training import train_geometry_bound_structure
from run_gse_partial_structure_training_v1 import evaluate_and_save, slice_target, predict, evaluate_partial_structure
from run_gse_supported_construction_teacher_v1 import sha


def contained(relative):
    path = (PROJECT_ROOT / relative).resolve()
    if not path.is_relative_to(PROJECT_ROOT): raise ValueError("path outside project")
    return path


def run_training(run, data, settings, evaluation, *, step_callback=None):
    config = PartialTrainingConfig(settings["seed"], settings["steps_per_branch"], settings["batch_size"], settings["lr"], settings["device"])
    hidden = settings["hidden"]
    for target in data["targets"].values():
        known = target.event_valid | target.member_valid.any(-1)
        if (known & ~target.center_valid).any(): raise ValueError("known semantic target lacks center")
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
    examples = {key: [PartialStructureExample(data["axes"][key][i:i+1], slice_target(data["targets"][key], slice(i, i+1)))
        for i in range(len(data["manifest"]))] for key in ("gt", "predicted")}
    trained = train_geometry_bound_structure({"gt_axes": examples["gt"], "predicted_axes": examples["predicted"],
        "predicted_no_relations": examples["predicted"]}, config, hidden=hidden, on_step=step_callback)
    if not all(torch.equal(trained.initial_state[k], value) for k, value in initial_state.items()):
        raise ValueError("training initial state differs from independently evaluated state")
    totals = {}
    for branch in BRANCHES:
        history = trained.history[branch]
        if len(history) != config.steps or [r["sample_indices"] for r in history] != trained.schedule:
            raise ValueError("paired update budget/schedule drift")
        totals[branch] = {group: {key: sum(row[group][key] for row in history) for key in history[0][group]}
            for group in ("counts", "geometry_counts", "supervision_counts")}
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in trained.heads[branch].state_dict().items()},
            "branch": branch, "settings": settings, "step": config.steps, "use_relations": trained.heads[branch].use_relations,
            "loss_policy": "geometry_only_unique_center_binding_v1"}, run / "artifacts" / (branch + "__final.pt"))
        outputs["final"][branch] = evaluate_and_save(run, branch, "final", trained.heads[branch], data, config, evaluation["member_threshold"])
    write_json(run / "artifacts/shared_schedule.json", trained.schedule)
    write_json(run / "artifacts/training_history.json", trained.history)
    write_json(run / "artifacts/manifest.json", data["manifest"])
    return {"evaluation": outputs, "optimizer_steps": sum(len(h) for h in trained.history.values()),
        "head_inference_windows": 2 * len(BRANCHES) * len(data["manifest"]), "backbone_windows": 0,
        "new_sensor_frames": 0, "same_initial_state_verified": True, "same_schedule_verified": True,
        "loss_policy": "geometry_only_unique_center_binding_v1", "supervision_totals": totals}


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
        report = validate_geometry_bound_training_card(card)
        if (not report.passed or spec["operation"] != "training" or spec["wall_time_cap_s"] != 1800
                or load_json(run / "config/data_card.json") != card or spec["training"] != card["training"]
                or spec["evaluation"] != card["evaluation"]):
            raise ValueError(f"frozen geometry-bound training scope drift: {report.errors}")
        frozen = {**card["sealed_sources"], **spec["source_sha256"]}
        for path, digest in frozen.items():
            if sha(contained(path)) != digest: raise ValueError("frozen source drift: " + path)
        versions = {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__, "zarr": zarr.__version__, "scipy": scipy.__version__}
        if versions != spec["expected_versions"]: raise ValueError("frozen environment drift")
        if not torch.cuda.is_available(): raise RuntimeError("required CUDA device unavailable; no CPU fallback")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8": raise ValueError("deterministic CUDA environment missing")
        torch.use_deterministic_algorithms(True); torch.cuda.reset_peak_memory_stats()
        write_json(run / "config/execution_environment.json", {**versions, "platform": platform.platform(),
            "device": card["training"]["device"], "gpu_name": torch.cuda.get_device_name(0), "deterministic_algorithms": True})
        data = load_training_inputs(PROJECT_ROOT, card["base_training_card"])
        reads = data["read_hashes"]
        with (run / "logs/training.jsonl").open("x") as log:
            def callback(branch, record):
                log.write(json.dumps({"branch": branch, "elapsed_s": time.monotonic() - started, **record}) + "\n"); log.flush()
                if record["step"] == 1 or record["step"] % 30 == 0:
                    print(json.dumps({"branch": branch, "step": record["step"], "loss": record["loss"]["total"],
                        "geometry_counts": record["geometry_counts"], "elapsed_s": time.monotonic() - started}), flush=True)
                if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > 4 * 1024**3 or torch.cuda.max_memory_reserved() > 4 * 1024**3:
                    raise RuntimeError("host/GPU4GiB budget exceeded")
            result = run_training(run, data, card["training"], card["evaluation"], step_callback=callback)
        for key in ("optimizer_steps", "head_inference_windows", "backbone_windows", "new_sensor_frames"):
            if result[key] != spec["expected_counts"][key]: raise ValueError("execution population drift: " + key)
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
    status = "GEOMETRY_BOUND_FIXED_BUDGET_COMPLETE" if error is None else "GEOMETRY_BOUND_TRAINING_FAIL"
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


if __name__ == "__main__": raise SystemExit(main())
