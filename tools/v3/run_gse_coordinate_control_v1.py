#!/usr/bin/env python3
"""C01 raw versus mean-broadcast coordinates: same head/update budget."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import resource
import signal
import time
import traceback

import numpy as np
import torch

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import build_run_id, load_json, write_json
from mtare_topo.governance_coordinate_control import validate_coordinate_control_card
from mtare_topo.data.gse_scoped_model_input import ScopedCompositionModelReader
from mtare_topo.representation.gse_coordinate_ablation import VARIANTS, coordinate_inputs
from mtare_topo.representation.gse_point_axis_readout import PointAxisReadout
from mtare_topo.representation.gse_point_axis_probe import configure_variant, evaluate, fit, sample_schedule, tensor_state_sha
from mtare_topo.representation.gse_point_axis_loss import axis_set_loss
from mtare_topo.evaluation.gse_axis_observation_dependence import score_observation, summarize
from run_gse_point_axis_probe_v1 import build_cache
from run_gse_composition_field_recovery_v1 import sha, contained, load_model, state_sha

FIELDS = {"sensor": frozenset(("range_m", "valid_mask")),
          "teacher": frozenset(("frame_row", "source_global_sequence_index", "primitive_mask",
                                "axis_control_current_sensor_m", "relative_translation_current_sensor_m",
                                "relative_yaw_current_sensor_deg"))}


def source_index(card):
    """Filter text BEFORE path resolution; unselected/C07 paths never resolve."""
    tasks = sorted({row["task"] for row in card["selected_rows"]})
    expected = {}
    for role, seal in card["source_seals"].items():
        if role == "reference":
            exact = {card["prediction_reference_root"] + "/" + task + ".npz" for task in tasks}
            prefixes = ()
        else:
            exact = set()
            prefixes = tuple(card["source_roots"][role] + "/" + task + ".zarr/" for task in tasks)
        with contained(seal).open() as stream:
            for line in stream:
                digest, path = line.strip().split(None, 1)
                selected = path in exact
                if not selected:
                    suffix = next((path[len(prefix):] for prefix in prefixes if path.startswith(prefix)), None)
                    if suffix is not None:
                        if suffix in (".zgroup", ".zattrs"):
                            selected = True
                        else:
                            pieces = suffix.split("/")
                            selected = (len(pieces) == 2 and pieces[0] in FIELDS[role]
                                        and (pieces[1] in (".zarray", ".zattrs")
                                             or re.fullmatch(r"[0-9]+(?:\.[0-9]+)*", pieces[1]) is not None))
                if not selected:
                    continue
                if re.fullmatch(r"[a-f0-9]{64}", digest) is None:
                    raise ValueError("invalid selected source digest")
                key = str(contained(path))
                if key in expected and expected[key] != digest:
                    raise ValueError("conflicting selected source seals")
                expected[key] = digest
    return expected


def variant_cache(cache, variant):
    """Only six student tensors change; targets/masks/order stay identical."""
    return [{**entry, "student": coordinate_inputs(entry["student"], variant)} for entry in cache]


def evaluate_all(axes, cache, records):
    details, summary = {}, {}
    for name, values in axes.items():
        if values.shape != (180, 32, 3, 3) or not np.isfinite(values).all():
            raise ValueError("all180/all32 finite axis predictions required")
        details[name] = [score_observation(values[i], entry["target"], entry["mask"])
                         for i, entry in enumerate(cache)]
        summary[name] = summarize(details[name], records)
    raw, mean = [summary[v] for v in VARIANTS]
    deltas = {parent: mean["parents"][parent]["coordinate_mae_m"] - raw["parents"][parent]["coordinate_mae_m"]
              for parent in raw["parents"]}
    summary["comparison"] = {
        "mean_minus_raw_coordinate_mae_m": mean["macro"]["coordinate_mae_m"] - raw["macro"]["coordinate_mae_m"],
        "per_parent_mean_minus_raw_coordinate_mae_m": deltas,
        "parents_raw_better": sum(delta > 0 for delta in deltas.values()),
        "parents_tied": sum(delta == 0 for delta in deltas.values()),
        "scientific_gate_pass": False, "generalization_claim": False, "detection_claim": False,
        "interpretation": "Same-budget coordinate intervention only; not historical token decoder. No surplus-query penalty or event/graph qualification."}
    return details, summary


def save_previews(run, cache, axes, result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    parents = sorted(result[VARIANTS[0]]["parents"])
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, variant in enumerate(VARIANTS):
        ax.bar(np.arange(10) + (i - .5) * .35,
               [result[variant]["parents"][parent]["coordinate_mae_m"] for parent in parents], .35, label=variant)
    ax.set_xticks(np.arange(10), [parent[:3] for parent in parents]); ax.set_ylabel("Coordinate MAE (m)")
    ax.set_title("C01 same180 FIT coordinate control; not detection or held-out evaluation"); ax.legend()
    fig.tight_layout(); fig.savefig(run / "previews/parent_geometry_mae.svg"); plt.close(fig)
    for parent in parents:
        positions = [i for i, entry in enumerate(cache) if entry["task"] == parent]
        fig, axs = plt.subplots(6, 6, figsize=(18, 18))
        for row, index in enumerate(positions):
            target = cache[index]["target"][0][cache[index]["mask"][0]].numpy()
            for view, dims in enumerate(((0, 1), (0, 2))):
                ax = axs.flat[2 * row + view]
                for variant, color in zip(VARIANTS, ("#3587ba", "#df7726")):
                    for line in axes[variant][index]:
                        ax.plot(line[:, dims[0]], line[:, dims[1]], color=color, alpha=.3, lw=.6)
                for line in target: ax.plot(line[:, dims[0]], line[:, dims[1]], color="black", lw=1.)
                ax.set_title(f"observation {index} {'XY' if view == 0 else 'XZ'} (m)", fontsize=8)
                ax.set_aspect("equal", adjustable="datalim"); ax.tick_params(labelsize=6)
        fig.suptitle(f"{parent}: all18 observations/all32 queries; black teacher, blue raw, orange mean broadcast\nSame head/parameters/order/budget; initial outputs need not agree", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, .96)); fig.savefig(run / f"previews/{parent}.svg"); plt.close(fig)


def make_initial_head(device):
    torch.manual_seed(0)
    return PointAxisReadout().to(device)


def environment_versions():
    return {"python": platform.python_version(), **{name: importlib.metadata.version(name)
            for name in ("numpy", "torch", "zarr", "scipy", "matplotlib")}}


def device_report(model):
    if next(model.parameters()).device.type != "cuda": raise ValueError("CUDA required; no silent CPU run")
    return {"device": str(next(model.parameters()).device), "gpu_name": torch.cuda.get_device_name(0)}


class LedgerLog:
    """Count completed frozen cache batches without changing original builder."""
    def __init__(self, stream, ledger): self.stream, self.ledger = stream, ledger
    def write(self, value):
        row = json.loads(value)
        if row.get("stage") == "cache":
            self.ledger["backbone_full_prediction_windows"] += row["rows"]
            self.ledger["backbone_memory_cache_windows"] += row["rows"]
        return self.stream.write(value)
    def flush(self): return self.stream.flush()


def train_control(run, model, cache, legacy, records, log, guard, ledger):
    """Real training/evaluation orchestration, also exercised on synthetic CPU inputs."""
    before = state_sha(model); device = next(model.parameters()).device
    initial = make_initial_head(device); initial_sha = tensor_state_sha(initial)
    schedule = sample_schedule()
    write_json(run / "artifacts/sample_schedule.json", schedule)
    write_json(run / "artifacts/sample_manifest.json", records)
    torch.save(initial.state_dict(), run / "artifacts/initial_head.pt")
    axes = {"legacy_frozen": legacy}; fit_metrics = {}; training = {}; readiness = []
    for variant in VARIANTS:
        selected = variant_cache(cache, variant)
        head = configure_variant(initial, "raw_no_offset")
        if tensor_state_sha(head) != initial_sha: raise ValueError("initial parameters differ")
        torch.save(head.state_dict(), run / f"artifacts/{variant}_initial.pt")
        # This forward intentionally has coordinate-dependent output. No
        # initial-axis equality assertion belongs in this intervention.
        axes[variant + "_initial"], fit_metrics[variant + "_initial"] = evaluate(head, selected)
        ledger["head_initial_final_evaluation_windows"] += len(selected)
        entry = selected[0]
        output = head(*[tensor.to(device) for tensor in entry["student"]]).votes.axis_control_m
        loss = axis_set_loss(output, entry["target"].to(device), entry["mask"].to(device)); loss.backward(); guard()
        ledger["zero_update_gradient_windows"] += 1
        if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in head.parameters() if p.requires_grad):
            raise ValueError("readiness gradient missing/nonfinite")
        readiness.append({"variant": variant, "loss": float(loss.detach()), "optimizer_steps": 0,
                          "trainable_parameters": sum(p.numel() for p in head.parameters() if p.requires_grad)})
        del output, loss
        def log_step(row):
            ledger["optimizer_steps"] += 1
            log.write(json.dumps({"stage": "train", "variant": variant, **row}) + "\n"); log.flush()
        trained = fit(head, selected, schedule, log_step, guard=guard)
        if trained["steps"] != 540: raise ValueError("exact step count drift")
        if any(p.requires_grad or bool(p.any()) or p.grad is not None
               for layer in (head.offset, head.slot_offset) for p in layer.parameters()):
            raise ValueError("coordinate-only control changed frozen zero offsets")
        axes[variant], fit_metrics[variant] = evaluate(head, selected)
        ledger["head_initial_final_evaluation_windows"] += len(selected)
        torch.save({"schema_version": "gse_coordinate_control_head_v1", "variant": variant, "seed": 0,
                    "head_state_dict": head.state_dict(), "training": trained,
                    "backbone_state_sha256": before}, run / f"artifacts/{variant}_final.pt")
        training[variant] = {key: value for key, value in trained.items() if key != "optimizer_state_dict"}
        del head, selected
        guard()
    if state_sha(model) != before or any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise ValueError("frozen backbone changed state/gradient")
    if tensor_state_sha(initial) != initial_sha: raise ValueError("initial head mutated")
    targets = np.concatenate([item["target"].numpy() for item in cache])
    masks = np.concatenate([item["mask"].numpy() for item in cache])
    np.savez_compressed(run / "artifacts/all_predictions.npz", **axes)
    np.savez_compressed(run / "artifacts/existing_scoring_targets.npz", axis_control_m=targets, mask=masks)
    write_json(run / "artifacts/fit_observation_metrics.json", fit_metrics)
    write_json(run / "artifacts/pretraining_readiness.json", readiness)
    details, result = evaluate_all(axes, cache, records)
    for name, rows in details.items(): write_json(run / f"artifacts/{name}_layout_metrics.json", rows)
    save_previews(run, cache, axes, result)
    result.update({"training": training, "initial_parameters_sha256": initial_sha,
                   "same_initial_parameters": True, "initial_output_equality_required": False,
                   "backbone_state_sha256": before, "backbone_unchanged": True,
                   "observations": 180, "parent_count": 10, "sensor_frames": 900,
                   "visible_fragments": 1452, "new_labels": 0, "scientific_gate_pass": False})
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--spec", required=True, type=Path); parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args(); spec = load_json(args.spec); run = args.run_dir.resolve()
    if (run != PROJECT_ROOT / "results/gate3_semantics" / build_run_id(spec)
            or load_json(run / "RUN_STATE.json")["state"] != "CREATED_NOT_EXECUTED"
            or load_json(run / "config/run_spec.json") != spec):
        raise ValueError("exact fresh coordinate control run required; no overwrite/retry")
    started = time.monotonic(); error = None; result = {}; reader = None
    ledger = {"optimizer_steps": 0, "backbone_full_prediction_windows": 0, "backbone_memory_cache_windows": 0,
              "head_initial_final_evaluation_windows": 0, "zero_update_gradient_windows": 0}
    def guard():
        if (time.monotonic() - started > 1200 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > 4 * 1024**3
                or (torch.cuda.is_initialized() and torch.cuda.max_memory_reserved() > 4 * 1024**3)):
            raise RuntimeError("coordinate control 1200s/4GiB host/GPU cap exceeded")
    def timeout(signum, frame): raise TimeoutError("coordinate control interrupted/deadline")
    signal.signal(signal.SIGALRM, timeout); signal.alarm(1200)
    signal.signal(signal.SIGTERM, timeout)
    write_json(run / "RUN_STATE.json", {"state": "RUNNING", "run_id": run.name})
    try:
        card = load_json(contained(spec["data_card"])); report = validate_coordinate_control_card(card)
        if (not report.passed or spec["operation"] != "training" or load_json(run / "config/data_card.json") != card
                or spec["training"] != card["training"] or spec.get("wall_time_cap_s") != 1200):
            raise ValueError(f"coordinate control card/scope drift: {report.errors}")
        frozen = {**card["sealed_sources"], **spec["source_sha256"]}
        for path, digest in frozen.items():
            if sha(contained(path)) != digest: raise ValueError(f"frozen input/tool drift: {path}")
        versions = environment_versions()
        if versions != spec["expected_versions"]: raise ValueError("environment drift")
        expected = source_index(card); records = card["selected_rows"]
        reader = ScopedCompositionModelReader(contained(card["source_roots"]["sensor"]),
                    contained(card["source_roots"]["teacher"]), records, expected_sha256=expected)
        model = load_model(contained(card["checkpoint"]["path"])); before = state_sha(model)
        write_json(run / "config/execution_environment.json", {**versions, **device_report(model), "tf32": False,
                   "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG")})
        with (run / "logs/progress.jsonl").open("x") as log:
            cache, legacy, manifest = build_cache(model, reader, records,
                        contained(card["prediction_reference_root"]), expected, LedgerLog(log, ledger), guard)
            if state_sha(model) != before: raise ValueError("backbone drift in cache construction")
            result = train_control(run, model, cache, legacy, manifest, log, guard, ledger)
        if ledger != {"optimizer_steps": 1080, "backbone_full_prediction_windows": 180, "backbone_memory_cache_windows": 180,
                      "head_initial_final_evaluation_windows": 720, "zero_update_gradient_windows": 2}:
            raise ValueError("actual computation ledger drift")
        for path, digest in {**frozen, **reader.opened}.items():
            if sha(contained(path)) != digest: raise ValueError("source drift during coordinate control")
        guard()
    except Exception:
        error = traceback.format_exc(); (run / "logs/error.log").write_text(error)
    finally:
        signal.alarm(0)
    if reader is not None:
        write_json(run / "artifacts/source_reads_sha256.json", {str(Path(path).relative_to(PROJECT_ROOT)): digest
                   for path, digest in sorted(reader.opened.items())})
    write_json(run / "metrics/summary.json", {"status": "SYSTEM_FAIL" if error else "COORDINATE_CONTROL_EVALUATED",
               "result": result, "error": error, "ledger": ledger, "scientific_gate_pass": False,
               "elapsed_s": time.monotonic() - started,
               "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
               "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0})
    write_json(run / "RUN_STATE.json", {"state": "FAILED" if error else "COMPLETED", "run_id": run.name, "error": error})
    seal = run / "artifacts/evidence_sha256.txt"
    seal.write_text("".join(f"{sha(path)}  {path.relative_to(PROJECT_ROOT)}\n" for path in sorted(run.rglob("*")) if path.is_file() and path != seal))
    print(json.dumps({"error": error, "comparison": result.get("comparison"), "ledger": ledger, "seal_sha256": sha(seal)}))
    return int(error is not None)


if __name__ == "__main__": raise SystemExit(main())
