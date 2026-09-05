#!/usr/bin/env python3
"""Recover shared-frame fields from the exact frozen 180-observation model run."""
import argparse
import hashlib
import json
import os
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
from mtare_topo.data.gse_scoped_model_input import ScopedCompositionModelReader
from mtare_topo.governance import build_run_id, load_json, write_json
from mtare_topo.representation.gse_structural_node_student import structural_node_student_inputs
from mtare_topo.representation.primitive_relation_observable_model import ObservableSparsePortRelationNet


FIELDS = {"axis_control_current_sensor_m": (32, 3, 3), "endpoint_half_axes_m": (32, 2, 2),
          "endpoint_shape_exponent": (32, 2), "existence_logits": (32,),
          "geometry_uncertainty": (32,), "endpoint_evidence_logits": (32, 2)}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def contained(relative):
    path = (PROJECT_ROOT / relative).resolve()
    if not path.is_relative_to(PROJECT_ROOT):
        raise ValueError("source outside project")
    return path


def state_sha(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous().numpy()
        h.update(name.encode()); h.update(str(value.dtype).encode())
        h.update(np.asarray(value.shape, dtype="<i8").tobytes()); h.update(value.tobytes())
    return h.hexdigest()


def load_model(path):
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS workspace differs from source run")
    np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if not torch.cuda.is_available():
        raise RuntimeError("frozen CUDA execution environment unavailable")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != "primitive_relation_observable_checkpoint_v1" or checkpoint.get("seed") != 0:
        raise ValueError("checkpoint schema/seed drift")
    model = ObservableSparsePortRelationNet()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval().to("cuda")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    torch.cuda.reset_peak_memory_stats()
    return model


@torch.no_grad()
def predict(model, student):
    device = next(model.parameters()).device
    values = [torch.from_numpy(np.stack([getattr(row, name) for row in student])).to(device)
              for name in ("range_valid", "relative_translation_current_sensor_m", "relative_yaw_current_sensor_deg")]
    prediction = model(*values)
    features, confidence = structural_node_student_inputs(prediction)
    output = {name: getattr(prediction, name).detach().cpu().numpy() for name in FIELDS}
    for name, shape in FIELDS.items():
        if output[name].shape != (len(student), *shape) or output[name].dtype != np.float32 or not np.isfinite(output[name]).all():
            raise ValueError(f"prediction field drift: {name}")
    return output, features.cpu().numpy(), confidence.cpu().numpy()


def recover(model, reader, records, cache, artifact_dir, log, *, repeat_task, ledger=None):
    """Exact-cache parity before each immutable task export; no label access."""
    if (cache["endpoint_features"].shape != (180, 64, 44)
            or cache["endpoint_confidence"].shape != (180, 64)
            or not np.array_equal(cache["source_global_sequence_index"], [r["source_global_sequence_index"] for r in records])):
        raise ValueError("legacy cache shape/row alignment drift")
    tasks, primary, repeats, unique_frames = [], 0, 0, 0
    ledger = {} if ledger is None else ledger
    for task in reader.selection:
        batch = reader.read_task(task)
        indices = [i for i, row in enumerate(records) if row["task"] == task]
        if len(indices) != 18 or len(batch.student) != 18:
            raise ValueError("each frozen task must contain exactly 18 observations")
        ledger["primary_requested"] = ledger.get("primary_requested", 0) + len(indices)
        raw, features, confidence = predict(model, batch.student)
        primary += len(indices)
        ledger["primary_completed"] = primary
        differences = {"features_max_abs": float(np.max(np.abs(features - cache["endpoint_features"][indices]))),
                       "confidence_max_abs": float(np.max(np.abs(confidence - cache["endpoint_confidence"][indices])))}
        row = {"task": task, "observations": len(indices), **differences}
        log.write(json.dumps(row) + "\n"); log.flush()
        if not np.array_equal(features, cache["endpoint_features"][indices]) or not np.array_equal(confidence, cache["endpoint_confidence"][indices]):
            raise ValueError(f"legacy cache parity failed: {row}")
        if task == repeat_task:
            ledger["repeat_requested"] = ledger.get("repeat_requested", 0) + len(indices)
            raw_repeat, f_repeat, c_repeat = predict(model, batch.student)
            repeats += len(indices)
            ledger["repeat_completed"] = repeats
            if (not np.array_equal(features, f_repeat) or not np.array_equal(confidence, c_repeat)
                    or any(not np.array_equal(raw[name], raw_repeat[name]) for name in FIELDS)):
                raise ValueError("repeat inference not bitwise deterministic")
        unique_frames += len(np.unique(batch.frame_rows))
        with (artifact_dir / (task + ".npz")).open("xb") as stream:
            np.savez_compressed(stream, **raw)
        row["source_sequence_indices"] = batch.source_sequence_indices.tolist()
        row["frame_rows"] = batch.frame_rows.tolist()
        tasks.append(row)
    if (primary, repeats, unique_frames) != (180, 18, 900):
        raise ValueError("inference/source population drift")
    return {"primary_inference_observations": primary, "repeat_inference_observations": repeats,
            "total_inference_observations": primary + repeats, "unique_sensor_frames": unique_frames,
            "optimizer_steps": 0, "new_teacher_labels": 0, "legacy_features_exact": True,
            "legacy_confidence_exact": True, "raw_repeat_exact": True, "tasks": tasks}


def main():
    from mtare_topo.governance_field_recovery import validate_scoped_field_recovery_card
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    spec, run = load_json(args.spec), args.run_dir.resolve()
    if (run != PROJECT_ROOT / "results/gate3_semantics" / build_run_id(spec)
            or load_json(run / "RUN_STATE.json")["state"] != "CREATED_NOT_EXECUTED"
            or load_json(run / "config/run_spec.json") != spec):
        raise RuntimeError("requires one fresh immutable run; no overwrite or retry")
    started, error, result, reader = time.monotonic(), None, {"inference_ledger": {}}, None
    def timeout_signal(signum, frame):
        raise TimeoutError("external execution deadline reached")
    signal.signal(signal.SIGTERM, timeout_signal)
    write_json(run / "RUN_STATE.json", {"state": "RUNNING", "run_id": run.name})
    try:
        card = load_json(contained(spec["data_card"]))
        report = validate_scoped_field_recovery_card(card)
        if not report.passed or load_json(run / "config/data_card.json") != card or spec["operation"] != "data_export":
            raise ValueError(f"exact export card invalid/drift: {report.errors}")
        if any(spec["source_sha256"][key] != value for key, value in card["sealed_sources"].items() if key in spec["source_sha256"]):
            raise ValueError("conflicting card/spec source hashes")
        frozen = {**card["sealed_sources"], **spec["source_sha256"]}
        for relative, digest in frozen.items():
            if sha(contained(relative)) != digest:
                raise ValueError(f"source/tool drift: {relative}")
        versions = {"python": platform.python_version(), "numpy": np.__version__, "zarr": zarr.__version__, "torch": torch.__version__}
        if versions != spec["expected_versions"]:
            raise ValueError(f"environment drift: {versions}")
        records = json.loads(contained(spec["selection_manifest"]).read_text())
        selection = [{k: row[k] for k in ("task", "row_index", "source_global_sequence_index")} for row in records]
        if selection != card["selected_rows"] or len({(r["task"], r["target_node_scoring_only"]) for r in records}) != 100:
            raise ValueError("same-180 row/node inventory drift")
        seals = {}
        for relative in spec["source_seals"]:
            if relative not in frozen:
                raise ValueError("unfrozen source seal")
            for line in contained(relative).read_text().splitlines():
                digest, name = line.split(None, 1)
                path = str(contained(name))
                if path in seals and seals[path] != digest:
                    raise ValueError("inconsistent source seals")
                seals[path] = digest
        reader = ScopedCompositionModelReader(contained(spec["sensor_root"]), contained(spec["teacher_root"]),
                                              selection, expected_sha256=seals)
        model = load_model(contained(card["checkpoint"]["path"]))
        before = state_sha(model)
        write_json(run / "config/execution_environment.json", {**versions, "device": torch.cuda.get_device_name(0),
                   "platform": platform.platform(), "tf32": False, "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG")})
        with np.load(contained(spec["legacy_cache"]), allow_pickle=False) as archive:
            cache = {k: archive[k] for k in ("endpoint_features", "endpoint_confidence", "source_global_sequence_index")}
        output = run / "artifacts/shared_frame_predictions"; output.mkdir()
        with (run / "logs/recovery.log").open("x") as log:
            result.update(recover(model, reader, records, cache, output, log,
                                  repeat_task=card["inference"]["repeat_task"], ledger=result["inference_ledger"]))
        if state_sha(model) != before or any(p.requires_grad or p.grad is not None for p in model.parameters()):
            raise ValueError("frozen model state/gradient drift")
        result.update({"model_state_sha256": before, "model_unchanged": True, "training_ready": False,
                       "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                       "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                       "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(), "source_chunk_files_read": len(reader.opened)})
        write_json(run / "artifacts/prediction_manifest.json", result.pop("tasks"))
        for p, digest in reader.opened.items():
            if sha(Path(p)) != digest:
                raise ValueError("read source changed during export")
        for p, digest in frozen.items():
            if sha(contained(p)) != digest:
                raise ValueError("frozen source/tool changed during export")
        if (result["peak_rss_bytes"] > 8 * 1024**3 or result["gpu_peak_reserved_bytes"] > 4 * 1024**3
                or time.monotonic() - started > 600):
            raise RuntimeError("field recovery resource cap exceeded")
    except Exception:
        error = traceback.format_exc()
        (run / "logs/error.log").write_text(error)
    if reader is not None:
        write_json(run / "artifacts/source_reads_sha256.json", {str(Path(p).relative_to(PROJECT_ROOT)): h for p, h in sorted(reader.opened.items())})
    write_json(run / "metrics/summary.json", {"status": "FIELD_RECOVERY_PASS" if error is None else "FIELD_RECOVERY_FAIL",
               "elapsed_s": time.monotonic() - started, "result": result, "error": error, "scientific_gate_pass": False})
    write_json(run / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "state": "COMPLETED" if error is None else "FAILED",
                                      "run_id": run.name, "error": error})
    seal = run / "artifacts/evidence_sha256.txt"
    seal.write_text("".join(f"{sha(p)}  {p.relative_to(PROJECT_ROOT)}\n" for p in sorted(run.rglob("*")) if p.is_file() and p != seal))
    print(json.dumps({"error": error, "result": result, "seal_sha256": sha(seal)}))
    return int(error is not None)


if __name__ == "__main__":
    raise SystemExit(main())
