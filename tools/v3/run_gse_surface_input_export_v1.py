#!/usr/bin/env python3
"""One immutable selected-sensor export; no labels, model, or training."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shlex
import signal
import subprocess
import sys
import time
import traceback
import zipfile

import numpy as np

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import build_run_id, load_json
from mtare_topo.governance_surface_input import validate_surface_input_card, RESOURCES
from mtare_topo.data.gse_surface_input_scope_v1 import compile_input_scope, access_summary, checked_read, safe_path
from mtare_topo.data.gse_surface_input_export_v1 import SurfaceInputReader
from freeze_gse_surface_input_export_v1 import current_environment


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                   separators=(",", ":"), allow_nan=False) + "\n", encoding="utf8")


def array_hash(value):
    header = json.dumps({"shape": list(value.shape), "dtype": value.dtype.str}, sort_keys=True).encode()
    return hashlib.sha256(header + b"\0" + value.tobytes(order="C")).hexdigest()


def execute(spec, run, root=PROJECT_ROOT):
    root, run = Path(root).resolve(strict=True), Path(run).resolve(strict=True)
    if (run != root / "results/gate3_semantics" / build_run_id(spec)
            or load_json(run / "RUN_STATE.json")["state"] != "CREATED_NOT_EXECUTED"
            or load_json(run / "config/run_spec.json") != spec):
        raise ValueError("exact fresh create_run output required; no overwrite/retry")
    started, error, reads, entries = time.monotonic(), None, {}, []
    reader, source_scope, decoded = None, None, {}
    old_signal = signal.getsignal(signal.SIGALRM)
    def expired(signum, frame):
        raise TimeoutError("3600s input export budget exceeded")
    signal.signal(signal.SIGALRM, expired); signal.alarm(RESOURCES["wall_time_s"])
    write(run / "RUN_STATE.json", {"state": "RUNNING", "run_id": run.name, "pid": os.getpid(),
        "started_at_utc": datetime.now(timezone.utc).isoformat()})
    try:
        card = load_json(safe_path(root, spec["data_card"]))
        report = validate_surface_input_card(card)
        if (not report.passed or spec["operation"] != "data_export"
                or load_json(run / "config/data_card.json") != card):
            raise ValueError("exact frozen sensor input card invalid: " + "; ".join(report.errors))
        if not spec.get("source_sha256"):
            raise ValueError("source freeze required")
        with zipfile.ZipFile(run / "artifacts/source_snapshot.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for path, h in sorted(spec["source_sha256"].items()):
                data = checked_read(root, path, h, reads)
                archive.writestr(path, data)
        executable = Path(sys.executable).resolve(strict=True)
        env = current_environment()
        freeze = subprocess.check_output([str(executable), "-m", "pip", "freeze", "--all"], text=True)
        if (env != spec["environment"] or env != card["scope"]["environment"]
                or sha(executable) != spec["python_executable_sha256"]
                or hashlib.sha256(freeze.encode()).hexdigest() != spec["sidecar_freeze_sha256"]):
            raise ValueError("bound data-sidecar environment drift")
        command = shlex.join(spec["command"]) + "\n"
        if (run / "config/command.txt").read_text() != command:
            raise ValueError("command snapshot drift")
        write(run / "config/execution_environment.json", {**env, "platform": platform.platform(),
            "executable": str(executable), "executable_sha256": sha(executable),
            "sidecar_freeze_sha256": spec["sidecar_freeze_sha256"],
            "command_sha256": hashlib.sha256(command.encode()).hexdigest(), "gpu": False})
        (run / "config/sidecar_freeze.txt").write_text(freeze)
        # Independently regenerate exact keys and collateral from pinned source
        # headers and immutable selection BEFORE permitting any array payload.
        source_scope, manifest, metadata_reads = compile_input_scope(root)
        reads.update(metadata_reads)
        if source_scope != card["scope"] or access_summary(source_scope) != spec["access_summary"]:
            raise ValueError("recomputed exact input/chunk scope differs from card")
        reader = SurfaceInputReader(root, task_sources=source_scope["task_sources"],
            selection=manifest["observations"], sealed_keys=source_scope["file_sha256"])
        (run / "artifacts/inputs").mkdir()
        with (run / "logs/tasks.jsonl").open("x", encoding="utf8") as log:
            for task in sorted(source_scope["task_sources"]):
                batch = reader.read_task(task)
                payload = {"ranges_m": batch.ranges_m, "valid_mask": batch.valid_mask,
                    "relative_translation_current_sensor_m": batch.relative_translation_current_sensor_m,
                    "relative_yaw_current_sensor_deg": batch.relative_yaw_current_sensor_deg,
                    "frame_rows": batch.frame_rows, "source_sequence_ids": batch.source_sequence_ids}
                expected = {key: array_hash(value) for key, value in payload.items()}
                path = run / "artifacts/inputs" / (task + ".npz")
                if path.exists():
                    raise FileExistsError("input shard already exists")
                np.savez_compressed(path, **payload)
                with np.load(path, allow_pickle=False) as restored:
                    if set(restored.files) != set(payload) or {key: array_hash(restored[key]) for key in restored.files} != expected:
                        raise ValueError("lossless input shard roundtrip failed")
                identity = source_scope["task_sources"][task]
                row = {"task": task, "parent_id": identity["parent_id"], "variant": identity["variant"],
                    "path": str(path.relative_to(run)), "sha256": sha(path), "array_sha256": expected,
                    "valid_returns": int(np.count_nonzero(batch.valid_mask)),
                    "empty_history_frames": int(np.count_nonzero(~batch.valid_mask.any(axis=(-2, -1)))),
                    "empty_windows": int(np.count_nonzero(~batch.valid_mask.any(axis=(-3, -2, -1)))),
                    "read_report": batch.read_report, "elapsed_s": time.monotonic() - started}
                entries.append(row)
                log.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n"); log.flush()
                print(json.dumps({"completed_tasks": len(entries), "task": task, "observations": len(entries) * 16,
                    "elapsed_s": row["elapsed_s"], "valid_returns": row["valid_returns"]}), flush=True)
                if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > RESOURCES["host_ram_bytes"]:
                    raise MemoryError("4GiB host cap exceeded")
                del batch, payload
        if set(reader.opened) != set(source_scope["file_sha256"]):
            raise ValueError("actual source-file reads differ from frozen exact input set")
        reads.update(reader.opened)
        # Recheck every actually read file after export. This byte-integrity pass
        # does not decode extra rows or grant any new source paths.
        for path, h in reads.items():
            if sha(safe_path(root, path)) != h:
                raise ValueError("input/source changed during export: " + path)
        if current_environment() != spec["environment"]:
            raise ValueError("environment changed during export")
        decoded = access_summary(source_scope)
        write(run / "artifacts/input_manifest.json", {"schema_version": "gse_surface_sensor_inputs_v1",
            "selection": source_scope["selection"], "population": source_scope["population"],
            "observations": manifest["observations"], "task_shards": entries,
            "student_fields": ["ranges_m", "valid_mask", "relative_translation_current_sensor_m", "relative_yaw_current_sensor_deg"],
            "provenance_only_fields": ["frame_rows", "source_sequence_ids"],
            "absolute_pose_exported": False, "teacher_labels": 0, "training_eligible": False,
            "independent_continuous_route_evidence": False, "physical_root_labels": "PAUSED_ROBOT_CONTRACT_FALLBACK"})
    except Exception:
        error = traceback.format_exc()
        (run / "logs/error.log").write_text(error)
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, old_signal)
        if reader is not None:
            reads.update(reader.opened)
        write(run / "artifacts/source_reads_sha256.json", reads)
        summary = {"status": "INPUT_EXPORT_FAIL" if error else "INPUT_EXPORT_COMPLETE", "error": error,
            "completed_tasks": len(entries), "exported_observations": len(entries) * 16,
            "exported_unique_variant_frames": len(entries) * 80, "elapsed_s": time.monotonic() - started,
            "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "actual_input_files_read": len(reader.opened) if reader else 0, "field_access": decoded,
            "valid_returns": sum(x["valid_returns"] for x in entries),
            "empty_history_frames": sum(x["empty_history_frames"] for x in entries),
            "empty_windows": sum(x["empty_windows"] for x in entries),
            "teacher_labels": 0, "model_windows": 0, "optimizer_steps": 0, "scientific_gate_pass": False}
        write(run / "metrics/summary.json", summary)
        write(run / "RUN_STATE.json", {"state": "FAILED" if error else "COMPLETED", "run_id": run.name, "error": error})
        seal = run / "artifacts/evidence_sha256.txt"
        def seal_evidence():
            seal.write_text("".join(f"{sha(p)}  {p.relative_to(root)}\n" for p in sorted(run.rglob("*")) if p.is_file() and p != seal))
        seal_evidence()
        output_bytes = sum(p.stat().st_size for p in run.rglob("*") if p.is_file())
        if error is None and (time.monotonic() - started > RESOURCES["wall_time_s"]
                or summary["peak_host_rss_bytes"] > RESOURCES["host_ram_bytes"] or output_bytes > RESOURCES["output_bytes"]):
            error = "Final sealed output/time/host budget exceeded; no retry"
            summary.update(status="INPUT_EXPORT_FAIL", error=error)
            write(run / "metrics/summary.json", summary)
            write(run / "RUN_STATE.json", {"state": "FAILED", "run_id": run.name, "error": error})
            (run / "logs/error.log").write_text(error + "\n")
            seal_evidence()
    print(json.dumps({"error": error, "completed_tasks": len(entries), "seal_sha256": sha(seal), "output_bytes": output_bytes}), flush=True)
    return int(error is not None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_AS, (RESOURCES["host_ram_bytes"], RESOURCES["host_ram_bytes"]))
    raise SystemExit(execute(load_json(args.spec), args.run_dir))
