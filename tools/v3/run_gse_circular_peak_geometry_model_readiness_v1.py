#!/usr/bin/env python3
"""Execute and seal one immutable circular peak geometry model readiness."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, validate_data_card, write_json


PASS = "PASS_GSE_CIRCULAR_PEAK_GEOMETRY_MODEL_READINESS_V1"
FAIL = "FAIL_GSE_CIRCULAR_PEAK_GEOMETRY_MODEL_READINESS_V1"
PYTHON = Path("/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python")
DATASET = PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
TEACHER = PROJECT_ROOT / "results/gate2_representation/gate2_20260829_gse_circular_exit_geometry_field_teacher_export_v1_seed0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seal(run_dir: Path) -> int:
    target = run_dir / "artifacts/evidence_sha256.txt"
    files = sorted(path for path in run_dir.rglob("*") if path.is_file() and path != target)
    with target.open("w", encoding="utf-8") as stream:
        for path in files:
            stream.write(f"{sha256(path)}  {path.relative_to(PROJECT_ROOT)}\n")
    return len(files)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    spec = load_json(args.spec.resolve())
    run_dir = args.run_dir.resolve()
    run_id = f"gate{spec['gate']}_{spec['date']}_{spec['slug']}_seed{spec['seed']}"
    if run_dir.name != run_id or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("circular peak model readiness executes exactly once")
    started = time.monotonic()
    overall = FAIL
    error = None
    result: dict = {}
    before: dict[str, str] = {}
    after: dict[str, str] = {}
    returncode = None
    try:
        card = load_json(PROJECT_ROOT / spec["data_card"])
        validation = validate_data_card(card)
        if not validation.passed or card.get("status") != "APPROVED_FOR_ONE_IMMUTABLE_GSE_CIRCULAR_PEAK_GEOMETRY_MODEL_READINESS_V1" or card.get("approval", {}).get("authorized_operations") != ["audit"] or card.get("approval", {}).get("authorized_gates") != [3]:
            raise RuntimeError(f"circular model readiness Data Card invalid: {validation.errors}")
        for record in spec["frozen_tools"].values():
            if sha256(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"tool drift: {record['path']}")
        for relative, expected in spec["frozen_inputs"].items():
            before[relative] = sha256(PROJECT_ROOT / relative)
            if before[relative] != expected:
                raise RuntimeError(f"input drift: {relative}")
        environment = json.loads(subprocess.check_output([str(PYTHON), "-c", "import json,numpy,torch,zarr,sys;print(json.dumps({'python':sys.version.split()[0],'numpy':numpy.__version__,'torch':torch.__version__,'zarr':zarr.__version__},sort_keys=True))"], text=True))
        expected = {"python": "3.13.5", "numpy": "2.1.3", "torch": "2.9.0+cu129", "zarr": "2.18.7"}
        if environment != expected:
            raise RuntimeError(f"circular model readiness environment drift: {environment}")
        write_json(run_dir / "config/environment.json", {"executable": str(PYTHON), "versions": environment, "device": "cpu", "deterministic_algorithms": True})
        write_json(run_dir / "config/source_integrity_before.json", before)
        command = [str(PYTHON), str(PROJECT_ROOT / "tools/v3/execute_gse_circular_peak_geometry_model_readiness_v1.py"), "--teacher-root", str(TEACHER / "artifacts/export/teacher"), "--source-root", str(DATASET / "artifacts/dataset/train"), "--output-dir", str(run_dir / "artifacts/readiness")]
        write_json(run_dir / "config/commands.json", [command])
        write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": run_id, "state": "RUNNING"})
        environment_process = os.environ.copy()
        environment_process["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3")
        environment_process["OMP_NUM_THREADS"] = "4"
        with (run_dir / "logs/00_readiness.log").open("w", encoding="utf-8") as stream:
            returncode = subprocess.run(command, cwd=PROJECT_ROOT, env=environment_process, stdout=stream, stderr=subprocess.STDOUT, text=True, timeout=900, check=False).returncode
        if returncode not in (0, 2):
            raise RuntimeError(f"circular model readiness executor system failure: {returncode}")
        result = load_json(run_dir / "artifacts/readiness/summary.json")
        scientific_pass = result.get("status") == PASS and result.get("scientific_pass") is True
        if (returncode == 0) != scientific_pass:
            raise RuntimeError("circular model readiness status/return mismatch")
        population = result.get("population", {})
        if population.get("worlds") != 80 or population.get("observations") != 188126 or population.get("peaks") != 396913 or population.get("real_readiness_rows") != 8 or result.get("model", {}).get("parameters") != 769268:
            raise RuntimeError("circular model readiness population/interface drift")
        if any(result.get(name) != 0 for name in ("optimizer_steps", "model_inference_frames", "model_updates", "checkpoint_writes", "normalization_steps", "threshold_selection_steps", "graph_replays", "c09_worlds_read", "c10_worlds_read", "mtare_worlds_read")):
            raise RuntimeError("forbidden circular model readiness operation occurred")
        required = [run_dir / "artifacts/readiness/summary.json", run_dir / "artifacts/readiness/figure_source.json", run_dir / "artifacts/readiness/readiness_checks.csv", *[run_dir / f"artifacts/readiness/gse_circular_peak_geometry_model_readiness_v1.{suffix}" for suffix in ("png", "pdf", "svg")]]
        if any(not path.is_file() or path.stat().st_size == 0 for path in required):
            raise RuntimeError("circular model readiness evidence incomplete")
        after = {relative: sha256(PROJECT_ROOT / relative) for relative in before}
        if before != after:
            raise RuntimeError("circular model readiness changed a frozen source")
        overall = PASS if scientific_pass else FAIL
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run_dir / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
    write_json(run_dir / "metrics/summary.json", {"schema_version": "gse_circular_peak_geometry_model_readiness_outer_v1", "overall_status": overall, "scientific_pass": overall == PASS, "error": error, "executor_returncode": returncode, "result": result, "duration_seconds": time.monotonic() - started, "source_unchanged": bool(before and before == after), "optimizer_steps": 0, "model_inference_frames": 0, "model_updates": 0, "checkpoint_writes": 0, "normalization_steps": 0, "threshold_selection_steps": 0, "graph_replays": 0, "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0})
    write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": run_id, "state": "COMPLETED" if overall == PASS and error is None else "FAILED", "overall_status": overall, "error": error})
    entries = seal(run_dir)
    print(json.dumps({"overall_status": overall, "error": error, "decision": result.get("decision"), "seal_entries": entries}, indent=2))
    return 0 if overall == PASS and error is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
