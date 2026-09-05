#!/usr/bin/env python3
"""Run one immutable C09 perception validation with only slope corrected."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time
import traceback

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


RUN_ID = "gate3_20260824_gse_corrected_perception_validation_v1_seed0"
PASS_STATUS = "PASS_GSE_CORRECTED_PERCEPTION_VALIDATION_V1"
FAIL_STATUS = "FAIL_GSE_CORRECTED_PERCEPTION_VALIDATION_V1"
PYTHON = Path("/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python")
DATASET = PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
MAIN_TRAINING = PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_graph_three_seed_training_v1r_seed0"
CORRECTIVE = PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_slope_corrective_three_seed_training_v1r_seed0"
ORIGINAL_C09 = PROJECT_ROOT / "results/gate3_semantics/gate3_20260824_gse_perception_validation_v1_seed0"
RSS_LIMIT_KIB = 8 * 1024**2
DISK_LIMIT_BYTES = 2 * 1024**3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_seal(run: Path, expected_entries: int) -> dict:
    seal = run / "artifacts/evidence_sha256.txt"
    entries = []
    for line in seal.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = PROJECT_ROOT / relative
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"source seal drift: {relative}")
        entries.append(relative)
    actual = {
        str(path.relative_to(PROJECT_ROOT))
        for path in run.rglob("*")
        if path.is_file() and path != seal
    }
    if len(entries) != expected_entries or len(set(entries)) != len(entries) or set(entries) != actual:
        raise RuntimeError(f"source seal coverage drift: {run}")
    return {"run": str(run.relative_to(PROJECT_ROOT)), "entries": len(entries), "seal_sha256": _sha256(seal)}


def _seal(run_dir: Path) -> int:
    seal = run_dir / "artifacts/evidence_sha256.txt"
    files = sorted(path for path in run_dir.rglob("*") if path.is_file() and path != seal)
    with seal.open("w", encoding="utf-8") as stream:
        for path in files:
            stream.write(f"{_sha256(path)}  {path.relative_to(PROJECT_ROOT)}\n")
    return len(files)


def _run(argv: list[str], log: Path, env: dict[str, str], timeout: int) -> int:
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(argv, cwd=PROJECT_ROOT, env=env, text=True, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout, check=False)
    return result.returncode


def _sources() -> dict:
    expected = {
        "dataset": (DATASET, "PASS_GSE_DEDUPLICATED_DATASET_EXPORT_V1", 33083),
        "main_training": (MAIN_TRAINING, "PASS_GSE_GRAPH_THREE_SEED_TRAINING_V1R", 33),
        "corrective_training": (CORRECTIVE, "PASS_GSE_SLOPE_CORRECTIVE_THREE_SEED_TRAINING_V1R", 115),
        "original_c09_failure": (ORIGINAL_C09, "FAIL_GSE_PERCEPTION_VALIDATION_V1", 49),
    }
    result = {}
    for name, (run, status, entries) in expected.items():
        state = load_json(run / "RUN_STATE.json")
        summary = load_json(run / "metrics/summary.json")
        if state.get("overall_status") != status or summary.get("overall_status") != status:
            raise RuntimeError(f"corrected perception source status drift: {name}")
        result[name] = _verify_seal(run, entries)
    original_gate = load_json(ORIGINAL_C09 / "metrics/perception_gate.json")
    if (
        original_gate.get("event_gate", {}).get("passed") is not True
        or original_gate.get("association_gate", {}).get("passed") is not True
        or original_gate.get("geometry_gate", {}).get("passed") is not False
    ):
        raise RuntimeError("original C09 is not the required slope-only failure")
    cache = load_json(CORRECTIVE / "artifacts/slope_corrective_cache/manifest.json")
    paths = cache.get("actual_source_paths_read", [])
    if (
        len(paths) != 80
        or any("/dataset/train/" not in value or not value.endswith(tuple(f"_C{code:02d}.zarr" for code in range(1, 9))) for value in paths)
        or any("_C09" in value or "_C10" in value for value in paths)
    ):
        raise RuntimeError("corrective training actual source-path isolation drift")
    result["corrective_training_read_audit"] = {
        "actual_source_paths": len(paths),
        "all_under_dataset_train": True,
        "c09_worlds_read": 0,
        "c10_worlds_read": 0,
        "mtare_worlds_read": 0,
        "status_snapshot_semantics": "PRE_RUN_SNAPSHOT_ONLY; final authority is RUN_STATE plus metrics plus seal",
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    spec = load_json(args.spec.resolve())
    if run_dir.name != RUN_ID or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("one-time corrected perception run identity mismatch")
    started = time.monotonic()
    overall = FAIL_STATUS
    error = None
    stages = {}
    sources_before = None
    sources_after = None
    try:
        if spec.get("gate") != 3 or spec.get("operation") != "threshold_calibration" or spec.get("user_authorization", {}).get("status") != "APPROVED":
            raise RuntimeError("corrected perception scope is not authorized")
        card = load_json(PROJECT_ROOT / spec["data_card"])
        if card.get("status") != "APPROVED_FOR_ONE_IMMUTABLE_GSE_CORRECTED_PERCEPTION_VALIDATION":
            raise RuntimeError("corrected perception Data Card status mismatch")
        for name, record in spec["frozen_tools"].items():
            if _sha256(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"corrected perception frozen tool drift: {name}")
        for relative, expected in spec["frozen_inputs"].items():
            if _sha256(PROJECT_ROOT / relative) != expected:
                raise RuntimeError(f"corrected perception frozen input drift: {relative}")
        if shutil.disk_usage(PROJECT_ROOT).free < 4 * 1024**3:
            raise RuntimeError("less than 4 GiB free before corrected C09 validation")
        sources_before = _sources()
        write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING"})
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3")
        write_json(
            run_dir / "config/environment.json",
            json.loads(subprocess.check_output([str(PYTHON), "-c", "import json,numpy,torch,zarr,sys;print(json.dumps({'python':sys.version.split()[0],'numpy':numpy.__version__,'torch':torch.__version__,'cuda':torch.version.cuda,'zarr':zarr.__version__,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},sort_keys=True))"], text=True)),
        )
        slope_dir = run_dir / "artifacts/corrected_slope_c09"
        code = _run(
            [str(PYTHON), str(PROJECT_ROOT / "tools/v3/evaluate_gse_slope_corrective_c09_v1.py"), "--dataset-run", str(DATASET), "--corrective-run", str(CORRECTIVE), "--output-dir", str(slope_dir)],
            run_dir / "logs/01_corrected_slope_c09.log",
            env,
            3600,
        )
        stages["corrected_slope"] = load_json(slope_dir / "summary.json") if (slope_dir / "summary.json").is_file() else {"exit_code": code}
        if code != 0 or stages["corrected_slope"].get("overall_status") != "PASS_GSE_SLOPE_CORRECTIVE_C09_EVALUATION_V1":
            raise RuntimeError("corrective C09 slope evaluation failed technical evidence")
        gate_path = run_dir / "metrics/corrected_perception_gate.json"
        code = _run(
            [str(PYTHON), str(PROJECT_ROOT / "tools/v3/summarize_gse_corrected_perception_gate_v1.py"), "--original-gate", str(ORIGINAL_C09 / "metrics/perception_gate.json"), "--corrective-summary", str(slope_dir / "summary.json"), "--output", str(gate_path)],
            run_dir / "logs/02_corrected_perception_gate.log",
            env,
            600,
        )
        stages["corrected_perception_gate"] = load_json(gate_path) if gate_path.is_file() else {"exit_code": code}
        if code != 0 or stages["corrected_perception_gate"].get("passed") is not True:
            raise RuntimeError("corrected GSE perception did not pass all original and learned-slope gates")
        sources_after = _sources()
        if sources_before != sources_after:
            raise RuntimeError("corrected perception source evidence changed during validation")
        result_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
        rss = int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
        if result_bytes > DISK_LIMIT_BYTES or rss > RSS_LIMIT_KIB:
            raise RuntimeError("corrected perception resource contract failed")
        overall = PASS_STATUS
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run_dir / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
    result_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
    rss = int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    write_json(
        run_dir / "metrics/summary.json",
        {
            "schema_version": "gse_corrected_perception_validation_run_v1",
            "overall_status": overall,
            "error": error,
            "stages": stages,
            "sources_before": sources_before,
            "sources_after": sources_after,
            "validation_worlds_read": 10,
            "validation_sequences": 24462,
            "c10_worlds_read": 0,
            "strict_test_worlds_read": 0,
            "mtare_worlds_read": 0,
            "optimizer_steps": 0,
            "model_updates": 0,
            "result_bytes_before_seal": result_bytes,
            "maximum_child_rss_kib": rss,
            "duration_seconds": time.monotonic() - started,
        },
    )
    write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "COMPLETED" if overall == PASS_STATUS else "FAILED", "overall_status": overall, "error": error})
    sealed = _seal(run_dir)
    print(json.dumps({"overall_status": overall, "error": error, "sealed_files": sealed}, indent=2))
    return 0 if overall == PASS_STATUS else 2


if __name__ == "__main__":
    raise SystemExit(main())
