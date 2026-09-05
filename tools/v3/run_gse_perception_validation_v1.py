#!/usr/bin/env python3
"""One immutable validation-only GSE perception/baseline qualification run."""

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


RUN_ID = "gate3_20260824_gse_perception_validation_v1_seed0"
PASS_STATUS = "PASS_GSE_PERCEPTION_VALIDATION_V1"
FAIL_STATUS = "FAIL_GSE_PERCEPTION_VALIDATION_V1"
PYTHON = Path("/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python")
TRAINING = PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_graph_three_seed_training_v1r_seed0"
DATASET = PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
CHECKPOINTS = (
    (0, PROJECT_ROOT / "results/gate2_representation/gate2_20260812_cano_phase3_masking_corrective_ray_dropout_v1r3_seed0/artifacts/models/m1d_seed0/best.pt", "55f6602fb7fd74e3a44c3a4697a7c4be4d31469f46348f631dc606612125f9fe"),
    (1, PROJECT_ROOT / "results/gate2_representation/gate2_20260812_cano_phase3_masking_corrective_ray_dropout_v1r3_seed0/artifacts/models/m1d_seed1/best.pt", "3822d27af6d5fb0ef920da3e310927d42c5a112c5d671b4cbdc81d9db251f8df"),
    (2, PROJECT_ROOT / "results/gate2_representation/gate2_20260813_cano_phase3_masking_corrective_recovery_seed2_v1_seed2/artifacts/models/m1d_seed2/best.pt", "20b1e9c8198b06473b68e31f4d8a3eb50eb5f2fc1b7aa577aa1aac92ba35346e"),
)
DISK_LIMIT_BYTES = 2 * 1024**3
RSS_LIMIT_KIB = 16 * 1024**2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_seal(run: Path) -> dict:
    seal = run / "artifacts/evidence_sha256.txt"
    checked = 0
    for line in seal.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        if _sha256(PROJECT_ROOT / relative) != expected:
            raise RuntimeError(f"source seal drift: {relative}")
        checked += 1
    return {"run": str(run.relative_to(PROJECT_ROOT)), "entries": checked, "seal_sha256": _sha256(seal)}


def _seal(run_dir: Path) -> int:
    destination = run_dir / "artifacts/evidence_sha256.txt"
    files = sorted(path for path in run_dir.rglob("*") if path.is_file() and path != destination)
    with destination.open("w", encoding="utf-8") as stream:
        for path in files:
            stream.write(f"{_sha256(path)}  {path.relative_to(PROJECT_ROOT)}\n")
    return len(files)


def _run(argv: list[str], log: Path, env: dict[str, str], timeout: int) -> int:
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            argv,
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    spec = load_json(args.spec.resolve())
    if run_dir.name != RUN_ID or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("one-time GSE perception validation run identity mismatch")
    started = time.monotonic()
    overall = FAIL_STATUS
    error = None
    stages: dict[str, dict] = {}
    sources_before = None
    sources_after = None
    result_bytes = 0
    maximum_child_rss_kib = 0
    try:
        if shutil.disk_usage(PROJECT_ROOT).free < 4 * 1024**3:
            raise RuntimeError("less than 4 GiB free before GSE perception validation")
        if spec.get("gate") != 3 or spec.get("operation") != "threshold_calibration" or spec.get("user_authorization", {}).get("status") != "APPROVED":
            raise RuntimeError("GSE perception validation scope is not authorized")
        card = load_json(PROJECT_ROOT / spec["data_card"])
        if card.get("status") != "APPROVED_FOR_ONE_IMMUTABLE_GSE_PERCEPTION_VALIDATION":
            raise RuntimeError("operation-bound GSE perception Data Card mismatch")
        for name, record in spec["frozen_tools"].items():
            if _sha256(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"frozen validation tool drift: {name}")
        for relative, expected in spec["frozen_inputs"].items():
            if _sha256(PROJECT_ROOT / relative) != expected:
                raise RuntimeError(f"frozen validation input drift: {relative}")
        if load_json(TRAINING / "RUN_STATE.json").get("overall_status") != "PASS_GSE_GRAPH_THREE_SEED_TRAINING_V1R":
            raise RuntimeError("GSE training source is not the required PASS")
        if load_json(DATASET / "RUN_STATE.json").get("overall_status") != "PASS_GSE_DEDUPLICATED_DATASET_EXPORT_V1":
            raise RuntimeError("GSE dataset source is not the required PASS")
        for seed, checkpoint, expected in CHECKPOINTS:
            if _sha256(checkpoint) != expected:
                raise RuntimeError(f"historical M1D seed {seed} checkpoint drift")
        sources_before = {"training": _verify_seal(TRAINING), "dataset": _verify_seal(DATASET)}
        write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING"})
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3")
        write_json(
            run_dir / "config/environment.json",
            json.loads(
                subprocess.check_output(
                    [str(PYTHON), "-c", "import json,numpy,torch,zarr,sys;print(json.dumps({'python':sys.version.split()[0],'numpy':numpy.__version__,'torch':torch.__version__,'cuda':torch.version.cuda,'zarr':zarr.__version__,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},sort_keys=True))"],
                    text=True,
                )
            ),
        )
        exit_dir = run_dir / "artifacts/exit_only_baseline"
        argv = [str(PYTHON), str(PROJECT_ROOT / "tools/v3/evaluate_gse_exit_only_baseline_v1.py"), "--dataset-run", str(DATASET), "--output-dir", str(exit_dir)]
        for seed, checkpoint, expected in CHECKPOINTS:
            argv.extend(("--checkpoint", f"{seed}:{checkpoint}:{expected}"))
        code = _run(argv, run_dir / "logs/01_exit_only_baseline.log", env, 7200)
        stages["exit_only"] = load_json(exit_dir / "summary.json") if (exit_dir / "summary.json").is_file() else {"exit_code": code}
        if code != 0 or stages["exit_only"].get("status") != "PASS_GSE_EXIT_ONLY_BASELINE_VALIDATION_V1":
            raise RuntimeError("exit-only validation baseline failed")

        geometry_dir = run_dir / "artifacts/nonlearning_geometry"
        code = _run(
            [str(PYTHON), str(PROJECT_ROOT / "tools/v3/evaluate_gse_nonlearning_geometry_v1.py"), "--dataset-run", str(DATASET), "--output-dir", str(geometry_dir)],
            run_dir / "logs/02_nonlearning_geometry.log",
            env,
            7200,
        )
        stages["nonlearning_geometry"] = load_json(geometry_dir / "summary.json") if (geometry_dir / "summary.json").is_file() else {"exit_code": code}
        if code != 0 or stages["nonlearning_geometry"].get("overall_status") != "PASS_GSE_NONLEARNING_GEOMETRY_VALIDATION_V1":
            raise RuntimeError("non-learning geometry validation failed")

        calibration_dir = run_dir / "artifacts/calibration"
        code = _run(
            [str(PYTHON), str(PROJECT_ROOT / "tools/v3/evaluate_gse_validation_outputs_v1.py"), "--training-run", str(TRAINING), "--output-dir", str(calibration_dir)],
            run_dir / "logs/03_validation_calibration.log",
            env,
            7200,
        )
        stages["calibration"] = load_json(calibration_dir / "summary.json") if (calibration_dir / "summary.json").is_file() else {"exit_code": code}
        if code != 0 or stages["calibration"].get("overall_status") != "PASS_GSE_VALIDATION_CALIBRATION_V1":
            raise RuntimeError("validation association calibration failed the non-vacuous safety contract")

        gate_path = run_dir / "metrics/perception_gate.json"
        code = _run(
            [str(PYTHON), str(PROJECT_ROOT / "tools/v3/summarize_gse_perception_gate_v1.py"), "--training-run", str(TRAINING), "--calibration-dir", str(calibration_dir), "--exit-only-dir", str(exit_dir), "--geometry-dir", str(geometry_dir), "--output", str(gate_path)],
            run_dir / "logs/04_perception_gate.log",
            env,
            600,
        )
        gate = load_json(gate_path) if gate_path.is_file() else {"passed": False}
        stages["perception_gate"] = gate
        if code != 0 or gate.get("passed") is not True:
            raise RuntimeError("GSE perception did not pass the pre-registered scientific gate")
        sources_after = {"training": _verify_seal(TRAINING), "dataset": _verify_seal(DATASET)}
        if sources_before != sources_after:
            raise RuntimeError("source evidence changed during validation")
        result_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
        maximum_child_rss_kib = int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
        if result_bytes > DISK_LIMIT_BYTES:
            raise RuntimeError("GSE perception evidence exceeded the frozen 2 GiB limit")
        if maximum_child_rss_kib > RSS_LIMIT_KIB:
            raise RuntimeError("GSE perception validation exceeded the frozen 16 GiB child RSS limit")
        overall = PASS_STATUS
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run_dir / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
    result_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
    maximum_child_rss_kib = int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    write_json(
        run_dir / "metrics/summary.json",
        {
            "schema_version": "gse_perception_validation_run_v1",
            "overall_status": overall,
            "error": error,
            "stages": stages,
            "sources_before": sources_before,
            "sources_after": sources_after,
            "validation_worlds": 10,
            "validation_sequences": 24462,
            "strict_test_worlds_read": 0,
            "mtare_worlds_read": 0,
            "optimizer_steps": 0,
            "model_updates": 0,
            "result_bytes_before_seal": result_bytes,
            "disk_limit_bytes": DISK_LIMIT_BYTES,
            "maximum_child_rss_kib": maximum_child_rss_kib,
            "rss_limit_kib": RSS_LIMIT_KIB,
            "duration_seconds": time.monotonic() - started,
        },
    )
    write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "COMPLETED" if overall == PASS_STATUS else "FAILED", "overall_status": overall, "error": error})
    sealed = _seal(run_dir)
    print(json.dumps({"overall_status": overall, "error": error, "sealed_files": sealed}, indent=2))
    return 0 if overall == PASS_STATUS else 2


if __name__ == "__main__":
    raise SystemExit(main())
