#!/usr/bin/env python3
"""Execute and seal one immutable typed dual-Composer readiness run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
import traceback
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, validate_data_card, write_json


PASS = "PASS_GSE_TYPED_COMPOSER_READINESS_V1"
FAIL = "FAIL_GSE_TYPED_COMPOSER_READINESS_V1"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_GSE_TYPED_COMPOSER_READINESS_V1"
PYTHON = Path("/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python")
TEACHER = PROJECT_ROOT / "results/gate2_representation/gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0"
BASELINE = PROJECT_ROOT / "results/gate3_semantics/gate3_20260829_gse_sparse_circular_relation_transport_three_seed_training_v2r5_seed0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seal(run: Path) -> int:
    target = run / "artifacts/evidence_sha256.txt"
    files = sorted(path for path in run.rglob("*") if path.is_file() and path != target)
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
    run = args.run_dir.resolve()
    run_id = f"gate{spec['gate']}_{spec['date']}_{spec['slug']}_seed{spec['seed']}"
    if run.name != run_id or load_json(run / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("typed Composer readiness executes exactly once")

    started = time.monotonic()
    overall = FAIL
    error = None
    before: dict[str, str] = {}
    after: dict[str, str] = {}
    subprocesses = []
    peak_rss_kib = None
    result: dict[str, object] = {}
    try:
        card = load_json(PROJECT_ROOT / spec["data_card"])
        validation = validate_data_card(card)
        if not validation.passed or card.get("status") != CARD_STATUS:
            raise RuntimeError(f"typed Composer readiness Data Card invalid: {validation.errors}")
        for record in spec["frozen_tools"].values():
            if sha256(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"tool drift: {record['path']}")
        for relative, expected in spec["frozen_inputs"].items():
            before[relative] = sha256(PROJECT_ROOT / relative)
            if before[relative] != expected:
                raise RuntimeError(f"input drift: {relative}")
        environment = json.loads(subprocess.check_output([
            str(PYTHON), "-c",
            "import json,numpy,torch,zarr,sys;print(json.dumps({'python':sys.version.split()[0],'numpy':numpy.__version__,'torch':torch.__version__,'cuda':torch.version.cuda,'zarr':zarr.__version__},sort_keys=True))",
        ], text=True))
        expected_environment = {
            "python": "3.13.5", "numpy": "2.1.3", "torch": "2.9.0+cu129",
            "cuda": "12.9", "zarr": "2.18.7",
        }
        if environment != expected_environment:
            raise RuntimeError(f"environment drift: {environment}")
        write_json(run / "config/environment.json", {
            "executable": str(PYTHON), "versions": environment, "cpu_only": True,
            "deterministic_algorithms": True,
        })
        write_json(run / "config/source_integrity_before.json", before)
        write_json(run / "RUN_STATE.json", {
            "schema_version": "v3_run_state_v1", "run_id": run_id, "state": "RUNNING",
        })
        process_environment = os.environ.copy()
        process_environment["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3")
        unit_log = run / "logs/00_unit_tests.log"
        with unit_log.open("w", encoding="utf-8") as stream:
            unit = subprocess.run([
                str(PYTHON), "-m", "pytest", "-q", "tests/v3/unit/test_gse_typed_composers.py",
            ], cwd=PROJECT_ROOT, env=process_environment, stdout=stream, stderr=subprocess.STDOUT,
                text=True, timeout=600, check=False)
        subprocesses.append({"stage": "unit_tests", "returncode": int(unit.returncode)})
        if unit.returncode != 0:
            raise RuntimeError("typed Composer unit tests failed")

        command = [
            str(PYTHON), str(PROJECT_ROOT / "tools/v3/evaluate_gse_typed_composer_readiness_v1.py"),
            "--prediction-root", str(BASELINE / "artifacts/models/seed0/development_predictions"),
            "--teacher", str(TEACHER / "artifacts/teacher_observations.jsonl"),
            "--output-dir", str(run / "metrics/readiness"),
        ]
        (run / "config/command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
        readiness_log = run / "logs/01_readiness.log"
        with readiness_log.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(
                ["/usr/bin/time", "-v", *command], cwd=PROJECT_ROOT,
                env=process_environment, stdout=stream, stderr=subprocess.STDOUT,
                text=True, timeout=600, check=False,
            )
        subprocesses.append({"stage": "readiness", "returncode": int(completed.returncode)})
        match = re.search(
            r"Maximum resident set size \(kbytes\):\s*(\d+)",
            readiness_log.read_text(encoding="utf-8"),
        )
        peak_rss_kib = int(match.group(1)) if match else None
        summary_path = run / "metrics/readiness/summary.json"
        if not summary_path.is_file() or completed.returncode not in (0, 2):
            raise RuntimeError("typed Composer evaluator did not produce a scientific result")
        result = load_json(summary_path)
        overall = str(result["status"])
        if completed.returncode == 0 and overall != PASS:
            raise RuntimeError("readiness return/status mismatch")
        if completed.returncode == 2 and overall != FAIL:
            raise RuntimeError("readiness failure return/status mismatch")

        for relative in spec["frozen_inputs"]:
            after[relative] = sha256(PROJECT_ROOT / relative)
        if after != before:
            raise RuntimeError("frozen readiness input changed during execution")
        write_json(run / "config/source_integrity_after.json", after)
        write_json(run / "metrics/summary.json", {
            "schema_version": "gse_typed_composer_readiness_runner_v1",
            "status": overall,
            "scientific_pass": overall == PASS,
            "result": result,
            "subprocesses": subprocesses,
            "peak_host_rss_kib": peak_rss_kib,
            "duration_seconds": time.monotonic() - started,
            "optimizer_steps": 0, "checkpoints_created": 0,
            "c08_worlds_read": 0, "c09_worlds_read": 0, "c10_worlds_read": 0,
            "mtare_worlds_read": 0, "graph_replays": 0,
        })
    except Exception as exception:
        error = f"{type(exception).__name__}: {exception}"
        (run / "logs/runner_error.log").write_text(traceback.format_exc(), encoding="utf-8")
        overall = FAIL
        if not (run / "metrics/summary.json").is_file():
            write_json(run / "metrics/summary.json", {
                "schema_version": "gse_typed_composer_readiness_runner_v1",
                "status": overall, "scientific_pass": False, "error": error,
                "subprocesses": subprocesses, "duration_seconds": time.monotonic() - started,
                "optimizer_steps": 0, "checkpoints_created": 0,
            })

    write_json(run / "RUN_STATE.json", {
        "schema_version": "v3_run_state_v1", "run_id": run_id,
        "state": "COMPLETED" if error is None else "FAILED",
        "overall_status": overall, "error": error,
        "duration_seconds": time.monotonic() - started,
    })
    evidence_files = seal(run)
    print(json.dumps({
        "run_id": run_id, "state": "COMPLETED" if error is None else "FAILED",
        "overall_status": overall, "error": error, "evidence_files": evidence_files,
        "peak_host_rss_kib": peak_rss_kib,
    }, indent=2, sort_keys=True))
    return 0 if error is None and overall == PASS else 2


if __name__ == "__main__":
    raise SystemExit(main())
