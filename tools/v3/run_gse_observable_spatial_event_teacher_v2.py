#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import traceback

import numpy
import torch
import zarr

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, validate_data_card, write_json


PASS = "PASS_GSE_OBSERVABLE_SPATIAL_EVENT_TEACHER_V2"
FAIL = "FAIL_GSE_OBSERVABLE_SPATIAL_EVENT_TEACHER_V2"
PYTHON = Path("/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python")
SOURCE_RUN = PROJECT_ROOT / "results/gate2_representation/gate2_20260828_gse_spatial_multi_event_teacher_export_v1_seed0"


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
        raise RuntimeError("observable Teacher V2 executes exactly once")

    started = time.monotonic()
    overall = FAIL
    error = None
    result: dict[str, object] = {}
    before: dict[str, str] = {}
    after: dict[str, str] = {}
    executor_returncode = None
    try:
        card = load_json(PROJECT_ROOT / spec["data_card"])
        validation = validate_data_card(card)
        if (
            not validation.passed
            or card.get("status") != "APPROVED_FOR_ONE_IMMUTABLE_GSE_OBSERVABLE_SPATIAL_EVENT_TEACHER_V2"
            or card.get("approval", {}).get("authorized_operations") != ["teacher_generation"]
            or card.get("approval", {}).get("authorized_gates") != [2]
        ):
            raise RuntimeError(f"observable Teacher V2 Data Card invalid: {validation.errors}")
        for record in spec["frozen_tools"].values():
            if sha256(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"tool drift: {record['path']}")
        for relative, expected in spec["frozen_inputs"].items():
            before[relative] = sha256(PROJECT_ROOT / relative)
            if before[relative] != expected:
                raise RuntimeError(f"input drift: {relative}")
        write_json(run_dir / "config/source_integrity_before.json", before)
        write_json(
            run_dir / "config/environment.json",
            {
                "executable": str(PYTHON),
                "python": platform.python_version(),
                "numpy": numpy.__version__,
                "torch": torch.__version__,
                "zarr": zarr.__version__,
                "gpu_used": False,
            },
        )
        write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": run_id, "state": "RUNNING"})
        command = [
            str(PYTHON),
            str(PROJECT_ROOT / "tools/v3/execute_gse_observable_spatial_event_teacher_v2.py"),
            "--source-teacher-root",
            str(SOURCE_RUN / "artifacts/export/teacher"),
            "--source-summary",
            str(SOURCE_RUN / "artifacts/export/summary.json"),
            "--source-identity-map",
            str(SOURCE_RUN / "artifacts/export/event_identity_map.json"),
            "--output-dir",
            str(run_dir / "artifacts/export"),
        ]
        (run_dir / "config/command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3")
        environment["OMP_NUM_THREADS"] = "2"
        with (run_dir / "logs/export.log").open("w", encoding="utf-8") as stream:
            executor_returncode = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=600,
                check=False,
            ).returncode
        if executor_returncode not in (0, 2):
            raise RuntimeError(f"observable Teacher V2 executor system failure: {executor_returncode}")
        result = load_json(run_dir / "artifacts/export/summary.json")
        population = result.get("population", {})
        if (
            result.get("status") not in (PASS, FAIL)
            or population.get("worlds") != 80
            or population.get("observations") != 188126
            or population.get("event_identities") != 1076
            or population.get("observable_tokens") != 131424
            or population.get("removed_unobservable_tokens") != 1631
            or any(
                result.get(name) != 0
                for name in (
                    "optimizer_steps",
                    "model_inference_frames",
                    "model_updates",
                    "normalization_steps",
                    "threshold_selection_steps",
                    "c09_worlds_read",
                    "c10_worlds_read",
                    "mtare_worlds_read",
                )
            )
        ):
            raise RuntimeError("observable Teacher V2 result contract drift")
        if bool(result.get("checks", {}).get("all_passed")) != bool(result["status"] == PASS):
            raise RuntimeError("observable Teacher V2 gate/status mismatch")
        required = [
            run_dir / "artifacts/export/event_identity_map.json",
            run_dir / "artifacts/export/teacher_shard_manifest.jsonl",
            run_dir / "artifacts/export/removed_unobservable_tokens.jsonl",
            run_dir / "artifacts/export/figure_source.json",
            run_dir / "artifacts/export/gse_observable_spatial_event_teacher_v2.png",
            run_dir / "artifacts/export/gse_observable_spatial_event_teacher_v2.pdf",
            run_dir / "artifacts/export/gse_observable_spatial_event_teacher_v2.svg",
        ]
        if (
            len(list((run_dir / "artifacts/export/teacher").glob("*/*.zarr"))) != 80
            or any(not path.is_file() or path.stat().st_size == 0 for path in required)
        ):
            raise RuntimeError("observable Teacher V2 evidence incomplete")
        after = {relative: sha256(PROJECT_ROOT / relative) for relative in before}
        if before != after:
            raise RuntimeError("observable Teacher V2 source changed")
        overall = str(result["status"])
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run_dir / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")

    write_json(
        run_dir / "metrics/summary.json",
        {
            "schema_version": "gse_observable_spatial_event_teacher_outer_v2",
            "overall_status": overall,
            "error": error,
            "executor_returncode": executor_returncode,
            "result": result,
            "duration_seconds": time.monotonic() - started,
            "source_unchanged": bool(before and before == after),
        },
    )
    write_json(
        run_dir / "RUN_STATE.json",
        {
            "schema_version": "v3_run_state_v1",
            "run_id": run_id,
            "state": "COMPLETED" if overall == PASS and error is None else "FAILED",
            "overall_status": overall,
            "error": error,
        },
    )
    entries = seal(run_dir)
    print(json.dumps({"overall_status": overall, "error": error, "seal_entries": entries}, indent=2))
    return 0 if overall == PASS and error is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
