#!/usr/bin/env python3
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
from mtare_topo.governance import load_json, write_json


PASS = "PASS_GSE_GEOMETRY_ANCHORED_SPATIAL_EVENT_READINESS_V1"
FAIL = "FAIL_GSE_GEOMETRY_ANCHORED_SPATIAL_EVENT_READINESS_V1"
PYTHON = Path("/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python")


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
        raise RuntimeError("geometry-anchored readiness executes exactly once")

    started = time.monotonic()
    overall = FAIL
    error = None
    result: dict[str, object] = {}
    before: dict[str, str] = {}
    after: dict[str, str] = {}
    try:
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
                "torch": __import__("torch").__version__,
                "gpu_used": False,
            },
        )
        write_json(
            run_dir / "RUN_STATE.json",
            {"schema_version": "v3_run_state_v1", "run_id": run_id, "state": "RUNNING"},
        )
        teacher = PROJECT_ROOT / "results/gate2_representation/gate2_20260828_gse_spatial_multi_event_teacher_export_v1_seed0/artifacts/export/teacher"
        source = PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0/artifacts/dataset"
        command = [
            str(PYTHON),
            str(PROJECT_ROOT / "tools/v3/execute_gse_geometry_anchored_spatial_event_readiness_v1.py"),
            "--teacher-root",
            str(teacher),
            "--source-root",
            str(source),
            "--output-dir",
            str(run_dir / "artifacts/audit"),
        ]
        (run_dir / "config/command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3")
        with (run_dir / "logs/audit.log").open("w", encoding="utf-8") as stream:
            code = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=1800,
                check=False,
            ).returncode
        if code not in (0, 2):
            raise RuntimeError(f"readiness executor system failure: {code}")
        result = load_json(run_dir / "artifacts/audit/summary.json")
        if (
            result.get("status") not in (PASS, FAIL)
            or result.get("population", {}).get("observations") != 188126
            or any(
                result.get(name) != 0
                for name in (
                    "optimizer_steps",
                    "trained_model_inference_frames",
                    "threshold_selection_steps",
                    "c09_worlds_read",
                    "c10_worlds_read",
                    "mtare_worlds_read",
                )
            )
        ):
            raise RuntimeError("readiness result contract drift")
        after = {relative: sha256(PROJECT_ROOT / relative) for relative in before}
        if before != after:
            raise RuntimeError("frozen source changed")
        overall = str(result["status"])
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run_dir / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")

    write_json(
        run_dir / "metrics/summary.json",
        {
            "schema_version": "gse_geometry_anchored_spatial_event_readiness_outer_v1",
            "overall_status": overall,
            "error": error,
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
