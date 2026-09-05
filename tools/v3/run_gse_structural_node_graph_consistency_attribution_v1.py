#!/usr/bin/env python3
"""Run and seal structural-node graph-consistency attribution."""

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


PYTHON = Path("/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python")
EXECUTOR = PROJECT_ROOT / "tools/v3/execute_gse_structural_node_graph_consistency_attribution_v1.py"
P1A = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1a_lossless_storage_corrective_v1_seed0"
P1B = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1b_teacher_materialization_v1r_seed0"
SOURCE = PROJECT_ROOT / "results/gate3_semantics/gate3_20260904_gse_structural_node_evidence_funnel_v1r_seed0"
PASS = "PASS_GSE_STRUCTURAL_NODE_GRAPH_CONSISTENCY_ATTRIBUTION_V1"
FAIL = "FAIL_GSE_STRUCTURAL_NODE_GRAPH_CONSISTENCY_ATTRIBUTION_V1"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify(spec: dict) -> dict[str, str]:
    observed = {}
    for relative, expected in spec["frozen_inputs"].items():
        actual = _sha(PROJECT_ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"graph-consistency frozen input drift: {relative}")
        observed[relative] = actual
    return observed


def _seal(run_dir: Path) -> int:
    seal = run_dir / "artifacts/evidence_sha256.txt"
    files = sorted(path for path in run_dir.rglob("*") if path.is_file() and path != seal)
    with seal.open("w", encoding="utf-8") as stream:
        for path in files:
            stream.write(f"{_sha(path)}  {path.relative_to(PROJECT_ROOT)}\n")
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
        raise RuntimeError("graph-consistency attribution may execute only once")
    started = time.monotonic()
    overall, error, result, returncode = FAIL, None, {}, None
    before: dict[str, str] = {}; after: dict[str, str] = {}
    try:
        for record in spec["frozen_tools"].values():
            if _sha(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"graph-consistency frozen tool drift: {record['path']}")
        before = _verify(spec)
        write_json(run_dir / "config/source_integrity_before.json", before)
        write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": run_id, "state": "RUNNING"})
        env = os.environ.copy()
        env.update({"PYTHONPATH": str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3"), "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})
        command = [
            str(PYTHON), str(EXECUTOR),
            "--construction-root", str(P1A / "artifacts/constructions"),
            "--teacher-root", str(P1B / "artifacts/teacher"),
            "--traversal-manifest", str(P1A / "artifacts/traversal_manifest.jsonl"),
            "--source-summary", str(SOURCE / "metrics/funnel/summary.json"),
            "--output-dir", str(run_dir / "metrics/attribution"),
        ]
        with (run_dir / "logs/executor.log").open("w", encoding="utf-8") as stream:
            completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env, text=True, stdout=stream, stderr=subprocess.STDOUT, timeout=3600, check=False)
        returncode = int(completed.returncode)
        if returncode not in (0, 2):
            raise RuntimeError("graph-consistency executor failed as a program")
        result = load_json(run_dir / "metrics/attribution/summary.json")
        scientific_pass = result.get("status") == PASS
        if scientific_pass != (returncode == 0):
            raise RuntimeError("graph-consistency status/return mismatch")
        after = _verify(spec)
        if before != after:
            raise RuntimeError("graph-consistency source changed during audit")
        overall = PASS if scientific_pass else FAIL
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run_dir / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
    summary = {"schema_version": "gse_structural_node_graph_consistency_attribution_outer_v1", "overall_status": overall, "scientific_pass": overall == PASS, "error": error, "duration_seconds": time.monotonic() - started, "executor_returncode": returncode, "attribution": result, "source_unchanged": bool(before and before == after), "optimizer_steps": 0, "model_inference_frames": 0, "c08_c09_c10_worlds_read": 0, "graph_replays": 0, "mtare_runs": 0}
    write_json(run_dir / "metrics/summary.json", summary)
    write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": run_id, "state": "COMPLETED", "overall_status": overall, "error": error})
    entries = _seal(run_dir)
    print(json.dumps({"overall_status": overall, "error": error, "seal_entries": entries}, indent=2))
    return 0 if overall == PASS else 2


if __name__ == "__main__":
    raise SystemExit(main())
