#!/usr/bin/env python3
"""Execute and seal the read-only trace-commit failure-funnel audit."""

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


RUN_ID = "gate3_20260828_gse_trace_commit_failure_funnel_audit_v1_seed0"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_GSE_TRACE_COMMIT_FAILURE_FUNNEL_AUDIT_V1"
PYTHON = Path("/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python")
PASS = "PASS_GSE_TRACE_COMMIT_FAILURE_FUNNEL_AUDIT_V1"
FAIL = "FAIL_GSE_TRACE_COMMIT_FAILURE_FUNNEL_AUDIT_V1"
RESULT_STATUS = PASS


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify(spec: dict) -> dict[str, str]:
    result = {}
    for relative, expected in spec["frozen_inputs"].items():
        actual = _sha(PROJECT_ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"failure-funnel frozen input drift: {relative}")
        result[relative] = actual
    return result


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
    run_dir = args.run_dir.resolve()
    spec = load_json(args.spec.resolve())
    if run_dir.name != RUN_ID or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("failure-funnel audit may execute only once")
    started = time.monotonic()
    overall, error, result = FAIL, None, {}
    before = after = {}
    returncode = None
    try:
        if spec.get("gate") != 3 or spec.get("operation") != "audit" or spec.get("seed") != 0:
            raise RuntimeError("failure-funnel scope drift")
        card = load_json(PROJECT_ROOT / spec["data_card"])
        if card.get("status") != CARD_STATUS:
            raise RuntimeError("failure-funnel Data Card drift")
        for record in spec["frozen_tools"].values():
            if _sha(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"failure-funnel frozen tool drift: {record['path']}")
        before = _verify(spec)
        write_json(run_dir / "config/source_integrity_before.json", before)
        versions = json.loads(subprocess.check_output([
            str(PYTHON), "-c",
            "import json,matplotlib,numpy,sys;print(json.dumps({'python':sys.version.split()[0],'numpy':numpy.__version__,'matplotlib':matplotlib.__version__},sort_keys=True))",
        ], text=True))
        expected = {"python": "3.13.5", "numpy": "2.1.3", "matplotlib": "3.10.0"}
        if versions != expected:
            raise RuntimeError(f"failure-funnel environment drift: {versions}")
        write_json(run_dir / "config/environment.json", {"executable": str(PYTHON), "versions": versions, "gpu_used": False})
        write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING"})
        current = PROJECT_ROOT / "results/gate3_semantics/gate3_20260828_gse_spatial_trace_commit_requalification_v1r2_seed0"
        predecessor = PROJECT_ROOT / "results/gate3_semantics/gate3_20260828_gse_projected_trace_commit_capacity_v1_seed0"
        teacher = PROJECT_ROOT / "results/gate2_representation/gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0/artifacts/teacher_observations.jsonl"
        pair_cache = PROJECT_ROOT / "results/gate3_semantics/gate3_20260826_gse_exit_token_association_corrective_v2_seed0/artifacts/pair_cache/pairs.npz"
        command = [
            str(PYTHON), str(PROJECT_ROOT / "tools/v3/execute_gse_trace_commit_failure_funnel_audit_v1.py"),
            "--current-run", str(current), "--predecessor-run", str(predecessor),
            "--teacher", str(teacher), "--pair-cache", str(pair_cache),
            "--output-dir", str(run_dir / "artifacts/audit"),
        ]
        (run_dir / "config/audit_command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3")
        with (run_dir / "logs/audit.log").open("w", encoding="utf-8") as stream:
            completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env, text=True, stdout=stream, stderr=subprocess.STDOUT, timeout=600, check=False)
        returncode = int(completed.returncode)
        if returncode != 0:
            raise RuntimeError(f"failure-funnel executor failed with code {returncode}")
        result = load_json(run_dir / "artifacts/audit/summary.json")
        if (
            result.get("status") != RESULT_STATUS
            or result.get("population", {}).get("true_node_identities") != 274
            or result.get("population", {}).get("true_trace_relations") != 13
            or result.get("population", {}).get("committed_nodes") != 207
            or result.get("population", {}).get("verified_edges") != 2
            or result.get("optimizer_steps") != 0
            or result.get("model_updates") != 0
            or any(result.get(key) != 0 for key in ("c09_worlds_read", "c10_worlds_read", "mtare_worlds_read"))
        ):
            raise RuntimeError("failure-funnel result contract drift")
        required = [
            "identity_audit.jsonl", "relation_audit.jsonl", "metric_comparison.csv",
            "node_funnel.csv", "edge_funnel.csv", "gse_trace_commit_failure_funnel.png",
            "gse_trace_commit_failure_funnel.pdf", "gse_trace_commit_failure_funnel.svg",
            "figure_source.json",
        ]
        if any(not (run_dir / "artifacts/audit" / name).is_file() for name in required):
            raise RuntimeError("failure-funnel evidence incomplete")
        after = _verify(spec)
        if before != after:
            raise RuntimeError("failure-funnel sources changed")
        overall = PASS
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run_dir / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
    summary = {
        "schema_version": "gse_trace_commit_failure_funnel_outer_v1",
        "overall_status": overall, "error": error, "result": result,
        "executor_returncode": returncode, "duration_seconds": time.monotonic() - started,
        "source_unchanged": bool(before and before == after),
        "optimizer_steps": 0, "model_updates": 0,
        "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0,
    }
    write_json(run_dir / "metrics/summary.json", summary)
    write_json(run_dir / "RUN_STATE.json", {
        "schema_version": "v3_run_state_v1", "run_id": RUN_ID,
        "state": "COMPLETED" if overall == PASS else "FAILED",
        "overall_status": overall, "error": error,
    })
    entries = _seal(run_dir)
    print(json.dumps({"overall_status": overall, "error": error, "seal_entries": entries}, indent=2))
    return 0 if overall == PASS else 2


if __name__ == "__main__":
    raise SystemExit(main())
