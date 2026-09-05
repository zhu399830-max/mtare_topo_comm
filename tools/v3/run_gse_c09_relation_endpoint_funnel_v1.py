#!/usr/bin/env python3
"""Execute and seal one immutable C09 relation-endpoint funnel audit."""

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


RUN_ID = "gate3_20260828_gse_c09_relation_endpoint_funnel_v1_seed0"
PASS = "PASS_GSE_C09_RELATION_ENDPOINT_FAILURE_FUNNEL_V1"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_GSE_C09_RELATION_ENDPOINT_FUNNEL_V1"
PYTHON = Path("/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify(spec: dict) -> dict[str, str]:
    values = {}
    for relative, expected in spec["frozen_inputs"].items():
        actual = _sha(PROJECT_ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"C09 endpoint-funnel frozen input drift: {relative}")
        values[relative] = actual
    return values


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
    state = load_json(run_dir / "RUN_STATE.json")
    if run_dir.name != RUN_ID or state.get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("C09 endpoint-funnel may execute only once")
    started = time.monotonic()
    overall, error, result = "FAIL_SYSTEM_GSE_C09_RELATION_ENDPOINT_FUNNEL_V1", None, {}
    before = after = {}
    returncode = None
    try:
        if spec.get("gate") != 3 or spec.get("operation") != "audit" or spec.get("seed") != 0:
            raise RuntimeError("C09 endpoint-funnel scope drift")
        card = load_json(PROJECT_ROOT / spec["data_card"])
        if card.get("status") != CARD_STATUS:
            raise RuntimeError("C09 endpoint-funnel Data Card drift")
        for record in spec["frozen_tools"].values():
            if _sha(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"C09 endpoint-funnel frozen tool drift: {record['path']}")
        before = _verify(spec)
        write_json(run_dir / "config/source_integrity_before.json", before)
        versions = json.loads(subprocess.check_output([
            str(PYTHON), "-c",
            "import json,matplotlib,numpy,scipy,sys;print(json.dumps({'python':sys.version.split()[0],'numpy':numpy.__version__,'scipy':scipy.__version__,'matplotlib':matplotlib.__version__},sort_keys=True))",
        ], text=True))
        expected = {"python": "3.13.5", "numpy": "2.1.3", "scipy": "1.15.3", "matplotlib": "3.10.0"}
        if versions != expected:
            raise RuntimeError(f"C09 endpoint-funnel environment drift: {versions}")
        write_json(run_dir / "config/environment.json", {"executable": str(PYTHON), "versions": versions, "gpu_used": False})
        write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING"})
        validation = PROJECT_ROOT / "results/gate3_semantics/gate3_20260828_gse_endpoint_geometry_c09_validation_v1_seed0"
        teacher = PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_teacher_manifest_v1_seed0/artifacts/teacher_observations.jsonl"
        parents = PROJECT_ROOT / "results/gate0_baseline/gate0_20260811_cano_100_topology_parent_recipe_reclassification_v2r_seed0/artifacts/accepted_parent_manifest.json"
        command = [
            str(PYTHON), str(PROJECT_ROOT / "tools/v3/execute_gse_c09_relation_endpoint_funnel_v1.py"),
            "--validation-run", str(validation), "--teacher", str(teacher),
            "--parent-manifest", str(parents), "--output-dir", str(run_dir / "artifacts/audit"),
        ]
        (run_dir / "config/command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3")
        with (run_dir / "logs/audit.log").open("w", encoding="utf-8") as stream:
            completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env, text=True, stdout=stream, stderr=subprocess.STDOUT, timeout=300, check=False)
        returncode = int(completed.returncode)
        if returncode != 0:
            raise RuntimeError(f"C09 endpoint-funnel executor failed with code {returncode}")
        result = load_json(run_dir / "artifacts/audit/summary.json")
        population = result.get("population", {})
        if (
            result.get("status") != PASS or population.get("objective_relations") != 8
            or population.get("relation_endpoints") != 16 or population.get("raw_edges") != 6
            or population.get("final_edges") != 1 or result.get("endpoint_funnel", {}).get("final_recovered") != 5
            or result.get("relation_stage_counts", {}).get("recovered") != 1
            or any(result.get(key) != 0 for key in (
                "optimizer_steps", "model_updates", "model_inference_frames",
                "threshold_selection_steps", "c10_worlds_read", "mtare_worlds_read",
            ))
        ):
            raise RuntimeError("C09 endpoint-funnel result contract drift")
        required = [
            "summary.json", "endpoint_audit.jsonl", "relation_audit.jsonl", "raw_edge_audit.jsonl",
            "endpoint_funnel.csv", "relation_stage.csv", "figure_source.json",
            "gse_c09_relation_endpoint_funnel.png", "gse_c09_relation_endpoint_funnel.pdf",
            "gse_c09_relation_endpoint_funnel.svg",
        ]
        if any(not (run_dir / "artifacts/audit" / name).is_file() for name in required):
            raise RuntimeError("C09 endpoint-funnel evidence incomplete")
        after = _verify(spec)
        if before != after:
            raise RuntimeError("C09 endpoint-funnel sources changed")
        if sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file()) > 100 * 1024**2:
            raise RuntimeError("C09 endpoint-funnel output exceeds 100 MiB")
        overall = PASS
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run_dir / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
    write_json(run_dir / "metrics/summary.json", {
        "schema_version": "gse_c09_relation_endpoint_funnel_outer_v1",
        "overall_status": overall, "error": error, "result": result,
        "executor_returncode": returncode, "duration_seconds": time.monotonic() - started,
        "source_unchanged": bool(before and before == after),
        "optimizer_steps": 0, "model_updates": 0, "model_inference_frames": 0,
        "threshold_selection_steps": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0,
    })
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
