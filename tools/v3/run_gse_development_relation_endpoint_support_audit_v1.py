#!/usr/bin/env python3
"""Execute and seal the C01-C08 relation-endpoint support audit."""

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


RUN_ID = "gate3_20260828_gse_development_relation_endpoint_support_audit_v1_seed0"
PASS = "PASS_GSE_DEVELOPMENT_RELATION_ENDPOINT_SUPPORT_AUDIT_V1"
PYTHON = Path("/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python")


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
        if actual != expected: raise RuntimeError(f"development endpoint-support input drift: {relative}")
        result[relative] = actual
    return result


def _seal(run_dir: Path) -> int:
    seal = run_dir / "artifacts/evidence_sha256.txt"
    files = sorted(path for path in run_dir.rglob("*") if path.is_file() and path != seal)
    with seal.open("w", encoding="utf-8") as stream:
        for path in files: stream.write(f"{_sha(path)}  {path.relative_to(PROJECT_ROOT)}\n")
    return len(files)


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--spec", required=True, type=Path); parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args(); run_dir = args.run_dir.resolve(); spec = load_json(args.spec.resolve())
    if run_dir.name != RUN_ID or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("development endpoint-support audit may execute only once")
    started = time.monotonic(); overall = "FAIL_SYSTEM_GSE_DEVELOPMENT_RELATION_ENDPOINT_SUPPORT_AUDIT_V1"; error = None; result = {}; before = after = {}; returncode = None
    try:
        if spec.get("gate") != 3 or spec.get("operation") != "audit" or spec.get("seed") != 0: raise RuntimeError("development endpoint-support scope drift")
        card = load_json(PROJECT_ROOT / spec["data_card"])
        if card.get("status") != "APPROVED_FOR_ONE_IMMUTABLE_GSE_DEVELOPMENT_RELATION_ENDPOINT_SUPPORT_AUDIT_V1": raise RuntimeError("development endpoint-support Data Card drift")
        for record in spec["frozen_tools"].values():
            if _sha(PROJECT_ROOT / record["path"]) != record["sha256"]: raise RuntimeError(f"development endpoint-support tool drift: {record['path']}")
        before = _verify(spec); write_json(run_dir / "config/source_integrity_before.json", before)
        versions = json.loads(subprocess.check_output([str(PYTHON), "-c", "import json,matplotlib,numpy,scipy,sys;print(json.dumps({'python':sys.version.split()[0],'numpy':numpy.__version__,'scipy':scipy.__version__,'matplotlib':matplotlib.__version__},sort_keys=True))"], text=True))
        expected = {"python": "3.13.5", "numpy": "2.1.3", "scipy": "1.15.3", "matplotlib": "3.10.0"}
        if versions != expected: raise RuntimeError(f"development endpoint-support environment drift: {versions}")
        write_json(run_dir / "config/environment.json", {"executable": str(PYTHON), "versions": versions, "gpu_used": False})
        write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING"})
        teacher = PROJECT_ROOT / "results/gate2_representation/gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0/artifacts/teacher_observations.jsonl"
        replay = PROJECT_ROOT / "results/gate3_semantics/gate3_20260828_gse_spatial_trace_commit_requalification_v1r2_seed0/artifacts"
        pair_cache = PROJECT_ROOT / "results/gate3_semantics/gate3_20260826_gse_exit_token_association_corrective_v2_seed0/artifacts/pair_cache/pairs.npz"
        capacity = PROJECT_ROOT / "results/gate3_semantics/gate3_20260828_gse_endpoint_geometry_capacity_v1_seed0"
        command = [str(PYTHON), str(PROJECT_ROOT / "tools/v3/execute_gse_development_relation_endpoint_support_audit_v1.py"), "--teacher", str(teacher), "--pair-cache", str(pair_cache), "--action-ensemble", str(replay / "replay/action_ensemble.npz"), "--spatial-projection", str(replay / "projection/spatial_center_ensemble_all_rows.npz"), "--association-pairs", str(replay / "replay/association_pairs.npz"), "--capacity-run", str(capacity), "--output-dir", str(run_dir / "artifacts/audit")]
        (run_dir / "config/command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
        env = os.environ.copy(); env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3")
        with (run_dir / "logs/audit.log").open("w", encoding="utf-8") as stream:
            completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env, text=True, stdout=stream, stderr=subprocess.STDOUT, timeout=600, check=False)
        returncode = int(completed.returncode)
        if returncode != 0: raise RuntimeError(f"development endpoint-support executor failed with code {returncode}")
        result = load_json(run_dir / "artifacts/audit/summary.json")
        if result.get("status") != PASS or result.get("population", {}).get("fit", {}).get("relation_endpoints") != 72 or result.get("population", {}).get("selection", {}).get("relation_endpoints") != 26 or any(result.get(key) != 0 for key in ("optimizer_steps", "model_updates", "model_inference_frames", "c09_worlds_read", "c10_worlds_read", "mtare_worlds_read")): raise RuntimeError("development endpoint-support result contract drift")
        required = ["summary.json", "endpoint_support_audit.jsonl", "relation_audit.jsonl", "support_bin_stats.csv", "figure_source.json", "gse_development_relation_endpoint_support.png", "gse_development_relation_endpoint_support.pdf", "gse_development_relation_endpoint_support.svg"]
        if any(not (run_dir / "artifacts/audit" / name).is_file() for name in required): raise RuntimeError("development endpoint-support evidence incomplete")
        after = _verify(spec)
        if before != after: raise RuntimeError("development endpoint-support sources changed")
        overall = PASS
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"; (run_dir / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
    write_json(run_dir / "metrics/summary.json", {"schema_version": "gse_development_relation_endpoint_support_outer_v1", "overall_status": overall, "error": error, "result": result, "executor_returncode": returncode, "duration_seconds": time.monotonic() - started, "source_unchanged": bool(before and before == after), "optimizer_steps": 0, "model_updates": 0, "model_inference_frames": 0, "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0})
    write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "COMPLETED" if overall == PASS else "FAILED", "overall_status": overall, "error": error})
    entries = _seal(run_dir); print(json.dumps({"overall_status": overall, "error": error, "seal_entries": entries}, indent=2)); return 0 if overall == PASS else 2


if __name__ == "__main__": raise SystemExit(main())
