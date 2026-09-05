#!/usr/bin/env python3
"""Execute the ERCSS V1R environment-only corrective and seal evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import traceback

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, validate_data_card, write_json


RUN_ID = "gate3_20260829_gse_registered_skeleton_feasibility_v1r_seed0"
PASS = "PASS_GSE_REGISTERED_SKELETON_FEASIBILITY_V1"
FAIL = "FAIL_GSE_REGISTERED_SKELETON_FEASIBILITY_V1"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_GSE_REGISTERED_SKELETON_FEASIBILITY_V1R"
SIDECAR = Path("/tmp/mtare_gate4_meshing_sidecar_v1/bin/python")
TEST_PYTHON = Path("/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python")
DATASET = PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
SUPERVISION = PROJECT_ROOT / "results/gate3_semantics/gate3_20260829_gse_composer_supervision_v1r_seed0"
MESH = PROJECT_ROOT / "results/gate0_baseline/gate0_20260811_cano_100_parent_perception_mesh_m1r_sanitized_assets_seed0"
EVALUATOR = PROJECT_ROOT / "tools/v3/evaluate_gse_registered_skeleton_feasibility_v1.py"
SIDECAR_CONTRACT = PROJECT_ROOT / "configs/v3/gate4/environments/gate4_meshing_sidecar_v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seal(run_dir: Path) -> int:
    destination = run_dir / "artifacts/evidence_sha256.txt"
    paths = sorted(path for path in run_dir.rglob("*") if path.is_file() and path != destination)
    with destination.open("w", encoding="utf-8") as stream:
        for path in paths: stream.write(f"{sha256(path)}  {path.relative_to(PROJECT_ROOT)}\n")
    return len(paths)


def verify_mesh_assets() -> int:
    expected: dict[str, str] = {}
    for line in (MESH / "artifacts/evidence_sha256.txt").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1); expected[relative] = digest
    count = 0
    for primary in sorted((MESH / "artifacts/meshes").glob("*_C0[1-8]/primary")):
        for name in ("mesh.obj", "graph.json", "splines.json", "geometry_parameters.json"):
            path = primary / name; relative = str(path.relative_to(PROJECT_ROOT))
            if relative not in expected or sha256(path) != expected[relative]:
                raise RuntimeError(f"sealed ERCSS mesh asset drift: {relative}")
            count += 1
    if count != 320: raise RuntimeError(f"expected 320 C01-C08 mesh documents, got {count}")
    return count


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--spec", required=True, type=Path); parser.add_argument("--run-dir", required=True, type=Path); args = parser.parse_args()
    run_dir = args.run_dir.resolve(); spec = load_json(args.spec.resolve())
    if run_dir.name != RUN_ID or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED": raise RuntimeError("ERCSS V1R executes exactly once")
    started = time.monotonic(); overall = FAIL; error = None; result: dict = {}; returncode = None; peak_rss = None; verified_mesh_files = 0
    before: dict[str, str] = {}; after: dict[str, str] = {}
    try:
        card = load_json(PROJECT_ROOT / spec["data_card"]); report = validate_data_card(card)
        if spec.get("gate") != 3 or spec.get("operation") != "audit" or not report.passed or card.get("status") != CARD_STATUS: raise RuntimeError("ERCSS V1R scope/card mismatch")
        v1 = load_json(PROJECT_ROOT / "results/gate3_semantics/gate3_20260829_gse_registered_skeleton_feasibility_v1_seed0/metrics/summary.json")
        if v1.get("error") is None or v1.get("registered_skeleton_feasibility") or v1.get("verified_mesh_files") != 320: raise RuntimeError("V1 was not the expected pre-science Open3D system failure")
        for record in spec["frozen_tools"].values():
            if sha256(PROJECT_ROOT / record["path"]) != record["sha256"]: raise RuntimeError(f"frozen tool drift: {record['path']}")
        for relative, expected in spec["frozen_inputs"].items():
            observed = sha256(PROJECT_ROOT / relative)
            if observed != expected: raise RuntimeError(f"frozen input drift: {relative}")
            before[relative] = observed
        contract = load_json(SIDECAR_CONTRACT)
        if str(contract.get("executable")) != str(SIDECAR): raise RuntimeError("sidecar executable contract drift")
        versions = json.loads(subprocess.check_output([str(SIDECAR), "-c", "import json,matplotlib,numpy,open3d,scipy,zarr,sys;print(json.dumps({'python':sys.version.split()[0],'numpy':numpy.__version__,'matplotlib':matplotlib.__version__,'open3d':open3d.__version__,'scipy':scipy.__version__,'zarr':zarr.__version__},sort_keys=True))"], text=True))
        if versions != {"python": "3.12.3", "numpy": "1.26.4", "matplotlib": "3.11.1", "open3d": "0.19.0", "scipy": "1.12.0", "zarr": "2.18.7"}: raise RuntimeError(f"sidecar version drift: {versions}")
        verified_mesh_files = verify_mesh_assets()
        write_json(run_dir / "config/environment.json", {"evaluator_executable": str(SIDECAR), "test_executable": str(TEST_PYTHON), "versions": versions, "cpu_only": True})
        write_json(run_dir / "config/source_integrity_before.json", before)
        write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING"})
        env = os.environ.copy(); env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3"); env["OMP_NUM_THREADS"] = "4"; env["MKL_NUM_THREADS"] = "4"; env["CUDA_VISIBLE_DEVICES"] = ""
        with (run_dir / "logs/00_unit_tests.log").open("w", encoding="utf-8") as stream:
            tests = subprocess.run([str(TEST_PYTHON), "-m", "pytest", "-q", "tests/v3/unit/test_gse_registered_structural_skeleton.py", "tests/v3/unit/test_gse_registered_structural_skeleton_teacher.py"], cwd=PROJECT_ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, text=True, timeout=300, check=False)
        if tests.returncode: raise RuntimeError("ERCSS V1R unit tests failed")
        output = run_dir / "metrics/registered_skeleton_feasibility"
        command = ["/usr/bin/time", "-v", str(SIDECAR), str(EVALUATOR), "--dataset", str(DATASET), "--supervision", str(SUPERVISION), "--mesh-root", str(MESH / "artifacts/meshes"), "--output-dir", str(output)]
        (run_dir / "config/command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
        log = run_dir / "logs/01_registered_skeleton_feasibility.log"
        with log.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, text=True, timeout=10800, check=False)
        returncode = int(completed.returncode); match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", log.read_text(encoding="utf-8")); peak_rss = int(match.group(1)) if match else None
        result = load_json(output / "summary.json") if (output / "summary.json").is_file() else {}; after = {relative: sha256(PROJECT_ROOT / relative) for relative in before}
        required = [output / "summary.json", output / "event_observation_audit.jsonl", output / "identity_coverage.csv", output / "world_summary.jsonl", *[output / f"gse_registered_skeleton_feasibility_v1.{suffix}" for suffix in ("png", "pdf", "svg")]]
        output_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
        if before != after or verified_mesh_files != 320 or returncode not in (0, 2) or result.get("overall_status") not in (PASS, FAIL) or (returncode == 0) != (result.get("overall_status") == PASS) or result.get("worlds") != 80 or result.get("frames") != 252430 or result.get("observations") != 188126 or sum(result.get("event_rows", {}).values()) != 37162 or result.get("optimizer_steps") != 0 or result.get("model_inference_frames") != 0 or any(result.get(name) != 0 for name in ("c09_worlds_read", "c10_worlds_read", "mtare_worlds_read", "graph_replays")) or peak_rss is None or peak_rss > 8 * 1024**2 or output_bytes > 2 * 1024**3 or not all(path.is_file() and path.stat().st_size > 0 for path in required): raise RuntimeError("ERCSS V1R execution/evidence contract failed")
        overall = str(result["overall_status"])
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"; (run_dir / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
    summary = {"schema_version": "gse_registered_skeleton_feasibility_outer_v1r", "overall_status": overall, "scientific_pass": overall == PASS and error is None, "error": error, "returncode": returncode, "peak_host_rss_kib": peak_rss, "verified_mesh_files": verified_mesh_files, "source_unchanged": bool(before and before == after), "v1_system_failure_corrected": True, "registered_skeleton_feasibility": result, "optimizer_steps": 0, "model_inference_frames": 0, "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0, "graph_replays": 0, "duration_seconds": time.monotonic() - started}
    write_json(run_dir / "metrics/summary.json", summary); write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "COMPLETED" if error is None else "FAILED", "overall_status": overall, "error": error})
    entries = seal(run_dir); print(json.dumps({"overall_status": overall, "error": error, "evidence_files": entries}, indent=2)); return 0 if overall == PASS and error is None else 2


if __name__ == "__main__": raise SystemExit(main())
