#!/usr/bin/env python3
"""Run and seal the approved Gate-4 C09 causal topology validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


RUN_ID = "gate4_20260820_cano_c09_causal_topology_validation_v1_seed0"
E1 = Path("/tmp/mtare_gate4_meshing_sidecar_v1/bin/python")
TORCH = Path("/tmp/mtare_gate4_torch290_cu129_zarr2187/bin/python")
STAGES = (
    ("01_sensor", E1, PROJECT_ROOT / "tools/v3/execute_cano_c09_causal_sensor_stage_v1.py", 9000),
    ("02_frozen_inference", TORCH, PROJECT_ROOT / "tools/v3/execute_cano_c09_frozen_inference_stage_v1.py", 1800),
    ("03_causal_graph_validation", TORCH, PROJECT_ROOT / "tools/v3/execute_cano_c09_causal_graph_validation_stage_v1.py", 1800),
)
EXPECTED_ENVIRONMENT_FREEZES = {
    "sensor": "6447ba5efb58f9458e17aa9cb28db38b5aa4dfef5d59633e1072d1ea119c4035",
    "inference_graph": "fccb0fe667ca7164294619bef3f9f6c5be81deb42b1c5c9554e7b562bd9b23ae",
}
WORLD_NAMES = (
    "flat_tree_small", "3d_tree_small", "flat_unicyclic_small", "3d_unicyclic_small",
    "flat_branch_medium", "3d_branch_medium", "flat_loop_rich", "3d_loop_rich",
    "flat_complex", "3d_complex",
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seal(run_dir: Path) -> int:
    destination = run_dir / "artifacts/evidence_sha256.txt"
    files = sorted(path for path in run_dir.rglob("*") if path.is_file() and path != destination)
    destination.write_text("".join(f"{sha(path)}  {path.relative_to(PROJECT_ROOT)}\n" for path in files), encoding="utf-8")
    return len(files)


def verify_source_seal(path: Path) -> int:
    entries = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        if "_C10" in relative or "M-TARE" in relative or "mtare" in relative.lower():
            raise RuntimeError(f"forbidden source in seal: {relative}")
        if sha(PROJECT_ROOT / relative) != expected:
            raise RuntimeError(f"source seal mismatch: {relative}")
        entries += 1
    return entries


def native_c09_bundle_sha256() -> tuple[str, int]:
    base = PROJECT_ROOT / "results/gate0_baseline/gate0_20260811_cano_100_parent_perception_mesh_m1r_sanitized_assets_seed0/artifacts/meshes"
    worlds = [f"S{index:02d}_{name}_C09" for index, name in enumerate(WORLD_NAMES, 1)]
    files = sorted(base / world / "primary" / name for world in worlds
                   for name in ("geometry_parameters.json", "graph.json", "mesh.obj", "splines.json"))
    lines = "".join(f"{sha(path)}  {path.relative_to(PROJECT_ROOT)}\n" for path in files)
    return hashlib.sha256(lines.encode()).hexdigest(), len(files)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args(); spec = load_json(args.spec.resolve()); run_dir = args.run_dir.resolve()
    if run_dir.name != RUN_ID or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("run identity/state mismatch")
    if spec.get("gate") != 4 or spec.get("operation") != "topology_replay" or spec.get("causal_replay_only") is not True:
        raise RuntimeError("scope mismatch")
    card = load_json(PROJECT_ROOT / spec["data_card"])
    if card.get("approval", {}).get("status") != "APPROVED" or "topology_replay" not in card["approval"].get("authorized_operations", []):
        raise RuntimeError("data card is not approved")
    for relative, expected in spec["frozen_inputs"].items():
        if sha(PROJECT_ROOT / relative) != expected:
            raise RuntimeError(f"frozen input drift: {relative}")
    geometry_seal = PROJECT_ROOT / spec["geometry_source_seal"]
    corrective_seal = PROJECT_ROOT / spec["corrective_audit_seal"]
    source_seals = {"geometry_entries": verify_source_seal(geometry_seal),
                    "corrective_entries": verify_source_seal(corrective_seal)}
    native_bundle, native_files = native_c09_bundle_sha256()
    if native_bundle != spec.get("native_c09_asset_bundle_sha256") or native_files != 40:
        raise RuntimeError(f"native C09 asset bundle drift: {native_bundle}/{native_files}")
    write_json(run_dir / "config/input_integrity.json", {
        **source_seals, "native_c09_files": native_files,
        "native_c09_asset_bundle_sha256": native_bundle,
    })
    observed = {}
    for name, item in spec["frozen_tools"].items():
        observed[name] = sha(PROJECT_ROOT / item["path"])
        if observed[name] != item["sha256"]:
            raise RuntimeError(f"frozen tool drift: {name}")
    write_json(run_dir / "config/tool_hashes.json", observed)
    identities = {}
    for name, executable in (("sensor", E1), ("inference_graph", TORCH)):
        frozen = subprocess.check_output([str(executable), "-m", "pip", "freeze"], text=True)
        normalized = "\n".join(sorted(line for line in frozen.splitlines() if line.strip())) + "\n"
        freeze_sha256 = hashlib.sha256(normalized.encode()).hexdigest()
        if freeze_sha256 != EXPECTED_ENVIRONMENT_FREEZES[name]:
            raise RuntimeError(f"{name} sidecar freeze drift: {freeze_sha256}")
        check = subprocess.run([str(executable), "-m", "pip", "check"], text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        if check.returncode != 0:
            raise RuntimeError(f"{name} sidecar pip check failed: {check.stdout}")
        code = "import json,sys,numpy,zarr;d={'python':sys.version.split()[0],'numpy':numpy.__version__,'zarr':zarr.__version__};"
        code += "\ntry:\n import open3d;d['open3d']=open3d.__version__\nexcept ImportError:pass"
        code += "\ntry:\n import torch;d['torch']=torch.__version__;d['cuda']=torch.cuda.is_available()\nexcept ImportError:pass\nprint(json.dumps(d))"
        identities[name] = json.loads(subprocess.check_output([str(executable), "-c", code], text=True))
        identities[name].update({"pip_freeze_sha256": freeze_sha256, "pip_check": check.stdout.strip()})
        (run_dir / f"config/{name}_pip_freeze.txt").write_text(normalized, encoding="utf-8")
    write_json(run_dir / "config/executor_environment_identity.json", identities)
    write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING"})
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "tools/v3")))
    environment.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    stage_records, all_passed, started = [], True, time.monotonic()
    for name, executable, script, timeout_s in STAGES:
        argv = [str(executable), str(script), "--run-dir", str(run_dir)]
        stage_started = time.monotonic()
        completed = subprocess.run(argv, cwd=PROJECT_ROOT, env=environment, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   timeout=timeout_s, check=False)
        duration = time.monotonic() - stage_started
        log_path = run_dir / f"logs/{name}.log"
        log_path.write_text(f"argv={json.dumps(argv)}\nfinished_at_utc={datetime.now(timezone.utc).isoformat()}\n{completed.stdout}\nduration_seconds={duration:.6f}\nexit_code={completed.returncode}\n", encoding="utf-8")
        print(completed.stdout, end="", flush=True)
        stage_records.append({"name": name, "exit_code": completed.returncode, "duration_seconds": duration, "log": str(log_path.relative_to(run_dir))})
        if completed.returncode != 0:
            all_passed = False; break
    sensor = load_json(run_dir / "metrics/sensor_summary.json") if (run_dir / "metrics/sensor_summary.json").is_file() else {}
    inference = load_json(run_dir / "metrics/inference_summary.json") if (run_dir / "metrics/inference_summary.json").is_file() else {}
    graph = load_json(run_dir / "metrics/graph_summary.json") if (run_dir / "metrics/graph_summary.json").is_file() else {}
    passed = bool(all_passed
        and sensor.get("overall_status") == "PASS_C09_CAUSAL_SENSOR_STAGE_V1"
        and sensor.get("frames") == 15833 and sensor.get("dual_scene_full_scan_rays") == 364792320
        and sensor.get("all_scans_finite") is True and sensor.get("all_branch_los_passed") is True
        and sensor.get("all_dual_scene_frames_equal") is True
        and inference.get("overall_status") == "PASS_C09_FROZEN_INFERENCE_STAGE_V1"
        and inference.get("model_inference_frames") == 47499
        and graph.get("overall_status") == "PASS_C09_CAUSAL_GRAPH_VALIDATION_V1"
        and graph.get("parameter_configurations") == 1 and graph.get("graph_replays") == 50
        and graph.get("frozen_parameter_id") == "sf2_tr8_lr6_hh20_te45_da20"
        and graph.get("selection_performed_on_c09") is False and graph.get("oracle_integrity_passed") is True
        and graph.get("c10_worlds_read") == graph.get("mtare_worlds_read") == 0)
    overall = "PASS_CANO_C09_CAUSAL_TOPOLOGY_VALIDATION_V1" if passed else "FAIL_CANO_C09_CAUSAL_TOPOLOGY_VALIDATION_V1"
    write_json(run_dir / "metrics/summary.json", {
        "schema_version": "cano_c09_causal_topology_validation_runner_v1", "overall_status": overall,
        "stages": stage_records, "duration_seconds": time.monotonic() - started,
        "unique_sensor_frames": sensor.get("frames", 0), "dual_scene_full_scan_rays": sensor.get("dual_scene_full_scan_rays", 0),
        "model_inference_frames": inference.get("model_inference_frames", 0), "graph_replays": graph.get("graph_replays", 0),
        "frozen_parameter_id": graph.get("frozen_parameter_id"),
        "result_bytes_before_seal": sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file()),
        "free_disk_bytes_after": shutil.disk_usage(PROJECT_ROOT).free,
        "claim_boundary": "C09 frozen-parameter validation only; C09 participated in checkpoint selection, so this is not strict end-to-end unseen generalization. No C10, planner or M-TARE."})
    write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID,
                                             "state": "COMPLETED" if passed else "FAILED", "overall_status": overall})
    count = seal(run_dir)
    print(json.dumps({"overall_status": overall, "sealed_files": count}, indent=2), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
