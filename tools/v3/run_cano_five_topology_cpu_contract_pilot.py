#!/usr/bin/env python3
"""Run and seal the approved five-topology Cano CPU contract exactly once."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


RUN_ID = "gate0_20260811_cano_five_topology_cpu_contract_pilot_v2b_runner_paths_seed0"
E1_PYTHON = Path("/tmp/mtare_cano_compat_e1_py312_np126_sp112/bin/python")
FREEZE_EVIDENCE = PROJECT_ROOT / (
    "results/gate0_baseline/gate0_20260810_cano_dependency_compatibility_matrix_v1_seed0/"
    "config/pip_freeze_E1.txt"
)
PREREQUISITE_DIR = PROJECT_ROOT / (
    "results/gate0_baseline/gate0_20260810_cano_cpu_raycast_lidar_contract_v1_seed0"
)
UPSTREAM_ROOT = PROJECT_ROOT / "external/procedural-subt-gen"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _approved_project_file(raw_path: Any, field_name: str) -> Path:
    """Resolve one spec-bound approval artifact without permitting path escape."""
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise RuntimeError(f"{field_name} must be a non-empty project-relative path")
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"{field_name} must stay inside the project: {raw_path}")
    project_root = PROJECT_ROOT.resolve()
    resolved = (project_root / relative).resolve()
    if project_root not in resolved.parents or not resolved.is_file():
        raise RuntimeError(
            f"{field_name} does not resolve to an existing project file: {raw_path}"
        )
    return resolved


def _e1_identity() -> dict[str, Any]:
    if not E1_PYTHON.is_file():
        raise RuntimeError(f"frozen E1 Python is absent: {E1_PYTHON}")
    program = (
        "import json,matplotlib,numpy,open3d,scipy,sys;"
        "print(json.dumps({'python':sys.version.split()[0],'executable':sys.executable,"
        "'numpy':numpy.__version__,'scipy':scipy.__version__,"
        "'open3d':open3d.__version__,'matplotlib':matplotlib.__version__},sort_keys=True))"
    )
    identity = json.loads(
        subprocess.check_output([str(E1_PYTHON), "-c", program], text=True)
    )
    expected = {
        "python": "3.12.3",
        "numpy": "1.26.4",
        "scipy": "1.12.0",
        "open3d": "0.19.0",
        "matplotlib": "3.11.1",
    }
    observed = {key: identity[key] for key in expected}
    if observed != expected:
        raise RuntimeError(
            f"frozen E1 version mismatch: observed={observed}, expected={expected}"
        )
    pip_check = subprocess.run(
        [str(E1_PYTHON), "-m", "pip", "check"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=120,
        check=False,
    )
    identity["pip_check_exit_code"] = pip_check.returncode
    identity["pip_check_output"] = pip_check.stdout.strip()
    if pip_check.returncode != 0:
        raise RuntimeError(f"frozen E1 pip check failed: {pip_check.stdout}")
    identity["pip_freeze_evidence"] = str(FREEZE_EVIDENCE.relative_to(PROJECT_ROOT))
    identity["pip_freeze_evidence_sha256"] = _sha256(FREEZE_EVIDENCE)
    return identity


def _upstream_identity() -> dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=UPSTREAM_ROOT, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1"], cwd=UPSTREAM_ROOT, text=True
    )
    return {
        "commit": commit,
        "tracked_clean": not bool(status),
        "entrypoint_sha256": _sha256(UPSTREAM_ROOT / "scripts/generate_environments.py"),
    }


def _executor_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join(
                (
                    str(UPSTREAM_ROOT / "src"),
                    str(PROJECT_ROOT / "src"),
                )
            ),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    return environment


def _subt_proc_gen_import_identity(environment: dict[str, str]) -> dict[str, Any]:
    program = (
        "import json,subt_proc_gen.tunnel;"
        "print(json.dumps({'module':subt_proc_gen.tunnel.__file__},sort_keys=True))"
    )
    identity = json.loads(
        subprocess.check_output(
            [str(E1_PYTHON), "-c", program],
            cwd=PROJECT_ROOT,
            env=environment,
            text=True,
        )
    )
    imported = Path(identity["module"]).resolve()
    expected_root = (UPSTREAM_ROOT / "src").resolve()
    identity["from_fixed_checkout"] = imported.is_relative_to(expected_root)
    if not identity["from_fixed_checkout"]:
        raise RuntimeError(f"subt_proc_gen import is not pinned to {expected_root}: {imported}")
    return identity


def _seal_manifest(run_dir: Path) -> int:
    destination = run_dir / "artifacts/evidence_sha256.txt"
    files = sorted(
        path for path in run_dir.rglob("*") if path.is_file() and path != destination
    )
    destination.write_text(
        "".join(
            f"{_sha256(path)}  {path.relative_to(PROJECT_ROOT)}\n" for path in files
        ),
        encoding="utf-8",
    )
    return len(files)


def _run_executor(run_dir: Path) -> tuple[int, float]:
    argv = [
        str(E1_PYTHON),
        str(PROJECT_ROOT / "tools/v3/execute_cano_five_topology_cpu_contract_pilot.py"),
        "--run-dir",
        str(run_dir),
    ]
    environment = _executor_environment()
    started = time.monotonic()
    output = ""
    exit_code = 124
    try:
        completed = subprocess.run(
            argv,
            cwd=PROJECT_ROOT,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=7200,
            check=False,
        )
        output = completed.stdout
        exit_code = completed.returncode
    except subprocess.TimeoutExpired as exc:
        captured = exc.stdout or ""
        output = captured.decode() if isinstance(captured, bytes) else captured
        output += "\nTIMEOUT: frozen 7200 second limit reached.\n"
    duration = time.monotonic() - started
    log_path = run_dir / "logs/01_five_topology_cpu_contract.log"
    log_path.write_text(
        "argv="
        + json.dumps(argv, ensure_ascii=False)
        + "\n"
        + f"finished_at_utc={datetime.now(timezone.utc).isoformat()}\n"
        + output
        + f"\nduration_seconds={duration:.6f}\nexit_code={exit_code}\n",
        encoding="utf-8",
    )
    print(output, end="", flush=True)
    return exit_code, duration


def _inspect_outputs(run_dir: Path) -> dict[str, Any]:
    missing: list[str] = []
    required = [
        run_dir / "metrics/summary.json",
        run_dir / "metrics/replay_and_seed_separation.json",
        run_dir / "artifacts/observation_manifest.json",
        run_dir / "previews/provenance.json",
        run_dir / "previews/five_topology_branch_summary.png",
    ]
    for path in required:
        if not path.is_file():
            missing.append(str(path.relative_to(run_dir)))

    worlds = sorted((run_dir / "artifacts/worlds").glob("P*"))
    shards = sorted((run_dir / "artifacts/shards").glob("*.npz"))
    maps = sorted((run_dir / "previews/world_maps").glob("*.png"))
    pages = sorted((run_dir / "previews/contact_pages").glob("*.png"))
    parent_metrics = sorted((run_dir / "metrics").glob("P*.json"))
    required_world_files = (
        "mesh.obj",
        "graph.json",
        "splines.json",
        "axis.txt",
        "fta_dist.txt",
        "anchors.json",
        "metadata.json",
    )
    for world in worlds:
        for name in required_world_files:
            path = world / name
            if not path.is_file():
                missing.append(str(path.relative_to(run_dir)))

    shard_contracts = []
    for path in shards:
        with np.load(path, allow_pickle=False) as archive:
            keys = sorted(archive.files)
            ranges = archive["student_range_m"] if "student_range_m" in keys else None
            valid = archive["student_valid_mask"] if "student_valid_mask" in keys else None
            labels = archive["teacher_exit_label_720"] if "teacher_exit_label_720" in keys else None
            shard_contracts.append(
                {
                    "path": str(path.relative_to(PROJECT_ROOT)),
                    "keys": keys,
                    "range_shape": list(ranges.shape) if ranges is not None else None,
                    "range_dtype": str(ranges.dtype) if ranges is not None else None,
                    "valid_shape": list(valid.shape) if valid is not None else None,
                    "valid_dtype": str(valid.dtype) if valid is not None else None,
                    "label_shape": list(labels.shape) if labels is not None else None,
                    "label_dtype": str(labels.dtype) if labels is not None else None,
                    "sha256": _sha256(path),
                }
            )
    expected_keys = [
        "student_range_m",
        "student_valid_mask",
        "teacher_exit_label_720",
    ]
    shards_pass = len(shard_contracts) == 5 and all(
        item["keys"] == expected_keys
        and item["range_shape"] == [150, 16, 720]
        and item["range_dtype"] == "float32"
        and item["valid_shape"] == [150, 16, 720]
        and item["valid_dtype"] == "uint8"
        and item["label_shape"] == [150, 720]
        and item["label_dtype"] == "float32"
        for item in shard_contracts
    )
    manifest_count = None
    if (run_dir / "artifacts/observation_manifest.json").is_file():
        manifest = load_json(run_dir / "artifacts/observation_manifest.json")
        manifest_count = len(manifest.get("observations", []))
    return {
        "missing_required_evidence": missing,
        "counts": {
            "world_bundles": len(worlds),
            "diagnostic_shards": len(shards),
            "parent_metric_files": len(parent_metrics),
            "world_maps": len(maps),
            "contact_pages": len(pages),
            "manifest_observations": manifest_count,
        },
        "shard_contracts": shard_contracts,
        "shards_pass": shards_pass,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    spec = load_json(args.spec.resolve())
    run_dir = args.run_dir.resolve()
    if run_dir.name != RUN_ID or not run_dir.is_dir():
        raise RuntimeError(f"runner accepts only the created run directory {RUN_ID}")
    run_state = load_json(run_dir / "RUN_STATE.json")
    if run_state.get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("run is not in the one-time CREATED_NOT_EXECUTED state")
    authorization = spec.get("user_authorization", {})
    if (
        spec.get("operation") != "sensor_contract_pilot"
        or spec.get("gate") != 0
        or spec.get("seed") != 0
        or authorization.get("status") != "APPROVED"
    ):
        raise RuntimeError("spec is not the approved Gate-0 five-topology contract pilot")
    expected_scope = {
        "topology_parents": 5,
        "primary_topology_constructions": 5,
        "topology_replay_constructions": 5,
        "native_perception_meshes": 5,
        "canonical_anchors": 250,
        "diagnostic_observations": 750,
        "rays_per_observation": 11520,
        "independent_scene_passes": 2,
        "total_primary_rays": 17280000,
        "formal_dataset_worlds": 0,
        "formal_dataset_samples": 0,
        "labels_for_training": 0,
        "training_samples": 0,
        "models": 0,
        "ai_labels": 0,
        "trajectories": 0,
        "isaac_runs": 0,
        "gazebo_runs": 0,
        "topology_runtime_changes": 0,
        "mtare_changes": 0,
    }
    if spec.get("data_scope") != expected_scope:
        raise RuntimeError(f"approved pilot scope drifted: {spec.get('data_scope')}")

    tool_paths = {
        "runner": Path(__file__).resolve(),
        "executor": PROJECT_ROOT / "tools/v3/execute_cano_five_topology_cpu_contract_pilot.py",
        "support": PROJECT_ROOT / "tools/v3/cano_five_topology_support.py",
        "contract": PROJECT_ROOT / "src/mtare_topo/data/cano_contract_pilot.py",
        "sensor": PROJECT_ROOT / "src/mtare_topo/data/cano_sensor_smoke.py",
        "baseline": PROJECT_ROOT / "src/mtare_topo/semantics/range_exit_baseline.py",
        "governance": PROJECT_ROOT / "src/mtare_topo/governance.py",
        "unit_test": PROJECT_ROOT / "tests/v3/unit/test_cano_contract_pilot.py",
        "external_test": PROJECT_ROOT / "tests/v3/external/test_cano_adapter_contract.py",
        "proposal": _approved_project_file(spec.get("config_path"), "config_path"),
        "data_card": _approved_project_file(spec.get("data_card"), "data_card"),
    }
    observed_hashes = {name: _sha256(path) for name, path in tool_paths.items()}
    for name, frozen in spec.get("frozen_tools", {}).items():
        if observed_hashes.get(name) != frozen.get("sha256"):
            raise RuntimeError(
                f"frozen hash mismatch for {name}: "
                f"{observed_hashes.get(name)} != {frozen.get('sha256')}"
            )

    prerequisite = load_json(PREREQUISITE_DIR / "metrics/summary.json")
    prerequisite_state = load_json(PREREQUISITE_DIR / "RUN_STATE.json")
    if (
        prerequisite.get("overall_status") != "PASS_CANO_CPU_RAYCAST_LIDAR_CONTRACT"
        or prerequisite_state.get("state") != "COMPLETED"
    ):
        raise RuntimeError("one-world CPU LiDAR prerequisite is not a completed PASS")
    upstream = _upstream_identity()
    if upstream != spec.get("upstream_identity"):
        raise RuntimeError(f"upstream identity mismatch: {upstream}")
    environment = _e1_identity()
    if environment["pip_freeze_evidence_sha256"] != spec.get(
        "pip_freeze_evidence_sha256"
    ):
        raise RuntimeError("frozen E1 pip-freeze evidence identity mismatch")
    executor_environment = _executor_environment()
    import_identity = _subt_proc_gen_import_identity(executor_environment)

    write_json(run_dir / "config/tool_hashes.json", observed_hashes)
    write_json(run_dir / "config/upstream_identity.json", upstream)
    write_json(run_dir / "config/e1_environment.json", environment)
    write_json(run_dir / "config/subt_proc_gen_import_identity.json", import_identity)
    write_json(
        run_dir / "config/prerequisite_identity.json",
        {
            "path": str(PREREQUISITE_DIR.relative_to(PROJECT_ROOT)),
            "summary_sha256": _sha256(PREREQUISITE_DIR / "metrics/summary.json"),
            "run_state_sha256": _sha256(PREREQUISITE_DIR / "RUN_STATE.json"),
            "overall_status": prerequisite["overall_status"],
        },
    )
    write_json(
        run_dir / "RUN_STATE.json",
        {
            "schema_version": "v3_run_state_v1",
            "run_id": RUN_ID,
            "state": "RUNNING",
            "note": "Approved five-topology diagnostic contract; zero formal data and training.",
        },
    )

    exit_code, duration = _run_executor(run_dir)
    inspection = _inspect_outputs(run_dir)
    executor_summary_path = run_dir / "metrics/summary.json"
    executor_summary = (
        load_json(executor_summary_path) if executor_summary_path.is_file() else None
    )
    counts = inspection["counts"]
    result_bytes = sum(
        path.stat().st_size for path in run_dir.rglob("*") if path.is_file()
    )
    expected_executor_scope = {
        "topology_parents": 5,
        "native_perception_meshes": 5,
        "canonical_anchors": 250,
        "diagnostic_observations": 750,
        "rays_per_observation": 11520,
        "independent_scene_passes": 2,
        "total_primary_rays": 17280000,
        "formal_dataset_worlds": 0,
        "formal_dataset_samples": 0,
        "labels_for_training": 0,
        "training_samples": 0,
        "models": 0,
        "ai_labels": 0,
        "isaac_runs": 0,
        "gazebo_runs": 0,
        "topology_runtime_changes": 0,
        "mtare_changes": 0,
        "trajectories": 0,
        "online_graph_metrics": "NOT_APPLICABLE_STATIC_ANCHORS_HAVE_NO_CAUSAL_MOVEMENT_EDGES",
    }
    counts_pass = counts == {
        "world_bundles": 5,
        "diagnostic_shards": 5,
        "parent_metric_files": 5,
        "world_maps": 5,
        "contact_pages": 30,
        "manifest_observations": 750,
    }
    passed = bool(
        exit_code == 0
        and duration <= 7200
        and executor_summary is not None
        and executor_summary.get("overall_status")
        == "PASS_CANO_FIVE_TOPOLOGY_CPU_CONTRACT_PILOT"
        and executor_summary.get("scope") == expected_executor_scope
        and executor_summary.get("source_unchanged") is True
        and not inspection["missing_required_evidence"]
        and inspection["shards_pass"]
        and counts_pass
        and result_bytes <= 2 * 1024**3
    )
    overall = (
        "PASS_CANO_FIVE_TOPOLOGY_CPU_CONTRACT_PILOT"
        if passed
        else "FAIL_CANO_FIVE_TOPOLOGY_CPU_CONTRACT_PILOT"
    )
    runner_summary = {
        "schema_version": "cano_five_topology_cpu_contract_runner_summary_v1",
        "run_id": RUN_ID,
        "overall_status": overall,
        "executor_exit_code": exit_code,
        "duration_seconds": duration,
        "inspection": inspection,
        "counts_pass": counts_pass,
        "result_bytes_before_seal": result_bytes,
        "disk_limit_bytes": 2 * 1024**3,
        "host": {
            "platform": platform.platform(),
            "free_disk_bytes_after": shutil.disk_usage(PROJECT_ROOT).free,
        },
        "claim_boundary": (
            "This run is diagnostic static-view evidence only. It creates no formal dataset, "
            "learned model, causal trajectory graph, navigation result or M-TARE comparison."
        ),
    }
    write_json(run_dir / "metrics/runner_summary.json", runner_summary)
    write_json(
        run_dir / "RUN_STATE.json",
        {
            "schema_version": "v3_run_state_v1",
            "run_id": RUN_ID,
            "state": "COMPLETED" if passed else "FAILED",
            "overall_status": overall,
            "note": runner_summary["claim_boundary"],
        },
    )
    sealed_files = _seal_manifest(run_dir)
    print(json.dumps({"overall_status": overall, "sealed_files": sealed_files}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
