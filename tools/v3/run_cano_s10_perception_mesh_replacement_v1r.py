#!/usr/bin/env python3
"""Run the approved S10 replacement with the frozen Cano M1R environment."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json
from run_cano_five_topology_cpu_contract_pilot import (
    _executor_environment,
    _upstream_identity,
)


RUN_ID = "gate4_20260813_cano_s10_perception_mesh_replacement_v1r_seed0"
EXECUTOR = PROJECT_ROOT / "tools/v3/execute_cano_s10_perception_mesh_replacement_v1.py"
E1_PYTHON = Path("/tmp/mtare_cano_compat_e1_rebuilt_20260813/bin/python")
HISTORICAL_FREEZE = PROJECT_ROOT / (
    "results/gate0_baseline/gate0_20260810_cano_dependency_compatibility_matrix_v1_seed0/"
    "config/pip_freeze_E1.txt"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seal(run_dir: Path) -> int:
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


def _normalized_freeze(lines: str) -> list[str]:
    normalized = []
    for line in lines.splitlines():
        if line.startswith("SubterraneanProceduralGeneration @ file://"):
            line = "SubterraneanProceduralGeneration @ LOCAL_CHECKOUT"
        normalized.append(line.lower())
    return sorted(normalized)


def _rebuilt_e1_identity() -> dict:
    if not E1_PYTHON.is_file():
        raise RuntimeError(f"rebuilt E1 is absent: {E1_PYTHON}")
    program = (
        "import json,matplotlib,numpy,open3d,perlin_numpy,scipy,sys;"
        "print(json.dumps({'python':sys.version.split()[0],'executable':sys.executable,"
        "'numpy':numpy.__version__,'scipy':scipy.__version__,'open3d':open3d.__version__,"
        "'matplotlib':matplotlib.__version__,'perlin_numpy_file':perlin_numpy.__file__},sort_keys=True))"
    )
    identity = json.loads(subprocess.check_output([str(E1_PYTHON), "-c", program], text=True))
    expected = {
        "python": "3.12.3",
        "numpy": "1.26.4",
        "scipy": "1.12.0",
        "open3d": "0.19.0",
        "matplotlib": "3.11.1",
    }
    if {key: identity[key] for key in expected} != expected:
        raise RuntimeError(f"rebuilt E1 version mismatch: {identity}")
    pip_check = subprocess.run(
        [str(E1_PYTHON), "-m", "pip", "check"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=120,
        check=False,
    )
    freeze = subprocess.check_output([str(E1_PYTHON), "-m", "pip", "freeze"], text=True)
    historical = HISTORICAL_FREEZE.read_text(encoding="utf-8")
    identity.update(
        {
            "pip_check_exit_code": pip_check.returncode,
            "pip_check_output": pip_check.stdout.strip(),
            "pip_freeze_line_count": len(freeze.splitlines()),
            "pip_freeze_sha256": hashlib.sha256(freeze.encode()).hexdigest(),
            "historical_freeze_sha256": _sha256(HISTORICAL_FREEZE),
            "normalized_freeze_equal": _normalized_freeze(freeze)
            == _normalized_freeze(historical),
        }
    )
    if pip_check.returncode != 0 or identity["pip_freeze_line_count"] != 98:
        raise RuntimeError(f"rebuilt E1 integrity failure: {identity}")
    if not identity["normalized_freeze_equal"]:
        raise RuntimeError("rebuilt E1 freeze differs from sealed historical freeze")
    return identity


def _rebuilt_executor_environment() -> dict[str, str]:
    environment = _executor_environment()
    return environment


def _rebuilt_import_identity(environment: dict[str, str]) -> dict:
    program = (
        "import json,perlin_numpy,subt_proc_gen.mesh_generation;"
        "print(json.dumps({'module':subt_proc_gen.mesh_generation.__file__,"
        "'perlin_numpy':perlin_numpy.__file__},sort_keys=True))"
    )
    identity = json.loads(
        subprocess.check_output(
            [str(E1_PYTHON), "-c", program],
            cwd=PROJECT_ROOT,
            env=environment,
            text=True,
        )
    )
    expected_root = (PROJECT_ROOT / "external/procedural-subt-gen/src").resolve()
    identity["from_fixed_checkout"] = Path(identity["module"]).resolve().is_relative_to(
        expected_root
    )
    identity["perlin_numpy_present"] = Path(identity["perlin_numpy"]).is_file()
    if not identity["from_fixed_checkout"] or not identity["perlin_numpy_present"]:
        raise RuntimeError(f"rebuilt full import identity failed: {identity}")
    return identity


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    spec = load_json(args.spec.resolve())
    run_dir = args.run_dir.resolve()

    if run_dir.name != RUN_ID:
        raise RuntimeError("run identity mismatch")
    if load_json(run_dir / "RUN_STATE.json")["state"] != "CREATED_NOT_EXECUTED":
        raise RuntimeError("run state mismatch")
    if spec.get("gate") != 4 or spec.get("s10_mesh_replacement_only") is not True:
        raise RuntimeError("scope mismatch")
    card = load_json(PROJECT_ROOT / spec["data_card"])
    if card["approval"]["status"] != "APPROVED":
        raise RuntimeError("data card is not approved")

    for relative, expected in spec["frozen_inputs"].items():
        if _sha256(PROJECT_ROOT / relative) != expected:
            raise RuntimeError(f"input drift: {relative}")
    observed_tools = {}
    for name, item in spec["frozen_tools"].items():
        observed_tools[name] = _sha256(PROJECT_ROOT / item["path"])
        if observed_tools[name] != item["sha256"]:
            raise RuntimeError(f"tool drift: {name}")

    environment = _rebuilt_executor_environment()
    environment_identity = _rebuilt_e1_identity()
    upstream_identity = _upstream_identity()
    import_identity = _rebuilt_import_identity(environment)
    write_json(run_dir / "config/tool_hashes.json", observed_tools)
    write_json(run_dir / "config/e1_identity.json", environment_identity)
    write_json(run_dir / "config/upstream_identity.json", upstream_identity)
    write_json(run_dir / "config/subt_proc_gen_import_identity.json", import_identity)
    write_json(
        run_dir / "RUN_STATE.json",
        {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING"},
    )

    argv = [str(E1_PYTHON), str(EXECUTOR), "--run-dir", str(run_dir)]
    started = time.monotonic()
    completed = subprocess.run(
        argv,
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=300,
        check=False,
    )
    duration = time.monotonic() - started
    (run_dir / "logs/01_s10_mesh_replacement.log").write_text(
        "argv="
        + json.dumps(argv)
        + "\n"
        + f"finished_at_utc={datetime.now(timezone.utc).isoformat()}\n"
        + completed.stdout
        + f"\nduration_seconds={duration:.6f}\nexit_code={completed.returncode}\n",
        encoding="utf-8",
    )
    print(completed.stdout, end="", flush=True)

    summary = (
        load_json(run_dir / "metrics/summary.json")
        if (run_dir / "metrics/summary.json").is_file()
        else {}
    )
    passed = (
        completed.returncode == 0
        and summary.get("overall_status") == "PASS_CANO_S10_PERCEPTION_MESH_REPLACEMENT_V1"
        and summary.get("trajectory_frames_checked") == 2558
        and summary.get("floor_query_rays") == 2558
        and summary.get("mesh_audit_passed") is True
        and summary.get("sanitation_passed") is True
        and summary.get("floor_support", {}).get("passed") is True
        and summary.get("training_samples_consumed") == 0
        and summary.get("inference_frames") == 0
        and summary.get("graph_updates") == 0
        and summary.get("c09_worlds_read") == 0
        and summary.get("c10_worlds_read") == 0
        and summary.get("mtare_worlds_read") == 0
        and upstream_identity.get("tracked_clean") is True
        and import_identity.get("from_fixed_checkout") is True
    )
    status = (
        "PASS_CANO_S10_PERCEPTION_MESH_REPLACEMENT_V1R"
        if passed
        else "FAIL_CANO_S10_PERCEPTION_MESH_REPLACEMENT_V1R"
    )
    write_json(
        run_dir / "metrics/runner_summary.json",
        {
            "schema_version": "cano_s10_perception_mesh_replacement_runner_v1r",
            "overall_status": status,
            "executor_exit_code": completed.returncode,
            "duration_seconds": duration,
            "floor_support_passed": summary.get("floor_support", {}).get("passed"),
            "unsupported_frames": summary.get("floor_support", {}).get("unsupported_count"),
            "frozen_e1_verified": True,
            "fixed_checkout_import_verified": import_identity.get("from_fixed_checkout"),
            "claim_boundary": "Replacement perception asset qualification only.",
        },
    )
    write_json(
        run_dir / "RUN_STATE.json",
        {
            "schema_version": "v3_run_state_v1",
            "run_id": RUN_ID,
            "state": "COMPLETED" if passed else "FAILED",
            "overall_status": status,
        },
    )
    print(json.dumps({"overall_status": status, "sealed_files": _seal(run_dir)}))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
