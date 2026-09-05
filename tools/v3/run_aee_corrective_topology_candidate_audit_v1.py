#!/usr/bin/env python3
"""Run and seal the approved C13--C24 corrective topology audit once."""

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
from mtare_topo.data.cano_topology_parent_audit import (
    build_arithmetic_candidate_registry,
    canonical_json_hash,
)
from mtare_topo.governance import load_json, write_json
from run_cano_five_topology_cpu_contract_pilot import (
    _approved_project_file,
    _seal_manifest,
    _sha256,
)


RUN_ID = "gate2_20260821_aee_corrective_topology_candidate_audit_v1_seed20260821"
RUNNER = Path(__file__).resolve()
EXECUTOR = PROJECT_ROOT / "tools/v3/execute_aee_corrective_topology_candidate_audit_v1.py"
E1 = Path("/home/zeng-workstation/.local/share/mtare_topo_comm/envs/cano_e1_topology_v1/bin/python")
EXPECTED_SCOPE = {
    "candidate_topology_constructions": 120,
    "same_seed_topology_replays_maximum": 120,
    "selected_topology_parents": 20,
    "corrective_train_parents": 10,
    "corrective_validation_parents": 10,
    "meshes": 0,
    "lidar_observations": 0,
    "teacher_labels": 0,
    "training_samples": 0,
    "models": 0,
    "c09_reads": 0,
    "c10_reads": 0,
    "mtare_changes": 0,
}
HISTORICAL_FREEZE = PROJECT_ROOT / (
    "results/gate0_baseline/"
    "gate0_20260810_cano_dependency_compatibility_matrix_v1_seed0/"
    "config/pip_freeze_E1.txt"
)
GATE0_V2_SEAL = PROJECT_ROOT / (
    "results/gate0_baseline/"
    "gate0_20260811_cano_100_topology_parent_candidate_audit_v2_bounded_resampling_seed0/"
    "artifacts/evidence_sha256.txt"
)
AEE_PARITY_SEAL = PROJECT_ROOT / (
    "results/gate2_representation/"
    "gate2_20260821_aee_sensor_operator_parity_v1r_seed20260820/"
    "artifacts/evidence_sha256.txt"
)


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    paths = [
        str(PROJECT_ROOT / "tools/v3"),
        str(PROJECT_ROOT / "external/procedural-subt-gen/src"),
        str(PROJECT_ROOT / "src"),
    ]
    if environment.get("PYTHONPATH"):
        paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(paths)
    environment["MPLBACKEND"] = "Agg"
    return environment


def _run_executor(run_dir: Path) -> tuple[int, float]:
    argv = [str(E1), str(EXECUTOR), "--run-dir", str(run_dir)]
    started = time.monotonic()
    output = ""
    exit_code = 124
    try:
        completed = subprocess.run(
            argv,
            cwd=PROJECT_ROOT,
            env=_environment(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=7200,
            check=False,
        )
        output = completed.stdout
        exit_code = completed.returncode
    except subprocess.TimeoutExpired as error:
        captured = error.stdout or ""
        output = captured.decode() if isinstance(captured, bytes) else captured
        output += "\nTIMEOUT: frozen 7200-second limit reached.\n"
    duration = time.monotonic() - started
    (run_dir / "logs/01_candidate_audit.log").write_text(
        "argv=" + json.dumps(argv) + "\n"
        + f"finished_at_utc={datetime.now(timezone.utc).isoformat()}\n"
        + output
        + f"\nduration_seconds={duration:.6f}\nexit_code={exit_code}\n",
        encoding="utf-8",
    )
    print(output, end="", flush=True)
    return exit_code, duration


def _normalized_freeze_sha256(text: str) -> str:
    lines = sorted(line.strip() for line in text.splitlines() if line.strip())
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def _verify_environment_and_sources(spec: dict) -> dict:
    environment_record_path = _approved_project_file(
        spec.get("environment", {}).get("path"), "environment.path"
    )
    environment_record = load_json(environment_record_path)
    if environment_record.get("normalized_freeze_equal") is not True:
        raise RuntimeError("persistent Cano E1 is not qualified against the historical freeze")
    if _sha256(HISTORICAL_FREEZE) != spec["environment"]["historical_freeze_sha256"]:
        raise RuntimeError("historical Cano E1 freeze drifted")

    live_freeze = subprocess.check_output(
        [str(E1), "-m", "pip", "freeze"], text=True, env=_environment()
    )
    historical_text = HISTORICAL_FREEZE.read_text(encoding="utf-8")
    live_normalized = _normalized_freeze_sha256(live_freeze)
    historical_normalized = _normalized_freeze_sha256(historical_text)
    if live_normalized != historical_normalized:
        raise RuntimeError("persistent Cano E1 package freeze drifted")
    pip_check = subprocess.run(
        [str(E1), "-m", "pip", "check"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=_environment(),
        check=False,
    )
    if pip_check.returncode != 0 or "No broken requirements found." not in pip_check.stdout:
        raise RuntimeError(f"persistent Cano E1 pip check failed: {pip_check.stdout}")

    source_evidence = spec.get("source_evidence", {})
    if _sha256(GATE0_V2_SEAL) != source_evidence.get("gate0_v2_evidence_seal_sha256"):
        raise RuntimeError("Gate-0 V2 evidence-seal identity drifted")
    if _sha256(AEE_PARITY_SEAL) != source_evidence.get("aee_parity_v1r_evidence_seal_sha256"):
        raise RuntimeError("AEE parity evidence-seal identity drifted")
    upstream_root = PROJECT_ROOT / "external/procedural-subt-gen"
    upstream_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=upstream_root, text=True
    ).strip()
    upstream_status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=upstream_root, text=True
    )
    if upstream_commit != environment_record.get("upstream_commit") or upstream_status.strip():
        raise RuntimeError("pinned Cano upstream commit or clean status drifted")
    return {
        "live_normalized_freeze_sha256": live_normalized,
        "historical_normalized_freeze_sha256": historical_normalized,
        "pip_check": pip_check.stdout.strip(),
        "upstream_commit": upstream_commit,
        "upstream_clean": True,
        "gate0_v2_seal_sha256": _sha256(GATE0_V2_SEAL),
        "aee_parity_seal_sha256": _sha256(AEE_PARITY_SEAL),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    spec_path = args.spec.resolve()
    spec = load_json(spec_path)
    run_dir = args.run_dir.resolve()
    if run_dir.name != RUN_ID or not run_dir.is_dir():
        raise RuntimeError(f"runner accepts only {RUN_ID}")
    if load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("run is not in one-time CREATED_NOT_EXECUTED state")
    authorization = spec.get("user_authorization", {})
    if (
        spec.get("gate") != 2
        or spec.get("operation") != "audit"
        or authorization.get("status") != "APPROVED"
    ):
        raise RuntimeError("spec is not an approved Gate-2 audit")
    if not E1.is_file():
        raise RuntimeError("frozen persistent Cano E1 is absent")

    proposal_path = _approved_project_file(spec.get("config_path"), "config_path")
    card_path = _approved_project_file(spec.get("data_card"), "data_card")
    proposal = load_json(proposal_path)
    card = load_json(card_path)
    if (
        proposal.get("status") != "APPROVED_FOR_ONE_EXECUTION"
        or proposal.get("approval", {}).get("status") != "APPROVED"
        or card.get("status") != "APPROVED_FOR_ONE_EXECUTION"
        or card.get("approval", {}).get("status") != "APPROVED"
    ):
        raise RuntimeError("proposal or Data Card lacks exact one-shot approval")

    scope = proposal["frozen_candidate_scope"]
    registry = build_arithmetic_candidate_registry(
        scope["strata"],
        first_candidate_index=int(scope["first_candidate_index"]),
        candidates_per_stratum=int(scope["candidates_per_stratum"]),
        topology_seed_base=int(scope["topology_seed_base"]),
        geometry_seed_base=int(scope["reserved_geometry_seed_base"]),
    )
    registry_hash = canonical_json_hash({"candidates": registry})
    if registry_hash != spec.get("candidate_registry_sha256"):
        raise RuntimeError("frozen candidate registry hash mismatch")

    tools = {
        "runner": RUNNER,
        "executor": EXECUTOR,
        "audit_helpers": PROJECT_ROOT / "src/mtare_topo/data/cano_topology_parent_audit.py",
        "v2_generator": PROJECT_ROOT / "tools/v3/execute_cano_100_topology_parent_candidate_audit_v2.py",
        "v1_generator_helpers": PROJECT_ROOT / "tools/v3/execute_cano_100_topology_parent_candidate_audit_v1.py",
        "cano_adapter": PROJECT_ROOT / "tools/v3/generate_cano_audited_bundle.py",
        "proposal": proposal_path,
        "data_card": card_path,
        "environment": PROJECT_ROOT / "configs/v3/gate2/environments/aee_corrective_cano_e1_topology_v1.json",
    }
    observed = {name: _sha256(path) for name, path in tools.items()}
    for name, frozen in spec.get("frozen_tools", {}).items():
        if observed.get(name) != frozen.get("sha256"):
            raise RuntimeError(f"frozen tool hash mismatch: {name}")

    version_probe = subprocess.check_output(
        [
            str(E1),
            "-c",
            "import json,sys,numpy,scipy,open3d,matplotlib;"
            "print(json.dumps({'python':sys.version.split()[0],'numpy':numpy.__version__,"
            "'scipy':scipy.__version__,'open3d':open3d.__version__,"
            "'matplotlib':matplotlib.__version__},sort_keys=True))",
        ],
        cwd=PROJECT_ROOT,
        env=_environment(),
        text=True,
    )
    versions = json.loads(version_probe)
    if versions != {
        "python": "3.12.3",
        "numpy": "1.26.4",
        "scipy": "1.12.0",
        "open3d": "0.19.0",
        "matplotlib": "3.11.1",
    }:
        raise RuntimeError(f"Cano E1 version drift: {versions}")
    environment_audit = _verify_environment_and_sources(spec)

    write_json(run_dir / "config/frozen_candidate_registry.json", {"candidates": registry})
    write_json(run_dir / "config/tool_hashes.json", observed)
    write_json(run_dir / "config/environment_probe.json", versions)
    write_json(run_dir / "config/environment_and_source_audit.json", environment_audit)
    shutil.copy2(card_path, run_dir / "config/data_card.json")
    write_json(
        run_dir / "RUN_STATE.json",
        {
            "schema_version": "v3_run_state_v1",
            "run_id": RUN_ID,
            "state": "RUNNING",
            "note": "Topology-only C13-C24 audit; zero mesh, sensor data or training.",
        },
    )
    exit_code, duration = _run_executor(run_dir)
    summary_path = run_dir / "metrics/summary.json"
    summary = load_json(summary_path) if summary_path.is_file() else {}
    candidate_metrics = list((run_dir / "metrics").glob("S*_C*.json"))
    selected_path = run_dir / "artifacts/selected_parent_manifest.json"
    selected = load_json(selected_path).get("parents", []) if selected_path.is_file() else []
    result_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
    passed = bool(
        exit_code == 0
        and duration <= 7200
        and summary.get("overall_status") == "PASS_AEE_CORRECTIVE_TOPOLOGY_CANDIDATE_AUDIT_V1"
        and summary.get("scope") == EXPECTED_SCOPE
        and summary.get("selection_audit", {}).get("passed") is True
        and summary.get("source_unchanged") is True
        and len(candidate_metrics) == 120
        and len(selected) == 20
        and result_bytes <= 2 * 1024**3
    )
    overall = (
        "PASS_AEE_CORRECTIVE_TOPOLOGY_CANDIDATE_AUDIT_V1"
        if passed
        else "FAIL_AEE_CORRECTIVE_TOPOLOGY_CANDIDATE_AUDIT_V1"
    )
    write_json(
        run_dir / "metrics/runner_summary.json",
        {
            "schema_version": "aee_corrective_topology_candidate_runner_v1",
            "overall_status": overall,
            "executor_exit_code": exit_code,
            "duration_seconds": duration,
            "candidate_metric_files": len(candidate_metrics),
            "selected_parent_count": len(selected),
            "result_bytes_before_seal": result_bytes,
            "claim_boundary": "Topology-only parent qualification; no sensor, teacher, model or planner conclusion.",
        },
    )
    write_json(
        run_dir / "RUN_STATE.json",
        {
            "schema_version": "v3_run_state_v1",
            "run_id": RUN_ID,
            "state": "COMPLETED" if passed else "FAILED",
            "overall_status": overall,
        },
    )
    sealed = _seal_manifest(run_dir)
    print(json.dumps({"overall_status": overall, "sealed_files": sealed}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
