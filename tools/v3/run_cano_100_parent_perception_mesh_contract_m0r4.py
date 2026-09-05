#!/usr/bin/env python3
"""Run and seal the approved Cano perception-mesh M0R4 exactly once."""

from __future__ import annotations

import argparse, json, platform, shutil, subprocess, time
from datetime import datetime, timezone
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json
from run_cano_100_parent_perception_mesh_contract_m0 import EXPECTED_SCOPE, SOURCE_V2, SOURCE_V2R, _verify_seal
from run_cano_five_topology_cpu_contract_pilot import E1_PYTHON, _approved_project_file, _e1_identity, _executor_environment, _seal_manifest, _sha256, _subt_proc_gen_import_identity, _upstream_identity

RUN_ID = "gate0_20260811_cano_100_parent_perception_mesh_contract_m0r4_unified_discretization_seed0"
EXECUTOR = PROJECT_ROOT / "tools/v3/execute_cano_100_parent_perception_mesh_contract_m0r4.py"
FAILED = PROJECT_ROOT / "results/gate0_baseline/gate0_20260811_cano_100_parent_perception_mesh_contract_m0r3_point_to_surface_seed0"


def _run_executor(run_dir: Path) -> tuple[int, float]:
    argv = [str(E1_PYTHON), str(EXECUTOR), "--run-dir", str(run_dir)]
    started = time.monotonic(); output = ""; code = 124
    try:
        done = subprocess.run(argv, cwd=PROJECT_ROOT, env=_executor_environment(), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=3600, check=False)
        output, code = done.stdout, done.returncode
    except subprocess.TimeoutExpired as exc:
        captured = exc.stdout or ""; output = captured.decode() if isinstance(captured, bytes) else captured
        output += "\nTIMEOUT: frozen 3600 second M0R4 limit reached.\n"
    duration = time.monotonic() - started
    (run_dir / "logs/01_cano_100_parent_perception_mesh_contract_m0r4.log").write_text(
        "argv=" + json.dumps(argv, ensure_ascii=False) + "\n" + f"finished_at_utc={datetime.now(timezone.utc).isoformat()}\n" + output + f"\nduration_seconds={duration:.6f}\nexit_code={code}\n", encoding="utf-8")
    print(output, end="", flush=True)
    return code, duration


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--spec", required=True, type=Path); parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args(); spec = load_json(args.spec.resolve()); run_dir = args.run_dir.resolve()
    if run_dir.name != RUN_ID or not run_dir.is_dir(): raise RuntimeError(f"runner accepts only {RUN_ID}")
    if load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED": raise RuntimeError("run is not in one-time CREATED_NOT_EXECUTED state")
    if spec.get("operation") != "infrastructure" or spec.get("gate") != 0 or spec.get("seed") != 0 or spec.get("user_authorization", {}).get("status") != "APPROVED" or spec.get("data_scope") != EXPECTED_SCOPE:
        raise RuntimeError("spec approval or unchanged M0R4 scope is incompatible")
    proposal_path = _approved_project_file(spec.get("config_path"), "config_path"); card_path = _approved_project_file(spec.get("topology_card"), "topology_card")
    proposal, card = load_json(proposal_path), load_json(card_path)
    if proposal.get("status") != "APPROVED_FOR_ONE_EXECUTION" or proposal.get("approval", {}).get("status") != "APPROVED" or card.get("status") != "APPROVED_FOR_ONE_EXECUTION" or card.get("approval", {}).get("status") != "APPROVED":
        raise RuntimeError("approved M0R4 proposal/card is missing")
    tool_paths = {
        "runner": Path(__file__).resolve(), "executor": EXECUTOR,
        "m0r_executor_helpers": PROJECT_ROOT / "tools/v3/execute_cano_100_parent_perception_mesh_contract_m0r.py",
        "m0_executor": PROJECT_ROOT / "tools/v3/execute_cano_100_parent_perception_mesh_contract_m0.py",
        "m0_runner_helpers": PROJECT_ROOT / "tools/v3/run_cano_100_parent_perception_mesh_contract_m0.py",
        "mesh_contract": PROJECT_ROOT / "src/mtare_topo/data/cano_perception_mesh_contract.py",
        "topology_audit": PROJECT_ROOT / "src/mtare_topo/data/cano_topology_parent_audit.py",
        "v2_executor": PROJECT_ROOT / "tools/v3/execute_cano_100_topology_parent_candidate_audit_v2.py",
        "v1_executor": PROJECT_ROOT / "tools/v3/execute_cano_100_topology_parent_candidate_audit_v1.py",
        "cano_adapter": PROJECT_ROOT / "tools/v3/generate_cano_audited_bundle.py",
        "runner_helpers": PROJECT_ROOT / "tools/v3/run_cano_five_topology_cpu_contract_pilot.py",
        "governance": PROJECT_ROOT / "src/mtare_topo/governance.py",
        "unit_test": PROJECT_ROOT / "tests/v3/unit/test_cano_perception_mesh_contract.py",
        "external_test": PROJECT_ROOT / "tests/v3/external/test_cano_perception_mesh_reconstruction.py",
        "proposal": proposal_path, "data_card": card_path,
    }
    observed = {name: _sha256(path) for name, path in tool_paths.items()}
    if set(spec.get("frozen_tools", {})) != set(tool_paths): raise RuntimeError("frozen M0R4 tool set is not exact")
    for name, frozen in spec["frozen_tools"].items():
        if observed[name] != frozen.get("sha256"): raise RuntimeError(f"frozen tool mismatch: {name}")
    state = load_json(FAILED / "RUN_STATE.json")
    if state.get("state") != "FAILED" or state.get("overall_status") != "FAIL_CANO_100_PARENT_PERCEPTION_MESH_CONTRACT_M0R3": raise RuntimeError("M0R3 predecessor is not the sealed vertex-count FAIL")
    previous_seal, v2r_seal, v2_seal = _verify_seal(FAILED, 130), _verify_seal(SOURCE_V2R, 17), _verify_seal(SOURCE_V2, 391)
    for relative, expected in spec.get("frozen_source_files", {}).items():
        if _sha256(_approved_project_file(relative, "frozen_source_files")) != expected: raise RuntimeError(f"frozen source mismatch: {relative}")
    upstream = _upstream_identity(); environment = _e1_identity()
    if upstream != spec.get("upstream_identity"): raise RuntimeError("upstream identity mismatch")
    if environment["pip_freeze_evidence_sha256"] != spec.get("pip_freeze_evidence_sha256"): raise RuntimeError("E1 identity mismatch")
    imports = _subt_proc_gen_import_identity(_executor_environment())
    if shutil.disk_usage(PROJECT_ROOT).free < 2 * 1024**3: raise RuntimeError("less than 2 GiB free disk")
    for name, value in (("tool_hashes", observed), ("upstream_identity", upstream), ("e1_environment", environment), ("subt_proc_gen_import_identity", imports), ("m0r3_source_seal", previous_seal), ("v2r_source_seal", v2r_seal), ("v2_source_seal", v2_seal)):
        write_json(run_dir / f"config/{name}.json", value)
    shutil.copy2(card_path, run_dir / "config/topology_card.json")
    write_json(run_dir / "RUN_STATE.json", {"schema_version":"v3_run_state_v1","run_id":RUN_ID,"state":"RUNNING","note":"Approved M0R4 unified discretization; zero data/training."})
    code, duration = _run_executor(run_dir)
    summary_path = run_dir / "metrics/summary.json"; summary = load_json(summary_path) if summary_path.is_file() else None
    meshes = sorted(run_dir.glob("artifacts/meshes/S*_C01/*/mesh.obj")); primary = [p for p in meshes if p.parent.name == "primary"]; replay = [p for p in meshes if p.parent.name == "replay"]
    metrics = sorted(run_dir.glob("metrics/S*_C01.json")); maps = sorted(run_dir.glob("previews/train_complete_maps/*.png")); size = sum(p.stat().st_size for p in run_dir.rglob("*") if p.is_file())
    passed = bool(code == 0 and duration <= 3600 and summary and summary.get("overall_status") == "PASS_CANO_100_PARENT_PERCEPTION_MESH_CONTRACT_M0R4" and summary.get("scope") == EXPECTED_SCOPE and summary.get("batch_audit", {}).get("passed") is True and len(primary) == len(replay) == len(metrics) == len(maps) == 10 and (run_dir / "artifacts/mesh_manifest.json").is_file() and (run_dir / "previews/provenance.json").is_file() and size <= int(1.5 * 1024**3))
    overall = "PASS_CANO_100_PARENT_PERCEPTION_MESH_CONTRACT_M0R4" if passed else "FAIL_CANO_100_PARENT_PERCEPTION_MESH_CONTRACT_M0R4"
    runner_summary = {"schema_version":"cano_100_parent_perception_mesh_contract_runner_summary_m0r4","run_id":RUN_ID,"overall_status":overall,"executor_exit_code":code,"duration_seconds":duration,"counts":{"primary_meshes":len(primary),"replay_meshes":len(replay),"parent_metrics":len(metrics),"train_complete_previews":len(maps)},"source_seals":{"m0r3":previous_seal,"v2r":v2r_seal,"v2":v2_seal},"result_bytes_before_seal":size,"disk_limit_bytes":int(1.5*1024**3),"free_disk_bytes_after":shutil.disk_usage(PROJECT_ROOT).free,"host":{"platform":platform.platform()},"claim_boundary":"Train-only native perception-surface M0R4; zero LiDAR, labels, formal data, training, models, trajectories and M-TARE changes."}
    write_json(run_dir / "metrics/runner_summary.json", runner_summary); write_json(run_dir / "RUN_STATE.json", {"schema_version":"v3_run_state_v1","run_id":RUN_ID,"state":"COMPLETED" if passed else "FAILED","overall_status":overall,"note":runner_summary["claim_boundary"]})
    sealed = _seal_manifest(run_dir); print(json.dumps({"overall_status":overall,"sealed_files":sealed}, indent=2)); return 0 if passed else 2


if __name__ == "__main__": raise SystemExit(main())
