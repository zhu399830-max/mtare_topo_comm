#!/usr/bin/env python3
"""Execute and seal one immutable geometry-conditioned event-risk capacity proof."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from _bootstrap import PROJECT_ROOT
from mtare_topo.evaluation.gse_evidence_integrity import (
    verify_complete_run_seal,
    verify_failed_component_run_seal,
)
from mtare_topo.governance import load_json, write_json
from run_gse_distance_aware_ensemble_calibration_v1 import _peak_rss, _seal, _sha256


RUN_ID = "gate3_20260827_gse_geometry_conditioned_identity_risk_capacity_v1_seed0"
PASS_STATUS = "PASS_GSE_GEOMETRY_CONDITIONED_IDENTITY_RISK_CAPACITY_V1"
FAIL_STATUS = "FAIL_GSE_GEOMETRY_CONDITIONED_IDENTITY_RISK_CAPACITY_V1"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_GSE_GEOMETRY_CONDITIONED_IDENTITY_RISK_CAPACITY_V1"
PYTHON = Path(
    "/home/zeng-workstation/.local/share/mtare_topo_comm/"
    "envs/phase3_torch290_cu129_zarr2187_v1/bin/python"
)
EXECUTOR = PROJECT_ROOT / "tools/v3/execute_gse_geometry_conditioned_identity_risk_capacity_v1.py"
TEACHER = PROJECT_ROOT / (
    "results/gate2_representation/gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0"
)
VERIFIER = PROJECT_ROOT / (
    "results/gate3_semantics/gate3_20260826_gse_exit_token_association_corrective_v2_seed0"
)
AUDIT = PROJECT_ROOT / (
    "results/gate3_semantics/gate3_20260827_gse_causal_geometry_risk_conflict_audit_v1_seed0"
)
TIME_LIMIT_SECONDS = 7200
DISK_LIMIT_BYTES = 256 * 1024 * 1024


def _sources() -> dict:
    return {
        "corrected_teacher": verify_complete_run_seal(
            PROJECT_ROOT, TEACHER, "PASS_GSE_CORRECTED_CAUSAL_TEACHER_MANIFEST_V1R"
        ),
        "frozen_features": verify_failed_component_run_seal(
            PROJECT_ROOT, VERIFIER, "FAIL_GSE_EXIT_TOKEN_ASSOCIATION_CORRECTIVE_V2"
        ),
        "mechanism_audit": verify_complete_run_seal(
            PROJECT_ROOT, AUDIT, "PASS_GSE_CAUSAL_GEOMETRY_RISK_CONFLICT_AUDIT_V1"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    spec = load_json(args.spec.resolve())
    if run_dir.name != RUN_ID:
        raise RuntimeError("identity-risk capacity run identity mismatch")
    if load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("identity-risk capacity proof may execute only once")

    started = time.monotonic()
    overall = FAIL_STATUS
    error = None
    returncode = None
    peak_rss = None
    result: dict = {}
    before: dict = {}
    after: dict = {}
    try:
        if spec.get("gate") != 3 or spec.get("operation") != "training" or spec.get("seed") != 0:
            raise RuntimeError("identity-risk capacity scope mismatch")
        if spec.get("user_authorization", {}).get("status") != "APPROVED":
            raise RuntimeError("standing user authorization is not bound to the capacity proof")
        card = load_json(PROJECT_ROOT / spec["data_card"])
        if card.get("status") != CARD_STATUS or card.get("approval", {}).get("status") != "APPROVED":
            raise RuntimeError("identity-risk capacity Data Card mismatch")
        for name, record in spec["frozen_tools"].items():
            if _sha256(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"frozen identity-risk tool drift: {name}")
        for relative, expected in spec["frozen_inputs"].items():
            if _sha256(PROJECT_ROOT / relative) != expected:
                raise RuntimeError(f"frozen identity-risk input drift: {relative}")
        before = _sources()
        audit_summary = load_json(AUDIT / "metrics/risk_conflict_summary.json")
        mechanism = audit_summary.get("mechanism_conclusion", {})
        if (
            mechanism.get("recommended_next")
            != "GEOMETRY_CONDITIONED_IDENTITY_LEVEL_RISK_CAPACITY_PROOF"
            or mechanism.get("longer_history_signal_material") is not True
            or mechanism.get("scalar_one_percent_risk_sufficient") is not False
        ):
            raise RuntimeError("sealed mechanism audit does not authorize the capacity proof")
        environment = json.loads(
            subprocess.check_output(
                [
                    str(PYTHON),
                    "-c",
                    "import json,numpy,scipy,sys;print(json.dumps({'python':sys.version.split()[0],'numpy':numpy.__version__,'scipy':scipy.__version__},sort_keys=True))",
                ],
                text=True,
            )
        )
        if environment != {"python": "3.13.5", "numpy": "2.1.3", "scipy": "1.15.3"}:
            raise RuntimeError(f"identity-risk sidecar drift: {environment}")
        write_json(
            run_dir / "config/environment.json",
            {"executable": str(PYTHON), "versions": environment, "cpu_only": True},
        )
        write_json(run_dir / "config/source_integrity_before.json", before)
        write_json(
            run_dir / "RUN_STATE.json",
            {
                "schema_version": "v3_run_state_v1",
                "run_id": RUN_ID,
                "state": "RUNNING",
                "note": "C01-C06 fit/C07-C08 selection deterministic geometry-conditioned risk capacity proof.",
            },
        )
        command = [
            "/usr/bin/time", "-v", str(PYTHON), str(EXECUTOR),
            "--run-dir", str(run_dir),
            "--teacher", str(TEACHER / "artifacts/teacher_observations.jsonl"),
            "--pair-cache", str(VERIFIER / "artifacts/pair_cache/pairs.npz"),
        ]
        for seed in (0, 1, 2):
            command.extend(
                ["--feature", str(VERIFIER / f"artifacts/models/seed{seed}/frozen_observation_features.npy")]
            )
        (run_dir / "config/command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3")
        env["OMP_NUM_THREADS"] = "2"
        env["MKL_NUM_THREADS"] = "2"
        log = run_dir / "logs/00_capacity.log"
        with log.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=env,
                text=True,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=TIME_LIMIT_SECONDS,
                check=False,
            )
        returncode = int(completed.returncode)
        peak_rss = _peak_rss(log)
        result_path = run_dir / "metrics/capacity_summary.json"
        result = load_json(result_path) if result_path.is_file() else {}
        after = _sources()
        output_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
        required = [
            run_dir / "artifacts/geometry_conditioned_risk_readout.npz",
            run_dir / "artifacts/selection_outputs.npz",
            run_dir / "artifacts/fit_contract.npz",
            run_dir / "artifacts/leave_family_out.csv",
        ]
        if (
            before != after
            or returncode not in (0, 2)
            or result.get("overall_status") not in (PASS_STATUS, FAIL_STATUS)
            or (returncode == 0) != (result.get("overall_status") == PASS_STATUS)
            or result.get("fit_worlds") != 60
            or result.get("fit_observations") != 142184
            or result.get("selection_worlds") != 20
            or result.get("selection_observations") != 45942
            or result.get("selection_change_identities") != 17
            or result.get("feature_count") != 60
            or result.get("lags") != [2, 4, 6, 8, 10, 12]
            or len(result.get("leave_family_out", [])) != 10
            or result.get("risk_readout_scored_observations") != 330310
            or result.get("upstream_model_inference_frames") != 0
            or not 0 < int(result.get("optimizer_iterations", 0)) <= 5500
            or any(result.get(name) != 0 for name in (
                "c09_worlds_read", "strict_test_worlds_read", "mtare_worlds_read"
            ))
            or peak_rss is None
            or peak_rss > 4 * 1024**2
            or output_bytes > DISK_LIMIT_BYTES
            or not all(path.is_file() for path in required)
        ):
            raise RuntimeError("identity-risk capacity violated its execution/evidence contract")
        overall = str(result["overall_status"])
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run_dir / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")

    write_json(
        run_dir / "metrics/summary.json",
        {
            "schema_version": "gse_geometry_conditioned_identity_risk_capacity_outer_v1",
            "overall_status": overall,
            "scientific_pass": overall == PASS_STATUS,
            "error": error,
            "duration_seconds": time.monotonic() - started,
            "returncode": returncode,
            "peak_host_rss_kib": peak_rss,
            "capacity": result,
            "source_before": before,
            "source_after": after,
            "source_unchanged": bool(before and before == after),
            "c09_worlds_read": 0,
            "strict_test_worlds_read": 0,
            "mtare_worlds_read": 0,
        },
    )
    write_json(
        run_dir / "RUN_STATE.json",
        {
            "schema_version": "v3_run_state_v1",
            "run_id": RUN_ID,
            "state": "COMPLETED" if overall == PASS_STATUS else "FAILED",
            "overall_status": overall,
            "error": error,
        },
    )
    entries = _seal(run_dir)
    print(json.dumps({"overall_status": overall, "error": error, "seal_entries": entries}, indent=2))
    return 0 if overall == PASS_STATUS else 2


if __name__ == "__main__":
    raise SystemExit(main())
