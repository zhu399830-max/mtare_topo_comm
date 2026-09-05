#!/usr/bin/env python3
"""Run and seal one immutable frozen-threshold C09 association qualification."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from _bootstrap import PROJECT_ROOT
from mtare_topo.evaluation.gse_evidence_integrity import verify_complete_run_seal
from mtare_topo.governance import load_json, write_json
from run_gse_distance_aware_ensemble_calibration_v1 import _peak_rss, _seal, _sha256


RUN_ID = "gate3_20260827_gse_factorized_association_c09_qualification_v1_seed0"
PASS_STATUS = "PASS_GSE_FACTORIZED_ASSOCIATION_C09_QUALIFICATION_V1"
FAIL_STATUS = "FAIL_GSE_FACTORIZED_ASSOCIATION_C09_QUALIFICATION_V1"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_GSE_FACTORIZED_ASSOCIATION_C09_QUALIFICATION_V1"
PYTHON = Path("/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python")
EXECUTOR = PROJECT_ROOT / "tools/v3/execute_gse_factorized_association_c09_qualification_v1.py"
DATASET = PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
TEACHER = PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_teacher_manifest_v1_seed0"
TRAINING = PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_graph_three_seed_training_v1r_seed0"
RISK = PROJECT_ROOT / "results/gate3_semantics/gate3_20260825_gse_risk_calibrated_perception_validation_v2_seed0"
CAPACITY = PROJECT_ROOT / "results/gate3_semantics/gate3_20260827_gse_factorized_association_capacity_v1r_seed0"


def _sources() -> dict[str, object]:
    return {
        "dataset": verify_complete_run_seal(PROJECT_ROOT, DATASET, "PASS_GSE_DEDUPLICATED_DATASET_EXPORT_V1"),
        "teacher": verify_complete_run_seal(PROJECT_ROOT, TEACHER, "PASS_GSE_TEACHER_MANIFEST_V1"),
        "training": verify_complete_run_seal(PROJECT_ROOT, TRAINING, "PASS_GSE_GRAPH_THREE_SEED_TRAINING_V1R"),
        "risk_calibrated_perception": verify_complete_run_seal(PROJECT_ROOT, RISK, "PASS_GSE_RISK_CALIBRATED_PERCEPTION_VALIDATION_V2"),
        "unified_capacity": verify_complete_run_seal(PROJECT_ROOT, CAPACITY, "PASS_GSE_FACTORIZED_ASSOCIATION_CAPACITY_V1R"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve(); spec = load_json(args.spec.resolve())
    if run_dir.name != RUN_ID or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("C09 factorized qualification may execute only once")
    started = time.monotonic(); overall = FAIL_STATUS; error = None
    returncode = peak_rss = None; result: dict[str, object] = {}
    before: dict[str, str] = {}; after: dict[str, str] = {}
    try:
        if spec.get("gate") != 3 or spec.get("operation") != "audit" or spec.get("seed") != 0:
            raise RuntimeError("C09 factorized qualification scope drift")
        card = load_json(PROJECT_ROOT / spec["data_card"])
        if card.get("status") != CARD_STATUS or card.get("approval", {}).get("status") != "APPROVED":
            raise RuntimeError("C09 factorized qualification Data Card drift")
        for name, record in spec["frozen_tools"].items():
            if _sha256(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"frozen C09 qualification tool drift: {name}")
        for relative, expected in spec["frozen_inputs"].items():
            actual = _sha256(PROJECT_ROOT / relative)
            if actual != expected:
                raise RuntimeError(f"frozen C09 qualification input drift: {relative}")
            before[relative] = actual
        sources_before = _sources()
        environment = json.loads(subprocess.check_output([
            str(PYTHON), "-c", "import json,numpy,torch,zarr,matplotlib,sys;print(json.dumps({'python':sys.version.split()[0],'numpy':numpy.__version__,'torch':torch.__version__,'zarr':zarr.__version__,'matplotlib':matplotlib.__version__},sort_keys=True))",
        ], text=True))
        expected = {"python": "3.13.5", "numpy": "2.1.3", "torch": "2.9.0+cu129", "zarr": "2.18.7", "matplotlib": "3.10.0"}
        if environment != expected:
            raise RuntimeError(f"C09 factorized qualification environment drift: {environment}")
        write_json(run_dir / "config/environment.json", {"executable": str(PYTHON), "versions": environment, "association_inference_device": "cpu", "selection_effect": "NONE"})
        write_json(run_dir / "config/source_integrity_before.json", sources_before)
        write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING", "note": "Frozen unified-slope models and thresholds; C09 only; C10/M-TARE forbidden."})
        command = [
            "/usr/bin/time", "-v", str(PYTHON), str(EXECUTOR), "--run-dir", str(run_dir),
            "--teacher", str(TEACHER / "artifacts/teacher_observations.jsonl"),
            "--dataset-run", str(DATASET), "--training-run", str(TRAINING),
            "--risk-run", str(RISK), "--capacity-run", str(CAPACITY),
        ]
        (run_dir / "config/command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
        env = os.environ.copy(); env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3")
        env["OMP_NUM_THREADS"] = env["MKL_NUM_THREADS"] = "2"
        log = run_dir / "logs/00_c09_factorized_association_qualification.log"
        with log.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env, text=True, stdout=stream, stderr=subprocess.STDOUT, timeout=1800, check=False)
        returncode = int(completed.returncode); peak_rss = _peak_rss(log)
        metric = run_dir / "metrics/factorized_association_c09_qualification.json"
        result = load_json(metric) if metric.is_file() else {}
        sources_after = _sources()
        for relative, expected_sha in before.items():
            actual = _sha256(PROJECT_ROOT / relative)
            if actual != expected_sha:
                raise RuntimeError(f"source changed during C09 qualification: {relative}")
            after[relative] = actual
        required = [
            run_dir / "artifacts/c09_balanced_alias_manifest.jsonl",
            run_dir / "artifacts/c09_balanced_alias_pairs.npz",
            run_dir / "artifacts/c09_runtime_candidate_pairs.npz",
            run_dir / "previews/gse_factorized_association_c09_qualification.png",
            run_dir / "previews/gse_factorized_association_c09_qualification.pdf",
            run_dir / "previews/gse_factorized_association_c09_qualification.svg",
            run_dir / "previews/gse_factorized_association_c09_qualification_source.json",
        ]
        for seed in range(3):
            required.extend([run_dir / f"artifacts/seed{seed}_c09_unified_observation_features.npy", run_dir / f"artifacts/seed{seed}_c09_qualification_scores.npz"])
        output_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
        if (
            sources_before != sources_after or returncode not in (0, 2)
            or result.get("overall_status") not in (PASS_STATUS, FAIL_STATUS)
            or (returncode == 0) != bool(result.get("scientific_pass"))
            or result.get("validation_worlds") != 10 or result.get("validation_sequences") != 24462
            or result.get("balanced_pairs") != 260 or result.get("runtime_pairs") != 4201
            or result.get("optimizer_steps") != 0 or result.get("model_updates") != 0
            or result.get("threshold_selection_steps") != 0 or result.get("c10_worlds_read") != 0
            or result.get("strict_test_worlds_read") != 0 or result.get("mtare_worlds_read") != 0
            or peak_rss is None or peak_rss > 4 * 1024**2 or output_bytes > 512 * 1024**2
            or not all(path.is_file() for path in required)
        ):
            raise RuntimeError("C09 factorized qualification evidence contract drift")
        overall = str(result["overall_status"])
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run_dir / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
    write_json(run_dir / "metrics/summary.json", {
        "schema_version": "gse_factorized_association_c09_qualification_outer_v1",
        "overall_status": overall, "scientific_pass": overall == PASS_STATUS,
        "error": error, "duration_seconds": time.monotonic() - started,
        "returncode": returncode, "peak_host_rss_kib": peak_rss,
        "source_unchanged": bool(before and before == after), "qualification": result,
        "association_inference_pairs": 3 * (260 + 4201),
        "checkpoint_selection_steps": 0, "threshold_selection_steps": 0,
        "optimizer_steps": 0, "model_updates": 0, "c10_worlds_read": 0,
        "strict_test_worlds_read": 0, "mtare_worlds_read": 0,
    })
    write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "COMPLETED" if overall == PASS_STATUS else "FAILED", "overall_status": overall, "error": error})
    entries = _seal(run_dir)
    print(json.dumps({"overall_status": overall, "error": error, "seal_entries": entries}, indent=2))
    return 0 if overall == PASS_STATUS else 2


if __name__ == "__main__":
    raise SystemExit(main())
