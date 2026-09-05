#!/usr/bin/env python3
"""Run one immutable two-process consensus/metric association corrective."""

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


RUN_ID = "gate3_20260828_gse_factorized_consensus_metric_corrective_v1_seed0"
PASS_STATUS = "PASS_GSE_FACTORIZED_CONSENSUS_METRIC_C09_V1"
FAIL_STATUS = "FAIL_GSE_FACTORIZED_CONSENSUS_METRIC_C09_V1"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_GSE_FACTORIZED_CONSENSUS_METRIC_CORRECTIVE_V1"
PYTHON = Path("/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python")
SELECTOR = PROJECT_ROOT / "tools/v3/select_gse_factorized_consensus_metric_v1.py"
APPLICATOR = PROJECT_ROOT / "tools/v3/apply_gse_factorized_consensus_metric_c09_v1.py"
DATASET = PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
TEACHER = PROJECT_ROOT / "results/gate2_representation/gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0"
TOKEN = PROJECT_ROOT / "results/gate3_semantics/gate3_20260826_gse_exit_token_association_corrective_v2_seed0"
CAPACITY = PROJECT_ROOT / "results/gate3_semantics/gate3_20260827_gse_factorized_association_capacity_v1r_seed0"
C09_FAIL = PROJECT_ROOT / "results/gate3_semantics/gate3_20260827_gse_factorized_association_c09_qualification_v1_seed0"


def _sources() -> dict[str, object]:
    return {
        "dataset": verify_complete_run_seal(PROJECT_ROOT, DATASET, "PASS_GSE_DEDUPLICATED_DATASET_EXPORT_V1"),
        "corrected_teacher": verify_complete_run_seal(PROJECT_ROOT, TEACHER, "PASS_GSE_CORRECTED_CAUSAL_TEACHER_MANIFEST_V1R"),
        "frozen_token_features": verify_failed_component_run_seal(PROJECT_ROOT, TOKEN, "FAIL_GSE_EXIT_TOKEN_ASSOCIATION_CORRECTIVE_V2"),
        "unified_capacity": verify_complete_run_seal(PROJECT_ROOT, CAPACITY, "PASS_GSE_FACTORIZED_ASSOCIATION_CAPACITY_V1R"),
        "archived_c09_failure": verify_failed_component_run_seal(PROJECT_ROOT, C09_FAIL, "FAIL_GSE_FACTORIZED_ASSOCIATION_C09_QUALIFICATION_V1"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    spec = load_json(args.spec.resolve())
    if run_dir.name != RUN_ID or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("consensus/metric corrective may execute only once")
    started = time.monotonic()
    overall = FAIL_STATUS
    error = None
    selector_rc = applicator_rc = None
    selection_rss = application_rss = None
    result: dict[str, object] = {}
    before: dict[str, str] = {}
    after: dict[str, str] = {}
    try:
        if spec.get("gate") != 3 or spec.get("operation") != "threshold_calibration" or spec.get("seed") != 0:
            raise RuntimeError("consensus/metric corrective scope drift")
        card = load_json(PROJECT_ROOT / spec["data_card"])
        if card.get("status") != CARD_STATUS or card.get("approval", {}).get("status") != "APPROVED":
            raise RuntimeError("consensus/metric corrective Data Card drift")
        for name, record in spec["frozen_tools"].items():
            if _sha256(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"frozen corrective tool drift: {name}")
        for relative, expected in spec["frozen_inputs"].items():
            actual = _sha256(PROJECT_ROOT / relative)
            if actual != expected:
                raise RuntimeError(f"frozen corrective input drift: {relative}")
            before[relative] = actual
        sources_before = _sources()
        environment = json.loads(subprocess.check_output([
            str(PYTHON), "-c",
            "import json,numpy,torch,zarr,matplotlib,sys;print(json.dumps({'python':sys.version.split()[0],'numpy':numpy.__version__,'torch':torch.__version__,'zarr':zarr.__version__,'matplotlib':matplotlib.__version__},sort_keys=True))",
        ], text=True))
        expected_environment = {
            "python": "3.13.5", "numpy": "2.1.3", "torch": "2.9.0+cu129",
            "zarr": "2.18.7", "matplotlib": "3.10.0",
        }
        if environment != expected_environment:
            raise RuntimeError(f"corrective environment drift: {environment}")
        write_json(run_dir / "config/environment.json", {
            "executable": str(PYTHON), "versions": environment,
            "association_inference_device": "cpu", "backbone_updates": 0,
        })
        write_json(run_dir / "config/source_integrity_before.json", sources_before)
        write_json(run_dir / "RUN_STATE.json", {
            "schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING",
            "note": "Process 1 selects only on C01-C08; only after exit may process 2 read archived C09; C10/M-TARE forbidden.",
        })
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3")
        env["OMP_NUM_THREADS"] = env["MKL_NUM_THREADS"] = "2"
        selection_dir = run_dir / "artifacts/selection"
        selector = [
            "/usr/bin/time", "-v", str(PYTHON), str(SELECTOR),
            "--teacher", str(TEACHER / "artifacts/teacher_observations.jsonl"),
            "--sequence-manifest", str(DATASET / "artifacts/sequence_manifest.jsonl"),
            "--pair-cache", str(TOKEN / "artifacts/pair_cache/pairs.npz"),
            "--capacity-run", str(CAPACITY), "--output-dir", str(selection_dir),
        ]
        for seed in range(3):
            selector.extend([
                f"--observation{seed}", str(CAPACITY / f"artifacts/unified_observation/seed{seed}_unified_observation_features.npy"),
                f"--tokens{seed}", str(TOKEN / f"artifacts/models/seed{seed}/frozen_exit_token_outputs.npz"),
            ])
        (run_dir / "config/command.txt").write_text(" ".join(selector) + "\n", encoding="utf-8")
        selection_log = run_dir / "logs/00_c01_c08_selection.log"
        with selection_log.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(
                selector, cwd=PROJECT_ROOT, env=env, text=True, stdout=stream,
                stderr=subprocess.STDOUT, timeout=1800, check=False,
            )
        selector_rc = int(completed.returncode)
        selection_rss = _peak_rss(selection_log)
        calibration = load_json(selection_dir / "calibration.json") if (selection_dir / "calibration.json").is_file() else {}
        if (
            selector_rc != 0 or calibration.get("c09_worlds_read") != 0
            or calibration.get("votes_required") != 2 or calibration.get("distance_cap_m") != 4.0
            or calibration.get("runtime_decision_queries") != 8_839
            or calibration.get("runtime_pairs") != 9_380
        ):
            raise RuntimeError("C01-C08-only selection process evidence contract drift")

        application_dir = run_dir / "artifacts/c09_application"
        applicator = [
            "/usr/bin/time", "-v", str(PYTHON), str(APPLICATOR),
            "--calibration", str(selection_dir / "calibration.json"),
            "--c09-failed-run", str(C09_FAIL), "--output-dir", str(application_dir),
        ]
        with (run_dir / "config/command.txt").open("a", encoding="utf-8") as stream:
            stream.write(" ".join(applicator) + "\n")
        application_log = run_dir / "logs/01_c09_application.log"
        with application_log.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(
                applicator, cwd=PROJECT_ROOT, env=env, text=True, stdout=stream,
                stderr=subprocess.STDOUT, timeout=600, check=False,
            )
        applicator_rc = int(completed.returncode)
        application_rss = _peak_rss(application_log)
        result = load_json(application_dir / "metrics.json") if (application_dir / "metrics.json").is_file() else {}
        sources_after = _sources()
        for relative, expected_sha in before.items():
            actual = _sha256(PROJECT_ROOT / relative)
            if actual != expected_sha:
                raise RuntimeError(f"source changed during corrective: {relative}")
            after[relative] = actual
        required = [
            selection_dir / "calibration.json", selection_dir / "selection_grid.json",
            selection_dir / "c07_c08_runtime_selection.npz",
            application_dir / "metrics.json", application_dir / "c09_consensus_metric_decisions.npz",
            application_dir / "gse_factorized_consensus_metric_c09.png",
            application_dir / "gse_factorized_consensus_metric_c09.pdf",
            application_dir / "gse_factorized_consensus_metric_c09.svg",
            application_dir / "gse_factorized_consensus_metric_c09_source.json",
        ]
        output_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
        if (
            sources_before != sources_after or applicator_rc not in (0, 2)
            or result.get("overall_status") not in (PASS_STATUS, FAIL_STATUS)
            or (applicator_rc == 0) != bool(result.get("scientific_pass"))
            or result.get("c09_worlds") != 10 or result.get("c09_sequences") != 24_462
            or result.get("c09_balanced_pairs") != 260 or result.get("c09_runtime_pairs") != 4_201
            or result.get("optimizer_steps") != 0 or result.get("model_updates") != 0
            or result.get("threshold_selection_steps") != 0 or result.get("c10_worlds_read") != 0
            or result.get("strict_test_worlds_read") != 0 or result.get("mtare_worlds_read") != 0
            or selection_rss is None or selection_rss > 4 * 1024**2
            or application_rss is None or application_rss > 4 * 1024**2
            or output_bytes > 512 * 1024**2 or not all(path.is_file() for path in required)
        ):
            raise RuntimeError("consensus/metric corrective evidence contract drift")
        write_json(run_dir / "metrics/factorized_consensus_metric_c09.json", result)
        overall = str(result["overall_status"])
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run_dir / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
    write_json(run_dir / "metrics/summary.json", {
        "schema_version": "gse_factorized_consensus_metric_corrective_outer_v1",
        "overall_status": overall, "scientific_pass": overall == PASS_STATUS,
        "error": error, "duration_seconds": time.monotonic() - started,
        "selector_returncode": selector_rc, "applicator_returncode": applicator_rc,
        "selection_peak_host_rss_kib": selection_rss,
        "application_peak_host_rss_kib": application_rss,
        "source_unchanged": bool(before and before == after), "qualification": result,
        "association_inference_pairs": 3 * 9_380,
        "backbone_inference_frames": 0, "checkpoint_selection_steps": 0,
        "optimizer_steps": 0, "model_updates": 0, "c10_worlds_read": 0,
        "strict_test_worlds_read": 0, "mtare_worlds_read": 0,
    })
    write_json(run_dir / "RUN_STATE.json", {
        "schema_version": "v3_run_state_v1", "run_id": RUN_ID,
        "state": "COMPLETED" if overall == PASS_STATUS else "FAILED",
        "overall_status": overall, "error": error,
    })
    entries = _seal(run_dir)
    print(json.dumps({"overall_status": overall, "error": error, "seal_entries": entries}, indent=2))
    return 0 if overall == PASS_STATUS else 2


if __name__ == "__main__":
    raise SystemExit(main())
