#!/usr/bin/env python3
"""Run and seal one immutable two-process C09 causal-node qualification."""

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


RUN_ID = "gate3_20260828_gse_factorized_causal_node_c09_v1_seed0"
PASS_STATUS = "PASS_GSE_FACTORIZED_CAUSAL_NODE_C09_V1"
FAIL_STATUS = "FAIL_GSE_FACTORIZED_CAUSAL_NODE_C09_V1"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_GSE_FACTORIZED_CAUSAL_NODE_C09_V1"
PYTHON = Path("/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python")
INFER = PROJECT_ROOT / "tools/v3/infer_gse_factorized_causal_nodes_c09_v1.py"
EVALUATE = PROJECT_ROOT / "tools/v3/evaluate_gse_factorized_causal_nodes_c09_v1.py"
DATASET = PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
TEACHER = PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_teacher_manifest_v1_seed0"
TRAINING = PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_graph_three_seed_training_v1r_seed0"
EPISODE = PROJECT_ROOT / "results/gate3_semantics/gate3_20260827_gse_causal_episode_training_v1r_seed0"


def _sources() -> dict[str, object]:
    return {
        "dataset": verify_complete_run_seal(
            PROJECT_ROOT, DATASET, "PASS_GSE_DEDUPLICATED_DATASET_EXPORT_V1"
        ),
        "teacher": verify_complete_run_seal(
            PROJECT_ROOT, TEACHER, "PASS_GSE_TEACHER_MANIFEST_V1"
        ),
        "base_training": verify_complete_run_seal(
            PROJECT_ROOT, TRAINING, "PASS_GSE_GRAPH_THREE_SEED_TRAINING_V1R"
        ),
        "causal_episode_component": verify_failed_component_run_seal(
            PROJECT_ROOT, EPISODE, "FAIL_GSE_CAUSAL_EPISODE_TRAINING_V1R"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    spec = load_json(args.spec.resolve())
    if run_dir.name != RUN_ID or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("C09 causal-node qualification may execute only once")
    started = time.monotonic()
    overall = FAIL_STATUS
    error = None
    inference_rc = evaluation_rc = None
    inference_rss = evaluation_rss = None
    result: dict[str, object] = {}
    before: dict[str, str] = {}
    after: dict[str, str] = {}
    try:
        if spec.get("gate") != 3 or spec.get("operation") != "audit" or spec.get("seed") != 0:
            raise RuntimeError("C09 causal-node qualification scope drift")
        card = load_json(PROJECT_ROOT / spec["data_card"])
        if card.get("status") != CARD_STATUS or card.get("approval", {}).get("status") != "APPROVED":
            raise RuntimeError("C09 causal-node Data Card drift")
        for name, record in spec["frozen_tools"].items():
            if _sha256(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"frozen causal-node tool drift: {name}")
        for relative, expected in spec["frozen_inputs"].items():
            actual = _sha256(PROJECT_ROOT / relative)
            if actual != expected:
                raise RuntimeError(f"frozen causal-node input drift: {relative}")
            before[relative] = actual
        sources_before = _sources()
        environment = json.loads(subprocess.check_output([
            str(PYTHON), "-c",
            "import json,numpy,torch,zarr,matplotlib,sys;print(json.dumps({'python':sys.version.split()[0],'numpy':numpy.__version__,'torch':torch.__version__,'zarr':zarr.__version__,'matplotlib':matplotlib.__version__,'cuda':torch.version.cuda},sort_keys=True))",
        ], text=True))
        expected_environment = {
            "python": "3.13.5", "numpy": "2.1.3", "torch": "2.9.0+cu129",
            "zarr": "2.18.7", "matplotlib": "3.10.0", "cuda": "12.9",
        }
        if environment != expected_environment:
            raise RuntimeError(f"C09 causal-node environment drift: {environment}")
        write_json(run_dir / "config/environment.json", {
            "executable": str(PYTHON), "versions": environment,
            "inference_device": "cuda", "optimizer_steps": 0,
        })
        write_json(run_dir / "config/source_integrity_before.json", sources_before)
        write_json(run_dir / "RUN_STATE.json", {
            "schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING",
            "note": "Process 1 has no Teacher argument and freezes all C09 outputs before process 2 evaluates them; C10/M-TARE forbidden.",
        })
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3")
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        env["OMP_NUM_THREADS"] = env["MKL_NUM_THREADS"] = "4"
        inference_dir = run_dir / "artifacts/inference"
        infer_command = [
            "/usr/bin/time", "-v", str(PYTHON), str(INFER),
            "--dataset-run", str(DATASET), "--training-run", str(TRAINING),
            "--episode-run", str(EPISODE), "--output-dir", str(inference_dir),
            "--frame-batch-size", "128", "--episode-batch-size", "128",
        ]
        (run_dir / "config/command.txt").write_text(
            " ".join(infer_command) + "\n", encoding="utf-8"
        )
        inference_log = run_dir / "logs/00_teacher_free_c09_inference.log"
        with inference_log.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(
                infer_command, cwd=PROJECT_ROOT, env=env, text=True,
                stdout=stream, stderr=subprocess.STDOUT, timeout=3600, check=False,
            )
        inference_rc = int(completed.returncode)
        inference_rss = _peak_rss(inference_log)
        manifest = load_json(inference_dir / "inference_manifest.json") if (inference_dir / "inference_manifest.json").is_file() else {}
        if (
            inference_rc != 0 or manifest.get("worlds") != 10
            or manifest.get("unique_lidar_frames") != 32_678
            or manifest.get("causal_observations") != 24_462
            or manifest.get("directed_traversals") != 2_054
            or manifest.get("valid_past_reference_cells") != 237_678
            or manifest.get("future_reference_cells") != 0
            or manifest.get("teacher_inputs_read") != 0
            or manifest.get("teacher_identity_inputs_read") != 0
            or manifest.get("spatial_encoder_inference_frames") != 98_034
            or manifest.get("episode_detector_inference_observations") != 73_386
            or manifest.get("optimizer_steps") != 0 or manifest.get("model_updates") != 0
            or manifest.get("checkpoint_selection_steps") != 0
            or manifest.get("c10_worlds_read") != 0 or manifest.get("mtare_worlds_read") != 0
        ):
            raise RuntimeError("Teacher-free C09 inference evidence contract drift")

        evaluation_dir = run_dir / "artifacts/evaluation"
        evaluation_command = [
            "/usr/bin/time", "-v", str(PYTHON), str(EVALUATE),
            "--inference-dir", str(inference_dir),
            "--teacher", str(TEACHER / "artifacts/teacher_observations.jsonl"),
            "--selection-summary", str(EPISODE / "metrics/ensemble/summary.json"),
            "--output-dir", str(evaluation_dir),
        ]
        with (run_dir / "config/command.txt").open("a", encoding="utf-8") as stream:
            stream.write(" ".join(evaluation_command) + "\n")
        evaluation_log = run_dir / "logs/01_posthoc_teacher_evaluation.log"
        with evaluation_log.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(
                evaluation_command, cwd=PROJECT_ROOT, env=env, text=True,
                stdout=stream, stderr=subprocess.STDOUT, timeout=600, check=False,
            )
        evaluation_rc = int(completed.returncode)
        evaluation_rss = _peak_rss(evaluation_log)
        result = load_json(evaluation_dir / "metrics.json") if (evaluation_dir / "metrics.json").is_file() else {}
        sources_after = _sources()
        for relative, expected_sha in before.items():
            actual = _sha256(PROJECT_ROOT / relative)
            if actual != expected_sha:
                raise RuntimeError(f"source changed during causal-node qualification: {relative}")
            after[relative] = actual
        required = [
            inference_dir / "inference_manifest.json",
            inference_dir / "past_only_references.npz",
            inference_dir / "ensemble_deployment_outputs.npz",
            inference_dir / "decision_triggers.jsonl",
            *(inference_dir / f"seed{seed}_outputs.npz" for seed in range(3)),
            evaluation_dir / "metrics.json",
            evaluation_dir / "gse_factorized_causal_nodes_c09.png",
            evaluation_dir / "gse_factorized_causal_nodes_c09.pdf",
            evaluation_dir / "gse_factorized_causal_nodes_c09.svg",
            evaluation_dir / "gse_factorized_causal_nodes_c09_source.json",
        ]
        output_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
        if (
            sources_before != sources_after or evaluation_rc not in (0, 2)
            or result.get("overall_status") not in (PASS_STATUS, FAIL_STATUS)
            or (evaluation_rc == 0) != bool(result.get("scientific_pass"))
            or result.get("worlds") != 10 or result.get("causal_observations") != 24_462
            or result.get("true_junction_episodes") != 426
            or result.get("true_terminal_episodes") != 114
            or result.get("true_junction_identities") != 71
            or result.get("true_terminal_identities") != 59
            or result.get("threshold_selection_steps") != 0
            or result.get("optimizer_steps") != 0 or result.get("model_updates") != 0
            or result.get("inference_teacher_inputs_read") != 0
            or result.get("c10_worlds_read") != 0 or result.get("mtare_worlds_read") != 0
            or inference_rss is None or inference_rss > 8 * 1024**2
            or evaluation_rss is None or evaluation_rss > 4 * 1024**2
            or output_bytes > 2 * 1024**3 or not all(path.is_file() for path in required)
        ):
            raise RuntimeError("C09 causal-node qualification evidence contract drift")
        write_json(run_dir / "metrics/factorized_causal_node_c09.json", result)
        overall = str(result["overall_status"])
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run_dir / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
    write_json(run_dir / "metrics/summary.json", {
        "schema_version": "gse_factorized_causal_node_c09_outer_v1",
        "overall_status": overall, "scientific_pass": overall == PASS_STATUS,
        "error": error, "duration_seconds": time.monotonic() - started,
        "inference_returncode": inference_rc, "evaluation_returncode": evaluation_rc,
        "inference_peak_host_rss_kib": inference_rss,
        "evaluation_peak_host_rss_kib": evaluation_rss,
        "source_unchanged": bool(before and before == after), "qualification": result,
        "spatial_encoder_inference_frames": 98_034,
        "episode_detector_inference_observations": 73_386,
        "checkpoint_selection_steps": 0, "optimizer_steps": 0, "model_updates": 0,
        "c10_worlds_read": 0, "strict_test_worlds_read": 0, "mtare_worlds_read": 0,
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
