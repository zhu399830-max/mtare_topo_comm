#!/usr/bin/env python3
"""Execute, evaluate and seal the immutable three-seed primitive model run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import numpy as np
import scipy
import torch
import zarr

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, validate_data_card, write_json
from run_primitive_relation_nonlearning_readiness_v1 import _seal, _sha


RUN_ID = "gate3_20260830_primitive_relation_three_seed_training_v1_seed0"
PASS = "PASS_PRIMITIVE_RELATION_THREE_SEED_V1"
FAIL = "FAIL_PRIMITIVE_RELATION_THREE_SEED_V1"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_PRIMITIVE_RELATION_THREE_SEED_TRAINING_V1"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"
P1A = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1a_lossless_storage_corrective_v1_seed0"
P1B = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1b_teacher_materialization_v1r_seed0"
MODEL_READINESS = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_model_readiness_v1r_seed0"
BASELINE_READINESS = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_nonlearning_readiness_v1_seed0"
BASELINE_C07 = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_nonlearning_c07_v1_seed0"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    spec = load_json(args.spec.resolve())
    run = args.run_dir.resolve()
    started = time.monotonic()
    overall, error = FAIL, None
    before: dict[str, str] = {}
    subprocesses: list[dict] = []
    optimizer_steps = 0
    try:
        if run.name != RUN_ID or load_json(run/"RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
            raise RuntimeError("primitive-relation training executes exactly once")
        card = load_json(PROJECT_ROOT/spec["data_card"])
        report = validate_data_card(card)
        if not report.passed or card.get("status") != CARD_STATUS:
            raise RuntimeError(f"primitive training Data Card invalid: {report.errors}")
        for relative, expected in spec["frozen_inputs"].items():
            before[relative] = _sha(PROJECT_ROOT/relative)
            if before[relative] != expected:
                raise RuntimeError(f"primitive training frozen input drift: {relative}")
        for record in spec["frozen_tools"].values():
            if _sha(PROJECT_ROOT/record["path"]) != record["sha256"]:
                raise RuntimeError(f"primitive training frozen tool drift: {record['path']}")
        for source in (P1A,P1B,MODEL_READINESS,BASELINE_READINESS,BASELINE_C07):
            state = load_json(source/"RUN_STATE.json"); summary = load_json(source/"metrics/summary.json")
            if state.get("state") != "COMPLETED" or state.get("error") is not None or not summary.get("scientific_pass"):
                raise RuntimeError(f"primitive training prerequisite failed: {source.name}")
        environment = {
            "python":sys.version.split()[0],"executable":sys.executable,
            "numpy":np.__version__,"scipy":scipy.__version__,
            "torch":torch.__version__,"cuda":torch.version.cuda,"zarr":zarr.__version__,
        }
        expected_environment = {
            "python":"3.13.5","executable":PYTHON,"numpy":"2.1.3",
            "scipy":"1.15.3","torch":"2.9.0+cu129","cuda":"12.9","zarr":"2.18.7",
        }
        if environment != expected_environment:
            raise RuntimeError(f"primitive training environment drift: {environment}")
        write_json(run/"config/source_integrity_before.json", before)
        write_json(run/"config/environment.json", {"versions":environment,"platform":platform.platform(),"device":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
        write_json(run/"RUN_STATE.json", {"schema_version":"v3_run_state_v1","run_id":RUN_ID,"state":"RUNNING"})
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT/"src")+os.pathsep+str(PROJECT_ROOT/"tools/v3")
        env["PYTHONHASHSEED"] = "0"
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        tests = [
            "tests/v3/unit/test_primitive_relation_training.py",
            "tests/v3/unit/test_primitive_relation_model.py",
            "tests/v3/unit/test_primitive_relation_nonlearning.py",
            "tests/v3/unit/test_primitive_relation_metrics.py",
            "tests/v3/unit/test_primitive_relation_training_schedule.py",
            "tests/v3/unit/test_evaluate_primitive_relation_three_seed_v1.py",
        ]
        with (run/"logs/00_unit_tests.log").open("w",encoding="utf-8") as stream:
            result = subprocess.run(
                [PYTHON,"-m","pytest","-q",*tests], cwd=PROJECT_ROOT,env=env,
                stdout=stream,stderr=subprocess.STDOUT,text=True,timeout=600,check=False,
            )
        subprocesses.append({"stage":"unit_tests","returncode":result.returncode})
        if result.returncode:
            raise RuntimeError("primitive training unit tests failed")

        models = run/"artifacts/models"; models.mkdir(parents=True)
        fit_sensor = P1A/"artifacts/dataset/fit"; fit_teacher = P1B/"artifacts/teacher/fit"
        c07_sensor = P1A/"artifacts/dataset/c07"; c07_teacher = P1B/"artifacts/teacher/c07"
        for seed in range(3):
            command = [
                PYTHON,str(PROJECT_ROOT/"tools/v3/train_primitive_relation_model_v1.py"),
                "--fit-sensor-root",str(fit_sensor),"--fit-teacher-root",str(fit_teacher),
                "--c07-sensor-root",str(c07_sensor),"--c07-teacher-root",str(c07_teacher),
                "--output-dir",str(models/f"seed{seed}"),"--seed",str(seed),
                "--epochs","6","--batch-size","16","--evaluation-batch-size","128",
                "--learning-rate","0.0003","--weight-decay","0.0001",
            ]
            (run/f"config/seed{seed}_command.txt").write_text(" ".join(command)+"\n",encoding="utf-8")
            with (run/f"logs/0{seed+1}_seed{seed}_training.log").open("w",encoding="utf-8") as stream:
                result = subprocess.run(command,cwd=PROJECT_ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,text=True,timeout=43_200,check=False)
            subprocesses.append({"stage":f"seed{seed}_training","returncode":result.returncode})
            if result.returncode:
                raise RuntimeError(f"primitive training seed{seed} failed")
            summary = load_json(models/f"seed{seed}/summary.json")
            if summary.get("c08_rows_read") != 0 or summary.get("c09_c10_worlds_read") != 0:
                raise RuntimeError(f"primitive seed{seed} selection isolation drift")
            optimizer_steps += int(summary["optimizer_steps"])

        evaluation = run/"metrics/evaluation"
        command = [
            PYTHON,str(PROJECT_ROOT/"tools/v3/evaluate_primitive_relation_three_seed_v1.py"),
            "--models-root",str(models),
            "--sensor-root",str(P1A/"artifacts/dataset"),
            "--teacher-root",str(P1B/"artifacts/teacher"),
            "--baseline-c07-summary",str(BASELINE_C07/"metrics/summary.json"),
            "--output-dir",str(evaluation),
        ]
        (run/"config/evaluation_command.txt").write_text(" ".join(command)+"\n",encoding="utf-8")
        with (run/"logs/04_evaluation.log").open("w",encoding="utf-8") as stream:
            result = subprocess.run(command,cwd=PROJECT_ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,text=True,timeout=14_400,check=False)
        subprocesses.append({"stage":"evaluation","returncode":result.returncode})
        evaluation_summary = load_json(evaluation/"summary.json") if (evaluation/"summary.json").is_file() else {}
        if result.returncode not in (0,2) or evaluation_summary.get("overall_status") not in (
            "PASS_PRIMITIVE_RELATION_THREE_SEED_V1",
            "FAIL_PRIMITIVE_RELATION_THREE_SEED_C07",
            "FAIL_PRIMITIVE_RELATION_THREE_SEED_C08",
        ):
            raise RuntimeError("primitive evaluation produced no recognized scientific result")
        scientific_pass = bool(evaluation_summary.get("scientific_pass"))
        overall = PASS if scientific_pass else FAIL
        for suffix in ("png","pdf","svg"):
            source = evaluation/f"primitive_relation_comparison.{suffix}"
            if source.is_file():
                (run/"previews"/source.name).write_bytes(source.read_bytes())
        after = {relative:_sha(PROJECT_ROOT/relative) for relative in before}
        if after != before:
            raise RuntimeError("primitive training frozen sources changed")
        write_json(run/"config/source_integrity_after.json", after)
        write_json(run/"metrics/summary.json", {
            "schema_version":"primitive_relation_three_seed_training_runner_v1",
            "overall_status":overall,"scientific_pass":scientific_pass,
            "evaluation":evaluation_summary,"subprocesses":subprocesses,
            "optimizer_steps":optimizer_steps,
            "fit_rows_per_seed_epoch":426552,"c07_rows_per_seed_epoch":64644,
            "c08_rows_read":int(evaluation_summary.get("c08_rows_read",0)),
            "c09_c10_worlds_read":0,"graph_replays":0,"mtare_worlds_read":0,
            "duration_seconds":time.monotonic()-started,"error":None,
        })
    except Exception as exc:
        error=f"{type(exc).__name__}: {exc}"
        (run/"logs/failure_traceback.log").write_text(traceback.format_exc(),encoding="utf-8")
        write_json(run/"metrics/summary.json", {
            "schema_version":"primitive_relation_three_seed_training_runner_v1",
            "overall_status":FAIL,"scientific_pass":False,"error":error,
            "subprocesses":subprocesses,"optimizer_steps":optimizer_steps,
            "c09_c10_worlds_read":0,"graph_replays":0,
            "duration_seconds":time.monotonic()-started,
        })
    write_json(run/"RUN_STATE.json", {
        "schema_version":"v3_run_state_v1","run_id":RUN_ID,
        "state":"COMPLETED" if error is None else "FAILED",
        "overall_status":overall,"error":error,
        "duration_seconds":time.monotonic()-started,
    })
    entries=_seal(run)
    print(json.dumps({"overall_status":overall,"error":error,"evidence_files":entries},indent=2))
    return 0 if overall==PASS and error is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
