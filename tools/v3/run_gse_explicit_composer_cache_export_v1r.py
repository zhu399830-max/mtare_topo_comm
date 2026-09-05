#!/usr/bin/env python3
"""V1R corrective runner for one-binary16-epsilon probability parity."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import run_gse_explicit_composer_cache_export_v1 as v1
from _bootstrap import PROJECT_ROOT
from mtare_topo.data.gse_explicit_composer_cache_parity import BINARY16_ABSOLUTE_EPSILON
from mtare_topo.governance import load_json, validate_data_card, write_json


PASS = "PASS_GSE_EXPLICIT_COMPOSER_CACHE_EXPORT_V1R"
FAIL = "FAIL_GSE_EXPLICIT_COMPOSER_CACHE_EXPORT_V1R"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_GSE_EXPLICIT_COMPOSER_CACHE_EXPORT_V1R"
FAILED_V1 = PROJECT_ROOT / "results/gate3_semantics/gate3_20260829_gse_explicit_composer_cache_export_v1_seed0"


def _verify_failed_v1() -> dict[str, object]:
    state = load_json(FAILED_V1 / "RUN_STATE.json")
    summary = load_json(FAILED_V1 / "metrics/summary.json")
    if (
        state.get("state") != "FAILED" or state.get("overall_status") != v1.FAIL
        or summary.get("overall_status") != v1.FAIL
        or "token_count_probability" not in (FAILED_V1 / "logs/01_seed0_export.log").read_text(encoding="utf-8")
    ):
        raise RuntimeError("failed V1 probability-parity evidence drift")
    seal = FAILED_V1 / "artifacts/evidence_sha256.txt"; covered = set()
    for line in seal.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1); path = (PROJECT_ROOT / relative).resolve()
        path.relative_to(FAILED_V1.resolve())
        if path in covered or not path.is_file() or v1.sha256(path) != expected:
            raise RuntimeError(f"failed V1 seal drift: {relative}")
        covered.add(path)
    actual = {path.resolve() for path in FAILED_V1.rglob("*") if path.is_file() and path.resolve() != seal.resolve()}
    if covered != actual:
        raise RuntimeError("failed V1 seal coverage drift")
    return {"run": str(FAILED_V1.relative_to(PROJECT_ROOT)), "entries": len(covered), "seal_sha256": v1.sha256(seal)}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--spec", required=True, type=Path); parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args(); spec = load_json(args.spec.resolve()); run = args.run_dir.resolve()
    run_id = f"gate{spec['gate']}_{spec['date']}_{spec['slug']}_seed{spec['seed']}"
    if run.name != run_id or load_json(run / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("explicit Composer cache V1R executes exactly once")
    started = time.monotonic(); overall = FAIL; error = None; before = {}; after = {}
    subprocesses = []; manifests = []; peak_rss_kib = 0; source_evidence = {}
    try:
        card = load_json(PROJECT_ROOT / spec["data_card"]); report = validate_data_card(card)
        if not report.passed or card.get("status") != CARD_STATUS:
            raise RuntimeError(f"V1R Data Card invalid: {report.errors}")
        for record in spec["frozen_tools"].values():
            if v1.sha256(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"tool drift: {record['path']}")
        for relative, expected in spec["frozen_inputs"].items():
            before[relative] = v1.sha256(PROJECT_ROOT / relative)
            if before[relative] != expected:
                raise RuntimeError(f"input drift: {relative}")
        source_evidence["dataset"] = v1.verify_run_seal(v1.DATASET, "PASS_GSE_DEDUPLICATED_DATASET_EXPORT_V1")
        source_evidence["baseline"] = v1.verify_run_seal(v1.BASELINE, "FAIL_GSE_SPARSE_CIRCULAR_RELATION_TRANSPORT_THREE_SEED_TRAINING_V2")
        source_evidence["readiness"] = v1.verify_run_seal(v1.READINESS, "PASS_GSE_TYPED_COMPOSER_READINESS_V1")
        source_evidence["failed_v1"] = _verify_failed_v1()
        environment = json.loads(subprocess.check_output([
            str(v1.PYTHON), "-c",
            "import json,numpy,torch,zarr,sys;print(json.dumps({'python':sys.version.split()[0],'numpy':numpy.__version__,'torch':torch.__version__,'cuda':torch.version.cuda,'zarr':zarr.__version__,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},sort_keys=True))",
        ], text=True))
        expected_environment = {"python":"3.13.5","numpy":"2.1.3","torch":"2.9.0+cu129","cuda":"12.9","zarr":"2.18.7","gpu":"NVIDIA GeForce RTX 5090 D"}
        if environment != expected_environment: raise RuntimeError(f"environment drift: {environment}")
        write_json(run / "config/environment.json", {"executable":str(v1.PYTHON),"versions":environment,"deterministic":True})
        write_json(run / "config/source_evidence.json", source_evidence); write_json(run / "config/source_integrity_before.json", before)
        write_json(run / "RUN_STATE.json", {"schema_version":"v3_run_state_v1","run_id":run_id,"state":"RUNNING"})
        env = os.environ.copy(); env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3"); env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        unit_log = run / "logs/00_unit_tests.log"
        with unit_log.open("w", encoding="utf-8") as stream:
            unit = subprocess.run([str(v1.PYTHON),"-m","pytest","-q","tests/v3/unit/test_gse_explicit_composer_cache.py","tests/v3/unit/test_gse_explicit_composer_cache_parity.py","tests/v3/unit/test_gse_typed_composers.py"],cwd=PROJECT_ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,text=True,timeout=600,check=False)
        subprocesses.append({"stage":"unit_tests","returncode":int(unit.returncode)})
        if unit.returncode != 0: raise RuntimeError("V1R unit tests failed")
        cache_root = run / "artifacts/cache"; cache_root.mkdir()
        for seed in range(3):
            command = [str(v1.PYTHON),str(PROJECT_ROOT/"tools/v3/export_gse_explicit_composer_cache_v1r.py"),"--dataset-root",str(v1.DATASET/"artifacts/dataset/train"),"--sequence-manifest",str(v1.DATASET/"artifacts/sequence_manifest.jsonl"),"--checkpoint",str(v1.BASELINE/f"artifacts/models/seed{seed}/best.pt"),"--development-prediction-root",str(v1.BASELINE/f"artifacts/models/seed{seed}/development_predictions"),"--output-dir",str(cache_root/f"seed{seed}"),"--seed",str(seed)]
            (run/f"config/seed{seed}_command.txt").write_text(" ".join(command)+"\n",encoding="utf-8")
            log=run/f"logs/0{seed+1}_seed{seed}_export.log"
            with log.open("w",encoding="utf-8") as stream:
                completed=subprocess.run(["/usr/bin/time","-v",*command],cwd=PROJECT_ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,text=True,timeout=3600,check=False)
            subprocesses.append({"stage":f"seed{seed}_export","returncode":int(completed.returncode)})
            if completed.returncode != 0: raise RuntimeError(f"V1R seed{seed} export failed")
            peak_rss_kib=max(peak_rss_kib,v1._peak_rss(log) or 0)
            manifest=load_json(cache_root/f"seed{seed}/manifest.json")
            parity=[record["sealed_development_parity"] for record in manifest["records"] if record["sealed_development_parity"] is not None]
            if (manifest.get("seed")!=seed or manifest.get("worlds")!=80 or manifest.get("split_rows")!={"fit":142184,"c07":21548,"c08":24394} or len(parity)!=20 or float(manifest.get("maximum_development_parity_error",99))>BINARY16_ABSOLUTE_EPSILON or not all(row.get("all_primary_fields_exact") and row.get("all_derived_probabilities_within_binary16_epsilon") and row.get("all_discrete_decisions_equal") for row in parity) or manifest.get("optimizer_steps")!=0 or int(manifest.get("peak_gpu_memory_bytes",v1.GPU_LIMIT_BYTES+1))>v1.GPU_LIMIT_BYTES):
                raise RuntimeError(f"V1R seed{seed} manifest contract failed")
            manifests.append(manifest)
        disk_bytes=sum(path.stat().st_size for path in run.rglob("*") if path.is_file())
        if disk_bytes>v1.DISK_LIMIT_BYTES: raise RuntimeError(f"V1R disk limit exceeded: {disk_bytes}")
        v1._plot(run/"metrics",manifests)
        for relative in spec["frozen_inputs"]: after[relative]=v1.sha256(PROJECT_ROOT/relative)
        if after!=before: raise RuntimeError("V1R frozen source changed")
        write_json(run/"config/source_integrity_after.json",after); overall=PASS
        write_json(run/"metrics/summary.json",{"schema_version":"gse_explicit_composer_cache_export_runner_v1r","overall_status":overall,"scientific_pass":True,"seeds":3,"worlds_per_seed":80,"observations_per_seed":188126,"model_forward_observations":564378,"development_parity_worlds":60,"primary_field_maximum_error":0.0,"derived_probability_maximum_error":max(float(m["maximum_development_parity_error"]) for m in manifests),"binary16_absolute_epsilon":BINARY16_ABSOLUTE_EPSILON,"all_discrete_decisions_equal":True,"fields":manifests[0]["fields"],"forbidden_fields":manifests[0]["forbidden_fields"],"cache_files":240,"disk_bytes_before_seal":disk_bytes,"peak_host_rss_kib":peak_rss_kib,"peak_gpu_memory_bytes":max(int(m["peak_gpu_memory_bytes"]) for m in manifests),"subprocesses":subprocesses,"source_evidence":source_evidence,"optimizer_steps":0,"checkpoints_created":0,"c09_worlds_read":0,"c10_worlds_read":0,"strict_test_worlds_read":0,"mtare_worlds_read":0,"graph_replays":0,"planner_calls":0,"duration_seconds":time.monotonic()-started})
    except Exception as exception:
        error=f"{type(exception).__name__}: {exception}"; overall=FAIL
        (run/"logs/runner_error.log").write_text(traceback.format_exc(),encoding="utf-8")
        write_json(run/"metrics/summary.json",{"schema_version":"gse_explicit_composer_cache_export_runner_v1r","overall_status":overall,"scientific_pass":False,"error":error,"subprocesses":subprocesses,"optimizer_steps":0,"checkpoints_created":0,"c09_worlds_read":0,"c10_worlds_read":0,"strict_test_worlds_read":0,"mtare_worlds_read":0,"graph_replays":0,"duration_seconds":time.monotonic()-started})
    write_json(run/"RUN_STATE.json",{"schema_version":"v3_run_state_v1","run_id":run_id,"state":"COMPLETED" if error is None else "FAILED","overall_status":overall,"error":error,"duration_seconds":time.monotonic()-started})
    evidence=v1.seal(run); print(json.dumps({"run_id":run_id,"overall_status":overall,"error":error,"evidence_files":evidence},indent=2,sort_keys=True))
    return 0 if error is None and overall==PASS else 2


if __name__ == "__main__": raise SystemExit(main())
