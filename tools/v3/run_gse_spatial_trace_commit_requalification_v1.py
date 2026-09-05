#!/usr/bin/env python3
"""Execute and seal the frozen spatial-center trace-commit requalification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

import numpy as np

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


RUN_ID = "gate3_20260828_gse_spatial_trace_commit_requalification_v1_seed0"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_GSE_SPATIAL_TRACE_COMMIT_REQUALIFICATION_V1"
PYTHON = Path("/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python")
PASS = "PASS_GSE_SPATIAL_TRACE_COMMIT_REQUALIFICATION_V1"
FAIL = "FAIL_GSE_SPATIAL_TRACE_COMMIT_REQUALIFICATION_V1"
DATASET = PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
GSE = PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_graph_three_seed_training_v1r_seed0"
CAUSAL_TEACHER = PROJECT_ROOT / "results/gate2_representation/gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0"
SOURCE = PROJECT_ROOT / "results/gate3_semantics/gate3_20260826_gse_exit_token_association_corrective_v2_seed0"
CAPACITY = PROJECT_ROOT / "results/gate3_semantics/gate3_20260827_gse_factorized_association_capacity_v1r_seed0"
ACTION = PROJECT_ROOT / "results/gate3_semantics/gate3_20260828_gse_action_set_node_training_v1_seed0"
CENTER = PROJECT_ROOT / "results/gate3_semantics/gate3_20260828_gse_event_center_offset_training_v1r3_seed0"
SCALAR = PROJECT_ROOT / "results/gate3_semantics/gate3_20260828_gse_event_center_pair_consistency_training_v1r3_seed0"
SPATIAL = PROJECT_ROOT / "results/gate3_semantics/gate3_20260828_gse_spatial_event_center_training_v1_seed0"
CORRECTIVE = PROJECT_ROOT / "results/gate3_semantics/gate3_20260828_gse_spatial_longitudinal_corrective_v1_seed0"
PREDECESSOR = PROJECT_ROOT / "results/gate3_semantics/gate3_20260828_gse_projected_trace_commit_capacity_v1_seed0"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify(spec: dict) -> dict[str, str]:
    result = {}
    for relative, expected in spec["frozen_inputs"].items():
        actual = _sha(PROJECT_ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"spatial trace-commit frozen input drift: {relative}")
        result[relative] = actual
    return result


def _peak_rss(path: Path) -> int | None:
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "Maximum resident set size (kbytes):" in line:
            return int(line.rsplit(":", 1)[1].strip())
    return None


def _run(
    command: list[str], log: Path, env: dict[str, str], timeout: int,
    allowed: tuple[int, ...] = (0,),
) -> tuple[int, int | None]:
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            ["/usr/bin/time", "-v", *command], cwd=PROJECT_ROOT, env=env,
            text=True, stdout=stream, stderr=subprocess.STDOUT,
            timeout=timeout, check=False,
        )
    code = int(completed.returncode)
    if code not in allowed:
        raise RuntimeError(f"spatial trace-commit subprocess failed: {command[1]} code={code}")
    return code, _peak_rss(log)


def _seal(run_dir: Path) -> int:
    seal = run_dir / "artifacts/evidence_sha256.txt"
    files = sorted(path for path in run_dir.rglob("*") if path.is_file() and path != seal)
    with seal.open("w", encoding="utf-8") as stream:
        for path in files:
            stream.write(f"{_sha(path)}  {path.relative_to(PROJECT_ROOT)}\n")
    return len(files)


def _verify_training_reproduction(seed_outputs: list[Path], ensemble: Path) -> dict:
    tolerance_m = 2.5e-4
    diagnostics = {"tolerance_m": tolerance_m, "seeds": {}}
    for seed, path in enumerate(seed_outputs):
        with np.load(path, allow_pickle=False) as current, np.load(
            CORRECTIVE / f"artifacts/models/seed{seed}/selection_outputs.npz",
            allow_pickle=False,
        ) as frozen:
            rows = frozen["observation_row"].astype(np.int64)
            local_error = float(np.max(np.abs(
                current["predicted_local_vector_m"][rows].astype(np.float64)
                - frozen["predicted_local_vector_m"].astype(np.float64)
            )))
            center_error = float(np.max(np.abs(
                current["predicted_center_xyz_m"][rows].astype(np.float64)
                - frozen["predicted_center_xyz_m"].astype(np.float64)
            )))
            diagnostics["seeds"][str(seed)] = {
                "local_vector_max_abs_m": local_error,
                "world_center_max_abs_m": center_error,
            }
            if (
                not np.array_equal(current["global_sequence_index"][rows], frozen["global_sequence_index"])
                or local_error > tolerance_m or center_error > tolerance_m
            ):
                raise RuntimeError(f"seed{seed} all-row inference does not reproduce qualification")
    with np.load(ensemble, allow_pickle=False) as current, np.load(
        CORRECTIVE / "metrics/ensemble/ensemble_selection_outputs.npz", allow_pickle=False,
    ) as frozen:
        rows = frozen["observation_row"].astype(np.int64)
        local_error = float(np.max(np.abs(
            current["predicted_local_vector_m"][rows].astype(np.float64)
            - frozen["predicted_local_vector_m"].astype(np.float64)
        )))
        center_error = float(np.max(np.abs(
            current["projected_center_xyz_m"][rows].astype(np.float64)
            - frozen["predicted_center_xyz_m"].astype(np.float64)
        )))
        diagnostics["ensemble"] = {
            "local_vector_max_abs_m": local_error,
            "world_center_max_abs_m": center_error,
        }
        if (
            not np.array_equal(current["global_sequence_index"][rows], frozen["global_sequence_index"])
            or local_error > tolerance_m or center_error > tolerance_m
        ):
            raise RuntimeError("all-row spatial ensemble does not reproduce qualification")
    return diagnostics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    spec = load_json(args.spec.resolve())
    if run_dir.name != RUN_ID or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("spatial trace-commit requalification may execute only once")
    started = time.monotonic()
    overall, error, result = FAIL, None, {}
    before = after = {}
    processes = []
    deletions = []
    predecessor_delta = {}
    try:
        if spec.get("gate") != 3 or spec.get("operation") != "audit" or spec.get("seed") != 0:
            raise RuntimeError("spatial trace-commit scope drift")
        card = load_json(PROJECT_ROOT / spec["data_card"])
        if (
            card.get("status") != CARD_STATUS
            or card.get("approval", {}).get("status") != "APPROVED"
        ):
            raise RuntimeError("spatial trace-commit Data Card drift")
        for record in spec["frozen_tools"].values():
            if _sha(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"spatial trace-commit frozen tool drift: {record['path']}")
        before = _verify(spec)
        write_json(run_dir / "config/source_integrity_before.json", before)
        environment = json.loads(subprocess.check_output([
            str(PYTHON), "-c",
            "import json,numcodecs,numpy,sys,torch,zarr;print(json.dumps({'python':sys.version.split()[0],'numpy':numpy.__version__,'torch':torch.__version__,'cuda':torch.version.cuda,'zarr':zarr.__version__,'numcodecs':numcodecs.__version__,'gpu':torch.cuda.get_device_name(0)},sort_keys=True))",
        ], text=True))
        expected = {"python": "3.13.5", "numpy": "2.1.3", "torch": "2.9.0+cu129", "cuda": "12.9", "zarr": "2.18.7", "numcodecs": "0.15.1", "gpu": "NVIDIA GeForce RTX 5090 D"}
        if environment != expected:
            raise RuntimeError(f"spatial trace-commit environment drift: {environment}")
        if shutil.disk_usage(PROJECT_ROOT).free < 8 * 1024**3:
            raise RuntimeError("less than 8 GiB free before spatial trace-commit caches")
        write_json(run_dir / "config/environment.json", {
            "executable": str(PYTHON), "versions": environment,
            "deterministic_algorithms": True, "sequential_temporary_cache": True,
        })
        write_json(run_dir / "RUN_STATE.json", {
            "schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING",
        })
        env = os.environ.copy()
        env.update({
            "PYTHONPATH": str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3"),
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
        })
        scratch = run_dir / "scratch"
        projection_dir = run_dir / "artifacts/projection"
        manifest_dir = run_dir / "artifacts/cache_manifests"
        scratch.mkdir()
        projection_dir.mkdir(parents=True)
        manifest_dir.mkdir(parents=True)
        seed_outputs = []
        for seed in (0, 1, 2):
            cache = scratch / f"seed{seed}_cache"
            build = [
                str(PYTHON), str(PROJECT_ROOT / "tools/v3/build_gse_spatial_event_center_cache_v1.py"),
                "--dataset-run", str(DATASET),
                "--teacher", str(CAUSAL_TEACHER / "artifacts/teacher_observations.jsonl"),
                "--checkpoint", str(GSE / f"artifacts/models/seed{seed}/best.pt"),
                "--output-dir", str(cache), "--seed", str(seed), "--batch-size", "128",
            ]
            code, rss = _run(build, run_dir / f"logs/{seed * 2:02d}_seed{seed}_cache.log", env, 1800)
            processes.append({"stage": "cache", "seed": seed, "returncode": code, "peak_host_rss_kib": rss})
            current_manifest = load_json(cache / "manifest.json")
            old_manifest = load_json(SPATIAL / f"artifacts/cache_manifests/seed{seed}_manifest.json")
            if (
                current_manifest.get("cache_bytes") != 2_526_043_720
                or current_manifest.get("array_sha256") != old_manifest.get("array_sha256")
                or current_manifest.get("checkpoint_sha256") != old_manifest.get("checkpoint_sha256")
                or current_manifest.get("teacher_sha256") != old_manifest.get("teacher_sha256")
                or any(current_manifest.get(key) != 0 for key in ("c09_worlds_read", "c10_worlds_read", "mtare_worlds_read"))
            ):
                raise RuntimeError(f"seed{seed} spatial trace-commit cache does not reproduce V1")
            shutil.copy2(cache / "manifest.json", manifest_dir / f"seed{seed}_manifest.json")
            output = projection_dir / f"seed{seed}_all_rows.npz"
            infer = [
                str(PYTHON), str(PROJECT_ROOT / "tools/v3/infer_gse_spatial_longitudinal_center_all_v1.py"),
                "--spatial-cache", str(cache), "--action-cache", str(ACTION / "scratch/action_set_cache"),
                "--center-teacher", str(CENTER / "artifacts/teacher/event_center_teacher.npz"),
                "--pair-cache", str(SOURCE / "artifacts/pair_cache/pairs.npz"),
                "--action-checkpoint", str(ACTION / f"artifacts/models/seed{seed}/best.pt"),
                "--scalar-checkpoint", str(SCALAR / f"artifacts/models/seed{seed}/best.pt"),
                "--spatial-checkpoint", str(SPATIAL / f"artifacts/models/seed{seed}/best.pt"),
                "--corrective-checkpoint", str(CORRECTIVE / f"artifacts/models/seed{seed}/best.pt"),
                "--seed", str(seed), "--batch-size", "128", "--output", str(output),
            ]
            code, rss = _run(infer, run_dir / f"logs/{seed * 2 + 1:02d}_seed{seed}_inference.log", env, 1800)
            processes.append({"stage": "inference", "seed": seed, "returncode": code, "peak_host_rss_kib": rss})
            seed_summary = load_json(output.with_suffix(".json"))
            if (
                seed_summary.get("rows") != 188_126
                or seed_summary.get("optimizer_steps") != 0
                or seed_summary.get("model_updates") != 0
                or any(seed_summary.get(key) != 0 for key in ("c09_worlds_read", "c10_worlds_read", "mtare_worlds_read"))
            ):
                raise RuntimeError(f"seed{seed} spatial trace-commit inference evidence drift")
            seed_outputs.append(output)
            cache_bytes = sum(path.stat().st_size for path in cache.rglob("*") if path.is_file())
            shutil.rmtree(cache)
            deletions.append({
                "seed": seed, "deleted_regenerable_cache_bytes": cache_bytes,
                "retained_manifest": str((manifest_dir / f"seed{seed}_manifest.json").relative_to(run_dir)),
            })
        scratch.rmdir()
        write_json(run_dir / "artifacts/cache_deletion_audit.json", {
            "schema_version": "gse_spatial_trace_commit_cache_deletion_audit_v1",
            "records": deletions,
            "reason": "Exact frozen spatial features are deterministic scratch; all digests are retained.",
        })

        ensemble = projection_dir / "spatial_center_ensemble_all_rows.npz"
        combine = [str(PYTHON), str(PROJECT_ROOT / "tools/v3/combine_gse_spatial_longitudinal_center_v1.py")]
        for seed, path in enumerate(seed_outputs):
            combine.extend((f"--seed{seed}", str(path)))
        combine.extend(("--output", str(ensemble)))
        code, rss = _run(combine, run_dir / "logs/06_combine.log", env, 600)
        processes.append({"stage": "combine", "returncode": code, "peak_host_rss_kib": rss})
        reproduction = _verify_training_reproduction(seed_outputs, ensemble)
        write_json(run_dir / "artifacts/projection/qualification_reproduction.json", {
            "schema_version": "gse_spatial_center_qualification_reproduction_v1",
            "selection_rows": 8_839,
            "canonical_inference_batching": "all_188126_rows_global_order_batch128",
            "qualification_subset_batching": "8839_selection_rows_packed_batch128",
            "within_fixed_submillimetre_tolerance": True,
            "numeric_reproduction": reproduction,
            "qualification_source": str(CORRECTIVE.relative_to(PROJECT_ROOT)),
        })

        replay = [
            str(PYTHON), str(PROJECT_ROOT / "tools/v3/execute_gse_trace_commit_capacity_v1.py"),
            "--teacher", str(CAUSAL_TEACHER / "artifacts/teacher_observations.jsonl"),
            "--sequence-manifest", str(DATASET / "artifacts/sequence_manifest.jsonl"),
            "--pair-cache", str(SOURCE / "artifacts/pair_cache/pairs.npz"),
            "--action-cache", str(ACTION / "scratch/action_set_cache"),
            "--action-model-run", str(ACTION), "--association-capacity-run", str(CAPACITY),
            "--center-offsets", str(ensemble), "--fixed-proposal-threshold", "0.97",
        ]
        for seed in range(3):
            replay.extend((f"--observation{seed}", str(CAPACITY / f"artifacts/unified_observation/seed{seed}_unified_observation_features.npy")))
            replay.extend((f"--tokens{seed}", str(SOURCE / f"artifacts/models/seed{seed}/frozen_exit_token_outputs.npz")))
        replay.extend(("--output-dir", str(run_dir / "artifacts/replay")))
        (run_dir / "config/replay_command.txt").write_text(" ".join(replay) + "\n", encoding="utf-8")
        code, rss = _run(replay, run_dir / "logs/07_replay.log", env, 3600, (0, 2))
        processes.append({"stage": "replay", "returncode": code, "peak_host_rss_kib": rss})
        result = load_json(run_dir / "artifacts/replay/summary.json")
        scientific_pass = result.get("status") == "PASS_GSE_TRACE_COMMIT_CAPACITY_V1"
        if (code == 0) != scientific_pass:
            raise RuntimeError("spatial trace-commit replay status mismatch")
        if (
            result.get("proposal_threshold_policy") != "fixed_predecessor_threshold"
            or result.get("selected_proposal_threshold") != 0.97
        ):
            raise RuntimeError("spatial trace-commit fixed threshold drift")
        predecessor = load_json(PREDECESSOR / "artifacts/replay/summary.json")
        old = predecessor["selection"]["gse_trace_commit"]
        new = result["selection"]["gse_trace_commit"]
        predecessor_delta = {
            "node_precision": new["node_precision"] - old["node_precision"],
            "node_recall": new["node_recall"] - old["node_recall"],
            "node_f1": new["node_f1"] - old["node_f1"],
            "edge_precision": new["edge_precision"] - old["edge_precision"],
            "edge_recall": new["edge_recall"] - old["edge_recall"],
            "edge_f1": new["edge_f1"] - old["edge_f1"],
            "node_edge_macro_f1": new["node_edge_macro_f1"] - old["node_edge_macro_f1"],
            "committed_nodes": new["committed_nodes"] - old["committed_nodes"],
            "committed_edges": new["committed_edges"] - old["committed_edges"],
        }
        after = _verify(spec)
        if before != after:
            raise RuntimeError("spatial trace-commit sources changed during replay")
        overall = PASS if scientific_pass else FAIL
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run_dir / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")

    summary = {
        "schema_version": "gse_spatial_trace_commit_requalification_outer_v1",
        "overall_status": overall,
        "scientific_pass": overall == PASS,
        "error": error,
        "result": result,
        "predecessor_delta": predecessor_delta,
        "subprocesses": processes,
        "duration_seconds": time.monotonic() - started,
        "source_unchanged": bool(before and before == after),
        "deleted_regenerable_cache_bytes": sum(value["deleted_regenerable_cache_bytes"] for value in deletions),
        "optimizer_steps": 0,
        "model_updates": 0,
        "c09_worlds_read": 0,
        "c10_worlds_read": 0,
        "mtare_worlds_read": 0,
    }
    write_json(run_dir / "metrics/summary.json", summary)
    write_json(run_dir / "RUN_STATE.json", {
        "schema_version": "v3_run_state_v1", "run_id": RUN_ID,
        "state": "COMPLETED" if overall == PASS else "FAILED",
        "overall_status": overall, "error": error,
    })
    entries = _seal(run_dir)
    print(json.dumps({"overall_status": overall, "error": error, "seal_entries": entries}, indent=2))
    return 0 if overall == PASS else 2


if __name__ == "__main__":
    raise SystemExit(main())
