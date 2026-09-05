#!/usr/bin/env python3
"""Execute and seal one immutable frozen C09 endpoint-geometry validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


RUN_ID = "gate3_20260828_gse_endpoint_geometry_c09_validation_v1_seed0"
PASS = "PASS_GSE_ENDPOINT_GEOMETRY_C09_VALIDATION_V1"
FAIL = "FAIL_GSE_ENDPOINT_GEOMETRY_C09_VALIDATION_V1"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_GSE_ENDPOINT_GEOMETRY_C09_VALIDATION_V1"
PYTHON = Path("/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python")


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
            raise RuntimeError(f"C09 endpoint-geometry frozen input drift: {relative}")
        result[relative] = actual
    return result


def _verify_dataset_subset(dataset_run: Path) -> int:
    seal = dataset_run / "artifacts/evidence_sha256.txt"
    count = 0
    for line in seal.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        selected = (
            relative.endswith("/artifacts/frame_manifest.jsonl")
            or relative.endswith("/artifacts/sequence_manifest.jsonl")
            or "/artifacts/dataset/validation/" in relative
        )
        if not selected:
            continue
        if _sha(PROJECT_ROOT / relative) != expected:
            raise RuntimeError(f"C09 dataset seal mismatch: {relative}")
        count += 1
    if count < 100:
        raise RuntimeError(f"C09 dataset seal coverage drift: {count}")
    return count


def _seal(run_dir: Path) -> int:
    seal = run_dir / "artifacts/evidence_sha256.txt"
    files = sorted(path for path in run_dir.rglob("*") if path.is_file() and path != seal)
    with seal.open("w", encoding="utf-8") as stream:
        for path in files:
            stream.write(f"{_sha(path)}  {path.relative_to(PROJECT_ROOT)}\n")
    return len(files)


def _peak_rss(path: Path) -> int | None:
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "Maximum resident set size (kbytes):" in line:
            return int(line.rsplit(":", 1)[1].strip())
    return None


def _run(command: list[str], log: Path, env: dict[str, str], timeout: int, allowed=(0,)) -> tuple[int, int | None]:
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            ["/usr/bin/time", "-v", *command], cwd=PROJECT_ROOT, env=env,
            text=True, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout, check=False,
        )
    code = int(completed.returncode)
    if code not in allowed:
        raise RuntimeError(f"C09 endpoint-geometry subprocess failed: {Path(command[1]).name} code={code}")
    return code, _peak_rss(log)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve(); spec = load_json(args.spec.resolve())
    if run_dir.name != RUN_ID or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("C09 endpoint-geometry validation may execute only once")
    started = time.monotonic(); overall = FAIL; error = None
    before = after = {}; result = {}; inference = {}; processes = []
    dataset_entries = 0
    try:
        if spec.get("gate") != 3 or spec.get("operation") != "audit" or spec.get("seed") != 0:
            raise RuntimeError("C09 endpoint-geometry validation scope drift")
        card = load_json(PROJECT_ROOT / spec["data_card"])
        if card.get("status") != CARD_STATUS or card.get("approval", {}).get("status") != "APPROVED":
            raise RuntimeError("C09 endpoint-geometry Data Card drift")
        for record in spec["frozen_tools"].values():
            if _sha(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"C09 endpoint-geometry tool drift: {record['path']}")
        before = _verify(spec)
        dataset = PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
        dataset_entries = _verify_dataset_subset(dataset)
        write_json(run_dir / "config/source_integrity_before.json", {
            "frozen_inputs": before, "dataset_subset_seal_entries": dataset_entries,
        })
        versions = json.loads(subprocess.check_output([
            str(PYTHON), "-c",
            "import json,matplotlib,numpy,scipy,sys,torch,zarr;print(json.dumps({'python':sys.version.split()[0],'numpy':numpy.__version__,'scipy':scipy.__version__,'matplotlib':matplotlib.__version__,'torch':torch.__version__,'cuda':torch.version.cuda,'zarr':zarr.__version__,'gpu':torch.cuda.get_device_name(0)},sort_keys=True))",
        ], text=True))
        expected = {
            "python": "3.13.5", "numpy": "2.1.3", "scipy": "1.15.3",
            "matplotlib": "3.10.0", "torch": "2.9.0+cu129", "cuda": "12.9",
            "zarr": "2.18.7", "gpu": "NVIDIA GeForce RTX 5090 D",
        }
        if versions != expected:
            raise RuntimeError(f"C09 endpoint-geometry environment drift: {versions}")
        write_json(run_dir / "config/environment.json", {
            "executable": str(PYTHON), "versions": versions,
            "deterministic_algorithms": True, "optimizer_steps": 0,
        })
        write_json(run_dir / "RUN_STATE.json", {
            "schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING",
            "note": "Process 1 freezes a Teacher-free C09 graph; process 2 opens Teacher only for posthoc scoring.",
        })
        env = os.environ.copy()
        env.update({
            "PYTHONPATH": str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3"),
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
        })
        training = PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_graph_three_seed_training_v1r_seed0"
        teacher = PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_teacher_manifest_v1_seed0"
        action = PROJECT_ROOT / "results/gate3_semantics/gate3_20260828_gse_action_set_node_training_v1_seed0"
        scalar = PROJECT_ROOT / "results/gate3_semantics/gate3_20260828_gse_event_center_pair_consistency_training_v1r3_seed0"
        spatial = PROJECT_ROOT / "results/gate3_semantics/gate3_20260828_gse_spatial_event_center_training_v1_seed0"
        corrective = PROJECT_ROOT / "results/gate3_semantics/gate3_20260828_gse_spatial_longitudinal_corrective_v1_seed0"
        association = PROJECT_ROOT / "results/gate3_semantics/gate3_20260827_gse_factorized_association_c09_qualification_v1_seed0"
        consensus = PROJECT_ROOT / "results/gate3_semantics/gate3_20260828_gse_factorized_consensus_metric_corrective_v1_seed0"
        development = PROJECT_ROOT / "results/gate3_semantics/gate3_20260828_gse_endpoint_geometry_capacity_v1_seed0"
        graph_dir = run_dir / "artifacts/teacher_free_graph"
        infer = [
            str(PYTHON), str(PROJECT_ROOT / "tools/v3/infer_gse_endpoint_geometry_c09_v1.py"),
            "--dataset-run", str(dataset),
            "--sequence-manifest", str(dataset / "artifacts/sequence_manifest.jsonl"),
            "--frame-manifest", str(dataset / "artifacts/frame_manifest.jsonl"),
            "--traversal-manifest", str(teacher / "artifacts/traversal_manifest.jsonl"),
            "--action-model-run", str(action), "--training-run", str(training),
            "--scalar-run", str(scalar), "--spatial-run", str(spatial),
            "--corrective-run", str(corrective),
            "--association-pairs", str(association / "artifacts/c09_runtime_candidate_pairs.npz"),
            "--association-decisions", str(consensus / "artifacts/c09_application/c09_consensus_metric_decisions.npz"),
            "--output-dir", str(graph_dir),
        ]
        (run_dir / "config/command.txt").write_text(" ".join(infer) + "\n", encoding="utf-8")
        code, rss = _run(infer, run_dir / "logs/00_teacher_free_inference.log", env, 3600)
        processes.append({"stage": "teacher_free_inference", "returncode": code, "peak_host_rss_kib": rss})
        inference = load_json(graph_dir / "inference_manifest.json")
        if (
            inference.get("status") != "PASS_TEACHER_FREE_C09_GRAPH_FREEZE_V1"
            or inference.get("worlds") != 10 or inference.get("causal_observations") != 24_462
            or inference.get("unique_lidar_frames") != 32_678 or inference.get("directed_traversals") != 2_054
            or inference.get("teacher_inputs_read") != 0 or inference.get("teacher_identity_inputs_read") != 0
            or any(inference.get(key) != 0 for key in (
                "optimizer_steps", "model_updates", "checkpoint_selection_steps",
                "threshold_selection_steps", "c10_worlds_read", "mtare_worlds_read",
            ))
            or rss is None or rss > 10 * 1024**2
        ):
            raise RuntimeError("C09 teacher-free graph evidence contract drift")
        evaluation_dir = run_dir / "artifacts/evaluation"
        evaluate = [
            str(PYTHON), str(PROJECT_ROOT / "tools/v3/evaluate_gse_endpoint_geometry_c09_v1.py"),
            "--frozen-graph-dir", str(graph_dir),
            "--teacher", str(teacher / "artifacts/teacher_observations.jsonl"),
            "--parent-manifest", str(PROJECT_ROOT / "results/gate0_baseline/gate0_20260811_cano_100_topology_parent_recipe_reclassification_v2r_seed0/artifacts/accepted_parent_manifest.json"),
            "--development-summary", str(development / "artifacts/capacity/summary.json"),
            "--output-dir", str(evaluation_dir),
        ]
        with (run_dir / "config/command.txt").open("a", encoding="utf-8") as stream:
            stream.write(" ".join(evaluate) + "\n")
        code, rss = _run(evaluate, run_dir / "logs/01_posthoc_teacher_evaluation.log", env, 600, (0, 2))
        processes.append({"stage": "posthoc_teacher_evaluation", "returncode": code, "peak_host_rss_kib": rss})
        result = load_json(evaluation_dir / "summary.json")
        if (
            result.get("overall_status") not in (PASS, FAIL)
            or (code == 0) != bool(result.get("scientific_pass"))
            or result.get("objective_nodes") != 130
            or result.get("teacher_rows_read_posthoc") != 24_462
            or result.get("graph_frozen_before_teacher") is not True
            or any(result.get(key) != 0 for key in (
                "optimizer_steps", "model_updates", "checkpoint_selection_steps",
                "threshold_selection_steps", "c10_worlds_read", "mtare_worlds_read",
            ))
            or rss is None or rss > 4 * 1024**2
        ):
            raise RuntimeError("C09 posthoc evaluation evidence contract drift")
        required = [
            graph_dir / "verified_nodes.jsonl", graph_dir / "verified_edges.jsonl",
            graph_dir / "hypothesis_qualification.jsonl", graph_dir / "decision_trace.jsonl",
            graph_dir / "action_ensemble.npz", graph_dir / "spatial_center_ensemble.npz",
            evaluation_dir / "summary.json", evaluation_dir / "scores.csv",
            evaluation_dir / "gse_endpoint_geometry_c09.png",
            evaluation_dir / "gse_endpoint_geometry_c09.pdf",
            evaluation_dir / "gse_endpoint_geometry_c09.svg", evaluation_dir / "figure_source.json",
        ]
        if not all(path.is_file() for path in required):
            raise RuntimeError("C09 endpoint-geometry evidence incomplete")
        after = _verify(spec)
        if before != after or _verify_dataset_subset(dataset) != dataset_entries:
            raise RuntimeError("C09 endpoint-geometry sources changed")
        output_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
        if output_bytes > 3 * 1024**3:
            raise RuntimeError("C09 endpoint-geometry output exceeds 3 GiB")
        overall = str(result["overall_status"])
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run_dir / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
    write_json(run_dir / "metrics/summary.json", {
        "schema_version": "gse_endpoint_geometry_c09_validation_outer_v1",
        "overall_status": overall, "scientific_pass": overall == PASS, "error": error,
        "inference": inference, "evaluation": result, "processes": processes,
        "dataset_subset_seal_entries": dataset_entries,
        "source_unchanged": bool(before and before == after),
        "optimizer_steps": 0, "model_updates": 0, "checkpoint_selection_steps": 0,
        "threshold_selection_steps": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0,
        "duration_seconds": time.monotonic() - started,
    })
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
