#!/usr/bin/env python3
"""Execute and seal C01-C08 execution-endpoint geometry capacity."""

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


RUN_ID = "gate3_20260828_gse_endpoint_geometry_capacity_v1_seed0"
PASS = "PASS_GSE_ENDPOINT_GEOMETRY_CAPACITY_V1"
FAIL = "FAIL_GSE_ENDPOINT_GEOMETRY_CAPACITY_V1"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_GSE_ENDPOINT_GEOMETRY_CAPACITY_V1"
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
            raise RuntimeError(f"endpoint geometry frozen input drift: {relative}")
        result[relative] = actual
    return result


def _verify_axis_subset(dataset_run: Path) -> int:
    seal = dataset_run / "artifacts/evidence_sha256.txt"
    count = 0
    for line in seal.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        is_frame_manifest = relative.endswith("/artifacts/frame_manifest.jsonl")
        is_axis = "/artifacts/dataset/train/" in relative and ".zarr/axis_xyz_m/" in relative
        if not (is_frame_manifest or is_axis):
            continue
        path = PROJECT_ROOT / relative
        if _sha(path) != expected:
            raise RuntimeError(f"endpoint geometry dataset seal mismatch: {relative}")
        count += 1
    if count < 81:
        raise RuntimeError(f"endpoint geometry dataset seal coverage drift: {count}")
    return count


def _seal(run_dir: Path) -> int:
    seal = run_dir / "artifacts/evidence_sha256.txt"
    files = sorted(path for path in run_dir.rglob("*") if path.is_file() and path != seal)
    with seal.open("w", encoding="utf-8") as stream:
        for path in files:
            stream.write(f"{_sha(path)}  {path.relative_to(PROJECT_ROOT)}\n")
    return len(files)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    spec = load_json(args.spec.resolve())
    if run_dir.name != RUN_ID or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("endpoint geometry capacity may execute only once")
    started = time.monotonic()
    overall, error, result = FAIL, None, {}
    before = after = {}
    returncode = None
    axis_seal_entries = 0
    try:
        if spec.get("gate") != 3 or spec.get("operation") != "audit":
            raise RuntimeError("endpoint geometry scope drift")
        if load_json(PROJECT_ROOT / spec["data_card"]).get("status") != CARD_STATUS:
            raise RuntimeError("endpoint geometry Data Card drift")
        for record in spec["frozen_tools"].values():
            if _sha(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"endpoint geometry tool drift: {record['path']}")
        before = _verify(spec)
        dataset_run = PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
        axis_seal_entries = _verify_axis_subset(dataset_run)
        write_json(run_dir / "config/source_integrity_before.json", {"frozen_inputs": before, "axis_subset_seal_entries": axis_seal_entries})
        versions = json.loads(subprocess.check_output([
            str(PYTHON), "-c",
            "import json,matplotlib,numpy,scipy,sys,zarr;print(json.dumps({'python':sys.version.split()[0],'numpy':numpy.__version__,'scipy':scipy.__version__,'matplotlib':matplotlib.__version__,'zarr':zarr.__version__},sort_keys=True))",
        ], text=True))
        if versions != {"python": "3.13.5", "numpy": "2.1.3", "scipy": "1.15.3", "matplotlib": "3.10.0", "zarr": "2.18.7"}:
            raise RuntimeError(f"endpoint geometry environment drift: {versions}")
        write_json(run_dir / "config/environment.json", {"executable": str(PYTHON), "versions": versions, "gpu_used": False})
        write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING"})
        teacher_root = PROJECT_ROOT / "results/gate2_representation/gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0/artifacts"
        replay_root = PROJECT_ROOT / "results/gate3_semantics/gate3_20260828_gse_spatial_trace_commit_requalification_v1r2_seed0/artifacts"
        command = [
            str(PYTHON), str(PROJECT_ROOT / "tools/v3/execute_gse_endpoint_geometry_capacity_v1.py"),
            "--teacher", str(teacher_root / "teacher_observations.jsonl"),
            "--traversals", str(teacher_root / "traversal_manifest.jsonl"),
            "--frame-manifest", str(dataset_run / "artifacts/frame_manifest.jsonl"),
            "--dataset-root", str(dataset_run / "artifacts/dataset"),
            "--pair-cache", str(PROJECT_ROOT / "results/gate3_semantics/gate3_20260826_gse_exit_token_association_corrective_v2_seed0/artifacts/pair_cache/pairs.npz"),
            "--action-ensemble", str(replay_root / "replay/action_ensemble.npz"),
            "--spatial-projection", str(replay_root / "projection/spatial_center_ensemble_all_rows.npz"),
            "--association-pairs", str(replay_root / "replay/association_pairs.npz"),
            "--objective-teacher", str(PROJECT_ROOT / "results/gate3_semantics/gate3_20260828_gse_event_center_offset_training_v1r3_seed0/artifacts/teacher/event_center_teacher.npz"),
            "--baseline-summary", str(PROJECT_ROOT / "results/gate3_semantics/gate3_20260828_gse_post_commit_endpoint_inventory_v1_seed0/artifacts/inventory/summary.json"),
            "--output-dir", str(run_dir / "artifacts/capacity"),
        ]
        (run_dir / "config/capacity_command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3")
        with (run_dir / "logs/capacity.log").open("w", encoding="utf-8") as stream:
            completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env, text=True, stdout=stream, stderr=subprocess.STDOUT, timeout=600, check=False)
        returncode = int(completed.returncode)
        if returncode != 0:
            raise RuntimeError(f"endpoint geometry executor failed with code {returncode}")
        result = load_json(run_dir / "artifacts/capacity/summary.json")
        fit = result.get("scores", {}).get("fit", {})
        selection = result.get("scores", {}).get("selection", {})
        method = result.get("method", {})
        inventory = result.get("geometry_inventory", {})
        if (
            result.get("status") != PASS or not result.get("gates", {}).get("all_passed")
            or fit.get("committed_nodes") != 598 or fit.get("correct_unique_nodes") != 598
            or fit.get("committed_edges") != 13 or fit.get("correct_unique_edges") != 13
            or selection.get("committed_nodes") != 193 or selection.get("correct_unique_nodes") != 193
            or selection.get("committed_edges") != 4 or selection.get("correct_unique_edges") != 4
            or inventory.get("declared_directed_traversals") != 16078
            or inventory.get("framed_directed_traversals") != 16076
            or inventory.get("frames_read") != 252430
            or method.get("c07_c08_used_for_method_confirmation") is not True
            or method.get("c07_c08_used_for_numeric_threshold_selection") is not False
            or any(result.get(key) != 0 for key in ("optimizer_steps", "model_inference_frames", "model_updates", "c09_worlds_read", "c10_worlds_read", "mtare_worlds_read"))
        ):
            raise RuntimeError("endpoint geometry result contract drift")
        required = (
            "summary.json", "capacity_scores.csv", "fit_hypothesis_qualification.jsonl",
            "selection_hypothesis_qualification.jsonl", "fit_edge_audit.jsonl", "selection_edge_audit.jsonl",
            "gse_endpoint_geometry_capacity.png", "gse_endpoint_geometry_capacity.pdf",
            "gse_endpoint_geometry_capacity.svg", "figure_source.json",
        )
        if any(not (run_dir / "artifacts/capacity" / name).is_file() for name in required):
            raise RuntimeError("endpoint geometry evidence incomplete")
        after = _verify(spec)
        if before != after or _verify_axis_subset(dataset_run) != axis_seal_entries:
            raise RuntimeError("endpoint geometry sources changed")
        overall = PASS
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run_dir / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
    write_json(run_dir / "metrics/summary.json", {
        "schema_version": "gse_endpoint_geometry_capacity_outer_v1", "overall_status": overall,
        "error": error, "result": result, "executor_returncode": returncode,
        "axis_subset_seal_entries": axis_seal_entries, "duration_seconds": time.monotonic() - started,
        "source_unchanged": bool(before and before == after),
        "optimizer_steps": 0, "model_inference_frames": 0, "model_updates": 0,
        "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0,
    })
    write_json(run_dir / "RUN_STATE.json", {
        "schema_version": "v3_run_state_v1", "run_id": RUN_ID,
        "state": "COMPLETED" if overall == PASS else "FAILED", "overall_status": overall, "error": error,
    })
    entries = _seal(run_dir)
    print(json.dumps({"overall_status": overall, "error": error, "seal_entries": entries}, indent=2))
    return 0 if overall == PASS else 2


if __name__ == "__main__":
    raise SystemExit(main())
