#!/usr/bin/env python3
"""Execute and seal the post-commit endpoint inventory."""

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


RUN_ID = "gate3_20260828_gse_post_commit_endpoint_inventory_v1_seed0"
PASS = "PASS_GSE_POST_COMMIT_ENDPOINT_INVENTORY_V1"
FAIL = "FAIL_GSE_POST_COMMIT_ENDPOINT_INVENTORY_V1"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_GSE_POST_COMMIT_ENDPOINT_INVENTORY_V1"
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
            raise RuntimeError(f"post-commit endpoint frozen input drift: {relative}")
        result[relative] = actual
    return result


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
        raise RuntimeError("post-commit endpoint inventory may execute only once")
    started = time.monotonic()
    overall, error, result = FAIL, None, {}
    before = after = {}
    returncode = None
    try:
        if spec.get("gate") != 3 or spec.get("operation") != "audit":
            raise RuntimeError("post-commit endpoint scope drift")
        if load_json(PROJECT_ROOT / spec["data_card"]).get("status") != CARD_STATUS:
            raise RuntimeError("post-commit endpoint Data Card drift")
        for record in spec["frozen_tools"].values():
            if _sha(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"post-commit endpoint tool drift: {record['path']}")
        before = _verify(spec)
        write_json(run_dir / "config/source_integrity_before.json", before)
        versions = json.loads(subprocess.check_output([
            str(PYTHON), "-c",
            "import json,matplotlib,numpy,scipy,sys;print(json.dumps({'python':sys.version.split()[0],'numpy':numpy.__version__,'scipy':scipy.__version__,'matplotlib':matplotlib.__version__},sort_keys=True))",
        ], text=True))
        if versions != {"python": "3.13.5", "numpy": "2.1.3", "scipy": "1.15.3", "matplotlib": "3.10.0"}:
            raise RuntimeError(f"post-commit endpoint environment drift: {versions}")
        write_json(run_dir / "config/environment.json", {"executable": str(PYTHON), "versions": versions, "gpu_used": False})
        write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING"})
        teacher_root = PROJECT_ROOT / "results/gate2_representation/gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0/artifacts"
        replay_root = PROJECT_ROOT / "results/gate3_semantics/gate3_20260828_gse_spatial_trace_commit_requalification_v1r2_seed0/artifacts"
        command = [
            str(PYTHON), str(PROJECT_ROOT / "tools/v3/execute_gse_post_commit_endpoint_inventory_v1.py"),
            "--teacher", str(teacher_root / "teacher_observations.jsonl"),
            "--traversals", str(teacher_root / "traversal_manifest.jsonl"),
            "--pair-cache", str(PROJECT_ROOT / "results/gate3_semantics/gate3_20260826_gse_exit_token_association_corrective_v2_seed0/artifacts/pair_cache/pairs.npz"),
            "--action-ensemble", str(replay_root / "replay/action_ensemble.npz"),
            "--spatial-projection", str(replay_root / "projection/spatial_center_ensemble_all_rows.npz"),
            "--association-pairs", str(replay_root / "replay/association_pairs.npz"),
            "--objective-teacher", str(PROJECT_ROOT / "results/gate3_semantics/gate3_20260828_gse_event_center_offset_training_v1r3_seed0/artifacts/teacher/event_center_teacher.npz"),
            "--output-dir", str(run_dir / "artifacts/inventory"),
        ]
        (run_dir / "config/inventory_command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3")
        with (run_dir / "logs/inventory.log").open("w", encoding="utf-8") as stream:
            completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env, text=True, stdout=stream, stderr=subprocess.STDOUT, timeout=600, check=False)
        returncode = int(completed.returncode)
        if returncode != 0:
            raise RuntimeError(f"post-commit endpoint executor failed with code {returncode}")
        result = load_json(run_dir / "artifacts/inventory/summary.json")
        fit = result.get("scores", {}).get("fit", {}).get("factorized_hybrid", {})
        selection = result.get("scores", {}).get("selection", {}).get("factorized_hybrid", {})
        if (
            result.get("status") != PASS or not result.get("gates", {}).get("inventory_complete")
            or result.get("gates", {}).get("training_authorized") is not False
            or result.get("gates", {}).get("factorized_hybrid_full_graph_capacity") is not False
            or fit.get("committed_nodes") != 667 or fit.get("correct_unique_nodes") != 654
            or fit.get("committed_edges") != 18 or fit.get("correct_unique_edges") != 15
            or selection.get("committed_nodes") != 212 or selection.get("correct_unique_nodes") != 212
            or selection.get("committed_edges") != 4 or selection.get("correct_unique_edges") != 4
            or any(result.get(key) != 0 for key in ("optimizer_steps", "model_inference_frames", "model_updates", "c09_worlds_read", "c10_worlds_read", "mtare_worlds_read"))
        ):
            raise RuntimeError("post-commit endpoint result contract drift")
        required = (
            "summary.json", "capacity_scores.csv", "gse_post_commit_endpoint_inventory.png",
            "gse_post_commit_endpoint_inventory.pdf", "gse_post_commit_endpoint_inventory.svg",
            "figure_source.json", "fit_factorized_hybrid_edges.jsonl", "selection_factorized_hybrid_edges.jsonl",
            "fit_support2_endpoint_candidates.jsonl", "selection_support2_endpoint_candidates.jsonl",
        )
        if any(not (run_dir / "artifacts/inventory" / name).is_file() for name in required):
            raise RuntimeError("post-commit endpoint evidence incomplete")
        after = _verify(spec)
        if before != after:
            raise RuntimeError("post-commit endpoint sources changed")
        overall = PASS
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run_dir / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
    write_json(run_dir / "metrics/summary.json", {
        "schema_version": "gse_post_commit_endpoint_inventory_outer_v1", "overall_status": overall,
        "error": error, "result": result, "executor_returncode": returncode,
        "duration_seconds": time.monotonic() - started, "source_unchanged": bool(before and before == after),
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
