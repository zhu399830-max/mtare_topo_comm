#!/usr/bin/env python3
"""Execute and seal the spatial event-center residual audit."""

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


RUN_ID = "gate3_20260828_gse_spatial_center_residual_audit_v1_seed0"
PYTHON = Path("/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python")
PASS = "PASS_GSE_SPATIAL_CENTER_RESIDUAL_AUDIT_V1"
FAIL = "FAIL_GSE_SPATIAL_CENTER_RESIDUAL_AUDIT_V1"
TRAINING = PROJECT_ROOT / "results/gate3_semantics/gate3_20260828_gse_spatial_event_center_training_v1_seed0"
QUALIFICATION = PROJECT_ROOT / "results/gate3_semantics/gate3_20260828_gse_spatial_event_center_qualification_v1r_seed0"
ACTION = PROJECT_ROOT / "results/gate3_semantics/gate3_20260828_gse_action_set_node_training_v1_seed0"
CENTER = PROJECT_ROOT / "results/gate3_semantics/gate3_20260828_gse_event_center_offset_training_v1r3_seed0"
PROJECTION = PROJECT_ROOT / "results/gate3_semantics/gate3_20260828_gse_projected_trace_commit_capacity_v1_seed0"


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
            raise RuntimeError(f"residual audit frozen input drift: {relative}")
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
    parser = argparse.ArgumentParser(); parser.add_argument("--spec", required=True, type=Path); parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args(); run_dir = args.run_dir.resolve(); spec = load_json(args.spec.resolve())
    if run_dir.name != RUN_ID or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("residual audit may execute only once")
    started = time.monotonic(); overall = FAIL; error = None; audit = {}; before = after = {}; returncode = None
    try:
        if spec.get("gate") != 3 or spec.get("operation") != "audit":
            raise RuntimeError("residual audit scope drift")
        for record in spec["frozen_tools"].values():
            if _sha(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"residual audit frozen tool drift: {record['path']}")
        before = _verify(spec); write_json(run_dir / "config/source_integrity_before.json", before)
        write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING"})
        env = os.environ.copy(); env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3")
        output = run_dir / "artifacts/audit"
        command = [
            str(PYTHON), str(PROJECT_ROOT / "tools/v3/execute_gse_spatial_center_residual_audit_v1.py"),
            "--ensemble", str(QUALIFICATION / "metrics/ensemble/ensemble_selection_outputs.npz"),
            "--training-root", str(TRAINING / "artifacts/models"),
            "--teacher", str(CENTER / "artifacts/teacher/event_center_teacher.npz"),
            "--action-cache", str(ACTION / "scratch/action_set_cache"),
            "--baseline-projection", str(PROJECTION / "artifacts/projection/event_center_projection.npz"),
            "--output-dir", str(output),
        ]
        with (run_dir / "logs/01_audit.log").open("w", encoding="utf-8") as stream:
            completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env, text=True, stdout=stream, stderr=subprocess.STDOUT, timeout=240, check=False)
        returncode = int(completed.returncode)
        if returncode != 0:
            raise RuntimeError("residual audit executor failed")
        audit = load_json(output / "summary.json")
        if (
            audit.get("selection_rows") != 8839 or audit.get("identities") != 272
            or audit.get("cross_traversal_pairs") != 144533
            or any(audit.get(key) != 0 for key in ("c09_worlds_read", "c10_worlds_read", "mtare_worlds_read"))
            or set(audit.get("variant_metrics", {})) != {
                "scalar_zero_transverse", "spatial_full", "predicted_longitudinal_oracle_transverse",
                "oracle_longitudinal_predicted_transverse", "oracle_all",
                "diagnostic_coordinate_median", "diagnostic_per_row_medoid",
            }
        ):
            raise RuntimeError("residual audit evidence contract drift")
        sealed = load_json(QUALIFICATION / "metrics/ensemble/summary.json")
        scalar = audit["variant_metrics"]["scalar_zero_transverse"]
        spatial = audit["variant_metrics"]["spatial_full"]
        if (
            abs(scalar["identity_macro_within_fraction"] - sealed["baseline_cross_view"]["identity_macro_within_4m_fraction"]) > 1e-12
            or abs(spatial["identity_macro_within_fraction"] - sealed["ensemble_cross_view"]["identity_macro_within_4m_fraction"]) > 1e-12
            # The sealed ensemble metric is computed before its centers are
            # persisted as float32. Reconstructing from that archive is exact
            # to float32 quantization, not to float64 accumulation order.
            or abs(spatial["identity_macro_mean_distance_m"] - sealed["ensemble_cross_view"]["identity_macro_relative_vector_error_m"]) > 1e-6
        ):
            raise RuntimeError("residual audit failed to reproduce sealed metrics")
        after = _verify(spec)
        if before != after:
            raise RuntimeError("residual audit sources changed")
        overall = PASS
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"; (run_dir / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
    summary = {
        "schema_version": "gse_spatial_center_residual_audit_outer_v1",
        "overall_status": overall, "error": error, "executor_returncode": returncode,
        "audit": audit, "optimizer_steps": 0, "duration_seconds": time.monotonic() - started,
        "source_unchanged": bool(before and before == after),
        "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0,
    }
    write_json(run_dir / "metrics/summary.json", summary)
    write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "COMPLETED" if overall == PASS else "FAILED", "overall_status": overall, "error": error})
    entries = _seal(run_dir); print(json.dumps({"overall_status": overall, "error": error, "seal_entries": entries}, indent=2)); return 0 if overall == PASS else 2


if __name__ == "__main__":
    raise SystemExit(main())
