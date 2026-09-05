#!/usr/bin/env python3
from __future__ import annotations

import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import write_json


RUN_ID = "gate3_20260828_gse_structured_polar_multidepth_readiness_v1_seed0"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_structured_polar_multidepth_readiness_v1.json"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if SPEC.exists():
        raise RuntimeError("structured-polar readiness spec already exists")
    dataset = "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
    teacher = "results/gate2_representation/gate2_20260828_gse_observable_spatial_event_teacher_v2_seed0"
    feasibility = "results/gate3_semantics/gate3_20260828_gse_structured_polar_teacher_feasibility_v1_seed0"
    inputs = [
        f"{dataset}/RUN_STATE.json", f"{dataset}/metrics/summary.json", f"{dataset}/artifacts/evidence_sha256.txt", f"{dataset}/artifacts/shard_manifest.json",
        f"{teacher}/RUN_STATE.json", f"{teacher}/metrics/summary.json", f"{teacher}/artifacts/evidence_sha256.txt", f"{teacher}/artifacts/export/summary.json", f"{teacher}/artifacts/export/teacher_shard_manifest.jsonl",
        f"{feasibility}/RUN_STATE.json", f"{feasibility}/metrics/summary.json", f"{feasibility}/artifacts/audit/summary.json", f"{feasibility}/artifacts/evidence_sha256.txt",
    ]
    tools = {
        "model": "src/mtare_topo/representation/gse_structured_polar_multidepth_event.py",
        "set_contract": "src/mtare_topo/representation/gse_spatial_event_set.py",
        "executor": "tools/v3/execute_gse_structured_polar_multidepth_readiness_v1.py",
        "runner": "tools/v3/run_gse_structured_polar_multidepth_readiness_v1.py",
        "freezer": "tools/v3/freeze_gse_structured_polar_multidepth_readiness_v1_spec.py",
        "model_tests": "tests/v3/unit/test_gse_structured_polar_multidepth_event.py",
        "feasibility_tests": "tests/v3/unit/test_gse_structured_polar_teacher_feasibility.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py",
        "create_run": "tools/v3/create_run.py",
    }
    authorization = {"status": "APPROVED", "approved_by": "user-standing-authorization", "approved_at": "2026-08-28T23:45:00+08:00", "authorized_gates": [3], "authorized_operations": ["audit"], "scope": "One immutable zero-training C01-C08 180x2 structured-polar multi-depth model readiness.", "confirmation_reference": "User instructed automatic best-choice execution without routine approval prompts."}
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "execution_phase": 3, "date": "20260828", "slug": "gse_structured_polar_multidepth_readiness_v1", "seed": 0, "operation": "audit",
        "question": "Can the fixed 180x2 range/valid-only event field round-trip all observable tokens and satisfy physical, rotation, deterministic-top-k and gradient contracts before training?",
        "method": "Rasterize every C01-C08 Teacher token into distance-sorted 180x2 slots using the exact five-frame local free-range profile; audit lossless reconstruction and run a 172430-parameter dense circular encoder on real cardinality 0-5 plus a same-ray pair.",
        "baseline": "The 16-free-query model failed spatial proposal oracle coverage; feasibility proved exactly 277 second-depth tokens and maximum occupancy two.",
        "fallback": "Any overflow, support, rotation, top-k, bound or gradient failure stops training and revises only the declared raster/model interface without test access.",
        "user_authorization": authorization,
        "expected_counts": {"worlds": 80, "observations": 188126, "tokens": 131424, "event_identities": 1076, "fit_tokens": 98279, "selection_tokens": 33145, "second_depth_tokens": 277, "parameters": 172430, "optimizer_steps": 0, "trained_model_inference_frames": 0, "threshold_selection_steps": 0, "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0},
        "acceptance_criteria": ["Exact C01-C08 population, cardinality, identity, split and 277 second-depth tokens.", "All 131424 tokens rasterize and reconstruct within 0.1mm without overflow; radial fraction <=1 and azimuth residual <=1 degree.", "Model accepts only five-frame range/valid, has exactly 172430 parameters, 180x2 dense slots and top16 without neighbor NMS.", "NumPy/Torch support parity <=1e-6, range/azimuth bounds, finite empty/multi-depth backward, unique deterministic top16 and seed replay.", "20-degree dense and top16 rotation errors <=0.2mm with exact bin/slot shift; zero training/threshold/C09/C10/M-TARE/graph/planner."],
        "expected_evidence": ["80-world rasterizer log, per-world slot counts and exact round-trip/bound metrics.", "Model/interface/rotation/top-k/determinism/gradient metrics, environment, RUN_STATE and seal."],
        "estimated_cost": {"compute": "Sequential CPU read of C01-C08 shards, dense target rasterization and seven small CPU model forwards plus one backward.", "wall_time_hours": 0.4, "host_ram_gb": 5, "gpu_memory_gb": 0, "disk_gb": 0.05, "gpu": "none"},
        "frozen_inputs": {path: _sha(PROJECT_ROOT / path) for path in sorted(inputs)},
        "frozen_tools": {name: {"path": path, "sha256": _sha(PROJECT_ROOT / path)} for name, path in tools.items()},
        "working_directory": str(PROJECT_ROOT),
        "command": ["/usr/bin/timeout", "2400s", PYTHON, "tools/v3/run_gse_structured_polar_multidepth_readiness_v1.py", "--spec", str(SPEC), "--run-dir", str(PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}")],
    }
    write_json(SPEC, spec)
    print(SPEC.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
