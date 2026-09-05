#!/usr/bin/env python3
from __future__ import annotations

import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import write_json


RUN_ID = "gate3_20260828_gse_structured_polar_teacher_feasibility_v1_seed0"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_structured_polar_teacher_feasibility_v1.json"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if SPEC.exists():
        raise RuntimeError("structured-polar feasibility spec already exists")
    teacher = "results/gate2_representation/gate2_20260828_gse_observable_spatial_event_teacher_v2_seed0"
    readiness = "results/gate3_semantics/gate3_20260828_gse_geometry_anchored_spatial_event_readiness_v2_seed0"
    attribution = "results/gate3_semantics/gate3_20260828_gse_geometry_anchored_proposal_failure_attribution_v1r_seed0"
    inputs = [
        f"{teacher}/RUN_STATE.json", f"{teacher}/metrics/summary.json", f"{teacher}/artifacts/evidence_sha256.txt", f"{teacher}/artifacts/export/summary.json", f"{teacher}/artifacts/export/teacher_shard_manifest.jsonl",
        f"{readiness}/RUN_STATE.json", f"{readiness}/metrics/summary.json", f"{readiness}/artifacts/audit/summary.json", f"{readiness}/artifacts/audit/support_audit.json", f"{readiness}/artifacts/evidence_sha256.txt",
        f"{attribution}/RUN_STATE.json", f"{attribution}/metrics/summary.json", f"{attribution}/artifacts/audit/summary.json", f"{attribution}/artifacts/evidence_sha256.txt",
    ]
    tools = {
        "teacher_loader": "src/mtare_topo/data/gse_observable_spatial_event_dataset.py",
        "executor": "tools/v3/execute_gse_structured_polar_teacher_feasibility_v1.py",
        "runner": "tools/v3/run_gse_structured_polar_teacher_feasibility_v1.py",
        "freezer": "tools/v3/freeze_gse_structured_polar_teacher_feasibility_v1_spec.py",
        "tests": "tests/v3/unit/test_gse_structured_polar_teacher_feasibility.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py",
        "create_run": "tools/v3/create_run.py",
    }
    authorization = {"status": "APPROVED", "approved_by": "user-standing-authorization", "approved_at": "2026-08-28T23:20:00+08:00", "authorized_gates": [3], "authorized_operations": ["audit"], "scope": "One immutable zero-training C01-C08 structured-polar Teacher slot feasibility audit.", "confirmation_reference": "User instructed automatic best-choice execution without routine approval prompts."}
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "execution_phase": 3, "date": "20260828", "slug": "gse_structured_polar_teacher_feasibility_v1", "seed": 0, "operation": "audit",
        "question": "Can all 131424 observable structure tokens receive deterministic collision-resolved polar proposal supervision with the sealed five-frame free-range support?",
        "method": "Assign each Teacher token to the nearest fixed 2-degree azimuth bin and nearest fixed four-band elevation center; audit exact/neighbor bin collisions, per-row occupancy, angular residual, integer circular rotation, and bind the sealed all-token free-range support proof.",
        "baseline": "The failed 16-free-query model has only 11.6-12.5% all-query typed oracle recall and approximately 29000 unsupported target positions per seed.",
        "fallback": "Select the least complex lossless slot scheme: 180-bin local maxima, 180-bin top-k without neighbor suppression, 180x4 azimuth-elevation, or multi-depth polar slots; any support/index failure stops implementation.",
        "user_authorization": authorization,
        "hyperparameters": {"azimuth_bins": 180, "bin_width_deg": 2.0, "bin_center_offset_deg": 0.75, "neighbor_radius_bins": 1, "elevation_centers_deg": [-12, -4, 4, 12], "rotation_test_bins": 17},
        "expected_counts": {"worlds": 80, "observations": 188126, "tokens": 131424, "event_identities": 1076, "fit_tokens": 98279, "selection_tokens": 33145, "multi_event_rows": 23617, "optimizer_steps": 0, "model_inference_frames": 0, "threshold_selection_steps": 0, "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0},
        "acceptance_criteria": ["Exact 80 worlds, 188126 observations, 131424 tokens, 1076 identities and fixed fit/selection counts.", "Every token maps within one degree of a fixed azimuth center and a deterministic 34-degree rotation shifts exactly 17 bins.", "Reuse of the sealed five-frame local support proof shows zero unsupported tokens and positive minimum margin.", "Emit exactly one least-complex lossless polar slot scheme and preserve every collision example; no label deletion.", "Zero optimizer, model inference, threshold selection, C09/C10/M-TARE, graph or planner; inputs unchanged and evidence sealed."],
        "expected_evidence": ["Exact same/neighbor-bin and azimuth-elevation conflicts, occupancies, minimum separations and collision examples.", "Full 180-bin/elevation histograms, PNG/PDF/SVG/source, logs, RUN_STATE and seal."],
        "estimated_cost": {"compute": "CPU read-only pass over Observable Teacher V2 plus sealed readiness summary; no scan reread or model inference.", "wall_time_hours": 0.08, "host_ram_gb": 2, "gpu_memory_gb": 0, "disk_gb": 0.05, "gpu": "none"},
        "frozen_inputs": {path: _sha(PROJECT_ROOT / path) for path in sorted(inputs)},
        "frozen_tools": {name: {"path": path, "sha256": _sha(PROJECT_ROOT / path)} for name, path in tools.items()},
        "working_directory": str(PROJECT_ROOT),
        "command": ["/usr/bin/timeout", "1200s", PYTHON, "tools/v3/run_gse_structured_polar_teacher_feasibility_v1.py", "--spec", str(SPEC), "--run-dir", str(PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}")],
    }
    write_json(SPEC, spec)
    print(SPEC.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
