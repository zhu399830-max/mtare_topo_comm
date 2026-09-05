#!/usr/bin/env python3
"""Freeze the C01--C08 event-center teacher generation."""

from __future__ import annotations

import hashlib
import json

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import write_json


RUN_ID = "gate2_20260828_gse_event_center_teacher_v1_seed0"
SPEC = PROJECT_ROOT / "configs/v3/gate2/gse_event_center_teacher_v1.json"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def _sha(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if SPEC.exists():
        raise RuntimeError("event-center teacher spec exists; overwrite is forbidden")
    parent_manifest = "results/gate0_baseline/gate0_20260811_cano_100_topology_parent_recipe_reclassification_v2r_seed0/artifacts/accepted_parent_manifest.json"
    inputs = [
        "results/gate2_representation/gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0/artifacts/teacher_observations.jsonl",
        "results/gate3_semantics/gate3_20260826_gse_exit_token_association_corrective_v2_seed0/artifacts/pair_cache/pairs.npz",
        parent_manifest,
    ]
    registry = json.loads((PROJECT_ROOT / parent_manifest).read_text(encoding="utf-8"))
    inputs += sorted(
        str(value["source_graph"]) for value in registry["parents"]
        if str(value["parent_id"]).endswith(tuple(f"_C{i:02d}" for i in range(1, 9)))
    )
    if len(inputs) != 83:
        raise RuntimeError("event-center graph input count drift")
    tools = {
        "teacher_contract": "src/mtare_topo/teacher/gse_event_center_teacher.py",
        "generator": "tools/v3/generate_gse_event_center_teacher_v1.py",
        "runner": "tools/v3/run_gse_event_center_teacher_v1.py",
        "freezer": "tools/v3/freeze_gse_event_center_teacher_spec_v1.py",
        "data_card": "configs/v3/gate3/data_cards/gse_event_center_offset_v1.json",
        "preflight": "tools/v3/preflight.py", "create_run": "tools/v3/create_run.py",
    }
    run_dir = PROJECT_ROOT / f"results/gate2_representation/{RUN_ID}"
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 2, "execution_phase": 2,
        "date": "20260828", "slug": "gse_event_center_teacher_v1", "seed": 0,
        "operation": "teacher_generation",
        "question": "Can objective graph centers define a complete, traversal-local signed center-offset target for all C01-C08 decision observations?",
        "method": "Subtract sealed sensor xyz from the typed TNG junction/terminal node xyz and project the vector onto a central/one-sided tangent computed only within the current directed traversal.",
        "baseline": "Sensor-pose graph has no center target and fragments C07-C08 identities.",
        "fallback": "None; any missing identity, degenerate tangent or support violation fails generation.",
        "data_card": "configs/v3/gate3/data_cards/gse_event_center_offset_v1.json",
        "config_path": "configs/v3/gate3/data_cards/gse_event_center_offset_v1.json",
        "user_authorization": {
            "status": "APPROVED", "approved_by": "user-standing-authorization",
            "approved_at": "2026-08-28T03:10:00+08:00", "authorized_gates": [2],
            "authorized_operations": ["teacher_generation"],
            "scope": "One immutable C01-C08 signed event-center teacher generation; zero model inference/training/C09/C10/M-TARE.",
            "confirmation_reference": "User instructed automatic best-choice execution without routine approval prompts."
        },
        "acceptance_criteria": [
            "Exactly 188126 rows, 34133 valid decision rows split 25294/8839 and 792/274 fit/selection identities.",
            "All signed offsets finite and within +/-12 m; all longitudinal projection transverse residuals <=5 m.",
            "Zero inference, optimizer, C09, C10 and M-TARE reads; all 80 graph hashes and sources remain unchanged."
        ],
        "expected_counts": {"worlds": 80, "observations": 188126, "valid_rows": 34133, "fit_rows": 25294, "selection_rows": 8839, "optimizer_steps": 0, "model_inference_frames": 0, "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0},
        "estimated_cost": {"compute": "CPU-only objective teacher generation", "wall_time_hours": 0.1, "host_ram_gb": 2, "disk_gb": 0.1},
        "expected_evidence": ["Typed NPZ, count/distribution/source summary, logs, source integrity, RUN_STATE and SHA-256 seal."],
        "frozen_inputs": {relative: _sha(PROJECT_ROOT / relative) for relative in inputs},
        "frozen_tools": {name: {"path": relative, "sha256": _sha(PROJECT_ROOT / relative)} for name, relative in tools.items()},
        "command": ["/usr/bin/timeout", "--signal=INT", "--kill-after=30s", "600s", PYTHON, "tools/v3/run_gse_event_center_teacher_v1.py", "--spec", str(SPEC), "--run-dir", str(run_dir)],
        "working_directory": str(PROJECT_ROOT),
    }
    write_json(SPEC, spec)
    print({"spec": str(SPEC), "frozen_inputs": len(inputs), "frozen_tools": len(tools)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
