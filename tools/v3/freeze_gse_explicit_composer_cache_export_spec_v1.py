#!/usr/bin/env python3
"""Freeze Data Card/spec for the three-seed explicit Composer cache export."""

from __future__ import annotations

import hashlib
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, validate_data_card, validate_run_spec, write_json


RUN_ID = "gate3_20260829_gse_explicit_composer_cache_export_v1_seed0"
CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_explicit_composer_cache_export_v1.json"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_explicit_composer_cache_export_v1.json"
DATASET = "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
BASELINE = "results/gate3_semantics/gate3_20260829_gse_sparse_circular_relation_transport_three_seed_training_v2r5_seed0"
READINESS = "results/gate3_semantics/gate3_20260829_gse_typed_composer_readiness_v1_seed0"
SOURCE_CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_sparse_circular_relation_transport_three_seed_training_v2r5.json"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    readiness = load_json(PROJECT_ROOT / READINESS / "RUN_STATE.json")
    if readiness.get("state") != "COMPLETED" or readiness.get("error") is not None or readiness.get("overall_status") != "PASS_GSE_TYPED_COMPOSER_READINESS_V1":
        raise RuntimeError("typed Composer readiness PASS is required")
    source = load_json(SOURCE_CARD)
    card = {
        "schema_version": "v3_data_card_v1", "card_id": "gse_explicit_composer_cache_export_v1",
        "status": "APPROVED_FOR_ONE_IMMUTABLE_GSE_EXPLICIT_COMPOSER_CACHE_EXPORT_V1",
        "purpose": "Run each sealed V2R5 checkpoint once on C01-C08 and export only the explicit geometry state permitted by the typed Composers, eliminating repeated frozen-backbone inference during small-Composer training.",
        "approval": {
            "status": "APPROVED", "approved_by": "user-standing-authorization",
            "approved_at": "2026-08-29T21:10:00+08:00",
            "scope": "One immutable three-seed C01-C08 geometry-only cache export; 564,378 frozen model forwards, no optimizer/checkpoint update, C09/C10, graph, planner or M-TARE.",
            "authorized_operations": ["audit"], "authorized_gates": [3],
            "confirmation_reference": "User explicitly authorized autonomous optimal continuation without further approval prompts.",
        },
        "source": {
            "raw_sources": [
                "Sealed C01-C08 deduplicated 16x720 five-frame causal LiDAR dataset.",
                "Three sealed V2R5 best checkpoints; the failed scientific run is reused only for its frozen explicit token/transport/geometry components.",
                "Sealed C07/C08 development predictions used for exact re-inference parity before any fit cache is accepted.",
            ],
            "teacher_run": DATASET, "prediction_run": BASELINE,
            "license_or_allowed_use": "Local project-generated procedural worlds and frozen learned components.",
        },
        "worlds": dict(source["worlds"]),
        "trajectories": list(source["trajectories"]),
        "sampling": {
            "raw_frame_count": 188126, "effective_sample_count": 188126,
            "effective_structure_event_count": 188126,
            "structure_event_counts": {
                "corridor": 150964, "junction": 26608, "terminal": 7525,
                "turn": 1998, "geometry_transition": 1031,
            },
            "independent_sampling_units": "80 world-disjoint C01-C08 procedural worlds; three frozen seed outputs per observation.",
            "spatial_interval_m": 1.0, "temporal_window_frames": 5,
            "rule": "Every C01-C08 observation is forwarded exactly once per seed at frozen batch256. Export only ten explicit numeric state arrays plus global row id; no label, event output, descriptor, pose or identity field.",
        },
        "teacher": {
            "source": "No Teacher target is materialized by the exporter; it reads only range_m, valid_mask, local causal frame references and global row identity from the sealed dataset.",
            "valid_mask": "Five-frame references are sealed past/current LiDAR indices. C07/C08 cached arrays must exactly equal the pre-existing sealed V2R5 predictions field by field.",
            "planner_consistency_plan": "No graph or planner. Corrected causal Teacher enters only the later small-Composer training labels, not this cache export.",
        },
        "split": {
            "world_disjoint": True, "trajectory_disjoint": True,
            "historical_pollution_audit": "C01-C06 are future Composer fit, C07 future selection/calibration, C08 future one-shot transfer. Export performs no fitting or selection and C09/C10/M-TARE remain unread.",
            "selection": "None during export; checkpoint and all V2R5 values are already sealed.",
            "transfer": "Exact C07/C08 parity only, with no threshold or parameter update.",
        },
        "leakage_audit": dict(source["leakage_audit"]),
        "estimated_cost": {"compute": "Sequential deterministic RTX 5090 frozen inference", "wall_time_hours": 1.5, "host_ram_gb": 16, "gpu_memory_gb": 16, "disk_gb": 4},
        "cache_contract": {
            "allowed": [
                "global_sequence_index", "token_bearing_deg", "token_existence_logits",
                "token_opening_width_m", "token_vertical_profile_m", "token_geometry_uncertainty",
                "token_count_probability", "transport_row_probability",
                "transport_reveal_probability", "geometry", "observation_uncertainty",
            ],
            "forbidden": [
                "event_logits", "event_probability", "context", "hidden", "place_descriptor",
                "token_descriptor", "exit_descriptor", "pose", "world", "tng", "identity",
            ],
        },
        "failure_policy": "Any source/seal/environment/count drift, forbidden field, C07/C08 parity error, nonfinite value, resource excess or forbidden test/graph access seals FAIL; do not train Composers from a partial cache.",
    }
    report = validate_data_card(card)
    if not report.passed:
        raise RuntimeError("generated explicit cache Data Card invalid: " + "; ".join(report.errors))
    write_json(CARD, card)
    inputs = [
        str(SOURCE_CARD.relative_to(PROJECT_ROOT)),
        f"{DATASET}/RUN_STATE.json", f"{DATASET}/metrics/summary.json",
        f"{DATASET}/artifacts/evidence_sha256.txt", f"{DATASET}/artifacts/sequence_manifest.jsonl",
        f"{BASELINE}/RUN_STATE.json", f"{BASELINE}/metrics/summary.json", f"{BASELINE}/artifacts/evidence_sha256.txt",
        f"{READINESS}/RUN_STATE.json", f"{READINESS}/metrics/summary.json", f"{READINESS}/artifacts/evidence_sha256.txt",
        "configs/v3/gate2/environments/phase3_torch290_cu129_zarr2187_v1r2.json",
    ]
    for seed in range(3):
        inputs.append(f"{BASELINE}/artifacts/models/seed{seed}/best.pt")
        predictions = sorted((PROJECT_ROOT / BASELINE / f"artifacts/models/seed{seed}/development_predictions").glob("*.npz"))
        if len(predictions) != 20:
            raise RuntimeError(f"seed{seed} development prediction population drift")
        inputs.extend(str(path.relative_to(PROJECT_ROOT)) for path in predictions)
    tools = {
        "data_card": str(CARD.relative_to(PROJECT_ROOT)),
        "cache_schema": "src/mtare_topo/data/gse_explicit_composer_cache.py",
        "composer_module": "src/mtare_topo/representation/gse_typed_composers.py",
        "schema_tests": "tests/v3/unit/test_gse_explicit_composer_cache.py",
        "composer_tests": "tests/v3/unit/test_gse_typed_composers.py",
        "exporter": "tools/v3/export_gse_explicit_composer_cache_v1.py",
        "runner": "tools/v3/run_gse_explicit_composer_cache_export_v1.py",
        "freezer": "tools/v3/freeze_gse_explicit_composer_cache_export_spec_v1.py",
        "governance": "src/mtare_topo/governance.py", "preflight": "tools/v3/preflight.py",
        "create_run": "tools/v3/create_run.py",
    }
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "execution_phase": 3,
        "date": "20260829", "slug": "gse_explicit_composer_cache_export_v1", "seed": 0,
        "operation": "audit",
        "question": "Can the three frozen V2R5 seeds produce a complete geometry-only C01-C08 Composer cache with exact C07/C08 parity and no event/descriptor bypass?",
        "method": "One frozen inference pass per seed and world; typed allow-list serialization only.",
        "baseline": "Repeated frozen-backbone inference inside each Composer epoch, which is computationally redundant and weakens input provenance.",
        "fallback": "Any parity/schema/resource failure stops before Composer training; no partial seed or world cache may be used.",
        "data_card": str(CARD.relative_to(PROJECT_ROOT)), "config_path": str(CARD.relative_to(PROJECT_ROOT)),
        "user_authorization": card["approval"],
        "acceptance_criteria": [
            "Exactly 3 seeds x 80 worlds x 188,126 rows, split 142,184/21,548/24,394 per seed.",
            "Every cache contains exactly the eleven allow-listed fields and no event/context/descriptor/pose/world/TNG/identity field.",
            "All 60 C07/C08 seed-world archives reproduce sealed V2R5 arrays with exact zero error.",
            "All values finite/probabilistically valid; source seals exact; <=16GiB GPU and <=4GiB result.",
            "Zero optimizer/checkpoint/C09/C10/M-TARE/graph/planner and complete evidence seal.",
        ],
        "expected_counts": {
            "seeds": 3, "worlds_per_seed": 80, "observations_per_seed": 188126,
            "model_forward_observations": 564378, "cache_files": 240,
            "development_parity_worlds": 60, "optimizer_steps": 0, "checkpoints_created": 0,
            "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0, "graph_replays": 0,
        },
        "expected_evidence": [
            "Three seed manifests, 240 typed NPZs, exact parity/error report, sizes/runtime/GPU/RSS figure, environment, commands, logs, RUN_STATE and SHA-256 seal.",
        ],
        "estimated_cost": card["estimated_cost"],
        "frozen_inputs": {path: sha256(PROJECT_ROOT / path) for path in inputs},
        "frozen_tools": {name: {"path": path, "sha256": sha256(PROJECT_ROOT / path)} for name, path in tools.items()},
        "working_directory": str(PROJECT_ROOT),
        "command": [
            "/usr/bin/timeout", "14400s", PYTHON,
            "tools/v3/run_gse_explicit_composer_cache_export_v1.py", "--spec", str(SPEC),
            "--run-dir", str(PROJECT_ROOT / "results/gate3_semantics" / RUN_ID),
        ],
    }
    spec_report = validate_run_spec(spec)
    if not spec_report.passed:
        raise RuntimeError("generated explicit cache run spec invalid: " + "; ".join(spec_report.errors))
    write_json(SPEC, spec)
    print(CARD.relative_to(PROJECT_ROOT)); print(SPEC.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
