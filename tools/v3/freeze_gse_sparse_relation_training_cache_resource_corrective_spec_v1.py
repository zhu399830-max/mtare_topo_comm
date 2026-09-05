#!/usr/bin/env python3
from __future__ import annotations

import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, validate_data_card, write_json


RUN_ID = "gate3_20260829_gse_sparse_relation_training_cache_resource_corrective_v1_seed0"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_sparse_relation_training_cache_resource_corrective_v1.json"
CARD = "configs/v3/gate3/data_cards/gse_sparse_relation_training_cache_resource_corrective_v1.json"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if SPEC.exists():
        raise RuntimeError("training cache resource corrective spec exists")
    card = load_json(PROJECT_ROOT / CARD)
    validation = validate_data_card(card)
    if not validation.passed:
        raise RuntimeError(str(validation.errors))
    dataset = "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
    slot = "results/gate3_semantics/gate3_20260829_gse_cardinality_conditioned_circular_slot_transport_three_seed_training_v1_seed0"
    v2r3 = "results/gate3_semantics/gate3_20260829_gse_sparse_circular_relation_transport_three_seed_training_v2r3_seed0"
    v2r4 = "results/gate3_semantics/gate3_20260829_gse_sparse_circular_relation_transport_three_seed_training_v2r4_seed0"
    validation_resource = "results/gate3_semantics/gate3_20260829_gse_sparse_relation_evaluation_cache_resource_corrective_v1r_seed0"
    inputs = [
        CARD,
        f"{dataset}/RUN_STATE.json", f"{dataset}/metrics/summary.json", f"{dataset}/artifacts/evidence_sha256.txt", f"{dataset}/artifacts/sequence_manifest.jsonl",
        f"{slot}/artifacts/models/seed0/best.pt",
        *[item for run in (v2r3, v2r4, validation_resource) for item in (f"{run}/RUN_STATE.json", f"{run}/metrics/summary.json", f"{run}/artifacts/evidence_sha256.txt")],
        f"{v2r3}/logs/01_seed0_training.log", f"{v2r3}/artifacts/models/seed0/best.pt",
        f"{v2r4}/logs/01_seed0_training.log",
        "configs/v3/gate2/environments/phase3_torch290_cu129_zarr2187_v1r2.json",
    ]
    tools = {
        "model": "src/mtare_topo/representation/gse_sparse_circular_relation_transport.py",
        "trainer": "tools/v3/train_gse_sparse_circular_relation_transport_v2.py",
        "base_loader": "tools/v3/train_gse_axis_anchored_event_relation_v1.py",
        "probe": "tools/v3/probe_gse_sparse_relation_training_cache_resource_v1.py",
        "executor": "tools/v3/execute_gse_sparse_relation_training_cache_resource_corrective_v1.py",
        "runner": "tools/v3/run_gse_sparse_relation_training_cache_resource_corrective_v1.py",
        "freezer": "tools/v3/freeze_gse_sparse_relation_training_cache_resource_corrective_spec_v1.py",
        "evaluation_tests": "tests/v3/unit/test_evaluate_gse_sparse_circular_relation_transport_v2.py",
        "training_tests": "tests/v3/unit/test_train_gse_sparse_circular_relation_transport_v2.py",
        "model_tests": "tests/v3/unit/test_gse_sparse_circular_relation_transport.py",
        "count_tests": "tests/v3/unit/test_gse_sparse_relation_cardinality_corrective.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py",
        "create_run": "tools/v3/create_run.py",
    }
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "execution_phase": 3, "date": "20260829",
        "slug": "gse_sparse_relation_training_cache_resource_corrective_v1", "seed": 0,
        "operation": "training", "data_card": CARD,
        "question": "Can per-training-world CUDA cache release preserve the exact seed0 epoch0 state and metrics while keeping all 60 fit worlds below16GiB?",
        "method": "Execute the exact frozen seed0 epoch0 schedule over all60 C01-C06 worlds at batch128 with unused CUDA cache release before/after every independently loaded world, then full C07 at unchanged batch256. Compare all77 model tensors and all epoch0 train/C07 metrics to sealed V2R3 and record allocated/reserved/process memory per training world.",
        "baseline": "Sealed V2R3 seed0 epoch0 without cache lifecycle correction; V2R4 is the immutable resource-stop evidence.",
        "fallback": "If exact state/metric parity or <=16GiB fails, stop and qualify process isolation. Do not raise the limit, reduce batch, reuse partial checkpoints or change science.",
        "user_authorization": card["approval"],
        "acceptance_criteria": [
            "Exactly60 fit worlds,142184 core rows,6347 descriptor rows and1223 optimizer steps; full10-world/21548-row C07 validation.",
            "All77 model tensors exact and every train/C07 numeric metric within3e-6 of sealed V2R3 epoch0.",
            "Every training world allocated/reserved/nvidia process memory <=16GiB with at least120 cache boundary calls.",
            "Zero selectable checkpoint, C08-C10/M-TARE/graph/planner; all inputs/tools unchanged and evidence complete.",
        ],
        "expected_counts": {"fit_worlds": 60, "fit_observations": 142184, "descriptor_rows": 6347, "c07_worlds": 10, "c07_observations": 21548, "optimizer_steps": 1223, "core_optimizer_steps": 1137, "descriptor_optimizer_steps": 86, "selection_checkpoints_written": 0, "c08_worlds_read": 0, "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0, "graph_replays": 0},
        "expected_evidence": ["Parity state and all-tensor exact comparison.", "Per-world training memory and epoch0 metric parity.", "PNG/PDF/SVG/source, tests, logs, RUN_STATE and SHA-256 seal."],
        "estimated_cost": {"compute": "One RTX5090 seed0 epoch plus full C07", "wall_time_hours": 0.25, "host_ram_gb": 8, "gpu_memory_gb": 16, "disk_gb": 0.5},
        "frozen_inputs": {path: sha(PROJECT_ROOT / path) for path in sorted(inputs)},
        "frozen_tools": {name: {"path": path, "sha256": sha(PROJECT_ROOT / path)} for name, path in tools.items()},
        "working_directory": str(PROJECT_ROOT),
        "command": ["/usr/bin/timeout", "2400s", PYTHON, "tools/v3/run_gse_sparse_relation_training_cache_resource_corrective_v1.py", "--spec", str(SPEC), "--run-dir", str(PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}")],
    }
    write_json(SPEC, spec)
    print(SPEC.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
