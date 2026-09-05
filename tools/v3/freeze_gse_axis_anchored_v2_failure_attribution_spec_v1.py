#!/usr/bin/env python3
"""Freeze the one formal Axis-Anchored V2 failure-attribution spec."""
from __future__ import annotations

import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, validate_data_card, write_json


RUN_ID = "gate3_20260829_gse_axis_anchored_v2_failure_attribution_v1_seed0"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_axis_anchored_v2_failure_attribution_v1.json"
DATA_CARD = "configs/v3/gate3/data_cards/gse_axis_anchored_v2_failure_attribution_v1.json"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if SPEC.exists():
        raise RuntimeError("V2 attribution spec already exists")
    card = load_json(PROJECT_ROOT / DATA_CARD); validation = validate_data_card(card)
    if not validation.passed:
        raise RuntimeError(f"V2 attribution card invalid: {validation.errors}")
    dataset = "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
    training = "results/gate3_semantics/gate3_20260829_gse_axis_anchored_event_relation_three_seed_training_v1_seed0"
    slot = "results/gate3_semantics/gate3_20260829_gse_cardinality_conditioned_circular_slot_transport_three_seed_training_v1_seed0"
    inputs = [
        DATA_CARD, f"{dataset}/RUN_STATE.json", f"{dataset}/metrics/summary.json", f"{dataset}/artifacts/evidence_sha256.txt",
        f"{dataset}/artifacts/sequence_manifest.jsonl", f"{dataset}/artifacts/association_pairs_numeric.jsonl",
        f"{training}/RUN_STATE.json", f"{training}/metrics/summary.json", f"{training}/metrics/selection/summary.json", f"{training}/artifacts/evidence_sha256.txt",
        f"{slot}/RUN_STATE.json", f"{slot}/metrics/summary.json", f"{slot}/metrics/selection/summary.json", f"{slot}/artifacts/evidence_sha256.txt",
        "configs/v3/gate2/environments/phase3_torch290_cu129_zarr2187_v1r2.json",
    ]
    for seed in range(3):
        inputs.extend(str(path.relative_to(PROJECT_ROOT)) for path in sorted((PROJECT_ROOT / training / f"artifacts/models/seed{seed}/development_predictions").glob("*.npz")))
    tools = {
        "current_model": "src/mtare_topo/representation/gse_axis_anchored_event_relation.py",
        "predecessor_model": "src/mtare_topo/representation/gse_circular_peak_geometry_model.py",
        "training_loader": "tools/v3/train_gse_axis_anchored_event_relation_v1.py",
        "selection_loader": "tools/v3/evaluate_gse_axis_anchored_event_relation_selection_v1.py",
        "executor": "tools/v3/execute_gse_axis_anchored_v2_failure_attribution_v1.py",
        "runner": "tools/v3/run_gse_axis_anchored_v2_failure_attribution_v1.py",
        "freezer": "tools/v3/freeze_gse_axis_anchored_v2_failure_attribution_spec_v1.py",
        "tests": "tests/v3/unit/test_gse_axis_anchored_v2_failure_attribution.py",
        "governance": "src/mtare_topo/governance.py", "preflight": "tools/v3/preflight.py", "create_run": "tools/v3/create_run.py",
    }
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "execution_phase": 3, "date": "20260829",
        "slug": "gse_axis_anchored_v2_failure_attribution_v1", "seed": 0, "operation": "audit", "data_card": DATA_CARD,
        "question": "Are missing five-frame directional fusion and dense extreme-imbalance relation learning the concrete causes of Axis-Anchored V1 failure, and does frozen local ranking justify soft peaks or require object-level transport?",
        "method": "Reproduce the formal C07/C08 ensemble; compare source contracts and sealed predecessor axis metrics; quantify relation rates, effective weights, AP, safe PR envelope, positive-slice AUC and oracle-count circular top-k localization at fixed radii.",
        "baseline": "Axis-Anchored V1 C07/C08 axis 82.7165/83.0120 degrees and all safe relation recalls zero; sealed Slot predecessor axis 5.4982/5.7835 degrees.",
        "fallback": "If local relation ranking is weak, require object-centric transport; if it is strong within fixed circular support, allow sparse soft-peak transport readiness. Never continue dense BCE or enter graph.",
        "user_authorization": card["approval"],
        "acceptance_criteria": [
            "Exactly 10 C07 plus 10 C08 worlds, 21548/24394 observations, 66752/77490 valid relation pairs and exact channel-positive counts.",
            "Reproduce formal event and axis ensemble metrics to 1e-12.",
            "Machine-confirm current last-frame axis path and predecessor five-frame directional-temporal path, with at least 70 degree error gap and predecessor at most 10 degrees.",
            "Report exact AP, threshold-0.5 precision/recall, maximum safe recall, positive-slice AUC and oracle-count localization for all three channels on both splits.",
            "Zero optimizer, new inference, C09/C10/M-TARE/graph/planner; all frozen inputs unchanged."
        ],
        "expected_counts": {"c07_worlds": 10, "c08_worlds": 10, "c07_observations": 21548, "c08_observations": 24394, "c07_valid_relation_pairs": 66752, "c08_valid_relation_pairs": 77490, "seeds": 3, "optimizer_steps": 0, "new_model_inference_observations": 0, "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0, "graph_replays": 0},
        "expected_evidence": ["Per-channel ranking/localization CSV and mechanism summary.", "PNG/PDF/SVG/source figure showing axis, imbalance, AP and localization.", "Environment, raw log, RUN_STATE and SHA-256 seal."],
        "estimated_cost": {"compute": "CPU-only deterministic frozen-output attribution", "wall_time_hours": 0.15, "host_ram_gb": 8, "gpu_memory_gb": 0, "disk_gb": 0.1},
        "frozen_inputs": {path: sha256(PROJECT_ROOT / path) for path in sorted(inputs)},
        "frozen_tools": {name: {"path": path, "sha256": sha256(PROJECT_ROOT / path)} for name, path in tools.items()},
        "working_directory": str(PROJECT_ROOT),
        "command": ["/usr/bin/timeout", "7200s", PYTHON, "tools/v3/run_gse_axis_anchored_v2_failure_attribution_v1.py", "--spec", str(SPEC), "--run-dir", str(PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}")],
    }
    write_json(SPEC, spec); print(SPEC.relative_to(PROJECT_ROOT)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
