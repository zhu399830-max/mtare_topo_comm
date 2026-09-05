#!/usr/bin/env python3
"""Freeze the one formal three-seed JCGS training spec."""
from __future__ import annotations

import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, validate_data_card, write_json


RUN_ID = "gate3_20260829_gse_joint_cyclic_gap_simplex_three_seed_training_v1_seed0"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_joint_cyclic_gap_simplex_three_seed_training_v1.json"
DATA_CARD = "configs/v3/gate3/data_cards/gse_joint_cyclic_gap_simplex_three_seed_training_v1.json"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if SPEC.exists():
        raise RuntimeError("JCGS training spec already exists")
    card = load_json(PROJECT_ROOT / DATA_CARD)
    validation = validate_data_card(card)
    if not validation.passed:
        raise RuntimeError(f"JCGS training card invalid: {validation.errors}")
    dataset = "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
    teacher = "results/gate2_representation/gate2_20260829_gse_circular_exit_geometry_field_teacher_export_v1_seed0"
    readiness = "results/gate3_semantics/gate3_20260829_gse_joint_cyclic_gap_simplex_readiness_v3_seed0"
    baseline = "results/gate3_semantics/gate3_20260829_gse_cyclic_ordered_unimodal_slot_transport_three_seed_training_v1_seed0"
    attribution = "results/gate3_semantics/gate3_20260829_gse_coust_complex_cardinality_failure_attribution_v1_seed0"
    inputs = [
        DATA_CARD,
        f"{dataset}/RUN_STATE.json", f"{dataset}/metrics/summary.json", f"{dataset}/artifacts/shard_manifest.json", f"{dataset}/artifacts/evidence_sha256.txt",
        f"{teacher}/RUN_STATE.json", f"{teacher}/metrics/summary.json", f"{teacher}/artifacts/export/teacher_shard_manifest.jsonl", f"{teacher}/artifacts/evidence_sha256.txt",
        f"{readiness}/RUN_STATE.json", f"{readiness}/metrics/summary.json", f"{readiness}/artifacts/evidence_sha256.txt",
        f"{baseline}/RUN_STATE.json", f"{baseline}/metrics/summary.json", f"{baseline}/artifacts/evidence_sha256.txt",
        f"{attribution}/RUN_STATE.json", f"{attribution}/metrics/summary.json", f"{attribution}/artifacts/evidence_sha256.txt",
        "configs/v3/gate2/environments/phase3_torch290_cu129_zarr2187_v1r2.json",
    ]
    tools = {
        "gap_simplex": "src/mtare_topo/representation/gse_joint_cyclic_gap_simplex.py",
        "parent_slot_transport": "src/mtare_topo/representation/gse_cardinality_conditioned_circular_slot_transport.py",
        "backbone": "src/mtare_topo/representation/gse_circular_peak_geometry_model.py",
        "circular_layers": "src/mtare_topo/representation/phase3_structural_semantics.py",
        "training_core": "tools/v3/train_gse_circular_peak_geometry_v1.py",
        "trainer": "tools/v3/train_gse_joint_cyclic_gap_simplex_v1.py",
        "baseline_evaluator": "tools/v3/evaluate_gse_cardinality_conditioned_circular_slot_transport_selection_v1.py",
        "alignment_evaluator": "tools/v3/evaluate_gse_cyclic_ordered_unimodal_slot_transport_selection_v1.py",
        "evaluator": "tools/v3/evaluate_gse_joint_cyclic_gap_simplex_selection_v1.py",
        "runner": "tools/v3/run_gse_joint_cyclic_gap_simplex_three_seed_training_v1.py",
        "freezer": "tools/v3/freeze_gse_joint_cyclic_gap_simplex_three_seed_training_spec_v1.py",
        "model_tests": "tests/v3/unit/test_gse_joint_cyclic_gap_simplex.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py",
        "create_run": "tools/v3/create_run.py",
    }
    spec = {
        "schema_version": "v3_run_spec_v1",
        "gate": 3,
        "execution_phase": 3,
        "date": "20260829",
        "slug": "gse_joint_cyclic_gap_simplex_three_seed_training_v1",
        "seed": 0,
        "operation": "training",
        "data_card": DATA_CARD,
        "question": "Can a joint circular phase and closed positive gap simplex recover strict K>=3 exit geometry and transferable safe confidence enough to establish GSE-Graph structural semantics?",
        "method": "Three from-scratch 789650-parameter seeds, ten epochs, C07 total-loss checkpointing, one joint phase distribution plus K positive closed gaps, cyclic-start-invariant supervision, periodically bound exit geometry, and one C07 refusal threshold.",
        "baseline": "Frozen COUST overall exact2 0.4156/0.3870, K3 0.00240/0.00179 and K4 0/0.00433 on C07/C08, with failed confidence transfer.",
        "fallback": "Any safety, K>=3 recovery, registered COUST gain, geometry or system failure seals FAIL before graph; no oracle count, wider matching, retry, seed selection or planner compensation.",
        "user_authorization": card["approval"],
        "acceptance_criteria": [
            "Exactly three from-scratch seeds times ten epochs and 11370 steps; 789650 parameters; no independent slot localizers or loss weights.",
            "One C07-selected refusal at <=2 degrees gives exit precision >=0.995 and recall >=0.50 on C07 and C08.",
            "Raw count accuracy >=0.80, deployed action macro-F1 >=0.80, overall raw and safe exact-set >=0.50 on both.",
            "K3 raw/safe exact >=0.40/0.30 and K4 >=0.20/0.10 on both.",
            "Overall, K3 and K4 exact2 each gain >=0.05 absolute over frozen COUST on both.",
            "Bearing <=1 degree, exit width <=3m, profile <=1m and global geometry bounds including slope <=2 degrees.",
            "C08 checkpoint observations zero; C09/C10/M-TARE/graph/planner zero; frozen inputs unchanged."
        ],
        "expected_counts": {
            "fit_worlds": 60, "c07_worlds": 10, "c08_worlds": 10,
            "fit_observations": 142184, "c07_observations": 21548, "c08_observations": 24394,
            "fit_exits": 299872, "c07_exits": 45504, "c08_exits": 51537,
            "fit_cardinality": {"1": 5551, "2": 117018, "3": 18175, "4": 1440},
            "parameters": 789650, "seeds": 3, "epochs_per_seed": 10,
            "optimizer_steps_per_seed": 11370, "optimizer_steps": 34110,
            "c08_checkpoint_observations": 0, "c09_worlds_read": 0, "c10_worlds_read": 0,
            "mtare_worlds_read": 0, "graph_replays": 0
        },
        "expected_evidence": [
            "Three checkpoints, histories, seed summaries and compact joint-set/count/phase/gap/global-geometry predictions.",
            "C07 refusal selection and one C08 transfer with strict continuous-set, K-stratified, COUST-gain, geometry and per-world metrics.",
            "PNG/PDF/SVG/source, environment, raw logs, RUN_STATE and SHA-256 seal."
        ],
        "estimated_cost": {"compute": "Three sequential CUDA trainings and frozen development evaluation", "wall_time_hours": 2, "host_ram_gb": 16, "gpu_memory_gb": 16, "disk_gb": 3, "gpu": "one NVIDIA GeForce RTX 5090 D"},
        "frozen_inputs": {path: sha256(PROJECT_ROOT / path) for path in sorted(inputs)},
        "frozen_tools": {name: {"path": path, "sha256": sha256(PROJECT_ROOT / path)} for name, path in tools.items()},
        "working_directory": str(PROJECT_ROOT),
        "command": ["/usr/bin/timeout", "86400s", PYTHON, "tools/v3/run_gse_joint_cyclic_gap_simplex_three_seed_training_v1.py", "--spec", str(SPEC), "--run-dir", str(PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}")]
    }
    write_json(SPEC, spec)
    print(SPEC.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
