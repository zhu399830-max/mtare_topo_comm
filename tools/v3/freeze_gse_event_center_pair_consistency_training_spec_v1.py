#!/usr/bin/env python3
"""Freeze the one three-seed cross-traversal event-center training run."""

from __future__ import annotations

import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import write_json


RUN_ID = "gate3_20260828_gse_event_center_vector_training_v2_seed0"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_event_center_vector_training_v2.json"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def _sha(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if SPEC.exists(): raise RuntimeError("pair-consistency training spec exists; overwrite is forbidden")
    action = "results/gate3_semantics/gate3_20260828_gse_action_set_node_training_v1_seed0"
    prior = "results/gate3_semantics/gate3_20260828_gse_event_center_offset_training_v1r3_seed0"
    graph = "results/gate3_semantics/gate3_20260828_gse_projected_trace_commit_capacity_v1_seed0"
    scalar_corrective = "results/gate3_semantics/gate3_20260828_gse_event_center_pair_consistency_training_v1r3_seed0"
    inputs = [
        f"{prior}/artifacts/teacher/event_center_teacher.npz", f"{prior}/metrics/summary.json",
        f"{prior}/artifacts/evidence_sha256.txt", f"{graph}/artifacts/projection/event_center_projection.npz",
        f"{graph}/artifacts/evidence_sha256.txt", f"{action}/scratch/action_set_cache/manifest.json",
    ]
    inputs += [f"{action}/scratch/action_set_cache/{name}.npy" for name in (
        "raw_tokens", "history_references", "history_mask", "normalization_mean", "normalization_scale",
        "global_sequence_index", "traversal_id", "sequence_index", "partition_code",
    )]
    inputs += [f"{action}/artifacts/models/seed{seed}/best.pt" for seed in range(3)]
    inputs += [f"{prior}/artifacts/models/seed{seed}/best.pt" for seed in range(3)]
    inputs += [f"{scalar_corrective}/artifacts/models/seed{seed}/best.pt" for seed in range(3)]
    inputs += [f"{scalar_corrective}/metrics/summary.json", f"{scalar_corrective}/artifacts/evidence_sha256.txt"]
    tools = {
        "action_model": "src/mtare_topo/representation/gse_action_set_node.py",
        "action_cache": "src/mtare_topo/data/gse_action_set_cache.py",
        "center_head": "src/mtare_topo/representation/gse_event_center_offset.py",
        "center_teacher": "src/mtare_topo/teacher/gse_event_center_teacher.py",
        "trainer_helper": "tools/v3/train_gse_event_center_pair_consistency_v1.py",
        "trainer": "tools/v3/train_gse_event_center_vector_v1.py",
        "evaluator": "tools/v3/evaluate_gse_event_center_vector_ensemble_v1.py",
        "runner": "tools/v3/run_gse_event_center_pair_consistency_training_v1.py",
        "freezer": "tools/v3/freeze_gse_event_center_pair_consistency_training_spec_v1.py",
        "data_card": "configs/v3/gate3/data_cards/gse_event_center_vector_v1.json",
        "preflight": "tools/v3/preflight.py", "create_run": "tools/v3/create_run.py",
    }
    run_dir = PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}"
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "execution_phase": 3, "date": "20260828",
        "slug": "gse_event_center_vector_training_v2", "seed": 0, "operation": "training",
        "question": "Can a route-local forward/lateral/up event-center vector remove scalar tangent residual and pass the frozen cross-view association gates?",
        "method": "Initialize from each dual-batch scalar corrective, split the vector output, freeze hidden plus forward output exactly, and train only zero-initialized lateral/up with separate row-balanced direct and identity-balanced pair batches at fixed 1:1 loss.",
        "baseline": "Sealed independent-row V1R3 ensemble: validation MAE 3.520097 m, identity-macro relative vector error 4.589575 m and within-4m fraction 0.523952.",
        "fallback": "None inside the run. Scientific failure stops local event-center regression and no graph replay is launched.",
        "data_card": "configs/v3/gate3/data_cards/gse_event_center_vector_v1.json", "config_path": "configs/v3/gate3/data_cards/gse_event_center_vector_v1.json",
        "user_authorization": {"status": "APPROVED", "approved_by": "user-standing-authorization", "approved_at": "2026-08-28T07:40:00+08:00", "authorized_gates": [3], "authorized_operations": ["training"], "scope": "One immutable C01-C08 frozen-forward local 3D corrective V2; zero C09/C10/M-TARE.", "confirmation_reference": "User instructed automatic best-choice execution without routine approval prompts."},
        "hyperparameters": {"epochs": 10, "identities_per_event": 16, "learning_rate": .001, "weight_decay": .0001, "gradient_clip": 5.0, "direct_to_relative_loss_weight": "1:1", "local_vector_scale_m": [12.0, 5.0, 5.0], "seeds": [0, 1, 2]},
        "acceptance_criteria": [
            "Exact 25294/8839 rows, 792/274 identities, 417435/144533 cross-traversal pairs and 782/272 multi-traversal identities.",
            "C07-C08 ensemble relative error improves scalar V1R3 by >=10%, within-4m fraction improves by >=0.10, forward MAE does not regress and global-center Euclidean MAE improves.",
            "All three seeds improve relative vector error; zero backbone steps and zero C09/C10/M-TARE reads; sources unchanged and complete seal."
        ],
        "expected_counts": {"worlds": 80, "fit_rows": 25294, "selection_rows": 8839, "fit_cross_traversal_pairs": 417435, "selection_cross_traversal_pairs": 144533, "heads": 3, "epochs_per_seed": 10, "optimizer_steps_per_seed": 3960, "backbone_optimizer_steps": 0, "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0},
        "estimated_cost": {"compute": "one GPU, three frozen-backbone small heads sequentially", "wall_time_hours": 1.0, "gpu_memory_gb": 2, "host_ram_gb": 4, "disk_gb": .5},
        "expected_evidence": ["Three checkpoints/histories, per-seed and ensemble point/cross-view metrics, logs, source integrity, RUN_STATE and SHA-256 seal."],
        "frozen_inputs": {relative: _sha(PROJECT_ROOT / relative) for relative in inputs},
        "frozen_tools": {name: {"path": relative, "sha256": _sha(PROJECT_ROOT / relative)} for name, relative in tools.items()},
        "command": ["/usr/bin/timeout", "--signal=INT", "--kill-after=60s", "3600s", PYTHON, "tools/v3/run_gse_event_center_pair_consistency_training_v1.py", "--spec", str(SPEC), "--run-dir", str(run_dir)],
        "working_directory": str(PROJECT_ROOT),
    }
    write_json(SPEC, spec); print({"spec": str(SPEC), "frozen_inputs": len(inputs), "frozen_tools": len(tools)}); return 0


if __name__ == "__main__": raise SystemExit(main())
