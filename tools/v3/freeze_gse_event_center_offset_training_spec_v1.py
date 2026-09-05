#!/usr/bin/env python3
"""Freeze the three-seed event-center offset training run."""

from __future__ import annotations

import hashlib
import json

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import write_json


RUN_ID = "gate3_20260828_gse_event_center_offset_training_v1r3_seed0"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_event_center_offset_training_v1r3.json"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def _sha(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if SPEC.exists(): raise RuntimeError("event-center training spec exists; overwrite is forbidden")
    parent_manifest = "results/gate0_baseline/gate0_20260811_cano_100_topology_parent_recipe_reclassification_v2r_seed0/artifacts/accepted_parent_manifest.json"
    action = "results/gate3_semantics/gate3_20260828_gse_action_set_node_training_v1_seed0"
    inputs = [
        "results/gate2_representation/gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0/artifacts/teacher_observations.jsonl",
        "results/gate3_semantics/gate3_20260826_gse_exit_token_association_corrective_v2_seed0/artifacts/pair_cache/pairs.npz",
        parent_manifest, f"{action}/scratch/action_set_cache/manifest.json",
    ]
    inputs += [f"{action}/scratch/action_set_cache/{name}.npy" for name in ("raw_tokens", "history_references", "history_mask", "normalization_mean", "normalization_scale", "global_sequence_index", "decision_target", "decision_episode_id", "partition_code")]
    inputs += [f"{action}/artifacts/models/seed{seed}/best.pt" for seed in range(3)]
    registry = json.loads((PROJECT_ROOT / parent_manifest).read_text(encoding="utf-8"))
    inputs += sorted(str(value["source_graph"]) for value in registry["parents"] if str(value["parent_id"]).endswith(tuple(f"_C{i:02d}" for i in range(1, 9))))
    tools = {
        "action_model": "src/mtare_topo/representation/gse_action_set_node.py",
        "action_cache": "src/mtare_topo/data/gse_action_set_cache.py",
        "center_teacher": "src/mtare_topo/teacher/gse_event_center_teacher.py",
        "center_head": "src/mtare_topo/representation/gse_event_center_offset.py",
        "teacher_generator": "tools/v3/generate_gse_event_center_teacher_v1.py",
        "trainer": "tools/v3/train_gse_event_center_offset_v1.py",
        "evaluator": "tools/v3/evaluate_gse_event_center_offset_ensemble_v1.py",
        "runner": "tools/v3/run_gse_event_center_offset_training_v1.py",
        "freezer": "tools/v3/freeze_gse_event_center_offset_training_spec_v1.py",
        "data_card": "configs/v3/gate3/data_cards/gse_event_center_offset_v1.json",
        "preflight": "tools/v3/preflight.py", "create_run": "tools/v3/create_run.py",
    }
    run_dir = PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}"
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "execution_phase": 3, "date": "20260828",
        "slug": "gse_event_center_offset_training_v1r3", "seed": 0, "operation": "training",
        "question": "Can frozen causal LiDAR action-set contexts regress a useful signed junction/terminal center offset on disjoint C07-C08 worlds?",
        "method": "Generate objective route-tangent targets inside the approved run, freeze each action-set detector, train only one 128-64-1 bounded offset head per seed with equal junction/terminal batch mass, and average three predictions.",
        "baseline": "Fit-only per-event median signed offset; graph baseline remains sealed sensor-pose trace commit.",
        "fallback": "None; if the ensemble fails regression gates, do not run projected graph qualification.",
        "data_card": "configs/v3/gate3/data_cards/gse_event_center_offset_v1.json", "config_path": "configs/v3/gate3/data_cards/gse_event_center_offset_v1.json",
        "user_authorization": {"status": "APPROVED", "approved_by": "user-standing-authorization", "approved_at": "2026-08-28T03:10:00+08:00", "authorized_gates": [3], "authorized_operations": ["training"], "scope": "One immutable C01-C08 three-seed event-center offset training run with deterministic teacher preprocessing; zero C09/C10/M-TARE.", "confirmation_reference": "User instructed automatic best-choice execution without routine approval prompts."},
        "hyperparameters": {"epochs": 10, "batch_per_event": 32, "learning_rate": .001, "weight_decay": .0001, "gradient_clip": 5.0, "maximum_offset_m": 12.0, "seeds": [0, 1, 2]},
        "acceptance_criteria": ["Exact 34133 target rows split 25294/8839 and 792/274 identities; finite +/-12m support.", "C07-C08 ensemble MAE improves the fit-only per-event median baseline by at least 10%; junction and terminal each improve and all three seeds improve aggregate baseline.", "Zero backbone optimizer steps and zero C09/C10/M-TARE reads; sources unchanged and complete seal."],
        "expected_counts": {"worlds": 80, "fit_rows": 25294, "selection_rows": 8839, "heads": 3, "epochs_per_seed": 10, "backbone_optimizer_steps": 0, "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0},
        "estimated_cost": {"compute": "one GPU, three frozen-backbone lightweight heads sequentially", "wall_time_hours": 1.0, "gpu_memory_gb": 2, "host_ram_gb": 4, "disk_gb": .5},
        "expected_evidence": ["Target archive/manifest, three checkpoints/histories/outputs, ensemble regression metrics, logs, source integrity, RUN_STATE and SHA-256 seal."],
        "frozen_inputs": {relative: _sha(PROJECT_ROOT / relative) for relative in inputs},
        "frozen_tools": {name: {"path": relative, "sha256": _sha(PROJECT_ROOT / relative)} for name, relative in tools.items()},
        "command": ["/usr/bin/timeout", "--signal=INT", "--kill-after=60s", "3600s", PYTHON, "tools/v3/run_gse_event_center_offset_training_v1.py", "--spec", str(SPEC), "--run-dir", str(run_dir)],
        "working_directory": str(PROJECT_ROOT),
    }
    write_json(SPEC, spec); print({"spec": str(SPEC), "frozen_inputs": len(inputs), "frozen_tools": len(tools)})
    return 0


if __name__ == "__main__": raise SystemExit(main())
