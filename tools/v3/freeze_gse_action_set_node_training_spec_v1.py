#!/usr/bin/env python3
"""Freeze the action-set node Data Card and one-run specification."""

from __future__ import annotations

import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


RUN_ID = "gate3_20260828_gse_action_set_node_training_v1_seed0"
CARD_PATH = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_action_set_node_training_v1.json"
SPEC_PATH = PROJECT_ROOT / "configs/v3/gate3/gse_action_set_node_training_v1.json"
SOURCE_CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_causal_episode_training_v1r.json"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"
TEACHER = "results/gate2_representation/gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0"
TOKENS = "results/gate3_semantics/gate3_20260826_gse_exit_token_association_corrective_v2_seed0"


def _sha(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if CARD_PATH.exists() or SPEC_PATH.exists():
        raise RuntimeError("action-set Data Card/spec already exists; overwrite is forbidden")
    source = load_json(SOURCE_CARD)
    approval = {
        "status": "APPROVED", "approved_by": "user",
        "approved_at": "2026-08-28T00:40:00+08:00",
        "authorized_operations": ["training"], "authorized_gates": [3],
        "scope": "One immutable C01-C08 three-seed action-set decision-node capacity run using only frozen learned exit tokens and five past/current observations; zero C09/C10/M-TARE access.",
        "confirmation_reference": "User explicitly instructed automatic selection of the best in-scope option without further routine approval prompts.",
    }
    card = {
        "schema_version": "v3_data_card_v1",
        "card_id": "gse_action_set_node_training_v1",
        "status": "APPROVED_FOR_ONE_IMMUTABLE_GSE_ACTION_SET_NODE_TRAINING_V1",
        "purpose": "Test whether learned exit/action token sets directly support safe junction/terminal node proposals after categorical event gates failed.",
        "approval": approval,
        "worlds": source["worlds"],
        "trajectories": source["trajectories"],
        "sampling": {
            "independent_units": "80 topology-parent worlds and 16076 observed directed traversals; contiguous junction/terminal Teacher episodes are the positive bags.",
            "raw_unique_lidar_frames": 252430,
            "effective_causal_observations": 188126,
            "fit_observations": 142184,
            "selection_observations": 45942,
            "spatial_interval_m": 1.0,
            "temporal_context": "Exactly the current and up to four earlier observations from the same traversal, left padded with a validity mask.",
            "fit_event_rows": {"junction": 19743, "terminal": 5551, "other_negative": 116890},
            "selection_event_rows": {"junction": 6865, "terminal": 1974, "other_negative": 37103},
            "fit_event_episodes": {"junction": 2542, "terminal": 740},
            "selection_event_episodes": {"junction": 882, "terminal": 254},
            "raw_frame_count": 252430,
            "effective_sample_count": 188126,
            "effective_structure_event_count": 34127,
            "rule": "At 1 m traversal-arc spacing, each sample uses the current and at most four prior observations from the same directed traversal; no future or cross-traversal row is referenced.",
            "structure_event_counts": {"junction_rows": 26608, "terminal_rows": 7525, "decision_episodes": 4418},
        },
        "split": {
            "fit": "C01-C06: 60 worlds, 142184 observations, 3282 decision episodes; fits token normalization and detector weights.",
            "checkpoint_and_threshold_selection": "C07-C08: 20 worlds, 45942 observations, 1136 decision episodes; selects epoch and one ensemble threshold.",
            "future_validation": "C09 is unread and may be used once only after this capacity run passes.",
            "strict_test": "C10 and all M-TARE benchmark worlds remain unread.",
            "world_disjoint": True, "trajectory_disjoint": True,
            "historical_pollution_audit": "Frozen exit-token backbones used only their predeclared development split. This head fits C01-C06 and selects on C07-C08; prior C09 diagnostics provide zero inputs, labels, normalization, checkpoints or thresholds.",
        },
        "source": {
            "teacher": f"{TEACHER}/artifacts/teacher_observations.jsonl",
            "partition": f"{TOKENS}/artifacts/pair_cache/pairs.npz",
            "token_outputs": [f"{TOKENS}/artifacts/models/seed{seed}/frozen_exit_token_outputs.npz" for seed in (0, 1, 2)],
            "student_input": "For each of three frozen perception seeds: six unordered exit tokens containing confidence, robot-frame heading, opening width, four-value vertical profile and 32D learned descriptor; plus a five-observation past-only mask.",
            "teacher_only_fields": "junction/terminal class, contiguous episode membership and objective identity are used only for loss/evaluation, never detector inputs.",
            "raw_sources": [
                "Sealed C01-C08 deduplicated 16x720 organized LiDAR dataset and frozen three-seed learned exit-token outputs.",
                "Sealed corrected causal Teacher V1R and strict C01-C06/C07-C08 partition manifest.",
                "No C09/C10/strict/M-TARE sensor, Teacher, model, graph or planner artifact is read."
            ],
            "license_or_allowed_use": "Local research use of project procedural Cano worlds, derived LiDAR, objective TNG/spline/mesh Teacher and locally trained checkpoints with full provenance.",
        },
        "teacher": {
            "source": "Sealed corrected causal Teacher V1R; junction and terminal are action-changing decision events.",
            "negative_definition": "corridor, turn and geometry-transition rows are explicit no-decision-node negatives for this factorized graph.",
            "valid_mask": "Every positive row belongs to exactly one same-traversal contiguous decision episode; all negative rows use episode_id=-1.",
            "planner_consistency_plan": "A passing detector may propose one junction/terminal node per contiguous response; edge creation remains forbidden until a recorded traversal verifies it.",
        },
        "leakage_audit": {
            "world_isolation": "C01-C06 fit and C07-C08 selection are disjoint; C09/C10/M-TARE are absent.",
            "future_frames_excluded": True,
            "identity_excluded_at_inference": True,
            "event_label_excluded_at_inference": True,
            "pose_and_objective_geometry_excluded": True,
            "normalization": "Only width/profile/descriptor dimensions are standardized using C01-C06 tokens; confidence and heading remain physical, C07-C10 contribute zero statistics.",
            "test_excluded_from_supervised_training": True,
            "test_excluded_from_checkpoint_selection": True,
            "test_excluded_from_threshold_calibration": True,
            "test_excluded_from_ssl": True,
            "test_excluded_from_normalization": True,
            "test_excluded_from_teacher_calibration": True,
            "test_excluded_from_augmentation_tuning": True,
        },
        "methods": {
            "main": "A 237443-parameter permutation-invariant token MLP pools six exits per seed, mean/std fuses three frozen perception seeds, and a 5-observation causal GRU predicts no-node/junction/terminal. Episode MIL requires one correct confirmation per event.",
            "baseline": "The sealed 12-frame categorical detector reached C09 false-trigger 2.464%; the sealed five-frame decision-mass fallback found no safe nonvacuous C07-C08 threshold.",
            "fallback": "None inside the run. Scientific failure stops this node representation; thresholds, graph logic and safety gates are not relaxed.",
        },
        "metrics_and_pre_registered_gates": {
            "aggregate": "C07-C08 ensemble precision >=0.995, false trigger fraction <=0.005 and episode recall >=0.25.",
            "per_event": "junction and terminal each precision >=0.99 and recall >=0.25.",
            "diversity": "All ten topology families must contain at least one correct trigger.",
            "diagnostics": "Report all three seeds at the frozen ensemble threshold, per-family results and junction/terminal unique-identity coverage; diagnostics do not replace gates.",
        },
        "estimated_cost": {"compute": "One RTX 5090 D; three lightweight heads trained sequentially", "wall_time_hours": 2.0, "host_ram_gb": 4, "gpu_memory_gb": 4, "scratch_disk_gb": 1.0, "retained_disk_gb": 0.5, "disk_gb": 1.5},
        "retention": "Retain three checkpoints/histories/selection outputs, ensemble output and metrics, fit-only normalization, cache manifest, deletion audit, logs, RUN_STATE and SHA-256 seal. Delete only the regenerable raw-token cache after evaluation.",
        "failure_policy": "Any identity/split/source/environment drift, future/test read, nonfinite value, resource excess, program error or unmet gate seals FAIL. No retry or gate relaxation.",
    }
    write_json(CARD_PATH, card)

    inputs = [
        f"{TEACHER}/RUN_STATE.json", f"{TEACHER}/metrics/summary.json",
        f"{TEACHER}/artifacts/evidence_sha256.txt", f"{TEACHER}/artifacts/teacher_observations.jsonl",
        f"{TOKENS}/RUN_STATE.json", f"{TOKENS}/metrics/summary.json",
        f"{TOKENS}/artifacts/evidence_sha256.txt", f"{TOKENS}/artifacts/pair_cache/pairs.npz",
    ] + [f"{TOKENS}/artifacts/models/seed{seed}/frozen_exit_token_outputs.npz" for seed in (0, 1, 2)]
    tools = {
        "data_card": str(CARD_PATH.relative_to(PROJECT_ROOT)),
        "action_model": "src/mtare_topo/representation/gse_action_set_node.py",
        "action_cache": "src/mtare_topo/data/gse_action_set_cache.py",
        "episode_sampler": "src/mtare_topo/data/gse_causal_episode_sampler.py",
        "episode_metrics": "src/mtare_topo/evaluation/gse_causal_episode_metrics.py",
        "cache_builder": "tools/v3/build_gse_action_set_node_cache_v1.py",
        "trainer": "tools/v3/train_gse_action_set_node_v1.py",
        "evaluator": "tools/v3/evaluate_gse_action_set_node_selection_v1.py",
        "runner": "tools/v3/run_gse_action_set_node_training_v1.py",
        "freezer": "tools/v3/freeze_gse_action_set_node_training_spec_v1.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py", "create_run": "tools/v3/create_run.py",
    }
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "execution_phase": 3,
        "date": "20260828", "slug": "gse_action_set_node_training_v1", "seed": 0,
        "operation": "training",
        "question": "Can frozen learned exit-token sets and five-observation causal stability safely generate junction/terminal decision nodes on C01-C08?",
        "method": card["methods"]["main"], "baseline": card["methods"]["baseline"], "fallback": card["methods"]["fallback"],
        "data_card": str(CARD_PATH.relative_to(PROJECT_ROOT)), "config_path": str(CARD_PATH.relative_to(PROJECT_ROOT)),
        "user_authorization": approval,
        "hyperparameters": {"epochs": 6, "training_batch_size": 64, "evaluation_batch_size": 256, "learning_rate": 0.0003, "weight_decay": 0.0001, "gradient_clip": 5.0, "history_observations": 5, "threshold_grid": "0.000..1.000 inclusive at 0.001", "seeds": [0, 1, 2]},
        "expected_counts": {"worlds": 80, "fit_observations": 142184, "selection_observations": 45942, "fit_decision_episodes": 3282, "selection_decision_episodes": 1136, "optimizer_steps": 39996, "backbone_optimizer_steps": 0, "trainable_parameters_per_seed": 237443, "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0},
        "acceptance_criteria": list(card["metrics_and_pre_registered_gates"].values()),
        "expected_evidence": [card["retention"]],
        "estimated_cost": card["estimated_cost"],
        "frozen_inputs": {relative: _sha(PROJECT_ROOT / relative) for relative in inputs},
        "frozen_tools": {name: {"path": relative, "sha256": _sha(PROJECT_ROOT / relative)} for name, relative in tools.items()},
        "command": ["/usr/bin/systemd-inhibit", "--what=sleep:shutdown", "--why=GSE action-set node capacity", "--mode=block", "/usr/bin/timeout", "--signal=INT", "--kill-after=300s", "7500s", PYTHON, "tools/v3/run_gse_action_set_node_training_v1.py", "--spec", str(SPEC_PATH), "--run-dir", str(PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}")],
        "working_directory": str(PROJECT_ROOT),
    }
    write_json(SPEC_PATH, spec)
    print({"data_card": str(CARD_PATH), "spec": str(SPEC_PATH), "frozen_inputs": len(inputs), "frozen_tools": len(tools)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
