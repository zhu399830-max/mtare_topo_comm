#!/usr/bin/env python3
"""Freeze the explicit causal geometry-delta multitask Data Card and run spec."""

from __future__ import annotations

from copy import deepcopy
import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


RUN_ID = "gate3_20260827_gse_causal_geometry_delta_multitask_training_v1_seed0"
CARD_PATH = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_causal_geometry_delta_multitask_training_v1.json"
SPEC_PATH = PROJECT_ROOT / "configs/v3/gate3/gse_causal_geometry_delta_multitask_training_v1.json"
SOURCE_CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_corrected_causal_event_training_v1.json"
DATASET = "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
TRAINING = "results/gate2_representation/gate2_20260824_gse_graph_three_seed_training_v1r_seed0"
TEACHER = "results/gate2_representation/gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0"
VERIFIER = "results/gate3_semantics/gate3_20260826_gse_exit_token_association_corrective_v2_seed0"
PREDECESSOR = "results/gate3_semantics/gate3_20260827_gse_corrected_causal_event_training_v1_seed0"
OBSERVABILITY = "results/gate3_semantics/gate3_20260827_gse_causal_geometry_delta_observability_v1_seed0"
OLD_DIRECTIONAL = "results/gate3_semantics/gate3_20260826_gse_directional_structural_event_training_v1_seed0"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def _sha(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    card = deepcopy(load_json(SOURCE_CARD))
    card.update(
        {
            "card_id": "gse_causal_geometry_delta_multitask_training_v1",
            "status": "APPROVED_FOR_ONE_IMMUTABLE_GSE_CAUSAL_GEOMETRY_DELTA_MULTITASK_TRAINING_V1",
            "purpose": "Train an explicit shared causal geometry-delta and structural-event decoder after the sealed observability audit proved transferable five-frame geometry-change signal.",
        }
    )
    card["approval"] = {
        "status": "APPROVED",
        "approved_by": "user",
        "approved_at": "2026-08-27T20:30:00+08:00",
        "authorized_operations": ["training"],
        "authorized_gates": [3],
        "scope": "One immutable three-seed frozen-backbone multitask training: C01-C06 fit, C07-C08 checkpoint/threshold selection, six epochs, 52236 head steps, zero backbone/C09/C10/M-TARE.",
        "confirmation_reference": "User instructed Codex to continue autonomously, choose the evidence-supported optimal in-scope implementation, and stop asking for routine approvals.",
    }
    card["source"].update(
        {
            "observability_run": OBSERVABILITY,
            "observability_status": "PASS_GSE_CAUSAL_GEOMETRY_DELTA_OBSERVABILITY_V1",
            "observability_seal_sha256": "c75787626ecc09a27870276e86a172d6434081034f4704a1f5286d2afe52fa50",
            "failed_pure_classifier_run": PREDECESSOR,
            "failed_pure_classifier_status": "FAIL_GSE_CORRECTED_CAUSAL_EVENT_TRAINING_V1",
            "failed_pure_classifier_seal_sha256": "4ca5aa2581a170679d14b3681f840c79d3b0f9d2b59e5e7dcc1f2af57fe17e5e",
            "partial_reuse": "READ_ONLY_SEALED_C01_C08_DATA_TEACHER_THREE_BACKBONES_FEATURE_BASELINES_AND_OBSERVABILITY; NO_FAILED_CHECKPOINT_REUSE",
        }
    )
    card["source"]["raw_sources"] = [
        "Sealed C01-C08 deduplicated Zarr: 252430 unique LiDAR frames and 188126 five-frame causal observations.",
        "Sealed corrected Teacher V1R: 1031 persistent change-point labels and continuous width/height/slope/curvature targets.",
        "Three sealed GSE backbone checkpoints; all backbone parameters remain frozen.",
        "Sealed observability PASS: 92845/29920 fit/selection geometry-valid four-step delta pairs.",
        "Failed pure-classifier outputs and old directional outputs are comparison evidence only; their checkpoints are not loaded.",
        "No C09/C10/M-TARE sensor, Teacher, graph, model output or planner artifact is read.",
    ]
    sampling = card["sampling"]
    sampling.update(
        {
            "fit_geometry_valid_lag_pairs": 92845,
            "selection_geometry_valid_lag_pairs": 29920,
            "geometry_invalid_lag_pairs_excluded": 1595,
            "causal_lag_steps": 4,
            "delta_components": ["width_m", "height_m", "slope_deg", "curvature_per_m"],
            "fit_only_delta_scale": [3.047925538538476, 0.5562241064740382, 0.5709059013471545, 0.005093988709664383],
            "normalization_rule": "Fixed C01-C06 standard deviation of signed four-step Teacher deltas. It is not learned and is never recomputed on C07-C08/C09/C10.",
            "optimization_draws": {
                "epochs_per_seed": 6,
                "seeds": 3,
                "event_identity_balanced_draws_per_epoch": 24000,
                "unique_geometry_regression_rows_per_epoch": 92845,
                "total_event_draws": 432000,
                "total_geometry_row_presentations": 1671210,
                "total_optimizer_steps": 52236,
            },
            "rule": "Each epoch visits all 92845 C01-C06 geometry-valid four-step lag pairs exactly once for signed delta regression and adds exactly 24000 identity-balanced event draws. C07-C08 is evaluated only at epochs 2/4/6; invalid geometry is never imputed.",
        }
    )
    card["teacher"].update(
        {
            "source": "Corrected persistent bidirectional causal Teacher V1R. Event labels preserve junction/terminal priority and back-projected change-point identity; continuous delta is current minus exactly four sequence steps earlier on the same traversal, with both endpoints geometry-valid.",
            "valid_mask": "A delta row is valid only when a same-traversal sequence_index-4 predecessor exists and both rows have finite geometry_valid width/height/slope/curvature. Exact counts are fit=92845 and selection=29920.",
            "student_input": "Only the existing five causal 16x720 range/mask frames. TNG, spline/mesh geometry, identity, pose, world, future observations and validation statistics are forbidden model inputs.",
            "planner_consistency_plan": "The delta branch and event branch share the azimuth-preserving causal change encoder. Event evidence may create a node only after the unchanged uncertainty/open-set gate; edges remain traversal-only.",
        }
    )
    card["methods"] = {
        "main": "For each frozen GSE seed, train a zero-initialized CausalGeometryDeltaEventHead. The shared circular change encoder predicts normalized signed delta-width/height/slope/curvature and binary-plus-conditional structural events. Geometry regression uses every valid fit pair; event supervision uses identity-balanced draws.",
        "baseline": "Strong event baseline is the sealed old directional corrected-label ensemble macro-F1=0.6879041032. Geometry baseline is the sealed three-seed frozen metric-geometry delta normalized MAE=0.5871666431. Failed pure classification is retained as an ablation.",
        "fallback": "If this frozen-representation decoder fails, stop this run and use one new predeclared bounded last-encoder-block fine-tune. Do not add epochs, weaken the Teacher, lower safety thresholds or tune graph/planner settings.",
    }
    card["metrics_and_pre_registered_gates"] = {
        "event": "Ensemble corrected event gate: macro-F1>=0.7379041032; precision>=0.98, false accept<=0.01, recall>=0.40; junction/terminal/turn identity>=0.90/0.90/0.40; change-point>=7/17. Ensemble gain>=0.05, >=2 seeds gain>=0.05 and no seed regression.",
        "geometry": "Ensemble normalized delta MAE improves >=10% over frozen metric-delta baseline; >=2 seeds improve >=10%, and no seed regresses by more than 5%.",
        "selection": "One lexicographic checkpoint per seed using corrected event gate, change-point identity coverage, lower normalized delta MAE, macro-F1 and earlier epoch; C07-C08 only.",
        "nonvacuous": True,
    }
    card["acceptance"].update(
        {
            "baseline": "Old directional corrected-label macro-F1=0.6879041032; frozen three-seed metric-delta normalized MAE=0.5871666431.",
            "delta_normalized_mae_improvement_min": 0.10,
            "seeds_reaching_delta_improvement_min": 2,
            "seed_delta_regression_max": 0.05,
            "final_generalization_claim": False,
        }
    )
    card["split"].update(
        {
            "fit": "C01-C06: 60 worlds, 142184 observations, 92845 valid delta pairs. Only new multitask-head parameters update.",
            "checkpoint_and_threshold_selection": "C07-C08: 20 worlds, 45942 observations, 29920 valid delta pairs and 17 change-point identities. Select epochs and event threshold only here.",
            "future_validation": "C09/C10 and M-TARE remain unread. Passing this run only permits a newly frozen offline topology evaluation.",
        }
    )
    card["estimated_cost"] = {
        "compute": "One RTX 5090 D; three frozen-backbone multitask heads serially",
        "wall_time_hours": 8.0,
        "disk_gb": 2.0,
        "host_ram_gb": 8,
        "gpu_memory_gb": 10,
    }
    card["evidence"] = {
        "machine_metrics": "Exact lag validity/counts and normalization; zero-step proof; three checkpoints; 18 epoch records; event/identity/open-set metrics; component and normalized delta MAE; baseline gains; optimizer/backbone/test read counts; source seals and environment.",
        "complete_visual_review": "Publish event-identity and delta-error figures only from a sealed PASS/FAIL summary with CSV/JSON provenance; preserve scientifically useful failure figures.",
        "failure_policy": "Any count/index/split/normalization/source drift, invalid target imputation, backbone gradient/update, nonfinite/resource failure, C09/C10/M-TARE access or unmet scientific gate seals FAIL. No retry or threshold relaxation.",
    }
    card["retention"] = "Retain three best heads, epoch records, selection outputs, sampling contract, summary, config, raw log, RUN_STATE and full seal. Source LiDAR/features remain in their existing sealed runs."
    write_json(CARD_PATH, card)

    inputs = [
        f"{DATASET}/RUN_STATE.json", f"{DATASET}/metrics/summary.json", f"{DATASET}/artifacts/evidence_sha256.txt", f"{DATASET}/artifacts/sequence_manifest.jsonl",
        f"{TRAINING}/RUN_STATE.json", f"{TRAINING}/metrics/summary.json", f"{TRAINING}/artifacts/evidence_sha256.txt",
        f"{TEACHER}/RUN_STATE.json", f"{TEACHER}/metrics/summary.json", f"{TEACHER}/artifacts/evidence_sha256.txt", f"{TEACHER}/artifacts/teacher_observations.jsonl",
        f"{VERIFIER}/RUN_STATE.json", f"{VERIFIER}/metrics/summary.json", f"{VERIFIER}/artifacts/evidence_sha256.txt", f"{VERIFIER}/artifacts/pair_cache/pairs.npz",
        f"{PREDECESSOR}/RUN_STATE.json", f"{PREDECESSOR}/metrics/summary.json", f"{PREDECESSOR}/artifacts/evidence_sha256.txt",
        f"{OBSERVABILITY}/RUN_STATE.json", f"{OBSERVABILITY}/metrics/summary.json", f"{OBSERVABILITY}/artifacts/evidence_sha256.txt",
        f"{OLD_DIRECTIONAL}/RUN_STATE.json", f"{OLD_DIRECTIONAL}/metrics/summary.json", f"{OLD_DIRECTIONAL}/artifacts/evidence_sha256.txt", f"{OLD_DIRECTIONAL}/artifacts/training/ensemble_selection_outputs.npz",
    ]
    for seed in (0, 1, 2):
        inputs.extend(
            [
                f"{TRAINING}/artifacts/models/seed{seed}/best.pt",
                f"{VERIFIER}/artifacts/models/seed{seed}/frozen_observation_features.npy",
                f"{OLD_DIRECTIONAL}/artifacts/training/seed{seed}/selection_outputs.npz",
            ]
        )
    tool_paths = {
        "data_card": str(CARD_PATH.relative_to(PROJECT_ROOT)),
        "multitask_head": "src/mtare_topo/representation/gse_causal_geometry_delta.py",
        "directional_head": "src/mtare_topo/representation/gse_directional_structural_event.py",
        "corrected_event_contract": "src/mtare_topo/representation/gse_corrected_causal_event.py",
        "backbone": "src/mtare_topo/representation/gse_graph.py",
        "trainer": "tools/v3/train_gse_causal_geometry_delta_multitask_v1.py",
        "runner": "tools/v3/run_gse_causal_geometry_delta_multitask_training_v1.py",
        "freezer": "tools/v3/freeze_gse_causal_geometry_delta_multitask_training_spec_v1.py",
        "evidence_integrity": "src/mtare_topo/evaluation/gse_evidence_integrity.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py",
        "create_run": "tools/v3/create_run.py",
    }
    spec = {
        "schema_version": "v3_run_spec_v1",
        "gate": 3,
        "execution_phase": 3,
        "date": "20260827",
        "slug": "gse_causal_geometry_delta_multitask_training_v1",
        "seed": 0,
        "operation": "training",
        "question": "Can explicit signed geometry-delta supervision turn the observed five-frame signal into transferable, safe persistent structural events?",
        "method": card["methods"]["main"],
        "baseline": card["methods"]["baseline"],
        "fallback": card["methods"]["fallback"],
        "data_card": str(CARD_PATH.relative_to(PROJECT_ROOT)),
        "config_path": str(CARD_PATH.relative_to(PROJECT_ROOT)),
        "user_authorization": {
            "status": "APPROVED",
            "approved_by": "user",
            "approved_at": "2026-08-27T20:30:00+08:00",
            "scope": card["approval"]["scope"],
            "confirmation_reference": card["approval"]["confirmation_reference"],
        },
        "acceptance_criteria": [
            card["metrics_and_pre_registered_gates"]["event"],
            card["metrics_and_pre_registered_gates"]["geometry"],
            "Exactly 52236 head optimizer steps, zero backbone updates and zero C09/C10/M-TARE reads; complete immutable evidence and seal.",
        ],
        "expected_counts": {
            "fit_worlds": 60, "fit_observations": 142184, "fit_geometry_valid_lag_pairs": 92845,
            "selection_worlds": 20, "selection_observations": 45942, "selection_geometry_valid_lag_pairs": 29920,
            "seeds": 3, "epochs": 6, "optimizer_steps": 52236, "backbone_optimizer_steps": 0,
            "c09_worlds_read": 0, "strict_test_worlds_read": 0, "mtare_worlds_read": 0,
        },
        "hyperparameters": {
            "epochs": 6, "event_draws_per_epoch": 24000, "regression_batch_size": 32,
            "evaluation_batch_size": 48, "evaluate_every": 2, "learning_rate": 0.001,
            "weight_decay": 0.0001, "gradient_clip": 5.0, "optimizer": "AdamW", "seeds": [0, 1, 2],
        },
        "expected_evidence": [
            "Three checkpoints, 18 epoch records, sampling/normalization contract, selection event and delta outputs, baseline gains, per-identity/open-set metrics, component MAE, environment, log, RUN_STATE and SHA-256 seal."
        ],
        "estimated_cost": {"compute": "One RTX 5090 D, serial three-seed frozen-backbone training", "disk_gb": 2.0, "wall_time_hours": 8.0},
        "frozen_inputs": {path: _sha(PROJECT_ROOT / path) for path in inputs},
        "frozen_tools": {name: {"path": path, "sha256": _sha(PROJECT_ROOT / path)} for name, path in tool_paths.items()},
        "working_directory": str(PROJECT_ROOT),
        "command": [
            "/usr/bin/systemd-inhibit", "--what=sleep:shutdown", "--why=GSE causal geometry-delta multitask training", "--mode=block",
            "/usr/bin/timeout", "--signal=INT", "--kill-after=300s", "29400s", PYTHON,
            "tools/v3/run_gse_causal_geometry_delta_multitask_training_v1.py",
            "--spec", str(SPEC_PATH),
            "--run-dir", str(PROJECT_ROOT / "results/gate3_semantics" / RUN_ID),
        ],
    }
    write_json(SPEC_PATH, spec)
    print(CARD_PATH.relative_to(PROJECT_ROOT))
    print(SPEC_PATH.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
