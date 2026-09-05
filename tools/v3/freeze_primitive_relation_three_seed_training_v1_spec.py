#!/usr/bin/env python3
"""Freeze the sole P2 three-seed primitive-relation training run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT


CARD = PROJECT_ROOT/"configs/v3/gate3/data_cards/primitive_relation_three_seed_training_v1.json"
SPEC = PROJECT_ROOT/"configs/v3/gate3/primitive_relation_three_seed_training_v1.json"
RUN_ID = "gate3_20260830_primitive_relation_three_seed_training_v1_seed0"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"
P1A = PROJECT_ROOT/"results/gate3_semantics/gate3_20260830_primitive_relation_p1a_lossless_storage_corrective_v1_seed0"
P1B = PROJECT_ROOT/"results/gate3_semantics/gate3_20260830_primitive_relation_p1b_teacher_materialization_v1r_seed0"
MODEL_READINESS = PROJECT_ROOT/"results/gate3_semantics/gate3_20260830_primitive_relation_model_readiness_v1r_seed0"
BASELINE_READINESS = PROJECT_ROOT/"results/gate3_semantics/gate3_20260830_primitive_relation_nonlearning_readiness_v1_seed0"
BASELINE_C07 = PROJECT_ROOT/"results/gate3_semantics/gate3_20260830_primitive_relation_nonlearning_c07_v1_seed0"


def sha(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def write(path: Path,value:dict)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,sort_keys=True)+"\n",encoding="utf-8")


def main()->None:
    sources=(P1A,P1B,MODEL_READINESS,BASELINE_READINESS,BASELINE_C07)
    for source in sources:
        state=json.loads((source/"RUN_STATE.json").read_text())
        summary=json.loads((source/"metrics/summary.json").read_text())
        if state.get("state")!="COMPLETED" or state.get("error") is not None or not summary.get("scientific_pass"):
            raise RuntimeError(f"primitive training prerequisite failed: {source.name}")
    approval={
        "status":"APPROVED","approved_by":"user-standing-authorization",
        "approved_at":"2026-08-30T00:00:00+08:00","authorized_gates":[3],
        "authorized_operations":["training","checkpoint_selection","threshold_calibration"],
        "confirmation_reference":"User explicitly authorized continuous autonomous execution and instructed the agent not to request repeated routine approvals while preserving paper evidence.",
        "scope":"One immutable three-seed primitive-relation training/evaluation run: C01-C06 gradients, C07 checkpoint and threshold selection, and C08 only after the pre-registered C07 perception gate passes in at least two seeds; no C09/C10/graph/M-TARE.",
    }
    families=((1,"flat_tree_small"),(2,"3d_tree_small"),(3,"flat_unicyclic_small"),(4,"3d_unicyclic_small"),(5,"flat_branch_medium"),(6,"3d_branch_medium"),(7,"flat_loop_rich"),(8,"3d_loop_rich"),(9,"flat_complex"),(10,"3d_complex"))
    fit=[f"S{family:02d}_{name}_C{split:02d}" for split in range(1,7) for family,name in families]
    c07=[f"S{family:02d}_{name}_C07" for family,name in families]
    c08=[f"S{family:02d}_{name}_C08" for family,name in families]
    strict=[f"S{family:02d}_{name}_C{split:02d}" for split in (9,10) for family,name in families]
    trajectories=[
        {"id":"all_C01_C06_fit_directed_traversals_three_geometries","world":fit[0],"split":"train","independent":True,"duration_s":178488.76052725562,"distance_m":178488.76052725562,"spatial_coverage_m":178488.76052725562},
        {"id":"all_C07_selection_directed_traversals_three_geometries","world":c07[0],"split":"validation","independent":True,"duration_s":27421.160671011814,"distance_m":27421.160671011814,"spatial_coverage_m":27421.160671011814},
        {"id":"all_C08_zero_adaptation_directed_traversals_three_geometries","world":c08[0],"split":"validation","independent":True,"duration_s":30434.24064490384,"distance_m":30434.24064490384,"spatial_coverage_m":30434.24064490384},
    ]
    card={
        "schema_version":"v3_data_card_v1","card_id":"primitive_relation_three_seed_training_v1",
        "status":"APPROVED_FOR_ONE_IMMUTABLE_PRIMITIVE_RELATION_THREE_SEED_TRAINING_V1","approval":approval,
        "purpose":"Train the construction-supervised explicit swept-primitive and relation model with three fixed seeds, select only on C07, and test whether learned geometry/relations exceed the frozen same-input robust fitter before any topology graph is permitted.",
        "source":{"raw_sources":["sealed P1a corrected 16x720 range/valid shards for fit/C07 and conditionally C08","sealed P1b 32-slot visible primitive, attachment, disconnected-overlap and temporal Teacher"],"license_or_allowed_use":"Local research use of project-generated Cano assets; redistribution remains subject to upstream licensing."},
        "worlds":{"train":fit,"validation":c07+c08,"ssl":[],"normalization":fit,"teacher_calibration":[],"threshold_calibration":c07,"augmentation_tuning":[],"checkpoint_selection":c07,"strict_test":strict},
        "trajectories":trajectories,
        "sampling":{"independent_sampling_units":"80 topology parents; three geometry realizations are paired repeated measures and never cross parent split.","raw_frame_count":757290,"effective_sample_count":188126,"effective_structure_event_count":564378,"spatial_interval_m":1.0,"structure_event_counts":{"fit_parent_worlds":60,"fit_geometry_tasks":180,"fit_unique_source_frames":190600,"fit_paired_frames":571800,"fit_unique_source_sequences":142184,"fit_paired_sequences":426552,"fit_primitives":18159,"fit_directed_attachments":6602968,"fit_disconnected_overlaps":2249595,"fit_temporal_dustbins":405311,"c07_parent_worlds":10,"c07_paired_sequences":64644,"c08_parent_worlds":10,"c08_paired_sequences":73182},"rule":"Every fit row appears once per epoch in deterministic shard-local shuffle; no frame deletion or geometry reweighting. C07 is evaluated in fixed order after each epoch. C08 is read only if at least two seeds pass every frozen C07 perception comparison."},
        "split":{"world_disjoint":True,"trajectory_disjoint":True,"fit":"C01-C06 only provide gradients and fit-only normalization implicit in fixed physical units.","selection":"C07 selects the minimum equal-six-family loss checkpoint and 91-point [0.05,0.95] existence/relation thresholds.","development_transfer":"C08 receives no gradient, checkpoint choice, threshold choice, normalization update or fallback. It remains unread if C07 fails.","strict_test":"C09/C10 and all M-TARE benchmark worlds remain unread.","historical_pollution_audit":"Historical exit/event/Composer/ERCSS predictions and checkpoints do not initialize or select this model. Only the transparent fitter's sealed C07 metrics are used as the pre-registered comparison."},
        "teacher":{"source":"P1b construction program provenance, cropped by actual five-frame ray support and expressed in current sensor coordinates.","valid_mask":"Primitive mask, binary relation arrays and temporal visibility are preserved; every fit row has 2-23 visible primitives and a nonempty attachment target.","planner_consistency_plan":"No planner or graph runs. C08 compact predictions are retained only if the C07 perception gate opens transfer.","student_forbidden_inputs":"Forward receives only [5,2,16,720] normalized range/valid and five current-sensor relative translations/yaws; absolute pose, world, TNG, traversal and construction identity are absent."},
        "leakage_audit":{"optimizer_step_count":481248,"model_inference_count":0,"future_sensor_frames_excluded":True,"absolute_pose_not_retained_in_student_representation":True,"mtare_benchmark_excluded":True,"test_excluded_from_supervised_training":True,"test_excluded_from_ssl":True,"test_excluded_from_normalization":True,"test_excluded_from_teacher_calibration":True,"test_excluded_from_threshold_calibration":True,"test_excluded_from_augmentation_tuning":True,"test_excluded_from_checkpoint_selection":True},
        "metrics_and_pre_registered_gates":{
            "implementation":"Exactly 27 frozen unit tests pass, including a guard that C08 shards are not opened before the complete C07 stop branch; deterministic full-FP32 CUDA, TF32 off, model parameters/losses/split unchanged from readiness.",
            "training":"Seeds 0/1/2, six epochs, batch16, AdamW lr=3e-4 weight_decay=1e-4, clip=1.0; epoch schedule is geometry, relation, temporal+uncertainty, then three equal-six-family joint epochs. Expected 160,416 steps per seed and 481,248 total.",
            "selection":"For each seed choose minimum C07 equal-six-family total loss across six epochs; tune only the frozen 91 thresholds on C07. Ensemble may not rescue fewer than two passing seeds.",
            "c07_perception_gate":"At least two seeds must each improve sampled-surface Chamfer >=10%, macro axis/width/height/exponent/slope/curvature MAE >=10%, and attachment F1 >=5 points over the sealed non-learning C07 baseline, while primitive F1 and target coverage do not regress. Otherwise stop before C08.",
            "c08_transfer_gate":"If C07 opens transfer, the unchanged checkpoint/thresholds must satisfy the same five checks against the same frozen non-learning method on C08 in at least two seeds. Failure stops before graph.",
            "safety_diagnostics":"C07 separately selects maximum-recall relation thresholds at precision >=0.98; unavailable safe thresholds are reported as failure diagnostics and never relaxed on C08.",
            "resources":"Full run <=40 h, process GPU memory <=16 GiB, RAM <=16 GiB and output <=6 GiB; any overrun stops. Zero C09/C10/graph/M-TARE.",
        },
        "estimated_cost":{"compute":"One RTX 5090 sequentially trains three seeds. Measured full forward/backward batch16 throughput is about 102 samples/s; validation batch128 about 145 samples/s.","wall_time_hours":32,"host_ram_gb":16,"gpu":1,"gpu_memory_gb":16,"disk_gb":6},
        "retention":"Keep all 18 epoch checkpoints/history, three selected checkpoints, C07 thresholds/metrics, conditional compact C08 outputs, baseline comparisons, paper figure, environment, raw logs, RUN_STATE and seal.",
        "failure_policy":"Any source/tool/environment drift, nonfinite gradient, row/task mismatch, >16 GiB process memory, C07 scientific failure, C08 regression or forbidden-world read stops. Do not add losses, epochs, initialization, threshold grids, rules, delete frames or enter graph to compensate.",
    }
    write(CARD,card)
    tools={
        "runner":"tools/v3/run_primitive_relation_three_seed_training_v1.py",
        "trainer":"tools/v3/train_primitive_relation_model_v1.py",
        "evaluator":"tools/v3/evaluate_primitive_relation_three_seed_v1.py",
        "batch_reader":"src/mtare_topo/data/primitive_relation_batches.py",
        "single_reader":"src/mtare_topo/data/primitive_relation_training.py",
        "training_contract":"src/mtare_topo/representation/primitive_relation_training.py",
        "model":"src/mtare_topo/representation/primitive_relation_model.py",
        "loss":"src/mtare_topo/representation/primitive_relation_losses.py",
        "metrics":"src/mtare_topo/evaluation/primitive_relation_metrics.py",
        "nonlearning":"src/mtare_topo/semantics/primitive_relation_nonlearning.py",
        "exit_baseline":"src/mtare_topo/semantics/range_exit_baseline.py",
        "reader_tests":"tests/v3/unit/test_primitive_relation_training.py",
        "model_tests":"tests/v3/unit/test_primitive_relation_model.py",
        "baseline_tests":"tests/v3/unit/test_primitive_relation_nonlearning.py",
        "metric_tests":"tests/v3/unit/test_primitive_relation_metrics.py",
        "schedule_tests":"tests/v3/unit/test_primitive_relation_training_schedule.py",
        "evaluation_tests":"tests/v3/unit/test_evaluate_primitive_relation_three_seed_v1.py",
        "governance":"src/mtare_topo/governance.py","preflight":"tools/v3/preflight.py","create_run":"tools/v3/create_run.py",
    }
    inputs=[CARD,*[source/name for source in sources for name in ("RUN_STATE.json","metrics/summary.json","artifacts/evidence_sha256.txt")],P1A/"artifacts/task_manifest.json",P1B/"artifacts/task_manifest.json",PROJECT_ROOT/"configs/v3/gate2/environments/phase3_torch290_cu129_zarr2187_v1r2.json",PROJECT_ROOT/"configs/v3/gate2/environments/phase3_torch290_cu129_zarr2187_pip_freeze.txt"]
    spec={
        "schema_version":"v3_run_spec_v1","gate":3,"execution_phase":3,"operation":"training","date":"20260830","slug":"primitive_relation_three_seed_training_v1","seed":0,
        "question":"Can construction-supervised five-frame learning recover explicit swept geometry and attachment relations better than the same-input robust fitter on C07 and then transfer without adaptation to C08?",
        "method":"Train the frozen 32-query causal circular encoder/registered-point decoder with staged use of the six frozen dimensionless losses, select each seed only by C07 total loss, calibrate existence/relation thresholds only on C07, then conditionally score C08 once.",
        "baseline":"Sealed same-input non-learning robust superellipse/axis/port fitter on C07 and unchanged C08; Cano-like rule graph remains for P3.",
        "fallback":"Scientific failure stops before C08 or graph as specified. Preserve checkpoints as learned-primitive ablation/failure evidence; do not add event rules, losses, epochs or planner tuning.",
        "data_card":str(CARD.relative_to(PROJECT_ROOT)),"config_path":str(CARD.relative_to(PROJECT_ROOT)),"user_authorization":approval,
        "acceptance_criteria":list(card["metrics_and_pre_registered_gates"].values()),
        "expected_counts":{"fit_parent_worlds":60,"fit_geometry_tasks":180,"fit_rows_per_epoch":426552,"epochs_per_seed":6,"seeds":3,"optimizer_steps_per_seed":160416,"optimizer_steps_total":481248,"c07_parent_worlds":10,"c07_rows":64644,"conditional_c08_parent_worlds":10,"conditional_c08_rows_per_method":73182,"unit_tests":27,"c09_c10_worlds_read":0,"graph_replays":0,"mtare_worlds_read":0},
        "expected_evidence":["Three complete six-epoch histories and selected checkpoints; C07 loss/threshold/geometry/relation comparisons; conditional C08 compact outputs and transfer comparisons; raw logs, environment, paper figure, RUN_STATE and seal."],
        "estimated_cost":card["estimated_cost"],
        "frozen_tools":{name:{"path":path,"sha256":sha(PROJECT_ROOT/path)} for name,path in tools.items()},
        "frozen_inputs":{str(path.relative_to(PROJECT_ROOT)):sha(path) for path in inputs},
        "working_directory":str(PROJECT_ROOT),
        "command":["/usr/bin/systemd-inhibit","--what=sleep:shutdown","--why=GSE primitive relation three-seed training","--mode=block","/usr/bin/timeout","--signal=INT","--kill-after=60s","144000s","/usr/bin/env","CUBLAS_WORKSPACE_CONFIG=:4096:8",f"PYTHONPATH={PROJECT_ROOT/'src'}:{PROJECT_ROOT/'tools/v3'}",PYTHON,"tools/v3/run_primitive_relation_three_seed_training_v1.py","--spec",str(SPEC),"--run-dir",str(PROJECT_ROOT/f"results/gate3_semantics/{RUN_ID}")],
    }
    write(SPEC,spec);print(SPEC)


if __name__=="__main__":main()
