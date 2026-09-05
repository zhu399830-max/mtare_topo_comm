#!/usr/bin/env python3
"""Freeze the full C07 same-input non-learning primitive baseline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT


CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/primitive_relation_nonlearning_c07_v1.json"
SPEC = PROJECT_ROOT / "configs/v3/gate3/primitive_relation_nonlearning_c07_v1.json"
RUN_ID = "gate3_20260830_primitive_relation_nonlearning_c07_v1_seed0"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"
P1A = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1a_lossless_storage_corrective_v1_seed0"
P1B = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1b_teacher_materialization_v1r_seed0"
READINESS = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_nonlearning_readiness_v1_seed0"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    for source in (P1A, P1B, READINESS):
        state = json.loads((source / "RUN_STATE.json").read_text())
        summary = json.loads((source / "metrics/summary.json").read_text())
        if state.get("state") != "COMPLETED" or state.get("error") is not None or not summary.get("scientific_pass"):
            raise RuntimeError(f"non-learning C07 prerequisite failed: {source.name}")
    approval = {
        "status":"APPROVED", "approved_by":"user-standing-authorization",
        "approved_at":"2026-08-30T00:00:00+08:00", "authorized_gates":[3],
        "authorized_operations":["audit"],
        "confirmation_reference":"User authorized continuous autonomous execution of the frozen primitive-relation paper plan without repeated routine approvals.",
        "scope":"One immutable C07-only evaluation of the already frozen same-input non-learning primitive/relation baseline on all 10 selection parents and three paired geometries; zero parameter tuning, C08/C09/C10/model training/graph/M-TARE.",
    }
    families = (
        (1,"flat_tree_small",1452,112,1340.0),
        (2,"3d_tree_small",1362,110,1251.8794531176775),
        (3,"flat_unicyclic_small",1386,120,1266.0),
        (4,"3d_unicyclic_small",1646,128,1517.8997292602726),
        (5,"flat_branch_medium",2804,184,2620.0),
        (6,"3d_branch_medium",2364,172,2191.6761742119097),
        (7,"flat_loop_rich",3646,242,3404.0),
        (8,"3d_loop_rich",4442,246,4195.854476280583),
        (9,"flat_complex",4756,294,4462.0),
        (10,"3d_complex",5522,350,5171.850838141369),
    )
    c07 = [f"S{family:02d}_{name}_C07" for family,name,_,_,_ in families]
    fit = [
        f"S{family:02d}_{name}_C{split:02d}"
        for split in range(1,7)
        for family,name,_,_,_ in families
    ]
    strict = [
        f"S{family:02d}_{name}_C{split:02d}"
        for split in (9,10)
        for family,name,_,_,_ in families
    ]
    trajectories = [
        {
            "id":f"{world}_all_{traversals}_directed_traversals",
            "world":world, "split":"validation", "independent":True,
            "duration_s":distance, "distance_m":distance,
            "spatial_coverage_m":distance, "raw_frames_per_geometry":frames,
            "directed_traversals":traversals,
        }
        for world, (_,_,frames,traversals,distance) in zip(c07, families, strict=True)
    ]
    card = {
        "schema_version":"v3_data_card_v1",
        "card_id":"primitive_relation_nonlearning_c07_v1",
        "status":"APPROVED_FOR_ONE_IMMUTABLE_PRIMITIVE_RELATION_NONLEARNING_C07_V1",
        "approval":approval,
        "purpose":"Freeze the principal non-learning comparison on the complete C07 selection population before learned-model checkpoint selection and quantify how much primitive geometry and relation structure can be recovered from the same five causal LiDAR scans without learned weights.",
        "source":{
            "raw_sources":["sealed P1a corrected C07 causal LiDAR shards","sealed P1b C07 construction Teacher used only after prediction for scoring"],
            "license_or_allowed_use":"Local research use of project-generated Cano assets; redistribution remains subject to upstream licensing.",
        },
        "worlds":{"train":fit,"validation":c07,"ssl":[],"normalization":[],"teacher_calibration":[],"threshold_calibration":[],"augmentation_tuning":[],"checkpoint_selection":[],"strict_test":strict},
        "trajectories":trajectories,
        "sampling":{
            "independent_sampling_units":"10 topology parents; three geometry realizations are paired repeated measures within each parent.",
            "raw_frame_count":88140, "effective_sample_count":21548,
            "effective_structure_event_count":64644,
            "spatial_interval_m":1.0,
            "structure_event_counts":{"parent_worlds":10,"directed_traversals":1958,"unique_source_frames":29380,"paired_geometry_frames":88140,"unique_source_sequences":21548,"paired_geometry_sequences":64644,"paired_geometry_tasks":30},
            "rule":"Evaluate every C07 five-frame sequence exactly once in deterministic task/row order. Baseline parameters were frozen before this run; no C07 value can modify candidate construction or fitting.",
        },
        "split":{
            "world_disjoint":True,"trajectory_disjoint":True,
            "fit":"C01--C06 are not read by this evaluation.",
            "selection":"All 10 C07 parents and all three paired geometries are scored.",
            "development_transfer":"C08 is not read.",
            "strict_test":"C09/C10 and M-TARE worlds remain unread.",
            "historical_pollution_audit":"The transparent fitter and all four numeric parameters passed readiness and were frozen before full C07 scoring; no learned prediction or graph result selected them.",
        },
        "teacher":{
            "source":"P1b construction provenance supplies current visible primitive parameters, attachment, disconnected overlap and temporal visibility only to the evaluator after baseline.predict returns.",
            "valid_mask":"P1a valid returns and P1b primitive masks are preserved; invalid rays are not surfaces and missing targets remain false negatives.",
            "planner_consistency_plan":"No planner or graph executes. The same evaluator is used later for learned checkpoints.",
            "student_forbidden_inputs":"World, topology, traversal, absolute pose, primitive identity, construction graph, future frames and Teacher arrays never enter baseline.predict.",
        },
        "leakage_audit":{"optimizer_step_count":0,"model_inference_count":0,"future_sensor_frames_excluded":True,"absolute_pose_not_retained_in_student_representation":True,"mtare_benchmark_excluded":True,"test_excluded_from_supervised_training":True,"test_excluded_from_ssl":True,"test_excluded_from_normalization":True,"test_excluded_from_teacher_calibration":True,"test_excluded_from_threshold_calibration":True,"test_excluded_from_augmentation_tuning":True,"test_excluded_from_checkpoint_selection":True},
        "metrics_and_pre_registered_gates":{
            "population":"Exactly 10 C07 parents, 30 paired tasks, 64,644 sequences and 323,220 causal frame references; all paired shard tree hashes match sealed manifests.",
            "method":"The readiness-frozen configuration is used unchanged and only five range/valid scans plus relative odometry enter prediction.",
            "geometry":"Report matched target coverage and prediction precision plus axis-control, width, height, half-axis, exponent, slope, curvature and sampled-surface errors; no accuracy threshold is required for the baseline itself.",
            "relations":"Report objective primitive, attachment, disconnected-overlap and temporal presence P/R/F1 plus temporal correspondence accuracy; missing predicted primitives/relations remain false negatives.",
            "resources":"Complete within 2 CPU hours, <=4 GiB RAM and <=0.25 GiB output; zero optimizer/model/checkpoint/C08/C09/C10/graph/M-TARE.",
        },
        "estimated_cost":{"compute":"One CPU process; about 0.03 seconds per sequence plus deterministic tree verification and metric alignment.","wall_time_hours":1.0,"host_ram_gb":4,"gpu":0,"disk_gb":0.25},
        "retention":"Keep per-task timings/hashes, complete aggregate metrics, three-format paper candidate figure, environment, raw test log, RUN_STATE and seal.",
        "failure_policy":"Any source/environment drift, input leakage, row/task mismatch, nondeterminism, nonfinite output, scorer/test failure or resource excess stops the baseline run. Do not tune the fitter, delete difficult rows or read C08.",
    }
    write(CARD, card)
    tools = {
        "runner":"tools/v3/run_primitive_relation_nonlearning_c07_v1.py",
        "readiness_runner":"tools/v3/run_primitive_relation_nonlearning_readiness_v1.py",
        "baseline":"src/mtare_topo/semantics/primitive_relation_nonlearning.py",
        "exit_baseline":"src/mtare_topo/semantics/range_exit_baseline.py",
        "batch_reader":"src/mtare_topo/data/primitive_relation_batches.py",
        "single_reader":"src/mtare_topo/data/primitive_relation_training.py",
        "torch_batch":"src/mtare_topo/representation/primitive_relation_training.py",
        "metrics":"src/mtare_topo/evaluation/primitive_relation_metrics.py",
        "loss":"src/mtare_topo/representation/primitive_relation_losses.py",
        "model":"src/mtare_topo/representation/primitive_relation_model.py",
        "baseline_tests":"tests/v3/unit/test_primitive_relation_nonlearning.py",
        "metric_tests":"tests/v3/unit/test_primitive_relation_metrics.py",
        "governance":"src/mtare_topo/governance.py",
        "preflight":"tools/v3/preflight.py",
        "create_run":"tools/v3/create_run.py",
    }
    inputs = [
        CARD,
        *[source/name for source in (P1A,P1B,READINESS) for name in ("RUN_STATE.json","metrics/summary.json","artifacts/evidence_sha256.txt")],
        P1A/"artifacts/task_manifest.json", P1B/"artifacts/task_manifest.json",
        PROJECT_ROOT/"configs/v3/gate2/environments/phase3_torch290_cu129_zarr2187_v1r2.json",
        PROJECT_ROOT/"configs/v3/gate2/environments/phase3_torch290_cu129_zarr2187_pip_freeze.txt",
    ]
    spec = {
        "schema_version":"v3_run_spec_v1","gate":3,"execution_phase":3,
        "operation":"audit","date":"20260830",
        "slug":"primitive_relation_nonlearning_c07_v1","seed":0,
        "question":"How accurately can a transparent, same-input non-learning robust primitive fitter recover explicit swept geometry and port/temporal relations on all C07 development topologies before learned-model selection?",
        "method":"Register five causal scans with relative odometry, derive opening-supported axis candidates, robustly fit three superellipse sections, infer symmetric contacts/overlaps and score every output with the common permutation/reversal-safe evaluator.",
        "baseline":"This is the principal non-learning baseline that the learned primitive-relation model must exceed; Cano-like exit+rule graph remains a separate later graph baseline.",
        "fallback":"Failure stops the P2 comparison. Do not tune on C07, use Teacher identity in fitting, remove rows, read C08 or substitute a historical geometry result.",
        "data_card":str(CARD.relative_to(PROJECT_ROOT)),
        "config_path":str(CARD.relative_to(PROJECT_ROOT)),
        "user_authorization":approval,
        "acceptance_criteria":list(card["metrics_and_pre_registered_gates"].values()),
        "expected_counts":{"c07_parent_worlds":10,"paired_geometry_tasks":30,"source_independent_sequences":21548,"paired_sequences":64644,"causal_frame_references":323220,"unit_tests":9,"optimizer_steps":0,"model_inference_frames":0,"checkpoint_writes":0,"c08_rows_read":0,"c09_c10_worlds_read":0,"graph_replays":0},
        "expected_evidence":["Complete per-task source hashes/timings, geometry and objective relation metrics, three-format paper candidate figure, environment, logs, RUN_STATE and SHA-256 seal."],
        "estimated_cost":card["estimated_cost"],
        "frozen_tools":{name:{"path":path,"sha256":sha(PROJECT_ROOT/path)} for name,path in tools.items()},
        "frozen_inputs":{str(path.relative_to(PROJECT_ROOT)):sha(path) for path in inputs},
        "working_directory":str(PROJECT_ROOT),
        "command":["/usr/bin/systemd-inhibit","--what=sleep:shutdown","--why=GSE non-learning primitive C07 baseline","--mode=block","/usr/bin/timeout","--signal=INT","--kill-after=60s","7200s","/usr/bin/env",f"PYTHONPATH={PROJECT_ROOT/'src'}:{PROJECT_ROOT/'tools/v3'}",PYTHON,"tools/v3/run_primitive_relation_nonlearning_c07_v1.py","--spec",str(SPEC),"--run-dir",str(PROJECT_ROOT/f"results/gate3_semantics/{RUN_ID}")],
    }
    write(SPEC, spec)
    print(SPEC)


if __name__ == "__main__":
    main()
