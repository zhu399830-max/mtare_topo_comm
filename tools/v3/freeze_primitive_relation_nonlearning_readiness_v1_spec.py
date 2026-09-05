#!/usr/bin/env python3
"""Freeze the same-input non-learning primitive-relation readiness audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT


CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/primitive_relation_nonlearning_readiness_v1.json"
SPEC = PROJECT_ROOT / "configs/v3/gate3/primitive_relation_nonlearning_readiness_v1.json"
RUN_ID = "gate3_20260830_primitive_relation_nonlearning_readiness_v1_seed0"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"
P1A = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1a_lossless_storage_corrective_v1_seed0"
P1B = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1b_teacher_materialization_v1r_seed0"
MODEL_READINESS = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_model_readiness_v1r_seed0"


def sha(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b""): digest.update(block)
    return digest.hexdigest()


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value,indent=2,ensure_ascii=False,sort_keys=True)+"\n",encoding="utf-8")


def main() -> None:
    for source in (P1A,P1B,MODEL_READINESS):
        summary=json.loads((source/"metrics/summary.json").read_text()); state=json.loads((source/"RUN_STATE.json").read_text())
        if not summary.get("scientific_pass") or state.get("state")!="COMPLETED" or state.get("error") is not None: raise RuntimeError(f"non-learning readiness prerequisite failed: {source.name}")
    approval={"status":"APPROVED","approved_by":"user-standing-authorization","approved_at":"2026-08-30T00:00:00+08:00","authorized_gates":[3],"authorized_operations":["audit"],"confirmation_reference":"User authorized continuous autonomous execution of the frozen primitive-relation paper plan without repeated routine approvals.","scope":"One immutable zero-training readiness of the same-input non-learning primitive/relation baseline on one fit source sequence and three paired geometry realizations; no C07/C08/C09/C10, model, optimizer, checkpoint, graph or M-TARE read."}
    strict=[f"S{family:02d}_{name}_C{split:02d}" for split in (9,10) for family,name in ((1,"flat_tree_small"),(2,"3d_tree_small"),(3,"flat_unicyclic_small"),(4,"3d_unicyclic_small"),(5,"flat_branch_medium"),(6,"3d_branch_medium"),(7,"flat_loop_rich"),(8,"3d_loop_rich"),(9,"flat_complex"),(10,"3d_complex"))]
    card={
        "schema_version":"v3_data_card_v1","card_id":"primitive_relation_nonlearning_readiness_v1","status":"APPROVED_FOR_ONE_IMMUTABLE_PRIMITIVE_RELATION_NONLEARNING_READINESS_V1","approval":approval,
        "purpose":"Prove that a transparent same-input five-frame geometric fitter can deterministically emit the complete typed primitive/relation interface before it is frozen as the principal non-learning comparison.",
        "source":{"raw_sources":["sealed P1a corrected causal LiDAR shards","sealed P1b Teacher used only to bind the paired row and report target cardinality, never passed to baseline.predict"],"license_or_allowed_use":"Local research use of project-generated Cano assets; redistribution remains subject to upstream licensing."},
        "worlds":{"train":["S10_3d_complex_C04"],"validation":["S10_3d_complex_C07"],"ssl":[],"normalization":[],"teacher_calibration":[],"threshold_calibration":[],"augmentation_tuning":[],"checkpoint_selection":[],"strict_test":strict},
        "trajectories":[{"id":"S10_3d_complex_C04_source_sequence_188724_nonlearning_readiness","world":"S10_3d_complex_C04","split":"train","independent":True,"duration_s":4.0,"distance_m":4.0,"spatial_coverage_m":4.0}],
        "sampling":{"independent_sampling_units":"One source topology sequence; three geometry realizations are paired repeated measures.","raw_frame_count":15,"effective_sample_count":1,"effective_structure_event_count":5,"spatial_interval_m":1.0,"structure_event_counts":{"paired_geometry_rows":3,"causal_frame_observations":15,"unique_source_primitives":5,"expected_nonlearning_candidates_per_row":3},"rule":"Read only row 2 of the three sealed C04 fit shards. Baseline configuration is fixed before any accuracy metric and receives only range/valid plus relative odometry."},
        "split":{"world_disjoint":True,"trajectory_disjoint":True,"fit":"Only one C04 row is read for interface readiness.","selection":"C07 zero rows read.","development_transfer":"C08 zero rows read.","strict_test":"C09/C10 and M-TARE worlds remain unread.","historical_pollution_audit":"No prior model, graph, event label or performance result chooses sectors, candidates or geometric fit parameters."},
        "teacher":{"source":"P1b is used only to verify paired-row identity and target cardinality outside baseline.predict; no Teacher field enters candidate generation or fitting.","valid_mask":"The baseline consumes exactly the P1a valid-return channel and never treats invalid maximum-range cells as surfaces.","planner_consistency_plan":"No graph/planner runs. The same frozen output will later enter the common offline evaluator and graph interface.","student_forbidden_inputs":"World, parent, traversal, primitive identity, TNG, construction graph, absolute pose and future frames are absent."},
        "leakage_audit":{"optimizer_step_count":0,"model_inference_count":0,"future_sensor_frames_excluded":True,"absolute_pose_not_retained_in_student_representation":True,"mtare_benchmark_excluded":True,"test_excluded_from_supervised_training":True,"test_excluded_from_ssl":True,"test_excluded_from_normalization":True,"test_excluded_from_teacher_calibration":True,"test_excluded_from_threshold_calibration":True,"test_excluded_from_augmentation_tuning":True,"test_excluded_from_checkpoint_selection":True},
        "metrics_and_pre_registered_gates":{"tests":"Exactly five synthetic tests cover corridor merge, multi-branch stubs/relations, determinism, rotation up to endpoint reversal and fail-closed inputs.","real_interface":"Each paired real row emits 1--32 finite candidates, finite geometry/uncertainty, symmetric relations and bit-exact repeated output.","student_boundary":"Only five range/valid scans and causal relative odometry enter predict; Teacher identity never enters.","resources":"Every real row completes in <=1 s CPU; total output <=0.05 GiB; zero model/optimizer/checkpoint/graph/C07--C10/M-TARE."},
        "estimated_cost":{"compute":"One CPU process in the frozen NumPy/SciPy/Zarr environment; five tests and two repeated predictions for each of three real rows.","wall_time_hours":0.25,"host_ram_gb":4,"gpu":0,"disk_gb":0.05},
        "retention":"Keep source/config hashes, tests, real predictions, candidate counts, runtime, environment, RUN_STATE and seal.",
        "failure_policy":"Any source/environment drift, nondeterminism, nonfinite/invalid geometry, asymmetric relation, forbidden input, test failure or resource excess stops baseline qualification. Do not use Teacher to repair candidates or tune on C07/C08.",
    }
    write(CARD,card)
    tools={"runner":"tools/v3/run_primitive_relation_nonlearning_readiness_v1.py","baseline":"src/mtare_topo/semantics/primitive_relation_nonlearning.py","exit_baseline":"src/mtare_topo/semantics/range_exit_baseline.py","reader":"src/mtare_topo/data/primitive_relation_training.py","tests":"tests/v3/unit/test_primitive_relation_nonlearning.py","governance":"src/mtare_topo/governance.py","preflight":"tools/v3/preflight.py","create_run":"tools/v3/create_run.py"}
    inputs=[CARD,*[source/name for source in (P1A,P1B,MODEL_READINESS) for name in ("RUN_STATE.json","metrics/summary.json","artifacts/evidence_sha256.txt")],P1A/"artifacts/task_manifest.json",P1B/"artifacts/task_manifest.json",PROJECT_ROOT/"configs/v3/gate2/environments/phase3_torch290_cu129_zarr2187_v1r2.json",PROJECT_ROOT/"configs/v3/gate2/environments/phase3_torch290_cu129_zarr2187_pip_freeze.txt"]
    spec={"schema_version":"v3_run_spec_v1","gate":3,"execution_phase":3,"operation":"audit","date":"20260830","slug":"primitive_relation_nonlearning_readiness_v1","seed":0,"question":"Can a transparent same-input five-frame geometric fitter provide a deterministic, finite and leakage-safe primitive/relation baseline before common evaluation and model training?","method":"Register five scans with relative odometry, form opening-sector axis hypotheses, merge only an antipodal corridor pair, robustly fit three axis sections and endpoint superellipse parameters, and infer symmetric port contacts/observed overlaps without learned weights or Teacher input.","baseline":"This run qualifies the principal non-learning comparison itself; the learned 32-slot primitive-relation model is not invoked.","fallback":"Failure stops baseline qualification. Do not add GT identity, tune on C07/C08, lower interface checks or substitute the old single-tunnel current-frame estimator.","data_card":str(CARD.relative_to(PROJECT_ROOT)),"config_path":str(CARD.relative_to(PROJECT_ROOT)),"user_authorization":approval,"acceptance_criteria":list(card["metrics_and_pre_registered_gates"].values()),"expected_counts":{"independent_source_sequences":1,"paired_geometry_rows":3,"causal_frames":15,"unit_tests":5,"optimizer_steps":0,"model_inference_frames":0,"checkpoint_writes":0,"c07_c08_rows_read":0,"c09_c10_worlds_read":0,"graph_replays":0},"expected_evidence":["Five-test log, three real deterministic typed predictions, source/tree hashes, runtime, environment, RUN_STATE and SHA-256 seal."],"estimated_cost":card["estimated_cost"],"frozen_tools":{name:{"path":path,"sha256":sha(PROJECT_ROOT/path)} for name,path in tools.items()},"frozen_inputs":{str(path.relative_to(PROJECT_ROOT)):sha(path) for path in inputs},"working_directory":str(PROJECT_ROOT),"command":["/usr/bin/timeout","--signal=INT","--kill-after=60s","900s",PYTHON,"tools/v3/run_primitive_relation_nonlearning_readiness_v1.py","--spec",str(SPEC),"--run-dir",str(PROJECT_ROOT/f"results/gate3_semantics/{RUN_ID}")]}
    write(SPEC,spec); print(SPEC)


if __name__=="__main__": main()
