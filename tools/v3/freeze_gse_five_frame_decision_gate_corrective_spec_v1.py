#!/usr/bin/env python3
"""Freeze Data Card/spec for the five-frame decision-mass fallback."""

from __future__ import annotations

from copy import deepcopy
import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


RUN_ID="gate3_20260828_gse_five_frame_decision_gate_corrective_v1_seed0"
CARD=PROJECT_ROOT/"configs/v3/gate3/data_cards/gse_five_frame_decision_gate_corrective_v1.json"
SPEC=PROJECT_ROOT/"configs/v3/gate3/gse_five_frame_decision_gate_corrective_v1.json"
SOURCE_CARD=PROJECT_ROOT/"configs/v3/gate3/data_cards/gse_factorized_consensus_metric_corrective_v1.json"
DATASET="results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
TEACHER="results/gate2_representation/gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0"
C09_TEACHER="results/gate2_representation/gate2_20260824_gse_teacher_manifest_v1_seed0"
TRAINING="results/gate2_representation/gate2_20260824_gse_graph_three_seed_training_v1r_seed0"
TOKEN="results/gate3_semantics/gate3_20260826_gse_exit_token_association_corrective_v2_seed0"
CAPACITY="results/gate3_semantics/gate3_20260827_gse_factorized_association_capacity_v1r_seed0"
FAILED12="results/gate3_semantics/gate3_20260828_gse_factorized_causal_node_c09_v2_seed0"
PYTHON="/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def _sha(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b""): h.update(block)
    return h.hexdigest()


def main():
    card=deepcopy(load_json(SOURCE_CARD))
    card.update({"card_id":"gse_five_frame_decision_gate_corrective_v1","status":"APPROVED_FOR_ONE_IMMUTABLE_GSE_FIVE_FRAME_DECISION_GATE_CORRECTIVE_V1","purpose":"Select a junction/terminal-specific five-frame probability gate on C07-C08 only, then apply it once to archived C09 outputs after the pre-registered 12-frame fallback failed safety."})
    card["approval"]={"status":"APPROVED","approved_by":"user-standing-authorization","approved_at":"2026-08-28T11:00:00+08:00","authorized_gates":[3],"authorized_operations":["threshold_calibration"],"scope":"One immutable two-process corrective: C07-C08-only decision threshold selection followed by one C09 application; zero model inference/update, C10, M-TARE, graph or planner.","confirmation_reference":"User explicitly instructed automatic best-choice execution without routine approval prompts."}
    card["sampling"]={"raw_frame_count":45_942+24_462,"effective_sample_count":45_942+24_462,"effective_structure_event_count":1_136+540,"structure_event_counts":{"junction":1_308,"terminal":368},"spatial_interval_m":1.0,"rule":"All five-frame causal observations in C07-C08 select one junction+terminal mass threshold; all C09 observations receive the frozen gate once.","independent_units":"Twenty C07-C08 worlds select; ten disjoint C09 worlds validate. Structural episode and decision identity are reporting units.","selection":{"observations":45_942,"junction_episodes":882,"terminal_episodes":254,"junction_identities":146,"terminal_identities":128},"validation":{"observations":24_462,"junction_episodes":426,"terminal_episodes":114,"junction_identities":71,"terminal_identities":59}}
    card["source"]={"dataset_run":DATASET,"corrected_teacher_run":TEACHER,"c09_teacher_run":C09_TEACHER,"base_training_run":TRAINING,"unified_observation_run":CAPACITY,"pair_cache_run":TOKEN,"failed_12frame_run":FAILED12,"raw_sources":["Three frozen five-frame GSE event distributions; no new model inference.","C07-C08 corrected Teacher and partition identities for selection only.","C09 original Teacher and frozen validation archives opened only by process 2."],"license_or_allowed_use":"Local research use with sealed provenance."}
    card["split"]={"model_fit":"Upstream C01-C06 only; no fit occurs here.","selection":"Exactly all C07-C08 observations. Selector has no C09 path or argument.","validation":"Exactly all C09 observations, opened only after calibration is written and selector exits.","strict_test":"C10 and all M-TARE/closed-loop worlds remain unread.","world_disjoint":True,"trajectory_disjoint":True,"historical_pollution_audit":"C09 exposed the 12-frame safety miss but does not select the five-frame threshold. The five-frame fallback and decision-only mass were pre-registered before this run.","leakage_audit":"Two OS processes enforce separation; selector reports c09_worlds_read=0 before applicator starts."}
    card["teacher"]={"source":"Sealed corrected C01-C08 Teacher for selection and sealed C09 Teacher for validation.","event":"Only junction/terminal episodes are decision positives; corridor, turn and transition are negatives for node triggering.","association":"A trigger is correct only for a previously unmatched true decision episode with the correct junction/terminal class.","valid_mask":"All rows retained; no event, identity or world masking.","student_forbidden_fields":["event label as input","episode id","identity","world ID","absolute position","future frame"],"planner_consistency_plan":"The gate only proposes decision nodes; association and physical edge verification remain separately frozen."}
    card["methods"]={"main":"Average the three frozen five-frame event distributions, score P(junction)+P(terminal), collapse each contiguous accepted response to its maximum-mass row, and classify between junction/terminal.","selection":"Grid 0.000--1.000 at 0.001. Require aggregate precision>=0.995/recall>=0.25 and each event precision>=0.99/recall>=0.25; maximize aggregate recall, then macro-F1, then fewer triggers, then lower threshold.","baseline":"Frozen 12-frame total-structural trigger: C09 precision 0.975359, false fraction 0.024641, recall 0.879630.","fallback":"If no C07-C08 threshold satisfies margins or frozen C09 fails, stop categorical node generation and redesign the proposal representation; no second threshold or graph compensation."}
    card["metrics_and_pre_registered_gates"]={"selection":"Nonvacuous threshold satisfying aggregate P>=0.995/R>=0.25 and per-event P>=0.99/R>=0.25.","validation_aggregate":"C09 precision>=0.98, false fraction<=0.01 and recall>=0.40.","validation_junction":"Precision>=0.98, recall>=0.40 and identity coverage>=0.80.","validation_terminal":"Precision>=0.98, recall>=0.60 and identity coverage>=0.80.","population":"Exact C07-C08 20 worlds/45942 observations and C09 10 worlds/24462 observations with the frozen episode/identity counts.","forbidden":"C09 threshold/checkpoint selection, model inference/update, optimizer, C10, M-TARE, graph and planner operations all zero."}
    card["estimated_cost"]={"compute":"CPU-only archived probability aggregation and 1001-point C07-C08 threshold grid","disk_gb":.1,"host_ram_gb":4,"gpu_memory_gb":0,"wall_time_hours":.25}
    card["retention"]="Retain calibration, selection/C09 arrays, metrics, PNG/PDF/SVG/source, environment, commands, logs, RUN_STATE and exact SHA-256 seal."
    card["failure_policy"]="Any input/process separation/population/resource drift or selection/C09 gate failure seals FAIL. No retry, threshold change, frame masking, C10 read, graph or planner tuning."
    card["evidence"]={"machine_metrics":"Selection and C09 per-event decision precision/recall/F1 and identity coverage.","complete_visual_review":"Selection-versus-C09 precision/recall PNG/PDF/SVG with source JSON.","failure_policy":card["failure_policy"]}
    write_json(CARD,card)
    inputs=[]
    for run,files in {DATASET:["RUN_STATE.json","metrics/summary.json","artifacts/evidence_sha256.txt"],TEACHER:["RUN_STATE.json","metrics/summary.json","artifacts/evidence_sha256.txt","artifacts/teacher_observations.jsonl"],C09_TEACHER:["RUN_STATE.json","metrics/summary.json","artifacts/evidence_sha256.txt","artifacts/teacher_observations.jsonl"],TRAINING:["RUN_STATE.json","metrics/summary.json","artifacts/evidence_sha256.txt"],TOKEN:["RUN_STATE.json","metrics/summary.json","artifacts/evidence_sha256.txt","artifacts/pair_cache/pairs.npz"],CAPACITY:["RUN_STATE.json","metrics/summary.json","artifacts/evidence_sha256.txt"],FAILED12:["RUN_STATE.json","metrics/summary.json","artifacts/evidence_sha256.txt","artifacts/evaluation/metrics.json"]}.items():
        inputs.extend(f"{run}/{file}" for file in files)
    for seed in range(3): inputs.extend([f"{CAPACITY}/artifacts/unified_observation/seed{seed}_unified_observation_features.npy",f"{TRAINING}/artifacts/models/seed{seed}/validation_outputs.npz"])
    tools={"data_card":str(CARD.relative_to(PROJECT_ROOT)),"method_plan":"docs/GSE_GRAPH_FACTORIZED_RESEARCH_PLAN_V2.md","metrics":"src/mtare_topo/evaluation/gse_causal_episode_metrics.py","runtime_alignment":"src/mtare_topo/representation/gse_causal_episode_runtime.py","selector":"tools/v3/select_gse_five_frame_decision_gate_v1.py","applicator":"tools/v3/apply_gse_five_frame_decision_gate_c09_v1.py","runner":"tools/v3/run_gse_five_frame_decision_gate_corrective_v1.py","freezer":"tools/v3/freeze_gse_five_frame_decision_gate_corrective_spec_v1.py","tests_metrics":"tests/v3/unit/test_gse_causal_episode_metrics.py","tests_alignment":"tests/v3/unit/test_gse_causal_episode_runtime.py","governance":"src/mtare_topo/governance.py","preflight":"tools/v3/preflight.py","create_run":"tools/v3/create_run.py"}
    criteria=card["metrics_and_pre_registered_gates"]
    spec={"schema_version":"v3_run_spec_v1","gate":3,"execution_phase":3,"date":"20260828","slug":"gse_five_frame_decision_gate_corrective_v1","seed":0,"operation":"threshold_calibration","question":"Can a C07-C08-selected five-frame junction/terminal probability gate satisfy the C09 one-percent false-node budget?","method":card["methods"]["main"],"baseline":card["methods"]["baseline"],"fallback":card["methods"]["fallback"],"data_card":str(CARD.relative_to(PROJECT_ROOT)),"config_path":str(CARD.relative_to(PROJECT_ROOT)),"user_authorization":deepcopy(card["approval"]),"acceptance_criteria":[criteria[name] for name in ("selection","validation_aggregate","validation_junction","validation_terminal","population","forbidden")],"expected_counts":{"selection_worlds":20,"selection_observations":45_942,"validation_worlds":10,"validation_observations":24_462,"threshold_grid_points":1001,"model_inference_frames":0,"optimizer_steps":0,"model_updates":0,"c10_worlds_read":0,"mtare_worlds_read":0},"expected_evidence":[card["retention"]],"estimated_cost":card["estimated_cost"],"frozen_inputs":{path:_sha(PROJECT_ROOT/path) for path in inputs},"frozen_tools":{name:{"path":path,"sha256":_sha(PROJECT_ROOT/path)} for name,path in tools.items()},"working_directory":str(PROJECT_ROOT),"command":["/usr/bin/timeout","--signal=INT","--kill-after=30s","1600s",PYTHON,"tools/v3/run_gse_five_frame_decision_gate_corrective_v1.py","--spec",str(SPEC),"--run-dir",str(PROJECT_ROOT/"results/gate3_semantics"/RUN_ID)]}
    write_json(SPEC,spec); print(CARD.relative_to(PROJECT_ROOT)); print(SPEC.relative_to(PROJECT_ROOT)); return 0


if __name__=="__main__": raise SystemExit(main())
