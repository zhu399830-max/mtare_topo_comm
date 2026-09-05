#!/usr/bin/env python3
from __future__ import annotations
import hashlib
from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, validate_data_card, write_json

RUN_ID="gate3_20260829_gse_sparse_relation_proposal_distinctness_corrective_v1_seed0"
SPEC=PROJECT_ROOT/"configs/v3/gate3/gse_sparse_relation_proposal_distinctness_corrective_v1.json"
CARD="configs/v3/gate3/data_cards/gse_sparse_relation_proposal_distinctness_corrective_v1.json"
PYTHON="/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"
def sha(path):
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b""): digest.update(block)
    return digest.hexdigest()
def main():
    if SPEC.exists(): raise RuntimeError("proposal corrective spec exists")
    card=load_json(PROJECT_ROOT/CARD); validation=validate_data_card(card)
    if not validation.passed: raise RuntimeError(str(validation.errors))
    dataset="results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"; readiness="results/gate3_semantics/gate3_20260829_gse_axis_anchored_event_relation_readiness_v2_seed0"; geometry="results/gate3_semantics/gate3_20260829_gse_sparse_relation_geometry_shape_corrective_v1_seed0"
    inputs=[CARD,f"{dataset}/RUN_STATE.json",f"{dataset}/metrics/summary.json",f"{dataset}/artifacts/evidence_sha256.txt",f"{dataset}/artifacts/sequence_manifest.jsonl",f"{readiness}/RUN_STATE.json",f"{readiness}/metrics/summary.json",f"{readiness}/artifacts/evidence_sha256.txt",f"{geometry}/RUN_STATE.json",f"{geometry}/metrics/summary.json",f"{geometry}/artifacts/evidence_sha256.txt","configs/v3/gate2/environments/phase3_torch290_cu129_zarr2187_v1r2.json"]
    tools={"model":"src/mtare_topo/representation/gse_sparse_circular_relation_transport.py","trainer":"tools/v3/train_gse_sparse_circular_relation_transport_v2.py","base_loader":"tools/v3/train_gse_axis_anchored_event_relation_v1.py","executor":"tools/v3/execute_gse_sparse_relation_proposal_distinctness_corrective_v1.py","runner":"tools/v3/run_gse_sparse_relation_proposal_distinctness_corrective_v1.py","freezer":"tools/v3/freeze_gse_sparse_relation_proposal_distinctness_corrective_spec_v1.py","model_tests":"tests/v3/unit/test_gse_sparse_circular_relation_transport.py","training_tests":"tests/v3/unit/test_train_gse_sparse_circular_relation_transport_v2.py","governance":"src/mtare_topo/governance.py","preflight":"tools/v3/preflight.py","create_run":"tools/v3/create_run.py"}
    spec={"schema_version":"v3_run_spec_v1","gate":3,"execution_phase":3,"date":"20260829","slug":"gse_sparse_relation_proposal_distinctness_corrective_v1","seed":0,"operation":"audit","data_card":CARD,"question":"Can the six-token capacity guarantee distinct circular proposals without adding a new hyperparameter?","method":"Replace top-6 with deterministic circular NMS using the existing fixed four-bin soft-support radius; verify synthetic wrap, real minimum separation, rotation, repeat and full real backward.","baseline":"Unconstrained top-6 could select adjacent bins from one physical opening.","fallback":"Any failure stops training; no new radius, threshold or graph compensation.","user_authorization":card["approval"],"acceptance_criteria":["All six proposals have pairwise circular separation greater than four bins on synthetic and real batches.","Synthetic wrap and 10-degree real rotation move every selected bin exactly.","Parameter count remains 784513 and full core-plus-descriptor backward has all gradients finite/nonzero.","Zero optimizer/checkpoint/C07-C10/M-TARE/graph/planner and frozen inputs unchanged."],"expected_counts":{"parameters":784513,"support_radius_bins":4,"fit_worlds_read":3,"real_observations":8,"optimizer_steps":0,"checkpoints_created":0,"c07_worlds_read":0,"c08_worlds_read":0,"c09_worlds_read":0,"c10_worlds_read":0,"mtare_worlds_read":0,"graph_replays":0},"expected_evidence":["Synthetic and real pairwise separation report.","Rotation/repeat/full-gradient checks.","PNG/PDF/SVG/source, CSV, environment, logs, RUN_STATE and seal."],"estimated_cost":{"compute":"Deterministic CUDA corrective only","wall_time_hours":0.05,"host_ram_gb":4,"gpu_memory_gb":12,"disk_gb":0.1},"frozen_inputs":{p:sha(PROJECT_ROOT/p) for p in sorted(inputs)},"frozen_tools":{n:{"path":p,"sha256":sha(PROJECT_ROOT/p)} for n,p in tools.items()},"working_directory":str(PROJECT_ROOT),"command":["/usr/bin/timeout","1800s",PYTHON,"tools/v3/run_gse_sparse_relation_proposal_distinctness_corrective_v1.py","--spec",str(SPEC),"--run-dir",str(PROJECT_ROOT/f"results/gate3_semantics/{RUN_ID}")]}
    write_json(SPEC,spec); print(SPEC.relative_to(PROJECT_ROOT)); return 0
if __name__=="__main__": raise SystemExit(main())
