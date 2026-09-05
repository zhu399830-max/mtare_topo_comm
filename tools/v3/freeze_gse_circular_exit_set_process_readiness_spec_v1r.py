#!/usr/bin/env python3
"""Freeze metric-only V1R exit-set readiness spec."""
from __future__ import annotations
import hashlib
from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json,validate_data_card,write_json
RUN_ID="gate3_20260829_gse_circular_exit_set_process_readiness_v1r_seed0";SPEC=PROJECT_ROOT/"configs/v3/gate3/gse_circular_exit_set_process_readiness_v1r.json";DATA_CARD="configs/v3/gate3/data_cards/gse_circular_exit_set_process_readiness_v1r.json";PYTHON="/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"
def sha256(path):
 d=hashlib.sha256()
 with path.open("rb") as s:
  for b in iter(lambda:s.read(4*1024*1024),b""):d.update(b)
 return d.hexdigest()
def main():
 if SPEC.exists():raise RuntimeError("V1R spec exists")
 card=load_json(PROJECT_ROOT/DATA_CARD);v=validate_data_card(card)
 if not v.passed:raise RuntimeError(f"V1R card invalid:{v.errors}")
 dataset="results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0";teacher="results/gate2_representation/gate2_20260829_gse_circular_exit_geometry_field_teacher_export_v1_seed0";v1="results/gate3_semantics/gate3_20260829_gse_circular_exit_set_process_readiness_v1_seed0"
 inputs=[DATA_CARD,f"{dataset}/RUN_STATE.json",f"{dataset}/metrics/summary.json",f"{dataset}/artifacts/evidence_sha256.txt",f"{teacher}/RUN_STATE.json",f"{teacher}/metrics/summary.json",f"{teacher}/artifacts/evidence_sha256.txt",f"{v1}/RUN_STATE.json",f"{v1}/metrics/summary.json",f"{v1}/artifacts/evidence_sha256.txt","configs/v3/gate2/environments/phase3_torch290_cu129_zarr2187_v1r2.json"]
 tools={"set_process":"src/mtare_topo/representation/gse_circular_exit_set_process.py","backbone":"src/mtare_topo/representation/gse_circular_peak_geometry_model.py","executor":"tools/v3/execute_gse_circular_exit_set_process_readiness_v1.py","source_audit":"tools/v3/execute_gse_circular_peak_geometry_model_readiness_v1.py","runner":"tools/v3/run_gse_circular_exit_set_process_readiness_v1r.py","freezer":"tools/v3/freeze_gse_circular_exit_set_process_readiness_spec_v1r.py","tests":"tests/v3/unit/test_gse_circular_exit_set_process.py","metric_test":"tests/v3/unit/test_gse_circular_exit_set_process_executor.py","governance":"src/mtare_topo/governance.py","preflight":"tools/v3/preflight.py","create_run":"tools/v3/create_run.py"}
 spec={"schema_version":"v3_run_spec_v1","gate":3,"execution_phase":3,"date":"20260829","slug":"gse_circular_exit_set_process_readiness_v1r","seed":0,"operation":"audit","data_card":DATA_CARD,
 "question":"Does the exact V1 set-process interface pass when its 5.96e-8 synthetic rotation difference is compared with the pre-registered bounded floating contract instead of exact zero?",
 "method":"Exact V1 data/model/loss/rows/checks; only synthetic set-NLL rotation comparison changes from ==0 to <=1e-6, below the original 3e-5 contract.","baseline":"Immutable V1 system-normal scientific FAIL caused only by exact floating equality.","fallback":"Any other check change or rotation error above 1e-6 stops before training.","user_authorization":card["approval"],
 "acceptance_criteria":["Exact V1 population/model/loss/rows and all ten non-defective checks unchanged.","Synthetic rotation error <=1e-6 and original model equivariance errors <=3e-5.","All eleven checks PASS with zero optimizer/checkpoint/test/graph and frozen inputs unchanged."],
 "expected_counts":{"worlds":80,"observations":188126,"visible_exits":396913,"parameters":769784,"real_rows":8,"optimizer_steps":0,"checkpoint_writes":0,"graph_replays":0,"c09_worlds_read":0,"c10_worlds_read":0,"mtare_worlds_read":0},
 "expected_evidence":["V1 failure binding and explicit metric-only repair record.","Exact readiness summary/figures/rows plus environment/log/RUN_STATE/seal."],"estimated_cost":{"compute":"CPU-only exact V1 repeat","wall_time_hours":0.08,"host_ram_gb":4,"gpu_memory_gb":0,"disk_gb":0.1,"gpu":"none"},
 "frozen_inputs":{p:sha256(PROJECT_ROOT/p) for p in sorted(inputs)},"frozen_tools":{n:{"path":p,"sha256":sha256(PROJECT_ROOT/p)} for n,p in tools.items()},"working_directory":str(PROJECT_ROOT),"command":["/usr/bin/timeout","1200s",PYTHON,"tools/v3/run_gse_circular_exit_set_process_readiness_v1r.py","--spec",str(SPEC),"--run-dir",str(PROJECT_ROOT/f"results/gate3_semantics/{RUN_ID}")]}
 write_json(SPEC,spec);print(SPEC.relative_to(PROJECT_ROOT));return 0
if __name__=="__main__":raise SystemExit(main())
