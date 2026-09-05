#!/usr/bin/env python3
from __future__ import annotations
import hashlib
from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json,write_json
RUN_ID="gate3_20260828_gse_spatial_event_set_readiness_v1r_seed0";SPEC=PROJECT_ROOT/"configs/v3/gate3/gse_spatial_event_set_readiness_v1r.json";OLD=PROJECT_ROOT/"configs/v3/gate3/gse_spatial_event_set_readiness_v1.json";OLD_RUN="results/gate3_semantics/gate3_20260828_gse_spatial_event_set_readiness_v1_seed0";PYTHON="/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
 return h.hexdigest()
def main():
 if SPEC.exists():raise RuntimeError("readiness corrective spec exists")
 spec=load_json(OLD);inputs=dict(spec["frozen_inputs"])
 for p in (str(OLD.relative_to(PROJECT_ROOT)),f"{OLD_RUN}/RUN_STATE.json",f"{OLD_RUN}/metrics/summary.json",f"{OLD_RUN}/artifacts/evidence_sha256.txt",f"{OLD_RUN}/artifacts/audit/summary.json"):inputs[p]=sha(PROJECT_ROOT/p)
 tools={"model":"src/mtare_topo/representation/gse_spatial_event_set.py","teacher_interface":"src/mtare_topo/teacher/gse_spatial_multi_event_teacher.py","executor":"tools/v3/execute_gse_spatial_event_set_readiness_v1.py","runner":"tools/v3/run_gse_spatial_event_set_readiness_v1.py","freezer":"tools/v3/freeze_gse_spatial_event_set_readiness_spec_v1r.py","tests":"tests/v3/unit/test_gse_spatial_event_set.py","governance":"src/mtare_topo/governance.py","preflight":"tools/v3/preflight.py","create_run":"tools/v3/create_run.py"}
 approval={"status":"APPROVED","approved_by":"user-standing-authorization","approved_at":"2026-08-28T00:00:00+08:00","authorized_gates":[3],"authorized_operations":["audit"],"scope":"One corrective readiness run after V1 proved all data/rotation/backward gates but exposed query-index-dependent assignment tie-breaking. Only content-canonical matching changes.","confirmation_reference":"User instructed automatic best-choice execution without routine approval prompts."}
 spec.update({"slug":"gse_spatial_event_set_readiness_v1r","user_authorization":approval,"corrective_provenance":{"original_run":OLD_RUN,"original_failure":"query permutation loss error 0.000293731689453125 exceeded 1e-6; every data, rotation and gradient gate passed","single_change":"canonicalize prediction rows by content before exact assignment so tie-breaking is independent of query enumeration","teacher_data_threshold_changes":False},"frozen_inputs":dict(sorted(inputs.items())),"frozen_tools":{n:{"path":p,"sha256":sha(PROJECT_ROOT/p)} for n,p in tools.items()},"command":["/usr/bin/timeout","360s",PYTHON,"tools/v3/run_gse_spatial_event_set_readiness_v1.py","--spec",str(SPEC),"--run-dir",str(PROJECT_ROOT/f"results/gate3_semantics/{RUN_ID}")]});write_json(SPEC,spec);print(SPEC.relative_to(PROJECT_ROOT));return 0
if __name__=="__main__":raise SystemExit(main())
