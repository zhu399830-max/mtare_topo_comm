#!/usr/bin/env python3
"""Freeze the V1R environment-checker-only corrective."""

from __future__ import annotations

from copy import deepcopy
import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


RUN_ID="gate3_20260828_gse_five_frame_decision_gate_corrective_v1r_seed0"
FAILED="results/gate3_semantics/gate3_20260828_gse_five_frame_decision_gate_corrective_v1_seed0"
CARD=PROJECT_ROOT/"configs/v3/gate3/data_cards/gse_five_frame_decision_gate_corrective_v1r.json"
SPEC=PROJECT_ROOT/"configs/v3/gate3/gse_five_frame_decision_gate_corrective_v1r.json"
PYTHON="/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def _sha(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b""): h.update(block)
    return h.hexdigest()


def main():
    card=deepcopy(load_json(PROJECT_ROOT/"configs/v3/gate3/data_cards/gse_five_frame_decision_gate_corrective_v1.json"))
    card.update({"card_id":"gse_five_frame_decision_gate_corrective_v1r","status":"APPROVED_FOR_ONE_IMMUTABLE_GSE_FIVE_FRAME_DECISION_GATE_CORRECTIVE_V1R","purpose":"Execute the unchanged five-frame decision fallback after correcting only the recorded SciPy patch version from 1.15.2 to the actual sealed 1.15.3."})
    card["approval"]={**card["approval"],"scope":"One immutable V1R; only the expected SciPy patch version changes. Data, method, threshold grid, gates and process separation are byte-identical in meaning."}
    card["source"]["failed_environment_checker_v1"]=FAILED
    card["split"]["historical_pollution_audit"] += " V1 stopped before selector launch because the runner expected SciPy 1.15.2 while the sealed sidecar is 1.15.3; C09 reads and threshold steps were zero."
    write_json(CARD,card)
    spec=deepcopy(load_json(PROJECT_ROOT/"configs/v3/gate3/gse_five_frame_decision_gate_corrective_v1.json"))
    spec.update({"slug":"gse_five_frame_decision_gate_corrective_v1r","data_card":str(CARD.relative_to(PROJECT_ROOT)),"config_path":str(CARD.relative_to(PROJECT_ROOT)),"user_authorization":deepcopy(card["approval"])})
    for relative in (f"{FAILED}/RUN_STATE.json",f"{FAILED}/metrics/summary.json",f"{FAILED}/artifacts/evidence_sha256.txt"):
        spec["frozen_inputs"][relative]=_sha(PROJECT_ROOT/relative)
    paths={name:record["path"] for name,record in spec["frozen_tools"].items() if name not in {"data_card","runner","freezer"}}
    paths.update({"data_card":str(CARD.relative_to(PROJECT_ROOT)),"runner_base":"tools/v3/run_gse_five_frame_decision_gate_corrective_v1.py","runner":"tools/v3/run_gse_five_frame_decision_gate_corrective_v1r.py","freezer":"tools/v3/freeze_gse_five_frame_decision_gate_corrective_spec_v1r.py"})
    spec["frozen_tools"]={name:{"path":path,"sha256":_sha(PROJECT_ROOT/path)} for name,path in paths.items()}
    spec["command"]=["/usr/bin/timeout","--signal=INT","--kill-after=30s","1600s",PYTHON,"tools/v3/run_gse_five_frame_decision_gate_corrective_v1r.py","--spec",str(SPEC),"--run-dir",str(PROJECT_ROOT/"results/gate3_semantics"/RUN_ID)]
    write_json(SPEC,spec); print(CARD.relative_to(PROJECT_ROOT)); print(SPEC.relative_to(PROJECT_ROOT)); return 0


if __name__=="__main__": raise SystemExit(main())
