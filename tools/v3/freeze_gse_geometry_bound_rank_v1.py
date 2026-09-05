#!/usr/bin/env python3
"""Freeze only final-cache ranking diagnostics; metadata reads, no execution."""
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import platform

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import build_run_id,load_json,preflight,write_json
from mtare_topo.governance_geometry_bound_rank import (
    SCHEMA,SOURCE_NAMES,CORRECTIVE_SEAL_SHA256,RANK_POLICY,RESOURCES,RESTRICTIONS,validate_geometry_bound_rank_card,
)
from mtare_topo.governance_partial_structure_training import IDENTITY_FIELDS,EXPORT_SEAL_SHA256
from freeze_gse_partial_structure_export_v1 import selected_seal_entries

SLUG="gse_geometry_bound_rank_v1r"
CARD=f"configs/v3/gate3/data_cards/{SLUG}.json"
SPEC=f"configs/v3/gate3/{SLUG}.json"
CORRECTIVE_SPEC="configs/v3/gate3/gse_geometry_bound_training_v1.json"
METHOD_DOC="docs/GSE_GEOMETRY_BOUND_RANK_V1.md"


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def documents():
    corrective=load_json(PROJECT_ROOT/CORRECTIVE_SPEC);old=load_json(PROJECT_ROOT/corrective["data_card"])
    root="results/gate3_semantics/"+build_run_id(corrective);seal=root+"/artifacts/evidence_sha256.txt"
    sources=deepcopy(old["sources"])
    newkeys=("gt_final","predicted_final","no_relations_final","corrective_summary")
    for key in newkeys:sources[key]={"path":root+("/metrics/" if key=="corrective_summary" else "/artifacts/")+SOURCE_NAMES[key]}
    wanted={sources[key]["path"] for key in newkeys}|{root+"/config/run_spec.json",root+"/config/data_card.json"}
    entries=selected_seal_entries(PROJECT_ROOT/seal,wanted,CORRECTIVE_SEAL_SHA256)
    bound={seal:CORRECTIVE_SEAL_SHA256}
    for key in newkeys:
        source=sources[key];source["sha256"]=entries[source["path"]];bound[source["path"]]=source["sha256"]
    for path,snapshot in ((CORRECTIVE_SPEC,"run_spec.json"),(corrective["data_card"],"data_card.json")):
        digest=sha(PROJECT_ROOT/path)
        if digest!=entries[root+"/config/"+snapshot]:raise ValueError("corrective metadata snapshot drift")
        bound[path]=digest
    exportseal=old["export_reference"]["seal"]
    exported=selected_seal_entries(PROJECT_ROOT/exportseal,{v["path"] for v in old["sources"].values()},EXPORT_SEAL_SHA256)
    bound[exportseal]=EXPORT_SEAL_SHA256
    for source in old["sources"].values():
        if exported[source["path"]]!=source["sha256"]:raise ValueError("original exported source drift")
        bound[source["path"]]=source["sha256"]
    bound[METHOD_DOC]=sha(PROJECT_ROOT/METHOD_DOC)
    purpose="Determine whether final geometry-bound member/junction probabilities rank positives above negatives despite fixed-threshold collapse; only eight existing caches, unchanged unique-center scoring population, no threshold search or training."
    card={key:deepcopy(old[key]) for key in (*IDENTITY_FIELDS,"effective_counts")}
    card.update({"schema_version":SCHEMA,"operation":"data_export","card_id":SLUG,"purpose":purpose,
        "teacher_source":"Unchanged four exported input/target/manifest/summary files plus three final corrective prediction caches and its summary; corrective_card.base_training_card is identity/reader evidence only.",
        "scope_limitations":"Same-sample final-cache ranking diagnostic, not calibrated detection or generalization. Fixed0.5 membership and original event argmax must reproduce; tie-aware AP/AUROC and final BCE versus empirical constant prior are descriptive, with single-class AP/AUROC null. No threshold selection or new optimization.",
        "corrective_card":deepcopy(old),"corrective_reference":{"spec":CORRECTIVE_SPEC,"card":corrective["data_card"],"seal":seal},
        "sources":sources,"sealed_sources":bound,"rank_policy":deepcopy(RANK_POLICY),"resources":dict(RESOURCES),
        "restrictions":dict.fromkeys(RESTRICTIONS,True),"scientific_gate_pass":False,
        "approval":{"status":"APPROVED","approved_by":"user","approved_at":"2026-09-05","scope":purpose,
            "authorized_operations":["data_export"],"authorized_gates":[3],"selection_sha256":old["selection_sha256"],
            "sources":deepcopy(sources),"rank_policy":deepcopy(RANK_POLICY),
            "confirmation_reference":"Existing user-approved GSE-Graph plan and standing scoped authorization cover this zero-training diagnosis following the completed corrective. No new conversation is invented; this independent data_export card permits only eight cached payloads and fixed ranking analysis, not old training authority or threshold tuning."}})
    report=validate_geometry_bound_rank_card(card)
    if not report.passed:raise ValueError(report.errors)
    spec={"schema_version":"v3_run_spec_v1","gate":3,"date":"20260905","slug":SLUG,"seed":0,"operation":"data_export",
        "data_card":CARD,"config_path":CARD,"user_authorization":deepcopy(card["approval"]),
        "question":"Does fixed0.5 all-negative collapse hide usable probability ordering, or are final member/junction outputs no better than constant priors?",
        "method":"Three final180 cached outputs, existing unique_center_query/scored_member_mask; tie-grouped AP/half-credit AUROC,global and parent positive/negative distributions and final-logit BCE versus constant priors. Original0.5/argmax reproduced, no thresholds selected.",
        "baseline":"Empirical constant-prior probability diagnostics plus exact already sealed0.5/argmax scoring. Single-class AP/AUROC explicitly null; prior estimates are descriptive only, never fed into calibration/training.",
        "fallback":"Hash/identity/finite-score/mask/original-count or fixed-score drift fails and seals once. No history/model/checkpoint/optimizer/threshold search; ranking improvement is not scientific PASS.",
        "rank_policy":deepcopy(RANK_POLICY),"wall_time_cap_s":300,
        "expected_counts":{"observations":180,"parents":10,"unique_frames":900,"visible_fragments":1452,
            "cached_prediction_observations":540,"centers_per_branch":1206,"events_per_branch":886,
            "positive_members_per_branch":2152,"negative_members_per_branch":13489,"model_windows":0,"checkpoint_reads":0,"optimizer_steps":0},
        "estimated_cost":{"compute":"CPU-only <=300s/8cachedfiles/540observations;0model/history/optimizer", "wall_time_hours":300/3600,"host_ram_gb":4,"gpu_vram_gb":0,"disk_gb":.5},
        "acceptance_criteria":["Exact180C01/900frames/1452fragments and final540cached outputs; same8sealed sources and original unique-center/member masks,1206centers886events2152positive13489negative per branch.",
            "Original0.5member and eventargmax scoring reproduced; tie-aware AP/AUROC and probability/BCE diagnostics global/per-parent,undefined single-class scores null. No selected thresholds or changed population.",
            "0history/model/checkpoint/optimizer/newscan/newlabels/C02-C10/graph/scientificPASS;300s4GiB.5GB; full diagnostics/previews/logs/env/state/seal,never overwrite/retry."],
        "expected_evidence":["Eight source hashes/identities, original scoring reproduction, final member/junction ranks/positive-negative probabilities/BCE-prior comparisons perbranch/parent, complete plots/logs/environment/RUN_STATE/seal."],
        "expected_versions":{"python":platform.python_version(),**{name:importlib.metadata.version(name) for name in ("numpy","torch","zarr","scipy")}}}
    executable=corrective["command"][6]
    spec["command"]=["env","CUDA_VISIBLE_DEVICES=","OMP_NUM_THREADS=1","OPENBLAS_NUM_THREADS=1","MKL_NUM_THREADS=1","PYTHONHASHSEED=0",executable,
        "tools/v3/run_gse_geometry_bound_rank_v1.py","--spec",str(PROJECT_ROOT/SPEC),"--run-dir",str(PROJECT_ROOT/"results/gate3_semantics"/build_run_id(spec))]
    spec["command_sha256"]=hashlib.sha256(json.dumps(spec["command"],separators=(",",":")).encode()).hexdigest()
    paths={str(p.relative_to(PROJECT_ROOT)) for p in (PROJECT_ROOT/"src").rglob("*.py")}
    paths.update({"tools/v3/run_gse_geometry_bound_rank_v1.py","tools/v3/freeze_gse_geometry_bound_rank_v1.py",
        "tools/v3/freeze_gse_partial_structure_export_v1.py","tools/v3/run_gse_partial_structure_training_v1.py",
        "tools/v3/run_gse_supported_construction_teacher_v1.py","tools/v3/_bootstrap.py",
        "tools/v3/run_gse_assignment_attribution_v1.py","tests/v3/unit/test_gse_binary_ranking.py",
        "tests/v3/unit/test_gse_assignment_runner.py"})
    paths.update(str(p.relative_to(PROJECT_ROOT)) for p in (PROJECT_ROOT/"tests/v3/unit").glob("test_gse_geometry_bound_rank*.py"))
    spec["source_sha256"]={p:sha(PROJECT_ROOT/p) for p in sorted(paths)}
    return card,spec


def main():
    if (PROJECT_ROOT/CARD).exists() or (PROJECT_ROOT/SPEC).exists():raise RuntimeError("refuse overwrite of rank card/spec")
    card,spec=documents();write_json(PROJECT_ROOT/CARD,card);write_json(PROJECT_ROOT/SPEC,spec)
    report=preflight(spec,load_json(PROJECT_ROOT/"results/project_status.json"),PROJECT_ROOT)
    print(json.dumps({"passed":report.passed,"errors":report.errors,"warnings":report.warnings,"card":CARD,"spec":SPEC}))
    return int(not report.passed)


if __name__=="__main__":raise SystemExit(main())
