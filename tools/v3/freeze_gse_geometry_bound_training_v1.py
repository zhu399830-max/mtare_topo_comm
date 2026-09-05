#!/usr/bin/env python3
"""Prepare the single loss-binding correction; metadata only, never run it."""
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import platform

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import build_run_id,load_json,preflight,write_json
from mtare_topo.governance_geometry_bound import (
    SCHEMA,LOSS_CONTRACT,SHARED_FIELDS,RESTRICTIONS,ATTRIBUTION_SEAL_SHA256,validate_geometry_bound_training_card,
)
from mtare_topo.governance_partial_structure_training import EXPORT_SEAL_SHA256
from mtare_topo.governance_assignment_attribution import TRAINING_SEAL_SHA256
from freeze_gse_partial_structure_export_v1 import selected_seal_entries

SLUG="gse_geometry_bound_training_v1"
CARD=f"configs/v3/gate3/data_cards/{SLUG}.json"
SPEC=f"configs/v3/gate3/{SLUG}.json"
BASE_SPEC="configs/v3/gate3/gse_partial_structure_training_v1.json"
ATTRIBUTION_SPEC="configs/v3/gate3/gse_assignment_attribution_v1.json"
METHOD_DOC="docs/GSE_GEOMETRY_BOUND_TRAINING_V1.md"
UNCHANGED_SOURCE_FILES=("src/mtare_topo/representation/gse_region_queries.py",
    "src/mtare_topo/representation/gse_partial_structure_training.py",
    "src/mtare_topo/evaluation/gse_partial_structure.py","src/mtare_topo/data/gse_partial_training_inputs.py")


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def documents():
    baseline=load_json(PROJECT_ROOT/BASE_SPEC);base=load_json(PROJECT_ROOT/baseline["data_card"])
    unchanged={path:sha(PROJECT_ROOT/path) for path in UNCHANGED_SOURCE_FILES}
    if any(baseline.get("source_sha256",{}).get(path)!=digest for path,digest in unchanged.items()):
        raise ValueError("original head/loss/training/evaluation/loader source changed")
    attribution=load_json(PROJECT_ROOT/ATTRIBUTION_SPEC)
    bound={};references={}
    for name,specpath,source,sealhash in (("base_reference",BASE_SPEC,baseline,TRAINING_SEAL_SHA256),
            ("attribution_reference",ATTRIBUTION_SPEC,attribution,ATTRIBUTION_SEAL_SHA256)):
        root="results/gate3_semantics/"+build_run_id(source)
        seal=root+"/artifacts/evidence_sha256.txt"
        entries=selected_seal_entries(PROJECT_ROOT/seal,{root+"/config/run_spec.json",root+"/config/data_card.json"},sealhash)
        bound[seal]=sealhash
        for path,snapshot in ((specpath,"run_spec.json"),(source["data_card"],"data_card.json")):
            digest=sha(PROJECT_ROOT/path)
            if digest!=entries[root+"/config/"+snapshot]:raise ValueError("completed metadata snapshot drift")
            bound[path]=digest
        references[name]={"spec":specpath,"card":source["data_card"],"seal":seal}
    exportseal=base["export_reference"]["seal"]
    exported=selected_seal_entries(PROJECT_ROOT/exportseal,{v["path"] for v in base["sources"].values()},EXPORT_SEAL_SHA256)
    bound[exportseal]=EXPORT_SEAL_SHA256
    for source in base["sources"].values():
        if exported[source["path"]]!=source["sha256"]:raise ValueError("unchanged exported input hash drift")
        bound[source["path"]]=source["sha256"]
    bound[METHOD_DOC]=sha(PROJECT_ROOT/METHOD_DOC)
    purpose="One single-variable cached-head correction: geometry-only unique center binding for all five original loss terms, unchanged data/head/weights/scales/init/order/300steps per branch; no old checkpoint warm start."
    card={key:deepcopy(base[key]) for key in SHARED_FIELDS}
    card.update({"schema_version":SCHEMA,"operation":"training","card_id":SLUG,"purpose":purpose,
        "teacher_source":"Unchanged four sealed partial export payloads, same original partial labels/unknowns and correspondence. Baseline training and attribution are metadata evidence only; old weights/predictions are not read.",
        "scope_limitations":"Only loss-to-candidate binding changes. Known partial targets require centers; nonunique geometry matches are unknown for every loss, fixed original denominators. Empty supervised batch fails. Same-sample partial diagnostic, not complete classification/detection/generalization/scientific PASS.",
        "base_training_card":deepcopy(base),**references,**deepcopy(LOSS_CONTRACT),"sealed_sources":bound,
        "restrictions":dict.fromkeys(RESTRICTIONS,True),"capacity_ready":False,"scientific_gate_pass":False,
        "approval":{"status":"APPROVED","approved_by":"user","approved_at":"2026-09-05","scope":purpose,
            "authorized_operations":["training"],"authorized_gates":[3],"selection_sha256":base["selection_sha256"],
            "sources":deepcopy(base["sources"]),"loss_contract":deepcopy(LOSS_CONTRACT),
            "confirmation_reference":"Standing user authorization for the approved GSE-Graph plan covers the publicly selected single-variable correction after sealed assignment attribution. No new user exchange is fabricated; this independent training card binds the unchanged four data files and the new geometry-only loss rule, not a reuse of old training permission."}})
    report=validate_geometry_bound_training_card(card)
    if not report.passed:raise ValueError(report.errors)
    spec={"schema_version":"v3_run_spec_v1","gate":3,"date":"20260905","slug":SLUG,"seed":0,"operation":"training",
        "data_card":CARD,"config_path":CARD,"user_authorization":deepcopy(card["approval"]),
        "question":"Does binding every loss to the unique geometric center candidate fix the observed remote-class-candidate assignment conflict, with every other training/evaluation setting held fixed?",
        "method":"Geometry-only unique center assignment before all five unchanged original loss terms; unknown ties receive no loss with fixed denominators; partial known labels require centers,empty-supervision batchFAIL. Original three heads/seed0/hidden64/Adam.001/B18/300steps each.",
        "baseline":"Original sealed partial structure training uses joint label-conditioned assignment. It remains immutable; comparison uses original summary evidence, not loading its weights or predictions into this run.",
        "fallback":"Hash/population/known-center/finite-gradient/empty-batch/budget drift FAIL and seal once. No extra epochs,loss reweighting,threshold search,checkpoint selection or scientific PASS.",
        "training":deepcopy(base["training"]),"evaluation":deepcopy(base["evaluation"]),**deepcopy(LOSS_CONTRACT),
        "wall_time_cap_s":1800,"expected_counts":deepcopy(baseline["expected_counts"]),"immutable_baseline_source_sha256":unchanged,
        "estimated_cost":{"compute":"One CUDA0 <=1800s single-variable correction;900head updates1080initial/final eval;0backbone/oldcheckpoint/newscan", "wall_time_hours":.5,
            "host_ram_gb":4,"gpu_vram_gb":4,"disk_gb":.5},
        "acceptance_criteria":["Same four inputs/180C01/900frames/1452fragments and all partial raw/effective denominators; original model/head/loss weights/scales unchanged.",
            "Only geometry-only unique center binding changes; all five losses use it; ambiguous assignments unknown with original fixed denominators,known partial labels need centers,empty supervised batchFAIL.",
            "Three seed0heads same initialization/schedule/B18/Adam.001/300steps each;900updates1080initial-final all180 evaluations; final-only independent center score and unchanged0.5,0backbone/newscan/oldcheckpoint.",
            "Preserve all weights/predictions/matches/loss components/unknowns/full previews/logs/env/state/seal;1800s4GiBhost/GPU.5GB; no overwrite/retry or scientificPASS."],
        "expected_evidence":["Exact source/method binding,shared initial state/order,900updates plus geometry-match supervision ledger and unknown counts,3finalweights,allinitial-final independent scores/previews,rawlogs/environment/RUN_STATE/SHAseal."],
        "expected_versions":{"python":platform.python_version(),**{name:importlib.metadata.version(name) for name in ("numpy","torch","zarr","scipy")}}}
    executable=baseline["command"][6]
    spec["command"]=["env","CUBLAS_WORKSPACE_CONFIG=:4096:8","OMP_NUM_THREADS=1","OPENBLAS_NUM_THREADS=1","MKL_NUM_THREADS=1","PYTHONHASHSEED=0",executable,
        "tools/v3/run_gse_geometry_bound_training_v1.py","--spec",str(PROJECT_ROOT/SPEC),"--run-dir",str(PROJECT_ROOT/"results/gate3_semantics"/build_run_id(spec))]
    spec["command_sha256"]=hashlib.sha256(json.dumps(spec["command"],separators=(",",":")).encode()).hexdigest()
    paths={str(p.relative_to(PROJECT_ROOT)) for p in (PROJECT_ROOT/"src").rglob("*.py")}
    paths.update({"tools/v3/run_gse_geometry_bound_training_v1.py","tools/v3/freeze_gse_geometry_bound_training_v1.py",
        "tools/v3/freeze_gse_partial_structure_export_v1.py","tools/v3/run_gse_partial_structure_training_v1.py",
        "tools/v3/run_gse_supported_construction_teacher_v1.py","tools/v3/_bootstrap.py"})
    paths.update(str(p.relative_to(PROJECT_ROOT)) for p in (PROJECT_ROOT/"tests/v3/unit").glob("test_gse_geometry_bound*.py"))
    spec["source_sha256"]={p:sha(PROJECT_ROOT/p) for p in sorted(paths)}
    return card,spec


def main():
    if (PROJECT_ROOT/CARD).exists() or (PROJECT_ROOT/SPEC).exists():raise RuntimeError("refuse overwrite of correction card/spec")
    card,spec=documents();write_json(PROJECT_ROOT/CARD,card);write_json(PROJECT_ROOT/SPEC,spec)
    report=preflight(spec,load_json(PROJECT_ROOT/"results/project_status.json"),PROJECT_ROOT)
    print(json.dumps({"passed":report.passed,"errors":report.errors,"warnings":report.warnings,"card":CARD,"spec":SPEC}))
    return int(not report.passed)


if __name__=="__main__":raise SystemExit(main())
