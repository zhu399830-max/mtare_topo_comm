#!/usr/bin/env python3
"""Metadata-only freeze for cached assignment diagnosis; never execute it."""
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import platform

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import build_run_id,load_json,preflight,write_json
from mtare_topo.governance_assignment_attribution import (
    SCHEMA,SOURCE_NAMES,TRAINING_SEAL_SHA256,ATTRIBUTION,RESOURCES,RESTRICTIONS,
    validate_assignment_attribution_card,
)
from mtare_topo.governance_partial_structure_training import IDENTITY_FIELDS,EXPORT_SEAL_SHA256
from freeze_gse_partial_structure_export_v1 import selected_seal_entries

SLUG="gse_assignment_attribution_v1"
CARD=f"configs/v3/gate3/data_cards/{SLUG}.json"
SPEC=f"configs/v3/gate3/{SLUG}.json"
TRAINING_SPEC="configs/v3/gate3/gse_partial_structure_training_v1.json"
METHOD_DOC="docs/GSE_ASSIGNMENT_ATTRIBUTION_V1.md"


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def documents():
    training=load_json(PROJECT_ROOT/TRAINING_SPEC);old=load_json(PROJECT_ROOT/training["data_card"])
    root="results/gate3_semantics/"+build_run_id(training)
    seal=root+"/artifacts/evidence_sha256.txt"
    sources=deepcopy(old["sources"])
    for key in ("gt_final","predicted_final","no_relations_final","history","training_summary"):
        sources[key]={"path":root+("/metrics/" if key=="training_summary" else "/artifacts/")+SOURCE_NAMES[key]}
    wanted={sources[k]["path"] for k in ("gt_final","predicted_final","no_relations_final","history","training_summary")}
    wanted.update((root+"/config/run_spec.json",root+"/config/data_card.json"))
    entries=selected_seal_entries(PROJECT_ROOT/seal,wanted,TRAINING_SEAL_SHA256)
    bound={seal:TRAINING_SEAL_SHA256}
    for key in ("gt_final","predicted_final","no_relations_final","history","training_summary"):
        path=sources[key]["path"];sources[key]["sha256"]=entries[path];bound[path]=entries[path]
    for path,snapshot in ((TRAINING_SPEC,"run_spec.json"),(training["data_card"],"data_card.json")):
        digest=sha(PROJECT_ROOT/path)
        if digest!=entries[root+"/config/"+snapshot]:raise ValueError("completed training spec/card snapshot drift")
        bound[path]=digest
    export_seal=old["export_reference"]["seal"]
    exported=selected_seal_entries(PROJECT_ROOT/export_seal,{v["path"] for v in old["sources"].values()},EXPORT_SEAL_SHA256)
    bound[export_seal]=EXPORT_SEAL_SHA256
    for source in old["sources"].values():
        if exported[source["path"]]!=source["sha256"]:raise ValueError("original exported source digest drift")
        bound[source["path"]]=source["sha256"]
    bound[METHOD_DOC]=sha(PROJECT_ROOT/METHOD_DOC)
    purpose="Read-only assignment attribution on three final180 cached predictions: compare original label-conditioned joint matching with independent center-geometric evaluation and inspect last10 B18 training batches; no model/checkpoint/optimizer or threshold search."
    card={key:deepcopy(old[key]) for key in (*IDENTITY_FIELDS,"effective_counts")}
    card.update({"schema_version":SCHEMA,"operation":"data_export","card_id":SLUG,"purpose":purpose,
        "teacher_source":"Unmodified completed partial export plus fixed300-step final prediction caches and training history. Training card is identity/structure evidence only, not new training authority.",
        "scope_limitations":"Joint matching conditions on target labels and is diagnostic, not deployment/scientific scoring. No new model, threshold, label or old-result changes. Matching mismatch remains an unconfirmed hypothesis until this read-only audit.",
        "training_card":deepcopy(old),"training_reference":{"spec":TRAINING_SPEC,"card":training["data_card"],"seal":seal},
        "sources":sources,"sealed_sources":bound,"attribution":deepcopy(ATTRIBUTION),"resources":dict(RESOURCES),
        "restrictions":dict.fromkeys(RESTRICTIONS,True),"scientific_gate_pass":False,
        "approval":{"status":"APPROVED","approved_by":"user","approved_at":"2026-09-05","scope":purpose,
            "authorized_operations":["data_export"],"authorized_gates":[3],"selection_sha256":old["selection_sha256"],"sources":deepcopy(sources),
            "confirmation_reference":"Existing user-approved GSE-Graph plan and standing in-scope authorization allow this bounded diagnosis of the completed experiment. No new user exchange is asserted; separate data_export authority binds only nine cached files, not old training authority or any new training/test access."}})
    report=validate_assignment_attribution_card(card)
    if not report.passed:raise ValueError(report.errors)
    spec={"schema_version":"v3_run_spec_v1","gate":3,"date":"20260905","slug":SLUG,"seed":0,"operation":"data_export",
        "data_card":CARD,"config_path":CARD,"user_authorization":deepcopy(card["approval"]),
        "question":"Does target-conditioned training assignment differ from independent geometric scoring, and can existing final predictions/last-epoch matches explain member/event failure without another training run?",
        "method":"Exact nine sealed cached files; original region_set_losses joint assignment versus original center-only independent evaluation at unchanged0.5; last10 B18 batches must partition180 once per branch.0model/checkpoint/optimizer.",
        "baseline":"Completed independent scoring is reproduced unchanged; joint target-conditioned matching is diagnostic only and never substitutes for model success.",
        "fallback":"Any seal/identity/count/history partition or original-score drift fails and seals once; geometric ambiguity stays UNKNOWN; joint solver representatives are explicitly not uniqueness-verified. No threshold search, score rewrite, training expansion or scientific PASS.",
        "attribution":deepcopy(ATTRIBUTION),"wall_time_cap_s":300,
        "expected_counts":{"observations":180,"parents":10,"unique_frames":900,"visible_fragments":1452,
            "cached_prediction_observations":540,"centers_per_branch":1206,"events_per_branch":886,
            "positive_members_per_branch":2152,"negative_members_per_branch":13489,"last_epoch_batches":10,"batch_size":18,
            "model_windows":0,"checkpoint_reads":0,"optimizer_steps":0},
        "estimated_cost":{"compute":"CPU only <=300s;540cached observation outputs,0model/checkpoint/optimizer", "wall_time_hours":300/3600,
            "host_ram_gb":4,"gpu_vram_gb":0,"disk_gb":.5},
        "acceptance_criteria":["Exact original180/900/1452 and three final180 caches,1206centers/886events/2152positive/13489negative per branch;9files only, all hashes and identities checked.",
            "Last10 B18 training batches form exact180 partition; reproduce original independent evaluation at0.5; compare unchanged original joint loss assignments without presenting label-conditioned scores as science.",
            "0model/checkpoint/optimizer/threshold search/newlabels/newscans/C02-C10/graph;<=300s/4GiB/.5GB; immutable once, full attribution records/unknowns/previews/logs/state/SHA256seal."],
        "expected_evidence":["Exact card/spec/9input hashes/environment/command, all540cached correspondence attribution and10last-batch records per branch, original-versus-joint diagnostic metrics, previews, logs, RUN_STATE and seal."],
        "expected_versions":{"python":platform.python_version(),**{name:importlib.metadata.version(name) for name in ("numpy","torch","zarr","scipy")}}}
    executable=training["command"][6]
    spec["command"]=["env","OMP_NUM_THREADS=1","OPENBLAS_NUM_THREADS=1","MKL_NUM_THREADS=1","PYTHONHASHSEED=0","CUDA_VISIBLE_DEVICES=",executable,
        "tools/v3/run_gse_assignment_attribution_v1.py","--spec",str(PROJECT_ROOT/SPEC),"--run-dir",str(PROJECT_ROOT/"results/gate3_semantics"/build_run_id(spec))]
    spec["command_sha256"]=hashlib.sha256(json.dumps(spec["command"],separators=(",",":")).encode()).hexdigest()
    paths={str(p.relative_to(PROJECT_ROOT)) for p in (PROJECT_ROOT/"src").rglob("*.py")}
    paths.update({"tools/v3/run_gse_assignment_attribution_v1.py","tools/v3/freeze_gse_assignment_attribution_v1.py",
                  "tools/v3/freeze_gse_partial_structure_export_v1.py","tools/v3/_bootstrap.py"})
    paths.update(str(p.relative_to(PROJECT_ROOT)) for p in (PROJECT_ROOT/"tests/v3/unit").glob("test_gse_assignment*.py"))
    spec["source_sha256"]={p:sha(PROJECT_ROOT/p) for p in sorted(paths)}
    return card,spec


def main():
    if (PROJECT_ROOT/CARD).exists() or (PROJECT_ROOT/SPEC).exists():raise RuntimeError("refuse overwrite of attribution card/spec")
    card,spec=documents();write_json(PROJECT_ROOT/CARD,card);write_json(PROJECT_ROOT/SPEC,spec)
    report=preflight(spec,load_json(PROJECT_ROOT/"results/project_status.json"),PROJECT_ROOT)
    print(json.dumps({"passed":report.passed,"errors":report.errors,"warnings":report.warnings,"card":CARD,"spec":SPEC}))
    return int(not report.passed)


if __name__=="__main__":raise SystemExit(main())
