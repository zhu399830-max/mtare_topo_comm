#!/usr/bin/env python3
"""Freeze exact existing-cache export; metadata only, never create/execute run."""
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import platform
import subprocess

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import build_run_id, load_json, preflight, write_json
from mtare_topo.governance_field_recovery import selection_sha256
from mtare_topo.governance_partial_structure import (
    SCHEMA, METHOD, RAW_COUNTS, RESOURCES, RESTRICTIONS, PRODUCER_FILES,
    COORDINATE_COMMIT, TEACHER_COMMIT, validate_partial_structure_export_card,
)

SLUG="gse_partial_structure_export_v1"
CARD=f"configs/v3/gate3/data_cards/{SLUG}.json"
SPEC=f"configs/v3/gate3/{SLUG}.json"
COORDINATE_SPEC="configs/v3/gate3/gse_coordinate_control_v1.json"
TEACHER_SPEC="configs/v3/gate3/gse_supported_construction_teacher_v1r.json"
AUDIT_SPEC="configs/v3/gate3/gse_local_teacher_audit_v1.json"
METHOD_DOC="docs/GSE_PARTIAL_STRUCTURE_HEAD_SOFTWARE_V1.md"
SEAL_HASHES={"coordinate_control":"a40c9b386e366164ab98f73dabafcb678b143d8e723b4b5621a93f53c14836ba",
    "local_teacher":"69f17706af3a3dc36f3cd01ca7e9ae5d64b8d60fe906270133f50212c09e4642",
    "supported_teacher":"47d1d33cd116268405599205412d649a042a9a782acfa783a6b417e402c39a14"}


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def git_source_sha(commit,path):
    data=subprocess.run(["git","show",f"{commit}:{path}"],cwd=PROJECT_ROOT,check=True,capture_output=True).stdout
    return hashlib.sha256(data).hexdigest()


def selected_seal_entries(path, expected_paths, expected_digest):
    """Read an existing seal only; never open/resolve payloads mentioned in it."""
    data=path.read_bytes()
    if hashlib.sha256(data).hexdigest()!=expected_digest: raise ValueError("original producer seal drift")
    selected={}
    for line in data.decode().splitlines():
        digest,relative=line.split(None,1)
        if relative in expected_paths:
            if relative in selected: raise ValueError("duplicate selected seal entry")
            selected[relative]=digest
    if set(selected)!=set(expected_paths): raise ValueError("selected payload/metadata absent from original seal")
    return selected


def documents():
    old_specs={role:load_json(PROJECT_ROOT/path) for role,path in (
        ("coordinate_control",COORDINATE_SPEC),("local_teacher",AUDIT_SPEC),("supported_teacher",TEACHER_SPEC))}
    old_cards={role:load_json(PROJECT_ROOT/spec["data_card"]) for role,spec in old_specs.items()}
    previous=old_cards["coordinate_control"]
    roots={role:f"results/gate3_semantics/{build_run_id(spec)}" for role,spec in old_specs.items()}
    sources={name:{"path":roots[role]+"/artifacts/"+basename} for name,role,basename in (
        ("predictions","coordinate_control","all_predictions.npz"),
        ("scoring_targets","coordinate_control","existing_scoring_targets.npz"),
        ("sample_manifest","coordinate_control","sample_manifest.json"),
        ("identity_audit","local_teacher","observation_audit.json"),
        ("partial_targets","supported_teacher","observation_targets.json"))}
    seals={role:root+"/artifacts/evidence_sha256.txt" for role,root in roots.items()}
    bound={}
    for role,spec in old_specs.items():
        wanted={v["path"] for v in sources.values() if v["path"].startswith(roots[role]+"/")}
        wanted.update(roots[role]+"/config/"+name for name in ("run_spec.json","data_card.json"))
        entries=selected_seal_entries(PROJECT_ROOT/seals[role],wanted,SEAL_HASHES[role])
        bound[seals[role]]=SEAL_HASHES[role]
        for value in sources.values():
            if value["path"] in entries: value["sha256"]=entries[value["path"]];bound[value["path"]]=value["sha256"]
        specpath={"coordinate_control":COORDINATE_SPEC,"local_teacher":AUDIT_SPEC,"supported_teacher":TEACHER_SPEC}[role]
        for actual,snapshot in ((specpath,"run_spec.json"),(spec["data_card"],"data_card.json")):
            digest=sha(PROJECT_ROOT/actual)
            if digest!=entries[roots[role]+"/config/"+snapshot]: raise ValueError("original spec/card differs from sealed snapshot")
            bound[actual]=digest
    selection=selection_sha256(previous["selected_rows"])
    for original in old_cards.values():
        # The original local audit card predates an explicit selection digest.
        # Bind its actual rows, without inventing a field or changing that card.
        if (selection_sha256(original["selected_rows"])!=selection
                or ("selection_sha256" in original and original["selection_sha256"]!=selection)):
            raise ValueError("all three original cards must bind the identical180 selection")
    provenance={}
    for role,commit in (("coordinate_control",COORDINATE_COMMIT),("supported_teacher",TEACHER_COMMIT)):
        hashes={}
        for path in PRODUCER_FILES[role]:
            digest=git_source_sha(commit,path)
            if digest!=old_specs[role]["source_sha256"].get(path) or digest!=sha(PROJECT_ROOT/path):
                raise ValueError("producer Git bytes/current source/frozen spec disagree")
            hashes[path]=digest
        provenance[role]={"git_commit":commit,"spec":COORDINATE_SPEC if role=="coordinate_control" else TEACHER_SPEC,
                          "card":old_specs[role]["data_card"],"source_sha256":hashes}
    bound[METHOD_DOC]=sha(PROJECT_ROOT/METHOD_DOC)
    keys=("sampling_rule","license_or_allowed_use","partition","time_basis","duration_s","independent_sampling_unit",
          "observation_count","parent_count","unique_source_frame_count","frames_per_observation","worlds","selected_rows")
    card={key:deepcopy(previous[key]) for key in keys}
    purpose="Export loss-only partial structure targets from the identical180 C01 cached axis predictions/GT and supported construction labels; freeze actual GT/predicted effective denominators before separately authorized head training."
    card.update({"schema_version":SCHEMA,"operation":"data_export","card_id":SLUG,"purpose":purpose,
        "tasks":sorted({r["task"] for r in card["selected_rows"]}),"selection_sha256":selection,
        "legacy_node_count_inventory_only":100,"visible_fragment_count":1452,
        "teacher_source":"Sealed supported-construction V1R partial targets; original coordinate-control32-slot GT axes/mask; original local-teacher audit supplies exact row/source/frame identity, not forward features.",
        "identity_provenance":"Coordinate arrays embed no frame/source identities. Bind producer version/spec/card and its original exact frame parity; join cache task/row through sealed audit to source ID and exact frame rows. This is inherited provenance, not an independent embedded-frame-ID check.",
        "cache_embeds_frame_identity":False,"prediction_branch":"raw_coordinates","native_method":METHOD,
        "raw_teacher_counts":deepcopy(RAW_COUNTS),"effective_counts":{"gt":None,"predicted":None},"capacity_ready":False,
        "resources":dict(RESOURCES),"restrictions":dict.fromkeys(RESTRICTIONS,True),"sources":sources,
        "sealed_sources":bound,"source_seals":seals,"producer_provenance":provenance,
        "approval":{"status":"APPROVED","approved_by":"user","approved_at":"2026-09-05","scope":purpose,
            "authorized_operations":["data_export"],"authorized_gates":[3],"selection_sha256":selection,"sources":deepcopy(sources),
            "confirmation_reference":"Existing user-approved GSE-Graph composition implementation and parallel convergence plan, with standing in-scope authorization to choose efficient steps. No new user exchange is asserted. This card only authorizes the exact cached export, not training, model inference, new teacher generation or test access."}})
    report=validate_partial_structure_export_card(card)
    if not report.passed: raise ValueError(report.errors)
    spec={"schema_version":"v3_run_spec_v1","gate":3,"date":"20260905","slug":SLUG,"seed":0,
        "operation":"data_export","data_card":CARD,"config_path":CARD,"user_authorization":deepcopy(card["approval"]),
        "question":"Which cached partial structure labels remain uniquely attributable to all32 GT or predicted axes, without GT-filtering predictions or inventing terminal labels?",
        "method":"Exact five sealed cached sources; task/row identity audit bridge; mean Euclidean control-point unique assignment and orientation; unknown ambiguity preserved; no model or raw sensor reads.",
        "baseline":"Same supported targets mapped to original GT32 axes versus frozen final raw_coordinates32 axes; export denominators only, not trained model performance.",
        "fallback":"Any hash/identity/padding/population drift fails and seals once. Unknown global/orientation ties remain unknown; absent terminal capacity cannot become three-class PASS.",
        "native_method":METHOD,"wall_time_cap_s":120,
        "expected_counts":{"observations":180,"parents":10,"unique_frames":900,"visible_fragments":1452,**deepcopy(RAW_COUNTS)},
        "effective_counts":{"gt":None,"predicted":None},
        "estimated_cost":{"compute":"CPU only <=120s;0 model/checkpoint/optimizer/new sensor reads", "wall_time_hours":120/3600,
                          "disk_gb":.25,"host_ram_gb":4,"gpu_vram_gb":0},
        "acceptance_criteria":["Exact180 C01 observations/10parents/900source frames/1452fragments and raw partial counts; sealed identity joins, all32 prediction slots retained.",
            "GT and predicted direction maps/RegionTargets/actual effective denominator ledgers retained; all nonunique matches UNKNOWN,0terminal. No complete three-class or scientific PASS.",
            "Five cached payloads only;0model/checkpoint/rawsensor/optimizer/newteacher/test/graph;<=120s/4GiB/0.25GB, immutable single run and SHA256seal."],
        "expected_evidence":["Exact input card/spec/command/environment/source hashes, original axes and manifest, all180 GT/predicted mappings/targets/ledgers, full previews, logs, metrics, RUN_STATE and seal."],
        "expected_versions":{"python":platform.python_version(),**{name:importlib.metadata.version(name) for name in ("numpy","torch","zarr","scipy")}}}
    executable=old_specs["coordinate_control"]["command"][6]
    spec["command"]=["env","OMP_NUM_THREADS=1","OPENBLAS_NUM_THREADS=1","MKL_NUM_THREADS=1","PYTHONHASHSEED=0",executable,
        "tools/v3/run_gse_partial_structure_export_v1.py","--spec",str(PROJECT_ROOT/SPEC),"--run-dir",str(PROJECT_ROOT/"results/gate3_semantics"/build_run_id(spec))]
    spec["command_sha256"]=hashlib.sha256(json.dumps(spec["command"],separators=(",",":")).encode()).hexdigest()
    paths={str(p.relative_to(PROJECT_ROOT)) for p in (PROJECT_ROOT/"src").rglob("*.py")}
    paths.update({"tools/v3/run_gse_partial_structure_export_v1.py","tools/v3/freeze_gse_partial_structure_export_v1.py","tools/v3/_bootstrap.py"})
    paths.update(str(p.relative_to(PROJECT_ROOT)) for p in (PROJECT_ROOT/"tests/v3/unit").glob("test_gse_partial_structure*.py"))
    spec["source_sha256"]={p:sha(PROJECT_ROOT/p) for p in sorted(paths)}
    return card,spec


def main():
    if (PROJECT_ROOT/CARD).exists() or (PROJECT_ROOT/SPEC).exists(): raise RuntimeError("refuse overwrite of frozen export card/spec")
    card,spec=documents();write_json(PROJECT_ROOT/CARD,card);write_json(PROJECT_ROOT/SPEC,spec)
    report=preflight(spec,load_json(PROJECT_ROOT/"results/project_status.json"),PROJECT_ROOT)
    print(json.dumps({"passed":report.passed,"errors":report.errors,"warnings":report.warnings,"card":CARD,"spec":SPEC}))
    return int(not report.passed)


if __name__=="__main__": raise SystemExit(main())
