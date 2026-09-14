"""Input-only30-observation material scope. No teacher/annotation authority."""
import hashlib
import json
from pathlib import Path
import re
from mtare_topo.governance_identity_inventory import FAMILIES, VARIANTS
from mtare_topo.governance_surface_selection import digest
from mtare_topo.governance_partial_structure_training import _exact

SCHEMA="v3_surface_observed_material_card_v1"
SOURCE="results/gate3_semantics/gate3_20260907_gse_surface_input_export_v1r_seed20260906"
SOURCE_SEAL="5e7c150959fb4eebfe837ea05015c0de8aa27b6e8fa03991c15c2e98a323c88b"
PARENTS=sorted(f+"_C01" for f in FAMILIES)
TASKS=sorted(p+"__"+v for p in PARENTS for v in VARIANTS)
POLICY={"edge_selection_rank":0,"surface_voxel_m":.5,"surface_radius_m":10.,"max_patches":4096,
        "neighbor_count":8,"ray_voxel_m":.25,"ray_cube_m":[-10.,10.],"labels":0,"model_windows":0}
COUNTS={"parents":10,"physical_edges":10,"tasks":30,"observations_processed":30,"frames_processed":150,
        "container_observations":480,"container_frames":2400,"split":"fit","new_labels":0}
RESOURCES={"wall_time_s":600,"host_ram_bytes":4294967296,"gpu_bytes":0,"output_bytes":536870912}


def read_pinned(root,relative,h):
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("explicit project relative input required")
    path=root/relative
    if path.resolve(strict=True)!=path:raise ValueError("input symlink forbidden")
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=h:raise ValueError("pinned input changed:"+relative)
    return raw


def compile_scope(root):
    root=Path(root).resolve(strict=True)
    seal=read_pinned(root,SOURCE+"/artifacts/evidence_sha256.txt",SOURCE_SEAL)
    allowed={SOURCE+"/artifacts/input_manifest.json"}|{SOURCE+"/artifacts/inputs/"+t+".npz" for t in TASKS}
    hashes={}
    for line in seal.decode().splitlines():
        h,p=line.split("  ",1)
        if p in allowed:
            if p in hashes:raise ValueError("duplicate sealed material input")
            hashes[p]=h
    if set(hashes)!=allowed:raise ValueError("exact31 input paths absent from completed export seal")
    mp=SOURCE+"/artifacts/input_manifest.json"
    manifest=json.loads(read_pinned(root,mp,hashes[mp]))
    rows={t:[] for t in TASKS}
    for r in manifest["observations"]:
        if r["task"] in rows:rows[r["task"]].append(r)
    selected=[]
    for task in TASKS:
        candidates=[(i,r) for i,r in enumerate(rows[task]) if r["edge_selection_rank"]==0]
        if len(rows[task])!=16 or len(candidates)!=1:raise ValueError("fixed rank0 selection not unique")
        i,r=candidates[0]
        if r["split"]!="fit" or r["parent_id"] not in PARENTS or len(r["frame_rows"])!=5:
            raise ValueError("material source scope drift")
        selected.append({"task":task,"input_path":SOURCE+"/artifacts/inputs/"+task+".npz",
                         "input_row":i,"source":r,"view_id":"obs_"+digest([task,r["source_sequence_id"]])[:16]})
    if len({r["view_id"] for r in selected})!=len(selected):raise ValueError("view identity collision")
    scope={"source_seal":{"path":SOURCE+"/artifacts/evidence_sha256.txt","sha256":SOURCE_SEAL},
           "input_files_sha256":hashes,"selected":selected,"counts":COUNTS,"policy":POLICY,"resources":RESOURCES,
           "spacing":"Original sealed causal fiveframe histories and decision_route_arc_m; no acquisition clock or newly asserted history spacing.",
           "teacher":"None. Observed material only; physical root labels paused; no manual/AI labels or targets.",
           "leakage":"Only C01 rank0 fixed observation per parent x3 variants; no score selection.10parent units,not30independent places. Does not replace full3360population; no C07-C10,construction,absolute pose,models or checkpoints."}
    return scope


def validate_surface_material_card(card):
    from mtare_topo.governance import ValidationReport
    errors=[]
    if type(card) is not dict or set(card)!={"schema_version","card_id","operation","scope","scope_sha256","approval"}:
        return ValidationReport(False,("closed observed-material card required",))
    if (card["schema_version"],card["card_id"],card["operation"])!=(SCHEMA,"gse_surface_observed_material_v1","data_export"):
        errors.append("input-only material operation required")
    s=card["scope"]
    if type(s) is not dict or set(s)!={"source_seal","input_files_sha256","selected","counts","policy","resources","spacing","teacher","leakage"}:
        return ValidationReport(False,("closed exact material scope required",))
    if (not _exact(s["counts"],COUNTS) or not _exact(s["policy"],POLICY) or not _exact(s["resources"],RESOURCES)
        or s["source_seal"]!={"path":SOURCE+"/artifacts/evidence_sha256.txt","sha256":SOURCE_SEAL}):
        errors.append("fixed population/observed-only policy/resources/source required")
    expected={SOURCE+"/artifacts/input_manifest.json"}|{SOURCE+"/artifacts/inputs/"+t+".npz" for t in TASKS}
    h=s["input_files_sha256"]
    if type(h) is not dict or set(h)!=expected or any(type(v) is not str or re.fullmatch("[a-f0-9]{64}",v) is None for v in h.values()):
        errors.append("exact31 sealed inputs required")
    selections=s["selected"]
    if type(selections) is not list or len(selections)!=30:
        errors.append("exact30 records required")
    else:
        for index,r in enumerate(selections):
            if (type(r) is not dict or set(r)!={"task","input_path","input_row","source","view_id"}
                or r.get("task")!=TASKS[index] or r.get("input_path")!=SOURCE+"/artifacts/inputs/"+TASKS[index]+".npz"
                or type(r.get("input_row")) is not int or not 0<=r["input_row"]<16
                or type(r.get("source")) is not dict or r["source"].get("edge_selection_rank")!=0):
                errors.append("fixed selected record drift");break
    for k in ("spacing","teacher","leakage"):
        if type(s[k]) is not str or not s[k]:errors.append(k+" disclosure required")
    try:sh=digest(s)
    except (TypeError,ValueError):return ValidationReport(False,("finite JSON scope required",))
    a=card["approval"]
    if (type(a) is not dict or a.get("status")!="APPROVED" or a.get("scope_sha256")!=sh
        or a.get("authorized_operations")!=["data_export"] or not _exact(a.get("authorized_gates"),[3])
        or card["scope_sha256"]!=sh):errors.append("exact input-export standing authorization required")
    for k in ("approved_by","approved_at","scope","confirmation_reference"):
        if type(a) is not dict or type(a.get(k)) is not str or not a[k]:errors.append("approval."+k+" required")
    return ValidationReport(not errors,tuple(errors))
