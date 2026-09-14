"""Metadata-only plan for the fixed30 observed-material teacher inputs.

Does not read construction JSON, codebook, compressed arrays, or write labels.
This plan is not an approved data card or experiment execution permission.
"""
import json
import math
from pathlib import Path
import re

from mtare_topo.governance_surface_material import compile_scope, read_pinned
from mtare_topo.governance_surface_input import SEALS
from mtare_topo.governance_identity_inventory import P1A


FIELDS = {"sensor_xyz_m": ([3],"<f8",8), "yaw_deg": ([],"<f8",8),
          "primitive_membership_code": ([16,720],"<u2",2)}


def plan_field(header, field, rows, count):
    tail,dtype,itemsize = FIELDS[field]
    shape,chunks = header.get("shape"),header.get("chunks")
    if (shape != [count]+tail or type(chunks) is not list or len(chunks)!=len(shape)
        or any(type(x) is not int or x<=0 for x in chunks) or chunks[1:]!=tail
        or header.get("dtype")!=dtype or header.get("zarr_format")!=2
        or header.get("order")!="C" or header.get("dimension_separator",".")!="."):
        raise ValueError("teacher source header mismatch")
    if type(rows) is not list or len(rows)!=5 or any(type(x) is not int or not 0<=x<count for x in rows) or sorted(set(rows))!=rows:
        raise ValueError("exact causal five selected frames required")
    if chunks[0] != (16 if field=="primitive_membership_code" else min(4096,count)):
        raise ValueError("source chunk contract drift")
    indices = sorted({r//chunks[0] for r in rows})
    decoded = sum(min(count,(i+1)*chunks[0])-i*chunks[0] for i in indices)
    padded = len(indices)*math.prod(chunks)*itemsize
    if padded > 32*1024**2:
        raise ValueError("per-field decoded allocation exceeds32MiB")
    return {"shape":shape,"chunks":chunks,"dtype":dtype,"selected_rows":rows,
        "chunk_keys":[".".join(map(str,[i]+[0]*len(tail))) for i in indices],
        "decoded_rows_including_collateral":decoded,"decoded_padded_bytes":padded}


def compile_teacher_source_scope(root):
    root = Path(root).resolve(strict=True)
    material = compile_scope(root)
    selected = material["selected"]
    tasks = [row["task"] for row in selected]
    seal = SEALS["sensor"]
    index = read_pinned(root,seal["path"],seal["sha256"])
    json_paths = {P1A+"/artifacts/"+role+"/fit/"+t+".json" for t in tasks for role in ("constructions","codebooks")}
    prefixes = {P1A+"/artifacts/dataset/fit/"+t+".zarr/"+f+"/" for t in tasks for f in FIELDS}
    candidates = {}
    for line in index.decode().splitlines():
        h,p = line.split(None,1)
        # Lexical filtering before resolution or file access: excluded world
        # names in a shared seal are metadata, not permission to read payload.
        if p not in json_paths and not any(p.startswith(prefix) for prefix in prefixes):
            continue
        if p in candidates or re.fullmatch("[a-f0-9]{64}",h) is None:
            raise ValueError("duplicate/invalid selected seal entry")
        candidates[p] = h
    if not json_paths.issubset(candidates):
        raise ValueError("construction/codebook absent from pinned source seal")
    files = {p:candidates[p] for p in sorted(json_paths)}
    plans = {}; headers_read = {}
    for row in selected:
        for field in FIELDS:
            prefix = P1A+"/artifacts/dataset/fit/"+row["task"]+".zarr/"+field
            path = prefix+"/.zarray"
            if path not in candidates: raise ValueError("missing header")
            header = json.loads(read_pinned(root,path,candidates[path]))
            headers_read[path] = candidates[path]; files[path] = candidates[path]
            plan = plan_field(header,field,row["source"]["frame_rows"],row["source"]["source_frame_count"])
            plans[prefix] = plan
            for chunk in plan["chunk_keys"]:
                path = prefix+"/"+chunk
                if path not in candidates: raise ValueError("missing selected compressed chunk")
                files[path] = candidates[path]
    return {"schema":"gse_surface_teacher_source_plan_v1","status":"PREPARATION_ONLY_NOT_EXECUTION_AUTHORITY",
        "source_seal":seal,"selected":selected,"file_sha256":files,"array_access":plans,
        "existing_sensor_inputs_sha256":material["input_files_sha256"],
        "metadata_headers_read_sha256":headers_read,
        "counts":{"parents":10,"physical_edge_units":10,"observations":30,"selected_frames":150,
                  "construction_files":30,"codebook_files":30,"array_plans":90,"actual_payload_reads":0,"labels":0},
        "restrictions":["teacher_only_absolute_pose_and_source_identity","no_student_candidate_filter",
            "no_C07_C10_payload","no_root_reach_labels","no_automatic_training_authority"],
        "spacing":"Existing fixed rank0 fiveframe selection; source history clock and arc spacing remain unasserted."}
