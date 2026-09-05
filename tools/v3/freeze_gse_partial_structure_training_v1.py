#!/usr/bin/env python3
"""Prepare one fixed-budget CUDA cached-head run; metadata only, never run it."""
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import platform

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import build_run_id, load_json, preflight, write_json
from mtare_topo.governance_partial_structure_training import (
    SCHEMA,SOURCE_NAMES,EXPORT_SEAL_SHA256,IDENTITY_FIELDS,TRAINING,EVALUATION,RESOURCES,
    EFFECTIVE_COUNTS,RESTRICTIONS,validate_partial_structure_training_card,
)
from freeze_gse_partial_structure_export_v1 import selected_seal_entries

SLUG="gse_partial_structure_training_v1"
CARD=f"configs/v3/gate3/data_cards/{SLUG}.json"
SPEC=f"configs/v3/gate3/{SLUG}.json"
EXPORT_SPEC="configs/v3/gate3/gse_partial_structure_export_v1.json"
METHOD_DOC="docs/GSE_PARTIAL_STRUCTURE_TRAINING_V1.md"


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def documents():
    export=load_json(PROJECT_ROOT/EXPORT_SPEC)
    original=load_json(PROJECT_ROOT/export["data_card"])
    root=f"results/gate3_semantics/{build_run_id(export)}"
    seal=root+"/artifacts/evidence_sha256.txt"
    sources={name:{"path":root+("/metrics/" if name=="summary" else "/artifacts/")+basename}
             for name,basename in SOURCE_NAMES.items()}
    wanted={v["path"] for v in sources.values()}|{root+"/config/run_spec.json",root+"/config/data_card.json"}
    entries=selected_seal_entries(PROJECT_ROOT/seal,wanted,EXPORT_SEAL_SHA256)
    bound={seal:EXPORT_SEAL_SHA256}
    for value in sources.values():
        value["sha256"]=entries[value["path"]];bound[value["path"]]=value["sha256"]
    for path,snapshot in ((EXPORT_SPEC,"run_spec.json"),(export["data_card"],"data_card.json")):
        digest=sha(PROJECT_ROOT/path)
        if digest!=entries[root+"/config/"+snapshot]: raise ValueError("completed export spec/card snapshot drift")
        bound[path]=digest
    bound[METHOD_DOC]=sha(PROJECT_ROOT/METHOD_DOC)
    card={key:deepcopy(original[key]) for key in IDENTITY_FIELDS}
    purpose="One fixed seed0 three-branch cached partial structure-head diagnostic on original180 C01 observations; same initial parameters/order/300-step budget, no backbone or new scans, final step only."
    card.update({"schema_version":SCHEMA,"operation":"training","card_id":SLUG,"purpose":purpose,
        "teacher_source":"Exact completed partial-structure export: GT/predicted32 cached axes, loss-only masks/RegionTargets and preserved unknown direction transport. Four payloads only; original teacher construction/identity never forwarded.",
        "scope_limitations":"C01 same-sample fit diagnosis, partial centers/members and two known event classes only. No terminal target, calibrated detector, held-out generalization or scientific Gate PASS. The full original three-class and downstream qualification criteria are unchanged.",
        "export_identity_reference":deepcopy(original),
        "export_reference":{"spec":EXPORT_SPEC,"card":export["data_card"],"seal":seal},
        "sources":sources,"sealed_sources":bound,"training":deepcopy(TRAINING),"evaluation":deepcopy(EVALUATION),
        "resources":dict(RESOURCES),"effective_counts":deepcopy(EFFECTIVE_COUNTS),"capacity_ready":False,
        "restrictions":dict.fromkeys(RESTRICTIONS,True),
        "approval":{"status":"APPROVED","approved_by":"user","approved_at":"2026-09-05",
            "authorized_operations":["training"],"authorized_gates":[3],"scope":purpose,
            "selection_sha256":original["selection_sha256"],"sources":deepcopy(sources),"training":deepcopy(TRAINING),
            "confirmation_reference":"Standing user authorization for the approved GSE-Graph implementation and parallel convergence plan covers this scoped next head-only experiment. No new user exchange is fabricated. This is new training-only authority bound to the exact exported population and900 total head steps; old export authority is not reused to authorize training."}})
    report=validate_partial_structure_training_card(card)
    if not report.passed: raise ValueError(report.errors)
    spec={"schema_version":"v3_run_spec_v1","gate":3,"date":"20260905","slug":SLUG,"seed":0,
        "operation":"training","data_card":CARD,"config_path":CARD,"user_authorization":deepcopy(card["approval"]),
        "question":"On identical cached180 observations, can partial structure heads fit supported centers/members, and what changes under GT versus predicted axes and removal of explicit relations?",
        "method":"Three seed0 region-query heads with shared initial weights and sample schedule; original GT axes/predicted axes/predicted axes without explicit relations; hidden64/26826parameters,Adam.001,B18,300steps per branch,CUDA0; no backbone or extra scans.",
        "baseline":"Same-budget GT-axis head is component reference; same predicted axes with explicit relations disabled is direct software ablation. None is a new detector or generalization result.",
        "fallback":"Population/hash/identity/effective-label/finite-gradient/budget drift fails and seals once. Fixed final300 only, no best checkpoint or additional steps. Unknown labels and original complete research criteria remain unchanged.",
        "training":deepcopy(TRAINING),"evaluation":deepcopy(EVALUATION),"wall_time_cap_s":1800,
        "expected_counts":{"observations":180,"parents":10,"unique_frames":900,"visible_fragments":1452,
            "raw_teacher":deepcopy(original["raw_teacher_counts"]),"effective":deepcopy(EFFECTIVE_COUNTS),
            "optimizer_steps":900,"head_inference_windows":1080,"backbone_windows":0,"new_sensor_frames":0},
        "estimated_cost":{"compute":"One CUDA0 run <=1800s;3x300 head updates,1080initial/final evaluation windows;0backbone/newscan", "wall_time_hours":.5,
                          "host_ram_gb":4,"gpu_vram_gb":4,"disk_gb":.5},
        "acceptance_criteria":["Exact exported180 C01/10parents/900existing frames/1452fragments; known raw/effective denominators hash-bound, all32 cached axes retained, identity only metadata/loss.",
            "Three branches share initial parameter hash and seed0 schedule, each300Adam.001/B18 updates,26826parameters; use_relations true/true/false only; frozen axes,0backbone/newscan/checkpoint selection.",
            "Initial and final all180 per branch; evaluation uses pure center-geometric matching rather than training.matches; member threshold0.5 explicitly uncalibrated; report known corridor/junction classes and missing terminal.",
            "Preserve shared initial state, branch flags/schedule, all final weights/initial-final predictions/scoring traces/parent metrics/complete previews, raw logs, command/environment, RUN_STATE and SHA256seal.",
            "<=1800s/4GiBhost/4GiBGPU/.5GBresults; no overwrite/retry/best-selection. Budgeted FIT diagnosis only, no three-class/detection/generalization or scientific Gate PASS."],
        "expected_evidence":["Full fixed budget/initial-state/schedule evidence,900-step logs,3finalweights,1080initial-final raw predictions and independent scoring traces, actual labels/unknowns and per-parent/full previews, environment/source hashes/state/seal."],
        "expected_versions":{"python":platform.python_version(),**{name:importlib.metadata.version(name) for name in ("numpy","torch","zarr","scipy")}}}
    executable=export["command"][5]
    spec["command"]=["env","CUBLAS_WORKSPACE_CONFIG=:4096:8","OMP_NUM_THREADS=1","OPENBLAS_NUM_THREADS=1",
        "MKL_NUM_THREADS=1","PYTHONHASHSEED=0",executable,"tools/v3/run_gse_partial_structure_training_v1.py",
        "--spec",str(PROJECT_ROOT/SPEC),"--run-dir",str(PROJECT_ROOT/"results/gate3_semantics"/build_run_id(spec))]
    spec["command_sha256"]=hashlib.sha256(json.dumps(spec["command"],separators=(",",":")).encode()).hexdigest()
    paths={str(p.relative_to(PROJECT_ROOT)) for p in (PROJECT_ROOT/"src").rglob("*.py")}
    paths.update({"tools/v3/run_gse_partial_structure_training_v1.py","tools/v3/freeze_gse_partial_structure_training_v1.py",
        "tools/v3/freeze_gse_partial_structure_export_v1.py","tools/v3/_bootstrap.py",
        "tools/v3/run_gse_supported_construction_teacher_v1.py"})
    paths.update(str(p.relative_to(PROJECT_ROOT)) for p in (PROJECT_ROOT/"tests/v3/unit").glob("test_gse_partial_structure*.py"))
    paths.add("tests/v3/unit/test_gse_partial_training_inputs.py")
    spec["source_sha256"]={p:sha(PROJECT_ROOT/p) for p in sorted(paths)}
    return card,spec


def main():
    if (PROJECT_ROOT/CARD).exists() or (PROJECT_ROOT/SPEC).exists(): raise RuntimeError("refuse overwrite of frozen training card/spec")
    card,spec=documents();write_json(PROJECT_ROOT/CARD,card);write_json(PROJECT_ROOT/SPEC,spec)
    report=preflight(spec,load_json(PROJECT_ROOT/"results/project_status.json"),PROJECT_ROOT)
    print(json.dumps({"passed":report.passed,"errors":report.errors,"warnings":report.warnings,"card":CARD,"spec":SPEC}))
    return int(not report.passed)


if __name__=="__main__": raise SystemExit(main())
