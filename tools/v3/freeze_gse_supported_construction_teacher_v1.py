#!/usr/bin/env python3
"""Prepare the exact supported-construction teacher pilot; never execute it."""
from copy import deepcopy
import importlib.metadata
import json
import platform
from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import build_run_id, load_json, preflight, write_json
from mtare_topo.governance_supported_teacher import (
    SCHEMA,TEACHER_FIELDS,SENSOR_FIELDS,NATIVE_GEOMETRY,RESTRICTIONS,validate_supported_teacher_card,
)
from mtare_topo.governance_field_recovery import selection_sha256
from run_gse_composition_field_recovery_v1 import sha

SLUG="gse_supported_construction_teacher_v1"
CARD=f"configs/v3/gate3/data_cards/{SLUG}.json"
SPEC=f"configs/v3/gate3/{SLUG}.json"
PROBE_SPEC="configs/v3/gate3/gse_point_axis_probe_v1.json"
AUDIT_SPEC="configs/v3/gate3/gse_local_teacher_audit_v1.json"
METHOD="docs/GSE_SUPPORTED_CONSTRUCTION_TEACHER_V1.md"


def documents():
    probe=load_json(PROJECT_ROOT/PROBE_SPEC);audit=load_json(PROJECT_ROOT/AUDIT_SPEC)
    previous=load_json(PROJECT_ROOT/probe["data_card"])
    keys=("sampling_rule","license_or_allowed_use","partition","time_basis","duration_s","independent_sampling_unit",
          "observation_count","parent_count","unique_source_frame_count","frames_per_observation","worlds","selected_rows")
    card={key:deepcopy(previous[key]) for key in keys}
    tasks=sorted({row["task"] for row in card["selected_rows"]})
    selection=selection_sha256(card["selected_rows"])
    if selection!=previous["selection_sha256"]:raise ValueError("original180 selection hash drift")
    if "/constructions/" not in audit["construction_root"]:raise ValueError("unexpected original construction layout")
    roots={"sensor":probe["sensor_root"],"teacher":probe["teacher_root"],"construction":audit["construction_root"],
           "codebook":audit["construction_root"].replace("/constructions/","/codebooks/")}
    seals={"sensor":probe["source_seals"][0],"teacher":probe["source_seals"][1]}
    bound={}
    for path in (PROBE_SPEC,AUDIT_SPEC,probe["data_card"],audit["data_card"],audit["selection_manifest"],METHOD,*seals.values()):
        bound[path]=sha(PROJECT_ROOT/path)
    for path in seals.values():
        if bound[path]!=previous["sealed_sources"][path]:raise ValueError("original sensor/teacher seal changed")
    expected_json={roots[role]+"/"+task+".json" for role in ("construction","codebook") for task in tasks}
    json_hashes={}
    for line in (PROJECT_ROOT/seals["sensor"]).read_text().splitlines():
        digest,path=line.split(None,1)
        if path in expected_json:
            if path in json_hashes and json_hashes[path]!=digest:raise ValueError("conflicting selected JSON hash")
            json_hashes[path]=digest
    if set(json_hashes)!=expected_json:raise ValueError("exact20 source JSONs missing from original seal")
    purpose="Same180 C01 construction truth plus local five-frame native-axis support teacher pilot; all visible construction instances, no old target_node filtering; terminal without source cap evidence remains UNKNOWN."
    card.update({"schema_version":SCHEMA,"card_id":SLUG,"purpose":purpose,"tasks":tasks,"selection_sha256":selection,
        "teacher_source":"Existing P1b13 fields and selected P1a900 range/valid/source-code/teacher-only pose rows; exact20 realized construction/codebook JSONs. Recompute original native axis support, no sensor rerender or new worlds.",
        "legacy_node_count_inventory_only":100,"visible_fragment_count":1452,"teacher_fields":list(TEACHER_FIELDS),
        "sensor_fields":list(SENSOR_FIELDS),"native_geometry":dict(NATIVE_GEOMETRY),
        "terminal_cap_policy":"UNKNOWN_UNLESS_POSITIVE_SOURCE_CAP_EVIDENCE","terminal_cap_evidence_implemented":False,
        "region_count":None,"event_label_counts":None,"member_label_count":None,"capacity_ready":False,
        "support_proxy_limitations":"Axis-index min/max intervals can bridge unobserved sample gaps; construction members are not physically verified openings/connections. Unknown supports/conflicts persist. New class counts unknown; missing terminal evidence prevents complete three-class capacity PASS.",
        "restrictions":dict.fromkeys(RESTRICTIONS,True),"source_roots":roots,"source_seals":seals,
        "sealed_sources":bound,"task_json_sha256":json_hashes,
        "approval":{"status":"APPROVED","approved_by":"user","approved_at":"2026-09-05",
            "authorized_operations":["teacher_generation"],"authorized_gates":[3],"scope":purpose,
            "selection_sha256":selection,
            "confirmation_reference":"User-approved GSE-Graph implementation, latest parallel convergence plan, and standing in-scope authorization to select the reasonable method. The explicit construction-truth plus observation-support operational definition is recorded in docs/GSE_SUPPORTED_CONSTRUCTION_TEACHER_V1.md and current PLAN; no fabricated new user exchange, no reuse of audit/training permissions."}})
    report=validate_supported_teacher_card(card)
    if not report.passed:raise ValueError(report.errors)
    spec={"schema_version":"v3_run_spec_v1","gate":3,"date":"20260905","slug":SLUG,"seed":0,
        "operation":"teacher_generation","data_card":CARD,"config_path":CARD,"user_authorization":deepcopy(card["approval"]),
        "question":"On the same180 observations, which construction centers, members and local events have the explicitly defined native support proxy, while unsupported/conflicting/terminal-no-cap targets remain UNKNOWN?",
        "method":"Strict C01-only reader; reproduce original0.025m native support/13field parity from selected900 source rays and teacher-only poses; all local supported construction anchors become loss-only targets; supported members only, no old target_node selection; no model or rerender.",
        "baseline":"Sealed original P1b visible geometry/relations/support/odometry for exact parity, not an oracle student's performance result.",
        "fallback":"Input/seal/causality/support parity failure stops. Ambiguous/unsupported/conflicting labels remain UNKNOWN with reasons. Terminal cap evidence absent means no terminal event target; partial targets never pass full three-class capacity or scientificGate.",
        "native_geometry":dict(NATIVE_GEOMETRY),"wall_time_cap_s":600,
        "expected_counts":{"observations":180,"parents":10,"unique_frames":900,"visible_fragments":1452,
                           "legacy_nodes_inventory_only":100,"new_regions":None,"event_labels":None,"member_labels":None,
                           "model_windows":0,"optimizer_steps":0,"scan_rerenders":0,"test_world_reads":0},
        "estimated_cost":{"compute":"CPU only <=600s; original900 frames,0model/optimizer/rerender; strict single-task memory boundary",
                          "disk_gb":.25,"wall_time_hours":1/6,"host_ram_gb":4,"gpu_vram_gb":0},
        "acceptance_criteria":[
            "Exact180 C01 rows/10mixed parents/900 unique source frames/1452 old visible fragments; source/task/field allowlist before resolve; all selected reads hash-verified.",
            "Original13 teacher fields/frame-support/relations/relative odometry reproduced under native0.025 and source-mesh0.05/64 record; no radius change/new geometry or rerender.",
            "All locally supported construction instances, GTcenter/identity only in target/loss boundary; no old100-node target filtering. Unsupported or numerically ambiguous members/conflicts UNKNOWN, source min/max/count retained.",
            "Terminal requires real source-cap positive evidence; implementation absent in this pilot means UNKNOWN. Print true class/target counts and capacity_ready=false; no partial/full scientificPASS substitution.",
            "<=600s/4GiB host/0.25GB output/zeroGPU/model/optimizer;180-row targets/unknown reasons/source support summaries/previews/logs/state/hashseal; no overwrite or retry."],
        "expected_evidence":["Exact card/spec/source index/read hashes/environment/command; all180 construction-support target rows with unknown reasons and counts, parity details, source support, previews, RUN_STATE and SHA256 seal."],
        "expected_versions":{"python":platform.python_version(),**{name:importlib.metadata.version(name) for name in ("numpy","torch","zarr","scipy")}}}
    executable=probe["command"][6]
    spec["command"]=["env","OMP_NUM_THREADS=1","OPENBLAS_NUM_THREADS=1","MKL_NUM_THREADS=1","PYTHONHASHSEED=0",executable,
        "tools/v3/run_gse_supported_construction_teacher_v1.py","--spec",str(PROJECT_ROOT/SPEC),"--run-dir",str(PROJECT_ROOT/"results/gate3_semantics"/build_run_id(spec))]
    paths={str(path.relative_to(PROJECT_ROOT)) for path in (PROJECT_ROOT/"src").rglob("*.py")}
    paths.update({"tools/v3/run_gse_supported_construction_teacher_v1.py","tools/v3/freeze_gse_supported_construction_teacher_v1.py",
                  "tests/v3/unit/test_gse_supported_teacher_card.py","tests/v3/unit/test_gse_supported_teacher_reader.py",
                  "tests/v3/unit/test_gse_supported_teacher_freezer.py","tools/v3/_bootstrap.py"})
    # Include the main-agent kernel/runner tests when present, after they are
    # stable; docs+all source are bound and no actual run is made by freezer.
    paths.update(str(path.relative_to(PROJECT_ROOT)) for path in (PROJECT_ROOT/"tests/v3/unit").glob("test_gse_supported_construction*.py"))
    spec["source_sha256"]={path:sha(PROJECT_ROOT/path) for path in sorted(paths)}
    return card,spec


def main():
    if (PROJECT_ROOT/CARD).exists() or (PROJECT_ROOT/SPEC).exists():raise RuntimeError("refuse overwrite of frozen teacher card/spec")
    card,spec=documents();write_json(PROJECT_ROOT/CARD,card);write_json(PROJECT_ROOT/SPEC,spec)
    report=preflight(spec,load_json(PROJECT_ROOT/"results/project_status.json"),PROJECT_ROOT)
    print(json.dumps({"passed":report.passed,"errors":report.errors,"warnings":report.warnings,"card":CARD,"spec":SPEC}));return int(not report.passed)


if __name__=="__main__":raise SystemExit(main())
