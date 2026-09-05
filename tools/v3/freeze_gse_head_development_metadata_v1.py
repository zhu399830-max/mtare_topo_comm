#!/usr/bin/env python3
"""Register metadata-only C02 selection, before reading new shard contents."""
import argparse
import json
from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json,write_json,preflight
from mtare_topo.governance_head_development import SCHEMA,FIELDS,SAMPLING_RULE,HEAD_EXPOSURE,RESTRICTIONS,validate_head_development_metadata_card
from run_gse_composition_inventory_v1 import sha

SLUG="gse_head_development_metadata_v1"
CARD=f"configs/v3/gate3/data_cards/{SLUG}.json"
SPEC=f"configs/v3/gate3/{SLUG}.json"
PYTHON="/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument("--freeze",action="store_true");args=parser.parse_args()
    if (PROJECT_ROOT/CARD).exists() or (PROJECT_ROOT/SPEC).exists():raise RuntimeError("registration exists")
    old_path="configs/v3/gate3/data_cards/gse_point_axis_probe_v1.json";old=load_json(PROJECT_ROOT/old_path)
    tasks=sorted({r["task"].replace("_C01__","_C02__") for r in old["selected_rows"]})
    previous=load_json(PROJECT_ROOT/"configs/v3/gate3/gse_point_axis_probe_v1.json")
    seal=previous["source_seals"][1];digest="f629b511e9a945bbe15249e3bc212be57a01b1aa0118d0d7d228824aa25f7d47"
    if sha(PROJECT_ROOT/seal)!=digest:raise ValueError("teacher seal drift")
    scope="Exact10 C02 c1_mixed parent metadata: read only root Zarr group/attributes and frame_row,source_global_sequence_index,primitive_mask integer headers/selected chunks. Select18 midpoint-quantile rows per parent usingN only. No geometry/sensor/model/checkpoint/optimizer; count real frames/fragments before separate frozen-head inference card."
    approval={"status":"APPROVED","approved_by":"user_standing_scope","approved_at":"2026-09-05",
        "authorized_operations":["audit"],"authorized_gates":[3],"scope":scope,
        "confirmation_reference":"Approved GSE-Graph implementation and continuous in-scope execution; PLAN/PROGRESS authorize exact C02 metadata selection/card preparation after cached observation dependence. No new model inference or training authority."}
    card=dict(schema_version=SCHEMA,card_id=SLUG,purpose=scope,teacher_source="Existing P1b immutable reference identities,frame rows and visible masks only; no geometry/new teacher. Integer chunks may contain neighboring metadata rows; logical selection exact180.",
        license_or_allowed_use=old["license_or_allowed_use"],head_exposure=HEAD_EXPOSURE,partition="fit",
        independent_sampling_unit="topology_parent",duration_s=None,time_basis="distance_sampled_no_acquisition_clock",
        target_observations=180,parent_count=10,rows_per_parent=18,source_global_sequence_indices=None,unique_source_frame_count=None,visible_fragment_count=None,
        fields=list(FIELDS),sampling_rule=SAMPLING_RULE,tasks=tasks,worlds=[t.split("__")[0] for t in tasks],
        source_roots={"teacher":previous["teacher_root"]},source_seals={"teacher":seal},
        sealed_sources={seal:digest,old_path:sha(PROJECT_ROOT/old_path)},restrictions=dict.fromkeys(RESTRICTIONS,True),approval=approval)
    validation=validate_head_development_metadata_card(card)
    if not validation.passed:raise ValueError(validation.errors)
    files=sorted(str(p.relative_to(PROJECT_ROOT)) for p in (PROJECT_ROOT/"src/mtare_topo").rglob("*.py"))
    files += ["tools/v3/"+name for name in ("_bootstrap.py","preflight.py","create_run.py","run_gse_composition_inventory_v1.py","run_gse_head_development_metadata_v1.py","freeze_gse_head_development_metadata_v1.py")]
    files += ["tests/v3/unit/test_gse_head_development_card.py","tests/v3/unit/test_gse_head_development_selection.py"]
    run=f"results/gate3_semantics/gate3_20260905_{SLUG}_seed0"
    spec=dict(schema_version="v3_run_spec_v1",gate=3,date="20260905",slug=SLUG,seed=0,operation="audit",data_card=CARD,config_path=CARD,
        question="What exactC02 head-development rows/frame references/visible fragment counts follow a score-independent fixed selector, without prematurely claiming unseen-model generalization?",
        method=scope,baseline="Original C01-only head-fitting card verifies disjoint head parents; frozen source identities and masks are reference metadata, not performance baseline.",
        fallback="Stop on seal/split/shape/causality/population/resource mismatch; no substitute rows or score-based sampling. No inference/geometry/training until separate exact approval/card.",
        user_authorization=approval,command=["env","OPENBLAS_NUM_THREADS=1","OMP_NUM_THREADS=1","PYTHONHASHSEED=0",PYTHON,"tools/v3/run_gse_head_development_metadata_v1.py","--spec",str(PROJECT_ROOT/SPEC),"--run-dir",str(PROJECT_ROOT/run)],
        expected_versions={"python":"3.13.5","numpy":"2.1.3","zarr":"2.18.7"},source_sha256={p:sha(PROJECT_ROOT/p) for p in files},
        estimated_cost={"disk_gb":.01,"wall_time_hours":1/30,"host_ram_gb":1,"compute":"CPU<=120s/1GiB;zeroGPU,raw/geometry/checkpoint/model/optimizer/newlabels;metadata selected chunks only."},
        acceptance_criteria=["Exact10 C02 parents and18 quantile rows each; sampling independent of IDs/mask/scores, no duplicates.","Identity,source refs, five consecutive frame rows and existing visible counts valid; unknown counts resolved honestly,allreadbytessealed.","Report180 source IDs,unique frames/traversals,nominal index spacing and unmeasured physical distances; oldbackbone C01-C06 exposure explicit.","No data/model/teacher-generation Gate pass;metadata scope only; CPU120s/1GiB,logs/manifest/state/seal preserved."],
        expected_evidence=["180-row manifest withsourceIDs/frame refs/fragment counts,traversal metadata,perparent population,source-read hashes,card/spec/versions,log,state,seal. No performance plot invented."])
    if args.freeze:
        write_json(PROJECT_ROOT/CARD,card);write_json(PROJECT_ROOT/SPEC,spec)
        report=preflight(spec,load_json(PROJECT_ROOT/"results/project_status.json"),PROJECT_ROOT)
        print(json.dumps({"passed":report.passed,"errors":report.errors}));return int(not report.passed)
    print(json.dumps({"valid_card":True,"files_written":0,"tasks":tasks}));return 0


if __name__=="__main__":raise SystemExit(main())
