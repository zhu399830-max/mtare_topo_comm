#!/usr/bin/env python3
"""Freeze exact cached-data derangement diagnostic before any scoring."""
import argparse
import json
import freeze_gse_point_axis_evidence_v1 as original
from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json,write_json,preflight
from mtare_topo.evaluation.gse_axis_observation_dependence import same_parent_derangement
from run_gse_composition_inventory_v1 import sha


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument("--freeze",action="store_true");args=parser.parse_args()
    original.SLUG="gse_axis_observation_dependence_v1"
    original.CARD=f"configs/v3/gate3/data_cards/{original.SLUG}.json";original.SPEC=f"configs/v3/gate3/{original.SLUG}.json"
    if (PROJECT_ROOT/original.CARD).exists() or (PROJECT_ROOT/original.SPEC).exists():raise RuntimeError("registration exists")
    card,spec=original.documents()
    scope="Read identical180 C01 cached predictions and sealed existing axis targets only. Fixed within-parent manifest-order shift9, correct versus shuffled output correspondence; geometry/layout/query-redundancy diagnosis. No training, checkpoint, raw scan, new teacher or scientificGate promotion."
    card["purpose"]=scope;card["approval"].update(scope=scope,confirmation_reference="Standing approved GSE-Graph implementation, follow fixed-budget failed-offset probe with the exact cached-output dependence diagnostic specified in PLAN/PROGRESS. No larger training or Gate advance.")
    card["allowed_cache_fields"]=["all_predictions","observation_metrics","sample_manifest","cached_teacher_control_points"]
    spec.update(question="Does lower raw-axis fitting error depend on correct observation correspondence and reflect layout, instead of only redundant32-query coverage?",method=scope,
        baseline="Identical32 cached query sets from the same parent shifted by9 of18 frozen rows. Includes initial,raw,offset,frozenold; not a clean raw-vs-mean learning ablation.",
        fallback="Stop any source/score/matching/resource drift. Report weak or absent correspondence dependence without increasing training or changing permutation. Even positive dependence is not detection/generalization.",
        user_authorization=card["approval"],shuffled_prediction_indices=same_parent_derangement(card["selected_rows"]),
        acceptance_criteria=["Exact180 same rows/10 parents/1452 fragments; fixed shift9 has no self-match and is a within-parent bijection.","Correct scores and identities reproduce all4 sealed methods; all32queries retained for correct and shuffled.","Report all rows/parents, geometry coordinate/direction/transverse/polyline errors with undefined direction counts, query-use histogram and nearest-query redundancy; no detector filtering or changed teacher.","CPU<=120s/4GiB;0model/checkpoint/raw/new_labels/optimizer. All evidence sealed; no new scientific pass claim."],
        expected_evidence=["Pre-frozen180 derangement manifest,8 full per-observation match/layout files,per-parent/aggregate metrics,comparisonSVG,versions,sourcehashes,log,state,seal."])
    spec["command"][spec["command"].index("tools/v3/run_gse_point_axis_evidence_v1.py")]="tools/v3/run_gse_axis_observation_dependence_v1.py"
    for path in ("tools/v3/run_gse_point_axis_evidence_v1r.py","tools/v3/run_gse_axis_observation_dependence_v1.py","tools/v3/freeze_gse_axis_observation_dependence_v1.py","tests/v3/unit/test_gse_axis_observation_dependence.py"):
        spec["source_sha256"][path]=sha(PROJECT_ROOT/path)
    if args.freeze:
        write_json(PROJECT_ROOT/original.CARD,card);write_json(PROJECT_ROOT/original.SPEC,spec)
        report=preflight(spec,load_json(PROJECT_ROOT/"results/project_status.json"),PROJECT_ROOT)
        print(json.dumps({"passed":report.passed,"errors":report.errors}));return int(not report.passed)
    print(json.dumps({"valid_card":True,"files_written":0,"observations":180}));return 0


if __name__=="__main__":raise SystemExit(main())
