from copy import deepcopy
import pytest

from mtare_topo.governance_geometry_bound_rank import (
    SCHEMA,SOURCE_NAMES,CORRECTIVE_SEAL_SHA256,RANK_POLICY,RESOURCES,RESTRICTIONS,validate_geometry_bound_rank_card,
)
from mtare_topo.governance_partial_structure_training import IDENTITY_FIELDS
from tests.v3.unit.test_gse_geometry_bound_card import card as corrective_card


def card():
    old=corrective_card();sources=deepcopy(old["sources"])
    for key in ("gt_final","predicted_final","no_relations_final","corrective_summary"):
        sources[key]={"path":"corrective/"+SOURCE_NAMES[key],"sha256":"d"*64}
    reference={"spec":"configs/corrective_spec.json","card":"configs/corrective_card.json","seal":"corrective/seal.txt"}
    bound={s["path"]:s["sha256"] for s in sources.values()}
    bound.update(dict.fromkeys(reference.values(),"f"*64));bound[reference["seal"]]=CORRECTIVE_SEAL_SHA256
    exportseal=old["export_reference"]["seal"];bound[exportseal]=old["sealed_sources"][exportseal]
    c={k:deepcopy(old[k]) for k in (*IDENTITY_FIELDS,"effective_counts")}
    c.update({"schema_version":SCHEMA,"operation":"data_export","card_id":"synthetic_rank",
        "purpose":"Rank-only diagnostic","teacher_source":"Eight cached outputs","scope_limitations":"No threshold search or training",
        "corrective_card":old,"corrective_reference":reference,"sources":sources,"sealed_sources":bound,
        "rank_policy":deepcopy(RANK_POLICY),"resources":dict(RESOURCES),"restrictions":dict.fromkeys(RESTRICTIONS,True),
        "scientific_gate_pass":False,"approval":{**old["approval"],"authorized_operations":["data_export"],
            "sources":deepcopy(sources),"rank_policy":deepcopy(RANK_POLICY)}})
    return c


def test_valid_rank_card_has_no_training_authority():
    c=card();before=deepcopy(c)
    assert validate_geometry_bound_rank_card(c).passed and c==before
    assert not validate_geometry_bound_rank_card(corrective_card()).passed


@pytest.mark.parametrize("change",["threshold","rank_population","single_class","history","training","missing_cache",
    "extra_cache","wrong_hash","changed_old_input","seal","float_count","split","no_bce","science","malformed_nested"])
def test_only_exact_final_ranking_diagnostic_allowed(change):
    c=card()
    if change=="threshold":c["rank_policy"]["threshold_search"]=True
    elif change=="rank_population":c["rank_policy"]["member_population"]="best_score_candidate"
    elif change=="single_class":c["rank_policy"]["undefined_ap"]="zero"
    elif change=="history":c["sealed_sources"]["old/training_history.json"]="f"*64
    elif change=="training":c["approval"]["authorized_operations"]=["training"]
    elif change=="missing_cache":c["sources"].pop("gt_final")
    elif change=="extra_cache":c["sealed_sources"]["extra/initial.npz"]="f"*64
    elif change=="wrong_hash":c["sources"]["gt_final"]["sha256"]="f"*64
    elif change=="changed_old_input":
        c["sources"]["inputs"]["sha256"]="f"*64;c["sealed_sources"][c["sources"]["inputs"]["path"]]="f"*64
        c["approval"]["sources"]=deepcopy(c["sources"])
    elif change=="seal":c["sealed_sources"][c["corrective_reference"]["seal"]]="f"*64
    elif change=="float_count":c["observation_count"]=180.
    elif change=="split":c["selected_rows"][0]["task"]="S01_wrong_C07__c1_mixed"
    elif change=="no_bce":c["rank_policy"].pop("probability_diagnostic")
    elif change=="science":c["scientific_gate_pass"]=True
    else:c["corrective_card"]["sources"]=None
    assert not validate_geometry_bound_rank_card(c).passed
