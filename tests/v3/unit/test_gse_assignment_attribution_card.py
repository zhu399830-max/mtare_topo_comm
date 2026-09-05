from copy import deepcopy
import pytest

from mtare_topo.governance_assignment_attribution import (
    SCHEMA,SOURCE_NAMES,TRAINING_SEAL_SHA256,ATTRIBUTION,RESOURCES,RESTRICTIONS,
    validate_assignment_attribution_card,
)
from mtare_topo.governance_partial_structure_training import IDENTITY_FIELDS
from tests.v3.unit.test_gse_partial_structure_training_card import card as training_card


def card():
    old=training_card();sources=deepcopy(old["sources"])
    for name in ("gt_final","predicted_final","no_relations_final","history","training_summary"):
        sources[name]={"path":"training/"+SOURCE_NAMES[name],"sha256":"d"*64}
    reference={"spec":"configs/training_spec.json","card":"configs/training_card.json","seal":"training/seal.txt"}
    bound={v["path"]:v["sha256"] for v in sources.values()}
    bound.update(dict.fromkeys(reference.values(),"a"*64));bound[reference["seal"]]=TRAINING_SEAL_SHA256
    exportseal=old["export_reference"]["seal"];bound[exportseal]=old["sealed_sources"][exportseal]
    value={k:deepcopy(old[k]) for k in (*IDENTITY_FIELDS,"effective_counts")}
    value.update({"schema_version":SCHEMA,"operation":"data_export","card_id":"synthetic_assignment_attribution",
        "purpose":"Cached assignment attribution only","teacher_source":"Exact nine sealed cached files",
        "scope_limitations":"Label-conditioned diagnostic only, no science","training_card":old,
        "training_reference":reference,"sources":sources,"sealed_sources":bound,"attribution":deepcopy(ATTRIBUTION),
        "resources":dict(RESOURCES),"restrictions":dict.fromkeys(RESTRICTIONS,True),"scientific_gate_pass":False,
        "approval":{**old["approval"],"authorized_operations":["data_export"],"sources":deepcopy(sources)}})
    return value


def test_new_export_authority_preserves_training_identity_without_authorizing_training():
    c=card();before=deepcopy(c)
    assert validate_assignment_attribution_card(c).passed and c==before
    assert not validate_assignment_attribution_card(training_card()).passed
    c["approval"]=c["training_card"]["approval"]
    assert not validate_assignment_attribution_card(c).passed


@pytest.mark.parametrize("change",["extra_payload","old_source_swap","missing_final","seal","split","float_count",
    "different_threshold","short_epoch","training_authority","model_authority","scope_relaxed","science_claim","invalid_embedded"])
def test_high_value_scope_or_identity_failures(change):
    c=card()
    if change=="extra_payload":c["sealed_sources"]["extra/raw.npz"]="f"*64
    elif change=="old_source_swap":
        c["sources"]["inputs"]["sha256"]="f"*64
        c["sealed_sources"][c["sources"]["inputs"]["path"]]="f"*64
        c["approval"]["sources"]=deepcopy(c["sources"])
    elif change=="missing_final":c["sources"].pop("gt_final")
    elif change=="seal":c["sealed_sources"][c["training_reference"]["seal"]]="f"*64
    elif change=="split":c["selected_rows"][0]["task"]="S01_synthetic_C07__c1_mixed"
    elif change=="float_count":c["attribution"]["cached_prediction_observations"]=540.
    elif change=="different_threshold":c["attribution"]["member_threshold"]=.4
    elif change=="short_epoch":c["attribution"]["last_epoch_batches"]=9
    elif change=="training_authority":c["approval"]["authorized_operations"]=["training"]
    elif change=="model_authority":c["inference"]={}
    elif change=="scope_relaxed":c["restrictions"]["no_threshold_search"]=False
    elif change=="science_claim":c["scientific_gate_pass"]=True
    else:c["training_card"]["sources"]=None
    assert not validate_assignment_attribution_card(c).passed
