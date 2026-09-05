from copy import deepcopy
import pytest

from mtare_topo.governance_geometry_bound import (
    SCHEMA,LOSS_CONTRACT,SHARED_FIELDS,RESTRICTIONS,ATTRIBUTION_SEAL_SHA256,validate_geometry_bound_training_card,
)
from mtare_topo.governance_assignment_attribution import TRAINING_SEAL_SHA256
from tests.v3.unit.test_gse_partial_structure_training_card import card as original_card


def card():
    base=original_card();value={k:deepcopy(base[k]) for k in SHARED_FIELDS}
    bound={s["path"]:s["sha256"] for s in base["sources"].values()}
    bound[base["export_reference"]["seal"]]=base["sealed_sources"][base["export_reference"]["seal"]]
    references={name:{"spec":f"configs/{name}_spec.json","card":f"configs/{name}_card.json","seal":f"fixtures/{name}_seal.txt"}
                for name in ("base_reference","attribution_reference")}
    for name,reference in references.items():
        bound.update(dict.fromkeys(reference.values(),"f"*64))
        bound[reference["seal"]]=TRAINING_SEAL_SHA256 if name=="base_reference" else ATTRIBUTION_SEAL_SHA256
    value.update({"schema_version":SCHEMA,"operation":"training","card_id":"synthetic_geometry_bound",
        "purpose":"Single-variable geometry loss binding","teacher_source":"Same four sealed files",
        "scope_limitations":"No full scientific qualification or extra training budget","base_training_card":base,
        **references,**deepcopy(LOSS_CONTRACT),"sealed_sources":bound,"restrictions":dict.fromkeys(RESTRICTIONS,True),
        "capacity_ready":False,"scientific_gate_pass":False,
        "approval":{**base["approval"],"loss_contract":deepcopy(LOSS_CONTRACT)}})
    return value


def test_new_loss_authority_does_not_reuse_old_training_permission():
    value=card();before=deepcopy(value)
    assert validate_geometry_bound_training_card(value).passed and value==before
    assert not validate_geometry_bound_training_card(original_card()).passed
    value["approval"]=deepcopy(value["base_training_card"]["approval"])
    assert not validate_geometry_bound_training_card(value).passed


@pytest.mark.parametrize("change",["budget","batch","init","evaluation","data","loss_weights","loss_rule","ambiguity",
    "center_required","empty_batch","oldcheckpoint","oldprediction","oldsummary","attributionseal","baseseal","science"])
def test_single_variable_boundary(change):
    c=card()
    if change=="budget":c["training"]["steps_per_branch"]=301
    elif change=="batch":c["training"]["batch_size"]=1
    elif change=="init":c["training"]["seed"]=1
    elif change=="evaluation":c["evaluation"]["member_threshold"]=.4
    elif change=="data":c["sources"]["inputs"]["sha256"]="f"*64
    elif change=="loss_weights":c["loss_weights"]={"event":2}
    elif change=="loss_rule":c["loss_policy"]="joint"
    elif change=="ambiguity":c["ambiguous_assignment"]="pick_first"
    elif change=="center_required":c["partial_known_targets_require_centers"]=1
    elif change=="empty_batch":c["supervised_empty_batch"]="skip"
    elif change=="oldcheckpoint":c["sealed_sources"]["old/head.pt"]="f"*64
    elif change=="oldprediction":c["sealed_sources"]["old/predictions.npz"]="f"*64
    elif change=="oldsummary":c["sealed_sources"]["old/summary.json"]="f"*64
    elif change=="attributionseal":c["sealed_sources"][c["attribution_reference"]["seal"]]="f"*64
    elif change=="baseseal":c["sealed_sources"][c["base_reference"]["seal"]]="f"*64
    else:c["scientific_gate_pass"]=True
    assert not validate_geometry_bound_training_card(c).passed


def test_invalid_base_and_missing_metadata_fail_without_io():
    c=card();c["base_training_card"]={}
    assert not validate_geometry_bound_training_card(c).passed
    c=card();c["attribution_reference"]["card"]=None
    assert not validate_geometry_bound_training_card(c).passed
