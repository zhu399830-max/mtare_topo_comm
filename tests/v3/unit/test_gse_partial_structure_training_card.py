from copy import deepcopy
import pytest

from mtare_topo.governance_partial_structure_training import (
    SCHEMA,SOURCE_NAMES,EXPORT_SEAL_SHA256,IDENTITY_FIELDS,TRAINING,EVALUATION,RESOURCES,
    EFFECTIVE_COUNTS,RESTRICTIONS,validate_partial_structure_training_card,
)
from tests.v3.unit.test_gse_partial_structure_card import card as exported


def card():
    old=exported()
    value={k:deepcopy(old[k]) for k in IDENTITY_FIELDS}
    sources={k:{"path":"fixtures/"+name,"sha256":"e"*64} for k,name in SOURCE_NAMES.items()}
    reference={"spec":"configs/export_spec.json","card":"configs/export_card.json","seal":"fixtures/export_seal.txt"}
    bound={s["path"]:s["sha256"] for s in sources.values()}
    bound.update(dict.fromkeys(reference.values(),"f"*64));bound[reference["seal"]]=EXPORT_SEAL_SHA256
    value.update({"schema_version":SCHEMA,"operation":"training","card_id":"synthetic_partial_training",
        "purpose":"Same180 fixed-budget head-only diagnostic","teacher_source":"Sealed four exported inputs only",
        "scope_limitations":"Partial fit only, no generalization/complete three-class claim",
        "export_identity_reference":old,"export_reference":reference,"sources":sources,"sealed_sources":bound,
        "training":deepcopy(TRAINING),"evaluation":deepcopy(EVALUATION),"resources":dict(RESOURCES),
        "effective_counts":deepcopy(EFFECTIVE_COUNTS),"capacity_ready":False,
        "restrictions":dict.fromkeys(RESTRICTIONS,True),
        "approval":{**old["approval"],"authorized_operations":["training"],"sources":deepcopy(sources),"training":deepcopy(TRAINING)}})
    return value


def test_independent_training_authority_not_export_alias():
    c=card();before=deepcopy(c)
    assert validate_partial_structure_training_card(c).passed and c==before
    assert not validate_partial_structure_training_card(exported()).passed
    c["approval"]=deepcopy(c["export_identity_reference"]["approval"])
    assert not validate_partial_structure_training_card(c).passed


@pytest.mark.parametrize("field",IDENTITY_FIELDS)
def test_inherited_identity_or_version_cannot_drift(field):
    c=card();c[field]=0 if field=="duration_s" else None
    assert not validate_partial_structure_training_card(c).passed


@pytest.mark.parametrize("field,value",[("steps_per_branch",301),("steps_per_branch",300.),("total_steps",1080),
    ("batch_size",1),("seed",True),("seed",1),("hidden",128),("lr",.01),("optimizer","SGD"),
    ("weight_decay",.001),("device","cpu"),("device","cuda:1"),("parameters_per_head",100),
    ("checkpoint_selection","best_loss"),("branches",["predicted_axes"]),
    ("sample_schedule","different_per_branch"),("same_initial_state",1),
    ("use_relations",{"gt_axes":True,"predicted_axes":True,"predicted_no_relations":True})])
def test_no_budget_or_method_drift_even_if_reapproved(field,value):
    c=card();c["training"][field]=value;c["approval"]["training"]=deepcopy(c["training"])
    assert not validate_partial_structure_training_card(c).passed


@pytest.mark.parametrize("field,value",[("stages",["final"]),("observations_per_stage_branch",18),
    ("total_head_inference_windows",180),("member_threshold",.6),("threshold_calibration","validation"),
    ("region_matching","training.matches"),("read_training_matches",True),
    ("known_event_classes",["corridor","junction","terminal"]),("scientific_gate_pass",True)])
def test_no_evaluation_shortcut(field,value):
    c=card();c["evaluation"][field]=value;assert not validate_partial_structure_training_card(c).passed


@pytest.mark.parametrize("field",RESTRICTIONS)
def test_no_scope_expansion(field):
    c=card();c["restrictions"][field]=False;assert not validate_partial_structure_training_card(c).passed


@pytest.mark.parametrize("branch",["gt","predicted"])
@pytest.mark.parametrize("field",list(EFFECTIVE_COUNTS["gt"]))
def test_effective_denominators_fixed(branch,field):
    c=card();c["effective_counts"][branch][field]+=1;assert not validate_partial_structure_training_card(c).passed


@pytest.mark.parametrize("role",SOURCE_NAMES)
@pytest.mark.parametrize("change",["path","hash","missing"])
def test_four_explicit_sources_bound(role,change):
    c=card()
    if change=="path":c["sources"][role]["path"]="../outside"
    elif change=="hash":c["sources"][role]["sha256"]="a"*64
    else:c["sources"].pop(role)
    assert not validate_partial_structure_training_card(c).passed


@pytest.mark.parametrize("field",["checkpoint","checkpoints","backbone","sensor_root","teacher_root","model_selection"])
def test_no_input_model_or_raw_authority(field):
    c=card();c[field]={};assert not validate_partial_structure_training_card(c).passed


@pytest.mark.parametrize("field,value",[("operation","data_export"),("capacity_ready",True),("export_identity_reference",None),
    ("export_reference",{}),("sources",None),("sealed_sources",{}),("training",None),("evaluation",None),("effective_counts",None)])
def test_malformed_card(field,value):
    c=card();c[field]=value;assert not validate_partial_structure_training_card(c).passed


def test_original_export_evidence_does_not_become_training_authority():
    c=card();c["export_identity_reference"]["approval"]["authorized_operations"]=["training"]
    assert not validate_partial_structure_training_card(c).passed


def test_export_seal_and_extra_old_numeric_cache_cannot_change():
    c=card();c["sealed_sources"][c["export_reference"]["seal"]]="f"*64
    assert not validate_partial_structure_training_card(c).passed
    c=card();c["sealed_sources"]["old/all_predictions.npz"]="f"*64
    assert not validate_partial_structure_training_card(c).passed


def test_resource_count_type_strict():
    c=card();c["resources"]["wall_time_cap_s"]=1800.
    assert not validate_partial_structure_training_card(c).passed


def test_all_branches_keep_explicit_relations_flags_and_fixed_final_eval():
    c=card()
    assert c["training"]["use_relations"]=={"gt_axes":True,"predicted_axes":True,"predicted_no_relations":False}
    assert c["evaluation"]["total_head_inference_windows"]==3*2*180
