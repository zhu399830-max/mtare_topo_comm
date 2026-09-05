import copy
import pytest
from mtare_topo.governance_supported_teacher import (
    SCHEMA,TEACHER_FIELDS,SENSOR_FIELDS,NATIVE_GEOMETRY,RESTRICTIONS,validate_supported_teacher_card,
)
from tests.v3.unit.test_gse_point_axis_training_card import card as previous


def card():
    old=previous()
    keys=("purpose","teacher_source","sampling_rule","license_or_allowed_use","partition","time_basis","duration_s",
          "independent_sampling_unit","observation_count","parent_count","unique_source_frame_count","frames_per_observation",
          "worlds","selected_rows","selection_sha256")
    value={key:copy.deepcopy(old[key]) for key in keys}
    tasks=sorted({row["task"] for row in value["selected_rows"]})
    roots={role:"fixtures/"+role+"/fit" for role in ("sensor","teacher","construction","codebook")}
    value.update({"schema_version":SCHEMA,"card_id":"synthetic_supported_teacher","tasks":tasks,
        "legacy_node_count_inventory_only":100,"visible_fragment_count":1452,"teacher_fields":list(TEACHER_FIELDS),
        "sensor_fields":list(SENSOR_FIELDS),"native_geometry":dict(NATIVE_GEOMETRY),
        "terminal_cap_policy":"UNKNOWN_UNLESS_POSITIVE_SOURCE_CAP_EVIDENCE","terminal_cap_evidence_implemented":False,
        "region_count":None,"event_label_counts":None,"member_label_count":None,"capacity_ready":False,
        "support_proxy_limitations":"Intervals may bridge unsampled gaps; no physical connectivity or complete visibility claim",
        "restrictions":dict.fromkeys(RESTRICTIONS,True),"source_roots":roots,
        "source_seals":{"sensor":"fixtures/sensor_seal.txt","teacher":"fixtures/teacher_seal.txt"},
        "sealed_sources":{"fixtures/sensor_seal.txt":"a"*64,"fixtures/teacher_seal.txt":"b"*64},
        "task_json_sha256":{roots[role]+"/"+task+".json":"c"*64 for role in ("construction","codebook") for task in tasks},
        "approval":{**old["approval"],"authorized_operations":["teacher_generation"],"scope":"Synthetic new teacher-only pilot approval"}})
    value["approval"].pop("checkpoint_sha256",None)
    return value


def test_exact_card_is_valid_and_previous_training_not_authority():
    value=card();before=copy.deepcopy(value)
    assert validate_supported_teacher_card(value).passed and value==before
    assert not validate_supported_teacher_card(previous()).passed


@pytest.mark.parametrize("field,value",[("observation_count",180.),("parent_count",True),("unique_source_frame_count",899),
    ("visible_fragment_count",1489),("legacy_node_count_inventory_only",99),("capacity_ready",True),("region_count",100),
    ("event_label_counts",{}),("member_label_count",0),("duration_s",90),("time_basis","made_up_seconds"),
    ("source_roots",[]),("source_seals",{}),("task_json_sha256",{}),("selected_rows",None),("worlds",[]),
    ("teacher_fields",list(TEACHER_FIELDS)+["node_id"]),("sensor_fields",list(SENSOR_FIELDS)+["axis_xyz_m"]),
    ("terminal_cap_evidence_implemented",True),("terminal_cap_policy","degree1_is_terminal")])
def test_contract_drift(field,value):
    c=card();c[field]=value;assert not validate_supported_teacher_card(c).passed


@pytest.mark.parametrize("restriction",RESTRICTIONS)
def test_restrictions_required(restriction):
    c=card();c["restrictions"][restriction]=False;assert not validate_supported_teacher_card(c).passed


@pytest.mark.parametrize("key,value",[("axis_spacing_m",.05),("mesh_axial_spacing_m",.025),("mesh_angular_segments",64.)])
def test_native_geometry_fixed(key,value):
    c=card();c["native_geometry"][key]=value;assert not validate_supported_teacher_card(c).passed


@pytest.mark.parametrize("key",["checkpoint","checkpoints","training","inference","prediction_root","input_fields"])
def test_no_model_or_training_authority(key):
    c=card();c[key]={};assert not validate_supported_teacher_card(c).passed


@pytest.mark.parametrize("operation",["audit","data_export","training"])
def test_only_new_teacher_generation_authorized(operation):
    c=card();c["approval"]["authorized_operations"]=[operation];assert not validate_supported_teacher_card(c).passed


def test_no_other_split_even_with_matching_hash():
    from mtare_topo.governance_field_recovery import selection_sha256
    c=card()
    for row in c["selected_rows"]:row["task"]=row["task"].replace("C01","C02")
    c["tasks"]=[t.replace("C01","C02") for t in c["tasks"]];c["worlds"]=[w.replace("C01","C02") for w in c["worlds"]]
    c["selection_sha256"]=c["approval"]["selection_sha256"]=selection_sha256(c["selected_rows"])
    assert not validate_supported_teacher_card(c).passed


def test_dispatch_only_new_teacher_generation(tmp_path):
    import json
    from mtare_topo.governance import preflight
    from tests.v3.unit.test_governance import make_spec,make_status
    (tmp_path/"card.json").write_text(json.dumps(card()))
    spec=make_spec(3,"teacher_generation");spec["data_card"]="card.json"
    assert preflight(spec,make_status(3),tmp_path).passed
    spec["operation"]="training";assert not preflight(spec,make_status(3),tmp_path).passed
