from copy import deepcopy
import pytest

from mtare_topo.governance_partial_structure import (
    SCHEMA,METHOD,SOURCE_NAMES,RAW_COUNTS,RESOURCES,RESTRICTIONS,PRODUCER_FILES,
    COORDINATE_COMMIT,TEACHER_COMMIT,validate_partial_structure_export_card,
)
from mtare_topo.governance_field_recovery import selection_sha256
from tests.v3.unit.test_gse_supported_teacher_card import card as previous


def card():
    old=previous()
    keys=("purpose","teacher_source","sampling_rule","license_or_allowed_use","partition","time_basis","duration_s",
        "independent_sampling_unit","observation_count","parent_count","unique_source_frame_count","frames_per_observation",
        "worlds","selected_rows","selection_sha256","tasks","legacy_node_count_inventory_only","visible_fragment_count")
    value={k:deepcopy(old[k]) for k in keys}
    sources={k:{"path":"fixtures/"+basename,"sha256":"a"*64} for k,basename in SOURCE_NAMES.items()}
    seals={k:"fixtures/"+k+"_seal.txt" for k in ("coordinate_control","local_teacher","supported_teacher")}
    provenance={role:{"git_commit":COORDINATE_COMMIT if role=="coordinate_control" else TEACHER_COMMIT,
        "spec":"configs/"+role+"_spec.json","card":"configs/"+role+"_card.json",
        "source_sha256":dict.fromkeys(paths,"b"*64)} for role,paths in PRODUCER_FILES.items()}
    bound={v["path"]:v["sha256"] for v in sources.values()}
    bound.update(dict.fromkeys(seals.values(),"c"*64))
    bound.update({v[k]:"d"*64 for v in provenance.values() for k in ("spec","card")})
    value.update({"schema_version":SCHEMA,"operation":"data_export","card_id":"synthetic_partial_export",
        "identity_provenance":"Explicit sealed old audit bridge; cached source IDs are inherited, not embedded",
        "cache_embeds_frame_identity":False,"prediction_branch":"raw_coordinates","native_method":METHOD,
        "raw_teacher_counts":deepcopy(RAW_COUNTS),"effective_counts":{"gt":None,"predicted":None},"capacity_ready":False,
        "resources":dict(RESOURCES),"restrictions":dict.fromkeys(RESTRICTIONS,True),"sources":sources,
        "sealed_sources":bound,"source_seals":seals,"producer_provenance":provenance,
        "approval":{**old["approval"],"authorized_operations":["data_export"],"sources":deepcopy(sources)}})
    return value


def test_exact_export_card_no_mutation_or_old_teacher_authority():
    value=card();before=deepcopy(value)
    assert validate_partial_structure_export_card(value).passed and value==before
    assert not validate_partial_structure_export_card(previous()).passed


@pytest.mark.parametrize("key,value",[("operation","training"),("schema_version","v3_supported_construction_teacher_card_v1"),
    ("observation_count",180.),("parent_count",True),("unique_source_frame_count",899),("visible_fragment_count",1453),
    ("legacy_node_count_inventory_only",99),("frames_per_observation",4),("duration_s",90),("duration_s",0),
    ("partition","dev"),("time_basis","seconds"),("independent_sampling_unit","frame"),("worlds",[]),("tasks",[]),
    ("selected_rows",None),("selection_sha256","f"*64),("effective_counts",{"gt":1206,"predicted":None}),
    ("effective_counts",{}),("capacity_ready",True),("cache_embeds_frame_identity",True),
    ("prediction_branch","mean_broadcast_coordinates"),("native_method","l1"),("sources",{}),("source_seals",[]),
    ("producer_provenance",None),("sealed_sources",{}),("raw_teacher_counts",None),("resources",None)])
def test_top_level_drift(key,value):
    c=card();c[key]=value;assert not validate_partial_structure_export_card(c).passed


@pytest.mark.parametrize("key",RESTRICTIONS)
@pytest.mark.parametrize("value",[False,1])
def test_no_relaxed_restrictions(key,value):
    c=card();c["restrictions"][key]=value;assert not validate_partial_structure_export_card(c).passed


@pytest.mark.parametrize("key",["training","optimizer","checkpoint","checkpoints","inference","sensor_root","teacher_root"])
def test_no_extra_authority(key):
    c=card();c[key]={};assert not validate_partial_structure_export_card(c).passed


@pytest.mark.parametrize("patch",[{"task":"S01_synthetic_C02__c1_mixed"},{"row_index":True},
    {"source_global_sequence_index":0.0},{"source_global_sequence_index":-1}])
def test_row_identity_drift_even_after_hash_update(patch):
    c=card();c["selected_rows"][0].update(patch)
    c["selection_sha256"]=c["approval"]["selection_sha256"]=selection_sha256(c["selected_rows"])
    assert not validate_partial_structure_export_card(c).passed


@pytest.mark.parametrize("field",["row_index","source_global_sequence_index"])
def test_duplicate_id_rejected(field):
    c=card();c["selected_rows"][1][field]=c["selected_rows"][0][field]
    c["selection_sha256"]=c["approval"]["selection_sha256"]=selection_sha256(c["selected_rows"])
    assert not validate_partial_structure_export_card(c).passed


@pytest.mark.parametrize("key",SOURCE_NAMES)
@pytest.mark.parametrize("patch",[{"path":"../outside.npz"},{"sha256":"bad"},{"sha256":"f"*64},{"path":[]}])
def test_all_source_path_hash_binding(key,patch):
    c=card();c["sources"][key].update(patch)
    assert not validate_partial_structure_export_card(c).passed


@pytest.mark.parametrize("operation",[["training"],["teacher_generation"],["data_export","training"]])
def test_approval_only_export(operation):
    c=card();c["approval"]["authorized_operations"]=operation
    assert not validate_partial_structure_export_card(c).passed


def test_approval_hash_drift_and_extra_payload_denied():
    for change in ("source","checkpoint","numeric"):
        c=card()
        if change=="source": c["approval"]["sources"]["predictions"]["sha256"]="f"*64
        else: c["sealed_sources"]["extra/model.pt" if change=="checkpoint" else "extra/raw.npz"]="f"*64
        assert not validate_partial_structure_export_card(c).passed


def test_even_rehashed_and_approved_test_source_path_is_out_of_scope():
    c=card();old=c["sources"]["predictions"]["path"]
    new="fixtures/C07/all_predictions.npz"
    c["sources"]["predictions"]["path"]=new
    c["sealed_sources"][new]=c["sealed_sources"].pop(old)
    c["approval"]["sources"]=deepcopy(c["sources"])
    assert not validate_partial_structure_export_card(c).passed


@pytest.mark.parametrize("role",PRODUCER_FILES)
@pytest.mark.parametrize("field,value",[("git_commit","f"*40),("spec","unbound/spec.json"),("card",[]),("source_sha256",{})])
def test_producer_binding(role,field,value):
    c=card();c["producer_provenance"][role][field]=value
    assert not validate_partial_structure_export_card(c).passed


def test_counts_and_resources_do_not_accept_float_or_bool():
    c=card();c["raw_teacher_counts"]["events"]["terminal"]=False
    assert not validate_partial_structure_export_card(c).passed
    c=card();c["resources"]["wall_time_cap_s"]=120.
    assert not validate_partial_structure_export_card(c).passed


def test_source_schema_matches_reader_without_importing_numpy_into_governance():
    from mtare_topo.data.gse_partial_structure_cache import SOURCE_NAMES as reader_names
    assert SOURCE_NAMES==reader_names
