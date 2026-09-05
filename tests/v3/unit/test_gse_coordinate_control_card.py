import copy
import pytest

from mtare_topo.governance_coordinate_control import SCHEMA, TRAINING, RESTRICTIONS, validate_coordinate_control_card
from mtare_topo.governance_point_axis import validate_point_axis_training_card
from tests.v3.unit.test_gse_point_axis_training_card import card as old_card


def card():
    value = old_card()
    value.update({"schema_version": SCHEMA, "training": copy.deepcopy(TRAINING), "compute_device": "cuda",
                  "visible_fragment_count": 1452, "restrictions": dict.fromkeys(RESTRICTIONS, True),
                  "source_roots": {"sensor": "fixtures/sensor/fit", "teacher": "fixtures/teacher/fit"},
                  "source_seals": {role: f"fixtures/{role}_seal.txt" for role in ("sensor", "teacher", "reference")},
                  "prediction_reference_root": "fixtures/predictions"})
    value["sealed_sources"].update({path: "c" * 64 for path in value["source_seals"].values()})
    value["approval"]["scope"] = "Synthetic independent coordinate control training approval"
    return value


def test_new_card_valid_without_mutation():
    value = card(); before = copy.deepcopy(value)
    assert validate_coordinate_control_card(value).passed
    assert value == before
    assert not validate_point_axis_training_card(value).passed
    assert not validate_coordinate_control_card(old_card()).passed


@pytest.mark.parametrize("field,value", [("compute_device", "cpu"), ("visible_fragment_count", 1452.0),
    ("visible_fragment_count", 1489), ("source_roots", {}), ("source_seals", []),
    ("prediction_reference_root", "../escape"), ("observation_count", 181), ("parent_count", True),
    ("worlds", []), ("selection_sha256", "a" * 64), ("training", None)])
def test_card_drift(field, value):
    actual = card(); actual[field] = value
    assert not validate_coordinate_control_card(actual).passed


@pytest.mark.parametrize("field,value", [("variants", ["raw_no_offset", "raw_slot_offset"]),
    ("variants", list(reversed(TRAINING["variants"]))), ("epochs", 3.0), ("batch_size", True),
    ("steps_per_variant", 541), ("total_steps", 1081), ("optimizer", "AdamW"),
    ("learning_rate", .01), ("checkpoint_selection", "best")])
def test_training_contract_exact(field, value):
    actual = card(); actual["training"][field] = value
    assert not validate_coordinate_control_card(actual).passed


@pytest.mark.parametrize("key", RESTRICTIONS)
def test_restrictions_not_relaxed(key):
    value = card(); value["restrictions"][key] = False
    assert not validate_coordinate_control_card(value).passed


@pytest.mark.parametrize("operation", ["audit", "data_export", "training_and_export"])
def test_no_other_authorization(operation):
    value = card(); value["approval"]["authorized_operations"] = [operation]
    assert not validate_coordinate_control_card(value).passed


def test_no_c02_expansion_even_when_selection_rehashed():
    from mtare_topo.governance_field_recovery import selection_sha256
    value = card()
    for row in value["selected_rows"]: row["task"] = row["task"].replace("C01", "C02")
    value["worlds"] = [p.replace("C01", "C02") for p in value["worlds"]]
    value["selection_sha256"] = value["approval"]["selection_sha256"] = selection_sha256(value["selected_rows"])
    assert not validate_coordinate_control_card(value).passed


def test_only_new_training_schema_dispatches_through_preflight(tmp_path):
    import json
    from mtare_topo.governance import preflight
    from tests.v3.unit.test_governance import make_spec, make_status
    value = card(); (tmp_path / "card.json").write_text(json.dumps(value))
    spec = make_spec(3, "training"); spec["data_card"] = "card.json"
    assert preflight(spec, make_status(3), tmp_path).passed
    spec["operation"] = "data_export"
    assert not preflight(spec, make_status(3), tmp_path).passed


def test_source_root_cannot_smuggle_c07_beneath_fit_name():
    value = card(); value["source_roots"]["teacher"] = "fixtures/C07/fit"
    assert not validate_coordinate_control_card(value).passed
