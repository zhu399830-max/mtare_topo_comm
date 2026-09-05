"""Synthetic exact-coordinate audit authorization; never model/training scope."""
import json

import pytest

from mtare_topo.governance import preflight
from mtare_topo.governance_field_recovery import INPUT_FIELDS, selection_sha256
from mtare_topo.governance_inventory import (
    validate_scoped_coordinate_audit_card,
    validate_scoped_inventory_card,
)
from tests.v3.unit.test_governance import make_spec, make_status
from tests.v3.unit.test_gse_scoped_inventory import card


def coordinate_card():
    value = card()
    rows = [{"task": f"S{family:02d}_fixture_C01__c1_mixed", "row_index": index,
             "source_global_sequence_index": family * 100 + index}
            for family in range(1, 11) for index in range(18)]
    value.update({
        "schema_version": "v3_scoped_coordinate_audit_card_v1",
        "selected_rows": rows, "selection_sha256": selection_sha256(rows),
        "worlds": sorted({row["task"].split("__")[0] for row in rows}),
        "observation_count": 180, "parent_count": 10, "node_count": 100,
        "unique_source_frame_count": 900, "frames_per_observation": 5,
        "input_fields": list(INPUT_FIELDS),
        "sensor_usage": "existing_coordinate_support_only_no_model_or_new_labels",
    })
    value["approval"]["selection_sha256"] = value["selection_sha256"]
    value["restrictions"].pop("no_sensor_decoding")
    value["restrictions"]["existing_selected_sensor_only"] = True
    return value


def _rebind(value):
    value["selection_sha256"] = selection_sha256(value["selected_rows"])
    value["approval"]["selection_sha256"] = value["selection_sha256"]
    value["worlds"] = sorted({row["task"].split("__")[0] for row in value["selected_rows"]})


def test_exact_static_coordinate_audit_card_is_valid_without_model_permission():
    value = coordinate_card()
    report = validate_scoped_coordinate_audit_card(value)
    assert report.passed, report.errors
    assert value["duration_s"] is None
    assert value["restrictions"]["no_model_or_checkpoint"] is True
    assert not validate_scoped_inventory_card(value).passed


@pytest.mark.parametrize("cohort", ["C02", "C07", "C10"])
def test_coordinate_audit_never_authorizes_other_cohorts_even_if_hashes_are_rebound(cohort):
    value = coordinate_card()
    for row in value["selected_rows"]:
        row["task"] = row["task"].replace("C01", cohort)
    _rebind(value)
    assert not validate_scoped_coordinate_audit_card(value).passed


@pytest.mark.parametrize("where", ["card", "approval", "rows", "row_order"])
def test_selection_hash_is_bound_to_full_ordered_selection_and_approval(where):
    value = coordinate_card()
    if where == "card":
        value["selection_sha256"] = "0" * 64
    elif where == "approval":
        value["approval"]["selection_sha256"] = "0" * 64
    elif where == "rows":
        value["selected_rows"][0]["source_global_sequence_index"] += 9999
    else:
        value["selected_rows"].reverse()
    assert not validate_scoped_coordinate_audit_card(value).passed


@pytest.mark.parametrize("restriction", [
    "no_model_or_checkpoint", "no_optimizer", "no_teacher_generation", "no_calibration",
    "existing_selected_sensor_only", "explicit_task_rows_only", "no_test_worlds", "read_only_sources",
])
def test_required_scope_restrictions_cannot_be_disabled(restriction):
    value = coordinate_card()
    value["restrictions"][restriction] = False
    assert not validate_scoped_coordinate_audit_card(value).passed


@pytest.mark.parametrize("change", ["missing_field", "identity_field", "wrong_usage", "no_fields"])
def test_raw_input_fields_and_use_are_exact(change):
    value = coordinate_card()
    if change == "missing_field":
        value["input_fields"].pop()
    elif change == "identity_field":
        value["input_fields"].append("primitive_membership_code")
    elif change == "wrong_usage":
        value["sensor_usage"] = "model_inference"
    else:
        del value["input_fields"]
    assert not validate_scoped_coordinate_audit_card(value).passed


@pytest.mark.parametrize("field", [
    "observation_count", "parent_count", "node_count", "unique_source_frame_count", "frames_per_observation",
])
def test_population_counts_cannot_be_changed_or_omitted(field):
    value = coordinate_card()
    value[field] += 1
    assert not validate_scoped_coordinate_audit_card(value).passed
    del value[field]
    assert not validate_scoped_coordinate_audit_card(value).passed


@pytest.mark.parametrize("field", [
    "observation_count", "parent_count", "node_count", "unique_source_frame_count", "frames_per_observation",
])
def test_population_counts_require_integers_not_equal_valued_floats(field):
    value = coordinate_card()
    value[field] = float(value[field])
    assert not validate_scoped_coordinate_audit_card(value).passed


def test_exact_eighteen_per_task_is_required_not_just_180_total():
    value = coordinate_card()
    value["selected_rows"][0]["task"] = value["selected_rows"][18]["task"]
    value["selected_rows"][0]["row_index"] = 1000
    value["selected_rows"][0]["source_global_sequence_index"] = 1000
    _rebind(value)
    assert not validate_scoped_coordinate_audit_card(value).passed


@pytest.mark.parametrize("duplicate", ["row", "source"])
def test_duplicate_row_or_source_identity_is_rejected_even_with_new_hash(duplicate):
    value = coordinate_card()
    key = "row_index" if duplicate == "row" else "source_global_sequence_index"
    value["selected_rows"][1][key] = value["selected_rows"][0][key]
    _rebind(value)
    assert not validate_scoped_coordinate_audit_card(value).passed


@pytest.mark.parametrize("approval", [None, [], "APPROVED"])
def test_malformed_approval_is_a_validation_failure_not_an_exception(approval):
    value = coordinate_card()
    value["approval"] = approval
    assert not validate_scoped_coordinate_audit_card(value).passed


@pytest.mark.parametrize("operation", ["training", "data_export", "teacher_generation", "threshold_calibration"])
def test_card_cannot_authorize_other_operations(operation):
    value = coordinate_card()
    value["approval"]["authorized_operations"] = [operation]
    assert not validate_scoped_coordinate_audit_card(value).passed


def test_old_inventory_schema_still_refuses_raw_sensor_access():
    value = card()
    assert validate_scoped_inventory_card(value).passed
    value["restrictions"]["no_sensor_decoding"] = False
    value["restrictions"]["existing_selected_sensor_only"] = True
    value["input_fields"] = list(INPUT_FIELDS)
    assert not validate_scoped_inventory_card(value).passed


def test_preflight_dispatches_coordinate_card_only_for_gate3_audit(tmp_path):
    value = coordinate_card()
    path = tmp_path / "card.json"
    path.write_text(json.dumps(value))
    spec = make_spec(3, "audit")
    spec["data_card"] = "card.json"
    report = preflight(spec, make_status(3), tmp_path)
    assert report.passed, report.errors
    for operation in ("training", "data_export", "teacher_generation"):
        spec["operation"] = operation
        value["approval"]["authorized_operations"] = [operation]
        path.write_text(json.dumps(value))
        assert not preflight(spec, make_status(3), tmp_path).passed
