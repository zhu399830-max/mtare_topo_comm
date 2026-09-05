"""Exact-budget training-card software tests; no real sample/checkpoint I/O."""
import copy

import pytest

from mtare_topo.governance import validate_data_card
from mtare_topo.governance_field_recovery import selection_sha256, validate_scoped_field_recovery_card
from mtare_topo.governance_point_axis import (
    GEOMTEACHER_FIELDS, RESTRICTIONS, SCHEMA, TRAINING, validate_point_axis_training_card,
)
from tests.v3.unit.test_gse_field_recovery_card import card as export_card


def card():
    """A new training approval is required; this is only a synthetic fixture."""
    value = export_card()
    value.update({"schema_version": SCHEMA, "card_id": "synthetic_point_axis_training",
                  "purpose": "Compare two point axis readouts with identical fixed small training budget",
                  "training": copy.deepcopy(TRAINING), "geomteacher_fields": list(GEOMTEACHER_FIELDS),
                  "restrictions": dict.fromkeys(RESTRICTIONS, True)})
    del value["inference"], value["output_fields"]
    value["approval"].update({"authorized_operations": ["training"],
                              "scope": "Synthetic new exact geometry training approval only",
                              "confirmation_reference": "Synthetic fixture, not an actual user approval"})
    return value


def reject(value):
    result = validate_point_axis_training_card(value)
    assert not result.passed and result.errors


def reseal_rows(value):
    digest = selection_sha256(value["selected_rows"])
    value["selection_sha256"] = value["approval"]["selection_sha256"] = digest


def test_exact_new_training_card_valid_without_source_io(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("validator cannot read source data or checkpoint")
    monkeypatch.setattr("builtins.open", forbidden)
    value = card()
    before = copy.deepcopy(value)
    assert validate_point_axis_training_card(value).passed
    assert value == before
    assert not validate_data_card(value).passed
    assert not validate_scoped_field_recovery_card(value).passed


def test_old_export_card_retains_export_only_authority():
    value = export_card()
    assert validate_scoped_field_recovery_card(value).passed
    reject(value)
    value["schema_version"] = SCHEMA
    value["approval"]["authorized_operations"] = ["training"]
    reject(value)
    value["training"] = copy.deepcopy(TRAINING)
    value["geomteacher_fields"] = list(GEOMTEACHER_FIELDS)
    reject(value)  # Old export bans/fields are contradictory, not new authority.


@pytest.mark.parametrize("value", [None, [], "x", False, 1])
def test_bad_top_level_type(value):
    reject(value)


@pytest.mark.parametrize("key,value", [
    ("observation_count", 180.0), ("observation_count", 181),
    ("parent_count", True), ("parent_count", 180), ("node_count", "100"),
    ("node_count", 99), ("unique_source_frame_count", 899),
    ("frames_per_observation", 5.0), ("duration_s", 90),
    ("time_basis", "invented_seconds"), ("independent_sampling_unit", "frames"),
    ("partition", "development"), ("worlds", []), ("selected_rows", []),
    ("selected_rows", None), ("selection_sha256", "f" * 64),
    ("sealed_sources", {}), ("sealed_sources", None), ("checkpoint", []),
    ("training", None), ("training", []), ("approval", []), ("restrictions", None),
    ("teacher_source", ""), ("purpose", None),
])
def test_bad_metadata_contracts(key, value):
    actual = card(); actual[key] = value
    reject(actual)


def test_clock_absence_is_not_explicit_null():
    value = card(); del value["duration_s"]
    reject(value)


@pytest.mark.parametrize("split", ["C02", "C06", "C07", "C08", "C09", "C10"])
def test_no_other_split_even_after_rehash(split):
    value = card()
    for row in value["selected_rows"]:
        row["task"] = row["task"].replace("C01", split)
    value["worlds"] = [world.replace("C01", split) for world in value["worlds"]]
    reseal_rows(value)
    reject(value)


@pytest.mark.parametrize("patch", [
    {"row_index": True}, {"row_index": 0.0}, {"row_index": -1},
    {"source_global_sequence_index": False}, {"source_global_sequence_index": -1},
    {"task": "../S01_synthetic_C01__c1_mixed"}, {"task": "S11_synthetic_C01__c1_mixed"},
])
def test_invalid_row_types_and_identity(patch):
    value = card(); value["selected_rows"][0].update(patch)
    reseal_rows(value)
    reject(value)


@pytest.mark.parametrize("field", ["row_index", "source_global_sequence_index"])
def test_duplicate_row_or_source_rejected_after_rehash(field):
    value = card(); value["selected_rows"][1][field] = value["selected_rows"][0][field]
    reseal_rows(value)
    reject(value)


def test_same_count_changed_rows_need_new_approval_binding():
    value = card(); value["selected_rows"][0]["row_index"] = 99
    value["selection_sha256"] = selection_sha256(value["selected_rows"])
    reject(value)


@pytest.mark.parametrize("field,value", [
    ("variants", ["raw_slot_offset"]), ("variants", ["raw_slot_offset", "raw_no_offset"]),
    ("epochs", 4), ("epochs", 3.0), ("batch_size", True), ("batch_size", 2),
    ("steps_per_variant", 540.0), ("steps_per_variant", 541),
    ("total_steps", 1081), ("total_steps", 1080.0), ("seed", False), ("seed", 1),
    ("optimizer", "AdamW"), ("learning_rate", 0.01), ("learning_rate", float("nan")),
    ("weight_decay", False), ("weight_decay", 0.1), ("weight_decay", 0),
    ("sample_schedule", "with_replacement"), ("checkpoint_selection", "best_train_loss"),
    ("loss", "geometry_and_event_loss"),
])
def test_training_values_and_types_are_frozen(field, value):
    actual = card(); actual["training"][field] = value
    reject(actual)


def test_unfrozen_optimizer_options_cannot_be_added():
    value = card(); value["training"]["extra_epochs"] = 1
    reject(value)


@pytest.mark.parametrize("key", TRAINING)
def test_every_training_parameter_is_required(key):
    value = card(); del value["training"][key]
    reject(value)


@pytest.mark.parametrize("field", ["node_id", "degree", "traversal_id", "future_pose"])
def test_identity_and_future_information_are_not_model_inputs(field):
    value = card(); value["input_fields"].append(field)
    reject(value)


@pytest.mark.parametrize("field", ["primitive_index", "node_id", "event_target", "width_m"])
def test_no_additional_supervision_fields(field):
    value = card(); value["geomteacher_fields"].append(field)
    reject(value)


@pytest.mark.parametrize("field", RESTRICTIONS)
def test_training_restrictions_mandatory(field):
    value = card(); value["restrictions"][field] = False
    reject(value)


@pytest.mark.parametrize("field", ["no_training", "no_optimizer"])
@pytest.mark.parametrize("value", [True, False])
def test_export_bans_not_silently_ignored(field, value):
    actual = card(); actual["restrictions"][field] = value
    reject(actual)


@pytest.mark.parametrize("field", ["inference", "output_fields", "identity_reference"])
def test_old_export_fields_rejected(field):
    value = card(); value[field] = {}
    reject(value)


@pytest.mark.parametrize("path", ["/absolute", "../escape", "./x", "a//b", "a\\b", "file://x"])
def test_source_paths_strictly_relative(path):
    value = card(); value["sealed_sources"][path] = "c" * 64
    reject(value)


@pytest.mark.parametrize("digest", [None, 123, "a" * 63, "G" * 64, "A" * 64])
def test_source_digest_type_and_format(digest):
    value = card(); value["sealed_sources"]["fixtures/other"] = digest
    reject(value)


@pytest.mark.parametrize("patch", [
    {"seed": False}, {"seed": 1}, {"frozen": 1}, {"frozen": False},
    {"selection": "best_validation"}, {"path": []}, {"path": "missing.pt"},
    {"sha256": "c" * 64},
])
def test_frozen_backbone_checkpoint_enforced(patch):
    value = card(); value["checkpoint"].update(patch)
    reject(value)


def test_changed_checkpoint_hash_needs_new_training_approval():
    value = card(); value["checkpoint"]["sha256"] = "c" * 64
    value["sealed_sources"]["fixtures/seed0.pt"] = "c" * 64
    reject(value)


@pytest.mark.parametrize("patch", [
    {"authorized_operations": ["data_export"]}, {"authorized_operations": ["training", "data_export"]},
    {"authorized_operations": ["audit"]}, {"authorized_gates": [3.0]},
    {"authorized_gates": [4]}, {"authorized_gates": [3, 4]},
    {"status": "PROPOSED"}, {"scope": ""}, {"confirmation_reference": ""},
    {"selection_sha256": None}, {"checkpoint_sha256": None},
])
def test_new_training_approval_scope_is_exact(patch):
    value = card(); value["approval"].update(patch)
    reject(value)
