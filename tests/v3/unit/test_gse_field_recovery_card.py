"""Software-only cards: no source dataset or checkpoint is opened."""
import copy

import pytest

from mtare_topo.governance import validate_data_card
from mtare_topo.governance_field_recovery import (
    INPUT_FIELDS, OUTPUT_FIELDS, RESTRICTIONS, SCHEMA,
    selection_sha256, validate_scoped_field_recovery_card,
)


def card():
    """Complete synthetic fixture; production replaces rows and source hashes."""
    rows = [{"task": f"S{family:02d}_synthetic_C01__c1_mixed", "row_index": row,
             "source_global_sequence_index": (family - 1) * 18 + row}
            for family in range(1, 11) for row in range(18)]
    digest = selection_sha256(rows)
    return {
        "schema_version": SCHEMA, "card_id": "synthetic_field_recovery",
        "purpose": "Restore shared-frame geometry, not labels or model selection",
        "teacher_source": "Sealed inventory metadata for alignment only, never student input",
        "sampling_rule": "Same fixed eighteen observations per topology parent",
        "license_or_allowed_use": "Synthetic software fixture only",
        "partition": "fit", "duration_s": None,
        "time_basis": "distance_sampled_no_acquisition_clock",
        "independent_sampling_unit": "topology_parent",
        "observation_count": 180, "parent_count": 10, "node_count": 100,
        "unique_source_frame_count": 900, "frames_per_observation": 5,
        "worlds": [f"S{family:02d}_synthetic_C01" for family in range(1, 11)],
        "selected_rows": rows, "selection_sha256": digest,
        "input_fields": list(INPUT_FIELDS), "output_fields": list(OUTPUT_FIELDS),
        "restrictions": dict.fromkeys(RESTRICTIONS, True),
        "sealed_sources": {"fixtures/inventory.json": "a" * 64,
                           "fixtures/seed0.pt": "b" * 64},
        "checkpoint": {"path": "fixtures/seed0.pt", "sha256": "b" * 64,
                       "seed": 0, "frozen": True, "selection": "none"},
        "inference": {"primary_observations": 180, "repeat_observations": 18,
                      "total_observations": 198,
                      "repeat_task": "S01_synthetic_C01__c1_mixed"},
        "approval": {"status": "APPROVED", "approved_by": "software_fixture",
                     "approved_at": "2026-09-05", "scope": "Synthetic validation only",
                     "confirmation_reference": "Synthetic fixture, not a user authorization",
                     "authorized_operations": ["data_export"], "authorized_gates": [3],
                     "selection_sha256": digest, "checkpoint_sha256": "b" * 64},
    }


def assert_rejected(value):
    report = validate_scoped_field_recovery_card(value)
    assert not report.passed
    assert report.errors


def test_exact_card_is_valid_without_io(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("validator must perform no file I/O")
    monkeypatch.setattr("builtins.open", forbidden)
    value = card()
    assert validate_scoped_field_recovery_card(value).passed
    assert not validate_data_card(value).passed  # Cannot authorize normal training.


@pytest.mark.parametrize("value", [None, [], "card", 180, False])
def test_non_object_card_rejected(value):
    assert_rejected(value)


@pytest.mark.parametrize("key,value", [
    ("schema_version", "v3_scoped_inventory_card_v1"), ("partition", "validation"),
    ("observation_count", 179), ("observation_count", 180.0), ("parent_count", True),
    ("parent_count", 180), ("node_count", 99), ("node_count", "100"),
    ("unique_source_frame_count", 899), ("frames_per_observation", 1),
    ("duration_s", 90), ("time_basis", "invented_1Hz"),
    ("independent_sampling_unit", "adjacent_frame"), ("worlds", []),
    ("purpose", None), ("sealed_sources", []), ("sealed_sources", {}),
    ("selected_rows", None), ("selection_sha256", "a" * 64),
    ("checkpoint", []), ("inference", None), ("restrictions", None),
    ("approval", None), ("input_fields", list(INPUT_FIELDS) + ["node_id"]),
    ("output_fields", list(OUTPUT_FIELDS) + ["target_event"]),
])
def test_invalid_top_level_contracts(key, value):
    value_card = card(); value_card[key] = value
    assert_rejected(value_card)


def test_duration_must_be_explicitly_null():
    value = card(); del value["duration_s"]
    assert_rejected(value)


@pytest.mark.parametrize("split", ["C02", "C06", "C07", "C08", "C09", "C10"])
def test_other_splits_cannot_enter_even_if_world_and_hash_updated(split):
    value = card()
    for row in value["selected_rows"]:
        row["task"] = row["task"].replace("C01", split)
    value["worlds"] = [world.replace("C01", split) for world in value["worlds"]]
    value["selection_sha256"] = selection_sha256(value["selected_rows"])
    value["approval"]["selection_sha256"] = value["selection_sha256"]
    assert_rejected(value)


@pytest.mark.parametrize("patch", [
    {"task": "../S01_escape_C01__c1_mixed"}, {"task": "S11_extra_C01__c1_mixed"},
    {"task": "S01_synthetic_C01__c0_baseline"}, {"row_index": True},
    {"row_index": -1}, {"row_index": 0.0}, {"source_global_sequence_index": False},
])
def test_bad_row_types_and_paths(patch):
    value = card(); value["selected_rows"][0].update(patch)
    assert_rejected(value)


def test_changed_row_without_selection_reapproval_rejected():
    value = card(); value["selected_rows"][0]["row_index"] = 100
    value["selection_sha256"] = selection_sha256(value["selected_rows"])
    assert_rejected(value)


@pytest.mark.parametrize("source_duplicate", [False, True])
def test_duplicate_rows_or_source_sequences_rejected(source_duplicate):
    value = card()
    if source_duplicate:
        value["selected_rows"][1]["source_global_sequence_index"] = 0
    else:
        value["selected_rows"][1] = copy.deepcopy(value["selected_rows"][0])
    value["selection_sha256"] = selection_sha256(value["selected_rows"])
    value["approval"]["selection_sha256"] = value["selection_sha256"]
    assert_rejected(value)


@pytest.mark.parametrize("bad_row", [None, [], 4, {"extra": float("nan")}])
def test_malformed_rows_are_reported_not_crashes(bad_row):
    value = card(); value["selected_rows"][0] = bad_row
    assert_rejected(value)


@pytest.mark.parametrize("path", ["/absolute", "../outside", "a/../outside", "./x", "a//b", "", "a\\b", "file://x"])
def test_source_paths_must_be_repository_relative(path):
    value = card(); value["sealed_sources"][path] = "c" * 64
    assert_rejected(value)


@pytest.mark.parametrize("digest", [None, [], "x" * 64, "a" * 63, "A" * 64, 123])
def test_bad_source_hashes(digest):
    value = card(); value["sealed_sources"]["fixtures/other"] = digest
    assert_rejected(value)


@pytest.mark.parametrize("patch", [
    {"seed": True}, {"seed": 1}, {"seed": 0.0}, {"frozen": 1},
    {"selection": "best_validation"}, {"sha256": "c" * 64},
    {"path": []}, {"path": "../outside.pt"}, {"path": "fixtures/unsealed.pt"},
])
def test_checkpoint_drift_and_selection_rejected(patch):
    value = card(); value["checkpoint"].update(patch)
    assert_rejected(value)


def test_changed_checkpoint_requires_matching_approval_not_only_new_source_hash():
    value = card(); value["checkpoint"]["sha256"] = "c" * 64
    value["sealed_sources"]["fixtures/seed0.pt"] = "c" * 64
    assert_rejected(value)


@pytest.mark.parametrize("patch", [
    {"primary_observations": 181}, {"repeat_observations": 0},
    {"repeat_observations": 18.0}, {"total_observations": 199},
    {"repeat_task": "S02_synthetic_C01__c1_mixed"},
])
def test_inference_ledger_cannot_expand(patch):
    value = card(); value["inference"].update(patch)
    assert_rejected(value)


@pytest.mark.parametrize("patch", [
    {"status": "PROPOSED"}, {"authorized_operations": ["training"]},
    {"authorized_operations": ["data_export", "training"]}, {"authorized_operations": ["audit"]},
    {"authorized_gates": [4]}, {"authorized_gates": [3.0]},
    {"authorized_gates": [3, 4]}, {"confirmation_reference": ""},
    {"checkpoint_sha256": "c" * 64}, {"selection_sha256": None},
])
def test_authorization_scope_drift(patch):
    value = card(); value["approval"].update(patch)
    assert_rejected(value)


@pytest.mark.parametrize("restriction", RESTRICTIONS)
def test_no_restriction_can_be_relaxed(restriction):
    value = card(); value["restrictions"][restriction] = False
    assert_rejected(value)
