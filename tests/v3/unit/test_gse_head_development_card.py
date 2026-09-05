"""Synthetic metadata-card validation only; no actual source arrays are read."""
import copy

import pytest

from mtare_topo.governance import validate_data_card
from mtare_topo.governance_head_development import (
    FIELDS, HEAD_EXPOSURE, RESTRICTIONS, SAMPLING_RULE, SCHEMA, UNKNOWN_FIELDS,
    validate_head_development_metadata_card,
)


def card():
    return {
        "schema_version": SCHEMA, "card_id": "synthetic_C02_metadata",
        "purpose": "Choose metadata-only C02 development rows without viewing model scores",
        "teacher_source": "Existing immutable integer teacher metadata only",
        "license_or_allowed_use": "Synthetic software fixture only",
        "head_exposure": HEAD_EXPOSURE, "partition": "fit",
        "independent_sampling_unit": "topology_parent", "duration_s": None,
        "time_basis": "distance_sampled_no_acquisition_clock",
        "target_observations": 180, "parent_count": 10, "rows_per_parent": 18,
        "source_global_sequence_indices": None, "unique_source_frame_count": None,
        "visible_fragment_count": None, "fields": list(FIELDS), "sampling_rule": SAMPLING_RULE,
        "tasks": [f"S{i:02d}_synthetic_C02__c1_mixed" for i in range(1, 11)],
        "worlds": [f"S{i:02d}_synthetic_C02" for i in range(1, 11)],
        "source_roots": {"teacher": "fixtures/teacher/fit"},
        "source_seals": {"teacher": "fixtures/evidence_sha256.txt"},
        "sealed_sources": {"fixtures/evidence_sha256.txt": "a" * 64},
        "restrictions": dict.fromkeys(RESTRICTIONS, True),
        "approval": {"status": "APPROVED", "approved_by": "software_fixture",
                     "approved_at": "2026-09-05", "scope": "Exact synthetic C02 metadata audit only",
                     "confirmation_reference": "Synthetic fixture, not actual user authorization",
                     "authorized_operations": ["audit"], "authorized_gates": [3]},
    }


def reject(value):
    result = validate_head_development_metadata_card(value)
    assert not result.passed and result.errors


def test_honest_unknown_population_card_valid_without_io(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("metadata card validator must not open source files")
    monkeypatch.setattr("builtins.open", forbidden)
    value = card(); before = copy.deepcopy(value)
    assert validate_head_development_metadata_card(value).passed
    assert value == before
    assert not validate_data_card(value).passed


@pytest.mark.parametrize("value", [None, [], "x", True, 180])
def test_bad_top_level_type(value):
    reject(value)


@pytest.mark.parametrize("field,value", [
    ("schema_version", "v3_scoped_inventory_card_v1"), ("head_exposure", "strict_test"),
    ("purpose", ""), ("teacher_source", None), ("partition", "dev"),
    ("independent_sampling_unit", "adjacent_frame"), ("duration_s", 90),
    ("time_basis", "invented_seconds"), ("target_observations", 180.0),
    ("target_observations", 179), ("parent_count", True), ("parent_count", 9),
    ("rows_per_parent", 18.0), ("rows_per_parent", 19),
    ("fields", list(FIELDS) + ["range_m"]), ("fields", list(FIELDS) + ["axis_control_current_sensor_m"]),
    ("fields", list(FIELDS)[::-1]), ("sampling_rule", "best_score"),
    ("worlds", []), ("tasks", None), ("tasks", []), ("source_roots", None),
    ("source_seals", []), ("sealed_sources", {}), ("sealed_sources", []),
    ("restrictions", None), ("approval", []),
])
def test_scope_type_and_count_changes_rejected(field, value):
    actual = card(); actual[field] = value
    reject(actual)


@pytest.mark.parametrize("field", UNKNOWN_FIELDS + ("duration_s",))
def test_unknown_counts_and_clock_must_be_explicit_null(field):
    value = card(); del value[field]
    reject(value)


@pytest.mark.parametrize("field,value", [
    ("source_global_sequence_indices", []), ("source_global_sequence_indices", list(range(180))),
    ("unique_source_frame_count", 900), ("visible_fragment_count", 1452),
])
def test_no_invented_completed_population(field, value):
    actual = card(); actual[field] = value
    reject(actual)


@pytest.mark.parametrize("split", ["C01", "C03", "C06", "C07", "C08", "C09", "C10"])
def test_only_c02_even_if_worlds_updated(split):
    value = card()
    value["tasks"] = [task.replace("C02", split) for task in value["tasks"]]
    value["worlds"] = [world.replace("C02", split) for world in value["worlds"]]
    reject(value)


@pytest.mark.parametrize("task", [None, [], 1, "../escape", "S11_synthetic_C02__c1_mixed",
                                  "S01_synthetic_C02__c0_baseline"])
def test_bad_tasks_do_not_crash(task):
    value = card(); value["tasks"][0] = task
    reject(value)


def test_duplicate_task_rejected():
    value = card(); value["tasks"][1] = value["tasks"][0]
    value["worlds"][1] = value["worlds"][0]
    reject(value)


def test_task_order_must_be_canonical():
    value = card(); value["tasks"].reverse()
    reject(value)


@pytest.mark.parametrize("root", ["/absolute/fit", "../escape/fit", "fixtures/C10/fit",
                                  "fixtures/dev", "fixtures//fit", "fixtures/fit/", [], None])
def test_source_root_must_be_fit_and_repository_relative(root):
    value = card(); value["source_roots"]["teacher"] = root
    reject(value)


def test_no_second_data_source_root():
    value = card(); value["source_roots"]["sensor"] = "fixtures/sensor/fit"
    reject(value)


@pytest.mark.parametrize("path", ["/absolute", "../escape", "./x", "a//b", "a\\b", "file://x"])
def test_source_hash_paths_strictly_relative(path):
    value = card(); value["sealed_sources"][path] = "c" * 64
    reject(value)


@pytest.mark.parametrize("digest", [None, 123, "a" * 63, "G" * 64, "A" * 64])
def test_source_sha256_format(digest):
    value = card(); value["sealed_sources"]["fixtures/other"] = digest
    reject(value)


@pytest.mark.parametrize("seal", [[], None, "../outside", "fixtures/not_hashed.txt"])
def test_source_seal_requires_exact_hashed_path(seal):
    value = card(); value["source_seals"]["teacher"] = seal
    reject(value)


@pytest.mark.parametrize("field", RESTRICTIONS)
def test_each_metadata_only_restriction_enforced(field):
    value = card(); value["restrictions"][field] = False
    reject(value)


@pytest.mark.parametrize("field", ["checkpoint", "inference", "training", "selected_rows",
                                  "selection_sha256", "observation_count", "model_inputs", "geomteacher_fields"])
def test_no_export_or_training_authority_smuggled_in(field):
    value = card(); value[field] = None
    reject(value)


@pytest.mark.parametrize("patch", [
    {"authorized_operations": ["data_export"]}, {"authorized_operations": ["training"]},
    {"authorized_operations": ["audit", "data_export"]}, {"authorized_gates": [3.0]},
    {"authorized_gates": [4]}, {"authorized_gates": [3, 4]}, {"status": "PROPOSED"},
    {"scope": ""}, {"confirmation_reference": ""},
])
def test_approval_bound_to_metadata_audit_gate3(patch):
    value = card(); value["approval"].update(patch)
    reject(value)
