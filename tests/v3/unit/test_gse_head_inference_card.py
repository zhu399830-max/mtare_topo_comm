"""Synthetic inference authorization tests; no real data/checkpoint reads."""
import copy

import pytest

from mtare_topo.governance_field_recovery import INPUT_FIELDS, selection_sha256
from mtare_topo.governance_head_development import validate_head_development_metadata_card
from mtare_topo.governance_head_inference import (
    GEOMTEACHER_FIELDS, INFERENCE_COUNTS, OUTPUT_FIELDS, RESTRICTIONS, SCHEMA,
    validate_head_development_inference_card,
)
from tests.v3.unit.test_gse_head_development_card import card as metadata_card


def card():
    value = metadata_card()
    rows = [{"task": task, "row_index": i, "source_global_sequence_index": p * 18 + i,
             "frame_rows": list(range(i * 5, i * 5 + 5)),
             "visible_fragments": 9 if p * 18 + i < 49 else 8}
            for p, task in enumerate(value["tasks"]) for i in range(18)]
    for key in ("target_observations", "fields", "source_global_sequence_indices"):
        del value[key]
    digest = selection_sha256(rows)
    value.update({"schema_version": SCHEMA, "card_id": "synthetic_C02_inference",
                  "selected_rows": rows, "selection_sha256": digest, "observation_count": 180,
                  "unique_source_frame_count": 900, "visible_fragment_count": 1489,
                  "frames_per_observation": 5, "input_fields": list(INPUT_FIELDS),
                  "geomteacher_fields": list(GEOMTEACHER_FIELDS), "output_fields": list(OUTPUT_FIELDS),
                  "restrictions": dict.fromkeys(RESTRICTIONS, True),
                  "source_roots": {"sensor": "fixtures/sensor/fit", "teacher": "fixtures/teacher/fit"},
                  "source_seals": {"sensor": "fixtures/sensor_seal.txt", "teacher": "fixtures/teacher_seal.txt"},
                  "metadata_selection_path": "fixtures/selected_rows.json",
                  "checkpoints": {role: {"path": f"fixtures/{role}.pt", "sha256": char * 64,
                                         "seed": 0, "frozen": True, "selection": "none"}
                                  for role, char in (("backbone", "b"), ("head", "c"))},
                  "sealed_sources": {"fixtures/sensor_seal.txt": "a" * 64,
                                     "fixtures/teacher_seal.txt": "d" * 64,
                                     "fixtures/selected_rows.json": "e" * 64,
                                     "fixtures/backbone.pt": "b" * 64,
                                     "fixtures/head.pt": "c" * 64},
                  "inference": {**INFERENCE_COUNTS, "repeat_task": value["tasks"][0]}})
    value["approval"].update({"authorized_operations": ["data_export"], "selection_sha256": digest,
                              "checkpoint_sha256": {"backbone": "b" * 64, "head": "c" * 64},
                              "scope": "Synthetic exact inference approval only"})
    return value


def reject(value):
    result = validate_head_development_inference_card(value)
    assert not result.passed and result.errors


def reseal(value):
    value["selection_sha256"] = selection_sha256(value["selected_rows"])
    value["approval"]["selection_sha256"] = value["selection_sha256"]


def test_valid_inference_card_no_io_no_mutation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("card validation must not open actual arrays or checkpoints")
    monkeypatch.setattr("builtins.open", forbidden)
    value = card(); before = copy.deepcopy(value)
    assert validate_head_development_inference_card(value).passed
    assert value == before
    assert not validate_head_development_metadata_card(value).passed


def test_metadata_approval_cannot_be_used_for_inference():
    value = metadata_card()
    assert validate_head_development_metadata_card(value).passed
    reject(value)
    value["schema_version"] = SCHEMA
    value["approval"]["authorized_operations"] = ["data_export"]
    reject(value)


@pytest.mark.parametrize("value", [None, [], "card", True, 180])
def test_non_object_card(value):
    reject(value)


@pytest.mark.parametrize("field,value", [
    ("schema_version", "v3_head_development_metadata_card_v1"), ("purpose", ""),
    ("head_exposure", "strict_test"), ("partition", "test"),
    ("independent_sampling_unit", "frame"), ("duration_s", 900),
    ("time_basis", "fixed_1hz"), ("observation_count", 180.0), ("observation_count", 181),
    ("parent_count", True), ("rows_per_parent", 19), ("unique_source_frame_count", None),
    ("visible_fragment_count", 1452), ("frames_per_observation", 5.0),
    ("selected_rows", None), ("selected_rows", []), ("tasks", []), ("worlds", []),
    ("selection_sha256", "f" * 64), ("input_fields", list(INPUT_FIELDS) + ["node_id"]),
    ("geomteacher_fields", list(GEOMTEACHER_FIELDS) + ["event_target"]),
    ("output_fields", list(OUTPUT_FIELDS) + ["raw_slot_offset"]),
    ("source_roots", None), ("source_seals", []), ("sealed_sources", {}),
    ("metadata_selection_path", None), ("checkpoints", []), ("inference", []),
    ("restrictions", None), ("approval", None),
])
def test_contract_drift(field, value):
    actual = card(); actual[field] = value
    reject(actual)


def test_clock_null_must_be_explicit():
    value = card(); del value["duration_s"]
    reject(value)


@pytest.mark.parametrize("split", ["C01", "C03", "C06", "C07", "C08", "C09", "C10"])
def test_other_split_rejected_even_if_rehashed(split):
    value = card()
    for row in value["selected_rows"]:
        row["task"] = row["task"].replace("C02", split)
    value["tasks"] = [task.replace("C02", split) for task in value["tasks"]]
    value["worlds"] = [world.replace("C02", split) for world in value["worlds"]]
    reseal(value)
    reject(value)


@pytest.mark.parametrize("patch", [
    {"row_index": True}, {"row_index": 0.0}, {"row_index": -1},
    {"source_global_sequence_index": False}, {"task": "../outside"},
    {"frame_rows": [0, 1, 2, 3, 5]}, {"frame_rows": [4, 3, 2, 1, 0]},
    {"frame_rows": [0, 1, 2, 3]}, {"frame_rows": [0.0, 1, 2, 3, 4]},
    {"frame_rows": [False, 1, 2, 3, 4]}, {"frame_rows": None},
    {"visible_fragments": True}, {"visible_fragments": 9.0},
    {"visible_fragments": 0}, {"visible_fragments": 33},
])
def test_selected_row_integrity(patch):
    value = card(); value["selected_rows"][0].update(patch)
    reseal(value)
    reject(value)


@pytest.mark.parametrize("field", ["row_index", "source_global_sequence_index", "frame_rows"])
def test_duplicate_rows_ids_or_frames_rejected(field):
    value = card(); value["selected_rows"][1][field] = value["selected_rows"][0][field]
    reseal(value)
    reject(value)


def test_fragment_total_must_be_realized_not_merely_declared():
    value = card(); value["selected_rows"][0]["visible_fragments"] = 8
    reseal(value)
    reject(value)


def test_reselection_needs_new_exact_approval():
    value = card(); value["selected_rows"][0]["row_index"] = 100
    value["selection_sha256"] = selection_sha256(value["selected_rows"])
    reject(value)


@pytest.mark.parametrize("role", ["sensor", "teacher"])
@pytest.mark.parametrize("root", ["/absolute/fit", "../escape/fit", "fixtures/C10/fit", "fixtures/dev", []])
def test_source_fit_roots(role, root):
    value = card(); value["source_roots"][role] = root
    reject(value)


@pytest.mark.parametrize("field", ["source_roots", "source_seals", "checkpoints"])
def test_no_extra_source_or_checkpoint(field):
    value = card(); value[field]["offset"] = "fixtures/offset"
    reject(value)


@pytest.mark.parametrize("path", ["../escape", "/absolute", "./x", "a//b", "a\\b"])
def test_source_paths_repository_relative(path):
    value = card(); value["sealed_sources"][path] = "a" * 64
    reject(value)


@pytest.mark.parametrize("digest", [None, [], "a" * 63, "X" * 64])
def test_hash_format(digest):
    value = card(); value["sealed_sources"]["fixtures/extra"] = digest
    reject(value)


@pytest.mark.parametrize("key", ["metadata_selection_path", "sensor", "teacher"])
def test_every_selection_and_seal_path_is_hashed(key):
    value = card()
    path = value[key] if key == "metadata_selection_path" else value["source_seals"][key]
    del value["sealed_sources"][path]
    reject(value)


@pytest.mark.parametrize("role", ["backbone", "head"])
@pytest.mark.parametrize("patch", [
    {"seed": False}, {"seed": 1}, {"frozen": 1}, {"selection": "best"},
    {"path": None}, {"path": "fixtures/unsealed.pt"}, {"sha256": "f" * 64},
])
def test_both_checkpoints_frozen_and_exact(role, patch):
    value = card(); value["checkpoints"][role].update(patch)
    reject(value)


def test_changed_head_checkpoint_needs_matching_approval():
    value = card(); value["checkpoints"]["head"]["sha256"] = "f" * 64
    value["sealed_sources"]["fixtures/head.pt"] = "f" * 64
    reject(value)


def test_stopped_offset_head_is_forbidden_even_if_hash_bound():
    value = card(); value["checkpoints"]["head"]["path"] = "fixtures/raw_slot_offset.pt"
    value["sealed_sources"]["fixtures/raw_slot_offset.pt"] = "c" * 64
    reject(value)


def test_backbone_and_head_cannot_alias_same_file():
    value = card(); value["checkpoints"]["head"] = copy.deepcopy(value["checkpoints"]["backbone"])
    value["approval"]["checkpoint_sha256"]["head"] = "b" * 64
    reject(value)


@pytest.mark.parametrize("key", INFERENCE_COUNTS)
def test_inference_counts_exact_integer(key):
    value = card(); value["inference"][key] = float(INFERENCE_COUNTS[key])
    reject(value)
    value["inference"][key] = INFERENCE_COUNTS[key] + 1
    reject(value)


def test_no_extra_forward_or_repeat_task():
    value = card(); value["inference"]["repeat_task"] = value["tasks"][1]
    reject(value)
    value = card(); value["inference"]["extra_repeat"] = 1
    reject(value)


@pytest.mark.parametrize("restriction", RESTRICTIONS)
def test_no_restriction_relaxed(restriction):
    value = card(); value["restrictions"][restriction] = False
    reject(value)


@pytest.mark.parametrize("patch", [
    {"authorized_operations": ["audit"]}, {"authorized_operations": ["training"]},
    {"authorized_operations": ["data_export", "training"]}, {"authorized_gates": [3.0]},
    {"authorized_gates": [4]}, {"status": "PROPOSED"}, {"scope": ""},
    {"confirmation_reference": ""}, {"selection_sha256": None}, {"checkpoint_sha256": {}},
])
def test_approval_exact_export_scope(patch):
    value = card(); value["approval"].update(patch)
    reject(value)
