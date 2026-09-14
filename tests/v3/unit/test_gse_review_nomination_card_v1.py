"""Mutations of the new policy card only; no dataset payload is read."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from mtare_topo.governance_review_nomination import (
    EXPECTED_SCOPE_SHA256, SCHEMA, scope_sha256, validate_review_nomination_card,
)


ROOT = Path(__file__).resolve().parents[3]
CARD = ROOT / "configs/v3/gate3/data_cards/gse_review_nomination_v1.json"


def card():
    return json.loads(CARD.read_text())


def rebind_claimed_digest(value):
    digest = scope_sha256(value["scope"])
    value["scope_sha256"] = digest
    value["approval"]["scope_sha256"] = digest


def test_exact_prepared_card_passes_without_reading_sources():
    value = card()
    before = deepcopy(value)
    report = validate_review_nomination_card(value)
    assert report.passed, report.errors
    assert value == before
    assert value["scope_sha256"] == EXPECTED_SCOPE_SHA256 == scope_sha256(value["scope"])
    assert value["schema_version"] == SCHEMA
    assert value["scope"]["source_scope"]["new_codebook_files_required"] == 0  # Provenance, not current permission.
    actual = value["scope"]["actual_read_scope"]
    assert actual["codebook_files_required"] == len(actual["codebook_paths"]) == 165
    assert actual["construction_whole_json_includes_unused_geometry_bytes"] is True
    assert actual["codebook_whole_json_includes_unused_source_sets_bytes"] is True
    assert value["scope"]["sampling"]["duration_s"] is None


def test_policy_and_draft_hashes_match_local_documents_not_runtime_trust():
    value = card()
    for key in ("source_draft_document", "nomination_policy"):
        item = value["scope"][key]
        assert hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest() == item["sha256"]


@pytest.mark.parametrize("field,value", [("operation", "training"), ("operation", "data_export"),
    ("operation", "annotation"), ("gate", True), ("gate", 4), ("gate", 3.0),
    ("schema_version", "v3_identity_inventory_card_v1"), ("card_id", "other"),
    ("scientific_gate_pass", True), ("real_nomination_completed", True), ("training_eligibility", True),
    ("purpose", ""), ("limitations", None)])
def test_authority_and_false_completion_mutations_rejected(field, value):
    item = card(); item[field] = value
    assert not validate_review_nomination_card(item).passed


@pytest.mark.parametrize("field,value", [("authorized_operations", ["audit", "training"]),
    ("authorized_gates", [True]), ("authorized_gates", [3, 4]), ("authorized_gates", "3"),
    ("scope_sha256", "a" * 64), ("status", "PENDING"), ("approved_by", "new-user-reply"),
    ("confirmation_reference", ""), ("extra_authority", True)])
def test_approval_must_be_closed_standing_scope(field, value):
    item = card(); item["approval"][field] = value
    assert not validate_review_nomination_card(item).passed


@pytest.mark.parametrize("mutation", ("task_count", "row_count_bool", "row_order", "row_digest", "parent_test",
    "codebook_outside", "codebook_missing", "codebook_omitted", "resource", "scan", "scope_path", "unknown_count", "policy"))
def test_rehashing_submitted_scope_cannot_bypass_frozen_expected_scope(mutation):
    item = card(); scope = item["scope"]; source = scope["source_scope"]
    if mutation == "task_count":
        source["task_row_summary"].pop()
    elif mutation == "row_count_bool":
        source["task_row_summary"][0]["row_count"] = True
    elif mutation == "row_order":
        source["task_row_summary"].reverse()
    elif mutation == "row_digest":
        source["task_row_summary"][0]["unique_rows_sha256"] = "a" * 64
    elif mutation == "parent_test":
        source["eligible_parents"][0] = "S01_flat_tree_small_C10"
    elif mutation == "codebook_outside":
        paths = scope["actual_read_scope"]["codebook_paths"]
        paths[next(iter(paths))] = "../../C10.json"
    elif mutation == "codebook_missing":
        paths = scope["actual_read_scope"]["codebook_paths"]; paths.pop(next(iter(paths)))
    elif mutation == "codebook_omitted":
        scope["actual_read_scope"]["codebook_files_required"] = 0
    elif mutation == "resource":
        scope["resources"]["wall_time_cap_s"] = 601
    elif mutation == "scan":
        scope["actual_read_scope"]["p1b_arrays"].append("range_m")
    elif mutation == "scope_path":
        source["source_roots"]["construction"] = "results"
    elif mutation == "unknown_count":
        source["unknown_population"]["human_accepted_labels"] = 48
    else:
        scope["nomination_policy"]["sha256"] = "b" * 64
    rebind_claimed_digest(item)
    assert not validate_review_nomination_card(item).passed


@pytest.mark.parametrize("bad", [None, [], True, float("nan"), float("inf"), ("tuple",)])
def test_non_card_and_non_json_values_rejected(bad):
    assert not validate_review_nomination_card(bad).passed
    item = card(); item["extra"] = bad
    assert not validate_review_nomination_card(item).passed


def test_missing_nested_structures_return_errors_not_crashes():
    for path in (("scope",), ("scope", "source_scope"), ("scope", "actual_read_scope"),
                 ("approval",), ("scope_sha256",)):
        item = card(); container = item
        for key in path[:-1]:
            container = container[key]
        del container[path[-1]]
        assert not validate_review_nomination_card(item).passed
