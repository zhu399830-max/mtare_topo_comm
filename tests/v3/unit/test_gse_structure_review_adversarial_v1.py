"""Synthetic adversarial review contracts; no data, model, or human labels."""
from copy import deepcopy
from dataclasses import replace

import pytest

from mtare_topo.data.gse_structure_review_v1 import (
    BlindReviewSession, canonical_sha, validate_annotation, validate_blind_bundle,
)
from tests.v3.unit.test_gse_structure_review_v1 import bundle, annotation, reference
from tests.v3.unit.test_gse_local_structure_v2 import observation


def test_reversed_source_windows_cannot_be_relabelled_as_forward_decisions():
    b = bundle()
    windows = deepcopy(b["decisions"])[::-1]
    for i, row in enumerate(windows): row["decision_index"] = i
    b["decisions"] = windows
    with pytest.raises(ValueError): validate_blind_bundle(b)


def test_one_source_window_cannot_fabricate_twenty_one_successive_decisions():
    b = bundle()
    first = deepcopy(b["decisions"][0])
    b["decisions"] = [dict(deepcopy(first), decision_index=i) for i in range(21)]
    with pytest.raises(ValueError): validate_blind_bundle(b)


def test_same_frame_identity_cannot_change_order_between_decisions():
    b = bundle()
    # Each row remains internally unique/causal, but f-1 now claims order 2.
    b["decisions"][2]["source_frame_keys"][0] = "f-1"
    with pytest.raises(ValueError): validate_blind_bundle(b)


def test_same_source_order_cannot_change_frame_identity_between_decisions():
    b = bundle()
    b["decisions"][1]["source_frame_keys"][0] = "replacement-frame-for-order-1"
    with pytest.raises(ValueError): validate_blind_bundle(b)


def test_decision_and_source_indices_need_not_be_equal():
    b = bundle()
    for row in b["decisions"]:
        row["decision_index"] += 100
        row["source_order_indices"] = [x + 300 for x in row["source_order_indices"]]
    assert validate_blind_bundle(b) == b


@pytest.mark.parametrize("mutate", [
    lambda b: b["decisions"][0].update(decision_index=False),
    lambda b: b["decisions"][0]["source_order_indices"].__setitem__(0, False),
    lambda b: b["decisions"][0].update(points_xyz_m=[[True, 0, 0]]),
    lambda b: b["decisions"][0].update(points_xyz_m=[[float("inf"), 0, 0]]),
])
def test_untrusted_boolean_and_nonfinite_source_fields_rejected(mutate):
    b = bundle(); mutate(b)
    with pytest.raises(ValueError): validate_blind_bundle(b)


def test_reference_cannot_be_seen_before_last_blind_or_backfill_after_reveal():
    b = bundle(); session = BlindReviewSession(b, "synthetic-reviewer-not-real-human")
    for i in range(20): session.commit_blind(i, annotation())
    before = session.export()
    with pytest.raises(ValueError): session.reveal_reference(reference(b))
    assert session.export() == before
    session.commit_blind(20, annotation())
    frozen = session.export()["blind_records"]
    ref = reference(b)
    ref["decisions"][0]["reference_annotation"]["notes"] = "hidden-reference-content"
    returned = session.reveal_reference(ref)
    returned["decisions"][0]["reference_annotation"]["notes"] = "overwrite"
    ref["decisions"][0]["reference_annotation"]["notes"] = "overwrite-again"
    with pytest.raises(ValueError): session.commit_blind(0, annotation())
    assert session.export()["blind_records"] == frozen
    assert session.export()["reference_sha256"] != canonical_sha(ref)


def test_unknown_empty_annotations_do_not_become_automatic_background_or_eligibility():
    b = bundle(); session = BlindReviewSession(b, "synthetic-reviewer")
    a = annotation()
    a["unknown_regions"] = [{"min_xyz_m": [-1, -1, -1], "max_xyz_m": [1, 1, 1], "reason": "occluded"}]
    for i in range(21): session.commit_blind(i, a)
    session.reveal_reference(reference(b)); session.record_reference_notes("No unknown backfill")
    result = session.export()
    assert result["review_complete"] is True and result["automatic_training_eligibility"] is False
    assert all(not row["annotation"]["complete_regions"] and not row["annotation"]["structures"]
               and row["annotation"]["unknown_regions"] for row in result["blind_records"])


@pytest.mark.parametrize("field,value", [("local_id", True), ("center_xyz_m", [0, False, 0]),
                                       ("center_xyz_m", [float("nan"), 0, 0])])
def test_untrusted_annotation_types(field, value):
    a = annotation()
    item = {"local_id": 0, "center_xyz_m": [0, 0, 0], "event": "unknown", "openings": [], "evidence_note": "uncertain"}
    item[field] = value; a["structures"] = [item]
    with pytest.raises(ValueError): validate_annotation(a)


@pytest.mark.parametrize("change", [dict(current_source_index=True), dict(decision_index=1.),
    dict(uncertainty_calibrated=1), dict(input_binding_sha256="A" * 64)])
def test_v2_dto_strict_scalar_bindings(change):
    with pytest.raises(ValueError): replace(observation(), **change)
