"""Synthetic source-evidence tests; no dataset reads or training runs.

Masks come from the retained reference rules. Binding verification flags below
stand for a synthetic verified reader, not validation of any real evidence.
"""
from copy import deepcopy

import numpy as np
import pytest
import torch

from mtare_topo.data.gse_structure_review_v1 import canonical_sha
from mtare_topo.representation.candidate_center_verifier_v1 import balanced_validity_loss
from mtare_topo.representation.center_localization_quality_v1 import localization_quality
from mtare_topo.teacher.gse_reference_exclusion_v1 import reference_exclusion
from mtare_topo.teacher.gse_reference_query_coverage_v2 import reference_query_coverage


def fixture(positions=None, states=None, references=None, confirmed=(0,)):
    q = np.asarray(positions if positions is not None else [
        [.2, 0., 0.], [.6, 0., 0.], [2., 0., 0.],
        [3., 0., 0.], [7., 0., 0.], [8., 0., 0.],
    ], dtype=np.float64).reshape(-1, 3)
    states = np.asarray(states if states is not None else [0, 1, 1, 0, 2, 0], dtype=np.uint8)
    refs = np.asarray(references if references is not None else [[0., 0., 0.]], dtype=np.float64).reshape(-1, 3)
    targets = refs[list(confirmed)]
    source = dict(parent="synthetic-only", frame_rows=[0, 1, 2, 3, 4])
    binding = dict(source=source, geometry_sha256="synthetic-geometry")
    grid_sha = canonical_sha(dict(positions=q.tolist(), states=states.tolist()))
    target_sha = canonical_sha(dict(positions=targets.tolist(), source=source))
    bg = reference_exclusion(query_xyz_m=q, observed_state=states,
        all_anchor_xyz_m=refs, inventory_complete=True, matching_radius_m=4.)
    bg.update(binding=deepcopy(binding), observation_identity_binding_verified=True,
        inventory_completeness_supplied_not_verified=False,
        observed_states_supplied_not_verified=False,
        query_observed_states=states.tolist(), grid_content_sha256=grid_sha)

    def coverage(radius):
        result = reference_query_coverage(query_xyz_m=q, observed_state=states,
            all_reference_xyz_m=refs, confirmed_reference_indices=list(confirmed),
            matching_radius_m=radius, score_center_m=[0., 0., 0.], score_radius_m=10.)
        result.update(source_binding=deepcopy(binding), grid_content_sha256=grid_sha,
            target_record_sha256=target_sha, query_xyz_m=q.tolist(),
            reference_inventory_complete_supplied_not_verified=False)
        return result

    evidence = dict(background_radius_m=4., original_background=bg, coverage=coverage(4.))
    return q, targets, evidence, coverage(1.), source


def test_original_background_unchanged_and_distinct_from_position_quality():
    args = fixture()
    before = deepcopy(args[2])
    result = localization_quality(*args)
    assert args[2] == before
    assert result["original_background_radius_m"] == 4.
    assert result["original_background_negative"] == [False, False, False, False, True, False]
    assert result["positive"] == [True, True, False, False, False, False]
    assert result["negative"] == [False, False, True, True, True, False]
    assert result["unknown"] == [False, False, False, False, False, True]
    assert result["candidates"][2]["localization_basis"] == "observed_reference_exclusion1m"
    assert not result["candidates"][2]["background_claim"]
    assert result["candidates"][4]["background_claim"]


def test_unobserved_offcenter_requires_confirmed_context():
    args = fixture([[3., 0., 0.], [4.01, 0., 0.], [8., 0., 0.]], [0, 0, 0])
    result = localization_quality(*args)
    assert result["negative"] == [True, False, False]
    assert result["unknown"] == [False, True, True]
    record = result["candidates"][0]
    assert record["support_type"] == "confirmed_structure_context"
    assert record["localization_basis"] == "confirmed_structure_offcenter"
    assert record["observed_state"] == 0 and not record["background_claim"]
    args = fixture([[3., 0., 0.]], [0], confirmed=())
    assert localization_quality(*args)["unknown"] == [True]


def test_multiple_legal_centers_are_positive_not_position_negatives():
    result = localization_quality(*fixture([[.1, 0., 0.], [.1, 0., 0.], [.9, 0., 0.]], [0, 0, 0]))
    assert result["positive"] == [True, True, True]
    assert result["negative"] == [False, False, False]
    assert all(c["set_redundancy"] == "not_a_position_label" for c in result["candidates"])


def test_domain_boundary_does_not_create_new_outside_labels():
    args = fixture([[10.1, 0., 0.], [12., 0., 0.], [16., 0., 0.]],
        [0, 0, 1], [[10., 0., 0.]])
    result = localization_quality(*args)
    # A near confirmed center outside the domain is not a new positive;
    # outside confirmed context is not a new position-quality negative.
    assert result["positive"] == [False, False, False]
    assert result["unknown"] == [True, True, False]
    # Existing independently supplied original background remains untouched.
    assert result["negative"] == [False, False, True]
    assert result["original_background_negative"] == [False, False, True]
    assert result["candidates"][2]["background_claim"]


def test_position_labels_permute_with_candidates():
    args = fixture()
    baseline = localization_quality(*args)
    order = np.array([5, 2, 0, 4, 1, 3])
    states = np.asarray(args[2]["original_background"]["query_observed_states"])
    changed = localization_quality(*fixture(args[0][order], states[order]))
    for key in ("positive", "negative", "unknown", "original_background_negative"):
        assert np.asarray(baseline[key])[order].tolist() == changed[key]
    for slot, original in enumerate(order):
        assert baseline["candidates"][original]["localization_basis"] == changed["candidates"][slot]["localization_basis"]


def test_scores_evaluation_fp_and_query_scoreable_do_not_construct_labels():
    args = fixture()
    baseline = localization_quality(*args)
    changed = deepcopy(args)
    for container in (changed[2], changed[2]["coverage"], changed[3]):
        container["logits"] = [100., -100., 100., -100., 100., -100.]
        container["known_region_false_positive_mask"] = [True] * 6
        container["query_scoreable_mask"] = [False] * 6
        container["used_duplicate_mask"] = [True] * 6
    assert localization_quality(*changed) == baseline


@pytest.mark.parametrize("references,positions,states", [
    ([[0., 0., 0.], [2., 0., 0.]], [[2., 0., 0.], [3., 0., 0.]], [1, 0]),
    ([[0., 0., 0.], [0., 0., 3.]], [[0., 0., 2.5], [0., 0., 3.5]], [1, 0]),
    ([[0., 0., 0.], [4.5, 0., 0.]], [[2., 0., 0.]], [0]),
])
def test_hidden_adjacent_and_stacked_competitors_do_not_create_negatives(references, positions, states):
    result = localization_quality(*fixture(positions, states, references))
    assert result["unknown"] == [True] * len(positions)
    assert not any(result["negative"])


def test_hidden_competitor_vetoes_even_near_confirmed_center():
    args = fixture([[.2, 0., 0.]], [1], [[0., 0., 0.], [.5, 0., 0.]])
    result = localization_quality(*args)
    assert result["positive"] == [False] and result["unknown"] == [True]


@pytest.mark.parametrize("tamper", ["source", "grid", "target", "verification", "coordinates"])
def test_source_evidence_mismatch_stops(tamper):
    args = list(fixture())
    if tamper == "source":
        args[4] = dict(parent="wrong-source")
    elif tamper == "grid":
        args[3]["grid_content_sha256"] = "wrong-grid"
    elif tamper == "target":
        args[3]["target_record_sha256"] = "wrong-target"
    elif tamper == "verification":
        args[2]["original_background"]["observed_states_supplied_not_verified"] = True
    else:
        args[3]["query_xyz_m"][0][0] += .01
    with pytest.raises(ValueError):
        localization_quality(*args)


def test_unobserved_unknown_is_not_background_and_has_zero_loss_gradient():
    result = localization_quality(*fixture())
    logits = torch.tensor([.1, .2, .3, .4, .5, 100.], requires_grad=True)
    positive = torch.tensor(result["positive"])
    negative = torch.tensor(result["negative"])
    loss = balanced_validity_loss(logits, positive, negative)
    loss.backward()
    assert torch.all(logits.grad[positive] < 0)
    assert torch.all(logits.grad[negative] > 0)
    assert logits.grad[5] == 0
    unknown_logits = torch.tensor([100.], requires_grad=True)
    mask = torch.zeros(1, dtype=torch.bool)
    loss = balanced_validity_loss(unknown_logits, mask, mask)
    loss.backward()
    assert loss == 0 and unknown_logits.grad.item() == 0
