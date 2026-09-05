"""Small synthetic same-snapshot attribution, with frozen loss/evaluator."""
from dataclasses import fields, replace
import json

import pytest
import torch

import mtare_topo.evaluation.gse_assignment_attribution as attribution
from mtare_topo.evaluation.gse_partial_structure import evaluate_partial_structure
from tests.v3.unit.test_gse_partial_structure_evaluation import fixture


def inputs(prediction=None, target=None, batch=1):
    if prediction is None: prediction, target = fixture()
    if batch != 1:
        prediction = type(prediction)(**{f.name: getattr(prediction, f.name).repeat(batch, *([1]*(getattr(prediction, f.name).ndim-1)))
                                         for f in fields(prediction)})
        target = type(target)(**{f.name: getattr(target, f.name).repeat(batch, *([1]*(getattr(target, f.name).ndim-1)))
                               for f in fields(target)})
    manifest = [dict(task="synthetic_a" if i % 2 == 0 else "synthetic_b", source_sequence=i)
                for i in range(len(prediction.centers_m))]
    bridge = []
    for row in range(len(prediction.centers_m)):
        counts = {name: int((target.member_valid[row] & (target.members[row] == value)).sum())
                  for name, value in (("positive", 1), ("negative", 0))}
        bridge.append(dict(target_transport={prefix + name: counts[name] if prefix != "unknown_correspondence_member_" else 0
            for name in counts for prefix in ("original_member_", "transferred_member_", "unknown_correspondence_member_")}))
    return prediction, target, manifest, bridge


def test_exact_snapshot_matches_geometry_and_json_serializes():
    p, t, manifest, bridge = inputs()
    result = attribution.diagnose_assignments(p, t, manifest, bridge, [[0]])
    expected = evaluate_partial_structure(p, t, membership_threshold=.5, manifest=manifest,
                                          direction_bridge_ledger=[r["target_transport"] for r in bridge])
    assert result["geometry_summary"] == expected.summary
    assert result["comparison"]["same_query"] == 2
    assert len(result["junction_table"]) == 1
    assert result["joint_diagnostic"]["members"]["known_scored_f1"] == 1
    assert not result["joint_assignment_uniqueness_verified"]
    assert not result["joint_diagnostic"]["scientific_score"]
    assert result["model_forwards"] == result["optimizer_steps"] == 0
    json.dumps(result, allow_nan=False)


def test_semantic_correct_far_candidate_reveals_joint_geometry_split():
    p, t = fixture(m=1)
    p.event_logits[0, 0] = torch.tensor([-8., 8., -8.])
    p.membership_logits[0, 0, :2] = torch.tensor([-8., 8.])
    p.centers_m[0, 1] = torch.tensor([1., 0., 0.])
    p.event_logits[0, 1] = torch.tensor([8., -8., -8.])
    p.membership_logits[0, 1, :2] = torch.tensor([8., -8.])
    result = attribution.diagnose_assignments(*inputs(p, t), [[0]])
    record = result["target_comparisons"][0]
    assert record["geometry_query"] == 0 and record["joint_query"] == 1
    assert not record["same_query"]
    assert record["teacher_center_m"] == record["geometry_center_m"] == [0., 0., 0.]
    assert record["joint_center_m"] == [1., 0., 0.]
    assert record["geometry_distance_m"] == 0 and record["joint_distance_m"] == 1
    assert record["geometry"]["event_argmax"] == 1 and record["joint"]["event_argmax"] == 0
    assert record["geometry"]["members"] == dict(tp=0, fn=1, fp=1, tn=0)
    assert record["joint"]["members"] == dict(tp=1, fn=0, fp=0, tn=1)
    assert result["geometry_summary"]["members"]["known_scored_f1"] == 0
    assert result["joint_diagnostic"]["members"]["known_scored_f1"] == 1
    assert not result["replaces_independent_scoring"]


def test_exact_original_batches_call_frozen_loss_under_no_grad(monkeypatch):
    p, t, manifest, bridge = inputs(batch=3)
    p.centers_m.requires_grad_()
    original = attribution.region_set_losses
    calls = []
    def capture(prediction, target):
        assert not torch.is_grad_enabled()
        calls.append(len(prediction.centers_m))
        return original(prediction, target)
    monkeypatch.setattr(attribution, "region_set_losses", capture)
    result = attribution.diagnose_assignments(p, t, manifest, bridge, [[2, 0], [1]])
    assert calls == [2, 1]
    assert result["last_epoch_batches"] == [[2, 0], [1]]
    assert result["batch_joint_losses"][0]["observation_indices"] == [2, 0]
    assert {r["observation_index"] for r in result["batch_joint_losses"][0]["joint_representative_matches"]} == {0, 2}
    assert p.centers_m.grad is None
    assert set(result["parents"]) == {"synthetic_a", "synthetic_b"}
    assert result["parents"]["synthetic_a"]["comparison"]["known_targets"] == 4
    assert len(result["junction_table"]) == 3


@pytest.mark.parametrize("batches", [[[0], [0]], [[0]], [[0, 3], [1]], [[True, 1]], [[]], [], [[0, 1.]], [[-1, 1]]])
def test_epoch_must_be_exact_population_partition(batches):
    with pytest.raises(ValueError):
        attribution.diagnose_assignments(*inputs(batch=2), batches)


def test_geometry_ambiguity_does_not_become_joint_uniqueness_claim():
    p, t = fixture(m=1)
    p.centers_m[0, 1] = p.centers_m[0, 0]
    p.event_logits[0, 1] = p.event_logits[0, 0]
    p.membership_logits[0, 1] = p.membership_logits[0, 0]
    result = attribution.diagnose_assignments(*inputs(p, t), [[0]])
    record = result["target_comparisons"][0]
    assert record["geometry_query"] is None and record["geometry_center_m"] is None
    assert record["geometry_center_status"] == "center_ambiguous"
    assert record["joint_query"] in (0, 1)
    assert record["same_query"] is None
    assert not record["joint_assignment_unique_verified"]
    assert result["geometry_summary"]["members"]["coverage"] == 0
    assert result["joint_diagnostic"]["members"]["coverage"] == 1


def test_original_bridge_unknown_and_probability_population_preserved():
    p, t, manifest, bridge = inputs()
    ledger = bridge[0]["target_transport"]
    ledger["original_member_positive"] += 3; ledger["unknown_correspondence_member_positive"] += 3
    ledger["original_member_negative"] += 4; ledger["unknown_correspondence_member_negative"] += 4
    result = attribution.diagnose_assignments(p, t, manifest, bridge, [[0]])
    m = result["joint_diagnostic"]["members"]
    assert m["original_positive"] == 5 and m["original_negative"] == 6
    assert m["coverage"] == m["worst_case_f1"] == 4/11
    assert m["best_case_f1"] == 1
    assert m["positive_probability"]["count"] == m["negative_probability"]["count"] == 2
    assert set(m["positive_probability"]) == {"count", "mean", "median", "p10", "p90"}
    assert all("_positive_values" not in r["joint"] for r in result["target_comparisons"])


def test_unsupported_direction_and_exact_event_tie_are_explicit():
    p, t = fixture()
    p.query_supported[0, 3] = False
    p.member_supported[:] = p.query_supported[:, :, None] & p.query_supported[:, None]
    t.member_valid[0, 0, 3] = True; t.members[0, 0, 3] = 1
    p.event_logits[:] = 0
    result = attribution.diagnose_assignments(*inputs(p, t), [[0]])
    assert result["joint_diagnostic"]["members"]["rejected"]["unsupported_direction"]["positive"] == 1
    assert all(r["joint"]["event_maximum_tied"] for r in result["target_comparisons"])
    assert result["joint_diagnostic"]["events"]["rejected_events"] == 2


def test_no_targets_is_not_zero_loss_scientific_pass():
    p, t = fixture(n=64, m=0)
    result = attribution.diagnose_assignments(*inputs(p, t), [[0]])
    assert result["target_comparisons"] == result["junction_table"] == []
    assert result["joint_diagnostic"]["members"]["known_scored_f1"] is None
    assert result["geometry_summary"]["geometry"]["unselected_queries"] == 64
    assert not result["joint_diagnostic"]["scientific_score"]


def test_same_tensors_are_unmodified_and_repeated_results_exact():
    args = inputs()
    before = {f.name: getattr(args[0], f.name).clone() for f in fields(args[0])}
    first = attribution.diagnose_assignments(*args, [[0]])
    second = attribution.diagnose_assignments(*args, [[0]])
    assert first == second
    assert all(torch.equal(v, getattr(args[0], k)) for k, v in before.items())


def test_semantic_target_changes_never_change_geometry_assignment():
    p, t, manifest, bridge = inputs()
    first = attribution.diagnose_assignments(p, t, manifest, bridge, [[0]])
    changed = replace(t, events=1-t.events, members=1-t.members)
    second = attribution.diagnose_assignments(p, changed, manifest, bridge, [[0]])
    assert [r["geometry_query"] for r in first["target_comparisons"]] == [r["geometry_query"] for r in second["target_comparisons"]]
