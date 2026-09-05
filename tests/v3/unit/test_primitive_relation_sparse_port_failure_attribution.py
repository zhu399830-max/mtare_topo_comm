import numpy as np
import pytest
import torch

from mtare_topo.evaluation.primitive_relation_sparse_port_failure_attribution import (
    attachment_score_slice,
    best_link_pair_mask,
    endpoint_pair_eligibility,
    exact_ranked_selection,
    teacher_cardinality_topk_mask,
)


def test_teacher_cardinality_topk_uses_count_not_identity():
    logits = torch.arange(32, dtype=torch.float32)[None]
    target = torch.zeros(1, 32, dtype=torch.bool); target[0, :3] = True
    selected = teacher_cardinality_topk_mask(logits, target)
    assert selected.sum() == 3
    assert selected[0, 29:].all()
    assert not torch.any(selected & target)


def test_teacher_cardinality_topk_supports_formal_batch_population():
    logits = torch.arange(32, dtype=torch.float32)[None].expand(128, -1).clone()
    logits += torch.arange(128, dtype=torch.float32)[:, None] * 1e-3
    target = torch.zeros(128, 32, dtype=torch.bool)
    target[:, :5] = True
    selected = teacher_cardinality_topk_mask(logits, target)
    assert selected.shape == (128, 32)
    assert torch.all(selected.sum(dim=1) == 5)
    assert torch.all(selected[:, 27:])


def test_pair_eligibility_excludes_same_primitive_and_lower_triangle():
    mask = torch.zeros(1, 32, dtype=torch.bool); mask[0, :2] = True
    eligible = endpoint_pair_eligibility(mask)
    assert int(eligible.sum()) == 4
    assert not eligible[0, 0, 1]
    assert eligible[0, 0, 2]
    assert not eligible[0, 2, 0]


def test_best_link_union_and_mutual_are_score_only():
    eligible = torch.zeros(1, 64, 64, dtype=torch.bool)
    eligible[0, 0, 2] = eligible[0, 0, 3] = True
    eligible[0, 1, 2] = eligible[0, 1, 3] = True
    eligible = eligible | eligible.transpose(1, 2)
    score = torch.zeros(1, 64, 64)
    score[0, 0, 2] = score[0, 2, 0] = 0.9
    score[0, 1, 2] = score[0, 2, 1] = 0.8
    score[0, 1, 3] = score[0, 3, 1] = 0.7
    union = best_link_pair_mask(score, eligible, mutual=False)
    mutual = best_link_pair_mask(score, eligible, mutual=True)
    assert union[0, 0, 2] and union[0, 1, 2] and union[0, 1, 3]
    assert mutual[0, 0, 2]
    assert int(mutual.sum()) == 1


def test_exact_ranked_selection_preserves_ties_and_safe_nonempty():
    score = np.asarray([0.9, 0.9, 0.8, 0.7], dtype=np.float32)
    target = np.asarray([True, False, True, True])
    best = exact_ranked_selection(score, target, total_positive=4)
    assert best["threshold"] == pytest.approx(0.7)
    assert best["true_positive"] == 3
    safe = exact_ranked_selection(score, target, total_positive=4, minimum_precision=0.98)
    assert safe["available"] is False
    score2 = np.asarray([0.99, 0.7, 0.6], dtype=np.float32)
    target2 = np.asarray([True, False, True])
    safe2 = exact_ranked_selection(score2, target2, total_positive=3, minimum_precision=0.98)
    assert safe2["available"] and safe2["true_positive"] == 1


def test_attachment_slice_reports_overlap_hard_negative():
    logits = torch.full((1, 32, 2, 32, 2), -10.0)
    uncertainty = torch.zeros_like(logits)
    target = torch.zeros_like(logits, dtype=torch.bool)
    overlap = torch.zeros(1, 32, 32, dtype=torch.bool)
    mask = torch.zeros(1, 32, dtype=torch.bool); mask[0, :2] = True
    logits[0, 0, 0, 1, 0] = logits[0, 1, 0, 0, 0] = 4.0
    overlap[0, 0, 1] = overlap[0, 1, 0] = True
    result = attachment_score_slice(logits, uncertainty, target, overlap, mask)
    assert result.eligible_pairs == 4
    assert int(result.overlap_hard_negative.sum()) == 4
    assert float(result.raw_score.max()) > 0.98


def test_invalid_decoder_fails_closed():
    logits = torch.zeros(1, 32, 2, 32, 2)
    with pytest.raises(ValueError):
        attachment_score_slice(
            logits, logits, logits.bool(), torch.zeros(1, 32, 32, dtype=torch.bool),
            torch.zeros(1, 32, dtype=torch.bool), decoder="rules",
        )


def test_outer_runner_declares_three_required_seed_files():
    import run_primitive_relation_sparse_port_failure_attribution_v1 as runner
    assert runner.SEEDS == (0, 1, 2)
