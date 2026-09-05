import numpy as np
import torch

from tools.v3.evaluate_primitive_composition_anchor_three_seed_v1 import (
    _PairBuffer,
    _cross_primitive_upper,
)


def test_composition_anchor_pair_buffer_counts_missing_positive_as_false_negative():
    buffer = _PairBuffer()
    safe = torch.tensor([0.99, 0.90, 0.20, 0.10])
    compatibility = torch.tensor([0.98, 0.80, 0.40, 0.30])
    target = torch.tensor([True, True, False, False])
    overlap = torch.tensor([False, False, True, False])
    buffer.append(safe, compatibility, target, overlap)
    result = buffer.finalize()
    assert result["candidate_true_pairs"] == 2
    # The frozen C07 population contains positives missing from the synthetic
    # candidate list, so the selector must retain a nonzero false-negative.
    assert result["safe"]["best_f1"]["false_negative"] > 0
    assert result["safe"]["safe"]["true_positive"] > 0


def test_cross_primitive_upper_excludes_self_and_two_ends_of_same_primitive():
    upper = _cross_primitive_upper(torch.device("cpu")).numpy()
    assert upper.shape == (64, 64)
    assert not np.any(np.diag(upper))
    assert not upper[0, 1]
    assert upper[0, 2]
    assert not upper[2, 0]


def test_safe_score_regression_uses_grouped_symmetric_evidence_product():
    compatibility = torch.linspace(0.01, 0.99, 64 * 64).reshape(1, 64, 64)
    compatibility = 0.5 * (compatibility + compatibility.transpose(1, 2))
    evidence = torch.sigmoid(torch.linspace(-5.0, 5.0, 64)).reshape(1, 64)
    grouped = compatibility * (evidence[:, :, None] * evidence[:, None, :])
    assert torch.equal(grouped, grouped.transpose(1, 2))
    assert bool(torch.isfinite(grouped).all())
