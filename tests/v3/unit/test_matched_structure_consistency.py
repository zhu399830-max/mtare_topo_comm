import pytest
import torch

from mtare_topo.representation.gse_matched_structure_consistency import matched_structure_consistency


def test_known_correspondence_mse_and_gradients_reach_both_views():
    x = torch.tensor([[0.0, 2.0], [2.0, 0.0]], requires_grad=True)
    y = torch.tensor([[1.0, 2.0], [2.0, -1.0]], requires_grad=True)
    terms = matched_structure_consistency(x, y, torch.tensor([True, True]))
    assert terms.effective_pair_count == 2 and terms.regularizer_valid
    assert terms.invariance_mse.item() == 0.5
    terms.invariance_mse.backward()
    assert torch.equal(x.grad, torch.tensor([[-0.5, 0.0], [0.0, 0.5]]))
    assert torch.equal(y.grad, -x.grad)


def test_unknown_nan_rows_do_not_pollute_any_term_or_gradient():
    x = torch.tensor([[0.0, 2.0], [float("nan"), float("nan")], [2.0, 0.0]], requires_grad=True)
    y = torch.tensor([[0.5, 2.0], [float("nan"), float("nan")], [2.0, 0.0]], requires_grad=True)
    known = torch.tensor([True, False, True])
    terms = matched_structure_consistency(x, y, known)
    loss = terms.invariance_mse + terms.standard_deviation_hinge + terms.covariance_off_diagonal
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(y.grad).all()
    assert torch.equal(x.grad[1], torch.zeros(2)) and torch.equal(y.grad[1], torch.zeros(2))
    assert terms.effective_pair_count == 2


def test_all_unknown_is_zero_with_invalid_regularizer_not_a_success():
    x = torch.full((3, 4), float("nan"), requires_grad=True)
    y = torch.full((3, 4), float("nan"), requires_grad=True)
    terms = matched_structure_consistency(x, y, torch.zeros(3, dtype=torch.bool))
    assert terms.effective_pair_count == 0 and not terms.regularizer_valid
    loss = terms.invariance_mse + terms.standard_deviation_hinge + terms.covariance_off_diagonal
    assert loss.item() == 0.0
    loss.backward()
    assert torch.equal(x.grad, torch.zeros_like(x)) and torch.equal(y.grad, torch.zeros_like(y))


def test_one_pair_can_match_but_cannot_estimate_variance_or_covariance():
    x = torch.tensor([[1.0, 2.0]], requires_grad=True)
    y = torch.tensor([[2.0, 2.0]], requires_grad=True)
    terms = matched_structure_consistency(x, y, torch.tensor([True]))
    assert terms.invariance_mse.item() == 0.5
    assert terms.effective_pair_count == 1 and not terms.regularizer_valid
    assert terms.standard_deviation_hinge.item() == terms.covariance_off_diagonal.item() == 0.0
    (terms.standard_deviation_hinge + terms.covariance_off_diagonal).backward()
    assert torch.equal(x.grad, torch.zeros_like(x)) and torch.equal(y.grad, torch.zeros_like(y))


def test_constant_collapse_has_nonzero_variance_penalty_and_finite_gradients():
    x = torch.full((3, 4), 2.0, requires_grad=True)
    y = torch.full((3, 4), 2.0, requires_grad=True)
    terms = matched_structure_consistency(x, y, torch.ones(3, dtype=torch.bool))
    assert terms.invariance_mse.item() == terms.covariance_off_diagonal.item() == 0.0
    assert terms.standard_deviation_hinge.item() == 1.0
    terms.standard_deviation_hinge.backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(y.grad).all()


def test_sample_statistics_and_dimension_normalization_have_analytic_values():
    x = torch.tensor([[0.0, 0.0], [2.0, 2.0]])
    terms = matched_structure_consistency(x, x, torch.tensor([True, True]))
    assert terms.standard_deviation_hinge.item() == 0.0
    # Unbiased covariance [[2, 2], [2, 2]]: off-diagonal squares / D = 4.
    assert terms.covariance_off_diagonal.item() == 4.0


def test_joint_row_permutation_and_view_swap_preserve_all_terms():
    x = torch.tensor([[0.0, 1.0], [1.0, 3.0], [2.0, 0.5], [float("nan"), float("nan")]])
    y = torch.tensor([[0.2, 0.8], [1.3, 3.0], [2.0, 0.4], [float("nan"), float("nan")]])
    known = torch.tensor([True, True, True, False])
    permutation = torch.tensor([3, 1, 0, 2])
    reference = matched_structure_consistency(x, y, known)
    permuted = matched_structure_consistency(y[permutation], x[permutation], known[permutation])
    for field in ("invariance_mse", "standard_deviation_hinge", "covariance_off_diagonal"):
        torch.testing.assert_close(getattr(reference, field), getattr(permuted, field))
    assert reference.effective_pair_count == permuted.effective_pair_count


def test_known_nonfinite_values_are_rejected():
    x = torch.tensor([[float("nan"), 1.0]])
    with pytest.raises(ValueError, match="qualified token values"):
        matched_structure_consistency(x, torch.zeros_like(x), torch.tensor([True]))


@pytest.mark.parametrize("known", [torch.tensor([1, 0]), torch.tensor([[True, False]]), [True, False]])
def test_known_mask_is_required_and_not_inferred_from_similarity(known):
    x = torch.zeros((2, 3))
    with pytest.raises(ValueError, match="qualification mask"):
        matched_structure_consistency(x, x, known)


def test_one_dimension_has_no_off_diagonal_and_empty_population_is_valid_masked_input():
    terms = matched_structure_consistency(torch.tensor([[0.0], [1.0]]), torch.tensor([[0.0], [1.0]]),
                                          torch.ones(2, dtype=torch.bool))
    assert terms.covariance_off_diagonal.item() == 0.0 and terms.regularizer_valid
    empty = matched_structure_consistency(torch.empty(0, 2), torch.empty(0, 2), torch.empty(0, dtype=torch.bool))
    assert empty.effective_pair_count == 0 and not empty.regularizer_valid
