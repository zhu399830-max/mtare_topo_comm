from __future__ import annotations

import torch

from mtare_topo.representation.gse_structural_node_dual_readout import (
    ASSOCIATION_DIM,
    GEOMETRY_SCALE,
    StructuralNodeDualReadout,
    association_distance,
    structural_node_dual_losses,
)
from mtare_topo.representation.gse_structural_node_evidence import NODE_EVIDENCE_DIM
from mtare_topo.representation.gse_structural_node_student import (
    ENDPOINTS,
    NODE_STUDENT_ENDPOINT_FEATURE_DIM,
)


def _batch() -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(17)
    feature = torch.randn(6, ENDPOINTS, NODE_STUDENT_ENDPOINT_FEATURE_DIM, generator=generator)
    confidence = torch.sigmoid(torch.randn(6, ENDPOINTS, generator=generator))
    target = torch.randn(6, NODE_EVIDENCE_DIM, generator=generator)
    degree = torch.tensor([1, 2, 2, 3, 3, 4], dtype=torch.long)
    identity = torch.tensor([0, 1, 1, 2, 2, 3])
    same = identity[:, None] == identity[None, :]
    candidate = torch.ones(6, 6, dtype=torch.bool)
    return feature, confidence, target, degree, same, candidate


def test_dual_readout_shapes_norms_and_scales() -> None:
    feature, confidence, *_ = _batch()
    prediction = StructuralNodeDualReadout()(feature, confidence)
    assert prediction.geometry.shape == (6, NODE_EVIDENCE_DIM)
    assert prediction.association.shape == (6, ASSOCIATION_DIM)
    assert torch.allclose(torch.linalg.vector_norm(prediction.association, dim=1), torch.ones(6), atol=2e-5)
    assert GEOMETRY_SCALE.shape == (NODE_EVIDENCE_DIM,)
    assert torch.all(GEOMETRY_SCALE > 0)


def test_dual_readout_is_endpoint_permutation_invariant() -> None:
    feature, confidence, *_ = _batch()
    model = StructuralNodeDualReadout().eval()
    permutation = torch.randperm(ENDPOINTS, generator=torch.Generator().manual_seed(9))
    with torch.no_grad():
        left = model(feature, confidence)
        right = model(feature[:, permutation], confidence[:, permutation])
    assert torch.allclose(left.geometry, right.geometry, atol=2e-6, rtol=2e-6)
    assert torch.allclose(left.association, right.association, atol=2e-6, rtol=2e-6)
    assert torch.allclose(left.degree_logits, right.degree_logits, atol=2e-6, rtol=2e-6)


def test_dual_losses_are_finite_and_all_parameters_receive_gradients() -> None:
    feature, confidence, target, degree, same, candidate = _batch()
    model = StructuralNodeDualReadout()
    losses = structural_node_dual_losses(model(feature, confidence), target, degree, same, candidate)
    assert all(torch.isfinite(value) for value in losses.values())
    losses["total"].backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_association_distance_is_symmetric_and_separates_identity() -> None:
    embedding = torch.zeros(3, ASSOCIATION_DIM)
    embedding[0, 0] = embedding[1, 0] = 1
    embedding[2, 1] = 1
    distance = association_distance(embedding)
    assert torch.equal(distance, distance.T)
    assert distance[0, 1] == 0
    assert distance[0, 2] == 1


def test_cross_graph_pairs_do_not_enter_relation_loss() -> None:
    feature, confidence, target, degree, same, candidate = _batch()
    candidate[4:, :4] = False
    candidate[:4, 4:] = False
    same[4:, :4] = False
    same[:4, 4:] = False
    prediction = StructuralNodeDualReadout()(feature, confidence)
    losses = structural_node_dual_losses(prediction, target, degree, same, candidate)
    assert torch.isfinite(losses["relation"])
