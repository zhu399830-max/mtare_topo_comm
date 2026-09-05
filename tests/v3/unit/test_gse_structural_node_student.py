import torch

from mtare_topo.representation.gse_structural_node_evidence import NODE_EVIDENCE_DIM
from mtare_topo.representation.gse_structural_node_student import (
    NODE_STUDENT_ENDPOINT_FEATURE_DIM,
    StructuralNodeAggregationHead,
    structural_node_student_losses,
)


def test_node_head_is_endpoint_permutation_invariant():
    torch.manual_seed(7)
    model = StructuralNodeAggregationHead().eval()
    features = torch.randn(3, 64, NODE_STUDENT_ENDPOINT_FEATURE_DIM)
    confidence = torch.sigmoid(torch.randn(3, 64))
    permutation = torch.randperm(64)
    first = model(features, confidence)
    second = model(features[:, permutation], confidence[:, permutation])
    torch.testing.assert_close(first.descriptor, second.descriptor, atol=2e-6, rtol=1e-6)
    torch.testing.assert_close(first.degree_logits, second.degree_logits, atol=2e-6, rtol=1e-6)


def test_node_losses_are_finite_and_all_parameters_receive_gradient():
    torch.manual_seed(8)
    model = StructuralNodeAggregationHead()
    features = torch.randn(6, 64, NODE_STUDENT_ENDPOINT_FEATURE_DIM)
    confidence = torch.sigmoid(torch.randn(6, 64))
    target = torch.randn(6, NODE_EVIDENCE_DIM)
    degree = torch.tensor((1, 2, 3, 2, 3, 2), dtype=torch.long)
    identity = torch.tensor((0, 0, 1, 1, 2, 3))
    same = identity[:, None] == identity[None, :]
    losses = structural_node_student_losses(model(features, confidence), target, degree, same)
    losses["total"].backward()
    assert all(torch.isfinite(value) for value in losses.values())
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_node_head_rejects_bad_confidence():
    model = StructuralNodeAggregationHead()
    features = torch.zeros(1, 64, NODE_STUDENT_ENDPOINT_FEATURE_DIM)
    confidence = torch.ones(1, 64); confidence[0, 0] = 2.0
    try:
        model(features, confidence)
    except ValueError as error:
        assert "contract" in str(error)
    else:
        raise AssertionError("invalid confidence was accepted")
