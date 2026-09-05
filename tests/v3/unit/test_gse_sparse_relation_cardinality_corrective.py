import torch

from mtare_topo.representation.gse_sparse_circular_relation_transport import (
    SparseCircularRelationTransportNet,
    parameter_count,
    token_count_loss,
)


def _scans() -> torch.Tensor:
    generator = torch.Generator().manual_seed(31)
    scans = torch.rand(2, 5, 2, 16, 720, generator=generator)
    scans[:, :, 1] = (scans[:, :, 1] > 0.2).float()
    return scans


def test_cardinality_head_is_exact_minimal_delta() -> None:
    model = SparseCircularRelationTransportNet()
    assert parameter_count() == 784513
    assert model.token_count_head.in_features == 128
    assert model.token_count_head.out_features == 7
    assert sum(parameter.numel() for parameter in model.token_count_head.parameters()) == 903


def test_cardinality_output_is_causal_and_rotation_invariant() -> None:
    torch.manual_seed(0)
    model = SparseCircularRelationTransportNet().eval()
    scans = _scans()
    changed = scans.clone()
    changed[:, 3:] = torch.flip(changed[:, 3:], dims=(-1,))
    with torch.no_grad():
        base = model(scans)["token_count_logits"]
        future = model(changed)["token_count_logits"]
        rotated = model(torch.roll(scans, 20, dims=-1))["token_count_logits"]
    assert torch.equal(base[:, :3], future[:, :3])
    assert torch.allclose(base, rotated, atol=3e-5, rtol=0)


def test_real_count_objective_reaches_count_head() -> None:
    model = SparseCircularRelationTransportNet()
    scans = _scans()
    presence = torch.zeros(2, 5, 180, dtype=torch.bool)
    presence[0, :, :2] = True
    presence[1, :, :3] = True
    output = model(scans)
    loss = token_count_loss(output["token_count_logits"], presence)
    loss.backward()
    for parameter in model.token_count_head.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert bool((parameter.grad != 0).any())
