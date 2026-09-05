import torch

from mtare_topo.representation.gse_sparse_circular_relation_transport import (
    MAX_TOKENS,
    SparseCircularRelationTransportNet,
    circular_nms_indices,
    parameter_count,
    relation_targets_from_token_identities,
    reverse_transport_logits,
    soft_circular_proposal_loss,
    sparse_circular_relation_transport_input_contract,
    sparse_relation_transport_loss,
    token_count_loss,
)


def _scans(batch: int = 2) -> torch.Tensor:
    generator = torch.Generator().manual_seed(29)
    value = torch.rand(batch, 5, 2, 16, 720, generator=generator)
    value[:, :, 1] = (value[:, :, 1] > 0.2).float()
    return value


def test_forward_is_lidar_only_and_typed() -> None:
    model = SparseCircularRelationTransportNet().eval()
    with torch.no_grad():
        output = model(_scans(1))
    assert sparse_circular_relation_transport_input_contract()["forward_parameters"] == ("self", "scans")
    assert output["proposal_logits"].shape == (1, 5, 180)
    assert output["token_bearing_deg"].shape == (1, 5, MAX_TOKENS)
    assert output["transport_row_logits"].shape == (1, 4, MAX_TOKENS, MAX_TOKENS + 1)
    assert output["transport_reveal_logits"].shape == (1, 4, MAX_TOKENS)
    assert torch.isfinite(torch.cat([value.reshape(-1) for value in output.values() if value.is_floating_point()])).all()
    assert output["token_count_logits"].shape == (1, 5, 7)
    assert output["token_geometry_uncertainty"].shape[-1] == 1 + output["token_vertical_profile_m"].shape[-1]
    assert parameter_count() == 784513


def test_identity_transport_preserves_simultaneous_reveal_and_withdraw() -> None:
    identity = torch.full((1, 5, MAX_TOKENS), -1)
    identity[0, 0, :2] = torch.tensor([10, 20])
    identity[0, 1, :2] = torch.tensor([10, 30])
    identity[0, 2:] = identity[0, 1]
    row, reveal = relation_targets_from_token_identities(identity)
    assert row[0, 0, 0] == 0
    assert row[0, 0, 1] == MAX_TOKENS
    assert reveal[0, 0, 0] == 0
    assert reveal[0, 0, 1] == 1


def test_circular_nms_is_distinct_wrap_aware_and_rotation_equivariant() -> None:
    logits = torch.full((1, 5, 180), -10.0)
    logits[..., 179] = 10.0
    logits[..., 0] = 9.0
    logits[..., 1] = 8.0
    logits[..., 40] = 7.0
    logits[..., 80] = 6.0
    logits[..., 120] = 5.0
    logits[..., 150] = 4.0
    logits[..., 20] = 3.0
    selected = circular_nms_indices(logits)
    rotated = circular_nms_indices(torch.roll(logits, 5, dims=-1))
    assert torch.equal(rotated, torch.remainder(selected + 5, 180))
    distance = torch.abs(selected[..., :, None] - selected[..., None, :])
    distance = torch.minimum(distance, 180 - distance)
    distinct = distance + torch.eye(6, dtype=distance.dtype) * 180
    assert int(distinct.min()) > 4
    assert 179 in selected
    assert 0 not in selected and 1 not in selected


def test_reverse_transport_is_exact_involution() -> None:
    generator = torch.Generator().manual_seed(3)
    row = torch.randn(2, 4, MAX_TOKENS, MAX_TOKENS + 1, generator=generator)
    reveal = torch.randn(2, 4, MAX_TOKENS, generator=generator)
    reverse_row, reverse_reveal = reverse_transport_logits(row, reveal)
    restored_row, restored_reveal = reverse_transport_logits(reverse_row, reverse_reveal)
    assert torch.equal(restored_row, row)
    assert torch.equal(restored_reveal, reveal)


def test_rotation_moves_tokens_and_preserves_transport() -> None:
    model = SparseCircularRelationTransportNet().eval()
    scans = _scans(1)
    with torch.no_grad():
        base = model(scans)
        shifted = model(torch.roll(scans, 20, dims=-1))
    expected = torch.remainder(base["token_bin_index"] + 5, 180)
    assert torch.equal(shifted["token_bin_index"], expected)
    assert torch.allclose(shifted["transport_row_logits"], base["transport_row_logits"], atol=3e-5, rtol=0)
    assert torch.allclose(shifted["transport_reveal_logits"], base["transport_reveal_logits"], atol=3e-5, rtol=0)


def test_sparse_transport_loss_has_finite_gradients() -> None:
    model = SparseCircularRelationTransportNet()
    output = model(_scans(2))
    presence = torch.zeros(2, 5, 180, dtype=torch.bool)
    identity = torch.full((2, 5, MAX_TOKENS), -1)
    for batch in range(2):
        for frame in range(5):
            bins = output["token_bin_index"][batch, frame, :2].detach()
            presence[batch, frame, bins] = True
            identity[batch, frame, :2] = torch.tensor([batch * 10 + 1, batch * 10 + 2])
    identity[0, 2, 1] = 99
    losses = sparse_relation_transport_loss(output, proposal_presence=presence, aligned_token_identity=identity)
    losses["total"].backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad and parameter.grad is not None]
    assert gradients and all(torch.isfinite(value).all() for value in gradients)


def test_causal_directional_outputs_ignore_future_frames() -> None:
    model = SparseCircularRelationTransportNet().eval()
    scans = _scans(1)
    changed = scans.clone(); changed[:, 3:] = torch.flip(changed[:, 3:], dims=(-1,))
    with torch.no_grad():
        base = model(scans); alternative = model(changed)
    assert torch.equal(base["proposal_logits"][:, :3], alternative["proposal_logits"][:, :3])


def test_soft_circular_proposal_prefers_target_over_distant_peak() -> None:
    presence = torch.zeros(1, 5, 180, dtype=torch.bool); presence[:, :, 0] = True
    correct = torch.full((1, 5, 180), -5.0); correct[:, :, 0] = 5.0
    distant = torch.roll(correct, 30, dims=-1)
    assert soft_circular_proposal_loss(correct, presence) < soft_circular_proposal_loss(distant, presence)


def test_count_loss_uses_explicit_zero_to_six_targets_and_mask() -> None:
    presence = torch.zeros(1, 5, 180, dtype=torch.bool)
    for frame, count in enumerate((0, 1, 2, 3, 6)):
        presence[0, frame, :count] = True
    correct = torch.full((1, 5, 7), -8.0)
    wrong = correct.clone()
    for frame, count in enumerate((0, 1, 2, 3, 6)):
        correct[0, frame, count] = 8.0
        wrong[0, frame, (count + 1) % 7] = 8.0
    valid = torch.tensor([[False, True, True, True, True]])
    assert token_count_loss(correct, presence, valid_frame_mask=valid) < token_count_loss(
        wrong, presence, valid_frame_mask=valid
    )


def test_proposal_and_transport_masks_exclude_invalid_history() -> None:
    model = SparseCircularRelationTransportNet()
    output = model(_scans(1))
    presence = torch.zeros(1, 5, 180, dtype=torch.bool)
    identity = torch.full((1, 5, MAX_TOKENS), -1)
    for frame in range(5):
        bins = output["token_bin_index"][0, frame, :2].detach()
        presence[0, frame, bins] = True
        identity[0, frame, :2] = torch.tensor([1, 2])
    frame_valid = torch.tensor([[False, True, True, True, True]])
    pair_valid = torch.tensor([[False, True, True, True]])
    losses = sparse_relation_transport_loss(
        output,
        proposal_presence=presence,
        aligned_token_identity=identity,
        proposal_valid_frame_mask=frame_valid,
        relation_pair_valid_mask=pair_valid,
        withdraw_weight=9.2,
        reveal_positive_weight=9.4,
    )
    assert torch.isfinite(losses["total"])
