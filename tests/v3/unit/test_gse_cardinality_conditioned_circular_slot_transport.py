from __future__ import annotations

import torch

from mtare_topo.representation.gse_cardinality_conditioned_circular_slot_transport import (
    BRANCH_SLICE,
    CardinalityConditionedCircularSlotTransportNet,
    circular_slot_transport_contract,
)


def test_contract_has_required_slots_without_existence_query() -> None:
    contract = circular_slot_transport_contract()
    assert contract["slots_by_cardinality"] == {1: 1, 2: 2, 3: 3, 4: 4}
    assert contract["existence_objectness"] is None
    model = CardinalityConditionedCircularSlotTransportNet()
    assert not any("query" in name or "objectness" in name for name, _ in model.named_parameters())


def test_branch_slices_are_disjoint_and_complete() -> None:
    values = []
    for count in range(1, 5):
        branch = BRANCH_SLICE[count]
        assert branch.stop - branch.start == count
        values.extend(range(branch.start, branch.stop))
    assert values == list(range(10))


def test_forward_has_exact_predicted_cardinality() -> None:
    torch.manual_seed(3)
    model = CardinalityConditionedCircularSlotTransportNet().eval()
    scans = torch.rand(2, 5, 2, 16, 720)
    scans[:, :, 1] = (scans[:, :, 1] > 0.5).float()
    with torch.no_grad():
        output = model(scans)
    assert output["slot_mass"].shape == (2, 10, 180)
    assert torch.allclose(output["slot_mass"].sum(-1), torch.ones(2, 10), atol=1e-6)
    assert torch.equal(output["decoded_valid_mask"].sum(-1), output["decoded_count"])
    assert bool(torch.isfinite(output["decoded_set_confidence"]).all())
