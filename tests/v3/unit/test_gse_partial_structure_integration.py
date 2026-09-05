"""Synthetic teacher -> loss-only direction bridge -> paired small-head updates."""
from dataclasses import fields

import torch

from mtare_topo.representation.gse_region_queries import RegionTargets
from mtare_topo.representation.gse_region_target_bridge import align_axis_directions, partial_region_targets
from mtare_topo.representation.gse_partial_structure_training import (
    PartialStructureExample, PartialTrainingConfig, train_partial_structure,
)


def test_partial_targets_to_three_heads_without_identity_forward_or_slot_filtering():
    gt = torch.zeros(3, 32, 3, 3)
    gt[:, 0] = torch.tensor([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
    gt[:, 1] = torch.tensor([[0., 0., 0.], [0., 1., 0.], [0., 2., 0.]])
    mask = torch.zeros(3, 32, dtype=torch.bool); mask[:, :2] = True
    pred = gt.clone(); pred[:, :2] += .125
    for slot in range(2, 32):
        pred[:, slot] = torch.tensor([[10.+slot, 0., 0.], [10.+slot, 1., 0.], [10.+slot, 2., 0.]])
    rows = [{"observation_label_complete": False, "regions": [{
        "construction_node_id_teacher_only": "never_forwarded",
        "center_current_sensor_m": [0., 0., 0.], "center_valid": True,
        "event_target": "corridor", "event_valid": True,
        "directional_member_target": [1, 0, 1, 0] + [0] * 60,
        "directional_member_valid": [True, False, True, False] + [False] * 60,
    }]} for _ in range(3)]
    gt_bridge = partial_region_targets(rows, align_axis_directions(gt, gt, mask), dtype=gt.dtype)
    pred_bridge = partial_region_targets(rows, align_axis_directions(pred, gt, mask), dtype=pred.dtype)
    assert sum(x["transferred_member_positive"] for x in pred_bridge.ledger) == 6
    assert not pred_bridge.targets.label_complete.any()
    def examples(axes, target):
        return [PartialStructureExample(axes[i:i+1], RegionTargets(**{
            f.name: getattr(target, f.name)[i:i+1] for f in fields(RegionTargets)})) for i in range(3)]
    predicted = examples(pred, pred_bridge.targets)
    result = train_partial_structure({"gt_axes": examples(gt, gt_bridge.targets),
        "predicted_axes": predicted, "predicted_no_relations": predicted},
        PartialTrainingConfig(seed=0, steps=2, batch_size=2, lr=.001, device="cpu"), hidden=8)
    assert set(result.heads) == {"gt_axes", "predicted_axes", "predicted_no_relations"}
    assert all(len(history) == 2 for history in result.history.values())
    assert all(record["gradient_l2"] > 0 for history in result.history.values() for record in history)
    assert all(example.axes.shape == (1, 32, 3, 3) for example in predicted)
    assert all(torch.equal(example.axes, pred[i:i+1]) for i, example in enumerate(predicted))
