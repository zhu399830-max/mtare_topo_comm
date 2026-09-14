from copy import deepcopy

import pytest

from mtare_topo.teacher.gse_surface_target_adapter_v1 import observed_targets


def record():
    return dict(schema="gse_surface_observed_targets_v1", coordinate_frame="current_sensor_m",
        source_frame_indices=[1,2,3,4,5], anchors=[dict(position_m=[0,0,0],evidence="synthetic")],
        openings=[dict(position_m=[2,0,0],direction=None,width_m=None,height_m=2,evidence="synthetic")],
        membership=[[None]], score_region=dict(center_m=[0,0,0],radius_m=10,
            anchors_complete=False,openings_complete=False,evidence="not completely observed"))


def test_unknown_is_not_negative_or_physics():
    target = observed_targets([record()])
    assert target.anchor_valid.all() and target.opening_valid.all()
    assert target.dimension_valid.tolist() == [[[False,True]]]
    assert not target.direction_valid.any() and not target.membership_valid.any()
    assert not target.reachability_valid.any() and not target.physical_reference_valid.any()
    assert not target.anchor_region_complete.any() and not target.opening_region_complete.any()


def test_multiple_anchors_and_independent_membership():
    row = record(); row["anchors"].append(dict(position_m=[0,0,4],evidence="stacked"))
    row["membership"] = [[True,False]]
    t = observed_targets([row])
    assert t.membership.tolist() == [[[1,0]]] and t.membership_valid.all()


def test_opening_without_anchor_and_empty_padding():
    row = record(); row["anchors"] = []; row["membership"] = [[]]
    empty = deepcopy(row); empty["openings"] = []; empty["membership"] = []
    t = observed_targets([row,empty])
    assert t.anchor_position_m.shape == (2,0,3)
    assert t.opening_valid.tolist() == [[True],[False]]
    assert not t.opening_region_complete.any()


@pytest.mark.parametrize("field,value", [("source_frame_indices",[1,2,3,5,4]),("coordinate_frame","world"),
    ("membership",[[0]]),("node_id",42)])
def test_reject_bad_provenance_and_gt_fields(field,value):
    row = record(); row[field] = value
    with pytest.raises(ValueError): observed_targets([row])


def test_capacity_never_truncates():
    row = record(); row["anchors"] *= 33
    with pytest.raises(OverflowError): observed_targets([row])


def test_input_owned_and_complete_not_inferred():
    row = record(); original = deepcopy(row); target = observed_targets([row])
    assert row == original
    row["anchors"][0]["position_m"][0] = 9
    assert target.anchor_position_m[0,0,0].item() == 0


@pytest.mark.parametrize("key,value", [("width_m",0),("direction",[2,0,0]),("evidence","")])
def test_bad_known_attributes(key,value):
    row = record(); row["openings"][0][key] = value
    with pytest.raises(ValueError): observed_targets([row])


def test_adapter_targets_reach_actual_loss_without_unknown_supervision():
    from dataclasses import fields
    import torch
    from test_gse_surface_losses_v1 import prediction
    from mtare_topo.representation.gse_surface_relation_model_v1 import SurfaceRelationPrediction
    from mtare_topo.representation.gse_surface_losses_v1 import surface_relation_losses
    old = prediction()
    p = SurfaceRelationPrediction(**{f.name: (getattr(old,f.name).float().detach().requires_grad_()
        if getattr(old,f.name).dtype != torch.bool else getattr(old,f.name)) for f in fields(old)})
    result = surface_relation_losses(p, observed_targets([record()]))
    assert result.has_supervision
    assert result.denominators["reachability"] == result.denominators["membership"] == 0
    assert result.denominators["opening_dimensions"] == 1
    assert result.denominators["anchor_presence"] == result.denominators["opening_presence"] == 1
    result.total.backward()
    assert torch.isfinite(p.anchor_presence_logits.grad).all()
    assert p.reachability_logits.grad is None or not p.reachability_logits.grad.any()
