from dataclasses import replace
import json
import pytest
import torch

from test_gse_surface_prediction_binding_v1 import binder, forward, IdealSyntheticModel
from test_gse_surface_graph_v1 import graph, pose


@pytest.fixture(autouse=True)
def single_thread():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def partial_binder():
    b=binder(IdealSyntheticModel())
    b.readout=replace(b.readout,supervision_policy='partial_geometry_v1')
    return b


def test_partial_default_unknown_not_untrained_scale_or_dimensions():
    b=partial_binder();result=forward(b)
    assert all(a.uncertainty_m is None for a in result.observation.anchors)
    assert all(o.width_m is None and o.height_m is None and o.traversability=='unknown'
               for o in result.observation.openings)
    assert not json.loads(result.proof_json)['robot_uncertainty_available']


def test_extreme_untrained_heads_do_not_change_partial_readout():
    b=partial_binder()
    captured=[]
    handle=b.model.register_forward_hook(lambda module,args,output: captured.append(output))
    forward(b);handle.remove();p=captured[0]
    changed=replace(p,anchor_uncertainty_m=torch.full_like(p.anchor_uncertainty_m,1e6),
        opening_dimension_evidence_logits=torch.full_like(p.opening_dimension_evidence_logits,-100.),
        opening_support_logits=torch.full_like(p.opening_support_logits,-100.),
        membership_validity_logits=torch.full_like(p.membership_validity_logits,-100.),
        reachability_logits=torch.full_like(p.reachability_logits,100.))
    from test_gse_surface_prediction_binding_v1 import I
    _,a,o=b._readout(p,I);_,aa,oo=b._readout(changed,I)
    assert (a,o)==(aa,oo)
    assert any(v for opening in o for v in opening.anchor_relation_valid)
    assert any(opening.position_robot_m is not None for opening in o)


def test_partial_graph_policy_must_be_explicit_and_keeps_stable_nodes():
    b=partial_binder();g=graph(stable=1);result=forward(b)
    with pytest.raises(ValueError,match='explicit known-pose'):
        b.update_graph(result,g,pose(0))
    g.config=replace(g.config,anchor_uncertainty_policy='known_pose_partial_diagnostic')
    b.update_graph(result,g,pose(0))
    assert len(g.snapshot()['nodes'])>0


def test_empty_observation_does_not_gain_geometry_from_partial_policy():
    from test_gse_surface_prediction_binding_v1 import packets
    b=partial_binder();result=forward(b,packets=packets(supported=False))
    assert all(a.position_robot_m is None for a in result.observation.anchors)
    assert all(o.position_robot_m is None and not any(o.anchor_relation_valid)
               for o in result.observation.openings)


def test_partial_mode_keeps_multiframe_stability_and_no_untraversed_edges():
    b=partial_binder();g=graph(stable=2)
    g.config=replace(g.config,anchor_uncertainty_policy='known_pose_partial_diagnostic')
    b.update_graph(forward(b,step=0),g,pose(0))
    assert not g.snapshot()['nodes']
    b.update_graph(forward(b,step=1),g,pose(1))
    assert g.snapshot()['nodes']
    assert not g.snapshot()['edges']
