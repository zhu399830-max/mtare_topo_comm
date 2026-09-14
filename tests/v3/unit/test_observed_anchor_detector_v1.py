from copy import deepcopy
import numpy as np
import pytest
import torch
from mtare_topo.representation.gse_block_points import bind_block_points
from mtare_topo.representation.gse_block_point_encoder import BlockPointEncoder
from mtare_topo.representation.observed_anchor_detector_v1 import (
    ObservedAnchorDetectorV1, residual_positions)


def observation():
    xyz=np.asarray([[-2,0,0],[-1.9,.1,0],[2,0,0],[2,.1,.1],[0,0,2]],np.float32)
    return bind_block_points(xyz,np.arange(5,dtype=np.int64),np.asarray([0,0,1,1,2],np.int64))


def test_source_indices_bounded_output_backward_and_frozen_context():
    torch.manual_seed(0)
    blocks=observation(); encoder=BlockPointEncoder()
    model=ObservedAnchorDetectorV1(relation_attributes=False)
    context=torch.randn(3,128,requires_grad=True)
    output=model(blocks,encoder(blocks),context)
    assert len(output.query_source_indices)==5
    assert torch.equal(output.query_positions_m,torch.tensor(blocks.xyz_m[output.query_source_indices.numpy()]))
    assert output.prediction.directions.shape==(5,64,3)
    assert torch.all(torch.linalg.vector_norm(output.prediction.position_m,dim=-1)<=10.00001)
    assert torch.all(output.residual_m.abs()<=20.)
    loss=output.prediction.position_m.square().mean()+output.prediction.presence_logits.sum()
    loss.backward()
    assert context.grad is None
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.position_encoding.parameters())
    assert not any(name=='queries' or name.startswith('queries.') for name,_ in model.named_parameters())


def test_paired_model_identical_state_distinct_relation_input():
    torch.manual_seed(1)
    blocks=observation(); features=BlockPointEncoder()(blocks); context=torch.zeros(3,128)
    a=ObservedAnchorDetectorV1(relation_attributes=False); b=deepcopy(a); b.relation_attributes=True
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    x=a(blocks,features,context); y=b(blocks,features,context)
    assert torch.equal(x.query_source_indices,y.query_source_indices)
    assert not torch.allclose(x.prediction.presence_logits,y.prediction.presence_logits)


def test_point_permutation_preserves_observed_query_coordinates_and_predictions():
    torch.manual_seed(4)
    blocks=observation();encoder=BlockPointEncoder()
    model=ObservedAnchorDetectorV1(relation_attributes=False)
    order=np.asarray([4,2,0,3,1])
    permuted=bind_block_points(blocks.xyz_m[order],blocks.frame_index[order],blocks.point_to_block[order])
    context=torch.zeros(3,128)
    a=model(blocks,encoder(blocks),context);b=model(permuted,encoder(permuted),context)
    assert torch.equal(a.query_positions_m,b.query_positions_m)
    assert torch.allclose(a.prediction.position_m,b.prediction.position_m,atol=2e-5)
    assert torch.allclose(a.prediction.presence_logits,b.prediction.presence_logits,atol=2e-5)


def test_residual_reaches_air_center_and_domain_projection():
    surface=torch.tensor([[9.,0,0],[9,0,0]])
    raw=torch.tensor([[-.48470028,0,0],[10,10,10]],requires_grad=True)
    position,residual=residual_positions(surface,raw)
    assert torch.linalg.vector_norm(position[0])<1e-5
    assert torch.allclose(torch.linalg.vector_norm(position[1]),torch.tensor(10.))
    position.sum().backward()
    assert torch.isfinite(raw.grad).all()


def test_empty_input_returns_no_guessed_queries():
    blocks=bind_block_points(np.empty((0,3),np.float32),np.empty(0,np.int64),np.empty(0,np.int64))
    features=dict(embedding=torch.empty(0,128),centers_m=torch.empty(0,3),
                  extent_m=torch.empty(0,3),degenerate=torch.empty(0,dtype=torch.bool))
    assert ObservedAnchorDetectorV1(relation_attributes=False)(blocks,features,torch.empty(0,128)) is None


def test_projection_respects_existing_double_precision_scoring_tolerance():
    torch.manual_seed(0)
    q=torch.randn(100000,3);q=q/q.norm(dim=1,keepdim=True)*9
    p,_=residual_positions(q,torch.randn_like(q)*5)
    assert bool(torch.all(p.double().norm(dim=1)<=10.000001))


def test_teacher_extra_field_and_mismatched_geometry_rejected():
    blocks=observation(); features=BlockPointEncoder()(blocks)
    model=ObservedAnchorDetectorV1(relation_attributes=False)
    with pytest.raises(ValueError): model(blocks,dict(features,teacher_center=torch.zeros(3)),torch.zeros(3,128))
    with pytest.raises(ValueError): model(blocks,dict(features,centers_m=features['centers_m']+1),torch.zeros(3,128))
