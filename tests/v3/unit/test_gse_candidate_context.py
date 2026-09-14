from dataclasses import replace
import torch
import pytest
from mtare_topo.representation.gse_candidate_context import ObservationBinding,bind_candidate_context
from mtare_topo.representation.gse_dual_path_encoder_adapter_v1 import CompactFrozenDualPathFeatures


def fixture():
    idx=(torch.arange(5)[:,None,None]*180+torch.arange(720)[None,None]//4).expand(5,16,720).reshape(1,-1)
    p=torch.zeros(1,57600,3);p[0,0,0]=-1;p[0,4,0]=1
    v=torch.zeros(1,57600,dtype=torch.bool);v[0,[0,4]]=True
    f=torch.zeros(1,900,128);f[0,0]=2;f[0,1]=3
    c=CompactFrozenDualPathFeatures(p,f,v,idx)
    s=ObservationBinding('test',(0,1,2,3,4),'current_sensor','a'*64,'b'*64)
    return c,s


def test_lookup_ties_and_distance_not_surface_claim():
    c,s=fixture();r=bind_candidate_context(c,torch.tensor([[0.,0,0],[1.,0,0]]),s,s)
    assert r.source_point_index.tolist()==[0,4]
    assert r.sensor_token_index.tolist()==[0,1]
    assert r.nearest_return_distance_m.tolist()==[1.,0.]
    assert r.features[:,0].tolist()==[2.,3.]


@pytest.mark.parametrize('field,value',[('observation_id','other'),('frame_indices',(1,2,3,4,5)),('input_sha256','c'*64),('encoder_sha256','c'*64)])
def test_mismatched_sources_rejected(field,value):
    c,s=fixture()
    with pytest.raises(ValueError):bind_candidate_context(c,torch.zeros(1,3),s,replace(s,**{field:value}))


def test_empty_valid_unknown_and_wrong_layout_fail():
    c,s=fixture();r=bind_candidate_context(replace(c,valid=torch.zeros_like(c.valid)),torch.zeros(2,3),s,s)
    assert not r.supported.any() and (r.source_point_index==-1).all()
    assert torch.isnan(r.nearest_return_distance_m).all()
    with pytest.raises(ValueError):bind_candidate_context(replace(c,sensor_token_index=torch.zeros_like(c.sensor_token_index)),torch.zeros(1,3),s,s)
