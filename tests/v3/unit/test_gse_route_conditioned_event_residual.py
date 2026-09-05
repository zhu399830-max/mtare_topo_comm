from __future__ import annotations
import numpy as np
import pytest
import torch
from mtare_topo.representation.gse_route_conditioned_event_residual import RouteConditionedEventResidual,corrected_event_distribution,route_flow_features

def _history(n):
 ref=np.zeros((n,5),dtype=np.int64);mask=np.zeros((n,5),dtype=bool)
 for row in range(n):
  values=np.arange(max(0,row-4),row+1);ref[row,-len(values):]=values;mask[row,-len(values):]=True
 return ref,mask
def test_zero_init_preserves_base_distribution():
 torch.manual_seed(2);base={"structural_logit":torch.randn(5),"conditional_decision_logits":torch.randn(5,2)};feature=torch.randn(5,16);model=RouteConditionedEventResidual();out=corrected_event_distribution(base,model,feature)
 torch.testing.assert_close(out["structural_logit"],base["structural_logit"],rtol=0,atol=0);torch.testing.assert_close(out["conditional_decision_logits"],base["conditional_decision_logits"],rtol=0,atol=0);assert sum(p.numel() for p in model.parameters())==643
def test_backward_flow_score_exceeds_forward_flow_score():
 raw=np.zeros((2,3,6,40),dtype=np.float32);raw[...,0]=1;raw[0,...,2]=-1;raw[1,...,2]=1;ref,mask=_history(2);feature=route_flow_features(raw,ref,mask);assert feature[0,3]>0.99 and feature[1,3]<-0.99
def test_route_flow_rejects_future_reference():
 raw=np.zeros((2,3,6,40),dtype=np.float32);raw[...,0]=1;ref,mask=_history(2);ref[0,-1]=1
 with pytest.raises(ValueError,match="causal"):route_flow_features(raw,ref,mask)
def test_route_flow_is_token_permutation_invariant():
 rng=np.random.default_rng(4);raw=rng.normal(size=(5,3,6,40));raw[...,0]=rng.uniform(0,1,size=(5,3,6));ref,mask=_history(5);first=route_flow_features(raw,ref,mask);second=route_flow_features(raw[:,:,[5,2,0,3,1,4]],ref,mask);np.testing.assert_allclose(first,second,rtol=1e-6,atol=1e-6)
