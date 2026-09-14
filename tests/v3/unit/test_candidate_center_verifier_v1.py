import numpy as np
import pytest
import torch
from mtare_topo.representation.candidate_center_verifier_v1 import CandidateCenterVerifierV1,balanced_validity_loss
from mtare_topo.representation.candidate_validity_v1 import candidate_validity
from mtare_topo.evaluation.candidate_center_selection_v1 import select_candidates

def packet():
    torch.manual_seed(0)
    return dict(candidate_xyz=torch.randn(3,3),block_xyz=torch.randn(7,3),block_extent=torch.rand(7,3),block_features=torch.randn(7,128),
        block_valid=torch.tensor([True]*6+[False]),block_degenerate=torch.zeros(7,1))

def test_architecture_padding_and_permutations():
    p=packet();model=CandidateCenterVerifierV1();assert sum(x.numel() for x in model.parameters())==21185
    a=model(**p);idx=torch.tensor([6,3,0,5,1,4,2]);q={k:v[idx] if k.startswith('block_') else v for k,v in p.items()}
    torch.testing.assert_close(a,model(**q),atol=1e-5,rtol=1e-5)
    q=dict(p,candidate_xyz=p['candidate_xyz'][[2,0,1]])
    torch.testing.assert_close(a[[2,0,1]],model(**q),atol=1e-5,rtol=1e-5)
    for k in ('block_xyz','block_extent','block_features','block_degenerate'):p[k][-1]=float('nan')
    torch.testing.assert_close(a,model(**p))

def test_same_center_same_score_and_relative_geometry_gradient():
    p=packet();p['candidate_xyz'][1]=p['candidate_xyz'][0];model=CandidateCenterVerifierV1();out=model(**p)
    assert out[0]==out[1]
    p['candidate_xyz']=p['candidate_xyz'].clone().requires_grad_();out=model(**p);out[0].backward()
    assert p['candidate_xyz'].grad[0].abs().sum()>0 and p['candidate_xyz'].grad[1:].abs().sum()==0
    assert model.pair[0].weight.grad[:,-7:-4].abs().sum()>0
    assert all(v.grad is not None for v in model.parameters())

def test_empty_observation_finite_no_background():
    p=packet()
    for k in p:
        if k.startswith('block_'):p[k]=p[k][:0]
    out=CandidateCenterVerifierV1()(**p);assert torch.isfinite(out).all() and torch.equal(out,torch.zeros(3))
    loss=balanced_validity_loss(out,torch.zeros(3,dtype=torch.bool),torch.zeros(3,dtype=torch.bool));assert loss==0

def test_unknown_direct_gradient_zero_and_negative_direction():
    z=torch.tensor([.1,.2,.3],requires_grad=True);loss=balanced_validity_loss(z,torch.tensor([1,0,0],dtype=torch.bool),torch.tensor([0,1,0],dtype=torch.bool));loss.backward()
    assert z.grad[0]<0 and z.grad[1]>0 and z.grad[2]==0

def evidence_fixture():
    source={'case':1};q=np.array([[0.,0,0],[.5,0,0],[5.,0,0],[2.,0,0]])
    cov=dict(source_binding={'source':source},query_xyz_m=q.tolist(),possible_unconfirmed_reference_mask=[False]*4)
    bg=dict(binding={'source':source},observation_identity_binding_verified=True,inventory_completeness_supplied_not_verified=False,observed_states_supplied_not_verified=False,
        reference_negative_mask=[False,False,True,False],reason=['UNKNOWN','UNKNOWN','BOUND4M_NEGATIVE','UNKNOWN'],grid_content_sha256='test')
    e=dict(original_background=bg,coverage=cov,background_radius_m=4.,used_duplicate_mask=[False,True,False,True])
    old=dict(positive=[True,False,False,False],negative=[False,True,True,True],unknown=[False]*4)
    return source,q,cov,e,old

def test_validity_not_unique_query_labels():
    source,q,cov,e,old=evidence_fixture();r=candidate_validity(q,np.array([[0.,0,0]]),e,cov,source,old)
    assert r['positive']==[True,True,False,False] and r['negative']==[False,False,True,False] and r['unknown']==[False,False,False,True]
    assert r['transition']['negative->positive']==1 and r['transition']['negative->unknown']==1

def test_conflicting_positive_negative_stops():
    source,q,cov,e,old=evidence_fixture();e['original_background']['reference_negative_mask'][0]=True
    with pytest.raises(ValueError,match='CONFLICT'):candidate_validity(q,np.array([[0.,0,0]]),e,cov,source,old)

def test_greedy_not_transitive_and_xyz_not_xy():
    q=np.array([[0.,0,0],[1.5,0,0],[3.,0,0],[0.,0,3.]])
    r=select_candidates(q,np.array([4.,3.,2.,1.]));assert r['after']==[0,2,3] and r['suppressed'][0]['suppressor']==0

def test_deterministic_ties_and_boundary():
    q=np.array([[2.,0,0],[0.,0,0],[5.,0,0]])
    r=select_candidates(q,np.zeros(3,dtype=np.float32));assert r['after']==[1,2] and r['suppressed'][0]['candidate']==0
    assert select_candidates(q,np.full(3,-1.))['after']==[]
