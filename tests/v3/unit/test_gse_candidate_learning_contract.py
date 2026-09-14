import numpy as np
import torch
from mtare_topo.representation.gse_candidate_neighbors import candidate_geometry
from mtare_topo.representation.gse_candidate_objective import candidate_targets,balanced_candidate_loss
from mtare_topo.evaluation.gse_candidate_selection import score_candidate_selection


def test_neighbors_keep_unknown_normals_and_rays():
    p=torch.tensor([[0.,0,0],[1.,0,0],[0.,0,4]])
    g=candidate_geometry(p,torch.zeros(3,18),torch.zeros(3,18,dtype=torch.bool))
    assert g.neighbor_valid.sum()==6 and not g.relation[:,:,:,5].any()
    assert g.relation[:,:,:,8].all() and not g.relation[:,:,:,6:8].any()
    assert torch.equal(g.neighbor_index[0,:,0],torch.tensor([1,0,0]))


def test_singleton_has_no_fake_neighbor():
    g=candidate_geometry(torch.zeros(1,3),torch.zeros(1,18),torch.zeros(1,18,dtype=torch.bool))
    assert not g.neighbor_valid.any() and (g.neighbor_index==-1).all()


def test_duplicate_targets_are_consistent_and_unknown_background_masked():
    p=np.array([[0.,0,0],[0.,0,0],[.25,0,0],[4.,0,0]])
    y,m,missed=candidate_targets(p,np.zeros((1,3)),complete_region=False)
    np.testing.assert_allclose(y,[1,1,.75,0]);assert m.tolist()==[True,True,True,False] and missed==0
    x=torch.zeros(4,requires_grad=True);loss,counts=balanced_candidate_loss(x,torch.from_numpy(y),torch.from_numpy(m))
    loss.backward();assert x.grad[-1]==0 and counts==dict(positive=3,negative=0)


def test_complete_straight_background_penalized():
    y,m,missed=candidate_targets(np.zeros((3,3)),np.empty((0,3)),complete_region=True)
    x=torch.ones(3,requires_grad=True);loss,counts=balanced_candidate_loss(x,torch.from_numpy(y),torch.from_numpy(m))
    loss.backward();assert (x.grad>0).all() and counts['negative']==3


def test_suppression_preserves_raw_false_positives_and_can_lose_nearby_nodes():
    p=np.array([[0.,0,0],[.1,0,0],[5.,0,0]])
    r=score_candidate_selection(p,np.array([.9,.8,.1]),np.array([[0.,0,0]]),complete_region=True)
    assert r['raw']['scores']['1.0']['false_positives']==1
    assert r['kept_indices']==[0] and r['suppressed']==[dict(candidate=1,by=0)]
    r=score_candidate_selection(p[:2],np.array([.9,.8]),p[:2],complete_region=True)
    assert r['after_suppression']['scores']['1.0']['missed']==1


def test_unknown_predictions_no_precision_claim():
    p=np.array([[0.,0,0]])
    r=score_candidate_selection(p,np.array([.9]),np.empty((0,3)),complete_region=False)
    assert r['after_suppression']['scores']['1.0']['precision'] is None
