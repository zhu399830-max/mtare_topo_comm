import numpy as np
import torch
from mtare_topo.representation.fixed_score_numerics_v1 import equivalent_weights,objective,separability,solve_once
from mtare_topo.representation.geometry_match_presence_v1 import restore_presence

def test_equivalent_group_weights_and_gradient():
    masks=[([True,False,False],[False,True,False]),([False,False],[True,True])]
    schedule=[[0,1,0,1],[1,1,0,1]]
    w,counts=equivalent_weights(masks,schedule);assert counts.tolist()==[3,5] and w[2]==0
    X=np.arange(15,dtype=float).reshape(5,3)/10;theta=np.array([.2,-.1,.5]);y=np.array([1,-1,1,-1,-1])
    value,g=objective(theta,X,y,w);t=torch.tensor(theta,requires_grad=True);logits=torch.tensor(X)@t
    total=0.;offset=0
    for (p,n),count in zip(masks,counts):
        r=restore_presence(dict(assignment=torch.tensor(np.flatnonzero(p)),negative_mask=torch.tensor(n),position=torch.tensor(0.)),logits[offset:offset+len(p)])
        total=total+r['presence']*count/8;offset+=len(p)
    total.backward();assert abs(float(total.detach())-value)<1e-12
    np.testing.assert_allclose(t.grad.numpy(),g,atol=1e-12)

def test_separable_with_exact_witness():
    X=np.array([[-1.,1.],[1.,1.]]);r=separability(X,np.array([-1,1]))
    assert r['status']=='STRICTLY_SEPARABLE_CERTIFIED' and r['min_signed_margin']>0

def test_contradictory_features_certified():
    r=separability(np.array([[1.,1.],[1.,1.]]),np.array([-1,1]))
    assert r['status']=='NOT_STRICTLY_SEPARABLE_CERTIFIED'

def test_one_solve_and_analytic_gradient():
    X=np.array([[-1.,1.],[1.,1.]]);y=np.array([-1,1]);w=np.ones(2)/2
    t,r=solve_once(X,y,w,np.zeros(2))
    assert np.all(y*(X@t)>0) and r['loss']<1e-8
    assert r['first_order_tolerance_met']

def test_unknown_does_not_change_objective():
    w,_=equivalent_weights([([True,False,False],[False,True,False])],[[0,0,0,0]])
    X=np.array([[1.,1.],[-1.,1.],[999.,1.]]);y=np.array([1,-1,1]);t=np.array([.2,.3])
    a=objective(t,X,y,w);X[-1]=[-888.,1.];b=objective(t,X,y,w)
    assert a[0]==b[0];np.testing.assert_array_equal(a[1],b[1])

def test_serialized_pair_equivalence_preserves_actual_values():
    from fixed_score_numerics_v1 import json_value_equal
    assert json_value_equal([(2,0),(5,1)],[[2,0],[5,1]])
    assert not json_value_equal([(2,0),(5,1)],[[2,1],[5,0]])
