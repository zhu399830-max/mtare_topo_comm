import numpy as np
from mtare_topo.representation.fixed_score_l2_v1 import regularized,solve,stable_scores

def test_penalty_gradient_and_bias_exclusion():
    X=np.array([[1.,1.],[-1.,1.]]);y=np.array([1.,-1.]);w=np.ones(2)/2;t=np.array([.4,.2])
    f,g=regularized(t,X,y,w,.01)
    for j in range(2):
        delta=np.eye(2)[j]*1e-5
        numerical=(regularized(t+delta,X,y,w,.01)[0]-regularized(t-delta,X,y,w,.01)[0])/2e-5
        assert abs(numerical-g[j])<1e-9
    _,g0=regularized(t,X,y,w,0.)
    np.testing.assert_allclose(g-g0,[.004,0.],atol=1e-15)

def test_bounded_solver_and_stability_reject_boundary():
    X=np.array([[1.,1.],[-1.,1.]]);y=np.array([1.,-1.]);w=np.ones(2)/2
    t,r=solve(X,y,w,np.zeros(2),.01)
    assert r['converged'] and np.all(y*(X@t)>0) and np.linalg.norm(t)<10
    assert not stable_scores(np.array([1.,-1.]),[np.array([0.,-1.])],np.array([True,True]))['pass_stability']

def test_fixed_region_varies_matching_not_known_masks_or_output():
    from fixed_score_l2_v1 import fixed_region_replay
    row=dict(source={'id':1},feature=np.zeros((3,128)),position_m=np.array([[.7,0,0],[3.,0,0],[8.,0,0]]),targets=np.array([[0.,0,0]]))
    ref=dict(observations=[dict(source=row['source'],scores={'4.0':dict(query_indices=[0,1,2],coverage=dict(query_scoreable_mask=[True,True,False],possible_unconfirmed_reference_mask=[False,False,True]))})])
    result=fixed_region_replay([row],np.ones(3,dtype=np.float32),ref)
    for score in result['observations'][0]['scores'].values():
        assert score['output_count']==3 and score['query_indices']==[0,1,2] and score['ignored']==1
    assert result['summary']['0.5']['fp']==2 and result['summary']['1.0']['fp']==1
