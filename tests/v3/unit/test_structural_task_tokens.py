import numpy as np
import pytest
from mtare_topo.topology.structural_task_tokens import bind_task_tokens,retrieval_candidates,bind_directional_context

KEYS=[str(i) for i in range(5)]
def proposal(refs):return dict(source_refs=refs,axis_start_xyz_m=[0,0,0],axis_target_xyz_m=[1,0,0])

def test_unknown_kept_and_no_fake_edges():
    rows=bind_task_tokens([proposal(['4/ray:0'])],KEYS,np.full(57600,-1),np.ones((2,128)))
    assert rows[0]['descriptor'] is None and not rows[0]['creates_edge']
    assert retrieval_candidates(rows,rows)[0]['status']=='UNKNOWN'

def test_exact_sources_and_permutation():
    m=np.full(57600,-1);m[4*11520]=0;m[4*11520+1]=1
    t=np.zeros((2,128));t[0,0]=1;t[1,1]=1
    a=bind_task_tokens([proposal(['4/ray:0','4/ray:1','4/ray:0'])],KEYS,m,t)
    b=bind_task_tokens([proposal(['4/ray:1','4/ray:0'])],KEYS,m,t)
    assert a==b and a[0]['candidate_source_rays']==2
    assert not retrieval_candidates(a,b)[0]['merge_committed']

def test_foreign_support_rejected():
    with pytest.raises(ValueError):bind_task_tokens([proposal(['future/ray:0'])],KEYS,np.full(57600,-1),np.ones((1,128)))

def test_equal_descriptors_do_not_merge():
    t=np.ones((1,128));m=np.zeros(57600,dtype=int)
    rows=bind_task_tokens([proposal(['4/ray:0']),proposal(['4/ray:1'])],KEYS,m,t)
    matches=retrieval_candidates(rows,rows)
    assert all(x['best_previous']==[0,1] and not x['merge_committed'] for x in matches)

def test_side_surface_context_and_no_membership_claim():
    c=np.array([[2,3,0],[-2,3,0],[0,3,0]],float);t=np.eye(3,128)
    r=bind_directional_context([proposal([])],c,t,np.eye(4))[0]
    assert r['context_patch_ids']==[0] and r['descriptor'][0]==1
    assert not r['membership_certified'] and not r['creates_edge']

def test_context_rotation_and_permutation():
    c=np.array([[2,3,1],[4,-2,0],[-2,3,0]],float);t=np.eye(3,128)
    a=bind_directional_context([proposal([])],c,t,np.eye(4))[0]
    pose=np.eye(4);pose[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]]
    rotated=dict(proposal([]),axis_target_xyz_m=[0,1,0])
    b=bind_directional_context([rotated],c[::-1],t[::-1],pose)[0]
    assert np.allclose(a['descriptor'],b['descriptor'])

def test_context_all_behind_remains_unknown():
    r=bind_directional_context([proposal([])],np.array([[-1,0,0.]]),np.ones((1,128)),np.eye(4))[0]
    assert r['descriptor'] is None
