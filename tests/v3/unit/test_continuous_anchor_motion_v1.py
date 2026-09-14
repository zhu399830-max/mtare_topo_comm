import numpy as np
import pytest
from mtare_topo.evaluation.continuous_anchor_motion_v1 import rotation,compose_current_frames,transform_candidates,set_motion_summary


def fixture():
    f=np.array([np.arange(i,i+5) for i in range(6)])
    yaw=np.arange(10)*7.;xyz=np.array([[i,i*i*.1,i*.05] for i in range(10)])
    t=np.array([(xyz[row]-xyz[row[-1]])@rotation(yaw[row[-1]]) for row in f])
    y=np.array([yaw[row]-yaw[row[-1]] for row in f])
    return t,y,f,xyz,yaw


def test_rotation_translation_and_prefix():
    t,y,f,xyz,yaw=fixture();r,p=compose_current_frames(t,y,f)
    np.testing.assert_allclose(p,(xyz[4:]-xyz[4])@rotation(yaw[4]),atol=1e-12)
    for n in range(1,7):
        a,b=compose_current_frames(t[:n],y[:n],f[:n]);np.testing.assert_array_equal(a,r[:n]);np.testing.assert_array_equal(b,p[:n])
    fixed=np.tile([3.,4.,1.],(32,1))
    local=np.array([(fixed-pi)@ri for ri,pi in zip(r,p)])
    world=transform_candidates(local,r,p)
    np.testing.assert_allclose(world,np.broadcast_to(fixed,world.shape),atol=1e-12)
    assert max(set_motion_summary(world,np.ones((6,32),bool)))<1e-12


def test_robot_attached_candidates_move_and_query_permutation_is_irrelevant():
    t,y,f,_,_=fixture();r,p=compose_current_frames(t,y,f)
    x=np.tile(np.arange(96).reshape(32,3)*.01,(6,1,1));world=transform_candidates(x,r,p);mask=np.ones((6,32),bool)
    a=set_motion_summary(world,mask);assert min(a)>.5
    shuffled=world.copy();shuffled[::2]=shuffled[::2,::-1]
    np.testing.assert_allclose(set_motion_summary(shuffled,mask),a)


def test_wrong_history_or_motion_rejected():
    t,y,f,_,_=fixture()
    bad=f.copy();bad[1]+=1
    with pytest.raises(ValueError):compose_current_frames(t,y,bad)
    t[2,0,0]+=.1
    with pytest.raises(ValueError,match='overlapping'):compose_current_frames(t,y,f)
