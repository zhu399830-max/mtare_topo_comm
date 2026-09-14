import pytest
import torch
from mtare_topo.representation.primitive_relation_model import register_causal_lidar_points


def inputs():
    x=torch.ones(1,5,2,16,720);x[:,:,0]=.2
    return x,torch.zeros(1,5,3),torch.zeros(1,5)


def test_identity_preserves_old_points_and_roll_rotates_z():
    x,t,y=inputs();r=torch.eye(3).repeat(1,5,1,1)
    old,mask=register_causal_lidar_points(x,t,y)
    new,valid=register_causal_lidar_points(x,t,y,relative_rotation_current_sensor=r)
    torch.testing.assert_close(new,old,rtol=0,atol=0)
    assert torch.equal(mask,valid)
    r[:,0]=torch.tensor([[1.,0,0],[0,0,-1],[0,1,0]])
    new,_=register_causal_lidar_points(x,t,y,relative_rotation_current_sensor=r)
    torch.testing.assert_close(new[:,0,:,:,2],old[:,0,:,:,1])
    torch.testing.assert_close(new[:,0,:,:,1],-old[:,0,:,:,2])


def test_reflection_and_nonidentity_current_rejected():
    x,t,y=inputs();r=torch.eye(3).repeat(1,5,1,1);r[:,0,0,0]=-1
    with pytest.raises(ValueError,match='proper rotations'):
        register_causal_lidar_points(x,t,y,relative_rotation_current_sensor=r)
    r=torch.eye(3).repeat(1,5,1,1);r[:,-1]=torch.tensor([[0.,-1,0],[1,0,0],[0,0,1]])
    with pytest.raises(ValueError,match='exact current identity'):
        register_causal_lidar_points(x,t,y,relative_rotation_current_sensor=r)


def test_yaw_matrix_matches_legacy_with_translation_and_invalid_returns():
    x,t,y=inputs();y[0,:4]=torch.tensor([-30.,12.,90.,-4.]);t[0,0]=torch.tensor([1.,2.,3.])
    x[:,:,1,0,0]=0
    a=torch.deg2rad(y);c=a.cos();s=a.sin();r=torch.eye(3).repeat(1,5,1,1)
    r[...,0,0]=c;r[...,0,1]=-s;r[...,1,0]=s;r[...,1,1]=c
    old,valid=register_causal_lidar_points(x,t,y)
    new,mask=register_causal_lidar_points(x,t,y,relative_rotation_current_sensor=r)
    torch.testing.assert_close(new,old,rtol=1e-6,atol=1e-6)
    assert torch.equal(valid,mask) and torch.equal(new[:,:,0,0],torch.zeros(1,5,3))


def test_observable_derived_forward_accepts_rigid_without_parameter_changes():
    from mtare_topo.representation.primitive_relation_observable_model import ObservableSparsePortRelationNet
    torch.manual_seed(7);model=ObservableSparsePortRelationNet().eval()
    before={k:v.clone() for k,v in model.state_dict().items()}
    x,t,y=inputs();r=torch.eye(3).repeat(1,5,1,1)
    with torch.no_grad():
        old=model(x,t,y)
        new=model(x,t,y,relative_rotation_current_sensor=r)
    for name in vars(old):torch.testing.assert_close(getattr(new,name),getattr(old,name),rtol=1e-5,atol=1e-6)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
