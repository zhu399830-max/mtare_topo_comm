import pytest
import torch
from mtare_topo.representation.branch_relation_learning import observed_ray_pairs

def stable(ids,policy):
    x=torch.tensor(ids,dtype=torch.long)
    return {tuple(sorted((ids[a],ids[b]))) for a,b in observed_ray_pairs(x,policy=policy).tolist()}

def test_opposite_directions_and_missing_intermediate_rows():
    ids=[0,1,360,361,2*720,2*720+1]
    old=stable(ids,'local_v1');new=stable(ids,'dyadic_v2')
    assert (0,360) not in old and (0,360) in new
    assert (0,1440) not in old and (0,1440) in new
    assert old<=new

def test_new_policy_is_order_independent_and_does_not_invent_rays():
    ids=[0,719,720,1440,360,512]
    pairs=stable(ids,'dyadic_v2')
    assert pairs==stable(ids[::-1],'dyadic_v2')
    assert all(a in ids and b in ids and a!=b for a,b in pairs)

def test_azimuth_seam_not_elevation_wrap():
    pairs=stable([0,719,15*720],'dyadic_v2')
    assert (0,719) in pairs and (0,15*720) not in pairs

def test_invalid_policy_and_identity_rejected():
    with pytest.raises(ValueError):stable([0,0],'dyadic_v2')
    with pytest.raises(ValueError):stable([11520],'dyadic_v2')
    with pytest.raises(ValueError):stable([0,1],'made_up')
