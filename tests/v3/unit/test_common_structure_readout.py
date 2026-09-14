import numpy as np
import torch
from dataclasses import replace
from mtare_topo.representation.gse_common_structure_readout import CommonStructureReadout,section_positions
from mtare_topo.representation.gse_common_observation import assemble_common_observation
from mtare_topo.representation.gse_dual_path_encoder_adapter_v1 import CompactFrozenDualPathFeatures


def observation():
    xyz=torch.zeros(1,57600,3);xyz[0,0]=torch.tensor([20.,0,0]);xyz[0,4]=torch.tensor([0.,20,0])
    valid=torch.zeros(1,57600,dtype=torch.bool);valid[0,[0,4]]=True
    idx=torch.from_numpy(np.repeat(np.arange(5),11520)*180+np.tile(np.arange(720)//4,80))[None]
    return assemble_common_observation(CompactFrozenDualPathFeatures(xyz,torch.zeros(1,900,128),valid,idx),np.zeros((5,3)))


def test_no_local_surface_still_produces_observation_seeded_hypotheses():
    torch.manual_seed(0);m=CommonStructureReadout('A');obs=observation();r=m(obs)
    assert len(obs.surface_patches.centers_m)==0
    assert len(r.prediction.structure_positions_m)==2
    assert set(r.section_query_ray_indices)=={0,4}
    assert torch.all(torch.linalg.vector_norm(r.prediction.structure_positions_m,dim=-1)<=10.000001)
    assert torch.allclose(torch.linalg.vector_norm(r.prediction.window_section_positions_m,dim=-1),torch.full((2,),10.))
    assert not r.prediction.full_detection_qualified


def test_geometry_change_changes_initial_positions_without_labels():
    torch.manual_seed(0);m=CommonStructureReadout('A');obs=observation()
    a=m(obs).prediction.structure_positions_m
    rays=obs.local_rays
    rotated=rays.end_xyz_m[:,[1,0,2]]*np.array([1,-1,1])
    altered=replace(obs,local_rays=replace(rays,end_xyz_m=rotated))
    b=m(altered).prediction.structure_positions_m
    assert not torch.equal(a,b)


def test_position_and_relation_outputs_have_gradients():
    torch.manual_seed(0);m=CommonStructureReadout('A');p=m(observation()).prediction
    loss=p.structure_positions_m.sum()+p.window_section_positions_m.sum()+p.section_structure_logits.sum()
    loss.backward()
    assert m.structure_offset.weight.grad.abs().sum()>0
    assert m.section_offset.weight.grad.abs().sum()>0
    assert m.section_relation.weight.grad.abs().sum()>0


def test_section_domain_can_reverse_seed_without_large_offset():
    seed=torch.tensor([[1.,0,0]]);raw=torch.tensor([[-1.,0,0]])
    assert torch.allclose(section_positions(seed,raw),torch.tensor([[-10.,0,0]]))
