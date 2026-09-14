"""Shared observed geometry, not teacher labels or trained accuracy."""
from dataclasses import fields
import numpy as np
import torch

from mtare_topo.representation.gse_surface_observed_inputs_v1 import build_observed_surface_inputs
from mtare_topo.representation.gse_surface_relation_model_v1 import collate_surface_patches


def test_both_production_entrypoints_use_same_builder():
    from mtare_topo.data import gse_surface_training_join_v1 as training
    from mtare_topo.topology import gse_surface_prediction_binding_v1 as runtime
    assert training.build_observed_surface_inputs is build_observed_surface_inputs
    assert runtime.build_observed_surface_inputs is build_observed_surface_inputs


def test_observed_free_gap_replaces_old_all_unknown_input_deterministically():
    xyz=np.array([[.125,.125,.125],[2.125,.125,.125]],dtype=np.float32)
    origins=xyz[::-1].copy();mask=np.ones(2,dtype=bool);slots=np.array([0,4],dtype=np.int64)
    patches,batch,grid,gap=build_observed_surface_inputs(xyz,mask,slots,origins)
    old=collate_surface_patches([patches])
    assert gap.gap_defined[0,0]
    assert batch.relation[0,0,0,-3:].tolist()==[1.,0.,0.]
    assert old.relation[0,0,0,-3:].tolist()==[0.,0.,1.]
    again=build_observed_surface_inputs(xyz,mask,slots,origins)
    for field in fields(batch):
        assert torch.equal(getattr(batch,field.name),getattr(again[1],field.name))
    assert grid.content_sha256==again[2].content_sha256
    np.testing.assert_array_equal(patches.ray_origins_m,origins)


def test_invalid_returns_cannot_create_observed_space():
    xyz=np.array([[.125,.125,.125],[2.125,.125,.125]],dtype=np.float32)
    patches,batch,grid,gap=build_observed_surface_inputs(xyz,np.zeros(2,dtype=bool),
        np.array([0,4],dtype=np.int64),xyz[::-1].copy())
    assert not batch.valid.any()
    assert not grid.state.any()
    assert batch.relation[0,0,0,-3:].tolist()==[0.,0.,1.]


def test_training_join_matches_shared_inputs_without_using_teacher_grid():
    from test_gse_surface_training_join_v1 import args
    from mtare_topo.data.gse_surface_training_join_v1 import join_training_observation
    kwargs=args();example,context=join_training_observation(**kwargs)
    compact=example.compact;source=kwargs['source_input']
    origins=np.broadcast_to(source.translation_m[:,None,None,:],(5,16,720,3)).reshape(-1,3)
    _,expected,grid,_=build_observed_surface_inputs(compact.points_xyz_m[0].numpy(),
        compact.valid[0].numpy(),np.repeat(np.arange(5,dtype=np.int64),11520),origins)
    for field in fields(expected):
        assert torch.equal(getattr(example.patches,field.name),getattr(expected,field.name))
    assert example.header['student_grid_content_sha256']==grid.content_sha256
    assert context.grid is not grid
