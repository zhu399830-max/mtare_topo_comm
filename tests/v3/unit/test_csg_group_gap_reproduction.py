"""Execute real event reducer with synthetic intersection IO (not raycasting).

Open3D Tensor/scene are explicit fixtures only. No mesh or scan is generated.
"""
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from mtare_topo.teacher.csg_mesh_provenance import CSGMeshProvenanceRaycaster


class Array:
    def __init__(self,value):self.value=np.asarray(value)
    def numpy(self):return self.value


def reduce_events(monkeypatch,entry):
    monkeypatch.setitem(sys.modules,'open3d',SimpleNamespace(core=SimpleNamespace(Tensor=lambda x:x)))
    caster=CSGMeshProvenanceRaycaster.__new__(CSGMeshProvenanceRaycaster)
    caster.distance_group_tolerance_m=.01
    caster.operand_signed_distances=None;caster.union_signed_distance=None
    caster.require_unique_qualified_candidate=False;caster.rescue_missing_with_interval_winding=False
    caster.primitive_ids=('main','branch');caster.geometry_to_operand={0:0,1:1}
    caster.meshes=(SimpleNamespace(triangle_normals=np.array([[1.,0,0]])),
                   SimpleNamespace(triangle_normals=np.array([[-1.,0,0],[1.,0,0]])))
    raw={k:Array(v) for k,v in dict(ray_splits=[0,3],t_hit=np.array([1.998137966459527,entry,30.04574649293542],dtype=np.float32),
        geometry_ids=[0,1,1],primitive_ids=[0,0,1]).items()}
    caster.scene=SimpleNamespace(list_intersections=lambda _:raw)
    return caster.ray_exit_hits(np.array([[0.,0,0]]),np.array([[1.,0,0]]),np.array([[True,False]]))[0]


def test_original_reducer_bridges_saved_positive_gap(monkeypatch):
    hit=reduce_events(monkeypatch,2.004893607443446)
    assert hit.distance_m==float(np.float32(30.04574649293542))
    assert hit.distance_m>1.998137966459527+28


def test_original_reducer_stops_when_gap_exceeds_grouping(monkeypatch):
    hit=reduce_events(monkeypatch,2.02)
    assert hit.distance_m==float(np.float32(1.998137966459527))
