from dataclasses import replace
import numpy as np
import pytest
from tests.v3.unit.test_mesh_interval_exit import mesh
from mtare_topo.teacher.csg_interval_raycaster_v2 import verified_interval_hit


def run(first_end,second_start,maximum=50):
    meshes=[replace(mesh([(-1.,first_end)]),primitive_id='main'),replace(mesh([(second_start,30.)]),primitive_id='branch')]
    ts=sorted([first_end,second_start,30.]);sources=[0 if t==first_end else 1 for t in ts]
    return verified_interval_hit(meshes,[0,.13,.17],[1,0,0],[True,False],ts,sources,maximum_m=maximum)


@pytest.mark.parametrize('gap',[.0001,.006755640983918898,.02])
def test_any_resolved_positive_gap_stops_at_first_exit(gap):
    hit=run(1.998137966459527,1.998137966459527+gap)
    assert hit.distance_m==1.998137966459527 and hit.source_primitive_ids==('main',)


def test_actual_overlap_continues_to_far_wall():
    assert run(2.,1.5).distance_m==30.


@pytest.mark.parametrize('second_start',[2.0001,2.006755640983918898,2.02,1.5])
def test_native_open3d_first_exit(second_start):
    pytest.importorskip('open3d')
    from mtare_topo.teacher.csg_interval_raycaster_v2 import CSGIntervalRaycasterV2
    from mtare_topo.teacher.csg_mesh_provenance import CSGMeshProvenanceRaycaster
    meshes=[replace(mesh([(-1.,2.)]),primitive_id='main'),
            replace(mesh([(second_start,30.)]),primitive_id='branch')]
    args=([[0.,.13,.17]],[[1.,0.,0.]],[[True,False]])
    new=CSGIntervalRaycasterV2(meshes).ray_exit_hits(*args)[0]
    old=CSGMeshProvenanceRaycaster(meshes).ray_exit_hits(*args)[0]
    expected=30. if second_start<2. else 2.
    assert new is not None
    assert new.distance_m==pytest.approx(expected,abs=1e-5)
    if 2.<second_start<2.01:
        assert old.distance_m==pytest.approx(30.,abs=1e-5)
    else:
        assert old.distance_m==pytest.approx(expected,abs=1e-5)


def test_out_of_range_is_not_replaced_by_legacy_return():
    assert run(2.,1.5,maximum=10.) is None


def test_bad_source_and_missing_intersections_reject():
    m=mesh([(-1,1)])
    with pytest.raises(ValueError):verified_interval_hit([m],[0,0,0],[1,0,0],[True],[1],[2])
    assert verified_interval_hit([m],[0,0,0],[1,0,0],[True],[],[]) is None


def test_versioned_batch_reducer_with_explicit_synthetic_intersection_io(monkeypatch):
    """Exercise the real V2 batch path; not a native Open3D rendering test."""
    import sys
    from types import SimpleNamespace
    from mtare_topo.teacher.csg_interval_raycaster_v2 import CSGIntervalRaycasterV2
    monkeypatch.setitem(sys.modules, 'open3d', SimpleNamespace(core=SimpleNamespace(Tensor=lambda x:x)))
    caster=CSGIntervalRaycasterV2.__new__(CSGIntervalRaycasterV2)
    caster.meshes=[replace(mesh([(-1.,2.)]),primitive_id='main'),
                   replace(mesh([(2.00675564,30.)]),primitive_id='branch')]
    caster.geometry_to_operand={4:0,9:1}
    arrays={'ray_splits':np.array([0,3]),
            't_hit':np.array([2.,2.00675564,30.],dtype=np.float32),
            'geometry_ids':np.array([4,9,9])}
    caster.scene=SimpleNamespace(list_intersections=lambda rays:{
        k:SimpleNamespace(numpy=lambda value=v:value) for k,v in arrays.items()})
    hit,=caster.ray_exit_hits([[0,.13,.17]],[[2,0,0]],[[True,False]])
    assert hit.distance_m==2.
    assert hit.xyz_m==(2.,.13,.17)
    assert hit.source_primitive_ids==('main',)
