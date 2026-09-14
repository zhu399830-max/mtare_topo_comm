from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from mtare_topo.integration.geometry_structure_trace import geometry_structure_record
from mtare_topo.semantics.observed_axis_structure import structure_from_observed_primitives
from mtare_topo.semantics.supported_primitive_fit import SupportedPrimitiveFit


def observation(z=0.):
    def primitive(a, b, ray):
        return SimpleNamespace(fit=SupportedPrimitiveFit('synthetic_fit', (ray,),
            axis_controls_m=(tuple(a), tuple((np.asarray(a)+b)/2), tuple(b))))
    return structure_from_observed_primitives([
        primitive([-3,0,0], [3,0,0], 0), primitive([0,0,z], [0,3,z], 11520)],
        max_residual_m=.01, min_crossing_sine=.1, endpoint_tolerance_m=1e-8,
        maximum_candidates=32)


def record(obs, matrix=None, rays=(0, 11520)):
    return geometry_structure_record(obs, source_frame_keys=('a','b','c','d','e'),
        source_ray_indices=np.asarray(rays), sensor_to_world=np.eye(4) if matrix is None else matrix,
        coordinate_frame='odom', timestamp=4.)


def test_composition_sources_and_3d_transform():
    matrix=np.eye(4); matrix[:3,3]=[5,6,7]
    matrix[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]]
    result=record(observation(), matrix)
    assert result['structures'][0]['position_world_m'] == [5.,6.,7.]
    assert [0.,-1.,0.] in result['structures'][0]['directions_world']
    assert result['primitives'][1]['source_rays'] == [dict(frame_key='b',ray_index=0)]
    assert not result['physical_openings_confirmed']
    assert result['traversed_edges'] == []


def test_stacked_geometry_changes_structure_not_just_color():
    assert len(record(observation())['structures']) == 1
    assert record(observation(2.))['structures'] == []


def test_no_fabricated_source_support():
    with pytest.raises(ValueError, match='absent'):
        record(observation(), rays=(0,))


def test_invalid_pose_rejected():
    matrix=np.eye(4);matrix[2,2]=-1
    with pytest.raises(ValueError, match='rigid'):
        record(observation(),matrix)
