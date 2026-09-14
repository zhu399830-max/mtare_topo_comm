"""Native mesh tests of existing lateral witnesses, not junction labels."""
from types import SimpleNamespace

import numpy as np
import open3d as o3d

from mtare_topo.teacher.gse_observed_operand_entries_v1 import observed_operand_entries


def caster_for_box(lower=(-1.,0.,-1.), upper=(1.,6.,1.)):
    extent = np.array(upper)-lower
    mesh = o3d.geometry.TriangleMesh.create_box(*extent)
    mesh.translate(lower)
    mesh.compute_triangle_normals()
    scene = o3d.t.geometry.RaycastingScene()
    geometry = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return SimpleNamespace(scene=scene, geometry_to_operand={geometry:0},
        primitive_ids=['north'], meshes=[SimpleNamespace(
            triangle_normals=np.asarray(mesh.triangle_normals))])


def query(caster, *, obstacle=False, owners=None):
    origins = np.array([[x,0.,0.] for x in [1.2,1.3,1.4,1.5,1.6]])
    targets = np.tile([-1.,3.,0.],(5,1))
    directions = targets-origins
    lengths = np.linalg.norm(directions,axis=1)
    directions /= lengths[:,None]
    packed = np.column_stack([origins,directions]).astype(np.float32)
    if obstacle:
        # Actual opaque plane x=1.1 lies before the north side at x=1.
        # Range follows its analytic first intersection, not the hidden wall.
        lengths = (1.1-origins[:,0])/directions[:,0]
    return observed_operand_entries(caster,packed,
        first_return=lengths.astype(np.float32),valid=np.ones(5,bool),
        return_sources=owners if owners is not None else [['north'] for _ in range(5)],
        center_m=origins[-1])


def test_original_side_entry_is_recovered_without_crossing_north_end_cap():
    result = query(caster_for_box())
    assert {e['ray_index'] for e in result['entries']} == set(range(5))
    for entry in result['entries']:
        x,y,z = entry['intersection_world_m']
        assert abs(x-1.) < 1e-6 and y>0 and z==0
    assert result['membership'] is None
    assert not result['physical_separation_certified']


def test_same_occluded_returns_do_not_reveal_changed_hidden_extent():
    # Same five observed rays, two different hidden north extents.
    a = query(caster_for_box(),obstacle=True,owners=[['occluder']]*5)
    b = query(caster_for_box(upper=(1.,9.,1.)),obstacle=True,owners=[['occluder']]*5)
    assert a == b and a['entries'] == []


def test_range_guard_rejects_hidden_intersection_even_with_wrong_owner():
    # Adversarial owner cannot override a nearer observed first return.
    assert query(caster_for_box(),obstacle=True)['entries'] == []


def test_multisource_return_is_not_forced_to_unique_branch():
    assert query(caster_for_box(),owners=[['north','other']]*5)['entries'] == []
