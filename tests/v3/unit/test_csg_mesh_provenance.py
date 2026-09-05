import numpy as np
import pytest

from mtare_topo.teacher.csg_mesh_provenance import CSGMeshProvenanceRaycaster, mesh_swept_superellipse
from mtare_topo.teacher.swept_superellipse_field import SweptSuperellipsePrimitive, SweptSuperellipseProvenanceField


def _primitive(identity, points, axes=((1.,1.),(1.,1.)), exponent=(2.,2.)):
    return SweptSuperellipsePrimitive(identity, np.asarray(points,float), axes, exponent)


def _cast(primitives, origin, direction):
    field = SweptSuperellipseProvenanceField(primitives, spacing_m=.01)
    origin = np.asarray([origin],float); direction=np.asarray([direction],float)
    inside = field.operand_signed_distances(origin) <= 0
    raycaster = CSGMeshProvenanceRaycaster(
        [mesh_swept_superellipse(x, axial_spacing_m=.05, angular_segments=64) for x in primitives],
        operand_signed_distances=field.operand_signed_distances,
    )
    mesh_hit = raycaster.ray_exit_hits(origin,direction,inside,maximum_m=20)[0]
    field_hit = field.ray_exit_hits(origin,direction,maximum_m=20)[0]
    assert mesh_hit is not None and field_hit is not None
    return mesh_hit, field_hit


def test_csg_mesh_matches_ellipse_exit_and_identity():
    mesh, field = _cast([_primitive("ellipse",[[-2,0,0],[2,0,0]],((1.2,.8),(1.2,.8)))],[0,0,0],[0,1,0])
    assert mesh.distance_m == pytest.approx(field.distance_m,abs=.01)
    assert mesh.source_primitive_ids == field.source_primitive_ids == ("ellipse",)


def test_csg_mesh_skips_internal_t_branch_surface():
    primitives=[_primitive("trunk",[[-3,0,0],[3,0,0]],((.5,.5),(.5,.5))),_primitive("branch",[[0,0,0],[0,3,0]],((.5,.5),(.5,.5)))]
    mesh, field = _cast(primitives,[0,0,0],[0,1,0])
    assert mesh.distance_m == pytest.approx(field.distance_m,abs=.03)
    assert mesh.source_primitive_ids == field.source_primitive_ids == ("branch",)


def test_csg_mesh_stacked_tunnel_does_not_steal_lower_exit():
    primitives=[_primitive("lower",[[-2,0,0],[2,0,0]],((1,.8),(1,.8)),(8,8)),_primitive("upper",[[-2,0,3],[2,0,3]],((1,.8),(1,.8)),(8,8))]
    mesh, field = _cast(primitives,[0,0,0],[0,0,1])
    assert mesh.distance_m == pytest.approx(field.distance_m,abs=.02)
    assert mesh.source_primitive_ids == field.source_primitive_ids == ("lower",)


def test_csg_mesh_preserves_identical_operand_ambiguity():
    primitives=[_primitive("one",[[-2,0,0],[2,0,0]],exponent=(8,8)),_primitive("two",[[-2,0,0],[2,0,0]],exponent=(8,8))]
    mesh, field = _cast(primitives,[0,0,0],[0,1,0])
    assert mesh.distance_m == pytest.approx(field.distance_m,abs=.02)
    assert mesh.source_primitive_ids == field.source_primitive_ids == ("one","two")
    assert not mesh.provenance_unique
