#!/usr/bin/env python3
"""Dependency-light analytic checks for the Open3D CSG provenance backend."""

import json
import numpy as np

from _bootstrap import PROJECT_ROOT  # noqa: F401
from mtare_topo.teacher.csg_mesh_provenance import CSGMeshProvenanceRaycaster, mesh_swept_superellipse
from mtare_topo.teacher.swept_superellipse_field import SweptSuperellipsePrimitive, SweptSuperellipseProvenanceField


def primitive(identity, points, axes=((1.,1.),(1.,1.)), exponent=(2.,2.)):
    return SweptSuperellipsePrimitive(identity, np.asarray(points,float), axes, exponent)


def compare(primitives, origin, direction):
    field=SweptSuperellipseProvenanceField(primitives,spacing_m=.01)
    origins=np.asarray([origin],float); directions=np.asarray([direction],float)
    inside=field.operand_signed_distances(origins)<=0
    raycaster=CSGMeshProvenanceRaycaster(
        [mesh_swept_superellipse(value,axial_spacing_m=.05,angular_segments=64) for value in primitives],
        operand_signed_distances=field.operand_signed_distances_sparse,
    )
    mesh=raycaster.ray_exit_hits(origins,directions,inside,maximum_m=20)[0]
    implicit=field.ray_exit_hits(origins,directions,maximum_m=20)[0]
    if mesh is None or implicit is None: raise RuntimeError("analytic ray missed")
    return mesh,implicit


def main():
    cases=[
        ("ellipse",[primitive("ellipse",[[-2,0,0],[2,0,0]],((1.2,.8),(1.2,.8)))],[0,0,0],[0,1,0]),
        ("t_union",[primitive("trunk",[[-3,0,0],[3,0,0]],((.5,.5),(.5,.5))),primitive("branch",[[0,0,0],[0,3,0]],((.5,.5),(.5,.5)))],[0,0,0],[0,1,0]),
        ("stacked",[primitive("lower",[[-2,0,0],[2,0,0]],((1,.8),(1,.8)),(8,8)),primitive("upper",[[-2,0,3],[2,0,3]],((1,.8),(1,.8)),(8,8))],[0,0,0],[0,0,1]),
        ("ambiguity",[primitive("one",[[-2,0,0],[2,0,0]],exponent=(8,8)),primitive("two",[[-2,0,0],[2,0,0]],exponent=(8,8))],[0,0,0],[0,1,0]),
    ]
    rows=[]
    for name,values,origin,direction in cases:
        mesh,implicit=compare(values,origin,direction)
        error=abs(mesh.distance_m-implicit.distance_m)
        if error>.03 or mesh.source_primitive_ids!=implicit.source_primitive_ids: raise RuntimeError(f"{name} mismatch")
        rows.append({"case":name,"range_error_m":error,"source_primitive_ids":list(mesh.source_primitive_ids),"provenance_unique":mesh.provenance_unique})
    if rows[-1]["provenance_unique"]: raise RuntimeError("ambiguity was collapsed")
    print(json.dumps({"status":"PASS_CSG_MESH_PROVENANCE_ANALYTIC_CONTRACT","cases":rows},sort_keys=True))


if __name__=="__main__": main()
