"""Demonstrate why return-storage ULP alone is not a caster error bound."""
import numpy as np
from mtare_topo.teacher.gse_cap_return_evidence_v1 import cap_return_evidence
from mtare_topo.teacher.gse_portal_ray_evidence import CausalRaySegments


def test_float32_mesh_rounding_can_exceed_return_storage_ulp():
    vertices=np.array([[1000.00003,0.,0.],[1000.00003,1.,0.],[1000.00003,0.,1.]])
    stored=vertices.astype(np.float32).astype(np.float64)
    rays=CausalRaySegments(np.array([[999.,.2,.2]]),np.array([[1.,0.,0.]]),
                          np.array([1.],dtype=np.float32),np.array([True]),np.array([0]),0,0.)
    kwargs=dict(cap_face_indices=np.array([0]),rays=rays)
    original=cap_return_evidence(vertices,np.array([[0,1,2]]),**kwargs)
    quantized=cap_return_evidence(stored,np.array([[0,1,2]]),**kwargs)
    assert not original['cap_surface_observed']
    assert quantized['cap_surface_observed']
    assert abs(vertices[0,0]-stored[0,0])>100*np.spacing(np.float32(1.))
    # This is a counterexample, not permission to change actual labels.
    assert original['terminal_label'] is quantized['terminal_label'] is None
