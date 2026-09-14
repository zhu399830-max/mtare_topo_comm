"""Bind window axis references to their unique source-section contour."""
from .gse_window_opening_diagnostic_v1 import _diagnose_observation
from .gse_window_opening_proposals_v1 import propose_window_openings
from .gse_mesh_sections_v1 import mesh_section
from .gse_section_reference_ownership_v1 import section_reference_owner


def propose_bound_window_openings(vertices_m,triangles,**kwargs):
    result=propose_window_openings(vertices_m,triangles,**kwargs)
    section=mesh_section(vertices_m,triangles,center_m=kwargs['section_center_m'],normal=kwargs['outward_direction'])
    ownership=section_reference_owner(section.loops_m,reference_m=kwargs['section_center_m'],normal=kwargs['outward_direction'])
    if len(section.loops_m)!=len(result['proposals']):
        raise ValueError('section ordering/population changed')
    for i,row in enumerate(result['proposals']):
        row['reference_contour_index']=i
        row['axis_reference_owned']=i==ownership['owner_loop_index']
        row['axis_reference_ownership']=ownership
        # Keep the complete original candidate and all witness lists. Positive
        # target eligibility is decided separately, not by deleting contours.
    return result


def diagnose_observation(bundle):
    return _diagnose_observation(bundle,propose_bound_window_openings)
