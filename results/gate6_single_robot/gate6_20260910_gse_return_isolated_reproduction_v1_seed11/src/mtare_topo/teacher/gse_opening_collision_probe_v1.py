"""Explain duplicate reference positions without merging/filtering proposals."""
from collections import defaultdict
import numpy as np
from .gse_window_opening_diagnostic_v1 import diagnose_observation
from .gse_mesh_sections_v1 import mesh_section
from .csg_mesh_provenance import mesh_swept_superellipse
from mtare_topo.data.primitive_relation_materialization import load_p1a_realized_construction


def inspect_collision(bundle):
    diagnostic=diagnose_observation(bundle)
    groups=defaultdict(list)
    for i,row in enumerate(diagnostic['proposals']):
        if row['exclusive_outward_crossing_ray_indices'] and row['surface_return_ray_indices']:
            groups[tuple(row['reference_position_m'])].append(i)
    duplicates=[v for v in groups.values() if len(v)>1]
    _,primitives=load_p1a_realized_construction(bundle['construction_teacher_only'])
    lookup={p.primitive_id:p for p in primitives};sections=[];seen=set()
    for indices in duplicates:
        for i in indices:
            row=diagnostic['proposals'][i]
            key=(row['primitive_id_teacher_only'],row['reference_arc_m'])
            if key in seen:continue
            seen.add(key);mesh=mesh_swept_superellipse(lookup[key[0]],axial_spacing_m=.05,angular_segments=64)
            section=mesh_section(mesh.vertices_xyz_m,mesh.triangle_vertex_indices,
                center_m=row['reference_position_m'],normal=row['reference_direction'])
            sections.append(dict(source_key=list(key),reference_position_m=row['reference_position_m'],
                loops_m=[p.tolist() for p in section.loops_m],
                loop_vertex_means_m=[np.mean(p,axis=0).tolist() for p in section.loops_m]))
    return dict(source=bundle['source'],diagnostic=diagnostic,duplicate_supported_proposal_indices=duplicates,
                sections=sections,labels_generated=0,training_eligible=False)


def inspect_corrected_collision(bundle):
    from .gse_junction_interface_diagnostic_v1 import diagnose_observation as interfaces
    from .gse_joint_reference_targets_v4 import produce_joint_reference_targets
    before=inspect_collision(bundle)
    raw=interfaces(bundle)
    after=produce_joint_reference_targets(bundle,raw)
    return dict(before=before,raw_interfaces=raw,produced_targets=after,
        duplicate_supported_proposal_indices=before['duplicate_supported_proposal_indices'],
        corrected_openings=len(after['record']['openings']),
        nonowning_contours=sum(r['reason']=='NO_UNIQUE_AXIS_REFERENCE_CONTOUR' for r in after['unknown_candidates']['openings']),
        labels_generated=0,training_eligible=False)
