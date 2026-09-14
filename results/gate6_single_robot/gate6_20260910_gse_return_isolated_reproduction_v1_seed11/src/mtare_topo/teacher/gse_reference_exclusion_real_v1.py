"""Evaluate fixed queries against authenticated, archived grids (no recasting)."""
import io
import itertools
import numpy as np
from mtare_topo.governance_surface_material import read_pinned
from mtare_topo.data.gse_structure_review_v1 import canonical_sha
from mtare_topo.representation.gse_surface_ray_evidence_v1 import ObservedRayGrid, _numerical_bound
from .gse_reference_exclusion_binding_v1 import bound_reference_exclusion

def evaluate_archived(root,row,bundle,policy,opened):
    raw=read_pinned(root,row['numeric_path'],row['numeric_sha256'])
    opened[row['numeric_path']]=row['numeric_sha256']
    with np.load(io.BytesIO(raw),allow_pickle=False) as archive:
        valid=archive['first_return_valid']
        origins=archive['ray_origins_current_sensor_m']
        points=archive['points_current_sensor_m']
        # Scalar ambiguity count was not persisted by the original material
        # exporter. -1 means unavailable, never measured zero. Query validation
        # uses authenticated state/bits/bound only, not this diagnostic count.
        grid=ObservedRayGrid(archive['grid_state'],archive['grid_free_frame_bits'],
            archive['grid_occupied_frame_bits'],row['grid_source_sha256'],
            row['grid_content_sha256'],_numerical_bound(origins[valid],points[valid]),
            int(valid.sum()),int((~valid).sum()),-1)
    q=np.asarray(list(itertools.product(policy['coordinates_per_axis_m'],repeat=3)),dtype=np.float64)
    source=bundle['source']
    if any(source[k]!=row['source'][k] for k in ('task','source_sequence_id','frame_rows')):
        raise ValueError('scope observation mismatch')
    # Construction bytes were already verified against the frozen source plan
    # by SurfaceTeacherReader; its complete document, never positive subsets.
    binding=dict(source={k:source[k] for k in ('task','source_sequence_id','frame_rows')},
        construction_sha256=canonical_sha(bundle['construction_teacher_only']))
    result=bound_reference_exclusion(bundle,grid,q,expected_binding=binding,
        matching_radius_m=policy['matching_radius_m'])
    result.update(query_xyz_m=q.tolist(),archived_ambiguous_ray_count=None,
        real_semantic_labels=0,human_reviewed=False)
    return result
