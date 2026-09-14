"""Reuse source-cap evidence; record branch source presence without label promotion."""
from collections import Counter
import numpy as np
from .gse_construction_paths_v3 import construction_incident_paths
from .gse_terminal_diagnostic_v2 import diagnose_observation as terminal_diagnostic


def branch_source_presence(groups,book,codes,center):
    counts=np.bincount(np.asarray(codes).reshape(-1),minlength=len(book['source_sets']))
    source_counts=Counter()
    for code,sources in enumerate(book['source_sets']):
        if len(sources)==1:
            source_counts[book['primitive_ids'][sources[0]]]+=int(counts[code])
    rows=[]
    for group in groups:
        if len(group['paths'])<3:continue
        distance=float(np.linalg.norm(np.asarray(group['anchor_world_m'])-center))
        if distance>=10.:continue
        repeats=Counter(p['endpoint_key'][0] for p in group['paths'])
        rows.append(dict(node_id_teacher_only=group['node_id_teacher_only'],distance_m=distance,
            branches=[dict(endpoint_key=list(p['endpoint_key']),
                uniquely_owned_return_count=source_counts[p['endpoint_key'][0]],
                source_shared_by_incident_arcs=repeats[p['endpoint_key'][0]]>1)
                for p in group['paths']],anchor_label_qualified=False,
            limitation='Source presence anywhere in five scans is not local branch visibility or connection evidence.'))
    return rows


def diagnose_observation(bundle,*,range_error_bound_m):
    output=terminal_diagnostic(bundle,range_error_bound_m=range_error_bound_m)
    output['junction_source_presence']=branch_source_presence(
        construction_incident_paths(bundle['construction_teacher_only']),bundle['codebook_teacher_only'],
        bundle['sensor_teacher_only']['primitive_membership_code'],bundle['sensor_teacher_only']['sensor_xyz_m'][-1])
    return output
