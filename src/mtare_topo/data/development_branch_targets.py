"""Loss-side adapter for archived junction/branch evidence, never model input.

Construction-reference completeness is not physical-opening completeness.
All file hashes and yaw-array provenance are authenticated by the caller.
"""
import numpy as np
import torch
from .gse_structure_review_v1 import canonical_sha
from mtare_topo.teacher.gse_construction_paths_v3 import construction_incident_paths
from mtare_topo.teacher.gse_directed_interface_binding_v1 import bind_axes
from mtare_topo.teacher.saved_branch_evidence import read_saved_junction_branches
from mtare_topo.representation.anchor_branch_loss import PartialAnchorBranches


def archived_junction_targets(produced, raw_interfaces, construction, *, current_yaw_deg,
                              expected_binding, expected_record_sha256,
                              dtype=torch.float32, device='cpu'):
    record=produced['record'];provenance=produced['teacher_provenance']
    if (produced['source_binding']!=expected_binding
            or canonical_sha(construction)!=expected_binding['construction_sha256']
            or produced['target_record_sha256']!=expected_record_sha256
            or canonical_sha(record)!=expected_record_sha256
            or record['source_frame_indices']!=expected_binding['source']['frame_rows']
            or any(raw_interfaces['source'][k]!=v for k,v in expected_binding['source'].items())):
        raise ValueError('archived target/source/construction binding mismatch')
    groups=construction_incident_paths(construction)
    lookup={g['node_id_teacher_only']:g for g in groups}
    interfaces=raw_interfaces['interfaces_teacher_only']
    axes=bind_axes(groups,interfaces)
    branches=read_saved_junction_branches(record,provenance,axes,current_yaw_deg=current_yaw_deg,ray_count=57600)
    count=provenance['terminal_anchor_start']
    if count>32:raise ValueError('anchor capacity exceeded; no truncation')
    positions=[];directions=[];complete=[];missing=[];seen=set();reference_count=0
    for i,evidence in enumerate(provenance['anchors']):
        node=evidence['node_id_teacher_only']
        if node in seen or node not in lookup:raise ValueError('duplicate or missing reference node')
        seen.add(node);paths=lookup[node]['paths']
        if len(paths)<3:raise ValueError('junction target is not a construction junction')
        expected_endpoints={tuple(p['endpoint_key']) for p in paths}
        node_interfaces=[v for v in interfaces if v['node_id_teacher_only']==node]
        endpoints=[tuple(v['endpoint_key_teacher_only']) for v in node_interfaces]
        # A restricted interface list must not manufacture completeness.
        if len(endpoints)!=len(set(endpoints)) or set(endpoints)!=expected_endpoints:
            raise ValueError('full construction incidence inventory required')
        all_ids={v['interface_id_teacher_only'] for v in node_interfaces}
        observed=[b for b in branches if b.node_id_teacher_only==node and b.supported]
        if len(observed)>64:raise ValueError('branch capacity exceeded; no truncation')
        known_ids={b.interface_id_teacher_only for b in observed}
        if not known_ids.issubset(all_ids):raise ValueError('foreign observed branch')
        positions.append(record['anchors'][i]['position_m'])
        directions.append(torch.tensor([b.reference_direction_current_sensor for b in observed],dtype=dtype,device=device).reshape(-1,3))
        absent=tuple(sorted(all_ids-known_ids));missing.append(absent);complete.append(not absent)
        reference_count+=len(all_ids)
    xyz=np.asarray(positions,dtype=float).reshape(-1,3)
    if not np.isfinite(xyz).all() or np.any(np.linalg.norm(xyz,axis=1)>10.+1e-5):
        raise ValueError('saved target outside observation domain')
    target=PartialAnchorBranches(torch.tensor(xyz,dtype=dtype,device=device),tuple(directions),False,tuple(complete))
    return dict(target=target,
                frozen_manifest=dict(source_binding=expected_binding,target_record_sha256=expected_record_sha256,
                                     target_reference_indices=tuple(range(count))),
                reference_branch_count=reference_count,observed_branch_count=sum(len(d) for d in directions),
                missing_interface_ids_teacher_only=tuple(missing),
                completeness_scope='frozen construction incidences only; not all physical openings',
                full_detection_eligible=False,aperture_supervised=False)
