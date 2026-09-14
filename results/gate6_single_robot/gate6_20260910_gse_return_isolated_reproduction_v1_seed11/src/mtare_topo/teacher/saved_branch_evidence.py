"""Typed view of existing teacher evidence; not new opening labels.

All IDs and directions are reference-only. This module has no student/model
API and does not infer hidden branches, apertures, or graph connectivity.
"""
from dataclasses import dataclass
import numpy as np


KINDS = ('entering_witness_ray_indices', 'interior_witness_ray_indices',
         'surface_entry_witness_ray_indices', 'surface_departure_witness_ray_indices')


def read_saved_junction_branches(record, provenance, axes, *, current_yaw_deg, ray_count):
    """Read the explicit junction prefix of a mixed archived reference record.

    Terminal references remain untouched and are not turned into empty branch
    targets. Missing prefix metadata is rejected, never inferred from geometry.
    """
    start = provenance.get('terminal_anchor_start')
    terminals = provenance.get('terminals')
    if (type(start) is not int or start < 0 or not isinstance(terminals, list)
            or start != len(provenance['anchors'])
            or len(record['anchors']) != start + len(terminals)):
        raise ValueError('explicit aligned junction/terminal partition required')
    junction_record = dict(record, anchors=record['anchors'][:start])
    return read_saved_branches(junction_record, provenance, axes,
                               current_yaw_deg=current_yaw_deg, ray_count=ray_count)


@dataclass(frozen=True)
class SavedBranchEvidence:
    node_id_teacher_only: str
    interface_id_teacher_only: int
    source_id_teacher_only: str
    anchor_position_current_sensor_m: tuple[float, float, float]
    reference_direction_current_sensor: tuple[float, float, float]
    witness_ray_indices: tuple[int, ...]
    witness_by_kind: tuple[tuple[str, tuple[int, ...]], ...]
    supported: bool
    aperture_position_m: None = None
    width_m: None = None
    height_m: None = None
    physical_reachable: None = None
    training_eligible: bool = False


def read_saved_branches(record, provenance, axes, *, current_yaw_deg, ray_count):
    if (record['coordinate_frame'] != 'current_sensor_m'
            or not np.isfinite(current_yaw_deg) or type(ray_count) is not int or ray_count <= 0):
        raise ValueError('explicit source coordinate frame and ray population required')
    if len(record['anchors']) != len(provenance['anchors']):
        raise ValueError('anchor provenance alignment differs')
    lookup={}
    for axis in axes:
        key=axis['interface_id_teacher_only'];direction=np.asarray(axis['inward_direction'],dtype=float)
        if (type(key) is not int or key < 0 or key in lookup or direction.shape != (3,)
                or not np.isfinite(direction).all() or abs(np.linalg.norm(direction)-1)>1e-12):
            raise ValueError('unique interface and original unit axis required')
        lookup[key]=axis
    angle=np.deg2rad(current_yaw_deg);c,s=np.cos(angle),np.sin(angle)
    rotation=np.array([[c,s,0],[-s,c,0],[0,0,1]])
    def rays(values):
        if (not isinstance(values,list) or any(type(i) is not int or not 0<=i<ray_count for i in values)
                or values != sorted(set(values))):
            raise ValueError('ordered unique bound witness indices required')
        return tuple(values)
    result=[];seen=set()
    for anchor, evidence in zip(record['anchors'],provenance['anchors'],strict=True):
        position=np.asarray(anchor['position_m'],dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError('finite saved anchor position required')
        ids=evidence['interface_ids'];all_rays=evidence['witness_ray_indices']
        if len(ids)!=len(all_rays) or any(len(evidence[k])!=len(ids) for k in KINDS if k in evidence):
            raise ValueError('branch evidence arrays are not aligned')
        for i,key in enumerate(ids):
            if key in seen or key not in lookup or lookup[key]['node_id_teacher_only']!=evidence['node_id_teacher_only']:
                raise ValueError('interface assigned to wrong or multiple anchors')
            seen.add(key);axis=lookup[key];witness=rays(all_rays[i]);by_kind=[]
            for kind in KINDS:
                if kind in evidence:
                    values=rays(evidence[kind][i])
                    if not set(values).issubset(witness):raise ValueError('witness kind outside combined evidence')
                    by_kind.append((kind,values))
            direction=rotation@np.asarray(axis['inward_direction'])
            result.append(SavedBranchEvidence(evidence['node_id_teacher_only'],key,
                axis['source_key_teacher_only'],tuple(position),tuple(direction),witness,
                tuple(by_kind),bool(witness)))
    return tuple(result)
