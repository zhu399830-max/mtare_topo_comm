"""Analytic T-union counterexample for frozen evidence functions, not labels.

No real dataset, mesh approximation, training, or new teacher policy is used.
Only 15 physical rays are valid; remaining slots pad the historical interface.
"""
import hashlib
import importlib
import json
import numpy as np
from teacher_snapshot_probe_v1 import load_teacher, ZIP, SHA


def interval(origin, direction, lower, upper):
    lo, hi = -np.inf, np.inf
    for o, d, a, b in zip(origin, direction, lower, upper):
        if abs(d) < 1e-14:
            if not a <= o <= b:
                return None
        else:
            p, q = sorted(((a-o)/d, (b-o)/d))
            lo, hi = max(lo, p), min(hi, q)
    return (lo, hi) if lo <= hi else None


def first_union_exit(origin, direction, boxes):
    spans = sorted(s for a, b in boxes if (s := interval(origin, direction, a, b)) is not None)
    end = None
    for lo, hi in spans:
        if hi < 0:
            continue
        if end is None:
            assert lo <= 0 <= hi, 'origin must lie inside union'
            end = hi
        elif lo <= end + 1e-12:
            end = max(end, hi)
        else:
            break
    assert end is not None and np.isfinite(end) and end > 0
    return end


def main():
    load_teacher()
    interior_module = importlib.import_module('mtare_topo.teacher.gse_interior_branch_evidence_v1')
    directed_module = importlib.import_module('mtare_topo.teacher.gse_directed_interface_evidence_v1')
    for module in (interior_module, directed_module):
        assert module.__file__.startswith(str(ZIP)+'/src/')
    boxes = [(np.array(a, float), np.array(b, float)) for a, b in [
        ((-6,-1,-1),(0,1,1)), ((0,-1,-1),(6,1,1)), ((-1,0,-1),(1,6,1))]]
    axes = [dict(interface_id_teacher_only=i, node_id_teacher_only='synthetic_T',
                 source_key_teacher_only=str(i), inward_direction=d)
            for i,d in enumerate([[-1.,0.,0.],[1.,0.,0.],[0.,1.,0.]])]
    directions = np.tile([1.,0.,0.], (57600,1))
    valid = np.zeros(57600, bool)
    ranges = np.zeros(57600, np.float32)
    owners = [[] for _ in range(57600)]
    distances = np.empty((5,3))
    records, physical = [], []
    for frame, x in enumerate([1.2,1.3,1.4,1.5,1.6]):
        origin = np.array([x,0.,0.])
        for j,(a,b) in enumerate(boxes):
            q = np.abs(origin-(a+b)/2)-(b-a)/2
            distances[frame,j] = np.linalg.norm(np.maximum(q,0))+min(float(q.max()),0.)
        for branch, target in enumerate([[-6.,0.,0.],[6.,0.,0.],[-1.,3.,0.]]):
            ray = frame*11520+branch
            direction = np.array(target)-origin
            direction /= np.linalg.norm(direction)
            t = first_union_exit(origin, direction, boxes)
            hit = origin+t*direction
            assert np.allclose(hit,target,atol=1e-10), 'target is occluded'
            source = [str(j) for j,(a,b) in enumerate(boxes)
                      if np.all(hit >= a-1e-10) and np.all(hit <= b+1e-10)]
            assert source == [str(branch)], 'first-return ownership must be unique'
            directions[ray],valid[ray],ranges[ray],owners[ray] = direction,True,t,source
            # Construction caps: west/east at x=0, north at y=0.
            for j in range(3):
                axis = 0 if j < 2 else 1
                if abs(direction[axis]) < 1e-14:
                    continue
                cap_t = -origin[axis]/direction[axis]
                point = origin+cap_t*direction
                if abs(point[1-axis]) <= 1 and abs(point[2]) <= 1:
                    records.append(dict(ray_index=ray,interface_id_teacher_only=j,
                                        t=float(cap_t),inside_roi=True))
            physical.append(dict(frame=frame, origin=origin.tolist(), first_return=hit.tolist(),
                                 visible_branch=branch, distance_m=float(t)))
    entering = directed_module.directed_evidence(records,axes,directions=directions,
        first_return=ranges,valid=valid,return_sources=owners)
    interior = interior_module.interior_branch_evidence(axes,source_ids=['0','1','2'],
        origin_operand_distance=distances,directions=directions,valid=valid,return_sources=owners)
    counts = [len(entering['interfaces'][i]['entering_ray_indices']) for i in range(3)]
    inside_counts = [len(interior['interface_ray_indices'][i]) for i in range(3)]
    assert counts == [5,0,0] and inside_counts == [0,0,0]
    print(json.dumps(dict(status='SYNTHETIC_SURFACE_VISIBILITY_NOT_EQUIVALENT_TO_CAP_EVIDENCE',
        archive_sha256=SHA, script_sha256=hashlib.sha256(open(__file__,'rb').read()).hexdigest(),
        visible_first_returns_per_branch=[5,5,5], entering_per_branch=counts,
        interior_per_branch=inside_counts, origin_operand_distances=distances.tolist(),
        rays=physical, real_data_reads=0, full_teacher_calls=0, training_updates=0,
        limitation='Analytic surface visibility only; not full junction qualification, structural identifiability, or ground traversability.'), indent=2))


if __name__ == '__main__':
    main()
