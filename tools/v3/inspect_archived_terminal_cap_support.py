"""Read-only single archived fixture; no tolerance tuning or label export."""
import _bootstrap
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import numpy as np
from mtare_topo.data.gse_synthetic_fit_scope import compile_scope
from mtare_topo.data.cano_sensor_smoke import lidar_local_directions, world_directions
from mtare_topo.evaluation.gse_synthetic_straight_oracle import visible_straight_cap_rays


def inspect_row(root, row):
    data = {}
    for name, pin in [('input_path','input_sha256'), ('source_record_path','source_record_sha256')]:
        raw = (root/row[name]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != row[pin]:
            raise ValueError('sealed input drift')
        data[name] = raw
    record = json.loads(gzip.decompress(data['source_record_path']))
    case = record['case']
    if case['program']['type'] not in ('straight', 'terminal', 'visible_blocker'):
        return dict(case_id=row['case_id'], status='JUNCTION_SUPPORT_UNDEFINED',
            point_structure_support_available=False,
            reason='existing central-star oracle gives anchor/opening positions, not point support')
    indices = visible_straight_cap_rays(case)
    local = lidar_local_directions().reshape(-1,3).astype(np.float64)
    directions = np.array([world_directions(local,float(y)) for y in case['yaw_deg']])
    unit = directions / np.linalg.norm(directions,axis=-1,keepdims=True)
    poses = np.array(case['poses_world_m'])
    rows = []
    with np.load(io.BytesIO(data['input_path']),allow_pickle=False) as a:
        ranges = a['ranges_m'].reshape(-1); valid = a['valid_mask'].reshape(-1)
        for side, selected in indices.items():
            ix = np.array(selected,dtype=int); slots = ix//11520; beams = ix%11520
            out = dict(side=side, count=len(ix), per_frame=np.bincount(slots,minlength=5).tolist())
            if len(ix):
                if len(np.unique(ix)) != len(ix) or not valid[ix].all():
                    raise ValueError('support indices duplicate or invalid')
                cap = case['program']['edges'][0]['points'][side][0]
                out['numeric_paths'] = {}
                for name, p, d in [('analytic',poses,directions), ('unit64',poses,unit),
                                    ('raycaster_packed32',poses.astype(np.float32).astype(float),unit.astype(np.float32).astype(float))]:
                    expected = (cap-p[slots,0])/d[slots,beams,0]
                    error = np.abs(ranges[ix].astype(float)-expected)
                    out['numeric_paths'][name] = dict(max_range_error_m=float(error.max()),
                        max_float32_ulp_error=float((error/np.spacing(ranges[ix])).max()),
                        exact_float32_matches=int((ranges[ix]==expected.astype(np.float32)).sum()))
                out['source_codes'] = np.unique(a['primitive_membership_code'].reshape(-1)[ix]).tolist()
            rows.append(out)
    return dict(case_id=row['case_id'], sides=rows, new_labels=0, renders=0,
        tolerance_selected=False, general_teacher_qualified=False)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--population', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    scope = compile_scope(root)
    selected = scope['observations'] if args.population else [next(r for r in scope['observations'] if r['case_id']=='terminal__circle__view2')]
    rows = [inspect_row(root, r) for r in selected]
    groups = {}
    for row in rows:
        kind = row['case_id'].split('__')[0]
        out = groups.setdefault(kind, dict(observations=0, supported_caps=0, support_rays=0,
            observations_without_cap_support=0, undefined_support_observations=0,
            min_frames_per_supported_cap=None, max_analytic_error_m=0.))
        out['observations'] += 1
        if 'sides' not in row:
            out['undefined_support_observations'] += 1
            continue
        supports = [s for s in row['sides'] if s['count']]
        out['observations_without_cap_support'] += int(not supports)
        out['supported_caps'] += len(supports)
        for side in supports:
            out['support_rays'] += side['count']
            frames = sum(n>0 for n in side['per_frame'])
            old = out['min_frames_per_supported_cap']
            out['min_frames_per_supported_cap'] = frames if old is None else min(old,frames)
            out['max_analytic_error_m'] = max(out['max_analytic_error_m'],side['numeric_paths']['analytic']['max_range_error_m'])
    print(json.dumps(dict(groups=groups, per_case=rows, new_labels=0, renders=0,
        optimizer_steps=0, general_teacher_qualified=False),indent=2))


if __name__ == '__main__':
    main()
