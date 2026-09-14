"""Read-only loss-mask census of the exact sealed 45 scored fixtures.

Does not promote fixture oracles to general-world completeness or export labels.
"""
import _bootstrap
import gzip
import argparse
import io
import hashlib
import json
from collections import Counter
from pathlib import Path
from mtare_topo.teacher.gse_surface_target_adapter_v1 import observed_targets

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'results/gate3_semantics/gate3_20260908_gse_synthetic_matrix_v1_seed20260906'


def main(check_inputs=False):
    pins = {p:h for h,p in (line.split('  ',1) for line in
        (RUN/'artifacts/evidence_sha256.txt').read_text().splitlines())}
    def read(path):
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != pins[str(path.relative_to(ROOT))]:
            raise ValueError('sealed input drift')
        return data
    summary = json.loads(read(RUN/'metrics/summary.json'))
    selected = [x for x in summary['per_case'] if x['status']=='FIXTURE_GEOMETRY_PASS']
    if len(selected) != 45: raise ValueError('exact 45 fixture population required')
    counts = Counter(); groups = Counter()
    for score in selected:
        case_id = score['case_id']
        source = json.loads(gzip.decompress(read(RUN/'artifacts'/(case_id+'.json.gz'))))
        if source['independent_score'] != score: raise ValueError('summary/record score mismatch')
        record = source['produced_targets']['record']
        if check_inputs:
            import numpy as np
            from mtare_topo.data.gse_feature_projection_v1 import project_input
            with np.load(io.BytesIO(read(RUN/'artifacts'/(case_id+'.npz'))),allow_pickle=False) as payload:
                ranges=payload['ranges_m']; valid=payload['valid_mask']
                translation=payload['relative_translation_current_sensor_m']; yaw=payload['relative_yaw_current_sensor_deg']
                if ranges.shape!=(5,16,720) or ranges.dtype!=np.float32 or valid.shape!=ranges.shape or valid.dtype!=np.uint8:
                    raise ValueError('five-frame sensor payload drift')
                if not np.isin(valid,[0,1]).all() or not valid.reshape(5,-1).any(axis=1).all():
                    raise ValueError('encoder requires five supported frames')
                if record['source_frame_indices']!=source['source']['frame_rows']:
                    raise ValueError('loss/input source frame mismatch')
                image=np.stack((ranges/np.float32(50),valid.astype(np.float32)),axis=1)
                points,mask=project_input(image,translation,yaw,device='cpu')
                if not np.isfinite(points.numpy()[mask.numpy()]).all():
                    raise ValueError('nonfinite projected valid points')
                if not np.array_equal(mask.numpy().reshape(5,16,720),valid.astype(bool)):
                    raise ValueError('projection changed validity')
                counts['input_observations_checked']+=1
                counts['input_frame_occurrences']+=5
                counts['input_ray_positions']+=ranges.size
                counts['input_valid_returns']+=int(valid.sum())
                counts['compact_cache_bytes_float32_without_token_index']+=57600*3*4+900*128*4+57600
        target = observed_targets([record])
        row = dict(case_id=case_id)
        for name in ('anchor_valid','opening_valid','direction_valid','dimension_valid',
                     'reachability_valid','physical_reference_valid','membership_valid',
                     'anchor_region_complete','opening_region_complete'):
            row[name] = int(getattr(target,name).sum().item()); counts[name] += row[name]
        counts['membership_true'] += int(target.membership[target.membership_valid].sum().item())
        counts['observations_without_anchor'] += int(row['anchor_valid']==0)
        row['general_training_qualified'] = False
        row['complete_loss_background'] = bool(row['anchor_region_complete'] and row['opening_region_complete'])
        groups[case_id.split('__')[0]] += 1
        print(json.dumps(row),flush=True)
    print(json.dumps(dict(observations=len(selected), groups=dict(groups), counts=dict(counts),
        new_labels=0, optimizer_steps=0, full_detection_training_qualified=False,
        conclusion='Fixture oracle scoring and incomplete producer loss masks are separate contracts. '
                   'Positive locations enter loss; unmatched background does not. '
                   'Do not silently flip completeness or generalize fixture PASS.')),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--check-inputs',action='store_true')
    main(parser.parse_args().check_inputs)
