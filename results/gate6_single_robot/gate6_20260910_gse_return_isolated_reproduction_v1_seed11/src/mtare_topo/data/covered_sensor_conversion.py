"""Stream sealed diagnostic records into twelve non-training input bundles."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .cano_sensor_smoke import lidar_local_directions, world_directions
from .gse_synthetic_matrix import matrix
from .covered_sensor_adapter import diagnostic_frame
from .covered_five_frame_bundle import assemble_five_frame_bundle
from .primitive_relation_dataset import PrimitiveMembershipCodebook


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def convert(root, run, scope, progress):
    root = Path(root); run = Path(run)
    def verify_inputs():
        for p, h in scope['input_sha256'].items():
            if file_sha(root/p) != h:
                raise ValueError('sealed input drift: '+p)
    verify_inputs()
    source = root/scope['source_run']
    summary = json.loads((source/'metrics/summary.json').read_text())
    if summary['state'] != 'DIAGNOSTIC_PASS' or summary['checked_rays'] != 691200:
        raise ValueError('full source diagnostic required')
    manifest = json.loads((source/'config/source_config').read_text())['frame_inputs']
    cases = {c['case_id']: c for c in matrix()}
    local = lidar_local_directions().reshape(-1,3).astype(np.float64)
    n = 0; max_storage = 0.; multisource = 0; exported = []
    with (source/'metrics/rays.jsonl').open() as stream:
        for name in scope['cases']:
            case = cases[name]; scans = []
            book = PrimitiveMembershipCodebook([e['id'] for e in case['program']['edges']])
            for frame in range(5):
                target = manifest[n//11520]
                if (target['case_id'],target['frame_index']) != (name,frame):
                    raise ValueError('manifest order drift')
                case_hash = hashlib.sha256(json.dumps(case,sort_keys=True,separators=(',',':')).encode()).hexdigest()
                if case_hash != target['case_sha256']:
                    raise ValueError('case declaration drift')
                origin = np.asarray(case['poses_world_m'][frame],float)
                directions = world_directions(local,case['yaw_deg'][frame])
                results = []
                for ray in range(11520):
                    line = stream.readline()
                    if not line: raise ValueError('truncated source frame')
                    row = json.loads(line)
                    if (row['case_id'],row['frame_index'],row['ray_index']) != (name,frame,ray):
                        raise ValueError('record order drift')
                    if not np.array_equal(directions[ray], row['direction']):
                        raise ValueError('direction drift')
                    if not row['score']['passed']:
                        raise ValueError('failed source record')
                    results.append(row['result']); n += 1
                scan = diagnostic_frame(results,sensor_xyz_m=origin,directions_xyz=directions,codebook=book)
                for ray, result in enumerate(results):
                    if scan.valid_mask.reshape(-1)[ray]:
                        max_storage = max(max_storage,abs(float(scan.range_m.reshape(-1)[ray])-result['distance_m']))
                decoded = book.decode(scan.primitive_membership_code)
                for ray, source_set in enumerate(decoded):
                    if source_set and {book.primitive_ids[i] for i in source_set} != set(results[ray]['sources']):
                        raise ValueError('source set lost in conversion')
                multisource += scan.ambiguous_ray_count
                scans.append((frame,scan))
            bundle = assemble_five_frame_bundle(case=case,indexed_frames=scans,codebook=book)
            bundle['source']['input_run'] = scope['source_run']
            bundle['source']['input_sha256'] = scope['input_sha256']
            bundle['qualification']['distance_evidence'] = 'bound_to_exact_sealed_diagnostic_records'
            prefix = run/'artifacts'/name
            with Path(str(prefix)+'.student.npz').open('xb') as out:
                np.savez_compressed(out,**bundle.pop('student'))
            diagnostic = bundle.pop('diagnostic_only')
            bundle['reported_source_codebook'] = diagnostic.pop('codebook')
            with Path(str(prefix)+'.diagnostic.npz').open('xb') as out:
                np.savez_compressed(out,**diagnostic)
            with Path(str(prefix)+'.json').open('x') as out:
                json.dump(bundle,out,indent=2)
            # Verify actual stored arrays, not only pre-write memory.
            with np.load(str(prefix)+'.student.npz',allow_pickle=False) as stored:
                assert stored['ranges_m'].shape == (5,16,720)
                assert np.array_equal(stored['ranges_m'],np.stack([s.range_m for _,s in scans]))
            exported.append(name)
            progress(dict(observations=len(exported),frames=len(exported)*5,rays=n))
        if stream.readline(): raise ValueError('extra source records')
    verify_inputs()
    if n != 691200: raise ValueError('incomplete conversion')
    return dict(observations=len(exported),frames=60,rays=n,case_ids=exported,
                max_float32_storage_error_m=max_storage,multisource_valid_rays=multisource,
                training_eligible=False,structure_labels_generated=0,optimizer_steps=0)
