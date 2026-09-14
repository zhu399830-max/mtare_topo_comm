"""Plan full fixed-population teacher reads, without reading any payload.

All 16 observations per task remain, including empty spatial background. No
coverage result, detector score, construction degree or source ID selects rows.
"""
import math
from pathlib import Path

from .gse_surface_coverage_scope_v1 import compile_coverage_scope
from .gse_surface_teacher_scope_v1 import FIELDS, plan_field
from .gse_surface_input_scope_v1 import checked_read, _object
from mtare_topo.governance_surface_material import SOURCE, SOURCE_SEAL
from mtare_topo.governance_surface_input import expected_tasks, SEALS
from mtare_topo.governance_identity_inventory import P1A


def combine_fiveframe_plans(header, field, observations):
    if len(observations) != 16:
        raise ValueError('fixed sixteen observations required')
    plans = [plan_field(header, field, r['frame_rows'], r['source_frame_count']) for r in observations]
    selected = [f for r in observations for f in r['frame_rows']]
    if len(set(selected)) != 80:
        raise ValueError('original eighty unique source frames required')
    keys = sorted({key for p in plans for key in p['chunk_keys']}, key=lambda x: int(x.split('.')[0]))
    count = header['shape'][0]; chunk = header['chunks'][0]
    return dict(shape=header['shape'], chunks=header['chunks'], dtype=header['dtype'],
                selected_rows=sorted(selected), observation_rows=[r['frame_rows'] for r in observations],
                chunk_keys=keys, decoded_rows_including_collateral=sum(
                    min(count, (int(k.split('.')[0])+1)*chunk)-int(k.split('.')[0])*chunk for k in keys),
                decoded_padded_bytes=len(keys)*math.prod(header['chunks'])*FIELDS[field][2])


def compile_population_teacher_scope(root):
    root = Path(root).resolve(strict=True)
    coverage = compile_coverage_scope(root)
    reads = dict(coverage['metadata_reads_sha256'])
    tasks = expected_tasks()
    sensor_seal = SEALS['sensor']
    raw = checked_read(root, sensor_seal['path'], sensor_seal['sha256'], reads)
    json_paths = {P1A+'/artifacts/'+role+'/'+s['partition']+'/'+t+'.json'
                  for t, s in tasks.items() for role in ('constructions', 'codebooks')}
    prefixes = {s['sensor']+'/'+field+'/' for s in tasks.values() for field in FIELDS}
    candidates = {}
    for line in raw.decode().splitlines():
        h, path = line.split(None, 1)
        if path not in json_paths and not any(path.startswith(prefix) for prefix in prefixes):
            continue
        if path in candidates:
            raise ValueError('duplicate selected seal entry')
        candidates[path] = h
    exported = checked_read(root, SOURCE+'/artifacts/evidence_sha256.txt', SOURCE_SEAL, reads)
    inputs = {}; wanted = {SOURCE+'/artifacts/input_manifest.json'} | {
        SOURCE+'/artifacts/inputs/'+task+'.npz' for task in tasks}
    for line in exported.decode().splitlines():
        h, path = line.split(None, 1)
        if path in wanted:
            if path in inputs:raise ValueError('duplicate exported input')
            inputs[path] = h
    if set(inputs) != wanted:
        raise ValueError('complete original input export required')
    mp = SOURCE+'/artifacts/input_manifest.json'
    manifest = _object(checked_read(root, mp, inputs[mp], reads))
    manifest_rows = {t: [] for t in tasks}
    for r in manifest['observations']:
        if r['task'] not in tasks:raise ValueError('export outside selected population')
        manifest_rows[r['task']].append(r)
    files = {p: candidates[p] for p in sorted(json_paths)}
    entries = []; plans = {}
    for entry in coverage['entries']:
        task = entry['task']; observations = entry['observations']
        if manifest_rows[task] != observations:
            raise ValueError('selection and exported input row order mismatch')
        sources = tasks[task]
        for field in FIELDS:
            prefix = sources['sensor']+'/'+field
            path = prefix+'/.zarray'
            header = _object(checked_read(root, path, candidates[path], reads))
            plan = combine_fiveframe_plans(header, field, observations)
            plans[prefix] = plan
            for path in [path, *(prefix+'/'+k for k in plan['chunk_keys'])]:
                files[path] = candidates[path]
        entries.append(dict(task=task, observations=observations,
                            construction_path=entry['construction_path'],
                            codebook_path=P1A+'/artifacts/codebooks/'+sources['partition']+'/'+task+'.json',
                            sensor_prefix=sources['sensor'], input_path=SOURCE+'/artifacts/inputs/'+task+'.npz'))
    return dict(schema='gse_surface_population_teacher_scope_v1', status='PREPARATION_NOT_EXECUTION_AUTHORITY',
                entries=entries, file_sha256=files, input_sha256=inputs, array_access=plans,
                metadata_reads_sha256=reads,
                counts=dict(parents=70,tasks=210,physical_edge_units=1120,observations=3360,
                            selected_frames=16800,construction_files=210,codebook_files=210,
                            input_npz_files=210,array_headers=630,
                            source_chunk_files=sum(len(p['chunk_keys']) for p in plans.values()),
                            decoded_padded_bytes=sum(p['decoded_padded_bytes'] for p in plans.values()),
                            actual_payload_reads=0,labels=0),
                restrictions=['C01_C07_only','fixed_all_observations_no_positive_filter',
                              'teacher_identity_never_student_input','unknown_not_negative','no_training'])
