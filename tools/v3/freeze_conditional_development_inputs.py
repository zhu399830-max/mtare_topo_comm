"""Bind only original source headers/seals for the frozen 240 observations.

No scan chunks, construction payloads, targets, or model tensors are decoded.
The resulting scope requires a separate preflighted export before payload reads.
"""
from _bootstrap import PROJECT_ROOT as ROOT
import hashlib
import json
import math
from ai_junction_pilot import sha, write
from mtare_topo.governance_surface_input import SEALS, expected_tasks
from mtare_topo.governance_identity_inventory import P1A

MANIFEST = 'configs/v3/gate3/gse_conditional_development_manifest_v1.json'
OUTPUT = 'configs/v3/gate3/gse_conditional_development_inputs_v1.json'
SENSOR = ('range_m', 'valid_mask', 'sensor_xyz_m', 'yaw_deg', 'primitive_membership_code')
SEQUENCE = ('relative_translation_current_sensor_m', 'relative_yaw_current_sensor_deg',
            'frame_row', 'source_global_sequence_index')


def plan_rows(header, rows):
    """Exact selected rows and unavoidable compressed-chunk collateral."""
    shape, chunks = header['shape'], header['chunks']
    if not shape or len(shape) != len(chunks) or any(type(i) is not int or i < 1 for i in shape + chunks):
        raise ValueError('invalid array dimensions')
    if chunks[1:] != shape[1:] or header['filters'] or header['order'] != 'C':
        raise ValueError('unsupported archived layout')
    selected = sorted(set(rows))
    if not selected or any(type(i) is not int or not 0 <= i < shape[0] for i in selected):
        raise ValueError('invalid selected source row')
    dtype = header['dtype']
    if dtype not in ('<f4', '<f8', '|u1', '<i4', '<i8', '<u4', '<u8', '<u2', '|b1'):
        raise ValueError('unsupported source dtype: ' + str(dtype))
    itemsize = int(dtype[2:])
    indices = sorted({i // chunks[0] for i in selected})
    intervals = [[i * chunks[0], min((i + 1) * chunks[0], shape[0])] for i in indices]
    return dict(header=header, selected_rows=selected,
                chunk_keys=['.'.join(map(str, [i] + [0] * (len(shape) - 1))) for i in indices],
                decoded_intervals=intervals,
                collateral_rows=sum(b-a for a,b in intervals)-len(selected),
                decoded_padded_bytes=len(indices)*math.prod(chunks)*itemsize)


def main():
    if (ROOT/OUTPUT).exists():
        raise FileExistsError('immutable scope already exists')
    manifest = json.loads((ROOT/MANIFEST).read_text())
    entries = manifest['entries']
    assert len(entries) == 240 and manifest['unique_variant_source_frames'] == 1200
    tasks = expected_tasks()
    selected_tasks = sorted({e['task'] for e in entries})
    assert len(selected_tasks) == 15 and all(tasks[t]['partition'] == 'c07' for t in selected_tasks)
    prefixes = {tasks[t][role] + '/' + f + '/'
                for t in selected_tasks for role,fields in [('sensor', SENSOR), ('teacher', SEQUENCE)] for f in fields}
    documents = {P1A+'/artifacts/'+kind+'/c07/'+t+'.json'
                 for t in selected_tasks for kind in ('constructions', 'codebooks')}
    pins = {MANIFEST:sha(ROOT/MANIFEST)}
    available = {}
    for seal in SEALS.values():
        raw = (ROOT/seal['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != seal['sha256']:
            raise ValueError('original source seal drift')
        pins[seal['path']] = seal['sha256']
        for line in raw.decode().splitlines():
            digest,path = line.split(None,1)
            if path in documents or any(path.startswith(p) for p in prefixes):
                available[path] = digest
    plans, task_files = {}, {}
    for task in selected_tasks:
        rows = [e for e in entries if e['task'] == task]
        assert len(rows) == 16
        task_files[task] = {}
        for kind in ('constructions','codebooks'):
            path = P1A+'/artifacts/'+kind+'/c07/'+task+'.json'
            pins[path] = available[path]
            task_files[task][kind] = path
        for role,fields in [('sensor', SENSOR), ('teacher', SEQUENCE)]:
            indices = [i for e in rows for i in e['frame_rows']] if role == 'sensor' else [e['sequence_row'] for e in rows]
            for field in fields:
                prefix = tasks[task][role]+'/'+field
                path = prefix+'/.zarray'
                raw = (ROOT/path).read_bytes()
                if hashlib.sha256(raw).hexdigest() != available[path]:
                    raise ValueError('source header drift: '+path)
                plan = plan_rows(json.loads(raw), indices)
                plans[prefix] = plan
                pins[path] = available[path]
                pins.update({prefix+'/'+k:available[prefix+'/'+k] for k in plan['chunk_keys']})
    scope = dict(schema_version='conditional_development_source_scope_v1',
        status='EXACT_METADATA_BOUND_PAYLOAD_NOT_EXPORTED', manifest=MANIFEST,
        entries=entries, parents=5, observations=240, physical_edges=80, tasks=15, frames=1200,
        task_prefixes={t:tasks[t] for t in selected_tasks}, task_files=task_files,
        array_plans=plans, input_sha256=pins,
        decoded_padded_bytes=sum(p['decoded_padded_bytes'] for p in plans.values()),
        student_fields=['ranges_m','valid_mask',*SEQUENCE[:2]],
        teacher_only_fields=list(SENSOR[2:]),
        split_audit=manifest['split_audit'],
        labels_generated=0, training_steps=0, payloads_decoded=0,
        limitation='Header binding only. Collateral chunk rows must not become examples; absolute poses and source identities are loss/evidence-only. No continuous-route claim.')
    write(ROOT/OUTPUT, scope)
    print(json.dumps({k:scope[k] for k in ('status','parents','observations','physical_edges','tasks','frames','decoded_padded_bytes','payloads_decoded')}))


if __name__ == '__main__':
    main()
