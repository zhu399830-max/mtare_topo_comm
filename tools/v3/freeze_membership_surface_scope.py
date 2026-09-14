"""Bind only existing16 windows to historical surface inputs; metadata only.

This is preparation, not a data card approval or permission to create labels.
"""
from _bootstrap import PROJECT_ROOT as ROOT
import hashlib
import json
import math
from mtare_topo.data.gse_surface_teacher_scope_v1 import FIELDS, plan_field
from mtare_topo.governance_identity_inventory import P1A
from mtare_topo.governance_surface_input import SEALS


def checked(path, digest):
    raw = (ROOT/path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError('source hash drift: '+path)
    return raw


def compile_scope():
    path = 'configs/v3/gate3/gse_membership_fit_input_binding_v1.json'
    raw = (ROOT/path).read_bytes()
    binding = json.loads(raw)
    files = {path: hashlib.sha256(raw).hexdigest()}
    seal = SEALS['sensor']
    index = dict((p,h) for h,p in (line.split(None,1) for line in
                 checked(seal['path'],seal['sha256']).decode().splitlines()))
    files[seal['path']] = seal['sha256']
    entries = []; allocations = {}; selected_frames = set(); parents = set()
    for i, b in enumerate(binding['entries']):
        source = b['source']; task = source['task']
        if source['split'] != 'fit' or not any('_C%02d__'%k in task for k in range(1,7)):
            raise ValueError('only original fit parents allowed')
        parents.add(source['parent_id'])
        selected_frames.update((task,f) for f in b['frame_rows'])
        docs = {role: P1A+'/artifacts/'+role+'/fit/'+task+'.json'
                for role in ('constructions','codebooks')}
        for p in docs.values(): files[p] = index[p]
        plans = {}
        for field in FIELDS:
            prefix = P1A+'/artifacts/dataset/fit/'+task+'.zarr/'+field
            header_path = prefix+'/.zarray'
            h = json.loads(checked(header_path,index[header_path]))
            files[header_path] = index[header_path]
            # Single-window exports omit corpus length; use its authenticated
            # original Zarr header, not a guessed or selected-row count.
            count = source.get('source_frame_count',h['shape'][0])
            plan = plan_field(h,field,b['frame_rows'],count)
            plans[field] = dict(prefix=prefix,header=h,access=plan)
            for key in plan['chunk_keys']:
                p = prefix+'/'+key; files[p] = index[p]
                allocations[p] = math.prod(h['chunks'])*FIELDS[field][2]
        for role in ('student','target'):
            files[b[role+'_path']] = b[role+'_sha256']
        entries.append(dict(observation=i,original_binding=b,documents=docs,arrays=plans))
    if len(entries)!=16 or len(selected_frames)!=80 or len(parents)!=8:
        raise ValueError('fixed16 population drift')
    return dict(schema_version='gse_membership_surface_scope_v1',
        status='PREPARATION_ONLY_NO_LABEL_OR_EXECUTION_AUTHORITY',entries=entries,
        input_sha256=files,parents=sorted(parents),observations=16,unique_frames=80,
        maximum_return_slots=16*5*16*720,
        unique_source_chunks=len(allocations),decoded_unique_chunk_bytes=sum(allocations.values()),
        student_decoded_observations=binding['decoded_observations_per_uncached_pass'],
        payload_reads=0,labels_generated=0,training_steps=0,
        purpose='Return to original candidate-source surface arc evidence only; not structural membership',
        source_code_semantics='Original runner enables operand_signed_distances; codes are qualified active field source sets, not saved triangle identities',
        restrictions=['Retain all ambiguous source sets','No nearest GT structure assignment',
            'No unknown to background','No model forward or training','No protected worlds',
            'Matching tolerance and numerical residual must be reported, not silently widened'])


if __name__ == '__main__':
    scope = compile_scope()
    out = ROOT/'configs/v3/gate3/gse_membership_surface_scope_v1.json'
    with out.open('x') as f: json.dump(scope,f,indent=2)
    print(json.dumps({k:v for k,v in scope.items() if k not in ('entries','input_sha256')}))
