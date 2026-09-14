"""Bind the already selected new12 to original source/mesh inputs, metadata only."""
from _bootstrap import PROJECT_ROOT as ROOT
import hashlib,json,math
from ai_junction_pilot import sha,write
from mtare_topo.data.gse_surface_teacher_scope_v1 import FIELDS,plan_field
from mtare_topo.governance_identity_inventory import P1A
from mtare_topo.governance_surface_input import SEALS

CARD='configs/v3/gate3/data_cards/gse_ai_new_fit_review_v1.json'
ORIGINAL='configs/v3/gate3/gse_original_surface_source_binding_v1.json'
OUT='configs/v3/gate3/gse_new12_surface_source_scope_v1.json'


def checked(path,digest):
    raw=(ROOT/path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('source drift '+path)
    return raw


def compile_scope():
    old=json.loads((ROOT/CARD).read_text())['scope'];entries=old['entries']
    if (len(entries)!=12 or len({e['parent'] for e in entries})!=12
            or [e['case'] for e in entries]!=list(range(12))):raise ValueError('fixed population drift')
    seal=SEALS['sensor'];raw=checked(seal['path'],seal['sha256'])
    index=dict((p,h) for h,p in (line.split(None,1) for line in raw.decode().splitlines()))
    files={CARD:sha(ROOT/CARD),seal['path']:seal['sha256'],ORIGINAL:sha(ROOT/ORIGINAL)}
    archive=json.loads((ROOT/ORIGINAL).read_text())
    files[archive['archive_path']]=archive['archive_sha256']
    bound=[];allocations={};headers={};frames=set()
    for e in entries:
        task=e['task']
        if e['split']!='fit' or not any('_C%02d__'%k in task for k in range(1,7)):
            raise ValueError('fit C01-C06 only')
        if e['parent'] in old['excluded_parents']:raise ValueError('excluded historical parent')
        frames.update((task,f) for f in e['frame_rows'])
        docs={k:P1A+'/artifacts/'+k+'/fit/'+task+'.json' for k in ('constructions','codebooks')}
        for p in docs.values():files[p]=index[p]
        arrays={}
        for field in FIELDS:
            prefix=P1A+'/artifacts/dataset/fit/'+task+'.zarr/'+field
            path=prefix+'/.zarray';h=json.loads(checked(path,index[path]))
            files[path]=index[path];headers[path]=index[path]
            access=plan_field(h,field,e['frame_rows'],h['shape'][0])
            arrays[field]=dict(prefix=prefix,header=h,access=access)
            for key in access['chunk_keys']:
                p=prefix+'/'+key;files[p]=index[p]
                allocations[p]=math.prod(h['chunks'])*FIELDS[field][2]
        bundle=next(p for p in old['indexed_bundles'] if p.endswith('/case_'+str(e['case'])+'.json'))
        files[bundle]=old['indexed_bundles'][bundle]
        files[e['student_path']]=e['student_sha256']
        bound.append(dict(observation=e['case'],identity=e,documents=docs,arrays=arrays,indexed_bundle=bundle))
    if len(frames)!=60:raise ValueError('unique frame population drift')
    return dict(schema_version='gse_new12_surface_source_scope_v1',
        status='METADATA_BOUND_NOT_LABEL_OR_TRAINING_AUTHORITY',entries=bound,input_sha256=files,
        original_surface_binding=archive,observations=12,independent_parents=12,unique_frames=60,
        maximum_return_slots=691200,effective_valid_returns=sum(old['valid_returns']),
        valid_returns_per_observation=old['valid_returns'],
        unique_source_chunks=len(allocations),decoded_unique_chunk_bytes=sum(allocations.values()),
        selected_metadata_headers=headers,payload_reads=0,labels_generated=0,training_steps=0,
        spacing=old['spacing'],selection_bias=old['bias'],selection_seed=old['selection_seed'],
        source_code_semantics='Active field candidate source sets, not unique triangle identity; preserve all candidates',
        supervision_status='Requires source-to-return surface evidence and explicit mixed/unknown handling before patch labels',
        restrictions=['No old16 reads/runs','No C08-C10 payload','No teacher ID in model forward',
            'No source-majority forced purity','No nearest structure labels','No structural success from source affinity'])


if __name__=='__main__':
    scope=compile_scope();write(ROOT/OUT,scope)
    print(json.dumps({k:v for k,v in scope.items() if k not in ('entries','input_sha256','original_surface_binding','selected_metadata_headers')},ensure_ascii=False))
