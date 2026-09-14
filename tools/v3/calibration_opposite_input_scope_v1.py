"""Exact existing six-field missing-input access; metadata only until execution."""
import hashlib
import json
from pathlib import Path
from mtare_topo.governance_surface_input import FIELDS,SEALS,expected_tasks
from mtare_topo.data.gse_surface_input_export_v1 import plan_array_access,SurfaceInputReader

INVENTORY='configs/v3/gate3/calibration_opposite_inventory_v1.json'
INVENTORY_SHA='6a7431c5cefe876ef07d2244d76d5334bb71325ca7ebc242448a19b5ea341e39'


def compile_scope(root):
    root=Path(root);opened={}
    def read(p,h):
        raw=(root/p).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=h:raise ValueError('metadata drift: '+p)
        opened[p]=h;return raw
    doc=json.loads(read(INVENTORY,INVENTORY_SHA))
    missing=[dict(r) for r in doc['rows'] if r['cached_input'] is None]
    if len(missing)!=289 or any(r['split']!='calibration' or '_C07__' not in r['task'] for r in missing):
        raise ValueError('exact289 C07 calibration required')
    source=expected_tasks();tasks={r['task']:source[r['task']] for r in missing};sealed={}
    for s in SEALS.values():
        for l in read(s['path'],s['sha256']).decode().splitlines():
            h,p=l.split('  ',1);sealed[p]=h
    files={};plans={}
    for task,source in sorted(tasks.items()):
        rows=[r for r in missing if r['task']==task]
        for role,names in FIELDS.items():
            prefix=source[role]
            for name in ('.zgroup','.zattrs'):
                p=prefix+'/'+name;read(p,sealed[p]);files[p]=sealed[p]
            selected=[f for r in rows for f in r['frame_rows']] if role=='sensor' else [r['sequence_row'] for r in rows]
            for name in names:
                prefix2=prefix+'/'+name;p=prefix2+'/.zarray';header=json.loads(read(p,sealed[p]));files[p]=sealed[p]
                count_key='source_frame_count' if role=='sensor' else 'source_sequence_count'
                for r in rows:
                    if count_key in r and r[count_key]!=header['shape'][0]:raise ValueError('source count mismatch')
                    r[count_key]=header['shape'][0]
                    r['parent_id'],r['variant']=task.split('__')
                plan=plan_array_access(header,name,role,selected);plans[prefix2]=plan
                for chunk in plan['chunk_keys']:
                    p=prefix2+'/'+chunk;files[p]=sealed[p]
    reader=SurfaceInputReader(root,task_sources=tasks,selection=missing,sealed_keys=files,variable_population=True)
    return dict(schema='calibration_opposite289_input_scope_v1',status='METADATA_ONLY_NOT_EXPORT_AUTHORITY',
        metadata_sha256=opened,source_seals=SEALS,task_sources=tasks,missing=missing,
        input_files_sha256=files,array_access=plans,
        counts=dict(observations=len(missing),tasks=len(tasks),parents=len({r['parent_id'] for r in missing}),
            selected_unique_variant_frames=len({(r['task'],f) for r in missing for f in r['frame_rows']}),
            decoded_padded_bytes=sum(p['decoded_padded_bytes'] for p in plans.values()),
            source_files=len(files)),
        restrictions=['six_student_fields_only','calibration_not_fit','no_labels_or_training','no_rerender',
                      'opposite_of_exact_original27_variants_only','shared_chunk_rows_not_new_samples'])


if __name__=='__main__':
    print(json.dumps(compile_scope(Path(__file__).resolve().parents[2]),separators=(',',':')))
