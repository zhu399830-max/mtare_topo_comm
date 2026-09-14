"""Restrict original multi-package reader to whole selected tasks; metadata only."""
from copy import deepcopy
from pathlib import Path
from v8_multiview_manifest_v1 import compile_manifest


def compile_scope(root):
    root=Path(root).resolve(strict=True)
    from mtare_topo.data.multiview_teacher_scope_v1 import compile_scope as original
    full=original(root);manifest=compile_manifest()
    selected={(r['task'],r['source_sequence_id']) for r in manifest['observations']}
    tasks={t for t,_ in selected}
    entries=[deepcopy(e) for e in full['entries'] if e['task'] in tasks]
    actual={(s['task'],s['source_sequence_id']) for e in entries for s in e['observations']}
    if actual!=selected:raise ValueError('selected tasks must contain exactly the fixed141 windows')
    prefixes={e['sensor_prefix']+'/' for e in entries}
    arrays={k:v for k,v in full['array_access'].items() if any(k.startswith(p) for p in prefixes)}
    files={e[k] for e in entries for k in ('construction_path','codebook_path')}
    for prefix,plan in arrays.items():
        files.add(prefix+'/.zarray')
        files.update(prefix+'/'+k for k in plan['chunk_keys'])
    packages={r['input_path'] for e in entries for r in e['input_references']}
    scope=dict(full,entries=entries,array_access=arrays,
        file_sha256={p:full['file_sha256'][p] for p in sorted(files)},
        input_sha256={p:full['input_sha256'][p] for p in sorted(packages)})
    scope['schema']='v8_multiview_exact141_teacher_scope_v1'
    scope['counts']=dict(manifest['counts'],
        construction_files=len(entries),codebook_files=len(entries),
        decoded_padded_bytes=sum(p['decoded_padded_bytes'] for p in arrays.values()))
    containers={}
    for entry in entries:
        for reference in entry['input_references']:
            path=reference['input_path']
            item=containers.setdefault(path,dict(observation_count=reference['observation_count'],selected_rows=set()))
            if item['observation_count']!=reference['observation_count']:
                raise ValueError('shared container population drift')
            item['selected_rows'].add(reference['input_row'])
    scope['input_container_population']={p:dict(observation_count=v['observation_count'],
        selected_rows=sorted(v['selected_rows']),
        incidental_rows=v['observation_count']-len(v['selected_rows'])) for p,v in sorted(containers.items())}
    scope['restrictions']=list(full['restrictions'])+[
        'exact141_only','shared_NPZ_container_rows_not_new_samples',
        'mechanism_diagnostic_not_training_or_calibration']
    return scope
