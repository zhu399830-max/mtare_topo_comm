"""Missing-only six-field input scope; original3360 inputs are not re-exported."""
from collections import Counter, defaultdict
from pathlib import Path

from mtare_topo.governance_surface_selection import PARENTS
from mtare_topo.governance_surface_input import SELECTION, SELECTION_SEAL, SEALS, FIELDS, expected_tasks
from .gse_surface_input_scope_v1 import checked_read, _object, selected_rows
from .gse_surface_input_export_v1 import plan_array_access
from .gse_structural_supplement_v1 import merge_source_requests

SUPPLEMENT = 'results/gate3_semantics/gate3_20260907_gse_structural_supplement_v1_seed20260906'
SUPPLEMENT_SEAL = 'ab4405bf0b934ece6e6528595651aa464f65bc1b5bc444d234901efd7e19f0ec'


def missing_population(background, selections):
    """Preserve background; enrich supplemental identities without teacher features."""
    tasks = expected_tasks()
    by_task = selected_rows(background, tasks)
    originals = background['observations']
    merged = merge_source_requests(originals, selections)
    old_keys = {(r['task'],r['source_sequence_id']) for r in originals}
    new = {}
    for selection in selections:
        for source in selection['sources']:
            task = source['task']
            if task not in tasks:
                raise ValueError('supplement outside original C01-C07 scope')
            original = by_task[task][0]
            if (selection['parent_id'] != original['parent_id'] or selection['split'] != original['split']
                    or source['variant'] != original['variant']
                    or not selection['traversal_id'].startswith(original['parent_id']+':')):
                raise ValueError('supplement parent/split/variant mismatch')
            frames = source['frame_rows']; seq = source['sequence_row']
            if (len(frames)!=5 or any(type(f) is not int for f in frames)
                    or frames != list(range(frames[0],frames[0]+5)) or frames[0]<0
                    or frames[-1]>=original['source_frame_count'] or type(seq) is not int
                    or not 0<=seq<original['source_sequence_count']):
                raise ValueError('supplement row/history bounds invalid')
            key = (task,source['source_sequence_id'])
            if key in old_keys:
                continue
            new[key] = dict(source,parent_id=original['parent_id'],split=original['split'],
                traversal_id=selection['traversal_id'], source_frame_count=original['source_frame_count'],
                source_sequence_count=original['source_sequence_count'], labels_generated=False,
                continuous_route_evidence=False)
    new_rows = [new[k] for k in sorted(new)]
    return dict(original_observations=originals, combined_requests=merged, new_observations=new_rows,
        counts=dict(original_observations=len(originals),new_observations=len(new_rows),
            combined_unique_observations=len(merged),
            new_unique_variant_frames=len({(r['task'],f) for r in new_rows for f in r['frame_rows']}),
            new_split_observations=dict(Counter(r['split'] for r in new_rows)),labels=0))


def compile_missing_input_scope(root):
    root = Path(root).resolve(strict=True); reads = {}
    def sealed_json_files(run,seal_sha,paths):
        raw = checked_read(root,run+'/artifacts/evidence_sha256.txt',seal_sha,reads)
        hashes = {}
        for line in raw.decode().splitlines():
            sha,path = line.split('  ',1)
            if path in paths:
                if path in hashes:raise ValueError('duplicate selected seal path')
                hashes[path] = sha
        if set(hashes)!=set(paths):raise ValueError('selected source missing in seal')
        return {p:_object(checked_read(root,p,hashes[p],reads)) for p in sorted(paths)}
    path=SELECTION+'/artifacts/selection_manifest.json'
    original=sealed_json_files(SELECTION,SELECTION_SEAL,{path})[path]
    supplements=sealed_json_files(SUPPLEMENT,SUPPLEMENT_SEAL,
        {SUPPLEMENT+'/artifacts/'+p+'.json' for p in PARENTS})
    selections=[]
    for path,out in supplements.items():
        if path!=SUPPLEMENT+'/artifacts/'+out['parent_id']+'.json':
            raise ValueError('supplement path-parent mismatch')
        selections.extend(out['selections'])
    population=missing_population(original,selections)
    by_task=defaultdict(list)
    for row in population['new_observations']:by_task[row['task']].append(row)
    tasks={t:s for t,s in expected_tasks().items() if t in by_task}
    files,plans={},{}
    for role,fields in FIELDS.items():
        seal=SEALS[role]
        raw=checked_read(root,seal['path'],seal['sha256'],reads)
        prefixes={s[role]+'/' for s in tasks.values()}
        candidates={}
        for line in raw.decode().splitlines():
            sha,path=line.split(None,1)
            marker=path.find('.zarr/')
            if marker<0 or path[:marker+6] not in prefixes:continue
            key=path[marker+6:]
            if key not in ('.zgroup','.zattrs') and key.split('/')[0] not in fields:continue
            if path in candidates:raise ValueError('duplicate input source seal path')
            candidates[path]=sha
        for task,source in sorted(tasks.items()):
            prefix=source[role]+'/'
            rows=by_task[task]
            indices=([f for r in rows for f in r['frame_rows']] if role=='sensor' else [r['sequence_row'] for r in rows])
            for key in ('.zgroup','.zattrs',*(f+'/.zarray' for f in fields)):
                path=prefix+key
                value=_object(checked_read(root,path,candidates[path],reads));files[path]=candidates[path]
                if key=='.zgroup' and value!={'zarr_format':2}:raise ValueError('group format drift')
                if key=='.zattrs' and any(value.get(k)!=source[s] for k,s in
                    [('parent_id','parent_id'),('partition','partition'),('geometry_realization','variant')]):
                    raise ValueError('stored source identity drift')
                if key.endswith('/.zarray'):
                    field=key.split('/')[0];plan=plan_array_access(value,field,role,indices)
                    bound='source_frame_count' if role=='sensor' else 'source_sequence_count'
                    if any(r[bound]!=plan['shape'][0] for r in rows):raise ValueError('source shape drift')
                    plans[prefix+field]=plan
                    for chunk in plan['chunk_keys']:
                        p=prefix+field+'/'+chunk;files[p]=candidates[p]
    for path,sha in list(reads.items()):checked_read(root,path,sha,{})
    return dict(schema='gse_supplement_input_scope_v1',status='PREPARATION_NOT_EXECUTION_AUTHORITY',
        population=population,task_sources=tasks,file_sha256=files,array_access=plans,
        metadata_reads_sha256=reads,counts=dict(population['counts'],tasks=len(tasks),
            chunk_files=sum(p['decoded_chunk_count'] for p in plans.values()),
            decoded_padded_bytes=sum(p['decoded_padded_bytes'] for p in plans.values()),
            actual_scan_motion_chunk_reads=0),
        restrictions=['six_fields_only','no_absolute_pose_or_labels_in_student','no_C08_C10',
                      'reuse_original3360','no_training','history_overlap_not_independent_population'])
