"""Single archived S07 case, compiled from sealed contract metadata only."""
import json
from copy import deepcopy
from pathlib import Path
from mtare_topo.governance_surface_material import read_pinned

RUN='results/gate3_semantics/gate3_20260908_gse_v8_original_ten_probe_v1r_seed20260906'
SEAL_SHA='81e5dd95eb086f4bf6b5c9185dc8081bf8e0cf79013b7823027f2ff370cb0676'
TASK='S07_flat_loop_rich_C03__rounded_rectangle'
SEQUENCE=97000


def compile_scope(root):
    root=Path(root).resolve(strict=True);seal_path=RUN+'/artifacts/evidence_sha256.txt'
    seal=read_pinned(root,seal_path,SEAL_SHA)
    pins={p:h for h,p in (line.split('  ',1) for line in seal.decode().splitlines())}
    card_path=RUN+'/config/data_card.json'
    old=json.loads(read_pinned(root,card_path,pins[card_path]))['scope']
    rows=[r for r in old['entries'] if (r['source']['task'],r['source']['source_sequence_id'])==(TASK,SEQUENCE)]
    if len(rows)!=1 or rows[0]['source']['frame_rows']!=[3084,3085,3086,3087,3088]:
        raise ValueError('exact original S07 five frames required')
    row=deepcopy(rows[0]);needed={row[k] for k in ('construction_path','codebook_path','input_path')}
    for plan in row['arrays'].values():
        needed.update(plan['prefix']+'/'+k for k in ['.zarray',*plan['chunk_keys']])
    files={p:old['file_sha256'][p] for p in sorted(needed)}
    files.update({pin['path']:pin['sha256'] for pin in row['references']})
    baseline=RUN+'/artifacts/'+TASK+'_'+str(SEQUENCE)+'.json.gz'
    files[baseline]=pins[baseline]
    return dict(schema='gse_s07_precision_scope_v1',entries=[row],file_sha256=files,
        metadata_reads_sha256={seal_path:SEAL_SHA,card_path:pins[card_path]},
        baseline=dict(path=baseline,sha256=pins[baseline]),
        geometry_settings=old['geometry_settings'],
        counts=dict(parents=1,tasks=1,physical_traversals=1,observations=1,frame_occurrences=5,
            unique_variant_frames=5,container_observations=row['container_observations']),
        sampling='Only the already diagnosed S07 error; not representative evaluation.',
        split='C03 fit development only; no C08-C10 payloads, training or calibration.',
        spacing='Original rows 3084-3088 and their original motion; no resampling.',
        collateral='Sealed card contains other-case metadata; NPZ/array collateral specified by the single entry. Only this observation is evaluated.')


def reader(root,scope):
    from .gse_v8_probe_reader import V8ProbeReader
    class SingleReader(V8ProbeReader):
        def __init__(self):
            self.root=Path(root).resolve(strict=True)
            if compile_scope(self.root)!=scope:raise ValueError('single S07 source drift')
            self.scope=deepcopy(scope)
            self.entries={(TASK,SEQUENCE):self.scope['entries'][0]}
            self.opened=dict(scope['metadata_reads_sha256']);self.completed=set()
    return SingleReader()
