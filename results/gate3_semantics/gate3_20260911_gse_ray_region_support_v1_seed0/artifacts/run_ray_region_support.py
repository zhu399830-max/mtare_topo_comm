"""Bind all original case11 rays to the already sealed grid; no teacher calls."""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse, hashlib, json, resource, signal, time, traceback, shutil
from pathlib import Path
from ai_junction_pilot import sha, write

NAME='gse_ray_region_support_v1'
CARD=f'configs/v3/gate3/data_cards/{NAME}.json'
SPEC=f'configs/v3/gate3/{NAME}.json'
RUN=f'results/gate3_semantics/gate3_20260911_{NAME}_seed0'
OLD='results/gate3_semantics/gate3_20260911_gse_multi_ray_regions_v1_seed0'


def freeze():
    old=json.loads((ROOT/OLD/'config/run_spec.json').read_text())
    card=json.loads((ROOT/OLD/'config/data_card.json').read_text())
    s=card['scope'];s['inputs'].pop(next(p for p in s['inputs'] if p.endswith('comparison.json')))
    s['inputs'][OLD+'/artifacts/observed_grid.npz']=sha(ROOT/OLD/'artifacts/observed_grid.npz')
    s['reuse_frozen_grid']=True;s['reference_payload_reads']=0
    a=card['approval'];a['scope']='Same case11 all original ray support binding to sealed regions; no reference reads, labels or training'
    a['scope_sha256']=hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest()
    write(ROOT/CARD,card)
    old.update(slug=NAME,data_card=CARD,user_authorization=a,
        question='Can each original ray preserve free-region and actual surface support without center snapping or structural labels?',
        method='Reuse frozen grid; authenticated original ray DDA with frame bits; separately retain surface returns',
        fallback='Keep empty/unknown support and no structural membership; do not change the grid',
        acceptance_criteria=['All56792 valid rays retained with exact IDs','Original grid unchanged','Surface and free support disjoint','No reference payload or membership labels'],
        expected_evidence=['All ray support arrays, counts, source snapshots, runtime, logs and seal'])
    old['command'][-2]='tools/v3/run_ray_region_support.py'
    old['input_sha256']={CARD:sha(ROOT/CARD),**s['inputs']}
    sources=['tools/v3/run_ray_region_support.py','tools/v3/ai_junction_pilot.py',
        'src/mtare_topo/representation/gse_ray_region_support.py',
        'src/mtare_topo/representation/gse_observed_regions.py',
        'src/mtare_topo/representation/gse_surface_ray_evidence_v1.py',
        'src/mtare_topo/governance.py','src/mtare_topo/governance_membership_fit.py']
    old['source_sha256']={p:sha(ROOT/p) for p in sources}
    write(ROOT/SPEC,old)


def execute():
    import numpy as np
    from mtare_topo.representation.gse_surface_ray_evidence_v1 import ObservedRayGrid,_digest,_numerical_bound
    from mtare_topo.representation.gse_observed_regions import ObservedRegions
    from mtare_topo.representation.gse_ray_region_support import bind_ray_support
    run=ROOT/RUN;spec=json.loads((ROOT/SPEC).read_text());scope=json.loads((ROOT/CARD).read_text())['scope']
    if json.loads((run/'RUN_STATE.json').read_text())['state']!='CREATED_NOT_EXECUTED':
        raise ValueError('fresh run required')
    write(run/'RUN_STATE.json',{'state':'RUNNING'},'w');start=time.monotonic();error=None;metrics={}
    def timeout(*_):raise TimeoutError('600s cap')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(600)
    try:
        for p,h in {**spec['input_sha256'],**spec['source_sha256']}.items():
            if sha(ROOT/p)!=h:raise ValueError('input/source drift '+p)
        for p in spec['source_sha256']:shutil.copyfile(ROOT/p,run/'artifacts'/Path(p).name)
        resource.setrlimit(resource.RLIMIT_AS,(scope['limits']['host_bytes'],)*2)
        bundle=json.loads((ROOT/scope['bundle']).read_text())
        with np.load(ROOT/scope['entry']['student_path'],allow_pickle=False) as f:
            valid=f['valid_mask'].reshape(-1).astype(bool)
            origins=np.repeat(f['relative_translation_current_sensor_m'].astype(np.float32),11520,axis=0)
        ids=np.asarray(bundle['point_history_slots'])*11520+np.asarray(bundle['point_ray_indices'])
        if (valid.sum()!=56792 or not np.array_equal(np.sort(ids),np.flatnonzero(valid))
                or bundle['source_frame_indices']!=scope['entry']['frame_rows']):raise ValueError('identity mismatch')
        ends=origins.copy();ends[ids]=np.asarray(bundle['points_xyz_m'],dtype=np.float32)
        frames=np.repeat(np.arange(5,dtype=np.int64),11520)
        tol=_numerical_bound(origins[valid],ends[valid])
        source=_digest(np.where(valid[:,None],origins,0.).astype('<f8'),
            np.where(valid[:,None],ends,0.).astype('<f8'),valid,frames.astype('<i8'),
            np.asarray([origins.dtype.str,ends.dtype.str],dtype='S3'))
        with np.load(ROOT/OLD/'artifacts/observed_grid.npz',allow_pickle=False) as f:
            state=f['state'];free=f['free_frame_bits'];occ=f['occupied_frame_bits'];labels=f['labels']
            sizes=tuple(map(int,f['sizes']));bits=tuple(map(int,f['frame_bits']))
        digest=_digest(state,free,occ,np.asarray([tol],dtype='<f8'))
        # Metadata reconstructed from bound original rays; voxel evidence is NOT rebuilt.
        grid=ObservedRayGrid(state,free,occ,source,digest,tol,int(valid.sum()),int((~valid).sum()),-1)
        regions=ObservedRegions(labels,sizes,bits,digest)
        print('Binding all valid original rays to sealed grid; no reference payload',flush=True)
        records=bind_ray_support(grid,regions,origins,ends,valid,frames)
        if len(records)!=56792:raise ValueError('record count')
        arrays={'ray_index':np.asarray([r.ray_index for r in records],dtype=np.int32),
                'frame_slot':np.asarray([r.frame_slot for r in records],dtype=np.int8)}
        for name in ('free_cells','surface_cells','components'):
            sizes_array=np.asarray([len(getattr(r,name)) for r in records],dtype=np.int64)
            arrays[name+'_offsets']=np.r_[0,np.cumsum(sizes_array)]
            arrays[name]=np.fromiter((x for r in records for x in getattr(r,name)),dtype=np.int32)
        np.savez_compressed(run/'artifacts/ray_support.npz',**arrays)
        for p,h in scope['inputs'].items():
            if sha(ROOT/p)!=h:raise ValueError('input changed during binding')
        metrics=dict(records=len(records),with_free_support=sum(bool(r.free_cells) for r in records),
            with_surface_support=sum(bool(r.surface_cells) for r in records),
            both=sum(bool(r.free_cells and r.surface_cells) for r in records),
            with_multiple_components=sum(len(r.components)>1 for r in records),
            per_frame=[sum(r.frame_slot==i for r in records) for i in range(5)],
            source_sha256=source,grid_content_sha256=digest,
            free_cell_references=len(arrays['free_cells']),surface_cell_references=len(arrays['surface_cells']),
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        if sum(p.stat().st_size for p in run.rglob('*') if p.is_file())>scope['limits']['output_bytes']:
            raise MemoryError('output cap')
    except BaseException:error=traceback.format_exc()
    finally:signal.alarm(0)
    summary=dict(status='GATE_FAIL' if error else 'GATE_MIXED',error=error,elapsed_s=time.monotonic()-start,
        **metrics,reference_payload_reads=0,new_labels=0,training_steps=0,training_qualified=False,
        rebuilt_grid=False,ambiguity_count_metadata_reconstructed=False)
    write(run/'metrics/summary.json',summary);write(run/'logs/raw.json',summary)
    write(run/'RUN_STATE.json',{'state':'FAILED' if error else 'COMPLETED'},'w')
    with (run/'artifacts/evidence_sha256.txt').open('x') as f:
        for p in sorted(run.rglob('*')):
            if p.is_file() and p.name!='evidence_sha256.txt':f.write(sha(p)+'  '+str(p.relative_to(ROOT))+'\n')
    print(json.dumps(summary));return int(error is not None)


if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--freeze',action='store_true');g.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.freeze:freeze()
    else:raise SystemExit(execute())
