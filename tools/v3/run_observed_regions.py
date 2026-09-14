"""One fixed full-five-frame observation-only component diagnostic."""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse,hashlib,json,time,traceback,signal,resource,sys,shutil
from ai_junction_pilot import sha,write
NAME='gse_multi_ray_regions_v1';CARD='configs/v3/gate3/data_cards/'+NAME+'.json';SPEC='configs/v3/gate3/'+NAME+'.json'
RUN='results/gate3_semantics/gate3_20260911_'+NAME+'_seed0'
PYTHON='/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python'
REF='results/gate3_semantics/gate3_20260911_gse_new_fit_reference_review_v1_seed0/artifacts/comparison.json'

def freeze():
    old=json.loads((ROOT/'configs/v3/gate3/data_cards/gse_ai_new_fit_review_v1.json').read_text())
    e=old['scope']['entries'][11];bundle=next(p for p in old['scope']['indexed_bundles'] if p.endswith('/case_11.json'))
    scope=dict(entry=e,bundle=bundle,observations=1,parents=1,frames=5,raw_ray_slots=57600,valid_returns=56792,
        spacing=old['scope']['spacing'],bias=old['scope']['bias'],resolution_m=.25,radius_m=10,
        inputs={e['student_path']:e['student_sha256'],bundle:old['scope']['indexed_bundles'][bundle],REF:sha(ROOT/REF)},
        references_only_after_component_freeze=True,training=False,new_labels=0,unknown_is_background=False,
        limits=dict(wall_seconds=600,host_bytes=4*1024**3,output_bytes=100*1024**2))
    approval=dict(status='APPROVED',authorized_operations=['audit'],authorized_gates=[3],approved_by='user',approved_at='2026-09-11',
        confirmation_reference='User explicitly: 全部都允许按照最优方案去推进, in reply to multi-ray/multi-frame supervision-scope question',
        scope='One original case11 full-observation proxy diagnostic; no labels or training',scope_sha256=hashlib.sha256(json.dumps(scope,sort_keys=True).encode()).hexdigest())
    write(ROOT/CARD,dict(schema_version='gse_observed_regions_card_v1',scope=scope,approval=approval))
    sources=['tools/v3/run_observed_regions.py','tools/v3/ai_junction_pilot.py','src/mtare_topo/representation/gse_observed_regions.py',
        'src/mtare_topo/representation/gse_surface_ray_evidence_v1.py','src/mtare_topo/governance_membership_fit.py','src/mtare_topo/governance.py']
    write(ROOT/SPEC,dict(schema_version='v3_run_spec_v1',gate=3,date='20260911',slug=NAME,seed=0,operation='audit',data_card=CARD,user_authorization=approval,
        command=['env','OPENBLAS_NUM_THREADS=1','MPLCONFIGDIR=/tmp/gse-mpl',PYTHON,'tools/v3/run_observed_regions.py','--execute'],
        question='Can existing five-frame observations support3D multi-ray region correspondence without construction IDs?',
        method='Existing .25m observed-ray grid, occupied precedence,6-neighbor full-in-sphere free cells; reference queries only after freeze',
        baseline='Single-ray rule cannot witness case11 full relation; component result only an observation proxy, not method advantage',
        fallback='Record unknown/disconnected/overconnected outcomes; no grid tuning, labels, teacher or training',
        estimated_cost=dict(compute='CPU56792 valid rays, fixed80cubed grid',host_ram_gb=4,gpu_vram_gb=0,disk_gb=.1,wall_time_hours=1/6),
        acceptance_criteria=['Exact original frames and all valid rays','Components frozen before reference queries','Unknown/occupied/ROI semantics unchanged; no label qualification'],
        expected_evidence=['Grid, component arrays, coordinate queries, fixed slices, logs,source hashes,seal'],input_sha256={CARD:sha(ROOT/CARD),**scope['inputs']},source_sha256={p:sha(ROOT/p) for p in sources}))

def execute():
    import numpy as np
    from mtare_topo.representation.gse_surface_ray_evidence_v1 import build_surface_ray_grid
    from mtare_topo.representation.gse_observed_regions import build_observed_regions,query_observed_pair
    run=ROOT/RUN;s=json.loads((ROOT/SPEC).read_text());c=json.loads((ROOT/CARD).read_text());scope=c['scope'];start=time.monotonic();error=None;metrics={}
    if json.loads((run/'RUN_STATE.json').read_text())['state']!='CREATED_NOT_EXECUTED':raise ValueError('fresh run required')
    write(run/'RUN_STATE.json',dict(state='RUNNING'),'w')
    def timeout(*args):raise TimeoutError('600s wall cap')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(600)
    try:
        for p,h in {**s['input_sha256'],**s['source_sha256']}.items():
            if sha(ROOT/p)!=h:raise ValueError('drift '+p)
        for p in s['source_sha256']:shutil.copyfile(ROOT/p,run/'artifacts'/__import__('pathlib').Path(p).name)
        resource.setrlimit(resource.RLIMIT_AS,(scope['limits']['host_bytes'],scope['limits']['host_bytes']))
        b=json.loads((ROOT/scope['bundle']).read_text())
        with np.load(ROOT/scope['entry']['student_path'],allow_pickle=False) as a:
            valid=a['valid_mask'].reshape(-1).astype(bool);translations=a['relative_translation_current_sensor_m'].astype(np.float32)
        if int(valid.sum())!=scope['valid_returns']:raise ValueError('valid count drift')
        if b['source_frame_indices']!=scope['entry']['frame_rows']:raise ValueError('frame mismatch')
        ids=np.asarray(b['point_history_slots'])*11520+np.asarray(b['point_ray_indices'])
        if not np.array_equal(np.sort(ids),np.flatnonzero(valid)):raise ValueError('indexed returns differ')
        origins=np.repeat(translations,11520,axis=0);ends=origins.copy();ends[ids]=np.asarray(b['points_xyz_m'],dtype=np.float32)
        frames=np.repeat(np.arange(5),11520)
        print('Building full causal observation grid; no reference payload read',flush=True)
        grid=build_surface_ray_grid(origins,ends,valid,frames);regions=build_observed_regions(grid)
        np.savez_compressed(run/'artifacts/observed_grid.npz',state=grid.state,free_frame_bits=grid.free_frame_bits,
            occupied_frame_bits=grid.occupied_frame_bits,labels=regions.labels,sizes=np.asarray(regions.sizes),frame_bits=np.asarray(regions.frame_bits))
        frozen=sha(run/'artifacts/observed_grid.npz')
        print('Components saved and hashed; now reading reference coordinates for diagnostic only',flush=True)
        r=json.loads((ROOT/REF).read_text())['cases'][11];queries=[]
        for oi,o in enumerate(r['openings']):
            for ai,a in enumerate(r['anchors']):
                queries.append(dict(opening=oi,anchor=ai,old_membership=r['membership'][oi][ai],
                    result=query_observed_pair(regions,grid,o['position_m'],a['position_m'])))
        if sha(run/'artifacts/observed_grid.npz')!=frozen:raise ValueError('reference altered components')
        write(run/'artifacts/reference_queries.json',dict(grid_file_sha256=frozen,queries=queries,
            caveat='Window sections lie on10m boundary; no nearest-free-cell snapping, so endpoint may remain unknown'))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axs=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
        axs[0].imshow(grid.state[:,:,40].T,origin='lower',extent=(-10,10,-10,10),vmin=0,vmax=2);axs[0].set_title('z=0.125m: unknown/free/occupied')
        axs[1].imshow(regions.labels[:,:,40].T,origin='lower',extent=(-10,10,-10,10),cmap='tab20');axs[1].set_title('Same slice: observed components')
        axs[2].imshow(grid.state[:,40,:].T,origin='lower',extent=(-10,10,-10,10),vmin=0,vmax=2);axs[2].set_title('y=0.125m: vertical evidence')
        for ax in axs:ax.set_xlabel('X m')
        fig.suptitle('Case11 observation proxy, NOT predicted structures or robot traversability')
        fig.savefig(run/'previews/observed_regions.png',dpi=130);plt.close(fig)
        metrics=dict(valid_rays=int(valid.sum()),free_cells=int((grid.state==1).sum()),occupied_cells=int((grid.state==2).sum()),
            region_count=len(regions.sizes),region_cells=sum(regions.sizes),largest_region_cells=max(regions.sizes,default=0),
            grid_file_sha256=frozen,queries=queries,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        if sum(p.stat().st_size for p in run.rglob('*') if p.is_file())>scope['limits']['output_bytes']:raise MemoryError('output cap')
    except BaseException:error=traceback.format_exc()
    finally:signal.alarm(0)
    summary=dict(status='GATE_FAIL' if error else 'GATE_MIXED',error=error,elapsed_s=time.monotonic()-start,**metrics,training_steps=0,new_labels=0,physical_connectivity=False)
    write(run/'metrics/summary.json',summary);write(run/'logs/raw.json',summary);write(run/'RUN_STATE.json',dict(state='FAILED' if error else 'COMPLETED'),'w')
    with (run/'artifacts/evidence_sha256.txt').open('x') as f:
        for p in sorted(run.rglob('*')):
            if p.is_file() and p.name!='evidence_sha256.txt':f.write(sha(p)+'  '+str(p.relative_to(ROOT))+'\n')
    print(json.dumps(summary));return int(error is not None)

if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--freeze',action='store_true');g.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.freeze:freeze()
    else:raise SystemExit(execute())
