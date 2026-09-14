"""Once-only cached source-affinity target export; no scans, fitting or GT forward."""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse, hashlib, json, resource, time, traceback, zipfile
from ai_junction_pilot import sha, write

NAME='gse_new12_affinity_targets_v1'
CARD=f'configs/v3/gate3/data_cards/{NAME}.json'
SPEC=f'configs/v3/gate3/{NAME}.json'
RUN=f'results/gate3_semantics/gate3_20260911_{NAME}_seed0'
FEATURE='results/gate3_semantics/gate3_20260911_gse_new12_features_v1_seed0'
SOURCE='results/gate3_semantics/gate3_20260911_gse_new12_surface_residual_v1_seed0'
BIND='configs/v3/gate3/gse_original_surface_source_binding_v1.json'
PYTHON='/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python'


def freeze():
    original=json.loads((ROOT/BIND).read_text())
    features=json.loads((ROOT/'configs/v3/gate3/data_cards/gse_new12_features_v1.json').read_text())
    s=dict(entries=features['scope']['entries'],parents=12,frames=60,roi_returns=446184,patches=10193,
           spacing=features['scope']['spacing'],selection_bias=features['scope']['selection_bias'],
           source_run=SOURCE,feature_run=FEATURE,producer_archive=original['archive_path'],
           source_rule='pinned nonempty face_sources subset active; singleton only',
           patch_rule='all original returns qualified singleton and same operand; otherwise unknown',
           target_semantics='same construction surface operand, NOT same junction or connected tunnel',
           training_steps=0,raw_sensor_reads=0,structural_membership=False,
           limits=dict(wall_seconds=300,host_bytes=4*1024**3,output_bytes=64*1024**2))
    approval=dict(status='APPROVED',approved_by='user-standing-new-route-authorization',approved_at='2026-09-11',
        authorized_gates=[3],authorized_operations=['data_export'],
        scope_sha256=hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest(),
        confirmation_reference='Active user goal explicitly authorizes fixed new12 pure/mixed/unknown supervision and feature binding; standing execution authorization',
        scope='Same fixed new12 cached local operand target binding only')
    write(ROOT/CARD,dict(schema_version='gse_new12_affinity_targets_card_v1',scope=s,approval=approval))
    files={BIND:sha(ROOT/BIND),CARD:sha(ROOT/CARD),original['archive_path']:original['archive_sha256']}
    for base in (SOURCE,FEATURE):
        if json.loads((ROOT/base/'RUN_STATE.json').read_text())['state']!='COMPLETED':raise ValueError('input incomplete')
        seal=ROOT/base/'artifacts/evidence_sha256.txt'
        sealed={p:h for h,p in (line.split('  ',1) for line in seal.read_text().splitlines())}
        files[str(seal.relative_to(ROOT))]=sha(seal)
        for i in range(12):
            p=f'{base}/artifacts/window_{i:02d}.npz'
            if sha(ROOT/p)!=sealed[p]:raise ValueError('cache seal mismatch')
            files[p]=sealed[p]
    sources={str(p.relative_to(ROOT)):sha(p) for folder in ('src/mtare_topo','tools/v3') for p in (ROOT/folder).rglob('*.py')}
    write(ROOT/SPEC,dict(schema_version='v3_run_spec_v1',gate=3,date='20260911',slug=NAME,seed=0,operation='data_export',
        data_card=CARD,user_authorization=approval,
        command=['env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','PYTHONPATH=src',PYTHON,'tools/v3/export_new12_affinity_targets.py','--execute'],
        question='Which original source-pure patches have source-affinity supervision in the fixed new12?',
        method=s['patch_rule'],baseline='All A/B/C consume identical observed patches; target export is not a comparison',
        fallback='Preserve mixed, multi-source and missing as unknown; seal any failure, no patch removal',
        acceptance_criteria=['12 inventories match by original ray identity','10193 patches retained','No nearest-source or score-based labels','Unknown never becomes background'],
        expected_evidence=['12 target arrays with reasons and masks, directed/unique pair counts, logs, source snapshot, seal'],
        estimated_cost=dict(compute='CPU cached target binding, zero training/GPU',host_ram_gb=4,gpu_vram_gb=0,disk_gb=.063,wall_time_hours=.084),
        input_sha256=files,source_sha256=sources))


def execute():
    import numpy as np
    from collections import Counter
    from mtare_topo.governance_new12_targets import validate_card
    from mtare_topo.data.gse_surface_source_join import join_surface_sources
    from mtare_topo.data.gse_singleton_source_evidence import singleton_return_evidence
    from mtare_topo.data.gse_surface_affinity_targets import bind_pure_patch_targets
    from mtare_topo.representation.gse_surface_patches_v1 import SurfacePatches,build_patch_neighbors
    run=ROOT/RUN;spec=json.loads((ROOT/SPEC).read_text());card=json.loads((ROOT/CARD).read_text());s=card['scope']
    if not validate_card(card).passed or json.loads((run/'RUN_STATE.json').read_text())['state']!='CREATED_NOT_EXECUTED':raise ValueError('fresh valid run required')
    start=time.monotonic();rows=[];error=None
    write(run/'RUN_STATE.json',dict(state='RUNNING'),'w')
    try:
        for p,h in {**spec['input_sha256'],**spec['source_sha256']}.items():
            if sha(ROOT/p)!=h:raise ValueError('bound input drift '+p)
        with zipfile.ZipFile(run/'artifacts/source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
            for p in spec['source_sha256']:z.write(ROOT/p,p)
        write(run/'config/runtime_environment.json',dict(numpy=np.__version__,python=__import__('sys').version))
        for i in range(12):
            with np.load(ROOT/SOURCE/f'artifacts/window_{i:02d}.npz',allow_pickle=False) as source, np.load(ROOT/FEATURE/f'artifacts/window_{i:02d}.npz',allow_pickle=False) as feature:
                joined=join_surface_sources(source['records'],source['roi_return_indices'],feature['surface_return_indices'])
                qualified=singleton_return_evidence(joined,producer_archive=ROOT/s['producer_archive'])
                patches=SurfacePatches(**{k:feature['patch_'+k] for k in SurfacePatches.__dataclass_fields__ if k not in ('voxel_size_m','roi_radius_m')},voxel_size_m=.5,roi_radius_m=10.)
                m=len(patches.centers_m);neighbors=build_patch_neighbors(patches).neighbor_index
                assigned=patches.point_patch_index[joined.ray_indices]
                if np.any(assigned<0) or np.any(assigned>=m):raise ValueError('unassigned ROI return')
                point_lists=[np.flatnonzero(assigned==p) for p in range(m)]
                if not np.array_equal(np.array([len(p) for p in point_lists]),patches.point_count):raise ValueError('point inventory drift')
                targets=bind_pure_patch_targets(joined.candidates,qualified,point_lists,neighbors)
                pos=int(np.sum(targets.known&(targets.values==1)));neg=int(np.sum(targets.known&(targets.values==0)))
                unique={tuple(sorted((p,int(neighbors[p,k])))) for p,k in zip(*np.nonzero(targets.known))}
                np.savez_compressed(run/f'artifacts/window_{i:02d}.npz',values=targets.values,known=targets.known,pure_patch=targets.pure_patch,
                    patch_reason=np.array(targets.patch_reason),neighbor_index=neighbors,original_ray_indices=joined.ray_indices,return_source_known=qualified)
                row=dict(observation=i,task=s['entries'][i]['task'],returns=len(joined.ray_indices),patches=m,
                    pure_patches=int(targets.pure_patch.sum()),reasons=dict(Counter(targets.patch_reason)),
                    positive_directed=pos,negative_directed=neg,unknown_directed=int(np.sum((neighbors>=0)&~targets.known)),known_unique_pairs=len(unique))
                rows.append(row);print(json.dumps(row),flush=True)
            if time.monotonic()-start>s['limits']['wall_seconds']:raise TimeoutError('wall cap')
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>s['limits']['host_bytes']:raise MemoryError('host cap')
            if sum(p.stat().st_size for p in run.rglob('*') if p.is_file())>s['limits']['output_bytes']:raise MemoryError('output cap')
        if sum(r['patches'] for r in rows)!=10193 or sum(r['returns'] for r in rows)!=446184:raise ValueError('population drift')
    except BaseException:error=traceback.format_exc()
    summary=dict(status='GATE_FAIL' if error else 'GATE_MIXED',error=error,completed=len(rows),windows=rows,
        training_steps=0,structural_membership_qualified=False,elapsed_s=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    write(run/'metrics/summary.json',summary);write(run/'logs/raw.json',summary)
    write(run/'RUN_STATE.json',dict(state='FAILED' if error else 'COMPLETED'),'w')
    with (run/'artifacts/evidence_sha256.txt').open('x') as f:
        for p in sorted(run.rglob('*')):
            if p.is_file() and p.name!='evidence_sha256.txt':f.write(sha(p)+'  '+str(p.relative_to(ROOT))+'\n')
    print(json.dumps(summary));return int(error is not None)


if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--freeze',action='store_true');g.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.freeze:freeze()
    else:raise SystemExit(execute())
