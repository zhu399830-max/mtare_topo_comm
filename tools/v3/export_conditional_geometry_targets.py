"""One frozen export of revised construction-conditioned geometry references."""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse, hashlib, json, resource, signal, time, traceback, zipfile
from ai_junction_pilot import sha, write
from export_new12_affinity_targets import FEATURE, PYTHON
NAME = 'gse_new12_conditional_geometry_v1'
CARD = f'configs/v3/gate3/data_cards/{NAME}.json'
SPEC = f'configs/v3/gate3/{NAME}.json'
RUN = f'results/gate3_semantics/gate3_20260911_{NAME}_seed0'
REFERENCE = 'results/gate3_semantics/gate3_20260911_gse_new12_interior_support_v1_seed0'
AFFINITY = 'results/gate3_semantics/gate3_20260911_gse_new12_affinity_targets_v1_seed0'


def freeze():
    old = json.loads((ROOT/'configs/v3/gate3/data_cards/gse_new12_affinity_targets_v1.json').read_text())['scope']
    scope = dict(entries=old['entries'], parents=12, frames=60, roi_returns=446184, patches=10193,
        spacing=old['spacing'], selection_bias=old['selection_bias'], feature_run=FEATURE, reference_run=REFERENCE,
        target_schema='construction_conditioned_geometry_targets_v1',
        target_fields=['axis_abs_dot', 'reference_center_height_difference_m'],
        observability_certified=False, connectivity_certified=False, training_steps=0, raw_sensor_reads=0,
        sampling_unit='12 fit parents / 12 windows; patch pairs and source components are not independent places',
        source='sealed original mesh midpoint references and original singleton-source arc component returns',
        membership='all ROI returns of patch uniquely in one saved component; ambiguity/mixed/missing unknown',
        label_meaning='conditional construction reference, not independently observed channel identity or floor height',
        inference='all patches unchanged; target identities restricted to loss and independent scoring',
        limits=dict(wall_seconds=300, host_bytes=4*1024**3, output_bytes=128*1024**2))
    approval = dict(status='APPROVED', approved_by='user-explicit-supervision-contract-confirmation', approved_at='2026-09-11',
        authorized_operations=['data_export'], authorized_gates=[3],
        scope_sha256=hashlib.sha256(json.dumps(scope, sort_keys=True).encode()).hexdigest(),
        confirmation_reference='User explicitly confirmed construction-conditioned statistical supervision; standing execution authorization; same fixed12 development population',
        scope='Same12 cached reference target export, no training or new scenes')
    pins = {}
    for base, names in ((FEATURE, [f'window_{i:02d}.npz' for i in range(12)]),
                        (AFFINITY, [f'window_{i:02d}.npz' for i in range(12)]),
                        (REFERENCE, [f'case_{i:02d}.json' for i in range(12)])):
        if json.loads((ROOT/base/'RUN_STATE.json').read_text())['state'] != 'COMPLETED': raise ValueError('input incomplete')
        seal = ROOT/base/'artifacts/evidence_sha256.txt'
        sealed = {p:h for h,p in (l.split('  ',1) for l in seal.read_text().splitlines())}
        pins[str(seal.relative_to(ROOT))] = sha(seal)
        for name in names:
            p = f'{base}/artifacts/{name}'
            if sha(ROOT/p) != sealed[p]: raise ValueError('input drift '+p)
            pins[p] = sealed[p]
    write(ROOT/CARD, dict(schema_version='gse_conditional_geometry_card_v1', scope=scope, approval=approval))
    pins[CARD] = sha(ROOT/CARD)
    sources = {str(p.relative_to(ROOT)):sha(p) for folder in ('src/mtare_topo','tools/v3') for p in (ROOT/folder).rglob('*.py')}
    write(ROOT/SPEC, dict(schema_version='v3_run_spec_v1', gate=3, date='20260911', slug=NAME, seed=0,
        operation='data_export', data_card=CARD, user_authorization=approval,
        command=['env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','PYTHONPATH=src',PYTHON,'tools/v3/export_conditional_geometry_targets.py','--execute'],
        question='What conditional construction geometry targets exist on fixed12 cached patch pairs?',
        method=scope['membership'], baseline='Constant axis dot1 and height difference0, separately on cross-component pairs',
        fallback='Unknown remains unknown; preserve old reference statuses; no re-rendering, reference relocation or invented correspondence',
        acceptance_criteria=['All12 parents and10193 patches retained','Conditional and observable semantics never conflated','Each target bound to saved reference and original return indices','Same/cross component and parent counts reported'],
        expected_evidence=['12 target arrays, reference mapping, counts and constant baseline errors, code snapshot, environment, raw logs, seal'],
        estimated_cost=dict(compute='CPU cached export, no GPU/training', host_ram_gb=4, gpu_vram_gb=0, disk_gb=.125, wall_time_hours=.084),
        input_sha256=pins, source_sha256=sources))


def execute():
    import numpy as np
    from collections import Counter
    from mtare_topo.governance_conditional_geometry import validate_card
    from mtare_topo.teacher.gse_conditional_geometry_targets import conditional_geometry_targets
    run = ROOT/RUN; spec = json.loads((ROOT/SPEC).read_text()); card = json.loads((ROOT/CARD).read_text())
    if not validate_card(card).passed or json.loads((run/'RUN_STATE.json').read_text())['state'] != 'CREATED_NOT_EXECUTED':
        raise ValueError('fresh valid run required')
    start = time.monotonic(); rows = []; error = None
    write(run/'RUN_STATE.json', dict(state='RUNNING'), 'w')
    def expire(*_): raise TimeoutError('300s wall limit')
    signal.signal(signal.SIGALRM, expire); signal.alarm(300)
    try:
        for p,h in {**spec['input_sha256'], **spec['source_sha256']}.items():
            if sha(ROOT/p) != h: raise ValueError('bound input drift '+p)
        resource.setrlimit(resource.RLIMIT_AS, (4*1024**3,)*2)
        with zipfile.ZipFile(run/'artifacts/source_snapshot.zip', 'x', compression=zipfile.ZIP_DEFLATED) as z:
            for p in spec['source_sha256']: z.write(ROOT/p,p)
        write(run/'config/runtime_environment.json', dict(python=__import__('sys').version, numpy=np.__version__, target_semantics=card['scope']['label_meaning']))
        for i in range(12):
            reference = json.loads((ROOT/REFERENCE/f'artifacts/case_{i:02d}.json').read_text())
            with np.load(ROOT/FEATURE/f'artifacts/window_{i:02d}.npz',allow_pickle=False) as f, np.load(ROOT/AFFINITY/f'artifacts/window_{i:02d}.npz',allow_pickle=False) as a:
                roi = f['surface_return_indices']; neighbors = a['neighbor_index']
                if not np.array_equal(roi,a['original_ray_indices']): raise ValueError('ROI drift')
                out = conditional_geometry_targets(reference['candidates'], f['patch_point_patch_index'], roi, neighbors)
                if len(neighbors) != len(f['patch_centers_m']): raise ValueError('patch count drift')
                out['original_roi_indices'] = roi
                np.savez_compressed(run/f'artifacts/window_{i:02d}.npz', **out)
                known = out['axis_known']; cross = known & ~out['same_reference_component']
                comps = out['patch_reference_component']
                independent = {tuple(sorted((int(comps[p]),int(comps[neighbors[p,k]])))) for p,k in zip(*np.nonzero(cross))}
                baseline = {}
                for name, mask in [('all',known),('cross_component',cross)]:
                    baseline[name] = dict(count=int(mask.sum()), axis_mae=float(np.mean(1-out['axis_abs_dot'][mask])) if mask.any() else None,
                                          height_mae_m=float(np.mean(abs(out['height_difference_m'][mask]))) if mask.any() else None)
                row = dict(observation=i, parent=card['scope']['entries'][i]['parent'], patches=len(neighbors), roi_returns=len(roi),
                    known_directed=int(known.sum()), cross_component_directed=int(cross.sum()), independent_cross_component_pairs=len(independent),
                    unique_bound_components=int(len(set(comps[comps>=0]))), unknown_directed=int(((neighbors>=0)&~known).sum()),
                    patch_reasons=dict(Counter(out['patch_reason'].tolist())), constant_baseline=baseline)
                rows.append(row); print(json.dumps(row),flush=True)
                write(run/f'artifacts/reference_{i:02d}.json', dict(source=f'{REFERENCE}/artifacts/case_{i:02d}.json',
                    reference_components=[dict(index=j,source_index=c['source_index'],component_index=c['component_index'],
                    selected_slab_m=c['selected_slab_m'],center_m=c.get('center_m'),normal=c.get('normal')) for j,c in enumerate(reference['candidates'])],
                    observability_certified=False, semantic_channel_identity=False))
            if sum(p.stat().st_size for p in run.rglob('*') if p.is_file()) > 128*1024**2: raise RuntimeError('output limit')
        if sum(r['patches'] for r in rows)!=10193 or sum(r['roi_returns'] for r in rows)!=446184: raise ValueError('population drift')
    except BaseException: error=traceback.format_exc()
    finally: signal.alarm(0)
    summary = dict(status='GATE_FAIL' if error else 'GATE_MIXED', error=error, completed=len(rows), windows=rows,
        target_schema=card['scope']['target_schema'], observability_certified=False, training_steps=0,
        elapsed_s=time.monotonic()-start, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    write(run/'metrics/summary.json',summary); write(run/'logs/raw.json',summary)
    write(run/'RUN_STATE.json',dict(state='FAILED' if error else 'COMPLETED'),'w')
    with (run/'artifacts/evidence_sha256.txt').open('x') as f:
        for p in sorted(run.rglob('*')):
            if p.is_file() and p.name!='evidence_sha256.txt': f.write(sha(p)+'  '+str(p.relative_to(ROOT))+'\n')
    print(json.dumps(summary)); return int(error is not None)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); g=parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--freeze',action='store_true');g.add_argument('--execute',action='store_true');args=parser.parse_args()
    if args.freeze: freeze()
    else: raise SystemExit(execute())
