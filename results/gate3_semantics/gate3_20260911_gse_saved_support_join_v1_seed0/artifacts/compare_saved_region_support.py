"""Read-only support joins; no geometry extraction or reference relabeling."""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse,gzip,hashlib,json,time,traceback,shutil
from pathlib import Path
from ai_junction_pilot import sha,write
NAME='gse_saved_support_join_v1'
CARD=f'configs/v3/gate3/data_cards/{NAME}.json';SPEC=f'configs/v3/gate3/{NAME}.json'
RUN=f'results/gate3_semantics/gate3_20260911_{NAME}_seed0'
PREV='results/gate3_semantics/gate3_20260911_gse_ray_region_support_v1_seed0'
REF='results/gate3_semantics/gate3_20260910_gse_local_pair_support_pilot_v1_seed20260906/artifacts/observation_00028.json.gz'
SUPPORT=PREV+'/artifacts/ray_support.npz'


def freeze():
    c=json.loads((ROOT/PREV/'config/data_card.json').read_text())
    s=c['scope'];s['inputs']={p:sha(ROOT/p) for p in (REF,SUPPORT)}
    s['reference_payload_reads']=1;s['comparison_only']=True
    s['reference_groups']='2 window crossings and surfaces; 3 junction interfaces with original witness roles; no new targets'
    c['approval']['scope']='Same case11 frozen per-ray support and original reference witness indices; read-only comparison, no labels or training'
    c['approval']['scope_sha256']=hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest()
    write(ROOT/CARD,c)
    spec=json.loads((ROOT/PREV/'config/run_spec.json').read_text())
    spec.update(slug=NAME,data_card=CARD,user_authorization=c['approval'],
        question='Which original window/interface witnesses share rays, free cells or actual return surfaces?',
        method='Join exact saved ray IDs; retain original role and separate free/surface overlaps',
        baseline='Original same-ray correspondence and common-component ambiguity; not a learned method comparison',
        fallback='Report absent and ambiguous support; no overlap threshold or new structural labels',
        acceptance_criteria=['All reference ray IDs resolve in saved observation','No recomputed rays/grid','Overlaps not structural labels'],
        expected_evidence=['Group supports and complete pairwise counts, fixed visual comparison, logs and SHA seal'])
    spec['command'][-2]='tools/v3/compare_saved_region_support.py'
    spec['input_sha256']={CARD:sha(ROOT/CARD),**s['inputs']}
    spec['source_sha256']={p:sha(ROOT/p) for p in ['tools/v3/compare_saved_region_support.py','tools/v3/ai_junction_pilot.py']}
    write(ROOT/SPEC,spec)


def execute():
    import numpy as np
    run=ROOT/RUN;spec=json.loads((ROOT/SPEC).read_text());start=time.monotonic();error=None;result={}
    if json.loads((run/'RUN_STATE.json').read_text())['state']!='CREATED_NOT_EXECUTED':raise ValueError('fresh run required')
    write(run/'RUN_STATE.json',dict(state='RUNNING'),'w')
    try:
        for p,h in {**spec['input_sha256'],**spec['source_sha256']}.items():
            if sha(ROOT/p)!=h:raise ValueError('drift '+p)
        for p in spec['source_sha256']:shutil.copyfile(ROOT/p,run/'artifacts'/Path(p).name)
        with np.load(ROOT/SUPPORT,allow_pickle=False) as z:data={k:z[k] for k in z.files}
        lookup={int(v):i for i,v in enumerate(data['ray_index'])}
        with gzip.open(ROOT/REF,'rt') as f:prov=json.load(f)['produced_targets']['teacher_provenance']
        groups={}
        for i,o in enumerate(prov['openings']):
            for role in ('crossing','surface'):groups[f'window_{i}_{role}']=set(o[role+'_ray_indices'])
        for i in range(3):
            for role in ('witness','entering_witness','interior_witness','surface_entry_witness','surface_departure_witness'):
                groups[f'interface_{i}_{role}']=set(prov['anchors'][0][role+'_ray_indices'][i])
        cells={};counts={}
        for name,ids in groups.items():
            missing=ids-lookup.keys()
            if missing:raise ValueError(f'{name}: {len(missing)} invalid/missing ray IDs')
            cells[name]={}
            for kind in ('free_cells','surface_cells','components'):
                offsets=data[kind+'_offsets'];values=data[kind]
                cells[name][kind]=set(int(c) for rid in ids for c in values[offsets[lookup[rid]]:offsets[lookup[rid]+1]])
            counts[name]=dict(rays=len(ids),**{k:len(v) for k,v in cells[name].items()},
                per_frame=[sum(rid//11520==f for rid in ids) for f in range(5)])
        pairs=[]
        for a,ids in groups.items():
            for b,other in groups.items():
                pairs.append(dict(first=a,second=b,shared_rays=len(ids&other),
                    **{'shared_'+k:len(cells[a][k]&cells[b][k]) for k in ('free_cells','surface_cells','components')},
                    structural_membership=None))
        write(run/'artifacts/comparison.json',dict(groups=counts,pairs=pairs,new_labels=0))
        np.savez_compressed(run/'artifacts/group_support.npz',
            **{name+'__'+kind:np.asarray(sorted(values),dtype=np.int32)
               for name,kinds in cells.items() for kind,values in kinds.items()})
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        names=['window_0_crossing','window_1_crossing']+[f'interface_{i}_witness' for i in range(3)]
        fig,axes=plt.subplots(2,5,figsize=(16,7),constrained_layout=True)
        for j,name in enumerate(names):
            for i,kind in enumerate(('free_cells','surface_cells')):
                flat=np.asarray(sorted(cells[name][kind]),dtype=np.int64)
                mask=np.zeros((80,80),dtype=bool)
                mask[flat//6400,(flat//80)%80]=True
                axes[i,j].imshow(mask.T,origin='lower',extent=(-10,10,-10,10),vmin=0,vmax=1,cmap='Greys')
                axes[i,j].set_title(name+'\n'+kind,fontsize=9)
                axes[i,j].set_xlabel('X m');axes[i,j].set_ylabel('Y m')
        fig.suptitle('Saved observation support, XY projection over all heights; NOT predicted branches or membership')
        fig.savefig(run/'previews/support_groups.png',dpi=130);plt.close(fig)
        for p,h in spec['input_sha256'].items():
            if sha(ROOT/p)!=h:raise ValueError('input modified '+p)
        result=dict(groups=counts,selected_pairs=[p for p in pairs if p['first']=='window_0_crossing' and p['second'] in names[2:]],
            pair_count=len(pairs),resolved_reference_rays=len(set().union(*groups.values())))
    except BaseException:error=traceback.format_exc()
    summary=dict(status='GATE_FAIL' if error else 'GATE_MIXED',error=error,elapsed_s=time.monotonic()-start,
        **result,new_labels=0,training_steps=0,grid_rebuilds=0)
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
