"""One bounded in-session AI region annotation on sealed indexed observations."""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse,hashlib,json,shutil,subprocess
from ai_junction_pilot import sha,write
NAME='gse_ai_region_annotation_v1'
CARD=f'configs/v3/gate3/data_cards/{NAME}.json';SPEC=f'configs/v3/gate3/{NAME}.json'
RUN=f'results/gate3_semantics/gate3_20260911_{NAME}_seed0'
SOURCE='results/gate3_semantics/gate3_20260911_gse_ai_indexed_review_v1r1_seed0'
PYTHON='/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python'

def main(mode):
    run=ROOT/RUN
    if mode=='freeze':
        card=json.loads((ROOT/'configs/v3/gate3/data_cards/gse_ai_three_case_pilot_v1.json').read_text())
        manifest=json.loads((ROOT/SOURCE/'artifacts/review_manifest.json').read_text())
        card['scope']['indexed_bundles']={SOURCE+'/'+e['bundle']:e['sha256'] for e in manifest['observations']}
        card['scope']['protocol']='docs/figures/gse_supervision_acquisition_pilot_v1/blind_observations.md'
        card['scope']['protocol_sha256']=sha(ROOT/card['scope']['protocol'])
        card['annotation']['output_schema']='Source-indexed measured surface regions; AI relationship judgments with explicit unknowns, not automatic training labels'
        card['approval']['scope']='Same-three-case AI point-region annotation using sealed v2 inputs; no training'
        card['approval']['scope_sha256']=hashlib.sha256(json.dumps(card['scope'],sort_keys=True).encode()).hexdigest();write(ROOT/CARD,card)
        inputs={CARD:sha(ROOT/CARD),**card['scope']['indexed_bundles'],card['scope']['protocol']:card['scope']['protocol_sha256']}
        spec=dict(schema_version='v3_run_spec_v1',gate=3,date='20260911',slug=NAME,seed=0,operation='ai_annotation',data_card=CARD,user_authorization=card['approval'],
            command=['env','MPLCONFIGDIR=/tmp/gse-mpl','OPENBLAS_NUM_THREADS=1',PYTHON,'tools/v3/ai_region_annotation.py','--begin'],
            question='Which measured 3D regions and structural relations can AI substantiate in these three observations?',method='AI selects 3D surface regions after viewing registered observations; exact raw ray identities saved; nonblind reference exposure explicit',
            baseline='Existing teacher is not an independent gold label; later correspondence remains separate',fallback='Save unknown relations; no forced center or full background and no training',
            acceptance_criteria=['All three cases reviewed with point-index evidence','No structural identity inferred from crop boundary alone','Raw AI response and selection geometry retained; unknown explicit'],
            expected_evidence=['Views, raw response, source-indexed regions, judgments, metrics, seal'],estimated_cost=dict(compute='CPU rendering and in-session AI annotation',host_ram_gb=4,gpu_vram_gb=0,disk_gb=.1,wall_time_hours=.25),
            input_sha256=inputs,source_sha256={p:sha(ROOT/p) for p in ['tools/v3/ai_region_annotation.py','tools/v3/ai_junction_pilot.py','tools/v3/review/gse_surface_review.js']})
        write(ROOT/SPEC,spec);return
    spec=json.loads((ROOT/SPEC).read_text());card=json.loads((ROOT/CARD).read_text())
    for p,h in {**spec['input_sha256'],**spec['source_sha256']}.items():
        if sha(ROOT/p)!=h:raise ValueError('drift '+p)
    state=json.loads((run/'RUN_STATE.json').read_text())['state']
    if mode=='begin':
        if state!='CREATED_NOT_EXECUTED':raise ValueError('fresh run required')
        write(run/'RUN_STATE.json',dict(state='RUNNING',activity='in-session AI annotation; not a background process'),'w')
        shutil.copyfile(ROOT/'tools/v3/ai_region_annotation.py',run/'artifacts/runner_source.py')
        import numpy as np
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for i,p in enumerate(card['scope']['indexed_bundles']):
            b=json.loads((ROOT/p).read_text());xyz=np.asarray(b['points_xyz_m']);slots=np.asarray(b['point_history_slots'])
            mask=(np.linalg.norm(xyz,axis=1)<=10)&(slots==4);x=xyz[mask]
            fig=plt.figure(figsize=(15,10))
            for j,(elev,azim) in enumerate([(20,-60),(20,120),(80,0),(0,0)]):
                ax=fig.add_subplot(2,2,j+1,projection='3d');ax.scatter(*x.T,s=1,c=x[:,2],cmap='viridis');ax.scatter([0],[0],[0],c='red',marker='+');ax.view_init(elev,azim);ax.set(xlim=(-10,10),ylim=(-10,10),zlim=(-6,6),xlabel='X m',ylabel='Y m',zlabel='Z m');ax.set_box_aspect((20,20,12))
            fig.suptitle(f'Case {i}: current frame / current sensor; color=height; no teacher; previous exposure disclosed')
            fig.savefig(run/f'previews/case_{i}_3d.png',dpi=130);plt.close(fig)
        print('Three 3D views ready; AI must provide actual raw response and region judgments.');return
    if state!='RUNNING':raise ValueError('active annotation required')
    raw=run/'logs/raw_ai_response.md';assert raw.is_file() and raw.stat().st_size>0
    selections=json.loads((run/'artifacts/region_selections.json').read_text());assert len(selections['cases'])==3
    rows=[]
    for i,((p,h),case) in enumerate(zip(card['scope']['indexed_bundles'].items(),selections['cases'])):
        assert case['case']==i and case['regions'] and case['complete_background'] is False
        # Use the tested browser selection implementation, not a second selector.
        program="const fs=require('fs'),m=require('./tools/v3/review/gse_surface_review.js');const x=JSON.parse(fs.readFileSync(0,'utf8'));console.log(JSON.stringify(x.regions.map(r=>m.selectSurfaceRegion(x.bundle,r.bounds_m,r.history_slots,r.kind,r.evidence))));"
        b=json.loads((ROOT/p).read_text())
        result=json.loads(subprocess.check_output(['node','-e',program],input=json.dumps(dict(bundle=b,regions=case['regions'])),text=True,cwd=ROOT))
        record=dict(case=i,bundle_sha256=h,coordinate_frame='current_sensor_m',blind=False,prior_reference_exposure=True,regions=result,relationship_judgment=case['relationship_judgment'],complete_background=False,training_qualified=False)
        write(run/f'artifacts/case_{i}_regions.json',record);rows.append(dict(case=i,region_counts=[len(r['points']) for r in result],relationship_judgment=case['relationship_judgment']))
    write(run/'metrics/summary.json',dict(status='GATE_MIXED',cases=rows,point_region_annotations=sum(len(r['region_counts']) for r in rows),training_steps=0,training_qualified=False,meaning='AI source-indexed regions recorded; structural qualification separately required'))
    write(run/'RUN_STATE.json',dict(state='COMPLETED'),'w')
    with (run/'artifacts/evidence_sha256.txt').open('x') as f:
        for p in sorted(run.rglob('*')):
            if p.is_file() and p.name!='evidence_sha256.txt':f.write(sha(p)+'  '+str(p.relative_to(ROOT))+'\n')
    print((run/'metrics/summary.json').read_text())

if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    for m in ['freeze','begin','seal']:g.add_argument('--'+m,action='store_true')
    a=p.parse_args();main(next(m for m in ['freeze','begin','seal'] if getattr(a,m)))
