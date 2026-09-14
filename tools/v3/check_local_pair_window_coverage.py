"""Exact saved-window coordinate coverage, not ray visibility or new labels."""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

SLUG='gse_local_pair_window_coverage_v1'
CARD='configs/v3/gate3/data_cards/'+SLUG+'.json'
SPEC='configs/v3/gate3/'+SLUG+'.json'
RUN='results/gate3_semantics/gate3_20260910_'+SLUG+'_seed20260906'
SOURCE='configs/v3/gate3/data_cards/gse_supplement_joint_v3.json'
OPPORTUNITIES='docs/figures/gse_graph/local_structure_opportunities_20260910.json'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p,v,mode='x'):
    with p.open(mode) as f:json.dump(v,f,indent=2,allow_nan=False)


def freeze():
    if (ROOT/RUN).exists():raise FileExistsError('no refreeze after run creation')
    original=json.loads((ROOT/SOURCE).read_text())['scope']
    opportunities=json.loads((ROOT/OPPORTUNITIES).read_text())
    parents={r['parent']:r for r in opportunities['parents'] if r['junction_terminal']}
    if len(parents)!=44 or sum(len(r['junction_terminal']) for r in parents.values())!=77:
        raise ValueError('fixed opportunity population drift')
    files={SOURCE:sha(ROOT/SOURCE),OPPORTUNITIES:sha(ROOT/OPPORTUNITIES)};entries=[]
    for old in original['entries']:
        parent=old['observations'][0]['parent_id']
        if parent not in parents:continue
        if any(o['parent_id']!=parent or o['split']!=parents[parent]['split'] for o in old['observations']):
            raise ValueError('mixed source split')
        prefix=old['sensor_prefix']+'/sensor_xyz_m'
        plan=original['array_access'][prefix]
        entry=dict(parent=parent,split=parents[parent]['split'],task=old['task'],observations=old['observations'],
            construction=old['construction_path'],coordinate_prefix=prefix,coordinate_plan=plan,
            reference_pairs=parents[parent]['junction_terminal'])
        entries.append(entry)
        for path in [old['construction_path'],prefix+'/.zarray',*(prefix+'/'+k for k in plan['chunk_keys'])]:
            files[path]=original['file_sha256'][path]
    if len(entries)!=132:raise ValueError('exact three variants per parent required')
    counts={}
    for split in ['fit','calibration','development']:
        selected=[e for e in entries if e['split']==split]
        counts[split]=dict(parents=len({e['parent'] for e in selected}),tasks=len(selected),
            observations=sum(len(e['observations']) for e in selected),
            unique_variant_frames=sum(len({f for o in e['observations'] for f in o['frame_rows']}) for e in selected))
    scope=dict(entries=entries,counts=counts,source_sha256=files,world_scope='C01-C07 historical development only',
        sampling='all previously selected windows of all44 opportunity parents; no scores, no new source rows',
        independent_unit='parent map; variants and overlapping windows are not independent',
        teacher='none; membership, visibility and training eligibility remain unknown',
        radius_m=10,reference_pairs=77,training=False,scan_reads=False,
        coordinate_decoded_padded_bytes=sum(e['coordinate_plan']['decoded_padded_bytes'] for e in entries),
        selected_coordinate_rows=sum(len(e['coordinate_plan']['selected_rows']) for e in entries),
        incidental_coordinate_rows=sum(e['coordinate_plan']['decoded_rows_including_collateral']-len(e['coordinate_plan']['selected_rows']) for e in entries))
    approval=dict(status='APPROVED',approved_by='user-standing-development-authorization',approved_at='2026-09-10',
        authorized_gates=[3],authorized_operations=['audit'],scope='Exact existing-source coordinate coverage; no labels/training',
        confirmation_reference='Active user goal: verify independent observable supervision and freeze exact population',
        scope_sha256=hashlib.sha256(json.dumps(scope,sort_keys=True).encode()).hexdigest())
    write(ROOT/CARD,dict(schema_version='gse_local_pair_coordinate_card_v1',scope=scope,approval=approval),'w')
    sources=[Path(__file__),ROOT/'tools/v3/_bootstrap.py',ROOT/CARD,ROOT/'src/mtare_topo/governance.py',ROOT/'src/mtare_topo/governance_branch_place_replay.py']
    write(ROOT/SPEC,dict(schema_version='v3_run_spec_v1',gate=3,date='20260910',slug=SLUG,seed=20260906,
        operation='audit',data_card=CARD,config_path=CARD,user_authorization=approval,
        command=[sys.executable,'tools/v3/check_local_pair_window_coverage.py','--execute'],
        question='Which of the existing windows can geometrically include both declared junction and terminal references?',
        method='Same fixed 10m ball at original current sensor position; no ray visibility or label inference',
        baseline='77 construction-neighborhood opportunities; not model comparison',
        estimated_cost=dict(compute='CPU, coordinates only, no models or scan arrays',disk_gb=.1,wall_time_hours=.1),
        acceptance_criteria=['Exact declared windows and source hashes','Keep uncovered pairs and all splits','No incidental rows used as new observations','Do not treat coordinate coverage as visible labels'],
        expected_evidence=['all window coverage, all pair coverage, unknowns, source hashes, logs, environment, seal'],
        input_sha256=files,source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources}),'w')
    print(json.dumps(dict(counts=counts,coordinate_bytes=scope['coordinate_decoded_padded_bytes'],spec=SPEC)))


def execute():
    import numpy as np
    import numcodecs
    from mtare_topo.governance_branch_place_replay import validate_local_pair_card
    out=ROOT/RUN;spec=json.loads((ROOT/SPEC).read_text());card=json.loads((ROOT/CARD).read_text());s=card['scope']
    assert json.loads((out/'RUN_STATE.json').read_text())['state']=='CREATED_NOT_EXECUTED'
    assert json.loads((out/'config/run_spec.json').read_text())==spec
    assert validate_local_pair_card(card).passed
    for path,h in dict(spec['input_sha256'],**spec['source_sha256']).items():
        if sha(ROOT/path)!=h:raise ValueError('source drift: '+path)
    write(out/'RUN_STATE.json',dict(state='RUNNING'),'w');start=time.monotonic();error=None;summary={}
    try:
        pair_rows={};observations=0;covered_windows={split:0 for split in s['counts']}
        with (out/'artifacts/window_coverage.jsonl').open('x') as log:
            for e in s['entries']:
                base=json.loads((ROOT/e['construction']).read_text())['base_construction']
                nodes={n['node_id']:n for n in base['composition_operations']}
                header=json.loads((ROOT/(e['coordinate_prefix']+'/.zarray')).read_text());plan=e['coordinate_plan']
                if header['filters'] or header['order']!='C' or header['dtype']!='<f8' or header['shape'][1:]!=[3]:
                    raise ValueError('unsupported coordinate storage; no silent decoder substitution')
                allowed=set(plan['selected_rows']);coordinates={}
                codec=numcodecs.get_codec(header['compressor'])
                for key in plan['chunk_keys']:
                    array=np.frombuffer(codec.decode((ROOT/(e['coordinate_prefix']+'/'+key)).read_bytes()),dtype='<f8').reshape(header['chunks'])
                    begin=int(key.split('.')[0])*header['chunks'][0]
                    for index in allowed:
                        if begin<=index<begin+len(array):coordinates[index]=array[index-begin].copy()
                if set(coordinates)!=allowed:raise ValueError('coordinate coverage mismatch')
                for pair in e['reference_pairs']:
                    identity=(e['parent'],pair['junction'],pair['terminal'])
                    pair_rows.setdefault(identity,dict(parent=e['parent'],split=e['split'],junction=pair['junction'],terminal=pair['terminal'],
                        minimum_required_radius_m=None,covered_observations=[],tested_observations=0))
                for o in e['observations']:
                    xyz=coordinates[o['frame_rows'][-1]]
                    if not np.isfinite(xyz).all():raise ValueError('nonfinite original coordinate')
                    values=[]
                    for pair in e['reference_pairs']:
                        a=nodes[pair['junction']]['anchor_xyz_m'];b=nodes[pair['terminal']]['anchor_xyz_m']
                        distance=[float(np.linalg.norm(xyz-np.asarray(p))) for p in [a,b]]
                        covered=max(distance)<=s['radius_m'];identity=(e['parent'],pair['junction'],pair['terminal'])
                        result=pair_rows[identity];result['tested_observations']+=1
                        previous=result['minimum_required_radius_m']
                        result['minimum_required_radius_m']=max(distance) if previous is None else min(previous,max(distance))
                        if covered:result['covered_observations'].append(dict(task=e['task'],sequence=o['source_sequence_id'],frame_rows=o['frame_rows']))
                        values.append(dict(junction=pair['junction'],terminal=pair['terminal'],distances_m=distance,both_in_ball=covered))
                    observations+=1;covered_windows[e['split']]+=any(v['both_in_ball'] for v in values)
                    log.write(json.dumps(dict(source=o,current_sensor_xyz_m=xyz.tolist(),pairs=values,visibility='NOT_EVALUATED'))+'\n')
        if observations!=sum(c['observations'] for c in s['counts'].values()) or len(pair_rows)!=77:
            raise ValueError('population drift')
        rows=list(pair_rows.values());write(out/'artifacts/pair_coverage.json',rows)
        summary=dict(status='GATE_MIXED',observations=observations,covered_windows=covered_windows,
            covered_pairs={split:sum(bool(r['covered_observations']) for r in rows if r['split']==split) for split in s['counts']},
            covered_parents={split:len({r['parent'] for r in rows if r['split']==split and r['covered_observations']}) for split in s['counts']},
            unobserved_or_outside_not_negative=True,labels_generated=0,scan_reads=0,training_steps=0,
            meaning='Necessary center-domain coverage only. No visibility, membership or training qualification.')
        write(out/'artifacts/environment.json',dict(python=sys.version,numpy=np.__version__,numcodecs=numcodecs.__version__))
    except Exception:error=traceback.format_exc()
    summary.update(error=error,elapsed_s=time.monotonic()-start)
    if error:summary['status']='GATE_FAIL'
    write(out/'metrics/summary.json',summary);write(out/'logs/raw.json',summary)
    write(out/'RUN_STATE.json',dict(state='FAILED' if error else 'COMPLETED'),'w')
    with (out/'artifacts/evidence_sha256.txt').open('x') as f:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.name!='evidence_sha256.txt':f.write(sha(p)+'  '+str(p.relative_to(ROOT))+'\n')
    print(json.dumps(summary));return int(error is not None)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.freeze:freeze()
    elif a.execute:raise SystemExit(execute())
    else:p.error('choose action')
