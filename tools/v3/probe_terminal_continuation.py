"""Five saved-source prerequisites only; never relabel or rerun ray casting."""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse
import gzip
import hashlib
import json
import resource
import signal
import sys
import time
import traceback
from check_local_pair_window_coverage import sha,write

SLUG='gse_terminal_continuation_probe_v1'
CARD='configs/v3/gate3/data_cards/'+SLUG+'.json'
SPEC='configs/v3/gate3/'+SLUG+'.json'
RUN='results/gate3_semantics/gate3_20260910_'+SLUG+'_seed20260906'
INPUT='results/gate3_semantics/gate3_20260910_gse_local_pair_pilot_inputs_v1_seed20260906'
TEACHER='results/gate3_semantics/gate3_20260910_gse_local_pair_support_pilot_v1_seed20260906'
CHOICES='docs/figures/gse_local_pair_support_pilot_v1/teacher_expressivity.json'


def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()


def freeze():
    if any((ROOT/p).exists() for p in (CARD,SPEC,RUN)):raise FileExistsError('no refreeze')
    files={CHOICES:sha(ROOT/CHOICES)};pins={}
    for run in (INPUT,TEACHER):
        p=run+'/artifacts/evidence_sha256.txt';files[p]=sha(ROOT/p)
        pins.update({p:h for h,p in (l.split('  ',1) for l in (ROOT/p).read_text().splitlines())})
    manifests=[]
    for p in (INPUT+'/artifacts/manifest.json',TEACHER+'/artifacts/target_manifest.json'):
        if sha(ROOT/p)!=pins[p]:raise ValueError('manifest drift')
        files[p]=pins[p];manifests.append(json.loads((ROOT/p).read_text()))
    rows=[]
    for choice in json.loads((ROOT/CHOICES).read_text())['pairs']:
        task=choice['task'];seq=choice['source']
        item=next(r for r in manifests[0] if r['source']['task']==task and r['source']['source_global_sequence_index']==seq)
        target=next(r for r in manifests[1] if r['source']['task']==task and r['source']['source_sequence_id']==seq)
        paths=dict(student=INPUT+'/'+item['student_path'],sensor=INPUT+'/'+item['source_evidence_path'],
            construction=INPUT+'/artifacts/source_evidence/'+task+'_constructions.json',target=TEACHER+'/artifacts/'+target['evidence_file'])
        files.update({p:pins[p] for p in paths.values()})
        rows.append(dict(choice=choice,source=item['source'],paths=paths))
    scope=dict(rows=rows,files=files,observations=5,parents=2,variant_frames=25,field_spacing_m=.025,
        labels_changed=False,ray_recast=False,training=False,independent_evaluation=False,
        resources=dict(address_space_bytes=3*1024**3,wall_seconds=600,output_bytes=1024**3),
        question='Are continuous realized source geometry, local first exit and same-frame containment compatible with the saved five unknown references?')
    approval=dict(status='APPROVED',approved_by='user-standing-development-authorization',approved_at='2026-09-10',
        authorized_gates=[3],authorized_operations=['data_export'],scope_sha256=digest(scope),
        scope='Five saved fit windows: prerequisite diagnostics, no new labels or training',
        confirmation_reference='Active user goal and PLAN authorize exact five missing-evidence diagnostic after21 software checks; no repeating159 or changing old labels.')
    write(ROOT/CARD,dict(schema_version='gse_terminal_continuation_probe_card_v1',scope=scope,approval=approval))
    sources={str(p.relative_to(ROOT)):sha(p) for folder in ('src/mtare_topo','tools/v3') for p in (ROOT/folder).rglob('*.py')};sources[CARD]=sha(ROOT/CARD)
    write(ROOT/SPEC,dict(schema_version='v3_run_spec_v1',gate=3,date='20260910',slug=SLUG,seed=20260906,operation='data_export',
        data_card=CARD,config_path=CARD,user_authorization=approval,
        command=[sys.executable,'tools/v3/probe_terminal_continuation.py','--execute'],question=scope['question'],
        method='Existing exact realized reference continuations and .025 source distance field; saved ray witnesses; prerequisite results only',
        baseline='Original unknown labels unchanged; primitive-ID limitation diagnostic, not model comparison',
        fallback='Stop and seal; no tolerance changes, ray rerun, inferred negative or retry',
        estimated_cost=dict(compute='CPU only5 windows;3GiB address cap',host_ram_gb=3,gpu_vram_gb=0,disk_gb=1,wall_time_hours=1/6),
        acceptance_criteria=['Exactly five source bindings','Unknown unchanged; all prerequisites separate','No casting/training; failures retained'],
        expected_evidence=['source checks,per-pair component and origin diagnostics,environment,logs,seal'],
        input_sha256=files,source_sha256=sources))
    print(json.dumps(dict(spec=SPEC,observations=5,parents=2,variant_frames=25)))


def execute():
    import numpy as np
    from mtare_topo.governance_branch_place_replay import validate_continuation_probe_card
    from mtare_topo.data.gse_structure_review_v1 import canonical_sha
    from mtare_topo.data.primitive_relation_materialization import load_p1a_realized_construction
    from mtare_topo.teacher.gse_construction_continuations_v1 import construction_reference_continuations
    from mtare_topo.teacher.gse_terminal_continuation_contract import terminal_continuation_component,continuation_origin_witness
    from mtare_topo.teacher.swept_superellipse_field import SweptSuperellipseProvenanceField
    run=ROOT/RUN;spec=json.loads((ROOT/SPEC).read_text());card=json.loads((ROOT/CARD).read_text());s=card['scope']
    if json.loads((run/'RUN_STATE.json').read_text())['state']!='CREATED_NOT_EXECUTED' or json.loads((run/'config/run_spec.json').read_text())!=spec or not validate_continuation_probe_card(card).passed:raise ValueError('fresh bound run required')
    for p,h in dict(spec['input_sha256'],**spec['source_sha256']).items():
        if sha(ROOT/p)!=h:raise ValueError('source drift '+p)
    start=time.monotonic();outputs=[];error=None
    write(run/'RUN_STATE.json',dict(state='RUNNING'),'w')
    def expire(*args):raise TimeoutError('600s diagnostic cap')
    signal.signal(signal.SIGALRM,expire);signal.alarm(600)
    resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3))
    try:
        for row in s['rows']:
            paths=row['paths'];c=row['choice'];doc=json.loads((ROOT/paths['construction']).read_text())
            data=json.loads(gzip.decompress((ROOT/paths['target']).read_bytes()));t=data['produced_targets'];v=t['teacher_provenance'];raw=data['raw_interfaces']
            if t['source_binding']['construction_sha256']!=canonical_sha(doc) or canonical_sha(t['record'])!=t['target_record_sha256']:raise ValueError('target/source binding')
            reference=construction_reference_continuations(doc,expected_document_sha256=canonical_sha(doc));_,primitives=load_p1a_realized_construction(doc)
            with np.load(ROOT/paths['sensor'],allow_pickle=False) as a:origins=a['sensor_xyz_m'].copy()
            with np.load(ROOT/paths['student'],allow_pickle=False) as a:ranges=a['ranges_m'].copy()
            component=terminal_continuation_component(reference['sources'],{p.primitive_id:p.centerline_xyz_m for p in primitives},terminal_node=c['terminal'],center_m=origins[-1])
            ti=next(i for i,x in enumerate(v['terminals']) if x['node_id_teacher_only']==c['terminal'])
            oi=next(i for i,x in enumerate(v['openings']) if x['primitive_id_teacher_only']==c['opening_primitive'])
            if t['record']['membership'][oi][v['terminal_anchor_start']+ti] is not None:raise ValueError('original unknown changed')
            detail=dict(choice=c,source=row['source'],component=component,original_membership=None,new_membership=None)
            if component['status']=='REFERENCE_COMPONENT_ONLY':
                distances=SweptSuperellipseProvenanceField(primitives,spacing_m=.025).operand_signed_distances_sparse(origins)
                witness=continuation_origin_witness(continuation_source_ids=component['source_ids'],source_ids=[p.primitive_id for p in primitives],origin_distances=distances,
                    node_id=c['terminal'],cap_rays=[w['ray_index'] for w in v['terminals'][ti]['witnesses']],opening_rays=v['openings'][oi]['crossing_ray_indices'],
                    interface_hits=raw['raw_interface_intersections'],interfaces=raw['interfaces_teacher_only'],first_return=ranges,frame_rows=row['source']['frame_rows'])
                offsets=next(x for x in component['source_arc_offsets'] if x['source_id']==c['opening_primitive'])
                arc=v['openings'][oi]['reference_arc_m'];global_arc=offsets['offset_m']+(arc if offsets['entry_side']==0 else offsets['length_m']-arc)
                detail.update(witness=witness,opening_continuation_arc_m=global_arc,
                    first_exit_arc_difference_m=global_arc-component['opening_reference_arc_m'],
                    original_known_contained_sources=[[primitives[i].primitive_id for i in np.flatnonzero(d<0)] for d in distances])
            outputs.append(detail)
            print(json.dumps(dict(completed=len(outputs),component_status=component['status'])),flush=True)
        if sum(p.stat().st_size for p in run.rglob('*') if p.is_file())>1024**3:raise OSError('output cap')
    except BaseException:error=traceback.format_exc()
    finally:signal.alarm(0)
    write(run/'artifacts/diagnostics.json',outputs)
    summary=dict(status='GATE_FAIL' if error else 'GATE_MIXED',completed=len(outputs),error=error,elapsed_s=time.monotonic()-start,labels_changed=0,training_steps=0,ray_recasts=0,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,python=sys.version,numpy=np.__version__)
    write(run/'metrics/summary.json',summary);write(run/'logs/raw.json',summary);write(run/'RUN_STATE.json',dict(state='FAILED' if error else 'COMPLETED'),'w')
    with (run/'artifacts/evidence_sha256.txt').open('x') as f:
        for p in sorted(run.rglob('*')):
            if p.is_file() and p.name!='evidence_sha256.txt':f.write(sha(p)+'  '+str(p.relative_to(ROOT))+'\n')
    print(json.dumps(summary));return int(error is not None)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.freeze:freeze()
    elif a.execute:raise SystemExit(execute())
    else:p.error('choose action')
