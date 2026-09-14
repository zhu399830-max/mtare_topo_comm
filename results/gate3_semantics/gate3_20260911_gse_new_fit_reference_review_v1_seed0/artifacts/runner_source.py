"""Reveal only the twelve frozen first-pass references; no teacher rerun or labels."""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse
import gzip
import hashlib
import json
import math
import shutil
from ai_junction_pilot import sha, write

NAME = 'gse_new_fit_reference_review_v1'
CARD = 'configs/v3/gate3/data_cards/' + NAME + '.json'
SPEC = 'configs/v3/gate3/' + NAME + '.json'
RUN = 'results/gate3_semantics/gate3_20260911_' + NAME + '_seed0'
FIRST = 'results/gate3_semantics/gate3_20260911_gse_ai_new_fit_review_v1_seed0'
SOURCE = 'results/gate3_semantics/gate3_20260910_gse_local_pair_support_pilot_v1_seed20260906'

def pinned(path, digest):
    raw = (ROOT / path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError('drift: ' + path)
    return json.loads(gzip.decompress(raw) if path.endswith('.gz') else raw)

def freeze():
    card = json.loads((ROOT / 'configs/v3/gate3/data_cards/gse_ai_new_fit_review_v1.json').read_text())
    seals = {p:h for h,p in (line.split('  ',1) for line in (ROOT/SOURCE/'artifacts/evidence_sha256.txt').read_text().splitlines())}
    manifest = SOURCE + '/artifacts/target_manifest.json'
    rows = pinned(manifest, seals[manifest])
    selected = []
    for e in card['scope']['entries']:
        matches = [r for r in rows if r['source']['task'] == e['task'] and r['source']['frame_rows'] == e['frame_rows']]
        if len(matches) > 1:
            raise ValueError('exact reference identity not unique')
        if not matches:
            selected.append(dict(case=e['case'],path=None,sha256=None,status='MISSING_IN_ORIGINAL_PARTIAL_RUN'))
            continue
        r = matches[0]
        if r['split'] != 'fit':
            raise ValueError('non-fit reference')
        path = SOURCE + '/artifacts/' + r['evidence_file']
        selected.append(dict(case=e['case'], path=path, sha256=seals[path]))
    first = FIRST + '/artifacts/first_pass.json'
    first_pins = {p:h for h,p in (line.split('  ',1) for line in (ROOT/FIRST/'artifacts/evidence_sha256.txt').read_text().splitlines())}
    frozen = pinned(first, first_pins[first])
    if frozen['reference_payload_read'] or frozen['training_qualified']:
        raise ValueError('first-pass boundary invalid')
    card['scope']['reference_reveal'] = selected
    card['scope']['frozen_first_pass'] = {first:first_pins[first]}
    card['scope']['review_protocol'] = 'Reveal same12 original references after first-pass seal; compare coordinates and per-frame source support, no targets changed'
    card['annotation']['input_bundle_contract'] = 'Frozen first-pass judgments plus exact existing12 reference payloads; no new maps or teacher calls'
    card['annotation']['output_schema'] = 'Reference coordinates, source witness counts and AI correspondence interpretation; not new training labels'
    card['approval']['scope'] = 'Same twelve fixed fit observations: post-freeze reference reveal and interpretation only'
    card['approval']['scope_sha256'] = hashlib.sha256(json.dumps(card['scope'],sort_keys=True).encode()).hexdigest()
    write(ROOT/CARD, card)
    inputs = {CARD:sha(ROOT/CARD), manifest:seals[manifest], first:first_pins[first]}
    inputs.update({r['path']:r['sha256'] for r in selected if r['path'] is not None})
    spec = dict(schema_version='v3_run_spec_v1',gate=3,date='20260911',slug=NAME,seed=0,
        operation='ai_annotation',data_card=CARD,user_authorization=card['approval'],
        command=['python3','tools/v3/review_new_fit_references.py','--execute'],
        question='Which of the same12 references were saved, and where do the available ones lie relative to observed structure hypotheses?',
        method='Read sealed records only; preserve first-pass judgments, quantify3D positions and frame support; interpret without relabeling',
        baseline='Frozen pre-reveal qualitative judgments; source support is not independent observability proof',
        fallback='Report missing/ambiguous correspondence, no resampling, new targets or training',
        acceptance_criteria=['Exactly12 identities accounted for, missing records explicit','Available reference positions and original source support retained','No unknown to background or label qualification'],
        expected_evidence=['Per-case comparison, original first-pass hashes, interpretation, summary and seal'],
        estimated_cost=dict(compute='CPU read-only12 saved references; zero GPU/teacher/model',host_ram_gb=2,gpu_vram_gb=0,disk_gb=.05,wall_time_hours=.1),
        input_sha256=inputs,source_sha256={'tools/v3/review_new_fit_references.py':sha(ROOT/'tools/v3/review_new_fit_references.py'),'tools/v3/ai_junction_pilot.py':sha(ROOT/'tools/v3/ai_junction_pilot.py')})
    write(ROOT/SPEC,spec)

def execute():
    run = ROOT/RUN
    spec = json.loads((ROOT/SPEC).read_text())
    if json.loads((run/'RUN_STATE.json').read_text())['state'] != 'CREATED_NOT_EXECUTED':
        raise ValueError('fresh run required')
    for p,h in {**spec['input_sha256'],**spec['source_sha256']}.items():
        if sha(ROOT/p) != h: raise ValueError('drift ' + p)
    card = json.loads((ROOT/CARD).read_text())
    first = json.loads((ROOT/FIRST/'artifacts/first_pass.json').read_text())
    write(run/'RUN_STATE.json',dict(state='RUNNING',activity='bounded reference comparison'),'w')
    shutil.copyfile(ROOT/'tools/v3/review_new_fit_references.py',run/'artifacts/runner_source.py')
    output=[]
    for e,b in zip(card['scope']['entries'],card['scope']['reference_reveal']):
        if b['path'] is None:
            output.append(dict(case=e['case'],task=e['task'],frames=e['frame_rows'],reference_status='MISSING_IN_ORIGINAL_PARTIAL_RUN',first_pass=first['cases'][e['case']],anchors=[],labels_changed=0))
            continue
        full=pinned(b['path'],b['sha256'])
        target=full['produced_targets']; record=target['record']; prov=target['teacher_provenance']
        if record['source_frame_indices'] != e['frame_rows']: raise ValueError('frame mismatch')
        source=full['raw_interfaces']['source']
        if source['task'] != e['task'] or source['frame_rows'] != e['frame_rows']: raise ValueError('source mismatch')
        refs=prov['anchors']+prov['terminals']
        if len(refs) != len(record['anchors']): raise ValueError('anchor alignment')
        anchors=[]
        for i,(a,p) in enumerate(zip(record['anchors'],refs)):
            xyz=a['position_m']; distance=math.sqrt(sum(v*v for v in xyz))
            witnesses=p.get('witness_ray_indices',[])
            support=[]
            if i<len(prov['anchors']):
                for interface,ids in zip(p['interface_ids'],witnesses):
                    if any(not 0<=r<57600 for r in ids): raise ValueError('ray outside history')
                    support.append(dict(interface=interface,per_frame=[sum(r//11520==s for r in set(ids)) for s in range(5)]))
            anchors.append(dict(kind='junction' if i<len(prov['anchors']) else 'terminal',
                node_teacher_only=p['node_id_teacher_only'],position_m=xyz,distance_m=distance,
                inside_original_10m=distance<=10,interface_source_support=support,
                original_provenance=p))
        output.append(dict(case=e['case'],task=e['task'],frames=e['frame_rows'],reference_status='AVAILABLE',reference_sha256=b['sha256'],
            first_pass=first['cases'][e['case']],anchors=anchors,openings=record['openings'],membership=record['membership'],
            score_region=record['score_region'],limitations=target.get('limitations',[]),labels_changed=0))
        print(json.dumps(dict(case=e['case'],anchors=[{k:a[k] for k in ('kind','position_m','distance_m','inside_original_10m')} for a in anchors],openings=len(record['openings'])),ensure_ascii=False),flush=True)
    write(run/'artifacts/comparison.json',dict(cases=output,training_steps=0,new_labels=0))
    write(run/'metrics/summary.json',dict(status='GATE_MIXED',cases=len(output),
        available_references=sum(c['reference_status']=='AVAILABLE' for c in output),
        missing_reference_cases=[c['case'] for c in output if c['reference_status']!='AVAILABLE'],
        junctions=sum(a['kind']=='junction' for c in output for a in c['anchors']),
        terminals=sum(a['kind']=='terminal' for c in output for a in c['anchors']),
        anchors_outside10m=sum(not a['inside_original_10m'] for c in output for a in c['anchors']),
        training_qualified=False,training_steps=0,teacher_calls=0,new_labels=0))
    write(run/'RUN_STATE.json',dict(state='AWAITING_IN_SESSION_INTERPRETATION'),'w')

def seal():
    run=ROOT/RUN
    if json.loads((run/'RUN_STATE.json').read_text())['state']!='AWAITING_IN_SESSION_INTERPRETATION': raise ValueError('not ready')
    if not (run/'logs/interpretation.md').stat().st_size: raise ValueError('interpretation absent')
    write(run/'RUN_STATE.json',dict(state='COMPLETED'),'w')
    with (run/'artifacts/evidence_sha256.txt').open('x') as f:
        for p in sorted(run.rglob('*')):
            if p.is_file() and p.name!='evidence_sha256.txt': f.write(sha(p)+'  '+str(p.relative_to(ROOT))+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    for mode in ('freeze','execute','seal'): g.add_argument('--'+mode,action='store_true')
    a=p.parse_args();globals()[next(m for m in ('freeze','execute','seal') if getattr(a,m))]()
