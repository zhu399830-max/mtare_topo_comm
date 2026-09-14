"""Read sealed support outputs; report evidence population, never model scores."""
from _bootstrap import PROJECT_ROOT as ROOT
from collections import Counter, defaultdict
import gzip
import json
import numpy as np
from local_pair_support_pilot import RUN
from check_local_pair_window_coverage import sha
from mtare_topo.data.gse_structure_review_v1 import canonical_sha


def summarize():
    root=ROOT/RUN
    state=json.loads((root/'RUN_STATE.json').read_text())['state']
    if state not in ('COMPLETED','FAILED'):
        raise ValueError('terminal sealed run required')
    pins={p:h for h,p in (l.split('  ',1) for l in (root/'artifacts/evidence_sha256.txt').read_text().splitlines())}
    for p,h in pins.items():
        if sha(ROOT/p)!=h:raise ValueError('evidence drift '+p)
    rows=json.loads((root/'artifacts/target_manifest.json').read_text())
    splits={};checks=Counter()
    for split in ('fit','calibration','development'):
        counts=Counter();parents=defaultdict(set);entities=defaultdict(set)
        for row in [r for r in rows if r['split']==split]:
            data=json.loads(gzip.decompress((root/'artifacts'/row['evidence_file']).read_bytes()))
            target=data['produced_targets'];record=target['record'];prov=target['teacher_provenance'];source=row['source']
            if canonical_sha(record)!=target['target_record_sha256']:raise ValueError('target hash')
            if record['source_frame_indices']!=source['frame_rows']:raise ValueError('frame identity')
            refs=[*prov['anchors'],*prov['terminals']];start=prov['terminal_anchor_start']
            if len(refs)!=len(record['anchors']) or start!=len(prov['anchors']):raise ValueError('reference alignment')
            exclusions={(r['opening_index'],r['anchor_index']):r for r in prov.get('terminal_nonmembership',[])}
            counts['observations']+=1
            checks['target_hash_frame_provenance']+=1
            for oi,members in enumerate(record['membership']):
                for ai,value in enumerate(members):
                    kind='junction' if ai<start else 'terminal'
                    key=kind+':'+{True:'positive',False:'negative',None:'unknown'}[value]
                    counts[key]+=1;parents[key].add(source['parent'])
                    entities[key].add((source['parent'],refs[ai]['node_id_teacher_only'],
                        prov['openings'][oi]['primitive_id_teacher_only']))
                    if value is False:
                        e=exclusions.get((oi,ai))
                        if not e or not e['opening_junction_ray_indices'] or not (
                            e['cap_entering_ray_indices'] or (e['cap_source_interior_ray_indices'] and e['terminal_outgoing_opening_ray_indices'])):
                            raise ValueError('negative lacks original observed-exclusion evidence')
                        if ai<start:raise ValueError('unexpected V8 negative mechanism')
                        checks['negative_observed_witness_present']+=1
                    if value is True:
                        relation=prov['relations'][oi]
                        if relation.get('anchor_index')!=ai:raise ValueError('positive provenance assignment differs')
                        if not relation.get('ray_indices') and not relation.get('evidence_indices'):
                            # Original cap-entry reference may nest its ray evidence.
                            if not any(k.endswith('ray_indices') and v for k,v in relation.items()):
                                raise ValueError('positive missing explicit evidence indices')
                        checks['positive_assignment_evidence_present']+=1
            # Partial labels must never become complete scoring regions.
            if record['score_region']['anchors_complete'] or record['score_region']['openings_complete']:
                raise ValueError('unexpected complete-background claim')
            checks['partial_regions_preserved']+=1
        known=sum(v for k,v in counts.items() if k.endswith(':positive') or k.endswith(':negative'))
        # A diagnostic of label confounding using teacher types, NOT a deployable model.
        trivial_correct=counts['junction:positive']+counts['terminal:negative']
        splits[split]=dict(counts=dict(counts),parent_counts={k:len(v) for k,v in parents.items()},
            unique_parent_anchor_source_relations={k:len(v) for k,v in entities.items()},
            teacher_type_only_shortcut=dict(known_relations=known,correct=trivial_correct,
                fraction=trivial_correct/known if known else None,rule='junction=yes, terminal=no; diagnostic only; GT type never model input'))
    return dict(run=RUN,run_state=state,completed_observations=len(rows),intended_observations=159,
        incomplete_population=len(rows)!=159,seal_files_verified=len(pins),checks=dict(checks),splits=splits,
        limitations=['partial automatic reference, not human confirmed','one development parent only',
            'relation counts are correlated, not independent observations','no model trained or method advantage demonstrated'])


if __name__=='__main__':print(json.dumps(summarize(),indent=2,ensure_ascii=False))
