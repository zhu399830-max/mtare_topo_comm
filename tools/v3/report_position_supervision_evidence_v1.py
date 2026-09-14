"""Join sealed evidence for every candidate; no inference, relocalization or training.

The 1m evaluation exclusion mask is NOT permission to change 4m training negatives.
Output v1r retains only independently allowed labels, even when it changes none.
"""
from _bootstrap import PROJECT_ROOT as ROOT
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from mtare_topo.governance_surface_selection import digest

R=ROOT/'results/gate3_semantics/gate3_20260910_gse_candidate_center_verifier_v1_seed0'
S=ROOT/'results/gate3_semantics/gate3_20260910_gse_fixed_candidate_scoring_v1_seed0'
OUT=ROOT/'docs/figures/gse_graph/position_supervision_evidence_20260910'

def main():
    if OUT.exists():raise FileExistsError('non-overwriting evidence reduction')
    reads={};indices={}
    def read(p):
        raw=p.read_bytes();h=hashlib.sha256(raw).hexdigest()
        for run in (R,S):
            if p.is_relative_to(run):
                if run not in indices:
                    indices[run]={n:x for x,n in (l.split('  ',1) for l in (run/'artifacts/evidence_sha256.txt').read_text().splitlines())}
                assert indices[run][str(p.relative_to(ROOT))]==h
        reads[str(p.relative_to(ROOT))]=h
        return json.loads(raw)
    labels=read(R/'metrics/validity_inventory.json');final=read(R/'metrics/evaluation_1000.json')
    ref=read(S/'metrics/evaluation_0000.json')
    prior_errors=read(ROOT/'docs/figures/gse_graph/candidate_center_verifier_20260910/error_analysis.json')
    error_set={(digest(e['source']),e['query_slot'],e['region']) for e in prior_errors['errors']}
    rows=[];stats=Counter();errors=Counter();newlabels=[];minimal=[]
    for v,o,f in zip(labels['observations'],ref['observations'],final['observations']):
        assert v['source']==o['source']==f['source'];key=digest(v['source'])
        ev=read(S/'artifacts'/f'fixed_supervision_evidence_{key}.json');bg=ev['original_background']
        assert bg['binding']==v['source_binding'] and bg['observation_identity_binding_verified']
        assert not bg['inventory_completeness_supplied_not_verified'] and not bg['observed_states_supplied_not_verified']
        assert ev['background_radius_m']==4
        c=o['scores']['1.0']['coverage'];c4=o['scores']['4.0']['coverage']
        for q,lab in enumerate(v['candidates']):
            neg4=bg['reference_negative_mask'][q];neg1=c['reference_negative_mask'][q]
            near=c['confirmed_structure_coverage_mask'][q];conflict=c['possible_unconfirmed_reference_mask'][q]
            # Use only already sealed provenance; do not calculate new positions/matches.
            positive=lab['new_label']=='positive';assert not(positive and neg4)
            label='positive' if positive else 'negative' if neg4 else 'unknown'
            assert label==lab['new_label']
            evals={}
            for region,cov in [('old',c),('fixed',c4)]:
                raw=o['scores']['1.0' if region=='old' else '4.0']
                # Fixed-region identity uses the final 1m match, not historical 4m pairs.
                result=f['scores'][region]['after'];slots=result['original_query_slots']
                matched={slots[j]:k for j,k in result['pairs']}
                status='not_selected' if q not in slots else 'matched' if q in matched else 'false_positive' if cov['query_scoreable_mask'][q] else 'unresolved'
                assert (status=='false_positive')==((key,q,region) in error_set)
                evals[region]=dict(identity=status,scoreable=cov['query_scoreable_mask'][q],observed_inside=cov['observed_inside_score_region_mask'][q],
                    confirmed_structure_coverage=cov['confirmed_structure_coverage_mask'][q],unknown_reference_competitor=cov['possible_unconfirmed_reference_mask'][q],
                    matched_reference_index=matched.get(q),basis='sealed coverage + selected slots + original one-to-one pairs; not-selected is not an FP')
            if positive:category='position_valid'
            elif neg4:category='position_negative_original4m'
            elif neg1:category='observed_excluded_at1m_but_not_original4m'
            elif conflict:category='unknown_reference_competitor'
            else:category='missing_observation_or_exclusion_evidence'
            # Redundancy requires a valid nearby location that lost set matching.
            redundancy=bool(near and not conflict and any(e['identity']=='false_positive' for e in evals.values()))
            if redundancy:category='valid_position_set_redundancy'
            row=dict(source=v['source'],query_slot=q,position_m=lab['position_m'],training_identity=label,
                original_training_basis=lab['reason'],old_duplicate_source_name_only=lab['old_duplicate'],
                independent_position_negative4m=neg4,exclusion_at1m=neg1,observed_state=bg['query_observed_states'][q],
                original4m_reason=bg['reason'][q],confirmed_within1m=near,confirmed_within4m=c4['confirmed_structure_coverage_mask'][q],
                reference_inventory_bound=True,reference_universe=bg['inventory_definition'],whole_region_complete=False,
                evaluation=evals,category=category,true_set_redundancy=redundancy,
                source_binding=v['source_binding'],grid_sha256=v['grid_content_sha256'],target_record_sha256=c['target_record_sha256'])
            rows.append(row);stats[category]+=1
            for region,e in evals.items():
                if e['identity']=='false_positive':errors[region+'/'+category]+=1
            newlabels.append(dict(source=v['source'],query_slot=q,label=label,basis='confirmed positive retained' if positive else bg['reason'][q] if neg4 else 'same source rules do not authorize extra position label',changed=False))
            if label=='unknown':
                minimal.append(dict(source=v['source'],query_slot=q,category=category,observed_state=bg['query_observed_states'][q],
                    minimum_needed='no additional reference inventory needed to state 1m exclusion; new training meaning would require an explicit separate contract, not automatic relabeling' if neg1 else 'candidate-local observation/visibility and reference exclusion evidence; no whole-world relabeling',
                    annotation_performed=False))
    assert len(rows)==512 and len({(digest(x['source']),x['query_slot']) for x in rows})==512
    unknown=sum(x['label']=='unknown' for x in newlabels)
    summary=dict(status='GATE_MIXED',scope='evidence reconciliation complete; position supervision incomplete',candidates=512,
        observations=16,counts=dict(stats),selected_errors=dict(errors),unused_original4m_labels=0,
        changed_labels=0,retained_labels=dict(positive=12,negative=132,unknown=unknown),complete_local_supervision=False,
        true_set_redundancy_selected=sum(r['true_set_redundancy'] for r in rows),missing_inventory_binding=0,
        training_updates=0,new_inference=0,position_or_float_checks_repeated=False,old_4m_rule_changed=False,
        conclusion='1m exclusion evidence is not unused original4m supervision; unknown is retained. No training authorized.')
    OUT.mkdir(parents=True)
    for name,data in [('candidates512.json',rows),('summary.json',summary),('candidate_validity_v1r_evidence_only.json',dict(version='v1r_evidence_only',training_authorized=False,rule_changed=False,labels=newlabels)),('minimal_evidence_scope.json',minimal)]:
        (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    with (OUT/'candidates512.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=['task','sequence','query','training','old_eval','fixed_eval','category','old_duplicate_name','negative4m','excluded1m','observed_state']);writer.writeheader()
        for r in rows:writer.writerow(dict(task=r['source']['task'],sequence=r['source']['source_sequence_id'],query=r['query_slot'],training=r['training_identity'],old_eval=r['evaluation']['old']['identity'],fixed_eval=r['evaluation']['fixed']['identity'],category=r['category'],old_duplicate_name=r['old_duplicate_source_name_only'],negative4m=r['independent_position_negative4m'],excluded1m=r['exclusion_at1m'],observed_state=r['observed_state']))
    manifest=dict(reads_sha256=reads,tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),outputs_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir()})
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':main()
