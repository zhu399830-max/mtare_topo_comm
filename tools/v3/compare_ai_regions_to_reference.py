"""Read-only join of frozen point-region annotations and existing source claims."""
from _bootstrap import PROJECT_ROOT as ROOT
import gzip,hashlib,json
RUN='results/gate3_semantics/gate3_20260911_gse_ai_region_annotation_v1_seed0'
OUT='docs/figures/gse_supervision_acquisition_pilot_v1/ai_region_reference_comparison.json'

def read(path,digest):
    b=(ROOT/path).read_bytes()
    if hashlib.sha256(b).hexdigest()!=digest:raise ValueError('drift '+path)
    return json.loads(gzip.decompress(b) if path.endswith('.gz') else b)

def main():
    card=json.loads((ROOT/'configs/v3/gate3/data_cards/gse_acquisition_junction_review_v1.json').read_text())
    pins={p:h for h,p in (l.split('  ',1) for l in (ROOT/RUN/'artifacts/evidence_sha256.txt').read_text().splitlines())}
    rows=[]
    roles=['entering_witness_ray_indices','interior_witness_ray_indices','surface_entry_witness_ray_indices','surface_departure_witness_ray_indices']
    for i,e in enumerate(card['entries']):
        p=RUN+f'/artifacts/case_{i}_regions.json';ai=read(p,pins[p]);ref=read(e['target_path'],e['target_sha256'])['produced_targets']
        frames=e['frame_rows'];sets=[]
        for region in ai['regions']:
            ids={frames.index(p['frame_row'])*11520+p['ring']*720+p['azimuth_bin'] for p in region['points']}
            if len(ids)!=len(region['points']):raise ValueError('duplicate selected identity')
            sets.append(ids)
        anchors=[]
        for n,a in enumerate(ref['teacher_provenance']['anchors']):
            branches=[];witness=[]
            for j,interface in enumerate(a['interface_ids']):
                w=set(a['witness_ray_indices'][j]);witness.append(w)
                branches.append(dict(interface=interface,total_five_frame_witnesses=len(w),current_frame_witnesses=sum(r//11520==4 for r in w),
                    selected_region_overlap=[len(w&s) for s in sets],
                    selected_region_role_overlap={role:[len(set(a[role][j])&s) for s in sets] for role in roles}))
            selected=set().union(*sets);multiplicity={r:sum(r in w for w in witness) for r in selected}
            counts={str(k):sum(v==k for v in multiplicity.values()) for k in range(len(witness)+1)}
            anchors.append(dict(reference_node=a['node_id_teacher_only'],position_m=ref['record']['anchors'][n]['position_m'],branches=branches,
                selected_ray_branch_multiplicity=counts,multi_claim_examples=[r for r in sorted(selected) if multiplicity[r]>1][:10],
                note='A witness may support several construction interfaces; this is not point-instance membership or independently visible branch count.'))
        rows.append(dict(case=i,task=e['source']['task'],annotation_sha256=pins[p],reference_sha256=e['target_sha256'],
            selected_counts=[len(s) for s in sets],selected_history_slots=[4],anchors=anchors,
            existing_relation_matrix=ref['record']['membership'],ai_relationship_judgment=ai['relationship_judgment'],
            labels_changed=0))
    output=dict(status='READ_ONLY_SOURCE_VS_OBSERVATION_COMPARISON',cases=rows,new_labels=0,training_steps=0,
        scope='Only selected current-frame regions; noncoverage is not evidence of absent five-frame support',
        limitation='Source-ray overlap neither independently validates nor disproves structural observability; AI unknown is not contradiction.')
    with (ROOT/OUT).open('x') as f:json.dump(output,f,ensure_ascii=False,indent=2)
    for r in rows:
        print(json.dumps(dict(case=r['case'],selected_counts=r['selected_counts'],anchors=[dict(position=a['position_m'],overlaps=[b['selected_region_overlap'] for b in a['branches']],multiplicity=a['selected_ray_branch_multiplicity']) for a in r['anchors']])) )

if __name__=='__main__':main()
