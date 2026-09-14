"""Index existing supervision by meaning; never export new point labels.

Opening crossing witnesses describe rays crossing a reference section, not
point-instance membership. Structural correspondence records remain distinct.
"""
from _bootstrap import PROJECT_ROOT as ROOT
import hashlib,json


def main():
    run=ROOT/'results/gate3_semantics/gate3_20260911_gse_membership_fit_v1r1_seed0'
    seals={p:h for h,p in (line.split('  ',1) for line in (run/'artifacts/evidence_sha256.txt').read_text().splitlines())}
    rows=[];counts={'openings':0,'known_positive':0,'known_negative':0,'unknown':0,'missing_known_evidence':0}
    inputs={}
    for i in range(16):
        relative=f'artifacts/reference_{i:02d}.json';path=run/relative;raw=path.read_bytes()
        assert hashlib.sha256(raw).hexdigest()==seals[relative]
        inputs[str(path.relative_to(ROOT))]=seals[relative]
        target=json.loads(raw);record=target['record'];provenance=target['teacher_provenance']
        if len(record['openings'])!=len(provenance['openings']):raise ValueError('opening provenance alignment')
        opening_rows=[]
        for j,p in enumerate(provenance['openings']):
            crossing=p['crossing_ray_indices'];surface=p['surface_ray_indices']
            assert crossing and surface and all(type(x) is int and 0<=x<57600 for x in crossing+surface)
            opening_rows.append(dict(index=j,source_teacher_only=p['primitive_id_teacher_only'],
                reference_arc_m=p['reference_arc_m'],position_source=p['position_source'],
                crossing_ray_count=len(crossing),surface_ray_count=len(surface),
                crossing_is_point_instance_label=False,source_identity_is_structural_membership=False))
        relations=[]
        for o,values in enumerate(record['membership']):
            for a,label in enumerate(values):
                field='unknown' if label is None else 'known_positive' if label else 'known_negative'
                counts[field]+=1
                collection='relations' if label is True else 'terminal_nonmembership' if label is False else None
                supports=[] if collection is None else [k for k,p in enumerate(provenance.get(collection,[]))
                    if p.get('opening_index')==o and p.get('anchor_index')==a]
                if label is not None and not supports:counts['missing_known_evidence']+=1
                relations.append(dict(opening_index=o,structure_index=a,original_membership=label,
                    evidence_collection=collection,evidence_indices=supports,
                    new_label_generated=False,point_mask_derived=False,global_edge_inferred=False))
        counts['openings']+=len(opening_rows)
        rows.append(dict(observation=i,source_binding=target['source_binding'],openings=opening_rows,relations=relations))
    result=dict(scope='Saved16 reference semantics only',counts=counts,observations=rows,input_sha256=inputs,
        point_mask_export_qualified=False,new_training_qualified=False,
        missing_bridge='Observed point/patch to local-channel association with ambiguity and visible support; structural relation must retain its separate evidence',
        teacher_calls=0,scan_reads=0,training_steps=0)
    path=ROOT/'docs/figures/gse_membership_fit_v1r1/reference_semantics.json'
    with path.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(counts))


if __name__=='__main__':main()
