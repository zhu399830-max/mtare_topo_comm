"""Saved shared-ray/interface ordering coverage, not a new geometry teacher."""
from _bootstrap import PROJECT_ROOT as ROOT
import json,gzip,hashlib
import numpy as np

def main():
    card=json.loads((ROOT/'configs/v3/gate3/data_cards/gse_acquisition_junction_review_v1.json').read_text())
    rows=[]
    for e in card['entries']:
        raw=(ROOT/e['target_path']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=e['target_sha256']:raise ValueError('reference drift')
        full=json.loads(gzip.decompress(raw));a=full['produced_targets']['teacher_provenance']['anchors'][0]
        ids=a['interface_ids'];support=[set(x) for x in a['witness_ray_indices']]
        hits={}
        for h in full['raw_interfaces']['raw_interface_intersections']:
            if h['inside_roi']:hits.setdefault(h['ray_index'],{}).setdefault(h['interface_id_teacher_only'],[]).append(h['t'])
        pairs=[]
        for i,j in [(0,1),(0,2),(1,2)]:
            shared=sorted(support[i]&support[j]);both=[];one=0;neither=0;gaps=[];examples=[]
            for ray in shared:
                h=hits.get(ray,{});x=h.get(ids[i],[]);y=h.get(ids[j],[])
                if x and y:
                    both.append(ray);gaps.append(min(abs(t-u) for t in x for u in y))
                    if len(examples)<3:examples.append(dict(ray_index=ray,ordered=sorted([(t,ids[i]) for t in x]+[(t,ids[j]) for t in y])))
                elif x or y:one+=1
                else:neither+=1
            pairs.append(dict(interfaces=[ids[i],ids[j]],shared_rays=len(shared),both_saved_interface_hits=len(both),one_saved_interface_hit=one,
                neither_saved_interface_hit=neither,minimum_separation_m=None if not gaps else float(min(gaps)),
                median_separation_m=None if not gaps else float(np.median(gaps)),first_three_examples=examples))
        rows.append(dict(task=e['source']['task'],pairs=pairs,
            interpretation='Ordered raw interface hits are geometry intersections, not proof of enter/depart orientation or complete structure; lateral surface witnesses can lack these cap-interface hits. Missing is not false.'))
    out=ROOT/'docs/figures/gse_supervision_acquisition_pilot_v1/junction_crossing_coverage.json'
    with out.open('x') as f:json.dump(dict(rows=rows,teacher_calls=0,training_steps=0,new_labels=0),f,ensure_ascii=False,indent=2)
    for r in rows:print(json.dumps(dict(task=r['task'],pairs=[{k:v for k,v in p.items() if k!='first_three_examples'} for p in r['pairs']]),ensure_ascii=False))

if __name__=='__main__':main()
