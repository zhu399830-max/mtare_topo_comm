"""Explain saved rejection chronology without changing predictions or grouping."""
from _bootstrap import PROJECT_ROOT as ROOT
from explain_local_conflict_replay import RUN, FIT, REVIEW, OUT
from run_short_observation_chain import write, sha
from collections import deque
import json
import numpy as np


def main():
    cases=json.loads((OUT/'split_attribution.json').read_text())['cases']
    results=[]
    for case in cases:
        i=case['observation'];target=tuple(case['same_core_positive_cut']['ray_ids'])
        with np.load(FIT/f'artifacts/prediction_0500_{i:02d}.npz',allow_pickle=False) as d:
            ids=d['ray_ids'];pairs=d['pairs'];prob=d['probabilities']
        with np.load(REVIEW/f'artifacts/ai_core_{i:02d}.npz',allow_pickle=False) as d:
            assert np.array_equal(ids,d['ray_ids']);labels=dict(zip(ids.tolist(),d['core_labels'].tolist()))
        parent={int(x):int(x) for x in ids};forest={int(x):[] for x in ids}
        def root(x):
            while parent[x]!=x:
                parent[x]=parent[parent[x]];x=parent[x]
            return x
        def path(a,b):
            previous={a:None};todo=deque([a])
            while todo and b not in previous:
                u=todo.popleft()
                for v,p,step in forest[u]:
                    if v not in previous:previous[v]=(u,p,step);todo.append(v)
            assert b in previous
            answer=[];v=b
            while previous[v] is not None:
                u,p,step=previous[v]
                answer.append(dict(ray_ids=[u,v],p_same=p,decision_index=step,reference_labels=[labels[u],labels[v]]));v=u
            return answer[::-1]
        log=RUN/f'artifacts/decisions_{i:02d}.jsonl'
        for step,line in enumerate(log.read_text().splitlines()):
            a,b,p,action=json.loads(line)
            if (a,b)==target:
                assert action=='REJECT_REPULSION'
                ra,rb=root(a),root(b);witness=[]
                for j in np.flatnonzero(prob<=np.float32(.1)):
                    u,v=map(int,ids[pairs[j]])
                    if root(u)==rb and root(v)==ra:u,v=v,u
                    if root(u)==ra and root(v)==rb:
                        witness.append((u,v,float(prob[j])))
                assert witness
                u,v,pn=min(witness,key=lambda e:(e[0],e[1]))
                results.append(dict(observation=i,rejected_edge=[a,b],p_same=p,decision_index=step,
                    action=action,active_repulsions_between_components=len(witness),
                    witness=dict(ray_ids=[u,v],p_same=pn,reference_labels=[labels[u],labels[v]]),
                    accepted_path_left=path(a,u),accepted_path_right=path(b,v),
                    interpretation='The retained repulsion and earlier accepted paths already forbid this correct core attraction at rejection time; unknown endpoints do not identify which model relation is wrong.'))
                break
            if action=='ACCEPT_CONSISTENT' and root(a)!=root(b):
                parent[root(b)]=root(a);forest[a].append((b,p,step));forest[b].append((a,p,step))
        else:raise AssertionError('saved rejection absent')
    write(OUT/'rejection_chronology.json',dict(cases=results,
        source_decision_hashes={str(i['observation']):sha(RUN/f"artifacts/decisions_{i['observation']:02d}.jsonl") for i in results},
        code_sha256=sha(ROOT/'tools/v3/trace_local_conflict_rejections.py'),training_steps_added=0,
        predictions_changed=False,labels_changed=False,thresholds_changed=False))
    print(json.dumps(results,ensure_ascii=False))


if __name__=='__main__':main()
