"""Independently check saved contradiction witnesses against frozen predictions."""
from _bootstrap import PROJECT_ROOT as ROOT
from explain_local_conflict_replay import OUT,RUN,FIT,REVIEW
from run_short_observation_chain import write,sha
import json
import numpy as np

def main():
    rows=[]
    for c in json.loads((OUT/'rejection_chronology.json').read_text())['cases']:
        i=c['observation']
        with np.load(FIT/f'artifacts/prediction_0500_{i:02d}.npz',allow_pickle=False) as d:
            ids=d['ray_ids'];lookup={tuple(sorted(map(int,ids[p]))):float(s) for p,s in zip(d['pairs'],d['probabilities'])}
        with np.load(REVIEW/f'artifacts/ai_core_{i:02d}.npz',allow_pickle=False) as d:labels=dict(zip(d['ray_ids'].tolist(),d['core_labels'].tolist()))
        decisions=[json.loads(s) for s in (RUN/f'artifacts/decisions_{i:02d}.jsonl').read_text().splitlines()]
        a,b=c['rejected_edge'];u,v=c['witness']['ray_ids']
        assert labels[a]==labels[b]>=0
        assert lookup[tuple(sorted((a,b)))]==c['p_same']>=np.float32(.9)
        assert decisions[c['decision_index']]==[a,b,c['p_same'],'REJECT_REPULSION']
        assert lookup[tuple(sorted((u,v)))]==c['witness']['p_same']<=np.float32(.1)
        known=unknown=0
        for start,end,path in [(a,u,c['accepted_path_left']),(b,v,c['accepted_path_right'])]:
            cursor=start
            for edge in path:
                x,y=edge['ray_ids'];assert x==cursor;cursor=y
                assert lookup[tuple(sorted((x,y)))]==edge['p_same']>=np.float32(.9)
                decision=decisions[edge['decision_index']]
                assert set(decision[:2])=={x,y} and decision[2]==edge['p_same'] and decision[3]=='ACCEPT_CONSISTENT'
                assert edge['decision_index']<c['decision_index']
                if labels[x]>=0 and labels[y]>=0:
                    assert labels[x]==labels[y];known+=1
                else:unknown+=1
            assert cursor==end
        rows.append(dict(observation=i,verified=True,known_positive_path_edges=known,unknown_path_edges=unknown,
            conclusion='Transitivity of both accepted attraction paths plus the known core attraction contradicts the retained repulsion. At least one relation in this certificate cannot be a valid same-group partition constraint; partial labels do not identify the incorrect unknown relation.'))
    write(OUT/'certificate_verification.json',dict(cases=rows,training_steps=0,labels_added=0,code_sha256=sha(ROOT/'tools/v3/verify_conflict_certificates.py'),certificate_sha256=sha(OUT/'rejection_chronology.json')))
    print(json.dumps(rows,ensure_ascii=False))

if __name__=='__main__':main()
