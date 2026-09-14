"""Frozen prediction diagnosis; no inference, training, threshold selection or writes."""
import _bootstrap
import hashlib,io,json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'results/gate3_semantics/gate3_20260908_gse_synthetic_fit_v2_seed0'
SEAL='158da6743e0e17086a9115fbd0557b7697cb5c7a6674a94ee934549c1c053334'


def main():
    seal=(RUN/'artifacts/evidence_sha256.txt').read_bytes()
    if hashlib.sha256(seal).hexdigest()!=SEAL:raise ValueError('result seal drift')
    pins={p:h for h,p in (line.split('  ',1) for line in seal.decode().splitlines())}
    def load(name):
        p=RUN/'artifacts'/name;b=p.read_bytes()
        if hashlib.sha256(b).hexdigest()!=pins[str(p.relative_to(ROOT))]:raise ValueError('result bytes drift')
        return torch.load(io.BytesIO(b),map_location='cpu',weights_only=True)
    result=load('paired_result.pt');examples=load('shared_examples.pt')
    for branch,stages in result['evaluations'].items():
        groups=defaultdict(list);extra=defaultdict(int)
        for example,value in zip(examples,stages['final'],strict=True):
            p=value['prediction'];t=example['targets'];case=example['header']['observation_id']
            for kind in ('anchor','opening'):
                xyz=p[kind+'_position_m'][0].numpy();prob=p[kind+'_presence_logits'][0].sigmoid().numpy()
                truth=t[kind+'_position_m'][0][t[kind+'_valid'][0]].numpy()
                assigned=set()
                if len(truth):
                    distance=np.linalg.norm(truth[:,None]-xyz[None],axis=-1)
                    rows,cols=linear_sum_assignment(distance)
                    for i,j in zip(rows,cols):
                        assigned.add(int(j));groups[(kind,case.split('__')[0])].append((float(distance[i,j]),float(prob[j])))
                outside=np.linalg.norm(xyz,axis=1)>10.
                unassigned=np.array([j not in assigned for j in range(len(xyz))])
                extra[kind+'_unmatched_outside_high_confidence']+=int((unassigned&outside&(prob>=.5)).sum())
                extra[kind+'_matched_outside']+=int((~unassigned&outside).sum())
        output={}
        for (kind,program),values in groups.items():
            a=np.asarray(values)
            output[kind+'/'+program]=dict(targets=len(a),median_assignment_error_m=float(np.median(a[:,0])),
                maximum_assignment_error_m=float(a[:,0].max()),median_assigned_confidence=float(np.median(a[:,1])),
                located_within_1m=int((a[:,0]<=1).sum()),located_but_below_confidence=int(((a[:,0]<=1)&(a[:,1]<.5)).sum()))
        print(json.dumps(dict(branch=branch,assignment_geometry_only=True,groups=output,unpenalized_candidates=dict(extra),
            changed_metrics=False,optimizer_steps=0)),flush=True)


if __name__=='__main__':main()
