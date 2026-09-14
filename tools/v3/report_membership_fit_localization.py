"""Read-only decomposition of sealed v1r1 predictions; no model or data calls."""
from _bootstrap import PROJECT_ROOT as ROOT
import hashlib,json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

RUN=ROOT/'results/gate3_semantics/gate3_20260911_gse_membership_fit_v1r1_seed0'


def decompose(positions, logits, truth):
    p=np.asarray(positions,float);t=np.asarray(truth,float).reshape(-1,3)
    scores=1/(1+np.exp(-np.clip(np.asarray(logits,float),-700,700)))
    distances=np.linalg.norm(p[:,None]-t[None],axis=-1)
    def matched(indices):
        if not len(indices) or not len(t):return {}
        d=distances[indices];cost=(d>1.)*(1+d.max())*(1+min(d.shape))+d
        q,r=linear_sum_assignment(cost)
        return {int(ri):int(indices[qi]) for qi,ri in zip(q,r) if d[qi,ri]<=1.}
    all_matches=matched(np.arange(len(p)));filtered=matched(np.flatnonzero(scores>=.5))
    rows=[]
    for i in range(len(t)):
        nearest=int(np.argmin(distances[:,i]));near=np.flatnonzero(distances[:,i]<=1.)
        reason=('recovered' if i in filtered else 'no_prediction_within_1m' if not len(near)
                else 'all_nearby_below_score' if not np.any(scores[near]>=.5) else 'one_to_one_competition')
        rows.append(dict(reference_index=i,nearest_query=nearest,nearest_distance_m=float(distances[nearest,i]),
            nearest_score=float(scores[nearest]),nearby_queries=near.tolist(),reason=reason,
            all_query_match=all_matches.get(i),filtered_match=filtered.get(i)))
    return dict(rows=rows,all_query_correct=len(all_matches),filtered_correct=len(filtered),references=len(t),
        selected=int((scores>=.5).sum()),position_norm_range_m=[float(np.linalg.norm(p,axis=1).min()),float(np.linalg.norm(p,axis=1).max())])


def main():
    sealed={}
    for line in (RUN/'artifacts/evidence_sha256.txt').read_text().splitlines():
        digest,relative=line.split('  ',1)
        assert hashlib.sha256((RUN/relative).read_bytes()).hexdigest()==digest,relative
        sealed[relative]=digest
    predictions=json.loads((RUN/'artifacts/predictions.json').read_text())
    references=[json.loads((RUN/f'artifacts/reference_{i:02d}.json').read_text())['record'] for i in range(16)]
    result=dict(source_run=str(RUN.relative_to(ROOT)),sealed_files_verified=len(sealed),source_sha256=sealed,stages={})
    for stage in ('initial','final'):
        stage_rows=[]
        for i,(saved,record) in enumerate(zip(predictions[stage],references)):
            raw=saved['all_predictions'];row=dict(observation=i)
            for role in ('anchor','opening'):
                row[role]=decompose(raw[role+'_position_m'],raw[role+'_presence_logits'],[r['position_m'] for r in record[role+'s']])
                assert row[role]['filtered_correct']==saved[role]['correct']
            stage_rows.append(row)
        totals={}
        for role in ('anchor','opening'):
            rr=[r for row in stage_rows for r in row[role]['rows']]
            totals[role]=dict(all_query_correct=sum(x[role]['all_query_correct'] for x in stage_rows),
                filtered_correct=sum(x[role]['filtered_correct'] for x in stage_rows),references=len(rr),
                reasons={reason:sum(r['reason']==reason for r in rr) for reason in (
                    'recovered','no_prediction_within_1m','all_nearby_below_score','one_to_one_competition')},
                mean_nearest_distance_m=float(np.mean([r['nearest_distance_m'] for r in rr])))
        result['stages'][stage]=dict(totals=totals,observations=stage_rows)
    updates=[json.loads(x) for x in (RUN/'logs/updates.jsonl').read_text().splitlines()]
    result['loss_terms_first_last_update']=[{k:float(np.mean([r[k] for r in u['loss_terms']])) for k in u['loss_terms'][0]} for u in (updates[0],updates[-1])]
    output=ROOT/'docs/figures/gse_membership_fit_v1r1';output.mkdir(parents=True,exist_ok=True)
    target=output/'localization_decomposition.json'
    with target.open('x') as f:json.dump(result,f,indent=2)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,role in zip(axes,('anchor','opening')):
        for stage in ('initial','final'):
            values=sorted(r['nearest_distance_m'] for row in result['stages'][stage]['observations'] for r in row[role]['rows'])
            ax.plot(range(1,len(values)+1),values,label=stage)
        ax.axhline(1,color='black',linestyle='--',label='fixed 1m criterion')
        ax.set(title=role+' localization (24 partial references)',xlabel='Reference rank (sorted independently)',ylabel='Nearest prediction distance (m)');ax.legend()
    fig.suptitle('Saved predictions only; no retraining or threshold selection')
    fig.tight_layout();fig.savefig(output/'localization.png',dpi=150);plt.close(fig)
    print(json.dumps({k:v['totals'] for k,v in result['stages'].items()}))
    print(json.dumps(result['loss_terms_first_last_update']))


if __name__=='__main__':main()
