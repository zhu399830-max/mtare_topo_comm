"""Paper-retainable comparison of two completed fixed-cohort fitting runs.

This renders saved predictions only. No checkpoints, scans, training or new
thresholds. Cases retain original order; neither successful cases nor epochs
are selected. Raw errors are not claimed to be independent development results.
"""
from _bootstrap import PROJECT_ROOT as ROOT
import hashlib,json
import numpy as np


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=['gse_membership_fit_v1r1','gse_membership_fit_gradient_isolated_v1']
    runs=[ROOT/'results/gate3_semantics'/('gate3_20260911_'+name+'_seed0') for name in names]
    sources={};payload=[];records=[]
    for run in runs:
        for line in (run/'artifacts/evidence_sha256.txt').read_text().splitlines():
            h,p=line.split('  ',1)
            assert hashlib.sha256((run/p).read_bytes()).hexdigest()==h,p
            sources[str((run/p).relative_to(ROOT))]=h
        payload.append(json.loads((run/'artifacts/predictions.json').read_text()))
        records.append([json.loads((run/f'artifacts/reference_{i:02d}.json').read_text())['record'] for i in range(16)])
    assert records[0]==records[1]
    assert payload[0]['schedule']==payload[1]['schedule']
    comparison=[]
    for i,record in enumerate(records[0]):
        row={'observation':i}
        for role in ('anchor','opening'):
            truth=np.asarray([r['position_m'] for r in record[role+'s']],float).reshape(-1,3)
            values=[]
            for data in payload:
                points=np.asarray(data['final'][i]['all_predictions'][role+'_position_m'],float)
                distance=np.linalg.norm(points[:,None]-truth[None],axis=-1)
                values.append(dict(mean_nearest_m=float(distance.min(axis=0).mean()),
                    correct=data['final'][i][role]['correct'],references=len(truth)))
            row[role]=values
        comparison.append(row)
    out=ROOT/'docs/figures/gse_membership_fit_v1r1'
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    labels=['Joint training','Gradient isolated']
    for ax,role in zip(axes[:2],('anchor','opening')):
        for branch,label in enumerate(labels):
            ax.plot(range(16),[r[role][branch]['mean_nearest_m'] for r in comparison],marker='.',label=label)
        ax.axhline(1,color='black',linestyle='--',linewidth=1)
        ax.set(title=role+' position error',xlabel='Original observation index (all16)',ylabel='Mean nearest distance (m)')
        ax.legend()
    metrics=[]
    for p in payload:
        rows=p['final']
        metrics.append([sum(r['anchor']['correct'] for r in rows),sum(r['opening']['correct'] for r in rows),
                        sum(r['membership']['positive_correct'] for r in rows),sum(r['membership']['negative_correct'] for r in rows)])
    x=np.arange(4)
    for branch,label in enumerate(labels):
        bars=axes[2].bar(x+(branch-.5)*.35,metrics[branch],.35,label=label)
        axes[2].bar_label(bars)
    axes[2].set_xticks(x,['Anchor\n/24','Opening\n/24','Positive\n/22','Negative\n/13'])
    axes[2].set(title='Correct counts: fixed1m / score0.5',ylabel='Correct partial references',ylim=(0,24))
    axes[2].legend();fig.suptitle('Fixed16 fitting diagnosis, NOT an independent method comparison')
    fig.tight_layout();image=out/'joint_vs_isolated.png';fig.savefig(image,dpi=160);plt.close(fig)
    result=dict(observations=comparison,correct_counts=metrics,source_sha256=sources,
        tool_sha256=hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest(),
        figure_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),training_steps=0,
        conclusion='Corrective configuration did not meet fit criteria. No geometry advantage or graph qualification.')
    with (out/'joint_vs_isolated.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({'counts':metrics,'figure':str(image.relative_to(ROOT))}))


if __name__=='__main__':main()
