"""Summarize sealed residuals; no inference, labels, tolerance or rerun."""
from _bootstrap import PROJECT_ROOT as ROOT
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUN=ROOT/'results/gate3_semantics/gate3_20260911_gse_membership_surface_residual_v1_seed0'
OUT=ROOT/'docs/figures/gse_membership_surface_residual_v1'


def main():
    sealed={}
    for line in (RUN/'artifacts/evidence_sha256.txt').read_text().splitlines():
        h,p=line.split('  ',1)
        if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h:raise ValueError('seal drift '+p)
        sealed[p]=h
    summary=json.loads((RUN/'metrics/summary.json').read_text())
    if summary['completed']!=16 or summary['error'] is not None:raise ValueError('complete16 required')
    rows=[];best_all=[];worst_all=[];delta_all=[];single=[];multiple=[]
    for i in range(16):
        with np.load(RUN/f'artifacts/window_{i:02d}.npz',allow_pickle=False) as z:
            a=z['records'];ids,inverse,counts=np.unique(a[:,0],return_inverse=True,return_counts=True)
            if not np.array_equal(ids,z['roi_return_indices']):raise ValueError('missing return')
            best=np.full(len(ids),np.inf);worst=np.zeros(len(ids))
            np.minimum.at(best,inverse,a[:,2]);np.maximum.at(worst,inverse,a[:,2])
            if not np.isfinite(a).all():raise ValueError('nonfinite residual')
            best_all.extend(best);worst_all.extend(worst);delta_all.extend(z['point_quantization_delta_m'])
            single.extend(best[counts==1]);multiple.extend(best[counts>1])
            rows.append(dict(observation=i,returns=len(ids),pairs=len(a),ambiguous_returns=int((counts>1).sum()),
                maximum_best_source_residual_m=float(best.max()),maximum_any_source_residual_m=float(worst.max())))
    q=lambda x:np.quantile(x,[0,.5,.9,.99,1]).tolist()
    result=dict(status='COMPLETED_DIAGNOSTIC_NOT_LABEL_QUALIFICATION',windows=rows,
        returns=len(best_all),pairs=sum(r['pairs'] for r in rows),ambiguous_returns=len(multiple),
        best_source_residual_quantiles_m=q(best_all),worst_source_residual_quantiles_m=q(worst_all),
        single_source_residual_quantiles_m=q(single),multiple_source_best_residual_quantiles_m=q(multiple),
        reported_vs_scene_delta_quantiles_m=q(delta_all),seal_entries_verified=len(sealed),
        original_status=summary['status'],elapsed_s=summary['elapsed_s'],peak_rss_bytes=summary['peak_rss_bytes'],
        labels_generated=0,training_steps=0,acceptance_distance_m=None,
        interpretation='Each return has a nearby candidate source surface, but active source sets are not exact surface ownership. Nearest face arc does not prove continuous local channel or structural membership.')
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'summary.json').open('x') as f:json.dump(result,f,indent=2)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    x=np.arange(16)
    axes[0].semilogy(x,[r['maximum_best_source_residual_m'] for r in rows],'o-',label='Closest coded source')
    axes[0].semilogy(x,[r['maximum_any_source_residual_m'] for r in rows],'s-',label='Farthest coded source')
    axes[0].set(xlabel='Window (fixed original order)',ylabel='Maximum surface distance (m)',title='All 16 windows: residuals, not label accuracy')
    axes[0].legend();axes[0].grid(alpha=.25)
    axes[1].bar(x,[r['ambiguous_returns'] for r in rows])
    axes[1].set(xlabel='Window (fixed original order)',ylabel='Returns with multiple coded sources',title='Ambiguity retained, not forced into one source')
    fig.savefig(OUT/'residuals.png',dpi=150);plt.close(fig)
    print(json.dumps({k:v for k,v in result.items() if k!='windows'}))


if __name__=='__main__':main()
