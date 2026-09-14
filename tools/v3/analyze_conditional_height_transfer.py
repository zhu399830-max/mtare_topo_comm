"""Saved-prediction attribution only. No model, refit, calibration or new labels."""
from _bootstrap import PROJECT_ROOT as ROOT
import json
from collections import Counter
import numpy as np
from ai_junction_pilot import sha,write
from mtare_topo.representation.gse_conditional_geometry_loss import compile_conditional_loss_targets

BASE='results/gate3_semantics/'
DEV=BASE+'gate3_20260911_gse_conditional_development_evaluation_v1_seed0'
TARGET=BASE+'gate3_20260911_gse_conditional_development_geometry_v1_seed0'
FIT=BASE+'gate3_20260911_gse_new12_axis_logit_correction_v1_seed0'
FIT_TARGET=BASE+'gate3_20260911_gse_new12_conditional_geometry_v1_seed0'
OUT='docs/figures/gse_conditional_geometry_fit_v1'


def checked(path,pins):
    if sha(ROOT/path)!=pins[path]:raise ValueError('saved evidence drift '+path)
    return ROOT/path


def stats(y,p,w):
    w=w/w.sum()
    ma=lambda x:float(w@x)
    signal=np.abs(y)>=.1
    sw=w[signal];sw=sw/sw.sum() if sw.size and sw.sum() else sw
    return dict(reference_abs_mean_m=ma(abs(y)),prediction_abs_mean_m=ma(abs(p)),mae_m=ma(abs(y-p)),
        amplitude_ratio=ma(abs(p))/ma(abs(y)) if ma(abs(y)) else None,
        prediction_abs_le_005m_mass=ma((abs(p)<=.05).astype(float)),
        abs_reference_ge_01m_mass=ma(signal.astype(float)),
        sign_agreement_on_abs_reference_ge_01m=float(sw@(np.sign(y[signal])==np.sign(p[signal]))) if len(sw) else None,
        wrong_sign_on_abs_reference_ge_01m=float(sw@(y[signal]*p[signal]<0)) if len(sw) else None,
        reference_abs_max_m=float(abs(y).max()),prediction_abs_max_m=float(abs(p).max()))


def load_population(entries,variants,pins):
    chunks=[]
    for e in entries:
        with np.load(checked(e['target'],pins),allow_pickle=False) as y:
            t=compile_conditional_loss_targets(y,'cpu');mask=(t.known&~t.same).numpy()[0]
            if not mask.any():continue
            weights=t.weights.numpy()[0][mask].astype(float);weights/=weights.sum()
            truth=t.height.numpy()[0][mask].astype(float)
            pp={}
            for v in variants:
                with np.load(checked(e['predictions'][v],pins),allow_pickle=False) as p:
                    if not np.array_equal(p['neighbors'][0],y['neighbor_index']):raise ValueError('query identity drift')
                    pp[v]=p['height'][0][mask].astype(float)
            chunks.append(dict(parent=e['parent'],case=e['case'],truth=truth,weights=weights,predictions=pp))
    counts=Counter(c['parent'] for c in chunks);nparents=len(counts)
    for c in chunks:c['weights']/=counts[c['parent']]*nparents
    truth=np.concatenate([c['truth'] for c in chunks]);weights=np.concatenate([c['weights'] for c in chunks])
    predictions={v:np.concatenate([c['predictions'][v] for c in chunks]) for v in variants}
    return chunks,truth,weights,predictions


def main():
    pins={}
    for base in (DEV,TARGET,FIT,FIT_TARGET):
        pins.update({p:h for h,p in (l.split('  ',1) for l in (ROOT/base/'artifacts/evidence_sha256.txt').read_text().splitlines())})
    dev=json.loads((ROOT/'configs/v3/gate3/data_cards/gse_conditional_development_evaluation_v1.json').read_text())['scope']['entries']
    fit=json.loads((ROOT/'configs/v3/gate3/data_cards/gse_new12_conditional_geometry_v1.json').read_text())['scope']['entries']
    de=[dict(case=e['identity']['case'],parent=e['identity']['parent_id'],target=e['target'],
        predictions={v:DEV+f"/artifacts/{v}_{e['identity']['case']:03d}.npz" for v in 'ABC'}) for e in dev]
    fe=[dict(case=i,parent=e['parent'],target=FIT_TARGET+f'/artifacts/window_{i:02d}.npz',predictions={'C':FIT+f'/artifacts/C_step2000_{i:02d}.npz'}) for i,e in enumerate(fit)]
    fc,fy,fw,fp=load_population(fe,['C'],pins);dc,dy,dw,dp=load_population(de,list('ABC'),pins)
    result=dict(scope='12fit saved references/C predictions and240development saved references/ABC predictions only',
        model_forwards=0,training_steps=0,fit_cross_observations=len(fc),development_cross_observations=len(dc),
        diagnostic_bins='abs(reference) [0,.1,.5,1,2,infinity] m; sign restricted to abs(reference)>=.1m; near-zero |prediction|<=.05m. Descriptive, not new acceptance thresholds.',
        fit_C=stats(fy,fp['C'],fw),development={v:stats(dy,dp[v],dw) for v in 'ABC'},per_parent={},bins=[])
    for parent in sorted({c['parent'] for c in dc}):
        cs=[c for c in dc if c['parent']==parent];y=np.concatenate([c['truth'] for c in cs]);w=np.concatenate([c['weights'] for c in cs])
        result['per_parent'][parent]={v:stats(y,np.concatenate([c['predictions'][v] for c in cs]),w) for v in 'ABC'}
    bounds=[0.,.1,.5,1.,2.,float('inf')]
    for lo,hi in zip(bounds[:-1],bounds[1:]):
        mask=(abs(dy)>=lo)&(abs(dy)<hi);mass=float(dw[mask].sum())
        result['bins'].append(dict(lower_m=lo,upper_m=hi if np.isfinite(hi) else None,mass=mass,
            methods={v:stats(dy[mask],dp[v][mask],dw[mask]) for v in 'ABC'} if mass else {}))
    result['development_mass_beyond_fit_max_abs_reference']=float(dw[abs(dy)>abs(fy).max()].sum())
    result['limitations']='Output statistics identify underestimation/sign behavior, not unique internal cause or observability of this conditional target. No fitted rescaling, model updates, target revisions or graph qualification.'
    write(ROOT/OUT/'height_transfer_attribution.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams['font.family']=font_manager.FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc').get_name();plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,y,p,title in [(axes[0],fy,fp['C'],'原12例拟合人口'),(axes[1],dy,dp['C'],'完整240例开发人口')]:
        ax.scatter(y,p,s=5,alpha=.2);span=max(abs(y).max(),abs(p).max());ax.plot([-span,span],[-span,span],'k--',lw=1)
        ax.set(xlabel='构造参考中心高差（米）',ylabel='C模型预测高差（米）',title=title);ax.grid(alpha=.2)
    fig.suptitle('C模型高差迁移：显示全部跨组件可评分预测；不是地面高度或通道连通真值\n散点含相关的重复面片关系；定量统计按组件对、观察、父地图加权',fontsize=11)
    fig.tight_layout(rect=[0,0,1,.88]);path=ROOT/OUT/'height_transfer_attribution.png'
    if path.exists():raise FileExistsError('preserve existing figure')
    fig.savefig(path,dpi=150);plt.close(fig)
    print(json.dumps({k:result[k] for k in ['fit_C','development','development_mass_beyond_fit_max_abs_reference']},indent=2))


if __name__=='__main__':main()
