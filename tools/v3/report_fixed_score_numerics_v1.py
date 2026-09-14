"""Post-run evidence reduction only. Never changes weights or invokes a solver."""
from _bootstrap import PROJECT_ROOT as ROOT
import json
import numpy as np
from report_geometry_presence_a_v1 import verify,load
from mtare_topo.governance_surface_selection import digest

RUN=ROOT/'results/gate3_semantics/gate3_20260910_gse_fixed_score_numerics_v1r_seed0'
PARENT=ROOT/'results/gate3_semantics/gate3_20260910_gse_fixed_candidate_scoring_v1_seed0'
OUT=ROOT/'docs/figures/gse_graph/fixed_score_numerics_20260910'

def main():
    assert not OUT.with_suffix('.json').exists()
    seals=dict(numerics=verify(RUN),parent=verify(PARENT),zero_solve_failure=verify(ROOT/'results/gate3_semantics/gate3_20260910_gse_fixed_score_numerics_v1_seed0'))
    d=np.load(RUN/'artifacts/fixed_design.npz');a=np.load(RUN/'artifacts/solved_affine.npz')
    evaluation=load(RUN/'metrics/evaluation_solved.json');opt=load(RUN/'metrics/optimization.json')
    old=load(PARENT/'metrics/candidate_scores_1000.json');z=[];z64=[];cases=[];offset=0
    for row,oldrow in zip(evaluation['observations'],old['observations']):
        assert row['source']==oldrow['source'];key=digest(row['source']);p=np.load(RUN/'artifacts'/('solved_prediction_'+key+'.npz'))
        known=p['positive']|p['negative'];zero=np.flatnonzero(known & (p['presence_logits']==0))
        cases.append(dict(source=row['source'],detection=row['scores']['1.0'],known_boundary_slots=zero.tolist(),
            positive_logits_float32=p['presence_logits'][p['positive']].tolist(),positive_logits_float64=p['presence_logits_float64'][p['positive']].tolist(),
            negative_selected=int((p['presence_logits'][p['negative']]>=0).sum()),unknown_selected=int((p['presence_logits'][p['unknown']]>=0).sum())))
        z.extend(p['presence_logits']);z64.extend(p['presence_logits_float64']);offset+=len(known)
    assert offset==512
    z=np.array(z);z64=np.array(z64);known=d['weights']>0;theta=a['theta_float64'];X=d['X'];w=d['weights'];y=d['y']
    stability=dict(head_l2_norm=float(np.linalg.norm(theta)),head_max_absolute=float(np.max(abs(theta))),
        smallest_design_singular_value=opt['conditioning']['singular_values'][-1],
        condition_ratio=opt['conditioning']['singular_values'][0]/opt['conditioning']['singular_values'][-1],
        all_query_max_logit_fp32_vs_fp64_error=float(np.max(abs(z-z64))),known_zero_logits=int(np.sum(z[known]==0)),
        float32_actual_weighted_loss=float(w@np.logaddexp(0.,-y*z)),
        float64_with_float32_weights_loss=float(w@np.logaddexp(0.,-y*(X@theta.astype(np.float32).astype(float)))),
        parent_final_weighted_loss=float(w@np.logaddexp(0.,-y*(X@a['initial_theta']))),
        numerically_fragile=True,robust_float32_success_claim=False,
        kernel='original affine layer float32 CPU; original GPU head execution not validated for solved large coefficients',
        threshold_unchanged=True,no_solver_or_head_update=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.text import Text
    fig,ax=plt.subplots(1,3,figsize=(16,4.8))
    ax[0].semilogy([h['iteration'] for h in opt['history']],[h['loss'] for h in opt['history']],'-o',markersize=3)
    ax[0].set_title('一次全批求解：32次迭代');ax[0].set_xlabel('数值迭代（不是重新训练1000批）');ax[0].set_ylabel('原加权损失：float64')
    for mask,c,label in [(d['positive'],'#0072b2','12个正例'),(d['negative'],'#d55e00','162个确认负例')]:
        ax[1].scatter(z64[mask],z[mask],s=14,color=c,label=label,alpha=.7)
    ax[1].axhline(0,color='gray',linestyle='--');ax[1].axvline(0,color='gray',linestyle='--')
    ax[1].set_xlabel('float64分数（sigmoid之前）');ax[1].set_ylabel('原float32线性头分数');ax[1].set_title('标签全对，但一个正例落在阈值上');ax[1].legend()
    xx=np.arange(4);ax[2].bar(xx-.18,[5,25,7,28],.36,label='第14节末轮');ax[2].bar(xx+.18,[12,0,0,152],.36,label='一次数值求解后')
    ax[2].set_xticks(xx,['正确','确认误检','漏检','未知预测']);ax[2].set_title('原1米部分参考评价；未知不是背景');ax[2].legend()
    for v in ax:v.grid(alpha=.2)
    for t in fig.findobj(Text):t.set_fontproperties(FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',size=9))
    fig.tight_layout();fig.savefig(OUT.with_suffix('.png'),dpi=160);plt.close(fig)
    result=dict(seals=seals,summary=load(RUN/'metrics/summary.json'),loss_equivalence=load(RUN/'metrics/loss_equivalence.json'),
        numerical_stability=stability,cases=cases,radii=evaluation['summary'],scope_status='GATE_MIXED_NUMERICAL_FIT_PASS_FLOAT32_FRAGILE',
        next='report only; no expanded training, network or mapping; exact stored feature separability is not robust generalization')
    OUT.with_suffix('.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(seals=seals,stability=stability,boundary_cases=[c for c in cases if c['known_boundary_slots']]),ensure_ascii=False))

if __name__=='__main__':main()
