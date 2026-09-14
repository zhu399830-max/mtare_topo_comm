"""Reduce sealed three-strength evidence; no additional optimization/inference."""
from _bootstrap import PROJECT_ROOT as ROOT
import json
import numpy as np
import torch
from report_geometry_presence_a_v1 import load,verify

RUN=ROOT/'results/gate3_semantics/gate3_20260910_gse_fixed_score_l2_v1_seed0'
OUT=ROOT/'docs/figures/gse_graph/fixed_score_l2_20260910'

def main():
    assert not OUT.with_suffix('.json').exists()
    seal=verify(RUN);summary=load(RUN/'metrics/summary.json');tables={};thresholds={};masks=None
    assert summary['numerical_solves']==3 and not summary['baseline_frozen']
    for r in summary['results']:
        tag=format(r['lambda_l2'],'.0e');old=load(RUN/f'metrics/old_evaluation_{tag}.json');fixed=load(RUN/f'metrics/fixed_region_{tag}.json')
        checks=load(RUN/f'metrics/full_forward_{tag}.json');assert checks['all_queries']==512 and len(checks['checks'])==16
        assert all(all(c[k] for k in ('feature_bitwise_equal','position_bitwise_equal','cache_cuda_full_bitwise_equal','forward_label_fields_absent')) for c in checks['checks'])
        current=[(c['source'],c['fixed_scoreable_mask'],c['fixed_allowed_mask']) for c in fixed['observations']]
        if masks is None:masks=current
        else:assert current==masks
        z=np.load(RUN/f'artifacts/head_{tag}.npz')
        kinds=('logits_float64','logits_cpu','logits_cuda','logits_full')
        selected=[(torch.from_numpy(z[k]).sigmoid().numpy()>=.5) for k in kinds]
        assert all(np.array_equal(s,selected[0]) for s in selected)
        thresholds[tag]=dict(actual_sigmoid_threshold_all512_equal=True,selected=int(selected[0].sum()))
        tables[tag]=dict(old=old['summary'],fixed_region=fixed['summary'],optimization=load(RUN/f'metrics/optimization_{tag}.json'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.text import Text
    fig,axs=plt.subplots(1,3,figsize=(16,4.8));x=np.arange(3)
    axs[0].bar(x-.16,[r['old']['tp'] for r in summary['results']],.32,label='旧1米正确数')
    axs[0].bar(x+.16,[r['old']['fp'] for r in summary['results']],.32,label='旧1米确认误检')
    axs[0].axhline(12,color='gray',linestyle='--');axs[0].set_xticks(x,['λ=0.01','λ=0.0001','λ=0.000001']);axs[0].set_title('三档均未成为有效检测基线');axs[0].legend()
    for tag in tables:
        t=tables[tag]['fixed_region'];rad=[.5,1.,2.,4.]
        axs[1].plot(rad,[t[str(v)]['precision'] for v in rad],'-o',label='λ='+tag)
        axs[2].semilogy(range(1,len(tables[tag]['optimization']['history'])+1),[v['total'] for v in tables[tag]['optimization']['history']],label='λ='+tag)
    axs[1].axhline(.9,color='gray',linestyle='--');axs[1].set_ylim(0,1);axs[1].set_title('评分区域固定后，只改变匹配距离');axs[1].set_xlabel('匹配距离（米）');axs[1].set_ylabel('精确率');axs[1].legend()
    axs[2].set_title('求解均收敛，不是继续等训练');axs[2].set_xlabel('各档数值迭代');axs[2].set_ylabel('含L2的本档目标（不可跨档直接排名）');axs[2].legend()
    for a in axs:a.grid(alpha=.2)
    for t in fig.findobj(Text):t.set_fontproperties(FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',size=9))
    fig.tight_layout();fig.savefig(OUT.with_suffix('.png'),dpi=160);plt.close(fig)
    result=dict(seal=seal,summary=summary,tables=tables,actual_threshold_checks=thresholds,fixed_masks_identical_across_all_strengths=True,
        full_forward_checks=48,no_model_search=True,no_new_solve=True)
    OUT.with_suffix('.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(seal=seal,thresholds=thresholds,decision='STOP_FIXED_FEATURE_SCORING_IMPLEMENTATION'),ensure_ascii=False))

if __name__=='__main__':main()
