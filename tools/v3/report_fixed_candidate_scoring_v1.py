"""Read-only sealed-run verification and fixed candidate scoring figure."""
from _bootstrap import PROJECT_ROOT as ROOT
from report_geometry_presence_a_v1 import verify, load
import json
import numpy as np
import torch

RUN=ROOT/'results/gate3_semantics/gate3_20260910_gse_fixed_candidate_scoring_v1_seed0'
OUT=ROOT/'docs/figures/gse_graph/fixed_candidate_scoring_20260910'

def main():
    assert not OUT.with_suffix('.json').exists()
    seal=verify(RUN)
    records=[]
    first=load(RUN/'metrics/candidate_scores_0000.json')
    for step in range(0,1001,100):
        scores=load(RUN/f'metrics/candidate_scores_{step:04d}.json')
        check=load(RUN/f'metrics/frozen_check_{step:04d}.json')
        assert check['all_queries']==512
        assert all(check[k] for k in ('all_nonexistence_parameters_equal','all_positions_bitwise_parent_equal','cache_full_forward_bitwise_equal'))
        for a,b in zip(first['observations'],scores['observations']):
            for k in ('source','assignment','positive','negative','unknown'): assert a[k]==b[k]
        records.append(dict(step=step,groups=scores['groups'],detection=load(RUN/f'metrics/evaluation_{step:04d}.json')['summary']))
    initial=torch.load(RUN/'checkpoints/step_0000.pt',map_location='cpu',weights_only=False)
    final=torch.load(RUN/'checkpoints/step_1000.pt',map_location='cpu',weights_only=False)
    initial=initial['model'];final=final['model']
    changed=[]
    for k in initial:
        if not torch.equal(initial[k],final[k]):
            assert k in ('head.head.anchor.weight','head.head.anchor.bias')
            assert torch.equal(initial[k][:3],final[k][:3]);changed.append(k)
    assert len(changed)==2
    logs=[json.loads(l) for l in (RUN/'logs/updates.jsonl').read_text().splitlines()]
    assert len(logs)==1000
    for r in logs: assert set(r['gradient_norms'])<=set(changed)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.text import Text
    fig,axs=plt.subplots(1,3,figsize=(16,4.7))
    x=[r['step'] for r in records]
    axs[0].plot(x,[12]*11,'--',label='固定位置一对一覆盖：12/12')
    axs[0].plot(x,[r['detection']['tp'] for r in records],'-o',label='评分后实际检出')
    axs[0].plot(x,[r['detection']['fp'] for r in records],'-s',label='确认误检')
    axs[0].set_title('候选没移动，评分仍未分清');axs[0].set_xlabel('固定训练批次')
    for k,label in [('positive','正例（12）'),('negative','确认负例（162）'),('unknown','未知（338，不监督）')]:
        axs[1].plot(x,[r['groups'][k]['mean'] for r in records],'-o',label=label)
        axs[2].plot(x,[r['groups'][k]['selected'] for r in records],'-o',label=label)
    axs[1].axhline(.5,color='gray',linestyle='--');axs[1].set_title('固定监督分组的平均分数')
    axs[2].set_title('分数达到原0.5阈值的候选数')
    for ax in axs: ax.legend();ax.grid(alpha=.2)
    for text in fig.findobj(Text): text.set_fontproperties(FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',size=9))
    fig.tight_layout();fig.savefig(OUT.with_suffix('.png'),dpi=160);plt.close(fig)
    result=dict(seal=seal,changed_parameter_tensors=changed,all_11_frozen_checks_pass=True,all_masks_fixed=True,updates=1000,records=records,decision='STOP_FIXED_CANDIDATE_SCORING_FIT_FAIL',new_training_started=False)
    OUT.with_suffix('.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(seal=seal,changed=changed,final=records[-1]),ensure_ascii=False))

if __name__=='__main__':main()
