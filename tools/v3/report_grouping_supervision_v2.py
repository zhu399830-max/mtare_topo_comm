"""Read-only sealed-run verification and descriptive learning-curve rendering.

No new model inference, teacher reads, scoring changes or checkpoint selection.
Outputs only to docs/figures, never back into the sealed runs.
"""
from _bootstrap import PROJECT_ROOT as ROOT
import hashlib
import json
import math
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.text import Text

RUN=ROOT/'results/gate3_semantics/gate3_20260909_gse_grouping_center_supervision_v2_seed0'
PARENT=ROOT/'results/gate3_semantics/gate3_20260909_gse_grouping_center_fit_v1r_seed0'
OUT=ROOT/'docs/figures/gse_graph'


def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def verify(run):
    seal=run/'artifacts/evidence_sha256.txt';files=set()
    for line in seal.read_text().splitlines():
        h,p=line.split('  ',1);p=ROOT/p
        if sha(p)!=h:raise ValueError('sealed evidence drift: '+str(p))
        files.add(p.resolve())
    if files!={p.resolve() for p in run.rglob('*') if p.is_file() and p!=seal}:
        raise ValueError('sealed file inventory drift')
    return dict(files=len(files),seal_sha256=sha(seal))


def main():
    outjson=OUT/'supervision_repair_audit_20260909_fontfix.json'
    figure=OUT/'supervision_repair_learning_20260909_fontfix.png'
    if outjson.exists() or figure.exists():raise FileExistsError('immutable report outputs exist')
    verification=dict(parent=verify(PARENT),rerun=verify(RUN))
    audits={};checked=0
    for prefix,label in [('parent_','old'),('','new')]:
        audits[label]=[]
        for step in range(0,1001,100):
            a=load(RUN/'metrics'/f'{prefix}supervision_audit_{step:04d}.json')
            for o in a['observations']:
                for q in o['queries']:
                    assert q['old_position_gradient']==q['new_position_gradient']
                    if q['new_negative'] and not q['old_negative']:
                        assert q['observed_inside'] and not q['unconfirmed_conflict'] and q['duplicate_candidate']
                    if not q['new_negative'] and not q['training_positive']:assert q['new_presence_gradient']==0
                    if q['new_negative']:assert q['new_presence_gradient']>0
                    checked+=1
            audits[label].append(a)
    logs=[json.loads(x) for x in (RUN/'logs/updates.jsonl').read_text().splitlines()]
    assert len(logs)==1000 and logs[-1]['optimizer_steps']==1000
    assert all(math.isfinite(v) for r in logs for v in r['gradient_norms'].values())
    assert not any('head.head.branch' in k for r in logs for k in r['gradient_norms'])
    assert load(RUN/'config/schedule.json')==load(PARENT/'config/schedule.json')
    changes=0;flips=0
    for a,b in zip(audits['new'][7]['observations'],audits['new'][8]['observations']):
        assert a['source']==b['source']
        changes+=([t['assigned_slot'] for t in a['targets']] != [t['assigned_slot'] for t in b['targets']])
        flips+=sum(q['new_negative']!=r['new_negative'] for q,r in zip(a['queries'],b['queries']))
    curves={}
    for label,steps in audits.items():
        curves[label]=[dict(a['summary'],nearest_mean_m=statistics.mean(t['raw_nearest_distance_m'] for o in a['observations'] for t in o['targets']),
            position_loss_positive_mean=statistics.mean(o['losses'][label]['position'] for o in a['observations'] if o['targets']),
            presence_loss_all_observation_mean=statistics.mean(o['losses'][label]['presence'] for o in a['observations']),
            projected_queries=sum(q['projected'] for o in a['observations'] for q in o['queries'])) for a in steps]
    plt.rcParams['font.sans-serif']=['Noto Sans CJK SC','DejaVu Sans']
    plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for label,title,color in [('old','修复前','#d55e00'),('new','本次修复','#0072b2')]:
        rows=curves[label];x=[r['step'] for r in rows]
        axes[0].plot(x,[r['nearest_mean_m'] for r in rows],'-o',label=title,color=color)
        axes[1].plot(x,[r['tp'] for r in rows],'-o',label=title,color=color)
        axes[2].plot(x,[r['fp'] for r in rows],'-o',label=title+'：已知误检',color=color)
        axes[2].plot(x,[r['ignored'] for r in rows],'--',label=title+'：未知预测',color=color)
    axes[0].set_title('全部12个参考的最近候选误差');axes[0].set_ylabel('米')
    axes[1].set_title('1米内正确检测数（最多12）');axes[1].set_ylim(-.5,12.5)
    axes[2].set_title('误检减少≠未知已解决')
    for ax in axes:
        ax.set_xlabel('实际更新次数');ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.suptitle('固定16样本：保留所有检查点，只采用第1000步结论',fontsize=14)
    for text in fig.findobj(Text):
        text.set_fontproperties(FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',size=text.get_fontsize()))
    fig.tight_layout();OUT.mkdir(parents=True,exist_ok=True);fig.savefig(figure,dpi=170);plt.close(fig)
    result=dict(verification=verification,checked_candidate_records=checked,
        source_script_sha256=sha(Path(__file__)),figure_sha256=sha(figure),
        optimizer_steps=1000,all_parameter_gradients_finite=True,branch_gradients=0,same_schedule=True,
        step700_to800=dict(changed_positive_assignments=changes,changed_negative_query_memberships=flips,
            causal_attribution='correlated instability evidence, not isolated causal proof'),
        curves=curves,cases=load(RUN/'metrics/fixed_nine_error_outcomes.json'),
        method_advantage_proven=False,representation_no_go=False)
    outjson.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(verification=verification,checked_candidate_records=checked,figure=str(figure),report=str(outjson))))


if __name__=='__main__':main()
