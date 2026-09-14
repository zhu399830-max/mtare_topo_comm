"""Read saved predictions/logs only; no forward, gradients or training."""
from _bootstrap import PROJECT_ROOT as ROOT
import hashlib,json
import numpy as np
import torch
from mtare_topo.representation.gse_partial_structure_contract import bind_partial_targets
from mtare_topo.representation.gse_structure_prediction_contract import PartialStructuralPrediction,partial_structure_loss

RUN=ROOT/'results/gate3_semantics/gate3_20260911_gse_common_structure_fit_v1_seed0'
OUT=ROOT/'docs/figures/gse_common_structure_fit_v1'

def main():
    lines=(RUN/'artifacts/evidence_sha256.txt').read_text().splitlines()
    for line in lines:
        h,p=line.split('  ',1)
        if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h:raise ValueError('run seal drift')
    card=json.loads((RUN/'config/run_spec.json').read_text())
    task=json.loads((ROOT/'configs/v3/gate3/gse_common_partial_structure_task_v1.json').read_text())
    bound=card['input_sha256']
    tp='configs/v3/gate3/gse_common_partial_structure_task_v1.json'
    if hashlib.sha256((ROOT/tp).read_bytes()).hexdigest()!=bound[tp]:raise ValueError('task drift')
    targets=[]
    for e in task['entries']:
        raw=(ROOT/e['reference_path']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=e['reference_sha256']:raise ValueError('reference drift')
        targets.append(bind_partial_targets(json.loads(raw)['record']))
    result=dict(seal_entries_verified=len(lines),model_forwards=0,training_steps=0,windows=[],stages={})
    for stage in ('initial','final'):
        rows=json.loads((RUN/f'artifacts/{stage}_predictions.json').read_text())
        all_dist={'structure':[],'section':[]};terms=[]
        for i,(row,target) in enumerate(zip(rows,targets)):
            p=PartialStructuralPrediction(**{k:torch.tensor(v,dtype=torch.float32) for k,v in row['predictions'].items()})
            loss=partial_structure_loss(p,target)
            terms.append({k:float(v) for k,v in loss['terms'].items()})
            if stage=='initial':result['windows'].append(dict(observation=i))
            record=dict(loss=terms[-1])
            for name,pos,truth in [('structure',p.structure_positions_m,target.anchor_positions_m),('section',p.window_section_positions_m,target.window_section_positions_m)]:
                d=np.linalg.norm(pos.numpy()[:,None]-truth[None],axis=-1)
                nearest=d.min(axis=0);assigned=d[loss['matches'][name],np.arange(len(truth))]
                all_dist[name].extend(assigned.tolist())
                record[name]=dict(nearest_m=nearest.tolist(),assigned_m=assigned.tolist(),matched_query=loss['matches'][name].tolist(),correct_1m=row[name]['correct'])
            result['windows'][i][stage]=record
        result['stages'][stage]=dict(mean_loss={k:float(np.mean([x[k] for x in terms])) for k in terms[0]},
            distances={k:dict(mean=float(np.mean(v)),median=float(np.median(v)),max=float(max(v))) for k,v in all_dist.items()})
    logs=[json.loads(x) for x in (RUN/'logs/updates.jsonl').read_text().splitlines()]
    result['loss_blocks']=[]
    for start in range(0,500,50):
        batch=logs[start:start+50];values=[t for row in batch for t in row['loss_terms']]
        result['loss_blocks'].append(dict(first_step=start+1,last_step=start+50,means={k:float(np.mean([v[k] for v in values])) for k in values[0]}))
    result['interpretation']='Loss scale alone is not gradient conflict evidence. Missing 1m references are not scored false positives. No threshold or label changes.'
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'saved_prediction_analysis.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(dict(stages=result['stages'],loss_blocks=result['loss_blocks']),indent=2))

def plot_saved():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font_manager.fontManager.addfont(font)
    plt.rcParams['font.family']=font_manager.FontProperties(fname=font).get_name()
    plt.rcParams['axes.unicode_minus']=False
    data=json.loads((OUT/'saved_prediction_analysis.json').read_text())
    fig,axes=plt.subplots(2,1,figsize=(12,7),layout='constrained')
    for ax,kind,title in zip(axes,('structure','section'),('结构位置','窗口截面位置（不等于物理开口检测已通过）')):
        a=[];b=[];labels=[]
        for row in data['windows']:
            a.extend(row['initial'][kind]['assigned_m']);b.extend(row['final'][kind]['assigned_m'])
            labels.extend(f"{row['observation']}:{j}" for j in range(len(row['initial'][kind]['assigned_m'])))
        ax.plot(a,'o-',label='初始一对一位置误差');ax.plot(b,'o-',label='最终一对一位置误差')
        ax.axhline(1,color='red',linestyle='--',label='固定1米评价线')
        ax.set(title=title,ylabel='米',xticks=range(len(labels)),xticklabels=labels)
        ax.legend(fontsize=8);ax.grid(alpha=.2)
    axes[-1].set_xlabel('观察编号:参考编号；全部24项，未挑选案例')
    fig.suptitle('500步训练结案：平均误差下降，不等于1米内恢复数量增加\n曲线为训练几何对应误差；正式计数使用独立最大恢复匹配，无阈值修改')
    fig.savefig(OUT/'localization_change.png',dpi=150);plt.close(fig)

if __name__=='__main__':
    import sys
    if '--plot-only' in sys.argv:plot_saved()
    else:main()
