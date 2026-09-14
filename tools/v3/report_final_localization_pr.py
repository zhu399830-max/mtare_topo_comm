"""Final saved scores only: exhaustive shared-threshold PR, no new run/model."""
from _bootstrap import PROJECT_ROOT as ROOT
import csv
import gzip
import hashlib
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from mtare_topo.governance_surface_selection import digest
from mtare_topo.evaluation.candidate_center_selection_v1 import select_candidates
from mtare_topo.evaluation.grouping_center_scoring_v1 import center_score
from fixed_score_numerics_v1 import aggregate

RUN=ROOT/'results/gate3_semantics/gate3_20260910_gse_localization_quality_fit_v1_seed0'
PARENT=ROOT/'results/gate3_semantics/gate3_20260910_gse_candidate_center_verifier_v1_seed0'
REF=ROOT/'results/gate3_semantics/gate3_20260910_gse_fixed_candidate_scoring_v1_seed0'
OUT=ROOT/'docs/figures/gse_graph/localization_quality_20260910'

def seal_cached_report():
    """Finish provenance without rerunning a completed threshold scan."""
    paths=[RUN/'metrics/evaluation_1000.json',REF/'metrics/evaluation_0000.json']
    for o in json.loads(paths[0].read_text())['observations']:
        key=digest(o['source']);paths.extend([RUN/'artifacts'/f'prediction_1000_{key}.npz',PARENT/'artifacts'/f'input_{key}.npz'])
    reads={}
    for p in paths:
        run=next(r for r in (RUN,PARENT,REF) if p.is_relative_to(r))
        index={n:h for h,n in (l.split('  ',1) for l in (run/'artifacts/evidence_sha256.txt').read_text().splitlines())}
        key=str(p.relative_to(ROOT));h=hashlib.sha256(p.read_bytes()).hexdigest();assert index[key]==h;reads[key]=h
    manifest=dict(reads=reads,tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),outputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('pr_final_*') if p.is_file() and p.name!='pr_final_sha256.json'})
    with (OUT/'pr_final_sha256.json').open('x') as f:json.dump(manifest,f,indent=2)

def selection(xyz,probability,threshold):
    before=np.flatnonzero(probability>=threshold).tolist()
    order=sorted(before,key=lambda i:(-float(probability[i]),*xyz[i].tolist(),i))
    kept=[];suppressed=[]
    for i in order:
        blockers=[j for j in kept if np.linalg.norm(xyz[i]-xyz[j])<=2.]
        if blockers:suppressed.append(dict(candidate=i,suppressor=blockers[0],distance_m=float(np.linalg.norm(xyz[i]-xyz[blockers[0]]))))
        else:kept.append(i)
    return dict(before=before,after=kept,suppressed=suppressed)

def main():
    if (OUT/'pr_final_summary.json').exists():raise FileExistsError('preserve derivative result')
    reads={};indices={}
    def read(path,kind='json'):
        raw=path.read_bytes();h=hashlib.sha256(raw).hexdigest();run=next(r for r in (RUN,PARENT,REF) if path.is_relative_to(r))
        if run not in indices:indices[run]={p:x for x,p in (l.split('  ',1) for l in (run/'artifacts/evidence_sha256.txt').read_text().splitlines())}
        assert indices[run][str(path.relative_to(ROOT))]==h;reads[str(path.relative_to(ROOT))]=h
        if kind=='json':return json.loads(raw)
        import io
        with np.load(io.BytesIO(raw)) as z:return {k:z[k].copy() for k in z.files}
    final=read(RUN/'metrics/evaluation_1000.json');reference=read(REF/'metrics/evaluation_0000.json');rows=[]
    for e,r in zip(final['observations'],reference['observations']):
        assert e['source']==r['source'];key=digest(e['source'])
        saved=read(RUN/'artifacts'/f'prediction_1000_{key}.npz','npz');data=read(PARENT/'artifacts'/f'input_{key}.npz','npz')
        assert np.array_equal(saved['position_m'],data['position_m']) and np.array_equal(saved['logits'],np.asarray(e['logits'],dtype=saved['logits'].dtype))
        xyz=saved['position_m'].astype(float);prob=torch.as_tensor(saved['logits'].copy()).sigmoid().numpy()
        sel=selection(xyz,prob,.5);oldsel=select_candidates(xyz,saved['logits'])
        assert all(sel[k]==oldsel[k]==e['selection'][k] for k in ('before','after','suppressed'))
        rows.append(dict(source=e['source'],xyz=xyz,prob=prob,target=data['target_positions_m'],reference=r))
    assert len(rows)==16 and sum(len(r['prob']) for r in rows)==512
    probabilities=np.unique(np.concatenate([r['prob'] for r in rows]).astype(float))
    thresholds=sorted(set([0.,.5,1.]+probabilities.tolist()));curve=[];details=[]
    for threshold in thresholds:
        records=[]
        for r in rows:
            sel=selection(r['xyz'],r['prob'],threshold);scores={}
            for region,key in [('old','1.0'),('fixed','4.0')]:
                cov=r['reference']['scores'][key]['coverage'];scoreable=np.asarray(cov['query_scoreable_mask'],bool);allowed=~np.asarray(cov['possible_unconfirmed_reference_mask'],bool)
                scores[region]={}
                for stage in ('before','after'):
                    chosen=np.asarray(sel[stage],int)
                    s=center_score(r['xyz'][chosen],r['target'],scoreable[chosen],allowed[chosen],radius=1.)
                    s['original_query_slots']=chosen.tolist();scores[region][stage]=s
            records.append(dict(source=r['source'],selection=sel,scores=scores))
        summary={region:{stage:aggregate([r['scores'][region][stage] for r in records]) for stage in ('before','after')} for region in ('old','fixed')}
        if threshold==.5:assert summary==final['summary']
        curve.append(dict(threshold=threshold,summary=summary));details.append(dict(threshold=threshold,observations=records))
    minima={}
    for region in ('old','fixed'):
        eligible=[r for r in curve if r['summary'][region]['after']['tp']>=11]
        fp=min(r['summary'][region]['after']['fp'] for r in eligible)
        best=[r for r in eligible if r['summary'][region]['after']['fp']==fp]
        representative=max(best,key=lambda r:(r['summary'][region]['after']['tp'],r['threshold']))
        minima[region]=dict(min_fp=fp,representative=representative,all_minimum_representatives=[r['threshold'] for r in best],tie_rule='more TP then highest threshold; descriptive only')
    passing=[r for r in curve if all(r['summary'][region]['after']['precision']>=.9 and r['summary'][region]['after']['recall']>=.9 for region in ('old','fixed'))]
    result=dict(candidates=512,observations=16,checkpoint_step=1000,distinct_probabilities=len(probabilities),threshold_count=len(thresholds),
        exhaustive_rule='All distinct original float32 sigmoid probabilities plus0,.5,1; inclusive>=; between consecutive scores selection is constant; one shared threshold across16',
        baseline=next(r for r in curve if r['threshold']==.5),min_fp_at_least11=minima,unified_90_90_exists=bool(passing),passing=passing,
        historical_status='GATE_FAIL',training_updates=0,new_weights=False,gt_inference_filter=False,independent_validation=False)
    with (OUT/'pr_final_all_thresholds.csv').open('x') as f:
        writer=csv.DictWriter(f,fieldnames=['threshold','region','selection','tp','fp','fn','ignored','output_count','precision','recall','f1']);writer.writeheader()
        for r in curve:
            for region,stages in r['summary'].items():
                for stage,v in stages.items():writer.writerow(dict(threshold=r['threshold'],region=region,selection=stage,**v))
    with gzip.open(OUT/'pr_final_matching_records.json.gz','xt') as f:json.dump(details,f,ensure_ascii=False)
    (OUT/'pr_final_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.text import Text
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for region,label in [('old','旧评价'),('fixed','固定区域评价')]:
        valid=[r for r in curve if r['summary'][region]['after']['tp']+r['summary'][region]['after']['fp']>0]
        axes[0].plot([r['summary'][region]['after']['recall'] for r in valid],[r['summary'][region]['after']['precision'] for r in valid],'.-',label=label)
        axes[1].plot([r['summary'][region]['after']['tp'] for r in curve],[r['summary'][region]['after']['fp'] for r in curve],'.',label=label)
    axes[0].set(xlabel='召回率',ylabel='精确率',title='最终检查点：共用阈值，原2米去重');axes[0].axhline(.9,c='gray',ls=':');axes[0].axvline(.9,c='gray',ls=':')
    axes[1].set(xlabel='正确检出数量 / 共12',ylabel='已知误检数量',title='不能用大量漏检换取通过');axes[1].axvline(11,c='gray',ls=':')
    for ax in axes:ax.grid(alpha=.2);ax.legend(prop=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'))
    for t in fig.findobj(Text):t.set_fontproperties(FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',size=t.get_fontsize()))
    fig.tight_layout();fig.savefig(OUT/'pr_final_curve.png',dpi=140);plt.close(fig)
    seal_cached_report()
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
