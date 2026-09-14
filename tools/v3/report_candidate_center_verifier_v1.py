"""Read-only reduction and complete previews of the sealed single verifier run."""
from _bootstrap import PROJECT_ROOT as ROOT
import json
from collections import Counter
import numpy as np
from report_geometry_presence_a_v1 import verify, sha, load
from mtare_topo.governance_surface_selection import digest

RUN=ROOT/'results/gate3_semantics/gate3_20260910_gse_candidate_center_verifier_v1_seed0'
OUT=ROOT/'docs/figures/gse_graph/candidate_center_verifier_20260910'

def main():
    if OUT.exists(): raise FileExistsError('derivative evidence is non-overwriting')
    seal=verify(RUN)
    final=load(RUN/'metrics/evaluation_1000.json')
    labels=load(RUN/'metrics/validity_inventory.json')
    baseline=load(RUN/'metrics/old_l2_1e-06.json')
    reference=load(ROOT/'results/gate3_semantics/gate3_20260910_gse_fixed_candidate_scoring_v1_seed0/metrics/evaluation_0000.json')
    OUT.mkdir(parents=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.text import Text
    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    def save(fig,path):
        for t in fig.findobj(Text): t.set_fontproperties(FontProperties(fname=font,size=t.get_fontsize()))
        fig.tight_layout();fig.savefig(path,dpi=125);plt.close(fig)
    errors=[];cases=[];counts={r:Counter() for r in ('old','fixed')}
    for i,(o,v,b,ref) in enumerate(zip(final['observations'],labels['observations'],baseline['observations'],reference['observations']),1):
        assert o['source']==b['source']==ref['source']
        key=digest(o['source'])
        with np.load(RUN/'artifacts'/f'input_{key}.npz') as z:
            xyz=z['xyz_m'];p=z['position_m'];t=z['target_positions_m']
        categories={}
        for region,radius in [('old','1.0'),('fixed','4.0')]:
            s=o['scores'][region]['after']; slots=s['original_query_slots'];matched={slots[j] for j,_ in s['pairs']}
            cov=ref['scores'][radius]['coverage']['query_scoreable_mask']
            categories[region]={q:('tp' if q in matched else 'fp' if cov[q] else 'unknown') for q in slots}
            assert sum(c=='fp' for c in categories[region].values())==s['fp']
            for q,c in categories[region].items():
                if c!='fp':continue
                lab=v['candidates'][q];kind=lab['new_label']+('/old_duplicate' if lab['old_duplicate'] else '/other')
                counts[region][kind]+=1
                errors.append(dict(observation=i,region=region,source=o['source'],**lab,logit=o['logits'][q],nearest_reference_m=float(np.linalg.norm(t-p[q],axis=1).min()) if len(t) else None))
        fig,axes=plt.subplots(2,4,figsize=(17,9))
        for col,(name,e,stage) in enumerate([('旧L2=1e-6',b,'before'),('旧L2=1e-6',b,'after'),('新验证器',o,'before'),('新验证器',o,'after')]):
            s=e['scores']['old'][stage];slots=s['original_query_slots'];matched={slots[j] for j,_ in s['pairs']}
            cov=ref['scores']['1.0']['coverage']['query_scoreable_mask']
            for row,dims in enumerate([(0,1),(0,2)]):
                ax=axes[row,col];a,c=dims
                ax.scatter(xyz[:,a],xyz[:,c],s=.2,c='#aaaaaa',alpha=.25,rasterized=True)
                ax.scatter(p[:,a],p[:,c],s=8,c='#cccccc',label='全部冻结候选')
                for q in slots:
                    color='#16823b' if q in matched else '#d62728' if cov[q] else '#ee9b00'
                    ax.scatter(p[q,a],p[q,c],marker='x',s=60,c=color)
                    ax.annotate(str(q),(p[q,a],p[q,c]),fontsize=7)
                if len(t):ax.scatter(t[:,a],t[:,c],s=65,facecolors='none',edgecolors='black',marker='D',label='参考中心')
                ax.set(xlim=(-10.5,10.5),ylim=(-10.5,10.5),aspect='equal',xlabel='X / m',ylabel=('Y' if c==1 else 'Z')+' / m',title=f'{name} '+('去重前' if stage=='before' else '去重后')+f"\n正确{s['tp']} 误检{s['fp']} 漏检{s['fn']} 未知{s['ignored']}")
                ax.grid(alpha=.15)
        fig.suptitle(f"{i:02d} {o['source']['task']} / 序列{o['source']['source_sequence_id']}\n绿叉=匹配正确；红叉=旧评价误检；橙叉=旧评价未知；数字=原查询槽。不是新的地图节点。",fontsize=11)
        path=OUT/f'case_{i:02d}.png';save(fig,path)
        cases.append(dict(index=i,source=o['source'],labels=v['counts'],old=o['scores']['old']['after'],fixed=o['scores']['fixed']['after'],image=str(path.relative_to(ROOT))))
    curve=load(RUN/'metrics/curve.json');logs=[json.loads(l) for l in (RUN/'logs/updates.jsonl').read_text().splitlines()]
    fig,axes=plt.subplots(1,3,figsize=(15,4.5));steps=[c['step'] for c in curve]
    axes[0].plot([x['update'] for x in logs],[x['loss'] for x in logs]);axes[0].set_title('实际训练损失（不作为通过标准）')
    for region,label in [('old','旧评价'),('fixed','固定区域评价')]:
        axes[1].plot(steps,[c['summary'][region]['after']['fp'] for c in curve],'-o',label=label+'误检')
        axes[2].plot(steps,[c['summary'][region]['after']['precision'] for c in curve],'-o',label=label+'精确率')
    axes[1].plot(steps,[c['summary']['old']['after']['tp'] for c in curve],'--',label='正确检出 / 共12')
    axes[2].plot(steps,[c['summary']['old']['after']['recall'] for c in curve],'--',label='召回率');axes[2].axhline(.9,c='gray',ls=':')
    for ax in axes:ax.set_xlabel('实际更新数');ax.grid(alpha=.2)
    for ax in axes[1:]:ax.legend(prop=FontProperties(fname=font,size=8))
    save(fig,OUT/'learning_curve.png')
    result=dict(seal=seal,training_rerun=False,model_inference=False,counts={k:dict(v) for k,v in counts.items()},errors=errors,cases=cases,summary=load(RUN/'metrics/summary.json'))
    (OUT/'error_analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    manifest={str(p.relative_to(ROOT)):sha(p) for p in OUT.iterdir() if p.is_file()}
    (OUT/'sha256.json').write_text(json.dumps(dict(run_seal=seal,tool_sha256=sha(__file__ and ROOT/'tools/v3/report_candidate_center_verifier_v1.py'),files=manifest),indent=2)+'\n')
    print(json.dumps(dict(seal=seal,counts=result['counts'],cases=len(cases)),ensure_ascii=False))

if __name__=='__main__':main()
