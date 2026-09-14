"""Read sealed predictions only; publish derived figures without changing runs."""
import _bootstrap
import hashlib, io, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from mtare_topo.data.gse_synthetic_fit_scope import declared_cases
from mtare_topo.evaluation.gse_synthetic_field_scoring import expected_geometry

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'results/gate3_semantics/gate3_20260908_gse_block_representation_fit_v1_seed0'
PART=ROOT/'results/gate3_semantics/gate3_20260908_gse_spg_paired_extraction_v1_seed0'
OUT=ROOT/'docs/figures/gse_block_fit_diagnosis_20260908'
NAMES={'straight':'直道','terminal':'尽头','visible_blocker':'可见阻挡','T':'T形路口','Y':'Y形路口','four_way':'四叉路口'}
METHODS={'r0':'扫描分组','r1':'体素分块','r2':'几何超点'}

def main():
    opened={}
    def reader(run, seal_sha):
        seal=(run/'artifacts/evidence_sha256.txt').read_bytes()
        assert hashlib.sha256(seal).hexdigest()==seal_sha
        pins={p:h for h,p in (line.split('  ',1) for line in seal.decode().splitlines())}
        def read(rel):
            p=run/rel; key=str(p.relative_to(ROOT)); b=p.read_bytes()
            assert hashlib.sha256(b).hexdigest()==pins[key]
            opened[key]=pins[key]; return b
        return read
    read=reader(RUN,'9993ec4cbbc08521a099ceba8b81fdfafeef7b91477b2f5fd9a0d3bb383bdfdf')
    part=reader(PART,'66e056fa4f235694d9fb7b39f155a6a8696324514febf47ca263acf584a0cadd')
    spec=json.loads(read('config/run_spec.json'))
    for rel in ('src/mtare_topo/data/gse_synthetic_fit_scope.py','src/mtare_topo/evaluation/gse_synthetic_field_scoring.py'):
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==spec['source_sha256'][rel]
        opened[rel]=spec['source_sha256'][rel]
    cases={c['case_id']:c for c in declared_cases()}
    predictions={m:{r['observation_id']:r for r in json.loads(read('artifacts/'+m+'_final_predictions.json'))} for m in METHODS}
    assert all(set(rows)==set(cases) for rows in predictions.values())
    totals={}; details=[]
    for m,rows in predictions.items():
        groups={}
        for cid,row in rows.items():
            ref=expected_geometry(cases[cid]); p=row['prediction']
            a=np.asarray(p['anchor_all_positions_m']); scores=np.asarray(p['anchor_all_scores'])
            truth=np.asarray(ref['anchors']).reshape(-1,3)
            within=(np.linalg.norm(truth[:,None]-a[None],axis=2)<=1) if len(truth) else np.empty((0,len(a)),bool)
            covered=within.any(axis=1); confident=(within&(scores>=.5)[None]).any(axis=1)
            kind=cid.split('__')[0]
            g=groups.setdefault(kind,dict(observations=0,targets=0,no_candidate_within_1m=0,located_but_low_confidence=0,background_false_observations=0,selected_anchors=0,tp=0,fp=0,fn=0))
            g['observations']+=1;g['targets']+=len(truth)
            g['no_candidate_within_1m']+=int((~covered).sum())
            g['located_but_low_confidence']+=int((covered&~confident).sum())
            g['background_false_observations']+=int(not len(truth) and bool(p['anchor_selected_indices']))
            g['selected_anchors']+=len(p['anchor_selected_indices'])
            for key in ('tp','fp','fn'):g[key]+=row['score']['anchors']['1.0'][key]
            details.append(dict(method=m,case_id=cid,nearest_any_anchor_m=(np.linalg.norm(truth[:,None]-a[None],axis=2).min(axis=1).tolist() if len(truth) else []),selected_indices=p['anchor_selected_indices'],anchor_score=row['score']['anchors']['1.0']))
        totals[m]=groups
    if OUT.exists():raise FileExistsError('no figure overwrite')
    OUT.mkdir(parents=True)
    font=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(font)).get_name(),'axes.unicode_minus':False})
    for kind,title in NAMES.items():
        cid=kind+'__circle__view2';ref=expected_geometry(cases[cid])
        with np.load(io.BytesIO(part('artifacts/partitions/'+cid+'.npz')),allow_pickle=False) as data:xyz=data['points_xyz_m'][::16]
        fig,axes=plt.subplots(2,3,figsize=(15,9),constrained_layout=True)
        for col,(m,name) in enumerate(METHODS.items()):
            p=predictions[m][cid]['prediction']
            for row,dims in enumerate(((0,1),(0,2))):
                ax=axes[row,col];ax.scatter(xyz[:,dims[0]],xyz[:,dims[1]],s=1,c='.7',alpha=.3,label='观测点（仅显示抽稀）')
                sets=[(ref['anchors'],'真实结构位置','green','o'),(p['anchor_positions_m'],'预测结构位置','red','x'),([o['position_m'] for o in ref['openings']],'真实窗口开口','blue','s'),(p['opening_positions_m'],'预测窗口开口','orange','+')]
                for values,label,color,marker in sets:
                    v=np.asarray(values).reshape(-1,3);ax.scatter(v[:,dims[0]],v[:,dims[1]],s=65,c=color,marker=marker,label=label)
                ax.set(xlim=(-11,11),ylim=(-11,11),xlabel='X / 米',ylabel=('Y' if row==0 else 'Z')+' / 米',title=name+(' · 俯视' if row==0 else ' · 侧视'))
                ax.set_aspect('equal');ax.grid(alpha=.2)
        handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=5)
        fig.suptitle(title+'｜固定圆形截面、中央视点；同输入三种分组最终预测\n合成训练集拟合，不是未见测试或拓扑图；置信度≥0.5，无去重')
        fig.savefig(OUT/(kind+'.png'),dpi=130);plt.close(fig)
    (OUT/'diagnosis.json').write_text(json.dumps(dict(groups=totals,per_case=details),ensure_ascii=False,indent=2))
    body='<html lang="zh"><meta charset="utf-8"><title>三种表示：真实预测对比</title><body><h1>几何分组是否帮助找准结构？</h1><p>同45个合成训练观察。以下固定展示六类场景的圆形截面、中央视点，不挑最佳结果。绿色为真实结构位置，红色为预测；蓝色为真实窗口开口，橙色为预测。窗口开口不是拓扑节点。本轮三组均未通过拟合。</p>'
    for kind,title in NAMES.items():body+='<h2>'+title+'</h2><img style="max-width:100%" src="'+kind+'.png">'
    (OUT/'index.html').write_text(body+'</body></html>')
    (OUT/'provenance.json').write_text(json.dumps(dict(input_sha256=opened,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),font_sha256=hashlib.sha256(font.read_bytes()).hexdigest(),selection='all six types circle view2; display stride16 only',new_optimizer_steps=0,scoring_unchanged=True),indent=2))
    (OUT/'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n' for p in sorted(OUT.iterdir()) if p.is_file()))
    print(json.dumps(dict(output=str(OUT),groups=totals),ensure_ascii=False))

if __name__=='__main__':main()
