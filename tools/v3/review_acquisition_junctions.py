"""Bound three saved-window review; no encoder, teacher or model execution."""
from _bootstrap import PROJECT_ROOT as ROOT
import json,hashlib,argparse
from pathlib import Path
INPUT='results/gate3_semantics/gate3_20260910_gse_local_pair_pilot_inputs_v1_seed20260906'
NOM='configs/v3/gate3/gse_acquisition_junction_nomination_v1.json'
CARD='configs/v3/gate3/data_cards/gse_acquisition_junction_review_v1.json'
OUT=ROOT/'docs/figures/gse_supervision_acquisition_pilot_v1'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def freeze():
    nomination=json.loads((ROOT/NOM).read_text())
    manifest=ROOT/INPUT/'artifacts/manifest.json'
    seals={p:h for h,p in (x.split('  ',1) for x in (ROOT/INPUT/'artifacts/evidence_sha256.txt').read_text().splitlines())}
    assert sha(manifest)==seals[str(manifest.relative_to(ROOT))]
    rows=json.loads(manifest.read_text());entries=[]
    for n in nomination['selected']:
        r=rows[n['index']];s=r['source']
        assert s['task']==n['source']['task'] and s['frame_rows']==n['source']['frame_rows'] and s['split']=='fit'
        p=INPUT+'/'+r['student_path'];assert r['student_sha256']==seals[p]
        entries.append(dict(source=s,student_path=p,student_sha256=seals[p],layout='saved_single_window',decoded_observations=1,
            target_path=n['reference_path'],target_sha256=n['reference_sha256'],frame_rows=s['frame_rows']))
    card=dict(schema='gse_saved_junction_review_scope_v1',operation='read_only_evidence_review',entries=entries,
        nomination_sha256=sha(ROOT/NOM),observations=3,parents=3,frames=15,raw_rays=172800,
        spacing='Original consecutive source frame rows; actual temporal/metric history spacing not independently confirmed',
        selection='Teacher-stratified fit-only positive nomination; not blind selection, not model score selection',
        authority='User continue to bind and inspect exactly these3 saved junction observations and witnesses',
        renderer='Existing register_causal_lidar_points; no encoder. Existing yaw contract not newly certified.',
        new_labels=0,training_steps=0,teacher_calls=0,full_training_authorized=False)
    with (ROOT/CARD).open('x') as f:json.dump(card,f,ensure_ascii=False,indent=2)
    print(json.dumps(dict(bound=len(entries),payload_reads=0)))

def render(witnesses=False):
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from mtare_topo.data.gse_membership_fit_reader import load_student_window,load_partial_reference
    from mtare_topo.representation.primitive_relation_model import register_causal_lidar_points,MAXIMUM_RANGE_M
    card=json.loads((ROOT/CARD).read_text())
    assert sha(ROOT/NOM)==card['nomination_sha256']
    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';font_manager.fontManager.addfont(font)
    plt.rcParams['font.family']=font_manager.FontProperties(fname=font).get_name();plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(3,2,figsize=(12,13),layout='constrained');counts=[]
    for i,e in enumerate(card['entries']):
        s=load_student_window(ROOT,e)
        rv=torch.stack((torch.tensor(s.ranges_m.copy())/MAXIMUM_RANGE_M,torch.tensor(s.valid_mask.copy(),dtype=torch.float32)),dim=1)[None]
        xyz,valid=register_causal_lidar_points(rv,torch.tensor(s.relative_translation_current_sensor_m.copy())[None],torch.tensor(s.relative_yaw_current_sensor_deg.copy())[None])
        xyz=xyz[0].reshape(-1,3).numpy();valid=valid[0].reshape(-1).numpy();slots=np.repeat(np.arange(5),11520)
        keep=valid & (np.linalg.norm(xyz,axis=1)<=10)
        target=load_partial_reference(ROOT,e);record=target['record'];prov=target['teacher_provenance'];n=prov['terminal_anchor_start']
        # Full raw points go through the same student registration before any
        # teacher payload is read. Reference locations only enter this overlay.
        anchors=np.array([a['position_m'] for a in record['anchors']]).reshape(-1,3)
        support=[np.asarray(x,dtype=int) for x in prov['anchors'][0]['witness_ray_indices']]
        if any(np.any((x<0)|(x>=len(xyz))) or not valid[x].all() for x in support):raise ValueError('witness not a valid saved ray')
        counts.append(dict(task=e['source']['task'],valid_returns=int(valid.sum()),local_returns=int(keep.sum()),junctions=n,terminals=len(anchors)-n,
            openings=len(record['openings']),score_region=record['score_region'],labels_generated=0))
        counts[-1]['branch_witnesses']=[dict(total=len(x),local_returns=int(keep[x].sum()),outside_returns=int((~keep[x]).sum())) for x in support]
        counts[-1]['pairwise_shared_ray_counts']=[len(set(support[a])&set(support[b])) for a,b in [(0,1),(0,2),(1,2)]]
        for ax,dims,title in zip(axes[i],[(0,1),(0,2)],['XY','XZ']):
            ax.scatter(xyz[keep,dims[0]],xyz[keep,dims[1]],s=.3,c=slots[keep],cmap='viridis',vmin=0,vmax=4)
            if witnesses:
                ax.clear()
                ax.scatter(xyz[keep,dims[0]],xyz[keep,dims[1]],s=.3,c='.8')
                for b,x in enumerate(support):
                    x=x[keep[x]]
                    ax.scatter(xyz[x,dims[0]],xyz[x,dims[1]],s=1.2,c=['blue','green','purple'][b],alpha=.55,label=f'支路{b+1}见证回波')
            if n:ax.scatter(anchors[:n,dims[0]],anchors[:n,dims[1]],facecolors='none',edgecolors='red',s=100,label='教师路口参考（非模型预测）')
            if len(anchors)>n:ax.scatter(anchors[n:,dims[0]],anchors[n:,dims[1]],marker='s',facecolors='none',edgecolors='orange',s=80,label='教师尽头参考')
            ax.set(title=e['source']['task']+'\n'+title,xlim=(-10,10),ylim=(-10,10),aspect='equal',xlabel='米',ylabel='米');ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('教师三支路见证对应的局部真实回波：颜色是见证分组，不是学习分割\n窗口外见证未显示；同射线可有多组身份，绘制覆盖不代表唯一归属' if witnesses else '三个不同父地图的路口复核案例：点云与隔离教师参考叠加\n颜色是五帧来源；圆/方框不是模型预测；未知区域未标为负例')
    fig.savefig(OUT/('junction_witness_overlay.png' if witnesses else 'junction_cases_reference_overlay.png'),dpi=160);plt.close(fig)
    with (OUT/('junction_witness_counts.json' if witnesses else 'junction_review_counts.json')).open('x') as f:json.dump(counts,f,ensure_ascii=False,indent=2)
    print(json.dumps(counts,ensure_ascii=False))

def render_observation_context():
    """Full saved returns, no reference payload; not a blind-review claim."""
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from mtare_topo.data.gse_membership_fit_reader import load_student_window
    from mtare_topo.representation.primitive_relation_model import register_causal_lidar_points,MAXIMUM_RANGE_M
    card=json.loads((ROOT/CARD).read_text())
    assert sha(ROOT/NOM)==card['nomination_sha256']
    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';font_manager.fontManager.addfont(font)
    plt.rcParams['font.family']=font_manager.FontProperties(fname=font).get_name()
    plt.rcParams['axes.unicode_minus']=False
    destination=OUT/'junction_observation_context.png'
    if destination.exists():raise FileExistsError(destination)
    fig,axes=plt.subplots(3,3,figsize=(17,15),layout='constrained')
    for i,e in enumerate(card['entries']):
        s=load_student_window(ROOT,e)
        rv=torch.stack((torch.tensor(s.ranges_m.copy())/MAXIMUM_RANGE_M,torch.tensor(s.valid_mask.copy(),dtype=torch.float32)),dim=1)[None]
        xyz,valid=register_causal_lidar_points(rv,torch.tensor(s.relative_translation_current_sensor_m.copy())[None],torch.tensor(s.relative_yaw_current_sensor_deg.copy())[None])
        xyz=xyz[0].reshape(-1,3).numpy();valid=valid[0].reshape(-1).numpy();slots=np.repeat(np.arange(5),11520)
        keep=valid & (np.linalg.norm(xyz,axis=1)<=10)
        for col,(dims,mask,title) in enumerate([((0,1),valid,'完整返回 XY'),((0,1),keep,'10米局部 XY'),((0,2),valid,'完整返回 XZ')]):
            ax=axes[i,col];ax.scatter(xyz[mask,dims[0]],xyz[mask,dims[1]],s=.2,c=slots[mask],cmap='viridis',vmin=0,vmax=4)
            ax.scatter([0],[0],c='black',marker='+',s=35,label='当前传感器原点')
            ax.set(title=e['source']['task']+'\n'+title,aspect='equal',xlabel='米',ylabel='米');ax.grid(alpha=.2)
            if col==1:ax.set(xlim=(-10,10),ylim=(-10,10))
    fig.suptitle('同三案例的五帧观测上下文：颜色仅表示帧序；无教师中心、来源分组或模型输出\n仅展示范围变化，模型及评分10米域不变；已看过参考，不能宣称盲审')
    fig.savefig(destination,dpi=130);plt.close(fig)
    print(destination)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--render',action='store_true');p.add_argument('--witnesses',action='store_true');p.add_argument('--observation-context',action='store_true');a=p.parse_args()
    if a.freeze:freeze()
    elif a.observation_context:render_observation_context()
    elif a.render:render(a.witnesses)
