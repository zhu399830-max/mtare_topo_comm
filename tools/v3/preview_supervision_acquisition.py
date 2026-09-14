"""Reuse sealed blind review files; no teacher, model, or annotation export."""
from _bootstrap import PROJECT_ROOT as ROOT
import hashlib,json
import numpy as np
RUN=ROOT/'results/gate3_semantics/gate3_20260907_gse_surface_review_export_v1_seed20260906'
OUT=ROOT/'docs/figures/gse_supervision_acquisition_pilot_v1'
CARD=ROOT/'configs/v3/gate3/data_cards/gse_supervision_acquisition_preview_v1.json'

def bound_metadata():
    seal=(RUN/'artifacts/evidence_sha256.txt').read_bytes()
    if hashlib.sha256(seal).hexdigest()!='8cd3037b74c5c0fac6b4b8e92dadda1121660ab3c679c249035c84c0e4d26cb5':raise ValueError('seal drift')
    hashes={p:h for h,p in (line.split('  ',1) for line in seal.decode().splitlines())}
    def read(rel):
        p=RUN/rel;raw=p.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=hashes[str(p.relative_to(ROOT))]:raise ValueError('metadata drift')
        return json.loads(raw)
    manifest=read('artifacts/review_manifest.json')
    original=read('config/data_card.json')
    selected=original['scope']['selected'][:3]
    entries=[]
    for e,m in zip(selected,manifest['observations'][:3]):
        assert e['view_id']==m['view_id']
        p=str((RUN/m['blind']).relative_to(ROOT))
        entries.append(dict(view_id=e['view_id'],source=e['source'],path=p,sha256=hashes[p],valid_points=m['valid_points']))
    return dict(schema='gse_saved_blind_preview_scope_v1',operation='read_only_review_preview',entries=entries,
        parents=1,physical_edge_units=1,observations=3,selected_frames=15,
        spacing='Original rows1333..1337; temporal/history metric spacing unknown; no invented values',
        purpose='Inspect saved observed geometry before any annotation or reference reveal',
        reference_payload_reads=0,new_labels=0,training_steps=0,
        authorization='User continue after explicit proposal to acquire/validate supervision; preview only, no training or annotation authority inferred',
        training_authorized=False,annotation_authorized=False)

def main():
    scope=bound_metadata()
    if CARD.exists():
        if json.loads(CARD.read_text())!=scope:raise ValueError('preview scope drift')
    else:
        with CARD.open('x') as f:json.dump(scope,f,ensure_ascii=False,indent=2)
    # Scope is bound before any blind point payload is opened.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';font_manager.fontManager.addfont(font)
    plt.rcParams['font.family']=font_manager.FontProperties(fname=font).get_name()
    plt.rcParams['axes.unicode_minus']=False
    import sys
    detail='--detail' in sys.argv
    fig=plt.figure(figsize=(15,12),layout='constrained') if detail else plt.figure(figsize=(12,14),layout='constrained')
    axes=None if detail else fig.subplots(3,2)
    for i,e in enumerate(scope['entries']):
        raw=(ROOT/e['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=e['sha256']:raise ValueError('blind point drift')
        b=json.loads(raw)
        assert set(b)=={'schema','observation_id','coordinate_frame','source_frame_indices','points_xyz_m','point_history_slots'}
        xyz=np.asarray(b['points_xyz_m']);slots=np.asarray(b['point_history_slots'])
        assert len(xyz)==e['valid_points'] and b['source_frame_indices']==e['source']['frame_rows']
        if detail:
            roi=np.linalg.norm(xyz,axis=1)<=10
            for j in range(4):
                ax=fig.add_subplot(3,4,i*4+j+1,projection='3d' if j==3 else None)
                keep=roi & ((slots==0) if j==0 else (slots==4) if j==1 else np.ones(len(slots),bool))
                points=xyz[keep]
                if j==3:
                    ax.scatter(*points.T,s=.2,c=slots[keep],cmap='viridis',vmin=0,vmax=4)
                    ax.view_init(elev=25,azim=125);ax.set_zlim(-10,10)
                else:
                    ax.scatter(points[:,0],points[:,1],s=.35,c=slots[keep],cmap='viridis',vmin=0,vmax=4)
                    ax.set_aspect('equal');ax.grid(alpha=.2)
                ax.set_xlim(-10,10);ax.set_ylim(-10,10)
                ax.set_title(e['source']['variant']+'\n'+['第1帧 XY','第5帧 XY','五帧 XY','五帧三维'][j],fontsize=9)
            continue
        for ax,dims,title in zip(axes[i],[(0,1),(0,2)],['XY俯视','XZ侧视']):
            ax.scatter(xyz[:,dims[0]],xyz[:,dims[1]],s=.25,c=slots,cmap='viridis',vmin=0,vmax=4,rasterized=True)
            ax.scatter([0],[0],marker='+',c='red',s=45)
            ax.set(title=e['source']['variant']+' / '+title,xlabel='米',ylabel='米',aspect='equal')
            ax.grid(alpha=.2)
    fig.suptitle('前3个封存盲观测：10米显示域、首末帧与三维；只显示返回点，不代表空白区域已观察' if detail else '封存清单前3个观测：仅点云，颜色代表五帧来源；红色加号是当前传感器原点\n未加载构造标签、未运行模型、未确认结构；同一地点三种实现，不是三个独立样本')
    OUT.mkdir(parents=True,exist_ok=True)
    fig.savefig(OUT/('blind_first3_detail.png' if detail else 'blind_first3.png'),dpi=150);plt.close(fig)
    print(json.dumps(dict(observations=3,parents=1,points=sum(e['valid_points'] for e in scope['entries']),labels=0,reference_reads=0)))

if __name__=='__main__':main()
