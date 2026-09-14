"""Reproducible explanatory figures from sealed window proposals, no labels."""
import hashlib
import json
import numpy as np
from _bootstrap import PROJECT_ROOT
from mtare_topo.data.gse_surface_teacher_reader_v1 import SurfaceTeacherReader
from mtare_topo.data.cano_sensor_smoke import lidar_local_directions, world_directions
from mtare_topo.data.primitive_relation_dataset import _current_sensor_transform

RUN='results/gate3_semantics/gate3_20260907_gse_surface_window_openings_v1_seed20260906'
SEAL='6c0af39c3054b2ad597609cdbac7b119f773e2d026e12d4741d2c33c0df9b88a'
CASES=('S08_3d_loop_rich_C01__c1_mixed','S01_flat_tree_small_C01__c1_mixed')


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font=next(x.name for x in font_manager.fontManager.ttflist if 'Noto Sans CJK' in x.name)
    plt.rcParams['font.family']=font
    plt.rcParams['axes.unicode_minus']=False
    root=PROJECT_ROOT;run=root/RUN
    raw=(run/'artifacts/evidence_sha256.txt').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=SEAL:raise ValueError('seal drift')
    pins={p:h for h,p in (line.split('  ',1) for line in raw.decode().splitlines())}
    def load(relative):
        p=RUN+'/'+relative;b=(root/p).read_bytes()
        if hashlib.sha256(b).hexdigest()!=pins[p]:raise ValueError('sealed file drift')
        return json.loads(b)
    card=load('config/data_card.json')
    reader=SurfaceTeacherReader(root,card['scope'])
    out=root/'docs/figures/gse_window_opening_evidence_v1'
    if out.exists():raise FileExistsError('preserve report figures')
    figures=[]
    for task in CASES:
        row=next(r for r in card['scope']['selected'] if r['task']==task)
        result=load('artifacts/'+row['view_id']+'.json');bundle=reader.read_task(task)
        if result['source']!=bundle['source']:raise ValueError('source mismatch')
        sensor=bundle['sensor_teacher_only'];student=bundle['student']
        local=lidar_local_directions().reshape(-1,3).astype(np.float64)
        directions=np.concatenate([world_directions(local,float(y)) for y in sensor['yaw_deg']])
        directions/=np.linalg.norm(directions,axis=1,keepdims=True)
        origins=np.repeat(sensor['sensor_xyz_m'],11520,axis=0)
        world=origins+directions*student['ranges_m'].reshape(-1,1)
        origin=sensor['sensor_xyz_m'][-1];yaw=float(sensor['yaw_deg'][-1])
        points=_current_sensor_transform(world,origin,yaw)
        valid=student['valid_mask'].reshape(-1).astype(bool)
        visible=valid&(np.linalg.norm(points,axis=1)<=15.)
        fig,axes=plt.subplots(1,2,figsize=(13,6),constrained_layout=True)
        detail=[]
        for ax,dim in zip(axes,(1,2)):
            ax.scatter(points[visible,0],points[visible,dim],s=.4,c='#aeb5bd',label='五帧回波（15米球内）',rasterized=True)
            theta=np.linspace(0,2*np.pi,200)
            ax.plot(10*np.cos(theta),10*np.sin(theta),'k--',lw=.7,label='10米球截面参照')
            ax.plot(0,0,'k+',markersize=8)
        for index,proposal in enumerate(result['proposals']):
            c=np.array(proposal['reference_position_m'])
            current=_current_sensor_transform(c,origin,yaw)
            end=_current_sensor_transform(c+2*np.array(proposal['reference_direction']),origin,yaw)
            ids=np.array(proposal['surface_return_ray_indices'],dtype=int)
            color=('#008b8b','#d98200','#7851a9')[index%3]
            supported=proposal['proposal_supported_after_competition']
            for ax,dim in zip(axes,(1,2)):
                displayed=ids[visible[ids]]
                ax.scatter(points[displayed,0],points[displayed,dim],s=5,c=color,rasterized=True)
                ax.scatter(current[0],current[dim],s=90,facecolors='none',edgecolors=color,
                    marker='o' if supported else 's',label=f'候选{index+1}：'+('有支持' if supported else '未知'))
                ax.annotate('',xy=end[[0,dim]],xytext=current[[0,dim]],arrowprops=dict(color=color,arrowstyle='->'))
            detail.append(dict(primitive=proposal['primitive_id_teacher_only'],supported=supported,
                surface_return_count=len(ids),surface_within10m=int((np.linalg.norm(points[ids],axis=1)<10).sum()),
                surface_shown_within15m=int(visible[ids].sum()),
                exclusive_crossing_count=len(proposal['exclusive_outward_crossing_ray_indices']),
                reference_center_current_m=current.tolist()))
        for ax,dim in zip(axes,(1,2)):
            ax.set(xlim=(-16,16),ylim=(-16,16),aspect='equal',xlabel='X（米）',ylabel=('Y' if dim==1 else 'Z')+'（米）')
            ax.legend(fontsize=8,loc='upper left');ax.grid(alpha=.15)
        fig.suptitle(task+'\n彩色点：同源边界回波；箭头与空心标记：参考方向/位置，不是模型预测\n'
                     '灰点含10米外原始观测；图像展示不代表完整轮廓、可通行或人工标注',fontsize=11)
        figures.append((fig,dict(task=task,source=result['source'],valid_returns=int(valid.sum()),
                                shown_returns=int(visible.sum()),proposals=detail)))
    out.mkdir()
    records=[]
    for i,(fig,record) in enumerate(figures,1):
        path=out/f'case_{i}.png';fig.savefig(path,dpi=150);plt.close(fig)
        records.append(dict(**record,file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (out/'provenance.json').write_text(json.dumps(dict(source_run=RUN,seal_sha256=SEAL,
        selection='Previously diagnosed S08 junction and S01 unsupported case; explanatory, not performance selection',
        source_files_sha256=reader.opened,figures=records,labels=0),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(records,ensure_ascii=False))


if __name__=='__main__':main()
