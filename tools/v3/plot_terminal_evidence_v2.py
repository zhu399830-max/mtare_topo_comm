#!/usr/bin/env python3
"""Derived report figures from frozen diagnostic sources; no new labels.

Explicit two-case explanatory selection, not an evaluation population.
Never modify the sealed run. Refuse to overwrite existing report assets.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.gse_surface_population_teacher_reader_v1 import PopulationTeacherReader
from mtare_topo.data.primitive_relation_materialization import load_p1a_realized_construction
from mtare_topo.data.primitive_relation_dataset import _current_sensor_transform
from mtare_topo.data.cano_sensor_smoke import lidar_local_directions, world_directions
from mtare_topo.teacher.csg_mesh_provenance import mesh_swept_superellipse
from mtare_topo.teacher.gse_cap_return_evidence_v1 import source_endpoint_cap_faces

RUN='results/gate3_semantics/gate3_20260907_gse_population_terminal_v2_seed20260906'
SEAL='837ead2dee3a375ff6aeae1f72817d566e6858d5160ba48db2ff7c5ceaf2097f'
CASES=[('S08_3d_loop_rich_C07__c1_mixed',133465,'node_0006','支持例：实际回波命中参考端面'),
       ('S01_flat_tree_small_C06__ellipse',6074,'node_0008','反例：参考端面不是当前可见终止面')]


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font=next((x.name for x in font_manager.fontManager.ttflist if 'Noto Sans CJK' in x.name),None)
    if font:plt.rcParams['font.family']=font
    plt.rcParams['axes.unicode_minus']=False
    root=PROJECT_ROOT;run=root/RUN
    raw=(run/'artifacts/evidence_sha256.txt').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=SEAL:raise ValueError('result seal drift')
    pins=dict((p,h) for h,p in (line.split('  ',1) for line in raw.decode().splitlines()))
    def result_json(relative):
        p=RUN+'/'+relative;b=(root/p).read_bytes()
        if hashlib.sha256(b).hexdigest()!=pins[p]:raise ValueError('result file drift')
        return json.loads(b)
    card=result_json('config/data_card.json')
    out=root/'docs/figures/gse_terminal_evidence_v2'
    if out.exists():raise FileExistsError('report figures are not overwritten')
    reader=PopulationTeacherReader(root,card['scope']);figures=[]
    # Prepare everything before creating any report files.
    for task,sequence,node,title in CASES:
        bundles=reader.read_task(task)
        bundle=next(b for b in bundles if b['source']['source_sequence_id']==sequence)
        result=next(o for o in result_json('artifacts/'+task+'.json')['observations']
                    if o['source']['source_sequence_id']==sequence)
        reference=next(r for r in result['terminal_references'] if r['node_id_teacher_only']==node)
        if result['source']!=bundle['source']:raise ValueError('figure source mismatch')
        sensor=bundle['sensor_teacher_only'];student=bundle['student']
        local=lidar_local_directions().reshape(-1,3).astype(float)
        directions=np.concatenate([world_directions(local,float(y)) for y in sensor['yaw_deg']])
        directions/=np.linalg.norm(directions,axis=1,keepdims=True)
        origins=np.repeat(sensor['sensor_xyz_m'],11520,axis=0)
        world=origins+student['ranges_m'].reshape(-1,1)*directions
        origin=sensor['sensor_xyz_m'][-1];yaw=float(sensor['yaw_deg'][-1])
        points=_current_sensor_transform(world,origin,yaw)
        valid=student['valid_mask'].reshape(-1).astype(bool)&(np.linalg.norm(points,axis=1)<10.)
        ids=np.array(sorted({w['ray_index'] for w in reference['accepted_witnesses']}),dtype=int)
        supported=np.zeros(len(points),bool);supported[ids]=True
        _,primitives=load_p1a_realized_construction(bundle['construction_teacher_only'])
        pid,side=reference['endpoint_key_teacher_only']
        primitive=next(p for p in primitives if p.primitive_id==pid)
        mesh=mesh_swept_superellipse(primitive,axial_spacing_m=.05,angular_segments=64)
        cap=source_endpoint_cap_faces(mesh,angular_segments=64,endpoint_index=side)
        cap_points=_current_sensor_transform(mesh.vertices_xyz_m.astype(np.float32).astype(float),origin,yaw)
        fig,axes=plt.subplots(1,2,figsize=(11,5),constrained_layout=True)
        for ax,dim,name in zip(axes,[1,2],['XY俯视','XZ侧视']):
            ax.scatter(points[valid,0],points[valid,dim],s=.35,c='#9ca3af',rasterized=True,label='五帧实际回波')
            keep=valid&supported
            if keep.any():ax.scatter(points[keep,0],points[keep,dim],s=.5,c='#059669',rasterized=True,label='精确端面见证')
            for j,face in enumerate(mesh.triangle_vertex_indices[cap]):
                p=cap_points[np.r_[face,face[0]]]
                ax.plot(p[:,0],p[:,dim],color='#d97706',lw=.35,alpha=.7,label='原构造端面' if j==0 else None)
            ax.scatter([0],[0],marker='^',c='#2563eb',s=50,label='当前传感器')
            ax.set(xlim=(-10,10),ylim=(-10,10),xlabel='X（米）',ylabel=('Y' if dim==1 else 'Z')+'（米）',title=name,aspect='equal')
            ax.grid(alpha=.15);ax.legend(fontsize=7,loc='upper left')
        fig.suptitle(title+'\n同一当前传感器坐标系；不是模型预测或安全证明',fontsize=12)
        figures.append((fig,dict(task=task,sequence=sequence,node_teacher_only=node,
            selected_frame_rows=bundle['source']['frame_rows'],all_valid_returns=int(student['valid_mask'].sum()),
            displayed_returns=int(valid.sum()),all_exact_witnesses=len(ids),support=reference['terminal_reference_observed'])))
    for path,h in reader.opened.items():
        if hashlib.sha256((root/path).read_bytes()).hexdigest()!=h:raise ValueError('source drift during plotting')
    out.mkdir(parents=True)
    for i,(fig,record) in enumerate(figures):
        path=out/f'case_{i+1}.png';fig.savefig(path,dpi=180);plt.close(fig)
        record.update(figure=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (out/'provenance.json').write_text(json.dumps(dict(source_run=RUN,seal_sha256=SEAL,
        selection='Explicit supported/counterexample explanation, not representative evaluation sampling',
        cases=[r for _,r in figures],source_reads_sha256=reader.opened,labels_changed=0),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(output=str(out),cases=[r for _,r in figures]),ensure_ascii=False))


if __name__=='__main__':main()
