"""Fixed first window per split; teacher evidence, explicitly not predictions."""
from _bootstrap import PROJECT_ROOT as ROOT
import gzip
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from local_pair_support_pilot import RUN, INPUT
from summarize_local_pair_support import summarize
from check_local_pair_window_coverage import sha, write
from mtare_topo.data.cano_sensor_smoke import lidar_local_directions


def main():
    report=summarize()  # Require terminal state and verify every sealed output.
    run=ROOT/RUN
    rows=json.loads((run/'artifacts/target_manifest.json').read_text())
    inputs=json.loads((ROOT/INPUT/'artifacts/manifest.json').read_text())
    index={(r['source']['task'],r['source']['source_global_sequence_index']):r for r in inputs}
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    fig,axes=plt.subplots(3,2,figsize=(11,15),constrained_layout=True)
    directions=lidar_local_directions().reshape(16,720,3).astype(np.float64)
    provenance=[]
    for i,split in enumerate(('fit','calibration','development')):
        row=next(r for r in rows if r['split']==split)
        s=row['source'];item=index[s['task'],s['source_global_sequence_index']]
        path=ROOT/INPUT/item['student_path']
        if sha(path)!=item['student_sha256']:raise ValueError('student source drift')
        with np.load(path,allow_pickle=False) as a:
            points=[]
            for j in range(5):
                xyz=directions*a['ranges_m'][j,...,None]
                yaw=np.deg2rad(float(a['relative_yaw_current_sensor_deg'][j]));c=np.cos(yaw);q=np.sin(yaw)
                rotation=np.array([[c,-q,0],[q,c,0],[0,0,1]])
                xyz=xyz@rotation.T+a['relative_translation_current_sensor_m'][j]
                valid=a['valid_mask'][j].astype(bool)&(np.linalg.norm(xyz,axis=-1)<=10.)
                points.append(xyz[valid])
            cloud=np.concatenate(points)[::10]  # Display decimation only.
        evidence=run/'artifacts'/row['evidence_file']
        data=json.loads(gzip.decompress(evidence.read_bytes()))['produced_targets']
        r=data['record'];start=data['teacher_provenance']['terminal_anchor_start']
        for ax,dims,name in zip(axes[i],((0,1),(0,2)),('俯视 XY','侧视 XZ')):
            ax.scatter(cloud[:,dims[0]],cloud[:,dims[1]],s=.5,c='#a7a7a7',alpha=.5)
            ax.scatter([0],[0],marker='^',s=60,c='black')
            for ai,a in enumerate(r['anchors']):
                xyz=np.asarray(a['position_m']);kind='路口' if ai<start else '尽头'
                ax.scatter(xyz[dims[0]],xyz[dims[1]],s=65,facecolors='white',edgecolors='#8250b5',linewidths=2,zorder=5)
                ax.annotate(kind+str(ai),xy=xyz[list(dims)],xytext=(4,5),textcoords='offset points',fontproperties=font,fontsize=9)
            for oi,o in enumerate(r['openings']):
                xyz=np.asarray(o['position_m']);direction=np.asarray(o['direction'])
                ax.scatter(xyz[dims[0]],xyz[dims[1]],s=35,marker='s',c='#226cb0',zorder=5)
                ax.arrow(xyz[dims[0]],xyz[dims[1]],direction[dims[0]],direction[dims[1]],width=.03,color='#226cb0')
                for ai,v in enumerate(r['membership'][oi]):
                    target=np.asarray(r['anchors'][ai]['position_m'])
                    color,style={True:('#248d4c','-'),False:('#d04636','--'),None:('#c18a22',':')}[v]
                    ax.plot([xyz[dims[0]],target[dims[0]]],[xyz[dims[1]],target[dims[1]]],color=color,ls=style,lw=1.3)
            ax.set(xlim=(-12,12),ylim=(-12,12),aspect='equal',xlabel='x (m)',ylabel=('y' if dims[1]==1 else 'z')+' (m)')
            ax.grid(alpha=.2);ax.set_title(split+' / '+name+'\n'+s['task']+' / '+str(s['source_sequence_id']),fontproperties=font,fontsize=10)
        provenance.append(dict(source=s,input_sha256=sha(path),evidence_sha256=sha(evidence),evidence_file=row['evidence_file']))
    fig.suptitle('真实五帧点云与自动监督证据——不是模型预测\n已完成'+str(report['completed_observations'])+'/159窗；未完成部分不纳入统计\n紫圈：参考结构；蓝方块：窗口通道截面（不是物理门口）\n绿线：有归属证据；红虚线：有排除证据；黄点线：未知；黑三角：当前传感器',fontproperties=font,fontsize=13)
    out=ROOT/'docs/figures/gse_local_pair_support_pilot_v1';out.mkdir(exist_ok=True)
    image=out/'fixed_examples.png'
    if image.exists():raise FileExistsError('no figure overwrite')
    fig.savefig(image,dpi=160);plt.close(fig)
    write(out/'support_population.json',report)
    write(out/'provenance.json',dict(selection='First source in frozen order per split; no model/target-score selection',
        display_only_decimation=10,sources=provenance,figure_sha256=sha(image),tool_sha256=sha(ROOT/'tools/v3/render_local_pair_support.py')))
    print(str(image))


if __name__=='__main__':main()
