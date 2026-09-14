"""Saved student-only first/last views. Does not assign correspondence labels."""
from _bootstrap import PROJECT_ROOT as ROOT
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from mtare_topo.semantics.primitive_relation_nonlearning import ELEVATION_DEG

RUN=ROOT/'results/gate3_semantics/gate3_20260912_gse_direction_task_pilot_inputs_v1_seed20260912'
OUT=ROOT/'docs/figures/gse_conditional_geometry_fit_v1/direction_task_pilot'

def main():
    rows=json.loads((RUN/'artifacts/manifest.json').read_text());selected=[]
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    el=np.deg2rad(ELEVATION_DEG);az=np.deg2rad(np.arange(720)*.5)
    rays=np.stack(np.broadcast_arrays(np.cos(el)[:,None]*np.cos(az),np.cos(el)[:,None]*np.sin(az),np.sin(el)[:,None]+np.zeros((16,720))),axis=-1)
    fig,axes=plt.subplots(3,4,figsize=(16,12));evidence=[]
    for block in range(3):
        group=sorted([r for r in rows if r['source']['fragment_id']==block],key=lambda r:r['source']['sequence_row'])
        for col,r in enumerate([group[0],group[-1]]):
            p=RUN/r['student_path']
            if hashlib.sha256(p.read_bytes()).hexdigest()!=r['student_sha256']:raise ValueError('student drift')
            data=np.load(p,allow_pickle=False);points=[]
            for i in range(5):
                xyz=(rays*data['ranges_m'][i,:,:,None])[data['valid_mask'][i].astype(bool)]
                a=np.deg2rad(data['relative_yaw_current_sensor_deg'][i]);rot=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
                xyz=xyz@rot.T+data['relative_translation_current_sensor_m'][i];points.append(xyz[np.linalg.norm(xyz,axis=1)<=10])
            xyz=np.concatenate(points);trajectory=data['relative_translation_current_sensor_m']
            evidence.append(dict(fragment=block,source=r['source'],student_sha256=r['student_sha256'],display_local_returns=len(xyz),display_stride=5,reference_payload_read=False,labels_generated=0))
            for view,(a,b) in enumerate([(0,1),(0,2)]):
                ax=axes[block,col*2+view];q=xyz[::5]
                ax.scatter(q[:,a],q[:,b],s=1,c=q[:,2],cmap='viridis',vmin=-5,vmax=5)
                ax.plot(trajectory[:,a],trajectory[:,b],'r.-');ax.scatter(0,0,c='red',s=15)
                ax.set_xlim(-10,10);ax.set_ylim(-10,10);ax.set_aspect('equal');ax.grid(alpha=.2)
                ax.set_title(f"片段{block+1} / {'首' if col==0 else '末'}窗口 / {'俯视' if view==0 else '侧视'}",fontproperties=font)
                ax.set_xlabel('x (m)');ax.set_ylabel(('y' if b==1 else 'z')+' (m)')
    fig.suptitle('三个固定训练区路口邻域片段：仅原五帧观测，红线为历史运动；尚未赋通道对应标签',fontproperties=font)
    fig.tight_layout();OUT.mkdir(parents=True,exist_ok=True);fig.savefig(OUT/'observations.png',dpi=140);plt.close(fig)
    with (OUT/'preview_sources.json').open('x') as f:json.dump(evidence,f,indent=2)
    print(json.dumps(dict(preview_observations=len(evidence),labels_generated=0,path=str(OUT/'observations.png'))))

if __name__=='__main__':main()
