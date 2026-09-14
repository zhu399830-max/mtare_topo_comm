"""Read sealed outputs only; no rerun, teacher or tuning."""
from _bootstrap import PROJECT_ROOT as ROOT
import hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

RUN=ROOT/'results/gate3_semantics/gate3_20260912_gse_short_observation_chain_v1_seed0'
DEST=ROOT/'docs/figures/gse_conditional_geometry_fit_v1/short_observation_chain'

def main():
    count=0
    for line in (RUN/'artifacts/evidence_sha256.txt').read_text().splitlines():
        h,p=line.split('  ',1)
        if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h:raise ValueError('sealed evidence drift '+p)
        count+=1
    DEST.mkdir(parents=True,exist_ok=True)
    data=np.load(RUN/'artifacts/inputs.npz',allow_pickle=False)
    poses=data['poses'];ranges=data['ranges_m'];valid=data['valid_mask']
    from mtare_topo.semantics.primitive_relation_nonlearning import ELEVATION_DEG
    az=np.deg2rad(np.arange(720)*.5);el=np.deg2rad(ELEVATION_DEG)
    rays=np.stack(np.broadcast_arrays(np.cos(el)[:,None]*np.cos(az)[None,:],np.cos(el)[:,None]*np.sin(az)[None,:],np.sin(el)[:,None]+np.zeros((16,720))),axis=-1)
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    fig,axs=plt.subplots(2,3,figsize=(15,9))
    # Fixed first window, not selected for attractive outputs.
    row=json.loads((RUN/'artifacts/window_00.json').read_text());points=[]
    for i in range(5):
        p=(rays*ranges[i,:,:,None])[valid[i].astype(bool)]
        p=p@poses[i,:3,:3].T+poses[i,:3,3];points.append(p)
    points=np.concatenate(points);origin=poses[4,:3,3]
    local=points[np.linalg.norm(points-origin,axis=1)<=10]
    titles=['原始五帧点云','距离扇区：4个候选方向','表面拟合：1条轴、2个方向']
    for line,(a,b) in enumerate([(0,1),(0,2)]):
        for col,ax in enumerate(axs[line]):
            # Display-only deterministic thinning; full inputs and outputs retained.
            p=local[::5];ax.scatter(p[:,a]-origin[a],p[:,b]-origin[b],s=1,c='lightgray',rasterized=True)
            ax.plot(poses[:5,a,3]-origin[a],poses[:5,b,3]-origin[b],'k.-',lw=1.5)
            if col:
                method=['range_sectors','observed_surface_fit'][col-1]
                for proposal in row['method_proposals'][method]:
                    x=np.asarray([proposal['axis_start_xyz_m'],proposal['axis_target_xyz_m']])-origin
                    ax.annotate('',xy=x[1,[a,b]],xytext=x[0,[a,b]],arrowprops=dict(arrowstyle='->',color='tab:orange',lw=2))
            if col==2:
                for p in row['geometry']['primitives']:
                    if p['axis_controls_world_m'] is not None:
                        x=np.asarray(p['axis_controls_world_m'])-origin;ax.plot(x[:,a],x[:,b],color='tab:blue',lw=2)
            ax.scatter(0,0,c='red',s=24);ax.set_xlim(-10,10);ax.set_ylim(-10,10);ax.set_aspect('equal');ax.grid(alpha=.2)
            ax.set_title(titles[col]+('（俯视）' if line==0 else '（侧视）'),fontproperties=font)
            ax.set_xlabel('x (m)');ax.set_ylabel(('y' if b==1 else 'z')+' (m)')
    fig.suptitle('同一输入的实际输出：灰色为回波，橙色为候选，蓝色为拟合轴；不是已确认开口或自动探索',fontproperties=font)
    fig.tight_layout();fig.savefig(DEST/'first_window_evidence.png',dpi=160);plt.close(fig)
    evidence=dict(seal_verified=count,source_run=str(RUN.relative_to(ROOT)),window=0,full_local_return_count=len(local),
        teacher_reads=0,new_predictions=0,display_stride=5,interpretation='Candidate count disagreement, not false-positive or missed-branch ground truth',
        result=json.loads((RUN/'metrics/chain.json').read_text()))
    with (DEST/'evidence.json').open('x') as f:json.dump(evidence,f,indent=2)
    print(json.dumps(evidence))

if __name__=='__main__':main()
