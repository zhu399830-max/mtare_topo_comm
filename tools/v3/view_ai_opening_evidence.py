"""Three-state causal ray evidence for annotation; no opening labels."""
from _bootstrap import PROJECT_ROOT as ROOT
import json
import numpy as np
import torch
from mtare_topo.data.gse_membership_fit_reader import load_student_window
from mtare_topo.representation.primitive_relation_model import register_causal_lidar_points,MAXIMUM_RANGE_M
from mtare_topo.representation.gse_local_ray_segments import clip_observed_rays

def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib import font_manager
    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';font_manager.fontManager.addfont(font)
    plt.rcParams['font.family']=font_manager.FontProperties(fname=font).get_name();plt.rcParams['axes.unicode_minus']=False
    card=json.loads((ROOT/'configs/v3/gate3/data_cards/gse_ai_three_case_pilot_v1.json').read_text())
    out=ROOT/'docs/figures/gse_supervision_acquisition_pilot_v1'
    if (out/'ai_ray_evidence.json').exists() or (out/'ai_ray_evidence.png').exists():raise FileExistsError('preserve prior evidence')
    fig,axes=plt.subplots(3,5,figsize=(20,9),layout='constrained');rows=[]
    for case,e in enumerate(card['scope']['entries']):
        s=load_student_window(ROOT,dict(e,layout='saved_single_window',decoded_observations=1))
        rv=torch.stack((torch.tensor(s.ranges_m.copy())/MAXIMUM_RANGE_M,torch.tensor(s.valid_mask.copy(),dtype=torch.float32)),dim=1)[None]
        xyz,valid=register_causal_lidar_points(rv,torch.tensor(s.relative_translation_current_sensor_m.copy())[None],torch.tensor(s.relative_yaw_current_sensor_deg.copy())[None])
        xyz=xyz[0].reshape(-1,3).numpy();valid=valid[0].reshape(-1).numpy()
        origins=np.repeat(s.relative_translation_current_sensor_m,11520,axis=0)
        seg=clip_observed_rays(origins,xyz,valid)
        state=np.zeros(57600,dtype=np.uint8)
        state[seg.ray_indices]=np.where(seg.end_is_observed_return,1,2)
        frames=[]
        for f in range(5):
            frame=state[f*11520:(f+1)*11520].reshape(16,720)
            ax=axes[case,f]
            ax.imshow(frame,origin='lower',aspect='auto',extent=[0,360,-15,15],vmin=0,vmax=2,cmap=ListedColormap(['#9b9b9b','#ba6828','#27a8b4']),interpolation='nearest')
            ax.set(title=f'案例{case+1} / 帧{f+1}\n相对朝向{s.relative_yaw_current_sensor_deg[f]:.2f}°',xlabel='该帧传感器方位角（度）',ylabel='仰角（度）')
            frames.append(dict(frame_row=e['frame_rows'][f],relative_yaw_deg=float(s.relative_yaw_current_sensor_deg[f]),
                unknown_count=int((frame==0).sum()),local_return_count=int((frame==1).sum()),
                boundary_crossing_count=int((frame==2).sum()),boundary_ray_indices=(np.flatnonzero(frame.reshape(-1)==2)+f*11520).tolist()))
        rows.append(dict(task=e['task'],frames=frames))
        print(json.dumps(dict(task=e['task'],crossing_counts=[f['boundary_crossing_count'] for f in frames])))
    fig.suptitle('原始射线证据：青色=有效首返前穿出10米球，棕色=球内表面返回，灰色=未知/无局部段\n颜色不是开口标签；角度为各帧传感器坐标；穿出范围不证明地面可通行或全局连通')
    fig.savefig(out/'ai_ray_evidence.png',dpi=130);plt.close(fig)
    with (out/'ai_ray_evidence.json').open('x') as f:json.dump(dict(rows=rows,new_labels=0,training_steps=0,crop_boundary_is_opening=False),f,ensure_ascii=False)

if __name__=='__main__':main()
