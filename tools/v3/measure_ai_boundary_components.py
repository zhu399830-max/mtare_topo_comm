"""Unfiltered angular visibility components and measured boundary pairs."""
from _bootstrap import PROJECT_ROOT as ROOT
import json
import numpy as np
import torch
from mtare_topo.data.gse_membership_fit_reader import load_student_window
from mtare_topo.representation.primitive_relation_model import register_causal_lidar_points,MAXIMUM_RANGE_M
from mtare_topo.representation.gse_local_ray_segments import clip_observed_rays


def components(state):
    height,width=state.shape;seen=set();result=[]
    for seed in np.flatnonzero(state.reshape(-1)==2):
        seed=int(seed)
        if seed in seen:continue
        seen.add(seed);queue=[seed];members=[];boundary=[]
        while queue:
            index=queue.pop();members.append(index);r,c=divmod(index,width)
            for nr,nc in [(r,(c-1)%width),(r,(c+1)%width),(r-1,c),(r+1,c)]:
                if not 0<=nr<height:
                    boundary.append([index,None,'fov']);continue
                j=nr*width+nc
                if state[nr,nc]==2:
                    if j not in seen:seen.add(j);queue.append(j)
                else:boundary.append([index,j,'surface' if state[nr,nc]==1 else 'unknown'])
        result.append(dict(indices=sorted(members),boundary=boundary))
    return result


def main():
    card=json.loads((ROOT/'configs/v3/gate3/data_cards/gse_ai_three_case_pilot_v1.json').read_text());rows=[]
    for e in card['scope']['entries']:
        s=load_student_window(ROOT,dict(e,layout='saved_single_window',decoded_observations=1))
        rv=torch.stack((torch.tensor(s.ranges_m.copy())/MAXIMUM_RANGE_M,torch.tensor(s.valid_mask.copy(),dtype=torch.float32)),dim=1)[None]
        xyz,valid=register_causal_lidar_points(rv,torch.tensor(s.relative_translation_current_sensor_m.copy())[None],torch.tensor(s.relative_yaw_current_sensor_deg.copy())[None])
        xyz=xyz[0].reshape(-1,3).numpy();valid=valid[0].reshape(-1).numpy()
        seg=clip_observed_rays(np.repeat(s.relative_translation_current_sensor_m,11520,axis=0),xyz,valid)
        state=np.zeros(57600,dtype=np.uint8);state[seg.ray_indices]=np.where(seg.end_is_observed_return,1,2)
        ends={int(i):p for i,p in zip(seg.ray_indices,seg.end_xyz_m)};frames=[]
        for f in range(5):
            groups=components(state[f*11520:(f+1)*11520].reshape(16,720));records=[]
            for group in groups:
                ids=[i+f*11520 for i in group['indices']];pairs=[]
                for i,j,kind in group['boundary']:
                    i+=f*11520;j=None if j is None else j+f*11520
                    pairs.append(dict(crossing_ray=i,neighbor_ray=j,kind=kind,
                        crop_xyz_m=ends[i].tolist(),surface_xyz_m=xyz[j].tolist() if kind=='surface' else None))
                records.append(dict(ray_indices=ids,boundary_pairs=pairs))
            frames.append(dict(frame_row=e['frame_rows'][f],components=records))
        rows.append(dict(task=e['task'],frames=frames))
        print(json.dumps(dict(task=e['task'],component_sizes=[[len(c['ray_indices']) for c in f['components']] for f in frames])))
    p=ROOT/'docs/figures/gse_supervision_acquisition_pilot_v1/ai_boundary_components.json'
    with p.open('x') as f:json.dump(dict(rows=rows,new_labels=0,training_steps=0,
        interpretation='4-neighbor visibility only; azimuth wraps, elevation does not. All components retained. Adjacent rays do not guarantee continuous physical surfaces or connectivity.'),f)

if __name__=='__main__':main()
