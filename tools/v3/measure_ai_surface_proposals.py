"""Read-only measurement of sealed AI search regions, not new annotations."""
from _bootstrap import PROJECT_ROOT as ROOT
import hashlib,json
import numpy as np
import torch
from mtare_topo.data.gse_membership_fit_reader import load_student_window
from mtare_topo.representation.primitive_relation_model import register_causal_lidar_points,MAXIMUM_RANGE_M
RUN='results/gate3_semantics/gate3_20260911_gse_ai_three_case_pilot_v1_seed0'


def describe(points):
    if len(points)<3:return dict(count=len(points),plane=None)
    center=points.mean(0);_,singular,vh=np.linalg.svd(points-center,full_matrices=False)
    normal=vh[-1];normal*=1 if normal[np.argmax(abs(normal))]>=0 else -1
    residual=np.abs((points-center)@normal)
    return dict(count=len(points),center_m=center.tolist(),extent_m=np.ptp(points,axis=0).tolist(),
        normal=normal.tolist(),singular_values=singular.tolist(),
        plane_rms_m=float(np.sqrt(np.mean(residual**2))),plane_max_m=float(residual.max()))


def main():
    card=json.loads((ROOT/'configs/v3/gate3/data_cards/gse_ai_three_case_pilot_v1.json').read_text())
    pins={p:h for h,p in (x.split('  ',1) for x in (ROOT/RUN/'artifacts/evidence_sha256.txt').read_text().splitlines())}
    path=RUN+'/artifacts/ai_proposals.json';raw=(ROOT/path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=pins[path]:raise ValueError('proposal drift')
    proposals=json.loads(raw);rows=[]
    for case,e in zip(proposals['cases'],card['scope']['entries']):
        if case['task']!=e['task']:raise ValueError('case mismatch')
        s=load_student_window(ROOT,dict(e,layout='saved_single_window',decoded_observations=1))
        rv=torch.stack((torch.tensor(s.ranges_m.copy())/MAXIMUM_RANGE_M,torch.tensor(s.valid_mask.copy(),dtype=torch.float32)),dim=1)[None]
        xyz,valid=register_causal_lidar_points(rv,torch.tensor(s.relative_translation_current_sensor_m.copy())[None],torch.tensor(s.relative_yaw_current_sensor_deg.copy())[None])
        xyz=xyz[0].reshape(5,-1,3).numpy().astype(np.float64);valid=valid[0].reshape(5,-1).numpy()
        regions=[]
        for proposal in case['proposals']:
            for box in proposal.get('support_regions_xy_m',[]):
                x0,y0,x1,y1=box
                keep=valid&(xyz[:,:,0]>=x0)&(xyz[:,:,0]<=x1)&(xyz[:,:,1]>=y0)&(xyz[:,:,1]<=y1)
                measurements=[describe(xyz[f,keep[f]]) for f in range(5)]
                regions.append(dict(proposal=proposal['id'],box_xy_m=box,pooled=describe(xyz[keep]),
                    frames=measurements,frame_point_indices=[np.flatnonzero(keep[f]).tolist() for f in range(5)]))
        rows.append(dict(task=e['task'],regions=regions,nonmetric_relation_proposals=[p['id'] for p in case['proposals'] if 'support_regions_xy_m' not in p]))
        print(json.dumps(dict(task=e['task'],region_counts=[[f['count'] for f in r['frames']] for r in regions],pooled_rms=[r['pooled'].get('plane_rms_m') for r in regions])))
    output=dict(status='MEASUREMENT_ONLY_NO_LABEL_PROMOTION',proposal_sha256=pins[path],rows=rows,
        limits='Fixed AI XY boxes contain all heights; a PCA plane need not be one wall. Point counts alone do not establish connectivity or temporal correspondence.',
        new_labels=0,training_steps=0)
    path=ROOT/'docs/figures/gse_supervision_acquisition_pilot_v1/ai_surface_measurements.json'
    with path.open('x') as f:json.dump(output,f,ensure_ascii=False)


if __name__=='__main__':main()
