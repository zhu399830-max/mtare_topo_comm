"""Fixed-checkpoint, same-input development replay; no training or GT scores."""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from mtare_topo.representation.primitive_relation_observable_model import ObservableSparsePortRelationNet
from mtare_topo.integration.live_geometry_pipeline import LiveGeometryPipeline
from mtare_topo.integration.learned_geometry_trace import learned_geometry_record
from mtare_topo.integration.geometry_navigation_replay import GeometryNavigationReplay
from mtare_topo.topology.geometry_branch_places import GeometryBranchPlaces
from mtare_topo.semantics.range_exit_baseline import RangeExitBaseline
from mtare_topo.semantics.primitive_relation_nonlearning import ELEVATION_DEG,_sector_mask


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],required=True);a=parser.parse_args()
    out=Path(a.output);out.mkdir(exist_ok=False)
    scope=json.loads((ROOT/'configs/v3/gate6/data_cards/gse_learned_geometry_development_inputs_v1.json').read_text())['scope']
    torch.set_num_threads(1);torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    checkpoint=torch.load(ROOT/scope['checkpoint'],map_location='cpu',weights_only=False)
    model=ObservableSparsePortRelationNet();model.load_state_dict(checkpoint['model_state_dict'],strict=True)
    model.to(a.device).eval()
    data=np.load(a.input,allow_pickle=False);values=data['range_valid'];poses=data['poses'];keys=data['source_keys'];stamps=data['stamps']
    if len(values)!=scope['raw_frames']:raise ValueError('population drift')
    baseline=LiveGeometryPipeline(composition_policy=dict(max_residual_m=.01,min_crossing_sine=.1,endpoint_tolerance_m=1e-8,maximum_candidates=32),
        anchor_spacing_m=4,lookahead_m=4,rigid_motion=True)
    learned=GeometryNavigationReplay(anchor_spacing_m=4,lookahead_m=4)
    policy=dict(direction_tolerance_deg=15,target_match_m=2,place_radius_m=4,support_observations=3)
    places={k:GeometryBranchPlaces(**policy) for k in ['learned','nonlearning']}
    counters={k:dict(observations=0,primitives=0,proposals=0,current_supported=0) for k in places}
    started=time.monotonic();raw=[]
    with (out/'predictions.jsonl').open('x') as log:
        for i in range(len(values)):
            base=baseline.push(values[i,0]*50,values[i,1],sensor_to_map=poses[i],stamp_sec=float(stamps[i]),source_key=str(keys[i]),coordinate_frame='map')
            if i<4:continue
            window=scope['windows'][i-4]
            if list(keys[i-4:i+1])!=window['source_frame_keys']:raise ValueError('window identity mismatch')
            rotation=poses[i,:3,:3];relative=np.array([rotation.T@p[:3,:3] for p in poses[i-4:i+1]])
            trans=np.array([rotation.T@(p[:3,3]-poses[i,:3,3]) for p in poses[i-4:i+1]])
            relative[-1]=np.eye(3);trans[-1]=0.;yaw=np.degrees(np.arctan2(relative[:,1,0],relative[:,0,0]))
            def tensor(x):return torch.as_tensor(x,dtype=torch.float32,device=a.device).unsqueeze(0)
            with torch.inference_mode():prediction=model(tensor(values[i-4:i+1]),tensor(trans),tensor(yaw),relative_rotation_current_sensor=tensor(relative))
            arrays={k:v.detach().cpu().numpy()[0] for k,v in vars(prediction).items()}
            raw.append(arrays)
            mask=np.zeros(720,bool)
            for sector in RangeExitBaseline().predict(values[i,0]*50,values[i,1],np.asarray(ELEVATION_DEG))['sectors']:
                mask|=_sector_mask(sector['heading_robot_deg'],sector['angular_width_deg'])
            record=learned_geometry_record(axis_controls_sensor_m=arrays['axis_control_current_sensor_m'],
                existence_probabilities=torch.sigmoid(prediction.existence_logits)[0].cpu().numpy(),
                existence_threshold=scope['existence_threshold'],checkpoint_sha256=scope['source_sha256'][scope['checkpoint']],
                source_frame_keys=list(keys[i-4:i+1]),sensor_to_world=poses[i],coordinate_frame='map',timestamp=float(stamps[i]),
                current_sector_columns=np.flatnonzero(mask).tolist())
            decision=learned.update(record)
            pair={'learned':(record,decision),'nonlearning':(base['geometry'],base['decision'])}
            item=dict(timestamp=float(stamps[i]),methods={})
            for name,(r,d) in pair.items():
                event=places[name].observe(r,d['proposals'],navigation_node_id=d['node'])
                c=counters[name];c['observations']+=1;c['primitives']+=len(r['primitives']);c['proposals']+=len(d['proposals'])
                c['current_supported']+=sum(p.get('current_direction_supported') is True for p in d['proposals'])
                # Avoid copying millions of source references into paired summaries.
                item['methods'][name]=dict(axes=[p['axis_controls_world_m'] for p in r['primitives']],
                    targets=[p['axis_target_xyz_m'] for p in d['proposals']],current_support=[p.get('current_direction_supported') for p in d['proposals']],place_event=event)
            log.write(json.dumps(item,allow_nan=False)+'\n');log.flush()
    with (out/'raw_model_outputs.npz').open('xb') as f:np.savez_compressed(f,**{k:np.stack([r[k] for r in raw]) for k in raw[0]})
    for name,graph in [('learned',learned),('nonlearning',baseline.graph)]:
        with (out/(name+'_graph.json')).open('x') as f:json.dump(dict(graph=graph.snapshot(),places=places[name].snapshot()),f)
    summary=dict(status='DEVELOPMENT_INFERENCE_COMPLETE',methods=counters,elapsed_s=time.monotonic()-started,
        training_steps=0,teacher_reads=0,semantic_accuracy_measured=False,advantage_proven=False,
        note='Graph metric anchors share recorded poses by design; counts are not quality scores.')
    with (out/'summary.json').open('x') as f:json.dump(summary,f,indent=2)
    print(json.dumps(summary))


if __name__=='__main__':main()
