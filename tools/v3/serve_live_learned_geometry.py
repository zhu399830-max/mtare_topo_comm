"""Frozen host-side inference for one local development simulation."""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse
import hashlib
import json
from pathlib import Path
import socket
import time
import numpy as np
import torch
from mtare_topo.integration.local_geometry_inference import receive,send
from mtare_topo.representation.primitive_relation_observable_model import ObservableSparsePortRelationNet

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--sha256',required=True);p.add_argument('--socket',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()!=a.sha256:raise ValueError('checkpoint drift')
    if a.socket.exists():raise ValueError('socket path already exists')
    a.output.mkdir(exist_ok=False)
    torch.set_num_threads(1);torch.manual_seed(0);torch.use_deterministic_algorithms(True)
    if torch.__version__!='2.9.0+cu129':raise ValueError('environment drift')
    model=ObservableSparsePortRelationNet()
    model.load_state_dict(torch.load(a.checkpoint,map_location='cpu',weights_only=False)['model_state_dict'],strict=True)
    model.eval()
    def forward(request):
        def t(key):return torch.as_tensor(request[key],dtype=torch.float32).unsqueeze(0)
        with torch.inference_mode():
            result=model(t('values'),t('translation'),t('yaw'),relative_rotation_current_sensor=t('rotation'))
        arrays={k:v.cpu().numpy()[0] for k,v in vars(result).items()}
        if any(not np.isfinite(v).all() for v in arrays.values()):raise ValueError('nonfinite model result')
        return arrays,torch.sigmoid(result.existence_logits)[0].cpu().numpy()
    # Synthetic warmup is not a map observation or training step.
    warmup=np.zeros((5,2,16,720));warmup[:,0,0,0]=.2;warmup[:,1,0,0]=1.
    forward(dict(values=warmup,translation=np.zeros((5,3)),yaw=np.zeros(5),rotation=np.tile(np.eye(3),(5,1,1))))
    count=0;previous=None;started=time.monotonic()
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as server:
        server.bind(str(a.socket));server.listen(1);server.settimeout(1.)
        (a.output/'ready.json').write_text(json.dumps(dict(checkpoint_sha256=a.sha256,device='cpu',training_steps=0)))
        with (a.output/'inference.jsonl').open('x') as log:
            while time.monotonic()-started<240 and count<400:
                try:connection,_=server.accept()
                except socket.timeout:continue
                with connection:
                    connection.settimeout(2.);request=receive(connection)
                    keys=request['source_frame_keys'].tolist();stamp=float(request['timestamp'])
                    if len(keys)!=5 or len(set(keys))!=5 or (previous is not None and stamp<=previous):raise ValueError('noncausal request')
                    start=time.monotonic();arrays,probabilities=forward(request)
                    evidence='prediction_%04d.npz'%count
                    with (a.output/evidence).open('xb') as f:np.savez_compressed(f,**arrays,source_frame_keys=keys,timestamp=stamp)
                    send(connection,dict(axes=arrays['axis_control_current_sensor_m'],probabilities=probabilities,
                        source_frame_keys=keys,timestamp=stamp,checkpoint_sha256=a.sha256,evidence=evidence))
                    log.write(json.dumps(dict(index=count,timestamp=stamp,source_frame_keys=keys,evidence=evidence,
                        inference_wall_s=time.monotonic()-start,checkpoint_sha256=a.sha256))+'\n');log.flush()
                    previous=stamp;count+=1

if __name__=='__main__':main()
