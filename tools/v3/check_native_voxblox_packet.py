"""Synthetic protocol smoke; no data/checkpoint files or research run."""
import json
import os
from pathlib import Path
import subprocess
import numpy as np
from mtare_topo.data.native_voxblox_packet import encode_observation

root=Path(__file__).resolve().parents[2]
exe=root/'build/gse_voxblox_cpu_v1/cmake/observation_graph'
env=dict(os.environ,LD_LIBRARY_PATH=str(root/'build/gse_voxblox_cpu_v1/prefix/usr/lib/x86_64-linux-gnu'))
v=np.zeros((5,2,16,720),np.float32)
packet=encode_observation(v,np.zeros((5,3)),np.zeros(5))
def call(raw):return subprocess.run([str(exe)],input=raw,capture_output=True,env=env,timeout=30)
result=call(packet)
assert result.returncode==0, result.stderr.decode()
empty=json.loads(result.stdout)
assert empty['rays']==0 and empty['nodes']==0 and empty['edges']==0
for bad in (b'bad\n', packet+b'teacher_id 42\n', packet.replace(b'GSE_RANGE_V1 5',b'GSE_RANGE_V1 6'),packet[:-20]):
    assert call(bad).returncode==2
v[:,0]=.1;v[:,1]=1
result=call(encode_observation(v,np.zeros((5,3)),np.zeros(5)))
assert result.returncode==0,result.stderr.decode()
sphere=json.loads(result.stdout)
assert sphere['rays']==57600 and sphere['missing_tsdf_support']==0
assert sphere['hallucinated']==0
print(json.dumps(dict(status='SYNTHETIC_PROTOCOL_PASS_NOT_GEOMETRY_QUALIFICATION',
    empty_nodes=empty['nodes'],rejected_packets=4,nonempty_rays=sphere['rays'],
    nonempty_nodes=sphere['nodes'],nonempty_edges=sphere['edges'])))
