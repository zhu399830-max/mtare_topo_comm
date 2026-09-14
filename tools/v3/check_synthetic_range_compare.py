"""Two preregistered synthetic distance ranges; no real data or tuning loop."""
from _bootstrap import PROJECT_ROOT as ROOT
import hashlib
import json
import os
import resource
import subprocess


def main():
    out=ROOT/'docs/figures/gse_graph/synthetic_range_compare_20260909'
    out.mkdir(parents=True,exist_ok=False)
    exe=ROOT/'build/gse_voxblox_cpu_v1/synthetic_range_compare'
    env=dict(os.environ,GLOG_logtostderr='1',OMP_NUM_THREADS='1',
        LD_LIBRARY_PATH=str(ROOT/'build/gse_voxblox_cpu_v1/prefix/usr/lib/x86_64-linux-gnu'))
    def limits():
        resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
        resource.setrlimit(resource.RLIMIT_CPU,(120,120))
        resource.setrlimit(resource.RLIMIT_FSIZE,(16*1024**2,16*1024**2))
        resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    graphs={}
    for distance in (2,5):
        with (out/f'range_{distance}.json').open('xb') as stdout,(out/f'range_{distance}.log').open('xb') as stderr:
            subprocess.run([str(exe),str(distance)],stdout=stdout,stderr=stderr,env=env,
                           preexec_fn=limits,timeout=150,check=True)
        graphs[distance]=json.loads((out/f'range_{distance}.json').read_text())
    a,b=graphs[2],graphs[5]
    for key in ('rays','observed_esdf','allocated_unknown','hallucinated','missing_tsdf_support'):
        if a[key]!=b[key]:raise ValueError('range comparison changed observation support')
    old=ROOT/'docs/figures/gse_graph/official_skeleton_synthetic_lidar_20260909/graph.json'
    raw=old.read_bytes()
    if hashlib.sha256(raw).hexdigest()!='32514f32f0bd709e1d4ee720dc67a73fe5d90ee48bcb0abb44610c6736a05e53':
        raise ValueError('original synthetic result drift')
    for key,value in json.loads(raw).items():
        if a[key]!=value:raise ValueError('2m control did not reproduce original graph')
    summary=dict(status='SYNTHETIC_RANGE_COMPARISON_NOT_RESEARCH',
        same_observation_support=True,original_2m_graph_reproduced=True,
        central_region='sphere radius1m at T junction coordinate origin',
        results={str(k):{name:v for name,v in g.items() if name not in ('vertices','connections')}
                 for k,g in graphs.items()},
        executable_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256((ROOT/'tools/v3/native_voxblox/synthetic_range_compare.cpp').read_bytes()).hexdigest())
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
