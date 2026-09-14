"""Read sealed outputs only; fixed-order overlays, no inference or selection."""
from _bootstrap import PROJECT_ROOT as ROOT
import hashlib
import json
import os
import numpy as np
from mtare_topo.data.cano_sensor_smoke import lidar_local_directions

RUN=ROOT/'results/gate6_single_robot/gate6_20260910_gse_learned_geometry_pair_v1_seed0'
OUT=ROOT/'docs/figures/gse_learned_geometry_pair_v1'


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()


def main():
    files=[RUN/'artifacts/inputs.npz',RUN/'artifacts/pair/predictions.jsonl']
    seal={line.split('  ',1)[1]:line.split('  ',1)[0] for line in (RUN/'artifacts/evidence_sha256.txt').read_text().splitlines()}
    for p in files:
        if sha(p)!=seal[str(p.relative_to(ROOT))]:raise ValueError('evidence drift')
    rows=[json.loads(l) for l in files[1].open()];data=np.load(files[0],allow_pickle=False)
    directions=np.asarray(lidar_local_directions());indices=np.linspace(0,len(rows)-1,6,dtype=int).tolist()
    OUT.mkdir(parents=True,exist_ok=False)
    os.environ.setdefault('MPLCONFIGDIR','/tmp/gse-learned-pair-mpl')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    diagnostics=[]
    for j,row in enumerate(rows):
        pose=data['poses'][j+4];origin=pose[:3,3];entry=dict(index=j,timestamp=row['timestamp'],methods={})
        for name,m in row['methods'].items():
            outside=0;intersect=0;extent=[]
            for controls in m['axes']:
                if controls is None:continue
                axis=np.asarray(controls)-origin;extent.append(float(np.linalg.norm(axis,axis=1).max()))
                delta=np.diff(axis,axis=0);den=np.sum(delta*delta,axis=1)
                t=np.clip(-np.sum(axis[:-1]*delta,axis=1)/np.maximum(den,1e-30),0,1)
                nearest=np.min(np.linalg.norm(axis[:-1]+t[:,None]*delta,axis=1))
                if nearest>10:outside+=1
                else:intersect+=1
            entry['methods'][name]=dict(axes=len(m['axes']),wholly_outside_10m=outside,intersects_10m=intersect,
                maximum_control_distance_m=max(extent,default=None),proposals=len(m['targets']),
                current_supported=sum(v is True for v in m['current_support']))
        diagnostics.append(entry)
    for view,dim in [('XY',1),('XZ',2)]:
        fig,axes=plt.subplots(3,2,figsize=(12,14))
        for ax,j in zip(axes.flat,indices):
            row=rows[j];pose=data['poses'][j+4];origin=pose[:3,3];points=[]
            for k in range(j,j+5):
                v=data['range_valid'][k];valid=v[1].astype(bool)
                p=directions[valid]*(v[0][valid,None]*50)
                points.append(p@data['poses'][k,:3,:3].T+data['poses'][k,:3,3]-origin)
            p=np.concatenate(points)[::8]  # fixed display thinning only
            ax.scatter(p[:,0],p[:,dim],s=.2,color='gray',alpha=.35,label='5-frame observed returns')
            for name,color in [('learned','tab:orange'),('nonlearning','tab:blue')]:
                first=True
                for controls in row['methods'][name]['axes']:
                    if controls is None:continue
                    a=np.asarray(controls)-origin
                    ax.plot(a[:,0],a[:,dim],'-o',color=color,lw=1.5,ms=3,label=name if first else None);first=False
            circle=plt.Circle((0,0),10,fill=False,color='black',linestyle='--',label='10m baseline fitting domain')
            ax.add_patch(circle);ax.scatter([0],[0],marker='^',c='red',s=35,label='current sensor')
            ax.set(xlim=(-50,50),ylim=(-50,50),aspect='equal',xlabel='relative world X (m)',ylabel=f'relative world {view[1]} (m)',title=f'window {j}, t={row["timestamp"]:.3f}s')
            ax.legend(fontsize=6);ax.grid(alpha=.2)
        fig.suptitle('Fixed-order same-input overlays; axes are NOT confirmed junctions or edges')
        fig.tight_layout();fig.savefig(OUT/(view+'.png'),dpi=130);plt.close(fig)
    summary={name:dict(total_axes=sum(d['methods'][name]['axes'] for d in diagnostics),
        wholly_outside_10m=sum(d['methods'][name]['wholly_outside_10m'] for d in diagnostics),
        intersects_10m=sum(d['methods'][name]['intersects_10m'] for d in diagnostics)) for name in ['learned','nonlearning']}
    result=dict(source_sha256={str(p.relative_to(ROOT)):sha(p) for p in files},fixed_window_indices=indices,
        diagnostic='10m spatial-domain comparability; NOT false-positive labels or accuracy',summary=summary,per_window=diagnostics)
    with (OUT/'diagnostics.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(summary))


if __name__=='__main__':main()
