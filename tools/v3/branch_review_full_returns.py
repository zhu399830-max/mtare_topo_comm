"""Interactive review view extension: same sealed arrays, no new labels."""
from ai_branch_review import ROOT,RUN,CARD,CACHE,sha,write
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    out=ROOT/RUN
    assert json.loads((out/'RUN_STATE.json').read_text())['state']=='RUNNING'
    card=json.loads((ROOT/CARD).read_text())
    write(out/'config/full_return_view_extension.json',dict(reason='10m crop hides evidence already present in full causal returns; inspect it without changing model domain or adding data',
        source='tools/v3/branch_review_full_returns.py',sha256=sha(ROOT/'tools/v3/branch_review_full_returns.py'),labels_generated=0,model_domain_changed=False))
    for i,e in enumerate(card['scope']['entries']):
        with np.load(ROOT/CACHE/f'artifacts/common_{i:02d}.npz',allow_pickle=False) as d:xyz=d['registered_returns_xyz_m'].copy()
        with np.load(ROOT/e['student_path'],allow_pickle=False) as d:valid=d['valid_mask'].astype(bool).reshape(-1)
        fig,axs=plt.subplots(1,2,figsize=(16,8))
        xyz=xyz[valid&np.isfinite(xyz).all(1)]
        for ax,(a,b) in zip(axs,[(0,1),(0,2)]):
            q=xyz[::2];ax.scatter(q[:,a],q[:,b],s=.4,c=q[:,2],cmap='viridis');ax.scatter(0,0,c='red',marker='+');ax.set_aspect('equal');ax.grid(alpha=.2);ax.set(xlabel='X m',ylabel='Y m' if b==1 else 'Z m')
            t=np.linspace(0,2*np.pi,200);ax.plot(10*np.cos(t),10*np.sin(t),'r--',alpha=.4)
        fig.suptitle(f'Observation {i:02d}: all valid five-frame returns; red circle=10m crop, NOT physical boundary; no GT')
        fig.tight_layout();fig.savefig(out/f'previews/full_{i:02d}.png',dpi=100);plt.close(fig)
    print('All12 full-return views created;0labels')

if __name__=='__main__':main()
