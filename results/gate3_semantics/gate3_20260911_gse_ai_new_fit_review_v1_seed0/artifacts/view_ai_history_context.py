"""Read-only review of already exported five-frame returns; no label changes."""
from _bootstrap import PROJECT_ROOT as ROOT
import json,hashlib
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
SOURCE='results/gate3_semantics/gate3_20260911_gse_ai_indexed_review_v1r1_seed0'
OUT='docs/figures/gse_supervision_acquisition_pilot_v1'

def main():
    manifest=json.loads((ROOT/SOURCE/'artifacts/review_manifest.json').read_text())
    for entry in manifest['observations']:
        raw=(ROOT/SOURCE/entry['bundle']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=entry['sha256']:raise ValueError('bundle drift')
        b=json.loads(raw);xyz=np.asarray(b['points_xyz_m']);slots=np.asarray(b['point_history_slots'])
        fig,axes=plt.subplots(5,3,figsize=(15,20))
        for f in range(5):
            x=xyz[slots==f];local=x[np.linalg.norm(x,axis=1)<=10]
            for col,(pts,ij,title) in enumerate([(x,(0,1),'all XY'),(local,(0,1),'local10m XY'),(x,(0,2),'all XZ')]):
                ax=axes[f,col];ax.scatter(pts[:,ij[0]],pts[:,ij[1]],s=.6,c=pts[:,2],cmap='viridis',vmin=-5,vmax=5);ax.plot(0,0,'r+');ax.grid(alpha=.2);ax.set_aspect('equal',adjustable='datalim');ax.set_title(f'frame {b["source_frame_indices"][f]} / {title}')
                if col==1:ax.set_xlim(-10,10);ax.set_ylim(-10,10)
        fig.suptitle(f'Case {entry["case"]}: five original frames in current sensor coordinates; color=height\nDisplay only; no teacher or model output. Prior reference exposure disclosed.')
        fig.tight_layout(rect=(0,0,1,.96));p=ROOT/OUT/f'ai_history_case_{entry["case"]}.png'
        if p.exists():raise FileExistsError('do not overwrite history evidence')
        fig.savefig(p,dpi=130);plt.close(fig);print(p)

if __name__=='__main__':main()
