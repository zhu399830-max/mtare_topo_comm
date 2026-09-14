"""Render saved common-domain graphs, without treating counts as accuracy."""
from _bootstrap import PROJECT_ROOT as ROOT
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUN=ROOT/'results/gate6_single_robot/gate6_20260910_gse_common_geometry_domain_v1_seed0'
OUT=ROOT/'docs/figures/gse_learned_geometry_pair_v1'

def main():
    seal={p:h for h,p in (s.split('  ',1) for s in (RUN/'artifacts/evidence_sha256.txt').read_text().splitlines())}
    graphs={}
    for n in ['learned','nonlearning']:
        p=RUN/('artifacts/'+n+'_graph.json');raw=p.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=seal[str(p.relative_to(ROOT))]:raise ValueError('seal mismatch')
        graphs[n]=json.loads(raw)
    fig,axs=plt.subplots(2,2,figsize=(12,8),sharex='col',sharey='col',constrained_layout=True)
    for row,(n,g) in enumerate(graphs.items()):
        nodes=np.array([p['xyz_m'] for p in g['graph']['nodes']])
        places=np.array([p['observer_xyz_m'] for p in g['places']['places']])
        for col,vertical in enumerate([1,2]):
            ax=axs[row,col];ax.plot(nodes[:,0],nodes[:,vertical],'o-',ms=3,color='0.35',label='Recorded metric graph')
            ax.scatter(places[:,0],places[:,vertical],facecolors='none',edgecolors='tab:orange',s=100,label='Observation places (NOT GT junctions)')
            ax.set_title(n+' / X'+('Y' if vertical==1 else 'Z'));ax.set_xlabel('X (m)');ax.set_ylabel(('Y' if vertical==1 else 'Z')+' (m)')
            ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('Same 10 m domain, same poses/backend: 290 windows. Counts are NOT accuracy.')
    fig.savefig(OUT/'common_domain_graphs.png',dpi=170);plt.close(fig)
    print(OUT/'common_domain_graphs.png')

if __name__=='__main__':main()
