"""Saved-output analysis and fixed first-development scene; no inference."""
from _bootstrap import PROJECT_ROOT as ROOT
import hashlib
import io
import json
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.font_manager import FontProperties
from verify_native_population_v2 import verify, RUN


def main():
    verification=verify();run=ROOT/RUN
    manifest=json.loads((run/'artifacts/native_graph_manifest.json').read_text())
    card=json.loads((run/'config/data_card.json').read_text())
    stats=defaultdict(list); graphs={}; opened={}
    def pinned(path,h):
        raw=(ROOT/path).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=h:raise ValueError('source drift '+path)
        opened[path]=h;return raw
    for entry in manifest['observations']:
        graph=json.loads(pinned(RUN+'/'+entry['graph_file'],entry['graph_sha256']))
        adj={v['id']:set() for v in graph['vertices']}
        for a,b in graph['connections']:adj[a].add(b);adj[b].add(a)
        unseen=set(adj);sizes=[]
        while unseen:
            stack=[unseen.pop()];n=0
            while stack:
                v=stack.pop();n+=1;new=adj[v]&unseen;unseen-=new;stack.extend(new)
            sizes.append(n)
        row=dict(source=entry['source'],parent_id=entry['parent_id'],nodes=graph['nodes'],edges=graph['edges'],
            components=len(sizes),largest_component_nodes=max(sizes,default=0),
            self_loops=entry['self_loop_edges'],unknown_edges=entry['unknown_sample_edges'],
            low_clearance_edges=entry['low_clearance_sample_edges'])
        stats[entry['split']].append(row)
        if entry['split']=='development' and not graphs:graphs=dict(entry=entry,graph=graph)
    summary={}
    for split,rows in stats.items():
        summary[split]=dict(observations=len(rows),parents=len({r['parent_id'] for r in rows}),
            median_nodes=float(np.median([r['nodes'] for r in rows])),
            median_components=float(np.median([r['components'] for r in rows])),
            observations_with_unknown_edges=sum(r['unknown_edges']>0 for r in rows),
            observations_with_low_clearance_edges=sum(r['low_clearance_edges']>0 for r in rows),
            totals={k:sum(r[k] for r in rows) for k in ('nodes','edges','self_loops','unknown_edges','low_clearance_edges')})
    e,g=graphs['entry'],graphs['graph'];source=e['source']
    asset=next(r for r in card['scope']['observations'] if r['source']==source)
    with np.load(io.BytesIO(pinned(asset['cache_path'],asset['cache_sha256'])),allow_pickle=False) as a:
        xyz=a['points_xyz_m'][0][a['valid'][0]]
    prior=ROOT/'docs/figures/gse_graph/corrective_selection_20260909_reference_axes/provenance.json'
    evidence=json.loads(prior.read_text())
    if evidence['source']!=source:raise ValueError('not same fixed development scene')
    predpath=next(p for p in evidence['inputs'] if '/artifacts/r2_final/' in p)
    with np.load(io.BytesIO(pinned(predpath,evidence['inputs'][predpath])),allow_pickle=False) as p:
        selected=p['presence_logits']>=0;pos=p['position_m'][selected]
        dirs=[d[l>=0] for d,l in zip(p['directions'][selected],p['branch_logits'][selected])]
    nodes={v['id']:np.array(v['xyz']) for v in g['vertices']}
    segments=np.array([[nodes[a],nodes[b]] for a,b in g['connections']])
    colors=['#9c27b0' if a['unknown_samples'] and a['below_0_4m_samples'] else
            '#e69f00' if a['unknown_samples'] else '#d62728' if a['below_0_4m_samples'] else
            '#2578a8' for a in g['edge_audit']]
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    fig,axes=plt.subplots(2,3,figsize=(16,10),constrained_layout=True)
    titles=['官方候选骨架：完整观测范围','同一骨架：局部放大','学习输出：路口位置与方向']
    for row,dim in enumerate((1,2)):
        for col in range(3):
            ax=axes[row,col]
            ax.scatter(xyz[:,0],xyz[:,dim],s=.15,c='#999999',alpha=.18,rasterized=True)
            if col<2:
                ax.add_collection(LineCollection(segments[:,:,[0,dim]],colors=colors,linewidths=.55,alpha=.75))
                loops=np.array([nodes[a] for a,b in g['connections'] if a==b])
                if len(loops):ax.scatter(loops[:,0],loops[:,dim],s=14,facecolors='none',edgecolors='black',linewidths=.5)
            else:
                ax.scatter(pos[:,0],pos[:,dim],marker='x',c='red',s=65)
                for a,ds in zip(pos,dirs):
                    for d in ds:ax.plot([a[0],a[0]+2*d[0]],[a[dim],a[dim]+2*d[dim]],c='#e28c16')
            ax.scatter([0],[0],marker='^',c='black',s=35)
            if col>0:ax.set(xlim=(-10,10),ylim=(-10,10))
            else:ax.autoscale_view()
            ax.set_aspect('equal');ax.set_xlabel('X (m)');ax.set_ylabel(('Y' if dim==1 else 'Z')+' (m)')
            ax.set_title(titles[col],fontproperties=font);ax.grid(alpha=.15)
    fig.suptitle('同一五帧扫描，不同输出含义；不是优劣排名\n蓝：其余候选边（未证明安全）  橙：未知采样  红：低净空  紫：两者都有  空心圈：自连接\n右列红叉：预测路口，橙线：方向符号，不是图边；黑三角：传感器',fontproperties=font,fontsize=13)
    out=ROOT/'docs/figures/gse_graph/native_population_v2_20260909'
    out.mkdir(parents=True,exist_ok=False)
    fig.savefig(out/'first_development.png',dpi=150);fig.savefig(out/'first_development.pdf');plt.close(fig)
    result=dict(verification=verification,split_summary=summary,observations=dict(stats),
        fixed_scene=source,selection='first development entry in frozen native manifest; matches previous learning preview',
        input_sha256=opened,figure_scope='Native full graph and 20m square viewport versus saved R2 direction glyphs; not same output task or safety ranking')
    (out/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(output=str(out),summary=summary),indent=2))


if __name__=='__main__':main()
