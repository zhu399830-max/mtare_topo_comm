"""Saved input coverage and a preidentified missing-surface case, not accuracy."""
from _bootstrap import PROJECT_ROOT as ROOT
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUN=ROOT/'results/gate3_semantics/gate3_20260911_gse_common_observation_export_v1_seed0'
REF=ROOT/'results/gate3_semantics/gate3_20260911_gse_membership_fit_v1r1_seed0'
OUT=ROOT/'docs/figures/gse_common_observation_v1'


def main():
    lines=(RUN/'artifacts/evidence_sha256.txt').read_text().splitlines()
    for line in lines:
        h,p=line.split('  ',1)
        if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h:raise ValueError('export seal drift')
    seals={p:h for h,p in (line.split('  ',1) for line in (REF/'artifacts/evidence_sha256.txt').read_text().splitlines())}
    rows=[]
    for i in range(16):
        path=f'artifacts/reference_{i:02d}.json';raw=(REF/path).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=seals[path]:raise ValueError('reference seal drift')
        target=json.loads(raw)
        with np.load(RUN/f'artifacts/window_{i:02d}.npz',allow_pickle=False) as z:
            rays=set(z['ray_ray_indices'].tolist());points=set(z['surface_return_indices'].tolist())
            for j,o in enumerate(target['teacher_provenance']['openings']):
                rows.append(dict(observation=i,opening=j,surface_witnesses=len(o['surface_ray_indices']),
                    surface_witnesses_in_local_points=len(points.intersection(o['surface_ray_indices'])),
                    surface_witnesses_in_local_rays=len(rays.intersection(o['surface_ray_indices'])),
                    crossing_witnesses=len(o['crossing_ray_indices']),crossing_witnesses_in_local_rays=len(rays.intersection(o['crossing_ray_indices']))))
            if i==3:
                o=target['teacher_provenance']['openings'][0]
                keep=np.isin(z['ray_ray_indices'],o['surface_ray_indices'])
                starts=z['ray_start_xyz_m'][keep];ends=z['ray_end_xyz_m'][keep]
                pts=z['registered_returns_xyz_m'][z['surface_return_indices']][::20]
    summary=json.loads((RUN/'metrics/summary.json').read_text())
    totals={k:sum(w[k] for w in summary['windows']) for k in ('measured_local_returns','local_segments','clipped_non_surface_endpoints')}
    result=dict(coverage=rows,totals=totals,seal_entries_verified=len(lines),training_steps=0,
        reference_use='Post-export coverage report only, not model forward or input selection',
        interpretation='Retained evidence does not establish detection or geometry advantage')
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'coverage.json'
    if path.exists():
        if json.loads(path.read_text())!=result:raise ValueError('existing coverage differs')
    else:
        with path.open('x') as f:json.dump(result,f,indent=2)
    from matplotlib import font_manager
    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font_manager.fontManager.addfont(font)
    plt.rcParams['font.family']=font_manager.FontProperties(fname=font).get_name()
    plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(1,2,figsize=(11,5),layout='constrained')
    for ax,dims,title in zip(axes,((0,1),(0,2)),('俯视 XY','侧视 XZ')):
        ax.scatter(pts[:,dims[0]],pts[:,dims[1]],s=2,c='0.65',label='局部真实返回点（间隔显示）')
        for k,(a,b) in enumerate(zip(starts,ends)):
            ax.plot([a[dims[0]],b[dims[0]]],[a[dims[1]],b[dims[1]]],c='teal',alpha=.6,label='被保留的观测射线段' if k==0 else None)
        ax.scatter(ends[:,dims[0]],ends[:,dims[1]],marker='s',s=18,c='orange',label='裁剪端点：不是表面或开口')
        ax.set(title=title,xlim=(-11,11),ylim=(-11,11),aspect='equal',xlabel='米',ylabel='米');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8,loc='lower left')
    fig.suptitle('此前定位的第4窗：18条表面见证均在局部点集之外，现保留其窗内射线\n这是输入覆盖示意，不是模型预测结果',fontsize=11)
    fig.savefig(OUT/'outside_return_evidence.png',dpi=150);plt.close(fig)
    print(json.dumps(totals))


if __name__=='__main__':main()
