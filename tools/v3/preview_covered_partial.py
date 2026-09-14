"""Read-only post-run visualization, not teacher execution or rescoring."""
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from _bootstrap import PROJECT_ROOT as ROOT
from mtare_topo.data.cano_sensor_smoke import lidar_local_directions, world_directions

RUN=ROOT/'results/gate3_semantics/gate3_20260908_gse_covered_partial_diagnostic_v1_seed20260906'
SEAL='b9f98274007ce324720d177f82585ebc82dfc4e420e946334e77a090bfedbdbd'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert sha(RUN/'SHA256_SEAL.json')==SEAL
    seal=json.loads((RUN/'SHA256_SEAL.json').read_text())
    scope=json.loads((RUN/'config/data_card.json').read_text())['scope']
    output=ROOT/'docs/figures/gse_graph/covered_partial_20260908'
    output.mkdir(exist_ok=False)
    fig,axes=plt.subplots(2,4,figsize=(16,8),constrained_layout=True)
    summaries=[];local=lidar_local_directions().reshape(-1,3).astype(float)
    for name in scope['cases']:
        relative='artifacts/'+name+'.json.gz';path=RUN/relative
        assert sha(path)==seal[relative]
        d=json.loads(gzip.decompress(path.read_bytes()));t=d['partial_targets'];r=t['record']
        summary=dict(d['summary'],unknown_candidates=t['unknown_candidates'],
            anchor_ids=[a['node_id_teacher_only'] for a in t['teacher_provenance']['anchors']],
            opening_sources=[a['primitive_id_teacher_only'] for a in t['teacher_provenance']['openings']],
            relation_status=[a['status'] for a in t['teacher_provenance']['relations']],
            anchors_complete=r['score_region']['anchors_complete'],openings_complete=r['score_region']['openings_complete'])
        summaries.append(summary)
        if '__circle__' not in name:continue # fixed representative section, not selected by score
        view=int(name[-1]);prefix=scope['export_run']+'/artifacts/'+name
        for suffix in ('.student.npz','.diagnostic.npz'):
            assert sha(ROOT/(prefix+suffix))==scope['input_sha256'][prefix+suffix]
        with np.load(ROOT/(prefix+'.student.npz')) as s,np.load(ROOT/(prefix+'.diagnostic.npz')) as pose:
            center=pose['sensor_xyz_m'][-1];yaw=pose['yaw_deg'][-1]
            points=[]
            for frame in range(5):
                directions=world_directions(local,float(pose['yaw_deg'][frame]))
                directions/=np.linalg.norm(directions,axis=1,keepdims=True)
                valid=s['valid_mask'][frame].reshape(-1).astype(bool)
                xyz=pose['sensor_xyz_m'][frame]+directions*s['ranges_m'][frame].reshape(-1,1)
                points.append(xyz[valid])
            points=np.concatenate(points)-center
            a=np.deg2rad(yaw);rot=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
            points=points@rot
        points=points[np.linalg.norm(points,axis=1)<=12][::8] # display only, never label filtering
        for ax,coords,label in ((axes[0,view],(0,1),'XY'),(axes[1,view],(0,2),'XZ')):
            i,j=coords
            ax.scatter(points[:,i],points[:,j],s=.5,c='#8296a8',alpha=.25,rasterized=True)
            ax.scatter(0,0,c='black',marker='^',s=40,label='Sensor')
            for ai,anchor in enumerate(r['anchors']):
                p=np.array(anchor['position_m']);ax.scatter(p[i],p[j],c='#007a58',marker='s',s=50)
                ax.annotate(summary['anchor_ids'][ai],(p[i],p[j]),xytext=(3,4),textcoords='offset points',fontsize=8)
            for oi,opening in enumerate(r['openings']):
                p=np.array(opening['position_m']);direction=np.array(opening['direction'])
                ax.scatter(p[i],p[j],c='#d97400',marker='o',s=35)
                ax.arrow(p[i],p[j],direction[i],direction[j],color='#d97400',head_width=.25,length_includes_head=True)
                ax.annotate('O'+str(oi),(p[i],p[j]),xytext=(3,3),textcoords='offset points',fontsize=7)
                for ai,member in enumerate(r['membership'][oi]):
                    if member is True:
                        q=np.array(r['anchors'][ai]['position_m'])
                        ax.plot([p[i],q[i]],[p[j],q[j]],color='#007a58',ls='--',lw=.8)
            ax.set(xlim=(-12,12),ylim=(-12,12) if j==1 else (-4,4),xlabel='X (m)',ylabel=label[1]+' (m)',title=f'view{view} {label}: {len(r["anchors"])} anchors / {len(r["openings"])} openings')
            ax.set_aspect('equal');ax.grid(alpha=.15)
    fig.suptitle('Automatic partial references, NOT model predictions | square: reference anchor; orange: window opening\nDashed lines: supported membership, NOT traversed graph edges; unknown pairs omitted',fontsize=11)
    fig.savefig(output/'circle_four_views.png',dpi=170)
    fig.savefig(output/'circle_four_views.pdf')
    plt.close(fig)
    (output/'case_summary.json').write_text(json.dumps(summaries,indent=2))
    (output/'provenance.json').write_text(json.dumps(dict(source_run=str(RUN.relative_to(ROOT)),source_seal=SEAL,
        script_sha256=sha(Path(__file__)),display_only_stride=8,display_radius_m=12,
        section_selection='fixed circle all four views; all12 summarized',
        model_predictions=False,new_labels=False),indent=2))
    print(json.dumps(dict(output=str(output),cases=len(summaries))))


if __name__=='__main__':main()
