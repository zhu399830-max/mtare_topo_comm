"""Derived preview only: first saved development observation, no inference."""
import gzip
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mtare_topo.data.development_paired_scope import compile_scope, identity


def saved_reference_directions(root, source, target):
    import zarr
    from mtare_topo.teacher.gse_construction_paths_v3 import construction_incident_paths
    from mtare_topo.teacher.gse_directed_interface_binding_v1 import bind_axes
    from mtare_topo.teacher.saved_branch_evidence import read_saved_junction_branches
    wrapper=json.loads((root/'configs/v3/gate3/saved_branch_binding_scope_v1.json').read_text())
    raw=gzip.decompress(base64.b64decode(wrapper['payload']))
    if hashlib.sha256(raw).hexdigest()!='9965eeac86b827266b76b0318c3bf99f3dee39641cba47ccab3a80c8b25b2bcb':
        raise ValueError('direction scope drift')
    scope=json.loads(raw);opened={}
    row=next(r for r in scope['rows'] if r['source']==source)
    def read(path):
        data=(root/path).read_bytes();digest=hashlib.sha256(data).hexdigest()
        if digest!=scope['file_sha256'][path]:raise ValueError('direction source drift')
        opened[path]=digest;return data
    name=row['yaw_array'];info=scope['arrays'][name]
    array=zarr.open_array(store={k:read(name+'/'+k) for k in ['.zarray',*info['chunk_keys']]},mode='r')
    original=json.loads(gzip.decompress(read(row['reference_chain'][-1])))['raw_interfaces']
    groups=construction_incident_paths(json.loads(read(row['construction_path'])))
    branches=read_saved_junction_branches(target['record'],target['teacher_provenance'],
        bind_axes(groups,original['interfaces_teacher_only']),current_yaw_deg=float(array[row['yaw_row']]),ray_count=57600)
    result=[]
    for i,a in enumerate(target['teacher_provenance']['anchors']):
        for b in branches:
            if b.supported and b.node_id_teacher_only==a['node_id_teacher_only']:
                result.append(dict(anchor_index=i,direction=list(b.reference_direction_current_sensor)))
    return result,opened


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--corrective',action='store_true')
    parser.add_argument('--reference-directions',action='store_true')
    args=parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    tag='corrective' if args.corrective else 'paired'
    run = root / f'results/gate3_semantics/gate3_20260909_gse_development_{tag}_train_v1_seed0'
    scope = compile_scope(root)
    rows = [json.loads(line) for line in (run/'metrics/r0_final.jsonl').read_text().splitlines()]
    first = next(row for row in rows if row['scores'][0]['split'] == 'development')
    source = first['source']
    asset = next(a for a in scope['observations'] if identity(a['source']) == identity(source))
    def pinned(path, digest):
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError('input drift: '+str(path))
        return data
    cache = pinned(root/asset['cache_path'], asset['cache_sha256'])
    with np.load(io.BytesIO(cache), allow_pickle=False) as data:
        xyz = data['points_xyz_m'][0]
        xyz = xyz[data['valid'][0] & (np.linalg.norm(xyz.astype(float), axis=1) <= 10)]
    target = json.loads(gzip.decompress(pinned(root/asset['target_path'], asset['target_sha256'])))['produced_targets']
    n = target['teacher_provenance']['terminal_anchor_start']
    reference = np.array([a['position_m'] for a in target['record']['anchors'][:n]])
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    evidence = {'source': source, 'selection': 'first development record in sealed r0 final score order',
                'points': len(xyz), 'scope': 'junction positions and predicted directions, not graph edges',
                'reference_directions_shown': False, 'inputs': {asset['cache_path']: asset['cache_sha256'],
                asset['target_path']: asset['target_sha256']}, 'methods': {}}
    refs=[]
    if args.reference_directions:
        refs,opened=saved_reference_directions(root,source,target)
        evidence['reference_directions_shown']=True
        evidence['reference_directions']=refs;evidence['inputs'].update(opened)
    for column, method in enumerate(('r0', 'r1', 'r2')):
        score_path = run/'metrics'/f'{method}_final.jsonl'
        row = next(v for v in map(json.loads, score_path.read_text().splitlines()) if v['source'] == source)
        s = next(v for v in row['scores'] if v['matching_radius_m'] == 4 and v['matching_angle_deg'] == 10)
        prediction_path = run/'artifacts'/row['prediction_file']
        evidence['inputs'][str(score_path.relative_to(root))] = hashlib.sha256(score_path.read_bytes()).hexdigest()
        evidence['inputs'][str(prediction_path.relative_to(root))] = hashlib.sha256(prediction_path.read_bytes()).hexdigest()
        with np.load(prediction_path, allow_pickle=False) as p:
            selected = np.flatnonzero(p['presence_logits'] >= 0)
            if selected.tolist() != s['selected_anchor_query_indices']:
                raise ValueError('saved selection mismatch')
            evidence['methods'][method] = {'selected_queries': selected.tolist(), 'anchor_score': s['anchors'],
                                          'branch_score': s['branches']}
            for row_index, coordinate in enumerate((1, 2)):
                ax = axes[row_index, column]
                ax.scatter(xyz[:, 0], xyz[:, coordinate], s=.3, c='#a8afb8', alpha=.32, rasterized=True)
                ax.scatter(reference[:, 0], reference[:, coordinate], s=110, facecolors='none', edgecolors='#1261b0', linewidths=2, label='Reference junction')
                for ref in refs:
                    a=reference[ref['anchor_index']];d=np.array(ref['direction'])
                    ax.plot([a[0],a[0]+3*d[0]],[a[coordinate],a[coordinate]+3*d[coordinate]],'--',color='#148c46',linewidth=2)
                if refs:ax.plot([],[], '--',color='#148c46',label='Reference axis (3 m glyph)')
                pos = p['position_m'][selected]
                ax.scatter(pos[:, 0], pos[:, coordinate], marker='x', s=55, c='#cc3030', label='Predicted junction')
                for q in selected:
                    directions = p['directions'][q][p['branch_logits'][q] >= 0]
                    for d in directions:
                        a = p['position_m'][q]
                        ax.plot([a[0], a[0]+2*d[0]], [a[coordinate], a[coordinate]+2*d[coordinate]], color='#e28c16', alpha=.6, linewidth=1)
                ax.plot([], [], color='#e28c16', label='Selected direction (2 m glyph)')
                ax.scatter([0], [0], marker='^', c='black', s=45, label='Sensor')
                ax.set(xlim=(-10,10), ylim=(-10,10), xlabel='X (m)', ylabel=('Y' if coordinate==1 else 'Z')+' (m)',
                       title=f'{method.upper()} | {len(selected)} predicted junctions')
                ax.set_aspect('equal'); ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7, loc='lower left')
    fig.suptitle('Same observed cloud, frozen predictions, threshold 0.5\n'+source['task']+' / sequence '+str(source['source_sequence_id'])+'\nBlue circles: reference locations. Red crosses: predictions, NOT confirmed nodes. Orange: directions, NOT traversed edges.', fontsize=11)
    out = root/('docs/figures/gse_graph/corrective_selection_20260909' if args.corrective else 'docs/figures/gse_graph/paired_selection_20260909')
    if args.reference_directions:out=out.with_name(out.name+'_reference_axes')
    out.mkdir(parents=True, exist_ok=False)
    fig.savefig(out/'first_development.png', dpi=160)
    plt.close(fig)
    (out/'provenance.json').write_text(json.dumps(evidence, indent=2)+'\n')
    print(json.dumps({'output': str(out), 'source': source, 'points': len(xyz)}))


if __name__ == '__main__':
    main()
