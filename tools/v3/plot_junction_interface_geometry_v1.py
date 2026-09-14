"""Derived source-geometry illustration, not labels or a new experiment."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.primitive_relation_materialization import load_p1a_realized_construction
from mtare_topo.teacher.csg_mesh_provenance import mesh_swept_superellipse
from mtare_topo.teacher.gse_cap_return_evidence_v1 import source_endpoint_cap_faces


def main():
    root = PROJECT_ROOT
    card_path = root/'configs/v3/gate3/data_cards/gse_surface_junction_interfaces_v1.json'
    card = json.loads(card_path.read_text())
    out = root/'docs/figures/gse_junction_interface_geometry_v1r'
    if out.exists():
        raise FileExistsError('preserve existing figure evidence')
    from matplotlib import font_manager
    font = next((x.name for x in font_manager.fontManager.ttflist if 'Noto Sans CJK' in x.name), None)
    if font is None:
        raise RuntimeError('Chinese font required before exporting report figure')
    plt.rcParams['font.family'] = font
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    selected = [(p, h) for p, h in card['scope']['file_sha256'].items()
                if '/constructions/' in p and 'S08_3d_loop_rich_C01__c1_mixed.json' in p]
    if len(selected) != 1:
        raise ValueError('exact previously diagnosed source required')
    path, expected = selected[0]
    raw = (root/path).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == expected
    document = json.loads(raw)
    _, primitives = load_p1a_realized_construction(document)
    lookup = {p.primitive_id: p for p in primitives}
    records = {p['primitive_id']: p for p in document['realized_primitives']}
    origin = np.array(records['primitive:edge_0086']['centerline_xyz_m'][-1])
    evidence = []
    for identity, side, color, label in (
            ('primitive:edge_0084', 1, '#2c7bb6', '支路'),
            ('primitive:edge_0086', 1, '#d7191c', '主通道一侧'),
            ('primitive:edge_0092', 0, '#1a9641', '主通道另一侧')):
        p = lookup[identity]
        mesh = mesh_swept_superellipse(p, axial_spacing_m=.05, angular_segments=64)
        cap = source_endpoint_cap_faces(mesh, angular_segments=64, endpoint_index=side)
        vertices = mesh.vertices_xyz_m[np.unique(mesh.triangle_vertex_indices[cap])]
        axis = np.array(records[identity]['centerline_xyz_m'])
        end = axis[0 if side == 0 else -1]
        away = axis[1 if side == 0 else -2]-end
        away /= np.linalg.norm(away)
        evidence.append(dict(primitive=identity, side=side, endpoint=end.tolist(),
                             away_direction=away.tolist()))
        local = vertices-origin
        for ax, dims in zip(axes, [(0, 1), (0, 2)]):
            a, b = dims
            ax.scatter(local[:, a], local[:, b], s=12, alpha=.65, color=color,
                       label=label+'：原端面顶点')
            start = end-origin
            finish = start + 4*away
            ax.annotate('', xy=finish[[a,b]], xytext=start[[a,b]],
                        arrowprops=dict(arrowstyle='->', color=color, lw=2.5))
    for ax, ylabel in zip(axes, ['Y', 'Z']):
        ax.set(xlabel='相对 X（米）', ylabel=f'相对 {ylabel}（米）',
               xlim=(-7,7), ylim=(-7,7), aspect='equal')
        ax.grid(alpha=.2)
        ax.legend(fontsize=9, loc='upper left')
    fig.suptitle('真实构造几何：两侧分支的端面近乎重合，但离开方向相反\n'
                 '箭头仅表示原轴线方向；这些构造端面不是模型检测到的开口', fontsize=13)
    out.mkdir()
    target = out/'source_interfaces.png'
    fig.savefig(target, dpi=150)
    plt.close(fig)
    (out/'provenance.json').write_text(json.dumps(dict(
        source=path, source_sha256=expected,
        selection='Previously diagnosed S08 mixed case; explanatory selection, not performance population',
        origin_world_m=origin.tolist(), interfaces=evidence,
        image_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        new_labels=0, model_predictions=False), ensure_ascii=False, indent=2)+'\n')
    print(target)


if __name__ == '__main__':
    main()
