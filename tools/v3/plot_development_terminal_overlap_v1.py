"""Derived illustration of the already diagnosed source overlap, not labels."""
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from _bootstrap import PROJECT_ROOT
from mtare_topo.governance_surface_material import read_pinned
from mtare_topo.data.primitive_relation_materialization import load_p1a_realized_construction
from mtare_topo.teacher.csg_mesh_provenance import mesh_swept_superellipse
from mtare_topo.teacher.gse_cap_return_evidence_v1 import source_endpoint_cap_faces
from mtare_topo.teacher.gse_mesh_sections_v1 import mesh_section


def main():
    out = PROJECT_ROOT / 'docs/figures/gse_development_terminal_overlap_v1'
    if out.exists():
        raise FileExistsError('preserve previous evidence')
    source = 'results/gate3_semantics/gate3_20260830_primitive_relation_p1a_lossless_storage_corrective_v1_seed0/artifacts/constructions/c07/S04_3d_unicyclic_small_C07__ellipse.json'
    expected = 'e1bdc1833da43d46dd9eaadf722517d71407f1e0fb512ac145864d5290f9783f'
    document = json.loads(read_pinned(PROJECT_ROOT, source, expected))
    _, primitives = load_p1a_realized_construction(document)
    lookup = {p.primitive_id: p for p in primitives}
    meshes = {key: mesh_swept_superellipse(lookup[key], axial_spacing_m=.05, angular_segments=64)
              for key in ('primitive:edge_0055', 'primitive:edge_0056')}
    origin = np.array([39.33273842476193, -54.71449656073344, 8.529901182326272])
    end = meshes['primitive:edge_0056']; other = meshes['primitive:edge_0055']
    cap_faces = source_endpoint_cap_faces(end, angular_segments=64, endpoint_index=0)
    cap = end.vertices_xyz_m[np.unique(end.triangle_vertex_indices[cap_faces])] - origin
    font = next((f.name for f in font_manager.fontManager.ttflist if 'Noto Sans CJK' in f.name), None)
    if font is None:
        raise RuntimeError('Chinese font missing')
    plt.rcParams.update({'font.family': font, 'axes.unicode_minus': False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    sections = []
    for ax, dims, normal in zip(axes, ((0, 1), (0, 2)), ((0, 0, 1), (0, 1, 0))):
        section = mesh_section(other.vertices_xyz_m, other.triangle_vertex_indices,
                               center_m=origin, normal=normal)
        sections.append([loop.tolist() for loop in section.loops_m])
        for i, loop in enumerate(section.loops_m):
            local = loop - origin
            ax.fill(local[:, dims[0]], local[:, dims[1]], color='#63a5d5', alpha=.3,
                    label='另一通道：通过参考中心的平面截面' if i == 0 else None)
            closed = np.vstack((local, local[:1]))
            ax.plot(closed[:, dims[0]], closed[:, dims[1]], color='#2171b5', lw=1.5)
        ax.scatter(cap[:, dims[0]], cap[:, dims[1]], s=15, c='#d94801',
                   label='自身参考端面顶点（投影，非检测结果）')
        ax.scatter([0], [0], s=100, marker='*', c='black', zorder=5, label='TNG参考末端中心')
        ax.set(xlim=(-7, 7), ylim=(-7, 7), aspect='equal', xlabel='相对X（米）',
               ylabel='相对' + ('Y' if dims[1] == 1 else 'Z') + '（米）')
        ax.grid(alpha=.2); ax.legend(fontsize=8, loc='lower left')
    fig.suptitle('为何构造末端不能直接当作可见端墙？\n'
                 'S04 C07 / node_0055 / ellipse：中心位于另一通道内部约0.273米', fontsize=13)
    out.mkdir(parents=True)
    target = out / 'source_overlap.png'; fig.savefig(target, dpi=150); plt.close(fig)
    (out / 'provenance.json').write_text(json.dumps(dict(source=source, source_sha256=expected,
        origin_world_m=origin.tolist(), sections_world_m=sections,
        image_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        selection='Previously diagnosed development unknown; explanatory case, not performance sample',
        method='Original .05m/64 source mesh plane intersection; red cap vertices are projected',
        labels=0, model_predictions=False, physical_safety_proven=False), ensure_ascii=False, indent=2)+'\n')
    print(target)


if __name__ == '__main__':
    main()
