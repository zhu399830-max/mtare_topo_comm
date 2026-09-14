"""Render existing synthetic unit-test outputs, not a research experiment."""
import html
import json
from pathlib import Path
import runpy

from mtare_topo.representation.gse_observed_connection_probe import local_connections


def main():
    root = Path(__file__).resolve().parents[2]
    fixture = runpy.run_path(str(root / "tests/v3/unit/test_observed_connection_probe.py"))["fixture"]
    target = root / "docs/figures/gse_conditional_geometry_fit_v1/connection_software_probe"
    target.mkdir(exist_ok=True)
    names = ["straight", "t_junction", "nearby_junctions", "stacked_crossing", "occluded", "ramp"]
    labels = ["直道", "T形连接", "相邻连接区域", "上下层交叉", "遮挡：证据缺失", "同一坡面"]
    rows = []
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="700" viewBox="0 0 1080 700">',
           '<rect width="1080" height="700" fill="white"/>',
           '<g font-family="sans-serif" fill="#152536">',
           '<text x="25" y="30" font-size="20">不训练的连接接口测试：点观测与面片组合结果相同</text>',
           '<text x="25" y="55" font-size="14">理想表面与显式连接见证；不是实际LiDAR检测，不是网络成绩，不是导航图</text>']
    for i, (name, label) in enumerate(zip(names, labels)):
        surfaces, witnesses, expected = fixture(name)
        point = local_connections(surfaces, witnesses)
        primitive = local_connections(surfaces, witnesses, representation="primitives")
        rows.append({"case": name, "point_route": point, "primitive_route": primitive,
                     "fixture_expected_links": expected, "equal": point == primitive})
        ox, oy = 25 + (i % 3)*350, 90 + (i // 3)*260
        svg.append(f'<text x="{ox}" y="{oy}" font-size="18">{label}</text>')
        positions = {}
        for s in surfaces:
            def project(p):
                return ox + 115 + p[0]*70 + p[1]*25, oy + 140 - p[1]*60 - p[2]*42
            polygon = [project(s.points[j]) for j in (0, 1, 3, 2)]
            coords = ' '.join(f'{x:.1f},{y:.1f}' for x, y in polygon)
            svg.append(f'<polygon points="{coords}" fill="#d8eaf3" stroke="#537b92"/>')
            x, y = project(s.points.mean(axis=0)); positions[s.key] = (x, y)
            svg.append(f'<text x="{x+3:.1f}" y="{y-7:.1f}" font-size="11">{html.escape(s.key)}</text>')
        for field, color, dash in (("local_evidence_links", "#167442", ""), ("unknown_pairs", "#bd6b15", 'stroke-dasharray="5 4"')):
            for a, b in primitive[field]:
                x1, y1 = positions[a]; x2, y2 = positions[b]
                svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="3" {dash}/>')
        svg.append(f'<text x="{ox}" y="{oy+205}" font-size="13">局部证据连接 {len(primitive["local_evidence_links"])}；未知 {len(primitive["unknown_pairs"])}；导航边 0</text>')
    svg.extend(['<text x="25" y="650" font-size="14">蓝色：合成表面输入　绿色：已有见证支持的局部连接　橙虚线：未知，不补边</text>',
                '<text x="25" y="676" font-size="14">表面身份是夹具索引，不是预测路口；邻接／共面本身不能证明机器人可通过。</text>', '</g></svg>'])
    (target / "preview.svg").write_text('\n'.join(svg), encoding="utf-8")
    (target / "outputs.json").write_text(json.dumps({"scope": "synthetic_software_test_preview",
        "training_steps": 0, "real_world_reads": 0, "geometry_advantage_proven": False,
        "bridge_extraction_tested": False, "cases": rows}, indent=2), encoding="utf-8")
    print(json.dumps({"cases": len(rows), "equal_routes": sum(r["equal"] for r in rows), "preview": str(target / "preview.svg")}))


if __name__ == "__main__":
    main()
