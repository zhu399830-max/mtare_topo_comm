"""Read-only multigraph analysis of the pinned synthetic native output.

No native edges are removed or changed. Simple-graph counts are diagnostics,
not a replacement score; no coordinate-dependent semantic node selection.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path

PATH = 'docs/figures/gse_graph/official_skeleton_synthetic_t_20260909/graph.json'
SHA = '5e43ae1edd39683320f9c5d3046bd4e61280df5178defa73dddf729194995ef1'


def components(vertices, edges, removed=None):
    remaining = set(vertices) - ({removed} if removed is not None else set())
    adjacency = {v: set() for v in remaining}
    for a, b in edges:
        if a in remaining and b in remaining:
            adjacency[a].add(b); adjacency[b].add(a)
    groups = []
    while remaining:
        todo = [min(remaining)]; group = set()
        while todo:
            v = todo.pop()
            if v in group: continue
            group.add(v); todo.extend(adjacency[v]-group)
        remaining -= group; groups.append(sorted(group))
    return groups


def analyze(graph):
    vertices = {v['id']: v for v in graph['vertices']}
    edges = graph['connections']
    if len(vertices) != len(graph['vertices']): raise ValueError('duplicate vertex ID')
    if any(a not in vertices or b not in vertices or a == b for a,b in edges):
        raise ValueError('invalid edge endpoint')
    counts = Counter(tuple(sorted(e)) for e in edges)
    degrees = Counter(v for e in edges for v in e)
    if any(degrees[v] != vertices[v]['degree'] for v in vertices):
        raise ValueError('saved degree disagrees with raw edges')
    base = components(vertices, edges)
    articulation = []
    for v in sorted(vertices):
        groups = components(vertices, edges, v)
        if len(groups) <= len(base): continue
        rows = []
        for group in groups:
            members = set(group)
            internal = [e for e in edges if all(x in members for x in e)]
            rows.append(dict(vertices=group, edges=len(internal),
                             cycle_rank=len(internal)-len(group)+1))
        articulation.append(dict(removed_vertex=v, remaining_components=rows))
    return dict(source_path=PATH, source_sha256=SHA, vertices=len(vertices), raw_edges=len(edges),
        unique_undirected_edges=len(counts), components=len(base),
        raw_multigraph_cycle_rank=len(edges)-len(vertices)+len(base),
        simple_graph_cycle_rank=len(counts)-len(vertices)+len(base),
        parallel_edges=[dict(endpoints=list(e), multiplicity=n) for e,n in sorted(counts.items()) if n>1],
        articulation=articulation, original_graph_modified=False,
        semantic_success=False, scope='synthetic_native_graph_structure_diagnostic')


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]
    raw = (root/PATH).read_bytes()
    if hashlib.sha256(raw).hexdigest() != SHA: raise ValueError('source graph drift')
    print(json.dumps(analyze(json.loads(raw)), indent=2))
