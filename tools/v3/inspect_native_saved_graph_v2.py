"""Read-only interpretation of the one sealed native output; no sensor access."""
from _bootstrap import PROJECT_ROOT as ROOT
import hashlib
import json
from collections import Counter
from mtare_topo.evaluation.native_candidate_graph_v2 import validate_native_graph

PATH='results/gate3_semantics/gate3_20260909_gse_native_graph_population_v1_seed0/artifacts/native_graphs/71a5f8b79e338a3ef5466b7ba323fbe276b3835cfbbccf2c9954f643ec382caf.json'
EXPECTED='38b90ecffdc20d45ed27c2558cbfd81fdd6df22ea9b618511121e5114b973699'


def inspect():
    raw=(ROOT/PATH).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=EXPECTED: raise ValueError('sealed graph drift')
    g=validate_native_graph(raw)
    adjacency={v['id']:set() for v in g['vertices']}
    pairs=Counter(tuple(sorted(e)) for e in g['connections'])
    for a,b in g['connections']:
        adjacency[a].add(b);adjacency[b].add(a)
    remaining=set(adjacency);sizes=[]
    while remaining:
        stack=[remaining.pop()];size=0
        while stack:
            node=stack.pop();size+=1
            new=adjacency[node]&remaining
            remaining.difference_update(new);stack.extend(new)
        sizes.append(size)
    audits=g['edge_audit']
    result=dict(status='SAVED_NATIVE_MULTIGRAPH_READABLE_NOT_SEMANTIC_OR_SAFETY_PASS',
        source=PATH,source_sha256=EXPECTED,nodes=g['nodes'],edges=g['edges'],
        self_loop_ids=[a['edge_id'] for a in audits if a['endpoints'][0]==a['endpoints'][1]],
        repeated_undirected_edges=sum(n-1 for n in pairs.values()),
        components=len(sizes),component_sizes=sorted(sizes,reverse=True),
        multigraph_cycle_rank=g['edges']-g['nodes']+len(sizes),
        unknown_sample_edges=sum(a['unknown_samples']>0 for a in audits),
        low_clearance_sample_edges=sum(a['below_0_4m_samples']>0 for a in audits),
        minimum_sampled_esdf_m=min((a['minimum_sampled_esdf_m'] for a in audits
            if a['minimum_sampled_esdf_m'] is not None),default=None),
        graph_changed=False,semantic_node_count=None,verified_traversal_count=None,
        sensor_reads=0,native_invocations=0)
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':inspect()
