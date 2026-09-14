"""Independent adjacency traversal of complete source endpoint inventory.

Shares the file decoder, not the producer's continuation grouping or labels.
Exact degree-two joins only; reference topology is not observed connectivity.
"""
from mtare_topo.data.primitive_relation_materialization import load_p1a_realized_construction
from mtare_topo.teacher.gse_construction_paths_v3 import construction_incident_paths
from mtare_topo.data.gse_structure_review_v1 import canonical_sha


def audit_reference_chains(target,document):
    if canonical_sha(document)!=target['source_binding']['construction_sha256']:
        raise ValueError('construction binding mismatch')
    _,primitives=load_p1a_realized_construction(document)
    points={(p.primitive_id,side):tuple(p.centerline_xyz_m[0 if side==0 else -1])
            for p in primitives for side in (0,1)}
    nodes={};owners={}
    for group in construction_incident_paths(document):
        node=group['node_id_teacher_only']
        for path in group['paths']:
            end=tuple(path['endpoint_key'])
            if end in owners or end not in points:raise ValueError('invalid endpoint inventory')
            owners[end]=node;nodes.setdefault(node,[]).append(end)
    if set(owners)!=set(points):raise ValueError('missing complete endpoint ownership')
    adjacent={p.primitive_id:set() for p in primitives};unresolved=set()
    for node,ends in nodes.items():
        if len(ends)!=2:continue
        a,b=ends
        if points[a]!=points[b]:unresolved.add(node);continue
        adjacent[a[0]].add(b[0]);adjacent[b[0]].add(a[0])
    def component(source):
        reached=set();queue=[source]
        while queue:
            p=queue.pop()
            if p in reached:continue
            reached.add(p);queue.extend(adjacent[p]-reached)
        boundary=[]
        for p in reached:
            for side in (0,1):
                node=owners[p,side]
                if node in unresolved:raise ValueError('candidate chain contains unresolved join')
                if len(nodes[node])!=2:boundary.append(node)
        return reached,sorted(boundary)
    provenance=target['teacher_provenance'];rows=[]
    for proof in provenance['terminal_nonmembership']:
        terminal=provenance['terminals'][proof['anchor_index']-provenance['terminal_anchor_start']]
        opening=provenance['openings'][proof['opening_index']]
        tc,tb=component(terminal['endpoint_key_teacher_only'][0]);oc,ob=component(opening['primitive_id_teacher_only'])
        tn=terminal['node_id_teacher_only'];jn=proof['junction_node_teacher_only']
        if (tc!=set(proof['terminal_continuation_sources']) or oc!=set(proof['opening_continuation_sources'])
                or tc&oc or tb!=sorted([tn,jn]) or jn not in ob or tn in ob):
            raise ValueError('saved exclusion contradicts independent complete reference chain')
        rows.append(dict(opening=proof['opening_index'],terminal_sources=len(tc),opening_sources=len(oc),
            terminal_boundaries=tb,opening_boundaries=ob,reference_only_not_physical_separation=True))
    return rows
