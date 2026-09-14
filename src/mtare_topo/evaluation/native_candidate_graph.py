"""Validate raw native multigraph output without pruning or semantic claims."""
import json
import math


def validate_native_graph(raw):
    def reject_constant(value): raise ValueError('nonfinite JSON constant')
    g=json.loads(raw,parse_constant=reject_constant)
    if g.get('scope')!='fiveframe_candidate_graph_not_verified': raise ValueError('scope drift')
    fields=('rays','below_min_ray_count','observed_esdf','allocated_unknown','hallucinated',
            'missing_tsdf_support','nodes','edges')
    if any(type(g.get(k)) is not int or g[k]<0 for k in fields): raise ValueError('invalid counts')
    if (g['rays']>57600 or g['below_min_ray_count']>g['rays'] or g['hallucinated']
            or g['missing_tsdf_support'] or g.get('audit_is_continuous_safety_proof') is not False
            or g.get('audit_spacing_m')!=.125): raise ValueError('observation contract drift')
    if len(g['vertices'])!=g['nodes'] or len(g['connections'])!=g['edges'] or len(g['edge_audit'])!=g['edges']:
        raise ValueError('graph arrays/counts disagree')
    ids=set()
    for v in g['vertices']:
        if type(v['id']) is not int or v['id'] in ids: raise ValueError('duplicate/invalid node')
        p=v['xyz']
        if len(p)!=3 or any(type(x) not in (int,float) or not math.isfinite(x) for x in p):
            raise ValueError('nonfinite node position')
        ids.add(v['id'])
    audit_ids=set()
    for edge,a in zip(g['connections'],g['edge_audit']):
        if (len(edge)!=2 or any(type(x) is not int or x not in ids for x in edge)
                or edge[0]==edge[1] or a['endpoints']!=edge): raise ValueError('edge/audit mismatch')
        if type(a['edge_id']) is not int or a['edge_id'] in audit_ids: raise ValueError('duplicate audit ID')
        audit_ids.add(a['edge_id'])
        if any(type(a[k]) is not int or a[k]<0 for k in ('samples','unknown_samples','below_0_4m_samples')):
            raise ValueError('invalid sampling counts')
        if a['samples']<2 or a['unknown_samples']+a['below_0_4m_samples']>a['samples']:
            raise ValueError('sampling count contradiction')
        m=a['minimum_sampled_esdf_m']
        if a['unknown_samples']==a['samples']:
            if m is not None: raise ValueError('unknown-only minimum must be null')
        elif type(m) not in (int,float) or not math.isfinite(m): raise ValueError('missing finite minimum')
        elif (m<.4)!=(a['below_0_4m_samples']>0): raise ValueError('clearance flag contradiction')
    # Parallel edges are legal native output and deliberately retained.
    return g
