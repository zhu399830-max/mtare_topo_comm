"""Necessary teacher-only evidence across artificial degree-two cuts.

Not connected to V8, a label writer or a model. Construction continuity alone
never proves observed membership, free space or physical traversability.
"""
import numpy as np
from .gse_reference_continuations_v1 import reference_continuations
from .gse_terminal_window_component_v1 import terminal_window_component
from .gse_terminal_relation_witness_v1 import terminal_relation_witness


def recheck_saved_positive_origin(target, *, opening_index, terminal_index,
                                 source_ids, origin_distances, field_spacing_m):
    """Recheck only the changed containment prerequisite of an old positive.

    Caller independently pins target bytes, construction and source ordering.
    Original two-sided witness frames can be removed, never newly invented.
    No new membership label or complete-reference qualification is returned.
    """
    from mtare_topo.data.gse_structure_review_v1 import canonical_sha
    if field_spacing_m!=.025:raise ValueError('explicit common .025 field required')
    record=target['record'];proof=target['teacher_provenance']
    if canonical_sha(record)!=target['target_record_sha256']:raise ValueError('target hash mismatch')
    if any(type(i) is not int or i<0 for i in (opening_index,terminal_index)):raise ValueError('explicit target indices required')
    ai=proof['terminal_anchor_start']+terminal_index;oi=opening_index
    unchanged=dict(membership=None,complete_reference_qualified=False,retained_witness_frames=[])
    if record['membership'][oi][ai] is not True:
        return dict(unchanged,status='NOT_AN_EXISTING_POSITIVE')
    terminal=proof['terminals'][terminal_index];opening=proof['openings'][oi]
    identity=terminal['endpoint_key_teacher_only'][0]
    if opening['primitive_id_teacher_only']!=identity:raise ValueError('old positive source semantics changed')
    relation=proof['relations'][oi]
    if relation.get('anchor_index')!=ai or relation.get('opening_index')!=oi or relation.get('status')!='PARTIAL_TERMINAL_SOURCE_COMPONENT_CORRESPONDENCE':
        raise ValueError('old correspondence proof missing')
    frames=set()
    for index in relation['evidence_indices']:
        evidence=proof['terminal_relations'][index]
        if (evidence['anchor_index']!=ai or evidence['opening_index']!=oi
            or evidence['status']!='TWO_SIDED_SOURCE_WITNESS_ONLY' or evidence['blocked_ray_indices']
            or evidence['component']['status']!='REFERENCE_COMPONENT_ONLY'):
            raise ValueError('original observed reference prerequisite differs')
        for frame in evidence['common_frame_slots']:
            if type(frame) is not int or not 0<=frame<5:raise ValueError('invalid original witness frame')
            frames.add(frame)
    if not frames:raise ValueError('positive without original supported frames')
    ids=tuple(source_ids);d=np.asarray(origin_distances,dtype=float)
    if len(set(ids))!=len(ids) or identity not in ids or d.shape!=(5,len(ids)) or np.isnan(d).any() or np.isneginf(d).any():
        raise ValueError('complete ordered operand distances required')
    i=ids.index(identity)
    retained=[f for f in sorted(frames) if d[f,i]<0 and np.sum(d[f]<0)==1]
    return dict(unchanged,status='ORIGIN_PREREQUISITE_RETAINED' if retained else 'ORIGIN_PREREQUISITE_NOT_RETAINED',
        original_witness_frames=sorted(frames),retained_witness_frames=retained)


def terminal_continuation_component(sources, polylines, *, terminal_node, center_m):
    sources=tuple(sources)
    components=reference_continuations(sources)
    by_id={s.source_id:s for s in sources}
    if set(polylines)!=set(by_id):raise ValueError('complete realized source geometry required')
    lines={};incidence={}
    for source in sources:
        p=np.asarray(polylines[source.source_id],dtype=float)
        if (p.ndim!=2 or p.shape[1]!=3 or len(p)<2 or not np.isfinite(p).all()
            or np.any(np.linalg.norm(np.diff(p,axis=0),axis=1)==0)
            or not np.array_equal(p[[0,-1]],source.endpoint_xyz_m)):
            raise ValueError('finite realized polyline with exact bound endpoints required')
        lines[source.source_id]=p
        for side,node in enumerate(source.node_ids):incidence.setdefault(node,[]).append((source.source_id,side))
    unknown=dict(status='UNKNOWN',membership=None,physical_traversability=False)
    if len(incidence.get(terminal_node,[]))!=1:return dict(unknown,reason='NOT_A_TERMINAL_BOUNDARY')
    first=incidence[terminal_node][0]
    component=next(c for c in components if first[0] in c.source_ids)
    if component.unresolved_degree_two_nodes or component.closed_reference_cycle or len(component.structural_boundaries)!=2:
        return dict(unknown,reason='UNRESOLVED_OR_NON_SIMPLE_CONTINUATION')
    ordered=[];pieces=[];seen=set();current=first
    while True:
        identity,side=current
        if identity in seen:raise ValueError('unexpected source cycle')
        seen.add(identity);p=lines[identity] if side==0 else lines[identity][::-1]
        if pieces and not np.array_equal(pieces[-1][-1],p[0]):raise ValueError('no invented connector')
        pieces.append(p);ordered.append((identity,side))
        node=by_id[identity].node_ids[1-side]
        if len(incidence[node])!=2:break  # never cross a junction
        other=[x for x in incidence[node] if x!=(identity,1-side)]
        if len(other)!=1:raise ValueError('ambiguous incidence')
        current=other[0]
    if seen!=set(component.source_ids):raise ValueError('incomplete ordered continuation')
    offsets=[];offset=0.
    for (identity,side),p in zip(ordered,pieces,strict=True):
        length=float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum())
        offsets.append(dict(source_id=identity,entry_side=side,offset_m=offset,length_m=length))
        offset+=length
    joined=np.concatenate([pieces[0],*[p[1:] for p in pieces[1:]]])
    # A straight artificial cut exactly on the ROI must not introduce a new
    # ambiguous corner. Remove only EXACT collinear forward vertices, no snap.
    reduced=[]
    for point in joined:
        reduced.append(point)
        while len(reduced)>=3:
            a,b,c=reduced[-3:];u=b-a;v=c-b
            if np.all(np.cross(u,v)==0) and np.dot(u,v)>0:reduced.pop(-2)
            else:break
    result=terminal_window_component(np.asarray(reduced),endpoint_index=0,center_m=center_m)
    return dict(result,source_ids=list(component.source_ids),ordered_sources=ordered,source_arc_offsets=offsets,
        terminal_node_teacher_only=terminal_node,observation_support_verified=False)


def continuation_origin_witness(*, continuation_source_ids, source_ids, origin_distances, **witness):
    """Reuse the old two-sided ray test, with exact union containment columns.

    The source group must come from terminal_continuation_component. This is
    only an observation prerequisite: it cannot certify a label on its own.
    """
    group=tuple(continuation_source_ids);ids=tuple(source_ids)
    distances=np.asarray(origin_distances,dtype=float)
    if (not group or len(set(group))!=len(group) or len(set(ids))!=len(ids)
        or not set(group)<=set(ids) or distances.shape!=(5,len(ids))
        or np.isnan(distances).any() or np.isneginf(distances).any()):
        raise ValueError('complete finite-or-positive-infinite source distances and unique group required')
    others=[i for i,s in enumerate(ids) if s not in group]
    # Internal overlap between two pieces of the SAME validated continuation
    # is not an unrelated layer; any contained outside source remains a veto.
    union=np.min(distances[:,[ids.index(s) for s in group]],axis=1)
    merged=np.column_stack([union,distances[:,others]])
    synthetic='__teacher_continuation__'
    while synthetic in ids:synthetic+='_'  # metadata only, never a model input
    result=terminal_relation_witness(source_id=synthetic,source_ids=[synthetic,*[ids[i] for i in others]],
        origin_distances=merged,**witness)
    return dict(result,continuation_source_ids=list(group),membership=None,
        complete_correspondence_verified=False)
