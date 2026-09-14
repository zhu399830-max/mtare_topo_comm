"""Independent archival witness checks, not label or reachability certification.

Does not invoke any joint/anchor/opening teacher producer. Checks actual stored
returns against candidate ray sets; geometric visibility and reference quality
must still be established separately.
"""
import numpy as np
from mtare_topo.data.gse_structure_review_v1 import canonical_sha


def audit_saved_negative_witnesses(target, bundle):
    record=target['record'];provenance=target['teacher_provenance']
    if canonical_sha(record)!=target['target_record_sha256']:
        raise ValueError('target record hash mismatch')
    source=bundle['source'];student=bundle['student'];book=bundle['codebook_teacher_only']
    if target['source_binding']['source']!={k:source[k] for k in ('task','source_sequence_id','frame_rows')}:
        raise ValueError('source mismatch')
    valid=student['valid_mask'].reshape(-1).astype(bool)
    ranges=student['ranges_m'].reshape(-1)
    codes=bundle['sensor_teacher_only']['primitive_membership_code'].reshape(-1)
    if valid.shape!=(57600,) or ranges.shape!=valid.shape or codes.shape!=valid.shape:
        raise ValueError('exact five-frame source arrays required')
    def rays(values):
        if type(values) is not list or any(type(v) is not int or not 0<=v<57600 for v in values):
            raise ValueError('ray identity out of bounds')
        if len(values)!=len(set(values)):
            raise ValueError('duplicate witness rays')
        if not valid[values].all():
            raise ValueError('witness includes invalid first return')
        return set(values)
    def owner(ray):
        return [book['primitive_ids'][i] for i in book['source_sets'][int(codes[ray])]]
    expected={(i,j) for i,row in enumerate(record['membership']) for j,v in enumerate(row) if v is False}
    seen=set();rows=[]
    for proof in provenance['terminal_nonmembership']:
        oi,ai=proof['opening_index'],proof['anchor_index']
        if (oi,ai) not in expected or (oi,ai) in seen:
            raise ValueError('negative witness index mismatch or duplicate')
        seen.add((oi,ai))
        ti=ai-provenance['terminal_anchor_start']
        if not 0<=ti<len(provenance['terminals']):
            raise ValueError('terminal index mismatch')
        terminal=provenance['terminals'][ti];terminal_source=terminal['endpoint_key_teacher_only'][0]
        cap=rays([w['ray_index'] for w in terminal['witnesses']])
        for witness in terminal['witnesses']:
            ray=witness['ray_index']
            if (owner(ray)!=[terminal_source] or float(ranges[ray])!=witness['stored_t']
                    or source['frame_rows'][ray//11520]!=witness['source_frame_index']):
                raise ValueError('terminal witness source/range/frame mismatch')
        entering=rays(proof['cap_entering_ray_indices']);inside=rays(proof['cap_source_interior_ray_indices'])
        if not entering<=cap or not inside<=cap:
            raise ValueError('claimed cap witness absent from terminal evidence')
        relation=provenance['relations'][oi];ji=relation.get('anchor_index')
        if type(ji) is not int or not 0<=ji<provenance['terminal_anchor_start']:
            raise ValueError('opening lacks positive junction relation')
        if (record['membership'][oi][ji] is not True or
                provenance['anchors'][ji]['node_id_teacher_only']!=proof['junction_node_teacher_only']):
            raise ValueError('positive junction identity mismatch')
        opening=rays(proof['opening_junction_ray_indices'])
        if opening!=rays(relation.get('ray_indices',[])) or not opening:
            raise ValueError('opening relation witness mismatch')
        if not opening<=rays(provenance['openings'][oi]['crossing_ray_indices']):
            raise ValueError('junction witness does not cross this opening')
        outgoing=rays(proof['terminal_outgoing_opening_ray_indices'])
        if not outgoing<=opening or not (entering or (inside and outgoing)):
            raise ValueError('missing cap/transition witness')
        terminal_sources=set(proof['terminal_continuation_sources'])
        opening_sources=set(proof['opening_continuation_sources'])
        if terminal_source not in terminal_sources or terminal_sources & opening_sources:
            raise ValueError('continuation exclusion contradicts source lists')
        if any(len(owner(r))!=1 or owner(r)[0] not in opening_sources for r in opening):
            raise ValueError('opening witness ownership mismatch')
        rows.append(dict(opening_index=oi,anchor_index=ai,cap_rays=len(cap),
            cap_entering=len(entering),cap_inside=len(inside),opening_rays=len(opening),
            transition_rays=len(outgoing),witness_frames=sorted({r//11520 for r in opening}),
            status='ARCHIVED_RAY_SOURCE_CONSISTENT_NOT_LABEL_QUALIFICATION'))
    if seen!=expected:
        raise ValueError('negative label without archived evidence')
    return dict(source=source,checked_negative_candidates=len(rows),candidates=rows,
        first_return_binding_checked=True,geometric_surface_intersection_rechecked=False,
        reference_continuity_independently_rechecked=False,full_label_qualification=False)
