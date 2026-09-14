"""Teacher-side finite-ray witnesses of oriented construction cross-sections.

These are partial, construction-conditioned references, not traversability,
complete task labels, or an independently validated teacher. No model score,
source-code equality, nearest-angle assignment or hidden free-space flood fill.
"""
import numpy as np
from mtare_topo.data.primitive_relation_materialization import load_p1a_realized_construction


def outward_section(line, half_axes, origin, radius=10.0):
    """Unique outgoing source-axis crossing of the existing local sphere.

    No anchor connector is added. Multiple crossings, tangent crossings or an
    axis starting outside the sphere are unavailable, not repaired. The disk
    is inscribed in the superellipse (source exponents are >=2).
    """
    line=np.asarray(line,float);axes=np.asarray(half_axes,float);origin=np.asarray(origin,float)
    if line.ndim!=2 or line.shape[1]!=3 or len(line)<2 or axes.shape!=(2,2) or origin.shape!=(3,):raise ValueError('shape')
    if not all(np.isfinite(v).all() for v in (line,axes,origin)) or np.any(axes<=0) or radius<=0:raise ValueError('finite positive geometry')
    if np.linalg.norm(line[0]-origin)>=radius:return None
    d=np.diff(line,axis=0);lengths=np.linalg.norm(d,axis=1)
    if np.any(lengths==0):raise ValueError('zero source segment')
    arc=np.r_[0,np.cumsum(lengths)];crossings=[]
    for i,delta in enumerate(d):
        rel=line[i]-origin;a=float(delta@delta);b=2*float(rel@delta);c=float(rel@rel-radius**2)
        disc=b*b-4*a*c
        if disc<=0:continue
        for t in ((-b-np.sqrt(disc))/(2*a),(-b+np.sqrt(disc))/(2*a)):
            # Half-open interval avoids counting the same vertex twice.
            if not 0<t<=1:continue
            center=line[i]+t*delta
            crossings.append((i,t,center,float((center-origin)@delta)))
    if len(crossings)!=1 or crossings[0][3]<=0:return None
    i,t,center,_=crossings[0];fraction=(arc[i]+t*lengths[i])/arc[-1]
    interpolated=axes[0]+fraction*(axes[1]-axes[0])
    return dict(center_world_m=center.tolist(),normal_world=(d[i]/lengths[i]).tolist(),
                inscribed_radius_m=float(min(interpolated)),source_segment=i,source_fraction=float(fraction))


def construction_sections(document, sensor_world, radius=10.0):
    graph,realized=load_p1a_realized_construction(document)
    primitives={p.primitive_id:p for p in realized};sections=[];unavailable=[]
    for group in graph.compositions:
        if len(group.member_endpoints)<3 or np.linalg.norm(np.asarray(group.anchor_xyz_m)-sensor_world)>=radius:continue
        for end in group.member_endpoints:
            p=primitives[end.primitive_id];line=p.centerline_xyz_m;axes=np.asarray(p.endpoint_half_axes_m)
            if end.endpoint_index==1:line=line[::-1];axes=axes[::-1]
            key=f'{group.node_id}/{end.primitive_id}/endpoint{end.endpoint_index}'
            section=outward_section(line,axes,sensor_world,radius)
            if section is None:unavailable.append(key)
            else:sections.append(dict(port_reference=key,node_reference=group.node_id,**section))
    return sections,unavailable


def crossed_sections(origins, endpoints, sections):
    """Strictly before finite first return; a surface hit is not free space."""
    origins=np.asarray(origins,float);endpoints=np.asarray(endpoints,float)
    if origins.ndim!=2 or origins.shape[1]!=3 or endpoints.shape!=origins.shape:raise ValueError('ray shape')
    if not np.isfinite(origins).all() or not np.isfinite(endpoints).all():raise ValueError('finite rays')
    delta=endpoints-origins;hits=np.zeros((len(origins),len(sections)),bool)
    for j,section in enumerate(sections):
        center=np.asarray(section['center_world_m']);normal=np.asarray(section['normal_world']);radius=section['inscribed_radius_m']
        if not np.isclose(np.linalg.norm(normal),1) or radius<=0:raise ValueError('section normal/radius')
        denom=delta@normal;numerator=(center-origins)@normal
        ratio=np.divide(numerator,denom,out=np.zeros_like(denom),where=denom>0)
        point=origins+ratio[:,None]*delta
        hits[:,j]=(denom>0)&(ratio>0)&(ratio<1)&(np.linalg.norm(point-center,axis=1)<radius)
    return hits


def candidate_witnesses(ray_indices, hits, sections):
    selected=np.asarray(ray_indices,dtype=int)
    if selected.ndim!=1 or np.any(selected<0) or np.any(selected>=len(hits)):raise ValueError('ray indices')
    counts=hits[selected].sum(axis=0);active=np.flatnonzero(counts)
    return dict(conditional_port_reference=sections[active[0]]['port_reference'] if len(active)==1 else None,
        witnessed_ports=[dict(port_reference=sections[i]['port_reference'],ray_count=int(counts[i])) for i in active],
        reason='UNIQUE_REFERENCE_SECTION_WITNESS' if len(active)==1 else ('COMPETING_SECTIONS' if len(active)>1 else 'NO_SECTION_WITNESS'),
        training_qualified=False,physical_passage_verified=False,reference_complete=False)
