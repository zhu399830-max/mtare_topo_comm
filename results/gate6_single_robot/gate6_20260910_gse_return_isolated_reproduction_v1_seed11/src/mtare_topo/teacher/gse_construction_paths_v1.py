"""Source-owned incident reference paths, retaining anchor/axis displacement.

Straight anchor-to-source-endpoint connectors are candidate observation probes,
NOT generated physical edges or certified free paths. No nearest projection,
radius adjustment, tunnel-ID collapse, or semantic labels are produced.
"""
import numpy as np


def construction_incident_paths(document):
    base=document.get("base_construction")
    if type(base) is not dict or base.get("coordinate_frame")!="world":
        raise ValueError("explicit world-frame source construction required")
    primitives=base.get("primitives");groups=base.get("compositions")
    if type(primitives) is not list or type(groups) is not list:
        raise ValueError("complete source primitive/composition lists required")
    lines={}
    for primitive in primitives:
        key=primitive["primitive_id"]
        line=np.asarray(primitive["centerline_xyz_m"],dtype=np.float64)
        if key in lines or type(key) is not str or not key:
            raise ValueError("duplicate or invalid source primitive identity")
        if line.ndim!=2 or line.shape[1:]!=(3,) or len(line)<2 or not np.isfinite(line).all() or np.any(np.linalg.norm(np.diff(line,axis=0),axis=1)==0):
            raise ValueError("invalid source reference polyline")
        lines[key]=line
    result=[];nodes=set();used=set()
    for group in groups:
        node=group["node_id"];anchor=np.asarray(group["anchor_xyz_m"],dtype=np.float64)
        if type(node) is not str or not node or node in nodes or anchor.shape!=(3,) or not np.isfinite(anchor).all():
            raise ValueError("invalid unique source anchor")
        nodes.add(node);members=group["member_endpoints"]
        if type(members) is not list or not members or type(group.get("degree")) is not int or group["degree"]!=len(members):
            raise ValueError("source degree/incidence mismatch")
        paths=[]
        for member in members:
            key=member["primitive_id"];side=member["endpoint_index"]
            if key not in lines or type(side) is not int or side not in (0,1) or (key,side) in used or member["node_id"]!=node:
                raise ValueError("invalid or multiply owned endpoint incidence")
            if not np.array_equal(member["composition_anchor_xyz_m"],anchor):
                raise ValueError("member/group anchor mismatch")
            line=lines[key] if side==0 else lines[key][::-1]
            if not np.array_equal(member["xyz_m"],line[0]):
                raise ValueError("source endpoint differs from its stored axis; no snapping")
            used.add((key,side))
            offset=float(np.linalg.norm(anchor-line[0]))
            # Exact same endpoint needs no zero-length connector. A nonzero
            # source displacement is preserved verbatim, even when tiny.
            points=line.copy() if offset==0 else np.vstack([anchor,line])
            points.setflags(write=False)
            paths.append(dict(endpoint_key=(key,side),points_world_m=points,
                anchor_axis_offset_m=offset,connector_is_observation_probe_only=True))
        result.append(dict(node_id_teacher_only=node,anchor_world_m=tuple(map(float,anchor)),paths=tuple(paths)))
    return tuple(result)
