"""Read only exact source chunks; validate ordering and report kinematics."""
import numpy as np
import zarr
from .gse_surface_input_export_v1 import _ExactStore


def inspect_arrays(arrays):
    """No guessed motion/safety tolerance, model inference, or graph update."""
    r,v=arrays['range_m'],arrays['valid_mask']
    if r.shape != (14,16,720) or v.shape != r.shape or np.any((v!=0)&(v!=1)):
        raise ValueError('14 source frames and binary validity required')
    valid=v.astype(bool)
    if not np.isfinite(r).all() or np.any(r[valid]<=0) or np.any(r[valid]>50):
        raise ValueError('finite ranges and valid first returns in(0,50] required')
    xyz,yaw,arc=(arrays[n] for n in ('sensor_xyz_m','yaw_deg','route_arc_m'))
    if xyz.shape != (14,3) or yaw.shape != (14,) or arc.shape != (14,) or not all(np.isfinite(a).all() for a in (xyz,yaw,arc)):
        raise ValueError('finite source poses/arcs required')
    local,t=arrays['local_frame_index'],arrays['traversal_index']
    if local.shape != (14,) or t.shape != (14,) or not np.all(np.diff(local)==1) or len(set(t.tolist()))!=1:
        raise ValueError('source crosses traversal or skips local frames')
    if not np.array_equal(arc[4:],np.arange(4.,14.)) or np.any(np.diff(arc)<=0):
        raise ValueError('source route arcs differ from fixed identity windows')
    return dict(valid_returns=int(valid.sum()),invalid_returns=int((~valid).sum()),
                local_frame_indices=local.tolist(),source_traversal_index=int(t[0]),
                route_arc_m=arc.tolist(),sensor_step_m=np.linalg.norm(np.diff(xyz,axis=0),axis=1).tolist(),
                yaw_step_deg=((np.diff(yaw)+180)%360-180).tolist(),
                physical_safety_verified=False,continuous_global_route=False)


def check_population(root,scope):
    opened={}; results=[]
    for entry in scope['entries']:
        arrays={}
        for a in entry['arrays']:
            store=_ExactStore(root,a['path'],scope['input_files_sha256'],opened)
            store.allowed={'.zarray',*a['chunk_keys']}
            array=zarr.Array(store,read_only=True)
            arrays[a['name']]=np.asarray(array[4012:4026])
            if set(store.read_counts)!={'.zarray',*a['chunk_keys']}:
                raise ValueError('unexpected source chunk consumption')
        results.append(dict(task=entry['task'],**inspect_arrays(arrays)))
    return dict(status='SOURCE_ORDER_AND_KINEMATIC_DIAGNOSTIC_COMPLETE',variants=results,
                observations=30,unique_variant_frames=42,optimizer_steps=0),opened
