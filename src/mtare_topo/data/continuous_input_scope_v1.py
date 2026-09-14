"""Metadata-only exact scope for the fixed single-traversal diagnostic."""
import hashlib
import json
import math
from pathlib import Path

from mtare_topo.governance_surface_material import read_pinned
from mtare_topo.governance_surface_input import SEALS
from mtare_topo.governance_identity_inventory import P1A

SELECTION = 'configs/v3/gate3/continuous_input_selection_v1.json'
PARENT = 'S09_flat_complex_C04'
VARIANTS = ('ellipse', 'rounded_rectangle', 'c1_mixed')
CONTRACT = {
    'range_m': ([4736,16,720], [16,16,720], '<f4'),
    'valid_mask': ([4736,16,720], [32,16,720], '|u1'),
    'sensor_xyz_m': ([4736,3], [4096,3], '<f8'),
    'yaw_deg': ([4736], [4096], '<f8'),
    'route_arc_m': ([4736], [4096], '<f8'),
    'local_frame_index': ([4736], [4096], '<i4'),
    'traversal_index': ([4736], [4096], '<i4'),
}


def compile_scope(root):
    root = Path(root).resolve()
    raw = (root/SELECTION).read_bytes()
    selection = json.loads(raw)
    if (selection['parent_id'] != PARENT or selection['split'] != 'fit'
            or selection['traversal_id'] != PARENT+':edge_0122:d1'
            or selection['variants'] != list(VARIANTS)
            or selection['unique_frame_rows_each_variant'] != list(range(4012,4026))
            or selection['sequence_rows_each_variant'] != list(range(3032,3042))
            or selection['five_frame_windows_each_variant'] != [list(range(i,i+5)) for i in range(4012,4022)]):
        raise ValueError('fixed metadata selection drift')
    document = json.loads(read_pinned(root, selection['metadata_path'], selection['metadata_sha256']))
    interval = [i for i in document['intervals'] if i['traversal_id'] == selection['traversal_id']]
    if len(interval) != 1: raise ValueError('unique traversal required')
    for record in interval[0]['variants']:
        if (record['frame_rows'] != selection['five_frame_windows_each_variant']
                or record['sequence_rows'] != selection['sequence_rows_each_variant']
                or record['source_sequence_ids'] != selection['source_sequence_ids_each_variant']
                or record['decision_arc_m'] != selection['decision_arc_m_each_variant']):
            raise ValueError('sealed interval identity differs')
    seal = SEALS['sensor']
    lines = read_pinned(root, seal['path'], seal['sha256']).decode().splitlines()
    sealed = {}
    for line in lines:
        h,p = line.split('  ',1)
        if p in sealed: raise ValueError('duplicate seal path')
        sealed[p] = h
    files = {}; entries = []; padded = 0
    for variant in VARIANTS:
        prefix = P1A+'/artifacts/dataset/fit/'+PARENT+'__'+variant+'.zarr'
        group = prefix+'/.zattrs'
        attrs = json.loads(read_pinned(root, group, sealed[group])); files[group] = sealed[group]
        if (attrs.get('parent_id') != PARENT or attrs.get('geometry_realization') != variant
                or attrs.get('partition') != 'fit' or attrs.get('sensor_shape') != [16,720]
                or attrs.get('maximum_range_m') != 50. or attrs.get('student_pose_input_forbidden') is not True):
            raise ValueError('sensor source contract drift')
        arrays = []
        for name,(shape,chunks,dtype) in CONTRACT.items():
            path = prefix+'/'+name; header = path+'/.zarray'
            h = json.loads(read_pinned(root,header,sealed[header])); files[header] = sealed[header]
            if (h.get('shape') != shape or h.get('chunks') != chunks or h.get('dtype') != dtype
                    or h.get('order') != 'C' or h.get('zarr_format') != 2
                    or h.get('dimension_separator','.') != '.'):
                raise ValueError('array header drift: '+name)
            keys = [str(i)+'.0'*(len(chunks)-1) for i in range(4012//chunks[0],4025//chunks[0]+1)]
            for key in keys:
                p=path+'/'+key
                if p not in sealed: raise ValueError('chunk missing from original seal')
                files[p] = sealed[p]  # No chunk payload read by scope compiler.
            nbytes = len(keys)*math.prod(chunks)*int(dtype[-1]); padded += nbytes
            arrays.append(dict(name=name,path=path,header=h,chunk_keys=keys,decoded_padded_bytes=nbytes))
        entries.append(dict(task=PARENT+'__'+variant,variant=variant,arrays=arrays))
    if padded != 6119424: raise ValueError('source budget drift')
    return dict(schema='continuous_input_scope_v1',selection=selection,
                selection_sha256=hashlib.sha256(raw).hexdigest(),source_seal=seal,
                input_files_sha256=files,entries=entries,
                counts=dict(parents=1,traversals=1,variants=3,observations=30,unique_variant_frames=42,
                            frame_references=150,decoded_padded_bytes=padded),
                restrictions=['fit_C04_only','absolute_pose_diagnostic_only','no_teacher_labels',
                              'no_model_or_training','not_safety_or_revisit_proof'])
