"""Exact six-field existing model input for the fixed30 rolling windows."""
import json
from .continuous_input_scope_v1 import compile_scope as compile_sensor_scope, PARENT, VARIANTS
from .gse_surface_input_export_v1 import plan_array_access, SurfaceInputReader
from mtare_topo.governance_surface_input import FIELDS, SEALS
from mtare_topo.governance_identity_inventory import P1A, P1B
from mtare_topo.governance_surface_material import read_pinned


def compile_scope(root):
    bound=compile_sensor_scope(root); chosen=bound['selection']
    doc=json.loads(read_pinned(root,chosen['metadata_path'],chosen['metadata_sha256']))
    sealed={}
    for seal in SEALS.values():
        for line in read_pinned(root,seal['path'],seal['sha256']).decode().splitlines():
            h,p=line.split('  ',1)
            if p in sealed and sealed[p]!=h:raise ValueError('conflicting source seals')
            sealed[p]=h
    tasks={};selection=[];files={};plans={};padded=0
    for v in VARIANTS:
        task=PARENT+'__'+v
        tasks[task]=dict(parent_id=PARENT,variant=v,partition='fit',
            sensor=P1A+'/artifacts/dataset/fit/'+task+'.zarr',
            teacher=P1B+'/artifacts/teacher/fit/'+task+'.zarr')
        for i,frames in enumerate(chosen['five_frame_windows_each_variant']):
            selection.append(dict(task=task,parent_id=PARENT,variant=v,split='fit',
                sequence_row=chosen['sequence_rows_each_variant'][i],
                source_sequence_id=chosen['source_sequence_ids_each_variant'][i],frame_rows=frames,
                source_frame_count=doc['counts']['variants'][v]['frames'],
                source_sequence_count=doc['counts']['variants'][v]['sequences']))
        for role,names in FIELDS.items():
            prefix=tasks[task][role]
            for name in ('.zgroup','.zattrs'):
                p=prefix+'/'+name;read_pinned(root,p,sealed[p]);files[p]=sealed[p]
            rows=chosen['unique_frame_rows_each_variant'] if role=='sensor' else chosen['sequence_rows_each_variant']
            for name in names:
                path=prefix+'/'+name; p=path+'/.zarray'
                h=json.loads(read_pinned(root,p,sealed[p]));files[p]=sealed[p]
                plan=plan_array_access(h,name,role,rows);plans[path]=plan;padded+=plan['decoded_padded_bytes']
                for key in plan['chunk_keys']:
                    p=path+'/'+key
                    if p not in sealed:raise ValueError('missing chunk in original seal')
                    files[p]=sealed[p]
    # Validate all task/row contracts without opening any payload.
    SurfaceInputReader(root,task_sources=tasks,selection=selection,sealed_keys=files,variable_population=True)
    return dict(schema='continuous_model_input_scope_v1',selection_sha256=bound['selection_sha256'],
        source_seals=SEALS,task_sources=tasks,selection=selection,input_files_sha256=files,array_access=plans,
        counts=dict(parents=1,traversals=1,variants=3,observations=30,unique_variant_frames=42,
                    frame_references=150,decoded_padded_bytes=padded),
        restrictions=['only_existing_six_fields','no_absolute_pose_model_input','no_structure_labels',
                      'no_test_worlds','no_training','single_traversal_not_global_graph'])
