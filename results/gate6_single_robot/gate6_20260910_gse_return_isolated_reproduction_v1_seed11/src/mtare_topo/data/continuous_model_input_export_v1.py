"""Reuse original six-field reader and encoder binding; no feature inference."""
import hashlib
import io
import numpy as np
from .gse_surface_input_export_v1 import SurfaceInputReader
from .gse_supplement_feature_input_v1 import bind_feature_input

FIELDS=('ranges_m','valid_mask','relative_translation_current_sensor_m',
        'relative_yaw_current_sensor_deg','frame_rows','source_sequence_ids')


def encode_and_validate(task_input):
    out=io.BytesIO()
    np.savez_compressed(out,**{k:getattr(task_input,k) for k in FIELDS})
    payload=out.getvalue();h=hashlib.sha256(payload).hexdigest()
    if task_input.frame_rows.shape!=(10,5):raise ValueError('exact10 rolling windows per variant required')
    for i in range(10):
        bound=bind_feature_input(payload,expected_sha256=h,task=task_input.task,row=i,
            source_sequence_id=int(task_input.source_sequence_ids[i]),
            frame_rows=task_input.frame_rows[i].tolist(),observation_count=10)
        if (bound.range_valid.shape!=(5,2,16,720) or bound.translation_m.shape!=(5,3)
                or bound.yaw_deg.shape!=(5,)):raise ValueError('existing encoder tensor contract differs')
    # Shared frame bytes must be identical across consecutive windows; motion
    # fields use different current origins and are not expected to be equal.
    for field in ('ranges_m','valid_mask','frame_rows'):
        values=getattr(task_input,field)
        if not np.array_equal(values[:-1,1:],values[1:,:-1]):raise ValueError('overlap differs: '+field)
    return payload,h


def export_population(root,run,scope,opened):
    reader=SurfaceInputReader(root,task_sources=scope['task_sources'],selection=scope['selection'],
                              sealed_keys=scope['input_files_sha256'],variable_population=True)
    directory=run/'artifacts/inputs';directory.mkdir()
    rows=[]
    try:
        for task in sorted(scope['task_sources']):
            value=reader.read_task(task)
            payload,h=encode_and_validate(value)
            path=directory/(task+'.npz')
            with path.open('xb') as f:f.write(payload)
            rows.append(dict(task=task,file=str(path.relative_to(run)),sha256=h,bytes=len(payload),
                             observations=10,source_frames=value.frame_rows.tolist(),
                             source_sequence_ids=value.source_sequence_ids.tolist(),read_report=value.read_report))
    finally:
        opened.update(reader.opened)
    return dict(status='SIX_FIELD_INPUT_EXPORT_COMPLETE',observations=30,unique_variant_frames=42,
                tasks=rows,model_calls=0,optimizer_steps=0,graph_pass=False)
