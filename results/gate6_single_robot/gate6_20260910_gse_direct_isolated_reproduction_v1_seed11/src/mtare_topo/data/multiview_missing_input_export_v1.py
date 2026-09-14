"""Missing-only export through the original student input reader and binder."""
import hashlib
import io
import json
import numpy as np
from .gse_surface_input_export_v1 import SurfaceInputReader
from .gse_supplement_feature_input_v1 import bind_feature_input
from .continuous_model_input_export_v1 import FIELDS


def encode_and_validate(value,expected):
    n=len(expected)
    if not 1<=n<=64 or value.frame_rows.tolist()!=[r['frame_rows'] for r in expected] or value.source_sequence_ids.tolist()!=[r['source_sequence_id'] for r in expected]:
        raise ValueError('exact scoped missing windows required')
    out=io.BytesIO();np.savez_compressed(out,**{k:getattr(value,k) for k in FIELDS})
    raw=out.getvalue();h=hashlib.sha256(raw).hexdigest();frames={}
    for i,row in enumerate(expected):
        bind_feature_input(raw,expected_sha256=h,task=value.task,row=i,source_sequence_id=row['source_sequence_id'],
                           frame_rows=row['frame_rows'],observation_count=n)
        # Missing-only windows may have gaps where an old input is reused.
        # Compare actual common frame IDs, never assume every pair is adjacent.
        for j,f in enumerate(row['frame_rows']):
            current=(value.ranges_m[i,j],value.valid_mask[i,j])
            if f in frames and any(not np.array_equal(a,b) for a,b in zip(frames[f],current)):
                raise ValueError('overlapping source frame differs')
            frames[f]=current
    return raw,h


def export_population(root,run,scope,opened):
    reader=SurfaceInputReader(root,task_sources=scope['task_sources'],selection=scope['missing'],
        sealed_keys=scope['input_files_sha256'],variable_population=True)
    output=run/'artifacts/inputs';output.mkdir();entries=[];total=0
    try:
        with (run/'logs/tasks.jsonl').open('x') as log:
            for task in sorted(scope['task_sources']):
                expected=sorted([r for r in scope['missing'] if r['task']==task],key=lambda r:r['sequence_row'])
                value=reader.read_task(task);payload,h=encode_and_validate(value,expected)
                path=output/(task+'.npz')
                with path.open('xb') as f:f.write(payload)
                entry=dict(task=task,file=str(path.relative_to(run)),sha256=h,bytes=len(payload),observations=len(expected),
                    source_frames=value.frame_rows.tolist(),source_sequence_ids=value.source_sequence_ids.tolist(),
                    split=expected[0]['split'],read_report=value.read_report)
                entries.append(entry);total+=len(expected);log.write(json.dumps(entry)+'\n');log.flush()
                if len(entries)%15==0:print(json.dumps(dict(tasks=len(entries),observations=total)),flush=True)
                if sum(p.stat().st_size for p in run.rglob('*') if p.is_file())>2*1024**3:raise OSError('2GiB output cap')
    finally:opened.update(reader.opened)
    if total!=2055 or len(entries)!=195:raise ValueError('missing-only population incomplete')
    return dict(status='SIX_FIELD_INPUT_EXPORT_COMPLETE',observations=total,tasks=entries,reused_inputs=scope['reuse'],
                model_calls=0,optimizer_steps=0,graph_pass=False)
