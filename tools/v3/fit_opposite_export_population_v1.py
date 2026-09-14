"""Reuse original six-field encoding/binding; exact fit138 population."""
import json
from mtare_topo.data.traversal_input_packages_v1 import package_indices,slice_package
from mtare_topo.data.multiview_missing_input_export_v1 import SurfaceInputReader,encode_and_validate

def export_population(root,run,scope,opened):
    reader=SurfaceInputReader(root,task_sources=scope['task_sources'],selection=scope['missing'],
        sealed_keys=scope['input_files_sha256'],variable_population=True)
    output=run/'artifacts/inputs';output.mkdir();entries=[];total=0
    try:
        with (run/'logs/tasks.jsonl').open('x') as log:
            for task in sorted(scope['task_sources']):
                expected=sorted([r for r in scope['missing'] if r['task']==task],key=lambda r:r['sequence_row'])
                value=reader.read_task(task)
                for indices in package_indices(expected):
                    part,selected=slice_package(value,expected,indices)
                    payload,h=encode_and_validate(part,selected)
                    path=output/(task+'__part'+str(len(entries)).zfill(3)+'.npz')
                    with path.open('xb') as f:f.write(payload)
                    entry=dict(task=task,file=str(path.relative_to(run)),sha256=h,bytes=len(payload),observations=len(selected),
                        source_frames=part.frame_rows.tolist(),source_sequence_ids=part.source_sequence_ids.tolist(),
                        split=selected[0]['split'],traversal_id=selected[0]['traversal_id'],read_report=part.read_report)
                    entries.append(entry);total+=len(selected);log.write(json.dumps(entry)+'\n');log.flush()
                    if sum(p.stat().st_size for p in run.rglob('*') if p.is_file())>2*1024**3:raise OSError('2GiB output cap')
    finally:opened.update(reader.opened)
    if total!=138 or len(entries)!=13:raise ValueError('missing-only population incomplete')
    return dict(status='SIX_FIELD_INPUT_EXPORT_COMPLETE',observations=total,tasks=entries,reused_inputs=[], reuse_binding_pending=True,
                model_calls=0,optimizer_steps=0,graph_pass=False)

