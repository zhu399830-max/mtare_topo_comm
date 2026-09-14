"""Verify existing exported inputs only; no new data, labels, model or tuning."""
import hashlib
import io
import json
from pathlib import Path
import time
import numpy as np
from _bootstrap import PROJECT_ROOT as ROOT
from mtare_topo.data.multiview_joined_inputs_v1 import compile_join
from mtare_topo.data.gse_supplement_feature_input_v1 import bind_feature_input
from mtare_topo.evaluation.continuous_anchor_motion_v1 import compose_current_frames
from mtare_topo.governance_surface_material import read_pinned


if __name__=='__main__':
    output=ROOT/'docs/figures/gse_graph/multiview_joined_input_check_20260909'
    if output.exists():raise FileExistsError('no overwrite')
    scope=compile_join(ROOT);opened=dict(scope['metadata_sha256']);records=[];start=time.monotonic()
    for task in scope['tasks']:
        cache={};frames={};translations=[];yaws=[];source_frames=[]
        for row in task['observations']:
            path=row['input_path'];h=row['input_sha256'];s=row['source']
            if path not in cache:
                raw=read_pinned(ROOT,path,h);opened[path]=h
                with np.load(io.BytesIO(raw),allow_pickle=False) as data:
                    cache[path]=(raw,data['ranges_m'].copy(),data['valid_mask'].copy())
            raw,ranges,valid=cache[path]
            w=bind_feature_input(raw,expected_sha256=h,task=task['task'],row=row['input_row'],
                source_sequence_id=s['source_sequence_id'],frame_rows=s['frame_rows'],observation_count=row['observation_count'])
            translations.append(w.translation_m);yaws.append(w.yaw_deg);source_frames.append(s['frame_rows'])
            for j,f in enumerate(s['frame_rows']):
                values=(ranges[row['input_row'],j],valid[row['input_row'],j])
                if f in frames and any(not np.array_equal(a,b) for a,b in zip(frames[f],values)):
                    raise ValueError('cross-package raw frame mismatch: '+task['task'])
                frames[f]=values
        rotations,positions=compose_current_frames(translations,yaws,source_frames)
        records.append(dict(task=task['task'],split=task['split'],observations=len(source_frames),unique_frames=len(frames),
            packages=len(cache),step_m=np.linalg.norm(np.diff(positions,axis=0),axis=1).tolist(),
            first_frame_positions_m=positions.tolist(),first_frame_rotations=rotations.tolist(),
            ground_safety_verified=False,structure_labels_verified=False))
        if len(records)%30==0:print(json.dumps(dict(tasks=len(records),elapsed_s=time.monotonic()-start)),flush=True)
    for p,h in opened.items():
        if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h:raise ValueError('input drift')
    output.mkdir(parents=True)
    (output/'input_references.json').write_text(json.dumps(scope,ensure_ascii=False,allow_nan=False)+'\n')
    report=dict(status='INPUT_JOIN_SCAN_AND_RELATIVE_MOTION_CONSISTENT',observations=sum(r['observations'] for r in records),
        unique_variant_frames=sum(r['unique_frames'] for r in records),tasks=records,source_sha256=opened,
        elapsed_s=time.monotonic()-start,teacher_reads=0,model_calls=0,graph_pass=False)
    (output/'check.json').write_text(json.dumps(report,ensure_ascii=False,allow_nan=False)+'\n')
    files=[Path(__file__),ROOT/'src/mtare_topo/data/multiview_joined_inputs_v1.py',ROOT/'src/mtare_topo/evaluation/continuous_anchor_motion_v1.py']
    (output/'code_sha256.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})+'\n')
    seal=output/'evidence_sha256.txt';seal.write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(ROOT))+'\n' for p in sorted(output.iterdir()) if p.is_file() and p!=seal))
    print(json.dumps({k:v for k,v in report.items() if k not in ('tasks','source_sha256')}))
