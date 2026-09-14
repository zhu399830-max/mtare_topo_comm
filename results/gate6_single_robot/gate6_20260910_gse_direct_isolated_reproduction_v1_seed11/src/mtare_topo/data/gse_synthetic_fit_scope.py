"""Exact archived synthetic fitting scope; no sensor/checkpoint deserialization.

Selection uses declared oracle coverage, not producer/model scores. This is not
the real-world 32-example gate and must never authorize real-world training.
"""
import gzip
import hashlib
import json
from pathlib import Path
from .gse_synthetic_matrix import matrix
from .gse_structure_review_v1 import canonical_sha

SOURCE='results/gate3_semantics/gate3_20260908_gse_synthetic_matrix_v1_seed20260906'
SEAL_SHA='5d491bd1c6c89caeb47ea416eb135fee8a1d8a3507f70d95256cb55a91d9a395'
CHECKPOINT='results/gate3_semantics/gate3_20260902_primitive_relation_observable_three_seed_training_v1_seed0/artifacts/models/seed0/selected.pt'
CHECKPOINT_SHA='8d5d2e0ec9779c28b068d0c7db80aeed38ebdb22dff5b781c26234d9001287fb'


def declared_cases():
    return [c for c in matrix() if c['program']['type'] in ('straight','terminal','visible_blocker')
            or (c['program']['type'] in ('T','Y','four_way') and c['case_id'].endswith('view2'))]


def compile_scope(root):
    root=Path(root)
    seal_path=SOURCE+'/artifacts/evidence_sha256.txt'
    seal=(root/seal_path).read_bytes()
    if hashlib.sha256(seal).hexdigest()!=SEAL_SHA: raise ValueError('archived seal drift')
    pins={p:h for h,p in (line.split('  ',1) for line in seal.decode().splitlines())}
    rows=[];inputs={seal_path:SEAL_SHA};units=set();frame_keys=set()
    for case in declared_cases():
        key=case['case_id']; record_path=SOURCE+'/artifacts/'+key+'.json.gz'
        payload=(root/record_path).read_bytes()
        if hashlib.sha256(payload).hexdigest()!=pins[record_path]:raise ValueError('archived declaration drift')
        record=json.loads(gzip.decompress(payload))
        if canonical_sha(record['case'])!=canonical_sha(case):raise ValueError('case declaration differs')
        frames=record['source']['frame_rows']
        if len(frames)!=5 or sorted(set(frames))!=frames:raise ValueError('five causal source rows required')
        input_path=SOURCE+'/artifacts/'+key+'.npz'
        if not (root/input_path).is_file():raise FileNotFoundError(input_path)
        # Sensor/checkpoint bytes are verified immediately before use by runner.
        inputs[record_path]=pins[record_path];inputs[input_path]=pins[input_path]
        rows.append(dict(case_id=key,case_sha256=canonical_sha(case),frame_rows=frames,
                         input_path=input_path,input_sha256=pins[input_path],
                         source_record_path=record_path,source_record_sha256=pins[record_path]))
        units.add(case['program']['type'])
        frame_keys.update((input_path,f) for f in frames)
    if len(rows)!=45 or len(units)!=6 or len(frame_keys)!=225:raise ValueError('exact fixture scope changed')
    inputs[CHECKPOINT]=CHECKPOINT_SHA
    return dict(schema_version='gse_synthetic_fit_scope_v1',observations=rows,input_sha256=inputs,
        primary_observations=45,frame_occurrences=225,unique_archive_frame_keys=225,
        independent_program_types=6,section_variants=3,real_worlds=0,ray_positions=2592000,
        history_spacing_m=.1,view_offsets_m=[-4.,-2.,0.,2.],
        sampling='All declared straight/terminal/blocker views and exact T/Y/four-way central views; no score selection.',
        split='Synthetic interface fitting only; no held-out/generalization claim, no real-world reads.',
        teacher='Independent declared convex-tube/star geometry oracle; no production targets or physical reachability.',
        unknown_fields=['dimensions','reachability','membership'],
        checkpoint=dict(path=CHECKPOINT,sha256=CHECKPOINT_SHA,seed=0,epoch=2),
        model_inputs=['ranges_m','valid_mask','relative_translation_current_sensor_m','relative_yaw_current_sensor_deg'],
        forbidden_forward_inputs=['case_id','construction','codebook','primitive_membership_code','oracle_targets'],
        methods=['A','B','C'],updates_per_method=500,seed=0,microbatch=1,accumulation_steps=4,
        optimizer=dict(name='AdamW',learning_rate=.001,weight_decay=.0001),
        evaluation='Initial and final only, all45; no best checkpoint selection. Fit only, not comparative research gain.',
        acceptance=dict(anchor_f1=.90,opening_f1=.90,anchor_match_radius_m=1.,opening_match_radius_m=1.),
        no_real_training_gate_advance=True,no_retry=True)
