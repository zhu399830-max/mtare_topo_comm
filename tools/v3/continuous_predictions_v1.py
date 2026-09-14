#!/usr/bin/env python3
"""Single immutable C0/C1 final-checkpoint continuous inference export."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import resource
import shlex
import signal
import sys
import time
import traceback
import zipfile
from _bootstrap import PROJECT_ROOT as ROOT
from continuous_features_v1 import PYTHON, environment, sha, write
from mtare_topo.governance import load_json, build_run_id
from mtare_topo.governance_surface_selection import digest
from mtare_topo.governance_surface_material import read_pinned
from mtare_topo.governance_continuous_predictions_v1 import SCHEMA, SLUG, POLICY, validate_card
from mtare_topo.data.continuous_prediction_scope_v1 import compile_scope

CARD='configs/v3/gate3/data_cards/'+SLUG+'.json'
SPEC='configs/v3/gate3/'+SLUG+'.json'


def freeze():
    if (ROOT/CARD).exists() or (ROOT/SPEC).exists():raise FileExistsError('no refreeze')
    scope=compile_scope(ROOT)
    approval=dict(status='APPROVED',approved_by='user-standing-scope-authorization',approved_at='2026-09-09',
        authorized_operations=['data_export'],authorized_gates=[3],scope_sha256=digest(scope),
        confirmation_reference='User standing autonomous authorization; PLAN permits exact30 continuous C04 windows with completed C0/C1 final weights; no training.',
        scope='One fit parent,one traversal,three variants,30 windows,60 predictions; no teacher,test,calibration or graph claims.')
    card=dict(schema_version=SCHEMA,card_id=SLUG,operation='data_export',scope=scope,scope_sha256=digest(scope),policy=POLICY,approval=approval)
    report=validate_card(card)
    if not report.passed:raise ValueError(report.errors)
    with (ROOT/CARD).open('x') as f:json.dump(card,f,ensure_ascii=False,indent=2)
    files=sorted({str(p.relative_to(ROOT)) for folder in ('src/mtare_topo','tools/v3') for p in (ROOT/folder).rglob('*.py')}|{CARD})
    run='results/gate3_semantics/gate3_20260909_'+SLUG+'_seed0'
    command=['env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','PYTHONHASHSEED=0',PYTHON,
        'tools/v3/continuous_predictions_v1.py','--spec',str(ROOT/SPEC),'--run-dir',str(ROOT/run)]
    spec=dict(schema_version='v3_run_spec_v1',gate=3,date='20260909',slug=SLUG,seed=0,operation='data_export',
        data_card=CARD,config_path=CARD,user_authorization=approval,command=command,
        question='What do unchanged final C0/C1 models predict across30 genuinely overlapping causal windows?',
        method='Original authenticated R2 compact points and partitions; original forward; both final2000update checkpoints; eval,no gradients; retain all logits and coordinates.',
        baseline='C0 versus C1 on identical windows; fit-only continuous diagnostic,not graph quality or generalization.',
        fallback='Fail and seal on input,environment,model,numeric or resource drift; no retry,filter,threshold tuning or replacement.',
        wall_time_cap_s=1800,estimated_cost=dict(compute='5090D sixty inference calls only',host_ram_gb=32,gpu_vram_gb=28,disk_gb=2,wall_time_hours=.5),
        acceptance_criteria=['Exact30 observations per final model,60 complete finite raw predictions and original >=0 logit DTOs.',
            'Weights unchanged,no teacher input,no optimization,no cross-traversal stitching.',
            '1800s,32GiBhost,28GiBGPU,2GiBoutput caps; input hashes,logs,source and evidence seal.'],
        expected_evidence=['60 raw NPZ predictions,60 observation DTOs,manifest,checkpoints hashes,environment,logs,summary,RUN_STATE,seal'],
        source_sha256={p:sha(ROOT/p) for p in files},environment=environment())
    with (ROOT/SPEC).open('x') as f:json.dump(spec,f,ensure_ascii=False,indent=2)
    print(json.dumps(dict(spec=SPEC,counts=scope['counts'],executed=False)))


def execute(spec,run):
    import numpy as np
    import torch
    from mtare_topo.data.development_compact_blocks import read_compact_points,bind_partition_payload
    from mtare_topo.data.development_partition_scope import ENCODER
    from mtare_topo.representation.frozen_relation_inference_v1 import load_final_model,predict_r2
    from mtare_topo.representation.block_relation_training_v1 import state_sha256
    from mtare_topo.topology.saved_anchor_branch_adapter_v1 import convert_saved_prediction
    run=run.resolve(strict=True)
    if (run!=ROOT/'results/gate3_semantics'/build_run_id(spec) or load_json(run/'RUN_STATE.json')['state']!='CREATED_NOT_EXECUTED'
            or load_json(run/'config/run_spec.json')!=spec):raise ValueError('fresh exact run required')
    entries=[];opened={};error=None;start=time.monotonic()
    def expired(*args):raise TimeoutError('1800s cap')
    old=signal.signal(signal.SIGALRM,expired);signal.alarm(1800)
    write(run/'RUN_STATE.json',dict(state='RUNNING',run_id=run.name))
    try:
        card=load_json(ROOT/spec['data_card'])
        if not validate_card(card).passed or card!=load_json(run/'config/data_card.json'):raise ValueError('card drift')
        if environment()!=spec['environment'] or Path(sys.executable)!=Path(PYTHON):raise ValueError('environment drift')
        if (run/'config/command.txt').read_text()!=shlex.join(spec['command'])+'\n':raise ValueError('command drift')
        with zipfile.ZipFile(run/'artifacts/source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
            for p,h in spec['source_sha256'].items():z.writestr(p,read_pinned(ROOT,p,h))
        scope=compile_scope(ROOT)
        if scope!=card['scope']:raise ValueError('scope drift')
        opened.update(scope['metadata_sha256'])
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')
        torch.manual_seed(0);torch.cuda.manual_seed_all(0)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        torch.cuda.set_per_process_memory_fraction(POLICY['gpu_bytes']/torch.cuda.get_device_properties(0).total_memory)
        write(run/'config/prediction_runtime.json',dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(0),**spec['environment']))
        def read(p,h):
            raw=read_pinned(ROOT,p,h);opened[p]=h;return raw
        out=run/'artifacts/predictions';out.mkdir()
        with (run/'logs/observations.jsonl').open('x') as log:
            for method,ck in scope['checkpoints'].items():
                model=load_final_model(read(ck['path'],ck['sha256']),expected_sha256=ck['sha256'],
                    initial_sha256=scope['initial_sha256'],relation_attributes=ck['relation_attributes'],device='cuda')
                before=state_sha256(model)
                for row in scope['observations']:
                    compact=read_compact_points(read(row['cache_path'],row['cache_sha256']),manifest_row=row['feature_entry'],
                        expected_source=row['source'],encoder_sha256=ENCODER)
                    representations=bind_partition_payload(compact,read(row['partition_path'],row['partition_sha256']),
                        expected_sha256=row['partition_sha256'],expected_source=row['source'])
                    arrays=predict_r2(model,representations['r2'])
                    source=row['source']; binding=row['feature_entry']['input_binding_sha256']
                    dto=convert_saved_prediction(arrays,stream_key=source['task'],decision_index=source['frame_rows'][-1],
                        frame_orders=source['frame_rows'],input_binding_sha256=binding)
                    name=method+'_'+digest(source);path=out/(name+'.npz')
                    with path.open('xb') as f:np.savez_compressed(f,**arrays)
                    write(out/(name+'.json'),asdict(dto))
                    entry=dict(method=method,source=source,input_binding_sha256=binding,file=str(path.relative_to(run)),sha256=sha(path),
                        selected_anchors=len(dto.anchors),model_state_sha256=before)
                    entries.append(entry);log.write(json.dumps(entry)+'\n');log.flush()
                    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>POLICY['host_ram_bytes'] or torch.cuda.max_memory_reserved()>POLICY['gpu_bytes']:
                        raise MemoryError('RAM/VRAM cap')
                    if sum(p.stat().st_size for p in run.rglob('*') if p.is_file())>POLICY['output_bytes']:raise RuntimeError('output cap')
                if state_sha256(model)!=before or any(p.grad is not None for p in model.parameters()):raise ValueError('model mutation')
                print(json.dumps(dict(method=method,completed=len(entries),elapsed_s=time.monotonic()-start)),flush=True)
                del model
        if len(entries)!=60:raise ValueError('incomplete predictions')
        for p,h in opened.items():
            if sha(ROOT/p)!=h:raise ValueError('input changed during inference')
    except Exception:
        error=traceback.format_exc();(run/'logs/error.log').write_text(error)
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,old)
        summary=dict(status='FAILED' if error else 'PREDICTION_EXPORT_COMPLETE',error=error,predictions=len(entries),
            elapsed_s=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0,optimizer_steps=0,scientific_gate_pass=False)
        write(run/'artifacts/prediction_manifest.json',dict(predictions=entries,source_reads=opened))
        write(run/'metrics/summary.json',summary);write(run/'RUN_STATE.json',dict(state='FAILED' if error else 'COMPLETED',run_id=run.name))
        seal=run/'artifacts/evidence_sha256.txt'
        seal.write_text(''.join(sha(p)+'  '+str(p.relative_to(ROOT))+'\n' for p in sorted(run.rglob('*')) if p.is_file() and p!=seal))
    print(json.dumps(summary),flush=True)
    return int(error is not None)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--freeze',action='store_true');parser.add_argument('--spec',type=Path);parser.add_argument('--run-dir',type=Path)
    args=parser.parse_args()
    if args.freeze:freeze()
    elif args.spec and args.run_dir:raise SystemExit(execute(load_json(args.spec),args.run_dir))
    else:parser.error('--freeze or --spec/--run-dir required')
