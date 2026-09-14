"""One sealed CPU-only extraction of both representations on identical45 inputs."""
import _bootstrap
import argparse
from dataclasses import fields
import html
import io
import json
import resource
import shlex
import signal
import time
import traceback
import zipfile
from pathlib import Path
import numpy as np
import libcp
import libply_c
from mtare_topo.governance import load_json,build_run_id
from mtare_topo.governance_spg_extraction import scope,validate_card,SCHEMA,SLUG
from mtare_topo.governance_surface_selection import digest
from mtare_topo.governance_surface_material import read_pinned
from mtare_topo.representation.gse_spg_runtime import sha,runtime_snapshot
from mtare_topo.representation.gse_paired_partitions import registered_returns,paired_partitions

ROOT=Path(__file__).resolve().parents[2]
CARD='configs/v3/gate3/data_cards/'+SLUG+'.json'
SPEC='configs/v3/gate3/'+SLUG+'.json'
RUNTIME='configs/v3/environments/gse_spg_runtime_v1.json'
RUN='results/gate3_semantics/gate3_20260908_'+SLUG+'_seed0'
PREFIX='build/gse_spg_cpu_v1'


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,sort_keys=True)+'\n')


def freeze():
    if any((ROOT/p).exists() for p in (CARD,SPEC,RUNTIME)):
        raise FileExistsError('no refreeze')
    s=scope(ROOT)
    a=dict(status='APPROVED',approved_by='user-standing-scope-authorization',approved_at='2026-09-08',
        scope='Exact same45 synthetic CPU partition extraction, no training or teacher generation',scope_sha256=digest(s),
        authorized_operations=['audit'],authorized_gates=[3],
        confirmation_reference='User: 全自动执行; all routine actions authorized; PLAN same45 R1/R2 extraction next step')
    card=dict(schema_version=SCHEMA,card_id=SLUG,operation='audit',scope=s,scope_sha256=digest(s),approval=a)
    if not validate_card(card).passed: raise ValueError('card invalid')
    runtime=runtime_snapshot(ROOT)
    for path,value in ((CARD,card),(RUNTIME,runtime)):
        with (ROOT/path).open('x') as f: json.dump(value,f,ensure_ascii=False,indent=2)
    files=sorted({str(p.relative_to(ROOT)) for folder in ('src/mtare_topo','tools/v3') for p in (ROOT/folder).rglob('*.py')}|{CARD,RUNTIME,'configs/v3/environments/gse_spg_cpu_build_v1.json'})
    command=['env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','PYTHONHASHSEED=0',
        'PYTHONPATH='+PREFIX+':'+PREFIX+'/prefix/usr/lib/python3/dist-packages',
        'LD_LIBRARY_PATH='+PREFIX+'/prefix/usr/lib/x86_64-linux-gnu','/usr/bin/python3',
        'tools/v3/spg_paired_extraction.py','--spec',str(ROOT/SPEC),'--run-dir',str(ROOT/RUN)]
    spec=dict(schema_version='v3_run_spec_v1',gate=3,date='20260908',slug=SLUG,seed=0,operation='audit',
        data_card=CARD,config_path=CARD,user_authorization=a,command=command,
        question='Can R1/R2 retain identical causal ROI returns with complete mappings and common finite attributes?',
        method=s['method'],baseline='R1 recalculated from shared float32 inputs; no reuse of historical detection scores',
        fallback=s['fallback'],estimated_cost=dict(compute='CPU45 paired extractions; no GPU, model or optimizer',
            host_ram_gb=4,gpu_vram_gb=0,disk_gb=2,wall_time_hours=1),
        acceptance_criteria=[s['acceptance'],'No teacher/model input; zero research Gate advancement','1hour/4GiBprocess/2GiB output limits; no overwrite or rerun'],
        expected_evidence=['45 raw mappings and common attributes,raw/connectedR2,XY/XZ previews,percase runtime/counts,logs,snapshot,environment,seal'],
        source_sha256={p:sha(ROOT/p) for p in files},environment_manifest=RUNTIME,environment_sha256=sha(ROOT/RUNTIME))
    with (ROOT/SPEC).open('x') as f: json.dump(spec,f,indent=2)
    print(json.dumps(dict(spec=SPEC,observations=45,frames=225,runtime_files=len(runtime['files']),scope_sha256=digest(s))),flush=True)


def preview(points,groups,axis):
    # Rendering decimation only: all raw points and memberships remain in NPZ.
    stride=max(1,(len(points)+1499)//1500)
    pieces=['<svg viewBox="0 0 320 320" width="320" height="320" style="background:#101820">']
    for p,g in zip(points[::stride],groups[::stride]):
        x=160+15*float(p[0]);y=160-15*float(p[axis]);hue=(int(g)*137)%360
        pieces.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="1" fill="hsl({hue},75%,65%)"/>')
    pieces.append('</svg>');return ''.join(pieces)


def execute(spec,run):
    run=run.resolve(strict=True)
    if run!=ROOT/'results/gate3_semantics'/build_run_id(spec) or load_json(run/'RUN_STATE.json')['state']!='CREATED_NOT_EXECUTED':
        raise ValueError('fresh exact run required')
    if load_json(run/'config/run_spec.json')!=spec: raise ValueError('spec drift')
    started=time.monotonic();rows=[];error=None
    write(run/'RUN_STATE.json',dict(state='RUNNING',run_id=run.name))
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('1hour cap')));signal.alarm(3600)
    # Native memory allocations also count toward this hard process bound.
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
    try:
        card=load_json(ROOT/spec['data_card'])
        if not validate_card(card).passed or card!=load_json(run/'config/data_card.json'):raise ValueError('card drift')
        runtime=load_json(ROOT/spec['environment_manifest'])
        if sha(ROOT/spec['environment_manifest'])!=spec['environment_sha256'] or runtime_snapshot(ROOT)!=runtime:
            raise ValueError('runtime drift')
        if (run/'config/command.txt').read_text()!=shlex.join(spec['command'])+'\n':raise ValueError('command drift')
        with zipfile.ZipFile(run/'artifacts/source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
            for p,h in spec['source_sha256'].items():z.writestr(p,read_pinned(ROOT,p,h))
        write(run/'config/environment.json',runtime)
        output=run/'artifacts/partitions';output.mkdir()
        with (run/'logs/observations.jsonl').open('x') as log,(run/'previews/partitions.html').open('x') as page:
            page.write('<!doctype html><meta charset="utf-8"><title>同输入几何分块对照</title><h1>45例同输入几何分块对照</h1><p>左：固定体素面片；右：几何超点。每组依次XY、XZ。颜色仅标分块，不是结构语义、通行或图节点。预览最多1500点，完整点与映射保存在NPZ；不同方法颜色不表示对应。</p>')
            for r in card['scope']['observations']:
                tick=time.monotonic()
                raw=read_pinned(ROOT,r['input_path'],r['input_sha256'])
                with np.load(io.BytesIO(raw),allow_pickle=False) as p:
                    points,frames,indices=registered_returns(p['ranges_m'],p['valid_mask'],p['relative_translation_current_sensor_m'],p['relative_yaw_current_sensor_deg'])
                a,b,r2=paired_partitions(points,frames,geof_backend=libply_c.compute_geof,partition_backend=libcp.cutpursuit)
                payload=dict(points_xyz_m=points,frame_index=frames,source_flat_ray_index=indices,
                    r2_voxel_centers_m=r2['voxels'].centers_m,r2_point_to_voxel=r2['voxels'].point_to_voxel,
                    r2_voxel_frame_counts=r2['voxels'].frame_counts)
                for label,value in (('r1',a),('r2',b)):
                    if value is not None:
                        payload.update({label+'_'+f.name:getattr(value,f.name) for f in fields(value)})
                if b is not None:
                    payload.update(r2_raw_assignment=r2['raw_assignment'],r2_voxel_to_connected=r2['connected'].point_to_component,
                        r2_connected_to_raw=r2['connected'].component_to_raw)
                with (output/(r['case_id']+'.npz')).open('xb') as f:np.savez_compressed(f,**payload)
                row=dict(case_id=r['case_id'],input_sha256=r['input_sha256'],roi_returns=len(points),
                    frame_counts=np.bincount(frames,minlength=5).tolist(),r1_groups=len(a.group_ids),
                    r1_unknown_normals=int((~a.normal_valid).sum()),r2_status=r2['status'],
                    r2_voxels=len(r2['voxels'].centers_m),r2_raw_groups=None if b is None else int(len(np.unique(r2['raw_assignment']))),
                    r2_groups=None if b is None else len(b.group_ids),r2_unknown_normals=None if b is None else int((~b.normal_valid).sum()),
                    elapsed_s=time.monotonic()-tick)
                rows.append(row);log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
                page.write('<details><summary>'+html.escape(r['case_id'])+': '+str(len(points))+'个回波</summary><h3>固定体素面片</h3>')
                page.write(preview(points,a.point_to_group,1)+preview(points,a.point_to_group,2))
                page.write('<h3>几何超点</h3>')
                if b is not None:page.write(preview(points,b.point_to_group,1)+preview(points,b.point_to_group,2))
                else:page.write(html.escape(r2['status']))
                page.write('</details>');page.flush()
                if sum(p.stat().st_size for p in run.rglob('*') if p.is_file())>2*1024**3:raise OSError('2GiB evidence cap')
        for p,h in spec['source_sha256'].items():read_pinned(ROOT,p,h)
        if runtime_snapshot(ROOT)!=runtime:raise ValueError('post-run runtime drift')
        if len(rows)!=45:raise ValueError('population incomplete')
    except Exception:
        error=traceback.format_exc();(run/'logs/error.log').write_text(error)
    finally:
        signal.alarm(0)
        summary=dict(status='FAILED' if error else 'EXTRACTION_COMPLETE_NOT_SEMANTICALLY_QUALIFIED',error=error,
            observations=len(rows),roi_returns=sum(r['roi_returns'] for r in rows),
            r1_groups=sum(r['r1_groups'] for r in rows),r2_groups=sum(r['r2_groups'] or 0 for r in rows),
            unpartitioned_observations=sum(r['r2_groups'] is None for r in rows),optimizer_steps=0,
            learned_model_inference=0,scientific_gate_pass=False,elapsed_s=time.monotonic()-started,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        write(run/'metrics/summary.json',summary)
        write(run/'RUN_STATE.json',dict(state='FAILED' if error else 'COMPLETED',run_id=run.name))
        seal=run/'artifacts/evidence_sha256.txt'
        seal.write_text(''.join(sha(p)+'  '+str(p.relative_to(ROOT))+'\n' for p in sorted(run.rglob('*')) if p.is_file() and p!=seal))
    print(json.dumps(summary),flush=True);return int(error is not None)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--freeze',action='store_true');parser.add_argument('--spec',type=Path);parser.add_argument('--run-dir',type=Path)
    args=parser.parse_args()
    if args.freeze:freeze()
    elif args.spec and args.run_dir:raise SystemExit(execute(load_json(args.spec),args.run_dir))
    else:parser.error('freeze or spec/run-dir required')
