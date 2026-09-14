#!/usr/bin/env python3
"""Read-only posthoc motion plots of a fixed completed run; no scoring/tuning."""
import io
import json
import hashlib
from pathlib import Path
import numpy as np
from _bootstrap import PROJECT_ROOT as ROOT
from mtare_topo.governance_surface_material import read_pinned
from mtare_topo.data.continuous_feature_scope_v1 import compile_feature_scope
from mtare_topo.data.gse_supplement_feature_input_v1 import bind_feature_input
from mtare_topo.topology.saved_anchor_branch_adapter_v1 import convert_saved_prediction
from mtare_topo.evaluation.continuous_anchor_motion_v1 import compose_current_frames,transform_candidates,set_motion_summary

RUN='results/gate3_semantics/gate3_20260909_gse_continuous_predictions_v1_seed0'
SEAL='ba04ea01feaaea62e251a41fbf1e75964d2352f6b2c72e1f15b87abfde98bb7e'
OUTPUT='docs/figures/gse_graph/continuous_anchor_motion_20260909'


def main():
    out=ROOT/OUTPUT
    if out.exists():raise FileExistsError('never overwrite existing review')
    opened={}
    def read(p,h):
        raw=read_pinned(ROOT,p,h);opened[p]=h;return raw
    raw=read(RUN+'/artifacts/evidence_sha256.txt',SEAL)
    pins={p:h for h,p in (line.split('  ',1) for line in raw.decode().splitlines())}
    path=RUN+'/artifacts/prediction_manifest.json';manifest=json.loads(read(path,pins[path]))
    entries=manifest['predictions']
    if len(entries)!=60:raise ValueError('exact60 predictions required')
    indexed={(r['method'],r['source']['task'],r['source']['source_sequence_id']):r for r in entries}
    if len(indexed)!=60:raise ValueError('duplicate predictions')
    scope=compile_feature_scope(ROOT);opened.update(scope['metadata_sha256'])
    records=[];plot_data=[]
    for task in scope['tasks']:
        raw=read(task['input_path'],task['input_sha256'])
        windows=[bind_feature_input(raw,expected_sha256=task['input_sha256'],task=task['task'],row=r['input_row'],
            source_sequence_id=r['source_sequence_id'],frame_rows=r['frame_rows'],observation_count=10) for r in task['observations']]
        frames=[w.provenance['frame_rows'] for w in windows]
        rotations,poses=compose_current_frames([w.translation_m for w in windows],[w.yaw_deg for w in windows],frames)
        for method in ('c0','c1'):
            local=[];selected=[]
            for w in windows:
                entry=indexed.pop((method,task['task'],w.provenance['source_sequence_id']))
                if entry['source']['frame_rows']!=w.provenance['frame_rows'] or entry['input_binding_sha256']!=w.input_binding_sha256:
                    raise ValueError('prediction motion source mismatch')
                path=RUN+'/'+entry['file']
                if pins.get(path)!=entry['sha256']:raise ValueError('prediction seal mismatch')
                with np.load(io.BytesIO(read(path,entry['sha256'])),allow_pickle=False) as a:arrays={k:a[k] for k in a.files}
                convert_saved_prediction(arrays,stream_key=task['task'],decision_index=w.provenance['frame_rows'][-1],
                    frame_orders=w.provenance['frame_rows'],input_binding_sha256=w.input_binding_sha256)
                local.append(arrays['position_m']);selected.append(arrays['presence_logits']>=0)
            local=np.array(local);selected=np.array(selected);fixed=transform_candidates(local,rotations,poses)
            slots=[]
            for q in range(32):
                if selected[:,q].all():
                    slots.append(dict(query=q,local_end_displacement_m=float(np.linalg.norm(local[-1,q]-local[0,q])),
                        compensated_end_displacement_m=float(np.linalg.norm(fixed[-1,q]-fixed[0,q]))))
            records.append(dict(method=method,task=task['task'],source_frame_rows=frames,sensor_positions_first_frame_m=poses.tolist(),
                sensor_displacement_m=float(np.linalg.norm(poses[-1])),selected_counts=selected.sum(axis=1).tolist(),
                local_symmetric_nearest_set_step_m=set_motion_summary(local,selected),
                compensated_symmetric_nearest_set_step_m=set_motion_summary(fixed,selected),
                always_selected_query_traces=slots,query_identity_is_physical_identity=False,
                local_positions_m=local.tolist(),compensated_positions_m=fixed.tolist(),selected=selected.tolist()))
            plot_data.append((task['task'],method,poses,fixed,selected))
    if indexed:raise ValueError('unused predictions')
    # All source binding and motion consistency checks precede output creation.
    out.mkdir(parents=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif']=['Noto Sans CJK SC','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(3,2,figsize=(12,13),constrained_layout=True)
    for ax,(task,method,poses,fixed,mask) in zip(axes.flat,plot_data):
        ax.plot(poses[:,0],poses[:,1],'k.-',label='机器人轨迹')
        for i in range(10):
            p=fixed[i,mask[i]];ax.scatter(p[:,0],p[:,1],c=np.full(len(p),i),cmap='viridis',vmin=0,vmax=9,s=28)
            if i in (0,9):
                for q in np.flatnonzero(mask[i]):ax.annotate(f'{i}:q{q}',fixed[i,q,:2],fontsize=7)
        ax.set_title(task.split('__')[-1]+' / '+method.upper());ax.set_xlabel('首个决策坐标系 X / 米');ax.set_ylabel('Y / 米')
        ax.axis('equal');ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.suptitle('候选路口经运动补偿后的位置：紫色早期，黄色晚期\n数字是时刻与查询槽，不是真实路口身份；未使用地图真值',fontsize=13)
    fig.savefig(out/'motion_xy.png',dpi=160);fig.savefig(out/'motion_xy.pdf');plt.close(fig)
    report=dict(status='POSTHOC_MOTION_DIAGNOSTIC_ONLY',records=records,source_sha256=opened,
        teacher_reads=0,model_calls=0,graph_accuracy_claim=False,
        limitation='One fit traversal with three geometry variants; nearest-set distances are not identity associations or accuracy. Stable counts do not prove stable positions.')
    (out/'motion_review.json').write_text(json.dumps(report,ensure_ascii=False,allow_nan=False)+'\n')
    for p,h in opened.items():
        if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h:raise ValueError('source drift')
    source={str(Path(__file__).relative_to(ROOT)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    module=ROOT/'src/mtare_topo/evaluation/continuous_anchor_motion_v1.py';source[str(module.relative_to(ROOT))]=hashlib.sha256(module.read_bytes()).hexdigest()
    (out/'analysis_source_sha256.json').write_text(json.dumps(source)+'\n')
    seal=out/'evidence_sha256.txt'
    seal.write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(ROOT))+'\n' for p in sorted(out.iterdir()) if p.is_file() and p!=seal))
    for r in records:print(json.dumps({k:v for k,v in r.items() if k in ('task','method','sensor_displacement_m','always_selected_query_traces','local_symmetric_nearest_set_step_m','compensated_symmetric_nearest_set_step_m')},ensure_ascii=False))


if __name__=='__main__':main()
