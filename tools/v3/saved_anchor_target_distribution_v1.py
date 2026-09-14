"""Read-only archived target distribution; no teacher generation or training."""
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from _bootstrap import PROJECT_ROOT as ROOT
from mtare_topo.data.development_paired_scope import compile_scope
from mtare_topo.data.gse_structure_review_v1 import canonical_sha
from mtare_topo.governance_surface_material import read_pinned


def main():
    output=ROOT/'docs/figures/gse_graph/saved_anchor_target_distribution_20260909'
    if output.exists():raise FileExistsError('immutable diagnostic')
    scope=compile_scope(ROOT);rows=[];opened=dict(scope['manifest_sha256'])
    for row in scope['observations']:
        raw=read_pinned(ROOT,row['target_path'],row['target_sha256']);opened[row['target_path']]=row['target_sha256']
        produced=json.loads(gzip.decompress(raw))['produced_targets'];record=produced['record']
        if (produced['source_binding']!=row['source_binding'] or canonical_sha(record)!=produced['target_record_sha256']
                or record['coordinate_frame']!='current_sensor_m' or record['source_frame_indices']!=row['source']['frame_rows']):
            raise ValueError('archived source/coordinate mismatch')
        count=produced['teacher_provenance']['terminal_anchor_start']
        anchors=record['anchors'][:count]
        if count!=len(produced['teacher_provenance']['anchors']):raise ValueError('junction target count drift')
        xyz=np.array([a['position_m'] for a in anchors],dtype=float).reshape(-1,3)
        if not np.isfinite(xyz).all():raise ValueError('nonfinite target')
        rows.append(dict(source=row['source'],split=row['split'],junction_positions_m=xyz.tolist()))
    summaries={}
    for split in ('fit','calibration','development'):
        chosen=[r for r in rows if r['split']==split]
        xyz=np.array([p for r in chosen for p in r['junction_positions_m']])
        summaries[split]=dict(observations=len(chosen),targets=len(xyz),
            parent_maps=len({r['source']['task'].split('__')[0] for r in chosen}),
            per_axis_min=xyz.min(0).tolist(),per_axis_max=xyz.max(0).tolist(),per_axis_median=np.median(xyz,0).tolist(),
            per_axis_std=xyz.std(0).tolist(),x_quantiles=np.quantile(xyz[:,0],[0,.05,.25,.5,.75,.95,1]).tolist(),
            radial_quantiles=np.quantile(np.linalg.norm(xyz,axis=1),[0,.05,.5,.95,1]).tolist(),
            targets_behind_sensor=int((xyz[:,0]<0).sum()),
            per_observation_target_counts=[len(r['junction_positions_m']) for r in chosen])
    output.mkdir(parents=True)
    result=dict(status='ARCHIVED_TARGET_DISTRIBUTION_NOT_NEW_SCORE',summaries=summaries,rows=rows,source_sha256=opened,
        teacher_generation=0,model_calls=0,optimizer_steps=0,limitation='Marginal position distribution does not verify physical label correctness or explain all model behavior.')
    (output/'distribution.json').write_text(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n')
    (output/'analysis_source_sha256.json').write_text(json.dumps({str(Path(__file__).relative_to(ROOT)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})+'\n')
    for p,h in opened.items():
        if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h:raise ValueError('source drift')
    (output/'evidence_sha256.txt').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(ROOT))+'\n' for p in sorted(output.iterdir()) if p.is_file()))
    print(json.dumps({k:{a:b for a,b in v.items() if a!='per_observation_target_counts'} for k,v in summaries.items()},ensure_ascii=False))


if __name__=='__main__':main()
