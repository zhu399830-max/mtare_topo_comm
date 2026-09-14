"""Stream the fixed observation-axis comparison, targets opened after forward.

Called only by an authorized frozen executor. This module grants no data scope.
"""
import json
from pathlib import Path
import numpy as np
from mtare_topo.representation.gse_observation_axis_pair import estimate_axes, pair_predictions
from mtare_topo.representation.gse_surface_patches_v1 import SurfacePatches, build_patch_neighbors
from mtare_topo.representation.gse_conditional_geometry_loss import compile_conditional_loss_targets


def metrics(predicted, valid, target, known, same, weights):
    out={}
    for category, mask in [('all', known), ('same', known & same), ('cross', known & ~same)]:
        w=weights[mask].astype(float); total=float(w.sum())
        ok=valid[mask]; truth=target[mask]; values=predicted[mask]
        if np.any(ok & ~np.isfinite(values)):
            raise ValueError('nonfinite valid prediction')
        error=np.ones(len(w)); error[ok]=abs(values[ok]-truth[ok])
        out[category]=dict(reference_count=int(mask.sum()), valid_count=int(ok.sum()),
            weighted_coverage=float(w[ok].sum()/total) if total else None,
            full_population_error=float(w@error/total) if total else None,
            valid_only_mae=float(w[ok]@error[ok]/w[ok].sum()) if w[ok].sum() else None,
            constant_one_mae=float(w@abs(1-truth)/total) if total else None)
    return out


def evaluate(root, run, entries, check_limits):
    root,run=Path(root),Path(run)
    rows=[]
    with (run/'logs/observations.jsonl').open('x') as log:
        for entry in entries:
            identity=entry['identity']; case=identity['case']
            with np.load(root/entry['feature']['path'],allow_pickle=False) as f:
                patches=SurfacePatches(**{k:f['patch_'+k] for k in SurfacePatches.__dataclass_fields__
                    if k not in ('voxel_size_m','roi_radius_m')},voxel_size_m=.5,roi_radius_m=10.)
                neighbors=build_patch_neighbors(patches).neighbor_index
                roi=f['surface_return_indices'].copy()
                points=f['registered_returns_xyz_m']
                mapping=patches.point_patch_index
                # Non-ROI/invalid entries can contain nonfinite coordinates;
                # only original assigned observed returns participate.
                selected=mapping>=0
                estimates=estimate_axes(points[selected],mapping[selected],neighbors,len(patches.centers_m))
                predictions={name:pair_predictions(e,neighbors) for name,e in estimates.items()}
                centers=patches.centers_m.copy()
            # NO source identities or reference values above this line.
            with np.load(root/entry['target']['path'],allow_pickle=False) as y:
                if not np.array_equal(roi,y['original_roi_indices']) or not np.array_equal(neighbors,y['neighbor_index']):
                    raise ValueError('frozen observation/reference indexing mismatch')
                target=compile_conditional_loss_targets(y,'cpu')
            axis=target.axis[0].numpy(); known=target.known[0].numpy()
            same=target.same[0].numpy(); weights=target.weights[0].numpy()
            for name,(predicted,valid) in predictions.items():
                e=estimates[name]
                record=dict(case=case,parent=identity['parent_id'],task=identity['task'],method=name,
                    metrics=metrics(predicted,valid,axis,known,same,weights),
                    unknown_reference_outputs=int((valid & ~known).sum()),
                    undefined_axes=int((~e.valid).sum()),patch_count=len(e.valid))
                rows.append(record);log.write(json.dumps(record,allow_nan=False)+'\n')
                np.savez_compressed(run/f'artifacts/{name}_{case:03d}.npz',axes=e.axes,axis_valid=e.valid,
                    eigenvalues=e.eigenvalues,support_points=e.support_points,centers=centers,
                    neighbor_index=neighbors,predictions=predicted,prediction_valid=valid,
                    reference=axis,reference_known=known,same_reference=same,weights=weights)
            log.flush();check_limits()
            print(json.dumps(dict(completed=case+1,total=len(entries))),flush=True)
    return rows


def summarize(rows):
    parents=sorted({r['parent'] for r in rows}); results={}
    for method in ('POINT','NORMAL'):
        results[method]={}
        for category in ('all','same','cross'):
            per_parent={}
            for parent in parents:
                subset=[r['metrics'][category] for r in rows if r['parent']==parent and r['method']==method]
                per_parent[parent]={key:float(np.mean([x[key] for x in subset if x[key] is not None]))
                    if any(x[key] is not None for x in subset) else None
                    for key in ('weighted_coverage','full_population_error','valid_only_mae','constant_one_mae')}
            results[method][category]=dict(per_parent=per_parent,
                macro={key:float(np.mean([x[key] for x in per_parent.values() if x[key] is not None]))
                       if any(x[key] is not None for x in per_parent.values()) else None
                       for key in next(iter(per_parent.values()))})
    return dict(parents=parents,results=results,meaning='Conditional local axis diagnostic, not graph advantage')
