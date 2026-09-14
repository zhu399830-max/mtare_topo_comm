"""Known-domain centre metrics; maximum-cardinality then minimum-distance.

Unlike the historical scorer exact duplicate ties have deterministic matches,
not an exception. Remaining known-domain duplicates are FP, never discarded.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from mtare_topo.teacher.gse_reference_query_coverage_v2 import bound_anchor_query_coverage


def center_score(predicted,reference,scoreable,allowed,*,radius=1.):
    p=np.asarray(predicted,dtype=float);t=np.asarray(reference,dtype=float)
    s=np.asarray(scoreable);a=np.asarray(allowed)
    if (p.ndim!=2 or p.shape[1]!=3 or t.ndim!=2 or t.shape[1]!=3
            or not np.isfinite(p).all() or not np.isfinite(t).all()
            or s.shape!=(len(p),) or a.shape!=(len(p),) or s.dtype!=bool or a.dtype!=bool
            or not np.isfinite(radius) or not 0<radius<=10):raise ValueError('aligned finite centre inputs required')
    d=np.linalg.norm(p[:,None]-t[None],axis=2)
    eligible=(d<=radius)&a[:,None]
    # Invalid distances all have equal penalty; eligible costs total < min(n,m)+1.
    cost=np.where(eligible,d/radius,(min(len(p),len(t))+1)*2.)
    ii,jj=linear_sum_assignment(cost)
    pairs=[(int(i),int(j)) for i,j in zip(ii,jj) if eligible[i,j]]
    matched={i for i,j in pairs}
    fp=sum(bool(s[i]) for i in range(len(p)) if i not in matched)
    ignored=sum(not bool(s[i]) for i in range(len(p)) if i not in matched)
    delta=np.asarray([p[i]-t[j] for i,j in pairs]).reshape(-1,3)
    tp=len(pairs);fn=len(t)-tp
    return dict(tp=tp,fp=fp,fn=fn,ignored=ignored,output_count=len(p),pairs=pairs,
        precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn) if tp+fn else None,
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
        horizontal_errors_m=np.linalg.norm(delta[:,:2],axis=1).tolist(),
        height_errors_m=np.abs(delta[:,2]).tolist(),distance_errors_m=np.linalg.norm(delta,axis=1).tolist(),
        partial_known_domain_only=True,radius_m=radius)


def score_prediction(prediction,observation,*,radius=1.,threshold=.5):
    keep=(prediction.presence_logits.detach().sigmoid()>=threshold).cpu().numpy()
    xyz=prediction.position_m.detach().cpu().double().numpy()[keep]
    loss=observation.loss_only
    coverage=bound_anchor_query_coverage(loss['bundle'],loss['grid'],xyz,
        produced_targets=loss['produced_targets'],
        manifest_row={k:loss['frozen_manifest'][k] for k in ('source_binding','target_record_sha256')},
        matching_radius_m=radius)
    result=center_score(xyz,loss['target'].position_m.cpu().numpy(),
        np.asarray(coverage['query_scoreable_mask'],bool),
        ~np.asarray(coverage['possible_unconfirmed_reference_mask'],bool),radius=radius)
    result.update(coverage=coverage,query_indices=np.flatnonzero(keep).tolist(),threshold=threshold)
    return result
