"""Complete declared-window opening scoring, independent of training matching.

Every selected query counts. Radius is a schema check, never a discard mask.
Unknown windows cannot receive full detection F1 through this interface.
"""
import numpy as np
from .gse_synthetic_field_scoring import match_positions


def score_complete_window_openings(expected,predicted,probabilities,*,complete_window):
    if complete_window is not True:raise ValueError('independently complete window required')
    for xyz in (expected,predicted):
        if not isinstance(xyz,np.ndarray) or xyz.ndim!=2 or xyz.shape[1]!=3 or xyz.dtype not in (np.dtype('float32'),np.dtype('float64')):
            raise ValueError('floating N,3 surface coordinates required')
        if not np.isfinite(xyz).all():raise ValueError('nonfinite positions')
        error=np.abs(np.linalg.norm(xyz.astype(np.float64),axis=1)-10.)
        if (error>64*np.finfo(xyz.dtype).eps*10.).any():raise ValueError('surface schema failure; never hide off-surface predictions')
    p=np.asarray(probabilities)
    if p.shape!=(len(predicted),) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('finite probabilities in[0,1] required')
    selected=p>=.5
    result=match_positions(expected,predicted[selected],1.)
    result.update(selected_queries=int(selected.sum()),discarded_by_radius=0,confidence_threshold=.5,matching_radius_m=1.)
    return result
