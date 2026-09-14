"""Independent binary source-affinity counts; unknown is never background."""
import numpy as np


def affinity_metrics(probabilities, values, known, valid):
    p,y,k,v=map(np.asarray,(probabilities,values,known,valid))
    if p.shape!=y.shape or p.shape!=k.shape or p.shape!=v.shape or k.dtype!=bool or v.dtype!=bool:
        raise ValueError('aligned explicit masks required')
    if np.any(k&~v) or not np.isfinite(p[v]).all() or np.any((p[v]<0)|(p[v]>1)):
        raise ValueError('invalid predictions/masks')
    if not np.isin(y[k],[0,1]).all():raise ValueError('known target must be binary')
    positive=k&(y==1);negative=k&(y==0);pred=p>=.5
    tp=int(np.sum(positive&pred));fn=int(np.sum(positive&~pred))
    tn=int(np.sum(negative&~pred));fp=int(np.sum(negative&pred))
    def div(a,b):return a/b if b else None
    pr=div(tp,tp+fn);nr=div(tn,tn+fp)
    return dict(tp=tp,fn=fn,tn=tn,fp=fp,positive_recall=pr,negative_recall=nr,
        balanced_accuracy=(pr+nr)/2 if pr is not None and nr is not None else None,
        positive_f1=div(2*tp,2*tp+fp+fn),negative_f1=div(2*tn,2*tn+fp+fn),
        unknown=int(np.sum(v&~k)),unknown_predicted_positive=int(np.sum(v&~k&pred)))
