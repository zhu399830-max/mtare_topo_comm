"""Metrics for explicit semantics and learned structural-role representations."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np


def decode_direction_components(logits: np.ndarray, threshold: float = 0.5) -> list[float]:
    """Decode circular active runs using a probability-weighted circular center."""

    values=np.asarray(logits,dtype=np.float64)
    if values.shape != (720,): raise ValueError("direction logits must have shape [720]")
    probability=1.0/(1.0+np.exp(-values));active=probability>=threshold
    if not np.any(active):return []
    if np.all(active):return [float(np.argmax(probability)*.5)]
    first_inactive=int(np.flatnonzero(~active)[0]);rolled=np.roll(active,-first_inactive);headings=[];index=0
    while index<720:
        if not rolled[index]:index+=1;continue
        end=index
        while end<720 and rolled[end]:end+=1
        columns=(np.arange(index,end)+first_inactive)%720
        radians=columns*(2*np.pi/720.0);weights=probability[columns]
        angle=np.arctan2(np.sum(weights*np.sin(radians)),np.sum(weights*np.cos(radians)))%(2*np.pi)
        heading=float(angle*180.0/np.pi)
        headings.append(0.0 if abs(heading)<1e-12 or abs(heading-360.0)<1e-12 else heading);index=end
    return sorted(headings)


def match_headings(predicted,truth,tolerance_deg=20.0):
    remaining=list(float(x) for x in truth);errors=[]
    for value in sorted(float(x) for x in predicted):
        if not remaining:break
        index,_=min(enumerate(remaining),key=lambda x:abs((value-x[1]+180)%360-180));error=abs((value-remaining[index]+180)%360-180)
        if error<=tolerance_deg:errors.append(float(error));remaining.pop(index)
    return len(errors),len(predicted),len(truth),errors


def confusion_matrix(target: np.ndarray, prediction: np.ndarray, classes: int) -> np.ndarray:
    target=np.asarray(target,dtype=np.int64).reshape(-1);prediction=np.asarray(prediction,dtype=np.int64).reshape(-1)
    if target.shape!=prediction.shape or np.any(target<0) or np.any(target>=classes) or np.any(prediction<0) or np.any(prediction>=classes):
        raise ValueError("invalid classification arrays")
    matrix=np.zeros((classes,classes),dtype=np.int64)
    np.add.at(matrix,(target,prediction),1);return matrix


def per_class_scores(matrix: np.ndarray) -> dict:
    matrix=np.asarray(matrix,dtype=np.int64);tp=np.diag(matrix).astype(np.float64);support=matrix.sum(axis=1);predicted=matrix.sum(axis=0)
    precision=np.divide(tp,predicted,out=np.zeros_like(tp),where=predicted>0);recall=np.divide(tp,support,out=np.zeros_like(tp),where=support>0)
    f1=np.divide(2*precision*recall,precision+recall,out=np.zeros_like(tp),where=(precision+recall)>0)
    present=support>0
    return {"precision":precision.tolist(),"recall":recall.tolist(),"f1":f1.tolist(),"support":support.tolist(),"macro_f1_present":float(f1[present].mean()) if np.any(present) else 0.0,"accuracy":float(tp.sum()/max(matrix.sum(),1))}


def same_cluster_cosine(embedding: np.ndarray, cluster_ids: Iterable[str]) -> dict:
    values=np.asarray(embedding,dtype=np.float64);ids=list(cluster_ids)
    if values.ndim!=2 or len(ids)!=len(values): raise ValueError("embedding/id shape mismatch")
    norms=np.linalg.norm(values,axis=1,keepdims=True);values=np.divide(values,norms,out=np.zeros_like(values),where=norms>0)
    groups=defaultdict(list)
    for index,key in enumerate(ids):groups[str(key)].append(index)
    similarities=[]
    for indices in groups.values():
        if len(indices)<2:continue
        block=values[indices]@values[indices].T
        similarities.extend(block[np.triu_indices(len(indices),1)].tolist())
    if not similarities: raise ValueError("no same-cluster pairs")
    array=np.asarray(similarities)
    return {"pairs":len(similarities),"mean":float(array.mean()),"p05":float(np.quantile(array,.05)),"minimum":float(array.min())}


def circular_direction_equivariance_error(reference_logits: np.ndarray, rotated_logits: np.ndarray, azimuth_shift_bins: int) -> dict:
    reference=np.asarray(reference_logits,dtype=np.float64);rotated=np.asarray(rotated_logits,dtype=np.float64)
    if reference.shape!=rotated.shape or reference.shape[-1]!=720:raise ValueError("direction logits must share [...,720] shape")
    expected=np.roll(reference,int(azimuth_shift_bins),axis=-1);error=np.abs(expected-rotated)
    return {"mean_absolute_logit_error":float(error.mean()),"p95_absolute_logit_error":float(np.quantile(error,.95)),"maximum_absolute_logit_error":float(error.max())}
